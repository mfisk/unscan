//! Font matching pipeline stages extracted from `run()`.
//!
//! - [`LineMatch`]: per-line font matching result
//! - [`match_lines`]: Pass 1 — parallel font matching with SSIM fast path
//! - [`update_dominant_font`]: dominant font candidate update after Pass 1
//! - [`paragraph_font_grouping`]: Pass 1.5 — paragraph-level font grouping

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use image::GrayImage;

use crate::audit;
use crate::char_render;
use crate::classifier;
use crate::cli;
use crate::color;
use crate::features;
use crate::font_cache;
use crate::font_match;
use crate::font_scan;
use crate::ground_truth;
use crate::ocr::{TextLine, TextRegion};
use crate::glyph_map::NgramGlyphMap;
use crate::segment;
use crate::verify;

/// Per-line font matching result produced by [`match_lines`].
pub struct LineMatch {
    pub font_result: Option<font_match::FontMatchResult>,
    pub text_color: (u8, u8, u8),
    pub font_scores: Vec<(String, Option<f32>)>,
    pub observations: Vec<font_match::ObservationDetail>,
    pub font_scores_lig: Vec<(String, Option<f32>)>,
    pub observations_lig: Vec<font_match::ObservationDetail>,
    pub seg_winner: Option<String>,
    pub diag_seg_dir: Option<PathBuf>,
    /// Per-observation rank (1-based) of the chosen font among all fonts, keyed by crop_index.
    pub chosen_obs_ranks: HashMap<usize, usize>,
    /// Per-observation rank (1-based) of the ground-truth font among all fonts, keyed by crop_index.
    pub gt_font_obs_ranks: HashMap<usize, usize>,
    /// Per-observation calibrated probability of the chosen font, keyed by crop_index.
    pub chosen_obs_probs: HashMap<usize, f32>,
    /// Per-observation calibrated probability of the ground-truth font, keyed by crop_index.
    pub gt_font_obs_probs: HashMap<usize, f32>,
    /// font tie-break candidates with per-candidate SSIM scores.
    pub tie_candidates: Vec<audit::TieCandidate>,
}

/// Minimum SSIM score for the fast-path dominant-font check.
const FAST_PATH_MIN_SSIM: f32 = 0.90;

/// Pass 1: parallel font matching with SSIM fast path.
///
/// For each line, tries the dominant font candidate via SSIM first; lines that
/// pass skip segmentation and font matching entirely. Misses fall through to the full
/// pipeline: segmentation → font search → font selection with tie-break.
///
/// Returns `(line_matches, fast_path_hit_count)`.
pub fn match_lines(
    lines: &[TextLine],
    gray_page: &image::GrayImage,
    page_img: &image::DynamicImage,
    page_num: usize,
    font_registry: &font_scan::FontRegistry,
    font_cache: &font_cache::FontCache,
    classifier: &dyn classifier::Classifier,
    glyph_map: &NgramGlyphMap,
    ground_truth: Option<&ground_truth::GroundTruth>,
    dominant_font_candidate: Option<&font_match::FontMatchResult>,
    args: &cli::Args,
    // When set, only these line indices get diag/audit output on disk.
    audit_line_filter: Option<&std::collections::HashSet<usize>>,
) -> (Vec<LineMatch>, u64) {
    let fast_path_font_data: Option<std::sync::Arc<Vec<u8>>> = dominant_font_candidate
        .and_then(|fm| font_cache.load(&fm.font_path).ok());
    let fast_path_hits = AtomicU64::new(0);

    // Profiling accumulators (microseconds, atomic for par_iter)
    let prof_seg_us = AtomicU64::new(0);
    let prof_ci_us = AtomicU64::new(0);
    let prof_pcd_us = AtomicU64::new(0);
    let prof_fp_us = AtomicU64::new(0);
    let prof_full_us = AtomicU64::new(0);
    let line_matches: Vec<LineMatch> = lines.par_iter().enumerate().map(|(li, line)| {
        let line_num = li + 1; // 1-indexed for output
        let line_start = std::time::Instant::now();

        // Crop and contrast-normalize the word-union bbox once for all verify calls.
        let norm_crop = {
            let iw = gray_page.width();
            let ih = gray_page.height();
            let cx = line.x.min(iw.saturating_sub(1));
            let cy = line.y.min(ih.saturating_sub(1));
            let cw = line.width.min(iw - cx);
            let ch = line.height.min(ih - cy);
            let raw = image::imageops::crop_imm(gray_page, cx, cy, cw, ch).to_image();
            features::contrast_normalize_char(&raw)
        };

        // ── Fast path: try dominant font via SSIM ────────────────
        if let (Some(fm), Some(ref fd)) = (dominant_font_candidate, &fast_path_font_data) {
            let vr = verify::verify_text_region(
                &norm_crop,
                fd.as_slice(),
                &line.text,
                &line.words,
                line.x, line.y,
                fm.glyph_overrides.as_deref(),
                &fm.variant_tag,
                fm.variations.as_deref(),
                None,
                Some(FAST_PATH_MIN_SSIM),
            );
            if vr.score >= FAST_PATH_MIN_SSIM {
                fast_path_hits.fetch_add(1, Ordering::Relaxed);
                prof_fp_us.fetch_add(line_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                let text_color = color::detect_text_color(
                    page_img,
                    &TextRegion {
                        text: line.text.clone(),
                        x: line.x, y: line.y,
                        width: line.width, height: line.height,
                        font_size_pt: line.font_size_pt,
                        confidence: line.confidence,
                        level: 5, block_num: 0, par_num: 0, line_num: 0, word_num: 0,
                    },
                );
                let mut result = fm.clone();
                result.best_dy = vr.dy;
                return LineMatch {
                    font_result: Some(result),
                    text_color,
                    font_scores: Vec::new(),
                    observations: Vec::new(),
                    font_scores_lig: Vec::new(),
                    observations_lig: Vec::new(),
                    seg_winner: None,
                    diag_seg_dir: None,
                    chosen_obs_ranks: HashMap::new(),
                    gt_font_obs_ranks: HashMap::new(),
                    chosen_obs_probs: HashMap::new(),
                    gt_font_obs_probs: HashMap::new(),
                    tie_candidates: Vec::new(),
                };
            } else if li < 3 {
            }
        }

        // ── Full pipeline: segmentation → font search → font match ─
        let _preview_end = {
            let mut end = line.text.len().min(30);
            while end > 0 && !line.text.is_char_boundary(end) { end -= 1; }
            end
        };
        let debug_mem = std::env::var("UNPRINT_DEBUG_MEM").is_ok();
        if debug_mem {
        }
        // Dump total mapped size from /proc/self/maps
        if debug_mem && (li == 2 || li == 45) {
            if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                let mut _total: u64 = 0;
                for l in maps.lines() {
                    if let Some(range) = l.split_whitespace().next() {
                        if let Some((start_s, end_s)) = range.split_once('-') {
                            if let (Ok(s), Ok(e)) = (u64::from_str_radix(start_s, 16), u64::from_str_radix(end_s, 16)) {
                                _total += e - s;
                            }
                        }
                    }
                }
            }
        }
        let line_start = std::time::Instant::now();
        let text_color = color::detect_text_color(
            page_img,
            &TextRegion {
                text: line.text.clone(),
                x: line.x, y: line.y,
                width: line.width, height: line.height,
                font_size_pt: line.font_size_pt,
                confidence: line.confidence,
                level: 5, block_num: 0, par_num: 0, line_num: 0, word_num: 0,
            },
        );
        let font_scores: Vec<(String, Option<f32>)>;
        let observations: Vec<font_match::ObservationDetail>;
        let font_scores_lig: Vec<(String, Option<f32>)>;
        let observations_lig: Vec<font_match::ObservationDetail>;
        let seg_winner: Option<String>;
        let diag_seg_dir: Option<PathBuf> = args.diag_seg_dir()
            .filter(|_| audit_line_filter.map_or(true, |f| f.contains(&li)))
            .map(|d| {
            let line_slug: String = line.text.chars().take(30)
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            let p = d.join(format!("p{}_L{:03}_{}", page_num, line_num, line_slug));
            let _ = std::fs::create_dir_all(&p);
            p
        });
        // Segment line (lazy crop data) (before font matching block so they're available after)
        let word_placements: Vec<crate::verify::WordPlacement> = line.words.iter()
            .map(|w| crate::verify::WordPlacement {
                text: w.text.clone(),
                x_off: w.x,
                y_off: w.y,
                width: w.width,
                height: w.height,
            })
            .collect();
        let word_height = line.words.iter().map(|w| w.height).max().unwrap_or(0);
        let seg_t0 = std::time::Instant::now();
        let line_crops = segment::segment_line(
            gray_page, &word_placements, word_height,
            diag_seg_dir.as_deref(),
            &args.render_params(),
        );
        prof_seg_us.fetch_add(seg_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        if debug_mem {
        }

        let mut crop_store_plain: Vec<GrayImage> = Vec::new();
        let mut crop_store_lig: Vec<GrayImage> = Vec::new();
        let (font_result, tie_candidates_audit, gt_font_key) = {

            // Crop PNGs are saved after font matching, gated by ground-truth
            // miss detection when --audit is set (see below).

            // ── Resolve ground-truth font key (if available) ─────
            let gt_font_key: Option<String> = ground_truth.as_ref().and_then(|gt| {
                let bbox_px = [line.x as f32, line.y as f32,
                               (line.x + line.width) as f32,
                               (line.y + line.height) as f32];
                let gt_font_name = gt.lookup_font(page_num, &bbox_px, args.dpi)?;
                let gt_ps = ground_truth::strip_subset_prefix_str(gt_font_name);
                font_registry.iter()
                    .find(|fe| fe.postscript_name == gt_ps)
                    .map(|fe| fe.font_key())
            });
            let ensure_keys: Vec<&str> = gt_font_key.as_deref().into_iter().collect();

            // ── Score: build sliding-window observations, run identify_fonts ──
            let score_t0 = std::time::Instant::now();

            let plain_windows = crate::ngram::build_scoring_windows(
                &line_crops.word_segs, classifier, glyph_map,
                &mut crop_store_plain,
            );
            let scoring_plain = font_match::identify_fonts(&plain_windows, classifier, glyph_map, args.thoroughness, args.full_audit(), &ensure_keys);

            // ── Score ligature path (if present) ─────────────────
            let scoring_lig = if let Some(ref lig_segs) = line_crops.lig_word_segs {
                let lig_windows = crate::ngram::build_scoring_windows(
                    lig_segs, classifier, glyph_map,
                    &mut crop_store_lig,
                );
                Some(font_match::identify_fonts(&lig_windows, classifier, glyph_map, args.thoroughness, args.full_audit(), &ensure_keys))
            } else {
                None
            };
            prof_ci_us.fetch_add(score_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            // ── Pick the winner: ligature vs plain segmentation ──
            // We compare using unweighted (uniform) mean log-probs so
            // weight doesn't bias the decision.
            let plain_top = scoring_plain.unweighted_top;
            let lig_top = scoring_lig.as_ref()
                .map(|r| r.unweighted_top)
                .unwrap_or(f32::MIN);
            let use_lig = scoring_lig.is_some() && lig_top > plain_top;

            let scoring = if use_lig {
                scoring_lig.as_ref().unwrap()
            } else {
                &scoring_plain
            };

            // Store both paths for audit
            font_scores = scoring.scores.iter()
                .map(|(fk, score)| (fk.clone(), Some(*score))).collect();
            observations = scoring.observations.clone();

            // Store the alternate path for audit
            let (scores_lig_audit, obs_lig_audit) = if let Some(ref lig_result) = scoring_lig {
                (lig_result.scores.iter().map(|(fk, s)| (fk.clone(), Some(*s))).collect::<Vec<_>>(),
                 lig_result.observations.clone())
            } else {
                (Vec::new(), Vec::new())
            };
            let (scores_plain_audit, obs_plain_audit) = (
                scoring_plain.scores.iter().map(|(fk, s)| (fk.clone(), Some(*s))).collect::<Vec<_>>(),
                scoring_plain.observations.clone(),
            );

            // Store both in the LineMatch for audit output
            font_scores_lig = if use_lig { scores_plain_audit } else { scores_lig_audit };
            observations_lig = if use_lig { obs_plain_audit } else { obs_lig_audit };
            seg_winner = if scoring_lig.is_some() {
                Some(if use_lig { "ligature".to_string() } else { "plain".to_string() })
            } else {
                None
            };

            if debug_mem {
            }

            // Crop PNGs saved after font matching (see below).

            // ── Font selection: font #1, with SSIM tie-break ───────
            let mut tie_candidates_audit: Vec<audit::TieCandidate> = Vec::new();
            if let Some((ref _top_key, top_score)) = scoring.scores.first() {
                let top_score = *top_score;
                // Collect all candidates that share the top font score
                let tied: Vec<&(String, f32)> = scoring.scores.iter()
                    .take_while(|(_, s)| *s == top_score)
                    .collect();

                if tied.len() >= 2 {
                    // Multiple fonts tied — similarity (ZNCC) decides
                    let mut best: Option<(font_match::FontMatchResult, f32)> = None;
                    let mut log_parts: Vec<String> = Vec::new();
                    let mut tie_sim_results: Vec<(String, String, f32)> = Vec::new();
                    let mut ti = 0usize;
                    for (font_key, _) in tied.iter().map(|&&(ref fk, s)| (fk, s)) {
                        let fe = match font_registry.by_key(font_key) {
                            Some(fe) => fe,
                            None => continue,
                        };
                        let fd = match font_cache.load(&fe.path).ok() {
                            Some(fd) => fd,
                            None => continue,
                        };
                        // Save per-candidate comparison images when audit dir exists
                        let tie_audit_dir = diag_seg_dir.as_ref().map(|d| {
                            let p = d.join(format!("tie_{}", ti));
                            let _ = std::fs::create_dir_all(&p);
                            p
                        });
                        let vr = verify::verify_text_region(
                            &norm_crop, &fd, &line.text,
                            &line.words,
                            line.x, line.y,
                            fe.glyph_overrides.as_deref(), &fe.variant_tag,
                            fe.variations.as_deref(),
                            tie_audit_dir.as_deref(), None,
                        );
                        log_parts.push(format!("{:.4}({})", vr.score, fe.family_name));
                        tie_sim_results.push((fe.font_key(), fe.family_name.clone(), vr.score));
                        if best.as_ref().map_or(true, |(prev, bs)| {
                            vr.score > *bs || (vr.score == *bs && !prev.variant_tag.is_empty() && fe.variant_tag.is_empty())
                        }) {
                            best = Some((font_match::FontMatchResult {
                                font_name: fe.font_key(),
                                font_path: fe.path.clone(),
                                font_key: fe.font_key(),
                                variant_tag: fe.variant_tag.clone(),
                                glyph_overrides: fe.glyph_overrides.clone(),
                                variations: fe.variations.clone(),
                                score: top_score,
                                best_dy: vr.dy,
                            }, vr.score));
                        }
                        ti += 1;
                    }
                    // Build tie_candidates for audit
                    let winner_key = best.as_ref().map(|(fm, _)| fm.font_key.clone());
                    for (fk, fname, sim) in tie_sim_results {
                        tie_candidates_audit.push(audit::TieCandidate {
                            font_key: fk.clone(),
                            family_name: fname,
                            similarity_score: sim,
                            winner: Some(&fk) == winner_key.as_ref(),
                        });
                    }
                    if let Some((ref _winner, _)) = best {
                    }
                    (best.map(|(fm, _)| fm), tie_candidates_audit, gt_font_key)
                } else {
                    // No tie — use font #1 directly, font_key already resolved
                    let (ref font_key, score) = *tied[0];
                    let fm = font_registry.by_key(font_key)
                        .map(|fe| font_match::FontMatchResult {
                            font_name: fe.font_key(),
                            font_path: fe.path.clone(),
                            font_key: fe.font_key(),
                            variant_tag: fe.variant_tag.clone(),
                            glyph_overrides: fe.glyph_overrides.clone(),
                            variations: fe.variations.clone(),
                            score,
                            best_dy: 0,
                        });
                    (fm, Vec::new(), gt_font_key)
                }
            } else {
                (None, Vec::new(), gt_font_key)
            }
        };
        let line_elapsed = line_start.elapsed();
        if line_elapsed.as_millis() > 500 {
        }
        // ── Ground-truth gated audit detail ─────────────────────────
        // When --audit is set, check if this line is a miss before
        // doing expensive audit I/O.  Without --audit, all lines
        // get full audit.  "Miss" means: ground-truth font mismatch, no
        // font matched, OCR too low, or font confidence too low.
        let is_miss = if let Some(ref gt) = ground_truth {
            // OCR too low → line will be kept raster, treat as miss
            let ocr_ok = line.confidence >= args.min_ocr_confidence as f32
                && !line.text.trim().is_empty();
            if !ocr_ok {
                true
            } else if let Some(ref fr) = font_result {
                    let bbox_px = [line.x as f32, line.y as f32,
                                   (line.x + line.width) as f32,
                                   (line.y + line.height) as f32];
                    // Look up chosen font's PostScript name for exact comparison
                    let chosen_ps = font_registry.by_key(&fr.font_key)
                        .map(|fe| fe.postscript_name.as_str())
                        .unwrap_or("");
                    !gt.is_hit(page_num, &bbox_px, args.dpi, chosen_ps)
            } else {
                true // no font matched → treat as miss
            }
        } else {
            true // no ground truth → full audit for all lines
        };

        // The winning crop store contains the actual crops used during scoring —
        // both unigram and bigram crops, indexed by crop_index in observations.
        let winning_crops: &[GrayImage] = if seg_winner.as_deref() == Some("ligature") {
            &crop_store_lig
        } else {
            &crop_store_plain
        };

        let pcd_t0 = std::time::Instant::now();
        // Per-observation probabilities and audit detail: only for miss lines when full audit is active
        let (chosen_obs_ranks, chosen_obs_probs, gt_font_obs_ranks, gt_font_obs_probs) = if is_miss && args.full_audit() {
            // Resolve chosen and GT font keys
            let chosen_font_key: Option<String> = font_result.as_ref()
                .filter(|fr| !fr.font_key.is_empty())
                .map(|fr| fr.font_key.clone());

            // Per-observation probabilities using the actual scoring crops and
            // the correct classifier/glyph_map for each observation's seq length.
            let mut c_ranks = HashMap::new();
            let mut c_probs = HashMap::new();
            let mut g_ranks = HashMap::new();
            let mut g_probs = HashMap::new();

            for d in &observations {
                let crop = match winning_crops.get(d.crop_index) {
                    Some(c) => c,
                    None => continue,
                };
                let feat = match features::compute_features(crop, true) {
                    Some(f) => f,
                    None => continue,
                };

                let probs = classifier.probabilities(&d.seq, &feat);
                if probs.is_empty() { continue; }

                let lookup = |gid: usize| -> Option<(usize, f32)> {
                    probs.iter().enumerate()
                        .find(|(_, (id, _))| *id == gid)
                        .map(|(pos, (_, p))| (pos + 1, *p))
                };

                if let Some(ref fk) = chosen_font_key {
                    if let Some(gid) = glyph_map.glyph_id_for_font(&d.seq, fk) {
                        if let Some((rank, prob)) = lookup(gid) {
                            c_ranks.insert(d.crop_index, rank);
                            c_probs.insert(d.crop_index, prob);
                        }
                    }
                }
                if let Some(ref gtk) = gt_font_key {
                    if let Some(gid) = glyph_map.glyph_id_for_font(&d.seq, gtk) {
                        if let Some((rank, prob)) = lookup(gid) {
                            g_ranks.insert(d.crop_index, rank);
                            g_probs.insert(d.crop_index, prob);
                        }
                    }
                }
            }

            // Render font ref glyphs for miss lines
            if let (Some(ref audit_root), Some(ref fr)) = (&args.audit, &font_result) {
                let fe_opt = font_registry.by_key(&fr.font_key);
                let label = fe_opt.map(|fe| {
                    let mut s = fe.family_name.replace(' ', "");
                    if fe.is_bold { s.push_str("-Bold"); }
                    if fe.is_italic { s.push_str("-Italic"); }
                    if !fe.variant_tag.is_empty() {
                        s.push('_');
                        s.push_str(&fe.variant_tag);
                    }
                    s
                }).unwrap_or_else(|| {
                    fr.font_path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string()
                });
                let font_ref_dir = audit_root.join("font_refs").join(&label);
                let _ = std::fs::create_dir_all(&font_ref_dir);

                let override_map: HashMap<char, u16> = fe_opt
                    .and_then(|fe| fe.glyph_overrides.as_ref())
                    .map(|v| v.iter().cloned().collect())
                    .unwrap_or_default();
                let font_data = font_cache.load(&fr.font_path).ok();
                if let Some(ref fdata) = font_data {
                    if let Ok(mut font) = ab_glyph::FontRef::try_from_slice(fdata) {
                        // Apply variable-font axis coordinates
                        if let Some(vars) = fe_opt.and_then(|fe| fe.variations.as_ref()) {
                            use ab_glyph::VariableFont;
                            for (tag, val) in vars {
                                font.set_variation(tag, *val);
                            }
                        }
                        for d in &observations {
                            // Filename: bigrams use "U+0041_U+0042.png", unigrams "U+0041.png"
                            let fname: String = d.seq.iter()
                                .map(|&c| format!("U+{:04X}", c as u32))
                                .collect::<Vec<_>>()
                                .join("_")
                                + ".png";
                            let path = font_ref_dir.join(&fname);
                            if path.exists() { continue; }
                            let gid_overrides: Vec<Option<ab_glyph::GlyphId>> = d.seq.iter()
                                .map(|c| override_map.get(c).map(|&gid| ab_glyph::GlyphId(gid)))
                                .collect();
                            let ref_img = char_render::render_ngram_default(&font, &d.seq, &gid_overrides);
                            if let Some((_hash, img)) = ref_img {
                                let _ = img.save(&path);
                            }
                        }
                    }
                }
            }

            (c_ranks, c_probs, g_ranks, g_probs)
        } else {
            (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new())
        };

        // Save crop PNGs and scan line image for ALL audited lines (not just
        // misses), so similarity-failure lines have crops in the report too.
        if let Some(ref ddir) = diag_seg_dir {
            if !observations.is_empty() {
                let crop_dir = ddir.join("crops");
                let _ = std::fs::create_dir_all(&crop_dir);
                for d in &observations {
                    if let Some(img) = winning_crops.get(d.crop_index) {
                        let seq_label: String = d.seq.iter().map(|&c| {
                            if c.is_alphanumeric() { format!("{}", c) }
                            else { format!("U{:04X}", c as u32) }
                        }).collect();
                        let path = crop_dir.join(format!("crop_{:02}_{}.png", d.crop_index, seq_label));
                        let _ = img.save(&path);
                    }
                }
            }

            // Save full-colour scan line crop for report overlay.
            {
                let pad = 4u32;
                let mut sx0 = line.x;
                let mut sy0 = line.y;
                let mut sx1 = line.x + line.width;
                let mut sy1 = line.y + line.height;
                for rw in &line.raw_words {
                    sx0 = sx0.min(rw.x);
                    sy0 = sy0.min(rw.y);
                    sx1 = sx1.max(rw.x + rw.width);
                    sy1 = sy1.max(rw.y + rw.height);
                }
                let surr_x = sx0.saturating_sub(pad).min(page_img.width().saturating_sub(1));
                let surr_y = sy0.saturating_sub(pad).min(page_img.height().saturating_sub(1));
                let surr_r = sx1.saturating_add(pad).min(page_img.width());
                let surr_b = sy1.saturating_add(pad).min(page_img.height());
                let surr_w = surr_r - surr_x;
                let surr_h = surr_b - surr_y;
                if surr_w >= 3 && surr_h >= 3 {
                    let crop = image::imageops::crop_imm(page_img, surr_x, surr_y, surr_w, surr_h).to_image();
                    let crop = features::contrast_normalize_rgba(&crop);
                    let _ = crop.save(ddir.join("scan_line.png"));
                    let _ = std::fs::write(
                        ddir.join("scan_line_origin.json"),
                        format!("{{\"x\":{},\"y\":{}}}", surr_x, surr_y),
                    );
                }
            }
        }

        prof_pcd_us.fetch_add(pcd_t0.elapsed().as_micros() as u64, Ordering::Relaxed);

        prof_full_us.fetch_add(line_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        LineMatch { font_result, text_color, font_scores, observations, font_scores_lig, observations_lig, seg_winner, diag_seg_dir, chosen_obs_ranks, gt_font_obs_ranks, chosen_obs_probs, gt_font_obs_probs, tie_candidates: tie_candidates_audit }
    }).collect();

    let fp_hits = fast_path_hits.load(Ordering::Relaxed);
    (line_matches, fp_hits)
}

/// Update the dominant font candidate from this page's match results.
///
/// Returns the new dominant font candidate (most frequently matched font key).
pub fn update_dominant_font(line_matches: &[LineMatch]) -> Option<font_match::FontMatchResult> {
    let mut font_freq: HashMap<String, usize> = HashMap::new();
    for lm in line_matches {
        if let Some(ref fr) = lm.font_result {
            *font_freq.entry(fr.font_key.clone()).or_insert(0) += 1;
        }
    }
    if let Some((top_key, _)) = font_freq.iter().max_by_key(|(_, c)| *c) {
        line_matches.iter()
            .find_map(|lm| lm.font_result.as_ref()
                .filter(|fr| fr.font_key == *top_key)
                .cloned())
    } else {
        None
    }
}

/// Pass 1.5: paragraph-level font grouping.
///
/// Finds the dominant body font: most common font among matched lines at the
/// most common font size (±1pt tolerance). Currently diagnostic-only.
pub fn paragraph_font_grouping(lines: &[TextLine], line_matches: &[LineMatch]) {
    // Collect (font_size_bucket) frequencies
    let mut size_freq: HashMap<i32, u32> = HashMap::new();
    for (i, lm) in line_matches.iter().enumerate() {
        if lm.font_result.is_some() {
            let bucket = lines[i].font_size_pt.round() as i32;
            *size_freq.entry(bucket).or_default() += 1;
        }
    }
    // Find most common size bucket
    let body_size = size_freq.iter()
        .max_by_key(|(_, &v)| v)
        .map(|(&k, _)| k);

    if let Some(body_size) = body_size {
        // Count fonts at body size (±1pt)
        let mut font_freq: HashMap<String, (u32, PathBuf)> = HashMap::new();
        for (i, lm) in line_matches.iter().enumerate() {
            let sz = lines[i].font_size_pt.round() as i32;
            if (sz - body_size).abs() <= 1 {
                if let Some(ref fm) = lm.font_result {
                    let entry = font_freq.entry(fm.font_name.clone())
                        .or_insert_with(|| (0, fm.font_path.clone()));
                    entry.0 += 1;
                }
            }
        }
        // Find majority font
        if let Some((_majority_name, (_majority_count, _majority_path))) = font_freq.iter()
            .max_by_key(|(_, (count, _))| *count)
        {
            let _total_body: u32 = font_freq.values().map(|(c, _)| c).sum();
        }
    }
}

/// Compute font size in points from the OCR bounding-box height and the
/// font's ink-height ratio (ascent − descent).  Falls back to a simple
/// height-to-pt conversion when the font can't be loaded.
pub fn compute_font_size_pt(
    font_result: &Option<font_match::FontMatchResult>,
    line_height: u32,
    dpi: u32,
    font_cache: &font_cache::FontCache,
) -> f32 {
    let dpi_f = dpi as f32;
    let fallback_pt = line_height as f32 * 72.0 / dpi_f;
    let fm = match font_result {
        Some(ref fm) => fm,
        None => return fallback_pt,
    };
    let font_bytes = match font_cache.load(&fm.font_path) {
        Ok(b) => b,
        Err(_) => return fallback_pt,
    };
    let mut f = match ab_glyph::FontRef::try_from_slice(font_bytes.as_slice()) {
        Ok(f) => f,
        Err(_) => return fallback_pt,
    };
    if let Some(ref vars) = fm.variations {
        use ab_glyph::VariableFont;
        for (tag, val) in vars {
            f.set_variation(tag, *val);
        }
    }
    use ab_glyph::{Font, PxScale, ScaleFont};
    let ref_h = 100.0f32;
    let sf_ref = f.as_scaled(PxScale::from(ref_h));
    let ref_ink = sf_ref.ascent() - sf_ref.descent();
    let line_h = line_height as f32;
    if line_h > 1.0 {
        let em_px = ref_h * (line_h / ref_ink);
        em_px * 72.0 / dpi_f
    } else {
        fallback_pt
    }
}
