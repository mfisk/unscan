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
use crate::glyph_map::GlyphMap;
use crate::segment;
use crate::verify;

/// Per-line font matching result produced by [`match_lines`].
pub struct LineMatch {
    pub font_result: Option<font_match::FontMatchResult>,
    pub text_color: (u8, u8, u8),
    pub ci_top_for_audit: Vec<(String, Option<f32>)>,
    pub ci_char_detail: Vec<font_match::CharMatchDetail>,
    pub ci_top_for_audit_lig: Vec<(String, Option<f32>)>,
    pub ci_char_detail_lig: Vec<font_match::CharMatchDetail>,
    pub seg_winner: Option<String>,
    pub diag_seg_dir: Option<PathBuf>,
    /// Per-char rank (1-based) of the chosen font among all fonts, keyed by crop_index.
    pub chosen_char_ranks: HashMap<usize, usize>,
    /// Per-char rank (1-based) of the ground-truth font among all fonts, keyed by crop_index.
    pub gt_font_char_ranks: HashMap<usize, usize>,
    /// Per-char calibrated probability of the chosen font, keyed by crop_index.
    pub chosen_char_probs: HashMap<usize, f32>,
    /// Per-char calibrated probability of the ground-truth font, keyed by crop_index.
    pub gt_font_char_probs: HashMap<usize, f32>,
    /// CI tie-break candidates with per-candidate SSIM scores.
    pub tie_candidates: Vec<audit::TieCandidate>,
}

/// Minimum SSIM score for the fast-path dominant-font check.
const FAST_PATH_MIN_SSIM: f32 = 0.90;

/// Pass 1: parallel font matching with SSIM fast path.
///
/// For each line, tries the dominant font candidate via SSIM first; lines that
/// pass skip segmentation and CI entirely. Misses fall through to the full
/// pipeline: segmentation → CI search → font selection with tie-break.
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
    glyph_map: &GlyphMap,
    ground_truth: Option<&ground_truth::GroundTruth>,
    dominant_font_candidate: Option<&font_match::FontMatchResult>,
    args: &cli::Args,
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
            let (score, _dy) = verify::verify_text_region(
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
            if score >= FAST_PATH_MIN_SSIM {
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
                result.best_dy = _dy;
                return LineMatch {
                    font_result: Some(result),
                    text_color,
                    ci_top_for_audit: Vec::new(),
                    ci_char_detail: Vec::new(),
                    ci_top_for_audit_lig: Vec::new(),
                    ci_char_detail_lig: Vec::new(),
                    seg_winner: None,
                    diag_seg_dir: None,
                    chosen_char_ranks: HashMap::new(),
                    gt_font_char_ranks: HashMap::new(),
                    chosen_char_probs: HashMap::new(),
                    gt_font_char_probs: HashMap::new(),
                    tie_candidates: Vec::new(),
                };
            } else if li < 3 {
            }
        }

        // ── Full pipeline: segmentation → CI search → font match ─
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
        let mut ci_top_for_audit: Vec<(String, Option<f32>)>;
        let ci_char_detail: Vec<font_match::CharMatchDetail>;
        let ci_top_for_audit_lig: Vec<(String, Option<f32>)>;
        let ci_char_detail_lig: Vec<font_match::CharMatchDetail>;
        let seg_winner: Option<String>;
        let diag_seg_dir: Option<PathBuf> = args.diag_seg_dir().map(|d| {
            let line_slug: String = line.text.chars().take(30)
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            let p = d.join(format!("p{}_L{:03}_{}", page_num, line_num, line_slug));
            let _ = std::fs::create_dir_all(&p);
            p
        });
        // Extract char crops (before font matching block so they're available after)
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
        let line_crops = segment::extract_line_chars(
            gray_page, &word_placements, word_height,
            diag_seg_dir.as_deref(),
            &args.render_params(),
        );
        prof_seg_us.fetch_add(seg_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        let char_crops = &line_crops.plain;
        if debug_mem {
        }

        let (font_result, tie_candidates_audit) = {

            // Crop PNGs are saved after font matching, gated by ground-truth
            // miss detection when --audit is set (see below).

            // ── Score plain path ─────────────────────────────────
            let ci_t0 = std::time::Instant::now();
            let ci_result_plain = font_match::identify_glyph(char_crops, args.thoroughness, args.full_audit(), classifier, glyph_map);

            // ── Score ligature path (if present) ─────────────────
            let ci_result_lig = line_crops.ligature.as_ref().map(|lig_crops| {
                font_match::identify_glyph(lig_crops, args.thoroughness, args.full_audit(), classifier, glyph_map)
            });
            prof_ci_us.fetch_add(ci_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            // ── Pick the winner: ligature vs plain segmentation ──
            // We compare using unweighted (uniform) mean log-probs so
            // char_weight doesn't bias the decision.  Ligature chars
            // carry weight 2.0 for font ranking (they're highly
            // discriminative), but that same bonus would unfairly tip
            // the path comparison toward the ligature segmentation.
            // Unweighted means treat every character equally, so the
            // decision reflects which segmentation genuinely fits
            // better, not which one has heavier-weighted chars.
            let plain_top = ci_result_plain.unweighted_top;
            let lig_top = ci_result_lig.as_ref()
                .map(|r| r.unweighted_top)
                .unwrap_or(f32::MIN);
            let use_lig = ci_result_lig.is_some() && lig_top > plain_top;

            let (ci_result, _winning_crops) = if use_lig {
                (ci_result_lig.as_ref().unwrap(), line_crops.ligature.as_ref().unwrap().as_slice())
            } else {
                (&ci_result_plain, char_crops.as_slice())
            };

            // Store both paths for audit
            ci_top_for_audit = ci_result.scores.iter()
                .map(|(fk, score)| (fk.clone(), Some(*score))).collect();
            ci_char_detail = ci_result.char_detail.clone();

            // Store the alternate path for audit
            let (ci_top_lig_audit, ci_char_lig_audit) = if let Some(ref lig_result) = ci_result_lig {
                (lig_result.scores.iter().map(|(fk, s)| (fk.clone(), Some(*s))).collect::<Vec<_>>(),
                 lig_result.char_detail.clone())
            } else {
                (Vec::new(), Vec::new())
            };
            let (ci_top_plain_audit, ci_char_plain_audit) = (
                ci_result_plain.scores.iter().map(|(fk, s)| (fk.clone(), Some(*s))).collect::<Vec<_>>(),
                ci_result_plain.char_detail.clone(),
            );

            // Store both in the LineMatch for audit output
            ci_top_for_audit_lig = if use_lig { ci_top_plain_audit } else { ci_top_lig_audit };
            ci_char_detail_lig = if use_lig { ci_char_plain_audit } else { ci_char_lig_audit };
            seg_winner = if ci_result_lig.is_some() {
                Some(if use_lig { "ligature".to_string() } else { "plain".to_string() })
            } else {
                None
            };

            if debug_mem {
            }

            // Crop PNGs saved after font matching (see below).

            // ── Font selection: CI #1, with SSIM tie-break ───────
            let mut tie_candidates_audit: Vec<audit::TieCandidate> = Vec::new();
            if let Some((ref _top_key, top_score)) = ci_result.scores.first() {
                let top_score = *top_score;
                // Collect all candidates that share the top CI score
                let tied: Vec<&(String, f32)> = ci_result.scores.iter()
                    .take_while(|(_, s)| *s == top_score)
                    .collect();

                if tied.len() >= 2 {
                    // Multiple fonts tied — SSIM decides
                    let mut best: Option<(font_match::FontMatchResult, f32)> = None;
                    let mut log_parts: Vec<String> = Vec::new();
                    let mut tie_ssim_results: Vec<(String, String, f32)> = Vec::new();
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
                        // Save per-candidate SSIM images when audit dir exists
                        let tie_audit_dir = diag_seg_dir.as_ref().map(|d| {
                            let p = d.join(format!("tie_{}", ti));
                            let _ = std::fs::create_dir_all(&p);
                            p
                        });
                        let (ssim, dy) = verify::verify_text_region(
                            &norm_crop, &fd, &line.text,
                            &line.words,
                            line.x, line.y,
                            fe.glyph_overrides.as_deref(), &fe.variant_tag,
                            fe.variations.as_deref(),
                            tie_audit_dir.as_deref(), None,
                        );
                        log_parts.push(format!("{:.4}({})", ssim, fe.family_name));
                        tie_ssim_results.push((fe.font_key(), fe.family_name.clone(), ssim));
                        if best.as_ref().map_or(true, |(_, bs)| ssim > *bs) {
                            best = Some((font_match::FontMatchResult {
                                font_name: fe.font_key(),
                                font_path: fe.path.clone(),
                                font_key: fe.font_key(),
                                variant_tag: fe.variant_tag.clone(),
                                glyph_overrides: fe.glyph_overrides.clone(),
                                variations: fe.variations.clone(),
                                score: top_score,
                                best_dy: dy,
                            }, ssim));
                        }
                        ti += 1;
                    }
                    // Build tie_candidates for audit
                    let winner_key = best.as_ref().map(|(fm, _)| fm.font_key.clone());
                    for (fk, fname, ssim) in tie_ssim_results {
                        tie_candidates_audit.push(audit::TieCandidate {
                            font_key: fk.clone(),
                            family_name: fname,
                            ssim_score: ssim,
                            winner: Some(&fk) == winner_key.as_ref(),
                        });
                    }
                    if let Some((ref _winner, _)) = best {
                    }
                    (best.map(|(fm, _)| fm), tie_candidates_audit)
                } else {
                    // No tie — use CI #1 directly, font_key already resolved
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
                    (fm, Vec::new())
                }
            } else {
                (None, Vec::new())
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

        // Compute per-char distances for the chosen font.
        // Use corrected characters when the OCR correction gate fired.
        // Use winning path's crops (plain or ligature).
        let effective_crops: &[(char, image::GrayImage)] = if seg_winner.as_deref() == Some("ligature") {
            line_crops.ligature.as_ref().map(|v| v.as_slice()).unwrap_or(char_crops)
        } else {
            char_crops
        };
        // Build a correction map: crop_index → corrected char, without cloning images
        let char_corrections: HashMap<usize, char> = ci_char_detail.iter()
            .filter_map(|d| d.ocr_corrected_from.as_ref().map(|_| (d.crop_index, d.ch)))
            .collect();
        let corrected_char_crops: Vec<(char, &image::GrayImage)> = effective_crops.iter()
            .enumerate()
            .map(|(i, (ch, img))| {
                let effective_ch = char_corrections.get(&i).copied().unwrap_or(*ch);
                (effective_ch, img)
            })
            .collect();

        let pcd_t0 = std::time::Instant::now();

        // Per-char probabilities and audit detail: only for miss lines when full audit is active
        let (chosen_char_ranks, chosen_char_probs, gt_font_char_ranks, gt_font_char_probs) = if is_miss && args.full_audit() {
            // Compute raw features for each crop
            let crop_feats: Vec<(usize, char, features::CharFeatures)> = corrected_char_crops
                .iter()
                .enumerate()
                .filter_map(|(i, (c, img))| {
                    features::compute_features(img, true).map(|f| (i, *c, f))
                })
                .collect();

            // Resolve chosen and GT font keys
            let chosen_font_key: Option<String> = font_result.as_ref()
                .filter(|fr| !fr.font_key.is_empty())
                .map(|fr| fr.font_key.clone());

            let gt_font_key: Option<String> = ground_truth.as_ref().and_then(|gt| {
                let bbox_px = [line.x as f32, line.y as f32,
                               (line.x + line.width) as f32,
                               (line.y + line.height) as f32];
                let gt_font_name = gt.lookup_font(page_num, &bbox_px, args.dpi)?;
                let gt_ps = ground_truth::strip_subset_prefix_str(gt_font_name);
                let gt_key = font_registry.iter()
                    .find(|fe| fe.postscript_name == gt_ps)
                    .map(|fe| fe.font_key())?;
                // Inject GT font into ci_top_for_audit if its font_key is missing
                if !ci_top_for_audit.iter().any(|(fk, _)| *fk == gt_key) {
                    if let Some(&(_, _ch, _)) = crop_feats.first() {
                        let gt_log_probs: Vec<(f32, f32)> = crop_feats.iter()
                            .filter_map(|&(_, ch2, ref feat)| {
                                let gid = glyph_map.glyph_id_for_font(ch2, &gt_key)?;
                                let p = classifier.probability(ch2, feat, gid)?;
                                Some((p.max(1e-30).ln(), font_match::char_weight(ch2)))
                            })
                            .collect();
                        if !gt_log_probs.is_empty() {
                            let score = font_match::aggregate_font_score(&gt_log_probs, crop_feats.len());
                            if score.is_finite() {
                                ci_top_for_audit.push((gt_key.clone(), Some(score)));
                            }
                        }
                    }
                }
                Some(gt_key)
            });

            // One probabilities() call per character; extract
            // rank and probability for chosen and GT glyph IDs (per-char via GlyphMap).
            let mut c_ranks = HashMap::new();
            let mut c_probs = HashMap::new();
            let mut g_ranks = HashMap::new();
            let mut g_probs = HashMap::new();

            for &(crop_idx, ch, ref feat) in &crop_feats {
                let probs = classifier.probabilities(ch, feat);
                if probs.is_empty() { continue; }

                // Extract rank (1-based position in probability-sorted list)
                // and probability for a glyph_id
                let lookup = |gid: usize| -> Option<(usize, f32)> {
                    probs.iter().enumerate()
                        .find(|(_, (id, _))| *id == gid)
                        .map(|(pos, (_, p))| (pos + 1, *p))
                };

                if let Some(ref fk) = chosen_font_key {
                    if let Some(gid) = glyph_map.glyph_id_for_font(ch, fk) {
                        if let Some((rank, prob)) = lookup(gid) {
                            c_ranks.insert(crop_idx, rank);
                            c_probs.insert(crop_idx, prob);
                        }
                    }
                }
                if let Some(ref gtk) = gt_font_key {
                    if let Some(gid) = glyph_map.glyph_id_for_font(ch, gtk) {
                        if let Some((rank, prob)) = lookup(gid) {
                            g_ranks.insert(crop_idx, rank);
                            g_probs.insert(crop_idx, prob);
                        }
                    }
                }
            }

            // Save crop PNGs for miss lines
            if let Some(ref ddir) = diag_seg_dir {
                if !effective_crops.is_empty() {
                    let crop_dir = ddir.join("crops");
                    let _ = std::fs::create_dir_all(&crop_dir);
                    for (i, (ch, img)) in effective_crops.iter().enumerate() {
                        let path = crop_dir.join(format!("crop_{:02}_{}.png", i,
                            if ch.is_alphanumeric() { format!("{}", ch) }
                            else { format!("U{:04X}", *ch as u32) }));
                        let _ = img.save(&path);
                    }
                }

                // Save full-colour scan line crop for report overlay.
                // "Surroundings" bbox: union of line bbox (= word-union)
                // and raw word bboxes, plus 4px padding, so the report
                // can show both original and expanded bbox sets.
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
                        for (ch, _crop) in corrected_char_crops.iter() {
                            let fname = format!("U+{:04X}.png", *ch as u32);
                            let path = font_ref_dir.join(&fname);
                            if path.exists() { continue; }
                            let gid_override = override_map.get(ch).map(|&gid| ab_glyph::GlyphId(gid));
                            let ref_img = char_render::get_rendered_char_default(&font, *ch, gid_override);
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
        prof_pcd_us.fetch_add(pcd_t0.elapsed().as_micros() as u64, Ordering::Relaxed);

        prof_full_us.fetch_add(line_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        LineMatch { font_result, text_color, ci_top_for_audit, ci_char_detail, ci_top_for_audit_lig, ci_char_detail_lig, seg_winner, diag_seg_dir, chosen_char_ranks, gt_font_char_ranks, chosen_char_probs, gt_font_char_probs, tie_candidates: tie_candidates_audit }
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
