//! Font matching pipeline stages extracted from `run()`.
//!
//! - [`LineMatch`]: per-line font matching result
//! - [`match_lines`]: Pass 1 — parallel font matching with SSIM fast path
//! - [`update_dominant_font`]: dominant font candidate update after Pass 1
//! - [`paragraph_font_grouping`]: Pass 1.5 — paragraph-level font grouping

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

/// Per-observation rank and probability data for a font path (winner or alt).
#[derive(Default)]
pub struct ObsRankProbs {
    pub chosen_ranks: HashMap<usize, usize>,
    pub chosen_probs: HashMap<usize, f32>,
    pub gt_ranks: HashMap<usize, usize>,
    pub gt_probs: HashMap<usize, f32>,
}

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
    /// Per-observation ranks/probs for the winner path.
    pub obs_rank_probs: ObsRankProbs,
    /// Per-observation ranks/probs for the alt (losing) path.
    pub alt_obs_rank_probs: ObsRankProbs,
    /// font tie-break candidates with per-candidate SSIM scores.
    pub tie_candidates: Vec<audit::TieCandidate>,
    /// When pflda OCR correction fires, the corrected word regions
    /// for use in ZNCC verification (replacing line.words).
    pub corrected_words: Option<Vec<crate::ocr::TextRegion>>,
    /// Whether this line was matched via the dominant-font fast path.
    pub fast_path: bool,
    /// ZNCC verify score from the fast-path check (so pass 2a can skip re-verification).
    pub fast_path_score: Option<f32>,
    /// Per-word segmentation summaries for audit integration.
    pub word_seg_summaries: Vec<crate::audit::WordSegSummary>,
    /// PFLDA OCR corrections with decision data.
    pub ocr_corrections: Vec<crate::audit::OcrCorrection>,
}

/// Minimum SSIM score for the fast-path dominant-font check.
const FAST_PATH_MIN_SSIM: f32 = 0.95;

/// Pass 1: parallel font matching with SSIM fast path.
///
/// For each line, tries the dominant font candidate via SSIM first; lines that
/// pass skip segmentation and font matching entirely. Misses fall through to the full
/// pipeline: segmentation → font search → font selection with tie-break.
///
/// Returns `(line_matches, fast_path_hit_count)`.
/// Compute per-observation ranks and calibrated probabilities for a set of
/// observations against their crops, comparing a chosen font and a GT font.
fn compute_obs_rank_probs(
    observations: &[font_match::ObservationDetail],
    crops: &[GrayImage],
    chosen_font_key: Option<&str>,
    gt_font_key: Option<&str>,
    classifier: &dyn classifier::Classifier,
    glyph_map: &NgramGlyphMap,
) -> ObsRankProbs {
    let mut result = ObsRankProbs::default();
    for d in observations {
        let crop = match crops.get(d.crop_index) {
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

        if let Some(fk) = chosen_font_key {
            if let Some(gid) = glyph_map.glyph_id_for_font(&d.seq, fk) {
                if let Some((rank, prob)) = lookup(gid) {
                    result.chosen_ranks.insert(d.crop_index, rank);
                    result.chosen_probs.insert(d.crop_index, prob);
                }
            }
        }
        if let Some(gtk) = gt_font_key {
            if let Some(gid) = glyph_map.glyph_id_for_font(&d.seq, gtk) {
                if let Some((rank, prob)) = lookup(gid) {
                    result.gt_ranks.insert(d.crop_index, rank);
                    result.gt_probs.insert(d.crop_index, prob);
                }
            }
        }
    }
    result
}

/// Save observation crops to a subdirectory under diag_dir.
fn save_obs_crops(
    diag_dir: &Path,
    subdir: &str,
    observations: &[font_match::ObservationDetail],
    crops: &[GrayImage],
) {
    let crop_dir = diag_dir.join(subdir);
    let _ = std::fs::create_dir_all(&crop_dir);
    for d in observations {
        if let Some(img) = crops.get(d.crop_index) {
            let seq_label: String = d.seq.iter().map(|&c| {
                if c.is_alphanumeric() { format!("{}", c) }
                else { format!("U{:04X}", c as u32) }
            }).collect();
            let path = crop_dir.join(format!("crop_{:02}_{}.png", d.crop_index, seq_label));
            let _ = img.save(&path);
        }
    }
}

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
    // Per-font LDA OCR correction data (None = skip correction)
    training_data: Option<&crate::train::RuntimeTrainingData>,
) -> (Vec<LineMatch>, u64) {
    let fast_path_font_data: Option<std::sync::Arc<Vec<u8>>> = dominant_font_candidate
        .and_then(|fm| font_cache.load(&fm.font_path).ok());
    let fast_path_hits = AtomicU64::new(0);

    // Profiling accumulators (microseconds, atomic for par_iter)
    let line_matches: Vec<LineMatch> = lines.par_iter().enumerate().map(|(li, line)| {
        let line_num = li + 1; // 1-indexed for output
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
                    obs_rank_probs: ObsRankProbs::default(),
                    alt_obs_rank_probs: ObsRankProbs::default(),
                    tie_candidates: Vec::new(),
                    corrected_words: None,
                    fast_path: true,
                    fast_path_score: Some(vr.score),
                    word_seg_summaries: Vec::new(),
                    ocr_corrections: Vec::new(),
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
        let mut observations: Vec<font_match::ObservationDetail>;
        let font_scores_lig: Vec<(String, Option<f32>)>;
        let observations_lig: Vec<font_match::ObservationDetail>;
        let seg_winner: Option<String>;
        let diag_seg_dir: Option<PathBuf> = args.diag_seg_dir()
            .filter(|_| audit_line_filter.map_or(true, |f| f.contains(&li)))
            .map(|d| {
            let line_slug: String = line.text.chars().take(30)
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            let prefix = format!("p{}_L{:03}_", page_num, line_num);
            // Remove stale diag dirs from prior runs with different word splits
            if let Ok(rd) = std::fs::read_dir(&d) {
                for entry in rd.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with(&prefix)
                        && entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                    {
                        let _ = std::fs::remove_dir_all(entry.path());
                    }
                }
            }
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
        let line_crops = segment::segment_line(
            gray_page, &word_placements, word_height,
            diag_seg_dir.as_deref(),
            &args.render_params(),
        );

        let mut crop_store_plain: Vec<GrayImage> = Vec::new();
        let mut crop_store_lig: Vec<GrayImage> = Vec::new();
        let mut plain_pos_map: Vec<(usize, usize)> = Vec::new();
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

            let plain_windows;
            {
                let (w, pm) = crate::ngram::build_scoring_windows(
                    &line_crops.word_segs, classifier, glyph_map,
                    &mut crop_store_plain,
                );
                plain_windows = w;
                plain_pos_map = pm;
            }
            let scoring_plain = font_match::identify_fonts(&plain_windows, classifier, glyph_map, args.thoroughness, args.full_audit(), &ensure_keys, args.min_ngram_prob);

            // ── Score ligature path (if present) ─────────────────
            let scoring_lig = if let Some(ref lig_segs) = line_crops.lig_word_segs {
                let (lig_windows, _lig_pos_map) = crate::ngram::build_scoring_windows(
                    lig_segs, classifier, glyph_map,
                    &mut crop_store_lig,
                );
                Some(font_match::identify_fonts(&lig_windows, classifier, glyph_map, args.thoroughness, args.full_audit(), &ensure_keys, args.min_ngram_prob))
            } else {
                None
            };
            // ── Pick the winner: ligature vs plain segmentation ──
            // Compare lig vs plain using OOD-weighted path scores.
            // Position weights are excluded so they don't bias the decision.
            let plain_top = scoring_plain.path_score;
            let lig_top = scoring_lig.as_ref()
                .map(|r| r.path_score)
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

        // ── Per-font LDA OCR correction (probability-gated) ─────────
        // Iterate directly over OCR characters in the winning word_segs,
        // crop each from segmentation data, classify with per-font LDA,
        // and apply probability-gated corrections.  Positions are known
        // because we walk the source text, so corrections propagate
        // directly to line.words via source_word_idx.
        let mut corrected_words: Option<Vec<crate::ocr::TextRegion>> = None;
        let mut ocr_correction_audit: Vec<crate::audit::OcrCorrection> = Vec::new();
        if let (Some(ref fr), Some(rtd)) = (&font_result, training_data) {
            eprintln!("[pflda] OCR correction pass for font_key={}", fr.font_key);
            let ctx = rtd.as_context(glyph_map);
            if let Some(pf_lda) = classifier::PerFontLda::load_or_train(&fr.font_key, &ctx) {
                let winning_word_segs: &[segment::WordSeg] = if seg_winner.as_deref() == Some("ligature") {
                    line_crops.lig_word_segs.as_deref().unwrap_or(&line_crops.word_segs)
                } else {
                    &line_crops.word_segs
                };
                eprintln!("[pflda] Loaded/trained OK, checking chars across {} word_segs", winning_word_segs.len());

                // -- Load font and compute glyph metric ratios ----------
                // Used to validate OCR corrections: reject replacements
                // whose vertical geometry is incompatible with the crop.
                let glyph_metrics: std::collections::HashMap<char, (f32, f32)> = {
                    let font_entry = ctx.catalog.iter().find(|fe| fe.font_key() == fr.font_key);
                    if let Some(fe) = font_entry {
                        if let Ok(font_data) = std::fs::read(&fe.path) {
                            if let Ok(mut font) = ab_glyph::FontVec::try_from_vec(font_data) {
                                if let Some(ref vars) = fe.variations {
                                    use ab_glyph::VariableFont;
                                    for (tag, val) in vars {
                                        font.set_variation(tag, *val);
                                    }
                                }
                                // Gather all chars that appear in any segment
                                let all_chars: Vec<char> = winning_word_segs.iter()
                                    .flat_map(|seg| seg.chars.iter().copied())
                                    .collect::<std::collections::HashSet<_>>()
                                    .into_iter()
                                    .collect();
                                crate::char_render::glyph_metric_ratios(
                                    &font,
                                    &all_chars,
                                    fe.glyph_overrides.as_deref(),
                                )
                            } else { std::collections::HashMap::new() }
                        } else { std::collections::HashMap::new() }
                    } else { std::collections::HashMap::new() }
                };
                if !glyph_metrics.is_empty() {
                    eprintln!("[pflda] Loaded glyph metrics for {} chars", glyph_metrics.len());
                }

                // -- Build reverse map: (seg_idx, char_pos) → obs index ---
                // Used to update observation audit fields alongside corrected_words.
                let winning_pos_map: &[(usize, usize)] = if seg_winner.as_deref() == Some("ligature") {
                    &[] // ligature pos_map not tracked (rare path)
                } else {
                    &plain_pos_map
                };
                let mut char_to_obs: std::collections::HashMap<(usize, usize), usize> =
                    std::collections::HashMap::new();
                for (obs_i, obs) in observations.iter().enumerate() {
                    if obs.seq.len() != 1 { continue; }
                    if let Some(&(si, cp)) = winning_pos_map.get(obs.crop_index) {
                        char_to_obs.insert((si, cp), obs_i);
                    }
                }

                // -- Pass 1: iterate chars, crop, classify ----------------
                struct PfldaChar {
                    seg_idx: usize,
                    char_pos: usize,
                    ocr_char: char,
                    dists: Vec<(char, f32)>, // (char, d²) sorted by d² asc
                }
                let mut pflda_chars: Vec<PfldaChar> = Vec::new();

                for (seg_idx, seg) in winning_word_segs.iter().enumerate() {
                    for (char_pos, &ocr_char) in seg.chars.iter().enumerate() {
                        if !crate::features::is_supported(ocr_char) { continue; }
                        let crop = match crate::segment::crop_ngram(
                            &seg.word_img, char_pos, 1,
                            &seg.boundaries, &seg.seam_paths, seg.crop_h,
                        ) {
                            Some(c) => c,
                            None => continue,
                        };
                        let hog = match crate::hog::compute_hog(&crop) {
                            Some(h) => h,
                            None => continue,
                        };
                        // Build feature vector: HOG + glyph metric ratios
                        let (metric_top, metric_bot) = glyph_metrics.get(&ocr_char)
                            .copied()
                            .unwrap_or((0.0, 0.0));
                        let mut feats = Vec::with_capacity(hog.len() + 2);
                        feats.extend_from_slice(&hog);
                        feats.push(metric_top);
                        feats.push(metric_bot);
                        let preds_d = pf_lda.predict_with_distances(&feats, 200);
                        if preds_d.is_empty() { continue; }
                        let dists: Vec<(char, f32)> = preds_d.iter()
                            .map(|&(c, _, d)| (c, d))
                            .collect();
                        pflda_chars.push(PfldaChar {
                            seg_idx,
                            char_pos,
                            ocr_char,
                            dists,
                        });
                    }
                }

                // -- Inference σ² = median of nearest-centroid d² ---------
                let inference_sigma_sq: f32 = {
                    let mut top1_d2: Vec<f32> = pflda_chars.iter()
                        .map(|pc| pc.dists[0].1)
                        .collect();
                    if top1_d2.is_empty() {
                        1.0
                    } else {
                        top1_d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        top1_d2[top1_d2.len() / 2]
                    }
                };
                eprintln!("[pflda] inference σ²={:.6} (training σ²={:.6}, {} chars)",
                    inference_sigma_sq, pf_lda.sigma_sq(), pflda_chars.len());

                // -- Pass 2: softmax with inference σ², apply gate --------
                // corrections: (seg_idx, char_pos, from_char, to_char)
                let mut corrections: Vec<(usize, usize, char, char)> = Vec::new();

                for pc in &pflda_chars {
                    let min_d2 = pc.dists[0].1;
                    let weights: Vec<f32> = pc.dists.iter()
                        .map(|&(_, d)| (-(d - min_d2) / (2.0 * inference_sigma_sq)).exp())
                        .collect();
                    let total: f32 = weights.iter().sum();
                    let probs: Vec<(char, f32, f32)> = pc.dists.iter()
                        .zip(weights.iter())
                        .map(|(&(c, d), &w)| (c, if total > 0.0 { w / total } else { 0.0 }, d))
                        .collect();
                    if probs.is_empty() { continue; }

                    let (top_char, top_p, top_d2) = probs[0];
                    let d2_next = if probs.len() > 1 { probs[1].2 } else { 0.0 };

                    let ocr_rank = probs.iter().position(|(ch, _, _)| *ch == pc.ocr_char);
                    let ocr_p = probs.iter().find(|(ch, _, _)| *ch == pc.ocr_char)
                        .map(|(_, p, _)| *p);

                    eprintln!("[pflda] seg[{}][{}] ocr=\'{}\' | top1=\'{}\' p={:.4} d²={:.4} | gap1-2={:.4} | σ²_inf={:.4} | ocr_rank={} ocr_p={} | top5: {}",
                        pc.seg_idx,
                        pc.char_pos,
                        pc.ocr_char,
                        top_char,
                        top_p,
                        top_d2,
                        d2_next - top_d2,
                        inference_sigma_sq,
                        ocr_rank.map(|r| format!("{}", r + 1)).unwrap_or("ABSENT".into()),
                        ocr_p.map(|p| format!("{:.4}", p)).unwrap_or("?".into()),
                        probs.iter().take(5).map(|(c, p, d)| format!("\'{}\' ={:.4}(d²={:.3})", c, p, d)).collect::<Vec<_>>().join(" "),
                    );

                    // Update observation audit fields if we have the mapping
                    if let Some(&obs_i) = char_to_obs.get(&(pc.seg_idx, pc.char_pos)) {
                        if top_char != pc.ocr_char {
                            observations[obs_i].best_alt_char = Some(top_char);
                            observations[obs_i].best_alt_dist = Some(top_p);
                        } else if probs.len() > 1 {
                            observations[obs_i].best_alt_char = Some(probs[1].0);
                            observations[obs_i].best_alt_dist = Some(probs[1].1);
                        }
                        observations[obs_i].pflda_top_char = Some(top_char);
                        observations[obs_i].pflda_top_p = Some(top_p);
                        observations[obs_i].pflda_ocr_p = ocr_p;
                    }

                    // Probability-gated correction
                    let ocr_p_val = ocr_p.unwrap_or(0.0);
                    let ratio = if ocr_p_val > 1e-6 { top_p / ocr_p_val } else { f32::INFINITY };

                    if top_p > 0.235 && ratio > 3.0 && top_char != pc.ocr_char {
                        corrections.push((pc.seg_idx, pc.char_pos, pc.ocr_char, top_char));
                        ocr_correction_audit.push(crate::audit::OcrCorrection {
                            char_pos: pc.char_pos,
                            seg_idx: pc.seg_idx,
                            ocr_char: pc.ocr_char,
                            replacement: top_char,
                            replacement_p: top_p,
                            ocr_p,
                            ratio,
                        });
                        // Update observation
                        if let Some(&obs_i) = char_to_obs.get(&(pc.seg_idx, pc.char_pos)) {
                            observations[obs_i].ocr_corrected_from = Some(pc.ocr_char);
                            observations[obs_i].seq = vec![top_char];
                            observations[obs_i].pflda_replaced = true;
                        }
                        eprintln!("[pflda] CORRECTED \'{}\' → \'{}\' at seg[{}][{}] (word_idx={})",
                            pc.ocr_char, top_char, pc.seg_idx, pc.char_pos,
                            winning_word_segs[pc.seg_idx].source_word_idx);
                    }
                }

                // -- Build corrected_words from corrections ---------------
                if !corrections.is_empty() {
                    let mut words = line.words.clone();
                    for &(seg_idx, char_pos, _from, to) in &corrections {
                        let word_idx = winning_word_segs[seg_idx].source_word_idx;
                        if word_idx < words.len() {
                            let mut chars: Vec<char> = words[word_idx].text.chars().collect();
                            if char_pos < chars.len() {
                                chars[char_pos] = to;
                                words[word_idx].text = chars.into_iter().collect();
                            }
                        }
                    }
                    corrected_words = Some(words);
                }
            }
        }

        // Per-observation probabilities and audit detail: only for miss lines when full audit is active
        let has_ocr_correction = corrected_words.is_some();
        let obs_rank_probs = if (is_miss || has_ocr_correction) && args.full_audit() {
            let chosen_font_key: Option<String> = font_result.as_ref()
                .filter(|fr| !fr.font_key.is_empty())
                .map(|fr| fr.font_key.clone());
            let rp = compute_obs_rank_probs(
                &observations, winning_crops,
                chosen_font_key.as_deref(), gt_font_key.as_deref(),
                classifier, glyph_map,
            );

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
                            let ref_img = char_render::render_ngram_fresh(&font, &d.seq, &gid_overrides, &char_render::RenderParams::default());
                            if let Some(img) = ref_img {
                                let _ = img.save(&path);
                            }
                        }
                    }
                }
            }

            rp
        } else {
            ObsRankProbs::default()
        };

        // Alt path per-observation ranks/probs (using losing path's crops)
        let alt_obs_rank_probs = if !observations_lig.is_empty() && (is_miss || has_ocr_correction) && args.full_audit() {
            let losing_crops: &[GrayImage] = if seg_winner.as_deref() == Some("ligature") {
                &crop_store_plain
            } else {
                &crop_store_lig
            };
            let alt_font_key: Option<String> = font_scores_lig.first().map(|(k, _)| k.clone());
            compute_obs_rank_probs(
                &observations_lig, losing_crops,
                alt_font_key.as_deref(), gt_font_key.as_deref(),
                classifier, glyph_map,
            )
        } else {
            ObsRankProbs::default()
        };

        // Save crop PNGs and scan line image for ALL audited lines (not just
        // misses), so similarity-failure lines have crops in the report too.
        if let Some(ref ddir) = diag_seg_dir {
            if !observations.is_empty() {
                save_obs_crops(ddir, "crops", &observations, winning_crops);
            }
            if !observations_lig.is_empty() && seg_winner.is_some() {
                let losing_crops: &[GrayImage] = if seg_winner.as_deref() == Some("ligature") {
                    &crop_store_plain
                } else {
                    &crop_store_lig
                };
                save_obs_crops(ddir, "crops_alt", &observations_lig, losing_crops);
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


        // Build per-word segmentation summaries for audit integration
        let word_seg_summaries: Vec<crate::audit::WordSegSummary> = {
            let winning_segs: &[segment::WordSeg] = if seg_winner.as_deref() == Some("ligature") {
                line_crops.lig_word_segs.as_deref().unwrap_or(&line_crops.word_segs)
            } else {
                &line_crops.word_segs
            };
            winning_segs.iter().map(|ws| crate::audit::WordSegSummary {
                word_text: ws.word_text.clone(),
                source_word_idx: ws.source_word_idx,
                image_w: ws.image_w,
                image_h: ws.image_h,
                n_chars_expected: ws.n_chars_expected,
                n_segments_produced: ws.n_segments_produced,
                mismatch: ws.mismatch,
                ws_splits: ws.ws_splits.clone(),
                seam_splits: ws.seam_splits.clone(),
                seam_paths: ws.seam_paths.clone(),
                seam_costs: ws.seam_costs.clone(),
            }).collect()
        };

        LineMatch { font_result, text_color, font_scores, observations, font_scores_lig, observations_lig, seg_winner, diag_seg_dir, obs_rank_probs, alt_obs_rank_probs, tie_candidates: tie_candidates_audit, corrected_words, fast_path: false, fast_path_score: None, word_seg_summaries, ocr_corrections: ocr_correction_audit }
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
