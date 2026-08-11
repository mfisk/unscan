//! Font matching pipeline stages extracted from `run()`.
//!
//! - [`LineMatch`]: per-line font matching result
//! - [`match_lines`]: Pass 1 — parallel font matching with SSIM fast path
//! - [`update_dominant_font`]: dominant font candidate update after Pass 1
//! - [`paragraph_font_grouping`]: Pass 1.5 — paragraph-level font grouping
//!
//! Rule-out via infinite penalty: if a font does not contain a character that
//! appears in the string, `per_char_geo` returns `None` (cmap miss). The
//! per-character geometry log-likelihood is `-infinity` (infinitely bad), so the
//! whole-font score is `-infinity`. The pipeline inserts the font index into
//! `cannot_render: HashSet<usize>` and skips it before softmax. This skip is
//! mathematically correct because `exp(-inf)=0`, so softmax probability is 0.
//! The abort is a valid short-circuit: a missing glyph cannot be rendered, so
//! the font is ruled out without further scoring. Empty `Vec` is distinct: it
//! means ligature mismatch (no usable words) and is kept as `Some(empty)` with
//! SSIM-only scoring, not infinite penalty.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use image::GrayImage;

use crate::audit;
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
    // Raw glyph scores ( -d²/(2σ²) ) before geo, for report display
    pub chosen_glyph_scores: HashMap<usize, f32>,
    pub gt_glyph_scores: HashMap<usize, f32>,
    // Geo scores per observation (crop_index -> value)
    pub chosen_geo_h_ll: HashMap<usize, f32>,
    pub chosen_geo_v_ll: HashMap<usize, f32>,
    pub chosen_geo_h_err: HashMap<usize, f32>,
    pub chosen_geo_v_err: HashMap<usize, f32>,
    pub gt_geo_h_ll: HashMap<usize, f32>,
    pub gt_geo_v_ll: HashMap<usize, f32>,
    pub gt_geo_h_err: HashMap<usize, f32>,
    pub gt_geo_v_err: HashMap<usize, f32>,
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
    /// Median em_px derived from midpoint center-span scales (obs_span/pred_span * upem).
    /// This reuses the exact scale calculation from geometry scoring for font-size,
    /// fixing L9 fox/jumps too-small issue where width-matched median was dragged down.
    pub midpoint_em_px: Option<f32>,
    /// GT font's own midpoint em_px computed from GT's predicted span, not winner's.
    /// Fixes p1:L7 1.6% scale error where GT rendered at winner size.
    pub gt_midpoint_em_px: Option<f32>,
    /// Per-word segmentation summaries for audit integration.
    pub word_seg_summaries: Vec<crate::audit::WordSegSummary>,
    /// GT font segmentation summaries – always included when GT scored, even on misses.
    pub gt_word_seg_summaries: Vec<crate::audit::WordSegSummary>,
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

/// Save observation crops to a subdirectory under diag_dir.
fn save_obs_crops(
    diag_dir: &Path,
    subdir: &str,
    observations: &[font_match::ObservationDetail],
    crops: &[GrayImage],
) {
    use std::fs::File;
    use std::io::BufWriter;
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    let crop_dir = diag_dir.join(subdir);
    let _ = std::fs::create_dir_all(&crop_dir);
    for d in observations {
        if let Some(img) = crops.get(d.crop_index) {
            let c = d.ch;
            let seq_label: String = if c.is_alphanumeric() { format!("{}", c) } else { format!("U{:04X}", c as u32) };
            let path = crop_dir.join(format!("crop_{:02}_{}.png", d.crop_index, seq_label));
            // Diag output: best-effort atomic, don't block pipeline on failure.
            // Must use PngEncoder explicitly — `img.save(.tmp)` fails because extension is `.tmp`.
            let tmp = crate::atomic_file::tmp_for(&path);
            let ok = (|| -> std::io::Result<()> {
                let f = File::create(&tmp)?;
                let mut w = BufWriter::new(f);
                let enc = PngEncoder::new(&mut w);
                enc.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::L8)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                use std::io::Write;
                w.flush()?;
                Ok(())
            })()
            .is_ok();
            if ok {
                let _ = std::fs::rename(&tmp, &path);
            } else {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
}

pub fn match_lines(
    lines: &[TextLine],
    gray_page: &image::GrayImage,
    rgba_page: &image::RgbaImage,
    page_img: &image::DynamicImage,
    page_num: usize,
    font_registry: &font_scan::FontRegistry,
    font_cache: &font_cache::FontCache,
    geo_cache: &crate::geo_cache::GeometryCache,
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

    // ── Serialized diag-dir stale cleanup (before parallel work) ──
    // Previously this was inside the par_iter with `let _ = remove_dir_all`,
    // which could race and would silently ignore errors, and also deleted
    // the dir that Pass1b had just recreated when old and new slugs are
    // identical (line.text unchanged after word-split). We now clean once,
    // serially, before any workers start.
    if let Some(diag_root) = args.diag_seg_dir() {
        let mut prefixes: Vec<String> = Vec::new();
        for (li, _line) in lines.iter().enumerate() {
            if audit_line_filter.as_ref().map_or(true, |f| f.contains(&li)) {
                let line_num = li + 1;
                prefixes.push(format!("p{}_L{:03}_", page_num, line_num));
            }
        }
        if !prefixes.is_empty() {
            match std::fs::read_dir(&diag_root) {
                Ok(rd) => {
                    for entry in rd.flatten() {
                        if let Ok(ft) = entry.file_type() {
                            if ft.is_dir() {
                                let name = entry.file_name();
                                let name_str = name.to_string_lossy();
                                for pref in &prefixes {
                                    if name_str.starts_with(pref) {
                                        let path = entry.path();
                                        if let Err(e) = std::fs::remove_dir_all(&path) {
                                            eprintln!(
                                                "[diag-clean] failed to remove stale {}: {}",
                                                path.display(),
                                                e
                                            );
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[diag-clean] failed to read diag root {}: {}",
                        diag_root.display(),
                        e
                    );
                }
            }
        }
    }

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
            features::contrast_normalize_char(raw)
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
                true,
                None,
                Some(FAST_PATH_MIN_SSIM),
                None,
            );
            if vr.score >= FAST_PATH_MIN_SSIM {
                fast_path_hits.fetch_add(1, Ordering::Relaxed);
                let text_color = color::detect_text_color_from_buffers(
                    gray_page,
                    rgba_page,
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
                    midpoint_em_px: None,
                    gt_midpoint_em_px: None,
                    word_seg_summaries: Vec::new(),
                    gt_word_seg_summaries: Vec::new(),
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

        let text_color = color::detect_text_color_from_buffers(
            gray_page,
            rgba_page,
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
        let gt_observations: Vec<font_match::ObservationDetail>;
        let font_scores_lig: Vec<(String, Option<f32>)>;
        let observations_lig: Vec<font_match::ObservationDetail>;
        let seg_winner: Option<String>;
        let diag_seg_dir: Option<PathBuf> = args
            .diag_seg_dir()
            .filter(|_| audit_line_filter.map_or(true, |f| f.contains(&li)))
            .map(|d| {
                let line_slug: String = line
                    .text
                    .chars()
                    .take(30)
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let p = d.join(format!("p{}_L{:03}_{}", page_num, line_num, line_slug));
                let _ = std::fs::create_dir_all(&p);
                p
            });
        // Segment line (lazy crop data) (before font matching block so they're available after)
        let _word_placements: Vec<crate::verify::WordPlacement> = line.words.iter()
            .map(|w| crate::verify::WordPlacement {
                text: w.text.clone(),
                x_off: w.x,
                y_off: w.y,
                width: w.width,
                height: w.height,
            })
            .collect();
        // Per-font pipeline: segmentation now happens inside identify_fonts,
        // so we don't pre-segment here with segment_line.
        // ── Per-font ligature handling: build word image infos for inside identify_fonts ──
        use std::sync::Arc;
        // Build filtered word infos (replicates segment_line filtering but without segmentation)
        let word_infos: Vec<font_match::WordImgInfo> = {
            let pw = gray_page.width();
            let ph = gray_page.height();
            let audit_all = crate::features::audit_all_chars_enabled();
            let mut char_counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
            let mut sorted: Vec<(usize, &crate::ocr::TextRegion)> = line.words.iter().enumerate().collect();
            sorted.sort_by(|a,b| b.1.text.chars().count().cmp(&a.1.text.chars().count()));
            let mut infos = Vec::new();
            for (orig_idx, w) in sorted {
                if w.width == 0 || w.height == 0 { continue; }
                let wx = w.x; let wy = w.y; let ww = w.width; let wh = w.height;
                if wx >= pw || wy >= ph { continue; }
                let cw = ww.min(pw - wx);
                let ch = wh.min(ph - wy);
                if cw < 2 || ch < 2 { continue; }
                let chars_supported: Vec<char> = if audit_all { w.text.chars().collect() } else { w.text.chars().filter(|c| crate::features::is_supported(*c)).collect() };
                if !audit_all && chars_supported.len() <= 1 { continue; }
                let need_any = if audit_all { true } else { chars_supported.iter().any(|c| char_counts.get(c).copied().unwrap_or(0) < 2) };
                if !audit_all && chars_supported.len() > 2 && !need_any { continue; }
                for &c in &chars_supported { if audit_all || crate::features::is_supported(c) { *char_counts.entry(c).or_insert(0) += 1; } }
                let word_img = image::imageops::crop_imm(gray_page, wx, wy, cw, ch).to_image();
                let word_img = crate::features::contrast_normalize_char(word_img);
                let all_chars: Vec<char> = w.text.chars().collect();
                if all_chars.is_empty() { continue; }
                infos.push(font_match::WordImgInfo::new(Arc::new(word_img), all_chars, orig_idx, w.text.clone()));
            }
            infos
        };

        // Helper to build per-font WordSegs + wib for any font entry (used for tie and winner recompute)
        let build_for_font = |fe: &crate::font_scan::FontEntry, infos: &[font_match::WordImgInfo]| -> Option<(Vec<crate::segment::WordSeg>, Vec<crate::geometry_classifier::WordGeoMeasurement>, Vec<image::GrayImage>, Vec<(usize,usize)>)> {
            let allowed = fe.collapsed_lig_set();
            let mut segs: Vec<crate::segment::WordSeg> = Vec::with_capacity(infos.len());
            for wi in infos {
                let collapsed = crate::segment::collapse_ligature_chars_for_allowed(&wi.orig_chars, &allowed);
                let k = collapsed.len();
                if k == 0 { continue; }
                let (bounds, seams, summary) = crate::segment::segment_characters(&wi.img, k);
                segs.push(crate::segment::WordSeg {
                    source_word_idx: wi.source_idx,
                    word_img: wi.img.clone(),
                    chars: collapsed,
                    boundaries: bounds,
                    seam_paths: Arc::new(seams),
                    seam_costs: Arc::new(summary.seam_costs.clone()),
                    crop_h: wi.img.height(),
                    word_text: wi.text.clone(),
                    image_w: summary.image_w,
                    image_h: summary.image_h,
                    n_chars_expected: summary.n_chars_expected,
                    n_segments_produced: summary.n_segments_produced,
                    mismatch: summary.mismatch,
                    ws_splits: summary.ws_splits.clone(),
                    seam_splits: summary.seam_splits.clone(),
                });
            }
            if segs.is_empty() { return None; }
            let mut crop_store: Vec<image::GrayImage> = Vec::new();
            let (_windows, p_map, wib) = crate::ngram::build_scoring_windows_with_geo(&segs, &mut crop_store);
            Some((segs, wib, crop_store, p_map))
        };

        let (font_result, tie_candidates_audit, gt_font_key, winning_segs, winning_wib, winning_pos_map, winning_crops) = {

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

            // ── Score: per-font collapse + segmentation cached inside identify_fonts ──
            let scoring = font_match::identify_fonts(
                &word_infos,
                classifier,
                glyph_map,
                args.thoroughness,
                args.full_audit(),
                &ensure_keys,
                args.min_ngram_prob,
                font_registry,
                font_cache,
                geo_cache,
            );

            font_scores = scoring.scores.into_iter().map(|(k,s)| (k, Some(s))).collect();
            observations = scoring.observations;
            gt_observations = scoring.gt_observations;
            font_scores_lig = Vec::new();
            observations_lig = Vec::new();
            seg_winner = None;
            // keep winning crops empty for now – filled after tie resolution
            // (winning_* computed below for reuse)

            let mut tie_candidates_audit: Vec<audit::TieCandidate> = Vec::new();
            let mut winning_segs: Vec<crate::segment::WordSeg> = Vec::new();
            let mut winning_wib: Vec<crate::geometry_classifier::WordGeoMeasurement> = Vec::new();
            let mut winning_pos: Vec<(usize,usize)> = Vec::new();
            let mut winning_crops_vec: Vec<image::GrayImage> = Vec::new();

            if let Some((ref _top_key, Some(top_score_opt))) = font_scores.first() {
                let top_score = *top_score_opt;
                let tied: Vec<String> = font_scores.iter()
                    .take_while(|(_, os)| os.map_or(false, |s| s == top_score))
                    .map(|(k,_)| k.clone())
                    .collect();

                if tied.len() >= 2 {
                    let mut best: Option<(font_match::FontMatchResult, f32, Vec<crate::segment::WordSeg>, Vec<crate::geometry_classifier::WordGeoMeasurement>, Vec<image::GrayImage>, Vec<(usize,usize)>)> = None;
                    let mut tie_sim_results: Vec<(String,String,f32)> = Vec::new();
                    let mut ti = 0usize;
                    for font_key in tied.iter() {
                        let fe = match font_registry.by_key(font_key) {
                            Some(v) => v,
                            None => continue,
                        };
                        let fd = match font_cache.load(&fe.path).ok() {
                            Some(v) => v,
                            None => continue,
                        };
                        let tie_audit_dir = diag_seg_dir.as_ref().map(|d| {
                            let p = d.join(format!("tie_{}", ti));
                            let _ = std::fs::create_dir_all(&p);
                            p
                        });
                        let (segs, wib, _crops, _pmap) = match build_for_font(fe, &word_infos) {
                            Some(v) => v,
                            None => continue,
                        };
                        let midpoint_em_px = crate::geometry_classifier::median_em_px_from_midpoints(font_key, &segs, &wib, geo_cache);
                        let vr = verify::verify_text_region(
                            &norm_crop, &fd, &line.text, &line.words, line.x, line.y,
                            fe.glyph_overrides.as_deref(), &fe.variant_tag, fe.variations.as_deref(),
                            true, tie_audit_dir.as_deref(), None, midpoint_em_px,
                        );
                        tie_sim_results.push((fe.font_key(), fe.family_name.clone(), vr.score));
                        if best.as_ref().map_or(true, |(_, bs, _, _, _, _)| {
                            vr.score > *bs || (vr.score == *bs && prev_variant_pref(best.as_ref(), fe))
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
                            }, vr.score, segs, wib, _crops, _pmap));
                        }
                        ti += 1;
                    }
                    let winner_key = best.as_ref().map(|(fm,_,_,_,_,_)| fm.font_key.clone());
                    for (fk,fname,sim) in tie_sim_results {
                        tie_candidates_audit.push(audit::TieCandidate {
                            font_key: fk.clone(),
                            family_name: fname,
                            similarity_score: sim,
                            winner: Some(&fk) == winner_key.as_ref(),
                        });
                    }
                    if let Some((fm, _sim, segs, wib, crops, pmap)) = best {
                        winning_segs = segs;
                        winning_wib = wib;
                        winning_crops_vec = crops;
                        winning_pos = pmap;
                        (Some(fm), tie_candidates_audit, gt_font_key, winning_segs, winning_wib, winning_pos, winning_crops_vec)
                    } else {
                        (None, tie_candidates_audit, gt_font_key, Vec::new(), Vec::new(), Vec::new(), Vec::new())
                    }
                } else {
                    // Single winner – rebuild its segs for downstream audit/midpoint
                    let font_key = &tied[0];
                    let fe_opt = font_registry.by_key(font_key);
                    if let Some(fe) = fe_opt {
                        if let Some((segs, wib, crops, pmap)) = build_for_font(fe, &word_infos) {
                            winning_segs = segs;
                            winning_wib = wib;
                            winning_crops_vec = crops;
                            winning_pos = pmap;
                        }
                    }
                    let score = top_score;
                    let fm = font_registry.by_key(font_key).map(|fe| font_match::FontMatchResult {
                        font_name: fe.font_key(),
                        font_path: fe.path.clone(),
                        font_key: fe.font_key(),
                        variant_tag: fe.variant_tag.clone(),
                        glyph_overrides: fe.glyph_overrides.clone(),
                        variations: fe.variations.clone(),
                        score,
                        best_dy: 0,
                    });
                    (fm, Vec::new(), gt_font_key, winning_segs, winning_wib, winning_pos, winning_crops_vec)
                }
            } else {
                (None, Vec::new(), gt_font_key, Vec::new(), Vec::new(), Vec::new(), Vec::new())
            }
        };

        // helper for tie variant preference – moved outside loop capture
        fn prev_variant_pref(prev: Option<&(font_match::FontMatchResult, f32, Vec<crate::segment::WordSeg>, Vec<crate::geometry_classifier::WordGeoMeasurement>, Vec<image::GrayImage>, Vec<(usize,usize)>)>, fe: &crate::font_scan::FontEntry) -> bool {
            if let Some((prev_fm, _, _, _, _, _)) = prev {
                !prev_fm.variant_tag.is_empty() && fe.variant_tag.is_empty()
            } else { false }
        }

        // ── Ground-truth gated audit detail ─────────────────────────
        // When --audit is set, check if this line is a miss before
        // doing expensive audit I/O.  Without --audit, all lines
        // get full audit.  "Miss" means: ground-truth font mismatch, no
        // font matched, OCR too low, or font confidence too low.
        let _is_miss = if let Some(ref gt) = ground_truth {
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

        let winning_crops_slice: &[GrayImage] = winning_crops.as_slice();
        let winning_crops = winning_crops_slice;

        // ── Per-font LDA OCR correction (probability-gated) ─────────
        // Iterate directly over OCR characters in the winning word_segs,
        // crop each from segmentation data, classify with per-font LDA,
        // and apply probability-gated corrections.  Positions are known
        // because we walk the source text, so corrections propagate
        // directly to line.words via source_word_idx.
        let mut corrected_words: Option<Vec<crate::ocr::TextRegion>> = None;
        let mut ocr_correction_audit: Vec<crate::audit::OcrCorrection> = Vec::new();
        if args.skip_ocr_correction
            || std::env::var("UNPRINT_SKIP_OCR_CORRECTION").is_ok()
        {
            // skip pflda for t64 fast path
        } else if let (Some(ref fr), Some(rtd)) = (&font_result, training_data) {
            if std::env::var("UNPRINT_VERBOSE_PFLDA").is_ok() { eprintln!("[pflda] OCR correction pass for font_key={}", fr.font_key); }
            let ctx = rtd.as_context(glyph_map);
            if let Some(pf_lda) = classifier::PerFontLda::load_or_train(&fr.font_key, &ctx) {
                let winning_word_segs: &[segment::WordSeg] = &winning_segs;
                if std::env::var("UNPRINT_VERBOSE_PFLDA").is_ok() { eprintln!("[pflda] Loaded/trained OK, checking chars across {} word_segs", winning_word_segs.len()); }
                // -- Load font and compute glyph metric ratios ----------
                // Used to validate OCR corrections: reject replacements
                // whose vertical geometry is incompatible with the crop.
                let glyph_metrics: std::collections::HashMap<char, (f32, f32)> = {
                    let font_entry = ctx.catalog.iter().find(|fe| fe.font_key() == fr.font_key);
                    if let Some(fe) = font_entry {
                        if let Ok(font_data) = std::fs::read(&fe.path) {
                            if let Ok(mut font) = unprint_fonts::ab_glyph::FontVec::try_from_vec(font_data) {
                                if let Some(ref vars) = fe.variations {
                                    use unprint_fonts::ab_glyph::VariableFont;
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
                    if std::env::var("UNPRINT_VERBOSE_PFLDA").is_ok() { eprintln!("[pflda] Loaded glyph metrics for {} chars", glyph_metrics.len()); }
                }

                let winning_pos_map: &[(usize, usize)] = &winning_pos_map;
                let mut char_to_obs: std::collections::HashMap<(usize, usize), usize> =
                    std::collections::HashMap::new();
                for (obs_i, obs) in observations.iter().enumerate() {
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
                        if !crate::features::audit_all_chars_enabled() && !crate::features::is_supported(ocr_char) { continue; }
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
                if std::env::var("UNPRINT_VERBOSE_PFLDA").is_ok() { eprintln!("[pflda] inference σ²={:.6} (training σ²={:.6}, {} chars)",
                    inference_sigma_sq, pf_lda.sigma_sq(), pflda_chars.len()); }
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

                    if std::env::var("UNPRINT_VERBOSE_PFLDA").is_ok() { eprintln!("[pflda] seg[{}][{}] ocr=\'{}\' | top1=\'{}\' p={:.4} d²={:.4} | gap1-2={:.4} | σ²_inf={:.4} | ocr_rank={} ocr_p={} | top5: {}",
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
                    ); }
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
                            observations[obs_i].ch = top_char;
                            observations[obs_i].pflda_replaced = true;
                        }
                        if std::env::var("UNPRINT_VERBOSE_PFLDA").is_ok() { eprintln!("[pflda] CORRECTED \'{}\' → \'{}\' at seg[{}][{}] (word_idx={})",
                            pc.ocr_char, top_char, pc.seg_idx, pc.char_pos,
                            winning_word_segs[pc.seg_idx].source_word_idx); }
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

        // ── Audit: pull per-char rank/prob/glyph/geo from mainline work (no recompute) ──
        // `identify_fonts` already computed best_prob, logit (as glyph_score), geo_ll
        // for the winning font path + GT path. Populate ObsRankProbs so the report's
        // per-char table has data.
        let obs_rank_probs = {
            let mut rp = ObsRankProbs::default();
            for obs in &observations {
                rp.chosen_ranks.insert(obs.crop_index, 1);
                // prefer actual prob from classifier, fall back to best_prob
                if let Some(p) = obs.prob { rp.chosen_probs.insert(obs.crop_index, p); }
                else if obs.best_prob > 0.0 { rp.chosen_probs.insert(obs.crop_index, obs.best_prob); }
                if let Some(gs) = obs.glyph_score { rp.chosen_glyph_scores.insert(obs.crop_index, gs); }
                else if obs.best_prob > 0.0 { rp.chosen_glyph_scores.insert(obs.crop_index, obs.best_prob.ln()); }
                if let Some(h) = obs.geo_h_ll { rp.chosen_geo_h_ll.insert(obs.crop_index, h); }
                if let Some(v) = obs.geo_v_ll { rp.chosen_geo_v_ll.insert(obs.crop_index, v); }
                if let Some(he) = obs.geo_h_err { rp.chosen_geo_h_err.insert(obs.crop_index, he); }
                if let Some(ve) = obs.geo_v_err { rp.chosen_geo_v_err.insert(obs.crop_index, ve); }
            }
            // GT side – from scoring.gt_observations (ensure font)
            // When window counts differ (ligature difference, e.g. Office 4 vs 5 glyphs),
            // direct crop_index mapping mis-aligns later chars (c/e vs T...). Use
            // sequential char-equality mapping for differing k, keep direct ti
            // mapping when k equal (common case, including p3:L34).
            if observations.len() == gt_observations.len() {
                for obs in &gt_observations {
                    rp.gt_ranks.insert(obs.crop_index, 1);
                    if let Some(p) = obs.prob { rp.gt_probs.insert(obs.crop_index, p); }
                    else if obs.best_prob > 0.0 { rp.gt_probs.insert(obs.crop_index, obs.best_prob); }
                    if let Some(gs) = obs.glyph_score { rp.gt_glyph_scores.insert(obs.crop_index, gs); }
                    if let Some(h) = obs.geo_h_ll { rp.gt_geo_h_ll.insert(obs.crop_index, h); }
                    if let Some(v) = obs.geo_v_ll { rp.gt_geo_v_ll.insert(obs.crop_index, v); }
                    if let Some(he) = obs.geo_h_err { rp.gt_geo_h_err.insert(obs.crop_index, he); }
                    if let Some(ve) = obs.geo_v_err { rp.gt_geo_v_err.insert(obs.crop_index, ve); }
                }
            } else {
                // Differing k – align by glyph identity, not ordinal.
                let mut win_sorted: Vec<&font_match::ObservationDetail> = observations.iter().collect();
                win_sorted.sort_by_key(|o| o.crop_index);
                let mut gt_sorted: Vec<&font_match::ObservationDetail> = gt_observations.iter().collect();
                gt_sorted.sort_by_key(|o| o.crop_index);
                let mut gi = 0usize;
                for wo in win_sorted {
                    // If we've exhausted GT, remaining winner glyphs (e.g. ffi) have no GT counterpart.
                    if gi >= gt_sorted.len() { break; }
                    if gt_sorted[gi].ch == wo.ch {
                        let go = gt_sorted[gi];
                        rp.gt_ranks.insert(wo.crop_index, 1);
                        if let Some(p) = go.prob { rp.gt_probs.insert(wo.crop_index, p); }
                        else if go.best_prob > 0.0 { rp.gt_probs.insert(wo.crop_index, go.best_prob); }
                        if let Some(gs) = go.glyph_score { rp.gt_glyph_scores.insert(wo.crop_index, gs); }
                        if let Some(h) = go.geo_h_ll { rp.gt_geo_h_ll.insert(wo.crop_index, h); }
                        if let Some(v) = go.geo_v_ll { rp.gt_geo_v_ll.insert(wo.crop_index, v); }
                        if let Some(he) = go.geo_h_err { rp.gt_geo_h_err.insert(wo.crop_index, he); }
                        if let Some(ve) = go.geo_v_err { rp.gt_geo_v_err.insert(wo.crop_index, ve); }
                        gi += 1;
                    } else {
                        // Look ahead for same char – skip GT extras (e.g. f, fi before c).
                        let mut found: Option<usize> = None;
                        for look in gi+1..(gi+6).min(gt_sorted.len()) {
                            if gt_sorted[look].ch == wo.ch {
                                found = Some(look);
                                break;
                            }
                        }
                        if let Some(fidx) = found {
                            let go = gt_sorted[fidx];
                            rp.gt_ranks.insert(wo.crop_index, 1);
                            if let Some(p) = go.prob { rp.gt_probs.insert(wo.crop_index, p); }
                            else if go.best_prob > 0.0 { rp.gt_probs.insert(wo.crop_index, go.best_prob); }
                            if let Some(gs) = go.glyph_score { rp.gt_glyph_scores.insert(wo.crop_index, gs); }
                            if let Some(h) = go.geo_h_ll { rp.gt_geo_h_ll.insert(wo.crop_index, h); }
                            if let Some(v) = go.geo_v_ll { rp.gt_geo_v_ll.insert(wo.crop_index, v); }
                            if let Some(he) = go.geo_h_err { rp.gt_geo_h_err.insert(wo.crop_index, he); }
                            if let Some(ve) = go.geo_v_err { rp.gt_geo_v_err.insert(wo.crop_index, ve); }
                            gi = fidx + 1;
                        } else {
                            // No GT counterpart (e.g. winner ffi not supported by GT Caladea) – leave —.
                            // Do not advance gi, keep GT pointer on current for next winner char.
                        }
                    }
                }
            }
            rp
        };
        let alt_obs_rank_probs = ObsRankProbs::default();

        // Save crop PNGs and scan line image for ALL audited lines (not just
        // misses), so similarity-failure lines have crops in the report too.
        if let Some(ref ddir) = diag_seg_dir {
            if !observations.is_empty() {
                save_obs_crops(ddir, "crops", &observations, winning_crops);
            }
            // (old plain/lig dual path removed — per-font collapse is now inside identify_fonts;
            //  alt crops no longer exist, only winning path)

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
                // Reapply upward expansion by 20% to capture i-dots / diacritics
                // that sit above the tight word bbox (e.g. Georgia 'i' dot = 5px
                // above stem with 4px gap). Clamped by page bounds below.
                let h = sy1.saturating_sub(sy0);
                let up_expand = ((h as f32 * 0.20).ceil() as u32).max(1);
                sy0 = sy0.saturating_sub(up_expand);
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


        // ── GT font segmentation / crops – always emit even on misses ──
        // Build GT segs using same per-word word_infos + GT collapsed_lig_set, so k differs correctly
        // for ligature fonts (e.g. Office k=4 vs k=5).  No extra compute beyond build_for_font.
        let (gt_segs_opt, gt_wib_opt, gt_crops_opt) = if let Some(ref gfk) = gt_font_key {
            if let Some(gt_fe) = font_registry.by_key(gfk) {
                if let Some((s, w, c, _pm)) = build_for_font(gt_fe, &word_infos) {
                    (Some(s), Some(w), Some(c))
                } else { (None, None, None) }
            } else { (None, None, None) }
        } else { (None, None, None) };

        // If GT segs exist and diag dir exists, save GT crops alongside winner crops.
        // Use subdir "gt_crops" – report can read directly.  Also mirror into "crops"
        // with gt_ prefix for p4:L23 backward-compat naming reviewer expects.
        if let (Some(ref ddir), Some(ref g_segs), Some(ref g_crops)) = (diag_seg_dir.as_ref(), gt_segs_opt.as_ref(), gt_crops_opt.as_ref()) {
            // Save GT observation crops via a pseudo observation list derived from gt_observations
            // When gt_observations length matches char count, we can reuse; otherwise generate sequential.
            if !gt_observations.is_empty() && gt_observations.len() == g_crops.len() {
                // Reuse existing save helper – maps crop_index -> char
                save_obs_crops(ddir, "gt_crops", &gt_observations, g_crops);
            } else {
                // Fallback: synthesize minimal ObservationDetail list from segs
                let mut synth: Vec<font_match::ObservationDetail> = Vec::with_capacity(g_crops.len());
                let mut idx = 0usize;
                for (_si, ws) in g_segs.iter().enumerate() {
                    for &ch in ws.chars.iter() {
                        if idx >= g_crops.len() { break; }
                        if !crate::features::is_supported(ch) { continue; }
                        synth.push(font_match::ObservationDetail {
                            ch,
                            weight: 1.0,
                            crop_index: idx,
                            best_prob: 0.0,
                            passed_gate: true,
                            nearest: Vec::new(),
                            ocr_corrected_from: None,
                            best_alt_char: None,
                            best_alt_dist: None,
                            pflda_top_char: None,
                            pflda_top_p: None,
                            pflda_ocr_p: None,
                            pflda_replaced: false,
                            obs_stats: None,
                            glyph_score: None,
                            prob: None,
                            geo_h_ll: None,
                            geo_v_ll: None,
                            geo_h_err: None,
                            geo_v_err: None,
                        });
                        idx += 1;
                    }
                }
                if !synth.is_empty() {
                    save_obs_crops(ddir, "gt_crops", &synth, g_crops);
                }
            }
            // Also emit gt_ prefixed files into crops/ for direct side-by-side viz
            if let Ok(entries) = std::fs::read_dir(ddir.join("gt_crops")) {
                for ent in entries.flatten() {
                    let src = ent.path();
                    if let Some(fname) = src.file_name().and_then(|n| n.to_str()) {
                        let dst = ddir.join("crops").join(format!("gt_{fname}"));
                        let _ = std::fs::copy(&src, &dst);
                    }
                }
            }
        }


        let word_seg_summaries: Vec<crate::audit::WordSegSummary> = {
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

        let gt_word_seg_summaries: Vec<crate::audit::WordSegSummary> = if let Some(ref g_segs) = gt_segs_opt {
            g_segs.iter().map(|ws| crate::audit::WordSegSummary {
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
        } else {
            Vec::new()
        };

        let (midpoint_em_px, gt_midpoint_em_px) = {
            let mp = if let Some(ref fm) = font_result {
                crate::geometry_classifier::median_em_px_from_midpoints(&fm.font_key, &winning_segs, &winning_wib, geo_cache)
            } else { None };
            let gt_mp = if let Some(ref gfk) = gt_font_key {
                if let Some(ref gs) = gt_segs_opt {
                    if let Some(ref gw) = gt_wib_opt {
                        crate::geometry_classifier::median_em_px_from_midpoints(gfk, gs, gw, geo_cache)
                    } else {
                        crate::geometry_classifier::median_em_px_from_midpoints(gfk, &winning_segs, &winning_wib, geo_cache)
                    }
                } else {
                    crate::geometry_classifier::median_em_px_from_midpoints(gfk, &winning_segs, &winning_wib, geo_cache)
                }
            } else { None };
            (mp, gt_mp)
        };

        LineMatch { font_result, text_color, font_scores, observations, font_scores_lig, observations_lig, seg_winner, diag_seg_dir, obs_rank_probs, alt_obs_rank_probs, tie_candidates: tie_candidates_audit, corrected_words, fast_path: false, fast_path_score: None, midpoint_em_px, gt_midpoint_em_px, word_seg_summaries, gt_word_seg_summaries, ocr_corrections: ocr_correction_audit }
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
    let mut f = match unprint_fonts::ab_glyph::FontRef::try_from_slice(font_bytes.as_slice()) {
        Ok(f) => f,
        Err(_) => return fallback_pt,
    };
    if let Some(ref vars) = fm.variations {
        use unprint_fonts::ab_glyph::VariableFont;
        for (tag, val) in vars {
            f.set_variation(tag, *val);
        }
    }
    use unprint_fonts::ab_glyph::{Font, PxScale, ScaleFont};
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
