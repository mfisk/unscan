//! Font match result type and font identification.
//!
//! Rule-out semantics: we effectively skip any font that does not contain a
//! character present in the string. If `per_char_geo_cached` /
//! `predict_glyph_positions_and_extents` returns `None` because the font lacks a
//! cmap entry for any required character, that character's geometry log-likelihood
//! is `-infinity` (infinitely bad). Therefore the whole-font log-likelihood is
//! `-infinity`. The font is inserted into `cannot_render: HashSet<String>` and
//! pruned as `f32::NEG_INFINITY`; `exp(-inf)=0` gives softmax probability 0, so
//! the font contributes 0 probability and is excluded from ranking. Empty `Vec`
//! is not an infinite penalty: it denotes ligature mismatch (no words had usable
//! geo, e.g. “ff” → single glyph) and returns `Some(empty)` as valid, falling
//! back to SSIM / n-gram only. Abort `None` is a valid short-circuit for an
//! infinitely bad score because a missing glyph makes rendering impossible.

use std::collections::HashSet;
use std::path::PathBuf;
use image::GrayImage;
use crate::features::{CropFeatures, compute_features};
use crate::classifier::{self, ObsStats};

const GEO_WEIGHT: f32 = 1.0;
// Aggregation mode: false = squared gap (current), true = simple sum (generative)
const USE_SUM_AGG: bool = true;

/// Midpoint pruning: worst log-prob of a letter in a correct font measured on BAP
/// specimen is -10.1537 (SourceSerif4-400 'T' p5:23, 416 font-correct letters,
/// ocr_correct=true, hit/minor_miss, p1=-4.99 p5=-1.90 median=-0.39). 
/// We prune fonts where min_{chars}(h_ll+v_ll) < threshold.
/// Threshold is scaled/loosened by --thoroughness: thr=1 => -12, thr=2 => -24 (looser),
/// thr=0.5 => -6 (tighter). Clamped to >=0.1 to avoid zero.
/// MIN_KEEP=10 ensures best_lps stability by keeping at least 10 best pruned fonts.
const MIDPOINT_PRUNE_BASE: f32 = -12.0;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FontMatchResult {
    pub font_name: String,
    pub font_path: PathBuf,
    /// Full font key (path + optional variant tag) for font lookups.
    pub font_key: String,
    /// OT variant tag (e.g. "smcp", "onum") — empty for base fonts.
    pub variant_tag: String,
    /// Glyph overrides for OT variant rendering.
    pub glyph_overrides: crate::font_scan::GlyphOverrides,
    /// Variable-font axis coordinates to apply before rendering.
    pub variations: crate::font_scan::Variations,
    pub score: f32,
    /// Best vertical pixel shift from SSIM alignment search (0 if coarse-only).
    pub best_dy: i32,
}

pub fn aggregate_font_score(log_probs: &[(f32, f32)], best_lps: &[f32]) -> f32 {
    if USE_SUM_AGG {
        // Generative: sum w*lp (proper log-likelihood under independence)
        return log_probs.iter().map(|&(lp, w)| lp * w).sum();
    }
    // Sum of squared deviations from the best log-prob at each observation,
    // weighted by observation weight.  Negated so higher = better.
    // Observations where all fonts score similarly contribute near-zero;
    // observations where this font falls behind contribute quadratically.
    debug_assert_eq!(log_probs.len(), best_lps.len());
    let penalty: f32 = log_probs.iter().enumerate()
        .map(|(i, &(lp, w))| {
            let gap = best_lps[i] - lp;
            gap * gap * w
        })
        .sum();
    -penalty
}

/// Compute the overall font score for a single font using calibrated
/// probabilities.  Returns `None` if the font has no data for any observation.

/// A single scored character.
pub struct ScoringWindow<'a> {
    pub ch: char,
    /// The cropped image for this observation.
    pub crop: &'a GrayImage,
    pub weight: f32,
}

// ---------------------------------------------------------------------------
// Matching — brute-force nearest-neighbor
// ---------------------------------------------------------------------------

/// Per-character font-matching detail.
#[derive(Debug, Clone)]
pub struct ObservationDetail {
    pub ch: char,
    /// Weight used for this observation in scoring.
    pub weight: f32,
    pub crop_index: usize,
    pub best_prob: f32,
    pub passed_gate: bool,
    /// Top-3 fonts by probability (name, prob), highest first.
    pub nearest: Vec<(usize, f32)>,
    /// When the OCR correction gate fires, the original OCR character
    /// that was replaced.  `seq` then holds the corrected sequence,
    /// and `nearest`/`best_prob` reflect the corrected observation's scoring.
    pub ocr_corrected_from: Option<char>,
    /// Best alternative character considered (even if correction gate
    /// didn't fire).  The character with the highest probability among
    /// all confusables/alternatives tested, if any were tested.
    pub best_alt_char: Option<char>,
    /// Distance of the best alternative character.
    pub best_alt_dist: Option<f32>,
    /// Per-font LDA top-1 predicted character.
    pub pflda_top_char: Option<char>,
    /// Per-font LDA probability of the top-1 prediction.
    pub pflda_top_p: Option<f32>,
    /// Per-font LDA probability of the OCR character.
    pub pflda_ocr_p: Option<f32>,
    /// Whether the pflda gate fired and replaced the OCR char.
    pub pflda_replaced: bool,
    /// Raw classifier distance stats (populated when UNPRINT_OBS_STATS=1).
    pub obs_stats: Option<ObsStats>,
}

/// Result of `identify_fonts`: ranked font scores + per-observation detail.
/// Minimum probability for an ngram observation to be included in font scoring.
/// Observations with p < this threshold are excluded as noise.
#[allow(dead_code)] // threshold for future ngram-based filtering
const MIN_NGRAM_PROB: f32 = 0.001;

/// scores are (font_key, aggregated_score) — globally consistent across observations.
/// observations[].nearest still uses per-observation glyph_ids (valid within each seq's classifier).
#[derive(Debug)]
pub struct FontIdResult {
    pub scores: Vec<(String, f32)>,
    pub observations: Vec<ObservationDetail>,
    /// Top score computed with uniform weights (all observations weight 1.0).
    /// Used for path comparison (ligature vs plain) so observation weights
    /// don't bias the decision.
    pub path_score: f32,
}

pub fn identify_fonts(
    windows: &[ScoringWindow],
    classifier: &dyn crate::classifier::Classifier,
    glyph_map: &crate::glyph_map::NgramGlyphMap,
    thoroughness: f32,
    audit: bool,
    ensure_font_keys: &[&str],
    min_ngram_prob: f32,
    word_segs: &[crate::segment::WordSeg],
    wib: &[crate::geometry_classifier::WordGeoMeasurement],
    font_registry: &crate::font_scan::FontRegistry,
    font_cache: &crate::font_cache::FontCache,
    geo_cache: &crate::geo_cache::GeometryCache,
    position_map: &[(usize, usize)],
) -> FontIdResult {
    if windows.is_empty() {
        return FontIdResult { scores: Vec::new(), observations: Vec::new(), path_score: f32::MIN };
    }

    // ── Pre-compute features ────────────────────────────────────────
    struct WindowData {
        window_idx: usize,
        ch: char,
        feat: CropFeatures,
        weight: f32,
        #[allow(dead_code)]
        ood_weight: f32,
    }

    let crop_data: Vec<WindowData> = windows
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            let f = compute_features(w.crop, false)?;
            Some(WindowData { window_idx: i, ch: w.ch, feat: f, weight: w.weight, ood_weight: 1.0 })
        })
        .collect();

    if crop_data.is_empty() {
        return FontIdResult { scores: Vec::new(), observations: Vec::new(), path_score: f32::MIN };
    }

    let n_windows = crop_data.len();
    // Minimum coverage: require at least 40% of windows or 3 observations
    // to prevent a font matching only 1/20 chars from getting a perfect
    // score (gap=0 on a single observation → score=0).
    let min_coverage = ((n_windows as f32 * 0.4).ceil() as usize)
        .max(3)
        .min(n_windows);

    // ── Stage 1: per-window classification → candidate set ─────────
    let mut candidate_set: HashSet<String> = HashSet::new();
    let mut observations: Vec<ObservationDetail> = Vec::with_capacity(n_windows);
    let mut ood_weights: Vec<f32> = Vec::with_capacity(n_windows);

    for wd in &crop_data {
        let seq = [wd.ch];
        let picks = classifier.classify(&seq, &wd.feat, 3);
        let ood_w = classifier::take_ood_weight();
        ood_weights.push(ood_w);
        if picks.is_empty() {
            continue;
        }

        // Expand glyph_ids to font_keys
        for &(glyph_id, _prob) in &picks {
            for fk in glyph_map.fonts_for_glyph(&seq, glyph_id) {
                candidate_set.insert(fk.clone());
            }
        }

        let best_prob = picks.iter()
            .map(|(_, p)| *p)
            .fold(0.0f32, f32::max);

        let nearest: Vec<(usize, f32)> = picks.iter()
            .take(3)
            .map(|&(id, p)| (id, p))
            .collect();

        observations.push(ObservationDetail {
            ch: wd.ch,
            weight: wd.weight,
            crop_index: wd.window_idx,
            best_prob,
            passed_gate: true,
            nearest,
            ocr_corrected_from: None,
            best_alt_char: None,
            best_alt_dist: None,
            pflda_top_char: None,
            pflda_top_p: None,
            pflda_ocr_p: None,
            pflda_replaced: false,
            obs_stats: classifier::take_obs_stats(),
        });
    }

    // Ensure requested font_keys are always scored (e.g. ground-truth font)
    for fk in ensure_font_keys {
        candidate_set.insert(fk.to_string());
    }

    // Apply font allowlist (fontkey format, exact match) - filters at matching time
    // This keeps main cache untouched when using default cache dir.
    if let Some(allow) = crate::cache::font_allowlist() {
        candidate_set.retain(|fk| allow.contains(fk));
    }

    if candidate_set.is_empty() {
        return FontIdResult { scores: Vec::new(), observations, path_score: f32::MIN };
    }

    // ── Per-char cache: compute logits/probs once, not per-candidate
    let mut window_logit_maps: Vec<std::collections::HashMap<usize, f32>> = Vec::with_capacity(crop_data.len());
    let mut window_prob_maps: Vec<std::collections::HashMap<usize, f32>> = Vec::with_capacity(crop_data.len());
    for wd in &crop_data {
        let seq = [wd.ch];
        let logits = classifier.raw_logits(&seq, &wd.feat);
        let probs = classifier.probabilities(&seq, &wd.feat);
        window_logit_maps.push(logits.into_iter().collect());
        window_prob_maps.push(probs.into_iter().collect());
    }

    // ── Stage 2a: filter chars — keep only chars where at least one
    // candidate font scores above (min_char_prob × uniform).
    let candidate_vec: Vec<String> = candidate_set.into_iter().collect();
    let scoring_window_indices: Vec<usize> = (0..crop_data.len())
        .filter(|&i| {
            let wd = &crop_data[i];
            let seq = [wd.ch];
            let n_glyphs = classifier.glyph_count(&seq).max(1) as f32;
            let threshold = min_ngram_prob / n_glyphs;
            let prob_map = &window_prob_maps[i];
            candidate_vec.iter().any(|font_key| {
                glyph_map.glyph_id_for_font(&seq, font_key)
                    .and_then(|gid| prob_map.get(&gid))
                    .map_or(false, |p| *p >= threshold)
            })
        })
        .collect();

    // ── Stage 2b: score each candidate font on the surviving chars ──
    let n_scoring = scoring_window_indices.len();

    // ── Geo precompute: per-font per-char geometry log-likelihoods ──
    let mut geo_per_font: std::collections::HashMap<String, std::collections::HashMap<(usize, usize), f32>> = std::collections::HashMap::new();
    let mut cannot_render: std::collections::HashSet<String> = std::collections::HashSet::new();
    if !word_segs.is_empty() && !wib.is_empty() {
        for font_key in &candidate_vec {
            if let Some(geos) = crate::geometry_classifier::per_char_geo_for_font(
                font_key, word_segs, wib, font_cache, geo_cache, font_registry
            ) {
                let mut map = std::collections::HashMap::new();
                for g in geos {
                    let ll = (g.h_ll + g.v_ll) as f32;
                    map.insert((g.seg_idx, g.orig_idx), ll);
                }
                geo_per_font.insert(font_key.clone(), map);
            } else {
                // Rule-out: per_char_geo returned None means the font lacks a cmap
                // entry for a required character, so it cannot render that char.
                // That char's geometry log-likelihood would be -infinity
                // (infinitely bad), making the whole-font score -infinity.
                // We short-circuit here — valid abort for an infinitely bad score —
                // and mark the font as cannot_render so it gets pruned as -inf.
                cannot_render.insert(font_key.clone());
            }
        }
    }

    // ── Midpoint pruning: prune fonts with very negative h_ll+v_ll ─────────
    // Worst correct-font letter on BAP is -10.1537 (416 letters, ocr_correct, hit/minor_miss).
    // Threshold = BASE * thoroughness, thr=1 => -12, thr=2 => -24 looser, thr=0.5 => -6 tighter.
    // We use min_ll (worst letter) to decide; keep ensure keys and fonts with no geo data.
    // To keep best_lps stable across prune levels, we also keep at least 10 fonts with
    // highest min_ll (least negative) so per-position best doesn't shift dramatically.
    let prune_threshold = MIDPOINT_PRUNE_BASE * thoroughness.max(0.1);
    let total_candidates = candidate_vec.len();
    let mut kept_candidates: Vec<String> = Vec::with_capacity(total_candidates);
    let mut pruned_with_ll: Vec<(String, f32)> = Vec::new();
    let mut pruned_count: usize = 0;
    for fk in &candidate_vec {
        if ensure_font_keys.contains(&fk.as_str()) {
            kept_candidates.push(fk.clone());
            continue;
        }
        if cannot_render.contains(fk) {
            // Rule-out short-circuit: font cannot render a required char → its
            // per-char geometry ll is -infinity, so its total score is
            // -infinity (infinitely bad). We prune it here as NEG_INFINITY;
            // aborting instead of scoring the rest is valid because an
            // infinitely bad component dominates the sum.
            pruned_count += 1;
            pruned_with_ll.push((fk.clone(), f32::NEG_INFINITY));
            continue;
        }
        if let Some(gmap) = geo_per_font.get(fk) {
            if gmap.is_empty() {
                kept_candidates.push(fk.clone());
                continue;
            }
            // min_ll = worst (most negative) midpoint log-prob for this font on this line
            let min_ll = gmap.values().cloned().fold(f32::INFINITY, f32::min);
            if min_ll < prune_threshold {
                pruned_count += 1;
                pruned_with_ll.push((fk.clone(), min_ll));
                continue;
            }
        }
        // No geo data -> cannot evaluate, keep for safety
        kept_candidates.push(fk.clone());
    }
    // Keep at least MIN_KEEP best pruned fonts by min_ll to stabilize per-position best
    const MIN_KEEP: usize = 10;
    if kept_candidates.len() < MIN_KEEP && !pruned_with_ll.is_empty() {
        pruned_with_ll.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let need = MIN_KEEP.saturating_sub(kept_candidates.len());
        for (fk, _ll) in pruned_with_ll.iter().take(need) {
            kept_candidates.push(fk.clone());
            pruned_count = pruned_count.saturating_sub(1);
        }
    }
    // Avoid empty candidate set: keep at least the best by max geo (least negative min_ll)
    let candidate_vec: Vec<String> = if kept_candidates.is_empty() && !candidate_vec.is_empty() {
        let mut best: Option<(String, f32)> = None;
        for fk in &candidate_vec {
            let best_ll_for_font = geo_per_font
                .get(fk)
                .map(|m| m.values().cloned().fold(f32::NEG_INFINITY, f32::max))
                .unwrap_or(f32::NEG_INFINITY);
            if best.is_none() || best_ll_for_font > best.as_ref().unwrap().1 {
                best = Some((fk.clone(), best_ll_for_font));
            }
        }
        if let Some((bfk, _)) = best {
            pruned_count = total_candidates.saturating_sub(1);
            vec![bfk]
        } else {
            kept_candidates
        }
    } else {
        kept_candidates
    };
    if pruned_count > 0 && audit {
        eprintln!(
            "midpoint prune: pruned {}/{} fonts at threshold {:.2} (base {:.1} * thoroughness {:.2})",
            pruned_count, total_candidates, prune_threshold, MIDPOINT_PRUNE_BASE, thoroughness
        );
    }

    // First pass: collect log-probs for all candidate fonts
    let font_lps: Vec<(String, Vec<(f32, f32)>)> = candidate_vec.into_iter()
        .filter_map(|font_key| {
            let log_probs: Vec<(f32, f32)> = scoring_window_indices.iter()
                .filter_map(|&i| {
                    let wd = &crop_data[i];
                    let seq = [wd.ch];
                    let glyph_id = glyph_map.glyph_id_for_font(&seq, &font_key)?;
                    // Use raw logit -d²/(2σ²)
                    let logit = *window_logit_maps[i].get(&glyph_id)?;
                    let mut lp = logit;
                    // Geo scoring: per character, scaled by GEO_WEIGHT
                    if let Some(geo_map) = geo_per_font.get(&font_key) {
                        if let Some(&(seg_idx, char_pos)) = position_map.get(wd.window_idx) {
                            if let Some(&ll) = geo_map.get(&(seg_idx, char_pos)) {
                                lp += GEO_WEIGHT * ll;
                            }
                        }
                    }
                    Some((lp, wd.weight * ood_weights[i]))
                })
                .collect();
            if log_probs.len() < n_scoring { return None; }
            // Require minimum character coverage so a font matching only
            // 1/20 windows cannot get a perfect score.
            if log_probs.len() < min_coverage { return None; }
            Some((font_key, log_probs))
        })
        .collect();

    // Find best log-prob per observation across all candidate fonts
    let mut best_lps = vec![f32::NEG_INFINITY; n_scoring];
    for (_, lps) in &font_lps {
        for (i, &(lp, _)) in lps.iter().enumerate() {
            if lp > best_lps[i] { best_lps[i] = lp; }
        }
    }

    // Second pass: score using squared deviations from per-observation best
    let mut best_path_score = f32::MIN;
    let mut scores: Vec<(String, f32)> = font_lps.into_iter()
        .filter_map(|(font_key, log_probs)| {
            let score = aggregate_font_score(&log_probs, &best_lps);
            if score.is_finite() {
                // Path comparison score: OOD-weighted (data quality) but not
                // position-weighted, so garbage observations are downweighted
                // without position bias affecting lig-vs-plain selection.
                let ood_probs: Vec<(f32, f32)> = scoring_window_indices.iter()
                    .zip(log_probs.iter())
                    .map(|(&i, &(lp, _))| (lp, ood_weights[i]))
                    .collect();
                let ps = aggregate_font_score(&ood_probs, &best_lps);
                if ps > best_path_score { best_path_score = ps; }
                Some((font_key, score))
            } else { None }
        })
        .collect();

    // Sort descending (higher = better = closer match). Unstable sort: equal-score order irrelevant.
    scores.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    FontIdResult { scores, observations, path_score: best_path_score }
}

