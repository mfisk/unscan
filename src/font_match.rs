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

use rustc_hash::{FxHashMap, FxHashSet};
use std::path::PathBuf;
use image::GrayImage;
use crate::features::{CropFeatures, compute_features};
use crate::classifier::{self, ObsStats};

const GEO_WEIGHT: f32 = 1.0;
// Aggregation mode: false = squared gap (current), true = simple sum (generative)
const USE_SUM_AGG: bool = true;

/// Midpoint pruning (geo-first): prune fonts where min_{chars}(h_ll+v_ll) < threshold.
/// Threshold = BASE * thoroughness. Base -12.0 keeps worst correct
/// letter -10.1537 (SourceSerif4-400 'T') with margin. Sweep showed -7.2
/// prunes more but caused regression on Mayr-Duffner lig path where math
/// fonts without ff ligature get empty geo (0 penalty) and beat GT.
/// Restoring -12.0 + MIN_KEEP=10 stabilizes best_lps and avoids
/// 2742/2743 catastrophic prune on plain path where -1381 sentinels hide
/// behind weight 0.
/// Clamped thoroughness >=0.1.
const MIDPOINT_PRUNE_BASE: f32 = -12.0;
const MIN_KEEP: usize = 10;

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

pub fn identify_fonts<'a>(
    windows: &'a [ScoringWindow<'a>],
    classifier: &dyn crate::classifier::Classifier,
    glyph_map: &'a crate::glyph_map::NgramGlyphMap,
    thoroughness: f32,
    audit: bool,
    ensure_font_keys: &'a [&'a str],
    min_ngram_prob: f32,
    word_segs: &'a [crate::segment::WordSeg],
    wib: &'a [crate::geometry_classifier::WordGeoMeasurement],
    font_registry: &'a crate::font_scan::FontRegistry,
    font_cache: &'a crate::font_cache::FontCache,
    geo_cache: &'a crate::geo_cache::GeometryCache,
    position_map: &'a [(usize, usize)],
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

    // ── Stage 1: per-window classification for observations (glyph candidate set kept only for fallback) ──
    let mut observations: Vec<ObservationDetail> = Vec::with_capacity(n_windows);
    let mut ood_weights: Vec<f32> = Vec::with_capacity(n_windows);
    let mut glyph_candidate_set: FxHashSet<&'a str> = FxHashSet::default();

    for wd in &crop_data {
        let seq = [wd.ch];
        let picks = classifier.classify(&seq, &wd.feat, 3);
        let ood_w = classifier::take_ood_weight();
        ood_weights.push(ood_w);
        if picks.is_empty() {
            continue;
        }

        for &(glyph_id, _prob) in &picks {
            for fk in glyph_map.fonts_for_glyph(&seq, glyph_id) {
                glyph_candidate_set.insert(fk.as_str());
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

    // ── Per-char cache: compute logits/probs once, not per-candidate ──
    let mut window_logit_maps: Vec<Vec<Option<f32>>> = Vec::with_capacity(crop_data.len());
    let mut window_prob_maps: Vec<Vec<Option<f32>>> = Vec::with_capacity(crop_data.len());
    for wd in &crop_data {
        let seq = [wd.ch];
        let logits = classifier.raw_logits(&seq, &wd.feat);
        let count = classifier.glyph_count(&seq).max(1);
        let max_gid = logits.iter().map(|(gid, _)| *gid).max().unwrap_or(0);
        let vec_len = count.max(max_gid + 1);
        let mut logit_vec: Vec<Option<f32>> = vec![None; vec_len];
        let mut prob_vec: Vec<Option<f32>> = vec![None; vec_len];
        if logits.is_empty() {
            window_logit_maps.push(logit_vec);
            window_prob_maps.push(prob_vec);
            continue;
        }
        let max_logit = logits.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut exps: Vec<f32> = Vec::with_capacity(logits.len());
        for (_, logit) in &logits {
            let e = (*logit - max_logit).exp();
            sum_exp += e;
            exps.push(e);
        }
        let uniform = sum_exp < 1e-30;
        let uniform_p = if uniform { 1.0 / exps.len() as f32 } else { 0.0 };
        for (i, (gid, logit)) in logits.into_iter().enumerate() {
            if gid >= logit_vec.len() {
                let new_len = gid + 1;
                logit_vec.resize(new_len, None);
                prob_vec.resize(new_len, None);
            }
            logit_vec[gid] = Some(logit);
            let p = if uniform { uniform_p } else { exps[i] / sum_exp };
            prob_vec[gid] = Some(p);
        }
        window_logit_maps.push(logit_vec);
        window_prob_maps.push(prob_vec);
    }

    // ── Geo-first: score geometry before any glyph work ──
    // No MIN_KEEP - threshold alone decides. Vertical-first short-circuit inside
    // per_char_geo (h<=0, so if v < threshold, sum will be < threshold).
    let prune_threshold = MIDPOINT_PRUNE_BASE * thoroughness.max(0.1);
    let mut geo_per_font: FxHashMap<&'a str, FxHashMap<(usize, usize), f32>> = FxHashMap::default();
    let mut cannot_render: FxHashSet<&'a str> = FxHashSet::default();
    let mut kept_candidates: Vec<&'a str> = Vec::new();
    let mut pruned_with_ll: Vec<(&'a str, f32)> = Vec::new();
    let mut pruned_count: usize = 0;
    let mut total_candidates: usize = 0;

    let mut ensure_set: FxHashSet<&'a str> = FxHashSet::default();
    for &fk in ensure_font_keys {
        ensure_set.insert(fk);
    }
    let allow_opt = crate::cache::font_allowlist();

    if !word_segs.is_empty() && !wib.is_empty() {
        total_candidates = font_registry.iter().len();
        // Universe = all registry fonts (filtered by allowlist, but ensure bypasses)
        for entry in font_registry.iter() {
            let fk: &'a str = entry.font_key_ref();
            if let Some(ref allow) = allow_opt {
                if !allow.contains(fk) && !ensure_set.contains(fk) {
                    continue;
                }
            }
            let is_ensure = ensure_set.contains(fk);
            // Compute geometry without early threshold abort — threshold is applied here
            // based on min_ll, so we keep stable best_lps and allow ensure recovery.
            // per_char_geo_for_font_with_threshold no longer aborts early (removed
            // vertical-first short-circuit that produced -690 sentinels as infinite penalty).
            let geo_opt = crate::geometry_classifier::per_char_geo_for_font(
                fk, word_segs, wib, font_cache, geo_cache, font_registry
            );
            match geo_opt {
                None => {
                    if is_ensure {
                        // Ensure fonts are never pruned — keep glyph-only if truly cannot render
                        kept_candidates.push(fk);
                    } else {
                        cannot_render.insert(fk);
                        pruned_count += 1;
                    }
                }
                Some(geos) => {
                    if geos.is_empty() {
                        // Empty after fix should only happen for plain-path non-lig mismatch
                        // (e.g. single-word line where plain "ff" → 1 glyph). Keep glyph-only;
                        // for lig path we now return None on lig mismatch, so empty won't occur.
                        geo_per_font.insert(fk, FxHashMap::default());
                        kept_candidates.push(fk);
                        continue;
                    }
                    let mut map = FxHashMap::default();
                    let mut min_ll = f32::INFINITY;
                    for g in &geos {
                        let ll = (g.h_ll + g.v_ll) as f32;
                        if ll < min_ll {
                            min_ll = ll;
                        }
                        map.insert((g.seg_idx, g.orig_idx), ll);
                    }
                    if !is_ensure && min_ll < prune_threshold {
                        pruned_with_ll.push((fk, min_ll));
                        pruned_count += 1;
                        continue;
                    }
                    geo_per_font.insert(fk, map);
                    kept_candidates.push(fk);
                }
            }
        }
        // Keep at least MIN_KEEP best pruned fonts to stabilize per-position best_lps
        if kept_candidates.len() < MIN_KEEP && !pruned_with_ll.is_empty() {
            pruned_with_ll.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let need = MIN_KEEP.saturating_sub(kept_candidates.len());
            for (fk, ll) in pruned_with_ll.iter().take(need) {
                // Recompute map for restored font (we already pruned it, need to rebuild)
                if let Some(geos) = crate::geometry_classifier::per_char_geo_for_font(
                    fk, word_segs, wib, font_cache, geo_cache, font_registry
                ) {
                    let mut map = FxHashMap::default();
                    for g in geos {
                        map.insert((g.seg_idx, g.orig_idx), (g.h_ll + g.v_ll) as f32);
                    }
                    geo_per_font.insert(fk, map);
                }
                kept_candidates.push(*fk);
                pruned_count = pruned_count.saturating_sub(1);
            }
        }
        // Ensure keys that may not be in registry (variants)
        for &fk in ensure_font_keys {
            if !kept_candidates.contains(&fk) {
                if let Some(geos) = crate::geometry_classifier::per_char_geo_for_font(
                    fk, word_segs, wib, font_cache, geo_cache, font_registry
                ) {
                    let mut map = FxHashMap::default();
                    for g in geos {
                        map.insert((g.seg_idx, g.orig_idx), (g.h_ll + g.v_ll) as f32);
                    }
                    geo_per_font.insert(fk, map);
                }
                kept_candidates.push(fk);
            }
        }
    } else {
        // No geometry available: fallback to old glyph candidate set
        let mut candidate_set = glyph_candidate_set;
        for &fk in ensure_font_keys {
            candidate_set.insert(fk);
        }
        if let Some(ref allow) = allow_opt {
            candidate_set.retain(|fk| allow.contains(*fk) || ensure_set.contains(*fk));
        }
        if candidate_set.is_empty() {
            return FontIdResult { scores: Vec::new(), observations, path_score: f32::MIN };
        }
        kept_candidates = candidate_set.into_iter().collect();
        total_candidates = kept_candidates.len();
    }

    let candidate_vec: Vec<&'a str> = kept_candidates;
    if candidate_vec.is_empty() {
        return FontIdResult { scores: Vec::new(), observations, path_score: f32::MIN };
    }
    if pruned_count > 0 && audit {
        eprintln!(
            "midpoint prune: pruned {}/{} fonts at threshold {:.2} (base {:.1} * thoroughness {:.2})",
            pruned_count, total_candidates, prune_threshold, MIDPOINT_PRUNE_BASE, thoroughness
        );
    }

    // ── Stage 2a: filter chars — keep only chars where at least one geo-kept font scores above threshold
    let scoring_window_indices: Vec<usize> = (0..crop_data.len())
        .filter(|&i| {
            let wd = &crop_data[i];
            let seq = [wd.ch];
            let n_glyphs = classifier.glyph_count(&seq).max(1) as f32;
            let threshold = min_ngram_prob / n_glyphs;
            let prob_vec = &window_prob_maps[i];
            candidate_vec.iter().any(|&font_key| {
                glyph_map.glyph_id_for_font(&seq, font_key)
                    .and_then(|gid| prob_vec.get(gid).copied().flatten())
                    .map_or(false, |p| p >= threshold)
            })
        })
        .collect();

    // ── Stage 2b: score each candidate font on the surviving chars ──
    let n_scoring = scoring_window_indices.len();

    // First pass: collect log-probs for all candidate fonts
    let font_lps: Vec<(&'a str, Vec<(f32, f32)>)> = candidate_vec.into_iter()
        .filter_map(|font_key| {
            let log_probs: Vec<(f32, f32)> = scoring_window_indices.iter()
                .filter_map(|&i| {
                    let wd = &crop_data[i];
                    let seq = [wd.ch];
                    let glyph_id = glyph_map.glyph_id_for_font(&seq, font_key)?;
                    // Dense Vec<Option<f32>> → preserve None as missing, same as HashMap get(&gid)? 
                    let logit = window_logit_maps[i].get(glyph_id).copied().flatten()?;
                    let mut lp = logit;
                    // Geo scoring: per character, scaled by GEO_WEIGHT
                    if let Some(geo_map) = geo_per_font.get(font_key) {
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
    // Perf: sort borrowed &str keys to avoid moving/cloning String during sort,
    // then materialize owned Strings after sort (preserves output, avoids String moves).
    let mut best_path_score = f32::MIN;
    let mut borrowed_scores: Vec<(&'a str, f32)> = font_lps.into_iter()
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

    // Sort descending (higher = better = closer match). Deterministic tie-break by name.
    borrowed_scores.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });

    let scores: Vec<(String, f32)> = borrowed_scores.into_iter()
        .map(|(k, s)| (k.to_owned(), s))
        .collect();

    FontIdResult { scores, observations, path_score: best_path_score }
}

