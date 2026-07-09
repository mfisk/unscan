//! Font match result type and font identification.

use std::collections::HashSet;
use std::path::PathBuf;
use image::GrayImage;
use crate::features::{CropFeatures, compute_features};

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

pub fn aggregate_font_score(log_probs: &[(f32, f32)], n_windows: usize) -> f32 {
    let matched = log_probs.len();
    let mut total_weight = 0.0_f32;
    let mut weighted_sum = 0.0_f32;
    for &(lp, w) in log_probs {
        weighted_sum += lp * w;
        total_weight += w;
    }
    // Missing glyphs: probability 0 → log-prob −∞ → score −∞
    if matched < n_windows {
        return f32::NEG_INFINITY;
    }
    weighted_sum / total_weight.max(1e-9)
}

/// Compute the overall font score for a single font using calibrated
/// probabilities.  Returns `None` if the font has no data for any observation.

/// A single scored observation in the sliding-window pipeline.
/// Can be a 1-gram (single character) or 2-gram (bigram) or any length.
pub struct ScoringWindow<'a> {
    /// The character sequence.
    pub seq: Vec<char>,
    /// The cropped image for this observation.
    pub crop: &'a GrayImage,
    /// Weight for score aggregation (0.5 for unigram fallback, 1.0 for bigram).
    pub weight: f32,
}

// ---------------------------------------------------------------------------
// Matching — brute-force nearest-neighbor
// ---------------------------------------------------------------------------

/// Per-observation font-matching detail: one entry per scoring window (unigram or bigram).
#[derive(Debug, Clone)]
pub struct ObservationDetail {
    /// Full scored sequence (e.g. ['T','i'] for bigram, ['a'] for unigram).
    pub seq: Vec<char>,
    /// Weight used for this observation in scoring (0.5 for unigram, 1.0 for bigram).
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
}

/// Result of `identify_fonts`: ranked font scores + per-observation detail.
/// scores are (font_key, aggregated_score) — globally consistent across observations.
/// observations[].nearest still uses per-observation glyph_ids (valid within each seq's classifier).
#[derive(Debug)]
pub struct FontIdResult {
    pub scores: Vec<(String, f32)>,
    pub observations: Vec<ObservationDetail>,
    /// Top score computed with uniform weights (all observations weight 1.0).
    /// Used for path comparison (ligature vs plain) so observation weights
    /// don't bias the decision.
    pub unweighted_top: f32,
}

pub fn identify_fonts(
    windows: &[ScoringWindow],
    classifier: &dyn crate::classifier::Classifier,
    glyph_map: &crate::glyph_map::NgramGlyphMap,
    _thoroughness: f32,
    _audit: bool,
    ensure_font_keys: &[&str],
) -> FontIdResult {
    if windows.is_empty() {
        return FontIdResult { scores: Vec::new(), observations: Vec::new(), unweighted_top: f32::MIN };
    }

    // ── Pre-compute features ────────────────────────────────────────
    struct WindowData {
        window_idx: usize,
        seq: Vec<char>,
        feat: CropFeatures,
        weight: f32,
    }

    let crop_data: Vec<WindowData> = windows
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            let f = compute_features(w.crop, false)?;
            Some(WindowData { window_idx: i, seq: w.seq.clone(), feat: f, weight: w.weight })
        })
        .collect();

    if crop_data.is_empty() {
        return FontIdResult { scores: Vec::new(), observations: Vec::new(), unweighted_top: f32::MIN };
    }

    let n_windows = crop_data.len();

    // ── Stage 1: per-window classification → candidate set ─────────
    let mut candidate_set: HashSet<String> = HashSet::new();
    let mut observations: Vec<ObservationDetail> = Vec::with_capacity(n_windows);

    for wd in &crop_data {
        let picks = classifier.classify(&wd.seq, &wd.feat, 3);
        if picks.is_empty() {
            continue;
        }

        // Expand glyph_ids to font_keys
        for &(glyph_id, _prob) in &picks {
            for fk in glyph_map.fonts_for_glyph(&wd.seq, glyph_id) {
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
            seq: wd.seq.clone(),
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
        });
    }

    // Ensure requested font_keys are always scored (e.g. ground-truth font)
    for fk in ensure_font_keys {
        candidate_set.insert(fk.to_string());
    }

    if candidate_set.is_empty() {
        return FontIdResult { scores: Vec::new(), observations, unweighted_top: f32::MIN };
    }

    // ── Stage 2: score each candidate font_key across all windows ──
    let mut best_unweighted = f32::MIN;
    let mut scores: Vec<(String, f32)> = candidate_set.into_iter()
        .filter_map(|font_key| {
            let log_probs: Vec<(f32, f32)> = crop_data.iter()
                .filter_map(|wd| {
                    let glyph_id = glyph_map.glyph_id_for_font(&wd.seq, &font_key)?;
                    let p = classifier.probability(&wd.seq, &wd.feat, glyph_id)?;
                    Some((p.max(1e-30).ln(), wd.weight))
                })
                .collect();
            if log_probs.is_empty() {
                return None;
            }
            let score = aggregate_font_score(&log_probs, crop_data.len());
            if score.is_finite() {
                // Unweighted mean for path comparison (ignore weight bias)
                let uw = log_probs.iter().map(|(lp, _)| lp).sum::<f32>()
                    / log_probs.len() as f32;
                if uw > best_unweighted { best_unweighted = uw; }
                Some((font_key, score))
            } else { None }
        })
        .collect();

    // Sort descending (higher = better = closer match).
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    FontIdResult { scores, observations, unweighted_top: best_unweighted }
}

