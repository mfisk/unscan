//! Font match result type and font identification.

use std::collections::HashSet;
use std::path::PathBuf;
use image::GrayImage;
use crate::features::{CharFeatures, compute_features};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FontMatchResult {
    pub font_name: String,
    pub font_path: PathBuf,
    /// Full font key (path + optional variant tag) for CI lookups.
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

pub fn aggregate_font_score(log_probs: &[(f32, f32)], n_total_chars: usize) -> f32 {
    let matched = log_probs.len();
    let mut total_weight = 0.0_f32;
    let mut weighted_sum = 0.0_f32;
    for &(lp, w) in log_probs {
        weighted_sum += lp * w;
        total_weight += w;
    }
    // Missing glyphs: probability 0 → log-prob −∞ → score −∞
    if matched < n_total_chars {
        return f32::NEG_INFINITY;
    }
    weighted_sum / total_weight.max(1e-9)
}

/// Compute the overall CI score for a single font using calibrated
/// probabilities.  Returns `None` if the font has no data for any crop char.
fn score_font(
    classifier: &dyn crate::classifier::Classifier,
    font_id: usize,
    crop_data: &[(usize, char, CharFeatures)],
) -> Option<f32> {
    let log_probs: Vec<(f32, f32)> = crop_data.iter()
        .filter_map(|&(_, ch, ref feat)| {
            let p = classifier.probability(ch, feat, font_id)?;
            // Clamp to avoid ln(0); 1e-30 is ~−69 in log space
            Some((p.max(1e-30).ln(), char_weight(ch)))
        })
        .collect();
    if log_probs.is_empty() {
        return None;
    }
    let score = aggregate_font_score(&log_probs, crop_data.len());
    if score.is_finite() { Some(score) } else { None }
}

/// Character discriminativeness weight for scoring.
pub fn char_weight(c: char) -> f32 {
    match c {
        'g' | 'a' | 'e' | 'R' | 'Q' | 'G' | 'S' | 'f' | 't' | 'y' | '&' | '@' => 1.5,
        'I' | 'l' | '1' | '|' | '!' | '.' | ',' | ':' | ';' | '-' => 0.5,
        'b' | 'd' | 'p' | 'q' | 'n' | 'u' | 'o' | 'c' | 'O' | 'C' | 'D' => 0.8,
        'k' | 'w' | 'x' | 'z' | 'A' | 'B' | 'E' | 'F' | 'K' | 'M' | 'N' | 'W' => 1.2,
        // Ligatures — highly discriminative (font either has the glyph or doesn't)
        '\u{FB00}' | '\u{FB01}' | '\u{FB02}' | '\u{FB03}' | '\u{FB04}' => 2.0,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Matching — brute-force nearest-neighbor
// ---------------------------------------------------------------------------

/// Per-character CI detail, collected from the identify_font loop.
#[derive(Debug, Clone)]
pub struct CharMatchDetail {
    pub ch: char,
    pub crop_index: usize,
    pub best_prob: f32,
    pub passed_gate: bool,
    /// Top-3 fonts by probability (name, prob), highest first.
    pub nearest: Vec<(String, f32)>,
    /// When the OCR correction gate fires, the original OCR character
    /// that was replaced.  `ch` then holds the corrected character,
    /// and `nearest`/`best_prob` reflect the corrected char's CI.
    pub ocr_corrected_from: Option<char>,
    /// Best alternative character considered (even if correction gate
    /// didn't fire).  Always the char with the highest probability among
    /// all confusables/alternatives tested, if any were tested.
    pub best_alt_char: Option<char>,
    /// Distance of the best alternative character.
    pub best_alt_dist: Option<f32>,
}

/// Result of `identify_font`: ranked font scores + per-character CI detail.
#[derive(Debug)]
pub struct FontIdResult {
    pub scores: Vec<(String, f32)>,
    pub char_detail: Vec<CharMatchDetail>,
}

pub fn identify_font(
    char_crops: &[(char, GrayImage)],
    _thoroughness: f32,
    _audit: bool,
    classifier: &dyn crate::classifier::Classifier,
) -> FontIdResult {
    if char_crops.is_empty() {
        return FontIdResult { scores: Vec::new(), char_detail: Vec::new() };
    }

    // ── Pre-compute features ────────────────────────────────────────
    let crop_data: Vec<(usize, char, CharFeatures)> = char_crops
        .iter()
        .enumerate()
        .filter_map(|(i, (c, img))| {
            let f = compute_features(img, false)?;
            Some((i, *c, f))
        })
        .collect();

    if crop_data.is_empty() {
        return FontIdResult { scores: Vec::new(), char_detail: Vec::new() };
    }

    let n_chars = crop_data.len();

    // ── Stage 1: per-crop classification ────────────────────────────
    // For each crop, the classifier picks the best font(s).
    // Union all picks into the candidate set.
    let mut candidate_set: HashSet<usize> = HashSet::new();
    let mut char_detail: Vec<CharMatchDetail> = Vec::with_capacity(n_chars);

    for &(crop_idx, ch, ref raw_feat) in &crop_data {
        // Classifier picks top fonts — it owns the font vectors internally
        let picks = classifier.classify(ch, raw_feat, 3);
        if picks.is_empty() {
            continue;
        }

        for &(font_id, _prob) in &picks {
            candidate_set.insert(font_id);
        }

        let best_prob = picks.iter()
            .map(|(_, p)| *p)
            .fold(0.0f32, f32::max);

        let nearest: Vec<(String, f32)> = picks.iter()
            .take(3)
            .filter_map(|(id, p)| {
                Some((classifier.font_name(*id)?.to_string(), *p))
            })
            .collect();

        char_detail.push(CharMatchDetail {
            ch,
            crop_index: crop_idx,
            best_prob,
            passed_gate: true,
            nearest,
            ocr_corrected_from: None,
            best_alt_char: None,
            best_alt_dist: None,
        });
    }

    if candidate_set.is_empty() {
        return FontIdResult { scores: Vec::new(), char_detail };
    }

    // ── Stage 2: score each candidate across all crops ──────────────
    let mut scores: Vec<(String, f32)> = candidate_set.iter()
        .filter_map(|&font_id| {
            let name = classifier.font_name(font_id)?.to_string();
            let score = score_font(
                classifier, font_id, &crop_data,
            )?;
            Some((name, score))
        })
        .collect();

    scores.retain(|(_, s)| s.is_finite());

    // Sort descending (higher = better = closer match).
    // Tiebreaker: prefer base (untagged) font over OT variants.
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.contains('|').cmp(&b.0.contains('|')))
    });

    // No candidate pruning — SSIM is the real arbiter.

    FontIdResult { scores, char_detail }
}

