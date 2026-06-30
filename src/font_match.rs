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

/// Per-character CI detail, collected from the identify_glyph loop.
#[derive(Debug, Clone)]
pub struct CharMatchDetail {
    pub ch: char,
    pub crop_index: usize,
    pub best_prob: f32,
    pub passed_gate: bool,
    /// Top-3 fonts by probability (name, prob), highest first.
    pub nearest: Vec<(usize, f32)>,
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

/// Result of `identify_glyph`: ranked font scores + per-character CI detail.
/// scores are (font_key, aggregated_score) — globally consistent across characters.
/// char_detail.nearest still uses per-char glyph_ids (valid within each character).
#[derive(Debug)]
pub struct GlyphIdResult {
    pub scores: Vec<(String, f32)>,
    pub char_detail: Vec<CharMatchDetail>,
    /// Top score computed with uniform weights (all chars weight 1.0).
    /// Used for path comparison (ligature vs plain) so char_weight
    /// doesn't bias toward the ligature path.
    pub unweighted_top: f32,
}

pub fn identify_glyph(
    char_crops: &[(char, GrayImage)],
    _thoroughness: f32,
    _audit: bool,
    classifier: &dyn crate::classifier::Classifier,
    glyph_map: &crate::glyph_map::GlyphMap,
) -> GlyphIdResult {
    if char_crops.is_empty() {
        return GlyphIdResult { scores: Vec::new(), char_detail: Vec::new(), unweighted_top: f32::MIN };
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
        return GlyphIdResult { scores: Vec::new(), char_detail: Vec::new(), unweighted_top: f32::MIN };
    }

    let n_chars = crop_data.len();

    // ── Stage 1: per-crop classification → per-char glyph_ids ──────
    // For each crop, classifier picks top glyph_ids (per-char dense indices).
    // Expand each to font_keys via GlyphMap and union into candidate set.
    let mut candidate_set: HashSet<String> = HashSet::new();
    let mut char_detail: Vec<CharMatchDetail> = Vec::with_capacity(n_chars);

    for &(crop_idx, ch, ref raw_feat) in &crop_data {
        let picks = classifier.classify(ch, raw_feat, 3);
        if picks.is_empty() {
            continue;
        }

        // Expand per-char glyph_ids to font_keys
        for &(glyph_id, _prob) in &picks {
            for fk in glyph_map.fonts_for_glyph(ch, glyph_id) {
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
        return GlyphIdResult { scores: Vec::new(), char_detail, unweighted_top: f32::MIN };
    }

    // ── Stage 2: score each candidate font_key across all crops ────
    // For each font_key, look up its per-char glyph_id and get the
    // classifier probability. This is globally consistent because
    // font_keys are the same across characters.
    let mut best_unweighted = f32::MIN;
    let mut scores: Vec<(String, f32)> = candidate_set.into_iter()
        .filter_map(|font_key| {
            let log_probs: Vec<(f32, f32)> = crop_data.iter()
                .filter_map(|&(_, ch, ref feat)| {
                    let glyph_id = glyph_map.glyph_id_for_font(ch, &font_key)?;
                    let p = classifier.probability(ch, feat, glyph_id)?;
                    Some((p.max(1e-30).ln(), char_weight(ch)))
                })
                .collect();
            if log_probs.is_empty() {
                return None;
            }
            let score = aggregate_font_score(&log_probs, crop_data.len());
            if score.is_finite() {
                // Unweighted mean for path comparison (ignore char_weight bias)
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

    GlyphIdResult { scores, char_detail, unweighted_top: best_unweighted }
}

