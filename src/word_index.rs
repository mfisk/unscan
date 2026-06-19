//! Word-level font index.
//!
//! Instead of comparing per-character crops (which depend on fragile Tesseract
//! char-level bounding boxes), this module builds an index of **common words**
//! rendered in every font.  At match time we crop whole words (reliable bboxes),
//! compute a visual fingerprint (downscaled thumbnail), and look up nearest fonts.
//!
//! The feature vector is a simple downscaled image — not the char-designed features
//! from char_index (which lose discriminative power on multi-character words).
//!
//! The word index is keyed by lowercase word text.  Only words that appear in
//! the common-words list get indexed and queried.

use std::collections::HashMap;

use ab_glyph::{Font, FontRef};
use image::GrayImage;

use crate::font_scan::FontEntry;

// ---------------------------------------------------------------------------
// Common words — high-frequency English words that appear in most text
// ---------------------------------------------------------------------------

const COMMON_WORDS: &[&str] = &[
    "the", "and", "for", "that", "with", "this", "from", "have", "was",
    "are", "been", "were", "its", "has", "not", "but", "can", "had",
    "one", "our", "out", "you", "all", "her", "she", "him", "his",
    "how", "man", "new", "now", "old", "see", "way", "who", "did",
    "get", "let", "say", "may", "two", "more", "some", "time", "very",
    "when", "come", "make", "like", "long", "look", "many", "over",
    "such", "take", "than", "them", "then", "will", "each", "made",
    "after", "also", "back", "most", "much", "only", "other", "right",
    "still", "their", "these", "think", "those", "under", "where",
    "which", "while", "world", "about", "could", "every", "first",
    "found", "great", "house", "large", "never", "since", "small",
    "state", "there", "three", "water", "would", "people", "should",
    "before", "between", "through",
    // Add uppercase variants for section headers
    "The", "And", "For", "This", "From", "With", "Bold", "Italic",
    "Regular", "Font", "Lining",
];

/// Thumbnail grid dimensions for visual fingerprint.
/// Height kept small, width proportional — captures the overall visual rhythm.
const THUMB_H: u32 = 12;
const THUMB_W: u32 = 48;

/// Feature vector length = THUMB_W * THUMB_H (flattened pixel grid).
pub const WORD_FEAT_LEN: usize = (THUMB_W * THUMB_H) as usize;

/// Normalized height for word rendering in the index.
const WORD_NORM_H: u32 = 48;

/// Minimum word length to index/query.
const MIN_WORD_LEN: usize = 3;

// ---------------------------------------------------------------------------
// Feature computation — visual thumbnail fingerprint
// ---------------------------------------------------------------------------

/// Compute a visual fingerprint for a word image.
///
/// 1. Trim whitespace
/// 2. Resize to THUMB_W × THUMB_H (stretch to fit — same word, same target size)
/// 3. Normalize pixel values to [0, 1] range
/// 4. Return as flat f32 array
fn compute_word_features(img: &GrayImage) -> Option<[f32; WORD_FEAT_LEN]> {
    let trimmed = crate::ssim::trim_whitespace_simple(img);
    let (w, h) = trimmed.dimensions();
    if w < 3 || h < 3 {
        return None;
    }

    // Resize to fixed grid — preserves visual shape while normalizing dimensions
    let resized = image::imageops::resize(
        &trimmed,
        THUMB_W,
        THUMB_H,
        image::imageops::FilterType::Lanczos3,
    );

    let mut features = [0.0f32; WORD_FEAT_LEN];
    for (i, px) in resized.pixels().enumerate() {
        // Invert: dark ink = high value (like char_index convention)
        features[i] = (255.0 - px.0[0] as f32) / 255.0;
    }

    Some(features)
}

// ---------------------------------------------------------------------------
// Index structure
// ---------------------------------------------------------------------------

/// Pre-computed word-level font index.
pub struct WordIndex {
    /// word_text (lowercase) → Vec<(font_id, features)>
    entries: HashMap<String, Vec<(usize, [f32; WORD_FEAT_LEN])>>,
    /// font_id → font name
    pub font_names: Vec<String>,
}

impl WordIndex {
    pub fn is_queryable(&self, word: &str) -> bool {
        word.len() >= MIN_WORD_LEN && self.entries.contains_key(&word.to_lowercase())
    }
}

// ---------------------------------------------------------------------------
// Index building
// ---------------------------------------------------------------------------

/// Build the word index from the system font catalog.
pub fn build_word_index(
    font_catalog: &[FontEntry],
) -> WordIndex {
    let words_set: Vec<&str> = COMMON_WORDS.iter()
        .filter(|w| w.len() >= MIN_WORD_LEN)
        .copied()
        .collect();


    let mut entries: HashMap<String, Vec<(usize, [f32; WORD_FEAT_LEN])>> = HashMap::new();
    let mut font_names: Vec<String> = Vec::with_capacity(font_catalog.len());

    for (font_id, fe) in font_catalog.iter().enumerate() {
        let font_key = fe.font_key();
        font_names.push(font_key.clone());

        let font = match FontRef::try_from_slice(&fe.data) {
            Ok(f) => f,
            Err(_) => continue,
        };

        for &word in &words_set {
            let rendered = match render_word_for_index(&font, word, WORD_NORM_H) {
                Some(img) => img,
                None => continue,
            };

            let features = match compute_word_features(&rendered) {
                Some(f) => f,
                None => continue,
            };

            let key = word.to_lowercase();
            entries.entry(key).or_default().push((font_id, features));
        }
    }

    let total: usize = entries.values().map(|v| v.len()).sum();

    WordIndex { entries, font_names }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Search the word index for candidate fonts matching observed word crops.
///
/// `word_crops` is a list of (word_text, cropped_image) pairs from the scan.
/// Returns a ranked list of (font_name, score) with higher = better.
pub fn search_word_index(
    index: &WordIndex,
    word_crops: &[(String, GrayImage)],
) -> Vec<(String, f32)> {
    if word_crops.is_empty() {
        return Vec::new();
    }

    // Compute features for each queryable word crop
    let crop_feats: Vec<(&str, [f32; WORD_FEAT_LEN])> = word_crops
        .iter()
        .filter(|(text, _)| index.is_queryable(text))
        .filter_map(|(text, img)| {
            compute_word_features(img).map(|f| (text.as_str(), f))
        })
        .collect();

    if crop_feats.is_empty() {
        return Vec::new();
    }

    let n_words = crop_feats.len();
    let quorum = ((n_words + 1) / 2).max(1);

    // For each word, find nearest fonts and accumulate distances
    let mut font_log_dists: HashMap<usize, Vec<f32>> = HashMap::new();

    for (text, query_feat) in &crop_feats {
        let key = text.to_lowercase();
        let candidates = match index.entries.get(&key) {
            Some(c) => c,
            None => continue,
        };

        // Find nearest neighbor distance for radius calculation
        let mut best_dist = f32::INFINITY;
        for (_, feat) in candidates {
            let d = squared_distance(query_feat, feat);
            if d < best_dist {
                best_dist = d;
            }
        }

        // Keep all within 2× of nearest (tighter than before)
        let radius = best_dist * 4.0; // 2² since we compare squared distances
        for (font_id, feat) in candidates {
            let d = squared_distance(query_feat, feat);
            if d <= radius {
                let log_d = (d + 1e-10_f32).ln();
                font_log_dists.entry(*font_id).or_default().push(log_d);
            }
        }
    }


    // Aggregate: geometric mean of distances (same as char_index)
    let mut scores: Vec<(String, f32)> = font_log_dists
        .into_iter()
        .filter_map(|(font_id, log_dists)| {
            if log_dists.len() < quorum {
                return None;
            }
            let name = index.font_names.get(font_id)?.clone();
            let mean_log_dist = log_dists.iter().sum::<f32>() / log_dists.len() as f32;
            Some((name, -mean_log_dist))
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // σ cutoff
    if scores.len() >= 2 {
        let top_n = 50.min(scores.len());
        let vals: Vec<f32> = scores.iter().take(top_n).map(|(_, s)| *s).collect();
        let n = vals.len() as f32;
        let mean = vals.iter().sum::<f32>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let sigma = variance.sqrt();
        let best = vals[0];
        let cutoff = best - 0.5 * sigma;
        let before = scores.len();
        scores.retain(|(_, s)| *s >= cutoff);
    }

    scores
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Render a word in the given font at the target height.
fn render_word_for_index(font: &FontRef, text: &str, target_h: u32) -> Option<GrayImage> {
    let canvas = crate::layout::render_word_ab_glyph(
        font, text, target_h as f32,
        None, Some(target_h),
        |f, c| f.glyph_id(c),
    )?;
    Some(crate::ssim::trim_whitespace_simple(&canvas))
}

/// Squared Euclidean distance between two feature vectors.
fn squared_distance(a: &[f32; WORD_FEAT_LEN], b: &[f32; WORD_FEAT_LEN]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..WORD_FEAT_LEN {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}
