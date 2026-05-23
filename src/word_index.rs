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

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma};
use log::{debug, info};

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
    let trimmed = trim_whitespace_simple(img);
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

    info!("Building word index: {} words × {} fonts", words_set.len(), font_catalog.len());

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
    info!("Word index built: {} word keys, {} total entries", entries.len(), total);

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

    eprintln!(
        "  WI: {} queryable words, quorum={}, {} fonts in voting",
        crop_feats.len(), quorum, font_log_dists.len(),
    );

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
        eprintln!(
            "  WI sigma cutoff: best={:.3} σ={:.3} cutoff={:.3} → {} of {} kept",
            best, sigma, cutoff, scores.len(), before,
        );
    }

    scores
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Render a word in the given font at the target height.
fn render_word_for_index(font: &FontRef, text: &str, target_h: u32) -> Option<GrayImage> {
    if text.is_empty() {
        return None;
    }

    let em_px = target_h as f32;
    let scale = PxScale::from(em_px);
    let sf = font.as_scaled(scale);

    let ink_h = sf.ascent() - sf.descent();
    if ink_h <= 0.0 {
        return None;
    }

    let baseline = (target_h as f32 - ink_h) / 2.0 + sf.ascent();

    // Compute total advance width and min pixel extent
    let mut min_px_x = 0i32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    let mut cx = 0.0f32;

    for c in text.chars() {
        let gid = font.glyph_id(c);
        if let Some(p) = prev {
            cx += sf.kern(p, gid);
        }
        let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
        if let Some(og) = font.outline_glyph(glyph) {
            min_px_x = min_px_x.min(og.px_bounds().min.x as i32);
        }
        cx += sf.h_advance(gid);
        prev = Some(gid);
    }
    let total_advance = cx;

    let x_offset = if min_px_x < 0 { -min_px_x } else { 0 };
    let canvas_w = (total_advance as i32 + x_offset + 2).max(4) as u32;
    let mut canvas = GrayImage::from_pixel(canvas_w, target_h, Luma([255u8]));

    let mut cx = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    let (cw, ch) = canvas.dimensions();

    for c in text.chars() {
        let gid = font.glyph_id(c);
        if let Some(p) = prev {
            cx += sf.kern(p, gid);
        }
        let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
        if let Some(og) = font.outline_glyph(glyph) {
            let bounds = og.px_bounds();
            let bx = bounds.min.x as i32 + x_offset;
            let by = bounds.min.y as i32;
            og.draw(|gx, gy, cov| {
                let px = gx as i32 + bx;
                let py = gy as i32 + by;
                if px >= 0 && py >= 0 && (px as u32) < cw && (py as u32) < ch {
                    let val = (255.0 * (1.0 - cov)) as u8;
                    let cur = canvas.get_pixel(px as u32, py as u32).0[0];
                    canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
                }
            });
        }
        cx += sf.h_advance(gid);
        prev = Some(gid);
    }

    Some(trim_whitespace_simple(&canvas))
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

/// Simple whitespace trimming (all 4 edges, threshold 240).
fn trim_whitespace_simple(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }

    let thresh = 240u8;
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;

    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y).0[0] < thresh {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }

    if min_x > max_x || min_y > max_y {
        return img.clone();
    }

    image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        .to_image()
}
