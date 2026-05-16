//! Per-character font index and matcher.
//!
//! Instead of assembling a magic word, we build a feature index for **every
//! common character** rendered in each candidate font.  At match time we crop
//! individual characters from the longest OCR words in the scan and compare
//! their feature vectors against the index.
//!
//! ## Indexed characters
//! Printable ASCII (a-z, A-Z, 0-9, common punctuation) plus typographic
//! specials: em dash, en dash, smart quotes, ellipsis.
//!
//! ## Feature vector per character
//! - Column ink density profile (32 bins)
//! - Aspect ratio (ink width / ink height)
//! - Ink density (total ink / bbox area)
//! - Vertical centre of mass (0.0 = top, 1.0 = bottom)
//! - Horizontal balance (left-half ink / total ink)
//!
//! ## Extraction strategy
//! Pick the longest words in the line (≥3 chars), segment each into
//! per-character crops via column-ink valley detection, normalise to
//! `NORM_H`, return `(char, image)` pairs.
//!
//! ## Lookup
//! Uses a per-character k-d tree for O(D·log N) spatial lookup with
//! tolerance-band radius queries instead of brute-force cosine scan.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};
use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::ocr::CharBox;
use crate::verify::WordPlacement;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Normalised character height for all feature computation.
const NORM_H: u32 = 48;

/// Number of bins in the column ink density profile.
const PROFILE_BINS: usize = 32;

/// Number of horizontal crossing scan lines.
const CROSSING_BINS: usize = 8;

/// Number of terminal angle bins (up/right/down/left).
const TERMINAL_ANGLE_BINS: usize = 4;

/// Number of original scalar features (aspect..xh_cap_ratio).
const SCALAR_V1: usize = 7;

/// Number of new discriminative scalar features.
const SCALAR_V2: usize = 4 + TERMINAL_ANGLE_BINS + 2 + CROSSING_BINS;
// counter_area_ratio(1) + counter_centroid(2) + counter_aspect(1) = 4
// terminal_count is folded into terminal_angles normalization
// terminal_angles(4) + ink_perimeter(1) + compactness(1) = 6
// h_crossings(8)

/// Feature vector length: PROFILE_BINS + original scalars + new features.
pub const FEAT_LEN: usize = PROFILE_BINS + SCALAR_V1 + SCALAR_V2;

/// Minimum word length (characters) for extraction.
const MIN_WORD_LEN: usize = 3;

// ---------------------------------------------------------------------------
// Indexed character set
// ---------------------------------------------------------------------------

/// Returns the full set of characters we index.
pub fn indexed_chars() -> &'static [char] {
    static CHARS: std::sync::LazyLock<Vec<char>> = std::sync::LazyLock::new(|| {
        let mut v: Vec<char> = Vec::with_capacity(128);
        for c in 'a'..='z' { v.push(c); }
        for c in 'A'..='Z' { v.push(c); }
        for c in '0'..='9' { v.push(c); }
        for c in &[
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
            '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\',
            ']', '^', '_', '`', '{', '|', '}', '~',
        ] {
            v.push(*c);
        }
        v.push('\u{2014}'); // em dash
        v.push('\u{2013}'); // en dash
        v.push('\u{2018}'); // left single quote
        v.push('\u{2019}'); // right single quote
        v.push('\u{201C}'); // left double quote
        v.push('\u{201D}'); // right double quote
        v.push('\u{2026}'); // ellipsis
        v
    });
    &CHARS
}

/// Quick membership test.
fn is_indexed(c: char) -> bool {
    indexed_chars().contains(&c)
}

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/// Compact feature vector for one character rendering.
#[derive(Debug, Clone)]
pub struct CharFeatures {
    /// 32-bin column ink density profile (each 0.0–1.0).
    pub profile: [f32; PROFILE_BINS],
    /// Ink width / ink height.
    pub aspect: f32,
    /// Total ink pixels / bounding-box area.
    pub ink_density: f32,
    /// Vertical centre of mass (0.0 = top, 1.0 = bottom).
    pub v_center: f32,
    /// Left-half ink / total ink.
    pub h_balance: f32,
    /// Serif confidence score (0.0 = sans-serif, 1.0 = definite serif).
    pub serif_score: f32,
    /// Stroke contrast: ratio of thickest to thinnest strokes.
    pub stroke_contrast: f32,
    /// x-height to cap-height ratio (per-font metric).
    pub xh_cap_ratio: f32,

    // ── v2 discriminative features ──────────────────────────────────
    /// Counter (enclosed whitespace) area / total bbox area. 0 if no counter.
    pub counter_area_ratio: f32,
    /// Counter centroid X, normalised to [0,1] within ink bbox.
    pub counter_centroid_x: f32,
    /// Counter centroid Y, normalised to [0,1] within ink bbox.
    pub counter_centroid_y: f32,
    /// Counter aspect ratio (width/height of counter bbox). 0 if no counter.
    pub counter_aspect: f32,
    /// Terminal angle histogram: 4 bins (up/right/down/left), normalised.
    pub terminal_angles: [f32; TERMINAL_ANGLE_BINS],
    /// Ink perimeter / sqrt(ink_area). Normalised boundary complexity.
    pub ink_perimeter: f32,
    /// Compactness: 4π × area / perimeter². Circle=1.0, complex<1.0.
    pub compactness: f32,
    /// Horizontal crossings at 8 evenly-spaced scan lines. Each value is
    /// the number of ink↔white transitions, normalised by dividing by 20.
    pub h_crossings: [f32; CROSSING_BINS],
}

impl CharFeatures {
    /// Raw feature vector for serialisation (no normalisation).
    pub fn as_slice(&self) -> [f32; FEAT_LEN] {
        let mut v = [0.0f32; FEAT_LEN];
        let mut i = 0;
        // Column profile (32)
        v[i..i + PROFILE_BINS].copy_from_slice(&self.profile);
        i += PROFILE_BINS;
        // Original scalars (7)
        v[i] = self.aspect;           i += 1;
        v[i] = self.ink_density;      i += 1;
        v[i] = self.v_center;         i += 1;
        v[i] = self.h_balance;        i += 1;
        v[i] = self.serif_score;      i += 1;
        v[i] = self.stroke_contrast;  i += 1;
        v[i] = self.xh_cap_ratio;     i += 1;
        // v2: counter features (4)
        v[i] = self.counter_area_ratio;   i += 1;
        v[i] = self.counter_centroid_x;   i += 1;
        v[i] = self.counter_centroid_y;   i += 1;
        v[i] = self.counter_aspect;       i += 1;
        // v2: terminal angles (4)
        v[i..i + TERMINAL_ANGLE_BINS].copy_from_slice(&self.terminal_angles);
        i += TERMINAL_ANGLE_BINS;
        // v2: boundary features (2)
        v[i] = self.ink_perimeter;    i += 1;
        v[i] = self.compactness;      i += 1;
        // v2: horizontal crossings (8)
        v[i..i + CROSSING_BINS].copy_from_slice(&self.h_crossings);
        // i += CROSSING_BINS;  // last group
        debug_assert_eq!(i + CROSSING_BINS, FEAT_LEN);
        v
    }

    /// Weighted-normalised feature vector for matching.
    ///
    /// L2-normalises three groups independently then weights them:
    ///   - 32-bin column profile: weight 0.40
    ///   - 7 original scalars:    weight 0.30
    ///   - 18 new v2 features:    weight 0.30
    pub fn as_weighted_slice(&self) -> [f32; FEAT_LEN] {
        let mut v = [0.0f32; FEAT_LEN];

        // Group 1: column profile (32 bins) → weight 0.40
        let prof_mag: f32 = self.profile.iter().map(|x| x * x).sum::<f32>().sqrt();
        if prof_mag > 1e-9 {
            for i in 0..PROFILE_BINS {
                v[i] = self.profile[i] / prof_mag * 0.40;
            }
        }

        // Group 2: original scalars (7) → weight 0.30
        let scalars = [
            self.aspect,
            self.ink_density,
            self.v_center,
            self.h_balance,
            self.serif_score,
            self.stroke_contrast,
            self.xh_cap_ratio,
        ];
        let sc_mag: f32 = scalars.iter().map(|x| x * x).sum::<f32>().sqrt();
        if sc_mag > 1e-9 {
            for (i, &s) in scalars.iter().enumerate() {
                v[PROFILE_BINS + i] = s / sc_mag * 0.30;
            }
        }

        // Group 3: v2 features (4 counter + 4 terminal + 2 boundary + 8 crossings = 18) → weight 0.30
        let v2_start = PROFILE_BINS + SCALAR_V1;
        let mut v2 = [0.0f32; SCALAR_V2];
        let mut j = 0;
        v2[j] = self.counter_area_ratio;   j += 1;
        v2[j] = self.counter_centroid_x;   j += 1;
        v2[j] = self.counter_centroid_y;   j += 1;
        v2[j] = self.counter_aspect;       j += 1;
        for k in 0..TERMINAL_ANGLE_BINS {
            v2[j] = self.terminal_angles[k]; j += 1;
        }
        v2[j] = self.ink_perimeter;         j += 1;
        v2[j] = self.compactness;           j += 1;
        for k in 0..CROSSING_BINS {
            v2[j] = self.h_crossings[k];    j += 1;
        }

        let v2_mag: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v2_mag > 1e-9 {
            for (i, &val) in v2.iter().enumerate() {
                v[v2_start + i] = val / v2_mag * 0.30;
            }
        }

        v
    }
}

// ---------------------------------------------------------------------------
// Brute-force nearest-neighbor search
// ---------------------------------------------------------------------------
// At 59 dimensions the KD-tree degrades to near-linear scan anyway (the
// single-axis pruning test checks 1/59th of total distance — far branch is
// almost always explored). A flat vector with linear scan is simpler, faster
// (cache-friendly + LLVM auto-vectorizes the distance loop), and trivially
// correct.

/// Find the nearest neighbor, then return ALL points within `factor`× that
/// distance. Returns `(font_id, squared_distance)` pairs sorted by distance.
fn nearest_within_factor_brute(
    points: &[(usize, [f32; FEAT_LEN])],
    query: &[f32; FEAT_LEN],
    factor: f32,
) -> Vec<(usize, f32)> {
    if points.is_empty() {
        return Vec::new();
    }
    // Single pass: find min squared distance
    let mut best_dist_sq = f32::MAX;
    for (_, coords) in points {
        let d = squared_distance(coords, query);
        if d < best_dist_sq {
            best_dist_sq = d;
        }
    }
    let cutoff = factor * factor * best_dist_sq.max(1e-12);
    // Second pass: collect everything within cutoff
    let mut results: Vec<(usize, f32)> = points.iter()
        .filter_map(|(id, coords)| {
            let d = squared_distance(coords, query);
            if d <= cutoff { Some((*id, d)) } else { None }
        })
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Squared Euclidean distance between two feature vectors.
fn squared_distance(a: &[f32; FEAT_LEN], b: &[f32; FEAT_LEN]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..FEAT_LEN {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

// ---------------------------------------------------------------------------
// New typographic feature extraction
// ---------------------------------------------------------------------------

/// Detect whether a glyph image shows serifs.
fn detect_serif(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    if w < 3 || h < 8 {
        return 0.0;
    }

    let threshold = 200u8;
    let mut row_ink = vec![0u32; h as usize];
    let mut min_y = h;
    let mut max_y = 0u32;

    for y in 0..h {
        let mut count = 0u32;
        for x in 0..w {
            if img.get_pixel(x, y).0[0] < threshold {
                count += 1;
            }
        }
        row_ink[y as usize] = count;
        if count > 0 {
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }
    }

    let ink_h = max_y.saturating_sub(min_y) + 1;
    if ink_h < 8 {
        return 0.0;
    }

    let terminal_rows = (ink_h / 8).max(2).min(4);

    let top_ink: f32 = (min_y..min_y + terminal_rows)
        .map(|y| row_ink[y as usize] as f32)
        .sum::<f32>() / terminal_rows as f32;
    let bot_ink: f32 = ((max_y + 1 - terminal_rows)..=max_y)
        .map(|y| row_ink[y as usize] as f32)
        .sum::<f32>() / terminal_rows as f32;

    let mid_start = min_y + ink_h / 3;
    let mid_end = min_y + 2 * ink_h / 3;
    let mid_rows = (mid_end - mid_start).max(1);
    let mid_ink: f32 = (mid_start..mid_end)
        .map(|y| row_ink[y as usize] as f32)
        .sum::<f32>() / mid_rows as f32;

    if mid_ink < 1.0 {
        return 0.0;
    }

    let top_ratio = top_ink / mid_ink;
    let bot_ratio = bot_ink / mid_ink;
    let avg_ratio = (top_ratio + bot_ratio) / 2.0;

    ((avg_ratio - 1.0)).clamp(0.0, 1.0)
}

/// Compute a per-font serif confidence score.
pub fn compute_font_serif_score<F: Font>(font: &F) -> f32 {
    let diag_chars = ['I', 'l'];
    let mut scores = Vec::new();

    for &c in &diag_chars {
        if let Some(img) = render_char_normalised(font, c) {
            let s = detect_serif(&img);
            scores.push(s);
        }
    }

    if scores.is_empty() {
        return 0.0;
    }

    let sum: f32 = scores.iter().sum();
    (sum / scores.len() as f32).clamp(0.0, 1.0)
}

/// Measure stroke contrast (thick-to-thin ratio) from a glyph image.
fn measure_stroke_contrast(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    if w < 4 || h < 4 {
        return 1.0;
    }

    let threshold = 200u8;
    let mut all_runs: Vec<u32> = Vec::new();

    for y in 0..h {
        let mut run = 0u32;
        for x in 0..w {
            if img.get_pixel(x, y).0[0] < threshold {
                run += 1;
            } else {
                if run >= 2 { all_runs.push(run); }
                run = 0;
            }
        }
        if run >= 2 { all_runs.push(run); }
    }

    for x in 0..w {
        let mut run = 0u32;
        for y in 0..h {
            if img.get_pixel(x, y).0[0] < threshold {
                run += 1;
            } else {
                if run >= 2 { all_runs.push(run); }
                run = 0;
            }
        }
        if run >= 2 { all_runs.push(run); }
    }

    if all_runs.len() < 4 {
        return 1.0;
    }

    all_runs.sort_unstable();
    let p10 = all_runs[all_runs.len() / 10].max(1);
    let p90 = all_runs[all_runs.len() * 9 / 10].max(1);

    p90 as f32 / p10 as f32
}

/// Compute x-height / cap-height ratio for a font.
pub fn compute_xh_cap_ratio<F: Font>(font: &F) -> f32 {
    let scale = PxScale::from(200.0);

    let ink_height_at_scale = |c: char| -> Option<f32> {
        let gid = font.glyph_id(c);
        if gid.0 == 0 { return None; }
        let sf = font.as_scaled(scale);
        let glyph = gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
        let outlined = font.outline_glyph(glyph)?;
        let b = outlined.px_bounds();
        let ih = b.max.y - b.min.y;
        if ih > 0.5 { Some(ih) } else { None }
    };

    match (ink_height_at_scale('x'), ink_height_at_scale('H')) {
        (Some(xh), Some(ch)) if ch > 0.0 => (xh / ch).clamp(0.0, 1.0),
        _ => 0.65,
    }
}

/// Character discriminativeness weight for scoring.
fn char_weight(c: char) -> f32 {
    match c {
        'g' | 'a' | 'e' | 'R' | 'Q' | 'G' | 'S' | 'f' | 't' | 'y' | '&' | '@' => 1.5,
        'I' | 'l' | '1' | '|' | '!' | '.' | ',' | ':' | ';' | '-' => 0.5,
        'b' | 'd' | 'p' | 'q' | 'n' | 'u' | 'o' | 'c' | 'O' | 'C' | 'D' => 0.8,
        'k' | 'w' | 'x' | 'z' | 'A' | 'B' | 'E' | 'F' | 'K' | 'M' | 'N' | 'W' => 1.2,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Feature computation
// ---------------------------------------------------------------------------

/// Compute features from a normalised grayscale glyph image.
pub fn compute_features(img: &GrayImage) -> Option<CharFeatures> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }

    let threshold = 200u8;
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;
    let mut total_ink = 0u64;
    let mut ink_pixels = 0u64;
    let mut wy_sum = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y).0[0];
            if px < threshold {
                let ink_val = (255 - px) as u64;
                total_ink += ink_val;
                ink_pixels += 1;
                wy_sum += y as f64 * ink_val as f64;
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }
    }

    if ink_pixels == 0 || total_ink == 0 {
        return None;
    }

    let ink_w = (max_x - min_x + 1) as f32;
    let ink_h = (max_y - min_y + 1) as f32;
    let aspect = ink_w / ink_h.max(1.0);
    let bbox_area = ink_w * ink_h;
    let ink_density = ink_pixels as f32 / bbox_area.max(1.0);
    let v_center = (wy_sum / total_ink as f64) as f32 / h as f32;

    let ink_mid_x = (min_x + max_x) / 2;
    let mut left_ink = 0u64;
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y).0[0];
            if px < threshold {
                let ink_val = (255 - px) as u64;
                if x <= ink_mid_x {
                    left_ink += ink_val;
                }
            }
        }
    }
    let h_balance = left_ink as f32 / total_ink as f32;

    let mut col_ink = vec![0.0f32; ink_w as usize];
    for x in min_x..=max_x {
        let mut col_sum = 0.0f32;
        for y in min_y..=max_y {
            let px = img.get_pixel(x, y).0[0];
            if px < threshold {
                col_sum += (255 - px) as f32;
            }
        }
        col_ink[(x - min_x) as usize] = col_sum;
    }
    let col_max = col_ink.iter().cloned().fold(0.0f32, f32::max);
    if col_max > 0.0 {
        for v in &mut col_ink {
            *v /= col_max;
        }
    }
    let profile = resample(&col_ink, PROFILE_BINS);

    let serif_score = detect_serif(img);
    let stroke_contrast_val = measure_stroke_contrast(img);

    // ── v2 features ────────────────────────────────────────────────

    // Build a binary ink mask within ink bbox for reuse
    let ink_w_u = (max_x - min_x + 1) as usize;
    let ink_h_u = (max_y - min_y + 1) as usize;
    let mut ink_mask = vec![false; ink_w_u * ink_h_u];
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if img.get_pixel(x, y).0[0] < threshold {
                ink_mask[(y - min_y) as usize * ink_w_u + (x - min_x) as usize] = true;
            }
        }
    }

    // Counter shape analysis: flood-fill from edges, remaining white = counter
    let (counter_area_ratio, counter_centroid_x, counter_centroid_y, counter_aspect) =
        compute_counter_features(&ink_mask, ink_w_u, ink_h_u);

    // Terminal / endpoint analysis
    let terminal_angles = compute_terminal_angles(&ink_mask, ink_w_u, ink_h_u);

    // Boundary complexity
    let (ink_perimeter, compactness) =
        compute_boundary_features(&ink_mask, ink_w_u, ink_h_u, ink_pixels as f32);

    // Horizontal crossings
    let h_crossings = compute_h_crossings(&ink_mask, ink_w_u, ink_h_u);

    Some(CharFeatures {
        profile,
        aspect,
        ink_density,
        v_center,
        h_balance,
        serif_score,
        stroke_contrast: stroke_contrast_val,
        xh_cap_ratio: 0.0,
        counter_area_ratio,
        counter_centroid_x,
        counter_centroid_y,
        counter_aspect,
        terminal_angles,
        ink_perimeter,
        compactness,
        h_crossings,
    })
}

// ---------------------------------------------------------------------------
// v2 discriminative feature helpers
// ---------------------------------------------------------------------------

/// Counter shape analysis via edge flood-fill.
///
/// Flood-fills white pixels from edges of the ink bounding box. Any white pixels
/// NOT reached are enclosed counters (e.g. the hole in 'o', 'a', 'e').
fn compute_counter_features(ink_mask: &[bool], w: usize, h: usize) -> (f32, f32, f32, f32) {
    if w == 0 || h == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let total = w * h;
    let mut reachable = vec![false; total];
    let mut queue = std::collections::VecDeque::new();

    // Seed from all edge pixels that are NOT ink
    for x in 0..w {
        if !ink_mask[x] && !reachable[x] {
            reachable[x] = true;
            queue.push_back((x, 0usize));
        }
        let idx = (h - 1) * w + x;
        if !ink_mask[idx] && !reachable[idx] {
            reachable[idx] = true;
            queue.push_back((x, h - 1));
        }
    }
    for y in 0..h {
        let idx = y * w;
        if !ink_mask[idx] && !reachable[idx] {
            reachable[idx] = true;
            queue.push_back((0, y));
        }
        let idx = y * w + (w - 1);
        if !ink_mask[idx] && !reachable[idx] {
            reachable[idx] = true;
            queue.push_back((w - 1, y));
        }
    }

    // BFS flood fill
    while let Some((x, y)) = queue.pop_front() {
        let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in &neighbors {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let nidx = ny as usize * w + nx as usize;
            if !ink_mask[nidx] && !reachable[nidx] {
                reachable[nidx] = true;
                queue.push_back((nx as usize, ny as usize));
            }
        }
    }

    // Counter pixels: white (not ink) AND not reachable from edges
    let mut counter_area = 0u32;
    let mut cx_sum = 0.0f64;
    let mut cy_sum = 0.0f64;
    let mut counter_min_x = w;
    let mut counter_max_x = 0usize;
    let mut counter_min_y = h;
    let mut counter_max_y = 0usize;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !ink_mask[idx] && !reachable[idx] {
                counter_area += 1;
                cx_sum += x as f64;
                cy_sum += y as f64;
                if x < counter_min_x { counter_min_x = x; }
                if x > counter_max_x { counter_max_x = x; }
                if y < counter_min_y { counter_min_y = y; }
                if y > counter_max_y { counter_max_y = y; }
            }
        }
    }

    if counter_area == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let area_ratio = counter_area as f32 / total as f32;
    let centroid_x = (cx_sum / counter_area as f64) as f32 / w.max(1) as f32;
    let centroid_y = (cy_sum / counter_area as f64) as f32 / h.max(1) as f32;
    let cw = (counter_max_x - counter_min_x + 1) as f32;
    let ch = (counter_max_y - counter_min_y + 1) as f32;
    let c_aspect = cw / ch.max(1.0);

    (area_ratio, centroid_x, centroid_y, c_aspect)
}

/// Terminal / endpoint analysis.
///
/// A terminal pixel has exactly 1 ink neighbor in 8-connected. We classify
/// the direction based on where the single neighbor is relative to the terminal.
/// Returns a 4-bin normalised histogram: [up, right, down, left].
fn compute_terminal_angles(ink_mask: &[bool], w: usize, h: usize) -> [f32; TERMINAL_ANGLE_BINS] {
    let mut bins = [0.0f32; TERMINAL_ANGLE_BINS]; // up, right, down, left
    if w < 3 || h < 3 {
        return bins;
    }

    let deltas: [(i32, i32); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];

    let mut total_terminals = 0u32;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !ink_mask[idx] { continue; }

            // Count ink neighbors and find the single neighbor if count==1
            let mut neighbor_count = 0u32;
            let mut nb_dx = 0i32;
            let mut nb_dy = 0i32;

            for &(dx, dy) in &deltas {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    if ink_mask[ny as usize * w + nx as usize] {
                        neighbor_count += 1;
                        nb_dx = dx;
                        nb_dy = dy;
                    }
                }
            }

            if neighbor_count == 1 {
                total_terminals += 1;
                // The terminal points AWAY from its single neighbor
                // If neighbor is below (dy=1), terminal points up → bin 0
                // If neighbor is left (dx=-1), terminal points right → bin 1
                // If neighbor is above (dy=-1), terminal points down → bin 2
                // If neighbor is right (dx=1), terminal points left → bin 3
                let angle = (-nb_dy as f64).atan2(-nb_dx as f64); // direction away
                let deg = angle.to_degrees();
                // -180..180 → bin: up=[-135,-45), right=[-45,45), down=[45,135), left=else
                let bin = if deg >= -135.0 && deg < -45.0 {
                    0 // up
                } else if deg >= -45.0 && deg < 45.0 {
                    1 // right
                } else if deg >= 45.0 && deg < 135.0 {
                    2 // down
                } else {
                    3 // left
                };
                bins[bin] += 1.0;
            }
        }
    }

    // Normalise by total terminals (so histogram sums to 1.0)
    if total_terminals > 0 {
        for b in &mut bins {
            *b /= total_terminals as f32;
        }
    }

    bins
}

/// Boundary complexity features.
///
/// Returns (normalised_perimeter, compactness).
/// - normalised_perimeter = boundary_pixel_count / sqrt(ink_area)
///   Dividing by sqrt(area) makes it scale-independent.
/// - compactness = 4π × area / perimeter² (1.0 = circle).
fn compute_boundary_features(ink_mask: &[bool], w: usize, h: usize, ink_count: f32) -> (f32, f32) {
    if w == 0 || h == 0 || ink_count < 1.0 {
        return (0.0, 0.0);
    }

    let mut boundary_count = 0u32;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !ink_mask[idx] { continue; }

            // Ink pixel borders white if any 4-connected neighbor is white or out-of-bounds
            let is_boundary = (x == 0 || !ink_mask[idx - 1])
                || (x + 1 >= w || !ink_mask[idx + 1])
                || (y == 0 || !ink_mask[idx - w])
                || (y + 1 >= h || !ink_mask[idx + w]);

            if is_boundary {
                boundary_count += 1;
            }
        }
    }

    let perim = boundary_count as f32;
    let norm_perimeter = perim / ink_count.sqrt();
    let compactness = if perim > 0.0 {
        (4.0 * std::f32::consts::PI * ink_count) / (perim * perim)
    } else {
        0.0
    };

    // Cap normalised perimeter to a reasonable range for feature vector
    let norm_perimeter_capped = (norm_perimeter / 10.0).min(1.0);

    (norm_perimeter_capped, compactness)
}

/// Horizontal crossings at 8 evenly-spaced scan lines.
///
/// For each scan line, counts the number of ink→white or white→ink transitions.
/// Normalised by dividing by 20 (a reasonable cap for complex chars).
fn compute_h_crossings(ink_mask: &[bool], w: usize, h: usize) -> [f32; CROSSING_BINS] {
    let mut crossings = [0.0f32; CROSSING_BINS];
    if w == 0 || h == 0 {
        return crossings;
    }

    for bin in 0..CROSSING_BINS {
        // Evenly spaced: first at 1/(N+1), last at N/(N+1) of height
        let y = ((bin + 1) as f32 * h as f32 / (CROSSING_BINS + 1) as f32) as usize;
        let y = y.min(h - 1);

        let mut transitions = 0u32;
        let mut prev_ink = false;
        for x in 0..w {
            let is_ink = ink_mask[y * w + x];
            if is_ink != prev_ink && x > 0 {
                transitions += 1;
            }
            prev_ink = is_ink;
        }
        // Normalise (most chars have ≤10 transitions; 20 is generous cap)
        crossings[bin] = (transitions as f32 / 20.0).min(1.0);
    }

    crossings
}

/// Linearly resample a 1-D signal to `n` bins.
fn resample(src: &[f32], n: usize) -> [f32; PROFILE_BINS] {
    let mut out = [0.0f32; PROFILE_BINS];
    if src.is_empty() || n == 0 {
        return out;
    }
    let src_len = src.len() as f32;
    for i in 0..n {
        let pos = i as f32 * (src_len - 1.0) / (n as f32 - 1.0).max(1.0);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(src.len() - 1);
        let frac = pos - lo as f32;
        out[i] = src[lo] * (1.0 - frac) + src[hi] * frac;
    }
    out
}

// ---------------------------------------------------------------------------
// Character index
// ---------------------------------------------------------------------------

/// Per-font feature for a single character.
#[derive(Debug, Clone)]
pub struct FontCharEntry {
    pub font_name: String,
    pub features: CharFeatures,
}

/// The full index: for each indexed character, a list of (font, features)
/// plus flat per-character vectors for brute-force spatial lookup.
#[derive(Debug)]
pub struct CharIndex {
    /// char → Vec<(font_name, features)> — raw entries for serialization + merge
    pub entries: HashMap<char, Vec<FontCharEntry>>,
    /// Fonts that were scanned but produced no indexable characters.
    pub skipped_fonts: std::collections::HashSet<String>,
    /// Ordered font name table: font_id → font_name
    font_names_table: Vec<String>,
    /// Per-character flat vectors: (font_id, weighted_features) for brute-force search
    flat_vecs: HashMap<char, Vec<(usize, [f32; FEAT_LEN])>>,
    /// Per-character per-dimension standard deviations (for diagnostics)
    dim_sigmas: HashMap<char, [f32; FEAT_LEN]>,
}

impl Clone for CharIndex {
    fn clone(&self) -> Self {
        let mut idx = CharIndex {
            entries: self.entries.clone(),
            skipped_fonts: self.skipped_fonts.clone(),
            font_names_table: Vec::new(),
            flat_vecs: HashMap::new(),
            dim_sigmas: HashMap::new(),
        };
        idx.rebuild_vecs();
        idx
    }
}

impl CharIndex {
    /// Build flat per-character vectors and compute per-dimension σ from entries.
    pub fn rebuild_vecs(&mut self) {
        // Build font name → id mapping
        let mut name_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entries in self.entries.values() {
            for e in entries {
                name_set.insert(e.font_name.clone());
            }
        }
        let mut names: Vec<String> = name_set.into_iter().collect();
        names.sort();
        self.font_names_table = names;
        let name_to_id: HashMap<&str, usize> = self.font_names_table.iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        self.flat_vecs.clear();
        self.dim_sigmas.clear();

        for (c, char_entries) in &self.entries {
            if char_entries.is_empty() {
                continue;
            }

            // Build flat vec of (font_id, weighted_features)
            let mut points: Vec<(usize, [f32; FEAT_LEN])> = Vec::with_capacity(char_entries.len());
            let mut all_weighted: Vec<[f32; FEAT_LEN]> = Vec::with_capacity(char_entries.len());

            for e in char_entries {
                let weighted = e.features.as_weighted_slice();
                if let Some(&font_id) = name_to_id.get(e.font_name.as_str()) {
                    points.push((font_id, weighted));
                    all_weighted.push(weighted);
                }
            }

            if points.is_empty() {
                continue;
            }

            // Compute per-dimension mean and σ
            let n = all_weighted.len() as f32;
            let mut means = [0.0f32; FEAT_LEN];
            for w in &all_weighted {
                for d in 0..FEAT_LEN {
                    means[d] += w[d];
                }
            }
            for d in 0..FEAT_LEN {
                means[d] /= n;
            }

            let mut sigmas = [0.0f32; FEAT_LEN];
            for w in &all_weighted {
                for d in 0..FEAT_LEN {
                    let diff = w[d] - means[d];
                    sigmas[d] += diff * diff;
                }
            }
            for d in 0..FEAT_LEN {
                sigmas[d] = (sigmas[d] / n).sqrt();
            }

            self.dim_sigmas.insert(*c, sigmas);
            self.flat_vecs.insert(*c, points);
        }
    }
}

/// Build the per-character index for all provided fonts.
pub fn build_char_index(font_paths: &[(String, Vec<u8>)]) -> CharIndex {
    use rayon::prelude::*;

    let chars = indexed_chars();

    // Per-font result: Vec of (char, entry) pairs, plus font_name if skipped.
    struct FontResult {
        char_entries: Vec<(char, FontCharEntry)>,
        skipped: Option<String>,
    }

    // Phase 1: process each font in parallel (CPU-bound: render + features)
    let results: Vec<FontResult> = font_paths.par_iter().map(|(font_name, font_data)| {
        let font = match FontRef::try_from_slice(font_data) {
            Ok(f) => f,
            Err(_) => {
                return FontResult {
                    char_entries: Vec::new(),
                    skipped: Some(font_name.clone()),
                };
            }
        };

        let xh_ratio = compute_xh_cap_ratio(&font);
        let serif = compute_font_serif_score(&font);

        let mut char_entries = Vec::with_capacity(chars.len());
        for &c in chars {
            if let Some(img) = render_char_normalised(&font, c) {
                if let Some(mut feats) = compute_features(&img) {
                    feats.xh_cap_ratio = xh_ratio;
                    feats.serif_score = serif;
                    char_entries.push((c, FontCharEntry {
                        font_name: font_name.clone(),
                        features: feats,
                    }));
                }
            }
        }

        let skipped = if char_entries.is_empty() { Some(font_name.clone()) } else { None };
        FontResult { char_entries, skipped }
    }).collect();

    // Phase 2: merge results (sequential, fast — just moving Vecs)
    let mut entries: HashMap<char, Vec<FontCharEntry>> = HashMap::new();
    for c in chars {
        entries.insert(*c, Vec::with_capacity(font_paths.len()));
    }
    let mut skipped_fonts: std::collections::HashSet<String> = std::collections::HashSet::new();

    for result in results {
        if let Some(name) = result.skipped {
            skipped_fonts.insert(name);
        }
        for (c, entry) in result.char_entries {
            entries.entry(c).or_default().push(entry);
        }
    }

    let mut index = CharIndex {
        entries,
        skipped_fonts,
        font_names_table: Vec::new(),
        flat_vecs: HashMap::new(),
        dim_sigmas: HashMap::new(),
    };
    index.rebuild_vecs();
    index
}

/// Render a single character in `font` at `NORM_H` ink height.
pub fn render_char_normalised<F: Font>(font: &F, c: char) -> Option<GrayImage> {
    let gid = font.glyph_id(c);
    if gid.0 == 0 {
        return None;
    }

    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);

    let glyph = gid.with_scale_and_position(ref_scale, point(0.0, sf_ref.ascent()));
    let outlined = font.outline_glyph(glyph)?;
    let bounds = outlined.px_bounds();
    let ink_h_ref = bounds.max.y - bounds.min.y;
    if ink_h_ref < 1.0 {
        return None;
    }

    let target_scale = ref_h * (NORM_H as f32 / ink_h_ref);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);

    let glyph2 = gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
    let outlined2 = font.outline_glyph(glyph2)?;
    let b2 = outlined2.px_bounds();

    let img_w = (b2.max.x - b2.min.x).ceil() as u32 + 2;
    let img_h = (b2.max.y - b2.min.y).ceil() as u32 + 2;
    if img_w == 0 || img_h == 0 || img_w > 500 || img_h > 500 {
        return None;
    }

    let mut canvas = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));
    let ox = b2.min.x.floor() as i32;
    let oy = b2.min.y.floor() as i32;

    outlined2.draw(|gx, gy, cov| {
        let px = gx as i32 + (b2.min.x.floor() as i32) - ox + 1;
        let py = gy as i32 + (b2.min.y.floor() as i32) - oy + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
        }
    });

    Some(canvas)
}

// ---------------------------------------------------------------------------
// Character extraction from scan
// ---------------------------------------------------------------------------

/// Extract individual character crops from a scan line.
pub fn extract_line_chars(
    page: &GrayImage,
    words: &[WordPlacement],
    line_height: u32,
    page_char_boxes: &[CharBox],
) -> Vec<(char, GrayImage)> {
    if words.is_empty() || line_height == 0 {
        return Vec::new();
    }

    // If we have Tesseract makebox char boxes, use them directly
    // instead of the column-valley segmenter.
    if !page_char_boxes.is_empty() {
        return extract_line_chars_from_charboxes(page, words, line_height, page_char_boxes);
    }

    let mut sorted: Vec<&WordPlacement> = words
        .iter()
        .filter(|w| w.text.chars().count() >= MIN_WORD_LEN && w.width > 0)
        .collect();
    sorted.sort_by(|a, b| b.text.chars().count().cmp(&a.text.chars().count()));

    let mut char_counts: HashMap<char, usize> = HashMap::new();
    let mut results: Vec<(char, GrayImage)> = Vec::new();

    for word in &sorted {
        let chars_in_word: Vec<char> = word.text.chars().filter(|c| is_indexed(*c)).collect();
        if chars_in_word.is_empty() {
            continue;
        }

        let need_any = chars_in_word.iter().any(|c| {
            char_counts.get(c).copied().unwrap_or(0) < 2
        });
        if !need_any {
            continue;
        }

        let wx = word.x_off;
        let mut wy = word.y_off;
        let ww = word.width;
        let mut wh = word.height.max(line_height);

        let (pw, ph) = page.dimensions();
        if wx >= pw || wy >= ph {
            continue;
        }

        // Ink-expand: search ±15 px above/below for actual ink
        let margin: u32 = 15;
        let search_top = wy.saturating_sub(margin);
        let search_bot = (wy + wh + margin).min(ph);
        let clamped_w = ww.min(pw - wx);
        let (ink_top, ink_bot) = crate::ocr::ink_vertical_extent(
            page, wx, clamped_w, search_top, search_bot, 200,
        );
        let new_y = ink_top.min(wy);
        let new_bot = ink_bot.max(wy + wh);
        wy = new_y;
        wh = new_bot.saturating_sub(new_y);

        let crop_w = ww.min(pw - wx);
        let crop_h = wh.min(ph - wy);
        if crop_w < 2 || crop_h < 2 {
            continue;
        }

        let word_img = image::imageops::crop_imm(page, wx, wy, crop_w, crop_h).to_image();

        let boundaries = segment_characters(&word_img, chars_in_word.len());
        let all_chars: Vec<char> = word.text.chars().collect();

        if boundaries.len() != all_chars.len() + 1 {
            let uniform = uniform_boundaries(crop_w, all_chars.len());
            extract_chars_from_boundaries(
                &word_img, &all_chars, &uniform, crop_h,
                &mut char_counts, &mut results,
            );
        } else {
            extract_chars_from_boundaries(
                &word_img, &all_chars, &boundaries, crop_h,
                &mut char_counts, &mut results,
            );
        }
    }

    results
}

/// Given character boundaries, crop and normalise each character.
/// Extract character crops from Tesseract makebox char-level bounding boxes.
/// Filters the page-level char boxes to those overlapping the line's words,
/// then crops and scales each character directly from the page.
fn extract_line_chars_from_charboxes(
    page: &GrayImage,
    words: &[WordPlacement],
    line_height: u32,
    page_char_boxes: &[CharBox],
) -> Vec<(char, GrayImage)> {
    let (pw, ph) = page.dimensions();
    let mut char_counts: HashMap<char, usize> = HashMap::new();
    let mut results: Vec<(char, GrayImage)> = Vec::new();

    // Compute the line's bounding box from its words
    let line_y_min = words.iter().map(|w| w.y_off).min().unwrap_or(0);
    let line_y_max = words.iter().map(|w| w.y_off + w.height.max(line_height)).max().unwrap_or(0);
    let line_x_min = words.iter().map(|w| w.x_off).min().unwrap_or(0);
    let line_x_max = words.iter().map(|w| w.x_off + w.width).max().unwrap_or(0);

    // Match char boxes against individual word bounding boxes rather than the
    // whole line bbox. This prevents cross-line contamination where chars from
    // adjacent lines (e.g. "Font:" reference line) leak into body text lines.
    // A charbox must have its center within a word bbox (with small vertical tolerance).
    let v_tol = line_height / 4;  // small vertical tolerance for alignment jitter
    let line_chars: Vec<&CharBox> = page_char_boxes
        .iter()
        .filter(|cb| {
            if cb.width < 2 || cb.height < 2 {
                return false;
            }
            // Match charbox center to word bounding boxes.
            // With HOCR, charboxes are structurally nested in words so
            // image-area contamination is already eliminated.
            let cb_cx = cb.x + cb.width / 2;
            let cb_cy = cb.y + cb.height / 2;
            words.iter().any(|w| {
                if w.confidence < 10.0 {
                    return false;
                }
                let w_top = w.y_off.saturating_sub(v_tol);
                let w_bot = w.y_off + w.height.max(line_height) + v_tol;
                let w_left = w.x_off;
                let w_right = w.x_off + w.width;
                cb_cx >= w_left && cb_cx <= w_right
                    && cb_cy >= w_top && cb_cy <= w_bot
            })
        })
        .collect();

    for cb in &line_chars {
        let c = cb.ch;
        // Filter on per-character OCR confidence (from HOCR x_conf).
        // Low-confidence chars are likely image fragments or misdetections.
        if cb.confidence < 75.0 {
            continue;
        }
        if !is_indexed(c) {
            continue;
        }
        if char_counts.get(&c).copied().unwrap_or(0) >= 3 {
            continue;
        }

        // Clamp to page bounds
        let cx = cb.x.min(pw.saturating_sub(1));
        let cy = cb.y.min(ph.saturating_sub(1));
        let cw = cb.width.min(pw - cx);
        let ch_px = cb.height.min(ph - cy);
        if cw < 2 || ch_px < 2 {
            continue;
        }

        // Skip impossibly narrow crops (aspect ratio filter)
        let aspect = cw as f32 / ch_px as f32;
        if aspect < 0.15 {
            continue;
        }

        let char_crop = image::imageops::crop_imm(page, cx, cy, cw, ch_px).to_image();

        // Scale to NORM_H, preserving aspect ratio
        let scaled_w = ((cw as f32 * NORM_H as f32 / ch_px as f32).ceil() as u32).max(1);
        let scaled = image::imageops::resize(
            &char_crop,
            scaled_w,
            NORM_H,
            image::imageops::FilterType::Lanczos3,
        );

        // Crop quality gate: reject image fragments and non-text crops.
        // Real text characters have near-black ink (min ≈ 0) on near-white
        // background (max ≈ 255) with high pixel variance (std > 80).
        // Image fragments lack this contrast — their min pixel is well
        // above 0, their max is well below 255, or they have low variance.
        {
            let mut pmin = 255u8;
            let mut pmax = 0u8;
            let n = scaled.pixels().len() as f64;
            let mut sum = 0f64;
            let mut sum_sq = 0f64;
            for p in scaled.pixels() {
                let v = p.0[0];
                if v < pmin { pmin = v; }
                if v > pmax { pmax = v; }
                let vf = v as f64;
                sum += vf;
                sum_sq += vf * vf;
            }
            let variance = (sum_sq / n) - (sum / n).powi(2);
            let std_dev = variance.max(0.0).sqrt();
            // Reject if no real ink, no real background, or too little contrast
            if pmin > 20 || pmax < 235 || std_dev < 75.0 {
                continue;
            }
        }

        results.push((c, scaled));
        *char_counts.entry(c).or_insert(0) += 1;
    }

    results
}

fn extract_chars_from_boundaries(
    word_img: &GrayImage,
    chars: &[char],
    boundaries: &[u32],
    crop_h: u32,
    char_counts: &mut HashMap<char, usize>,
    results: &mut Vec<(char, GrayImage)>,
) {
    let (ww, _wh) = word_img.dimensions();

    for (i, &c) in chars.iter().enumerate() {
        if !is_indexed(c) {
            continue;
        }
        if char_counts.get(&c).copied().unwrap_or(0) >= 3 {
            continue;
        }

        if i + 1 >= boundaries.len() {
            break;
        }
        let x0 = boundaries[i].min(ww);
        let x1 = boundaries[i + 1].min(ww);
        if x1 <= x0 || (x1 - x0) < 2 {
            continue;
        }

        let char_crop = image::imageops::crop_imm(word_img, x0, 0, x1 - x0, crop_h).to_image();

        let scaled = image::imageops::resize(
            &char_crop,
            ((x1 - x0) as f32 * NORM_H as f32 / crop_h as f32).ceil() as u32,
            NORM_H,
            image::imageops::FilterType::Lanczos3,
        );

        results.push((c, scaled));
        *char_counts.entry(c).or_insert(0) += 1;
    }
}

/// Segment a word image into N characters using column ink valleys.
fn segment_characters(img: &GrayImage, n_chars: usize) -> Vec<u32> {
    let (w, h) = img.dimensions();
    if n_chars <= 1 {
        return vec![0, w];
    }

    let threshold = 200u8;
    let mut col_ink = vec![0.0f32; w as usize];
    for x in 0..w {
        let mut s = 0.0f32;
        for y in 0..h {
            let px = img.get_pixel(x, y).0[0];
            if px < threshold {
                s += (255 - px) as f32;
            }
        }
        col_ink[x as usize] = s;
    }

    let smoothed = smooth_signal(&col_ink, 3);

    let mut valleys: Vec<(u32, f32)> = Vec::new();
    for i in 1..smoothed.len().saturating_sub(1) {
        if smoothed[i] <= smoothed[i - 1] && smoothed[i] <= smoothed[i + 1] {
            valleys.push((i as u32, smoothed[i]));
        }
    }

    valleys.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let need = n_chars - 1;

    if valleys.len() >= need {
        let mut splits: Vec<u32> = valleys[..need].iter().map(|v| v.0).collect();
        splits.sort();
        let mut bounds = Vec::with_capacity(n_chars + 1);
        bounds.push(0);
        bounds.extend_from_slice(&splits);
        bounds.push(w);
        bounds
    } else {
        uniform_boundaries(w, n_chars)
    }
}

/// Uniform character boundaries.
fn uniform_boundaries(width: u32, n: usize) -> Vec<u32> {
    let mut b = Vec::with_capacity(n + 1);
    for i in 0..=n {
        b.push((i as f32 * width as f32 / n as f32).round() as u32);
    }
    b
}

/// Simple box-filter smoothing.
fn smooth_signal(src: &[f32], radius: usize) -> Vec<f32> {
    let n = src.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        let sum: f32 = src[lo..hi].iter().sum();
        out[i] = sum / (hi - lo) as f32;
    }
    out
}

// ---------------------------------------------------------------------------
// Matching — k-d tree based
// ---------------------------------------------------------------------------

/// Search result from the k-d tree for a single character.
/// Contains font candidates within the tolerance radius.
#[derive(Debug, Clone)]
pub struct CharSearchResult {
    /// The character queried
    pub ch: char,
    /// Candidate fonts within tolerance: (font_name, cosine_similarity)
    pub candidates: Vec<(String, f32)>,
    /// The search radius used
    pub radius: f32,
    /// Number of fonts within the radius
    pub n_within_radius: usize,
}

/// Search the index for candidate fonts matching the given character crops.
///
/// This is the single shared search function used by both the pipeline and tests.
///
/// # Scoring: geometric mean of k-d tree distances
///
/// Each character's k-d tree gives us a squared Euclidean distance d²ᵢ for
/// font F in the 57-dimensional weighted feature space.  Under the null
/// hypothesis (random font), these distances follow a scaled χ² distribution
/// — fonts that are genuinely similar will have consistently small distances
/// across multiple independent character observations.
///
/// We aggregate via the **geometric mean of distances**:
///
/// ```text
///     score(F) = exp( (1/N) · Σᵢ log(d²ᵢ + ε) )
/// ```
///
/// This is equivalent to multiplying per-character p-values (Fisher's method)
/// in log space.  It has two key properties:
///
/// 1. **Multiplicative sensitivity:** a font that's close on 14 of 15 chars
///    but far on 1 gets a much worse score than a font that's moderately close
///    on all 15.  One bad character tanks the geometric mean — this naturally
///    penalizes the "619 fonts within 2× on 'e'" problem because those fonts
///    will have large distances on other characters.
///
/// 2. **No arbitrary union:** we don't union per-character candidate sets.
///    Instead, we collect (font_id, d²) pairs per character, then score every
///    font that appears in ≥ quorum characters.  Fonts appearing in only 1-2
///    character neighborhoods are discarded — they're statistical noise.
///
/// The quorum threshold (currently ≥ 50% of queried characters) acts as a
/// lightweight pre-filter before the geometric mean calculation.  A font must
/// be a plausible match on at least half the characters to be considered.
///
/// Returns top `top_n` fonts by geometric-mean distance (ascending = best).
/// The returned f32 score is negated log-geometric-mean so that higher = better,
/// matching the convention expected by downstream callers.
pub fn search_candidates(
    index: &CharIndex,
    char_crops: &[(char, GrayImage)],
    thoroughness: f32,
) -> Vec<(String, f32)> {
    if char_crops.is_empty() {
        return Vec::new();
    }

    // Pre-compute weighted feature vectors for all crops
    let crop_feats: Vec<(char, [f32; FEAT_LEN])> = char_crops
        .iter()
        .filter_map(|(c, img)| {
            compute_features(img).map(|f| (*c, f.as_weighted_slice()))
        })
        .collect();

    if crop_feats.is_empty() {
        return Vec::new();
    }

    let n_chars = crop_feats.len();
    // Quorum: font must appear in at least half the character neighborhoods.
    let quorum = ((n_chars + 1) / 2).max(1) as f32 / thoroughness;
    let quorum = (quorum.ceil() as usize).max(1);


    // For each character, find nearby fonts and record their distances.
    // font_id → Vec<(log_dist, weight)>
    let mut font_log_dists: HashMap<usize, Vec<(f32, f32)>> = HashMap::new();
    let mut quality_gate_pass = 0usize;
    let mut quality_gate_fail = 0usize;
    let mut no_tree = 0usize;

    for (c, query_feat) in &crop_feats {
        let weight = char_weight(*c);

        // Find nearest neighbor + everything within (1.5 × thoroughness)× that distance.
        let hits: Vec<(usize, f32)> = if let Some(points) = index.flat_vecs.get(c) {
            nearest_within_factor_brute(points, query_feat, 1.5 * thoroughness)
        } else {
            no_tree += 1;
            continue;
        };

        // Quality gate: scaled by thoroughness (default 0.5, higher = more permissive)
        let min_dist_sq = hits.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
        if min_dist_sq > 0.5 * thoroughness {
            quality_gate_fail += 1;
            continue;
        }
        quality_gate_pass += 1;

        for (font_id, dist_sq) in &hits {

            // ε = 1e-10 avoids log(0) for self-matches / identical OT variants
            let log_d = (dist_sq + 1e-10_f32).ln();
            font_log_dists.entry(*font_id).or_default().push((log_d, weight));
        }
    }

    eprintln!(
        "  CI: {} crops, {} pass gate, {} fail gate, {} no_tree → {} fonts in voting",
        crop_feats.len(),
        quality_gate_pass,
        quality_gate_fail,
        no_tree,
        font_log_dists.len(),
    );

    // Aggregate: geometric mean of distances per font.
    //
    // For fonts missing from some characters, we penalize with a large
    // distance (the penalty is the max log-distance seen across all fonts
    // for that character slot).  But first we enforce the quorum — fonts
    // appearing in fewer than ceil(n/2) characters are dropped entirely.
    let penalty_log_dist = 0.0_f32; // log(1.0) = 0; i.e. d²=1.0, very far in normalized space

    // Keep a backup for the "at least 1" fallback (quorum may drop everything)
    let font_log_dists_backup: HashMap<usize, Vec<(f32, f32)>> = font_log_dists.clone();

    let mut scores: Vec<(String, f32)> = font_log_dists
        .into_iter()
        .filter_map(|(font_id, log_dists)| {
            let matched = log_dists.len();
            // Quorum gate: must appear in enough character neighborhoods
            if matched < quorum {
                return None;
            }
            let name = index.font_names_table.get(font_id)?.clone();

            // Weighted mean of log-distances.
            // Pad missing characters with the penalty distance.
            let mut total_weight = 0.0_f32;
            let mut weighted_sum = 0.0_f32;
            for (ld, w) in &log_dists {
                weighted_sum += ld * w;
                total_weight += w;
            }
            // Penalty for missing characters
            for _ in matched..n_chars {
                weighted_sum += penalty_log_dist * 1.0;
                total_weight += 1.0;
            }

            let mean_log_dist = weighted_sum / total_weight.max(1e-9);
            // Negate so higher = better (smaller distance = better match)
            let score = -mean_log_dist;
            Some((name, score))
        })
        .collect();

    scores.retain(|(_, s)| s.is_finite());
    // Sort descending (higher = better = closer match)
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.contains('[').cmp(&b.0.contains('[')))
    });

    // ── Statistical cutoff: keep best + near-ties ────────────────────
    // CI is the recall stage — high recall, lower precision (no spacing
    // info).  We keep the top score and anything within k·σ of it.
    //
    // σ is computed on the top 50 scores only (the contender pool), not
    // the full distribution.  The full distribution has a long left tail
    // of clearly-wrong fonts that inflates σ and makes the cutoff too
    // generous.  The top-50 σ measures spread among actual contenders.
    //
    // k = 0.5 * thoroughness: default thoroughness=1.0 → k=0.5.
    // Higher thoroughness widens the window (more candidates, slower).
    if scores.len() >= 2 {
        let top_n = 50.min(scores.len());
        let vals: Vec<f32> = scores.iter().take(top_n).map(|(_, s)| *s).collect();
        let n = vals.len() as f32;
        let mean = vals.iter().sum::<f32>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let sigma = variance.sqrt();
        let best = vals[0];
        let k = 0.5 * thoroughness;
        let cutoff = best - k * sigma;
        let before = scores.len();
        scores.retain(|(_, s)| *s >= cutoff);
        eprintln!(
            "  CI sigma cutoff: best={:.3} top50_σ={:.3} cutoff={:.3} → {} of {} kept",
            best, sigma, cutoff, scores.len(), before,
        );
    }

    // Guarantee at least 1 result: if quorum dropped everything, return the
    // single font with the best (lowest) average distance across whatever
    // characters it did appear in.
    if scores.is_empty() && !font_log_dists_backup.is_empty() {
        if let Some(best) = font_log_dists_backup
            .into_iter()
            .filter_map(|(font_id, log_dists)| {
                let name = index.font_names_table.get(font_id)?.clone();
                let total_weight: f32 = log_dists.iter().map(|(_, w)| w).sum();
                let weighted_sum: f32 = log_dists.iter().map(|(ld, w)| ld * w).sum();
                let mean = weighted_sum / total_weight.max(1e-9);
                Some((name, -mean))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            scores.push(best);
        }
    }

    scores
}

/// Search the index for a single character and return detailed results.
/// Useful for diagnostics — shows which fonts fall within the tolerance radius.
pub fn search_single_char(
    index: &CharIndex,
    ch: char,
    crop: &GrayImage,
) -> Option<CharSearchResult> {
    let feats = compute_features(crop)?;
    let query = feats.as_weighted_slice();

    let points = index.flat_vecs.get(&ch)?;

    // Find nearest neighbor + everything within 2× that distance
    let hits = nearest_within_factor_brute(points, &query, 2.0);
    let n_within_radius = hits.len();

    // Score with cosine similarity
    let char_entries = index.entries.get(&ch)?;
    let entry_map: HashMap<&str, &CharFeatures> = char_entries.iter()
        .map(|e| (e.font_name.as_str(), &e.features))
        .collect();

    let mut candidates: Vec<(String, f32)> = hits.iter()
        .filter_map(|(font_id, _)| {
            let name = index.font_names_table.get(*font_id)?;
            let feats = entry_map.get(name.as_str())?;
            let idx_weighted = feats.as_weighted_slice();
            let sim = cosine_similarity(&query, &idx_weighted);
            Some((name.clone(), sim))
        })
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Some(CharSearchResult {
        ch,
        candidates,
        radius: 0.0, // kNN mode — no radius
        n_within_radius,
    })
}

/// Legacy API — thin wrapper around search_candidates for backward compat.
pub fn match_line_chars(
    crops: &[(char, GrayImage)],
    index: &CharIndex,
    _top_n: usize,
) -> Vec<(String, f32)> {
    search_candidates(index, crops, 1.0)
}

/// Cosine similarity between two feature vectors.
fn cosine_similarity(a: &[f32; FEAT_LEN], b: &[f32; FEAT_LEN]) -> f32 {
    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;
    for i in 0..FEAT_LEN {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// Serialisation (simple binary format)
// ---------------------------------------------------------------------------

/// Bump this whenever the feature vector, normalization, or serialization
/// layout changes. Stale caches auto-rebuild on mismatch.
const INDEX_VERSION: u32 = 6;
const INDEX_MAGIC: &[u8; 4] = b"UCIX";

/// Save the character index to a binary file.
///
/// Format v5:
/// - [u8; 4]: magic bytes b"UCIX"
/// - u32: format version (4)
/// - u32: feature vector length (FEAT_LEN)
/// - u32: number of characters
/// - For each character:
///   - u32: char as u32
///   - u32: number of font entries
///   - For each entry:
///     - u32: font name length
///     - [u8]: font name bytes
///     - [f32; FEAT_LEN]: raw feature vector
///   - [f32; FEAT_LEN]: per-dimension σ values
/// - u32: number of skipped fonts
/// - For each skipped font:
///   - u32: name length
///   - [u8]: name bytes
pub fn save_index(index: &CharIndex, path: &Path) -> io::Result<()> {
    use std::io::Write;
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(INDEX_MAGIC);
    buf.extend_from_slice(&INDEX_VERSION.to_le_bytes());
    buf.extend_from_slice(&(FEAT_LEN as u32).to_le_bytes());

    let n_chars = index.entries.len() as u32;
    buf.extend_from_slice(&n_chars.to_le_bytes());

    for (c, entries) in &index.entries {
        buf.extend_from_slice(&(*c as u32).to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in entries {
            let name_bytes = e.font_name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            let feat = e.features.as_slice();
            for &v in feat.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Write per-dimension σ for this character
        let sigmas = index.dim_sigmas.get(c)
            .cloned()
            .unwrap_or([0.0f32; FEAT_LEN]);
        for &s in sigmas.iter() {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    // Append skipped_fonts section
    buf.extend_from_slice(&(index.skipped_fonts.len() as u32).to_le_bytes());
    for name in &index.skipped_fonts {
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
    }

    let mut f = std::fs::File::create(path)?;
    f.write_all(&buf)?;
    Ok(())
}

/// Load a character index from a binary file.
pub fn load_index(path: &Path) -> io::Result<CharIndex> {
    use std::io::Read;
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    let mut pos;

    if data.len() < 12 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "index file too small"));
    }
    if &data[0..4] != INDEX_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            "stale index: missing magic header (pre-v2 format)"));
    }
    pos = 4;
    let version = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
    pos += 4;
    if version != INDEX_VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("stale index: version {version}, expected {INDEX_VERSION}")));
    }
    let feat_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
    pos += 4;
    if feat_len as usize != FEAT_LEN {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("stale index: feat_len {feat_len}, expected {FEAT_LEN}")));
    }

    let read_u32 = |pos: &mut usize| -> io::Result<u32> {
        if *pos + 4 > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let read_f32 = |pos: &mut usize| -> io::Result<f32> {
        if *pos + 4 > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };

    let n_chars = read_u32(&mut pos)?;
    let mut entries: HashMap<char, Vec<FontCharEntry>> = HashMap::new();
    let mut dim_sigmas: HashMap<char, [f32; FEAT_LEN]> = HashMap::new();

    for _ in 0..n_chars {
        let c_u32 = read_u32(&mut pos)?;
        let c = char::from_u32(c_u32).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid char")
        })?;
        let n_fonts = read_u32(&mut pos)?;
        let mut font_entries = Vec::with_capacity(n_fonts as usize);

        for _ in 0..n_fonts {
            let name_len = read_u32(&mut pos)? as usize;
            if pos + name_len > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated name"));
            }
            let font_name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;

            let mut profile = [0.0f32; PROFILE_BINS];
            for p in &mut profile {
                *p = read_f32(&mut pos)?;
            }
            let aspect = read_f32(&mut pos)?;
            let ink_density = read_f32(&mut pos)?;
            let v_center = read_f32(&mut pos)?;
            let h_balance = read_f32(&mut pos)?;
            let serif_score = read_f32(&mut pos)?;
            let stroke_contrast = read_f32(&mut pos)?;
            let xh_cap_ratio = read_f32(&mut pos)?;

            // v2 features
            let counter_area_ratio = read_f32(&mut pos)?;
            let counter_centroid_x = read_f32(&mut pos)?;
            let counter_centroid_y = read_f32(&mut pos)?;
            let counter_aspect = read_f32(&mut pos)?;
            let mut terminal_angles = [0.0f32; TERMINAL_ANGLE_BINS];
            for t in &mut terminal_angles {
                *t = read_f32(&mut pos)?;
            }
            let ink_perimeter = read_f32(&mut pos)?;
            let compactness = read_f32(&mut pos)?;
            let mut h_crossings = [0.0f32; CROSSING_BINS];
            for hc in &mut h_crossings {
                *hc = read_f32(&mut pos)?;
            }

            font_entries.push(FontCharEntry {
                font_name,
                features: CharFeatures {
                    profile,
                    aspect,
                    ink_density,
                    v_center,
                    h_balance,
                    serif_score,
                    stroke_contrast,
                    xh_cap_ratio,
                    counter_area_ratio,
                    counter_centroid_x,
                    counter_centroid_y,
                    counter_aspect,
                    terminal_angles,
                    ink_perimeter,
                    compactness,
                    h_crossings,
                },
            });
        }

        // Read per-dimension σ
        let mut sigmas = [0.0f32; FEAT_LEN];
        for s in &mut sigmas {
            *s = read_f32(&mut pos)?;
        }
        dim_sigmas.insert(c, sigmas);

        entries.insert(c, font_entries);
    }

    // Read skipped_fonts section
    let mut skipped_fonts = std::collections::HashSet::new();
    if pos < data.len() {
        if let Ok(n_skipped) = read_u32(&mut pos) {
            for _ in 0..n_skipped {
                let name_len = read_u32(&mut pos)? as usize;
                if pos + name_len > data.len() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated skipped font name"));
                }
                let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
                pos += name_len;
                skipped_fonts.insert(name);
            }
        }
    }

    let mut index = CharIndex {
        entries,
        skipped_fonts,
        font_names_table: Vec::new(),
        flat_vecs: HashMap::new(),
        dim_sigmas,
    };
    // Build flat vecs from loaded entries
    index.rebuild_vecs();
    Ok(index)
}

/// Read just the 12-byte header from a cached index file.
pub fn peek_header(path: &Path) -> io::Result<(u32, u32)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut header = [0u8; 12];
    f.read_exact(&mut header)?;

    if &header[0..4] != INDEX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing UCIX magic header",
        ));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let feat_len = u32::from_le_bytes(header[8..12].try_into().unwrap());
    Ok((version, feat_len))
}

/// Return the current expected version and feature length.
pub fn expected_header() -> (u32, u32) {
    (INDEX_VERSION, FEAT_LEN as u32)
}

/// Count the number of unique font names across all characters in the index.
pub fn count_fonts(index: &CharIndex) -> usize {
    index.font_names().len()
}

impl CharIndex {
    /// Collect all unique font names known to this index (both indexed and skipped).
    pub fn font_names(&self) -> std::collections::HashSet<String> {
        let mut names = self.skipped_fonts.clone();
        for entries in self.entries.values() {
            for e in entries {
                names.insert(e.font_name.clone());
            }
        }
        names
    }

    /// Collect only the font names that have actual indexed entries (not skipped).
    pub fn indexed_font_names(&self) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        for entries in self.entries.values() {
            for e in entries {
                names.insert(e.font_name.clone());
            }
        }
        names
    }

    /// Merge another index into this one.
    pub fn merge(&mut self, other: CharIndex) {
        for (c, new_entries) in other.entries {
            let existing = self.entries.entry(c).or_default();
            let existing_names: std::collections::HashSet<String> =
                existing.iter().map(|e| e.font_name.clone()).collect();
            for entry in new_entries {
                if !existing_names.contains(&entry.font_name) {
                    existing.push(entry);
                }
            }
        }
        self.skipped_fonts.extend(other.skipped_fonts);
        // Rebuild vecs after merge
        self.rebuild_vecs();
    }

    /// Remove all entries for the given font names.
    pub fn remove_fonts(&mut self, names: &std::collections::HashSet<String>) {
        for entries in self.entries.values_mut() {
            entries.retain(|e| !names.contains(&e.font_name));
        }
        self.skipped_fonts.retain(|n| !names.contains(n));
        // Rebuild vecs after removal
        self.rebuild_vecs();
    }

    /// Get the per-dimension σ values for a given character (for diagnostics).
    pub fn get_dim_sigmas(&self, c: char) -> Option<&[f32; FEAT_LEN]> {
        self.dim_sigmas.get(&c)
    }

    /// Get the font name for a font_id (for diagnostics).
    pub fn font_name_by_id(&self, id: usize) -> Option<&str> {
        self.font_names_table.get(id).map(|s| s.as_str())
    }

    /// Number of fonts indexed for a given character.
    pub fn tree_size(&self, c: char) -> usize {
        self.flat_vecs.get(&c).map(|v| v.len()).unwrap_or(0)
    }
}
