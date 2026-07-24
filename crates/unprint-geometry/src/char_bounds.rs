//! Character ink bounds measurement - pure geometry, no font deps.
//! Batch API: measures all chars in a word at once, returns Vec<CharInkBounds>.
//! Hot per-pixel loops stay inline inside this crate.

use std::collections::HashMap;
use image::GrayImage;

use crate::bbox::{CharInkBounds, WordGeoMeasurement};

/// Type alias for seam paths: boundary x -> list of [row, seam_x]
pub type SeamPaths = HashMap<u32, Vec<[u32; 2]>>;

/// Measure ink bounds for each character in a word.
///
/// Batch API: takes whole word image + char slice + boundaries + seam paths,
/// returns one `CharInkBounds` per character.
/// This is the batch entry point - one call per word, not per char.
///
/// Mirrors `crop_ngram` masking logic to avoid including adjacent ink.
pub fn measure_char_ink_bounds(
    word_img: &GrayImage,
    chars: &[char],
    boundaries: &[u32],
    seam_paths: &SeamPaths,
) -> WordGeoMeasurement {
    let (w, h) = word_img.dimensions();
    let n_chars = chars.len();
    if n_chars == 0 {
        return WordGeoMeasurement { chars: Vec::new() };
    }
    let owned_uniform: Vec<u32> = (0..=n_chars)
        .map(|i| ((i as f32 * w as f32 / n_chars as f32).round() as u32).min(w))
        .collect();
    let bounds: &[u32] = if boundaries.len() < n_chars + 1 {
        &owned_uniform
    } else {
        boundaries
    };

    let mut result = Vec::with_capacity(n_chars);
    for i in 0..n_chars {
        let b_left = bounds[i];
        let b_right = bounds[i + 1];
        let x0_rect = b_left.min(w.saturating_sub(1)) as usize;
        let x1_rect = b_right.min(w) as usize;
        if x0_rect >= x1_rect {
            result.push(CharInkBounds {
                cx: x0_rect as f64,
                cy: h as f64 / 2.0,
                width: 0.0,
                height: 0.0,
                x_min: x0_rect as u32,
                x_max: x0_rect as u32,
                y_min: 0,
                y_max: h,
            });
            continue;
        }

        let left_seam = seam_paths.get(&b_left);
        let right_seam = seam_paths.get(&b_right);

        let mut x_min = x1_rect;
        let mut x_max = x0_rect;
        let mut y_min = h as usize;
        let mut y_max = 0usize;
        let mut has_ink = false;

        for y in 0..h as usize {
            let mut left_limit = x0_rect;
            let mut right_limit = x1_rect;

            if let Some(sp) = left_seam {
                if let Some(seam_x) = sp.iter().filter(|p| p[0] as usize == y).map(|p| p[1] as usize).min() {
                    left_limit = seam_x.max(x0_rect).min(x1_rect);
                }
            }
            if let Some(sp) = right_seam {
                if let Some(seam_x) = sp.iter().filter(|p| p[0] as usize == y).map(|p| p[1] as usize).max() {
                    right_limit = seam_x.min(x1_rect).max(left_limit);
                }
            }

            for x in left_limit..right_limit {
                let pixel = word_img.get_pixel(x as u32, y as u32).0[0];
                if pixel < 200 {
                    has_ink = true;
                    if x < x_min { x_min = x; }
                    if x > x_max { x_max = x; }
                    if y < y_min { y_min = y; }
                    if y > y_max { y_max = y; }
                }
            }
        }

        let cb = if !has_ink {
            let cx = (x0_rect + x1_rect) as f64 / 2.0;
            let cy = h as f64 / 2.0;
            CharInkBounds {
                cx,
                cy,
                width: (x1_rect - x0_rect) as f64,
                height: h as f64,
                x_min: x0_rect as u32,
                x_max: x1_rect as u32,
                y_min: 0,
                y_max: h,
            }
        } else {
            let cx = (x_min + x_max) as f64 / 2.0;
            let cy = (y_min + y_max) as f64 / 2.0;
            CharInkBounds {
                cx,
                cy,
                width: (x_max - x_min + 1) as f64,
                height: (y_max - y_min + 1) as f64,
                x_min: x_min as u32,
                x_max: x_max as u32,
                y_min: y_min as u32,
                y_max: y_max as u32,
            }
        };
        result.push(cb);
    }
    WordGeoMeasurement { chars: result }
}

/// Batch version: measures many words at once.
/// Takes slices of (image, chars, boundaries, seams), returns Vec<WordGeoMeasurement>.
/// One call, many words - keeps cross-crate call volume low.
pub fn measure_words_ink_bounds_batch(
    inputs: &[(&GrayImage, &[char], &[u32], &SeamPaths)],
) -> Vec<WordGeoMeasurement> {
    inputs.iter().map(|(img, chars, bounds, seams)| {
        measure_char_ink_bounds(img, chars, bounds, seams)
    }).collect()
}

/// Compute per-char vertical/horizontal errors in batch.
/// Pure math, no font deps. Takes slices of obs/pred centres.
#[derive(Debug, Clone)]
pub struct PerCharError {
    pub h_err: Option<f64>,
    pub v_err: f64,
    pub h_ll: f64,
    pub v_ll: f64,
}

const SIGMA_CENTER_PX: f64 = 0.284;
const SIGMA_PITCH_PX: f64 = 0.435;

pub fn batch_per_char_errors(
    obs_cx: &[f64],
    obs_cy: &[f64],
    pred_cx: &[f64],
    pred_cy: &[f64],
) -> Vec<PerCharError> {
    if obs_cx.len() != pred_cx.len() || obs_cy.len() != pred_cy.len() {
        return Vec::new();
    }
    let obs_word_cy = obs_cy.iter().sum::<f64>() / obs_cy.len().max(1) as f64;
    let pred_word_cy = pred_cy.iter().sum::<f64>() / pred_cy.len().max(1) as f64;

    let mut out = Vec::with_capacity(obs_cx.len());
    for i in 0..obs_cx.len() {
        let v_err = (obs_cy[i] - obs_word_cy) - (pred_cy[i] - pred_word_cy);
        let v_ll = -v_err * v_err / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);
        let (h_err, h_ll) = if i == 0 {
            (None, 0.0)
        } else {
            let obs_pitch = obs_cx[i] - obs_cx[i-1];
            let pred_pitch = pred_cx[i] - pred_cx[i-1];
            let he = obs_pitch - pred_pitch;
            let hl = -he * he / (2.0 * SIGMA_PITCH_PX * SIGMA_PITCH_PX);
            (Some(he), hl)
        };
        out.push(PerCharError { h_err, v_err, h_ll, v_ll });
    }
    out
}
