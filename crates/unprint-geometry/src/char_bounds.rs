//! Character ink bounds measurement - pure geometry, no font deps.
//! Batch API: measures all chars in a word at once, returns Vec<CharInkBounds>.
//! Hot per-pixel loops stay inline inside this crate.

use std::collections::HashMap;
use std::sync::OnceLock;
use image::GrayImage;

use crate::bbox::{CharInkBounds, WordGeoMeasurement};

/// Type alias for seam paths: boundary x -> list of [row, seam_x]
pub type SeamPaths = HashMap<u32, Vec<[u32; 2]>>;

fn get_gamma() -> f64 {
    static GAMMA_CACHE: OnceLock<f64> = OnceLock::new();
    *GAMMA_CACHE.get_or_init(|| {
        std::env::var("UNPRINT_GAMMA")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.2)
    })
}

fn get_frac_enabled() -> bool {
    static FRAC_CACHE: OnceLock<bool> = OnceLock::new();
    *FRAC_CACHE.get_or_init(|| {
        if let Ok(v) = std::env::var("UNPRINT_FRAC") {
            match v.to_ascii_lowercase().as_str() {
                "0" | "false" | "off" | "no" => return false,
                _ => return true,
            }
        }
        true
    })
}

fn get_blend_mode() -> u8 {
    static BLEND_CACHE: OnceLock<u8> = OnceLock::new();
    *BLEND_CACHE.get_or_init(|| {
        if let Ok(v) = std::env::var("UNPRINT_BLEND") {
            if let Ok(n) = v.parse::<u8>() {
                return n.min(2);
            }
            match v.to_ascii_lowercase().as_str() {
                "frac" | "0" => return 0,
                "blend" | "1" | "mix" => return 1,
                "weighted" | "2" => return 2,
                _ => {}
            }
        }
        0 // default: pure fractional center as requested
    })
}

fn coverage_lut() -> &'static [f64; 256] {
    static LUT: OnceLock<[f64; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let gamma = get_gamma();
        let mut lut = [0.0f64; 256];
        if gamma <= 1.0 || gamma <= 1.0001 {
            for p in 0..256 {
                lut[p] = (255 - p) as f64 / 255.0;
            }
        } else {
            for p in 0..256 {
                let s = p as f64 / 255.0;
                let c = 1.0 - s.powf(gamma);
                lut[p] = if c < 0.0 { 0.0 } else if c > 1.0 { 1.0 } else { c };
            }
        }
        lut
    })
}

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

    let frac_enabled = get_frac_enabled();

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
                frac_left: x0_rect as f64,
                frac_right: x0_rect as f64,
            });
            continue;
        }

        let left_seam = seam_paths.get(&b_left);
        let right_seam = seam_paths.get(&b_right);

        // Expanded bounds to include winding seam excursions (prevents 1-px gap like col63=142)
        let x0_exp = left_seam
            .map(|sp| sp.iter().map(|p| p[1] as usize).min().unwrap_or(b_left as usize).min(b_left as usize))
            .unwrap_or(b_left as usize)
            .min(w as usize);
        let x1_exp = right_seam
            .map(|sp| sp.iter().map(|p| p[1] as usize).max().unwrap_or(b_right as usize).max(b_right as usize))
            .unwrap_or(b_right as usize)
            .saturating_add(1)
            .min(w as usize);
        let x0_rect = x0_exp;
        let x1_rect = x1_exp;

        let mut x_min = x1_rect;
        let mut x_max = x0_rect;
        let mut y_min = h as usize;
        let mut y_max = 0usize;
        let mut has_ink = false;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_m = 0.0f64;

        let cov_lut = coverage_lut(); // [c] where c = coverage, mass = c*255

        for y in 0..h as usize {
            let mut left_limit = x0_rect;
            let mut right_limit = x1_rect;

            if let Some(sp) = left_seam {
                let cols: Vec<usize> = sp
                    .iter()
                    .filter(|p| p[0] as usize == y)
                    .map(|p| p[1] as usize)
                    .collect();
                if !cols.is_empty() {
                    let s_min = *cols.iter().min().unwrap();
                    let s_max = *cols.iter().max().unwrap();
                    let s_min_c = s_min.min(w as usize - 1);
                    let s_max_c = s_max.min(w as usize - 1);
                    let left_adj = if s_min_c > 0 {
                        word_img.get_pixel((s_min_c - 1) as u32, y as u32).0[0]
                    } else {
                        255
                    };
                    let right_adj = if s_max_c + 1 < w as usize {
                        word_img.get_pixel((s_max_c + 1) as u32, y as u32).0[0]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj;
                    left_limit = if assign_to_left {
                        (s_max_c + 1).max(x0_rect).min(x1_rect)
                    } else {
                        s_min_c.max(x0_rect).min(x1_rect)
                    };
                }
            }
            if let Some(sp) = right_seam {
                let cols: Vec<usize> = sp
                    .iter()
                    .filter(|p| p[0] as usize == y)
                    .map(|p| p[1] as usize)
                    .collect();
                if !cols.is_empty() {
                    let s_min = *cols.iter().min().unwrap();
                    let s_max = *cols.iter().max().unwrap();
                    let s_min_c = s_min.min(w as usize - 1);
                    let s_max_c = s_max.min(w as usize - 1);
                    let left_adj = if s_min_c > 0 {
                        word_img.get_pixel((s_min_c - 1) as u32, y as u32).0[0]
                    } else {
                        255
                    };
                    let right_adj = if s_max_c + 1 < w as usize {
                        word_img.get_pixel((s_max_c + 1) as u32, y as u32).0[0]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj;
                    right_limit = if assign_to_left {
                        (s_max_c + 1).min(x1_rect).max(left_limit)
                    } else {
                        s_min_c.min(x1_rect).max(left_limit)
                    };
                }
            }

            for x in left_limit..right_limit {
                let pixel = word_img.get_pixel(x as u32, y as u32).0[0];
                if pixel < 255 {
                    has_ink = true;
                    if x < x_min { x_min = x; }
                    if x > x_max { x_max = x; }
                    if y < y_min { y_min = y; }
                    if y > y_max { y_max = y; }
                }
                // weighted centroid now uses any non-white fringe, mass = coverage*255 (gamma-decoded)
                // coverage for 255 = 0 gives zero mass, so include <255
                if pixel < 255 {
                    let m = cov_lut[pixel as usize] * 255.0;
                    sum_x += m * x as f64;
                    sum_y += m * y as f64;
                    sum_m += m;
                }
            }
        }

        let cb = if sum_m == 0.0 {
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
                frac_left: x0_rect as f64,
                frac_right: x1_rect as f64,
            }
        } else {
            // weighted center fixes right-edge light fringe being dropped by <200 binary cutoff
            let cx_w = sum_x / sum_m;
            let cy_w = sum_y / sum_m;
            if !has_ink {
                CharInkBounds {
                    cx: cx_w,
                    cy: cy_w,
                    width: (x1_rect - x0_rect) as f64,
                    height: h as f64,
                    x_min: x0_rect as u32,
                    x_max: x1_rect as u32,
                    y_min: 0,
                    y_max: h,
                    frac_left: x0_rect as f64,
                    frac_right: x1_rect as f64,
                }
            } else {
                // fractional edge vote
                let (frac_left, frac_right, width_frac) = if frac_enabled {
                    let mut darkest_left: u8 = 255;
                    let mut darkest_right: u8 = 255;
                    let mut found_left = false;
                    let mut found_right = false;
                    let y0 = y_min.min(h as usize - 1);
                    let y1 = y_max.min(h as usize - 1);
                    for y in y0..=y1 {
                        let mut left_limit = x0_rect;
                        let mut right_limit = x1_rect;
                        if let Some(sp) = left_seam {
                            let cols: Vec<usize> = sp.iter().filter(|p| p[0] as usize == y).map(|p| p[1] as usize).collect();
                            if !cols.is_empty() {
                                let s_min = *cols.iter().min().unwrap();
                                let s_max = *cols.iter().max().unwrap();
                                let s_min_c = s_min.min(w as usize - 1);
                                let s_max_c = s_max.min(w as usize - 1);
                                let left_adj = if s_min_c > 0 { word_img.get_pixel((s_min_c - 1) as u32, y as u32).0[0] } else { 255 };
                                let right_adj = if s_max_c + 1 < w as usize { word_img.get_pixel((s_max_c + 1) as u32, y as u32).0[0] } else { 255 };
                                let assign_to_left = left_adj <= right_adj;
                                left_limit = if assign_to_left { (s_max_c + 1).max(x0_rect).min(x1_rect) } else { s_min_c.max(x0_rect).min(x1_rect) };
                            }
                        }
                        if let Some(sp) = right_seam {
                            let cols: Vec<usize> = sp.iter().filter(|p| p[0] as usize == y).map(|p| p[1] as usize).collect();
                            if !cols.is_empty() {
                                let s_min = *cols.iter().min().unwrap();
                                let s_max = *cols.iter().max().unwrap();
                                let s_min_c = s_min.min(w as usize - 1);
                                let s_max_c = s_max.min(w as usize - 1);
                                let left_adj = if s_min_c > 0 { word_img.get_pixel((s_min_c - 1) as u32, y as u32).0[0] } else { 255 };
                                let right_adj = if s_max_c + 1 < w as usize { word_img.get_pixel((s_max_c + 1) as u32, y as u32).0[0] } else { 255 };
                                let assign_to_left = left_adj <= right_adj;
                                right_limit = if assign_to_left { (s_max_c + 1).min(x1_rect).max(left_limit) } else { s_min_c.min(x1_rect).max(left_limit) };
                            }
                        }
                        if x_min >= left_limit && x_min < right_limit {
                            let p = word_img.get_pixel(x_min as u32, y as u32).0[0];
                            if p < darkest_left {
                                darkest_left = p;
                                found_left = true;
                            }
                        }
                        if x_max >= left_limit && x_max < right_limit {
                            let p = word_img.get_pixel(x_max as u32, y as u32).0[0];
                            if p < darkest_right {
                                darkest_right = p;
                                found_right = true;
                            }
                        }
                    }
                    // If column never visited due to seam narrowing, fallback to binary edge coverage = 1.0 (full)
                    let p_left = if found_left { darkest_left } else { 0 };
                    let p_right = if found_right { darkest_right } else { 0 };
                    let c_left = cov_lut[p_left as usize];
                    let c_right = cov_lut[p_right as usize];
                    let l = x_min as f64 + (1.0 - c_left);
                    let r = x_max as f64 + c_right;
                    // R = (x_max+1) - (1-c_right) = x_max + c_right, same as above
                    let mut wf = r - l;
                    if wf < 1.0 { wf = 1.0; }
                    (l, r, wf)
                } else {
                    let l = x_min as f64;
                    let r = x_max as f64 + 1.0;
                    (l, r, (x_max - x_min + 1) as f64)
                };

                // fractional geometry fix: feed L/R into cx, not just width
                let blend = get_blend_mode();
                let center_frac = (frac_left + frac_right) * 0.5;
                let cx_final = if !frac_enabled {
                    cx_w
                } else {
                    match blend {
                        0 => center_frac, // pure fractional: cx = (L+R)/2
                        1 => 0.5 * center_frac + 0.5 * cx_w, // blended: preserves fringe robustness
                        _ => cx_w, // fallback weighted only
                    }
                };
                CharInkBounds {
                    cx: cx_final,
                    cy: cy_w,
                    width: width_frac,
                    height: (y_max - y_min + 1) as f64,
                    x_min: x_min as u32,
                    x_max: x_max as u32,
                    y_min: y_min as u32,
                    y_max: y_max as u32,
                    frac_left: frac_left,
                    frac_right: frac_right,
                }
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

use crate::params::{quant_half_width_center_px, quant_half_width_pitch_px, quantized_ll, SIGMA_CENTER_PX, SIGMA_PITCH_PX};

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
        let v_ll = quantized_ll(v_err, SIGMA_CENTER_PX, quant_half_width_center_px());
        let (h_err, h_ll) = if i == 0 {
            (None, 0.0)
        } else {
            let obs_pitch = obs_cx[i] - obs_cx[i-1];
            let pred_pitch = pred_cx[i] - pred_cx[i-1];
            let he = obs_pitch - pred_pitch;
            let hl = quantized_ll(he, SIGMA_PITCH_PX, quant_half_width_pitch_px());
            (Some(he), hl)
        };
        out.push(PerCharError { h_err, v_err, h_ll, v_ll });
    }
    out
}
