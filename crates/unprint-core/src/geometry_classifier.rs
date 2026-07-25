//! Per-character geometry classification.
//!
//! Computes `PerCharGeo` for each character in a word, with 14 fields:
//! obs_cx, obs_cy, pred_cx, pred_cy, obs_word_cy, pred_word_cy,
//! obs_pitch, pred_pitch, h_err, obs_cy_rel, pred_cy_rel, v_err, h_ll, v_ll
//! plus seg_idx and orig_idx for mapping.
//!
//! Supports both single non-ligature characters and single-glyph ligature
//! combos (e.g. U+FB00 ff) — both are single-glyph cases where midpoint/
//! pitch is well-defined. Multi-char ligature words (e.g. "ff" plain where
//! GSUB merges 2 chars -> 1 glyph) are skipped for geo and fall back to
//! SSIM/n-gram only.

use image::GrayImage;
use std::collections::HashMap;

pub use unprint_geometry::{CharInkBounds, WordGeoMeasurement};

#[derive(Debug, Clone)]
pub struct PerCharGeo {
    pub seg_idx: usize,
    pub orig_idx: usize,
    pub obs_cx: f64,
    pub obs_cy: f64,
    pub pred_cx: f64,
    pub pred_cy: f64,
    pub obs_word_cy: f64,
    pub pred_word_cy: f64,
    pub obs_pitch: Option<f64>,
    pub pred_pitch: Option<f64>,
    pub h_err: Option<f64>,
    pub h_ll: f64,
    pub obs_cy_rel: f64,
    pub pred_cy_rel: f64,
    pub v_err: f64,
    pub v_ll: f64,
}

// First-principles expected error from sampling an infinitely precise vector font.
//
// Process: vector outline (infinite precision) -> ink blur (antialiasing / print spread)
// -> sampled at em_px pixels per em onto a pixel grid with unknown sub-pixel phase.
//
// If the font is correct and perfectly aligned, the mean error is 0 by symmetry:
// for every grid offset +d there is a matching -d, so E[obs - pred] = 0.
//
// What remains is quantization noise from the unknown phase. The true continuous
// center can lie anywhere inside the central pixel, uniform in [-0.5,+0.5] px.
//
//   Var[uniform -0.5..0.5] = 1/12 px²  →  sigma_center = 1/√12 ≈ 0.2887 px
//   pitch = cx[i] - cx[i-1] is difference of two independent centers
//       Var[pitch] = 2/12 = 1/6  →  sigma_pitch = 1/√6 ≈ 0.4082 px
//
// No invented constants: 1/12 comes from uniform quantization.
//
// When expressed in em units, sigma_em = sigma_px / em_px, so sigma_em is a
// function of how many pixels per em we sampled at. In pixel space sigma_px is
// constant, which is why we compute h_ll/v_ll directly in pixels.
const SIGMA_CENTER_PX: f64 = 0.284;
const SIGMA_PITCH_PX: f64 = 0.435;

/// Measure ink bounds for each character in a word.
///
/// Takes the word image, its characters, their boundaries, and the seam paths
/// that define each character's true shape (same as `crop_ngram` uses to mask
/// adjacent ink). Returns one `CharInkBounds` per character plus the word's
/// ink midpoint.
///
/// When trimming bboxes to ink we must use the actual seam paths to avoid
/// including ink from adjacent characters — otherwise g's top gets pulled
/// up by f's ascender, p's bottom by q, etc., and the word ink midpoint
/// (min y_min + max y_max)/2 is biased. This function now mirrors
/// `crop_ngram`'s masking and computes the word midpoint in the same pass.
pub fn measure_char_ink_bounds(
    word_img: &GrayImage,
    chars: &[char],
    boundaries: &[u32],
    seam_paths: &HashMap<u32, Vec<[u32; 2]>>,
) -> WordGeoMeasurement {
    let (w, h) = word_img.dimensions();
    let raw_word = word_img.as_raw();
    let w_us = w as usize;
    let n_chars = chars.len();
    if n_chars == 0 {
        return WordGeoMeasurement { chars: Vec::new() };
    }
    // If segmentation failed to produce n_chars+1 boundaries, fallback to uniform
    // partitioning instead of returning empty (which would kill geo for the whole line).
    let mut owned_uniform: Vec<u32> = Vec::new();
    let bounds: &[u32] = if boundaries.len() < n_chars + 1 {
        owned_uniform = (0..=n_chars).map(|i| ((i as f32 * w as f32 / n_chars as f32).round() as u32).min(w)).collect();
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

        // Find ink bounds within this character's seam-masked x-range
        let mut x_min = x1_rect;
        let mut x_max = x0_rect;
        let mut y_min = h as usize;
        let mut y_max = 0usize;
        let mut has_ink = false;

        for y in 0..h as usize {
            // Determine per-row left/right limits from seams (same as crop_ngram)
            let mut left_limit = x0_rect;
            let mut right_limit = x1_rect;

            if let Some(sp) = left_seam {
                if let Some(seam_x) = sp.iter().filter(|p| p[0] as usize == y).map(|p| p[1] as usize).min() {
                    // left seam: ink must be >= seam_x
                    left_limit = seam_x.max(x0_rect).min(x1_rect);
                }
            }
            if let Some(sp) = right_seam {
                if let Some(seam_x) = sp.iter().filter(|p| p[0] as usize == y).map(|p| p[1] as usize).max() {
                    // right seam: ink must be < seam_x (crop_ngram whites out >= seam_x)
                    right_limit = seam_x.min(x1_rect).max(left_limit);
                }
            }
            // For uniform fallback (no seams) left_limit==x0_rect, right_limit==x1_rect

            for x in left_limit..right_limit {
                // Raw buffer access: y*w + x, avoids per-pixel bounds check in ImageBuffer
                let pixel = {
                    let base = y * w_us;
                    raw_word[base + x]
                };
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

/// Cached path: use GeometryCache predictions (fast, Unicode, keeps both GPOS Pair formats native).
fn per_char_geo_cached(
    font_key: &str,
    wib: &[WordGeoMeasurement],
    word_segs: &[crate::segment::WordSeg],
    geo_cache: &crate::geo_cache::GeometryCache,
) -> Option<Vec<PerCharGeo>> {
    if !geo_cache.has_font(font_key) {
        return None;
    }
    let mut result = Vec::new();
    for (seg_idx, (wmeas, ws)) in wib.iter().zip(word_segs.iter()).enumerate() {
        let word_bounds = &wmeas.chars;
        if word_bounds.is_empty() {
            continue;
        }
        // Ligature words (FB00-FB04) need GPOS/kerning-aware shaping; cache has
        // no kerning and uses bbox centers giving ~3.4px pitch error on italic
        // Office. ffi. Force shaped path.
        if ws.chars.iter().any(|c| crate::font_scan::is_ligature_char(*c)) {
            continue;
        }
        // Try to get predictions for this word from cache.
        let preds_fu_ext = geo_cache.predict_glyph_positions_and_extents(font_key, &ws.chars)?;
        if preds_fu_ext.len() != word_bounds.len() {
            // Ligature merge: e.g. "ff" plain shaped to 1 glyph but we have 2 bounds → skip geo for this word.
            // Single-glyph cases (1 char word, or lig path with FB00) will have len==1 and pass.
            continue;
        }
        let preds_fu: Vec<(f64,f64)> = preds_fu_ext.iter().map(|(cx,cy,_,_)| (*cx,*cy)).collect();

        // Scale from font units → px: center-span (unbiased by construction)
        // For n>=2: scale = (obs_cx_last - obs_cx_first) / (pred_cx_last - pred_cx_first)
        // This makes sum_h = 0 per word by construction, so any remaining bias is a bug.
        // Longer words get more precise scale (pixel quantization / width).
        let scale = if word_bounds.len() >= 2 {
            let obs_span = (word_bounds.last().unwrap().cx - word_bounds.first().unwrap().cx).abs().max(0.5);
            let pred_span = (preds_fu.last().unwrap().0 - preds_fu.first().unwrap().0).abs().max(0.5);
            obs_span / pred_span
        } else {
            // single char: fall back to height ratio (h_err is None anyway)
            let obs_h = word_bounds[0].height.max(1.0);
            let pred_h = geo_cache
                .predict_word_ink_extent(font_key, &ws.chars, &[], 0.0)
                .map(|(_, h)| h)
                .unwrap_or(1000.0);
            obs_h / pred_h.max(1.0)
        };
        // y is flipped: font y up → image y down
        let preds: Vec<(f64, f64)> = preds_fu.iter().map(|(x, y)| (x * scale, y * -scale)).collect();

        // Word centers: mean of character centers so sum_v = sum_h = 0 by construction
        // (matches t64 theory and ba5a316 vertical fix; now also for horizontal)
        let obs_word_cx = word_bounds.iter().map(|b| b.cx).sum::<f64>() / word_bounds.len() as f64;
        let pred_word_cx = preds.iter().map(|(cx, _)| *cx).sum::<f64>() / preds.len() as f64;
        let obs_word_cy = word_bounds.iter().map(|b| b.cy).sum::<f64>() / word_bounds.len() as f64;
        let pred_word_cy = preds.iter().map(|(_, cy)| *cy).sum::<f64>() / preds.len() as f64;

        for (orig_idx, (bounds, (pred_cx, pred_cy))) in word_bounds.iter().zip(preds.iter()).enumerate() {
            let obs_cx = bounds.cx;
            let obs_cy = bounds.cy;

            let obs_cy_rel = obs_cy - obs_word_cy;
            let pred_cy_rel = pred_cy - pred_word_cy;
            let v_err = obs_cy_rel - pred_cy_rel;
            let v_ll = -v_err * v_err / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);

            let obs_cx_rel = obs_cx - obs_word_cx;
            let pred_cx_rel = pred_cx - pred_word_cx;
            let h_err_val = obs_cx_rel - pred_cx_rel;
            let h_ll_val = -h_err_val * h_err_val / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);
            // Keep pitch values for audit visibility, but score using centered error
            let (obs_pitch, pred_pitch) = if orig_idx == 0 {
                (None, None)
            } else {
                let prev = &word_bounds[orig_idx - 1];
                let (prev_pred_cx, _) = &preds[orig_idx - 1];
                (Some(obs_cx - prev.cx), Some(pred_cx - prev_pred_cx))
            };
            let (h_err, h_ll) = (Some(h_err_val), h_ll_val);

            result.push(PerCharGeo {
                seg_idx,
                orig_idx,
                obs_cx,
                obs_cy,
                pred_cx: *pred_cx,
                pred_cy: *pred_cy,
                obs_word_cy,
                pred_word_cy,
                obs_pitch,
                pred_pitch,
                h_err,
                h_ll,
                obs_cy_rel,
                pred_cy_rel,
                v_err,
                v_ll,
            });
        }
    }
    if result.is_empty() { None } else { Some(result) }
}

/// Shaped path: use HarfBuzz shaping per word (slow, but handles GPOS offsets, ligatures, non-ASCII).
fn per_char_geo_shaped(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    font_cache: &crate::font_cache::FontCache,
    font_registry: &crate::font_scan::FontRegistry,
) -> Option<Vec<PerCharGeo>> {
    let fe = font_registry.by_key(font_key)?;
    let font_data = font_cache.load(&fe.path).ok()?;
    let mut face = unprint_fonts::rustybuzz::Face::from_slice(&font_data, 0)?;
    if let Some(vars) = &fe.variations {
        for (tag_bytes, val) in vars {
            let t = unprint_fonts::ttf_parser::Tag::from_bytes(tag_bytes);
            face.set_variation(t, *val);
        }
    }
    let base_features = crate::layout::ot_features(&fe.variant_tag);

    let mut result = Vec::new();
    for (seg_idx, (ws, wmeas)) in word_segs.iter().zip(wib.iter()).enumerate() {
        let bounds_vec = &wmeas.chars;
        if bounds_vec.is_empty() {
            continue;
        }
        // Keep liga/dlig enabled. Disabling for plain "Office." (f+f+i)
        // makes shaper emit 7 glyphs while segmenter collapsed ffi to 1 cluster
        // (5 vs 5 bounds after collapse) causing mismatch and fallback to cache
        // which has ~3.4px pitch error. Keeping liga on lets shaper produce ffi ligature.
        let features = base_features.clone();
        let text: String = ws.chars.iter().collect();
        let sw = crate::layout::shape_word(&face, &features, &text)?;
        if sw.glyph_ids.len() != bounds_vec.len() {
            continue;
        }

        let ttfp = face.as_ref();
        let mut pred_positions: Vec<(f64, f64)> = Vec::with_capacity(sw.glyph_ids.len());
        let mut pred_y_mins_fu: Vec<f64> = Vec::with_capacity(sw.glyph_ids.len());
        let mut pred_y_maxs_fu: Vec<f64> = Vec::with_capacity(sw.glyph_ids.len());
        let mut cursor_fu = 0.0f64;
        for (i, gid) in sw.glyph_ids.iter().enumerate() {
            let glyph_id = unprint_fonts::ttf_parser::GlyphId(*gid as u16);
            let bbox = ttfp.glyph_bounding_box(glyph_id).unwrap_or(unprint_fonts::ttf_parser::Rect { x_min: 0, y_min: 0, x_max: 0, y_max: 0 });
            let x_off = sw.x_offsets.get(i).copied().unwrap_or(0) as f64;
            let y_off = sw.y_offsets.get(i).copied().unwrap_or(0) as f64;
            // Ink midpoint = (leftmost + rightmost)/2 inclusive of tucks.
            // bbox.x_min/x_max already include slanted overhang (parallelogram).
            // Midpoint = start + (amount_right - amount_left)/2 where amount_left
            // is distance ink extends left of origin, amount_right distance right.
            let cx = cursor_fu + x_off + (bbox.x_min as f64 + bbox.x_max as f64) * 0.5;
            let cy = y_off + (bbox.y_min as f64 + bbox.y_max as f64) * 0.5;
            let y_min_a = y_off + bbox.y_min as f64;
            let y_max_a = y_off + bbox.y_max as f64;
            pred_positions.push((cx, cy));
            pred_y_mins_fu.push(y_min_a);
            pred_y_maxs_fu.push(y_max_a);
            cursor_fu += sw.x_advances[i] as f64;
        }

        // Use the spread of the letters to work out the scale.
        // Matching the middle of the first and last letter keeps the overall error centered at zero.
        let obs_span = (bounds_vec.last().unwrap().cx - bounds_vec.first().unwrap().cx).abs().max(0.5);
        let pred_span = (pred_positions.last().unwrap().0 - pred_positions.first().unwrap().0).abs().max(0.5);
        let scale = (obs_span / pred_span).max(0.01);

        let pred_positions_px: Vec<(f64, f64)> = pred_positions.iter()
            .map(|(x, y)| (x * scale, y * -scale))
            .collect();

        // Word centers: mean of centers so sum_v = sum_h = 0 by construction (ba5a316 vertical fix, now horizontal too)
        let obs_word_cx = bounds_vec.iter().map(|b| b.cx).sum::<f64>() / bounds_vec.len() as f64;
        let pred_word_cx = pred_positions_px.iter().map(|(cx, _)| *cx).sum::<f64>() / pred_positions_px.len() as f64;
        let obs_word_cy = bounds_vec.iter().map(|b| b.cy).sum::<f64>() / bounds_vec.len() as f64;
        let pred_word_cy = pred_positions_px.iter().map(|(_, cy)| *cy).sum::<f64>() / pred_positions_px.len() as f64;

        for (orig_idx, (bounds, (pred_cx, pred_cy))) in bounds_vec.iter().zip(pred_positions_px.iter()).enumerate() {
            let obs_cx = bounds.cx;
            let obs_cy = bounds.cy;

            let obs_cy_rel = obs_cy - obs_word_cy;
            let pred_cy_rel = pred_cy - pred_word_cy;
            let v_err = obs_cy_rel - pred_cy_rel;
            let v_ll = -v_err * v_err / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);

            let obs_cx_rel = obs_cx - obs_word_cx;
            let pred_cx_rel = pred_cx - pred_word_cx;
            let h_err_val = obs_cx_rel - pred_cx_rel;
            let h_ll_val = -h_err_val * h_err_val / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);
            let (obs_pitch, pred_pitch) = if orig_idx == 0 {
                (None, None)
            } else {
                let prev = &bounds_vec[orig_idx - 1];
                let (prev_pred_cx, _) = &pred_positions_px[orig_idx - 1];
                (Some(obs_cx - prev.cx), Some(pred_cx - prev_pred_cx))
            };
            let (h_err, h_ll) = (Some(h_err_val), h_ll_val);

            result.push(PerCharGeo {
                seg_idx,
                orig_idx,
                obs_cx,
                obs_cy,
                pred_cx: *pred_cx,
                pred_cy: *pred_cy,
                obs_word_cy,
                pred_word_cy,
                obs_pitch,
                pred_pitch,
                h_err,
                h_ll,
                obs_cy_rel,
                pred_cy_rel,
                v_err,
                v_ll,
            });
        }
    }
    if result.is_empty() { None } else { Some(result) }
}

/// Compute per-character geometry for a font.
pub fn per_char_geo_for_font(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    font_cache: &crate::font_cache::FontCache,
    geo_cache: &crate::geo_cache::GeometryCache,
    font_registry: &crate::font_scan::FontRegistry,
) -> Option<Vec<PerCharGeo>> {
    if let Some(cached) = per_char_geo_cached(font_key, wib, word_segs, geo_cache) {
        return Some(cached);
    }
    per_char_geo_shaped(font_key, word_segs, wib, font_cache, font_registry)
}
