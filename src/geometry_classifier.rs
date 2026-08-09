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
//!
//! Rule-out short-circuit: if `predict_glyph_positions_and_extents` returns
//! `None` because the font lacks a cmap entry for a required character, the
//! character cannot be rendered, so its geometry log-likelihood is `-infinity`
//! (infinitely bad). Whole-font score is `-infinity`, so the font is ruled out
//! immediately. We return `None` to signal abort; caller inserts into
//! `cannot_render` and prunes as `NEG_INFINITY` with softmax prob 0. Empty
//! `Vec` is not abort: it indicates ligature mismatch (0 usable words) and
//! returns `Some(empty)` valid, keeping the font with SSIM-only scoring.

use image::GrayImage;
use std::collections::HashMap;

pub use unprint_geometry::{CharInkBounds, WordGeoMeasurement};

#[allow(dead_code)]
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
use unprint_geometry::params::{quant_half_width_center_px, quant_half_width_pitch_px, quantized_ll, SIGMA_CENTER_PX, SIGMA_PITCH_PX};

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
#[allow(dead_code, unused_assignments)]
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

        // Find ink bounds within this character's seam-masked x-range
        let mut x_min = x1_rect;
        let mut x_max = x0_rect;
        let mut y_min = h as usize;
        let mut y_max = 0usize;
        let mut has_ink = false;

        for y in 0..h as usize {
            // Determine per-row left/right limits – handles horizontal moves (multiple seam cols in one row)
            // Ownership: darkest adjacent pixel (s_min-1 vs s_max+1) decides which char gets the whole seam run
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
                        raw_word[y * w_us + (s_min_c - 1)]
                    } else {
                        255
                    };
                    let right_adj = if s_max_c + 1 < w_us {
                        raw_word[y * w_us + (s_max_c + 1)]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj; // darkest adjacent left → seam belongs to left char
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
                        raw_word[y * w_us + (s_min_c - 1)]
                    } else {
                        255
                    };
                    let right_adj = if s_max_c + 1 < w_us {
                        raw_word[y * w_us + (s_max_c + 1)]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj; // seam belongs to current (left side) if left adjacent darker
                    right_limit = if assign_to_left {
                        (s_max_c + 1).min(x1_rect).max(left_limit)
                    } else {
                        s_min_c.min(x1_rect).max(left_limit)
                    };
                }
            }
            // For uniform fallback (no seams) left_limit==x0_rect, right_limit==x1_rect

            for x in left_limit..right_limit {
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
                frac_left: x0_rect as f64,
                frac_right: x1_rect as f64,
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
                frac_left: x_min as f64,
                frac_right: x_max as f64 + 1.0,
            }
        };
        result.push(cb);
    }
    WordGeoMeasurement { chars: result }
}

/// Cached path: use GeometryCache predictions (fast, Unicode, keeps both GPOS Pair formats native).
/// If `prune_threshold` is Some(t), we do vertical-first short-circuit:
/// `h_ll <= 0` always, so `h_ll+v_ll <= v_ll`. If any `v_ll < t`, the sum will be `< t`
/// regardless of h, so the font will be pruned. We can abort without computing h or remaining chars.
/// Same for `v_ll + h_ll < t` on subsequent chars.
fn per_char_geo_cached_with_threshold(
    font_key: &str,
    wib: &[WordGeoMeasurement],
    word_segs: &[crate::segment::WordSeg],
    geo_cache: &crate::geo_cache::GeometryCache,
    prune_threshold: Option<f32>,
) -> Option<Vec<PerCharGeo>> {
    let _ = prune_threshold; // threshold no longer aborts; pruning happens in font_match on min_ll
    if !geo_cache.has_font(font_key) {
        return None;
    }
    let mut result = Vec::new();
    for (seg_idx, (wmeas, ws)) in wib.iter().zip(word_segs.iter()).enumerate() {
        let word_bounds = &wmeas.chars;
        if word_bounds.is_empty() {
            continue;
        }
        // Try to get predictions for this word from cache.
        // Cache contains full Unicode (cmap) + FB00-FB04 ligature codepoints.
        // Non-BMP / missing cmap entries will miss and fall back to shaped path.
        // Ligature codepoints (FB00-FB04) ARE in cache and score as single glyphs.
        // Plain "ff" (['f','f']) is 2 chars, stays 2 glyphs (liga disabled for plain).
        //
        // Rule-out short-circuit: if the font lacks a cmap entry for a required
        // character it cannot render that character at all. Its geometry
        // log-likelihood for that char would be -infinity (infinitely bad score),
        // so the whole-font score is -infinity regardless of other chars.
        // Abort (return None) is therefore a valid early short-circuit for an
        // infinitely bad score — we rule the font out immediately without
        // scoring remaining chars.
        let Some(preds_fu_ext) = geo_cache.predict_glyph_positions_and_extents(font_key, &ws.chars) else { return None; };
        if preds_fu_ext.len() != word_bounds.len() {
            // Ligature path mismatch: font cannot render the ligature segmentation (e.g. lacks ff liga
            // or lacks FB00 cmap with correct advance). For lig path this font is invalid for this
            // segmentation — it would get empty geo (0 penalty) and unfairly beat fonts that do
            // support the ligature and incur small negative geo. Prune it from this path.
            if ws.chars.iter().any(|c| crate::font_scan::is_ligature_char(*c)) {
                return None;
            }
            // Plain path "ff" → 1 glyph (liga enabled elsewhere) but we have 2 bounds — skip word, not font.
            continue;
        }
        // Perf slice 3: avoid two intermediate Vec allocs (preds_fu, preds) per word per font.
        // preds_fu_ext is Vec<(cx,cy,y_min,y_max)> in font units. Scale from font units → px
        // is center-span (unbiased). We compute scale via shared helper
        // `geometry_scale::center_span_scale` — single source of truth for midpoint scaling.
        let scale = if word_bounds.len() >= 2 {
            let obs_first = word_bounds.first().unwrap().cx;
            let obs_last = word_bounds.last().unwrap().cx;
            let pred_first = preds_fu_ext.first().unwrap().0;
            let pred_last = preds_fu_ext.last().unwrap().0;
            crate::geometry_scale::center_span_scale(obs_first, obs_last, pred_first, pred_last)
                .unwrap_or_else(|| {
                    // fallback to height ratio if span degenerate (should be rare)
                    let obs_h = word_bounds.iter().map(|b| b.height).fold(0.0_f64, f64::max).max(1.0);
                    let pred_h = preds_fu_ext.iter().map(|(_,_,y_min,y_max)| (y_max - y_min).abs()).fold(0.0_f64, f64::max).max(1.0);
                    obs_h / pred_h.max(1.0)
                })
        } else {
            // single char: fall back to height ratio (h_err is None anyway)
            let obs_h = word_bounds[0].height.max(1.0);
            let (_,_, y_min, y_max) = preds_fu_ext[0];
            let pred_h = (y_max - y_min).abs().max(1.0);
            obs_h / pred_h.max(1.0)
        };

        // Word vertical center: mean of centers so sum_v = 0 by construction
        let obs_word_cy = word_bounds.iter().map(|b| b.cy).sum::<f64>() / word_bounds.len() as f64;
        // pred_word_cy = mean(pred_cy) where pred_cy = cy_fu * -scale
        let pred_word_cy = {
            let sum: f64 = preds_fu_ext.iter().map(|(_, cy, _, _)| cy * -scale).sum();
            sum / preds_fu_ext.len() as f64
        };

        // Second pass: emit PerCharGeo without allocating preds Vec. Keep prev_pred_cx for pitch.
        let mut prev_obs_cx: Option<f64> = None;
        let mut prev_pred_cx: Option<f64> = None;
        for (orig_idx, (bounds, (cx_fu, cy_fu, _, _))) in word_bounds.iter().zip(preds_fu_ext.iter()).enumerate() {
            let pred_cx = cx_fu * scale;
            let pred_cy = cy_fu * -scale;
            let obs_cx = bounds.cx;
            let obs_cy = bounds.cy;

            let obs_cy_rel = obs_cy - obs_word_cy;
            let pred_cy_rel = pred_cy - pred_word_cy;
            let v_err = obs_cy_rel - pred_cy_rel;
            let v_ll = quantized_ll(v_err, SIGMA_CENTER_PX, quant_half_width_center_px());

            let (obs_pitch, pred_pitch, h_err, h_ll) = if orig_idx == 0 {
                (None, None, None, 0.0)
            } else {
                let prev_cx = prev_obs_cx.unwrap();
                let ppcx = prev_pred_cx.unwrap();
                let obs_pitch_val = obs_cx - prev_cx;
                let pred_pitch_val = pred_cx - ppcx;
                let h_err_val = obs_pitch_val - pred_pitch_val;
                let h_ll_val = quantized_ll(h_err_val, SIGMA_PITCH_PX, quant_half_width_pitch_px());
                (Some(obs_pitch_val), Some(pred_pitch_val), Some(h_err_val), h_ll_val)
            };

            result.push(PerCharGeo {
                seg_idx,
                orig_idx,
                obs_cx,
                obs_cy,
                pred_cx,
                pred_cy,
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
            prev_obs_cx = Some(obs_cx);
            prev_pred_cx = Some(pred_cx);
        }
    }
    // Return Some even if empty: empty = no words had usable geo (ligature mismatch), not missing glyph.
    // None is reserved for infinite penalty (cannot render char).
    Some(result)
}

fn per_char_geo_cached(
    font_key: &str,
    wib: &[WordGeoMeasurement],
    word_segs: &[crate::segment::WordSeg],
    geo_cache: &crate::geo_cache::GeometryCache,
) -> Option<Vec<PerCharGeo>> {
    per_char_geo_cached_with_threshold(font_key, wib, word_segs, geo_cache, None)
}

/// Shaped path: use HarfBuzz shaping per word (slow, but handles GPOS offsets, ligatures, non-ASCII).
fn per_char_geo_shaped_with_threshold(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    font_cache: &crate::font_cache::FontCache,
    font_registry: &crate::font_scan::FontRegistry,
    prune_threshold: Option<f32>,
) -> Option<Vec<PerCharGeo>> {
    let _ = prune_threshold; // see cached version: no early abort

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
        // Ligature control is now propagated from segmentation winner:
        // - ligature WordSegs have chars containing FB00..FB04 (collapsed)
        // - plain WordSegs have no FB00..FB04
        // For ligature segs we shape the original word_text ("figures") with
        // liga enabled, so HarfBuzz's GSUB produces the fi glyph and glyph
        // count matches the ligature segmentation (fi + g + ...). For plain
        // segs we disable liga so "ff" stays two glyphs.
        // Previously we shaped "\u{FB01}gures" which fails because most fonts
        // lack a cmap entry for FB01; shaping the plain text with liga enabled
        // is the correct way to get the ligature glyph.
        let is_lig_word = ws.chars.iter().any(|c| crate::font_scan::is_ligature_char(*c));
        let allow_liga = is_lig_word;
        // Perf: word_text already holds the original string for both lig and plain.
        // Previous code cloned word_text for lig and collected chars for plain,
        // both equivalent to &word_text. Avoid String alloc per word.
        let text = &ws.word_text;
        // base_features is immutable per font; no need to clone per word.
        // Rule-out: shape_word returns None when HarfBuzz cannot shape because
        // the font lacks a cmap entry — the char cannot be rendered. Its geometry
        // ll would be -infinity (infinitely bad), so whole-font score is -infinity.
        // The `?` here propagates None to the caller, which is a valid abort
        // short-circuit: we rule the font out immediately as impossible.
        let sw = crate::layout::shape_word(&face, &base_features, text, allow_liga)?;
        if sw.glyph_ids.len() != bounds_vec.len() {
            if is_lig_word {
                // Font on lig path without ligature support would get empty geo (0 penalty)
                // and beat fonts that do support the ligature. For correctness, prune it
                // from lig path entirely — it can only compete on plain path.
                return None;
            }
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
            let cx = cursor_fu + x_off + (bbox.x_min as f64 + bbox.x_max as f64) * 0.5;
            let cy = y_off + (bbox.y_min as f64 + bbox.y_max as f64) * 0.5;
            let y_min_a = y_off + bbox.y_min as f64;
            let y_max_a = y_off + bbox.y_max as f64;
            pred_positions.push((cx, cy));
            pred_y_mins_fu.push(y_min_a);
            pred_y_maxs_fu.push(y_max_a);
            cursor_fu += sw.x_advances[i] as f64;
        }

        // Center-span scaling (unbiased): single source of truth via geometry_scale
        let scale = if bounds_vec.len() >= 2 {
            let obs_first = bounds_vec.first().unwrap().cx;
            let obs_last = bounds_vec.last().unwrap().cx;
            let pred_first = pred_positions.first().unwrap().0;
            let pred_last = pred_positions.last().unwrap().0;
            crate::geometry_scale::center_span_scale(obs_first, obs_last, pred_first, pred_last)
                .unwrap_or_else(|| {
                    let obs_h = bounds_vec.iter().map(|b| b.height).fold(0.0_f64, f64::max).max(1.0);
                    let glyph_id = unprint_fonts::ttf_parser::GlyphId(sw.glyph_ids[0] as u16);
                    let bbox = ttfp.glyph_bounding_box(glyph_id).unwrap_or(unprint_fonts::ttf_parser::Rect { x_min: 0, y_min: -1000, x_max: 0, y_max: 0 });
                    let pred_h = (bbox.y_max - bbox.y_min) as f64;
                    obs_h / pred_h.max(1.0)
                })
        } else {
            let obs_h = bounds_vec[0].height.max(1.0);
            // Use ink height from bbox for single char
            let glyph_id = unprint_fonts::ttf_parser::GlyphId(sw.glyph_ids[0] as u16);
            let bbox = ttfp.glyph_bounding_box(glyph_id).unwrap_or(unprint_fonts::ttf_parser::Rect { x_min: 0, y_min: -1000, x_max: 0, y_max: 0 });
            let pred_h = (bbox.y_max - bbox.y_min) as f64;
            obs_h / pred_h.max(1.0)
        };

        let pred_positions_px: Vec<(f64, f64)> = pred_positions.iter()
            .map(|(x, y)| (x * scale, y * -scale))
            .collect();

        // Word vertical center: mean of centers so sum_v = 0 by construction
        let obs_word_cy = bounds_vec.iter().map(|b| b.cy).sum::<f64>() / bounds_vec.len() as f64;
        let pred_word_cy = pred_positions_px.iter().map(|(_, cy)| *cy).sum::<f64>() / pred_positions_px.len() as f64;

        for (orig_idx, (bounds, (pred_cx, pred_cy))) in bounds_vec.iter().zip(pred_positions_px.iter()).enumerate() {
            let obs_cx = bounds.cx;
            let obs_cy = bounds.cy;

            let obs_cy_rel = obs_cy - obs_word_cy;
            let pred_cy_rel = pred_cy - pred_word_cy;
            let v_err = obs_cy_rel - pred_cy_rel;
            let v_ll = quantized_ll(v_err, SIGMA_CENTER_PX, quant_half_width_center_px());

            let (obs_pitch, pred_pitch, h_err, h_ll) = if orig_idx == 0 {
                (None, None, None, 0.0)
            } else {
                let prev = &bounds_vec[orig_idx - 1];
                let (prev_pred_cx, _) = &pred_positions_px[orig_idx - 1];
                let obs_pitch_val = obs_cx - prev.cx;
                let pred_pitch_val = pred_cx - prev_pred_cx;
                let h_err_val = obs_pitch_val - pred_pitch_val;
                let h_ll_val = quantized_ll(h_err_val, SIGMA_PITCH_PX, quant_half_width_pitch_px());
                (Some(obs_pitch_val), Some(pred_pitch_val), Some(h_err_val), h_ll_val)
            };

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
    // Empty = no usable words (ligature mismatch), not missing glyph → keep font, not infinite penalty.
    Some(result)
}

fn per_char_geo_shaped(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    font_cache: &crate::font_cache::FontCache,
    font_registry: &crate::font_scan::FontRegistry,
) -> Option<Vec<PerCharGeo>> {
    per_char_geo_shaped_with_threshold(font_key, word_segs, wib, font_cache, font_registry, None)
}

/// Compute per-character geometry for a font.
pub fn per_char_geo_for_font_with_threshold(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    font_cache: &crate::font_cache::FontCache,
    geo_cache: &crate::geo_cache::GeometryCache,
    font_registry: &crate::font_scan::FontRegistry,
    prune_threshold: Option<f32>,
) -> Option<Vec<PerCharGeo>> {
    if let Some(cached) = per_char_geo_cached_with_threshold(font_key, wib, word_segs, geo_cache, prune_threshold) {
        return Some(cached);
    }
    per_char_geo_shaped_with_threshold(font_key, word_segs, wib, font_cache, font_registry, prune_threshold)
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

/// Median font size (em_px) derived from midpoint center-span scales.
///
/// This reuses the exact scale calculation that geometry scoring uses for
/// midpoint log-likelihoods: for each word,
///   obs_span  = last observed cx - first observed cx   (px, from segmentation)
///   pred_span = last predicted cx - first predicted cx (font units, from GPOS-aware cache)
///   scale     = obs_span / pred_span                    (px per font unit)
///   em_px     = scale * upem
///
/// The median across words is robust and does NOT use advance-width matching,
/// so it is not distorted by sidebearings, punctuation, or numeric fragments.
/// This is the "midpoint scale" the user asked to leverage for font-size.
///
/// We filter to alphabetic anchors (at least one alphabetic char) so pure
/// numbers like "1,234,567,890" don't participate, matching the sizing-anchor
/// heuristic but now via geometry rather than width.
pub fn median_em_px_from_midpoints(
    font_key: &str,
    segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    geo_cache: &crate::geo_cache::GeometryCache,
) -> Option<f32> {
    let upem = geo_cache.units_per_em(font_key)? as f64;
    if upem <= 0.0 {
        return None;
    }
    let scales = word_scales_for_font(font_key, segs, wib, geo_cache);
    em_from_word_scales(&scales, upem)
}


/// Compute per-word center-span scales for a font without re-shaping twice.
/// This is the shared path that both geometry scoring and sizing can use,
/// removing the previous redundant `predict_glyph_positions_and_extents` call
/// in sizing that geometry had already done.
///
/// Returns scales in same order as `segs`/`wib` (only for valid anchors).
pub fn word_scales_for_font(
    font_key: &str,
    segs: &[crate::segment::WordSeg],
    wib: &[WordGeoMeasurement],
    geo_cache: &crate::geo_cache::GeometryCache,
) -> Vec<f64> {
    let mut scales = Vec::with_capacity(segs.len());
    if segs.len() != wib.len() {
        return scales;
    }
    for (seg, meas) in segs.iter().zip(wib.iter()) {
        if meas.chars.len() != seg.chars.len() || meas.chars.len() < 2 {
            continue;
        }
        if !crate::geometry_scale::is_sizing_anchor(&seg.chars) {
            continue;
        }
        let obs_first = meas.chars.first().unwrap().cx;
        let obs_last = meas.chars.last().unwrap().cx;
        let Some(pred) = geo_cache.predict_glyph_positions(font_key, &seg.chars) else { continue };
        if pred.len() < 2 { continue; }
        let Some(s) = crate::geometry_scale::center_span_scale(obs_first, obs_last, pred[0].0, pred[pred.len()-1].0) else { continue };
        scales.push(s);
    }
    scales
}


pub fn em_from_word_scales(scales: &[f64], upem: f64) -> Option<f32> {
    if scales.is_empty() || upem <= 0.0 { return None; }
    let mut ems: Vec<f32> = scales.iter().map(|&s| crate::geometry_scale::em_from_scale(s, upem)).collect();
    crate::geometry_scale::median_f32(&mut ems)
}
