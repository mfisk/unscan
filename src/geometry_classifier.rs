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

#[derive(Debug, Clone)]
pub struct CharInkBounds {
    pub cx: f64,
    pub cy: f64,
    pub width: f64,
    pub height: f64,
    pub x_min: u32,
    pub x_max: u32,
    pub y_min: u32,
    pub y_max: u32,
}

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
const SIGMA_CENTER_PX: f64 = 0.2886751345948129; // 1/√12
const SIGMA_PITCH_PX: f64 = 0.4082482904638630;  // 1/√6

/// Measure ink bounds for each character in a word.
///
/// Takes the word image, its characters, and their boundaries (x positions).
/// Returns one `CharInkBounds` per character.
pub fn measure_char_ink_bounds(
    word_img: &GrayImage,
    chars: &[char],
    boundaries: &[u32],
) -> Vec<CharInkBounds> {
    let (w, h) = word_img.dimensions();
    let n_chars = chars.len();
    if n_chars == 0 {
        return Vec::new();
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
        let x0 = bounds[i].min(w.saturating_sub(1)) as usize;
        let x1 = bounds[i + 1].min(w) as usize;
        if x0 >= x1 {
            result.push(CharInkBounds {
                cx: x0 as f64,
                cy: h as f64 / 2.0,
                width: 0.0,
                height: 0.0,
                x_min: x0 as u32,
                x_max: x0 as u32,
                y_min: 0,
                y_max: h,
            });
            continue;
        }

        // Find ink bounds within this character's x-range
        let mut x_min = x1;
        let mut x_max = x0;
        let mut y_min = h as usize;
        let mut y_max = 0usize;
        let mut has_ink = false;

        for x in x0..x1 {
            for y in 0..h as usize {
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

        if !has_ink {
            let cx = (x0 + x1) as f64 / 2.0;
            let cy = h as f64 / 2.0;
            result.push(CharInkBounds {
                cx,
                cy,
                width: (x1 - x0) as f64,
                height: h as f64,
                x_min: x0 as u32,
                x_max: x1 as u32,
                y_min: 0,
                y_max: h,
            });
        } else {
            let cx = (x_min + x_max) as f64 / 2.0;
            let cy = (y_min + y_max) as f64 / 2.0;
            result.push(CharInkBounds {
                cx,
                cy,
                width: (x_max - x_min + 1) as f64,
                height: (y_max - y_min + 1) as f64,
                x_min: x_min as u32,
                x_max: x_max as u32,
                y_min: y_min as u32,
                y_max: y_max as u32,
            });
        }
    }
    result
}

/// Cached path: use GeometryCache predictions (fast, Unicode, keeps both GPOS Pair formats native).
fn per_char_geo_cached(
    font_key: &str,
    wib: &[Vec<CharInkBounds>],
    word_segs: &[crate::segment::WordSeg],
    geo_cache: &crate::geo_cache::GeometryCache,
) -> Option<Vec<PerCharGeo>> {
    if !geo_cache.has_font(font_key) {
        return None;
    }
    let mut result = Vec::new();
    for (seg_idx, (word_bounds, ws)) in wib.iter().zip(word_segs.iter()).enumerate() {
        if word_bounds.is_empty() {
            continue;
        }
        // Try to get predictions for this word from cache.
        // Cache contains full Unicode (cmap) + FB00-FB04 ligature codepoints.
        // Non-BMP / missing cmap entries will miss and fall back to shaped path.
        // Ligature codepoints (FB00-FB04) ARE in cache and score as single glyphs.
        // Plain "ff" (['f','f']) is 2 chars, stays 2 glyphs (liga disabled for plain).
        let preds_fu = geo_cache.predict_glyph_positions(font_key, &ws.chars)?;
        if preds_fu.len() != word_bounds.len() {
            // Ligature merge: e.g. "ff" plain shaped to 1 glyph but we have 2 bounds → skip geo for this word.
            // Single-glyph cases (1 char word, or lig path with FB00) will have len==1 and pass.
            continue;
        }

        // Scale from font units → px: use ink-width ratio (sum obs widths / sum pred ink widths)
        // This is apples-to-apples vs advance which includes side bearings.
        let obs_total_width = word_bounds.iter().map(|b| b.width).sum::<f64>().max(1.0);
        let scale = if let Some(pred_ink_sum) = geo_cache.predict_word_ink_width_sum(font_key, &ws.chars) {
            obs_total_width / pred_ink_sum.max(1.0)
        } else {
            // fallback: height ratio (robust when ink width unavailable)
            let obs_avg_h = word_bounds.iter().map(|b| b.height).sum::<f64>() / word_bounds.len() as f64;
            let pred_h = geo_cache
                .predict_word_ink_extent(font_key, &ws.chars, &[], 0.0)
                .map(|(_, h)| h)
                .unwrap_or(1000.0);
            obs_avg_h / pred_h.max(1.0)
        };
        // y is flipped: font y up → image y down
        let preds: Vec<(f64, f64)> = preds_fu.iter().map(|(x, y)| (x * scale, y * -scale)).collect();

        let obs_word_cy = word_bounds.iter().map(|b| b.cy).sum::<f64>() / word_bounds.len() as f64;
        let pred_word_cy = preds.iter().map(|(_, y)| *y).sum::<f64>() / preds.len() as f64;

        for (orig_idx, (bounds, (pred_cx, pred_cy))) in word_bounds.iter().zip(preds.iter()).enumerate() {
            let obs_cx = bounds.cx;
            let obs_cy = bounds.cy;

            let obs_cy_rel = obs_cy - obs_word_cy;
            let pred_cy_rel = pred_cy - pred_word_cy;
            let v_err = obs_cy_rel - pred_cy_rel;
            let v_ll = -v_err * v_err / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);

            let (obs_pitch, pred_pitch, h_err, h_ll) = if orig_idx == 0 {
                (None, None, None, 0.0)
            } else {
                let prev = &word_bounds[orig_idx - 1];
                let (_, prev_pred_cx) = &preds[orig_idx - 1];
                let obs_pitch_val = obs_cx - prev.cx;
                let pred_pitch_val = pred_cx - prev_pred_cx;
                let h_err_val = obs_pitch_val - pred_pitch_val;
                let h_ll_val = -h_err_val * h_err_val / (2.0 * SIGMA_PITCH_PX * SIGMA_PITCH_PX);
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
    if result.is_empty() { None } else { Some(result) }
}

/// Shaped path: use HarfBuzz shaping per word (slow, but handles GPOS offsets, ligatures, non-ASCII).
fn per_char_geo_shaped(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[Vec<CharInkBounds>],
    font_cache: &crate::font_cache::FontCache,
    font_registry: &crate::font_scan::FontRegistry,
) -> Option<Vec<PerCharGeo>> {
    let fe = font_registry.by_key(font_key)?;
    let font_data = font_cache.load(&fe.path).ok()?;
    let mut face = rustybuzz::Face::from_slice(&font_data, 0)?;
    if let Some(vars) = &fe.variations {
        for (tag_bytes, val) in vars {
            let t = rustybuzz::ttf_parser::Tag::from_bytes(tag_bytes);
            face.set_variation(t, *val);
        }
    }
    let base_features = crate::layout::ot_features(&fe.variant_tag);

    let mut result = Vec::new();
    for (seg_idx, (ws, bounds_vec)) in word_segs.iter().zip(wib.iter()).enumerate() {
        if bounds_vec.is_empty() {
            continue;
        }
        // Per-word liga control using canonical ligature list.
        // Plain words like ['f','f'] have no FB00..FB04 -> disable liga/dlig so "ff" stays two glyphs.
        // Ligature words like ['\u{FB00}'] or ['a','\u{FB04}','u',...] contain a ligature codepoint -> keep liga on.
        let is_lig_word = ws.chars.iter().any(|c| crate::font_scan::is_ligature_char(*c));
        let mut features = base_features.clone();
        if !is_lig_word {
            features.push(rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(b"liga"), 0, ..));
            features.push(rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(b"dlig"), 0, ..));
        }
        let text: String = ws.chars.iter().collect();
        let sw = crate::layout::shape_word(&face, &features, &text)?;
        if sw.glyph_ids.len() != bounds_vec.len() {
            continue;
        }

        let ttfp = face.as_ref();
        let mut pred_positions: Vec<(f64, f64)> = Vec::with_capacity(sw.glyph_ids.len());
        let mut pred_ink_width_sum = 0.0f64;
        let mut cursor_fu = 0.0f64;
        for (i, gid) in sw.glyph_ids.iter().enumerate() {
            let glyph_id = rustybuzz::ttf_parser::GlyphId(*gid as u16);
            let bbox = ttfp.glyph_bounding_box(glyph_id).unwrap_or(rustybuzz::ttf_parser::Rect { x_min: 0, y_min: 0, x_max: 0, y_max: 0 });
            pred_ink_width_sum += (bbox.x_max - bbox.x_min) as f64;
            let x_off = sw.x_offsets.get(i).copied().unwrap_or(0) as f64;
            let y_off = sw.y_offsets.get(i).copied().unwrap_or(0) as f64;
            let cx = cursor_fu + x_off + (bbox.x_min as f64 + bbox.x_max as f64) * 0.5;
            let cy = y_off + (bbox.y_min as f64 + bbox.y_max as f64) * 0.5;
            pred_positions.push((cx, cy));
            cursor_fu += sw.x_advances[i] as f64;
        }

        let obs_total_width: f64 = bounds_vec.iter().map(|b| b.width).sum::<f64>().max(1.0);
        // Use ink-width ratio (not advance) — apples-to-apples with obs_total_width
        let pred_total_ink_width = pred_ink_width_sum.max(1.0);
        let scale = obs_total_width / pred_total_ink_width;

        let pred_positions_px: Vec<(f64, f64)> = pred_positions.iter()
            .map(|(x, y)| (x * scale, y * -scale))
            .collect();

        let obs_word_cy = bounds_vec.iter().map(|b| b.cy).sum::<f64>() / bounds_vec.len() as f64;
        let pred_word_cy = pred_positions_px.iter().map(|(_, y)| *y).sum::<f64>() / pred_positions_px.len() as f64;

        for (orig_idx, (bounds, (pred_cx, pred_cy))) in bounds_vec.iter().zip(pred_positions_px.iter()).enumerate() {
            let obs_cx = bounds.cx;
            let obs_cy = bounds.cy;

            let obs_cy_rel = obs_cy - obs_word_cy;
            let pred_cy_rel = pred_cy - pred_word_cy;
            let v_err = obs_cy_rel - pred_cy_rel;
            let v_ll = -v_err * v_err / (2.0 * SIGMA_CENTER_PX * SIGMA_CENTER_PX);

            let (obs_pitch, pred_pitch, h_err, h_ll) = if orig_idx == 0 {
                (None, None, None, 0.0)
            } else {
                let prev = &bounds_vec[orig_idx - 1];
                let (prev_pred_cx, _) = &pred_positions_px[orig_idx - 1];
                let obs_pitch_val = obs_cx - prev.cx;
                let pred_pitch_val = pred_cx - prev_pred_cx;
                let h_err_val = obs_pitch_val - pred_pitch_val;
                let h_ll_val = -h_err_val * h_err_val / (2.0 * SIGMA_PITCH_PX * SIGMA_PITCH_PX);
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
    if result.is_empty() { None } else { Some(result) }
}

/// Compute per-character geometry for a font.
pub fn per_char_geo_for_font(
    font_key: &str,
    word_segs: &[crate::segment::WordSeg],
    wib: &[Vec<CharInkBounds>],
    font_cache: &crate::font_cache::FontCache,
    geo_cache: &crate::geo_cache::GeometryCache,
    font_registry: &crate::font_scan::FontRegistry,
) -> Option<Vec<PerCharGeo>> {
    if let Some(cached) = per_char_geo_cached(font_key, wib, word_segs, geo_cache) {
        return Some(cached);
    }
    per_char_geo_shaped(font_key, word_segs, wib, font_cache, font_registry)
}
