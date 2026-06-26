//! SSIM verification — render vector text back to raster and compare with the
//! original to catch bad replacements before they make it into the output.
//!
//! v5: width-scaled SSIM for glyph-shape comparison + aspect-ratio penalty
//! for catching fonts whose proportions don't match the original.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma};
use crate::ocr::TextRegion;

// ---------------------------------------------------------------------------
// FreeType variable-font axis coordination helper
// ---------------------------------------------------------------------------

/// Apply variable-font design coordinates to a FreeType face.
///
/// Reads the fvar axis order from the raw font data (via ttf_parser) and maps
/// the provided tag→value pairs to `FT_Set_Var_Design_Coordinates`.
fn set_ft_variations<B>(ft_face: &freetype::Face<B>, font_data: &[u8], vars: &[([u8; 4], f32)]) {
    use rustybuzz::ttf_parser;

    let parsed = match ttf_parser::Face::parse(font_data, 0) {
        Ok(f) => f,
        Err(_) => return,
    };

    // Get the axis list in fvar order
    let axes: Vec<ttf_parser::VariationAxis> = parsed.variation_axes().into_iter().collect();
    if axes.is_empty() { return; }

    // Build coordinate array: each axis gets its value from vars, or its default
    let coords: Vec<i64> = axes.iter().map(|ax| {
        let val = vars.iter()
            .find(|(tag, _)| *tag == ax.tag.to_bytes())
            .map(|(_, v)| *v)
            .unwrap_or(ax.def_value);
        // FT_Fixed is 16.16 fixed-point
        (val as f64 * 65536.0).round() as i64
    }).collect();

    // Call FT_Set_Var_Design_Coordinates via raw FFI
    let raw_face: freetype::freetype_sys::FT_Face = ft_face.raw() as *const _ as *mut _;
    unsafe {
        freetype::freetype_sys::FT_Set_Var_Design_Coordinates(
            raw_face,
            coords.len() as u32,
            coords.as_ptr() as *const freetype::freetype_sys::FT_Fixed,
        );
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct WordPlacement {
    pub text: String,
    pub x_off: u32,
    pub y_off: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

/// Verify a vectorised text region by:
/// 1. Computing an aspect-ratio penalty (natural advance widths vs OCR bbox widths)
/// 2. Rendering with per-word width scaling (so SSIM can compare glyph shapes)
/// 3. Computing windowed SSIM on the width-scaled render
/// 4. Returning `ssim * aspect_penalty`
pub fn verify_text_region(
    original_gray: &GrayImage,
    font_data: &[u8],
    _text: &str,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    words: &[TextRegion],
    overrides: Option<&[(char, u16)]>,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
    audit_dir: Option<&std::path::Path>,
    bail_below: Option<f32>,
) -> (f32, i32) {
    let (iw, ih) = original_gray.dimensions();
    let x = x.min(iw.saturating_sub(1));
    let y = y.min(ih.saturating_sub(1));
    let w = width.min(iw - x);
    let h = height.min(ih - y);

    if w < 3 || h < 3 {
        return (0.0, 0);
    }

    let full_scan = image::imageops::crop_imm(original_gray, x, y, w, h).to_image();

    // Page-level Hough deskew already corrected the full page before we get
    // here, so no per-line rotation needed.
    let placements: Vec<WordPlacement> = words
        .iter()
        .map(|wr| WordPlacement {
            text: wr.text.clone(),
            x_off: wr.x.saturating_sub(x),
            y_off: wr.y.saturating_sub(y),
            width: wr.width,
            height: wr.height,
            confidence: wr.confidence,
        })
        .collect();

    // Try multiple render scales and pick the best SSIM.
    let scales = vec![2, 4];

    // Both scan and render use the word-union bbox — the union of all
    // final (post-processed) word bboxes.  This matches what the report
    // draws as the "final" cyan-dashed boxes.

    let mut best_score = 0.0f32;
    let mut best_dy = 0i32;
    let mut best_scan_crop: Option<GrayImage> = None;
    let mut best_render_ink: Option<GrayImage> = None;
    let mut best_diff: Option<GrayImage> = None;

    for &scale in &scales {
        // Render into the expanded line bbox canvas
        let full_render = match render_via_freetype_scaled(font_data, &placements, w, h, scale, overrides, variant_tag, variations) {
            Some(r) => r,
            None => continue,
        };

        let scan_crop = full_scan.clone();
        let render_for_ssim = full_render.clone();

        let scan_blur = crate::ssim::gaussian_blur_3x3(&scan_crop);
        let render_blur = crate::ssim::gaussian_blur_3x3(&render_for_ssim);
        let (score, dy) = crate::ssim::ssim_windowed_best_vshift(&scan_blur, &render_blur, 12, bail_below);

        if score > best_score {
            best_score = score;
            best_dy = dy;
            best_scan_crop = Some(scan_crop);

            // Render for audit: ink-crop for clean display
            let ink_threshold = 240u8;
            let (rw, rh) = full_render.dimensions();
            let (r_top, r_bot) = crate::ocr::ink_vertical_extent(&full_render, 0, rw, 0, rh, ink_threshold);
            let ink_h = r_bot.saturating_sub(r_top);
            let render_ink = if ink_h >= 3 {
                image::imageops::crop_imm(&full_render, 0, r_top, rw, ink_h).to_image()
            } else {
                render_for_ssim
            };

            // Diff the displayed pair (compute_abs_diff resizes if heights differ)
            best_diff = Some(compute_abs_diff(
                best_scan_crop.as_ref().unwrap(), &render_ink,
            ));
            best_render_ink = Some(render_ink);
        }

        // Early exit: if 2× already gives a strong match, skip the
        // more expensive 4× render — the glyph shape is clearly right.
        if best_score >= 0.75 {
            break;
        }
    }

    // Save SSIM audit images.
    if let Some(audit_path) = audit_dir {
        if let Some(ref sc) = best_scan_crop {
            let _ = sc.save(audit_path.join("ssim_scan.png"));
        }
        if let Some(ref ri) = best_render_ink {
            let _ = ri.save(audit_path.join("ssim_render.png"));
        }
        if let Some(ref d) = best_diff {
            let _ = d.save(audit_path.join("ssim_diff.png"));
        }
    }

    (best_score, best_dy)
}

/// Compute absolute pixel difference between two grayscale images.
/// Images are resized to match dimensions if needed (using the scan crop size).
pub fn compute_abs_diff(a: &GrayImage, b: &GrayImage) -> GrayImage {
    let (aw, ah) = a.dimensions();
    let b_resized = if b.dimensions() != (aw, ah) {
        image::imageops::resize(b, aw, ah, image::imageops::FilterType::Lanczos3)
    } else {
        b.clone()
    };
    let mut diff = GrayImage::new(aw, ah);
    for y in 0..ah {
        for x in 0..aw {
            let pa = a.get_pixel(x, y).0[0] as i16;
            let pb = b_resized.get_pixel(x, y).0[0] as i16;
            diff.put_pixel(x, y, Luma([(pa - pb).unsigned_abs() as u8]));
        }
    }
    diff
}

// ---------------------------------------------------------------------------
// Word-by-word renderer — height-matched scale, natural advance width
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Word-by-word renderer — width-matched scale, natural height
// ---------------------------------------------------------------------------

fn render_via_freetype_scaled(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    render_scale: u32,
    _overrides: Option<&[(char, u16)]>,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
) -> Option<GrayImage> {
    // NOTE: ab_glyph fallback disabled.  width_matched_em_px (ab_glyph) does
    // not compensate for sidebearings, so it systematically underestimates font
    // size (~3% too small).  The shaped path (rustybuzz + FreeType) handles
    // sidebearing correction and should be the only render path.
    render_via_freetype(font_data, words, canvas_w, canvas_h, render_scale, variant_tag, variations)
}

/// Render text using rustybuzz (OT shaping) + FreeType (rasterisation).
/// Subpixel glyph positioning, no intermediate PDF, no subprocess.
fn render_via_freetype(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    render_scale: u32,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
) -> Option<GrayImage> {
    use std::cell::RefCell;

    thread_local! {
        static FT_LIB: RefCell<Option<freetype::Library>> = RefCell::new(None);
    }

    // Compute font size from ab_glyph (consistent with coarse scoring).
    // Compute per-word em_px using rustybuzz (shaped advances with
    // sidebearing correction).  No ab_glyph fallback — it lacks sidebearing
    // compensation and systematically underestimates font size by ~3%.
    let mut font_ref = FontRef::try_from_slice(font_data).ok()?;
    // Apply variable-font axis coordinates
    if let Some(vars) = variations {
        use ab_glyph::VariableFont;
        for (tag, val) in vars {
            font_ref.set_variation(tag, *val);
        }
    }
    let mut all_em: Vec<f32> = words.iter()
        .filter(|w| !w.text.is_empty() && w.width >= 1)
        .filter_map(|w| {
            crate::layout::width_matched_em_px_shaped(font_data, &w.text, w.width as f32, variant_tag, variations)
        })
        .collect();
    if all_em.is_empty() {
        return None;
    }
    all_em.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let line_em_px = all_em[all_em.len() / 2];

    // Pad the render canvas so italic overshoot / wide terminal strokes
    // aren't clipped.  After rendering, we trim back to the scan width.
    let render_pad: u32 = 20;
    let render_w = (canvas_w + render_pad) * render_scale;
    let final_w = canvas_w * render_scale;  // target width after trim
    let render_em = line_em_px * render_scale as f32;

    // The caller's canvas_h is the OCR-expanded bbox — sized to the scan's
    // ink extent.  The rendered font's metric height (ascent − descent) may
    // exceed that, so use whichever is larger to avoid vertical clipping.
    // Add vertical padding so glyphs whose descenders exceed the font's
    // declared descent (e.g. Q tail, swash g) aren't clipped.
    let vert_pad: u32 = 10 * render_scale;
    let sf2 = font_ref.as_scaled(PxScale::from(render_em));
    let ink_h2 = sf2.ascent() - sf2.descent();
    let min_render_h = canvas_h * render_scale;
    let render_h = min_render_h.max(ink_h2.ceil() as u32) + vert_pad;

    // Baseline centred in the (possibly taller) canvas
    let baseline_y = ((render_h as f32 - ink_h2) / 2.0 + sf2.ascent()) as f64;

    // Set up FreeType (reuse thread-local library)
    let ft_result: Option<GrayImage> = FT_LIB.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = freetype::Library::init().ok();
        }
        let lib = borrow.as_ref()?;
        let ft_face = lib.new_memory_face2(font_data.to_vec(), 0).ok()?;
        let size_26_6 = (render_em as f64 * 64.0) as isize;
        ft_face.set_char_size(size_26_6, size_26_6, 72, 72).ok()?;

        // Apply variable-font axis coordinates to FreeType face
        if let Some(vars) = variations {
            set_ft_variations(&ft_face, font_data, vars);
        }

    // Set up rustybuzz for shaping
    let mut buzz_face = rustybuzz::Face::from_slice(font_data, 0)?;
    if let Some(vars) = variations {
        for (tag, val) in vars {
            let t = rustybuzz::ttf_parser::Tag::from_bytes(tag);
            buzz_face.set_variation(t, *val);
        }
    }
    let ot_features = crate::layout::ot_features(variant_tag);
    let units_per_em = buzz_face.units_per_em() as f64;
    let px_per_unit = render_em as f64 / units_per_em;

    let mut canvas = GrayImage::from_pixel(render_w, render_h, Luma([255u8]));

    for word in words {
        if word.text.is_empty() || word.width < 1 {
            continue;
        }

        // Shape with shared helper
        let shaped = match crate::layout::shape_word(&buzz_face, &ot_features, &word.text) {
            Some(s) => s,
            None => continue,
        };

        // Walk glyphs, accumulating pen position in subpixel floats
        // Start at ink edge: subtract first glyph's LSB (in pixels) so
        // rendered ink aligns with the OCR bbox edge.
        let lsb_px = shaped.first_lsb_fu * px_per_unit;
        let mut pen_x = word.x_off as f64 * render_scale as f64 - lsb_px;
        let pen_y = baseline_y;

        for i in 0..shaped.glyph_ids.len() {
            let glyph_id = shaped.glyph_ids[i];
            let x_offset = shaped.x_offsets[i] as f64 * px_per_unit;
            let y_offset = shaped.y_offsets[i] as f64 * px_per_unit;

            // Load glyph in FreeType
            ft_face.load_glyph(glyph_id, freetype::face::LoadFlag::RENDER | freetype::face::LoadFlag::NO_HINTING).ok()?;
            let glyph = ft_face.glyph();
            let bitmap = glyph.bitmap();
            let bmp_w = bitmap.width() as usize;
            let bmp_h = bitmap.rows() as usize;
            let bmp_buf = bitmap.buffer();
            let bmp_pitch = bitmap.pitch().unsigned_abs() as usize;

            if bmp_w == 0 || bmp_h == 0 || bmp_buf.is_empty() {
                pen_x += shaped.x_advances[i] as f64 * px_per_unit;
                continue;
            }

            // Glyph bitmap origin: (pen_x + x_offset + bitmap_left, pen_y - y_offset - bitmap_top)
            let blit_x = (pen_x + x_offset + glyph.bitmap_left() as f64).round() as i32;
            let blit_y = (pen_y - y_offset - glyph.bitmap_top() as f64).round() as i32;

            // Blit the glyph bitmap onto the canvas
            for row in 0..bmp_h {
                for col in 0..bmp_w {
                    let cx = blit_x + col as i32;
                    let cy = blit_y + row as i32;
                    if cx < 0 || cy < 0 || cx >= render_w as i32 || cy >= render_h as i32 {
                        continue;
                    }
                    let alpha = bmp_buf[row * bmp_pitch + col] as f32 / 255.0;
                    if alpha < 0.01 {
                        continue;
                    }
                    let existing = canvas.get_pixel(cx as u32, cy as u32).0[0] as f32;
                    let blended = existing * (1.0 - alpha); // black ink on white
                    canvas.put_pixel(cx as u32, cy as u32, Luma([blended as u8]));
                }
            }

            pen_x += shaped.x_advances[i] as f64 * px_per_unit;
        }
    }

    // Measure rendered ink extent and correct for advance-vs-ink mismatch.
    // width_matched_em_px matches advance width to target, but the OCR bbox
    // is ink extent (excluding sidebearings). Resize to match.
    let target_ink_w = words.iter().map(|w| w.x_off as u32 + w.width).max().unwrap_or(canvas_w);
    let rend_ink_right = {
        let mut right = 0u32;
        for x in (0..canvas.width()).rev() {
            if (0..canvas.height()).any(|y| canvas.get_pixel(x, y).0[0] < 240) {
                right = x + 1;
                break;
            }
        }
        right
    };
    let rend_ink_left = {
        let mut left = canvas.width();
        for x in 0..canvas.width() {
            if (0..canvas.height()).any(|y| canvas.get_pixel(x, y).0[0] < 240) {
                left = x;
                break;
            }
        }
        left
    };
    // Scale canvas horizontally so rendered ink extent matches scan ink extent (the OCR bbox).
    // Only apply if there's a meaningful difference (>1px) and rendered ink is non-empty.
    let rendered_ink_w = rend_ink_right.saturating_sub(rend_ink_left);
    let target_ink_w_rs = target_ink_w * render_scale;
    if rendered_ink_w > 0 && rendered_ink_w.abs_diff(target_ink_w_rs) > render_scale {
        let scale_x = target_ink_w_rs as f64 / rendered_ink_w as f64;
        let new_w = (canvas.width() as f64 * scale_x).round() as u32;
        canvas = image::imageops::resize(&canvas, new_w, canvas.height(), image::imageops::FilterType::Lanczos3);
    }

    // Trim render padding: crop (not resize!) back to the scan bbox width.
    // The extra padding was only to avoid clipping during glyph rasterisation.
    let trim_w = final_w.min(canvas.width());
    canvas = image::imageops::crop_imm(&canvas, 0, 0, trim_w, canvas.height()).to_image();

    // Downsample from render resolution to 1x (keep expanded height if canvas was grown)
    if render_scale > 1 {
        let out_h = render_h / render_scale;
        Some(image::imageops::resize(&canvas, canvas_w, out_h, image::imageops::FilterType::Lanczos3))
    } else {
        Some(canvas)
    }
    }); // end FT_LIB.with

    ft_result
}

/// Fallback: ab_glyph-based rendering (no OT shaping, legacy kern only).
/// DISABLED: lacks sidebearing compensation, systematically underestimates
/// font size by ~3%.  Kept for reference only.
#[allow(dead_code)]
fn render_words_ab_glyph(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    overrides: Option<&[(char, u16)]>,
) -> Option<GrayImage> {
    use ab_glyph::point;
    let font = FontRef::try_from_slice(font_data).ok()?;
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));

    let (cw, ch) = canvas.dimensions();

    let mut all_em: Vec<f32> = words.iter()
        .filter(|w| !w.text.is_empty() && w.width >= 1)
        .filter_map(|w| crate::layout::width_matched_em_px(&font, &w.text, w.width as f32, overrides))
        .collect();
    if all_em.is_empty() {
        return Some(canvas);
    }
    all_em.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let line_em_px = all_em[all_em.len() / 2];
    let line_scale = PxScale::from(line_em_px);
    let sf_line = font.as_scaled(line_scale);
    let line_baseline = crate::layout::ink_centered_baseline_px(&font, line_em_px, canvas_h as f32);

    for word in words {
        if word.text.is_empty() || word.width < 1 {
            continue;
        }

        let natural_adv = {
            let mut adv = 0.0f32;
            let mut prev: Option<ab_glyph::GlyphId> = None;
            for c in word.text.chars() {
                let gid = crate::char_index::resolve_glyph(&font, c, overrides);
                if let Some(p) = prev {
                    adv += sf_line.kern(p, gid);
                }
                adv += sf_line.h_advance(gid);
                prev = Some(gid);
            }
            adv
        };
        let h_scale = if natural_adv > 0.1 { word.width as f32 / natural_adv } else { 1.0 };

        let mut cx = word.x_off as f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;

        for c in word.text.chars() {
            let gid = crate::char_index::resolve_glyph(&font, c, overrides);
            if let Some(p) = prev {
                cx += sf_line.kern(p, gid) * h_scale;
            }
            let glyph = gid.with_scale_and_position(line_scale, point(cx, line_baseline));
            if let Some(og) = font.outline_glyph(glyph) {
                let bounds = og.px_bounds();
                let bx = bounds.min.x as i32;
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
            cx += sf_line.h_advance(gid) * h_scale;
            prev = Some(gid);
        }
    }

    Some(canvas)
}

// ---------------------------------------------------------------------------
// Public: render a line for visual comparison in the miss report
// ---------------------------------------------------------------------------

/// Render a line of text for visual comparison.
/// Returns the ink-cropped render image, or None if rendering fails.
pub fn render_line_for_comparison(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    overrides: Option<&[(char, u16)]>,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
) -> Option<GrayImage> {
    let rendered = render_via_freetype_scaled(
        font_data, words, canvas_w, canvas_h, 2, overrides, variant_tag, variations,
    )?;

    // Ink-crop vertically for clean display
    let ink_threshold = 240u8;
    let (rw, rh) = rendered.dimensions();
    let (r_top, r_bot) = crate::ocr::ink_vertical_extent(&rendered, 0, rw, 0, rh, ink_threshold);
    let ink_h = r_bot.saturating_sub(r_top);
    if ink_h >= 3 {
        Some(image::imageops::crop_imm(&rendered, 0, r_top, rw, ink_h).to_image())
    } else {
        Some(rendered)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
