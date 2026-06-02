//! SSIM verification — render vector text back to raster and compare with the
//! original to catch bad replacements before they make it into the output.
//!
//! v5: width-scaled SSIM for glyph-shape comparison + aspect-ratio penalty
//! for catching fonts whose proportions don't match the original.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma};
use crate::ocr::TextRegion;

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
    audit_dir: Option<&std::path::Path>,
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
    let scales = if let Ok(s) = std::env::var("UNSCAN_RENDER_SCALE") {
        vec![s.parse::<u32>().unwrap_or(2)]
    } else {
        vec![2, 4]
    };

    // Ink-crop the render to where the glyphs actually are; crop the scan to
    // the word-union bbox (tight to OCR word bounds, no adjacent-line bleed).
    // SSIM scores word-union crops of both (vshift handles baseline offset);
    // audit render image uses the tighter ink crop for visual clarity.
    let word_top = placements.iter().map(|p| p.y_off).min().unwrap_or(0);
    let word_bot = placements.iter().map(|p| p.y_off + p.height).max().unwrap_or(h);
    let word_h = word_bot.saturating_sub(word_top).min(h - word_top);

    let mut best_score = 0.0f32;
    let mut best_dy = 0i32;
    let mut best_scan_crop: Option<GrayImage> = None;
    let mut best_render_ink: Option<GrayImage> = None;
    let mut best_diff: Option<GrayImage> = None;

    for &scale in &scales {
        // Render into the full line bbox canvas (word placements are relative to it)
        let full_render = match render_via_freetype_scaled(font_data, &placements, w, h, scale, overrides, variant_tag) {
            Some(r) => r,
            None => continue,
        };

        // Scan: word-union bbox crop (no adjacent-line bleed)
        let scan_crop = if word_h >= 3 && word_h < h {
            image::imageops::crop_imm(&full_scan, 0, word_top, w, word_h).to_image()
        } else {
            full_scan.clone()
        };

        // Render for SSIM: same word-union region (vshift handles baseline offset)
        let render_for_ssim = if word_h >= 3 && word_h < h {
            image::imageops::crop_imm(&full_render, 0, word_top, w, word_h).to_image()
        } else {
            full_render.clone()
        };

        let scan_blur = gaussian_blur_3x3(&scan_crop);
        let render_blur = gaussian_blur_3x3(&render_for_ssim);
        let (score, dy) = ssim_windowed_best_vshift(&scan_blur, &render_blur, 12);

        if std::env::var("UNSCAN_DUMP_SSIM").is_ok() {
            log::info!("SSIM debug: scan ({}x{}) render ({}x{}) scale={} dy={} score={:.4}",
                scan_crop.width(), scan_crop.height(),
                render_for_ssim.width(), render_for_ssim.height(),
                scale, dy, score);
        }

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
fn compute_abs_diff(a: &GrayImage, b: &GrayImage) -> GrayImage {
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
    overrides: Option<&[(char, u16)]>,
    variant_tag: &str,
) -> Option<GrayImage> {
    // Try PDF-based rendering first (proper OT shaping via the PDF renderer).
    // Falls back to ab_glyph if PDF rendering fails.
    if let Some(img) = render_via_freetype(font_data, words, canvas_w, canvas_h, render_scale, variant_tag) {
        return Some(img);
    }
    if render_scale == 2 {
        // Only fall back for the default scale
        log::warn!("FreeType rendering failed, falling back to ab_glyph for SSIM");
        render_words_ab_glyph(font_data, words, canvas_w, canvas_h, overrides)
    } else {
        None
    }
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
) -> Option<GrayImage> {
    use std::cell::RefCell;

    thread_local! {
        static FT_LIB: RefCell<Option<freetype::Library>> = RefCell::new(None);
    }

    // Compute font size from ab_glyph (consistent with coarse scoring).
    let font_ref = FontRef::try_from_slice(font_data).ok()?;
    let mut all_em: Vec<f32> = words.iter()
        .filter(|w| !w.text.is_empty() && w.width >= 1)
        .filter_map(|w| {
            // Prefer rustybuzz-shaped advance for consistency with FreeType rendering
            crate::layout::width_matched_em_px_shaped(font_data, &w.text, w.width as f32, variant_tag)
                .or_else(|| crate::layout::width_matched_em_px(&font_ref, &w.text, w.width as f32, None))
        })
        .collect();
    if all_em.is_empty() {
        return None;
    }
    all_em.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let line_em_px = all_em[all_em.len() / 2];
    if std::env::var("UNSCAN_DUMP_SSIM").is_ok() {
        log::info!("render: line_em_px={:.2}, canvas={}x{}, scale={}", line_em_px, canvas_w, canvas_h, render_scale);
    }

    let render_w = canvas_w * render_scale;
    let render_h = canvas_h * render_scale;
    let render_em = line_em_px * render_scale as f32;

    // Baseline for the 2× canvas
    let sf2 = font_ref.as_scaled(PxScale::from(render_em));
    let ink_h2 = sf2.ascent() - sf2.descent();
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

    // Set up rustybuzz for shaping
    let buzz_face = rustybuzz::Face::from_slice(font_data, 0)?;
    let units_per_em = buzz_face.units_per_em() as f64;
    let px_per_unit = render_em as f64 / units_per_em;

    // OT variant features for shaping (e.g. smcp, onum)
    let ot_features: Vec<rustybuzz::Feature> = if !variant_tag.is_empty() && variant_tag.len() <= 4 {
        let mut tag_bytes = [b' '; 4];
        for (i, b) in variant_tag.as_bytes().iter().enumerate().take(4) {
            tag_bytes[i] = *b;
        }
        let tag = rustybuzz::ttf_parser::Tag::from_bytes(&tag_bytes);
        vec![rustybuzz::Feature::new(tag, 1, ..)]
    } else {
        vec![]
    };

    let mut canvas = GrayImage::from_pixel(render_w, render_h, Luma([255u8]));

    for word in words {
        if word.text.is_empty() || word.width < 1 {
            continue;
        }

        // Shape with rustybuzz
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(&word.text);
        let glyphs = rustybuzz::shape(&buzz_face, &ot_features, buffer);
        let infos = glyphs.glyph_infos();
        let positions = glyphs.glyph_positions();

        // Walk glyphs, accumulating pen position in subpixel floats
        let mut pen_x = word.x_off as f64 * render_scale as f64;
        let pen_y = baseline_y;

        // Compensate for first glyph's left side bearing so ink aligns
        // with the OCR bbox edge (which is ink-extent, not advance-extent).
        let mut lsb_compensated = false;

        for (info, pos) in infos.iter().zip(positions.iter()) {
            let glyph_id = info.glyph_id; // after shaping = glyph ID
            let x_offset = pos.x_offset as f64 * px_per_unit;
            let y_offset = pos.y_offset as f64 * px_per_unit;

            // Load glyph in FreeType
            ft_face.load_glyph(glyph_id, freetype::face::LoadFlag::RENDER | freetype::face::LoadFlag::NO_HINTING).ok()?;
            let glyph = ft_face.glyph();
            let bitmap = glyph.bitmap();
            let bmp_w = bitmap.width() as usize;
            let bmp_h = bitmap.rows() as usize;
            let bmp_buf = bitmap.buffer();
            let bmp_pitch = bitmap.pitch().unsigned_abs() as usize;

            if bmp_w == 0 || bmp_h == 0 || bmp_buf.is_empty() {
                pen_x += pos.x_advance as f64 * px_per_unit;
                continue;
            }

            // Shift pen left by first glyph's bitmap_left so ink starts at crop edge
            if !lsb_compensated {
                pen_x -= glyph.bitmap_left() as f64;
                lsb_compensated = true;
            }

            if bmp_w == 0 || bmp_h == 0 || bmp_buf.is_empty() {
                pen_x += pos.x_advance as f64 * px_per_unit;
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

            pen_x += pos.x_advance as f64 * px_per_unit;
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
        // Crop or pad back to render_w
        if canvas.width() > render_w {
            canvas = image::imageops::crop_imm(&canvas, 0, 0, render_w, canvas.height()).to_image();
        } else if canvas.width() < render_w {
            let mut padded = GrayImage::from_pixel(render_w, canvas.height(), Luma([255u8]));
            image::imageops::overlay(&mut padded, &canvas, 0, 0);
            canvas = padded;
        }
    }

    // Downsample from render resolution to canvas size
    if render_scale > 1 {
        Some(image::imageops::resize(&canvas, canvas_w, canvas_h, image::imageops::FilterType::Lanczos3))
    } else {
        Some(canvas)
    }
    }); // end FT_LIB.with

    ft_result
}

/// Fallback: ab_glyph-based rendering (no OT shaping, legacy kern only).
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
// Windowed SSIM (11×11 Gaussian-weighted, ink-aware)
// ---------------------------------------------------------------------------

/// Precomputed 11×11 Gaussian kernel with sigma ≈ 1.5.
/// Values sum to 1.0.
fn gaussian_kernel_11x11() -> [[f64; 11]; 11] {
    const SIGMA: f64 = 1.5;
    let mut kernel = [[0.0f64; 11]; 11];
    let mut sum = 0.0f64;
    for iy in 0..11 {
        for ix in 0..11 {
            let dx = ix as f64 - 5.0;
            let dy = iy as f64 - 5.0;
            let v = (-0.5 * (dx * dx + dy * dy) / (SIGMA * SIGMA)).exp();
            kernel[iy][ix] = v;
            sum += v;
        }
    }
    // Normalise
    for row in &mut kernel {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    kernel
}

/// Windowed SSIM on grayscale images.
///
/// - 11×11 Gaussian-weighted windows
/// - Only windows containing ink (pixels < 240 in either image) contribute
/// - Returns mean SSIM over ink-containing windows, or 0.0 if none
/// Try vertical shifts of the rendered image from -max_shift to +max_shift pixels
/// 3×3 Gaussian blur with σ≈0.7 (kernel [1,2,1]/4 separable).
fn gaussian_blur_3x3(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return img.clone();
    }
    let mut tmp = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = |dx: i32| -> u32 {
                let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                img.get_pixel(xx, y).0[0] as u32
            };
            let v = p(-1) + 2 * p(0) + p(1);
            tmp.put_pixel(x, y, Luma([((v + 2) / 4) as u8]));
        }
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = |dy: i32| -> u32 {
                let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                tmp.get_pixel(x, yy).0[0] as u32
            };
            let v = p(-1) + 2 * p(0) + p(1);
            out.put_pixel(x, y, Luma([((v + 2) / 4) as u8]));
        }
    }
    out
}


/// and return the best (highest) SSIM and the shift that produced it.
/// Positive dy = rendered image moved DOWN.
/// Searches from center outward (0, -1, 1, -2, 2, …) and exits early if SSIM ≥ 0.92.
fn ssim_windowed_best_vshift(a: &GrayImage, b: &GrayImage, max_shift: i32) -> (f32, i32) {
    const EARLY_EXIT_THRESHOLD: f32 = 0.92;
    let mut best = 0.0f32;
    let mut best_dy = 0i32;

    // Search center-outward: 0, -1, 1, -2, 2, …
    let mut shifts = Vec::with_capacity((2 * max_shift + 1) as usize);
    shifts.push(0i32);
    for d in 1..=max_shift {
        shifts.push(-d);
        shifts.push(d);
    }

    for dy in shifts {
        let score = ssim_windowed(a, b, dy);
        if score > best {
            best = score;
            best_dy = dy;
            if best >= EARLY_EXIT_THRESHOLD {
                break;
            }
        }
    }
    (best, best_dy)
}

fn ssim_windowed(a: &GrayImage, b: &GrayImage, b_dy: i32) -> f32 {
    let (w, h) = a.dimensions();
    if w < 11 || h < 11 {
        // Fallback to global for tiny images (shift not applied in global path)
        return ssim_global(a, b);
    }

    let kernel = gaussian_kernel_11x11();
    let c1: f64 = (0.01 * 255.0_f64).powi(2);
    let c2: f64 = (0.03 * 255.0_f64).powi(2);

    // Ink threshold: a pixel is "ink" if its value < 240
    const INK_THRESHOLD: u8 = 240;
    // Minimum number of ink pixels in a window to count it
    const MIN_INK_PIXELS: u32 = 3;

    let half = 5i32; // 11/2
    let bw = b.width() as i32;
    let bh = b.height() as i32;

    let mut ssim_sum = 0.0f64;
    let mut window_count = 0u64;

    // Step by 4 pixels for speed (still plenty of overlap at 11×11)
    let step = 4u32;

    let mut cy = half as u32;
    while cy + (half as u32) < h {
        let mut cx = half as u32;
        while cx + (half as u32) < w {
            // Single pass: accumulate ink count, weighted means, and weighted
            // squared/cross terms simultaneously.
            let mut ink_count = 0u32;
            let mut mu_a = 0.0f64;
            let mut mu_b = 0.0f64;
            let mut sum_wa2 = 0.0f64;
            let mut sum_wb2 = 0.0f64;
            let mut sum_wab = 0.0f64;

            for ky in 0..11u32 {
                let py = (cy as i32 - half + ky as i32) as u32;
                // b pixel y with shift
                let by = py as i32 + b_dy;
                for kx in 0..11u32 {
                    let px = (cx as i32 - half + kx as i32) as u32;
                    let va_u8 = a.get_pixel(px, py).0[0];
                    // Read b with offset; out-of-bounds → 255 (white background)
                    let vb_u8 = if by >= 0 && by < bh && (px as i32) < bw {
                        b.get_pixel(px, by as u32).0[0]
                    } else {
                        255u8
                    };

                    if va_u8 < INK_THRESHOLD || vb_u8 < INK_THRESHOLD {
                        ink_count += 1;
                    }

                    let wt = kernel[ky as usize][kx as usize];
                    let va = va_u8 as f64;
                    let vb = vb_u8 as f64;
                    mu_a += wt * va;
                    mu_b += wt * vb;
                    sum_wa2 += wt * va * va;
                    sum_wb2 += wt * vb * vb;
                    sum_wab += wt * va * vb;
                }
            }

            if ink_count >= MIN_INK_PIXELS {
                // One-pass variance: sig2 = E[x²] - (E[x])²
                let sig_a2 = sum_wa2 - mu_a * mu_a;
                let sig_b2 = sum_wb2 - mu_b * mu_b;
                let sig_ab = sum_wab - mu_a * mu_b;

                let num = (2.0 * mu_a * mu_b + c1) * (2.0 * sig_ab + c2);
                let den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a2 + sig_b2 + c2);
                let local_ssim = if den < 1e-10 { 1.0 } else { num / den };

                ssim_sum += local_ssim;
                window_count += 1;
            }

            cx += step;
        }
        cy += step;
    }

    if window_count == 0 {
        // No ink windows found — fall back to global
        return ssim_global(a, b);
    }

    (ssim_sum / window_count as f64).clamp(0.0, 1.0) as f32
}

/// Fallback global SSIM for very small images (< 11×11).
fn ssim_global(a: &GrayImage, b: &GrayImage) -> f32 {
    let c1: f64 = (0.01 * 255.0_f64).powi(2);
    let c2: f64 = (0.03 * 255.0_f64).powi(2);
    let n = (a.width() as u64 * a.height() as u64) as f64;
    if n == 0.0 { return 0.0; }

    let (mut sa, mut sb, mut sa2, mut sb2, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let va = pa.0[0] as f64;
        let vb = pb.0[0] as f64;
        sa += va; sb += vb;
        sa2 += va * va; sb2 += vb * vb;
        sab += va * vb;
    }

    let mu_a = sa / n;
    let mu_b = sb / n;
    let sig_a2 = (sa2 / n) - mu_a * mu_a;
    let sig_b2 = (sb2 / n) - mu_b * mu_b;
    let sig_ab = (sab / n) - mu_a * mu_b;

    let num = (2.0 * mu_a * mu_b + c1) * (2.0 * sig_ab + c2);
    let den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a2 + sig_b2 + c2);
    if den < 1e-10 { return 0.0; }
    (num / den).clamp(0.0, 1.0) as f32
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
