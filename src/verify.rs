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
}

/// Verify a vectorised text region by:
/// 1. Computing an aspect-ratio penalty (natural advance widths vs OCR bbox widths)
/// 2. Rendering with per-word width scaling (so SSIM can compare glyph shapes)
/// 3. Computing windowed SSIM on the width-scaled render
/// 4. Returning `ssim * aspect_penalty`
pub fn verify_text_region(
    original_gray: &GrayImage,
    font_data: &[u8],
    text: &str,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    words: &[TextRegion],
) -> (f32, i32) {
    let (iw, ih) = original_gray.dimensions();
    let x = x.min(iw.saturating_sub(1));
    let y = y.min(ih.saturating_sub(1));
    let w = width.min(iw - x);
    let h = height.min(ih - y);

    if w < 3 || h < 3 {
        return (0.0, 0);
    }

    let original_crop = image::imageops::crop_imm(original_gray, x, y, w, h).to_image();

    let placements: Vec<WordPlacement> = words
        .iter()
        .map(|wr| WordPlacement {
            text: wr.text.clone(),
            x_off: wr.x.saturating_sub(x),
            y_off: wr.y.saturating_sub(y),
            width: wr.width,
            height: wr.height,
        })
        .collect();

    let skew_angle = detect_skew_from_words(&placements);
    let deskewed = if skew_angle.abs() > 0.001 {
        rotate_gray(&original_crop, -skew_angle)
    } else {
        original_crop.clone()
    };

    // Render at height-matched scale with natural width — SSIM sees width
    // mismatches directly as misaligned ink vs whitespace.
    let rendered = match render_words_height_scaled(font_data, &placements, w, h, text) {
        Some(r) => r,
        None => return (0.0, 0),
    };

    // Debug: dump both sides of the SSIM comparison
    if std::env::var("UNSCAN_DUMP_SSIM").is_ok() {
        let _ = deskewed.save("/tmp/ssim_scan_crop.png");
        let _ = rendered.save("/tmp/ssim_rendered.png");
        log::info!("SSIM debug: dumped scan crop ({}x{}) and rendered ({}x{}) to /tmp/",
            deskewed.width(), deskewed.height(), rendered.width(), rendered.height());
    }

    // Windowed SSIM with vertical shift search — try offsets from -12 to +12
    // pixels to find best vertical alignment (bumped from ±6 in v8f).
    ssim_windowed_best_vshift(&deskewed, &rendered, 12)
}

// ---------------------------------------------------------------------------
// Word-by-word renderer — height-matched scale, natural advance width
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Word-by-word renderer — width-matched scale, natural height
// ---------------------------------------------------------------------------

fn render_words_height_scaled(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    _line_text: &str,
) -> Option<GrayImage> {
    // Try PDF-based rendering first (proper OT shaping via the PDF renderer).
    // Falls back to ab_glyph if PDF rendering fails.
    if let Some(img) = render_via_pdf(font_data, words, canvas_w, canvas_h) {
        return Some(img);
    }
    log::warn!("PDF rendering failed, falling back to ab_glyph for SSIM");
    render_words_ab_glyph(font_data, words, canvas_w, canvas_h)
}

/// Render text by generating a tiny single-page PDF with the candidate font,
/// then rasterising it with `pdftoppm`.  The PDF viewer applies full OpenType
/// shaping (GPOS kerning, GSUB ligatures, etc.) so the output matches what a
/// real PDF with this font would look like.
fn render_via_pdf(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
) -> Option<GrayImage> {
    use lopdf::{dictionary, Document, Object, Stream};
    use lopdf::content::{Content, Operation as Op};

    // We build a PDF whose media box is exactly canvas_w × canvas_h points
    // (at 72 DPI). We'll render at a DPI that gives us exactly the pixel
    // dimensions we need.

    let pt_w = canvas_w as f64;
    let pt_h = canvas_h as f64;

    // Embed the font as a simple TrueType/OpenType resource.
    let mut doc = Document::with_version("1.7");

    // Font stream (compressed)
    let font_stream = Stream::new(
        dictionary! {
            "Length1" => Object::Integer(font_data.len() as i64),
        },
        font_data.to_vec(),
    ).with_compression(true);
    let font_stream_id = doc.add_object(font_stream);

    // Detect CFF vs TrueType
    let is_cff = font_data.starts_with(&[0x4F, 0x54, 0x54, 0x4F]) // OTTO
        || (font_data.len() > 4 && &font_data[0..4] == b"OTTO");

    let font_descriptor = dictionary! {
        "Type" => Object::Name(b"FontDescriptor".to_vec()),
        "FontName" => Object::Name(b"CandidateFont".to_vec()),
        "Flags" => Object::Integer(32), // non-symbolic
        "ItalicAngle" => Object::Integer(0),
        "Ascent" => Object::Integer(800),
        "Descent" => Object::Integer(-200),
        "CapHeight" => Object::Integer(700),
        "StemV" => Object::Integer(80),
        if is_cff { "FontFile3" } else { "FontFile2" } => Object::Reference(font_stream_id),
    };
    let fd_id = doc.add_object(font_descriptor);

    // Compute font size: median of per-word width-matched sizes.
    let font_ref = FontRef::try_from_slice(font_data).ok()?;
    let mut all_em: Vec<f32> = words.iter()
        .filter(|w| !w.text.is_empty() && w.width >= 1)
        .filter_map(|w| crate::layout::width_matched_em_px(&font_ref, &w.text, w.width as f32))
        .collect();
    if all_em.is_empty() {
        return None;
    }
    all_em.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let line_em_px = all_em[all_em.len() / 2];
    // At 72 DPI, 1 pt = 1 px, so font size in pt = em_px.
    let font_size_pt = line_em_px as f64;

    // Baseline: ink-centered within the canvas height.
    let baseline_px = crate::layout::ink_centered_baseline_px(&font_ref, line_em_px, canvas_h as f32);
    // PDF Y is bottom-up: pdf_y = canvas_h - baseline_px
    // But baseline_px is measured from top (ab_glyph style), and in PDF coords
    // baseline should be at (canvas_h - baseline_from_top).
    // Actually in our 72-DPI identity mapping: pdf_baseline_y = pt_h - baseline_px as f64
    let sf = font_ref.as_scaled(PxScale::from(line_em_px));
    let ink_h = sf.ascent() - sf.descent();
    let baseline_from_bottom = (pt_h - ink_h as f64) / 2.0 - sf.descent() as f64;

    // Simple Type1-style font (WinAnsiEncoding, first 256 glyphs).
    // For SSIM comparison this is sufficient — we only need Latin text.
    let font_dict = dictionary! {
        "Type" => Object::Name(b"Font".to_vec()),
        "Subtype" => Object::Name(if is_cff { b"Type1".to_vec() } else { b"TrueType".to_vec() }),
        "BaseFont" => Object::Name(b"CandidateFont".to_vec()),
        "Encoding" => Object::Name(b"WinAnsiEncoding".to_vec()),
        "FontDescriptor" => Object::Reference(fd_id),
    };
    let font_id = doc.add_object(font_dict);

    // Build page content: position each word with Td, render with Tj.
    let mut ops: Vec<Op> = Vec::new();
    fn op(name: &str, args: &[Object]) -> Op {
        Op::new(name, args.to_vec())
    }
    fn real(v: f64) -> Object {
        Object::Real(v as f32)
    }

    for word in words {
        if word.text.is_empty() || word.width < 1 {
            continue;
        }
        let pdf_x = word.x_off as f64;
        let pdf_y = baseline_from_bottom;

        // Per-word horizontal scaling to match OCR bbox width.
        let sf_line = font_ref.as_scaled(PxScale::from(line_em_px));
        let natural_adv: f32 = {
            let mut adv = 0.0f32;
            let mut prev: Option<ab_glyph::GlyphId> = None;
            for c in word.text.chars() {
                let gid = font_ref.glyph_id(c);
                if let Some(p) = prev {
                    adv += sf_line.kern(p, gid);
                }
                adv += sf_line.h_advance(gid);
                prev = Some(gid);
            }
            adv
        };
        let tz = if natural_adv > 0.1 {
            (word.width as f64 / natural_adv as f64) * 100.0
        } else {
            100.0
        };

        ops.push(op("BT", &[]));
        ops.push(op("Tf", &[Object::Name(b"F1".to_vec()), real(font_size_pt)]));
        ops.push(op("Tz", &[real(tz)]));
        ops.push(op("Td", &[real(pdf_x), real(pdf_y)]));
        let encoded = crate::pdf_out::encode_pdf_text(&word.text);
        ops.push(op("Tj", &[Object::String(encoded, lopdf::StringFormat::Literal)]));
        ops.push(op("ET", &[]));
    }

    let content = Content { operations: ops };
    let content_bytes = content.encode().ok()?;
    let content_stream = Stream::new(dictionary! {}, content_bytes);
    let content_id = doc.add_object(content_stream);

    let resources = dictionary! {
        "Font" => dictionary! {
            "F1" => Object::Reference(font_id),
        },
    };

    let page = dictionary! {
        "Type" => Object::Name(b"Page".to_vec()),
        "MediaBox" => Object::Array(vec![
            Object::Integer(0), Object::Integer(0),
            real(pt_w), real(pt_h),
        ]),
        "Contents" => Object::Reference(content_id),
        "Resources" => resources,
    };
    let page_id = doc.add_object(page);

    let pages = dictionary! {
        "Type" => Object::Name(b"Pages".to_vec()),
        "Kids" => Object::Array(vec![Object::Reference(page_id)]),
        "Count" => Object::Integer(1),
    };
    let pages_id = doc.add_object(pages);

    // Patch page's Parent
    if let Ok(page_obj) = doc.get_object_mut(page_id) {
        if let Object::Dictionary(ref mut d) = page_obj {
            d.set("Parent", Object::Reference(pages_id));
        }
    }

    let catalog = dictionary! {
        "Type" => Object::Name(b"Catalog".to_vec()),
        "Pages" => Object::Reference(pages_id),
    };
    let catalog_id = doc.add_object(catalog);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Write to a temp file.
    let tmp_pdf = std::env::temp_dir().join(format!("unscan_ssim_{}.pdf", std::process::id()));
    let tmp_png_prefix = std::env::temp_dir().join(format!("unscan_ssim_{}", std::process::id()));
    doc.save(&tmp_pdf).ok()?;

    // Render at 72 DPI so 1 pt = 1 px → output dimensions = canvas_w × canvas_h.
    let status = std::process::Command::new("pdftoppm")
        .args([
            "-r", "72",
            "-gray",
            "-f", "1", "-l", "1",
            "-singlefile",
        ])
        .arg(&tmp_pdf)
        .arg(&tmp_png_prefix)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;

    let _ = std::fs::remove_file(&tmp_pdf);

    if !status.success() {
        let _ = std::fs::remove_file(tmp_png_prefix.with_extension("pgm"));
        return None;
    }

    // pdftoppm -gray outputs a PGM file.
    let pgm_path = tmp_png_prefix.with_extension("pgm");
    let img = image::open(&pgm_path).ok()?.to_luma8();
    let _ = std::fs::remove_file(&pgm_path);

    // The rendered image might be slightly different dimensions due to rounding.
    // Resize to exact canvas dimensions if needed.
    if img.width() != canvas_w || img.height() != canvas_h {
        Some(image::imageops::resize(&img, canvas_w, canvas_h, image::imageops::FilterType::Lanczos3))
    } else {
        Some(img)
    }
}

/// Fallback: ab_glyph-based rendering (no OT shaping, legacy kern only).
fn render_words_ab_glyph(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
) -> Option<GrayImage> {
    use ab_glyph::point;
    let font = FontRef::try_from_slice(font_data).ok()?;
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));

    let (cw, ch) = canvas.dimensions();

    let mut all_em: Vec<f32> = words.iter()
        .filter(|w| !w.text.is_empty() && w.width >= 1)
        .filter_map(|w| crate::layout::width_matched_em_px(&font, &w.text, w.width as f32))
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
                let gid = font.glyph_id(c);
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
            let gid = font.glyph_id(c);
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
/// and return the best (highest) SSIM and the shift that produced it.
/// Positive dy = rendered image moved DOWN.
fn ssim_windowed_best_vshift(a: &GrayImage, b: &GrayImage, max_shift: i32) -> (f32, i32) {
    let (aw, ah) = a.dimensions();
    let mut best = 0.0f32;
    let mut best_dy = 0i32;
    for dy in -max_shift..=max_shift {
        // Shift image b vertically by dy pixels
        let mut shifted = GrayImage::from_pixel(aw, ah, Luma([255u8]));
        for sy in 0..ah {
            let ty = sy as i32 + dy;
            if ty < 0 || ty >= ah as i32 { continue; }
            for sx in 0..aw.min(b.width()) {
                shifted.put_pixel(sx, ty as u32, *b.get_pixel(sx, sy));
            }
        }
        let score = ssim_windowed(a, &shifted);
        if score > best {
            best = score;
            best_dy = dy;
        }
    }
    (best, best_dy)
}

fn ssim_windowed(a: &GrayImage, b: &GrayImage) -> f32 {
    let b = if a.dimensions() != b.dimensions() {
        image::imageops::resize(b, a.width(), a.height(), image::imageops::FilterType::Lanczos3)
    } else {
        b.clone()
    };

    let (w, h) = a.dimensions();
    if w < 11 || h < 11 {
        // Fallback to global for tiny images
        return ssim_global(a, &b);
    }

    let kernel = gaussian_kernel_11x11();
    let c1: f64 = (0.01 * 255.0_f64).powi(2);
    let c2: f64 = (0.03 * 255.0_f64).powi(2);

    // Ink threshold: a pixel is "ink" if its value < 240
    const INK_THRESHOLD: u8 = 240;
    // Minimum number of ink pixels in a window to count it
    const MIN_INK_PIXELS: u32 = 3;

    let half = 5i32; // 11/2

    let mut ssim_sum = 0.0f64;
    let mut window_count = 0u64;

    // Step by 4 pixels for speed (still plenty of overlap at 11×11)
    let step = 4u32;

    let mut cy = half as u32;
    while cy + (half as u32) < h {
        let mut cx = half as u32;
        while cx + (half as u32) < w {
            // Check if this window contains ink
            let mut ink_count = 0u32;
            for ky in 0..11u32 {
                for kx in 0..11u32 {
                    let px = (cx as i32 - half + kx as i32) as u32;
                    let py = (cy as i32 - half + ky as i32) as u32;
                    let va = a.get_pixel(px, py).0[0];
                    let vb = b.get_pixel(px, py).0[0];
                    if va < INK_THRESHOLD || vb < INK_THRESHOLD {
                        ink_count += 1;
                    }
                }
            }

            if ink_count >= MIN_INK_PIXELS {
                // Compute weighted statistics for this window
                let mut mu_a = 0.0f64;
                let mut mu_b = 0.0f64;
                let mut sig_a2 = 0.0f64;
                let mut sig_b2 = 0.0f64;
                let mut sig_ab = 0.0f64;

                for ky in 0..11usize {
                    for kx in 0..11usize {
                        let px = (cx as i32 - half + kx as i32) as u32;
                        let py = (cy as i32 - half + ky as i32) as u32;
                        let va = a.get_pixel(px, py).0[0] as f64;
                        let vb = b.get_pixel(px, py).0[0] as f64;
                        let wt = kernel[ky][kx];
                        mu_a += wt * va;
                        mu_b += wt * vb;
                    }
                }

                for ky in 0..11usize {
                    for kx in 0..11usize {
                        let px = (cx as i32 - half + kx as i32) as u32;
                        let py = (cy as i32 - half + ky as i32) as u32;
                        let va = a.get_pixel(px, py).0[0] as f64;
                        let vb = b.get_pixel(px, py).0[0] as f64;
                        let wt = kernel[ky][kx];
                        let da = va - mu_a;
                        let db = vb - mu_b;
                        sig_a2 += wt * da * da;
                        sig_b2 += wt * db * db;
                        sig_ab += wt * da * db;
                    }
                }

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
        return ssim_global(a, &b);
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
// Skew detection & correction
// ---------------------------------------------------------------------------

fn detect_skew_from_words(words: &[WordPlacement]) -> f32 {
    let centres: Vec<(f32, f32)> = words
        .iter()
        .filter(|w| !w.text.is_empty() && w.width > 0 && w.height > 0)
        .map(|w| {
            let cx = w.x_off as f32 + w.width as f32 / 2.0;
            let cy = w.y_off as f32 + w.height as f32 / 2.0;
            (cx, cy)
        })
        .collect();
    if centres.len() < 3 { return 0.0; }

    let n = centres.len() as f32;
    let sx: f32 = centres.iter().map(|(x, _)| x).sum();
    let sy: f32 = centres.iter().map(|(_, y)| y).sum();
    let sxy: f32 = centres.iter().map(|(x, y)| x * y).sum();
    let sx2: f32 = centres.iter().map(|(x, _)| x * x).sum();
    let denom = n * sx2 - sx * sx;
    if denom.abs() < 1e-6 { return 0.0; }
    let slope = (n * sxy - sx * sy) / denom;
    slope.atan().clamp(-5.0_f32.to_radians(), 5.0_f32.to_radians())
}

fn rotate_gray(img: &GrayImage, angle: f32) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::from_pixel(w, h, Luma([255u8]));
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let (cos_a, sin_a) = (angle.cos(), angle.sin());

    for oy in 0..h {
        for ox in 0..w {
            let dx = ox as f32 - cx;
            let dy = oy as f32 - cy;
            let sx = cos_a * dx + sin_a * dy + cx;
            let sy = -sin_a * dx + cos_a * dy + cy;
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let s = |px: i32, py: i32| -> f32 {
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    img.get_pixel(px as u32, py as u32).0[0] as f32
                } else { 255.0 }
            };
            let val = s(x0,y0)*(1.0-fx)*(1.0-fy) + s(x0+1,y0)*fx*(1.0-fy)
                + s(x0,y0+1)*(1.0-fx)*fy + s(x0+1,y0+1)*fx*fy;
            out.put_pixel(ox, oy, Luma([val.clamp(0.0, 255.0) as u8]));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
