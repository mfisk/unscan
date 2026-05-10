//! SSIM verification — render vector text back to raster and compare with the
//! original to catch bad replacements before they make it into the output.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
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

/// Verify a vectorised text region by rendering each word at its OCR-reported
/// position, then computing SSIM against the (optionally deskewed) original.
pub fn verify_text_region(
    original_gray: &GrayImage,
    font_data: &[u8],
    _text: &str,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    words: &[TextRegion],
) -> f32 {
    let (iw, ih) = original_gray.dimensions();
    let x = x.min(iw.saturating_sub(1));
    let y = y.min(ih.saturating_sub(1));
    let w = width.min(iw - x);
    let h = height.min(ih - y);

    if w < 3 || h < 3 {
        return 0.0;
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

    let rendered = match render_words(font_data, &placements, w, h) {
        Some(r) => r,
        None => return 0.0,
    };

    let a = binarise(&deskewed);
    let b = binarise(&rendered);
    ssim(&a, &b)
}

// ---------------------------------------------------------------------------
// Word-by-word renderer (v5 approach — per-word width scaling, shared baseline)
// ---------------------------------------------------------------------------

fn render_words(
    font_data: &[u8],
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
) -> Option<GrayImage> {
    let font = FontRef::try_from_slice(font_data).ok()?;
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));

    let ref_h = 100.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);
    let ref_ink = sf_ref.ascent() - sf_ref.descent();

    // Median word bbox height for the shared baseline scale.
    let mut heights: Vec<u32> = words
        .iter()
        .filter(|w| !w.text.is_empty() && w.height > 0)
        .map(|w| w.height)
        .collect();
    if heights.is_empty() {
        return Some(canvas);
    }
    heights.sort();
    let median_h = heights[heights.len() / 2] as f32;

    // Line-level scale: ink height ≈ median word bbox height.
    let line_px_h = ref_h * (median_h / ref_ink);
    let line_scale = PxScale::from(line_px_h);
    let sf_line = font.as_scaled(line_scale);
    let line_ink = sf_line.ascent() - sf_line.descent();
    let baseline_y = (canvas_h as f32 - line_ink) / 2.0 + sf_line.ascent();

    let (cw, ch) = canvas.dimensions();

    for word in words {
        if word.text.is_empty() || word.width < 1 {
            continue;
        }

        // Per-word width-derived scale: advance width matches OCR bbox width.
        let mut adv = 0.0f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for c in word.text.chars() {
            let gid = font.glyph_id(c);
            if let Some(p) = prev {
                adv += sf_ref.kern(p, gid);
            }
            adv += sf_ref.h_advance(gid);
            prev = Some(gid);
        }
        if adv < 0.1 {
            continue;
        }

        let word_px_h = (ref_h * (word.width as f32 / adv)).clamp(4.0, 500.0);
        let word_scale = PxScale::from(word_px_h);
        let sf_word = font.as_scaled(word_scale);

        let mut cx = word.x_off as f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;

        for c in word.text.chars() {
            let gid = font.glyph_id(c);
            if let Some(p) = prev {
                cx += sf_word.kern(p, gid);
            }
            let glyph = gid.with_scale_and_position(word_scale, point(cx, baseline_y));
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
            cx += sf_word.h_advance(gid);
            prev = Some(gid);
        }
    }

    Some(canvas)
}

// ---------------------------------------------------------------------------
// SSIM
// ---------------------------------------------------------------------------

fn ssim(a: &GrayImage, b: &GrayImage) -> f32 {
    let b = if a.dimensions() != b.dimensions() {
        image::imageops::resize(b, a.width(), a.height(), image::imageops::FilterType::Lanczos3)
    } else {
        b.clone()
    };

    let c1: f64 = (0.01 * 255.0_f64).powi(2);
    let c2: f64 = (0.03 * 255.0_f64).powi(2);
    let n = (a.width() as u64 * a.height() as u64) as f64;
    if n == 0.0 { return 1.0; }

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
    if den < 1e-10 { return 1.0; }
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
    if centres.len() < 2 { return 0.0; }

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

fn binarise(img: &GrayImage) -> GrayImage {
    let mut hist = [0u32; 256];
    for p in img.pixels() { hist[p.0[0] as usize] += 1; }
    let total = (img.width() * img.height()) as f64;
    let mut sum_total = 0.0f64;
    for (i, &c) in hist.iter().enumerate() { sum_total += i as f64 * c as f64; }
    let (mut sum_bg, mut w_bg, mut max_var, mut thr) = (0.0f64, 0u32, 0.0f64, 128u8);
    for (t, &c) in hist.iter().enumerate() {
        w_bg += c;
        if w_bg == 0 { continue; }
        let w_fg = (total as u32) - w_bg;
        if w_fg == 0 { break; }
        sum_bg += t as f64 * c as f64;
        let m_bg = sum_bg / w_bg as f64;
        let m_fg = (sum_total - sum_bg) / w_fg as f64;
        let var = w_bg as f64 * w_fg as f64 * (m_bg - m_fg).powi(2);
        if var > max_var { max_var = var; thr = t as u8; }
    }
    let mut out = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, Luma([if p.0[0] <= thr { 0 } else { 255 }]));
    }
    out
}
