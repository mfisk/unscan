//! Color utilities — background / text colour detection and text erasure.

use crate::ocr::TextRegion;
use image::{DynamicImage, Rgba};

pub type Rgb = (u8, u8, u8);

// ---------------------------------------------------------------------------
// Text colour
// ---------------------------------------------------------------------------

/// Detect the foreground text colour in a bounding box.
pub fn detect_text_color(page_img: &DynamicImage, region: &TextRegion) -> Rgb {
    let rgba = page_img.to_rgba8();
    let gray = page_img.to_luma8();
    let (iw, ih) = rgba.dimensions();

    let x0 = region.x.min(iw.saturating_sub(1));
    let y0 = region.y.min(ih.saturating_sub(1));
    let x1 = (region.x + region.width).min(iw);
    let y1 = (region.y + region.height).min(ih);

    if x1 <= x0 || y1 <= y0 {
        return (0, 0, 0);
    }

    // Collect luminance values for Otsu thresholding.
    let mut vals: Vec<u8> = Vec::new();
    for py in y0..y1 {
        for px in x0..x1 {
            vals.push(gray.get_pixel(px, py).0[0]);
        }
    }
    if vals.is_empty() {
        return (0, 0, 0);
    }

    // Otsu threshold to separate text from background.
    let threshold = otsu_threshold(&vals);

    // Determine if text is dark or light: whichever side of the threshold
    // has fewer pixels is the text (text is the minority).
    let dark_count = vals.iter().filter(|&&v| v <= threshold).count();
    let text_is_dark = dark_count <= vals.len() / 2;

    // Average the colour of text pixels.
    let (mut rs, mut gs, mut bs, mut cnt) = (0u64, 0u64, 0u64, 0u64);
    for py in y0..y1 {
        for px in x0..x1 {
            let gv = gray.get_pixel(px, py).0[0];
            let is_text = if text_is_dark { gv <= threshold } else { gv > threshold };
            if is_text {
                let Rgba([r, g, b, _]) = *rgba.get_pixel(px, py);
                rs += r as u64;
                gs += g as u64;
                bs += b as u64;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        return (0, 0, 0);
    }
    ((rs / cnt) as u8, (gs / cnt) as u8, (bs / cnt) as u8)
}

/// Otsu's method: find threshold that maximises between-class variance.
pub fn otsu_threshold(vals: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &v in vals {
        hist[v as usize] += 1;
    }
    let total = vals.len() as f64;
    let mut sum_total = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum_total += i as f64 * c as f64;
    }
    let mut sum_bg = 0.0f64;
    let mut w_bg = 0u32;
    let mut max_var = 0.0f64;
    let mut thr = 128u8;
    for (t, &c) in hist.iter().enumerate() {
        w_bg += c;
        if w_bg == 0 {
            continue;
        }
        let w_fg = (total as u32) - w_bg;
        if w_fg == 0 {
            break;
        }
        sum_bg += t as f64 * c as f64;
        let m_bg = sum_bg / w_bg as f64;
        let m_fg = (sum_total - sum_bg) / w_fg as f64;
        let var = w_bg as f64 * w_fg as f64 * (m_bg - m_fg).powi(2);
        if var > max_var {
            max_var = var;
            thr = t as u8;
        }
    }
    thr
}

// ---------------------------------------------------------------------------
// Background colour
// ---------------------------------------------------------------------------

/// Detect the dominant background colour (average of border pixels).
pub fn detect_background_color(page_img: &DynamicImage) -> Rgb {
    let rgba = page_img.to_rgba8();
    let (w, h) = rgba.dimensions();
    if w == 0 || h == 0 {
        return (255, 255, 255);
    }
    let margin = 5.min(w / 4).min(h / 4).max(1);
    let (mut rs, mut gs, mut bs, mut cnt) = (0u64, 0u64, 0u64, 0u64);
    for py in 0..h {
        for px in 0..w {
            if px < margin || px >= w - margin || py < margin || py >= h - margin {
                let Rgba([r, g, b, _]) = *rgba.get_pixel(px, py);
                rs += r as u64;
                gs += g as u64;
                bs += b as u64;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        return (255, 255, 255);
    }
    ((rs / cnt) as u8, (gs / cnt) as u8, (bs / cnt) as u8)
}

// ---------------------------------------------------------------------------
// Erasure
// ---------------------------------------------------------------------------

/// Paint over given regions with the background colour.
pub fn erase_regions(
    page_img: &DynamicImage,
    rects: &[(u32, u32, u32, u32)], // (x, y, w, h)
    bg: Rgb,
    margin: u32,
) -> DynamicImage {
    let mut rgba = page_img.to_rgba8();
    let (iw, ih) = rgba.dimensions();
    for &(rx, ry, rw, rh) in rects {
        let x0 = rx.saturating_sub(margin).min(iw);
        let y0 = ry.saturating_sub(margin).min(ih);
        let x1 = (rx + rw + margin).min(iw);
        let y1 = (ry + rh + margin).min(ih);
        for py in y0..y1 {
            for px in x0..x1 {
                rgba.put_pixel(px, py, Rgba([bg.0, bg.1, bg.2, 255]));
            }
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

// ---------------------------------------------------------------------------
// Content detection
// ---------------------------------------------------------------------------

/// Check if a region has visual content (not just uniform background).
pub fn region_has_content(gray: &image::GrayImage, x: u32, y: u32, w: u32, h: u32) -> bool {
    let (iw, ih) = gray.dimensions();
    let x0 = x.min(iw);
    let y0 = y.min(ih);
    let x1 = (x + w).min(iw);
    let y1 = (y + h).min(ih);
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let mut cnt = 0u64;
    for py in y0..y1 {
        for px in x0..x1 {
            let v = gray.get_pixel(px, py).0[0] as u64;
            sum += v;
            sum_sq += v * v;
            cnt += 1;
        }
    }
    if cnt == 0 {
        return false;
    }
    let mean = sum as f64 / cnt as f64;
    let variance = (sum_sq as f64 / cnt as f64) - mean * mean;
    variance > 100.0
}
