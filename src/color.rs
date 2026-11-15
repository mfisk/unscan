//! Color utilities — background / text colour detection and text erasure.

use crate::ocr::TextRegion;
use image::{DynamicImage, GrayImage, RgbaImage};

pub type Rgb = (u8, u8, u8);

// ---------------------------------------------------------------------------
// Text colour
// ---------------------------------------------------------------------------

/// Detect the foreground text colour in a bounding box.
///
/// Fast path: crop to the region first so we only convert the small
/// ROI to luma/rgba instead of the whole page, and iterate over raw
/// buffers instead of `get_pixel`.
#[allow(dead_code)]
pub fn detect_text_color(page_img: &DynamicImage, region: &TextRegion) -> Rgb {
    let (iw, ih) = (page_img.width(), page_img.height());

    let x0 = region.x.min(iw.saturating_sub(1));
    let y0 = region.y.min(ih.saturating_sub(1));
    let x1 = (region.x + region.width).min(iw);
    let y1 = (region.y + region.height).min(ih);

    if x1 <= x0 || y1 <= y0 {
        return (0, 0, 0);
    }
    let w = x1 - x0;
    let h = y1 - y0;

    // Crop first — converting only the ROI is ~100-300x cheaper than
    // converting the whole page (6 Mpx -> ~25 kpx for a line).
    // Fast variant: avoid CicpRgb generic cast 8.7% leaf by using as_raw + manual luma.
    let sub = page_img.crop_imm(x0, y0, w, h);
    // Try to reuse underlying buffer type to avoid generic cast.
    let (gray, rgba) = match &sub {
        DynamicImage::ImageLuma8(l) => {
            let (sw, sh) = l.dimensions();
            // luma -> rgba expansion manual
            let mut rbuf = Vec::with_capacity((sw * sh * 4) as usize);
            for &v in l.as_raw() { rbuf.extend_from_slice(&[v, v, v, 255]); }
            (l.clone(), image::RgbaImage::from_raw(sw, sh, rbuf).expect("rgba"))
        }
        DynamicImage::ImageRgba8(r) => {
            // rgba -> luma fast Rec709 to match image 0.25
            let (sw, sh) = r.dimensions();
            let mut gbuf = Vec::with_capacity((sw * sh) as usize);
            for chunk in r.as_raw().chunks_exact(4) {
                gbuf.push(((chunk[0] as u32 * 2126 + chunk[1] as u32 * 7152 + chunk[2] as u32 * 722) / 10000) as u8);
            }
            (image::GrayImage::from_raw(sw, sh, gbuf).expect("gray"), r.clone())
        }
        _ => {
            // fallback – small ROI so generic cost negligible
            (sub.to_luma8(), sub.to_rgba8())
        }
    };

    let gray_raw = gray.as_raw();
    if gray_raw.is_empty() {
        return (0, 0, 0);
    }

    // Build histogram directly — no Vec<u8> allocation for vals.
    let mut hist = [0u32; 256];
    for &v in gray_raw {
        hist[v as usize] += 1;
    }

    // Otsu from histogram (same math as before).
    let threshold = otsu_from_hist(&hist, gray_raw.len());

    // Dark count from histogram instead of re-scanning vals.
    let mut dark_count: u32 = 0;
    for i in 0..=threshold as usize {
        dark_count += hist[i];
    }
    let text_is_dark = (dark_count as usize) <= gray_raw.len() / 2;

    // Average colour of text pixels using raw buffers (no bounds checks).
    let rgba_raw = rgba.as_raw(); // w*h*4
    let (mut rs, mut gs, mut bs, mut cnt) = (0u64, 0u64, 0u64, 0u64);
    if text_is_dark {
        for i in 0..gray_raw.len() {
            if gray_raw[i] <= threshold {
                let j = i * 4;
                rs += rgba_raw[j] as u64;
                gs += rgba_raw[j + 1] as u64;
                bs += rgba_raw[j + 2] as u64;
                cnt += 1;
            }
        }
    } else {
        for i in 0..gray_raw.len() {
            if gray_raw[i] > threshold {
                let j = i * 4;
                rs += rgba_raw[j] as u64;
                gs += rgba_raw[j + 1] as u64;
                bs += rgba_raw[j + 2] as u64;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        return (0, 0, 0);
    }
    ((rs / cnt) as u8, (gs / cnt) as u8, (bs / cnt) as u8)
}

/// Zero-alloc version for callers that already have the full-page
/// Gray + RGBA buffers (hoisted conversion). No cropping allocation,
/// just raw indexing into the full-page buffers.
pub fn detect_text_color_from_buffers(
    gray: &GrayImage,
    rgba: &RgbaImage,
    region: &TextRegion,
) -> Rgb {
    let (iw_u32, ih_u32) = gray.dimensions();
    let iw = iw_u32 as usize;
    // Clamp region
    let x0 = (region.x.min(iw_u32.saturating_sub(1))) as usize;
    let y0 = (region.y.min(ih_u32.saturating_sub(1))) as usize;
    let x1 = ((region.x + region.width).min(iw_u32)) as usize;
    let y1 = ((region.y + region.height).min(ih_u32)) as usize;
    if x1 <= x0 || y1 <= y0 {
        return (0, 0, 0);
    }
    let w = x1 - x0;
    let h = y1 - y0;
    let total = w * h;
    if total == 0 {
        return (0, 0, 0);
    }

    let gray_raw = gray.as_raw();
    let rgba_raw = rgba.as_raw();

    // Build histogram from ROI by strided raw access
    let mut hist = [0u32; 256];
    for py in y0..y1 {
        let row_base = py * iw;
        let base = row_base + x0;
        for px in 0..w {
            let v = gray_raw[base + px];
            hist[v as usize] += 1;
        }
    }

    let threshold = otsu_from_hist(&hist, total);
    let mut dark_count: u32 = 0;
    for i in 0..=threshold as usize {
        dark_count += hist[i];
    }
    let text_is_dark = (dark_count as usize) <= total / 2;

    let (mut rs, mut gs, mut bs, mut cnt) = (0u64, 0u64, 0u64, 0u64);
    if text_is_dark {
        for py in y0..y1 {
            let row_base = py * iw;
            let rgba_row_base = row_base * 4;
            for px in 0..w {
                let x = x0 + px;
                let gray_idx = row_base + x;
                if gray_raw[gray_idx] <= threshold {
                    let rgba_idx = rgba_row_base + x * 4;
                    rs += rgba_raw[rgba_idx] as u64;
                    gs += rgba_raw[rgba_idx + 1] as u64;
                    bs += rgba_raw[rgba_idx + 2] as u64;
                    cnt += 1;
                }
            }
        }
    } else {
        for py in y0..y1 {
            let row_base = py * iw;
            let rgba_row_base = row_base * 4;
            for px in 0..w {
                let x = x0 + px;
                let gray_idx = row_base + x;
                if gray_raw[gray_idx] > threshold {
                    let rgba_idx = rgba_row_base + x * 4;
                    rs += rgba_raw[rgba_idx] as u64;
                    gs += rgba_raw[rgba_idx + 1] as u64;
                    bs += rgba_raw[rgba_idx + 2] as u64;
                    cnt += 1;
                }
            }
        }
    }
    if cnt == 0 {
        return (0, 0, 0);
    }
    ((rs / cnt) as u8, (gs / cnt) as u8, (bs / cnt) as u8)
}

/// Otsu's method: find threshold that maximises between-class variance.
#[allow(dead_code)]
pub fn otsu_threshold(vals: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &v in vals {
        hist[v as usize] += 1;
    }
    otsu_from_hist(&hist, vals.len())
}

pub(crate) fn otsu_from_hist(hist: &[u32; 256], total_len: usize) -> u8 {
    if total_len == 0 {
        return 128;
    }
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
        let w_fg = (total_len as u32) - w_bg;
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
    detect_background_color_from_buffer(&rgba)
}

pub fn detect_background_color_from_buffer(rgba: &RgbaImage) -> Rgb {
    let (w_u32, h_u32) = rgba.dimensions();
    if w_u32 == 0 || h_u32 == 0 {
        return (255, 255, 255);
    }
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    let margin = (5usize).min(w / 4).min(h / 4).max(1);
    let raw = rgba.as_raw();
    let (mut rs, mut gs, mut bs, mut cnt) = (0u64, 0u64, 0u64, 0u64);

    // Top and bottom margins - contiguous runs
    for py in 0..margin {
        let row_base = py * w * 4;
        for px in 0..w {
            let j = row_base + px * 4;
            rs += raw[j] as u64;
            gs += raw[j + 1] as u64;
            bs += raw[j + 2] as u64;
            cnt += 1;
        }
    }
    if h > margin {
        for py in (h - margin)..h {
            if py < margin { continue; } // avoid double count when h < 2*margin
            let row_base = py * w * 4;
            for px in 0..w {
                let j = row_base + px * 4;
                rs += raw[j] as u64;
                gs += raw[j + 1] as u64;
                bs += raw[j + 2] as u64;
                cnt += 1;
            }
        }
    }
    // Left/right for middle band
    if h > 2 * margin {
        for py in margin..(h - margin) {
            let row_base = py * w * 4;
            for px in 0..margin {
                let j = row_base + px * 4;
                rs += raw[j] as u64;
                gs += raw[j + 1] as u64;
                bs += raw[j + 2] as u64;
                cnt += 1;
            }
            for px in (w - margin)..w {
                let j = row_base + px * 4;
                rs += raw[j] as u64;
                gs += raw[j + 1] as u64;
                bs += raw[j + 2] as u64;
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
    erase_regions_inplace(&mut rgba, rects, bg, margin);
    DynamicImage::ImageRgba8(rgba)
}

pub fn erase_regions_inplace(
    rgba: &mut RgbaImage,
    rects: &[(u32, u32, u32, u32)],
    bg: Rgb,
    margin: u32,
) {
    let (iw_u32, ih_u32) = rgba.dimensions();
    let iw = iw_u32 as usize;
    let w_u32 = iw_u32;
    let h_u32 = ih_u32;
    let raw = rgba.as_mut();
    // Prebuild bg pixel bytes
    let bg_r = bg.0;
    let bg_g = bg.1;
    let bg_b = bg.2;
    for &(rx, ry, rw, rh) in rects {
        let x0 = (rx.saturating_sub(margin).min(w_u32)) as usize;
        let y0 = (ry.saturating_sub(margin).min(h_u32)) as usize;
        let x1 = ((rx + rw + margin).min(w_u32)) as usize;
        let y1 = ((ry + rh + margin).min(h_u32)) as usize;
        if x1 <= x0 || y1 <= y0 { continue; }
        for py in y0..y1 {
            let row_base = py * iw * 4 + x0 * 4;
            // Fill span [x0, x1) with bg color
            let mut j = row_base;
            let end = row_base + (x1 - x0) * 4;
            while j < end {
                raw[j] = bg_r;
                raw[j + 1] = bg_g;
                raw[j + 2] = bg_b;
                raw[j + 3] = 255;
                j += 4;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Content detection
// ---------------------------------------------------------------------------

/// Check if a region has visual content (not just uniform background).
pub fn region_has_content(gray: &image::GrayImage, x: u32, y: u32, w: u32, h: u32) -> bool {
    let (iw_u32, ih_u32) = gray.dimensions();
    let iw = iw_u32 as usize;
    let x0 = (x.min(iw_u32)) as usize;
    let y0 = (y.min(ih_u32)) as usize;
    let x1 = ((x + w).min(iw_u32)) as usize;
    let y1 = ((y + h).min(ih_u32)) as usize;
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    let raw = gray.as_raw();
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let mut cnt = 0u64;
    for py in y0..y1 {
        let row_base = py * iw;
        for px in x0..x1 {
            let v = raw[row_base + px] as u64;
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
