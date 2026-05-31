//! `--diag-seg` segmentation diagnostics.
//!
//! Dumps the full OCR → segmentation pipeline for every word:
//!   word_NNN_TEXT.png         — raw word crop from Tesseract bbox
//!   word_NNN_TEXT_vp.png      — VP pass: zero-ink runs highlighted cyan, split midpoints red
//!   word_NNN_TEXT_seam.png    — VP (red) + seam splits (blue) overlaid
//!   word_NNN_TEXT_final.png   — all splits (VP red, seam blue) overlaid
//!   word_NNN_TEXT_chars/      — per-character crop PNGs (00_A.png, 01_B.png, ...)
//!   summary.json              — structured dump of everything

use image::{GrayImage, Rgb, RgbImage};
use std::path::Path;

/// Save a grayscale image as RGB.
fn gray_to_rgb(img: &GrayImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut rgb = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let g = img.get_pixel(x, y).0[0];
            rgb.put_pixel(x, y, Rgb([g, g, g]));
        }
    }
    rgb
}

/// Overlay VP whitespace runs on the word image.
/// Runs are tinted cyan; split midpoints are red vertical lines.
pub fn save_vp_overlay(
    img: &GrayImage,
    ws_runs: &[(u32, u32)],  // (start, end_exclusive)
    path: &Path,
) {
    let (w, h) = img.dimensions();
    let mut rgb = gray_to_rgb(img);
    for &(rs, re) in ws_runs {
        for x in rs..re.min(w) {
            for y in 0..h {
                let p = rgb.get_pixel(x, y).0;
                rgb.put_pixel(x, y, Rgb([
                    p[0].saturating_sub(40),
                    p[1].saturating_add(60),
                    p[2].saturating_add(100),
                ]));
            }
        }
        let mid = (rs + re) / 2;
        if mid < w {
            for y in 0..h {
                rgb.put_pixel(mid, y, Rgb([255, 0, 0]));
            }
        }
    }
    let _ = rgb.save(path);
}

/// Overlay split lines on the word image.
/// `vp` in red, `seam` in blue.
pub fn save_split_overlay(
    img: &GrayImage,
    vp: &[u32],
    seam: &[u32],
    extra: &[u32],
    path: &Path,
) {
    save_split_overlay_with_paths(img, vp, seam, extra, &std::collections::HashMap::new(), path);
}

/// Like save_split_overlay but draws actual diagonal seam paths instead of
/// vertical lines for seam splits.
pub fn save_split_overlay_with_paths(
    img: &GrayImage,
    vp: &[u32],
    _seam: &[u32],
    _extra: &[u32],
    seam_paths: &std::collections::HashMap<u32, Vec<u32>>,
    path: &Path,
) {
    let (w, h) = img.dimensions();
    let mut rgb = gray_to_rgb(img);
    // VP splits: red vertical lines
    for &x in vp {
        if x < w {
            for y in 0..h { rgb.put_pixel(x, y, Rgb([255, 0, 0])); }
        }
    }
    // Seam splits: blue diagonal paths
    for (_col, sp) in seam_paths {
        for (y, &x) in sp.iter().enumerate() {
            if x < w && (y as u32) < h {
                rgb.put_pixel(x, y as u32, Rgb([0, 100, 255]));
                // Thicken: draw ±1 pixel horizontally for visibility
                if x > 0 { rgb.put_pixel(x - 1, y as u32, Rgb([0, 100, 255])); }
                if x + 1 < w { rgb.put_pixel(x + 1, y as u32, Rgb([0, 100, 255])); }
            }
        }
    }
    let _ = rgb.save(path);
}

/// Save individual character crops from final boundaries.
pub fn save_char_crops(
    img: &GrayImage,
    bounds: &[u32],
    chars: &[char],
    dir: &Path,
) {
    let _ = std::fs::create_dir_all(dir);
    let (_, h) = img.dimensions();
    for (i, window) in bounds.windows(2).enumerate() {
        let x0 = window[0];
        let x1 = window[1];
        if x1 <= x0 { continue; }
        let crop = image::imageops::crop_imm(img, x0, 0, x1 - x0, h).to_image();
        let label = chars.get(i).copied().unwrap_or('?');
        let fname = format!("{:02}_{}.png", i, sanitize_char(label));
        let _ = crop.save(dir.join(fname));
    }
}

pub fn sanitize_char(c: char) -> String {
    if c.is_alphanumeric() { c.to_string() }
    else { format!("U{:04X}", c as u32) }
}

/// Sanitize text for filenames.
pub fn sanitize_text(s: &str) -> String {
    s.chars()
        .take(25)
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}
