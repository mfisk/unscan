//! HOG (Histogram of Oriented Gradients) features for character classification.
//!
//! Designed to discriminate between different characters rendered in the same
//! font — the complement of the main `features.rs` which discriminates between
//! fonts rendering the same character.
//!
//! Feature vector layout:
//!   4×4 spatial grid of cells, 8 unsigned-gradient orientation bins per cell.
//!   Total: 128 dimensions.
//!
//! Input: a grayscale crop already normalized to `NORM_H` height (24 px).
//! The crop is padded/resized to a fixed 24×24 square before HOG extraction.

use image::GrayImage;

/// Number of orientation bins (unsigned gradients, 0°–180°).
pub const HOG_ORIENT_BINS: usize = 8;
/// Spatial grid: cells per side.
pub const HOG_CELLS_PER_SIDE: usize = 4;
/// Total HOG feature length.
pub const HOG_FEAT_LEN: usize = HOG_CELLS_PER_SIDE * HOG_CELLS_PER_SIDE * HOG_ORIENT_BINS;

/// Compute HOG features from a normalized grayscale glyph crop.
///
/// Returns `None` if the image is degenerate (< 3px in either dimension).
pub fn compute_hog(img: &GrayImage) -> Option<[f32; HOG_FEAT_LEN]> {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return None;
    }

    // ── 1. Resize to fixed 24×24 square ──────────────────────────────
    // Pad the shorter axis to make a square, then resize.
    let target = 24u32;
    let square = if w == target && h == target {
        img.clone()
    } else {
        // Letterbox: fit into target×target preserving aspect ratio,
        // padding with white (255).
        let scale = (target as f32 / w as f32).min(target as f32 / h as f32);
        let new_w = (w as f32 * scale).round().max(1.0) as u32;
        let new_h = (h as f32 * scale).round().max(1.0) as u32;
        let resized = image::imageops::resize(
            img, new_w, new_h,
            image::imageops::FilterType::Lanczos3,
        );
        let mut sq = GrayImage::from_pixel(target, target, image::Luma([255u8]));
        let ox = (target - new_w) / 2;
        let oy = (target - new_h) / 2;
        image::imageops::overlay(&mut sq, &resized, ox as i64, oy as i64);
        sq
    };

    let sw = square.width() as usize;
    let sh = square.height() as usize;

    // ── 2. Compute Sobel gradients ───────────────────────────────────
    // Central differences; border pixels use forward/backward difference.
    let pixels = square.as_raw();
    let mut gx = vec![0.0f32; sw * sh];
    let mut gy = vec![0.0f32; sw * sh];

    for y in 0..sh {
        for x in 0..sw {
            let dx = if x == 0 {
                pixels[y * sw + 1] as f32 - pixels[y * sw] as f32
            } else if x == sw - 1 {
                pixels[y * sw + x] as f32 - pixels[y * sw + x - 1] as f32
            } else {
                (pixels[y * sw + x + 1] as f32 - pixels[y * sw + x - 1] as f32) * 0.5
            };

            let dy = if y == 0 {
                pixels[sw + x] as f32 - pixels[x] as f32
            } else if y == sh - 1 {
                pixels[y * sw + x] as f32 - pixels[(y - 1) * sw + x] as f32
            } else {
                (pixels[(y + 1) * sw + x] as f32 - pixels[(y - 1) * sw + x] as f32) * 0.5
            };

            gx[y * sw + x] = dx;
            gy[y * sw + x] = dy;
        }
    }

    // ── 3. Build cell histograms ─────────────────────────────────────
    // Divide the image into a HOG_CELLS_PER_SIDE × HOG_CELLS_PER_SIDE grid.
    // Each cell covers (target/cells_per_side) pixels per axis = 6×6 pixels.
    let cell_w = sw as f32 / HOG_CELLS_PER_SIDE as f32;
    let cell_h = sh as f32 / HOG_CELLS_PER_SIDE as f32;
    let bin_width = std::f32::consts::PI / HOG_ORIENT_BINS as f32; // ~22.5°

    let mut hog = [0.0f32; HOG_FEAT_LEN];

    for y in 0..sh {
        for x in 0..sw {
            let dx = gx[y * sw + x];
            let dy = gy[y * sw + x];
            let mag = (dx * dx + dy * dy).sqrt();
            if mag < 1e-6 {
                continue; // skip flat regions
            }

            // Unsigned angle in [0, π)
            let mut angle = dy.atan2(dx); // [-π, π]
            if angle < 0.0 {
                angle += std::f32::consts::PI; // map to [0, π)
            }
            if angle >= std::f32::consts::PI {
                angle -= std::f32::consts::PI;
            }

            // Bilinear interpolation across orientation bins
            let bin_f = angle / bin_width;
            let bin0 = bin_f.floor() as usize % HOG_ORIENT_BINS;
            let bin1 = (bin0 + 1) % HOG_ORIENT_BINS;
            let frac = bin_f - bin_f.floor();

            // Which cell does this pixel belong to?
            let cx = ((x as f32 / cell_w).floor() as usize).min(HOG_CELLS_PER_SIDE - 1);
            let cy = ((y as f32 / cell_h).floor() as usize).min(HOG_CELLS_PER_SIDE - 1);
            let cell_offset = (cy * HOG_CELLS_PER_SIDE + cx) * HOG_ORIENT_BINS;

            hog[cell_offset + bin0] += mag * (1.0 - frac);
            hog[cell_offset + bin1] += mag * frac;
        }
    }

    // ── 4. L2 normalize the whole descriptor ─────────────────────────
    let norm: f32 = hog.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for v in &mut hog {
            *v /= norm;
        }
    }

    Some(hog)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hog_returns_correct_length() {
        // 24×24 white image → all zeros (no gradients)
        let img = GrayImage::from_pixel(24, 24, image::Luma([255u8]));
        let hog = compute_hog(&img);
        assert!(hog.is_some());
        let h = hog.unwrap();
        assert_eq!(h.len(), HOG_FEAT_LEN);
        // Flat image → zero gradients → all zeros
        assert!(h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn hog_nonzero_for_real_image() {
        // Create a simple image with some structure
        let mut img = GrayImage::from_pixel(24, 24, image::Luma([255u8]));
        // Draw a vertical line
        for y in 2..22 {
            img.put_pixel(12, y, image::Luma([0u8]));
        }
        let hog = compute_hog(&img);
        assert!(hog.is_some());
        let h = hog.unwrap();
        // Should have nonzero gradient energy
        assert!(h.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn hog_narrow_image() {
        // Narrow image (typical character crop: 12×24)
        let img = GrayImage::from_pixel(12, 24, image::Luma([128u8]));
        let hog = compute_hog(&img);
        assert!(hog.is_some());
        assert_eq!(hog.unwrap().len(), HOG_FEAT_LEN);
    }
}
