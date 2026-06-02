//! Whole-page deskewing via Hough line transform.
//!
//! Detects slight rotational skew (typically 0.1–2°) introduced by scanning
//! and corrects it before OCR, improving character segmentation and font
//! matching accuracy.

use image::{GrayImage, Luma};




/// Detect the dominant skew angle of text lines using a simplified
/// Hough line transform focused on near-horizontal lines.
pub fn detect_skew(img: &GrayImage) -> f32 {
    let (w, h) = (img.width() as usize, img.height() as usize);

    // 1. Downsample for speed — work at ~600px wide max
    let scale = if w > 600 { 600.0 / w as f32 } else { 1.0 };
    let sw = (w as f32 * scale).max(1.0) as usize;
    let sh = (h as f32 * scale).max(1.0) as usize;

    // 2. Build binary edge image (Sobel-Y to find horizontal text edges)
    let edges = compute_edges(img, sw, sh, scale);

    // 3. Hough accumulator for near-horizontal lines
    //    Standard Hough: x·cos(θ) + y·sin(θ) = ρ
    //    A horizontal line has normal at θ = 90°.
    //    Skew = detected_θ - 90°.
    let theta_center: f32 = 90.0;
    let theta_range: f32 = 5.0;
    let theta_step: f32 = 0.05;
    let n_theta: usize = ((2.0 * theta_range / theta_step) as usize) + 1;

    let diag = ((sw * sw + sh * sh) as f32).sqrt().ceil() as usize;
    let rho_max = diag;
    let rho_offset = rho_max;
    let n_rho = 2 * rho_max + 1;

    // Pre-compute sin/cos for each θ
    let thetas: Vec<(f32, f32, f32)> = (0..n_theta)
        .map(|i| {
            let deg = (theta_center - theta_range) + i as f32 * theta_step;
            let rad = deg.to_radians();
            (rad.sin(), rad.cos(), deg)
        })
        .collect();

    // Accumulate votes
    let mut accum = vec![0u32; n_theta * n_rho];

    for y in 0..sh {
        for x in 0..sw {
            if edges[y * sw + x] {
                for (ti, &(sin_t, cos_t, _)) in thetas.iter().enumerate() {
                    let rho = (x as f32 * cos_t + y as f32 * sin_t).round() as i32;
                    let ri = (rho + rho_offset as i32) as usize;
                    if ri < n_rho {
                        accum[ti * n_rho + ri] += 1;
                    }
                }
            }
        }
    }

    // 4. Find peak angle by selecting the θ with maximum total votes,
    //    then refine with parabolic interpolation for sub-step accuracy.
    //    (Previous weighted-average approach was biased toward 0° by noise.)
    let max_votes = accum.iter().copied().max().unwrap_or(0);
    if max_votes < 10 {
        return 0.0;
    }
    let threshold = (max_votes as f32 * 0.3) as u32;

    let mut theta_sums: Vec<u64> = vec![0; n_theta];
    for ti in 0..n_theta {
        for ri in 0..n_rho {
            let votes = accum[ti * n_rho + ri];
            if votes >= threshold {
                theta_sums[ti] += votes as u64;
            }
        }
    }

    let (best_ti, best_sum) = theta_sums
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| *s)
        .map(|(i, &s)| (i, s))
        .unwrap_or((n_theta / 2, 0));

    if best_sum == 0 {
        return 0.0;
    }

    // Parabolic interpolation around peak for sub-step accuracy
    let skew = if best_ti > 0 && best_ti < n_theta - 1 {
        let left = theta_sums[best_ti - 1] as f64;
        let center = best_sum as f64;
        let right = theta_sums[best_ti + 1] as f64;
        let denom = 2.0 * center - left - right;
        if denom > 0.0 {
            let delta = (right - left) / (2.0 * denom);
            (thetas[best_ti].2 - theta_center) as f64 + delta * theta_step as f64
        } else {
            (thetas[best_ti].2 - theta_center) as f64
        }
    } else {
        (thetas[best_ti].2 - theta_center) as f64
    };

    skew as f32
}

/// Compute a binary edge map at the target resolution.
/// Uses Sobel-Y gradient to find horizontal text edges (baselines, tops).
fn compute_edges(img: &GrayImage, tw: usize, th: usize, scale: f32) -> Vec<bool> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let inv_scale = 1.0 / scale;

    let mut grad = vec![0i32; tw * th];
    let mut max_grad: i32 = 0;

    for ty in 1..th.saturating_sub(1) {
        for tx in 1..tw.saturating_sub(1) {
            let sx = (tx as f32 * inv_scale) as usize;
            let sy = (ty as f32 * inv_scale) as usize;

            if sx < 1 || sy < 1 || sx >= w - 1 || sy >= h - 1 {
                continue;
            }

            // Sobel-Y: detects horizontal edges
            let p = |dx: i32, dy: i32| -> i32 {
                let px = (sx as i32 + dx) as usize;
                let py = (sy as i32 + dy) as usize;
                img.get_pixel(px as u32, py as u32).0[0] as i32
            };

            let gy = -p(-1, -1) - 2 * p(0, -1) - p(1, -1)
                + p(-1, 1) + 2 * p(0, 1) + p(1, 1);

            let abs_gy = gy.abs();
            grad[ty * tw + tx] = abs_gy;
            if abs_gy > max_grad {
                max_grad = abs_gy;
            }
        }
    }

    let thresh = (max_grad as f32 * 0.20) as i32;
    grad.iter().map(|&g| g > thresh).collect()
}

/// Rotate a grayscale image by `angle_deg` degrees around its center using
/// bilinear interpolation, filling new pixels with white (255).
pub fn rotate_gray(img: &GrayImage, angle_deg: f32) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    let mut out = GrayImage::from_pixel(w, h, Luma([255u8]));

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let src = img.as_raw();
    let dst = out.as_mut();
    let stride = w as usize;

    for oy in 0..h {
        let dy = oy as f32 - cy;
        let base_sx = sin_a * dy + cx;
        let base_sy = cos_a * dy + cy;

        for ox in 0..w {
            let dx = ox as f32 - cx;
            let sx = cos_a * dx + base_sx;
            let sy = -sin_a * dx + base_sy;

            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;

            if x0 < 0 || y0 < 0 || x0 + 1 >= w as i32 || y0 + 1 >= h as i32 {
                continue;
            }

            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let i00 = y0 as usize * stride + x0 as usize;
            let val = src[i00] as f32 * (1.0 - fx) * (1.0 - fy)
                + src[i00 + 1] as f32 * fx * (1.0 - fy)
                + src[i00 + stride] as f32 * (1.0 - fx) * fy
                + src[i00 + stride + 1] as f32 * fx * fy;

            dst[oy as usize * stride + ox as usize] = val.round() as u8;
        }
    }

    out
}

