//! Shared image‐comparison implementations for grayscale images.
//!
//! Provides SSIM (legacy), Gaussian-windowed ink-aware SSIM with vertical
//! shift search, and ZNCC (Zero-mean Normalised Cross-Correlation).
//! ZNCC is the primary scoring metric for font verification — it is
//! inherently invariant to linear brightness/contrast differences between
//! scan and render.

use image::GrayImage;

// ---------------------------------------------------------------------------
// Global SSIM
// ---------------------------------------------------------------------------

/// Global SSIM between two grayscale images.
/// If sizes differ, the smaller is padded with white (255).
pub fn ssim_global(a: &GrayImage, b: &GrayImage) -> f32 {
    let w = a.width().max(b.width());
    let h = a.height().max(b.height());
    if w == 0 || h == 0 {
        return 0.0;
    }

    let c1: f64 = (0.01 * 255.0_f64).powi(2);
    let c2: f64 = (0.03 * 255.0_f64).powi(2);
    let n = (w as u64 * h as u64) as f64;
    if n == 0.0 {
        return 0.0;
    }

    let get_a = |x: u32, y: u32| -> f64 {
        if x < a.width() && y < a.height() {
            a.get_pixel(x, y).0[0] as f64
        } else {
            255.0
        }
    };
    let get_b = |x: u32, y: u32| -> f64 {
        if x < b.width() && y < b.height() {
            b.get_pixel(x, y).0[0] as f64
        } else {
            255.0
        }
    };

    let (mut sa, mut sb, mut sa2, mut sb2, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for y in 0..h {
        for x in 0..w {
            let va = get_a(x, y);
            let vb = get_b(x, y);
            sa += va;
            sb += vb;
            sa2 += va * va;
            sb2 += vb * vb;
            sab += va * vb;
        }
    }

    let mu_a = sa / n;
    let mu_b = sb / n;
    let sig_a2 = (sa2 / n) - mu_a * mu_a;
    let sig_b2 = (sb2 / n) - mu_b * mu_b;
    let sig_ab = (sab / n) - mu_a * mu_b;

    let num = (2.0 * mu_a * mu_b + c1) * (2.0 * sig_ab + c2);
    let den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a2 + sig_b2 + c2);
    if den < 1e-10 {
        return 0.0;
    }
    (num / den).clamp(0.0, 1.0) as f32
}

// ---------------------------------------------------------------------------
// Gaussian blur (3×3, σ≈0.7)
// ---------------------------------------------------------------------------

/// 3×3 Gaussian blur with σ≈0.7 (kernel [1,2,1]/4 separable).
pub fn gaussian_blur_3x3(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return img.clone();
    }
    let w_us = w as usize;
    let raw = img.as_raw();

    // Horizontal pass → tmp
    let mut tmp_buf = vec![0u8; w_us * h as usize];
    for y in 0..h as usize {
        let row_off = y * w_us;
        // Left edge
        tmp_buf[row_off] = ((raw[row_off] as u32 * 3 + raw[row_off + 1] as u32 + 2) / 4) as u8;
        // Interior (no bounds check)
        for x in 1..w_us - 1 {
            let v = raw[row_off + x - 1] as u32 + 2 * raw[row_off + x] as u32 + raw[row_off + x + 1] as u32;
            tmp_buf[row_off + x] = ((v + 2) / 4) as u8;
        }
        // Right edge
        let last = w_us - 1;
        tmp_buf[row_off + last] = ((raw[row_off + last - 1] as u32 + raw[row_off + last] as u32 * 3 + 2) / 4) as u8;
    }

    // Vertical pass → out
    let mut out_buf = vec![0u8; w_us * h as usize];
    // Top edge row
    for x in 0..w_us {
        let v = tmp_buf[x] as u32 * 3 + tmp_buf[w_us + x] as u32;
        out_buf[x] = ((v + 2) / 4) as u8;
    }
    // Interior rows (no bounds check)
    for y in 1..h as usize - 1 {
        let prev_off = (y - 1) * w_us;
        let curr_off = y * w_us;
        let next_off = (y + 1) * w_us;
        for x in 0..w_us {
            let v = tmp_buf[prev_off + x] as u32 + 2 * tmp_buf[curr_off + x] as u32 + tmp_buf[next_off + x] as u32;
            out_buf[curr_off + x] = ((v + 2) / 4) as u8;
        }
    }
    // Bottom edge row
    let last_row = (h as usize - 1) * w_us;
    let prev_row = (h as usize - 2) * w_us;
    for x in 0..w_us {
        let v = tmp_buf[prev_row + x] as u32 + tmp_buf[last_row + x] as u32 * 3;
        out_buf[last_row + x] = ((v + 2) / 4) as u8;
    }

    GrayImage::from_raw(w, h, out_buf).expect("blur output size mismatch")
}

// ---------------------------------------------------------------------------
// Windowed SSIM with vertical shift search
// ---------------------------------------------------------------------------

/// Precomputed 11×11 Gaussian kernel with sigma ≈ 1.5.
/// Computed once on first access, then reused.
fn gaussian_kernel_11x11() -> &'static [[f64; 11]; 11] {
    use std::sync::OnceLock;
    static KERNEL: OnceLock<[[f64; 11]; 11]> = OnceLock::new();
    KERNEL.get_or_init(|| {
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
        for row in &mut kernel {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        kernel
    })
}

/// Try vertical shifts of the rendered image from -max_shift to +max_shift
/// pixels and return the best (highest) SSIM and the shift that produced it.
/// Positive dy = rendered image moved DOWN.
/// Searches from center outward (0, -1, 1, -2, 2, …) and exits early if SSIM ≥ 0.92.
pub fn ssim_windowed_best_vshift(a: &GrayImage, b: &GrayImage, max_shift: i32, bail_below: Option<f32>) -> (f32, i32) {
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
        let score = ssim_windowed(a, b, dy, bail_below);
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

/// Windowed SSIM on grayscale images with a vertical shift applied to image b.
///
/// - 11×11 Gaussian-weighted windows, stepped by 4 pixels
/// - Only windows containing ink (pixels < 240 in either image) contribute
/// - Falls back to global SSIM for images smaller than 11×11
/// - `bail_below`: if Some(threshold), bail early when the running average
///   drops below threshold after processing each row of windows.
pub fn ssim_windowed(a: &GrayImage, b: &GrayImage, b_dy: i32, bail_below: Option<f32>) -> f32 {
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

        // Early bail: if running average is clearly below threshold, stop
        if let Some(bail_thresh) = bail_below {
            if window_count >= 8 {
                let running = (ssim_sum / window_count as f64) as f32;
                if running < bail_thresh {
                    return running.clamp(0.0, 1.0);
                }
            }
        }

        cy += step;
    }

    if window_count == 0 {
        // No ink windows found — fall back to global
        return ssim_global(a, b);
    }

    (ssim_sum / window_count as f64).clamp(0.0, 1.0) as f32
}

// ---------------------------------------------------------------------------
// Image trimming
// ---------------------------------------------------------------------------

/// Trim whitespace using ink-density ratios. Rows/columns with less than 1%
/// ink pixels are considered empty. Preserves dots on j/i, diacritics, and
/// descenders.
pub fn trim_whitespace(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w == 0 || h < 6 {
        return img.clone();
    }

    const INK_THRESH: u8 = 230;
    const MIN_INK_ROW: f32 = 0.01;
    const MIN_INK_COL: f32 = 0.01;

    // Vertical: find first/last ink rows
    let row_ink: Vec<f32> = (0..h).map(|y| {
        let dark: u32 = (0..w).map(|x| {
            if img.get_pixel(x, y).0[0] < INK_THRESH { 1u32 } else { 0 }
        }).sum();
        dark as f32 / w as f32
    }).collect();

    let first_row = match row_ink.iter().position(|&d| d > MIN_INK_ROW) {
        Some(y) => y as u32,
        None => return img.clone(),
    };
    let last_row = match row_ink.iter().rposition(|&d| d > MIN_INK_ROW) {
        Some(y) => y as u32,
        None => return img.clone(),
    };

    // Horizontal: find first/last ink columns
    let col_ink: Vec<f32> = (0..w).map(|x| {
        let dark: u32 = (0..h).map(|y| {
            if img.get_pixel(x, y).0[0] < INK_THRESH { 1u32 } else { 0 }
        }).sum();
        dark as f32 / h as f32
    }).collect();

    let first_col = match col_ink.iter().position(|&d| d > MIN_INK_COL) {
        Some(x) => x as u32,
        None => 0,
    };
    let last_col = match col_ink.iter().rposition(|&d| d > MIN_INK_COL) {
        Some(x) => x as u32,
        None => w - 1,
    };

    let band_h = last_row - first_row + 1;
    let band_w = last_col - first_col + 1;
    if band_h < 4 || band_w < 4 {
        return img.clone();
    }

    image::imageops::crop_imm(img, first_col, first_row, band_w, band_h).to_image()
}

/// Simple whitespace trim: crop to the tight bounding box of all pixels below
/// threshold 240. No density filtering — any single dark pixel extends the box.
pub fn trim_whitespace_simple(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }

    let thresh = 240u8;
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;

    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y).0[0] < thresh {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }

    if min_x > max_x || min_y > max_y {
        return img.clone();
    }

    image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        .to_image()
}

// ---------------------------------------------------------------------------
// SSIM compare (trim + resize + global SSIM)
// ---------------------------------------------------------------------------

/// Result from SSIM comparison including the actual processed images.
pub struct SsimResult {
    pub score: f32,
    pub dy: i32,
    /// The crop image as actually compared (trimmed + resized).
    pub crop_compared: GrayImage,
    /// The render image as actually compared (trimmed + resized).
    pub render_compared: GrayImage,
}

/// SSIM with ink-band normalization: trim both images to their ink content,
/// resize to the same height, then compare.
pub fn ssim_compare(crop: &GrayImage, render: &GrayImage) -> SsimResult {
    let a_trimmed = trim_whitespace(crop);
    let b_trimmed = trim_whitespace(render);

    if a_trimmed.width() == 0 || a_trimmed.height() == 0
        || b_trimmed.width() == 0 || b_trimmed.height() == 0
    {
        return SsimResult {
            score: 0.0, dy: 0,
            crop_compared: a_trimmed.clone(),
            render_compared: b_trimmed.clone(),
        };
    }

    // Use the larger dimensions so neither gets upscaled much
    let target_w = a_trimmed.width().max(b_trimmed.width());
    let target_h = a_trimmed.height().max(b_trimmed.height());

    let a_resized = image::imageops::resize(
        &a_trimmed, target_w, target_h, image::imageops::FilterType::Lanczos3,
    );
    let b_resized = image::imageops::resize(
        &b_trimmed, target_w, target_h, image::imageops::FilterType::Lanczos3,
    );

    let score = ssim_global(&a_resized, &b_resized);
    SsimResult {
        score,
        dy: 0,
        crop_compared: a_resized,
        render_compared: b_resized,
    }
}

// ---------------------------------------------------------------------------
// ZNCC — Zero-mean Normalised Cross-Correlation
// ---------------------------------------------------------------------------

/// Global ZNCC with vertical shift search and Cauchy-Schwarz early bail.
///
/// For each candidate vertical shift, computes global (whole-image) ZNCC
/// in two passes:
///   Pass 1: compute per-image means and variances (no bail — cheap).
///   Pass 2: accumulate cross-covariance with early bail using a
///           Cauchy-Schwarz upper bound on the remaining contribution.
///
/// Tries vertical offsets from 0 outward, keeps the best score, and
/// exits early when a strong match is found.
pub fn zncc_windowed_best_vshift(
    a: &GrayImage,
    b: &GrayImage,
    max_shift: i32,
    bail_below: Option<f32>,
) -> (f32, i32) {
    const EARLY_EXIT: f32 = 0.96;
    let mut best = -1.0f32;
    let mut best_dy = 0i32;

    let mut shifts = Vec::with_capacity((2 * max_shift + 1) as usize);
    shifts.push(0i32);
    for d in 1..=max_shift {
        shifts.push(-d);
        shifts.push(d);
    }

    for dy in shifts {
        // bail_below is in normalized [0,1] space; convert to raw [-1,1] for zncc_global_bailable
        let raw_bail = bail_below.map(|t| t * 2.0 - 1.0);
        let score = zncc_global_bailable(a, b, dy, raw_bail);
        if score > best {
            best = score;
            best_dy = dy;
            if best >= EARLY_EXIT {
                break;
            }
        }
    }
    // Map from [-1, 1] to [0, 1] for compatibility with SSIM thresholds
    let clamped = best.clamp(-1.0, 1.0);
    let normalized = (clamped + 1.0) / 2.0;
    (normalized, best_dy)
}

/// Global ZNCC with vertical shift and Cauchy-Schwarz early bail.
///
/// Pass 1: compute μ_a, μ_b, σ_a², σ_b² over ink pixels.
/// Pass 2: accumulate Σ(a-μ_a)(b-μ_b) row by row; after each row,
///         compute an upper bound on the final ZNCC. If the upper bound
///         falls below `bail_below`, return early.
fn zncc_global_bailable(
    a: &GrayImage,
    b: &GrayImage,
    b_dy: i32,
    bail_below: Option<f32>,
) -> f32 {
    let (w, h) = a.dimensions();
    let bw = b.width() as i32;
    let bh = b.height() as i32;

    // Collect all pixel pairs (including background — global ZNCC needs
    // the full image to avoid being dominated by edge noise).
    let mut pixels: Vec<(f64, f64)> = Vec::with_capacity((w * h) as usize);

    for y in 0..h {
        let by = y as i32 + b_dy;
        for x in 0..w {
            let va = a.get_pixel(x, y).0[0] as f64;
            let vb = if by >= 0 && by < bh && (x as i32) < bw {
                b.get_pixel(x, by as u32).0[0] as f64
            } else {
                255.0
            };
            pixels.push((va, vb));
        }
    }

    let n = pixels.len();
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;

    // Pass 1: means, variances, and per-pixel squared deviations.
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;
    for &(va, vb) in &pixels {
        sum_a += va;
        sum_b += vb;
        sum_a2 += va * va;
        sum_b2 += vb * vb;
    }
    let mu_a = sum_a / nf;
    let mu_b = sum_b / nf;
    let total_var_a = sum_a2 - sum_a * sum_a / nf;  // = Σ(a - μ_a)²
    let total_var_b = sum_b2 - sum_b * sum_b / nf;  // = Σ(b - μ_b)²
    let denom = (total_var_a * total_var_b).sqrt();
    if denom < 1e-10 {
        return 1.0; // both constant → perfect match
    }

    // Pass 2: accumulate cross-covariance with Cauchy-Schwarz bail.
    let mut partial_cov = 0.0f64;
    let mut partial_var_a = 0.0f64;
    let mut partial_var_b = 0.0f64;

    // Check every ~200 pixels (cheap check, tight bound)
    let check_interval = 200.min(n / 4).max(1);

    for (i, &(va, vb)) in pixels.iter().enumerate() {
        let da = va - mu_a;
        let db = vb - mu_b;
        partial_cov += da * db;
        partial_var_a += da * da;
        partial_var_b += db * db;

        if let Some(bail_thresh) = bail_below {
            if (i + 1) % check_interval == 0 && i + 1 < n {
                // Cauchy-Schwarz upper bound on remaining covariance
                let rem_var_a = (total_var_a - partial_var_a).max(0.0);
                let rem_var_b = (total_var_b - partial_var_b).max(0.0);
                let max_rem_cov = (rem_var_a * rem_var_b).sqrt();
                let max_zncc = ((partial_cov + max_rem_cov) / denom) as f32;
                if max_zncc < bail_thresh {
                    return max_zncc.clamp(-1.0, 1.0);
                }
            }
        }
    }

    (partial_cov / denom).clamp(-1.0, 1.0) as f32
}

/// Windowed ZNCC on grayscale images with a vertical shift applied to b.
///
/// Uses 11×11 Gaussian-weighted windows stepped by 4 pixels, only counting
/// windows that contain ink in either image.
///
/// ZNCC per window: Σ w·(a-μa)·(b-μb) / sqrt(Σ w·(a-μa)² · Σ w·(b-μb)²)
pub fn zncc_windowed(a: &GrayImage, b: &GrayImage, b_dy: i32, bail_below: Option<f32>) -> f32 {
    let (w, h) = a.dimensions();
    if w < 11 || h < 11 {
        return zncc_global(a, b);
    }

    let kernel = gaussian_kernel_11x11();
    const INK_THRESHOLD: u8 = 240;
    const MIN_INK_PIXELS: u32 = 3;

    let half = 5i32;
    let bw = b.width() as i32;
    let bh = b.height() as i32;

    let mut zncc_sum = 0.0f64;
    let mut window_count = 0u64;
    let step = 4u32;

    let mut cy = half as u32;
    while cy + (half as u32) < h {
        let mut cx = half as u32;
        while cx + (half as u32) < w {
            let mut ink_count = 0u32;
            let mut mu_a = 0.0f64;
            let mut mu_b = 0.0f64;
            let mut sum_wa2 = 0.0f64;
            let mut sum_wb2 = 0.0f64;
            let mut sum_wab = 0.0f64;

            for ky in 0..11u32 {
                let py = (cy as i32 - half + ky as i32) as u32;
                let by = py as i32 + b_dy;
                for kx in 0..11u32 {
                    let px = (cx as i32 - half + kx as i32) as u32;
                    let va_u8 = a.get_pixel(px, py).0[0];
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
                // Weighted variance/covariance via one-pass: sig² = E[x²] - E[x]²
                let var_a = sum_wa2 - mu_a * mu_a;
                let var_b = sum_wb2 - mu_b * mu_b;
                let cov_ab = sum_wab - mu_a * mu_b;

                let denom = (var_a * var_b).sqrt();
                let local_zncc = if denom < 1e-10 {
                    // Both images constant in this window → perfect match
                    1.0
                } else {
                    cov_ab / denom
                };

                zncc_sum += local_zncc;
                window_count += 1;
            }

            cx += step;
        }

        if let Some(bail_thresh) = bail_below {
            if window_count >= 8 {
                let running = (zncc_sum / window_count as f64) as f32;
                if running < bail_thresh {
                    return running.clamp(-1.0, 1.0);
                }
            }
        }

        cy += step;
    }

    if window_count == 0 {
        return zncc_global(a, b);
    }

    (zncc_sum / window_count as f64).clamp(-1.0, 1.0) as f32
}

/// Global ZNCC between two grayscale images.
/// Public wrapper for `zncc_global` — used by ZnccClassifier.
pub fn zncc_global_pub(a: &GrayImage, b: &GrayImage) -> f32 {
    zncc_global(a, b)
}

fn zncc_global(a: &GrayImage, b: &GrayImage) -> f32 {
    let w = a.width().max(b.width()) as usize;
    let h = a.height().max(b.height()) as usize;
    let n = (w * h) as f64;
    if n < 1.0 { return 0.0; }

    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;
    let mut sum_ab = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let va = if (x as u32) < a.width() && (y as u32) < a.height() {
                a.get_pixel(x as u32, y as u32).0[0] as f64
            } else { 255.0 };
            let vb = if (x as u32) < b.width() && (y as u32) < b.height() {
                b.get_pixel(x as u32, y as u32).0[0] as f64
            } else { 255.0 };
            sum_a += va;
            sum_b += vb;
            sum_a2 += va * va;
            sum_b2 += vb * vb;
            sum_ab += va * vb;
        }
    }

    let var_a = sum_a2 / n - (sum_a / n).powi(2);
    let var_b = sum_b2 / n - (sum_b / n).powi(2);
    let cov = sum_ab / n - (sum_a / n) * (sum_b / n);
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 { 1.0 } else { (cov / denom).clamp(-1.0, 1.0) as f32 }
}
