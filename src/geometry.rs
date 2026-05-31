//! Geometry vectorisation — detect horizontal / vertical lines, rectangles,
//! and solid-colour fill regions in the page raster so they can be replaced
//! with native PDF vector primitives.
//!
//! Approach: run-length analysis for axis-aligned lines, then rectangle
//! assembly from intersecting H/V lines, plus variance-based solid-fill
//! detection on a coarse grid.

use crate::color::Rgb;
use image::DynamicImage;
use log::debug;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DetectedLine {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
    pub thickness: u32,
    pub color: Rgb,
}

#[derive(Debug, Clone)]
pub struct DetectedFill {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub color: Rgb,
}

/// Everything detected on one page.
#[derive(Debug, Clone)]
pub struct GeometryResult {
    pub lines: Vec<DetectedLine>,
    pub fills: Vec<DetectedFill>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect vectorisable geometry on `img`, ignoring pixels that fall inside
/// any of the supplied `text_bboxes` (x, y, w, h).
pub fn detect_geometry(
    img: &DynamicImage,
    text_bboxes: &[(u32, u32, u32, u32)],
    min_line_length_px: u32,
) -> GeometryResult {
    let gray = img.to_luma8();
    let rgba = img.to_rgba8();
    let (w, h) = gray.dimensions();

    // Build a mask of text regions so we skip them.
    let mut text_mask = vec![false; (w as usize) * (h as usize)];
    for &(bx, by, bw, bh) in text_bboxes {
        let x0 = bx.min(w);
        let y0 = by.min(h);
        let x1 = (bx + bw).min(w);
        let y1 = (by + bh).min(h);
        for py in y0..y1 {
            for px in x0..x1 {
                text_mask[(py as usize) * (w as usize) + (px as usize)] = true;
            }
        }
    }

    let is_text = |x: u32, y: u32| -> bool {
        text_mask[(y as usize) * (w as usize) + (x as usize)]
    };

    // Binarise with Otsu
    let threshold = otsu_threshold(&gray);

    // ── Horizontal lines ─────────────────────────────────────────────
    let mut h_lines: Vec<DetectedLine> = Vec::new();
    for y in 0..h {
        let mut run_start: Option<u32> = None;
        for x in 0..=w {
            let dark = if x < w {
                gray.get_pixel(x, y).0[0] <= threshold && !is_text(x, y)
            } else {
                false
            };
            if dark {
                if run_start.is_none() {
                    run_start = Some(x);
                }
            } else if let Some(start) = run_start {
                let len = x - start;
                if len >= min_line_length_px {
                    let mid_x = start + len / 2;
                    let px = rgba.get_pixel(mid_x.min(w - 1), y);
                    h_lines.push(DetectedLine {
                        x1: start,
                        y1: y,
                        x2: x.saturating_sub(1),
                        y2: y,
                        thickness: 1,
                        color: (px.0[0], px.0[1], px.0[2]),
                    });
                }
                run_start = None;
            }
        }
    }

    // Merge adjacent horizontal runs into thicker lines.
    let h_lines = merge_horizontal(&mut h_lines);

    // ── Vertical lines ───────────────────────────────────────────────
    let mut v_lines: Vec<DetectedLine> = Vec::new();
    for x in 0..w {
        let mut run_start: Option<u32> = None;
        for y in 0..=h {
            let dark = if y < h {
                gray.get_pixel(x, y).0[0] <= threshold && !is_text(x, y)
            } else {
                false
            };
            if dark {
                if run_start.is_none() {
                    run_start = Some(y);
                }
            } else if let Some(start) = run_start {
                let len = y - start;
                if len >= min_line_length_px {
                    let mid_y = start + len / 2;
                    let px = rgba.get_pixel(x, mid_y.min(h - 1));
                    v_lines.push(DetectedLine {
                        x1: x,
                        y1: start,
                        x2: x,
                        y2: y.saturating_sub(1),
                        thickness: 1,
                        color: (px.0[0], px.0[1], px.0[2]),
                    });
                }
                run_start = None;
            }
        }
    }

    let v_lines = merge_vertical(&mut v_lines);

    let mut all_lines = h_lines;
    all_lines.extend(v_lines);

    debug!("  geometry: {} lines detected", all_lines.len());

    // ── Solid fills ──────────────────────────────────────────────────
    let fills = detect_fills(img, &text_mask, w, h);
    debug!("  geometry: {} fills detected", fills.len());

    GeometryResult {
        lines: all_lines,
        fills,
    }
}

/// Return (x, y, w, h) bounding boxes for every detected geometry element
/// so the caller can erase them from the raster.
pub fn erase_bboxes(geo: &GeometryResult) -> Vec<(u32, u32, u32, u32)> {
    let mut out = Vec::new();
    for l in &geo.lines {
        let x = l.x1.min(l.x2);
        let y = l.y1.min(l.y2);
        let w = l.x1.abs_diff(l.x2) + l.thickness;
        let h = l.y1.abs_diff(l.y2) + l.thickness;
        out.push((x, y, w, h));
    }
    for f in &geo.fills {
        out.push((f.x, f.y, f.width, f.height));
    }
    out
}

// ---------------------------------------------------------------------------
// Merging
// ---------------------------------------------------------------------------

/// Merge single-pixel horizontal runs on adjacent rows into thicker lines.
fn merge_horizontal(runs: &mut Vec<DetectedLine>) -> Vec<DetectedLine> {
    if runs.is_empty() {
        return Vec::new();
    }
    runs.sort_by_key(|l| (l.x1, l.y1));

    let mut merged: Vec<DetectedLine> = Vec::new();
    let mut current = runs[0].clone();

    for r in runs.iter().skip(1) {
        // Same horizontal span (within 2px tolerance) and adjacent row?
        if r.y1 <= current.y2 + 2
            && r.x1.abs_diff(current.x1) <= 2
            && r.x2.abs_diff(current.x2) <= 2
        {
            current.y2 = r.y2;
            current.thickness = if current.y2 >= current.y1 { current.y2 - current.y1 + 1 } else { 1 };
        } else {
            merged.push(current.clone());
            current = r.clone();
        }
    }
    merged.push(current);
    merged
}

/// Merge single-pixel vertical runs on adjacent columns into thicker lines.
fn merge_vertical(runs: &mut Vec<DetectedLine>) -> Vec<DetectedLine> {
    if runs.is_empty() {
        return Vec::new();
    }
    runs.sort_by_key(|l| (l.y1, l.x1));

    let mut merged: Vec<DetectedLine> = Vec::new();
    let mut current = runs[0].clone();

    for r in runs.iter().skip(1) {
        if r.x1 <= current.x2 + 2
            && r.y1.abs_diff(current.y1) <= 2
            && r.y2.abs_diff(current.y2) <= 2
        {
            current.x2 = r.x2;
            current.thickness = if current.x2 >= current.x1 { current.x2 - current.x1 + 1 } else { 1 };
        } else {
            merged.push(current.clone());
            current = r.clone();
        }
    }
    merged.push(current);
    merged
}

// ---------------------------------------------------------------------------
// Solid fill detection
// ---------------------------------------------------------------------------

/// Find large rectangular regions of uniform non-white colour.
fn detect_fills(
    img: &DynamicImage,
    text_mask: &[bool],
    w: u32,
    h: u32,
) -> Vec<DetectedFill> {
    let rgba = img.to_rgba8();
    let cell = 50u32; // grid cell size
    let cols = (w + cell - 1) / cell;
    let rows = (h + cell - 1) / cell;

    // For each cell compute (mean_r, mean_g, mean_b, variance, has_text).
    struct CellInfo {
        r: u8,
        g: u8,
        b: u8,
        uniform: bool,
        bg: bool, // close to white → skip
    }

    let mut cells: Vec<CellInfo> = Vec::with_capacity((cols * rows) as usize);

    for row in 0..rows {
        for col in 0..cols {
            let cx = col * cell;
            let cy = row * cell;
            let cw = cell.min(w - cx);
            let ch = cell.min(h - cy);

            let (mut sr, mut sg, mut sb, mut cnt) = (0u64, 0u64, 0u64, 0u64);
            let mut has_text = false;
            let _var_sum = 0u64;
            let mut gray_sum = 0u64;
            let mut gray_sq = 0u64;

            for dy in 0..ch {
                for dx in 0..cw {
                    let px = cx + dx;
                    let py = cy + dy;
                    if text_mask[(py as usize) * (w as usize) + (px as usize)] {
                        has_text = true;
                    }
                    let p = rgba.get_pixel(px, py);
                    sr += p.0[0] as u64;
                    sg += p.0[1] as u64;
                    sb += p.0[2] as u64;
                    let g = (p.0[0] as u64 * 30 + p.0[1] as u64 * 59 + p.0[2] as u64 * 11) / 100;
                    gray_sum += g;
                    gray_sq += g * g;
                    cnt += 1;
                }
            }

            if cnt == 0 || has_text {
                cells.push(CellInfo { r: 255, g: 255, b: 255, uniform: false, bg: true });
                continue;
            }

            let mr = (sr / cnt) as u8;
            let mg = (sg / cnt) as u8;
            let mb = (sb / cnt) as u8;
            let mean_g = gray_sum as f64 / cnt as f64;
            let var = (gray_sq as f64 / cnt as f64) - mean_g * mean_g;

            // "Uniform" = low variance; "bg" = close to white (skip).
            let is_bg = mr > 240 && mg > 240 && mb > 240;
            cells.push(CellInfo {
                r: mr, g: mg, b: mb,
                uniform: var < 60.0,
                bg: is_bg,
            });
        }
    }

    // BFS to find connected components of uniform, non-bg cells with similar colour.
    let mut visited = vec![false; cells.len()];
    let mut fills = Vec::new();

    for idx in 0..cells.len() {
        if visited[idx] || !cells[idx].uniform || cells[idx].bg {
            continue;
        }
        visited[idx] = true;
        let seed = &cells[idx];
        let sr = seed.r;
        let sg = seed.g;
        let sb = seed.b;

        let mut queue = vec![idx];
        let mut min_c = (idx % cols as usize) as u32;
        let mut max_c = min_c;
        let mut min_r = (idx / cols as usize) as u32;
        let mut max_r = min_r;

        while let Some(ci) = queue.pop() {
            let cr = (ci / cols as usize) as u32;
            let cc = (ci % cols as usize) as u32;
            min_r = min_r.min(cr);
            max_r = max_r.max(cr);
            min_c = min_c.min(cc);
            max_c = max_c.max(cc);

            for (dr, dc) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nr = cr as i32 + dr;
                let nc = cc as i32 + dc;
                if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                    continue;
                }
                let ni = (nr as u32 * cols + nc as u32) as usize;
                if visited[ni] || !cells[ni].uniform || cells[ni].bg {
                    continue;
                }
                // Colour similarity
                let c = &cells[ni];
                if (c.r as i32 - sr as i32).unsigned_abs() > 15
                    || (c.g as i32 - sg as i32).unsigned_abs() > 15
                    || (c.b as i32 - sb as i32).unsigned_abs() > 15
                {
                    continue;
                }
                visited[ni] = true;
                queue.push(ni);
            }
        }

        let fx = min_c * cell;
        let fy = min_r * cell;
        let fw = ((max_c + 1) * cell).min(w) - fx;
        let fh = ((max_r + 1) * cell).min(h) - fy;

        // Only keep if the region is at least 2 cells in each dimension.
        if fw >= cell * 2 && fh >= cell * 2 {
            fills.push(DetectedFill {
                x: fx,
                y: fy,
                width: fw,
                height: fh,
                color: (sr, sg, sb),
            });
        }
    }

    fills
}

// ---------------------------------------------------------------------------
// Otsu threshold
// ---------------------------------------------------------------------------

fn otsu_threshold(gray: &image::GrayImage) -> u8 {
    let mut hist = [0u32; 256];
    for p in gray.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let total = gray.width() * gray.height();
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
        if w_bg == 0 { continue; }
        let w_fg = total - w_bg;
        if w_fg == 0 { break; }
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
