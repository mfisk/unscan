//! Geometry vectorisation — detect horizontal / vertical lines, rectangles,
//! and solid-colour fill regions in the page raster so they can be replaced
//! with native PDF vector primitives.
//!
//! Optimized: raw buffer access, no get_pixel, histogram Otsu without Vec alloc.

use crate::color::Rgb;
use image::{DynamicImage, GrayImage, RgbaImage};

pub use unprint_geometry::{DetectedLine, DetectedFill, GeometryResult};

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
    detect_geometry_from_buffers(&gray, &rgba, text_bboxes, min_line_length_px)
}

pub fn detect_geometry_from_buffers(
    gray: &GrayImage,
    rgba: &RgbaImage,
    text_bboxes: &[(u32, u32, u32, u32)],
    min_line_length_px: u32,
) -> GeometryResult {
    let (w_u32, h_u32) = gray.dimensions();
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    if w == 0 || h == 0 {
        return GeometryResult { lines: Vec::new(), fills: Vec::new() };
    }

    let gray_raw = gray.as_raw();
    let rgba_raw = rgba.as_raw();

    // Build a mask of text regions so we skip them.
    let mut text_mask = vec![false; w * h];
    for &(bx, by, bw, bh) in text_bboxes {
        let x0 = (bx.min(w_u32)) as usize;
        let y0 = (by.min(h_u32)) as usize;
        let x1 = ((bx + bw).min(w_u32)) as usize;
        let y1 = ((by + bh).min(h_u32)) as usize;
        if x1 <= x0 || y1 <= y0 { continue; }
        for py in y0..y1 {
            let base = py * w + x0;
            for px in x0..x1 {
                text_mask[base + (px - x0)] = true;
            }
        }
    }

    // Binarise with Otsu — histogram from raw, no vals Vec
    let mut hist = [0u32; 256];
    for &v in gray_raw {
        hist[v as usize] += 1;
    }
    let threshold = crate::color::otsu_from_hist(&hist, gray_raw.len());

    // ── Horizontal lines ─────────────────────────────────────────────
    let mut h_lines: Vec<DetectedLine> = Vec::new();
    for y in 0..h {
        let row_base = y * w;
        let mut run_start: Option<usize> = None;
        for x in 0..=w {
            let dark = if x < w {
                let idx = row_base + x;
                !text_mask[idx] && gray_raw[idx] <= threshold
            } else {
                false
            };
            if dark {
                if run_start.is_none() {
                    run_start = Some(x);
                }
            } else if let Some(start) = run_start {
                let len = x - start;
                if len >= min_line_length_px as usize {
                    let mid_x = start + len / 2;
                    let mid_idx = row_base + mid_x.min(w - 1);
                    let j = mid_idx * 4;
                    // rgba_raw has 4 bytes per pixel
                    let r = rgba_raw[j];
                    let g = rgba_raw[j + 1];
                    let b = rgba_raw[j + 2];
                    h_lines.push(DetectedLine {
                        x1: start as u32,
                        y1: y as u32,
                        x2: (x as u32).saturating_sub(1),
                        y2: y as u32,
                        thickness: 1,
                        color: (r, g, b),
                    });
                }
                run_start = None;
            }
        }
    }

    let h_lines = merge_horizontal(&mut h_lines);

    // ── Vertical lines ───────────────────────────────────────────────
    let mut v_lines: Vec<DetectedLine> = Vec::new();
    for x in 0..w {
        let mut run_start: Option<usize> = None;
        for y in 0..=h {
            let dark = if y < h {
                let idx = y * w + x;
                !text_mask[idx] && gray_raw[idx] <= threshold
            } else {
                false
            };
            if dark {
                if run_start.is_none() {
                    run_start = Some(y);
                }
            } else if let Some(start) = run_start {
                let len = y - start;
                if len >= min_line_length_px as usize {
                    let mid_y = start + len / 2;
                    let mid_y_clamped = mid_y.min(h - 1);
                    let mid_idx = mid_y_clamped * w + x;
                    let j = mid_idx * 4;
                    let r = rgba_raw[j];
                    let g = rgba_raw[j + 1];
                    let b = rgba_raw[j + 2];
                    v_lines.push(DetectedLine {
                        x1: x as u32,
                        y1: start as u32,
                        x2: x as u32,
                        y2: (y as u32).saturating_sub(1),
                        thickness: 1,
                        color: (r, g, b),
                    });
                }
                run_start = None;
            }
        }
    }

    let v_lines = merge_vertical(&mut v_lines);

    let mut all_lines = h_lines;
    all_lines.extend(v_lines);

    // ── Solid fills ──────────────────────────────────────────────────
    let fills = detect_fills_from_rgba(rgba, &text_mask, w_u32, h_u32);

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
// Merging (unchanged)
// ---------------------------------------------------------------------------

fn merge_horizontal(runs: &mut Vec<DetectedLine>) -> Vec<DetectedLine> {
    if runs.is_empty() {
        return Vec::new();
    }
    runs.sort_by_key(|l| (l.x1, l.y1));
    let mut merged: Vec<DetectedLine> = Vec::new();
    let mut current = runs[0].clone();
    for r in runs.iter().skip(1) {
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
// Solid fill detection — raw buffer version
// ---------------------------------------------------------------------------

fn detect_fills_from_rgba(
    rgba: &RgbaImage,
    text_mask: &[bool],
    w_u32: u32,
    h_u32: u32,
) -> Vec<DetectedFill> {
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    let raw = rgba.as_raw();
    let cell = 50usize;
    let cols = (w + cell - 1) / cell;
    let rows = (h + cell - 1) / cell;

    struct CellInfo {
        r: u8,
        g: u8,
        b: u8,
        uniform: bool,
        bg: bool,
    }

    let mut cells: Vec<CellInfo> = Vec::with_capacity(cols * rows);

    for row in 0..rows {
        for col in 0..cols {
            let cx = col * cell;
            let cy = row * cell;
            let cw = cell.min(w - cx);
            let ch = cell.min(h - cy);

            let (mut sr, mut sg, mut sb, mut cnt) = (0u64, 0u64, 0u64, 0u64);
            let mut has_text = false;
            let mut gray_sum = 0u64;
            let mut gray_sq = 0u64;

            for dy in 0..ch {
                let py = cy + dy;
                let row_base = py * w;
                let rgba_row_base = row_base * 4;
                for dx in 0..cw {
                    let px = cx + dx;
                    let idx = row_base + px;
                    if text_mask[idx] {
                        has_text = true;
                    }
                    let j = rgba_row_base + px * 4;
                    let r = raw[j] as u64;
                    let g = raw[j + 1] as u64;
                    let b = raw[j + 2] as u64;
                    sr += r;
                    sg += g;
                    sb += b;
                    let gv = (r * 30 + g * 59 + b * 11) / 100;
                    gray_sum += gv;
                    gray_sq += gv * gv;
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
        let seed_r = cells[idx].r;
        let seed_g = cells[idx].g;
        let seed_b = cells[idx].b;

        let mut queue = vec![idx];
        let mut min_c = (idx % cols) as u32;
        let mut max_c = min_c;
        let mut min_r = (idx / cols) as u32;
        let mut max_r = min_r;

        while let Some(ci) = queue.pop() {
            let cr = (ci / cols) as u32;
            let cc = (ci % cols) as u32;
            min_r = min_r.min(cr);
            max_r = max_r.max(cr);
            min_c = min_c.min(cc);
            max_c = max_c.max(cc);

            // 4-neighbor
            let neighbors = [
                (cr as i32 - 1, cc as i32),
                (cr as i32 + 1, cc as i32),
                (cr as i32, cc as i32 - 1),
                (cr as i32, cc as i32 + 1),
            ];
            for (nr, nc) in neighbors {
                if nr < 0 || nc < 0 { continue; }
                let nr = nr as usize;
                let nc = nc as usize;
                if nr >= rows || nc >= cols { continue; }
                let ni = nr * cols + nc;
                if visited[ni] { continue; }
                if !cells[ni].uniform || cells[ni].bg { continue; }
                // colour similarity (within 15)
                let dr = (cells[ni].r as i16 - seed_r as i16).abs();
                let dg = (cells[ni].g as i16 - seed_g as i16).abs();
                let db = (cells[ni].b as i16 - seed_b as i16).abs();
                if dr > 15 || dg > 15 || db > 15 { continue; }
                visited[ni] = true;
                queue.push(ni);
            }
        }

        let x = min_c * cell as u32;
        let y = min_r * cell as u32;
        let w_fill = ((max_c - min_c + 1) * cell as u32).min(w_u32 - x);
        let h_fill = ((max_r - min_r + 1) * cell as u32).min(h_u32 - y);
        if w_fill >= cell as u32 && h_fill >= cell as u32 {
            fills.push(DetectedFill {
                x, y,
                width: w_fill,
                height: h_fill,
                color: (seed_r, seed_g, seed_b),
            });
        }
    }

    fills
}

// Backward compat wrapper that was previously used — now delegates to raw version.
fn detect_fills(
    img: &DynamicImage,
    text_mask: &[bool],
    w: u32,
    h: u32,
) -> Vec<DetectedFill> {
    let rgba = img.to_rgba8();
    detect_fills_from_rgba(&rgba, text_mask, w, h)
}
