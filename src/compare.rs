//! Diagnostic comparison output: side-by-side scan crop vs rendered font match
//! for every vectorized line. Activated by `--compare`.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma, RgbImage, Rgb as ImgRgb};
use crate::pdf_out::PlacedText;
use std::path::Path;

/// Render a comparison strip for all vectorized lines on a page.
/// Returns a tall RGB image with scan crop / rendered crop pairs.
pub fn generate_comparison(
    gray_page: &GrayImage,
    placed_texts: &[PlacedText],
    page_idx: usize,
    output_dir: &Path,
    font_cache: &crate::font_cache::FontCache,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let (page_w, page_h) = gray_page.dimensions();
    let target_width: u32 = 800;
    let label_height: u32 = 18;
    let separator: u32 = 2;

    let mut panels: Vec<RgbImage> = Vec::new();
    let mut line_idx = 0u32;

    for pt in placed_texts {
        if pt.keep_raster {
            continue;
        }
        let fm = match pt.font_match.as_ref() {
            Some(f) => f,
            None => continue,
        };

        let x = (pt.x as u32).min(page_w.saturating_sub(1));
        let y = (pt.y as u32).min(page_h.saturating_sub(1));
        let w = (pt.width as u32).min(page_w - x).max(1);
        let h = (pt.height as u32).min(page_h - y).max(1);

        // ── Scan crop ────────────────────────────────────────────
        // Add margin to see if ink extends beyond bbox
        let margin: u32 = 8;
        let crop_x = x.saturating_sub(margin);
        let crop_y = y.saturating_sub(margin);
        let crop_w = (w + 2 * margin).min(page_w - crop_x);
        let crop_h = (h + 2 * margin).min(page_h - crop_y);

        let scan_crop = image::imageops::crop_imm(gray_page, crop_x, crop_y, crop_w, crop_h)
            .to_image();

        // ── Rendered crop ────────────────────────────────────────
        // Render the matched font using the same rendering as verify.rs
        let font_data_loaded = match font_cache.load(&fm.font_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let rendered_crop = render_font_crop(
            &font_data_loaded,
            &pt.words,
            x, y, crop_x, crop_y,
            crop_w, crop_h,
            &pt.text,
            fm.glyph_overrides.as_deref(),
        );

        // ── Scale both to target width ───────────────────────────
        let scale_factor = target_width as f32 / crop_w as f32;
        let scaled_h = (crop_h as f32 * scale_factor).ceil() as u32;

        let scan_scaled = image::imageops::resize(
            &scan_crop,
            target_width,
            scaled_h,
            image::imageops::FilterType::Lanczos3,
        );

        let rendered_scaled = image::imageops::resize(
            &rendered_crop,
            target_width,
            scaled_h,
            image::imageops::FilterType::Lanczos3,
        );

        // ── Build panel ──────────────────────────────────────────
        let panel_h = label_height + scaled_h + separator + label_height + scaled_h + separator;
        let mut panel = RgbImage::from_pixel(target_width, panel_h, ImgRgb([255, 255, 255]));

        // Label "SCAN" bar — light blue background
        for py in 0..label_height {
            for px in 0..target_width {
                panel.put_pixel(px, py, ImgRgb([200, 220, 255]));
            }
        }

        // Paste scan crop
        let scan_y_start = label_height;
        for py in 0..scaled_h {
            for px in 0..target_width {
                let gray = scan_scaled.get_pixel(px, py).0[0];
                panel.put_pixel(px, scan_y_start + py, ImgRgb([gray, gray, gray]));
            }
        }

        // Separator line — red
        let sep1_y = scan_y_start + scaled_h;
        for py in 0..separator {
            for px in 0..target_width {
                panel.put_pixel(px, sep1_y + py, ImgRgb([255, 0, 0]));
            }
        }

        // Label "RENDERED" bar — light green background
        let render_label_y = sep1_y + separator;
        for py in 0..label_height {
            for px in 0..target_width {
                panel.put_pixel(px, render_label_y + py, ImgRgb([200, 255, 200]));
            }
        }

        // Paste rendered crop
        let render_y_start = render_label_y + label_height;
        for py in 0..scaled_h.min(panel_h - render_y_start) {
            for px in 0..target_width {
                let gray = rendered_scaled.get_pixel(px, py).0[0];
                panel.put_pixel(px, render_y_start + py, ImgRgb([gray, gray, gray]));
            }
        }

        // Bottom separator — dark gray
        let sep2_y = render_y_start + scaled_h;
        if sep2_y + separator <= panel_h {
            for py in 0..separator {
                for px in 0..target_width {
                    panel.put_pixel(px, sep2_y + py, ImgRgb([100, 100, 100]));
                }
            }
        }

        // Save individual panel
        let text_preview: String = pt.text.chars().take(40).collect();
        let fname = format!(
            "p{}-line{:03}-{}.png",
            page_idx + 1,
            line_idx,
            sanitize_filename(&text_preview),
        );
        let panel_path = output_dir.join(&fname);
        panel.save(&panel_path).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;


        panels.push(panel);
        line_idx += 1;
    }

    // ── Combined strip ───────────────────────────────────────────
    if !panels.is_empty() {
        let total_h: u32 = panels.iter().map(|p| p.height()).sum();
        let mut combined = RgbImage::from_pixel(target_width, total_h, ImgRgb([255, 255, 255]));
        let mut y_off = 0u32;
        for panel in &panels {
            for py in 0..panel.height() {
                for px in 0..target_width.min(panel.width()) {
                    combined.put_pixel(px, y_off + py, *panel.get_pixel(px, py));
                }
            }
            y_off += panel.height();
        }
        let combined_path = output_dir.join(format!("p{}-combined.png", page_idx + 1));
        combined.save(&combined_path).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
    }

    Ok(())
}

/// Render the matched font onto a canvas matching the scan crop dimensions.
/// Uses the same width-matched rendering as verify.rs.
fn render_font_crop(
    font_data: &[u8],
    words: &[crate::pdf_out::WordBox],
    line_x: u32,
    line_y: u32,
    crop_x: u32,
    _crop_y: u32,
    canvas_w: u32,
    canvas_h: u32,
    line_text: &str,
    overrides: Option<&[(char, u16)]>,
) -> GrayImage {
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));

    let font = match FontRef::try_from_slice(font_data) {
        Ok(f) => f,
        Err(_) => return canvas,
    };

    // If we have word-level boxes, render per-word (same as verify.rs)
    if !words.is_empty() {
        for word in words {
            if word.text.is_empty() || word.width < 1.0 {
                continue;
            }

            let word_em_px = match crate::layout::width_matched_em_px(
                &font,
                &word.text,
                word.width,
                overrides,
            ) {
                Some(v) => v,
                None => continue,
            };

            let word_scale = PxScale::from(word_em_px);
            let sf = font.as_scaled(word_scale);

            // Word position relative to crop origin
            let word_x_in_crop = word.x - crop_x as f32;
            // Use the line bbox height for baseline calc, not canvas_h
            let _line_h = (line_y + (word.height as u32)) as f32 - line_y as f32;
            let baseline = crate::layout::ink_centered_baseline_px(
                &font,
                word_em_px,
                canvas_h as f32,
            );

            let mut cx = word_x_in_crop;
            let mut prev: Option<ab_glyph::GlyphId> = None;

            for c in word.text.chars() {
                let gid = crate::char_render::resolve_glyph(&font, c, overrides);
                if let Some(p) = prev {
                    cx += sf.kern(p, gid);
                }
                let glyph = gid.with_scale_and_position(
                    word_scale,
                    point(cx, baseline),
                );
                if let Some(og) = font.outline_glyph(glyph) {
                    let bounds = og.px_bounds();
                    let bx = bounds.min.x as i32;
                    let by = bounds.min.y as i32;
                    og.draw(|gx, gy, cov| {
                        let px = gx as i32 + bx;
                        let py = gy as i32 + by;
                        if px >= 0
                            && py >= 0
                            && (px as u32) < canvas_w
                            && (py as u32) < canvas_h
                        {
                            let val = (255.0 * (1.0 - cov)) as u8;
                            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
                            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
                        }
                    });
                }
                cx += sf.h_advance(gid);
                prev = Some(gid);
            }
        }
    } else {
        // Fallback: render whole line text at line-level width match
        let em_px = match crate::layout::width_matched_em_px(
            &font,
            line_text,
            (line_x + (canvas_w - 8)) as f32 - line_x as f32,
            overrides,
        ) {
            Some(v) => v,
            None => return canvas,
        };

        let scale = PxScale::from(em_px);
        let sf = font.as_scaled(scale);
        let baseline = crate::layout::ink_centered_baseline_px(&font, em_px, canvas_h as f32);

        // Start at left margin
        let margin = 4.0f32;
        let mut cx = margin;
        let mut prev: Option<ab_glyph::GlyphId> = None;

        for c in line_text.chars() {
            let gid = crate::char_render::resolve_glyph(&font, c, overrides);
            if let Some(p) = prev {
                cx += sf.kern(p, gid);
            }
            let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
            if let Some(og) = font.outline_glyph(glyph) {
                let bounds = og.px_bounds();
                let bx = bounds.min.x as i32;
                let by = bounds.min.y as i32;
                og.draw(|gx, gy, cov| {
                    let px = gx as i32 + bx;
                    let py = gy as i32 + by;
                    if px >= 0
                        && py >= 0
                        && (px as u32) < canvas_w
                        && (py as u32) < canvas_h
                    {
                        let val = (255.0 * (1.0 - cov)) as u8;
                        let cur = canvas.get_pixel(px as u32, py as u32).0[0];
                        canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
                    }
                });
            }
            cx += sf.h_advance(gid);
            prev = Some(gid);
        }
    }

    canvas
}

/// Sanitize a string for use as a filename.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(30)
        .collect()
}
