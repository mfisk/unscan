//! OCR module — runs Tesseract via CLI to extract text with bounding boxes.

use crate::error::ScanTextError;
use image::{DynamicImage, GrayImage};
use log::{debug, info};
use std::process::Command;

/// A character-level bounding box from Tesseract HOCR output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CharBox {
    pub ch: char,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

/// A detected text region from OCR (word-level).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextRegion {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub font_size_pt: f32,
    pub confidence: f32,
    pub level: u32,
    pub block_num: u32,
    pub par_num: u32,
    pub line_num: u32,
    #[allow(dead_code)]
    pub word_num: u32,
}

/// A line of text assembled from word-level regions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextLine {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub font_size_pt: f32,
    pub confidence: f32,
    pub words: Vec<TextRegion>,
}

/// Run Tesseract on a page image and return word-level regions plus
/// character-level bounding boxes (from makebox output).
/// Applies contrast enhancement and sharpening before OCR to improve
/// recognition on scanned documents.
pub fn extract_text_regions(
    page_img: &DynamicImage,
    dpi: u32,
) -> Result<(Vec<TextRegion>, Vec<CharBox>), ScanTextError> {
    let tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .map_err(ScanTextError::Io)?;

    // Pre-process: convert to grayscale, then enhance contrast via
    // adaptive normalisation.  This helps Tesseract on noisy scans.
    let gray = page_img.to_luma8();
    let (w, h) = gray.dimensions();
    let mut enhanced = gray.clone();

    // Simple global contrast stretch: map [min, max] → [0, 255]
    let mut lo = 255u8;
    let mut hi = 0u8;
    for p in gray.pixels() {
        lo = lo.min(p.0[0]);
        hi = hi.max(p.0[0]);
    }
    if hi > lo {
        let range = (hi - lo) as f32;
        for y_px in 0..h {
            for x_px in 0..w {
                let v = gray.get_pixel(x_px, y_px).0[0];
                let stretched = ((v as f32 - lo as f32) / range * 255.0) as u8;
                enhanced.put_pixel(x_px, y_px, image::Luma([stretched]));
            }
        }
    }

    // Sharpen via unsharp mask: out = 1.5*original - 0.5*blurred
    let blurred = image::imageops::blur(&enhanced, 1.0);
    let mut sharpened = enhanced.clone();
    for y_px in 0..h {
        for x_px in 0..w {
            let orig = enhanced.get_pixel(x_px, y_px).0[0] as f32;
            let blur_v = blurred.get_pixel(x_px, y_px).0[0] as f32;
            let sharp = (1.5 * orig - 0.5 * blur_v).clamp(0.0, 255.0) as u8;
            sharpened.put_pixel(x_px, y_px, image::Luma([sharp]));
        }
    }

    DynamicImage::ImageLuma8(sharpened)
        .save(tmp.path())
        .map_err(|e| ScanTextError::Ocr(format!("Failed to save temp image: {e}")))?;

    let output = Command::new("tesseract")
        .args([
            tmp.path().to_str().unwrap(),
            "stdout",
            "--dpi",
            &dpi.to_string(),
            "-l",
            "eng",
            "tsv",
        ])
        .output()
        .map_err(|e| {
            ScanTextError::Ocr(format!(
                "Failed to run tesseract (install tesseract-ocr): {e}"
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ScanTextError::Ocr(format!("tesseract failed: {stderr}")));
    }

    let regions = parse_tsv(&String::from_utf8_lossy(&output.stdout), dpi)?;

    // Second pass: get character-level bounding boxes via HOCR
    // HOCR with hocr_char_boxes=1 gives per-character bboxes with confidence,
    // structurally nested inside words (eliminates image-area contamination).
    let hocr_output = Command::new("tesseract")
        .args([
            tmp.path().to_str().unwrap(),
            "stdout",
            "--dpi",
            &dpi.to_string(),
            "-l",
            "eng",
            "-c",
            "hocr_char_boxes=1",
            "hocr",
        ])
        .output()
        .map_err(|e| {
            ScanTextError::Ocr(format!(
                "Failed to run tesseract hocr: {e}"
            ))
        })?;

    let char_boxes = if hocr_output.status.success() {
        parse_hocr(&String::from_utf8_lossy(&hocr_output.stdout))
    } else {
        debug!("  HOCR pass failed, no character boxes");
        Vec::new()
    };

    info!("  OCR: {} words, {} char boxes from HOCR", regions.len(), char_boxes.len());

    Ok((regions, char_boxes))
}

/// Group word regions into lines using Tesseract's block/par/line numbering.
pub fn assemble_lines(words: &[TextRegion]) -> Vec<TextLine> {
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<(u32, u32, u32), Vec<&TextRegion>> = BTreeMap::new();
    for w in words {
        if w.level == 5 && !w.text.trim().is_empty() {
            groups
                .entry((w.block_num, w.par_num, w.line_num))
                .or_default()
                .push(w);
        }
    }

    let mut lines = Vec::new();
    for (_, mut gw) in groups {
        gw.sort_by_key(|w| w.x);
        let text = gw.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        let x = gw.iter().map(|w| w.x).min().unwrap_or(0);
        let y = gw.iter().map(|w| w.y).min().unwrap_or(0);
        let x_max = gw.iter().map(|w| w.x + w.width).max().unwrap_or(0);
        let y_max = gw.iter().map(|w| w.y + w.height).max().unwrap_or(0);
        let avg_conf = gw.iter().map(|w| w.confidence).sum::<f32>() / gw.len() as f32;
        let font_size_pt = gw.first().map(|w| w.font_size_pt).unwrap_or(12.0);

        lines.push(TextLine {
            text,
            x,
            y,
            width: x_max.saturating_sub(x),
            height: y_max.saturating_sub(y),
            font_size_pt,
            confidence: avg_conf,
            words: gw.into_iter().cloned().collect(),
        });
    }
    lines
}

// ---------------------------------------------------------------------------
// Ink-extent expansion: Tesseract bboxes often clip descenders.
// Scan the actual grayscale pixels to find true ink boundaries.
// ---------------------------------------------------------------------------

/// Expand each line's (and its words') bounding boxes vertically so they
/// encompass the actual ink extent on the page.  Tesseract frequently
/// under-reports height by clipping descenders (measured 10 px / ~3 pt on
/// 300 dpi Bodoni body text).
///
/// Only **expands** — never shrinks a bbox.  Horizontal bounds are untouched.
pub fn expand_bbox_to_ink(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8) {
    let (page_w, page_h) = gray.dimensions();
    let margin: u32 = 20; // search this many px above/below the OCR bbox
    let mut expanded_count = 0u32;

    for line in lines.iter_mut() {
        // ── expand the line bbox ────────────────────────────────────
        let lx = line.x.min(page_w.saturating_sub(1));
        let lw = line.width.min(page_w - lx);
        let search_top = line.y.saturating_sub(margin);
        let search_bot = (line.y + line.height + margin).min(page_h);

        let (ink_top, ink_bot) = ink_vertical_extent(gray, lx, lw, search_top, search_bot, ink_threshold);

        // Only expand, never shrink
        let new_y = ink_top.min(line.y);
        let new_bottom = ink_bot.max(line.y + line.height);
        let new_h = new_bottom.saturating_sub(new_y);

        if new_h > line.height {
            expanded_count += 1;
            debug!(
                "  bbox expand: '{}' y {}→{} h {}→{} (+{}px)",
                &line.text[..line.text.len().min(40)],
                line.y, new_y, line.height, new_h, new_h - line.height
            );
        }
        line.y = new_y;
        line.height = new_h;

        // ── NOTE: word bboxes are NOT expanded ─────────────────────
        // The line-level expansion above covers ascenders/descenders.
        // Expanding individual word bboxes grabs adjacent-line text
        // that corrupts character segmentation downstream.
    }

    if expanded_count > 0 {
        info!("  Expanded {expanded_count}/{} line bboxes to ink extent", lines.len());
    }
}

/// Scan a horizontal strip of the grayscale image for the topmost and
/// bottommost rows that contain ink (pixel value < `threshold`).
/// Returns (ink_top_row, ink_bottom_row) — both inclusive pixel rows.
/// If no ink is found, returns (search_top, search_top) so the caller's
/// `min()`/`max()` logic leaves the bbox unchanged.
pub fn ink_vertical_extent(
    gray: &GrayImage,
    x: u32,
    w: u32,
    search_top: u32,
    search_bot: u32,
    threshold: u8,
) -> (u32, u32) {
    let mut first_ink: Option<u32> = None;
    let mut last_ink: u32 = search_top;

    for row in search_top..search_bot {
        for col in x..x + w {
            if gray.get_pixel(col, row).0[0] < threshold {
                if first_ink.is_none() {
                    first_ink = Some(row);
                }
                last_ink = row;
                break; // found ink in this row, move to next
            }
        }
    }

    match first_ink {
        Some(top) => (top, last_ink + 1), // +1 so bottom is exclusive (height = bot - top)
        None => (search_top, search_top), // no ink found — don't change anything
    }
}

// ---------------------------------------------------------------------------

fn parse_tsv(tsv: &str, dpi: u32) -> Result<Vec<TextRegion>, ScanTextError> {
    let mut regions = Vec::new();
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 {
            continue;
        }
        let level: u32 = cols[0].parse().unwrap_or(0);
        if level != 5 {
            continue;
        }
        let text = cols[11].trim().to_string();
        if text.is_empty() {
            continue;
        }
        let block_num: u32 = cols[2].parse().unwrap_or(0);
        let par_num: u32 = cols[3].parse().unwrap_or(0);
        let line_num: u32 = cols[4].parse().unwrap_or(0);
        let word_num: u32 = cols[5].parse().unwrap_or(0);
        let x: u32 = cols[6].parse().unwrap_or(0);
        let y: u32 = cols[7].parse().unwrap_or(0);
        let width: u32 = cols[8].parse().unwrap_or(0);
        let height: u32 = cols[9].parse().unwrap_or(0);
        let confidence: f32 = cols[10].parse().unwrap_or(0.0);
        let font_size_pt = height as f32 * 72.0 / dpi as f32;

        debug!(
            "  OCR word: '{}' at ({},{} {}x{}) conf={:.0} size={:.1}pt",
            text, x, y, width, height, confidence, font_size_pt
        );

        regions.push(TextRegion {
            text,
            x,
            y,
            width,
            height,
            font_size_pt,
            confidence,
            level,
            block_num,
            par_num,
            line_num,
            word_num,
        });
    }
    Ok(regions)
}

/// Parse Tesseract HOCR output into CharBox structs.
///
/// HOCR `ocrx_cinfo` spans provide per-character bboxes with confidence,
/// structurally nested inside `ocrx_word` spans. Coordinates use top-left
/// origin (matching image coordinates — no y-flip needed unlike makebox).
///
/// Format:
/// ```html
/// <span class='ocrx_cinfo' title='x_bboxes 132 88 158 121; x_conf 99.44'>T</span>
/// ```
fn parse_hocr(hocr: &str) -> Vec<CharBox> {
    let mut boxes = Vec::new();

    for line in hocr.lines() {
        let line = line.trim();
        if !line.contains("ocrx_cinfo") {
            continue;
        }

        // Extract title attribute content: title='x_bboxes ... ; x_conf ...'
        let title_start = match line.find("title='") {
            Some(pos) => pos + 7,
            None => continue,
        };
        let title_end = match line[title_start..].find('\'') {
            Some(pos) => title_start + pos,
            None => continue,
        };
        let title = &line[title_start..title_end];

        // Parse x_bboxes: "x_bboxes x1 y1 x2 y2"
        let bbox_start = match title.find("x_bboxes ") {
            Some(pos) => pos + 9,
            None => continue,
        };
        let bbox_end = title[bbox_start..].find(';')
            .map(|p| bbox_start + p)
            .unwrap_or(title.len());
        let bbox_str = title[bbox_start..bbox_end].trim();
        let coords: Vec<u32> = bbox_str.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if coords.len() < 4 {
            continue;
        }
        let (x1, y1, x2, y2) = (coords[0], coords[1], coords[2], coords[3]);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        // Parse x_conf: "x_conf 99.44"
        let conf = if let Some(conf_start) = title.find("x_conf ") {
            let val_start = conf_start + 7;
            let val_str = title[val_start..].trim();
            val_str.parse::<f32>().unwrap_or(0.0)
        } else {
            0.0
        };

        // Extract character text content: >T</span>
        let ch = {
            let content_start = match line.rfind("'>") {
                Some(pos) => pos + 2,
                None => continue,
            };
            let content_end = match line[content_start..].find("</span>") {
                Some(pos) => content_start + pos,
                None => continue,
            };
            let text = &line[content_start..content_end];
            // Handle HTML entities
            let text = text.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&#39;", "'");
            match text.chars().next() {
                Some(c) => c,
                None => continue,
            }
        };

        boxes.push(CharBox {
            ch,
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
            confidence: conf,
        });
    }

    // Resolve horizontal overlaps within each word.
    // HOCR chars are nested in words — we track word boundaries by detecting
    // ocrx_word lines, then resolve overlaps for each word's chars.
    resolve_hocr_overlaps(&mut boxes, hocr);

    boxes
}

/// Resolve horizontal overlaps between adjacent HOCR charboxes within each word.
/// When two adjacent chars overlap, split at the midpoint of the overlap region.
fn resolve_hocr_overlaps(boxes: &mut [CharBox], hocr: &str) {
    // Count chars per word by scanning ocrx_word spans and their nested ocrx_cinfo spans.
    let mut word_char_counts: Vec<usize> = Vec::new();
    let mut in_word = false;
    let mut count = 0usize;
    for line in hocr.lines() {
        let trimmed = line.trim();
        if trimmed.contains("ocrx_word") && !trimmed.contains("ocrx_cinfo") {
            // New word starting — save previous word's count if any
            if in_word && count > 0 {
                word_char_counts.push(count);
            }
            in_word = true;
            count = 0;
        } else if trimmed.contains("ocrx_cinfo") {
            count += 1;
        }
    }
    // Don't forget the last word
    if in_word && count > 0 {
        word_char_counts.push(count);
    }

    // Now walk through boxes in word-sized groups and resolve overlaps
    let mut offset = 0usize;
    for &wc in &word_char_counts {
        if offset + wc > boxes.len() {
            break;
        }
        let word_chars = &mut boxes[offset..offset + wc];

        // Sort by x position within the word (should already be, but be safe)
        word_chars.sort_by_key(|c| c.x);

        // Resolve overlaps between adjacent chars.
        // If one box is a superset of (or wider than) the other, the wider box
        // becomes the complement of the narrower one. Otherwise fall back to
        // trimming the wider side.
        for i in 0..word_chars.len().saturating_sub(1) {
            let a_left = word_chars[i].x;
            let a_right = word_chars[i].x + word_chars[i].width;
            let b_left = word_chars[i + 1].x;
            let b_right = word_chars[i + 1].x + word_chars[i + 1].width;

            if a_right <= b_left {
                continue; // no overlap
            }

            let a_width = word_chars[i].width;
            let b_width = word_chars[i + 1].width;

            if a_width >= b_width {
                // A is wider (or equal) — shrink A to the complement: A becomes [a_left, b_left)
                word_chars[i].width = b_left.saturating_sub(a_left).max(1);
            } else {
                // B is wider — shrink B to the complement: B becomes [a_right, b_right)
                let old_right = b_right;
                word_chars[i + 1].x = a_right;
                word_chars[i + 1].width = old_right.saturating_sub(a_right).max(1);
            }
        }

        offset += wc;
    }
}
