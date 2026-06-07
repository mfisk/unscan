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
/// Tesseract handles its own binarization and preprocessing internally
/// via Leptonica — we just convert to grayscale and pass it through.
pub fn extract_text_regions(
    page_img: &DynamicImage,
    dpi: u32,
) -> Result<(Vec<TextRegion>, Vec<CharBox>), ScanTextError> {
    let tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .map_err(ScanTextError::Io)?;

    let gray = page_img.to_luma8();

    DynamicImage::ImageLuma8(gray)
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

/// Merge lines that Tesseract split from a single physical line.
///
/// Tesseract's layout analysis sometimes assigns blobs from one physical line
/// to two different `line_num` values (e.g. when a nearby image blob shifts
/// the row geometry, or when italic serifs extend outside the row band).
/// This produces two `TextLine`s with heavily overlapping y-ranges and
/// overlapping x-ranges.  We detect these and merge them into one line,
/// then merge overlapping or abutting word boxes within the result.
///
/// Must run immediately after `assemble_lines()`, before any other
/// post-processing.
pub fn merge_overlapping_lines(lines: &mut Vec<TextLine>) {
    if lines.len() < 2 {
        return;
    }

    let mut merged_count = 0u32;
    let mut i = 0;
    while i < lines.len() {
        let mut j = i + 1;
        while j < lines.len() {
            if lines_overlap(&lines[i], &lines[j]) {
                // Merge line j into line i
                debug!(
                    "  merge overlapping lines: '{}' + '{}'",
                    &lines[i].text.chars().take(40).collect::<String>(),
                    &lines[j].text.chars().take(40).collect::<String>(),
                );
                let donor = lines.remove(j);
                merge_line_into(&mut lines[i], donor);
                merged_count += 1;
                // Don't increment j — the next candidate shifted into position
            } else {
                j += 1;
            }
        }
        i += 1;
    }

    if merged_count > 0 {
        info!("  Merged {} overlapping line pair(s)", merged_count);
    }
}

/// Two lines "overlap" if their vertical ranges overlap by ≥ 50% of the
/// shorter line's height AND their horizontal ranges overlap at all.
fn lines_overlap(a: &TextLine, b: &TextLine) -> bool {
    // Vertical overlap
    let a_top = a.y;
    let a_bot = a.y + a.height;
    let b_top = b.y;
    let b_bot = b.y + b.height;
    let v_overlap_start = a_top.max(b_top);
    let v_overlap_end = a_bot.min(b_bot);
    if v_overlap_start >= v_overlap_end {
        return false; // no vertical overlap
    }
    let v_overlap = v_overlap_end - v_overlap_start;
    let min_height = a.height.min(b.height);
    if min_height == 0 || (v_overlap as f64) < min_height as f64 * 0.5 {
        return false; // less than 50% vertical overlap
    }

    // Horizontal overlap — any overlap at all means same physical line
    let a_left = a.x;
    let a_right = a.x + a.width;
    let b_left = b.x;
    let b_right = b.x + b.width;
    a_left < b_right && b_left < a_right
}

/// Merge donor line into target: combine word lists, merge overlapping/abutting
/// words, then recompute line-level fields.
fn merge_line_into(target: &mut TextLine, donor: TextLine) {
    target.words.extend(donor.words);
    // Sort all words by x position
    target.words.sort_by_key(|w| w.x);
    // Merge overlapping or abutting words
    merge_overlapping_words(&mut target.words);
    // Recompute line fields
    target.text = target.words.iter()
        .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    target.x = target.words.iter().map(|w| w.x).min().unwrap_or(0);
    target.y = target.words.iter().map(|w| w.y).min().unwrap_or(0);
    let x_max = target.words.iter().map(|w| w.x + w.width).max().unwrap_or(0);
    let y_max = target.words.iter().map(|w| w.y + w.height).max().unwrap_or(0);
    target.width = x_max.saturating_sub(target.x);
    target.height = y_max.saturating_sub(target.y);
    target.confidence = target.words.iter().map(|w| w.confidence).sum::<f32>()
        / target.words.len() as f32;
    target.font_size_pt = target.words.first()
        .map(|w| w.font_size_pt).unwrap_or(12.0);
}

/// Merge word boxes that overlap or abut horizontally within a sorted word list.
///
/// Two words are candidates for merging when the gap between them is ≤ 2px
/// (abutting) or they overlap. When merged, keep the word with higher confidence
/// if their texts cover the same span, otherwise concatenate.
fn merge_overlapping_words(words: &mut Vec<TextRegion>) {
    if words.len() < 2 {
        return;
    }
    let mut i = 0;
    while i + 1 < words.len() {
        let a_right = words[i].x + words[i].width;
        let b_left = words[i + 1].x;

        // Gap ≤ 2px or overlapping → merge candidate
        if a_right + 2 >= b_left {
            let b = words.remove(i + 1);
            let a = &mut words[i];

            // Union the bounding boxes
            let new_x = a.x.min(b.x);
            let new_y = a.y.min(b.y);
            let new_right = (a.x + a.width).max(b.x + b.width);
            let new_bot = (a.y + a.height).max(b.y + b.height);

            // If the words overlap significantly, pick the longer/higher-confidence one;
            // otherwise concatenate texts
            let overlap_amount = a_right.saturating_sub(b_left);
            if overlap_amount > a.width / 2 || overlap_amount > b.width / 2 {
                // Substantial overlap — pick the better word
                if b.confidence > a.confidence && b.text.len() >= a.text.len() {
                    a.text = b.text;
                }
                a.confidence = a.confidence.max(b.confidence);
            } else {
                // Abutting or minor overlap — concatenate
                a.text = format!("{}{}", a.text, b.text);
                a.confidence = (a.confidence + b.confidence) / 2.0;
            }

            a.x = new_x;
            a.y = new_y;
            a.width = new_right - new_x;
            a.height = new_bot - new_y;
            // Don't increment — check if the merged word now overlaps the next one
        } else {
            i += 1;
        }
    }
}

/// Clip overlapping word bboxes within each line.
///
/// Tesseract occasionally returns word bboxes that extend into the next word's
/// space, especially after contrast enhancement / sharpening.  When word A's
/// right edge crosses word B's left edge, clip A's width so it stops at B's
/// left edge (with a 1px gap).  Words must already be sorted by x within each
/// line (assemble_lines does this).
pub fn clip_word_overlaps(lines: &mut [TextLine]) {
    let mut clipped = 0u32;
    for line in lines.iter_mut() {
        let n = line.words.len();
        if n < 2 {
            continue;
        }
        // Words are already sorted by x from assemble_lines.
        for i in 0..n - 1 {
            let a_right = line.words[i].x + line.words[i].width;
            let b_left = line.words[i + 1].x;
            if a_right > b_left {
                let new_width = b_left.saturating_sub(line.words[i].x);
                if new_width > 0 {
                    debug!(
                        "  word overlap clip: '{}' width {}→{} (was overlapping '{}')",
                        line.words[i].text, line.words[i].width, new_width,
                        line.words[i + 1].text,
                    );
                    line.words[i].width = new_width;
                    clipped += 1;
                }
            }
        }
    }
    if clipped > 0 {
        info!("  Clipped {} overlapping word bbox(es)", clipped);
    }
}

/// Remove outlier words that are likely OCR artifacts from images.
///
/// When a line contains words with very different heights and some have low OCR
/// confidence, the tall low-confidence words are probably image artifacts (e.g.
/// a logo that Tesseract tried to read as text).  Dropping them prevents the
/// line bbox from ballooning and contaminating font matching / ground-truth
/// spatial lookups.
///
/// Heuristic: if a word's height is ≥ 1.8× the median word height in the line
/// AND its confidence is below 70, drop it.  After dropping, recompute the
/// line's text and bbox from the surviving words.
pub fn drop_outlier_words(lines: &mut Vec<TextLine>) {
    let mut dropped_total = 0u32;
    lines.retain_mut(|line| {
        if line.words.len() < 2 {
            return true;
        }

        // Use the median word height as baseline — min_h is too fragile
        // (a 1px em-dash makes everything look like an outlier).
        let mut heights: Vec<u32> = line.words.iter().map(|w| w.height).collect();
        heights.sort_unstable();
        let median_h = heights[heights.len() / 2];

        if median_h == 0 {
            return true;
        }

        let before = line.words.len();
        line.words.retain(|w| {
            let height_outlier = w.height as f64 >= median_h as f64 * 1.8;
            let low_conf = w.confidence < 70.0;
            if height_outlier && low_conf {
                debug!(
                    "  drop outlier word '{}' (h={}, median_h={}, conf={:.1})",
                    w.text, w.height, median_h, w.confidence
                );
                false
            } else {
                true
            }
        });

        let dropped = before - line.words.len();
        if dropped > 0 {
            dropped_total += dropped as u32;
            if line.words.is_empty() {
                return false; // entire line was artifacts
            }
            // Recompute line text and bbox from surviving words.
            line.text = line.words.iter()
                .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
            line.x = line.words.iter().map(|w| w.x).min().unwrap_or(0);
            line.y = line.words.iter().map(|w| w.y).min().unwrap_or(0);
            let x_max = line.words.iter().map(|w| w.x + w.width).max().unwrap_or(0);
            let y_max = line.words.iter().map(|w| w.y + w.height).max().unwrap_or(0);
            line.width = x_max.saturating_sub(line.x);
            line.height = y_max.saturating_sub(line.y);
            line.confidence = line.words.iter().map(|w| w.confidence).sum::<f32>()
                / line.words.len() as f32;
        }
        true
    });
    if dropped_total > 0 {
        info!("  Dropped {} outlier word(s) (image artifacts)", dropped_total);
    }
}

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
    let margin: u32 = 20; // search this many px beyond the OCR bbox
    let mut expanded_count = 0u32;

    for line in lines.iter_mut() {
        // ── expand the line bbox vertically ─────────────────────────
        let lx = line.x.min(page_w.saturating_sub(1));
        let lw = line.width.min(page_w - lx);
        let search_top = line.y.saturating_sub(margin);
        let search_bot = (line.y + line.height + margin).min(page_h);

        let (ink_top, ink_bot) = ink_vertical_extent(gray, lx, lw, search_top, search_bot, ink_threshold);

        // Only expand, never shrink
        let new_y = ink_top.min(line.y);
        let new_bottom = ink_bot.max(line.y + line.height);
        let new_h = new_bottom.saturating_sub(new_y);

        let vert_grew = new_h > line.height;
        line.y = new_y;
        line.height = new_h;

        // ── expand the line bbox horizontally ───────────────────────
        // Check leftward: walk columns left from line.x within the
        // vertical extent, looking for ink.
        let y_top = line.y;
        let y_bot = (line.y + line.height).min(page_h);
        let search_left = line.x.saturating_sub(margin);
        let mut new_x = line.x;
        for col in (search_left..line.x).rev() {
            let col_has_ink = (y_top..y_bot)
                .any(|row| gray.get_pixel(col, row).0[0] < ink_threshold);
            if col_has_ink {
                new_x = col;
            } else {
                break;
            }
        }

        // Check rightward
        let old_right = line.x + line.width;
        let search_right = (old_right + margin).min(page_w);
        let mut new_right = old_right;
        for col in old_right..search_right {
            let col_has_ink = (y_top..y_bot)
                .any(|row| gray.get_pixel(col, row).0[0] < ink_threshold);
            if col_has_ink {
                new_right = col + 1;
            } else {
                break;
            }
        }

        let horiz_grew = new_x < line.x || new_right > old_right;
        line.x = new_x;
        line.width = new_right.saturating_sub(new_x);

        if vert_grew || horiz_grew {
            expanded_count += 1;
            debug!(
                "  bbox expand: '{}' x {}→{} y {}→{} w {}→{} h {}→{}",
                &line.text.chars().take(40).collect::<String>(),
                lx, line.x, search_top.max(ink_top), line.y,
                lw, line.width, new_h.saturating_sub(new_h - line.height), line.height
            );
        }

        // ── NOTE: word bboxes are NOT expanded here ────────────────
        // expand_words_to_ink handles word-level expansion separately.
    }

    if expanded_count > 0 {
        info!("  Expanded {expanded_count}/{} line bboxes to ink extent", lines.len());
    }
}

/// Expand word bboxes horizontally when ink is present at the edge and there
/// is free space before the next word.  Italic glyphs frequently overshoot
/// the OCR bbox to the right.  For each word, if the rightmost column of its
/// crop contains ink, extend rightward column-by-column as long as (a) we
/// find ink and (b) we haven't reached the next word's left edge.
///
/// Only **expands** — never shrinks.  Must run AFTER `clip_word_overlaps` so
/// the word list is already gap-safe.
pub fn expand_words_to_ink(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8) {
    let (page_w, page_h) = gray.dimensions();
    let mut expanded = 0u32;

    for line in lines.iter_mut() {
        let n = line.words.len();
        let line_top = line.y;
        let line_bot = (line.y + line.height).min(page_h);

        for i in 0..n {
            let mut changed = false;

            // ── Rightward expansion ─────────────────────────────────
            {
                let w = &line.words[i];
                let right_edge = w.x + w.width;
                let limit = if i + 1 < n {
                    line.words[i + 1].x
                } else {
                    (line.x + line.width).min(page_w)
                };

                if right_edge < limit && right_edge < page_w {
                    let check_col = right_edge.saturating_sub(1);
                    let y_top = w.y;
                    let y_bot = w.y + w.height;
                    let has_edge_ink = (y_top..y_bot)
                        .any(|row| gray.get_pixel(check_col.min(page_w - 1), row).0[0] < ink_threshold);

                    if has_edge_ink {
                        let mut new_right = right_edge;
                        for col in right_edge..limit.min(page_w) {
                            let col_has_ink = (y_top..y_bot)
                                .any(|row| gray.get_pixel(col, row).0[0] < ink_threshold);
                            if col_has_ink {
                                new_right = col + 1;
                            } else {
                                break;
                            }
                        }
                        if new_right > right_edge {
                            let growth = new_right - right_edge;
                            debug!(
                                "  word ink-expand right: '{}' width {}→{} (+{}px)",
                                line.words[i].text, line.words[i].width, line.words[i].width + growth, growth
                            );
                            line.words[i].width += growth;
                            changed = true;
                        }
                    }
                }
            }

            // ── Leftward expansion ──────────────────────────────────
            {
                let left_edge = line.words[i].x;
                let word_y = line.words[i].y;
                let word_h = line.words[i].height;
                let mut limit = if i > 0 {
                    line.words[i - 1].x + line.words[i - 1].width
                } else {
                    line.x
                };

                // Trim the previous word's trailing empty columns so
                // the boundary between adjacent words sits in actual
                // whitespace, not inside dead space that Tesseract
                // assigned to the wrong word.
                //
                // Two-phase approach:
                // Phase 1 (legacy): walk backward from prev_right looking
                //   for fully-empty columns.  Handles the common case.
                // Phase 2 (rebalance): if phase 1 found nothing to trim
                //   AND the words are flush, scan the boundary region for
                //   a zero-ink column — the actual inter-word gap that
                //   the previous word's rightward expansion overshot
                //   (common with italic text where glyphs lean into the
                //   next word's space).
                if i > 0 {
                    let prev_x = line.words[i - 1].x;
                    let prev_right = prev_x + line.words[i - 1].width;
                    let prev_y_top = line.words[i - 1].y;
                    let prev_y_bot = prev_y_top + line.words[i - 1].height;

                    // Phase 1: trim trailing empty columns.
                    let mut shrink_to = prev_right;
                    for col in (prev_x..prev_right).rev() {
                        let col_has_ink = (prev_y_top..prev_y_bot)
                            .any(|row| gray.get_pixel(col.min(page_w - 1), row).0[0] < ink_threshold);
                        if col_has_ink {
                            shrink_to = col + 1;
                            break;
                        }
                        shrink_to = col;
                    }
                    if shrink_to < prev_right {
                        let old_w = line.words[i - 1].width;
                        line.words[i - 1].width = shrink_to.saturating_sub(prev_x);
                        debug!(
                            "  word ink-shrink trailing: '{}' width {}→{} (freed {}px for '{}')",
                            line.words[i - 1].text, old_w, line.words[i - 1].width,
                            prev_right - shrink_to, line.words[i].text
                        );
                        limit = shrink_to;
                    } else if left_edge <= prev_right {
                        // Phase 2: phase 1 found nothing (rightmost column
                        // has ink) and the words are flush or overlapping.
                        // The prev word's expansion likely overshot through
                        // a narrow gap into the current word's glyph ink.
                        // Scan backward from the boundary for a zero-ink
                        // column — the actual inter-word whitespace gap.
                        let prev_h = line.words[i - 1].height;
                        let scan_y_top = prev_y_top.min(word_y);
                        let scan_y_bot = prev_y_bot.max(word_y + word_h);
                        // Only scan back at most one word-height (roughly
                        // one character width) from the boundary.
                        let scan_left = prev_right.saturating_sub(prev_h).max(prev_x);
                        let mut gap_col: Option<u32> = None;
                        for col in (scan_left..prev_right).rev() {
                            let ink: u32 = (scan_y_top..scan_y_bot)
                                .map(|row| {
                                    let px = gray.get_pixel(col.min(page_w - 1), row).0[0];
                                    if px < ink_threshold { 1 } else { 0 }
                                })
                                .sum();
                            if ink == 0 {
                                gap_col = Some(col);
                                break;
                            }
                        }
                        if let Some(gc) = gap_col {
                            let old_w = line.words[i - 1].width;
                            line.words[i - 1].width = gc.saturating_sub(prev_x);
                            debug!(
                                "  word boundary rebalance: '{}' width {}→{} (freed {}px for '{}')",
                                line.words[i - 1].text, old_w, line.words[i - 1].width,
                                prev_right - gc, line.words[i].text
                            );
                            limit = gc;
                        }
                    }
                }

                if left_edge > limit {
                    let y_top = word_y;
                    let y_bot = word_y + word_h;

                    // Check for ink anywhere between limit and left_edge,
                    // not just at the current edge — Tesseract may have
                    // placed the box several pixels right of the glyph.
                    let gap_has_ink = (limit..left_edge)
                        .any(|col| (y_top..y_bot)
                            .any(|row| gray.get_pixel(col.min(page_w - 1), row).0[0] < ink_threshold));

                    let has_edge_ink = (y_top..y_bot)
                        .any(|row| gray.get_pixel(left_edge.min(page_w - 1), row).0[0] < ink_threshold);

                    if has_edge_ink || gap_has_ink {
                        let mut new_left = left_edge;
                        for col in (limit..left_edge).rev() {
                            let col_has_ink = (y_top..y_bot)
                                .any(|row| gray.get_pixel(col, row).0[0] < ink_threshold);
                            if col_has_ink {
                                new_left = col;
                            } else {
                                break;
                            }
                        }
                        if new_left < left_edge {
                            let growth = left_edge - new_left;
                            debug!(
                                "  word ink-expand left: '{}' x {}→{} (+{}px)",
                                line.words[i].text, line.words[i].x, new_left, growth
                            );
                            line.words[i].x = new_left;
                            line.words[i].width += growth;
                            changed = true;
                        }
                    }
                }
            }

            // ── Vertical expansion (bounded by line bbox) ───────────
            {
                let w = &line.words[i];
                let wx = w.x;
                let wr = (w.x + w.width).min(page_w);
                let word_top = w.y;
                let word_bot = w.y + w.height;

                // Expand upward
                let mut new_top = word_top;
                for row in (line_top..word_top).rev() {
                    let row_has_ink = (wx..wr)
                        .any(|col| gray.get_pixel(col, row).0[0] < ink_threshold);
                    if row_has_ink {
                        new_top = row;
                    } else {
                        break;
                    }
                }

                // Expand downward
                let mut new_bot = word_bot;
                for row in word_bot..line_bot {
                    let row_has_ink = (wx..wr)
                        .any(|col| gray.get_pixel(col, row).0[0] < ink_threshold);
                    if row_has_ink {
                        new_bot = row + 1;
                    } else {
                        break;
                    }
                }

                // Add 1px anti-alias padding (bounded by line bbox).
                // Rasterized text has sub-threshold anti-aliased edges
                // that the ink walk misses; the padding captures them.
                if new_top > line_top { new_top -= 1; }
                if new_bot < line_bot { new_bot += 1; }

                if new_top < word_top || new_bot > word_bot {
                    let old_h = line.words[i].height;
                    line.words[i].y = new_top;
                    line.words[i].height = new_bot - new_top;
                    debug!(
                        "  word ink-expand vert: '{}' y {}→{} h {}→{}",
                        line.words[i].text, word_top, new_top, old_h, line.words[i].height
                    );
                    changed = true;
                }
            }

            if changed {
                expanded += 1;
            }
        }
    }

    if expanded > 0 {
        info!("  Expanded {} word bbox(es) to ink (horiz+vert)", expanded);
    }
}

/// Split words that contain wide internal bands of whitespace into separate
/// words.  Tesseract sometimes merges spaced-out characters (e.g. "0 1 2 3")
/// into a single word bbox with text "0123456789".  This function scans each
/// word's image for zero-ink column runs wider than a threshold and splits
/// the word at each gap.
///
/// When `char_index` is provided, we do a quick CI font lookup on each word
/// and use the matched font's advance widths to determine the expected
/// inter-character gap.  We only split where the observed ink-to-ink gap
/// exceeds the font's natural spacing.  When no char_index is available,
/// falls back to a fixed threshold of 18% of line height.
///
/// Must run AFTER `expand_words_to_ink` so bboxes are ink-tight.
pub fn split_wide_whitespace_words(
    lines: &mut [TextLine],
    gray: &GrayImage,
    ink_threshold: u8,
    char_index: Option<&crate::char_index::CharIndex>,
    font_cache: Option<&crate::font_cache::FontCache>,
) {
    let (page_w, page_h) = gray.dimensions();
    let mut total_splits = 0u32;
    let debug_gaps = std::env::var("UNSCAN_DEBUG_GAPS").is_ok();

    for line in lines.iter_mut() {
        let mut new_words: Vec<TextRegion> = Vec::new();
        let line_h = line.height;
        // Fallback: split at gaps ≥ 18% of the line height.
        let fallback_min_gap = (line_h * 18 / 100).max(4);

        for word in line.words.drain(..) {
            let wx = word.x.min(page_w.saturating_sub(1));
            let wy = word.y.min(page_h.saturating_sub(1));
            let ww = word.width.min(page_w - wx);
            let wh = word.height.min(page_h - wy);

            let chars: Vec<char> = word.text.chars().collect();
            let n_chars = chars.len();

            if ww < 4 || wh < 2 || n_chars < 2 {
                new_words.push(word);
                continue;
            }

            // Crop the word image and run CI segmentation to get per-char boundaries
            let word_img = image::imageops::crop_imm(gray, wx, wy, ww, wh).to_image();
            let (boundaries, _seams) = crate::segment::segment_characters(&word_img, n_chars);

            // boundaries: [0, b1, b2, ..., ww] — n_chars+1 entries
            // Character i occupies columns [boundaries[i] .. boundaries[i+1])
            if boundaries.len() != n_chars + 1 {
                new_words.push(word);
                continue;
            }

            // --- Font identification for metric-based splitting ---
            // If we have a char index, do a quick CI match on this word's char
            // crops to identify the font.  We'll use the font's glyph outlines
            // to compute per-pair expected gaps inline below.
            let matched_font: Option<(std::sync::Arc<Vec<u8>>, String)> = char_index.and_then(|ci| {
                // Build char crops from segmentation
                let mut crops: Vec<(char, GrayImage)> = Vec::new();
                for (ci_idx, &ch) in chars.iter().enumerate() {
                    if !crate::char_index::is_indexed(ch) {
                        continue;
                    }
                    let left = boundaries[ci_idx] as u32;
                    let right = boundaries[ci_idx + 1] as u32;
                    if right <= left { continue; }
                    let crop = image::imageops::crop_imm(&word_img, left, 0, right - left, wh).to_image();
                    if let Some(norm) = crate::char_index::normalize_to_ink_bounds(&crop) {
                        crops.push((ch, norm));
                    }
                }
                if crops.is_empty() { return None; }

                // Quick CI search — just need the top font name
                let result = crate::char_index::search_candidates(ci, &crops, 1.0, false);
                let font_key = result.scores.first().map(|(name, _)| name.clone())?;

                let font_path = if font_key.contains('|') {
                    font_key.split('|').next().unwrap()
                } else {
                    &font_key
                };
                let font_data: std::sync::Arc<Vec<u8>> = if let Some(fc) = font_cache {
                    fc.load(std::path::Path::new(font_path)).ok()?
                } else {
                    std::sync::Arc::new(std::fs::read(font_path).ok()?)
                };
                Some((font_data, font_key))
            });
            let matched_font_ref = matched_font.as_ref().and_then(|(data, _key)| {
                ab_glyph::FontRef::try_from_slice(data).ok()
            });

            // For each adjacent pair of characters, measure the zero-ink gap
            // between the rightmost ink of char i and the leftmost ink of char i+1.
            let mut wide_gap_indices: Vec<usize> = Vec::new(); // indices into chars where a word break occurs AFTER char[i]

            for i in 0..n_chars - 1 {

                // Find leftmost and rightmost ink columns in char i's segment
                let seg_i_left = boundaries[i] as usize;
                let seg_i_right = boundaries[i + 1] as usize;
                let mut left_ink_i = seg_i_right; // fallback: no ink
                for col in seg_i_left..seg_i_right {
                    if col < ww as usize {
                        let has_ink = (0..wh).any(|row| {
                            word_img.get_pixel(col as u32, row).0[0] < ink_threshold
                        });
                        if has_ink {
                            left_ink_i = col;
                            break;
                        }
                    }
                }
                let mut right_ink = seg_i_left; // fallback
                for col in (seg_i_left..seg_i_right).rev() {
                    if col < ww as usize {
                        let has_ink = (0..wh).any(|row| {
                            word_img.get_pixel(col as u32, row).0[0] < ink_threshold
                        });
                        if has_ink {
                            right_ink = col;
                            break;
                        }
                    }
                }

                // Find leftmost ink column in char i+1's segment
                let seg_j_left = boundaries[i + 1] as usize;
                let seg_j_right = boundaries[i + 2] as usize;
                let mut left_ink = seg_j_right; // fallback
                for col in seg_j_left..seg_j_right {
                    if col < ww as usize {
                        let has_ink = (0..wh).any(|row| {
                            word_img.get_pixel(col as u32, row).0[0] < ink_threshold
                        });
                        if has_ink {
                            left_ink = col;
                            break;
                        }
                    }
                }

                let gap = if left_ink > right_ink {
                    (left_ink - right_ink) as u32
                } else {
                    0
                };

                // Font-metric expected gap: compute the expected inter-ink
                // gap for this character pair at the derived scale, round it,
                // and add 5px margin for AA/scan error.
                let min_gap_for_pair = matched_font_ref.as_ref()
                    .and_then(|font| {
                        let observed_ink_w = if right_ink > left_ink_i {
                            (right_ink - left_ink_i + 1) as f32
                        } else {
                            return None;
                        };
                        let ref_scale = 100.0_f32;
                        let ref_px = ab_glyph::PxScale { x: ref_scale, y: ref_scale };
                        let font_ink_w = crate::char_index::font_ink_width(font, ref_px, chars[i])?;
                        if font_ink_w <= 0.0 { return None; }
                        let s = observed_ink_w / font_ink_w * ref_scale;
                        let scale = ab_glyph::PxScale { x: s, y: s };
                        let expected = crate::char_index::font_pair_ink_gap(font, scale, chars[i], chars[i + 1]);
                        let threshold = expected.round() as u32 + 5;
                        if debug_gaps {
                            let word_text: String = chars.iter().collect();
                            eprintln!("[gap-debug] '{}'→'{}' word=\"{}\" obs_ink={:.1} font_ink={:.1} scale={:.1} expected_gap={:.1} threshold={} scan_gap={} fallback={}",
                                chars[i], chars[i+1], word_text, observed_ink_w, font_ink_w, s, expected, threshold, gap, fallback_min_gap);
                        }
                        Some(threshold)
                    })
                    .unwrap_or_else(|| {
                        if debug_gaps {
                            let word_text: String = chars.iter().collect();
                            eprintln!("[gap-debug] '{}'→'{}' word=\"{}\" FALLBACK (no font metric) scan_gap={} fallback={}",
                                chars[i], chars[i+1], word_text, gap, fallback_min_gap);
                        }
                        fallback_min_gap
                    });

                if gap >= min_gap_for_pair {
                    wide_gap_indices.push(i);
                }
            }

            if wide_gap_indices.is_empty() {
                new_words.push(word);
                continue;
            }

            // Group characters into new words, splitting at wide gaps.
            // Word breaks occur AFTER chars at wide_gap_indices.
            let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
            let mut start = 0usize;
            for &gap_after in &wide_gap_indices {
                groups.push(start..gap_after + 1);
                start = gap_after + 1;
            }
            groups.push(start..n_chars);

            for group in &groups {
                let seg_text: String = chars[group.clone()].iter().collect();

                // Bbox spans from the left edge of the first char's segment
                // to the right edge of the last char's segment.
                let seg_x_start = boundaries[group.start];
                let seg_x_end = boundaries[group.end];
                let seg_w = seg_x_end - seg_x_start;

                // Trim horizontally to ink within this segment
                let abs_x = wx + seg_x_start;
                let mut ink_left = seg_w;
                let mut ink_right = 0u32;
                for col_off in 0..seg_w {
                    let col = abs_x + col_off;
                    if col < page_w {
                        let has_ink = (wy..wy + wh).any(|row| {
                            gray.get_pixel(col, row).0[0] < ink_threshold
                        });
                        if has_ink {
                            ink_left = ink_left.min(col_off);
                            ink_right = col_off;
                        }
                    }
                }

                let (trimmed_x, trimmed_w) = if ink_right >= ink_left {
                    (abs_x + ink_left, ink_right - ink_left + 1)
                } else {
                    (abs_x, seg_w) // no ink found, keep original
                };

                new_words.push(TextRegion {
                    text: seg_text,
                    x: trimmed_x,
                    y: wy,
                    width: trimmed_w,
                    height: wh,
                    font_size_pt: word.font_size_pt,
                    confidence: word.confidence,
                    level: word.level,
                    block_num: word.block_num,
                    par_num: word.par_num,
                    line_num: word.line_num,
                    word_num: word.word_num,
                });
            }

            total_splits += wide_gap_indices.len() as u32;
            debug!(
                "  split word '{}' into {} pieces at {} gap(s) (min_gap={}px, line_h={}px)",
                word.text, groups.len(), wide_gap_indices.len(), fallback_min_gap, line_h
            );
        }

        // Replace words and rebuild line text
        line.words = new_words;
        line.text = line.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    }

    if total_splits > 0 {
        info!("split_wide_whitespace_words: {} splits across all lines", total_splits);
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
