//! OCR module — runs Tesseract via CLI to extract text with bounding boxes.
//!
//! # Word box pipeline
//!
//! Tesseract returns word boxes that often include trailing or leading
//! whitespace and that occasionally overlap the next word. Unprint cleans them
//! in stages so downstream font matching and segmentation see tight,
//! ink-faithful boxes.
//!
//! 1. **Geometry-only assembly** in `postprocess_words` (no image needed):
//!    - `assemble_lines` groups raw `TextRegion`s into `TextLine`s using
//!      Tesseract's block/par/line numbers.
//!    - `snapshot_raw_bboxes` saves original Tesseract boxes for debugging.
//!    - `merge_overlapping_lines` is currently disabled; `split_merged_lines`
//!      handles multi-line merges via vertical projection.
//!    - `drop_outlier_words` removes tall, low-confidence image artifacts.
//!    Pipeline: `assemble → snapshot → merge-disabled → split → drop`.
//!
//! 2. **Ink-aware refinement** in `page_cache.rs` after background detection:
//!    - `expand_words_to_ink` grows each word to its true ink using a strict
//!      ink threshold as a gate and a softer blur walk. It only expands when
//!      the current box edge already sits on ink or the gap contains ink.
//!      Vertical expansion is bounded by margin.
//!    - `fix_overlapping_words_by_ink` reflows overlapping words to the natural
//!      whitespace gap: scans between word centers for zero-ink runs, picks the
//!      run closest to `(a_right+b_x)/2`, splits at `run_start` / `run_end+1`.
//!      Single-word lines (`n<2`) are unchanged.
//!    - `trim_words_to_ink` shrinks each word to its ink bounds using middle
//!      80% of height to avoid adjacent-line contamination, removing trailing/
//!      leading whitespace (e.g. Georgia final 'a' 10px ws). This replaces the
//!      prior 90% shrink which regressed `abcdefghijklmnopqrstuvwxyz.` 489w→440w,
//!      and the prior 60% which cut off baseline '.'.

use crate::error::ScanTextError;
use unprint_fonts::ab_glyph::{Font, PxScale, ScaleFont, point};
use image::{DynamicImage, GrayImage};
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
    /// Snapshot of word bboxes as Tesseract originally reported them,
    /// before merge/clip/expand post-processing.  Populated by
    /// `snapshot_raw_bboxes()` right after `assemble_lines()`.
    pub raw_words: Vec<RawWordBBox>,
}

impl TextLine {
}

/// Lightweight snapshot of a Tesseract word bbox before post-processing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RawWordBBox {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

/// Capture a raw-bbox snapshot into each line's `raw_words` field.
/// Call once, right after `assemble_lines()` and before any merge/clip/expand.
pub fn snapshot_raw_bboxes(lines: &mut [TextLine]) {
    for line in lines.iter_mut() {
        line.raw_words = line.words.iter().map(|w| RawWordBBox {
            text: w.text.clone(),
            x: w.x,
            y: w.y,
            width: w.width,
            height: w.height,
            confidence: w.confidence,
        }).collect();
    }
}

/// Run Tesseract on a page image and return word-level regions plus
/// character-level bounding boxes (from makebox output).
/// Tesseract handles its own binarization and preprocessing internally
/// via Leptonica — we just convert to grayscale and pass it through.
pub fn extract_text_regions_from_gray(
    gray: &image::GrayImage,
    dpi: u32,
) -> Result<(Vec<TextRegion>, Vec<CharBox>), ScanTextError> {
    let tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .map_err(ScanTextError::Io)?;

    gray.save(tmp.path())
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
        Vec::new()
    };

    Ok((regions, char_boxes))
}

#[allow(dead_code)]
pub fn extract_text_regions(
    page_img: &DynamicImage,
    dpi: u32,
) -> Result<(Vec<TextRegion>, Vec<CharBox>), ScanTextError> {
    // Fast path: if already Luma8, avoid to_luma8() clone
    if let Some(gray_ref) = page_img.as_luma8() {
        return extract_text_regions_from_gray(gray_ref, dpi);
    }
    let gray = page_img.to_luma8();
    extract_text_regions_from_gray(&gray, dpi)
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
            raw_words: Vec::new(), // populated by snapshot_raw_bboxes()
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
#[allow(unreachable_code, unused_variables)]
pub fn merge_overlapping_lines(lines: &mut Vec<TextLine>) {
    return; // DISABLED — split_merged_lines handles the multi-line case
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
    target.raw_words.extend(donor.raw_words);
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

/// Reflow overlapping word bboxes to the natural whitespace gap using ink.
///
/// When two word boxes overlap, the system has assigned trailing whitespace
/// from one word to the next. The natural gap is found by looking at column
/// ink totals between the two word centers. Every stretch of pure whitespace
/// is collected, and the stretch whose middle is closest to where the boxes
/// originally overlapped is chosen. The left word ends at the start of that
/// stretch and the right word starts after it. This restores the true
/// inter-word gap and removes trailing whitespace that simple clipping would
/// leave. If no pure whitespace exists between the centers, the left word is
/// cut at the left edge of the right word as a fallback.
pub fn fix_overlapping_words_by_ink(
    lines: &mut [TextLine],
    gray: &GrayImage,
    ink_threshold: u8,
) {
    let page_w = gray.width();
    let page_h = gray.height();

    for line in lines.iter_mut() {
        let n = line.words.len();
        if n < 2 {
            continue;
        }
        for i in 0..n - 1 {
            let a_x = line.words[i].x;
            let a_right = a_x + line.words[i].width;
            let b_x = line.words[i + 1].x;
            let b_right = b_x + line.words[i + 1].width;

            if a_right <= b_x {
                continue;
            }

            let a_center = a_x + line.words[i].width / 2;
            let b_center = b_x + line.words[i + 1].width / 2;

            let search_left = a_center.min(page_w.saturating_sub(1));
            let mut search_right = b_center.min(page_w);
            if search_right <= search_left {
                let new_w = b_x.saturating_sub(a_x);
                if new_w > 0 {
                    line.words[i].width = new_w;
                }
                continue;
            }
            let union_left = a_x.min(b_x);
            let union_right = a_right.max(b_right).min(page_w);
            let search_left = search_left.max(union_left);
            search_right = search_right.min(union_right);
            if search_right <= search_left {
                let new_w = b_x.saturating_sub(a_x);
                if new_w > 0 {
                    line.words[i].width = new_w;
                }
                continue;
            }

            let y_top = line.words[i].y.min(line.words[i + 1].y);
            let y_bot = (line.words[i].y + line.words[i].height)
                .max(line.words[i + 1].y + line.words[i + 1].height)
                .min(page_h);
            if y_top >= y_bot {
                let new_w = b_x.saturating_sub(a_x);
                if new_w > 0 {
                    line.words[i].width = new_w;
                }
                continue;
            }

            let mut col_has_ink = Vec::with_capacity((search_right - search_left) as usize);
            for col in search_left..search_right {
                let mut has = false;
                for row in y_top..y_bot {
                    if gray.as_raw()[(row) as usize * gray.width() as usize + (col) as usize] < ink_threshold {
                        has = true;
                        break;
                    }
                }
                col_has_ink.push(has);
            }

            let mut runs: Vec<(u32, u32)> = Vec::new();
            let mut run_start: Option<u32> = None;
            for (idx, has) in col_has_ink.iter().enumerate() {
                let col = search_left + idx as u32;
                if !*has {
                    if run_start.is_none() {
                        run_start = Some(col);
                    }
                } else if let Some(rs) = run_start.take() {
                    runs.push((rs, col - 1));
                }
            }
            if let Some(rs) = run_start {
                runs.push((rs, search_right - 1));
            }

            if runs.is_empty() {
                let new_w = b_x.saturating_sub(a_x);
                if new_w > 0 {
                    line.words[i].width = new_w;
                }
                continue;
            }

            let overlap_center = (a_right + b_x) / 2;

            let mut best_idx = 0;
            let mut best_dist = u32::MAX;
            let mut best_width = 0;
            for (idx, (rs, re)) in runs.iter().enumerate() {
                let run_center = (rs + re) / 2;
                let dist = run_center.abs_diff(overlap_center);
                let width = re - rs;
                if dist < best_dist || (dist == best_dist && width > best_width) {
                    best_dist = dist;
                    best_width = width;
                    best_idx = idx;
                }
            }
            let (best_rs, best_re) = runs[best_idx];

            let new_a_width = best_rs.saturating_sub(a_x);
            let new_b_x = best_re + 1;
            if new_a_width == 0 || new_b_x >= b_right || new_b_x <= a_x {
                let new_w = b_x.saturating_sub(a_x);
                if new_w > 0 {
                    line.words[i].width = new_w;
                }
                continue;
            }
            let new_b_width = b_right.saturating_sub(new_b_x);
            if new_b_width == 0 {
                continue;
            }

            line.words[i].width = new_a_width;
            line.words[i + 1].x = new_b_x;
            line.words[i + 1].width = new_b_width;
        }
    }
}

/// Trim each word bbox to its actual ink bounds.
///
/// Removes trailing/leading whitespace that Tesseract included in the word
/// box (e.g. Georgia final 'a' with 10px trailing ws: image_w 153, ink 123-142).
/// Must run after `expand_words_to_ink` and `fix_overlapping_words_by_ink`
/// so boxes are already roughly correct.
///
/// To avoid false trimming from vertical overlap with other lines, we scan
/// only the middle 80% of the word height (or full height if very short),
/// and require at least 1 ink pixel per column.
///
/// This replaces the prior 90% shrink which could place edges inside
/// inter-glyph whitespace and prevent later expansion from recovering.
pub fn trim_words_to_ink(
    lines: &mut [TextLine],
    gray: &GrayImage,
    ink_threshold: u8,
) {
    let page_w = gray.width();
    let page_h = gray.height();

    for line in lines.iter_mut() {
        for word in line.words.iter_mut() {
            if word.width <= 2 || word.height <= 2 {
                continue;
            }
            let wx = word.x.min(page_w.saturating_sub(1));
            let wy = word.y.min(page_h.saturating_sub(1));
            let ww = word.width.min(page_w - wx);
            let wh = word.height.min(page_h - wy);
            if ww == 0 || wh == 0 {
                continue;
            }

            // Use middle 80% of height to avoid ascenders/descenders from
            // adjacent lines that vertically overlap (Georgia p3 L54) while
            // keeping baseline punctuation like '.' which sits low (y 49-53 for h=59).
            let (y_top, y_bot) = if wh >= 10 {
                let margin = wh * 10 / 100; // 10% top and bottom -> 80% middle
                (wy + margin, wy + wh - margin)
            } else {
                (wy, wy + wh)
            };

            // Find leftmost ink column
            let mut left_ink = None;
            for col in wx..wx+ww {
                let has_ink = (y_top..y_bot).any(|row| {
                    gray.as_raw()[(row.min(page_h-1)) as usize * gray.width() as usize + (col.min(page_w-1)) as usize] < ink_threshold
                });
                if has_ink {
                    left_ink = Some(col);
                    break;
                }
            }
            // Find rightmost ink column (inclusive)
            let mut right_ink = None;
            for col in (wx..wx+ww).rev() {
                let has_ink = (y_top..y_bot).any(|row| {
                    gray.as_raw()[(row.min(page_h-1)) as usize * gray.width() as usize + (col.min(page_w-1)) as usize] < ink_threshold
                });
                if has_ink {
                    right_ink = Some(col);
                    break;
                }
            }

            if let (Some(li), Some(ri)) = (left_ink, right_ink) {
                // ri is inclusive, so new width = ri - li + 1
                if ri >= li {
                    let new_x = li;
                    let new_w = ri - li + 1;
                    // Only shrink, never expand (expand already did)
                    // Allow up to 2px expansion for anti-aliasing edge?
                    // For now, shrink only if it reduces width.
                    if new_x >= wx && new_w <= ww {
                        // Ensure we don't shift too far (keep within original)
                        word.x = new_x;
                        word.width = new_w;
                    } else if new_x > wx {
                        // Left trim even if width slightly larger due to off-by-1
                        let shift = new_x - wx;
                        if shift < ww {
                            word.x = new_x;
                            word.width = ww - shift;
                            if new_w < word.width {
                                word.width = new_w;
                            }
                        }
                    } else if new_w < ww {
                        word.width = new_w;
                    }
                }
            }
        }
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

/// Split lines that contain words from multiple physical text lines.
///
/// Uses a confidence-weighted vertical projection profile: for each y-pixel
/// in the line's bounding box, accumulate `confidence / 100` for every word
/// box that covers it.  Contiguous interior runs where the profile drops
/// below 0.5 (less than one word at 50% confidence — the point where
/// Tesseract itself considers recognition a coin-flip) are valleys.
///
/// Words whose vertical midpoint falls in a valley and whose confidence is
/// below 60% are dropped (likely image artefacts or decorations that span
/// the gap between real text lines).  The remaining words are partitioned
/// into bands separated by the valleys, each band becoming its own TextLine.
pub fn split_merged_lines(lines: &mut Vec<TextLine>) {
    // ── First pass: split lines with extreme height-ratio words ──────
    // Words at drastically different scales belong to different lines
    // regardless of confidence. Pure geometric check — if words in the
    // same Tesseract line have heights differing by 2×+, they're from
    // different text regions.
    let mut i = 0;
    while i < lines.len() {
        if lines[i].words.len() < 2 {
            i += 1;
            continue;
        }

        let mut heights: Vec<(usize, u32)> = lines[i].words.iter()
            .enumerate()
            .map(|(idx, w)| (idx, w.height))
            .collect();
        heights.sort_by_key(|&(_, h)| h);

        // Find the largest relative gap between consecutive sorted heights.
        let mut best_ratio = 1.0f64;
        let mut best_split = 0usize;
        for k in 0..heights.len() - 1 {
            let lo = heights[k].1.max(1) as f64;
            let hi = heights[k + 1].1.max(1) as f64;
            let ratio = hi / lo;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_split = k;
            }
        }

        if best_ratio < 2.0 {
            i += 1;
            continue;
        }

        // Don't split out words under 10px tall — they're punctuation
        // strokes (em-dashes, dots), not text from a separate line.
        let tallest_small = heights[best_split].1;
        if tallest_small < 10 {
            i += 1;
            continue;
        }

        let small_indices: std::collections::HashSet<usize> = heights[..=best_split]
            .iter().map(|&(idx, _)| idx).collect();

        let mut small_words: Vec<TextRegion> = Vec::new();
        let mut large_words: Vec<TextRegion> = Vec::new();
        for (idx, w) in lines[i].words.iter().enumerate() {
            if small_indices.contains(&idx) {
                small_words.push(w.clone());
            } else {
                large_words.push(w.clone());
            }
        }

        if small_words.is_empty() || large_words.is_empty() {
            i += 1;
            continue;
        }

        lines.remove(i);
        for (j, group) in [small_words, large_words].into_iter().enumerate() {
            let text = group.iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let bx = group.iter().map(|w| w.x).min().unwrap_or(0);
            let by = group.iter().map(|w| w.y).min().unwrap_or(0);
            let bx_max = group.iter().map(|w| w.x + w.width).max().unwrap_or(0);
            let by_max = group.iter().map(|w| w.y + w.height).max().unwrap_or(0);
            let avg_conf = group.iter().map(|w| w.confidence).sum::<f32>()
                / group.len() as f32;
            let fsize = group.first()
                .map(|w| w.font_size_pt).unwrap_or(12.0);

            lines.insert(i + j, TextLine {
                text,
                x: bx,
                y: by,
                width: bx_max.saturating_sub(bx),
                height: by_max.saturating_sub(by),
                font_size_pt: fsize,
                confidence: avg_conf,
                words: group,
                raw_words: Vec::new(),
            });
        }
        i += 2;
    }

    // ── Second pass: valley-based vertical split ─────────────────────
    let mut i = 0;
    while i < lines.len() {
        if lines[i].words.len() < 3 {
            i += 1;
            continue;
        }

        let y_min = match lines[i].words.iter().map(|w| w.y).min() {
            Some(v) => v,
            None => { i += 1; continue; }
        };
        let y_max = match lines[i].words.iter().map(|w| w.y + w.height).max() {
            Some(v) => v,
            None => { i += 1; continue; }
        };
        if y_max <= y_min {
            i += 1;
            continue;
        }

        let h = (y_max - y_min) as usize;

        // Confidence-weighted vertical projection profile.
        let mut profile = vec![0.0f32; h];
        for w in &lines[i].words {
            let top = w.y.saturating_sub(y_min) as usize;
            let bot = (w.y + w.height).saturating_sub(y_min) as usize;
            let weight = w.confidence / 100.0;
            for y_px in top..bot.min(h) {
                profile[y_px] += weight;
            }
        }

        // Valleys: contiguous runs where weighted coverage < 0.5.
        let threshold = 0.5f32;
        let mut valleys: Vec<(usize, usize)> = Vec::new();
        let mut in_valley = false;
        let mut vs = 0usize;
        for (y_px, &val) in profile.iter().enumerate() {
            if val < threshold {
                if !in_valley {
                    vs = y_px;
                    in_valley = true;
                }
            } else if in_valley {
                valleys.push((vs, y_px));
                in_valley = false;
            }
        }
        // Trailing valley (below all words) is ignored.

        // Must be interior and at least 2px wide.
        valleys.retain(|&(s, e)| e - s >= 2 && s > 0 && e < h);

        if valleys.is_empty() {
            i += 1;
            continue;
        }

        // Partition words into bands separated by valleys.
        let n_bands = valleys.len() + 1;
        let mut bands: Vec<Vec<TextRegion>> = vec![Vec::new(); n_bands];

        for w in &lines[i].words {
            let mid_y = (w.y + w.height / 2).saturating_sub(y_min) as usize;

            let in_valley_region = valleys.iter()
                .any(|&(s, e)| mid_y >= s && mid_y < e);

            if in_valley_region && w.confidence < 60.0 {
                continue; // drop low-confidence bridge word
            }

            let mut band = 0;
            for (vi, &(_, ve)) in valleys.iter().enumerate() {
                if mid_y >= ve {
                    band = vi + 1;
                }
            }
            bands[band].push(w.clone());
        }

        // Need at least 2 non-empty bands.
        let non_empty: Vec<Vec<TextRegion>> = bands.into_iter()
            .filter(|b| !b.is_empty())
            .collect();

        if non_empty.len() < 2 {
            i += 1;
            continue;
        }

        lines.remove(i);
        let n_new = non_empty.len();
        for (j, band_words) in non_empty.into_iter().enumerate() {
            let text = band_words.iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let bx = band_words.iter().map(|w| w.x).min().unwrap_or(0);
            let by = band_words.iter().map(|w| w.y).min().unwrap_or(0);
            let bx_max = band_words.iter().map(|w| w.x + w.width).max().unwrap_or(0);
            let by_max = band_words.iter().map(|w| w.y + w.height).max().unwrap_or(0);
            let avg_conf = band_words.iter().map(|w| w.confidence).sum::<f32>()
                / band_words.len() as f32;
            let fsize = band_words.first()
                .map(|w| w.font_size_pt).unwrap_or(12.0);

            lines.insert(i + j, TextLine {
                text,
                x: bx,
                y: by,
                width: bx_max.saturating_sub(bx),
                height: by_max.saturating_sub(by),
                font_size_pt: fsize,
                confidence: avg_conf,
                words: band_words,
                raw_words: Vec::new(),
            });
        }
        i += n_new;
    }

}

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
    }
}

// Scan the actual grayscale pixels to find true ink boundaries.
// ---------------------------------------------------------------------------

/// Walk columns outward from an edge, returning the furthest column that
/// contains non-background pixels.  `blur` is the threshold (bg − 15):
/// any pixel darker counts as ink or blur/bleed from real rasterisation.
/// The walk stops at the first fully-background column.
///
/// Rightward (direction > 0): checks columns `start..limit`, returns new
/// exclusive right edge.
/// Leftward (direction < 0): checks columns `(limit..start).rev()`,
/// returns new inclusive left edge.
fn walk_ink_edge(
    gray: &GrayImage,
    start: u32,
    limit: u32,
    y_top: u32,
    y_bot: u32,
    blur: u8,
    direction: i32,
) -> u32 {
    let page_w = gray.width();
    let mut edge = start;
    if direction > 0 {
        for col in start..limit.min(page_w) {
            if (y_top..y_bot).any(|row| gray.as_raw()[(row) as usize * gray.width() as usize + (col) as usize] < blur) {
                edge = col + 1;
            } else {
                break;
            }
        }
    } else {
        for col in (limit..start).rev() {
            if (y_top..y_bot).any(|row| gray.as_raw()[(row) as usize * gray.width() as usize + (col.min(page_w - 1)) as usize] < blur) {
                edge = col;
            } else {
                break;
            }
        }
    }
    edge
}

/// Walk rows outward, same semantics as `walk_ink_edge` but vertical.
/// Allows bridging small gaps (diacritics, i-dots) up to 5px, which is
/// less than inter-line whitespace (15-18px at 9pt/300dpi) so it won't
/// merge adjacent lines. Dot gap is ~4-5px.
fn walk_ink_edge_vertical(
    gray: &GrayImage,
    start: u32,
    limit: u32,
    x_left: u32,
    x_right: u32,
    blur: u8,
    direction: i32,
) -> u32 {
    let page_w = gray.width();
    let page_h = gray.height();
    let mut edge = start;
    const GAP_TOL: u32 = 5;
    if direction > 0 {
        let mut empty = 0u32;
        for row in start..limit.min(page_h) {
            if (x_left..x_right.min(page_w)).any(|col| gray.as_raw()[(row) as usize * gray.width() as usize + (col) as usize] < blur) {
                edge = row + 1;
                empty = 0;
            } else {
                empty += 1;
                if empty > GAP_TOL {
                    break;
                }
            }
        }
    } else {
        let mut empty = 0u32;
        for row in (limit..start).rev() {
            if (x_left..x_right.min(page_w)).any(|col| gray.as_raw()[(row) as usize * gray.width() as usize + (col) as usize] < blur) {
                edge = row;
                empty = 0;
            } else {
                empty += 1;
                if empty > GAP_TOL {
                    break;
                }
            }
        }
    }
    edge
}

/// Expand word bboxes when ink/blur is present beyond the current edge.
/// Uses `blur` threshold for expansion walks, `ink_threshold` for gates.
/// `margin` bounds edge-word and vertical searches.
///
/// Only **expands** — never shrinks. It assumes words do not overlap
/// heavily because overlaps are fixed later by `fix_overlapping_words_by_ink`
/// which uses column ink totals between centers.
pub fn expand_words_to_ink(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8, blur: u8, margin: u32) {
    let (page_w, page_h) = gray.dimensions();
    // Snapshot of original word bboxes for inter-line overlap checks.
    // Used to prevent vertical expansion from merging adjacent lines when
    // GAP_TOL would otherwise bridge the small inter-line whitespace.
    // We clone only positions, not the whole image.
    let orig_snapshot: Vec<Vec<(u32,u32,u32,u32)>> = lines
        .iter()
        .map(|l| l.words.iter().map(|w| (w.x, w.y, w.width, w.height)).collect())
        .collect();

    for (li_idx, line) in lines.iter_mut().enumerate() {
        let n = line.words.len();

        for i in 0..n {
            // ── Rightward expansion ─────────────────────────────────
            {
                let w = &line.words[i];
                let right_edge = w.x + w.width;
                let limit = if i + 1 < n {
                    line.words[i + 1].x
                } else {
                    (right_edge + margin).min(page_w)
                };

                if right_edge < limit && right_edge < page_w {
                    let check_col = right_edge.saturating_sub(1);
                    let y_top = w.y;
                    let y_bot = w.y + w.height;
                    let has_edge_ink = (y_top..y_bot)
                        .any(|row| gray.as_raw()[(row) as usize * gray.width() as usize + (check_col.min(page_w - 1)) as usize] < ink_threshold);

                    if has_edge_ink {
                        let new_right = walk_ink_edge(gray, right_edge, limit, y_top, y_bot, blur, 1);
                        if new_right > right_edge {
                            line.words[i].width = new_right - line.words[i].x;
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
                    left_edge.saturating_sub(margin)
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
                            .any(|row| gray.as_raw()[(row) as usize * gray.width() as usize + (col.min(page_w - 1)) as usize] < ink_threshold);
                        if col_has_ink {
                            shrink_to = col + 1;
                            break;
                        }
                        shrink_to = col;
                    }
                    if shrink_to < prev_right {
                        line.words[i - 1].width = shrink_to.saturating_sub(prev_x);
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
                                    let px = gray.as_raw()[(row) as usize * gray.width() as usize + (col.min(page_w - 1)) as usize];
                                    if px < ink_threshold { 1 } else { 0 }
                                })
                                .sum();
                            if ink == 0 {
                                gap_col = Some(col);
                                break;
                            }
                        }
                        if let Some(gc) = gap_col {
                            line.words[i - 1].width = gc.saturating_sub(prev_x);
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
                            .any(|row| gray.as_raw()[(row) as usize * gray.width() as usize + (col.min(page_w - 1)) as usize] < ink_threshold));

                    let has_edge_ink = (y_top..y_bot)
                        .any(|row| gray.as_raw()[(row) as usize * gray.width() as usize + (left_edge.min(page_w - 1)) as usize] < ink_threshold);

                    if has_edge_ink || gap_has_ink {
                        let new_left = walk_ink_edge(gray, left_edge, limit, y_top, y_bot, blur, -1);
                        if new_left < left_edge {
                            let growth = left_edge - new_left;
                            line.words[i].x = new_left;
                            line.words[i].width += growth;
                        }
                    }
                }
            }

            // ── Vertical expansion (bounded by ±margin) ─────────────
            // Prevent merging adjacent lines: clamp search range to stay
            // outside other lines' word bboxes that overlap horizontally.
            // This allows GAP_TOL to bridge internal i-dot gaps (4-5px)
            // which are inside the same line, but not the 3px inter-line
            // gap in the 9-line synthetic test.  Dot gap < inter-line gap
            // is false for that test, so we must distinguish by overlap
            // with other lines rather than by absolute distance.
            {
                let w = &line.words[i];
                let wx = w.x;
                let wr = (w.x + w.width).min(page_w);
                let word_top = w.y;
                let word_bot = w.y + w.height;
                let mut search_top = word_top.saturating_sub(margin);
                let mut search_bot = (word_bot + margin).min(page_h);

                // Find nearest overlapping word above/below from snapshot.
                let mut max_prev_bottom: Option<u32> = None;
                let mut min_next_top: Option<u32> = None;
                for (other_li, other_words) in orig_snapshot.iter().enumerate() {
                    if other_li == li_idx { continue; }
                    for &(ox, oy, ow, oh) in other_words {
                        // Horizontal overlap?
                        if ox >= wr || ox + ow <= wx { continue; }
                        if oy + oh <= word_top {
                            // Above
                            let bottom = oy + oh;
                            max_prev_bottom = Some(max_prev_bottom.map_or(bottom, |b| b.max(bottom)));
                        } else if oy >= word_bot {
                            let top = oy;
                            min_next_top = Some(min_next_top.map_or(top, |t| t.min(top)));
                        }
                    }
                }
                // Clamp to preserve at least 2px gap from neighboring lines.
                if let Some(prev_bot) = max_prev_bottom {
                    // Don't search into or past the previous line's ink.
                    // Keep at least 2px whitespace.
                    let clamped = prev_bot.saturating_add(2);
                    if clamped > search_top {
                        search_top = clamped.min(word_top);
                    }
                }
                if let Some(next_top) = min_next_top {
                    let clamped = next_top.saturating_sub(2);
                    if clamped < search_bot {
                        search_bot = clamped.max(word_bot);
                    }
                }

                let new_top = walk_ink_edge_vertical(gray, word_top, search_top, wx, wr, blur, -1);
                let new_bot = walk_ink_edge_vertical(gray, word_bot, search_bot, wx, wr, blur, 1);

                if new_top < word_top || new_bot > word_bot {
                    line.words[i].y = new_top;
                    line.words[i].height = new_bot - new_top;
                }
            }
        }
    }

    // Line bbox = union of expanded word bboxes.
    for line in lines.iter_mut() {
        if let (Some(x0), Some(y0), Some(x1), Some(y1)) = (
            line.words.iter().map(|w| w.x).min(),
            line.words.iter().map(|w| w.y).min(),
            line.words.iter().map(|w| w.x + w.width).max(),
            line.words.iter().map(|w| w.y + w.height).max(),
        ) {
            line.x = x0;
            line.y = y0;
            line.width = x1 - x0;
            line.height = y1 - y0;
        }
    }
}

/// Split words that contain wide internal bands of whitespace into separate
/// words.  Tesseract sometimes merges spaced-out characters (e.g. "0 1 2 3")
/// into a single word bbox with text "0123456789".  This function scans each
/// word's image for zero-ink column runs wider than a threshold and splits
/// the word at each gap.
///
/// When a `classifier` is provided, we do a quick CI font lookup on each word
/// and use the matched font's advance widths to determine the expected
/// inter-character gap.  We only split where the observed ink-to-ink gap
/// exceeds the font's natural spacing.  When no classifier is available,
/// falls back to a fixed threshold of 18% of line height.
///
/// Must run AFTER `expand_words_to_ink` so bboxes are ink-tight.
pub fn split_wide_whitespace_words(
    lines: &mut [TextLine],
    gray: &GrayImage,
    ink_threshold: u8,
    line_fonts: &[Option<std::sync::Arc<Vec<u8>>>],
) -> Vec<usize> {
    let (page_w, page_h) = gray.dimensions();
    let mut total_splits = 0u32;
    let mut split_line_indices: Vec<usize> = Vec::new();

    for (line_idx, line) in lines.iter_mut().enumerate() {
        let mut new_words: Vec<TextRegion> = Vec::new();
        let line_h = line.height;
        // Fallback: split at gaps ≥ 18% of the line height.
        let fallback_min_gap = (line_h * 18 / 100).max(4);

        // Use the pre-identified font for this line (from the main font matching pass)
        let line_font_ref = line_fonts.get(line_idx)
            .and_then(|opt| opt.as_ref())
            .and_then(|data| unprint_fonts::ab_glyph::FontRef::try_from_slice(data).ok());

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
            let (boundaries, _seams, _seg_summary) = crate::segment::segment_characters(&word_img, n_chars);

            // boundaries: [0, b1, b2, ..., ww] — n_chars+1 entries
            // Character i occupies columns [boundaries[i] .. boundaries[i+1])
            if boundaries.len() != n_chars + 1 {
                new_words.push(word);
                continue;
            }

            // Use the line-level font match for gap metric computation
            let matched_font_ref = line_font_ref.as_ref();

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
                            word_img.as_raw()[(row) as usize * word_img.width() as usize + (col as u32) as usize] < ink_threshold
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
                            word_img.as_raw()[(row) as usize * word_img.width() as usize + (col as u32) as usize] < ink_threshold
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
                            word_img.as_raw()[(row) as usize * word_img.width() as usize + (col as u32) as usize] < ink_threshold
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
                let min_gap_for_pair = matched_font_ref
                    .and_then(|font| {
                        let observed_ink_w = if right_ink > left_ink_i {
                            (right_ink - left_ink_i + 1) as f32
                        } else {
                            return None;
                        };
                        let ref_scale = 100.0_f32;
                        let ref_px = unprint_fonts::ab_glyph::PxScale { x: ref_scale, y: ref_scale };
                        let font_ink_w = font_ink_width(font, ref_px, chars[i])?;
                        if font_ink_w <= 0.0 { return None; }
                        let s = observed_ink_w / font_ink_w * ref_scale;
                        let scale = unprint_fonts::ab_glyph::PxScale { x: s, y: s };
                        let expected = font_pair_ink_gap(font, scale, chars[i], chars[i + 1]);
                        let threshold = expected.round() as u32 + 5;
                        Some(threshold)
                    })
                    .unwrap_or(fallback_min_gap);

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
                            gray.as_raw()[(row) as usize * gray.width() as usize + (col) as usize] < ink_threshold
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
            split_line_indices.push(line_idx);
        }

        // Replace words and rebuild line text
        line.words = new_words;
        line.text = line.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    }

    if total_splits > 0 {
    }
    split_line_indices
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
            if gray.as_raw()[(row) as usize * gray.width() as usize + (col) as usize] < threshold {
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

/// Return the ink width of a glyph in pixels from outline bounds (no rasterization).
fn font_ink_width<F: Font>(font: &F, scale: PxScale, ch: char) -> Option<f32> {
    let gid = font.glyph_id(ch);
    let glyph = gid.with_scale_and_position(scale, point(0.0, 0.0));
    let outlined = font.outline_glyph(glyph)?;
    let b = outlined.px_bounds();
    Some(b.max.x - b.min.x)
}

/// Return the expected ink-to-ink gap between two adjacent glyphs from outline bounds.
fn font_pair_ink_gap<F: Font>(font: &F, scale: PxScale, ch_a: char, ch_b: char) -> f32 {
    let sf = font.as_scaled(scale);
    let gid_a = font.glyph_id(ch_a);
    let gid_b = font.glyph_id(ch_b);
    let adv_a = sf.h_advance(gid_a);

    let glyph_a = gid_a.with_scale_and_position(scale, point(0.0, 0.0));
    let glyph_b = gid_b.with_scale_and_position(scale, point(0.0, 0.0));

    let outlined_a = match font.outline_glyph(glyph_a) {
        Some(o) => o,
        None => return 0.0,
    };
    let outlined_b = match font.outline_glyph(glyph_b) {
        Some(o) => o,
        None => return 0.0,
    };

    let bounds_a = outlined_a.px_bounds();
    let bounds_b = outlined_b.px_bounds();

    // gap = advance_a - ink_right_a + ink_left_b
    (adv_a - bounds_a.max.x + bounds_b.min.x).max(0.0)
}


// ---------------------------------------------------------------------------
// Detect stage entry point
// ---------------------------------------------------------------------------

/// Assemble and post-process text lines from raw OCR word regions.
///
/// Performs: line assembly → raw-bbox snapshot → overlap merging (currently
/// disabled) → split merged lines → drop.
///
/// Does NOT include `expand_words_to_ink` (needs ink threshold from
/// background-colour detection), `fix_overlapping_words_by_ink` (needs gray and
/// ink threshold, finds natural gap between overlapping words by looking at
/// column ink totals between centers), or `split_wide_whitespace_words` (needs
/// matched font data). Those run as separate refinement passes in
/// `page_cache.rs`.
pub fn postprocess_words(word_regions: &[TextRegion]) -> Vec<TextLine> {
    let mut lines = assemble_lines(word_regions);
    snapshot_raw_bboxes(&mut lines);
    merge_overlapping_lines(&mut lines);
    split_merged_lines(&mut lines);
    drop_outlier_words(&mut lines);
    lines
}
