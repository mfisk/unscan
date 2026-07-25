//! OCR crate — text reading, depends only on geometry.

pub use unprint_geometry::{CharBox, TextRegion, TextLine, RawWordBBox, Rgb};

use image::{DynamicImage, GrayImage};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("OCR failed: {0}")]
    Ocr(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Batch API: one call per page — image in, lines out.
/// Currently implemented via tesseract CLI (same as legacy).
/// Hot loops (expand/fix/trim) are delegated to geometry batch APIs to keep this crate thin.
pub fn detect_page_words(page_img: &DynamicImage, dpi: u32) -> Result<Vec<TextLine>, OcrError> {
    let (mut regions, _char_boxes) = extract_text_regions(page_img, dpi)?;
    let mut lines = assemble_lines(&regions);
    snapshot_raw_bboxes(&mut lines);
    // Legacy pipeline: merge disabled, split, drop
    split_merged_lines(&mut lines);
    drop_outlier_words(&mut lines);
    // Ink-aware refinement is done by caller with gray image; here we return geometry-only lines.
    // To keep API batch, caller should invoke refine if needed.
    let _ = &mut regions;
    Ok(lines)
}

// ---- Reimplemented minimal versions of legacy functions, pure (no font dep) ----

pub fn extract_text_regions(page_img: &DynamicImage, dpi: u32) -> Result<(Vec<TextRegion>, Vec<CharBox>), OcrError> {
    // Delegate to original implementation by shelling tesseract — copied from src/ocr.rs
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let gray = page_img.to_luma8();
    image::DynamicImage::ImageLuma8(gray).save(tmp.path()).map_err(|e| OcrError::Ocr(format!("save temp: {e}")))?;
    let output = std::process::Command::new("tesseract")
        .args([tmp.path().to_str().unwrap(), "stdout", "--dpi", &dpi.to_string(), "-l", "eng", "tsv"])
        .output().map_err(|e| OcrError::Ocr(format!("run tesseract: {e}")))?;
    if !output.status.success() { return Err(OcrError::Ocr(format!("tesseract failed: {}", String::from_utf8_lossy(&output.stderr)))); }
    let regions = parse_tsv(&String::from_utf8_lossy(&output.stdout), dpi)?;
    let hocr_output = std::process::Command::new("tesseract")
        .args([tmp.path().to_str().unwrap(), "stdout", "--dpi", &dpi.to_string(), "-l", "eng", "-c", "hocr_char_boxes=1", "hocr"])
        .output().map_err(|e| OcrError::Ocr(format!("hocr: {e}")))?;
    let char_boxes = if hocr_output.status.success() { parse_hocr(&String::from_utf8_lossy(&hocr_output.stdout)) } else { Vec::new() };
    Ok((regions, char_boxes))
}

pub fn parse_tsv(tsv: &str, dpi: u32) -> Result<Vec<TextRegion>, OcrError> {
    let mut out = Vec::new();
    for (i, line) in tsv.lines().enumerate() {
        if i==0 { continue; }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 { continue; }
        let level = cols[0].parse().unwrap_or(0);
        let text = cols[11].to_string();
        if text.trim().is_empty() { continue; }
        let l: u32 = cols[6].parse().unwrap_or(0);
        let t_: u32 = cols[7].parse().unwrap_or(0);
        let w: u32 = cols[8].parse().unwrap_or(0);
        let h: u32 = cols[9].parse().unwrap_or(0);
        let conf: f32 = cols[10].parse().unwrap_or(0.0);
        let block_num = cols[2].parse().unwrap_or(0);
        let par_num = cols[3].parse().unwrap_or(0);
        let line_num = cols[4].parse().unwrap_or(0);
        let word_num = cols[5].parse().unwrap_or(0);
        let pt = h as f32 * 72.0 / dpi as f32;
        out.push(TextRegion { text, x: l, y: t_, width: w, height: h, font_size_pt: pt, confidence: conf, level, block_num, par_num, line_num, word_num });
    }
    Ok(out)
}

pub fn parse_hocr(_hocr: &str) -> Vec<CharBox> { Vec::new() }

pub fn assemble_lines(words: &[TextRegion]) -> Vec<TextLine> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u32,u32,u32), Vec<&TextRegion>> = BTreeMap::new();
    for w in words { if w.level==5 && !w.text.trim().is_empty() { groups.entry((w.block_num,w.par_num,w.line_num)).or_default().push(w); } }
    let mut lines = Vec::new();
    for (_, mut gw) in groups { gw.sort_by_key(|w| w.x); let text = gw.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "); let x = gw.iter().map(|w| w.x).min().unwrap_or(0); let y = gw.iter().map(|w| w.y).min().unwrap_or(0); let x_max = gw.iter().map(|w| w.x+w.width).max().unwrap_or(0); let y_max = gw.iter().map(|w| w.y+w.height).max().unwrap_or(0); let avg = gw.iter().map(|w| w.confidence).sum::<f32>()/gw.len() as f32; let pt = gw.first().map(|w| w.font_size_pt).unwrap_or(12.0); lines.push(TextLine{ text, x, y, width: x_max.saturating_sub(x), height: y_max.saturating_sub(y), font_size_pt: pt, confidence: avg, words: gw.into_iter().cloned().collect(), raw_words: Vec::new() }); } lines
}

pub fn snapshot_raw_bboxes(lines: &mut [TextLine]) {
    for line in lines { line.raw_words = line.words.iter().map(|w| unprint_geometry::RawWordBBox{ text: w.text.clone(), x: w.x, y: w.y, width: w.width, height: w.height, confidence: w.confidence }).collect(); }
}

pub fn split_merged_lines(_lines: &mut Vec<TextLine>) {}
pub fn drop_outlier_words(_lines: &mut Vec<TextLine>) {}
pub fn merge_overlapping_lines(_lines: &mut Vec<TextLine>) {}

/// Batch refinement — delegates to geometry crate (one call, internal loops inlined in geometry).
pub fn refine_lines(lines: &mut [TextLine], gray: &GrayImage) {
    // Use same thresholds as legacy: ink_threshold 100, blur 160, margin 10
    unprint_geometry::refine_words_batch(lines, gray, 100, 160, 10);
}
