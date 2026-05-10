//! OCR module — runs Tesseract via CLI to extract text with bounding boxes.

use crate::error::ScanTextError;
use image::DynamicImage;
use log::debug;
use std::process::Command;

/// A detected text region from OCR (word-level).
#[derive(Debug, Clone)]
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
    pub word_num: u32,
}

/// A line of text assembled from word-level regions.
#[derive(Debug, Clone)]
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

/// Run Tesseract on a page image and return word-level regions.
/// Applies contrast enhancement and sharpening before OCR to improve
/// recognition on scanned documents.
pub fn extract_text_regions(
    page_img: &DynamicImage,
    dpi: u32,
) -> Result<Vec<TextRegion>, ScanTextError> {
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

    parse_tsv(&String::from_utf8_lossy(&output.stdout), dpi)
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
