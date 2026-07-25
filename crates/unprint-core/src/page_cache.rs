//! Page-level cache for rasterized images and OCR results.
//!
//! Cache location: `/tmp/unprint-page-cache/<key>/`
//! Key: `<filename>-<file_size>-<dpi>`
//!
//! Staleness: if the source PDF is newer than the cached page-0 image,
//! the cache is invalidated and pages are re-rasterized.
//!
//! Per page:
//!   - `page-N.png`       — rasterized page image
//!   - `page-N-ocr.json`  — OCR word regions + char boxes

use crate::ocr::{CharBox, TextRegion};
use image::DynamicImage;
use std::path::{Path, PathBuf};

/// Cached OCR results for a single page.
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedOcr {
    word_regions: Vec<TextRegion>,
    char_boxes: Vec<CharBox>,
}

/// Build a cache key string from input file metadata + DPI.
/// The key is stable across runs for the same file path + size + DPI;
/// staleness is checked separately via `is_cache_stale`.
pub fn cache_key(path: &Path, dpi: u32) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    // Sanitize filename: replace anything that isn't alphanumeric, dot, or hyphen
    let safe_name: String = file_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    Some(format!("{}-{}-{}dpi", safe_name, size, dpi))
}

/// Return true if the source file is newer than the cached page-0 image,
/// meaning the cache is stale and should be regenerated.
pub fn is_cache_stale(cache_dir: &Path, source: &Path) -> bool {
    let source_mtime = match std::fs::metadata(source).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return true, // can't stat source → treat as stale
    };
    let cached_page0 = cache_dir.join("page-0.png");
    match std::fs::metadata(&cached_page0).and_then(|m| m.modified()) {
        Ok(cache_mtime) => source_mtime > cache_mtime,
        Err(_) => true, // no cached file → stale
    }
}

/// Return the cache directory for a given key.
pub fn cache_dir(key: &str) -> Option<PathBuf> {
    let base = std::env::var("TMPDIR").ok().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"));
    Some(base.join("unprint-page-cache").join(key))
}

/// Try to load a cached page image from disk.
pub fn load_cached_image(dir: &Path, page_idx: usize) -> Option<DynamicImage> {
    let png_path = dir.join(format!("page-{}.png", page_idx));
    image::open(&png_path).ok()
}

/// Try to load cached OCR results from disk.
pub fn load_cached_ocr(dir: &Path, page_idx: usize) -> Option<(Vec<TextRegion>, Vec<CharBox>)> {
    let json_path = dir.join(format!("page-{}-ocr.json", page_idx));
    let data = std::fs::read_to_string(&json_path).ok()?;
    let cached: CachedOcr = serde_json::from_str(&data).ok()?;
    Some((cached.word_regions, cached.char_boxes))
}

/// Save a page image to the cache.
pub fn save_cached_image(dir: &Path, page_idx: usize, img: &DynamicImage) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let png_path = dir.join(format!("page-{}.png", page_idx));
    let _ = img.save(&png_path);
}

/// Save OCR results to the cache.
pub fn save_cached_ocr(
    dir: &Path,
    page_idx: usize,
    word_regions: &[TextRegion],
    char_boxes: &[CharBox],
) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let json_path = dir.join(format!("page-{}-ocr.json", page_idx));
    let cached = CachedOcr {
        word_regions: word_regions.to_vec(),
        char_boxes: char_boxes.to_vec(),
    };
    if let Ok(data) = serde_json::to_string(&cached) {
        let _ = std::fs::write(&json_path, data);
    }
}

// ---------------------------------------------------------------------------
// Rasterize stage entry point
// ---------------------------------------------------------------------------

/// Load rasterized page images for an input file, with transparent caching.
///
/// Returns `(pages, cached)` where `cached` is true when all pages came
/// from the on-disk cache.  Handles staleness checks, cache miss
/// rasterization, and write-back automatically.
pub fn get_pages(
    input: &Path,
    dpi: u32,
) -> Result<(Vec<DynamicImage>, bool), crate::error::ScanTextError> {
    let cdir = cache_key(input, dpi).and_then(|key| cache_dir(&key));

    if let Some(ref dir) = cdir {
        if !is_cache_stale(dir, input) {
            if let Some(first) = load_cached_image(dir, 0) {
                let mut pages = vec![first];
                let mut idx = 1;
                while let Some(img) = load_cached_image(dir, idx) {
                    pages.push(img);
                    idx += 1;
                }
                return Ok((pages, true));
            }
        }
    }

    // Cache miss — rasterize fresh.
    let pages = rasterize(input, dpi)?;

    // Write back to cache, clearing stale OCR so prepare_page re-runs Tesseract.
    if let Some(ref dir) = cdir {
        if dir.exists() {
            for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().ends_with("-ocr.json") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        for (i, img) in pages.iter().enumerate() {
            save_cached_image(dir, i, img);
        }
    }

    Ok((pages, false))
}

/// Rasterize an input file (PDF or image) into page images.
fn rasterize(path: &Path, dpi: u32) -> Result<Vec<DynamicImage>, crate::error::ScanTextError> {
    use crate::error::ScanTextError;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "pdf" => rasterize_pdf(path, dpi),
        "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "gif" | "webp" => {
            let img = image::open(path).map_err(|e| ScanTextError::ImageLoad(e.to_string()))?;
            Ok(vec![img])
        }
        _ => Err(ScanTextError::UnsupportedFormat(ext)),
    }
}

fn rasterize_pdf(path: &Path, dpi: u32) -> Result<Vec<DynamicImage>, crate::error::ScanTextError> {
    use crate::error::ScanTextError;
    use std::process::Command;

    let tmp_dir = tempfile::tempdir().map_err(ScanTextError::Io)?;
    let prefix = tmp_dir.path().join("page");

    let status = Command::new("pdftoppm")
        .args([
            "-r",
            &dpi.to_string(),
            "-png",
            &path.to_string_lossy(),
            &prefix.to_string_lossy(),
        ])
        .status()
        .map_err(|e| {
            ScanTextError::ImageLoad(format!("Failed to run pdftoppm (install poppler-utils): {e}"))
        })?;
    if !status.success() {
        return Err(ScanTextError::ImageLoad("pdftoppm exited with error".into()));
    }

    let mut pngs: Vec<_> = std::fs::read_dir(tmp_dir.path())
        .map_err(ScanTextError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    pngs.sort();

    let mut pages = Vec::new();
    for png_path in &pngs {
        pages.push(image::open(png_path).map_err(|e| ScanTextError::ImageLoad(e.to_string()))?);
    }
    if pages.is_empty() {
        return Err(ScanTextError::ImageLoad("pdftoppm produced no pages".into()));
    }
    Ok(pages)
}


// ---------------------------------------------------------------------------
// Page preparation: deskew → OCR (cached) → background colour → ink expansion
// ---------------------------------------------------------------------------

/// Result of preparing a single page for font matching.
pub struct PreparedPage {
    /// OCR text lines with word bboxes expanded to actual ink.
    pub lines: Vec<crate::ocr::TextLine>,
    /// Deskewed grayscale image for character segmentation / matching.
    pub gray: image::GrayImage,
    /// Dominant background colour.
    pub bg_color: crate::color::Rgb,
    /// Ink threshold (bg − 56, saturating) used for binarisation.
    pub ink_thresh: u8,
}

/// Deskew the page, run OCR (with disk cache), detect background colour,
/// and expand word bounding boxes to actual ink extent.
pub fn prepare_page(
    page_img: &DynamicImage,
    page_idx: usize,
    dpi: u32,
    cache_dir: Option<&std::path::Path>,
) -> Result<PreparedPage, crate::error::ScanTextError> {
    // Deskew
    let orig_gray = page_img.to_luma8();
    let skew_angle = crate::deskew::detect_skew(&orig_gray);
    let deskewed_gray = if skew_angle.abs() > 5.0 {
        // Large skew: treat as not deskewing (original was cloned before, now moved to avoid page-size clone)
        orig_gray
    } else if skew_angle.abs() > 0.5 {
        crate::deskew::rotate_gray(&orig_gray, skew_angle)
    } else {
        orig_gray
    };

    // OCR (with cache) — direct GrayImage path avoids DynamicImage clone + to_luma8 clone
    let word_regions = if let Some((wr, _cb)) =
        cache_dir.and_then(|d| load_cached_ocr(d, page_idx))
    {
        wr
    } else {
        let (wr, cb) = crate::ocr::extract_text_regions_from_gray(&deskewed_gray, dpi)?;
        if let Some(cdir) = cache_dir {
            save_cached_ocr(cdir, page_idx, &wr, &cb);
        }
        wr
    };
    let mut lines = crate::ocr::postprocess_words(&word_regions);

    // Background colour
    let bg_color = crate::color::detect_background_color(page_img);

    // Expand word bboxes to actual ink
    let ink_thresh = bg_color.0.saturating_sub(56);
    let blur_thresh = bg_color.0.saturating_sub(15);
    crate::ocr::expand_words_to_ink(&mut lines, &deskewed_gray, ink_thresh, blur_thresh, 20);
    crate::ocr::fix_overlapping_words_by_ink(&mut lines, &deskewed_gray, ink_thresh);
    crate::ocr::trim_words_to_ink(&mut lines, &deskewed_gray, ink_thresh);

    Ok(PreparedPage { lines, gray: deskewed_gray, bg_color, ink_thresh })
}
