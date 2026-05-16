//! Page-level cache for rasterized images and OCR results.
//!
//! Cache location: `~/.cache/unscan/page-cache/<key>/`
//! Key: `<filename>-<file_size>-<mtime_secs>-<dpi>`
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
pub fn cache_key(path: &Path, dpi: u32) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    // Sanitize filename: replace anything that isn't alphanumeric, dot, or hyphen
    let safe_name: String = file_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    Some(format!("{}-{}-{}-{}dpi", safe_name, size, mtime, dpi))
}

/// Return the cache directory for a given key.
pub fn cache_dir(key: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".cache/unscan/page-cache").join(key))
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
