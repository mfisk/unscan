//! Page-level cache for rasterized images and OCR results.
//!
//! Cache location: `$TMPDIR/unprint-page-cache/<key>/`
//! Key: `<filename>-<file_size>-<dpi>`
//!
//! Staleness: source-meta.json stores mtime+size. If either changes, cache is
//! invalidated. Legacy fallback: if source is newer than cached page-0.png,
//! invalidate.
//!
//! Per page:
//!   - `page-N.png`       — rasterized page image
//!   - `page-N-ocr.json`  — OCR word regions + char boxes
//!   - `source-meta.json` — mtime + size for invalidation

use crate::ocr::{CharBox, TextRegion};
use image::{DynamicImage, GrayImage, Luma};
use std::path::{Path, PathBuf};

/// Cached OCR results for a single page.
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedOcr {
    word_regions: Vec<TextRegion>,
    char_boxes: Vec<CharBox>,
}

/// Source file metadata used for cache invalidation.
/// Stored as `source-meta.json` in the cache dir.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
struct SourceMeta {
    size: u64,
    mtime_secs: u64,
    mtime_nanos: u32,
}

fn current_source_meta(path: &Path) -> Option<SourceMeta> {
    let md = std::fs::metadata(path).ok()?;
    let size = md.len();
    let mtime = md.modified().ok()?;
    let dur = mtime.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(SourceMeta {
        size,
        mtime_secs: dur.as_secs(),
        mtime_nanos: dur.subsec_nanos(),
    })
}

fn source_meta_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("source-meta.json")
}

fn load_source_meta(cache_dir: &Path) -> Option<SourceMeta> {
    let p = source_meta_path(cache_dir);
    let data = std::fs::read_to_string(&p).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_source_meta(cache_dir: &Path, meta: &SourceMeta) {
    if std::fs::create_dir_all(cache_dir).is_err() {
        return;
    }
    let p = source_meta_path(cache_dir);
    if let Ok(s) = serde_json::to_string(meta) {
        let _ = std::fs::write(p, s);
    }
}

/// Build a cache key string from input file metadata + DPI.
/// The key is stable across runs for the same file path + size + DPI;
/// staleness is checked separately via `is_cache_stale`.
pub fn cache_key(path: &Path, dpi: u32) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    let safe_name: String = file_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    Some(format!("{}-{}-{}dpi", safe_name, size, dpi))
}

/// Return true if the source file is newer / different size than the cached
/// metadata, meaning the cache is stale.
pub fn is_cache_stale(cache_dir: &Path, source: &Path) -> bool {
    let cur = match current_source_meta(source) {
        Some(m) => m,
        None => return true,
    };
    if let Some(stored) = load_source_meta(cache_dir) {
        if stored.size != cur.size
            || stored.mtime_secs != cur.mtime_secs
            || stored.mtime_nanos != cur.mtime_nanos
        {
            return true;
        }
        // meta matches → not stale, even if page-0.png mtime is older (we wrote meta after)
        return false;
    }
    // Legacy fallback: compare mtime vs page-0.png
    let cached_page0 = cache_dir.join("page-0.png");
    match (std::fs::metadata(source).and_then(|m| m.modified()), std::fs::metadata(&cached_page0).and_then(|m| m.modified())) {
        (Ok(src_mtime), Ok(cache_mtime)) => src_mtime > cache_mtime,
        _ => true,
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
// Fast single-raster extraction (all pages are one DeviceGray Flate Image)
// ---------------------------------------------------------------------------

/// Try to extract pages that are each a single raster image (scan PDFs) without
/// invoking `pdftoppm`.
///
/// Criteria:
/// - PDF loads with `lopdf`
/// - Every page has exactly one image XObject covering the page
/// - Image is 8bpc DeviceGray or DeviceRGB, filter FlateDecode or DCTDecode or no filter (raw)
/// - Width*height matches decompressed size (or JPEG decodes)
/// - Returns Gray/RGB converted to DynamicImage
///
/// On any mismatch we return `None` so caller falls back to pdftoppm.
fn try_extract_raster_pages(path: &Path) -> Option<Vec<DynamicImage>> {
    use std::io::Read;

    let doc = lopdf::Document::load(path).ok()?;
    let pages = doc.get_pages();
    if pages.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(pages.len());

    // pages BTreeMap<u32, ObjectId> sorted by page number already, but get_pages()
    // uses page_iter which is in document order. BTreeMap ensures 1..N ordering.
    for (_pnum, &page_id) in pages.iter() {
        let images = doc.get_page_images(page_id).ok()?;
        if images.len() != 1 {
            return None;
        }
        let img = &images[0];

        // Validate basics
        let w = img.width as u32;
        let h = img.height as u32;
        if w == 0 || h == 0 || w > 10000 || h > 10000 {
            return None;
        }
        let bpc = img.bits_per_component.unwrap_or(8);
        if bpc != 8 {
            // Only 8bpc fast path for now; could add 1bpc later
            return None;
        }
        let cs = img.color_space.as_deref().unwrap_or("DeviceGray");
        let is_gray = cs == "DeviceGray" || cs == "DeviceGray" || cs.contains("Gray");
        let is_rgb = cs == "DeviceRGB" || cs.contains("RGB");
        if !is_gray && !is_rgb {
            return None;
        }

        let filters = img.filters.clone().unwrap_or_default();
        let data: Vec<u8> = if filters.iter().any(|f| f == "FlateDecode" || f == "Fl") {
            // Decompress zlib/deflate stream
            // lopdf stores raw compressed bytes in `content`; decompress via flate2
            let mut decoder = flate2::read::ZlibDecoder::new(img.content);
            let mut buf = Vec::with_capacity((w as usize) * (h as usize) * if is_gray {1} else {3});
            if decoder.read_to_end(&mut buf).is_err() {
                // Fallback: try raw deflate (no zlib header) – some producers use it
                let mut decoder2 = flate2::read::DeflateDecoder::new(img.content);
                buf.clear();
                if decoder2.read_to_end(&mut buf).is_err() {
                    return None;
                }
                buf
            } else {
                buf
            }
        } else if filters.iter().any(|f| f == "DCTDecode" || f == "DCT") {
            // JPEG bytes – decode via image crate
            let dyn_img = image::load_from_memory(img.content).ok()?;
            out.push(dyn_img);
            continue;
        } else if filters.is_empty() {
            // Raw uncompressed
            img.content.to_vec()
        } else {
            // Unsupported filter (JPXDecode, etc) → fall back
            return None;
        };

        // Validate decompressed size
        let expected = (w as usize) * (h as usize) * if is_gray {1} else {3};
        if data.len() != expected {
            // Some PDFs have PNG predictor bytes (1 byte per row)
            let row_len = w as usize * if is_gray {1} else {3};
            if data.len() == (row_len + 1) * h as usize {
                let mut stripped = Vec::with_capacity(expected);
                let mut offset = 0usize;
                for _ in 0..h {
                    offset += 1; // predictor tag
                    if offset + row_len > data.len() { return None; }
                    stripped.extend_from_slice(&data[offset..offset+row_len]);
                    offset += row_len;
                }
                if is_gray {
                    let gray = GrayImage::from_raw(w, h, stripped)?;
                    out.push(DynamicImage::ImageLuma8(gray));
                } else {
                    let rgb = image::RgbImage::from_raw(w, h, stripped)?;
                    let dyn_rgb = DynamicImage::ImageRgb8(rgb);
                    out.push(dyn_rgb);
                }
                continue;
            } else {
                return None;
            }
        }

        if is_gray {
            let gray = GrayImage::from_raw(w, h, data)?;
            out.push(DynamicImage::ImageLuma8(gray));
        } else {
            // DeviceRGB → to_luma8 for OCR pipeline compatibility (same as pdftoppm png → to_luma8 later)
            let rgb = image::RgbImage::from_raw(w, h, data)?;
            let dyn_rgb = DynamicImage::ImageRgb8(rgb);
            out.push(dyn_rgb);
        }
    }

    if out.len() == pages.len() {
        eprintln!("fast raster extract: {} pages via Flate/DCT ({}x{} DeviceGray)", out.len(), out[0].width(), out[0].height());
        Some(out)
    } else {
        None
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
    let cur_meta = current_source_meta(input);

    if let Some(ref dir) = cdir {
        if !is_cache_stale(dir, input) {
            if let Some(first) = load_cached_image(dir, 0) {
                let mut pages = vec![first];
                let mut idx = 1;
                while let Some(img) = load_cached_image(dir, idx) {
                    pages.push(img);
                    idx += 1;
                }
                eprintln!("page-cache hit: {} pages from {}", pages.len(), dir.display());
                return Ok((pages, true));
            }
        } else if dir.exists() {
            // Stale → remove old OCR jsons but keep dir for reuse; source-meta will be overwritten later
            eprintln!("page-cache stale (mtime/size mismatch): {}", dir.display());
        }
    }

    // Cache miss — try fast single-raster extraction before pdftoppm.
    if let Some(fast_pages) = try_extract_raster_pages(input) {
        if let Some(ref dir) = cdir {
            if dir.exists() {
                for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                    let name = entry.file_name();
                    if name.to_string_lossy().ends_with("-ocr.json") {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
            for (i, img) in fast_pages.iter().enumerate() {
                save_cached_image(dir, i, img);
            }
            if let Some(ref meta) = cur_meta {
                save_source_meta(dir, meta);
            }
        }
        return Ok((fast_pages, false));
    }

    // Fallback: rasterize via pdftoppm.
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
        if let Some(ref meta) = cur_meta {
            save_source_meta(dir, meta);
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
