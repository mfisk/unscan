mod audit;
mod cli;
mod color;
mod deskew;
mod error;
mod font_cache;
mod font_match;
mod font_scan;
mod geometry;
pub mod char_index;
pub mod layout;
mod compare;
pub mod seg_diag;
pub mod ssim;
mod ocr;
mod page_cache;
mod pdf_out;
mod segment;
mod smooth;
// mod word_match; // disabled: CI ranking used directly, word-level SSIM rerank removed
pub(crate) mod verify;

use crate::audit::{AuditEntry, AuditLog, BBox, Decision, GeometryEntry, PageSummary};

fn mem_info() -> String {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0u64;
    let mut vsz = 0u64;
    for l in s.lines() {
        if l.starts_with("VmRSS:") {
            rss = l.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        }
        if l.starts_with("VmSize:") {
            vsz = l.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        }
    }
    format!("RSS={}MB VSZ={}MB", rss / 1024, vsz / 1024)
}

fn dump_limits() {
    if let Ok(s) = std::fs::read_to_string("/proc/self/limits") {
        for l in s.lines() {
            if l.contains("address space") || l.contains("data size") || l.contains("stack size") {
                eprintln!("  LIMIT: {}", l);
            }
        }
    }
    // Check overcommit and committed memory
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for l in s.lines() {
            if l.starts_with("CommitLimit:") || l.starts_with("Committed_AS:") || l.starts_with("MemAvailable:") {
                eprintln!("  MEMINFO: {}", l.trim());
            }
        }
    }
    if let Ok(s) = std::fs::read_to_string("/proc/sys/vm/overcommit_memory") {
        eprintln!("  OVERCOMMIT: {}", s.trim());
    }
    // cgroup memory limit
    for path in &["/sys/fs/cgroup/memory/memory.limit_in_bytes",
                   "/sys/fs/cgroup/memory.max"] {
        if let Ok(s) = std::fs::read_to_string(path) {
            eprintln!("  CGROUP: {} = {}", path, s.trim());
        }
    }
}
use crate::error::ScanTextError;
use crate::ocr::TextRegion;
use image::DynamicImage;
use log::{debug, info, warn};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Minimum SSIM score for SSIM verification to consider a font match acceptable.
/// Correct matches on scanned documents typically score 0.5–0.8; truly wrong fonts
/// score much lower. 0.3 catches garbage without false-rejecting legitimate matches.
const MIN_VERIFY_SSIM: f32 = 0.3;

/// Standalone char rendering: render characters using the index-time
/// render_char_normalised() pipeline and save as PNGs.
fn render_ref_chars_and_exit(json_str: &str) -> ! {
    use ab_glyph::{Font, FontVec};

    #[derive(serde::Deserialize)]
    struct Req {
        font: String,
        chars: String,
        output_dir: String,
    }

    let req: Req = serde_json::from_str(json_str).unwrap_or_else(|e| {
        eprintln!("Invalid --render-ref-chars JSON: {e}");
        std::process::exit(1);
    });

    let font_data = std::fs::read(&req.font).unwrap_or_else(|e| {
        eprintln!("Cannot read font {:?}: {e}", req.font);
        std::process::exit(1);
    });
    let font = FontVec::try_from_vec(font_data).unwrap_or_else(|e| {
        eprintln!("Cannot parse font {:?}: {e}", req.font);
        std::process::exit(1);
    });

    let out = std::path::Path::new(&req.output_dir);
    std::fs::create_dir_all(out).unwrap_or_else(|e| {
        eprintln!("Cannot create output dir {:?}: {e}", req.output_dir);
        std::process::exit(1);
    });

    let mut rendered = 0u32;
    for c in req.chars.chars() {
        if font.glyph_id(c).0 == 0 {
            continue; // no glyph for this char
        }
        if let Some(img) = char_index::render_char_normalised(&font, c) {
            let fname = format!("U+{:04X}.png", c as u32);
            let _ = img.save(out.join(&fname));
            rendered += 1;
        }
    }
    eprintln!("Rendered {rendered} chars to {:?}", req.output_dir);
    std::process::exit(0);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let args = cli::parse();
    if let Err(msg) = args.validate() {
        eprintln!("Error: {msg}");
        std::process::exit(1);
    }

    // ── render-ref-chars: standalone char rendering, no PDF needed ───
    if let Some(ref json_str) = args.render_ref_chars {
        render_ref_chars_and_exit(json_str);
    }

    if args.index {
        if let Err(e) = run_index(&args) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    } else {
        if let Err(e) = run(&args) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// Scan available fonts, compare against the cached index, and incrementally
/// update: add new fonts, remove stale fonts, or report "Index is current".
/// With `--rebuild-index`, forces a full rebuild.
fn run_index(args: &cli::Args) -> Result<(), ScanTextError> {
    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    info!("Scanning for fonts…");
    let font_catalog = font_scan::scan_fonts(&font_dirs);
    info!("Found {} system fonts", font_catalog.len());
    if font_catalog.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    let index_path = args.resolved_index_path();

    // Collect system font keys → paths + glyph overrides for lookup.
    // Keys are unique per weight/style/variant (path + optional variant tag).
    let system_fonts: std::collections::HashMap<String, (PathBuf, char_index::GlyphOverrides)> = font_catalog
        .iter()
        .map(|e| (e.font_key(), (e.path.clone(), e.glyph_overrides.clone())))
        .collect();
    let system_names: std::collections::HashSet<String> =
        system_fonts.keys().cloned().collect();

    // ── Full rebuild path ──────────────────────────────────────────
    if args.rebuild_index {
        info!("Forced full rebuild requested");
        return do_full_build(&system_fonts, &index_path);
    }

    // ── Try loading existing index ─────────────────────────────────
    if !index_path.exists() {
        info!("No cached index found — building from scratch");
        return do_full_build(&system_fonts, &index_path);
    }

    // Fast header check first (12 bytes, no full deserialize).
    match char_index::peek_header(&index_path) {
        Ok((version, feat_len)) => {
            let (exp_ver, exp_fl) = char_index::expected_header();
            if version != exp_ver || feat_len != exp_fl {
                info!(
                    "Index format changed (v{version}/feat{feat_len} → v{exp_ver}/feat{exp_fl}) — full rebuild"
                );
                return do_full_build(&system_fonts, &index_path);
            }
        }
        Err(e) => {
            info!("Cannot read index header ({e}) — full rebuild");
            return do_full_build(&system_fonts, &index_path);
        }
    }

    // Header OK — load the full index for incremental comparison.
    info!("Loading cached index from {}", index_path.display());
    let start = std::time::Instant::now();
    let mut index = match char_index::load_index(&index_path) {
        Ok(idx) => idx,
        Err(e) => {
            warn!("Failed to load index ({e}) — full rebuild");
            return do_full_build(&system_fonts, &index_path);
        }
    };
    let load_time = start.elapsed();
    let indexed_names = index.font_names(); // includes both indexed + skipped
    let indexed_count = index.indexed_font_names().len();
    let skipped_count = index.skipped_fonts.len();
    info!(
        "  Loaded {} fonts ({} indexed, {} skipped) in {:.1}s",
        indexed_names.len(),
        indexed_count,
        skipped_count,
        load_time.as_secs_f64()
    );

    // ── Diff: new vs removed ───────────────────────────────────────
    let new_fonts: Vec<String> = system_names
        .difference(&indexed_names)
        .cloned()
        .collect();
    let removed_fonts: std::collections::HashSet<String> = indexed_names
        .difference(&system_names)
        .cloned()
        .collect();

    if new_fonts.is_empty() && removed_fonts.is_empty() {
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("  Index is current ({} indexed, {} skipped, {} chars)",
            indexed_count, skipped_count, index.n_chars());
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        return Ok(());
    }

    // ── Remove stale fonts ─────────────────────────────────────────
    if !removed_fonts.is_empty() {
        info!("Removing {} stale fonts from index", removed_fonts.len());
        for name in removed_fonts.iter().take(10) {
            debug!("  - {name}");
        }
        if removed_fonts.len() > 10 {
            debug!("  … and {} more", removed_fonts.len() - 10);
        }
        index.remove_fonts(&removed_fonts);
    }

    // ── Build entries for new fonts only ────────────────────────────
    if !new_fonts.is_empty() {
        info!("Indexing {} new fonts…", new_fonts.len());
        for name in new_fonts.iter().take(10) {
            debug!("  + {name}");
        }
        if new_fonts.len() > 10 {
            debug!("  … and {} more", new_fonts.len() - 10);
        }
        let pairs: Vec<(String, PathBuf, char_index::GlyphOverrides)> = new_fonts
            .iter()
            .filter_map(|name| system_fonts.get(name).map(|(p, g)| (name.clone(), p.clone(), g.clone())))
            .collect();
        let start = std::time::Instant::now();
        let partial = char_index::build_char_index(&pairs);
        info!("  Built entries for {} fonts in {:.1}s", new_fonts.len(), start.elapsed().as_secs_f64());
        index.merge(partial);
    }

    // ── Save updated index ─────────────────────────────────────────
    if let Some(parent) = index_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    char_index::save_index(&index, &index_path).map_err(ScanTextError::Io)?;
    index.compact(); // drop raw entries — flat_vecs is all we need for search
    let file_size = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
    let final_count = char_index::count_fonts(&index);

    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("          CHARACTER INDEX UPDATED");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if !new_fonts.is_empty() {
        info!("  Added:              {} fonts", new_fonts.len());
    }
    if !removed_fonts.is_empty() {
        info!("  Removed:            {} stale fonts", removed_fonts.len());
    }
    info!("  Total fonts:        {}", final_count);
    info!("  Characters:         {}", index.n_chars());
    info!("  Index size:         {:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    info!("  Saved to:           {}", index_path.display());
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

/// Full index build from scratch (used for first build, format changes, --rebuild-index).
fn do_full_build(
    system_fonts: &std::collections::HashMap<String, (PathBuf, char_index::GlyphOverrides)>,
    index_path: &Path,
) -> Result<(), ScanTextError> {
    let pairs: Vec<(String, PathBuf, char_index::GlyphOverrides)> = system_fonts
        .iter()
        .map(|(n, (p, g))| (n.clone(), p.clone(), g.clone()))
        .collect();

    info!("Building character index ({} fonts × {} chars)…",
        pairs.len(), char_index::indexed_chars().len());

    let start = std::time::Instant::now();
    let mut index = char_index::build_char_index(&pairs);
    let elapsed = start.elapsed();

    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent).map_err(ScanTextError::Io)?;
    }
    char_index::save_index(&index, index_path).map_err(ScanTextError::Io)?;

    let file_size = std::fs::metadata(index_path).map(|m| m.len()).unwrap_or(0);
    let n_entries: usize = index.n_entries();
    index.compact();

    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("          CHARACTER INDEX BUILT");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("  Fonts indexed:      {}", pairs.len());
    info!("  Characters:         {}", index.n_chars());
    info!("  Total entries:      {} (font × char)", n_entries);
    info!("  Time:               {:.1}s", elapsed.as_secs_f64());
    info!("  Index size:         {:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    info!("  Saved to:           {}", index_path.display());
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

/// Load or build the character index with caching.
fn load_or_build_index(
    args: &cli::Args,
    font_catalog: &[font_scan::FontEntry],
) -> Result<char_index::CharIndex, ScanTextError> {
    let index_path = args.resolved_index_path();

    if !args.rebuild_index && index_path.exists() {
        info!("Loading cached character index from {}", index_path.display());
        let start = std::time::Instant::now();
        match char_index::load_index(&index_path) {
            Ok(mut index) => {
                let n_entries: usize = index.n_entries();
                index.compact(); // free raw entries — only flat_vecs needed for search
                if std::env::var("UNSCAN_DEBUG_MEM").is_ok() {
                    eprintln!("  MEM after compact: {}", mem_info());
                }
                info!(
                    "  Loaded {} chars × {} entries in {:.1}s",
                    index.n_chars(),
                    n_entries,
                    start.elapsed().as_secs_f64()
                );
                return Ok(index);
            }
            Err(e) => {
                warn!("  Stale/corrupt index cache: {e}");
                info!("  Auto-rebuilding index…");
            }
        }
    }

    info!("Building character index ({} fonts × {} chars)…",
        font_catalog.len(), char_index::indexed_chars().len());
    let start = std::time::Instant::now();
    // Content-hash font dedup disabled — hurt accuracy (AA 471→470, noaa failed 93% threshold).
    // Keeping code commented out for reference.
    // let pairs: Vec<(String, Vec<u8>)> = {
    //     use std::collections::HashSet;
    //     let mut seen = HashSet::new();
    //     let mut deduped = Vec::new();
    //     for e in font_catalog.iter() {
    //         let content_hash = {
    //             use std::hash::{Hash, Hasher};
    //             let mut h = std::collections::hash_map::DefaultHasher::new();
    //             e.data.hash(&mut h);
    //             h.finish()
    //         };
    //         let key = (content_hash, e.variant_tag.clone());
    //         if seen.insert(key) {
    //             deduped.push((e.font_key(), e.data.clone()));
    //         }
    //     }
    //     let n_removed = font_catalog.len() - deduped.len();
    //     if n_removed > 0 {
    //         info!("  Deduped {} identical font file+variant entries ({} → {} unique)",
    //             n_removed, font_catalog.len(), deduped.len());
    //     }
    //     deduped
    // };
    let pairs: Vec<(String, PathBuf, char_index::GlyphOverrides)> = font_catalog
        .iter()
        .map(|e| (e.font_key(), e.path.clone(), e.glyph_overrides.clone()))
        .collect();
    let mut index = char_index::build_char_index(&pairs);
    let elapsed = start.elapsed();
    info!("  Built index in {:.1}s", elapsed.as_secs_f64());

    // Cache to disk
    if let Some(parent) = index_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match char_index::save_index(&index, &index_path) {
        Ok(()) => {
            let sz = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
            info!("  Cached index to {} ({:.2} MB)", index_path.display(), sz as f64 / (1024.0 * 1024.0));
        }
        Err(e) => {
            warn!("  Failed to cache index: {e}");
        }
    }
    index.compact();

    Ok(index)
}

fn run(args: &cli::Args) -> Result<(), ScanTextError> {
    dump_limits();
    let input = args.input.as_ref().expect("input validated");
    let output = args.output.as_ref().expect("output validated");

    // ── Audit directory (created when --audit is set) ──────────────
    if let Some(ref dir) = args.audit {
        let _ = std::fs::create_dir_all(dir);
    }
    let audit_image_dir: Option<audit::AuditImageDir> = {
        if output.to_str() == Some("/dev/null") && args.audit.is_none() {
            None
        } else {
            let audit_path = args.audit_log_path();
            audit::AuditImageDir::from_audit_path(&audit_path).ok()
        }
    };

    let input_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    info!(
        "Input: {} ({:.2} MB)",
        input.display(),
        input_size as f64 / (1024.0 * 1024.0)
    );

    // ── 1. Discover fonts ────────────────────────────────────────────
    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    info!("Scanning for fonts…");
    let font_catalog = font_scan::scan_fonts(&font_dirs);
    info!("Found {} candidate fonts", font_catalog.len());
    if font_catalog.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    // ── 1b. Load or build character index ────────────────────────────
    let char_index = load_or_build_index(args, &font_catalog)?;

    // Font bytes already cleared by scan_fonts() — catalog holds metadata + paths only.
    // All font access goes through the shared cache below.

    // ── 1b''. Shared font cache for all post-index font access ──────
    let font_cache = font_cache::FontCache::new(font_cache::DEFAULT_CAPACITY);

    // ── 2. Load input pages (with raster cache) ──────────────────────
    let cache_dir = page_cache::cache_key(input, args.dpi)
        .and_then(|key| page_cache::cache_dir(&key));

    info!("Loading input pages…");
    let raster_start = std::time::Instant::now();

    // Try loading all pages from cache first.
    let (pages, raster_cached) = if let Some(ref cdir) = cache_dir {
        // Probe page-0 to see if cache is populated; then load all sequentially.
        if page_cache::load_cached_image(cdir, 0).is_some() {
            let mut cached_pages = Vec::new();
            let mut idx = 0;
            while let Some(img) = page_cache::load_cached_image(cdir, idx) {
                cached_pages.push(img);
                idx += 1;
            }
            if !cached_pages.is_empty() {
                (cached_pages, true)
            } else {
                (load_pages(input, args.dpi)?, false)
            }
        } else {
            (load_pages(input, args.dpi)?, false)
        }
    } else {
        (load_pages(input, args.dpi)?, false)
    };

    // If we rasterized fresh, save to cache for next time.
    if !raster_cached {
        if let Some(ref cdir) = cache_dir {
            for (i, img) in pages.iter().enumerate() {
                page_cache::save_cached_image(cdir, i, img);
            }
        }
    }

    let raster_elapsed = raster_start.elapsed();
    info!(
        "Loaded {} page(s) ({:.1}s{})",
        pages.len(),
        raster_elapsed.as_secs_f32(),
        if raster_cached { ", cached" } else { "" },
    );
    if std::env::var("UNSCAN_DEBUG_MEM").is_ok() {
        eprintln!("  MEM after page load: {}", mem_info());
    }

    // ── 2b. Extract source image data for pass-through ───────────────
    let source_images = if input.extension().and_then(|e| e.to_str()) == Some("pdf") {
        extract_source_images(input)
    } else {
        Vec::new()
    };

    // ── 3. Process each page ─────────────────────────────────────────
    let mut all_pages: Vec<pdf_out::PageContent> = Vec::new();
    let mut audit_text: Vec<AuditEntry> = Vec::new();
    let mut audit_geo: Vec<GeometryEntry> = Vec::new();
    let mut page_summaries: Vec<PageSummary> = Vec::new();

    let mut stat_lines_vectorized = 0u32;
    let mut stat_lines_raster = 0u32;
    let mut stat_geo_elements = 0u32;
    let mut stat_raster_frags = 0u32;

    for (page_idx, page_img) in pages.iter().enumerate() {
        info!(
            "━━━ Page {} ({} × {} px) ━━━",
            page_idx + 1,
            page_img.width(),
            page_img.height()
        );

        // 3a-pre. Deskew ──────────────────────────────────────────────
        // Detect and correct skew on the grayscale image used for OCR
        // and font matching. The original colour page_img is kept for
        // background colour detection, geometry, and raster fragments.
        let orig_gray = page_img.to_luma8();
        let skew_angle = deskew::detect_skew(&orig_gray);
        let (deskewed_gray, did_deskew) = if skew_angle.abs() > 5.0 {
            info!("  Deskew: {:.2}° detected (too large, skipped)", skew_angle);
            (orig_gray.clone(), false)
        } else if skew_angle.abs() > 0.5 {
            info!("  Deskew: {:.2}° rotation corrected", skew_angle);
            (deskew::rotate_gray(&orig_gray, skew_angle), true)
        } else {
            (orig_gray, false)
        };

        // Build a DynamicImage from the deskewed gray for OCR input
        let ocr_img: std::borrow::Cow<'_, image::DynamicImage> = if did_deskew {
            std::borrow::Cow::Owned(image::DynamicImage::ImageLuma8(deskewed_gray.clone()))
        } else {
            std::borrow::Cow::Borrowed(page_img)
        };

        // 3a. OCR (with cache) ─────────────────────────────────────────
        let ocr_start = std::time::Instant::now();
        let (word_regions, _page_char_boxes, ocr_cached) =
            if let Some((wr, cb)) = cache_dir.as_ref().and_then(|d| page_cache::load_cached_ocr(d, page_idx)) {
                (wr, cb, true)
            } else {
                let (wr, cb) = ocr::extract_text_regions(&ocr_img, args.dpi)?;
                if let Some(ref cdir) = cache_dir {
                    page_cache::save_cached_ocr(cdir, page_idx, &wr, &cb);
                }
                (wr, cb, false)
            };
        let mut lines = ocr::assemble_lines(&word_regions);
        // Snapshot raw Tesseract word bboxes before post-processing
        let raw_word_bboxes: Vec<Vec<audit::WordBBox>> = lines.iter().map(|line| {
            line.words.iter().map(|w| audit::WordBBox {
                text: w.text.clone(),
                x: w.x,
                y: w.y,
                width: w.width,
                height: w.height,
                confidence: w.confidence,
            }).collect()
        }).collect();
        ocr::merge_overlapping_lines(&mut lines);
        ocr::clip_word_overlaps(&mut lines);
        ocr::drop_outlier_words(&mut lines);
        let ocr_elapsed = ocr_start.elapsed();
        info!(
            "  OCR: {} words → {} lines ({:.1}s{})",
            word_regions.len(),
            lines.len(),
            ocr_elapsed.as_secs_f32(),
            if ocr_cached { ", cached" } else { "" },
        );

        // 3b. Background colour ───────────────────────────────────────
        let bg_color = color::detect_background_color(page_img);
        info!("  Background: #{:02x}{:02x}{:02x}", bg_color.0, bg_color.1, bg_color.2);

        // 3c. Font match + decision matrix ────────────────────────────
        // Use the deskewed grayscale for character segmentation and matching
        let gray_page = deskewed_gray;

        // Expand OCR bboxes to actual ink extent — Tesseract often clips
        // descenders, under-reporting line height by up to 10 px.
        // Use a threshold relative to the background: anything darker than
        // (bg - 56) counts as ink (works for both light and dark backgrounds).
        let ink_thresh = bg_color.0.saturating_sub(56);
        ocr::expand_bbox_to_ink(&mut lines, &gray_page, ink_thresh);
        ocr::expand_words_to_ink(&mut lines, &gray_page, ink_thresh);
        ocr::split_wide_whitespace_words(&mut lines, &gray_page, ink_thresh, Some(&char_index), Some(&font_cache));
        let mut placed_texts: Vec<pdf_out::PlacedText> = Vec::new();
        let mut pg_vec = 0u32;
        let mut pg_raster = 0u32;

        // ── Pass 1: Match all lines ──────────────────────────────────
        struct LineMatch {
            font_result: Option<font_match::FontMatchResult>,
            text_color: (u8, u8, u8),
            ci_top_for_audit: Vec<(String, f32)>,
            ci_char_detail: Vec<char_index::CharCiDetail>,
            ci_top_for_audit_lig: Vec<(String, f32)>,
            ci_char_detail_lig: Vec<char_index::CharCiDetail>,
            seg_winner: Option<String>,
            diag_seg_dir: Option<std::path::PathBuf>,
            /// Per-char distances to the chosen font, keyed by crop_index.
            chosen_char_dists: std::collections::HashMap<usize, f32>,
            /// Per-char distances to all fontmap fonts, keyed by crop_index.
            fontmap_char_dists: std::collections::HashMap<usize, Vec<(String, f32)>>,
        }

        // Pre-load fontmap font_keys once for per-char ground truth distances.
        let fontmap_keys: Vec<String> = if args.audit.is_some() && args.include_fontmap.is_some() {
            if let Some(ref fontmap_path) = args.include_fontmap {
                if let Ok(data) = std::fs::read_to_string(fontmap_path) {
                    if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(&data) {
                        let mut keys = Vec::new();
                        for font_path_str in map.values() {
                            let fp = std::path::Path::new(font_path_str);
                            for fe in &font_catalog {
                                if fe.path == fp {
                                    let key = fe.font_key();
                                    if !keys.contains(&key) {
                                        keys.push(key);
                                    }
                                }
                            }
                        }
                        keys
                    } else { Vec::new() }
                } else { Vec::new() }
            } else { Vec::new() }
        } else { Vec::new() };

        let fontmatch_start = std::time::Instant::now();
        let line_matches: Vec<LineMatch> = lines.par_iter().enumerate().map(|(li, line)| {
            let preview_end = {
                let mut end = line.text.len().min(30);
                while end > 0 && !line.text.is_char_boundary(end) { end -= 1; }
                end
            };
            let debug_mem = std::env::var("UNSCAN_DEBUG_MEM").is_ok();
            if debug_mem {
                eprintln!("  MEM line {} start: {} (\"{}\")", li, mem_info(), &line.text[..preview_end]);
            }
            // Dump total mapped size from /proc/self/maps
            if debug_mem && (li == 2 || li == 45) {
                if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                    let mut total: u64 = 0;
                    for l in maps.lines() {
                        if let Some(range) = l.split_whitespace().next() {
                            if let Some((start_s, end_s)) = range.split_once('-') {
                                if let (Ok(s), Ok(e)) = (u64::from_str_radix(start_s, 16), u64::from_str_radix(end_s, 16)) {
                                    total += e - s;
                                }
                            }
                        }
                    }
                    eprintln!("  MEM line {} maps total: {}MB, {} mappings", li, total / (1024*1024), maps.lines().count());
                }
            }
            let line_start = std::time::Instant::now();
            let text_color = color::detect_text_color(
                page_img,
                &TextRegion {
                    text: line.text.clone(),
                    x: line.x, y: line.y,
                    width: line.width, height: line.height,
                    font_size_pt: line.font_size_pt,
                    confidence: line.confidence,
                    level: 5, block_num: 0, par_num: 0, line_num: 0, word_num: 0,
                },
            );
            let mut ci_top_for_audit: Vec<(String, f32)>;
            let ci_char_detail: Vec<char_index::CharCiDetail>;
            let mut ci_top_for_audit_lig: Vec<(String, f32)>;
            let ci_char_detail_lig: Vec<char_index::CharCiDetail>;
            let seg_winner: Option<String>;
            let diag_seg_dir: Option<std::path::PathBuf> = args.diag_seg_dir().map(|d| {
                let line_slug: String = line.text.chars().take(30)
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let p = d.join(format!("p{}_L{:03}_{}", page_idx + 1, li, line_slug));
                let _ = std::fs::create_dir_all(&p);
                p
            });
            // Extract char crops (before font matching block so they're available after)
            let word_placements: Vec<crate::verify::WordPlacement> = line.words.iter()
                .map(|w| crate::verify::WordPlacement {
                    text: w.text.clone(),
                    x_off: w.x,
                    y_off: w.y,
                    width: w.width,
                    height: w.height,
                    confidence: w.confidence,
                })
                .collect();
            let line_height = line.words.iter().map(|w| w.height).max().unwrap_or(0);
            let line_crops = char_index::extract_line_chars(
                &gray_page, &word_placements, line_height,
                diag_seg_dir.as_deref(),
            );
            let char_crops = &line_crops.plain;
            if debug_mem {
                eprintln!("  MEM line {} after segmentation: {} ({} plain crops, {} lig crops)",
                    li, mem_info(), char_crops.len(),
                    line_crops.ligature.as_ref().map_or(0, |v| v.len()));
            }

            let font_result = {

                // Dump per-line crops into the audit dir.
                // NOTE: We dump plain crops here before scoring; after the winner
                // is decided we overwrite with winning crops if different (see below).
                if let Some(ref ddir) = diag_seg_dir {
                    if !char_crops.is_empty() {
                        let crop_dir = ddir.join("crops");
                        let _ = std::fs::create_dir_all(&crop_dir);
                        for (i, (ch, img)) in char_crops.iter().enumerate() {
                            let path = crop_dir.join(format!("crop_{:02}_{}.png", i,
                                if ch.is_alphanumeric() { format!("{}", ch) }
                                else { format!("U{:04X}", *ch as u32) }));
                            let _ = img.save(&path);
                        }
                    }
                }

                // ── Score plain path ─────────────────────────────────
                let ci_result_plain = char_index::search_candidates(&char_index, char_crops, args.thoroughness, args.audit.is_some());

                // ── Score ligature path (if present) ─────────────────
                let ci_result_lig = line_crops.ligature.as_ref().map(|lig_crops| {
                    char_index::search_candidates(&char_index, lig_crops, args.thoroughness, args.audit.is_some())
                });

                // ── Pick the winner: higher top score wins ───────────
                let plain_top = ci_result_plain.scores.first().map(|(_, s)| *s).unwrap_or(f32::MIN);
                let lig_top = ci_result_lig.as_ref()
                    .and_then(|r| r.scores.first().map(|(_, s)| *s))
                    .unwrap_or(f32::MIN);
                let use_lig = ci_result_lig.is_some() && lig_top > plain_top;

                let (ci_result, _winning_crops) = if use_lig {
                    (ci_result_lig.as_ref().unwrap(), line_crops.ligature.as_ref().unwrap().as_slice())
                } else {
                    (&ci_result_plain, char_crops.as_slice())
                };

                // Store both paths for audit
                ci_top_for_audit = ci_result.scores.iter()
                    .map(|(name, score)| (name.clone(), *score)).collect();
                ci_char_detail = ci_result.char_detail.clone();

                // Store the alternate path for audit
                let (ci_top_lig_audit, ci_char_lig_audit) = if let Some(ref lig_result) = ci_result_lig {
                    (lig_result.scores.iter().map(|(n, s)| (n.clone(), *s)).collect(),
                     lig_result.char_detail.clone())
                } else {
                    (Vec::new(), Vec::new())
                };
                let (ci_top_plain_audit, ci_char_plain_audit) = (
                    ci_result_plain.scores.iter().map(|(n, s)| (n.clone(), *s)).collect::<Vec<_>>(),
                    ci_result_plain.char_detail.clone(),
                );

                // Store both in the LineMatch for audit output
                ci_top_for_audit_lig = if use_lig { ci_top_plain_audit } else { ci_top_lig_audit };
                ci_char_detail_lig = if use_lig { ci_char_plain_audit } else { ci_char_lig_audit };
                seg_winner = if ci_result_lig.is_some() {
                    Some(if use_lig { "ligature".to_string() } else { "plain".to_string() })
                } else {
                    None
                };

                if debug_mem {
                    eprintln!("  MEM line {} after CI search: {}{}", li, mem_info(),
                        if let Some(ref w) = seg_winner { format!(" [seg: {}]", w) } else { String::new() });
                }

                // Overwrite diag-seg crops/ and refs/ with winning path's data
                // (initial dump above used plain; if ligature won, replace).
                if use_lig {
                    if let Some(ref ddir) = diag_seg_dir {
                        let lig_crops = line_crops.ligature.as_ref().unwrap();
                        if !lig_crops.is_empty() {
                            let crop_dir = ddir.join("crops");
                            // Remove old plain crops
                            if crop_dir.is_dir() {
                                let _ = std::fs::remove_dir_all(&crop_dir);
                            }
                            let _ = std::fs::create_dir_all(&crop_dir);
                            for (i, (ch, img)) in lig_crops.iter().enumerate() {
                                let path = crop_dir.join(format!("crop_{:02}_{}.png", i,
                                    if ch.is_alphanumeric() { format!("{}", ch) }
                                    else { format!("U{:04X}", *ch as u32) }));
                                let _ = img.save(&path);
                            }
                        }
                    }
                }

                // --include-font: inject into CI audit list so it shows in audit
                if let Some(ref include) = args.include_font {
                    let include_lc = include.to_lowercase();
                    for fe in &font_catalog {
                        let key = fe.font_key();
                        if key.to_lowercase().contains(&include_lc) && !ci_top_for_audit.iter().any(|(n, _)| n == &key) {
                            ci_top_for_audit.push((key, -999.0)); // sentinel score = included
                        }
                    }
                }

                // --include-fontmap: inject all fonts from a fontmap JSON into CI audit list
                if let Some(ref fontmap_path) = args.include_fontmap {
                    if let Ok(data) = std::fs::read_to_string(fontmap_path) {
                        if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(&data) {
                            for font_path_str in map.values() {
                                let fp = std::path::Path::new(font_path_str);
                                for fe in &font_catalog {
                                    if fe.path == fp {
                                        let key = fe.font_key();
                                        if !ci_top_for_audit.iter().any(|(n, _)| n == &key) {
                                            ci_top_for_audit.push((key, -999.0));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Font selection: use CI #1 directly ───────────────
                if let Some((top_key, top_score)) = ci_result.scores.first() {
                    font_catalog.iter().find(|fe| fe.font_key() == *top_key)
                        .map(|fe| font_match::FontMatchResult {
                            font_name: fe.family_name.clone(),
                            font_path: fe.path.clone(),
                            font_key: fe.font_key(),
                            variant_tag: fe.variant_tag.clone(),
                            glyph_overrides: fe.glyph_overrides.clone(),
                            score: *top_score,
                            best_dy: 0,
                        })
                } else {
                    None
                }
            };
            let line_elapsed = line_start.elapsed();
            if line_elapsed.as_millis() > 500 {
                eprintln!("  LINE {}: {:.1}s '{:.30}…'", li, line_elapsed.as_secs_f32(),
                    line.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "));
            }
            // Compute per-char distances for the chosen font.
            // Use corrected characters when the OCR correction gate fired.
            // Use winning path's crops (plain or ligature).
            let effective_crops: &[(char, image::GrayImage)] = if seg_winner.as_deref() == Some("ligature") {
                line_crops.ligature.as_ref().map(|v| v.as_slice()).unwrap_or(char_crops)
            } else {
                char_crops
            };
            // Build a correction map: crop_index → corrected char, without cloning images
            let char_corrections: std::collections::HashMap<usize, char> = ci_char_detail.iter()
                .filter_map(|d| d.ocr_corrected_from.as_ref().map(|_| (d.crop_index, d.ch)))
                .collect();
            let corrected_char_crops: Vec<(char, &image::GrayImage)> = effective_crops.iter()
                .enumerate()
                .map(|(i, (ch, img))| {
                    let effective_ch = char_corrections.get(&i).copied().unwrap_or(*ch);
                    (effective_ch, img)
                })
                .collect();
            let chosen_char_dists: std::collections::HashMap<usize, f32> = if let Some(ref fr) = font_result {
                if !fr.font_key.is_empty() {
                    char_index::per_char_distances(&char_index, &fr.font_key, &corrected_char_crops)
                        .into_iter()
                        .map(|(_, crop_idx, d2)| (crop_idx, d2))
                        .collect()
                } else {
                    std::collections::HashMap::new()
                }
            } else {
                std::collections::HashMap::new()
            };

            // Per-char distances to all fontmap fonts (ground truth coverage for audit)
            let fontmap_char_dists: std::collections::HashMap<usize, Vec<(String, f32)>> = if !fontmap_keys.is_empty() {
                let mut result: std::collections::HashMap<usize, Vec<(String, f32)>> = std::collections::HashMap::new();
                for fk in &fontmap_keys {
                    for (_, crop_idx, d2) in char_index::per_char_distances(&char_index, fk, &corrected_char_crops) {
                        result.entry(crop_idx).or_default().push((fk.clone(), d2));
                    }
                }
                result
            } else {
                std::collections::HashMap::new()
            };

            // Dump CI reference images into shared font_refs/<label>/U+XXXX.png
            // One copy per font/variant, shared across all lines.
            if let (Some(ref audit_root), Some(ref fr)) = (&args.audit, &font_result) {
                let fe_opt = font_catalog.iter().find(|e| e.font_key() == fr.font_key);
                let label = fe_opt.map(|fe| {
                    let mut s = fe.family_name.replace(' ', "");
                    if fe.is_bold { s.push_str("-Bold"); }
                    if fe.is_italic { s.push_str("-Italic"); }
                    if !fe.variant_tag.is_empty() {
                        s.push('_');
                        s.push_str(&fe.variant_tag);
                    }
                    s
                }).unwrap_or_else(|| {
                    fr.font_path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string()
                });
                let font_ref_dir = audit_root.join("font_refs").join(&label);
                let _ = std::fs::create_dir_all(&font_ref_dir);

                let override_map: std::collections::HashMap<char, u16> = fe_opt
                    .and_then(|fe| fe.glyph_overrides.as_ref())
                    .map(|v| v.iter().cloned().collect())
                    .unwrap_or_default();
                let font_data = font_cache.load(&fr.font_path).ok();
                if let Some(ref fdata) = font_data {
                    if let Ok(font) = ab_glyph::FontRef::try_from_slice(fdata) {
                        for (ch, _crop) in corrected_char_crops.iter() {
                            let fname = format!("U+{:04X}.png", *ch as u32);
                            let path = font_ref_dir.join(&fname);
                            if path.exists() { continue; } // already rendered
                            let ref_img = if let Some(&gid) = override_map.get(ch) {
                                char_index::render_glyph_normalised(&font, ab_glyph::GlyphId(gid))
                            } else {
                                char_index::render_char_normalised(&font, *ch)
                            };
                            if let Some(img) = ref_img {
                                let _ = img.save(&path);
                            }
                        }
                    }
                }
            }
            LineMatch { font_result, text_color, ci_top_for_audit, ci_char_detail, ci_top_for_audit_lig, ci_char_detail_lig, seg_winner, diag_seg_dir, chosen_char_dists, fontmap_char_dists }
        }).collect();
        let fontmatch_elapsed = fontmatch_start.elapsed();
        eprintln!("  Font matching: {:.1}s ({} lines)", fontmatch_elapsed.as_secs_f32(), lines.len());

        // ── Pass 1.5: Paragraph-level font grouping ─────────────────
        // Find the dominant body font: most common font among matched lines
        // at the most common font size (±1pt tolerance).
        {
            use std::collections::HashMap;
            // Collect (font_name, font_size_bucket) frequencies
            let mut size_freq: HashMap<i32, u32> = std::collections::HashMap::new();
            for (i, lm) in line_matches.iter().enumerate() {
                if let Some(ref fm) = lm.font_result {
                    if fm.score >= args.min_font_confidence {
                        let bucket = lines[i].font_size_pt.round() as i32;
                        *size_freq.entry(bucket).or_default() += 1;
                    }
                }
            }
            // Find most common size bucket
            let body_size = size_freq.iter()
                .max_by_key(|(_, &v)| v)
                .map(|(&k, _)| k);

            debug!("  paragraph grouping: size_freq={:?} body_size={:?}", size_freq, body_size);

            if let Some(body_size) = body_size {
                // Count fonts at body size (±1pt)
                let mut font_freq: HashMap<String, (u32, PathBuf)> = std::collections::HashMap::new();
                for (i, lm) in line_matches.iter().enumerate() {
                    let sz = lines[i].font_size_pt.round() as i32;
                    if (sz - body_size).abs() <= 1 {
                        if let Some(ref fm) = lm.font_result {
                            if fm.score >= args.min_font_confidence {
                                let entry = font_freq.entry(fm.font_name.clone())
                                    .or_insert_with(|| (0, fm.font_path.clone()));
                                entry.0 += 1;
                            }
                        }
                    }
                }
                // Find majority font
                if let Some((majority_name, (majority_count, _majority_path))) = font_freq.iter()
                    .max_by_key(|(_, (count, _))| *count)
                {
                    let total_body: u32 = font_freq.values().map(|(c, _)| c).sum();
                    debug!("  paragraph grouping: font_freq={:?} majority='{}' {}/{}", 
                        font_freq.iter().map(|(k,(c,_))| (k.as_str(), *c)).collect::<Vec<_>>(),
                        majority_name, majority_count, total_body);
                }
            }
        }

        // ── Pass 2: Decision matrix + output ──────────────────────────
        for (li, line) in lines.iter().enumerate() {
            let lm = &line_matches[li];
            let text_color = lm.text_color;
            let font_result = &lm.font_result;

            // ── Decision matrix ──────────────────────────────────────
            let ocr_ok = line.confidence >= args.min_ocr_confidence as f32
                && !line.text.trim().is_empty();
            let font_ok = font_result
                .as_ref()
                .map(|f| f.score >= args.min_font_confidence)
                .unwrap_or(false);

            let (keep_raster, reason) = if !ocr_ok {
                (true, format!("OCR confidence too low ({:.0}%)", line.confidence))
            } else if !font_ok {
                let best = font_result
                    .as_ref()
                    .map(|f| format!("{} at {:.3}", f.font_name, f.score))
                    .unwrap_or_else(|| "none".into());
                (
                    true,
                    format!(
                        "No confident font match (best: {best}). Kept as raster."
                    ),
                )
            } else {
                (false, "Vectorised".into())
            };

            // ── SSIM verification (if vectorising) ───────────────────
            let (ssim_score, ssim_pass): (Option<f32>, Option<bool>) = if !keep_raster {
                if let Some(ref fm) = font_result {
                    let font_data = font_cache.load(&fm.font_path).ok();
                    if let Some(ref fd) = font_data {
                        let (score, _dy) = verify::verify_text_region(
                            &gray_page,
                            fd.as_slice(),
                            &line.text,
                            line.x, line.y,
                            line.width, line.height,
                            &line.words,
                            fm.glyph_overrides.as_deref(),
                            &fm.variant_tag,
                            lm.diag_seg_dir.as_deref(),
                        );
                        let pass = score >= MIN_VERIFY_SSIM;
                        (Some(score), Some(pass))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            // ── Logging ──────────────────────────────────────────────
            if keep_raster {
                pg_raster += 1;
                warn!(
                    "  ⚠ '{}' — {}",
                    truncate_str(&line.text, 50),
                    reason
                );
            } else {
                pg_vec += 1;
                let fname = font_result.as_ref().map(|f| f.font_name.as_str()).unwrap_or("?");
                let fscore = font_result.as_ref().map(|f| f.score).unwrap_or(0.0);
                let ssim_part = ssim_score
                    .map(|s| format!(" ssim={s:.3}"))
                    .unwrap_or_default();
                info!(
                    "  ✓ '{}' → {} (score={:.3}{}) {:.1}pt #{:02x}{:02x}{:02x}",
                    truncate_str(&line.text, 50),
                    fname,
                    fscore,
                    ssim_part,
                    line.font_size_pt,
                    text_color.0,
                    text_color.1,
                    text_color.2,
                );
            }

            // ── Diag-seg line-level summary ──────────────────────────
            if let Some(ref ddir) = lm.diag_seg_dir {
                let fname = font_result.as_ref().map(|f| f.font_name.as_str()).unwrap_or("?");
                let fscore = font_result.as_ref().map(|f| f.score).unwrap_or(0.0);
                let line_summary = serde_json::json!({
                    "page": page_idx + 1,
                    "line_index": li,
                    "text": &line.text,
                    "font_matched": fname,
                    "font_score": fscore,
                    "ssim_score": ssim_score,
                    "ci_top_5": lm.ci_top_for_audit.iter().take(5)
                        .map(|(k, s)| serde_json::json!({"font": k, "score": s}))
                        .collect::<Vec<_>>(),
                    "decision": if keep_raster { "raster" } else { "vectorized" },
                });
                let _ = std::fs::write(
                    ddir.join("line_summary.json"),
                    serde_json::to_string_pretty(&line_summary).unwrap_or_default(),
                );
            }

            // ── Audit entry ──────────────────────────────────────────
            audit_text.push(AuditEntry {
                page: page_idx + 1,
                line_index: li,
                text: line.text.clone(),
                ocr_confidence: line.confidence,
                font_matched: font_result.as_ref().map(|f| f.font_name.clone()),
                font_confidence: font_result.as_ref().map(|f| f.score),
                ssim_score,
                ssim_pass,
                decision: if keep_raster { Decision::KeptRaster } else { Decision::Vectorized },
                reason: reason.clone(),
                bbox: BBox {
                    x: line.x,
                    y: line.y,
                    width: line.width,
                    height: line.height,
                },
                ci_candidates: lm.ci_top_for_audit.iter()
                    .map(|(k, s)| audit::CiCandidate { font_key: k.clone(), score: *s })
                    .collect(),
                ci_char_votes: lm.ci_char_detail.iter()
                    .map(|d| {
                        let chosen_d2 = lm.chosen_char_dists.get(&d.crop_index).copied();
                        let fm_dists = lm.fontmap_char_dists.get(&d.crop_index)
                            .cloned().unwrap_or_default();
                        audit::CharCiVote {
                            ch: d.ch,
                            crop_index: d.crop_index,
                            min_dist_sq: d.min_dist_sq,
                            passed_gate: d.passed_gate,
                            nearest: d.nearest.clone(),
                            crop_path: None,
                            chosen_dist_sq: chosen_d2,
                            ocr_corrected_from: d.ocr_corrected_from,
                            best_alt_char: d.best_alt_char,
                            best_alt_dist: d.best_alt_dist,
                            fontmap_dists: fm_dists,
                        }
                    })
                    .collect(),
                ci_candidates_lig: lm.ci_top_for_audit_lig.iter()
                    .map(|(k, s)| audit::CiCandidate { font_key: k.clone(), score: *s })
                    .collect(),
                ci_char_votes_lig: lm.ci_char_detail_lig.iter()
                    .map(|d| {
                        audit::CharCiVote {
                            ch: d.ch,
                            crop_index: d.crop_index,
                            min_dist_sq: d.min_dist_sq,
                            passed_gate: d.passed_gate,
                            nearest: d.nearest.clone(),
                            crop_path: None,
                            chosen_dist_sq: None,
                            ocr_corrected_from: d.ocr_corrected_from,
                            best_alt_char: d.best_alt_char,
                            best_alt_dist: d.best_alt_dist,
                            fontmap_dists: Vec::new(),
                        }
                    })
                    .collect(),
                seg_winner: lm.seg_winner.clone(),
                word_bboxes: line.words.iter().map(|w| audit::WordBBox {
                    text: w.text.clone(),
                    x: w.x,
                    y: w.y,
                    width: w.width,
                    height: w.height,
                    confidence: w.confidence,
                }).collect(),
                word_bboxes_raw: if li < raw_word_bboxes.len() {
                    raw_word_bboxes[li].clone()
                } else {
                    Vec::new()
                },
            });

            placed_texts.push(pdf_out::PlacedText {
                text: line.text.clone(),
                x: line.x as f32,
                y: line.y as f32,
                width: line.width as f32,
                height: line.height as f32,
                font_size_pt: {
                    // Height-based sizing: map OCR bbox height to em-square
                    // using the font's ink height ratio (ascent - descent).
                    let dpi_f = args.dpi as f32;
                    let fallback_pt = line.height as f32 * 72.0 / dpi_f;
                    if let Some(ref fm) = font_result {
                        let font_bytes = font_cache.load(&fm.font_path).ok();
                        if let Some(ref fb) = font_bytes {
                        if let Ok(f) = ab_glyph::FontRef::try_from_slice(fb.as_slice()) {
                            use ab_glyph::{Font, PxScale, ScaleFont};
                            let ref_h = 100.0f32;
                            let sf_ref = f.as_scaled(PxScale::from(ref_h));
                            let ref_ink = sf_ref.ascent() - sf_ref.descent();
                            // Use the line bbox height — spans full ascender-to-descender
                            let line_h = line.height as f32;
                            if line_h > 1.0 {
                                let em_px = ref_h * (line_h / ref_ink);
                                em_px * 72.0 / dpi_f
                            } else {
                                fallback_pt
                            }
                        } else {
                            fallback_pt
                        }
                        } else {
                            fallback_pt
                        }
                    } else {
                        fallback_pt
                    }
                },
                font_match: font_result.clone(),
                keep_raster,
                color: text_color,
                confidence: line.confidence,
                words: line.words.iter().map(|w| pdf_out::WordBox {
                    text: w.text.clone(),
                    x: w.x as f32,
                    y: w.y as f32,
                    width: w.width as f32,
                    height: w.height as f32,
                    smoothed_em_px: None,
                }).collect(),
            });
        }

        stat_lines_vectorized += pg_vec;
        stat_lines_raster += pg_raster;

        // 3d. Geometry detection ──────────────────────────────────────
        let (det_lines, det_fills) = if args.no_geometry {
            (Vec::new(), Vec::new())
        } else {
            // All text bounding boxes (vectorised AND raster) — skip them
            // during geometry detection.
            let text_bboxes: Vec<(u32, u32, u32, u32)> = placed_texts
                .iter()
                .map(|pt| (pt.x as u32, pt.y as u32, pt.width as u32, pt.height as u32))
                .collect();

            let min_line_len = (args.dpi / 4).max(30); // ~¼ inch
            let geo = geometry::detect_geometry(page_img, &text_bboxes, min_line_len);

            for l in &geo.lines {
                audit_geo.push(GeometryEntry {
                    page: page_idx + 1,
                    kind: "line",
                    bbox: BBox {
                        x: l.x1.min(l.x2),
                        y: l.y1.min(l.y2),
                        width: l.x1.abs_diff(l.x2).max(l.thickness),
                        height: l.y1.abs_diff(l.y2).max(l.thickness),
                    },
                });
            }
            for f in &geo.fills {
                audit_geo.push(GeometryEntry {
                    page: page_idx + 1,
                    kind: "fill",
                    bbox: BBox { x: f.x, y: f.y, width: f.width, height: f.height },
                });
            }

            let count = geo.lines.len() + geo.fills.len();
            stat_geo_elements += count as u32;
            if count > 0 {
                info!(
                    "  Geometry: {} lines, {} fills → vectorised",
                    geo.lines.len(),
                    geo.fills.len()
                );
            }
            (geo.lines, geo.fills)
        };

        // 3e. Erase vectorised content from raster ────────────────────
        let mut erase_rects: Vec<(u32, u32, u32, u32)> = Vec::new();

        if !args.overlay {
            // Erase only text that was successfully vectorised.
            for pt in &placed_texts {
                if !pt.keep_raster {
                    erase_rects.push((pt.x as u32, pt.y as u32, pt.width as u32, pt.height as u32));
                }
            }

            // Erase vectorised geometry.
            let geo_result = geometry::GeometryResult {
                lines: det_lines.clone(),
                fills: det_fills.clone(),
            };
            erase_rects.extend(geometry::erase_bboxes(&geo_result));
        }

        let cleaned_img = color::erase_regions(page_img, &erase_rects, bg_color, 2);

        // 3f. Build raster fragments — prefer source pass-through ─────
        let anything_vectorized = pg_vec > 0 || !det_lines.is_empty() || !det_fills.is_empty();
        let source_img = source_images.get(page_idx).and_then(|s| s.as_ref());

        let raster_fragments = if !anything_vectorized && source_img.is_some() {
            // Nothing was vectorized/erased — pass through the original
            // compressed image stream verbatim.
            let si = source_img.unwrap().clone();
            let scale = 72.0 / args.dpi as f32;
            let pw = page_img.width();
            let ph = page_img.height();
            info!(
                "  Raster pass-through: {:?} {}×{} ({:.1} KB)",
                si.filter,
                si.width,
                si.height,
                si.stream_bytes.len() as f64 / 1024.0
            );
            vec![pdf_out::RasterFragment {
                raw_rgb: Vec::new(), // not used — passthrough takes precedence
                width_px: si.width,
                height_px: si.height,
                x_pt: 0.0,
                y_pt: 0.0,
                width_pt: pw as f32 * scale,
                height_pt: ph as f32 * scale,
                passthrough: Some(si),
            }]
        } else {
            // Something was vectorized — we modified the image, must re-encode
            extract_raster_fragments(
                &cleaned_img,
                args.dpi,
                page_img.height(),
            )
        };

        stat_raster_frags += raster_fragments.len() as u32;

        if !raster_fragments.is_empty() {
            let is_passthrough = raster_fragments.iter().any(|f| f.passthrough.is_some());
            if !is_passthrough {
                let frag_bytes: usize = raster_fragments.iter().map(|f| f.raw_rgb.len()).sum();
                info!(
                    "  Raster fragments: {} ({:.1} KB raw RGB)",
                    raster_fragments.len(),
                    frag_bytes as f64 / 1024.0
                );
            }
        }

        page_summaries.push(PageSummary {
            page: page_idx + 1,
            width_px: page_img.width(),
            height_px: page_img.height(),
            lines_vectorized: pg_vec,
            lines_kept_raster: pg_raster,
            geometry_elements: (det_lines.len() + det_fills.len()) as u32,
            raster_fragments: raster_fragments.len() as u32,
        });

        // 3f. Comparison output ───────────────────────────────────────
        if args.compare {
            let compare_dir = output.with_extension("").to_string_lossy().to_string()
                + "-compare";
            let compare_path = std::path::PathBuf::from(&compare_dir);
            if let Err(e) = compare::generate_comparison(
                &gray_page,
                &placed_texts,
                page_idx,
                &compare_path,
                &font_cache,
            ) {
                warn!("  Compare output failed: {e}");
            }
        }

        all_pages.push(pdf_out::PageContent {
            width_px: page_img.width(),
            height_px: page_img.height(),
            dpi: args.dpi,
            text_regions: placed_texts,
            raster_fragments,
            lines: det_lines,
            fills: det_fills,
            bg_color,
        });
    }

    // ── 4. Generate output PDF ───────────────────────────────────────
    // Optional smoothing pass: unify per-word font sizes within same-font runs.
    if args.smooth {
        for page in all_pages.iter_mut() {
            smooth::smooth_font_sizes(&mut page.text_regions, page.dpi as f32, &font_cache);
        }
    }

    info!("Generating output PDF: {}", output.display());
    pdf_out::generate_pdf(output, &all_pages, args.overlay, &font_cache)?;

    let output_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    let ratio = if output_size > 0 { input_size as f64 / output_size as f64 } else { 0.0 };

    // ── 5. Write audit log (skip for /dev/null output) ────────────────
    let audit_path = args.audit_log_path();
    let audit = AuditLog {
        input_file: input.display().to_string(),
        output_file: output.display().to_string(),
        input_size_bytes: input_size,
        output_size_bytes: output_size,
        compression_ratio: ratio,
        images_dir: audit_image_dir.as_ref().map(|aid| aid.rel_dir()),
        pages: page_summaries,
        text_entries: audit_text,
        geometry_entries: audit_geo,
    };
    if output.to_str() != Some("/dev/null") || args.audit.is_some() {
        if let Err(e) = audit.write_to_file(&audit_path) {
            warn!("Failed to write audit log: {}", e);
        } else {
            info!("Audit log: {}", audit_path.display());
        }
    }

    // ── 6. Report ────────────────────────────────────────────────────
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("              SCANTEXT RESULTS");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("  Pages:                  {}", all_pages.len());
    info!("  Text lines vectorised:  {}", stat_lines_vectorized);
    info!("  Text lines kept raster: {}", stat_lines_raster);
    info!("  Geometry elements:      {}", stat_geo_elements);
    info!("  Raster fragments:       {}", stat_raster_frags);
    info!("  ──────────────────────────────────────────────");
    info!(
        "  Input size:  {:.2} MB",
        input_size as f64 / (1024.0 * 1024.0)
    );
    info!(
        "  Output size: {:.2} MB",
        output_size as f64 / (1024.0 * 1024.0)
    );
    info!("  Ratio:       {:.1}× smaller", ratio);
    let (cache_hits, cache_misses) = font_cache.stats();
    info!("  Font cache:  {} hits / {} misses ({} cached)",
        cache_hits, cache_misses, font_cache.len());
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("Output: {}", output.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Page loading
// ---------------------------------------------------------------------------

fn load_pages(path: &Path, dpi: u32) -> Result<Vec<DynamicImage>, ScanTextError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "pdf" => load_pdf_pages(path, dpi),
        "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "gif" | "webp" => {
            let img = image::open(path).map_err(|e| ScanTextError::ImageLoad(e.to_string()))?;
            Ok(vec![img])
        }
        _ => Err(ScanTextError::UnsupportedFormat(ext)),
    }
}

fn load_pdf_pages(path: &Path, dpi: u32) -> Result<Vec<DynamicImage>, ScanTextError> {
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
// Raster fragment extraction (lossless)
// ---------------------------------------------------------------------------

fn extract_raster_fragments(
    cleaned_img: &DynamicImage,
    dpi: u32,
    page_height_px: u32,
) -> Vec<pdf_out::RasterFragment> {
    let w = cleaned_img.width();
    let h = cleaned_img.height();
    let cell = 100u32;
    let cols = (w + cell - 1) / cell;
    let rows = (h + cell - 1) / cell;

    let mut interesting = vec![false; (cols * rows) as usize];
    let gray = cleaned_img.to_luma8();
    for row in 0..rows {
        for col in 0..cols {
            let cx = col * cell;
            let cy = row * cell;
            let cw = cell.min(w - cx);
            let ch = cell.min(h - cy);
            if color::region_has_content(&gray, cx, cy, cw, ch) {
                interesting[(row * cols + col) as usize] = true;
            }
        }
    }

    let mut visited = vec![false; (cols * rows) as usize];
    let mut fragments = Vec::new();

    for row in 0..rows {
        for col in 0..cols {
            let idx = (row * cols + col) as usize;
            if !interesting[idx] || visited[idx] {
                continue;
            }
            visited[idx] = true;
            let mut queue = vec![(row, col)];
            let (mut min_r, mut max_r, mut min_c, mut max_c) = (row, row, col, col);

            while let Some((r, c)) = queue.pop() {
                min_r = min_r.min(r);
                max_r = max_r.max(r);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
                for (dr, dc) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                        continue;
                    }
                    let ni = (nr as u32 * cols + nc as u32) as usize;
                    if !interesting[ni] || visited[ni] {
                        continue;
                    }
                    visited[ni] = true;
                    queue.push((nr as u32, nc as u32));
                }
            }

            let fx = min_c * cell;
            let fy = min_r * cell;
            let fw = ((max_c + 1) * cell).min(w) - fx;
            let fh = ((max_r + 1) * cell).min(h) - fy;

            if let Some(frag) =
                pdf_out::extract_raster_fragment(cleaned_img, fx, fy, fw, fh, dpi, page_height_px)
            {
                fragments.push(frag);
            }
        }
    }
    fragments
}

fn truncate_str(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() <= max {
        s
    } else {
        // Walk backwards from `max` to find the nearest char boundary,
        // avoiding panics on multi-byte UTF-8 (e.g. em-dash '—' is 3 bytes).
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

// ---------------------------------------------------------------------------
// Source image extraction — preserve original encoding
// ---------------------------------------------------------------------------

/// Extract the original compressed image stream from each page of the input PDF.
/// Returns one `SourceImageInfo` per page, or `None` if extraction fails for
/// that page (multiple images, unsupported structure, etc.).
fn extract_source_images(path: &Path) -> Vec<Option<pdf_out::SourceImageInfo>> {
    let doc = match lopdf::Document::load(path) {
        Ok(d) => d,
        Err(e) => {
            log::debug!("Could not load source PDF for pass-through: {e}");
            return Vec::new();
        }
    };

    let page_map = doc.get_pages(); // BTreeMap<u32, ObjectId>
    let mut results: Vec<Option<pdf_out::SourceImageInfo>> = Vec::new();

    for page_num in 1..=(page_map.len() as u32) {
        let info = (|| -> Option<pdf_out::SourceImageInfo> {
            let &page_id = page_map.get(&page_num)?;
            let page_obj = doc.get_object(page_id).ok()?;
            let page_dict = page_obj.as_dict().ok()?;

            let resources = page_dict.get(b"Resources").ok()?;
            let res_dict = resolve_dict(&doc, resources)?;

            let xobj = res_dict.get(b"XObject").ok()?;
            let xobj_dict = resolve_dict(&doc, xobj)?;

            // We only handle the simple case: exactly one image XObject per page
            let entries: Vec<_> = xobj_dict.iter().collect();
            if entries.len() != 1 {
                log::debug!(
                    "Page {page_num}: {} XObjects, skipping pass-through",
                    entries.len()
                );
                return None;
            }

            let (_name, obj) = entries[0];
            let stream_id = match obj {
                lopdf::Object::Reference(r) => *r,
                _ => return None,
            };

            let stream_obj = doc.get_object(stream_id).ok()?;
            let stream = match stream_obj {
                lopdf::Object::Stream(ref s) => s,
                _ => return None,
            };

            // Verify it's an Image XObject
            let subtype = stream.dict.get(b"Subtype").ok()?;
            match subtype {
                lopdf::Object::Name(ref n) if n == b"Image" => {}
                _ => return None,
            }

            // Extract filter
            let filter = match stream.dict.get(b"Filter") {
                Ok(lopdf::Object::Name(ref n)) => filter_from_name(n),
                Ok(lopdf::Object::Array(ref arr)) => {
                    // Single-element filter array (common)
                    if arr.len() == 1 {
                        if let lopdf::Object::Name(ref n) = arr[0] {
                            filter_from_name(n)
                        } else {
                            pdf_out::ImageFilter::Other("unknown".into())
                        }
                    } else {
                        // Chained filters — too complex to pass through
                        log::debug!("Page {page_num}: chained filters, skipping pass-through");
                        return None;
                    }
                }
                _ => pdf_out::ImageFilter::None,
            };

            let width = get_integer(&stream.dict, b"Width")? as u32;
            let height = get_integer(&stream.dict, b"Height")? as u32;
            let bpc = get_integer(&stream.dict, b"BitsPerComponent").unwrap_or(8) as u32;

            // Resolve ColorSpace to a name string
            let color_space = match stream.dict.get(b"ColorSpace") {
                Ok(lopdf::Object::Name(ref n)) => {
                    String::from_utf8_lossy(n).to_string()
                }
                Ok(lopdf::Object::Reference(r)) => {
                    match doc.get_object(*r) {
                        Ok(lopdf::Object::Name(ref n)) => {
                            String::from_utf8_lossy(n).to_string()
                        }
                        // ICCBased or other array-style colorspace — fall back
                        // to DeviceRGB/DeviceGray based on BPC heuristic
                        _ => {
                            // Try to infer from stream size
                            let expected_rgb = (width * height * 3) as usize;
                            let expected_gray = (width * height) as usize;
                            if filter == pdf_out::ImageFilter::None
                                || filter == pdf_out::ImageFilter::FlateDecode
                            {
                                // Can't reliably determine from compressed size,
                                // but for FlateDecode the uncompressed size would
                                // tell us.  Use the referenced object's structure
                                // if it's an ICCBased array.
                                if let Ok(lopdf::Object::Array(ref arr)) = doc.get_object(*r) {
                                    if arr.len() >= 1 {
                                        if let lopdf::Object::Name(ref n) = arr[0] {
                                            if n == b"ICCBased" {
                                                // Check the ICC profile stream's N value
                                                if arr.len() >= 2 {
                                                    if let lopdf::Object::Reference(icc_ref) = arr[1] {
                                                        if let Ok(lopdf::Object::Stream(ref icc_stream)) = doc.get_object(icc_ref) {
                                                            let n_val = get_integer(&icc_stream.dict, b"N").unwrap_or(3);
                                                            return Some(pdf_out::SourceImageInfo {
                                                                stream_bytes: stream.content.clone(),
                                                                filter,
                                                                width,
                                                                height,
                                                                color_space: if n_val == 1 {
                                                                    "DeviceGray".into()
                                                                } else {
                                                                    "DeviceRGB".into()
                                                                },
                                                                bits_per_component: bpc,
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Default fallback
                            let _ = (expected_rgb, expected_gray);
                            "DeviceRGB".into()
                        }
                    }
                }
                _ => "DeviceRGB".into(),
            };

            log::debug!(
                "Page {page_num}: source image {}×{} {:?} {} bpc={} ({} bytes)",
                width,
                height,
                filter,
                color_space,
                bpc,
                stream.content.len()
            );

            Some(pdf_out::SourceImageInfo {
                stream_bytes: stream.content.clone(),
                filter,
                width,
                height,
                color_space,
                bits_per_component: bpc,
            })
        })();

        results.push(info);
    }

    results
}

fn filter_from_name(name: &[u8]) -> pdf_out::ImageFilter {
    match name {
        b"DCTDecode" => pdf_out::ImageFilter::DCTDecode,
        b"FlateDecode" => pdf_out::ImageFilter::FlateDecode,
        _ => pdf_out::ImageFilter::Other(String::from_utf8_lossy(name).to_string()),
    }
}

fn resolve_dict<'a>(
    doc: &'a lopdf::Document,
    obj: &'a lopdf::Object,
) -> Option<lopdf::Dictionary> {
    match obj {
        lopdf::Object::Reference(r) => doc
            .get_object(*r)
            .ok()
            .and_then(|o| o.as_dict().ok().cloned()),
        lopdf::Object::Dictionary(d) => Some(d.clone()),
        _ => None,
    }
}

fn get_integer(dict: &lopdf::Dictionary, key: &[u8]) -> Option<i64> {
    match dict.get(key).ok()? {
        lopdf::Object::Integer(i) => Some(*i),
        _ => None,
    }
}

