mod audit;
mod classifier;
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
pub mod ground_truth;
pub mod report;

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
            }
        }
    }
    // Check overcommit and committed memory
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for l in s.lines() {
            if l.starts_with("CommitLimit:") || l.starts_with("Committed_AS:") || l.starts_with("MemAvailable:") {
            }
        }
    }
    if let Ok(s) = std::fs::read_to_string("/proc/sys/vm/overcommit_memory") {
    }
    // cgroup memory limit
    for path in &["/sys/fs/cgroup/memory/memory.limit_in_bytes",
                   "/sys/fs/cgroup/memory.max"] {
        if let Ok(s) = std::fs::read_to_string(path) {
        }
    }
}
use crate::error::ScanTextError;
use crate::ocr::TextRegion;
use image::DynamicImage;
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
    std::process::exit(0);
}

fn main() {
    let args = cli::parse();
    if let Err(msg) = args.validate() {
        eprintln!("Error: {msg}");
        std::process::exit(1);
    }

    // ── render-ref-chars: standalone char rendering, no PDF needed ───
    if let Some(ref json_str) = args.render_ref_chars {
        render_ref_chars_and_exit(json_str);
    }

    // ── weight-explicit: normalize PS names and exit ─────────────────
    if !args.weight_explicit.is_empty() {
        for spec in &args.weight_explicit {
            if let Some((ps, w_str)) = spec.rsplit_once(':') {
                if let Ok(w) = w_str.parse::<u16>() {
                    println!("{}", font_scan::make_weight_explicit(ps, w));
                } else {
                    eprintln!("Bad weight in {:?} — expected PSName:weight", spec);
                    std::process::exit(1);
                }
            } else {
                eprintln!("Bad format {:?} — expected PSName:weight", spec);
                std::process::exit(1);
            }
        }
        std::process::exit(0);
    }

    // ── Build the classifier based on --classifier flag ──────────────
    let clf: Box<dyn classifier::Classifier> = make_classifier(&args);

    if args.index {
        if let Err(e) = run_index(&args, &*clf) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    } else {
        if let Err(e) = run(&args, &*clf) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// Create the classifier selected by CLI args.
fn make_classifier(args: &cli::Args) -> Box<dyn classifier::Classifier> {
    match args.classifier.as_str() {
        "fisher" => Box::new(classifier::FisherClassifier),
        "triplet" => {
            let weights_path = args.triplet_weights.as_ref().unwrap_or_else(|| {
                eprintln!("Error: --triplet-weights is required when using --classifier=triplet");
                std::process::exit(1);
            });
            match classifier::TripletClassifier::load(weights_path) {
                Ok(c) => Box::new(c),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        "global-triplet" => {
            let weights_path = args.triplet_weights.as_ref().unwrap_or_else(|| {
                eprintln!("Error: --triplet-weights is required when using --classifier=global-triplet");
                std::process::exit(1);
            });
            match classifier::GlobalTripletClassifier::load(weights_path) {
                Ok(c) => Box::new(c),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        other => {
            // Check for weight-file-based classifiers
            match other {
                "lda" => {
                    // LDA has built-in weights; external file is optional override
                    if let Some(ref weights_path) = args.triplet_weights {
                        match classifier::LdaClassifier::load(weights_path) {
                            Ok(c) => Box::new(c),
                            Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                        }
                    } else {
                        static LDA_WEIGHTS: &[u8] = include_bytes!("../lda28-weights.bin");
                        match classifier::LdaClassifier::from_bytes(LDA_WEIGHTS) {
                            Ok(c) => Box::new(c),
                            Err(e) => { eprintln!("Error loading built-in LDA weights: {e}"); std::process::exit(1); }
                        }
                    }
                }
                "fusion" => {
                    // Rank-fusion of LDA + Fisher (no external weights needed)
                    static LDA_WEIGHTS: &[u8] = include_bytes!("../lda28-weights.bin");
                    let lda = match classifier::LdaClassifier::from_bytes(LDA_WEIGHTS) {
                        Ok(c) => c,
                        Err(e) => { eprintln!("Error loading built-in LDA weights: {e}"); std::process::exit(1); }
                    };
                    let fisher = classifier::FisherClassifier;
                    Box::new(classifier::FusionClassifier::new(vec![
                        (0.5, Box::new(lda)),
                        (0.5, Box::new(fisher)),
                    ]))
                }
                _ => {
                    let weights_path = args.triplet_weights.as_ref().unwrap_or_else(|| {
                        eprintln!("Error: --triplet-weights is required when using --classifier={other}");
                        std::process::exit(1);
                    });
                    match other {
                        "perchar-fisher" => {
                            match classifier::PerCharFisherClassifier::load(weights_path) {
                                Ok(c) => Box::new(c),
                                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                            }
                        }
                        "mahalanobis" => {
                            match classifier::MahalanobisClassifier::load(weights_path) {
                                Ok(c) => Box::new(c),
                                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                            }
                        }
                        "mlp" => {
                            match classifier::MlpClassifier::load(weights_path) {
                                Ok(c) => Box::new(c),
                                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                            }
                        }
                        _ => {
                            eprintln!("Error: unknown classifier '{other}'. Use 'lda', 'fisher', 'perchar-fisher', 'triplet', 'global-triplet', 'mahalanobis', 'mlp', or 'fusion'.");
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
    }
}

/// Scan available fonts, compare against the cached index, and incrementally
/// update: add new fonts, remove stale fonts, or report "Index is current".
/// With `--rebuild-index`, forces a full rebuild.
fn run_index(args: &cli::Args, classifier: &dyn classifier::Classifier) -> Result<(), ScanTextError> {
    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    let font_catalog = font_scan::scan_fonts(&font_dirs);
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
        return do_full_build(&system_fonts, &index_path, classifier);
    }

    // ── Try loading existing index ─────────────────────────────────
    if !index_path.exists() {
        return do_full_build(&system_fonts, &index_path, classifier);
    }

    // Fast header check first (12 bytes, no full deserialize).
    match char_index::peek_header(&index_path) {
        Ok((version, feat_len)) => {
            let (exp_ver, exp_fl) = char_index::expected_header();
            if version != exp_ver || feat_len != exp_fl {
                return do_full_build(&system_fonts, &index_path, classifier);
            }
        }
        Err(e) => {
            return do_full_build(&system_fonts, &index_path, classifier);
        }
    }

    // Header OK — load the full index for incremental comparison.
    let start = std::time::Instant::now();
    let mut index = match char_index::load_index(&index_path, classifier) {
        Ok(idx) => idx,
        Err(e) => {
            return do_full_build(&system_fonts, &index_path, classifier);
        }
    };
    let load_time = start.elapsed();
    let indexed_names = index.font_names(); // includes both indexed + skipped
    let indexed_count = index.indexed_font_names().len();
    let skipped_count = index.skipped_fonts.len();

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
        return Ok(());
    }

    // ── Remove stale fonts ─────────────────────────────────────────
    if !removed_fonts.is_empty() {
        for name in removed_fonts.iter().take(10) {
        }
        if removed_fonts.len() > 10 {
        }
        index.remove_fonts(&removed_fonts, classifier);
    }

    // ── Build entries for new fonts only ────────────────────────────
    if !new_fonts.is_empty() {
        for name in new_fonts.iter().take(10) {
        }
        if new_fonts.len() > 10 {
        }
        let pairs: Vec<(String, PathBuf, char_index::GlyphOverrides)> = new_fonts
            .iter()
            .filter_map(|name| system_fonts.get(name).map(|(p, g)| (name.clone(), p.clone(), g.clone())))
            .collect();
        let start = std::time::Instant::now();
        let partial = char_index::build_char_index(&pairs, classifier);
        index.merge(partial, classifier);
    }

    // ── Save updated index ─────────────────────────────────────────
    if let Some(parent) = index_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    char_index::save_index(&index, &index_path).map_err(ScanTextError::Io)?;
    index.compact(); // drop raw entries — flat_vecs is all we need for search
    let file_size = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
    let final_count = char_index::count_fonts(&index);

    if !new_fonts.is_empty() {
    }
    if !removed_fonts.is_empty() {
    }

    Ok(())
}

/// Reconstruct a font catalog from cached index metadata.
fn catalog_from_meta(index: &char_index::CharIndex) -> Vec<font_scan::FontEntry> {
    index.font_meta.iter().map(|(_key, meta)| {
        font_scan::FontEntry {
            path: meta.path.clone(),
            family_name: meta.family_name.clone(),
            postscript_name: meta.postscript_name.clone(),
            is_bold: meta.is_bold,
            is_italic: meta.is_italic,
            class: match meta.class {
                0 => font_scan::FontClass::Serif,
                1 => font_scan::FontClass::Sans,
                2 => font_scan::FontClass::Mono,
                _ => font_scan::FontClass::Unknown,
            },
            data: Vec::new(),
            oldstyle_figures: meta.oldstyle_figures,
            variant_tag: meta.variant_tag.clone(),
            glyph_overrides: meta.glyph_overrides.clone(),
        }
    }).collect()
}

/// Scan fonts from filesystem, build char index, and return both.
fn scan_and_build_index(
    args: &cli::Args,
    classifier: &dyn classifier::Classifier,
) -> Result<(char_index::CharIndex, Vec<font_scan::FontEntry>), ScanTextError> {
    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    let _t_scan = std::time::Instant::now();
    let font_catalog = font_scan::scan_fonts(&font_dirs);
    if font_catalog.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    let index_path = args.resolved_index_path();
    let start = std::time::Instant::now();
    let pairs: Vec<(String, PathBuf, char_index::GlyphOverrides)> = font_catalog
        .iter()
        .map(|e| (e.font_key(), e.path.clone(), e.glyph_overrides.clone()))
        .collect();
    let mut index = char_index::build_char_index(&pairs, classifier);
    let elapsed = start.elapsed();

    index.populate_font_meta(&font_catalog);

    if let Some(parent) = index_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match char_index::save_index(&index, &index_path) {
        Ok(()) => {
            let sz = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
        }
        Err(e) => {
        }
    }
    index.compact();

    Ok((index, font_catalog))
}

/// Full index build from scratch (used for first build, format changes, --rebuild-index).
fn do_full_build(
    system_fonts: &std::collections::HashMap<String, (PathBuf, char_index::GlyphOverrides)>,
    index_path: &Path,
    classifier: &dyn classifier::Classifier,
) -> Result<(), ScanTextError> {
    let pairs: Vec<(String, PathBuf, char_index::GlyphOverrides)> = system_fonts
        .iter()
        .map(|(n, (p, g))| (n.clone(), p.clone(), g.clone()))
        .collect();


    let start = std::time::Instant::now();
    let mut index = char_index::build_char_index(&pairs, classifier);
    let elapsed = start.elapsed();

    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent).map_err(ScanTextError::Io)?;
    }
    char_index::save_index(&index, index_path).map_err(ScanTextError::Io)?;

    let file_size = std::fs::metadata(index_path).map(|m| m.len()).unwrap_or(0);
    let n_entries: usize = index.n_entries();
    index.compact();


    Ok(())
}

/// Load or build the character index with caching.
fn run(args: &cli::Args, classifier: &dyn classifier::Classifier) -> Result<(), ScanTextError> {
    dump_limits();
    let input = args.input.as_ref().expect("input validated");
    let dev_null = std::path::PathBuf::from("/dev/null");
    let output = args.output.as_ref().unwrap_or(&dev_null);
    let test_mode = args.test.is_some();

    // ── Audit directory (created when --audit is set) ─────────────
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

    // ── 1. Load or build character index + font catalog ────────────
    let index_path = args.resolved_index_path();
    let (char_index, font_catalog) = if !args.rebuild_index && index_path.exists() {
        let start = std::time::Instant::now();
        match char_index::load_index(&index_path, classifier) {
            Ok(mut index) if !index.font_meta.is_empty() => {
                let n_entries: usize = index.n_entries();
                let catalog = catalog_from_meta(&index);
                index.compact();
                (index, catalog)
            }
            Ok(_) => {
                let (idx, cat) = scan_and_build_index(args, classifier)?;
                (idx, cat)
            }
            Err(e) => {
                let (idx, cat) = scan_and_build_index(args, classifier)?;
                (idx, cat)
            }
        }
    } else {
        scan_and_build_index(args, classifier)?
    };
    if font_catalog.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    // All font access goes through the shared cache below.

    // ── 1b''. Shared font cache for all post-index font access ──────
    let font_cache = font_cache::FontCache::new(font_cache::DEFAULT_CAPACITY);

    // ── 2. Load input pages (with raster cache) ──────────────────────
    let cache_dir = page_cache::cache_key(input, args.dpi)
        .and_then(|key| page_cache::cache_dir(&key));

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
    if std::env::var("UNSCAN_DEBUG_MEM").is_ok() {
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
    // Dominant font candidate for fast-path SSIM, persists across pages.
    let mut dominant_font_candidate: Option<font_match::FontMatchResult> = None;

    // Load ground truth from vector PDF (--audit or --test).
    let ground_truth: Option<ground_truth::GroundTruth> = if let Some(vpath) = args.gt_vector_pdf() {
        match ground_truth::GroundTruth::load(vpath) {
            Ok(gt) => Some(gt),
            Err(e) => {
                None
            }
        }
    } else {
        None
    };

    // Parse --pages filter (1-indexed page numbers).
    let page_filter: Option<std::collections::HashSet<usize>> = args
        .pages
        .as_deref()
        .map(|spec| cli::parse_pages(spec).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }));

    for (page_idx, page_img) in pages.iter().enumerate() {
        let page_num = page_idx + 1; // 1-indexed

        // Skip pages not in the --pages filter.
        if let Some(ref filter) = page_filter {
            if !filter.contains(&page_num) {
                // Still need a placeholder for PDF output ordering.
                all_pages.push(pdf_out::PageContent {
                    width_px: 0,
                    height_px: 0,
                    dpi: args.dpi,
                    text_regions: Vec::new(),
                    raster_fragments: Vec::new(),
                    lines: Vec::new(),
                    fills: Vec::new(),
                    bg_color: (255, 255, 255),
                });
                continue;
            }
        }


        // 3a-pre. Deskew ──────────────────────────────────────────────
        // Detect and correct skew on the grayscale image used for OCR
        // and font matching. The original colour page_img is kept for
        // background colour detection, geometry, and raster fragments.
        let orig_gray = page_img.to_luma8();
        let skew_angle = deskew::detect_skew(&orig_gray);
        let (deskewed_gray, did_deskew) = if skew_angle.abs() > 5.0 {
            (orig_gray.clone(), false)
        } else if skew_angle.abs() > 0.5 {
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
        ocr::snapshot_raw_bboxes(&mut lines);
        ocr::merge_overlapping_lines(&mut lines);
        ocr::clip_word_overlaps(&mut lines);
        ocr::drop_outlier_words(&mut lines);
        let ocr_elapsed = ocr_start.elapsed();

        // 3b. Background colour ───────────────────────────────────────
        let bg_color = color::detect_background_color(page_img);

        // 3c. Font match + decision matrix ────────────────────────────
        // Use the deskewed grayscale for character segmentation and matching
        let gray_page = deskewed_gray;

        // Expand OCR bboxes to actual ink extent — Tesseract often clips
        // descenders, under-reporting line height by up to 10 px.
        // Use a threshold relative to the background: anything darker than
        // (bg - 56) counts as ink (works for both light and dark backgrounds).
        let ink_thresh = bg_color.0.saturating_sub(56);
        let _t_pre = std::time::Instant::now();
        ocr::expand_bbox_to_ink(&mut lines, &gray_page, ink_thresh);
        ocr::expand_words_to_ink(&mut lines, &gray_page, ink_thresh);
        let mut placed_texts: Vec<pdf_out::PlacedText> = Vec::new();
        let mut pg_vec = 0u32;
        let mut pg_raster = 0u32;

        // ── Pass 1: Match all lines ──────────────────────────────────
        struct LineMatch {
            font_result: Option<font_match::FontMatchResult>,
            text_color: (u8, u8, u8),
            ci_top_for_audit: Vec<(String, Option<f32>)>,
            ci_char_detail: Vec<char_index::CharCiDetail>,
            ci_top_for_audit_lig: Vec<(String, Option<f32>)>,
            ci_char_detail_lig: Vec<char_index::CharCiDetail>,
            seg_winner: Option<String>,
            diag_seg_dir: Option<std::path::PathBuf>,
            /// Per-char distances to the chosen font, keyed by crop_index.
            chosen_char_dists: std::collections::HashMap<usize, f32>,
            /// Per-char distances to the ground-truth font (--audit mode), keyed by crop_index.
            gt_font_char_dists: std::collections::HashMap<usize, f32>,
            /// CI tie-break candidates with per-candidate SSIM scores.
            tie_candidates: Vec<audit::TieCandidate>,
        }

        let fontmatch_start = std::time::Instant::now();

        // ── Parallel font matching with SSIM fast path ───────────────
        // If we have a dominant font candidate (from a previous page or
        // seeded from this page), each thread tries it via SSIM first.
        // Lines that pass skip segmentation and CI entirely; misses fall
        // through to the full pipeline.  Everything runs in parallel.
        const FAST_PATH_MIN_SSIM: f32 = 0.90;
        let fast_path_candidate: Option<&font_match::FontMatchResult> =
            dominant_font_candidate.as_ref();
        let fast_path_font_data: Option<std::sync::Arc<Vec<u8>>> = fast_path_candidate
            .and_then(|fm| font_cache.load(&fm.font_path).ok());
        let fast_path_hits = std::sync::atomic::AtomicU64::new(0);

        // Profiling accumulators (microseconds, atomic for par_iter)
        let prof_seg_us = std::sync::atomic::AtomicU64::new(0);
        let prof_ci_us = std::sync::atomic::AtomicU64::new(0);
        let prof_pcd_us = std::sync::atomic::AtomicU64::new(0);
        let prof_fp_us = std::sync::atomic::AtomicU64::new(0);
        let prof_full_us = std::sync::atomic::AtomicU64::new(0);
        let line_matches: Vec<LineMatch> = lines.par_iter().enumerate().map(|(li, line)| {
            let line_num = li + 1; // 1-indexed for output
            let line_start = std::time::Instant::now();
            // ── Fast path: try dominant font via SSIM ────────────────
            if let (Some(fm), Some(ref fd)) = (fast_path_candidate, &fast_path_font_data) {
                let (score, _dy) = verify::verify_text_region(
                    &gray_page,
                    fd.as_slice(),
                    &line.text,
                    line.x, line.y,
                    line.width, line.height,
                    &line.words,
                    fm.glyph_overrides.as_deref(),
                    &fm.variant_tag,
                    None,
                    Some(FAST_PATH_MIN_SSIM),
                );
                if score >= FAST_PATH_MIN_SSIM {
                    fast_path_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    prof_fp_us.fetch_add(line_start.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
                    let mut result = fm.clone();
                    result.best_dy = _dy;
                    return LineMatch {
                        font_result: Some(result),
                        text_color,
                        ci_top_for_audit: Vec::new(),
                        ci_char_detail: Vec::new(),
                        ci_top_for_audit_lig: Vec::new(),
                        ci_char_detail_lig: Vec::new(),
                        seg_winner: None,
                        diag_seg_dir: None,
                        chosen_char_dists: std::collections::HashMap::new(),
                        gt_font_char_dists: std::collections::HashMap::new(),
                        tie_candidates: Vec::new(),
                    };
                } else if li < 3 {
                }
            }

            // ── Full pipeline: segmentation → CI search → font match ─
            let preview_end = {
                let mut end = line.text.len().min(30);
                while end > 0 && !line.text.is_char_boundary(end) { end -= 1; }
                end
            };
            let debug_mem = std::env::var("UNSCAN_DEBUG_MEM").is_ok();
            if debug_mem {
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
            let mut ci_top_for_audit: Vec<(String, Option<f32>)>;
            let ci_char_detail: Vec<char_index::CharCiDetail>;
            let ci_top_for_audit_lig: Vec<(String, Option<f32>)>;
            let ci_char_detail_lig: Vec<char_index::CharCiDetail>;
            let seg_winner: Option<String>;
            let diag_seg_dir: Option<std::path::PathBuf> = args.diag_seg_dir().map(|d| {
                let line_slug: String = line.text.chars().take(30)
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let p = d.join(format!("p{}_L{:03}_{}", page_num, line_num, line_slug));
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
            let seg_t0 = std::time::Instant::now();
            let line_crops = char_index::extract_line_chars(
                &gray_page, &word_placements, line_height,
                diag_seg_dir.as_deref(),
            );
            prof_seg_us.fetch_add(seg_t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            let char_crops = &line_crops.plain;
            if debug_mem {
            }

            let (font_result, tie_candidates_audit) = {

                // Crop PNGs are saved after font matching, gated by ground-truth
                // miss detection when --audit is set (see below).

                // ── Score plain path ─────────────────────────────────
                let ci_t0 = std::time::Instant::now();
                let ci_result_plain = char_index::search_candidates(&char_index, char_crops, args.thoroughness, args.full_audit(), classifier);

                // ── Score ligature path (if present) ─────────────────
                let ci_result_lig = line_crops.ligature.as_ref().map(|lig_crops| {
                    char_index::search_candidates(&char_index, lig_crops, args.thoroughness, args.full_audit(), classifier)
                });
                prof_ci_us.fetch_add(ci_t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
                    .map(|(name, score)| (name.clone(), Some(*score))).collect();
                ci_char_detail = ci_result.char_detail.clone();

                // Store the alternate path for audit
                let (ci_top_lig_audit, ci_char_lig_audit) = if let Some(ref lig_result) = ci_result_lig {
                    (lig_result.scores.iter().map(|(n, s)| (n.clone(), Some(*s))).collect(),
                     lig_result.char_detail.clone())
                } else {
                    (Vec::new(), Vec::new())
                };
                let (ci_top_plain_audit, ci_char_plain_audit) = (
                    ci_result_plain.scores.iter().map(|(n, s)| (n.clone(), Some(*s))).collect::<Vec<_>>(),
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
                }

                // Crop PNGs saved after font matching (see below).

                // ── Font selection: CI #1, with SSIM tie-break ───────
                let mut tie_candidates_audit: Vec<audit::TieCandidate> = Vec::new();
                if let Some((_top_key, top_score)) = ci_result.scores.first() {
                    // Collect all candidates that share the top CI score
                    let tied: Vec<&(String, f32)> = ci_result.scores.iter()
                        .take_while(|(_, s)| *s == *top_score)
                        .collect();

                    if tied.len() >= 2 {
                        // Multiple candidates tied — SSIM decides
                        let mut best: Option<(font_match::FontMatchResult, f32)> = None;
                        let mut log_parts: Vec<String> = Vec::new();
                        let mut tie_ssim_results: Vec<(String, String, f32)> = Vec::new();
                        for (ti, (key, _)) in tied.iter().enumerate() {
                            let fe = match font_catalog.iter().find(|fe| fe.font_key() == *key) {
                                Some(fe) => fe,
                                None => continue,
                            };
                            let fd = match font_cache.load(&fe.path).ok() {
                                Some(fd) => fd,
                                None => continue,
                            };
                            // Save per-candidate SSIM images when audit dir exists
                            let tie_audit_dir = diag_seg_dir.as_ref().map(|d| {
                                let p = d.join(format!("tie_{}", ti));
                                let _ = std::fs::create_dir_all(&p);
                                p
                            });
                            let (ssim, dy) = verify::verify_text_region(
                                &gray_page, &fd, &line.text,
                                line.x, line.y, line.width, line.height,
                                &line.words,
                                fe.glyph_overrides.as_deref(), &fe.variant_tag,
                                tie_audit_dir.as_deref(), None,
                            );
                            log_parts.push(format!("{:.4}({})", ssim, fe.family_name));
                            tie_ssim_results.push((fe.font_key(), fe.family_name.clone(), ssim));
                            if best.as_ref().map_or(true, |(_, bs)| ssim > *bs) {
                                best = Some((font_match::FontMatchResult {
                                    font_name: fe.family_name.clone(),
                                    font_path: fe.path.clone(),
                                    font_key: fe.font_key(),
                                    variant_tag: fe.variant_tag.clone(),
                                    glyph_overrides: fe.glyph_overrides.clone(),
                                    score: *top_score,
                                    best_dy: dy,
                                }, ssim));
                            }
                        }
                        // Build tie_candidates for audit
                        let winner_key = best.as_ref().map(|(fm, _)| fm.font_key.clone());
                        for (fk, fname, ssim) in tie_ssim_results {
                            tie_candidates_audit.push(audit::TieCandidate {
                                font_key: fk.clone(),
                                family_name: fname,
                                ssim_score: ssim,
                                winner: Some(&fk) == winner_key.as_ref(),
                            });
                        }
                        if let Some((ref winner, _)) = best {
                        }
                        (best.map(|(fm, _)| fm), tie_candidates_audit)
                    } else {
                        // No tie — use CI #1 directly
                        let (ref key, score) = *tied[0];
                        (font_catalog.iter().find(|fe| fe.font_key() == *key)
                            .map(|fe| font_match::FontMatchResult {
                                font_name: fe.family_name.clone(),
                                font_path: fe.path.clone(),
                                font_key: fe.font_key(),
                                variant_tag: fe.variant_tag.clone(),
                                glyph_overrides: fe.glyph_overrides.clone(),
                                score,
                                best_dy: 0,
                            }), Vec::new())
                    }
                } else {
                    (None, Vec::new())
                }
            };
            let line_elapsed = line_start.elapsed();
            if line_elapsed.as_millis() > 500 {
            }
            // ── Ground-truth gated audit detail ─────────────────────────
            // When --audit is set, check if this line is a miss before
            // doing expensive audit I/O.  Without --audit, all lines
            // get full audit.  "Miss" means: ground-truth font mismatch, no
            // font matched, OCR too low, or font confidence too low.
            let is_miss = if let Some(ref gt) = ground_truth {
                // OCR too low → line will be kept raster, treat as miss
                let ocr_ok = line.confidence >= args.min_ocr_confidence as f32
                    && !line.text.trim().is_empty();
                if !ocr_ok {
                    true
                } else if let Some(ref fr) = font_result {
                    // Font confidence too low → kept raster, treat as miss
                    if fr.score < args.min_font_confidence {
                        true
                    } else {
                        let bbox_px = [line.x as f32, line.y as f32,
                                       (line.x + line.width) as f32,
                                       (line.y + line.height) as f32];
                        // Look up chosen font's PostScript name for exact comparison
                        let chosen_ps = font_catalog.iter()
                            .find(|fe| fe.font_key() == fr.font_key)
                            .map(|fe| fe.postscript_name.as_str())
                            .unwrap_or("");
                        !gt.is_hit(page_num, &bbox_px, args.dpi, chosen_ps)
                    }
                } else {
                    true // no font matched → treat as miss
                }
            } else {
                true // no ground truth → full audit for all lines
            };

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

            let pcd_t0 = std::time::Instant::now();

            // Per-char distances and audit detail: only for miss lines when full audit is active
            let (chosen_char_dists, gt_font_char_dists) = if is_miss && args.full_audit() {
                // Precompute features once for all per-char distance lookups
                let crop_feats = char_index::precompute_crop_features(&corrected_char_crops, classifier);

                let chosen: std::collections::HashMap<usize, f32> = if let Some(ref fr) = font_result {
                    if !fr.font_key.is_empty() {
                        char_index::per_char_distances_precomputed(&char_index, &fr.font_key, &crop_feats)
                            .into_iter()
                            .map(|(_, crop_idx, d2)| (crop_idx, d2))
                            .collect()
                    } else {
                        std::collections::HashMap::new()
                    }
                } else {
                    std::collections::HashMap::new()
                };

                // Per-char distances to the ground-truth font (if known)
                let gt_dists: std::collections::HashMap<usize, f32> = if let Some(ref gt) = ground_truth {
                    let bbox_px = [line.x as f32, line.y as f32,
                                   (line.x + line.width) as f32,
                                   (line.y + line.height) as f32];
                    if let Some(gt_font_name) = gt.lookup_font(page_num, &bbox_px, args.dpi) {
                        // Inject GT font into ci_top_for_audit so it appears in audit output
                        let gt_ps = ground_truth::strip_subset_prefix_str(gt_font_name);
                        let gt_key = font_catalog.iter()
                            .find(|fe| fe.postscript_name == gt_ps)
                            .map(|fe| fe.font_key());
                        if let Some(ref gk) = gt_key {
                            if !ci_top_for_audit.iter().any(|(n, _)| n == gk) {
                                ci_top_for_audit.push((gk.clone(), None));
                            }
                            // Compute per-char distances to the GT font
                            char_index::per_char_distances_precomputed(&char_index, gk, &crop_feats)
                                .into_iter()
                                .map(|(_, crop_idx, d2)| (crop_idx, d2))
                                .collect()
                        } else {
                            std::collections::HashMap::new()
                        }
                    } else {
                        std::collections::HashMap::new()
                    }
                } else {
                    std::collections::HashMap::new()
                };

                // Save crop PNGs for miss lines
                if let Some(ref ddir) = diag_seg_dir {
                    if !effective_crops.is_empty() {
                        let crop_dir = ddir.join("crops");
                        let _ = std::fs::create_dir_all(&crop_dir);
                        for (i, (ch, img)) in effective_crops.iter().enumerate() {
                            let path = crop_dir.join(format!("crop_{:02}_{}.png", i,
                                if ch.is_alphanumeric() { format!("{}", ch) }
                                else { format!("U{:04X}", *ch as u32) }));
                            let _ = img.save(&path);
                        }
                    }

                    // Save full-colour scan line crop for report overlay
                    // Crop region = union of all word bboxes (raw + final) with padding
                    let all_wbs: Vec<(u32, u32, u32, u32)> = line.words.iter()
                        .map(|w| (w.x, w.y, w.width, w.height))
                        .chain(
                            line.raw_words.iter()
                                .map(|w| (w.x, w.y, w.width, w.height))
                        )
                        .collect();
                    if !all_wbs.is_empty() {
                        let pad = 4u32;
                        let ux = all_wbs.iter().map(|b| b.0).min().unwrap().saturating_sub(pad);
                        let uy = all_wbs.iter().map(|b| b.1).min().unwrap().saturating_sub(pad);
                        let ur = all_wbs.iter().map(|b| b.0 + b.2).max().unwrap().saturating_add(pad).min(page_img.width());
                        let ub = all_wbs.iter().map(|b| b.1 + b.3).max().unwrap().saturating_add(pad).min(page_img.height());
                        if ur > ux && ub > uy {
                            let crop = image::imageops::crop_imm(page_img, ux, uy, ur - ux, ub - uy).to_image();
                            let _ = crop.save(ddir.join("scan_line.png"));
                            let _ = std::fs::write(
                                ddir.join("scan_line_origin.json"),
                                format!("{{\"x\":{},\"y\":{}}}", ux, uy),
                            );
                        }
                    }
                }

                // Render font ref glyphs for miss lines
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
                                if path.exists() { continue; }
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

                (chosen, gt_dists)
            } else {
                (std::collections::HashMap::new(), std::collections::HashMap::new())
            };
            prof_pcd_us.fetch_add(pcd_t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);

            prof_full_us.fetch_add(line_start.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            LineMatch { font_result, text_color, ci_top_for_audit, ci_char_detail, ci_top_for_audit_lig, ci_char_detail_lig, seg_winner, diag_seg_dir, chosen_char_dists, gt_font_char_dists, tie_candidates: tie_candidates_audit }
        }).collect();

        // Update dominant font candidate for next page from this page's results
        {
            let mut font_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for lm in &line_matches {
                if let Some(ref fr) = lm.font_result {
                    *font_freq.entry(fr.font_key.clone()).or_insert(0) += 1;
                }
            }
            if let Some((top_key, count)) = font_freq.iter().max_by_key(|(_, c)| *c) {
                dominant_font_candidate = line_matches.iter()
                    .find_map(|lm| lm.font_result.as_ref()
                        .filter(|fr| fr.font_key == *top_key)
                        .cloned());
            }
        }

        let fp_hits = fast_path_hits.load(std::sync::atomic::Ordering::Relaxed);
        let ci_lines = lines.len() as u64 - fp_hits;
        let fontmatch_elapsed = fontmatch_start.elapsed();
        if fp_hits > 0 {
        }
        if ci_lines > 0 {
        }

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
                }
            }
        }

        // ── Word split: split wide whitespace using matched fonts ──
        let _t_split = std::time::Instant::now();
        {
            let line_fonts: Vec<Option<std::sync::Arc<Vec<u8>>>> = line_matches.iter()
                .map(|lm| {
                    lm.font_result.as_ref().and_then(|fm| {
                        font_cache.load(&fm.font_path).ok()
                    })
                })
                .collect();
            ocr::split_wide_whitespace_words(&mut lines, &gray_page, ink_thresh, &line_fonts);
        }

        // ── Pass 2a: Parallel SSIM verification ─────────────────────
        let verify_start = std::time::Instant::now();
        let ssim_results: Vec<(Option<f32>, Option<bool>)> = lines.par_iter()
            .enumerate()
            .map(|(li, line)| {
                let lm = &line_matches[li];
                let font_result = &lm.font_result;

                let ocr_ok = line.confidence >= args.min_ocr_confidence as f32
                    && !line.text.trim().is_empty();
                let font_ok = font_result
                    .as_ref()
                    .map(|f| f.score >= args.min_font_confidence)
                    .unwrap_or(false);

                if !ocr_ok || !font_ok {
                    return (None, None);
                }

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
                            None,
                        );
                        let pass = score >= MIN_VERIFY_SSIM;
                        (Some(score), Some(pass))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            })
            .collect();
        let verify_count = ssim_results.iter().filter(|(s, _)| s.is_some()).count() as u32;

        // ── Pass 2b: Decision matrix + output ────────────────────────
        for (li, line) in lines.iter().enumerate() {
            let line_num = li + 1; // 1-indexed for output
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

            let (ssim_score, ssim_pass) = ssim_results[li];

            // ── Logging ──────────────────────────────────────────────
            if keep_raster {
                pg_raster += 1;
            } else {
                pg_vec += 1;
                let fname = font_result.as_ref().map(|f| f.font_name.as_str()).unwrap_or("?");
                let fscore = font_result.as_ref().map(|f| f.score).unwrap_or(0.0);
                let ssim_part = ssim_score
                    .map(|s| format!(" ssim={s:.3}"))
                    .unwrap_or_default();
            }

            // ── Diag-seg line-level summary ──────────────────────────
            if let Some(ref ddir) = lm.diag_seg_dir {
                let fname = font_result.as_ref().map(|f| f.font_name.as_str()).unwrap_or("?");
                let fscore = font_result.as_ref().map(|f| f.score).unwrap_or(0.0);
                let line_summary = serde_json::json!({
                    "page": page_num,
                    "line_index": line_num,
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
                page: page_num,
                line_index: line_num,
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
                        let gt_d2 = lm.gt_font_char_dists.get(&d.crop_index).copied();
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
                            gt_font_dist_sq: gt_d2,
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
                            gt_font_dist_sq: None,
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
                word_bboxes_raw: line.raw_words.iter().map(|w| audit::WordBBox {
                    text: w.text.clone(),
                    x: w.x,
                    y: w.y,
                    width: w.width,
                    height: w.height,
                    confidence: w.confidence,
                }).collect(),
                tie_candidates: lm.tie_candidates.clone(),
                miss_type: None,
                expected_font: None,
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
                    page: page_num,
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
                    page: page_num,
                    kind: "fill",
                    bbox: BBox { x: f.x, y: f.y, width: f.width, height: f.height },
                });
            }

            let count = geo.lines.len() + geo.fills.len();
            stat_geo_elements += count as u32;
            if count > 0 {
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
            }
        }

        page_summaries.push(PageSummary {
            page: page_num,
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

    if !test_mode {
        pdf_out::generate_pdf(output, &all_pages, args.overlay, &font_cache)?;
    }

    let output_size = if test_mode { 0 } else { std::fs::metadata(output).map(|m| m.len()).unwrap_or(0) };
    let ratio = if output_size > 0 { input_size as f64 / output_size as f64 } else { 0.0 };

    // ── 5. Accuracy & audit ──────────────────────────────────────────
    // Build a minimal AuditLog for classification (needed by both --test
    // and full audit paths).
    let mut audit = AuditLog {
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
    // Enrich entries with ground-truth classification before writing JSON.
    report::enrich_audit_entries(
        &mut audit.text_entries,
        ground_truth.as_ref(),
        args.dpi,
        &font_catalog,
    );

    if test_mode {
        // ── Test mode: compute accuracy, print JSON to stdout ────────
        let acc = report::compute_accuracy(
            &audit.text_entries,
            ground_truth.as_ref(),
            args.dpi,
            &font_catalog,
        );
        // JSON output to stdout
        let test_json = serde_json::json!({
            "primary_hits": acc.primary_hits,
            "compared": acc.compared,
            "pct": (acc.pct * 10.0).round() / 10.0,
            "major_misses": acc.major_misses,
            "minor_misses": acc.minor_misses,
            "ssim_failures": acc.ssim_failures,
            "hits": acc.hits,
            "kept_raster": acc.kept_raster,
        });
        println!("{}", serde_json::to_string_pretty(&test_json).unwrap());

        // If --audit is also set, write audit artifacts + HTML report
        if let Some(ref audit_root) = args.audit {
            let audit_path = args.audit_log_path();
            if let Err(e) = audit.write_to_file(&audit_path) {
            } else {
            }
            let report_path = audit_root.join("report.html");
            if let Err(e) = report::generate_report(
                &report_path,
                audit_root,
                &audit.text_entries,
                ground_truth.as_ref(),
                args.dpi,
                &font_catalog,
            ) {
            }
        }
        return Ok(());
    }

    let audit_path = args.audit_log_path();

    if output.to_str() != Some("/dev/null") || args.audit.is_some() {
        if let Err(e) = audit.write_to_file(&audit_path) {
        } else {
        }
    }

    // ── 5b. HTML miss report ─────────────────────────────────────────
    if let Some(ref audit_root) = args.audit {
        let report_path = audit_root.join("report.html");
        if let Err(e) = report::generate_report(
            &report_path,
            audit_root,
            &audit.text_entries,
            ground_truth.as_ref(),
            args.dpi,
            &font_catalog,
        ) {
        }
    }

    // ── 6. Report ────────────────────────────────────────────────────
    let (cache_hits, cache_misses) = font_cache.stats();

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

