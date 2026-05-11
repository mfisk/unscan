mod audit;
mod cli;
mod color;
mod error;
mod font_match;
mod font_scan;
mod geometry;
pub mod char_index;
pub mod layout;
mod compare;
mod ocr;
mod pdf_out;
mod smooth;
pub(crate) mod verify;

use crate::audit::{AuditEntry, AuditLog, BBox, Decision, GeometryEntry, PageSummary};
use crate::error::ScanTextError;
use crate::ocr::TextRegion;
use image::DynamicImage;
use log::{debug, info, warn};
use rayon::prelude::*;
use std::path::Path;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let args = cli::parse();
    if let Err(msg) = args.validate() {
        eprintln!("Error: {msg}");
        std::process::exit(1);
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

    // Collect system font keys → data for lookup.
    // Keys are unique per weight/style/variant (path + optional variant tag).
    let system_fonts: std::collections::HashMap<String, Vec<u8>> = font_catalog
        .iter()
        .map(|e| (e.font_key(), e.data.clone()))
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
            indexed_count, skipped_count, index.entries.len());
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
        let pairs: Vec<(String, Vec<u8>)> = new_fonts
            .iter()
            .filter_map(|name| system_fonts.get(name).map(|d| (name.clone(), d.clone())))
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
    info!("  Characters:         {}", index.entries.len());
    info!("  Index size:         {:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    info!("  Saved to:           {}", index_path.display());
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

/// Full index build from scratch (used for first build, format changes, --rebuild-index).
fn do_full_build(
    system_fonts: &std::collections::HashMap<String, Vec<u8>>,
    index_path: &Path,
) -> Result<(), ScanTextError> {
    let pairs: Vec<(String, Vec<u8>)> = system_fonts
        .iter()
        .map(|(n, d)| (n.clone(), d.clone()))
        .collect();

    info!("Building character index ({} fonts × {} chars)…",
        pairs.len(), char_index::indexed_chars().len());

    let start = std::time::Instant::now();
    let index = char_index::build_char_index(&pairs);
    let elapsed = start.elapsed();

    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent).map_err(ScanTextError::Io)?;
    }
    char_index::save_index(&index, index_path).map_err(ScanTextError::Io)?;

    let file_size = std::fs::metadata(index_path).map(|m| m.len()).unwrap_or(0);
    let n_entries: usize = index.entries.values().map(|v| v.len()).sum();

    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("          CHARACTER INDEX BUILT");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("  Fonts indexed:      {}", pairs.len());
    info!("  Characters:         {}", index.entries.len());
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
            Ok(index) => {
                let n_entries: usize = index.entries.values().map(|v| v.len()).sum();
                info!(
                    "  Loaded {} chars × {} entries in {:.1}s",
                    index.entries.len(),
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
    let pairs: Vec<(String, Vec<u8>)> = font_catalog
        .iter()
        .map(|e| (e.font_key(), e.data.clone()))
        .collect();
    let index = char_index::build_char_index(&pairs);
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

    Ok(index)
}

fn run(args: &cli::Args) -> Result<(), ScanTextError> {
    let input = args.input.as_ref().expect("input validated");
    let output = args.output.as_ref().expect("output validated");

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

    // ── 2. Load input pages ──────────────────────────────────────────
    info!("Loading input pages…");
    let pages = load_pages(input, args.dpi)?;
    info!("Loaded {} page(s)", pages.len());

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

        // 3a. OCR ─────────────────────────────────────────────────────
        let (word_regions, page_char_boxes) = ocr::extract_text_regions(page_img, args.dpi)?;
        let mut lines = ocr::assemble_lines(&word_regions);
        info!("  OCR: {} words → {} lines", word_regions.len(), lines.len());

        // 3b. Background colour ───────────────────────────────────────
        let bg_color = color::detect_background_color(page_img);
        info!("  Background: #{:02x}{:02x}{:02x}", bg_color.0, bg_color.1, bg_color.2);

        // 3c. Font match + decision matrix ────────────────────────────
        let gray_page = page_img.to_luma8();

        // Expand OCR bboxes to actual ink extent — Tesseract often clips
        // descenders, under-reporting line height by up to 10 px.
        // Use a threshold relative to the background: anything darker than
        // (bg - 56) counts as ink (works for both light and dark backgrounds).
        let ink_thresh = bg_color.0.saturating_sub(56);
        ocr::expand_bbox_to_ink(&mut lines, &gray_page, ink_thresh);
        let mut placed_texts: Vec<pdf_out::PlacedText> = Vec::new();
        let mut pg_vec = 0u32;
        let mut pg_raster = 0u32;

        // ── Pass 1: Match all lines ──────────────────────────────────
        struct LineMatch {
            font_result: Option<font_match::FontMatchResult>,
            text_color: (u8, u8, u8),
        }

        // Pre-parse all catalog fonts once (avoids per-candidate per-line re-parsing)
        let parsed_fonts: Vec<Option<ab_glyph::FontRef>> = font_catalog.iter()
            .map(|e| ab_glyph::FontRef::try_from_slice(&e.data).ok())
            .collect();

        let mut line_matches: Vec<LineMatch> = lines.par_iter().map(|line| {
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
            let font_result = {
                // Query char index for candidate font keys
                let word_placements: Vec<crate::verify::WordPlacement> = line.words.iter()
                    .map(|w| crate::verify::WordPlacement {
                        text: w.text.clone(),
                        x_off: w.x,
                        y_off: w.y,
                        width: w.width,
                        height: w.height,
                    })
                    .collect();
                let line_height = line.words.iter().map(|w| w.height).max().unwrap_or(0);
                let char_crops = char_index::extract_line_chars(&gray_page, &word_placements, line_height, &page_char_boxes);
                let ci_results = char_index::search_candidates(&char_index, &char_crops, 50);
                let ci_keys: std::collections::HashSet<String> = ci_results.into_iter().map(|(name, _score)| name).collect();
                let ci_arg = if ci_keys.is_empty() { None } else { Some(&ci_keys) };
                let result = font_match::match_font(
                    &gray_page, line, &font_catalog, &parsed_fonts,
                    args.min_font_confidence, args.dpi,
                    ci_arg,
                );
                // Fallback: if gated match returned None but we had ci_keys, retry ungated
                if result.is_none() && ci_arg.is_some() {
                    font_match::match_font(
                        &gray_page, line, &font_catalog, &parsed_fonts,
                        args.min_font_confidence, args.dpi,
                        None,
                    )
                } else {
                    result
                }
            };
            LineMatch { font_result, text_color }
        }).collect();

        // ── Pass 1.5: Paragraph-level font grouping ─────────────────
        // Find the dominant body font: most common font among matched lines
        // at the most common font size (±1pt tolerance).
        {
            use std::collections::HashMap;
            // Collect (font_name, font_size_bucket) frequencies
            let mut size_freq: HashMap<i32, u32> = HashMap::new();
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
                let mut font_freq: HashMap<String, (u32, Vec<u8>)> = HashMap::new();
                for (i, lm) in line_matches.iter().enumerate() {
                    let sz = lines[i].font_size_pt.round() as i32;
                    if (sz - body_size).abs() <= 1 {
                        if let Some(ref fm) = lm.font_result {
                            if fm.score >= args.min_font_confidence {
                                let entry = font_freq.entry(fm.font_name.clone())
                                    .or_insert_with(|| (0, fm.font_data.clone()));
                                entry.0 += 1;
                            }
                        }
                    }
                }
                // Find majority font
                if let Some((majority_name, (majority_count, majority_data))) = font_freq.iter()
                    .max_by_key(|(_, (count, _))| *count)
                {
                    let total_body: u32 = font_freq.values().map(|(c, _)| c).sum();
                    debug!("  paragraph grouping: font_freq={:?} majority='{}' {}/{}", 
                        font_freq.iter().map(|(k,(c,_))| (k.as_str(), *c)).collect::<Vec<_>>(),
                        majority_name, majority_count, total_body);
                    // Only apply grouping if majority font has ≥40% of body lines
                    if *majority_count as f32 / total_body as f32 >= 0.4 && *majority_count >= 3 {
                        info!("  Paragraph grouping: '{}' is body font ({}/{} body lines)",
                            majority_name, majority_count, total_body);

                        // For body-size lines with a DIFFERENT font, try the majority font
                        for (i, lm) in line_matches.iter_mut().enumerate() {
                            let sz = lines[i].font_size_pt.round() as i32;
                            if (sz - body_size).abs() > 1 { continue; }

                            let current_name = lm.font_result.as_ref()
                                .map(|f| f.font_name.as_str()).unwrap_or("");
                            if current_name == majority_name.as_str() { continue; }

                            // Run SSIM with majority font
                            let (majority_ssim, _majority_dy) = verify::verify_text_region(
                                &gray_page,
                                majority_data,
                                "",
                                lines[i].x, lines[i].y,
                                lines[i].width, lines[i].height,
                                &lines[i].words,
                            );

                            let current_score = lm.font_result.as_ref()
                                .map(|f| f.score).unwrap_or(0.0);

                            // Accept majority font if its SSIM is within 90% of current winner,
                            // OR if current line is raster (no good match) and majority SSIM
                            // meets threshold.
                            let switch = if current_score >= args.min_font_confidence {
                                majority_ssim >= current_score * 0.90
                            } else {
                                majority_ssim >= args.min_font_confidence
                            };

                            if switch {
                                debug!("  paragraph regroup: '{}' line {}: {} ({:.3}) → {} ({:.3})",
                                    truncate_str(&lines[i].text, 30), i,
                                    current_name, current_score,
                                    majority_name, majority_ssim);
                                // Find the majority font's path from the catalog
                                let majority_path = font_catalog.iter()
                                    .find(|e| e.family_name == *majority_name)
                                    .map(|e| e.path.clone())
                                    .unwrap_or_default();
                                lm.font_result = Some(font_match::FontMatchResult {
                                    font_name: majority_name.clone(),
                                    font_path: majority_path,
                                    score: majority_ssim,
                                    font_data: majority_data.clone(),
                                    best_dy: 0, // majority override, no shift data
                                    ssim_verified: false,
                                });
                            }
                        }
                    }
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

            let (mut keep_raster, mut reason) = if !ocr_ok {
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
            let mut ssim_score: Option<f32> = None;
            if !keep_raster && !args.no_verify {
                if let Some(ref fm) = font_result {
                    if fm.ssim_verified {
                        // Score already comes from SSIM rerank — skip redundant verify
                        ssim_score = Some(fm.score);
                        if fm.score < args.min_verify_ssim {
                            keep_raster = true;
                            reason = format!(
                                "SSIM verification failed ({:.3} < {:.2}). Reverted to raster.",
                                fm.score, args.min_verify_ssim
                            );
                        }
                    } else {
                        let (score, _dy) = verify::verify_text_region(
                            &gray_page,
                            &fm.font_data,
                            &line.text,
                            line.x,
                            line.y,
                            line.width,
                            line.height,
                            &line.words,
                        );
                        ssim_score = Some(score);
                        if score < args.min_verify_ssim {
                            keep_raster = true;
                            reason = format!(
                                "SSIM verification failed ({score:.3} < {:.2}). Reverted to raster.",
                                args.min_verify_ssim
                            );
                        }
                    }
                }
            }

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

            // ── Audit entry ──────────────────────────────────────────
            audit_text.push(AuditEntry {
                page: page_idx + 1,
                line_index: li,
                text: line.text.clone(),
                ocr_confidence: line.confidence,
                font_matched: font_result.as_ref().map(|f| f.font_name.clone()),
                font_confidence: font_result.as_ref().map(|f| f.score),
                ssim_score,
                decision: if keep_raster { Decision::KeptRaster } else { Decision::Vectorized },
                reason: reason.clone(),
                bbox: BBox {
                    x: line.x,
                    y: line.y,
                    width: line.width,
                    height: line.height,
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
                        if let Ok(f) = ab_glyph::FontRef::try_from_slice(&fm.font_data) {
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
            smooth::smooth_font_sizes(&mut page.text_regions, page.dpi as f32);
        }
    }

    info!("Generating output PDF: {}", output.display());
    pdf_out::generate_pdf(output, &all_pages, args.overlay)?;

    let output_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    let ratio = if output_size > 0 { input_size as f64 / output_size as f64 } else { 0.0 };

    // ── 5. Write audit log ───────────────────────────────────────────
    let audit_path = args.audit_log_path();
    let audit = AuditLog {
        input_file: input.display().to_string(),
        output_file: output.display().to_string(),
        input_size_bytes: input_size,
        output_size_bytes: output_size,
        compression_ratio: ratio,
        pages: page_summaries,
        text_entries: audit_text,
        geometry_entries: audit_geo,
    };
    audit.write_to_file(&audit_path)?;
    info!("Audit log: {}", audit_path.display());

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
    for row in 0..rows {
        for col in 0..cols {
            let cx = col * cell;
            let cy = row * cell;
            let cw = cell.min(w - cx);
            let ch = cell.min(h - cy);
            if color::region_has_content(cleaned_img, cx, cy, cw, ch) {
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

