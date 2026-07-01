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
pub mod features;
pub mod glyph_map;
pub mod char_render;
pub mod train;
pub mod layout;
mod compare;
pub mod seg_diag;
pub mod compare_rasters;
mod ocr;
mod page_cache;
mod pdf_out;
mod segment;
mod smooth;
pub(crate) mod verify;
pub mod ground_truth;
pub mod report;
mod font_pipeline;
mod zncc_classifier;
mod ngram;

use crate::audit::{AuditEntry, AuditLog, BBox, GeometryEntry, PageSummary};

use crate::error::ScanTextError;
use rayon::prelude::*;

/// Minimum SSIM score for SSIM verification to consider a font match acceptable.
const MIN_VERIFY_SIMILARITY: f32 = 0.8;

/// Standalone char rendering: render characters using the index-time
/// render_glyph_at_ink_height() pipeline and save as PNGs.

fn main() {
    let args = cli::parse();
    if let Err(msg) = args.validate() {
        eprintln!("Error: {msg}");
        std::process::exit(1);
    }

    // ── render-ref-chars: standalone char rendering, no PDF needed ───
    if let Some(ref json_str) = args.render_ref_chars {
        char_render::render_ref_chars_and_exit(json_str);
    }

    // ── train-lda: train LDA classifier and exit ────────────────────
    if args.train_lda {
        let train_args = train::TrainArgs {
            font_dir: args.font_dir.clone(),
            render_params: args.render_params(),
            ..train::TrainArgs::default()
        };
        train::run_train(train_args);
        std::process::exit(0);
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
    let mut clf: Box<dyn classifier::Classifier> = classifier::build_classifier(
        &args.classifier,
        args.triplet_weights.as_deref(),
        Some((&args.font_dir, &args.render_params())),
    );

    if let Err(e) = run(&args, &mut *clf) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}


/// Scan available fonts and return a FontRegistry.
/// Writes catalog.bin so classifier loaders can validate catalog_hash.
/// Centroids are already baked into classifier .bin files from training,
/// so no runtime render+embed step is needed.
fn load_fonts(args: &cli::Args, _classifier: &mut dyn classifier::Classifier) -> Result<font_scan::FontRegistry, ScanTextError> {
    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    let entries = font_scan::scan_fonts(&font_dirs);
    if entries.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    let registry = font_scan::FontRegistry::new(entries);

    // Write catalog.bin so classifier loaders can validate against it.
    let catalog_path = classifier::default_catalog_path();
    if let Err(e) = registry.write_fonts_bin(&catalog_path) {
        eprintln!("warning: could not write {}: {e}", catalog_path.display());
    }

    Ok(registry)
}

/// Load or build the character index with caching.
/// Build an [`AuditEntry`] from a line match, OCR line, and decision results.
fn build_audit_entry(
    lm: &font_pipeline::LineMatch,
    line: &ocr::TextLine,
    page_num: usize,
    line_num: usize,
    similarity_score: Option<f32>,
    similarity_pass: Option<bool>,
    keep_raster: bool,
    reason: &str,
) -> AuditEntry {
    use audit::{BBox, FontCandidate, ObservationVote, Decision, WordBBox};
    let font_result = &lm.font_result;
    let obs_vote = |d: &font_match::ObservationDetail, with_ranks: bool| {
        ObservationVote {
            seq: d.seq.clone(),
            weight: d.weight,
            crop_index: d.crop_index,
            best_prob: d.best_prob,
            passed_gate: d.passed_gate,
            nearest: d.nearest.clone(),
            crop_path: None,
            chosen_rank: if with_ranks { lm.chosen_obs_ranks.get(&d.crop_index).copied() } else { None },
            ocr_corrected_from: d.ocr_corrected_from,
            best_alt_char: d.best_alt_char,
            best_alt_dist: d.best_alt_dist,
            gt_font_rank: if with_ranks { lm.gt_font_obs_ranks.get(&d.crop_index).copied() } else { None },
            chosen_prob: if with_ranks { lm.chosen_obs_probs.get(&d.crop_index).copied() } else { None },
            gt_font_prob: if with_ranks { lm.gt_font_obs_probs.get(&d.crop_index).copied() } else { None },
        }
    };
    AuditEntry {
        page: page_num,
        line_index: line_num,
        text: line.text.clone(),
        ocr_confidence: line.confidence,
        font_matched: font_result.as_ref().map(|f| f.font_name.clone()),
        font_key_matched: font_result.as_ref().map(|f| f.font_key.clone()),
        font_confidence: font_result.as_ref().map(|f| f.score),
        similarity_score,
        similarity_pass,
        decision: if keep_raster { Decision::KeptRaster } else { Decision::Vectorized },
        reason: reason.to_string(),
        bbox: BBox { x: line.x, y: line.y, width: line.width, height: line.height },
        font_candidates: lm.font_scores.iter()
            .map(|(fk, s)| FontCandidate { font_key: fk.clone(), score: *s })
            .collect(),
        obs_votes: lm.observations.iter().map(|d| obs_vote(d, true)).collect(),
        font_candidates_lig: lm.font_scores_lig.iter()
            .map(|(fk, s)| FontCandidate { font_key: fk.clone(), score: *s })
            .collect(),
        obs_votes_lig: lm.observations_lig.iter().map(|d| obs_vote(d, false)).collect(),
        seg_winner: lm.seg_winner.clone(),
        word_bboxes: line.words.iter().map(|w| WordBBox {
            text: w.text.clone(), x: w.x, y: w.y, width: w.width, height: w.height, confidence: w.confidence,
        }).collect(),
        word_bboxes_raw: line.raw_words.iter().map(|w| WordBBox {
            text: w.text.clone(), x: w.x, y: w.y, width: w.width, height: w.height, confidence: w.confidence,
        }).collect(),
        tie_candidates: lm.tie_candidates.clone(),
        miss_type: None,
        expected_font: None,
    }
}

/// Write the HTML audit report to `<audit_root>/report.html`.
fn write_audit_report(
    audit_root: &std::path::Path,
    entries: &[AuditEntry],
    ground_truth: Option<&ground_truth::GroundTruth>,
    dpi: u32,
    font_entries: &[font_scan::FontEntry],
    glyph_map: &glyph_map::NgramGlyphMap,
    args: &cli::Args,
    elapsed: std::time::Duration,
) {
    let report_path = audit_root.join("report.html");
    let meta = report::ReportMeta {
        classifier: args.classifier.clone(),
        render_scale: args.render_scale,
        render_aa: args.render_aa.clone(),
        render_binarize: args.render_binarize,
        elapsed,
    };
    let _ = report::generate_report(
        &report_path,
        audit_root,
        entries,
        ground_truth,
        dpi,
        font_entries,
        glyph_map,
        &meta,
    );
}

fn run(args: &cli::Args, classifier: &mut dyn classifier::Classifier) -> Result<(), ScanTextError> {
    let run_start = std::time::Instant::now();
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

    // ── 1. Scan fonts and populate classifier ──────────────────────
    let font_registry = load_fonts(args, classifier)?;
    if font_registry.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    // Load glyph map (glyph dedup groups) — required; run --train-lda first.
    let gmap_path = glyph_map::NgramGlyphMap::default_path();
    let glyph_map = glyph_map::NgramGlyphMap::load(&gmap_path)
        .unwrap_or_else(|e| {
            eprintln!("Error: could not load glyph-map.bin ({e})");
            eprintln!("  expected at: {}", gmap_path.display());
            eprintln!("  Run with --train-lda first to generate it.");
            std::process::exit(1);
        });

    // All font access goes through the shared cache below.

    // ── 1b''. Shared font cache for all post-index font access ──────
    let font_cache = font_cache::FontCache::new(font_cache::DEFAULT_CAPACITY);

    // ── 2. Load input pages (with raster cache) ──────────────────────
    let cache_dir = page_cache::cache_key(input, args.dpi)
        .and_then(|key| page_cache::cache_dir(&key));

    let raster_start = std::time::Instant::now();
    let (pages, _raster_cached) = page_cache::get_pages(input, args.dpi)?;
    let _raster_elapsed = raster_start.elapsed();
    if std::env::var("UNPRINT_DEBUG_MEM").is_ok() {
    }

    // ── 2b. Extract source image data for pass-through ───────────────
    let source_images = if input.extension().and_then(|e| e.to_str()) == Some("pdf") {
        pdf_out::extract_source_images(input)
    } else {
        Vec::new()
    };

    // ── 3. Process each page ─────────────────────────────────────────
    let mut all_pages: Vec<pdf_out::PageContent> = Vec::new();
    let mut audit_text: Vec<AuditEntry> = Vec::new();
    let mut audit_geo: Vec<GeometryEntry> = Vec::new();
    let mut page_summaries: Vec<PageSummary> = Vec::new();

    let mut _stat_lines_vectorized = 0u32;
    let mut _stat_lines_raster = 0u32;
    let mut _stat_geo_elements = 0u32;
    let mut _stat_raster_frags = 0u32;
    // Dominant font candidate for fast-path SSIM, persists across pages.
    let mut dominant_font_candidate: Option<font_match::FontMatchResult> = None;

    // Load ground truth from vector PDF (--audit or --test).
    let ground_truth: Option<ground_truth::GroundTruth> = if let Some(vpath) = args.gt_vector_pdf() {
        match ground_truth::GroundTruth::load(vpath) {
            Ok(mut gt) => {
                gt.canonicalize_names(font_registry.entries());
                Some(gt)
            },
            Err(_e) => {
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


        let prepared = page_cache::prepare_page(page_img, page_idx, args.dpi, cache_dir.as_deref())?;
        let mut lines = prepared.lines;
        let gray_page = prepared.gray;
        let bg_color = prepared.bg_color;
        let ink_thresh = prepared.ink_thresh;

        let mut placed_texts: Vec<pdf_out::PlacedText> = Vec::new();
        let mut pg_vec = 0u32;
        let mut pg_raster = 0u32;

        // ── Pass 1: Match all lines ──────────────────────────────────
        let fontmatch_start = std::time::Instant::now();
        let (mut line_matches, fp_hits) = font_pipeline::match_lines(
            &lines, &gray_page, page_img, page_num,
            &font_registry, &font_cache, classifier,
            &glyph_map,
            ground_truth.as_ref(),
            dominant_font_candidate.as_ref(),
            args,
            None,
        );

        // Update dominant font candidate for next page
        if let Some(new_dom) = font_pipeline::update_dominant_font(&line_matches) {
            dominant_font_candidate = Some(new_dom);
        }

        let scored_lines = lines.len() as u64 - fp_hits;
        let _fontmatch_elapsed = fontmatch_start.elapsed();
        if fp_hits > 0 {
        }
        if scored_lines > 0 {
        }

        // ── Pass 1.5: Paragraph-level font grouping ─────────────────
        font_pipeline::paragraph_font_grouping(&lines, &line_matches);

        // ── Word split: split wide whitespace using matched fonts ──
        let _t_split = std::time::Instant::now();
        let split_indices;
        {
            let line_fonts: Vec<Option<std::sync::Arc<Vec<u8>>>> = line_matches.iter()
                .map(|lm| {
                    lm.font_result.as_ref().and_then(|fm| {
                        font_cache.load(&fm.font_path).ok()
                    })
                })
                .collect();
            split_indices = ocr::split_wide_whitespace_words(&mut lines, &gray_page, ink_thresh, &line_fonts);
        }

        // ── Pass 1b: Re-score only lines whose words were split ─────
        if !split_indices.is_empty() {
            let split_set: std::collections::HashSet<usize> = split_indices.iter().copied().collect();
            let (mut new_matches, _) = font_pipeline::match_lines(
                &lines, &gray_page, page_img, page_num,
                &font_registry, &font_cache, classifier,
                &glyph_map,
                ground_truth.as_ref(),
                dominant_font_candidate.as_ref(),
                args,
                Some(&split_set),
            );
            for li in split_indices {
                std::mem::swap(&mut line_matches[li], &mut new_matches[li]);
                // After swap, new_matches[li] holds pass 1's stale LineMatch.
                // Remove its diag dir so the report finds pass 2's dir.
                if let Some(old_dir) = new_matches[li].diag_seg_dir.as_ref() {
                    let _ = std::fs::remove_dir_all(old_dir);
                }
            }
        }

        // ── Pass 2a: Parallel similarity (ZNCC) verification ────────
        let _verify_start = std::time::Instant::now();
        let similarity_results: Vec<(Option<f32>, Option<bool>)> = lines.par_iter()
            .enumerate()
            .map(|(li, line)| {
                let lm = &line_matches[li];
                let font_result = &lm.font_result;

                let ocr_ok = line.confidence >= args.min_ocr_confidence as f32
                    && !line.text.trim().is_empty();
                let font_ok = font_result.is_some();

                if !ocr_ok || !font_ok {
                    return (None, None);
                }

                if let Some(ref fm) = font_result {
                    let font_data = font_cache.load(&fm.font_path).ok();
                    if let Some(ref fd) = font_data {
                        let norm_crop = {
                            let iw = gray_page.width();
                            let ih = gray_page.height();
                            let cx = line.x.min(iw.saturating_sub(1));
                            let cy = line.y.min(ih.saturating_sub(1));
                            let cw = line.width.min(iw - cx);
                            let ch = line.height.min(ih - cy);
                            let raw = image::imageops::crop_imm(&gray_page, cx, cy, cw, ch).to_image();
                            features::contrast_normalize_char(&raw)
                        };
                        let vr = verify::verify_text_region(
                            &norm_crop,
                            fd.as_slice(),
                            &line.text,
                            &line.words,
                            line.x, line.y,
                            fm.glyph_overrides.as_deref(),
                            &fm.variant_tag,
                            fm.variations.as_deref(),
                            lm.diag_seg_dir.as_deref(),
                            None,
                        );
                        let pass = vr.score >= MIN_VERIFY_SIMILARITY;
                        (Some(vr.score), Some(pass))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            })
            .collect();
        let _verify_count = similarity_results.iter().filter(|(s, _)| s.is_some()).count() as u32;

        // ── Pass 2b: Decision matrix + output ────────────────────────
        for (li, line) in lines.iter().enumerate() {
            let line_num = li + 1; // 1-indexed for output
            let lm = &line_matches[li];
            let text_color = lm.text_color;
            let font_result = &lm.font_result;

            // ── Decision matrix ──────────────────────────────────────
            let ocr_ok = line.confidence >= args.min_ocr_confidence as f32
                && !line.text.trim().is_empty();
            let font_ok = font_result.is_some();

            let (keep_raster, reason) = if !ocr_ok {
                (true, format!("OCR confidence too low ({:.0}%)", line.confidence))
            } else if !font_ok {
                (true, "No font match. Kept as raster.".into())
            } else {
                (false, "Vectorised".into())
            };

            let (similarity_score, similarity_pass) = similarity_results[li];

            // ── Logging ──────────────────────────────────────────────
            if keep_raster {
                pg_raster += 1;
            } else {
                pg_vec += 1;
                let _fname = font_result.as_ref().map(|f| f.font_name.as_str()).unwrap_or("?");
                let _fscore = font_result.as_ref().map(|f| f.score).unwrap_or(0.0);
                let _sim_part = similarity_score
                    .map(|s| format!(" sim={s:.3}"))
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
                    "similarity_score": similarity_score,
                    "font_scores_top_5": lm.font_scores.iter().take(5)
                        .map(|(gid, s)| serde_json::json!({"glyph_id": gid, "score": s}))
                        .collect::<Vec<_>>(),
                    "decision": if keep_raster { "raster" } else { "vectorized" },
                });
                let _ = std::fs::write(
                    ddir.join("line_summary.json"),
                    serde_json::to_string_pretty(&line_summary).unwrap_or_default(),
                );
            }

            // ── Audit entry ──────────────────────────────────────────
            audit_text.push(build_audit_entry(
                lm, line, page_num, line_num, similarity_score, similarity_pass, keep_raster, &reason,
            ));

            placed_texts.push(pdf_out::PlacedText {
                text: line.text.clone(),
                x: line.x as f32,
                y: line.y as f32,
                width: line.width as f32,
                height: line.height as f32,
                font_size_pt: font_pipeline::compute_font_size_pt(
                    font_result, line.height, args.dpi, &font_cache,
                ),
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

        _stat_lines_vectorized += pg_vec;
        _stat_lines_raster += pg_raster;

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
            _stat_geo_elements += count as u32;
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
            pdf_out::extract_raster_fragments(
                &cleaned_img,
                args.dpi,
                page_img.height(),
            )
        };

        _stat_raster_frags += raster_fragments.len() as u32;

        if !raster_fragments.is_empty() {
            let is_passthrough = raster_fragments.iter().any(|f| f.passthrough.is_some());
            if !is_passthrough {
                let _frag_bytes: usize = raster_fragments.iter().map(|f| f.raw_rgb.len()).sum();
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
            if let Err(_e) = compare::generate_comparison(
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
        dpi: args.dpi,
        classifier: args.classifier.clone(),
        render_scale: args.render_scale,
        render_aa: args.render_aa.clone(),
        render_binarize: args.render_binarize,
        elapsed_secs: 0.0,  // updated before writing
        pages: page_summaries,
        text_entries: audit_text,
        geometry_entries: audit_geo,
    };
    // Enrich entries with ground-truth classification before writing JSON.
    report::enrich_audit_entries(
        &mut audit.text_entries,
        ground_truth.as_ref(),
        args.dpi,
        font_registry.entries(),
        &glyph_map,
    );

    if test_mode {
        // ── Test mode: compute accuracy, print JSON to stdout ────────
        let acc = report::compute_accuracy(
            &audit.text_entries,
            ground_truth.as_ref(),
            args.dpi,
            font_registry.entries(),
            &glyph_map,
        );
        // JSON output to stdout
        let test_json = serde_json::json!({
            "primary_hits": acc.primary_hits,
            "compared": acc.compared,
            "pct": (acc.pct * 10.0).round() / 10.0,
            "major_misses": acc.major_misses,
            "minor_misses": acc.minor_misses,
            "similarity_failures": acc.similarity_failures,
            "hits": acc.hits,
            "kept_raster": acc.kept_raster,
        });
        println!("{}", serde_json::to_string_pretty(&test_json).unwrap());

        // If --audit is also set, write audit artifacts + HTML report
        if let Some(ref audit_root) = args.audit {
            let audit_path = args.audit_log_path();
            audit.elapsed_secs = run_start.elapsed().as_secs_f64();
            if let Err(_e) = audit.write_to_file(&audit_path) {
            } else {
            }
            write_audit_report(
                audit_root, &audit.text_entries, ground_truth.as_ref(),
                args.dpi, font_registry.entries(), &glyph_map, args, run_start.elapsed(),
            );
        }
        return Ok(());
    }

    let audit_path = args.audit_log_path();

    if output.to_str() != Some("/dev/null") || args.audit.is_some() {
        audit.elapsed_secs = run_start.elapsed().as_secs_f64();
        if let Err(_e) = audit.write_to_file(&audit_path) {
        } else {
        }
    }

    // ── 5b. HTML miss report ─────────────────────────────────────────
    if let Some(ref audit_root) = args.audit {
        write_audit_report(
            audit_root, &audit.text_entries, ground_truth.as_ref(),
            args.dpi, font_registry.entries(), &glyph_map, args, run_start.elapsed(),
        );
    }

    // ── 6. Report ────────────────────────────────────────────────────
    let (_cache_hits, _cache_misses) = font_cache.stats();

    Ok(())
}

// ---------------------------------------------------------------------------
// Page loading
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Raster fragment extraction (lossless)
// ---------------------------------------------------------------------------


