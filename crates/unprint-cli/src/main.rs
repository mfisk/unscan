#[allow(dead_code)]

use unprint_report as report;
use unprint::audit;
use unprint::classifier;
use unprint::cli;
use unprint::color;
use unprint::font_cache;
use unprint::font_match;
use unprint::font_scan;
use unprint::geometry;
use unprint::features;
use unprint::glyph_map;
use unprint::char_render;
use unprint::train;
use unprint::compare;
use unprint::ocr;
use unprint::page_cache;
use unprint::pdf_out;
use unprint::smooth;
use unprint::verify;
use unprint::ground_truth;
use unprint::font_pipeline;
use unprint::vintage_cache;

use unprint::font_pipeline::ObsRankProbs;
use unprint::audit::{AuditEntry, AuditLog, BBox, GeometryEntry, PageSummary};

use unprint::error::ScanTextError;
use rayon::prelude::*;

/// Minimum SSIM score for SSIM verification to consider a font match acceptable.
const MIN_VERIFY_SIMILARITY: f32 = 0.9;

fn main() {
    // ── pprof flamegraph (cargo build --features profile) ────────
    #[cfg(feature = "profile")]
    let pprof_guard = pprof::ProfilerGuardBuilder::default()
        .frequency(997)
        .blocklist(&["libc", "libgcc", "libpthread", "vdso"])
        .build()
        .expect("pprof guard");
    let args = cli::parse();
    if let Err(msg) = args.validate() {
        eprintln!("Error: {msg}");
        std::process::exit(1);
    }

    // Initialize cache directory and font allowlist from CLI/env
    unprint::cache::init_cache_dir(args.cache_dir.as_deref());
    unprint::cache::init_allowlist(args.font_allowlist.as_deref());

    // Warn if allowlist used with default cache (safe but slower than alt cache)
    if let Some(ref allowlist_str) = args.font_allowlist {
        if unprint::cache::is_default_cache_dir() {
            eprintln!("Note: --font-allowlist used with default cache dir; filtering at matching time only (main cache untouched). For faster iteration, use --cache-dir /tmp/unprint-6fonts with --font-allowlist.");
        } else {
            let count = allowlist_str.split(',').filter(|s| !s.trim().is_empty()).count();
            eprintln!("Using alternate cache dir {:?} with font allowlist ({} entries)", unprint::cache::cache_dir(), count);
        }
    }

    // train-lda removed — training happens automatically when weights are stale.

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

    // ── Write pprof profile (cargo build --features profile) ────────
    #[cfg(feature = "profile")]
    {
        if let Ok(report) = pprof_guard.report().build() {
            use std::collections::HashMap;
            // Inclusive and exclusive sample counts per symbol
            let mut inclusive: HashMap<String, isize> = HashMap::new();
            let mut exclusive: HashMap<String, isize> = HashMap::new();
            let mut total: isize = 0;
            for (frames, count) in &report.data {
                total += *count;
                // leaf = first frame (pprof stores leaf-first)
                if let Some(leaf_frame) = frames.frames.first() {
                    for sym in leaf_frame {
                        *exclusive.entry(sym.name()).or_insert(0) += *count;
                    }
                }
                // inclusive: every frame in the stack
                // Use a set per sample to avoid double-counting same symbol appearing twice in one stack
                let mut seen_in_sample = std::collections::HashSet::new();
                for frame in &frames.frames {
                    for sym in frame {
                        let name = sym.name();
                        if seen_in_sample.insert(name.clone()) {
                            *inclusive.entry(name).or_insert(0) += *count;
                        }
                    }
                }
            }
            // Top 30 by inclusive
            let mut top_inc: Vec<(String, isize)> = inclusive.into_iter().collect();
            top_inc.sort_by(|a,b| b.1.cmp(&a.1));
            let mut top_exc: Vec<(String, isize)> = exclusive.into_iter().collect();
            top_exc.sort_by(|a,b| b.1.cmp(&a.1));

            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            let pid = std::process::id();
            let txt_path = "/tmp/pprof-top.txt";
            let txt_ts_path = format!("/tmp/unprint-pprof-top-{}-{}.txt", ts, pid);
            let write_top = |path: &str| -> std::io::Result<()> {
                use std::io::Write;
                let mut f = std::fs::File::create(path)?;
                writeln!(f, "pprof 997Hz total_samples={} pid={} ts={}", total, pid, ts)?;
                writeln!(f, "=== Top 30 Inclusive (time spent in func + callees) ===")?;
                for (i, (name, cnt)) in top_inc.iter().take(30).enumerate() {
                    let pct = if total>0 { *cnt as f64 * 100.0 / total as f64 } else { 0.0 };
                    writeln!(f, "{:2}. {:6} {:5.1}% {}", i+1, cnt, pct, name)?;
                }
                writeln!(f, "\n=== Top 30 Exclusive (time spent in func itself, leaf) ===")?;
                for (i, (name, cnt)) in top_exc.iter().take(30).enumerate() {
                    let pct = if total>0 { *cnt as f64 * 100.0 / total as f64 } else { 0.0 };
                    writeln!(f, "{:2}. {:6} {:5.1}% {}", i+1, cnt, pct, name)?;
                }
                Ok(())
            };
            let _ = write_top(txt_path);
            let _ = write_top(&txt_ts_path);

            // Keep flamegraph only if env says to, otherwise skip to save time
            if std::env::var("UNPRINT_FLAMEGRAPH").ok().as_deref() == Some("1") {
                let fg_path = std::path::Path::new("/tmp/lob-flamegraph.svg");
                let fg_ts_path = format!("/tmp/unprint-flamegraph-{}-{}.svg", ts, pid);
                if let Ok(file) = std::fs::File::create(fg_path) {
                    let _ = report.flamegraph(file);
                }
                if let Ok(file) = std::fs::File::create(&fg_ts_path) {
                    let _ = report.flamegraph(file);
                }
                if !args.quiet { eprintln!("[profile] Wrote flamegraph and top to {} {}", fg_path.display(), txt_path); }
            } else {
                if !args.quiet { eprintln!("[profile] Wrote pprof top to {} and {}", txt_path, txt_ts_path); }
            }
        }
    }
}


/// Scan available fonts and return a FontRegistry.
/// Writes catalog.bin so classifier loaders can validate catalog_hash.
/// Centroids are already baked into classifier .bin files from training,
/// so no runtime render+embed step is needed.
fn load_fonts(args: &cli::Args, _classifier: &mut dyn classifier::Classifier) -> Result<font_scan::FontRegistry, ScanTextError> {
    // ── Vintage font cache generation (lightweight, BEFORE heavy scan) ──
    // To avoid OOM on full rescan (685 files -> 2911 entries -> incremental 342k glyph renders),
    // we generate vintage ONLY from msttcorefonts dir via uncached scan (53 files -> 30 deduped).
    // This does NOT touch ~/.cache/unprint/font_scan.bin and stays <200MB RSS.
    let vintage_eras = vintage_cache::DEFAULT_ERAS;
    let vintage_paths = {
        let mstt_dir = std::path::PathBuf::from("/usr/share/fonts/truetype/msttcorefonts");
        let mstt_bases = if mstt_dir.exists() {
            // scan_fonts_uncached is crate-private but same crate root — lightweight, no cache write
            font_scan::scan_fonts_uncached(&[mstt_dir])
        } else {
            Vec::new()
        };
        vintage_cache::ensure_vintage_fonts(&mstt_bases, vintage_eras)
    };

    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    let base_entries = font_scan::scan_fonts(&font_dirs, args.quiet);
    if base_entries.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    // Load vintage entries via uncached scan of the returned paths (avoids touching global font_scan.bin)
    // We parse each vintage file to get a FontEntry, then tag it with era as variant so font_key is unique.
    let alias_table = font_scan::build_alias_table();
    let mut vintage_entries = Vec::with_capacity(vintage_paths.len());
    for vp in &vintage_paths {
        // Determine era from filename (format: {fam}-{era}-{hash}.ttf)
        let fname = vp.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let mut found_era = None;
        for &era in vintage_eras {
            if fname.contains(era.name()) {
                found_era = Some(era);
                break;
            }
        }
        // Fallback: try all eras if not in DEFAULT_ERAS (in case caller passes ALL_ERAS later)
        if found_era.is_none() {
            for &era in vintage_cache::ALL_ERAS {
                if fname.contains(era.name()) {
                    found_era = Some(era);
                    break;
                }
            }
        }
        let era = found_era.unwrap_or(vintage_cache::Era::PostScript);

        if let Some(mut fe) = font_scan::load_font_entry(vp, &alias_table) {
            // New font_key logic: keep postscript_name as base, store era in vintage_era field.
            // font_key = PSName | variant_tag | vintage=ERA — no need to mangle PS name.
            let era_tag = era.name().to_string();
            fe.vintage_era = Some(era_tag.clone());
            // Keep variant_tag as-is (preserves wght for variable instances). If base had no variant,
            // leave it empty — vintage distinction comes from vintage_era, not variant_tag.
            // Family name tagged for human readability only; canonical matching uses substring.
            if !fe.family_name.to_lowercase().contains(&era_tag) {
                fe.family_name = format!("{} [{}]", fe.family_name, era.name());
            }
            fe.data = Vec::new(); // drop bytes, will be loaded via FontCache on demand
            // Recompute cache to include |vintage=ERA|var so font_key is unique and sorted deterministically
            fe.recompute_font_key_cache();
            vintage_entries.push(fe);
        }
    }

    if !vintage_entries.is_empty() {
        eprintln!("[vintage] loaded {} vintage variants into registry ({} total with base)", vintage_entries.len(), base_entries.len() + vintage_entries.len());
    }

    let mut all_entries = base_entries;
    all_entries.extend(vintage_entries);

    // Deduplicate by font_key to eliminate duplicate vintage files created from
    // duplicate msttcorefonts paths (Arial.TTF vs Arial.ttf). Without this,
    // FontRegistry contains duplicate keys, owned_fonts HashMap overwrites them,
    // and geo-cache reports Wrote N < total.
    {
        use std::collections::HashSet;
        let mut seen: HashSet<String> = HashSet::with_capacity(all_entries.len());
        let before = all_entries.len();
        all_entries.retain(|e| seen.insert(e.font_key()));
        let after = all_entries.len();
        if before != after {
            eprintln!("[vintage] deduped all_entries {} -> {} (removed {} duplicate keys)", before, after, before - after);
        }
    }

    let registry = font_scan::FontRegistry::new(all_entries);

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
    classifier: &dyn classifier::Classifier,
) -> AuditEntry {
    use audit::{BBox, FontCandidate, ObservationVote, Decision, WordBBox};
    let font_result = &lm.font_result;
    let obs_vote = |d: &font_match::ObservationDetail, rp: &ObsRankProbs| {
        let seq = [d.ch];
        let n_glyphs = classifier.glyph_count(&seq).max(1) as f32;
        let ch_h_ll = rp.chosen_geo_h_ll.get(&d.crop_index).copied();
        let ch_v_ll = rp.chosen_geo_v_ll.get(&d.crop_index).copied();
        let gt_h_ll = rp.gt_geo_h_ll.get(&d.crop_index).copied();
        let gt_v_ll = rp.gt_geo_v_ll.get(&d.crop_index).copied();
        ObservationVote {
            seq: vec![d.ch],
            weight: d.weight,
            crop_index: d.crop_index,
            best_prob: d.best_prob * n_glyphs,
            passed_gate: d.passed_gate,
            nearest: d.nearest.clone(),
            crop_path: None,
            chosen_rank: rp.chosen_ranks.get(&d.crop_index).copied(),
            ocr_corrected_from: d.ocr_corrected_from,
            best_alt_char: d.best_alt_char,
            best_alt_dist: d.best_alt_dist,
            pflda_top_char: d.pflda_top_char,
            pflda_top_p: d.pflda_top_p,
            pflda_ocr_p: d.pflda_ocr_p,
            pflda_replaced: d.pflda_replaced,
            gt_font_rank: rp.gt_ranks.get(&d.crop_index).copied(),
            chosen_prob: rp.chosen_probs.get(&d.crop_index).copied().map(|p| p * n_glyphs),
            gt_font_prob: rp.gt_probs.get(&d.crop_index).copied().map(|p| p * n_glyphs),
            obs_stats: d.obs_stats.clone(),
            chosen_geo_h_ll: ch_h_ll,
            chosen_geo_v_ll: ch_v_ll,
            chosen_glyph_score: rp.chosen_glyph_scores.get(&d.crop_index).copied(),
            gt_glyph_score: rp.gt_glyph_scores.get(&d.crop_index).copied(),
            chosen_geo_h_err: rp.chosen_geo_h_err.get(&d.crop_index).copied(),
            chosen_geo_v_err: rp.chosen_geo_v_err.get(&d.crop_index).copied(),
            gt_geo_h_ll: gt_h_ll,
            gt_geo_v_ll: gt_v_ll,
            gt_geo_h_err: rp.gt_geo_h_err.get(&d.crop_index).copied(),
            gt_geo_v_err: rp.gt_geo_v_err.get(&d.crop_index).copied(),
            chosen_geo_ll: match (ch_h_ll, ch_v_ll) { (Some(h), Some(v)) => Some(h+v), (Some(h), None) => Some(h), (None, Some(v)) => Some(v), _ => None },
            gt_geo_ll: match (gt_h_ll, gt_v_ll) { (Some(h), Some(v)) => Some(h+v), (Some(h), None) => Some(h), (None, Some(v)) => Some(v), _ => None },
        }
    };
    // Use corrected text if OCR corrections were applied
    let display_text = if let Some(ref cw) = lm.corrected_words {
        cw.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    } else {
        line.text.clone()
    };
    AuditEntry {
        page: page_num,
        line_index: line_num,
        text: display_text,
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
        obs_votes: lm.observations.iter().map(|d| obs_vote(d, &lm.obs_rank_probs)).collect(),
        font_candidates_lig: lm.font_scores_lig.iter()
            .map(|(fk, s)| FontCandidate { font_key: fk.clone(), score: *s })
            .collect(),
        obs_votes_lig: lm.observations_lig.iter().map(|d| obs_vote(d, &lm.alt_obs_rank_probs)).collect(),
        seg_winner: lm.seg_winner.clone(),
        word_bboxes: lm.corrected_words.as_deref().unwrap_or(&line.words).iter().map(|w| WordBBox {
            text: w.text.clone(), x: w.x, y: w.y, width: w.width, height: w.height, confidence: w.confidence,
        }).collect(),
        word_bboxes_raw: line.raw_words.iter().map(|w| WordBBox {
            text: w.text.clone(), x: w.x, y: w.y, width: w.width, height: w.height, confidence: w.confidence,
        }).collect(),
        tie_candidates: lm.tie_candidates.clone(),
        miss_type: None,
        expected_font: None,
        gt_text: None,
        ocr_text: None,
        ocr_correct: None,
        midpoint_em_px: lm.midpoint_em_px,
        gt_midpoint_em_px: lm.gt_midpoint_em_px,
        fast_path: lm.fast_path,
        word_segmentation: lm.word_seg_summaries.clone(),
        gt_word_segmentation: lm.gt_word_seg_summaries.clone(),
        ocr_corrections: lm.ocr_corrections.clone(),
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
        report_all: args.report_all || args.audit_all,
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
    let mut font_registry = load_fonts(args, classifier)?;
    if !args.quiet { eprintln!("[timing] load_fonts {:.2}s", run_start.elapsed().as_secs_f64()); }
    if font_registry.is_empty() {
        return Err(ScanTextError::NoFonts);
    }

    // Load glyph map (glyph dedup groups).  If missing or stale, retrain
    // the LDA classifier (which rebuilds the glyph map as a side effect).
    let gmap_path = glyph_map::NgramGlyphMap::default_path();
    let t_gmap = std::time::Instant::now();
    let mut glyph_map = match glyph_map::NgramGlyphMap::load(&gmap_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Glyph map at {} stale or missing ({e}), retraining...", gmap_path.display());
            let font_dirs = font_scan::default_font_dirs(&args.font_dir);
            let lda_path = classifier::default_lda_weights_path();
            train::run_train(train::TrainArgs {
                output: lda_path,
                font_dir: font_dirs,
                render_params: args.render_params(),
                lda: true,
                ..train::TrainArgs::default()
            });
            glyph_map::NgramGlyphMap::load(&gmap_path)
                .unwrap_or_else(|e2| {
                    eprintln!("Error: retraining did not produce a valid glyph map ({e2})");
                    std::process::exit(1);
                })
        }
    };
    if !args.quiet { eprintln!("[timing] glyph_map load {:.2}s (total {:.2}s)", t_gmap.elapsed().as_secs_f64(), run_start.elapsed().as_secs_f64()); }

    // ── Incremental update: detect new fonts and add to index ──────
    // Compare cached font_meta (glyph_map) against installed fonts.
    // New fonts are indexed and added; removed fonts are kept.
    {
        // Fast path: if catalog hash matches, no new fonts (common warm case)
        // Use mtime+size for fast check (per Mike) - catalog_hash is hash of sorted font_keys
        if font_registry.catalog_hash() == glyph_map.catalog_hash {
            if !args.quiet { eprintln!("[incremental] No new fonts (catalog hash match)"); }
        } else {
        use std::collections::HashSet;
        let installed_keys: HashSet<String> = font_registry.entries().iter()
            .map(|e| e.font_key())
            .collect();
        let t_cached_keys = std::time::Instant::now();
        let cached_keys = glyph_map.cached_font_keys();
        if !args.quiet { eprintln!("[timing] incremental cached_keys {:.2}s", t_cached_keys.elapsed().as_secs_f64()); }

        if !args.quiet { eprintln!("[incremental] installed={} cached={} (glyph_map has {} unique fonts)", installed_keys.len(), cached_keys.len(), cached_keys.len()); }

        let t_diff = std::time::Instant::now();
        let mut new_keys: Vec<String> = installed_keys.difference(&cached_keys).cloned().collect();
        let removed_keys: Vec<String> = cached_keys.difference(&installed_keys).cloned().collect();
        if !args.quiet { eprintln!("[timing] incremental diff {:.2}s (new={} removed={})", t_diff.elapsed().as_secs_f64(), new_keys.len(), removed_keys.len()); }

        if !removed_keys.is_empty() {
            if !args.quiet { eprintln!("[incremental] Keeping {} removed fonts (trained data still valid for classification)", removed_keys.len()); }
        }

        // Filter out non-Latin fonts that cannot render any supported_chars
        // (NotoSansBamum, NotoSansGothic, NotoSerifLao etc. = 5898-5660 = 238).
        // These have zero Latin coverage and would never register in glyph_map,
        // causing the same 238 to be redetected every run.
        // Cache known non-Latin keys to avoid re-reading 238 font files every run (was 3s).
        let t_nonlatin = std::time::Instant::now();
        {
            let cache_path = unprint::cache::cache_dir().join("non-latin-cache.json");
            let mut known_non_latin: std::collections::HashSet<String> = if let Ok(data) = std::fs::read_to_string(&cache_path) {
                serde_json::from_str(&data).unwrap_or_default()
            } else {
                std::collections::HashSet::new()
            };
            let mut updated_non_latin = false;
            let supported = features::supported_chars();
            let mut filtered = Vec::with_capacity(new_keys.len());
            let mut skipped = 0usize;
            for font_key in &new_keys {
                // Fast path: known non-Latin from cache
                if known_non_latin.contains(font_key) {
                    skipped += 1;
                    continue;
                }
                let Some(fe) = font_registry.by_key(font_key) else { continue };
                let Ok(font_data) = std::fs::read(&fe.path) else { continue };
                let Ok(face) = unprint_fonts::ttf_parser::Face::parse(&font_data, 0) else { continue };
                // Require at least one Latin letter (a-z) to avoid NotoColorEmoji (which has only digits 0-9)
                // NotoColorEmoji has 0-9 but no letters -> would pass 'any supported char' and then hang indexing 11M CBDT
                let has_latin = ('a'..='z').any(|c| face.glyph_index(c).map_or(false, |g| g.0 != 0));
                if has_latin {
                    filtered.push(font_key.clone());
                } else {
                    // Check fallback via overrides (rare)
                    let has_override = fe.glyph_overrides.as_deref()
                        .map(|ovs| ovs.iter().any(|(ch,_)| supported.contains(ch)))
                        .unwrap_or(false);
                    if has_override {
                        filtered.push(font_key.clone());
                    } else {
                        // Remember this as known non-Latin for next run
                        known_non_latin.insert(font_key.clone());
                        updated_non_latin = true;
                        skipped += 1;
                    }
                }
            }
            if updated_non_latin {
                let _ = std::fs::create_dir_all(cache_path.parent().unwrap());
                let _ = std::fs::write(&cache_path, serde_json::to_string(&known_non_latin).unwrap_or_default());
            }
            if skipped > 0 {
                if !args.quiet { eprintln!("[incremental] Skipping {} non-Latin fonts (no supported_chars coverage)", skipped); }
            }
            new_keys = filtered;
        }
        if !args.quiet { eprintln!("[timing] incremental non-latin filter {:.2}s (skipped={})", t_nonlatin.elapsed().as_secs_f64(), 238); }

        if !new_keys.is_empty() {
            if !args.quiet { eprintln!("[incremental] Detected {} new fonts, indexing and adding to classifier...", new_keys.len()); }
            for k in &new_keys {
                eprintln!("  + {k}");
            }

            // Build sequence list (single-char only, bigrams disabled)
            let sequences: Vec<Vec<char>> = features::supported_chars().iter().map(|&c| vec![c]).collect();

            // Map font_key -> FontEntry for quick lookup
            let font_by_key: std::collections::HashMap<String, &font_scan::FontEntry> = font_registry.entries().iter()
                .map(|e| (e.font_key(), e))
                .collect();

            let render_params = args.render_params();
            let mut new_glyphs_added = 0usize;
            let mut fonts_indexed = 0usize;

            // Ensure classifier is in owned mode for mutation
            classifier.ensure_owned();

            for font_key in &new_keys {
                let fe = match font_by_key.get(font_key) {
                    Some(f) => *f,
                    None => continue,
                };

                // Load font data
                let font_data = match std::fs::read(&fe.path) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let font = match unprint_fonts::ab_glyph::FontRef::try_from_slice(&font_data) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

                let overrides = fe.glyph_overrides.as_deref();
                fonts_indexed += 1;

                for seq in &sequences {
                    let gid_overrides: Vec<Option<unprint_fonts::ab_glyph::GlyphId>> = seq.iter().map(|c| {
                        overrides.and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| unprint_fonts::ab_glyph::GlyphId(*g)))
                    }).collect();

                    let img = match char_render::render_ngram_fresh(&font, seq, &gid_overrides, &render_params) {
                        Some(img) => img,
                        None => continue,
                    };

                    let hash = glyph_map::hash_image(&img);
                    let (glyph_id, is_new_group) = glyph_map.register(seq, font_key, hash);

                    if is_new_group {
                        // New unique glyph — add centroid to classifier
                        if let Some(feats) = features::compute_features(&img, false) {
                            classifier.add_glyph(glyph_id, seq, &feats);
                            new_glyphs_added += 1;
                        }
                    }
                }
            }

            // Update catalog hashes to new installed hash (union semantics: keep removed fonts' data, but hash reflects installed)
            let new_catalog_hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = rustc_hash::FxHasher::default();
                let mut sorted_keys: Vec<&String> = installed_keys.iter().collect();
                sorted_keys.sort();
                for k in sorted_keys {
                    k.hash(&mut hasher);
                }
                hasher.finish()
            };
            glyph_map.set_catalog_hash(new_catalog_hash);
            classifier.set_catalog_hash(new_catalog_hash);
            classifier.recompute_stats();

            // Persist updated glyph_map (dirty flag will trigger write on Drop, but also write explicitly)
            if let Err(e) = glyph_map.write_bin(&gmap_path) {
                eprintln!("warning: failed to write updated glyph map: {e}");
            }

            // Persist updated classifier
            let clf_name = classifier.name().to_string();
            let save_result = match clf_name.as_str() {
                "lda" => classifier.save_to(&classifier::default_lda_weights_path(), b"LDAC", 8),
                "perchar-fisher" | "fisher" => classifier.save_to(&classifier::default_fisher_weights_path(), b"FISH", 3),
                "mahalanobis" => classifier.save_to(&classifier::default_mahalanobis_weights_path(), b"MAHA", 3),
                "triplet" => classifier.save_to(&classifier::default_triplet_weights_path(), b"TRIP", 3),
                "fusion" => {
                    // Fusion saves each child via its trait impl
                    classifier.save_to(&std::path::PathBuf::from("/tmp/fusion-dummy.bin"), b"LDAC", 8)
                }
                _ => Ok(()),
            };
            if let Err(e) = save_result {
                eprintln!("warning: failed to persist updated classifier ({clf_name}): {e}");
            }

            // Patch catalog.bin hash to new installed hash so next run's fast path succeeds
            // and we don't trigger an infinite retrain loop (registry hash == installed hash with FxHasher).
            // Also update in-memory registry hash for current process consistency.
            font_registry.set_catalog_hash(new_catalog_hash);
            let catalog_path = classifier::default_catalog_path();
            if let Err(e) = font_registry.write_fonts_bin(&catalog_path) {
                eprintln!("warning: failed to rewrite catalog with new hash: {e}");
                // Fallback: patch hash bytes in-place
                if let Ok(mut f) = std::fs::OpenOptions::new().read(true).write(true).open(&catalog_path) {
                    use std::io::{Seek, Write};
                    let _ = f.seek(std::io::SeekFrom::Start(8));
                    let _ = f.write_all(&new_catalog_hash.to_le_bytes());
                }
            }

            if !args.quiet { eprintln!("[incremental] Indexed {fonts_indexed} new fonts, {new_glyphs_added} new glyph groups created"); }
        }
        } // end else (catalog hash mismatch)
    }
    if !args.quiet { eprintln!("[timing] incremental {:.2}s", run_start.elapsed().as_secs_f64()); }

    // All font access goes through the shared cache below.

    // ── 1c. Build runtime training data for per-font OCR correction ─
    let t_rtd = std::time::Instant::now();
    let rtd = train::RuntimeTrainingData::from_registry(
        &font_registry, &glyph_map, &args.render_params(),
    );
    if !args.quiet { eprintln!("[timing] rtd from_registry {:.2}s (total {:.2}s) -> {}", t_rtd.elapsed().as_secs_f64(), run_start.elapsed().as_secs_f64(), if rtd.is_some() { "Some" } else { "None" }); }

    // ── 1b''. Shared font cache for all post-index font access ──────
    let font_cache = font_cache::FontCache::new(font_cache::DEFAULT_CAPACITY);
    if !args.quiet { eprintln!("[timing] font_cache new {:.2}s", run_start.elapsed().as_secs_f64()); }

    // ── 1c''. Geometry cache for GPOS kerning + ligature-aware positioning ──
    let t_geo = std::time::Instant::now();
    let geo_cache = unprint::geo_cache::GeometryCache::load_or_build(&font_registry, &font_cache, args.quiet);
    if !args.quiet { eprintln!("[timing] geo_cache load_or_build {:.2}s (total {:.2}s)", t_geo.elapsed().as_secs_f64(), run_start.elapsed().as_secs_f64()); }

    // ── 2. Load input pages (with raster cache) ──────────────────────
    let cache_dir = page_cache::cache_key(input, args.dpi)
        .and_then(|key| page_cache::cache_dir(&key));

    let t_pages = std::time::Instant::now();
    let (pages, _raster_cached) = page_cache::get_pages(input, args.dpi)?;
    if !args.quiet { eprintln!("[timing] get_pages {} pages {:.2}s (total {:.2}s)", pages.len(), t_pages.elapsed().as_secs_f64(), run_start.elapsed().as_secs_f64()); }

    // ── 2b. Extract source image data for pass-through ───────────────
    let t_extract = std::time::Instant::now();
    let source_images = if input.extension().and_then(|e| e.to_str()) == Some("pdf") {
        pdf_out::extract_source_images(input)
    } else {
        Vec::new()
    };
    if !args.quiet { eprintln!("[timing] extract_source_images {:.2}s (total {:.2}s)", t_extract.elapsed().as_secs_f64(), run_start.elapsed().as_secs_f64()); }

    // ── 3. Process each page ─────────────────────────────────────────
    if !args.quiet { eprintln!("[timing] before page loop {:.2}s", run_start.elapsed().as_secs_f64()); }
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
        // Fast RGBA reuse – avoid image 0.25 generic CicpRgb::cast_pixels_by_layout 8.7% leaf.
        // Same fast path as page_cache::fast_rgba8_from_dynamic but inline for bin crate visibility.
        let rgba_page = match page_img {
            image::DynamicImage::ImageRgba8(rgba) => rgba.clone(),
            image::DynamicImage::ImageRgb8(rgb) => {
                let (w, h) = rgb.dimensions();
                let raw = rgb.as_raw();
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for chunk in raw.chunks_exact(3) {
                    out.extend_from_slice(chunk);
                    out.push(255);
                }
                image::RgbaImage::from_raw(w, h, out).expect("rgba size")
            }
            image::DynamicImage::ImageLuma8(luma) => {
                let (w, h) = luma.dimensions();
                let raw = luma.as_raw();
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for &v in raw { out.extend_from_slice(&[v, v, v, 255]); }
                image::RgbaImage::from_raw(w, h, out).expect("rgba size")
            }
            _ => page_img.to_rgba8(),
        };
        let bg_color = prepared.bg_color;
        let ink_thresh = prepared.ink_thresh;

        let mut placed_texts: Vec<pdf_out::PlacedText> = Vec::new();
        let mut pg_vec = 0u32;
        let mut pg_raster = 0u32;

        // ── Pass 1: Match all lines ──────────────────────────────────
        let (mut line_matches, fp_hits) = font_pipeline::match_lines(
            &lines, &gray_page, &rgba_page, page_img, page_num,
            &font_registry, &font_cache, &geo_cache, classifier,
            &glyph_map,
            ground_truth.as_ref(),
            dominant_font_candidate.as_ref(),
            args,
            None,
            rtd.as_ref(),
        );

        // Update dominant font candidate for next page
        if let Some(new_dom) = font_pipeline::update_dominant_font(&line_matches) {
            dominant_font_candidate = Some(new_dom);
        }

        let scored_lines = lines.len() as u64 - fp_hits;
        if fp_hits > 0 {
        }
        if scored_lines > 0 {
        }

        // ── Pass 1.5: Paragraph-level font grouping ─────────────────
        font_pipeline::paragraph_font_grouping(&lines, &line_matches);

        // ── Word split: split wide whitespace using matched fonts ──
        let split_indices: Vec<usize>;
        {
            let line_fonts: Vec<Option<std::sync::Arc<Vec<u8>>>> = line_matches.iter()
                .map(|lm| {
                    lm.font_result.as_ref().and_then(|fm| {
                        font_cache.load(&fm.font_path).ok()
                    })
                })
                .collect();
            split_indices = ocr::split_wide_whitespace_words(&mut lines, &gray_page, ink_thresh, &line_fonts);
            // split_indices = vec![];
        }

        // ── Pass 1b: Re-score only lines whose words were split ─────
        if !split_indices.is_empty() {
            let split_set: std::collections::HashSet<usize> = split_indices.iter().copied().collect();
            let (mut new_matches, _) = font_pipeline::match_lines(
            &lines, &gray_page, &rgba_page, page_img, page_num,
                &font_registry, &font_cache, &geo_cache, classifier,
                &glyph_map,
                ground_truth.as_ref(),
                None,  // No fast path — re-score fully with new word splits
                args,
                Some(&split_set),
                rtd.as_ref(),
            );
            for li in split_indices {
                // new_matches[li] is the fresh Pass 1b result, line_matches[li] is the stale Pass 1 result.
                // If their diag dirs are the same path (slug derived from line.text, which
                // doesn't change after word-split), we must NOT delete it, otherwise we delete
                // the dir we just recreated.
                let new_dir = new_matches[li].diag_seg_dir.clone();
                std::mem::swap(&mut line_matches[li], &mut new_matches[li]);
                if let Some(old_dir) = new_matches[li].diag_seg_dir.as_ref() {
                    if Some(old_dir) != new_dir.as_ref() {
                        if let Err(e) = std::fs::remove_dir_all(old_dir) {
                            eprintln!(
                                "[diag-clean] failed to remove stale {}: {}",
                                old_dir.display(),
                                e
                            );
                        }
                    }
                    // else: same path as new_dir — keep it
                }
            }
        }

        // ── Pass 2a: Parallel similarity (ZNCC) verification ────────
        let similarity_results: Vec<(Option<f32>, Option<bool>)> = lines.par_iter()
            .enumerate()
            .map(|(li, line)| {
                let lm = &line_matches[li];
                let font_result = &lm.font_result;

                // Fast-path lines already verified — use stored score
                if let Some(fps) = lm.fast_path_score {
                    return (Some(fps), Some(fps >= MIN_VERIFY_SIMILARITY));
                }

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
                            features::contrast_normalize_char(raw)
                        };
                        let verify_words = lm.corrected_words.as_deref().unwrap_or(&line.words);
                        let allow_liga = lm.seg_winner.as_deref() == Some("ligature");
                        let vr = verify::verify_text_region(
                            &norm_crop,
                            fd.as_slice(),
                            &line.text,
                            verify_words,
                            line.x, line.y,
                            fm.glyph_overrides.as_deref(),
                            &fm.variant_tag,
                            fm.variations.as_deref(),
                            allow_liga,
                            lm.diag_seg_dir.as_deref(),
                            None,
                            lm.midpoint_em_px,
                        );
                        // Verify alt segmentation path's top font for ZNCC comparison
                        if lm.seg_winner.is_some() && !lm.font_scores_lig.is_empty() {
                            if let Some((ref alt_key, _)) = lm.font_scores_lig.first() {
                                if let Some(alt_fe) = font_registry.by_key(alt_key) {
                                    if let Ok(alt_fd) = font_cache.load(&alt_fe.path) {
                                        let alt_audit_dir = lm.diag_seg_dir.as_ref().map(|d| {
                                            let p = d.join("ssim_alt");
                                            let _ = std::fs::create_dir_all(&p);
                                            p
                                        });
                                        let allow_liga_alt = lm.seg_winner.as_deref() != Some("ligature");
                                        let _alt_vr = verify::verify_text_region(
                                            &norm_crop,
                                            alt_fd.as_slice(),
                                            &line.text,
                                            verify_words,
                                            line.x, line.y,
                                            alt_fe.glyph_overrides.as_deref(),
                                            &alt_fe.variant_tag,
                                            alt_fe.variations.as_deref(),
                                            allow_liga_alt,
                                            alt_audit_dir.as_deref(),
                                            None,
                                            None,
                                        );
                                    }
                                }
                            }
                        }
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
                let diag_text = if let Some(ref cw) = lm.corrected_words {
                    cw.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
                } else {
                    line.text.clone()
                };
                let line_summary = serde_json::json!({
                    "page": page_num,
                    "line_index": line_num,
                    "text": diag_text,
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
                lm, line, page_num, line_num, similarity_score, similarity_pass, keep_raster, &reason, classifier,
            ));

            let render_text = if let Some(ref cw) = lm.corrected_words {
                cw.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
            } else {
                line.text.clone()
            };
            placed_texts.push(pdf_out::PlacedText {
                text: render_text,
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
                words: lm.corrected_words.as_deref().unwrap_or(&line.words).iter().map(|w| pdf_out::WordBox {
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
            let geo = geometry::detect_geometry_from_buffers(&gray_page, &rgba_page, &text_bboxes, min_line_len);

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
        let zncc_avg: f64 = {
            let vals: Vec<f64> = audit.text_entries.iter()
                .filter_map(|e| e.similarity_score.map(|s| s as f64))
                .collect();
            if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 }
        };
        let test_json = serde_json::json!({
            "primary_hits": acc.primary_hits,
            "compared": acc.compared,
            "pct": (acc.pct * 10.0).round() / 10.0,
            "zncc_avg": (zncc_avg * 10000.0).round() / 10000.0,
            "major_misses": acc.major_misses,
            "minor_misses": acc.minor_misses,
            "similarity_failures": acc.similarity_failures,
            "hits": acc.hits,
            "kept_raster": acc.kept_raster,
            "ocr_correct_total": acc.ocr_correct_total,
            "ocr_correct_hits": acc.ocr_correct_hits,
            "ocr_wrong_total": acc.ocr_wrong_total,
            "ocr_wrong_hits": acc.ocr_wrong_hits,
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

