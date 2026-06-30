//! HTML miss report generation.
//!
//! Automatically generated into `<audit_dir>/report.html`.  When
//! `--audit` is also provided, classifies lines as hits/misses against
//! ground truth from the vector PDF.  Without `--audit`, reports all
//! kept-raster lines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ab_glyph::FontRef;
use base64::Engine;
use image::GrayImage;

use crate::audit::{AuditEntry, Decision};

use crate::char_render;
use crate::ground_truth::{self, GroundTruth};
use crate::font_scan::FontEntry;
use crate::glyph_map::GlyphMap;

/// Metadata about the run, displayed in the report header.
pub struct ReportMeta {
    pub classifier: String,
    pub render_scale: u32,
    pub render_aa: String,
    pub render_binarize: Option<u8>,
    pub elapsed: std::time::Duration,
}

// ── Glyph helpers ───────────────────────────────────────────────────────────

/// Shorten a font_key to just the filename with file extension stripped.
/// Variant tags ("|smcp") are preserved.
fn short_key(key: &str) -> String {
    let name = key.rsplit('/').next().unwrap_or(key);
    // Handle "Font.otf|tag" → "Font|tag"
    for ext in &[".otf", ".ttf", ".ttc"] {
        if let Some(pos) = name.find(ext) {
            let mut s = String::with_capacity(name.len() - 4);
            s.push_str(&name[..pos]);
            s.push_str(&name[pos + 4..]); // skip 4-char extension
            return s;
        }
    }
    name.to_string()
}

/// Resolve a glyph_id for a character to a display-friendly font key.
/// Falls back to "glyph#{id}" when the map has no entry.
fn glyph_display_key(glyph_map: &GlyphMap, ch: char, glyph_id: usize) -> String {
    glyph_map.fonts_for_glyph(ch, glyph_id)
        .first()
        .cloned()
        .unwrap_or_else(|| format!("glyph#{glyph_id}"))
}

// ── Image helpers ───────────────────────────────────────────────────────────

fn img_to_b64_uri(img: &GrayImage) -> String {
    let mut buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut buf);
    image::ImageEncoder::write_image(
        encoder,
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::L8,
    )
    .ok();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
    format!("data:image/png;base64,{b64}")
}

fn file_to_b64_uri(path: &Path) -> Option<String> {
    let data = std::fs::read(path).ok()?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let mime = match ext {
        "jpg" | "jpeg" => "image/jpeg",
        _ => "image/png",
    };
    Some(format!("data:{mime};base64,{b64}"))
}

fn img_td(uri: Option<&str>) -> String {
    match uri {
        Some(u) => format!("<img src=\"{u}\" class=\"ci\">"),
        None => "—".into(),
    }
}

fn dist_class(d2: f32) -> &'static str {
    if d2 > 0.05 {
        "bad"
    } else if d2 > 0.01 {
        "warn"
    } else {
        "ok"
    }
}

fn prob_class(p: f32) -> &'static str {
    if p < 0.01 {
        "bad"
    } else if p < 0.1 {
        "warn"
    } else {
        "ok"
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

// ── Font resolution ─────────────────────────────────────────────────────────

/// Try to find a font entry in the catalog matching a ground-truth font name.
/// GT BaseFont names are PostScript names (name ID 6), so we do an exact
/// lookup against each entry's postscript_name. No heuristics.
fn find_font_in_catalog<'a>(
    font_catalog: &'a [FontEntry],
    gt_font_name: &str,
) -> Option<&'a FontEntry> {
    let gt_stripped = ground_truth::strip_subset_prefix_str(gt_font_name);
    // Exact PostScript name match.  Variant entries carry "PSName|tag"
    // so they won't collide with base "PSName" — single-pass lookup.
    font_catalog.iter().find(|fe| fe.postscript_name == gt_stripped)
}

/// Try to find a font entry matching a CI candidate font_key.
fn find_font_by_key<'a>(font_catalog: &'a [FontEntry], font_key: &str) -> Option<&'a FontEntry> {
    font_catalog.iter().find(|fe| fe.font_key() == font_key)
}

// ── Character selection (mirrors Python pick_interesting_chars) ──────────────

fn pick_interesting_chars(
    chars: &[crate::audit::CharCiVote],
    n_worst: usize,
    n_normal: usize,
) -> Vec<(usize, &crate::audit::CharCiVote)> {
    let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut result: Vec<(usize, &crate::audit::CharCiVote)> = Vec::new();

    // 1. OCR-corrected characters (always show)
    for (i, c) in chars.iter().enumerate() {
        if c.ocr_corrected_from.is_some() {
            used.insert(i);
            result.push((i, c));
        }
    }

    // 2. Worst characters for the chosen/matched font (lowest chosen_prob)
    {
        let mut by_chosen: Vec<(usize, &crate::audit::CharCiVote)> =
            chars.iter().enumerate()
                .filter(|(_, c)| c.chosen_prob.is_some())
                .collect();
        by_chosen.sort_by(|a, b| {
            a.1.chosen_prob.unwrap()
                .partial_cmp(&b.1.chosen_prob.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, c) in by_chosen.iter().take(n_worst) {
            if used.insert(*i) {
                result.push((*i, c));
            }
        }
    }

    // 3. Worst characters for the ground-truth font (lowest gt_font_prob)
    {
        let mut by_gt: Vec<(usize, &crate::audit::CharCiVote)> =
            chars.iter().enumerate()
                .filter(|(_, c)| c.gt_font_prob.is_some())
                .collect();
        by_gt.sort_by(|a, b| {
            a.1.gt_font_prob.unwrap()
                .partial_cmp(&b.1.gt_font_prob.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, c) in by_gt.iter().take(n_worst) {
            if used.insert(*i) {
                result.push((*i, c));
            }
        }
    }

    // 4. Worst by best_prob (lowest probability = worst match)
    {
        let mut by_prob: Vec<(usize, &crate::audit::CharCiVote)> =
            chars.iter().enumerate().collect();
        by_prob.sort_by(|a, b| {
            a.1.best_prob
                .partial_cmp(&b.1.best_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, c) in by_prob.iter().take(n_worst) {
            if used.insert(*i) {
                result.push((*i, c));
            }
        }
    }

    // 5. A few normal characters for contrast (highest probability)
    {
        let mut by_prob: Vec<(usize, &crate::audit::CharCiVote)> =
            chars.iter().enumerate().collect();
        by_prob.sort_by(|a, b| {
            b.1.best_prob
                .partial_cmp(&a.1.best_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut count = 0;
        for (i, c) in by_prob.iter() {
            if count >= n_normal {
                break;
            }
            if c.best_prob > 0.5 && used.insert(*i) {
                result.push((*i, c));
                count += 1;
            }
        }
    }

    result.sort_by_key(|(i, _)| *i);
    result
}

// ── Crop / image lookup ─────────────────────────────────────────────────────

/// Find the diag-seg line directory for an audit entry.
fn find_diag_seg_dir(audit_root: &Path, page: usize, line_index: usize) -> Option<PathBuf> {
    let prefix = format!("p{page}_L{line_index:03}_");
    let rd = std::fs::read_dir(audit_root).ok()?;
    for entry in rd {
        if let Ok(e) = entry {
            let name = e.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix) && e.file_type().map(|t| t.is_dir()).unwrap_or(false)
            {
                return Some(e.path());
            }
        }
    }
    None
}

/// Find the crop PNG for a specific crop index in a diag-seg line directory.
fn find_crop_png(diag_dir: &Path, crop_index: usize) -> Option<PathBuf> {
    let crop_dir = diag_dir.join("crops");
    if !crop_dir.is_dir() {
        return None;
    }
    let prefix = format!("crop_{crop_index:02}_");
    for entry in std::fs::read_dir(&crop_dir).ok()? {
        if let Ok(e) = entry {
            let name = e.file_name();
            if name.to_string_lossy().starts_with(&prefix) {
                return Some(e.path());
            }
        }
    }
    None
}

/// Find the font ref glyph PNG for a character in the font_refs directory.
fn find_font_ref_png(audit_root: &Path, font_entry: &FontEntry, ch: char) -> Option<PathBuf> {
    let mut label = font_entry.family_name.replace(' ', "");
    if font_entry.is_bold {
        label.push_str("-Bold");
    }
    if font_entry.is_italic {
        label.push_str("-Italic");
    }
    if !font_entry.variant_tag.is_empty() {
        label.push('_');
        label.push_str(&font_entry.variant_tag);
    }
    let path = audit_root
        .join("font_refs")
        .join(&label)
        .join(format!("U+{:04X}.png", ch as u32));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

// ── Font cache for rendering correct-font ref glyphs ────────────────────────

struct FontDataCache {
    cache: HashMap<PathBuf, Option<Vec<u8>>>,
}

impl FontDataCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    fn load(&mut self, path: &Path) -> Option<&[u8]> {
        self.cache
            .entry(path.to_path_buf())
            .or_insert_with(|| std::fs::read(path).ok())
            .as_deref()
    }
}

// ── Miss classification ─────────────────────────────────────────────────────

struct ClassifiedEntry<'a> {
    entry: &'a AuditEntry,
    actual_font: Option<String>,
    kind: MissKind,
    /// Ground-truth effective font size in PDF points (if available).
    gt_font_size_pt: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissKind {
    Hit,
    MajorMiss,
    MinorMiss,
    SsimFailure,
    KeptRaster,
    NoGroundTruth,
}

impl MissKind {
    fn as_str(&self) -> &'static str {
        match self {
            MissKind::Hit => "hit",
            MissKind::MajorMiss => "major_miss",
            MissKind::MinorMiss => "minor_miss",
            MissKind::SsimFailure => "ssim_failure",
            MissKind::KeptRaster => "kept_raster",
            MissKind::NoGroundTruth => "no_ground_truth",
        }
    }
}

fn classify_entries<'a>(
    entries: &'a [AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
    _glyph_map: &GlyphMap,
) -> Vec<ClassifiedEntry<'a>> {
    entries
        .iter()
        .map(|e| {
            let bbox_px = [
                e.bbox.x as f32,
                e.bbox.y as f32,
                (e.bbox.x + e.bbox.width) as f32,
                (e.bbox.y + e.bbox.height) as f32,
            ];

            if let Some(gt) = gt {
                let (actual_font, gt_font_size_pt) = match gt
                    .lookup_font_and_size(e.page, &bbox_px, dpi)
                {
                    Some((name, size)) => (Some(ground_truth::strip_subset_prefix_str(name)), Some(size)),
                    None => (None, None),
                };

                let kind = if e.decision == Decision::KeptRaster {
                    MissKind::KeptRaster
                } else if let Some(ref actual) = actual_font {
                    // After GT canonicalization, both sides use canonical
                    // (weight-explicit) names.  Variant entries carry
                    // "PSName|tag" which won't match base "PSName", so
                    // exact equality is sufficient.
                    let ps_match = e.ci_candidates.first().and_then(|c| {
                        find_font_by_key(font_catalog, &c.font_key)
                    }).map_or(false, |fe| {
                        fe.postscript_name == *actual
                    });

                    if ps_match {
                        if e.ssim_pass == Some(false) {
                            MissKind::SsimFailure
                        } else {
                            MissKind::Hit
                        }
                    } else {
                        // Font miss — classify as major or minor.
                        // Read identity from both the picked font and the GT font.
                        let picked_path = e.ci_candidates.first()
                            .and_then(|c| find_font_by_key(font_catalog, &c.font_key))
                            .map(|fe| fe.path.clone());
                        let gt_path = font_catalog.iter()
                            .find(|fe| fe.postscript_name == *actual)
                            .map(|fe| fe.path.clone());

                        let is_minor = match (picked_path, gt_path) {
                            (Some(pp), Some(gp)) => {
                                match (crate::font_scan::read_font_identity(&pp),
                                       crate::font_scan::read_font_identity(&gp)) {
                                    (Some(pi), Some(gi)) => !pi.is_major_diff(&gi),
                                    _ => false, // can't read → assume major
                                }
                            }
                            _ => false,
                        };

                        if is_minor { MissKind::MinorMiss } else { MissKind::MajorMiss }
                    }
                } else {
                    MissKind::NoGroundTruth
                };

                ClassifiedEntry {
                    entry: e,
                    actual_font,
                    kind,
                    gt_font_size_pt,
                }
            } else {
                // No ground truth: only classify kept-raster vs hit
                let kind = if e.decision == Decision::KeptRaster {
                    MissKind::KeptRaster
                } else {
                    MissKind::Hit
                };
                ClassifiedEntry {
                    entry: e,
                    actual_font: None,
                    kind,
                    gt_font_size_pt: None,
                }
            }
        })
        .collect()
}

/// Enrich audit entries with ground-truth classification fields
/// (`miss_type` and `expected_font`).  Call this before writing
/// the audit JSON so downstream tools never have to parse the HTML
/// report.
pub fn enrich_audit_entries(
    entries: &mut [AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
    glyph_map: &GlyphMap,
) {
    // classify_entries produces a 1:1 parallel vec.
    let results: Vec<(String, Option<String>)> = {
        let classified = classify_entries(entries, gt, dpi, font_catalog, glyph_map);
        classified
            .iter()
            .map(|ce| (ce.kind.as_str().to_string(), ce.actual_font.clone()))
            .collect()
    };
    for (e, (kind, expected)) in entries.iter_mut().zip(results) {
        e.miss_type = Some(kind);
        e.expected_font = expected;
    }
}

// ── CI candidate lookup ─────────────────────────────────────────────────────

/// Strip directory and file extension from a font key while preserving the
fn find_correct_ci_candidate(
    entry: &AuditEntry,
    actual_font: &str,
    font_catalog: &[FontEntry],
    _glyph_map: &GlyphMap,
) -> (Option<String>, Option<f32>, Option<usize>) {
    let gt_ps = ground_truth::strip_subset_prefix_str(actual_font);

    // Match CI candidate's PostScript name against GT font.
    // After GT canonicalization, both names are canonical — exact equality.
    // Variant entries carry "PSName|tag" so they won't match base "PSName".
    for (i, c) in entry.ci_candidates.iter().enumerate() {
        if let Some(fe) = find_font_by_key(font_catalog, &c.font_key) {
            if fe.postscript_name == gt_ps {
                return (Some(c.font_key.clone()), c.score, Some(i + 1));
            }
        }
    }
    (None, None, None)
}

// ── HTML block generation ───────────────────────────────────────────────────

fn build_miss_block(
    ce: &ClassifiedEntry,
    audit_root: &Path,
    font_catalog: &[FontEntry],
    glyph_map: &GlyphMap,
    font_data_cache: &mut FontDataCache,
    dpi: u32,
) -> String {
    let entry = ce.entry;
    let actual_font = ce.actual_font.as_deref().map(short_key).unwrap_or_else(|| "?".into());
    let matched = entry.font_matched.as_deref().map(short_key).unwrap_or_else(|| "?".into());

    // Find correct font CI candidate
    let (gt_key, gt_score, gt_rank) =
        if let Some(ref af) = ce.actual_font {
            find_correct_ci_candidate(entry, af, font_catalog, glyph_map)
        } else {
            (None, None, None)
        };

    // Resolve font entries for correct and chosen fonts
    let correct_fe = ce.actual_font.as_deref().and_then(|af| {
        // Try CI candidate key first
        gt_key
            .as_deref()
            .and_then(|k| find_font_by_key(font_catalog, k))
            .or_else(|| find_font_in_catalog(font_catalog, af))
    });

    let chosen_fe = entry
        .font_matched
        .as_deref()
        .and_then(|m| {
            entry
                .ci_candidates
                .first()
                .and_then(|c| find_font_by_key(font_catalog, &c.font_key))
                .or_else(|| find_font_in_catalog(font_catalog, m))
        });

    // Find diag_seg_dir for this line
    let diag_dir = find_diag_seg_dir(audit_root, entry.page, entry.line_index);

    // Determine if font pick is correct (skip char table + GT render for SSIM-only failures)
    let font_is_correct = ce.kind == MissKind::SsimFailure;

    // SSIM comparison block — render correct font for side-by-side if this is a font miss
    let (gt_render_uri, gt_diff_uri, gt_ssim) = if !font_is_correct {
        render_correct_font_comparison(entry, correct_fe, font_data_cache, diag_dir.as_deref())
    } else {
        (None, None, None)
    };
    // Compute font sizes for comparison
    let gt_font_size_pt = ce.gt_font_size_pt;
    let inferred_size = chosen_fe.and_then(|fe| {
        let data = font_data_cache.load(&fe.path)?;
        compute_inferred_font_size(data, &entry.word_bboxes, &fe.variant_tag, fe.variations.as_deref(), dpi)
    });
    let unprint_font_size_pt = inferred_size.as_ref().map(|s| s.median_pt);

    let ssim_compare_html = if let Some(ref dd) = diag_dir {
        build_ssim_block(
            dd, &actual_font, &matched, entry.ssim_score,
            gt_render_uri.as_deref(), gt_diff_uri.as_deref(),
            gt_ssim,
            gt_font_size_pt, unprint_font_size_pt,
            inferred_size.as_ref().map(|s| s.per_word.as_slice()),
        )
    } else {
        String::new()
    };

    // CI tie-break comparison block
    let tie_break_html = if let Some(ref dd) = diag_dir {
        build_tie_break_block(entry, dd)
    } else {
        String::new()
    };

    // Scan line image with word bbox + segmentation overlays
    let scan_line_html = if let Some(ref dd) = diag_dir {
        build_scan_line_with_overlays(dd, entry)
    } else {
        String::new()
    };

    // SSIM info
    let ssim_html = match (entry.ssim_score, entry.ssim_pass) {
        (Some(score), Some(pass)) => {
            let cls = if pass { "ssim-pass" } else { "ssim-fail" };
            let label = if pass { "pass" } else { "FAIL" };
            format!(" <span class=\"{cls}\">ZNCC {score:.10} ({label})</span>")
        }
        _ => String::new(),
    };

    // Per-character comparison table
    let char_table_html = if font_is_correct || entry.ci_char_votes.is_empty() {
        String::new()
    } else {
        let chars_to_show = pick_interesting_chars(&entry.ci_char_votes, 4, 2);
        build_char_table(
            entry,
            &chars_to_show,
            audit_root,
            correct_fe,
            chosen_fe,
            &actual_font,
            &matched,
            gt_rank,
            gt_score,
            font_data_cache,
            diag_dir.as_deref(),
            font_catalog,
            glyph_map,
        )
    };

    let text_preview = truncate(&entry.text, 60);
    let miss_kind_label = match ce.kind {
        MissKind::MajorMiss => {
            // Show expected→got summary for font misses
            format!(" [MAJOR: expected {}, got {}]", actual_font, matched)
        },
        MissKind::MinorMiss => {
            format!(" [minor: expected {}, got {}]", actual_font, matched)
        },
        MissKind::SsimFailure => " [ZNCC failure]".to_string(),
        MissKind::KeptRaster => " [kept raster]".to_string(),
        _ => String::new(),
    };

    // Segmentation path comparison (ligature vs plain)
    let seg_path_html = if let Some(ref winner) = entry.seg_winner {
        if !entry.ci_candidates_lig.is_empty() {
            let (lig_font, lig_score, plain_font, plain_score) = if winner == "ligature" {
                // ci_candidates = ligature (winner), ci_candidates_lig = plain (loser)
                let lf = entry.ci_candidates.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ls = entry.ci_candidates.first().and_then(|c| c.score);
                let pf = entry.ci_candidates_lig.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ps = entry.ci_candidates_lig.first().and_then(|c| c.score);
                (lf, ls, pf, ps)
            } else {
                // ci_candidates = plain (winner), ci_candidates_lig = ligature (loser)
                let pf = entry.ci_candidates.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ps = entry.ci_candidates.first().and_then(|c| c.score);
                let lf = entry.ci_candidates_lig.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ls = entry.ci_candidates_lig.first().and_then(|c| c.score);
                (lf, ls, pf, ps)
            };
            let lig_marker = if winner == "ligature" { " ✓" } else { "" };
            let plain_marker = if winner == "plain" { " ✓" } else { "" };
            let ls = lig_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "—".into());
            let ps = plain_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "—".into());
            format!(
                "<div class=\"seg-paths\">\
                 <b>Seg paths:</b> \
                 ligature → {} ({}){} · \
                 plain → {} ({}){}\
                 </div>",
                lig_font, ls, lig_marker,
                plain_font, ps, plain_marker,
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    format!(
        "<div class=\"miss\">\
         <h3>p{}:L{} — \"{}\"{}{}</h3>\
         {}\
         {}\
         {}\
         {}\
         {}\
         </div>",
        entry.page, entry.line_index, text_preview, miss_kind_label, ssim_html,
        seg_path_html, scan_line_html, ssim_compare_html, tie_break_html, char_table_html,
    )
}

/// Build scan line image with word bbox and segmentation path overlays.
/// Replicates the Python char-misses.py `render_scan_line_with_word_boxes()`:
///   - scan_line.png as the background (colour crop of the word-union region)
///   - Raw Tesseract word bboxes: dotted orange outlines
///   - Final post-processed word bboxes: dashed cyan outlines
///   - VP splits: blue vertical lines with column labels
///   - Seam paths: magenta diagonal paths with column labels
///   - Pixel-scale ruler at top of each final word box
fn build_scan_line_with_overlays(diag_dir: &Path, entry: &AuditEntry) -> String {
    // Prefer scan_line.png (full-colour); fall back to ssim_scan.png (grayscale)
    let scan_path = diag_dir.join("scan_line.png");
    let origin_path = diag_dir.join("scan_line_origin.json");
    let (img_uri, crop_x, crop_y) = if scan_path.exists() {
        let uri = match file_to_b64_uri(&scan_path) {
            Some(u) => u,
            None => return String::new(),
        };
        // Read crop origin
        let (ox, oy) = if let Ok(data) = std::fs::read_to_string(&origin_path) {
            let v: serde_json::Value = serde_json::from_str(&data).unwrap_or_default();
            (
                v["x"].as_u64().unwrap_or(0) as u32,
                v["y"].as_u64().unwrap_or(0) as u32,
            )
        } else {
            (entry.bbox.x, entry.bbox.y)
        };
        (uri, ox, oy)
    } else {
        // Fall back to ssim_scan.png
        let ssim_path = diag_dir.join("ssim_scan.png");
        if !ssim_path.exists() {
            return String::new();
        }
        match file_to_b64_uri(&ssim_path) {
            Some(u) => (u, entry.bbox.x, entry.bbox.y),
            None => return String::new(),
        }
    };

    // Get image dimensions for the container
    let (img_w, img_h) = if scan_path.exists() {
        image::image_dimensions(&scan_path)
            .unwrap_or((entry.bbox.width, entry.bbox.height))
    } else {
        let ssim_path = diag_dir.join("ssim_scan.png");
        image::image_dimensions(&ssim_path)
            .unwrap_or((entry.bbox.width, entry.bbox.height))
    };

    let scale = 3u32; // 3× upscale to match Python
    let margin_top = 18 * scale;
    let margin_bot = 14 * scale;
    let canvas_w = img_w * scale;
    let canvas_h = img_h * scale + margin_top + margin_bot;

    let mut overlays = String::new();

    // Raw Tesseract boxes — dotted orange
    for wb in &entry.word_bboxes_raw {
        let bx = (wb.x.saturating_sub(crop_x)) * scale;
        let by = (wb.y.saturating_sub(crop_y)) * scale + margin_top;
        let bw = wb.width * scale;
        let bh = wb.height * scale;
        overlays.push_str(&format!(
            "<div style=\"position:absolute;left:{bx}px;top:{by}px;width:{bw}px;height:{bh}px;\
             border:1px dotted rgba(255,160,0,0.8);pointer-events:none;\"></div>"
        ));
    }

    // Final post-processed boxes — dashed cyan
    for wb in &entry.word_bboxes {
        let bx = (wb.x.saturating_sub(crop_x)) * scale;
        let by = (wb.y.saturating_sub(crop_y)) * scale + margin_top;
        let bw = wb.width * scale;
        let bh = wb.height * scale;
        overlays.push_str(&format!(
            "<div style=\"position:absolute;left:{bx}px;top:{by}px;width:{bw}px;height:{bh}px;\
             border:1px dashed rgba(0,200,220,0.85);pointer-events:none;\"></div>"
        ));

        // Pixel-scale ruler at top of each final word box
        let wb_px_w = wb.width;
        let mut col = 0u32;
        while col <= wb_px_w {
            let sx = bx + col * scale + scale / 2;
            if col % 10 == 0 {
                // Major tick + label
                overlays.push_str(&format!(
                    "<div style=\"position:absolute;left:{sx}px;top:{}px;width:1px;height:6px;\
                     background:rgba(140,140,140,0.7);pointer-events:none;\"></div>\
                     <div style=\"position:absolute;left:{}px;top:{}px;\
                     font-size:7px;color:rgba(120,120,120,0.8);pointer-events:none;\">{col}</div>",
                    by.saturating_sub(6), sx.saturating_sub(8), by.saturating_sub(18)
                ));
            } else {
                // Minor tick
                overlays.push_str(&format!(
                    "<div style=\"position:absolute;left:{sx}px;top:{}px;width:1px;height:3px;\
                     background:rgba(180,180,180,0.6);pointer-events:none;\"></div>",
                    by.saturating_sub(3)
                ));
            }
            col += 5;
        }
    }

    // Segmentation paths from diag-seg summary.json
    if let Ok(rd) = std::fs::read_dir(diag_dir) {
        let mut word_dirs: Vec<_> = rd
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name().to_string_lossy().starts_with("word_")
                    && e.path().is_dir()
            })
            .collect();
        word_dirs.sort_by_key(|d| d.file_name().to_string_lossy().to_string());

        for wd in &word_dirs {
            let wpath = wd.path();
            let data_path = {
                let sp = wpath.join("seg_plain");
                if sp.is_dir() { sp } else { wpath.clone() }
            };
            let summary_path = data_path.join("summary.json");
            if !summary_path.exists() {
                continue;
            }
            let summary: serde_json::Value = match std::fs::read_to_string(&summary_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
            {
                Some(v) => v,
                None => continue,
            };

            let word_text = summary["word_text"].as_str().unwrap_or("");
            let seg_img_w = summary["image_w"].as_u64().unwrap_or(0) as u32;
            let seg_img_h = summary["image_h"].as_u64().unwrap_or(0) as u32;
            if seg_img_w == 0 || seg_img_h == 0 {
                continue;
            }

            // Find matching word bbox
            let matching_wb = entry.word_bboxes.iter().find(|wb| wb.text == word_text);
            let matching_wb = match matching_wb {
                Some(wb) => wb,
                None => continue,
            };

            let wx = (matching_wb.x.saturating_sub(crop_x)) * scale;
            let wy = (matching_wb.y.saturating_sub(crop_y)) * scale + margin_top;
            let wb_h = matching_wb.height * scale;

            // Scale factors: seg boundaries are in word image pixels
            let sx_f = matching_wb.width as f64 / seg_img_w as f64 * scale as f64;
            let sy_f = matching_wb.height as f64 / seg_img_h as f64 * scale as f64;

            let label_y = wy + wb_h + 2;

            // VP splits — blue vertical lines
            if let Some(vp_arr) = summary["vp_splits"].as_array() {
                for vp in vp_arr {
                    if let Some(col) = vp.as_u64() {
                        let cx = wx + (col as f64 * sx_f) as u32;
                        overlays.push_str(&format!(
                            "<div style=\"position:absolute;left:{cx}px;top:{wy}px;width:1px;height:{wb_h}px;\
                             background:rgba(40,100,220,0.8);pointer-events:none;\"></div>\
                             <div style=\"position:absolute;left:{}px;top:{label_y}px;\
                             font-size:7px;color:rgba(40,100,220,0.9);pointer-events:none;\">{col}</div>",
                            cx.saturating_sub(6)
                        ));
                    }
                }
            }

            // Seam paths — magenta diagonal paths (one x per row)
            if let Some(seam_obj) = summary["seam_paths"].as_object() {
                for (col_key, path_arr) in seam_obj {
                    if let Some(path) = path_arr.as_array() {
                        for (row_idx, px) in path.iter().enumerate() {
                            if let Some(col_px) = px.as_u64() {
                                let px_x = wx + (col_px as f64 * sx_f) as u32;
                                let px_y = wy + (row_idx as f64 * sy_f) as u32;
                                let dot_pad = scale.max(1) / 3;
                                overlays.push_str(&format!(
                                    "<div style=\"position:absolute;left:{}px;top:{px_y}px;\
                                     width:{}px;height:{}px;\
                                     background:rgba(255,0,200,0.8);pointer-events:none;\"></div>",
                                    px_x.saturating_sub(dot_pad),
                                    dot_pad * 2 + 1,
                                    scale.max(1),
                                ));
                            }
                        }
                        // Column label
                        let nominal_col = col_key.parse::<u64>().unwrap_or(0);
                        let cx = wx + (nominal_col as f64 * sx_f) as u32;
                        overlays.push_str(&format!(
                            "<div style=\"position:absolute;left:{}px;top:{label_y}px;\
                             font-size:7px;color:rgba(255,0,200,0.9);pointer-events:none;\">{col_key}</div>",
                            cx.saturating_sub(6)
                        ));
                    }
                }
            }

            // Seam splits without paths — magenta vertical lines
            let seam_path_keys: std::collections::HashSet<String> = summary["seam_paths"]
                .as_object()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            if let Some(seam_arr) = summary["seam_splits"].as_array() {
                for seam in seam_arr {
                    if let Some(col) = seam.as_u64() {
                        if !seam_path_keys.contains(&col.to_string()) {
                            let cx = wx + (col as f64 * sx_f) as u32;
                            overlays.push_str(&format!(
                                "<div style=\"position:absolute;left:{cx}px;top:{wy}px;width:1px;height:{wb_h}px;\
                                 background:rgba(255,0,200,0.7);pointer-events:none;\"></div>\
                                 <div style=\"position:absolute;left:{}px;top:{label_y}px;\
                                 font-size:7px;color:rgba(255,0,200,0.9);pointer-events:none;\">{col}</div>",
                                cx.saturating_sub(6)
                            ));
                        }
                    }
                }
            }
        }
    }

    // Build the segmentation stats line
    let seg_stats = build_seg_stats(diag_dir, entry);

    format!(
        "<div class=\"scan-line-block\">\
         <div class=\"scan-line-label\">\
         Scan line: <span style=\"color:#ffa000\">···</span> raw box · \
         <span style=\"color:#00c8dc\">- -</span> final box · \
         <span style=\"color:#2864dc\">│</span> v-whitespace · \
         <span style=\"color:#ff00c8\">╲</span> seam</div>\
         <div style=\"position:relative;display:inline-block;width:{canvas_w}px;height:{canvas_h}px;overflow:visible;\">\
         <img src=\"{img_uri}\" style=\"position:absolute;left:0;top:{margin_top}px;\
         width:{cw}px;height:{ch}px;image-rendering:pixelated;\" class=\"scan-line-img\">\
         {overlays}\
         </div>\
         {seg_stats}\
         </div>",
        cw = img_w * scale,
        ch = img_h * scale,
    )
}

fn build_seg_stats(diag_dir: &Path, entry: &AuditEntry) -> String {
    // Build word text → x position map for left-to-right ordering
    let word_x_map: HashMap<&str, u32> = entry
        .word_bboxes
        .iter()
        .map(|wb| (wb.text.as_str(), wb.x))
        .collect();

    let mut seg_parts: Vec<(u32, String)> = Vec::new();

    // Read word directories
    let word_dirs = match std::fs::read_dir(diag_dir) {
        Ok(rd) => {
            let mut dirs: Vec<_> = rd
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name().to_string_lossy().starts_with("word_")
                        && e.path().is_dir()
                })
                .collect();
            dirs.sort_by_key(|d| d.file_name().to_string_lossy().to_string());
            dirs
        }
        Err(_) => return String::new(),
    };

    for wd in &word_dirs {
        let wpath = wd.path();
        // Prefer seg_plain subdirectory, fall back to flat layout
        let data_path = {
            let sp = wpath.join("seg_plain");
            if sp.is_dir() { sp } else { wpath.clone() }
        };
        let summary_path = data_path.join("summary.json");
        if !summary_path.exists() {
            continue;
        }
        let summary: serde_json::Value = match std::fs::read_to_string(&summary_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            Some(v) => v,
            None => continue,
        };
        let wtext = summary["word_text"].as_str().unwrap_or("?");
        let n_exp = summary["n_chars_expected"]
            .as_u64()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "?".to_string());
        let n_got = summary["n_segments_produced"]
            .as_u64()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "?".to_string());
        let mismatch = summary["mismatch"].as_bool().unwrap_or(false);
        let nvp = summary["vp_splits"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        let nseam = summary["seam_splits"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);

        let word_x = word_x_map.get(wtext).copied().unwrap_or(999999u32);
        let mut info = format!("&quot;{wtext}&quot; {n_got}/{n_exp}");
        if mismatch {
            info.push_str(" \u{26a0}");
        }
        let mut tags = Vec::new();
        if nvp > 0 {
            tags.push(format!("{nvp} vert"));
        }
        if nseam > 0 {
            tags.push(format!("{nseam} seam"));
        }
        if !tags.is_empty() {
            info.push_str(&format!(" ({})", tags.join(", ")));
        }
        seg_parts.push((word_x, info));
    }

    if seg_parts.is_empty() {
        return String::new();
    }
    seg_parts.sort_by_key(|t| t.0);
    let stats = seg_parts
        .iter()
        .map(|(_, info)| info.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    format!("<div class=\"scan-line-label\">Segmentation: {stats}</div>")
}

/// Render the correct (ground-truth) font for a miss entry and produce
/// base64 URIs for the render image and its diff against the scan crop.
fn render_correct_font_comparison(
    entry: &AuditEntry,
    correct_fe: Option<&FontEntry>,
    font_data_cache: &mut FontDataCache,
    diag_dir: Option<&Path>,
) -> (Option<String>, Option<String>, Option<f32>) {
    let fe = match correct_fe {
        Some(fe) => fe,
        None => return (None, None, None),
    };
    let dd = match diag_dir {
        Some(d) => d,
        None => return (None, None, None),
    };

    // Load font data
    let font_data = match font_data_cache.load(&fe.path) {
        Some(d) => d,
        None => return (None, None, None),
    };

    // Build word placements relative to line bbox
    let placements: Vec<crate::verify::WordPlacement> = entry
        .word_bboxes
        .iter()
        .map(|wb| crate::verify::WordPlacement {
            text: wb.text.clone(),
            x_off: wb.x.saturating_sub(entry.bbox.x),
            y_off: wb.y.saturating_sub(entry.bbox.y),
            width: wb.width,
            height: wb.height,
        })
        .collect();

    if placements.is_empty() {
        return (None, None, None);
    }

    let overrides: Option<Vec<(char, u16)>> = fe
        .glyph_overrides
        .as_ref()
        .map(|v| v.iter().cloned().collect());

    // Render the line in the correct font
    let render_img = match crate::verify::render_line_for_comparison(
        font_data,
        &placements,
        entry.bbox.width,
        entry.bbox.height,
        overrides.as_deref(),
        &fe.variant_tag,
        fe.variations.as_deref(),
    ) {
        Some(img) => img,
        None => return (None, None, None),
    };

    let render_uri = img_to_b64_uri(&render_img);

    // Load ssim_scan.png, compute diff and ZNCC (same path as verify_text_region)
    let scan_path = dd.join("ssim_scan.png");
    let (diff_uri, correct_ssim) = if scan_path.exists() {
        if let Ok(scan_dyn) = image::open(&scan_path) {
            let scan_gray = scan_dyn.to_luma8();
            let diff_img = crate::verify::compute_abs_diff(&scan_gray, &render_img);
            // Compute ZNCC: same pipeline as verify_text_region
            let (zncc_val, _dy) = crate::compare_rasters::zncc_windowed_best_vshift(
                &scan_gray, &render_img, 12, None,
            );
            (Some(img_to_b64_uri(&diff_img)), Some(zncc_val))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    (Some(render_uri), diff_uri, correct_ssim)
}

/// Compute the font size (in PDF points) that unprint would infer for the
/// picked font, based on word-width matching.
/// Per-word font-size detail for the audit report.
#[derive(Clone)]
struct WordSizeDetail {
    text: String,
    width_px: u32,
    em_px: f32,
    pt: f32,
}

struct InferredFontSize {
    median_pt: f32,
    per_word: Vec<WordSizeDetail>,
}

fn compute_inferred_font_size(
    font_data: &[u8],
    word_bboxes: &[crate::audit::WordBBox],
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
    dpi: u32,
) -> Option<InferredFontSize> {
    let scale = 72.0 / dpi as f32;
    let per_word: Vec<WordSizeDetail> = word_bboxes
        .iter()
        .filter(|wb| !wb.text.is_empty() && wb.width >= 1)
        .filter_map(|wb| {
            let em_px = crate::layout::width_matched_em_px_shaped(
                font_data,
                &wb.text,
                wb.width as f32,
                variant_tag,
                variations,
            )?;
            Some(WordSizeDetail {
                text: wb.text.clone(),
                width_px: wb.width,
                em_px,
                pt: em_px * scale,
            })
        })
        .collect();
    if per_word.is_empty() {
        return None;
    }
    let mut sorted_em: Vec<f32> = per_word.iter().map(|w| w.em_px).collect();
    sorted_em.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_em_px = sorted_em[sorted_em.len() / 2];
    Some(InferredFontSize {
        median_pt: median_em_px * scale,
        per_word,
    })
}


fn build_ssim_block(
    diag_dir: &Path,
    correct_font: &str,
    chosen_font: &str,
    ssim_score: Option<f32>,
    correct_render_uri: Option<&str>,
    correct_diff_uri: Option<&str>,
    correct_ssim: Option<f32>,
    gt_font_size_pt: Option<f32>,
    unprint_font_size_pt: Option<f32>,
    per_word_sizes: Option<&[WordSizeDetail]>,
) -> String {
    let scan_path = diag_dir.join("ssim_scan.png");
    let render_path = diag_dir.join("ssim_render.png");
    let diff_path = diag_dir.join("ssim_diff.png");

    if !scan_path.exists() || !render_path.exists() {
        return String::new();
    }

    let scan_uri = match file_to_b64_uri(&scan_path) {
        Some(u) => u,
        None => return String::new(),
    };
    let render_uri = match file_to_b64_uri(&render_path) {
        Some(u) => u,
        None => return String::new(),
    };

    let diff_row = if diff_path.exists() {
        if let Some(diff_uri) = file_to_b64_uri(&diff_path) {
            Some(diff_uri)
        } else {
            None
        }
    } else {
        None
    };

    let ssim_str = ssim_score
        .map(|s| format!("{s:.10}"))
        .unwrap_or_else(|| "—".into());

    let correct_ssim_str = correct_ssim
        .map(|s| format!("{s:.10}"))
        .unwrap_or_else(|| "—".into());

    // Render row: side-by-side if correct-font render is available
    let render_row = if let Some(gt_uri) = correct_render_uri {
        format!(
            "<tr><td class=\"ssim-label\">Render</td>\
             <td><img src=\"{gt_uri}\" class=\"ssim-compare-img\"></td>\
             <td><img src=\"{render_uri}\" class=\"ssim-compare-img\"></td></tr>"
        )
    } else {
        format!(
            "<tr><td class=\"ssim-label\">Render</td>\
             <td colspan=\"2\"><img src=\"{render_uri}\" class=\"ssim-compare-img\"></td></tr>"
        )
    };

    // Diff row: side-by-side if both diffs are available
    let diff_row_html = match (correct_diff_uri, &diff_row) {
        (Some(gt_diff), Some(picked_diff)) => format!(
            "<tr><td class=\"ssim-label\">Diff</td>\
             <td><img src=\"{gt_diff}\" class=\"ssim-compare-img\"></td>\
             <td><img src=\"{picked_diff}\" class=\"ssim-compare-img\"></td></tr>"
        ),
        (Some(gt_diff), None) => format!(
            "<tr><td class=\"ssim-label\">Diff</td>\
             <td><img src=\"{gt_diff}\" class=\"ssim-compare-img\"></td>\
             <td>—</td></tr>"
        ),
        (None, Some(picked_diff)) => format!(
            "<tr><td class=\"ssim-label\">Diff</td>\
             <td colspan=\"2\"><img src=\"{picked_diff}\" class=\"ssim-compare-img\"></td></tr>"
        ),
        (None, None) => String::new(),
    };

    // Font size comparison row
    let font_size_row = match (gt_font_size_pt, unprint_font_size_pt) {
        (Some(gt_sz), Some(us_sz)) => {
            let delta = us_sz - gt_sz;
            let pct = if gt_sz.abs() > 0.01 { delta / gt_sz * 100.0 } else { 0.0 };
            let delta_class = if pct.abs() > 5.0 { "bad" } else if pct.abs() > 2.0 { "warn" } else { "ok" };
            format!(
                "<tr><td class=\"ssim-label\">Size</td>\
                 <td class=\"correct\"><span class=\"num\">{gt_sz:.2}pt</span></td>\
                 <td class=\"chosen\"><span class=\"num\">{us_sz:.2}pt</span> \
                 <span class=\"num {delta_class}\">({delta:+.2}pt / {pct:+.1}%)</span></td></tr>"
            )
        }
        (Some(gt_sz), None) => format!(
            "<tr><td class=\"ssim-label\">Size</td>\
             <td class=\"correct\"><span class=\"num\">{gt_sz:.2}pt</span></td>\
             <td class=\"chosen\">—</td></tr>"
        ),
        (None, Some(us_sz)) => format!(
            "<tr><td class=\"ssim-label\">Size</td>\
             <td class=\"correct\">—</td>\
             <td class=\"chosen\"><span class=\"num\">{us_sz:.2}pt</span></td></tr>"
        ),
        (None, None) => String::new(),
    };

    // Per-word size breakdown row
    let per_word_row = if let Some(words) = per_word_sizes {
        if words.len() > 1 {
            let median_pt = unprint_font_size_pt.unwrap_or(0.0);
            let mut cells: Vec<String> = Vec::new();
            for w in words {
                let pct = if median_pt.abs() > 0.01 { (w.pt - median_pt) / median_pt * 100.0 } else { 0.0 };
                let cls = if pct.abs() > 10.0 { "bad" } else if pct.abs() > 5.0 { "warn" } else { "ok" };
                let txt_esc = w.text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
                cells.push(format!(
                    "<span class=\"word-size-cell {cls}\">\"{txt_esc}\" <b>{:.2}pt</b> ({pct:+.1}%) [{}px]</span>",
                    w.pt, w.width_px
                ));
            }
            format!(
                "<tr><td class=\"ssim-label\">Words</td>\
                 <td colspan=\"2\" class=\"per-word-sizes\">{}</td></tr>",
                cells.join(" ")
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    format!(
        "<div class=\"ssim-compare-block\">\
         <table class=\"ssim-compare-table\">\
         <tr><th></th><th>Correct</th><th>Picked (ZNCC verified)</th></tr>\
         <tr><td class=\"ssim-label\">Font</td>\
         <td class=\"correct\">{correct_font}</td>\
         <td class=\"chosen\">{chosen_font}</td></tr>\
         {font_size_row}\
         {per_word_row}\
         <tr><td class=\"ssim-label\">Scan</td>\
         <td><img src=\"{scan_uri}\" class=\"ssim-compare-img\"></td>\
         <td><img src=\"{scan_uri}\" class=\"ssim-compare-img\"></td></tr>\
         {render_row}\
         {diff_row_html}\
         <tr><td class=\"ssim-label\">ZNCC</td>\
         <td class=\"correct\">{correct_ssim_str}</td>\
         <td class=\"chosen\">{ssim_str}</td></tr>\
         </table></div>"
    )
}

/// Build an SSIM tie-break comparison block showing each tied candidate's
/// render, diff, and SSIM score.
fn build_tie_break_block(
    entry: &AuditEntry,
    diag_dir: &Path,
) -> String {
    if entry.tie_candidates.is_empty() {
        return String::new();
    }

    // Load scan image (shared across all candidates)
    let scan_uri = {
        // Try tie_0/ssim_scan.png first, then fall back to parent ssim_scan.png
        let tie0_scan = diag_dir.join("tie_0").join("ssim_scan.png");
        let parent_scan = diag_dir.join("ssim_scan.png");
        let scan_path = if tie0_scan.exists() { tie0_scan } else { parent_scan };
        match file_to_b64_uri(&scan_path) {
            Some(u) => u,
            None => return String::new(),
        }
    };

    let mut rows = String::new();

    // Header row with candidate names
    let mut header = String::from("<tr><th></th>");
    for tc in &entry.tie_candidates {
        let winner_marker = if tc.winner { " ✓" } else { "" };
        header.push_str(&format!(
            "<th class=\"{}\">{}{}</th>",
            if tc.winner { "tie-winner" } else { "tie-loser" },
            short_key(&tc.font_key),
            winner_marker,
        ));
    }
    header.push_str("</tr>");
    rows.push_str(&header);

    // Scan row (same image for all)
    rows.push_str(&format!(
        "<tr><td class=\"ssim-label\">Scan</td><td colspan=\"{}\"><img src=\"{}\" class=\"ssim-compare-img\"></td></tr>",
        entry.tie_candidates.len(), scan_uri
    ));

    // Render row — one image per candidate
    rows.push_str("<tr><td class=\"ssim-label\">Render</td>");
    for (ti, _tc) in entry.tie_candidates.iter().enumerate() {
        let render_path = diag_dir.join(format!("tie_{}", ti)).join("ssim_render.png");
        if let Some(uri) = file_to_b64_uri(&render_path) {
            rows.push_str(&format!("<td><img src=\"{}\" class=\"ssim-compare-img\"></td>", uri));
        } else {
            rows.push_str("<td>—</td>");
        }
    }
    rows.push_str("</tr>");

    // Diff row — one image per candidate
    rows.push_str("<tr><td class=\"ssim-label\">Diff</td>");
    for (ti, _tc) in entry.tie_candidates.iter().enumerate() {
        let diff_path = diag_dir.join(format!("tie_{}", ti)).join("ssim_diff.png");
        if let Some(uri) = file_to_b64_uri(&diff_path) {
            rows.push_str(&format!("<td><img src=\"{}\" class=\"ssim-compare-img\"></td>", uri));
        } else {
            rows.push_str("<td>—</td>");
        }
    }
    rows.push_str("</tr>");

    // ZNCC score row
    rows.push_str("<tr><td class=\"ssim-label\">ZNCC</td>");
    for tc in &entry.tie_candidates {
        let class = if tc.winner { "tie-winner" } else { "tie-loser" };
        rows.push_str(&format!(
            "<td class=\"{}\">{:.6}</td>", class, tc.ssim_score
        ));
    }
    rows.push_str("</tr>");

    format!(
        "<div class=\"tie-break-block\">\
         <div class=\"tie-break-title\">CI Tie-Break ({} candidates, ZNCC decides)</div>\
         <table class=\"ssim-compare-table\">{}</table></div>",
        entry.tie_candidates.len(), rows
    )
}

fn build_char_table(
    entry: &AuditEntry,
    chars_to_show: &[(usize, &crate::audit::CharCiVote)],
    audit_root: &Path,
    correct_fe: Option<&FontEntry>,
    chosen_fe: Option<&FontEntry>,
    correct_font_name: &str,
    chosen_font_name: &str,
    gt_rank: Option<usize>,
    gt_score: Option<f32>,
    font_data_cache: &mut FontDataCache,
    diag_dir: Option<&Path>,
    font_catalog: &[FontEntry],
    glyph_map: &GlyphMap,
) -> String {
    let _ = font_catalog; // retained for future use
    let mut rows = String::new();

    for &(_idx, cv) in chars_to_show {
        let ch = cv.ch;
        let original_ocr = cv.ocr_corrected_from.unwrap_or(ch);
        let best_p = cv.best_prob;

        // Crop image from disk
        let crop_uri = diag_dir.and_then(|dd| {
            find_crop_png(dd, cv.crop_index)
                .and_then(|p| file_to_b64_uri(&p))
        });

        // Correct font reference glyph — render via shared pipeline
        let correct_ref_uri = correct_fe.and_then(|fe| {
            let data = font_data_cache.load(&fe.path)?;
            let mut font = FontRef::try_from_slice(data).ok()?;
            if let Some(ref vars) = fe.variations {
                use ab_glyph::VariableFont;
                for (tag, val) in vars {
                    font.set_variation(tag, *val);
                }
            }
            let gid_override = fe.glyph_overrides.as_ref()
                .and_then(|ovs| ovs.iter().find(|(c, _)| *c == ch).map(|(_, g)| ab_glyph::GlyphId(*g)));
            let (_hash, img) = char_render::get_rendered_char_default(&font, ch, gid_override)?;
            Some(img_to_b64_uri(&img))
        });

        // Chosen font reference glyph — try on-disk font_refs first, then render
        let chosen_ref_uri = chosen_fe.and_then(|fe| {
            // Try on-disk first
            if let Some(path) = find_font_ref_png(audit_root, fe, ch) {
                return file_to_b64_uri(&path);
            }
            // Render via shared pipeline
            let data = font_data_cache.load(&fe.path)?;
            let mut font = FontRef::try_from_slice(data).ok()?;
            if let Some(ref vars) = fe.variations {
                use ab_glyph::VariableFont;
                for (tag, val) in vars {
                    font.set_variation(tag, *val);
                }
            }
            let gid_override = fe.glyph_overrides.as_ref()
                .and_then(|ovs| ovs.iter().find(|(c, _)| *c == ch).map(|(_, g)| ab_glyph::GlyphId(*g)));
            let (_hash, img) = char_render::get_rendered_char_default(&font, ch, gid_override)?;
            Some(img_to_b64_uri(&img))
        });

        // OCR cell
        let ocr_label = if cv.ocr_corrected_from.is_some() {
            format!(
                "<span class='ocr-fix'>'{original_ocr}' → '{ch}'</span>"
            )
        } else {
            format!("'{original_ocr}'")
        };

        let mut ocr_parts = vec![format!("OCR: <b>{ocr_label}</b>")];

        // Best-scoring font for the OCR char (stage-1 diagnostic)
        if let Some(&(gid, _np)) = cv.nearest.first() {
            let font_name = glyph_display_key(glyph_map, cv.ch, gid);
            let short = font_name.rsplit('/').next().unwrap_or(&font_name);
            ocr_parts.push(format!(
                "<span class='font-mini'>{short}</span>"
            ));
        }

        // Best alt char
        if let (Some(alt_ch), Some(alt_dist)) = (cv.best_alt_char, cv.best_alt_dist) {
            let dc = dist_class(alt_dist);
            ocr_parts.push(format!(
                "Alt: <b>'{alt_ch}'</b> <span class='num {dc}'>{alt_dist:.6}</span>"
            ));
        }

        let ocr_cell = ocr_parts.join("<br>");

        // Per-char probability labels
        let chosen_score_label = if let Some(p) = cv.chosen_prob {
            let pc = prob_class(p);
            let rank_part = cv.chosen_rank
                .map(|r| format!(" <span class='font-mini'>rank {r}</span>"))
                .unwrap_or_default();
            format!("<div class='sub'><span class='num {pc}'>{p:.6}</span>{rank_part}</div>")
        } else {
            String::new()
        };

        let correct_score_label = if let Some(p) = cv.gt_font_prob {
            let pc = prob_class(p);
            let rank_part = cv.gt_font_rank
                .map(|r| format!(" <span class='font-mini'>rank {r}</span>"))
                .unwrap_or_default();
            format!("<div class='sub'><span class='num {pc}'>{p:.6}</span>{rank_part}</div>")
        } else {
            String::new()
        };

        let _pc = prob_class(best_p);

        rows.push_str(&format!(
            "<tr>\
             <td class=\"img-td\">{}</td>\
             <td class=\"img-td\">{}{}</td>\
             <td class=\"img-td\">{}{}</td>\
             <td class=\"ocr-col\">{}</td>\
             </tr>",
            img_td(crop_uri.as_deref()),
            img_td(correct_ref_uri.as_deref()),
            correct_score_label,
            img_td(chosen_ref_uri.as_deref()),
            chosen_score_label,
            ocr_cell,
        ));
    }

    // Column headers
    let rank_str = match (gt_rank, gt_score) {
        (Some(r), Some(s)) => format!("CI #{r}, score {s:.10}"),
        _ => "not in CI".into(),
    };

    // The chosen font is always CI candidate #1
    let chosen_rank_info = entry
        .ci_candidates
        .first()
        .map(|c| {
            match c.score {
                Some(s) => format!("CI #1, score {:.10}", s),
                None => "CI #1".into(),
            }
        })
        .unwrap_or_default();

    // If chosen font has same PostScript name as correct, show the correct name for both
    let chosen_ps = chosen_fe.map(|fe| fe.postscript_name.as_str()).unwrap_or("");
    let correct_ps = ground_truth::strip_subset_prefix_str(correct_font_name);
    let display_chosen = if chosen_ps == correct_ps {
        correct_font_name
    } else {
        chosen_font_name
    };

    format!(
        "<table>\
         <tr>\
         <th>Scan</th>\
         <th class=\"correct\">Correct: {correct_font_name}<br><span class='score'>{rank_str}</span></th>\
         <th class=\"chosen\">Unscan pick: {display_chosen}<br><span class='score'>{chosen_rank_info}</span></th>\
         <th>OCR</th>\
         </tr>\
         {rows}\
         </table>"
    )
}

// ── CSS ──────────────────────────────────────────────────────────────────────

const CSS: &str = r#"<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
.ssim-pass { font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #d4edda; color: #155724; margin-left: 8px; }
.ssim-fail { font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #f8d7da; color: #721c24; margin-left: 8px; font-weight: bold; }
body {
  font-family: -apple-system, system-ui, sans-serif;
  font-size: 13px;
  color: #222;
  padding: 16px;
}
h2 { font-size: 16px; margin-bottom: 12px; color: #111; }
.summary { color: #555; font-size: 12px; margin-bottom: 8px; }
.score-legend { color: #666; font-size: 11px; margin-bottom: 16px; line-height: 1.6; }
.miss { margin-bottom: 28px; }
.miss h3 { font-size: 13px; margin-bottom: 6px; color: #111; }
table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
th {
  text-align: center; font-size: 11px; font-weight: 600;
  color: #444;
  padding: 4px 6px;
  border-bottom: 2px solid #ccc;
  line-height: 1.4;
}
th.correct { color: #2e7d32; }
th.chosen { color: #c62828; }
th .score { font-weight: 400; font-size: 10px; color: #666; }
td {
  padding: 6px; vertical-align: middle;
  border-bottom: 1px solid #eee;
}
.img-td { text-align: center; }
img.ci {
  height: 48px; image-rendering: pixelated; display: block; margin: 0 auto;
  border: 1px solid #ddd;
  background: #f5f5f5;
}
.sub { font-size: 10px; color: #777; margin-top: 2px; }
.ocr-fix { color: #c62828; }
.num { font-family: monospace; font-size: 12px; text-align: right; white-space: nowrap; }
.num.bad { color: #c62828; font-weight: bold; }
.num.warn { color: #e65100; }
.num.ok { color: #2e7d32; }
.per-word-sizes { font-family: monospace; font-size: 11px; line-height: 1.8; }
.word-size-cell { display: inline-block; padding: 1px 5px; margin: 1px 3px; border-radius: 3px; background: #f5f5f5; border: 1px solid #ddd; }
.word-size-cell.bad { background: #ffebee; border-color: #ef9a9a; }
.word-size-cell.warn { background: #fff3e0; border-color: #ffcc80; }
.ocr-col { text-align: center; font-size: 11px; vertical-align: middle; padding: 4px; }
.char-label { font-size: 14px; font-weight: 600; }
.font-mini { font-size: 9px; color: #888; word-break: break-all; max-width: 100px; display: inline-block; }
.dimmed { color: #bbb; font-size: 10px; }
.ratio { font-size: 11px; color: #888; }
.ssim-compare-block {
  margin: 8px 0 10px 0; padding: 8px; background: #f5f8ff;
  border: 1px solid #ccd; border-radius: 4px;
}
.ssim-compare-table { border-collapse: collapse; width: 100%; }
.ssim-compare-table th {
  text-align: center; font-size: 11px; font-weight: 600;
  padding: 4px 6px; border-bottom: 2px solid #ccc;
}
.ssim-compare-table td {
  padding: 4px 6px; border-bottom: 1px solid #dde; vertical-align: middle;
}
.ssim-compare-table .ssim-label {
  font-size: 10px; font-weight: 600; color: #555; width: 50px; text-align: right;
}
.ssim-compare-img {
  max-width: 100%; image-rendering: pixelated;
  border: 1px solid #ddd; display: block; margin: 2px 0;
}
.tie-break-block {
  margin: 8px 0 10px 0; padding: 8px; background: #fff8f0;
  border: 1px solid #dca; border-radius: 4px;
}
.tie-break-title {
  font-size: 11px; font-weight: 700; color: #c60; margin-bottom: 6px;
}
.tie-winner { background: #e8ffe8; font-weight: 600; }
.tie-loser { background: #fff0f0; }
.scan-line-block {
  margin: 6px 0 10px 0; padding: 8px; background: #f0f4f0;
  border: 1px solid #c0c8c0; border-radius: 4px; overflow-x: auto;
  box-sizing: border-box;
}
.scan-line-label { font-size: 10px; font-weight: 600; color: #555; margin-bottom: 4px; }
.scan-line-img {
  image-rendering: pixelated;
  border: 1px solid #ddd; display: block; margin: 2px 0;
}
</style>"#;

// ── Public API ──────────────────────────────────────────────────────────────

/// Accuracy result from classifying audit entries against ground truth.
pub struct AccuracyResult {
    pub hits: usize,
    pub major_misses: usize,
    pub minor_misses: usize,
    pub ssim_failures: usize,
    pub kept_raster: usize,
    pub compared: usize,
    pub primary_hits: usize,
    pub pct: f64,
}

/// Compute accuracy without generating any HTML or audit I/O.
/// Used by --test for fast scoring.
pub fn compute_accuracy(
    entries: &[AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
    glyph_map: &GlyphMap,
) -> AccuracyResult {
    let classified = classify_entries(entries, gt, dpi, font_catalog, glyph_map);

    let mut hits = 0usize;
    let mut major_misses = 0usize;
    let mut minor_misses = 0usize;
    let mut ssim_failures = 0usize;
    let mut kept_raster = 0usize;

    for ce in &classified {
        match ce.kind {
            MissKind::Hit => hits += 1,
            MissKind::MajorMiss => major_misses += 1,
            MissKind::MinorMiss => minor_misses += 1,
            MissKind::SsimFailure => ssim_failures += 1,
            MissKind::KeptRaster => kept_raster += 1,
            MissKind::NoGroundTruth => {}
        }
    }

    let all_misses = major_misses + minor_misses + ssim_failures;
    let compared = hits + all_misses;
    let major_total = major_misses + ssim_failures;
    let primary_hits = compared - major_total;
    let pct = if compared > 0 {
        primary_hits as f64 / compared as f64 * 100.0
    } else {
        100.0
    };

    AccuracyResult {
        hits,
        major_misses,
        minor_misses,
        ssim_failures,
        kept_raster,
        compared,
        primary_hits,
        pct,
    }
}

pub fn generate_report(
    report_path: &Path,
    audit_root: &Path,
    entries: &[AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
    glyph_map: &GlyphMap,
    meta: &ReportMeta,
) -> Result<(), String> {
    let classified = classify_entries(entries, gt, dpi, font_catalog, glyph_map);

    let mut hits = 0usize;
    let mut major_misses: Vec<&ClassifiedEntry> = Vec::new();
    let mut minor_misses: Vec<&ClassifiedEntry> = Vec::new();
    let mut ssim_failures: Vec<&ClassifiedEntry> = Vec::new();
    let mut kept_raster: Vec<&ClassifiedEntry> = Vec::new();
    let mut total_chars = 0usize;
    let mut corrected_chars = 0usize;

    for ce in &classified {
        // Count OCR corrections
        for cv in &ce.entry.ci_char_votes {
            total_chars += 1;
            if cv.ocr_corrected_from.is_some() {
                corrected_chars += 1;
            }
        }

        match ce.kind {
            MissKind::Hit => hits += 1,
            MissKind::MajorMiss => major_misses.push(ce),
            MissKind::MinorMiss => minor_misses.push(ce),
            MissKind::SsimFailure => ssim_failures.push(ce),
            MissKind::KeptRaster => kept_raster.push(ce),
            MissKind::NoGroundTruth => {} // excluded from hit/miss denominator
        }
    }

    let all_misses = major_misses.len() + minor_misses.len() + ssim_failures.len();
    let compared = hits + all_misses;
    // Primary metric: only major misses count against the score.
    let major_total = major_misses.len() + ssim_failures.len();
    let primary_hits = compared - major_total;
    let pct = if compared > 0 {
        primary_hits as f64 / compared as f64 * 100.0
    } else {
        100.0
    };

    let mut font_data_cache = FontDataCache::new();

    // Build major miss blocks
    let mut major_miss_blocks = String::new();
    for ce in &major_misses {
        major_miss_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            glyph_map,
            &mut font_data_cache,
            dpi,
        ));
    }

    // Build minor miss blocks
    let mut minor_miss_blocks = String::new();
    for ce in &minor_misses {
        minor_miss_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            glyph_map,
            &mut font_data_cache,
            dpi,
        ));
    }

    // Build SSIM failure blocks
    let mut ssim_blocks = String::new();
    for ce in &ssim_failures {
        ssim_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            glyph_map,
            &mut font_data_cache,
            dpi,
        ));
    }

    // Build kept-raster blocks
    let mut raster_blocks = String::new();
    for ce in &kept_raster {
        raster_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            glyph_map,
            &mut font_data_cache,
            dpi,
        ));
    }

    let ssim_section = if !ssim_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#c55;\">\
             ZNCC Failures (correct font, ZNCC rejected)</h2>{ssim_blocks}"
        )
    } else {
        String::new()
    };

    let raster_section = if !raster_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#888;\">\
             Kept Raster ({} lines)</h2>{raster_blocks}",
            kept_raster.len()
        )
    } else {
        String::new()
    };

    let ssim_miss_str = if !ssim_failures.is_empty() {
        format!(" ({} SSIM)", ssim_failures.len())
    } else {
        String::new()
    };

    let raster_str = if !kept_raster.is_empty() {
        format!(" | {} kept raster", kept_raster.len())
    } else {
        String::new()
    };

    let ocr_corr_str = if total_chars > 0 {
        format!(" | OCR corrections: {corrected_chars}/{total_chars}")
    } else {
        String::new()
    };

    // ZNCC percentiles across all lines that have a ZNCC score
    let ssim_percentile_str = {
        let mut ssim_vals: Vec<f32> = entries.iter()
            .filter_map(|e| e.ssim_score)
            .collect();
        ssim_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if ssim_vals.is_empty() {
            String::new()
        } else {
            let n = ssim_vals.len();
            let p50 = ssim_vals[n / 2];
            let p90_idx = (n as f64 * 0.9).ceil() as usize;
            let p90 = ssim_vals[p90_idx.min(n - 1)];
            format!(" | ZNCC p50={p50:.6} p90={p90:.6} (n={n})")
        }
    };

    // Per-character GT font rank statistics
    let gt_rank_str = {
        let mut all_ranks: Vec<usize> = classified.iter()
            .flat_map(|ce| ce.entry.ci_char_votes.iter())
            .filter_map(|cv| cv.gt_font_rank)
            .collect();
        all_ranks.sort();
        if all_ranks.is_empty() {
            String::new()
        } else {
            let n = all_ranks.len();
            let median = all_ranks[n / 2];
            let p90_idx = (n as f64 * 0.9).ceil() as usize;
            let p90 = all_ranks[p90_idx.min(n - 1)];
            let top1 = all_ranks.iter().filter(|&&r| r == 1).count();
            let top10 = all_ranks.iter().filter(|&&r| r <= 10).count();
            format!(
                " | GT char rank: median={median} p90={p90} top1={top1}/{n} ({:.0}%) top10={top10}/{n} ({:.0}%)",
                top1 as f64 / n as f64 * 100.0,
                top10 as f64 / n as f64 * 100.0,
            )
        }
    };

    // Run metadata line
    let elapsed_secs = meta.elapsed.as_secs_f64();
    let elapsed_str = if elapsed_secs >= 60.0 {
        format!("{:.0}m {:.0}s", elapsed_secs / 60.0, elapsed_secs % 60.0)
    } else {
        format!("{elapsed_secs:.1}s")
    };
    let render_opts = {
        let mut parts = vec![format!("scale={}", meta.render_scale)];
        parts.push(format!("aa={}", meta.render_aa));
        if let Some(t) = meta.render_binarize {
            parts.push(format!("binarize={t}"));
        }
        parts.join(", ")
    };
    let meta_str = format!(
        "classifier=<b>{}</b> | render: {} | runtime: <b>{}</b>",
        meta.classifier, render_opts, elapsed_str,
    );

    let html = format!(
        "<!DOCTYPE html>\n\
         <html>\n\
         <head>\n\
         <meta charset=\"utf-8\">\n\
         <title>unprint miss report — {primary_hits}/{compared} ({pct:.1}%)</title>\n\
         </head>\n\
         <body style=\"background: white; color: #222;\">\n\
         {CSS}\n\
         <h2>unprint miss report</h2>\n\
         <div class=\"summary\">{primary_hits}/{compared} correct ({pct:.1}%) — \
         {n_major} major + {n_minor} minor misses{ssim_miss_str}{raster_str}{ocr_corr_str}{ssim_percentile_str}{gt_rank_str}</div>\n\
         <div class=\"summary\">{meta_str}</div>\n\
         <div class=\"score-legend\">\n\
         <b>Score key:</b>\n\
         <b>CI score</b> (per-line) = mean(log(prob)) across characters, \
         weighted by character discriminativeness; \
         <b>higher = better match</b>.\n\
         <b>CI prob</b> (per-character) = calibrated posterior probability \
         via Gaussian kernel over embedding distances; \
         <b>0–1, higher = better</b>.\n\
         <b>ZNCC</b> (per-line) = zero-mean normalized cross-correlation between scanned line \
         and re-render; <b>-1–1, higher = more similar</b>.\n\
         </div>\n\
         <h2>Major Misses ({n_major})</h2>\n\
         {major_miss_blocks}\n\
         <h2 style=\"margin-top:2em; color:#e90;\">Minor Misses ({n_minor})</h2>\n\
         {minor_miss_blocks}\n\
         {ssim_section}\n\
         {raster_section}\n\
         </body>\n\
         </html>",
        n_major = major_misses.len(),
        n_minor = minor_misses.len(),
    );

    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create report directory: {e}"))?;
    }
    std::fs::write(report_path, &html)
        .map_err(|e| format!("Failed to write report: {e}"))?;

    Ok(())
}
