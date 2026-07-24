//! HTML miss report generation.
//!
//! Automatically generated into `<audit_dir>/report.html`.  When
//! `--audit` is also provided, classifies lines as hits/misses against
//! ground truth from the vector PDF.  Without `--audit`, reports all
//! kept-raster lines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use unprint_fonts::ab_glyph::FontRef;
use base64::Engine;
use image::{GrayImage, RgbImage};

use crate::audit::{AuditEntry, Decision};

use crate::char_render;
use crate::ground_truth::{self, GroundTruth};
use crate::font_scan::FontEntry;
use crate::glyph_map::NgramGlyphMap;

/// Metadata about the run, displayed in the report header.
pub struct ReportMeta {
    pub classifier: String,
    pub render_scale: u32,
    pub render_aa: String,
    pub render_binarize: Option<u8>,
    pub elapsed: std::time::Duration,
    pub report_all: bool,
}

// ── Glyph helpers ───────────────────────────────────────────────────────────

/// Shorten a font_key for display.  Now that font_key is the canonical name
/// from make_weight_explicit (not a file path), this is the identity function.
/// Variant tags ("|smcp") are already part of the key and preserved.
fn short_key(key: &str) -> String {
    key.to_string()
}

/// Resolve a glyph_id for a character to a display-friendly font key.
/// Falls back to "glyph#{id}" when the map has no entry.
fn glyph_display_key(glyph_map: &NgramGlyphMap, seq: &[char], glyph_id: usize) -> String {
    glyph_map.fonts_for_glyph(seq, glyph_id)
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

fn rgb_img_to_b64_uri(img: &RgbImage) -> String {
    let mut buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut buf);
    image::ImageEncoder::write_image(
        encoder,
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
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

/// Ink bounding-box midpoint (cx, cy) for a grayscale image.
/// Ink = pixel < 200 (same threshold as geometry_classifier).
/// Returns None for blank images.
fn ink_midpoint(img: &GrayImage) -> Option<(f32, f32)> {
    let (w, h) = img.dimensions();
    let mut x_min = w as i32;
    let mut x_max = -1i32;
    let mut y_min = h as i32;
    let mut y_max = -1i32;
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y).0[0] < 200 {
                if (x as i32) < x_min { x_min = x as i32; }
                if x as i32 > x_max { x_max = x as i32; }
                if (y as i32) < y_min { y_min = y as i32; }
                if y as i32 > y_max { y_max = y as i32; }
            }
        }
    }
    if x_max < 0 || y_max < 0 {
        None
    } else {
        let cx = (x_min as f32 + x_max as f32) * 0.5;
        let cy = (y_min as f32 + y_max as f32) * 0.5;
        Some((cx, cy))
    }
}

fn format_mid_delta(_mid_scan: Option<(f32, f32)>, _mid_ref: Option<(f32, f32)>) -> String {
    // was: ink midpoint delta (Δh Δv) — duplicates geo Δh (h_err is pitch error).
    // Removed to conserve space and avoid "2 delta h" per cell. Geo Δh/Δv in
    // format_geo is the scoring signal; keep only that.
    String::new()
}

fn format_log_prob(prob_x_u: Option<f32>) -> String {
    if let Some(pu) = prob_x_u {
        if pu > 0.0 {
            let lp = pu.ln();
            return format!("<span class='logprob'>{lp:.2}</span>");
        }
    }
    String::new()
}

fn format_geo(h_ll: Option<f32>, v_ll: Option<f32>, _ll: Option<f32>, h_err: Option<f32>, v_err: Option<f32>) -> String {
    // Legacy wrapper — kept for compatibility; new code uses format_char_detail
    let mut parts = Vec::new();
    if let Some(h) = h_ll { parts.push(format!("h {h:.2}")); }
    if let Some(v) = v_ll { parts.push(format!("v {v:.2}")); }
    let score = parts.join(" ");
    let mut deltas = Vec::new();
    if let Some(he) = h_err { deltas.push(format!("Δh{he:.2}")); }
    if let Some(ve) = v_err { deltas.push(format!("Δv{ve:.2}")); }
    if score.is_empty() && deltas.is_empty() { return String::new(); }
    if deltas.is_empty() {
        format!("<span class='geo'>{score}</span>")
    } else if score.is_empty() {
        format!("<span class='geo'>({}px)</span>", deltas.join(" "))
    } else {
        format!("<span class='geo'>{score} ({}px)</span>", deltas.join(" "))
    }
}

fn format_char_detail(
    rank: Option<usize>,
    prob_x_u: Option<f32>,
    glyph_score: Option<f32>,
    h_ll: Option<f32>,
    v_ll: Option<f32>,
    h_err: Option<f32>,
    v_err: Option<f32>,
) -> String {
    // Requested: "rank 1, p=#×u<br>glyph: #<br>midpoint x,y (delta x,ypx)"
    let mut out = String::new();
    if let Some(r) = rank {
        out.push_str(&format!("<span class='font-mini'>rank {r}</span>"));
    }
    if let Some(pu) = prob_x_u {
        if !out.is_empty() { out.push_str(", "); }
        out.push_str(&format!("<span class='num'>p={pu:.1}×u</span>"));
    }
    if out.is_empty() && glyph_score.is_none() && h_ll.is_none() && v_ll.is_none() && h_err.is_none() && v_err.is_none() {
        return String::new();
    }
    // line break between total score and glyph score
    if !out.is_empty() { out.push_str("<br>"); }
    // glyph: raw logit -d²/(2σ²) — not ln(joint prob)
    if let Some(gs) = glyph_score {
        out.push_str(&format!("<span class='logprob'>glyph: {gs:.2}</span>"));
    } else if let Some(pu) = prob_x_u {
        // fallback: if glyph_score missing (old audit), show ln(joint) as approximation
        if pu > 0.0 {
            out.push_str(&format!("<span class='logprob'>glyph: {:.2}</span>", pu.ln()));
        }
    }
    // midpoint scores + deltas
    let has_mid = h_ll.is_some() || v_ll.is_some();
    let has_delta = h_err.is_some() || v_err.is_some();
    if has_mid || has_delta {
        // line break between glyph score and midpoint scores
        if glyph_score.is_some() || prob_x_u.is_some() { out.push_str("<br>"); }
        out.push_str("<span class='geo'>midpoint ");
        if let Some(h) = h_ll { out.push_str(&format!("h {h:.2}")); }
        if let Some(v) = v_ll {
            if h_ll.is_some() { out.push(' '); }
            out.push_str(&format!("v {v:.2}"));
        }
        if !has_mid && has_delta {
            // no h_ll/v_ll, just deltas
            out.push_str("—");
        }
        if has_delta {
            out.push_str(" (");
            let mut d = Vec::new();
            if let Some(he) = h_err { d.push(format!("Δh{he:.2}")); }
            if let Some(ve) = v_err { d.push(format!("Δv{ve:.2}")); }
            out.push_str(&d.join(" "));
            out.push_str("px)");
        }
        out.push_str("</span>");
    }
    out
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

// ── Character selection (mirrors Python pick_interesting_observations) ──────────────

fn pick_interesting_observations<'a>(
    chars: &'a [crate::audit::ObservationVote],
    n_show: usize,
    _n_normal: usize,
) -> Vec<(usize, &'a crate::audit::ObservationVote)> {
    // Rank by the log-probability difference between chosen and ground-truth fonts.
    // Largest absolute difference first — these observations drive the font decision.
    let mut scored: Vec<(usize, &crate::audit::ObservationVote, f32)> = chars.iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let cp = c.chosen_prob?;
            let gp = c.gt_font_prob?;
            // log contribution delta: positive means chosen scores better on this obs
            let delta = cp.max(1e-30).ln() - gp.max(1e-30).ln();
            Some((i, c, delta))
        })
        .collect();
    // Sort by |delta| descending — biggest disagreements first
    scored.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<(usize, &crate::audit::ObservationVote)> = scored.iter()
        .take(n_show)
        .map(|&(i, c, _)| (i, c))
        .collect();

    // If we didn't get enough from the delta sort (e.g. GT probs missing),
    // fill with worst best_prob observations
    if result.len() < n_show {
        let used: std::collections::HashSet<usize> = result.iter().map(|(i, _)| *i).collect();
        let mut by_prob: Vec<(usize, &crate::audit::ObservationVote)> =
            chars.iter().enumerate()
                .filter(|(i, _)| !used.contains(i))
                .collect();
        by_prob.sort_by(|a, b| {
            a.1.best_prob.partial_cmp(&b.1.best_prob).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, c) in by_prob.into_iter().take(n_show - result.len()) {
            result.push((i, c));
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
fn find_crop_png_in(diag_dir: &Path, subdir: &str, crop_index: usize, seq: &[char]) -> Option<PathBuf> {
    let crop_dir = diag_dir.join(subdir);
    let seq_label: String = seq.iter().map(|&c| {
        if c.is_alphanumeric() { format!("{}", c) }
        else { format!("U{:04X}", c as u32) }
    }).collect();
    let path = crop_dir.join(format!("crop_{crop_index:02}_{seq_label}.png"));
    if path.is_file() { Some(path) } else { None }
}



/// Find the font ref glyph PNG for a character in the font_refs directory.
fn find_font_ref_ngram_png(audit_root: &Path, font_entry: &FontEntry, ch: char) -> Option<PathBuf> {
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
    SimilarityFailure,
    KeptRaster,
    NoGroundTruth,
}

impl MissKind {
    fn as_str(&self) -> &'static str {
        match self {
            MissKind::Hit => "hit",
            MissKind::MajorMiss => "major_miss",
            MissKind::MinorMiss => "minor_miss",
            MissKind::SimilarityFailure => "similarity_failure",
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
    _glyph_map: &NgramGlyphMap,
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
                    let ps_match = e.font_key_matched.as_ref().and_then(|fk| {
                        find_font_by_key(font_catalog, fk)
                    }).map_or(false, |fe| {
                        fe.postscript_name == *actual
                    });

                    if ps_match {
                        if e.similarity_pass == Some(false) {
                            MissKind::SimilarityFailure
                        } else {
                            MissKind::Hit
                        }
                    } else {
                        // Font miss — classify as major or minor.
                        // Read identity from both the picked font and the GT font.
                        let picked_path = e.font_key_matched.as_ref()
                            .and_then(|fk| find_font_by_key(font_catalog, fk))
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
                // No ground truth: classify kept-raster, similarity failure, or hit
                let kind = if e.decision == Decision::KeptRaster {
                    MissKind::KeptRaster
                } else if e.similarity_pass == Some(false) {
                    MissKind::SimilarityFailure
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
    glyph_map: &NgramGlyphMap,
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

        // Populate GT text and OCR text comparison
        if let Some(gt) = gt {
            let bbox_px = [e.bbox.x as f32, e.bbox.y as f32,
                           (e.bbox.x + e.bbox.width) as f32,
                           (e.bbox.y + e.bbox.height) as f32];
            e.gt_text = gt.lookup_text(e.page, &bbox_px, dpi).map(|s| s.to_string());
        }
        // OCR text: word_bboxes_raw = original tesseract output (before pflda corrections),
        // word_bboxes = corrected text (used for rendering and comparison).
        let ocr_raw: String = if e.word_bboxes_raw.is_empty() {
            e.word_bboxes.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
        } else {
            e.word_bboxes_raw.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
        };
        let ocr_corrected: String = e.word_bboxes.iter()
            .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        if !ocr_raw.is_empty() {
            e.ocr_text = Some(ocr_raw);
            if let Some(ref gt_t) = e.gt_text {
                // Compare corrected text against GT (not raw OCR).
                // Normalize: collapse whitespace, trim, lowercase.
                let gt_norm: String = gt_t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
                let corrected_norm: String = ocr_corrected.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
                e.ocr_correct = Some(gt_norm == corrected_norm);
            }
        }
    }
}

// ── CI candidate lookup ─────────────────────────────────────────────────────

/// Strip directory and file extension from a font key while preserving the
fn find_correct_ci_candidate(
    entry: &AuditEntry,
    actual_font: &str,
    font_catalog: &[FontEntry],
    _glyph_map: &NgramGlyphMap,
) -> (Option<String>, Option<f32>, Option<usize>) {
    let gt_ps = ground_truth::strip_subset_prefix_str(actual_font);

    // Match CI candidate's PostScript name against GT font.
    // After GT canonicalization, both names are canonical — exact equality.
    // Variant entries carry "PSName|tag" so they won't match base "PSName".
    for (i, c) in entry.font_candidates.iter().enumerate() {
        if let Some(fe) = find_font_by_key(font_catalog, &c.font_key) {
            if fe.postscript_name == gt_ps {
                return (Some(c.font_key.clone()), c.score, Some(i + 1));
            }
        }
    }
    (None, None, None)
}

// ── HTML block generation ───────────────────────────────────────────────────

/// Build a lightweight block for lines where the font matched but OCR got the text wrong.
fn build_ocr_miss_block(
    ce: &ClassifiedEntry,
    audit_root: &Path,
) -> String {
    let entry = ce.entry;
    let text_preview = truncate(&entry.text, 60);
    let matched = entry.font_matched.as_deref().map(short_key).unwrap_or_else(|| "?".into());

    let diag_dir = find_diag_seg_dir(audit_root, entry.page, entry.line_index);

    // Scan line image
    let scan_line_html = if let Some(ref dd) = diag_dir {
        build_scan_line_with_overlays(dd, entry)
    } else {
        String::new()
    };

    // Similarity images (scan vs render vs diff)
    let sim_images_html = if let Some(ref dd) = diag_dir {
        let scan_path = dd.join("ssim_scan.png");
        let render_path = dd.join("ssim_render.png");
        let diff_path = dd.join("ssim_diff.png");
        let scan_uri = file_to_b64_uri(&scan_path);
        let render_uri = file_to_b64_uri(&render_path);
        let diff_uri = file_to_b64_uri(&diff_path);
        match (scan_uri, render_uri) {
            (Some(s), Some(r)) => {
                let diff_row = diff_uri.map(|d| format!(
                    "<tr><td class=\"ssim-label\">Diff</td>\
                     <td colspan=\"2\"><img src=\"{d}\" class=\"ssim-compare-img\"></td></tr>"
                )).unwrap_or_default();
                format!(
                    "<table class=\"ssim-compare\" style=\"margin:0.5em 0;\">\
                     <tr><td class=\"ssim-label\">Scan</td>\
                     <td colspan=\"2\"><img src=\"{s}\" class=\"ssim-compare-img\"></td></tr>\
                     <tr><td class=\"ssim-label\">Render</td>\
                     <td colspan=\"2\"><img src=\"{r}\" class=\"ssim-compare-img\"></td></tr>\
                     {diff_row}\
                     </table>"
                )
            }
            _ => String::new(),
        }
    } else {
        String::new()
    };

    // Character-level diff between GT and OCR text
    let diff_html = match (entry.gt_text.as_deref(), entry.ocr_text.as_deref()) {
        (Some(gt), Some(ocr)) => {
            let gt_chars: Vec<char> = gt.chars().filter(|c| !c.is_whitespace()).collect();
            let ocr_chars: Vec<char> = ocr.chars().filter(|c| !c.is_whitespace()).collect();

            // Simple LCS-based diff for highlighting
            let (gt_marked, ocr_marked) = char_diff_markup(&gt_chars, &ocr_chars);

            format!(
                "<table class=\"ocr-diff\" style=\"margin:0.5em 0; border-collapse:collapse;\">\
                 <tr><td style=\"padding:2px 8px; color:#888;\">GT</td>\
                 <td style=\"padding:2px 8px; font-family:monospace;\">{}</td></tr>\
                 <tr><td style=\"padding:2px 8px; color:#888;\">OCR</td>\
                 <td style=\"padding:2px 8px; font-family:monospace;\">{}</td></tr>\
                 </table>",
                gt_marked, ocr_marked
            )
        }
        _ => String::new(),
    };

    // ZNCC info
    let sim_html = match (entry.similarity_score, entry.similarity_pass) {
        (Some(score), Some(pass)) => {
            let cls = if pass { "ssim-pass" } else { "ssim-fail" };
            let label = if pass { "pass" } else { "FAIL" };
            format!(" <span class=\"{cls}\">ZNCC {score:.4} ({label})</span>")
        }
        _ => String::new(),
    };

    format!(
        "<div class=\"miss\" style=\"border-left: 3px solid #e90; padding-left: 1em; margin-bottom: 1em;\">\
         <h3>p{}:L{} — \"{}\" [font: {}]{}</h3>\
         {}\
         {}\
         {}\
         </div>",
        entry.page, entry.line_index, text_preview, matched, sim_html,
        scan_line_html, sim_images_html, diff_html,
    )
}

/// Produce character-level diff markup: matching chars are plain,
/// mismatched/inserted/deleted chars are highlighted.
fn char_diff_markup(gt: &[char], ocr: &[char]) -> (String, String) {
    // LCS table
    let m = gt.len();
    let n = ocr.len();
    let mut dp = vec![vec![0u16; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if gt[i - 1] == ocr[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Backtrack to produce aligned sequences
    let mut gt_out = String::new();
    let mut ocr_out = String::new();
    let (mut i, mut j) = (m, n);
    let mut gt_parts: Vec<(char, bool)> = Vec::new();
    let mut ocr_parts: Vec<(char, bool)> = Vec::new();

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && gt[i - 1] == ocr[j - 1] {
            gt_parts.push((gt[i - 1], false));
            ocr_parts.push((ocr[j - 1], false));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            // Insertion in OCR
            ocr_parts.push((ocr[j - 1], true));
            j -= 1;
        } else {
            // Deletion from GT
            gt_parts.push((gt[i - 1], true));
            i -= 1;
        }
    }

    gt_parts.reverse();
    ocr_parts.reverse();

    for (ch, is_diff) in &gt_parts {
        if *is_diff {
            gt_out.push_str(&format!("<span style=\"background:#fdd; color:#900;\">{}</span>",
                                     html_escape_char(*ch)));
        } else {
            gt_out.push_str(&html_escape_char(*ch));
        }
    }
    for (ch, is_diff) in &ocr_parts {
        if *is_diff {
            ocr_out.push_str(&format!("<span style=\"background:#dfd; color:#090;\">{}</span>",
                                      html_escape_char(*ch)));
        } else {
            ocr_out.push_str(&html_escape_char(*ch));
        }
    }

    (gt_out, ocr_out)
}

fn html_escape_char(c: char) -> String {
    match c {
        '<' => "&lt;".to_string(),
        '>' => "&gt;".to_string(),
        '&' => "&amp;".to_string(),
        '"' => "&quot;".to_string(),
        _ => c.to_string(),
    }
}

/// Returns (html_block, chosen_zncc, gt_zncc).
fn build_miss_block(
    ce: &ClassifiedEntry,
    audit_root: &Path,
    font_catalog: &[FontEntry],
    glyph_map: &NgramGlyphMap,
    font_data_cache: &mut FontDataCache,
    dpi: u32,
) -> (String, Option<f32>, Option<f32>) {
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
                .font_candidates
                .first()
                .and_then(|c| find_font_by_key(font_catalog, &c.font_key))
                .or_else(|| find_font_in_catalog(font_catalog, m))
        });

    // Find diag_seg_dir for this line
    let diag_dir = find_diag_seg_dir(audit_root, entry.page, entry.line_index);

    // Similarity comparison block — always render for side-by-side comparison
    let (gt_render_uri, gt_diff_uri, gt_sim) =
        render_correct_font_comparison(entry, correct_fe, font_data_cache, diag_dir.as_deref());
    // Compute font sizes for comparison
    let gt_font_size_pt = ce.gt_font_size_pt;
    let inferred_size = chosen_fe.and_then(|fe| {
        let data = font_data_cache.load(&fe.path)?;
        compute_inferred_font_size(data, &entry.word_bboxes, &fe.variant_tag, fe.variations.as_deref(), dpi)
    });
    let unprint_font_size_pt = inferred_size.as_ref().map(|s| s.median_pt);

    let sim_compare_html = if let Some(ref dd) = diag_dir {
        build_similarity_block(
            dd, &actual_font, &matched, entry.similarity_score,
            gt_render_uri.as_deref(), gt_diff_uri.as_deref(),
            gt_sim,
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

    // Ligature vs plain segmentation visual comparison
    let lig_compare_html = if let Some(ref dd) = diag_dir {
        build_lig_comparison_block(dd, entry)
    } else {
        String::new()
    };

    // Scan line image with word bbox + segmentation overlays
    let scan_line_html = if let Some(ref dd) = diag_dir {
        build_scan_line_with_overlays(dd, entry)
    } else {
        String::new()
    };

    // Similarity (ZNCC) info
    let sim_html = match (entry.similarity_score, entry.similarity_pass) {
        (Some(score), Some(pass)) => {
            let cls = if pass { "ssim-pass" } else { "ssim-fail" };
            let label = if pass { "pass" } else { "FAIL" };
            format!(" <span class=\"{cls}\">ZNCC {score:.10} ({label})</span>")
        }
        _ => String::new(),
    };

    // Per-character comparison table (skip for similarity-only failures unless ligature path picked)
    let obs_table_html = if entry.obs_votes.is_empty() || (ce.kind == MissKind::SimilarityFailure && entry.seg_winner.is_none()) {
        String::new()
    } else {
        let obs_to_show = pick_interesting_observations(&entry.obs_votes, 6, 0);
        build_observation_table(
            entry,
            &obs_to_show,
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
            "crops",
        )
    };

    // Per-character comparison table for the alternate (lig) segmentation path
    let obs_table_lig_html = if entry.obs_votes_lig.is_empty() || entry.seg_winner.is_none() {
        String::new()
    } else {
        let winner = entry.seg_winner.as_deref().unwrap_or("?");
        let alt_label = if winner == "ligature" { "plain" } else { "ligature" };
        let alt_font_key = entry.font_candidates_lig.first()
            .map(|c| c.font_key.as_str()).unwrap_or("?");
        let alt_font = short_key(alt_font_key);
        let alt_chosen_fe = font_catalog.iter().find(|fe| fe.font_key() == alt_font_key);
        // GT rank/score in the alt path's candidate list (not winner's)
        let alt_gt = if let Some(ref af) = ce.actual_font {
            let gt_ps = ground_truth::strip_subset_prefix_str(af);
            entry.font_candidates_lig.iter().enumerate()
                .find(|(_, c)| {
                    find_font_by_key(font_catalog, &c.font_key)
                        .map_or(false, |fe| fe.postscript_name == gt_ps)
                })
                .map(|(i, c)| (Some(i + 1), c.score))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };
        let obs_to_show = pick_interesting_observations(&entry.obs_votes_lig, 6, 0);
        let table = build_observation_table(
            entry,
            &obs_to_show,
            audit_root,
            correct_fe,
            alt_chosen_fe,
            &actual_font,
            &alt_font,
            alt_gt.0,
            alt_gt.1,
            font_data_cache,
            diag_dir.as_deref(),
            font_catalog,
            glyph_map,
            "crops_alt",
        );
        if table.is_empty() {
            String::new()
        } else {
            format!(
                "<div class=\"seg-alt-obs\"><b>Alternate path ({alt_label}) \u{2192} {alt_font}</b>{table}</div>",
            )
        }
    };

    let text_preview = truncate(&entry.text, 60);
    let ocr_fail_tag = if entry.ocr_correct == Some(false) {
        " <span class=\"ocr-wrong-badge\">OCR WRONG</span>"
    } else {
        ""
    };
    let fast_path_tag = if entry.fast_path {
        " <span class=\"fast-path-badge\">fast-path</span>"
    } else {
        ""
    };
    let miss_kind_label = match ce.kind {
        MissKind::MajorMiss => {
            // Show expected→got summary for font misses
            format!(" [MAJOR: expected {}, got {}]{fast_path_tag}{ocr_fail_tag}", actual_font, matched)
        },
        MissKind::MinorMiss => {
            format!(" [minor: expected {}, got {}]{fast_path_tag}{ocr_fail_tag}", actual_font, matched)
        },
        MissKind::SimilarityFailure => format!(" [ZNCC failure]{fast_path_tag}{ocr_fail_tag}"),
        MissKind::KeptRaster => format!(" [kept raster]{ocr_fail_tag}"),
        _ => ocr_fail_tag.to_string(),
    };

    // Segmentation path comparison (ligature vs plain)
    let seg_path_html = if let Some(ref winner) = entry.seg_winner {
        if !entry.font_candidates_lig.is_empty() {
            let (lig_font, lig_score, plain_font, plain_score) = if winner == "ligature" {
                // font_candidates = ligature (winner), font_candidates_lig = plain (loser)
                let lf = entry.font_candidates.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ls = entry.font_candidates.first().and_then(|c| c.score);
                let pf = entry.font_candidates_lig.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ps = entry.font_candidates_lig.first().and_then(|c| c.score);
                (lf, ls, pf, ps)
            } else {
                // font_candidates = plain (winner), font_candidates_lig = ligature (loser)
                let pf = entry.font_candidates.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ps = entry.font_candidates.first().and_then(|c| c.score);
                let lf = entry.font_candidates_lig.first()
                    .map(|c| short_key(&c.font_key)).unwrap_or_else(|| "?".into());
                let ls = entry.font_candidates_lig.first().and_then(|c| c.score);
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

    // OCR override table — separate from font-matching obs table.
    // OCR corrections table — from audit ocr_corrections data.
    let ocr_override_html = {
        if entry.ocr_corrections.is_empty() {
            String::new()
        } else {
            let mut rows = String::new();
            for oc in &entry.ocr_corrections {
                let ocr_p_str = oc.ocr_p.map(|p| format!("{:.4}", p)).unwrap_or("—".into());
                let ratio_str = if oc.ratio.is_infinite() { "∞".into() } else { format!("{:.1}", oc.ratio) };
                rows.push_str(&format!(
                    "<tr><td>{}</td><td class=\"mono\">{}</td><td class=\"mono\">{}</td>                     <td>{:.4}</td><td>{}</td><td>{}</td></tr>",
                    oc.char_pos, oc.ocr_char, oc.replacement,
                    oc.replacement_p, ocr_p_str, ratio_str,
                ));
            }
            format!(
                "<details open><summary>OCR corrections (PFLDA)</summary>\
                 <table class=\"obs-table\"><thead><tr>\
                 <th>Pos</th><th>OCR</th><th>Replaced</th><th>PFLDA p</th><th>OCR p</th>                 <th>Ratio</th>\
                 </tr></thead><tbody>{}</tbody></table></details>",
                rows
            )
        }
    };

    let html = format!(
        "<div class=\"miss\">\
         <h3>p{}:L{}{}{}  </h3>\
         <div class=\"line-text-preview\">\"{}\"</div>\
         {}\
         {}\
         {}\
         {}\
         {}\
         {}\
         {}\
         {}\
         </div>",
        entry.page, entry.line_index, miss_kind_label, sim_html,
        text_preview,
        seg_path_html, lig_compare_html, scan_line_html, sim_compare_html, tie_break_html, obs_table_html,
        obs_table_lig_html, ocr_override_html,
    );
    (html, entry.similarity_score, gt_sim)
}

/// Build ligature vs plain segmentation visual comparison.
/// Finds word_*/seg_plain/word_crop.png and word_*/seg_lig/word_crop.png
/// in the diag directory and renders them side by side when both exist.
/// Also shows ZNCC render/diff images for both paths when available.
fn build_lig_comparison_block(diag_dir: &Path, entry: &AuditEntry) -> String {
    if entry.seg_winner.is_none() || entry.font_candidates_lig.is_empty() {
        return String::new();
    }
    let winner = entry.seg_winner.as_deref().unwrap_or("?");
    let (winner_label, loser_label) = if winner == "plain" {
        ("Plain ✓", "Ligature")
    } else {
        ("Plain", "Ligature ✓")
    };

    let mut word_dirs: Vec<_> = match std::fs::read_dir(diag_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name().to_string_lossy().starts_with("word_")
                    && e.path().is_dir()
            })
            .collect(),
        Err(_) => return String::new(),
    };
    word_dirs.sort_by_key(|d| d.file_name().to_string_lossy().to_string());

    let mut rows = String::new();
    for wd in &word_dirs {
        let wpath = wd.path();
        let plain_img = wpath.join("seg_plain").join("word_crop.png");
        let lig_img = wpath.join("seg_lig").join("word_crop.png");
        if !plain_img.exists() || !lig_img.exists() {
            continue;
        }
        let plain_uri = match file_to_b64_uri(&plain_img) {
            Some(u) => u,
            None => continue,
        };
        let lig_uri = match file_to_b64_uri(&lig_img) {
            Some(u) => u,
            None => continue,
        };
        let dirname = wd.file_name().to_string_lossy().to_string();
        let word_label = dirname.splitn(3, '_').nth(2).unwrap_or(&dirname);

        rows.push_str(&format!(
            "<tr>\
             <td style=\"vertical-align:middle;font-size:11px;font-weight:600;padding:4px 6px;\">\
             {word_label}</td>\
             <td style=\"padding:2px;\"><img src=\"{plain_uri}\" \
             style=\"max-width:100%;image-rendering:pixelated;\"/></td>\
             <td style=\"padding:2px;\"><img src=\"{lig_uri}\" \
             style=\"max-width:100%;image-rendering:pixelated;\"/></td></tr>",
        ));
    }

    // ZNCC comparison: winner path renders from diag_dir, alt path from ssim_alt/
    let mut zncc_rows = String::new();
    let winner_render = diag_dir.join("ssim_render.png");
    let winner_diff = diag_dir.join("ssim_diff.png");
    let alt_render = diag_dir.join("ssim_alt").join("ssim_render.png");
    let alt_diff = diag_dir.join("ssim_alt").join("ssim_diff.png");

    if winner_render.exists() && alt_render.exists() {
        if let (Some(wr_uri), Some(ar_uri)) = (file_to_b64_uri(&winner_render), file_to_b64_uri(&alt_render)) {
            let (left_uri, right_uri) = if winner == "plain" { (wr_uri, ar_uri) } else { (ar_uri, wr_uri) };
            zncc_rows.push_str(&format!(
                "<tr>\
                 <td style=\"vertical-align:middle;font-size:11px;font-weight:600;padding:4px 6px;\">\
                 ZNCC render</td>\
                 <td style=\"padding:2px;\"><img src=\"{left_uri}\" \
                 style=\"max-width:100%;image-rendering:pixelated;\"/></td>\
                 <td style=\"padding:2px;\"><img src=\"{right_uri}\" \
                 style=\"max-width:100%;image-rendering:pixelated;\"/></td></tr>",
            ));
        }
    }
    if winner_diff.exists() && alt_diff.exists() {
        if let (Some(wd_uri), Some(ad_uri)) = (file_to_b64_uri(&winner_diff), file_to_b64_uri(&alt_diff)) {
            let (left_uri, right_uri) = if winner == "plain" { (wd_uri, ad_uri) } else { (ad_uri, wd_uri) };
            zncc_rows.push_str(&format!(
                "<tr>\
                 <td style=\"vertical-align:middle;font-size:11px;font-weight:600;padding:4px 6px;\">\
                 ZNCC diff</td>\
                 <td style=\"padding:2px;\"><img src=\"{left_uri}\" \
                 style=\"max-width:100%;image-rendering:pixelated;\"/></td>\
                 <td style=\"padding:2px;\"><img src=\"{right_uri}\" \
                 style=\"max-width:100%;image-rendering:pixelated;\"/></td></tr>",
            ));
        }
    }

    let all_rows = format!("{rows}{zncc_rows}");
    if all_rows.is_empty() {
        return String::new();
    }

    format!(
        "<div class=\"seg-compare\">\
         <b>Segmentation comparison</b> (winner: {winner})\
         <table style=\"border-collapse:collapse;margin-top:4px;\">\
         <tr><th></th>\
         <th style=\"font-size:11px;padding:2px 6px;\">{winner_label}</th>\
         <th style=\"font-size:11px;padding:2px 6px;\">{loser_label}</th></tr>\
         {all_rows}\
         </table></div>",
        winner_label = winner_label,
        loser_label = loser_label,
        all_rows = all_rows,
    )
}

/// Build scan line image with word bbox and segmentation path overlays.
/// Replicates the Python report `render_scan_line_with_word_boxes()`:
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
        // Fall back to ssim_scan.png (legacy file name for similarity scan)
        let scan_fallback_path = diag_dir.join("ssim_scan.png");
        if !scan_fallback_path.exists() {
            return String::new();
        }
        match file_to_b64_uri(&scan_fallback_path) {
            Some(u) => (u, entry.bbox.x, entry.bbox.y),
            None => return String::new(),
        }
    };

    // Get image dimensions for the container
    let (img_w, img_h) = if scan_path.exists() {
        image::image_dimensions(&scan_path)
            .unwrap_or((entry.bbox.width, entry.bbox.height))
    } else {
        let scan_fallback_path = diag_dir.join("ssim_scan.png");
        image::image_dimensions(&scan_fallback_path)
            .unwrap_or((entry.bbox.width, entry.bbox.height))
    };

    let scale = 1u32; // 1× native resolution
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
            "<rect x=\"{bx}\" y=\"{by}\" width=\"{bw}\" height=\"{bh}\" \
             fill=\"none\" stroke=\"rgba(255,160,0,0.8)\" stroke-width=\"1\" stroke-dasharray=\"3,2\"/>"
        ));
    }

    // Final post-processed boxes — dashed cyan
    for wb in &entry.word_bboxes {
        let bx = (wb.x.saturating_sub(crop_x)) * scale;
        let by = (wb.y.saturating_sub(crop_y)) * scale + margin_top;
        let bw = wb.width * scale;
        let bh = wb.height * scale;
        overlays.push_str(&format!(
            "<rect x=\"{bx}\" y=\"{by}\" width=\"{bw}\" height=\"{bh}\" \
             fill=\"none\" stroke=\"rgba(0,200,220,0.85)\" stroke-width=\"1\" stroke-dasharray=\"4,2\"/>"
        ));

        // Pixel-scale ruler at top of each final word box
        let wb_px_w = wb.width;
        let mut col = 0u32;
        while col <= wb_px_w {
            let sx = bx + col * scale + scale / 2;
            if col % 10 == 0 {
                // Major tick + label
                overlays.push_str(&format!(
                    "<line x1=\"{sx}\" y1=\"{}\" x2=\"{sx}\" y2=\"{}\" \
                     stroke=\"rgba(140,140,140,0.7)\" stroke-width=\"1\"/>\
                     <text x=\"{sx}\" y=\"{}\" font-size=\"7\" fill=\"rgba(120,120,120,0.8)\" \
                     text-anchor=\"start\" transform=\"rotate(-90,{sx},{})\">{col}</text>",
                    by.saturating_sub(6), by, by.saturating_sub(9), by.saturating_sub(9)
                ));
            } else {
                // Minor tick
                overlays.push_str(&format!(
                    "<line x1=\"{sx}\" y1=\"{}\" x2=\"{sx}\" y2=\"{}\" \
                     stroke=\"rgba(180,180,180,0.6)\" stroke-width=\"1\"/>",
                    by.saturating_sub(3), by
                ));
            }
            col += 5;
        }
    }

    // Segmentation paths from audit entry word_segmentation
    for ws in &entry.word_segmentation {
        if ws.image_w == 0 || ws.image_h == 0 {
            continue;
        }
        let matching_wb = match entry.word_bboxes.get(ws.source_word_idx) {
            Some(wb) => wb,
            None => continue,
        };

        let wx = (matching_wb.x.saturating_sub(crop_x)) * scale;
        let wy = (matching_wb.y.saturating_sub(crop_y)) * scale + margin_top;
        let wb_h = matching_wb.height * scale;

        let sx_f = matching_wb.width as f64 / ws.image_w as f64 * scale as f64;
        let sy_f = matching_wb.height as f64 / ws.image_h as f64 * scale as f64;

        let label_y = wy + wb_h + 2;

        // Whitespace splits — blue vertical lines
        for &col in &ws.ws_splits {
            let cx = wx + (col as f64 * sx_f) as u32;
            overlays.push_str(&format!(
                "<line x1=\"{cx}\" y1=\"{wy}\" x2=\"{cx}\" y2=\"{}\" \
                 stroke=\"rgba(40,100,220,0.8)\" stroke-width=\"1\"/>\
                 <text x=\"{}\" y=\"{label_y}\" font-size=\"7\" \
                 fill=\"rgba(40,100,220,0.9)\">{col}</text>",
                wy + wb_h, cx.saturating_sub(6)
            ));
        }

        // Seam paths — magenta diagonal paths (one x per row)
        // seam_paths now includes candidate (unused) paths too; only draw accepted ones.
        let seam_split_set: std::collections::HashSet<u32> = ws.seam_splits.iter().copied().collect();
        for (col_key, path) in &ws.seam_paths {
            if !seam_split_set.contains(col_key) { continue; }
            for entry in path.iter() {
                let col_px = entry[1];
                let row_idx = entry[0];
                let px_x = wx + (col_px as f64 * sx_f) as u32;
                let px_y = wy + (row_idx as f64 * sy_f) as u32;
                overlays.push_str(&format!(
                    "<rect x=\"{px_x}\" y=\"{px_y}\" width=\"1\" height=\"1\" \
                     fill=\"rgba(255,0,200,0.8)\"/>",
                ));
            }
            // Column label
            let cx = wx + (*col_key as f64 * sx_f) as u32;
            overlays.push_str(&format!(
                "<text x=\"{}\" y=\"{label_y}\" font-size=\"7\" \
                 fill=\"rgba(255,0,200,0.9)\">{col_key}</text>",
                cx.saturating_sub(6)
            ));
        }

        // Seam splits without paths — magenta vertical lines
        let seam_path_keys: std::collections::HashSet<u32> = ws.seam_paths.keys().copied().collect();
        for &col in &ws.seam_splits {
            if !seam_path_keys.contains(&col) {
                let cx = wx + (col as f64 * sx_f) as u32;
                overlays.push_str(&format!(
                    "<line x1=\"{cx}\" y1=\"{wy}\" x2=\"{cx}\" y2=\"{}\" \
                     stroke=\"rgba(255,0,200,0.7)\" stroke-width=\"1\"/>\
                     <text x=\"{}\" y=\"{label_y}\" font-size=\"7\" \
                     fill=\"rgba(255,0,200,0.9)\">{col}</text>",
                    wy + wb_h, cx.saturating_sub(6)
                ));
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
         <div style=\"width:100%;position:relative;\">\
         <svg viewBox=\"0 0 {canvas_w} {canvas_h}\" style=\"width:100%;height:auto;display:block;\" xmlns=\"http://www.w3.org/2000/svg\">\
         <image href=\"{img_uri}\" x=\"0\" y=\"{margin_top}\" width=\"{cw}\" height=\"{ch}\" style=\"image-rendering:pixelated;\"/>\
         {overlays}\
         </svg></div>\
         {seg_stats}\
         </div>",
        cw = img_w * scale,
        ch = img_h * scale,
    )
}

fn build_seg_stats(_diag_dir: &Path, entry: &AuditEntry) -> String {
    if entry.word_segmentation.is_empty() {
        return String::new();
    }

    // Build word text → x position map for left-to-right ordering
    let word_x_map: HashMap<&str, u32> = entry
        .word_bboxes
        .iter()
        .map(|wb| (wb.text.as_str(), wb.x))
        .collect();

    let mut seg_parts: Vec<(u32, String)> = Vec::new();

    for ws in &entry.word_segmentation {
        let wtext = &ws.word_text;
        let n_exp = ws.n_chars_expected.to_string();
        let n_got = ws.n_segments_produced.to_string();

        let word_x = word_x_map.get(wtext.as_str()).copied().unwrap_or(999999u32);
        let mut info = format!("&quot;{wtext}&quot; {n_got}/{n_exp}");
        if ws.mismatch {
            info.push_str(" \u{26a0}");
        }
        let mut tags = Vec::new();
        let nvp = ws.ws_splits.len();
        let nseam = ws.seam_splits.len();
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

    let font_data = match font_data_cache.load(&fe.path) {
        Some(d) => d,
        None => return (None, None, None),
    };

    // Load the scan crop saved during the chosen font's verification
    let scan_path = dd.join("ssim_scan.png");
    let scan_gray = match image::open(&scan_path).ok().map(|d| d.to_luma8()) {
        Some(img) => img,
        None => return (None, None, None),
    };

    // Build TextRegions from word_bboxes — these already contain pflda-corrected
    // text (corrected_words written into audit in main.rs), so both the GT font
    // render and the chosen font render use the identical corrected text.
    let words: Vec<crate::ocr::TextRegion> = entry.word_bboxes.iter().map(|wb| {
        crate::ocr::TextRegion {
            text: wb.text.clone(),
            x: wb.x, y: wb.y,
            width: wb.width, height: wb.height,
            font_size_pt: 0.0, confidence: wb.confidence,
            level: 5, block_num: 0, par_num: 0, line_num: 0, word_num: 0,
        }
    }).collect();

    // Same pipeline as the chosen font — render, ZNCC, ink-crop for display
    // For GT comparison renders, allow ligatures so "fi" can shape correctly
    let vr = crate::verify::verify_text_region(
        &scan_gray, font_data, &entry.text, &words,
        entry.bbox.x, entry.bbox.y,
        fe.glyph_overrides.as_ref().map(|v| v.as_slice()),
        &fe.variant_tag, fe.variations.as_deref(),
        true,
        None, None,
    );

    let render_uri = vr.render_ink.as_ref().map(|r| img_to_b64_uri(r));
    let diff_uri = vr.diff.as_ref().map(|d| rgb_img_to_b64_uri(d));
    let score = if vr.score > 0.0 { Some(vr.score) } else { None };

    (render_uri, diff_uri, score)
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
                true,
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


fn build_similarity_block(
    diag_dir: &Path,
    correct_font: &str,
    chosen_font: &str,
    sim_score: Option<f32>,
    correct_render_uri: Option<&str>,
    correct_diff_uri: Option<&str>,
    correct_sim: Option<f32>,
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

    let sim_str = sim_score
        .map(|s| format!("{s:.10}"))
        .unwrap_or_else(|| "—".into());

    let correct_sim_str = correct_sim
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
         <td class=\"correct\">{correct_sim_str}</td>\
         <td class=\"chosen\">{sim_str}</td></tr>\
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
            "<td class=\"{}\">{:.6}</td>", class, tc.similarity_score
        ));
    }
    rows.push_str("</tr>");

    format!(
        "<div class=\"tie-break-block\">\
         <div class=\"tie-break-title\">Font Tie-Break ({} candidates, ZNCC decides)</div>\
         <table class=\"ssim-compare-table\">{}</table></div>",
        entry.tie_candidates.len(), rows
    )
}

fn build_observation_table(
    entry: &AuditEntry,
    obs_to_show: &[(usize, &crate::audit::ObservationVote)],
    audit_root: &Path,
    correct_fe: Option<&FontEntry>,
    chosen_fe: Option<&FontEntry>,
    correct_font_name: &str,
    chosen_font_name: &str,
    gt_rank: Option<usize>,
    gt_score: Option<f32>,
    font_data_cache: &mut FontDataCache,
    diag_dir: Option<&Path>,
    _font_catalog: &[FontEntry],
    _glyph_map: &NgramGlyphMap,
    crop_dir_name: &str,
) -> String {
    let mut rows = String::new();

    for &(_idx, cv) in obs_to_show {
        let seq: &[char] = &cv.seq;

        // Crop image + midpoint
        let (crop_uri, crop_mid) = if let Some(dd) = diag_dir {
            if let Some(p) = find_crop_png_in(dd, crop_dir_name, cv.crop_index, &cv.seq) {
                let uri = file_to_b64_uri(&p);
                let mid = image::open(&p).ok().map(|i| i.to_luma8()).and_then(|im| ink_midpoint(&im));
                (uri, mid)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Correct font reference glyph
        let mut correct_ref_uri: Option<String> = None;
        let mut correct_mid: Option<(f32, f32)> = None;
        if let Some(fe) = correct_fe {
            if let Some(data) = font_data_cache.load(&fe.path) {
                if let Ok(mut font) = FontRef::try_from_slice(data) {
                    if let Some(ref vars) = fe.variations {
                        use unprint_fonts::ab_glyph::VariableFont;
                        for (tag, val) in vars {
                            let _ = font.set_variation(tag, *val);
                        }
                    }
                    let gid_overrides: Vec<Option<unprint_fonts::ab_glyph::GlyphId>> = seq.iter().map(|c| {
                        fe.glyph_overrides.as_ref()
                            .and_then(|ovs| ovs.iter().find(|(gc, _)| *gc == *c).map(|(_, g)| unprint_fonts::ab_glyph::GlyphId(*g)))
                    }).collect();
                    if let Some(img) = char_render::render_ngram_fresh(&font, seq, &gid_overrides, &char_render::RenderParams::default()) {
                        correct_mid = ink_midpoint(&img);
                        correct_ref_uri = Some(img_to_b64_uri(&img));
                    }
                }
            }
        }

        // Chosen font reference glyph
        let mut chosen_ref_uri: Option<String> = None;
        let mut chosen_mid: Option<(f32, f32)> = None;
        if let Some(fe) = chosen_fe {
            if seq.len() == 1 {
                if let Some(path) = find_font_ref_ngram_png(audit_root, fe, seq[0]) {
                    chosen_ref_uri = file_to_b64_uri(&path);
                    if let Ok(img) = image::open(&path) {
                        chosen_mid = ink_midpoint(&img.to_luma8());
                    }
                }
            }
            if chosen_ref_uri.is_none() {
                if let Some(data) = font_data_cache.load(&fe.path) {
                    if let Ok(mut font) = FontRef::try_from_slice(data) {
                        if let Some(ref vars) = fe.variations {
                            use unprint_fonts::ab_glyph::VariableFont;
                            for (tag, val) in vars {
                                let _ = font.set_variation(tag, *val);
                            }
                        }
                        let gid_overrides: Vec<Option<unprint_fonts::ab_glyph::GlyphId>> = seq.iter().map(|c| {
                            fe.glyph_overrides.as_ref()
                                .and_then(|ovs| ovs.iter().find(|(gc, _)| *gc == *c).map(|(_, g)| unprint_fonts::ab_glyph::GlyphId(*g)))
                        }).collect();
                        if let Some(img) = char_render::render_ngram_fresh(&font, seq, &gid_overrides, &char_render::RenderParams::default()) {
                            chosen_mid = ink_midpoint(&img);
                            chosen_ref_uri = Some(img_to_b64_uri(&img));
                        }
                    }
                }
            }
        }

        let (chosen_win_class, correct_win_class) = match (cv.chosen_prob, cv.gt_font_prob) {
            (Some(cp), Some(gp)) => {
                if cp > gp {
                    ("prob-win", "prob-lose")
                } else if gp > cp {
                    ("prob-lose", "prob-win")
                } else {
                    ("", "")
                }
            }
            _ => ("", ""),
        };

        let chosen_detail = format_char_detail(
            cv.chosen_rank, cv.chosen_prob, cv.chosen_glyph_score,
            cv.chosen_geo_h_ll, cv.chosen_geo_v_ll,
            cv.chosen_geo_h_err, cv.chosen_geo_v_err
        );
        let correct_detail = format_char_detail(
            cv.gt_font_rank, cv.gt_font_prob, cv.gt_glyph_score,
            cv.gt_geo_h_ll, cv.gt_geo_v_ll,
            cv.gt_geo_h_err, cv.gt_geo_v_err
        );

        let chosen_score_label = if !chosen_detail.is_empty() {
            format!("<div class='sub {chosen_win_class}'>{chosen_detail}</div>")
        } else {
            String::new()
        };

        let correct_score_label = if !correct_detail.is_empty() {
            format!("<div class='sub {correct_win_class}'>{correct_detail}</div>")
        } else {
            String::new()
        };

        // OCR letter for this observation — show corrected char, and original if it was pflda-replaced
        let ocr_label = {
            let seq_str: String = cv.seq.iter().collect();
            let mut esc = seq_str.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
            if esc.trim().is_empty() {
                esc = "·".to_string();
            }
            if let Some(orig) = cv.ocr_corrected_from {
                let orig_esc = match orig {
                    '<' => "&lt;".to_string(),
                    '>' => "&gt;".to_string(),
                    '&' => "&amp;".to_string(),
                    _ => orig.to_string(),
                };
                if orig_esc != esc {
                    format!("<span class='ocr-orig'>{orig_esc}</span>→<b>{esc}</b>")
                } else {
                    esc
                }
            } else {
                esc
            }
        };

        rows.push_str(&format!(
            "<tr>\
             <td class=\"ocr-col\"><span class=\"char-label\">{ocr_label}</span></td>\
             <td class=\"img-td\">{}</td>\
             <td class=\"img-td {correct_win_class}\"><div class=\"img-stat\">{}{}</div></td>\
             <td class=\"img-td {chosen_win_class}\"><div class=\"img-stat\">{}{}</div></td>\
             </tr>",
            img_td(crop_uri.as_deref()),
            img_td(correct_ref_uri.as_deref()),
            correct_score_label,
            img_td(chosen_ref_uri.as_deref()),
            chosen_score_label,
        ));
    }

    // Column headers
    // Column headers
    let rank_str = match (gt_rank, gt_score) {
        (Some(r), Some(s)) => format!("font #{r}, score {s:.10}"),
        _ => "not scored".into(),
    };

    // The chosen font is CI candidate #1 from the appropriate path
    let path_candidates = if crop_dir_name == "crops_alt" {
        &entry.font_candidates_lig
    } else {
        &entry.font_candidates
    };
    let chosen_rank_info = path_candidates
        .first()
        .map(|c| {
            match c.score {
                Some(s) => format!("font #1, score {:.10}", s),
                None => "font #1".into(),
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
         <th>OCR</th>\
         <th>Scan</th>\
         <th class=\"correct\">Correct: {correct_font_name}<br><span class='score'>{rank_str}</span></th>\
         <th class=\"chosen\">Unscan pick: {display_chosen}<br><span class='score'>{chosen_rank_info}</span></th>\
         </tr>\
         {rows}\
         </table>"
    )
}

// ── CSS ──────────────────────────────────────────────────────────────────────

const CSS: &str = r#"<style>
@page { size: 841mm 1189mm landscape; margin: 10mm; }
@media print { body { width: 100%; } }
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
  padding: 2px 4px; vertical-align: middle;
  border-bottom: 1px solid #eee;
}
.img-td { text-align: center; vertical-align: top; }
.img-stat { display: flex; align-items: flex-start; gap: 4px; justify-content: center; }
img.ci {
  height: 19px; image-rendering: pixelated; flex-shrink: 0;
  border: 1px solid #ddd;
  background: #f5f5f5;
}
.sub { font-size: 9px; color: #777; line-height: 1.2; display: inline; white-space: nowrap; }
.ocr-fix { color: #c62828; }
.num { font-family: monospace; font-size: 10px; text-align: left; white-space: nowrap; display: inline; }
.mid-delta { font-family: monospace; font-size: 9px; color: #0066cc; white-space: nowrap; display: inline; }
.logprob { font-family: monospace; font-size: 9px; color: #8855aa; white-space: nowrap; display: inline; }
.geo { font-family: monospace; font-size: 9px; color: #d07a00; white-space: nowrap; display: inline; }
.prob-win { background: #e8f5e9; border-left: 3px solid #4caf50; padding-left: 4px; }
.prob-lose { background: #ffebee; border-left: 3px solid #ef5350; padding-left: 4px; }
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
.ssim-compare-table { border-collapse: collapse; width: 100%; table-layout: fixed; }
.ssim-compare-table th {
  text-align: center; font-size: 11px; font-weight: 600;
  padding: 4px 6px; border-bottom: 2px solid #ccc; overflow: hidden; word-break: break-all;
}
.ssim-compare-table td {
  padding: 4px 6px; border-bottom: 1px solid #dde; vertical-align: middle; overflow: hidden;
}
.ssim-compare-table img { max-width: 100%; height: auto; }
.ssim-compare-table .ssim-label {
  font-size: 10px; font-weight: 600; color: #555; width: 50px; text-align: right;
}
.ssim-compare-img {
  max-width: 100%; image-rendering: pixelated;
  display: block;
}
.tie-break-block {
  margin: 8px 0 10px 0; padding: 8px; background: #fff8f0;
  border: 1px solid #dca; border-radius: 4px;
  overflow-x: auto; max-width: 100%; box-sizing: border-box;
}
.tie-break-title {
  font-size: 11px; font-weight: 700; color: #c60; margin-bottom: 6px;
}
.tie-winner { background: #e8ffe8; font-weight: 600; }
.tie-loser { background: #fff0f0; }
.scan-line-block {
  margin: 6px 0 10px 0; padding: 0; background: #f0f4f0;
  border: 1px solid #c0c8c0; border-radius: 4px; overflow: hidden;
  box-sizing: border-box; width: 100%;
}
.scan-line-label { font-size: 10px; font-weight: 600; color: #555; margin: 8px 8px 4px 8px; }
.ocr-wrong-badge { background: #f0a020; color: white; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 700; }
.fast-path-badge { background: #8b5cf6; color: white; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 700; }
.line-text-preview { font-size: 12px; color: #666; margin: 2px 0 8px 0; font-style: italic; }
.scan-line-block svg { display: block; width: 100%; height: auto; }
</style>"#;

// ── Public API ──────────────────────────────────────────────────────────────

/// Accuracy result from classifying audit entries against ground truth.
pub struct AccuracyResult {
    pub hits: usize,
    pub major_misses: usize,
    pub minor_misses: usize,
    pub similarity_failures: usize,
    pub kept_raster: usize,
    pub compared: usize,
    pub primary_hits: usize,
    pub pct: f64,
    pub ocr_correct_total: usize,
    pub ocr_correct_hits: usize,
    pub ocr_wrong_total: usize,
    pub ocr_wrong_hits: usize,
}

/// Compute accuracy without generating any HTML or audit I/O.
/// Used by --test for fast scoring.
pub fn compute_accuracy(
    entries: &[AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
    glyph_map: &NgramGlyphMap,
) -> AccuracyResult {
    let classified = classify_entries(entries, gt, dpi, font_catalog, glyph_map);

    let mut hits = 0usize;
    let mut major_misses = 0usize;
    let mut minor_misses = 0usize;
    let mut similarity_failures = 0usize;
    let mut kept_raster = 0usize;

    for ce in &classified {
        match ce.kind {
            MissKind::Hit => hits += 1,
            MissKind::MajorMiss => major_misses += 1,
            MissKind::MinorMiss => minor_misses += 1,
            MissKind::SimilarityFailure => similarity_failures += 1,
            MissKind::KeptRaster => kept_raster += 1,
            MissKind::NoGroundTruth => {}
        }
    }

    let all_misses = major_misses + minor_misses + similarity_failures;
    let compared = hits + all_misses + kept_raster;
    let major_total = major_misses + similarity_failures + kept_raster;
    let primary_hits = compared - major_total;
    let pct = if compared > 0 {
        primary_hits as f64 / compared as f64 * 100.0
    } else {
        100.0
    };

    let (ocr_correct_total, ocr_correct_hits, ocr_wrong_total, ocr_wrong_hits) = {
        let mut c_total = 0usize;
        let mut c_hits = 0usize;
        let mut w_total = 0usize;
        let mut w_hits = 0usize;
        for ce in &classified {
            let is_hit = matches!(ce.kind, MissKind::Hit | MissKind::MinorMiss);
            match ce.entry.ocr_correct {
                Some(true) => { c_total += 1; if is_hit { c_hits += 1; } },
                Some(false) => { w_total += 1; if is_hit { w_hits += 1; } },
                None => {},
            }
        }
        (c_total, c_hits, w_total, w_hits)
    };

    AccuracyResult {
        hits,
        major_misses,
        minor_misses,
        similarity_failures,
        kept_raster,
        compared,
        primary_hits,
        pct,
        ocr_correct_total,
        ocr_correct_hits,
        ocr_wrong_total,
        ocr_wrong_hits,
    }
}

pub fn generate_report(
    report_path: &Path,
    audit_root: &Path,
    entries: &[AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
    glyph_map: &NgramGlyphMap,
    meta: &ReportMeta,
) -> Result<(), String> {
    let classified = classify_entries(entries, gt, dpi, font_catalog, glyph_map);

    let mut hits: Vec<&ClassifiedEntry> = Vec::new();
    let mut major_misses: Vec<&ClassifiedEntry> = Vec::new();
    let mut minor_misses: Vec<&ClassifiedEntry> = Vec::new();
    let mut similarity_failures: Vec<&ClassifiedEntry> = Vec::new();
    let mut kept_raster: Vec<&ClassifiedEntry> = Vec::new();
    let mut no_ground_truth: Vec<&ClassifiedEntry> = Vec::new();
    for ce in &classified {
        match ce.kind {
            MissKind::Hit => hits.push(ce),
            MissKind::MajorMiss => major_misses.push(ce),
            MissKind::MinorMiss => minor_misses.push(ce),
            MissKind::SimilarityFailure => similarity_failures.push(ce),
            MissKind::KeptRaster => kept_raster.push(ce),
            MissKind::NoGroundTruth => no_ground_truth.push(ce),
        }
    }

    // Collect OCR-wrong hits/minor misses (font matched, but OCR text wrong)
    let ocr_misses: Vec<&ClassifiedEntry> = classified.iter()
        .filter(|ce| matches!(ce.kind, MissKind::Hit)
                     && ce.entry.ocr_correct == Some(false))
        .collect();

    let all_misses = major_misses.len() + minor_misses.len() + similarity_failures.len();
    let compared = hits.len() + all_misses + kept_raster.len();
    // Primary metric: only major misses count against the score.
    let major_total = major_misses.len() + similarity_failures.len() + kept_raster.len();
    let primary_hits = compared - major_total;
    let pct = if compared > 0 {
        primary_hits as f64 / compared as f64 * 100.0
    } else {
        100.0
    };

    // OCR accuracy split: break down font matching accuracy by OCR correctness


    // Sort each miss category by increasing ZNCC (worst visual matches first)
    major_misses.sort_by(|a, b| {
        a.entry.similarity_score.partial_cmp(&b.entry.similarity_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    minor_misses.sort_by(|a, b| {
        a.entry.similarity_score.partial_cmp(&b.entry.similarity_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    similarity_failures.sort_by(|a, b| {
        a.entry.similarity_score.partial_cmp(&b.entry.similarity_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    if meta.report_all {
        hits.sort_by(|a, b| {
            a.entry.similarity_score.partial_cmp(&b.entry.similarity_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        no_ground_truth.sort_by(|a, b| {
            a.entry.similarity_score.partial_cmp(&b.entry.similarity_score).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut font_data_cache = FontDataCache::new();

    // Build major miss blocks
    let mut major_miss_blocks = String::new();
    for ce in &major_misses {
        let (html, _chosen_zncc, _gt_zncc) = build_miss_block(
            ce, audit_root, font_catalog, glyph_map, &mut font_data_cache, dpi,
        );
        major_miss_blocks.push_str(&html);
    }

    // Build minor miss blocks
    let mut minor_miss_blocks = String::new();
    for ce in &minor_misses {
        let (html, _chosen_zncc, _gt_zncc) = build_miss_block(
            ce, audit_root, font_catalog, glyph_map, &mut font_data_cache, dpi,
        );
        minor_miss_blocks.push_str(&html);
    }

    // Build similarity failure blocks
    let mut sim_fail_blocks = String::new();
    for ce in &similarity_failures {
        let (html, _chosen_zncc, _gt_zncc) = build_miss_block(
            ce, audit_root, font_catalog, glyph_map, &mut font_data_cache, dpi,
        );
        sim_fail_blocks.push_str(&html);
    }

    // Build kept-raster blocks
    let mut raster_blocks = String::new();
    for ce in &kept_raster {
        let (html, _chosen_zncc, _gt_zncc) = build_miss_block(
            ce, audit_root, font_catalog, glyph_map, &mut font_data_cache, dpi,
        );
        raster_blocks.push_str(&html);
    }

    // Build hits / no_ground_truth blocks when --report-all
    let mut hits_blocks = String::new();
    let mut no_gt_blocks = String::new();
    if meta.report_all {
        for ce in &hits {
            let (html, _chosen_zncc, _gt_zncc) = build_miss_block(
                ce, audit_root, font_catalog, glyph_map, &mut font_data_cache, dpi,
            );
            hits_blocks.push_str(&html);
        }
        for ce in &no_ground_truth {
            let (html, _chosen_zncc, _gt_zncc) = build_miss_block(
                ce, audit_root, font_catalog, glyph_map, &mut font_data_cache, dpi,
            );
            no_gt_blocks.push_str(&html);
        }
    }

    // Build OCR miss blocks (font matched, OCR text wrong)
    let mut ocr_miss_blocks = String::new();
    for ce in &ocr_misses {
        ocr_miss_blocks.push_str(&build_ocr_miss_block(ce, audit_root));
    }

    let sim_fail_section = if !sim_fail_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#c55;\">\
             ZNCC Failures (correct font, ZNCC rejected)</h2>{sim_fail_blocks}"
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

    let ocr_miss_section = if !ocr_miss_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#e90;\">\
             OCR Text Mismatches ({} lines — font correct, text wrong)</h2>{ocr_miss_blocks}",
            ocr_misses.len()
        )
    } else {
        String::new()
    };

    let hits_section = if meta.report_all && !hits_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#2a7;\">\
             Hits ({} lines — all correct)</h2>{hits_blocks}",
            hits.len()
        )
    } else {
        String::new()
    };

    let no_gt_section = if meta.report_all && !no_gt_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#888;\">\
             No Ground Truth ({} lines)</h2>{no_gt_blocks}",
            no_ground_truth.len()
        )
    } else {
        String::new()
    };

    // ── Summary line 1: Similarity ──────────────────────────────────
    let sim_summary = {
        let mut sim_vals: Vec<f32> = entries.iter()
            .filter_map(|e| e.similarity_score)
            .collect();
        sim_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if sim_vals.is_empty() {
            String::from("Similarity: no ZNCC data")
        } else {
            let n = sim_vals.len();
            let above = sim_vals.iter().filter(|&&v| v >= crate::MIN_VERIFY_SIMILARITY).count();
            let pct_above = above as f64 / n as f64 * 100.0;
            let avg: f64 = sim_vals.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let p50 = sim_vals[n / 2];
            let p90_idx = (n as f64 * 0.9).ceil() as usize;
            let p90 = sim_vals[p90_idx.min(n - 1)];
            format!(
                "Similarity: <b>avg={avg:.4}</b> · <b>{above}/{n} ({pct_above:.0}%)</b> above {:.1} threshold · p50={p50:.4} · p90={p90:.4}",
                crate::MIN_VERIFY_SIMILARITY,
            )
        }
    };

    // ── Summary line 2: Font accuracy ────────────────────────────────
    let font_summary = if compared > 0 {
        let major_correct = compared - major_misses.len() - similarity_failures.len();
        let major_pct = major_correct as f64 / compared as f64 * 100.0;
        let minor_correct = hits.len(); // hits = exact match
        let minor_pct = minor_correct as f64 / compared as f64 * 100.0;
        format!(
            "Font accuracy: <b>{major_correct}/{compared} ({major_pct:.0}%)</b> major correct ·              <b>{minor_correct}/{compared} ({minor_pct:.0}%)</b> exact match"
        )
    } else {
        String::from("Font accuracy: no GT data")
    };

    // ── Summary line 3: OCR ──────────────────────────────────────────
    // Compare ocr_text (tesseract raw) against gt_text for tesseract accuracy.
    // Compare word_bboxes text (pflda-corrected) against gt_text for post-pflda accuracy.
    // Count pflda true/false by diffing ocr vs corrected positionally.
    let (tess_correct, post_correct, ocr_total, pflda_true, pflda_false) = {
        let mut tc = 0usize;
        let mut pc = 0usize;
        let mut total = 0usize;
        let mut p_true = 0usize;
        let mut p_false = 0usize;
        for ce in &classified {
            let gt_text = match ce.entry.gt_text.as_deref() {
                Some(t) => t,
                None => continue,
            };
            let ocr_text = match ce.entry.ocr_text.as_deref() {
                Some(t) => t,
                None => continue,
            };
            let gt_chars: Vec<char> = gt_text.chars().filter(|c| !c.is_whitespace()).collect();
            let ocr_chars: Vec<char> = ocr_text.chars().filter(|c| !c.is_whitespace()).collect();

            // Corrected text from word_bboxes (same source as chosen font verify)
            let corrected_text: String = ce.entry.word_bboxes.iter()
                .map(|wb| wb.text.as_str()).collect::<Vec<_>>().join(" ");
            let corrected_chars: Vec<char> = corrected_text.chars()
                .filter(|c| !c.is_whitespace()).collect();

            // pflda true/false: diff ocr vs corrected positionally
            let n_diff = ocr_chars.len().min(corrected_chars.len()).min(gt_chars.len());
            for i in 0..n_diff {
                if ocr_chars[i] != corrected_chars[i] {
                    if gt_chars[i].to_lowercase().eq(corrected_chars[i].to_lowercase()) {
                        p_true += 1;
                    } else {
                        p_false += 1;
                    }
                }
            }

            // Char-by-char accuracy comparison
            let n = gt_chars.len().min(ocr_chars.len()).min(corrected_chars.len());
            for i in 0..n {
                total += 1;
                if gt_chars[i].to_lowercase().eq(ocr_chars[i].to_lowercase()) { tc += 1; }
                if gt_chars[i].to_lowercase().eq(corrected_chars[i].to_lowercase()) { pc += 1; }
            }
        }
        (tc, pc, total, p_true, p_false)
    };
    let pflda_total = pflda_true + pflda_false;

    let ocr_summary = {
        let mut parts = Vec::new();
        if ocr_total > 0 {
            let post_pct = post_correct as f64 / ocr_total as f64 * 100.0;
            let tess_pct = tess_correct as f64 / ocr_total as f64 * 100.0;
            parts.push(format!(
                "OCR: <b>{post_correct}/{ocr_total} ({post_pct:.1}%)</b> chars correct after pflda ·                  <b>{tess_correct}/{ocr_total} ({tess_pct:.1}%)</b> tesseract-only"
            ));
        }
        if pflda_total > 0 {
            parts.push(format!(
                "pflda replacements: <b>{pflda_true}</b> correct, <b>{pflda_false}</b> wrong ({pflda_total} total)"
            ));
        }
        parts.join(" · ")
    };

    // Per-character GT font rank statistics
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
         <div class=\"summary\">{sim_summary}</div>\n\
         <div class=\"summary\">{font_summary}</div>\n\
         <div class=\"summary\">{ocr_summary}</div>\n\
         <div class=\"summary\">{meta_str}</div>\n\
         <div class=\"score-legend\">\n\
         <b>Score key:</b>\n\
         <b>font score</b> (per-line) = mean(log(prob)) across characters, \
         weighted by character discriminativeness; \
         <b>higher = better match</b>.\n\
         <b>font prob</b> (per-observation) = probability as a multiple of uniform (1/N fonts); \
         <b>×u, higher = better</b>. Values below the threshold (default 6×u) are noise.\n\
         <b>ZNCC</b> (per-line) = zero-mean normalized cross-correlation between scanned line \
         and re-render; <b>-1–1, higher = more similar</b>.\n\
         </div>\n\
         <h2>Major Misses ({n_major})</h2>\n\
         {major_miss_blocks}\n\
         <h2 style=\"margin-top:2em; color:#e90;\">Minor Misses ({n_minor})</h2>\n\
         {minor_miss_blocks}\n\
         {sim_fail_section}\n\
         {raster_section}\n\
         {ocr_miss_section}\n\
         {hits_section}\n\
         {no_gt_section}\n\
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
