//! HTML miss report generation.
//!
//! Automatically generated into `<audit_dir>/report.html`.  When
//! `--audit-vector` is also provided, classifies lines as hits/misses against
//! ground truth (the Rust equivalent of `tools/char-misses.py`).  Without
//! `--audit-vector`, reports all kept-raster lines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ab_glyph::FontRef;
use base64::Engine;
use image::GrayImage;

use crate::audit::{AuditEntry, Decision};
use crate::char_index;
use crate::ground_truth::{self, GroundTruth};
use crate::font_scan::FontEntry;

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

/// Try to find a font entry in the catalog that matches a ground-truth font name.
fn find_font_in_catalog<'a>(
    font_catalog: &'a [FontEntry],
    gt_font_name: &str,
) -> Option<&'a FontEntry> {
    // Try each entry: compare family_name, font_key stem, etc.
    for fe in font_catalog {
        if ground_truth::fonts_match(&fe.family_name, gt_font_name) {
            return Some(fe);
        }
        // Also try the filename stem
        if let Some(stem) = fe.path.file_stem().and_then(|s| s.to_str()) {
            if ground_truth::fonts_match(stem, gt_font_name) {
                return Some(fe);
            }
        }
    }
    None
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
    let corrected: Vec<(usize, &crate::audit::CharCiVote)> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| c.ocr_corrected_from.is_some())
        .collect();

    let mut by_dist: Vec<(usize, &crate::audit::CharCiVote)> =
        chars.iter().enumerate().collect();
    by_dist.sort_by(|a, b| {
        b.1.min_dist_sq
            .partial_cmp(&a.1.min_dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let corrected_idxs: std::collections::HashSet<usize> =
        corrected.iter().map(|(i, _)| *i).collect();

    let worst: Vec<(usize, &crate::audit::CharCiVote)> = by_dist
        .iter()
        .filter(|(i, _)| !corrected_idxs.contains(i))
        .take(n_worst)
        .cloned()
        .collect();

    let used: std::collections::HashSet<usize> = corrected_idxs
        .iter()
        .chain(worst.iter().map(|(i, _)| i))
        .cloned()
        .collect();

    let mut normal: Vec<(usize, &crate::audit::CharCiVote)> = by_dist
        .iter()
        .filter(|(i, c)| !used.contains(i) && c.min_dist_sq < 0.008)
        .cloned()
        .collect();
    normal.reverse();
    normal.truncate(n_normal);

    let mut result: Vec<(usize, &crate::audit::CharCiVote)> = Vec::new();
    result.extend(corrected);
    result.extend(worst);
    result.extend(normal);
    result.sort_by_key(|(i, _)| *i);
    result
}

// ── Crop / image lookup ─────────────────────────────────────────────────────

/// Find the diag-seg line directory for an audit entry.
fn find_diag_seg_dir(audit_root: &Path, page: usize, line_index: usize) -> Option<PathBuf> {
    let prefix = format!("p{page}_L{line_index:03}_");
    for entry in std::fs::read_dir(audit_root).ok()? {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissKind {
    Hit,
    FontMiss,
    SsimFailure,
    KeptRaster,
    NoGroundTruth,
}

fn classify_entries<'a>(
    entries: &'a [AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
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
                let actual_font = gt
                    .lookup_font(e.page, &bbox_px, dpi)
                    .map(|s| s.to_string());

                let kind = if e.decision == Decision::KeptRaster {
                    MissKind::KeptRaster
                } else if let Some(ref actual) = actual_font {
                    let matched = e.font_matched.as_deref().unwrap_or("");
                    if ground_truth::fonts_match(matched, actual) {
                        if e.ssim_pass == Some(false) {
                            MissKind::SsimFailure
                        } else {
                            MissKind::Hit
                        }
                    } else {
                        MissKind::FontMiss
                    }
                } else {
                    MissKind::NoGroundTruth
                };

                ClassifiedEntry {
                    entry: e,
                    actual_font,
                    kind,
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
                }
            }
        })
        .collect()
}

// ── CI candidate lookup ─────────────────────────────────────────────────────

fn find_correct_ci_candidate(
    entry: &AuditEntry,
    actual_font: &str,
) -> (Option<String>, Option<f32>, Option<usize>) {
    for (i, c) in entry.ci_candidates.iter().enumerate() {
        if ground_truth::fonts_match(&c.font_key, actual_font) {
            return (Some(c.font_key.clone()), Some(c.score), Some(i + 1));
        }
        // Also try filename stem
        if let Some(stem) = Path::new(&c.font_key)
            .file_stem()
            .and_then(|s| s.to_str())
        {
            if ground_truth::fonts_match(stem, actual_font) {
                return (Some(c.font_key.clone()), Some(c.score), Some(i + 1));
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
    font_data_cache: &mut FontDataCache,
) -> String {
    let entry = ce.entry;
    let actual_font = ce.actual_font.as_deref().unwrap_or("?");
    let matched = entry.font_matched.as_deref().unwrap_or("?");

    // Find correct font CI candidate
    let (gt_key, gt_score, gt_rank) =
        if let Some(ref af) = ce.actual_font {
            find_correct_ci_candidate(entry, af)
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

    // SSIM comparison block
    let ssim_compare_html = if let Some(ref dd) = diag_dir {
        build_ssim_block(dd, actual_font, matched, entry.ssim_score)
    } else {
        String::new()
    };

    // Scan line image (ssim_scan.png shows the scanned line region)
    let scan_line_html = if let Some(ref dd) = diag_dir {
        let scan_path = dd.join("ssim_scan.png");
        if scan_path.exists() {
            if let Some(uri) = file_to_b64_uri(&scan_path) {
                format!(
                    "<div class=\"scan-line-block\">\
                     <div class=\"scan-line-label\">Scan line</div>\
                     <img src=\"{uri}\" class=\"scan-line-img\">\
                     </div>"
                )
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Segmentation stats (per-word segment info from word_*/seg_plain/summary.json)
    let seg_stats_html = if let Some(ref dd) = diag_dir {
        build_seg_stats(dd, entry)
    } else {
        String::new()
    };

    // SSIM info
    let ssim_html = match (entry.ssim_score, entry.ssim_pass) {
        (Some(score), Some(pass)) => {
            let cls = if pass { "ssim-pass" } else { "ssim-fail" };
            let label = if pass { "pass" } else { "FAIL" };
            format!(" <span class=\"{cls}\">SSIM {score:.10} ({label})</span>")
        }
        _ => String::new(),
    };

    // Determine if font pick is correct (skip char table for SSIM-only failures)
    let font_is_correct = ce.kind == MissKind::SsimFailure;

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
            actual_font,
            matched,
            gt_rank,
            gt_score,
            font_data_cache,
            diag_dir.as_deref(),
        )
    };

    let text_preview = truncate(&entry.text, 60);
    let miss_kind_label = match ce.kind {
        MissKind::FontMiss => "",
        MissKind::SsimFailure => " [SSIM failure]",
        MissKind::KeptRaster => " [kept raster]",
        _ => "",
    };

    format!(
        "<div class=\"miss\">\
         <h3>p{}:L{} — \"{}\"{}{}</h3>\
         {}\
         {}\
         {}\
         {}\
         </div>",
        entry.page, entry.line_index, text_preview, miss_kind_label, ssim_html,
        scan_line_html, seg_stats_html, ssim_compare_html, char_table_html,
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

fn build_ssim_block(
    diag_dir: &Path,
    correct_font: &str,
    chosen_font: &str,
    ssim_score: Option<f32>,
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
            format!(
                "<tr><td class=\"ssim-label\">Diff</td>\
                 <td colspan=\"2\"><img src=\"{diff_uri}\" class=\"ssim-compare-img\"></td></tr>"
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let ssim_str = ssim_score
        .map(|s| format!("{s:.10}"))
        .unwrap_or_else(|| "—".into());

    format!(
        "<div class=\"ssim-compare-block\">\
         <table class=\"ssim-compare-table\">\
         <tr><th></th><th>Correct</th><th>Picked (SSIM verified)</th></tr>\
         <tr><td class=\"ssim-label\">Font</td>\
         <td class=\"correct\">{correct_font}</td>\
         <td class=\"chosen\">{chosen_font}</td></tr>\
         <tr><td class=\"ssim-label\">Scan</td>\
         <td colspan=\"2\"><img src=\"{scan_uri}\" class=\"ssim-compare-img\"></td></tr>\
         <tr><td class=\"ssim-label\">Render</td>\
         <td colspan=\"2\"><img src=\"{render_uri}\" class=\"ssim-compare-img\"></td></tr>\
         {diff_row}\
         <tr><td class=\"ssim-label\">SSIM</td>\
         <td colspan=\"2\">{ssim_str}</td></tr>\
         </table></div>"
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
) -> String {
    let mut rows = String::new();

    for &(_idx, cv) in chars_to_show {
        let ch = cv.ch;
        let original_ocr = cv.ocr_corrected_from.unwrap_or(ch);
        let d2 = cv.min_dist_sq;

        // Crop image from disk
        let crop_uri = diag_dir.and_then(|dd| {
            find_crop_png(dd, cv.crop_index)
                .and_then(|p| file_to_b64_uri(&p))
        });

        // Correct font reference glyph — render on the fly
        let correct_ref_uri = correct_fe.and_then(|fe| {
            let data = font_data_cache.load(&fe.path)?;
            let font = FontRef::try_from_slice(data).ok()?;
            let override_map: HashMap<char, u16> = fe
                .glyph_overrides
                .as_ref()
                .map(|v| v.iter().cloned().collect())
                .unwrap_or_default();
            let img = if let Some(&gid) = override_map.get(&ch) {
                char_index::render_glyph_normalised(&font, ab_glyph::GlyphId(gid))
            } else {
                char_index::render_char_normalised(&font, ch)
            }?;
            Some(img_to_b64_uri(&img))
        });

        // Chosen font reference glyph — try on-disk font_refs first, then render
        let chosen_ref_uri = chosen_fe.and_then(|fe| {
            // Try on-disk first
            if let Some(path) = find_font_ref_png(audit_root, fe, ch) {
                return file_to_b64_uri(&path);
            }
            // Render on the fly
            let data = font_data_cache.load(&fe.path)?;
            let font = FontRef::try_from_slice(data).ok()?;
            let override_map: HashMap<char, u16> = fe
                .glyph_overrides
                .as_ref()
                .map(|v| v.iter().cloned().collect())
                .unwrap_or_default();
            let img = if let Some(&gid) = override_map.get(&ch) {
                char_index::render_glyph_normalised(&font, ab_glyph::GlyphId(gid))
            } else {
                char_index::render_char_normalised(&font, ch)
            }?;
            Some(img_to_b64_uri(&img))
        });

        // Per-char distance for correct font
        let correct_char_dist: Option<f32> = correct_fe.and_then(|fe| {
            let fk = fe.font_key();
            // Check fontmap_dists
            for (k, d) in &cv.fontmap_dists {
                if *k == fk || ground_truth::fonts_match(k, correct_font_name) {
                    return Some(*d);
                }
            }
            // Check nearest
            for (nf, nd) in &cv.nearest {
                if ground_truth::fonts_match(nf, correct_font_name) {
                    return Some(*nd);
                }
            }
            None
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

        // Best-scoring font for the OCR char
        if let Some((ref nf, nd)) = cv.nearest.first() {
            let font_name = nf.rsplit('/').next().unwrap_or(nf);
            let dc = dist_class(*nd);
            ocr_parts.push(format!(
                "<span class='font-mini'>{font_name}</span><br>\
                 <span class='num {dc}'>{nd:.6}</span>"
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

        // Per-char distance labels
        let chosen_score_label = cv
            .chosen_dist_sq
            .map(|d| {
                let dc = dist_class(d);
                format!("<div class='sub'><span class='num {dc}'>{d:.6}</span></div>")
            })
            .unwrap_or_default();

        let correct_score_label = correct_char_dist
            .map(|d| {
                let dc = dist_class(d);
                format!("<div class='sub'><span class='num {dc}'>{d:.6}</span></div>")
            })
            .unwrap_or_default();

        let _dc = dist_class(d2);

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

    let chosen_rank_info = entry
        .ci_candidates
        .iter()
        .enumerate()
        .find(|(_, c)| {
            ground_truth::fonts_match(&c.font_key, chosen_font_name)
                || Path::new(&c.font_key)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| ground_truth::fonts_match(s, chosen_font_name))
                    .unwrap_or(false)
        })
        .map(|(i, c)| format!("CI #{}, score {:.10}", i + 1, c.score))
        .unwrap_or_default();

    // If fonts_match(matched, correct), show the correct name for both
    let display_chosen = if ground_truth::fonts_match(chosen_font_name, correct_font_name) {
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

// ── CSS (matches char-misses.py) ────────────────────────────────────────────

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

pub fn generate_report(
    report_path: &Path,
    audit_root: &Path,
    entries: &[AuditEntry],
    gt: Option<&GroundTruth>,
    dpi: u32,
    font_catalog: &[FontEntry],
) -> Result<(), String> {
    let classified = classify_entries(entries, gt, dpi);

    let mut hits = 0usize;
    let mut font_misses: Vec<&ClassifiedEntry> = Vec::new();
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
            MissKind::FontMiss => font_misses.push(ce),
            MissKind::SsimFailure => ssim_failures.push(ce),
            MissKind::KeptRaster => kept_raster.push(ce),
            MissKind::NoGroundTruth => {} // excluded from hit/miss denominator
        }
    }

    let all_misses = font_misses.len() + ssim_failures.len();
    let compared = hits + all_misses;
    let pct = if compared > 0 {
        hits as f64 / compared as f64 * 100.0
    } else {
        100.0
    };

    let mut font_data_cache = FontDataCache::new();

    // Build miss blocks
    let mut miss_blocks = String::new();
    for ce in &font_misses {
        miss_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            &mut font_data_cache,
        ));
    }

    // Build SSIM failure blocks
    let mut ssim_blocks = String::new();
    for ce in &ssim_failures {
        ssim_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            &mut font_data_cache,
        ));
    }

    // Build kept-raster blocks
    let mut raster_blocks = String::new();
    for ce in &kept_raster {
        raster_blocks.push_str(&build_miss_block(
            ce,
            audit_root,
            font_catalog,
            &mut font_data_cache,
        ));
    }

    let ssim_section = if !ssim_blocks.is_empty() {
        format!(
            "<h2 style=\"margin-top:2em; color:#c55;\">\
             SSIM Failures (correct font, SSIM rejected)</h2>{ssim_blocks}"
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

    let html = format!(
        "<!DOCTYPE html>\n\
         <html>\n\
         <head>\n\
         <meta charset=\"utf-8\">\n\
         <title>unscan char-misses — {hits}/{compared} ({pct:.1}%)</title>\n\
         </head>\n\
         <body style=\"background: white; color: #222;\">\n\
         {CSS}\n\
         <h2>unscan char-misses</h2>\n\
         <div class=\"summary\">{hits}/{compared} correct ({pct:.1}%) — \
         {all_misses} misses shown below{ssim_miss_str}{raster_str}{ocr_corr_str}</div>\n\
         <div class=\"score-legend\">\n\
         <b>Score key:</b>\n\
         <b>CI score</b> (per-line) = −mean(log(dist²)) across characters; \
         <b>higher = better match</b>.\n\
         <b>CI dist²</b> (per-character) = squared Euclidean distance in \
         normalized feature space between scan crop and rendered glyph; \
         <b>lower = better</b> (good: &lt;1e-4, suspect: &gt;1e-3).\n\
         <b>SSIM</b> (per-line) = structural similarity between scanned line \
         and re-render; <b>0–1, higher = more similar</b>.\n\
         </div>\n\
         {miss_blocks}\n\
         {ssim_section}\n\
         {raster_section}\n\
         </body>\n\
         </html>"
    );

    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create report directory: {e}"))?;
    }
    std::fs::write(report_path, &html)
        .map_err(|e| format!("Failed to write report: {e}"))?;

    eprintln!(
        "Report: {hits}/{compared} ({pct:.1}%) — {all_misses} misses{ssim_miss_str} → {}",
        report_path.display()
    );

    Ok(())
}
