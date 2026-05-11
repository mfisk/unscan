//! Font matching — multi-signal fusion with normalised tight-crop comparison.
//!
//! Strategy:
//! 1. Select 1-2 sample words from the OCR line.
//! 2. For each candidate font, render each sample word at the OCR bbox height.
//! 3. Tight-crop both source and rendered to their ink bounding boxes.
//! 4. Normalise both tight-crops to NORM_H pixels tall (preserving aspect ratio).
//! 5. Pad the narrower one to match the wider one (centred).
//! 6. Compute IoU, NCC, Hu moments, and fill ratio on the normalised canvases.
//! 7. Width matching uses both bbox-level advance width AND tight-crop aspect
//!    ratio.  Width mismatch is a multiplicative gate on the detail score.

use crate::font_scan::{FontClass, FontEntry};
use crate::ocr::TextLine;
use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma};
use log::{debug, info, warn};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FontMatchResult {
    pub font_name: String,
    pub font_path: PathBuf,
    pub score: f32,
    pub font_data: Vec<u8>,
    /// Best vertical pixel shift from SSIM alignment search (0 if coarse-only).
    pub best_dy: i32,
    /// True when the score already comes from SSIM verification (rerank path),
    /// so Pass 2 can skip the redundant verify call.
    pub ssim_verified: bool,
}

/// Pre-processed source word ready for comparison.
struct SourceSample {
    text: String,
    bbox_width: u32,
    bbox_height: u32,
    /// Tight-cropped binarised image, normalised to NORM_H height.
    norm: GrayImage,
    norm_w: u32,
    hu: [f64; 7],
    fill_ratio: f32,
    /// Pre-computed Gaussian blur of the normalised image.
    blur: GrayImage,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Normalisation height for tight-crop comparison.
const NORM_H: u32 = 48;

/// SSIM reranking: keep all coarse candidates within this factor of the best
/// coarse score, rather than a fixed top-N.  E.g. 1.2 means keep everything
/// scoring ≥ best_coarse / 1.2.
const RERANK_SCORE_FACTOR: f32 = 1.2;

/// Warn when the within-factor pool exceeds this many candidates (suggests
/// the coarse scores are poorly separated).
const RERANK_WARN_THRESHOLD: usize = 40;

/// Detail-score signal weights.
const W_IOU: f32 = 0.35;
const W_NCC: f32 = 0.25;
const W_HU: f32 = 0.20;
const W_FILL: f32 = 0.20;

/// Bbox-level width ratio below this → skip font entirely.
const WIDTH_RATIO_FLOOR: f32 = 0.60;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn match_font<'a>(
    gray_page: &GrayImage,
    line: &TextLine,
    catalog: &'a [FontEntry],
    parsed_fonts: &[Option<FontRef<'a>>],
    threshold: f32,
    _dpi: u32,
    char_index_keys: Option<&std::collections::HashSet<String>>,
) -> Option<FontMatchResult> {
    // ── 1. Select sample words ───────────────────────────────────────
    let word_refs = select_sample_words(line);
    if word_refs.is_empty() {
        return None;
    }

    // ── 2. Pre-process source samples ────────────────────────────────
    let src_samples: Vec<SourceSample> = word_refs
        .iter()
        .filter_map(|w| {
            let crop = safe_crop(&gray_page, w.x, w.y, w.width, w.height);
            if crop.width() < 6 || crop.height() < 6 {
                return None;
            }
            let bin = otsu_binarize(&crop);
            let clean = morphological_open(&bin, 1);
            let tight = tight_crop(&clean);
            if tight.width() < 4 || tight.height() < 4 {
                return None;
            }
            // Normalise to NORM_H preserving aspect ratio
            let norm_w = ((tight.width() as f32) * NORM_H as f32 / tight.height() as f32)
                .round()
                .max(1.0) as u32;
            let norm_resized = resize_to(&tight, norm_w, NORM_H);
            let norm = threshold_mid(&norm_resized);
            let hu = hu_moments(&norm);
            let fill = fill_ratio(&norm);
            let blur = gaussian_blur_3x3(&norm);
            Some(SourceSample {
                text: w.text.clone(),
                bbox_width: w.width,
                bbox_height: w.height,
                norm,
                norm_w,
                hu,
                fill_ratio: fill,
                blur,
            })
        })
        .collect();

    if src_samples.is_empty() {
        return None;
    }

    // ── 3. Detect source characteristics ─────────────────────────────
    let is_mono_source = detect_monospace_source(line);
    let detected_class = guess_class_from_samples(&src_samples);
    let source_is_bold = detect_bold_source(&src_samples);

    // DEBUG: dump source sample images for the first line
    if std::env::var("SCANTEXT_DUMP").is_ok() {
        for (si, src) in src_samples.iter().enumerate() {
            let dump_dir = std::path::Path::new("/tmp/unscan-dump");
            let _ = std::fs::create_dir_all(dump_dir);
            let line_idx = line.words.first().map(|w| w.line_num).unwrap_or(0);
            let path = dump_dir.join(format!("src_L{}_S{}_{}.png", line_idx, si, &src.text[..src.text.len().min(12)]));
            let _ = src.norm.save(&path);
            debug!("  DUMP src: {:?} ({}x{})", path, src.norm.width(), src.norm.height());
        }
    }

    debug!(
        "  font_match: {} samples, class={:?} mono={}  [first='{}' bbox={}x{} norm_w={}]",
        src_samples.len(),
        detected_class,
        is_mono_source,
        &src_samples[0].text[..src_samples[0].text.len().min(25)],
        src_samples[0].bbox_width,
        src_samples[0].bbox_height,
        src_samples[0].norm_w,
    );

    // ── 4. Score each candidate ──────────────────────────────────────
    // Check if source text contains digits — if so, we'll prefer fonts
    // with matching figure style (lining vs old-style).
    let line_text: String = line.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    let _source_has_digits = line_text.chars().any(|c| c.is_ascii_digit());

    let mut best_score = f32::NEG_INFINITY;
    let mut best_entry: Option<&FontEntry> = None;
    let mut top_candidates: Vec<(f32, &FontEntry)> = Vec::new();

    // If char index returned candidates, ONLY score those (skip brute-force).
    // Fall back to full catalog only when char index is unavailable.
    let use_char_gate = char_index_keys.is_some();
    let score_entries: Vec<(usize, &FontEntry)> = if let Some(ci) = char_index_keys {
        catalog.iter().enumerate().filter(|(_, e)| ci.contains(&e.font_key())).collect()
    } else {
        catalog.iter().enumerate().collect()
    };

    info!(
        "  Coarse scoring: {} fonts (gated={}, ci_keys={}) for '{:.40}…'",
        score_entries.len(),
        use_char_gate,
        char_index_keys.map_or(0, |k| k.len()),
        line_text,
    );

    for &(cat_idx, entry) in &score_entries {
        // ── Pre-filter: mono ────────────────────────────────────────
        // Only apply mono filter — serif/sans classification from raster
        // is unreliable (transition heuristic returns wrong class).
        if is_mono_source
            && entry.class != FontClass::Mono
            && entry.class != FontClass::Unknown
        {
            continue;
        }

        // ── Pre-filter: bold mismatch ─────────────────────────────
        // Don't match bold fonts to regular text or vice versa.
        if entry.is_bold != source_is_bold {
            continue;
        }
        // NOTE: no italic pre-filter — centroid-based slant detection is
        // unreliable (sans uprights overlap with italic serif). Let the
        // pixel scoring + SSIM verification sort it out.

        // ── Pre-filter: skip CJK-primary fonts for Latin text ─────
        // Fonts like ipagp, IPAMincho, Noto CJK, etc. have Latin glyphs
        // but they look wrong for English body text.
        {
            let fl = entry.family_name.to_lowercase();
            if fl.starts_with("ipa")
                || fl.contains("gothic")  // CJK gothic != Western gothic
                || fl.contains("mincho")
                || fl.contains(" cjk")
                || fl.starts_with("cjk")
                || fl.contains("wqy")
                || fl.contains("noto sans jp")
                || fl.contains("noto serif jp")
                || fl.contains("noto sans kr")
                || fl.contains("noto serif kr")
                || fl.contains("noto sans sc")
                || fl.contains("noto serif sc")
                || fl.contains("noto sans tc")
                || fl.contains("noto serif tc")
            {
                continue;
            }
        }

        // ── Pre-filter: old-style figures ─────────────────────────
        // Detection is available (entry.oldstyle_figures) but not used
        // as a filter — documents can use either style. Let SSIM pick
        // the best pixel match naturally.
        // ── Pre-filter: parse font ──────────────────────────────────
        let font = match &parsed_fonts[cat_idx] {
            Some(f) => f,
            None => continue,
        };
        let overrides = entry.glyph_overrides.as_deref();

        // ── Pre-filter: glyph coverage ──────────────────────────────
        let first_text = &src_samples[0].text;
        let total_chars = first_text.chars().filter(|c| !c.is_whitespace()).count();
        if total_chars == 0 {
            continue;
        }
        let renderable = first_text
            .chars()
            .filter(|c| !c.is_whitespace())
            .filter(|&c| font.glyph_id(c).0 != 0)
            .count();
        if (renderable as f32 / total_chars as f32) < 0.8 {
            continue;
        }

        // ── Quick bbox-level width check on first sample ─────────────
        let first = &src_samples[0];
        let rend_w0 = rendered_text_width(&font, &first.text, first.bbox_height as f32, overrides);
        let wr_bbox0 = symmetric_ratio(first.bbox_width, rend_w0);
        if wr_bbox0 < WIDTH_RATIO_FLOOR {
            continue;
        }

        // ── Per-word scoring ─────────────────────────────────────────
        let mut word_scores: Vec<f32> = Vec::new();
        let mut width_ratios: Vec<f32> = Vec::new();

        for (si, src) in src_samples.iter().enumerate() {
            // Bbox-level width ratio
            let wr_bbox = if si == 0 {
                wr_bbox0
            } else {
                let rw = rendered_text_width(&font, &src.text, src.bbox_height as f32, overrides);
                let wr = symmetric_ratio(src.bbox_width, rw);
                if wr < WIDTH_RATIO_FLOOR {
                    continue;
                }
                wr
            };

            // Render, binarize, tight-crop candidate
            let rendered = render_text_gray(&font, &src.text, src.bbox_height as f32, overrides);
            if rendered.width() < 4 || rendered.height() < 4 {
                continue;
            }

            // DEBUG: dump raw render for selected fonts
            if std::env::var("SCANTEXT_DUMP").is_ok() {
                let fam_lc2 = entry.family_name.to_lowercase();
                if fam_lc2.contains("georgia") && si == 0 {
                    let dump_dir = std::path::Path::new("/tmp/unscan-dump");
                    let _ = std::fs::create_dir_all(dump_dir);
                    let line_idx = line.words.first().map(|w| w.line_num).unwrap_or(0);
                    let p = dump_dir.join(format!("raw_L{}_{}_{}.png", line_idx, si, entry.family_name.replace(' ', "_")));
                    let _ = rendered.save(&p);
                    debug!("  DUMP raw render {:?}: {}x{}", p, rendered.width(), rendered.height());
                }
            }

            let cand_bin = otsu_binarize(&rendered);
            let cand_clean = morphological_open(&cand_bin, 1);
            let cand_tight = tight_crop(&cand_clean);
            if cand_tight.width() < 3 || cand_tight.height() < 3 {
                continue;
            }

            // Normalise candidate tight to NORM_H (preserving aspect ratio)
            let cand_norm_w = ((cand_tight.width() as f32) * NORM_H as f32
                / cand_tight.height() as f32)
                .round()
                .max(1.0) as u32;
            let cand_norm_resized = resize_to(&cand_tight, cand_norm_w, NORM_H);
            let cand_norm = threshold_mid(&cand_norm_resized);

            // Tight-crop aspect ratio similarity
            let wr_tight = symmetric_ratio(src.norm_w, cand_norm_w);

            // Combined width ratio (average of bbox and tight-crop)
            let wr_combined = wr_bbox * 0.5 + wr_tight * 0.5;
            width_ratios.push(wr_combined);

            // ── Pad both to same canvas for pixel comparison ─────────
            let canvas_w = src.norm_w.max(cand_norm_w);
            let src_padded = center_pad(&src.norm, canvas_w, NORM_H);
            let cand_padded = center_pad(&cand_norm, canvas_w, NORM_H);

            // ── Signal 1: IoU with alignment search ──────────────────
            let iou = aligned_iou(&src_padded, &cand_padded, 2);

            // ── Signal 2: NCC on blurred images ──────────────────────
            let src_blur = center_pad(&src.blur, canvas_w, NORM_H);
            let cand_blur = gaussian_blur_3x3(&cand_padded);
            let ncc = ncc_score(&src_blur, &cand_blur);

            // ── Signal 3: Hu moment similarity ───────────────────────
            let cand_hu = hu_moments(&cand_norm);
            let hu_sim = hu_similarity(&src.hu, &cand_hu);

            // ── Signal 4: Fill ratio similarity ──────────────────────
            let cand_fill = fill_ratio(&cand_norm);
            let fill_diff = (src.fill_ratio - cand_fill).abs();
            let fill_sim = (1.0 - fill_diff * 3.0).max(0.0); // 33% diff → 0

            let detail = W_IOU * iou + W_NCC * ncc + W_HU * hu_sim + W_FILL * fill_sim;

            // DEBUG: dump candidate images for selected fonts
            if std::env::var("SCANTEXT_DUMP").is_ok() {
                let fam_lc = entry.family_name.to_lowercase();
                if fam_lc.contains("georgia") || fam_lc == "inter thin" {
                    let dump_dir = std::path::Path::new("/tmp/unscan-dump");
                    let _ = std::fs::create_dir_all(dump_dir);
                    let safe_name = entry.family_name.replace(' ', "_");
                    let line_idx = line.words.first().map(|w| w.line_num).unwrap_or(0);
                    let path = dump_dir.join(format!("cand_L{}_S{}_{}.png", line_idx, si, safe_name));
                    let _ = cand_norm.save(&path);
                    debug!(
                        "  DUMP cand {:?}: iou={:.3} ncc={:.3} hu={:.3} fill={:.3} detail={:.3} ({}x{})",
                        path, iou, ncc, hu_sim, fill_sim, detail, cand_norm.width(), cand_norm.height()
                    );
                }
            }

            word_scores.push(detail);
        }

        if word_scores.is_empty() {
            continue;
        }

        let avg_detail: f32 = word_scores.iter().sum::<f32>() / word_scores.len() as f32;
        let avg_wr: f32 = width_ratios.iter().sum::<f32>() / width_ratios.len() as f32;

        // Width factor: multiplicative gate  
        // Use a softer curve: wr^1.5 so 0.9→0.85, 0.8→0.72, 0.7→0.59
        let width_factor = avg_wr.powf(1.5);

        // Mono bonus
        let mono_bonus: f32 = if is_mono_source && entry.class == FontClass::Mono {
            0.03
        } else {
            0.0
        };

        let score = width_factor * avg_detail + mono_bonus;

        // ── Diagnostic logging ───────────────────────────────────────
        let fl = entry.family_name.to_lowercase();
        let should_log = fl.contains("georgia")
            || fl.contains("times")
            || fl.contains("courier")
            || fl.contains("prestige")
            || fl.contains("letter gothic")
            || fl.contains("liberation")
            || fl.contains("freemono")
            || fl.contains("arial")
            || fl.contains("garamond")
            || fl.contains("nimbus")
            || fl.contains("verdana")
            || fl.contains("trebuchet")
            || fl == "inter"
            || fl.starts_with("inter ")
            || score > best_score;

        if should_log {
            debug!(
                "    {} '{}': wR={:.3} detail={:.3} wFact={:.3} → {:.3}{}",
                if score > best_score { "★" } else { " " },
                entry.family_name,
                avg_wr,
                avg_detail,
                width_factor,
                score,
                if word_scores.len() > 1 {
                    format!("  [{} words]", word_scores.len())
                } else {
                    String::new()
                },
            );
        }

        if score > best_score {
            best_score = score;
            best_entry = Some(entry);
        }

        // Collect all candidates — we'll filter by score factor after the loop.
        top_candidates.push((score, entry));
    }

    // ── Stage 2: Re-rank top candidates with full-resolution SSIM ────
    if top_candidates.len() >= 2 {
        top_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let best_coarse = top_candidates[0].0;
        let score_floor = if best_coarse > 0.0 { best_coarse / RERANK_SCORE_FACTOR } else { 0.0 };
        top_candidates.retain(|&(s, _)| s >= score_floor);

        if top_candidates.len() > RERANK_WARN_THRESHOLD {
            warn!(
                "  rerank: {} candidates within {:.1}× of best coarse ({:.3}) — scores poorly separated for '{:.40}…'",
                top_candidates.len(),
                RERANK_SCORE_FACTOR,
                best_coarse,
                line_text,
            );
        }

        info!("  rerank: {} candidates within {:.1}× (floor={:.3}), coarse best='{}'({:.3}) for '{:.40}…'",
            top_candidates.len(),
            RERANK_SCORE_FACTOR,
            score_floor,
            top_candidates[0].1.family_name,
            top_candidates[0].0,
            line_text);

        // Use the full line bbox for SSIM
        let lx = line.words.iter().map(|w| w.x).min().unwrap_or(0);
        let ly = line.words.iter().map(|w| w.y).min().unwrap_or(0);
        let lx2 = line.words.iter().map(|w| w.x + w.width).max().unwrap_or(0);
        let ly2 = line.words.iter().map(|w| w.y + w.height).max().unwrap_or(0);
        let lw = lx2.saturating_sub(lx);
        let lh = ly2.saturating_sub(ly);

        let mut best_rerank_ssim = -1.0f32;
        let mut best_rerank_entry: Option<&FontEntry> = None;
        let mut _best_rerank_coarse = 0.0f32;
        let mut best_rerank_dy = 0i32;

        for &(coarse_score, entry) in &top_candidates {
            let (ssim, dy) = crate::verify::verify_text_region(
                gray_page,
                &entry.data,
                "",  // text not used
                lx, ly, lw, lh,
                &line.words,
            );
            debug!("    rerank '{}': coarse={:.3} ssim={:.3}", entry.family_name, coarse_score, ssim);
            if ssim > best_rerank_ssim {
                best_rerank_ssim = ssim;
                best_rerank_entry = Some(entry);
                _best_rerank_coarse = coarse_score;
                best_rerank_dy = dy;
            }
        }

        if let Some(entry) = best_rerank_entry {
            if best_rerank_ssim > 0.0 {
                debug!("  rerank WINNER: '{}' ssim={:.3} (coarse was '{}')",
                    entry.family_name, best_rerank_ssim,
                    top_candidates[0].1.family_name);
                // Use SSIM as the returned score — it's more meaningful
                // than the coarse score and is what the caller thresholds on.
                return Some(FontMatchResult {
                    font_name: entry.family_name.clone(),
                    font_path: entry.path.clone(),
                    score: best_rerank_ssim,
                    font_data: entry.data.clone(),
                    best_dy: best_rerank_dy,
                    ssim_verified: true,
                });
            }
        }
    }

    best_entry.map(|e| {
        debug!(
            "  font_match: BEST='{}' score={:.3} threshold={:.2}",
            e.family_name, best_score, threshold
        );
        FontMatchResult {
            font_name: e.family_name.clone(),
            font_path: e.path.clone(),
            score: best_score,
            font_data: e.data.clone(),
            best_dy: 0, // coarse-only path, no SSIM shift
            ssim_verified: false,
        }
    })
}

// ===========================================================================
// Sample word selection
// ===========================================================================

struct SampleWord {
    text: String,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn select_sample_words(line: &TextLine) -> Vec<SampleWord> {
    let mut candidates: Vec<&crate::ocr::TextRegion> = line
        .words
        .iter()
        .filter(|w| w.text.chars().filter(|c| c.is_alphanumeric()).count() >= 3)
        .collect();

    candidates.sort_by(|a, b| b.text.len().cmp(&a.text.len()));

    candidates
        .iter()
        .take(4)
        .map(|w| SampleWord {
            text: w.text.clone(),
            x: w.x,
            y: w.y,
            width: w.width,
            height: w.height,
        })
        .collect()
}

// ===========================================================================
// Quick rendered-width computation (no image allocation)
// ===========================================================================

fn rendered_text_width(font: &FontRef, text: &str, px_height: f32, overrides: Option<&[(char, u16)]>) -> u32 {
    let scale = PxScale::from(px_height);
    let sf = font.as_scaled(scale);

    let mut total_w = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for ch in text.chars() {
        let gid = resolve_glyph(font, ch, overrides);
        if let Some(p) = prev {
            total_w += sf.kern(p, gid);
        }
        total_w += sf.h_advance(gid);
        prev = Some(gid);
    }
    total_w.ceil().max(1.0) as u32
}

// ===========================================================================
// Image normalisation helpers
// ===========================================================================

/// Crop to tight ink bounding box (0=ink, 255=bg).
fn tight_crop(bin: &GrayImage) -> GrayImage {
    let (w, h) = bin.dimensions();
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;

    for y in 0..h {
        for x in 0..w {
            if bin.get_pixel(x, y).0[0] == 0 {
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }
    }

    if max_x < min_x || max_y < min_y {
        return GrayImage::from_pixel(1, 1, Luma([255u8]));
    }

    image::imageops::crop_imm(bin, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        .to_image()
}

/// Centre-pad an image to `target_w × target_h` (white bg).
fn center_pad(img: &GrayImage, target_w: u32, target_h: u32) -> GrayImage {
    let (w, h) = img.dimensions();
    if w == target_w && h == target_h {
        return img.clone();
    }
    let mut out = GrayImage::from_pixel(target_w, target_h, Luma([255u8]));
    let ox = target_w.saturating_sub(w) / 2;
    let oy = target_h.saturating_sub(h) / 2;
    for y in 0..h.min(target_h) {
        for x in 0..w.min(target_w) {
            let dx = x + ox;
            let dy = y + oy;
            if dx < target_w && dy < target_h {
                out.put_pixel(dx, dy, *img.get_pixel(x, y));
            }
        }
    }
    out
}

fn symmetric_ratio(a: u32, b: u32) -> f32 {
    if a == 0 || b == 0 {
        return 0.0;
    }
    a.min(b) as f32 / a.max(b) as f32
}

// ===========================================================================
// Signal: Aligned IoU
// ===========================================================================

fn aligned_iou(src: &GrayImage, cand: &GrayImage, range: i32) -> f32 {
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 || cand.dimensions() != (w, h) {
        return 0.0;
    }
    let max_dx = range.min(w as i32 / 4).max(0);
    let max_dy = range.min(h as i32 / 4).max(0);

    let mut best = 0.0f32;

    for dy in -max_dy..=max_dy {
        for dx in -max_dx..=max_dx {
            let mut inter = 0u32;
            let mut union_count = 0u32;

            for y in 0..h {
                for x in 0..w {
                    let src_ink = src.get_pixel(x, y).0[0] == 0;
                    let cx = x as i32 + dx;
                    let cy = y as i32 + dy;
                    let cand_ink = if cx >= 0 && cy >= 0 && (cx as u32) < w && (cy as u32) < h {
                        cand.get_pixel(cx as u32, cy as u32).0[0] == 0
                    } else {
                        false
                    };
                    if src_ink || cand_ink {
                        union_count += 1;
                        if src_ink && cand_ink {
                            inter += 1;
                        }
                    }
                }
            }

            let iou = if union_count == 0 {
                0.0
            } else {
                inter as f32 / union_count as f32
            };
            if iou > best {
                best = iou;
            }
        }
    }

    best
}

// ===========================================================================
// Signal: NCC
// ===========================================================================

fn ncc_score(a: &GrayImage, b: &GrayImage) -> f32 {
    if a.dimensions() != b.dimensions() {
        return 0.0;
    }
    let n = (a.width() as u64 * a.height() as u64) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let va = pa.0[0] as f64;
        let vb = pb.0[0] as f64;
        sa += va;
        sb += vb;
        saa += va * va;
        sbb += vb * vb;
        sab += va * vb;
    }
    let ma = sa / n;
    let mb = sb / n;
    let va = (saa / n) - ma * ma;
    let vb = (sbb / n) - mb * mb;
    let cv = (sab / n) - ma * mb;
    let d = (va * vb).sqrt();
    if d < 1e-10 {
        return 0.0;
    }
    ((cv / d) as f32).max(0.0)
}

// ===========================================================================
// Signal: Hu Moments
// ===========================================================================

fn hu_moments(bin: &GrayImage) -> [f64; 7] {
    let (w, h) = bin.dimensions();
    let (mut m00, mut m10, mut m01) = (0f64, 0f64, 0f64);
    let (mut m20, mut m11, mut m02) = (0f64, 0f64, 0f64);
    let (mut m30, mut m21, mut m12, mut m03) = (0f64, 0f64, 0f64, 0f64);

    for y in 0..h {
        for x in 0..w {
            if bin.get_pixel(x, y).0[0] != 0 {
                continue;
            }
            let xf = x as f64;
            let yf = y as f64;
            m00 += 1.0;
            m10 += xf;
            m01 += yf;
            m20 += xf * xf;
            m11 += xf * yf;
            m02 += yf * yf;
            m30 += xf * xf * xf;
            m21 += xf * xf * yf;
            m12 += xf * yf * yf;
            m03 += yf * yf * yf;
        }
    }

    if m00 < 1.0 {
        return [0.0; 7];
    }

    let cx = m10 / m00;
    let cy = m01 / m00;
    let mu20 = m20 - cx * m10;
    let mu11 = m11 - cx * m01;
    let mu02 = m02 - cy * m01;
    let mu30 = m30 - 3.0 * cx * m20 + 2.0 * cx * cx * m10;
    let mu21 = m21 - 2.0 * cx * m11 - cy * m20 + 2.0 * cx * cx * m01;
    let mu12 = m12 - 2.0 * cy * m11 - cx * m02 + 2.0 * cy * cy * m10;
    let mu03 = m03 - 3.0 * cy * m02 + 2.0 * cy * cy * m01;

    let norm = |mu: f64, p: i32, q: i32| -> f64 {
        let gamma = ((p + q) as f64 / 2.0) + 1.0;
        mu / m00.powf(gamma)
    };

    let n20 = norm(mu20, 2, 0);
    let n11 = norm(mu11, 1, 1);
    let n02 = norm(mu02, 0, 2);
    let n30 = norm(mu30, 3, 0);
    let n21 = norm(mu21, 2, 1);
    let n12 = norm(mu12, 1, 2);
    let n03 = norm(mu03, 0, 3);

    let h1 = n20 + n02;
    let h2 = (n20 - n02).powi(2) + 4.0 * n11.powi(2);
    let h3 = (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2);
    let h4 = (n30 + n12).powi(2) + (n21 + n03).powi(2);
    let h5 = (n30 - 3.0 * n12) * (n30 + n12)
        * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        + (3.0 * n21 - n03) * (n21 + n03)
            * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));
    let h6 = (n20 - n02)
        * ((n30 + n12).powi(2) - (n21 + n03).powi(2))
        + 4.0 * n11 * (n30 + n12) * (n21 + n03);
    let h7 = (3.0 * n21 - n03) * (n30 + n12)
        * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        - (n30 - 3.0 * n12) * (n21 + n03)
            * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));

    [h1, h2, h3, h4, h5, h6, h7]
}

fn hu_similarity(a: &[f64; 7], b: &[f64; 7]) -> f32 {
    let mut dist_sq = 0.0f64;
    for i in 0..7 {
        let la = if a[i].abs() > 1e-30 {
            -a[i].abs().log10().copysign(a[i])
        } else {
            0.0
        };
        let lb = if b[i].abs() > 1e-30 {
            -b[i].abs().log10().copysign(b[i])
        } else {
            0.0
        };
        dist_sq += (la - lb).powi(2);
    }
    let dist = dist_sq.sqrt();
    (1.0 / (1.0 + dist * 0.15)) as f32
}

// ===========================================================================
// Signal: Fill ratio
// ===========================================================================

fn fill_ratio(bin: &GrayImage) -> f32 {
    let total = (bin.width() * bin.height()) as f32;
    if total == 0.0 {
        return 0.0;
    }
    let ink: u32 = bin.pixels().filter(|p| p.0[0] == 0).count() as u32;
    ink as f32 / total
}

// ===========================================================================
// Source classification heuristics
// ===========================================================================

/// Detect monospace from the LINE (not individual words) by checking if word
/// widths are proportional to character count (uniform char-width).
/// Detect whether source text appears bold by measuring fill ratio.
/// Bold text has thicker strokes → higher fill ratio in the tight-cropped
/// normalised image.
fn detect_bold_source(samples: &[SourceSample]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let avg_fill: f32 = samples.iter().map(|s| s.fill_ratio).sum::<f32>() / samples.len() as f32;
    log::debug!("    bold_detect: avg_fill={:.3} fills=[{}]", avg_fill,
        samples.iter().map(|s| format!("{:.3}", s.fill_ratio)).collect::<Vec<_>>().join(", "));
    // Typical fill ratios: regular serif ≈ 0.19-0.25, bold serif ≈ 0.32-0.35
    // Regular sans ≈ 0.29-0.31, bold sans ≈ 0.39-0.41
    // Threshold 0.33 catches serif bold without false-positive on sans regular.
    avg_fill > 0.33
}

/// Italic detection result: definite, or indeterminate (gray zone).
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum ItalicGuess {
    No,        // slant < 0.04: definitely upright
    Maybe,     // 0.04..0.09: ambiguous (sans uprights overlap with mild italics)
    Yes,       // slant >= 0.09: definitely italic
}

/// Detect whether source text appears italic by measuring the horizontal
/// centroid shift between the top and bottom halves of the image.
#[allow(dead_code)]
fn detect_italic_source(samples: &[SourceSample]) -> ItalicGuess {
    if samples.is_empty() {
        return ItalicGuess::No;
    }
    let mut slant_sum = 0.0f32;
    let mut count = 0;
    for src in samples {
        let (w, h) = (src.norm.width(), src.norm.height());
        if h < 8 || w < 8 {
            continue;
        }
        let mid_y = h / 2;
        let centroid_x = |y0: u32, y1: u32| -> f32 {
            let mut sx = 0.0f64;
            let mut n = 0u64;
            for y in y0..y1 {
                for x in 0..w {
                    if src.norm.get_pixel(x, y).0[0] == 0 {
                        sx += x as f64;
                        n += 1;
                    }
                }
            }
            if n > 0 { (sx / n as f64) as f32 } else { w as f32 / 2.0 }
        };
        let top_cx = centroid_x(0, mid_y);
        let bot_cx = centroid_x(mid_y, h);
        let shift = (top_cx - bot_cx) / h as f32;
        slant_sum += shift;
        count += 1;
    }
    if count == 0 {
        return ItalicGuess::No;
    }
    let avg_slant = slant_sum / count as f32;
    log::debug!("    italic_detect: avg_slant={:.4}", avg_slant);
    // Bands:  < 0.04 = upright,  0.04-0.09 = gray zone,  >= 0.09 = italic
    // Upright serif: 0.001..0.027, upright sans: 0.038..0.079, italic: 0.058..0.105
    if avg_slant >= 0.09 {
        ItalicGuess::Yes
    } else if avg_slant < 0.04 {
        ItalicGuess::No
    } else {
        ItalicGuess::Maybe
    }
}

fn detect_monospace_source(line: &TextLine) -> bool {
    // Need at least 3 words of different lengths to judge
    let words: Vec<_> = line
        .words
        .iter()
        .filter(|w| {
            let alpha_count = w.text.chars().filter(|c| c.is_alphanumeric()).count();
            alpha_count >= 2 && w.width > 5
        })
        .collect();

    if words.len() < 3 {
        return false;
    }

    // For monospace: width should be proportional to character count.
    // Compute per-character width for each word and check consistency.
    let per_char_widths: Vec<f32> = words
        .iter()
        .map(|w| {
            let n = w.text.chars().count() as f32;
            w.width as f32 / n
        })
        .collect();

    let mean = per_char_widths.iter().sum::<f32>() / per_char_widths.len() as f32;
    if mean < 2.0 {
        return false;
    }

    let var = per_char_widths
        .iter()
        .map(|w| (w - mean).powi(2))
        .sum::<f32>()
        / per_char_widths.len() as f32;
    let cv = var.sqrt() / mean;

    debug!(
        "    mono_detect: {} words, mean_char_w={:.1} cv={:.3} → {}",
        words.len(), mean, cv, cv < 0.10
    );

    cv < 0.10 // very strict — char widths must be very uniform
}

/// Use all sample words to vote on serif vs sans.
/// Measures horizontal transition density at baseline vs midzone per sample.
fn guess_class_from_samples(samples: &[SourceSample]) -> FontClass {
    let mut serif_votes = 0i32;
    let mut sans_votes = 0i32;

    for src in samples {
        let bin = &src.norm;
        let h = bin.height();
        let w = bin.width();
        if h < 8 || w < 8 {
            continue;
        }

        let baseline_start = (h as f32 * 0.80) as u32;
        let mid_start = (h as f32 * 0.25) as u32;
        let mid_end = (h as f32 * 0.75) as u32;

        let transitions = |y0: u32, y1: u32| -> u64 {
            let mut count = 0u64;
            for y in y0..y1 {
                let mut prev = bin.get_pixel(0, y).0[0];
                for x in 1..w {
                    let cur = bin.get_pixel(x, y).0[0];
                    if cur != prev {
                        count += 1;
                    }
                    prev = cur;
                }
            }
            count
        };

        let base_rows = (h - baseline_start).max(1);
        let mid_rows = (mid_end - mid_start).max(1);
        let base_rate = transitions(baseline_start, h) as f64 / (base_rows as f64 * w as f64);
        let mid_rate = transitions(mid_start, mid_end) as f64 / (mid_rows as f64 * w as f64);

        if mid_rate > 0.0 {
            let ratio = base_rate / mid_rate;
            if ratio > 1.25 {
                serif_votes += 1; // extra transitions at baseline = serifs
            } else if ratio < 0.95 {
                sans_votes += 1; // fewer transitions at baseline = no serifs
            }
        }
    }

    log::debug!("    class_detect: serif_votes={} sans_votes={}", serif_votes, sans_votes);

    if serif_votes > sans_votes && serif_votes >= 2 {
        FontClass::Serif
    } else if sans_votes > serif_votes && sans_votes >= 2 {
        FontClass::Sans
    } else {
        FontClass::Unknown
    }
}

// ===========================================================================
// Image pre-processing
// ===========================================================================

fn otsu_binarize(img: &GrayImage) -> GrayImage {
    let mut hist = [0u32; 256];
    for p in img.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let total = img.width() * img.height();
    let mut sum_total = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum_total += i as f64 * c as f64;
    }
    let mut sum_bg = 0.0f64;
    let mut w_bg = 0u32;
    let mut max_var = 0.0f64;
    let mut thr = 128u8;
    for (t, &c) in hist.iter().enumerate() {
        w_bg += c;
        if w_bg == 0 {
            continue;
        }
        let w_fg = total - w_bg;
        if w_fg == 0 {
            break;
        }
        sum_bg += t as f64 * c as f64;
        let m_bg = sum_bg / w_bg as f64;
        let m_fg = (sum_total - sum_bg) / w_fg as f64;
        let var = w_bg as f64 * w_fg as f64 * (m_bg - m_fg).powi(2);
        if var > max_var {
            max_var = var;
            thr = t as u8;
        }
    }
    let mut out = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, Luma([if p.0[0] <= thr { 0 } else { 255 }]));
    }
    out
}

fn morphological_open(bin: &GrayImage, radius: u32) -> GrayImage {
    let eroded = morphological_erode(bin, radius);
    morphological_dilate(&eroded, radius)
}

fn morphological_erode(bin: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = bin.dimensions();
    let mut out = GrayImage::from_pixel(w, h, Luma([255u8]));
    let r = radius as i32;
    for y in 0..h {
        for x in 0..w {
            let mut is_black = true;
            'outer: for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        is_black = false;
                        break 'outer;
                    }
                    if bin.get_pixel(nx as u32, ny as u32).0[0] != 0 {
                        is_black = false;
                        break 'outer;
                    }
                }
            }
            if is_black {
                out.put_pixel(x, y, Luma([0u8]));
            }
        }
    }
    out
}

fn morphological_dilate(bin: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = bin.dimensions();
    let mut out = GrayImage::from_pixel(w, h, Luma([255u8]));
    let r = radius as i32;
    for y in 0..h {
        for x in 0..w {
            if bin.get_pixel(x, y).0[0] == 0 {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                            out.put_pixel(nx as u32, ny as u32, Luma([0u8]));
                        }
                    }
                }
            }
        }
    }
    out
}

fn gaussian_blur_3x3(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return img.clone();
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for dy in 0i32..3 {
                for dx in 0i32..3 {
                    let ny = (y as i32 + dy - 1).clamp(0, h as i32 - 1) as u32;
                    let nx = (x as i32 + dx - 1).clamp(0, w as i32 - 1) as u32;
                    let weight = match (dy, dx) {
                        (1, 1) => 4u32,
                        (0, 1) | (1, 0) | (1, 2) | (2, 1) => 2,
                        _ => 1,
                    };
                    acc += img.get_pixel(nx, ny).0[0] as u32 * weight;
                }
            }
            out.put_pixel(x, y, Luma([(acc / 16) as u8]));
        }
    }
    out
}

fn threshold_mid(img: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, Luma([if p.0[0] < 128 { 0 } else { 255 }]));
    }
    out
}

// ===========================================================================
// Rendering
// ===========================================================================

/// Resolve a character to a glyph ID, optionally using an OT feature
/// glyph override map for substitution.
fn resolve_glyph(font: &FontRef, ch: char, overrides: Option<&[(char, u16)]>) -> ab_glyph::GlyphId {
    if let Some(map) = overrides {
        if let Some(&(_c, gid)) = map.iter().find(|(c, _)| *c == ch) {
            return ab_glyph::GlyphId(gid);
        }
    }
    font.glyph_id(ch)
}

fn render_text_gray(font: &FontRef, text: &str, px_height: f32, overrides: Option<&[(char, u16)]>) -> GrayImage {
    let scale = PxScale::from(px_height);
    let sf = font.as_scaled(scale);

    let mut total_w = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for ch in text.chars() {
        let gid = resolve_glyph(font, ch, overrides);
        if let Some(p) = prev {
            total_w += sf.kern(p, gid);
        }
        total_w += sf.h_advance(gid);
        prev = Some(gid);
    }

    let img_w = (total_w.ceil() as u32).max(1);
    let img_h = (px_height.ceil() as u32).max(1);
    let mut img = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));

    let ascent = sf.ascent();
    let mut cx = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;

    for ch in text.chars() {
        let gid = resolve_glyph(font, ch, overrides);
        if let Some(p) = prev {
            cx += sf.kern(p, gid);
        }
        let glyph = gid.with_scale_and_position(scale, point(cx, ascent));
        if let Some(og) = font.outline_glyph(glyph) {
            let bounds = og.px_bounds();
            let bx = bounds.min.x as i32;
            let by = bounds.min.y as i32;
            og.draw(|gx, gy, cov| {
                let px = gx as i32 + bx;
                let py = gy as i32 + by;
                if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
                    let val = (255.0 * (1.0 - cov)) as u8;
                    let cur = img.get_pixel(px as u32, py as u32).0[0];
                    img.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
                }
            });
        }
        cx += sf.h_advance(gid);
        prev = Some(gid);
    }
    img
}

// ===========================================================================
// Helpers
// ===========================================================================

fn resize_to(img: &GrayImage, target_w: u32, target_h: u32) -> GrayImage {
    image::imageops::resize(
        img,
        target_w.max(1),
        target_h.max(1),
        image::imageops::FilterType::Lanczos3,
    )
}

fn safe_crop(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    let (iw, ih) = img.dimensions();
    let x = x.min(iw.saturating_sub(1));
    let y = y.min(ih.saturating_sub(1));
    let w = w.min(iw - x);
    let h = h.min(ih - y);
    if w == 0 || h == 0 {
        return GrayImage::new(1, 1);
    }
    image::imageops::crop_imm(img, x, y, w, h).to_image()
}
