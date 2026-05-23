//! Word-level SSIM font reranking.
//!
//! After the character-level index produces a ranked list of candidate fonts,
//! this module crops full words from the scanned page, renders the same text
//! in each candidate font, and picks the best match via SSIM.
//!
//! Word-level matching sidesteps Tesseract's imprecise character bounding
//! boxes — word bboxes are reliable.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma};
use log::debug;
use std::collections::HashMap;

use crate::audit;

/// Word rerank uses all CI candidates — no artificial cap.

/// Minimum word length (characters) to include in word-level voting.
const MIN_WORD_LEN: usize = 3;

/// Minimum OCR confidence to include a word in voting.
const MIN_WORD_CONF: f32 = 50.0;

/// Maximum vertical pixel shift to search during SSIM alignment.


// ── Public API ──────────────────────────────────────────────────────────────

/// A candidate font for word-level reranking.
pub struct WordMatchCandidate<'a> {
    pub name: String,
    pub font_data: &'a [u8],
}

/// A word with its bounding box and text, for word-level matching.
pub struct WordBBox {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
    /// Line-level vertical bounds for clamping (Tesseract word bboxes bleed into adjacent lines).
    pub line_y: u32,
    pub line_h: u32,
}

/// Context for diagnostic collection during word reranking.
pub struct WordRerankDiagCtx {
    pub page: usize,
    pub line: usize,
}

/// Rerank candidate fonts using word-level SSIM.
///
/// For each word in the line that meets quality thresholds, crops the word
/// from the page image, renders it in each candidate font, and computes SSIM.
/// Returns the font name that wins the majority vote across words, or None
/// if no words could be matched.
///
/// When `audit_imgs` is Some, saves crop/render images and populates word-level
/// audit entries.
pub fn word_level_rerank(
    page_gray: &GrayImage,
    words: &[WordBBox],
    candidates: &[WordMatchCandidate],
    audit_imgs: Option<(&audit::AuditImageDir, &WordRerankDiagCtx)>,
) -> (Option<String>, Vec<audit::WordAudit>) {
    if candidates.is_empty() || words.is_empty() {
        return (None, Vec::new());
    }

    let (pw, ph) = page_gray.dimensions();

    // Parse candidate fonts
    let parsed: Vec<(&str, FontRef)> = candidates
        .iter()
        .filter_map(|c| {
            FontRef::try_from_slice(c.font_data)
                .ok()
                .map(|f| (c.name.as_str(), f))
        })
        .collect();

    if parsed.is_empty() {
        return (None, Vec::new());
    }

    // Filter to usable words
    let mut usable_words: Vec<&WordBBox> = words
        .iter()
        .filter(|w| {
            w.text.len() >= MIN_WORD_LEN
                && w.confidence >= MIN_WORD_CONF
                && w.width >= 6
                && w.height >= 6
                && w.x + w.width <= pw
                && w.y + w.height <= ph
        })
        .collect();

    if usable_words.is_empty() {
        return (None, Vec::new());
    }

    // Cap to best 4 words — longer words have more signal.  Beyond 4 the
    // marginal gain is negligible while the render cost is linear.
    usable_words.sort_by(|a, b| b.text.len().cmp(&a.text.len()));
    usable_words.truncate(4);

    let mut font_votes: HashMap<&str, u32> = HashMap::new();
    let mut total_words = 0u32;
    let mut word_diags: Vec<audit::WordAudit> = Vec::new();

    for (wi, word) in usable_words.iter().enumerate() {
        // Crop word from page, clamped to line vertical bounds
        let crop_y = word.y.max(word.line_y);
        let word_bottom = word.y + word.height;
        let line_bottom = word.line_y + word.line_h;
        let crop_bottom = word_bottom.min(line_bottom);
        let crop_h = crop_bottom.saturating_sub(crop_y).max(1);

        let raw_crop = image::imageops::crop_imm(
            page_gray,
            word.x,
            crop_y,
            word.width,
            crop_h,
        )
        .to_image();

        // Trim vertical whitespace — Tesseract bboxes include interline gap
        let crop = trim_whitespace(&raw_crop);

        // Save crop for audit
        let crop_path = if let Some((aid, ctx)) = &audit_imgs {
            aid.save_crop(ctx.page, ctx.line, wi, &word.text, &crop)
        } else {
            String::new()
        };

        // Score all candidates
        let mut all_scores: Vec<(&str, f32, i32, Option<image::GrayImage>)> = Vec::new();

        // Save the crop as SSIM actually sees it (trimmed) for audit
        let mut audit_crop_saved = false;

        for (fname, font) in &parsed {
            let rendered = match render_word(font, &word.text, crop.width(), crop.height()) {
                Some(r) => r,
                None => continue,
            };

            let result = ssim_compare(&crop, &rendered);

            // Save the SSIM-processed crop once (same for all candidates)
            if !audit_crop_saved && audit_imgs.is_some() {
                if let Some((aid, ctx)) = &audit_imgs {
                    let compared_crop_path = format!("crops/p{}_l{}_w{}_{}_compared.png",
                        ctx.page, ctx.line, wi,
                        word.text.chars().take(15)
                            .map(|c| if c.is_alphanumeric() { c } else { '_' })
                            .collect::<String>());
                    let _ = result.crop_compared.save(aid.dir.join(&compared_crop_path));
                }
                audit_crop_saved = true;
            }

            let diag_render = if audit_imgs.is_some() {
                Some(result.render_compared)
            } else {
                None
            };
            all_scores.push((fname, result.score, result.dy, diag_render));
        }

        // Sort by SSIM descending
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Vote for the best
        if let Some((best_font, _best_ssim, _dy, _)) = all_scores.first() {
            if !best_font.is_empty() {
                *font_votes.entry(best_font).or_insert(0) += 1;
                total_words += 1;
            }
        }

        // Collect audit entry for this word
        if let Some((aid, ctx)) = &audit_imgs {
            let mut cand_entries: Vec<audit::WordCandidateAudit> = Vec::new();
            // Save top 5 renders — these are the actual images SSIM compared
            for (rank, (fname, ssim_score, dy, rendered_opt)) in all_scores.iter().enumerate() {
                let render_path = if rank < 5 {
                    if let Some(ref rendered) = rendered_opt {
                        aid.save_render(ctx.page, ctx.line, wi, fname, rendered)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                cand_entries.push(audit::WordCandidateAudit {
                    font_key: fname.to_string(),
                    ssim: *ssim_score,
                    dy: *dy,
                    render_path,
                });
                if rank >= 9 { break; } // top 10 in data
            }

            let winner = all_scores.first().map(|(n, _, _, _)| n.to_string());
            word_diags.push(audit::WordAudit {
                text: word.text.clone(),
                bbox: [word.x, crop_y, word.width, crop_h],
                crop_path,
                candidates: cand_entries,
                winner,
            });
        }
    }

    if total_words == 0 {
        return (None, Vec::new());
    }

    // Winner = font with most votes.
    // Tiebreaker: prefer the font that appears earlier in `parsed` (= higher CI rank).
    let winner = font_votes
        .iter()
        .max_by(|(a_name, a_votes), (b_name, b_votes)| {
            a_votes.cmp(b_votes).then_with(|| {
                // Lower index in parsed = higher CI rank = preferred
                let a_idx = parsed.iter().position(|(n, _)| n == *a_name).unwrap_or(usize::MAX);
                let b_idx = parsed.iter().position(|(n, _)| n == *b_name).unwrap_or(usize::MAX);
                b_idx.cmp(&a_idx) // reverse: lower index wins
            })
        })
        .map(|(&f, &v)| {
            debug!(
                "  word rerank: winner='{}' with {}/{} votes ({} candidates)",
                f,
                v,
                total_words,
                parsed.len()
            );
            f.to_string()
        });

    (winner, word_diags)
}

// ── Word rendering ──────────────────────────────────────────────────────────

/// Render `text` in `font` onto a canvas of `canvas_w × canvas_h`, width-matched.
fn render_word(font: &FontRef, text: &str, canvas_w: u32, canvas_h: u32) -> Option<GrayImage> {
    if text.is_empty() || canvas_w < 4 || canvas_h < 4 {
        return None;
    }

    let em_px = width_matched_em(font, text, canvas_w as f32)?;
    let scale = PxScale::from(em_px);
    let sf = font.as_scaled(scale);

    let ink_h = sf.ascent() - sf.descent();
    // Use enough height for full font metrics even if the crop is shorter
    let actual_h = (canvas_h as f32).max(ink_h + 4.0) as u32;
    let baseline = (actual_h as f32 - ink_h) / 2.0 + sf.ascent();

    // First pass: find leftmost pixel extent (glyphs like 'j' can go left of origin)
    let mut min_px_x = 0i32;
    {
        let mut cx = 0.0f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for c in text.chars() {
            let gid = font.glyph_id(c);
            if let Some(p) = prev {
                cx += sf.kern(p, gid);
            }
            let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
            if let Some(og) = font.outline_glyph(glyph) {
                let bx = og.px_bounds().min.x as i32;
                min_px_x = min_px_x.min(bx);
            }
            cx += sf.h_advance(gid);
            prev = Some(gid);
        }
    }

    // Offset so nothing is clipped on the left
    let x_offset = if min_px_x < 0 { -min_px_x } else { 0 };
    let padded_w = canvas_w as i32 + x_offset;
    let mut canvas = GrayImage::from_pixel(padded_w as u32, actual_h, Luma([255u8]));

    let mut cx = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    let (cw, ch) = canvas.dimensions();

    for c in text.chars() {
        let gid = font.glyph_id(c);
        if let Some(p) = prev {
            cx += sf.kern(p, gid);
        }
        let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
        if let Some(og) = font.outline_glyph(glyph) {
            let bounds = og.px_bounds();
            let bx = bounds.min.x as i32 + x_offset;
            let by = bounds.min.y as i32;
            og.draw(|gx, gy, cov| {
                let px = gx as i32 + bx;
                let py = gy as i32 + by;
                if px >= 0 && py >= 0 && (px as u32) < cw && (py as u32) < ch {
                    let val = (255.0 * (1.0 - cov)) as u8;
                    let cur = canvas.get_pixel(px as u32, py as u32).0[0];
                    canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
                }
            });
        }
        cx += sf.h_advance(gid);
        prev = Some(gid);
    }

    Some(canvas)
}

fn width_matched_em(font: &FontRef, text: &str, target_w: f32) -> Option<f32> {
    let ref_h = 100.0f32;
    let sf = font.as_scaled(PxScale::from(ref_h));
    let mut adv = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for c in text.chars() {
        let gid = font.glyph_id(c);
        if let Some(p) = prev {
            adv += sf.kern(p, gid);
        }
        adv += sf.h_advance(gid);
        prev = Some(gid);
    }
    if adv < 0.1 {
        return None;
    }
    Some((ref_h * (target_w / adv)).clamp(4.0, 500.0))
}

// ── SSIM ────────────────────────────────────────────────────────────────────

/// Simple global SSIM between two grayscale images of the same size.
/// If sizes differ, the smaller is padded with white (255).
fn ssim(a: &GrayImage, b: &GrayImage) -> f32 {
    let w = a.width().max(b.width());
    let h = a.height().max(b.height());
    if w == 0 || h == 0 {
        return 0.0;
    }

    let get_a = |x: u32, y: u32| -> f64 {
        if x < a.width() && y < a.height() {
            a.get_pixel(x, y).0[0] as f64
        } else {
            255.0
        }
    };
    let get_b = |x: u32, y: u32| -> f64 {
        if x < b.width() && y < b.height() {
            b.get_pixel(x, y).0[0] as f64
        } else {
            255.0
        }
    };

    let n = (w * h) as f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;
    let mut sum_ab = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let va = get_a(x, y);
            let vb = get_b(x, y);
            sum_a += va;
            sum_b += vb;
            sum_a2 += va * va;
            sum_b2 += vb * vb;
            sum_ab += va * vb;
        }
    }

    let mu_a = sum_a / n;
    let mu_b = sum_b / n;
    let var_a = (sum_a2 / n) - mu_a * mu_a;
    let var_b = (sum_b2 / n) - mu_b * mu_b;
    let cov = (sum_ab / n) - mu_a * mu_b;

    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2);
    let den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
    (num / den) as f32
}

/// Result from SSIM comparison including the actual processed images.
pub struct SsimResult {
    pub score: f32,
    pub dy: i32,
    /// The crop image as actually compared (trimmed + resized).
    pub crop_compared: GrayImage,
    /// The render image as actually compared (trimmed + resized).
    pub render_compared: GrayImage,
}

/// SSIM with ink-band normalization: trim both images to their ink content,
/// resize to the same height, then compare.
fn ssim_compare(crop: &GrayImage, render: &GrayImage) -> SsimResult {
    let a_trimmed = trim_whitespace(crop);
    let b_trimmed = trim_whitespace(render);

    if a_trimmed.width() == 0 || a_trimmed.height() == 0
        || b_trimmed.width() == 0 || b_trimmed.height() == 0
    {
        return SsimResult {
            score: 0.0, dy: 0,
            crop_compared: a_trimmed.clone(),
            render_compared: b_trimmed.clone(),
        };
    }

    // Use the larger dimensions so neither gets upscaled much
    let target_w = a_trimmed.width().max(b_trimmed.width());
    let target_h = a_trimmed.height().max(b_trimmed.height());

    let a_resized = image::imageops::resize(
        &a_trimmed, target_w, target_h, image::imageops::FilterType::Lanczos3,
    );
    let b_resized = image::imageops::resize(
        &b_trimmed, target_w, target_h, image::imageops::FilterType::Lanczos3,
    );

    let score = ssim(&a_resized, &b_resized);
    SsimResult {
        score,
        dy: 0,
        crop_compared: a_resized,
        render_compared: b_resized,
    }
}

/// Trim whitespace from all four edges. Crops to the bounding box of ink
/// pixels — dots on j/i, diacritics, and descenders are all preserved.
fn trim_whitespace(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w == 0 || h < 6 {
        return img.clone();
    }

    const INK_THRESH: u8 = 230;
    const MIN_INK_ROW: f32 = 0.01;
    const MIN_INK_COL: f32 = 0.01;

    // Vertical: find first/last ink rows
    let row_ink: Vec<f32> = (0..h).map(|y| {
        let dark: u32 = (0..w).map(|x| {
            if img.get_pixel(x, y).0[0] < INK_THRESH { 1u32 } else { 0 }
        }).sum();
        dark as f32 / w as f32
    }).collect();

    let first_row = match row_ink.iter().position(|&d| d > MIN_INK_ROW) {
        Some(y) => y as u32,
        None => return img.clone(),
    };
    let last_row = match row_ink.iter().rposition(|&d| d > MIN_INK_ROW) {
        Some(y) => y as u32,
        None => return img.clone(),
    };

    // Horizontal: find first/last ink columns
    let col_ink: Vec<f32> = (0..w).map(|x| {
        let dark: u32 = (0..h).map(|y| {
            if img.get_pixel(x, y).0[0] < INK_THRESH { 1u32 } else { 0 }
        }).sum();
        dark as f32 / h as f32
    }).collect();

    let first_col = match col_ink.iter().position(|&d| d > MIN_INK_COL) {
        Some(x) => x as u32,
        None => 0,
    };
    let last_col = match col_ink.iter().rposition(|&d| d > MIN_INK_COL) {
        Some(x) => x as u32,
        None => w - 1,
    };

    let band_h = last_row - first_row + 1;
    let band_w = last_col - first_col + 1;
    if band_h < 4 || band_w < 4 {
        return img.clone();
    }

    image::imageops::crop_imm(img, first_col, first_row, band_w, band_h).to_image()
}
