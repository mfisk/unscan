//! Word-level SSIM font reranking.
//!
//! After the character-level index produces a ranked list of candidate fonts,
//! this module crops full words from the scanned page, renders the same text
//! in each candidate font, and picks the best match via SSIM.
//!
//! Word-level matching sidesteps Tesseract's imprecise character bounding
//! boxes — word bboxes are reliable.

use ab_glyph::{Font, FontRef};
use image::GrayImage;
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
    /// OT variant tag (e.g. "smcp", "onum") — empty for base fonts.
    pub variant_tag: String,
    /// Glyph overrides for OT variant rendering.
    pub glyph_overrides: crate::char_index::GlyphOverrides,
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
    let parsed: Vec<(&str, FontRef, Option<&[(char, u16)]>)> = candidates
        .iter()
        .filter_map(|c| {
            FontRef::try_from_slice(c.font_data)
                .ok()
                .map(|f| (c.name.as_str(), f, c.glyph_overrides.as_deref()))
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

        for (fname, font, overrides) in &parsed {
            let rendered = match render_word(font, &word.text, crop.width(), crop.height(), *overrides) {
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
                let a_idx = parsed.iter().position(|(n, _, _)| n == *a_name).unwrap_or(usize::MAX);
                let b_idx = parsed.iter().position(|(n, _, _)| n == *b_name).unwrap_or(usize::MAX);
                b_idx.cmp(&a_idx) // reverse: lower index wins
            })
        })
        .map(|(&f, &v)| {
            f.to_string()
        });

    (winner, word_diags)
}

// ── Word rendering ──────────────────────────────────────────────────────────

/// Render `text` in `font` onto a canvas of `canvas_w × canvas_h`, width-matched.
fn render_word(font: &FontRef, text: &str, canvas_w: u32, canvas_h: u32, overrides: Option<&[(char, u16)]>) -> Option<GrayImage> {
    if text.is_empty() || canvas_w < 4 || canvas_h < 4 {
        return None;
    }

    let em_px = crate::layout::width_matched_em_px(font, text, canvas_w as f32, overrides)?;

    crate::layout::render_word_ab_glyph(
        font, text, em_px,
        Some(canvas_w), Some(canvas_h),
        |f, c| crate::char_index::resolve_glyph(f, c, overrides),
    )
}

// ── SSIM ────────────────────────────────────────────────────────────────────

// Re-export for local use
pub use crate::ssim::SsimResult;

fn ssim_compare(crop: &GrayImage, render: &GrayImage) -> SsimResult {
    crate::ssim::ssim_compare(crop, render)
}
