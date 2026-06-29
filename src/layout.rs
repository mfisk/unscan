// layout.rs — shared text layout arithmetic for SSIM rendering and PDF output.
//
// Both verify.rs (SSIM comparison) and pdf_out.rs (PDF generation) need the
// same two calculations per word:
//   1. Width-matched font size (em_px where advance width == OCR bbox width)
//   2. Ink-centered baseline (vertically center the font's ink in a bbox)
//
// Having a single source of truth prevents alignment drift between the
// overlay preview and the final output.

use ab_glyph::{point, Font, FontRef, GlyphId, PxScale, ScaleFont};
use image::{GrayImage, Luma};

/// Reference height used for measuring advance widths before rescaling.
const REF_H: f32 = 100.0;

/// Compute the font em-height in pixels such that the advance width of `text`
/// equals `target_width_px`.  Returns `None` when the font has zero advance
/// for the given text (e.g. missing glyphs).
pub fn width_matched_em_px<F: Font>(font: &F, text: &str, target_width_px: f32, overrides: Option<&[(char, u16)]>) -> Option<f32> {
    let sf_ref = font.as_scaled(PxScale::from(REF_H));
    let mut adv = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for c in text.chars() {
        let gid = crate::char_render::resolve_glyph(font, c, overrides);
        if let Some(p) = prev {
            adv += sf_ref.kern(p, gid);
        }
        adv += sf_ref.h_advance(gid);
        prev = Some(gid);
    }
    if adv < 0.1 {
        return None;
    }
    Some((REF_H * (target_width_px / adv)).clamp(4.0, 500.0))
}

/// Given a font scaled to `em_px`, return the baseline Y position that
/// vertically centers the font's ink within a canvas/bbox of `bbox_h_px`
/// pixels.  The returned value is in **pixel coordinates** measured from
/// the top of the bbox (suitable for ab_glyph's coordinate system where
/// Y increases downward).
pub fn ink_centered_baseline_px<F: Font>(font: &F, em_px: f32, bbox_h_px: f32) -> f32 {
    let sf = font.as_scaled(PxScale::from(em_px));
    let ink_h = sf.ascent() - sf.descent(); // positive
    (bbox_h_px - ink_h) / 2.0 + sf.ascent()
}

/// Same as [`ink_centered_baseline_px`] but returns the baseline as a
/// **PDF-coordinate Y offset** from the top of the OCR bbox.  Positive
/// means *below* the top (PDF Y increases upward, so this is subtracted
/// from the top-of-bbox PDF Y).
///
/// Returns `(baseline_offset_pt, ink_h_pt)` so callers can also use the
/// ink height if needed.
pub fn ink_centered_baseline_pt<F: Font>(
    font: &F,
    em_px: f32,
    bbox_h_px: f32,
    dpi: f32,
) -> (f32, f32) {
    let sf = font.as_scaled(PxScale::from(em_px));
    let ink_h = sf.ascent() - sf.descent();
    let ink_h_pt = ink_h * 72.0 / dpi;
    let bbox_h_pt = bbox_h_px * 72.0 / dpi;
    let ascent_pt = sf.ascent() * 72.0 / dpi;
    // Offset from top of bbox to the baseline (in pt, positive downward)
    let offset = (bbox_h_pt - ink_h_pt) / 2.0 + ascent_pt;
    (offset, ink_h_pt)
}

// ---------------------------------------------------------------------------
// Baseline-aligned model (v8f) — replaces ink-centered for production use
// ---------------------------------------------------------------------------

/// Characters that have typographic descenders (extend below the baseline).
const DESCENDER_CHARS: &[char] = &[
    'g', 'j', 'p', 'q', 'y',       // lowercase
    'Q',                              // uppercase Q sometimes has a descender tail
    // Less common but can descend in some fonts:
];

/// Check if the given text line contains any descender characters.
pub fn has_descenders(text: &str) -> bool {
    text.chars().any(|c| DESCENDER_CHARS.contains(&c))
}

/// Baseline-aligned vertical position in **pixel coordinates** (Y-down).
///
/// Instead of centering the full ink extent, this estimates where the
/// typographic baseline sits in the OCR bounding box and aligns the font's
/// ascent to that position.
///
/// - If the line has descenders: the baseline is proportionally placed so
///   that `ascent / (ascent + |descent|)` of the bbox is above the baseline.
/// - If the line has NO descenders: the baseline is at the **bottom** of the
///   bbox, because OCR trims the bbox to visible ink and there's nothing
///   below the baseline.
///
/// Returns the Y position for the baseline on a canvas of `bbox_h_px` height
/// (suitable for ab_glyph rendering where Y=0 is the top of the canvas).
pub fn baseline_aligned_baseline_px<F: Font>(
    font: &F,
    em_px: f32,
    bbox_h_px: f32,
    text: &str,
) -> f32 {
    let sf = font.as_scaled(PxScale::from(em_px));
    let ascent = sf.ascent();            // positive, pixels above baseline
    let descent = sf.descent();          // negative, pixels below baseline

    if has_descenders(text) {
        // Descenders present: OCR bbox spans from top of ascenders to bottom
        // of descenders.  Estimate baseline position proportionally.
        let total_ink = ascent - descent;  // ascent + |descent|
        if total_ink < 0.1 {
            // Degenerate font — fall back to ink-centered
            return ink_centered_baseline_px(font, em_px, bbox_h_px);
        }
        // baseline_y = bbox_h * (ascent / total_ink)
        // This places the ascent portion above and descent portion below.
        bbox_h_px * (ascent / total_ink)
    } else {
        // No descenders: OCR bbox bottom IS the baseline.
        // The font's ascent should fill the bbox.
        // baseline_y = bbox_h (bottom of bbox in Y-down coords)
        //
        // But we may need a small nudge: if the font's ascent at this size
        // is taller than the bbox, text would overflow above.  We cap it.
        bbox_h_px
    }
}

/// Baseline-aligned position for **PDF output** (returns offset from top of
/// OCR bbox, in points, positive downward — same convention as
/// `ink_centered_baseline_pt`).
///
/// Returns `(baseline_offset_pt, ink_h_pt)`.
pub fn baseline_aligned_baseline_pt<F: Font>(
    font: &F,
    em_px: f32,
    bbox_h_px: f32,
    dpi: f32,
    text: &str,
) -> (f32, f32) {
    let sf = font.as_scaled(PxScale::from(em_px));
    let ink_h = sf.ascent() - sf.descent();
    let ink_h_pt = ink_h * 72.0 / dpi;
    
    let baseline_px = baseline_aligned_baseline_px(font, em_px, bbox_h_px, text);
    let baseline_offset_pt = baseline_px * 72.0 / dpi;
    (baseline_offset_pt, ink_h_pt)
}

// ---------------------------------------------------------------------------
// Shared rustybuzz shaping
// ---------------------------------------------------------------------------

/// Build OT feature list from a variant tag (e.g. "smcp", "onum").
pub fn ot_features(variant_tag: &str) -> Vec<rustybuzz::Feature> {
    if !variant_tag.is_empty() && variant_tag.len() <= 4 {
        let mut tag_bytes = [b' '; 4];
        for (i, b) in variant_tag.as_bytes().iter().enumerate().take(4) {
            tag_bytes[i] = *b;
        }
        let tag = rustybuzz::ttf_parser::Tag::from_bytes(&tag_bytes);
        vec![rustybuzz::Feature::new(tag, 1, ..)]
    } else {
        vec![]
    }
}

/// Result of shaping a word with rustybuzz, including sidebearing info.
pub struct ShapedWord {
    /// Per-glyph IDs (post-shaping).
    pub glyph_ids: Vec<u32>,
    /// Per-glyph x_advance in font units.
    pub x_advances: Vec<i32>,
    /// Per-glyph x_offset in font units.
    pub x_offsets: Vec<i32>,
    /// Per-glyph y_offset in font units.
    pub y_offsets: Vec<i32>,
    /// Total advance width in font units (sum of x_advances).
    pub total_advance_fu: f64,
    /// Ink-only advance in font units (advance minus first LSB and last RSB).
    pub ink_advance_fu: f64,
    /// First glyph's left side-bearing in font units.
    pub first_lsb_fu: f64,
    /// Units per em for this font.
    pub units_per_em: f64,
}

impl ShapedWord {
    /// Compute em_px such that ink advance == target_width_px.
    pub fn ink_matched_em_px(&self, target_width_px: f32) -> Option<f32> {
        if self.ink_advance_fu < 0.1 {
            return None;
        }
        let em_px = (target_width_px as f64 * self.units_per_em / self.ink_advance_fu) as f32;
        Some(em_px.clamp(4.0, 500.0))
    }

    /// Pixels per font-unit at a given em_px.
    pub fn px_per_unit(&self, em_px: f64) -> f64 {
        em_px / self.units_per_em
    }
}

/// Shape a word using rustybuzz and compute sidebearing-corrected metrics.
pub fn shape_word(face: &rustybuzz::Face, features: &[rustybuzz::Feature], text: &str) -> Option<ShapedWord> {
    let units_per_em = face.units_per_em() as f64;

    let mut buffer = rustybuzz::UnicodeBuffer::new();
    buffer.push_str(text);
    let glyphs = rustybuzz::shape(face, features, buffer);
    let positions = glyphs.glyph_positions();
    let infos = glyphs.glyph_infos();

    let glyph_ids: Vec<u32> = infos.iter().map(|gi| gi.glyph_id).collect();
    let x_advances: Vec<i32> = positions.iter().map(|p| p.x_advance).collect();
    let x_offsets: Vec<i32> = positions.iter().map(|p| p.x_offset).collect();
    let y_offsets: Vec<i32> = positions.iter().map(|p| p.y_offset).collect();

    let total_advance_fu: f64 = x_advances.iter().map(|&a| a as f64).sum();
    if total_advance_fu < 0.1 {
        return None;
    }

    let ttfp = face.as_ref();
    let first_lsb_fu = infos.first().and_then(|gi| {
        let gid = rustybuzz::ttf_parser::GlyphId(gi.glyph_id as u16);
        ttfp.glyph_bounding_box(gid).map(|bb| bb.x_min as f64)
    }).unwrap_or(0.0);

    let last_rsb = if let (Some(last_info), Some(last_pos)) = (infos.last(), positions.last()) {
        let gid = rustybuzz::ttf_parser::GlyphId(last_info.glyph_id as u16);
        if let Some(bb) = ttfp.glyph_bounding_box(gid) {
            let adv = last_pos.x_advance as f64;
            (adv - bb.x_max as f64).max(0.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let ink_advance_fu = total_advance_fu - first_lsb_fu - last_rsb;

    Some(ShapedWord {
        glyph_ids,
        x_advances,
        x_offsets,
        y_offsets,
        total_advance_fu,
        ink_advance_fu,
        first_lsb_fu,
        units_per_em,
    })
}

/// Compute em_px using rustybuzz shaped advances (full OT shaping).
/// More accurate than `width_matched_em_px` which uses ab_glyph (no GPOS).
/// When `variant_tag` is non-empty (e.g. "smcp", "onum"), the corresponding
/// OT feature is activated during shaping.
pub fn width_matched_em_px_shaped(font_data: &[u8], text: &str, target_width_px: f32, variant_tag: &str, variations: Option<&[([u8; 4], f32)]>) -> Option<f32> {
    let mut face = rustybuzz::Face::from_slice(font_data, 0)?;
    if let Some(vars) = variations {
        for (tag, val) in vars {
            let t = rustybuzz::ttf_parser::Tag::from_bytes(tag);
            face.set_variation(t, *val);
        }
    }
    let features = ot_features(variant_tag);
    shape_word(&face, &features, text)?.ink_matched_em_px(target_width_px)
}

// ---------------------------------------------------------------------------
// Shared ab_glyph word rendering
// ---------------------------------------------------------------------------


/// Render a word into a grayscale canvas at a given em size.
///
/// `resolve` maps each char to a glyph ID — pass `char_render::resolve_glyph`
/// for override-aware matching, or `|f, c| f.glyph_id(c)` for plain lookup.
///
/// If `canvas_h` is None, height is auto-sized from font ascent/descent.
/// If `canvas_w` is None, width is auto-sized from total advance + padding.
///
/// Returns the rendered canvas (white background, black ink).
pub fn render_word_ab_glyph(
    font: &FontRef,
    text: &str,
    em_px: f32,
    canvas_w: Option<u32>,
    canvas_h: Option<u32>,
    resolve: impl Fn(&FontRef, char) -> GlyphId,
) -> Option<GrayImage> {
    if text.is_empty() || em_px < 1.0 {
        return None;
    }

    let scale = PxScale::from(em_px);
    let sf = font.as_scaled(scale);

    let ink_h = sf.ascent() - sf.descent();
    if ink_h <= 0.0 {
        return None;
    }

    let h = canvas_h.unwrap_or_else(|| (ink_h + 4.0) as u32).max(4);
    let baseline = (h as f32 - ink_h) / 2.0 + sf.ascent();

    // First pass: find min pixel X and total advance
    let mut min_px_x = 0i32;
    let mut cx = 0.0f32;
    let mut prev: Option<GlyphId> = None;
    for c in text.chars() {
        let gid = resolve(font, c);
        if let Some(p) = prev {
            cx += sf.kern(p, gid);
        }
        let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
        if let Some(og) = font.outline_glyph(glyph) {
            min_px_x = min_px_x.min(og.px_bounds().min.x as i32);
        }
        cx += sf.h_advance(gid);
        prev = Some(gid);
    }
    let total_advance = cx;

    let x_offset = if min_px_x < 0 { -min_px_x } else { 0 };
    let w = canvas_w.unwrap_or_else(|| (total_advance as i32 + x_offset + 2).max(4) as u32);
    let padded_w = (w as i32 + x_offset) as u32;

    let mut canvas = GrayImage::from_pixel(padded_w, h, Luma([255u8]));
    let (cw, ch) = canvas.dimensions();

    // Second pass: draw
    let mut cx = 0.0f32;
    let mut prev: Option<GlyphId> = None;
    for c in text.chars() {
        let gid = resolve(font, c);
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
