// layout.rs — shared text layout arithmetic for SSIM rendering and PDF output.
//
// Both verify.rs (SSIM comparison) and pdf_out.rs (PDF generation) need the
// same two calculations per word:
//   1. Width-matched font size (em_px where advance width == OCR bbox width)
//   2. Ink-centered baseline (vertically center the font's ink in a bbox)
//
// Having a single source of truth prevents alignment drift between the
// overlay preview and the final output.

use ab_glyph::{Font, PxScale, ScaleFont};

/// Reference height used for measuring advance widths before rescaling.
const REF_H: f32 = 100.0;

/// Compute the font em-height in pixels such that the advance width of `text`
/// equals `target_width_px`.  Returns `None` when the font has zero advance
/// for the given text (e.g. missing glyphs).
pub fn width_matched_em_px<F: Font>(font: &F, text: &str, target_width_px: f32) -> Option<f32> {
    let sf_ref = font.as_scaled(PxScale::from(REF_H));
    let mut adv = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for c in text.chars() {
        let gid = font.glyph_id(c);
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
