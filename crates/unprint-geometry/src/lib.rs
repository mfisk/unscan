pub mod text;
pub mod bbox;
pub mod char_bounds;
pub mod detect;
pub mod word;
pub mod params;

pub use params::{SIGMA_CENTER_PX, SIGMA_PITCH_PX, quant_half_width_px, quantized_ll, audit_all_chars_enabled};

pub use text::{CharBox, TextRegion, TextLine, RawWordBBox};
pub use bbox::{Bbox, GlyphBBox, CharInkBounds, WordGeoMeasurement, center_span_scale, batch_center_span_scales, glyph_bboxes_batch_pure};
pub use char_bounds::{measure_char_ink_bounds, measure_words_ink_bounds_batch, batch_per_char_errors, PerCharError, SeamPaths};
pub use detect::{GeometryResult, DetectedLine, DetectedFill, detect_geometry, erase_bboxes, otsu_threshold, Rgb};
pub use word::{expand_words_to_ink, fix_overlapping_words_by_ink, trim_words_to_ink, ink_vertical_extent, refine_words_batch};

/// Batch API for per-char errors across many words.
pub fn batch_measure_and_error(
    inputs: &[(&image::GrayImage, &[char], &[u32], &SeamPaths)],
    pred_sets: &[Vec<(f64,f64)>],
) -> Vec<Vec<PerCharError>> {
    let measurements = measure_words_ink_bounds_batch(inputs);
    measurements.iter().zip(pred_sets.iter()).map(|(m, preds)| {
        let obs_cx: Vec<f64> = m.chars.iter().map(|c| c.cx).collect();
        let obs_cy: Vec<f64> = m.chars.iter().map(|c| c.cy).collect();
        let pred_cx: Vec<f64> = preds.iter().map(|(x,_)| *x).collect();
        let pred_cy: Vec<f64> = preds.iter().map(|(_,y)| *y).collect();
        batch_per_char_errors(&obs_cx, &obs_cy, &pred_cx, &pred_cy)
    }).collect()
}
