//! Classify crate — font matching, depends on geometry + fonts.

pub use unprint_geometry::{Bbox, GlyphBBox, CharInkBounds, WordGeoMeasurement, PerCharError, SeamPaths};
pub use unprint_fonts;

use image::GrayImage;

/// Batch API: measure ink bounds for many words in one call, then score.
/// Hot loops stay inside geometry crate (inlined), this crate orchestrates font candidates.

pub struct MatchInput<'a> {
    pub gray: &'a GrayImage,
    pub text: &'a [char],
    pub bboxes: &'a [u32],
    pub seam: &'a SeamPaths,
}

pub struct MatchOutput {
    pub font_name: String,
    pub score: f64,
}

/// Batch measurement API: inputs: slice of match inputs -> Vec<WordGeoMeasurement>
pub fn measure_batch(inputs: &[(&GrayImage, &[char], &[u32], &SeamPaths)]) -> Vec<WordGeoMeasurement> {
    unprint_geometry::measure_words_ink_bounds_batch(inputs)
}

/// Batch bbox API: glyph bboxes for many chars in one call (pure).
pub fn glyph_bboxes_batch(
    _glyphs: &[GlyphBBox],
    _baselines: &[f64],
    _x_positions: &[f64],
    _target_word_height: f64,
    _actual_word_height: f64,
    _scales: &[f64],
) -> Vec<Bbox> {
    // Stub - real implementation lives in geometry after refactor
    Vec::new()
}

/// Simple scoring placeholder — real implementation moved from classifier.rs
/// For batch crate to compile, we provide stub that will be filled with per-char scoring.
pub fn match_piece(_input: &MatchInput) -> MatchOutput {
    MatchOutput { font_name: "NotoSans-Regular".to_string(), score: 0.0 }
}

/// Batch match: slice in, Vec out, rayon parallel over pieces.
pub fn match_pieces_batch(inputs: &[MatchInput]) -> Vec<MatchOutput> {
    use rayon::prelude::*;
    inputs.par_iter().map(|inp| match_piece(inp)).collect()
}
