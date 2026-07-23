pub mod types;
pub mod render;
pub mod shape;
pub mod scan;
pub mod geom;
pub mod verify;

/// Re-export underlying font crates for use by main crate without direct dep,
/// preserving single version and allowing hot inlinable functions to stay in main.
pub use ab_glyph;
pub use freetype;
pub use rustybuzz;
pub use ttf_parser;

pub use types::{Bbox, FontKey, GlyphId, RenderParams, ShapedGlyph, ShapedRun, ShapedWord, FontMeta, FeatureTag, Variation, KerningPair, GlyphBBox, RenderResult, AaMode};

pub mod prelude {
    pub use crate::types::*;
    pub use crate::render::{render_ngrams_batch, FontHandle, hash_image, hash_hex, render_ngram_single, render_glyph_at_ink_height, glyph_metric_ratios_batch};
    pub use crate::shape::{shape_words, shape_word, ot_features_for_variant, Features, FaceHandle, compute_em_px_batch};
    pub use crate::scan::{detect_ot_features, detect_ligatures, scan_font_meta, FontEntryMeta};
    pub use crate::geom::{glyph_bboxes_batch, kerning_table, GlyphKernTable};
    pub use crate::verify::{verify_render_batch, VerifyParams};
}
