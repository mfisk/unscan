pub mod kern;
pub mod types;
pub mod render;
pub mod shape;
pub mod verify;

pub use kern::{KernProvider, NoKern};

/// Re-export underlying font crates for use by main crate without direct dep,
/// preserving single version and allowing hot inlinable functions to stay in main.
pub use ab_glyph;
pub use freetype;
pub use rustybuzz;
pub use ttf_parser;

pub use types::{Bbox, FontKey, GlyphId, RenderParams, ShapedGlyph, ShapedRun, ShapedWord, FontMeta, FeatureTag, Variation, KerningPair, GlyphBBox, RenderResult, AaMode};

pub mod prelude {
    pub use crate::types::*;
    pub use crate::render::render_glyph_at_ink_height;
    pub use crate::shape::{shape_word, FaceHandle};
    pub use crate::verify::{verify_render_batch, VerifyParams};
}
