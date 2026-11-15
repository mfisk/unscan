pub mod kern;
pub mod types;
pub mod verify;

pub use kern::{KernProvider, NoKern};

/// Re-export underlying font crates for use by main crate without direct dep,
/// preserving single version and allowing hot inlinable functions to stay in main.
pub use ab_glyph;
pub use freetype;
pub use rustybuzz;
pub use ttf_parser;

pub use types::{Bbox, FontKey, GlyphId, RenderParams, FontMeta, FeatureTag, Variation, KerningPair, GlyphBBox, RenderResult, AaMode};

pub mod prelude {
    pub use crate::types::*;
    pub use crate::verify::{verify_render_batch, VerifyParams};
}
