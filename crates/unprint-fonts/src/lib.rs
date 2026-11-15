pub mod kern;

pub use kern::{KernProvider, NoKern};

/// Re-export underlying font crates for use by main crate without direct dep,
/// preserving single version and allowing hot inlinable functions to stay in main.
pub use ab_glyph;
pub use freetype;
pub use rustybuzz;
pub use ttf_parser;
