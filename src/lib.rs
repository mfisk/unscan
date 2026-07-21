// Library target for integration tests.
// Includes modules needed by the test chain + their transitive deps.

pub mod cache;
pub mod error;
pub mod color;
pub mod geometry;
pub mod ocr;
pub mod verify;
pub mod features;
pub mod layout;
pub mod segment;
pub mod font_cache;
pub mod font_scan;
pub mod font_match;
pub mod pdf_out;
pub mod smooth;
pub mod audit;
pub mod classifier;
pub mod seg_diag;
pub mod compare_rasters;
pub mod zncc_classifier;
pub mod glyph_map;
pub mod char_render;
pub mod train;
#[allow(dead_code)] // ngram training/LDA infrastructure — kept for future use
pub mod ngram;
pub mod hog;

pub mod atomic_file;
pub mod geo_cache;
pub mod geometry_classifier;
pub mod ground_truth;
