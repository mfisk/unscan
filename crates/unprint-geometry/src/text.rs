use serde::{Deserialize, Serialize};

/// A detected text region from OCR (word-level).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub font_size_pt: f32,
    pub confidence: f32,
    pub level: u32,
    pub block_num: u32,
    pub par_num: u32,
    pub line_num: u32,
    pub word_num: u32,
}

/// Lightweight snapshot of a Tesseract word bbox before post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawWordBBox {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

/// A line of text assembled from word-level regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub font_size_pt: f32,
    pub confidence: f32,
    pub words: Vec<TextRegion>,
    /// Snapshot of word bboxes before post-processing.
    pub raw_words: Vec<RawWordBBox>,
}

impl TextLine {
    pub fn new_empty() -> Self {
        Self {
            text: String::new(),
            x: 0, y: 0, width: 0, height: 0,
            font_size_pt: 12.0,
            confidence: 0.0,
            words: Vec::new(),
            raw_words: Vec::new(),
        }
    }
}
