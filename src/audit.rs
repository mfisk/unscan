//! Audit log — JSON sidecar documenting every decision made per text region.

use crate::error::ScanTextError;
use serde::Serialize;
use std::path::Path;

/// Per-text-region audit record.
#[derive(Debug, Serialize)]
pub struct AuditEntry {
    pub page: usize,
    pub line_index: usize,
    pub text: String,
    pub ocr_confidence: f32,
    pub font_matched: Option<String>,
    pub font_confidence: Option<f32>,
    pub ssim_score: Option<f32>,
    pub decision: Decision,
    pub reason: String,
    pub bbox: BBox,
}

#[derive(Debug, Serialize, Clone)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Vectorized,
    KeptRaster,
}

/// Per-geometry-element audit record.
#[derive(Debug, Serialize)]
pub struct GeometryEntry {
    pub page: usize,
    pub kind: &'static str,
    pub bbox: BBox,
}

/// Per-page summary.
#[derive(Debug, Serialize)]
pub struct PageSummary {
    pub page: usize,
    pub width_px: u32,
    pub height_px: u32,
    pub lines_vectorized: u32,
    pub lines_kept_raster: u32,
    pub geometry_elements: u32,
    pub raster_fragments: u32,
}

/// Top-level audit log.
#[derive(Debug, Serialize)]
pub struct AuditLog {
    pub input_file: String,
    pub output_file: String,
    pub input_size_bytes: u64,
    pub output_size_bytes: u64,
    pub compression_ratio: f64,
    pub pages: Vec<PageSummary>,
    pub text_entries: Vec<AuditEntry>,
    pub geometry_entries: Vec<GeometryEntry>,
}

impl AuditLog {
    pub fn write_to_file(&self, path: &Path) -> Result<(), ScanTextError> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| ScanTextError::Serialize(e.to_string()))?;
        std::fs::write(path, json).map_err(ScanTextError::Io)
    }
}
