//! Audit log — single JSON sidecar with all pipeline decisions, CI detail,
//! word-level SSIM scores, and image references (crops + renders).

use crate::error::ScanTextError;
use serde::Serialize;
use std::path::{Path, PathBuf};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssim_pass: Option<bool>,
    pub decision: Decision,
    pub reason: String,
    pub bbox: BBox,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_candidates: Vec<CiCandidate>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_char_votes: Vec<CharCiVote>,
}

/// CI candidate font score.
#[derive(Debug, Serialize, Clone)]
pub struct CiCandidate {
    pub font_key: String,
    pub score: f32,
}

/// Per-character CI vote detail.
#[derive(Debug, Serialize, Clone)]
pub struct CharCiVote {
    pub ch: char,
    pub crop_index: usize,
    pub min_dist_sq: f32,
    pub passed_gate: bool,
    pub nearest: Vec<(String, f32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_path: Option<String>,
    /// Squared distance to the chosen (font_matched) font's reference glyph.
    /// Populated after the winner is determined so the report can show
    /// per-character match quality for the unscan pick.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_dist_sq: Option<f32>,
    /// When the OCR correction gate fires, the original OCR character.
    /// `ch` holds the corrected (better-matching) character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_corrected_from: Option<char>,
    /// Best alternative character considered (even if correction didn't fire).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_alt_char: Option<char>,
    /// Distance of the best alternative character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_alt_dist: Option<f32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images_dir: Option<String>,
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

// ── Image directory for audit artifacts ─────────────────────────────────

/// Manages the audit image directory alongside the audit JSON.
pub struct AuditImageDir {
    pub dir: PathBuf,
}

impl AuditImageDir {
    /// Create the image directory structure.
    /// `audit_json_path` is the path to the audit JSON file; images go in
    /// a sibling directory named `<stem>.audit/`.
    pub fn from_audit_path(audit_json_path: &Path) -> std::io::Result<Self> {
        let stem = audit_json_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // Strip ".audit" if the stem already ends with it (from "foo.audit.json")
        let base = stem.strip_suffix(".audit").unwrap_or(&stem);
        let dir = audit_json_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}.audit", base));
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Relative path from the audit dir to use in JSON references.
    pub fn rel_dir(&self) -> String {
        self.dir.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }
}
