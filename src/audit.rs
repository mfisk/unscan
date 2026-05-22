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
    pub decision: Decision,
    pub reason: String,
    pub bbox: BBox,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_candidates: Vec<CiCandidate>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_char_votes: Vec<CharCiVote>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordAudit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_rerank_winner: Option<String>,
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
}

/// Per-word SSIM rerank detail.
#[derive(Debug, Serialize, Clone)]
pub struct WordAudit {
    pub text: String,
    pub bbox: [u32; 4],
    #[serde(skip_serializing_if = "String::is_empty")]
    pub crop_path: String,
    pub candidates: Vec<WordCandidateAudit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub winner: Option<String>,
}

/// Per-candidate SSIM score for a word.
#[derive(Debug, Serialize, Clone)]
pub struct WordCandidateAudit {
    pub font_key: String,
    pub ssim: f32,
    pub dy: i32,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub render_path: String,
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

/// Manages saving crop and render images alongside the audit JSON.
pub struct AuditImageDir {
    pub dir: PathBuf,
    crops_dir: PathBuf,
    renders_dir: PathBuf,
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
        let crops_dir = dir.join("crops");
        let renders_dir = dir.join("renders");
        std::fs::create_dir_all(&crops_dir)?;
        std::fs::create_dir_all(&renders_dir)?;
        Ok(Self { dir, crops_dir, renders_dir })
    }

    /// Relative path from the audit dir to use in JSON references.
    pub fn rel_dir(&self) -> String {
        self.dir.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }

    /// Save a word crop image, return path relative to the audit dir.
    pub fn save_crop(
        &self,
        page: usize,
        line: usize,
        word_idx: usize,
        text: &str,
        img: &image::GrayImage,
    ) -> String {
        let safe: String = text.chars().take(15)
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let rel = format!("crops/p{}_l{}_w{}_{}.png", page, line, word_idx, safe);
        let _ = img.save(self.dir.join(&rel));
        rel
    }

    /// Save a rendered word image, return path relative to the audit dir.
    pub fn save_render(
        &self,
        page: usize,
        line: usize,
        word_idx: usize,
        font_key: &str,
        img: &image::GrayImage,
    ) -> String {
        let font_base = font_key.rsplit('/').next().unwrap_or(font_key);
        let safe_font: String = font_base.chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
            .collect();
        let rel = format!("renders/p{}_l{}_w{}_{}.png", page, line, word_idx, safe_font);
        let _ = img.save(self.dir.join(&rel));
        rel
    }
}
