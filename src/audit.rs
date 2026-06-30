//! Audit log — single JSON sidecar with all pipeline decisions, CI detail,
//! word-level similarity scores, and image references (crops + renders).

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
    pub similarity_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_pass: Option<bool>,
    pub decision: Decision,
    pub reason: String,
    pub bbox: BBox,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_candidates: Vec<CiCandidate>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_char_votes: Vec<CharCiVote>,
    /// Ligature-segmented CI candidates (when ligature sequences are present).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_candidates_lig: Vec<CiCandidate>,
    /// Ligature-segmented per-char CI votes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ci_char_votes_lig: Vec<CharCiVote>,
    /// Which segmentation path won: "plain" or "ligature".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seg_winner: Option<String>,
    /// Word-level bounding boxes for this line (for miss-report visualisation).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub word_bboxes: Vec<WordBBox>,
    /// Raw Tesseract word bboxes before post-processing (clip/drop/expand).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub word_bboxes_raw: Vec<WordBBox>,
    /// CI tie-break candidates with per-candidate similarity (ZNCC) scores.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tie_candidates: Vec<TieCandidate>,
    /// Ground-truth classification: "hit", "major_miss", "minor_miss",
    /// "similarity_failure", "kept_raster", or "no_ground_truth".
    /// Populated when --audit is provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub miss_type: Option<String>,
    /// Ground-truth expected font (PostScript name from the vector PDF).
    /// Populated when --audit is provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_font: Option<String>,
}

/// Word-level bounding box with OCR text.
#[derive(Debug, Serialize, Clone)]
pub struct WordBBox {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

/// CI candidate font score.
#[derive(Debug, Serialize, Clone)]
pub struct CiCandidate {
    pub font_key: String,
    pub score: Option<f32>,
}

/// Similarity (ZNCC) tie-break candidate detail.
#[derive(Debug, Serialize, Clone)]
pub struct TieCandidate {
    pub font_key: String,
    pub family_name: String,
    pub similarity_score: f32,
    /// Whether this candidate was the tie-break winner.
    pub winner: bool,
}

/// Per-character CI vote detail.
#[derive(Debug, Serialize, Clone)]
pub struct CharCiVote {
    pub ch: char,
    pub crop_index: usize,
    pub best_prob: f32,
    pub passed_gate: bool,
    pub nearest: Vec<(usize, f32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_path: Option<String>,
    /// 1-based rank of the chosen font among all fonts for this character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_rank: Option<usize>,
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
    /// 1-based rank of the ground-truth font among all fonts for this
    /// character, sorted by probability (1 = highest).  `None` when GT font
    /// is unknown or not in the index for this character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_font_rank: Option<usize>,
    /// Calibrated posterior probability of the chosen font for this character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_prob: Option<f32>,
    /// Calibrated posterior probability of the ground-truth font for this character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_font_prob: Option<f32>,
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
