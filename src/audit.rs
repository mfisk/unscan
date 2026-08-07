//! Audit log — single JSON sidecar with all pipeline decisions, font-matching detail,
//! word-level similarity scores, and image references (crops + renders).
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ScanTextError;
use serde::Serialize;
use crate::classifier::ObsStats;

use std::path::{Path, PathBuf};

/// Per-word segmentation summary, embedded in the audit entry so the report
/// doesn't need to crawl summary.json files from disk.
#[derive(Debug, Serialize, Clone)]
pub struct WordSegSummary {
    pub word_text: String,
    pub source_word_idx: usize,
    pub image_w: u32,
    pub image_h: u32,
    pub n_chars_expected: u32,
    pub n_segments_produced: u32,
    pub mismatch: bool,
    pub ws_splits: Vec<u32>,
    pub seam_splits: Vec<u32>,
    pub seam_paths: Arc<HashMap<u32, Vec<[u32; 2]>>>,
    pub seam_costs: Arc<HashMap<u32, crate::segment::SeamCost>>,
}

/// Per-text-region audit record.
#[derive(Debug, Serialize)]
pub struct AuditEntry {
    pub page: usize,
    pub line_index: usize,
    pub text: String,
    pub ocr_confidence: f32,
    pub font_matched: Option<String>,
    /// Font key of the actual matched font (after ZNCC tie-breaking).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_key_matched: Option<String>,
    pub font_confidence: Option<f32>,
    pub similarity_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_pass: Option<bool>,
    pub decision: Decision,
    pub reason: String,
    pub bbox: BBox,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub font_candidates: Vec<FontCandidate>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub obs_votes: Vec<ObservationVote>,
    /// Ligature-segmented font candidates (when ligature sequences are present).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub font_candidates_lig: Vec<FontCandidate>,
    /// Ligature-segmented per-observation font-scoring votes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub obs_votes_lig: Vec<ObservationVote>,
    /// Which segmentation path won: "plain" or "ligature".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seg_winner: Option<String>,
    /// Word-level bounding boxes for this line (for miss-report visualisation).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub word_bboxes: Vec<WordBBox>,
    /// Raw Tesseract word bboxes before post-processing (clip/drop/expand).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub word_bboxes_raw: Vec<WordBBox>,
    /// font tie-break candidates with per-candidate similarity (ZNCC) scores.
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
    /// Ground-truth text from the vector PDF span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_text: Option<String>,
    /// OCR-extracted text (joined word texts for this line).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_text: Option<String>,
    /// Whether OCR text matches ground truth (None when GT unavailable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_correct: Option<bool>,
    /// Midpoint-derived font size (em_px) used for verification.
    /// When present, both chosen and GT renders should use this size
    /// to avoid the bug where chosen==GT but ZNCC differs because GT used width-matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midpoint_em_px: Option<f32>,
    /// GT font's own midpoint em_px computed from GT's predicted span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_midpoint_em_px: Option<f32>,
    /// Whether this line was matched via the dominant-font fast path
    /// (skipping full classification).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub fast_path: bool,
    /// Per-word segmentation summaries for this line.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub word_segmentation: Vec<WordSegSummary>,
    /// PFLDA OCR corrections with decision data.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ocr_corrections: Vec<OcrCorrection>,
}

/// A single PFLDA OCR correction with its decision data.
#[derive(Debug, Serialize, Clone)]
pub struct OcrCorrection {
    /// Index into the character sequence (word-relative position).
    pub char_pos: usize,
    /// Segment (word) index within the line.
    pub seg_idx: usize,
    /// Original OCR character.
    pub ocr_char: char,
    /// PFLDA replacement character.
    pub replacement: char,
    /// PFLDA probability of the replacement character.
    pub replacement_p: f32,
    /// PFLDA probability of the original OCR character (None if OCR char not in font).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_p: Option<f32>,
    /// Ratio of replacement_p / ocr_p.
    pub ratio: f32,
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

/// font candidate font score.
#[derive(Debug, Serialize, Clone)]
pub struct FontCandidate {
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

/// Per-window font-scoring vote detail (1-gram or bigram).
#[derive(Debug, Serialize, Clone)]
pub struct ObservationVote {
    /// Full scored sequence (e.g. ['T','i'] for bigram, ['a'] for 1-gram).
    pub seq: Vec<char>,
    /// Weight used in scoring (0.5 for 1-gram fallback, 1.0 for bigram).
    pub weight: f32,
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
    /// Per-font LDA top-1 predicted character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pflda_top_char: Option<char>,
    /// Per-font LDA probability of top-1 prediction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pflda_top_p: Option<f32>,
    /// Per-font LDA probability of the OCR character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pflda_ocr_p: Option<f32>,
    /// Whether pflda correction gate fired and replaced the OCR char.
    pub pflda_replaced: bool,
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
    /// Raw glyph log-likelihood ( -d²/(2σ²) ) before adding geo, for report display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_glyph_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_glyph_score: Option<f32>,
    /// Raw classifier distance stats (populated when UNPRINT_OBS_STATS=1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub obs_stats: Option<ObsStats>,
    // ── Geo scores ───────────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_h_ll: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_v_ll: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_h_err: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_v_err: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_h_ll: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_v_ll: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_h_err: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_v_err: Option<f32>,
    /// Combined geo log-likelihood (h_ll + v_ll) for chosen and GT, for report display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_ll: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_ll: Option<f32>,
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
    /// Rendering DPI used for OCR and rasterisation.
    pub dpi: u32,
    /// Classifier name (e.g. "lda-32").
    pub classifier: String,
    /// Render scale multiplier for ZNCC verification renders.
    pub render_scale: u32,
    /// Anti-aliasing mode for ZNCC verification renders.
    pub render_aa: String,
    /// Binarization threshold for ZNCC verification renders, if enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render_binarize: Option<u8>,
    /// Total pipeline elapsed time in seconds.
    pub elapsed_secs: f64,
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
