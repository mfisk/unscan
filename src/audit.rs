//! Audit log — single JSON sidecar with all pipeline decisions, font-matching detail,
//! word-level similarity scores, and image references (crops + renders).
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ScanTextError;
use serde::{Deserialize, Serialize, Serializer};
use crate::classifier::ObsStats;

use std::path::{Path, PathBuf};

/// Per-word segmentation summary, embedded in the audit entry so the report
/// doesn't need to crawl summary.json files from disk.
#[derive(Debug, Serialize, Deserialize, Clone)]
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

/// Serialize only the first 10 font candidates. The ranked list is
/// score-descending (see `font_match::identify_fonts`, which sorts
/// `scored` descending), so these are the top 10. The in-memory Vec
/// stays full — the report generator reads it in memory and the GT
/// rank/score are recorded separately (`gt_rank`/`gt_score`) — only
/// the JSON writer truncates. The full list (typically ~700 entries
/// per line) was the audit JSON's biggest component and no tool reads
/// past the top 10.
fn serialize_top10<S>(candidates: &Vec<FontCandidate>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let n = candidates.len().min(10);
    candidates[..n].serialize(serializer)
}

/// Per-text-region audit record.
#[derive(Debug, Serialize, Deserialize)]
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
    #[serde(default, skip_serializing_if = "Vec::is_empty", serialize_with = "serialize_top10")]
    pub font_candidates: Vec<FontCandidate>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obs_votes: Vec<ObservationVote>,
    /// GT font's own per-observation votes, on GT's own segmentation
    /// (GT's collapsed ligature set: collapsed for lig GT, uncollapsed
    /// for nolig GT). The report reads GT's per-char data on GT's own
    /// segmentation from these — the nolig-cell components and the
    /// GT-geo fallback when the winner's segmentation collapses a
    /// ligature the GT font renders as separate characters.
    /// Omitted from the JSON when identical to obs_votes; readers must
    /// treat a missing gt_obs_votes as equal to obs_votes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gt_obs_votes: Vec<ObservationVote>,
    /// Word-level bounding boxes for this line (for miss-report visualisation).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub word_bboxes: Vec<WordBBox>,
    /// Raw Tesseract word bboxes before post-processing (clip/drop/expand).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub word_bboxes_raw: Vec<WordBBox>,
    /// font tie-break candidates with per-candidate similarity (ZNCC) scores.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    /// 1-based rank of the ground-truth font in the full ranked
    /// `font_candidates` list (None when GT is absent from the list or
    /// no GT is available). Populated when --audit is provided.
    /// The JSON records only the top 10 candidates (see
    /// `serialize_top10`), so the rank is kept explicitly — GT ranks
    /// below the top 10 on a substantial minority of lines.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_rank: Option<usize>,
    /// Score of the ground-truth font in the full ranked
    /// `font_candidates` list (None when GT is absent from the list or
    /// no GT is available). Populated when --audit is provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_score: Option<f32>,
    /// Ground-truth font size in points, from the vector PDF span
    /// (None when no GT is available). Populated when --audit is
    /// provided. The report's Size row reads this; without it a
    /// regenerated report cannot show the GT size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gt_font_size_pt: Option<f32>,
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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fast_path: bool,
    /// Per-word segmentation summaries for this line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub word_segmentation: Vec<WordSegSummary>,
    /// GT font per-word segmentation summaries – always included when GT scored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gt_word_segmentation: Vec<WordSegSummary>,
    /// Per-character GT geometry log-likelihoods (h_ll + v_ll) in observation order.
    /// Enables offline evaluation of pruning statistics (min, mean, etc.) without re-running.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gt_per_char_geo_ll: Vec<f32>,
    /// Whether GT would have been pruned by the geometry gate (min_ll < threshold)
    /// if not protected by ensure_set. Flags lines where --test changes the outcome.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gt_pruned: bool,
    /// PFLDA OCR corrections with decision data.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ocr_corrections: Vec<OcrCorrection>,
}

/// A single PFLDA OCR correction with its decision data.
#[derive(Debug, Serialize, Deserialize, Clone)]
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
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WordBBox {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

/// font candidate font score.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FontCandidate {
    pub font_key: String,
    pub score: Option<f32>,
}

/// Similarity (ZNCC) tie-break candidate detail.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TieCandidate {
    pub font_key: String,
    pub family_name: String,
    pub similarity_score: f32,
    /// Whether this candidate was the tie-break winner.
    pub winner: bool,
}

/// Per-window font-scoring vote detail (1-gram or bigram).
#[derive(Debug, Serialize, Deserialize, Clone)]
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
    /// Likely overlap error (px) subtracted from |h_err| for the graded
    /// downweight applied to this pair's h_ll. Present only when the
    /// overlap downweight fired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_h_overlap_correction_px: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_v_ll: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_h_err: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_v_err: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_h_ll: Option<f32>,
    /// Likely overlap error (px) subtracted from |h_err| for the graded
    /// downweight applied to this pair's h_ll. Present only when the
    /// overlap downweight fired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_h_overlap_correction_px: Option<f32>,
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
    // ── Absolute midpoint/pitch data (Finding 10) ──────────────────
    // Observed and predicted character midpoints (word-image px), pitches
    // (px; None for word-initial chars, which have no previous char),
    // per-word scale (px per font unit), and observed ink width (px).
    // Recorded so per-line geometry can be re-derived from audit.json.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_obs_cx: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_pred_cx: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_obs_pitch: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_pred_pitch: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_geo_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_ink_width: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_obs_cx: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_pred_cx: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_obs_pitch: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_pred_pitch: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_geo_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt_ink_width: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
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
    /// Drop the gt_obs_votes key from entries where it duplicates
    /// obs_votes (same values, i.e. the winner's segmentation equals
    /// GT's own segmentation). Readers treat a missing gt_obs_votes as
    /// equal to obs_votes. A field-level skip_serializing_if cannot do
    /// this comparison (it only sees the one field), so the dedup runs
    /// here on the serialized Value. The in-memory entries are untouched.
    fn dedup_gt_obs_votes(value: &mut serde_json::Value) {
        if let Some(entries) = value
            .get_mut("text_entries")
            .and_then(|v| v.as_array_mut())
        {
            for entry in entries {
                let dup = matches!(
                    (entry.get("obs_votes"), entry.get("gt_obs_votes")),
                    (Some(o), Some(g)) if o == g
                );
                if dup {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.remove("gt_obs_votes");
                    }
                }
            }
        }
    }

    pub fn write_to_file(&self, path: &Path) -> Result<(), ScanTextError> {
        let mut value =
            serde_json::to_value(self).map_err(|e| ScanTextError::Serialize(e.to_string()))?;
        Self::dedup_gt_obs_votes(&mut value);
        let json =
            serde_json::to_string(&value).map_err(|e| ScanTextError::Serialize(e.to_string()))?;
        write_json_gz(path, &json)
    }
}

/// Write `json` (already serialized, compact) gzip-compressed to `path`
/// with `.gz` appended, e.g. `audit.json` -> `audit.json.gz`. Readers
/// accept both `.json.gz` and plain `.json` (see tests/common/mod.rs).
pub fn write_json_gz(path: &Path, json: &str) -> Result<(), ScanTextError> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut gz_name = path.as_os_str().to_owned();
    gz_name.push(".gz");
    let file = std::fs::File::create(PathBuf::from(gz_name)).map_err(ScanTextError::Io)?;
    let mut enc = GzEncoder::new(file, Compression::default());
    enc.write_all(json.as_bytes()).map_err(ScanTextError::Io)?;
    enc.finish().map_err(ScanTextError::Io)?;
    Ok(())
}

/// A saved audit file's entries plus the run metadata the report header
/// needs. Loaded by --report-from-audit so a report can be regenerated
/// without re-running the pipeline.
pub struct AuditFileData {
    pub entries: Vec<AuditEntry>,
    pub dpi: u32,
    pub classifier: String,
    pub render_scale: u32,
    pub render_aa: String,
    pub render_binarize: Option<u8>,
    pub elapsed_secs: f64,
}

/// Load a saved audit JSON (.json or .json.gz) for report regeneration.
/// A `.json` path whose `.gz` sibling exists resolves to the sibling,
/// matching the reader convention in tests/common/mod.rs. Errors loudly
/// on a missing file, bad gzip, malformed JSON, or missing entries —
/// never silently produces a partial report.
pub fn load_audit_file(path: &Path) -> Result<AuditFileData, ScanTextError> {
    use std::io::Read;
    let resolved: PathBuf = if path.exists() {
        path.to_path_buf()
    } else {
        let mut gz_name = path.as_os_str().to_owned();
        gz_name.push(".gz");
        let gz_path = PathBuf::from(gz_name);
        if gz_path.exists() {
            gz_path
        } else {
            return Err(ScanTextError::Serialize(format!(
                "audit file not found: {} (nor {}.gz)",
                path.display(),
                path.display()
            )));
        }
    };
    let is_gz = resolved.extension().map_or(false, |e| e == "gz");
    let json: String = if is_gz {
        let file = std::fs::File::open(&resolved).map_err(ScanTextError::Io)?;
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut buf = String::new();
        decoder
            .read_to_string(&mut buf)
            .map_err(|e| ScanTextError::Serialize(format!("bad gzip in {}: {e}", resolved.display())))?;
        buf
    } else {
        std::fs::read_to_string(&resolved).map_err(ScanTextError::Io)?
    };
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| ScanTextError::Serialize(format!("malformed audit JSON in {}: {e}", resolved.display())))?;
    let obj = value
        .as_object()
        .ok_or_else(|| ScanTextError::Serialize(format!("audit JSON in {} is not an object", resolved.display())))?;
    let entries_value = obj.get("text_entries").ok_or_else(|| {
        ScanTextError::Serialize(format!("audit JSON in {} has no text_entries", resolved.display()))
    })?;
    let mut entries: Vec<AuditEntry> = serde_json::from_value(entries_value.clone()).map_err(|e| {
        ScanTextError::Serialize(format!("cannot parse text_entries in {}: {e}", resolved.display()))
    })?;
    // The writer drops gt_obs_votes when it duplicates obs_votes, so a
    // missing key on a ground-truth line means "equal to obs_votes" —
    // restore it, or the report's GT-observation fallback table would
    // come up empty on those lines. Lines without ground truth never
    // had GT observations, so they stay empty; and a wrong restore is
    // harmless anyway, since the fallback map only keeps observations
    // whose gt_* fields are populated.
    for e in entries.iter_mut() {
        if e.gt_obs_votes.is_empty() && !e.obs_votes.is_empty() && e.expected_font.is_some() {
            e.gt_obs_votes = e.obs_votes.clone();
        }
    }
    // Run metadata for the report header. dpi is load-bearing (font-size
    // conversions); the rest only label the header, so a missing value
    // falls back to the live path's defaults rather than failing.
    let dpi = obj
        .get("dpi")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ScanTextError::Serialize(format!("audit JSON in {} has no dpi", resolved.display())))?
        as u32;
    let classifier = obj
        .get("classifier")
        .and_then(|v| v.as_str())
        .unwrap_or("lda")
        .to_string();
    let render_scale = obj
        .get("render_scale")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let render_aa = obj
        .get("render_aa")
        .and_then(|v| v.as_str())
        .unwrap_or("native")
        .to_string();
    let render_binarize = obj
        .get("render_binarize")
        .and_then(|v| v.as_u64())
        .map(|v| v as u8);
    let elapsed_secs = obj
        .get("elapsed_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    Ok(AuditFileData {
        entries,
        dpi,
        classifier,
        render_scale,
        render_aa,
        render_binarize,
        elapsed_secs,
    })
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
