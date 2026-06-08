use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "unscan",
    about = "Replace scanned (raster) text with native (vector) text.\n\n\
             Zero information loss: only replaces raster when BOTH OCR and font match\n\
             confidence are high. All remaining raster is kept at original quality.\n\
             Also vectorizes lines, rectangles, and solid fills.",
    version
)]
pub struct Args {
    /// Input file (PDF or image: PNG, JPEG, TIFF, BMP).
    /// Not required when using --index.
    pub input: Option<PathBuf>,

    /// Output PDF path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Additional font search directories (repeatable)
    #[arg(long)]
    pub font_dir: Vec<PathBuf>,

    /// Minimum OCR confidence to consider vectorizing text (0–100).
    /// Below this, the original raster is kept unconditionally.
    #[arg(long, default_value = "0")]
    pub min_ocr_confidence: u32,

    /// Minimum font-match confidence (0.0–1.0).
    /// Below this, text is kept as raster — we never replace with a wrong font.
    #[arg(long, default_value = "0.10")]
    pub min_font_confidence: f32,

    /// DPI for PDF page rasterization
    #[arg(long, default_value = "300")]
    pub dpi: u32,

    /// Skip geometry vectorization (lines, rectangles, fills)
    #[arg(long)]
    pub no_geometry: bool,

    /// Debug overlay mode: keep original raster in place and render vector
    /// text on top in semitransparent red. Useful for visually checking
    /// font matching and sizing accuracy.
    #[arg(long)]
    pub overlay: bool,

    /// Smooth font sizes: unify per-word sizes within consecutive same-font
    /// runs to their median, removing OCR bbox noise.  Outlier words (>1pt
    /// from the run mean) keep their natural size.
    #[arg(long)]
    pub smooth: bool,

    /// Audit directory.  Writes audit.json and per-word segmentation
    /// diagnostics (crops, seams, char overlays) into the given directory.
    /// Also used by tools/char-misses.py to generate the visual miss report.
    #[arg(long, value_name = "DIR")]
    pub audit: Option<PathBuf>,

    // ── Character index flags ──────────────────────────────────────
    /// Scan available fonts, update the cached character index if needed,
    /// and exit.  Incrementally adds new fonts and removes stale ones
    /// without rebuilding the entire index.
    #[arg(long)]
    pub index: bool,

    /// Path to the character index cache file.
    /// Default: ~/.cache/unscan/char-index.bin
    #[arg(long)]
    pub index_path: Option<PathBuf>,

    /// Force a full rebuild of the character index, ignoring any cache.
    #[arg(long)]
    pub rebuild_index: bool,

    /// Generate side-by-side comparison images (scan crop vs rendered font)
    /// for every vectorized line. Output goes to <output_base>-compare/ directory.
    #[arg(long)]
    pub compare: bool,

    /// Include a font in CI candidate evaluation for every line, even if the
    /// character index didn't rank it highly. Substring match against font name
    /// (case-insensitive). Useful for diagnosing why a known-correct font
    /// scores poorly in the audit.
    #[arg(long, value_name = "NAME")]
    pub include_font: Option<String>,

    /// Path to a fontmap JSON ({"FontName": "/path/to/font.ttf", ...}).
    /// All fonts referenced in the map are injected into the CI audit
    /// candidate list, like --include-font but in bulk.
    #[arg(long = "include-fontmap", value_name = "FILE")]
    pub include_fontmap: Option<std::path::PathBuf>,

    /// Thoroughness factor for font matching. Default 1.0.
    /// Higher values relax all CI thresholds (quorum, quality gate, search
    /// radius) so more candidate fonts survive to evaluation.
    /// Useful for diagnosing why a known font isn't being matched.
    #[arg(long, default_value_t = 1.0)]
    pub thoroughness: f32,

    /// Vector PDF for ground-truth comparison.  When set alongside --audit,
    /// only miss lines get full audit detail (crop PNGs, fontmap per-char
    /// distances, font ref glyphs).  Hit lines are logged with minimal data.
    /// A "miss" includes font mismatches, no font match, OCR rejection, and
    /// SSIM rejection.  Also generates an HTML miss report in the audit
    /// directory comparing unscan's font picks against ground truth.
    #[arg(long = "audit-vector", value_name = "FILE")]
    pub audit_vector: Option<std::path::PathBuf>,

    /// Render characters using the index-time render_char_normalised() pipeline
    /// and save as PNGs.  Takes a JSON object: {"font": "/path/to/font.ttf",
    /// "chars": "abc", "output_dir": "/tmp/refs"}.  Each character is saved as
    /// U+XXXX.png (e.g. U+0048.png for 'H').  Exits after rendering.
    #[arg(long, value_name = "JSON")]
    pub render_ref_chars: Option<String>,
}

impl Args {
    /// Resolve the audit JSON path.
    /// With --audit DIR, it's DIR/audit.json.
    /// Without --audit, falls back to <output>.audit.json.
    pub fn audit_log_path(&self) -> PathBuf {
        if let Some(ref dir) = self.audit {
            dir.join("audit.json")
        } else {
            let out = self.output.as_ref().expect("output required for audit log");
            let mut p = out.clone();
            let stem = p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            p.set_file_name(format!("{stem}.audit.json"));
            p
        }
    }

    /// Resolve the diag-seg directory (same as --audit DIR when set).
    pub fn diag_seg_dir(&self) -> Option<&Path> {
        self.audit.as_deref()
    }

    /// Resolve the character index path.
    /// Default: ~/.cache/unscan/char-index.bin
    pub fn resolved_index_path(&self) -> PathBuf {
        if let Some(ref p) = self.index_path {
            p.clone()
        } else {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home)
                .join(".cache")
                .join("unscan")
                .join("char-index.bin")
        }
    }

    /// Validate: if not --index, require input and output.
    pub fn validate(&self) -> Result<(), String> {
        if self.index || self.render_ref_chars.is_some() {
            return Ok(());
        }
        if self.input.is_none() {
            return Err("Input file required (or use --index)".to_string());
        }
        if self.output.is_none() {
            return Err("Output path required (use -o / --output)".to_string());
        }
        Ok(())
    }
}

pub fn parse() -> Args {
    Args::parse()
}
