use clap::Parser;
use std::path::PathBuf;

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

    /// Minimum SSIM score for the verification pass (0.0–1.0).
    /// After rendering vector text, compare with original. Below this → revert to raster.
    /// Disabled by default (0.0) — word-level SSIM in the matching stage handles quality.
    #[arg(long, default_value = "0.0")]
    pub min_verify_ssim: f32,

    /// DPI for PDF page rasterization
    #[arg(long, default_value = "300")]
    pub dpi: u32,

    /// Skip geometry vectorization (lines, rectangles, fills)
    #[arg(long)]
    pub no_geometry: bool,

    /// Skip SSIM verification pass
    #[arg(long)]
    pub no_verify: bool,

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

    /// Path for the audit-log JSON file (default: <output>.audit.json)
    #[arg(long)]
    pub audit_log: Option<PathBuf>,

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

    /// Generate a full diagnostic HTML report with CI candidates, word-level
    /// SSIM scores, crop images, and rendered comparisons. Writes to the
    /// specified directory (creates it if needed).
    #[arg(long, value_name = "DIR")]
    pub diagnostic: Option<PathBuf>,

    /// Thoroughness factor for font matching. Default 1.0.
    /// Higher values relax all CI thresholds (quorum, quality gate, kd-tree
    /// search radius) so more candidate fonts survive to word-level SSIM.
    /// Useful for diagnosing why a known font isn't being matched.
    #[arg(long, default_value_t = 1.0)]
    pub thoroughness: f32,

    /// Include a font in word-level SSIM reranking for every line, even if CI
    /// didn't select it. Substring match against font name (case-insensitive).
    /// Useful with --diagnostic to see how a known-correct font scores.
    #[arg(long, value_name = "NAME")]
    pub include_font: Option<String>,

    /// Dump per-word segmentation diagnostics: OCR bboxes, VP splits,
    /// seam splits, charbox fallback, final char crops, and images for
    /// each pass.  Writes to the specified directory.
    /// Use with a small test PDF to inspect the full segmentation pipeline.
    #[arg(long, value_name = "DIR")]
    pub diag_seg: Option<PathBuf>,
}

impl Args {
    pub fn audit_log_path(&self) -> PathBuf {
        self.audit_log.clone().unwrap_or_else(|| {
            let out = self.output.as_ref().expect("output required for audit log");
            let mut p = out.clone();
            let stem = p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            p.set_file_name(format!("{stem}.audit.json"));
            p
        })
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
        if self.index {
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
