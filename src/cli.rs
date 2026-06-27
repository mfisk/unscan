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

    /// Audit output directory.  Writes audit.json and per-word segmentation
    /// diagnostics (crops, seams, char overlays) into the given directory.
    /// When --test is also set, generates report.html with ground-truth
    /// hit/miss classification.
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

    /// Train LDA classifier weights and save to ~/.cache/unscan/lda-weights.bin.
    /// Scans all system fonts, renders characters, computes features, and trains
    /// the LDA projection.  Exits after training.
    #[arg(long)]
    pub train_lda: bool,

    /// Generate side-by-side comparison images (scan crop vs rendered font)
    /// for every vectorized line. Output goes to <output_base>-compare/ directory.
    #[arg(long)]
    pub compare: bool,

    /// Thoroughness factor for font matching. Default 1.0.
    /// Higher values relax all CI thresholds (quorum, quality gate, search
    /// radius) so more candidate fonts survive to evaluation.
    /// Useful for diagnosing why a known font isn't being matched.
    #[arg(long, default_value_t = 1.0)]
    pub thoroughness: f32,

    /// Process only the given pages (1-indexed, comma-separated, ranges ok).
    /// Examples: --pages 3  --pages 1,3,5  --pages 2-4,7
    /// Omit to process all pages.
    #[arg(long, value_name = "PAGES")]
    pub pages: Option<String>,

    /// Ground-truth vector PDF for accuracy evaluation.  Outputs performance
    /// stats as JSON to stdout.  When --audit is also set, the audit report
    /// includes ground-truth hit/miss classification.  Does not require --output.
    #[arg(long, value_name = "PDF")]
    pub test: Option<PathBuf>,

    /// Render characters using the index-time render_glyph_at_ink_height() pipeline
    /// and save as PNGs.  Takes a JSON object: {"font": "/path/to/font.ttf",
    /// "chars": "abc", "output_dir": "/tmp/refs"}.  Each character is saved as
    /// U+XXXX.png (e.g. U+0048.png for 'H').  Exits after rendering.
    #[arg(long, value_name = "JSON")]
    pub render_ref_chars: Option<String>,

    /// Font matching classifier: 'lda' (default, LDA-projected kNN),
    /// 'fisher' (Fisher-weighted kNN), 'triplet' (per-glyph learned embedding),
    /// 'global-triplet' (single learned embedding), 'mlp' (direct multi-class
    /// softmax per character), or 'fusion' (rank-fusion of LDA + Fisher).
    #[arg(long, default_value = "lda")]
    pub classifier: String,

    /// Path to classifier weights file.  Required for triplet, global-triplet,
    /// perchar-fisher, mahalanobis, and mlp.  Optional for lda (auto-trained
    /// if absent).  Not needed for fisher or fusion.
    #[arg(long)]
    pub triplet_weights: Option<PathBuf>,

    /// Normalize PostScript name(s) to include explicit weight keywords.
    /// Pass one or more "PSName:weight" pairs (e.g. "Lato-Italic:400").
    /// Prints the normalized name(s) to stdout and exits.
    #[arg(long, value_name = "PS:WEIGHT")]
    pub weight_explicit: Vec<String>,

    // ── Render pipeline parameters ─────────────────────────────────
    /// Render scale multiplier for reference character images.
    /// 1 = render directly at target height, 3 = render at 3× then downscale.
    #[arg(long, default_value = "1")]
    pub render_scale: u32,

    /// AA variant for reference character images: native, blur_0.5, sharpen.
    #[arg(long, default_value = "native")]
    pub render_aa: String,

    /// Binarize threshold for reference character images (0–255).
    /// Default: no binarization (keep native greyscale with AA).
    #[arg(long)]
    pub render_binarize: Option<u8>,
}

impl Args {
    /// Build RenderParams from CLI flags. This is the single source of truth
    /// for how reference characters are rendered — used by training, indexing,
    /// and inference.
    pub fn render_params(&self) -> crate::char_render::RenderParams {
        use crate::char_index::AaVariant;
        let aa = AaVariant::parse(&self.render_aa).unwrap_or(AaVariant::Native);
        crate::char_render::RenderParams {
            height: crate::char_index::NORM_H,
            render_scale: self.render_scale,
            aa,
            binarize_threshold: self.render_binarize.filter(|&v| v > 0),
        }
    }

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

    /// The ground-truth vector PDF path — from --test.
    pub fn gt_vector_pdf(&self) -> Option<&PathBuf> {
        self.test.as_ref()
    }

    /// Whether full audit I/O (crops, per-char diagnostics, HTML) is enabled.
    /// True when --audit is set.
    pub fn full_audit(&self) -> bool {
        self.audit.is_some()
    }

    /// Validate: if not --index, require input and output (unless --test).
    pub fn validate(&self) -> Result<(), String> {
        if self.index || self.render_ref_chars.is_some() || !self.weight_explicit.is_empty() || self.train_lda {
            return Ok(());
        }
        if self.input.is_none() {
            return Err("Input file required (or use --index)".to_string());
        }
        // --test mode doesn't require output
        if self.test.is_some() {
            return Ok(());
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

/// Parse a page specification string into a set of 1-indexed page numbers.
/// Accepts comma-separated values and ranges: "1,3,5" or "2-4,7" or "3".
pub fn parse_pages(spec: &str) -> Result<std::collections::HashSet<usize>, String> {
    let mut set = std::collections::HashSet::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.trim().parse().map_err(|_| format!("bad page number: {a}"))?;
            let end: usize = b.trim().parse().map_err(|_| format!("bad page number: {b}"))?;
            if start == 0 || end == 0 {
                return Err("page numbers are 1-indexed".into());
            }
            if start > end {
                return Err(format!("invalid range: {start}-{end}"));
            }
            for p in start..=end {
                set.insert(p);
            }
        } else {
            let p: usize = part.parse().map_err(|_| format!("bad page number: {part}"))?;
            if p == 0 {
                return Err("page numbers are 1-indexed".into());
            }
            set.insert(p);
        }
    }
    if set.is_empty() {
        return Err("empty page specification".into());
    }
    Ok(set)
}
