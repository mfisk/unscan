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
    /// Input file (PDF or image: PNG, JPEG, TIFF, BMP)
    pub input: PathBuf,

    /// Output PDF path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Additional font search directories (repeatable)
    #[arg(long)]
    pub font_dir: Vec<PathBuf>,

    /// Minimum OCR confidence to consider vectorizing text (0–100).
    /// Below this, the original raster is kept unconditionally.
    #[arg(long, default_value = "80")]
    pub min_ocr_confidence: u32,

    /// Minimum font-match confidence (0.0–1.0).
    /// Below this, text is kept as raster — we never replace with a wrong font.
    #[arg(long, default_value = "0.40")]
    pub min_font_confidence: f32,

    /// Minimum SSIM score for the verification pass (0.0–1.0).
    /// After rendering vector text, compare with original. Below this → revert to raster.
    #[arg(long, default_value = "0.30")]
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

    /// Path for the audit-log JSON file (default: <output>.audit.json)
    #[arg(long)]
    pub audit_log: Option<PathBuf>,
}

impl Args {
    pub fn audit_log_path(&self) -> PathBuf {
        self.audit_log.clone().unwrap_or_else(|| {
            let mut p = self.output.clone();
            let stem = p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            p.set_file_name(format!("{stem}.audit.json"));
            p
        })
    }
}

pub fn parse() -> Args {
    Args::parse()
}
