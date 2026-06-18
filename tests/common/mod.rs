//! Shared helpers for unscan integration tests (t30, t40, t50).
//!
//! Contains CLI binary resolution, fixture generation, output parsers,
//! and run wrappers used across the end-to-end test suites.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

pub const EB_GARAMOND: &str = "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf";

// ── Shared index pre-build ───────────────────────────────────────────

static INDEX_ONCE: Once = Once::new();

/// Ensure the character index exists before any test spawns unscan.
/// Uses `Once` so parallel test threads only build it once; the rest block
/// until it's ready, then all hit the cached file.
pub fn ensure_index() {
    INDEX_ONCE.call_once(|| {
        let bin = unscan_bin();
        eprintln!("[test setup] Pre-building character index via {:?} --index", bin);
        let output = Command::new(&bin)
            .arg("--index")
            .env("RUST_LOG", "info")
            .output()
            .unwrap_or_else(|e| panic!("failed to run {:?} --index: {}", bin, e));
        if !output.status.success() {
            panic!(
                "Index pre-build failed (exit {}):\n{}{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        eprintln!("[test setup] Index ready.");
    });
}

// ── Binary & path helpers ────────────────────────────────────────────

/// Locate the built `unscan` binary (release or debug).
pub fn unscan_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_unscan"))
}

/// Path into the test-docs directory.
pub fn test_doc(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-docs")
        .join(name)
}

// ── Fixture generation helpers ───────────────────────────────────────

/// Run a Python one-liner to generate a fixture. Panics on failure.
pub fn python3(code: &str) {
    let status = Command::new("python3")
        .arg("-c")
        .arg(code)
        .status()
        .expect("failed to launch python3");
    assert!(status.success(), "python3 fixture generation failed");
}

/// Run ImageMagick `convert` to embed a PNG as a PDF at a given DPI.
pub fn convert_png_to_pdf(png: &Path, pdf: &Path, dpi: u32) {
    let status = Command::new("convert")
        .arg(png)
        .args(["-density", &dpi.to_string()])
        .arg(pdf)
        .status()
        .expect("failed to launch convert (ImageMagick)");
    assert!(status.success(), "convert failed for {:?}", pdf);
}

/// Ensure all test fixtures exist, generating any that are missing.
/// Returns false if prerequisites (EB Garamond, python3, convert) are absent.
pub fn ensure_fixtures() -> bool {
    if !Path::new(EB_GARAMOND).exists() {
        eprintln!("SKIP: EB Garamond 12 not installed at {}", EB_GARAMOND);
        return false;
    }

    // punch-hires.pdf — 600 DPI source, "punchcutter" at 120px
    let punch_hires = test_doc("punch-hires.pdf");
    if !punch_hires.exists() {
        eprintln!("  Generating punch-hires.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 120)
img = Image.new('L', (1800, 240), 255)
ImageDraw.Draw(img).text((0, 60), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-hires.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(Path::new("/tmp/_punch-hires.png"), &punch_hires, 600);
    }

    // punch-garamond.pdf — 200 DPI
    let punch_garamond = test_doc("punch-garamond.pdf");
    if !punch_garamond.exists() {
        eprintln!("  Generating punch-garamond.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 120)
img = Image.new('L', (1800, 240), 255)
ImageDraw.Draw(img).text((0, 60), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-garamond.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(
            Path::new("/tmp/_punch-garamond.png"),
            &punch_garamond,
            200,
        );
    }

    // punch-gold.png — native 60px, no pdftoppm
    let punch_gold = test_doc("punch-gold.png");
    if !punch_gold.exists() {
        eprintln!("  Generating punch-gold.png...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 60)
img = Image.new('L', (900, 120), 255)
ImageDraw.Draw(img).text((100, 30), 'punchcutter', font=font, fill=0)
img.save('{}')
"#,
            EB_GARAMOND,
            punch_gold.display()
        ));
    }

    // punch-100dpi-big.pdf — 600 DPI source, 40pt text, loaded at 100 DPI
    let punch_100dpi = test_doc("punch-100dpi-big.pdf");
    if !punch_100dpi.exists() {
        eprintln!("  Generating punch-100dpi-big.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 333)
img = Image.new('L', (5100, 1800), 255)
ImageDraw.Draw(img).text((600, 600), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-100dpi-big.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(
            Path::new("/tmp/_punch-100dpi-big.png"),
            &punch_100dpi,
            600,
        );
    }

    // specimen-clean-raster.pdf — generated by t55_specimen_gen
    let specimen = test_doc("specimen-clean-raster.pdf");
    if !specimen.exists() {
        eprintln!("  NOTE: specimen-clean-raster.pdf missing — run t55_specimen_gen first");
    }

    true
}

/// Common preamble: ensure fixtures exist or skip.
/// Returns true if tests can proceed.
pub fn setup() -> bool {
    ensure_fixtures()
}

// ── Run wrappers ─────────────────────────────────────────────────────

/// Run unscan with output to /dev/null and return combined stdout+stderr.
pub fn run_unscan(input: &Path, extra_args: &[&str]) -> String {
    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(input)
        .args(["-o", "/dev/null"])
        .args(extra_args)
        .env("RUST_LOG", "info")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

/// Run unscan with a specific output path and return combined stdout+stderr.
pub fn run_unscan_to(input: &Path, output_path: &Path, extra_args: &[&str]) -> String {
    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(input)
        .args(["-o", output_path.to_str().unwrap()])
        .args(extra_args)
        .env("RUST_LOG", "info")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

// ── Output parsers ───────────────────────────────────────────────────

/// Extract the first `ssim=X.XXX` value from unscan output.
pub fn parse_ssim(output: &str) -> Option<f64> {
    for line in output.lines() {
        if let Some(pos) = line.find("ssim=") {
            let rest = &line[pos + 5..];
            let num_end = rest
                .find(|c: char| c != '.' && !c.is_ascii_digit())
                .unwrap_or(rest.len());
            return rest[..num_end].parse::<f64>().ok();
        }
    }
    None
}

/// Extract the matched font name from the first `✓` line (the `→ NAME (` part).
pub fn parse_font_match(output: &str) -> Option<String> {
    for line in output.lines() {
        if !line.contains('✓') {
            continue;
        }
        if let Some(arrow_pos) = line.find('→') {
            // "→ " is 4 bytes (UTF-8 arrow + space)
            let after_arrow = &line[arrow_pos + 4..];
            if let Some(paren) = after_arrow.find('(') {
                let name = after_arrow[..paren].trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

/// Extract the "Text lines vectorised: N" count.
pub fn parse_vectorized_count(output: &str) -> Option<u32> {
    for line in output.lines() {
        // Handle both British and American spelling
        if line.contains("vectorised:") || line.contains("vectorized:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(last) = parts.last() {
                return last.parse::<u32>().ok();
            }
        }
    }
    None
}

// ── Rasterization helpers ────────────────────────────────────────────
//
// All rasterization goes through tools/rasterize.py — one script for
// rasterization, fontmap generation, and all variant controls
// (DPI, AA, backend).

fn rasterize_py() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools").join("rasterize.py")
}

fn raster_cache_ok(src: &Path, out_path: &Path) -> bool {
    if out_path.exists() {
        if let (Ok(sm), Ok(om)) = (src.metadata(), out_path.metadata()) {
            if let (Ok(st), Ok(ot)) = (sm.modified(), om.modified()) {
                if ot >= st {
                    eprintln!("  Using cached {:?}", out_path.file_name().unwrap());
                    return true;
                }
            }
        }
    }
    false
}

/// Rasterize a vector PDF to a grayscale raster PDF at the given DPI.
/// `aa` controls anti-aliasing: true = default AA, false = hard pixel edges.
/// `backend` selects the rasterizer: "mupdf" (default) or "poppler".
/// Result is cached at `out_path`; regenerated only if the source is newer.
pub fn rasterize_pdf_with(src: &Path, out_path: &Path, dpi: u32, aa: bool, backend: &str) -> bool {
    rasterize_pdf_opts(src, out_path, dpi, aa, false, backend)
}

pub fn rasterize_pdf_opts(src: &Path, out_path: &Path, dpi: u32, aa: bool, threshold: bool, backend: &str) -> bool {
    if raster_cache_ok(src, out_path) { return true; }

    eprintln!("  Rasterizing {:?} → {:?} (dpi={}, aa={}, threshold={}, backend={})",
        src.file_name().unwrap(), out_path.file_name().unwrap(), dpi, aa, threshold, backend);

    let mut cmd = Command::new("python3");
    cmd.arg(rasterize_py())
        .arg("prepare")
        .arg(src)
        .arg("--rasterize-only")
        .args(["-o", out_path.to_str().unwrap()])
        .args(["--dpi", &dpi.to_string()])
        .args(["--backend", backend]);
    if !aa { cmd.arg("--no-aa"); }
    if threshold { cmd.arg("--threshold"); }

    let ok = cmd.status().map(|s| s.success()).unwrap_or(false);
    if ok {
        eprintln!("  Generated {:?}", out_path.file_name().unwrap());
    } else {
        eprintln!("  ERROR: {} rasterization failed", backend);
    }
    ok
}

/// Rasterize with MuPDF backend (default for tests).
pub fn rasterize_pdf(src: &Path, out_path: &Path, dpi: u32, aa: bool) -> bool {
    rasterize_pdf_with(src, out_path, dpi, aa, "mupdf")
}

/// Rasterize with MuPDF backend + binary threshold (1-bit).
pub fn rasterize_pdf_threshold(src: &Path, out_path: &Path, dpi: u32, aa: bool) -> bool {
    rasterize_pdf_opts(src, out_path, dpi, aa, true, "mupdf")
}

/// Rasterize with Poppler/Cairo backend (cross-renderer accuracy tests).
pub fn rasterize_pdf_poppler(src: &Path, out_path: &Path, dpi: u32, aa: bool) -> bool {
    rasterize_pdf_with(src, out_path, dpi, aa, "poppler")
}

// ── Accuracy measurement harness ─────────────────────────────────────

/// Result of a specimen accuracy measurement run.
pub struct AccuracyResult {
    pub hits: usize,
    pub misses: usize,
    pub unmatched: usize,
    pub skipped: usize,
    pub total: usize,
    pub compared: usize,
    pub accuracy: f64,
    pub report_path: PathBuf,
}

/// Run unscan with --audit and --audit-vector for spatial ground-truth
/// comparison against the vector PDF.  The built-in report.rs generates
/// report.html and prints the summary line to stderr.
///
/// `raster` is the rasterized input PDF.
/// `vector` is the original vector PDF (ground truth).
/// `label` is used to name the audit directory (e.g. "300dpi-aa").
pub fn measure_accuracy(raster: &Path, vector: &Path, label: &str) -> AccuracyResult {
    assert!(vector.exists(),
        "Vector specimen missing — run t55_specimen_gen first: {}",
        vector.display());
    assert!(raster.exists(),
        "Raster input missing — run t55_specimen_gen first: {}",
        raster.display());

    let audit_dir = std::env::temp_dir().join(format!("unscan-audit-{}", label));
    if audit_dir.exists() {
        let _ = std::fs::remove_dir_all(&audit_dir);
    }
    std::fs::create_dir_all(&audit_dir).expect("create audit dir");

    let output_pdf = audit_dir.join("out.pdf");

    // Run unscan with --audit and --test (ground-truth vector PDF)
    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(raster)
        .args(["--audit", audit_dir.to_str().unwrap()])
        .args(["--test", vector.to_str().unwrap()])
        .env("RUST_LOG", "info")
        .output()
        .unwrap_or_else(|e| panic!("failed to run unscan: {}", e));

    assert!(output.status.success(), "unscan failed (exit {:?}):\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr));

    assert!(audit_dir.join("audit.json").exists(),
        "audit.json not written");

    let report_path = audit_dir.join("report.html");
    assert!(report_path.exists(),
        "report.html not written — --audit-vector may not have triggered report generation");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let (hits, misses, unmatched, skipped, total) = parse_report_summary(&stderr);
    let compared = hits + misses;
    let accuracy = if compared > 0 { hits as f64 / compared as f64 } else { 0.0 };

    AccuracyResult { hits, misses, unmatched, skipped, total, compared, accuracy, report_path }
}

/// Parse: "Report: H/C (P%) — M misses ..."
/// from unscan's stderr (generated by report.rs).
pub fn parse_report_summary(output: &str) -> (usize, usize, usize, usize, usize) {
    let mut hits = 0;
    let mut misses = 0;
    let unmatched = 0; // not separately reported by report.rs
    let skipped = 0;   // not separately reported by report.rs

    for line in output.lines() {
        if !line.contains("Report:") || !line.contains("misses") {
            continue;
        }
        // Parse "Report: H/C"
        if let Some(after_report) = line.split("Report:").nth(1) {
            let trimmed = after_report.trim();
            if let Some(slash_pos) = trimmed.find('/') {
                hits = trimmed[..slash_pos].trim().parse().unwrap_or(0);
                // compared is everything between '/' and ' '
                let after_slash = &trimmed[slash_pos + 1..];
                if let Some(space_pos) = after_slash.find(' ') {
                    let compared: usize = after_slash[..space_pos].trim().parse().unwrap_or(0);
                    misses = compared.saturating_sub(hits);
                }
            }
        }
        break;
    }

    let total = hits + misses + skipped;
    (hits, misses, unmatched, skipped, total)
}
