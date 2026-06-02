//! Cross-renderer accuracy tests.
//!
//! The CI font index is built from ab_glyph renders (FreeType-based). These
//! tests rasterize the specimen PDF using Poppler/Cairo — a different rendering
//! engine with different hinting, stem placement, and anti-aliasing decisions.
//!
//! This is the same kind of variation seen between different printers/drivers
//! in real scanned documents: stems land on different pixel boundaries,
//! anti-aliasing kernels differ, and gray-level distributions shift.
//!
//! Accuracy is measured by shelling out to tools/char-misses.py, which
//! spatially matches each audit entry against the vector PDF's text spans
//! via PyMuPDF — the same methodology as t60.
//!
//! Run with:
//!   cargo test --release --test t62_cross_renderer_accuracy -- --nocapture
//!
//! Requires: pdftoppm (Poppler), img2pdf, PIL/numpy, PyMuPDF (Python).

mod common;

use common::{test_doc, ensure_index, unscan_bin};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Poppler renders with different hinting than the CI's FreeType backend.
/// Lower than t60's threshold to account for cross-renderer feature drift.
const MIN_ACCURACY_POPPLER_AA: f64 = 0.82;

struct AccuracyResult {
    hits: usize,
    misses: usize,
    unmatched: usize,
    skipped: usize,
    total: usize,
    compared: usize,
    accuracy: f64,
    report_path: PathBuf,
}

/// Ensure the vector specimen + fontmap exist (delegates to gen-specimen.py).
fn ensure_specimen() {
    let vector_pdf = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");

    if vector_pdf.exists() && fontmap.exists() {
        return;
    }

    eprintln!("[test setup] Generating specimen via gen-specimen.py ...");
    let gen_script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-docs")
        .join("gen-specimen.py");

    let output = Command::new("python3")
        .arg(&gen_script)
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("test-docs"))
        .output()
        .unwrap_or_else(|e| panic!("failed to run gen-specimen.py: {}", e));

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    eprintln!("{}", combined);

    assert!(output.status.success(), "gen-specimen.py failed");
    assert!(vector_pdf.exists(), "gen-specimen.py did not create vector PDF");
    assert!(fontmap.exists(), "gen-specimen.py did not create fontmap JSON");
}

/// Run unscan --audit on a rasterized PDF, then char-misses.py for spatial
/// ground-truth comparison against the vector PDF.
fn measure_accuracy(raster_pdf: &Path, label: &str) -> Option<AccuracyResult> {
    let vector_src = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");

    let audit_dir = std::env::temp_dir()
        .join(format!("unscan-t62-audit-{}", label));
    if audit_dir.exists() {
        let _ = std::fs::remove_dir_all(&audit_dir);
    }
    std::fs::create_dir_all(&audit_dir).expect("create audit dir");

    let output_pdf = audit_dir.join("out.pdf");

    // Run unscan with --audit
    let bin = unscan_bin();
    let mut cmd = Command::new(&bin);
    cmd.arg(raster_pdf)
        .args(["-o", output_pdf.to_str().unwrap()])
        .args(["--audit", audit_dir.to_str().unwrap()])
        .env("RUST_LOG", "info");
    if fontmap.exists() {
        cmd.args(["--include-fontmap", fontmap.to_str().unwrap()]);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run unscan: {}", e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("unscan failed (exit {:?}):\n{}", output.status.code(), stderr);
        return None;
    }

    let audit_json = audit_dir.join("audit.json");
    if !audit_json.exists() {
        eprintln!("SKIP: audit.json not written");
        return None;
    }

    // Run char-misses.py
    let report_path = audit_dir.join("misses.html");
    let tools_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools");
    let char_misses_py = tools_dir.join("char-misses.py");

    let mut py_cmd = Command::new("python3");
    py_cmd.arg(&char_misses_py)
        .arg(&audit_dir)
        .arg(&vector_src)
        .args(["-o", report_path.to_str().unwrap()]);
    if fontmap.exists() {
        py_cmd.args(["--fontmap", fontmap.to_str().unwrap()]);
    }

    let py_output = py_cmd.output()
        .unwrap_or_else(|e| panic!("failed to run char-misses.py: {}", e));

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&py_output.stdout),
        String::from_utf8_lossy(&py_output.stderr),
    );

    if !py_output.status.success() {
        eprintln!("char-misses.py failed:\n{}", combined);
        return None;
    }

    let (hits, misses, unmatched, skipped, total) = parse_summary(&combined);
    let compared = hits + misses;
    let accuracy = if compared > 0 { hits as f64 / compared as f64 } else { 0.0 };

    Some(AccuracyResult {
        hits,
        misses,
        unmatched,
        skipped,
        total,
        compared,
        accuracy,
        report_path,
    })
}

/// Parse char-misses.py summary: "Total: N  Hits: H  Misses: M (U unmatched)  Skipped: S"
fn parse_summary(output: &str) -> (usize, usize, usize, usize, usize) {
    let mut hits = 0;
    let mut misses = 0;
    let mut unmatched = 0;
    let mut skipped = 0;
    let mut total = 0;

    for line in output.lines() {
        if !line.contains("Total:") || !line.contains("Hits:") {
            continue;
        }
        for part in line.split_whitespace().collect::<Vec<_>>().windows(2) {
            match part[0] {
                "Total:" => total = part[1].parse().unwrap_or(0),
                "Hits:" => hits = part[1].parse().unwrap_or(0),
                "Misses:" => misses = part[1].parse().unwrap_or(0),
                "Skipped:" => skipped = part[1].parse().unwrap_or(0),
                _ => {}
            }
        }
        if let Some(pos) = line.find('(') {
            if let Some(end) = line[pos..].find(" unmatched)") {
                let n_str = &line[pos + 1..pos + end];
                unmatched = n_str.trim().parse().unwrap_or(0);
            }
        }
        break;
    }

    (hits, misses, unmatched, skipped, total)
}

// ── Tests ────────────────────────────────────────────────────────────

/// Accuracy against Poppler/Cairo-rendered specimen with anti-aliasing.
#[test]
fn specimen_font_accuracy_poppler() {
    ensure_index();
    ensure_specimen();

    let vector_src = test_doc("font-timeline-specimen.pdf");
    let poppler_pdf = test_doc("font-timeline-specimen-rasterized-poppler.pdf");
    if !common::rasterize_pdf_poppler(&vector_src, &poppler_pdf, 300, true) {
        eprintln!("SKIP: Poppler rasterization failed (pdftoppm missing?)");
        return;
    }

    let r = measure_accuracy(&poppler_pdf, "poppler-300dpi-aa")
        .expect("could not measure Poppler accuracy");

    eprintln!(
        "Poppler AA @ 300dpi: {}/{} = {:.1}% (threshold: {:.0}%)",
        r.hits, r.compared, r.accuracy * 100.0, MIN_ACCURACY_POPPLER_AA * 100.0,
    );
    eprintln!(
        "  {} hits, {} misses ({} unmatched, {} wrong), {} skipped, {} total",
        r.hits, r.misses, r.unmatched, r.misses.saturating_sub(r.unmatched), r.skipped, r.total,
    );
    eprintln!("  Miss report: {}", r.report_path.display());

    assert!(
        r.compared > 0,
        "No lines compared — audit or char-misses.py produced no results",
    );
    assert!(
        r.accuracy >= MIN_ACCURACY_POPPLER_AA,
        "Poppler accuracy {:.1}% below threshold {:.0}% ({}/{}) — see {}",
        r.accuracy * 100.0,
        MIN_ACCURACY_POPPLER_AA * 100.0,
        r.hits,
        r.compared,
        r.report_path.display(),
    );
}
