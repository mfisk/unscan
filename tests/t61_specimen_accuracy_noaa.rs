//! Specimen accuracy — 300 dpi, no anti-aliasing.
//!
//! Same methodology as t60 but with binarized (no-AA) rasterization.
//! This simulates high-contrast scans and photocopied documents where
//! gray-level anti-aliasing has been lost.
//!
//! Prerequisites: run t55_specimen_gen first to generate all test fixtures.
//!
//! Run with:
//!   cargo test --test t61_specimen_accuracy_noaa -- --nocapture

mod common;

use common::{test_doc, ensure_index, unscan_bin};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Lower threshold than t60 — binarization loses sub-pixel detail.
const MIN_ACCURACY: f64 = 0.82;

// ── Accuracy measurement ─────────────────────────────────────────────

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

fn measure_accuracy() -> AccuracyResult {
    let vector_src = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");
    let raster = test_doc("font-timeline-specimen-rasterized-noaa-300dpi.pdf");

    assert!(vector_src.exists(),
        "Vector specimen missing — run t55_specimen_gen first: {}",
        vector_src.display());
    assert!(fontmap.exists(),
        "Fontmap missing — run t55_specimen_gen first: {}",
        fontmap.display());
    assert!(raster.exists(),
        "No-AA raster missing — run t55_specimen_gen first: {}",
        raster.display());

    let audit_dir = std::env::temp_dir().join("unscan-t61-audit-300dpi-noaa");
    if audit_dir.exists() {
        let _ = std::fs::remove_dir_all(&audit_dir);
    }
    std::fs::create_dir_all(&audit_dir).expect("create audit dir");

    let output_pdf = audit_dir.join("out.pdf");

    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(&raster)
        .args(["-o", output_pdf.to_str().unwrap()])
        .args(["--audit", audit_dir.to_str().unwrap()])
        .args(["--include-fontmap", fontmap.to_str().unwrap()])
        .env("RUST_LOG", "info")
        .output()
        .unwrap_or_else(|e| panic!("failed to run unscan: {}", e));

    assert!(output.status.success(), "unscan failed: {}",
        String::from_utf8_lossy(&output.stderr));

    assert!(audit_dir.join("audit.json").exists(),
        "audit.json not written");

    let report_path = audit_dir.join("misses.html");
    let char_misses_py = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("char-misses.py");

    let py_output = Command::new("python3")
        .arg(&char_misses_py)
        .arg(&audit_dir)
        .arg(&vector_src)
        .args(["-o", report_path.to_str().unwrap()])
        .args(["--fontmap", fontmap.to_str().unwrap()])
        .output()
        .unwrap_or_else(|e| panic!("failed to run char-misses.py: {}", e));

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&py_output.stdout),
        String::from_utf8_lossy(&py_output.stderr),
    );
    assert!(py_output.status.success(), "char-misses.py failed:\n{}", combined);

    let (hits, misses, unmatched, skipped, total) = parse_summary(&combined);
    let compared = hits + misses;
    let accuracy = if compared > 0 { hits as f64 / compared as f64 } else { 0.0 };

    AccuracyResult { hits, misses, unmatched, skipped, total, compared, accuracy, report_path }
}

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

#[test]
fn specimen_font_accuracy_noaa() {
    ensure_index();

    let r = measure_accuracy();

    eprintln!(
        "noAA @ 300dpi: {}/{} = {:.1}% (threshold: {:.0}%)",
        r.hits, r.compared, r.accuracy * 100.0, MIN_ACCURACY * 100.0,
    );
    eprintln!(
        "  {} hits, {} misses ({} unmatched, {} wrong), {} skipped, {} total",
        r.hits, r.misses, r.unmatched, r.misses.saturating_sub(r.unmatched), r.skipped, r.total,
    );
    eprintln!("  Miss report: {}", r.report_path.display());

    assert!(r.compared > 0, "No lines compared");
    assert!(
        r.accuracy >= MIN_ACCURACY,
        "noAA accuracy {:.1}% below threshold {:.0}% ({}/{}) — see {}",
        r.accuracy * 100.0,
        MIN_ACCURACY * 100.0,
        r.hits,
        r.compared,
        r.report_path.display(),
    );
}
