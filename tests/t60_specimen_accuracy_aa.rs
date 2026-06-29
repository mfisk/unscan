//! Specimen accuracy — 300 dpi, anti-aliased.
//!
//! Runs unprint with --audit and --test against the AA-rasterized
//! specimen.  The --test flag compares each matched font against the
//! vector PDF's spatial ground truth and outputs JSON results to stdout.
//!
//! Prerequisites: run t55_specimen_gen first to generate all test fixtures.
//!
//! Run with:
//!   cargo test --test t60_specimen_accuracy -- --nocapture

mod common;

use common::{test_doc, ensure_index};

/// Minimum acceptable accuracy (primary_hits / compared).
const MIN_ACCURACY: f64 = 0.72;

#[test]
fn specimen_font_accuracy_aa() {
    ensure_index();

    let vector = test_doc("font-timeline-specimen.pdf");
    let raster = test_doc("font-timeline-specimen-rasterized.pdf");

    let r = common::measure_accuracy(&raster, &vector, "t60-300dpi-aa");

    eprintln!(
        "AA @ 300dpi: {}/{} = {:.1}% (threshold: {:.0}%)",
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
        "AA accuracy {:.1}% below threshold {:.0}% ({}/{}) — see {}",
        r.accuracy * 100.0,
        MIN_ACCURACY * 100.0,
        r.hits,
        r.compared,
        r.report_path.display(),
    );
}

#[test]
fn specimen_vectorizes_enough_lines() {
    ensure_index();

    let input = test_doc("font-timeline-specimen-rasterized.pdf");
    assert!(input.exists(),
        "AA raster missing — run t55_specimen_gen first: {}",
        input.display());

    // Run with --audit to get audit.json, then count vectorized lines
    let audit_dir = std::env::temp_dir().join("unscan-audit-t60-vec-count");
    if audit_dir.exists() {
        let _ = std::fs::remove_dir_all(&audit_dir);
    }
    std::fs::create_dir_all(&audit_dir).expect("create audit dir");

    let bin = common::unscan_bin();
    let result = std::process::Command::new(&bin)
        .arg(&input)
        .args(["--audit", audit_dir.to_str().unwrap()])
        .env("RUST_LOG", "info")
        .output()
        .expect("failed to run unprint");

    assert!(result.status.success(), "unprint failed: {}",
        String::from_utf8_lossy(&result.stderr));

    let audit_path = audit_dir.join("audit.json");
    assert!(audit_path.exists(), "audit.json not written");

    let audit_data = std::fs::read_to_string(&audit_path).expect("read audit.json");
    let json: serde_json::Value = serde_json::from_str(&audit_data)
        .expect("parse audit.json");

    // Count vectorized entries from text_entries
    let count = json["text_entries"].as_array()
        .map(|entries| entries.iter()
            .filter(|e| e["decision"].as_str() == Some("vectorized"))
            .count())
        .unwrap_or(0);

    eprintln!("specimen vectorized = {}", count);
    assert!(count >= 350, "specimen vectorized {} lines, expected >= 350", count);
}
