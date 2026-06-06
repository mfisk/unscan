//! Specimen accuracy — 300 dpi, anti-aliased.
//!
//! Runs unscan with --audit against the AA-rasterized specimen, then
//! invokes tools/char-misses.py to compare each matched font against the
//! vector PDF's spatial ground truth via PyMuPDF.
//!
//! Prerequisites: run t55_specimen_gen first to generate all test fixtures.
//!
//! Run with:
//!   cargo test --test t60_specimen_accuracy -- --nocapture

mod common;

use common::{test_doc, ensure_index};

/// Minimum acceptable accuracy (hits / compared).
const MIN_ACCURACY: f64 = 0.89;

#[test]
fn specimen_font_accuracy_aa() {
    ensure_index();

    let vector = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");
    let raster = test_doc("font-timeline-specimen-rasterized.pdf");

    let r = common::measure_accuracy(&raster, &vector, &fontmap, "t60-300dpi-aa");

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

    let output = common::run_unscan(&input, &[]);
    let count = common::parse_vectorized_count(&output)
        .expect("could not parse vectorized count");

    eprintln!("specimen vectorized = {}", count);
    assert!(count >= 350, "specimen vectorized {} lines, expected >= 350", count);
}
