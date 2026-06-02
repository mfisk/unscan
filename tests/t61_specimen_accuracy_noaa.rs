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

use common::{test_doc, ensure_index};

/// Lower threshold than t60 — binarization loses sub-pixel detail.
const MIN_ACCURACY: f64 = 0.82;

#[test]
fn specimen_font_accuracy_noaa() {
    ensure_index();

    let vector = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");
    let raster = test_doc("font-timeline-specimen-rasterized-noaa-300dpi.pdf");

    let r = common::measure_accuracy(&raster, &vector, &fontmap, "t61-300dpi-noaa");

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
