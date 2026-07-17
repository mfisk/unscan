//! Specimen accuracy — 300 dpi, 1-bit binary threshold.
//!
//! Simulates photocopied or faxed documents where anti-aliasing has been
//! completely lost — all pixels are either black or white.
//!
//! Prerequisites: run t55_specimen_gen first to generate all test fixtures.
//!
//! Run with:
//!   cargo test --release --test t63_specimen_accuracy_1bit -- --nocapture

mod common;

use common::{test_doc, ensure_index};

/// 1-bit is the hardest condition — expect lower accuracy than 8-bit no-AA.
/// Old metric (primary_hits) had 86% threshold.
/// Strict metric (hit only) scores 46.5–54.4% (high variance from 1-bit rasterization).
const MIN_ACCURACY: f64 = 0.44;

#[test]
fn specimen_font_accuracy_1bit() {
    ensure_index();

    let vector = test_doc("font-timeline-specimen.pdf");
    let raster = test_doc("font-timeline-specimen-rasterized-1bit-300dpi.pdf");

    let r = common::measure_accuracy(&raster, &vector, "t63-300dpi-1bit");

    eprintln!(
        "1-bit @ 300dpi: {}/{} = {:.1}% (threshold: {:.0}%)",
        r.hits, r.compared, r.accuracy * 100.0, MIN_ACCURACY * 100.0,
    );
    eprintln!(
        "  {} hits, {} minor, {} major, {} sim_fail",
        r.hits, r.minor_misses, r.major_misses, r.similarity_failures,
    );
    eprintln!("  Miss report: {}", r.report_path.display());

    assert!(r.compared > 0, "No lines compared");
    assert!(
        r.accuracy >= MIN_ACCURACY,
        "1-bit accuracy {:.1}% below threshold {:.0}% ({}/{}) — see {}",
        r.accuracy * 100.0,
        MIN_ACCURACY * 100.0,
        r.hits,
        r.compared,
        r.report_path.display(),
    );
}
