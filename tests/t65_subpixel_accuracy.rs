//! Subpixel accuracy — Trebuchet MS at fractional-pixel offsets.
//!
//! Runs unprint with --audit and --test against the 300-DPI rasterized
//! subpixel fixture (`test-xpos-kern.pdf`: preamble + H/V offset lines
//! at 0, 0.25, 0.5, 0.75, 1, 2.33, 5.67 px). The --test flag compares
//! each matched font against the vector PDF's spatial ground truth.
//!
//! Baseline (2026-11-05, regenerated specimens, tiebreaker unification
//! + has_gpos fix): 14/20 primary hits (70.0%), ZNCC avg 0.9214.
//! The 6 similarity failures all pick the correct font
//! (TrebuchetMS-400|kern|nolig) but score ZNCC 0.857-0.894, below
//! the 0.90 hit threshold — subpixel-phase rasterization, not
//! misidentification. There are 0 major misses.
//! (GT key was TrebuchetMS-400|nokern|nolig before the Nov 6 has_kern
//! preference fix; the fixture renders kerned, so the kerned key is correct.)
//!
//! Prerequisites: run ./regen-specimens.sh first to generate the fixture.
//!
//! Run with:
//!   cargo test --test t65_subpixel_accuracy -- --nocapture

mod common;

use common::{test_doc, ensure_index};

/// Minimum acceptable primary-hit rate (exact hits + minor misses).
/// Baseline 2026-11-05: 14/20 = 0.70.
const MIN_PRIMARY_ACCURACY: f64 = 0.70;

/// Minimum acceptable mean ZNCC similarity.
/// Baseline 2026-11-05: 0.9214.
const MIN_ZNCC_AVG: f64 = 0.92;

#[test]
fn subpixel_accuracy() {
    ensure_index();

    let vector = test_doc("test-xpos-kern.pdf");
    let raster = test_doc("test-xpos-kern-rasterized.pdf");

    let r = common::measure_accuracy(&raster, &vector, "t65-subpixel");

    let primary_accuracy = if r.compared > 0 {
        r.primary_hits as f64 / r.compared as f64
    } else {
        0.0
    };

    eprintln!(
        "Subpixel: {}/{} primary hits = {:.1}% (threshold: {:.0}%), ZNCC avg {:.4} (threshold: {:.2})",
        r.primary_hits, r.compared, primary_accuracy * 100.0,
        MIN_PRIMARY_ACCURACY * 100.0, r.zncc_avg, MIN_ZNCC_AVG,
    );
    eprintln!(
        "  {} hits, {} minor, {} major, {} sim_fail",
        r.hits, r.minor_misses, r.major_misses, r.similarity_failures,
    );
    eprintln!("  Miss report: {}", r.report_path.display());

    assert!(r.compared > 0, "No lines compared");
    assert_eq!(r.major_misses, 0,
        "Subpixel test has {} major misses — see {}",
        r.major_misses, r.report_path.display());
    assert!(
        primary_accuracy >= MIN_PRIMARY_ACCURACY,
        "Subpixel primary accuracy {:.1}% below threshold {:.0}% ({}/{}) — see {}",
        primary_accuracy * 100.0,
        MIN_PRIMARY_ACCURACY * 100.0,
        r.primary_hits,
        r.compared,
        r.report_path.display(),
    );
    assert!(
        r.zncc_avg >= MIN_ZNCC_AVG,
        "Subpixel ZNCC avg {:.4} below threshold {:.2} — see {}",
        r.zncc_avg, MIN_ZNCC_AVG, r.report_path.display(),
    );
}
