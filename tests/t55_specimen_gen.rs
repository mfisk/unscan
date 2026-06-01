//! Specimen generation test.
//!
//! Ensures the font-timeline specimen PDF, fontmap JSON, and rasterized
//! variants exist.  Runs gen-specimen.py if the vector PDF or fontmap are
//! missing, then rasterizes AA and no-AA versions at 300 dpi for downstream
//! accuracy tests (t60, t61, t62).
//!
//! Run with:
//!   cargo test --test t55_specimen_gen -- --nocapture

mod common;

use common::{test_doc, ensure_index};
use std::path::Path;
use std::process::Command;

/// Generate the vector specimen PDF and fontmap if they don't already exist.
fn ensure_specimen() {
    let vector_pdf = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");

    if vector_pdf.exists() && fontmap.exists() {
        return;
    }

    eprintln!("[t55] Generating specimen via gen-specimen.py ...");
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

#[test]
fn specimen_gen() {
    ensure_specimen();

    let vector_pdf = test_doc("font-timeline-specimen.pdf");
    let fontmap = test_doc("font-timeline-specimen-fontmap.json");

    eprintln!("[t55] Vector PDF: {}", vector_pdf.display());
    eprintln!("[t55] Fontmap:    {}", fontmap.display());

    // Pre-rasterize AA and no-AA at 300 dpi so t60/t61 don't have to wait
    let raster_aa = test_doc("font-timeline-specimen-rasterized.pdf");
    if !raster_aa.exists() {
        let alt = test_doc("font-timeline-specimen-rasterized-300dpi.pdf");
        if !alt.exists() {
            eprintln!("[t55] Rasterizing 300dpi AA ...");
            assert!(
                common::rasterize_pdf(&vector_pdf, &raster_aa, 300, true),
                "AA rasterization failed",
            );
        }
    }

    let raster_noaa = test_doc("font-timeline-specimen-rasterized-noaa-300dpi.pdf");
    if !raster_noaa.exists() {
        eprintln!("[t55] Rasterizing 300dpi no-AA ...");
        assert!(
            common::rasterize_pdf(&vector_pdf, &raster_noaa, 300, false),
            "no-AA rasterization failed",
        );
    }

    // Also ensure the character index is built
    ensure_index();

    eprintln!("[t55] Specimen ready.");
}
