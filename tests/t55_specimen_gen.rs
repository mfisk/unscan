//! Specimen generation test.
//!
//! Generates the font-timeline specimen PDF and rasterized
//! variants (AA and no-AA at 300 dpi).  Downstream accuracy tests (t60,
//! t61, t62) depend on these outputs and will fail if they're missing.
//!
//! Run with:
//!   cargo test --test t55_specimen_gen -- --nocapture

mod common;

use common::{test_doc, ensure_index};
use std::path::Path;
use std::process::Command;

#[test]
fn specimen_gen() {
    // Generate vector PDF + fontmap via gen-specimen.py
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

    let vector_pdf = test_doc("font-timeline-specimen.pdf");
    assert!(vector_pdf.exists(), "gen-specimen.py did not create vector PDF");

    eprintln!("[t55] Vector PDF: {}", vector_pdf.display());

    // Rasterize AA at 300 dpi
    let raster_aa = test_doc("font-timeline-specimen-rasterized.pdf");
    eprintln!("[t55] Rasterizing 300dpi AA ...");
    assert!(
        common::rasterize_pdf(&vector_pdf, &raster_aa, 300, true),
        "AA rasterization failed",
    );

    // Rasterize no-AA at 300 dpi (8-bit, renderer-native AA disabled)
    let raster_noaa = test_doc("font-timeline-specimen-rasterized-noaa-300dpi.pdf");
    eprintln!("[t55] Rasterizing 300dpi no-AA ...");
    assert!(
        common::rasterize_pdf(&vector_pdf, &raster_noaa, 300, false),
        "no-AA rasterization failed",
    );

    // Rasterize 1-bit threshold at 300 dpi (no-AA + binary threshold)
    let raster_1bit = test_doc("font-timeline-specimen-rasterized-1bit-300dpi.pdf");
    eprintln!("[t55] Rasterizing 300dpi 1-bit ...");
    assert!(
        common::rasterize_pdf_threshold(&vector_pdf, &raster_1bit, 300, false),
        "1-bit rasterization failed",
    );

    // Build character index
    ensure_index();

    eprintln!("[t55] Specimen ready.");
}
