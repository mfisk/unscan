//! Single-font, single-line end-to-end test.
//!
//! Runs unscan against bodoni-sentence-raster.pdf (one sentence in Libre
//! Bodoni 400 @ 24pt, rasterized at 300 DPI) and verifies the correct
//! font is identified and SSIM is reasonable.
//!
//! This is the simplest image-matching integration test — one line, one font.
//!
//! Run: cargo test --release --test t40_bodoni_sentence -- --nocapture

mod common;

use common::{test_doc, run_unscan, parse_ssim, parse_font_match, parse_vectorized_count};

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn bodoni_sentence_identifies_libre_bodoni() {
    let input = test_doc("bodoni-sentence-raster.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
        ],
    );
    eprintln!("Output:\n{}", output);

    let font = parse_font_match(&output)
        .expect("no font match found in bodoni-sentence output");
    eprintln!("Matched font: {}", font);

    assert!(
        font.to_lowercase().contains("bodoni"),
        "Expected Libre Bodoni, got '{}'",
        font
    );
}

#[test]
fn bodoni_sentence_vectorizes_one_line() {
    let input = test_doc("bodoni-sentence-raster.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
        ],
    );

    let count = parse_vectorized_count(&output)
        .expect("could not parse vectorized count from bodoni-sentence output");
    eprintln!("Vectorized lines: {}", count);

    assert_eq!(
        count, 1,
        "bodoni-sentence should vectorize exactly 1 line, got {}",
        count
    );
}

#[test]
fn bodoni_sentence_ssim_above_threshold() {
    let input = test_doc("bodoni-sentence-raster.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
        ],
    );

    let ssim = parse_ssim(&output)
        .expect("no SSIM found in bodoni-sentence output");
    eprintln!("Bodoni sentence SSIM = {:.4}", ssim);

    // Threshold is conservative — the important thing is that the font
    // is identified correctly (tested above). SSIM just catches regressions.
    assert!(
        ssim >= 0.30,
        "bodoni-sentence SSIM {:.4} below threshold 0.30",
        ssim
    );
}
