//! OTF-only font test.
//!
//! Inter Bold is only available as .otf (CFF outlines) on this system —
//! no .ttf exists.  This test verifies unscan indexes and identifies
//! OTF-only fonts correctly.
//!
//! Run: cargo test --release --test t41_otf_only_inter_bold -- --nocapture

mod common;

use common::{test_doc, run_unscan, parse_font_match, parse_ssim, parse_vectorized_count};

#[test]
fn inter_bold_otf_identifies_correctly() {
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found — run test-docs/gen-inter-bold.py first", input);
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
        .expect("no font match found in inter-bold output");
    eprintln!("Matched font: {}", font);

    // Must identify Inter Bold specifically — not InterDisplay, not Inter Regular
    let lower = font.to_lowercase();
    assert!(
        lower.contains("inter") && lower.contains("bold"),
        "Expected Inter Bold, got '{}'",
        font
    );
    // Must NOT be InterDisplay
    assert!(
        !lower.contains("display"),
        "Matched InterDisplay instead of Inter Bold: '{}'",
        font
    );
}

#[test]
fn inter_bold_otf_vectorizes_one_line() {
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() {
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
        .expect("could not parse vectorized count");
    assert_eq!(count, 1, "should vectorize exactly 1 line, got {}", count);
}

#[test]
fn inter_bold_otf_ssim_above_threshold() {
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() {
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
        .expect("no SSIM found in inter-bold output");
    eprintln!("Inter Bold SSIM = {:.4}", ssim);

    assert!(
        ssim >= 0.30,
        "inter-bold SSIM {:.4} below threshold 0.30",
        ssim
    );
}

#[test]
fn inter_bold_otf_source_is_otf() {
    // Verify the matched font path is actually an .otf file
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() {
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
        ],
    );

    assert!(
        output.contains(".otf"),
        "Expected .otf font path in output, but none found:\n{}",
        output
    );
}
