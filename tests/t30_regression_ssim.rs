//! SSIM regression tests — run the `unscan` binary against known test fixtures
//! and assert that SSIM scores, font identification, and vectorization counts
//! stay above established thresholds.
//!
//! Test fixtures are auto-generated from EB Garamond 12 (must be installed).
//! The binary must be built before running: `cargo build --release`.
//!
//! These tests invoke the CLI binary via `std::process::Command`, so they
//! exercise the full pipeline end-to-end (OCR → coarse scoring → SSIM verify).

mod common;

use common::{setup, test_doc, run_unscan, parse_ssim, parse_font_match, parse_vectorized_count};

// ---------- SSIM threshold tests ----------

#[test]
fn ssim_hires_above_threshold() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-hires.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let ssim = parse_ssim(&output).expect("no SSIM found in hires output");
    eprintln!("hires SSIM = {:.4}", ssim);
    assert!(
        ssim >= 0.85,
        "hires SSIM {:.4} below threshold 0.85",
        ssim
    );
}

#[test]
fn ssim_100dpi_above_threshold() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-100dpi-big.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--dpi", "100",
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let ssim = parse_ssim(&output).expect("no SSIM found in 100dpi output");
    eprintln!("100dpi SSIM = {:.4}", ssim);
    assert!(
        ssim >= 0.85,
        "100dpi SSIM {:.4} below threshold 0.85",
        ssim
    );
}

#[test]
fn ssim_garamond_above_threshold() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-garamond.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let ssim = parse_ssim(&output).expect("no SSIM found in garamond output");
    eprintln!("garamond SSIM = {:.4}", ssim);
    assert!(
        ssim >= 0.85,
        "garamond SSIM {:.4} below threshold 0.85",
        ssim
    );
}

#[test]
fn ssim_gold_png_above_threshold() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-gold.png");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let ssim = parse_ssim(&output).expect("no SSIM found in gold-png output");
    eprintln!("gold-png SSIM = {:.4}", ssim);
    assert!(
        ssim >= 0.85,
        "gold-png SSIM {:.4} below threshold 0.85",
        ssim
    );
}

// ---------- Font identification tests ----------

#[test]
fn font_match_hires_is_eb_garamond() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-hires.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let font = parse_font_match(&output).expect("no font match in hires output");
    eprintln!("hires font = {}", font);
    assert!(
        font.to_lowercase().contains("ebgaramond"),
        "expected EBGaramond, got '{}'",
        font
    );
}

#[test]
fn font_match_100dpi_is_eb_garamond() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-100dpi-big.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--dpi", "100",
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let font = parse_font_match(&output).expect("no font match in 100dpi output");
    eprintln!("100dpi font = {}", font);
    assert!(
        font.to_lowercase().contains("ebgaramond"),
        "expected EBGaramond, got '{}'",
        font
    );
}

#[test]
fn font_match_garamond_is_eb_garamond() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-garamond.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let font = parse_font_match(&output).expect("no font match in garamond output");
    eprintln!("garamond font = {}", font);
    assert!(
        font.to_lowercase().contains("ebgaramond"),
        "expected EBGaramond, got '{}'",
        font
    );
}

#[test]
fn font_match_gold_png_is_eb_garamond() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("punch-gold.png");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(
        &input,
        &[
            "--min-font-confidence", "0.0",
            "--min-ocr-confidence", "0",
            "--min-verify-ssim", "0.0",
        ],
    );
    let font = parse_font_match(&output).expect("no font match in gold-png output");
    eprintln!("gold-png font = {}", font);
    assert!(
        font.to_lowercase().contains("ebgaramond"),
        "expected EBGaramond, got '{}'",
        font
    );
}

// ---------- Specimen vectorization count ----------

#[test]
fn specimen_vectorizes_enough_lines() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }
    let input = test_doc("specimen-clean-raster.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found", input);
        return;
    }

    let output = run_unscan(&input, &["--overlay"]);
    let count =
        parse_vectorized_count(&output).expect("could not parse vectorized count from specimen");
    eprintln!("specimen vectorized = {}", count);
    assert!(
        count >= 90,
        "specimen vectorized {} lines, expected >= 90",
        count
    );
}
