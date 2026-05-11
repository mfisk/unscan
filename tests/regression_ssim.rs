//! SSIM regression tests — run the `unscan` binary against known test fixtures
//! and assert that SSIM scores, font identification, and vectorization counts
//! stay above established thresholds.
//!
//! Test fixtures are auto-generated from EB Garamond 12 (must be installed).
//! The binary must be built before running: `cargo build --release`.
//!
//! These tests invoke the CLI binary via `std::process::Command`, so they
//! exercise the full pipeline end-to-end (OCR → coarse scoring → SSIM verify).

use std::path::{Path, PathBuf};
use std::process::Command;

// ── Helpers ──────────────────────────────────────────────────────────

const EB_GARAMOND: &str = "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf";

/// Locate the built `unscan` binary (release or debug).
fn unscan_bin() -> PathBuf {
    // cargo sets OUT_DIR only for build scripts; for integration tests the
    // binary lives relative to the manifest directory.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let release = manifest.join("target/release/unscan");
    if release.exists() {
        return release;
    }
    let debug = manifest.join("target/debug/unscan");
    if debug.exists() {
        return debug;
    }
    panic!(
        "unscan binary not found. Run `cargo build --release` first.\n\
         Checked: {:?} and {:?}",
        release, debug
    );
}

/// Path into the test-docs directory.
fn test_doc(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-docs")
        .join(name)
}

/// Run a Python one-liner to generate a fixture. Panics on failure.
fn python3(code: &str) {
    let status = Command::new("python3")
        .arg("-c")
        .arg(code)
        .status()
        .expect("failed to launch python3");
    assert!(status.success(), "python3 fixture generation failed");
}

/// Run ImageMagick `convert` to embed a PNG as a PDF at a given DPI.
fn convert_png_to_pdf(png: &Path, pdf: &Path, dpi: u32) {
    let status = Command::new("convert")
        .arg(png)
        .args(["-density", &dpi.to_string()])
        .arg(pdf)
        .status()
        .expect("failed to launch convert (ImageMagick)");
    assert!(status.success(), "convert failed for {:?}", pdf);
}

// ── Fixture generation ───────────────────────────────────────────────

/// Ensure all test fixtures exist, generating any that are missing.
/// Returns false if prerequisites (EB Garamond, python3, convert) are absent.
fn ensure_fixtures() -> bool {
    if !Path::new(EB_GARAMOND).exists() {
        eprintln!("SKIP: EB Garamond 12 not installed at {}", EB_GARAMOND);
        return false;
    }

    // punch-hires.pdf — 600 DPI source, "punchcutter" at 120px
    let punch_hires = test_doc("punch-hires.pdf");
    if !punch_hires.exists() {
        eprintln!("  Generating punch-hires.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 120)
img = Image.new('L', (1800, 240), 255)
ImageDraw.Draw(img).text((0, 60), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-hires.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(Path::new("/tmp/_punch-hires.png"), &punch_hires, 600);
    }

    // punch-garamond.pdf — 200 DPI
    let punch_garamond = test_doc("punch-garamond.pdf");
    if !punch_garamond.exists() {
        eprintln!("  Generating punch-garamond.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 120)
img = Image.new('L', (1800, 240), 255)
ImageDraw.Draw(img).text((0, 60), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-garamond.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(
            Path::new("/tmp/_punch-garamond.png"),
            &punch_garamond,
            200,
        );
    }

    // punch-gold.png — native 60px, no pdftoppm
    let punch_gold = test_doc("punch-gold.png");
    if !punch_gold.exists() {
        eprintln!("  Generating punch-gold.png...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 60)
img = Image.new('L', (900, 120), 255)
ImageDraw.Draw(img).text((100, 30), 'punchcutter', font=font, fill=0)
img.save('{}')
"#,
            EB_GARAMOND,
            punch_gold.display()
        ));
    }

    // punch-100dpi-big.pdf — 600 DPI source, 40pt text, loaded at 100 DPI
    let punch_100dpi = test_doc("punch-100dpi-big.pdf");
    if !punch_100dpi.exists() {
        eprintln!("  Generating punch-100dpi-big.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 333)
img = Image.new('L', (5100, 1800), 255)
ImageDraw.Draw(img).text((600, 600), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-100dpi-big.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(
            Path::new("/tmp/_punch-100dpi-big.png"),
            &punch_100dpi,
            600,
        );
    }

    // specimen-clean-raster.pdf — generated by gen-specimen.py
    let specimen = test_doc("specimen-clean-raster.pdf");
    if !specimen.exists() {
        let gen_script = test_doc("gen-specimen.py");
        if gen_script.exists() {
            eprintln!("  Generating specimen-clean-raster.pdf...");
            let status = Command::new("python3")
                .arg(&gen_script)
                .current_dir(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-docs"),
                )
                .status()
                .expect("python3 gen-specimen.py failed to launch");
            assert!(status.success(), "gen-specimen.py failed");
        } else {
            eprintln!("  SKIP: gen-specimen.py not found");
        }
    }

    true
}

// ── Output parsers ───────────────────────────────────────────────────

/// Run unscan and return its combined stdout+stderr as a String.
fn run_unscan(input: &Path, extra_args: &[&str]) -> String {
    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(input)
        .args(["-o", "/dev/null"])
        .args(extra_args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

/// Extract the first `ssim=X.XXX` value from unscan output.
fn parse_ssim(output: &str) -> Option<f64> {
    for line in output.lines() {
        if let Some(pos) = line.find("ssim=") {
            let rest = &line[pos + 5..];
            let num_end = rest
                .find(|c: char| c != '.' && !c.is_ascii_digit())
                .unwrap_or(rest.len());
            return rest[..num_end].parse::<f64>().ok();
        }
    }
    None
}

/// Extract the matched font name from the first `✓` line (the `→ NAME (` part).
fn parse_font_match(output: &str) -> Option<String> {
    for line in output.lines() {
        if !line.contains('✓') {
            continue;
        }
        if let Some(arrow_pos) = line.find('→') {
            // "→ " is 4 bytes (UTF-8 arrow + space)
            let after_arrow = &line[arrow_pos + 4..];
            if let Some(paren) = after_arrow.find('(') {
                let name = after_arrow[..paren].trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

/// Extract the "Text lines vectorised: N" count.
fn parse_vectorized_count(output: &str) -> Option<u32> {
    for line in output.lines() {
        // Handle both British and American spelling
        if line.contains("vectorised:") || line.contains("vectorized:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(last) = parts.last() {
                return last.parse::<u32>().ok();
            }
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────

/// Common preamble: ensure fixtures exist or skip.
/// Returns true if tests can proceed.
fn setup() -> bool {
    ensure_fixtures()
}

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
        ssim >= 0.95,
        "hires SSIM {:.4} below threshold 0.95",
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
        ssim >= 0.95,
        "100dpi SSIM {:.4} below threshold 0.95",
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
        ssim >= 0.95,
        "gold-png SSIM {:.4} below threshold 0.95",
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
        count >= 25,
        "specimen vectorized {} lines, expected >= 25",
        count
    );
}
