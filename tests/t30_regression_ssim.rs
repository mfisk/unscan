//! SSIM regression tests — run the `unprint` binary against known test fixtures
//! and assert that SSIM scores and font identification stay above established
//! thresholds.
//!
//! Test fixtures are auto-generated from EB Garamond 12 (must be installed).
//! The binary must be built before running: `cargo build --release`.
//!
//! These tests invoke the CLI binary via `std::process::Command`, so they
//! exercise the full pipeline end-to-end (OCR → font match → SSIM verify).

mod common;

use common::{setup, test_doc, unscan_bin};
use std::process::Command;

/// Run unprint with --audit on a single-word fixture and parse audit.json
/// Returns (ssim_score, font_matched) from the first text entry.
fn run_and_parse(input: &std::path::Path, extra_args: &[&str]) -> (Option<f64>, Option<String>) {
    let audit_dir = std::env::temp_dir().join(format!("unprint-t30-{}-{}",
        std::process::id(),
        input.file_stem().unwrap().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&audit_dir);
    std::fs::create_dir_all(&audit_dir).unwrap();

    let bin = unscan_bin();
    let status = Command::new(&bin)
        .arg(input)
        .args(["-o", "/dev/null"])
        .args(["--audit", audit_dir.to_str().unwrap()])
        .args(extra_args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    assert!(status.status.success(), "unprint failed: {:?}\nstderr: {}",
        status.status.code(),
        String::from_utf8_lossy(&status.stderr));

    let json_path = audit_dir.join("audit.json");
    if !json_path.exists() {
        return (None, None);
    }

    let data = std::fs::read_to_string(&json_path).unwrap();
    // Minimal JSON parsing — extract ssim_score and font_matched from first text_entry
    let ssim = extract_json_f64(&data, "ssim_score");
    let font = extract_json_string(&data, "font_matched");

    let _ = std::fs::remove_dir_all(&audit_dir);
    (ssim, font)
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{}\":", key);
    if let Some(pos) = json.find(&needle) {
        let rest = &json[pos + needle.len()..];
        let rest = rest.trim_start();
        let end = rest.find(|c: char| c != '.' && c != '-' && !c.is_ascii_digit())
            .unwrap_or(rest.len());
        rest[..end].parse::<f64>().ok()
    } else {
        None
    }
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":", key);
    if let Some(pos) = json.find(&needle) {
        let rest = &json[pos + needle.len()..];
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            let inner = &rest[1..];
            if let Some(end) = inner.find('"') {
                return Some(inner[..end].to_string());
            }
        }
    }
    None
}

// ---------- SSIM threshold tests ----------

#[test]
fn ssim_hires_above_threshold() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-hires.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (ssim, _) = run_and_parse(&input, &["--min-ocr-confidence", "0"]);
    let ssim = ssim.expect("no SSIM in audit.json for hires");
    eprintln!("hires SSIM = {:.4}", ssim);
    assert!(ssim >= 0.83, "hires SSIM {:.4} below threshold 0.83", ssim);
}

#[test]
fn ssim_100dpi_above_threshold() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-100dpi-big.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (ssim, _) = run_and_parse(&input, &["--dpi", "100", "--min-ocr-confidence", "0"]);
    let ssim = ssim.expect("no SSIM in audit.json for 100dpi");
    eprintln!("100dpi SSIM = {:.4}", ssim);
    assert!(ssim >= 0.74, "100dpi SSIM {:.4} below threshold 0.74", ssim);
}

#[test]
fn ssim_garamond_above_threshold() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-garamond.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (ssim, _) = run_and_parse(&input, &["--min-ocr-confidence", "0"]);
    let ssim = ssim.expect("no SSIM in audit.json for garamond");
    eprintln!("garamond SSIM = {:.4}", ssim);
    assert!(ssim >= 0.83, "garamond SSIM {:.4} below threshold 0.83", ssim);
}

#[test]
fn ssim_gold_png_above_threshold() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-gold.png");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (ssim, _) = run_and_parse(&input, &["--min-ocr-confidence", "0"]);
    let ssim = ssim.expect("no SSIM in audit.json for gold-png");
    eprintln!("gold-png SSIM = {:.4}", ssim);
    assert!(ssim >= 0.81, "gold-png SSIM {:.4} below threshold 0.81", ssim);
}

// ---------- Font identification tests ----------

#[test]
fn font_match_hires_is_eb_garamond() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-hires.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (_, font) = run_and_parse(&input, &["--min-ocr-confidence", "0"]);
    let font = font.expect("no font match in audit.json for hires");
    eprintln!("hires font = {}", font);
    assert!(
        font.to_lowercase().replace(' ', "").contains("ebgaramond"),
        "expected EBGaramond, got '{}'", font
    );
}

#[test]
fn font_match_100dpi_is_eb_garamond() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-100dpi-big.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (_, font) = run_and_parse(&input, &["--dpi", "100", "--min-ocr-confidence", "0"]);
    let font = font.expect("no font match in audit.json for 100dpi");
    eprintln!("100dpi font = {}", font);
    assert!(
        font.to_lowercase().replace(' ', "").contains("ebgaramond"),
        "expected EBGaramond, got '{}'", font
    );
}

#[test]
fn font_match_garamond_is_eb_garamond() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-garamond.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (_, font) = run_and_parse(&input, &["--min-ocr-confidence", "0"]);
    let font = font.expect("no font match in audit.json for garamond");
    eprintln!("garamond font = {}", font);
    assert!(
        font.to_lowercase().replace(' ', "").contains("ebgaramond"),
        "expected EBGaramond, got '{}'", font
    );
}

#[test]
fn font_match_gold_png_is_eb_garamond() {
    if !setup() { eprintln!("SKIP: fixtures unavailable"); return; }
    let input = test_doc("punch-gold.png");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }

    let (_, font) = run_and_parse(&input, &["--min-ocr-confidence", "0"]);
    let font = font.expect("no font match in audit.json for gold-png");
    eprintln!("gold-png font = {}", font);
    assert!(
        font.to_lowercase().replace(' ', "").contains("ebgaramond"),
        "expected EBGaramond, got '{}'", font
    );
}
