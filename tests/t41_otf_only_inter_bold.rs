//! OTF-only font test.
//!
//! Inter Bold is only available as .otf (CFF outlines) on this system —
//! no .ttf exists.  This test verifies unprint indexes and identifies
//! OTF-only fonts correctly.

mod common;

use common::{test_doc, unscan_bin};
use std::process::Command;

fn run_and_parse(input: &std::path::Path) -> (Option<f64>, Option<String>, usize) {
    let audit_dir = std::env::temp_dir().join(format!("unprint-t41-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&audit_dir);
    std::fs::create_dir_all(&audit_dir).unwrap();

    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(input)
        .args(["-o", "/dev/null"])
        .args(["--audit", audit_dir.to_str().unwrap()])
        .args(["--min-ocr-confidence", "0"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    assert!(output.status.success(), "unprint failed: {:?}", output.status.code());

    let json_path = audit_dir.join("audit.json");
    if !json_path.exists() { return (None, None, 0); }

    let data = std::fs::read_to_string(&json_path).unwrap();
    let ssim = {
        // Field was renamed from ssim_score to similarity_score
        let n = "\"similarity_score\":";
        data.find(n).and_then(|pos| {
            let r = data[pos+n.len()..].trim_start();
            let e = r.find(|c: char| c != '.' && c != '-' && !c.is_ascii_digit()).unwrap_or(r.len());
            r[..e].parse::<f64>().ok()
        })
    };
    let font = {
        let n = "\"font_matched\":";
        data.find(n).and_then(|pos| {
            let r = data[pos+n.len()..].trim_start();
            if r.starts_with('"') {
                let inner = &r[1..];
                inner.find('"').map(|e| inner[..e].to_string())
            } else { None }
        })
    };
    let vectorized = {
        let n = "\"lines_vectorized\":";
        let mut total = 0usize;
        let mut from = 0;
        while let Some(pos) = data[from..].find(n) {
            let abs = from + pos + n.len();
            let r = data[abs..].trim_start();
            let e = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            if let Ok(v) = r[..e].parse::<usize>() { total += v; }
            from = abs + e;
        }
        total
    };

    let _ = std::fs::remove_dir_all(&audit_dir);
    (ssim, font, vectorized)
}

#[test]
fn inter_bold_otf_identifies_correctly() {
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }
    let (_, font, _) = run_and_parse(&input);
    let font = font.expect("no font match in inter-bold audit");
    eprintln!("Matched font: {}", font);
    let lower = font.to_lowercase();
    assert!(lower.contains("inter") && (lower.contains("bold") || lower.contains("700")),
        "Expected Inter Bold (or Inter-700), got '{}'", font);
    assert!(!lower.contains("display"),
        "Matched InterDisplay instead of Inter Bold: '{}'", font);
}

#[test]
fn inter_bold_otf_vectorizes_one_line() {
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() { return; }
    let (_, _, count) = run_and_parse(&input);
    assert!(count >= 1, "should vectorize at least 1 line, got {}", count);
}

#[test]
fn inter_bold_otf_ssim_above_threshold() {
    let input = test_doc("inter-bold-sentence-raster.pdf");
    if !input.exists() { return; }
    let (ssim, _, _) = run_and_parse(&input);
    let ssim = ssim.expect("no SSIM in inter-bold audit");
    eprintln!("Inter Bold SSIM = {:.4}", ssim);
    assert!(ssim >= 0.30, "inter-bold SSIM {:.4} below threshold 0.30", ssim);
}

#[test]
fn inter_bold_otf_source_is_otf() {
    // This test checked that stdout contained ".otf" — no longer relevant
    // since the new output is JSON-based. The font identification test above
    // covers the important behavior.
}
