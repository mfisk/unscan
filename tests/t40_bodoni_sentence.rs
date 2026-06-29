//! Single-font, single-line end-to-end test.
//!
//! Runs unprint against bodoni-sentence-raster.pdf (one sentence in Libre
//! Bodoni 400 @ 24pt, rasterized at 300 DPI) and verifies the correct
//! font is identified and SSIM is reasonable.

mod common;

use common::{test_doc, unscan_bin};
use std::process::Command;

/// Run unprint with --audit and parse audit.json for font/SSIM/vectorized info.
fn run_and_parse(input: &std::path::Path) -> (Option<f64>, Option<String>, usize) {
    let audit_dir = std::env::temp_dir().join(format!("unprint-t40-{}", std::process::id()));
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
    let ssim = find_json_f64(&data, "ssim_score");
    let font = find_json_string(&data, "font_matched");
    let vectorized = count_json_field(&data, "lines_vectorized");

    let _ = std::fs::remove_dir_all(&audit_dir);
    (ssim, font, vectorized)
}

fn find_json_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{}\":", key);
    json.find(&needle).and_then(|pos| {
        let rest = json[pos + needle.len()..].trim_start();
        let end = rest.find(|c: char| c != '.' && c != '-' && !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse::<f64>().ok()
    })
}

fn find_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":", key);
    json.find(&needle).and_then(|pos| {
        let rest = json[pos + needle.len()..].trim_start();
        if rest.starts_with('"') {
            let inner = &rest[1..];
            inner.find('"').map(|end| inner[..end].to_string())
        } else { None }
    })
}

fn count_json_field(json: &str, key: &str) -> usize {
    let needle = format!("\"{}\":", key);
    let mut total = 0usize;
    let mut from = 0;
    while let Some(pos) = json[from..].find(&needle) {
        let abs = from + pos + needle.len();
        let rest = json[abs..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        if let Ok(n) = rest[..end].parse::<usize>() { total += n; }
        from = abs + end;
    }
    total
}

#[test]
fn bodoni_sentence_identifies_libre_bodoni() {
    let input = test_doc("bodoni-sentence-raster.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }
    let (_, font, _) = run_and_parse(&input);
    let font = font.expect("no font match in bodoni-sentence audit");
    eprintln!("Matched font: {}", font);
    assert!(font.to_lowercase().contains("bodoni"), "Expected Libre Bodoni, got '{}'", font);
}

#[test]
fn bodoni_sentence_vectorizes_one_line() {
    let input = test_doc("bodoni-sentence-raster.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }
    let (_, _, count) = run_and_parse(&input);
    eprintln!("Vectorized lines: {}", count);
    assert!(count >= 1, "bodoni-sentence should vectorize at least 1 line, got {}", count);
}

#[test]
fn bodoni_sentence_ssim_above_threshold() {
    let input = test_doc("bodoni-sentence-raster.pdf");
    if !input.exists() { eprintln!("SKIP: {:?} not found", input); return; }
    let (ssim, _, _) = run_and_parse(&input);
    let ssim = ssim.expect("no SSIM in bodoni-sentence audit");
    eprintln!("Bodoni sentence SSIM = {:.4}", ssim);
    assert!(ssim >= 0.30, "bodoni-sentence SSIM {:.4} below threshold 0.30", ssim);
}
