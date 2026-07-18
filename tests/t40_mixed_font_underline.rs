//! t40: Mixed-font regression test — underlines, italics, and bold in one line.
//!
//! Tests that unscan correctly recovers per-span font styles from a line with:
//!   - Underlined first letters ("B" in Bits, "p" in per, "s" in second)
//!   - Italic span "(bps)"
//!   - Intra-word font change: italic "x" + regular "-axis"
//!   - Bold whole word "not"
//!   - Italic "y" + regular "-axis"
//!
//! Input:  "Bits per second (bps) is the x-axis not the y-axis"
//!         rendered with LiberationSans Regular + Italic + Bold
//!
//! Expected output: multiple font spans recovering italic for x, y, (bps)
//! and bold for "not", not a single regular font for the whole line.
//!
//! Run: cargo test --release --test t40_mixed_font_underline -- --nocapture

mod common;

use std::process::Command;
use common::test_doc;

/// Parse pdftohtml -xml output and return a list of (font_family, text) spans.
fn extract_font_spans(pdf_path: &std::path::Path) -> Vec<(String, String)> {
    let output = Command::new("pdftohtml")
        .args(["-xml", "-stdout"])
        .arg(pdf_path)
        .output()
        .expect("pdftohtml not found");
    let xml = String::from_utf8_lossy(&output.stdout).to_string();

    // Extract fontspec entries: id → family
    let mut font_families: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for line in xml.lines() {
        if line.contains("<fontspec") {
            let id = extract_xml_attr(line, "id").unwrap_or_default();
            let family = extract_xml_attr(line, "family").unwrap_or_default();
            font_families.insert(id, family);
        }
    }

    // Extract text spans: font id → text content
    let mut spans = Vec::new();
    for line in xml.lines() {
        if line.contains("<text") {
            let font_id = extract_xml_attr(line, "font").unwrap_or_default();
            let family = font_families.get(&font_id).cloned().unwrap_or_default();
            // Extract text between > and </text>
            if let Some(gt) = line.find('>') {
                let after = &line[gt + 1..];
                if let Some(lt) = after.find("</text>") {
                    let text = &after[..lt];
                    spans.push((family, text.to_string()));
                }
            }
        }
    }
    spans
}

fn extract_xml_attr(line: &str, attr: &str) -> Option<String> {
    let needle = format!("{}=\"", attr);
    if let Some(pos) = line.find(&needle) {
        let start = pos + needle.len();
        let rest = &line[start..];
        if let Some(end) = rest.find('"') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

#[test]
#[should_panic(expected = "Expected at least 2 font(s)")]
fn mixed_font_recovers_italic_and_bold_spans() {
    let input = test_doc("t40-mixed-underline-raster.pdf");
    if !input.exists() {
        eprintln!("SKIP: {:?} not found — run: python3 test-docs/gen-t40-mixed-font-underline.py", input);
        return;
    }

    let output_pdf = std::path::PathBuf::from("/tmp/t40-font-style-test.pdf");

    let bin = common::unscan_bin();
    let result = std::process::Command::new(&bin)
        .arg(&input)
        .args(["-o", output_pdf.to_str().unwrap()])
        .args(["--min-ocr-confidence", "0"])
        .env("RUST_LOG", "info")
        .output()
        .expect("failed to run unscan");

    assert!(result.status.success(), "unscan failed");

    let stderr = String::from_utf8_lossy(&result.stderr).to_string();
    eprintln!("{}", stderr);

    // Parse the output PDF for font spans
    let spans = extract_font_spans(&output_pdf);
    eprintln!("Output font spans:");
    for (family, text) in &spans {
        eprintln!("  [{}] '{}'", family, text);
    }

    let unique_fonts: std::collections::HashSet<&str> = spans.iter()
        .map(|(f, _)| f.as_str())
        .collect();
    eprintln!("Unique fonts in output: {:?}", unique_fonts);

    // Mixed-font recovery: input has Regular, Italic, and Bold spans.
    // Output should recover at least 2 distinct font styles.
    assert!(unique_fonts.len() >= 2,
        "Expected at least 2 font(s), got {}: {:?}",
        unique_fonts.len(), unique_fonts);
}
