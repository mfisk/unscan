//! Cross-renderer accuracy tests.
//!
//! The CI font index is built from ab_glyph renders (FreeType-based). These
//! tests rasterize the specimen PDF using Poppler/Cairo — a different rendering
//! engine with different hinting, stem placement, and anti-aliasing decisions.
//!
//! This is the same kind of variation seen between different printers/drivers
//! in real scanned documents: stems land on different pixel boundaries,
//! anti-aliasing kernels differ, and gray-level distributions shift.
//!
//! Run with:
//!   cargo test --release --test t62_cross_renderer_accuracy
//!
//! Requires: pdftoppm (Poppler), img2pdf, PIL/numpy (Python).

mod common;

use common::{test_doc, run_unscan};
use std::collections::HashMap;

/// Poppler renders with different hinting than the CI's FreeType backend.
/// This threshold is intentionally lower than t60's 95% to account for
/// cross-renderer feature drift while still proving the index is broadly
/// rendering-invariant.
const MIN_ACCURACY_POPPLER_AA: f64 = 0.85;

/// Parse ground truth: section index → lowercase font family (spaces removed).
fn load_ground_truth() -> HashMap<usize, String> {
    let path = test_doc("font-timeline-specimen.json");
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read ground truth"))
            .expect("parse ground truth JSON");
    let sections = data["sections"].as_array().expect("sections array");
    sections
        .iter()
        .map(|s| {
            let idx = s["index"].as_u64().expect("section index") as usize;
            let family = s["font_family"]
                .as_str()
                .expect("font_family")
                .to_lowercase()
                .replace(' ', "");
            (idx, family)
        })
        .collect()
}

/// Extract matched font names from unscan output lines containing ✓ or ✗.
fn parse_all_font_matches(output: &str) -> Vec<String> {
    let mut matches = Vec::new();
    for line in output.lines() {
        if !line.contains('✓') && !line.contains('✗') {
            continue;
        }
        if let Some(arrow_pos) = line.find('→') {
            let after_arrow = &line[arrow_pos + '→'.len_utf8()..];
            let after_arrow = after_arrow.trim_start();
            if let Some(paren) = after_arrow.find('(') {
                let name = after_arrow[..paren].trim().to_lowercase().replace(' ', "");
                if !name.is_empty() {
                    matches.push(name);
                }
            }
        }
    }
    matches
}

/// Count total OCR lines from unscan output.
fn count_total_ocr_lines(output: &str) -> usize {
    let mut total = 0;
    for line in output.lines() {
        if let Some(arrow) = line.find('→') {
            let after = &line[arrow + '→'.len_utf8()..];
            let after = after.trim_start();
            if after.contains("lines") {
                if let Some(n) = after.split_whitespace().next().and_then(|s| s.parse::<usize>().ok()) {
                    total += n;
                }
            }
        }
    }
    total
}

/// Known font renames / aliases (same as t60).
fn font_aliases() -> HashMap<String, Vec<String>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();
    m.insert("sourcesans3".into(), vec!["sourcesanspro".into()]);
    m.insert("couriernew".into(), vec!["nimbusmonops".into(), "freemono".into()]);
    m.insert("arial".into(), vec!["liberationsans".into(), "nimbussans".into(), "freesans".into()]);
    m.insert("ptserif".into(), vec!["nimbusroman".into(), "liberationserif".into(), "freeserif".into()]);
    m.insert("lato".into(), vec!["carlito".into()]);
    m.insert("caladea".into(), vec!["p052".into()]);
    m
}

/// Check if a matched font name corresponds to any ground truth font family.
fn is_correct(matched: &str, ground_truth: &HashMap<usize, String>) -> bool {
    let aliases = font_aliases();
    ground_truth.values().any(|expected| {
        if matched.contains(expected.as_str()) || expected.contains(matched) {
            return true;
        }
        if let Some(alias_list) = aliases.get(expected.as_str()) {
            if alias_list
                .iter()
                .any(|a| matched.contains(a.as_str()) || a.contains(matched))
            {
                return true;
            }
        }
        false
    })
}

/// Accuracy against Poppler/Cairo-rendered specimen with anti-aliasing.
///
/// Tests that the CI feature index generalises across rendering engines —
/// the same variation profile as different printers/drivers produce on paper.
#[test]
fn specimen_font_accuracy_poppler() {
    let vector_src = test_doc("font-timeline-specimen.pdf");
    if !vector_src.exists() {
        eprintln!("SKIP: font-timeline-specimen.pdf not found");
        return;
    }
    let gt_path = test_doc("font-timeline-specimen.json");
    if !gt_path.exists() {
        eprintln!("SKIP: font-timeline-specimen.json not found");
        return;
    }

    // Generate / cache the Poppler-rendered rasterized PDF
    let poppler_pdf = test_doc("font-timeline-specimen-rasterized-poppler.pdf");
    if !common::rasterize_pdf_poppler(&vector_src, &poppler_pdf, 300, true) {
        eprintln!("SKIP: Poppler rasterization failed (pdftoppm missing?)");
        return;
    }

    let ground_truth = load_ground_truth();

    let output = run_unscan(&poppler_pdf, &[]);
    let matched_fonts = parse_all_font_matches(&output);
    let total_lines = count_total_ocr_lines(&output);

    let matched = matched_fonts.len();
    assert!(matched > 0, "No font matches found in Poppler specimen output");
    assert!(total_lines > 0, "No OCR lines found in Poppler specimen output");

    let correct = matched_fonts
        .iter()
        .filter(|m| is_correct(m, &ground_truth))
        .count();

    let accuracy = correct as f64 / total_lines as f64;
    eprintln!(
        "Specimen Poppler accuracy: {}/{} = {:.1}% (threshold: {:.0}%)",
        correct,
        total_lines,
        accuracy * 100.0,
        MIN_ACCURACY_POPPLER_AA * 100.0,
    );
    eprintln!(
        "  ({} matched, {} unmatched, {} incorrect)",
        matched,
        total_lines - matched,
        matched - correct,
    );

    // Log misses
    let misses: Vec<&String> = matched_fonts
        .iter()
        .filter(|m| !is_correct(m, &ground_truth))
        .collect();
    if !misses.is_empty() {
        eprintln!("Misses ({}):", misses.len());
        for m in misses.iter().take(20) {
            eprintln!("  {}", m);
        }
        if misses.len() > 20 {
            eprintln!("  ... and {} more", misses.len() - 20);
        }
    }

    assert!(
        accuracy >= MIN_ACCURACY_POPPLER_AA,
        "Specimen Poppler accuracy {:.1}% below threshold {:.0}% ({}/{})",
        accuracy * 100.0,
        MIN_ACCURACY_POPPLER_AA * 100.0,
        correct,
        total_lines,
    );
}
