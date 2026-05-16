//! Font-timeline specimen accuracy test.
//!
//! Runs unscan against the 6-page, 30-section font-timeline-specimen-scanned.pdf
//! and compares every matched font line against the ground truth in
//! font-timeline-specimen.json.
//!
//! This is intentionally a separate test binary because the specimen takes
//! ~10 min uncached (or ~40s cached). Run with:
//!   cargo test --release --test t60_specimen_accuracy
//!
//! The ground truth JSON has sections with `font_family` names. Each section
//! occupies a vertical band of the specimen; all text lines rendered in that
//! section's font should match it (or a known variant).

mod common;

use common::{test_doc, run_unscan};
use std::collections::HashMap;

/// Minimum acceptable accuracy (correct / total matched lines).
const MIN_ACCURACY: f64 = 0.78;

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

/// Extract matched font names from unscan output lines containing ✓.
/// Returns vec of lowercase, space-stripped font names.
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

/// Check if a matched font name corresponds to any ground truth font family.
/// Returns true if the matched name contains or is contained by any expected family.
fn is_correct(matched: &str, ground_truth: &HashMap<usize, String>) -> bool {
    ground_truth
        .values()
        .any(|expected| matched.contains(expected.as_str()) || expected.contains(matched))
}

#[test]
fn specimen_font_accuracy() {
    let input = test_doc("font-timeline-specimen-scanned.pdf");
    if !input.exists() {
        eprintln!("SKIP: font-timeline-specimen-scanned.pdf not found");
        return;
    }
    let gt_path = test_doc("font-timeline-specimen.json");
    if !gt_path.exists() {
        eprintln!("SKIP: font-timeline-specimen.json not found");
        return;
    }

    let ground_truth = load_ground_truth();

    let output = run_unscan(&input, &[]);
    let matched_fonts = parse_all_font_matches(&output);

    let total = matched_fonts.len();
    assert!(total > 0, "No font matches found in specimen output");

    let correct = matched_fonts
        .iter()
        .filter(|m| is_correct(m, &ground_truth))
        .count();

    let accuracy = correct as f64 / total as f64;
    eprintln!(
        "Specimen accuracy: {}/{} = {:.1}% (threshold: {:.0}%)",
        correct,
        total,
        accuracy * 100.0,
        MIN_ACCURACY * 100.0,
    );

    // Log misses for debugging
    let misses: Vec<&String> = matched_fonts
        .iter()
        .filter(|m| !is_correct(m, &ground_truth))
        .collect();
    if !misses.is_empty() {
        eprintln!("Misses ({}):", misses.len());
        for m in misses.iter().take(15) {
            eprintln!("  {}", m);
        }
        if misses.len() > 15 {
            eprintln!("  ... and {} more", misses.len() - 15);
        }
    }

    assert!(
        accuracy >= MIN_ACCURACY,
        "Specimen accuracy {:.1}% below threshold {:.0}% ({}/{})",
        accuracy * 100.0,
        MIN_ACCURACY * 100.0,
        correct,
        total,
    );
}

#[test]
fn specimen_vectorizes_enough_lines() {
    let input = test_doc("font-timeline-specimen-scanned.pdf");
    if !input.exists() {
        eprintln!("SKIP: font-timeline-specimen-scanned.pdf not found");
        return;
    }

    let output = run_unscan(&input, &[]);
    let count = common::parse_vectorized_count(&output)
        .expect("could not parse vectorized count from specimen");

    eprintln!("specimen vectorized = {}", count);
    // 30 sections × ~16 lines each ≈ 480 lines; some go raster.
    // Threshold: at least 350 lines vectorized.
    assert!(
        count >= 350,
        "specimen vectorized {} lines, expected >= 350",
        count,
    );
}
