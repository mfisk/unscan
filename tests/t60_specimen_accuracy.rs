//! Font-timeline specimen accuracy test.
//!
//! Runs unscan against the 6-page, 30-section font-timeline-specimen-rasterized.pdf
//! (clean raster, no scan skew) and compares every matched font line against the
//! ground truth in font-timeline-specimen.json.
//!
//! This is intentionally a separate test binary because the specimen takes
//! ~3 min uncached (or ~40s cached). Run with:
//!   cargo test --release --test t60_specimen_accuracy
//!
//! The ground truth JSON has sections with `font_family` names. Each section
//! occupies a vertical band of the specimen; all text lines rendered in that
//! section's font should match it (or a known variant).

mod common;

use common::{test_doc, run_unscan, ensure_index};
use std::collections::HashMap;

/// Minimum acceptable accuracy (correct / total OCR lines).
/// The specimen is rasterized on-demand via PyMuPDF (same FreeType engine as
/// the CI index). Accuracy is lower than the original pre-made raster because
/// PyMuPDF's hinting choices differ from whatever originally generated it.
const MIN_ACCURACY_AA: f64 = 0.991;
const MIN_ACCURACY_NOAA: f64 = 0.965;

/// DPI-specific accuracy floors.  Lower DPI = less detail = lower accuracy.
/// These are non-gating (report-only) until we establish stable baselines.
const DPI_THRESHOLDS: &[(u32, f64)] = &[
    (300, 0.991),  // primary gate — must pass
    (200, 0.0),    // report-only for now
    (150, 0.0),    // report-only for now
];

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

/// Count total OCR lines from unscan output (the denominator for accuracy).
/// Parses "OCR: N words → M lines" from each page.
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

/// Known font renames / aliases.  If the matched font contains any alias
/// value for a ground-truth family, it counts as correct.
/// Includes metric-compatible clones (fonts designed as drop-in replacements).
fn font_aliases() -> HashMap<String, Vec<String>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();
    // Source Sans 3 was formerly Source Sans Pro
    m.insert("sourcesans3".into(), vec!["sourcesanspro".into()]);
    // Courier New ↔ NimbusMonoPS, FreeMono, CourierPrime (metric-compatible)
    m.insert("couriernew".into(), vec![
        "nimbusmonops".into(), "freemono".into(), "courierprime".into(),
    ]);
    // Arial ↔ Liberation Sans, Nimbus Sans, FreeSans (metric-compatible)
    m.insert("arial".into(), vec![
        "liberationsans".into(), "nimbussans".into(), "freesans".into(),
    ]);
    // Times New Roman ↔ NimbusRoman, LiberationSerif, FreeSerif, Tinos, P052, C059
    // (all metric-compatible Times clones)
    m.insert("timesnewroman".into(), vec![
        "nimbusroman".into(), "liberationserif".into(), "freeserif".into(),
        "tinos".into(), "p052".into(), "c059".into(),
    ]);
    // PT Serif — its own design family, no metric clones in the index
    // (NimbusRoman etc. are Times clones, NOT PT Serif clones)
    // Lato ↔ Carlito (designed as metric-compatible replacement)
    m.insert("lato".into(), vec!["carlito".into()]);
    // Caladea ↔ Cambria (metric-compatible)
    m.insert("caladea".into(), vec![]);
    // Verdana ↔ DejaVu Sans (derived from Bitstream Vera, very similar)
    m.insert("verdana".into(), vec!["dejavusans".into()]);
    m
}

/// Check if a matched font name corresponds to any ground truth font family.
/// Returns true if the matched name contains or is contained by any expected family
/// (or a known alias).
fn is_correct(matched: &str, ground_truth: &HashMap<usize, String>) -> bool {
    let aliases = font_aliases();
    ground_truth.values().any(|expected| {
        if matched.contains(expected.as_str()) || expected.contains(matched) {
            return true;
        }
        // Check aliases for this expected family
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

/// Result of a single accuracy measurement at a given DPI.
struct AccuracyResult {
    dpi: u32,
    aa: bool,
    correct: usize,
    total_lines: usize,
    matched: usize,
    accuracy: f64,
    misses: Vec<String>,
}

/// Run accuracy measurement for a single DPI + AA combination.
/// Rasterizes on demand, runs unscan with --dpi, returns structured result.
fn measure_accuracy(dpi: u32, aa: bool) -> Option<AccuracyResult> {
    let vector_src = test_doc("font-timeline-specimen.pdf");
    if !vector_src.exists() {
        eprintln!("SKIP: font-timeline-specimen.pdf not found");
        return None;
    }
    let gt_path = test_doc("font-timeline-specimen.json");
    if !gt_path.exists() {
        eprintln!("SKIP: font-timeline-specimen.json not found");
        return None;
    }

    let aa_suffix = if aa { "" } else { "-noaa" };
    let raster_name = format!("font-timeline-specimen-rasterized{}-{}dpi.pdf", aa_suffix, dpi);
    let raster_path = test_doc(&raster_name);

    if !common::rasterize_pdf(&vector_src, &raster_path, dpi, aa) {
        eprintln!("SKIP: rasterization failed for {} dpi aa={}", dpi, aa);
        return None;
    }

    let ground_truth = load_ground_truth();
    let dpi_args = format!("{}", dpi);
    let output = run_unscan(&raster_path, &["--dpi", &dpi_args]);
    let matched_fonts = parse_all_font_matches(&output);
    let total_lines = count_total_ocr_lines(&output);
    let matched = matched_fonts.len();

    if total_lines == 0 {
        eprintln!("WARNING: 0 OCR lines at {} dpi aa={}", dpi, aa);
        return None;
    }

    let correct = matched_fonts
        .iter()
        .filter(|m| is_correct(m, &ground_truth))
        .count();
    let accuracy = correct as f64 / total_lines as f64;

    let misses: Vec<String> = matched_fonts
        .iter()
        .filter(|m| !is_correct(m, &ground_truth))
        .cloned()
        .collect();

    Some(AccuracyResult {
        dpi,
        aa,
        correct,
        total_lines,
        matched,
        accuracy,
        misses,
    })
}

/// Print accuracy result in a consistent format.
fn report_accuracy(r: &AccuracyResult, label: &str, threshold: f64) {
    eprintln!(
        "{} accuracy @ {}dpi: {}/{} = {:.1}% (threshold: {:.0}%)",
        label, r.dpi, r.correct, r.total_lines, r.accuracy * 100.0, threshold * 100.0,
    );
    eprintln!(
        "  ({} matched, {} unmatched, {} incorrect)",
        r.matched,
        r.total_lines - r.matched,
        r.matched - r.correct,
    );
    if !r.misses.is_empty() {
        eprintln!("  Misses ({}):", r.misses.len());
        for m in r.misses.iter().take(15) {
            eprintln!("    {}", m);
        }
        if r.misses.len() > 15 {
            eprintln!("    ... and {} more", r.misses.len() - 15);
        }
    }
}

#[test]
fn specimen_font_accuracy() {
    ensure_index();
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

    // Generate (or reuse cached) clean raster with anti-aliasing
    let input = test_doc("font-timeline-specimen-rasterized.pdf");
    if !common::rasterize_pdf(&vector_src, &input, 300, true) {
        eprintln!("SKIP: rasterization failed (pdftoppm/img2pdf missing?)");
        return;
    }

    let ground_truth = load_ground_truth();

    let output = run_unscan(&input, &[]);
    let matched_fonts = parse_all_font_matches(&output);
    let total_lines = count_total_ocr_lines(&output);

    let matched = matched_fonts.len();
    assert!(matched > 0, "No font matches found in specimen output");
    assert!(total_lines > 0, "No OCR lines found in specimen output");

    let correct = matched_fonts
        .iter()
        .filter(|m| is_correct(m, &ground_truth))
        .count();

    let accuracy = correct as f64 / total_lines as f64;
    eprintln!(
        "Specimen accuracy: {}/{} = {:.1}% (threshold: {:.0}%)",
        correct,
        total_lines,
        accuracy * 100.0,
        MIN_ACCURACY_AA * 100.0,
    );
    eprintln!(
        "  ({} matched, {} unmatched, {} incorrect)",
        matched,
        total_lines - matched,
        matched - correct,
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
        accuracy >= MIN_ACCURACY_AA,
        "Specimen accuracy {:.1}% below threshold {:.0}% ({}/{})",
        accuracy * 100.0,
        MIN_ACCURACY_AA * 100.0,
        correct,
        total_lines,
    );
}

#[test]
fn specimen_vectorizes_enough_lines() {
    ensure_index();
    let input = test_doc("font-timeline-specimen-rasterized.pdf");
    if !input.exists() {
        eprintln!("SKIP: font-timeline-specimen-rasterized.pdf not found");
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

/// Same as specimen_font_accuracy but against a no-AA rasterized version.
/// Tests whether anti-aliasing blurring affects CI feature matching.
/// The no-AA PDF is generated on demand from the vector source and cached
/// between runs (regenerated if the source PDF is newer).
#[test]
fn specimen_font_accuracy_noaa() {
    ensure_index();
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

    // Generate / cache the no-AA rasterized PDF
    let noaa_pdf = test_doc("font-timeline-specimen-rasterized-noaa.pdf");
    if !common::rasterize_pdf(&vector_src, &noaa_pdf, 300, false) {
        eprintln!("SKIP: could not generate no-AA rasterized PDF (missing pdftoppm or img2pdf?)");
        return;
    }

    let ground_truth = load_ground_truth();

    let output = run_unscan(&noaa_pdf, &[]);
    let matched_fonts = parse_all_font_matches(&output);
    let total_lines = count_total_ocr_lines(&output);

    let matched = matched_fonts.len();
    assert!(matched > 0, "No font matches found in noaa specimen output");
    assert!(total_lines > 0, "No OCR lines found in noaa specimen output");

    let correct = matched_fonts
        .iter()
        .filter(|m| is_correct(m, &ground_truth))
        .count();

    let accuracy = correct as f64 / total_lines as f64;
    eprintln!(
        "Specimen noaa accuracy: {}/{} = {:.1}% (threshold: {:.0}%)",
        correct,
        total_lines,
        accuracy * 100.0,
        MIN_ACCURACY_NOAA * 100.0,
    );
    eprintln!(
        "  ({} matched, {} unmatched, {} incorrect)",
        matched,
        total_lines - matched,
        matched - correct,
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
        accuracy >= MIN_ACCURACY_NOAA,
        "Specimen noaa accuracy {:.1}% below threshold {:.0}% ({}/{})",
        accuracy * 100.0,
        MIN_ACCURACY_NOAA * 100.0,
        correct,
        total_lines,
    );
}

/// Multi-DPI accuracy sweep.  Rasterizes the specimen at each DPI in
/// DPI_THRESHOLDS, runs unscan with --dpi, and reports per-DPI accuracy.
/// The 300 DPI gate is enforced by the dedicated tests above; lower DPIs
/// are report-only until stable baselines are established.
#[test]
fn specimen_multi_dpi_accuracy() {
    ensure_index();

    let mut results: Vec<AccuracyResult> = Vec::new();

    for &(dpi, _threshold) in DPI_THRESHOLDS {
        if let Some(r) = measure_accuracy(dpi, true) {
            results.push(r);
        }
    }

    // Summary table
    eprintln!("\n╔══════════════════════════════════════════════════╗");
    eprintln!("║         MULTI-DPI ACCURACY SUMMARY (AA)         ║");
    eprintln!("╠═══════╦════════════╦═══════╦═══════╦════════════╣");
    eprintln!("║  DPI  ║  Accuracy  ║ Correct║ Total ║  Threshold ║");
    eprintln!("╠═══════╬════════════╬═══════╬═══════╬════════════╣");
    for r in &results {
        let threshold = DPI_THRESHOLDS.iter()
            .find(|(d, _)| *d == r.dpi)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        let gate = if threshold > 0.0 { "GATE" } else { "info" };
        eprintln!(
            "║  {:>3}  ║   {:>5.1}%   ║  {:>3}  ║  {:>3}  ║ {:>5.1}% {:>4} ║",
            r.dpi, r.accuracy * 100.0, r.correct, r.total_lines,
            threshold * 100.0, gate,
        );
    }
    eprintln!("╚═══════╩════════════╩═══════╩═══════╩════════════╝");

    // Report per-DPI detail
    for r in &results {
        let threshold = DPI_THRESHOLDS.iter()
            .find(|(d, _)| *d == r.dpi)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        report_accuracy(r, "AA", threshold);
    }

    // Only enforce thresholds > 0
    for r in &results {
        let threshold = DPI_THRESHOLDS.iter()
            .find(|(d, _)| *d == r.dpi)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        if threshold > 0.0 {
            assert!(
                r.accuracy >= threshold,
                "AA accuracy at {}dpi: {:.1}% below threshold {:.0}% ({}/{})",
                r.dpi, r.accuracy * 100.0, threshold * 100.0, r.correct, r.total_lines,
            );
        }
    }
}

/// Same multi-DPI sweep but with no-AA rasterization.
#[test]
fn specimen_multi_dpi_accuracy_noaa() {
    ensure_index();

    let noaa_thresholds: &[(u32, f64)] = &[
        (300, MIN_ACCURACY_NOAA),
        (200, 0.0),
        (150, 0.0),
    ];

    let mut results: Vec<AccuracyResult> = Vec::new();

    for &(dpi, _threshold) in noaa_thresholds {
        if let Some(r) = measure_accuracy(dpi, false) {
            results.push(r);
        }
    }

    // Summary table
    eprintln!("\n╔══════════════════════════════════════════════════╗");
    eprintln!("║       MULTI-DPI ACCURACY SUMMARY (NO-AA)        ║");
    eprintln!("╠═══════╦════════════╦═══════╦═══════╦════════════╣");
    eprintln!("║  DPI  ║  Accuracy  ║ Correct║ Total ║  Threshold ║");
    eprintln!("╠═══════╬════════════╬═══════╬═══════╬════════════╣");
    for r in &results {
        let threshold = noaa_thresholds.iter()
            .find(|(d, _)| *d == r.dpi)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        let gate = if threshold > 0.0 { "GATE" } else { "info" };
        eprintln!(
            "║  {:>3}  ║   {:>5.1}%   ║  {:>3}  ║  {:>3}  ║ {:>5.1}% {:>4} ║",
            r.dpi, r.accuracy * 100.0, r.correct, r.total_lines,
            threshold * 100.0, gate,
        );
    }
    eprintln!("╚═══════╩════════════╩═══════╩═══════╩════════════╝");

    for r in &results {
        let threshold = noaa_thresholds.iter()
            .find(|(d, _)| *d == r.dpi)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        report_accuracy(r, "noaa", threshold);
    }

    for r in &results {
        let threshold = noaa_thresholds.iter()
            .find(|(d, _)| *d == r.dpi)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        if threshold > 0.0 {
            assert!(
                r.accuracy >= threshold,
                "noaa accuracy at {}dpi: {:.1}% below threshold {:.0}% ({}/{})",
                r.dpi, r.accuracy * 100.0, threshold * 100.0, r.correct, r.total_lines,
            );
        }
    }
}
