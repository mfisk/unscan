//! Vintage legacy matching — real PostScript-era fonts plus Word6 quantized kern.
//!
//! Runs unprint with --audit and --test against the vintage specimen
//! (`test-docs/vintage-specimen-gt.pdf` + `-rasterized.pdf`): the same
//! 3-line passage rendered once per variant, all 5 variants on ONE page,
//! one section per variant, 15 scored lines total. The same text in every
//! variant makes the comparison apples-to-apples: only the font/variant
//! varies, not the text.
//!
//! Each section is self-documenting: a header at the top states the font
//! name and variant, the era, how that vintage is recreated (clone,
//! reconstructing, or quantizing), and the key characteristics (kern pair
//! count, kerning mechanism, outline format). The header lines are NOT
//! scored — they are rendered in DejaVu Sans and annotated with the
//! sentinel canonical "HEADER-NO-SCORE", and this test skips audit
//! entries whose expected font is the sentinel.
//!
//! Sections 1-4 are REAL vintage fonts — the URW Base 35 OTF fonts
//! (PostScript Level 2 core font set, 1991) — with STANDARD catalog font
//! keys (Mike Nov 12, 2026: use real fonts, not synthetic reconstructions).
//! Section 5 is the Word6 synthetic variant (Times New Roman, kern
//! quantized to multiples of 5): the one era that still needs synthetic
//! generation because no real font exhibits quantized kern, so no standard
//! font key represents it (Mike Nov 12, 2026 correction).
//!
//! Variants are identified by their expected font key (from the audit's
//! `expected_font`), not by page or line position — all sections share
//! one page. The test asserts the pipeline picks the expected font key
//! for each scored line; the picked key must match exactly.
//!
//! Thresholds (stated, conservative):
//!   MIN_CORRECT_FONT_RATE = 1.0 — every scored line must pick its
//!     expected font (15/15). The expected font IS the correct answer;
//!     there is no partial credit for picking a lookalike.
//!   MIN_ZNCC_AVG = 0.92 — same as t65, computed over the scored lines
//!     only (header lines excluded). The correct render should score at
//!     least as well as the t65 baseline.
//!   Wrong picks: 0 allowed — any scored line picking a font other than
//!     its expected key fails the test.
//!
//! Do NOT weaken the thresholds to make it pass.
//!
//! Run with:
//!   env CARGO_BIN_EXE_unprint=/home/hatch/workspace/repos/unscan-vintage/target/fastbuild/unprint cargo test --profile fastbuild -p unprint --test t66_vintage_legacy_matching -- --nocapture
//! (The env var is required: `cargo test -p unprint` never sets
//! CARGO_BIN_EXE_unprint, so without it the harness falls back to the main
//! repo's debug binary, which lacks the vintage fixes.)

mod common;

use common::{test_doc, ensure_index};

/// Minimum acceptable fraction of scored lines picking the expected font.
/// 15/15 — the expected font is the correct answer for each scored line.
const MIN_CORRECT_FONT_RATE: f64 = 1.0;

/// Minimum acceptable mean ZNCC similarity (same as t65), over scored lines only.
const MIN_ZNCC_AVG: f64 = 0.92;

/// Sentinel canonical marking header lines as unscored (must match the
/// generator's HEADER_SENTINEL in test-docs/gen-vintage-specimen.py).
const HEADER_SENTINEL: &str = "HEADER-NO-SCORE";

/// Expected font key -> (font display name, era label), for reporting only.
/// The pass/fail decision compares picked vs expected (both from the
/// audit), never against this table. Keys are the verbatim catalog keys
/// (--resolve-fontkey for the URW fonts, vintage_key_dump for Word6).
const KEY_TO_VARIANT: [(&str, &str, &str); 5] = [
    ("Nimbus-400|kern|nolig", "Times New Roman Regular (URW clone: Nimbus Roman Regular)", "PostScript Level 2 (1991)"),
    ("Nimbus-700|kern|nolig", "Times New Roman Bold (URW clone: Nimbus Roman Bold)", "PostScript Level 2 (1991)"),
    ("Nimbus-400Italic|kern|nolig", "Times New Roman Italic (URW clone: Nimbus Roman Italic)", "PostScript Level 2 (1991)"),
    ("NimbusMonoPS-400|nokern|nolig", "Courier Regular (URW clone: Nimbus Mono PS Regular)", "PostScript Level 2 (1991)"),
    ("TimesNewRomanPSMT-400|vintage=word6", "Times New Roman Word6", "Word 6.0 (1993)"),
];

#[test]
fn vintage_legacy_matching() {
    ensure_index();

    let vector = test_doc("vintage-specimen-gt.pdf");
    let raster = test_doc("vintage-specimen-rasterized.pdf");

    let r = common::measure_accuracy(&raster, &vector, "t66-vintage");

    // Per-line results from the audit (the --test summary has no per-line picks).
    let audit_dir = r.report_path.parent().expect("report_path has no parent");
    let audit_json: serde_json::Value = {
        let p = audit_dir.join("audit.json");
        let s = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", p.display(), e));
        serde_json::from_str(&s).expect("audit.json is not valid JSON")
    };
    let entries = audit_json["text_entries"]
        .as_array()
        .expect("audit.json has no text_entries array");

    // Score only the repeated test-text lines. Header lines (expected font
    // is the HEADER-NO-SCORE sentinel) are skipped — they document the
    // variant, they are not test lines. Metrics are computed over the
    // scored lines only; r.compared / r.zncc_avg include the headers.
    let mut correct = 0usize;
    let mut scored = 0usize;
    let mut skipped_headers = 0usize;
    let mut sim_sum = 0.0f64;
    let mut sim_count = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    for e in entries {
        let expected = e["expected_font"].as_str().unwrap_or("").to_string();
        if expected == HEADER_SENTINEL {
            skipped_headers += 1;
            continue;
        }
        let page = e["page"].as_u64().unwrap_or(0);
        let line_index = e["line_index"].as_u64().unwrap_or(0);
        let picked = e["font_key_matched"].as_str().unwrap_or("").to_string();
        let sim = e["similarity_score"].as_f64().unwrap_or(0.0);
        let miss_type = e["miss_type"].as_str().unwrap_or("?");
        let (font_name, era) = KEY_TO_VARIANT
            .iter()
            .find(|(k, _, _)| *k == expected)
            .map(|(_, n, e)| (*n, *e))
            .unwrap_or(("?", "?"));
        scored += 1;
        sim_sum += sim;
        sim_count += 1;
        let accepted = picked == expected;
        if accepted {
            correct += 1;
        } else {
            wrong.push(format!(
                "p{} l{} [{} | {}]: expected {} picked {} (sim {:.4}, {})",
                page, line_index, font_name, era, expected, picked, sim, miss_type
            ));
        }
        eprintln!(
            "  p{} l{} [{} | {}]: expected {} | picked {} | sim {:.4} | {} | {}",
            page,
            line_index,
            font_name,
            era,
            expected,
            picked,
            sim,
            miss_type,
            if accepted { "OK" } else { "WRONG" }
        );
    }

    let rate = if scored > 0 {
        correct as f64 / scored as f64
    } else {
        0.0
    };
    let zncc_avg = if sim_count > 0 {
        sim_sum / sim_count as f64
    } else {
        0.0
    };
    eprintln!(
        "Vintage: {}/{} scored lines picked the expected font = {:.1}% (threshold: {:.0}%), ZNCC avg {:.4} (threshold: {:.2}); {} header lines skipped",
        correct, scored, rate * 100.0,
        MIN_CORRECT_FONT_RATE * 100.0, zncc_avg, MIN_ZNCC_AVG, skipped_headers,
    );
    eprintln!(
        "  (unfiltered: {} compared, {} hits, {} minor, {} major, {} sim_fail)",
        r.compared, r.hits, r.minor_misses, r.major_misses, r.similarity_failures,
    );
    eprintln!("  Miss report: {}", r.report_path.display());

    assert_eq!(scored, 15,
        "Vintage specimen should score 15 lines (5 variants x 3 lines), got {} — see {}",
        scored, r.report_path.display());
    assert!(
        wrong.is_empty(),
        "Vintage test: {} scored line(s) picked the wrong font:\n{}",
        wrong.len(),
        wrong.join("\n"),
    );
    assert!(
        rate >= MIN_CORRECT_FONT_RATE,
        "Vintage correct-font rate {:.1}% below threshold {:.0}% ({}/{}) — see {}",
        rate * 100.0,
        MIN_CORRECT_FONT_RATE * 100.0,
        correct,
        scored,
        r.report_path.display(),
    );
    assert!(
        zncc_avg >= MIN_ZNCC_AVG,
        "Vintage ZNCC avg {:.4} below threshold {:.2} — see {}",
        zncc_avg, MIN_ZNCC_AVG, r.report_path.display(),
    );
}
