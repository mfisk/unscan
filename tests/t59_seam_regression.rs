//! t59: Seam split regression test.
//!
//! Generates an 18-line test PDF covering LibreBodoni, Georgia, OpenSans,
//! LibreBaskerville, PTSerif italic fox, etc. Runs unprint with --audit
//! and checks segmentation against recorded known-correct expectations.
//!
//! The EXPECTED table records, per word, what is KNOWN to be correct and
//! nothing else:
//! - `k`: the known-correct segment count, checked against the audit word
//!   object's `n_segments_produced`.
//! - `columns`: individual seam columns known to be correct; each must
//!   appear in the observed `seam_splits`. Columns not recorded are not
//!   checked at all.
//! - A word with neither k nor columns recorded is fully undefined: no
//!   assertion, counted and reported as unverified.
//!
//! Recorded values must be verified against primary evidence (the visible
//! word text, the audit's `n_chars_expected`, crops), never copied from a
//! previous run's output. The old snapshot table enshrined the segmenter's
//! own bugs (e.g. the col-34 M-split for "(Microsoft)"); only the 86 and
//! 202 columns for that word are recorded, because the seam overlay shows
//! them falling in clean gaps at true character boundaries.
//!
//! The test prints a summary (verified checks passed, misses, unverified
//! items) and fails on ANY miss among verified items.
//!
//! Run:  cargo test --release --test t59_seam_regression -- --nocapture

mod common;

use common::{run_unscan, audit_json_path, read_audit_json_text};
use std::path::PathBuf;
use std::process::Command;

/// Per-word expectation: known-correct segment count and/or known-correct
/// seam columns. Anything not recorded is unverified and not asserted.
struct WordExpect {
    /// Known-correct segment count (k), checked against `n_segments_produced`.
    k: Option<usize>,
    /// Individual seam columns known to be correct. Each must appear in the
    /// observed `seam_splits`. Unrecorded columns are not checked.
    columns: &'static [u32],
}

/// Word with only a known-correct segment count.
const fn k_only(k: usize) -> WordExpect {
    WordExpect { k: Some(k), columns: &[] }
}

/// Expected segmentation per test line, per word.
///
/// k values are verified against the visible word text and the audit's
/// `n_chars_expected` (three-way agreement with the observed and GT segment
/// counts, except where noted). Recorded columns are verified against the
/// seam overlay crops. The col-34 M-split for "(Microsoft)" is a known
/// segmenter bug and is deliberately NOT recorded; only 86 and 202 are,
/// both falling in clean gaps at true character boundaries.
/// Word_segmentation order is as emitted by Tesseract; first 10 lines share
/// page for word-gap stats.
const EXPECTED: &[(&str, &[WordExpect])] = &[
    // 0: LibreBodoni-400 "abcdefghijklmnopqrstuvwxyz."
    ("abcdefghijklmnopqrstuvwxyz.", &[k_only(27)]),
    // 1: LibreBodoni-400 "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[k_only(10), k_only(17)]),
    // 2: Georgia-400 "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[k_only(27)]),
    // 3: OpenSans-400 "abcdefghijklmnopqrstuvwxyz."
    ("abcdefghijklmnopqrstuvwxyz.", &[k_only(27)]),
    // 4: LibreBodoni-400Italic "dogs."
    ("dogs.", &[k_only(5)]),
    // 5: SourceSerif4-400Italic "Mayr-Duffner."
    // k=13 verified visually: the crop shows two separate f glyphs, each
    // with its own crossbar — no ff ligature in the (nolig) render. The GT
    // segmentation's 12 is a GT-path artifact (gt_ligature_mask collapses
    // ff via support+overlap even for nolig variants).
    ("Mayr-Duffner.", &[k_only(13)]),
    // 6: SourceSerif4-400Italic "Type"
    ("Type", &[k_only(4)]),
    // 7: LibreBaskerville-400 "abcdefghijklmnopqrstuvwxyz."
    ("abcdefghijklmnopqrstuvwxyz.", &[k_only(27)]),
    // 8: PTSerif-400Italic "Italic: The quick brown fox jumps over 1,234,567,890 lazy"
    // The number splits into "1,234," and "567,890" (wide-whitespace split).
    ("Italic: The quick brown fox jumps over 1,234,567,890 lazy", &[
        k_only(7), k_only(3), k_only(5), k_only(5), k_only(3),
        k_only(5), k_only(4), k_only(6), k_only(7), k_only(4),
    ]),
    // 9: SourceSerif4-400Italic "Font: Originally for IBM Executive typewriters — 12 characters per inch"
    ("Font: Originally for IBM Executive typewriters — 12 characters per inch", &[
        k_only(5), k_only(10), k_only(3), k_only(3), k_only(9),
        k_only(11), k_only(1), k_only(2), k_only(10), k_only(3), k_only(4),
    ]),
    // 10: Georgia-400 "Matthew Carter created Georgia in 1993."
    ("Matthew Carter created Georgia in 1993.", &[
        k_only(7), k_only(6), k_only(7), k_only(7), k_only(2), k_only(5),
    ]),
    // 11: Arial-ItalicMT-400Italic "Italic: The quick brown fox jumps over 1,234,567,890"
    // The number stays whole here (unlike line 8).
    ("Italic: The quick brown fox jumps over 1,234,567,890", &[
        k_only(7), k_only(3), k_only(5), k_only(5),
        k_only(3), k_only(5), k_only(4), k_only(13),
    ]),
    // 12: Nimbus-400Italic "AVAST! WAYWARD YOUTH TO YONDER VALLEY, AWAY AT DAWN. Ryan Tries."
    ("AVAST! WAYWARD YOUTH TO YONDER VALLEY, AWAY AT DAWN. Ryan Tries.", &[
        k_only(6), k_only(7), k_only(5), k_only(2), k_only(6),
        k_only(7), k_only(4), k_only(2), k_only(5), k_only(4), k_only(6),
    ]),
    // 13: SourceSerif4-400Italic "Font: fonts.google.com/specimen/Jost — OFL, Owen Earl"
    ("Font: fonts.google.com/specimen/Jost — OFL, Owen Earl", &[
        k_only(5), k_only(6), k_only(24), k_only(1),
        k_only(4), k_only(4), k_only(4),
    ]),
    // 14: LibreBodoni-400 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", &[k_only(26)]),
    // 15: SourceSerif4-400Italic "Hunt for Adobe"
    ("Hunt for Adobe", &[k_only(4), k_only(3), k_only(5)]),
    // 16: Georgia-700 "1993 — Georgia (Microsoft)"
    // "(Microsoft)": k=11, columns 86 and 202 verified against the seam
    // overlay (both fall in clean gaps at true character boundaries).
    // Column 34 is the known M-split bug and is deliberately NOT recorded.
    ("1993 — Georgia (Microsoft)", &[
        k_only(4), k_only(1), k_only(7),
        WordExpect { k: Some(11), columns: &[86, 202] },
    ]),
    // 17: LibreCaslonText-400 "saying: \"When in doubt, use Caslon.\""
    ("saying: \"When in doubt, use Caslon.\"", &[
        k_only(7), k_only(5), k_only(2), k_only(6), k_only(3), k_only(8),
    ]),
];


#[test]
fn seam_splits_match_ground_truth() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let gt_pdf = repo.join("test-docs/line-test-gt.pdf");
    let raster_pdf = repo.join("test-docs/line-test.pdf");
    let seams_gt = repo.join("test-docs/line-test-seams-gt.pdf");
    let seams_input = repo.join("test-docs/line-test-seams.pdf");

    // Only regenerate if missing or env var forces it, to avoid WeasyPrint memory pressure
    let need_gen = std::env::var("FORCE_T59_GEN").is_ok()
        || !gt_pdf.exists()
        || !raster_pdf.exists()
        || !seams_gt.exists()
        || !seams_input.exists();

    if need_gen {
        let gen_status = Command::new("python3")
            .arg(repo.join("test-docs/gen-line-test.py"))
            .arg("--hardcoded")
            .current_dir(&repo)
            .status()
            .expect("failed to run gen-line-test.py");
        assert!(gen_status.success(), "gen-line-test.py --hardcoded failed");
    } else {
        eprintln!("  Using cached line-test PDFs (skip --hardcoded gen)");
    }

    // Copy to expected filenames (if we skipped gen, these may already exist)
    if gt_pdf.exists() {
        let _ = std::fs::copy(&gt_pdf, &seams_gt);
    }
    if raster_pdf.exists() {
        let _ = std::fs::copy(&raster_pdf, &seams_input);
    }

    // Clear page caches (both old /tmp and new TMPDIR locations)
    let _ = std::fs::remove_dir_all("/tmp/unprint-page-cache/line-test-seams");
    let _ = std::fs::remove_dir_all("/home/hatch/workspace/tmp/unprint-page-cache/line-test-seams");
    let _ = std::fs::remove_dir_all("/home/hatch/workspace/tmp/unprint-page-cache");

    let audit_dir = repo.join("test-docs/t59-audit");
    let _ = std::fs::remove_dir_all(&audit_dir);

    let input = repo.join("test-docs/line-test-seams.pdf");
    let gt = repo.join("test-docs/line-test-seams-gt.pdf");
    assert!(input.exists(), "line-test-seams.pdf missing");
    assert!(gt.exists(), "line-test-seams-gt.pdf missing");

    // Retry loop for OOM-prone VM (7.8G no-swap). First unprint after WeasyPrint often OOMs.
    let mut last_output = String::new();
    for attempt in 1..=3 {
        eprintln!("  t59 attempt {attempt}/3 running unprint...");
        last_output = run_unscan(&input, &[
            "--test", gt.to_str().unwrap(),
            "--audit", audit_dir.to_str().unwrap(),
        ]);
        let audit_path = audit_json_path(&audit_dir);
        if audit_path.exists() {
            eprintln!("  audit.json produced on attempt {attempt}");
            break;
        } else {
            eprintln!("  attempt {attempt} failed to produce audit.json, retrying...");
            eprintln!("  output tail: {}", last_output.chars().rev().take(500).collect::<String>());
            // Clear partial audit dir and page cache before retry
            let _ = std::fs::remove_dir_all(&audit_dir);
            let _ = std::fs::remove_dir_all("/tmp/unprint-page-cache/line-test-seams");
            let _ = std::fs::remove_dir_all("/home/hatch/workspace/tmp/unprint-page-cache/line-test-seams");
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    // Parse audit.json for seam splits
    let audit_path = audit_json_path(&audit_dir);
    assert!(audit_path.exists(), "audit.json not produced after 3 attempts. last output: {}", last_output);

    let audit: serde_json::Value = serde_json::from_str(
        &read_audit_json_text(&audit_path)
    ).unwrap();

    let entries = audit["text_entries"].as_array().expect("no text_entries");
    assert_eq!(entries.len(), EXPECTED.len(),
        "expected {} lines, got {}", EXPECTED.len(), entries.len());

    // Verified checks passed, misses (verified items that failed), and
    // unverified items (recorded as nothing — reported, not asserted).
    let mut verified: usize = 0;
    let mut misses: Vec<String> = Vec::new();
    let mut unverified: Vec<String> = Vec::new();
    let mut n_k_recorded: usize = 0;
    let mut n_cols_recorded: usize = 0;
    let mut n_words_undef: usize = 0;

    for (i, (entry, &(expected_text, expected_words))) in entries.iter().zip(EXPECTED.iter()).enumerate() {
        // Use gt_text when available (robust against pflda false corrections like x->w),
        // fallback to display text.
        let text = entry.get("gt_text").and_then(|v| v.as_str())
            .or_else(|| entry.get("text").and_then(|v| v.as_str()))
            .unwrap_or("");
        assert_eq!(text, expected_text, "line {i}: text mismatch (gt_text vs expected)");

        let ws = entry["word_segmentation"].as_array().expect("no word_segmentation");
        assert_eq!(ws.len(), expected_words.len(),
            "line {i} ({text}): word count mismatch — got {} words, expected {}",
            ws.len(), expected_words.len());

        for (j, (word, exp)) in ws.iter().zip(expected_words.iter()).enumerate() {
            let wtext = word.get("word_text").and_then(|v| v.as_str()).unwrap_or("?");
            let n_seg = word["n_segments_produced"].as_u64().unwrap() as usize;
            let splits: Vec<u32> = match word["seam_splits"].as_array() {
                Some(arr) => arr.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
                None => vec![],
            };

            match exp.k {
                Some(k) => {
                    n_k_recorded += 1;
                    if n_seg == k {
                        verified += 1;
                    } else {
                        misses.push(format!(
                            "line {i} word {j} ({wtext}): k mismatch — got {n_seg} segments, expected {k}"));
                    }
                }
                None => unverified.push(format!(
                    "line {i} word {j} ({wtext}): k unverified")),
            }

            if exp.columns.is_empty() {
                unverified.push(format!(
                    "line {i} word {j} ({wtext}): columns unverified (0 recorded)"));
            } else {
                for &col in exp.columns {
                    n_cols_recorded += 1;
                    if splits.contains(&col) {
                        verified += 1;
                    } else {
                        misses.push(format!(
                            "line {i} word {j} ({wtext}): recorded column {col} missing from observed splits {splits:?}"));
                    }
                }
            }

            if exp.k.is_none() && exp.columns.is_empty() {
                n_words_undef += 1;
            }
        }
    }

    // Report: verified counts, misses, and every unverified item.
    eprintln!("t59 summary: {verified} verified checks passed, {} miss(es), {} unverified item(s)",
        misses.len(), unverified.len());
    eprintln!("  k recorded on {n_k_recorded} words, {n_cols_recorded} columns recorded, {n_words_undef} words fully undefined");
    for u in &unverified {
        eprintln!("  unverified: {u}");
    }
    // Break on misses: any verified item that fails, fails the test.
    assert!(misses.is_empty(),
        "t59 FAILED with {} miss(es):\n{}", misses.len(), misses.join("\n"));
}
