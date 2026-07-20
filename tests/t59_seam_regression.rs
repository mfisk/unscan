//! t59: Seam split regression test.
//!
//! Generates a 6-line test PDF (L2 removed) covering LibreBodoni (uppercase, lowercase,
//! lining figures) removed L2, EBGaramond (body text), Arial Bold, Roboto Italic, and
//! PlayfairDisplay (lining figures).  Runs unprint with --audit and verifies
//! seam splits match known-good positions.
//!
//! Run:  cargo test --release --test t59_seam_regression -- --nocapture

mod common;

use common::run_unscan;
use std::path::PathBuf;
use std::process::Command;

/// Expected seam splits for each test line, per word.
/// Hardcoded fonts/strings — no audit dependency, no empty lines.
/// Updated Jul 19 2026: switched to hardcoded fonts/strings.
const EXPECTED: &[(&str, &[&[u32]])] = &[
    // LibreBodoni-400 lowercase (L1)
    ("abcdefghijklmnopqrstuvwxyz.", &[
        &[17, 112, 166, 199, 208, 395, 413, 440, 460],
    ]),
    // LibreBodoni-400 uppercase (was L3)
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[
        &[25, 187, 216, 230, 276, 332, 471, 518, 545, 570, 604, 633, 656],
    ]),
    // EBGaramond-400 body text
    ("carved type into wood or imported it from Italy.", &[
        &[99], &[41, 58], &[24, 39, 48, 64], &[11, 26], &[], &[27], &[11],
    ]),
    // Arial-BoldMT-700 Bold
    ("Bold: The quick brown fox jumps over.", &[
        &[], &[], &[35], &[], &[], &[], &[12],
    ]),
    // Roboto-400It Italic
    ("Italic: The quick brown fox jumps over 1,234,567,890 lazy,", &[
        &[85, 150], &[50], &[], &[32, 80], &[32, 64], &[8, 27, 45, 54], &[36], &[19], &[10, 31],
    ]),
    // PlayfairDisplay-400 lining figures (clean)
    ("Lining figures: 0 1 2 3 4 5 6 7 8 9.", &[
        &[44], &[88], &[9], &[], &[], &[], &[], &[], &[], &[], &[], &[],
    ]),
];

#[test]
fn seam_splits_match_ground_truth() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Generate test PDFs from hardcoded fonts/strings (no audit dependency)
    let gen_status = Command::new("python3")
        .arg(repo.join("test-docs/gen-line-test.py"))
        .arg("--hardcoded")
        .current_dir(&repo)
        .status()
        .expect("failed to run gen-line-test.py");
    assert!(gen_status.success(), "gen-line-test.py --hardcoded failed");

    // Copy to expected filenames
    std::fs::copy(
        repo.join("test-docs/line-test-gt.pdf"),
        repo.join("test-docs/line-test-seams-gt.pdf"),
    ).expect("copy gt pdf");
    std::fs::copy(
        repo.join("test-docs/line-test.pdf"),
        repo.join("test-docs/line-test-seams.pdf"),
    ).expect("copy rasterized pdf");

    // Clear page cache
    let _ = std::fs::remove_dir_all("/tmp/unprint-page-cache/line-test-seams");

    let audit_dir = repo.join("test-docs/t59-audit");
    let _ = std::fs::remove_dir_all(&audit_dir);

    let input = repo.join("test-docs/line-test-seams.pdf");
    let gt = repo.join("test-docs/line-test-seams-gt.pdf");
    assert!(input.exists(), "line-test-seams.pdf missing");
    assert!(gt.exists(), "line-test-seams-gt.pdf missing");

    let _output = run_unscan(&input, &[
        "--test", gt.to_str().unwrap(),
        "--audit", audit_dir.to_str().unwrap(),
    ]);

    // Parse audit.json for seam splits
    let audit_path = audit_dir.join("audit.json");
    assert!(audit_path.exists(), "audit.json not produced");

    let audit: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&audit_path).unwrap()
    ).unwrap();

    let entries = audit["text_entries"].as_array().expect("no text_entries");
    assert_eq!(entries.len(), EXPECTED.len(),
        "expected {} lines, got {}", EXPECTED.len(), entries.len());

    for (i, (entry, &(expected_text, expected_word_splits))) in entries.iter().zip(EXPECTED.iter()).enumerate() {
        let text = entry["text"].as_str().unwrap_or("");
        assert_eq!(text, expected_text, "line {i}: text mismatch");

        let ws = entry["word_segmentation"].as_array().expect("no word_segmentation");
        assert_eq!(ws.len(), expected_word_splits.len(),
            "line {i} ({text}): word count mismatch — got {} words, expected {}",
            ws.len(), expected_word_splits.len());

        for (j, (word, expected_splits)) in ws.iter().zip(expected_word_splits.iter()).enumerate() {
            let splits: Vec<u32> = match word["seam_splits"].as_array() {
                Some(arr) => arr.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
                None => vec![],
            };

            assert_eq!(splits, *expected_splits,
                "line {i} word {j} ({text}): seam splits mismatch\n  got:      {splits:?}\n  expected: {expected_splits:?}");
        }
    }
}
