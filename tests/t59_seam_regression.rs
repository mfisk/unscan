//! t59: Seam split regression test.
//!
//! Generates a 3-line test PDF (p1:L72 LibreBodoni uppercase, p1:L73 LibreBodoni
//! lowercase, p3:L45 Georgia uppercase) from the BAP audit, runs unprint with
//! --audit, and verifies seam splits match known-good positions.
//!
//! Run:  cargo test --release --test t59_seam_regression -- --nocapture

mod common;

use common::run_unscan;
use std::path::PathBuf;
use std::process::Command;

/// Expected seam splits for each test line (verified correct Jul 12 2026).
const EXPECTED: &[(&str, &[u32])] = &[
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[25, 188, 232, 274, 331, 522, 548, 574, 612, 632, 650]),
    ("abcdefghijklmnopqrstuvwxyz.", &[19, 114, 165, 194, 399, 418, 438, 455]),
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[25, 267, 291, 506, 537, 562, 588, 613, 635]),
];

#[test]
fn seam_splits_match_ground_truth() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Generate test PDFs from BAP audit
    let gen_status = Command::new("python3")
        .arg(repo.join("test-docs/gen-line-test.py"))
        .args(["1:72", "1:73", "3:45"])
        .current_dir(&repo)
        .status()
        .expect("failed to run gen-line-test.py");
    assert!(gen_status.success(), "gen-line-test.py failed");

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

    for (i, (entry, &(expected_text, expected_splits))) in entries.iter().zip(EXPECTED.iter()).enumerate() {
        let text = entry["text"].as_str().unwrap_or("");
        assert_eq!(text, expected_text, "line {i}: text mismatch");

        let ws = entry["word_segmentation"].as_array().expect("no word_segmentation");
        assert!(!ws.is_empty(), "line {i}: no words");

        let splits: Vec<u32> = ws[0]["seam_splits"].as_array()
            .expect("no seam_splits")
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        assert_eq!(splits, expected_splits,
            "line {i} ({text}): seam splits mismatch\n  got:      {splits:?}\n  expected: {expected_splits:?}");
    }
}
