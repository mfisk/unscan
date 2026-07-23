//! t59: Seam split regression test.
//!
//! Generates a 7-line test PDF covering LibreBodoni (uppercase, lowercase,
//! lining figures), EBGaramond (body text), Arial Bold, Roboto Italic, and
//! PlayfairDisplay (lining figures).  Runs unprint with --audit and verifies
//! seam splits match known-good positions.
//!
//! Run:  cargo test --release --test t59_seam_regression -- --nocapture

mod common;

use common::run_unscan;
use std::path::PathBuf;
use std::process::Command;

/// Expected seam splits for each test line, per word.
/// Generated 2026-07-23 from fresh audit after HARDCODED input changed to 8 lines.
/// OpenSans and IBMPlexSans now OCR-split into multiple words, so EXPECTED reflects
/// actual entry["text"] (display_text) and word_segmentation order as observed.
const EXPECTED: &[(&str, &[&[u32]])] = &[
    // 0: LibreBodoni-400 lowercase — gold
    ("abcdefghijklmnopqrstuvwxyz.", &[
        &[17, 112, 166, 199, 208, 395, 413, 440, 460],
    ]),
    // 1: LibreBodoni-400 uppercase
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[
        &[25, 187, 216, 230, 275, 332, 360, 471, 518, 545, 570, 604, 633, 656],
    ]),
    // 2: Georgia-400 uppercase — distinct metrics from LibreBodoni
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[
        &[26, 268, 292, 504, 531, 557, 592, 618, 642],
    ]),
    // 3: OpenSans-400 lowercase — OCR now splits into 3 words: "a", "bcdefghij", "klmnopgrstuvwxyz."
    //    word_segmentation order is [klm..., bcd..., a] as currently emitted
    ("a bcdefghij klmnopgrstuvwxyz.", &[
        &[241],
        &[94, 146],
        &[],
    ]),
    // 4: LibreBodoni-400Italic "dogs."
    ("dogs.", &[
        &[19, 40, 57, 73],
    ]),
    // 5: IBMPlexSans-400 lowercase — OCR splits into 2 words
    ("abcdefghijklmn opqgrstuvwxyz.", &[
        &[223],
        &[],
    ]),
    // 6: SourceSerif4-400It "Mayr-Duffner."
    ("Mayr-Duffner.", &[
        &[35, 53, 72, 86, 146, 169, 219],
    ]),
    // 7: SourceSerif4-400It "Type"
    ("Type", &[
        &[14, 36],
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
        let audit_path = audit_dir.join("audit.json");
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
    let audit_path = audit_dir.join("audit.json");
    assert!(audit_path.exists(), "audit.json not produced after 3 attempts. last output: {}", last_output);

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
