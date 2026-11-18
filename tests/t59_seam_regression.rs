//! t59: Seam split regression test.
//!
//! Generates a 10-line test PDF covering LibreBodoni, Georgia, OpenSans,
//! LibreBaskerville, PTSerif italic fox, etc. Runs unprint with --audit
//! and verifies seam splits match known-good positions after 2-horiz DP.
//!
//! Run:  cargo test --release --test t59_seam_regression -- --nocapture

mod common;

use common::{run_unscan, audit_json_path, read_audit_json_text};
use std::path::PathBuf;
use std::process::Command;

/// Expected seam splits for each test line, per word.
/// Regenerated 2026-11-17 from fresh audit after HARDCODED expanded to 18 lines
/// (all 11 original lines re-captured because the font registry changed,
/// so winning fonts and seam columns moved).
/// Captured from test-docs/t59-audit/audit.json after 2-horiz DP. Word_segmentation
/// order is as emitted by Tesseract; first 10 lines share page for word-gap stats.
const EXPECTED: &[(&str, &[&[u32]])] = &[
    // 0: LibreBodoni-400 "abcdefghijklmnopqrstuvwxyz."
    ("abcdefghijklmnopqrstuvwxyz.", &[
        &[17, 112, 165, 198, 208, 373, 395, 424, 438, 459],
    ]),
    // 1: LibreBodoni-400 "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[
        &[24, 185, 216, 229],
        &[28, 84, 112, 223, 297, 321, 356, 384, 408],
    ]),
    // 2: Georgia-400 "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ.", &[
        &[26, 268, 290, 459, 530, 556, 592, 618, 642],
    ]),
    // 3: OpenSans-400 "abcdefghijklmnopqrstuvwxyz."
    ("abcdefghijklmnopqrstuvwxyz.", &[
        &[117, 167, 471],
    ]),
    // 4: LibreBodoni-400Italic "dogs."
    ("dogs.", &[
        &[20, 39, 58, 72],
    ]),
    // 5: SourceSerif4-400Italic "Mayr-Duffner."
    ("Mayr-Duffner.", &[
        &[35, 56, 94, 155, 166, 181, 237],
    ]),
    // 6: SourceSerif4-400Italic "Type"
    ("Type", &[
        &[14, 38],
    ]),
    // 7: LibreBaskerville-400 "abcdefghijklmnopqrstuvwxyz."
    ("abcdefghijklmnopqrstuvwxyz.", &[
        &[19, 127, 149, 187, 221, 375, 423, 448, 471, 503, 526],
    ]),
    // 8: PTSerif-400Italic "Italic: The quick brown fox jumps over 1,234,567,890 lazy"
    ("Italic: The quick brown fox jumps over 1,234,567,890 lazy", &[
        &[14, 26],
        &[20],
        &[65],
        &[34],
        &[14, 32],
        &[],
        &[35],
        &[46, 65, 87],
        &[19, 52],
        &[28, 45],
    ]),
    // 9: SourceSerif4-400Italic "Font: Originally for IBM Executive typewriters — 12 characte"
    ("Font: Originally for IBM Executive typewriters — 12 characters per inch", &[
        &[21, 78],
        &[53, 151],
        &[19],
        &[14, 36],
        &[22, 128],
        &[11, 30, 178],
        &[],
        &[],
        &[75, 159],
        &[],
        &[],
    ]),
    // 10: Georgia-400 "Matthew Carter created Georgia in 1993."
    ("Matthew Carter created Georgia in 1993.", &[
        &[53, 66, 79, 119],
        &[58],
        &[32, 67],
        &[79, 99],
        &[],
        &[],
    ]),
    // 11: Arial-ItalicMT-400Italic "Italic: The quick brown fox jumps over 1,234,567,890"
    ("Italic: The quick brown fox jumps over 1,234,567,890", &[
        &[9, 20, 49, 57],
        &[20],
        &[40, 50, 68],
        &[33, 81],
        &[9, 30],
        &[14, 64],
        &[38],
        &[162],
    ]),
    // 12: Nimbus-400Italic "AVAST! WAYWARD YOUTH TO YONDER VALLEY, AWAY AT DAWN. Ryan Tr"
    ("AVAST! WAYWARD YOUTH TO YONDER VALLEY, AWAY AT DAWN. Ryan Tries.", &[
        &[41, 107],
        &[25, 73, 100, 123],
        &[17, 72, 93],
        &[19],
        &[17, 43, 69, 96, 118],
        &[16, 63, 84, 125],
        &[50],
        &[],
        &[28, 80, 106],
        &[22, 59],
        &[16, 31, 57],
    ]),
    // 13: SourceSerif4-400Italic "Font: fonts.google.com/specimen/Jost — OFL, Owen Earl"
    ("Font: fonts.google.com/specimen/Jost — OFL, Owen Earl", &[
        &[21, 78],
        &[19, 76],
        &[21, 60, 191, 203, 217, 360, 368, 387, 406],
        &[],
        &[23, 45, 66],
        &[],
        &[],
    ]),
    // 14: LibreBodoni-400 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", &[
        &[24, 186, 216, 229, 276, 332, 360, 471, 545, 569, 604, 632, 656],
    ]),
    // 15: SourceSerif4-400Italic "Hunt for Adobe"
    ("Hunt for Adobe", &[
        &[30],
        &[19],
        &[],
    ]),
    // 16: Georgia-700 "1993 — Georgia (Microsoft)"
    ("1993 — Georgia (Microsoft)", &[
        &[],
        &[],
        &[29, 94, 116],
        &[34, 86, 202],
    ]),
    // 17: LibreCaslonText-400 "saying: \"When in doubt, use Caslon.\""
    ("saying: \"When in doubt, use Caslon.\"", &[
        &[31, 54, 89],
        &[53],
        &[],
        &[],
        &[],
        &[128],
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

    for (i, (entry, &(expected_text, expected_word_splits))) in entries.iter().zip(EXPECTED.iter()).enumerate() {
        // Use gt_text when available (robust against pflda false corrections like x->w),
        // fallback to display text.
        let text = entry.get("gt_text").and_then(|v| v.as_str())
            .or_else(|| entry.get("text").and_then(|v| v.as_str()))
            .unwrap_or("");
        assert_eq!(text, expected_text, "line {i}: text mismatch (gt_text vs expected)");

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
