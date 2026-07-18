//! t58: Single-word segmentation end-to-end test.
//!
//! Runs unscan with `--audit` on a minimal PDF containing a single line
//! of "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in Source Sans 3.  Verifies:
//!   1. Audit output includes per-line diagnostics (line_summary.json, overlays, char crops)
//!   2. Segmentation produces exactly 26 character crops for 26 characters
//!
//! Run:  cargo test --release --test t58_word_segmentation -- --nocapture

mod common;

use common::{test_doc, run_unscan};

/// Recursively find all files with a given name under a directory.
fn find_files(dir: &std::path::Path, target: &str) -> Vec<std::path::PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(find_files(&path, target));
            } else if path.file_name().map(|n| n == target).unwrap_or(false) {
                results.push(path);
            }
        }
    }
    results
}

#[test]
fn single_word_segmentation_e2e() {
    let input = test_doc("t58-sourcesans3-az-rasterized.pdf");
    assert!(input.exists(), "fixture missing — run: python3 gen_t58.py in test-docs/");

    let audit_dir = std::path::PathBuf::from("/tmp/t58-diag-seg");

    // Clean previous run
    let _ = std::fs::remove_dir_all(&audit_dir);

    // Run unscan with --audit
    let output = run_unscan(&input, &[
        "--audit", audit_dir.to_str().unwrap(),
    ]);

    eprintln!("--- unscan output ---\n{}\n---", output);

    // Audit structure: audit_dir/p{page}_L{line}_{text}/
    //   - line_summary.json
    //   - crops/crop_NN_X.png
    //   - word_NNN_{text}/seg_plain/{word_crop,final_overlay,seam_overlay}.png

    // Find all line_summary.json files
    let summaries = find_files(&audit_dir, "line_summary.json");
    assert!(!summaries.is_empty(),
        "no line_summary.json found in audit output at {:?}", audit_dir);

    for summary_path in &summaries {
        let line_dir = summary_path.parent().unwrap();
        let name = line_dir.file_name().unwrap().to_string_lossy().to_string();

        let data = std::fs::read_to_string(summary_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&data).unwrap();

        let text = json["text"].as_str().unwrap_or("");
        let decision = json["decision"].as_str().unwrap_or("");
        eprintln!("  {}: text='{}' decision='{}'", name, text, decision);

        // Verify character crops exist at the line level
        let crops_dir = line_dir.join("crops");
        assert!(crops_dir.exists(), "crops/ dir missing in {:?}", line_dir);

        let crop_count = std::fs::read_dir(&crops_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .count();

        // Input is "ABCDEFGHIJKLMNOPQRSTUVWXYZ" — expect 26 char crops
        assert_eq!(crop_count, 26,
            "expected 26 char crops in {:?}, got {}", crops_dir, crop_count);

        // Verify word subdirectory exists with segmentation overlays
        let word_dirs: Vec<_> = std::fs::read_dir(line_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir() && e.file_name().to_string_lossy().starts_with("word_"))
            .collect();
        assert!(!word_dirs.is_empty(), "no word_NNN_* dirs in {:?}", line_dir);

        for wd in &word_dirs {
            let seg_dir = wd.path().join("seg_plain");
            if seg_dir.exists() {
                assert!(seg_dir.join("word_crop.png").exists(),
                    "word_crop.png missing in {:?}", seg_dir);
                assert!(seg_dir.join("final_overlay.png").exists(),
                    "final_overlay.png missing in {:?}", seg_dir);
            }
        }
    }
}
