//! t58: Single-word segmentation end-to-end test.
//!
//! Runs unscan with `--diag-seg` on a minimal PDF containing a single line
//! of "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in Source Sans 3.  Verifies:
//!   1. Diag output is produced (summary.json, overlays, char crops)
//!   2. Segmentation produces exactly 26 segments for 26 characters
//!   3. No over-segmentation from charbox fallback
//!
//! Run:  cargo test --release --test t58_word_segmentation -- --nocapture

mod common;

use common::{test_doc, run_unscan};

#[test]
fn single_word_segmentation_e2e() {
    let input = test_doc("t58-sourcesans3-az-rasterized.pdf");
    assert!(input.exists(), "fixture missing — run: python3 gen_t58.py in test-docs/");

    let diag_dir = std::path::PathBuf::from("/tmp/t58-diag-seg");

    // Clean previous run
    let _ = std::fs::remove_dir_all(&diag_dir);

    // Run unscan with --diag-seg
    let output = run_unscan(&input, &[
        "--diag-seg", diag_dir.to_str().unwrap(),
    ]);

    eprintln!("--- unscan output ---\n{}\n---", output);

    // Find the diag output — should have a p1_* line directory
    let entries: Vec<_> = std::fs::read_dir(&diag_dir)
        .expect("diag dir not created")
        .filter_map(|e| e.ok())
        .collect();

    assert!(!entries.is_empty(), "no diag output produced in {:?}", diag_dir);

    // Find word subdirectories containing summary.json
    let mut summaries = Vec::new();
    for entry in &entries {
        let path = entry.path();
        if path.is_dir() {
            // Look for word_NNN_* subdirs inside line dirs
            for sub in std::fs::read_dir(&path).into_iter().flatten().filter_map(|e| e.ok()) {
                let sp = sub.path();
                let summary_path = sp.join("summary.json");
                if summary_path.exists() {
                    let data = std::fs::read_to_string(&summary_path).unwrap();
                    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
                    summaries.push((sp.file_name().unwrap().to_string_lossy().to_string(), json));
                }
            }
        }
    }

    assert!(!summaries.is_empty(), "no summary.json found in diag output");

    // Check each word
    for (name, json) in &summaries {
        let expected = json["n_chars_expected"].as_u64().unwrap() as usize;
        let produced = json["n_segments_produced"].as_u64().unwrap() as usize;
        let vp = json["vp_splits"].as_array().unwrap().len();
        let seam = json["seam_splits"].as_array().unwrap().len();
        let cb = json["charbox_added_splits"].as_array().unwrap().len();
        let mismatch = json["mismatch"].as_bool().unwrap();

        eprintln!(
            "  {}: {} expected, {} produced (VP:{} seam:{} cb:{})",
            name, expected, produced, vp, seam, cb
        );

        assert_eq!(
            produced, expected,
            "{}: expected {} segments, got {} — over/under-segmentation! VP:{} seam:{} cb:{}",
            name, expected, produced, vp, seam, cb
        );
        assert!(!mismatch, "{}: mismatch flag set", name);
    }

    // Verify overlay images and char crops exist
    for entry in &entries {
        let path = entry.path();
        if path.is_dir() {
            for sub in std::fs::read_dir(&path).into_iter().flatten().filter_map(|e| e.ok()) {
                let sp = sub.path();
                if sp.join("summary.json").exists() {
                    assert!(sp.join("word_crop.png").exists(), "word_crop.png missing");
                    assert!(sp.join("vp_overlay.png").exists(), "vp_overlay.png missing");
                    assert!(sp.join("final_overlay.png").exists(), "final_overlay.png missing");
                    assert!(sp.join("chars").exists(), "chars/ dir missing");
                    let char_count = std::fs::read_dir(sp.join("chars"))
                        .unwrap()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
                        .count();
                    assert_eq!(char_count, 26,
                        "expected 26 char crops, got {}", char_count);
                    return; // verified the first word, done
                }
            }
        }
    }
    panic!("no word directory with overlay files found");
}
