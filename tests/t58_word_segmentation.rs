//! t58: Single-word segmentation end-to-end test.
//!
//! Runs unscan with `--audit` on a minimal PDF containing a single line
//! of "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in Source Sans 3.  Verifies:
//!   1. Audit output includes segmentation diagnostics (summary.json, overlays, char crops)
//!   2. Segmentation produces exactly 26 segments for 26 characters
//!   3. No over-segmentation from charbox fallback
//!
//! Run:  cargo test --release --test t58_word_segmentation -- --nocapture

mod common;

use common::{test_doc, run_unscan};

/// Recursively find all `summary.json` files under a directory.
fn find_summaries(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(find_summaries(&path));
            } else if path.file_name().map(|n| n == "summary.json").unwrap_or(false) {
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

    // Run unscan with --audit (replaces the old --diag-seg flag)
    let output = run_unscan(&input, &[
        "--audit", audit_dir.to_str().unwrap(),
    ]);

    eprintln!("--- unscan output ---\n{}\n---", output);

    // Find all summary.json files recursively (audit nests them under
    // line_dir/word_dir/seg_variant/)
    let summaries = find_summaries(&audit_dir);
    assert!(!summaries.is_empty(), "no summary.json found in audit output at {:?}", audit_dir);

    // Check each word's segmentation
    for summary_path in &summaries {
        let seg_dir = summary_path.parent().unwrap();
        let name = seg_dir.file_name().unwrap().to_string_lossy().to_string();

        let data = std::fs::read_to_string(summary_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&data).unwrap();

        let expected = json["n_chars_expected"].as_u64().unwrap() as usize;
        let produced = json["n_segments_produced"].as_u64().unwrap() as usize;
        let seam = json.get("seam_splits")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let cb = json.get("charbox_added_splits")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let mismatch = json["mismatch"].as_bool().unwrap();

        // VP splits may or may not be present depending on segmentation path
        let vp = json.get("vp_splits")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

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

        // Verify overlay images and char crops exist in the same directory
        assert!(seg_dir.join("word_crop.png").exists(),
            "word_crop.png missing in {:?}", seg_dir);
        assert!(seg_dir.join("final_overlay.png").exists(),
            "final_overlay.png missing in {:?}", seg_dir);
        assert!(seg_dir.join("chars").exists(),
            "chars/ dir missing in {:?}", seg_dir);

        let char_count = std::fs::read_dir(seg_dir.join("chars"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .count();
        assert_eq!(char_count, 26,
            "expected 26 char crops, got {}", char_count);
    }
}
