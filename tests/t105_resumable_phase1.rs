//! t105: Phase 1 resume — interrupt a bounded cold run mid-Phase-1, resume
//! it, and verify byte-identical outputs vs an uninterrupted run on the
//! same inputs (perf proposal 4.4).
//!
//! Bounded training runs (max_fonts=100) in FRESH TEMP cache dirs via
//! UNPRINT_CACHE_DIR — never the real ~/.cache/unprint.
//!
//! The interruption uses the UNPRINT_TRAIN_ABORT_AFTER_FONTS fault-injection
//! hook (test-only): run_train panics at the next chunk boundary once N
//! fonts are done, leaving tmp feature files + the phase1 checkpoint on
//! disk with no manifests — the same on-disk state a SIGKILL/reboot
//! leaves. The panic unwinds after the previous chunk's flush +
//! checkpoint, so all state is consistent. The test catches it with
//! catch_unwind, then calls run_train again (resume).
//!
//! Three assertions:
//!   1. Resume actually skips the completed fonts: after aborting at 60,
//!      a second run with the same abort point must leave every tmp file
//!      byte-identical (a restart would truncate + rewrite them).
//!   2. The resumed run's outputs (feature files, glyph-map.bin, LDA
//!      weights) are byte-identical to an uninterrupted control run.
//!   3. A corrupted checkpoint is discarded safely: truncating a tmp file
//!      after the abort forces a full restart, and the final outputs are
//!      still byte-identical to the control run.
//!
//! Run single-threaded: the test switches UNPRINT_CACHE_DIR (process-global)
//! between phases. `cargo test --release -p unprint --test t105_resumable_phase1 -- --test-threads=1`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MAX_FONTS: usize = 100;
const ABORT_AFTER: usize = 60; // chunk boundary (chunk_size = 30): 2 chunks done

fn train_args() -> unprint::train::TrainArgs {
    unprint::train::TrainArgs {
        max_fonts: MAX_FONTS,
        min_fonts: 2,
        lda: true,
        ..Default::default()
    }
}

fn run_train_expect_abort(geo: &unprint::geo_cache::GeometryCache) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unprint::train::run_train(train_args(), geo);
    }));
    match result {
        Ok(()) => panic!("run_train should have aborted (UNPRINT_TRAIN_ABORT_AFTER_FONTS), but it completed"),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic>");
            assert!(
                msg.contains("UNPRINT_TRAIN_ABORT_AFTER_FONTS"),
                "unexpected panic (expected the abort hook): {msg}"
            );
        }
    }
}

fn run_train_expect_ok(geo: &unprint::geo_cache::GeometryCache) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unprint::train::run_train(train_args(), geo);
    }));
    if let Err(payload) = result {
        let msg = payload
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        panic!("run_train panicked unexpectedly: {msg}");
    }
}

/// (relative path -> file bytes) for every file under `dir`, recursively.
fn snapshot_dir(dir: &Path) -> HashMap<PathBuf, Vec<u8>> {
    let mut out = HashMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {}: {e}", d.display())) {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(dir).unwrap().to_path_buf();
                let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
                out.insert(rel, bytes);
            }
        }
    }
    out
}

fn assert_file_sets_identical(a: &HashMap<PathBuf, Vec<u8>>, b: &HashMap<PathBuf, Vec<u8>>, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: file count differs ({} vs {})", a.len(), b.len());
    for (rel, bytes_a) in a {
        match b.get(rel) {
            Some(bytes_b) => assert_eq!(
                bytes_a, bytes_b,
                "{label}: file differs: {} ({} vs {} bytes)",
                rel.display(),
                bytes_a.len(),
                bytes_b.len()
            ),
            None => panic!("{label}: file missing from second dir: {}", rel.display()),
        }
    }
}

/// Compare the training outputs that define the training result:
/// feature files (+ manifests), the glyph map, and the LDA weights.
fn assert_training_outputs_identical(dir_a: &Path, dir_b: &Path, label: &str) {
    let feats_a = snapshot_dir(&dir_a.join("glyph-features"));
    let feats_b = snapshot_dir(&dir_b.join("glyph-features"));
    assert_file_sets_identical(&feats_a, &feats_b, &format!("{label} glyph-features"));
    for name in ["glyph-map.bin", "lda-weights-global.bin"] {
        let a = std::fs::read(dir_a.join(name)).unwrap_or_else(|e| panic!("{label}: read {name} in A: {e}"));
        let b = std::fs::read(dir_b.join(name)).unwrap_or_else(|e| panic!("{label}: read {name} in B: {e}"));
        assert_eq!(a, b, "{label}: {name} differs ({} vs {} bytes)", a.len(), b.len());
    }
    eprintln!("[t105] {label}: identical ({} feature files + glyph map + LDA weights)", feats_a.len());
}

fn fresh_cache_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from("/home/hatch/workspace/tmp").join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove stale test cache dir");
    }
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn tmp_files(feat_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(feat_dir).expect("read feat dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_some_and(|e| e == "tmp") {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn checkpoint_fonts_done(feat_dir: &Path) -> usize {
    let json = std::fs::read_to_string(feat_dir.join("phase1_checkpoint.json"))
        .expect("checkpoint file must exist");
    let v: serde_json::Value = serde_json::from_str(&json).expect("checkpoint must be valid JSON");
    v["fonts_done"].as_u64().expect("fonts_done must be a number") as usize
}

#[test]
fn phase1_resume_byte_identical() {
    let dir_resume = fresh_cache_dir("unprint-cache-perf44-resume");
    let dir_control = fresh_cache_dir("unprint-cache-perf44-control");
    let dir_corrupt = fresh_cache_dir("unprint-cache-perf44-corrupt");

    // Geo cache is built once (full catalog) and reused in-memory for all runs.
    std::env::set_var("UNPRINT_CACHE_DIR", &dir_resume);
    let geo_cache = unprint::train::load_geo_cache_for_training(&[], true);
    eprintln!("[t105] geo cache built");

    // ── Run 1: abort at 60 fonts ──
    std::env::set_var("UNPRINT_TRAIN_ABORT_AFTER_FONTS", ABORT_AFTER.to_string());
    run_train_expect_abort(&geo_cache);
    std::env::remove_var("UNPRINT_TRAIN_ABORT_AFTER_FONTS");
    let feat_resume = dir_resume.join("glyph-features");
    assert_eq!(
        checkpoint_fonts_done(&feat_resume),
        ABORT_AFTER,
        "checkpoint must record 60 fonts done after the abort"
    );
    assert!(
        tmp_files(&feat_resume).len() > 100,
        "tmp feature files must exist after the abort"
    );
    let has_manifest = std::fs::read_dir(&feat_resume)
        .expect("read feat dir")
        .any(|e| {
            e.map(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("manifest_") && n.ends_with(".txt"))
            })
            .unwrap_or(false)
        });
    assert!(!has_manifest, "no manifest may exist after an interrupted Phase 1");
    eprintln!("[t105] run 1 aborted at 60; checkpoint + tmp files on disk, no manifests");

    // Snapshot the tmp files: a resume must not touch them until it
    // reaches font 60 again, while a restart truncates + rewrites them.
    let tmp_snapshot = snapshot_dir(&feat_resume);

    // ── Run 2: abort at the SAME point — must resume and abort immediately ──
    std::env::set_var("UNPRINT_TRAIN_ABORT_AFTER_FONTS", ABORT_AFTER.to_string());
    run_train_expect_abort(&geo_cache);
    std::env::remove_var("UNPRINT_TRAIN_ABORT_AFTER_FONTS");
    let tmp_after = snapshot_dir(&feat_resume);
    assert_file_sets_identical(
        &tmp_snapshot,
        &tmp_after,
        "resume at the abort point must leave tmp files untouched",
    );
    eprintln!("[t105] run 2 resumed from 60 and aborted immediately; tmp files untouched — resume path proven");

    // ── Run 3: resume to completion ──
    run_train_expect_ok(&geo_cache);
    assert!(
        !feat_resume.join("phase1_checkpoint.json").exists(),
        "checkpoint must be deleted when Phase 1 completes"
    );
    assert!(
        tmp_files(&feat_resume).is_empty(),
        "no tmp files may remain after Phase 1 completes"
    );
    eprintln!("[t105] run 3 resumed to completion");

    // ── Run 4: uninterrupted control ──
    std::env::set_var("UNPRINT_CACHE_DIR", &dir_control);
    run_train_expect_ok(&geo_cache);
    eprintln!("[t105] run 4 (control) completed");
    assert_training_outputs_identical(&dir_resume, &dir_control, "resumed vs control");

    // ── Corrupt-checkpoint test: truncate a tmp file, resume must restart ──
    std::env::set_var("UNPRINT_CACHE_DIR", &dir_corrupt);
    std::env::set_var("UNPRINT_TRAIN_ABORT_AFTER_FONTS", ABORT_AFTER.to_string());
    run_train_expect_abort(&geo_cache);
    std::env::remove_var("UNPRINT_TRAIN_ABORT_AFTER_FONTS");
    let feat_corrupt = dir_corrupt.join("glyph-features");
    let mut tmps = tmp_files(&feat_corrupt);
    tmps.retain(|p| std::fs::metadata(p).map(|m| m.len() > 1000).unwrap_or(false));
    let victim = tmps.first().expect("need a non-trivial tmp file to corrupt");
    {
        let f = std::fs::File::create(victim).expect("truncate victim tmp file");
        f.set_len(10).expect("truncate to 10 bytes");
    }
    eprintln!("[t105] corrupted tmp file {}", victim.display());
    run_train_expect_ok(&geo_cache);
    assert!(
        !feat_corrupt.join("phase1_checkpoint.json").exists(),
        "checkpoint must be deleted after the corrupt-checkpoint restart completes"
    );
    assert_training_outputs_identical(&dir_corrupt, &dir_control, "corrupt-restart vs control");
    eprintln!("[t105] corrupt checkpoint discarded safely; outputs identical to control");
}
