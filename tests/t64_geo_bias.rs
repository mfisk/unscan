//! t64: Geometry bias + all-chars GT jitter regression.
//!
//! Verifies GT midpoint jitter stays unbiased and RMS close to theory,
//! using the **all-chars GT path** that must be used for jitter stats:
//!   `UNPRINT_AUDIT_ALL_CHARS=1` includes 2-letter words (`is`/`in`/`or`) and
//!   punctuation filtered by `is_supported()`; keeps `len<=1` only (segment.rs:1364)
//!   and `need_any` gate `len>2` (segment.rs:1376). This is the path
//!   `tools/gen_hist_flat_top.py` now uses — GT-only, no fallback to chosen.
//!
//! Theory:
//!   sigma_center = 1/√12 ≈0.2887, sigma_pitch = 1/√6≈0.4082
//!   tuned from flat-top sweep: SIGMA_CENTER_PX=0.284 SIGMA_PITCH_PX=0.435 a=0.45
//!   quantized_ll(e,σ,a)=ln[Φ((e+a)/σ)-Φ((e-a)/σ)]-ln(2a) via libm::erf
//!   (crates/unprint-geometry/src/params.rs)
//!
//! Procedure:
//!   - gen-line-test.py --hardcoded (same HARDCODED as t59)
//!   - run unprint with UNPRINT_AUDIT_ALL_CHARS=1,
//!     --test GT --audit (no UNPRINT_EXTRA_SEAMS — not debugging seams)
//!   - collect gt_geo_h_err / gt_geo_v_err for ocr_correct==True only
//!   - assert mean≈0 within 2σ, RMS≤2.0, sum_h per word==0 (center-span)
//!   - filtered |err|<1.5 gives sv/sh; assert sv=0.284±0.15 sh=0.435±0.20
//!   - GT coverage pct_v>75% and total_obs≥200 proves all-chars path exercised
//!   - No fallback to chosen_geo — if gt_geo missing, skip obs (prevents pollution)
//!
//! Run: cargo test --test t64_geo_bias -- --nocapture

mod common;

use common::unscan_bin;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn geo_bias_is_zero() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let gen_status = Command::new("python3")
        .arg(repo.join("test-docs/gen-line-test.py"))
        .arg("--hardcoded")
        .current_dir(&repo)
        .status()
        .expect("gen-line-test.py launch failed");
    assert!(gen_status.success(), "gen-line-test.py --hardcoded failed");

    let gt_src = repo.join("test-docs/line-test-gt.pdf");
    let gt_dst = repo.join("test-docs/line-test-seams-gt.pdf");
    let raster_src = repo.join("test-docs/line-test.pdf");
    let raster_dst = repo.join("test-docs/line-test-seams.pdf");
    // Prev bug: *-seams.pdf were symlinks to line-test.pdf/gt -> std::fs::copy truncated source to 0 bytes (self-copy via symlink).
    // Remove dest symlink/file first.
    let _ = std::fs::remove_file(&gt_dst);
    let _ = std::fs::remove_file(&raster_dst);
    std::fs::copy(&gt_src, &gt_dst).expect("copy gt");
    std::fs::copy(&raster_src, &raster_dst).expect("copy raster");

    let _ = std::fs::remove_dir_all("/tmp/unprint-page-cache/line-test-seams");
    // TMPDIR is ~/workspace/tmp in pueue - test originally cleared wrong /tmp path
    let _ = std::fs::remove_dir_all("/home/hatch/workspace/tmp/unprint-page-cache/line-test-seams");
    // also clear any hashed variants
    for entry in std::fs::read_dir("/home/hatch/workspace/tmp/unprint-page-cache").unwrap_or_else(|_| std::fs::read_dir("/tmp").unwrap()) {
        // best-effort cleanup of stale line-test-seams caches that cause false hits
        if let Ok(e) = entry {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("line-test-seams") {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }

    let audit_dir = repo.join("test-docs/t64-audit");
    let _ = std::fs::remove_dir_all(&audit_dir);
    let _ = std::fs::create_dir_all(&audit_dir);

    let input = repo.join("test-docs/line-test-seams.pdf");
    let gt = repo.join("test-docs/line-test-seams-gt.pdf");
    assert!(input.exists());
    assert!(gt.exists());

    let bin = unscan_bin();

    // All-chars GT + all-lines: env = audit all chars (punct, 2-letter words),
    // CLI --audit-all-lines = audit all lines / hits in obs_votes + geo for t64.
    // Both are required per 43aad2c: Histogram BAP needs
    //   UNPRINT_AUDIT_ALL_CHARS=1 ... --audit-all-lines
    // Do NOT set UNPRINT_EXTRA_SEAMS — we are not debugging seam splits.
    // Do NOT set UNPRINT_FLAT_TOP — use params.rs defaults (SIGMA_CENTER_PX=0.284 etc).
    let output = Command::new(&bin)
        .arg(&input)
        .args(["-o", "/dev/null"])
        .args(["--test", gt.to_str().unwrap()])
        .args(["--audit", audit_dir.to_str().unwrap()])
        .args(["--audit-all-lines"])
        .env("RUST_LOG", "info")
        .env("RAYON_NUM_THREADS", "1")
        .env("MALLOC_ARENA_MAX", "1")
        .env("TMPDIR", "/home/hatch/workspace/tmp")
        .env("UNPRINT_CACHE_DIR", "/home/hatch/.cache/unprint")
        .env("UNPRINT_AUDIT_ALL_CHARS", "1")
        .env("UNPRINT_SKIP_OCR_CORRECTION", "true")
        .output()
        .expect("run unprint");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "unprint failed {:?}\n{}",
        output.status.code(),
        stderr
    );

    let audit_path = audit_dir.join("audit.json");
    assert!(audit_path.exists(), "audit.json missing");

    let audit: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&audit_path).unwrap()).unwrap();
    let entries = audit["text_entries"].as_array().expect("no text_entries");

    let mut hs: Vec<f64> = Vec::new();
    let mut vs: Vec<f64> = Vec::new();
    let mut per_word_sums: Vec<f64> = Vec::new();
    let mut total_obs: usize = 0;
    let mut gt_v_count: usize = 0;

    for entry in entries {
        let ocr_correct = entry["ocr_correct"].as_bool().unwrap_or(false);
        if !ocr_correct {
            continue;
        }
        let votes = match entry["obs_votes"].as_array() {
            Some(v) => v,
            None => continue,
        };
        let mut cur_word_h: Vec<f64> = Vec::new();
        let mut in_word = false;
        for v in votes {
            total_obs += 1;
            // Strict GT-only: no fallback to chosen_geo_v_err/h_err.
            let h_opt = v.get("gt_geo_h_err").and_then(|x| x.as_f64());
            let v_opt = v.get("gt_geo_v_err").and_then(|x| x.as_f64());
            if let Some(h) = h_opt {
                hs.push(h);
                if h.abs() < 1e-9 {
                    if !cur_word_h.is_empty() {
                        per_word_sums.push(cur_word_h.iter().sum());
                        cur_word_h.clear();
                    }
                    in_word = true;
                } else if in_word {
                    cur_word_h.push(h);
                }
            }
            if let Some(vv) = v_opt {
                vs.push(vv);
                gt_v_count += 1;
            }
        }
        if !cur_word_h.is_empty() {
            per_word_sums.push(cur_word_h.iter().sum());
        }
    }

    assert!(!hs.is_empty(), "no GT h samples");
    assert!(!vs.is_empty(), "no GT v samples");
    assert!(
        total_obs >= 200,
        "all-chars path not exercised: total_obs={total_obs} <200; UNPRINT_AUDIT_ALL_CHARS may be ignored"
    );
    let pct_v = 100.0 * gt_v_count as f64 / total_obs as f64;
    assert!(
        pct_v > 75.0,
        "GT v coverage {pct_v:.1}% ({gt_v_count}/{total_obs}) — GT font should have cmap for its own chars"
    );

    fn mean(a: &[f64]) -> f64 {
        a.iter().sum::<f64>() / a.len() as f64
    }
    fn rms(a: &[f64]) -> f64 {
        (a.iter().map(|x| x * x).sum::<f64>() / a.len() as f64).sqrt()
    }
    fn stddev(a: &[f64], m: f64) -> f64 {
        (a.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / a.len() as f64).sqrt()
    }
    fn min_max(a: &[f64]) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &x in a {
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
        }
        (lo, hi)
    }
    fn sd_filtered(a: &[f64]) -> (f64, usize) {
        let filt: Vec<f64> = a.iter().copied().filter(|x| x.abs() < 1.5).collect();
        let n = filt.len();
        let m = filt.iter().sum::<f64>() / n as f64;
        let sd = (filt.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64).sqrt();
        (sd, n)
    }

    let h_mean = mean(&hs);
    let h_rms = rms(&hs);
    let h_std = stddev(&hs, h_mean);
    let (h_min, h_max) = min_max(&hs);
    let v_mean = mean(&vs);
    let v_rms = rms(&vs);
    let v_std = stddev(&vs, v_mean);
    let (v_min, v_max) = min_max(&vs);
    let (sv, sv_n) = sd_filtered(&vs);
    let (sh, sh_n) = sd_filtered(&hs);

    eprintln!(
        "ALL_CHARS GT h mean {h_mean:.6} rms {h_rms:.3} std {h_std:.3} min {h_min:.3} max {h_max:.3} n={} sh(|<1.5)={sh:.4} n={sh_n}",
        hs.len()
    );
    eprintln!(
        "ALL_CHARS GT v mean {v_mean:.6} rms {v_rms:.3} std {v_std:.3} min {v_min:.3} max {v_max:.3} n={} sv(|<1.5)={sv:.4} n={sv_n} pct_v={pct_v:.1}%",
        vs.len()
    );
    eprintln!(
        "per-word sum_h (should be 0): {:?}",
        per_word_sums
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    let sigma_center_theory = 0.2886751345948129;
    let sigma_pitch_theory = 0.408248290463863;
    eprintln!("theory sigma_center={sigma_center_theory:.4} sigma_pitch={sigma_pitch_theory:.4} tuned 0.284/0.435 a=0.45");

    let h_allowed = (2.0 * h_rms / (hs.len() as f64).sqrt()).max(0.05);
    assert!(
        h_mean.abs() <= h_allowed,
        "GT h bias {h_mean:.4} > 2σ {h_allowed:.4}"
    );
    let v_allowed = (2.0 * v_rms / (vs.len() as f64).sqrt()).max(0.08);
    assert!(
        v_mean.abs() <= v_allowed,
        "GT v bias {v_mean:.4} > 2σ {v_allowed:.4}"
    );

    assert!(h_rms <= 2.0, "GT h RMS {h_rms:.3} >2.0");
    assert!(v_rms <= 2.0, "GT v RMS {v_rms:.3} >2.0");
    for (i, s) in per_word_sums.iter().enumerate() {
        assert!(s.abs() <= 1e-6, "word {i} sum_h {s:.6} !=0");
    }
    assert!(h_mean.abs() < 1.0 && v_mean.abs() < 0.5);

    // Lock tuned sigmas (params.rs SIGMA_CENTER_PX=0.284 SIGMA_PITCH_PX=0.435) via filtered sd.
    const SIGMA_V_TUNED: f64 = 0.284;
    const SIGMA_H_TUNED: f64 = 0.435;
    assert!(
        (sv - SIGMA_V_TUNED).abs() <= 0.15,
        "sv drift {sv:.4} vs {SIGMA_V_TUNED} ±0.15 — check quant_half_width_px/quantized_ll"
    );
    assert!(
        (sh - SIGMA_H_TUNED).abs() <= 0.20,
        "sh drift {sh:.4} vs {SIGMA_H_TUNED} ±0.20"
    );
}
