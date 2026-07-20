//! t64: Geometry bias regression test.
//!
//! Verifies that GT h_err and v_err are unbiased (mean ≈ 0) and
//! have RMS close to theoretical pixelation limits.
//!
//! Theory:
//!   - obs_cx = (x_min + x_max)/2 from scanning word image for ink pixels <200
//!   - pred_cx = cursor + x_pla + (x0+x1)/2 where x0/x1 are glyph ink bbox,
//!     cursor = sum of advances + GPOS xAdvance, x_pla = GPOS xPlacement
//!   - Both are ink centers, so comparison is apples-to-apples.
//!   - Spacing includes: advance (hmtx), ink bbox (glyf/CFF), GPOS Single
//!     (xPlacement/xAdvance), GPOS Pair Format1/Format2 (kerning + placement),
//!     kern table fallback, plus variation handling.
//!   - Scale = obs_span / pred_span (center-span) where
//!     obs_span = last.cx - first.cx, pred_span = last.pred_cx - first.pred_cx
//!     for n>=2, fallback to height for single char. This makes sum_h = 0
//!     per word by construction.
//!   - v_err = (obs_cy - obs_word_cy) - (pred_cy - pred_word_cy) where
//!     obs_word_cy = mean(obs_cy), pred_word_cy = mean(pred_cy), so sum_v =0.
//!   - Expected RMS from uniform quantization [-0.5,0.5]:
//!     sigma_center = 1/√12 ≈ 0.2887, sigma_pitch = √(2)/√12 = 1/√6 ≈ 0.4082
//!
//! Test procedure:
//!   - Generate 7-line hardcoded test PDFs (same as t59 + lob)
//!   - Run unprint --test GT --audit
//!   - Collect gt_geo_h_err / gt_geo_v_err for ocr_correct==True
//!   - Filter outliers: |v| > 3 px (large vertical errors from descenders etc),
//!     |h - median| > 3*MAD (robust outlier rejection)
//!   - Assert mean ≈ 0 within 2σ (mean error < 2*RMS/√n)
//!   - Assert RMS ≤ 2.0 px (generous, theory ~0.3-0.5, allow hinting etc)
//!   - Assert sum_h per word ≈ 0 (center-span unbiased by construction)
//!
//! Run: cargo test --test t64_geo_bias -- --nocapture

mod common;

use common::run_unscan;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn geo_bias_is_zero() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Generate test PDFs from hardcoded fonts/strings (no audit dependency)
    let gen_status = Command::new("python3")
        .arg(repo.join("test-docs/gen-line-test.py"))
        .arg("--hardcoded")
        .current_dir(&repo)
        .status()
        .expect("failed to run gen-line-test.py");
    assert!(gen_status.success(), "gen-line-test.py --hardcoded failed");

    // Copy to expected filenames for this test (reuse seam names)
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

    let audit_dir = repo.join("test-docs/t64-audit");
    let _ = std::fs::remove_dir_all(&audit_dir);

    let input = repo.join("test-docs/line-test-seams.pdf");
    let gt = repo.join("test-docs/line-test-seams-gt.pdf");
    assert!(input.exists(), "line-test-seams.pdf missing");
    assert!(gt.exists(), "line-test-seams-gt.pdf missing");

    let _output = run_unscan(&input, &[
        "--test", gt.to_str().unwrap(),
        "--audit", audit_dir.to_str().unwrap(),
        "--audit-all",
    ]);

    // Parse audit.json for geo errors
    let audit_path = audit_dir.join("audit.json");
    assert!(audit_path.exists(), "audit.json not produced: {:?}", audit_path);

    let audit: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&audit_path).unwrap()
    ).unwrap();

    let entries = audit["text_entries"].as_array().expect("no text_entries");

    let mut hs: Vec<f64> = Vec::new();
    let mut vs: Vec<f64> = Vec::new();
    // For sum_h per word check
    let mut per_word_sums: Vec<f64> = Vec::new();

    for entry in entries {
        // Only measure bias on ocr_correct lines (Good lines = ocr_correct==True)
        // per handoff: "Good lines = ocr_correct==True only; throw outliers"
        let ocr_correct = entry["ocr_correct"].as_bool().unwrap_or(false);
        if !ocr_correct {
            continue;
        }
        let votes = match entry["obs_votes"].as_array() {
            Some(v) => v,
            None => continue,
        };
        // Group by words via h_err==0 (first char of each word)
        let mut cur_word_h: Vec<f64> = Vec::new();
        let mut in_word = false;
        for v in votes {
            let h_opt = v.get("gt_geo_h_err").and_then(|x| x.as_f64());
            let v_opt = v.get("gt_geo_v_err").and_then(|x| x.as_f64());
            if let Some(h) = h_opt {
                hs.push(h);
                // word grouping
                if h.abs() < 1e-9 {
                    // first char of word
                    if !cur_word_h.is_empty() {
                        let sum: f64 = cur_word_h.iter().sum();
                        per_word_sums.push(sum);
                        cur_word_h.clear();
                    }
                    in_word = true;
                } else if in_word {
                    cur_word_h.push(h);
                }
            }
            if let Some(vv) = v_opt {
                vs.push(vv);
            }
        }
        if !cur_word_h.is_empty() {
            let sum: f64 = cur_word_h.iter().sum();
            per_word_sums.push(sum);
        }
    }

    assert!(!hs.is_empty(), "no GT h samples (need ocr_correct True misses)");
    assert!(!vs.is_empty(), "no GT v samples");

    // Stats helper
    fn mean(arr: &[f64]) -> f64 {
        arr.iter().sum::<f64>() / arr.len() as f64
    }
    fn rms(arr: &[f64]) -> f64 {
        (arr.iter().map(|x| x*x).sum::<f64>() / arr.len() as f64).sqrt()
    }
    fn median(arr: &mut [f64]) -> f64 {
        arr.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = arr.len();
        if n % 2 == 1 { arr[n/2] } else { (arr[n/2 -1] + arr[n/2]) / 2.0 }
    }
    fn mad(arr: &[f64], med: f64) -> f64 {
        let mut devs: Vec<f64> = arr.iter().map(|x| (x-med).abs()).collect();
        devs.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = devs.len();
        let m = if n % 2 ==1 { devs[n/2] } else { (devs[n/2 -1] + devs[n/2]) /2.0 };
        if m < 1e-9 { 1.0 } else { m }
    }

    // Horizontal: outlier rejection via MAD (3*MAD)
    let mut hs_sorted = hs.clone();
    let h_med = median(&mut hs_sorted);
    let h_mad = mad(&hs, h_med);
    let h_filtered: Vec<f64> = hs.iter().copied()
        .filter(|x| (x - h_med).abs() <= 3.0 * h_mad)
        .collect();
    let h_mean = mean(&hs);
    let h_rms = rms(&hs);
    let h_f_mean = mean(&h_filtered);
    let h_f_rms = rms(&h_filtered);
    let h_n = hs.len() as f64;
    let h_f_n = h_filtered.len() as f64;

    // Vertical: simple |v|>3 filter + MAD
    let v_filtered: Vec<f64> = vs.iter().copied()
        .filter(|x| x.abs() <= 3.0)
        .collect();
    let mut vs_sorted = vs.clone();
    let v_med = median(&mut vs_sorted);
    let v_mad = mad(&vs, v_med);
    let v_filtered2: Vec<f64> = vs.iter().copied()
        .filter(|x| (x - v_med).abs() <= 5.0 * v_mad && x.abs() <= 3.0)
        .collect();
    let v_mean = mean(&vs);
    let v_rms = rms(&vs);
    let v_f_mean = mean(&v_filtered);
    let v_f_rms = rms(&v_filtered);
    let v_f2_mean = mean(&v_filtered2);
    let v_f2_rms = rms(&v_filtered2);

    eprintln!("GT h raw mean {h_mean:.6} rms {h_rms:.3} n={} filtered mean {h_f_mean:.6} rms {h_f_rms:.3} n={}", hs.len(), h_filtered.len());
    eprintln!("GT v raw mean {v_mean:.6} rms {v_rms:.3} n={} filtered |v|<=3 mean {v_f_mean:.6} rms {v_f_rms:.3} n={} mad-filtered mean {v_f2_mean:.6} rms {v_f2_rms:.3} n={}", vs.len(), v_filtered.len(), v_filtered2.len());
    eprintln!("per-word sum_h (should be 0): {:?}", per_word_sums.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());

    // Theory
    let sigma_center_theory = 0.2886751345948129; // 1/√12
    let sigma_pitch_theory = 0.408248290463863;  // 1/√6
    eprintln!("theory sigma_center={sigma_center_theory:.4} sigma_pitch={sigma_pitch_theory:.4}");
    eprintln!("MLE sigma_pitch={:.4} (h_f_rms), sigma_center={:.4} (v_f2_rms or v_f_rms)", h_f_rms, v_f2_rms.min(v_f_rms));

    // Assert unbiased: mean within 2 sigma of zero
    // 2σ = 2 * RMS / sqrt(n)
    let h_mean_allowed = 2.0 * h_f_rms / h_f_n.sqrt();
    assert!(h_f_mean.abs() <= h_mean_allowed.max(0.05),
        "GT h bias not zero: mean {h_f_mean:.4} > 2σ {h_mean_allowed:.4} (raw mean {h_mean:.4}, rms {h_f_rms:.3}, n={})", h_f_n);

    let v_mean_allowed = 2.0 * v_f_rms / (v_filtered.len() as f64).sqrt();
    assert!(v_f_mean.abs() <= v_mean_allowed.max(0.08),
        "GT v bias not zero: mean {v_f_mean:.4} > 2σ {v_mean_allowed:.4} (raw mean {v_mean:.4}, mad mean {v_f2_mean:.4}, rms {v_f_rms:.3})");

    // Assert RMS reasonable (pixelation, not systematic)
    // Allow up to 2.0 px (generous), but expect close to theory ~0.3-0.6
    assert!(h_f_rms <= 2.0,
        "GT h RMS too large: {h_f_rms:.3} > 2.0 (mean {h_f_mean:.4}, n={})", h_f_n);
    assert!(v_f_rms <= 2.0,
        "GT v RMS too large: {v_f_rms:.3} > 2.0 (mean {v_f_mean:.4})");

    // Assert center-span unbiased by construction: sum_h per word == 0
    for (i, sum) in per_word_sums.iter().enumerate() {
        assert!(sum.abs() <= 1e-6,
            "word {i} sum_h not zero (center-span bug): sum={sum:.6}");
    }

    // Also check overall mean is near zero (redundant but explicit)
    assert!(h_mean.abs() < 1.0, "GT h raw mean {h_mean:.3} too large, indicates bias");
    assert!(v_f_mean.abs() < 0.5, "GT v filtered mean {v_f_mean:.3} too large");

    // ── Discriminative sigma fitting (how weights relate to sigmas) ─────
    // We have per-observation GT vs chosen geo errors. Want w_h,w_v to maximize
    // GT win rate where score = w_h*(-h_err^2) + w_v*(-v_err^2)
    // Relationship: w = 1/(2σ²), so σ = 1/√(2w). If we scale ll by α,
    // σ_eff = σ/√α — inverse square root, NOT exponential.
    // Fitting weights ≡ fitting sigmas. MLE sigmas = RMS(h), RMS(v).
    // Here do simple grid search over sigma_pitch, sigma_center to find
    // best discriminative pair that makes GT beat chosen most often.

    // Need paired GT+chosen errors; re-parse audit for paired data
    // (we already have hs/vs for GT only; now get chosen too)
    let mut paired: Vec<(f64,f64,f64,f64)> = Vec::new(); // (h_gt, v_gt, h_ch, v_ch)
    // Re-iterate entries to collect paired chosen/gt
    // (audit already loaded; do quick second pass)
    // We need audit value still in scope; reload quickly
    let audit_path = repo.join("test-docs/t64-audit/audit.json");
    let audit2: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&audit_path).unwrap()).unwrap();
    for entry in audit2["text_entries"].as_array().unwrap() {
        if entry["ocr_correct"].as_bool().unwrap_or(false) == false { continue; }
        for v in entry["obs_votes"].as_array().unwrap_or(&vec![]) {
            let hg = v.get("gt_geo_h_err").and_then(|x| x.as_f64());
            let vg = v.get("gt_geo_v_err").and_then(|x| x.as_f64());
            let hc = v.get("chosen_geo_h_err").and_then(|x| x.as_f64());
            let vc = v.get("chosen_geo_v_err").and_then(|x| x.as_f64());
            if let (Some(a), Some(b), Some(c), Some(d)) = (hg, vg, hc, vc) {
                // filter same as above: |v|<=3 and h MAD already applied? use simple
                if b.abs() <= 3.0 && d.abs() <= 3.0 {
                    paired.push((a,b,c,d));
                }
            }
        }
    }

    if !paired.is_empty() {
        let mut best_score = 0usize;
        let mut best_sp: f64 = 0.0;
        let mut best_sc: f64 = 0.0;
        // Grid 0.15..2.0 step 0.05
        let mut sigma = 0.15;
        while sigma <= 2.001 {
            let mut sigma_c = 0.15;
            while sigma_c <= 2.001 {
                let inv2p2 = 1.0/(2.0*sigma*sigma);
                let inv2c2 = 1.0/(2.0*sigma_c*sigma_c);
                let mut wins = 0;
                for (hg, vg, hc, vc) in &paired {
                    let ll_gt = -hg*hg*inv2p2 - vg*vg*inv2c2;
                    let ll_ch = -hc*hc*inv2p2 - vc*vc*inv2c2;
                    if ll_gt > ll_ch { wins += 1; }
                }
                if wins > best_score {
                    best_score = wins;
                    best_sp = sigma;
                    best_sc = sigma_c;
                }
                sigma_c += 0.05;
            }
            sigma += 0.05;
        }
        let total = paired.len();
        eprintln!("discriminative grid: best sigma_pitch={:.3} sigma_center={:.3} wins {}/{} ({:.1}%)",
            best_sp, best_sc, best_score, total, 100.0*best_score as f64/total as f64);
        let mle_wins = {
            let inv2p = 1.0/(2.0*h_f_rms*h_f_rms);
            let inv2c = 1.0/(2.0*v_f_rms*v_f_rms);
            paired.iter().filter(|(hg,vg,hc,vc)| {
                let ll_gt = -hg*hg*inv2p - vg*vg*inv2c;
                let ll_ch = -hc*hc*inv2p - vc*vc*inv2c;
                ll_gt > ll_ch
            }).count()
        };
        eprintln!("MLE sigma_pitch={:.3} sigma_center={:.3} wins {}/{} ({:.1}%)",
            h_f_rms, v_f_rms, mle_wins, total, 100.0*mle_wins as f64/total as f64);
        eprintln!("theory sigma_pitch={:.4} sigma_center={:.4}", sigma_pitch_theory, sigma_center_theory);
        eprintln!("weight relation: w=1/(2σ²), so σ=1/√(2w). Scaling ll by α → σ_eff=σ/√α (inverse sqrt, NOT exponential). Fitting weights ≡ fitting sigmas.");
        eprintln!("suggestion: set SIGMA_PITCH_PX = {:.4}, SIGMA_CENTER_PX = {:.4} (MLE) or {:.4}/{:.4} (discriminative best)", h_f_rms, v_f_rms, best_sp, best_sc);
    }
}
