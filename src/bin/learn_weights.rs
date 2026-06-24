/// Fisher discriminant weight analysis from exhaustive multi-DPI rendering.
///
/// For every (font × character × DPI) triple, renders the glyph, normalises
/// through the same pipeline as the character index, and computes features.
/// Identical feature vectors are deduplicated per character.
///
/// Signal = between-font variance (per char, pooled across DPIs)
/// Noise  = within-font across-DPI variance (per font×char)
/// Fisher = signal / noise per feature dimension
///
/// Usage:
///   learn_weights                           Run full analysis (all fonts × chars × DPIs)
///   learn_weights --dpis 300,200,100        Custom DPI set (default: 300,200,100)

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use unscan::char_index::{self, compute_features, FEAT_LEN};
use unscan::font_scan;

const FEAT_NAMES: &[&str] = &[
    // Group 1: Column ink profile (16)
    "col0","col1","col2","col3","col4","col5","col6","col7",
    "col8","col9","col10","col11","col12","col13","col14","col15",
    // Group 2: Scalar v1 (7)
    "aspect","ink_density","v_center","h_balance","serif_score","stroke_contrast","xh_cap_ratio",
    // Group 3: Scalar v2 (14)
    "counter_area","counter_cx","counter_cy","counter_asp",
    "term0","term1","term2","term3",
    "ink_perim","compactness",
    "cross0","cross1","cross2","cross3",
    // Group 4: Row ink profile (16)
    "row0","row1","row2","row3","row4","row5","row6","row7",
    "row8","row9","row10","row11","row12","row13","row14","row15",
    // Group 5: Scalar v3 (11)
    "hole_count","h_symmetry","v_symmetry","skel_branch","skel_endpt",
    "corner_count","quad_tl","quad_tr","quad_bl","quad_br","mean_stroke_w",
];

/// Render a character at a given simulated DPI.
///
/// The approach mirrors the real pipeline:
/// - Render at full quality using ab_glyph (same as index)
/// - Downscale to simulate capture at `dpi` (relative to 300 DPI reference)
/// - Run through `normalize_to_ink_bounds` (same as scan-side crop normalization)
///
/// At 300 DPI this is identical to the index render.  At lower DPIs, the
/// downscale→upscale cycle loses detail just like a real lower-resolution scan.
fn render_char_at_dpi<F: Font>(font: &F, c: char, dpi: u32) -> Option<GrayImage> {
    // Render at full quality (same as render_char_normalised)
    let full = char_index::render_char_normalised(font, c)?;

    if dpi >= 300 {
        // At reference DPI, return as-is (already normalized)
        return Some(full);
    }

    // Downscale to simulate lower DPI, then normalize back up
    let scale_factor = dpi as f32 / 300.0;
    let (w, h) = full.dimensions();
    let small_w = ((w as f32 * scale_factor).round() as u32).max(3);
    let small_h = ((h as f32 * scale_factor).round() as u32).max(3);

    // Downscale (simulates lower-resolution capture)
    let small = image::imageops::resize(
        &full,
        small_w,
        small_h,
        image::imageops::FilterType::Lanczos3,
    );

    // Normalize back through the same path as scan crops
    char_index::normalize_to_ink_bounds(&small)
}

/// One (font, char, dpi) → feature vector result.
struct RenderResult {
    font_name: String,
    ch: char,
    dpi: u32,
    features: [f32; FEAT_LEN],
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --dpis flag
    let dpis: Vec<u32> = args.iter()
        .position(|a| a == "--dpis")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').filter_map(|d| d.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![300, 200, 100]);

    eprintln!("DPIs: {:?}", dpis);

    // ── 1. Scan all system fonts ────────────────────────────────────
    let font_dirs = font_scan::default_font_dirs(&[]);
    eprintln!("Scanning fonts...");
    let catalog = font_scan::scan_fonts(&font_dirs);
    eprintln!("  {} font entries", catalog.len());

    let chars: &[char] = char_index::indexed_chars();
    eprintln!("  {} indexed characters", chars.len());
    eprintln!("  {} DPIs × {} fonts × {} chars = {} renders",
        dpis.len(), catalog.len(), chars.len(),
        dpis.len() * catalog.len() * chars.len());

    // ── 2. Render all (font × char × dpi) in parallel ──────────────
    // Build (font_name, font_path) pairs — same as index build
    let font_pairs: Vec<(String, PathBuf)> = catalog.iter()
        .map(|e| (e.font_key(), e.path.clone()))
        .collect();

    eprintln!("Rendering features...");
    let start = std::time::Instant::now();

    // Each work item is (font_idx, dpi)
    let work: Vec<(usize, u32)> = (0..font_pairs.len())
        .flat_map(|fi| dpis.iter().map(move |&d| (fi, d)))
        .collect();

    let results: Vec<Vec<RenderResult>> = work.par_iter().map(|&(fi, dpi)| {
        let (ref font_name, ref font_path) = font_pairs[fi];
        let font_data = match std::fs::read(font_path) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let font = match FontRef::try_from_slice(&font_data) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        let mut batch = Vec::new();
        for &c in chars {
            if let Some(img) = render_char_at_dpi(&font, c, dpi) {
                if let Some(feats) = compute_features(&img) {
                    batch.push(RenderResult {
                        font_name: font_name.clone(),
                        ch: c,
                        dpi,
                        features: feats.as_slice(),
                    });
                }
            }
        }
        batch
    }).collect();

    let all_results: Vec<RenderResult> = results.into_iter().flatten().collect();
    let elapsed = start.elapsed();
    eprintln!("  {} feature vectors in {:.1}s", all_results.len(), elapsed.as_secs_f64());

    // ── 2b. Optional CSV dump ──────────────────────────────────────
    if let Some(csv_path) = args.iter()
        .position(|a| a == "--dump-csv")
        .and_then(|i| args.get(i + 1))
    {
        eprintln!("Dumping CSV to {}...", csv_path);
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(csv_path).expect("create CSV"));
        // Header
        write!(f, "font_name,char,dpi").unwrap();
        for name in FEAT_NAMES {
            write!(f, ",{}", name).unwrap();
        }
        writeln!(f).unwrap();
        // Rows
        for r in &all_results {
            write!(f, "{},{},{}", r.font_name, r.ch as u32, r.dpi).unwrap();
            for &v in r.features.iter() {
                write!(f, ",{:.8}", v).unwrap();
            }
            writeln!(f).unwrap();
        }
        eprintln!("  CSV written ({} rows)", all_results.len());
    }

    // ── 3. Collect per-character feature vectors across all DPIs ──────
    eprintln!("Collecting feature vectors...");

    // Group by (char, dpi)
    let mut by_char_dpi: HashMap<(char, u32), Vec<&RenderResult>> = HashMap::new();
    for r in &all_results {
        by_char_dpi.entry((r.ch, r.dpi)).or_default().push(r);
    }

    // ── 4. Compute signal: between-font variance per feature ────────
    // For each character, pool all feature vectors across all DPIs,
    // compute variance across fonts.
    eprintln!("Computing between-font variance (signal)...");
    let unique_chars: Vec<char> = {
        let mut cs: Vec<char> = by_char_dpi.keys().map(|(c, _)| *c).collect::<HashSet<_>>().into_iter().collect();
        cs.sort();
        cs
    };

    let mut signal = [0.0f64; FEAT_LEN];
    let mut signal_n = 0usize;
    for &ch in &unique_chars {
        // Collect all feature vectors for this char across all DPIs
        let mut all_feats: Vec<[f32; FEAT_LEN]> = Vec::new();
        for &dpi in &dpis {
            if let Some(entries) = by_char_dpi.get(&(ch, dpi)) {
                for r in entries {
                    all_feats.push(r.features);
                }
            }
        }
        if all_feats.len() < 10 {
            continue;
        }
        // Variance across fonts
        let n = all_feats.len() as f64;
        let mut means = [0.0f64; FEAT_LEN];
        for f in &all_feats {
            for d in 0..FEAT_LEN {
                means[d] += f[d] as f64;
            }
        }
        for d in 0..FEAT_LEN {
            means[d] /= n;
        }
        let mut var = [0.0f64; FEAT_LEN];
        for f in &all_feats {
            for d in 0..FEAT_LEN {
                let diff = f[d] as f64 - means[d];
                var[d] += diff * diff;
            }
        }
        for d in 0..FEAT_LEN {
            signal[d] += var[d] / n;
        }
        signal_n += 1;
    }
    for d in 0..FEAT_LEN {
        signal[d] /= signal_n.max(1) as f64;
    }
    eprintln!("  {} chars with >=10 vectors", signal_n);

    // ── 5. Compute noise: within-font across-DPI variance ───────────
    // For each (font, char), compute variance of features across DPIs.
    // This measures how much resolution changes the features for the
    // same glyph — features unstable across DPI get high noise.
    eprintln!("Computing across-DPI variance (noise)...");
    let mut by_font_char: HashMap<(&str, char), Vec<&RenderResult>> = HashMap::new();
    for r in &all_results {
        by_font_char.entry((r.font_name.as_str(), r.ch)).or_default().push(r);
    }

    let mut noise = [0.0f64; FEAT_LEN];
    let mut noise_n = 0usize;
    for ((_, _), entries) in &by_font_char {
        if entries.len() < 2 {
            continue; // need at least 2 DPIs to compute variance
        }
        let n = entries.len() as f64;
        let mut means = [0.0f64; FEAT_LEN];
        for r in entries {
            for d in 0..FEAT_LEN {
                means[d] += r.features[d] as f64;
            }
        }
        for d in 0..FEAT_LEN {
            means[d] /= n;
        }
        let mut var = [0.0f64; FEAT_LEN];
        for r in entries {
            for d in 0..FEAT_LEN {
                let diff = r.features[d] as f64 - means[d];
                var[d] += diff * diff;
            }
        }
        for d in 0..FEAT_LEN {
            noise[d] += var[d] / n;
        }
        noise_n += 1;
    }
    for d in 0..FEAT_LEN {
        noise[d] /= noise_n.max(1) as f64;
    }
    eprintln!("  {} (font, char) pairs with 2+ DPI entries", noise_n);

    // ── 6. Fisher ratio and optimal weights ─────────────────────────
    // Two weight variants:
    //  A) sqrt(fisher) normalized — assumes features are pre-standardized
    //  B) sqrt(fisher)/std_total  — corrects for raw feature magnitude
    // Variant B is needed when applying weights directly to raw features
    // (no L2 group normalization).
    let mut fisher = [0.0f64; FEAT_LEN];
    let mut raw_weights = [0.0f64; FEAT_LEN];
    let mut scale_adj_weights = [0.0f64; FEAT_LEN];
    let mut weight_sum = 0.0f64;
    let mut scale_adj_sum = 0.0f64;
    for d in 0..FEAT_LEN {
        fisher[d] = signal[d] / (noise[d] + 1e-12);
        raw_weights[d] = fisher[d].max(0.0).sqrt();
        weight_sum += raw_weights[d];
        let std_total = (signal[d] + noise[d]).sqrt();
        if std_total > 1e-12 {
            scale_adj_weights[d] = raw_weights[d] / std_total;
        }
        scale_adj_sum += scale_adj_weights[d];
    }
    let mut norm_weights = [0.0f64; FEAT_LEN];
    let mut norm_scale_adj = [0.0f64; FEAT_LEN];
    for d in 0..FEAT_LEN {
        norm_weights[d] = raw_weights[d] / (weight_sum + 1e-12);
        norm_scale_adj[d] = scale_adj_weights[d] / (scale_adj_sum + 1e-12);
    }

    // ── 7. Output ───────────────────────────────────────────────────
    println!("={:=>94}", "");
    println!("FISHER DISCRIMINANT ANALYSIS — MULTI-DPI ({:?})", dpis);
    println!("={:=>94}", "");
    println!("\n{:>4} {:>4} {:>16} {:>12} {:>12} {:>12} {:>8}",
        "rank", "dim", "name", "signal", "noise", "fisher", "opt_wt");
    println!("{:-<82}", "");

    let mut order: Vec<usize> = (0..FEAT_LEN).collect();
    order.sort_by(|a, b| fisher[*b].partial_cmp(&fisher[*a]).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, &i) in order.iter().enumerate() {
        let group = if i < 16 { "col_prof" }
            else if i < 23 { "scal_v1" }
            else if i < 37 { "scal_v2" }
            else if i < 53 { "row_prof" }
            else { "scal_v3" };
        println!("{:>4} {:>4} {:>16} {:>12.6} {:>12.6} {:>12.2} {:>8.4}  {}",
            rank + 1, i, FEAT_NAMES[i], signal[i], noise[i], fisher[i], norm_weights[i], group);
    }

    // Group totals
    let col_prof_wt: f64 = norm_weights[..16].iter().sum();
    let scal_v1_wt: f64 = norm_weights[16..23].iter().sum();
    let scal_v2_wt: f64 = norm_weights[23..37].iter().sum();
    let row_prof_wt: f64 = norm_weights[37..53].iter().sum();
    let scal_v3_wt: f64 = norm_weights[53..].iter().sum();
    println!("\n={:=>94}", "");
    println!("GROUP WEIGHTS — CURRENT vs OPTIMAL");
    println!("={:=>94}", "");
    println!("  {:30}  current: 0.4000  optimal: {:.4}", "Col profile (16 dims)", col_prof_wt);
    println!("  {:30}  current: 0.3000  optimal: {:.4}", "Scalar v1 (7 dims)", scal_v1_wt);
    println!("  {:30}  current: 0.3000  optimal: {:.4}", "Scalar v2 (14 dims)", scal_v2_wt);
    println!("  {:30}  current: 0.3000  optimal: {:.4}", "Row profile (16 dims)", row_prof_wt);
    println!("  {:30}  current: 0.2000  optimal: {:.4}", "Scalar v3 (11 dims)", scal_v3_wt);

    // Scalar detail — all scalar groups
    println!("\n={:=>94}", "");
    println!("SCALAR FEATURES — v1");
    println!("={:=>94}", "");
    for j in 0..7 {
        let i = 16 + j;
        println!("  {:>16}  signal={:.6}  noise={:.6}  fisher={:.2}  wt={:.4}",
            FEAT_NAMES[i], signal[i], noise[i], fisher[i], norm_weights[i]);
    }
    println!("\n={:=>94}", "");
    println!("SCALAR FEATURES — v2");
    println!("={:=>94}", "");
    for j in 0..14 {
        let i = 23 + j;
        println!("  {:>16}  signal={:.6}  noise={:.6}  fisher={:.2}  wt={:.4}",
            FEAT_NAMES[i], signal[i], noise[i], fisher[i], norm_weights[i]);
    }
    println!("\n={:=>94}", "");
    println!("SCALAR FEATURES — v3");
    println!("={:=>94}", "");
    for j in 0..11 {
        let i = 53 + j;
        println!("  {:>16}  signal={:.6}  noise={:.6}  fisher={:.2}  wt={:.4}",
            FEAT_NAMES[i], signal[i], noise[i], fisher[i], norm_weights[i]);
    }

    // Rust array — scale-adjusted (for direct application to raw features)
    println!("\n={:=>94}", "");
    println!("RUST WEIGHTS ARRAY — SCALE-ADJUSTED (no group L2 normalization needed)");
    println!("={:=>94}", "");
    println!("const FISHER_WEIGHTS: [f32; FEAT_LEN] = [");
    for (start, end, label) in [
        (0, 16, "Col profile"),
        (16, 23, "Scalar v1"),
        (23, 37, "Scalar v2"),
        (37, 53, "Row profile"),
        (53, 64, "Scalar v3"),
    ] {
        let vals: Vec<String> = (start..end).map(|i| format!("{:.6}", norm_scale_adj[i])).collect();
        println!("    // {}", label);
        println!("    {},", vals.join(", "));
    }
    println!("];");
}
