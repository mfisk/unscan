/// Diagnostic: compare feature vectors of two character crop PNGs.
/// Usage: feat_diff <scan.png> <ref.png>
/// Both images must be normalized char crops (output of normalize_to_ink_bounds
/// or render_char_normalised).

use image::ImageReader;
use unscan::char_index::{compute_features, FEAT_LEN};

const NAMES: &[&str] = &[
    // 0..15: column profile bins (16)
    "prof[0]", "prof[1]", "prof[2]", "prof[3]", "prof[4]", "prof[5]", "prof[6]", "prof[7]",
    "prof[8]", "prof[9]", "prof[10]", "prof[11]", "prof[12]", "prof[13]", "prof[14]", "prof[15]",
    // 16..22: original scalars (7)
    "aspect", "ink_density", "v_center", "h_balance", "serif_score", "stroke_contrast", "xh_cap_ratio",
    // 23..26: counter features (4)
    "counter_area", "counter_cx", "counter_cy", "counter_asp",
    // 27..30: terminal angles (4)
    "term[0]", "term[1]", "term[2]", "term[3]",
    // 31..32: boundary (2)
    "ink_perim", "compactness",
    // 33..36: h_crossings (4)
    "cross[0]", "cross[1]", "cross[2]", "cross[3]",
    // 37..52: row profile bins (16)
    "row[0]", "row[1]", "row[2]", "row[3]", "row[4]", "row[5]", "row[6]", "row[7]",
    "row[8]", "row[9]", "row[10]", "row[11]", "row[12]", "row[13]", "row[14]", "row[15]",
    // 53..63: scalar v3 (11)
    "hole_count", "h_symmetry", "v_symmetry", "skel_branch", "skel_endpt",
    "corner_count", "quad_tl", "quad_tr", "quad_bl", "quad_br", "mean_stroke_w",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: feat_diff <scan.png> <ref.png>");
        std::process::exit(1);
    }

    let scan_img = ImageReader::open(&args[1]).unwrap().decode().unwrap().to_luma8();
    let ref_img = ImageReader::open(&args[2]).unwrap().decode().unwrap().to_luma8();

    let scan_feats = compute_features(&scan_img).expect("scan features failed");
    let ref_feats = compute_features(&ref_img).expect("ref features failed");

    let scan_w = scan_feats.weighted();
    let ref_w = ref_feats.weighted();
    let scan_raw = scan_feats.as_slice();
    let ref_raw = ref_feats.as_slice();

    assert_eq!(NAMES.len(), FEAT_LEN, "name table length mismatch");

    let mut total_sq = 0.0f32;
    let mut diffs: Vec<(usize, &str, f32, f32, f32, f32)> = Vec::new();
    for i in 0..FEAT_LEN {
        let d = scan_w[i] - ref_w[i];
        let sq = d * d;
        total_sq += sq;
        diffs.push((i, NAMES[i], scan_raw[i], ref_raw[i], d, sq));
    }

    // Sort by squared weighted diff, worst first
    diffs.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());

    println!("Total dist² = {:.6}", total_sq);
    println!();
    println!("{:>4} {:>14} {:>10} {:>10} {:>10} {:>10}  {:>5}", "dim", "name", "scan_raw", "ref_raw", "wt_diff", "wt_diff²", "cum%");
    println!("{}", "-".repeat(75));

    let mut cum = 0.0f32;
    for (i, name, sr, rr, d, sq) in &diffs {
        cum += sq;
        let pct = cum / total_sq * 100.0;
        println!("{:>4} {:>14} {:>10.4} {:>10.4} {:>10.6} {:>10.6}  {:>5.1}",
            i, name, sr, rr, d, sq, pct);
    }
}
