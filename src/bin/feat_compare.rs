//! Compare all 100 features between a scan crop and two font renderings of the same character.
//!
//! Usage: feat_compare <crop.png> <char> <font1.ttf> <font2.ttf>

use ab_glyph::{Font, FontRef};
use image::GrayImage;

use unscan::char_index::{self, compute_features, FEAT_LEN};

const FEAT_NAMES: &[&str] = &[
    // Group 1: Column ink profile (32)
    "col0","col1","col2","col3","col4","col5","col6","col7",
    "col8","col9","col10","col11","col12","col13","col14","col15",
    "col16","col17","col18","col19","col20","col21","col22","col23",
    "col24","col25","col26","col27","col28","col29","col30","col31",
    // Group 2: Scalar v1 (7)
    "aspect","ink_density","v_center","h_balance","serif_score","stroke_contrast","xh_cap_ratio",
    // Group 3: Scalar v2 (18)
    "counter_area","counter_cx","counter_cy","counter_asp",
    "term0","term1","term2","term3",
    "ink_perim","compactness",
    "cross0","cross1","cross2","cross3","cross4","cross5","cross6","cross7",
    // Group 4: Row ink profile (32)
    "row0","row1","row2","row3","row4","row5","row6","row7",
    "row8","row9","row10","row11","row12","row13","row14","row15",
    "row16","row17","row18","row19","row20","row21","row22","row23",
    "row24","row25","row26","row27","row28","row29","row30","row31",
    // Group 5: Scalar v3 (11)
    "hole_count","h_symmetry","v_symmetry","skel_branch","skel_endpt",
    "corner_count","quad_tl","quad_tr","quad_bl","quad_br","mean_stroke_w",
];

fn group_name(i: usize) -> &'static str {
    if i < 32 { "col_prof" }
    else if i < 39 { "scal_v1" }
    else if i < 57 { "scal_v2" }
    else if i < 89 { "row_prof" }
    else { "scal_v3" }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: feat_compare <crop.png> <char> <font1.ttf> <font2.ttf>");
        std::process::exit(1);
    }

    let crop_path = &args[1];
    let ch: char = args[2].chars().next().expect("need a character");
    let font1_path = &args[3];
    let font2_path = &args[4];

    // Load and normalise the scan crop
    let crop_img = image::open(crop_path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", crop_path, e))
        .into_luma8();
    let crop_norm = char_index::normalize_to_ink_bounds(&crop_img)
        .unwrap_or_else(|| panic!("normalize_to_ink_bounds returned None for crop"));
    let crop_feats = compute_features(&crop_norm)
        .unwrap_or_else(|| panic!("compute_features returned None for crop"));

    // Render char from font 1
    let f1_data = std::fs::read(font1_path).unwrap();
    let f1 = FontRef::try_from_slice(&f1_data).unwrap();
    let f1_img = char_index::render_char_normalised(&f1, ch)
        .unwrap_or_else(|| panic!("render_char_normalised returned None for font1 '{}'", ch));
    let f1_feats = compute_features(&f1_img)
        .unwrap_or_else(|| panic!("compute_features returned None for font1"));

    // Render char from font 2
    let f2_data = std::fs::read(font2_path).unwrap();
    let f2 = FontRef::try_from_slice(&f2_data).unwrap();
    let f2_img = char_index::render_char_normalised(&f2, ch)
        .unwrap_or_else(|| panic!("render_char_normalised returned None for font2 '{}'", ch));
    let f2_feats = compute_features(&f2_img)
        .unwrap_or_else(|| panic!("compute_features returned None for font2"));

    let scan = crop_feats.as_slice();
    let r1 = f1_feats.as_slice();
    let r2 = f2_feats.as_slice();

    // Also get weighted versions
    let scan_w = crop_feats.weighted();
    let r1_w = f1_feats.weighted();
    let r2_w = f2_feats.weighted();

    // Print CSV header
    println!("dim,name,group,scan,font1,font2,d1_raw,d2_raw,scan_w,font1_w,font2_w,d1_w,d2_w,closer");
    for i in 0..FEAT_LEN {
        let d1 = (scan[i] - r1[i]).abs();
        let d2 = (scan[i] - r2[i]).abs();
        let d1w = (scan_w[i] - r1_w[i]).abs();
        let d2w = (scan_w[i] - r2_w[i]).abs();
        let closer = if d1w < d2w { "font1" } else if d2w < d1w { "font2" } else { "tie" };
        println!("{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            i, FEAT_NAMES[i], group_name(i),
            scan[i], r1[i], r2[i], d1, d2,
            scan_w[i], r1_w[i], r2_w[i], d1w, d2w, closer);
    }

    // Summary
    let total_d1w: f32 = (0..FEAT_LEN).map(|i| (scan_w[i] - r1_w[i]).powi(2)).sum();
    let total_d2w: f32 = (0..FEAT_LEN).map(|i| (scan_w[i] - r2_w[i]).powi(2)).sum();
    let f1_closer: usize = (0..FEAT_LEN).filter(|&i| {
        let d1w = (scan_w[i] - r1_w[i]).abs();
        let d2w = (scan_w[i] - r2_w[i]).abs();
        d1w < d2w
    }).count();
    let f2_closer: usize = (0..FEAT_LEN).filter(|&i| {
        let d1w = (scan_w[i] - r1_w[i]).abs();
        let d2w = (scan_w[i] - r2_w[i]).abs();
        d2w < d1w
    }).count();

    eprintln!("\nWeighted L2 distance: font1={:.6}  font2={:.6}", total_d1w.sqrt(), total_d2w.sqrt());
    eprintln!("Dims closer to font1: {}  font2: {}  tie: {}", f1_closer, f2_closer, FEAT_LEN - f1_closer - f2_closer);
}
