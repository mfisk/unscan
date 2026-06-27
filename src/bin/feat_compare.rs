//! Compare all FEAT_LEN features between a scan crop and two font renderings of the same character.
//!
//! Usage: feat_compare <crop.png> <char> <font1.ttf> <font2.ttf>

use ab_glyph::{Font, FontRef};

use unscan::char_index::{self, compute_features, FEAT_LEN, group_name_for_dim};

use unscan::char_index::FEAT_NAMES;

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
    let crop_norm = char_index::normalize_to_ink_bounds(&crop_img, char_index::NORM_H)
        .unwrap_or_else(|| panic!("normalize_to_ink_bounds returned None for crop"));
    let crop_feats = compute_features(&crop_norm)
        .unwrap_or_else(|| panic!("compute_features returned None for crop"));

    // Render char from font 1
    let f1_data = std::fs::read(font1_path).unwrap();
    let f1 = FontRef::try_from_slice(&f1_data).unwrap();
    let f1_img = unscan::char_render::render_glyph_at_ink_height(&f1, f1.glyph_id(ch), char_index::NORM_H)
        .and_then(|img| char_index::normalize_to_ink_bounds(&img, char_index::NORM_H))
        .unwrap_or_else(|| panic!("render returned None for font1 '{}'", ch));
    let f1_feats = compute_features(&f1_img)
        .unwrap_or_else(|| panic!("compute_features returned None for font1"));

    // Render char from font 2
    let f2_data = std::fs::read(font2_path).unwrap();
    let f2 = FontRef::try_from_slice(&f2_data).unwrap();
    let f2_img = unscan::char_render::render_glyph_at_ink_height(&f2, f2.glyph_id(ch), char_index::NORM_H)
        .and_then(|img| char_index::normalize_to_ink_bounds(&img, char_index::NORM_H))
        .unwrap_or_else(|| panic!("render returned None for font2 '{}'", ch));
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
            i, FEAT_NAMES[i], group_name_for_dim(i),
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
