//! Integration test: render a known system font, extract characters,
//! query the char index, and verify the correct font wins.
//!
//! Uses DejaVu Sans (always installed at /usr/share/fonts/truetype/dejavu/).
//! Builds a small index from ~8 system fonts, renders "hamburgefontsiv" in
//! DejaVu Sans, extracts character crops, queries the index, and asserts
//! DejaVu Sans is the #1 match.

use image::{GrayImage, Luma};
use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use std::path::PathBuf;

use unscan::char_index::{
    build_char_index, compute_features, compute_font_serif_score, compute_xh_cap_ratio,
    render_char_normalised, search_candidates, GlyphOverrides, Variations, CharFeatures, FEAT_LEN,
};
use unscan::classifier::FisherClassifier;

/// Render a single character in the given font at NORM_H=48 ink height,
/// tight-cropped — same as what the index stores.
fn render_char_cropped(font_data: &[u8], c: char) -> Option<GrayImage> {
    let font = FontRef::try_from_slice(font_data).ok()?;
    let norm_h = 48u32;

    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);

    let gid = font.glyph_id(c);
    if gid.0 == 0 { return None; }

    let glyph = gid.with_scale_and_position(ref_scale, point(0.0, sf_ref.ascent()));
    let outlined = font.outline_glyph(glyph)?;
    let bounds = outlined.px_bounds();
    let ink_h_ref = bounds.max.y - bounds.min.y;
    if ink_h_ref < 1.0 { return None; }

    let target_scale = ref_h * (norm_h as f32 / ink_h_ref);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);

    let glyph2 = gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
    let outlined2 = font.outline_glyph(glyph2)?;
    let b2 = outlined2.px_bounds();

    let img_w = (b2.max.x - b2.min.x).ceil() as u32 + 2;
    let img_h = (b2.max.y - b2.min.y).ceil() as u32 + 2;
    if img_w == 0 || img_h == 0 || img_w > 500 || img_h > 500 { return None; }

    let mut canvas = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));
    let ox = b2.min.x.floor() as i32;
    let oy = b2.min.y.floor() as i32;

    outlined2.draw(|gx, gy, cov| {
        let px = gx as i32 + (b2.min.x.floor() as i32) - ox + 1;
        let py = gy as i32 + (b2.min.y.floor() as i32) - oy + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
        }
    });

    // Tight-crop to ink
    let (w, h) = canvas.dimensions();
    let mut min_r = h;
    let mut max_r = 0;
    let mut min_c = w;
    let mut max_c = 0;
    for y in 0..h {
        for x in 0..w {
            if canvas.get_pixel(x, y).0[0] < 200 {
                min_r = min_r.min(y);
                max_r = max_r.max(y);
                min_c = min_c.min(x);
                max_c = max_c.max(x);
            }
        }
    }
    if max_r < min_r { return None; }

    let cropped = image::imageops::crop_imm(
        &canvas, min_c, min_r,
        max_c - min_c + 1, max_r - min_r + 1
    ).to_image();

    let (cw, ch) = cropped.dimensions();
    let new_w = ((cw as f32 * norm_h as f32 / ch as f32).ceil() as u32).max(1);
    let resized = image::imageops::resize(
        &cropped, new_w, norm_h,
        image::imageops::FilterType::Lanczos3,
    );

    Some(resized)
}

/// Load font data from a system path
fn load_font(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("Cannot read font {}: {}", path, e))
}

/// System font paths for the test index.
const FONT_PATHS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
];

/// Collect system fonts as (name, PathBuf, GlyphOverrides, Variations) for build_char_index.
fn test_font_set() -> Vec<(String, PathBuf, GlyphOverrides, Variations)> {
    let mut fonts = Vec::new();
    for path in FONT_PATHS {
        if std::path::Path::new(path).exists() {
            fonts.push((path.to_string(), PathBuf::from(path), None, None));
        }
    }
    assert!(fonts.len() >= 4, "Need at least 4 system fonts for meaningful test");
    fonts
}

/// Build the test index and render crops (shared between tests)
fn setup() -> (unscan::char_index::CharIndex, Vec<(char, GrayImage)>, Vec<u8>) {
    let target_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    let target_data = load_font(target_path);
    let font_set = test_font_set();
    let index = build_char_index(&font_set, &FisherClassifier);

    let test_word = "hamburgefontsiv";
    let mut crops: Vec<(char, GrayImage)> = Vec::new();
    for c in test_word.chars() {
        if let Some(img) = render_char_cropped(&target_data, c) {
            crops.push((c, img));
        }
    }
    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    let unique_crops: Vec<(char, GrayImage)> = crops.into_iter()
        .filter(|(c, _)| seen.insert(*c))
        .collect();

    (index, unique_crops, target_data)
}

#[test]
fn char_index_identifies_dejavu_sans() {
    let target_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    let (index, unique_crops, _target_data) = setup();

    let font_set = test_font_set();
    eprintln!("\n=== Building index from {} fonts ===", font_set.len());
    for (key, _, _, _) in &font_set {
        eprintln!("  {}", key);
    }

    eprintln!("\n{} unique character crops", unique_crops.len());
    assert!(unique_crops.len() >= 5, "Should extract at least 5 unique characters, got {}", unique_crops.len());

    // Per-character nearest fonts (diagnostic output)
    eprintln!("\n=== Per-character nearest fonts ===");
    for (c, img) in &unique_crops {
        let single = vec![(*c, img.clone())];
        let char_results = search_candidates(&index, &single, 3.0, false, &FisherClassifier).scores;
        let top3: Vec<String> = char_results.iter()
            .map(|(n, s)| format!("{} ({:.4})", n, s))
            .collect();
        eprintln!("  '{}': {}", c, top3.join(" | "));
    }

    // Overall query
    let results = search_candidates(&index, &unique_crops, 5.0, false, &FisherClassifier).scores;

    eprintln!("\n=== Overall top {} matches ===", results.len());
    for (i, (name, score)) in results.iter().enumerate() {
        let marker = if name == target_font { " ✓" } else { "" };
        eprintln!("  #{}: {} (score: {:.4}){}", i + 1, name, score, marker);
    }

    // THE ASSERTION: DejaVu Sans must be #1
    assert!(!results.is_empty(), "Index returned no results");
    let (winner, winner_score) = &results[0];
    eprintln!("\nWinner: {} (score: {:.4})", winner, winner_score);

    assert_eq!(
        winner, target_font,
        "\nExpected '{}' to win but got '{}'.\nFull results: {:#?}",
        target_font, winner, results
    );
}

#[test]
fn ci_single_char_diagnostics() {
    let (index, unique_crops, _target_data) = setup();

    // Exercise search_candidates per character — show diagnostics
    for test_char in ['e', 'g', 'a'] {
        if let Some((_, crop)) = unique_crops.iter().find(|(c, _)| *c == test_char) {
            let single = vec![(test_char, crop.clone())];
            let result = search_candidates(&index, &single, 5.0, false, &FisherClassifier);

            eprintln!("\n=== CI search: '{}' ===", test_char);
            eprintln!("  Candidates:");
            for (i, (name, score)) in result.scores.iter().enumerate() {
                eprintln!("    #{}: {} ({:.4})", i + 1, name, score);
            }

            // The right font should be in the candidates
            assert!(result.scores.iter().any(|(n, _)| n.contains("DejaVuSans.ttf")),
                "DejaVuSans.ttf should be in candidates for '{}'", test_char);
        }
    }
}

#[test]
fn char_index_identifies_dejavu_sans_full_index() {
    let index_path = std::path::Path::new(env!("HOME"))
        .join(".cache/unscan/char-index.bin");

    if !index_path.exists() {
        eprintln!("SKIP: full index not found at {:?}", index_path);
        return;
    }

    let target_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    let target_data = load_font(target_font);

    eprintln!("\n=== Loading full index from {:?} ===", index_path);
    let index = unscan::char_index::load_index(&index_path, &FisherClassifier).expect("load index");
    let font_count = unscan::char_index::count_fonts(&index);
    eprintln!("  {} fonts in index", font_count);

    // Render "hamburgefontsiv" in DejaVu Sans
    let test_word = "hamburgefontsiv";
    let mut crops: Vec<(char, GrayImage)> = Vec::new();
    for c in test_word.chars() {
        if let Some(img) = render_char_cropped(&target_data, c) {
            crops.push((c, img));
        }
    }
    let mut seen = std::collections::HashSet::new();
    let unique_crops: Vec<(char, GrayImage)> = crops.into_iter()
        .filter(|(c, _)| seen.insert(*c))
        .collect();

    eprintln!("{} unique character crops", unique_crops.len());

    // Per-character top 5
    eprintln!("\n=== Per-character nearest fonts (full index) ===");
    for (c, img) in &unique_crops {
        let single = vec![(*c, img.clone())];
        let char_results = search_candidates(&index, &single, 5.0, false, &FisherClassifier).scores;
        let top5: Vec<String> = char_results.iter()
            .take(5)
            .map(|(n, s)| format!("{} ({:.4})", n, s))
            .collect();
        eprintln!("  '{}': {}", c, top5.join(" | "));
    }

    // Overall top 10
    let results = search_candidates(&index, &unique_crops, 10.0, false, &FisherClassifier).scores;

    eprintln!("\n=== Overall top 10 matches (full {} font index) ===", font_count);
    for (i, (name, score)) in results.iter().take(10).enumerate() {
        let marker = if name == target_font { " ✓" } else { "" };
        eprintln!("  #{}: {} (score: {:.4}){}", i + 1, name, score, marker);
    }

    assert!(!results.is_empty(), "Index returned no results");

    // With 5048 fonts, DejaVuSans.ttf should still be #1 or at least top-3
    let dejavu_pos = results.iter().position(|(n, _)| n.contains("DejaVuSans.ttf"));
    eprintln!("\nDejaVuSans.ttf position: {:?}", dejavu_pos.map(|p| p + 1));

    assert!(
        dejavu_pos.is_some() && dejavu_pos.unwrap() < 3,
        "DejaVuSans.ttf should be top-3 but was at position {:?}. Winner: {}",
        dejavu_pos.map(|p| p + 1), &results[0].0
    );
}

#[test]
fn compare_dejavu_vs_noto_per_char() {
    let index_path = std::path::Path::new(env!("HOME"))
        .join(".cache/unscan/char-index.bin");
    if !index_path.exists() {
        eprintln!("SKIP: full index not found");
        return;
    }

    let target_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    let target_data = load_font(target_path);
    let index = unscan::char_index::load_index(&index_path, &FisherClassifier).expect("load index");

    let test_word = "hamburgefontsiv";
    let mut seen = std::collections::HashSet::new();

    eprintln!("\n{:<6} {:<50} {:<10} {:<50} {:<10} {:<8}",
        "Char", "DejaVuSans.ttf", "Score", "NotoSansSymbols", "Score", "Delta");
    eprintln!("{}", "-".repeat(140));

    for c in test_word.chars() {
        if !seen.insert(c) { continue; }
        let img = match render_char_cropped(&target_data, c) {
            Some(i) => i,
            None => continue,
        };
        let single = vec![(c, img)];
        // Get ALL results with high thoroughness
        let results = search_candidates(&index, &single, 100.0, false, &FisherClassifier).scores;

        let dv = results.iter().enumerate()
            .find(|(_, (n, _))| n.contains("DejaVuSans.ttf"));
        let noto = results.iter().enumerate()
            .find(|(_, (n, _))| n.contains("NotoSansSymbols"));

        let (dv_rank, dv_score) = match dv {
            Some((i, (_, s))) => (format!("#{}", i+1), *s),
            None => ("N/A".into(), 0.0),
        };
        let (noto_rank, noto_score) = match noto {
            Some((i, (_, s))) => (format!("#{}", i+1), *s),
            None => ("N/A".into(), 0.0),
        };
        let delta = dv_score - noto_score;
        let winner = if delta > 0.0 { "DV" } else if delta < 0.0 { "Noto" } else { "tie" };

        eprintln!("  '{}'   DejaVu {} {:.6}    Noto {} {:.6}    {:+.6} ({})",
            c, dv_rank, dv_score, noto_rank, noto_score, delta, winner);
    }
}


/// Cosine similarity of two f32 slices.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut mag_a = 0.0f64;
    let mut mag_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        mag_a += (*x as f64) * (*x as f64);
        mag_b += (*y as f64) * (*y as f64);
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-12 { return 0.0; }
    (dot / denom) as f32
}

#[test]
fn feature_self_consistency() {
    // Verify that re-computing features on a font's own rendered character
    // produces feature vectors that are identical (within pixelisation margin)
    // to what the index stores from the build path.

    let test_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ];

    let test_chars = ['a', 'e', 'g', 'h', 'm', 'n', 'o', 'r', 's', 't'];

    // Self-match should be exact (same code path, deterministic rendering)
    let min_sim: f32 = 0.999;

    eprintln!("\n=== Feature Self-Consistency Test (FEAT_LEN={}) ===", FEAT_LEN);
    eprintln!("{:<50} {:<6} {:<12} {:<12} {}", "Font", "Char", "Cos(raw)", "Cos(weighted)", "Status");
    eprintln!("{}", "-".repeat(100));

    let mut total = 0u32;
    let mut passed = 0u32;
    let mut worst_sim = 1.0f32;
    let mut worst_case = String::new();

    for font_path in &test_fonts {
        if !std::path::Path::new(font_path).exists() {
            eprintln!("  SKIP: {} not found", font_path);
            continue;
        }

        let font_data = std::fs::read(font_path).unwrap();

        // Build a 1-font index from just this font
        let font_set = vec![(font_path.to_string(), PathBuf::from(*font_path), None, None)];
        let index = build_char_index(&font_set, &FisherClassifier);

        // Parse the font for per-font metric overrides (same as build path)
        let font = FontRef::try_from_slice(&font_data).expect("parse font");
        let xh_ratio = compute_xh_cap_ratio(&font);
        let serif = compute_font_serif_score(&font);

        for &c in &test_chars {
            // Render using the exact same function as build_char_index
            let rendered = match render_char_normalised(&font, c) {
                Some(img) => img,
                None => continue,
            };

            // Compute features and apply per-font overrides (same as build path)
            let mut query_feats = match compute_features(&rendered) {
                Some(f) => f,
                None => continue,
            };
            query_feats.xh_cap_ratio = xh_ratio;
            query_feats.serif_score = serif;

            // Get the index entry for this font+char (build path)
            let index_feats = match index.entries.get(&c) {
                Some(entries) => {
                    match entries.iter().find(|e| e.font_name == *font_path) {
                        Some(e) => e.features.clone(),
                        None => continue,
                    }
                }
                None => continue,
            };

            total += 1;

            // Compare raw feature vectors
            let raw_q = query_feats.as_slice();
            let raw_i = index_feats.as_slice();
            let raw_sim = cosine_sim(&raw_q, &raw_i);

            // Compare weighted feature vectors
            let w_q = query_feats.weighted();
            let w_i = index_feats.weighted();
            let weighted_sim = cosine_sim(&w_q, &w_i);

            let ok = raw_sim >= min_sim && weighted_sim >= min_sim;
            if ok { passed += 1; }
            let status = if ok { "✓" } else { "FAIL" };

            if raw_sim < worst_sim {
                worst_sim = raw_sim;
                worst_case = format!("{} '{}'", font_path, c);
            }

            eprintln!("{:<16} '{}'    {:.6}     {:.6}       {}",
                font_path, c, raw_sim, weighted_sim, status);

            assert!(raw_sim >= min_sim,
                "Self-consistency FAIL for {} '{}': raw cosine sim {:.6} < {}",
                font_path, c, raw_sim, min_sim);
            assert!(weighted_sim >= min_sim,
                "Self-consistency FAIL for {} '{}': weighted cosine sim {:.6} < {}",
                font_path, c, weighted_sim, min_sim);
        }
    }

    eprintln!("\n=== Summary ===");
    eprintln!("  {}/{} passed (min_sim={})", passed, total, min_sim);
    eprintln!("  Worst similarity: {:.6} ({})", worst_sim, worst_case);
    assert!(total >= 30, "Expected at least 30 font/char combos, got {}", total);
    assert_eq!(passed, total, "Not all self-consistency checks passed");
}
