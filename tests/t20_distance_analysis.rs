//! Distance analysis: for each specimen font, check how far the correct font
//! is from the nearest neighbor in the character index, per character.
//!
//! This tells us whether the 2× adaptive cutoff is sufficient.

use std::path::Path;
use std::collections::HashMap;

use unscan::char_index::{
    load_index, CharIndex, FEAT_LEN,
};
use unscan::classifier::FisherClassifier;

/// Extract a short display name from a font path or name
fn short_name(full: &str) -> &str {
    full.rsplit('/').next().unwrap_or(full)
}
fn sq_dist(a: &[f32; FEAT_LEN], b: &[f32; FEAT_LEN]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity between two weighted feature vectors.
fn cosine_sim(a: &[f32; FEAT_LEN], b: &[f32; FEAT_LEN]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a < 1e-12 || mag_b < 1e-12 { return 0.0; }
    dot / (mag_a * mag_b)
}

/// For a given font, look up its feature vector from the index entries,
/// then brute-force rank ALL fonts in the index by distance, and report
/// where the target font sits.
fn analyze_font_char(
    index: &CharIndex,
    target_font: &str,
    ch: char,
) -> Option<CharDistanceInfo> {
    let entries = index.entries.get(&ch)?;
    
    // Get the target font's features
    let target_entry = entries.iter().find(|e| e.font_name == target_font)?;
    let target_weighted = target_entry.features.weighted();
    
    // Compute distance from target to every other font for this char
    let mut distances: Vec<(String, f32, f32)> = Vec::new(); // (name, dist², cosine)
    for e in entries {
        let w = e.features.weighted();
        let d = sq_dist(&target_weighted, &w);
        let cos = cosine_sim(&target_weighted, &w);
        distances.push((e.font_name.clone(), d, cos));
    }
    
    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Find rank of target (should be #1 = self at dist 0)
    // But we want the rank when queried from a RENDERED image, so self is #1.
    // What we really want: nearest non-self neighbor distance, and how the target
    // compares to neighbors.
    
    // Since we're looking up the target from its OWN features, it will be at dist=0.
    // Instead, report: #2 neighbor (closest OTHER font), and how tightly packed the field is.
    
    let self_idx = distances.iter().position(|d| d.0 == target_font).unwrap_or(0);
    
    // First non-self neighbor
    let nearest_other = distances.iter()
        .find(|d| d.0 != target_font)
        .map(|d| (d.0.clone(), d.1, d.2));
    
    // Count fonts within various distance multiples of nearest other
    let nearest_dist_sq = nearest_other.as_ref().map(|n| n.1).unwrap_or(f32::MAX);
    let within_2x = distances.iter().filter(|d| d.0 != target_font && d.1 <= nearest_dist_sq * 4.0).count();
    let within_3x = distances.iter().filter(|d| d.0 != target_font && d.1 <= nearest_dist_sq * 9.0).count();
    let within_5x = distances.iter().filter(|d| d.0 != target_font && d.1 <= nearest_dist_sq * 25.0).count();
    let within_10x = distances.iter().filter(|d| d.0 != target_font && d.1 <= nearest_dist_sq * 100.0).count();
    
    let top10: Vec<(String, f32, f32)> = distances.iter()
        .filter(|d| d.0 != target_font)
        .take(10)
        .cloned()
        .collect();
    
    Some(CharDistanceInfo {
        target_font: target_font.to_string(),
        ch,
        nearest_other_name: nearest_other.as_ref().map(|n| n.0.clone()).unwrap_or_default(),
        nearest_other_dist_sq: nearest_dist_sq,
        nearest_other_cosine: nearest_other.as_ref().map(|n| n.2).unwrap_or(0.0),
        within_2x,
        within_3x,
        within_5x,
        within_10x,
        total_fonts: distances.len() - 1, // exclude self
        top10,
    })
}

struct CharDistanceInfo {
    target_font: String,
    ch: char,
    nearest_other_name: String,
    nearest_other_dist_sq: f32,
    nearest_other_cosine: f32,
    within_2x: usize,
    within_3x: usize,
    within_5x: usize,
    within_10x: usize,
    total_fonts: usize,
    top10: Vec<(String, f32, f32)>,
}

#[test]
fn distance_analysis_specimen_fonts() {
    let index_path = Path::new("/home/hatch/.cache/unscan/char-index.bin");
    
    eprintln!("Looking for index at {:?}, exists={}", index_path, index_path.exists());
    
    if !index_path.exists() {
        panic!("Char index not found at {:?}", index_path);
    }
    
    let index = load_index(&index_path, &FisherClassifier).expect("Failed to load index");
    eprintln!("Loaded index: {} chars indexed", index.entries.len());
    
    // Dump specimen-related font names to find the correct naming
    {
        let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (_ch, entries) in &index.entries {
            for e in entries {
                names.insert(e.font_name.clone());
            }
        }
        let mut sorted: Vec<&String> = names.iter().collect();
        sorted.sort();
        println!("\n=== Specimen-related font names in index ===");
        for n in &sorted {
            let nl = n.to_lowercase();
            if nl.contains("garamond") || nl.contains("caslon") || nl.contains("baskerville") 
                || nl.contains("bodoni") || nl.contains("zilla") {
                println!("  '{}'", n);
            }
        }
        println!("Total unique font names: {}", sorted.len());
    }
    
    let specimen_fonts = [
        "/usr/share/fonts/truetype/specimen-fonts/eb-garamond-400.ttf",
        "/usr/share/fonts/truetype/specimen-fonts/libre-caslon-text-400.ttf",
        "/usr/share/fonts/truetype/specimen-fonts/libre-baskerville-400.ttf",
        "/usr/share/fonts/truetype/specimen-fonts/libre-bodoni-400.ttf",
        "/usr/share/fonts/truetype/specimen-fonts/zilla-slab-400.ttf",
    ];
    
    let test_chars = ['e', 'a', 'o', 'n', 't', 's'];
    
    // First verify the fonts are in the index
    println!("\n=== Font Availability Check ===");
    for font in &specimen_fonts {
        let found_chars: Vec<char> = test_chars.iter()
            .filter(|&&c| {
                index.entries.get(&c)
                    .map(|es| es.iter().any(|e| e.font_name == *font))
                    .unwrap_or(false)
            })
            .copied()
            .collect();
        println!("  {}: found for chars {:?}", font, found_chars);
    }
    
    // Summary table header
    println!("\n{}", "=".repeat(130));
    println!("=== DISTANCE ANALYSIS: How far is the correct font from its nearest neighbor? ===");
    println!("{}", "=".repeat(130));
    println!("{:<30} {:>4} {:<30} {:>10} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Font", "Char", "Nearest Other", "Dist²", "Cosine", "≤2×", "≤3×", "≤5×", "≤10×", "Total");
    println!("{}", "-".repeat(130));
    
    for font in &specimen_fonts {
        for &ch in &test_chars {
            match analyze_font_char(&index, font, ch) {
                Some(info) => {
                    let short_target = short_name(&info.target_font);
                    let short_nearest = short_name(&info.nearest_other_name);
                    println!("{:<30} {:>4} {:<30} {:>10.6} {:>8.4} {:>6} {:>6} {:>6} {:>6} {:>6}",
                        short_target,
                        info.ch,
                        &short_nearest[..short_nearest.len().min(30)],
                        info.nearest_other_dist_sq,
                        info.nearest_other_cosine,
                        info.within_2x,
                        info.within_3x,
                        info.within_5x,
                        info.within_10x,
                        info.total_fonts,
                    );
                }
                None => {
                    println!("{:<30} {:>4} -- NOT IN INDEX --", short_name(font), ch);
                }
            }
        }
        println!("{}", "-".repeat(130));
    }
    
    // Detailed top-10 for each font (just first char 'e' to keep output manageable)
    println!("\n{}", "=".repeat(100));
    println!("=== DETAILED TOP-10 NEIGHBORS (per font × all test chars) ===");
    println!("{}", "=".repeat(100));
    
    for font in &specimen_fonts {
        println!("\n--- {} ---", short_name(font));
        for &ch in &test_chars {
            if let Some(info) = analyze_font_char(&index, font, ch) {
                println!("  '{}': nearest_dist²={:.6}, within_2x={}, within_3x={}, within_5x={}",
                    ch, info.nearest_other_dist_sq, info.within_2x, info.within_3x, info.within_5x);
                for (i, (name, dsq, cos)) in info.top10.iter().enumerate() {
                    let ratio = if info.nearest_other_dist_sq > 0.0 {
                        dsq.sqrt() / info.nearest_other_dist_sq.sqrt()
                    } else {
                        0.0
                    };
                    println!("    #{:>2}: {:<35} dist²={:.6}  ratio={:.2}×  cosine={:.4}",
                        i + 1, &short_name(name)[..short_name(name).len().min(35)], dsq, ratio, cos);
                }
            }
        }
    }
    
    // Aggregate: what's the worst case across all font×char combos?
    println!("\n{}", "=".repeat(100));
    println!("=== WORST CASES: Which font×char combos need the widest search? ===");
    println!("{}", "=".repeat(100));
    
    let mut all_results: Vec<(String, char, usize, usize, f32)> = Vec::new();
    for font in &specimen_fonts {
        for &ch in &test_chars {
            if let Some(info) = analyze_font_char(&index, font, ch) {
                all_results.push((
                    font.to_string(),
                    ch,
                    info.within_2x,
                    info.within_5x,
                    info.nearest_other_dist_sq,
                ));
            }
        }
    }
    
    // Sort by within_2x descending (most crowded neighborhoods)
    all_results.sort_by(|a, b| b.2.cmp(&a.2));
    println!("\nMost crowded neighborhoods (most fonts within 2× best distance):");
    for (font, ch, w2, w5, dist) in all_results.iter().take(10) {
        println!("  {:<30} '{}': {} within 2×, {} within 5× (nearest_dist²={:.6})",
            short_name(font), ch, w2, w5, dist);
    }
    
    println!("\n=== END OF ANALYSIS ===\n");
}
