/// Exhaustive labeled training data generator for ML font classification.
///
/// Renders every (font × character × native-height × AA variant) combination
/// through the same pipeline as the character index, producing feature vectors
/// and normalized glyph images with full font identity metadata.
///
/// The native-height parameter directly simulates what the glyph would look
/// like at different pixel sizes before normalization. For example:
///   height=48 → full quality (no downscale)
///   height=12 → 9pt text at 100 DPI (very aggressive upscale on normalize)
///   height=9  → worst-case tiny text
///
/// Usage:
///   gen_training_data                                  # defaults to ./training-data/
///   gen_training_data --output-dir /data/training       # custom output
///   gen_training_data --heights 48,24,12               # subset of native heights
///   gen_training_data --aa native,blur_0.5             # subset of AA variants
///   gen_training_data --max-fonts 50                   # limit font count (for testing)

use ab_glyph::{Font, FontRef};
use clap::Parser;
use image::{GrayImage, Luma};
use rayon::prelude::*;
use serde::Serialize;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use unscan::char_index::{self, compute_features, FEAT_LEN};
use unscan::font_scan::{self, FontClass};

// ---------------------------------------------------------------------------
// Feature names (copied from learn_weights.rs)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "gen_training_data", about = "Generate labeled training data for ML font classification")]
struct Args {
    /// Output directory for generated data.
    #[arg(long, default_value = "training-data")]
    output_dir: PathBuf,

    /// Comma-separated native pixel heights to simulate (default: NORM_H).
    /// Each value represents the glyph height in pixels before normalization.
    #[arg(long, value_delimiter = ',')]
    heights: Vec<u32>,

    /// Comma-separated AA variants (default: native,blur_0.5,sharpen,binary_128).
    /// Options: native, blur_0.5, sharpen, binary_128
    #[arg(long, default_value = "native,blur_0.5,sharpen,binary_128", value_delimiter = ',')]
    aa: Vec<String>,

    /// Maximum number of fonts to process (for testing). 0 = all.
    #[arg(long, default_value = "0")]
    max_fonts: usize,

    /// Extra font directories to scan.
    #[arg(long)]
    font_dir: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// AA variant — use shared definition from char_index
// ---------------------------------------------------------------------------

use unscan::char_index::AaVariant;

// ---------------------------------------------------------------------------
// Native-height simulation
// ---------------------------------------------------------------------------

/// Render a character at a simulated native pixel height with a specific AA variant.
///
/// Pipeline:
/// 1. Render at full quality using ab_glyph (height = NORM_H ≈ 48px)
/// 2. Apply AA variant (if not native)
/// 3. Downscale to simulate a glyph that was natively `target_height` pixels tall
/// 4. Normalize through normalize_to_ink_bounds (scales back to NORM_H)
///
/// This directly models what happens to features when the source glyph is small:
///   target_height=48 → no downscale, full quality features
///   target_height=12 → 4:1 downscale then 4:1 upscale, heavy interpolation blur
///   target_height=9  → extreme quality loss, worst-case scenario
///
/// When `overrides` is provided (OT variant entries), uses resolve_glyph to
/// pick the variant-specific glyph ID instead of the default cmap.
fn render_char_at_native_height<F: Font>(
    font: &F, c: char, target_height: u32, aa: AaVariant,
    overrides: Option<&[(char, u16)]>,
) -> Option<GrayImage> {
    // Render at full quality — use resolve_glyph for variant support
    let gid = char_index::resolve_glyph(font, c, overrides);
    let full = char_index::render_glyph_normalised(font, gid)?;

    // Apply AA transformation before downscaling
    let aa_applied = aa.apply(&full);

    let (_w, h) = aa_applied.dimensions();

    // If target height is >= actual rendered height, skip downscale/upscale
    if target_height >= h {
        return char_index::normalize_to_ink_bounds(&aa_applied);
    }

    // Downscale: height goes to target_height, width scales proportionally
    let scale_factor = target_height as f32 / h as f32;
    let small_h = target_height.max(3);
    let small_w = ((_w as f32 * scale_factor).round() as u32).max(3);

    // Downscale (simulates a natively small glyph)
    let small = image::imageops::resize(
        &aa_applied,
        small_w,
        small_h,
        image::imageops::FilterType::Lanczos3,
    );

    // Normalize back through the same path as scan crops
    // (scales back up to NORM_H — this is where feature drift happens)
    char_index::normalize_to_ink_bounds(&small)
}

// ---------------------------------------------------------------------------
// Font identity extraction
// ---------------------------------------------------------------------------

/// Read all available name table entries from a font file.
fn read_name_ids(data: &[u8]) -> (String, String, String, String) {
    use rustybuzz::ttf_parser;

    let face = match ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return (String::new(), String::new(), String::new(), String::new()),
    };

    let mut nid1 = String::new();  // family
    let mut nid2 = String::new();  // subfamily
    let mut nid4 = String::new();  // full name
    let mut nid6 = String::new();  // PostScript name

    for name in face.names() {
        match name.name_id {
            1 if nid1.is_empty() => { if let Some(s) = name.to_string() { nid1 = s; } }
            2 if nid2.is_empty() => { if let Some(s) = name.to_string() { nid2 = s; } }
            4 if nid4.is_empty() => { if let Some(s) = name.to_string() { nid4 = s; } }
            6 if nid6.is_empty() => { if let Some(s) = name.to_string() { nid6 = s; } }
            _ => {}
        }
    }

    (nid1, nid2, nid4, nid6)
}

/// Read OS/2 weight class from font data.
fn read_weight_class(data: &[u8]) -> u16 {
    use rustybuzz::ttf_parser;
    match ttf_parser::Face::parse(data, 0) {
        Ok(face) => face.tables().os2
            .map(|os2| os2.weight().to_number())
            .unwrap_or(400),
        Err(_) => 400,
    }
}

/// Check if italic via OS/2 fsSelection.
fn read_italic(data: &[u8]) -> bool {
    use rustybuzz::ttf_parser;
    match ttf_parser::Face::parse(data, 0) {
        Ok(face) => face.is_italic(),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Label record
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LabelRecord {
    font_path: String,
    font_key: String,
    postscript_name: String,
    family_name: String,
    nid1_family: String,
    nid2_subfamily: String,
    nid4_full_name: String,
    nid16_family: String,
    weight_class: u16,
    weight_bucket: u16,
    italic: bool,
    font_class: String,
    variant_tag: String,
    char_str: String,
    char_code: u32,
    native_height: u32,
    aa_variant: String,
    is_ligature: bool,
}

// ---------------------------------------------------------------------------
// Per-sample result
// ---------------------------------------------------------------------------

struct Sample {
    features: [f32; FEAT_LEN],
    image: GrayImage,
    label: LabelRecord,
}

// ---------------------------------------------------------------------------
// Ligature detection
// ---------------------------------------------------------------------------

fn is_ligature(c: char) -> bool {
    matches!(c,
        '\u{FB00}' | // ff
        '\u{FB01}' | // fi
        '\u{FB02}' | // fl
        '\u{FB03}' | // ffi
        '\u{FB04}'   // ffl
    )
}

fn font_class_str(class: FontClass) -> &'static str {
    match class {
        FontClass::Serif => "serif",
        FontClass::Sans => "sans",
        FontClass::Mono => "mono",
        FontClass::Unknown => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Manifest {
    n_samples: usize,
    n_fonts: usize,
    n_chars: usize,
    n_heights: usize,
    n_aa_variants: usize,
    feat_dim: usize,
    feature_names: Vec<String>,
    heights: Vec<u32>,
    aa_variants: Vec<String>,
    chars: Vec<String>,
    norm_h: u32,
    generated_at: String,
    git_commit: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let mut args = Args::parse();
    if args.heights.is_empty() {
        args.heights = vec![char_index::NORM_H];
    }

    let aa_variants: Vec<AaVariant> = args.aa.iter()
        .filter_map(|s| AaVariant::parse(s))
        .collect();
    if aa_variants.is_empty() {
        eprintln!("ERROR: no valid AA variants specified");
        std::process::exit(1);
    }

    eprintln!("=== unscan training data generator ===");
    eprintln!("Output dir: {}", args.output_dir.display());
    eprintln!("Native heights: {:?}", args.heights);
    eprintln!("AA variants: {:?}", aa_variants.iter().map(|a| a.name()).collect::<Vec<_>>());

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).expect("create output dir");

    // ── 1. Scan all system fonts ──────────────────────────────────
    let font_dirs = font_scan::default_font_dirs(&args.font_dir);
    eprintln!("Scanning fonts from {:?}...", font_dirs);
    let mut catalog = font_scan::scan_fonts(&font_dirs);
    eprintln!("  {} font entries found", catalog.len());

    if args.max_fonts > 0 && catalog.len() > args.max_fonts {
        eprintln!("  Limiting to {} fonts (--max-fonts)", args.max_fonts);
        catalog.truncate(args.max_fonts);
    }

    let chars: &[char] = char_index::indexed_chars();
    eprintln!("  {} indexed characters", chars.len());

    let total_renders = catalog.len() * chars.len() * args.heights.len() * aa_variants.len();
    eprintln!("  {} fonts × {} chars × {} heights × {} AA = {} total renders",
        catalog.len(), chars.len(), args.heights.len(), aa_variants.len(), total_renders);

    // ── 2. Render all combinations in parallel ────────────────────
    let progress = AtomicUsize::new(0);
    let total_fonts = catalog.len();

    let start = std::time::Instant::now();

    // Process fonts in parallel; each font produces a batch of samples.
    // Write incrementally per-chunk to avoid OOM with thousands of fonts.
    let chunk_size = 100;
    let features_path = args.output_dir.join("features.bin");
    let images_path = args.output_dir.join("images.bin");
    let labels_path = args.output_dir.join("labels.jsonl");

    // Create/truncate output files
    let mut feat_writer = BufWriter::with_capacity(
        8 * 1024 * 1024,
        std::fs::File::create(&features_path).expect("create features.bin"),
    );
    let mut img_writer = BufWriter::with_capacity(
        8 * 1024 * 1024,
        std::fs::File::create(&images_path).expect("create images.bin"),
    );
    let mut label_writer = BufWriter::with_capacity(
        4 * 1024 * 1024,
        std::fs::File::create(&labels_path).expect("create labels.jsonl"),
    );

    let mut n_samples = 0usize;

    for chunk_start in (0..catalog.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(catalog.len());
        let chunk = &catalog[chunk_start..chunk_end];

        let chunk_samples: Vec<Vec<Sample>> = chunk.par_iter().map(|fe| {
        // Font data is dropped after scan_fonts(); load from path on demand.
        let font_data = match std::fs::read(&fe.path) {
            Ok(d) => d,
            Err(_) => {
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 {
                    eprintln!("  [{}/{}] fonts processed...", done, total_fonts);
                }
                return Vec::new();
            }
        };
        let font = match FontRef::try_from_slice(&font_data) {
            Ok(f) => f,
            Err(_) => {
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 {
                    eprintln!("  [{}/{}] fonts processed...", done, total_fonts);
                }
                return Vec::new();
            }
        };

        // Read font identity metadata
        let (nid1, nid2, nid4, nid6) = read_name_ids(&font_data);
        let weight_class = read_weight_class(&font_data);
        let italic = read_italic(&font_data);
        let nid16_family = {
            // nid16 is the typographic family; read_font_identity extracts it
            // but we may need it separately. Try to get it from the name table.
            use rustybuzz::ttf_parser;
            ttf_parser::Face::parse(&font_data, 0).ok()
                .and_then(|face| {
                    face.names().into_iter()
                        .find(|n| n.name_id == 16)
                        .and_then(|n| n.to_string())
                })
                .unwrap_or_else(|| nid1.clone())
        };

        let font_key = fe.font_key();
        let font_path = fe.path.display().to_string();
        let font_class = font_class_str(fe.class);

        let overrides = fe.glyph_overrides.as_deref();

        let mut batch: Vec<Sample> = Vec::new();

        for &c in chars {
            for &ht in &args.heights {
                for &aa in &aa_variants {
                    let img = match render_char_at_native_height(&font, c, ht, aa, overrides) {
                        Some(img) => img,
                        None => continue,
                    };

                    let feats = match compute_features(&img) {
                        Some(f) => f.as_slice(),
                        None => continue,
                    };

                    let label = LabelRecord {
                        font_path: font_path.clone(),
                        font_key: font_key.clone(),
                        postscript_name: if !nid6.is_empty() { nid6.clone() } else { fe.postscript_name.clone() },
                        family_name: fe.family_name.clone(),
                        nid1_family: nid1.clone(),
                        nid2_subfamily: nid2.clone(),
                        nid4_full_name: nid4.clone(),
                        nid16_family: nid16_family.clone(),
                        weight_class,
                        weight_bucket: weight_class / 100,
                        italic,
                        font_class: font_class.to_string(),
                        variant_tag: fe.variant_tag.clone(),
                        char_str: c.to_string(),
                        char_code: c as u32,
                        native_height: ht,
                        aa_variant: aa.name().to_string(),
                        is_ligature: is_ligature(c),
                    };

                    batch.push(Sample {
                        features: feats,
                        image: img,
                        label,
                    });
                }
            }
        }

        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 50 == 0 || done == total_fonts {
            eprintln!("  [{}/{}] fonts processed ({} samples so far)...",
                done, total_fonts, batch.len());
        }

        batch
    }).collect();

        // Write this chunk's samples to disk immediately
        for batch in &chunk_samples {
            for sample in batch {
                for &f in &sample.features {
                    feat_writer.write_all(&f.to_le_bytes()).expect("write feature");
                }
                let (w, h) = sample.image.dimensions();
                img_writer.write_all(&h.to_le_bytes()).expect("write img h");
                img_writer.write_all(&w.to_le_bytes()).expect("write img w");
                img_writer.write_all(sample.image.as_raw()).expect("write img data");
                serde_json::to_writer(&mut label_writer, &sample.label).expect("write label");
                label_writer.write_all(b"\n").expect("write newline");
                n_samples += 1;
            }
        }
        // Drop chunk_samples to free memory before next chunk
        drop(chunk_samples);

        let done = chunk_end;
        eprintln!("  [{}/{}] fonts processed ({} samples so far)...",
            done, total_fonts, n_samples);
    }

    feat_writer.flush().expect("flush features.bin");
    img_writer.flush().expect("flush images.bin");
    label_writer.flush().expect("flush labels.jsonl");

    let elapsed = start.elapsed();
    eprintln!("\nRendering complete: {} samples in {:.1}s", n_samples, elapsed.as_secs_f64());

    eprintln!("  {} samples written", n_samples);

    // manifest.json
    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let chars_strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    let unique_fonts = catalog.len();

    let manifest = Manifest {
        n_samples,
        n_fonts: unique_fonts,
        n_chars: chars.len(),
        n_heights: args.heights.len(),
        n_aa_variants: aa_variants.len(),
        feat_dim: FEAT_LEN,
        feature_names: FEAT_NAMES.iter().map(|s| s.to_string()).collect(),
        heights: args.heights.clone(),
        aa_variants: aa_variants.iter().map(|a| a.name().to_string()).collect(),
        chars: chars_strs,
        norm_h: char_index::NORM_H,  // NORM_H from char_index
        generated_at: chrono_now(),
        git_commit,
    };

    let manifest_path = args.output_dir.join("manifest.json");
    let manifest_file = std::fs::File::create(&manifest_path).expect("create manifest.json");
    serde_json::to_writer_pretty(manifest_file, &manifest).expect("write manifest");

    let total_elapsed = start.elapsed();

    // Size report
    let feat_size = std::fs::metadata(&features_path).map(|m| m.len()).unwrap_or(0);
    let img_size = std::fs::metadata(&images_path).map(|m| m.len()).unwrap_or(0);
    let label_size = std::fs::metadata(&labels_path).map(|m| m.len()).unwrap_or(0);
    let manifest_size = std::fs::metadata(&manifest_path).map(|m| m.len()).unwrap_or(0);

    eprintln!("\n=== Generation complete ===");
    eprintln!("  Samples:      {}", n_samples);
    eprintln!("  Unique fonts: {}", unique_fonts);
    eprintln!("  features.bin: {} ({:.1} MB)", features_path.display(), feat_size as f64 / 1e6);
    eprintln!("  images.bin:   {} ({:.1} MB)", images_path.display(), img_size as f64 / 1e6);
    eprintln!("  labels.jsonl: {} ({:.1} MB)", labels_path.display(), label_size as f64 / 1e6);
    eprintln!("  manifest.json: {} ({:.1} KB)", manifest_path.display(), manifest_size as f64 / 1e3);
    eprintln!("  Total size:   {:.1} MB", (feat_size + img_size + label_size + manifest_size) as f64 / 1e6);
    eprintln!("  Total time:   {:.1}s", total_elapsed.as_secs_f64());
}

/// Simple ISO 8601 timestamp without pulling in chrono crate.
fn chrono_now() -> String {
    use std::process::Command;
    Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
