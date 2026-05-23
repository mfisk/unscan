/// Dump index features and scan crop features for regression analysis.
///
/// Usage:
///   learn_weights --index          Dump all index features (raw) as TSV
///   learn_weights --scans <dir>    Dump scan crop features from diag-seg dir
///   learn_weights --analyze <index.tsv> <scans.tsv>   Run Fisher discriminant analysis

use std::path::PathBuf;
use std::collections::HashMap;
use image::io::Reader as ImageReader;

use unscan::char_index::{self, compute_features, CharIndex, FEAT_LEN};

const FEAT_NAMES: &[&str] = &[
    "prof0","prof1","prof2","prof3","prof4","prof5","prof6","prof7",
    "prof8","prof9","prof10","prof11","prof12","prof13","prof14","prof15",
    "prof16","prof17","prof18","prof19","prof20","prof21","prof22","prof23",
    "prof24","prof25","prof26","prof27","prof28","prof29","prof30","prof31",
    "aspect","ink_density","v_center","h_balance","serif_score","stroke_contrast","xh_cap_ratio",
    "counter_area","counter_cx","counter_cy","counter_asp",
    "term0","term1","term2","term3",
    "ink_perim","compactness",
    "cross0","cross1","cross2","cross3","cross4","cross5","cross6","cross7",
];

fn dump_index() {
    // Load pre-built index
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let index_path = std::path::PathBuf::from(home).join(".cache").join("unscan").join("char-index.bin");
    eprintln!("Loading index from {:?}...", index_path);
    let index = char_index::load_index(&index_path).expect("failed to load char-index.bin");

    // Print header
    print!("font\tchar");
    for name in FEAT_NAMES {
        print!("\t{}", name);
    }
    println!();

    // Dump raw features from entries
    for (c, entries) in &index.entries {
        for e in entries {
            let raw = e.features.as_slice();
            print!("{}\t{}", e.font_name, c);
            for v in &raw {
                print!("\t{:.6}", v);
            }
            println!();
        }
    }
}

fn dump_scans(diag_dir: &str) {
    // Walk diag-seg output: line_dir/word_NNN_text/chars/NN_c.png
    // Print header
    print!("file\tchar");
    for name in FEAT_NAMES {
        print!("\t{}", name);
    }
    println!();

    let base = PathBuf::from(diag_dir);
    let mut line_dirs: Vec<_> = std::fs::read_dir(&base)
        .expect("cannot read diag dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    line_dirs.sort_by_key(|e| e.file_name());

    for line_entry in &line_dirs {
        let line_dir = line_entry.path();
        let mut word_dirs: Vec<_> = std::fs::read_dir(&line_dir)
            .into_iter().flatten().filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir() && e.file_name().to_str().map_or(false, |s| s.starts_with("word_")))
            .collect();
        word_dirs.sort_by_key(|e| e.file_name());

        for word_entry in &word_dirs {
            let chars_dir = word_entry.path().join("chars");
            if !chars_dir.is_dir() { continue; }
            let mut pngs: Vec<_> = std::fs::read_dir(&chars_dir)
                .into_iter().flatten().filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.ends_with(".png") && !name.contains("_ref")
                })
                .collect();
            pngs.sort_by_key(|e| e.file_name());

            for png_entry in &pngs {
                let path = png_entry.path();
                let fname = png_entry.file_name().to_string_lossy().to_string();
                // Parse char from filename: "02_i.png" -> 'i'
                let label = fname.trim_end_matches(".png");
                let ch = if let Some(idx) = label.find('_') {
                    &label[idx+1..]
                } else {
                    continue;
                };
                // Handle special char names
                let c = match ch {
                    "period" => '.',
                    "comma" => ',',
                    "dash" | "hyphen" => '-',
                    "slash" => '/',
                    "colon" => ':',
                    "semicolon" => ';',
                    "question" => '?',
                    "exclamation" | "bang" => '!',
                    "lparen" => '(',
                    "rparen" => ')',
                    "amp" => '&',
                    "at" => '@',
                    s if s.len() == 1 => s.chars().next().unwrap(),
                    _ => continue,
                };

                let img = match ImageReader::open(&path) {
                    Ok(r) => match r.decode() {
                        Ok(i) => i.to_luma8(),
                        Err(_) => continue,
                    },
                    Err(_) => continue,
                };

                if let Some(feats) = compute_features(&img) {
                    let raw = feats.as_slice();
                    let rel_path = path.strip_prefix(&base).unwrap_or(&path);
                    print!("{}\t{}", rel_path.display(), c);
                    for v in &raw {
                        print!("\t{:.6}", v);
                    }
                    println!();
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  learn_weights --index              Dump index features as TSV");
        eprintln!("  learn_weights --scans <diag_dir>   Dump scan crop features as TSV");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "--index" => dump_index(),
        "--scans" => {
            if args.len() < 3 {
                eprintln!("Need diag-seg directory path");
                std::process::exit(1);
            }
            dump_scans(&args[2]);
        }
        _ => {
            eprintln!("Unknown flag: {}", args[1]);
            std::process::exit(1);
        }
    }
}
