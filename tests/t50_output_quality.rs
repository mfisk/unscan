//! Output quality tests — verify that unscan produces correct, compact output.
//!
//! Checks:
//! - Output file size ≤ input (font subsetting + blank raster elimination)
//! - Visual regression: rasterized output SSIM-matches the rasterized input
//!
//! Test fixtures are auto-generated from EB Garamond 12 (must be installed).
//! The binary must be built before running: `cargo build --release`.

mod common;

use std::path::{Path, PathBuf};

use common::{setup, test_doc, run_unscan_to};

// ── PGM / rasterization helpers ──────────────────────────────────────

/// Rasterize a PDF page to raw grayscale pixels, returning (width, height, pixels).
fn rasterize_pdf_dims(pdf: &Path, dpi: u32) -> Option<(u32, u32, Vec<u8>)> {
    let tmp_prefix = std::env::temp_dir().join(format!("unscan-rast-{}", std::process::id()));
    let pgm_path = PathBuf::from(format!("{}.pgm", tmp_prefix.display()));
    let result = std::process::Command::new("pdftoppm")
        .args(["-gray", "-r", &dpi.to_string(), "-singlefile"])
        .arg(pdf)
        .arg(&tmp_prefix)
        .status();
    match result {
        Err(ref e) => { eprintln!("pdftoppm launch error: {}", e); return None; }
        Ok(ref s) if !s.success() => { eprintln!("pdftoppm exit {:?}", s.code()); return None; }
        _ => {}
    }
    if !pgm_path.exists() {
        eprintln!("pgm not found at {:?}", pgm_path);
        return None;
    }
    let data = std::fs::read(&pgm_path).ok()?;
    let _ = std::fs::remove_file(&pgm_path);
    parse_pgm_dims(&data)
}

/// Parse a PGM (P5) file into (width, height, pixels).
fn parse_pgm_dims(data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    // P5 format: "P5\n<width> <height>\n<maxval>\n<binary pixels>"
    let mut pos = 0usize;
    let mut header_lines: Vec<String> = Vec::new();

    while header_lines.len() < 3 && pos < data.len() {
        // Skip comments
        if data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' { pos += 1; }
            if pos < data.len() { pos += 1; }
            continue;
        }
        // Read a line
        let start = pos;
        while pos < data.len() && data[pos] != b'\n' { pos += 1; }
        if let Ok(line) = std::str::from_utf8(&data[start..pos]) {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                header_lines.push(trimmed.to_string());
            }
        }
        if pos < data.len() { pos += 1; } // skip \n
    }

    if header_lines.len() < 3 || header_lines[0] != "P5" { return None; }
    let dims: Vec<u32> = header_lines[1].split_whitespace()
        .filter_map(|t| t.parse().ok()).collect();
    if dims.len() < 2 { return None; }

    let pixels = data[pos..].to_vec();
    Some((dims[0], dims[1], pixels))
}

/// Compute SSIM between two same-size grayscale images.
/// Simplified SSIM: compares 8x8 blocks, returns mean.
fn compute_ssim(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;
    let block = 8usize;
    let c1: f64 = 6.5025;   // (0.01 * 255)^2
    let c2: f64 = 58.5225;  // (0.03 * 255)^2

    let mut sum = 0.0f64;
    let mut count = 0u64;

    for by in (0..h - block).step_by(block) {
        for bx in (0..w - block).step_by(block) {
            let mut ma = 0.0f64;
            let mut mb = 0.0f64;
            let n = (block * block) as f64;

            for dy in 0..block {
                for dx in 0..block {
                    let idx = (by + dy) * w + (bx + dx);
                    if idx >= a.len() || idx >= b.len() { continue; }
                    ma += a[idx] as f64;
                    mb += b[idx] as f64;
                }
            }
            ma /= n;
            mb /= n;

            let mut va = 0.0f64;
            let mut vb = 0.0f64;
            let mut cov = 0.0f64;
            for dy in 0..block {
                for dx in 0..block {
                    let idx = (by + dy) * w + (bx + dx);
                    if idx >= a.len() || idx >= b.len() { continue; }
                    let da = a[idx] as f64 - ma;
                    let db = b[idx] as f64 - mb;
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= n - 1.0;
            vb /= n - 1.0;
            cov /= n - 1.0;

            let ssim = (2.0 * ma * mb + c1) * (2.0 * cov + c2)
                     / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            sum += ssim;
            count += 1;
        }
    }
    if count == 0 { return 0.0; }
    sum / count as f64
}

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn output_smaller_than_input() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }

    let cases: &[(&str, &[&str])] = &[
        ("punch-hires.pdf", &["--min-font-confidence", "0.0", "--min-ocr-confidence", "0"]),
        ("punch-100dpi-big.pdf", &["--dpi", "100", "--min-font-confidence", "0.0", "--min-ocr-confidence", "0"]),
    ];

    for (name, args) in cases {
        let input = test_doc(name);
        if !input.exists() {
            eprintln!("SKIP: {:?} not found", input);
            continue;
        }

        let out_path = std::env::temp_dir().join(format!("unscan-size-{}", name));
        let _ = run_unscan_to(&input, &out_path, args);

        let in_size = std::fs::metadata(&input).unwrap().len();
        let out_size = std::fs::metadata(&out_path).unwrap().len();
        let _ = std::fs::remove_file(&out_path);

        eprintln!("{}: in={} out={} ratio={:.1}%", name, in_size, out_size, out_size as f64 * 100.0 / in_size as f64);
        assert!(
            out_size <= in_size,
            "{}: output ({}) larger than input ({})",
            name, out_size, in_size
        );
    }
}

#[test]
fn visual_regression_output_matches_input() {
    if !setup() {
        eprintln!("SKIP: fixtures unavailable");
        return;
    }

    let cases: &[(&str, u32, &[&str], f64)] = &[
        ("punch-hires.pdf", 300,
         &["--min-font-confidence", "0.0", "--min-ocr-confidence", "0"],
         0.90),
        ("punch-100dpi-big.pdf", 100,
         &["--dpi", "100", "--min-font-confidence", "0.0", "--min-ocr-confidence", "0"],
         0.85),
    ];

    for (name, dpi, args, threshold) in cases {
        let input = test_doc(name);
        if !input.exists() {
            eprintln!("SKIP: {:?} not found", input);
            continue;
        }

        let out_path = std::env::temp_dir().join(format!("unscan-visual-{}", name));
        let _ = run_unscan_to(&input, &out_path, args);

        // Rasterize both at same DPI
        let (iw, ih, input_px) = rasterize_pdf_dims(&input, *dpi)
            .expect(&format!("failed to rasterize input {}", name));
        let (ow, oh, output_px) = rasterize_pdf_dims(&out_path, *dpi)
            .expect(&format!("failed to rasterize output {}", name));
        let _ = std::fs::remove_file(&out_path);

        // Resize to smaller dimensions if they differ
        let w = iw.min(ow);
        let h = ih.min(oh);

        // Crop both to same size (top-left aligned)
        let crop = |px: &[u8], pw: u32| -> Vec<u8> {
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                let row_start = (y * pw) as usize;
                let row_end = row_start + w as usize;
                if row_end <= px.len() {
                    out.extend_from_slice(&px[row_start..row_end]);
                }
            }
            out
        };
        let a = crop(&input_px, iw);
        let b = crop(&output_px, ow);

        let ssim = compute_ssim(&a, &b, w, h);
        eprintln!("{}: visual SSIM = {:.4} ({}x{} vs {}x{}, compared at {}x{})",
            name, ssim, iw, ih, ow, oh, w, h);
        assert!(
            ssim >= *threshold,
            "{}: visual SSIM {:.4} below threshold {:.2}",
            name, ssim, threshold
        );
    }
}
