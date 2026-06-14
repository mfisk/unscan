#!/usr/bin/env cargo
//! All-in-one triplet network trainer for unscan font classification.
//!
//! Renders glyphs from system fonts, computes features, trains a per-character
//! 3-layer MLP (100→128→64→32) with triplet margin loss, and exports weights
//! in the binary format consumed by TripletClassifier (magic b"TRIP").
//!
//! Zero disk I/O for training data — everything stays in memory.
//!
//! Usage:
//!     train -o triplet-weights.bin
//!     train -o triplet-weights.bin --epochs 50 --max-fonts 100

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ab_glyph::Font;
use clap::Parser;
use image::{GrayImage, Luma};
use rayon::prelude::*;
use rand::prelude::*;
use rand::rngs::SmallRng;

use unscan::char_index::{self, compute_features, FEAT_LEN};
use unscan::font_scan;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "train", about = "Train triplet classifier from system fonts")]
struct Args {
    /// Output weights file path
    #[arg(short, long, default_value = "triplet-weights.bin")]
    output: PathBuf,

    /// Comma-separated native pixel heights to simulate
    #[arg(long, default_value = "48,36,24,18,12,9", value_delimiter = ',')]
    heights: Vec<u32>,

    /// Maximum number of fonts (0 = all)
    #[arg(long, default_value = "0")]
    max_fonts: usize,

    /// Extra font directories to scan
    #[arg(long)]
    font_dir: Vec<PathBuf>,

    /// Training epochs per character
    #[arg(long, default_value = "30")]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Triplet margin
    #[arg(long, default_value = "0.3")]
    margin: f32,

    /// Batch size for triplet sampling
    #[arg(long, default_value = "256")]
    batch_size: usize,

    /// Minimum fonts per character to train (need ≥2 for triplets)
    #[arg(long, default_value = "5")]
    min_fonts: usize,

    /// Directory for temporary per-character feature files (can be large).
    /// Defaults to a subdirectory next to the output file.
    /// Avoid tmpfs/ramdisk mounts — features can reach several GB.
    #[arg(long)]
    tmpdir: Option<PathBuf>,

    /// Fast mode: alternating heights and single AA variant (~6× fewer samples)
    #[arg(long)]
    fast: bool,

    /// Fisher scoring mode: compute per-feature discriminative weights
    /// instead of training a neural network. Reports MRR with and without
    /// Fisher weighting for comparison.
    #[arg(long)]
    fisher: bool,
}

// ---------------------------------------------------------------------------
// Network architecture — matches classifier.rs exactly
// ---------------------------------------------------------------------------

const L1_IN: usize = FEAT_LEN;  // 100
const L1_OUT: usize = 128;
const L2_OUT: usize = 64;
const L3_OUT: usize = 32;

/// Trainable linear layer with Adam optimizer state.
struct Linear {
    rows: usize,
    cols: usize,
    w: Vec<f32>,   // rows × cols, row-major (w[i * cols + j])
    b: Vec<f32>,   // cols
    // Gradients (accumulated, zeroed each step)
    dw: Vec<f32>,
    db: Vec<f32>,
    // Adam moment estimates
    mw: Vec<f32>,
    vw: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Linear {
    fn new(rows: usize, cols: usize, rng: &mut SmallRng) -> Self {
        // Kaiming/He initialization for ReLU layers
        let std_dev = (2.0 / rows as f32).sqrt();
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen::<f32>() * 2.0 * std_dev - std_dev)
            .collect();
        let b = vec![0.0f32; cols];
        let n = rows * cols;
        Self {
            rows, cols, w, b,
            dw: vec![0.0; n],
            db: vec![0.0; cols],
            mw: vec![0.0; n],
            vw: vec![0.0; n],
            mb: vec![0.0; cols],
            vb: vec![0.0; cols],
        }
    }

    /// Forward: output[j] = sum_i(input[i] * w[i*cols+j]) + b[j]
    /// Returns (output, input_clone_for_backprop)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.rows);
        let mut out = self.b.clone();
        for j in 0..self.cols {
            let mut sum = out[j];
            for i in 0..self.rows {
                sum += input[i] * self.w[i * self.cols + j];
            }
            out[j] = sum;
        }
        out
    }

    /// Accumulate gradients: dL/dW and dL/db given dL/dout.
    /// Returns dL/dinput for upstream backprop.
    fn backward(&mut self, input: &[f32], d_out: &[f32]) -> Vec<f32> {
        debug_assert_eq!(d_out.len(), self.cols);
        debug_assert_eq!(input.len(), self.rows);

        // dL/db += d_out
        for j in 0..self.cols {
            self.db[j] += d_out[j];
        }

        // dL/dW[i,j] += input[i] * d_out[j]
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.dw[i * self.cols + j] += input[i] * d_out[j];
            }
        }

        // dL/dinput[i] = sum_j(W[i,j] * d_out[j])
        let mut d_input = vec![0.0f32; self.rows];
        for i in 0..self.rows {
            let mut sum = 0.0f32;
            for j in 0..self.cols {
                sum += self.w[i * self.cols + j] * d_out[j];
            }
            d_input[i] = sum;
        }
        d_input
    }

    /// Adam update step
    fn adam_step(&mut self, lr: f32, t: usize, batch_size: usize) {
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let t_f = t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);
        let scale = 1.0 / batch_size as f32;

        for i in 0..self.w.len() {
            let g = self.dw[i] * scale;
            self.mw[i] = beta1 * self.mw[i] + (1.0 - beta1) * g;
            self.vw[i] = beta2 * self.vw[i] + (1.0 - beta2) * g * g;
            let m_hat = self.mw[i] / bc1;
            let v_hat = self.vw[i] / bc2;
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        for i in 0..self.b.len() {
            let g = self.db[i] * scale;
            self.mb[i] = beta1 * self.mb[i] + (1.0 - beta1) * g;
            self.vb[i] = beta2 * self.vb[i] + (1.0 - beta2) * g * g;
            let m_hat = self.mb[i] / bc1;
            let v_hat = self.vb[i] / bc2;
            self.b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Zero gradients
        self.dw.fill(0.0);
        self.db.fill(0.0);
    }
}

/// Per-character trainable network.
struct TrainableNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl TrainableNet {
    fn new(rng: &mut SmallRng) -> Self {
        Self {
            fc1: Linear::new(L1_IN, L1_OUT, rng),
            fc2: Linear::new(L1_OUT, L2_OUT, rng),
            fc3: Linear::new(L2_OUT, L3_OUT, rng),
        }
    }

    /// Forward pass with cached activations for backprop.
    fn forward(&self, input: &[f32]) -> ForwardCache {
        let z1 = self.fc1.forward(input);
        let h1: Vec<f32> = z1.iter().map(|&x| x.max(0.0)).collect();

        let z2 = self.fc2.forward(&h1);
        let h2: Vec<f32> = z2.iter().map(|&x| x.max(0.0)).collect();

        let z3 = self.fc3.forward(&h2);

        // L2 normalize
        let norm: f32 = z3.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        let out: Vec<f32> = z3.iter().map(|x| x / norm).collect();

        ForwardCache {
            input: input.to_vec(),
            z1, h1, z2, h2, z3, norm, out,
        }
    }

    /// Backward pass given dL/d(output).
    fn backward(&mut self, cache: &ForwardCache, d_out: &[f32]) {
        // Backprop through L2 normalization
        // d(x/||x||)/dx = (I - x*x^T/||x||^2) / ||x||
        let d_z3 = l2_norm_backward(&cache.z3, cache.norm, d_out);

        // Backprop through fc3 (linear, no activation)
        let d_h2 = self.fc3.backward(&cache.h2, &d_z3);

        // Backprop through ReLU on layer 2
        let d_z2: Vec<f32> = d_h2.iter().zip(cache.z2.iter())
            .map(|(&dh, &z)| if z > 0.0 { dh } else { 0.0 })
            .collect();

        let d_h1 = self.fc2.backward(&cache.h1, &d_z2);

        // Backprop through ReLU on layer 1
        let d_z1: Vec<f32> = d_h1.iter().zip(cache.z1.iter())
            .map(|(&dh, &z)| if z > 0.0 { dh } else { 0.0 })
            .collect();

        let _ = self.fc1.backward(&cache.input, &d_z1);
    }

    fn adam_step(&mut self, lr: f32, t: usize, batch_size: usize) {
        self.fc1.adam_step(lr, t, batch_size);
        self.fc2.adam_step(lr, t, batch_size);
        self.fc3.adam_step(lr, t, batch_size);
    }
}

struct ForwardCache {
    input: Vec<f32>,
    z1: Vec<f32>,
    h1: Vec<f32>,
    z2: Vec<f32>,
    h2: Vec<f32>,
    z3: Vec<f32>,
    norm: f32,
    out: Vec<f32>,
}

/// Gradient of L2 normalization: d(x/||x||)/dx applied to upstream gradient.
fn l2_norm_backward(z: &[f32], norm: f32, d_out: &[f32]) -> Vec<f32> {
    let n = z.len();
    let norm_sq = norm * norm;
    let dot: f32 = z.iter().zip(d_out.iter()).map(|(a, b)| a * b).sum();
    let mut grad = vec![0.0f32; n];
    for i in 0..n {
        // (d_out[i] - z[i] * dot / norm_sq) / norm
        grad[i] = (d_out[i] - z[i] * dot / norm_sq) / norm;
    }
    grad
}

/// Euclidean distance between two vectors.
fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ---------------------------------------------------------------------------
// AA variants — copied from gen_training_data.rs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum AaVariant { Native, Blur05, Sharpen }

impl AaVariant {
    fn apply(&self, img: &GrayImage) -> GrayImage {
        match self {
            Self::Native => img.clone(),
            Self::Blur05 => image::imageops::blur(img, 0.5),
            Self::Sharpen => contrast_stretch(img),
        }
    }
}

fn contrast_stretch(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y).0[0] as f32;
            let norm = (px - 128.0) / 128.0;
            let stretched = norm * (1.0 + 0.5 * norm.abs());
            let val = ((stretched * 128.0 + 128.0).clamp(0.0, 255.0)) as u8;
            out.put_pixel(x, y, Luma([val]));
        }
    }
    out
}

fn render_char_at_native_height<F: Font>(
    font: &F, c: char, target_height: u32, aa: AaVariant,
    overrides: Option<&[(char, u16)]>,
) -> Option<GrayImage> {
    let gid = char_index::resolve_glyph(font, c, overrides);
    let full = char_index::render_glyph_normalised(font, gid)?;
    let full = aa.apply(&full);

    if target_height >= 48 {
        return char_index::normalize_to_ink_bounds(&full);
    }

    let (fw, fh) = full.dimensions();
    if fh == 0 || fw == 0 { return None; }

    let scale = target_height as f32 / fh as f32;
    let new_w = ((fw as f32 * scale).round() as u32).max(1);
    let new_h = target_height;

    let small = image::imageops::resize(&full, new_w, new_h, image::imageops::FilterType::Lanczos3);
    char_index::normalize_to_ink_bounds(&small)
}

// ---------------------------------------------------------------------------
// Training data structure
// ---------------------------------------------------------------------------

struct TrainingSample {
    font_id: u32,     // compact font index for triplet mining
    features: [f32; FEAT_LEN],
}

/// Load per-character samples from multiple (height, aa) combo files.
fn load_char_combo_samples(
    feat_dir: &std::path::Path,
    ci: usize,
    combos: &[(u32, usize, Vec<usize>)], // (ht, aa_idx, per-char counts)
) -> Vec<TrainingSample> {
    let total: usize = combos.iter().map(|(_, _, counts)| counts[ci]).sum();
    let mut samples = Vec::with_capacity(total);
    let mut buf4 = [0u8; 4];
    for (ht, aa_idx, counts) in combos {
        let n = counts[ci];
        if n == 0 { continue; }
        let path = feat_dir.join(format!("char_{:04}_h{}_aa{}.bin", ci, ht, aa_idx));
        let file = std::fs::File::open(&path).expect("open combo feature file");
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        for _ in 0..n {
            reader.read_exact(&mut buf4).expect("read font_id");
            let font_id = u32::from_le_bytes(buf4);
            let mut features = [0.0f32; FEAT_LEN];
            for f in &mut features {
                reader.read_exact(&mut buf4).expect("read feature");
                *f = f32::from_le_bytes(buf4);
            }
            samples.push(TrainingSample { font_id, features });
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // In --fast mode: alternating heights, single AA variant
    let active_heights: Vec<u32> = if args.fast {
        args.heights.iter().step_by(2).copied().collect()
    } else {
        args.heights.clone()
    };
    let aa_variants: Vec<AaVariant> = if args.fast {
        vec![AaVariant::Native]
    } else {
        vec![AaVariant::Native, AaVariant::Blur05, AaVariant::Sharpen]
    };

    eprintln!("=== unscan all-in-one triplet trainer ===");
    eprintln!("Heights: {:?}{}", active_heights, if args.fast { " (fast)" } else { "" });
    eprintln!("AA variants: {}{}", aa_variants.len(), if args.fast { " (fast)" } else { "" });
    eprintln!("Epochs: {}", args.epochs);
    eprintln!("Learning rate: {}", args.lr);
    eprintln!("Margin: {}", args.margin);

    // ── 1. Scan fonts ─────────────────────────────────────────────
    let font_dirs: Vec<PathBuf> = font_scan::default_font_dirs(&args.font_dir);

    let mut catalog = font_scan::scan_fonts(&font_dirs);
    eprintln!("  {} font entries found", catalog.len());

    if args.max_fonts > 0 && catalog.len() > args.max_fonts {
        catalog.truncate(args.max_fonts);
        eprintln!("  Limiting to {} fonts (--max-fonts)", args.max_fonts);
    }

    let chars: &[char] = char_index::indexed_chars();
    eprintln!("  {} indexed characters", chars.len());

    // ── 2. Render & extract features ──────────────────────────────
    // Write per-char binary feature files to disk to avoid OOM.
    // Each file: sequence of (font_id: u32, features: [f32; 100]) = 404 bytes/sample.
    // With ~10M samples × 404 bytes ≈ 4 GB on disk, but only ~38 MB per char in memory.

    let total_fonts = catalog.len();
    let progress = AtomicUsize::new(0);
    let start = std::time::Instant::now();

    // Assign each unique font_key a compact integer ID
    let font_id_map: HashMap<String, u32> = catalog.iter().enumerate()
        .map(|(i, fe)| (fe.font_key(), i as u32))
        .collect();

    // Build family groups: font_id → family_id, family_id → Vec<font_id>
    let mut family_name_to_id: HashMap<&str, u32> = HashMap::new();
    let mut font_family: Vec<u32> = Vec::with_capacity(catalog.len()); // font_id → family_id
    let mut family_members: Vec<Vec<u32>> = Vec::new(); // family_id → [font_ids]
    for (i, fe) in catalog.iter().enumerate() {
        let fam_name = fe.family_name.as_str();
        let fam_id = if let Some(&id) = family_name_to_id.get(fam_name) {
            id
        } else {
            let id = family_members.len() as u32;
            family_name_to_id.insert(fam_name, id);
            family_members.push(Vec::new());
            id
        };
        font_family.push(fam_id);
        family_members[fam_id as usize].push(i as u32);
    }
    let n_families = family_members.len();
    let multi_variant_families = family_members.iter().filter(|m| m.len() > 1).count();
    eprintln!("  {} font families ({} with multiple variants)",
        n_families, multi_variant_families);

    let chunk_size = 200;
    let n_chars = chars.len();

    let char_to_idx: HashMap<char, usize> = chars.iter().enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // Create temp directory for per-char feature files.
    // Default: subdirectory next to the output file (avoids tmpfs/ramdisk OOM).
    let feat_dir = match &args.tmpdir {
        Some(d) => d.join(".train_feat_tmp"),
        None => {
            let parent = args.output.parent()
                .filter(|p| !p.as_os_str().is_empty() && *p != std::path::Path::new("/dev"))
                .unwrap_or(std::path::Path::new("."));
            parent.join(".train_feat_tmp")
        }
    };
    std::fs::create_dir_all(&feat_dir).expect("create feature temp dir");

    // ── Determine which (height, aa) combos need rendering ───────
    // All possible combos (always render the full set so fast/normal share files)
    let all_heights: &[u32] = &args.heights;
    let all_aa: &[AaVariant] = &[AaVariant::Native, AaVariant::Blur05, AaVariant::Sharpen];

    /// File path for a (char, height, aa) feature file.
    fn combo_path(feat_dir: &std::path::Path, ci: usize, ht: u32, aa_idx: usize) -> std::path::PathBuf {
        feat_dir.join(format!("char_{:04}_h{}_aa{}.bin", ci, ht, aa_idx))
    }

    /// Manifest path for a (height, aa) combo.
    fn manifest_combo_path(feat_dir: &std::path::Path, ht: u32, aa_idx: usize) -> std::path::PathBuf {
        feat_dir.join(format!("manifest_h{}_aa{}.txt", ht, aa_idx))
    }

    // A combo is cached if its manifest exists, matches the font count, and all char files exist.
    let combo_cached = |ht: u32, aa_idx: usize| -> Option<Vec<usize>> {
        let mpath = manifest_combo_path(&feat_dir, ht, aa_idx);
        let content = std::fs::read_to_string(&mpath).ok()?;
        let mut lines = content.lines();
        let header = lines.next()?;
        if header.trim() != format!("fonts={} chars={}", catalog.len(), n_chars) {
            return None;
        }
        let mut counts = Vec::with_capacity(n_chars);
        for line in lines {
            counts.push(line.trim().parse::<usize>().ok()?);
        }
        if counts.len() != n_chars { return None; }
        // Verify all files exist
        for ci in 0..n_chars {
            if !combo_path(&feat_dir, ci, ht, aa_idx).exists() { return None; }
        }
        Some(counts)
    };

    // Check which combos the current mode needs
    let mut needed_combos: Vec<(u32, usize)> = Vec::new(); // (height, aa_idx)
    let mut cached_combos: Vec<(u32, usize, Vec<usize>)> = Vec::new(); // (height, aa_idx, counts)

    for &ht in &active_heights {
        for (aa_idx, _) in aa_variants.iter().enumerate() {
            match combo_cached(ht, aa_idx) {
                Some(counts) => cached_combos.push((ht, aa_idx, counts)),
                None => needed_combos.push((ht, aa_idx)),
            }
        }
    }

    if needed_combos.is_empty() {
        let total_cached: usize = cached_combos.iter()
            .flat_map(|(_, _, c)| c.iter())
            .sum();
        eprintln!("\nReusing {} cached combos ({} samples)",
            cached_combos.len(), total_cached);
    } else {
        eprintln!("\nRendering {} combos ({} cached)...",
            needed_combos.len(), cached_combos.len());

        // Build a set of needed combos for the inner loop
        let needed_set: std::collections::HashSet<(u32, usize)> = needed_combos.iter().copied().collect();

        // Open writers: indexed by (ci, combo_index)
        // combo_index is position in needed_combos
        let n_combos = needed_combos.len();
        let mut combo_writers: Vec<Vec<BufWriter<std::fs::File>>> = (0..n_chars).map(|ci| {
            needed_combos.iter().map(|&(ht, aa_idx)| {
                let path = combo_path(&feat_dir, ci, ht, aa_idx);
                BufWriter::with_capacity(
                    64 * 1024,
                    std::fs::File::create(&path).expect("create combo feature file"),
                )
            }).collect()
        }).collect();

        // Per-combo per-char counts
        let mut combo_counts: Vec<Vec<usize>> = vec![vec![0usize; n_chars]; n_combos];

        for chunk_start in (0..catalog.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(catalog.len());
            let chunk = &catalog[chunk_start..chunk_end];

            // Each font produces samples tagged with (ci, combo_index)
            let chunk_results: Vec<Vec<(usize, usize, TrainingSample)>> = chunk.par_iter().map(|fe| {
                let font_data = match std::fs::read(&fe.path) {
                    Ok(d) => d,
                    Err(_) => {
                        progress.fetch_add(1, Ordering::Relaxed);
                        return Vec::new();
                    }
                };
                let font = match ab_glyph::FontRef::try_from_slice(&font_data) {
                    Ok(f) => f,
                    Err(_) => {
                        progress.fetch_add(1, Ordering::Relaxed);
                        return Vec::new();
                    }
                };

                let font_id = font_id_map[&fe.font_key()];
                let overrides = fe.glyph_overrides.as_deref();
                let mut samples = Vec::new();

                for &c in chars {
                    let ci = char_to_idx[&c];

                    let gid = char_index::resolve_glyph(&font, c, overrides);
                    let full = match char_index::render_glyph_normalised(&font, gid) {
                        Some(img) => img,
                        None => continue,
                    };

                    for (aa_idx_all, &aa) in all_aa.iter().enumerate() {
                        // Only process AA variants we need
                        let aa_needed = needed_combos.iter().any(|&(_, ai)| ai == aa_idx_all);
                        if !aa_needed { continue; }

                        let aa_full = aa.apply(&full);
                        let (fw, fh) = aa_full.dimensions();
                        if fh == 0 || fw == 0 { continue; }

                        let full_normed = char_index::normalize_to_ink_bounds(&aa_full);

                        for &ht in all_heights {
                            if !needed_set.contains(&(ht, aa_idx_all)) { continue; }

                            let combo_idx = needed_combos.iter()
                                .position(|&(h, a)| h == ht && a == aa_idx_all)
                                .unwrap();

                            let img = if ht >= 48 {
                                match full_normed {
                                    Some(ref img) => img.clone(),
                                    None => continue,
                                }
                            } else {
                                let scale = ht as f32 / fh as f32;
                                let new_w = ((fw as f32 * scale).round() as u32).max(1);
                                let small = image::imageops::resize(
                                    &aa_full, new_w, ht,
                                    image::imageops::FilterType::Lanczos3,
                                );
                                match char_index::normalize_to_ink_bounds(&small) {
                                    Some(img) => img,
                                    None => continue,
                                }
                            };

                            let feats = match compute_features(&img) {
                                Some(f) => f,
                                None => continue,
                            };

                            samples.push((ci, combo_idx, TrainingSample {
                                font_id,
                                features: feats.as_slice(),
                            }));
                        }
                    }
                }

                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 200 == 0 || done == total_fonts {
                    eprintln!("  [{}/{}] fonts rendered...", done, total_fonts);
                }

                samples
            }).collect();

            // Write chunk results to per-combo per-char files
            for font_samples in chunk_results {
                for (ci, combo_idx, sample) in font_samples {
                    let w = &mut combo_writers[ci][combo_idx];
                    w.write_all(&sample.font_id.to_le_bytes()).expect("write font_id");
                    for &f in &sample.features {
                        w.write_all(&f.to_le_bytes()).expect("write feature");
                    }
                    combo_counts[combo_idx][ci] += 1;
                }
            }
        }

        // Flush and close all writers
        for char_ws in &mut combo_writers {
            for w in char_ws {
                w.flush().expect("flush combo features");
            }
        }
        drop(combo_writers);

        // Write per-combo manifests
        for (combo_idx, &(ht, aa_idx)) in needed_combos.iter().enumerate() {
            let mpath = manifest_combo_path(&feat_dir, ht, aa_idx);
            let mut manifest = format!("fonts={} chars={}", catalog.len(), n_chars);
            for ci in 0..n_chars {
                manifest.push('\n');
                manifest.push_str(&combo_counts[combo_idx][ci].to_string());
            }
            std::fs::write(&mpath, &manifest).expect("write combo manifest");
        }

        let rendered_samples: usize = combo_counts.iter().flat_map(|c| c.iter()).sum();
        let render_secs_inner = start.elapsed().as_secs_f64();
        eprintln!("\nRendering complete: {} new samples in {:.1}s", rendered_samples, render_secs_inner);

        // Refresh cached_combos with freshly rendered ones
        for &(ht, aa_idx) in &needed_combos {
            if let Some(counts) = combo_cached(ht, aa_idx) {
                cached_combos.push((ht, aa_idx, counts));
            }
        }
    }

    // ── Load aggregate char_counts from all active combos ────────
    let mut char_counts = vec![0usize; n_chars];
    for (_, _, ref counts) in &cached_combos {
        for ci in 0..n_chars {
            char_counts[ci] += counts[ci];
        }
    }
    let total_samples: usize = char_counts.iter().sum();
    eprintln!("Total samples for evaluation: {} ({} combos)", total_samples, cached_combos.len());

    let render_secs = start.elapsed().as_secs_f64();

    // ── 3. Fisher scoring mode ────────────────────────────────────
    if args.fisher {
        eprintln!("\nFisher scoring {} characters...", chars.len());
        let fisher_start = std::time::Instant::now();
        let mut fisher_chars: Vec<(char, [f32; FEAT_LEN])> = Vec::new();
        let mut skipped = 0usize;

        // Accumulators for baseline (uniform) and Fisher-weighted MRR
        // Strict: each font_id is independent
        // Family: best rank among same-family variants counts
        let mut base_rr = 0.0f64;
        let mut base_top1 = 0usize;
        let mut fish_rr = 0.0f64;
        let mut fish_top1 = 0usize;
        let mut fish_top5 = 0usize;
        let mut fish_fam_rr = 0.0f64;
        let mut fish_fam_top1 = 0usize;
        let mut fish_fam_top5 = 0usize;
        let mut total_eval = 0usize;

        for (ci, &c) in chars.iter().enumerate() {
            let n_samples = char_counts[ci];
            if n_samples == 0 { skipped += 1; continue; }

            let samples = load_char_combo_samples(&feat_dir, ci, &cached_combos);

            // Group by font
            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.font_id).or_default().push(i);
            }
            if font_indices.len() < args.min_fonts.max(2) { skipped += 1; continue; }

            let n = samples.len();

            // ── Per-feature Fisher score ──
            // Global mean
            let mut global_mean = [0.0f64; FEAT_LEN];
            for s in &samples {
                for j in 0..FEAT_LEN { global_mean[j] += s.features[j] as f64; }
            }
            for j in 0..FEAT_LEN { global_mean[j] /= n as f64; }

            // Class means
            let class_means: HashMap<u32, Vec<f64>> = font_indices.iter().map(|(&fid, indices)| {
                let mut mean = vec![0.0f64; FEAT_LEN];
                for &i in indices {
                    for j in 0..FEAT_LEN { mean[j] += samples[i].features[j] as f64; }
                }
                let cnt = indices.len() as f64;
                for j in 0..FEAT_LEN { mean[j] /= cnt; }
                (fid, mean)
            }).collect();

            // Between-class variance: Σ_k n_k (μ_k - μ)² / N
            let mut var_between = [0.0f64; FEAT_LEN];
            for (&fid, indices) in &font_indices {
                let nk = indices.len() as f64;
                let cm = &class_means[&fid];
                for j in 0..FEAT_LEN {
                    let d = cm[j] - global_mean[j];
                    var_between[j] += nk * d * d;
                }
            }
            for j in 0..FEAT_LEN { var_between[j] /= n as f64; }

            // Within-class variance: Σ_k Σ_{i∈k} (x_i - μ_k)² / N
            let mut var_within = [0.0f64; FEAT_LEN];
            for (&fid, indices) in &font_indices {
                let cm = &class_means[&fid];
                for &i in indices {
                    for j in 0..FEAT_LEN {
                        let d = samples[i].features[j] as f64 - cm[j];
                        var_within[j] += d * d;
                    }
                }
            }
            for j in 0..FEAT_LEN { var_within[j] /= n as f64; }

            // Fisher score per feature: F_j = var_between / var_within
            let mut scores = [0.0f32; FEAT_LEN];
            for j in 0..FEAT_LEN {
                scores[j] = if var_within[j] > 1e-12 {
                    (var_between[j] / var_within[j]) as f32
                } else if var_between[j] > 1e-12 {
                    // Zero within-class variance but nonzero between = perfectly discriminative
                    f32::MAX
                } else {
                    0.0 // constant feature, useless
                };
            }

            fisher_chars.push((c, scores));

            // ── Evaluate: baseline (uniform) vs Fisher-weighted MRR ──
            // Subsample if too many samples (MRR estimate stabilizes well below full n)
            let max_eval = 2000usize;
            let eval_indices: Vec<usize> = if n <= max_eval {
                (0..n).collect()
            } else {
                let mut rng = SmallRng::seed_from_u64(c as u64);
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng);
                idx.truncate(max_eval);
                idx
            };
            let n_eval = eval_indices.len();

            // Flatten centroids for cache-friendly access
            let centroid_fids: Vec<u32> = class_means.keys().copied().collect();
            let centroid_feats: Vec<&Vec<f64>> = centroid_fids.iter()
                .map(|fid| &class_means[fid])
                .collect();
            let k = centroid_fids.len();

            // Precompute family membership for each centroid
            let centroid_famid: Vec<u32> = centroid_fids.iter()
                .map(|&fid| font_family[fid as usize])
                .collect();

            let mut c_base_rr = 0.0f64;
            let mut c_base_top1 = 0usize;
            let mut c_fish_rr = 0.0f64;
            let mut c_fish_top1 = 0usize;
            let mut c_fish_top5 = 0usize;
            let mut c_fish_fam_rr = 0.0f64;
            let mut c_fish_fam_top1 = 0usize;
            let mut c_fish_fam_top5 = 0usize;

            for &i in &eval_indices {
                let correct = samples[i].font_id;
                let correct_famid = font_family[correct as usize];

                // Compute Fisher distance to every centroid
                let mut fish_dists: Vec<(u32, f64)> = Vec::with_capacity(k);
                let mut d_base_correct = 0.0f64;
                for ci in 0..k {
                    let mut d_fish = 0.0f64;
                    for j in 0..FEAT_LEN {
                        let diff = samples[i].features[j] as f64 - centroid_feats[ci][j];
                        d_fish += scores[j] as f64 * diff * diff;
                    }
                    fish_dists.push((centroid_fids[ci], d_fish));

                    // Baseline distance for the correct font only
                    if centroid_fids[ci] == correct {
                        let mut d = 0.0f64;
                        for j in 0..FEAT_LEN {
                            let diff = samples[i].features[j] as f64 - centroid_feats[ci][j];
                            d += diff * diff;
                        }
                        d_base_correct = d;
                    }
                }

                // Baseline: count how many are closer (unweighted)
                let mut base_rank = 0usize;
                for ci in 0..k {
                    if centroid_fids[ci] == correct { continue; }
                    let mut d = 0.0f64;
                    for j in 0..FEAT_LEN {
                        let diff = samples[i].features[j] as f64 - centroid_feats[ci][j];
                        d += diff * diff;
                    }
                    if d < d_base_correct { base_rank += 1; }
                }
                c_base_rr += 1.0 / (base_rank as f64 + 1.0);
                if base_rank == 0 { c_base_top1 += 1; }

                // Strict Fisher: rank of exact font_id
                let d_fish_correct = fish_dists.iter()
                    .find(|&&(fid, _)| fid == correct).unwrap().1;
                let fish_rank = fish_dists.iter()
                    .filter(|&&(fid, d)| fid != correct && d < d_fish_correct)
                    .count();
                c_fish_rr += 1.0 / (fish_rank as f64 + 1.0);
                if fish_rank == 0 { c_fish_top1 += 1; }
                if fish_rank < 5 { c_fish_top5 += 1; }

                // Family Fisher: best rank among any font in same family
                let best_fam_dist = fish_dists.iter()
                    .filter(|&&(fid, _)| font_family[fid as usize] == correct_famid)
                    .map(|&(_, d)| d)
                    .fold(f64::MAX, f64::min);
                let fam_rank = fish_dists.iter()
                    .filter(|&&(fid, d)| {
                        font_family[fid as usize] != correct_famid && d < best_fam_dist
                    })
                    .count();
                c_fish_fam_rr += 1.0 / (fam_rank as f64 + 1.0);
                if fam_rank == 0 { c_fish_fam_top1 += 1; }
                if fam_rank < 5 { c_fish_fam_top5 += 1; }
            }

            base_rr += c_base_rr;
            base_top1 += c_base_top1;
            fish_rr += c_fish_rr;
            fish_top1 += c_fish_top1;
            fish_top5 += c_fish_top5;
            fish_fam_rr += c_fish_fam_rr;
            fish_fam_top1 += c_fish_fam_top1;
            fish_fam_top5 += c_fish_fam_top5;
            total_eval += n_eval;

            if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                eprintln!("  char '{}' base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    c,
                    c_base_rr / n_eval as f64,
                    c_fish_rr / n_eval as f64,
                    c_fish_top1 as f64 / n_eval as f64 * 100.0,
                    c_fish_fam_rr / n_eval as f64,
                    c_fish_fam_top1 as f64 / n_eval as f64 * 100.0);
            }
        }

        let fisher_elapsed = fisher_start.elapsed();

        let b_mrr = if total_eval > 0 { base_rr / total_eval as f64 } else { 0.0 };
        let b_top1 = if total_eval > 0 { base_top1 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        let f_mrr = if total_eval > 0 { fish_rr / total_eval as f64 } else { 0.0 };
        let f_top1 = if total_eval > 0 { fish_top1 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        let f_top5 = if total_eval > 0 { fish_top5 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        let ff_mrr = if total_eval > 0 { fish_fam_rr / total_eval as f64 } else { 0.0 };
        let ff_top1 = if total_eval > 0 { fish_fam_top1 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        let ff_top5 = if total_eval > 0 { fish_fam_top5 as f64 / total_eval as f64 * 100.0 } else { 0.0 };

        eprintln!("\nFisher scoring complete: {} characters, {} skipped, {:.1}s",
            fisher_chars.len(), skipped, fisher_elapsed.as_secs_f64());
        eprintln!("  Baseline:       MRR={:.3} top1={:.1}%", b_mrr, b_top1);
        eprintln!("  Fisher strict:  MRR={:.3} top1={:.1}% top5={:.1}%", f_mrr, f_top1, f_top5);
        eprintln!("  Fisher family:  MRR={:.3} top1={:.1}% top5={:.1}%", ff_mrr, ff_top1, ff_top5);
        eprintln!("  ({} families, {} multi-variant)", n_families, multi_variant_families);

        // ── Export Fisher weights ─────────────────────────────────
        eprintln!("Writing Fisher weights to {}...", args.output.display());
        let f = std::fs::File::create(&args.output).expect("create output file");
        let mut w = BufWriter::new(f);

        // Header: magic + version + n_chars + feat_len
        w.write_all(b"FISH").unwrap();
        w.write_all(&1u32.to_le_bytes()).unwrap();
        w.write_all(&(fisher_chars.len() as u32).to_le_bytes()).unwrap();
        w.write_all(&(FEAT_LEN as u32).to_le_bytes()).unwrap();

        for (c, scores) in &fisher_chars {
            w.write_all(&(*c as u32).to_le_bytes()).unwrap();
            for &v in scores { w.write_all(&v.to_le_bytes()).unwrap(); }
        }
        w.flush().unwrap();

        let file_size = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
        let total_elapsed = start.elapsed();

        eprintln!("\n=== Fisher scoring complete ===");
        eprintln!("  Characters: {}/{}", fisher_chars.len(), chars.len());
        eprintln!("  Families:   {} ({} multi-variant)", n_families, multi_variant_families);
        eprintln!("  Weights:    {} ({:.1} KB)", args.output.display(), file_size as f64 / 1e3);
        eprintln!("  Baseline:       MRR={:.3} top1={:.1}%", b_mrr, b_top1);
        eprintln!("  Fisher strict:  MRR={:.3} top1={:.1}% top5={:.1}%", f_mrr, f_top1, f_top5);
        eprintln!("  Fisher family:  MRR={:.3} top1={:.1}% top5={:.1}%", ff_mrr, ff_top1, ff_top5);
        eprintln!("  Render:     {:.1}s", render_secs);
        eprintln!("  Score:      {:.1}s", fisher_elapsed.as_secs_f64());
        eprintln!("  Total:      {:.1}s", total_elapsed.as_secs_f64());

        return;
    }

    // ── 3. Train per-character networks ───────────────────────────
    eprintln!("\nTraining {} characters (epochs={}, lr={}, margin={})...",
        chars.len(), args.epochs, args.lr, args.margin);

    let train_start = std::time::Instant::now();
    let mut trained_chars: Vec<(char, TrainableNet)> = Vec::new();
    let mut skipped = 0usize;
    let mut total_rr_sum = 0.0f64;   // sum of reciprocal ranks
    let mut total_top1 = 0usize;
    let mut total_top5 = 0usize;
    let mut total_eval = 0usize;

    for (ci, &c) in chars.iter().enumerate() {
        // Load this character's samples from disk
        let n_samples = char_counts[ci];
        if n_samples == 0 {
            skipped += 1;
            continue;
        }

        let samples = load_char_combo_samples(&feat_dir, ci, &cached_combos);

        // Count unique fonts
        let mut font_set: Vec<u32> = samples.iter().map(|s| s.font_id).collect();
        font_set.sort_unstable();
        font_set.dedup();

        if font_set.len() < args.min_fonts.max(2) {
            skipped += 1;
            continue;
        }

        let mut rng = SmallRng::seed_from_u64(c as u64);
        let mut net = TrainableNet::new(&mut rng);

        // Build font_id → sample indices map
        let mut font_samples: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, s) in samples.iter().enumerate() {
            font_samples.entry(s.font_id).or_default().push(i);
        }
        let font_ids: Vec<u32> = font_samples.keys().copied().collect();

        let mut adam_t = 0usize;

        for epoch in 0..args.epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_triplets = 0usize;

            // Sample batch_size triplets
            for _ in 0..args.batch_size {
                // Pick anchor font, then a different negative font
                let anchor_font = font_ids[rng.gen_range(0..font_ids.len())];
                let anchor_samples = &font_samples[&anchor_font];
                if anchor_samples.len() < 2 { continue; }

                // Anchor and positive from same font
                let ai = anchor_samples[rng.gen_range(0..anchor_samples.len())];
                let pi = anchor_samples[rng.gen_range(0..anchor_samples.len())];
                if ai == pi { continue; } // need different samples

                // Negative from different font
                let neg_font = loop {
                    let f = font_ids[rng.gen_range(0..font_ids.len())];
                    if f != anchor_font { break f; }
                };
                let neg_samples = &font_samples[&neg_font];
                let ni = neg_samples[rng.gen_range(0..neg_samples.len())];

                // Forward all three
                let a_cache = net.forward(&samples[ai].features);
                let p_cache = net.forward(&samples[pi].features);
                let n_cache = net.forward(&samples[ni].features);

                // Triplet margin loss: max(0, d(a,p) - d(a,n) + margin)
                let dp = dist_sq(&a_cache.out, &p_cache.out);
                let dn = dist_sq(&a_cache.out, &n_cache.out);
                let loss = (dp - dn + args.margin).max(0.0);

                if loss > 0.0 {
                    epoch_loss += loss;
                    n_triplets += 1;

                    // Gradient of triplet loss w.r.t. embeddings
                    // L = d(a,p)^2 - d(a,n)^2 + margin
                    // dL/d_a = 2(a-p) - 2(a-n)
                    // dL/d_p = 2(p-a)
                    // dL/d_n = -2(n-a) = 2(a-n)  [note: minus sign from -d(a,n)^2]
                    let mut d_a = vec![0.0f32; L3_OUT];
                    let mut d_p = vec![0.0f32; L3_OUT];
                    let mut d_n = vec![0.0f32; L3_OUT];
                    for j in 0..L3_OUT {
                        d_a[j] = 2.0 * (a_cache.out[j] - p_cache.out[j])
                               - 2.0 * (a_cache.out[j] - n_cache.out[j]);
                        d_p[j] = 2.0 * (p_cache.out[j] - a_cache.out[j]);
                        d_n[j] = 2.0 * (a_cache.out[j] - n_cache.out[j]);
                    }

                    // Backward all three (accumulates gradients)
                    net.backward(&a_cache, &d_a);
                    net.backward(&p_cache, &d_p);
                    net.backward(&n_cache, &d_n);
                }
            }

            // Adam step
            adam_t += 1;
            let effective_batch = n_triplets.max(1);
            net.adam_step(args.lr, adam_t, effective_batch);

            if (epoch + 1) % 10 == 0 || epoch == 0 {
                let avg_loss = if n_triplets > 0 { epoch_loss / n_triplets as f32 } else { 0.0 };
                if ci < 5 || ci == chars.len() - 1 {
                    eprintln!("  char '{}' epoch {}/{}: loss={:.4} ({} active triplets)",
                        c, epoch + 1, args.epochs, avg_loss, n_triplets);
                }
            }
        }

        trained_chars.push((c, net));

        // ── Retrieval quality: MRR + top-k via nearest-centroid ──
        if let Some((_, ref trained_net)) = trained_chars.last() {
            let embeddings: Vec<Vec<f32>> = samples.iter()
                .map(|s| trained_net.forward(&s.features).out)
                .collect();

            // Compute per-font centroids
            let mut centroid_sums: HashMap<u32, Vec<f32>> = HashMap::new();
            let mut centroid_counts: HashMap<u32, usize> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                let entry = centroid_sums.entry(s.font_id).or_insert_with(|| vec![0.0; L3_OUT]);
                for (j, &v) in embeddings[i].iter().enumerate() {
                    entry[j] += v;
                }
                *centroid_counts.entry(s.font_id).or_insert(0) += 1;
            }
            let centroid_fids: Vec<u32> = centroid_sums.keys().copied().collect();
            let centroid_vecs: Vec<Vec<f32>> = centroid_fids.iter().map(|fid| {
                let mut v = centroid_sums.remove(fid).unwrap();
                let cnt = centroid_counts[fid] as f32;
                for x in &mut v { *x /= cnt; }
                v
            }).collect();
            let k = centroid_fids.len();

            let n = embeddings.len();
            // Subsample for eval if too many
            let max_eval = 2000usize;
            let eval_indices: Vec<usize> = if n <= max_eval {
                (0..n).collect()
            } else {
                let mut rng = SmallRng::seed_from_u64(c as u64 + 0x1234);
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng);
                idx.truncate(max_eval);
                idx
            };
            let n_eval = eval_indices.len();

            let mut char_rr_sum = 0.0f64;
            let mut char_top1 = 0usize;
            let mut char_top5 = 0usize;
            for &i in &eval_indices {
                let correct_font = samples[i].font_id;
                let ci_pos = centroid_fids.iter().position(|&f| f == correct_font).unwrap();
                let d_correct = dist_sq(&embeddings[i], &centroid_vecs[ci_pos]);

                let mut rank = 0usize;
                for ci in 0..k {
                    if centroid_fids[ci] == correct_font { continue; }
                    if dist_sq(&embeddings[i], &centroid_vecs[ci]) < d_correct {
                        rank += 1;
                    }
                }
                char_rr_sum += 1.0 / (rank as f64 + 1.0);
                if rank == 0 { char_top1 += 1; }
                if rank < 5 { char_top5 += 1; }
            }
            let mrr = char_rr_sum / n_eval as f64;
            total_rr_sum += char_rr_sum;
            total_top1 += char_top1;
            total_top5 += char_top5;
            total_eval += n_eval;

            if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                eprintln!("  char '{}' MRR={:.3} top1={:.1}% top5={:.1}% (n={})",
                    c, mrr,
                    char_top1 as f64 / n_eval as f64 * 100.0,
                    char_top5 as f64 / n_eval as f64 * 100.0,
                    n_eval);
            }
        }
    }

    let train_elapsed = train_start.elapsed();

    let mrr = if total_eval > 0 { total_rr_sum / total_eval as f64 } else { 0.0 };
    let top1 = if total_eval > 0 { total_top1 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
    let top5 = if total_eval > 0 { total_top5 as f64 / total_eval as f64 * 100.0 } else { 0.0 };

    eprintln!("\nTraining complete: {} characters trained, {} skipped (<{} fonts), {:.1}s",
        trained_chars.len(), skipped, args.min_fonts, train_elapsed.as_secs_f64());
    eprintln!("Overall: MRR={:.3} top1={:.1}% top5={:.1}% (n={})",
        mrr, top1, top5, total_eval);

    // ── 4. Export weights ─────────────────────────────────────────
    eprintln!("Writing weights to {}...", args.output.display());

    let f = std::fs::File::create(&args.output).expect("create output file");
    let mut w = BufWriter::new(f);

    // Header: magic + version + n_chars
    w.write_all(b"TRIP").unwrap();
    w.write_all(&1u32.to_le_bytes()).unwrap();          // version
    w.write_all(&(trained_chars.len() as u32).to_le_bytes()).unwrap();

    for (c, net) in &trained_chars {
        // char_code: u32
        w.write_all(&(*c as u32).to_le_bytes()).unwrap();
        // W1: L1_IN × L1_OUT
        for &v in &net.fc1.w { w.write_all(&v.to_le_bytes()).unwrap(); }
        // b1: L1_OUT
        for &v in &net.fc1.b { w.write_all(&v.to_le_bytes()).unwrap(); }
        // W2: L1_OUT × L2_OUT
        for &v in &net.fc2.w { w.write_all(&v.to_le_bytes()).unwrap(); }
        // b2: L2_OUT
        for &v in &net.fc2.b { w.write_all(&v.to_le_bytes()).unwrap(); }
        // W3: L2_OUT × L3_OUT
        for &v in &net.fc3.w { w.write_all(&v.to_le_bytes()).unwrap(); }
        // b3: L3_OUT
        for &v in &net.fc3.b { w.write_all(&v.to_le_bytes()).unwrap(); }
    }

    w.flush().unwrap();

    let file_size = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
    let total_elapsed = start.elapsed();

    eprintln!("\n=== Training complete ===");
    eprintln!("  Characters: {}/{}", trained_chars.len(), chars.len());
    eprintln!("  Weights:    {} ({:.1} MB)", args.output.display(), file_size as f64 / 1e6);
    eprintln!("  Recall:     MRR={:.3} top1={:.1}% top5={:.1}%", mrr, top1, top5);
    eprintln!("  Render:     {:.1}s", render_secs);
    eprintln!("  Train:      {:.1}s", train_elapsed.as_secs_f64());
    eprintln!("  Total:      {:.1}s", total_elapsed.as_secs_f64());
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute triplet loss for given network and three inputs.
    fn triplet_loss_val(net: &TrainableNet, a: &[f32], p: &[f32], n: &[f32], margin: f32) -> f32 {
        let a_out = net.forward(a).out;
        let p_out = net.forward(p).out;
        let n_out = net.forward(n).out;
        let dp = dist_sq(&a_out, &p_out);
        let dn = dist_sq(&a_out, &n_out);
        (dp - dn + margin).max(0.0)
    }

    #[test]
    fn gradient_check() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut net = TrainableNet::new(&mut rng);

        // Random inputs
        let a: Vec<f32> = (0..L1_IN).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
        let p: Vec<f32> = (0..L1_IN).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
        let n: Vec<f32> = (0..L1_IN).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
        let margin = 0.5f32; // large margin to ensure active triplet

        // Forward
        let a_cache = net.forward(&a);
        let p_cache = net.forward(&p);
        let n_cache = net.forward(&n);

        let dp = dist_sq(&a_cache.out, &p_cache.out);
        let dn = dist_sq(&a_cache.out, &n_cache.out);
        let loss = (dp - dn + margin).max(0.0);

        eprintln!("Loss = {:.6}, dp = {:.6}, dn = {:.6}", loss, dp, dn);
        assert!(loss > 0.0, "Need active triplet for gradient check");

        // Analytic gradients via backward
        let mut d_a = vec![0.0f32; L3_OUT];
        let mut d_p = vec![0.0f32; L3_OUT];
        let mut d_n = vec![0.0f32; L3_OUT];
        for j in 0..L3_OUT {
            d_a[j] = 2.0 * (a_cache.out[j] - p_cache.out[j])
                   - 2.0 * (a_cache.out[j] - n_cache.out[j]);
            d_p[j] = 2.0 * (p_cache.out[j] - a_cache.out[j]);
            d_n[j] = 2.0 * (a_cache.out[j] - n_cache.out[j]);
        }

        net.fc1.dw.fill(0.0); net.fc1.db.fill(0.0);
        net.fc2.dw.fill(0.0); net.fc2.db.fill(0.0);
        net.fc3.dw.fill(0.0); net.fc3.db.fill(0.0);

        net.backward(&a_cache, &d_a);
        net.backward(&p_cache, &d_p);
        net.backward(&n_cache, &d_n);

        let eps = 1e-4f32;
        let mut max_rel_err = 0.0f32;
        let mut n_checked = 0usize;
        let mut n_bad = 0usize;

        // Check fc1.w (20 random indices)
        for _ in 0..20 {
            let idx = rng.gen_range(0..net.fc1.w.len());
            let analytic = net.fc1.dw[idx];
            let orig = net.fc1.w[idx];

            net.fc1.w[idx] = orig + eps;
            let lp = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc1.w[idx] = orig - eps;
            let lm = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc1.w[idx] = orig;

            let numerical = (lp - lm) / (2.0 * eps);
            let denom = analytic.abs().max(numerical.abs()).max(1e-7);
            let rel_err = (analytic - numerical).abs() / denom;

            if rel_err > 0.05 && analytic.abs().max(numerical.abs()) > 0.01 {
            }
            max_rel_err = max_rel_err.max(rel_err);
            n_checked += 1;
        }

        // Check fc2.w
        for _ in 0..20 {
            let idx = rng.gen_range(0..net.fc2.w.len());
            let analytic = net.fc2.dw[idx];
            let orig = net.fc2.w[idx];

            net.fc2.w[idx] = orig + eps;
            let lp = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc2.w[idx] = orig - eps;
            let lm = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc2.w[idx] = orig;

            let numerical = (lp - lm) / (2.0 * eps);
            let denom = analytic.abs().max(numerical.abs()).max(1e-7);
            let rel_err = (analytic - numerical).abs() / denom;

            if rel_err > 0.05 && analytic.abs().max(numerical.abs()) > 0.01 {
                eprintln!("  FAIL fc2.w[{}]: analytic={:.6}, numerical={:.6}, rel_err={:.4}", idx, analytic, numerical, rel_err);
                n_bad += 1;
            }
            max_rel_err = max_rel_err.max(rel_err);
            n_checked += 1;
        }

        // Check fc3.w
        for _ in 0..20 {
            let idx = rng.gen_range(0..net.fc3.w.len());
            let analytic = net.fc3.dw[idx];
            let orig = net.fc3.w[idx];

            net.fc3.w[idx] = orig + eps;
            let lp = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc3.w[idx] = orig - eps;
            let lm = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc3.w[idx] = orig;

            let numerical = (lp - lm) / (2.0 * eps);
            let denom = analytic.abs().max(numerical.abs()).max(1e-7);
            let rel_err = (analytic - numerical).abs() / denom;

            if rel_err > 0.05 && analytic.abs().max(numerical.abs()) > 0.01 {
                eprintln!("  FAIL fc3.w[{}]: analytic={:.6}, numerical={:.6}, rel_err={:.4}", idx, analytic, numerical, rel_err);
                n_bad += 1;
            }
            max_rel_err = max_rel_err.max(rel_err);
            n_checked += 1;
        }

        // Check biases (5 each)
        for idx in 0..5 {
            // fc1.b
            let analytic = net.fc1.db[idx];
            let orig = net.fc1.b[idx];
            net.fc1.b[idx] = orig + eps;
            let lp = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc1.b[idx] = orig - eps;
            let lm = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc1.b[idx] = orig;
            let numerical = (lp - lm) / (2.0 * eps);
            let denom = analytic.abs().max(numerical.abs()).max(1e-7);
            let rel_err = (analytic - numerical).abs() / denom;
            if rel_err > 0.05 && analytic.abs().max(numerical.abs()) > 0.01 {
                eprintln!("  FAIL fc1.b[{}]: analytic={:.6}, numerical={:.6}, rel_err={:.4}", idx, analytic, numerical, rel_err);
                n_bad += 1;
            }
            max_rel_err = max_rel_err.max(rel_err);
            n_checked += 1;

            // fc3.b
            let analytic = net.fc3.db[idx];
            let orig = net.fc3.b[idx];
            net.fc3.b[idx] = orig + eps;
            let lp = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc3.b[idx] = orig - eps;
            let lm = triplet_loss_val(&net, &a, &p, &n, margin);
            net.fc3.b[idx] = orig;
            let numerical = (lp - lm) / (2.0 * eps);
            let denom = analytic.abs().max(numerical.abs()).max(1e-7);
            let rel_err = (analytic - numerical).abs() / denom;
            if rel_err > 0.05 && analytic.abs().max(numerical.abs()) > 0.01 {
                eprintln!("  FAIL fc3.b[{}]: analytic={:.6}, numerical={:.6}, rel_err={:.4}", idx, analytic, numerical, rel_err);
                n_bad += 1;
            }
            max_rel_err = max_rel_err.max(rel_err);
            n_checked += 1;
        }

        eprintln!("\nGradient check: {n_checked} params, {n_bad} failures, max_rel_err={max_rel_err:.6}");
        assert!(n_bad == 0, "{n_bad} gradient mismatches (threshold: rel_err > 0.05)");
        eprintln!("✓ Gradient check PASSED");
    }

    #[test]
    fn loss_decreases_on_simple_data() {
        // Verify that training actually reduces loss on a trivial 3-font problem
        let mut rng = SmallRng::seed_from_u64(99);
        let mut net = TrainableNet::new(&mut rng);
        let margin = 0.3f32;
        let lr = 0.001f32;

        // 3 "fonts", each with slightly different feature patterns (overlapping enough to need training)
        let font_a: Vec<f32> = (0..L1_IN).map(|i| (i as f32 / L1_IN as f32) + 0.1).collect();
        let font_b: Vec<f32> = (0..L1_IN).map(|i| (i as f32 / L1_IN as f32) + 0.2).collect();
        let font_c: Vec<f32> = (0..L1_IN).map(|i| (i as f32 / L1_IN as f32) + 0.3).collect();

        // Add significant noise so samples from different fonts overlap
        let noisy = |base: &[f32], rng: &mut SmallRng| -> Vec<f32> {
            base.iter().map(|&v| v + rng.gen::<f32>() * 0.3 - 0.15).collect()
        };

        let samples_a: Vec<Vec<f32>> = (0..10).map(|_| noisy(&font_a, &mut rng)).collect();
        let samples_b: Vec<Vec<f32>> = (0..10).map(|_| noisy(&font_b, &mut rng)).collect();
        let samples_c: Vec<Vec<f32>> = (0..10).map(|_| noisy(&font_c, &mut rng)).collect();
        let all_samples = [&samples_a, &samples_b, &samples_c];

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for epoch in 0..100 {
            let mut epoch_loss = 0.0f32;
            let mut n_active = 0usize;

            for _ in 0..64 {
                let anchor_font = rng.gen_range(0..3);
                let neg_font = loop { let f = rng.gen_range(0..3); if f != anchor_font { break f; } };

                let ai = rng.gen_range(0..10);
                let pi = loop { let i = rng.gen_range(0..10); if i != ai { break i; } };
                let ni = rng.gen_range(0..10);

                let a_cache = net.forward(&all_samples[anchor_font][ai]);
                let p_cache = net.forward(&all_samples[anchor_font][pi]);
                let n_cache = net.forward(&all_samples[neg_font][ni]);

                let dp = dist_sq(&a_cache.out, &p_cache.out);
                let dn = dist_sq(&a_cache.out, &n_cache.out);
                let loss = (dp - dn + margin).max(0.0);

                if loss > 0.0 {
                    epoch_loss += loss;
                    n_active += 1;

                    let mut d_a = vec![0.0f32; L3_OUT];
                    let mut d_p = vec![0.0f32; L3_OUT];
                    let mut d_n = vec![0.0f32; L3_OUT];
                    for j in 0..L3_OUT {
                        d_a[j] = 2.0 * (a_cache.out[j] - p_cache.out[j])
                               - 2.0 * (a_cache.out[j] - n_cache.out[j]);
                        d_p[j] = 2.0 * (p_cache.out[j] - a_cache.out[j]);
                        d_n[j] = 2.0 * (a_cache.out[j] - n_cache.out[j]);
                    }
                    net.backward(&a_cache, &d_a);
                    net.backward(&p_cache, &d_p);
                    net.backward(&n_cache, &d_n);
                }
            }

            net.adam_step(lr, epoch + 1, n_active.max(1));

            let avg = if n_active > 0 { epoch_loss / n_active as f32 } else { 0.0 };
            if epoch == 0 { first_loss = avg; }
            if epoch == 99 { last_loss = avg; }

            if epoch % 20 == 0 || epoch == 99 {
                eprintln!("  epoch {}: loss={:.4}, active={}", epoch, avg, n_active);
            }
        }

        eprintln!("First loss: {:.4}, Last loss: {:.4}", first_loss, last_loss);
        assert!(last_loss < first_loss, "Loss should decrease! first={:.4} last={:.4}", first_loss, last_loss);
        eprintln!("✓ Loss decrease test PASSED");
    }
}
