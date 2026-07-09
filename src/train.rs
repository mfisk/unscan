//! LDA / triplet / Fisher trainer for unprint font classification.
//!
//! Moved from `src/bin/train.rs` into the main binary as a library module.
//! Called via `unprint --train-lda`.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::classifier;
use crate::features::{self, compute_features, FEAT_LEN};
use crate::font_scan;


// ---------------------------------------------------------------------------
// Shared training context
// ---------------------------------------------------------------------------

/// Everything a classifier's `train()` needs from the shared rendering pipeline.
pub struct TrainingContext<'a> {
    pub sequences: &'a [Vec<char>],
    pub seq_counts: &'a [usize],
    pub font_family: &'a [u32],
    pub font_id_map: &'a HashMap<String, u32>,
    pub glyph_map: &'a crate::glyph_map::NgramGlyphMap,
    pub n_families: usize,
    pub multi_variant_families: usize,
    pub min_fonts: usize,
    pub feat_dir: &'a std::path::Path,
    pub cached_combos: &'a [(u32, usize, Vec<usize>)],
    pub catalog: &'a [font_scan::FontEntry],
    pub catalog_hash: u64,
    pub render_params: &'a crate::char_render::RenderParams,
}

impl<'a> TrainingContext<'a> {
    /// Load all training samples for character index `ci`.
    pub fn load_samples(&self, si: usize) -> Vec<TrainingSample> {
        load_seq_combo_samples(self.feat_dir, &self.sequences[si], si, self.cached_combos, crate::features::AaVariant::all())
    }

    /// Build a glyph_id → family_id mapping for a given character.
    /// Uses the first font in each glyph group to determine the family.
    pub fn glyph_family_for_seq(&self, seq: &[char]) -> Vec<u32> {
        let n = self.glyph_map.glyph_count(seq);
        (0..n).map(|gid| {
            let fonts = self.glyph_map.fonts_for_glyph(seq, gid);
            if let Some(fk) = fonts.first() {
                if let Some(&fid) = self.font_id_map.get(fk.as_str()) {
                    return self.font_family[fid as usize];
                }
            }
            u32::MAX // shouldn't happen
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

pub struct TrainArgs {
    pub output: PathBuf,
    pub heights: Vec<u32>,
    pub max_fonts: usize,
    pub font_dir: Vec<PathBuf>,
    pub epochs: usize,
    pub lr: f32,
    pub margin: f32,
    pub batch_size: usize,
    pub min_fonts: usize,
    pub tmpdir: Option<PathBuf>,
    pub fast: bool,
    pub fisher: bool,
    pub mahalanobis: bool,
    pub lda: bool,
    pub lda_dims: usize,
    pub lda_reg: f64,
    pub mlp: bool,
    pub mlp_noise: f32,
    pub mlp_dropout: f32,
    pub render_params: crate::char_render::RenderParams,
}

impl Default for TrainArgs {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            output: PathBuf::from(home).join(".cache").join("unprint").join("lda-weights.bin"),
            heights: vec![],
            max_fonts: 0,
            font_dir: vec![],
            epochs: 30,
            lr: 0.001,
            margin: 0.3,
            batch_size: 256,
            min_fonts: 5,
            tmpdir: None,
            fast: false,
            fisher: false,
            mahalanobis: false,
            lda: true,
            lda_dims: 32,
            lda_reg: 0.01,
            mlp: false,
            mlp_noise: 0.02,
            mlp_dropout: 0.3,
            render_params: crate::char_render::RenderParams::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Network architecture — matches classifier.rs exactly
// ---------------------------------------------------------------------------

const L1_IN: usize = FEAT_LEN;  // 64 (was 100 at NORM_H=48)
const L1_OUT: usize = 128;
const L2_OUT: usize = 64;
const L3_OUT: usize = 32;

/// Trainable linear layer with Adam optimizer state.
pub struct Linear {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,   // rows × cols, row-major (w[i * cols + j])
    pub b: Vec<f32>,   // cols
    // Gradients (accumulated, zeroed each step)
    pub dw: Vec<f32>,
    pub db: Vec<f32>,
    // Adam moment estimates
    mw: Vec<f32>,
    vw: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Linear {
    pub fn new(rows: usize, cols: usize, rng: &mut SmallRng) -> Self {
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
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
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
    pub fn backward(&mut self, input: &[f32], d_out: &[f32]) -> Vec<f32> {
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
    pub fn adam_step(&mut self, lr: f32, t: usize, batch_size: usize) {
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
pub struct TrainableNet {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
}

impl TrainableNet {
    pub fn new(rng: &mut SmallRng) -> Self {
        Self {
            fc1: Linear::new(L1_IN, L1_OUT, rng),
            fc2: Linear::new(L1_OUT, L2_OUT, rng),
            fc3: Linear::new(L2_OUT, L3_OUT, rng),
        }
    }

    /// Forward pass with cached activations for backprop.
    pub fn forward(&self, input: &[f32]) -> ForwardCache {
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
    pub fn backward(&mut self, cache: &ForwardCache, d_out: &[f32]) {
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

    pub fn adam_step(&mut self, lr: f32, t: usize, batch_size: usize) {
        self.fc1.adam_step(lr, t, batch_size);
        self.fc2.adam_step(lr, t, batch_size);
        self.fc3.adam_step(lr, t, batch_size);
    }
}

pub struct ForwardCache {
    pub input: Vec<f32>,
    pub z1: Vec<f32>,
    pub h1: Vec<f32>,
    pub z2: Vec<f32>,
    pub h2: Vec<f32>,
    pub z3: Vec<f32>,
    pub norm: f32,
    pub out: Vec<f32>,
}

/// Gradient of L2 normalization: d(x/||x||)/dx applied to upstream gradient.
pub fn l2_norm_backward(z: &[f32], norm: f32, d_out: &[f32]) -> Vec<f32> {
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
pub fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ---------------------------------------------------------------------------
// Rendering — use shared char_render pipeline
use crate::features::AaVariant;
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Training data structure
// ---------------------------------------------------------------------------

pub struct TrainingSample {
    pub glyph_id: u32,     // compact font index for triplet mining
    pub features: [f32; FEAT_LEN],
}

/// Jacobi eigendecomposition of a symmetric matrix.
/// Returns the top-k eigenvectors (by eigenvalue magnitude) as rows of
/// a k × n matrix (flattened row-major).
pub fn jacobi_eigen_top_k(mat: &[f64], n: usize, k: usize) -> Vec<f64> {
    assert_eq!(mat.len(), n * n);
    let k = k.min(n);

    // Copy matrix (will be modified in place)
    let mut a = mat.to_vec();
    // Eigenvector matrix (starts as identity)
    let mut v = vec![0.0f64; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }

    let max_iter = 200;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in i+1..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 { break; }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Givens rotation to rows/cols p and q
        let mut new_a = a.clone();
        for i in 0..n {
            if i == p || i == q { continue; }
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            new_a[i * n + p] = cos_t * aip + sin_t * aiq;
            new_a[p * n + i] = new_a[i * n + p];
            new_a[i * n + q] = -sin_t * aip + cos_t * aiq;
            new_a[q * n + i] = new_a[i * n + q];
        }
        new_a[p * n + p] = cos_t * cos_t * app + 2.0 * cos_t * sin_t * apq + sin_t * sin_t * aqq;
        new_a[q * n + q] = sin_t * sin_t * app - 2.0 * cos_t * sin_t * apq + cos_t * cos_t * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = cos_t * vip + sin_t * viq;
            v[i * n + q] = -sin_t * vip + cos_t * viq;
        }
    }

    // Extract eigenvalues (diagonal of a)
    let mut eigen_pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    eigen_pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));

    // Return top-k eigenvectors as rows
    let mut result = vec![0.0f64; k * n];
    for d in 0..k {
        let col = eigen_pairs[d].1;
        for i in 0..n {
            result[d * n + i] = v[i * n + col];
        }
    }
    result
}

/// Load per-character samples from multiple (height, aa) combo files.
/// Codepoint-based key for a sequence. E.g. ['A'] → "0041", ['h','e'] → "0068_0065".
pub fn seq_key(seq: &[char]) -> String {
    seq.iter().map(|c| format!("{:04X}", *c as u32)).collect::<Vec<_>>().join("_")
}

pub fn load_seq_combo_samples(
    feat_dir: &std::path::Path,
    seq: &[char],
    si: usize,
    combos: &[(u32, usize, Vec<usize>)], // (ht, aa_idx, per-seq counts)
    all_aa: &[AaVariant],
) -> Vec<TrainingSample> {
    let total: usize = combos.iter().map(|(_, _, counts)| counts[si]).sum();
    let mut samples = Vec::with_capacity(total);
    let mut buf4 = [0u8; 4];
    let sk = seq_key(seq);
    for (ht, aa_idx, counts) in combos {
        let n = counts[si];
        if n == 0 { continue; }
        let aa_name = all_aa[*aa_idx].name();
        let path = feat_dir.join(format!("{}_h{}_{}.bin", sk, ht, aa_name));
        let file = std::fs::File::open(&path).expect("open combo feature file");
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        // Read and validate header
        reader.read_exact(&mut buf4).expect("read magic");
        assert!(&buf4 == b"UTFD", "invalid training feature magic in {}", path.display());
        reader.read_exact(&mut buf4).expect("read version");
        let version = u32::from_le_bytes(buf4);
        assert!(version == 1, "unsupported training feature version {version} in {}", path.display());
        reader.read_exact(&mut buf4).expect("read feat_len");
        let file_feat_len = u32::from_le_bytes(buf4) as usize;
        assert!(file_feat_len == FEAT_LEN, "FEAT_LEN mismatch: file has {file_feat_len}, code has {FEAT_LEN} in {}", path.display());
        for _ in 0..n {
            reader.read_exact(&mut buf4).expect("read glyph_id");
            let glyph_id = u32::from_le_bytes(buf4);
            let mut features = [0.0f32; FEAT_LEN];
            for f in &mut features {
                reader.read_exact(&mut buf4).expect("read feature");
                *f = f32::from_le_bytes(buf4);
            }
            samples.push(TrainingSample { glyph_id, features });
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// Centroid-based MRR evaluation
// ---------------------------------------------------------------------------

/// Accumulated MRR/top-k statistics from centroid-based ranking.
#[derive(Default)]
pub struct RankStats {
    /// Sum of 1/rank (strict: exact glyph_id)
    pub strict_rr: f64,
    pub strict_top1: usize,
    pub strict_top5: usize,
    /// Sum of 1/rank (family: best same-family variant)
    pub family_rr: f64,
    pub family_top1: usize,
    pub family_top5: usize,
    /// Baseline (unweighted Euclidean to centroid)
    pub base_rr: f64,
    pub base_top1: usize,
    /// Number of evaluated samples
    pub n_eval: usize,
}

impl RankStats {
    pub fn accumulate(&mut self, other: &RankStats) {
        self.strict_rr += other.strict_rr;
        self.strict_top1 += other.strict_top1;
        self.strict_top5 += other.strict_top5;
        self.family_rr += other.family_rr;
        self.family_top1 += other.family_top1;
        self.family_top5 += other.family_top5;
        self.base_rr += other.base_rr;
        self.base_top1 += other.base_top1;
        self.n_eval += other.n_eval;
    }

    pub fn strict_mrr(&self) -> f64 { if self.n_eval > 0 { self.strict_rr / self.n_eval as f64 } else { 0.0 } }
    pub fn strict_top1_pct(&self) -> f64 { if self.n_eval > 0 { self.strict_top1 as f64 / self.n_eval as f64 * 100.0 } else { 0.0 } }
    pub fn strict_top5_pct(&self) -> f64 { if self.n_eval > 0 { self.strict_top5 as f64 / self.n_eval as f64 * 100.0 } else { 0.0 } }
    pub fn family_mrr(&self) -> f64 { if self.n_eval > 0 { self.family_rr / self.n_eval as f64 } else { 0.0 } }
    pub fn family_top1_pct(&self) -> f64 { if self.n_eval > 0 { self.family_top1 as f64 / self.n_eval as f64 * 100.0 } else { 0.0 } }
    pub fn family_top5_pct(&self) -> f64 { if self.n_eval > 0 { self.family_top5 as f64 / self.n_eval as f64 * 100.0 } else { 0.0 } }
    pub fn base_mrr(&self) -> f64 { if self.n_eval > 0 { self.base_rr / self.n_eval as f64 } else { 0.0 } }
    pub fn base_top1_pct(&self) -> f64 { if self.n_eval > 0 { self.base_top1 as f64 / self.n_eval as f64 * 100.0 } else { 0.0 } }
}

/// Evaluate a set of samples against centroids using a caller-provided distance
/// function, computing strict, family, and baseline MRR.
///
/// - `samples`: the full sample array (each has `.glyph_id` and `.features`)
/// - `eval_indices`: which samples to evaluate (subsampled)
/// - `class_means`: glyph_id → centroid (in raw feature space, used for baseline)
/// - `centroid_fids`: ordered list of centroid font_ids
/// - `glyph_family`: glyph_id → family_id mapping
/// - `calc_dists`: given a sample index, returns `Vec<(glyph_id, distance)>` using
///    the classifier-specific metric
pub fn eval_mrr(
    samples: &[TrainingSample],
    eval_indices: &[usize],
    class_means: &HashMap<u32, Vec<f64>>,
    _centroid_fids: &[u32],
    glyph_family: &[u32],
    calc_dists: &dyn Fn(usize) -> Vec<(u32, f64)>,
) -> RankStats {
    let mut stats = RankStats::default();
    stats.n_eval = eval_indices.len();

    for &i in eval_indices {
        let correct = samples[i].glyph_id;
        let correct_famid = glyph_family.get(correct as usize).copied().unwrap_or(u32::MAX);

        // Classifier-specific distances
        let dists = calc_dists(i);

        // Strict: rank of exact glyph_id
        let d_correct = dists.iter()
            .find(|&&(fid, _)| fid == correct)
            .map(|&(_, d)| d)
            .unwrap_or(f64::MAX);
        let rank = dists.iter()
            .filter(|&&(fid, d)| fid != correct && d < d_correct)
            .count();
        stats.strict_rr += 1.0 / (rank as f64 + 1.0);
        if rank == 0 { stats.strict_top1 += 1; }
        if rank < 5 { stats.strict_top5 += 1; }

        // Family: best rank among any glyph in same family
        let best_fam_dist = dists.iter()
            .filter(|&&(fid, _)| glyph_family.get(fid as usize).copied().unwrap_or(u32::MAX) == correct_famid)
            .map(|&(_, d)| d)
            .fold(f64::MAX, f64::min);
        let fam_rank = dists.iter()
            .filter(|&&(fid, d)| glyph_family.get(fid as usize).copied().unwrap_or(u32::MAX) != correct_famid && d < best_fam_dist)
            .count();
        stats.family_rr += 1.0 / (fam_rank as f64 + 1.0);
        if fam_rank == 0 { stats.family_top1 += 1; }
        if fam_rank < 5 { stats.family_top5 += 1; }

        // Baseline: unweighted Euclidean to class means
        let correct_cm = &class_means[&correct];
        let d_base: f64 = (0..FEAT_LEN)
            .map(|j| { let d = samples[i].features[j] as f64 - correct_cm[j]; d * d })
            .sum();
        let base_rank = class_means.iter()
            .filter(|(&fid, cm)| {
                if fid == correct { return false; }
                let d: f64 = (0..FEAT_LEN)
                    .map(|j| { let diff = samples[i].features[j] as f64 - cm[j]; diff * diff })
                    .sum();
                d < d_base
            })
            .count();
        stats.base_rr += 1.0 / (base_rank as f64 + 1.0);
        if base_rank == 0 { stats.base_top1 += 1; }
    }

    stats
}

/// Select up to `max_eval` evaluation indices, subsampling with a deterministic
/// seed derived from the character.
pub fn subsample_eval(n: usize, max_eval: usize, seed: u64) -> Vec<usize> {
    if n <= max_eval {
        (0..n).collect()
    } else {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        idx.truncate(max_eval);
        idx
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------


pub fn run_train(mut args: TrainArgs) {
    // Limit rayon thread pool to avoid holding too many fonts in memory at once.
    // On memory-constrained machines (no swap), each thread holds one font's data.
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let max_threads = num_cpus.min(4); // cap at 4 to limit memory pressure
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global();

    if args.heights.is_empty() {
        args.heights = vec![features::NORM_H];
    }

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

    eprintln!("=== unprint all-in-one triplet trainer ===");
    eprintln!("Heights: {:?}{}", active_heights, if args.fast { " (fast)" } else { "" });
    eprintln!("AA variants: {}{}", aa_variants.len(), if args.fast { " (fast)" } else { "" });
    eprintln!("Epochs: {}", args.epochs);
    eprintln!("Learning rate: {}", args.lr);
    eprintln!("Margin: {}", args.margin);

    // ── 1. Scan fonts ─────────────────────────────────────────────
    let font_dirs: Vec<PathBuf> = font_scan::default_font_dirs(&args.font_dir);

    let mut catalog = font_scan::scan_fonts(&font_dirs);
    // Sort by font_key for deterministic font_id assignment, matching
    // FontRegistry::new() ordering so runtime and training agree.
    catalog.sort_by(|a, b| a.font_key().cmp(&b.font_key()));
    eprintln!("  {} font entries found", catalog.len());

    if args.max_fonts > 0 && catalog.len() > args.max_fonts {
        catalog.truncate(args.max_fonts);
        eprintln!("  Limiting to {} fonts (--max-fonts)", args.max_fonts);
    }

    // Build unified sequence list: unigrams + bigrams, all using the same code path.
    let sequences: Vec<Vec<char>> = {
        let mut seqs: Vec<Vec<char>> = features::supported_chars().iter().map(|&c| vec![c]).collect();
        seqs.extend(features::supported_sequences(2).iter().cloned());
        seqs
    };
    eprintln!("  {} sequences ({} unigrams + {} bigrams)",
        sequences.len(),
        features::supported_chars().len(),
        features::supported_sequences(2).len());

    // ── 2. Render & extract features ──────────────────────────────
    // Write per-char binary feature files to disk to avoid OOM.
    // Each file: sequence of (glyph_id: u32, features: [f32; FEAT_LEN]) per sample.

    let total_fonts = catalog.len();
    let progress = AtomicUsize::new(0);
    let start = std::time::Instant::now();

    // Assign each unique font_key a compact integer ID
    let font_id_map: HashMap<String, u32> = catalog.iter().enumerate()
        .map(|(i, fe)| (fe.font_key(), i as u32))
        .collect();

    // Compute catalog hash for cache validation (same algorithm as FontRegistry).
    let catalog_hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for fe in &catalog { fe.font_key().hash(&mut hasher); }
        hasher.finish()
    };

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
    let n_seqs = sequences.len();

    let seq_to_idx: HashMap<Vec<char>, usize> = sequences.iter().enumerate()
        .map(|(i, seq)| (seq.clone(), i))
        .collect();

    // Training feature cache directory.
    // Default: ~/.cache/unprint/training/ (XDG-compliant, persists across runs).
    // Override with --tmpdir for custom location.
    let feat_dir = match &args.tmpdir {
        Some(d) => d.clone(),
        None => {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            std::path::PathBuf::from(home)
                .join(".cache")
                .join("unprint")
                .join("training")
        }
    };
    std::fs::create_dir_all(&feat_dir).expect("create training cache dir");
    eprintln!("Training cache: {}", feat_dir.display());

    // ── Pre-warm character render cache + build GlyphMap ──
    // Render every (font, char) at default params, capturing the content hash.
    // Identical renders share a hash → same glyph equivalence class.
    let mut glyph_map = {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let prewarm_done = AtomicUsize::new(0);
        let prewarm_total = catalog.len();
        eprintln!("\nPre-warming character render cache + building GlyphMap ({} fonts × {} chars)...",
            prewarm_total, sequences.len());
        let prewarm_t0 = std::time::Instant::now();

        // Each font returns Vec<(Vec<char>, hash, font_key)> for GlyphMap construction
        let per_font_hashes: Vec<Vec<(Vec<char>, u64, String)>> = catalog.par_iter().map(|fe| {
            let font_data = match std::fs::read(&fe.path) {
                Ok(d) => d,
                Err(_) => {
                    let done = prewarm_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 500 == 0 || done == prewarm_total {
                        eprintln!("  Pre-render [{}/{}]...", done, prewarm_total);
                    }
                    return Vec::new();
                }
            };
            let font = match ab_glyph::FontRef::try_from_slice(&font_data) {
                Ok(f) => f,
                Err(_) => {
                    let done = prewarm_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 500 == 0 || done == prewarm_total {
                        eprintln!("  Pre-render [{}/{}]...", done, prewarm_total);
                    }
                    return Vec::new();
                }
            };
            let overrides = fe.glyph_overrides.as_deref();
            let fk = fe.font_key();
            let mut hashes = Vec::with_capacity(sequences.len());
            for seq in &sequences {
                let gid_overrides: Vec<Option<ab_glyph::GlyphId>> = seq.iter().map(|c| {
                    overrides.and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| ab_glyph::GlyphId(*g)))
                }).collect();
                let params = crate::char_render::RenderParams::default();
                if let Some(img) = crate::char_render::render_ngram_fresh(&font, seq, &gid_overrides, &params) {
                    let hash = crate::glyph_map::hash_image(&img);
                    let path = crate::char_render::ngram_cache_path(seq, hash, &params);
                    if !path.exists() {
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let _ = img.save(&path);
                    }
                    hashes.push((seq.clone(), hash, fk.clone()));
                }
            }
            let done = prewarm_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 500 == 0 || done == prewarm_total {
                eprintln!("  Pre-render [{}/{}]...", done, prewarm_total);
            }
            hashes
        }).collect();

        // Register all results into the glyph_map
        let mut gmap = crate::glyph_map::NgramGlyphMap::new(catalog_hash);
        for font_hashes in per_font_hashes {
            for (seq, hash, font_key) in font_hashes {
                gmap.register(&seq, &font_key, hash);
            }
        }

        let total_glyphs: usize = gmap.groups.values().map(|g| g.len()).sum();
        let total_deduped: usize = gmap.groups.values()
            .flat_map(|gs| gs.iter())
            .filter(|g| g.font_keys.len() > 1)
            .map(|g| g.font_keys.len() - 1)
            .sum();
        eprintln!("  GlyphMap: {} unique glyphs across {} sequences ({} duplicate renders eliminated)",
            total_glyphs, gmap.groups.len(), total_deduped);
        eprintln!("  Pre-warm + GlyphMap complete in {:.1}s", prewarm_t0.elapsed().as_secs_f64());
        gmap
    };

    // ── Determine which (height, aa) combos need rendering ───────
    // All possible combos (always render the full set so fast/normal share files)
    let _all_heights: &[u32] = &args.heights;
    let all_aa: &[AaVariant] = AaVariant::all();

    /// Build a stable codepoint-based key for a character sequence.
    /// E.g. ['A'] → "0041", ['h','e'] → "0068_0065".
    fn seq_key(seq: &[char]) -> String {
        seq.iter().map(|c| format!("{:04X}", *c as u32)).collect::<Vec<_>>().join("_")
    }

    /// File path for a (sequence, height, aa) feature file.
    fn combo_path(feat_dir: &std::path::Path, seq: &[char], ht: u32, aa_name: &str) -> std::path::PathBuf {
        feat_dir.join(format!("{}_h{}_{}.bin", seq_key(seq), ht, aa_name))
    }

    /// Manifest path for a (height, aa) combo.
    fn manifest_combo_path(feat_dir: &std::path::Path, ht: u32, aa_name: &str) -> std::path::PathBuf {
        feat_dir.join(format!("manifest_h{}_{}.txt", ht, aa_name))
    }

    // A combo is cached if its manifest exists, matches the font count, all
    // char files exist, and the manifest is not older than the font scan cache
    // (font key changes invalidate glyph IDs stored in the feature files).
    let scan_cache_mtime = std::fs::metadata(crate::font_scan::scan_cache_path())
        .and_then(|m| m.modified())
        .ok();
    let combo_cached = |ht: u32, aa_idx: usize| -> Option<Vec<usize>> {
        let aa_name = all_aa[aa_idx].name();
        let mpath = manifest_combo_path(&feat_dir, ht, aa_name);
        // Reject if manifest is older than font scan cache — glyph IDs may be stale
        if let Some(scan_t) = scan_cache_mtime {
            if let Ok(meta) = std::fs::metadata(&mpath) {
                if let Ok(manifest_t) = meta.modified() {
                    if manifest_t < scan_t {
                        return None;
                    }
                }
            }
        }
        let content = std::fs::read_to_string(&mpath).ok()?;
        let mut lines = content.lines();
        let header = lines.next()?;
        if header.trim() != format!("fonts={} seqs={}", catalog.len(), n_seqs) {
            return None;
        }
        let mut counts = Vec::with_capacity(n_seqs);
        for line in lines {
            counts.push(line.trim().parse::<usize>().ok()?);
        }
        if counts.len() != n_seqs { return None; }
        // Verify all files exist
        for si in 0..n_seqs {
            if !combo_path(&feat_dir, &sequences[si], ht, aa_name).exists() { return None; }
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
        let _needed_set: std::collections::HashSet<(u32, usize)> = needed_combos.iter().copied().collect();

        // Open writers: indexed by (ci, combo_index)
        // combo_index is position in needed_combos
        let n_combos = needed_combos.len();
        let mut combo_tmp_paths: Vec<Vec<(std::path::PathBuf, std::path::PathBuf)>> = Vec::new();
        let mut combo_writers: Vec<Vec<BufWriter<std::fs::File>>> = (0..n_seqs).map(|si| {
            let mut paths_row = Vec::new();
            let writers: Vec<_> = needed_combos.iter().map(|&(ht, aa_idx)| {
                let aa_name = all_aa[aa_idx].name();
                let final_path = combo_path(&feat_dir, &sequences[si], ht, aa_name);
                let tmp_path = crate::atomic_file::tmp_for(&final_path);
                let w = BufWriter::with_capacity(
                    64 * 1024,
                    std::fs::File::create(&tmp_path).expect("create combo feature file"),
                );
                paths_row.push((tmp_path, final_path));
                w
            }).collect();
            combo_tmp_paths.push(paths_row);
            writers
        }).collect();

        // Per-combo per-char counts
        let mut combo_counts: Vec<Vec<usize>> = vec![vec![0usize; n_seqs]; n_combos];

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

                let fk = fe.font_key();
                let overrides = fe.glyph_overrides.as_deref();
                let mut samples = Vec::new();

                for (si, seq) in sequences.iter().enumerate() {
                    // Look up this font's glyph_id for this sequence.
                    // If the font didn't render it (not in GlyphMap), skip.
                    let glyph_id = match glyph_map.glyph_id_for_font(seq, &fk) {
                        Some(id) => id as u32,
                        None => continue,
                    };

                    // Skip if we're not the representative font for this glyph group.
                    // All fonts in a group produce identical renders → identical features,
                    // so we only need one sample per glyph_id per combo.
                    let rep_font = &glyph_map.fonts_for_glyph(seq, glyph_id as usize)[0];
                    if *rep_font != fk {
                        continue;
                    }

                    let gid_overrides: Vec<Option<ab_glyph::GlyphId>> = seq.iter().map(|c| {
                        overrides.and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| ab_glyph::GlyphId(*g)))
                    }).collect();

                    for &(ht, aa_idx_all) in &needed_combos {
                        let mut params = args.render_params.clone();
                        params.height = ht;
                        params.aa = all_aa[aa_idx_all];
                        let img = match crate::char_render::render_ngram_fresh(
                            &font, seq, &gid_overrides, &params,
                        ) {
                            Some(img) => img,
                            None => continue,
                        };

                        let feats = match compute_features(&img, false) {
                            Some(f) => f,
                            None => continue,
                        };

                        let combo_idx = needed_combos.iter()
                            .position(|&(h, a)| h == ht && a == aa_idx_all)
                            .unwrap();

                        samples.push((si, combo_idx, TrainingSample {
                            glyph_id,
                            features: feats.as_slice(),
                        }));
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
                for (si, combo_idx, sample) in font_samples {
                    let w = &mut combo_writers[si][combo_idx];
                    // Write header on first sample
                    if combo_counts[combo_idx][si] == 0 {
                        use std::io::Write;
                        w.write_all(b"UTFD").expect("write magic");        // Unprint Training Feature Data
                        w.write_all(&1u32.to_le_bytes()).expect("write version");
                        w.write_all(&(FEAT_LEN as u32).to_le_bytes()).expect("write feat_len");
                    }
                    w.write_all(&sample.glyph_id.to_le_bytes()).expect("write glyph_id");
                    for &f in &sample.features {
                        w.write_all(&f.to_le_bytes()).expect("write feature");
                    }
                    combo_counts[combo_idx][si] += 1;
                }
            }
        }

        // Flush, close, and atomically rename all writers
        for char_ws in &mut combo_writers {
            for w in char_ws {
                w.flush().expect("flush combo features");
            }
        }
        drop(combo_writers);
        for paths_row in &combo_tmp_paths {
            for (tmp, final_path) in paths_row {
                std::fs::rename(tmp, final_path).expect("atomic rename combo feature");
            }
        }

        // Write per-combo manifests
        for (combo_idx, &(ht, aa_idx)) in needed_combos.iter().enumerate() {
            let aa_name = all_aa[aa_idx].name();
            let mpath = manifest_combo_path(&feat_dir, ht, aa_name);
            let tmp_mpath = crate::atomic_file::tmp_for(&mpath);
            let mut manifest = format!("fonts={} seqs={}", catalog.len(), n_seqs);
            for ci in 0..n_seqs {
                manifest.push('\n');
                manifest.push_str(&combo_counts[combo_idx][ci].to_string());
            }
            std::fs::write(&tmp_mpath, &manifest).expect("write combo manifest");
            std::fs::rename(&tmp_mpath, &mpath).expect("atomic rename manifest");
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
    let mut seq_counts = vec![0usize; n_seqs];
    for (_, _, ref counts) in &cached_combos {
        for si in 0..n_seqs {
            seq_counts[si] += counts[si];
        }
    }
    let total_samples: usize = seq_counts.iter().sum();
    eprintln!("Total samples for evaluation: {} ({} combos)", total_samples, cached_combos.len());


    // ── 3. Fisher scoring mode ────────────────────────────────────
    if args.fisher {
        let _ = std::fs::remove_file(&args.output); // force retrain
        let ctx = TrainingContext {
            sequences: &sequences,
            seq_counts: &seq_counts,
            font_family: &font_family,
            font_id_map: &font_id_map,
            glyph_map: &glyph_map,
            n_families,
            multi_variant_families,
            min_fonts: args.min_fonts,
            feat_dir: &feat_dir,
            cached_combos: &cached_combos,
            catalog: &catalog,
            catalog_hash,
            render_params: &args.render_params,
        };
        let _cls = classifier::PerCharFisherClassifier::load(&args.output, Some(&ctx))
            .unwrap_or_else(|e| { eprintln!("Error training Fisher: {e}"); std::process::exit(1); });
        return;
    }

    // ── 3b. Mahalanobis mode ─────────────────────────────────────
    if args.mahalanobis {
        let ctx = TrainingContext {
            sequences: &sequences,
            seq_counts: &seq_counts,
            font_family: &font_family,
            font_id_map: &font_id_map,
            glyph_map: &glyph_map,
            n_families,
            multi_variant_families,
            min_fonts: args.min_fonts,
            feat_dir: &feat_dir,
            cached_combos: &cached_combos,
            catalog: &catalog,
            catalog_hash,
            render_params: &args.render_params,
        };
        classifier::MahalanobisClassifier::train(&ctx, &args.output);
        return;
    }

    // ── 3c. LDA mode ─────────────────────────────────────────────
    if args.lda {
        let ctx = TrainingContext {
            sequences: &sequences,
            seq_counts: &seq_counts,
            font_family: &font_family,
            font_id_map: &font_id_map,
            glyph_map: &glyph_map,
            n_families,
            multi_variant_families,
            min_fonts: args.min_fonts,
            feat_dir: &feat_dir,
            cached_combos: &cached_combos,
            catalog: &catalog,
            catalog_hash,
            render_params: &args.render_params,
        };
        classifier::LdaClassifier::train_with_params(&ctx, &args.output, args.lda_dims, args.lda_reg);

        return;
    }

    // ── 3d. MLP mode ─────────────────────────────────────────────
    if args.mlp {
        let ctx = TrainingContext {
            sequences: &sequences,
            seq_counts: &seq_counts,
            font_family: &font_family,
            font_id_map: &font_id_map,
            glyph_map: &glyph_map,
            n_families,
            multi_variant_families,
            min_fonts: args.min_fonts,
            feat_dir: &feat_dir,
            cached_combos: &cached_combos,
            catalog: &catalog,
            catalog_hash,
            render_params: &args.render_params,
        };
        classifier::MlpClassifier::train(
            &ctx, &args.output,
            args.epochs, args.lr, args.batch_size,
            args.mlp_noise, args.mlp_dropout,
        );
        return;
    }

    // ── 3. Train per-character triplet networks ─────────────────────
        let ctx = TrainingContext {
            sequences: &sequences,
            seq_counts: &seq_counts,
            font_family: &font_family,
            font_id_map: &font_id_map,
            glyph_map: &glyph_map,
            n_families,
            multi_variant_families,
            min_fonts: args.min_fonts,
            feat_dir: &feat_dir,
            cached_combos: &cached_combos,
            catalog: &catalog,
            catalog_hash,
            render_params: &args.render_params,
        };
    classifier::TripletClassifier::train_with_params(
        &ctx, &args.output,
        args.epochs, args.lr, args.margin, args.batch_size,
    );
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

            if rel_err > 0.07 && analytic.abs().max(numerical.abs()) > 0.01 {
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

            if rel_err > 0.07 && analytic.abs().max(numerical.abs()) > 0.01 {
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

            if rel_err > 0.07 && analytic.abs().max(numerical.abs()) > 0.01 {
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
            if rel_err > 0.07 && analytic.abs().max(numerical.abs()) > 0.01 {
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
            if rel_err > 0.07 && analytic.abs().max(numerical.abs()) > 0.01 {
                eprintln!("  FAIL fc3.b[{}]: analytic={:.6}, numerical={:.6}, rel_err={:.4}", idx, analytic, numerical, rel_err);
                n_bad += 1;
            }
            max_rel_err = max_rel_err.max(rel_err);
            n_checked += 1;
        }

        eprintln!("\nGradient check: {n_checked} params, {n_bad} failures, max_rel_err={max_rel_err:.6}");
        assert!(n_bad == 0, "{n_bad} gradient mismatches (threshold: rel_err > 0.07)");
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

// ---------------------------------------------------------------------------
// Runtime training data — self-contained context for per-font LDA training
// ---------------------------------------------------------------------------
//
// Owns the data that TrainingContext borrows, so it can live in the runtime
// pipeline without tying callers to the training module's lifetimes.

pub struct RuntimeTrainingData {
    pub sequences: Vec<Vec<char>>,
    pub seq_counts: Vec<usize>,
    pub font_family: Vec<u32>,
    pub font_id_map: HashMap<String, u32>,
    pub n_families: usize,
    pub multi_variant_families: usize,
    pub feat_dir: std::path::PathBuf,
    pub cached_combos: Vec<(u32, usize, Vec<usize>)>,
    pub catalog: Vec<crate::font_scan::FontEntry>,
    pub catalog_hash: u64,
    pub render_params: crate::char_render::RenderParams,
}

impl RuntimeTrainingData {
    /// Build from a font registry and existing feature cache.
    /// Returns None if the feature cache doesn't exist.
    pub fn from_registry(
        font_registry: &crate::font_scan::FontRegistry,
        glyph_map: &crate::glyph_map::NgramGlyphMap,
        render_params: &crate::char_render::RenderParams,
    ) -> Option<Self> {
        let sequences: Vec<Vec<char>> = {
            let mut seqs: Vec<Vec<char>> = crate::features::supported_chars().iter().map(|&c| vec![c]).collect();
            seqs.extend(crate::features::supported_sequences(2).iter().cloned());
            seqs
        };
        let n_seqs = sequences.len();

        // Get sorted catalog (same order as training)
        let mut catalog: Vec<crate::font_scan::FontEntry> = font_registry.iter()
            .cloned()
            .collect();
        catalog.sort_by(|a, b| a.font_key().cmp(&b.font_key()));

        let catalog_hash = glyph_map.catalog_hash;

        // Build font_id_map and family info
        let font_id_map: HashMap<String, u32> = catalog.iter().enumerate()
            .map(|(i, fe)| (fe.font_key(), i as u32))
            .collect();

        let mut family_name_to_id: HashMap<&str, u32> = HashMap::new();
        let mut font_family: Vec<u32> = Vec::with_capacity(catalog.len());
        let mut family_members: Vec<Vec<u32>> = Vec::new();
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

        // Find feature cache directory (same as run_train default)
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let feat_dir = std::path::Path::new(&home)
            .join(".cache").join("unprint").join("training")
            .to_path_buf();
        if !feat_dir.exists() { return None; }

        // Scan cached combos (same logic as run_train)
        let all_aa = crate::features::AaVariant::all();
        let active_heights = vec![crate::features::NORM_H];
        let mut cached_combos: Vec<(u32, usize, Vec<usize>)> = Vec::new();

        for &ht in &active_heights {
            for (aa_idx, aa) in all_aa.iter().enumerate() {
                let aa_name = aa.name();
                let mpath = feat_dir.join(format!("manifest_h{}_{}.txt", ht, aa_name));
                if let Ok(content) = std::fs::read_to_string(&mpath) {
                    let mut lines = content.lines();
                    if let Some(header) = lines.next() {
                        let expected = format!("fonts={} seqs={}", catalog.len(), n_seqs);
                        if header.trim() == expected {
                            let counts: Vec<usize> = lines
                                .filter_map(|l| l.trim().parse::<usize>().ok())
                                .collect();
                            if counts.len() == n_seqs {
                                cached_combos.push((ht, aa_idx, counts));
                            }
                        }
                    }
                }
            }
        }

        if cached_combos.is_empty() { return None; }

        // Build char_counts from cached_combos
        let mut seq_counts = vec![0usize; n_seqs];
        for (_, _, counts) in &cached_combos {
            for (si, &cnt) in counts.iter().enumerate() {
                seq_counts[si] += cnt;
            }
        }

        Some(RuntimeTrainingData {
            sequences,
            seq_counts,
            font_family,
            font_id_map,
            n_families,
            multi_variant_families,
            feat_dir,
            cached_combos,
            catalog,
            catalog_hash,
            render_params: render_params.clone(),
        })
    }

    /// Borrow as a TrainingContext for use with per-font classifier training.
    pub fn as_context<'a>(&'a self, glyph_map: &'a crate::glyph_map::NgramGlyphMap) -> TrainingContext<'a> {
        TrainingContext {
            sequences: &self.sequences,
            seq_counts: &self.seq_counts,
            font_family: &self.font_family,
            font_id_map: &self.font_id_map,
            glyph_map,
            n_families: self.n_families,
            multi_variant_families: self.multi_variant_families,
            min_fonts: 2,
            feat_dir: &self.feat_dir,
            cached_combos: &self.cached_combos,
            catalog: &self.catalog,
            catalog_hash: self.catalog_hash,
            render_params: &self.render_params,
        }
    }
}
