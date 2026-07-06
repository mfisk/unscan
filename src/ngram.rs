//! Bigram (character pair) classifier pipeline.
//!
//! Training: render pairs → build NgramGlyphMap → extract features → train LDA.
//! Inference: sliding window over word segments → bigram features → classify.
//!
//! Falls back to unigram classification when a bigram pair has no trained
//! classifier or the scan doesn't have enough characters for pairs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use image::GrayImage;
use rayon::prelude::*;

use crate::classifier::{NgramModel, ImageModel};
use crate::features::{self, compute_features, FEAT_LEN};
use crate::font_scan::FontEntry;
use crate::glyph_map::NgramGlyphMap;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn ngram_feat_dir(seq_len: usize) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let dirname = if seq_len == 1 { "chars".to_string() } else { format!("ngram-{seq_len}") };
    Path::new(&home).join(".cache").join("unprint").join("training").join(dirname)
}

// ---------------------------------------------------------------------------
// NgramGlyphMap construction
// ---------------------------------------------------------------------------

/// Build the NgramGlyphMap by rendering every (font, bigram) pair at default
/// params and grouping by content hash.  Analogous to the unigram GlyphMap
/// build in train.rs.
/// Add bigram entries to an existing glyph map and write the combined result.
pub fn build_ngram_glyph_map(
    catalog: &[FontEntry],
    glyph_map: &mut NgramGlyphMap,
) -> () {
    let bigrams = features::supported_sequences(2);
    eprintln!("\nBuilding bigram glyph map ({} fonts × {} bigrams)...",
        catalog.len(), bigrams.len());
    let t0 = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let total = catalog.len();

    // Each font returns Vec<(seq, hash, font_key)>
    // Uses render_ngram_fresh directly for parallel rendering; results
    // are registered into the glyph_map after collection.
    let per_font_hashes: Vec<Vec<(Vec<char>, u64, String)>> = catalog.par_iter().map(|fe| {
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
        let overrides = fe.glyph_overrides.as_deref();
        let fk = fe.font_key();
        let params = crate::char_render::RenderParams::default();

        let mut hashes = Vec::with_capacity(bigrams.len());
        for seq in bigrams {
            let gid_overrides: Vec<Option<ab_glyph::GlyphId>> = seq.iter().map(|c| {
                overrides.and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| ab_glyph::GlyphId(*g)))
            }).collect();

            if let Some(img) = crate::char_render::render_ngram_fresh(
                &font, seq, &gid_overrides, &params,
            ) {
                let hash = crate::glyph_map::hash_image(&img);
                // Write image cache
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
        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 500 == 0 || done == total {
            eprintln!("  NgramGlyphMap [{}/{}]...", done, total);
        }
        hashes
    }).collect();

    // Register all results into the glyph_map
    for font_hashes in per_font_hashes {
        for (seq, hash, font_key) in font_hashes {
            glyph_map.register(&seq, &font_key, hash);
        }
    }

    let total_glyphs: usize = glyph_map.groups.values().map(|g| g.len()).sum();
    let total_deduped: usize = glyph_map.groups.values()
        .flat_map(|gs| gs.iter())
        .filter(|g| g.font_keys.len() > 1)
        .map(|g| g.font_keys.len() - 1)
        .sum();
    eprintln!("  Combined glyph map: {} unique glyphs across {} seqs ({} duplicate renders eliminated)",
        total_glyphs, glyph_map.groups.len(), total_deduped);
    eprintln!("  Bigram glyph map complete in {:.1}s", t0.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------------------
// Bigram training data generation
// ---------------------------------------------------------------------------

/// Training sample for a bigram: glyph_id + feature vector.
#[derive(Clone)]
pub struct BigramSample {
    pub glyph_id: u32,
    pub features: [f32; FEAT_LEN],
}

/// Generate bigram training features for all (font, bigram, height, aa) combos.
/// Writes per-bigram feature files under the bigram feature directory.
pub fn generate_ngram_training_data(
    catalog: &[FontEntry],
    bgmap: &NgramGlyphMap,
    render_params: &crate::char_render::RenderParams,
) {
    let bigrams = features::supported_sequences(2);
    let all_aa = features::AaVariant::all();
    let heights: &[u32] = &[features::NORM_H]; // bigrams at native height only for now
    let feat_dir = ngram_feat_dir(2);
    std::fs::create_dir_all(&feat_dir).expect("create bigram feat dir");

    eprintln!("\nGenerating bigram training data ({} fonts × {} bigrams × {} heights × {} AA)...",
        catalog.len(), bigrams.len(), heights.len(), all_aa.len());
    let t0 = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let total = catalog.len();

    // Create output files for each bigram
    // We write all samples for a bigram into a single file: bigram_feat_dir/c1_c2.bin
    // Format per sample: glyph_id(u32) + features(FEAT_LEN × f32)

    // Pre-create bigram index mapping
    let bigram_to_idx: HashMap<Vec<char>, usize> = bigrams.iter()
        .enumerate()
        .map(|(i, bg)| (bg.clone(), i))
        .collect();

    // Collect samples per bigram across all fonts in parallel
    let chunk_size = 100;
    let n_bigrams = bigrams.len();

    // Use per-bigram atomic counts for progress
    let bigram_counts: Vec<AtomicUsize> = (0..n_bigrams).map(|_| AtomicUsize::new(0)).collect();

    for chunk_start in (0..catalog.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(catalog.len());
        let chunk = &catalog[chunk_start..chunk_end];

        // Each font in chunk → Vec<(bigram_idx, BigramSample)>
        let chunk_results: Vec<Vec<(usize, BigramSample)>> = chunk.par_iter().map(|fe| {
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
            let overrides = fe.glyph_overrides.as_deref();
            let fk = fe.font_key();
            let mut samples = Vec::new();

            for seq in bigrams {
            let (c1, c2) = (seq[0], seq[1]);
                let bi = match bigram_to_idx.get(&vec![c1, c2]) {
                    Some(&i) => i,
                    None => continue,
                };

                // Look up this font's bigram glyph_id
                let glyph_id = match bgmap.glyph_id_for_font(&[c1, c2], &fk) {
                    Some(id) => id as u32,
                    None => continue,
                };

                // Only render for the representative font (first in group)
                let rep_font = &bgmap.fonts_for_glyph(&[c1, c2], glyph_id as usize)[0];
                if *rep_font != fk {
                    continue;
                }

                let gid_overrides: Vec<Option<ab_glyph::GlyphId>> = [c1, c2].iter().map(|c| {
                    overrides.and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| ab_glyph::GlyphId(*g)))
                }).collect();

                for &ht in heights {
                    for &aa in all_aa {
                        let mut params = render_params.clone();
                        params.height = ht;
                        params.aa = aa;

                        let img = match crate::char_render::render_ngram_fresh(
                            &font, &[c1, c2], &gid_overrides, &params,
                        ) {
                            Some(img) => img,
                            None => continue,
                        };

                        let feats = match compute_features(&img, false) {
                            Some(f) => f,
                            None => continue,
                        };

                        samples.push((bi, BigramSample {
                            glyph_id,
                            features: feats.as_slice(),
                        }));
                    }
                }
            }
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 500 == 0 || done == total {
                eprintln!("  Bigram features [{}/{}]...", done, total);
            }
            samples
        }).collect();

        // Write chunk results to per-bigram files
        for font_samples in chunk_results {
            for (bi, sample) in font_samples {
                bigram_counts[bi].fetch_add(1, Ordering::Relaxed);
                let seq = &bigrams[bi]; let (c1, c2) = (seq[0], seq[1]);
                let path = feat_dir.join(format!("{:04X}_{:04X}.bin", c1 as u32, c2 as u32));
                let mut f = std::fs::OpenOptions::new()
                    .create(true).append(true).open(&path)
                    .expect("open bigram feature file");
                use std::io::Write;
                f.write_all(&sample.glyph_id.to_le_bytes()).expect("write glyph_id");
                for &v in &sample.features {
                    f.write_all(&v.to_le_bytes()).expect("write feature");
                }
            }
        }
    }

    let total_samples: usize = bigram_counts.iter().map(|c| c.load(Ordering::Relaxed)).sum();
    let trained_bigrams = bigram_counts.iter().filter(|c| c.load(Ordering::Relaxed) > 0).count();
    eprintln!("  Bigram training data: {} samples across {} bigrams in {:.1}s",
        total_samples, trained_bigrams, t0.elapsed().as_secs_f64());
}

/// Load bigram training samples from disk for a specific bigram.
fn load_bigram_samples(c1: char, c2: char) -> Vec<BigramSample> {
    let path = ngram_feat_dir(2).join(format!("{:04X}_{:04X}.bin", c1 as u32, c2 as u32));
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let sample_size = 4 + FEAT_LEN * 4; // glyph_id(4) + features(FEAT_LEN*4)
    let n = data.len() / sample_size;
    let mut samples = Vec::with_capacity(n);

    for i in 0..n {
        let off = i * sample_size;
        let glyph_id = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        let mut features = [0.0f32; FEAT_LEN];
        for j in 0..FEAT_LEN {
            features[j] = f32::from_le_bytes(
                data[off + 4 + j * 4..off + 4 + (j + 1) * 4].try_into().unwrap()
            );
        }
        samples.push(BigramSample { glyph_id, features });
    }
    samples
}

// ---------------------------------------------------------------------------
// Bigram LDA training
// ---------------------------------------------------------------------------

/// Train LDA projections for all bigrams and merge them into the existing
/// ngram model (which already contains unigram entries).
pub fn train_ngram_lda(
    _bgmap: &NgramGlyphMap,
    _font_id_map: &HashMap<String, u32>,
    _font_family: &[u32],
    model_path: &std::path::Path,
    lda_dims: usize,
    lda_reg: f64,
) {
    let bigrams = features::supported_sequences(2);
    let out_dim = lda_dims.min(FEAT_LEN - 1);
    eprintln!("\nTraining bigram LDA (target dim={})...", out_dim);
    let t0 = std::time::Instant::now();

    let mut model_entries: HashMap<Vec<char>, ImageModel> = HashMap::new();
    let mut skipped = 0usize;
    let mut trained = 0usize;

    for seq in bigrams {
            let (c1, c2) = (seq[0], seq[1]);
        let samples = load_bigram_samples(c1, c2);
        if samples.len() < 2 {
            skipped += 1;
            continue;
        }

        // Group samples by glyph_id
        let mut by_glyph: HashMap<u32, Vec<&[f32; FEAT_LEN]>> = HashMap::new();
        for s in &samples {
            by_glyph.entry(s.glyph_id).or_default().push(&s.features);
        }

        let n_classes = by_glyph.len();
        if n_classes < 2 {
            skipped += 1;
            continue;
        }

        // Compute LDA projection for this bigram
        let actual_dim = out_dim.min(n_classes - 1);

        // Compute class means and global mean
        let mut global_mean = vec![0.0f64; FEAT_LEN];
        let mut n_total = 0usize;

        let mut class_data: Vec<(u32, Vec<Vec<f64>>)> = Vec::new();
        for (&gid, feat_refs) in &by_glyph {
            let mut class_samples: Vec<Vec<f64>> = Vec::with_capacity(feat_refs.len());
            for feat in feat_refs {
                let sample: Vec<f64> = feat.iter().map(|&v| v as f64).collect();
                for (j, &v) in sample.iter().enumerate() {
                    global_mean[j] += v;
                }
                n_total += 1;
                class_samples.push(sample);
            }
            class_data.push((gid, class_samples));
        }
        for v in &mut global_mean { *v /= n_total as f64; }

        // Compute S_b (between-class scatter) and S_w (within-class scatter)
        let mut s_b = vec![0.0f64; FEAT_LEN * FEAT_LEN];
        let mut s_w = vec![0.0f64; FEAT_LEN * FEAT_LEN];

        for (_gid, class_samples) in &class_data {
            let nc = class_samples.len() as f64;
            let mut class_mean = vec![0.0f64; FEAT_LEN];
            for s in class_samples {
                for (j, &v) in s.iter().enumerate() { class_mean[j] += v; }
            }
            for v in &mut class_mean { *v /= nc; }

            // S_b += nc * (mean_c - mean_global) * (mean_c - mean_global)^T
            let diff: Vec<f64> = class_mean.iter().zip(&global_mean).map(|(a, b)| a - b).collect();
            for i in 0..FEAT_LEN {
                for j in 0..FEAT_LEN {
                    s_b[i * FEAT_LEN + j] += nc * diff[i] * diff[j];
                }
            }

            // S_w += sum_x (x - mean_c) * (x - mean_c)^T
            for s in class_samples {
                let d: Vec<f64> = s.iter().zip(&class_mean).map(|(a, b)| a - b).collect();
                for i in 0..FEAT_LEN {
                    for j in 0..FEAT_LEN {
                        s_w[i * FEAT_LEN + j] += d[i] * d[j];
                    }
                }
            }
        }

        // Regularize S_w
        for i in 0..FEAT_LEN {
            s_w[i * FEAT_LEN + i] += lda_reg;
        }

        // Solve S_w^{-1} S_b via Cholesky decomposition of S_w
        // For simplicity, use a direct eigendecomposition approach:
        // Compute S_w^{-1} S_b and take top eigenvectors.
        // We'll use a power iteration approach for the top eigenvectors.

        // First, compute L = cholesky(S_w), then solve L^{-1} S_b L^{-T}
        // For now, use a simpler approach: invert S_w directly (it's only 63×63)
        let s_w_inv = invert_matrix(&s_w, FEAT_LEN);
        if s_w_inv.is_none() {
            skipped += 1;
            continue;
        }
        let s_w_inv = s_w_inv.unwrap();

        // M = S_w^{-1} S_b
        let mut m = vec![0.0f64; FEAT_LEN * FEAT_LEN];
        for i in 0..FEAT_LEN {
            for j in 0..FEAT_LEN {
                let mut sum = 0.0f64;
                for k in 0..FEAT_LEN {
                    sum += s_w_inv[i * FEAT_LEN + k] * s_b[k * FEAT_LEN + j];
                }
                m[i * FEAT_LEN + j] = sum;
            }
        }

        // Power iteration to extract top `actual_dim` eigenvectors
        let proj = power_iteration_eigenvectors(&m, FEAT_LEN, actual_dim);

        // Project centroids: for each glyph, project its mean feature vector
        let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(by_glyph.len());
        for (gid, class_samples) in &class_data {
            let mut mean = vec![0.0f64; FEAT_LEN];
            for s in class_samples {
                for (j, &v) in s.iter().enumerate() { mean[j] += v; }
            }
            for v in &mut mean { *v /= class_samples.len() as f64; }

            // Project
            let mut projected = vec![0.0f32; actual_dim];
            for d in 0..actual_dim {
                let mut dot = 0.0f64;
                for j in 0..FEAT_LEN {
                    dot += proj[d * FEAT_LEN + j] * mean[j];
                }
                projected[d] = dot as f32;
            }
            centroids.push((*gid, projected));
        }

        // Store projection as weights: [out_dim, proj_matrix...]
        let mut weights = Vec::with_capacity(1 + actual_dim * FEAT_LEN);
        weights.push(actual_dim as f32);
        for d in 0..actual_dim {
            for j in 0..FEAT_LEN {
                weights.push(proj[d * FEAT_LEN + j] as f32);
            }
        }

        let mut cm = ImageModel { weights, centroids, sigma_sq: 0.0 };
        cm.compute_sigma_sq();

        model_entries.insert(vec![c1, c2], cm);
        trained += 1;
    }

    // Load existing model (unigram entries) and merge bigram entries in
    let data = std::fs::read(model_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", model_path.display()));
    let mut model = NgramModel::read_bin(&data, b"LDAC", None)
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", model_path.display()));
    model.entries.extend(model_entries);

    let mut f = std::fs::File::create(model_path).expect("create model file");
    model.write_bin(&mut f, b"LDAC", 5).expect("write combined ngram model");

    eprintln!("  Bigram LDA: trained {} bigrams, skipped {} ({:.1}s)",
        trained, skipped, t0.elapsed().as_secs_f64());
    eprintln!("  Combined model: {} total entries", model.entries.len());
    eprintln!("  Wrote {}", model_path.display());
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (63×63 matrix ops)
// ---------------------------------------------------------------------------

/// Invert a square matrix using Gauss-Jordan elimination.
fn invert_matrix(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = mat[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 { return None; }

        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }

        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }

        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

/// Extract top `k` eigenvectors of an n×n matrix via deflated power iteration.
/// Returns a flat vec of k eigenvectors, each of length n (row-major: [k × n]).
fn power_iteration_eigenvectors(mat: &[f64], n: usize, k: usize) -> Vec<f64> {
    use rand::prelude::*;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut result = Vec::with_capacity(k * n);
    let mut deflated = mat.to_vec();

    for _ in 0..k {
        // Random initial vector
        let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
        let mut norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v { *x /= norm; }

        // Power iteration (200 steps is plenty for 63×63)
        for _ in 0..200 {
            let mut new_v = vec![0.0f64; n];
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..n {
                    s += deflated[i * n + j] * v[j];
                }
                new_v[i] = s;
            }
            norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-30 { break; }
            for (x, nv) in v.iter_mut().zip(&new_v) { *x = nv / norm; }
        }

        result.extend_from_slice(&v);

        // Deflate: M = M - eigenvalue * v * v^T
        // eigenvalue ≈ v^T M v
        let mut eigenvalue = 0.0f64;
        for i in 0..n {
            let mut s = 0.0f64;
            for j in 0..n {
                s += deflated[i * n + j] * v[j];
            }
            eigenvalue += v[i] * s;
        }
        for i in 0..n {
            for j in 0..n {
                deflated[i * n + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Inference: sliding window ngram scoring
// ---------------------------------------------------------------------------

/// Score a font against a line's crops using sliding-window bigram
/// classification, falling back to unigram when a bigram classifier doesn't
/// exist for the pair.
///
/// `bigram_crops` are the combined pair images extracted from word-image
/// boundaries via `segment::extract_bigram_crops()`.
///
/// Returns `(font_key, score)` pairs sorted descending, same shape as
/// `identify_fonts`.
/// Build sliding-window observations and delegate to `identify_fonts`.
///
/// Walk a 2-observation window over word segments. At each position:
///   - If a bigram model exists for the pair → use bigram classifier/glyph_map, weight 1.0.
///   - If no bigram model and the left position is uncovered → use unigram, weight 0.5.
///   - If no bigram model and the right position is last and uncovered → use unigram, weight 0.5.
///   - Ligature codepoints (U+FB00–FB04) get weight 1.0 like bigrams.
///
/// Returns the same `FontIdResult` as `identify_fonts` — same structure, same audit detail.

/// Weight for a 1-gram observation: ligature codepoints score 1.0 like bigrams.
fn unigram_weight(c: char) -> f32 {
    match c {
        '\u{FB00}'..='\u{FB04}' => 1.0,
        _ => 0.5,
    }
}

pub fn build_scoring_windows<'a>(
    word_segs: &'a [crate::segment::WordSeg],
    classifier: &'a dyn crate::classifier::Classifier,
    _glyph_map: &'a crate::glyph_map::NgramGlyphMap,
    crop_store: &'a mut Vec<GrayImage>,
) -> Vec<crate::font_match::ScoringWindow<'a>> {
    // Flatten characters across all words into (seg_idx, pos_in_word, character)
    let flat: Vec<(usize, usize, char)> = word_segs.iter().enumerate()
        .flat_map(|(seg_idx, seg)| {
            seg.chars.iter().enumerate()
                .filter(|(_, c)| crate::features::is_supported(**c))
                .map(move |(pos, &c)| (seg_idx, pos, c))
        })
        .collect();

    let n = flat.len();
    if n == 0 {
        return Vec::new();
    }

    // Pass 1: crop into crop_store, record entries.
    // Try every adjacent pair as a bigram; unigrams fill gaps.
    let mut covered = vec![false; n];
    // entries: (flat_idx, store_idx, is_bigram)
    let mut entries: Vec<(usize, usize, bool)> = Vec::new();

    for i in 0..n - 1 {
        let (seg_i, pos_i, c1) = flat[i];
        let (seg_j, pos_j, c2) = flat[i + 1];
        let pair = [c1, c2];

        if seg_i == seg_j && pos_j == pos_i + 1 && classifier.has_sequence(&pair) {
            let seg = &word_segs[seg_i];
            if let Some(crop) = crate::segment::crop_ngram(
                &seg.word_img, pos_i, 2, &seg.boundaries, &seg.seam_paths, seg.crop_h,
            ) {
                let idx = crop_store.len();
                crop_store.push(crop);
                entries.push((i, idx, true));
                covered[i] = true;
                covered[i + 1] = true;
            }
        } else {
            // Left position: score as unigram if not already covered by a prior bigram
            if !covered[i] {
                let seg = &word_segs[seg_i];
                if let Some(crop) = crate::segment::crop_ngram(
                    &seg.word_img, pos_i, 1, &seg.boundaries, &seg.seam_paths, seg.crop_h,
                ) {
                    let idx = crop_store.len();
                    crop_store.push(crop);
                    entries.push((i, idx, false));
                    covered[i] = true;
                }
            }
            // Right position: if it's the last and won't get another chance
            if i + 1 == n - 1 && !covered[i + 1] {
                let seg = &word_segs[seg_j];
                if let Some(crop) = crate::segment::crop_ngram(
                    &seg.word_img, pos_j, 1, &seg.boundaries, &seg.seam_paths, seg.crop_h,
                ) {
                    let idx = crop_store.len();
                    crop_store.push(crop);
                    entries.push((i + 1, idx, false));
                    covered[i + 1] = true;
                }
            }
        }
    }

    // Edge case: single character on the line
    if n == 1 {
        let (seg_i, pos_i, _c1) = flat[0];
        let seg = &word_segs[seg_i];
        if let Some(crop) = crate::segment::crop_ngram(
            &seg.word_img, pos_i, 1, &seg.boundaries, &seg.seam_paths, seg.crop_h,
        ) {
            let idx = crop_store.len();
            crop_store.push(crop);
            entries.push((0, idx, false));
        }
    }

    // Pass 2: build ScoringWindows from entries
    let mut windows: Vec<crate::font_match::ScoringWindow<'a>> = Vec::with_capacity(entries.len());
    for &(fi, store_idx, is_bigram) in &entries {
        if is_bigram {
            let c1 = flat[fi].2;
            let c2 = flat[fi + 1].2;
            windows.push(crate::font_match::ScoringWindow {
                seq: vec![c1, c2],
                crop: &crop_store[store_idx],
                weight: 1.0,
            });
        } else {
            let c1 = flat[fi].2;
            windows.push(crate::font_match::ScoringWindow {
                seq: vec![c1],
                crop: &crop_store[store_idx],
                weight: unigram_weight(c1),
            });
        }
    }

    windows
}
