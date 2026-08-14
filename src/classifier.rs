//! Pluggable font classifiers for the character index.
//!
//! A **classifier** transforms raw FEAT_LEN-dim feature vectors into an embedding
//! space and defines a distance metric in that space.  The character index
//! stores embedded vectors and uses the classifier's distance function for
//! nearest-neighbor search.
//!
//! Two implementations ship today:
//!
//! - [`FisherClassifier`]: the original diagonal Fisher-weighted Euclidean
//!   distance.  Equivalent to the pre-refactor behaviour.
//! - [`TripletClassifier`]: per-glyph learned 3-layer MLPs (FEAT_LEN→128→64→32)
//!   trained with triplet loss.  One network per indexed character.


use crate::features::{CropFeatures, FEAT_LEN};

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OOD observation diagnostics (gated by UNPRINT_OBS_STATS=1)
// ---------------------------------------------------------------------------

/// Raw distance metrics from a single probabilities() call.
/// Populated only when UNPRINT_OBS_STATS=1.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ObsStats {
    pub min_d: f32,
    pub d_second: f32,
    pub sigma_sq: f32,
    pub med_nn: f32,
    pub n_centroids: usize,
    /// Pre-blend softmax winner probability.
    pub softmax_max: f32,
    /// Shannon entropy of the raw softmax distribution (nats).
    pub softmax_entropy: f32,
}

/// Whether to collect OOD observation stats (checked once at startup).
static OBS_STATS_ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("UNPRINT_OBS_STATS").map_or(false, |v| v == "1")
});

thread_local! {
    /// Sidecar from the last probabilities() call, if stats collection is enabled.
    static LAST_OBS_STATS: std::cell::RefCell<Option<ObsStats>> = const { std::cell::RefCell::new(None) };
}

/// Retrieve and clear the last observation stats (call after probabilities()/classify()).
pub fn take_obs_stats() -> Option<ObsStats> {
    if !*OBS_STATS_ENABLED { return None; }
    LAST_OBS_STATS.with(|cell| cell.borrow_mut().take())
}

thread_local! {
    /// OOD confidence weight: min(1, med_nn / min_d). Always written by softmax_probs().
    static LAST_OOD_WEIGHT: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

/// Retrieve the OOD confidence weight from the last softmax_probs() call.
/// Returns min(1, med_nn / min_d): 1.0 when on-distribution, decaying toward 0 for OOD.
pub fn take_ood_weight() -> f32 {
    LAST_OOD_WEIGHT.with(|cell| {
        let w = cell.get();
        cell.set(1.0);
        w
    })
}



fn stash_obs_stats(min_d: f32, dists: &[(u32, f32)], sigma_sq: f32, med_nn: f32, softmax_probs: &[(u32, f32)]) {
    if !*OBS_STATS_ENABLED { return; }
    let n = dists.len();
    let mut d_second = f32::INFINITY;
    for &(_, d) in dists {
        if d > min_d && d < d_second {
            d_second = d;
        }
    }
    let softmax_max = softmax_probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
    let softmax_entropy = -softmax_probs.iter()
        .map(|(_, p)| if *p > 1e-30 { p * p.ln() } else { 0.0 })
        .sum::<f32>();
    LAST_OBS_STATS.with(|cell| {
        *cell.borrow_mut() = Some(ObsStats {
            min_d,
            d_second,
            sigma_sq,
            med_nn,
            n_centroids: n,
            softmax_max,
            softmax_entropy,
        });
    });
}

/// Shared softmax probability computation over pre-computed squared distances.
/// Both ImageModel and MmapNgramModel delegate here to avoid duplication.
/// Returns probabilities in input order plus uniform flag (true when all p equal).
#[inline]
fn softmax_unsorted(dists: &[(u32, f32)], sigma_sq: f32, med_nn: f32) -> (Vec<(u32, f32)>, bool) {
    if dists.is_empty() { return (Vec::new(), true); }
    let sigma = if sigma_sq > 1e-30 {
        sigma_sq
    } else {
        let p = 1.0 / dists.len() as f32;
        let uniform: Vec<(u32, f32)> = dists.iter().map(|(id, _)| (*id, p)).collect();
        // For degenerate sigma, OOD weight is 1.0 and min_d is 0 → stash with min_d=0
        // to match previous early-return behaviour (no min_d calc originally).
        // Compute min_d for stash consistency anyway.
        let min_d = dists.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
        let ood_w = 1.0;
        LAST_OOD_WEIGHT.with(|cell| cell.set(ood_w));
        stash_obs_stats(min_d, dists, sigma_sq, med_nn, &uniform);
        return (uniform, true);
    };
    let inv2s = 1.0 / (2.0 * sigma);
    let min_d = dists.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
    let raw: Vec<f32> = dists.iter().map(|(_, d)| (-(d - min_d) * inv2s).exp()).collect();
    let sum: f32 = raw.iter().sum();
    let (softmax, is_uniform) = if sum < 1e-30 {
        let p = 1.0 / dists.len() as f32;
        (dists.iter().map(|(id, _)| (*id, p)).collect::<Vec<(u32, f32)>>(), true)
    } else {
        (dists.iter().zip(raw.iter()).map(|((id, _), &r)| (*id, r / sum)).collect::<Vec<(u32, f32)>>(), false)
    };
    // Always stash OOD confidence weight (not gated by UNPRINT_OBS_STATS)
    let ood_w = if min_d > 1e-30 && med_nn > 1e-30 {
        (med_nn / min_d).min(1.0)
    } else {
        1.0
    };
    LAST_OOD_WEIGHT.with(|cell| cell.set(ood_w));
    stash_obs_stats(min_d, dists, sigma, med_nn, &softmax);
    (softmax, is_uniform)
}

/// Shared softmax probability computation over pre-computed squared distances.
/// Both ImageModel and MmapNgramModel delegate here to avoid duplication.
fn softmax_probs(dists: &[(u32, f32)], sigma_sq: f32, med_nn: f32) -> Vec<(u32, f32)> {
    let (mut probs, is_uniform) = softmax_unsorted(dists, sigma_sq, med_nn);
    if is_uniform {
        // Preserve original behaviour: uniform returns input order (no sort)
        return probs;
    }
    // Perf: prob tie → glyph_id asc for deterministic top-k (affects candidate fonts & path winner, needed for stable t59).
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    probs
}

/// Select top-k by (prob desc, id asc) without full sort.
/// Preserves original uniform-input-order behaviour for uniform case.
#[inline]
fn top_k_by_prob(mut probs: Vec<(u32, f32)>, k: usize, is_uniform: bool) -> Vec<(u32, f32)> {
    if probs.len() <= k || is_uniform {
        // Uniform: keep input order, just truncate, to match original softmax_probs→truncate
        // Non-uniform small vec: full sort is cheaper than select_nth overhead
        if !is_uniform && probs.len() > 1 {
            probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        }
        probs.truncate(k);
        return probs;
    }
    // Partial selection O(n) + sort top k O(k log k)
    let kth = k - 1;
    probs.select_nth_unstable_by(kth, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
    });
    probs.truncate(k);
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    probs
}

// ---------------------------------------------------------------------------
// Binary weight-file reader
// ---------------------------------------------------------------------------

/// Cursor-based reader for binary weight files (f32 LE arrays + u32 LE headers).
struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn read_f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let need = n * 4;
        if self.pos + need > self.data.len() {
            return Err("truncated weight data".into());
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap()));
            self.pos += 4;
        }
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("truncated header data".into());
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Shared dense linear layer
// ---------------------------------------------------------------------------

/// Inference-only linear layer (no gradients, no optimizer state).
struct InferenceLinear {
    rows: usize,
    cols: usize,
    w: Vec<f32>, // rows × cols, row-major
    b: Vec<f32>, // cols
}

impl InferenceLinear {
    /// Forward: output[j] = sum_i(input[i] * w[i*cols+j]) + b[j]
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
}

/// Dense matrix × vector: y[i] = sum_j(mat[i*FEAT_LEN + j] * x[j]).
/// Shared by Mahalanobis (square) and LDA (rectangular) classifiers.
fn dense_project(out_dim: usize, mat: &[f32], x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let row = &mat[i * FEAT_LEN..(i + 1) * FEAT_LEN];
        let mut sum = 0.0f32;
        for j in 0..FEAT_LEN {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
    y
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A font classifier that identifies fonts from character glyph images.
///
/// # Scoring model
///
/// All classifiers expose a single score space: calibrated posterior
/// probabilities `p(font | query, char)` in `[0, 1]`, summing to 1 across
/// all fonts for a given query.  These are comparable across classifiers,
/// characters, and queries.
///
/// `classify()` returns the top-k fonts by probability.  `probabilities()`
/// returns all fonts.  `probability()` returns a single font's posterior.
///
/// # Computing probabilities
///
/// Embedding-based classifiers (LDA, Fisher, Triplet, Mahalanobis) model
/// each font as a point in a learned embedding space.  The posterior is
/// derived via a Gaussian (RBF) kernel:
///
/// ```text
///   p(font_i | query, ch) = exp(-d_i / 2σ²_ch) / Σ_j exp(-d_j / 2σ²_ch)
/// ```
///
/// where `d_i` is the squared Euclidean distance from the query embedding to
/// font i's centroid, and `σ²_ch` is a per-character bandwidth parameter.
///
/// The bandwidth `σ²` is set to the **median pairwise squared distance**
/// between all font centroids for that character (the "median heuristic"
/// from kernel density estimation).  This choice maximizes the entropy of
/// the resulting kernel matrix, ensuring the probability distribution is
/// neither too peaked (only the nearest font gets any mass) nor too flat
/// (all fonts are equally likely).  It is computed at training time from the
/// projected class centroids and stored in the weight file.
///
/// MLP classifiers produce probabilities directly via softmax over learned
/// logits — no bandwidth parameter is needed.
///
/// Fusion classifiers compute probabilities from each child's probability
/// distribution, combined via a weighted geometric mean (equivalent to
/// weighted log-probability averaging), then renormalized.
///
/// # σ² storage
///
/// For embedding classifiers, `σ²` is computed during training and stored
/// per-character in the weight file (LDAC v2).  At runtime, `sigma_sq(ch)`
/// returns the stored value.  If the weight file predates v2, `σ²` is
/// computed on first access from the stored NgramModel centroids as a
/// fallback.
#[allow(dead_code)]
pub trait Classifier: Send + Sync {
    /// Return the top `k` glyph matches for a character crop.
    /// Returns `(glyph_id, probability)` sorted descending (highest = best).
    /// glyph_id indexes into the GlyphMap for the given character.
    fn classify(&self, seq: &[char], query: &CropFeatures, k: usize) -> Vec<(usize, f32)>;

    /// Return calibrated posterior probabilities for all glyphs, sorted
    /// descending by probability.  Probabilities sum to 1.
    fn probabilities(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        self.classify(seq, query, self.glyph_count(seq))
    }

    /// Posterior probability of a specific glyph given a query.
    fn probability(&self, seq: &[char], query: &CropFeatures, glyph_id: usize) -> Option<f32> {
        self.probabilities(seq, query).iter()
            .find(|(id, _)| *id == glyph_id)
            .map(|(_, p)| *p)
    }

    /// Raw classifier logits ` -d²/(2σ²)` — NOT softmaxed.
    /// Use for `softmax(logit + geo)`. Default falls back to `ln p`
    /// (which differs by a per-query constant, so ranking and
    /// `lp - best_lp` scoring are unchanged).
    fn raw_logits(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        self.probabilities(seq, query).into_iter()
            .map(|(id, p)| (id, if p > 0.0 { p.ln() } else { f32::NEG_INFINITY }))
            .collect()
    }

    fn raw_logit(&self, seq: &[char], query: &CropFeatures, glyph_id: usize) -> Option<f32> {
        self.raw_logits(seq, query).into_iter()
            .find(|(id, _)| *id == glyph_id)
            .map(|(_, l)| l)
    }

    /// Short name for logging and cache invalidation.
    fn name(&self) -> &str;

    /// Number of distinct glyphs for a sequence.
    fn glyph_count(&self, seq: &[char]) -> usize;

    /// Whether the classifier has a trained model for this sequence.
    fn has_sequence(&self, seq: &[char]) -> bool {
        self.glyph_count(seq) > 0
    }

    /// Feed a glyph's feature vector for a character into the classifier.
    /// Called once per (glyph_id, char) pair during index build.
    fn add_glyph(&mut self, _glyph_id: usize, _seq: &[char], _features: &CropFeatures) {}

    /// Ensure the model's storage is owned (convert mmap → owned) so add_glyph works.
    fn ensure_owned(&mut self) {}

    /// Recompute per-character σ² and med_nn after incremental additions.
    fn recompute_stats(&mut self) {}

    /// Update catalog_hash after incremental addition.
    fn set_catalog_hash(&mut self, _hash: u64) {}

    /// Persist the updated model to disk (for incremental updates).
    fn save_to(&self, _path: &std::path::Path, _magic: &[u8; 4], _version: u32) -> Result<(), String> { Ok(()) }

    /// Catalog hash baked into the model at training time, if available.
    /// Used by load_or_train to detect stale weights when the font catalog changes.
    fn catalog_hash(&self) -> Option<u64> { None }
}

/// Convert `(font_id, sq_dist)` pairs to `(font_id, prob)` pairs using a
/// Gaussian kernel with bandwidth `sigma_sq`.  Mutates in place — no extra
/// allocation beyond the input vector.
// ---------------------------------------------------------------------------
// Helpers for embedding-based classifiers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two slices.
fn sq_euclid(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum()
}


// ---------------------------------------------------------------------------
// ImageModel — per-character complete model state
// ---------------------------------------------------------------------------

/// Complete model state for a single character.  Co-locates the classifier-
/// specific weights (Fisher scores, LDA projection, Mahalanobis L_inv, …)
/// with the embedded font centroids and the probability-calibration σ².
///
/// Replaces the old split where weights lived in the Embedder and centroids
/// lived in a separate NgramModel.


// ---------------------------------------------------------------------------

#[allow(dead_code)] // called from ngram.rs (currently dead-code module)
pub fn pairwise_sigma_sq(centroids: &[(u32, Vec<f32>)]) -> f32 {
    let n = centroids.len();
    if n < 2 { return 0.0; }
    let mut dists: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(sq_euclid(&centroids[i].1, &centroids[j].1));
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists[dists.len() / 2]
}


pub struct ImageModel {
    /// Classifier-specific weights as a flat f32 blob.
    /// Interpretation depends on the classifier type (magic byte in .bin header).
    pub weights: Vec<f32>,
    /// Embedded font centroids: (font_id, embedded_vec).
    pub centroids: Vec<(u32, Vec<f32>)>,
    /// Gaussian bandwidth for probability calibration (median pairwise
    /// squared distance among centroids).  0.0 means not yet computed.
    pub sigma_sq: f32,
    /// OOD gating threshold: median nearest-foreign-centroid d².
    /// Used for Cauchy OOD confidence blending.  0.0 means not yet computed.
    pub med_nn: f32,
}

impl ImageModel {
    /// Probability of each font given a query vector, sorted descending.
    pub fn probabilities(&self, query: &[f32]) -> Vec<(u32, f32)> {
        let dists: Vec<(u32, f32)> = self.centroids.iter()
            .map(|(id, stored)| (*id, sq_euclid(query, stored)))
            .collect();
        softmax_probs(&dists, self.sigma_sq, self.med_nn)
    }

    /// Raw classifier logits (unnormalized log-probs) for each font.
    /// `logit = -d² / (2·sigma_sq)`. Does NOT softmax — caller must softmax
    /// after adding geometry terms. Returns uniform 0 when sigma_sq is degenerate
    /// (equivalent to uniform softmax).
    pub fn raw_logits(&self, query: &[f32]) -> Vec<(u32, f32)> {
        if self.sigma_sq <= 1e-30 {
            return self.centroids.iter().map(|(id, _)| (*id, 0.0f32)).collect();
        }
        let inv2s = 1.0 / (2.0 * self.sigma_sq);
        self.centroids.iter()
            .map(|(id, stored)| (*id, -sq_euclid(query, stored) * inv2s))
            .collect()
    }

    /// Top-k fonts by probability – uses partial selection O(n) + sort top k,
    /// not full O(n log n) sort, preserving deterministic tie-break (prob desc, id asc).
    pub fn classify(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if k == 0 { return Vec::new(); }
        let dists: Vec<(u32, f32)> = self.centroids.iter()
            .map(|(id, stored)| (*id, sq_euclid(query, stored)))
            .collect();
        let (probs, is_uniform) = softmax_unsorted(&dists, self.sigma_sq, self.med_nn);
        top_k_by_prob(probs, k, is_uniform)
    }

    /// Compute σ² from stored centroids (median pairwise squared distance).
    pub fn compute_sigma_sq(&mut self) {
        let n = self.centroids.len();
        if n < 2 { return; }
        let mut dists: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(sq_euclid(&self.centroids[i].1, &self.centroids[j].1));
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = dists[dists.len() / 2];
        if median > 1e-30 { self.sigma_sq = median; }
    }

    /// Compute median nearest-neighbor squared distance among centroids.
    pub fn compute_med_nn(&mut self) {
        let n = self.centroids.len();
        if n < 2 { return; }
        let mut nn_dists: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let mut best = f32::INFINITY;
            for j in 0..n {
                if i == j { continue; }
                let d = sq_euclid(&self.centroids[i].1, &self.centroids[j].1);
                if d < best { best = d; }
            }
            nn_dists.push(best);
        }
        nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = nn_dists[nn_dists.len() / 2];
        if median > 1e-30 { self.med_nn = median; }
    }
}

/// Per-character classifier with co-located weights, centroids, and σ².
/// Font names are shared across all characters; centroids reference fonts
/// by font_id (index into `font_names` / catalog).
pub struct NgramModel {
    pub entries: HashMap<Vec<char>, ImageModel>,
    /// Catalog hash at training time.  Used to reject stale .bin files
    /// when the font catalog changes.
    pub catalog_hash: u64,
}


/// On-disk header for the indexed mmap binary format (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct IndexedHeader {
    magic: [u8; 4],
    version: u32,
    catalog_hash: u64,
    n_fonts: u32,
    n_entries: u32,
    index_off: u32,
    data_off: u32,
}

/// Fixed fields of one index entry (28 bytes), written after the variable-
/// length codepoint array.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct IndexedEntryFixed {
    n_weights: u32,
    weights_off: u32,
    n_centroids: u32,
    gids_off: u32,
    vecs_off: u32,
    vec_dim: u32,
    sigma_sq_bits: u32,
    med_nn_bits: u32,
}

impl NgramModel {
    pub fn new(catalog_hash: u64) -> Self {
        Self { entries: HashMap::new(), catalog_hash }
    }

    /// Serialize to the indexed mmap-friendly .bin format.
    ///
    /// Version number has high bit set (0x8000_0000) to distinguish from
    /// the legacy sequential format.  Data section is 4-byte aligned so
    /// float arrays can be read as zero-copy slices from an mmap.
    pub fn write_bin(
        &self,
        w: &mut dyn std::io::Write,
        magic: &[u8; 4],
        version: u32,
    ) -> std::io::Result<()> {
        let n_entries = self.entries.len();

        // Build data section: for each entry, pack weights, glyph_ids, centroid vecs
        let mut data_buf: Vec<u8> = Vec::new();
        struct EntryLayout {
            seq: Vec<char>,
            fixed: IndexedEntryFixed,
        }
        let mut layouts: Vec<EntryLayout> = Vec::with_capacity(n_entries);

        // Sort entries for deterministic output
        let mut sorted_entries: Vec<_> = self.entries.iter().collect();
        sorted_entries.sort_by_key(|(seq, _)| (*seq).clone());

        for (seq, cm) in &sorted_entries {
            let weights_off = data_buf.len();
            for &wt in &cm.weights {
                data_buf.extend_from_slice(&wt.to_le_bytes());
            }
            let gids_off = data_buf.len();
            for &(gid, _) in &cm.centroids {
                data_buf.extend_from_slice(&gid.to_le_bytes());
            }
            let vecs_off = data_buf.len();
            let vec_dim = if cm.centroids.is_empty() { 0 }
                          else { cm.centroids[0].1.len() };
            for (_, v) in &cm.centroids {
                debug_assert!(v.len() == vec_dim || cm.centroids.is_empty());
                for &f in v {
                    data_buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            layouts.push(EntryLayout {
                seq: seq.to_vec(),
                fixed: IndexedEntryFixed {
                    n_weights: cm.weights.len() as u32,
                    weights_off: weights_off as u32,
                    n_centroids: cm.centroids.len() as u32,
                    gids_off: gids_off as u32,
                    vecs_off: vecs_off as u32,
                    vec_dim: vec_dim as u32,
                    sigma_sq_bits: cm.sigma_sq.to_bits(),
                    med_nn_bits: cm.med_nn.to_bits(),
                },
            });
        }

        // Compute section offsets (no font names — empty font table)
        let hdr_size = std::mem::size_of::<IndexedHeader>();
        let entry_fixed_size = std::mem::size_of::<IndexedEntryFixed>();
        let index_off = hdr_size; // no font section
        let index_size: usize = layouts.iter()
            .map(|l| 4 + l.seq.len() * 4 + entry_fixed_size)
            .sum();
        let data_off = (index_off + index_size + 3) & !3;

        // Header
        let hdr = IndexedHeader {
            magic: *magic,
            version: version | 0x8000_0000,
            catalog_hash: self.catalog_hash,
            n_fonts: 0,
            n_entries: n_entries as u32,
            index_off: index_off as u32,
            data_off: data_off as u32,
        };
        w.write_all(unsafe {
            std::slice::from_raw_parts(&hdr as *const _ as *const u8, hdr_size)
        })?;

        // Entry index
        for layout in &layouts {
            w.write_all(&(layout.seq.len() as u32).to_le_bytes())?;
            for &ch in &layout.seq {
                w.write_all(&(ch as u32).to_le_bytes())?;
            }
            w.write_all(unsafe {
                std::slice::from_raw_parts(
                    &layout.fixed as *const _ as *const u8, entry_fixed_size)
            })?;
        }

        // Alignment padding before data section
        let idx_end = index_off + index_size;
        let data_pad = data_off - idx_end;
        if data_pad > 0 {
            w.write_all(&vec![0u8; data_pad])?;
        }

        // Data section
        w.write_all(&data_buf)?;
        Ok(())
    }


}

/// Read a u32 LE from `data` at `*pos`, advancing `*pos`.
#[allow(dead_code)] // read-side pair of the write serialization above
fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() { return Err("truncated u32".into()); }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

/// Read `n` f32 LE values from `data` at `*pos`, advancing `*pos`.
#[allow(dead_code)]
fn read_f32s(data: &[u8], pos: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let need = n * 4;
    if *pos + need > data.len() { return Err("truncated f32 array".into()); }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()));
        *pos += 4;
    }
    Ok(out)
}


/// Reusable storage for classifiers that do brute-force distance search
/// over embedded font vectors. Each embedding-based classifier composes
/// this internally.

// ---------------------------------------------------------------------------
// EmbeddingClassifier — shared Classifier impl for embed-then-store classifiers
// ---------------------------------------------------------------------------

/// Trait for the embedding step: converts raw features into a classifier-specific vector.
#[allow(dead_code)]
pub trait Embedder: Send + Sync {
    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32>;
    fn name(&self) -> &str;
}

/// Generic classifier that embeds features via an [`Embedder`] then searches
/// pre-computed centroids stored in a [`NgramModel`].

// ---------------------------------------------------------------------------
// Memory-mapped model — zero-copy access to centroids via mmap
// ---------------------------------------------------------------------------

/// Index entry for one character in the mmap'd file.
struct MmapEntryIndex {
    weights_off: usize,   // byte offset to weights [f32; n_weights]
    n_weights: usize,
    glyph_ids_off: usize, // byte offset to [u32; n_centroids]
    vecs_off: usize,      // byte offset to [f32; n_centroids * vec_dim]
    n_centroids: usize,
    vec_dim: usize,
    sigma_sq: f32,
    med_nn: f32,
}

/// Zero-copy NgramModel backed by a memory-mapped indexed binary file.
///
/// File layout (all values little-endian, data section 4-byte aligned):
/// ```text
/// magic:          [u8; 4]
/// version:        u32
/// catalog_hash:   u64
/// n_fonts:        u32
/// n_entries:      u32
/// index_off:      u32       — byte offset to entry index section
/// data_off:       u32       — byte offset to data section
///
/// FONT NAMES (offset 32):
///   per font: name_len: u32, name_bytes: [u8; name_len]
///
/// ENTRY INDEX (at index_off, 32 bytes per entry):
///   seq_len:      u32
///   codepoints:   [u32; seq_len]  — one u32 per character
///   n_weights:    u32
///   weights_off:  u32       — relative to data_off
///   n_centroids:  u32
///   gids_off:     u32       — relative to data_off
///   vecs_off:     u32       — relative to data_off
///   vec_dim:      u32       — unused padding / future multi-char
///   sigma_sq_bits: u32      — f32 as u32 bits
///   (entry size = 4 + seq_len*4 + 28 bytes)
///
/// DATA SECTION (at data_off, 4-byte aligned):
///   [u32 glyph-id arrays and f32 weight/centroid arrays, contiguous]
/// ```
pub struct MmapNgramModel {
    mmap: memmap2::Mmap,
    _file: std::fs::File,
    pub catalog_hash: u64,
    entries: HashMap<Vec<char>, MmapEntryIndex>,
    #[allow(dead_code)] // stored for debugging/future use
    pub font_names: Vec<String>,
}

impl MmapNgramModel {
    #[inline]
    fn f32_slice(&self, off: usize, n: usize) -> &[f32] {
        let end = off + n * 4;
        debug_assert!(off % 4 == 0, "unaligned f32 read at offset {off}");
        debug_assert!(end <= self.mmap.len(), "f32 slice out of bounds");
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr().add(off) as *const f32, n) }
    }

    #[inline]
    fn u32_slice(&self, off: usize, n: usize) -> &[u32] {
        let end = off + n * 4;
        debug_assert!(off % 4 == 0, "unaligned u32 read at offset {off}");
        debug_assert!(end <= self.mmap.len(), "u32 slice out of bounds");
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr().add(off) as *const u32, n) }
    }

    /// Probability of each font given a query vector, for the given char.
    /// Returns None if the character is unknown.
    pub fn probabilities(&self, seq: &[char], query: &[f32]) -> Option<Vec<(u32, f32)>> {
        let e = self.entries.get(seq)?;
        if e.n_centroids == 0 { return Some(Vec::new()); }
        let gids = self.u32_slice(e.glyph_ids_off, e.n_centroids);
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(e.n_centroids);
        for i in 0..e.n_centroids {
            let v = self.f32_slice(e.vecs_off + i * e.vec_dim * 4, e.vec_dim);
            dists.push((gids[i], sq_euclid(query, v)));
        }
        Some(softmax_probs(&dists, e.sigma_sq, e.med_nn))
    }

    /// Raw logits ` -d² / (2·sigma_sq)` — NOT softmaxed. Use for `softmax(logit + geo)`.
    pub fn raw_logits(&self, seq: &[char], query: &[f32]) -> Option<Vec<(u32, f32)>> {
        let e = self.entries.get(seq)?;
        if e.n_centroids == 0 { return Some(Vec::new()); }
        if e.sigma_sq <= 1e-30 {
            let gids = self.u32_slice(e.glyph_ids_off, e.n_centroids);
            return Some(gids.iter().map(|&gid| (gid, 0.0f32)).collect());
        }
        let gids = self.u32_slice(e.glyph_ids_off, e.n_centroids);
        let inv2s = 1.0 / (2.0 * e.sigma_sq);
        let mut out = Vec::with_capacity(e.n_centroids);
        for i in 0..e.n_centroids {
            let v = self.f32_slice(e.vecs_off + i * e.vec_dim * 4, e.vec_dim);
            let d2 = sq_euclid(query, v);
            out.push((gids[i], -d2 * inv2s));
        }
        Some(out)
    }

    /// Top-k fonts by probability for a given char – O(n)+ O(k log k) partial selection.
    pub fn classify(&self, seq: &[char], query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if k == 0 { return Vec::new(); }
        let e = match self.entries.get(seq) {
            Some(e) => e,
            None => return Vec::new(),
        };
        if e.n_centroids == 0 { return Vec::new(); }
        let gids = self.u32_slice(e.glyph_ids_off, e.n_centroids);
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(e.n_centroids);
        for i in 0..e.n_centroids {
            let v = self.f32_slice(e.vecs_off + i * e.vec_dim * 4, e.vec_dim);
            dists.push((gids[i], sq_euclid(query, v)));
        }
        let (probs, is_uniform) = softmax_unsorted(&dists, e.sigma_sq, e.med_nn);
        top_k_by_prob(probs, k, is_uniform)
    }

    /// Number of centroids (fonts) for a character.
    pub fn glyph_count(&self, seq: &[char]) -> usize {
        self.entries.get(seq).map_or(0, |e| e.n_centroids)
    }

    /// Glyph ID slice for a character (zero-copy from mmap).
    #[allow(dead_code)]
    fn glyph_ids(&self, seq: &[char]) -> &[u32] {
        match self.entries.get(seq) {
            Some(e) if e.n_centroids > 0 => self.u32_slice(e.glyph_ids_off, e.n_centroids),
            _ => &[],
        }
    }

    /// Centroid vector data for a character — flat f32 slice, n_centroids × vec_dim (zero-copy).
    #[allow(dead_code)]
    fn centroid_vecs(&self, seq: &[char]) -> &[f32] {
        match self.entries.get(seq) {
            Some(e) if e.n_centroids > 0 && e.vec_dim > 0 => {
                self.f32_slice(e.vecs_off, e.n_centroids * e.vec_dim)
            }
            _ => &[],
        }
    }

    /// All character sequences in the model.
    pub fn entry_keys(&self) -> impl Iterator<Item = &Vec<char>> {
        self.entries.keys()
    }

    /// Reconstruct an owned NgramModel from the mmap (for merge-and-rewrite).
    #[allow(dead_code)]
    pub fn to_owned_model(&self) -> NgramModel {
        let mut entries = HashMap::with_capacity(self.entries.len());
        for (seq, idx) in &self.entries {
            let weights = self.weights(seq).map(|s| s.to_vec()).unwrap_or_default();
            let mut centroids = Vec::with_capacity(idx.n_centroids);
            let gids = self.glyph_ids(seq);
            let vecs = self.centroid_vecs(seq);
            for i in 0..idx.n_centroids {
                let gid = gids[i];
                let start = i * idx.vec_dim;
                let end = start + idx.vec_dim;
                centroids.push((gid, vecs[start..end].to_vec()));
            }
            entries.insert(seq.clone(), ImageModel { weights, centroids, sigma_sq: idx.sigma_sq, med_nn: idx.med_nn });
        }
        NgramModel { entries, catalog_hash: self.catalog_hash }
    }

    /// Get the weights slice for a character (used by embedders at load time).
    pub fn weights(&self, seq: &[char]) -> Option<&[f32]> {
        let e = self.entries.get(seq)?;
        Some(self.f32_slice(e.weights_off, e.n_weights))
    }

    /// Load an indexed mmap file.  Returns Err if format is wrong.
    pub fn load_indexed(
        path: &std::path::Path,
        magic: &[u8; 4],
    ) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("cannot mmap {}: {e}", path.display()))?;
        let data = &mmap[..];
        let hdr_size = std::mem::size_of::<IndexedHeader>();
        let entry_fixed_size = std::mem::size_of::<IndexedEntryFixed>();
        if data.len() < hdr_size {
            return Err("file too small for indexed header".into());
        }
        let hdr = unsafe { &*(data.as_ptr() as *const IndexedHeader) };
        if &hdr.magic != magic {
            return Err(format!("bad magic: expected {:?}, got {:?}", magic, hdr.magic));
        }
        if hdr.version & 0x8000_0000 == 0 {
            return Err(format!("not an indexed file (version {})", hdr.version));
        }
        let n_fonts = hdr.n_fonts as usize;
        let n_entries = hdr.n_entries as usize;
        let index_off = hdr.index_off as usize;
        let data_off = hdr.data_off as usize;

        // Read font names
        let mut pos = hdr_size;
        let mut font_names = Vec::with_capacity(n_fonts);
        for _ in 0..n_fonts {
            if pos + 4 > data.len() { return Err("truncated font name".into()); }
            let nlen = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + nlen > data.len() { return Err("truncated font name data".into()); }
            let name = String::from_utf8_lossy(&data[pos..pos+nlen]).into_owned();
            pos += nlen;
            font_names.push(name);
        }

        // Read entry index
        let mut entries = HashMap::with_capacity(n_entries);
        pos = index_off;
        for _ in 0..n_entries {
            if pos + 4 > data.len() { return Err("truncated entry index".into()); }
            let seq_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            let cp_bytes = seq_len * 4;
            if pos + cp_bytes + entry_fixed_size > data.len() {
                return Err("truncated entry index".into());
            }
            let mut seq = Vec::with_capacity(seq_len);
            for _ in 0..seq_len {
                let cp = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                seq.push(char::from_u32(cp).ok_or_else(|| format!("bad codepoint U+{cp:04X}"))?);
            }
            let ef = unsafe { &*(data.as_ptr().add(pos) as *const IndexedEntryFixed) };
            pos += entry_fixed_size;

            entries.insert(seq, MmapEntryIndex {
                weights_off: data_off + ef.weights_off as usize,
                n_weights: ef.n_weights as usize,
                glyph_ids_off: data_off + ef.gids_off as usize,
                vecs_off: data_off + ef.vecs_off as usize,
                n_centroids: ef.n_centroids as usize,
                vec_dim: ef.vec_dim as usize,
                sigma_sq: f32::from_bits(ef.sigma_sq_bits),
                med_nn: f32::from_bits(ef.med_nn_bits),
            });
        }

        Ok(MmapNgramModel { mmap, _file: file, catalog_hash: hdr.catalog_hash, entries, font_names })
    }
}



/// Dispatches between owned NgramModel and zero-copy MmapNgramModel.
pub enum CharModelStore {
    #[allow(dead_code)]
    Owned(NgramModel),
    Mmap(MmapNgramModel),
}

impl CharModelStore {
    pub fn probabilities(&self, seq: &[char], query: &[f32]) -> Vec<(u32, f32)> {
        match self {
            CharModelStore::Owned(m) => {
                m.entries.get(seq).map_or_else(Vec::new, |cm| cm.probabilities(query))
            }
            CharModelStore::Mmap(m) => {
                m.probabilities(seq, query).unwrap_or_default()
            }
        }
    }

    /// Raw logits ( -d² / 2σ² ), NOT softmaxed. For correct `softmax(logit + geo)`.
    pub fn raw_logits(&self, seq: &[char], query: &[f32]) -> Vec<(u32, f32)> {
        match self {
            CharModelStore::Owned(m) => {
                m.entries.get(seq).map_or_else(Vec::new, |cm| cm.raw_logits(query))
            }
            CharModelStore::Mmap(m) => {
                m.raw_logits(seq, query).unwrap_or_default()
            }
        }
    }

    pub fn classify(&self, seq: &[char], query: &[f32], k: usize) -> Vec<(u32, f32)> {
        match self {
            CharModelStore::Owned(m) => {
                m.entries.get(seq).map_or_else(Vec::new, |cm| cm.classify(query, k))
            }
            CharModelStore::Mmap(m) => {
                m.classify(seq, query, k)
            }
        }
    }

    pub fn glyph_count(&self, seq: &[char]) -> usize {
        match self {
            CharModelStore::Owned(m) => m.entries.get(seq).map_or(0, |cm| cm.centroids.len()),
            CharModelStore::Mmap(m) => m.glyph_count(seq),
        }
    }

    /// Mutable access to the owned model for add_glyph.  Panics if mmap.
    pub fn entries_mut(&mut self) -> &mut HashMap<Vec<char>, ImageModel> {
        match self {
            CharModelStore::Owned(m) => &mut m.entries,
            CharModelStore::Mmap(_) => panic!("cannot mutate mmap model"),
        }
    }

    pub fn catalog_hash(&self) -> u64 {
        match self {
            CharModelStore::Owned(m) => m.catalog_hash,
            CharModelStore::Mmap(m) => m.catalog_hash,
        }
    }
}

pub struct EmbeddingClassifier {
    model: CharModelStore,
    embedder: Box<dyn Embedder>,
}

impl Classifier for EmbeddingClassifier {
    fn classify(&self, seq: &[char], query: &CropFeatures, k: usize) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(seq, query);
        self.model.classify(seq, &q, k).into_iter().map(|(id, p)| (id as usize, p)).collect()
    }

    fn probabilities(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(seq, query);
        self.model.probabilities(seq, &q).into_iter().map(|(id, p)| (id as usize, p)).collect()
    }

    fn probability(&self, seq: &[char], query: &CropFeatures, glyph_id: usize) -> Option<f32> {
        let q = self.embedder.embed(seq, query);
        let probs = self.model.probabilities(seq, &q);
        probs.into_iter()
            .find(|(id, _)| *id as usize == glyph_id)
            .map(|(_, p)| p)
    }

    fn raw_logits(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(seq, query);
        self.model.raw_logits(seq, &q).into_iter().map(|(id, l)| (id as usize, l)).collect()
    }

    fn raw_logit(&self, seq: &[char], query: &CropFeatures, glyph_id: usize) -> Option<f32> {
        let q = self.embedder.embed(seq, query);
        self.model.raw_logits(seq, &q).into_iter()
            .find(|(id, _)| *id as usize == glyph_id)
            .map(|(_, l)| l)
    }

    fn name(&self) -> &str {
        self.embedder.name()
    }

    fn glyph_count(&self, seq: &[char]) -> usize {
        self.model.glyph_count(seq)
    }

    fn add_glyph(&mut self, glyph_id: usize, seq: &[char], features: &CropFeatures) {
        let embedded = self.embedder.embed(seq, features);
        let cm = self.model.entries_mut().entry(seq.to_vec()).or_insert_with(|| ImageModel {
            weights: Vec::new(),
            centroids: Vec::new(),
            sigma_sq: 0.0,
            med_nn: 0.0,
        });
        cm.centroids.push((glyph_id as u32, embedded));
    }

    fn ensure_owned(&mut self) {
        if let CharModelStore::Mmap(m) = &self.model {
            let owned = m.to_owned_model();
            self.model = CharModelStore::Owned(owned);
        }
    }

    fn recompute_stats(&mut self) {
        match &mut self.model {
            CharModelStore::Owned(m) => {
                for cm in m.entries.values_mut() {
                    cm.compute_sigma_sq();
                    cm.compute_med_nn();
                }
            }
            CharModelStore::Mmap(_) => {} // should have been ensure_owned first
        }
    }

    fn set_catalog_hash(&mut self, hash: u64) {
        match &mut self.model {
            CharModelStore::Owned(m) => m.catalog_hash = hash,
            CharModelStore::Mmap(m) => {
                // Convert to owned to allow mutation, then set
                let mut owned = m.to_owned_model();
                owned.catalog_hash = hash;
                self.model = CharModelStore::Owned(owned);
            }
        }
    }

    fn save_to(&self, path: &std::path::Path, magic: &[u8; 4], version: u32) -> Result<(), String> {
        let owned = match &self.model {
            CharModelStore::Owned(m) => m,
            CharModelStore::Mmap(m) => {
                // Need owned view to write; clone to owned temporarily
                // to_owned_model is cheap-ish compared to full retrain
                // We can't move out of &self, so we do a full owned copy
                // and write that.
                let tmp_owned = m.to_owned_model();
                // Write tmp_owned directly
                let tmp_path = crate::atomic_file::tmp_for(path);
                let f = std::fs::File::create(&tmp_path).map_err(|e| format!("create tmp: {e}"))?;
                let mut w = std::io::BufWriter::new(f);
                tmp_owned.write_bin(&mut w, magic, version).map_err(|e| format!("write: {e}"))?;
                use std::io::Write;
                w.flush().map_err(|e| format!("flush: {e}"))?;
                drop(w);
                std::fs::rename(&tmp_path, path).map_err(|e| format!("rename: {e}"))?;
                return Ok(());
            }
        };
        let tmp_path = crate::atomic_file::tmp_for(path);
        let f = std::fs::File::create(&tmp_path).map_err(|e| format!("create tmp: {e}"))?;
        let mut w = std::io::BufWriter::new(f);
        owned.write_bin(&mut w, magic, version).map_err(|e| format!("write: {e}"))?;
        use std::io::Write;
        w.flush().map_err(|e| format!("flush: {e}"))?;
        drop(w);
        std::fs::rename(&tmp_path, path).map_err(|e| format!("rename: {e}"))?;
        Ok(())
    }

    fn catalog_hash(&self) -> Option<u64> {
        Some(self.model.catalog_hash())
    }
}

// ---------------------------------------------------------------------------
// Triplet network (per-glyph)
// ---------------------------------------------------------------------------

pub(crate) const L1_IN: usize = FEAT_LEN;
pub(crate) const L1_OUT: usize = 128;
pub(crate) const L2_OUT: usize = 64;
pub(crate) const L3_OUT: usize = 32;

/// Per-character parameter count.
const PARAMS_PER_CHAR: usize =
    L1_IN * L1_OUT + L1_OUT + // W1, b1
    L1_OUT * L2_OUT + L2_OUT + // W2, b2
    L2_OUT * L3_OUT + L3_OUT;  // W3, b3

/// Weights for a single character's MLP.
struct GlyphNet {
    fc1: InferenceLinear, // L1_IN → L1_OUT
    fc2: InferenceLinear, // L1_OUT → L2_OUT
    fc3: InferenceLinear, // L2_OUT → L3_OUT
}

impl GlyphNet {
    /// Forward pass: ReLU(fc1) → ReLU(fc2) → fc3 → L2-normalize
    fn forward(&self, raw: &[f32]) -> Vec<f32> {
        // Layer 1: ReLU
        let mut h1 = self.fc1.forward(raw);
        for v in &mut h1 { *v = v.max(0.0); }

        // Layer 2: ReLU
        let mut h2 = self.fc2.forward(&h1);
        for v in &mut h2 { *v = v.max(0.0); }

        // Layer 3: linear (no activation)
        let mut out = self.fc3.forward(&h2);

        // L2-normalise to unit sphere
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut out {
                *v /= norm;
            }
        }
        out
    }
}

/// Per-glyph triplet network classifier.
///
/// Loads one 3-layer MLP per indexed character from a binary weights file.
/// Characters not in the weights file fall back to Fisher-weighted embedding
/// (truncated to 32 dims).
///
/// Stores projected font vectors internally and performs brute-force
/// nearest-neighbor search in the embedding space.
pub struct TripletClassifier {
    nets: HashMap<Vec<char>, GlyphNet>,
}

impl TripletClassifier {
    /// Load a triplet classifier from a TRIP v3 binary, or train one if
    /// missing/stale.  Returns a ready-to-use `EmbeddingClassifier`.
    pub fn load(
        path: &std::path::Path,
        ctx: Option<&crate::train::TrainingContext>,
    ) -> Result<EmbeddingClassifier, String> {
        let need_train = if !path.exists() {
            true
        } else {
            let data = std::fs::read(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if data.len() < 8 { true }
            else {
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap()) & 0x7FFF_FFFF;
                if version != 3 {
                    eprintln!("Triplet weights {} are v{version}, need v3 — retraining...", path.display());
                    true
                } else if let Some(c) = ctx {
                    let file_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());
                    if file_hash != c.catalog_hash {
                        eprintln!("Triplet weights {} stale (catalog changed), retraining...", path.display());
                        true
                    } else { false }
                } else { false }
            }
        };
        if need_train {
            let ctx = ctx.ok_or_else(|| format!(
                "Triplet weights {} not found and no training context provided", path.display()))?;
            Self::train(ctx, path);
        }

        let mmap_model = MmapNgramModel::load_indexed(path, b"TRIP")?;

        let mut nets = HashMap::new();
        for seq in mmap_model.entry_keys().cloned().collect::<Vec<_>>() {
            let w = mmap_model.weights(&seq)
                .ok_or_else(|| format!("Triplet seq '{:?}': no weights", seq))?;
            if w.len() != PARAMS_PER_CHAR {
                return Err(format!(
                    "Triplet seq '{:?}': expected {} params, got {}",
                    seq, PARAMS_PER_CHAR, w.len()
                ));
            }
            let mut pos = 0usize;
            let fc1_w = w[pos..pos + L1_IN * L1_OUT].to_vec(); pos += L1_IN * L1_OUT;
            let fc1_b = w[pos..pos + L1_OUT].to_vec(); pos += L1_OUT;
            let fc2_w = w[pos..pos + L1_OUT * L2_OUT].to_vec(); pos += L1_OUT * L2_OUT;
            let fc2_b = w[pos..pos + L2_OUT].to_vec(); pos += L2_OUT;
            let fc3_w = w[pos..pos + L2_OUT * L3_OUT].to_vec(); pos += L2_OUT * L3_OUT;
            let fc3_b = w[pos..pos + L3_OUT].to_vec();

            let net = GlyphNet {
                fc1: InferenceLinear { rows: L1_IN, cols: L1_OUT, w: fc1_w, b: fc1_b },
                fc2: InferenceLinear { rows: L1_OUT, cols: L2_OUT, w: fc2_w, b: fc2_b },
                fc3: InferenceLinear { rows: L2_OUT, cols: L3_OUT, w: fc3_w, b: fc3_b },
            };
            nets.insert(seq.clone(), net);
        }

        let embedder = Self { nets };
        Ok(EmbeddingClassifier { model: CharModelStore::Mmap(mmap_model), embedder: Box::new(embedder) })
    }

    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        if let Some(net) = self.nets.get(seq) {
            net.forward(&features.as_slice())
        } else {
            let raw = features.as_slice();
            raw[..L3_OUT.min(raw.len())].to_vec()
        }
    }

    /// Train per-character triplet networks and write a TRIP v2 binary
    /// (weights + font index).
    pub fn train(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
    ) {
        Self::train_with_params(ctx, output, 50, 0.001, 0.2, 128);
    }

    pub fn train_with_params(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
        epochs: usize,
        lr: f32,
        margin: f32,
        batch_size: usize,
    ) {
        use std::io::{BufWriter, Write};
        use rand::prelude::*;
        use rand::rngs::SmallRng;
        use crate::train::{TrainableNet, dist_sq};

        let sequences = ctx.sequences;
        eprintln!("\nTriplet training {} characters (epochs={}, lr={}, margin={})...",
            sequences.len(), epochs, lr, margin);

        let train_start = std::time::Instant::now();
        let mut trained_seqs: Vec<(Vec<char>, TrainableNet)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_rr_sum = 0.0f64;
        let mut total_top1 = 0usize;
        let mut total_top5 = 0usize;
        let mut total_eval = 0usize;

        // Collect per-char samples for centroid computation after training
        let mut per_seq_samples: Vec<(Vec<char>, Vec<crate::train::TrainingSample>)> = Vec::new();

        for (si, seq) in sequences.iter().enumerate() {
            if ctx.seq_counts[si] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(si);

            let mut font_set: Vec<u32> = samples.iter().map(|s| s.glyph_id).collect();
            font_set.sort_unstable();
            font_set.dedup();
            if font_set.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let mut rng = SmallRng::seed_from_u64(si as u64);
            let mut net = TrainableNet::new(&mut rng);

            let mut font_samples: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_samples.entry(s.glyph_id).or_default().push(i);
            }
            let font_ids: Vec<u32> = font_samples.keys().copied().collect();
            let mut adam_t = 0usize;

            for epoch in 0..epochs {
                let mut epoch_loss = 0.0f32;
                let mut n_triplets = 0usize;

                for _ in 0..batch_size {
                    let anchor_font = font_ids[rng.gen_range(0..font_ids.len())];
                    let anchor_samples = &font_samples[&anchor_font];
                    if anchor_samples.len() < 2 { continue; }

                    let ai = anchor_samples[rng.gen_range(0..anchor_samples.len())];
                    let pi = anchor_samples[rng.gen_range(0..anchor_samples.len())];
                    if ai == pi { continue; }

                    let neg_font = loop {
                        let f = font_ids[rng.gen_range(0..font_ids.len())];
                        if f != anchor_font { break f; }
                    };
                    let neg_samples = &font_samples[&neg_font];
                    let ni = neg_samples[rng.gen_range(0..neg_samples.len())];

                    let a_cache = net.forward(&samples[ai].features);
                    let p_cache = net.forward(&samples[pi].features);
                    let n_cache = net.forward(&samples[ni].features);

                    let dp = dist_sq(&a_cache.out, &p_cache.out);
                    let dn = dist_sq(&a_cache.out, &n_cache.out);
                    let loss = (dp - dn + margin).max(0.0);

                    if loss > 0.0 {
                        epoch_loss += loss;
                        n_triplets += 1;

                        let out_dim = a_cache.out.len();
                        let mut d_a = vec![0.0f32; out_dim];
                        let mut d_p = vec![0.0f32; out_dim];
                        let mut d_n = vec![0.0f32; out_dim];
                        for j in 0..out_dim {
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

                adam_t += 1;
                let effective_batch = n_triplets.max(1);
                net.adam_step(lr, adam_t, effective_batch);

                if (epoch + 1) % 10 == 0 || epoch == 0 {
                    let avg_loss = if n_triplets > 0 { epoch_loss / n_triplets as f32 } else { 0.0 };
                    if si < 5 || si == sequences.len() - 1 {
                        eprintln!("  seq {:?} epoch {}/{}: loss={:.4} ({} active triplets)",
                            seq, epoch + 1, epochs, avg_loss, n_triplets);
                    }
                }
            }

            trained_seqs.push((seq.clone(), net));

            // Retrieval quality: MRR via nearest-centroid
            if let Some((_, ref trained_net)) = trained_seqs.last() {
                let embeddings: Vec<Vec<f32>> = samples.iter()
                    .map(|s| trained_net.forward(&s.features).out)
                    .collect();

                let mut centroid_sums: HashMap<u32, Vec<f32>> = HashMap::new();
                let mut centroid_counts: HashMap<u32, usize> = HashMap::new();
                for (i, s) in samples.iter().enumerate() {
                    let entry = centroid_sums.entry(s.glyph_id).or_insert_with(|| vec![0.0; L3_OUT]);
                    for (j, &v) in embeddings[i].iter().enumerate() { entry[j] += v; }
                    *centroid_counts.entry(s.glyph_id).or_insert(0) += 1;
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
                let max_eval = 2000usize;
                let eval_indices: Vec<usize> = if n <= max_eval {
                    (0..n).collect()
                } else {
                    let mut rng2 = SmallRng::seed_from_u64(si as u64 + 0x1234);
                    let mut idx: Vec<usize> = (0..n).collect();
                    idx.shuffle(&mut rng2);
                    idx.truncate(max_eval);
                    idx
                };
                let n_eval = eval_indices.len();

                let mut char_rr_sum = 0.0f64;
                let mut char_top1 = 0usize;
                let mut char_top5 = 0usize;
                for &i in &eval_indices {
                    let correct_font = samples[i].glyph_id;
                    let pos = centroid_fids.iter().position(|&f| f == correct_font).unwrap();
                    let d_correct = dist_sq(&embeddings[i], &centroid_vecs[pos]);

                    let mut rank = 0usize;
                    for ci2 in 0..k {
                        if centroid_fids[ci2] == correct_font { continue; }
                        if dist_sq(&embeddings[i], &centroid_vecs[ci2]) < d_correct { rank += 1; }
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

                if si < 5 || si == sequences.len() - 1 || (si + 1) % 20 == 0 {
                    eprintln!("  seq {:?} MRR={:.3} top1={:.1}% top5={:.1}% (n={})",
                        seq, mrr,
                        char_top1 as f64 / n_eval as f64 * 100.0,
                        char_top5 as f64 / n_eval as f64 * 100.0,
                        n_eval);
                }
            }

            per_seq_samples.push((seq.clone(), samples));
        }

        let train_elapsed = train_start.elapsed();
        let mrr = if total_eval > 0 { total_rr_sum / total_eval as f64 } else { 0.0 };
        let top1 = if total_eval > 0 { total_top1 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        let top5 = if total_eval > 0 { total_top5 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        eprintln!("\nTriplet complete: {} seqs, {} skipped, {:.1}s",
            trained_seqs.len(), skipped, train_elapsed.as_secs_f64());
        eprintln!("  MRR={:.3} top1={:.1}% top5={:.1}% (n={})", mrr, top1, top5, total_eval);

        // Write TRIP v3 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let mut model = NgramModel::new(ctx.catalog_hash);
        for (tc_idx, (seq, net)) in trained_seqs.iter().enumerate() {
            // Flatten net params into weights blob
            let mut weights = Vec::with_capacity(PARAMS_PER_CHAR);
            weights.extend_from_slice(&net.fc1.w);
            weights.extend_from_slice(&net.fc1.b);
            weights.extend_from_slice(&net.fc2.w);
            weights.extend_from_slice(&net.fc2.b);
            weights.extend_from_slice(&net.fc3.w);
            weights.extend_from_slice(&net.fc3.b);

            // Build centroids: embed each sample, average per font
            let samples = &per_seq_samples[tc_idx].1;
            let mut sums: HashMap<u32, Vec<f32>> = HashMap::new();
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for s in samples {
                let emb = net.forward(&s.features).out;
                let entry = sums.entry(s.glyph_id).or_insert_with(|| vec![0.0; emb.len()]);
                for (j, &v) in emb.iter().enumerate() { entry[j] += v; }
                *counts.entry(s.glyph_id).or_insert(0) += 1;
            }
            let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(sums.len());
            for (&fid, sum) in &sums {
                let cnt = counts[&fid] as f32;
                let centroid: Vec<f32> = sum.iter().map(|&v| v / cnt).collect();
                centroids.push((fid, centroid));
            }
            let mut cm = ImageModel { weights, centroids, sigma_sq: 0.0, med_nn: 0.0 };
            cm.compute_sigma_sq();
                cm.compute_med_nn();
            model.entries.insert(seq.clone(), cm);
        }
        let tmp = crate::atomic_file::tmp_for(output);
        let f = std::fs::File::create(&tmp).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"TRIP", 3).expect("write TRIP v3");
        w.flush().unwrap();
        drop(w);
        std::fs::rename(&tmp, output).expect("atomic rename");

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e6, ctx.catalog.len());
    }

}

impl Embedder for TripletClassifier {
    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        TripletClassifier::embed(self, seq, features)
    }
    fn name(&self) -> &str { "triplet" }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Per-character Fisher (loads per-char weights from FISH file)
// ---------------------------------------------------------------------------

/// Per-character Fisher-weighted Euclidean distance.
///
/// Unlike [`FisherClassifier`] which uses a single global weight vector,
/// this variant loads per-character weights from a FISH file, so each
/// character uses its own discriminative feature weighting.
///
/// Binary format (same as FISH):
/// ```text
/// magic:    b"FISH" (4 bytes)
/// version:  u32 LE (1)
/// n_chars:  u32 LE
/// feat_len: u32 LE (FEAT_LEN)
/// Per character (repeated n_chars times):
///   char_code: u32 LE
///   weights:   [f32; FEAT_LEN] LE
/// ```
pub struct PerCharFisherClassifier {
    weights: HashMap<Vec<char>, [f32; FEAT_LEN]>,
}

impl PerCharFisherClassifier {
    /// Load a per-char Fisher classifier from a FISH v3 binary, or train one
    /// if the file doesn't exist or is stale.  Returns a ready-to-use
    /// `EmbeddingClassifier` with the font index populated — no separate
    /// `load_fonts` step needed.
    pub fn load(
        path: &std::path::Path,
        ctx: Option<&crate::train::TrainingContext>,
    ) -> Result<EmbeddingClassifier, String> {
        // Train if missing; retrain if stale (wrong version/hash).
        let need_train = if !path.exists() {
            true
        } else {
            // Peek at the file: reject v2 or catalog-hash mismatch.
            let data = std::fs::read(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if data.len() < 8 { true }
            else {
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap()) & 0x7FFF_FFFF;
                if version != 3 {
                    eprintln!("Fisher weights {} are v{version}, need v3 — retraining...", path.display());
                    true
                } else if let Some(c) = ctx {
                    let file_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());
                    if file_hash != c.catalog_hash {
                        eprintln!("Fisher weights {} stale (catalog changed), retraining...", path.display());
                        true
                    } else { false }
                } else { false }
            }
        };
        if need_train {
            let ctx = ctx.ok_or_else(|| format!(
                "Fisher weights {} not found and no training context provided", path.display()))?;
            Self::train(ctx, path);
        }

        let mmap_model = MmapNgramModel::load_indexed(path, b"FISH")?;

        // Extract normalized embedder weights from the model
        let mut weights = HashMap::new();
        for seq in mmap_model.entry_keys().cloned().collect::<Vec<_>>() {
            let wt = mmap_model.weights(&seq)
                .ok_or_else(|| format!("Fisher seq '{:?}': no weights", seq))?;
            let mut w = [0.0f32; FEAT_LEN];
            let n = wt.len().min(FEAT_LEN);
            w[..n].copy_from_slice(&wt[..n]);

            let max_finite = w.iter().filter(|v| v.is_finite()).copied()
                .fold(0.0f32, f32::max);
            for v in &mut w {
                if !v.is_finite() || *v > max_finite { *v = max_finite; }
                *v = v.sqrt();
            }
            let sum: f32 = w.iter().sum();
            if sum > 1e-12 { for v in &mut w { *v /= sum; } }

            weights.insert(seq.clone(), w);
        }

        let embedder = Self { weights };
        Ok(EmbeddingClassifier { model: CharModelStore::Mmap(mmap_model), embedder: Box::new(embedder) })
    }

    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some(w) = self.weights.get(seq) {
            let mut v = vec![0.0f32; FEAT_LEN];
            for i in 0..FEAT_LEN { v[i] = raw[i] * w[i]; }
            v
        } else {
            raw.to_vec()
        }
    }

    /// Normalize raw Fisher scores the same way load() does.
    fn normalize_scores(scores: &[f32; FEAT_LEN]) -> [f32; FEAT_LEN] {
        let mut nw = *scores;
        let max_finite = nw.iter().filter(|v| v.is_finite()).copied()
            .fold(0.0f32, f32::max);
        for v in &mut nw {
            if !v.is_finite() || *v > max_finite { *v = max_finite; }
            *v = v.sqrt();
        }
        let sum: f32 = nw.iter().sum();
        if sum > 1e-12 { for v in &mut nw { *v /= sum; } }
        nw
    }

    /// Train Fisher weights and write a FISH v2 binary (weights + font index).
    pub fn train(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
    ) {
        use std::io::{BufWriter, Write};

        let sequences = ctx.sequences;
        eprintln!("\nFisher scoring {} characters...", sequences.len());
        let fisher_start = std::time::Instant::now();
        let mut fisher_seqs: Vec<(Vec<char>, [f32; FEAT_LEN], HashMap<u32, Vec<f64>>)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (si, seq) in sequences.iter().enumerate() {
            if ctx.seq_counts[si] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(si);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.glyph_id).or_default().push(i);
            }
            if font_indices.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let n = samples.len();

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

            // Between-class variance
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

            // Within-class variance
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

            // Fisher score per feature
            let mut scores = [0.0f32; FEAT_LEN];
            for j in 0..FEAT_LEN {
                scores[j] = if var_within[j] > 1e-12 {
                    (var_between[j] / var_within[j]) as f32
                } else if var_between[j] > 1e-12 {
                    f32::MAX
                } else {
                    0.0
                };
            }

            // Evaluate
            let eval_indices = crate::train::subsample_eval(n, 2000, si as u64);
            let centroid_fids: Vec<u32> = class_means.keys().copied().collect();
            let centroid_feats: Vec<&Vec<f64>> = centroid_fids.iter()
                .map(|fid| &class_means[fid])
                .collect();

            let char_stats = crate::train::eval_mrr(
                &samples, &eval_indices, &class_means, &centroid_fids,
                &ctx.glyph_family_for_seq(seq),
                &|i| {
                    centroid_fids.iter().enumerate().map(|(ci2, &fid)| {
                        let mut d = 0.0f64;
                        for j in 0..FEAT_LEN {
                            let diff = samples[i].features[j] as f64 - centroid_feats[ci2][j];
                            d += scores[j] as f64 * diff * diff;
                        }
                        (fid, d)
                    }).collect()
                },
            );

            if si < 5 || si == sequences.len() - 1 || (si + 1) % 20 == 0 {
                eprintln!("  seq {:?} base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    seq, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }

            total_stats.accumulate(&char_stats);
            fisher_seqs.push((seq.clone(), scores, class_means));
        }

        let fisher_elapsed = fisher_start.elapsed();
        eprintln!("\nFisher scoring complete: {} seqs, {} skipped, {:.1}s",
            fisher_seqs.len(), skipped, fisher_elapsed.as_secs_f64());
        eprintln!("  Baseline:      MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  Fisher strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  Fisher family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write FISH v3 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut model = NgramModel::new(ctx.catalog_hash);
        for (seq, scores, class_means) in &fisher_seqs {
            let nw = Self::normalize_scores(scores);
            let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(class_means.len());
            for (&fid, mean) in class_means {
                let embedded: Vec<f32> = (0..FEAT_LEN)
                    .map(|j| mean[j] as f32 * nw[j])
                    .collect();
                centroids.push((fid, embedded));
            }
            let mut cm = ImageModel {
                weights: scores.to_vec(),
                centroids,
                sigma_sq: 0.0,
                med_nn: 0.0,
            };
            cm.compute_sigma_sq();
                cm.compute_med_nn();
            model.entries.insert(seq.clone(), cm);
        }
        let tmp = crate::atomic_file::tmp_for(output);
        let f = std::fs::File::create(&tmp).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"FISH", 3).expect("write FISH v3");
        w.flush().unwrap();
        drop(w);
        std::fs::rename(&tmp, output).expect("atomic rename");

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} KB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e3, ctx.catalog.len());
    }
}

impl Embedder for PerCharFisherClassifier {
    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        PerCharFisherClassifier::embed(self, seq, features)
    }
    fn name(&self) -> &str { "per_char_fisher" }
}

// Mahalanobis (per-character whitening via Cholesky)
// ---------------------------------------------------------------------------
// Mahalanobis (per-character whitening via Cholesky)
// ---------------------------------------------------------------------------

/// Per-character Mahalanobis distance classifier.
///
/// For each character, computes the within-class scatter matrix Sw,
/// then whitens features using the inverse Cholesky factor: L^{-1}
/// where Sw = L L^T.  In the whitened space, Euclidean distance equals
/// Mahalanobis distance, which accounts for feature correlations.
///
/// Binary format (magic `b"MAHA"`, version 2):
/// ```text
/// magic:    b"MAHA" (4 bytes)
/// version:  u32 LE (2)
/// n_chars:  u32 LE
/// feat_len: u32 LE
/// Per character (repeated n_chars times):
///   char_code: u32 LE
///   L_inv:     [f32; feat_len * feat_len] LE (row-major)
/// ```
pub struct MahalanobisClassifier {
    transforms: HashMap<Vec<char>, Vec<f32>>, // L_inv per char, FEAT_LEN × FEAT_LEN row-major
}

impl MahalanobisClassifier {
    /// Load a Mahalanobis classifier from a MAHA v3 binary, or train one if
    /// missing/stale.  Returns a ready-to-use `EmbeddingClassifier`.
    pub fn load(
        path: &std::path::Path,
        ctx: Option<&crate::train::TrainingContext>,
    ) -> Result<EmbeddingClassifier, String> {
        let need_train = if !path.exists() {
            true
        } else {
            let data = std::fs::read(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if data.len() < 8 { true }
            else {
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap()) & 0x7FFF_FFFF;
                if version != 3 {
                    eprintln!("Mahalanobis weights {} are v{version}, need v3 — retraining...", path.display());
                    true
                } else if let Some(c) = ctx {
                    let file_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());
                    if file_hash != c.catalog_hash {
                        eprintln!("Mahalanobis weights {} stale (catalog changed), retraining...", path.display());
                        true
                    } else { false }
                } else { false }
            }
        };
        if need_train {
            let ctx = ctx.ok_or_else(|| format!(
                "Mahalanobis weights {} not found and no training context provided", path.display()))?;
            Self::train(ctx, path);
        }

        let mmap_model = MmapNgramModel::load_indexed(path, b"MAHA")?;

        let mut transforms = HashMap::new();
        for seq in mmap_model.entry_keys().cloned().collect::<Vec<_>>() {
            let w = mmap_model.weights(&seq)
                .ok_or_else(|| format!("Mahalanobis seq '{:?}': no weights", seq))?;
            transforms.insert(seq.clone(), w.to_vec());
        }

        let embedder = Self { transforms };
        Ok(EmbeddingClassifier { model: CharModelStore::Mmap(mmap_model), embedder: Box::new(embedder) })
    }

    /// Apply L_inv transform: y = L_inv * x  (square FEAT_LEN×FEAT_LEN matrix multiply)
    fn apply_transform(linv: &[f32], x: &[f32]) -> Vec<f32> {
        // Special case of dense_project with out_dim = FEAT_LEN
        dense_project(FEAT_LEN, linv, x)
    }

    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some(linv) = self.transforms.get(seq) {
            Self::apply_transform(linv, &raw)
        } else {
            raw.to_vec()
        }
    }

    /// Train Mahalanobis weights and write a MAHA v2 binary (weights + font index).
    pub fn train(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
    ) {
        use std::io::{BufWriter, Write};
        use rand::prelude::*;
        use rand::rngs::SmallRng;

        let sequences = ctx.sequences;
        eprintln!("\nMahalanobis training {} characters...", sequences.len());
        let maha_start = std::time::Instant::now();
        let mut maha_seqs: Vec<(Vec<char>, Vec<f32>, HashMap<u32, Vec<f64>>)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (si, seq) in sequences.iter().enumerate() {
            if ctx.seq_counts[si] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(si);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.glyph_id).or_default().push(i);
            }
            if font_indices.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let n = samples.len();

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

            // ── New per-char forward cache: means/{code:08x}.bin ──────────
            // Populate means cache for lazy Unicode expansion. Format:
            //   b"MEAN" | version | char_code | feat_dim | n_entries
            //   per entry: font_key_len | font_key | file_hash | count | mean[feat_dim]
            // Atomic write via tmp+rename. file_hash = mtime+size FNV (same as geo_cache).
            if seq.len() == 1 {
                let code = seq[0] as u32;
                let mut mean_entries: Vec<crate::per_char_cache::MeanEntry> = Vec::new();
                mean_entries.reserve(class_means.len() * 2);
                for (&glyph_id, mean_f64) in &class_means {
                    let count = font_indices.get(&glyph_id).map(|v| v.len() as u32).unwrap_or(1);
                    // glyph_id is group index into glyph_map for this seq
                    let font_keys = ctx.glyph_map.fonts_for_glyph(seq, glyph_id as usize);
                    // mean as f32 — share Arc to avoid per-font Vec clone bloat (was Vec clone 200x per group).
                    let mean_arc: std::sync::Arc<[f32]> = mean_f64.iter().map(|&v| v as f32).collect::<Vec<_>>().into();
                    for fk in font_keys {
                        if let Some(&font_idx) = ctx.font_id_map.get(fk) {
                            if let Some(fe) = ctx.catalog.get(font_idx as usize) {
                                let fhash = crate::per_char_cache::file_meta_hash(&fe.path);
                                mean_entries.push(crate::per_char_cache::MeanEntry{
                                    font_key: fk.clone(),
                                    file_hash: fhash,
                                    count,
                                    mean: mean_arc.clone(),
                                });
                            }
                        }
                    }
                }
                if !mean_entries.is_empty() {
                    if let Err(e) = crate::per_char_cache::write_means_atomic(code, FEAT_LEN, &mean_entries) {
                        eprintln!("Warning: failed to write means cache for U+{:04X}: {e}", code);
                    }
                }
            }

            // Within-class scatter Sw
            let mut sw = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            for (&fid, indices) in &font_indices {
                let cm = &class_means[&fid];
                for &i in indices {
                    for a in 0..FEAT_LEN {
                        let da = samples[i].features[a] as f64 - cm[a];
                        for b in a..FEAT_LEN {
                            let db = samples[i].features[b] as f64 - cm[b];
                            sw[a * FEAT_LEN + b] += da * db;
                        }
                    }
                }
            }
            for a in 0..FEAT_LEN { for b in 0..a { sw[a * FEAT_LEN + b] = sw[b * FEAT_LEN + a]; } }
            for v in &mut sw { *v /= n as f64; }

            // Regularize: Ledoit-Wolf style shrinkage toward identity
            let trace: f64 = (0..FEAT_LEN).map(|j| sw[j * FEAT_LEN + j]).sum();
            let avg_var = trace / FEAT_LEN as f64;
            let alpha = 0.9;
            for a in 0..FEAT_LEN {
                for b in 0..FEAT_LEN {
                    sw[a * FEAT_LEN + b] *= 1.0 - alpha;
                }
                sw[a * FEAT_LEN + a] += alpha * avg_var;
            }

            // Cholesky: Sw = L L^T
            let mut l = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            let mut chol_ok = true;
            for i in 0..FEAT_LEN {
                for j in 0..=i {
                    let mut sum = sw[i * FEAT_LEN + j];
                    for k in 0..j { sum -= l[i * FEAT_LEN + k] * l[j * FEAT_LEN + k]; }
                    if i == j {
                        if sum <= 0.0 { chol_ok = false; break; }
                        l[i * FEAT_LEN + j] = sum.sqrt();
                    } else {
                        l[i * FEAT_LEN + j] = sum / l[j * FEAT_LEN + j];
                    }
                }
                if !chol_ok { break; }
            }
            if !chol_ok {
                eprintln!("  seq {:?} Cholesky failed, skipping", seq);
                skipped += 1;
                continue;
            }

            // L^{-1} by forward substitution
            let mut linv = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            for j in 0..FEAT_LEN {
                for i in 0..FEAT_LEN {
                    let mut sum = if i == j { 1.0 } else { 0.0 };
                    for k in 0..i { sum -= l[i * FEAT_LEN + k] * linv[k * FEAT_LEN + j]; }
                    linv[i * FEAT_LEN + j] = sum / l[i * FEAT_LEN + i];
                }
            }

            // Scale calibration
            let target_dist = 0.03f64;
            let mut within_dists: Vec<f64> = Vec::new();
            let mut rng_cal = SmallRng::seed_from_u64(si as u64 + 9999);
            for (&_fid, indices) in &font_indices {
                if indices.len() < 2 { continue; }
                let npairs = 5.min(indices.len() * (indices.len() - 1) / 2);
                for _ in 0..npairs {
                    let a = indices[rng_cal.gen_range(0..indices.len())];
                    let b = loop {
                        let x = indices[rng_cal.gen_range(0..indices.len())];
                        if x != a { break x; }
                    };
                    let mut d = 0.0f64;
                    for dim in 0..FEAT_LEN {
                        let mut ea = 0.0f64;
                        let mut eb = 0.0f64;
                        for j in 0..FEAT_LEN {
                            ea += linv[dim * FEAT_LEN + j] * samples[a].features[j] as f64;
                            eb += linv[dim * FEAT_LEN + j] * samples[b].features[j] as f64;
                        }
                        let diff = ea - eb;
                        d += diff * diff;
                    }
                    within_dists.push(d);
                }
            }
            let scale = if !within_dists.is_empty() {
                within_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = within_dists[within_dists.len() / 2];
                if median > 1e-15 { (target_dist / median).sqrt() } else { 1.0 }
            } else { 1.0 };
            for v in &mut linv { *v *= scale; }

            let linv_f32: Vec<f32> = linv.iter().map(|&v| v as f32).collect();

            // Evaluate MRR
            let eval_indices = crate::train::subsample_eval(n, 2000, si as u64);
            let centroid_fids: Vec<u32> = class_means.keys().copied().collect();
            let centroid_embeds: Vec<Vec<f64>> = centroid_fids.iter().map(|fid| {
                let cm = &class_means[fid];
                let mut emb = vec![0.0f64; FEAT_LEN];
                for i in 0..FEAT_LEN {
                    let mut sum = 0.0f64;
                    for j in 0..FEAT_LEN { sum += linv[i * FEAT_LEN + j] * cm[j]; }
                    emb[i] = sum;
                }
                emb
            }).collect();

            let char_stats = crate::train::eval_mrr(
                &samples, &eval_indices, &class_means, &centroid_fids,
                &ctx.glyph_family_for_seq(seq),
                &|i| {
                    let mut emb = vec![0.0f64; FEAT_LEN];
                    for a in 0..FEAT_LEN {
                        let mut sum = 0.0f64;
                        for b in 0..FEAT_LEN {
                            sum += linv[a * FEAT_LEN + b] * samples[i].features[b] as f64;
                        }
                        emb[a] = sum;
                    }
                    centroid_fids.iter().enumerate().map(|(ci2, &fid)| {
                        let mut d = 0.0f64;
                        for j in 0..FEAT_LEN {
                            let diff = emb[j] - centroid_embeds[ci2][j];
                            d += diff * diff;
                        }
                        (fid, d)
                    }).collect()
                },
            );

            if si < 5 || si == sequences.len() - 1 || (si + 1) % 20 == 0 {
                eprintln!("  seq {:?} base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    seq, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }
            total_stats.accumulate(&char_stats);
            maha_seqs.push((seq.clone(), linv_f32, class_means));
        }

        let maha_elapsed = maha_start.elapsed();
        eprintln!("\nMahalanobis complete: {} seqs, {} skipped, {:.1}s",
            maha_seqs.len(), skipped, maha_elapsed.as_secs_f64());
        eprintln!("  Baseline:    MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  Maha strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  Maha family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write MAHA v3 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let mut model = NgramModel::new(ctx.catalog_hash);
        for (seq, linv, class_means) in &maha_seqs {
            let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(class_means.len());
            for (&fid, mean) in class_means {
                let mut embedded = vec![0.0f32; FEAT_LEN];
                for i in 0..FEAT_LEN {
                    for j in 0..FEAT_LEN {
                        embedded[i] += linv[i * FEAT_LEN + j] * mean[j] as f32;
                    }
                }
                centroids.push((fid, embedded));
            }
            let mut cm = ImageModel {
                weights: linv.to_vec(),
                centroids,
                sigma_sq: 0.0,
                med_nn: 0.0,
            };
            cm.compute_sigma_sq();
                cm.compute_med_nn();
            model.entries.insert(seq.clone(), cm);
        }
        let tmp = crate::atomic_file::tmp_for(output);
        let f = std::fs::File::create(&tmp).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"MAHA", 3).expect("write MAHA v3");
        w.flush().unwrap();
        drop(w);
        std::fs::rename(&tmp, output).expect("atomic rename");

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e6, ctx.catalog.len());
    }

}

impl Embedder for MahalanobisClassifier {
    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        MahalanobisClassifier::embed(self, seq, features)
    }
    fn name(&self) -> &str { "mahalanobis" }
}

// ---------------------------------------------------------------------------
// LDA (per-character linear discriminant analysis)
// ---------------------------------------------------------------------------

/// Per-character LDA classifier.
///
/// Projects features into a lower-dimensional discriminant subspace that
/// maximizes class separability.  Uses the top-k eigenvectors of
/// Sw^{-1} * Sb.
///
/// Binary format (magic `b"LDAC"`, version 3):
/// ```text
/// magic:    b"LDAC" (4 bytes)
/// version:  u32 LE (3)
/// n_chars:  u32 LE
/// Per character:
///   char_code: u32 LE
///   out_dim:   u32 LE (number of projection dimensions)
///   proj:      [f32; out_dim * FEAT_LEN] LE (row-major, each row = one projection direction)
///   sigma_sq:  f32 LE
/// ```
pub struct LdaClassifier {
    projections: HashMap<Vec<char>, (usize, Vec<f32>)>, // (out_dim, proj matrix)
}

impl LdaClassifier {
    /// Load an LDA classifier from an LDAC v6 binary, or train one if the file
    /// doesn't exist or is stale.  Returns a ready-to-use `EmbeddingClassifier`.
    pub fn load(
        path: &std::path::Path,
        ctx: Option<&crate::train::TrainingContext>,
    ) -> Result<EmbeddingClassifier, String> {
        let need_train = if !path.exists() {
            true
        } else {
            use std::io::Read;
            let mut hdr = [0u8; 16];
            let ok = std::fs::File::open(path)
                .and_then(|mut f| f.read_exact(&mut hdr).map(|_| ()));
            if ok.is_err() { true }
            else {
                let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
                let base_ver = version & 0x7FFF_FFFF;
                if base_ver != 8 {
                    eprintln!("LDA weights {} are v{base_ver}, need v8 — retraining...", path.display());
                    true
                } else if let Some(c) = ctx {
                    let file_hash = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
                    if file_hash != c.catalog_hash {
                        eprintln!("LDA weights {} stale (catalog changed), retraining...", path.display());
                        true
                    } else { false }
                } else { false }
            }
        };
        if need_train {
            let ctx = ctx.ok_or_else(|| format!(
                "LDA weights {} not found and no training context provided", path.display()))?;
            Self::train(ctx, path);
        }

        let mmap_model = MmapNgramModel::load_indexed(path, b"LDAC")?;

        // Extract LDA projections from the mmap model
        let mut projections = HashMap::new();
        // Collect keys first to avoid borrow issues
        let keys: Vec<Vec<char>> = mmap_model.entry_keys().cloned().collect();

        for seq in &keys {
            let w = mmap_model.weights(seq)
                .ok_or_else(|| format!("LDA seq '{:?}': no weights", seq))?;
            if w.is_empty() {
                return Err(format!("LDA seq '{:?}': empty weights", seq));
            }
            let out_dim = w[0] as usize;
            let proj = w[1..].to_vec();
            if proj.len() != out_dim * FEAT_LEN {
                return Err(format!(
                    "LDA seq '{:?}': proj len {} != {} × {} = {}",
                    seq, proj.len(), out_dim, FEAT_LEN, out_dim * FEAT_LEN
                ));
            }
            projections.insert(seq.clone(), (out_dim, proj));
        }

        let embedder = Self { projections };
        Ok(EmbeddingClassifier { model: CharModelStore::Mmap(mmap_model), embedder: Box::new(embedder) })
    }

    fn project(out_dim: usize, proj: &[f32], x: &[f32]) -> Vec<f32> {
        dense_project(out_dim, proj, x)
    }

    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some((out_dim, proj)) = self.projections.get(seq) {
            Self::project(*out_dim, proj, &raw)
        } else {
            raw.to_vec()
        }
    }

    /// Train LDA weights and write an LDAC v6 binary (weights + font index).
    pub fn train(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
    ) {
        // Use default LDA params
        Self::train_with_params(ctx, output, 97, 0.01);
    }

    pub fn train_with_params(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
        lda_dims: usize,
        lda_reg: f64,
    ) {
        use std::io::{BufWriter, Write};
        use rand::prelude::*;
        use rand::rngs::SmallRng;

        let sequences = ctx.sequences;
        let out_dim = lda_dims.min(FEAT_LEN - 1);
        eprintln!("\nLDA training {} characters (target dim={})...", sequences.len(), out_dim);
        let lda_start = std::time::Instant::now();
        let mut lda_chars: Vec<(Vec<char>, usize, Vec<f32>, f32, HashMap<u32, Vec<f64>>, f32)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (si, seq) in sequences.iter().enumerate() {
            if ctx.seq_counts[si] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(si);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.glyph_id).or_default().push(i);
            }
            if font_indices.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let n = samples.len();

            // Class means and global mean
            let mut global_mean = vec![0.0f64; FEAT_LEN];
            for s in &samples {
                for j in 0..FEAT_LEN { global_mean[j] += s.features[j] as f64; }
            }
            for j in 0..FEAT_LEN { global_mean[j] /= n as f64; }

            let class_means: HashMap<u32, Vec<f64>> = font_indices.iter().map(|(&fid, indices)| {
                let mut mean = vec![0.0f64; FEAT_LEN];
                for &i in indices {
                    for j in 0..FEAT_LEN { mean[j] += samples[i].features[j] as f64; }
                }
                let cnt = indices.len() as f64;
                for j in 0..FEAT_LEN { mean[j] /= cnt; }
                (fid, mean)
            }).collect();

            // ── New per-char forward cache: means/{code:08x}.bin ──────────
            if seq.len() == 1 {
                let code = seq[0] as u32;
                let mut mean_entries: Vec<crate::per_char_cache::MeanEntry> = Vec::new();
                mean_entries.reserve(class_means.len() * 2);
                for (&glyph_id, mean_f64) in &class_means {
                    let count = font_indices.get(&glyph_id).map(|v| v.len() as u32).unwrap_or(1);
                    let font_keys = ctx.glyph_map.fonts_for_glyph(seq, glyph_id as usize);
                    let mean_arc: std::sync::Arc<[f32]> = mean_f64.iter().map(|&v| v as f32).collect::<Vec<_>>().into();
                    for fk in font_keys {
                        if let Some(&font_idx) = ctx.font_id_map.get(fk) {
                            if let Some(fe) = ctx.catalog.get(font_idx as usize) {
                                let fhash = crate::per_char_cache::file_meta_hash(&fe.path);
                                mean_entries.push(crate::per_char_cache::MeanEntry{
                                    font_key: fk.clone(),
                                    file_hash: fhash,
                                    count,
                                    mean: mean_arc.clone(),
                                });
                            }
                        }
                    }
                }
                if !mean_entries.is_empty() {
                    if let Err(e) = crate::per_char_cache::write_means_atomic(code, FEAT_LEN, &mean_entries) {
                        eprintln!("Warning: failed to write means cache for U+{:04X}: {e}", code);
                    }
                }
            }

            // Within-class scatter Sw
            let mut sw = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            for (&fid, indices) in &font_indices {
                let cm = &class_means[&fid];
                for &i in indices {
                    for a in 0..FEAT_LEN {
                        let da = samples[i].features[a] as f64 - cm[a];
                        for b in a..FEAT_LEN {
                            let db = samples[i].features[b] as f64 - cm[b];
                            sw[a * FEAT_LEN + b] += da * db;
                        }
                    }
                }
            }
            for a in 0..FEAT_LEN { for b in 0..a { sw[a * FEAT_LEN + b] = sw[b * FEAT_LEN + a]; } }
            for v in &mut sw { *v /= n as f64; }

            // Regularize Sw
            let trace: f64 = (0..FEAT_LEN).map(|j| sw[j * FEAT_LEN + j]).sum();
            let eps = (trace / FEAT_LEN as f64) * lda_reg + 1e-6;
            for j in 0..FEAT_LEN { sw[j * FEAT_LEN + j] += eps; }

            // Cholesky: Sw = L L^T
            let mut l = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            let mut chol_ok = true;
            for i in 0..FEAT_LEN {
                for j in 0..=i {
                    let mut sum = sw[i * FEAT_LEN + j];
                    for k in 0..j { sum -= l[i * FEAT_LEN + k] * l[j * FEAT_LEN + k]; }
                    if i == j {
                        if sum <= 0.0 { chol_ok = false; break; }
                        l[i * FEAT_LEN + j] = sum.sqrt();
                    } else {
                        l[i * FEAT_LEN + j] = sum / l[j * FEAT_LEN + j];
                    }
                }
                if !chol_ok { break; }
            }
            if !chol_ok { skipped += 1; continue; }

            // L^{-1}
            let mut linv = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            for j in 0..FEAT_LEN {
                for i in 0..FEAT_LEN {
                    let mut sum = if i == j { 1.0 } else { 0.0 };
                    for k in 0..i { sum -= l[i * FEAT_LEN + k] * linv[k * FEAT_LEN + j]; }
                    linv[i * FEAT_LEN + j] = sum / l[i * FEAT_LEN + i];
                }
            }

            // Whiten class means
            let centroid_fids: Vec<u32> = class_means.keys().copied().collect();
            let whitened_means: Vec<Vec<f64>> = centroid_fids.iter().map(|fid| {
                let cm = &class_means[fid];
                let mut centered = vec![0.0f64; FEAT_LEN];
                for j in 0..FEAT_LEN { centered[j] = cm[j] - global_mean[j]; }
                let mut wm = vec![0.0f64; FEAT_LEN];
                for i in 0..FEAT_LEN {
                    for j in 0..FEAT_LEN {
                        wm[i] += linv[i * FEAT_LEN + j] * centered[j];
                    }
                }
                wm
            }).collect();

            // PCA on whitened means
            let k_classes = centroid_fids.len();
            let mut wm_mean = vec![0.0f64; FEAT_LEN];
            for wm in &whitened_means { for j in 0..FEAT_LEN { wm_mean[j] += wm[j]; } }
            for j in 0..FEAT_LEN { wm_mean[j] /= k_classes as f64; }

            let mut cov = vec![0.0f64; FEAT_LEN * FEAT_LEN];
            for wm in &whitened_means {
                for a in 0..FEAT_LEN {
                    let da = wm[a] - wm_mean[a];
                    for b in a..FEAT_LEN {
                        let db = wm[b] - wm_mean[b];
                        cov[a * FEAT_LEN + b] += da * db;
                    }
                }
            }
            for a in 0..FEAT_LEN { for b in 0..a { cov[a * FEAT_LEN + b] = cov[b * FEAT_LEN + a]; } }

            let actual_dim = out_dim.min(FEAT_LEN);
            let eigvecs = crate::train::jacobi_eigen_top_k(&cov, FEAT_LEN, actual_dim);

            // Final projection: P = eigvecs^T * L^{-1}
            let mut proj = vec![0.0f64; actual_dim * FEAT_LEN];
            for d in 0..actual_dim {
                for j in 0..FEAT_LEN {
                    let mut sum = 0.0f64;
                    for k in 0..FEAT_LEN {
                        sum += eigvecs[d * FEAT_LEN + k] * linv[k * FEAT_LEN + j];
                    }
                    proj[d * FEAT_LEN + j] = sum;
                }
            }


            // Scale calibration — rescale projection so median within-class d²
            // equals target_dist (0.03).  Keeps f32 distances in a numerically
            // useful range for the softmax kernel.  Store scale² in med_nn so
            let target_dist = 0.03f64;
            let mut within_dists: Vec<f64> = Vec::new();
            let mut rng_cal = SmallRng::seed_from_u64(si as u64 + 9999);
            for (&_fid, indices) in &font_indices {
                if indices.len() < 2 { continue; }
                let npairs = 5.min(indices.len() * (indices.len() - 1) / 2);
                for _ in 0..npairs {
                    let a = indices[rng_cal.gen_range(0..indices.len())];
                    let b = loop {
                        let x = indices[rng_cal.gen_range(0..indices.len())];
                        if x != a { break x; }
                    };
                    let mut d = 0.0f64;
                    for dim in 0..actual_dim {
                        let mut ea = 0.0f64;
                        let mut eb = 0.0f64;
                        for j in 0..FEAT_LEN {
                            ea += proj[dim * FEAT_LEN + j] * samples[a].features[j] as f64;
                            eb += proj[dim * FEAT_LEN + j] * samples[b].features[j] as f64;
                        }
                        let diff = ea - eb;
                        d += diff * diff;
                    }
                    within_dists.push(d);
                }
            }
            let scale = if !within_dists.is_empty() {
                within_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = within_dists[within_dists.len() / 2];
                if median > 1e-15 { (target_dist / median).sqrt() } else { 1.0 }
            } else { 1.0 };
            for v in &mut proj { *v *= scale; }

            let proj_f32: Vec<f32> = proj.iter().map(|&v| v as f32).collect();

            // Evaluate MRR
            let eval_indices = crate::train::subsample_eval(n, 2000, si as u64);
            let centroid_embeds: Vec<Vec<f64>> = centroid_fids.iter().map(|fid| {
                let cm = &class_means[fid];
                let mut emb = vec![0.0f64; actual_dim];
                for d in 0..actual_dim {
                    for j in 0..FEAT_LEN { emb[d] += proj[d * FEAT_LEN + j] * cm[j]; }
                }
                emb
            }).collect();

            // Compute sigma_sq
            let sigma_sq: f32 = {
                let nc = centroid_embeds.len();
                let mut pairwise: Vec<f64> = Vec::with_capacity(nc * (nc - 1) / 2);
                for i in 0..nc {
                    for j in (i + 1)..nc {
                        let mut d = 0.0f64;
                        for k in 0..actual_dim {
                            let diff = centroid_embeds[i][k] - centroid_embeds[j][k];
                            d += diff * diff;
                        }
                        pairwise.push(d);
                    }
                }
                if pairwise.is_empty() { 0.0 }
                else {
                    pairwise.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    pairwise[pairwise.len() / 2] as f32
                }
            };

            let char_stats = crate::train::eval_mrr(
                &samples, &eval_indices, &class_means, &centroid_fids,
                &ctx.glyph_family_for_seq(seq),
                &|i| {
                    let mut emb = vec![0.0f64; actual_dim];
                    for d in 0..actual_dim {
                        for j in 0..FEAT_LEN {
                            emb[d] += proj[d * FEAT_LEN + j] * samples[i].features[j] as f64;
                        }
                    }
                    centroid_fids.iter().enumerate().map(|(ci2, &fid)| {
                        let mut d = 0.0f64;
                        for j in 0..actual_dim {
                            let diff = emb[j] - centroid_embeds[ci2][j];
                            d += diff * diff;
                        }
                        (fid, d)
                    }).collect()
                },
            );

            if si < 5 || si == sequences.len() - 1 || (si + 1) % 20 == 0 {
                eprintln!("  seq {:?} base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    seq, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }
            total_stats.accumulate(&char_stats);
            // Compute d95: nearest foreign centroid distance.
            // For each centroid, find the distance to its closest
            // *different-character* centroid.  Use the median of these
            // nearest-foreign distances as the OOD scale for this char.
            // Geometric interpretation: confidence ~ 0.5 when min_d equals
            // the inter-class gap, so observations inside the gap get high
            // confidence and those outside get suppressed.
            let d95: f32 = {
                let nc = centroid_embeds.len();
                if nc < 2 { 0.0 }
                else {
                    let mut nearest_foreign: Vec<f64> = Vec::with_capacity(nc);
                    for i in 0..nc {
                        let mut best = f64::MAX;
                        for j in 0..nc {
                            if i == j { continue; }
                            let mut d = 0.0f64;
                            for k in 0..actual_dim {
                                let diff = centroid_embeds[i][k] - centroid_embeds[j][k];
                                d += diff * diff;
                            }
                            if d < best { best = d; }
                        }
                        nearest_foreign.push(best);
                    }
                    nearest_foreign.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    nearest_foreign[nearest_foreign.len() / 2] as f32
                }
            };

            lda_chars.push((seq.clone(), actual_dim, proj_f32, sigma_sq, class_means, d95));
        }

        let lda_elapsed = lda_start.elapsed();
        eprintln!("\nLDA complete: {} seqs, {} skipped, {:.1}s",
            lda_chars.len(), skipped, lda_elapsed.as_secs_f64());
        eprintln!("  Baseline:   MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  LDA strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  LDA family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write LDAC v7 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let mut model = NgramModel::new(ctx.catalog_hash);
        for (seq, actual_dim, proj, sigma, class_means, d95) in &lda_chars {
            let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(class_means.len());
            for (&fid, mean) in class_means {
                let mut embedded = vec![0.0f32; *actual_dim];
                for d in 0..*actual_dim {
                    for j in 0..FEAT_LEN {
                        embedded[d] += proj[d * FEAT_LEN + j] * mean[j] as f32;
                    }
                }
                centroids.push((fid, embedded));
            }
            // Weights blob: [out_dim as f32, proj...]
            let mut weights = Vec::with_capacity(1 + proj.len());
            weights.push(*actual_dim as f32);
            weights.extend_from_slice(proj);

            let mut cm = ImageModel {
                weights,
                centroids,
                sigma_sq: *sigma,
                med_nn: *d95,
            };
            if cm.sigma_sq <= 1e-30 {
                cm.compute_sigma_sq();
            }
            model.entries.insert(seq.clone(), cm);
        }
        let tmp = crate::atomic_file::tmp_for(output);
        let f = std::fs::File::create(&tmp).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"LDAC", 8).expect("write LDAC v8");
        w.flush().unwrap();
        drop(w);
        std::fs::rename(&tmp, output).expect("atomic rename");

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e6, ctx.catalog.len());

        // ── New per-char forward cache: lda/{code:08x}.bin ────────────────
        // Atomic write per char: b"LDPC" | version | char_code | feat_dim | out_dim | sigma | med_nn | catalog_hash | proj
        // Enables lazy creation for unusual Unicode and fast per-char reload without monolithic file.
        let mut per_char_written = 0usize;
        for (seq, actual_dim, proj, sigma, _cm, d95) in &lda_chars {
            if seq.len() != 1 { continue; }
            let code = seq[0] as u32;
            let entry = crate::per_char_cache::LdaPerChar{
                char_code: code,
                feat_dim: FEAT_LEN,
                out_dim: *actual_dim,
                sigma_sq: *sigma,
                med_nn: *d95,
                catalog_hash: ctx.catalog_hash,
                projection: proj.clone(),
            };
            if let Err(e) = crate::per_char_cache::write_lda_atomic(&entry) {
                eprintln!("Warning: failed to write lda cache for U+{:04X}: {e}", code);
            } else {
                per_char_written += 1;
            }
        }
        eprintln!("  Per-char cache: {} lda files + {} means files (atomic)", per_char_written, per_char_written);
    }

}

impl Embedder for LdaClassifier {
    fn embed(&self, seq: &[char], features: &CropFeatures) -> Vec<f32> {
        LdaClassifier::embed(self, seq, features)
    }
    fn name(&self) -> &str { "lda" }
}

// ---------------------------------------------------------------------------
// MLP (per-character direct multi-class softmax classifier)
// ---------------------------------------------------------------------------

/// Inference-only linear layer (no gradients, no optimizer state).
/// Per-character MLP weights for direct multi-class classification.
struct MlpCharNet {
    fc1: InferenceLinear, // 100 → 256
    fc2: InferenceLinear, // 256 → 128
    fc3: InferenceLinear, // 128 → k
    class_map: Vec<u32>,  // class_index → font_id
}

impl MlpCharNet {
    /// Forward pass: ReLU(fc1) → ReLU(fc2) → fc3 (logits).
    /// Returns raw logits (length k). No softmax — caller handles scoring.
    fn forward(&self, raw: &[f32]) -> Vec<f32> {
        // Layer 1: ReLU(W1*x + b1)
        let mut h1 = self.fc1.forward(raw);
        for v in &mut h1 { *v = v.max(0.0); }

        // Layer 2: ReLU(W2*h1 + b2)
        let mut h2 = self.fc2.forward(&h1);
        for v in &mut h2 { *v = v.max(0.0); }

        // Layer 3: logits (no activation)
        self.fc3.forward(&h2)
    }
}

/// Per-character MLP classifier with direct softmax output.
///
/// Instead of embedding + nearest-centroid, this classifier runs a forward
/// pass through a per-character MLP and returns class probabilities directly.
/// Trained with cross-entropy loss and input noise augmentation for domain-gap
/// robustness.
///
/// Binary format (magic `b"MLPC"`):
/// ```text
/// magic:    b"MLPC" (4 bytes)
/// version:  u32 LE (2, only)
/// n_seqs:   u32 LE
/// Per sequence:
///   seq_len:   u32 LE (number of codepoints in this sequence)
///   codepoints: [u32; seq_len] LE
///   k:         u32 LE (number of classes)
///   class_map: [u32; k] LE (class_index → font_id)
///   W1: [f32; FEAT_LEN×256] LE, b1: [f32; 256] LE
///   W2: [f32; 256×128] LE, b2: [f32; 128] LE
///   W3: [f32; 128×k]   LE, b3: [f32; k]   LE
/// ```
pub struct MlpClassifier {
    nets: HashMap<Vec<char>, MlpCharNet>,
}

// MLP hidden layer sizes (must match trainer)
const MLP_H1: usize = 256;
const MLP_H2: usize = 128;

impl MlpClassifier {
    /// Load per-character MLP weights from a binary file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open MLP weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 {
            return Err("MLP file too small".into());
        }
        if &data[0..4] != b"MLPC" {
            return Err(format!("bad magic (expected MLPC, got {:?})", &data[0..4]));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 2 {
            return Err(format!("MLP weights are v{version}, need v2 — retraining required"));
        }
        let n_seqs = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

        let mut nets = HashMap::with_capacity(n_seqs);
        let mut r = BinaryReader::new(data, 12);

        for _ in 0..n_seqs {
            let seq_len = r.read_u32()? as usize;
            let mut seq = Vec::with_capacity(seq_len);
            for _ in 0..seq_len {
                let cp = r.read_u32()?;
                seq.push(char::from_u32(cp)
                    .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?);
            }
            let k = r.read_u32()? as usize;
            if k == 0 {
                return Err(format!("seq {:?}: zero classes", seq));
            }

            // class_map: k × u32
            let mut class_map = Vec::with_capacity(k);
            for _ in 0..k {
                class_map.push(r.read_u32()?);
            }

            // Layer weights
            let w1 = r.read_f32s(FEAT_LEN * MLP_H1)?;
            let b1 = r.read_f32s(MLP_H1)?;
            let w2 = r.read_f32s(MLP_H1 * MLP_H2)?;
            let b2 = r.read_f32s(MLP_H2)?;
            let w3 = r.read_f32s(MLP_H2 * k)?;
            let b3 = r.read_f32s(k)?;

            nets.insert(seq, MlpCharNet {
                fc1: InferenceLinear { rows: FEAT_LEN, cols: MLP_H1, w: w1, b: b1 },
                fc2: InferenceLinear { rows: MLP_H1, cols: MLP_H2, w: w2, b: b2 },
                fc3: InferenceLinear { rows: MLP_H2, cols: k, w: w3, b: b3 },
                class_map,
            });
        }

        Ok(Self { nets })
    }

    /// Train MLP weights from rendered training data and write an MLPC binary.
    pub fn train(
        ctx: &crate::train::TrainingContext,
        output: &std::path::Path,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        mlp_noise: f32,
        mlp_dropout: f32,
    ) {
        use std::io::{BufWriter, Write};
        use rand::prelude::*;
        use rand::rngs::SmallRng;
        use crate::train::Linear;

        let sequences = ctx.sequences;
        eprintln!("\nMLP training {} characters (epochs={}, noise={}, dropout={})...",
            sequences.len(), epochs, mlp_noise, mlp_dropout);
        let mlp_start = std::time::Instant::now();

        const MLP_H1: usize = 256;
        const MLP_H2: usize = 128;

        struct MlpNet {
            fc1: Linear,
            fc2: Linear,
            fc3: Linear,
        }

        struct MlpForwardCache {
            input: Vec<f32>,
            z1: Vec<f32>,
            h1: Vec<f32>,
            mask1: Vec<f32>,
            z2: Vec<f32>,
            h2: Vec<f32>,
            mask2: Vec<f32>,
            logits: Vec<f32>,
        }

        impl MlpNet {
            fn new(k: usize, rng: &mut SmallRng) -> Self {
                Self {
                    fc1: Linear::new(FEAT_LEN, MLP_H1, rng),
                    fc2: Linear::new(MLP_H1, MLP_H2, rng),
                    fc3: Linear::new(MLP_H2, k, rng),
                }
            }

            fn forward_train(&self, input: &[f32], dropout: f32, rng: &mut SmallRng) -> MlpForwardCache {
                let z1 = self.fc1.forward(input);
                let mut h1 = z1.clone();
                let mut mask1 = vec![1.0f32; MLP_H1];
                for j in 0..MLP_H1 {
                    h1[j] = h1[j].max(0.0);
                    if dropout > 0.0 && rng.gen::<f32>() < dropout {
                        h1[j] = 0.0;
                        mask1[j] = 0.0;
                    } else if dropout > 0.0 {
                        h1[j] /= 1.0 - dropout;
                    }
                }
                let z2 = self.fc2.forward(&h1);
                let mut h2 = z2.clone();
                let mut mask2 = vec![1.0f32; MLP_H2];
                for j in 0..MLP_H2 {
                    h2[j] = h2[j].max(0.0);
                    if dropout > 0.0 && rng.gen::<f32>() < dropout {
                        h2[j] = 0.0;
                        mask2[j] = 0.0;
                    } else if dropout > 0.0 {
                        h2[j] /= 1.0 - dropout;
                    }
                }
                let logits = self.fc3.forward(&h2);
                MlpForwardCache { input: input.to_vec(), z1, h1, mask1, z2, h2, mask2, logits }
            }

            fn forward_eval(&self, input: &[f32]) -> Vec<f32> {
                let z1 = self.fc1.forward(input);
                let h1: Vec<f32> = z1.iter().map(|&x| x.max(0.0)).collect();
                let z2 = self.fc2.forward(&h1);
                let h2: Vec<f32> = z2.iter().map(|&x| x.max(0.0)).collect();
                self.fc3.forward(&h2)
            }

            fn backward(&mut self, cache: &MlpForwardCache, d_logits: &[f32], dropout: f32) {
                let d_h2 = self.fc3.backward(&cache.h2, d_logits);
                let drop_scale2 = if dropout > 0.0 { 1.0 / (1.0 - dropout) } else { 1.0 };
                let d_z2: Vec<f32> = d_h2.iter().enumerate()
                    .map(|(j, &dh)| {
                        if cache.mask2[j] == 0.0 { return 0.0; }
                        let relu_grad = if cache.z2[j] > 0.0 { 1.0 } else { 0.0 };
                        dh * drop_scale2 * relu_grad
                    }).collect();
                let d_h1 = self.fc2.backward(&cache.h1, &d_z2);
                let drop_scale1 = if dropout > 0.0 { 1.0 / (1.0 - dropout) } else { 1.0 };
                let d_z1: Vec<f32> = d_h1.iter().enumerate()
                    .map(|(j, &dh)| {
                        if cache.mask1[j] == 0.0 { return 0.0; }
                        let relu_grad = if cache.z1[j] > 0.0 { 1.0 } else { 0.0 };
                        dh * drop_scale1 * relu_grad
                    }).collect();
                let _ = self.fc1.backward(&cache.input, &d_z1);
            }

            fn adam_step(&mut self, lr: f32, t: usize, batch_size: usize) {
                self.fc1.adam_step(lr, t, batch_size);
                self.fc2.adam_step(lr, t, batch_size);
                self.fc3.adam_step(lr, t, batch_size);
            }
        }

        let mut mlp_seqs: Vec<(Vec<char>, usize, Vec<u32>, MlpNet)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (si, seq) in sequences.iter().enumerate() {
            if ctx.seq_counts[si] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(si);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.glyph_id).or_default().push(i);
            }
            if font_indices.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let n = samples.len();

            let mut font_ids_sorted: Vec<u32> = font_indices.keys().copied().collect();
            font_ids_sorted.sort_unstable();
            let k = font_ids_sorted.len();
            let fid_to_class: HashMap<u32, usize> = font_ids_sorted.iter()
                .enumerate().map(|(ci2, &fid)| (fid, ci2)).collect();
            let class_map: Vec<u32> = font_ids_sorted.clone();

            let mut rng = SmallRng::seed_from_u64(si as u64);
            let mut net = MlpNet::new(k, &mut rng);
            let mut adam_t = 0usize;
            let mut sample_order: Vec<usize> = (0..n).collect();

            for epoch in 0..epochs {
                sample_order.shuffle(&mut rng);
                let mut epoch_loss = 0.0f64;

                for batch_start in (0..n).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(n);
                    let batch_len = batch_end - batch_start;

                    for &si in &sample_order[batch_start..batch_end] {
                        let label = fid_to_class[&samples[si].glyph_id];
                        let mut noisy = samples[si].features;
                        if mlp_noise > 0.0 {
                            for f in &mut noisy {
                                let u1: f32 = rng.gen::<f32>().max(1e-10);
                                let u2: f32 = rng.gen::<f32>();
                                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                                *f += z * mlp_noise;
                            }
                        }
                        let cache = net.forward_train(&noisy, mlp_dropout, &mut rng);
                        let max_logit = cache.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut probs = vec![0.0f32; k];
                        let mut sum_exp = 0.0f32;
                        for j in 0..k {
                            probs[j] = (cache.logits[j] - max_logit).exp();
                            sum_exp += probs[j];
                        }
                        for p in &mut probs { *p /= sum_exp; }
                        epoch_loss += -(probs[label].max(1e-10)).ln() as f64;
                        let mut d_logits = probs;
                        d_logits[label] -= 1.0;
                        net.backward(&cache, &d_logits, mlp_dropout);
                    }
                    adam_t += 1;
                    net.adam_step(lr, adam_t, batch_len);
                }

                if (epoch + 1) % 10 == 0 || epoch == 0 {
                    let avg_loss = epoch_loss / n as f64;
                    if si < 5 || si == sequences.len() - 1 {
                        eprintln!("  seq {:?} epoch {}/{}: loss={:.4}", seq, epoch + 1, epochs, avg_loss);
                    }
                }
            }

            // Evaluate
            let class_means: HashMap<u32, Vec<f64>> = font_indices.iter().map(|(&fid, indices)| {
                let mut mean = vec![0.0f64; FEAT_LEN];
                for &i in indices {
                    for j in 0..FEAT_LEN { mean[j] += samples[i].features[j] as f64; }
                }
                let cnt = indices.len() as f64;
                for j in 0..FEAT_LEN { mean[j] /= cnt; }
                (fid, mean)
            }).collect();

            let centroid_fids: Vec<u32> = class_means.keys().copied().collect();
            let eval_indices = crate::train::subsample_eval(n, 2000, si as u64);

            let char_stats = crate::train::eval_mrr(
                &samples, &eval_indices, &class_means, &centroid_fids,
                &ctx.glyph_family_for_seq(seq),
                &|i| {
                    let logits = net.forward_eval(&samples[i].features);
                    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs = vec![0.0f32; k];
                    let mut sum_exp = 0.0f32;
                    for j in 0..k {
                        probs[j] = (logits[j] - max_logit).exp();
                        sum_exp += probs[j];
                    }
                    for p in &mut probs { *p /= sum_exp; }
                    font_ids_sorted.iter().enumerate().map(|(ci2, &fid)| {
                        let neg_log_prob = -(probs[ci2].max(1e-10)).ln() as f64;
                        (fid, neg_log_prob)
                    }).collect()
                },
            );

            if si < 5 || si == sequences.len() - 1 || (si + 1) % 20 == 0 {
                eprintln!("  seq {:?} base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    seq, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }
            total_stats.accumulate(&char_stats);
            mlp_seqs.push((seq.clone(), k, class_map, net));
        }

        let mlp_elapsed = mlp_start.elapsed();
        eprintln!("\nMLP complete: {} seqs, {} skipped, {:.1}s",
            mlp_seqs.len(), skipped, mlp_elapsed.as_secs_f64());
        eprintln!("  Baseline:   MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  MLP strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  MLP family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write MLPC binary
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let tmp = crate::atomic_file::tmp_for(output);
        let f = std::fs::File::create(&tmp).expect("create output file");
        let mut w = BufWriter::new(f);
        w.write_all(b"MLPC").unwrap();
        w.write_all(&2u32.to_le_bytes()).unwrap();
        w.write_all(&(mlp_seqs.len() as u32).to_le_bytes()).unwrap();
        for (seq, k, class_map, net) in &mlp_seqs {
            // Write sequence length + codepoints
            w.write_all(&(seq.len() as u32).to_le_bytes()).unwrap();
            for &ch in seq { w.write_all(&(ch as u32).to_le_bytes()).unwrap(); }
            w.write_all(&(*k as u32).to_le_bytes()).unwrap();
            for &fid in class_map { w.write_all(&fid.to_le_bytes()).unwrap(); }
            for &v in &net.fc1.w { w.write_all(&v.to_le_bytes()).unwrap(); }
            for &v in &net.fc1.b { w.write_all(&v.to_le_bytes()).unwrap(); }
            for &v in &net.fc2.w { w.write_all(&v.to_le_bytes()).unwrap(); }
            for &v in &net.fc2.b { w.write_all(&v.to_le_bytes()).unwrap(); }
            for &v in &net.fc3.w { w.write_all(&v.to_le_bytes()).unwrap(); }
            for &v in &net.fc3.b { w.write_all(&v.to_le_bytes()).unwrap(); }
        }
        w.flush().unwrap();
        drop(w);
        std::fs::rename(&tmp, output).expect("atomic rename");

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB)", output.display(), file_size as f64 / 1e6);
    }

}

impl MlpClassifier {
    /// Compute softmax probabilities for a character query, returning (net, probs) if the char has a net.
    fn softmax_probs(&self, seq: &[char], query: &CropFeatures) -> Option<(&MlpCharNet, Vec<f32>)> {
        let net = self.nets.get(seq)?;
        let logits = net.forward(&query.as_slice());
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum_exp; }
        Some((net, probs))
    }
}

impl Classifier for MlpClassifier {
    fn classify(&self, seq: &[char], query: &CropFeatures, k: usize) -> Vec<(usize, f32)> {
        if let Some((net, probs)) = self.softmax_probs(seq, query) {
            let mut scored: Vec<(usize, f32)> = net.class_map.iter().enumerate()
                .map(|(ci, &fid)| (fid as usize, probs[ci]))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            scored
        } else {
            Vec::new()
        }
    }

    /// MLP produces probabilities natively via softmax — no σ² needed.
    fn probabilities(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        if let Some((net, probs)) = self.softmax_probs(seq, query) {
            let mut scored: Vec<(usize, f32)> = net.class_map.iter().enumerate()
                .map(|(ci, &fid)| (fid as usize, probs[ci]))
                .collect();
            // Sort descending by probability (highest first)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored
        } else {
            Vec::new()
        }
    }

    fn probability(&self, seq: &[char], query: &CropFeatures, glyph_id: usize) -> Option<f32> {
        let (net, probs) = self.softmax_probs(seq, query)?;
        for (ci, &fid) in net.class_map.iter().enumerate() {
            if fid as usize == glyph_id {
                return Some(probs[ci]);
            }
        }
        None
    }

    fn name(&self) -> &str {
        "mlp"
    }

    fn glyph_count(&self, seq: &[char]) -> usize {
        // MLP class_map length for the char's network
        self.nets.get(seq).map_or(0, |net| net.class_map.len())
    }
}

// ---------------------------------------------------------------------------
// Rank fusion (combines multiple classifiers)
// ---------------------------------------------------------------------------

/// Rank-fusion classifier that combines scores from multiple child classifiers.
///
/// At query time, calls each child's `classify()`, normalizes scores to [0,1],
/// and computes a weighted average.  The fused score determines final ranking.
pub struct FusionClassifier {
    children: Vec<(f32, Box<dyn Classifier>)>, // (weight, classifier)
    /// Cached sum of weights for normalization.
    weight_sum: f32,
}

impl FusionClassifier {
    /// Create a new fusion classifier from weighted children.
    /// Weights do not need to sum to 1 — they are normalized internally.
    pub fn new(children: Vec<(f32, Box<dyn Classifier>)>) -> Self {
        let weight_sum: f32 = children.iter().map(|(w, _)| *w).sum();
        Self { children, weight_sum }
    }
}

impl Classifier for FusionClassifier {
    fn classify(&self, seq: &[char], query: &CropFeatures, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self.probabilities(seq, query);
        probs.truncate(k);
        probs
    }

    /// Fusion probabilities via weighted geometric mean of child posteriors.
    ///
    /// For each font, computes `exp(Σ w_i * ln(p_i)) / Z` where `p_i` is
    /// child i's probability and `w_i` is its normalized weight.  This is
    /// equivalent to `(∏ p_i^w_i) / Z`, the weighted geometric mean of
    /// individual posteriors, renormalized.
    fn probabilities(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        // Collect child probability distributions
        let child_probs: Vec<(f32, HashMap<usize, f32>)> = self.children.iter()
            .map(|(weight, child)| {
                let probs = child.probabilities(seq, query);
                let map: HashMap<usize, f32> = probs.into_iter().collect();
                (*weight / self.weight_sum, map)
            })
            .collect();

        // Union of all font_ids
        let mut all_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (_, map) in &child_probs {
            for &id in map.keys() { all_ids.insert(id); }
        }

        // Weighted sum of log-probabilities (geometric mean in log space)
        let log_scores: Vec<(usize, f32)> = all_ids.into_iter().map(|id| {
            let mut log_p = 0.0f32;
            for (w, map) in &child_probs {
                let p = map.get(&id).copied().unwrap_or(1e-30);
                log_p += w * p.max(1e-30).ln();
            }
            (id, log_p)
        }).collect();

        // Softmax normalization
        let max_lp = log_scores.iter().map(|(_, lp)| *lp)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = log_scores.iter().map(|(_, lp)| (lp - max_lp).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

        let mut result: Vec<(usize, f32)> = log_scores.iter().zip(exps)
            .map(|(&(id, _), e)| (id, e * inv_sum))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    fn name(&self) -> &str {
        "fusion"
    }

    fn glyph_count(&self, seq: &[char]) -> usize {
        self.children.iter().map(|(_, c)| c.glyph_count(seq)).max().unwrap_or(0)
    }

    fn add_glyph(&mut self, glyph_id: usize, seq: &[char], features: &CropFeatures) {
        for (_, child) in &mut self.children {
            child.add_glyph(glyph_id, seq, features);
        }
    }

    fn ensure_owned(&mut self) {
        for (_, child) in &mut self.children {
            child.ensure_owned();
        }
    }

    fn recompute_stats(&mut self) {
        for (_, child) in &mut self.children {
            child.recompute_stats();
        }
    }

    fn set_catalog_hash(&mut self, hash: u64) {
        for (_, child) in &mut self.children {
            child.set_catalog_hash(hash);
        }
    }

    fn save_to(&self, _path: &std::path::Path, _magic: &[u8; 4], _version: u32) -> Result<(), String> {
        // Fusion doesn't have a single file; each child saves separately via build_classifier.
        // For incremental, we propagate save to children using their default paths.
        for (_, child) in &self.children {
            // Determine child's default path by name
            let (default_path, magic, version) = match child.name() {
                "lda" => (default_lda_weights_path(), *b"LDAC", 8u32),
                "perchar-fisher" | "fisher" => (default_fisher_weights_path(), *b"FISH", 3u32),
                "mahalanobis" => (default_mahalanobis_weights_path(), *b"MAHA", 3u32),
                "triplet" => (default_triplet_weights_path(), *b"TRIP", 3u32),
                _ => continue,
            };
            let _ = child.save_to(&default_path, &magic, version);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Classifier construction
// ---------------------------------------------------------------------------

/// Default path for cached LDA weights.
pub fn default_lda_weights_path() -> std::path::PathBuf {
    crate::cache::paths::lda_weights_bin()
}

/// Default path for cached per-char Fisher weights.
pub fn default_fisher_weights_path() -> std::path::PathBuf {
    crate::cache::paths::fisher_weights_bin()
}

/// Default path for cached triplet weights.
pub fn default_triplet_weights_path() -> std::path::PathBuf {
    crate::cache::paths::triplet_weights_bin()
}

/// Default path for cached Mahalanobis weights.
pub fn default_mahalanobis_weights_path() -> std::path::PathBuf {
    crate::cache::paths::mahalanobis_weights_bin()
}

/// Default path for cached MLP weights.
pub fn default_mlp_weights_path() -> std::path::PathBuf {
    crate::cache::paths::mlp_weights_bin()
}

/// Default path for the font catalog file.
pub fn default_catalog_path() -> std::path::PathBuf {
    crate::cache::paths::catalog_bin()
}

/// Try to load a classifier from its cached weights, auto-training if missing.
///
/// 1. If  is explicitly set, load from it (fail hard on error).
/// 2. Otherwise try .
/// 3. If missing, auto-train using  to produce .
/// Read just the catalog_hash from catalog.bin (FONT header).
/// Returns None if the file is missing, too small, or has wrong magic.
fn read_catalog_hash() -> Option<u64> {
    let path = default_catalog_path();
    let data = std::fs::read(&path).ok()?;
    if data.len() < 16 || &data[0..4] != b"FONT" { return None; }
    Some(u64::from_le_bytes(data[8..16].try_into().unwrap()))
}

/// Check if any training feature manifest is older than font_scan.bin.
/// If so, any classifier weights derived from those features are suspect
/// (glyph IDs may correspond to a stale catalog ordering).
fn training_features_stale() -> bool {
    if !crate::cache::is_default_cache_dir() {
        return false;
    }
    let scan_path = crate::font_scan::scan_cache_path();
    let scan_mtime = match std::fs::metadata(&scan_path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false, // no font_scan cache → nothing to compare
    };
    let feat_dir = crate::cache::paths::training_dir();
    let entries = match std::fs::read_dir(&feat_dir) {
        Ok(e) => e,
        Err(_) => return false, // no training dir → nothing stale
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("manifest_") { continue; }
        // Content check: the manifest header carries the ink threshold its
        // features were extracted at; a threshold change forces regeneration.
        let th_tag = format!("th={}", crate::INK_THRESH);
        if let Ok(content) = std::fs::read_to_string(entry.path()) {
            let header_ok = content
                .lines()
                .next()
                .map(|h| h.split_whitespace().any(|t| t == th_tag))
                .unwrap_or(false);
            if !header_ok {
                return true;
            }
        } else {
            return true;
        }
        if let Ok(meta) = entry.metadata() {
            if let Ok(mtime) = meta.modified() {
                if mtime < scan_mtime {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if the weights file is older than any training feature manifest.
/// This catches the case where features were re-rendered (fixing a stale
/// catalog) but the process died before retraining the weights.
fn weights_older_than_features(weights_path: &std::path::Path) -> bool {
    if !crate::cache::is_default_cache_dir() {
        return false;
    }
    let weights_mtime = match std::fs::metadata(weights_path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false, // no weights file → nothing to compare
    };
    let feat_dir = crate::cache::paths::training_dir();
    let entries = match std::fs::read_dir(&feat_dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("manifest_") { continue; }
        if let Ok(meta) = entry.metadata() {
            if let Ok(mtime) = meta.modified() {
                if weights_mtime < mtime {
                    return true;
                }
            }
        }
    }
    false
}

fn load_or_train<F, T>(
    name: &str,
    weights_path: Option<&std::path::Path>,
    default_path: &std::path::Path,
    load_fn: fn(&std::path::Path) -> Result<T, String>,
    auto_train: Option<(&[std::path::PathBuf], &crate::char_render::RenderParams)>,
    train_fn: F,
) -> Box<dyn Classifier>
where
    T: Classifier + 'static,
    F: FnOnce(&[std::path::PathBuf], &crate::char_render::RenderParams, &std::path::Path),
{
    // Explicit path: load or die.
    if let Some(wp) = weights_path {
        match load_fn(wp) {
            Ok(c) => return Box::new(c),
            Err(e) => { eprintln!("Error loading {name} weights from {}: {e}", wp.display()); std::process::exit(1); }
        }
    }

    // Try default cache path — also validate catalog_hash against catalog.bin.
    if default_path.exists() {
        match load_fn(default_path) {
            Ok(c) => {
                // Validate catalog_hash: if catalog.bin exists and the model
                // has a different hash, the font catalog changed since training.
                if let Some(expected) = read_catalog_hash() {
                    if let Some(model_hash) = c.catalog_hash() {
                        if model_hash != expected {
                            eprintln!("{name} weights at {} stale (catalog_hash {model_hash:#x} != {expected:#x}), retraining...",
                                default_path.display());
                            // fall through to auto-train below
                        } else if training_features_stale() {
                            eprintln!("{name} weights at {} suspect (training features older than font_scan cache), retraining...",
                                default_path.display());
                            // fall through to auto-train below
                        } else if weights_older_than_features(default_path) {
                            eprintln!("{name} weights at {} stale (older than training features), retraining...",
                                default_path.display());
                            // fall through to auto-train below
                        } else {
                            return Box::new(c);
                        }
                    } else {
                        return Box::new(c);
                    }
                } else {
                    return Box::new(c);
                }
            }
            Err(e) => eprintln!("Warning: cached {name} weights at {} are corrupt ({e}), retraining...", default_path.display()),
        }
    }

    // Auto-train.
    if let Some((font_dir, render_params)) = auto_train {
        eprintln!("No {name} weights found, training automatically...");
        train_fn(font_dir, render_params, default_path);
        if !default_path.exists() {
            eprintln!("Auto-training failed to produce {name} weights at {}", default_path.display());
            std::process::exit(1);
        }
        eprintln!("{name} auto-training complete.");
    } else {
        eprintln!("No {name} weights found at {} and auto-training not available", default_path.display());
        std::process::exit(1);
    }

    match load_fn(default_path) {
        Ok(c) => Box::new(c),
        Err(e) => { eprintln!("Error loading {name} weights from {}: {e}", default_path.display()); std::process::exit(1); }
    }
}

fn train_lda(font_dir: &[std::path::PathBuf], render_params: &crate::char_render::RenderParams, output: &std::path::Path) {
    crate::train::run_train(crate::train::TrainArgs {
        output: output.to_path_buf(),
        font_dir: font_dir.to_vec(),
        render_params: render_params.clone(),
        lda: true,
        ..crate::train::TrainArgs::default()
    });
}

fn train_fisher(font_dir: &[std::path::PathBuf], render_params: &crate::char_render::RenderParams, output: &std::path::Path) {
    crate::train::run_train(crate::train::TrainArgs {
        output: output.to_path_buf(),
        font_dir: font_dir.to_vec(),
        render_params: render_params.clone(),
        fisher: true,
        lda: false,
        ..crate::train::TrainArgs::default()
    });
}

fn train_mahalanobis(font_dir: &[std::path::PathBuf], render_params: &crate::char_render::RenderParams, output: &std::path::Path) {
    crate::train::run_train(crate::train::TrainArgs {
        output: output.to_path_buf(),
        font_dir: font_dir.to_vec(),
        render_params: render_params.clone(),
        mahalanobis: true,
        lda: false,
        ..crate::train::TrainArgs::default()
    });
}

fn train_triplet(font_dir: &[std::path::PathBuf], render_params: &crate::char_render::RenderParams, output: &std::path::Path) {
    crate::train::run_train(crate::train::TrainArgs {
        output: output.to_path_buf(),
        font_dir: font_dir.to_vec(),
        render_params: render_params.clone(),
        lda: false,
        ..crate::train::TrainArgs::default()
    });
}

fn train_mlp(font_dir: &[std::path::PathBuf], render_params: &crate::char_render::RenderParams, output: &std::path::Path) {
    crate::train::run_train(crate::train::TrainArgs {
        output: output.to_path_buf(),
        font_dir: font_dir.to_vec(),
        render_params: render_params.clone(),
        mlp: true,
        lda: false,
        ..crate::train::TrainArgs::default()
    });
}

/// Build a classifier by name.
///
/// : explicit weights file override.
/// : if Some, auto-train when weights are missing.
pub fn build_classifier(
    classifier_name: &str,
    weights_path: Option<&std::path::Path>,
    auto_train: Option<(&[std::path::PathBuf], &crate::char_render::RenderParams)>,
) -> Box<dyn Classifier> {
    match classifier_name {
        "lda" => load_or_train(
            "LDA", weights_path, &default_lda_weights_path(),
            |p| LdaClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
            auto_train, train_lda,
        ),
        "perchar-fisher" => load_or_train(
            "Fisher", weights_path, &default_fisher_weights_path(),
            |p| PerCharFisherClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
            auto_train, train_fisher,
        ),
        "triplet" => load_or_train(
            "Triplet", weights_path, &default_triplet_weights_path(),
            |p| TripletClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
            auto_train, train_triplet,
        ),
        "mahalanobis" => load_or_train(
            "Mahalanobis", weights_path, &default_mahalanobis_weights_path(),
            |p| MahalanobisClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
            auto_train, train_mahalanobis,
        ),
        "mlp" => load_or_train(
            "MLP", weights_path, &default_mlp_weights_path(),
            |p| MlpClassifier::load(p),
            auto_train, train_mlp,
        ),
        "fusion" => {
            let lda = load_or_train(
                "LDA", weights_path, &default_lda_weights_path(),
                |p| LdaClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
                auto_train, train_lda,
            );
            let fisher = load_or_train(
                "Fisher", None, &default_fisher_weights_path(),
                |p| PerCharFisherClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
                auto_train, train_fisher,
            );
            Box::new(FusionClassifier::new(vec![
                (0.5, lda),
                (0.5, fisher),
            ]))
        }
        "zncc" => {
            if let Some((font_dirs, render_params)) = auto_train {
                let gmap_path = crate::glyph_map::NgramGlyphMap::default_path();
                let glyph_map = match crate::glyph_map::NgramGlyphMap::load(&gmap_path) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("Glyph map stale or missing ({e}), retraining...");
                        let lda_path = default_lda_weights_path();
                        crate::train::run_train(crate::train::TrainArgs {
                            output: lda_path,
                            font_dir: font_dirs.to_vec(),
                            render_params: render_params.clone(),
                            lda: true,
                            ..crate::train::TrainArgs::default()
                        });
                        crate::glyph_map::NgramGlyphMap::load(&gmap_path)
                            .unwrap_or_else(|e2| {
                                eprintln!("Retraining did not produce a valid glyph map ({e2})");
                                std::process::exit(1);
                            })
                    }
                };
                Box::new(crate::zncc_classifier::ZnccClassifier::from_glyph_map(glyph_map, render_params))
            } else {
                eprintln!("ZNCC classifier requires font directories");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("Error: unknown classifier '{other}'. Use 'lda', 'perchar-fisher', 'triplet', 'mahalanobis', 'mlp', 'fusion', or 'zncc'.");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-font 1-gram LDA — classes = characters, one classifier per font
// ---------------------------------------------------------------------------
//
// Trained per font: given this font, which character is this crop?
// Used after font selection to find stray OCR errors.
//
// IMPORTANT: uses HOG features (128-dim) for character discrimination,
// NOT the main CropFeatures (63-dim) which are designed for font
// discrimination.  Different tasks need different features.
//
// Each font's classifier is cached to its own file under
// ~/.cache/unprint/per-font-lda/<hash>.bin

/// Conforms to `src/font_cache.rs` pattern (shared LRU, same eviction/promotion logic).
/// See `docs/debugging-segmentation.md` — "Font cache (shared LRU)" entry.

use std::collections::VecDeque;
use std::sync::Mutex;

/// Default number of per-font LDA classifiers to keep in memory.
/// Each classifier ~5-6 MB (HOG 144-dim → LDA 31-dim, ~100-200 centroids),
/// so 16 ≈ 80-100 MB peak vs unbounded 79-font ≈ 400 MB+ which OOMs on 7.8G VM.
pub const PER_FONT_CACHE_DEFAULT_CAPACITY: usize = 16;

struct PerFontLruInner {
    entries: HashMap<String, std::sync::Arc<PerFontLda>>,
    order: VecDeque<String>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

struct PerFontLruCache {
    inner: Mutex<PerFontLruInner>,
}

impl PerFontLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(PerFontLruInner {
                entries: HashMap::with_capacity(capacity),
                order: VecDeque::with_capacity(capacity),
                capacity,
                hits: 0,
                misses: 0,
            }),
        }
    }

    fn get(&self, key: &str) -> Option<std::sync::Arc<PerFontLda>> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(v) = inner.entries.get(key).cloned() {
            // Promote to most-recent
            if let Some(pos) = inner.order.iter().position(|k| k == key) {
                inner.order.remove(pos);
            }
            inner.order.push_back(key.to_string());
            inner.hits += 1;
            Some(v)
        } else {
            inner.misses += 1;
            None
        }
    }

    fn put(&self, key: String, value: std::sync::Arc<PerFontLda>) {
        let mut inner = self.inner.lock().unwrap();
        if inner.entries.contains_key(&key) {
            inner.entries.insert(key.clone(), value);
            if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                inner.order.remove(pos);
            }
            inner.order.push_back(key);
            return;
        }
        // Evict oldest if at capacity (while to handle shrink)
        while inner.entries.len() >= inner.capacity {
            if let Some(old) = inner.order.pop_front() {
                inner.entries.remove(&old);
            } else {
                break;
            }
        }
        inner.order.push_back(key.clone());
        inner.entries.insert(key, value);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    #[allow(dead_code)]
    fn stats(&self) -> (u64, u64) {
        let inner = self.inner.lock().unwrap();
        (inner.hits, inner.misses)
    }
}

/// In-memory LRU cache of loaded per-font classifiers, keyed by font_key.
/// Mirrors `src/font_cache.rs::FontCache` pattern: `Mutex<LruInner>`, `HashMap` + `VecDeque`,
/// `while len >= capacity` eviction, `hits`/`misses` counters, `len()`/`stats()`.
/// Cap 16 keeps peak ~80-100M vs unbounded ~400M+ for 79-font specimen.
static PER_FONT_CACHE: std::sync::LazyLock<PerFontLruCache> =
    std::sync::LazyLock::new(|| PerFontLruCache::new(PER_FONT_CACHE_DEFAULT_CAPACITY));

/// Per-font 1-gram LDA classifier using HOG features.
///
/// Self-contained: stores its own projection, centroids, and probability
/// parameters.  Does NOT depend on `EmbeddingClassifier` or the `Embedder`
/// trait (those are locked to CropFeatures for font identification).
pub struct PerFontLda {
    /// LDA projection matrix: [out_dim × feat_dim] row-major
    projection: Vec<f32>,
    out_dim: usize,
    /// Input feature dimension (HOG_FEAT_LEN or HOG_FEAT_LEN + metric features)
    feat_dim: usize,
    /// Per-class centroids in projected space: (class_index, embedding)
    centroids: Vec<Vec<f32>>,
    /// RBF kernel bandwidth for probability computation
    sigma_sq: f32,
    /// class_index → character
    char_map: Vec<char>,
    font_key: String,
    catalog_hash: u64,
}

impl PerFontLda {
    /// Cache directory for per-font LDA classifiers.
    fn cache_dir() -> std::path::PathBuf {
        crate::cache::paths::per_font_lda_dir()
    }

    /// Deterministic cache path for a font_key.
    /// Uses FNV-1a 64-bit to avoid DefaultHasher's random SipHash seed which
    /// makes cache file names non-deterministic across processes (causing cache misses + retraining OOM).
    fn cache_path(font_key: &str) -> std::path::PathBuf {
        // FNV-1a 64-bit
        let mut h: u64 = 0xcbf29ce484222325;
        for b in font_key.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Self::cache_dir().join(format!("{:016x}.bin", h))
    }

    /// Load or train the per-font classifier for `font_key`.
    /// Returns `None` if the font has fewer than 5 trainable characters.
    pub fn load_or_train(
        font_key: &str,
        ctx: &crate::train::TrainingContext,
    ) -> Option<std::sync::Arc<Self>> {
        // Check in-memory cache first (LRU, cap 16)
        if let Some(arc) = PER_FONT_CACHE.get(font_key) {
            return Some(arc);
        }

        // Try loading from disk
        let path = Self::cache_path(font_key);
        if path.exists() {
            if let Some(clf) = Self::load(&path, font_key, ctx.catalog_hash) {
                let arc = std::sync::Arc::new(clf);
                PER_FONT_CACHE.put(font_key.to_string(), arc.clone());
                return Some(arc);
            }
            // Stale or corrupt — fall through to retrain
        }

        // Train
        let clf = Self::train(font_key, ctx)?;
        if let Err(e) = clf.save() {
            eprintln!("Warning: could not cache per-font LDA for {font_key}: {e}");
        }
        let arc = std::sync::Arc::new(clf);
        PER_FONT_CACHE.put(font_key.to_string(), arc.clone());
        Some(arc)
    }

    /// Predict which character best matches the HOG features.
    /// Returns `(char, probability)` pairs sorted descending by probability.
    #[allow(dead_code)] // character-level classification — not yet wired into pipeline
    pub fn predict(&self, feats: &[f32], k: usize) -> Vec<(char, f32)> {
        let feat_dim = self.feat_dim;

        // Project features into LDA space
        let mut emb = vec![0.0f32; self.out_dim];
        for d in 0..self.out_dim {
            let row_off = d * feat_dim;
            let mut sum = 0.0f32;
            for j in 0..feat_dim {
                sum += self.projection[row_off + j] * feats[j];
            }
            emb[d] = sum;
        }

        // Compute squared distances to all centroids
        let mut dists: Vec<(usize, f32)> = self.centroids.iter().enumerate()
            .map(|(ci, cent)| {
                let d: f32 = emb.iter().zip(cent.iter())
                    .map(|(&a, &b)| { let d = a - b; d * d })
                    .sum();
                (ci, d)
            })
            .collect();

        // Sort by distance ascending
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to softmax probabilities via RBF kernel
        let min_dist = dists.first().map(|d| d.1).unwrap_or(0.0);
        let mut scores: Vec<(usize, f32)> = dists.iter()
            .map(|&(ci, d)| (ci, (-(d - min_dist) / (2.0 * self.sigma_sq)).exp()))
            .collect();
        let total: f32 = scores.iter().map(|(_, s)| s).sum();
        if total > 0.0 {
            for s in &mut scores { s.1 /= total; }
        }

        scores.iter()
            .take(k)
            .filter_map(|&(ci, prob)| self.char_map.get(ci).map(|&ch| (ch, prob)))
            .collect()
    }

    /// Predict top-1 character.
    #[allow(dead_code)]
    pub fn predict_top1(&self, hog: &[f32]) -> Option<(char, f32)> {
        self.predict(hog, 1).into_iter().next()
    }

    /// Expose sigma_sq for diagnostics.
    pub fn sigma_sq(&self) -> f32 { self.sigma_sq }

    /// Predict with raw squared distances for diagnostics.
    /// Returns (char, probability, squared_distance) triples sorted by distance ascending.
    pub fn predict_with_distances(&self, feats: &[f32], k: usize) -> Vec<(char, f32, f32)> {
        let feat_dim = self.feat_dim;

        let mut emb = vec![0.0f32; self.out_dim];
        for d in 0..self.out_dim {
            let row_off = d * feat_dim;
            let mut sum = 0.0f32;
            for j in 0..feat_dim {
                sum += self.projection[row_off + j] * feats[j];
            }
            emb[d] = sum;
        }

        let mut dists: Vec<(usize, f32)> = self.centroids.iter().enumerate()
            .map(|(ci, cent)| {
                let d: f32 = emb.iter().zip(cent.iter())
                    .map(|(&a, &b)| { let d = a - b; d * d })
                    .sum();
                (ci, d)
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let min_dist = dists.first().map(|d| d.1).unwrap_or(0.0);
        let mut scores: Vec<(usize, f32, f32)> = dists.iter()
            .map(|&(ci, d)| (ci, (-(d - min_dist) / (2.0 * self.sigma_sq)).exp(), d))
            .collect();
        let total: f32 = scores.iter().map(|(_, s, _)| s).sum();
        if total > 0.0 {
            for s in &mut scores { s.1 /= total; }
        }

        scores.iter()
            .take(k)
            .filter_map(|&(ci, prob, dist)| self.char_map.get(ci).map(|&ch| (ch, prob, dist)))
            .collect()
    }

    /// Train a per-font LDA classifier from rendered glyphs.
    ///
    /// For each supported character, renders the glyph at multiple
    /// degradation levels, computes HOG features, and trains LDA with
    /// characters as classes.
    fn train(font_key: &str, ctx: &crate::train::TrainingContext) -> Option<Self> {
        use crate::hog::{HOG_FEAT_LEN, compute_hog};
        use crate::features::NORM_H;

        eprintln!("[pflda] Training per-font LDA for {font_key}");
        // Find the font entry in the catalog
        let font_entry = match ctx.catalog.iter().find(|fe| fe.font_key() == font_key) {
            Some(fe) => fe,
            None => {
                eprintln!("[pflda] Font entry not found in catalog for {font_key}");
                return None;
            }
        };

        // Parse font data for rendering
        // Font bytes are dropped from FontEntry after initial scan to save memory.
        // Read from the file path instead.
        let font_data = match std::fs::read(&font_entry.path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[pflda] Cannot read font file {:?}: {e}", font_entry.path);
                return None;
            }
        };
        let mut font = match unprint_fonts::ab_glyph::FontVec::try_from_vec(font_data) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[pflda] FontVec parse failed for {font_key}: {e}");
                return None;
            }
        };

        // Apply variable-font axis coordinates (e.g. weight)
        if let Some(ref vars) = font_entry.variations {
            use unprint_fonts::ab_glyph::VariableFont;
            for (tag, val) in vars {
                font.set_variation(tag, *val);
            }
        }

        // Degradation scale factors for training augmentation.
        // These simulate different document resolutions so the within-class
        // scatter captures real-world variation.
        const DEGRADE_SCALES: &[f32] = &[1.0, 0.85, 0.70];

        let overrides = font_entry.glyph_overrides.as_deref();

        // Compute glyph metric ratios for all supported characters
        let all_chars: Vec<char> = ctx.sequences.iter()
            .filter(|s| s.len() == 1)
            .map(|s| s[0])
            .collect();
        let glyph_metrics = crate::char_render::glyph_metric_ratios(
            &font, &all_chars, overrides,
        );
        let feat_dim = HOG_FEAT_LEN + 2; // HOG + top_frac + bottom_frac

        // Gather per-character feature samples (HOG + glyph metrics)
        let mut char_samples: Vec<(char, Vec<Vec<f32>>)> = Vec::new();
        let mut total_samples = 0usize;

        for seq in ctx.sequences.iter() {
            // PerFontLda is per-character HOG — skip bigrams
            if seq.len() != 1 { continue; }
            let ch = seq[0];

            // Check that the glyph map recognizes this font for this character
            if ctx.glyph_map.glyph_id_for_font(seq, font_key).is_none() {
                continue;
            }

            // Resolve glyph ID (handles OT feature overrides)
            let gid = crate::char_render::resolve_glyph(&font, ch, overrides);
            if gid.0 == 0 { continue; } // .notdef

            // Render at NORM_H
            let base_img = crate::char_render::render_glyph_at_ink_height(
                &font, gid, NORM_H,
            );
            let base_img = match base_img {
                Some(img) if img.width() >= 3 && img.height() >= 3 => img,
                _ => continue,
            };

            // Get glyph metric features for this character
            let (metric_top, metric_bot) = glyph_metrics.get(&ch)
                .copied()
                .unwrap_or((0.0, 0.0));

            let mut feats: Vec<Vec<f32>> = Vec::new();

            for &scale in DEGRADE_SCALES {
                let img = if scale >= 0.999 {
                    base_img.clone()
                } else {
                    match crate::features::degrade_and_renormalize(&base_img, scale) {
                        Some(img) => img,
                        None => continue,
                    }
                };

                if let Some(h) = compute_hog(&img) {
                    let mut fv = Vec::with_capacity(feat_dim);
                    fv.extend_from_slice(&h);
                    fv.push(metric_top);
                    fv.push(metric_bot);
                    feats.push(fv);
                }
            }

            if feats.is_empty() { continue; }
            total_samples += feats.len();
            char_samples.push((ch, feats));
        }

        eprintln!("[pflda] {font_key}: {} chars, {} total HOG samples", char_samples.len(), total_samples);
        if char_samples.len() < 5 {
            eprintln!("[pflda] Only {} chars for {font_key}, skipping", char_samples.len());
            return None; // not enough characters to be useful
        }

        let n_classes = char_samples.len();
        let out_dim = (n_classes - 1).min(feat_dim - 1).min(32);

        // Build char_map: class_index → character
        let char_map: Vec<char> = char_samples.iter().map(|(ch, _)| *ch).collect();

        // Compute per-class means and global mean
        let mut global_mean = vec![0.0f64; feat_dim];
        for (_, feats) in &char_samples {
            for f in feats {
                for j in 0..feat_dim { global_mean[j] += f[j] as f64; }
            }
        }
        for j in 0..feat_dim { global_mean[j] /= total_samples as f64; }

        let class_means: Vec<Vec<f64>> = char_samples.iter().map(|(_, feats)| {
            let mut mean = vec![0.0f64; feat_dim];
            for f in feats {
                for j in 0..feat_dim { mean[j] += f[j] as f64; }
            }
            let cnt = feats.len() as f64;
            for j in 0..feat_dim { mean[j] /= cnt; }
            mean
        }).collect();

        // Within-class scatter Sw
        let mut sw = vec![0.0f64; feat_dim * feat_dim];
        for (ci, (_, feats)) in char_samples.iter().enumerate() {
            let cm = &class_means[ci];
            for f in feats {
                for a in 0..feat_dim {
                    let da = f[a] as f64 - cm[a];
                    for b in a..feat_dim {
                        let db = f[b] as f64 - cm[b];
                        sw[a * feat_dim + b] += da * db;
                    }
                }
            }
        }
        // Mirror lower triangle
        for a in 0..feat_dim { for b in 0..a { sw[a * feat_dim + b] = sw[b * feat_dim + a]; } }
        for v in &mut sw { *v /= total_samples as f64; }

        // Regularize
        let trace: f64 = (0..feat_dim).map(|j| sw[j * feat_dim + j]).sum();
        let eps = (trace / feat_dim as f64) * 0.01 + 1e-6;
        for j in 0..feat_dim { sw[j * feat_dim + j] += eps; }

        // Cholesky: Sw = L L^T
        let mut l = vec![0.0f64; feat_dim * feat_dim];
        let mut chol_ok = true;
        for i in 0..feat_dim {
            for j in 0..=i {
                let mut sum = sw[i * feat_dim + j];
                for k in 0..j { sum -= l[i * feat_dim + k] * l[j * feat_dim + k]; }
                if i == j {
                    if sum <= 0.0 { chol_ok = false; break; }
                    l[i * feat_dim + j] = sum.sqrt();
                } else {
                    l[i * feat_dim + j] = sum / l[j * feat_dim + j];
                }
            }
            if !chol_ok { break; }
        }
        if !chol_ok { eprintln!("[pflda] Cholesky failed for {font_key}"); return None; }

        // L^{-1}
        let mut linv = vec![0.0f64; feat_dim * feat_dim];
        for j in 0..feat_dim {
            for i in 0..feat_dim {
                let mut sum = if i == j { 1.0 } else { 0.0 };
                for k in 0..i { sum -= l[i * feat_dim + k] * linv[k * feat_dim + j]; }
                linv[i * feat_dim + j] = sum / l[i * feat_dim + i];
            }
        }

        // Whiten class means
        let whitened_means: Vec<Vec<f64>> = class_means.iter().map(|cm| {
            let mut centered = vec![0.0f64; feat_dim];
            for j in 0..feat_dim { centered[j] = cm[j] - global_mean[j]; }
            let mut wm = vec![0.0f64; feat_dim];
            for i in 0..feat_dim {
                for j in 0..feat_dim {
                    wm[i] += linv[i * feat_dim + j] * centered[j];
                }
            }
            wm
        }).collect();

        // PCA on whitened means to get top eigenvectors
        let mut wm_mean = vec![0.0f64; feat_dim];
        for wm in &whitened_means { for j in 0..feat_dim { wm_mean[j] += wm[j]; } }
        for j in 0..feat_dim { wm_mean[j] /= n_classes as f64; }

        let mut cov = vec![0.0f64; feat_dim * feat_dim];
        for wm in &whitened_means {
            for a in 0..feat_dim {
                let da = wm[a] - wm_mean[a];
                for b in a..feat_dim {
                    let db = wm[b] - wm_mean[b];
                    cov[a * feat_dim + b] += da * db;
                }
            }
        }
        for a in 0..feat_dim { for b in 0..a { cov[a * feat_dim + b] = cov[b * feat_dim + a]; } }

        let actual_dim = out_dim.min(feat_dim);
        let eigvecs = crate::train::jacobi_eigen_top_k(&cov, feat_dim, actual_dim);

        // Final projection: P = eigvecs^T * L^{-1}
        let mut proj = vec![0.0f64; actual_dim * feat_dim];
        for d in 0..actual_dim {
            for j in 0..feat_dim {
                let mut sum = 0.0f64;
                for k in 0..feat_dim {
                    sum += eigvecs[d * feat_dim + k] * linv[k * feat_dim + j];
                }
                proj[d * feat_dim + j] = sum;
            }
        }

        // Scale calibration
        let target_dist = 0.03f64;
        let mut within_dists: Vec<f64> = Vec::new();
        {
            use rand::prelude::*;
            use rand::rngs::SmallRng;
            let mut rng = SmallRng::seed_from_u64(0x484f4743); // "HOGC"
            for (_, feats) in &char_samples {
                if feats.len() < 2 { continue; }
                let npairs = 5.min(feats.len() * (feats.len() - 1) / 2);
                for _ in 0..npairs {
                    let a = rng.gen_range(0..feats.len());
                    let b = loop { let x = rng.gen_range(0..feats.len()); if x != a { break x; } };
                    let mut d = 0.0f64;
                    for dim in 0..actual_dim {
                        let mut ea = 0.0f64;
                        let mut eb = 0.0f64;
                        for j in 0..feat_dim {
                            ea += proj[dim * feat_dim + j] * feats[a][j] as f64;
                            eb += proj[dim * feat_dim + j] * feats[b][j] as f64;
                        }
                        let diff = ea - eb;
                        d += diff * diff;
                    }
                    within_dists.push(d);
                }
            }
        }
        let scale = if !within_dists.is_empty() {
            within_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = within_dists[within_dists.len() / 2];
            if median > 1e-15 { (target_dist / median).sqrt() } else { 1.0 }
        } else { 1.0 };
        for v in &mut proj { *v *= scale; }

        let proj_f32: Vec<f32> = proj.iter().map(|&v| v as f32).collect();

        // Compute centroids in projected space
        let centroids: Vec<Vec<f32>> = class_means.iter().map(|cm| {
            let mut emb = vec![0.0f32; actual_dim];
            for d in 0..actual_dim {
                for j in 0..feat_dim {
                    emb[d] += proj_f32[d * feat_dim + j] * cm[j] as f32;
                }
            }
            emb
        }).collect();

        // Compute sigma_sq from within-class scatter: for each training
        // sample, project it and measure d² to its class centroid.
        // σ² = median of those distances — the typical noise around a centroid.
        let sigma_sq: f32 = {
            let mut within_dists: Vec<f32> = Vec::new();
            for (ci, (_, feats)) in char_samples.iter().enumerate() {
                let cent = &centroids[ci];
                for f in feats {
                    let mut emb = vec![0.0f32; actual_dim];
                    for d in 0..actual_dim {
                        for j in 0..feat_dim {
                            emb[d] += proj_f32[d * feat_dim + j] * f[j] as f32;
                        }
                    }
                    let d_sq: f32 = emb.iter().zip(cent.iter())
                        .map(|(&a, &b)| { let dd = a - b; dd * dd })
                        .sum();
                    within_dists.push(d_sq);
                }
            }
            if within_dists.is_empty() { 1.0 }
            else {
                within_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                within_dists[within_dists.len() / 2]
            }
        };

        eprintln!("[pflda] Training complete for {font_key}: {} classes, out_dim={actual_dim}, sigma_sq={sigma_sq:.6}", n_classes);
        Some(PerFontLda {
            projection: proj_f32,
            out_dim: actual_dim,
            feat_dim,
            centroids,
            sigma_sq,
            char_map,
            font_key: font_key.to_string(),
            catalog_hash: ctx.catalog_hash,
        })
    }

    /// Save to per-font cache file.
    fn save(&self) -> Result<(), String> {
        use std::io::{BufWriter, Write};
        let dir = Self::cache_dir();
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        let path = Self::cache_path(&self.font_key);
        let f = std::fs::File::create(&path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);

        // Magic + version
        w.write_all(b"PFHG").map_err(|e| e.to_string())?; // Per-Font HOG
        w.write_all(&3u32.to_le_bytes()).map_err(|e| e.to_string())?; // version 3 (HOG + glyph metrics)
        w.write_all(&self.catalog_hash.to_le_bytes()).map_err(|e| e.to_string())?;

        // Dimensions
        w.write_all(&(self.feat_dim as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        w.write_all(&(self.out_dim as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        let n_classes = self.char_map.len() as u32;
        w.write_all(&n_classes.to_le_bytes()).map_err(|e| e.to_string())?;

        // char_map
        for &ch in &self.char_map {
            w.write_all(&(ch as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        }

        // sigma_sq
        w.write_all(&self.sigma_sq.to_le_bytes()).map_err(|e| e.to_string())?;

        // projection: [out_dim × feat_dim]
        for &v in &self.projection {
            w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        // centroids: [n_classes × out_dim]
        for cent in &self.centroids {
            for &v in cent {
                w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }

    /// Load from a per-font cache file.  Returns None if stale or corrupt.
    fn load(path: &std::path::Path, font_key: &str, catalog_hash: u64) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 28 { return None; } // magic(4) + ver(4) + hash(8) + feat_dim(4) + dims(8)

        let mut pos = 0usize;

        // Magic
        if &data[pos..pos+4] != b"PFHG" { return None; }
        pos += 4;

        // Version
        let version = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
        if version != 3 { return None; }
        pos += 4;

        // Catalog hash
        let file_hash = u64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
        if file_hash != catalog_hash { return None; }
        pos += 8;

        // Dimensions
        let feat_dim = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        let out_dim = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        let n_classes = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;

        // char_map
        if data.len() < pos + n_classes * 4 { return None; }
        let mut char_map = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let cp = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
            pos += 4;
            char_map.push(char::from_u32(cp)?);
        }

        // sigma_sq
        if data.len() < pos + 4 { return None; }
        let sigma_sq = f32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
        pos += 4;

        // projection
        let proj_len = out_dim * feat_dim;
        if data.len() < pos + proj_len * 4 { return None; }
        let mut projection = Vec::with_capacity(proj_len);
        for _ in 0..proj_len {
            projection.push(f32::from_le_bytes(data[pos..pos+4].try_into().ok()?));
            pos += 4;
        }

        // centroids
        if data.len() < pos + n_classes * out_dim * 4 { return None; }
        let mut centroids = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let mut cent = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                cent.push(f32::from_le_bytes(data[pos..pos+4].try_into().ok()?));
                pos += 4;
            }
            centroids.push(cent);
        }

        Some(PerFontLda {
            projection,
            out_dim,
            feat_dim,
            centroids,
            sigma_sq,
            char_map,
            font_key: font_key.to_string(),
            catalog_hash,
        })
    }
}
