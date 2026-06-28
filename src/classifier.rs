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

use crate::char_index::{CharFeatures, FEAT_LEN};

// Re-export the Fisher weights so FisherClassifier can use them without
// making them pub in char_index.
use crate::char_index::FISHER_WEIGHTS;

use std::collections::HashMap;

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

    fn read_f32(&mut self) -> Result<f32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("truncated weight data".into());
        }
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
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
/// computed on first access from the stored FontVecStore centroids as a
/// fallback.
pub trait Classifier: Send + Sync {
    /// Return the top `k` font matches for a character crop.
    /// Returns `(font_id, probability)` sorted descending (highest = best).
    fn classify(&self, ch: char, query: &CharFeatures, k: usize) -> Vec<(usize, f32)>;

    /// Return calibrated posterior probabilities for all fonts, sorted
    /// descending by probability.  Probabilities sum to 1.
    ///
    /// Default implementation delegates to `classify(ch, query, font_count())`.
    fn probabilities(&self, ch: char, query: &CharFeatures) -> Vec<(usize, f32)> {
        self.classify(ch, query, self.font_count())
    }

    /// Posterior probability of a specific font given a query.
    /// Equivalent to finding `font_id` in `probabilities()`.
    ///
    /// Default calls `probabilities` and scans.
    fn probability(&self, ch: char, query: &CharFeatures, font_id: usize) -> Option<f32> {
        self.probabilities(ch, query).iter()
            .find(|(id, _)| *id == font_id)
            .map(|(_, p)| *p)
    }

    /// Short name for logging and cache invalidation.
    fn name(&self) -> &str;

    /// Number of distinct fonts loaded.
    fn font_count(&self) -> usize;

    /// Whether this classifier requires the raster image (not just feature vectors).
    /// When true, `rebuild_vecs` will re-render glyphs from the font cache before
    /// calling `add_font`.
    fn needs_raster(&self) -> bool { false }

    /// Feed a font's feature vector for a character into the classifier.
    /// Called once per (font_id, char) pair during index build.
    /// Default implementation is a no-op (for classifiers like MLP that
    /// don't use font vectors).
    fn add_font(&mut self, _font_id: usize, _ch: char, _features: &CharFeatures) {}
}

/// Convert `(font_id, sq_dist)` pairs to `(font_id, prob)` pairs using a
/// Gaussian kernel with bandwidth `sigma_sq`.  Mutates in place — no extra
/// allocation beyond the input vector.
///
/// `p_i = exp(-d_i / 2σ²) / Σ_j exp(-d_j / 2σ²)`
pub fn distances_to_probs(mut ranked: Vec<(usize, f32)>, sigma_sq: f32) -> Vec<(usize, f32)> {
    let inv2s = 1.0 / (2.0 * sigma_sq);
    // Numerically stable softmax: subtract max score before exp
    let max_score = ranked.iter().map(|(_, d)| -d * inv2s)
        .fold(f32::NEG_INFINITY, f32::max);
    // Convert distances → unnormalized probabilities in place
    let mut sum = 0.0f32;
    for (_, d) in &mut ranked {
        let e = (-*d * inv2s - max_score).exp();
        *d = e;
        sum += e;
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for (_, p) in &mut ranked {
        *p *= inv_sum;
    }
    ranked
}

// ---------------------------------------------------------------------------
// Helpers for embedding-based classifiers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two slices.
fn sq_euclid(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum()
}


/// Reusable storage for classifiers that do brute-force distance search
/// over embedded font vectors. Each embedding-based classifier composes
/// this internally.
pub(crate) struct FontVecStore {
    /// Per-char font vectors: char → [(font_id, embedded_vec)]
    vecs: HashMap<char, Vec<(usize, Vec<f32>)>>,
    /// Per-char font_id → index into vecs for O(1) lookup
    idx: HashMap<char, HashMap<usize, usize>>,
    count: usize,
    /// Per-char Gaussian bandwidth (median pairwise squared distance).
    /// Loaded from weights if available; otherwise computed lazily from
    /// stored centroids on first `sigma_sq()` call.
    sigma_sq_cache: std::sync::RwLock<HashMap<char, f32>>,
}

impl FontVecStore {
    fn new() -> Self {
        Self {
            vecs: HashMap::new(),
            idx: HashMap::new(),
            count: 0,
            sigma_sq_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    fn add(&mut self, font_id: usize, ch: char, embedded: Vec<f32>) {
        let v = self.vecs.entry(ch).or_default();
        let i = v.len();
        v.push((font_id, embedded));
        self.idx.entry(ch).or_default().insert(font_id, i);
        if font_id >= self.count { self.count = font_id + 1; }
    }

    /// Return the top `k` fonts by probability (highest first).
    fn classify(&self, ch: char, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut probs = self.probabilities(ch, query);
        probs.truncate(k);
        probs
    }

    /// Return all fonts with calibrated probabilities, sorted descending.
    fn probabilities(&self, ch: char, query: &[f32]) -> Vec<(usize, f32)> {
        let points = match self.vecs.get(&ch) {
            Some(p) if !p.is_empty() => p,
            _ => return Vec::new(),
        };
        let dists: Vec<(usize, f32)> = points.iter()
            .map(|(id, stored)| (*id, sq_euclid(query, stored)))
            .collect();
        let sigma = match self.sigma_sq(ch) {
            Some(s) if s > 1e-30 => s,
            _ => {
                // No σ² — fall back to uniform
                let p = 1.0 / dists.len() as f32;
                return dists.into_iter().map(|(id, _)| (id, p)).collect();
            }
        };
        let mut probs = distances_to_probs(dists, sigma);
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs
    }

    /// Probability of a specific font for a character.
    fn probability(&self, ch: char, query: &[f32], font_id: usize) -> Option<f32> {
        self.probabilities(ch, query).into_iter()
            .find(|(id, _)| *id == font_id)
            .map(|(_, p)| p)
    }

    fn font_count(&self) -> usize { self.count }

    /// Set a pre-computed σ² for a character (from training-time weights).
    fn set_sigma_sq(&self, ch: char, val: f32) {
        self.sigma_sq_cache.write().unwrap().insert(ch, val);
    }

    /// Get σ² for a character.  Returns the training-time value if set,
    /// otherwise computes it from stored centroids (median pairwise squared
    /// distance) and caches the result.
    fn sigma_sq(&self, ch: char) -> Option<f32> {
        // Fast path: already cached
        if let Some(&s) = self.sigma_sq_cache.read().unwrap().get(&ch) {
            return Some(s);
        }
        // Compute from stored centroids
        let points = self.vecs.get(&ch)?;
        let n = points.len();
        if n < 2 { return None; }
        let mut dists: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(sq_euclid(&points[i].1, &points[j].1));
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = dists[dists.len() / 2];
        if median > 1e-30 {
            self.sigma_sq_cache.write().unwrap().insert(ch, median);
            Some(median)
        } else {
            None
        }
    }
}



// ---------------------------------------------------------------------------
// EmbeddingClassifier — shared Classifier impl for embed-then-store classifiers
// ---------------------------------------------------------------------------

/// Trait for the embedding step: converts raw features into a classifier-specific vector.
pub trait Embedder: Send + Sync {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32>;
    fn name(&self) -> &str;
}

/// Generic classifier that embeds features via an [`Embedder`] then searches a [`FontVecStore`].
/// Collapses the identical classify/distance/add_font boilerplate across Fisher, Triplet,
/// GlobalTriplet, PerCharFisher, Mahalanobis, and LDA.
pub struct EmbeddingClassifier {
    store: FontVecStore,
    embedder: Box<dyn Embedder>,
}

impl EmbeddingClassifier {
    pub fn new(embedder: Box<dyn Embedder>) -> Self {
        Self { store: FontVecStore::new(), embedder }
    }
}

impl Classifier for EmbeddingClassifier {
    fn classify(&self, ch: char, query: &CharFeatures, k: usize) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(ch, query);
        self.store.classify(ch, &q, k)
    }

    fn probabilities(&self, ch: char, query: &CharFeatures) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(ch, query);
        self.store.probabilities(ch, &q)
    }

    fn probability(&self, ch: char, query: &CharFeatures, font_id: usize) -> Option<f32> {
        let q = self.embedder.embed(ch, query);
        self.store.probability(ch, &q, font_id)
    }

    fn name(&self) -> &str {
        self.embedder.name()
    }

    fn font_count(&self) -> usize {
        self.store.font_count()
    }

    fn add_font(&mut self, font_id: usize, ch: char, features: &CharFeatures) {
        let embedded = self.embedder.embed(ch, features);
        self.store.add(font_id, ch, embedded);
    }
}

// ---------------------------------------------------------------------------
// Fisher (original)
// ---------------------------------------------------------------------------

/// Diagonal Fisher-weighted Euclidean distance — the original classifier.
///
/// Each raw feature dimension is multiplied by its learned Fisher weight
/// (√(between-font variance / within-font variance), normalised to sum = 1).
/// Distance is plain squared Euclidean in the weighted space.
///
/// Fisher stores weighted font vectors internally and performs brute-force
/// nearest-neighbor search.
/// Apply global Fisher weighting to raw feature vector.
fn fisher_embed(raw: &[f32]) -> Vec<f32> {
    let mut v = vec![0.0f32; FEAT_LEN];
    for i in 0..FEAT_LEN {
        v[i] = raw[i] * FISHER_WEIGHTS[i];
    }
    v
}

struct FisherEmbedder;

impl Embedder for FisherEmbedder {
    fn embed(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        fisher_embed(&features.as_slice())
    }
    fn name(&self) -> &str { "fisher" }
}

/// Create a new FisherClassifier.
pub fn new_fisher() -> EmbeddingClassifier {
    EmbeddingClassifier::new(Box::new(FisherEmbedder))
}// ---------------------------------------------------------------------------
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
    nets: HashMap<char, GlyphNet>,
}

impl TripletClassifier {
    /// Load per-glyph weights from a binary file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open triplet weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        if data.len() < 12 {
            return Err(format!(
                "triplet weights file too small ({} bytes, need ≥12 for header)",
                data.len()
            ));
        }

        if &data[0..4] != b"TRIP" {
            return Err(format!(
                "bad magic in triplet weights (expected TRIP, got {:?})",
                &data[0..4]
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported triplet weights version {version}"));
        }

        let n_chars = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let expected_bytes = 12 + n_chars * (4 + PARAMS_PER_CHAR * 4);
        if data.len() != expected_bytes {
            return Err(format!(
                "triplet weights: {} bytes, expected {} ({n_chars} chars × {} params + header)",
                data.len(),
                expected_bytes,
                PARAMS_PER_CHAR,
            ));
        }

        let mut r = BinaryReader::new(&data, 12);

        let mut nets = HashMap::with_capacity(n_chars);
        for _ in 0..n_chars {
            let codepoint = r.read_u32()?;
            let ch = char::from_u32(codepoint).ok_or_else(|| {
                format!("invalid codepoint U+{codepoint:04X} in triplet weights")
            })?;

            let net = GlyphNet {
                fc1: InferenceLinear {
                    rows: L1_IN, cols: L1_OUT,
                    w: r.read_f32s(L1_IN * L1_OUT)?, b: r.read_f32s(L1_OUT)?,
                },
                fc2: InferenceLinear {
                    rows: L1_OUT, cols: L2_OUT,
                    w: r.read_f32s(L1_OUT * L2_OUT)?, b: r.read_f32s(L2_OUT)?,
                },
                fc3: InferenceLinear {
                    rows: L2_OUT, cols: L3_OUT,
                    w: r.read_f32s(L2_OUT * L3_OUT)?, b: r.read_f32s(L3_OUT)?,
                },
            };
            nets.insert(ch, net);
        }

        Ok(Self { nets })
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        if let Some(net) = self.nets.get(&ch) {
            net.forward(&features.as_slice())
        } else {
            let f = fisher_embed(&features.as_slice());
            f[..L3_OUT.min(f.len())].to_vec()
        }
    }
}

impl Embedder for TripletClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        TripletClassifier::embed(self, ch, features)
    }
    fn name(&self) -> &str { "triplet" }
}

/// Load a triplet-embedding classifier from weights file.
pub fn load_triplet(path: &std::path::Path) -> Result<EmbeddingClassifier, String> {
    let tc = TripletClassifier::load(path)?;
    Ok(EmbeddingClassifier::new(Box::new(tc)))
}

// ---------------------------------------------------------------------------
// Global triplet network (single model, all characters)
// ---------------------------------------------------------------------------

/// Single global triplet network classifier.
///
/// Unlike [`TripletClassifier`] which loads one MLP per character, this
/// variant uses a single MLP for ALL characters.  The `embed()` method
/// ignores the `ch` parameter — the same network processes every glyph.
///
/// Because the embedding space captures both font *and* character identity,
/// nearest-neighbor search recovers both which character and which font
/// match best.
///
/// Binary format (magic `b"TRPG"`):
/// ```text
/// magic:   b"TRPG" (4 bytes)
/// version: u32 LE (1)
/// W1: L1_IN×128 f32 LE, b1: 128 f32 LE
/// W2: 128×64 f32 LE,  b2: 64 f32 LE
/// W3: 64×32 f32 LE,   b3: 32 f32 LE
/// Total: 8 + 23 264 × 4 = 93 064 bytes
/// ```
pub struct GlobalTripletClassifier {
    net: GlyphNet,
}

/// Header size for TRPG format: magic (4) + version (4) = 8 bytes.
/// Then PARAMS_PER_CHAR × 4 bytes of weights.
const GLOBAL_HEADER: usize = 8;

impl GlobalTripletClassifier {
    /// Load global weights from a binary file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open global-triplet weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        let expected_bytes = GLOBAL_HEADER + PARAMS_PER_CHAR * 4;
        if data.len() != expected_bytes {
            return Err(format!(
                "global-triplet weights: {} bytes, expected {} ({} header + {} params × 4)",
                data.len(),
                expected_bytes,
                GLOBAL_HEADER,
                PARAMS_PER_CHAR,
            ));
        }

        if &data[0..4] != b"TRPG" {
            return Err(format!(
                "bad magic in global-triplet weights (expected TRPG, got {:?})",
                &data[0..4]
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported global-triplet weights version {version}"));
        }

        let mut r = BinaryReader::new(&data, GLOBAL_HEADER);

        let net = GlyphNet {
            fc1: InferenceLinear {
                rows: L1_IN, cols: L1_OUT,
                w: r.read_f32s(L1_IN * L1_OUT)?, b: r.read_f32s(L1_OUT)?,
            },
            fc2: InferenceLinear {
                rows: L1_OUT, cols: L2_OUT,
                w: r.read_f32s(L1_OUT * L2_OUT)?, b: r.read_f32s(L2_OUT)?,
            },
            fc3: InferenceLinear {
                rows: L2_OUT, cols: L3_OUT,
                w: r.read_f32s(L2_OUT * L3_OUT)?, b: r.read_f32s(L3_OUT)?,
            },
        };

        Ok(Self { net })
    }
}

impl Embedder for GlobalTripletClassifier {
    fn embed(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        self.net.forward(&features.as_slice())
    }
    fn name(&self) -> &str { "global_triplet" }
}

/// Load a global-triplet-embedding classifier from weights file.
pub fn load_global_triplet(path: &std::path::Path) -> Result<EmbeddingClassifier, String> {
    let gt = GlobalTripletClassifier::load(path)?;
    Ok(EmbeddingClassifier::new(Box::new(gt)))
}

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
    weights: HashMap<char, [f32; FEAT_LEN]>,
}

impl PerCharFisherClassifier {
    /// Load per-character Fisher weights from a FISH binary file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open per-char Fisher weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        if data.len() < 16 {
            return Err("per-char Fisher file too small".into());
        }
        if &data[0..4] != b"FISH" {
            return Err(format!("bad magic (expected FISH, got {:?})", &data[0..4]));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported version {version}"));
        }
        let n_chars = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let feat_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        if feat_len != FEAT_LEN {
            return Err(format!("feat_len mismatch: file has {feat_len}, expected {FEAT_LEN}"));
        }

        let per_char_bytes = 4 + FEAT_LEN * 4;
        let expected = 16 + n_chars * per_char_bytes;
        if data.len() != expected {
            return Err(format!("size mismatch: {} bytes, expected {expected}", data.len()));
        }

        let mut weights = HashMap::with_capacity(n_chars);
        let mut r = BinaryReader::new(&data, 16);
        for _ in 0..n_chars {
            let cp = r.read_u32()?;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let wv = r.read_f32s(FEAT_LEN)?;
            let mut w = [0.0f32; FEAT_LEN];
            w.copy_from_slice(&wv);

            // Normalize: cap f32::MAX, take sqrt, scale to sum=1
            let max_finite = w.iter().filter(|v| v.is_finite()).copied()
                .fold(0.0f32, f32::max);
            for v in &mut w {
                if !v.is_finite() || *v > max_finite {
                    *v = max_finite;
                }
                *v = v.sqrt();
            }
            let sum: f32 = w.iter().sum();
            if sum > 1e-12 {
                for v in &mut w { *v /= sum; }
            }

            weights.insert(ch, w);
        }

        Ok(Self { weights })
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        let w = self.weights.get(&ch)
            .map(|w| w.as_slice())
            .unwrap_or(&FISHER_WEIGHTS);
        let mut v = vec![0.0f32; FEAT_LEN];
        for i in 0..FEAT_LEN {
            v[i] = raw[i] * w[i];
        }
        v
    }
}

impl Embedder for PerCharFisherClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        PerCharFisherClassifier::embed(self, ch, features)
    }
    fn name(&self) -> &str { "per_char_fisher" }
}

/// Load a per-char Fisher classifier from weights file.
pub fn load_per_char_fisher(path: &std::path::Path) -> Result<EmbeddingClassifier, String> {
    let pcf = PerCharFisherClassifier::load(path)?;
    Ok(EmbeddingClassifier::new(Box::new(pcf)))
}

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
/// Binary format (magic `b"MAHA"`):
/// ```text
/// magic:    b"MAHA" (4 bytes)
/// version:  u32 LE (1)
/// n_chars:  u32 LE
/// feat_len: u32 LE
/// Per character (repeated n_chars times):
///   char_code: u32 LE
///   L_inv:     [f32; feat_len * feat_len] LE (row-major)
/// ```
pub struct MahalanobisClassifier {
    transforms: HashMap<char, Vec<f32>>, // L_inv per char, FEAT_LEN × FEAT_LEN row-major
}

impl MahalanobisClassifier {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open Mahalanobis weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        if data.len() < 16 {
            return Err("Mahalanobis file too small".into());
        }
        if &data[0..4] != b"MAHA" {
            return Err(format!("bad magic (expected MAHA, got {:?})", &data[0..4]));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported version {version}"));
        }
        let n_chars = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let feat_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        if feat_len != FEAT_LEN {
            return Err(format!("feat_len mismatch: {feat_len} vs {FEAT_LEN}"));
        }

        let per_char = 4 + FEAT_LEN * FEAT_LEN * 4;
        let expected = 16 + n_chars * per_char;
        if data.len() != expected {
            return Err(format!("size mismatch: {} bytes, expected {expected}", data.len()));
        }

        let mut transforms = HashMap::with_capacity(n_chars);
        let mut r = BinaryReader::new(&data, 16);
        for _ in 0..n_chars {
            let cp = r.read_u32()?;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let linv = r.read_f32s(FEAT_LEN * FEAT_LEN)?;
            transforms.insert(ch, linv);
        }

        Ok(Self { transforms })
    }

    /// Apply L_inv transform: y = L_inv * x  (square FEAT_LEN×FEAT_LEN matrix multiply)
    fn apply_transform(linv: &[f32], x: &[f32]) -> Vec<f32> {
        // Special case of dense_project with out_dim = FEAT_LEN
        dense_project(FEAT_LEN, linv, x)
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some(linv) = self.transforms.get(&ch) {
            Self::apply_transform(linv, &raw)
        } else {
            fisher_embed(&raw)
        }
    }
}

impl Embedder for MahalanobisClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        MahalanobisClassifier::embed(self, ch, features)
    }
    fn name(&self) -> &str { "mahalanobis" }
}

/// Load a Mahalanobis classifier from weights file.
pub fn load_mahalanobis(path: &std::path::Path) -> Result<EmbeddingClassifier, String> {
    let m = MahalanobisClassifier::load(path)?;
    Ok(EmbeddingClassifier::new(Box::new(m)))
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
/// Binary format (magic `b"LDAC"`):
/// ```text
/// magic:    b"LDAC" (4 bytes)
/// version:  u32 LE (1)
/// n_chars:  u32 LE
/// Per character:
///   char_code: u32 LE
///   out_dim:   u32 LE (number of projection dimensions)
///   proj:      [f32; out_dim * FEAT_LEN] LE (row-major, each row = one projection direction)
/// ```
pub struct LdaClassifier {
    projections: HashMap<char, (usize, Vec<f32>)>, // (out_dim, proj matrix)
    /// Per-character σ² from training (LDAC v2+).  Empty for v1 files;
    /// FontVecStore will compute from centroids as fallback.
    pub(crate) sigma_sq: HashMap<char, f32>,
}

impl LdaClassifier {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open LDA weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        let result = Self::from_bytes(&data)?;
        Ok(result)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {

        if data.len() < 12 {
            return Err("LDA file too small".into());
        }
        if &data[0..4] != b"LDAC" {
            return Err(format!("bad magic (expected LDAC, got {:?})", &data[0..4]));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 1 && version != 2 {
            return Err(format!("unsupported version {version}"));
        }
        let n_chars = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

        let mut projections = HashMap::with_capacity(n_chars);
        let mut sigma_sq = HashMap::with_capacity(n_chars);
        let mut r = BinaryReader::new(&data, 12);
        for _ in 0..n_chars {
            let cp = r.read_u32()?;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let out_dim = r.read_u32()? as usize;
            let proj = r.read_f32s(out_dim * FEAT_LEN)?;
            if version >= 2 {
                let s = r.read_f32()?;
                if s > 1e-30 {
                    sigma_sq.insert(ch, s);
                }
            }
            projections.insert(ch, (out_dim, proj));
        }

        let dims: Vec<usize> = projections.values().map(|(d, _)| *d).collect();
        let _max_dim = dims.iter().max().copied().unwrap_or(0);
        Ok(Self { projections, sigma_sq })
    }

    fn project(out_dim: usize, proj: &[f32], x: &[f32]) -> Vec<f32> {
        dense_project(out_dim, proj, x)
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some((out_dim, proj)) = self.projections.get(&ch) {
            Self::project(*out_dim, proj, &raw)
        } else {
            fisher_embed(&raw)
        }
    }

}

impl Embedder for LdaClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        LdaClassifier::embed(self, ch, features)
    }
    fn name(&self) -> &str { "lda" }
}

/// Load an LDA classifier from weights file.
/// If the LDAC file contains per-character σ² (v2), seeds them into
/// the FontVecStore so `sigma_sq()` returns training-time values.
pub fn load_lda(path: &std::path::Path) -> Result<EmbeddingClassifier, String> {
    let lda = LdaClassifier::load(path)?;
    let sigmas = lda.sigma_sq.clone();
    let ec = EmbeddingClassifier::new(Box::new(lda));
    // Pre-seed σ² values — they become effective once FontVecStore
    // is populated during index build (the store still serves as the
    // fallback if no training-time σ² is available for a character).
    for (ch, s) in sigmas {
        ec.store.set_sigma_sq(ch, s);
    }
    Ok(ec)
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
/// version:  u32 LE (1)
/// n_chars:  u32 LE
/// Per character:
///   char_code: u32 LE
///   k:         u32 LE (number of classes)
///   class_map: [u32; k] LE (class_index → font_id)
///   W1: [f32; 100×256] LE, b1: [f32; 256] LE
///   W2: [f32; 256×128] LE, b2: [f32; 128] LE
///   W3: [f32; 128×k]   LE, b3: [f32; k]   LE
/// ```
pub struct MlpClassifier {
    nets: HashMap<char, MlpCharNet>,
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
        if version != 1 {
            return Err(format!("unsupported MLP version {version}"));
        }
        let n_chars = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

        let mut nets = HashMap::with_capacity(n_chars);
        let mut r = BinaryReader::new(data, 12);

        for _ in 0..n_chars {
            let cp = r.read_u32()?;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let k = r.read_u32()? as usize;
            if k == 0 {
                return Err(format!("char '{}': zero classes", ch));
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

            nets.insert(ch, MlpCharNet {
                fc1: InferenceLinear { rows: FEAT_LEN, cols: MLP_H1, w: w1, b: b1 },
                fc2: InferenceLinear { rows: MLP_H1, cols: MLP_H2, w: w2, b: b2 },
                fc3: InferenceLinear { rows: MLP_H2, cols: k, w: w3, b: b3 },
                class_map,
            });
        }

        Ok(Self { nets })
    }

    /// Total number of unique font IDs across all character nets.
    fn count_fonts(&self) -> usize {
        let mut ids = std::collections::HashSet::new();
        for net in self.nets.values() {
            for &fid in &net.class_map {
                ids.insert(fid);
            }
        }
        ids.len()
    }
}

impl MlpClassifier {
    /// Compute softmax probabilities for a character query, returning (net, probs) if the char has a net.
    fn softmax_probs(&self, ch: char, query: &CharFeatures) -> Option<(&MlpCharNet, Vec<f32>)> {
        let net = self.nets.get(&ch)?;
        let logits = net.forward(&query.as_slice());
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum_exp; }
        Some((net, probs))
    }
}

impl Classifier for MlpClassifier {
    fn classify(&self, ch: char, query: &CharFeatures, k: usize) -> Vec<(usize, f32)> {
        if let Some((net, probs)) = self.softmax_probs(ch, query) {
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
    fn probabilities(&self, ch: char, query: &CharFeatures) -> Vec<(usize, f32)> {
        if let Some((net, probs)) = self.softmax_probs(ch, query) {
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

    fn probability(&self, ch: char, query: &CharFeatures, font_id: usize) -> Option<f32> {
        let (net, probs) = self.softmax_probs(ch, query)?;
        for (ci, &fid) in net.class_map.iter().enumerate() {
            if fid as usize == font_id {
                return Some(probs[ci]);
            }
        }
        None
    }

    fn name(&self) -> &str {
        "mlp"
    }

    fn font_count(&self) -> usize {
        self.count_fonts()
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
    fn classify(&self, ch: char, query: &CharFeatures, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self.probabilities(ch, query);
        probs.truncate(k);
        probs
    }

    /// Fusion probabilities via weighted geometric mean of child posteriors.
    ///
    /// For each font, computes `exp(Σ w_i * ln(p_i)) / Z` where `p_i` is
    /// child i's probability and `w_i` is its normalized weight.  This is
    /// equivalent to `(∏ p_i^w_i) / Z`, the weighted geometric mean of
    /// individual posteriors, renormalized.
    fn probabilities(&self, ch: char, query: &CharFeatures) -> Vec<(usize, f32)> {
        // Collect child probability distributions
        let child_probs: Vec<(f32, HashMap<usize, f32>)> = self.children.iter()
            .map(|(weight, child)| {
                let probs = child.probabilities(ch, query);
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

    fn font_count(&self) -> usize {
        self.children.iter().map(|(_, c)| c.font_count()).max().unwrap_or(0)
    }

    fn add_font(&mut self, font_id: usize, ch: char, features: &CharFeatures) {
        for (_, child) in &mut self.children {
            child.add_font(font_id, ch, features);
        }
    }
}
