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


use crate::features::{CharFeatures, FEAT_LEN};

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
#[allow(dead_code)]
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

    /// Map a font_id back to its name.  Returns None for invalid ids.
    fn font_name(&self, font_id: usize) -> Option<&str>;

    /// Look up a font_id by name.  Returns None if the name is unknown.
    fn font_id(&self, name: &str) -> Option<usize> {
        (0..self.font_count()).find(|&id| self.font_name(id).map_or(false, |n| n == name))
    }

    /// Feed a font's feature vector for a character into the classifier.
    /// Called once per (font_id, char) pair during index build.
    /// Default implementation is a no-op (for classifiers like MLP that
    /// don't use font vectors).
    fn add_font(&mut self, _font_id: usize, _font_name: &str, _ch: char, _features: &CharFeatures) {}
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
// CharModel — per-character complete model state
// ---------------------------------------------------------------------------

/// Complete model state for a single character.  Co-locates the classifier-
/// specific weights (Fisher scores, LDA projection, Mahalanobis L_inv, …)
/// with the embedded font centroids and the probability-calibration σ².
///
/// Replaces the old split where weights lived in the Embedder and centroids
/// lived in a separate FontVecStore.
pub struct CharModel {
    /// Classifier-specific weights as a flat f32 blob.
    /// Interpretation depends on the classifier type (magic byte in .bin header).
    pub weights: Vec<f32>,
    /// Embedded font centroids: (font_id, embedded_vec).
    pub centroids: Vec<(u32, Vec<f32>)>,
    /// Gaussian bandwidth for probability calibration (median pairwise
    /// squared distance among centroids).  0.0 means not yet computed.
    pub sigma_sq: f32,
}

impl CharModel {
    /// Probability of each font given a query vector, sorted descending.
    pub fn probabilities(&self, query: &[f32]) -> Vec<(u32, f32)> {
        if self.centroids.is_empty() { return Vec::new(); }
        let dists: Vec<(u32, f32)> = self.centroids.iter()
            .map(|(id, stored)| (*id, sq_euclid(query, stored)))
            .collect();
        let sigma = if self.sigma_sq > 1e-30 {
            self.sigma_sq
        } else {
            let p = 1.0 / dists.len() as f32;
            return dists.into_iter().map(|(id, _)| (id, p)).collect();
        };
        let inv2s = 1.0 / (2.0 * sigma);
        let mut probs: Vec<(u32, f32)> = {
            // Standard Gaussian kernel with softmax max-subtraction for numerical stability
            let min_d = dists.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
            let raw: Vec<f32> = dists.iter().map(|(_, d)| (-(d - min_d) * inv2s).exp()).collect();
            let sum: f32 = raw.iter().sum();
            if sum < 1e-30 {
                let p = 1.0 / dists.len() as f32;
                dists.into_iter().map(|(id, _)| (id, p)).collect()
            } else {
                dists.iter().zip(raw.iter()).map(|((id, _), &r)| (*id, r / sum)).collect()
            }
        };
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs
    }

    /// Top-k fonts by probability.
    pub fn classify(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut probs = self.probabilities(query);
        probs.truncate(k);
        probs
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
}

/// Per-character classifier with co-located weights, centroids, and σ².
/// Font names are shared across all characters; centroids reference fonts
/// by font_id (index into `font_names` / catalog).
pub struct PerCharModel {
    pub chars: HashMap<char, CharModel>,
    /// font_id → font_key.  Shared across all characters.
    pub font_names: Vec<String>,
    /// Catalog hash at training time.  Used to reject stale .bin files
    /// when the font catalog changes.
    pub catalog_hash: u64,
}

impl PerCharModel {
    pub fn new(catalog_hash: u64) -> Self {
        Self { chars: HashMap::new(), font_names: Vec::new(), catalog_hash }
    }

    /// Serialize to the unified per-char .bin format.
    ///
    /// ```text
    /// magic:         [u8; 4]       (caller-chosen, e.g. b"FISH")
    /// version:       u32 le
    /// catalog_hash:  u64 le
    /// n_fonts:       u32 le
    /// per font:      name_len: u32 le, name: [u8]
    /// n_chars:       u32 le
    /// per char:
    ///   codepoint:   u32 le
    ///   n_weights:   u32 le
    ///   weights:     [f32; n_weights] le
    ///   n_centroids: u32 le
    ///   per centroid:
    ///     font_id:   u32 le
    ///     vec_len:   u32 le
    ///     vec:       [f32; vec_len] le
    ///   sigma_sq:    f32 le
    /// ```
    pub fn write_bin(
        &self,
        w: &mut dyn std::io::Write,
        magic: &[u8; 4],
        version: u32,
    ) -> std::io::Result<()> {
        w.write_all(magic)?;
        w.write_all(&version.to_le_bytes())?;
        w.write_all(&self.catalog_hash.to_le_bytes())?;

        // Font names
        w.write_all(&(self.font_names.len() as u32).to_le_bytes())?;
        for name in &self.font_names {
            let b = name.as_bytes();
            w.write_all(&(b.len() as u32).to_le_bytes())?;
            w.write_all(b)?;
        }

        // Per-char models
        w.write_all(&(self.chars.len() as u32).to_le_bytes())?;
        for (&ch, model) in &self.chars {
            w.write_all(&(ch as u32).to_le_bytes())?;

            // Weights
            w.write_all(&(model.weights.len() as u32).to_le_bytes())?;
            for &v in &model.weights { w.write_all(&v.to_le_bytes())?; }

            // Centroids
            w.write_all(&(model.centroids.len() as u32).to_le_bytes())?;
            for (font_id, vec) in &model.centroids {
                w.write_all(&font_id.to_le_bytes())?;
                w.write_all(&(vec.len() as u32).to_le_bytes())?;
                for &v in vec { w.write_all(&v.to_le_bytes())?; }
            }

            // σ²
            w.write_all(&model.sigma_sq.to_le_bytes())?;
        }
        Ok(())
    }

    /// Deserialize from the unified per-char .bin format.
    /// Validates magic and catalog_hash.  Returns an error if magic doesn't
    /// match `expected_magic` or if the catalog hash doesn't match
    /// `expected_catalog_hash` (when Some).
    pub fn read_bin(
        data: &[u8],
        expected_magic: &[u8; 4],
        expected_catalog_hash: Option<u64>,
    ) -> Result<Self, String> {
        if data.len() < 16 {
            return Err("file too small".into());
        }
        if &data[0..4] != expected_magic {
            return Err(format!("bad magic (expected {:?}, got {:?})", expected_magic, &data[0..4]));
        }
        let _version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let catalog_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());

        if let Some(expected) = expected_catalog_hash {
            if catalog_hash != expected {
                return Err(format!(
                    "stale classifier: catalog_hash {catalog_hash:#x} != current {expected:#x}"
                ));
            }
        }

        let mut pos = 16;

        // Font names
        let n_fonts = read_u32(data, &mut pos)? as usize;
        let mut font_names = Vec::with_capacity(n_fonts);
        for _ in 0..n_fonts {
            let len = read_u32(data, &mut pos)? as usize;
            if pos + len > data.len() { return Err("truncated font name".into()); }
            let name = std::str::from_utf8(&data[pos..pos + len])
                .map_err(|e| format!("bad font name UTF-8: {e}"))?
                .to_string();
            pos += len;
            font_names.push(name);
        }

        // Per-char models
        let n_chars = read_u32(data, &mut pos)? as usize;
        let mut chars = HashMap::with_capacity(n_chars);
        for _ in 0..n_chars {
            let cp = read_u32(data, &mut pos)?;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;

            // Weights
            let n_weights = read_u32(data, &mut pos)? as usize;
            let weights = read_f32s(data, &mut pos, n_weights)?;

            // Centroids
            let n_centroids = read_u32(data, &mut pos)? as usize;
            let mut centroids = Vec::with_capacity(n_centroids);
            for _ in 0..n_centroids {
                let font_id = read_u32(data, &mut pos)?;
                let vec_len = read_u32(data, &mut pos)? as usize;
                let vec = read_f32s(data, &mut pos, vec_len)?;
                centroids.push((font_id, vec));
            }

            // σ²
            if pos + 4 > data.len() { return Err("truncated sigma_sq".into()); }
            let sigma_sq = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;

            chars.insert(ch, CharModel { weights, centroids, sigma_sq });
        }

        Ok(Self { chars, font_names, catalog_hash })
    }
}

/// Read a u32 LE from `data` at `*pos`, advancing `*pos`.
fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() { return Err("truncated u32".into()); }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

/// Read `n` f32 LE values from `data` at `*pos`, advancing `*pos`.
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
/// Legacy font vector store — only retained for deserializing FVST blobs
/// in GlobalTriplet v2 files. New classifiers use PerCharModel directly.
pub(crate) struct FontVecStore {
    /// Per-char font vectors: char → [(font_id, embedded_vec)]
    pub(crate) vecs: HashMap<char, Vec<(usize, Vec<f32>)>>,
    /// font_id → font name
    pub(crate) font_names: Vec<String>,
}

impl FontVecStore {
    /// Deserialize a font vector store from a reader (FVST format).
    pub fn read_from(r: &mut dyn std::io::Read) -> Result<Self, String> {
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4).map_err(|e| format!("read FVST magic: {e}"))?;
        if &b4 != b"FVST" {
            return Err(format!("bad FontVecStore magic: {:?}", &b4[..]));
        }
        // Font names
        r.read_exact(&mut b4).map_err(|e| format!("read n_fonts: {e}"))?;
        let n_fonts = u32::from_le_bytes(b4) as usize;
        let mut font_names = Vec::with_capacity(n_fonts);
        for _ in 0..n_fonts {
            r.read_exact(&mut b4).map_err(|e| format!("read name len: {e}"))?;
            let len = u32::from_le_bytes(b4) as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf).map_err(|e| format!("read name: {e}"))?;
            font_names.push(String::from_utf8(buf).map_err(|e| format!("bad font name: {e}"))?);
        }
        // Char vectors
        r.read_exact(&mut b4).map_err(|e| format!("read n_chars: {e}"))?;
        let n_chars = u32::from_le_bytes(b4) as usize;
        let mut vecs: HashMap<char, Vec<(usize, Vec<f32>)>> = HashMap::with_capacity(n_chars);
        for _ in 0..n_chars {
            r.read_exact(&mut b4).map_err(|e| format!("read cp: {e}"))?;
            let ch = char::from_u32(u32::from_le_bytes(b4))
                .ok_or_else(|| "invalid codepoint".to_string())?;
            r.read_exact(&mut b4).map_err(|e| format!("read n_entries: {e}"))?;
            let n_entries = u32::from_le_bytes(b4) as usize;
            let mut cv = Vec::with_capacity(n_entries);
            for _ in 0..n_entries {
                r.read_exact(&mut b4).map_err(|e| format!("read fid: {e}"))?;
                let fid = u32::from_le_bytes(b4) as usize;
                r.read_exact(&mut b4).map_err(|e| format!("read vlen: {e}"))?;
                let vlen = u32::from_le_bytes(b4) as usize;
                let mut v = vec![0.0f32; vlen];
                for x in &mut v {
                    r.read_exact(&mut b4).map_err(|e| format!("read val: {e}"))?;
                    *x = f32::from_le_bytes(b4);
                }
                cv.push((fid, v));
            }
            vecs.insert(ch, cv);
        }
        // Sigma sq (read and discard — no longer used)
        r.read_exact(&mut b4).map_err(|e| format!("read sigma count: {e}"))?;
        let n_sigma = u32::from_le_bytes(b4) as usize;
        for _ in 0..n_sigma {
            r.read_exact(&mut b4).map_err(|_| "read sigma cp".to_string())?;
            r.read_exact(&mut b4).map_err(|_| "read sigma val".to_string())?;
        }
        Ok(Self { vecs, font_names })
    }
}



// ---------------------------------------------------------------------------
// EmbeddingClassifier — shared Classifier impl for embed-then-store classifiers
// ---------------------------------------------------------------------------

/// Trait for the embedding step: converts raw features into a classifier-specific vector.
#[allow(dead_code)]
pub trait Embedder: Send + Sync {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32>;
    fn name(&self) -> &str;
}

/// Generic classifier that embeds features via an [`Embedder`] then searches
/// pre-computed centroids stored in a [`PerCharModel`].
pub struct EmbeddingClassifier {
    model: PerCharModel,
    embedder: Box<dyn Embedder>,
}

impl Classifier for EmbeddingClassifier {
    fn classify(&self, ch: char, query: &CharFeatures, k: usize) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(ch, query);
        if let Some(cm) = self.model.chars.get(&ch) {
            cm.classify(&q, k).into_iter().map(|(id, p)| (id as usize, p)).collect()
        } else {
            Vec::new()
        }
    }

    fn probabilities(&self, ch: char, query: &CharFeatures) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(ch, query);
        if let Some(cm) = self.model.chars.get(&ch) {
            cm.probabilities(&q).into_iter().map(|(id, p)| (id as usize, p)).collect()
        } else {
            Vec::new()
        }
    }

    fn probability(&self, ch: char, query: &CharFeatures, font_id: usize) -> Option<f32> {
        let q = self.embedder.embed(ch, query);
        let cm = self.model.chars.get(&ch)?;
        let probs = cm.probabilities(&q);
        probs.into_iter()
            .find(|(id, _)| *id as usize == font_id)
            .map(|(_, p)| p)
    }

    fn name(&self) -> &str {
        self.embedder.name()
    }

    fn font_count(&self) -> usize {
        self.model.font_names.len()
    }

    fn font_name(&self, font_id: usize) -> Option<&str> {
        self.model.font_names.get(font_id).map(|s| s.as_str())
    }

    fn add_font(&mut self, font_id: usize, font_name: &str, ch: char, features: &CharFeatures) {
        if font_id >= self.model.font_names.len() {
            self.model.font_names.resize(font_id + 1, String::new());
        }
        self.model.font_names[font_id] = font_name.to_string();
        let embedded = self.embedder.embed(ch, features);
        let cm = self.model.chars.entry(ch).or_insert_with(|| CharModel {
            weights: Vec::new(),
            centroids: Vec::new(),
            sigma_sq: 0.0,
        });
        cm.centroids.push((font_id as u32, embedded));
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
    nets: HashMap<char, GlyphNet>,
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
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
                if version < 3 {
                    eprintln!("Triplet weights {} are v{version}, retraining as v3...", path.display());
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

        let data = std::fs::read(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let model = PerCharModel::read_bin(&data, b"TRIP", None)?;

        let mut nets = HashMap::with_capacity(model.chars.len());
        for (&ch, cm) in &model.chars {
            if cm.weights.len() != PARAMS_PER_CHAR {
                return Err(format!(
                    "Triplet char '{}': expected {} params, got {}",
                    ch, PARAMS_PER_CHAR, cm.weights.len()
                ));
            }
            let mut pos = 0usize;
            let fc1_w = cm.weights[pos..pos + L1_IN * L1_OUT].to_vec(); pos += L1_IN * L1_OUT;
            let fc1_b = cm.weights[pos..pos + L1_OUT].to_vec(); pos += L1_OUT;
            let fc2_w = cm.weights[pos..pos + L1_OUT * L2_OUT].to_vec(); pos += L1_OUT * L2_OUT;
            let fc2_b = cm.weights[pos..pos + L2_OUT].to_vec(); pos += L2_OUT;
            let fc3_w = cm.weights[pos..pos + L2_OUT * L3_OUT].to_vec(); pos += L2_OUT * L3_OUT;
            let fc3_b = cm.weights[pos..pos + L3_OUT].to_vec();

            let net = GlyphNet {
                fc1: InferenceLinear { rows: L1_IN, cols: L1_OUT, w: fc1_w, b: fc1_b },
                fc2: InferenceLinear { rows: L1_OUT, cols: L2_OUT, w: fc2_w, b: fc2_b },
                fc3: InferenceLinear { rows: L2_OUT, cols: L3_OUT, w: fc3_w, b: fc3_b },
            };
            nets.insert(ch, net);
        }

        let embedder = Self { nets };
        Ok(EmbeddingClassifier { model, embedder: Box::new(embedder) })
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        if let Some(net) = self.nets.get(&ch) {
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

        let chars = ctx.chars;
        eprintln!("\nTriplet training {} characters (epochs={}, lr={}, margin={})...",
            chars.len(), epochs, lr, margin);

        let train_start = std::time::Instant::now();
        let mut trained_chars: Vec<(char, TrainableNet)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_rr_sum = 0.0f64;
        let mut total_top1 = 0usize;
        let mut total_top5 = 0usize;
        let mut total_eval = 0usize;

        // Collect per-char samples for centroid computation after training
        let mut per_char_samples: Vec<(char, Vec<crate::train::TrainingSample>)> = Vec::new();

        for (ci, &c) in chars.iter().enumerate() {
            if ctx.char_counts[ci] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(ci);

            let mut font_set: Vec<u32> = samples.iter().map(|s| s.font_id).collect();
            font_set.sort_unstable();
            font_set.dedup();
            if font_set.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let mut rng = SmallRng::seed_from_u64(c as u64);
            let mut net = TrainableNet::new(&mut rng);

            let mut font_samples: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_samples.entry(s.font_id).or_default().push(i);
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
                    if ci < 5 || ci == chars.len() - 1 {
                        eprintln!("  char '{}' epoch {}/{}: loss={:.4} ({} active triplets)",
                            c, epoch + 1, epochs, avg_loss, n_triplets);
                    }
                }
            }

            trained_chars.push((c, net));

            // Retrieval quality: MRR via nearest-centroid
            if let Some((_, ref trained_net)) = trained_chars.last() {
                let embeddings: Vec<Vec<f32>> = samples.iter()
                    .map(|s| trained_net.forward(&s.features).out)
                    .collect();

                let mut centroid_sums: HashMap<u32, Vec<f32>> = HashMap::new();
                let mut centroid_counts: HashMap<u32, usize> = HashMap::new();
                for (i, s) in samples.iter().enumerate() {
                    let entry = centroid_sums.entry(s.font_id).or_insert_with(|| vec![0.0; L3_OUT]);
                    for (j, &v) in embeddings[i].iter().enumerate() { entry[j] += v; }
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
                let max_eval = 2000usize;
                let eval_indices: Vec<usize> = if n <= max_eval {
                    (0..n).collect()
                } else {
                    let mut rng2 = SmallRng::seed_from_u64(c as u64 + 0x1234);
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
                    let correct_font = samples[i].font_id;
                    let ci_pos = centroid_fids.iter().position(|&f| f == correct_font).unwrap();
                    let d_correct = dist_sq(&embeddings[i], &centroid_vecs[ci_pos]);

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

                if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                    eprintln!("  char '{}' MRR={:.3} top1={:.1}% top5={:.1}% (n={})",
                        c, mrr,
                        char_top1 as f64 / n_eval as f64 * 100.0,
                        char_top5 as f64 / n_eval as f64 * 100.0,
                        n_eval);
                }
            }

            per_char_samples.push((c, samples));
        }

        let train_elapsed = train_start.elapsed();
        let mrr = if total_eval > 0 { total_rr_sum / total_eval as f64 } else { 0.0 };
        let top1 = if total_eval > 0 { total_top1 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        let top5 = if total_eval > 0 { total_top5 as f64 / total_eval as f64 * 100.0 } else { 0.0 };
        eprintln!("\nTriplet complete: {} chars, {} skipped, {:.1}s",
            trained_chars.len(), skipped, train_elapsed.as_secs_f64());
        eprintln!("  MRR={:.3} top1={:.1}% top5={:.1}% (n={})", mrr, top1, top5, total_eval);

        // Write TRIP v3 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let mut model = PerCharModel::new(ctx.catalog_hash);
        for (font_id, fe) in ctx.catalog.iter().enumerate() {
            if font_id >= model.font_names.len() {
                model.font_names.resize(font_id + 1, String::new());
            }
            model.font_names[font_id] = fe.font_key();
        }
        for (tc_idx, (c, net)) in trained_chars.iter().enumerate() {
            // Flatten net params into weights blob
            let mut weights = Vec::with_capacity(PARAMS_PER_CHAR);
            weights.extend_from_slice(&net.fc1.w);
            weights.extend_from_slice(&net.fc1.b);
            weights.extend_from_slice(&net.fc2.w);
            weights.extend_from_slice(&net.fc2.b);
            weights.extend_from_slice(&net.fc3.w);
            weights.extend_from_slice(&net.fc3.b);

            // Build centroids: embed each sample, average per font
            let samples = &per_char_samples[tc_idx].1;
            let mut sums: HashMap<u32, Vec<f32>> = HashMap::new();
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for s in samples {
                let emb = net.forward(&s.features).out;
                let entry = sums.entry(s.font_id).or_insert_with(|| vec![0.0; emb.len()]);
                for (j, &v) in emb.iter().enumerate() { entry[j] += v; }
                *counts.entry(s.font_id).or_insert(0) += 1;
            }
            let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(sums.len());
            for (&fid, sum) in &sums {
                let cnt = counts[&fid] as f32;
                let centroid: Vec<f32> = sum.iter().map(|&v| v / cnt).collect();
                centroids.push((fid, centroid));
            }
            let mut cm = CharModel { weights, centroids, sigma_sq: 0.0 };
            cm.compute_sigma_sq();
            model.chars.insert(*c, cm);
        }
        let f = std::fs::File::create(output).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"TRIP", 3).expect("write TRIP v3");
        w.flush().unwrap();

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e6, ctx.catalog.len());
    }

}

impl Embedder for TripletClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        TripletClassifier::embed(self, ch, features)
    }
    fn name(&self) -> &str { "triplet" }
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
/// Binary format (magic `b"TRPG"`, version 2):
/// ```text
/// magic:   b"TRPG" (4 bytes)
/// version: u32 LE (2)
/// W1: L1_IN×128 f32 LE, b1: 128 f32 LE
/// W2: 128×64 f32 LE,  b2: 64 f32 LE
/// W3: 64×32 f32 LE,   b3: 32 f32 LE
/// FontVecStore (FVST section)
/// ```
pub struct GlobalTripletClassifier {
    net: GlyphNet,
}

/// Header size for TRPG format: magic (4) + version (4) = 8 bytes.
/// Then PARAMS_PER_CHAR × 4 bytes of weights.
const GLOBAL_HEADER: usize = 8;

impl GlobalTripletClassifier {
    /// Load global triplet weights and font index.  No auto-train (training
    /// is done through TripletClassifier's trainer).
    pub fn load(
        path: &std::path::Path,
        _ctx: Option<&crate::train::TrainingContext>,
    ) -> Result<EmbeddingClassifier, String> {
        use std::io::{Cursor, Read};

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("cannot open global-triplet weights {}: {e}", path.display()))?
            .read_to_end(&mut data)
            .map_err(|e| format!("read error on {}: {e}", path.display()))?;

        let weights_end = GLOBAL_HEADER + PARAMS_PER_CHAR * 4;
        if data.len() < weights_end {
            return Err(format!(
                "global-triplet weights: {} bytes, need at least {} ({} header + {} params × 4)",
                data.len(),
                weights_end,
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
        if version != 2 {
            return Err(format!("unsupported global-triplet weights version {version} (expected 2)"));
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

        let embedder = Self { net };

        // Read legacy FVST font index and convert to PerCharModel
        let model = if data.len() > weights_end {
            let mut cursor = Cursor::new(&data[weights_end..]);
            let store = FontVecStore::read_from(&mut cursor)?;
            let mut pcm = PerCharModel::new(0);
            pcm.font_names = store.font_names;
            for (ch, entries) in &store.vecs {
                let centroids: Vec<(u32, Vec<f32>)> = entries.iter()
                    .map(|(fid, v)| (*fid as u32, v.clone()))
                    .collect();
                let mut cm = CharModel {
                    weights: Vec::new(), // global net, no per-char weights
                    centroids,
                    sigma_sq: 0.0,
                };
                cm.compute_sigma_sq();
                pcm.chars.insert(*ch, cm);
            }
            pcm
        } else {
            PerCharModel::new(0)
        };

        Ok(EmbeddingClassifier { model, embedder: Box::new(embedder) })
    }
}

impl Embedder for GlobalTripletClassifier {
    fn embed(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        self.net.forward(&features.as_slice())
    }
    fn name(&self) -> &str { "global_triplet" }
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
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
                if version < 3 {
                    eprintln!("Fisher weights {} are v{version}, retraining as v3...", path.display());
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

        let data = std::fs::read(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let model = PerCharModel::read_bin(&data, b"FISH", None)?;

        // Extract normalized embedder weights from the model
        let mut weights = HashMap::with_capacity(model.chars.len());
        for (&ch, cm) in &model.chars {
            let mut w = [0.0f32; FEAT_LEN];
            let n = cm.weights.len().min(FEAT_LEN);
            w[..n].copy_from_slice(&cm.weights[..n]);

            let max_finite = w.iter().filter(|v| v.is_finite()).copied()
                .fold(0.0f32, f32::max);
            for v in &mut w {
                if !v.is_finite() || *v > max_finite { *v = max_finite; }
                *v = v.sqrt();
            }
            let sum: f32 = w.iter().sum();
            if sum > 1e-12 { for v in &mut w { *v /= sum; } }

            weights.insert(ch, w);
        }

        let embedder = Self { weights };
        Ok(EmbeddingClassifier { model, embedder: Box::new(embedder) })
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some(w) = self.weights.get(&ch) {
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

        let chars = ctx.chars;
        eprintln!("\nFisher scoring {} characters...", chars.len());
        let fisher_start = std::time::Instant::now();
        let mut fisher_chars: Vec<(char, [f32; FEAT_LEN], HashMap<u32, Vec<f64>>)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (ci, &c) in chars.iter().enumerate() {
            if ctx.char_counts[ci] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(ci);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.font_id).or_default().push(i);
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
            let eval_indices = crate::train::subsample_eval(n, 2000, c);
            let centroid_fids: Vec<u32> = class_means.keys().copied().collect();
            let centroid_feats: Vec<&Vec<f64>> = centroid_fids.iter()
                .map(|fid| &class_means[fid])
                .collect();

            let char_stats = crate::train::eval_mrr(
                &samples, &eval_indices, &class_means, &centroid_fids,
                ctx.font_family,
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

            if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                eprintln!("  char '{}' base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    c, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }

            total_stats.accumulate(&char_stats);
            fisher_chars.push((c, scores, class_means));
        }

        let fisher_elapsed = fisher_start.elapsed();
        eprintln!("\nFisher scoring complete: {} chars, {} skipped, {:.1}s",
            fisher_chars.len(), skipped, fisher_elapsed.as_secs_f64());
        eprintln!("  Baseline:      MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  Fisher strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  Fisher family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write FISH v3 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut model = PerCharModel::new(ctx.catalog_hash);
        for (font_id, fe) in ctx.catalog.iter().enumerate() {
            if font_id >= model.font_names.len() {
                model.font_names.resize(font_id + 1, String::new());
            }
            model.font_names[font_id] = fe.font_key();
        }
        for (c, scores, class_means) in &fisher_chars {
            let nw = Self::normalize_scores(scores);
            let mut centroids: Vec<(u32, Vec<f32>)> = Vec::with_capacity(class_means.len());
            for (&fid, mean) in class_means {
                let embedded: Vec<f32> = (0..FEAT_LEN)
                    .map(|j| mean[j] as f32 * nw[j])
                    .collect();
                centroids.push((fid, embedded));
            }
            let mut cm = CharModel {
                weights: scores.to_vec(),
                centroids,
                sigma_sq: 0.0,
            };
            cm.compute_sigma_sq();
            model.chars.insert(*c, cm);
        }
        let f = std::fs::File::create(output).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"FISH", 3).expect("write FISH v3");
        w.flush().unwrap();

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} KB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e3, ctx.catalog.len());
    }
}

impl Embedder for PerCharFisherClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        PerCharFisherClassifier::embed(self, ch, features)
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
/// FontVecStore (FVST section)
/// ```
pub struct MahalanobisClassifier {
    transforms: HashMap<char, Vec<f32>>, // L_inv per char, FEAT_LEN × FEAT_LEN row-major
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
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
                if version < 3 {
                    eprintln!("Mahalanobis weights {} are v{version}, retraining as v3...", path.display());
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

        let data = std::fs::read(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let model = PerCharModel::read_bin(&data, b"MAHA", None)?;

        let mut transforms = HashMap::with_capacity(model.chars.len());
        for (&ch, cm) in &model.chars {
            transforms.insert(ch, cm.weights.clone());
        }

        let embedder = Self { transforms };
        Ok(EmbeddingClassifier { model, embedder: Box::new(embedder) })
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

        let chars = ctx.chars;
        eprintln!("\nMahalanobis training {} characters...", chars.len());
        let maha_start = std::time::Instant::now();
        let mut maha_chars: Vec<(char, Vec<f32>, HashMap<u32, Vec<f64>>)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (ci, &c) in chars.iter().enumerate() {
            if ctx.char_counts[ci] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(ci);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.font_id).or_default().push(i);
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
                eprintln!("  char '{}' Cholesky failed, skipping", c);
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
            let mut rng_cal = SmallRng::seed_from_u64(c as u64 + 9999);
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
            let eval_indices = crate::train::subsample_eval(n, 2000, c);
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
                ctx.font_family,
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

            if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                eprintln!("  char '{}' base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    c, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }
            total_stats.accumulate(&char_stats);
            maha_chars.push((c, linv_f32, class_means));
        }

        let maha_elapsed = maha_start.elapsed();
        eprintln!("\nMahalanobis complete: {} chars, {} skipped, {:.1}s",
            maha_chars.len(), skipped, maha_elapsed.as_secs_f64());
        eprintln!("  Baseline:    MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  Maha strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  Maha family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write MAHA v3 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let mut model = PerCharModel::new(ctx.catalog_hash);
        for (font_id, fe) in ctx.catalog.iter().enumerate() {
            if font_id >= model.font_names.len() {
                model.font_names.resize(font_id + 1, String::new());
            }
            model.font_names[font_id] = fe.font_key();
        }
        for (c, linv, class_means) in &maha_chars {
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
            let mut cm = CharModel {
                weights: linv.to_vec(),
                centroids,
                sigma_sq: 0.0,
            };
            cm.compute_sigma_sq();
            model.chars.insert(*c, cm);
        }
        let f = std::fs::File::create(output).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"MAHA", 3).expect("write MAHA v3");
        w.flush().unwrap();

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e6, ctx.catalog.len());
    }

}

impl Embedder for MahalanobisClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        MahalanobisClassifier::embed(self, ch, features)
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
/// FontVecStore (FVST section)
/// ```
pub struct LdaClassifier {
    projections: HashMap<char, (usize, Vec<f32>)>, // (out_dim, proj matrix)
}

impl LdaClassifier {
    /// Load an LDA classifier from an LDAC v4 binary, or train one if the file
    /// doesn't exist or is stale.  Returns a ready-to-use `EmbeddingClassifier`.
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
                let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
                if version < 4 {
                    eprintln!("LDA weights {} are v{version}, retraining as v4...", path.display());
                    true
                } else if let Some(c) = ctx {
                    let file_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());
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

        let data = std::fs::read(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let model = PerCharModel::read_bin(&data, b"LDAC", None)?;

        // Extract LDA projections from the model
        let mut projections = HashMap::with_capacity(model.chars.len());

        for (&ch, cm) in &model.chars {
            if cm.weights.is_empty() {
                return Err(format!("LDA char '{}': empty weights", ch));
            }
            let out_dim = cm.weights[0] as usize;
            let proj = cm.weights[1..].to_vec();
            if proj.len() != out_dim * FEAT_LEN {
                return Err(format!(
                    "LDA char '{}': proj len {} != {} × {} = {}",
                    ch, proj.len(), out_dim, FEAT_LEN, out_dim * FEAT_LEN
                ));
            }
            projections.insert(ch, (out_dim, proj));
        }

        let embedder = Self { projections };
        Ok(EmbeddingClassifier { model, embedder: Box::new(embedder) })
    }

    fn project(out_dim: usize, proj: &[f32], x: &[f32]) -> Vec<f32> {
        dense_project(out_dim, proj, x)
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some((out_dim, proj)) = self.projections.get(&ch) {
            Self::project(*out_dim, proj, &raw)
        } else {
            raw.to_vec()
        }
    }

    /// Train LDA weights and write an LDAC v3 binary (weights + font index).
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

        let chars = ctx.chars;
        let out_dim = lda_dims.min(FEAT_LEN - 1);
        eprintln!("\nLDA training {} characters (target dim={})...", chars.len(), out_dim);
        let lda_start = std::time::Instant::now();
        let mut lda_chars: Vec<(char, usize, Vec<f32>, f32, HashMap<u32, Vec<f64>>)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (ci, &c) in chars.iter().enumerate() {
            if ctx.char_counts[ci] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(ci);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.font_id).or_default().push(i);
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

            // Scale calibration
            let target_dist = 0.03f64;
            let mut within_dists: Vec<f64> = Vec::new();
            let mut rng_cal = SmallRng::seed_from_u64(c as u64 + 9999);
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
            let eval_indices = crate::train::subsample_eval(n, 2000, c);
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
                ctx.font_family,
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

            if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                eprintln!("  char '{}' base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    c, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }
            total_stats.accumulate(&char_stats);
            lda_chars.push((c, actual_dim, proj_f32, sigma_sq, class_means));
        }

        let lda_elapsed = lda_start.elapsed();
        eprintln!("\nLDA complete: {} chars, {} skipped, {:.1}s",
            lda_chars.len(), skipped, lda_elapsed.as_secs_f64());
        eprintln!("  Baseline:   MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  LDA strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  LDA family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write LDAC v4 binary (per-char model: weights + centroids + σ²)
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let mut model = PerCharModel::new(ctx.catalog_hash);
        for (font_id, fe) in ctx.catalog.iter().enumerate() {
            if font_id >= model.font_names.len() {
                model.font_names.resize(font_id + 1, String::new());
            }
            model.font_names[font_id] = fe.font_key();
        }
        for (c, actual_dim, proj, sigma, class_means) in &lda_chars {
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

            let mut cm = CharModel {
                weights,
                centroids,
                sigma_sq: *sigma,
            };
            if cm.sigma_sq <= 1e-30 {
                cm.compute_sigma_sq();
            }
            model.chars.insert(*c, cm);
        }
        let f = std::fs::File::create(output).expect("create output file");
        let mut w = BufWriter::new(f);
        model.write_bin(&mut w, b"LDAC", 4).expect("write LDAC v4");
        w.flush().unwrap();

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB, {} fonts indexed)",
            output.display(), file_size as f64 / 1e6, ctx.catalog.len());
    }

}

impl Embedder for LdaClassifier {
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        LdaClassifier::embed(self, ch, features)
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
    font_names: Vec<String>,
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

        Ok(Self { nets, font_names: Vec::new() })
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

        let chars = ctx.chars;
        eprintln!("\nMLP training {} characters (epochs={}, noise={}, dropout={})...",
            chars.len(), epochs, mlp_noise, mlp_dropout);
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

        let mut mlp_chars: Vec<(char, usize, Vec<u32>, MlpNet)> = Vec::new();
        let mut skipped = 0usize;
        let mut total_stats = crate::train::RankStats::default();

        for (ci, &c) in chars.iter().enumerate() {
            if ctx.char_counts[ci] == 0 { skipped += 1; continue; }
            let samples = ctx.load_samples(ci);

            let mut font_indices: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, s) in samples.iter().enumerate() {
                font_indices.entry(s.font_id).or_default().push(i);
            }
            if font_indices.len() < ctx.min_fonts.max(2) { skipped += 1; continue; }

            let n = samples.len();

            let mut font_ids_sorted: Vec<u32> = font_indices.keys().copied().collect();
            font_ids_sorted.sort_unstable();
            let k = font_ids_sorted.len();
            let fid_to_class: HashMap<u32, usize> = font_ids_sorted.iter()
                .enumerate().map(|(ci2, &fid)| (fid, ci2)).collect();
            let class_map: Vec<u32> = font_ids_sorted.clone();

            let mut rng = SmallRng::seed_from_u64(c as u64);
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
                        let label = fid_to_class[&samples[si].font_id];
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
                    if ci < 5 || ci == chars.len() - 1 {
                        eprintln!("  char '{}' epoch {}/{}: loss={:.4}", c, epoch + 1, epochs, avg_loss);
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
            let eval_indices = crate::train::subsample_eval(n, 2000, c);

            let char_stats = crate::train::eval_mrr(
                &samples, &eval_indices, &class_means, &centroid_fids,
                ctx.font_family,
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

            if ci < 5 || ci == chars.len() - 1 || (ci + 1) % 20 == 0 {
                eprintln!("  char '{}' base={:.3} | strict={:.3} t1={:.1}% | family={:.3} t1={:.1}%",
                    c, char_stats.base_mrr(), char_stats.strict_mrr(),
                    char_stats.strict_top1_pct(), char_stats.family_mrr(),
                    char_stats.family_top1_pct());
            }
            total_stats.accumulate(&char_stats);
            mlp_chars.push((c, k, class_map, net));
        }

        let mlp_elapsed = mlp_start.elapsed();
        eprintln!("\nMLP complete: {} chars, {} skipped, {:.1}s",
            mlp_chars.len(), skipped, mlp_elapsed.as_secs_f64());
        eprintln!("  Baseline:   MRR={:.3} top1={:.1}%", total_stats.base_mrr(), total_stats.base_top1_pct());
        eprintln!("  MLP strict: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.strict_mrr(), total_stats.strict_top1_pct(), total_stats.strict_top5_pct());
        eprintln!("  MLP family: MRR={:.3} top1={:.1}% top5={:.1}%",
            total_stats.family_mrr(), total_stats.family_top1_pct(), total_stats.family_top5_pct());

        // Write MLPC binary
        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
        let f = std::fs::File::create(output).expect("create output file");
        let mut w = BufWriter::new(f);
        w.write_all(b"MLPC").unwrap();
        w.write_all(&1u32.to_le_bytes()).unwrap();
        w.write_all(&(mlp_chars.len() as u32).to_le_bytes()).unwrap();
        for (c, k, class_map, net) in &mlp_chars {
            w.write_all(&(*c as u32).to_le_bytes()).unwrap();
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

        let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Weights: {} ({:.1} MB)", output.display(), file_size as f64 / 1e6);
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

    fn font_name(&self, font_id: usize) -> Option<&str> {
        self.font_names.get(font_id).map(|s| s.as_str())
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

    fn font_name(&self, font_id: usize) -> Option<&str> {
        self.children.first().and_then(|(_, c)| c.font_name(font_id))
    }

    fn add_font(&mut self, font_id: usize, font_name: &str, ch: char, features: &CharFeatures) {
        for (_, child) in &mut self.children {
            child.add_font(font_id, font_name, ch, features);
        }
    }
}

// ---------------------------------------------------------------------------
// Classifier construction
// ---------------------------------------------------------------------------

/// Default path for cached LDA weights.
pub fn default_lda_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("lda-weights.bin")
}

/// Default path for cached per-char Fisher weights.
pub fn default_fisher_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("fisher-weights.bin")
}

/// Default path for cached triplet weights.
pub fn default_triplet_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("triplet-weights.bin")
}

/// Default path for cached global-triplet weights.
pub fn default_global_triplet_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("global-triplet-weights.bin")
}

/// Default path for cached Mahalanobis weights.
pub fn default_mahalanobis_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("mahalanobis-weights.bin")
}

/// Default path for cached MLP weights.
pub fn default_mlp_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("mlp-weights.bin")
}

/// Default path for the font catalog file.
pub fn default_catalog_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("unprint").join("catalog.bin")
}

/// Try to load a classifier from its cached weights, auto-training if missing.
///
/// 1. If  is explicitly set, load from it (fail hard on error).
/// 2. Otherwise try .
/// 3. If missing, auto-train using  to produce .
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

    // Try default cache path.
    if default_path.exists() {
        match load_fn(default_path) {
            Ok(c) => return Box::new(c),
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
        "global-triplet" => load_or_train(
            "Global-Triplet", weights_path, &default_global_triplet_weights_path(),
            |p| GlobalTripletClassifier::load(p, None).map(|c| { let ec: EmbeddingClassifier = c; ec }),
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
        "zncc" => Box::new(crate::zncc_classifier::ZnccClassifier::new()),
        other => {
            eprintln!("Error: unknown classifier '{other}'. Use 'lda', 'perchar-fisher', 'triplet', 'global-triplet', 'mahalanobis', 'mlp', 'fusion', or 'zncc'.");
            std::process::exit(1);
        }
    }
}
