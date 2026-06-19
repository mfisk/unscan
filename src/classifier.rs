//! Pluggable font classifiers for the character index.
//!
//! A **classifier** transforms raw 100-dim feature vectors into an embedding
//! space and defines a distance metric in that space.  The character index
//! stores embedded vectors and uses the classifier's distance function for
//! nearest-neighbor search.
//!
//! Two implementations ship today:
//!
//! - [`FisherClassifier`]: the original diagonal Fisher-weighted Euclidean
//!   distance.  Equivalent to the pre-refactor behaviour.
//! - [`TripletClassifier`]: per-glyph learned 3-layer MLPs (100→128→64→32)
//!   trained with triplet loss.  One network per indexed character.

use crate::char_index::{CharFeatures, FEAT_LEN};

// Re-export the Fisher weights so FisherClassifier can use them without
// making them pub in char_index.
use crate::char_index::FISHER_WEIGHTS;

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A font classifier that stores per-font representations at index time
/// and ranks candidates against a query glyph at search time.
pub trait Classifier: Send + Sync {
    /// Dimensionality of the stored representation per font entry.
    fn prepare_dim(&self) -> usize;

    /// Compute the stored representation for a font entry at index build time.
    ///
    /// The `ch` parameter identifies which character the features represent.
    /// The returned vector is stored in the index and passed back to [`rank`]
    /// at query time.
    fn prepare(&self, ch: char, features: &CharFeatures) -> Vec<f32>;

    /// Rank all candidates for a character against a query glyph.
    ///
    /// Returns an iterator of `(font_id, score)` in **best-first order**
    /// (lowest score = closest match). The caller can pull matches one at
    /// a time, stopping when one passes downstream verification (e.g. SSIM).
    ///
    /// `candidates` contains `(font_id, stored_representation)` pairs from
    /// the index, where each `stored_representation` was produced by a prior
    /// call to [`prepare`].
    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a>;

    /// Short name for logging and cache invalidation.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Helpers for embedding-based classifiers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two slices.
fn sq_euclid(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum()
}

/// Default rank implementation for embedding-based classifiers:
/// embed the query, compute squared Euclidean distance to every candidate,
/// return sorted by distance.
fn rank_by_embedding(
    query_embedded: Vec<f32>,
    candidates: &[(usize, Vec<f32>)],
) -> Box<dyn Iterator<Item = (usize, f32)> + '_> {
    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .map(|(id, stored)| (*id, sq_euclid(&query_embedded, stored)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Box::new(scored.into_iter())
}

// ---------------------------------------------------------------------------
// Fisher (original)
// ---------------------------------------------------------------------------

/// Diagonal Fisher-weighted Euclidean distance — the original classifier.
///
/// Each raw feature dimension is multiplied by its learned Fisher weight
/// (√(between-font variance / within-font variance), normalised to sum = 1).
/// Distance is plain squared Euclidean in the weighted space.
pub struct FisherClassifier;

impl Classifier for FisherClassifier {
    fn prepare_dim(&self) -> usize {
        FEAT_LEN
    }

    fn prepare(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        let mut v = vec![0.0f32; FEAT_LEN];
        for i in 0..FEAT_LEN {
            v[i] = raw[i] * FISHER_WEIGHTS[i];
        }
        v
    }

    fn rank<'a>(
        &'a self,
        _ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let q = self.prepare(_ch, query);
        rank_by_embedding(q, candidates)
    }

    fn name(&self) -> &str {
        "fisher"
    }
}

// ---------------------------------------------------------------------------
// Triplet network (per-glyph)
// ---------------------------------------------------------------------------

const L1_IN: usize = FEAT_LEN; // 100
const L1_OUT: usize = 128;
const L2_OUT: usize = 64;
const L3_OUT: usize = 32;

/// Per-character parameter count.
const PARAMS_PER_CHAR: usize =
    L1_IN * L1_OUT + L1_OUT + // W1, b1
    L1_OUT * L2_OUT + L2_OUT + // W2, b2
    L2_OUT * L3_OUT + L3_OUT;  // W3, b3

/// Weights for a single character's MLP.
struct GlyphNet {
    w1: Vec<f32>, // 100 × 128 row-major
    b1: Vec<f32>, // 128
    w2: Vec<f32>, // 128 × 64
    b2: Vec<f32>, // 64
    w3: Vec<f32>, // 64 × 32
    b3: Vec<f32>, // 32
}

impl GlyphNet {
    /// Forward pass: ReLU(W1*x + b1) → ReLU(W2*h + b2) → W3*h + b3 → L2-normalize
    fn forward(&self, raw: &[f32]) -> Vec<f32> {
        debug_assert_eq!(raw.len(), L1_IN);

        // Layer 1: h = ReLU(W1^T * x + b1)
        let mut h1 = self.b1.clone();
        for j in 0..L1_OUT {
            let mut sum = h1[j];
            for i in 0..L1_IN {
                sum += raw[i] * self.w1[i * L1_OUT + j];
            }
            h1[j] = sum.max(0.0);
        }

        // Layer 2: h = ReLU(W2^T * h1 + b2)
        let mut h2 = self.b2.clone();
        for j in 0..L2_OUT {
            let mut sum = h2[j];
            for i in 0..L1_OUT {
                sum += h1[i] * self.w2[i * L2_OUT + j];
            }
            h2[j] = sum.max(0.0);
        }

        // Layer 3: out = W3^T * h2 + b3 (linear, no activation)
        let mut out = self.b3.clone();
        for j in 0..L3_OUT {
            let mut sum = out[j];
            for i in 0..L2_OUT {
                sum += h2[i] * self.w3[i * L3_OUT + j];
            }
            out[j] = sum;
        }

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
/// Binary format:
/// ```text
/// magic:   b"TRIP" (4 bytes)
/// version: u32 LE (1)
/// n_chars: u32 LE
/// Per character (repeated n_chars times):
///   char_code: u32 LE (Unicode codepoint)
///   W1: 100×128 f32 LE, b1: 128 f32 LE
///   W2: 128×64 f32 LE,  b2: 64 f32 LE
///   W3: 64×32 f32 LE,   b3: 32 f32 LE
/// ```
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

        let mut pos = 12usize;
        let read_vec = |pos: &mut usize, n: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let val = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
                v.push(val);
                *pos += 4;
            }
            v
        };

        let mut nets = HashMap::with_capacity(n_chars);
        for _ in 0..n_chars {
            let codepoint = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let ch = char::from_u32(codepoint).ok_or_else(|| {
                format!("invalid codepoint U+{codepoint:04X} in triplet weights")
            })?;

            let net = GlyphNet {
                w1: read_vec(&mut pos, L1_IN * L1_OUT),
                b1: read_vec(&mut pos, L1_OUT),
                w2: read_vec(&mut pos, L1_OUT * L2_OUT),
                b2: read_vec(&mut pos, L2_OUT),
                w3: read_vec(&mut pos, L2_OUT * L3_OUT),
                b3: read_vec(&mut pos, L3_OUT),
            };
            nets.insert(ch, net);
        }

        Ok(Self { nets })
    }

    /// Fisher-weighted fallback for characters without a trained model.
    fn fisher_embed(features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        let mut v = vec![0.0f32; FEAT_LEN];
        for i in 0..FEAT_LEN {
            v[i] = raw[i] * FISHER_WEIGHTS[i];
        }
        v
    }
}

impl Classifier for TripletClassifier {
    fn prepare_dim(&self) -> usize {
        L3_OUT
    }

    fn prepare(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        if let Some(net) = self.nets.get(&ch) {
            net.forward(&features.as_slice())
        } else {
            let fisher = Self::fisher_embed(features);
            fisher[..L3_OUT.min(fisher.len())].to_vec()
        }
    }

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let q = self.prepare(ch, query);
        rank_by_embedding(q, candidates)
    }

    fn name(&self) -> &str {
        "triplet"
    }
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
/// W1: 100×128 f32 LE, b1: 128 f32 LE
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

        let mut pos = GLOBAL_HEADER;
        let read_vec = |pos: &mut usize, n: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let val = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
                v.push(val);
                *pos += 4;
            }
            v
        };

        let net = GlyphNet {
            w1: read_vec(&mut pos, L1_IN * L1_OUT),
            b1: read_vec(&mut pos, L1_OUT),
            w2: read_vec(&mut pos, L1_OUT * L2_OUT),
            b2: read_vec(&mut pos, L2_OUT),
            w3: read_vec(&mut pos, L2_OUT * L3_OUT),
            b3: read_vec(&mut pos, L3_OUT),
        };

        Ok(Self { net })
    }
}

impl Classifier for GlobalTripletClassifier {
    fn prepare_dim(&self) -> usize {
        L3_OUT
    }

    fn prepare(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        self.net.forward(&features.as_slice())
    }

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let q = self.prepare(ch, query);
        rank_by_embedding(q, candidates)
    }

    fn name(&self) -> &str {
        "global-triplet"
    }
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
/// feat_len: u32 LE (100)
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
        let mut pos = 16;
        for _ in 0..n_chars {
            let cp = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let mut w = [0.0f32; FEAT_LEN];
            for j in 0..FEAT_LEN {
                w[j] = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
            }

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
}

impl Classifier for PerCharFisherClassifier {
    fn prepare_dim(&self) -> usize {
        FEAT_LEN
    }

    fn prepare(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
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

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let q = self.prepare(ch, query);
        rank_by_embedding(q, candidates)
    }

    fn name(&self) -> &str {
        "perchar-fisher"
    }
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
        let mut pos = 16;
        for _ in 0..n_chars {
            let cp = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let mut linv = vec![0.0f32; FEAT_LEN * FEAT_LEN];
            for j in 0..FEAT_LEN * FEAT_LEN {
                linv[j] = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
            }
            transforms.insert(ch, linv);
        }

        Ok(Self { transforms })
    }

    /// Apply L_inv transform: y = L_inv * x
    fn apply_transform(linv: &[f32], x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; FEAT_LEN];
        for i in 0..FEAT_LEN {
            let mut sum = 0.0f32;
            let row = &linv[i * FEAT_LEN..(i + 1) * FEAT_LEN];
            for j in 0..FEAT_LEN {
                sum += row[j] * x[j];
            }
            y[i] = sum;
        }
        y
    }
}

impl Classifier for MahalanobisClassifier {
    fn prepare_dim(&self) -> usize {
        FEAT_LEN
    }

    fn prepare(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some(linv) = self.transforms.get(&ch) {
            Self::apply_transform(linv, &raw)
        } else {
            let mut v = vec![0.0f32; FEAT_LEN];
            for i in 0..FEAT_LEN {
                v[i] = raw[i] * FISHER_WEIGHTS[i];
            }
            v
        }
    }

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let q = self.prepare(ch, query);
        rank_by_embedding(q, candidates)
    }

    fn name(&self) -> &str {
        "mahalanobis"
    }
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
        if version != 1 {
            return Err(format!("unsupported version {version}"));
        }
        let n_chars = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

        let mut projections = HashMap::with_capacity(n_chars);
        let mut pos = 12;
        for _ in 0..n_chars {
            if pos + 8 > data.len() {
                return Err("truncated LDA file".into());
            }
            let cp = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let out_dim = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            let n_floats = out_dim * FEAT_LEN;
            if pos + n_floats * 4 > data.len() {
                return Err("truncated LDA projection data".into());
            }
            let mut proj = vec![0.0f32; n_floats];
            for j in 0..n_floats {
                proj[j] = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
            }
            projections.insert(ch, (out_dim, proj));
        }

        let dims: Vec<usize> = projections.values().map(|(d, _)| *d).collect();
        let max_dim = dims.iter().max().copied().unwrap_or(0);
        Ok(Self { projections })
    }

    fn project(out_dim: usize, proj: &[f32], x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            let row = &proj[i * FEAT_LEN..(i + 1) * FEAT_LEN];
            let mut sum = 0.0f32;
            for j in 0..FEAT_LEN {
                sum += row[j] * x[j];
            }
            y[i] = sum;
        }
        y
    }
}

impl Classifier for LdaClassifier {
    fn prepare_dim(&self) -> usize {
        // Variable per char; return max across all chars
        self.projections.values().map(|(d, _)| *d).max().unwrap_or(FEAT_LEN)
    }

    fn prepare(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        if let Some((out_dim, proj)) = self.projections.get(&ch) {
            Self::project(*out_dim, proj, &raw)
        } else {
            // Fallback to Fisher
            let mut v = vec![0.0f32; FEAT_LEN];
            for i in 0..FEAT_LEN {
                v[i] = raw[i] * FISHER_WEIGHTS[i];
            }
            v
        }
    }

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let q = self.prepare(ch, query);
        rank_by_embedding(q, candidates)
    }

    fn name(&self) -> &str {
        "lda"
    }
}

// ---------------------------------------------------------------------------
// MLP (per-character direct multi-class softmax classifier)
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
        let mut pos = 12;

        let read_f32s = |pos: &mut usize, n: usize, data: &[u8]| -> Result<Vec<f32>, String> {
            let need = n * 4;
            if *pos + need > data.len() {
                return Err("truncated MLP weight data".into());
            }
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()));
                *pos += 4;
            }
            Ok(v)
        };

        let read_u32 = |pos: &mut usize, data: &[u8]| -> Result<u32, String> {
            if *pos + 4 > data.len() {
                return Err("truncated MLP header data".into());
            }
            let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        };

        for _ in 0..n_chars {
            let cp = read_u32(&mut pos, data)?;
            let ch = char::from_u32(cp)
                .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?;
            let k = read_u32(&mut pos, data)? as usize;
            if k == 0 {
                return Err(format!("char '{}': zero classes", ch));
            }

            // class_map: k × u32
            let mut class_map = Vec::with_capacity(k);
            for _ in 0..k {
                class_map.push(read_u32(&mut pos, data)?);
            }

            // Layer weights
            let w1 = read_f32s(&mut pos, FEAT_LEN * MLP_H1, data)?;
            let b1 = read_f32s(&mut pos, MLP_H1, data)?;
            let w2 = read_f32s(&mut pos, MLP_H1 * MLP_H2, data)?;
            let b2 = read_f32s(&mut pos, MLP_H2, data)?;
            let w3 = read_f32s(&mut pos, MLP_H2 * k, data)?;
            let b3 = read_f32s(&mut pos, k, data)?;

            nets.insert(ch, MlpCharNet {
                fc1: InferenceLinear { rows: FEAT_LEN, cols: MLP_H1, w: w1, b: b1 },
                fc2: InferenceLinear { rows: MLP_H1, cols: MLP_H2, w: w2, b: b2 },
                fc3: InferenceLinear { rows: MLP_H2, cols: k, w: w3, b: b3 },
                class_map,
            });
        }

        Ok(Self { nets })
    }
}

impl Classifier for MlpClassifier {
    fn prepare_dim(&self) -> usize {
        // We store only the font_id (as f32) for each candidate.
        // The MLP does not use stored embeddings — it classifies the query
        // directly and maps outputs back to font_ids via the class map.
        1
    }

    fn prepare(&self, _ch: char, _features: &CharFeatures) -> Vec<f32> {
        // Minimal storage: we just need the font_id back from candidates.
        // The char_index stores font_id as the first element of the tuple,
        // so we return an empty-ish marker. But since prepare_dim() == 1,
        // we must return exactly 1 element. We store 0.0 as a placeholder.
        vec![0.0]
    }

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let raw = query.as_slice();

        if let Some(net) = self.nets.get(&ch) {
            let logits = net.forward(&raw);
            let k = logits.len();

            // Numerically stable softmax
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; k];
            let mut sum_exp = 0.0f32;
            for i in 0..k {
                probs[i] = (logits[i] - max_logit).exp();
                sum_exp += probs[i];
            }
            for p in &mut probs { *p /= sum_exp; }

            // Build font_id → probability lookup
            let mut fid_to_prob: HashMap<usize, f32> = HashMap::with_capacity(k);
            for (ci, &fid) in net.class_map.iter().enumerate() {
                fid_to_prob.insert(fid as usize, probs[ci]);
            }

            // Score each candidate: -probability (lower = better)
            let mut scored: Vec<(usize, f32)> = candidates
                .iter()
                .map(|(id, _)| {
                    let prob = fid_to_prob.get(id).copied().unwrap_or(0.0);
                    (*id, -prob)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            Box::new(scored.into_iter())
        } else {
            // Fallback to Fisher for unknown characters
            let mut v = vec![0.0f32; FEAT_LEN];
            for i in 0..FEAT_LEN {
                v[i] = raw[i] * FISHER_WEIGHTS[i];
            }
            rank_by_embedding(v, candidates)
        }
    }

    fn name(&self) -> &str {
        "mlp"
    }
}

// ---------------------------------------------------------------------------
// Rank fusion (combines multiple classifiers)
// ---------------------------------------------------------------------------

/// Rank-fusion classifier that combines scores from multiple child classifiers.
///
/// At index time, stores each child's prepared representation concatenated
/// with length prefixes. At query time, reconstructs per-child candidate
/// lists, calls each child's `rank()`, normalizes scores to [0,1], and
/// computes a weighted average. The fused score determines final ranking.
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
    fn prepare_dim(&self) -> usize {
        // Sum of all children's dims + 1 length header per child
        self.children.iter()
            .map(|(_, c)| 1 + c.prepare_dim())
            .sum()
    }

    fn prepare(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        // Concatenate: [child0_len, child0_data..., child1_len, child1_data..., ...]
        let mut result = Vec::new();
        for (_, child) in &self.children {
            let child_data = child.prepare(ch, features);
            result.push(child_data.len() as f32);
            result.extend_from_slice(&child_data);
        }
        result
    }

    fn rank<'a>(
        &'a self,
        ch: char,
        query: &CharFeatures,
        candidates: &'a [(usize, Vec<f32>)],
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let n_children = self.children.len();

        // For each child, reconstruct its candidate slice from the concatenated data
        // and call its rank() to get scores.
        let mut per_child_scores: Vec<HashMap<usize, f32>> = Vec::with_capacity(n_children);

        for (child_idx, (_, child)) in self.children.iter().enumerate() {
            // Reconstruct per-child candidates
            let child_candidates: Vec<(usize, Vec<f32>)> = candidates.iter().map(|(id, stored)| {
                // Parse stored data to extract this child's portion
                let mut pos = 0usize;
                for _ci in 0..child_idx {
                    if pos >= stored.len() { break; }
                    let len = stored[pos] as usize;
                    pos += 1 + len;
                }
                let child_data = if pos < stored.len() {
                    let len = stored[pos] as usize;
                    pos += 1;
                    if pos + len <= stored.len() {
                        stored[pos..pos + len].to_vec()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
                (*id, child_data)
            }).collect();

            let scores: Vec<(usize, f32)> = child.rank(ch, query, &child_candidates).collect();
            let score_map: HashMap<usize, f32> = scores.into_iter().collect();
            per_child_scores.push(score_map);
        }

        // Normalize each child's scores to [0, 1] range and compute weighted average
        let mut fused: HashMap<usize, f32> = HashMap::with_capacity(candidates.len());

        // First pass: find min/max per child
        let mut child_ranges: Vec<(f32, f32)> = Vec::with_capacity(n_children);
        for scores in &per_child_scores {
            let min = scores.values().copied().fold(f32::INFINITY, f32::min);
            let max = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
            child_ranges.push((min, max));
        }

        // Initialize fused scores
        for (id, _) in candidates {
            fused.insert(*id, 0.0);
        }

        // Accumulate weighted normalized scores
        for (child_idx, ((weight, _), (min, max))) in
            self.children.iter().zip(child_ranges.iter()).enumerate()
        {
            let range = max - min;
            let norm_weight = weight / self.weight_sum;

            for (id, _) in candidates {
                let raw_score = per_child_scores[child_idx]
                    .get(id)
                    .copied()
                    .unwrap_or(*max); // worst score for missing

                let normalized = if range > 1e-12 {
                    (raw_score - min) / range
                } else {
                    0.5 // all scores equal
                };

                *fused.get_mut(id).unwrap() += norm_weight * normalized;
            }
        }

        // Sort by fused score (lower = better)
        let mut scored: Vec<(usize, f32)> = fused.into_iter().collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Box::new(scored.into_iter())
    }

    fn name(&self) -> &str {
        "fusion"
    }
}
