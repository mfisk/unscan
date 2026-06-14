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

/// A font classifier that transforms raw character features into an
/// embedding space and computes distances in that space.
pub trait Classifier: Send + Sync {
    /// Dimensionality of the embedding (output length of [`embed`]).
    fn embed_dim(&self) -> usize;

    /// Transform raw character features into the classifier's embedding space.
    ///
    /// The `ch` parameter identifies which character the features represent.
    /// Global classifiers (like Fisher) ignore it; per-glyph classifiers
    /// use it to select the appropriate model.
    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32>;

    /// Squared distance between two embedded vectors.
    ///
    /// **Contract**: lower values mean closer match (more similar fonts).
    /// All classifiers must follow this convention so the downstream scoring
    /// pipeline (log-distance aggregation, negation to higher-is-better)
    /// produces consistent results regardless of classifier choice.
    fn distance_sq(&self, a: &[f32], b: &[f32]) -> f32;

    /// Short name for logging and cache invalidation.
    fn name(&self) -> &str;
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
    fn embed_dim(&self) -> usize {
        FEAT_LEN
    }

    fn embed(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        let raw = features.as_slice();
        let mut v = vec![0.0f32; FEAT_LEN];
        for i in 0..FEAT_LEN {
            v[i] = raw[i] * FISHER_WEIGHTS[i];
        }
        v
    }

    fn distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
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

        eprintln!("Loaded triplet weights: {} per-glyph models from {}", nets.len(), path.display());
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
    fn embed_dim(&self) -> usize {
        L3_OUT
    }

    fn embed(&self, ch: char, features: &CharFeatures) -> Vec<f32> {
        if let Some(net) = self.nets.get(&ch) {
            net.forward(&features.as_slice())
        } else {
            // Fallback: Fisher weights truncated to L3_OUT dims.
            // Only fires for chars not in the training data.
            let fisher = Self::fisher_embed(features);
            fisher[..L3_OUT.min(fisher.len())].to_vec()
        }
    }

    fn distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
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

        eprintln!("Loaded global-triplet weights from {}", path.display());
        Ok(Self { net })
    }
}

impl Classifier for GlobalTripletClassifier {
    fn embed_dim(&self) -> usize {
        L3_OUT
    }

    fn embed(&self, _ch: char, features: &CharFeatures) -> Vec<f32> {
        self.net.forward(&features.as_slice())
    }

    fn distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn name(&self) -> &str {
        "global-triplet"
    }
}
