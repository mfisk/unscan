//! Pure-Rust random forest classifier for font identification.
//!
//! Each per-character model is a collection of decision trees trained on
//! rendered glyph feature vectors.  Training data comes from the char index
//! entries (one sample per font/variant per character).
//!
//! ## Design choices
//!
//! * **No external ML crate** — avoids dependency churn and keeps the build
//!   hermetic.
//! * **Axis-aligned splits** — at each node we pick the feature + threshold
//!   that maximises a Gini-impurity reduction.  Simple, fast, and effective
//!   on 59-dimensional data.
//! * **Bagging** — each tree sees a bootstrap sample of the training data and
//!   a random subset of features at each split (√D features ≈ 8 out of 59).
//! * **Top-N predictions** — the forest returns class vote proportions so the
//!   caller can get ranked candidates with confidence scores.
//!
//! ## Serialization
//!
//! Models are compact: each node is 4 fields.  We provide `to_bytes` / `from_bytes`
//! round-trip so the trained forest can be cached alongside the char index.

use crate::char_index::FEAT_LEN;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

/// Number of trees per character model.
const N_TREES: usize = 10;
/// Maximum tree depth.  Keeps memory bounded.
const MAX_DEPTH: usize = 14;
/// Minimum samples at a leaf to allow further splitting.
const MIN_SAMPLES_SPLIT: usize = 2;
/// Minimum samples at a leaf.
const MIN_SAMPLES_LEAF: usize = 1;
/// Number of random threshold candidates per feature (ExtraTrees style).
/// Using random thresholds instead of best-split search makes training O(N)
/// per node instead of O(N log N), critical for 5000+ sample datasets.
const N_RANDOM_THRESHOLDS: usize = 1;

// ---------------------------------------------------------------------------
// Decision tree node
// ---------------------------------------------------------------------------

/// A single node in a decision tree.
#[derive(Debug, Clone)]
enum Node {
    /// Internal split node.
    Split {
        feature: u16,    // which feature dimension to split on
        threshold: f32,  // split threshold
        left: Box<Node>, // samples with feature <= threshold
        right: Box<Node>,
    },
    /// Leaf node with class vote counts.
    Leaf {
        /// class_id → count of training samples that reached this leaf.
        votes: Vec<(u32, u32)>, // (class_id, count) — sorted desc by count
    },
}

/// A single decision tree.
#[derive(Debug, Clone)]
struct DecisionTree {
    root: Node,
}

/// A random forest: an ensemble of decision trees.
#[derive(Debug, Clone)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_classes: usize,
}

/// Per-character trained model.
#[derive(Debug, Clone)]
pub struct FontClassifier {
    /// char → trained random forest
    pub models: HashMap<char, RandomForest>,
    /// char → class_id → font_name mapping
    pub class_names: HashMap<char, Vec<String>>,
}

// ---------------------------------------------------------------------------
// Simple RNG (xoshiro128+) — deterministic, no external dep
// ---------------------------------------------------------------------------

struct Rng {
    s: [u32; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let s0 = seed as u32;
        let s1 = (seed >> 32) as u32;
        Rng {
            s: [s0 ^ 0x12345678, s1 ^ 0x9abcdef0, s0.wrapping_mul(2654435761), s1.wrapping_mul(2246822519)],
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let result = self.s[0].wrapping_add(self.s[3]);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        result
    }

    /// Random usize in [0, n)
    #[inline]
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// Random f32 in [0, 1)
    #[inline]
    #[allow(dead_code)]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Training sample: feature vector + class label.
struct Sample {
    features: [f32; FEAT_LEN],
    class_id: u32,
}

impl RandomForest {
    /// Train a random forest on the given samples.
    ///
    /// `n_classes` is the total number of font classes for this character.
    pub fn train(samples: &[Sample], n_classes: usize, seed: u64) -> Self {
        let n = samples.len();
        let n_features_per_split = ((FEAT_LEN as f64).sqrt().ceil() as usize).max(1);
        let mut rng = Rng::new(seed);
        let mut trees = Vec::with_capacity(N_TREES);

        for _ in 0..N_TREES {
            // Bootstrap sample (sample with replacement)
            let mut indices: Vec<usize> = Vec::with_capacity(n);
            for _ in 0..n {
                indices.push(rng.next_usize(n));
            }

            let tree = build_tree(samples, &indices, n_features_per_split, 0, n_classes, &mut rng);
            trees.push(DecisionTree { root: tree });
        }

        RandomForest { trees, n_classes }
    }

    /// Predict class vote proportions for a query feature vector.
    ///
    /// Returns Vec of (class_id, vote_proportion) sorted descending by votes.
    pub fn predict_top_n(&self, query: &[f32; FEAT_LEN], top_n: usize) -> Vec<(u32, f32)> {
        let mut vote_counts: HashMap<u32, f32> = HashMap::new();
        let total_trees = self.trees.len() as f32;

        for tree in &self.trees {
            let leaf_votes = predict_tree(&tree.root, query);
            // Each tree contributes its leaf's class distribution
            let total: f32 = leaf_votes.iter().map(|(_, c)| *c as f32).sum();
            if total > 0.0 {
                for &(class_id, count) in leaf_votes {
                    *vote_counts.entry(class_id).or_insert(0.0) += count as f32 / total;
                }
            }
        }

        // Normalize by number of trees
        let mut results: Vec<(u32, f32)> = vote_counts
            .into_iter()
            .map(|(cid, v)| (cid, v / total_trees))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_n);
        results
    }
}

/// Walk a tree to the leaf and return its vote distribution.
fn predict_tree<'a>(node: &'a Node, query: &[f32; FEAT_LEN]) -> &'a [(u32, u32)] {
    match node {
        Node::Leaf { votes } => votes,
        Node::Split { feature, threshold, left, right } => {
            if query[*feature as usize] <= *threshold {
                predict_tree(left, query)
            } else {
                predict_tree(right, query)
            }
        }
    }
}

/// Build a decision tree recursively.
fn build_tree(
    samples: &[Sample],
    indices: &[usize],
    n_features_per_split: usize,
    depth: usize,
    n_classes: usize,
    rng: &mut Rng,
) -> Node {
    // Count classes in this subset using a flat Vec
    let mut class_counts = vec![0u32; n_classes];
    let mut n_distinct = 0usize;
    for &i in indices {
        let cid = samples[i].class_id as usize;
        if class_counts[cid] == 0 {
            n_distinct += 1;
        }
        class_counts[cid] += 1;
    }

    let n = indices.len();

    // Stopping conditions
    if n < MIN_SAMPLES_SPLIT || n_distinct <= 1 || depth >= MAX_DEPTH {
        return make_leaf_vec(&class_counts);
    }

    // Try to find the best split
    let (best_feature, best_threshold, best_impurity_decrease) =
        find_best_split_fast(samples, indices, n_features_per_split, &class_counts, n_classes, rng);

    if best_impurity_decrease <= 0.0 {
        return make_leaf_vec(&class_counts);
    }

    // Partition indices
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for &i in indices {
        if samples[i].features[best_feature] <= best_threshold {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    // Don't create empty children
    if left_indices.len() < MIN_SAMPLES_LEAF || right_indices.len() < MIN_SAMPLES_LEAF {
        return make_leaf_vec(&class_counts);
    }

    let left = build_tree(samples, &left_indices, n_features_per_split, depth + 1, n_classes, rng);
    let right = build_tree(samples, &right_indices, n_features_per_split, depth + 1, n_classes, rng);

    Node::Split {
        feature: best_feature as u16,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn make_leaf_vec(class_counts: &[u32]) -> Node {
    let mut votes: Vec<(u32, u32)> = class_counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i as u32, c))
        .collect();
    votes.sort_by(|a, b| b.1.cmp(&a.1));
    // Keep only top entries to save memory (leaf doesn't need all 5000 classes)
    votes.truncate(10);
    Node::Leaf { votes }
}

/// Maximum number of threshold candidates to try per feature.
/// Using quantile-based subsampling keeps training fast even with 5000+ samples.
#[allow(dead_code)]
const MAX_THRESHOLDS: usize = 32;

/// Find the best (feature, threshold) split among a random subset of features.
///
/// Uses ExtraTrees-style random threshold selection: for each candidate feature,
/// pick a random threshold between the min and max values. This is O(N) per
/// feature instead of O(N log N) — critical when N=5000 with 1 sample per class.
///
/// The incremental Gini approach requires sorted data which defeats the purpose.
/// Instead, we evaluate each random threshold by a single O(N) pass counting
/// left/right class distributions.
fn find_best_split_fast(
    samples: &[Sample],
    indices: &[usize],
    n_features_per_split: usize,
    parent_counts: &[u32],
    n_classes: usize,
    rng: &mut Rng,
) -> (usize, f32, f32) {
    let n = indices.len();
    let n_f = n as f32;
    let parent_gini = gini_impurity_vec(parent_counts, n_f);

    let mut best_feature = 0usize;
    let mut best_threshold = 0.0f32;
    let mut best_decrease = f32::NEG_INFINITY;

    // Pick random subset of features (Fisher-Yates partial shuffle)
    let mut feature_indices: Vec<usize> = (0..FEAT_LEN).collect();
    let m = n_features_per_split.min(FEAT_LEN);
    for i in 0..m {
        let j = i + rng.next_usize(FEAT_LEN - i);
        feature_indices.swap(i, j);
    }

    // Reusable left-counts vector
    let mut left_counts = vec![0u32; n_classes];
    // Precompute parent sum of squares (for incremental Gini)
    let parent_sum_sq: f64 = parent_counts.iter()
        .map(|&c| (c as f64) * (c as f64))
        .sum();

    for &f_idx in &feature_indices[..m] {
        // Find min and max for this feature
        let mut f_min = f32::INFINITY;
        let mut f_max = f32::NEG_INFINITY;
        for &i in indices {
            let v = samples[i].features[f_idx];
            if v < f_min { f_min = v; }
            if v > f_max { f_max = v; }
        }

        if (f_max - f_min).abs() < 1e-10 {
            continue; // constant feature, no useful split
        }

        // Try N_RANDOM_THRESHOLDS random thresholds
        for _ in 0..N_RANDOM_THRESHOLDS {
            // Random threshold between min and max
            let t = f_min + (f_max - f_min) * (rng.next_u32() as f32 / u32::MAX as f32);

            // Count left class distribution and track sum_sq incrementally.
            // Only zero the entries we actually touch (avoids O(n_classes) per attempt).
            let mut touched: Vec<usize> = Vec::new();
            let mut left_n = 0u32;
            let mut left_sum_sq = 0.0f64;
            let mut right_sum_sq = parent_sum_sq;

            for &i in indices {
                if samples[i].features[f_idx] <= t {
                    let cid = samples[i].class_id as usize;
                    let old_left = left_counts[cid] as f64;
                    let old_right = (parent_counts[cid] - left_counts[cid]) as f64;
                    left_sum_sq += 2.0 * old_left + 1.0;
                    right_sum_sq += -2.0 * old_right + 1.0;
                    if left_counts[cid] == 0 {
                        touched.push(cid);
                    }
                    left_counts[cid] += 1;
                    left_n += 1;
                }
            }

            // Clean up touched entries for reuse
            for &cid in &touched {
                left_counts[cid] = 0;
            }

            let right_n = n as u32 - left_n;
            if (left_n as usize) < MIN_SAMPLES_LEAF || (right_n as usize) < MIN_SAMPLES_LEAF {
                continue;
            }

            let left_gini = 1.0 - left_sum_sq / ((left_n as f64) * (left_n as f64));
            let right_gini = 1.0 - right_sum_sq / ((right_n as f64) * (right_n as f64));

            let decrease = parent_gini as f64
                - (left_n as f64 / n_f as f64) * left_gini
                - (right_n as f64 / n_f as f64) * right_gini;

            if decrease > best_decrease as f64 {
                best_decrease = decrease as f32;
                best_feature = f_idx;
                best_threshold = t;
            }
        }
    }

    (best_feature, best_threshold, best_decrease)
}

/// Gini impurity from a flat count Vec: 1 - Σ(p_i²)
#[inline]
fn gini_impurity_vec(counts: &[u32], n: f32) -> f32 {
    if n <= 0.0 {
        return 0.0;
    }
    let inv_n = 1.0 / n;
    let mut sum_sq = 0.0f32;
    for &c in counts {
        if c > 0 {
            let p = c as f32 * inv_n;
            sum_sq += p * p;
        }
    }
    1.0 - sum_sq
}

// ---------------------------------------------------------------------------
// FontClassifier — high-level API
// ---------------------------------------------------------------------------

impl FontClassifier {
    /// Train classifiers for all characters from char index entries.
    ///
    /// Training is parallelized across characters using rayon.
    pub fn train(entries: &HashMap<char, Vec<crate::char_index::FontCharEntry>>) -> Self {
        use rayon::prelude::*;

        // Collect characters that have enough entries
        let chars: Vec<char> = entries
            .keys()
            .filter(|c| entries.get(c).map(|e| e.len()).unwrap_or(0) >= 2)
            .copied()
            .collect();

        // Train each character's model in parallel
        let results: Vec<(char, RandomForest, Vec<String>)> = chars
            .par_iter()
            .filter_map(|c| {
                let font_entries = entries.get(c)?;

                // Build class mapping: font_name → class_id
                let mut name_to_id: HashMap<&str, u32> = HashMap::new();
                let mut names: Vec<String> = Vec::new();

                for e in font_entries {
                    if !name_to_id.contains_key(e.font_name.as_str()) {
                        let id = names.len() as u32;
                        name_to_id.insert(&e.font_name, id);
                        names.push(e.font_name.clone());
                    }
                }

                let n_classes = names.len();

                // Build training samples
                let samples: Vec<Sample> = font_entries
                    .iter()
                    .map(|e| Sample {
                        features: e.features.as_slice(),
                        class_id: *name_to_id.get(e.font_name.as_str()).unwrap(),
                    })
                    .collect();

                // Use character code as part of seed for reproducibility
                let seed = (*c as u64).wrapping_mul(0x517cc1b727220a95);
                let forest = RandomForest::train(&samples, n_classes, seed);

                Some((*c, forest, names))
            })
            .collect();

        let mut models = HashMap::with_capacity(results.len());
        let mut class_names_map = HashMap::with_capacity(results.len());
        for (c, forest, names) in results {
            models.insert(c, forest);
            class_names_map.insert(c, names);
        }

        FontClassifier {
            models,
            class_names: class_names_map,
        }
    }

    /// Predict font candidates for a single character's feature vector.
    ///
    /// Returns font names with confidence scores, sorted descending.
    pub fn predict(
        &self,
        c: char,
        features: &[f32; FEAT_LEN],
        top_n: usize,
    ) -> Vec<(String, f32)> {
        let forest = match self.models.get(&c) {
            Some(f) => f,
            None => return Vec::new(),
        };
        let names = match self.class_names.get(&c) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let raw = forest.predict_top_n(features, top_n);
        raw.into_iter()
            .filter_map(|(class_id, score)| {
                names.get(class_id as usize).map(|name| (name.clone(), score))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Serialization — compact binary format
// ---------------------------------------------------------------------------

impl FontClassifier {
    /// Serialize to bytes for caching alongside the char index.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic + version
        buf.extend_from_slice(b"MLFC"); // ML Font Classifier
        buf.extend_from_slice(&1u32.to_le_bytes()); // version

        // Number of characters with models
        buf.extend_from_slice(&(self.models.len() as u32).to_le_bytes());

        for (c, forest) in &self.models {
            buf.extend_from_slice(&(*c as u32).to_le_bytes());

            // Class names for this char
            let names = self.class_names.get(c).map(|n| n.as_slice()).unwrap_or(&[]);
            buf.extend_from_slice(&(names.len() as u32).to_le_bytes());
            for name in names {
                let bytes = name.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }

            // Forest
            buf.extend_from_slice(&(forest.trees.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(forest.n_classes as u32).to_le_bytes());
            for tree in &forest.trees {
                serialize_node(&tree.root, &mut buf);
            }
        }

        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("too small".to_string());
        }
        if &data[0..4] != b"MLFC" {
            return Err("bad magic".to_string());
        }
        let mut pos = 4;
        let version = read_u32_at(data, &mut pos)?;
        if version != 1 {
            return Err(format!("unsupported version {version}"));
        }

        let n_chars = read_u32_at(data, &mut pos)? as usize;
        let mut models = HashMap::with_capacity(n_chars);
        let mut class_names_map = HashMap::with_capacity(n_chars);

        for _ in 0..n_chars {
            let c = char::from_u32(read_u32_at(data, &mut pos)?)
                .ok_or_else(|| "invalid char".to_string())?;

            // Class names
            let n_names = read_u32_at(data, &mut pos)? as usize;
            let mut names = Vec::with_capacity(n_names);
            for _ in 0..n_names {
                let len = read_u32_at(data, &mut pos)? as usize;
                if pos + len > data.len() {
                    return Err("truncated name".to_string());
                }
                names.push(String::from_utf8_lossy(&data[pos..pos + len]).to_string());
                pos += len;
            }

            // Forest
            let n_trees = read_u32_at(data, &mut pos)? as usize;
            let n_classes = read_u32_at(data, &mut pos)? as usize;
            let mut trees = Vec::with_capacity(n_trees);
            for _ in 0..n_trees {
                let root = deserialize_node(data, &mut pos)?;
                trees.push(DecisionTree { root });
            }

            models.insert(c, RandomForest { trees, n_classes });
            class_names_map.insert(c, names);
        }

        Ok(FontClassifier {
            models,
            class_names: class_names_map,
        })
    }
}

fn serialize_node(node: &Node, buf: &mut Vec<u8>) {
    match node {
        Node::Split { feature, threshold, left, right } => {
            buf.push(0x01); // split tag
            buf.extend_from_slice(&feature.to_le_bytes());
            buf.extend_from_slice(&threshold.to_le_bytes());
            serialize_node(left, buf);
            serialize_node(right, buf);
        }
        Node::Leaf { votes } => {
            buf.push(0x00); // leaf tag
            buf.extend_from_slice(&(votes.len() as u32).to_le_bytes());
            for &(class_id, count) in votes {
                buf.extend_from_slice(&class_id.to_le_bytes());
                buf.extend_from_slice(&count.to_le_bytes());
            }
        }
    }
}

fn deserialize_node(data: &[u8], pos: &mut usize) -> Result<Node, String> {
    if *pos >= data.len() {
        return Err("truncated node".to_string());
    }
    let tag = data[*pos];
    *pos += 1;

    match tag {
        0x01 => {
            // Split
            if *pos + 6 > data.len() {
                return Err("truncated split".to_string());
            }
            let feature = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            let threshold = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            let left = deserialize_node(data, pos)?;
            let right = deserialize_node(data, pos)?;
            Ok(Node::Split {
                feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        0x00 => {
            // Leaf
            let n = read_u32_at(data, pos)? as usize;
            let mut votes = Vec::with_capacity(n);
            for _ in 0..n {
                let class_id = read_u32_at(data, pos)?;
                let count = read_u32_at(data, pos)?;
                votes.push((class_id, count));
            }
            Ok(Node::Leaf { votes })
        }
        _ => Err(format!("unknown node tag 0x{tag:02x}")),
    }
}

fn read_u32_at(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".to_string());
    }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_forest() {
        // Create simple 2-class problem
        let mut samples = Vec::new();
        for i in 0..50 {
            let mut f = [0.0f32; FEAT_LEN];
            f[0] = i as f32 / 50.0;
            f[1] = 0.2;
            samples.push(Sample { features: f, class_id: 0 });
        }
        for i in 0..50 {
            let mut f = [0.0f32; FEAT_LEN];
            f[0] = 0.5 + i as f32 / 100.0;
            f[1] = 0.8;
            samples.push(Sample { features: f, class_id: 1 });
        }

        let forest = RandomForest::train(&samples, 2, 42);
        assert_eq!(forest.trees.len(), N_TREES);

        // Query near class 0
        let mut q = [0.0f32; FEAT_LEN];
        q[0] = 0.1;
        q[1] = 0.2;
        let preds = forest.predict_top_n(&q, 2);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 0); // should predict class 0

        // Query near class 1
        q[0] = 0.9;
        q[1] = 0.8;
        let preds = forest.predict_top_n(&q, 2);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 1); // should predict class 1
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut samples = Vec::new();
        for i in 0..20 {
            let mut f = [0.0f32; FEAT_LEN];
            f[0] = i as f32 / 20.0;
            samples.push(Sample { features: f, class_id: (i % 3) as u32 });
        }

        let forest = RandomForest::train(&samples, 3, 123);
        let tree = &forest.trees[0];
        let mut buf = Vec::new();
        serialize_node(&tree.root, &mut buf);
        let mut pos = 0;
        let restored = deserialize_node(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());

        // Verify predictions match
        let mut q = [0.0f32; FEAT_LEN];
        q[0] = 0.5;
        let orig = predict_tree(&tree.root, &q);
        let rest = predict_tree(&restored, &q);
        assert_eq!(orig, rest);
    }
}
