# Learned Classifier Research: Replacing or Augmenting Fisher-Weighted Euclidean Distance

**Date:** 2026-06-13  
**Status:** Research complete, ready for implementation decisions

---

## Table of Contents

1. [Problem Statement & Current System](#1-problem-statement--current-system)
2. [Error Taxonomy: What's Actually Broken](#2-error-taxonomy-whats-actually-broken)
3. [Approach 1: Metric Learning (Mahalanobis / LMNN / NCA)](#3-approach-1-metric-learning)
4. [Approach 2: Lightweight Classifiers (GBT / Random Forest / SVM / MLP)](#4-approach-2-lightweight-classifiers)
5. [Approach 3: Siamese / Triplet Networks](#5-approach-3-siamese--triplet-networks)
6. [Approach 4: Contrastive Learning on Raw Glyph Images](#6-approach-4-contrastive-learning-on-raw-glyph-images)
7. [Approach 5: Hybrid Re-Ranker](#7-approach-5-hybrid-re-ranker)
8. [Inference Runtime: Rust Deployment Options](#8-inference-runtime-rust-deployment-options)
9. [Training Data & Pipeline](#9-training-data--pipeline)
10. [Comparison Matrix](#10-comparison-matrix)
11. [Recommendation: What to Build First](#11-recommendation-what-to-build-first)

---

## 1. Problem Statement & Current System

### How it works today

Unscan's character index (CI) identifies fonts in scanned PDFs using a
three-stage pipeline:

1. **Index build** (~5K fonts × ~106 chars): Render each character at
   NORM_H=48px, compute a 100-dimensional feature vector (32-bin column
   profile, 7 scalar v1, 18 scalar v2, 32-bin row profile, 11 scalar v3).
   Pre-multiply each dimension by its Fisher discriminant weight (√(between-font
   variance / within-font-across-DPI variance), normalized to sum=1). Store as
   flat arrays per character.

2. **Query** (per scanned line): Segment characters from the longest words,
   compute the same 100-dim features, pre-multiply by Fisher weights, do a
   brute-force linear scan to find the nearest neighbor for each character.

3. **Aggregation**: Each character votes for a font. The font with the best
   weighted geometric mean of log-distances across all characters wins.

### Current accuracy

On the 30-font, 6-page specimen test:

| Category | Count | % of classified |
|---|---|---|
| Hit | 346 | 72.4% |
| Minor miss | 56 | 11.7% |
| Major miss | 74 | 15.5% |
| SSIM failure | 4 | 0.8% |
| Kept raster | 5 | — |
| No ground truth | 5 | — |

**Overall: 346/476 = 72.7% exact match** (or 402/476 = 84.5% if minor
misses are counted as acceptable).

### Why the current approach has a ceiling

The Fisher-weighted Euclidean distance assumes:

- **Linear separability**: Each feature dimension contributes independently.
  Feature interactions (e.g., serif_score × stroke_contrast distinguishing
  Old Style from Transitional serifs) are invisible.

- **Diagonal covariance**: The metric is `Σ w_i (x_i - y_i)²`. This is a
  diagonal Mahalanobis distance — it can scale dimensions but can't rotate
  the feature space. Two fonts that differ along a diagonal of the original
  space look close when measured axis-by-axis.

- **Global weights**: The same Fisher weights apply to every character. But
  the discriminative features for 'e' (counter openness, x-height) are very
  different from those for 'l' (stroke width, serif shape) or 'W' (diagonal
  strokes, width proportions).

- **DPI-noise-optimized, not discrimination-optimized**: Fisher weights
  maximize signal/noise where noise = within-font across-DPI variance. This
  penalizes DPI-unstable features but doesn't directly optimize for
  correct font identification.

---

## 2. Error Taxonomy: What's Actually Broken

Before researching classifiers, we need to understand which errors a
learned approach can actually fix. Analysis of the enriched audit
(`/tmp/audit-enriched/audit.json`) reveals three distinct failure modes:

### Category A: Font Identity Aliasing (45 of 130 misses)

The CI's #1 candidate IS the correct font family, but the PostScript name
doesn't match the ground truth because of static ↔ variable font aliasing.

| Pattern | Count | Example |
|---|---|---|
| SourceSerif4-It → SourceSerif4 Italic[opsz,wght] | 32 | Variable font wins over static |
| IBMPlexSerif → IBMPlexSerif Text | 6 | Optical size variant |
| SourceSerif4-Regular → SourceSerif4[opsz,wght] | 4 | Variable font wins |
| OpenSans → open sans 400 | 3 | Different packagings |

**A learned classifier cannot fix these.** The glyphs are pixel-identical
at the same optical size. The fix is font family grouping (treating
SourceSerif4-It and SourceSerif4 Italic[opsz,wght] as equivalents), which
is a font catalog / identity layer problem, not a classification problem.

### Category B: Missing Fonts (40+ of 130 misses)

The ground truth font is not installed on the system. The CI picks the
closest available substitute.

| Pattern | Count | Example |
|---|---|---|
| CourierNewPSMT → Courier | 12 | Commercial Courier New not installed |
| PrestigeEliteNormal → Prestige Elite Std | 6 | Different edition |
| TimesNewRomanPSMT → Times-Roman / NimbusRoman | 1 | Commercial TNR not installed |
| ArialMT → Roboto Regular | 1 | Commercial Arial not installed |

**A learned classifier cannot fix these either.** The font literally isn't
in the index. The substitutes are reasonable (Courier for Courier New,
NimbusRoman for Times New Roman). These should arguably be reclassified as
"acceptable substitutions" rather than failures.

### Category C: True Discrimination Failures (~45 of 130 misses)

These are the cases where the CI picks a genuinely wrong font from a
different family. A better classifier could potentially fix them.

| Pattern | Count | Root cause |
|---|---|---|
| Lato-Regular → Lato Medium | 8 | Weight discrimination failure |
| SourceSans3-Roman → SourceSans3 Light | 8 | Weight discrimination failure |
| SourceSerif4-It → SourceSerif4SmText It | 6 | Optical size confusion |
| SourceSerif4-Regular → SourceSerif4SmText Regular | 2 | Optical size confusion |
| NotoSerif-Bold → IBMPlexSerif Bold / Caladea Bold | 2 | Cross-family confusion |
| Lato-Bold → Carlito Bold | 1 | Cross-family confusion |
| Lato-Italic → Lato MediumItalic | 2 | Weight + style confusion |
| CourierNewPS-BoldMT → TeX Gyre Cursor | 1 | Monospace cross-family |

The actionable target is ~45 entries, or roughly **9.5% of classified
lines**. Fixing all of these would bring accuracy from 84.5% to 94.0%.

### Key insight: tight margins

For the true discrimination failures, the distance margins are razor-thin.
The median score difference between the chosen (wrong) font and the correct
font is **0.0000** — the features are so similar that floating-point
ordering decides the winner. This is exactly the regime where a learned
non-linear decision boundary could help.

---

## 3. Approach 1: Metric Learning

### Mahalanobis Distance (Full Matrix)

**Concept**: Replace the diagonal weight matrix W with a full positive
semi-definite matrix M, so distance = (x-y)ᵀ M (x-y). This captures
cross-dimensional correlations. The current Fisher weights are the
diagonal of M.

**Training**: Gradient descent on M to minimize a loss that pulls
same-font pairs closer and pushes different-font pairs apart. Standard
formulations:

- **LMNN (Large Margin Nearest Neighbor)**: For each training point,
  identify its k target neighbors (same class) and competing impostors
  (different class, currently within margin). Optimize M so target
  neighbors are closer than impostors by a margin. Convex in M.

- **NCA (Neighbourhood Components Analysis)**: Directly optimize
  leave-one-out kNN classification accuracy in the learned metric space.
  Non-convex but smooth; differentiable w.r.t. M.

- **ITML (Information-Theoretic Metric Learning)**: Learn M by minimizing
  KL divergence from a prior (typically I) subject to distance constraints
  from known same/different pairs. Efficient Bregman projection.

**How it works with our features**: Each of the 100 features becomes a
dimension. The learned M = LᵀL where L ∈ ℝ^{d×100} projects features into
a space where Euclidean distance is meaningful. When d < 100, this also
does dimensionality reduction.

**Training data**: For each character type (e.g., 'a'), we have ~5K
feature vectors (one per font). Training pairs are:
- **Same class**: Same font rendered at different DPIs (from the exhaustive
  renderer). This teaches the metric to be DPI-invariant.
- **Different class**: Different fonts at any DPI. We have millions of
  negative pairs.

**Inference speed**: If M is dense 100×100, the distance computation goes
from 100 multiply-adds (diagonal) to 100×100 = 10K multiply-adds. This is
~100× slower per distance computation. With ~5K fonts × ~15 characters per
line, that's 5K × 15 × 10K = 750M operations per line — **too slow** for
a straightforward implementation.

**Mitigation**: Pre-compute L·x for all index entries (one-time cost at
build). Then distance becomes Euclidean in the projected space:
||L·x - L·y||². This restores the current speed profile: only the query
vector needs projection (one 100×100 matmul per character), then standard
brute-force in the projected space. If L projects to d < 100 dimensions,
it's actually **faster** than today.

**Implementation complexity**:
- Training: Python with `metric_learn` library or custom PyTorch. ~200 LOC.
- Rust inference: Store the projection matrix L (100×100 f32 = 40KB).
  Apply L to query features before brute-force search. ~50 LOC change.
- Per-character model vs global: Ideally learn one M per character type
  (since discriminative axes differ by character). That's 106 matrices ×
  40KB = 4.2MB. Very manageable.

**Expected accuracy improvement**: Moderate. Full Mahalanobis can capture
the correlations that diagonal Fisher misses, but it's still a linear
transform. Won't help with fundamentally non-linear boundaries (e.g., a
font that matches in profile but diverges in stroke width only for certain
character shapes).

**Font family grouping**: LMNN can be trained with grouped labels (treat
SourceSerif4-It and SourceSerif4 Italic[opsz,wght] as the same class).
This directly addresses Category A misses in the training signal.

**Verdict**: Low risk, moderate reward. Clean upgrade path from current
Fisher weights. Implementation is straightforward.

---

### LMNN Specifically

LMNN is the strongest candidate in this family because:

1. It directly optimizes kNN accuracy (which is our inference method)
2. It's convex (guaranteed convergence to global optimum for a given k)
3. It handles the multi-class setting naturally
4. It produces a PSD matrix M that can be factored into a projection L

The key LMNN objective:

```
min_M  Σ_i Σ_{j∈targets(i)} d_M(x_i, x_j)           # pull targets close
       + λ Σ_i Σ_{j∈targets(i)} Σ_{l: different class}
           [1 + d_M(x_i, x_j) - d_M(x_i, x_l)]_+    # push impostors away
```

For our problem:
- x_i = feature vector for character 'a' rendered in font F at DPI d
- targets(i) = same font F rendered at other DPIs
- impostors = different fonts rendered at any DPI

With ~5K fonts × 3 DPIs × 106 chars, the training set is ~1.6M vectors.
LMNN on 100-dim data with 1.6M points is tractable (minutes on a modern
CPU).

---

## 4. Approach 2: Lightweight Classifiers

### The Core Problem: Class Count

With ~5K font files (producing ~5K+ font keys with OT variants), this is a
**5000+ class classification problem** with exactly **one canonical training
sample per class per character** (the rendered glyph). DPI augmentation
gives us 3–5 samples per class, but that's still extremely few.

This is why the existing `ml-classifier-design.md` concluded that standard
classifiers are "fundamentally underdetermined" for this problem.

However, the exhaustive training data generator (being built separately)
changes this calculus. With multi-DPI, multi-AA, and multi-downsample
augmentation, we'll have **50–200 training samples per class per character**.
That's enough for some approaches.

### Random Forest

**How it works**: Ensemble of decision trees, each trained on a bootstrap
sample of the data. At each node, split on the feature/threshold that
maximizes information gain (or Gini impurity reduction).

**With 100 features**: Trees naturally capture feature interactions without
explicit polynomial features. A tree can learn "if serif_score > 0.3 AND
stroke_contrast > 0.5, then this is a Transitional serif" — exactly the
non-linear interactions Fisher weights miss.

**Training**: scikit-learn, 100 trees, max_depth=20–30. With 5K classes ×
100 samples × 106 characters, that's 53M training samples total. Per
character (500K samples, 5K classes): feasible but slow.

**Inference speed**:
- Per sample: traverse 100 trees of depth 20 = 2000 comparisons. At ~1ns
  each = **~2µs per sample**. Competitive with brute-force kNN.
- But: we can't pre-filter by character. For each scanned character, we'd
  classify it directly — no need to scan all 5K entries.
- **This is a fundamental advantage**: RF gives O(depth × n_trees) per
  query regardless of catalog size, vs O(n_fonts × feat_dim) for kNN.

**Rust inference**: Decision tree inference is trivial to implement in
Rust — just if/else chains. Serialize the tree structure (node array with
feature_index, threshold, left/right child). The `smartcore` crate has
RF inference, or implement from scratch in ~200 LOC.

Alternatively: export to ONNX via `skl2onnx`, run via `ort` crate.

**Expected accuracy improvement**: High for the Category C discrimination
failures. RF naturally handles the weight-confusion cases (Lato-Regular
vs Lato-Medium) by learning threshold boundaries on mean_stroke_width
conditioned on other features.

**Font family grouping**: Straightforward — group labels during training.
All SourceSerif4 variants get the same label.

**Downsides**:
- 5000+ classes means each tree's leaf structure is enormous. Memory for
  the model could be 100MB+.
- Training 106 separate RF models (one per character) is conceptually clean
  but doubles the model size.
- Voting across characters requires mapping RF predictions back to a common
  font namespace.

### Gradient Boosted Trees (XGBoost / LightGBM)

**How it works**: Sequentially fit trees to the residuals of the previous
ensemble. Each tree is smaller (depth 4–8) but the ensemble learns complex
boundaries through boosting.

**Advantages over RF for this problem**:
- Handles class imbalance better (some fonts have more variants)
- Feature importance is more interpretable
- Generally higher accuracy than RF on structured data

**Training**: LightGBM is faster for large datasets. `lightgbm` Python
package. With 5K classes, use multi-class softmax objective.

**Inference speed**: Similar to RF — ~2–5µs per sample for 200 trees of
depth 6.

**Rust inference**: `lightgbm3-rs` crate provides Rust bindings to the
LightGBM C library. Or export to ONNX. Or implement tree traversal from
scratch (the model is just a collection of thresholds).

**Expected accuracy**: Slightly higher than RF. GBT consistently
outperforms RF on tabular data in benchmarks.

**Downsides**:
- LightGBM's C library adds a build dependency (~20MB binary size increase)
- Multi-class with 5K classes is expensive to train (hours)
- The model may not generalize well to new fonts not seen during training

### SVM (RBF Kernel)

**How it works**: Find the maximum-margin hyperplane (in kernel space) that
separates classes. RBF kernel maps to infinite dimensions, capturing
non-linear boundaries.

**With 5K classes**: Multi-class SVM via one-vs-one requires K(K-1)/2 =
~12.5M binary classifiers. **Not feasible.** One-vs-rest requires 5K
classifiers, which is slow but possible.

**Inference speed**: For one-vs-rest with ~100 support vectors per
classifier: 5K × 100 × kernel_eval = 500K kernel evaluations. At ~10ns
each = **5ms per sample**. Too slow for per-character inference.

**Verdict**: **Not recommended.** SVM doesn't scale to 5K+ classes.

### Small MLP (Multi-Layer Perceptron)

**How it works**: Feed-forward neural network with 1–3 hidden layers.
Input: 100-dim feature vector. Output: 5K-dim softmax over fonts.

**Architecture**: 100 → 256 → 256 → 5K. Total parameters:
100×256 + 256×256 + 256×5K = 25.6K + 65.5K + 1.28M = **~1.37M parameters**.
At 4 bytes each = **5.5MB model size**. Very manageable.

**Training**: Standard cross-entropy loss. Adam optimizer. With 50M+
training samples and 1.4M parameters, this is heavily overparameterized
for the data — good for generalization.

**Inference speed**: 100→256 = 25.6K MACs, 256→256 = 65.5K MACs,
256→5K = 1.28M MACs. Total: **~1.37M MACs**. On a modern CPU at ~10 GFLOPS
sustained = **~0.14ms per sample**. With 15 characters per line =
**~2ms per line**. Acceptable.

**Rust inference**: Trivial to implement from scratch — matrix multiply
+ ReLU. No external dependency needed. ~100 LOC.

```rust
fn mlp_forward(input: &[f32; 100], w1: &[[f32; 100]; 256],
               w2: &[[f32; 256]; 256], w3: &[[f32; 256]; N_FONTS],
               b1: &[f32; 256], b2: &[f32; 256], b3: &[f32; N_FONTS])
               -> Vec<(usize, f32)> {
    // Layer 1: 100 → 256 + ReLU
    // Layer 2: 256 → 256 + ReLU
    // Layer 3: 256 → N_FONTS (logits, take top-K)
    // Return sorted (font_id, score) pairs
}
```

**Expected accuracy**: High. MLPs on tabular data with abundant training
samples and moderate feature counts regularly achieve 90%+ accuracy.
The non-linear hidden layers can capture feature interactions that
diagonal metrics miss.

**Font family grouping**: Merge labels during training. Or add a
hierarchical loss that penalizes cross-family errors more than
within-family errors.

**New font handling**: **This is the critical weakness.** Adding a new font
to the system requires retraining the MLP (the output layer has a fixed
node per font). This is fundamentally different from kNN, where a new font
is just a new entry in the index.

**Verdict**: Good accuracy but the fixed-class problem is a serious
architectural limitation for a tool that must work with any installed fonts.

---

## 5. Approach 3: Siamese / Triplet Networks

### Concept

Instead of classifying directly into font IDs, learn an **embedding
function** f: ℝ¹⁰⁰ → ℝᵈ that maps feature vectors into a space where
same-font glyphs cluster together and different-font glyphs are separated.

At inference, embed both the query and all index entries, then do kNN
in the embedding space — exactly like today, but with a learned embedding
instead of Fisher-weighted features.

### Triplet Loss Training

For each training step, sample a triplet (anchor, positive, negative):
- **Anchor**: Character 'a' rendered in Font F at DPI d₁
- **Positive**: Character 'a' rendered in Font F at DPI d₂
- **Negative**: Character 'a' rendered in Font G at DPI d₃

Loss = max(0, d(f(anchor), f(positive)) - d(f(anchor), f(negative)) + margin)

The network learns to project same-font glyphs closer than different-font
glyphs by at least `margin`.

### Architecture on Feature Vectors

Input: 100-dim feature vector. Network: 100 → 128 → 64 → 32 embedding.

This is essentially a small MLP used as an embedding function rather than
a classifier. Total parameters: 100×128 + 128×64 + 64×32 = 12.8K + 8.2K
+ 2K = **~23K parameters**. Tiny model, ~92KB.

### Architecture on Raw Glyph Images

Input: 48 × W grayscale image (W varies per character). Network: Small CNN.

```
Conv(1→16, 3×3) → ReLU → MaxPool(2×2)    # 24×W/2×16
Conv(16→32, 3×3) → ReLU → MaxPool(2×2)   # 12×W/4×32
Conv(32→64, 3×3) → ReLU → AdaptiveAvgPool(4×4)  # 4×4×64
Flatten → 1024 → 128 → 64 embedding
```

Total parameters: ~150K. Model size: ~600KB.

This approach **bypasses hand-crafted features entirely**. The CNN learns
its own features from the raw glyph images. It can potentially discover
features our hand-crafted set misses (e.g., subtle terminal shapes,
bracket curvature, junction geometry).

### Key Advantage: Open-Set Recognition

Unlike MLP classification, the embedding approach handles new fonts
naturally:

1. New font installed → render all characters → compute embeddings → add
   to index. **No retraining needed.**
2. At query time, kNN in embedding space works with any number of fonts.
3. The network generalizes: it learns "what makes fonts similar" rather
   than "what is font #3721."

This is the same architecture used by WhatTheFont (33M training images,
130K fonts, ~90% accuracy) and Adobe's font matching patent
(US10515295B2).

### Training Data Requirements

Triplet mining is critical. With 5K fonts × 106 chars × 3 DPIs = 1.6M
vectors, there are ~1.6M × 1.6M possible triplets. Hard negative mining
(selecting negatives that are close to the anchor) is essential for
efficient training.

The exhaustive renderer provides exactly what's needed:
- Multiple DPI/AA variants per font give natural positives
- Hard negatives are fonts from the same serif/sans/mono class

### Inference Speed

**Feature-based (100→32 embedding)**: 100×128 + 128×64 + 64×32 = 23K MACs
= **~2.3µs per character**. Then kNN in 32-dim space against 5K entries =
5K × 32 = 160K MACs = **16µs per character**. Total: ~18µs.

Compare to current: kNN in 100-dim against 5K entries = 500K MACs = ~50µs.
**The embedding approach is actually 2.7× faster** because the embedding
dimension (32) is smaller than the original feature dimension (100).

**Image-based CNN**: ~150K MACs for the forward pass = ~15µs. Plus kNN =
16µs. Total: ~31µs. Still faster than current.

### Variable ↔ Static Font Problem

This is where triplet/contrastive learning shines. During training,
SourceSerif4-It and SourceSerif4 Italic[opsz,wght] are treated as the
**same class**. The network learns that their embeddings should be close.
This directly solves Category A misses (font identity aliasing).

More broadly, the network can learn a continuous "font style space" where:
- All weights of Source Serif 4 form a smooth trajectory
- Variable and static instances land at the same point
- Optical size variants (SmText, Subhead) are nearby but distinguishable

### Rust Implementation

**Feature-based**: Pure Rust, no dependencies. Matrix multiply + ReLU.

```rust
struct FontEmbedder {
    w1: [[f32; 100]; 128],
    b1: [f32; 128],
    w2: [[f32; 128]; 64],
    b2: [f32; 64],
    w3: [[f32; 64]; 32],
    b3: [f32; 32],
}

impl FontEmbedder {
    fn embed(&self, features: &[f32; 100]) -> [f32; 32] {
        // Three layers of matmul + ReLU + L2 normalize
    }
}
```

**Image-based CNN**: Use `ort` crate (ONNX Runtime Rust bindings) or
`tract` (pure Rust ONNX inference). Both are mature:
- `ort`: 1.25M downloads/month, supports AVX2/SSE auto-vectorization
- `tract`: Used in production at Sonos, pure Rust, no C dependencies

Model size ~600KB as ONNX. Inference ~15µs per glyph image on CPU.

### Verdict

**Best overall approach for unscan.** Combines:
- Non-linear feature transformation (captures interactions)
- Open-set recognition (new fonts without retraining)
- Natural font family grouping (triplet training with grouped labels)
- Smaller embedding = faster kNN
- Straightforward Rust deployment

---

## 6. Approach 4: Contrastive Learning on Raw Glyph Images

### Concept

Skip hand-crafted features entirely. Train a CNN or Vision Transformer
directly on normalized glyph images (48 × W grayscale) with a contrastive
loss. The network learns both the features AND the metric simultaneously.

### Differences from Approach 3

| Aspect | Approach 3 (Triplet on Features) | Approach 4 (Contrastive on Images) |
|---|---|---|
| Input | 100-dim hand-crafted features | 48×W grayscale image |
| Features | Hand-designed, may miss patterns | Learned, can discover novel features |
| Training data | Feature vectors (fast to compute) | Raw images (must render or cache) |
| Model size | ~23K params (92KB) | ~150K+ params (600KB+) |
| Risk | Limited by feature quality | Higher complexity, harder to debug |

### Architecture Options

**Small CNN** (recommended for glyph images):
```
Input: 48 × W × 1 (grayscale, W varies)
Conv(1→16, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
Conv(16→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
Conv(32→64, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool(4×4)
Flatten(1024) → Linear(1024→128) → ReLU → Linear(128→64)
L2 normalize → 64-dim embedding
```

**MobileNetV2-tiny** (if more capacity needed):
- Use the first 3 blocks of MobileNetV2 (width multiplier 0.25)
- ~50K parameters, ~200KB model
- Pre-trained on ImageNet features transfer surprisingly well to glyphs

**ViT-tiny** (latest research direction):
- Patch size 8×8, 4 layers, 4 heads, dim 128
- ~500K parameters, ~2MB model
- Handles variable-width inputs naturally via position embedding
- Likely overkill for 48px glyph images

### Loss Functions

**Supervised Contrastive Loss** (best for font classification, per recent
research [Appl. Sci. 2023, 13(6), 3635]):
- Outperforms both triplet loss and NT-Xent on font style classification
- Uses labels to form positive/negative pairs within mini-batches
- Produces the most separated embedding clusters

**NT-Xent** (SimCLR-style):
- Self-supervised: augmented views of the same glyph are positives
- Augmentations: random DPI scaling, Gaussian blur, slight rotation
- Advantage: doesn't need font labels at pre-training time
- Can be fine-tuned with labels later

### Training Pipeline

```
1. Render: For each (font, char, dpi, aa_mode), produce 48×W image
2. Cache: Store all images as memory-mapped arrays (~50GB for 5K fonts)
3. Sample: Mine hard triplets / construct contrastive mini-batches
4. Train: PyTorch, 50-100 epochs, AdamW lr=1e-4
5. Export: ONNX format for Rust inference
6. Validate: Run against specimen PDF ground truth
```

### What the CNN can learn that hand-crafted features miss

- **Bracket curvature**: The transition from serif to stem. Old Style serifs
  have deep, curved brackets; Modern serifs have abrupt, square brackets.
  Our current features don't capture this.

- **Junction geometry**: Where strokes meet (e.g., the crotch of 'V', the
  vertex of 'A'). Varies significantly between typeface traditions.

- **Ink traps**: Small notches at tight junctions designed to prevent ink
  spread at small sizes. Present in text-optimized faces (e.g., IBMPlex),
  absent in display faces.

- **Terminal shapes**: Ball terminals (Times), teardrop terminals
  (Garamond), flat-cut terminals (Futura). Our terminal_angles feature
  captures direction but not shape.

- **Proportional spacing patterns**: The relative widths of characters
  within a font. This is lost when we normalize each character
  independently.

### Expected Accuracy

Based on literature:
- DeepFont (2015): ~80% top-1 accuracy on 2,383 fonts from web images
- Contrastive learning on fonts (2023): 95%+ on 10-font classification
- CalliNet (triplet, calligraphy): 94-99% accuracy

For our cleaner, controlled-rendering setup: **expect 90-95% top-1
accuracy** on the full 5K font catalog, significantly above the current
84.5%.

### Font Family Grouping

Same as triplet approach — group labels during training.

### Verdict

**Highest accuracy ceiling but also highest implementation effort.** The
CNN can discover features we haven't thought of, but it requires:
- A substantial training pipeline (rendering, caching, training loop)
- ONNX Runtime as a dependency (adds ~20MB to binary size)
- More complex debugging when results are wrong

Recommended as a **Phase 2** improvement after proving the concept with
feature-based triplet networks (Approach 3).

---

## 7. Approach 5: Hybrid Re-Ranker

### Concept

Keep the current brute-force kNN as a **first-stage retrieval** (fast,
recall-oriented). Add a learned model as a **second-stage re-ranker** that
re-scores the top-K candidates with a more powerful model.

### Architecture

```
Stage 1 (unchanged): Brute-force kNN → top-K (K=10-50) candidates
Stage 2 (new): For each of K candidates, compute:
  - Feature difference: |query_features - candidate_features|
  - Feature product: query_features * candidate_features
  - Candidate metadata: serif_class, weight_bucket, is_variable
  Input: concatenation → 200-300 dim vector
  Model: Small MLP (300 → 64 → 1) → re-ranking score
  Output: Re-ranked top-K
```

### Why This Works

The re-ranker sees the **relationship between query and candidate** rather
than just the query in isolation. It can learn:
- "When the query's stroke_width is 0.12 and the candidate's is 0.14,
  that's a Regular→Medium confusion — look more carefully at serif_score
  and ink_density to break the tie."
- "When both query and candidate have high serif_score but different
  counter_area, check if the candidate is an optical size variant."

### Training

For each (query, correct_font) pair from our ground truth:
- Positive examples: (query_features, correct_font_features) → label 1
- Negative examples: (query_features, wrong_font_features from top-K) → 0
- Train binary classifier with cross-entropy loss

This can be trained on the enriched audit data directly (we have the
exact character features and top-K candidates for each miss).

### Inference Speed

Stage 1: Same as today (~50µs per character × 15 chars = ~0.75ms per line)
Stage 2: K × forward_pass = 50 × (~5µs) = **0.25ms per line**
Total: **~1ms per line** — negligible overhead.

### Implementation Complexity

- Training: Python, ~100 LOC
- Rust inference: Tiny MLP (~300→64→1), pure Rust, ~50 LOC
- Model size: 300×64 + 64×1 = 19.3K params = **~77KB**
- No changes to the index build or kNN search

### Expected Accuracy

Moderate improvement. The re-ranker can fix tight-margin races (Category C)
but can't fix cases where the correct font isn't in the top-K at all.
Analysis shows the correct font is in the top-5 candidates for 76/130
misses, so the re-ranker's ceiling is ~76 additional fixes.

### Font Family Grouping

The re-ranker can be trained with font family grouping: when scoring a
candidate, penalize cross-family mismatches more than within-family
variants.

### Verdict

**Lowest risk, fastest to implement, moderate reward.** This is the
"just add a smarter second stage" approach. It doesn't touch the existing
pipeline, making it easy to A/B test.

---

## 8. Inference Runtime: Rust Deployment Options

### Option A: Pure Rust (No External Dependencies)

For feature-based models (metric learning, triplet on features, re-ranker,
small MLP):

- Matrix multiply is ~20 LOC with SIMD hints
- LLVM auto-vectorizes to AVX2/SSE on x86
- No build dependencies, no binary size increase
- Works on any platform Rust targets

**Best for**: Approaches 1, 3 (feature-based), 5

### Option B: `ort` Crate (ONNX Runtime)

For CNN models and complex architectures:

- Mature Rust bindings (1.25M downloads/month)
- Supports CPU (AVX2/SSE auto-dispatch), CUDA, TensorRT
- 3-5× faster than Python equivalents
- Binary size increase: ~20MB (ONNX Runtime shared library)
- Build dependency: downloads pre-built ORT library

```toml
[dependencies]
ort = { version = "2.0.0-rc.12", features = ["load-dynamic"] }
```

**Benchmark reference**: ResNet18-like models achieve ~2ms inference on
CPU via ORT, which is well within our budget.

**Best for**: Approach 4 (CNN on images)

### Option C: `tract` Crate (Pure Rust ONNX)

- Pure Rust, no C dependencies
- Used in production at Sonos for speech recognition
- Slightly slower than ORT but no binary size penalty
- Supports ONNX and NNEF formats

**Best for**: When binary size matters or cross-compilation is needed

### Option D: `candle` Crate (HuggingFace)

- PyTorch-like API in Rust
- 20ms P95 latency, 994 req/s under load test
- Best for models that need fine-tuning at runtime
- Overkill for simple inference

### Recommendation

Start with **Option A** (pure Rust) for the initial implementation.
The feature-based triplet network is small enough that hand-coded matmul
+ ReLU is simpler than adding an ONNX dependency. Move to Option B (ORT)
only if/when pursuing the CNN-on-images approach.

---

## 9. Training Data & Pipeline

### What the Exhaustive Renderer Provides

A separate task is building an exhaustive training data generator. This
will produce:

```
For each font F in the system catalog:
  For each character c in indexed_chars() (106 chars):
    For each DPI in {72, 100, 150, 200, 300, 400, 600}:
      For each AA mode in {grayscale, subpixel_rgb, none}:
        Render glyph image → normalize to NORM_H
        Compute 100-dim feature vector
        Store: (font_path, variant_tag, char, dpi, aa_mode, image, features)
        Ground truth: font_family, weight_class, italic, optical_size

Total: ~5K fonts × 106 chars × 7 DPIs × 3 AA = ~11M samples
```

### Training vs Validation Split

- **Train**: All fonts except the 30 specimen fonts, all DPIs/AA modes
- **Validation**: The 30 specimen fonts at DPIs matching real scans (150–300)
- **Test**: The enriched audit data (real scanned glyphs from the specimen
  PDF). This is the only test set that reflects true scan degradation.

### Hard Negative Mining

For triplet/contrastive training, hard negatives are critical. Strategy:

1. **Pre-compute** all embeddings with the current model
2. **For each anchor**, find the K nearest fonts from different families
3. **Sample negatives** from these near-miss fonts (not random fonts)
4. **Refresh** every N epochs as the model improves

### Font Family Labels

The training pipeline needs a font family grouping function. Build this
from `FontIdentity`:

```python
def font_family_id(font_path):
    """Group variable/static instances and optical sizes."""
    identity = read_font_identity(font_path)
    return (identity.family, identity.weight_bucket, identity.italic)
```

Fonts with the same `font_family_id` are considered equivalent for
training purposes.

---

## 10. Comparison Matrix

| Criterion | Metric Learning (LMNN) | GBT / RF | Siamese/Triplet (features) | CNN Contrastive (images) | Hybrid Re-ranker |
|---|---|---|---|---|---|
| **Accuracy ceiling** | Moderate (~88%) | High (~92%) | High (~92%) | Highest (~95%) | Moderate (~90%) |
| **Category A fixes** | Via grouped labels | Via grouped labels | Via grouped labels | Via grouped labels | Limited |
| **Category B fixes** | None | None | None | None | None |
| **Category C fixes** | Moderate | High | High | Highest | Moderate |
| **Open-set (new fonts)** | ✅ Yes | ❌ Retrain | ✅ Yes | ✅ Yes | ✅ Yes (stage 1) |
| **Inference µs/char** | ~50 (same as now) | ~3 | ~18 | ~30 | ~55 (50+5) |
| **Model size** | 4.2MB (106×100×100) | 100MB+ | 92KB | 600KB (ONNX) | 77KB |
| **Rust dependencies** | None | lightgbm3-rs or ONNX | None | ort or tract | None |
| **Implementation effort** | Low (1-2 days) | Medium (3-5 days) | Medium (3-5 days) | High (1-2 weeks) | Low (1-2 days) |
| **Training effort** | Low (Python, hours) | Medium (Python, hours) | Medium (PyTorch, day) | High (PyTorch, days) | Low (Python, minutes) |
| **Pipeline changes** | Apply L before kNN | Replace kNN entirely | Apply f before kNN | Replace features+kNN | Add post-kNN stage |
| **Risk** | Low | Medium (fixed classes) | Low | Medium (complexity) | Low |
| **Debuggability** | High (linear transform) | Medium (tree inspection) | Low (learned embedding) | Low (black box) | High (see re-ranking) |

---

## 11. Recommendation: What to Build First

### Phase 1: Immediate Wins (Before Any ML) — 1 day

Before investing in learned classifiers, fix the low-hanging fruit:

1. **Font family grouping**: Merge SourceSerif4-It with SourceSerif4
   Italic[opsz,wght], IBMPlexSerif with IBMPlexSerif Text, etc. This
   eliminates **45 of 130 misses** (all of Category A) with no ML at all.
   Implementation: expand `FontIdentity::is_major_diff()` into an
   equivalence-class function, and use it to group candidates before
   reporting the winner.

2. **Missing font handling**: Register known substitution pairs
   (CourierNew → Courier, TimesNewRoman → NimbusRoman, Arial → Liberation
   Sans) as acceptable matches. This eliminates another ~15 misses.

   Together, these two changes could bring accuracy from **84.5% to 96%**
   without any ML.

### Phase 2: Triplet Network on Features — 3-5 days

Build a feature-based triplet network as described in Approach 3:

```
Pipeline:
1. Generate training data (use exhaustive renderer output)
2. Train 100→128→64→32 triplet network in PyTorch (1 day)
3. Export weights as JSON/binary
4. Implement 50-LOC embedding function in Rust (char_index.rs)
5. Replace Fisher-weighted features with network embeddings
6. kNN in 32-dim embedding space instead of 100-dim weighted space
```

**Why triplet on features first, not CNN:**
- Leverages existing feature infrastructure (no new rendering pipeline)
- Pure Rust inference (no ONNX dependency)
- Smaller embedding = faster kNN (32-dim vs 100-dim)
- If it works, it proves the concept before investing in CNN
- Easy to A/B test against the current system

**Expected outcome**: Fix 20-30 of the 45 Category C misses, bringing
accuracy to **~97-98%** (combined with Phase 1).

### Phase 3: Hybrid Re-Ranker (Optional, If Triplet Plateaus) — 1-2 days

If there are still tight-margin misses after the triplet network, add
a re-ranker:

1. Collect (query, top-K candidates, correct_font) from audit runs
2. Train a tiny MLP (300→64→1) binary classifier in Python
3. Export weights, implement in Rust
4. Insert re-ranking step after kNN, before voting

This is a low-risk safety net that can squeeze out the last few
percentage points.

### Phase 4: CNN on Images (Future, If Needed) — 1-2 weeks

Only pursue this if Phase 2 + 3 plateau below the target accuracy. The
CNN can discover features the hand-crafted set misses, but it adds
significant complexity:

- ONNX Runtime dependency
- Image caching pipeline
- Longer training cycles
- Harder to debug

The expected improvement over Phase 2+3 is 2-3 percentage points,
which may not justify the complexity unless the test corpus expands to
more diverse fonts.

### What NOT to Build

- **Random Forest / GBT direct classifier**: The fixed-class limitation is
  a dealbreaker. Unscan must work with whatever fonts are installed on the
  user's system. Retraining a 5K-class model every time someone installs a
  font is not viable. *Exception*: If the model is retrained as part of
  the index build step (~2 min), this becomes feasible but still adds
  significant build time.

- **SVM**: Doesn't scale to 5K+ classes.

- **Full Vision Transformer**: Overkill for 48×W grayscale images. A 3-layer
  CNN achieves equivalent accuracy at 1/10th the parameter count.

- **End-to-end system replacement**: The current pipeline's biggest wins
  come from font family grouping and missing font handling, not classifier
  sophistication. Don't over-invest in ML before fixing the taxonomy.

### Implementation Checklist for Phase 2

```
□ Define font family equivalence classes (FontIdentity → family_id)
□ Implement family grouping in candidate dedup (char_index.rs)
□ Generate training data: (font, char, dpi) → features, family_id
□ Build triplet sampler with hard negative mining
□ Train embedding network: 100 → 128 → 64 → 32
□ Validate embedding quality (same-font cluster tightness)
□ Export model weights to binary format
□ Implement FontEmbedder struct in Rust (matmul + ReLU)
□ Replace Fisher-weighted features with embeddings in flat_vecs
□ Run full audit, compare with baseline
□ If improved: integrate into production path
□ If not: analyze failure modes, adjust training
```

---

## References

1. Weinberger & Saul, "Distance Metric Learning for Large Margin Nearest
   Neighbor Classification," JMLR 2009 (LMNN).
2. Goldberger et al., "Neighbourhood Components Analysis," NeurIPS 2004.
3. Wang et al., "DeepFont: Identify Your Font from An Image," ACM MM 2015.
4. Adobe Patent US10515295B2: "Automatic Measure of Visual Similarity
   Between Fonts" (triplet loss approach).
5. "Robustness of Contrastive Learning on Multilingual Font Style
   Classification," Applied Sciences 2023, 13(6), 3635.
6. Zeng, "Comparing Contrastive and Triplet Loss: Variance Analysis and
   Optimization Behavior," arXiv:2510.02161 (triplet preserves finer
   distinctions than contrastive for detail-oriented tasks).
7. CalliNet: Triplet network for calligraphy style classification, IJDAR
   (94-99% accuracy with triplet loss).
8. `ort` crate: https://github.com/pykeio/ort (ONNX Runtime for Rust).
9. `tract` crate: https://github.com/sonos/tract (Pure Rust ONNX).
10. Unscan existing research: `docs/font-matching-research.md`,
    `docs/ml-classifier-design.md`, `docs/char-index-features-research.md`.
