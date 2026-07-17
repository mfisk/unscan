# ML Classifier Design: Runtime-Trained SIMD Brute-Force Font Index

**Date:** 2025-07-05
**Status:** Implemented + benchmarked
**Replaces:** per-character lookup in former `char_index.rs` (eliminated)

## Problem

The char index previously used `nearest_within_factor(1.5)` per-character
range search in feature space. This has since been replaced by the LDA
classifier in `classifier.rs`.

## Constraints

1. **Pure Rust, no Python.** Different users have different fonts installed.
   The index trains at runtime during the indexing step.
2. **Fast build.** The current index build takes **1.54s** for 373K entries.
3. **Fast inference.** Must return top-N font candidates per character quickly
   enough that the total char-index phase stays under ~200ms per document.
4. **No external files.** Index is built in-memory during construction.

## Architecture: SIMD Brute-Force + Data-Driven Feature Weighting

### Why brute-force instead of ML?

With **4,694 classes** and exactly **1 training sample per class per character**
(each font produces one rendered glyph per char), traditional ML classifiers
are fundamentally underdetermined:

- **Random forest:** Can't learn decision boundaries from 1 sample/class
- **GBT:** Same problem, plus O(n_classes × n_rounds) training cost
- **Neural net:** Massive dependency, can't converge on 1 sample/class

This is a **retrieval** problem (find top-50 from 4,694), not a **classification**
problem. The right tool is nearest-neighbor with a good distance metric.

### Two-part approach

#### 1. SIMD Brute-Force Scan (now the production method)

Flat array of pre-weighted feature vectors scanned linearly per character.
LLVM auto-vectorizes the distance kernel to AVX2/SSE instructions.

**Memory layout:** Features stored as flat `[f32; PADDED_LEN × n_entries]`
per character, where PADDED_LEN=64 (FEAT_LEN=59 padded to 8-float boundary).

**Key advantage:** Guarantees exact nearest-neighbor — no tree-pruning misses.

#### 2. Data-Driven Feature Weighting (replaces hand-tuned 40/30/30)

During build, compute per-character, per-feature weights via inverse-σ
standardization (diagonal Mahalanobis distance):

```
weight[d] = 1.0 / (σ[d] + ε)     # standardize to unit variance
```

Plus typographic importance boosts:
- `serif_score`: 1.5×
- `stroke_contrast`: 1.5×
- `xh_cap_ratio`: 1.3×
- `stress_diagonal`: 1.5×
- `stress_vertical_balance`: 1.3×

Features are pre-multiplied by `sqrt(weight)` at build time, so the query-time
distance is plain squared Euclidean — no per-dimension multiply in the hot loop.

## Benchmark Results (real data, `target-cpu=native`)

| Metric | Tree-based | Brute-Force |
|---|---|---|
| Build time (373K entries) | — | **1.54s** (combined KD+BF) |
| Per-char query (self-lookup) | **12.8 µs** | **75.5 µs** |
| Results agreement | — | 100% (same 808 hits) |
| Memory (feature data) | ~90 MB + tree overhead | **92.5 MB** flat |
| Est. per-doc overhead (94 lines × 15 chars) | ~18ms | ~106ms |

**Note:** Self-lookup queries (font against itself) show worst-case BF/KD ratio
because tree-based factor-based pruning is maximally effective at zero distance.
On real scan queries with noise, tree search explores more subtrees, narrowing
the gap. The real payoff is accuracy: BF guarantees exact NN (no pruning misses).

## Implementation

### Files (historical — architecture has changed)

**Note:** The `char_index.rs` module and `BruteForceIndex` have been eliminated.
Font identification is now handled by the LDA classifier in `src/classifier.rs`
with search logic in `src/font_match.rs`. The design notes below document the
intermediate brute-force approach that preceded the LDA classifier.

1. ~~**`src/brute_force.rs`**~~ — eliminated
2. ~~**`src/char_index.rs`**~~ — eliminated; functionality split across `classifier.rs`, `font_match.rs`, `features.rs`, `segment.rs`
3. **`src/classifier.rs`** — LDA classifier with runtime-trained weights
4. **`src/font_match.rs`** — CI search + font selection with tie-break

### API

```rust
// In char_index.rs — new BF search function
pub fn search_candidates_bf(
    index: &CharIndex,
    char_crops: &[(char, GrayImage)],
    top_n: usize,
) -> Vec<(String, f32)>;

// In brute_force.rs
impl BruteForceIndex {
    pub fn build(entries, name_to_id) -> Self;
    pub fn query_topk(ch, raw_query, k) -> Vec<(font_id, dist²)>;
    pub fn query_within_factor(ch, raw_query, factor) -> Vec<(font_id, dist²)>;
}
```

### Integration (to switch from tree-based to brute-force)

In `src/main.rs`, line ~404, change:
```rust
// Before (tree-based):
let ci_results = char_index::search_candidates(&char_index, &char_crops, 500);

// After (BF):
let ci_results = char_index::search_candidates_bf(&char_index, &char_crops, 500);
```

Both return the same `Vec<(String, f32)>` type. The BF variant uses raw features
(not the 3-group weighted normalization) and the data-driven feature weights.

### Serialization

No change to the UCIX binary format. The BF index is rebuilt from the same
`entries` data during `rebuild_trees()`. Cache load remains fast (1.54s total).

## Next Steps

1. **A/B test on specimen PDF:** Run full pipeline with `search_candidates_bf()`
   and compare line-level accuracy vs tree-based baseline (currently 42%)
2. **Tune factor threshold:** BF's exact NN enables using a tighter factor
   (e.g., 1.3 instead of 1.5) since no pruning misses to compensate for
3. **Remove quality gate:** The 0.5 distance quality gate was tuned for the
   weighted feature space; may need recalibration for BF's data-driven weights
4. **Profile and optimize:** If 75µs/query is too slow, add inverted-index
   prefilter to reduce scan from 3,700 to ~500 candidates per character
5. **Resolved — brute-force is now the production search method.


## Classifier Confidence Blending (OOD Dampening)

`GlyphClassifier::probabilities()` in `src/classifier.rs` uses a Gaussian
kernel over squared Euclidean distances in LDA space to produce softmax
probabilities for each glyph.  A confidence blending step dampens scores
for out-of-distribution (OOD) query vectors.

### How it works

1. **Standard softmax**: for each centroid, compute
   `exp(-(d - d_min) / (2σ²))`, normalize to sum to 1.

2. **Confidence**: `exp(-d_min / (2σ²))` — the raw Gaussian density at the
   nearest centroid.  When the query sits on a centroid, confidence ≈ 1.0.
   When it's far from all centroids, confidence → 0.

3. **Blend**: final probability =
   `confidence × softmax_p + (1 - confidence) × uniform_p`.

The effect: an in-distribution query (small `d_min`) gets pure softmax
probabilities — the top glyph might be 28× uniform.  An OOD query (large
`d_min`) converges toward 1× uniform regardless of which glyph "wins" the
softmax, because the win is meaningless when the query is far from
everything.

### σ² computation

`compute_sigma_sq()` sets σ² to the **median pairwise squared distance**
between all centroids for a given character class.  This is computed once
per character at model load time.  Using the median makes the bandwidth
robust to outlier centroids without requiring per-font tuning.

### Motivation

Without confidence blending, a wrong crop (e.g., an `r` misclassified as
`-` due to a segmentation failure) can produce extreme ×uniform values
(28×, 36×) because the softmax still concentrates on whichever centroid
happens to be geometrically closest, even though the query is nowhere near
the true distribution of any glyph.

Two mechanisms address this:

1. **Confidence blending** (classifier level): an OOD crop that lands far
   from all centroids gets a confidence near 0, so its probability converges
   to uniform and its contribution to font scoring is neutralized.

2. **OOD observation weighting** (scoring level): each observation's weight
   is scaled by `min(1, med_nn / min_d)`.  When a crop is far from all
   centroids (`min_d >> med_nn`), the observation's weight drops toward zero
   regardless of softmax concentration.  This catches the case confidence
   blending misses: when a wrong crop lands *near* a real centroid.

### Font Score Aggregation

Font scoring uses sum of squared deviations from the per-observation best:

```
score(font) = −Σᵢ (ln p_best_i − ln p_font_i)² · wᵢ
```

where `p_best_i` is the highest probability any candidate font achieves at
observation `i`, and `wᵢ` incorporates both position weight and OOD weight.
Highest score (closest to zero) wins.  Squaring amplifies discriminative
observations: a single character where the font falls far behind contributes
quadratically more than many characters where all candidates score alike.

This replaces the earlier `weighted_mean(ln(p))` which was vulnerable to
many non-discriminative observations outvoting a single informative one.
