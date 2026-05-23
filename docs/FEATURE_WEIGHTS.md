# Feature Weight Optimization

How the per-feature weights in `as_weighted_slice()` are derived.

## Problem

The character index (CI) compares scan crops against index renders using
weighted Euclidean distance across a 57-dimensional feature vector. The
original weights were hand-tuned group weights (profile=0.40, scalars=0.30,
v2=0.30) with L2-normalization within each group. This gave equal weight to
discriminative features (stroke_contrast, fisher=34) and near-useless ones
(xh_cap_ratio, fisher≈0; v_center, fisher=0.3).

## Approach: Fisher Discriminant Analysis

For each feature dimension, compute two quantities:

- **Signal** (between-font variance): How much does this feature vary across
  different fonts rendering the same character? High signal means the feature
  can tell fonts apart.

- **Noise** (scan-index variance): How much does rasterization (anti-aliasing,
  resolution, blur) shift this feature between a scan crop and the
  corresponding clean index render? High noise means the feature is unreliable
  when matching scanned input.

The **Fisher ratio** = signal / noise gives the discrimination power of each
feature. The optimal weight is proportional to `sqrt(fisher_ratio)` (sqrt
because weights multiply features before squaring in L2 distance).

## Data

- **Index features**: 353,927 entries from `char-index.bin` covering 4,502
  font variants and 101 indexed characters. Dumped using
  `learn_weights --index`, which calls `compute_features()` on the actual
  index entries (post `render_char_normalised` → `normalize_to_ink_bounds`).

- **Scan features**: 9,354 character crops from the font-timeline specimen PDF
  (30 font sections, ~486 text lines), extracted via `--diag-seg` and feature-
  computed using `learn_weights --scans`. These go through the real scan path:
  `extract_chars_from_boundaries` → `normalize_to_ink_bounds` →
  `compute_features`.

Both sides use the **same Rust code paths** — no Python reimplementation of
features or normalization.

## Deduplication

The index contains many OpenType variant entries (`font.otf|lnum`,
`font.otf|smcp`, `font.otf|onum`, `font.otf|hist`, etc.) that render
identical glyphs for most characters. Without dedup, these identical vectors
contribute zero between-font variance, diluting the signal estimate by ~75%.

**Dedup rule**: For each character independently, round all feature vectors to
4 decimal places and collapse entries with identical rounded vectors. This
removes 264,447 duplicates (74.7% reduction: 353,927 → 89,480 unique vectors)
while preserving OT variants that genuinely render differently (e.g., an
old-style `g` in `|onum` has different counter features than the default `g`).

Noise estimation uses the full (non-deduped) index for nearest-neighbor lookup
so that the closest matching font is always found accurately.

## Key Findings

### Group-level rebalancing

| Group            | Dims | Old weight | Fisher optimal |
|------------------|------|-----------|----------------|
| Column profile   | 32   | 0.40      | **0.59**       |
| Original scalars | 7    | 0.30      | **0.17**       |
| V2 features      | 18   | 0.30      | **0.24**       |

Profile was under-weighted; scalars were nearly 2× over-weighted.

### Individual feature ranking (top 5)

| Rank | Feature          | Fisher | Note |
|------|-----------------|--------|------|
| 1    | stroke_contrast | 34.3   | Dominant discriminator — 5× signal of #2 |
| 2    | term1           | 8.4    | Terminal angle bin 1 |
| 3    | aspect          | 5.2    | Width/height ratio |
| 4    | term3           | 4.1    | Terminal angle bin 3 |
| 5    | prof0           | 3.2    | Left-edge column profile |

### Dead or harmful features

| Feature      | Fisher | Problem |
|-------------|--------|---------|
| xh_cap_ratio | 0.003  | Near-zero signal, massive noise (0.52) — ratio computed differently at index vs scan time |
| v_center     | 0.31   | Almost no between-font variance |
| h_balance    | 0.74   | More noise than signal |
| serif_score  | 1.11   | High signal (0.128) but **almost as much noise** (0.115) — rasterization blur destroys serif detection on scan crops |

The `serif_score` finding is critical: it was responsible for 24% of the
distance in the IBM Plex Serif `i` mismatch, but Fisher says it's barely
above noise. The anti-aliasing fringe from rasterization widens narrow strokes
and softens serifs, making a serifed scan crop look half as serifed as the
clean render.

## Methodology: No L2 Group Normalization

The old approach L2-normalized each group before applying group weights. This
made every feature's contribution depend on what else was in its group — a
noisy feature in a group with one strong feature would get amplified by the
normalization.

The Fisher weights are applied directly to raw feature values: `weighted[i] =
raw[i] * weight[i]`. No group normalization, no cross-feature dependency.
Each feature's influence is set independently by its measured discrimination
power.

## Reproducing

```bash
# 1. Build the tools
cargo build --release --bin learn_weights --bin feat_diff

# 2. Dump index features (requires char-index.bin in ~/.cache/unscan/)
./target/release/learn_weights --index > /tmp/index_features.tsv

# 3. Run specimen with --diag-seg to get scan crops
./target/release/unscan test-docs/font-timeline-specimen-rasterized.pdf \
    -o /dev/null --diag-seg /tmp/diag-crops

# 4. Dump scan features from diag-seg output
./target/release/learn_weights --scans /tmp/diag-crops > /tmp/scan_features.tsv

# 5. Run Fisher analysis (Python, requires numpy)
python3 tools/fisher_analysis.py
```

The analysis script is at `tools/fisher_analysis.py`. It outputs per-feature
signal/noise/fisher/weight and a ready-to-paste Rust `const FISHER_WEIGHTS`
array.

## Per-character comparison

To inspect individual character mismatches:

```bash
# Side-by-side scan crop vs specific font's index render
./target/release/unscan input.pdf -o /dev/null \
    --diag-seg /tmp/diag \
    --diag-ref-font /path/to/expected-font.ttf

# Per-dimension feature diff between two crop PNGs
./target/release/feat_diff /tmp/diag/.../chars/02_i.png \
                           /tmp/diag/.../chars/02_i_ref.png
```

`feat_diff` shows weighted distance² per feature, sorted worst-first with
cumulative percentage, so you can immediately see which features dominate a
mismatch.
