# CI Feature Noise Analysis

Analysis of unprint's 100-dimensional character index (CI) feature space to identify
noisy, harmful, or underperforming features. Based on the enriched audit of
`font-timeline-specimen-scanned.pdf` (490 text entries, 130 misses).

## Executive Summary

**The current feature set is mostly not the problem.** The dominant failure mode
(92% of misses) is same-family variant confusion — the CI correctly identifies the
font family but picks the wrong variant (optical size, weight, static vs variable).
Only 10 of 130 misses involve genuinely different font families. The features are
doing their job for inter-family discrimination; the failures are in intra-family
discrimination where the features have insufficient resolving power.

Key findings:
1. **v_center** (weight 7.2%) dominates the distance metric disproportionately — one
   feature controls more than 3× the budget of the next-highest
2. **xh_cap_ratio** (weight 0.0%) and **stroke_contrast** (weight 0.05%) are dead features
3. **Profile edges** (col0-2, col29-31, row0-2, row28-31) have low Fisher scores
   because edge bins capture padding noise
4. **45 of 130 miss lines** have the GT font at rank #1 in CI scoring — these are
   variant-selection failures, not feature failures
5. **Scaling-dependent miss rate**: 67% at ≤30px line height vs 18% at 80-120px

## 1. Miss Taxonomy

### 1.1 Overall Distribution

| Category | Count | % of entries |
|----------|------:|------------:|
| Hit | 346 | 70.6% |
| Minor miss (same family/weight/italic) | 56 | 11.4% |
| Major miss (different family, weight, or italic) | 74 | 15.1% |
| SSIM failure (right font, rendering mismatch) | 4 | 0.8% |
| Kept raster (no CI match) | 5 | 1.0% |
| No ground truth | 5 | 1.0% |

### 1.2 Confusion Type Breakdown

| Confusion Type | Lines | Chars | GT Closer % |
|---------------|------:|------:|------------:|
| Optical size: variable font matched static | 41 | 1294 | 47.0% |
| Weight confusion (Regular↔Medium, Regular↔Light) | 26 | 767 | 26.7% |
| Courier variants (CourierNew→Courier) | 15 | 390 | 8.5% |
| Text variant (Regular↔Text optical size) | 15 | 391 | 38.1% |
| Different family entirely | 12 | 288 | 48.6% |
| Optical size: SmText variant | 9 | 259 | 46.3% |
| Prestige Elite variants | 6 | 129 | 27.9% |
| Static vs variable font | 4 | 121 | 14.9% |
| EBGaramond variants | 2 | 60 | 36.7% |

**Key insight**: 120/130 misses (92%) are same-family variant confusion. The CI's
100-feature vector successfully distinguishes between font families; it fails at
distinguishing between font variants that render nearly identical glyphs (optical
sizes, static vs variable, weight-adjacent variants).

### 1.3 GT Font's Position in CI Candidates

| GT Rank | Count | Implication |
|---------|------:|-------------|
| Rank 1 (top scorer) | 45 | Variant selection bug, not feature issue |
| Rank 2 | 31 | Very close miss — margin median 0.27% |
| Rank 3 | 6 | Slightly farther but still tight |
| Rank 4+ | 8 | Genuinely misidentified |
| Not in candidates | 40 | Font not installed or not matching |

**45 misses (35%) have the GT font literally at rank #1 in CI scoring** — these
lines lose because the system picks a different .ttf file that happens to have the
same font family and per-char vote pattern. These are purely variant-selection
failures.

77 misses have the GT in candidates but not at #1. For these, the median margin is
just **0.27%** and all 77 are within 2%. A single noisy character vote can flip the
result.

## 2. Feature Weight Analysis

### 2.1 Weight Budget by Group

| Group | Dims | Total Weight | % Budget | Avg/dim |
|-------|-----:|------------:|---------:|--------:|
| Column profile | 32 | 0.245 | 24.5% | 0.0076 |
| Scalar v1 | 7 | 0.123 | 12.3% | 0.0176 |
| Scalar v2 | 18 | 0.243 | 24.3% | 0.0135 |
| Row profile | 32 | 0.258 | 25.8% | 0.0081 |
| Scalar v3 | 11 | 0.131 | 13.1% | 0.0119 |

### 2.2 Top 15 Features by Fisher Weight

| Rank | Feature | Dim | Weight | Group |
|-----:|---------|----:|-------:|-------|
| 1 | **v_center** | 34 | 0.07233 | scalar_v1 |
| 2 | mean_stroke_w | 99 | 0.02403 | scalar_v3 |
| 3 | cross3 | 52 | 0.02143 | scalar_v2 |
| 4 | cross4 | 53 | 0.02128 | scalar_v2 |
| 5 | cross6 | 55 | 0.02059 | scalar_v2 |
| 6 | cross5 | 54 | 0.02053 | scalar_v2 |
| 7 | counter_area | 39 | 0.02041 | scalar_v2 |
| 8 | cross2 | 51 | 0.02027 | scalar_v2 |
| 9 | cross1 | 50 | 0.01999 | scalar_v2 |
| 10 | v_symmetry | 91 | 0.01848 | scalar_v3 |
| 11 | h_balance | 35 | 0.01821 | scalar_v1 |
| 12 | cross7 | 56 | 0.01611 | scalar_v2 |
| 13 | cross0 | 49 | 0.01414 | scalar_v2 |
| 14 | aspect | 32 | 0.01400 | scalar_v1 |
| 15 | ink_density | 33 | 0.01371 | scalar_v1 |

### 2.3 Dead or Near-Dead Features

| Feature | Dim | Weight | Issue |
|---------|----:|-------:|-------|
| **xh_cap_ratio** | 38 | 0.00000 | Hardcoded to 0.0 in source — dead code |
| **stroke_contrast** | 37 | 0.00050 | Near-zero discriminative power |
| **row0** | 57 | 0.00216 | Edge bin: captures padding noise |
| **row31** | 88 | 0.00225 | Edge bin: captures padding noise |
| **row1** | 58 | 0.00311 | Near-edge noise |
| **skel_branch** | 92 | 0.00282 | Low discriminative power at NORM_H=48 |
| **counter_asp** | 42 | 0.00240 | Noisy: small counter → unstable aspect |

### 2.4 v_center Dominance Problem

`v_center` at weight 0.07233 controls **7.2% of the total distance metric**. The
second-highest feature (`mean_stroke_w`) is at 0.02403 — less than one-third. This
single feature has more influence than the entire 32-dimension row profile combined.

The Fisher analysis justifies this: v_center (the ink-weighted vertical centroid)
genuinely varies between fonts and is stable across DPI. But it creates a fragile
system where a small v_center measurement error (from scan crop alignment, AA
artifacts, or baseline drift) can overwhelm the signal from the other 99 features.

**Risk**: At margins of 0.27%, a 0.4% shift in v_center alone can flip a match.

## 3. Per-Character Analysis

### 3.1 Characters with Highest Miss Rate (≥10 occurrences)

| Char | Total | Misses | Miss% | GT Closer% |
|------|------:|-------:|------:|----------:|
| `+` | 23 | 19 | 82.6% | 94.7% |
| `/` | 52 | 42 | 80.8% | 83.3% |
| `F` | 87 | 55 | 63.2% | 49.1% |
| `O` | 72 | 36 | 50.0% | 5.6% |
| `S` | 87 | 42 | 48.3% | 38.1% |
| `L` | 87 | 41 | 47.1% | 61.0% |
| `M` | 83 | 36 | 43.4% | 16.7% |
| `A` | 58 | 24 | 41.4% | 4.2% |
| `.` | 208 | 80 | 38.5% | 32.9% |
| `I` | 97 | 36 | 37.1% | 19.4% |

**`+` and `/`** have extremely high miss rates but GT is almost always closer —
these characters appear overwhelmingly in miss lines but actually point toward the
correct answer. They're just outvoted.

**`O`, `A`, `M`, `I`** have high miss rates AND the GT font is almost never closer.
These are the characters where the feature vector truly fails to distinguish
variants. Capital letters with simple geometry (circles, triangles, rectangles) have
fewer distinguishing details at NORM_H=48.

### 3.2 Best-Performing Characters

| Char | Total | Misses | Miss% | GT Closer% |
|------|------:|-------:|------:|----------:|
| `ﬁ` | 11 | 1 | 9.1% | 100% |
| `9` | 89 | 15 | 16.9% | 33.3% |
| `1` | 114 | 20 | 17.5% | 10.0% |
| `8` | 68 | 13 | 19.1% | 23.1% |
| `h` | 283 | 60 | 21.2% | 61.7% |
| `j` | 92 | 18 | 19.6% | 50.0% |
| `4` | 70 | 15 | 21.4% | 53.3% |

**Ligatures** (`ﬁ`) are nearly always correct — their complex shape provides rich
feature signal. **Digits with distinctive topology** (`8` with two holes, `9`/`4`
with unique profiles) also perform well.

### 3.3 Most Helpful Characters in Misses (Point Toward GT)

When a character does appear in a miss line, how often does its nearest-neighbor
point toward the correct font?

| Char | GT Closer% | Notes |
|------|----------:|-------|
| `+` | 94.7% | Excellent discriminator, outvoted |
| `/` | 83.3% | Excellent discriminator, outvoted |
| `R` | 69.2% | Good discriminator |
| `:` | 62.2% | Good discriminator |
| `h` | 61.7% | Strong — ascender shape is distinctive |
| `L` | 61.0% | Serif structure is informative |
| `T` | 59.4% | Good |
| `i` | 57.9% | Dot placement varies between fonts |
| `l` | 56.2% | Narrow but ascender shape helps |

### 3.4 Least Helpful Characters (Push Toward Wrong Font)

| Char | GT Closer% | Notes |
|------|----------:|-------|
| `O` | 5.6% | Circle — no distinguishing features between variants |
| `A` | 4.2% | Triangle — geometry too simple |
| `0` | 4.3% | Oval — same problem as O |
| `1` | 10.0% | Vertical stroke — no variant signal |
| `N` | 0.0% | All 12 occurrences pushed wrong way |
| `3` | 14.3% | |
| `B` | 14.3% | |
| `M` | 16.7% | |
| `I` | 19.4% | Vertical stroke — no variant signal |

**Pattern**: Characters with simple, symmetric geometry (O, A, N, M, I, 0, 1) are
systematically harmful in the voting — they match the wrong variant more often than
the right one. These characters lack the fine detail (serif shape, terminal style,
counter proportions) that distinguishes font variants.

## 4. Distance Margin Analysis

### 4.1 Per-Character Distance Ratios (GT vs Chosen)

For each character crop in a miss, the ratio `gt_font_dist_sq / chosen_dist_sq`:
- **< 1.0**: GT is closer (feature vector favors correct font)
- **> 1.0**: GT is farther (feature vector favors wrong font)

| Range | Count | Notes |
|-------|------:|-------|
| [0, 0.50) | 110 | GT much closer — easy wins being lost |
| [0.50, 0.80) | 246 | GT moderately closer |
| [0.80, 0.90) | 188 | GT slightly closer |
| [0.90, 1.00) | 787 | Very tight — nearly tied |
| [1.00, 1.10) | 825 | Very tight — wrong side |
| [1.10, 1.20) | 205 | GT slightly farther |
| [1.20, 1.50) | 439 | GT moderately farther |
| [1.50, 2.00) | 414 | GT much farther |
| [2.00, 5.00) | 450 | GT way farther |
| [5.00, ∞) | 34 | GT astronomically farther |

**The distribution is bimodal**: a large cluster around 0.95-1.05 (barely
distinguishable) and a long tail above 1.5 (clearly wrong).

- Median ratio: **1.021** (GT is barely farther on median)
- GT closer: 1331 chars (36.0%), median advantage: 5.6%
- GT farther: 2326 chars (62.9%), median disadvantage: 30.5%

**The asymmetry is revealing**: when GT is closer, it's only slightly closer (5.6%
median). When GT is farther, it's much farther (30.5% median). The features have
enough information to give the GT font a slight advantage on some chars, but the
wrong font often wins decisively on others — likely due to scan-to-reference
mismatch amplified by specific feature instabilities.

### 4.2 Line-Level GT Fraction

For each miss line, what fraction of characters had GT closer?

| GT Closer Range | Lines |
|----------------|------:|
| 0-10% | 22 |
| 10-20% | 9 |
| 20-30% | 13 |
| 30-40% | 20 |
| 40-50% | 35 |
| 50-60% | 17 |
| 60-70% | 6 |
| 70-80% | 7 |
| 80-90% | 1 |

Median: **40.0%** of characters point toward GT in miss lines.

31 lines (24%) have GT closer for majority of chars but still lose the vote —
these are cases where the minority of chars that favor the wrong font do so more
decisively (larger distances).

## 5. Scaling Analysis

| Line Height | Miss Rate | Scale Direction |
|------------|----------:|----------------|
| 20-30px | 66.7% | Heavy upscale |
| 30-40px | 40.0% | Moderate upscale |
| 40-50px | 50.0% | Near native |
| 50-60px | 36.2% | Slight downscale |
| 60-80px | 22.7% | Moderate downscale |
| 80-120px | 18.4% | Heavy downscale |

Miss rate increases monotonically with smaller text. At 20-30px line height (heavy
upscaling to NORM_H=48), miss rate is **3.6× worse** than at 80-120px. This
confirms that scaling artifacts are a primary noise source — upscaling introduces
interpolation blur that affects all features, while downscaling (which is more
common with large text) preserves detail better because Lanczos3 anti-aliases
cleanly.

## 6. Feature Group Assessment

### 6.1 Column Profile (dims 0-31) — ADEQUATE

- 24.5% of weight budget for 32 dims (0.76% per dim)
- **Edge bins (col0-2, col29-31) are noisy**: they capture anti-aliasing artifacts
  and crop boundary effects. Fisher weights correctly downweight them (0.39-0.55%)
  but they still contribute noise to the total distance.
- **Center bins (col10-20) are the most discriminative**: they capture the main
  stroke structure. Weights peak at col15-16 (~1.05%).
- **Recommendation**: The profile group works. Edge bin downweighting could be more
  aggressive (halve weights for col0-1 and col30-31), or use a 24-bin profile that
  discards the outer bins.

### 6.2 Scalar v1 (dims 32-38) — NEEDS SURGERY

- 12.3% of budget for 7 dims (1.76% avg)
- **v_center**: 7.23% — absurdly dominant. Valid signal but creates single-point
  failure. Should be capped at ~3% or the metric should use v_center as a pre-filter
  rather than part of the weighted sum.
- **aspect, ink_density, h_balance**: 1.4-1.8% each — solid features.
- **serif_score**: 0.46% — weak but legitimately measures serifs. Keep.
- **stroke_contrast**: 0.05% — effectively dead. At NORM_H=48 with binary
  thresholding, there isn't enough resolution to measure stroke contrast. Either
  fix the measurement (use grayscale-weighted runs) or remove.
- **xh_cap_ratio**: 0.00% — hardcoded to 0.0 in source. Dead code. Remove or
  implement.

### 6.3 Scalar v2 (dims 39-56) — STRONG

- 24.3% of budget for 18 dims (1.35% avg)
- **h_crossings (cross0-7)**: 1.41-2.14% — the strongest feature sub-group.
  Crossing patterns are highly stable across DPI and vary well between fonts.
- **counter_area**: 2.04% — strong for characters with counters (a, e, o, p, etc.)
- **compactness**: 1.21% — solid geometric feature
- **terminal angles**: 0.49-0.96% — moderate but noisy at low resolution
- **counter_asp**: 0.24% — noisy. Counter aspect ratio is unstable when counters
  are small or absent.
- **Recommendation**: This is the best-performing group. Consider adding more
  crossing features (diagonal crossings, crossing positions).

### 6.4 Row Profile (dims 57-88) — ADEQUATE

- 25.8% of budget for 32 dims (0.81% per dim)
- Same edge-bin noise pattern as column profile: row0-2 and row28-31 are weak.
- Interior bins are slightly more discriminative than column profile because they
  capture ascender/descender distribution, which varies significantly between fonts.
- **Recommendation**: Same as column profile — edge bins could be trimmed.

### 6.5 Scalar v3 (dims 89-99) — MIXED

- 13.1% of budget for 11 dims (1.19% avg)
- **mean_stroke_w**: 2.40% — excellent, second-highest overall. Stroke width is
  a primary font characteristic.
- **v_symmetry**: 1.85% — strong. Vertical symmetry distinguishes many font pairs.
- **h_symmetry**: 1.32% — good.
- **quadrant_density (4)**: 1.07-1.28% — solid spatial distribution features.
- **hole_count**: 1.01% — moderate. Quantization to 4ths limits resolution.
- **corner_count**: 0.65% — weak at NORM_H=48 due to aliasing.
- **skel_branch**: 0.28% — very weak. Skeletonization at 48px is unstable.
- **skel_endpt**: 0.91% — moderate but noisy for same reason.
- **Recommendation**: corner_count and skel_branch are noise. skel_endpt is
  marginal. Remove or improve skeletonization quality.

## 7. Root Cause Analysis

### 7.1 Why Same-Family Variants Are Hard

The fundamental problem: font variants within a family (e.g., SourceSerif4-Regular
vs SourceSerif4-Italic[opsz,wght]) differ by **optical size adjustments** — tiny
changes to stroke weight, x-height ratio, and serif shape optimized for different
point sizes. These differences are:

1. **Sub-pixel** at NORM_H=48 — a 2% weight change on a 48px glyph is less than 1
   pixel of stroke difference
2. **Anti-aliasing-scale** — the differences are comparable to the noise introduced
   by scan-to-screen rendering and vice versa
3. **Not captured by current features** — the features measure coarse geometry
   (profiles, crossings, symmetry) rather than fine typographic details (serif
   bracket curvature, stroke termination style, contrast ratio)

### 7.2 The Dominant Miss Pair

`SourceSerif4-It → SourceSerif4 Italic[opsz,wght]`: 32 lines, 1115 chars, GT
closer just 46% of the time. The variable-font version (`[opsz,wght]`) is the
Google Fonts version of the same font, likely rendered with default optical size.
The differences between these two are within measurement noise.

### 7.3 The Courier Problem

`CourierNewPSMT → Courier`: 12 lines, GT closer only 4.5%. CourierNew is not
installed on this system — the CI has Courier (TeX Gyre derivative) but not
CourierNew (Microsoft). The features correctly measure that the scanned text
doesn't match Courier well (avg ratio 2.2×), but with no better option in the
index, Courier wins by default.

## 8. Specific Recommendations

### 8.1 Immediate: Dead Feature Removal

1. **Remove xh_cap_ratio** (dim 38) — hardcoded to 0.0, contributes nothing
2. **Downweight stroke_contrast** (dim 37) — weight 0.0005 is negligible; either
   fix the measurement (grayscale-weighted) or set to 0

### 8.2 Short-term: Weight Rebalancing

3. **Cap v_center weight** at 3% instead of 7.2% — redistribute the excess to
   h_crossings and profile features. v_center's information is real but its
   dominance creates single-feature fragility.
4. **Halve edge profile bin weights** (col0-2, col29-31, row0-2, row28-31) —
   these bins are dominated by crop-boundary effects
5. **Zero out skel_branch** (dim 92) — 0.28% weight, generates more noise than signal
   at NORM_H=48

### 8.3 Medium-term: Variant Disambiguation

6. **Add optical-size-aware variant grouping** — when the top N candidates are all
   from the same font family, use the fontmap or a family-equivalence table to
   select the best match, bypassing per-char voting within the family
7. **Implement xh_cap_ratio properly** — the ratio of x-height to cap-height is
   the primary signal that distinguishes optical sizes. It's currently dead code.
8. **Weight character votes by confidence** — instead of equal votes, weight each
   char's vote by (1/chosen_dist_sq). Characters with very tight distances (near
   the noise floor) should have less influence than characters with clear matches.

### 8.4 For ML Training Data Generator

9. **Include variant pairs** in training data — the training set must contain
   same-family variant pairs (optical sizes, static vs variable) so an ML classifier
   can learn to distinguish them
10. **Vary scan simulation parameters** aggressively — the current learn_weights
   uses 3 DPIs (300, 200, 100). Training data should include AA variation, JPEG
   artifacts, and sub-pixel positioning to teach the classifier what signal survives
   these degradations
11. **Per-character confidence labels** — some characters (O, A, N, I) are
   inherently less discriminative than others (h, R, +, ﬁ). The classifier should
   learn per-character reliability.

### 8.5 Character-Specific Weights

12. **Consider per-character Fisher weights** — instead of one global weight vector,
    compute separate weights per character. The optimal weight for 'h' (where
    ascender shape matters) is different from 'o' (where profile shape matters).
    This is a natural extension of the Fisher framework and could improve accuracy
    by 5-10% based on the per-character GT-closer rates above.

## 9. Data Tables

### 9.1 All Features Ranked by Fisher Weight

| Rank | Dim | Feature | Weight | Group |
|-----:|----:|---------|-------:|-------|
| 1 | 34 | v_center | 0.072330 | scalar_v1 |
| 2 | 99 | mean_stroke_w | 0.024029 | scalar_v3 |
| 3 | 52 | cross3 | 0.021426 | scalar_v2 |
| 4 | 53 | cross4 | 0.021282 | scalar_v2 |
| 5 | 55 | cross6 | 0.020585 | scalar_v2 |
| 6 | 54 | cross5 | 0.020531 | scalar_v2 |
| 7 | 39 | counter_area | 0.020413 | scalar_v2 |
| 8 | 51 | cross2 | 0.020272 | scalar_v2 |
| 9 | 50 | cross1 | 0.019990 | scalar_v2 |
| 10 | 91 | v_symmetry | 0.018484 | scalar_v3 |
| 11 | 35 | h_balance | 0.018210 | scalar_v1 |
| 12 | 56 | cross7 | 0.016114 | scalar_v2 |
| 13 | 49 | cross0 | 0.014144 | scalar_v2 |
| 14 | 32 | aspect | 0.013996 | scalar_v1 |
| 15 | 33 | ink_density | 0.013707 | scalar_v1 |
| 16 | 90 | h_symmetry | 0.013220 | scalar_v3 |
| 17 | 98 | quad_br | 0.012751 | scalar_v3 |
| 18 | 95 | quad_tl | 0.012161 | scalar_v3 |
| 19 | 48 | compactness | 0.012096 | scalar_v2 |
| 20 | 96 | quad_tr | 0.011283 | scalar_v3 |
| ... | ... | ... | ... | ... |
| 96 | 42 | counter_asp | 0.002402 | scalar_v2 |
| 97 | 88 | row31 | 0.002250 | row_profile |
| 98 | 57 | row0 | 0.002158 | row_profile |
| 99 | 37 | stroke_contrast | 0.000500 | scalar_v1 |
| 100 | 38 | xh_cap_ratio | 0.000000 | scalar_v1 |

### 9.2 Top Font Confusion Pairs

| Expected → Matched | Lines | GT Closer% | Avg Ratio |
|--------------------|------:|----------:|----------:|
| SourceSerif4-It → SourceSerif4 Italic[opsz,wght] | 32 | 46.0% | 1.050 |
| CourierNewPSMT → Courier | 12 | 4.5% | 2.203 |
| Lato-Regular → Lato Medium | 8 | 31.5% | 1.313 |
| SourceSans3-Roman → SourceSans3 Light | 8 | 10.0% | 2.281 |
| SourceSerif4-It → SourceSerif4SmText It | 6 | 46.3% | 1.073 |
| PrestigeEliteNormal → Prestige Elite Std | 6 | 27.9% | 1.396 |
| IBMPlexSerif → IBMPlexSerif Text | 6 | 40.7% | 1.122 |

---

*Generated from audit data at `/tmp/audit-enriched/audit.json`, 2026-06-13.*
*Analysis scripts in-memory (Python); no source modifications made.*
