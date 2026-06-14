# Scaling & Anti-Aliasing Normalization for Font Matching

> **Updated 2026-06-13** based on NORM_H=48 vs 32 line-by-line audit data.
> Previous version focused on AA fringe from upscaling; this revision
> addresses the broader problem of scale-induced feature instability in
> both directions.

## The Problem (Revised)

The character index (CI) compares scan-extracted glyph crops against
pre-rendered font reference glyphs. Both sides go through
`normalize_to_ink_bounds` → Lanczos3 resize to `NORM_H`. There are
**two interacting problems**:

### Problem 1: Scale-Induced Feature Instability

Any rescaling — up or down — changes the feature vector.
Lanczos3 interpolation, threshold-dependent binary features, and
pixel-counting features all behave differently at different
resolutions. The features are **not scale-invariant**.

**Evidence from NORM_H=48 vs 32 experiment:**
- Headline: 71 major both, minor 57→52, SSIM 4→6. Looks like a gentle
  improvement.
- Reality: **27 new regressions + 30 fixes + 17 category shifts**.
  Massive churn underneath a near-zero net change.
- Every line's scale factor changed by the same ratio (32/48 = 0.667×),
  yet lines flipped between hit and miss in both directions.
- Zero category-stable improvements — the "improvement" is noise that
  happened to net slightly positive.

### Problem 2: Asymmetric AA (Original Problem — Still Real but Secondary)

Font reference images are rasterized by `ab_glyph` natively at
(or near) NORM_H, producing crisp 1px AA fringes. Scan crops come
from MuPDF rasterization at 300 DPI, then get rescaled to NORM_H.

- **Upscaling** (crop < NORM_H) spreads AA fringe across multiple
  pixels, making thin fonts look heavier.
- **Downscaling** (crop > NORM_H) destroys AA non-uniformly, losing
  fine detail.

**But upscaling affects only 9.4% of lines** (46/490). The vast
majority of crops (89%) are downscaled.

### Problem 3: Per-Character Feature-Space Failures

For major misses, the wrong font is genuinely closer in feature space
for **71.8%** of individual characters. The GT font wins the per-char
majority in only 12.2% of major-miss lines. This is a fundamental
feature discrimination problem, not just a scoring/aggregation issue.

## Empirical Findings (NORM_H=48 Audit)

### Scale Factor Distribution

At NORM_H=48, scan crops have these actual heights:
```
  Median crop height: ~65px (downscaled to 48px by 0.74×)
  p10 crop height: ~48px (1:1)
  p90 crop height: ~84px (downscaled by 0.57×)
  Range: 24px to 167px
```

**89% of crops are downscaled**, 9.4% upscaled, 1.4% near 1:1.

### Miss Rate by Crop Height

```
Height range    Scale@48   Total    Hits   Miss    Rate
  0-30px         1.60×+       9       3      6    33.3%
 30-40px         1.37×       10       6      4    60.0%
 40-50px         1.07×       42      21     21    50.0%  *
 50-60px         0.87×       80      51     29    63.8%
 60-70px         0.74×       84      65     19    77.4%
 70-80px         0.64×      111      85     26    76.6%
 80-100px        0.53×      142     115     27    81.0%
```

(*) The 40-50px zone is heavily skewed by SourceSerif4-It, which has
a systemic variable-vs-static-font matching problem (9/49 = 18.4%
accuracy) regardless of scale. Excluding that font, the near-1:1
zone has ~76% accuracy, comparable to other zones.

### Key Observations

1. **Larger crops (more downscaling) have BETTER accuracy** — the
   opposite of "AA fringe from upscaling is the main problem."
2. **Upscaled crops (smallest text) have the worst accuracy**, but
   they're only 9.4% of lines and their problems are compounded
   by OCR difficulty and thin-stroke fragility at small sizes.
3. **The downscaling regime dominates the error budget**: 82 of 134
   total misses come from the 380 downscaled lines.
4. **Italic fonts have dramatically worse accuracy**: 42.9% vs 78.9%
   for non-italic. This is a feature-level discrimination problem,
   not a scaling problem.

### NORM_H=48 → 32: What Actually Changed

Every crop's scale factor decreased by 0.667×:
- An 80px crop went from 0.60× downscale to 0.40× — losing 33% more
  resolution.
- A 30px crop went from 1.60× upscale to 1.07× — much less AA bleed.

**New regressions (27 lines clean@48, miss@32):**
- Median crop height: 61px. These were moderately downscaled at 48
  (0.79×) and are now aggressively downscaled at 32 (0.52×).
- The extra information loss from 48→32 pixels pushed them past the
  discrimination threshold.

**Fixes (30 lines miss@48, clean@32):**
- Median crop height: 60px — same size distribution as regressions.
- No systematic pattern: some fixes came from reduced AA bleed on
  small crops, others from downscaling that happened to move features
  closer to the index side (since the index also renders at 32px now).

**Category shifts (17 lines):**
- 8 went minor→MAJOR (worse): reduced resolution lost discriminative
  detail that was keeping the match within the same font family.
- 7 went MAJOR→minor (better): coarser resolution smoothed out
  differences that were causing cross-family confusion.
- 2 went to SSIM failure: correct font identified but rendering at
  32px looks too different from the scan.

**Conclusion: 48→32 is a noisy wash, not a directional improvement.**
The "5 fewer minor misses" headline masks 57 lines that changed state.

## Current Architecture (for reference)

```
Scan crop ─→ normalize_to_ink_bounds() ─→ compute_features() ─→ 100-dim vector
                    │                           │
                    ├─ tight ink crop            ├─ col/row profiles (grayscale weighted)
                    ├─ 1px white padding         ├─ ink_mask (binary, threshold 200)
                    └─ Lanczos3 → NORM_H        ├─ stroke runs (binary, threshold 200)
                                                └─ morphological features (from ink_mask)

Font render ─→ render_char_normalised() ─→ [same normalize_to_ink_bounds()] ─→ same features
```

Both paths funnel through the same `normalize_to_ink_bounds`, so
padding and scaling logic are symmetric. The asymmetry is *what goes in*:
- **Index renders**: rasterized at a scale engineered to produce ~NORM_H
  ink height. `normalize_to_ink_bounds` barely rescales (~1:1).
- **Scan crops**: arrive at whatever height the source PDF used, then
  get rescaled (usually down) to NORM_H. The scale factor varies
  per-line from 0.29× to 2.0×.

### Feature Sensitivity Analysis

Fisher weights reveal which features carry the most discriminative
power and which are near-dead:

**Top features** (50% of total discrimination):
- `v_center` (7.2%) — vertical center of ink mass. Geometrically stable.
- `mean_stroke_w` (2.4%) — scale-sensitive (pixel-counted runs).
- `cross0..cross7` (15.9% total) — horizontal crossing counts. Integer
  features from binary ink mask — sensitive to threshold and resolution.
- `counter_area` (2.0%) — enclosed area ratio. Topology-based, more
  stable.
- `v_symmetry` (1.8%), `h_balance` (1.8%) — ratio features, stable.
- `aspect` (1.4%), `ink_density` (1.4%) — ratio features, stable.

**Near-dead features:**
- `xh_cap_ratio` (0.000) — literally zero weight, always 0.0.
- `stroke_contrast` (0.050%) — nearly zero Fisher ratio, extremely
  noisy across DPI changes.

**Profile features** (col 24.5%, row 25.8%):
- Each bin's weight is moderate (0.2–1.1% each). Edge bins (col0, col31,
  row0, row31) have the lowest weights — these are most affected by
  AA fringe and ink bounds.

---

## Approach 1: Otsu Global Thresholding (Binarize Before Features)

**Method**: Compute Otsu's optimal threshold on the normalized glyph
image, convert to binary (0/255), then run `compute_features`.

**Revised Assessment**: The original concern about destroying sub-pixel
weight discrimination is valid but needs nuancing. The data shows:
- `stroke_contrast` has near-zero Fisher weight (0.05%) — it's already
  too noisy to be useful, so binarizing it costs nothing.
- `mean_stroke_width` has 2.4% weight. Binarization quantizes
  1.8px→2px vs 2.2px→2px, losing Regular/Medium distinction.
  But the current per-char distance data shows the wrong font is
  closer 71.8% of the time anyway — the features aren't discriminating.

**Scale-stability benefit**: Binarization would make all features
insensitive to AA fringe AND to Lanczos3 ringing artifacts from
rescaling. A binary image rescaled by different amounts produces
more consistent features than a grayscale one.

**Verdict**: **Still probably harmful** for weight discrimination, but
less clearly so than originally thought. The existing features are
already failing at weight discrimination; binarization might trade a
failing grayscale signal for a more stable binary one. Worth testing
as a controlled experiment, not as a first-line fix.

---

## Approach 2: Sauvola / Niblack Adaptive Thresholding

**Verdict**: **Still overkill and counterproductive.** Our images have
clean white backgrounds. Nothing in the new data changes this
assessment. Skip.

---

## Approach 3: Sharpen + Threshold (AA removal)

**Revised Assessment**: With the new data showing downscaling is the
dominant regime, sharpening before downscale could help preserve edge
detail that Lanczos3 smears. But this only helps the scan side (the
index side renders near 1:1).

**Verdict**: **Still marginal.** Asymmetric preprocessing is
dangerous, and the benefit is speculative.

---

## Approach 4: Morphological Operations (Erosion to Counter Bleed)

**Revised Assessment**: The original concern about asymmetric processing
stands. With the new data showing 89% of crops are downscaled (not
upscaled), erosion to counter bleed would help only the minority of
upscaled crops and potentially hurt the majority.

**Verdict**: **Even less applicable than before.** The bleed problem
(upscaling) affects only 9.4% of lines.

---

## Approach 5: Sub-pixel Edge Detection / SWT

**Verdict**: **Still wrong tool for the scale.** Nothing changed.

---

## Approach 6: Feature-Level Normalization (AA-Invariant Features)

### 6a: Grayscale-weighted profiles (current behavior)

Current `ink_val = (255 - px)` weighting. The profiles are already
grayscale-weighted, which is both a strength (preserves weight info)
and a weakness (sensitive to AA fringe width and rescaling artifacts).

### 6b: Centroid-based profile normalization

Still interesting but addresses only the AA asymmetry, not the
broader scale-instability problem.

### 6c: Gamma-weighted ink values

**Revised Assessment**: Still valid for suppressing AA fringe, but the
data shows the main problem is NOT fringe — it's information loss from
downscaling. Gamma weighting helps the 9.4% of upscaled crops where
fringe is wider, but does nothing for the 89% of downscaled crops
where the problem is lost resolution.

**For downscaled crops**, gamma weighting may actually hurt: at lower
resolution, intermediate-value pixels carry real stroke-edge
information, not just fringe. Suppressing them would lose signal.

**Verdict**: **Still worth testing** but expectations should be modest.
Won't address the dominant error source (downscaling information loss).

### 6d: Grayscale-weighted stroke runs

**Revised Assessment**: Same situation as 6c. Helps with fringe-
inflated runs on upscaled crops, but the metric is already noisy
(`stroke_contrast` Fisher weight = 0.05%) and `mean_stroke_width`
at 2.4% Fisher weight is a small part of the total discrimination.

**Verdict**: Low-impact. The features this helps are already
low-importance in the Fisher ranking.

### 6e: Threshold tightening (200 → 128)

**Revised Assessment**: The original argument that threshold 128 =
"majority-ink rule" is principled. But the new data shows:

At NORM_H=48, downscaled crops have *fewer* fringe pixels because
downscaling averages them away. Tightening the threshold mainly
affects upscaled crops (9.4%). For the 89% of downscaled crops,
the effect is minimal.

**Verdict**: Low-impact on the dominant error source.

---

## NEW Approach 7: Multi-Height Index (Avoid Rescaling Entirely)

**Method**: Instead of one NORM_H, render index glyphs at several
heights (e.g., 24, 32, 48, 64, 96). At query time, pick the index
height closest to the scan crop's natural height, avoiding rescaling
entirely (or minimizing it to a very small factor).

**How it works**:
1. At index build time: for each (font, char), render and store
   features at each of {24, 32, 48, 64, 96} pixels.
2. At query time: measure the scan crop's ink height. Find the
   closest indexed height. Normalize the crop to that height (small
   rescale factor, typically < 1.3×). Compare against the index
   entries at that height.

**Pros**:
- **Directly addresses the root cause**: rescaling is the problem, so
  don't rescale. A crop at 60px native height would match against
  64px index entries (1.07× scale) instead of 48px entries (0.80×).
- Both sides experience minimal rescaling → features are comparable.
- Works for both upscaling and downscaling regimes.
- Fully symmetric — no asymmetric preprocessing needed.
- Index build is a one-time cost. 5× more entries per (font, char),
  but index search is already fast.

**Cons**:
- **Index size**: 5× more feature vectors. Currently ~500 fonts × 80
  chars × 100 dims = 4M entries. At 5 heights = 20M entries. Still
  fits in memory but the KNN search is 5× slower.
- **Height selection edge cases**: what if the crop height is between
  two indexed heights? Need a policy (nearest, or try both and pick
  the closer match).
- **Feature weights**: the Fisher weights were computed at one NORM_H.
  They may need to be height-dependent if feature noise profiles vary
  by resolution. Or: recompute Fisher weights across all heights.
- **Implementation complexity**: moderate. The index structure needs
  a height dimension, and the query path needs height selection.

**Mitigations**:
- Use a coarser set of heights (e.g., just {32, 48, 72}) to keep index
  size manageable. Even 3 heights would dramatically reduce the max
  scale factor from 3.4× to ~1.5×.
- Can still keep a single NORM_H as fallback for crops at extreme sizes.

**Verdict**: **Most promising approach for the scaling problem.** This
is the only approach that addresses the root cause (features are not
scale-invariant, so don't rescale) rather than trying to make features
more robust to rescaling.

---

## NEW Approach 8: Simulated-Scan Index

**Method**: Render index glyphs through the same downscale→upscale
pipeline that real scans experience. Instead of matching "clean index
render" vs "degraded scan crop", match "degraded index render" vs
"degraded scan crop".

**How it works**:
1. Render each index glyph at high resolution (e.g., 300 DPI equivalent).
2. Downscale to simulate a scan at each of several DPI levels.
3. Run through `normalize_to_ink_bounds` (which upscales back to
   NORM_H).
4. Store features from this degraded version.

This is essentially what `learn_weights.rs` already does for computing
Fisher ratios, but applied to the actual index.

**Pros**:
- Matches the degradation profile between scan and index.
- Both sides see Lanczos3 artifacts, threshold quantization effects,
  and AA bleed in the same way.
- Could be combined with multi-height: render at several DPI levels
  per height.

**Cons**:
- **Index explosion**: N DPI levels × M heights × F fonts × C chars.
  Quickly becomes huge.
- **Which DPI to simulate?** The scan's effective DPI varies per line
  (depends on original font size in the PDF). Hard to predict which
  DPI to match against.
- **Overfit risk**: if we simulate DPI=300 but the actual scan was
  at 200 effective DPI, the simulated degradation is wrong.
- learn_weights already uses this trick for weight computation. The
  Fisher weights should already downweight features that are
  DPI-sensitive. If they're not doing enough, the problem is in the
  weight computation, not the index.

**Verdict**: **Interesting but impractical.** The DPI matching problem
makes this hard to apply correctly. Better to pursue multi-height
indexing (Approach 7) or scale-invariant features (Approach 9).

---

## NEW Approach 9: Scale-Invariant Features

**Method**: Replace or supplement current features with ones that are
mathematically invariant to resolution changes.

**Candidates**:

### 9a: Normalized profiles (already partially done)

Col/row profiles are already normalized to their maximum value, making
them invariant to overall ink density. But they're NOT invariant to
resolution because the number of pixels contributing to each bin changes
with scale. At lower resolution, each bin covers fewer pixels, making
the profile noisier.

### 9b: Hu Moments / Zernike Moments

Classical shape descriptors that are theoretically invariant to scale,
rotation, and translation. Well-studied in pattern recognition.

**For this use case**: glyph shapes are already axis-aligned (no
rotation), and normalization handles translation. Scale invariance is
the useful property.

**Problem**: Hu moments capture global shape but lose local structure
(serifs, crossbar position, bowl shape). They might not discriminate
between similar fonts. Worth benchmarking but unlikely to replace the
full feature vector.

### 9c: Topological features (already partially used)

- `hole_count`: already scale-invariant (topology doesn't change with
  resolution, mostly). Fisher weight 1.0%.
- `h_crossings`: integer-valued, sensitive to threshold and resolution.
  Currently the single largest feature group by Fisher weight (15.9%).
  These are NOT scale-invariant — at lower resolution, AA pixels
  may flip above/below threshold, adding or removing crossings.
- `counter_area_ratio`: ratio feature, moderately scale-invariant.
  Fisher weight 2.0%.

### 9d: Relative-position features

Features like `v_center`, `h_balance`, `quadrant_density` are already
position-ratios and thus approximately scale-invariant. These have
solid Fisher weights and are the most reliable features in the current
set.

### 9e: Gradient-histogram features (HOG-like)

Histogram of Oriented Gradients computed on the normalized image.
HOG is designed to be robust to illumination and moderate geometric
changes. At the scale of these images (~30-50px), gradients are noisy,
but HOG's spatial binning and orientation binning may be more stable
than pixel-counting features.

**Verdict**: Scale-invariant features are the right long-term direction
but require significant R&D. The current 100-dim feature vector was
designed for a fixed NORM_H; making it scale-invariant is essentially
a redesign. **Multi-height indexing (Approach 7) is the faster win.**

---

## NEW Approach 10: Learned Feature Weighting / ML Classifier

**Method**: Replace the hand-tuned Fisher weights and Euclidean distance
with a learned classifier trained on labeled data.

**Key insight from the data**: For major misses, the wrong font is
genuinely closer in 71.8% of per-char comparisons. The Fisher weights
give `v_center` 7.2% of total weight — by far the largest single
feature. But `v_center` measures vertical ink position, which is
stable but not very discriminative for distinguishing similar fonts.
Meanwhile, h_crossings get 15.9% total but are scale-noisy. The
feature weighting may be suboptimal.

**Options**:
1. **Learn a distance metric** (Mahalanobis, LMNN) from labeled pairs.
2. **Train a small neural net** on the 100-dim feature vectors with
   font-family labels.
3. **Random forest / gradient-boosted classifier** on feature vectors.

A training set can be generated exhaustively by rendering every
(font × char × DPI) combination through the pipeline and labeling with
ground truth font identity. The `learn_weights.rs` binary already does
this rendering.

**Verdict**: **High potential, separate research track.** The current
Fisher-weighted Euclidean distance is a linear method applied to a
problem that may need nonlinear boundaries. But this is a larger
effort that should run in parallel with the multi-height approach.
(See separate research on ML classifier training.)

---

## The Core Insight (Revised)

The original research framed this as "asymmetric AA is the problem."
The NORM_H=48 vs 32 experiment disproves this — or rather, shows it's
a secondary concern:

**The primary problem is that the features are not scale-invariant,
and the scan side undergoes significant rescaling (median 0.74×
downscale) while the index side undergoes near-zero rescaling.**

The AA fringe asymmetry is a specific manifestation of this broader
problem, affecting only the 9.4% of upscaled crops. For the 89%
of downscaled crops, the problem is information loss from resolution
reduction, not AA bleed.

### Why Higher NORM_H Helps Accuracy

Counterintuitively, the data shows accuracy improves with more
downscaling (larger crops downscaled more). This is because:

1. **Larger crops start with more information**: an 80px crop
   downscaled to 48px loses 40% of its pixels, but 48px is still
   enough to extract meaningful features.
2. **The index side also renders at NORM_H**: at NORM_H=48, the
   index renders high-quality 48px glyphs. Matching a downscaled-to-48
   scan crop against a natively-rendered-at-48 index entry works
   because 48px is above the "feature discrimination threshold."
3. **At NORM_H=32, features degrade on BOTH sides**: the index renders
   at 32px too, so both scan and index have less information. Some
   features (crossings, skeleton, corners) become unreliable at 32px.

**The optimal NORM_H is a trade-off**: high enough that features are
reliable, but not so high that upscaled crops suffer excessive AA
bleed. Given that 89% of crops are downscaled and upscaled crops
are only 9.4%, NORM_H=48 is a reasonable choice (even NORM_H=64
might be better — the accuracy data shows no penalty for more
downscaling in the 80-100px range).

---

## Recommended Path (Updated)

### Priority 1: Multi-Height Index (Approach 7)

**Fastest path to reducing scaling error.** Render index at {32, 48, 72}
and match each scan crop against the closest-height index.

Implementation plan:
1. Modify the index structure to store a `height` alongside each entry.
2. Modify `render_char_normalised` to accept a target height parameter.
3. Build the index at all three heights.
4. At query time: compute `closest_height = argmin(|h - crop_height|)`
   over {32, 48, 72}. Query only that height's entries.
5. Re-run audit to measure improvement.

Expected impact: the worst-case scale factor drops from 3.4× to ~1.5×.

### Priority 2: Feature Noise Investigation

**The per-char data shows features are failing fundamentally** — the
wrong font is closer 71.8% of the time for major misses. Before
optimizing the index, we need to understand WHICH features are driving
the wrong match closer. A per-feature analysis of miss-causing chars
would reveal whether specific features (e.g., h_crossings) are adding
noise rather than signal.

### Priority 3: Gamma-Weighted Ink Values (Approach 6c)

Still a reasonable low-effort improvement for the 9.4% of upscaled
crops. Apply gamma=2.0 to ink weighting in profiles. Low risk, low
complexity, small expected benefit.

### Priority 4: Kill Dead Features

`xh_cap_ratio` has zero Fisher weight. `stroke_contrast` has 0.05%.
Remove them from the feature vector (or give them explicit zero
weight) to reduce noise and index size.

### Priority 5: ML Classifier (Approach 10)

Longer-term: learn a distance metric or classifier from the full
(font × char × DPI) training data. The `learn_weights.rs` binary
already generates this data; the next step is to use it for metric
learning rather than just Fisher ratios.

### What NOT to do:
- Don't reduce NORM_H to 32 — the churn-to-benefit ratio is terrible.
- Don't binarize as a first-line fix (try after multi-height).
- Don't apply asymmetric preprocessing (erosion, sharpening) to one
  side only.
- Don't increase NORM_H above 48 without testing — diminishing returns
  vs increased index render cost.

---

## Experimental Validation Plan (Updated)

1. **Baseline**: current audit at NORM_H=48 (done: 346/480 = 72.1%
   total, 74 major + 56 minor + 4 SSIM).
2. **Multi-height index {32, 48, 72}**: each crop matched to closest
   height. Measure total accuracy and per-height-bucket accuracy.
3. **Per-feature noise analysis**: for each miss, compute per-dimension
   contribution to the distance error. Identify features that
   systematically push toward the wrong font.
4. **Gamma=2.0 at NORM_H=48**: compare on the upscaled crop subset
   (46 lines) specifically, not just overall.
5. **NORM_H=64**: check if accuracy improves for the 80-100px crop
   range (would be less downscaling) without hurting the upscaled range.
6. **Kill stroke_contrast + xh_cap_ratio**: confirm zero regression.

Primary metric: total correct (compared - major - SSIM).
Secondary metric: per-height-bucket accuracy and total churn count.

---

## References

- Otsu, N. (1979). "A threshold selection method from gray-level
  histograms." IEEE Trans. Systems, Man, and Cybernetics.
- Sauvola, J. & Pietikäinen, M. (2000). "Adaptive document image
  binarization." Pattern Recognition, 33(2).
- Epshtein, B., Ofek, E., & Wexler, Y. (2010). "Detecting Text in
  Natural Scenes with Stroke Width Transform." CVPR 2010.
- Hu, M.-K. (1962). "Visual pattern recognition by moment invariants."
  IRE Trans. Information Theory.
- Weinberger, K. Q. & Saul, L. K. (2009). "Distance metric learning
  for large margin nearest neighbor classification." JMLR 10.
- DIBCO benchmark series (2009–2019): Document Image Binarization
  Competition.
