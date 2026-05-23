# Feature Weight Optimization

How the per-feature weights in `as_weighted_slice()` were investigated, what
experiments were run, and how the Noto family dedup delivered the real win.

## Current Architecture

### Weighting: Group L2 (0.40 / 0.30 / 0.30)

| Group            | Dims | Weight | Description |
|------------------|------|--------|-------------|
| Column profile   | 32   | 0.40   | Ink density per vertical slice |
| Original scalars | 7    | 0.30   | aspect, ink_density, v_center, h_balance, serif_score, stroke_contrast, xh_cap_ratio |
| V2 features      | 18   | 0.30   | counters, terminals, boundary, h_crossings |

Within each group, all features are weighted uniformly, then L2-normalized,
then multiplied by the group weight.  This treats the feature vector within
each group as a **shape comparison** — a geometric property that turns out to
be load-bearing (see experiments below).

### Noto Family Dedup

After OT variant dedup and score sorting, all Noto sans-serif fonts are
collapsed into a single candidate slot (keeping the best score).  Separate
groups for NotoSerif and NotoMono.

**Why**: The Noto superfamily shares Latin glyphs across dozens of font files.
NotoSans, NotoSansDisplay, NotoTraditionalNushu, NotoSansMath,
NotoSansSymbols, etc. all embed the same NotoSans Latin outlines.  Different
font metrics (ascent/descent/em) produce slightly different feature-vector
normalizations (score Δ ~0.01–0.03), but the underlying glyphs are identical.
Without dedup, these fonts occupy many CI candidate slots, crowding out the
correct font after the σ cutoff.

**Verification**: Pixel-level comparison of rendered Latin characters confirmed
NotoTraditionalNushu↔NotoSans mean pixel diff of 10–25 (same outlines,
minor hinting differences), vs 44–54 to SourceSans3 (genuinely different
glyphs).

### Score-Based Clone Dedup

A first pass merges adjacent sorted entries within ε=1e-5.  This catches
exact clones from different font packages that produce bitwise-identical
feature vectors.  The Noto family dedup (above) handles the broader case
where shared outlines produce similar but not identical features.

---

## Fisher Discriminant Analysis

### Tool: `learn_weights`

Self-contained binary that performs multi-DPI Fisher analysis across the full
system font catalog.  No dependency on the specimen PDF or external data.

**Pipeline:**
1. Scan all system fonts (4,742 entries)
2. Render every (font × char × DPI) triple — 101 indexed chars at 300, 200,
   and 100 DPI through the same `render_char_normalised` →
   `normalize_to_ink_bounds` pipeline as the index
3. Compute `compute_features()` on each render
4. Dedup identical feature vectors per character (OT variants)
5. Signal = between-font variance per feature (pooled across DPIs)
6. Noise = within-font across-DPI variance per feature
7. Fisher ratio = signal / noise per dimension

**DPI simulation**: Lower DPI is simulated by rendering at full quality, then
downscaling by `dpi/300` and normalizing back through `normalize_to_ink_bounds`.
This models real information loss — same interpolation artifacts as actual
low-resolution scans.

### Data (from full run)

- **1,436,826** (font × char × DPI) triples attempted
- **1,061,283** feature vectors produced
- **264,271** unique after per-character dedup (75% OT variant reduction)
- **353,927** (font, char) pairs with 2+ DPI entries for noise estimation

### Fisher Rankings

**Top 10 features by Fisher ratio:**

| Rank | Feature | Fisher | Note |
|------|---------|--------|------|
| 1 | aspect | 237.1 | Dominant — nearly invariant to DPI |
| 2 | prof10 | 21.5 | Mid-glyph profile bin |
| 3 | prof11 | 21.0 | |
| 4 | prof9 | 20.8 | |
| 5 | prof12 | 20.0 | |
| 6 | prof19 | 19.6 | |
| 7 | prof16 | 19.5 | |
| 8 | prof13 | 19.4 | |
| 9 | prof20 | 19.3 | |
| 10 | prof15 | 19.1 | |

Mid-glyph profile bins (8–25) are the reliable workhorses — high
discrimination, low DPI sensitivity.

**DPI-sensitive features:**

| Feature | 300-only Fisher | Multi-DPI Fisher | Problem |
|---------|----------------|------------------|---------|
| stroke_contrast | 34.3 | 5.6 | Rasterization blur smears contrast |
| term1 | 8.4 | 1.8 | Terminals disappear at low res |
| term3 | 4.1 | 1.9 | Same |
| serif_score | 1.1 | 7.2 | *Improved* — was under-estimated |

**Dead/misleading features:**

| Feature | Fisher | Problem |
|---------|--------|---------|
| xh_cap_ratio | 0.000 | Zero signal, zero noise — computed identically for all fonts |
| v_center | 12.1 | High Fisher but low within-class discrimination (see Experiment 3) |

### Fisher-Optimal Group Weights

| Group | Dims | Original | Fisher optimal |
|-------|------|----------|----------------|
| Column profile | 32 | 0.40 | **0.64** |
| Original scalars | 7 | 0.30 | **0.17** |
| V2 features | 18 | 0.30 | **0.19** |

Fisher says profile was under-weighted and scalars nearly 2× over-weighted.
In practice, applying these group weights regressed accuracy (Experiment 1).

---

## Experiments

All experiments measured against the 6-page font-timeline specimen (30 font
sections, ~486 lines AA at 300 DPI, ~495 lines noaa).

### Baseline

Group L2 with uniform features, weights 0.40/0.30/0.30.

| Test | Score | Lines |
|------|-------|-------|
| AA | 97.5% | 474/486 |
| noaa | 93.1% | 461/495 |

12 AA misses: 8 Noto variants (NotoSans, NotoSansDisplay,
NotoTraditionalNushu) stealing from Arial/Open Sans/SourceSans3/others,
2 Loma→Arial, 1 Impact→Verdana, 1 DejaVuSerif→Playfair.

### Experiment 1: Fisher Group Weights (0.64/0.17/0.19)

Replace hand-tuned group weights with Fisher-optimal ratios.  Still use
uniform features within each L2-normalized group.

| Test | Score | Δ from baseline |
|------|-------|-----------------|
| AA | 95.9% (466/486) | **−1.6%** |

**Why it failed**: Lower scalar weight crushed stroke_contrast, which is the
#1 discriminator at 300 DPI (Fisher=34.3 before multi-DPI dampening).  The
Fisher analysis optimized for DPI robustness, not peak 300 DPI performance.

### Experiment 2: Per-Feature Scale-Adjusted Weights

Each feature gets individual weight: `sqrt(fisher[i]) / std_total[i]`,
normalized to sum=1.  Applied directly to raw features — no group L2
normalization.

| Test | Score | Δ from baseline |
|------|-------|-----------------|
| AA | 95.9% (466/486) | **−1.6%** |
| noaa | 94.1% (466/495) | **+1.0%** |

20 AA misses analyzed:
- 5 short lines ("dogs.") — too few chars for reliable matching
- 12 NotoSans/NotoSansDisplay variants flooding results
- 3 Loma (Thai font) stealing Arial lines
- 3 NotoTraditionalNushu beating SourceSans3 on Latin text

The noaa improvement confirms Fisher weights model DPI robustness correctly.
The AA regression shows they sacrifice shape-comparison properties needed at
300 DPI.

**Per-character analysis** (`feat_diff`): For Source Sans 3 'a' vs both
SourceSans3 and NotoTraditionalNushu reference renders, the correct font was
closer (dist²=0.000013 vs 0.000018).  The problem was at the aggregation/
voting level — not individual character features.

### Experiment 3: v_center Zeroed

The per-feature Fisher analysis gave v_center a 12.2% weight — disproportionate
for a feature that doesn't discriminate between fonts within the same class.
`feat_diff` showed v_center made NotoTraditionalNushu *closer* to scan crops
than SourceSans3 (scan=0.5154, NotoTrad=0.5186 diff=0.003, SS3=0.5085
diff=0.007).

Starting from Experiment 2's per-feature weights, zeroed v_center:

| Test | Score | Δ from Exp 2 | Δ from baseline |
|------|-------|-------------|-----------------|
| AA | 96.7% (470/486) | **+0.8%** | **−0.8%** |
| noaa | 93.1% (461/495) | −1.0% | 0% |

Recovered 4 lines over Experiment 2 but still below baseline.

### Experiment 4: Hybrid — Fisher Per-Feature Within L2 Groups, Fisher Group Weights

Per-feature Fisher weights applied within each group before L2 normalization,
then Fisher group weights (0.64/0.17/0.19) applied to normalized groups.

| Test | Score | Δ from baseline |
|------|-------|-----------------|
| AA | 96.3% (468/486) | **−1.2%** |
| noaa | 89.7% (444/495) | **−3.4%** |

Worst result.  noaa cratered because the combination of per-feature distortion
AND wrong group balance compounded.

### Experiment 5: Hybrid — Fisher Per-Feature Within L2 Groups, Original Group Weights

Same as Experiment 4 but with original group weights (0.40/0.30/0.30).

| Test | Score | Δ from baseline |
|------|-------|-----------------|
| AA | 95.9% (466/486) | **−1.6%** |
| noaa | 92.9% (460/495) | **−0.2%** |

Per-feature weighting within groups hurts even with correct group balance.
Confirms the L2 normalization's shape-comparison property is the thing
that matters, and per-feature scaling destroys it.

### Experiment 6: Noto Family Dedup (final — committed)

Reverted to baseline weights.  Added two-pass clone dedup after OT variant
dedup:
1. Score-epsilon (ε=1e-5) — merges exact clones
2. Noto family grouping — collapses all Noto sans-serif fonts into one slot

| Test | Score | Δ from baseline |
|------|-------|-----------------|
| AA | **99.2%** (482/486) | **+1.7%** |
| noaa | **96.6%** (478/495) | **+3.5%** |

8 of the 12 baseline misses were Noto family variants.  Collapsing them into
one candidate freed CI slots for the correct font.

**Remaining 4 AA misses** (genuine glyph similarity, not indexing artifacts):
- 2× Loma → Arial (Thai font with similar Latin glyphs)
- 1× Impact → Verdana (short text "(Microsoft)")
- 1× DejaVuSerif Bold → Playfair Display (serif confusion)

**Remaining 10 noaa misses**: 4× Impact, 2× Loma, 1× URWBookmanDemi,
1× StandardSymbolsPS, 1× DejaVuSerif, 1× SourceCodePro.  These are harder
cases — different sans-serif confusions amplified at lower DPI.

---

## Summary Table

| Experiment | Change | AA | noaa |
|-----------|--------|-----|------|
| Baseline | Group L2, 0.40/0.30/0.30 | 97.5% | 93.1% |
| 1. Fisher group weights | 0.64/0.17/0.19 | 95.9% (−1.6) | — |
| 2. Per-feature scale-adjusted | No L2, individual weights | 95.9% (−1.6) | 94.1% (+1.0) |
| 3. v_center zeroed | Exp 2 + v_center=0 | 96.7% (−0.8) | 93.1% (±0) |
| 4. Hybrid Fisher/Fisher | Per-feat in L2, Fisher groups | 96.3% (−1.2) | 89.7% (−3.4) |
| 5. Hybrid Fisher/original | Per-feat in L2, original groups | 95.9% (−1.6) | 92.9% (−0.2) |
| **6. Noto family dedup** | **Baseline + clone dedup** | **99.2% (+1.7)** | **96.6% (+3.5)** |

---

## Key Lessons

1. **L2 group normalization is load-bearing.**  It creates an implicit
   shape-comparison geometry where the *distribution* of feature values
   matters, not just individual dimensions.  Every attempt to replace it
   with per-feature weighting regressed AA.

2. **The real problem was structural, not parametric.**  The Noto superfamily
   placing dozens of near-identical Latin-glyph fonts in the candidate pool
   was worth 8× more misses than any weight miscalibration.

3. **Fisher analysis is diagnostic gold, bad prescriptive weights.**  The
   multi-DPI Fisher ratios correctly identified dead features (xh_cap_ratio),
   DPI-fragile features (stroke_contrast, terminals), and misleading features
   (v_center).  But applying them as weights traded 300 DPI accuracy for DPI
   robustness — a trade-off that only pays off if most inputs are low-DPI.

4. **v_center is a trap.**  High Fisher ratio (12.1) because it's stable
   across DPI, but it doesn't discriminate between fonts in the same class.
   It made NotoTraditionalNushu *closer* to Source Sans 3 scan crops than the
   actual Source Sans 3 renders.

5. **noaa is the DPI-robustness signal.**  Experiments that improved noaa
   (per-feature weights: +1.0%) but hurt AA confirmed the Fisher analysis
   is correct about DPI robustness — just not useful for optimizing peak
   300 DPI performance.

---

## Reproducing

```bash
# Build and run Fisher analysis (~5 min, 1.4M renders)
cargo build --release --bin learn_weights
target/release/learn_weights

# Custom DPIs
target/release/learn_weights --dpis 300,200,150,100

# Per-dimension feature diff between two crop PNGs
./target/release/feat_diff scan_crop.png ref_render.png

# Run accuracy tests
cargo test --release --test t60_specimen_accuracy -- --nocapture
```
