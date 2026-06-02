# Char Index Feature Research: Discriminating Close Serif Cousins

**Date:** 2026-05-11
**Problem:** The char index (57 features) only surfaces the correct specimen font in
~26% of lines (top-50 candidates). The remaining 74% of lines never get a chance at correct
identification because the correct font isn't even in the candidate pool.

**Target pairs to discriminate:**
- EB Garamond vs Georgia vs Libre Caslon Text
- Libre Baskerville vs Times New Roman vs NimbusRoman
- Libre Bodoni vs PT Serif vs SourceSerif4
- Zilla Slab vs URWBookman vs Caladea

## 1. Current Feature Inventory (57 features)

### Group 1: Column Ink Profile (32 features, weight 0.40)
- 32-bin resampled column ink density profile
- Measures: horizontal ink distribution across the glyph

### Group 2: Original Scalars (7 features, weight 0.30)
1. `aspect` — ink width / ink height
2. `ink_density` — ink pixels / bbox area
3. `v_center` — vertical center of mass
4. `h_balance` — left-half ink / total ink
5. `serif_score` — serif confidence (row-ink analysis of 'I' and 'l') **per-font, not per-glyph**
6. `stroke_contrast` — p90/p10 horizontal+vertical run lengths
7. `xh_cap_ratio` — x-height / cap-height **per-font metric**

### Group 3: V2 Discriminative (18 features, weight 0.30)
1. `counter_area_ratio` — enclosed whitespace / bbox area
2. `counter_centroid_x` — normalized counter centroid X
3. `counter_centroid_y` — normalized counter centroid Y
4. `counter_aspect` — counter width / height
5-8. `terminal_angles[4]` — 4-bin endpoint direction histogram
9. `ink_perimeter` — boundary pixels / √ink_area
10. `compactness` — 4π × area / perimeter²
11-18. `h_crossings[8]` — ink↔white transitions at 8 scan lines

### Current Weighting
Three-group L2-normalize-then-scale: profile 40%, scalars 30%, v2 30%.
All 57 dimensions live in the same CI with squared Euclidean distance.

---

## 2. What Typography Says Distinguishes Serif Subclasses

The Vox-ATypI classification system identifies these axes (with our target fonts mapped):

| Feature | Old Style (Garamond, Caslon) | Transitional (Baskerville, Times) | Didone/Modern (Bodoni) | Slab (Zilla) |
|---|---|---|---|---|
| **Stress angle** | Oblique (~20-30°) | Nearly vertical (~5-10°) | Vertical (0°) | Vertical (0°) |
| **Stroke contrast** | Low-moderate (2:1–3:1) | Moderate (3:1–4:1) | Extreme (6:1+) | Very low (~1.2:1) |
| **Serif brackets** | Deep, generous | Moderate | None (unbracketed) | None or minimal |
| **Serif thickness** | Medium | Medium-thin | Hairline | Same as stem |
| **x-height** | Small (0.42–0.48) | Medium (0.48–0.52) | Medium (0.50) | Tall (0.53–0.58) |
| **Counter openness** | Open | Moderate | Narrow/tall | Open |
| **Terminal shape** | Teardrop/ball | Ball or tapered | Ball terminal | Flat/squared |

Georgia (screen serif) breaks the pattern: it has a very tall x-height (0.55+), wide set,
low contrast — designed for pixel rendering rather than print tradition.

---

## 3. Proposed New Features

### 3.1. Stress Angle (HIGH IMPACT)

**What it measures:** The angle of minimum stroke thickness in round characters ('o', 'e', 'c', 'O', 'C', 'D'). Old Style fonts have oblique stress (the thinnest part of 'o' is NE/SW), while Transitional/Modern fonts have vertical stress (thinnest at 3 o'clock/9 o'clock).

**Which pairs it separates:**
- **Garamond (oblique ~25°) vs Baskerville (near-vertical ~8°)** — the single most important distinction
- **Caslon (oblique ~20°) vs Times (vertical ~5°)**
- Doesn't help: Baskerville vs Bodoni (both vertical) or Garamond vs Caslon (both oblique)

**How to compute (from 48px binarized glyph render):**
1. Compute distance transform of ink region
2. Find the skeleton (medial axis)
3. At each skeleton point, the distance-transform value = local stroke width
4. In round portions, find the angular position of minimum width
5. Alternatively: for 'o'/'O'/'e', scan radially from centroid measuring ink run-length at each angle. The angle of minimum run-length is the stress angle.

**Simpler approximation:**
For round chars ('o', 'O', 'c', 'C'), divide the glyph into angular quadrants and compare ink density:
- `stress_ratio = density(NE+SW quadrants) / density(NW+SE quadrants)`
- Old Style: stress_ratio < 1.0 (less ink in NE/SW → thinner there)
- Modern: stress_ratio ≈ 1.0 (symmetric)

**Implementation complexity:** Medium. Distance transform is available in imageproc crate. Skeleton extraction needs thinning (Zhang-Suen or similar). The quadrant density approximation is simple — maybe 30 lines of Rust.

**Gotchas:** Only meaningful for round characters. Should be computed for a subset of chars ('o', 'O', 'e', 'c', 'C', 'D', 'Q') and set to 0.0 for others. At 48px, the angle resolution is coarse (~15° precision), but that's enough to separate oblique (25°) from vertical (5°).

**Recommendation: Implement the quadrant-density approximation first.** Two features:
- `stress_obliqueness` — ratio measuring how much NE/SW differs from NW/SE (0 = vertical stress, 1 = strong oblique)
- Only for round chars; 0.0 for all others

---

### 3.2. Serif Bracket Depth (HIGH IMPACT)

**What it measures:** The smoothness of the transition from serif to stem. Deep brackets = gradual curve (Old Style). No brackets = sharp right angle (Didone, Slab).

**Which pairs it separates:**
- **Garamond (deep bracket) vs Bodoni (no bracket)** — night and day
- **Baskerville (moderate bracket) vs Bodoni (none)**
- **Times (moderate) vs Bodoni (none)**
- Helps within Old Style too: Caslon has slightly sharper brackets than Garamond

**How to compute:**
For vertical-stemmed chars with serifs ('I', 'l', 'T', 'i', 'H', 'h', 'd', 'b'):
1. Find the stem: look for a column range where ink is continuous from top to bottom
2. At the base and top, find where the serif protrudes beyond the stem width
3. Measure the curvature of the transition zone:
   - **Sharp bracket (low value):** ink width changes abruptly (1-2 pixel rows)
   - **Deep bracket (high value):** ink width changes gradually (5+ pixel rows)
4. Measure = number of rows between "stem width" and "full serif width" / total height

**Simpler approximation:**
For 'I' or 'l' (or any character with a clear vertical stem + serif):
- Find the serif rows (top/bottom rows where ink extends wider than the stem)
- Compute the "bracket zone" = rows where width is between stem_width and serif_width
- `bracket_depth = bracket_zone_height / serif_height`

**Implementation complexity:** Medium. Finding the stem width is the tricky part — use the horizontal crossing profile to identify the stem region.

**Gotchas:** Requires serif to be present. Meaningless for sans-serif fonts (which will get filtered by existing serif_score). Also, some characters have no clean stem ('o', 's', etc.). Best computed for 'I', 'l', 'T', 'H' only.

---

### 3.3. Stroke Width Variance Profile (MEDIUM-HIGH IMPACT)

**What it measures:** The distribution of stroke widths within a glyph, not just the p90/p10 contrast ratio. Captures the *character* of the contrast, not just its magnitude.

**Current problem:** `stroke_contrast` uses a single p90/p10 ratio, which loses information about the distribution shape. Bodoni and Baskerville might have similar contrast ratios for some glyphs, but Bodoni has a bimodal distribution (very thin + very thick) while Baskerville has a more gradual transition.

**Which pairs it separates:**
- **Bodoni (bimodal: hairline serifs + thick stems) vs Baskerville (unimodal: moderate variation)**
- **Slab serif (tight distribution) vs Transitional (broader)**
- **Georgia (moderate, screen-optimized) vs Times (slightly more contrasty)**

**How to compute:**
Using the distance transform approach:
1. Compute the Euclidean distance transform of the ink region
2. Extract the skeleton (medial axis) — skeleton pixels have their distance-to-boundary
3. The distance-transform values along the skeleton = half-stroke-widths
4. From this distribution, compute:
   - `stroke_width_mean` — normalized by glyph height
   - `stroke_width_std` — standard deviation (high = more contrast)
   - `stroke_width_skewness` — positive skew = mostly thin with some thick (Bodoni); negative skew = mostly thick (Slab)
   - `stroke_width_bimodality` — Sarle's bimodality coefficient or simply (distance between modes) / std

**Simpler approximation (no skeleton needed):**
Use the existing horizontal + vertical run-length data more thoroughly:
- Instead of just p90/p10, compute a 4-bin histogram of run lengths
- Bins: [0-25th%, 25-50th%, 50-75th%, 75-100th%] of run-length range
- This captures the distribution shape, not just extremes

**Implementation complexity:** Medium-High for distance transform + skeleton. Low for run-length histogram.

**Gotchas:** Distance transform is O(W×H), skeleton extraction adds ~O(W×H×iterations). At 48px, both are trivial in practice.

**Recommendation: Start with 4-bin run-length histogram (4 features).** Graduate to distance-transform-based features if needed.

---

### 3.4. Row Ink Profile (MEDIUM IMPACT)

**What it measures:** The vertical ink distribution, analogous to the existing column profile but rotated 90°. A row profile captures ascender/descender proportions, where the crossbar sits, baseline thickness patterns, etc.

**Why the current index misses this:** The column profile (32 bins) captures horizontal structure well, but vertical structure — where the thickest/thinnest parts of the glyph sit vertically — is represented only by `v_center` (a single scalar). That's not enough.

**Which pairs it separates:**
- **Different crossbar positions:** Garamond's 'e' crossbar sits lower than Georgia's. The row profile captures this.
- **Different ascender/descender proportions:** Garamond has longer descenders relative to x-height than Georgia.
- **Serif thickness at baseline vs cap-line:** Bodoni has hairline top serifs but slightly thicker baseline serifs.

**How to compute:**
Exactly like the existing column profile, but scanning row-by-row:
- For each row in the ink bbox, sum ink values
- Resample to 16 bins (fewer than column profile since vertical info is less discriminative)
- Normalize

**Implementation complexity:** Very Low. Copy-paste the column profile code, swap x/y.

**Gotchas:** More affected by rendering height inconsistencies. Using 16 bins keeps the feature count modest.

**Recommendation: Add a 16-bin row profile.** Cheap, adds genuine new information.

---

### 3.5. Width-Normalized Character Width (MEDIUM IMPACT — Mike's Suggestion)

**What it measures:** The rendered width of the character relative to its height, but using the *actual font metrics* (advance width) rather than the ink bbox. This captures the "set width" or "tracking" — how wide the font's design intends each character to be.

**Why it matters for the index:** The existing `aspect` feature measures ink bbox width / height, which is close but not identical. Advance width includes side bearings, which vary between fonts. Garamond is narrower-set than Georgia. Times is narrower than Baskerville.

**Current situation:** Width ratio is already used as a **pre-filter** in font_match.rs, comparing rendered line width against OCR bbox width. Moving it into the index would let it contribute to CI distance rather than just being a binary pass/fail gate.

**Which pairs it separates:**
- **Georgia (wide set) vs Times (narrow set)** for the same character
- **Garamond (narrow) vs Caslon (slightly wider)**
- **Bodoni (moderate) vs SourceSerif4 (wider)**

**How to compute:**
Already have the rendered glyph. Compute:
- `advance_width_ratio = font.h_advance(glyph_id) / ascent_height` at reference scale
- Or simpler: `set_width = (ink_bbox_width + 2*side_bearing) / glyph_height`

The `aspect` feature already captures ink_width/ink_height, but adding the *advance width* (ink + side bearings) as a separate feature captures the font's intended spacing.

**Implementation complexity:** Very Low. Already have the font object and scale. One call to `h_advance()`.

**Gotchas:** Some fonts have broken metrics. Needs sanity check (0.1 < ratio < 3.0 or similar).

**Recommendation: Add `advance_width_ratio` as a single scalar feature.**

---

### 3.6. Diagonal Feature Profile (MEDIUM IMPACT)

**What it measures:** Ink density along the two diagonals of the bounding box (NW→SE and NE→SW). Captures the *stress pattern* from a different angle than the column/row profiles.

**Which pairs it separates:**
- **Old Style (oblique stress → more ink along NE-SW diagonal) vs Modern (vertical stress → symmetric diagonals)**
- Complementary to stress angle — captures the same typographic property but through a different measurement

**How to compute:**
Divide the ink bbox into 2 diagonal bands (NW→SE and NE→SW). For each:
- Sum ink along that diagonal direction in 8 bins
- Compute ratio = NE-SW ink / NW-SE ink

Or simpler: just 2 scalar features:
- `diag_balance_nwse` = ink in NW→SE diagonal half / total ink
- `diag_balance_nesw` = ink in NE→SW diagonal half / total ink

**Implementation complexity:** Low. A single pass over pixels, classifying each by which diagonal half it falls in.

**Gotchas:** Only meaningful for round/diagonal characters. For vertical-only chars like 'I' or 'l', both diagonals will be similar regardless of font.

---

### 3.7. Serif Protrusion Ratio (MEDIUM IMPACT)

**What it measures:** For stemmed characters, how much the serif extends beyond the stem width, relative to the stem width itself.

**Which pairs it separates:**
- **Slab (protrusion ≈ stem width) vs Old Style (protrusion < stem width)**
- **Bodoni (protrusion minimal, hairline) vs Baskerville (moderate protrusion)**
- **Georgia (wider serifs) vs Times (narrower serifs)**

**How to compute:**
For 'I', 'l', 'T', 'h', 'd':
1. Find the stem width (mode of column ink widths in the middle third)
2. Find the serif width (max ink width in top/bottom 15%)
3. `serif_protrusion = (serif_width - stem_width) / stem_width`

**Implementation complexity:** Low-Medium. Reuses concepts from the bracket depth measurement.

---

### 3.8. 'e' Aperture Openness (MEDIUM IMPACT)

**What it measures:** How open or closed the counter of 'e' is. The aperture — the gap between the bowl and the crossbar — varies dramatically between fonts and is one of the most distinctive features typographers use to identify fonts.

**Which pairs it separates:**
- **Garamond (very open 'e') vs Georgia (moderately open)**
- **Bodoni (narrow aperture) vs Baskerville (wider)**
- This feature is character-specific but the char index already stores per-character features

**How to compute:**
For 'e' specifically:
1. Find the counter (already have `compute_counter_features`)
2. Find the aperture opening — the gap in the right side of the counter
3. Measure its angular span or height relative to counter height

More generally, for any character with a counter ('a', 'e', 'c', 's', 'C', 'G', 'S'):
- `aperture_openness` = width of the opening / perimeter of the counter

**Implementation complexity:** Medium. Counter detection is already done; need to find the opening.

**Gotchas:** Sensitive to rendering quality at 48px. The aperture of 'e' in Garamond at 48px is only a few pixels wide.

---

### 3.9. Frequency Domain Features — DCT Coefficients (MEDIUM IMPACT)

**What it measures:** Low-frequency DCT (Discrete Cosine Transform) coefficients capture the overall shape/texture of the glyph in a resolution-independent way. Higher frequencies capture fine details like serifs and stroke modulation.

**Academic support:** Bozkurt et al. (2014, arXiv:1407.2649) used Complex Wavelet Transform features for font classification and achieved higher accuracy than spatial-domain features alone. Wavelet/DCT features capture texture and structural patterns that spatial profiles miss.

**Which pairs it separates:**
- Fine serif details that distinguish fonts at the texture level
- Stroke modulation patterns (gradual vs abrupt thickness changes)
- Overall glyph "feel" that's hard to capture with spatial features

**How to compute:**
1. Resize glyph render to 32×32 (or 16×16 for fewer features)
2. Apply 2D DCT (Type II)
3. Take the top-K low-frequency coefficients (zigzag scan)
4. E.g., first 16 coefficients from an 8×8 DCT = captures overall shape

Using the `rustdct` crate or manual DCT implementation (straightforward for small sizes).

**Implementation complexity:** Medium. DCT is standard but adds a dependency. For 8×8 blocks, a hand-written DCT is ~40 lines.

**Gotchas:** DCT coefficients are scale-sensitive — must normalize the glyph to a fixed size first (already do this: NORM_H=48). The number of useful coefficients needs tuning. Too many = noise in CI (curse of dimensionality).

**Recommendation: Add 8-16 DCT coefficients from an 8×8 block DCT of the normalized glyph.**

---

### 3.10. Learned Embeddings — Contrastive/Triplet Learning (HIGH IMPACT, HIGH EFFORT)

**What it measures:** A neural embedding that captures font identity in a low-dimensional space, trained specifically to pull same-font glyphs together and push different-font glyphs apart.

**Academic support:**
- **DeepFont (Adobe, 2015):** CNN achieves >80% top-5 accuracy on 2383 fonts. Uses domain adaptation for synthetic→real transfer.
- **Font Representation Learning via Paired-Glyph Matching (2022):** Trains on glyph pairs, achieves strong font retrieval.
- **FontCLIP (2024):** Vision-language model connecting CLIP embeddings with typographic attributes.
- **Total Disentanglement of Font Images (2024):** Separates style from character class, achieving font-agnostic style vectors.

**The approach for unscan:**
1. Render all 101 indexed characters × ~4900 fonts = ~495K training images
2. Train a small CNN (e.g., 3-layer ConvNet) with triplet loss:
   - Anchor: glyph 'a' in Garamond
   - Positive: glyph 'a' in Garamond (different rendering/augmentation)
   - Negative: glyph 'a' in Georgia
3. Output: 16-32 dimensional embedding per glyph
4. Use these embeddings as features in the CI (replacing or supplementing handcrafted features)

**Why it could be transformative:**
The model would learn exactly what distinguishes close serif cousins at the pixel level — subtle curve shapes, thickness transitions, ink traps, etc. that no handcrafted feature captures.

**Implementation complexity:** High. Requires:
- Training infrastructure (PyTorch or tch-rs)
- Training data generation (already have font rendering)
- Model training (hours of GPU time)
- Inference integration (ONNX runtime or embedded weights)
- Ongoing maintenance when font catalog changes

**Practical concern for unscan:** The tool is a Rust CLI that processes scanned PDFs. Adding a neural network dependency (ONNX runtime, libtorch, etc.) significantly increases binary size and build complexity. 

**Alternative: Offline pre-computation.** Train the model offline, extract embeddings for all fonts, store in the char index. At match time, you'd need to run inference on the scan crop — which requires the model at runtime. Unless you can use the learned embedding as a *training signal* to weight the handcrafted features...

**Recommendation: Park this for now.** Focus on handcrafted features first. If those get us to 60-70% accuracy, consider learned embeddings to push to 80%+. If handcrafted features plateau at 40-50%, jump straight to this.

---

## 4. Quick-Win Implementation Priority

Ordered by (expected impact × ease of implementation):

| Priority | Feature | New Dims | Impact | Effort | Notes |
|---|---|---|---|---|---|
| **1** | Row ink profile (16 bins) | +16 | Medium | Very Low | Copy of column profile, rotated 90° |
| **2** | Advance width ratio | +1 | Medium | Very Low | Mike's suggestion; move width ratio into index |
| **3** | Stress angle (quadrant density) | +1-2 | High | Low-Med | Only for round chars |
| **4** | Serif bracket depth | +1-2 | High | Medium | Only for stemmed chars |
| **5** | Stroke width histogram (4 bins) | +4 | Med-High | Low | Run-length distribution, not just p90/p10 |
| **6** | Serif protrusion ratio | +1 | Medium | Low-Med | serif_width / stem_width |
| **7** | Diagonal balance | +2 | Medium | Low | NW-SE vs NE-SW ink ratio |
| **8** | DCT coefficients | +8-16 | Medium | Medium | Frequency domain texture |
| **9** | 'e' aperture | +1 | Medium | Medium | Character-specific |
| **10** | Learned embeddings | +16-32 | High | Very High | Requires training pipeline |

**Proposed new total: ~57 + 28 = ~85 features** (adding priorities 1-7).

---

## 5. Weight Rebalancing Required

The current 3-group weighting (40/30/30) will need revision when adding features.
Possible new groupings:

- **Profile features** (32 column + 16 row = 48 dims): weight 0.35
- **Typographic scalars** (aspect, density, v_center, h_balance, serif_score, stroke_contrast, xh_cap_ratio, advance_width, stress_angle, bracket_depth, serif_protrusion = ~11 dims): weight 0.30
- **Shape features** (4 counter + 4 terminal + 2 boundary + 8 crossings + 4 stroke histogram + 2 diagonal = ~24 dims): weight 0.25
- **DCT/frequency** (8-16 dims, if added): weight 0.10

The CI with Euclidean distance in 85+ dimensions will suffer more from the curse
of dimensionality. Consider:
- **PCA reduction** to ~40 dimensions before building the tree
- **Per-dimension variance weighting** (already have σ per dimension — could use 1/σ weighting)
- **Multiple trees** (random projection forests / annoy-style) for approximate NN

---

## 6. Resolved: Brute-Force Replaced Tree-Based Search

The project now uses brute-force linear scan for nearest-neighbor search.
At 59+ dimensions, tree-based structures (k-d trees, ball trees) degrade to
near-linear scan anyway. The flat vector approach is simpler, cache-friendly,
and LLVM auto-vectorizes the distance loop.

**Previous alternatives considered** (no longer needed):
1. Ball trees — tighter bounding volumes than k-d trees
2. VP-trees — designed for metric spaces
3. Annoy — random projection forests
4. Product Quantization — compressed approximate search
5. PCA dimensionality reduction — reduce to 20-30 dims first

At the current scale (~4900 fonts × 101 chars = ~495K vectors), brute-force
with SIMD scans ~495K vectors in ~2ms. The investment is better spent on
feature quality than search structure optimization.

**Status:** Resolved. Brute-force linear scan is the production search method.

---

## 7. The Asymmetric Matching Problem

A subtlety the current approach may be missing: when comparing a **scan crop** to
**rendered reference glyphs**, the features should ideally be computed the same way.
But scan crops have:
- Anti-aliasing artifacts from rasterization
- Possible slight rotation/skew
- Different binarization thresholds
- Sub-pixel alignment differences

This means the feature vectors from scan crops are systematically offset from rendered
reference vectors. The CI search finds the nearest neighbor in *feature space*, but
the nearest neighbor might be the rendered version of a different font that happens to
have similar anti-aliasing artifacts rather than the correct font with slightly different
rendering.

**Mitigation strategies:**
1. **Augmented index:** Render each font at multiple slight variations (±1px shift,
   slight blur, different threshold) and index all variants. Increases index size 3-5×
   but makes the feature space more robust.
2. **Feature normalization by estimated noise:** If we know the typical feature offset
   between scan and render, subtract it before search.
3. **Mahalanobis distance** instead of Euclidean, using per-font covariance matrices
   to account for known rendering variability.

---

## 8. Academic References

1. **DeepFont** (Wang et al., 2015, arXiv:1507.03196) — CNN-based, 80%+ top-5 accuracy on 2383 fonts. Key insight: domain adaptation between synthetic training data and real-world photos.

2. **Complex Wavelet Transform for Fonts** (Bozkurt et al., 2014, arXiv:1407.2649) — CWT features + SVM outperform spatial features alone. Wavelet subbands capture texture patterns invisible to spatial profiles.

3. **Font Representation Learning via Paired-Glyph Matching** (2022, arXiv:2211.10967) — Contrastive learning on glyph pairs produces generalizable font embeddings.

4. **TrueType Transformer (T3)** (2022, arXiv:2203.05338) — Processes TrueType outlines directly (not rasterized) for font classification. Interesting for unscan since we have the font files.

5. **Total Disentanglement of Font Images** (2024, arXiv:2403.12784) — Separates style from character, achieving font-independent style vectors.

6. **Vox-ATypI Classification** — Industry-standard typographic classification based on stress angle, contrast, serif shape, x-height, and historical period.

---

## 9. Concrete Next Steps

### Phase 1: Quick handcrafted wins (est. 2-4 hours)
1. Add row ink profile (16 bins)
2. Add advance_width_ratio
3. Reweight groups appropriately
4. Bump INDEX_VERSION, rebuild, measure accuracy

### Phase 2: Typographic features (est. 4-8 hours)  
1. Implement stress_angle via quadrant density
2. Implement serif_bracket_depth for stemmed chars
3. Implement stroke_width_histogram (4 bins)
4. Measure accuracy delta per feature

### Phase 3: Structural improvements (est. 8-16 hours)
1. Benchmark brute-force vs CI at current scale
2. If needed, switch to ball tree or approximate NN
3. Consider PCA dimensionality reduction
4. Test augmented index (render with blur/shift variants)

### Phase 4: Learned embeddings (est. 40+ hours)
1. Only if Phase 1-3 plateau below 70% accuracy
2. Train triplet-loss CNN on rendered glyphs
3. Integrate via ONNX runtime or pre-computed embeddings
