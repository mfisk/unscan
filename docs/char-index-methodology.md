# Per-Character Font Index — Methodology & Assessment

**Module:** `src/char_index.rs`  
**Purpose:** Fast pre-filter to narrow 5,048 candidate fonts to ~50 before expensive SSIM reranking.  
**Date:** 2026-05-10

---

## 1. Architecture Overview

The existing font-matching pipeline in `font_match.rs` runs a multi-signal coarse scorer (IoU, NCC, Hu moments, fill ratio) against every font for every OCR line, then SSIM-reranks the top 30. At 5,048 fonts × ~94 lines per page, this is the dominant cost — roughly 7 minutes per page.

The char index is a **pre-filter stage** that slots in before the coarse scorer:

```
OCR line
  → extract characters from longest words (char_index.rs)
  → match against pre-built per-char index → top 50 fonts
  → coarse score those 50 (font_match.rs)
  → SSIM rerank top 30 of those (verify.rs)
  → best match
```

The goal is to reduce the coarse scoring from 5,048 candidates to ~50, cutting per-line matching time by ~100×.

---

## 2. What Gets Indexed

### Character set (101 characters)

| Group | Count | Characters |
|-------|-------|-----------|
| Lowercase | 26 | a–z |
| Uppercase | 26 | A–Z |
| Digits | 10 | 0–9 |
| ASCII punctuation | 32 | `! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ `` { \| } ~` |
| Typographic specials | 7 | em dash (—), en dash (–), '' "" … |

**Why these?** They cover virtually every character that appears in English-language scanned documents. The typographic specials matter because OCR frequently encounters smart quotes and em dashes in professionally typeset text, and these characters have high variance across font families (an em dash in Garamond looks nothing like one in Futura).

### What's NOT indexed

- Accented characters (é, ñ, ü, etc.) — limits non-English utility
- Ligatures (fi, fl, ff) — these are common in quality typography and highly discriminative
- Math symbols, currency symbols
- Characters above U+2026

**Assessment:** The 101-character set is adequate for English but the absence of ligatures is a missed opportunity. The fi ligature alone is one of the most font-distinctive characters — its presence or absence, and its shape, immediately separates font families.

---

## 3. Index Construction

### Per-character rendering

For each font × each character:

1. **Parse font** via `ab_glyph::FontRef`. Skip if the glyph is `.notdef` (font doesn't contain the character).

2. **Measure ink height** at a reference scale of 200px em-height. Render the glyph, measure its pixel bounding box height.

3. **Normalize to 48px ink height.** Compute the scale factor: `target_scale = 200 × (48 / measured_ink_height)`. Re-render at this scale.

4. **Tight-crop** to the glyph's pixel bounds plus 2px padding.

5. **Compute features** from the resulting grayscale image.

**Why 48px?** Large enough for the 32-bin column profile to have meaningful resolution (~1.5 px per bin for a typical character), small enough to keep rendering fast and memory bounded.

**Assessment — rendering approach is solid.** Using ink height (not em height) for normalization is correct — it means the features describe the character's actual visual appearance independent of the font's internal metrics. Two fonts that render 'a' identically but differ in their em-square definitions will produce identical features.

---

## 4. Feature Vector (36 floats)

### 4.1 Column Ink Density Profile (32 floats)

**What:** For each column of pixels within the ink bounding box, sum the ink darkness (255 − pixel value, thresholded at 200). Normalize so the maximum column is 1.0. Linearly resample to exactly 32 bins.

**What it captures:** The horizontal rhythm of ink distribution. Thick strokes produce high bins, thin strokes produce low bins, counters (enclosed whitespace) produce near-zero bins. This is the primary shape descriptor.

**Example discrimination:** An 'a' in Bodoni has extreme contrast (razor-thin horizontals, thick verticals) producing a spiky profile with high peaks at the vertical strokes and near-zero valleys at the hairlines. An 'a' in Futura has nearly uniform stroke width, producing a smoother, flatter profile. These are trivially separable.

**Weakness — bin count vs. character width:**

- Narrow characters ('i', 'l', '1', '.', '!') may have only 3–8 columns of ink before resampling to 32 bins. Linear interpolation of 5 values to 32 bins produces a smooth curve that loses all structural detail — the profile of 'i' in Times looks nearly identical to 'i' in Helvetica because there's simply not enough source data.

- Wide characters ('W', 'M', '@') have plenty of source columns and the profile is genuinely informative.

- **Net effect:** The pre-filter will work well when matching is based on wide characters (extracted from long words) but poorly when it must rely on narrow characters.

### 4.2 Aspect Ratio (1 float)

**What:** `ink_width / ink_height` of the tight bounding box.

**What it captures:** Whether the character is naturally wide or narrow at the normalized height. Condensed fonts produce lower aspect ratios; extended fonts produce higher ones.

**Assessment:** Genuinely useful. Even among similar-looking serifs, Caslon has wider 'e' than Baskerville. Cheap to compute, non-redundant with the profile.

### 4.3 Ink Density (1 float)

**What:** `ink_pixel_count / (ink_width × ink_height)`. How much of the bounding box is filled with ink.

**What it captures:** Stroke weight and counter openness. Bold fonts have high density; light weights have low density. Fonts with large open counters (Futura) have lower density than fonts with smaller counters (Bodoni).

**Assessment:** Moderately useful. The threshold at pixel value 200 means anti-aliased edges (which vary with the rendering engine) affect the count. At 48px this is a minor issue, but it means the metric is somewhat dependent on rendering parameters.

### 4.4 Vertical Center of Mass (1 float)

**What:** Ink-weighted average Y position, normalized to 0.0 (top) – 1.0 (bottom).

**What it captures:** Whether the character's visual weight sits high, low, or centered. An 'A' has weight at the bottom (high v_center); a 'V' has weight at the top (low v_center). More subtly, fonts with a high x-height shift the center of mass for lowercase letters.

**Assessment:** Useful for distinguishing font classes (high x-height vs. low x-height) but low discrimination between fonts within the same class. Garamond, Caslon, and Baskerville all have similar x-height ratios and will produce nearly identical v_center for the same character.

### 4.5 Horizontal Balance (1 float)

**What:** `ink_in_left_half / total_ink`, where left half is defined by the midpoint of the full image (not the ink bbox).

**What it captures:** Whether ink weight is left-biased, right-biased, or centered. An 'R' is left-heavy; a 'J' is right-heavy. Italic fonts shift horizontal balance relative to upright versions.

**Assessment:** Weakly useful. For most characters this is close to 0.5 for all fonts, providing almost no discrimination. It becomes informative mainly for asymmetric characters (f, r, J, 7) and italic detection.

**Bug:** `mid_x` is computed from the **full image** width (including padding), not the ink bounding box. Since the tight-crop adds variable 2px padding, this introduces noise into what should be a pure shape metric.

---

## 5. Feature Weighting Problem (CRITICAL)

The feature vector is 36 floats: 32 profile bins + 4 scalars. Matching uses flat cosine similarity over all 36 dimensions.

**The profile dominates.** In a cosine similarity computation, each dimension contributes proportionally to its magnitude. The 32 profile bins (each 0.0–1.0) will typically have L2 norm ~2–4, while the 4 scalars (each 0.0–1.0) have L2 norm ~1. The profile contributes roughly **88–90%** of the similarity score.

**The 4 carefully designed scalar features are nearly irrelevant.** A font that has a perfect profile match but wildly wrong aspect ratio will still score ~0.95 similarity. This defeats the purpose of the multi-feature design.

### Fix

Replace flat cosine similarity with **weighted block cosine** or **normalized concatenation**:

```rust
// Normalize each feature block to unit L2, then concatenate with weights
let profile_norm = normalize_l2(&profile);  // 32 floats, unit length
let scalars_norm = normalize_l2(&[aspect, density, v_center, h_balance]);  // 4 floats, unit length

// Weight: 60% profile, 40% scalars (or use learned weights)
similarity = 0.6 * cosine(query_profile, index_profile)
           + 0.4 * cosine(query_scalars, index_scalars)
```

This gives the scalar features 40% influence instead of ~11%.

---

## 6. Character Extraction from Scan

### Word selection

Words are sorted by character count descending. The algorithm takes characters from the longest words first (≥3 characters), collecting up to 3 samples per character and stopping when every observed character has ≥2 samples.

**Rationale — correct.** Longer words have more characters to extract per segmentation attempt, and the per-character width estimates are more reliable (bbox noise is amortized across more characters). Short words (1–2 chars) are excluded because segmentation is impossible.

### Character segmentation (valley detection)

1. Compute per-column ink sums across the word image
2. Smooth with a 3px box filter
3. Find all local minima (valleys) in the smoothed signal
4. Sort valleys by depth (lowest ink first — deepest valleys are best split points)
5. Take the N−1 deepest valleys as split points (for N characters)
6. If too few valleys found, fall back to uniform segmentation

**Assessment — fragile for proportional fonts:**

- **Touching/overlapping characters:** In tightly kerned or ligature-heavy text, characters share columns of ink. The valley between 'fi' may be shallower than the valley inside 'a's counter, causing misplacement.

- **Characters with disjoint parts:** 'i', 'j', '!', '?' have dots/descenders separated by whitespace that creates false valleys within a single character.

- **Uniform fallback is crude:** When valley detection fails, uniform segmentation assumes all characters have equal width. For proportional fonts, this places boundaries through the middle of wide characters ('m', 'W') and gives excess whitespace to narrow ones ('i', 'l').

**Mitigation idea:** Use the index itself for bootstrapping — render the OCR text in a "generic" font (say, the current best-guess font or a default serif/sans), measure the proportional widths, and use those as initial boundary estimates instead of either valley detection or uniform splitting. This is the approach used by WhatTheFont's extraction phase.

### Height normalization

Character crops are resized to `NORM_H` (48px) tall using Lanczos3 interpolation, with width scaled proportionally. This matches the index's normalization.

**Assessment — correct.** Using the full word-crop height (not the individual character's ink height) means all characters from the same word maintain consistent scale, which is critical for cross-character comparison.

---

## 7. Matching Algorithm

### Per-line matching flow

1. Extract character crops from the line's longest words
2. Compute feature vectors for each crop
3. For each font in the index, compute mean cosine similarity across all crop-character matches (only characters the font contains)
4. Return top N fonts by descending score

### Score aggregation

For a font F and extracted characters {c₁, c₂, ..., cₖ}:

```
score(F) = (1/k) × Σᵢ cosine(features(cᵢ_crop), features(cᵢ_index_F))
```

Characters that appear multiple times (e.g., 'e' extracted from three different words) each contribute independently to the average.

**Assessment — sound in principle but with issues:**

1. **No character weighting.** An 'e' contributes equally to an 'M'. But 'M' is far more discriminative (more ink, more structural complexity) and should have higher weight. A simple improvement: weight each character's contribution by its profile variance (high-variance profiles are more distinctive).

2. **Missing character penalty is absent.** If font F is missing glyph 'ë' but no extracted characters include 'ë', the font isn't penalized. This is correct. But if extracted characters include a character the font lacks, it's simply skipped (`count` isn't incremented). This means a font missing 90% of the query characters but matching perfectly on the remaining 10% scores 1.0. There should be a minimum coverage requirement.

3. **Duplicate samples are averaged, not aggregated.** Three samples of 'e' produce three similarity scores that get averaged together. This is fine — it's effectively a denoising step via the mean.

### O(n²) lookup construction bug

The `match_line_chars` function builds a per-font char map by iterating **all fonts × all characters × all entries per character** — O(fonts² × chars). With 5,048 fonts × 101 chars, this is ~2.6 billion iterations just to construct the lookup table, **before any actual matching begins**.

The fix is trivial: the index should be stored as `HashMap<(font_name, char), CharFeatures>` or, better, `HashMap<font_name, HashMap<char, CharFeatures>>` so lookup is O(1). The current structure (char → Vec<FontCharEntry>) is optimized for building but terrible for querying.

---

## 8. Serialization

Binary format: u32 character count, then per-character: char as u32, font count, then per-font: name length + bytes + 36 floats.

**Estimated size:** 5,048 fonts × 101 chars × (4 + 30 + 144) bytes ≈ **91 MB** uncompressed. With typical font name lengths averaging 25–30 bytes.

**Assessment:** Functional but wasteful. The font name is repeated ~101 times per font (once per indexed character). A more efficient format would use a font name string table with integer indices, reducing size to ~74 MB. Adding zstd or lz4 compression would likely bring this to ~15–25 MB. For a cached, build-once file this is not critical, but 91 MB is large enough to be annoying.

**No versioning.** If the feature vector changes (add a feature, change PROFILE_BINS), old index files will silently produce garbage. A magic number and version byte at the start would catch this.

---

## 9. Comparison to Known Approaches

### WhatTheFont (MyFonts/Monotype)

Uses a deep CNN trained on rendered font samples. Feature extraction is learned, not hand-designed. Operates on whole-word images. The CNN approach inherently learns which features matter for font discrimination, avoiding the manual feature weighting problem we have. However, it requires training data and GPU inference.

**Our approach vs.:** Much simpler, no ML dependency, runs on CPU. But our hand-designed features will never match the discriminative power of a learned representation.

### Identifont (feature-question approach)

Uses a questionnaire of ~20 typographic features: "Does the 'a' have a double story?", "Is the 'G' spurred?", "Are the serifs bracketed?" Each answer prunes the candidate set. This is a classification approach, not a continuous-similarity approach.

**Our approach vs.:** We don't capture any of these structural/categorical features. We can't distinguish single-story vs. double-story 'a' from density profiles — both can have similar ink distribution. Identifont's features are more semantically meaningful but require manual annotation of the font catalog.

### Adobe Fonts / Typekit matching

Uses glyph outline similarity (Bézier curve matching) rather than raster comparison. This is inherently more precise because it operates on the vector representation, avoiding rendering artifacts.

**Our approach vs.:** We can't do outline matching because we're starting from a raster scan, not a vector font. The raster-to-raster comparison is the correct approach for our problem. However, the index side could use outline-derived features (curvature histograms, stroke count, junction topology) since we have the font files.

---

## 10. Critical Assessment Summary

### What works well

1. **Per-character decomposition is the right architecture.** Comparing individual characters rather than whole lines avoids the problem of word spacing, line length, and text content variation. 

2. **Column density profile is the best single feature.** It captures horizontal stroke rhythm, which is the most discriminative visual property of a typeface. The 32-bin resampling provides a fixed-length representation.

3. **Longest-word-first extraction is smart.** More characters per word = better segmentation = less noise per character.

4. **Index-then-match is the right pattern.** One-time O(N) index build, then O(N) per-line matching, vs. the current O(N) coarse scoring that's effectively the same cost but without the pre-filtering benefit.

### What needs fixing

| Issue | Severity | Fix |
|-------|----------|-----|
| **Feature weighting imbalance** — profile bins dominate cosine similarity, 4 scalar features contribute ~11% | **High** | Normalize each feature group to unit L2, then weight: 60% profile + 40% scalars |
| **O(n²) lookup construction** — `match_line_chars` rebuilds per-font map by iterating all entries | **High** | Restructure index as `HashMap<String, HashMap<char, CharFeatures>>` |
| **No index versioning** — format changes silently corrupt results | **Medium** | Add 4-byte magic + version byte header |
| **Font name repetition** — name stored 101× per font in binary | **Medium** | String table with integer indices |
| **h_balance bug** — uses full-image midpoint instead of ink-bbox midpoint | **Medium** | Change `mid_x = w / 2` to `mid_x = min_x + (max_x - min_x) / 2` |
| **No minimum coverage threshold** — font matching 1 of 20 chars can score 1.0 | **Medium** | Require ≥50% character overlap; penalize missing chars |
| **Narrow character profiles** — 'i', 'l', '1' resampled from <8 columns to 32 bins | **Low** | Weight character contribution by source-column count, or use adaptive bin count |
| **Valley segmentation fragility** — fails on tight kerning, ligatures, 'i'/'j' dots | **Low** | Use proportional-width estimates from a reference font as initial boundaries |
| **Missing ligatures** — fi, fl, ff not indexed | **Low** | Add common ligatures to `indexed_chars()` |

### Bottom line

The per-character index is a **sound pre-filter architecture** with a **correct rendering/normalization pipeline** and a **reasonable feature set** that has **two critical implementation bugs** (weighting imbalance, O(n²) lookup) and several medium-severity issues. Once the weighting fix and lookup restructuring are applied, this should effectively cut the font catalog to 50–100 candidates, reducing per-line matching from ~4 seconds to ~0.1 seconds.

The feature set will reliably separate font *classes* (serif vs. sans, thin vs. bold, condensed vs. regular) and do a reasonable job within classes. It will struggle to distinguish very similar fonts within the same family (Libre Baskerville vs. Libre Caslon vs. Noto Serif) — but that's acceptable for a pre-filter, because the SSIM reranker handles fine discrimination. The pre-filter just needs to not accidentally eliminate the correct font from the top 50.
