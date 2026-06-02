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

### Character set (106 characters)

| Group | Count | Characters |
|-------|-------|-----------|
| Lowercase | 26 | a–z |
| Uppercase | 26 | A–Z |
| Digits | 10 | 0–9 |
| ASCII punctuation | 32 | `! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ `` { \| } ~` |
| Typographic specials | 7 | em dash (—), en dash (–), '' "" … |
| Ligatures | 5 | ff (U+FB00), fi (U+FB01), fl (U+FB02), ffi (U+FB03), ffl (U+FB04) |

**Why these?** They cover virtually every character that appears in English-language scanned documents. The typographic specials matter because OCR frequently encounters smart quotes and em dashes in professionally typeset text, and these characters have high variance across font families (an em dash in Garamond looks nothing like one in Futura). The five standard Latin ligatures are highly discriminative — a font either has a ligature substitution or it doesn't, and the ligature glyph shape varies dramatically across families.

### Character weights

Not all characters are equally discriminative. `char_weight()` assigns weights
that scale each character's contribution to the CI score:

| Weight | Characters | Rationale |
|--------|-----------|-----------|
| 2.0 | ff, fi, fl, ffi, ffl (ligatures) | Binary signal — font has it or doesn't. Highly distinctive shapes |
| 1.5 | g, a, e, R, Q, G, S, f, t, y, &, @ | Complex structure, high inter-font variance |
| 1.2 | k, w, x, z, A, B, E, F, K, M, N, W | Good structural complexity |
| 1.0 | *(default)* | Standard contribution |
| 0.8 | b, d, p, q, n, u, o, c, O, C, D | Symmetric/common shapes, moderate discrimination |
| 0.5 | I, l, 1, \|, !, ., ,, :, ;, - | Narrow or simple — prone to rasterization noise |

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

## 4. Feature Vector (99 floats, 5 groups)

The feature vector has evolved significantly from the original 36-float design.
See `FEATURES.md` for the complete layout. Summary:

| Group | Dims | Weight | What it captures |
|-------|------|--------|------------------|
| Column ink profile | 32 | 0.40 | Horizontal ink distribution rhythm |
| Scalar v1 | 7 | 0.30 | Core geometry: aspect, density, v_center, h_balance, serif_score, stroke_contrast, xh_cap_ratio |
| Scalar v2 | 18 | 0.30 | Counters (4), terminal angles (4), shape (2), horizontal crossings (8) |
| Row ink profile | 32 | 0.30 | Vertical ink distribution (ascender/x-height/baseline/descender) |
| Scalar v3 | 10 | 0.20 | Holes, symmetry, skeleton topology, corners, quadrant density |

Each group is independently L2-normalized to unit length, then scaled by its
weight. This ensures no group dominates the distance computation purely through
dimension count — a problem that plagued the original flat-vector design where
the 32-bin profile contributed ~88% of the similarity score.

---

## 5. Feature Weighting (FIXED)

Each of the five feature groups is independently L2-normalized to unit length,
then scaled by a per-group weight. This ensures every group contributes
proportionally to the distance metric regardless of its dimensionality.

| Group | Dims | Weight | Rationale |
|-------|------|--------|-----------|
| Column profile | 32 | 0.40 | Highest single-group discriminative power |
| Scalar v1 | 7 | 0.30 | Core geometric features |
| Scalar v2 | 18 | 0.30 | Structural/topological features |
| Row profile | 32 | 0.30 | Complements column profile |
| Scalar v3 | 10 | 0.20 | Fine-grained discriminators, lower individual Fisher ratios |

Weights were tuned via Fisher discriminant analysis over 353K index entries
and 9K rasterized scan crops. The old flat-cosine approach where the 32-bin
profile contributed ~88% of the score has been replaced.

---

## 6. Character Extraction from Scan

### Word selection

Words are sorted by character count descending. The algorithm takes characters from the longest words first (≥3 characters), collecting up to 3 samples per character and stopping when every observed character has ≥2 samples.

**Rationale — correct.** Longer words have more characters to extract per segmentation attempt, and the per-character width estimates are more reliable (bbox noise is amortized across more characters). Short words (1–2 chars) are excluded because segmentation is impossible.

### Character segmentation

See `SEGMENTATION.md` for the full algorithm description. Summary:

1. **VP Split:** find contiguous runs of zero-ink columns. Each run is
   a definitive character boundary — split at its midpoint. Both sides must
   have `min_ink_for_symbol` ink (height-scaled: `(0.07 × h)² × 255`) or
   the split is rejected.

2. **Greedy Seam Carving:** for remaining splits, dual-DP seam carving finds
   the cheapest vertical paths through each segment. Energy is ink darkness
   (0 for white, 255 for black) with an entry penalty when the path moves
   into a darker pixel. All candidates go onto a min-heap; the globally
   cheapest is accepted, the segment is split, children get diagonal masking
   from the accepted seam, and new candidates are computed. Repeat until
   enough splits or the heap is exhausted.

3. **Fallback:** if neither pass produces enough splits, fall back to
   uniform boundaries.

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

The CI computes per-character distances via kd-tree nearest-neighbor search
(not brute-force), then aggregates using a weighted geometric mean of
log-distances. Characters are weighted by `char_weight()` — highly
discriminative characters (ligatures at 2.0, structural letters at 1.5)
contribute more than simple/narrow ones (0.5).

Characters that appear multiple times (e.g., 'e' extracted from three different
words) each contribute independently to the score.

**Missing character handling:** If a font doesn't contain a query character,
that character gets a score contribution of `log(1.0) = 0` — neutral. It
doesn't help the font, but it doesn't penalize it either. This padding
mechanism handles the common case where a font lacks ligature glyphs or
obscure punctuation without unfairly punishing it.

**Nearest-neighbor search:** Per-character lookup uses a kd-tree built over
the 99-dimensional feature space for O(log N) retrieval, replacing the original
O(N) brute-force scan.

---

## 8. Serialization

Binary format: u32 character count, then per-character: char as u32, font count, then per-font: name length + bytes + 99 floats (396 bytes).

**Estimated size:** ~5,000 fonts × 106 chars × (4 + 30 + 396) bytes ≈ **228 MB** uncompressed. With typical font name lengths averaging 25–30 bytes.

**Assessment:** Functional but wasteful. The font name is repeated ~101 times per font (once per indexed character). A more efficient format would use a font name string table with integer indices, reducing size to ~74 MB. Adding zstd or lz4 compression would likely bring this to ~15–25 MB. For a cached, build-once file this is not critical, but 91 MB is large enough to be annoying.

**Index versioning:** The index file has a version header (`INDEX_VERSION = 8`).
If the feature vector or character set changes, the version is bumped, causing
stale index files to be rejected and rebuilt automatically.

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

| Issue | Severity | Status |
|-------|----------|--------|
| ~~Feature weighting imbalance~~ | ~~High~~ | **Fixed** — per-group L2 normalization + weights |
| ~~O(n²) lookup construction~~ | ~~High~~ | **Fixed** — kd-tree nearest-neighbor search |
| ~~No index versioning~~ | ~~Medium~~ | **Fixed** — INDEX_VERSION = 8 with auto-rebuild |
| **Font name repetition** — name stored 106× per font in binary | Medium | String table with integer indices would reduce size |
| ~~Missing ligatures~~ | ~~Low~~ | **Fixed** — ff, fi, fl, ffi, ffl indexed with weight 2.0 |
| ~~Valley segmentation fragility~~ | ~~Low~~ | **Fixed** — VP + seam carving DP with diagonal masking |
| **Narrow character profiles** — 'i', 'l', '1' resampled from <8 columns to 32 bins | Low | Mitigated by char_weight(0.5) for narrow chars |

### Bottom line

The per-character index is a **sound pre-filter architecture** with a **correct
rendering/normalization pipeline** and a **rich 99-dimensional feature set**
with proper per-group weighting. The kd-tree provides efficient O(log N)
nearest-neighbor lookup, and INDEX_VERSION guards against stale index files.

The feature set reliably separates font *classes* (serif vs. sans, thin vs.
bold, condensed vs. regular) and does a reasonable job within classes. It
struggles to distinguish very similar fonts within the same family (Libre
Baskerville vs. Libre Caslon vs. Noto Serif) — but that's acceptable because
the word-level SSIM reranker handles fine discrimination. The CI just needs to
not accidentally eliminate the correct font from its candidate list.

Current specimen accuracy: **85/94 lines (90.4%)** with dual-path ligature
support on the 6-page, 30-font timeline specimen.
