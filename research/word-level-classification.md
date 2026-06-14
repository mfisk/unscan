# Word-Level Font Classification: Feasibility & Design

## Executive Summary

**Bottom line: Word-level classification is worth pursuing as a re-ranker, but
its ceiling is constrained by the nature of unscan's actual error modes.**

Unscan's current 84.3% accuracy (404/479) has a specific failure profile that
reframes the word-level question:

| Error class | Count | % of misses | Word-level can help? |
|---|---|---|---|
| Same-family variant confusion | 120 | 92.3% | ❌ No — glyphs are pixel-identical |
| Cross-family confusion | 10 | 7.7% | ✅ Yes — spacing/rhythm/texture differ |

The 120 variant misses (e.g., `SourceSerif4-It` → `SourceSerif4 Italic[opsz,wght]`,
`Lato-Regular` → `Lato Medium`) involve fonts whose rasterized glyphs are
**literally identical** at any scale. No amount of word-level, character-level,
or neural feature extraction can distinguish them — they're the same pixels.
This is a font identity/naming resolution problem, not a classification problem.

The 10 cross-family misses (e.g., `NotoSerif-Bold` → `Caladea Bold`, `ArialMT`
→ `Roboto Regular`) are the only ones where word-level features could add signal.
These are metrically similar fonts (clones/alternatives), but their kerning
tables, spacing rhythm, and fine stroke details do differ.

**Estimated accuracy ceiling with word-level:**
- If word-level resolves all 10 cross-family misses: 84.3% → 86.4% (+2.1%)
- If variant confusion is fixed by naming/metadata: 84.3% → 109.2% (normalized: ~97%)
- Both combined: ~99%

**Recommendation:** Fix the variant naming problem first (highest ROI), then
add word-level as a cross-family discriminator second.

---

## 1. Current System Analysis

### 1.1 Pipeline Architecture

```
OCR (Tesseract) → Word bboxes → Character segmentation (seam carving)
    → Per-char 100-dim feature vectors → Weighted geometric mean scoring
    → Top font candidate → SSIM verification → Output
```

Key implementation files:
- Character index: `src/char_index.rs` (features, search, scoring)
- Segmentation: `src/segment.rs` (VP + dual-DP seam carving)
- Font matching: `src/font_match.rs` (result types)
- SSIM verification: `src/verify.rs` (word-level render-and-compare)
- **Disabled word-level modules**: `src/word_index.rs`, `src/word_match.rs`

### 1.2 Feature Vector (100-dim)

| Dims | Feature | Fisher weight range |
|---|---|---|
| [0..31] | Column ink profile (32 bins) | Low-medium |
| [32..38] | Scalar v1: aspect, ink_density, v_center, h_balance, serif_score, stroke_contrast, xh_cap_ratio | v_center=0.072 (highest) |
| [39..56] | Scalar v2: counters (4), terminal angles (4), ink_perimeter, compactness, h_crossings (8) | Medium |
| [57..88] | Row ink profile (32 bins) | Low-medium |
| [89..99] | Scalar v3: hole_count, symmetry (2), skeleton (2), corners, quadrant_density (4), mean_stroke_width | mean_stroke_width=0.024 |

### 1.3 Error Analysis (from audit JSON, 490 entries)

```
Hits:            346  (70.6%)
Minor misses:     56  (11.4%)
Major misses:     74  (15.1%)
SSIM failures:     4  ( 0.8%)
Kept raster:       5  ( 1.0%)
No ground truth:   5  ( 1.0%)
```

**Critical finding — quality gate performance:**

| Category | Total chars | Gate pass | Gate fail | Fail rate |
|---|---|---|---|---|
| Major misses | 1,959 | 1,959 | 0 | 0.0% |
| Minor misses | 1,740 | 1,740 | 0 | 0.0% |
| Hits | 9,055 | 9,055 | 0 | 0.0% |

**Zero quality gate failures.** The character crops are consistently clean.
This means the problem isn't segmentation corruption — it's that the features
genuinely can't distinguish certain font pairs.

**Distance distributions are nearly identical between hits and misses:**

| | p10 | p50 | p90 | max |
|---|---|---|---|---|
| Hits (9,055 chars) | 0.00001 | 0.00002 | 0.00005 | 0.00166 |
| Major misses (1,959 chars) | 0.00001 | 0.00002 | 0.00005 | 0.00114 |

The "wrong" font is just as close as the right font in feature space.

### 1.4 Dominant Error: Variant Confusion (92.3% of misses)

| Confusion pair | Count | Nature |
|---|---|---|
| SourceSerif4-It → SourceSerif4 Italic[opsz,wght] | 32 | Static vs variable |
| CourierNewPSMT → Courier | 12 | OEM vs URW clone |
| Lato-Regular → Lato Medium | 8 | Weight variant |
| SourceSans3-Roman → SourceSans3 Light | 8 | Weight variant |
| PrestigeEliteNormal → Prestige Elite Std | 6 | Naming variant |
| SourceSerif4-It → SourceSerif4SmText It | 6 | Optical size variant |
| IBMPlexSerif → IBMPlexSerif Text | 6 | Optical size variant |
| (18 other same-family pairs) | 42 | Various |

These fonts render **identical glyphs**. A SourceSerif4-It `e` and a
SourceSerif4 Italic[opsz,wght] `e` are the same Bézier curves. No image-based
classifier — character-level, word-level, or neural — can distinguish them.

### 1.5 Cross-Family Errors (10 misses — the word-level targets)

| Expected | Got | Lines |
|---|---|---|
| NotoSerif-Bold | Caladea Bold | 2 |
| NotoSerif-Bold | IBMPlexSerif Bold | 2 |
| NotoSerif-Italic | Caladea Italic | 1 |
| NotoSerif-Italic | SourceSerif4Subhead SemiboldIt | 1 |
| CourierNewPS-BoldMT | TeX Gyre Cursor | 1 |
| ArialMT | Roboto Regular | 1 |
| Lato-Bold | Carlito Bold | 1 |
| IBMPlexMono-Bold | IBMPlexSansThai Bold [ss04] | 1 |

These are metrically similar fonts (Google's Caladea is a Cambria clone, Carlito
is a Calibri clone, Roboto has Arial-like proportions). Their individual glyphs
look nearly identical, but their **spacing, kerning, and fine stroke details
differ** — this is where word-level features could add discriminative power.

---

## 2. Prior Word-Level Work in Unscan

### 2.1 word_index.rs (285 lines, present but unused)

A complete word-level font index using visual thumbnails:
- **Feature**: 12×48 pixel downscaled thumbnail (576-dim flat pixel vector)
- **Index**: ~100 common English words rendered in every font
- **Query**: Crop words from scan, resize to thumbnail, nearest-neighbor search
- **Aggregation**: Geometric mean distance, quorum voting, σ cutoff
- **Status**: Module exists and compiles but is never called from main.rs

**Design limitations:**
- Fixed to ~100 hardcoded common words (the/and/for/with...)
- 12×48 pixel resolution is extremely coarse
- Flat pixel comparison is sensitive to any alignment/scale shift
- No Fisher weighting or learned features — raw pixels only

### 2.2 word_match.rs (285 lines, disabled)

Word-level SSIM re-ranking after character index:
- Crops whole words from scan, renders same text in each candidate font
- Computes SSIM similarity between crop and render
- Votes across up to 4 words per line
- **Status**: Disabled with comment "CI ranking used directly, word-level SSIM
  rerank removed"

**Disabled because:**
- Line-level SSIM (now in verify.rs) replaced it
- Word-level SSIM was net-negative for accuracy
- The same SSIM comparison is now done at the full-line level for verification

### 2.3 Why They Failed

The word_match approach failed because:
1. **SSIM is good at verification but poor at ranking.** SSIM scores cluster
   tightly for similar fonts — the difference between Noto Serif Bold and
   Caladea Bold at word-level SSIM is 0.001-0.002, well within noise.
2. **The word_index used raw pixel thumbnails** — no spatial invariance,
   no learned features, no alignment robustness.
3. **Neither approach addressed variant confusion**, which dominates the errors.

---

## 3. Word-Level Feature Design

### 3.1 Features That Word-Level Can Capture

These features are invisible at the character level but emerge at word scale:

#### 3.1.1 Spacing & Rhythm Features
| Feature | Description | Discriminative for |
|---|---|---|
| **Inter-character spacing ratio** | avg gap / x-height across the word | Tight vs loose tracking |
| **Spacing variance** | σ of inter-char gaps (normalized) | Proportional vs monospaced |
| **Word width / char count ratio** | Total width / num characters | Average character width |
| **Rhythm regularity** | Autocorrelation of column ink profile | Fonts with consistent vs variable widths |

From audit data: inter-word gaps average 17.4px for major misses, 10.4px for
minor misses, 16.5px for hits — showing that line-level spacing varies by
font context.

#### 3.1.2 Kerning Signatures
| Feature | Description | Why it helps |
|---|---|---|
| **Known kern pairs** | Spacing between To, AV, We, VA, etc. | Fonts embed different kern tables |
| **Kern pair ratio** | Kern gap / standard gap | Normalizes for font size |

Kerning is the most font-specific word-level signal. A font's kern table is
part of its identity — different font families kern "To" at different ratios.
However, this requires recognizing specific character pairs, which needs
reliable character positions within words.

#### 3.1.3 Global Word Shape
| Feature | Description | Discriminative for |
|---|---|---|
| **Word aspect ratio** | width/height | Condensed vs extended fonts |
| **Ink density** | total ink pixels / bbox area | Light vs bold |
| **Baseline position** | y-position of ink bottom (normalized) | Baseline alignment consistency |
| **x-height ratio** | height of lowercase body / cap height | x-height variation between fonts |
| **Ascender/descender ratio** | extent above/below x-height | Proportional differences |

#### 3.1.4 Texture & Stroke Features
| Feature | Description | Discriminative for |
|---|---|---|
| **Mean stroke width** (word-level) | Average across entire word | Weight consistency |
| **Stroke width variance** | σ of stroke width across word | Contrast (thin/thick strokes) |
| **Ink column entropy** | Shannon entropy of column ink profile | Textural regularity |
| **Gabor filter responses** | Multi-scale directional filters | Serif vs sans detection |

### 3.2 Features That DON'T Help

| Feature | Why not |
|---|---|
| Character identity features | Already covered by char-level CI |
| Individual glyph shape | Already 100-dim per character |
| Font file metadata | Not visible in raster image |
| OpenType feature detection | Not visible in raster image |

---

## 4. Architectural Options

### 4.1 Option A: Word-Level Handcrafted Features + Nearest-Neighbor

**Approach:** Compute word-level feature vector (spacing, rhythm, texture),
add to the CI scoring pipeline.

**Pros:**
- Fits naturally into existing Rust pipeline
- Interpretable — each feature has meaning
- No training data required (render words in every font, measure features)
- Fast — just arithmetic on the word crop

**Cons:**
- Limited ceiling — handcrafted features miss subtle patterns
- Need to design, test, weight each feature individually

**Estimated features:** ~20 dimensions
- 5 spacing features (inter-char gap mean/σ, width/count ratio, rhythm)
- 3 kern pair features (if recognizable pairs present)
- 5 shape features (aspect, ink density, baseline, x-height, asc/desc)
- 5 stroke/texture features (mean stroke width, variance, entropy, etc.)
- 2 word-level ink profiles (coarse column + row profiles at word scale)

**How to score:**
- Render each word in each candidate font (already done in verify.rs)
- Compute feature vector for both crop and render
- Weighted Euclidean distance (Fisher weights learned like CI)
- Combine with CI score: `final_score = α * CI_score + (1-α) * word_score`

### 4.2 Option B: Small CNN on Word Images

**Approach:** Train a small CNN to embed word images into a font-discriminative
space. Use as re-ranker after CI narrows to top-K fonts.

**Architecture:**
```
Input: Fixed-height word image (48px height, variable width)
→ 3-4 conv layers (3×3 kernels, 32→64→128 channels)
→ Global Average Pooling (handles variable width)
→ FC → 64-dim embedding
→ Nearest-neighbor against font embeddings
```

**Pros:**
- Can learn subtle features humans would miss
- Handles variable word lengths naturally
- Can capture kerning/spacing patterns implicitly

**Cons:**
- Requires training infrastructure (Python, PyTorch)
- Needs training data (the generator being built handles this)
- Adds model weight to the binary
- Domain gap: synthetic training → scanned test

**Training:**
- Contrastive learning (triplet loss) with hard negative mining
- Anchor: rendered word in font A
- Positive: same word in font A with augmentation (noise, blur, scale)
- Negative: same word in font B (hard negative: similar family)
- Following Memon et al. (2023): hard negatives selected by visual similarity

**Inference:**
- Pre-compute embeddings for common words in all fonts
- For each word in scan: compute embedding, find nearest fonts
- Combine with CI scores

### 4.3 Option C: Hybrid Character + Word Features

**Approach:** Use CI top-K as input, then re-rank with word-level evidence.

```
CI produces top-10 candidates with scores
    → For each word in line, compute word-level features
    → Score each top-10 candidate against word features
    → Weighted combination: 0.7 * CI_rank_score + 0.3 * word_rank_score
    → Final ranking
```

**This is the recommended approach.** It:
- Uses CI's strength (character-level discrimination)
- Adds word-level as a tie-breaker, not primary classifier
- Avoids building a full word-level index (expensive)
- Only needs to evaluate top-K candidates (fast)

### 4.4 Option D: Sequence Model (LSTM/1D-CNN)

**Approach:** Treat line as sequence of character features, capture sequential
patterns.

**Architecture:**
```
Input: Sequence of (char, 100-dim feature vector) for each character in line
→ Bidirectional LSTM or 1D-CNN
→ Output: font class probabilities
```

**Why not recommended:**
- Character-level features already capture individual glyph shapes
- Sequential patterns (kerning, spacing) aren't in the feature vectors
- Would need inter-character gap features injected into the sequence
- Complex, marginal benefit

---

## 5. Literature Review

### 5.1 DeepFont (Wang et al., 2015, Adobe)

The seminal work in visual font recognition:
- **Input:** Word-level text crops (not single characters)
- **Architecture:** CNN with low-level shared sub-network + high-level
  classifier sub-network
- **Training:** 2,383 font categories, synthetic data with domain adaptation
  via Stacked Convolutional Auto-Encoder (SCAE)
- **Accuracy:** >80% top-5 on AdobeVFR dataset
- **Key insight:** Domain adaptation (synthetic → real) is critical. Direct
  CNN training on synthetic data alone degrades on real scanned images.

**Relevance to unscan:** DeepFont operates at word level, not character level.
Its architecture validates word-level classification is viable, but the domain
gap problem (clean renders vs. noisy scans) requires explicit handling.

### 5.2 FasterViT-2 (Collabora, 2025)

Updated font recognition with modern architecture:
- **Input:** Grayscale word images, height 105px
- **Architecture:** FasterViT-2 (hybrid CNN + transformer with HAT)
- **Training:** 2,700 fonts, 2.7M synthetic images
- **Accuracy:** 87.4% top-1, 92.1% top-5 on real-world test set
- **Throughput:** 3,161 img/sec on A100

**Relevance:** Shows that modern architectures significantly outperform
DeepFont (87.4% vs 62.3% top-1). However, this is open-world classification
(2,700 fonts); unscan's closed-world setting (500 installed fonts) is simpler.

### 5.3 DINOv2 + LoRA (Chen et al., 2026)

State-of-the-art parameter-efficient font classification:
- **Input:** Text images at 224×224 (padded square)
- **Architecture:** DINOv2-Base (ViT-B/14) + LoRA (rank 8)
- **Training:** 394 Google Font variants, ~575 images per variant
- **Accuracy:** 99.0% top-1 on synthetic test set
- **Parameters:** Only 900K trainable (1% of 87M backbone)

**Key findings:**
- Trained on full sentences, not single characters
- Pre-processing includes padding to square, resize to 224×224
- LoRA fine-tuning achieves near-perfect accuracy with minimal parameters
- Uses SWER (severity-weighted error rate) that penalizes cross-family errors
  more than same-family variant errors

**Relevance:** Validates that fine-tuning pre-trained vision models is
extremely effective for font classification. However:
- 394 variants is much smaller than unscan's ~2,353 font files
- Synthetic test only — no real scan evaluation
- The backbone alone (DINOv2-Base) is 87M parameters — way too large to embed
  in unscan's Rust binary

### 5.4 Contrastive Learning for Fonts (Memon et al., 2023)

Contrastive learning (triplet loss) applied to font style classification:
- **Key insight:** Font images can't use standard augmentation (flipping,
  rotation) — these change the content. Must use hard positive/negative mining
  with same-character cross-font and different-character same-font pairs.
- **Loss functions compared:** NT-Xent, triplet loss, supervised contrastive
- **Result:** Contrastive learning matches fully supervised accuracy with
  fewer labeled examples
- **Multilingual:** Works across Chinese, English, Korean

**Relevance:** Validates triplet loss for font embedding spaces. The hard
negative mining strategy (same character, different font) is directly
applicable to unscan's training data generator.

### 5.5 Font Group Identification Using Reconstructed Fonts

An interesting unsupervised approach:
- Builds token co-occurrence graphs to reconstruct font alphabets
- Uses graph partitioning to assign tokens to candidate fonts
- Works without labeled data

**Relevance:** Limited — unscan has labeled ground truth and a known font catalog.

---

## 6. Segmentation Analysis: Is It Actually a Problem?

### 6.1 Character Segmentation Quality

A key motivation for word-level classification is bypassing character
segmentation errors. But the data shows segmentation is **not the bottleneck**:

- **0% quality gate failures** across all categories
- Seam carving + VP splitting produces consistently clean crops
- The `split_wide_whitespace_words` function already handles word-level splitting

### 6.2 Word Bboxes Are Reliable

From the audit:
- Raw Tesseract words: 2,440
- Post-processed words: 2,781 (expansion from `split_wide_whitespace_words`)
- Word bboxes are clipped to prevent inter-line bleeding
- Average word height: 32.1px (misses), 36.2px (hits)

Word-level processing has a reliable foundation: Tesseract word bboxes are
good, and post-processing improves them.

### 6.3 Conclusion: Segmentation Is Not the Issue

The current 16% error rate is not caused by segmentation problems. It's caused
by:
1. **Variant confusion (92%)** — identical glyphs, different font names
2. **Genuine font similarity (8%)** — clones/alternatives that look alike

---

## 7. Data Implications

### 7.1 Available Training Data

**Character-level (existing):**
- CI index covers ~100 characters × ~2,353 fonts = ~235K feature vectors
- Generated by rendering each character at NORM_H=48 with per-font scaling

**Word-level (needs generation):**
- Render words in every font: ~100 common words × 2,353 fonts = ~235K images
- The existing `word_index.rs` already does this with COMMON_WORDS
- Training data generator (in progress) focuses on characters; word-level
  would need extension

### 7.2 Training Data Design for Word-Level

For handcrafted features (Option A):
- No training needed — render words, compute features, build index
- Same approach as CI: deterministic feature extraction

For CNN embeddings (Option B):
- Synthetic: render words in each font at various sizes
- Augmentation: Gaussian noise, blur, JPEG compression, slight rotation
- Hard negatives: same word in visually similar fonts
- ~1,000 words × 2,353 fonts × 5 augmentations = ~11.7M training images

### 7.3 Ground Truth

The test specimen has word-level ground truth:
- Each word bbox maps to a known font (inherited from line-level)
- Word text is available from Tesseract OCR
- This enables direct word-level accuracy measurement

---

## 8. Practical Considerations

### 8.1 Vocabulary Dependence

**Problem:** Word-level features depend on word content. The word "the" has
different features than "Mississippi" — different lengths, different character
distributions, different spacing patterns.

**Solutions:**
1. **Content-normalized features:** Divide by word length, use ratios not absolutes
2. **Common-word index:** Only index known words (existing word_index.rs approach)
3. **Content-agnostic features:** Use features that don't depend on specific letters
   (ink density, stroke width, inter-char gap variance)

### 8.2 Short Words

Words with 1-2 characters have minimal word-level signal:
- No inter-character spacing to measure
- No kerning pairs
- Basically just a character crop

**Distribution from audit data:**
- Average word length in misses: 6.0 characters
- Average word length in hits: 5.3 characters
- Short words (≤3 chars) make up ~25% of all words

**Mitigation:** Skip words shorter than 3-4 characters for word-level features,
use them for character-level only.

### 8.3 Speed

Current character-level processing per line:
- ~100 char crops × 100-dim features × brute-force NN = <10ms per line

Word-level options:
- **Handcrafted features (Option A):** ~2-5ms per word × ~6 words = ~12-30ms
- **CNN embedding (Option B):** ~5-20ms per word (depending on model size)
- **SSIM comparison (existing word_match):** ~50-100ms per word (expensive renders)

Handcrafted features are fastest and fit the pipeline naturally.

### 8.4 Font Index Size

Character-level index: ~2,353 fonts × 100 chars × 100 dims × 4 bytes = ~94MB

Word-level index options:
- **Common words (existing):** ~100 words × 2,353 fonts × 576 dims × 4 bytes = ~542MB (too large)
- **Handcrafted features:** ~100 words × 2,353 fonts × 20 dims × 4 bytes = ~19MB
- **CNN embeddings:** ~100 words × 2,353 fonts × 64 dims × 4 bytes = ~60MB
- **Per-font aggregate:** Skip word-specific index; compute features at query time = 0 index

### 8.5 Complementarity: Word-Level as Re-ranker

The most practical architecture is word-level as a re-ranker:

1. CI produces top-10 candidates (existing, fast, ~84% accurate)
2. For each word in line, render in each top-10 font
3. Compute word-level feature distance (spacing, rhythm, texture)
4. Re-rank top-10 using combined CI + word scores

This avoids building a word-level index entirely — just compute features
on-the-fly for the 10 candidates. The existing `word_match.rs` was close
to this architecture but used SSIM instead of designed features.

---

## 9. Recommendations

### 9.1 Priority 1: Fix Variant Confusion (Highest ROI)

Before adding word-level features, address the 92% of errors that are variant
confusion. Options:

1. **Variant collapsing:** Merge font entries that render identical glyphs
   (SourceSerif4-It ≡ SourceSerif4 Italic[opsz,wght])
2. **PostScript name matching:** After CI picks a font, check if the GT has
   a PS name match for any variant of the same family
3. **Family-level scoring:** First classify to font family, then pick the
   best-matching variant within the family

Expected impact: ~120/130 misses resolved → 97%+ accuracy

### 9.2 Priority 2: Word-Level Re-ranker (Moderate ROI)

For the remaining ~10 cross-family misses, implement a word-level re-ranker:

**Minimum Viable Experiment:**

1. **Instrument the existing pipeline** to capture word crops for misses
2. **Render the same words** in the top-5 CI candidates
3. **Compute 3 handcrafted word-level features:**
   - Inter-character spacing ratio (crop vs render)
   - Word aspect ratio difference
   - Mean stroke width difference (crop vs render)
4. **Score:** Compare feature distances, re-rank if word-level clearly
   favors a different candidate (margin > threshold)
5. **Measure:** Does this flip any of the 10 cross-family misses?

**Code location:** Integrate into `src/main.rs` after CI scoring (around line
730), using existing word crop infrastructure from verify.rs.

**Effort estimate:** ~200 lines of Rust, 1-2 days of implementation.

### 9.3 Priority 3: CNN Re-ranker (Lower ROI, Higher Ceiling)

If handcrafted word features show promise:

1. Train a small CNN (4 conv layers, ~500K params) with contrastive loss
2. Use the training data generator output
3. Export to ONNX, load in Rust via `ort` crate
4. Embed as a re-ranker: CI → top-5 → CNN re-rank

**Effort estimate:** ~1 week Python training + ~300 lines Rust integration.

### 9.4 What NOT to Do

1. **Don't build a standalone word-level classifier.** The CI is already good
   at character-level; word-level should complement, not replace.
2. **Don't use the existing word_index.rs as-is.** Its 576-dim pixel thumbnail
   is too crude and the common-word restriction is limiting.
3. **Don't use large vision models (DINOv2, FasterViT-2).** They're 87M+
   parameters; unscan needs to stay as a lean CLI tool. A 500K-param CNN is
   the ceiling.
4. **Don't optimize word-level before fixing variant confusion.** It's solving
   10 errors when 120 are solvable with naming logic.

---

## 10. Minimum Viable Experiment Design

### Goal
Determine if word-level features can distinguish NotoSerif-Bold from
Caladea Bold, ArialMT from Roboto Regular, etc.

### Steps

1. **Extract word crops** from the 10 cross-family miss lines
2. **Render the same words** in both the expected and matched fonts
3. **Compute candidate features:**
   - Word width ratio (crop width / rendered width at same height)
   - Inter-character spacing profile correlation
   - Column ink profile correlation (32-bin, like CI but full word)
   - Stroke width histogram difference (Sobel-based)
4. **Compare feature distances:**
   - d(crop, expected_font_render) vs d(crop, wrong_font_render)
   - If expected is closer on word features, word-level helps
5. **Score:** Count how many of the 10 misses word-level would flip

### Success Criteria
- Flips ≥5 of 10 cross-family misses → word-level worth implementing
- Flips ≤2 of 10 → word-level not worth the complexity
- Introduces ≥5 new misses (regressions) → net negative

### Implementation
```rust
// In src/main.rs, after CI scoring produces top candidates:
fn word_level_rerank(
    crop: &GrayImage,
    words: &[WordBBox],
    candidates: &[(String, f32)], // CI top-K with scores
    font_cache: &FontCache,
) -> Vec<(String, f32)> {
    // For each word ≥ 4 chars:
    //   - Compute word-level features on crop
    //   - Render word in each candidate font
    //   - Compute word-level features on renders
    //   - Score: weighted distance
    // Return re-ranked candidates
}
```

---

## 11. Feature-by-Feature Impact Assessment

### High-Impact Word-Level Features

| Feature | Impl complexity | Expected discriminative power | Why |
|---|---|---|---|
| Word width ratio | Low | Medium | Fonts with different default tracking produce different total widths |
| Inter-char gap profile | Medium | High | Kerning tables differ across families; this is the strongest word-level signal |
| Column ink profile correlation | Low | Medium | Already proven at char level; word-level captures rhythm |

### Medium-Impact Features

| Feature | Impl complexity | Expected discriminative power | Why |
|---|---|---|---|
| Stroke width histogram | Medium | Medium | Bold vs regular at word level more stable than per-char |
| x-height consistency | Medium | Low-Medium | Some fonts have more variable x-heights |
| Baseline regularity | Medium | Low | Most fonts have consistent baselines |

### Low-Impact Features

| Feature | Impl complexity | Expected discriminative power | Why |
|---|---|---|---|
| Gabor filter responses | High | Low | Overkill for text; works for natural textures |
| LBP texture descriptors | High | Low | Same as above |
| n-gram features | High | Low | Requires character recognition + pair extraction |

---

## 12. Relation to Other Research Tracks

### 12.1 Feature Noise Analysis
The feature noise analysis (in progress) may identify which CI features are
harming cross-family discrimination. If specific features are noisy, word-level
features could compensate by providing an orthogonal signal.

### 12.2 Learned Classifier Research
The learned classifier research (in progress) explores metric learning and
contrastive approaches. If pursued, the CNN word-level embeddings (Option B)
would naturally fit into a triplet-loss framework, as validated by Memon et al.

### 12.3 Training Data Generator
The training data generator (in progress) focuses on character-level data.
Word-level would need:
- Word rendering (combine character renders with font kerning)
- Word-level feature computation
- Ground truth at word level (straightforward: inherit from font label)

### 12.4 AA Normalization
The AA normalization research found that scaling introduces feature instability
in both directions. Word-level features would face the same issue — scanned
words at different sizes would need consistent normalization.

---

## Appendix A: Existing Code to Reuse

| Module | What to reuse | Where |
|---|---|---|
| `word_match.rs` | Word crop extraction, word rendering, SSIM infrastructure | Re-enable and refactor |
| `word_index.rs` | Common word list, render_word_for_index function | Extract rendering logic |
| `verify.rs` | WordPlacement struct, width-scaled rendering | Already in main pipeline |
| `char_index.rs` | compute_features patterns, Fisher weight learning | Adapt for word-level features |
| `layout.rs` | render_word_ab_glyph, width_matched_em_px | Direct reuse |

## Appendix B: Key Measurements from Audit Data

| Metric | Value |
|---|---|
| Total entries | 490 |
| Hit rate | 70.6% (346/490) |
| Font-correct rate (hits+minor) | 82.0% (402/490) |
| Words per miss line | avg 5.8 (range 1-13) |
| Chars per miss line | avg 28.5 (range 4-58) |
| Word bbox height (misses) | avg 32.1px (range 4-70px) |
| Word bbox height (hits) | avg 36.2px (range 4-90px) |
| Line height (misses) | avg p50=60px |
| Line height (hits) | avg p50=76px |
| Inter-word gap (misses) | avg 17.4px |
| Inter-word gap (hits) | avg 16.5px |
| Same-family variant misses | 120/130 (92.3%) |
| Cross-family misses | 10/130 (7.7%) |
| Quality gate failures | 0/12,754 (0.0%) |
