# Text Matching Approach

How unprint identifies fonts and places vector text to match the original scan.

## Pipeline Overview

```
Scanned PDF
  → Rasterize page (pdftoppm at configured DPI)
  → OCR (Tesseract HOCR) → words with bounding boxes
  → Expand word bboxes to ink bounds
  → Font-metric word splitting (split_wide_whitespace_words)
  → Parallel font matching per line:
      → SSIM fast path (dominant font candidate)
      → Full pipeline on miss: char segmentation → CI search → font match
  → Pass 1.5: Paragraph-level font grouping (dominant body font detection)
  → SSIM verification (MIN_VERIFY_SSIM = 0.3)
  → PDF output → subsetted font + positioned words
```

## 1. Character Index (CI) — Font Identification

**Files:** `src/classifier.rs`, `src/font_match.rs`, `src/features.rs`

Each installed font (~5000+) is rendered at a reference size for ~106
printable characters. A 63-dimensional feature vector is extracted per glyph
(see [`FEATURES.md`](../FEATURES.md)). The LDA classifier weights are
cached to `~/.cache/unprint/lda-weights.bin` and retrained automatically
when feature computation changes.

At query time, each OCR'd character's crop is feature-extracted and classified
via the LDA classifier, which projects features into a discriminant space and
identifies the best-matching font.

**Role:** The CI is the sole font identification stage. There is no separate
word-level SSIM reranking step — the CI #1 candidate wins directly. The SSIM
verification in Pass 2 is a *gate* (reject bad matches), not a *selector*
(choose between candidates).

**Score aggregation (generative log-likelihood):** For each observation (character crop),
the classifier produces a per-font logit `-d²/(2σ²)` and geometry adds
`GEO_WEIGHT * (h_ll+v_ll)` (see §1.1). The weighted log-prob is
`lp_i = logit_i + GEO_WEIGHT*geo_ll_i`, weight `w_i = crop.weight * ood_weight_i`.

When `USE_SUM_AGG=true` (default since 2026-07-26), the font score is the
proper generative log-likelihood under independence:

```
score(font) = Σᵢ wᵢ · lp_i
```

Highest score wins. This is order-preserving under subsetting (IIA): if A > B
on the full set, A > B remains true after pruning any third font C, because
`best_i` is not used. The previous squared-gap mode
`score = −Σ w·(best_i − lp_i)²` used a data-dependent `best_i = max_j lp_j[i]`;
pruning changes `best_i`, so a font could flip from winner to loser because it
was close to an artificially low `best_pruned`. That broke midpoint pruning
soundness — hence the switch.

`best_lps[i] = max_j lp_j[i]` is still computed for tie-breaking and for the
`MIN_KEEP` stabilization heuristic, but no longer part of the score when
`USE_SUM_AGG` is true.

**OOD observation weighting:** Each observation's weight is scaled by
`min(1, med_nn / min_d)`, where `min_d` is the distance to the nearest
centroid and `med_nn` is the median nearest-neighbor distance among centroids.
Observations where the crop is far from all known glyphs (bad segmentation,
unseen character) are downweighted geometrically.

**Dual-path ligature support:** Common Latin ligatures (ff, fi, fl, ffi, ffl)
are handled via dual-path segmentation: plain OCR characters vs.
ligature-collapsed characters, with the higher-scoring path winning.  Path
comparison uses OOD-weighted scores only (no position weights) so garbage
observations are downweighted without position bias affecting the selection.

**Geometry scoring:** `per_char_geo_for_font()` computes per-character
midpoint geometry `h_ll + v_ll` (horizontal/vertical midpoint log-likelihoods)
from `word_segs` / `wib` (word-image boxes). This is added to the classifier
logit with `GEO_WEIGHT=1.0`:

```
lp_i = logit_i + GEO_WEIGHT * geo_ll_i
```

If a font has no geometry data for a line, it is kept (cannot be pruned for
geometry). Empty `geo_per_font` maps keep the candidate safe.

**Midpoint pruning (pre-filter, ~85% reduction):** Before scoring, fonts are
pruned by worst-case geometry:

```
min_ll(font) = min_{chars on line} (h_ll+v_ll)
threshold = MIDPOINT_PRUNE_BASE * thoroughness.max(0.1)
  BASE = -12.0  (worst correct-font letter on BAP: SourceSerif4-400 'T' p5:23 = -10.15)
  thr=1.0 → -12, thr=2.0 → -24 (looser), thr=0.5 → -6 (tighter)

prune if min_ll < threshold
keep if font_key in ensure_font_keys or geo_map empty
```

Stabilization:
- `MIN_KEEP=10` — if kept <10, re-add the best pruned fonts by highest `min_ll`
- If kept becomes empty, keep the single font with highest `max geo_ll`
- Logs `midpoint prune: pruned X/Y fonts at threshold ...` when `--audit` is on

This is sound with `USE_SUM_AGG=true` because `Σ w·lp` does not depend on
per-position `best`. With squared-gap mode it was unsound — `best_pruned < best_full`
made all remaining fonts look better, but the wrong fonts benefited most from a
lowered `best`, flipping p5:84 `IBMPlexSans-400` (-4.458) vs `Devanagari-400` (-4.442).

BAP (font-timeline, 494 GT lines):
- no prune: 362 primary hits 73.3% 445s, major 77 minor 98 hit 264 sim 55
- prune thr1 -10 gap²: 334 primary 67.6% 105s (4.2×), prune 85.6% 126621/147985
- prune thr1 -12 log-p (current): 348 primary 70.4% (291 hit+57 minor), major 43 minor 57 sim 103, ZNCC 0.8987, 52s audit-only; OCR-correct 320/396=80.8% vs 324/404=80.2% before

## 2. SSIM Fast Path — Dominant Font Acceleration

**File:** `src/main.rs` (parallel font matching section)

Most documents use one dominant font for body text. The SSIM fast path
exploits this by trying a **candidate font** on every line via SSIM rendering
comparison before running the expensive CI pipeline:

1. **Candidate selection:** The dominant font from the previous page (tallied
   by font-key frequency after each page's font matching completes). For the
   first page, no candidate exists — all lines go through full CI.

2. **Parallel execution:** All lines run in `rayon::par_iter`. The candidate
   font and its loaded font data are shared immutably across threads. Each
   thread's first action is the fast-path check.

3. **SSIM gate:** `verify_text_region()` renders the line in the candidate
   font and computes SSIM against the scan. If SSIM ≥ 0.90
   (`FAST_PATH_MIN_SSIM`), the line accepts the candidate — skipping
   segmentation and CI entirely. The `bail_below` parameter is set to
   `FAST_PATH_MIN_SSIM`, enabling early bail-out in `ssim_windowed()`:
   after processing ≥8 windows per row, if the running average is below
   the threshold, it returns immediately without evaluating the rest of
   the image.

4. **Fallthrough:** Lines that fail the fast path fall through to the full
   pipeline (segmentation → CI search → font match) within the same
   parallel thread.

5. **Candidate update:** After each page, the dominant font is re-tallied
   from all matched lines (fast-path hits + CI results). This candidate
   propagates to the next page.

**Why 0.90 and not lower?** The fast-path threshold is intentionally much
higher than the final verification gate (`MIN_VERIFY_SSIM = 0.3`). At lower
thresholds (tested: 0.55, 0.65, 0.75, 0.85), wrong-font matches slip
through, causing accuracy regressions. At 0.90, accuracy is identical to the
no-fast-path baseline (454/480 = 94.6% on the 30-font specimen).

**Performance:** On a typical single-font document, 90%+ of lines hit the
fast path, avoiding CI entirely. On the adversarial 30-font specimen (every
section a different font), most lines miss — but the overhead is just one
wasted SSIM call per line, which is cheap compared to full CI.

## 3. Full CI Pipeline (per line, on fast-path miss)

For each line that misses the fast path:

1. **Character extraction:** Words are segmented into individual character
   crops via VP split + seam carving DP (see [`SEGMENTATION.md`](../SEGMENTATION.md)).
   Crops are normalized to NORM_H (48px) tall.

2. **Feature computation:** Each crop is converted to a 99-dimensional
   feature vector (see [`FEATURES.md`](../FEATURES.md)).

3. **CI search:** Per-character nearest-neighbor search against the
   pre-built index. Brute-force linear scan (~5000 fonts per character).

4. **Score aggregation:** Weighted sum of log-probs `Σ w_i·lp_i` (generative).
   `best_lps` still computed per position for tie-break stability, but not used
   in score when `USE_SUM_AGG=true`. OOD-weighted variant also computed for
   path comparison.

5. **OCR correction gate:** If the best match for a character is
   catastrophically bad (d² > 0.5), all indexed characters are scanned
   to check if Tesseract mis-identified the character. For moderate
   distances (0.1–0.5), only confusable pairs are checked (O↔0, l↔1, etc.).

6. **Result:** CI #1 wins directly. No word-level SSIM reranking.

## 4. Pass 1.5: Paragraph-Level Font Grouping

**File:** `src/main.rs`

After all lines on a page are matched, the pipeline identifies the **dominant
body font** — the most common font among matched lines at the most common
font size (±1pt tolerance). This detects the body font for potential
paragraph-level consistency enforcement.

Currently, Pass 1.5 logs the dominant font but does not override individual
line matches. The infrastructure exists for majority-vote font replacement
but is not active.

## 5. Font-Metric Word Splitting

**File:** `src/ocr.rs` → `split_wide_whitespace_words()`

Before font matching, Tesseract's word boundaries are refined using font
metrics to detect over-merged words (Tesseract sometimes joins distinct words).

**Algorithm:**

1. **Font identification (once per line):** The longest word in the line is
   segmented and CI-searched to identify the font. This result is shared
   across all words in the line.

2. **Per-character segmentation:** Each word is segmented into character
   crops via VP + seam carving.

3. **Ink gap measurement:** For each adjacent character pair, the zero-ink
   gap between the rightmost ink of character i and the leftmost ink of
   character i+1 is measured from the scan image.

4. **Font-metric expected gap:** Using `ab_glyph`'s `outline_glyph()` and
   `px_bounds()`, the pipeline derives a per-character rendering scale:
   - Measure the observed ink width of character i
   - Render the same character in the matched font at a reference scale (100px)
   - Compute `font_ink_width()` via `outlined.px_bounds()` extents
   - Derive the actual scale: `s = observed_ink / font_ink * reference_scale`
   - Compute expected inter-glyph gap via `font_pair_ink_gap()`:
     `gap = advance_a - ink_right_a + ink_left_b` at the derived scale

5. **Split threshold:** `round(expected_gap) + 5` pixels. The +5 margin
   absorbs anti-aliasing blur and scan noise. If the measured ink gap
   exceeds this threshold, the word is split at that point.

6. **Fallback:** If no font match is available, a fallback gap threshold of
   18% of line height (minimum 4px) is used.

## 6. SSIM Verification (Pass 2)

**File:** `src/verify.rs`

After font matching, each line that will be vectorized undergoes SSIM
verification:

1. The matched font renders the line's text via FreeType (thread-local
   library instance) with rustybuzz shaping.

2. Multi-scale rendering: the text is rendered at 2× and 4× resolution,
   each downscaled and compared via windowed SSIM.

3. Vertical shift search: ±12px offsets are tested, with early termination
   at SSIM ≥ 0.92 and center-outward search order.

4. `bail_below` parameter: `verify_text_region()` accepts an optional
   `bail_below: Option<f32>`. When set, this is passed through to
   `ssim_windowed()`, which bails early if the running SSIM average drops
   below the threshold after ≥8 windows per row. The fast-path SSIM check
   sets this to `FAST_PATH_MIN_SSIM` (0.90) so non-matching fonts are
   rejected cheaply. The normal verification path passes `None` (no
   early bail).

5. If SSIM < 0.3 (`MIN_VERIFY_SSIM`), the line reverts to raster. This
   threshold is intentionally loose — it catches only catastrophic
   mismatches (wrong font entirely), not marginal ones.

**Key distinction:** The verification gate (0.3) is much lower than the
fast-path gate (0.90). The fast path must be confident because it's
*accepting* a font without CI evidence. The verify gate only needs to reject
disasters because CI already chose the best available font.

## 7. Decision Matrix + Output

For each line, the pipeline checks:
- OCR confidence ≥ `--min-ocr-confidence` (default 0)
- Font match score ≥ `--min-font-confidence` (default 0.10)
- SSIM verification ≥ 0.3

Lines passing all thresholds are vectorized; all others keep the original
raster.

## 8. Font Size Determination

Font size is determined from the OCR bounding box **height**:

```
em_px = bbox_height * reference_em / font_ink_height
```

Where `font_ink_height = ascent - descent` at the reference em size.

Height-matching is more stable than width-matching because OCR line heights
are consistent and don't depend on word-level bbox precision.

## 9. PDF Text Placement

**File:** `src/pdf_out.rs`

Each word is placed at its OCR x-position with a uniform font size for the
line:

- One `em_px` computed from line bbox height
- One baseline computed via `ink_centered_baseline_pt()`
- Each word gets its own `BT/Tf/Td/Tj/ET` block at its OCR x-coordinate
- No character spacing adjustments (Tc), no horizontal scaling
- Font is subsetted to only the glyphs used (via `subsetter` crate)

## 10. Geometry Vectorization

**File:** `src/geometry.rs`

Horizontal/vertical lines, solid-color fills, and rectangles are detected
and replaced with native PDF paths.

## 11. Raster Handling

**File:** `src/main.rs`, `src/color.rs`

After vectorizing text, corresponding regions are erased from the raster.
Remaining raster is split into fragments via cell-based content detection
(100px cells classified as interesting/blank, flood-fill grouped). Fully
blank pages produce zero raster fragments.

## 12. Audit Mode — Per-Character Distances

**File:** `src/main.rs`, `src/font_pipeline.rs`

When `--audit` is enabled, the pipeline computes per-character feature-space
distances between each line's character crops and both:
- The **chosen font** (CI winner or fast-path hit)
- All **fontmap fonts** (ground-truth fonts injected via `--include-fontmap`)

To avoid redundant feature computation, crop features are precomputed once
via `precompute_crop_features()`, which extracts and weights the 99-dim
feature vector for each character crop. The precomputed vectors are then
passed to `per_char_distances_precomputed()` for each font, which looks up
the font's reference vector and computes squared Euclidean distance — no
`compute_features()` call per font.

This is critical for performance: the fontmap typically contains ~74 fonts,
and a page can have ~480 character crops. Without precomputation, the audit
would call `compute_features()` once per crop per font (~35,000 times per
page). With precomputation, it's called once per crop (~480 times), and the
per-font lookup is a simple vector distance.

The original `per_char_distances()` function still exists for callers that
have images rather than precomputed features — it internally calls
`precompute_crop_features()` and delegates to the precomputed version.

## Design Principles

1. **CI #1 wins directly.** No word-level SSIM reranking — the character
   index is the sole font selector. SSIM serves only as a verification gate.
2. **Parallel everything.** All lines run in `par_iter` with the SSIM fast
   path as a cheap speculative check before the expensive CI pipeline.
3. **Fix inputs, not algorithms.** Every SSIM failure traces to bad inputs
   (illustration contamination, garbage OCR, wrong crop geometry).
4. **Natural font metrics.** Don't distort spacing to match OCR bboxes.
5. **Smaller outputs.** Font subsetting + blank raster elimination.

## What's NOT Done

- **No word-level SSIM reranking.** The `word_match.rs` module exists but is
  disabled. CI ranking is used directly.
- **No ML classifier.** Nearest-neighbor on 99-dim feature vectors is the
  approach. A random forest module exists but is not integrated.
- **No paragraph font override.** Pass 1.5 detects the dominant font but
  does not override individual line matches.

## Known Weaknesses

- **Baskerville vs EB Garamond**: Body text in Baskerville sections sometimes
  matches EB Garamond. These are historically related faces with similar
  proportions.
- **Short fragments**: Single-word lines match poorly — too little signal.
- **Small/watermark text**: Attribution lines at low point sizes get unreliable
  OCR.
