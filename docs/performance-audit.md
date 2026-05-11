# Unscan Performance Audit

**Date:** 2025-07-13
**Codebase:** ~8,866 LOC across 18 `.rs` files
**Baseline:** ~3 min for 89-line specimen PDF, ~1-2 min for 33-line Berkeley PDF

---

## Executive Summary

The dominant bottleneck is the **coarse scoring loop** in `font_match.rs`, which for each of ~50 candidate fonts per line: parses the font binary, renders text to images, binarizes, tight-crops, resizes, computes morphological operations, and calculates four similarity signals — all of this per-word (up to 4 words per line). The SSIM reranking loop then does a similar render+compare pass for the top 30 candidates with an additional 25-position vertical shift search.

For a 89-line document: **~50 fonts × 89 lines × ~4 words × (render + binarize + morph_open + tight_crop + resize + blur + IoU_with_shift + NCC + Hu + fill) = ~17,800 full scoring iterations**, plus **30 × 89 × 25 SSIM verification renders = ~66,750 SSIM windows**.

The `page_img.to_luma8()` call (full-page grayscale conversion) is made **3 times per line** inside `match_font()` alone, and once more in `main.rs`.

---

## HIGH IMPACT — Estimated 50-70% of total runtime

### H1. Redundant `page_img.to_luma8()` — Triple conversion per line

**Files:** `src/font_match.rs` lines 89, 174, 500; `src/main.rs` line 355

`match_font()` is called once per line. Inside it, `page_img.to_luma8()` is called:
1. Line 89: for source sample pre-processing
2. Line 174: inside the char-index pre-filter block
3. Line 500: for the SSIM reranking phase

Each call allocates a fresh `GrayImage` by iterating over every pixel of the full page image. For a 2550×3300 px page at 300 DPI, that's **8.4M pixels × 3 calls × 89 lines = 2.24 billion pixel conversions**.

Meanwhile, `main.rs:355` does a 4th conversion for its own `gray_page` variable, which it passes to `verify_text_region()` — but `match_font()` never receives this pre-computed grayscale.

**Fix:** Pass `&GrayImage` into `match_font()` instead of `&DynamicImage`. Compute it once in `main.rs` and thread it through. **Estimated speedup: 15-25% of total runtime eliminated.**

### H2. Coarse scoring: per-candidate font parsing, rendering, and image processing

**File:** `src/font_match.rs` lines 280-480

For each of ~50 candidates per line, the inner loop does:
1. `FontRef::try_from_slice(&entry.data)` — parses the font binary (~200KB avg) every time
2. `rendered_text_width()` — computes advance widths (cheap)
3. For each of up to 4 sample words:
   - `render_text_gray()` — renders text to a fresh `GrayImage` (glyph rasterization)
   - `otsu_binarize()` — full-image histogram + threshold pass
   - `morphological_open()` — erode + dilate, each O(w×h×(2r+1)²) = O(9·w·h) for r=1
   - `tight_crop()` — full pixel scan for ink bounds
   - `resize_to()` with Lanczos3 — expensive resampling
   - `threshold_mid()` — another full-image pass
   - `center_pad()` — allocates padded image, pixel-by-pixel copy
   - `gaussian_blur_3x3()` — two separate images, 9-tap kernel per pixel
   - `aligned_iou()` with shift range ±2 — **5×5 = 25 full-image IoU passes**, each O(w×h)
   - `ncc_score()` — full-image pass
   - `hu_moments()` — full-image pass
   - `fill_ratio()` — full-image pass

**Iteration count:** 50 fonts × 89 lines × 4 words = **17,800 complete image processing pipelines**.

At NORM_H=48 and typical word widths of 80-200 px, each render+process cycle touches roughly 10K-20K pixels across ~15 image operations. That's ~180M-360M pixel operations in the coarse loop alone.

**Fixes (multiple, cumulative):**

- **Cache parsed `FontRef` objects**: Font binary parsing is done per-line per-candidate. A `HashMap<PathBuf, FontRef>` or even a `Vec<FontRef>` (pre-parsed once at startup) would eliminate ~4,450 redundant `try_from_slice` calls. `FontRef` borrows the `&[u8]` data, so this is zero-copy.
- **Morphological open is expensive for radius=1**: The current implementation is naive O(w × h × 9) per operation (erode + dilate = 2 passes). For binary images, separable 1D passes reduce this to O(w × h × 3 × 2). Or skip it entirely — at NORM_H=48, a 1px open on tiny comparison images may not be worth the cost.
- **IoU shift search**: `aligned_iou()` with range=2 does 25 full-image scans. Each scan is O(w×h). For NORM_H=48 × ~120px width, that's ~144K pixels × 25 = 3.6M pixels per word per font. Consider computing IoU on downsampled images or using integral images for O(1) rectangular IoU.
- **Gaussian blur**: Applied twice (source and candidate) per word. At NORM_H=48, these are small images, but the blur is computed from scratch each time. The source blur is identical across all candidates — compute it once outside the font loop.

**Estimated speedup: 20-40% if font parsing cached + morph_open simplified + source-side computation hoisted.**

### H3. SSIM reranking: 30 candidates × 25 vertical shifts × windowed SSIM

**File:** `src/verify.rs` function `ssim_windowed_best_vshift()`

For each of the top 30 coarse candidates, `verify_text_region()`:
1. Renders all words onto a canvas (font parsing + glyph rasterization)
2. Detects and corrects skew (if any) — `rotate_gray()` is O(w×h) with bilinear interpolation
3. Calls `ssim_windowed_best_vshift()` which:
   - Creates 25 shifted copies of the rendered image (each O(w×h) pixel copy)
   - For each shift, runs `ssim_windowed()` which:
     - May clone+resize the image (line 256: `b.clone()` even when dimensions match!)
     - Steps through with 11×11 windows at stride 4, computing per-window weighted statistics

**Per-line cost:** 30 candidates × (1 render + 25 × (image shift + SSIM scan)) = 30 renders + 750 SSIM evaluations.

**For 89 lines:** 2,670 font renders + 66,750 SSIM evaluations.

Each SSIM evaluation on a typical line region (~800×40 px = 32K pixels) with stride 4 produces ~200 windows × 121 kernel operations = 24K weighted ops. Total: 66,750 × 24K ≈ 1.6B floating-point operations.

**Fixes:**
- **`b.clone()` on line 256 of verify.rs**: When dimensions already match (common case), this needlessly clones the entire image. Already guarded by an if-statement but the else branch still clones. Change to borrow.
- **Early termination in shift search**: If a shift produces SSIM > 0.95 (or some threshold), stop searching. Most correct fonts will hit near-perfect alignment quickly.
- **Shift via offset arithmetic, not pixel copy**: Instead of creating 25 full image copies, pass the shift offset into the SSIM kernel and adjust coordinates on the fly.
- **Skip SSIM for obvious mismatches**: If coarse score is much lower than the leader, skip SSIM entirely (already top-N truncated, but could prune further).
- **Reduce shift range**: ±12 means 25 shifts. If typical misalignment is ±3-4 px, try ±6 first and only expand if the best is at the boundary.

**Estimated speedup: 15-25% of total runtime.**

### H4. Per-line processing is entirely sequential

**File:** `src/main.rs` lines 365-410 (the line match loop)

The main processing loop iterates over lines sequentially:
```rust
for line in lines.iter() {
    // ... font match (the expensive part) ...
}
```

Each line's font matching is independent — there's no data dependency between lines. This is a natural target for `rayon::par_iter()`.

**Caveats:**
- The paragraph-grouping pass (Pass 1.5) needs all line results before running, but it could be a sequential post-pass after parallel matching.
- Font catalog is read-only during matching — safe to share.
- `page_img` / `gray_page` are read-only — safe to share.
- The char index is read-only — safe to share.

**Estimated speedup: Near-linear with core count. On 4 cores: ~3-4× speedup = 60-75% reduction in wall time.**

---

## MEDIUM IMPACT — Estimated 10-20% of total runtime

### M1. Font data (`Vec<u8>`) cloned into `FontMatchResult`

**File:** `src/font_match.rs` lines 542, 558

When a font wins, its entire binary data (`Vec<u8>`, typically 100-400KB) is cloned into the result:
```rust
font_data: entry.data.clone(),  // clones ~200KB
```

This happens twice per line (once for SSIM rerank winner, once for coarse-only fallback). For 89 lines, that's ~89 × 200KB = ~17MB of unnecessary allocation+copy.

Later in `main.rs`, `font_data` is cloned again into `majority_data` for paragraph grouping (line 424), and yet again when the majority font replaces a line's result (line 488).

**Fix:** Use `Arc<Vec<u8>>` or `&[u8]` references instead of cloning raw bytes. The font data already lives in `FontEntry.data` for the lifetime of the processing.

### M2. Redundant SSIM verification in main.rs

**File:** `src/main.rs` lines 468-490

The SSIM reranking in `font_match.rs` already computed an SSIM score for the winning font. Then `main.rs` computes SSIM *again* for verification (lines 472-485):
```rust
if !keep_raster && !args.no_verify {
    let (score, _dy) = verify::verify_text_region(...);
```

And the paragraph grouping pass (line 456) also calls `verify::verify_text_region()` for each non-majority line at body size.

**Fix:** Return the SSIM score from the reranking phase and reuse it for the verification threshold check. Saves ~89 redundant full SSIM computations.

### M3. Font data stored in memory 5,000×

**File:** `src/font_scan.rs` `scan_fonts()`

`FontEntry.data: Vec<u8>` stores the full font file bytes for every catalog entry. With ~5,000 fonts averaging ~200KB each, that's **~1GB of font data in RAM**.

Additionally, when OT variants are detected, the font entry (including its `data`) is `clone()`d (line: `let mut var_entry = fe.clone();`), multiplying this further.

**Fix:** Use `Arc<Vec<u8>>` to share font data between base entries and their OT variants. Or memory-map font files and store `Mmap` handles instead of `Vec<u8>`, allowing the OS to page them in/out on demand.

### M4. Index serialization: font names stored 101× per font

**File:** `src/char_index.rs` `save_index()`

Each `FontCharEntry` stores the full font key string (typically a ~50-byte file path). With 4,965 fonts × 101 characters, that's 501,465 string copies in the index.

Estimated name overhead in the 114MB index file:
- ~4,965 fonts × 101 chars × (4 bytes length + ~50 bytes name) = **~27MB** (24% of the index)

**Fix:** String interning / dictionary encoding. Write a string table once, then store 2-byte or 4-byte indices per entry. This would reduce name storage from ~27MB to ~250KB (string table) + 501K × 2 bytes (indices) ≈ ~1.3MB. **Net savings: ~25MB (22% smaller index).**

### M5. Image operations in `aligned_iou` are O(n²) in shift range

**File:** `src/font_match.rs` `aligned_iou()`

With `range=2`, this iterates 25 shifts × every pixel. Each shift recomputes IoU from scratch. For images of size W×H:
- Total pixel operations: 25 × W × H
- With NORM_H=48, W≈120: 25 × 5,760 = 144K per word per font

This is called 50 fonts × 4 words × 89 lines = 17,800 times → 2.56B pixel operations.

**Fix:** Use integral images (summed area tables) to compute IoU in O(1) per rectangle after O(W×H) precomputation. Or reduce shift range from ±2 to ±1 (9 shifts instead of 25).

---

## LOW IMPACT — Estimated <5% each

### L1. Morphological operations use naive square kernel

**File:** `src/font_match.rs` `morphological_erode()`, `morphological_dilate()`

Each is O(W × H × (2r+1)²). For r=1 on NORM_H=48 × 120px images: 5,760 × 9 = 51.8K ops per call, ×2 (erode+dilate) = 103.6K per word per font.

**Fix:** Separable 1D passes: O(W×H×(2r+1)×2) = 5,760 × 6 = 34.6K. Or use a rolling min/max approach for O(W×H) regardless of radius.

### L2. Feature extraction `as_weighted_slice()` normalizes on every call

**File:** `src/char_index.rs` `CharFeatures::as_weighted_slice()`

L2-normalizes three groups of the feature vector (magnitude computation + division). Called once per crop per character during `search_candidates()`. With ~15 characters per line × 89 lines = ~1,335 calls, each doing √(Σx²) on 32+7+18 dimensions. Negligible individually but adds up.

**Fix:** Pre-compute and cache the weighted slice during feature extraction.

### L3. `score_entries` Vec rebuilt per line

**File:** `src/font_match.rs` lines 216-222

```rust
let score_entries: Vec<&FontEntry> = if use_char_gate {
    let ci = char_index_names.as_ref().unwrap();
    catalog.iter().filter(|e| ci.contains(&e.font_key())).collect()
} else {
    catalog.iter().collect()
};
```

This filters 5,000 entries through a `HashSet.contains()` check that calls `font_key()` (which allocates a `String` via `format!()` for variant entries) on every entry, 89 times.

**Fix:** Pre-compute `font_key()` for all catalog entries once and store it. Or use entry indices instead of string matching.

### L4. `top_candidates` periodic sort in coarse loop

**File:** `src/font_match.rs` lines 477-480

```rust
if top_candidates.len() > TOP_N_RERANK + 10 {
    top_candidates.sort_by(...);
    top_candidates.truncate(TOP_N_RERANK);
}
```

Sorts ~40 elements when the vec exceeds 40 entries. With ~50 candidates per line, this triggers once per line. Sorting 40 elements is cheap, but a min-heap (BinaryHeap) would be O(log n) per insertion instead of periodic O(n log n).

### L5. OCR pre-processing does per-pixel contrast stretching + sharpening

**File:** `src/ocr.rs` `extract_text_regions()`

Three full-image passes (contrast stretch, blur, sharpen) on the page image before OCR. At 2550×3300 = 8.4M pixels × 3 passes = 25.2M pixel operations. Done once per page, not per line, so relatively minor.

### L6. k-d tree leaf sort on every insertion

**File:** `src/char_index.rs` `knn_recursive()` line 404

Inside the kNN search, leaf node processing sorts the `best` vec after every insertion:
```rust
best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(...));
if best.len() > k { best.truncate(k); }
```

For k=1 (the common case in `nearest_within_factor`), this is trivially cheap. For larger k, a BinaryHeap would be better, but since k is always 1 here, no real impact.

### L7. `CharIndex` entries duplicate feature data between `entries` HashMap and `kd_trees`

**File:** `src/char_index.rs`

The `entries` HashMap stores `FontCharEntry` (with `CharFeatures`), and the `kd_trees` store `KdPoint` (with `coords: [f32; FEAT_LEN]`). These are two copies of the feature vectors in memory:
- entries: 4,965 × 101 × 57 × 4 bytes ≈ 109MB
- kd_trees: same ≈ 109MB
- Total: ~218MB RAM for feature data alone

**Fix:** Have kd_tree nodes store indices into the entries Vec instead of owning coordinate copies. Or drop the entries HashMap after building trees (if no longer needed for merge/save operations).

### L8. `ssim_windowed` clones image `b` even when dimensions match

**File:** `src/verify.rs` line 256

```rust
let b = if a.dimensions() != b.dimensions() {
    image::imageops::resize(b, ...)
} else {
    b.clone()  // unnecessary clone
};
```

Clones ~32K pixels (800×40) needlessly in the common case. Called 66,750 times → ~2.14 billion pixels of unnecessary copying.

**Fix:** Use a `Cow<GrayImage>` or just borrow when dimensions match.

---

## Parallelism Opportunities Summary

| Opportunity | Current | Potential | Complexity |
|---|---|---|---|
| **Per-line font matching** | Sequential | `rayon::par_iter()` | Low — lines are independent |
| **Per-page processing** | Sequential | `rayon::par_iter()` | Medium — needs shared font catalog |
| **Coarse scoring loop** | Sequential per line | `par_iter()` over candidates | Low — read-only data |
| **SSIM reranking loop** | Sequential per line | `par_iter()` over candidates | Low — read-only data |
| **Vertical shift search** | Sequential | `par_iter()` over shifts | Low — independent computations |
| Index build | Already parallelized | — | Done |

---

## Recommended Priority Order

1. **H4 — Parallelize per-line matching** (biggest bang, lowest risk, ~3-4× wall time improvement)
2. **H1 — Eliminate redundant `to_luma8()`** (trivial fix, 15-25% savings)
3. **H2 — Cache font parsing + hoist source-side computation** (moderate refactor, 20-40%)
4. **H3 — SSIM reranking optimizations** (offset-based shifts, early termination, 15-25%)
5. **M2 — Skip redundant SSIM verification** (reuse rerank score, easy)
6. **M3 — Arc/mmap font data** (reduce 1GB→shared, moderate refactor)
7. **M4 — Dictionary-encode index names** (22% smaller index, moderate)
8. **L8 — Fix ssim_windowed clone** (one-line fix, eliminates 2B pixel copies)

**Estimated combined speedup:** With items 1-4 implemented, expect **10-20× wall-time improvement** (parallel + eliminating redundant work), bringing the 89-line specimen from ~3 min to ~10-20 seconds.

---

## Appendix: Memory Budget Estimate

| Component | Estimated RAM |
|---|---|
| Font catalog (`Vec<FontEntry>`) | ~1GB (5,000 fonts × 200KB avg data) |
| Char index entries | ~109MB (feature vectors) |
| k-d trees | ~109MB (duplicate feature vectors) |
| Page image (DynamicImage) | ~32MB (2550×3300 RGBA) |
| Grayscale page (GrayImage) | ~8MB per copy, ×3-4 copies |
| Coarse scoring working set | ~50KB per candidate (small images) |
| SSIM working set | ~500KB per candidate (line-size images) |
| **Total peak** | **~1.3 GB** |

The font catalog dominates memory. `Arc<Vec<u8>>` sharing between base entries and OT variants, or memory-mapped font files, would reduce this significantly.
