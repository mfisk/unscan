# Unscan Optimization Plan

Analysis of `src/` for performance improvements with **no algorithm changes**
and **no regressions**. Focus: caching, precomputation, memory reuse,
complexity reduction.

## Implementation Status

Many of the optimizations identified below have been implemented. Status markers:
- ✅ **Done** — implemented and verified
- ⏳ **Partial** — partially addressed
- ❌ **Not started** — still available for future work

### Completed Optimizations

1. **✅ 1.1: Shared flood-fill** — `flood_fill_from_edges()` computed once,
   shared between counter features and hole count.
2. **✅ 1.2: Merged pixel scans** — `compute_features()` uses a two-step
   approach: first pass gets bounds + totals, second restricted pass over
   ink bbox builds col_ink, row_ink, left_ink, and ink_mask simultaneously.
3. **✅ 1.5: Zhang-Suen LUT** — Thinning uses a 256-entry lookup table
   (`trans_lut`) for transition counting instead of per-pixel arithmetic.
4. **✅ 3.1: FontCache in word splitting** — `split_wide_whitespace_words()`
   accepts an optional `&FontCache` parameter, using `fc.load()` instead
   of raw `std::fs::read()`.
5. **✅ 4.1: Line-level CI for word splitting** — `split_wide_whitespace_words()`
   runs CI once on the longest word per line, then reuses the result for
   all words. No longer runs CI per word.
6. **✅ 6.1: OnceLock Gaussian kernel** — `gaussian_kernel_11x11()` uses
   `std::sync::OnceLock` to compute the kernel exactly once.
7. **✅ 5.2: Thread-local FreeType** — `render_via_freetype()` uses
   `thread_local!` for the FreeType library instance.
8. **✅ NEW: SSIM fast path** — Parallel dominant-font SSIM check before
   CI (threshold 0.90), skipping segmentation + CI entirely on hit.
   Candidate propagates across pages via frequency tally.
9. **✅ NEW: Font-metric word splitting** — `font_pair_ink_gap()` and
   `font_ink_width()` use `ab_glyph`'s `outline_glyph().px_bounds()` to
   derive per-character rendering scale and predicted inter-glyph gaps.
   Threshold = `round(expected) + 5`.
10. **✅ NEW: SSIM bail-below early exit** — `ssim_windowed()` and
    `ssim_windowed_best_vshift()` accept `bail_below: Option<f32>`.
    After processing ≥8 windows per row, if the running SSIM average is
    below the threshold, computation returns early. Used by the fast-path
    SSIM check (bail threshold = 0.90) to reject non-matching fonts
    cheaply. `verify_text_region()` threads the parameter through.
11. **✅ NEW: Precomputed crop features for audit** —
    `precompute_crop_features()` extracts weighted 99-dim feature vectors
    once per character crop. `per_char_distances_precomputed()` takes these
    precomputed vectors and computes distances against any font's reference
    glyphs without re-calling `compute_features()`. The audit path in
    `main.rs` uses this to compute per-char distances against ~74 fontmap
    fonts without redundant feature extraction (~35K → ~480 feature
    computations per page).

---

## 1. char_index.rs — Hot Path: Feature Computation & Search

### 1.1 Duplicate flood-fill in `compute_counter_features` + `compute_hole_count`
- **File:** `src/char_index.rs`, lines ~520–600 (`compute_counter_features`) and ~680–750 (`compute_hole_count`)
- **Pattern:** Both functions perform an identical edge flood-fill on the ink mask (BFS from border white pixels). `compute_counter_features` builds a `reachable` array, then `compute_hole_count` builds its own identical `reachable` array from scratch.
- **Current complexity:** O(w×h) × 2 = two full BFS passes over the same data
- **Proposed optimization:** Compute `reachable` once and pass it into both functions. Or merge them into a single function returning `(counter_area_ratio, cx, cy, counter_aspect, hole_count)`.
- **Expected impact:** **Medium** — saves one full BFS + allocation per character per font during index build. With ~100 indexed chars × ~5000 fonts = 500K redundant flood-fills eliminated.

### 1.2 Redundant pixel scans in `compute_features`
- **File:** `src/char_index.rs`, lines ~300–450 (`compute_features`)
- **Pattern:** The function makes **four separate full-image scans**:
  1. Lines ~310–335: full scan for ink bounds + total_ink + v_center
  2. Lines ~340–350: full scan for `left_ink` (h_balance)
  3. Lines ~352–365: column scan for `col_ink` (column profile)
  4. Lines ~380–395: row scan for `row_ink` (row profile)
  5. Lines ~410–420: another full scan to build `ink_mask`
  
  The `ink_mask` is built in scan 5, but scans 1–4 already visit every pixel.
- **Proposed optimization:** Single-pass: accumulate all per-pixel statistics (ink bounds, total_ink, v_center, left_ink, col_ink, row_ink, ink_mask) in one traversal. Write into pre-allocated arrays.
- **Expected impact:** **Medium-High** — reduces 5 passes to 1 pass over every pixel for every character in every font. At 48×~30 pixels per glyph image this is modest per-glyph but multiplied by 500K+ glyph images during index build.

### 1.3 `ink_mask` allocation inside `compute_features`
- **File:** `src/char_index.rs`, line ~410
- **Pattern:** `let mut ink_mask = vec![false; ink_w_u * ink_h_u];` — fresh allocation per character.
- **Proposed optimization:** In the index-build path (`build_char_index`), each per-font closure processes chars sequentially. Reuse a thread-local `Vec<bool>` buffer, calling `.clear()` + `.resize()` instead of allocating fresh each time.
- **Expected impact:** **Low-Medium** — reduces allocator pressure. ~100 allocations per font × 5000 fonts = 500K Vec allocations eliminated.

### 1.4 `VecDeque` allocations in flood-fill helpers
- **File:** `src/char_index.rs`, `compute_counter_features` (~line 530), `compute_hole_count` (~line 700)
- **Pattern:** `VecDeque::new()` inside each call, used for BFS. Could reuse a thread-local deque.
- **Proposed optimization:** Thread-local `VecDeque` reuse (clear between calls).
- **Expected impact:** **Low** — minor allocator savings.

### 1.5 `compute_skeleton_features` — Zhang-Suen thinning is expensive
- **File:** `src/char_index.rs`, lines ~760–870
- **Pattern:** Copies ink_mask to `Vec<u8>`, then runs iterative Zhang-Suen thinning with two full-image scans per iteration, collecting `to_remove` vectors each step. The algorithm is inherently O(k × w × h) where k = number of thinning iterations.
- **Current complexity:** O(k × w × h) with k typically 5–15 for a 48px glyph
- **Proposed optimization:**
  - Avoid the `Vec<u8>` copy — work directly on a bitfield (8× denser → better cache)
  - Pre-collect only border pixels instead of scanning the entire image each iteration
  - Use a `to_remove` bitfield instead of `Vec<usize>` to avoid allocation
- **Expected impact:** **Medium** — thinning is one of the heaviest per-glyph operations. Constant-factor improvement, not algorithmic.

### 1.6 `nearest_within_factor_brute` — two-pass scan
- **File:** `src/char_index.rs`, lines ~200–225
- **Pattern:** Pass 1 finds min distance, pass 2 collects everything within cutoff. Two full linear scans of the per-character point set.
- **Proposed optimization:** Single pass: track min distance and collect all points, then filter at the end. Or use a running cutoff approach. The second pass also sorts results; since we only need candidates, defer sorting to the caller.
- **Expected impact:** **Low-Medium** — this is called once per (character crop × search) during the OCR pipeline. With ~15 chars per line and maybe 50 lines per page, it's ~750 calls per page, each scanning ~5000 points.

### 1.7 `search_candidates` — OCR correction gate scans all indexed chars
- **File:** `src/char_index.rs`, lines ~1030–1070
- **Pattern:** When `min_dist_sq > 0.5` (bad OCR match), the code collects ALL indexed chars and runs `nearest_within_factor_brute` for each — that's ~100 alternative characters × linear scan of ~5000 fonts. This is O(100 × 5000 × FEAT_LEN) per bad character.
- **Current complexity:** O(N_indexed_chars × N_fonts × FEAT_LEN) per crop with bad OCR
- **Proposed optimization:** Keep the fast-path check (confusables only) but for the `check_all` case, precompute a single flat array of all (char, font_id, features) and scan it once, tracking the overall minimum. This reduces ~100 separate brute-force scans to 1 scan over a concatenated array.
- **Expected impact:** **Medium-High** for pages with many low-confidence OCR characters. For clean pages this code path is rarely hit.

### 1.8 `rebuild_vecs` — clones all font names
- **File:** `src/char_index.rs`, lines ~940–1000
- **Pattern:** `name_set.insert(e.font_name.clone())` for every entry across all characters. String cloning for ~500K entries.
- **Proposed optimization:** Build `name_set` from `&str` references first, then clone only unique names into `font_names_table`. Currently it clones every font name even if already in the set.
- **Expected impact:** **Low** — this runs once at startup, not in the hot loop.

### 1.9 `CharFeatures::weighted()` allocates a stack array each time
- **File:** `src/char_index.rs`, line ~175
- **Pattern:** `as_slice()` + `weighted()` each produce `[f32; FEAT_LEN]` on the stack. `weighted()` calls `as_slice()` internally, producing two 99-element arrays.
- **Proposed optimization:** Fuse `as_slice()` and `weighted()` into a single method that writes directly to the output array.
- **Expected impact:** **Low** — stack arrays are cheap, but this is called ~500K times during index build.

---

## 2. segment.rs — Seam Carving DP

### 2.1 `candidate_seams` — energy lookup via closure per-pixel
- **File:** `src/segment.rs`, lines ~530–650
- **Pattern:** `masked_energy(r, c)` is a closure called for every pixel in the DP, checking left/right path bounds each time. Two full DP passes (forward + reverse) over the segment.
- **Current complexity:** O(seg_w × h) per DP pass
- **Proposed optimization:** Pre-compute a masked energy matrix for the segment once (replace out-of-bounds pixels with infinity), then index directly without per-pixel branching. This also enables SIMD-friendly contiguous memory access.
- **Expected impact:** **Low-Medium** — seam carving is used only when VP splitting fails, which is uncommon for well-spaced text. But when it fires (connected characters), it can be called multiple times per word.

### 2.2 `SeamDp` stores full DP matrices
- **File:** `src/segment.rs`, lines ~470–530
- **Pattern:** `cost_fwd` and `cost_rev` are `Vec<Vec<f32>>` — a Vec of Vecs (non-contiguous memory, poor cache behavior). Each is `h × seg_w` floats.
- **Proposed optimization:** Use a flat `Vec<f32>` with index arithmetic `[r * seg_w + c]` instead of `Vec<Vec<f32>>`. This eliminates inner Vec allocations and improves cache locality.
- **Expected impact:** **Low-Medium** — only affects seam carving paths.

### 2.3 Vertical candidate scoring scans horizontally per-pixel
- **File:** `src/segment.rs`, lines ~640–690
- **Pattern:** For each candidate column at each row, the code walks left and right to measure `run_len` (dark pixel run). This is O(run_len) per pixel × O(h) rows × O(seg_w) candidates = potentially O(seg_w² × h).
- **Proposed optimization:** Precompute horizontal dark-run lengths for each pixel in a single O(w × h) pass. Store as a 2D array: `run_len[y][x]` = length of the dark run containing pixel (x, y). Then the per-candidate lookup is O(1).
- **Expected impact:** **Medium** for words with many connected characters requiring seam carving. Most words skip this.

---

## 3. ocr.rs — Word Splitting & Bbox Expansion

### 3.1 `split_wide_whitespace_words` — CI search per word
- **File:** `src/ocr.rs`, lines ~766–920
- **Pattern:** For EVERY word in the document, the function:
  1. Calls `segment_characters` (full segmentation + DP)
  2. Crops and normalizes each character
  3. Calls `search_candidates` on the CI (full font matching) to identify the font
  4. Loads the matched font from disk (`std::fs::read`)
  5. Computes per-pair ink gaps using font metrics
  
  This is the **single most expensive function** in the pipeline.
- **Current complexity:** O(N_words × (segmentation + CI_search + font_load))
- **Proposed optimizations:**
  - **Font caching:** Use the `FontCache` instead of `std::fs::read(font_path)`. Currently the word splitter reads font files directly from disk, bypassing the cache entirely. Line ~855: `let font_data = std::fs::read(font_path).ok()?;`
  - **Batch font matching:** Instead of running CI search independently for each word, batch words into groups (e.g., by line) and share the font result. If the line-level CI match is already computed (it is — later in the pipeline), use that result here instead of re-running CI per word.
  - **Skip CI for short words:** Words with < 3 indexable characters produce low-quality CI matches anyway. Skip the CI search and use fallback gap detection for these.
  - **Cache font metrics:** `font_pair_ink_gap` and `font_ink_width` are called per character pair. For the same font at the same scale, these could be cached.
- **Expected impact:** **HIGH** — this is where most processing time goes. The font file read alone (bypassing cache) can add 1-5ms per word × hundreds of words = seconds.

### 3.2 `expand_words_to_ink` — per-pixel column scans
- **File:** `src/ocr.rs`, lines ~510–660
- **Pattern:** For each word, scans columns pixel-by-pixel to find ink extent. Inner loops: `(y_top..y_bot).any(|row| gray.get_pixel(col, row).0[0] < ink_threshold)`. Also does "phase 2 rebalance" scanning backward through previous word's columns.
- **Current complexity:** O(N_words × word_height × expansion_distance)
- **Proposed optimization:** Precompute column ink presence for the entire page (or per-line strip) as a `Vec<bool>`. One O(w × h) pass, then all word expansion queries become O(1) lookups.
- **Expected impact:** **Medium** — `expand_words_to_ink` runs on every word but the per-word work is typically small (expansion distance ≤ 20px).

### 3.3 `expand_bbox_to_ink` — redundant with `expand_words_to_ink`
- **File:** `src/ocr.rs`, lines ~435–505
- **Pattern:** Scans line-level ink extent, then `expand_words_to_ink` re-scans for word-level expansion. Both scan overlapping pixel regions.
- **Proposed optimization:** Combine into a single pass that computes both line and word ink extent together.
- **Expected impact:** **Low-Medium**

---

## 4. main.rs — Pipeline Orchestration

### 4.1 Per-line CI search runs twice: once in `split_wide_whitespace_words`, once in Pass 1
- **File:** `src/main.rs`, lines ~590–700 (Pass 1 font matching) + `src/ocr.rs` line ~820 (word splitting CI)
- **Pattern:** The word splitter calls `search_candidates` per word to identify the font for metric-based gap detection. Then Pass 1 calls `search_candidates` again per line (using the same char index). The word-level CI results are thrown away.
- **Current complexity:** Doubles the total CI search work
- **Proposed optimization:** Run word splitting AFTER the line-level CI match (Pass 1). Use the line's matched font for gap metrics instead of re-running CI per word. This requires reordering the pipeline slightly:
  1. OCR → assemble lines → expand bboxes
  2. Extract char crops and run line-level CI (current Pass 1)
  3. Use line-level CI winner to drive word splitting (currently runs before Pass 1)
  4. If splitting changes word boundaries, re-run CI only for affected lines
  
  Alternatively, pass the CI results from word splitting up to Pass 1 to avoid re-running.
- **Expected impact:** **HIGH** — eliminates ~50% of total CI search work.

### 4.2 `line_matches` collects clones of crop images for `corrected_char_crops`
- **File:** `src/main.rs`, lines ~695–710
- **Pattern:** `corrected_char_crops: Vec<(char, image::GrayImage)>` — clones every crop image just to change the char label. The GrayImage clone copies pixel data.
- **Proposed optimization:** Use a Vec of `(char, &GrayImage)` references instead of cloning. Or pass the char corrections as a separate mapping and let `per_char_distances` handle the indirection.
- **Expected impact:** **Medium** — saves ~15 GrayImage clones per line × 50 lines = 750 image copies per page.

### 4.3 Audit-related work even when `--audit` is not set
- **File:** `src/main.rs`, lines ~650–700
- **Pattern:** `fontmap_keys` computation, `fontmap_char_dists` calculation, and per-char distance computation for fontmap fonts run even when audit is disabled (they're gated by `args.audit.is_some()` in some places but not all).
- **Proposed optimization:** Verify all audit-only work is behind `args.audit.is_some()` gates. Currently `chosen_char_dists` is computed unconditionally.
- **Expected impact:** **Low** — `per_char_distances` is O(N_crops × N_fonts_in_index) but runs only on the winning font.

### 4.4 `mem_info()` and `/proc/self/maps` reads in hot loop
- **File:** `src/main.rs`, lines ~570–590
- **Pattern:** `eprintln!("  MEM line {} start: {}", li, mem_info())` — reads `/proc/self/status` for every line. At lines 2 and 45, reads and parses `/proc/self/maps` (iterates all memory mappings).
- **Proposed optimization:** Remove or gate behind a debug flag. These are I/O operations in the per-line parallel loop.
- **Expected impact:** **Low** — /proc reads are fast but unnecessary in production.

### 4.5 Font data re-loaded for SSIM verification + font-size computation
- **File:** `src/main.rs`, lines ~745–760 and ~780–810
- **Pattern:** `font_cache.load(&fm.font_path)` is called twice: once for SSIM verification, once for font-size computation. While the FontCache handles dedup, the Mutex lock is acquired twice.
- **Proposed optimization:** Load once, use for both purposes.
- **Expected impact:** **Low** — Mutex contention is minimal with 64-entry cache.

---

## 5. verify.rs — SSIM Verification

### 5.1 SSIM renders at multiple scales and picks the best
- **File:** `src/verify.rs`, lines ~80–130
- **Pattern:** `let scales = vec![2, 4];` — renders the text at 2× and 4× resolution, computes SSIM for each, picks the best. Each render involves FreeType shaping + rasterization + full-image resize.
- **Proposed optimization:** Start with the cheaper scale (2×). If SSIM > 0.8, skip the 4× render. Only compute 4× when the 2× result is ambiguous.
- **Expected impact:** **Medium** — eliminates ~50% of SSIM render work for good font matches (which are the common case).

### 5.2 `render_via_freetype` — thread-local FreeType library
- **File:** `src/verify.rs`, lines ~220–380
- **Pattern:** Already uses `thread_local!` for FreeType library — good. But `rustybuzz::Face::from_slice` is called every time. Rustybuzz face creation parses OT tables.
- **Proposed optimization:** Cache the `rustybuzz::Face` per font_data (by pointer or hash). Since the same font is used for many lines, this avoids re-parsing OT tables.
- **Expected impact:** **Low-Medium** — face parsing is fast but repeated per-line for the same font.

---

## 6. ssim.rs — SSIM Computation

### 6.1 `gaussian_kernel_11x11` — recomputed on every call
- **File:** `src/ssim.rs`, line ~100
- **Pattern:** `fn gaussian_kernel_11x11() -> [[f64; 11]; 11]` — computes the kernel from scratch each time, including exp() calls and normalization.
- **Proposed optimization:** Use a `const` or `LazyLock` for the kernel. Pre-compute at compile time or at first use.
- **Expected impact:** **Low** — the kernel computation itself is trivial (121 elements), but it's called per SSIM evaluation.

### 6.2 `gaussian_blur_3x3` — two-pass separable blur with per-pixel bounds checking
- **File:** `src/ssim.rs`, lines ~60–90
- **Pattern:** Separable 1D blur with `.clamp()` per pixel for boundary handling. Allocates two fresh `GrayImage`s.
- **Proposed optimization:** Skip boundary clamping for interior pixels (the vast majority). Process edges separately with clamping, interior without. Also reuse the tmp buffer.
- **Expected impact:** **Low** — blur runs twice per SSIM call (scan + render), but images are small (one line bbox).

---

## 7. font_scan.rs — Font Discovery

### 7.1 `scan_fonts` reads every font file from disk
- **File:** `src/font_scan.rs`, line ~135
- **Pattern:** `let data = std::fs::read(path).ok()?;` in `load_font_entry` — reads the full font file (~50-500KB) just to check if ab_glyph can parse it and to run `detect_oldstyle_figures` + `detect_ot_variants`.
- **Proposed optimization:** Already optimized — `fe.data = Vec::new()` drops the data after scanning. The font bytes are not retained. No change needed.
- **Expected impact:** N/A — already handled.

### 7.2 `detect_ot_variants` runs rustybuzz shaping ~25 times per font
- **File:** `src/font_scan.rs`, lines ~450–520
- **Pattern:** For each of 25+ OT features, creates a `UnicodeBuffer`, shapes the probe string, compares glyph IDs. This runs during font scanning, not the per-page pipeline.
- **Proposed optimization:** No change needed — this runs once during index build, not per-page.
- **Expected impact:** N/A — one-time cost.

---

## 8. geometry.rs — Geometry Detection

### 8.1 `text_mask` allocation is O(page_width × page_height)
- **File:** `src/geometry.rs`, line ~50
- **Pattern:** `let mut text_mask = vec![false; (w × h)]` — at 300 DPI, a letter-page is 2550×3300 = 8.4M booleans.
- **Proposed optimization:** Use a bitfield (`Vec<u64>`) — 8× denser, better cache.
- **Expected impact:** **Low** — one allocation per page.

### 8.2 Horizontal line detection scans every pixel on every row
- **File:** `src/geometry.rs`, lines ~70–100
- **Pattern:** Full page scan for horizontal runs, then full page scan for vertical runs. Two complete pixel traversals.
- **Proposed optimization:** Combine H/V line detection into a single pass, or skip geometry entirely for pages with no lines (early exit on uniform white pages).
- **Expected impact:** **Low** — geometry detection is a small fraction of total time.

---

## 9. color.rs / deskew.rs — Minor

### 9.1 `detect_text_color` runs per-line
- **File:** `src/color.rs` (not shown in detail)
- **Pattern:** Samples pixels within each line's bbox to determine text color. Runs once per line.
- **Proposed optimization:** No change — already per-line, constant work.

---

## Priority Summary

| Priority | Optimization | Expected Speedup | Risk | Status |
|----------|-------------|------------------|------|--------|
| **P0** | 4.1: Eliminate duplicate CI search (word split vs Pass 1) | 30-50% total time | Medium | ✅ Done — line-level CI in word splitter |
| **P0** | 3.1: Use FontCache in `split_wide_whitespace_words` | 10-20% total time | Low | ✅ Done |
| **P1** | 1.1: Merge duplicate flood-fills in char_index | 5-10% index build time | Low | ✅ Done — shared `flood_fill_from_edges()` |
| **P1** | 1.2: Single-pass pixel scan in `compute_features` | 5-10% index build time | Low | ✅ Done — merged pixel scans |
| **P1** | 5.1: Skip 4× SSIM render when 2× is good | 5-15% per-page time | Low | ❌ Not started |
| **P2** | 1.5: Optimize Zhang-Suen thinning | 3-5% index build time | Low | ✅ Done — LUT-based |
| **P2** | 1.7: Flatten OCR correction gate search | 2-5% per-page time (on bad OCR) | Low | ❌ Not started |
| **P2** | 2.2: Flat Vec for DP matrices | 1-3% on connected chars | Low | ❌ Not started |
| **P2** | 2.3: Precompute horizontal dark-run lengths | 1-3% on connected chars | Low | ❌ Not started |
| **P2** | 3.2: Precompute column ink for word expansion | 1-2% per-page | Low | ❌ Not started |
| **P3** | 4.2: Avoid GrayImage clones for char corrections | <1% | Low | ❌ Not started |
| **P3** | 4.4: Gate mem_info() behind debug flag | <1% | None | ✅ Done — gated behind UNSCAN_DEBUG_MEM |
| **P3** | 6.1: Const gaussian kernel | <0.1% | None | ✅ Done — OnceLock |
| **P3** | 8.1: Bitfield for text_mask | <0.5% | Low | ❌ Not started |
| **NEW** | SSIM fast path (dominant font from prev page) | High on typical docs | Low | ✅ Done |
| **NEW** | SSIM bail-below early exit | Medium (fast-path speedup) | Low | ✅ Done |
| **NEW** | Precomputed crop features for audit | High for audit mode | Low | ✅ Done |
| **NEW** | Font-metric word splitting | Accuracy improvement | Low | ✅ Done |

---

## Recommended Implementation Order

1. **P0: Use FontCache in word splitting** (src/ocr.rs ~line 855)
   - Change `std::fs::read(font_path)` → use a `FontCache` reference passed into the function
   - Zero risk, immediate payoff

2. **P0: Eliminate duplicate CI search** (src/main.rs + src/ocr.rs)
   - Pass the line-level CI winner into `split_wide_whitespace_words` instead of re-running CI per word
   - OR reorder pipeline: do word splitting after line-level CI match

3. **P1: Merge flood-fills** (src/char_index.rs)
   - Combine `compute_counter_features` and `compute_hole_count` to share `reachable` array

4. **P1: Single-pass features** (src/char_index.rs)
   - Merge the 5 pixel scans in `compute_features` into 1

5. **P1: Early-exit SSIM** (src/verify.rs)
   - Skip 4× render when 2× SSIM > threshold

6. **P2+: Remaining items** in priority order

---

## Validation

After each change:
```bash
source ~/.cargo/env
cd ~/workspace/repos/unscan
cargo build 2>&1 | tail -5
cargo test --test t60_specimen_accuracy_aa -- --nocapture 2>&1 | tail -20
```

Success criteria: t60 test passes with ≥93% accuracy (currently at 94.6%).
