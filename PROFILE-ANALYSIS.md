# Unprint Profiling Analysis — LDA Classifier
**Date**: 2026-06-14  
**Test document**: berkeley-output.pdf (1 page, 34 text lines, 31 vectorised)  
**Release build**, AMD EPYC 9D25

## Wall-time Budget (26.5s total)

| Phase | Time | % |
|---|---|---|
| Font scan (disk) | 4.1s | 15% |
| Char-index load | 1.7s | 6% |
| Page load (cached) | 0.2s | 1% |
| split_wide_whitespace | 6.1s | 23% |
| **Font matching** | **9.7s** | **37%** |
| SSIM verify | 3.9s | 15% |
| PDF output + misc | 0.8s | 3% |

## Font Matching Deep Dive (19.0s cumulative across threads → 9.7s wall)

71 CI calls across 34 lines (lines are processed in parallel, each line may call CI multiple times for candidate/recheck passes).

| Sub-phase | Cumulative | Per call avg | Notes |
|---|---|---|---|
| **alt (OCR correction)** | **15,344ms** | **216ms** | **81% of CI search time** |
| brute (primary search) | 350ms | 4.9ms | Fast — kNN on prepared vectors |
| feat (LDA projection) | 302ms | 4.3ms | Cheap |
| backfill (2nd pass) | — | — | 2.49M distance computations, ~63 passes |

### OCR Correction: The Dominant Cost

When a character crop's best match distance > 0.1, the code tries alternative characters. When > 0.5, it checks **ALL 106 indexed characters**. Each check runs `nearest_within_factor_brute` against that character's flat_vecs (~5K entries per char).

Top 6 worst CI calls (all dominated by `alt`):
- 1,686ms (brute=17ms, **alt=1,649ms**) — 52 crops, 41 fail gate
- 1,317ms (brute=14ms, **alt=1,287ms**)
- 1,298ms (brute=20ms, **alt=1,259ms**)
- 1,281ms (brute=19ms, **alt=1,240ms**)
- 1,220ms (brute=21ms, **alt=1,176ms**)
- 1,167ms (brute=14ms, **alt=1,129ms**)

Pattern: lines with many crops failing the quality gate (gate fail >> gate pass) trigger exhaustive alt search.

### Backfill: 2.49M Distance Computations

Top 10 backfill counts per CI call: 224K, 168K, 159K, 146K, 119K, 103K, 95K, 94K, 90K, 80K.

These are O(1) lookups per font (via `flat_vecs_font_idx`) but the sheer volume of `squared_distance()` calls on 28-dim LDA vectors adds up.

## split_wide_whitespace: 6.1s for 7 Splits

This function identifies word breaks within Tesseract lines. It does **per-line font identification** (a full CI search on the longest word per line) before analyzing gaps. That's 34 extra CI searches just for whitespace splitting — nearly as many as the main font matching pass.

## Optimization Opportunities (Ranked by Impact)

### 1. OCR Correction Early-Exit (~10-12s savings potential)
**Current**: When `min_dist_sq > 0.5`, checks all 106 characters with full brute-force search each.  
**Fix**: 
- Keep a running best-alt-distance; skip any alt char whose nearest entry is already worse than current best × 10 (the substitution threshold)
- Precompute per-char "centroid" vectors; quick-reject chars whose centroid distance is much larger than current best before running full brute search
- Or: limit `check_all` to top-N closest characters by centroid distance (e.g., top 10 instead of all 106)

### 2. split_wide_whitespace CI Bypass (~4-5s savings)
**Current**: Runs a full CI font identification per line (34 extra searches) before whitespace analysis.  
**Fix**: The main font matching pass already identifies the dominant font. Reorder so split_wide_whitespace runs AFTER font matching and reuses the already-identified font, or defer whitespace splitting until the font is known.

### 3. Backfill Vectorization (~1-2s savings)
**Current**: `squared_distance()` on 28-dim vectors one at a time.  
**Fix**: 
- Batch backfill: collect all query vectors for a font, compute distances in one vectorized pass
- Use SIMD (AVX2/AVX-512) for the inner product — 28 dims fits in one AVX-512 register (16 f32s × 2 loads)
- Since most backfill is "same N query vectors against one stored vector", transpose and use matrix-vector multiply

### 4. Font Scan Caching (~4s savings on subsequent runs)
**Current**: Scans all 6,157 system fonts from disk every run.  
**Fix**: Cache the font catalog (path → metadata) with filesystem timestamps; only rescan changed dirs.

### 5. Char-Index Load (~1.7s savings)
**Current**: Deserializes 513K entries from disk.  
**Fix**: Memory-map the index file instead of reading + deserializing. Store flat_vecs in a mmap-friendly layout.

### 6. SSIM Verify Parallelization (partial savings from 3.9s)
**Current**: Likely serial per-line SSIM computation.  
**Fix**: Parallelize across lines (already has a thread pool for font matching).

## LDA-Specific Observations

- **LDA projection** (the `feat=` timing) is NOT a bottleneck: 302ms total for 71 CI calls, ~4ms each.
- The 28-dim LDA vectors make `squared_distance()` cheap per call, but the volume (2.49M backfill + OCR correction) overwhelms.
- LDA's quality gate pass rate varies wildly: some CI calls have 0/3 pass, others 62/64 pass. The gate threshold (0.5 × thoroughness) may need tuning for LDA's different distance scale vs Fisher.

## Quick Wins Summary

| Fix | Estimated savings | Complexity |
|---|---|---|
| Skip split_wide_whitespace CI (reuse main-pass font) | 4-5s | Low |
| OCR alt centroid pre-filter | 8-10s | Medium |
| Font scan caching | 4s | Medium |
| Backfill SIMD batching | 1-2s | Medium |
| ~~Mmap char-index~~ | ~~1.7s~~ | ~~Done (index eliminated)~~ |
| Total potential | ~20s → 6-7s wall | |
