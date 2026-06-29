# Rendering Path Investigation — NORM_H=24 Training

## Changes Made

### 1. Training rendering path — now uses `render_glyph_normalised`
**File:** `src/bin/train.rs` (lines ~905-940)

The old training code used `render_char_at_native_height()` — a separate rendering function
with a different pipeline for heights < 48. It would manually resize and normalize, producing
images that differed from `char_render.rs`'s `render_char_normalised()`.

**Fix:** Replaced with `render_glyph_normalised()` — the **exact same function** used by:
- The classifier training data (used for LDA classification at runtime)
- The audit report reference images ("Correct" / "Picked" glyphs)
- The scan-side `normalize_to_ink_bounds()` path (shared geometry)

This ensures training, index, and audit all live in the same image domain.

### 2. Training data cache moved to `~/.cache/unprint/training/`
**File:** `src/bin/train.rs` (lines ~785-797)

Old: `.train_feat_tmp/` next to the output file (transient, in project directory).
New: `~/.cache/unprint/training/` (XDG-compliant, persists across runs, ~607MB).
Override with `--tmpdir`.

### 3. Added `render_glyph_hires()` / `render_char_hires()`
**File:** `src/char_render.rs`

New public functions that render at high resolution (ink height = 3×NORM_H = 72px)
then return the raw image for callers to downsample via `normalize_to_ink_bounds()`.
Intended for future scan-simulation augmentation. **Not currently used** in the
active pipeline — kept as infrastructure for domain-augmented training experiments.

## Experimental Results

### Why hires→downsample doesn't work (yet)

I tried three variants of the hires→downsample approach:

| Experiment | Render height | Index path | Compared | Primary Hits | Pct | Kept Raster |
|---|---|---|---|---|---|---|
| Hires 200px (both) | 200px→24px | hires→normalize | 7 | 1 | 14.3% | 492 |
| Hires 72px (both) | 72px→24px | hires→normalize | 15 | 3 | 20.0% | 481 |
| Hires 72px (train only) | 72px→24px | direct NORM_H | 51 | 2 | 3.9% | 443 |
| **Matched render (final)** | NORM_H direct | direct NORM_H | **437** | **82** | **18.8%** | **14** |
| Old 48px baseline | NORM_H=48 direct | direct NORM_H | 480 | 320 | 66.7% | 0 |

**Root cause:** The hires→downsample approach creates a domain mismatch. Real scan crops
come from 300dpi rasterization where typical body text is ~40-50px before normalization to
24px (a ~2× downsample). Index entries rendered at 72px or 200px and downsampled to 24px
(3×–8× downsample) have fundamentally different Lanczos3 artifact profiles. The LDA learns
to discriminate in the wrong pixel domain.

### The real NORM_H=24 problem

With the matched rendering path (final row above), the results are essentially identical to
the earlier "24px + LDA 24,18,12,9" run (18.6%, 83 primary hits) — confirming the rendering
path was **not the bottleneck**. The issue is intrinsic to NORM_H=24:

- At 24px, features carry less information than at 48px
- The quality gate (`min_dist_sq <= 0.5`) passes nearly everything (437 of 451 lines)
- But font discrimination is poor: 82 primary hits vs 320 at 48px

The old 48px-trained LDA weights worked better at 24px (64.7%) because they were trained on
48px data where features were more discriminative. When those weights project 24px features,
the projection somewhat compensates for the resolution loss. But retraining at 24px loses
that compensation.

## Downsample Algorithm Discovery

Both the scan path and the training/index path share `normalize_to_ink_bounds()`:
1. Find tight ink bounding box (pixels < threshold 200)
2. Crop to ink + 1px padding on all sides
3. Resize via **`image::imageops::resize()` with `FilterType::Lanczos3`** to height = NORM_H
4. Width scaled proportionally

This is the canonical downsample shared by all three domains:
scan crops, classifier training entries, and training images.
