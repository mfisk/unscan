# Feature Vector: 99 Dimensions

Each character crop is normalized to NORM_H (48px) tall, then converted to a
99-dimensional feature vector for nearest-neighbor matching against the font
index. Features are grouped into five blocks with per-group L2 normalization
and weighting.

## Vector Layout

```
[0..31]   Column ink profile        32 dims   weight 0.40
[32..38]  Scalar v1                  7 dims   weight 0.30
[39..56]  Scalar v2                 18 dims   weight 0.30
[57..88]  Row ink profile           32 dims   weight 0.30
[89..98]  Scalar v3                 10 dims   weight 0.20
                                    ─────────
                                    99 total
```

## Group 1: Column Ink Profile (32 dims)

Vertical projection — ink density per column, resampled to 32 bins.

Captures the horizontal distribution of ink mass. Left-heavy vs right-heavy vs
centered. Distinguishes 'F' (ink left) from 'J' (ink right) and 'T' (ink
centered top, split bottom) from 'L' (ink left+bottom).

## Group 2: Scalar v1 (7 dims)

| # | Feature | Range | What it captures |
|---|---------|-------|------------------|
| 0 | `aspect` | 0-∞ | Width/height ratio. 'M' ≈ 0.8, 'l' ≈ 0.15 |
| 1 | `ink_density` | 0-1 | Fraction of dark pixels. 'O' < 'Q' < 'M' |
| 2 | `v_center` | 0-1 | Vertical center of ink mass. 'g' > 'b' (descender shifts CoM down) |
| 3 | `h_balance` | -1..1 | Left-right balance. 'F' < 0, 'J' > 0, 'H' ≈ 0 |
| 4 | `serif_score` | 0-1 | Ink concentration at baseline/capline. High for Times, low for Helvetica |
| 5 | `stroke_contrast` | 0-1 | Variation in stroke width. High for Didone, low for geometric sans |
| 6 | `xh_cap_ratio` | 0-1 | x-height to cap-height ratio (lowercase context) |

## Group 3: Scalar v2 (18 dims)

### Counter features (4 dims)
| # | Feature | What it captures |
|---|---------|------------------|
| 0 | `counter_area_ratio` | Area of enclosed white regions vs total area. 'O' has large counter, 'S' has none |
| 1 | `counter_centroid_x` | Horizontal center of counter space. Distinguishes 'b' (right counter) from 'd' (left counter) |
| 2 | `counter_centroid_y` | Vertical center of counter space. 'p' counter is high, 'b' counter is low |
| 3 | `counter_aspect` | Aspect ratio of counter region. Tall narrow counter ('D') vs round ('O') |

### Terminal features (4 dims)
| # | Feature | What it captures |
|---|---------|------------------|
| 4-7 | `terminal_angles[0..3]` | Stroke-ending directions binned into 4 quadrants (up/right/down/left). 'T' has down-pointing terminals, 'L' has right-pointing. Normalized so sum = terminal count |

### Shape features (2 dims)
| # | Feature | What it captures |
|---|---------|------------------|
| 8 | `ink_perimeter` | Perimeter of ink region / area. High for complex shapes ('W'), low for simple ('O') |
| 9 | `compactness` | 4π·area/perimeter². Circle = 1.0. Measures how efficiently ink fills its boundary |

### Horizontal crossings (8 dims)
| # | Feature | What it captures |
|---|---------|------------------|
| 10-17 | `h_crossings[0..7]` | Number of ink-to-white transitions along 8 evenly-spaced horizontal scan lines. 'B' has 4 crossings at midline (two counters), 'I' has 2 everywhere |

## Group 4: Row Ink Profile (32 dims)

Horizontal projection — ink density per row, resampled to 32 bins.

The vertical analog of the column profile. Captures where ink mass sits on the
vertical axis: ascender, x-height, baseline, descender. 'g' has a descender
spike that 'q' shares but 'a' lacks. 'T' has a heavy top row and thin stem
below. Together with the column profile, gives a 2D density fingerprint.

## Group 5: Scalar v3 (10 dims)

| # | Feature | Range | What it captures |
|---|---------|-------|------------------|
| 0 | `hole_count` | 0+ (÷4) | Enclosed white regions via flood-fill. 'O' = 1, 'B' = 2, '8' = 2, 'C' = 0 |
| 1 | `h_symmetry` | 0-1 | Left-right mirror similarity. 'O', 'X', 'H' ≈ 1.0. 'F', 'P' < 0.5 |
| 2 | `v_symmetry` | 0-1 | Top-bottom mirror similarity. 'X', '=' high. 'P', 'b' low |
| 3 | `skeleton_branch_pts` | 0+ (÷10) | Branch points after Zhang-Suen thinning. 'T' = 1, 'X' = 1, 'I' = 0, 'M' = 3+ |
| 4 | `skeleton_end_pts` | 0+ (÷10) | Endpoints after thinning. 'T' = 3, 'O' = 0, 'C' = 2 |
| 5 | `corner_count` | 0+ (÷20) | L-shaped boundary junctions. Serifs create corners; sans-serif has fewer. 'M' > 'I' |
| 6-9 | `quadrant_density[0..3]` | 0-1 | Ink density in TL / TR / BL / BR quadrants. 'P' has ink TR but not BR. 'L' has ink BL+BR but not TR |

## Weighting

Each group is independently L2-normalized to unit length, then scaled by its
weight. This prevents high-dimensional groups (32-bin profiles) from
dominating lower-dimensional scalar groups purely through dimension count.

Weights were tuned via Fisher discriminant analysis over 353K index entries and
9K rasterized scan crops:

| Group | Dims | Weight | Rationale |
|-------|------|--------|-----------|
| Column profile | 32 | 0.40 | Highest single-group discriminative power |
| Scalar v1 | 7 | 0.30 | Core geometric features |
| Scalar v2 | 18 | 0.30 | Structural/topological features |
| Row profile | 32 | 0.30 | Complements column profile |
| Scalar v3 | 10 | 0.20 | Fine-grained discriminators, lower individual Fisher ratios |

## OCR Correction Gate

After per-character nearest-neighbor search, a correction gate checks whether
Tesseract's OCR character label is plausible:

- If the OCR char's best match distance > 0.1 AND > 10× worse than an
  alternative character's best match, substitute the alternative.
- For moderate distances (0.1-0.5): check confusable pairs only (O↔0, l↔1, etc.)
- For catastrophic distances (>0.5): scan all indexed characters.

This catches genuine OCR mangling (Q→dingbat at 50-100× gap) without
interfering with style confusion (e vs c at 2-5× gap, which is a font
discrimination signal, not an error).

## Index Version

INDEX_VERSION = 7. Changing FEAT_LEN or feature computation requires bumping
this to force a full index rebuild. The index stores pre-computed feature
vectors; stale vectors from a prior version produce meaningless distances.

## Normalization: `normalize_to_ink_bounds()`

Both index-time (rendered glyphs) and scan-time (extracted crops) pass through
the same normalization:

1. Find tight ink bounding box (threshold 200)
2. Add 1px padding on all sides
3. Resize to NORM_H (48px) tall, preserving aspect ratio

Using a single shared function ensures index and scan produce identical
geometry for the same glyph, preventing systematic bias in feature vectors
(see DEBUGGING.md for details on index/scan crop geometry mismatches).
