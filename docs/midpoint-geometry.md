# Midpoint Geometry Matching

Spec for how `unprint` measures character centers/pitches and scores fonts via midpoint geometry. See also [`text-matching-approach.md`](text-matching-approach.md) for pipeline context.

## 1. Goal

Match observed ink centers and inter-character pitches from the scan to predicted centers/pitches from each candidate font. Geometry is a generative log-likelihood added to the LDA classifier logit: `lp = logit + GEO_WEIGHT * (h_ll + v_ll)`. Focus is **midpoint-only** – no ligature vs non-ligature blending beyond choosing the winning segmentation path.

## 2. Definitions

For each word `w` with `n` characters:

- `b_left = boundaries[i]`, `b_right = boundaries[i+1]` – integer column bounds from segmentation (VP + seam carving).
- `seam_paths: HashMap<boundary_x, Vec<[row, seam_x]>>` – per-row seam x for each boundary.
- `WordGeoMeasurement { chars: Vec<CharInkBounds>}` – one per word.
- `CharInkBounds { cx, cy, x_min, x_max, y_min, y_max, width, height}` – in pixels.
- `obs_cx[i], obs_cy[i]` – measured from ink.
- `pred_cx[i], pred_cy[i]` – predicted from font (cached `GeometryCache` in font units → px via `center_span_scale`).
- `obs_word_cy = mean(obs_cy)`, `pred_word_cy = mean(pred_cy)` – zero-centers vertical by construction, `Σ v_err = 0`.
- `obs_pitch[i] = obs_cx[i] - obs_cx[i-1]` for `i>0`, similarly `pred_pitch[i]`.
- `v_err = (obs_cy[i]-obs_word_cy) - (pred_cy[i]-pred_word_cy)`
- `h_err = obs_pitch[i] - pred_pitch[i]` for `i>0`, else `None`

## 3. Measurement – `measure_char_ink_bounds`

**File:** `crates/unprint-geometry/src/char_bounds.rs:18-120`

For each row `y` in `[0,h)`:

```
left_limit(y) = max( x0_rect, seam_left[y]) // inclusive, from seam_paths[b_left]
right_limit(y) = min( x1_rect, seam_right[y]) // exclusive, seam goes to right char
scan x in left_limit.. right_limit
```

- **Bounds (`x_min/x_max/y_min/y_max`):** binary `pixel < 200` ⇒ dark enough to count as ink. Sets `has_ink`, updates extremes.
- **Weighted centroid (current, 2026-08-03):** `if pixel < 250 { m = 255 - pixel; sum_x += m*x; sum_y += m*y; sum_m += m}` → `cx_w = sum_x/sum_m`, `cy_w = sum_y/sum_m`. Includes light fringe `200..249` (e.g. 247→mass 8, 193→62) that binary `<200` dropped, fixing right-edge inset.
- Fallbacks: `sum_m==0` → `cx = (x0+x1)/2`; `!has_ink` → zero width but weighted center.

Legacy `src/geometry_classifier.rs:measure_char_ink_bounds` still uses binary `(x_min+x_max)/2` and is being migrated to the crate version.

**Seam semantics (critical):**
- Right seam is exclusive: `right: max(seam_x,b_left)..b_right` is **right's** interval? Actually current impl: `left_limit = seam_left[...].max(x0)` inclusive to right, `right_limit = seam_right[...]` exclusive (`left_limit..right_limit`). So pixel at `seam_x` belongs to **right** char. `left: b_left..min(seam_x,b_right)` exclusive of seam column, `right: max(seam_x,b_left)..b_right` inclusive.
- Example seam 63 in `dogs.` word: `seam_paths[63]=[[0,64]…[16,64],[16,63]…]`, `col63` at y13 `142/255` (dark 113), y14 `142`, `col64` white. With seam=63, `col63` is dead-gap or belongs to `s` when seam=63, so `g = 42..63` excludes it – explains why `g` not wider from `s` bleed. Confirmed via `python3 tools/seam_viz.py audit.json "dogs." 21 63 --line 15`.

## 4. Coverage vs Gamma – Why Simple `(255-p)/255` is Non-Linear

Simplest AA assumes `% coverage → gray` linearly, but real grayscale AA must account for gamma:

> you'd think each component will be set to half brightness... but actually it's more complex than that: you have to account for gamma

- Ideal: `coverage c ∈ [0,1]` linear in linear-light.
- Display: `gray = 255 * (1-c)^(1/γ)` or `255*(1-(1-c)^γ)`, γ≈2.2 sRGB. So 50% coverage ≠128, ~186. Our 193 (≈24% in 0-255) is `c_lin ≈ 1-(193/255)^2.2 ≈0.48` linear, not 0.24.
- ClearType/gamma registry tunables `FontSmoothingGamma 1000..2200` confirm gamma matters.
- Scanned ink adds PSF, ink spread, JPEG.

**Fractional edge proposal (next experiment):**

Current weighted centroid fixes center but still reports integer `x_min/x_max`. Next: treat fringe coverage as partial pixel width.

Pixel at `x` spans `[x, x+1)`:
- `c = (255-p)/255` (naive) or gamma-corrected `c_lin = 1-(p/255)^γ` or `((255-p)/255)^γ`.
- Left edge: ink is rightmost `c` of pixel → `L = x_min + (1-c_left) = (x_min+1) - c_left`.
- Right edge: ink is leftmost `c` → `R = x_max + c_right = (x_max+1) - (1-c_right)` – same as user correction: subtract `(1-c_right)` from outer bound.
- Width `= R-L = (x_max-x_min+1) - (2 - c_left - c_right)`? Actually `(x_max + c_r) - (x_min+1 - c_l) = x_max-x_min-1 + c_l + c_r`; when `c=1` → `x_max-x_min+1` full width.
- Then `cx = (L+R)/2`, similarly for y. Keep seam limits.

For `s` right fringe `p=247` → `c=8/255=0.031` naive, `R=77.031` vs old `78` (or `76` binary). Trims 0.97px inset needed.

Plan: sweep γ 1.0 vs 1.8 vs 2.2, verify `'.'` leak (col16 115/96) not pulled into `s`.

## 5. Scoring – `quantized_ll`

**File:** `crates/unprint-geometry/src/params.rs:52-78`

First-principles uniform quantization:

```
σ_center = sqrt(1/12) ≈ 0.28867513459481287 // center uniform [-0.5,0.5]
σ_pitch = sqrt(2/12) = sqrt(1/6) ≈ 0.40824829046386302 // difference of two centers
```

User requirement: inflection points same for pitch and vertical – based on `0.75*σ` per dimension, not shared absolute.

- `thresh_center = 0.75*σ_center ≈0.2165`
- `thresh_pitch = 0.75*σ_pitch ≈0.3062`
- `K_INNER = 2.0`, `OFFSET = 0.2109375` (ensures continuity)
- Inside: `|e| < thresh → -0.5*(e/(K_INNER*σ))^2` → 0.2px center -0.06, pitch -0.03 (almost free, penalty near zero per "0.2 should get almost none")
- Outside: `|e| ≥ thresh → -0.5*(e/σ)^2 + OFFSET`
- Continuity check: `inner(thresh)= -0.5*(0.75/2)^2 = -0.0703125`, `outer_raw(thresh)= -0.5*0.75^2 = -0.28125`, `+OFFSET = -0.0703125` continuous.
- At 0.6px: center `-1.95`, pitch `-0.87` (per "0.6px should get quite a bit of penalty").

Flat-top alternative: `FLAT_TOP_DEFAULT = 0.45*0.75 = 0.3375` (env `UNPRINT_FLAT_TOP`, `QUANT_HALF_WIDTH_PX`). Currently `quant_half_width_*` params are stored but not used in `quantized_ll` when conditional path active (`_half_width` ignored); kept for env override compatibility.

## 6. Pruning – Midpoint Prune

**File:** `src/font_match.rs` / `src/main.rs` logging

```
BASE = -12.0 // worst correct-font letter on BAP: SourceSerif4-400 'T' p5:23 = -10.15
threshold = BASE * thoroughness.max(0.1)
thr=1.0 → -12, thr=2.0 → -24 (looser), etc.
min_ll(font) = min_{chars} (h_ll+v_ll)
prune if min_ll < threshold, unless in ensure_font_keys or geo_map empty
MIN_KEEP=10: re-add best pruned by highest min_ll if kept<10
If kept empty: keep single max geo_ll
```

Sound with `USE_SUM_AGG=true` (`Σ w·lp`) because score doesn't depend on per-position `best`. Previous squared-gap mode was unsound: `best_pruned < best_full` flipped `IBMPlexSans-400` vs `Devanagari-400`.

Recent BAP logs show `pruned 2749/2750 at -7.20` (base -7.2 variant in testing).

## 7. Worked Example – `dogs.` word 87×42

From debug dump `your_files/debug-p1l15/`:

- Splits `[0,21,42,63,78,87]` ink cx `10.0,31.0,52.5,69.5,83.0`
- `g` rightmost dark `x_max=62` (6 dark pixels), gap 64-65 white, `s` leftmost dark at 63 but seam excludes → `g` not wider.
- `s` right fringe: binary `<200` excludes `p=247/251` at edge, includes `193`; binary `(x_min+x_max)/2` → `cx=69.5`. Weighted `cx = sum(m*x)/sum(m)` with `m=255-p`, `p<250` → edge mass 8+4 pulls `cx` to ~70.0, error `g→s` `-0.63 → -0.13` inside `0.75σ=0.3062`.
- `'.'` leak col16 `115/96` – currently `<250` mass would include if seam mask allowed, but seam mask excludes, so safe. Check when implementing fractional edge.

Seam 63: `col63` rows 13-15 `142,142,242` with `col64=255` white; `col64` rows 16 `251`, 17 `232` dark=4/23 → mass 4/23 not width. Current inclusive/exclusive logic puts col63 dark in dead-gap or `s`.

Audit baseline: `d3f76a2` 428/84.4%/0.9425 ZNCC with `K_INNER=2σ OFFSET=0.2109375`; `1.5σ` trial 425/83.8%/0.9419 worse. Last BAP task 52 running with weighted center.

## 8. Implementation Map

- `crates/unprint-geometry/src/params.rs:6-11` – sigmas, `FLAT_*`
- `params.rs:52-78` – `quantized_ll`, `K_INNER`, `OFFSET`, continuity comment
- `crates/unprint-geometry/src/char_bounds.rs:18-120` – `measure_char_ink_bounds`, seam `left_limit`/`right_limit`, `<200` bounds vs `<250` weighted, `sum_x/sum_y/sum_m`
- `char_bounds.rs:78-110` – fallback paths, `width = x_max-x_min+1`
- `src/geometry_classifier.rs:296-310` – `v_err`, `h_err`, `v_ll=quantized_ll(v_err, SIGMA_CENTER)`, `h_ll=quantized_ll(h_err, SIGMA_PITCH)`
- `src/geometry_classifier.rs:46-70` – comment deriving `σ=1/√12`, `σ_pitch=1/√6`, `E[err]=0` symmetry
- `src/font_match.rs` – `per_char_geo_for_font`, `GEO_WEIGHT=1.0`, `midpoint prune`

## 9. Next Experiments

1. **Gamma-corrected mass:** `c_lin = 1-(p/255)^γ` with γ sweep 1.0,1.8,2.2 in `char_bounds.rs` mass calc; measure `s` cx shift and `'.'` contamination.
2. **Fractional edge coverage:** `L = x_min + (1-c_left)`, `R = (x_max+1)-(1-c_right)`, `width = R-L`, `cx=(L+R)/2`, same for y. Compare hits/ZNCC vs weighted-center BAP (current task 52).
3. **Seam-aware mass:** ensure `col16` style leak (115) not counted as `s` fringe – keep seam mask before mass accumulation.
4. **Threshold sensitivity:** ablate `<200` vs `<220` for `x_min/x_max` while keeping `<250` mass – ensure not pulling raster noise.
5. **Prune base tuning:** re-evaluate `BASE -12` vs `-10.15` observed worst correct after weighted fix; may tighten to -11 without regression.

## 10. References

- `docs/text-matching-approach.md` § Geometry scoring, Mid point pruning – now points here
- `your_files/sigma-flat-top-search-report.md` – flat-top sweep history
- `tools/seam_viz.py` – seam visualization for `dogs.` case
- Previous BAP audits `test-docs/audit/audit.json` (507 GT, 364 hits minor 61 major 41 etc.)

---
Generated 2026-08-03 from live code `a918e3d` + weighted-center uncommitted diff.
