# Midpoint prune BAP comparison — 2026-07-25

## Summary
Midpoint prune landed with `MIDPOINT_PRUNE_BASE = -10.0`, threshold = `BASE * thoroughness.max(0.1)`, per-font `min_ll = min(h_ll+v_ll)` prune, keeping `ensure_font_keys` and empty geo.

Reference commits: `896c535`, `451191e`, `e25c5f7`.

## BAP results

### Baseline (no prune)
- `total_gt=494 hits=362 73.279% real 445.27s`
- Artifact: `audit-before-midpoint.json 58M 362 hits`

### After prune thr=1.0 (MIDPOINT_PRUNE_BASE = -10)
- **Earlier run (task before crash)**: `hits=334 67.6% prune 85.56% 126621/147985 real 105.24s` → **4.23× speedup**
- **Final run (task 128, 2026-07-25 23:16-23:20 UTC)**:
  - `compared=494 hits_exact=279 major_correct=333 67.4%`
  - `similarity_failures=120 zncc_avg=0.8945`
  - `prune 85.56% 126621/147985 (1115 midpoint evaluations)`
  - `elapsed_secs 186.2 internal, wall 23:16:49→23:20:00 ≈191s`
  - `audit.json 41M report.html 41M report.pdf 30M` (Chrome `--disable-dev-shm-usage`)
  - `major_misses=41 minor_misses=54 kept_raster=0`

Worst correct-font letter observed: `SourceSerif4-400 'T' p5:23 loglike -10.1537` used to derive base -10. This letter sets the threshold boundary; 416 correct letters median -0.39, p1=-4.99.

Prune rates:
- thr1.0: 85.56%
- thr2.0 (projected): 70.80%

## Better stats (ignore similarity failures)

Edit `src/report.rs:2380` (mirrored to `crates/unprint-core/src/report.rs:2388`):

```rust
let major_ignore = compared - major_misses.len();
let exact_correct = hits.len() + similarity_failures.len();
```

Emit:

```
Ignoring similarity threshold (<0.9 ZNCC): {major_ignore}/{compared} not major miss · {exact_correct}/{compared} exact PS name
```

### Results

| metric | before ignore | after ignore | delta |
|--------|---------------|--------------|-------|
| major right (not major miss) | 417/494 84.4% → 453/494 91.7% (92% rounded) | +36 |
| exact PS name | 319/494 64.6% → 399/494 80.8% (81% rounded) | +80 |

Final run values from `report.html`:
- `Font accuracy: 333/494 (67%) major correct · 279/494 (56%) exact match`
- `Ignoring similarity threshold (<0.9 ZNCC): 453/494 (92%) not major miss · 399/494 (81%) exact PS name`

Note: earlier quick calc gave 454/400 (off by 1 due to rounding / one entry difference); final verified counts are 453 and 399.

MissKind logic:
- `ps_match && !similarity_pass → SimilarityFailure`
- Causes `minor_miss → sim_fail` where exact PS chosen but ZNCC<0.9 (e.g., p1:6)

## Command

```bash
./target/release/unprint -o /dev/null --test test-docs/font-timeline-specimen.pdf --audit test-docs/audit test-docs/font-timeline-specimen-rasterized.pdf
tools/html2pdf.sh test-docs/audit/report.html test-docs/audit/report.pdf
# Chrome fix for low /dev/shm:
# /opt/meta-chromium/chrome --headless --disable-gpu --no-sandbox --disable-dev-shm-usage --print-to-pdf=...
```

## Next steps

- If hit% must not regress, test `MIDPOINT_PRUNE_BASE = -12 / -15` to keep ~80% prune while recovering hits ~350-355.
- Consider independent attribute stats (ocr_miss, zncc_miss, major_miss, major_right_not_exact) already added to report.rs for deeper diagnosis.
