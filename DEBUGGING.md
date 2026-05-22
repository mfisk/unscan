# Debugging Font Matching

How to diagnose and fix font identification errors in unscan.

## Pipeline Stages

**Critical invariant:** Index build and scan lookup must produce identical
images for the same glyph. Not "equivalent" — **identical code path**.

Two functions that both say "crop to ink + 1px pad + scale to NORM_H" are NOT
the same. `render_char_normalised` creates a white canvas sized to ink+2 then
draws vector outlines — guaranteed 1px pad, crisp edges. `normalize_to_ink_bounds`
receives a rasterized slice and crops/scales it — rasterization blur widens thin
strokes, anti-aliased pixels bleed past ink bounds, and sub-pixel coverage differs.

These differences are **not padding bugs** — they're fundamental to the two paths
producing different effective images. Key symptoms:
- **Thin strokes widen in scans**: Lato `l` renders at 7px in the index but
  the scan crop measures 12px wide. The blur makes it look like a heavier-weight
  font (SS3-Medium at 11px), and CI correctly picks the closer match.
- **ink_density diverges**: a 7px-wide `l` at 48px tall = density ~1.0. A 12px
  scan `l` = density ~0.69. This single character can swing the geometric mean
  enough to push the correct font out of the candidate list entirely.
- **Aspect ratio shifts**: every character is slightly wider in scans than in
  vector index renders, systematically biasing CI toward heavier-weight fonts.

**Debugging procedure for CI misses:**
1. Run `--diag-seg` to get the actual char crops CI receives.
2. Identify the **single worst-scoring character** for the correct font (compute
   per-char feature distances: aspect, v_center, ink_density between scan crop
   and index render). Fix the worst char first — geometric mean amplifies outliers.
3. Show the scan crop and index render **side by side at 8× zoom**. If the scan
   is visibly wider/blurrier, the problem is rasterization blur, not the crop code.
4. Only after per-char inspection rules out rasterization effects, check whether
   both paths call the same normalize function.

A line of scanned text goes through four stages. At each stage the correct font
can be lost:

```
  Scan Image
      │
      ▼
  ┌──────────────────────────────────┐
  │ 1. Character Extraction          │  Tesseract makebox → per-char crops
  │    extract_line_chars()          │  src/char_index.rs ~L1088
  └──────────────────────────────────┘
      │  (char, GrayImage) pairs
      ▼
  ┌──────────────────────────────────┐
  │ 2. CI Search                     │  Per-char feature vectors → brute-force
  │    search_candidates()           │  nearest neighbor → geometric mean score
  │                                  │  → σ cutoff
  │                                  │  src/char_index.rs ~L1429
  └──────────────────────────────────┘
      │  Vec<(font_key, score)> — ranked candidates
      ▼
  ┌──────────────────────────────────┐
  │ 3. Word SSIM Rerank              │  Crop full words from scan, render in
  │    word_level_rerank()           │  each CI candidate, pick best SSIM
  │                                  │  src/word_match.rs
  └──────────────────────────────────┘
      │  Winner font for the line
      ▼
  ┌──────────────────────────────────┐
  │ 4. Paragraph Grouping            │  Majority-vote body font replaces
  │    (main.rs ~L651)               │  outlier matches in same-size runs
  └──────────────────────────────────┘
      │  Final font assignment
      ▼
  PDF Output
```

## Stage 1 Detail: Character Segmentation Algorithm

Tesseract's character-level bounding boxes (HOCR) are unreliable — horizontal
positions drift (labeling the wrong glyph) and vertical extents clip descenders.
We use Tesseract only for **word** bounding boxes and **text** (both reliable),
then segment word crops into individual characters ourselves.

**Input:** Word crop image (trimmed to ink bounds via `ink_vertical_extent`),
known text from Tesseract (gives us the target character count N).

**Algorithm (two-pass):**

1. **Pass 1 — Vertical whitespace splits.**
   Compute column ink profile (sum of dark pixels per column, threshold < 200).
   Find contiguous runs of zero-ink columns. Each run is a definite character
   boundary — split at the midpoint. If we find N−1 gaps for N characters,
   we're done.

2. **Pass 2 — Whitespace path splits** (only for segments still containing
   multiple characters after Pass 1).

   For each under-split segment:

   a. Build a white-pixel reachability map: DP from top row to bottom row,
      8-connected, propagating only through white pixels (≥ ink threshold).
      A column in the bottom row is "reachable" if an all-white connected
      path exists from some top-row pixel down to it.

   b. Group reachable bottom-row columns into contiguous runs. Each run is
      a candidate split — a diagonal (or vertical) whitespace corridor
      between characters. This handles italic fonts where the inter-glyph
      gap slants rather than running straight down.

   c. Among the candidate paths, pick the one with the shortest (narrowest)
      corridor — this is the tightest character boundary, i.e. the split
      between the two closest characters in the segment.

   d. Split at that path's midpoint column. Recurse on the resulting
      sub-segments until all characters are separated or no more white
      paths exist.

3. **Fallback:** If neither pass produces enough splits (truly touching
   characters with no white path), fall back to uniform boundaries.

**Reference:** The whitespace-path concept is adapted from seam carving
(Avidan & Shamir, "Seam Carving for Content-Aware Image Resizing",
SIGGRAPH 2007), restricted to all-white connected paths rather than
minimum-energy seams that can cut through ink.

**Implementation:** `segment_characters()` in `src/char_index.rs`.

## Debug Tooling Reference

### `--diagnostic <DIR>`

Full HTML report with dark theme. Creates:

| Path | Contents |
|------|----------|
| `<DIR>/index.html` | Interactive HTML report — line-by-line CI candidates, word SSIM scores, inline crop/render images |
| `<DIR>/data.json` | Machine-readable: array of pages, each with lines containing `ci_candidates`, `words` (with per-word `candidates[].ssim`), `final_font`, `final_score` |
| `<DIR>/crops/` | Word-level scan crops: `p{page}_l{line}_w{word}_{text}.png` |
| `<DIR>/renders/` | Word-level font renders: `p{page}_l{line}_w{word}_{font}.png` |

The HTML report has collapsible CI candidate tables per line (showing score and
gap from #1) and inline word crop vs render images with SSIM scores.

```bash
unscan input.pdf -o /dev/null --diagnostic /tmp/diag
open /tmp/diag/index.html
```

### `UNSCAN_DUMP_CROPS=1`

Dumps the **character-level** crops that CI actually scores — the raw inputs to
stage 2. These are the individual character images extracted from the scan, 
normalized to `NORM_H` pixels tall.

Output: `/tmp/unscan-crops/p{page}_line_{text}/crop_{idx}_{char}.png`

```bash
UNSCAN_DUMP_CROPS=1 unscan input.pdf -o /dev/null
ls /tmp/unscan-crops/p2_line_1931__Times_New_Roman/
# crop_00_1.png  crop_04_T.png  crop_06_m.png ...
```

These are the actual images being turned into 59-dimensional feature vectors and
compared against the font index. If a character crop looks wrong (clipped,
merged with neighbor, includes artifacts), CI can't be expected to score it
correctly — fix the extraction first.

#### Diagnosing segmentation with crop dumps

When a line has too many or too few crops, segmentation is the culprit. Compare
crop count against expected character count (OCR text length minus spaces):

```bash
UNSCAN_DUMP_CROPS=1 unscan input.pdf -o /dev/null
ls /tmp/unscan-crops/p4_line_ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklm/ | wc -l
# 99 crops for 52 expected chars = over-segmentation
```

Over-segmentation (more crops than chars) means the charbox fallback or seam
pass is splitting inside glyphs. Each index in the dump gets two character
labels (e.g., `crop_03_D.png` and `crop_03_E.png` at the same index), so the
wrong glyph image is being compared against the wrong indexed character.

Under-segmentation (fewer crops) means multiple characters are stuck in one
crop, giving feature vectors that don't match any single indexed glyph.

To visualise just the VP (Pass 1) splits without seam carving, extract the
line/word crop and compute column ink profiles independently:

```python
import numpy as np
from PIL import Image

crop = Image.open("/tmp/line-crop.png").convert("L")
arr = np.array(crop)
col_ink = np.sum(arr < 200, axis=0)         # ink threshold 200
zero_cols = np.where(col_ink == 0)[0]       # columns with zero ink

# Group into contiguous runs — each run is a VP split
runs = []
if len(zero_cols) > 0:
    start = zero_cols[0]
    for i in range(1, len(zero_cols)):
        if zero_cols[i] != zero_cols[i-1] + 1:
            runs.append((start, zero_cols[i-1]))
            start = zero_cols[i]
    runs.append((start, zero_cols[-1]))

interior = [(s,e) for s,e in runs if s > 0 and e < arr.shape[1]-1]
print(f"VP splits: {len(interior)} interior runs for {N} expected chars")
```

If VP alone produces enough splits for N-1 boundaries, seam carving never
runs and you should see clean one-to-one crops. If VP falls short, seam
carving (Pass 2) fills the gap. If the total *still* falls short, the
charbox/uniform fallback fires — that's where over-splitting usually
originates, because the fallback doesn't respect glyph boundaries.

### `--include-font <NAME>`

Injects every font matching `<NAME>` (case-insensitive substring) into the
word SSIM reranking stage for **every line**, regardless of whether CI selected
it. Use this to see how a known-correct font renders and scores at the word
level even when CI pruned it.

```bash
unscan input.pdf -o /dev/null --diagnostic /tmp/diag --include-font merriweather
```

The included font appears in `data.json` CI candidates with score `-999.000`
(penalty marker). **Caution:** `--include-font` can cause the included font to
*win* lines it shouldn't (it bypasses CI entirely), so don't trust accuracy
numbers from an `--include-font` run.

### `--thoroughness <FLOAT>`

Default `1.0`. Scales CI thresholds:

| Threshold | Formula | Default (t=1.0) |
|-----------|---------|-----------------|
| kd-tree search radius | `2.5 × t` | 2.5× nearest |
| Quality gate | `0.5 × t` | 0.5 |
| Quorum divisor | `÷ t` | ÷ 1.0 |
| σ cutoff k | `0.5 × t` | 0.5 |

Higher values relax all gates — more candidates survive CI, slower but
more recall. Useful for diagnosing whether a font is being pruned by which
gate.

### `RUST_LOG=info` (stderr)

Per-line CI summary on stderr:

```
CI: 11 crops, 11 pass gate, 0 fail gate, 0 no_tree → 3259 fonts in voting
CI sigma cutoff: best=4.588 top50_σ=0.430 cutoff=4.373 → 2 of 209 kept
```

- **crops**: characters extracted from the line
- **pass/fail gate**: characters passing/failing the quality gate (min_dist² > 0.5)
- **no_tree**: characters with no index entry for that codepoint
- **fonts in voting**: fonts meeting quorum after per-char neighborhood search
- **σ cutoff**: how many fonts survive the statistical cutoff (top50 σ, k=0.5)

### `--compare`

Generates side-by-side overlay images (scan crop vs rendered winner) in
`<output_base>-compare/`. Quick visual sanity check of output quality.

### Test Harness

```bash
# Accuracy regression on clean raster specimen (30 fonts, 444 lines)
cargo test --release --test t60_specimen_accuracy

# Output quality + file size regression
cargo test --release --test t50_output_quality
```

`t60` compares every matched font against ground truth in
`test-docs/font-timeline-specimen.json`, including metric-compatible clone
aliases. Threshold: 95%.

---

## Procedure: Diagnosing a Font Miss on Clean Input

On a clean rasterized specimen (no physical scan skew), the correct font should
be **first or tied for first at every stage**. Anything else is a bug to fix.

**Always start with the character crops.** Every CI miss ultimately traces back
to what the model saw — the actual pixel images it scored. Before looking at
distances, features, or thresholds, look at the crops with your own eyes.

### Step 0: Establish Ground Truth

You need to know what font each line should be. For the specimen this is
`test-docs/font-timeline-specimen.json`. For other documents, identify the font
independently (original vector PDF, designer confirmation, etc.).

### Step 1: Dump and Visually Inspect Character Crops

This is the first step. Always.

```bash
rm -rf /tmp/unscan-crops
UNSCAN_DUMP_CROPS=1 RUST_LOG=info \
  unscan input.pdf -o /dev/null --diagnostic /tmp/diag 2>&1 | tee /tmp/unscan-stderr.log
```

For every failing line, open the crop PNGs and review them as a human:

```bash
ls /tmp/unscan-crops/p{PAGE}_line_{TEXT}*/
# Open the PNGs — look at them, don't just check sizes
```

**What to look for:**

- **Clipped glyphs:** Descenders chopped off ('g' bowl without tail looks like
  'o'; 'p' without descender looks like 'b'). This is the #1 cause of
  misidentification — a clipped character matches the wrong font because the
  wrong font's intact glyph accidentally resembles the clipped shape.
  *Example: Open Sans 'g' descender clipped → matches NotoSansMath because
  NotoSansMath's 'g' bowl is rounder, closer to the clipped circle shape.*

- **Neighbor fragments:** Part of an adjacent letter bleeds into the crop.
  A sliver of 'n' glued to 'a' changes 'a' into something unrecognizable.
  This poisons the feature vector for that character.

- **Merged characters:** Two glyphs in one crop (segmentation failure).
  The feature vector for 'fi' doesn't match any single indexed character.

- **Missing characters:** Check for gaps — if the line has 10 characters but
  only 7 crops, 3 were dropped. Fewer voters means less discriminative power
  and more noise sensitivity in the geometric mean.

- **Wrong character label:** The crop shows one glyph but is labeled as
  another (e.g., file says `crop_16_g.png` but the image is clearly an 'o').
  This means the crop is being compared against indexed 'g' shapes from all
  fonts — it will match whatever font has an 'o'-shaped feature vector for
  'g', which is nobody. Garbage in, garbage out.

**The crops are the ground truth of what CI scored.** If a crop is wrong, no
amount of scoring refinement can fix the match. Fix extraction first.
Everything downstream — distances, features, σ cutoffs — is irrelevant until
the inputs are clean.

### Step 2: Identify Wrong Lines

Parse `data.json` against ground truth. For each line, check:

1. Is `final_font` correct (or a known clone alias)?
2. If correct — is it CI rank #1 or tied for #1?

**Both** of these must be true on clean input. A line where the correct font
won but needed word reranking to rescue it is a CI scoring bug. A line where
the correct font lost entirely is worse — either a CI recall failure or a word
SSIM precision failure.

Categorize each miss:

| Category | Symptom | Investigate |
|----------|---------|-------------|
| **Bad crops** | Clipped, merged, or artifact-contaminated char images | Step 1 — fix extraction first |
| **CI recall failure** | Correct font not in CI candidates at all | Stage 2 (search radius, quorum, quality gate, σ cutoff) |
| **CI ranking failure** | Correct font in CI but not #1 | Stage 2 (per-char distances, geometric mean aggregation) |
| **Word SSIM failure** | Correct font is CI #1 but loses word rerank | Stage 3 (crop quality, render sizing, SSIM alignment) |
| **Paragraph grouping** | Correct font wins word rerank but overridden by majority vote | Stage 4 (grouping logic) |

#### Index/Scan Crop Geometry Mismatch

Both index-time and scan-time must produce the same crop geometry for the same
glyph. Both paths use `normalize_to_ink_bounds()` which: finds ink bounds
(threshold 200, matching `compute_features`), adds 1px padding each side, and
resizes to NORM_H height.

- **Index-time** (`render_char_normalised()`): renders at a scale where
  ink_h ≈ NORM_H into a canvas of (ink_w+2) × (ink_h+2). The image is
  already nearly at target geometry; the +2 padding matches normalize's 1px
  each side.

- **Scan-time** (`extract_chars_from_boundaries()`): slices the char from
  the word crop, then calls `normalize_to_ink_bounds()` to trim and resize.

If you see scan crops with visible padding that rendered crops don't have (or
vice versa), the normalize step is broken or being bypassed. This causes
geometry-dependent features (v_center, aspect ratio, ink density) to diverge
systematically between index and scan.

### Step 3: Inspect CI Scoring (Stage 2)

For CI ranking failures, look at `data.json` → `ci_candidates` for the line.
Note:

- **Score gap**: How far is the correct font from #1? (< 0.05 = noise-level,
  the geometric mean can't distinguish them)
- **What's beating it**: Is it a metric clone? An OT variant explosion? A
  genuinely different font?
- **Candidate count**: How many fonts survived the σ cutoff?

Common patterns:

| Pattern | Example | Root Cause |
|---------|---------|------------|
| Clone beats original | NimbusRoman > Times New Roman | Clone has slightly better char-level distances (expected, add alias) |
| OT variant swamp | 14 SourceSans3 variants at 4.517, correct font at 4.515 | One font×14 OT variants occupies 14 slots, pushes correct font to rank #14 despite 0.002 gap |
| Genuinely wrong | DejaVuSerif > IBM Plex Serif by 0.07 | Feature vectors don't distinguish these fonts — need better features or more characters |

To see the per-character distances for a specific font, increase log verbosity
or add temporary debug output in `search_candidates()` around the
`font_log_dists` map.

### Step 4: Inspect Word SSIM (Stage 3)

Open `/tmp/diag/index.html` in a browser. For the failing line:

1. Expand "Word SSIM" — see scan crop alongside each candidate's render
2. Check SSIM scores — the correct font should have highest SSIM on most words
3. Look for render problems: wrong sizing, clipped glyphs, baseline misalignment

The `data.json` word entries have `candidates[].ssim` and `candidates[].dy`
(vertical shift). If the correct font has good SSIM on all words but one, that
one word's crop may have an extraction issue.

### Step 5: Check Paragraph Grouping (Stage 4)

Paragraph grouping only overrides a line match when a different font has
majority vote among same-size lines in a paragraph run. Check `RUST_LOG=debug`
output for `paragraph regroup` messages.

This stage should be transparent on a specimen (each section has its own font,
no majority vote applies). If it's interfering, the bug is upstream — the
majority font shouldn't have won enough lines to trigger grouping.

---

## Key Source Locations

| What | Where |
|------|-------|
| CLI args | `src/cli.rs` |
| Pipeline orchestration | `src/main.rs` ~L445–650 |
| Character extraction (makebox) | `src/char_index.rs` `extract_line_chars()` ~L1088 |
| Feature computation | `src/char_index.rs` `compute_features()` |
| CI search + scoring | `src/char_index.rs` `search_candidates()` ~L1429 |
| σ cutoff | `src/char_index.rs` ~L1595–1615 |
| Word SSIM reranking | `src/word_match.rs` `word_level_rerank()` |
| Paragraph grouping | `src/main.rs` ~L651 |
| Diagnostic report | `src/diagnostic.rs` |
| Ground truth | `test-docs/font-timeline-specimen.json` |
| Clean raster specimen | `test-docs/font-timeline-specimen-rasterized.pdf` |
| Skewed scan specimen | `test-docs/font-timeline-specimen-scanned.pdf` |
| Accuracy test | `tests/t60_specimen_accuracy.rs` |

## Build and Run

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cargo/bin:$PATH"
cargo build --release
cargo test --release --test t60_specimen_accuracy
```

Cached specimen run: ~45s. Uncached (index rebuild): ~3min.


## Finding CI Failures: Correct Font Not First

When investigating mismatches, the first question is: did the CI (character index)
put the correct font at rank #1 (or tied for #1)? If not, the bug is in CI scoring
(feature vectors, σ cutoff, segmentation quality). If CI had it right but the final
match is wrong, the bug is in word SSIM reranking.

### Step 1: Run with audit log

```bash
./target/release/unscan test-docs/font-timeline-specimen-rasterized.pdf \
    -o /dev/null --audit-log /tmp/audit.json
```

### Step 2: Cross-reference CI top vs matched font

The audit log `text_entries[].ci_top` is a ranked list of `[font_path, score]` pairs.
`text_entries[].font_matched` is the final winner (after word SSIM rerank).
`text_entries[].word_rerank_winner` shows what the word-level SSIM picked.

To find lines where word reranking flipped away from CI #1:

```python
import json

with open('/tmp/audit.json') as f:
    entries = json.load(f)['text_entries']

for e in entries:
    ci = e.get('ci_top', [])
    matched = e.get('font_matched', '')
    if not ci: continue
    ci1 = ci[0][0].rsplit('/', 1)[-1].split('.')[0].split('|')[0]
    ci1 = ci1.lower().replace('-','').replace('_','')
    mn = matched.lower().replace('-','').replace('_','').replace(' ','')
    if ci1 not in mn and mn not in ci1:
        print(f"p{e['page']}:l{e['line_index']} CI#1={ci[0][0].rsplit('/',1)[-1]} → matched={matched}")
```

### Step 3: Use --diag-seg for deep inspection

```bash
./target/release/unscan INPUT.pdf -o /dev/null --diag-seg /tmp/diag-seg
```

Produces per-line directories, each containing per-word subdirectories:

```
/tmp/diag-seg/
  p1_The_quick_brown_fox/
    word_000_quick/
      word_crop.png          # raw word image from Tesseract bbox
      vp_overlay.png         # VP pass: cyan runs, red split midpoints
      seam_overlay.png       # VP (red) + seam (blue) overlaid
      final_overlay.png      # all passes: VP red, seam blue, charbox green
      summary.json           # vp_splits, seam_splits, charbox_added_splits, boundaries
      chars/                 # individual char crop PNGs: 00_A.png, 01_B.png, ...
    line_summary.json        # CI top 5, font_matched, ssim_score, word_rerank_winner
```

### Diagnosis decision tree

1. **segmentation mismatch** (summary.json `n_segments_produced != n_chars_expected`):
   Check VP overlay — did VP find enough zero-ink runs? If not, the font has
   touching/overlapping characters. Check seam overlay — did seam carving produce
   reasonable splits? Check charbox — did charbox fallback over-split?

2. **CI wrong** (correct font not in ci_top, or ranked low):
   Examine char crops in `chars/` — are they cleanly segmented? Compare against
   the reference char from the font index. Bad crops → bad feature vectors → wrong CI.

3. **CI right, SSIM flipped** (correct font is CI #1 but matched font differs):
   The word-level SSIM rerank chose a different font. Compare the rendered word
   images at the winning vs correct font — often the "wrong" font renders nearly
   identically (e.g., Noto Traditional Nushu's Latin glyphs vs Source Sans).

### Ground Truth Mapping

**Do NOT assume `page × 5` maps to section indices.** The specimen has 5
sections per page nominally, but sections overflow across page boundaries when
the content is long. A line on page 4 might belong to a section that starts on
page 5's section list in the JSON.

The t60 test avoids this by checking each matched font against ALL 30 sections,
not per-page. Any CI failure analysis script must do the same. Mapping a line to
"the sections on this page" will produce false failures for every line whose
section overflowed from an adjacent page — which can be the majority of reported
failures.

**Always verify ground truth before investigating a CI miss.** Show the actual
uncropped page image to confirm which font the line is rendered in. If CI says
SourceSans3 and the line text says "Source Code Pro, is beloved by
programmers" — maybe CI is right.
