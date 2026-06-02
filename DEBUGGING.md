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
1. Run `--audit` to get the actual char crops CI receives.
2. From the audit JSON `ci_char_votes`, identify which characters of the correct
   font score worst. Show the crop images for those characters. Narrow glyphs
   (I, l, t, i) are the canary — they're most sensitive to geometry mismatches.
3. Use only actual tool output (diag-seg crops, audit JSON distances). Do not
   reimplement feature computation or character rendering in Python — subtle
   differences in thresholds, rounding, and resize filters produce misleading
   comparisons. If you need index-side reference images, add a diagnostic dump
   to the Rust code.
4. Fix the worst character first — geometric mean amplifies outliers.
5. Only after per-char inspection rules out rasterization effects, check whether
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
  │ 4. Paragraph Grouping (DISABLED) │  Majority-vote body font replaces
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

**Algorithm (two-pass cascade — see SEGMENTATION.md):**

1. **Pass 1 — Vertical whitespace splits (VP).**
   Compute column ink profile (sum of dark pixels per column, threshold < 200).
   Find contiguous runs of zero-ink columns. Each run is a definite character
   boundary — split at the midpoint. If we find N−1 gaps for N characters,
   we're done. Both sides of each split must have `min_ink_for_symbol` ink
   (height-scaled: `(0.07 × h)² × 255`).

2. **Pass 2 — Greedy seam carving** (only for segments still containing
   multiple characters after Pass 1).

   Dual-DP seam carving finds the cheapest vertical paths through each
   segment. Energy is ink darkness (0 for white, 255 for black) with an
   entry penalty (`ENTRY_PENALTY_WEIGHT × darkness_increase`) when the
   path moves into darker pixels — directly encoding "stay in whitespace,
   don't wander into ink." All candidate seams go on a min-heap; the
   globally cheapest is accepted, child segments get diagonal masking from
   the accepted seam path, and new candidates are computed. Repeat until
   enough splits.

3. **Fallback:** If neither pass produces enough splits (truly touching
   characters with no separable seam), fall back to uniform boundaries.

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

### `--audit DIR` segmentation images

The `--audit DIR` flag dumps **character-level** crops that CI actually scores — the raw inputs to
the character index. These are the individual character images extracted from the scan,
normalized to `NORM_H` pixels tall.

Output: `DIR/p{page}_{text_slug}/word_{idx}_{text}/chars/NN_c.png`

```bash
unscan input.pdf -o /dev/null --audit /tmp/audit-out
ls /tmp/audit-out/p2_Times_New_Roman/word_000_1931/chars/
# 00_1.png  04_T.png  06_m.png ...
```

These are the actual images being turned into 99-dimensional feature vectors and
compared against the font index (see FEATURES.md for the full vector layout). If a character crop looks wrong (clipped,
merged with neighbor, includes artifacts), CI can't be expected to score it
correctly — fix the extraction first.

#### Diagnosing segmentation with crop dumps

When a line has too many or too few crops, segmentation is the culprit. Compare
crop count against expected character count (OCR text length minus spaces):

```bash
unscan input.pdf -o /dev/null
ls /tmp/unscan-crops/p4_line_ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklm/ | wc -l
# 99 crops for 52 expected chars = over-segmentation
```

Over-segmentation (more crops than chars) means the seam pass is splitting
inside glyphs. Each index in the dump gets two character labels (e.g.,
`crop_03_D.png` and `crop_03_E.png` at the same index), so the wrong glyph
image is being compared against the wrong indexed character.

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
uniform fallback fires.

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
cargo test --release --test t60_specimen_accuracy_aa

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
RUST_LOG=info \
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

### Step 1b: Show the Worst-Scoring Characters of the Correct Font

After confirming crops are visually clean, identify which characters of the
**correct font** score worst in CI. Use `--audit` output — never
reimplement feature computation in Python or another language. The Rust code
is the source of truth; any reimplementation will have subtle differences
(thresholds, rounding, padding, resize filter) that produce misleading
comparisons.

**Use actual tool output only:**

1. Run with `--audit` to get the per-char crop PNGs that CI actually scored
   (these are in `word_NNN_text/chars/NN_c.png`).
2. Use the audit JSON `ci_char_votes` to find which characters of the correct
   font have the worst (largest) distances. If the correct font doesn't appear
   in the top-3 nearest neighbors for a character, it's a CI recall failure for
   that character.
3. Show the scan crop images for the worst-scoring characters. Look at them —
   compare narrow glyphs (I, l, t, i) against wider ones (e, S, M) to see if
   narrow characters are systematically worse (rasterization blur has
   proportionally larger impact on narrow glyphs).

**Why narrow characters matter:** A 1px width difference on a 17px-wide `l` is
a ~6% aspect shift and drastically changes the horizontal profile bins. The
same 1px difference on a 50px-wide `M` is ~2% and barely registers. Narrow
characters are where index/scan geometry mismatches hit hardest — they're the
canary.

**Do not** write Python scripts to compute feature vectors or render reference
glyphs for comparison. If you need side-by-side scan-vs-index images, add a
diagnostic mode to the Rust code that dumps both. The scan crop and the index
render must come from the same binary using the same normalization code paths.

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

### Step 5: Check Paragraph Grouping (Stage 4 — currently disabled)

Paragraph grouping is currently disabled. When active, it overrides a line
match when a different font has majority vote among same-size lines in a
paragraph run. Check `RUST_LOG=debug` output for `paragraph regroup` messages.

This stage should be transparent on a specimen (each section has its own font,
no majority vote applies). If it's interfering, the bug is upstream — the
majority font shouldn't have won enough lines to trigger grouping.

---

## Procedure: "Show Me" a Miss

**When to run:** Immediately after any test run that does not achieve 100%
accuracy. Before analyzing, theorizing, or proposing fixes — produce the
visual card for every miss first. This is step 1, not step N.

When asked to "show" a miss, produce a **visual card** — not text tables, not
raw JSON, not narration. The card must contain actual rendered images so the
human can see what CI saw and compare it to what the correct answer looks like.

### Card layout

Each missed character is a **row** with three images side by side, followed by
a score line below them:

```
┌─────────────┬─────────────┬─────────────┐
│  Scan Crop  │  OCR Char   │ Correct Font│
│  (48px img) │  (48px img) │  (48px img) │
│  label:     │  label:     │  label:     │
│  "crop 28"  │  "OCR: 'f'" │  "SrcSerif4"│
└─────────────┴─────────────┴─────────────┘
  d²=0.1002   #1 NotoSansDisplay...  Bold 'Tofu' T
```

**All three columns are images rendered at the same height (NORM_H = 48px).**
Text labels go underneath each image. Scores and notes go on a separate line
below the image group. Do NOT put OCR values as text in a table cell next to
48px images — they become invisible.

### Three images per character

| Image | What it shows | Source |
|-------|--------------|--------|
| **Scan Crop** | The actual crop CI scored | `--audit` output PNGs |
| **OCR Char** | What Tesseract said, rendered at 48px in a neutral sans font (DejaVu Sans) | PIL render of the original OCR character |
| **Correct Font** | The same character rendered from the ground-truth font | PIL render from the system font file at NORM_H |

For OCR corrections, the label reads `OCR: 'f' → '0'`. For normal chars,
just `OCR: 'm'`.

### Score line (text below images)

- **d²** value, color-coded: green (< 0.01), orange (0.01–0.05), red (> 0.05)
- **CI nearest**: top 1-2 font names + distances
- **Note**: brief description of what's wrong (e.g., "Bold 'Tofu' T",
  "merged crop", "normal — for contrast")

### Which characters to show

Don't dump all 40 characters. Show:
1. The **worst-scoring characters** of the correct font (highest d²)
2. Any characters with **OCR corrections** (`.ocr_corrected_from` is set)
3. One or two **normal characters** for contrast (d² in the 0.003–0.007 range)

### Header metadata

The card header should show:
- Line identifier (page:line, OCR text)
- What was matched (wrong answer)
- What should have matched (correct answer, its CI rank + score)

### Rendering reference characters

Use PIL/FreeType to render characters at NORM_H (48px) from font files on disk.
This is purely for **visual comparison** — it is NOT used for feature computation
or distance measurement. The caveat from Step 1b still applies: never
reimplement feature vectors or scoring in Python.

- **OCR character**: render in a neutral sans font (DejaVu Sans) so it's
  visually distinct from both the scan crop and the correct font render.
- **Correct font character**: render in the ground-truth font file.

### Output format

Produce an HTML widget card with inline base64 images. The card must be
theme-aware (light/dark) and self-contained. Present it with `present_now: true`
so it renders immediately — don't embed it in a text reply.

### Script: `tools/char-misses.py`

Automates the full procedure — finds all real misses (ignoring metric-compatible
clones), picks interesting characters, renders all three columns, and produces
a self-contained HTML report.

```bash
# 1. Run unscan with --audit
./target/debug/unscan INPUT.pdf \
    -o /dev/null --audit /tmp/audit-out

# 2. Generate the visual report
python3 tools/char-misses.py /tmp/audit-out test-docs/font-timeline-specimen.pdf \
    -o /tmp/char-misses.html

# 3. Present as widget card (present_now: true)
```

The script mirrors the clone/alias map from `t60_specimen_accuracy_aa.rs` so its
miss count matches the test. Output is a single HTML file with all base64
images inlined.

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
| Clean raster specimen | `test-docs/font-timeline-specimen-scanned.pdf` |
| Skewed scan specimen | `test-docs/font-timeline-specimen-scanned.pdf` |
| Accuracy test | `tests/t60_specimen_accuracy_aa.rs` |

## Build and Run

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cargo/bin:$PATH"
cargo build --release
cargo test --release --test t60_specimen_accuracy_aa
```

Cached specimen run: ~45s. Uncached (index rebuild): ~3min.


## Finding CI Failures: Correct Font Not First

When investigating mismatches, the first question is: did the CI (character index)
put the correct font at rank #1 (or tied for #1)? If not, the bug is in CI scoring
(feature vectors, σ cutoff, segmentation quality). If CI had it right but the final
match is wrong, the bug is in word SSIM reranking.

### Step 1: Run with audit log

```bash
./target/release/unscan test-docs/font-timeline-specimen-scanned.pdf \
    -o /dev/null --audit /tmp/audit-out
```

### Step 2: Cross-reference CI top vs matched font

The audit log `text_entries[].ci_top` is a ranked list of `[font_path, score]` pairs.
`text_entries[].font_matched` is the final winner (CI #1).
`text_entries[].ssim_score` shows the SSIM verification score.

To find lines where CI #1 might be wrong, check low SSIM scores:

```python
import json

with open('/tmp/audit-out/audit.json') as f:
    entries = json.load(f)['text_entries']

for e in entries:
    ssim = e.get('ssim_score')
    if ssim is not None and ssim < 0.4:
        print(f"p{e['page']}:l{e['line_index']} ssim={ssim:.3f} font={e.get('font_matched', '?')}")
```

### Step 3: Use --audit for deep inspection

```bash
./target/release/unscan INPUT.pdf -o /dev/null --audit /tmp/audit-out
```

Produces per-line directories, each containing per-word subdirectories:

```
/tmp/audit-out/
  p1_The_quick_brown_fox/
    word_000_quick/
      word_crop.png          # raw word image from Tesseract bbox
      vp_overlay.png         # VP pass: cyan runs, red split midpoints
      seam_overlay.png       # VP (red) + seam (blue) overlaid
      final_overlay.png      # all passes: VP red, seam blue
      summary.json           # vp_splits, seam_splits, boundaries
      chars/                 # individual char crop PNGs: 00_A.png, 01_B.png, ...
    line_summary.json        # CI top 5, font_matched, ssim_score
```

### Diagnosis decision tree

1. **segmentation mismatch** (summary.json `n_segments_produced != n_chars_expected`):
   Check VP overlay — did VP find enough zero-ink runs? If not, the font has
   touching/overlapping characters. Check seam overlay — did seam carving produce
   reasonable splits?

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
