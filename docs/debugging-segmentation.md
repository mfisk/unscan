# Debugging Character Segmentation & Font Identification

How to diagnose and fix character segmentation failures and font
identification misses in unscan.

**See also:** `DEBUGGING.md` (pipeline-level font matching walkthrough),
`SEGMENTATION.md` (algorithm overview), `docs/char-index-methodology.md`
(CI feature vectors).

---

## Diagnostic Tools

### `--audit-log <path>` — Pipeline Audit JSON

Produces a single JSON file with every pipeline decision. This is the
primary data source for diagnosing font identification failures.

```bash
./target/release/unscan INPUT.pdf -o /dev/null --audit-log /tmp/audit.json
```

**Top-level structure** (`AuditLog`):

| Field | Type | Description |
|-------|------|-------------|
| `input_file` | string | Input path |
| `output_file` | string | Output path |
| `input_size_bytes` | u64 | — |
| `output_size_bytes` | u64 | — |
| `compression_ratio` | f64 | output / input |
| `images_dir` | string? | Path to audit image directory (crops/renders) |
| `pages` | PageSummary[] | Per-page stats |
| `text_entries` | AuditEntry[] | Per-line audit records |
| `geometry_entries` | GeometryEntry[] | Per-geometry-element records |

**Per-line record** (`AuditEntry`):

| Field | Type | Description |
|-------|------|-------------|
| `page` | usize | 1-indexed page number |
| `line_index` | usize | Line index within page |
| `text` | string | OCR text |
| `ocr_confidence` | f32 | Tesseract confidence (0–100) |
| `font_matched` | string? | Final font name after all stages |
| `font_confidence` | f32? | CI score for matched font |
| `ssim_score` | f32? | Word-level SSIM of matched font |
| `decision` | "vectorized" \| "kept_raster" | — |
| `reason` | string | Why this decision was made |
| `bbox` | {x, y, width, height} | Pixel bounding box at render DPI |
| `ci_candidates` | CiCandidate[] | CI candidate fonts with scores |
| `ci_char_votes` | CharCiVote[] | Per-character CI voting detail |
| `words` | WordAudit[] | Per-word SSIM reranking detail |
| `word_rerank_winner` | string? | Font chosen by word-level SSIM |

**Per-character CI vote** (`CharCiVote`):

| Field | Type | Description |
|-------|------|-------------|
| `ch` | char | Character label from OCR |
| `crop_index` | usize | Index into the line's crop array |
| `min_dist_sq` | f32 | Squared distance to nearest indexed glyph |
| `passed_gate` | bool | Whether this char passed the quality gate |
| `nearest` | [string, f32][] | Top-N nearest font matches with distances |
| `crop_path` | string? | Path to the crop image (when audit images enabled) |

**Per-word SSIM detail** (`WordAudit`):

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Word text |
| `bbox` | [u32; 4] | [x, y, width, height] pixel bbox |
| `crop_path` | string | Path to word crop image |
| `candidates` | WordCandidateAudit[] | Per-candidate SSIM scores |
| `winner` | string? | SSIM winner for this word |

**Per-candidate SSIM** (`WordCandidateAudit`):

| Field | Type | Description |
|-------|------|-------------|
| `font_key` | string | Font identifier |
| `ssim` | f32 | Structural similarity score |
| `dy` | i32 | Vertical alignment shift used |
| `render_path` | string | Path to rendered comparison image |

---

### `--diag-seg <DIR>` — Segmentation Diagnostics

Dumps the full segmentation pipeline state for every word. Use this to
understand exactly how VP/seam/charbox splits were chosen.

```bash
./target/release/unscan INPUT.pdf -o /dev/null --diag-seg /tmp/diag-seg
```

**Output tree:**

```
/tmp/diag-seg/
  p1_Typography/                      # line-level directory
    word_000_Typography/              # per-word directory
      word_crop.png                   # raw word image from Tesseract bbox
      vp_overlay.png                  # VP pass: cyan low-ink runs, red split points
      seam_overlay.png                # VP (red) + seam (blue) splits overlaid
      final_overlay.png               # all passes: VP red, seam blue, charbox green
      summary.json                    # machine-readable split data (see below)
      chars/                          # per-character crops (exact CI inputs)
        00_T.png                      # character 0: 'T' — normalized to NORM_H (48px)
        00_T_ref.png                  # reference render from --diag-ref-font (if used)
        01_y.png
        ...
    line_summary.json                 # CI results, font match, SSIM score
```

**`summary.json` fields:**

```json
{
  "word_text": "Typography",
  "image_w": 492,
  "image_h": 89,
  "n_chars_expected": 10,
  "n_segments_produced": 10,
  "vp_splits": [52, 101, 153, 202, 253, 294, 339, 390, 444],
  "seam_splits": [],
  "charbox_input_splits": [52, 100, 153, 202, 253, 294, 339, 390, 445],
  "charbox_added_splits": [],
  "final_boundaries": [0, 52, 101, 153, 202, 253, 294, 339, 390, 444, 492],
  "mismatch": false
}
```

**`line_summary.json` fields:**

```json
{
  "page": 1,
  "line_index": 0,
  "text": "Typography",
  "font_matched": "PlayfairDisplay-Regular",
  "font_score": 4.123,
  "ssim_score": 0.91,
  "word_rerank_winner": "PlayfairDisplay-Regular",
  "ci_top_5": [
    {"font": "PlayfairDisplay-Regular", "score": 4.123},
    {"font": "CourierPrime", "score": 3.987}
  ],
  "decision": "vectorized"
}
```

**`--diag-ref-font <FONT>`** (requires `--diag-seg`): also renders each
character from a reference font file using the index-time normalization
path, saved as `NN_c_ref.png` alongside the scan crop `NN_c.png`. Useful
for side-by-side comparison of identical normalization.

---

### `UNSCAN_DUMP_CROPS=1` — Character Crop Dump

Dumps every character crop that CI actually scores. Quick visual check
without the full `--diag-seg` overhead.

```bash
rm -rf /tmp/unscan-crops
UNSCAN_DUMP_CROPS=1 ./target/release/unscan INPUT.pdf -o /dev/null
```

**Output:**

```
/tmp/unscan-crops/
  word_Typography.png                     # full word image
  p1_L001_Typography/                     # per-line directory
    crop_00_T.png                         # char 0: normalized 48px-tall crop
    crop_01_y.png
    ...
```

The crop files are the **exact images** CI converts to feature vectors.
If a crop looks wrong, CI can't identify it correctly — fix segmentation
before investigating scoring.

---

### Accuracy Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `tools/verify-accuracy.py` | Span-level accuracy vs vector PDF ground truth | `python3 tools/verify-accuracy.py /tmp/audit.json VECTOR.pdf [--verbose]` |
| `tools/char-misses.py` | Visual HTML report of line-level misses with inline crop images | `python3 tools/char-misses.py /tmp/audit.json VECTOR.pdf --crops /tmp/unscan-crops -o /tmp/misses.html` |

**After any accuracy test that is not 100%:** run `tools/verify-accuracy.py`
with `--misses-only` first. Understand every miss before proposing fixes.

---

### Other Diagnostic Flags

| Flag | Description |
|------|-------------|
| `--include-font <NAME>` | Force a font (case-insensitive substring) into word SSIM reranking for all lines, even if CI pruned it. Score shows as -999 in audit. |
| `--thoroughness <FLOAT>` | Scale CI thresholds (default 1.0). Higher = more candidates survive CI, slower. Useful for testing if a correct font is being over-pruned. |
| `--compare` | Generate side-by-side scan/render overlay images in `<output>-compare/` |
| `--overlay` | Debug render mode: original raster kept, vector text overlaid in semitransparent red |
| `RUST_LOG=info` | Per-line CI summary on stderr: crop count, gate pass/fail, σ cutoff stats |

---

## Segmentation Architecture

All segmentation lives in `src/char_index.rs`, function
`segment_characters_inner()` (~line 2076).

### Input

- `img`: word crop image (grayscale, from Tesseract's word bbox)
- `n_chars`: target character count (from OCR text length)
- `charbox_splits`: Tesseract's per-character boundary x-positions (unreliable, used only as fallback)

### Column Ink Profile

For each column x, compute `col_ink[x]` = Σ(255 − pixel) for all pixels
where pixel < 200 (ink threshold). This is a density-weighted ink measure:
solid black pixels contribute 255, gray pixels contribute less, pixels ≥ 200
contribute nothing.

Key derived values:
- `max_ink`: peak column ink across the word
- `col_has_ink[x]`: true if `col_ink[x] > 0`
- `col_has_ink_5[x]`: true if `col_ink[x] > max_ink / 17` (~5.9% of peak)

### Pass 1: Zero-Ink VP (Vertical Projection)

Iterative, ink-trimmed splitting at the deepest ink valley.

**Each iteration:**
1. For each segment, find the best run of consecutive zero-ink columns
   (interior only — not touching segment edges)
2. **Rank runs by minimum `col_ink` within the run** (lower = deeper
   valley = better split). Width breaks ties (wider preferred).
3. Split at the column with minimum ink in the winning run
4. After splitting, trim each child segment to its ink extent (columns
   where `col_has_ink` is true), so whitespace margins don't participate
   in future searches

**Why ink-trimmed?** Without trimming, adjacent zero-ink columns near a
split point appear in both children, causing near-duplicate splits (e.g.,
cols 173 and 174) that waste split slots.

**Repeat** until `n_chars - 1` splits found or no more zero-ink runs exist.

### Pass 2: ≤5% Ink VP

If Pass 1 didn't find enough splits, relax: a column counts as "whitespace"
if `col_ink[x] ≤ max_ink / 17` (~5.9% of peak ink).

Same valley-ranking logic: **minimum ink wins**, not widest run. This is
critical for high-contrast fonts (Didone/Bodoni) where hairline serifs
create wide low-ink regions within a single character. The actual inter-
character gap has even less ink than the hairline, so minimum-ink ranking
correctly prefers it.

**Example — PlayfairDisplay "Typography":**
The T's hairline serifs (cols 7–17) have col_ink ≈ 580–1049. The h-y gap
(cols 439–449) has col_ink ≈ 64–763. Both are ≤ 5% of peak. Both runs are
11 columns wide. But the h-y gap has min ink = 64 vs the T hairline's
min ink = 580, so the h-y gap wins — the correct split.

### Pass 3: Greedy Seam Carving

For remaining splits that VP couldn't find, use dynamic-programming seam
carving to find the cheapest vertical path through each under-split segment.

```
energy(r,c) = (255 - pixel) if pixel < 200, else 0
M(0, c) = energy(0, c)
M(r, c) = energy(r, c) + min(M(r-1, c-1), M(r-1, c), M(r-1, c+1))
```

All segments' cheapest seams go into a min-heap. Pop cheapest, split,
recompute children, repeat until enough splits.

### Pass 4: Charbox Fallback

After VP + seam, check for segments wider than `avg_char_width × 3`. For
those, inject Tesseract's charbox boundaries to split them. This handles
cases where VP and seam both fail (e.g., truly overlapping characters).

**Caution:** The charbox fallback adds splits without reducing the total
count. This can produce more segments than characters, causing a mismatch
between character labels and crop images (character N gets the wrong crop).
The `summary.json` `mismatch` field flags this.

### normalize_to_ink_bounds

After segmentation, each character cell is cropped and normalized:
1. Find bounding box of all pixels < 200 (ink threshold)
2. Add 1px padding on all sides
3. Scale to `NORM_H` (48px) tall, preserving aspect ratio

This is the exact image CI scores. Both index-time (`render_char_normalised`)
and scan-time (`normalize_to_ink_bounds`) use the same threshold and padding
so feature vectors are comparable.

---

## Common Failure Modes

### 1. High-Contrast Fonts (Didone/Bodoni)

**Symptom:** VP splits a character in half instead of splitting between
characters.

**Cause:** Fonts like Playfair Display have extreme stroke contrast — thick
vertical stems with razor-thin hairline serifs. The hairlines have very low
`col_ink` values (3–5 ink pixels per column vs 65 in the stem). When the
≤5% ink pass searches for split candidates, these hairline regions look
like inter-character gaps.

**Fix:** The valley-ranking algorithm ranks by minimum ink (deepest valley
wins), not by run width. Actual inter-character gaps have near-zero ink
even in the 5% pass, beating intra-character hairlines. If you're seeing
this failure, check that `best_low_ink_valley()` is being used (not an
older width-based ranking).

### 2. Segment/Character Count Mismatch

**Symptom:** `summary.json` shows `n_segments_produced ≠ n_chars_expected`.
Character crops are labeled wrong (crop_05_a.png actually shows 'b').

**Cause:** Usually the charbox fallback adding extra splits after VP+seam
already found enough. Can also be seam carving splitting too aggressively
in a segment that VP already handled.

**Diagnosis:**
```bash
# Find all mismatched words
find /tmp/diag-seg -name summary.json -exec \
  jq -r 'select(.mismatch) | "\(.word_text): \(.n_segments_produced) segs vs \(.n_chars_expected) expected"' {} \;
```

### 3. normalize_to_ink_bounds Clipping

**Symptom:** Character crop is narrower than expected — missing serifs
or thin strokes.

**Cause:** Pixels with value ≥ 200 are treated as background. Anti-aliased
edges on a screen render can have pixels in the 200–230 range that get
clipped. (Note: clean rasterized PDFs at 300 DPI typically use solid black,
so this mainly affects screen captures or low-DPI renders.)

**Diagnosis:** Examine the raw word image (`word_Typography.png` from
`UNSCAN_DUMP_CROPS`) and check pixel values in the clipped region:
```python
from PIL import Image
import numpy as np
word = np.array(Image.open("/tmp/unscan-crops/word_Typography.png"))
# Check a specific column
col = 51
for y in range(word.shape[0]):
    px = int(word[y, col])
    if px < 255:
        print(f"  row {y}: px={px} {'INK' if px < 200 else 'clipped'}")
```

### 4. Seam Carving Through Ink

**Symptom:** A character is split by seam carving even though it's a
single glyph. Crop shows half a character.

**Cause:** Seam carving finds a low-cost path through thin parts of a
glyph (e.g., the crossbar of 'e', the thin joint of 'k'). This happens
when VP didn't find enough splits, forcing seam carving to split a
single-character segment.

**Diagnosis:** Check `seam_overlay.png` — blue lines show seam split
points. If a blue line goes through a glyph, seam carving is cutting ink.
Compare against `vp_overlay.png` to see if VP missed a genuine gap.

---

## Debugging Workflow

### Step 1: Identify the Miss

```bash
# Run with audit
./target/release/unscan test-docs/font-timeline-specimen-rasterized.pdf \
    -o /dev/null --audit-log /tmp/audit.json

# Check accuracy
python3 tools/verify-accuracy.py /tmp/audit.json \
    test-docs/font-timeline-specimen.pdf --misses-only
```

### Step 2: Look at the Crops

```bash
rm -rf /tmp/unscan-crops
UNSCAN_DUMP_CROPS=1 ./target/release/unscan INPUT.pdf -o /dev/null
```

Open the crop PNGs for the failing line. Do they look right? Is each crop
a single, complete character? If not, the problem is segmentation — fix
that before investigating CI scoring.

### Step 3: Query the Audit JSON

```bash
# Find a specific line
jq '.text_entries[] | select(.text | contains("Typography"))' /tmp/audit.json

# Show CI candidates for that line
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_candidates' /tmp/audit.json

# Show per-character votes
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_char_votes[]' /tmp/audit.json

# Find characters that failed the quality gate
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_char_votes[] | select(.passed_gate == false)' /tmp/audit.json

# Find the worst-scoring characters (highest distance)
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_char_votes | sort_by(-.min_dist_sq) | .[0:3]' /tmp/audit.json

# Show word SSIM scores
jq '.text_entries[] | select(.text | contains("Typography")) | .words[].candidates | sort_by(-.ssim) | .[0:3]' /tmp/audit.json

# Find all lines where CI and final match disagree
jq '.text_entries[] | select(.word_rerank_winner != null and .word_rerank_winner != .font_matched)' /tmp/audit.json

# List all fonts matched, with counts
jq '[.text_entries[] | .font_matched] | group_by(.) | map({font: .[0], count: length}) | sort_by(-.count)' /tmp/audit.json
```

### Step 4: Inspect Segmentation

```bash
./target/release/unscan INPUT.pdf -o /dev/null --diag-seg /tmp/diag-seg
```

```bash
# Find all words with segment mismatches
find /tmp/diag-seg -name summary.json | xargs jq -r \
  'select(.mismatch) | "\(.word_text): \(.n_segments_produced) segs, expected \(.n_chars_expected)"'

# Show VP vs seam vs charbox split counts per word
find /tmp/diag-seg -name summary.json | xargs jq -r \
  '"\(.word_text): vp=\(.vp_splits | length) seam=\(.seam_splits | length) cb=\(.charbox_added_splits | length) final=\(.final_boundaries | length - 1)/\(.n_chars_expected)"'

# Find words where charbox fallback fired
find /tmp/diag-seg -name summary.json | xargs jq -r \
  'select(.charbox_added_splits | length > 0) | "\(.word_text): charbox added \(.charbox_added_splits)"'

# Show segment widths for a specific word
jq '[.final_boundaries | to_entries | .[] | select(.key > 0)] | map(.value) as $b |
  [range(0; $b | length - 1)] | map("\($b[.] - $b[. - 1])px")' \
  /tmp/diag-seg/p1_Typography/word_000_Typography/summary.json
```

### Step 5: Examine Column Ink Profile

When you need to understand why VP made a specific split decision, examine
the raw ink values:

```python
from PIL import Image
import numpy as np

word = np.array(Image.open("/tmp/unscan-crops/word_Typography.png"))
h, w = word.shape

# Compute col_ink as the Rust code does
col_ink = []
for x in range(w):
    s = sum(255 - int(word[y, x]) for y in range(h) if word[y, x] < 200)
    col_ink.append(s)

max_ink = max(col_ink)
cutoff_5pct = max_ink // 17  # ~5.9% of peak

print(f"max_ink={max_ink}, 5% cutoff={cutoff_5pct}")
for x in range(w):
    flag = ""
    if col_ink[x] == 0:
        flag = " *** ZERO INK"
    elif col_ink[x] <= cutoff_5pct:
        flag = " ≤5%"
    print(f"  col {x:3d}: ink={col_ink[x]:6d}{flag}")
```

### Step 6: Check CI Scoring

If segmentation is correct but CI picks the wrong font:

```bash
# Which characters of the correct font have worst distances?
jq '.text_entries[] | select(.text | contains("Typography")) |
  .ci_char_votes[] | {ch, min_dist_sq, top_match: .nearest[0]}' /tmp/audit.json

# Is the correct font in the candidate list at all?
jq '.text_entries[] | select(.text | contains("Typography")) |
  .ci_candidates[] | select(.font_key | test("Playfair"; "i"))' /tmp/audit.json
```

If the correct font isn't in `ci_candidates`, it was pruned by CI's σ cutoff.
Try `--thoroughness 2.0` to relax all gates, or `--include-font Playfair` to
force it into word SSIM reranking.

---

## Key Source Locations

| What | File | Function / Line |
|------|------|-----------------|
| CLI args & flags | `src/cli.rs` | `struct Args` |
| Pipeline orchestration | `src/main.rs` | ~L445–650 |
| Audit data structures | `src/audit.rs` | `AuditEntry`, `CharCiVote`, `WordAudit` |
| Character extraction | `src/char_index.rs` | `extract_line_chars()` ~L1733 |
| Segmentation algorithm | `src/char_index.rs` | `segment_characters_inner()` ~L2076 |
| Valley-finding (VP) | `src/char_index.rs` | `best_low_ink_valley()` ~L2156 |
| Charbox fallback | `src/char_index.rs` | Pass 4 in `segment_characters_inner()` ~L2375 |
| Seam carving | `src/char_index.rs` | Pass 3 in `segment_characters_inner()` ~L2258 |
| normalize_to_ink_bounds | `src/char_index.rs` | `normalize_to_ink_bounds()` ~L1672 |
| Boundary→crop extraction | `src/char_index.rs` | `extract_chars_from_boundaries()` ~L1987 |
| CI search + scoring | `src/char_index.rs` | `search_candidates()` |
| Feature computation | `src/char_index.rs` | `compute_features()` |
| Word SSIM reranking | `src/word_match.rs` | `word_level_rerank()` |
| Segmentation diag overlay | `src/seg_diag.rs` | `save_split_overlay()` |
| Span-level accuracy | `tools/verify-accuracy.py` | — |
| Line-level miss report | `tools/char-misses.py` | — |
| Test ground truth | `test-docs/font-timeline-specimen.pdf` | Vector PDF |
| Test raster input | `test-docs/font-timeline-specimen-rasterized.pdf` | 300 DPI raster |

---

## Quick Reference: Full Diagnostic Run

```bash
# Full diagnostic run: audit + crops + diag-seg
rm -rf /tmp/unscan-crops /tmp/diag-seg
UNSCAN_DUMP_CROPS=1 RUST_LOG=info \
  ./target/release/unscan test-docs/font-timeline-specimen-rasterized.pdf \
    -o /dev/null \
    --audit-log /tmp/audit.json \
    --diag-seg /tmp/diag-seg \
    2>&1 | tee /tmp/unscan.log

# Accuracy check
python3 tools/verify-accuracy.py /tmp/audit.json \
    test-docs/font-timeline-specimen.pdf --misses-only

# Visual miss report
python3 tools/char-misses.py /tmp/audit.json \
    test-docs/font-timeline-specimen.pdf \
    --crops /tmp/unscan-crops \
    -o ~/workspace/your_files/char-misses/index.html

# Find segmentation mismatches
find /tmp/diag-seg -name summary.json | xargs jq -r \
  'select(.mismatch) | .word_text'

# Dump all CI failures (correct font not #1)
jq '.text_entries[] | select(.ci_candidates | length > 0) |
  {text, matched: .font_matched, ci_top: .ci_candidates[0].font_key,
   ci_score: .ci_candidates[0].score}' /tmp/audit.json
```
