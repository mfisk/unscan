# Debugging Character Segmentation & Font Identification

How to diagnose and fix character segmentation failures and font
identification misses in unscan.

**See also:** `DEBUGGING.md` (pipeline-level font matching walkthrough),
`SEGMENTATION.md` (algorithm overview), `docs/char-index-methodology.md`
(CI feature vectors).

---

## Diagnostic Tools

### `--audit <DIR>` — Pipeline Audit

Produces an audit directory with `audit.json` (pipeline decisions) plus
per-line directories with per-word segmentation diagnostics. This is the
primary data source for diagnosing font identification failures.

```bash
./target/release/unscan INPUT.pdf -o /dev/null --audit /tmp/audit-out
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
| `seg_winner` | string? | Which segmentation path won: `"plain"` or `"ligature"` |
| `ci_candidates_lig` | CiCandidate[]? | CI candidates from the alternate (non-winning) path |
| `ci_char_votes_lig` | CharCiVote[]? | Per-character CI votes from the alternate path |
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
| `tools/verify-accuracy.py` | Span-level accuracy vs vector PDF ground truth | `python3 tools/verify-accuracy.py /tmp/audit-out/audit.json VECTOR.pdf [--verbose]` |
| `tools/char-misses.py` | Visual HTML report of line-level misses with inline crop images | `python3 tools/char-misses.py /tmp/audit-out/audit.json VECTOR.pdf --crops /tmp/unscan-crops -o /tmp/misses.html` |

**After any accuracy test that is not 100%:** run `tools/verify-accuracy.py`
with `--misses-only` first. Understand every miss before proposing fixes.

---

### Other Diagnostic Flags

| Flag | Description |
|------|-------------|
| `--include-font <NAME>` | Force a font (case-insensitive substring) into word SSIM reranking for all lines, even if CI pruned it. Score shows as -999 in audit. |
| `--include-fontmap <FILE>` | Inject all fonts from a fontmap JSON (`{"Name": "/path/to/font.ttf", ...}`) into CI audit candidate list. Like `--include-font` but in bulk. |
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

Find contiguous runs of zero-ink columns (interior only, not touching edges).
Split at each run's midpoint. Both sides must have at least
`min_ink_for_symbol` total column-ink or the split is rejected.  This
threshold scales with the word crop height: `(0.07 × h)² × 255`.

If this yields ≥ N-1 splits, pick the N-1 widest runs and stop.

### Pass 2: Greedy Seam Carving

If Pass 1 didn't find enough splits, seam carving takes over for the
remaining splits.

**Energy function:** each pixel's base cost is its darkness
(`255.0 - pixel_value`). White pixels are zero cost; black pixels are 255.
The DP adds an **entry penalty** when the path moves into a darker pixel:
`ENTRY_PENALTY_WEIGHT (4.0) × max(0, darkness_increase)`. This encodes
"stay in whitespace, don't wander into ink" — a path through a uniformly
dark stroke interior pays base cost but no entry penalty, while a path
crossing from a white gap into a stroke edge pays heavily.

**Dual DP (forward + reverse)** computes cost matrices from both top and
bottom. For each interior column at the midpoint row, the combined cost
gives a candidate seam score. Multiple candidates per segment are generated
and placed on a global min-heap.

**Greedy loop:** pop cheapest candidate, validate ink on both sides
(min_ink_for_symbol), accept the split, compute diagonal-masked candidates
for the two child segments, repeat.

See `SEGMENTATION.md` for the full algorithm description including diagonal
masking, midpoint tie-breaking, and straight-path preference.

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
./target/release/unscan test-docs/font-timeline-specimen-scanned.pdf \
    -o /dev/null --audit /tmp/audit-out

# Check accuracy
python3 tools/verify-accuracy.py /tmp/audit-out/audit.json \
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
jq '.text_entries[] | select(.text | contains("Typography"))' /tmp/audit-out/audit.json

# Show CI candidates for that line
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_candidates' /tmp/audit-out/audit.json

# Show per-character votes
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_char_votes[]' /tmp/audit-out/audit.json

# Find characters that failed the quality gate
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_char_votes[] | select(.passed_gate == false)' /tmp/audit-out/audit.json

# Find the worst-scoring characters (highest distance)
jq '.text_entries[] | select(.text | contains("Typography")) | .ci_char_votes | sort_by(-.min_dist_sq) | .[0:3]' /tmp/audit-out/audit.json

# Show word SSIM scores
jq '.text_entries[] | select(.text | contains("Typography")) | .words[].candidates | sort_by(-.ssim) | .[0:3]' /tmp/audit-out/audit.json

# Find all lines where CI and final match disagree
jq '.text_entries[] | select(.word_rerank_winner != null and .word_rerank_winner != .font_matched)' /tmp/audit-out/audit.json

# List all fonts matched, with counts
jq '[.text_entries[] | .font_matched] | group_by(.) | map({font: .[0], count: length}) | sort_by(-.count)' /tmp/audit-out/audit.json
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
  .ci_char_votes[] | {ch, min_dist_sq, top_match: .nearest[0]}' /tmp/audit-out/audit.json

# Is the correct font in the candidate list at all?
jq '.text_entries[] | select(.text | contains("Typography")) |
  .ci_candidates[] | select(.font_key | test("Playfair"; "i"))' /tmp/audit-out/audit.json
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
| Test raster input | `test-docs/font-timeline-specimen-scanned.pdf` | 6-page simulated scan |

---

## Quick Reference: Full Diagnostic Run

```bash
# Full diagnostic run: audit + crops
rm -rf /tmp/unscan-crops
UNSCAN_DUMP_CROPS=1 RUST_LOG=info \
  ./target/release/unscan test-docs/font-timeline-specimen-scanned.pdf \
    -o /dev/null \
    --audit /tmp/audit-out \
    2>&1 | tee /tmp/unscan.log

# Accuracy check
python3 tools/verify-accuracy.py /tmp/audit-out/audit.json \
    test-docs/font-timeline-specimen.pdf --misses-only

# Visual miss report
python3 tools/char-misses.py /tmp/audit-out/audit.json \
    test-docs/font-timeline-specimen.pdf \
    --fontmap test-docs/font-timeline-specimen-fontmap.json \
    -o ~/workspace/your_files/char-misses/index.html

# Find segmentation mismatches
find /tmp/audit-out -name summary.json | xargs jq -r \
  'select(.mismatch) | .word_text'

# Dump all CI failures (correct font not #1)
jq '.text_entries[] | select(.ci_candidates | length > 0) |
  {text, matched: .font_matched, ci_top: .ci_candidates[0].font_key,
   ci_score: .ci_candidates[0].score}' /tmp/audit-out/audit.json
```
