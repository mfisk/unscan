# Debugging Character Segmentation & Font Identification

How to diagnose and fix character segmentation failures and font
identification misses in unprint.

**See also:** `docs/char-index-methodology.md` (CI feature vectors).

---

## Diagnostic Tools

### `--audit <DIR>` — Unified Diagnostic Output

The single flag for all diagnostic data.  Produces:

1. `audit.json` — pipeline decisions (per-line font matches, CI votes,
   SSIM verification scores)
2. Per-line diag-seg directories with per-word segmentation diagnostics
   (word crops, seam overlays, summary.json, character crops)
3. Per-line `crops/` directories (the exact character images CI scores)

```bash
./target/release/unprint INPUT.pdf -o /dev/null --audit /tmp/audit-out
```

There is **no separate `--diag-seg` flag** — `--audit` enables everything.

**Output tree:**

```
/tmp/audit-out/
  audit.json                              # full pipeline audit
  crops/                                  # per-line character crops (CI inputs)
    crop_00_T.png
    crop_01_y.png
    ...
  p1_L000_Typography/                     # line-level diag-seg directory
    word_000_Typography/                  # per-word directory
      seg_plain/                          # plain segmentation path
        word_crop.png                     # raw word image from Tesseract bbox
        ws_overlay.png                    # whitespace pass: split points
        seam_overlay.png                  # whitespace (red) + seam (blue) splits overlaid
        final_overlay.png                 # all passes: whitespace red, seam blue, charbox green
        summary.json                      # machine-readable split data (see below)
        chars/                            # per-character crops
          00_T.png
          00_T_ref.png                    # reference render (if --diag-ref-font used)
          01_y.png
          ...
      seg_lig/                            # ligature segmentation path (if applicable)
        ...
```

**Top-level `audit.json`** (`AuditLog`):

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
| `ssim_score` | f32? | SSIM verification score (Pass 2 gate, threshold 0.3) |
| `decision` | "vectorized" \| "kept_raster" | — |
| `reason` | string | Why this decision was made |
| `bbox` | {x, y, width, height} | Pixel bounding box at render DPI |
| `ci_candidates` | CiCandidate[] | CI candidate fonts with scores |
| `ci_char_votes` | CharCiVote[] | Per-character CI voting detail |
| `seg_winner` | string? | Which segmentation path won: `"plain"` or `"ligature"` |
| `ci_candidates_lig` | CiCandidate[]? | CI candidates from the alternate (non-winning) path |
| `ci_char_votes_lig` | CharCiVote[]? | Per-character CI votes from the alternate path |
| `words` | WordAudit[] | Per-word detail (historical; word SSIM reranking is now disabled) |
| `word_rerank_winner` | string? | Font chosen by word-level SSIM |
| `word_bboxes` | WordBBox[] | Post-processed word bounding boxes (after clip/drop/expand) |
| `word_bboxes_raw` | WordBBox[] | Raw Tesseract word bounding boxes (before post-processing) |

**Per-character CI vote** (`CharCiVote`):

| Field | Type | Description |
|-------|------|-------------|
| `ch` | char | Character label from OCR |
| `crop_index` | usize | Index into the line's crop array |
| `min_dist_sq` | f32 | Squared distance to nearest indexed glyph |
| `passed_gate` | bool | Whether this char passed the quality gate |
| `nearest` | [string, f32][] | Top-N nearest font matches with distances |
| `crop_path` | string? | Path to the crop image (when audit images enabled) |

**Per-word detail** (`WordAudit`) — *Note: word-level SSIM reranking is
disabled; CI #1 wins directly. These fields may be empty in current output:*

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

### Per-word `summary.json`

Located at `<audit_dir>/p<N>_L<NNN>_<text>/word_<NNN>_<text>/seg_plain/summary.json`:

```json
{
  "word_text": "Typography",
  "image_w": 492,
  "image_h": 89,
  "n_chars_expected": 10,
  "n_segments_produced": 10,
  "vp_splits": [52, 101, 153, 202, 253, 294, 339, 390, 444],
  "seam_splits": [],
  "seam_paths": {},
  "final_boundaries": [0, 52, 101, 153, 202, 253, 294, 339, 390, 444, 492],
  "mismatch": false
}
```

The `seam_paths` field maps split column → full path (one x value per row
of the word image).  Whitespace splits are straight vertical lines and do
not appear in `seam_paths`.

---

### `--diag-ref-font <FONT>` (requires `--audit`)

Renders each character from a reference font file using the index-time
normalization path, saved as `NN_c_ref.png` alongside the scan crop
`NN_c.png`.  Useful for side-by-side comparison of identical normalization.

---

### Accuracy Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `tools/verify-accuracy.py` | Span-level accuracy vs vector PDF ground truth | `python3 tools/verify-accuracy.py <audit_dir>/audit.json VECTOR.pdf [--verbose]` |
| Built-in report (`report.rs`) | Visual HTML miss report with inline crop images, scan line overlays, and per-char distance tables | `unprint RASTER.pdf --audit DIR --test VECTOR.pdf` → `DIR/report.html` |

**After any accuracy test that is not 100%:** run `tools/verify-accuracy.py`
**After any accuracy test that is not 100%:** examine the miss report.
Understand every miss before proposing fixes.
---

### Other Diagnostic Flags

| Flag | Description |
|------|-------------|
| `--include-font <NAME>` | Force a font (case-insensitive substring) into CI candidate list for all lines, even if CI would normally prune it. Score shows as -999 in audit. |
| `--thoroughness <FLOAT>` | Scale CI thresholds (default 1.0). Higher = more candidates survive CI, slower. Useful for testing if a correct font is being over-pruned. |
| `--compare` | Generate side-by-side scan/render overlay images in `<output>-compare/` |
| `--overlay` | Debug render mode: original raster kept, vector text overlaid in semitransparent red |
| `RUST_LOG=info` | Per-line CI summary on stderr: crop count, gate pass/fail, σ cutoff stats |

---

## Segmentation Architecture

Segmentation lives in `src/segment.rs`, functions `segment_characters()`
and `segment_characters_inner()`.

Character extraction (cropping words from the page, calling segmentation,
producing normalized character images for CI) is in `src/segment.rs`,
function `segment_characters()`.

### Input

- `img`: word crop image (grayscale, from Tesseract's word bbox)
- `n_chars`: target character count (from OCR text length)

### Column Ink Profile

For each column x, compute `col_ink[x]` = count of pixels where
pixel < 200 (ink threshold).  This is a pixel-count measure: each dark
pixel contributes 1, regardless of how dark.

Key derived values:
- `col_has_ink[x]`: true if `col_ink[x] > 0`

### Pass 1: Whitespace Splitting

Scan across the word's ink extent for contiguous runs of zero-ink columns.
Split at each run's midpoint.  This handles the easy cases — obvious
whitespace gaps between letters — so the DP only has to carve through ink.

### Pass 2: Greedy Seam Carving

For remaining splits, seam carving finds the cheapest vertical path through
ink.

**Energy function:** each pixel's base cost is its darkness
(`255.0 - pixel_value`).  White pixels are zero cost; black pixels are 255.
The DP adds a **delta penalty** when the path moves into a darker pixel:
`delta_weight (4.0) × max(0, dark_cur - dark_prev)`.  This encodes
"stay in whitespace, don't wander into ink" — a path through a uniformly
dark stroke interior pays base cost but no delta penalty, while a path
crossing from a white gap into a stroke edge pays heavily.

**Dual DP (forward + reverse)** computes cost matrices from both top and
bottom, meeting at the midpoint row.  Movement is vertical and horizontal
(left-to-right + right-to-left chaining within each row).  For each interior
column at the midpoint row, the combined cost
`cost_fwd[mid][c] + cost_rev[mid][c] - ink_score(mid, c)` gives a candidate
seam score.  A post-hoc **width penalty** (`50 × path_width`) discourages
paths that wander far horizontally.  Runs of consecutive equal-cost
candidates are collapsed to a single candidate at the midpoint.

**Greedy loop:** pop cheapest candidate, validate ink on both sides
(min_ink_for_symbol), accept the split, compute candidates for the two child
segments, repeat.

### normalize_to_ink_bounds

After segmentation, each character cell is cropped and normalized:
1. Find bounding box of all pixels < 200 (ink threshold)
2. Add 1px padding on all sides
3. Scale to `NORM_H` (48px) tall, preserving aspect ratio

This is the exact image CI scores.  Both index-time (`render_char_normalised`)
and scan-time (`normalize_to_ink_bounds`) use the same threshold and padding
so feature vectors are comparable.

---

## Ligature Path Selection

Some fonts use ligature glyphs — a single glyph replacing multi-character
sequences like "ff", "fi", "fl", "ffi", "ffl".  When OCR reads "affluent",
the glyph on the page may actually be a single "ffl" ligature rather than
three separate letters.  Segmenting for 8 characters ("a-f-f-l-u-e-n-t")
will fail because the word image only has 6 visual units
("a-[ffl]-u-e-n-t").

### Dual-path approach

`segment_characters()` in `src/segment.rs` handles this by segmenting
each word **twice** when ligature-eligible character sequences are present:

1. **Plain path** (`seg_plain/`): segment targeting `n_chars = len(all_chars)`
   — treats every character independently.
2. **Ligature path** (`seg_lig/`): collapse ligature sequences into single
   Unicode codepoints (e.g., `['f','f','l']` → `['\u{FB04}']`) via
   `collapse_ligature_chars()`, then segment targeting
   `n_chars = len(lig_chars)` (fewer segments).

Collapse uses greedy longest-first matching: "ffi"/"ffl" (3→1) before
"ff"/"fi"/"fl" (2→1).

### Winner selection

Both paths produce character crops.  Both are scored independently by
`search_candidates()` in CI.  The winner is chosen in `src/main.rs` by
comparing the top CI candidate score from each path:

```
plain_top = ci_result_plain.scores[0].score
lig_top   = ci_result_lig.scores[0].score
use_lig   = lig_top > plain_top
```

**Higher top score wins.**  If the font actually uses ligature glyphs,
ligature-path segmentation will produce cleaner crops that match the
index better, yielding a higher CI score.  If it doesn't (plain "ff"
rendered as two separate glyphs), plain segmentation wins because the
character boundaries align with actual glyph boundaries.

### What happens after

- `seg_winner` is recorded in the audit entry as `"plain"` or `"ligature"`
  (or `null` if no ligature-eligible sequences existed in the line).
- The winning path's crops are used for all downstream work: font matching,
  SSIM verification, and per-char distance computation.
- The losing path's CI results are stored in `ci_candidates_lig` /
  `ci_char_votes_lig` in the audit JSON for diagnostic comparison.
- The miss report shows `seg_winner` in the audit data but no longer
  renders separate segmentation pictures for each path — the scan-line
  overlay with word bounding boxes covers the same information.

### Ligature probes

The font index also detects ligature support at index time
(`detect_ligature_glyphs()` in `src/font_scan.rs`).  It shapes probe
strings like "ff", "fi", "fl", "ffi", "ffl" through HarfBuzz with
`liga` and `dlig` features enabled.  If shaping produces a single glyph
(i.e., the font's GSUB table fired a ligature substitution), that glyph
ID is recorded and the ligature codepoint (e.g., U+FB00 for "ff") is
added to the font's character map.  This means CI can score a ligature
crop against the actual ligature glyph in fonts that have one.

---

## OCR Post-Processing Pipeline

Between Tesseract output and font matching, several post-processing steps
modify the word bounding boxes:

1. `assemble_lines()` — groups word-level regions into lines
2. `merge_overlapping_lines()` — merges vertically overlapping lines
3. `clip_word_overlaps()` — clips horizontally overlapping word bboxes
4. `drop_outlier_words()` — removes words with height ≥ 1.8× median
   word height AND confidence < 70 (filters image artifacts)
5. `expand_bbox_to_ink()` — expands line bboxes to actual ink extent
6. `expand_words_to_ink()` — expands word bboxes to actual ink extent

The audit JSON stores both `word_bboxes_raw` (after step 1, before
post-processing) and `word_bboxes` (after all steps) so the miss report
can show what Tesseract saw vs what unprint used.

---

## Common Failure Modes

### 1. High-Contrast Fonts (Didone/Bodoni)

**Symptom:** Whitespace splitter puts a split inside a character instead of
between characters.

**Cause:** Fonts like Playfair Display have extreme stroke contrast — thick
vertical stems with razor-thin hairline serifs.  The hairlines have very low
`col_ink` values.  But since the whitespace splitter only splits at columns
with truly zero ink, this only happens when a hairline is thin enough to
have zero-ink columns — rare in practice.

**Fix:** The whitespace splitter only fires on runs of truly zero-ink
columns.  Non-zero ink columns, even very low ones, are left for the DP
seam carver.

### 2. Segment/Character Count Mismatch

**Symptom:** `summary.json` shows `n_segments_produced ≠ n_chars_expected`.
Character crops are labeled wrong.

**Cause:** Usually seam carving splitting too aggressively or not finding
enough valid splits with ink on both sides.

**Diagnosis:**
```bash
find /tmp/audit-out -name summary.json -exec \
  jq -r 'select(.mismatch) | "\(.word_text): \(.n_segments_produced) segs vs \(.n_chars_expected) expected"' {} \;
```

### 3. normalize_to_ink_bounds Clipping

**Symptom:** Character crop is narrower than expected — missing serifs
or thin strokes.

**Cause:** Pixels with value ≥ 200 are treated as background.  Anti-aliased
edges can have pixels in the 200–230 range that get clipped.

**Diagnosis:** Examine the raw word image and check pixel values in the
clipped region.

### 4. Seam Carving Through Ink

**Symptom:** A character is split by seam carving even though it's a
single glyph.

**Cause:** Seam carving finds a low-cost path through thin parts of a
glyph (e.g., the crossbar of 'e', the thin joint of 'k').

**Diagnosis:** Check `seam_overlay.png` in the word's diag-seg directory.
Blue lines show seam paths.  If a path goes through a glyph, seam carving
is cutting ink.

### 5. drop_outlier_words False Positives

**Symptom:** A word disappears from the line (present in `word_bboxes_raw`
but missing from `word_bboxes`).

**Cause:** `drop_outlier_words()` uses median word height as baseline.  If
the line contains a degenerate-height word (e.g., an em-dash at h=1) that
happens to be the median, the threshold becomes too aggressive.  Also
triggers when a legitimate word has low Tesseract confidence (< 70).

**Diagnosis:** Compare `word_bboxes_raw` vs `word_bboxes` in the audit
entry.  Check height and confidence of the dropped word.

---

## Debugging Workflow

### Step 1: Full Diagnostic Run

```bash
RUST_LOG=info \
  ./target/release/unprint INPUT.pdf \
    -o /dev/null \
    --audit /tmp/audit-out \
    2>&1 | tee /tmp/unprint.log
```

This produces everything: `audit.json`, character crops, segmentation
diagnostics, word crops, and overlay images.

### Step 2: Check Accuracy

```bash
python3 tools/verify-accuracy.py /tmp/audit-out/audit.json \
    VECTOR.pdf --misses-only
```

### Step 3: Visual Miss Report

The miss report is generated automatically when `--test` is set.
Open `DIR/report.html` in a browser.

### Step 4: Query the Audit JSON

```bash
# Find a specific line
jq '.text_entries[] | select(.text | contains("Typography"))' \
    /tmp/audit-out/audit.json

# Show CI candidates for that line
jq '.text_entries[] | select(.text | contains("Typography")) |
    .ci_candidates' /tmp/audit-out/audit.json

# Per-character votes
jq '.text_entries[] | select(.text | contains("Typography")) |
    .ci_char_votes[]' /tmp/audit-out/audit.json

# Characters that failed the quality gate
jq '.text_entries[] | select(.text | contains("Typography")) |
    .ci_char_votes[] | select(.passed_gate == false)' \
    /tmp/audit-out/audit.json

# Worst-scoring characters (highest distance)
jq '.text_entries[] | select(.text | contains("Typography")) |
    .ci_char_votes | sort_by(-.min_dist_sq) | .[0:3]' \
    /tmp/audit-out/audit.json

# Compare raw vs post-processed word bboxes (find dropped words)
jq '.text_entries[] | select(.text | contains("Font:")) |
    {raw: [.word_bboxes_raw[].text], final: [.word_bboxes[].text]}' \
    /tmp/audit-out/audit.json

# Word SSIM scores
jq '.text_entries[] | select(.text | contains("Typography")) |
    .words[].candidates | sort_by(-.ssim) | .[0:3]' \
    /tmp/audit-out/audit.json

# All fonts matched, with counts
jq '[.text_entries[] | .font_matched] | group_by(.) |
    map({font: .[0], count: length}) | sort_by(-.count)' \
    /tmp/audit-out/audit.json
```

### Step 5: Inspect Segmentation

```bash
# Find all words with segment mismatches
find /tmp/audit-out -name summary.json | xargs jq -r \
  'select(.mismatch) | "\(.word_text): \(.n_segments_produced) segs, expected \(.n_chars_expected)"'

# Show whitespace vs seam split counts per word
find /tmp/audit-out -name summary.json | xargs jq -r \
  '"\(.word_text): vp=\(.vp_splits | length) seam=\(.seam_splits | length) final=\(.final_boundaries | length - 1)/\(.n_chars_expected)"'

# Show segment widths for a specific word
jq '[.final_boundaries | to_entries | .[] | select(.key > 0)] |
    map(.value) as $b |
    [range(0; $b | length - 1)] | map("\($b[.] - $b[. - 1])px")' \
    /tmp/audit-out/p1_L000_Typography/word_000_Typography/seg_plain/summary.json
```

### Step 6: Check CI Scoring

If segmentation is correct but CI picks the wrong font:

```bash
# Which characters of the correct font have worst distances?
jq '.text_entries[] | select(.text | contains("Typography")) |
    .ci_char_votes[] | {ch, min_dist_sq, top_match: .nearest[0]}' \
    /tmp/audit-out/audit.json

# Is the correct font in the candidate list?
jq '.text_entries[] | select(.text | contains("Typography")) |
    .ci_candidates[] | select(.font_key | test("Playfair"; "i"))' \
    /tmp/audit-out/audit.json
```

If the correct font isn't in `ci_candidates`, it was pruned by CI's σ cutoff.
Try `--thoroughness 2.0` to relax all gates, or `--include-font Playfair` to
force it into CI candidate list.

---

## Key Source Locations

| What | File | Function / Line |
|------|------|-----------------|
| CLI args & flags | `src/cli.rs` | `struct Args` |
| Pipeline orchestration | `src/main.rs` | `fn run()` |
| SSIM fast path | `src/main.rs` | Parallel font matching section |
| Pass 1.5 paragraph grouping | `src/main.rs` | Paragraph-level font grouping section |
| Audit data structures | `src/audit.rs` | `AuditEntry`, `CharCiVote`, `WordAudit` |
| OCR + post-processing | `src/ocr.rs` | `extract_text_regions()`, `assemble_lines()`, `merge_overlapping_lines()`, `clip_word_overlaps()`, `drop_outlier_words()`, `expand_words_to_ink()` |
| Font-metric word splitting | `src/ocr.rs` | `split_wide_whitespace_words()` |
| Character extraction | `src/segment.rs` | `extract_line_chars()` |
| Segmentation algorithm | `src/segment.rs` | `segment_characters_inner()` |
| Whitespace splitting | `src/segment.rs` | Pass 1 in `segment_characters_inner()` |
| Seam carving | `src/segment.rs` | Pass 2 in `segment_characters_inner()` |
| normalize_to_ink_bounds | `src/features.rs` | `normalize_to_ink_bounds()` |
| Boundary→crop extraction | `src/segment.rs` | `extract_chars_from_boundaries()` |
| CI search + scoring | `src/font_match.rs` | `identify_font()` |
| Feature computation | `src/features.rs` | `compute_features()` |
| Font-metric gap functions | `src/ocr.rs` | `font_ink_width()`, `font_pair_ink_gap()` |
| Font cache (shared LRU) | `src/font_cache.rs` | `FontCache` |
| SSIM verification | `src/verify.rs` | `verify_text_region()` |
| Segmentation diag overlay | `src/seg_diag.rs` | `save_split_overlay()` |
| Span-level accuracy | `tools/verify-accuracy.py` | — |
| Line-level miss report | Built-in `report.rs` (via `--test`) | `DIR/report.html` |
| Test ground truth | `test-docs/font-timeline-specimen.pdf` | Vector PDF |
| Test raster (Poppler) | `test-docs/font-timeline-specimen-rasterized-poppler.pdf` | Cross-renderer test |
