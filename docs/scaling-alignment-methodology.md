# Scaling & Alignment Methodology

How unscan scales and positions a candidate font to match an input raster line,
assessed for correctness, robustness, and efficiency.

---

## 1. Width-Matched Scaling

### What happens

`layout::width_matched_em_px(font, text, target_width_px)` computes a font size
(em-height in pixels) such that the string's total advance width exactly equals
the OCR bounding-box width.

**Algorithm:**

1. Render the string at a reference em-height of 100 px.
2. Walk every character, summing `h_advance(gid)` and inter-glyph `kern(prev, gid)`.
3. Compute `em_px = 100 × (target_width / advance_at_100)`.
4. Clamp to `[4, 500]`.

**Note (June 2026 update):** The word-level SSIM reranking described in this
document (`word_match.rs`) has been removed from the pipeline. The CI #1
candidate now wins directly, and the only SSIM stage is the verification gate
(Pass 2, threshold 0.3) plus the fast-path dominant-font check (threshold 0.90).
References to "SSIM reranking" below are historical. The font-size determination
now uses height-matching for both SSIM verification and PDF output.

- **SSIM renderer** (`verify.rs → render_words_height_scaled`) — per-word: each
  word's advance is forced to equal its OCR bbox width.
- **PDF output** (`pdf_out.rs`) — per-word: same formula, producing `word_pt`
  for the `Tf` operator.

### Why width-matched (history)

Versions 1–4 used height-matched scaling (font size = OCR bbox height).  That
hid width mismatches: a wrong font could be stretched or compressed and still
score well on SSIM because the horizontal dimension was whatever the font
naturally produced.  Width-matched scaling inverts this — width is locked, so a
font with wrong proportions ends up with a wrong *height*, and the SSIM canvas
(sized to OCR bbox height) sees ink that overflows or underflows vertically.
Wrong aspect ratio → worse SSIM.

### Assessment

**Correct?** Mostly.  The key assumption is that advance width scales linearly
with em-height (i.e., doubling em doubles advance).  In `ab_glyph`, advance is
a simple scale of the design-space width, so linearity holds.  However:

- **Hinting:** TrueType hinting snaps glyph widths to pixel boundaries at small
  sizes.  `ab_glyph` does *not* apply hinting (it uses unhinted outlines), so
  there is no hinting-induced non-linearity.  This is actually an advantage for
  the matching use-case.
- **Kerning variation:** Some OpenType `kern` or `GPOS` tables have size-specific
  entries.  `ab_glyph` uses only the `kern` table (not `GPOS`), and reports a
  single kern value independent of scale.  So kern is treated as linearly
  scaling, which is correct for the `kern` table but may miss GPOS adjustments.

**Robust?**

| Edge case | Behaviour | Problem? |
|---|---|---|
| 1-char word ("a", "—") | advance is one glyph width | Fine, but noisy — tiny bbox ÷ tiny advance amplifies measurement error |
| All-caps words | Works; caps have wider advances | OK |
| Ligatures (fi, fl) | `ab_glyph` does not substitute ligatures | Renders f+i as separate glyphs; advance may be wider than the ligature in the scan. Width mismatch tanks SSIM — a penalty for fonts that *should* match. |
| Condensed/extended variants | Width-match picks a very large or very small em | Can produce em > 500 or < 4, clamped. Clamped values are wrong → bad SSIM → line falls to raster. Correct outcome (we don't have the right variant). |
| CJK / RTL | Untested | advance walk assumes LTR; RTL would need bidi reordering. CJK is filtered out in pre-match. |
| Em dash (—) | Single-glyph advance | The scan's em dash may be a different width than the candidate font's. Per-word matching confines the error to one word rather than polluting the whole line. |

**Efficient?** Yes.  The advance walk is O(n) in character count.  It is run
once per word per candidate during SSIM rendering, and once per word during PDF
output.  The only waste is the reference-scale pass: we render at 100 px to
measure, then re-render at the target scale.  A minor optimisation would be to
just compute `advance_at_1px` and scale directly, avoiding the clamped reference
pass, but the runtime cost is negligible.

**Flaw — per-word vs per-line inconsistency:**  The SSIM renderer draws
word-by-word on a *line-height* canvas, each word with its own `em_px`.  The
SSIM comparison sees the line as a whole.  But if one word's width-matched em is
15% different from its neighbours (e.g., a short word "I" next to "the"), the
vertical ink heights will differ word-to-word in the rendered image.  The scan
doesn't have that height variation — every character in a line is the same font
size.  This probably has minimal impact on SSIM because the windows are local
(11×11), but it creates a visual texture that doesn't exist in the scan.

**Suggested improvement:** After computing per-word `em_px` values, take the
median across the line and use that as the uniform em for all words in the
*SSIM render*.  The per-word values remain useful for PDF output positioning
(that's what `smooth.rs` does), but the SSIM comparison should arguably see a
uniform size.  This is partially addressed by `smooth.rs`, but `smooth.rs` only
kicks in at the PDF stage and only when `--smooth` is passed — the SSIM render
path doesn't use it.

---

## 2. Baseline / Vertical Positioning

### What happens

Three layers operate here:

**A) SSIM canvas (`verify.rs`):**
`layout::ink_centered_baseline_px(font, em_px, canvas_h)` returns a baseline Y
(in pixel-down coords) that centers the font's ink extent `(ascent - descent)` in
a canvas of height `canvas_h` (= OCR bbox height).

```
baseline = (canvas_h - ink_h) / 2 + ascent
```

When `ink_h > canvas_h`, the centering pushes `(canvas_h - ink_h)/2` negative,
meaning the ink extends above and below the canvas.  Pixels outside the canvas
are clipped.

**B) Vertical shift search (`verify.rs`):**
`ssim_windowed_best_vshift(scan_crop, rendered, 12)` tries 25 shifts
(dy = -12..+12 px), searched center-outward (0, -1, 1, -2, 2, …).  For each,
the rendered image is offset by `dy` (via coordinate arithmetic in the SSIM
kernel, not by creating shifted copies) and SSIM is computed.  Early exit at
SSIM ≥ 0.92.  Returns `(best_ssim, best_dy)`.

An optional `bail_below: Option<f32>` parameter is threaded through from
`verify_text_region()` to `ssim_windowed()`. When set (e.g., by the fast-path
SSIM check), the SSIM computation bails early if the running average drops
below the threshold after processing ≥8 windows per row — avoiding full
evaluation of obviously bad matches.

**C) PDF output (`pdf_out.rs`):**
`layout::ink_centered_baseline_pt(font, em_px, bbox_h_px, dpi)` converts the
same centering to PDF points.  Then `best_dy` from the SSIM stage is applied:

```rust
let dy_pt = fm.best_dy as f32 * 72.0 / page.dpi as f32;
let pdf_y = page.px_to_pt_y(tr.y) - baseline_offset_pt - dy_pt;
```

### Assessment

**Is the ink-center model correct?**

No.  This is the biggest conceptual issue in the pipeline.

Real typography aligns on a **fixed baseline**.  The ascent goes up and the
descent goes down from that baseline.  The OCR bounding box from Tesseract
wraps the *ink*, so its top is roughly at the ascent and its bottom is roughly
at the descent — but not exactly, because Tesseract includes a small margin and
because characters without descenders (e.g., "THE QUICK") have a bbox that stops
at the baseline rather than extending to the descender line.

Ink-centering assumes the font's ink fills the OCR bbox symmetrically.  In
practice:

- A **line with no descenders** (e.g., "ABCDEFGHIJKLMNOPQRSTUVWXYZ") has a bbox
  that doesn't extend below the baseline.  The font's `descent` contributes to
  `ink_h` but the scan bbox doesn't include that space.  So `ink_h > bbox_h`
  and centering pushes the text up.  This is exactly the Bodoni title bug.

- A **line with descenders** (e.g., "Giambattista Bodoni pushed the contrast")
  has a bbox that extends below the baseline.  Here `ink_h ≈ bbox_h` and
  centering works acceptably.

The vertical shift search (±12 px) compensates partially.  If the centering is
off by 3–4 pixels, the shift search finds the optimal offset and returns it as
`best_dy`.  But:

1. The search range is limited to ±12 px.  At 300 DPI, 12 px = 1.0 pt.  For
   large title text (~12–18 pt), the misalignment from ink-centering on an
   all-caps line can be 8–10 px, which is within range but leaves little
   margin.

2. The `best_dy` is per-line, but the ink-centering error depends on the
   *character content* of the line (descenders present or not).  A uniform
   `best_dy` applied to all words is correct only if every word has the same
   descender situation — which is the common case since they're on the same
   line.

**Suggested fix — baseline-aligned model:**

Instead of ink-centering, align on the **typographic baseline** directly:

1. From the OCR data, estimate where the baseline is within the bbox.  Tesseract
   reports `ascenders` / `descenders` per line in the hOCR output (though the
   current code only uses word-level bboxes).  Alternatively, heuristic: if all
   characters in the line are uppercase + no descenders, the baseline is at the
   bottom of the bbox.  If the line has descenders (g, j, p, q, y), the baseline
   is roughly `ascent / (ascent + |descent|)` of the way down from the top.

2. For the candidate font at `em_px`, compute the baseline position as
   `font.ascent()` below the top of the line:
   ```
   candidate_baseline_from_top = ascent_px
   ```

3. Align the candidate's baseline at the estimated scan baseline.  The remaining
   error is just the estimation noise, which the shift search can handle.

This would eliminate the systematic shift on all-caps or no-descender lines.

**Is the vertical shift search efficient?**

It is brute-force: up to 25 shifts (center-outward with early exit at ≥ 0.92).
Each SSIM evaluation walks every
ink-containing 11×11 window at step 4.  For a typical line of 400×30 pixels,
that's ~75 windows per SSIM call, so up to 25 × 75 = ~1,875 window evaluations
(typically fewer due to early exit).  At
the cost of ~242 multiply-adds per window (11×11×2 passes), this is about
~450K floating-point operations per line in the worst case.  Negligible compared
to the font index scan.

The `bail_below` parameter further speeds this up: when the SSIM running average
drops below the threshold after processing ≥8 windows per row, the evaluation
returns early without processing remaining rows. This is particularly useful for
the fast-path check, where most non-matching fonts bail within the first few
rows.

But a **phase-correlation** approach could find the optimal shift in a single
pass: compute the cross-power spectrum of the two images, take the inverse FFT,
and read off the peak.  For 400×30 images this would be ~12K pixels → ~12K log₂
operations for the FFT.  Faster in theory, but the implementation complexity
(FFT library, handling non-power-of-two dimensions) isn't worth it given the
negligible runtime of the brute-force approach.  **Leave as-is.**

**Sub-pixel shift?**  The current search is integer-pixel only.  At 300 DPI,
1 pixel = 0.085 mm.  Sub-pixel accuracy would improve SSIM marginally but
wouldn't change font *ranking* — the right font will still win by a margin
larger than the sub-pixel improvement.  Not worth the complexity.

**Horizontal shift?**  The pipeline does no horizontal shift search.  The
assumption is that Tesseract's word-level bboxes provide accurate horizontal
positioning, and the width-matched advance width places glyphs where they belong.
This is reasonable: horizontal alignment is primarily a function of advance
width accuracy, which is handled by the width-matching stage.  A horizontal
shift search would help only if Tesseract's x-coordinates are consistently
biased, which is not typically the case.

---

## 3. Deskew

### What happens

`detect_skew_from_words(words)` computes a linear regression of word-center Y
positions as a function of word-center X positions.  The slope is converted to
an angle via `atan()`, clamped to ±5°.  If `|angle| > 0.001 rad` (~0.06°), the
scan crop is rotated by `-angle` to straighten it before SSIM comparison.

The rotation uses bilinear interpolation with white (255) fill for out-of-bounds
pixels.

### Assessment

**Correct?** The regression model assumes the baseline is a straight line across
the page.  This is true for most scans (slight rotation of the page on the
scanner bed).  Non-linear warping (e.g., a book spine) is not handled but is
out of scope.

**Robustness issues:**

| Case | Problem |
|---|---|
| 2-word line | Regression through 2 points is perfect but meaningless — any noise in bbox centers maps to a skew angle. False deskew can hurt SSIM. |
| Line with mixed sizes (title + body) | Larger words have higher centers, biasing the slope. Not relevant for single-font lines. |
| Vertical text | Slope → ±90°, clamped to ±5°. Irrelevant for English docs. |

**Edge case with 2 words:** The code checks `centres.len() < 2` and returns 0.
With exactly 2 words, the regression is determined but noisy.  A threshold of 3
would be safer.

**Efficiency:** O(n) in word count.  Trivial cost.

---

## 4. Per-Word vs Per-Line Placement

### What happens

**SSIM stage (`verify.rs`):** Renders each word at its own width-matched `em_px`
on a shared canvas.  Word x-positions are relative to the line bbox
(`w.x.saturating_sub(line_x)`).  The canvas size is the full line bbox.

**PDF output (`pdf_out.rs`):** Each word is a separate `BT/Tf/Td/Tj/ET` sequence.
Word x-positions are absolute page coordinates (`word.x`).  Each word gets its
own `em_px` (or `smoothed_em_px`).

### The em-dash origin story

Previously, the PDF output rendered the whole line as a single `Tj` with one
line-level `em_px`.  If the candidate font's em dash was wider than the
original's, it consumed more than its share of the line's total advance width,
compressing all other characters.  Per-word placement eliminated this: the em
dash gets its own `em_px` from its own bbox width, and other words are
unaffected.

### Assessment

**x-coordinate systems diverge:** In `verify.rs`, words use `x_off = w.x - line_x`
(relative to line bbox).  In `pdf_out.rs`, words use `word.x` (absolute).  The
shared `layout.rs` doesn't touch x-coordinates at all, so there's no single
source of truth for horizontal placement.  This isn't currently causing bugs
because the coordinates are correct in each context, but it's a potential source
of drift if either system is modified.

**Per-word em inconsistency (SSIM vs scan):**  As noted in §1, the SSIM render
uses per-word `em_px` values that can differ by 15%+ between adjacent words.
The scan has uniform sizing.  This creates an SSIM penalty for the *correct*
font (because the render doesn't match pixel-for-pixel) that is absent for an
*incorrect* font that happens to have more uniform per-word em values.  In
practice this effect is small because SSIM windows are local, but it's a latent
accuracy ceiling.

---

## 5. Font-Size Smoothing

### What happens

`smooth.rs` groups consecutive `PlacedText` entries by font file path (same file
= same face, so regular, bold, and italic are separate groups).  For each group,
it collects all per-word `em_px` values, drops outliers more than 1 pt from the
mean, and assigns the median of survivors as `word.smoothed_em_px`.  The PDF
renderer uses `smoothed_em_px` when set.

### Assessment

**Correct?** The assumption is that all words in a run of the same font file
are the same point size.  This is true for body text within a paragraph but
breaks at size transitions (e.g., a drop cap, a footnote number, or an inline
heading).  The grouping is purely by adjacency + same font file, so a
paragraph followed by a footnote in the same font would be merged into one
group, and the footnote's smaller em values would be treated as outliers and
discarded.

**Threshold of 1 pt:** At 300 DPI, 1 pt ≈ 4.17 px.  OCR bbox noise is
typically 1–2 px, so a 1-pt threshold is generous enough to keep genuine
values and tight enough to exclude true size differences (footnotes are
typically 2+ pt smaller than body).  Reasonable.

**Median vs mean:** Median is robust to the remaining outliers after filtering.
Good choice.

**Applied only to PDF, not SSIM:** The smoothing doesn't improve the SSIM
*ranking* — that ship has sailed by the time smoothing runs.  It only improves
the visual quality of the output.  This is the right design: smoothing should
not influence which font wins.

**Missing: per-word baseline recalculation.**  When `smoothed_em_px` overrides
the natural `em_px`, the ink height changes.  The baseline centering formula
uses `em_px` to compute ink height and ascent.  If `smoothed_em_px` differs
significantly from the natural value, the baseline will be slightly off.  The
code does recalculate the baseline using `smoothed_em_px` (the renderer uses
the same `em_px` variable that was set from `smoothed_em_px`), so this is
correct — but only because the renderer checks `smoothed_em_px` *before*
computing the baseline.

---

## 6. SSIM-to-PDF Alignment Drift

### Current state

The whole point of `layout.rs` was to unify the arithmetic.  Here is where each
stage gets its values:

| Calculation | SSIM path | PDF path | Shared? |
|---|---|---|---|
| Width-matched em | `layout::width_matched_em_px` | `layout::width_matched_em_px` (or `smoothed_em_px`) | ✅ |
| Baseline (ink-centered) | `layout::ink_centered_baseline_px` | `layout::ink_centered_baseline_pt` | ✅ (same formula, different units) |
| Vertical shift | `ssim_windowed_best_vshift → best_dy` (±12 px, center-outward, early exit at ≥ 0.92) | Applied as `dy_pt` | ✅ (carried via `FontMatchResult`) |
| Horizontal position | `word.x_off` (relative) | `word.x` (absolute) | ❌ (different coordinate systems, correct in each context) |
| Deskew | Applied to scan crop before SSIM | Not applied to PDF output | ❌ |

**Deskew gap:** The SSIM stage straightens the scan before comparison, but the
PDF output does not apply a compensating rotation.  If the scan is skewed by
0.3°, the SSIM match finds the right font using a deskewed image, but the PDF
text is placed without rotation.  The resulting text follows the page grid rather
than the scan's slight tilt.  For small angles this is arguably *better* (you
want straight text in the output), but the vertical position (`best_dy`) was
optimised for the deskewed alignment.  On a skewed scan, the left edge of the
line might be 2 px higher than the right edge, so a single `best_dy` is a
compromise.

**Smoothed em not used in SSIM:** As noted, `smooth.rs` operates only on the
PDF stage.  If a future change tried to use smoothed sizes during SSIM (e.g.,
to re-verify), the font might not be loaded in the same context.

---

## 7. Summary of Recommended Improvements

### High priority

1. **Baseline-aligned model instead of ink-centered.**  Estimate the scan's
   baseline position from character content (descenders present → baseline is
   higher; all-caps → baseline at bottom of bbox).  Align the candidate's
   `font.ascent()` to the estimated baseline.  This eliminates the systematic
   vertical shift on title / all-caps lines that the current ±12 px brute-force
   search struggles to compensate.

2. ~~**Increase shift search range to ±12 px**~~ — **Done.** The shift range
   was increased from ±6 to ±12 px and center-outward search with early exit
   at SSIM ≥ 0.92 was added.

3. **Uniform em for SSIM render.**  After computing per-word `em_px`, take the
   median across the line and use it for all words in the SSIM rendering pass.
   Keep per-word em for PDF output (where it handles em-dash width differences),
   but the SSIM comparison should see a uniform size to match what the scan
   actually looks like.

### Medium priority

4. **Minimum 3 words for deskew.**  Currently triggers at 2 words, where the
   regression is noise-dominated.  Require 3+ word centers before fitting a
   slope.

5. **Ligature awareness.**  Before computing advance width, check if the font
   has GSUB ligature substitutions for common pairs (fi, fl, ff, ffi, ffl).  If
   the text contains such pairs and the font supports them, use the ligature
   glyph's advance instead of the individual glyphs' advances.  This would
   improve width accuracy for serif fonts that use ligatures.  `ab_glyph` does
   not support GSUB; this would require a library like `rustybuzz` or
   `swash` for shaping.

### Low priority / not worth it

6. **Phase-correlation shift search:** Theoretically faster, but the brute-force
   approach is cheap enough (~1ms per line) that the implementation complexity
   isn't justified.

7. **Sub-pixel shift interpolation:** Would improve SSIM by ~0.01–0.02.
   Wouldn't change ranking outcomes.

8. **Horizontal shift search:** Not needed; Tesseract's x-coordinates are
   sufficiently accurate and the width-matching handles horizontal alignment.

---

## 8. Overall Verdict

The scaling and alignment pipeline is **fundamentally sound** — width-matched
scaling is the right strategy, per-word placement is correct, and the shared
`layout.rs` module successfully prevents most arithmetic drift.

The main weakness is the **ink-centered baseline model**, which produces
systematic vertical misalignment on lines lacking descenders.  The vertical
shift search partially compensates but has too small a range.  Switching to a
baseline-aligned model (improvement #1) would fix this class of bug and is the
single highest-value change.

The pipeline is **efficient enough**.  No stage is a bottleneck — the dominant
cost is the 5048-font coarse scan, not the alignment arithmetic.  The per-char
index pre-filter (being wired in separately) will address that.
