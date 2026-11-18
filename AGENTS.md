# AGENTS.md — Unprint Project

## Universal Principle

**Correctness is the universal goal in everything.** Not a narrow PT Serif fix, not a sub-task — every change is judged by correctness first. Metrics measure correctness; they are not to be gamed with looser tolerances, sigma tweaks, or fuzzy matching. Fix input quality at the source. When geometry or naming is wrong, fix the geometry or naming — don't wallpaper over it.

## Context Window Hygiene

**Your context is finite. Protect it.**

- **Use subagents for builds and tests.** Never run `cargo build`, `cargo test`, or full `unprint` runs in the main thread. Spawn a subagent, let it report back a summary.
- **Never read diagnostic output in its entirety.** Test runs, cargo builds, and `unprint` stderr can produce thousands of lines. Reading all of it into an LLM context will blow your token budget. Always redirect to a file, then grep/tail for the specific lines you need.
- **Reduce output verbosity.** Pipe through `tail -n`, grep for specific patterns, or redirect to files and read only what matters. Never dump raw cargo output or full test logs into main context.
- **Build wrapper:** `cargo build --release 2>&1 | tail -3` at most. Only care about errors or "Finished".
- **Test wrapper:** Capture full output to a file, then grep/parse for the summary line (accuracy %, pass/fail). Only surface misses if debugging them.
- **Crop dumps:** Write to files, inspect specific ones by name. Never ls + read entire directories into context.

## Subagent Discipline

- One subagent per task: "build and run t60, report accuracy and misses"
- Subagent reports back: accuracy %, pass/fail, list of misses (if any), timing
- Main thread stays in orchestration mode — reads results, decides next step

## Development Rules (from Mike)

- **Always fix things right. We're not afraid of refactors. We avoid creating tech debt (Mike Nov 10, 2026).** When a fix needs a structural change (a trait in a shared crate, a required parameter instead of an optional closure, threading a dependency into a call path), do the refactor. Do not ship a minimal workaround that leaves a design wart (an opt-in `Option` closure, a silently-skipped code path) to avoid the refactor.
- **Always crop to ink bounds first.** Before segmenting, matching, or computing features on a word or character image, trim it down to just the ink. Raw word crops from Tesseract include adjacent lines, extra whitespace, and other garbage. Use `trim_whitespace()` or `normalize_to_ink_bounds()` before doing anything with the image.
- **Always ask before committing.** Never auto-commit.
- **Commit messages report the BAP score (Mike Nov 9, 2026).** Every commit message includes the BAP score for that commit (hits out of 505, ZNCC, p50, p90), e.g. `BAP: 451/505, ZNCC 0.9275, p50 0.9552, p90 0.9772`. A child branch or worktree starting from that commit reads the BAP score from the commit message and uses it as its baseline instead of recomputing a BAP. If the commit has no BAP score in its message, the child runs a BAP baseline before making changes.
- **Report ligatures (Mike Nov 6, 2026):** when showing characters in the report, always include ligature characters. If a character is part of a ligature (e.g. "fi" in a lig variant), the report must show the ligature glyph (U+FB01), not just the separate f and i glyphs.
- **Report lig-vs-nolig comparison (Mike Nov 6, 2026):** when the report compares a lig font against a nolig font, show both renderings side by side — the nolig side as the two separate characters (f, i) as two glyphs, the lig side as the one combined ligature glyph (U+FB01) as a single glyph — so the difference is visible. Each side renders according to its own font key's lig/nolig state.
- **Report lig/nolig shown in addition to top 6 (Mike Nov 6, 2026):** the lig-vs-nolig comparison display is shown IN ADDITION TO the top 6 deltas in the observation table, not instead of them. The B′ picker still shows the top 6 observations by |weight × (glyphD + GEO_WEIGHT × geoD)|, and the lig/nolig comparison rendering is an additional display alongside it.
- **"Stop vibing shit"** — deliberate changes only, no speculative fixes.
- **"Stop doing 1-off diagnostics and ONLY use the main code path."**
- **"Focus on input quality, not fancier algorithms."** Fix images first.
- **Show crops visually** — always present character crops as inline images.
- **Both index-build and index-lookup must use identical crop geometry.**

## Architecture

- `segment_characters()`: VP (zero-ink columns) first, dual-DP seam carving fallback
- `extract_line_chars()`: word-level extraction — VP + seam carving handles crop boundaries
- `normalize_to_ink_bounds()`: scan-time crop normalization to match index-time rendering

## Conjunction (h-geo overlap downweighting)

**The problem: overlapping / abutting serifs.** When two adjacent glyphs' ink boxes overlap — negative side bearings, or serifs that abut and interpenetrate — the VP pass finds no clean zero-ink column between them, so Pass 2's DP seam must cut through ink to separate them. The seam truncates both glyphs' ink bounds to the seam column, so the measured pitch (distance between the two midpoints) is shifted by the overlap. The error IS the overlap: it is a systematic measurement artifact of the segmentation, not a genuine spacing difference between the fonts. It can exceed the nominal shift: when the seam has serifs on one side and none on the other, the ink gradient pushes the seam toward the empty side, and a complete run-on leaves no clean gap at all. The shift is systematic, not uniform or random. Example: lob line 2 (LibreBodoni uppercase A–Z) has gaps from +0.04 px (U|V, just touching) to −2.41 px (X|Y, deep interpenetration); every conjunction seam crosses real ink, but antialiased edge ink (darkness 73–241), never a solid stroke at 255 — which is why a Max-ink gate fired on none of them.

**How we handle it.** The conjunction corrects the h error instead of discarding it. A pair is corrected when both hold:
- The ink gap between the two glyphs is below 0.8824 px. Derived from INK_THRESH=240: each edge's thresholded ink extends 0.5 − 15/255 = 0.4412 px beyond the outline, so two facing edges need a gap above 2 × 0.4412 px for a clean zero-ink column. Below that, the serifs overlap or antialiasing bridges them.
- The seam cuts ink. The gate is the seam's darkest pixel reaching 0.4 of the word's `max_ink` (the p95 darkness, ~255 for normal text). Do NOT gate on `dp_cost > 0`: dp_cost includes the horizontal-movement penalty, so a seam moving diagonally through white space has dp_cost > 0 with no ink (e.g. col 216 H|I dp_cost 233.0 but max darkness only 125). Gate history: Max ink (Mike Nov 8, "I specifically said Max ink") → 0.6 (Mike Nov 9) → 0.4 (Mike Nov 9, "Lower the gate to 0.4") → 0.3 (Mike Nov 10, directed the 0.3 gate) → 0.4 (Mike Nov 11, "go back to 0.4 gate"). The 0.3 gate was measured score-neutral (BAP 588 at 0.3 and BAP 542 at 0.4 both 450/505), so the gate value does not move the score.

**Correction (Mike's method, as revised Nov 9).** Subtract the FULL overlap width from the error magnitude:
- `overlap_px = 1.0 + max(−gap_px, 0)` is the overlap width (`overlap_px` in `src/geometry_classifier.rs`): 1.0 plus the raw outline overlap. A positive gap still yields 1.0 (the seam itself displaces the measurement by up to a pixel); a negative gap (real outline overlap) adds the overlap on top.
- Subtract `overlap_px` from `|h_err|`, clamp the corrected magnitude at 0 (so the correction cannot overshoot and flip the sign), keep the sign of the original h_err, and compute h_ll from the corrected h_err with the original sigma (SIGMA_PITCH_PX = 1/√6). No sigma widening. No halving: the shift is systematic, not uniform, so there is no expected-value factor of 1/2, and the error can exceed the nominal shift when ink is asymmetric — the full overlap is the correction. (Earlier versions used w/4, `likely_shift`, w/2, and `0.8824 − gap_px`; all superseded by Mike's Nov 9 directions: the error IS overlap, use the full overlap, do not cut in half for expected.)
- When a correction does fire it is magnitude-aware: a small error within the overlap width is forgiven (corrected to ~0); a large genuine mismatch keeps most of its penalty. E.g. L2 V: h_err −1.276, correction 1.000, corrected to −0.276; L2 Z: h_err −1.325, correction 1.489, corrected to 0.

**Why not binary zeroing.** The binary version set h_ll to 0.0. It caused the L53 regression: it removed Merriweather's genuine −2.29 px 'y' mismatch (weighted −25.45) along with GT's +0.46 px noise error (weighted −0.71), flipping a 22.29 GT lead into a 0.58 Merriweather win. Zeroing cannot distinguish overlap noise from a genuine spacing mismatch.

**Metrics (0.4 gate + full-overlap correction, verified lob 556 + BAP 542/588).** Lob 556: 11/13, ZNCC 0.9250, p50 0.9650, p90 0.9793. BAP 542 (0.4 gate): 450/505, ZNCC 0.9271. BAP 588 (0.3 gate, same tree): 450/505 (174 exact, 276 minor, 35 major, 16 sim failures), ZNCC 0.9274, p50 0.9559, p90 0.9766 — the gate change is score-neutral. All eight L2 pairs fire (B 1.790, J 1.639, L 2.579, V 1.000, W 2.166, X 2.767, Y 3.407, Z 1.489); Z is fully forgiven. L2 is a hit at 0.9759, exact GT font. L53 is a hit at 0.9806, exact GT font. L9 is not recovered by any variant (separate VP limitation, sim 0.7301). Diagnostic BAP 613 (h-geo zeroed entirely, TEMP, reverted): 435/505, ZNCC 0.9066 — the horizontal geometry term is worth 15 hits.

**Audit/report display.** The audit records the correction (`*_geo_h_overlap_correction_px`) and the seam fields (`max_darkness`, `max_ink`) in `seam_costs`. The report renders a † marker (class `overlap-graded`, hover title giving the overlap width subtracted) after the h value when a correction was applied. The report only shows per-char detail for picker-selected lines, so L2's pairs get no visible marker unless L2 is selected.

## Seam audit data (summary.json)

The lob must produce all seam data; that is its whole reason for being.

**Lob seam data regression (fixed Nov 8, 2026).** Commit `a92d0bb` (2026-08-11) removed the only caller of `segment::segment_line` in `src/font_pipeline.rs`, which was the only path to `segment_characters_diag`, the only writer of `word_crop.png`/`seam_overlay.png`/`final_overlay.png` into `seg_plain/`, and the only caller whose diag dir set `audit_mode`. Fix: `src/audit.rs` no longer skips serializing `WordSegSummary.seam_costs`, and `src/font_pipeline.rs` restores a diagnostic-only `segment_line` call once per line, gated on `diag_seg_dir.is_some()`. Scoring is untouched.

**summary.json (restored Nov 8, 2026).** `summary.json` was deliberately removed in commit `10ad2fb` (2026-07-12) when the audit moved to mainline data. The seam viz tool still needs it for `candidate_seam_paths` / `seam_seed_candidates`. Root cause for the A|B case: col 25 was never a candidate seam path, because `candidate_seams` collapses tied runs to a single representative via the local-minimum rule (for the tied run [24, 25] it keeps col 24 only). Fix: `candidate_seams` / `candidate_seams_child` now return the non-representative members of tied runs as `diag_candidates`; `segment_characters_inner` traces them in audit mode into `candidate_paths`; the diag block writes `seg_plain/summary.json` from the segmenter's own traced paths and costs. Diag candidates never go on the heap, so split decisions are unchanged. `tools/seam_viz.py` primary mode falls back to `summary.json` for columns missing from audit.json (audit data wins on conflict). Do not reconstruct seam data by hand; the viz must run on real recorded data.

**A|B seam (columns 24 vs 25).** Ink ties exactly (one pixel each at row 28, darkness 241 in the word crop). Segment penalty is the tiebreaker: col 24 segment_size_penalty 113.38, col 25 104.49 (penalty is (10 × avg_char_width / min_child)²; splitting at 25 gives min_child 25 vs 24). Totals: col 24 = 354.38, col 25 = 345.49. Col 25 wins by 8.89. The DP picked col 24 because the ink cost tied and the choice fell to heap order, not to the segment penalty that favours col 25.

## Test Cheat Sheet

```bash
# Build
cargo build --release 2>&1 | tail -3

# t60 accuracy (the one that matters)
cargo test specimen_font_accuracy --release -- --nocapture --test-threads=1 2>&1 | grep -E "accuracy|FAILED|ok"

# Full t60 misses
cargo test specimen_font_accuracy --release -- --nocapture --test-threads=1 2>&1 | grep -E "accuracy|Misses|  " 

# Crop dump for specific line (via --audit)
./target/release/unprint test-docs/font-timeline-specimen-rasterized.pdf -o /dev/null --audit /tmp/audit-out
ls /tmp/audit-out/<line_dir>/crops/
```

## Current State

- Segmenter: VP + dual-DP seam carving with diagonal masking and midpoint tie-breaking
- Font matching: CI #1 wins directly (no word-level SSIM reranking), with parallel SSIM fast path (dominant font from previous page, threshold ≥0.90)
- SSIM bail-below: `ssim_windowed()` accepts `bail_below` for early exit when running average drops below threshold (used by fast path)
- Audit per-char distances: precomputed crop features via `precompute_crop_features()` + `per_char_distances_precomputed()` — avoids redundant feature extraction across multiple fonts
- Accuracy: 454/480 (94.6%) on font-timeline-specimen.pdf (t60 AA @ 300 DPI)
- Index version: 8, FEAT_LEN: 99, NORM_H: 48

## Context Assurance (Mike Oct 18, 2026)
- **Include 🖨️ in every response** (alongside 📋) to assure Mike that this AGENTS.md is in context. If 🖨️ is missing, the unscan project instructions were lost.
