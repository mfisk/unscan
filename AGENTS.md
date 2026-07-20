# AGENTS.md — Unprint Project

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

- **Always crop to ink bounds first.** Before segmenting, matching, or computing features on a word or character image, trim it down to just the ink. Raw word crops from Tesseract include adjacent lines, extra whitespace, and other garbage. Use `trim_whitespace()` or `normalize_to_ink_bounds()` before doing anything with the image.
- **Always ask before committing.** Never auto-commit.
- **"Stop vibing shit"** — deliberate changes only, no speculative fixes.
- **"Stop doing 1-off diagnostics and ONLY use the main code path."**
- **"Focus on input quality, not fancier algorithms."** Fix images first.
- **Show crops visually** — always present character crops as inline images.
- **Both index-build and index-lookup must use identical crop geometry.**

## Architecture

- `segment_characters()`: VP (zero-ink columns) first, dual-DP seam carving fallback
- `extract_line_chars()`: word-level extraction — VP + seam carving handles crop boundaries
- `normalize_to_ink_bounds()`: scan-time crop normalization to match index-time rendering

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

## Font Identity — search space is font keys

- **Each variant / weight / feature is a separate font key.** A single file (`MyFont-VF.ttf`) produces many distinct fonts in our search space: `MyFont-Regular` (base), `MyFont-VF|wght400` (`wght=400`), `MyFont-VF|wght700` (`wght=700`), `MyFont-Regular|onum`, `|smcp`, etc. Each has its own `font_key = postscript_name|variant_tag`, its own `variations: Option<Vec<([u8;4], f32)>>`, its own `glyph_overrides`. Classification, `geo-cache`, and `glyph-map` all key by `font_key` — never by file path alone.

- **Dedup storage only when truly identical — never collapse distinct keys.** Map many keys to one shared value only when provably identical (same advance/bbox for every gid, same outlines, same metrics). OT feature variants (`variations=None`, `onum`/`smcp`) share base geometry — they only remap unicode→gid via `glyph_overrides`, the underlying gid metrics are identical → share one `OwnedFont`/`GlyphMetrics` block. Variable instances (`wght400` vs `wght700`) have different `gvar` deltas and must be separate entries, even though same file. A missed share wastes memory/time; a false share is a correctness bug.

Generalize to any cache/map/index: key by logical identity, but internally `Arc`/reuse the value when the computed bytes are bit-identical. Verify with equality, not assumption.
