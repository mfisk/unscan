# AGENTS.md — Unscan Project

## Context Window Hygiene

**Your context is finite. Protect it.**

- **Use subagents for builds and tests.** Never run `cargo build`, `cargo test`, or full `unscan` runs in the main thread. Spawn a subagent, let it report back a summary.
- **Never read diagnostic output in its entirety.** Test runs, cargo builds, and `unscan` stderr can produce thousands of lines. Reading all of it into an LLM context will blow your token budget. Always redirect to a file, then grep/tail for the specific lines you need.
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
- **Tesseract charboxes are JUNK for segmentation** — use for character identity only, never for crop boundaries.
- **Both index-build and index-lookup must use identical crop geometry.**

## Architecture

- `segment_characters()`: VP (zero-ink columns) first, seam carving (Avidan & Shamir 2007) fallback
- `extract_line_chars()`: word-level extraction path — our segmenter handles crop boundaries
- `extract_line_chars_from_charboxes()`: DEAD CODE — charbox path bypassed, do not re-enable
- `normalize_to_ink_bounds()`: scan-time crop normalization to match index-time rendering

## Test Cheat Sheet

```bash
# Build
cargo build --release 2>&1 | tail -3

# t60 accuracy (the one that matters)
cargo test specimen_font_accuracy --release -- --nocapture --test-threads=1 2>&1 | grep -E "accuracy|FAILED|ok"

# Full t60 misses
cargo test specimen_font_accuracy --release -- --nocapture --test-threads=1 2>&1 | grep -E "accuracy|Misses|  " 

# Crop dump for specific line
UNSCAN_DUMP_CROPS=1 ./target/release/unscan -o /tmp/out.pdf test-docs/font-timeline-specimen-rasterized.pdf 2>/dev/null
ls /tmp/unscan-crops/<line_dir>/
```

## Current State

- Charbox path bypassed — word-level path with our segmenter is the only path
- Segmenter: VP + seam carving hybrid (in progress, not yet working)
- Accuracy baseline with charboxes: 86.8% (429/494) — this is what we need to beat
- Index version: 6, FEAT_LEN: 59, NORM_H: see code
