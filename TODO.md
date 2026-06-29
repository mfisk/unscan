# Unprint TODO

## Active

### 1. Faster Rust compilation for debug/hillclimb cycles
The release build takes ~2–4 minutes, which kills iteration speed when tweaking weights, thresholds, or feature code. Research options:
- **`cargo-nextest`** — parallel test runner, may help test-only cycles
- **`sccache`** or `mold` linker — shared compilation cache / faster linking
- **`cranelift` backend** (`-Zcodegen-backend=cranelift`) — faster debug builds, no optimizations
- **Profile-guided split**: keep hot paths in a small crate, cold code in a lib crate (incremental recompilation)
- **`cargo build --profile dev-opt`** — custom profile with `opt-level=1` (faster than release, faster to compile than `opt-level=3`)
- **LDA weights cache**: avoid retraining `lda-weights.bin` during test iteration (already cached to disk)
- **`cargo watch`** — auto-rebuild on save
- Measure: where does time go? `cargo build --timings` to find the bottleneck (codegen? linking? macro expansion?)

### 2. Multi-resolution regression test (300/200/100 DPI)
Current test rasterizes the font-timeline specimen at a single DPI (300). Real-world scanned PDFs vary widely in quality. Expanding to multiple resolutions tests robustness:
- Rasterize `test-docs/font-timeline-specimen.pdf` at 300, 200, and 100 DPI
- Run font identification at each resolution
- Track per-font accuracy at each DPI separately
- Identify fonts/features that degrade at lower resolution (e.g., serif detection, stroke contrast)
- Use multi-DPI results to weight features that are resolution-stable higher
- Consider: should the specimen rasterization happen once (cached) or per-test-run?
- Baseline to establish: what's the accuracy floor at 100 DPI?

## Completed

### Run accuracy regression test
All pending changes tested and verified:
- Fisher discriminant feature weights (replaced hand-tuned group weights)
- Per-char dedup in flat_vecs (354K → ~89K)
- Memory optimization: compact() after save, eliminated all_weighted temp
- **Font cache**: shared LRU cache (64 slots) for all post-index font access
- **Lazy font loading**: scan_fonts() drops bytes after OT detection, index build loads per-thread from disk
- FontMatchResult no longer carries font_data (just font_path)
- **SSIM fast path**: dominant font from previous page, threshold 0.90
- **SSIM bail-below**: early exit in ssim_windowed() when running average drops below threshold
- **Precomputed crop features**: precompute_crop_features() + per_char_distances_precomputed() for audit mode
- **Current baseline**: AA 454/480 = 94.6% (t60, PASS at 93% threshold)
