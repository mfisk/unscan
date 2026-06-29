# Code Cleanup Opportunities — unprint

Audit date: 2026-05-10  
**Status note (June 2026):** Many items below are completed. The `char_index.rs`
module has been eliminated entirely — its functionality was split across
`classifier.rs`, `font_match.rs`, `char_render.rs`, `segment.rs`, and
`features.rs`. Word-level SSIM reranking is removed. `main.rs` has been
substantially refactored with extracted modules (`font_pipeline.rs`,
`page_cache.rs`). Items marked ~~struck~~ are done.

---

## Quick Wins (< 30 min each)

### ~~1. Remove dead `radius_search` infrastructure~~
**Done.** `char_index.rs` eliminated entirely.

### 2. Remove unused dependencies from `Cargo.toml`
**File:** `Cargo.toml`  
**What:** Check for crates with zero references in `src/`. May have changed since original audit.

**Effort:** 2 min. Remove unused lines, `cargo build` to confirm.

### ~~3. Remove dead `ItalicGuess` / `detect_italic_source` code~~
**Status:** Check if still present in current `font_match.rs`.

### ~~4. Stale module doc comment in `char_index.rs`~~
**Done.** `char_index.rs` eliminated.

### ~~5. Lingering "hamburger" references~~
**Done.** `char_index.rs` and related test file are stale artifacts.

---

## Medium Effort (1-2 hours each)

### ~~6. Font name duplication in `CharIndex.entries`~~
**Done.** `char_index.rs` eliminated. Character data now lives in the LDA classifier.

### ~~7. Dead "Stage 2: Union merge" code in `font_match.rs`~~
**Status:** Check if still present in current `font_match.rs`.

### 8. Three separate font rendering functions
**Files:** `src/char_render.rs`, `src/verify.rs`, `src/layout.rs`  
**What:** Multiple rendering functions exist across modules. A shared
`render_glyphs_to_image(font, text, height, options)` utility could reduce duplication.

**Effort:** 2 hours.

### ~~9. `CharSearchResult` still carries `radius` and `n_within_radius` fields~~
**Done.** `char_index.rs` eliminated.

---

## Larger Refactors (half day+)

### ~~10. `char_index.rs` is 1969 lines — too big~~
**Done.** Module eliminated. Functionality distributed across `classifier.rs`,
`font_match.rs`, `char_render.rs`, `segment.rs`, and `features.rs`.

### 11. `font_match.rs` coarse scoring path is complex
**File:** `src/font_match.rs` (~205 lines now)  
**What:** Review complexity after char_index elimination. The CI search and
tie-break logic now lives here. May be cleaner than the original audit found.

### ~~12. Serialization format stores per-entry font names on disk~~
**Done.** `char_index.rs` eliminated. LDA uses its own weight format (`lda-weights.bin`).

---

## Test Coverage Gaps

### 13. No integration test for the full pipeline
**What:** The specimen and Berkeley tests are run via `--test` mode and BAP subagents.
No `cargo test` equivalent that takes a known raster → runs the full pipeline → verifies output.
`t10_char_index_roundtrip.rs` is stale (references eliminated module).

### 14. No test for character segmentation
**What:** The column-ink valley segmentation in `segment.rs` that splits words into
individual character crops has no isolated test.

### ~~15. No test for serialization round-trip at full scale~~
**Done.** Old index format eliminated.

---

## Naming Inconsistencies

### ~~16-18. CharSearchResult, CHAR_INDEX_TOP_N, search_candidates~~
**Done.** `char_index.rs` eliminated. These naming issues no longer apply.

---

## Summary by Priority

| # | Item | Effort | Status |
|---|------|--------|--------|
| 1 | Remove dead radius_search | — | ✅ Done |
| 2 | Remove unused deps | 2 min | Check |
| 3 | Remove dead ItalicGuess | — | Check |
| 4 | Update module doc | — | ✅ Done |
| 5 | Fix hamburger references | — | ✅ Done |
| 6 | Font name dedup | — | ✅ Done |
| 7 | Remove dead union merge | — | Check |
| 8 | Shared render function | 2 hr | Open |
| 9 | Clean up CharSearchResult | — | ✅ Done |
| 10 | Split char_index.rs | — | ✅ Done |
| 11 | Refactor font_match.rs | Review | Open |
| 12 | Serialization name table | — | ✅ Done |
| 13 | Full pipeline integration test | 4 hr | Open |
| 14 | Segmentation test | 2 hr | Open |
| 15 | Serialization round-trip test | — | ✅ Done |
