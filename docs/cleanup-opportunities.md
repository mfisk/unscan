# Code Cleanup Opportunities — unscan

Audit date: 2026-05-10  
**Status note (June 2026):** Some items below may be outdated due to subsequent
refactoring. The architecture has changed: word-level SSIM reranking is disabled,
CI #1 wins directly, and font data is no longer stored in FontMatchResult (lazy
load via FontCache). Review against current source before acting on any item.

Codebase: 8,526 lines across 16 `.rs` files + 1 test file

---

## Quick Wins (< 30 min each)

### 1. Remove dead `radius_search` infrastructure
**Files:** `src/char_index.rs`  
**What:** The search was switched from radius search to kNN. The following are now unused:
- `KdTree::radius_search()` (line 337) and `radius_search_recursive()` (line 346) — **0 callers** outside the impl
- `CharIndex::search_radii: HashMap<char, f32>` field (line 1049) — populated but never read from the main path
- `CharIndex::get_search_radius()` accessor (line 1951) — **0 callers**
- The radius computation in `rebuild_trees()` (line 1134-1140): `sigma_mean * sqrt(FEAT_LEN) * 1.5`
- All `search_radii.clear()` / `search_radii.insert()` calls

**Keep:** `dim_sigmas` is still used by the test diagnostic (`ci_single_char_diagnostics`) and by serialization. Could be pruned later if sigmas aren't needed.

**Effort:** 15 min. Delete ~60 lines.

### 2. Remove unused dependencies from `Cargo.toml`
**File:** `Cargo.toml`  
**What:** Two crates have zero references anywhere in `src/`:
- `rayon` — no `par_iter`, `par_bridge`, or any rayon import
- `ordered-float` — no `OrderedFloat` or `ordered_float` references

**Effort:** 2 min. Remove 2 lines, `cargo build` to confirm.

### 3. Remove dead `ItalicGuess` / `detect_italic_source` code
**File:** `src/font_match.rs` (lines 927-986)  
**What:** `ItalicGuess` enum and `detect_italic_source()` function are defined but never called. Compiler already warns about this.

**Effort:** 5 min. Delete ~60 lines.

### 4. Stale module doc comment in `char_index.rs`
**File:** `src/char_index.rs` (lines 1-26)  
**What:** Module doc says:
- "tolerance-band radius queries instead of brute-force cosine scan" — **stale**, now uses kNN
- Feature list only mentions 5 features (profile, aspect, ink_density, v_center, h_balance) — **missing** the 18 v2 features (counters, terminals, crossings, etc.) and serif_score, stroke_contrast, xh_cap_ratio

**Effort:** 10 min. Rewrite the doc comment to match current reality.

### 5. Lingering "hamburger" references in comments/strings
**File:** `tests/char_index_roundtrip.rs`  
**What:** 5 references to "hamburgefontsiv" (the test word) and comments mentioning "hamburger". The test word itself is fine (it's a typography standard), but the comment on line 5 says "hamburgefontsiv" as if explaining the module name — now misleading since the module is `char_index`.

**Effort:** 5 min. Update the doc comment at the top of the test file.

---

## Medium Effort (1-2 hours each)

### 6. Font name duplication in `CharIndex.entries`
**Files:** `src/char_index.rs`  
**What:** `FontCharEntry` stores `font_name: String` per entry. With ~101 indexed chars × 4714 fonts = 476K entries, each storing a ~25-byte font name string. That's **~11 MB** of duplicated name strings in memory and on disk.

The `font_names_table: Vec<String>` already exists as a dedup table for the character index. The entries could store a `font_id: usize` instead and look up names from the table.

**Savings:** ~9.5 MB in memory, ~10 MB on disk (97 MB → ~87 MB index file).

**Effort:** 1-2 hours. Refactor `FontCharEntry` to use `font_id: usize`, update `save_index`/`load_index` to write a name table + ID-based entries, update all code that reads `e.font_name`.

### 7. Dead "Stage 2: Union merge" code in `font_match.rs`
**File:** `src/font_match.rs` (lines 478-527)  
**What:** The union merge section (adding char-index candidates to coarse results) is now vestigial. The coarse scoring loop was changed to a gate: when char index returns candidates, it ONLY scores those. So there's nothing to "merge" afterward — everything is already in the set.

The union merge block iterates the entire `catalog` again looking for char-index fonts not in coarse results, but since the gate already limits coarse to only char-index fonts, this pass will always find 0 additions.

**Effort:** 30 min. Remove the block, verify no behavior change.

### 8. Three separate font rendering functions
**Files:** `src/char_index.rs`, `src/font_match.rs`, `src/verify.rs`, `src/compare.rs`  
**What:** Four different rendering functions exist:
- `render_char_normalised()` in char_index.rs — renders single char at NORM_H for indexing
- `render_text_gray()` in font_match.rs — renders text string at given height for coarse scoring
- `render_words_height_scaled()` in verify.rs — renders words at scaled height for SSIM
- `render_font_crop()` in compare.rs — renders for diagnostic comparison output

Each does its own glyph layout and rasterization. They're different enough that a naive merge would be forced, but a shared `render_glyphs_to_image(font, text, height, options)` utility could reduce duplication.

**Effort:** 2 hours. Define a shared rasterizer in `layout.rs` or a new `render.rs`.

### 9. `CharSearchResult` still carries `radius` and `n_within_radius` fields
**File:** `src/char_index.rs` (line 1461)  
**What:** The `CharSearchResult` struct has:
- `radius: f32` — now hardcoded to `0.0` after kNN switch
- `n_within_radius: usize` — now just equals `hits.len()` (always 50 from kNN)

These are meaningless in kNN mode. The test (`ci_single_char_diagnostics`) prints them but they no longer convey useful information.

**Effort:** 30 min. Remove fields, update test to not print them or replace with `k` and `n_candidates`.

---

## Larger Refactors (half day+)

### 10. `char_index.rs` is 1969 lines — too big
**File:** `src/char_index.rs`  
**What:** This single file contains:
- Feature computation (~400 lines): `compute_features`, counter/terminal/boundary/crossings
- Brute-force nearest-neighbor search (~160 lines): flat vector scan
- Font rendering (~60 lines): `render_char_normalised`
- Index building (~100 lines): `build_char_index`
- Index querying (~120 lines): `search_candidates`, `search_single_char`, `match_line_chars`
- Serialization (~200 lines): `save_index`, `load_index`, `peek_header`
- Character extraction (~200 lines): `extract_line_chars`, `segment_characters`
- `CharIndex` struct (~100 lines)
- Helper functions (~100 lines): weights, median, cosine, etc.

Could split into:
- `char_features.rs` — `CharFeatures` struct + `compute_features` + sub-functions
- `char_index.rs` — `CharIndex` struct, build, query, serialize

**Effort:** Half day. Lots of cross-references to untangle.

### 11. `font_match.rs` coarse scoring path is complex
**File:** `src/font_match.rs` (1320 lines)  
**What:** The main `match_font()` function is ~500 lines with deeply nested logic:
- Pre-filters (mono, bold, OT variants)
- Per-word rendering + width-matching score
- OT variant probing inside the scoring loop
- Digit style detection
- Union merge (now dead, see #7)
- SSIM re-ranking with its own rendering
- Best-of selection with coarse-only fallback

The function does too much. The OT variant probing (~200 lines, starting at line 250) is particularly convoluted — it renders the line text with each possible OT feature combination and compares widths.

**Effort:** 1 day. Extract stages into separate functions.

### 12. Serialization format stores per-entry font names on disk
**File:** `src/char_index.rs` (`save_index` / `load_index`)  
**What:** The on-disk format (v5) writes font name bytes for every entry:
```
for each char:
  for each font_entry:
    u32: name_length
    bytes: font_name     ← stored 101× per font
    [f32; 57]: features
```

A more compact format would write a name table once, then use font_id indices:
```
u32: n_fonts
for each font:
  u32: name_length
  bytes: font_name       ← stored 1× per font
for each char:
  for each entry:
    u32: font_id         ← 4 bytes vs ~25 bytes
    [f32; 57]: features
```

**Savings:** ~9.5 MB on disk (97 MB → ~87 MB). Also faster load since fewer allocations.

**Effort:** 2 hours. Bump INDEX_VERSION to 6.

---

## Test Coverage Gaps

### 13. No integration test for the full pipeline
**What:** `tests/char_index_roundtrip.rs` only tests the char index in isolation. There's no test that:
- Takes a known rasterized PDF
- Runs the full pipeline (OCR → char index → coarse → SSIM → PDF output)
- Verifies the output fonts match ground truth

The specimen and Berkeley tests are run ad-hoc via subagents, not in `cargo test`.

### 14. No test for `extract_line_chars` character segmentation
**What:** The column-ink valley segmentation that splits words into individual character crops has no test. If segmentation is wrong, features are computed on the wrong glyph slices, and everything downstream breaks.

### 15. No test for serialization round-trip at full scale
**What:** The `save_index` / `load_index` pair is implicitly tested by the full-index test (which loads from cache), but there's no explicit test that saves → loads → verifies every entry matches.

---

## Naming Inconsistencies

### 16. `CharSearchResult.n_within_radius` — no longer meaningful
See #9 above.

### 17. `CHAR_INDEX_TOP_N` constant in `font_match.rs` (line 69)
**What:** Set to 50, used when calling `match_line_chars()`. But now that the char index is used as a gate (not union), this constant controls how many fonts enter the coarse pass. The name suggests it's the "char index top N" but it's really "max fonts to coarse-score". Should be renamed to something like `MAX_COARSE_CANDIDATES`.

### 18. `search_candidates` vs `match_line_chars`
**What:** `match_line_chars()` is a 1-line wrapper around `search_candidates()`. The wrapper exists for backward compatibility but adds no value. Could remove `match_line_chars` and call `search_candidates` directly.

---

## Summary by Priority

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | Remove dead radius_search | 15 min | Clean up ~60 lines of dead code |
| 2 | Remove rayon + ordered-float | 2 min | Smaller builds |
| 3 | Remove dead ItalicGuess | 5 min | Clean compiler output |
| 7 | Remove dead union merge | 30 min | Simplify font_match.rs |
| 4 | Update module doc comment | 10 min | Accurate docs |
| 5 | Fix hamburger references | 5 min | Naming consistency |
| 9 | Clean up CharSearchResult | 30 min | Remove confusion |
| 6 | Font name dedup in entries | 1-2 hr | ~10 MB less RAM/disk |
| 12 | Serialization name table | 2 hr | ~10 MB smaller index file |
| 10 | Split char_index.rs | 4 hr | Maintainability |
| 11 | Refactor font_match.rs | 8 hr | Readability |
| 8 | Shared render function | 2 hr | Less duplication |
| 13-15 | Test coverage | 4 hr | Regression safety |
