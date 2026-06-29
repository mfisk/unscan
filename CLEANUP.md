# Unprint Codebase Cleanup List

Post-refactor housekeeping. Ranked roughly by impact.

---

## Dead Code

1. ~~**`src/image_cache.rs`**~~ — **Done.** Deleted.

2. ~~**`src/ml_forest.rs`**~~ — **Done.** Deleted.

3. ~~**`src/word_match.rs`**~~ — **Done.** Deleted.

4. ~~**`src/word_index.rs`**~~ — **Done.** Deleted.

5. ~~**`dump_limits()`** in `main.rs`~~ — **Done.** Deleted.

6. **`src/bin/train.rs`** — 2,577 lines, exact duplicate of the new `src/train.rs` module (which is now integrated into the main binary). The old binary should be deleted once `src/train.rs` is confirmed working.

7. ~~**`render_char_normalised` / `render_glyph_normalised` / `render_glyph_hires` / `render_char_hires`** in `char_index.rs`~~ — **Done.** `char_index.rs` eliminated; rendering consolidated in `char_render.rs`.

---

## Duplicate Functions

8. **`render_char_at_native_height`** — exists in 3 places:
   - `src/bin/gen_training_data.rs:114`
   - `src/bin/train.rs:354`
   - `src/train.rs:304`
   All should go through `char_render::get_rendered_char()`.

9. **`render_word`** — exists independently in `word_match.rs:252` and `bin/word_index_test.rs:53`. If word_match is deleted, only the test binary remains. Consider using `layout::render_word_ab_glyph` everywhere.

10. **`ssim_compare` wrapper** in `word_match.rs:271` — just calls `crate::ssim::ssim_compare`. Pointless indirection.

11. ~~**`normalize_to_ink_bounds` vs `normalize_to_height`**~~ — **Done.** `char_index.rs` eliminated; `normalize_to_height` in `char_render.rs` is the sole implementation.

---

## Magic Numbers / Scattered Constants

12. **`ref_h = 200.0f32`** — appears in multiple places across char_render, ci_diag, render_ff_compare. This is the "reference measurement scale" for determining glyph ink height. Should be a named constant.

13. **Binarize threshold `128`** — hardcoded in train.rs (2 places), gen_training_data.rs, and the `RenderParams::default()`. Already captured in `RenderParams` but the standalone usages in bin/ files still use raw `128`.

14. **Ink detection threshold `200`** — used in `extract_line_chars` and related functions. This is the "is this pixel ink?" threshold. Should be a named constant.

15. **Ink detection threshold `240`** — used in `verify.rs` (lines 144, 536) for a different "ink" test. Different value, same concept. At minimum document why these differ; ideally unify or name them.

16. **`180` threshold** in `ci_diag.rs:246` — yet another ink threshold variant.

17. **`NORM_H = 24`** vs **`norm_h = 48`** — `ci_diag.rs` and `render_ff_compare.rs` use `norm_h = 48` while the main pipeline uses `NORM_H = 24`. These diagnostic binaries diverge from the real pipeline. If they're meant to match it, they should use `NORM_H`.

---

## Structural Issues

18. ~~**`char_index.rs` is 3,074 lines**~~ — **Done.** Module eliminated. Functionality split across `classifier.rs` (feature extraction, search), `font_match.rs` (font matching), `char_render.rs` (rendering), `segment.rs` (character segmentation), and `features.rs` (feature types).

19. **`main.rs` is ~736 lines** — substantially reduced from 2,183 by extracting `font_pipeline.rs` (match_lines, update_dominant_font, paragraph_font_grouping, compute_font_size_pt), `page_cache::prepare_page()`, `build_audit_entry()`, and `write_audit_report()`. Remaining ~500 lines in `run()` are orchestration glue. Further extraction blocked by high parameter counts on remaining blocks.

20. **`report.rs` is 1,885 lines** of HTML string building — works but is a maintenance burden. No immediate action needed, but would benefit from a templating approach eventually.

21. ~~**`AaVariant` enum** lives in `char_index.rs`~~ — **Done.** Moved to `features.rs`.

22. **8 classifier implementations** in `classifier.rs` (1,259 lines) — Fisher, Triplet, GlobalTriplet, PerCharFisher, Mahalanobis, LDA, MLP, Fusion. With LDA as the default and proven winner, the others are experimental baggage. Consider gating behind a `--classifier` flag without all the `Classifier` trait ceremony, or pruning the ones that never won.

---

## Diagnostic Binaries

23. **`src/bin/ci_diag.rs`** (413 lines) — has its own `render_char_normalised` at `norm_h=48`, hardcoded font paths (`/usr/share/fonts/truetype/specimen-fonts/...`). Uses a different norm height than the main pipeline. Should use `char_render` or be deleted if it's one-off debugging.

24. **`src/bin/render_ff_compare.rs`** — also uses `norm_h = 48` and hardcoded `ref_h = 200.0`. Same story.

25. **`src/bin/learn_weights.rs`** (392 lines) — trains Fisher weights via signal/noise ratio. The LDA trainer supersedes this. Candidate for deletion.

26. **`src/bin/gen_training_data.rs`** (551 lines) — generates `images.bin`/`labels.jsonl` training data. Now that the trainer uses the per-character file cache from `char_render`, this bulk format is obsolete. Delete once confirmed nothing reads `images.bin` anymore.

27. **`src/bin/word_index_test.rs`** (446 lines) — hardcoded specimen font paths, builds word-level index. Word-level matching is disabled. Delete with `word_index.rs`.

---

## Minor

28. ~~**`font_match.rs` is only 24 lines**~~ — **Done.** Now 205 lines with CI search, tie-break logic, and font matching results.

29. **Page cache writes to `/tmp`** (`page_cache.rs:56`) — comment in TOOLS.md warns that `/tmp` is RAM-backed and can OOM. Large PDFs could fill it. Should use `~/.cache/unprint/pages/` instead.

30. **`sha256_prefix` in `char_render.rs`** — named "sha256" but actually uses `DefaultHasher` (SipHash). Not wrong for a cache key, but the name is misleading. Rename to `hash_prefix` or similar.

31. **`is_indexed()` uses linear scan** — `indexed_chars().contains(&c)` on a 111-element Vec. Called per-character during extraction. A `HashSet` or sorted array with binary search would be trivial and correct.

32. ~~**`contrast_stretch`** in `char_index.rs`~~ — **Done.** Eliminated with `char_index.rs`.
