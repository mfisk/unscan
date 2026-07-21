# Cache atomicity and validity

All persistent caches in `~/.cache/unprint/` are **read-only after creation** and built via **staging file + atomic rename**. Readers never see a partially-written file.

## Implementation

- `src/atomic_file.rs::tmp_for(path)` returns `path + ".tmp"` sibling.
- Every writer creates the temp file, writes fully, `flush()`, `drop()`, then `std::fs::rename(&tmp, path)`. `rename` is atomic on Linux (same filesystem).

Covered caches:

- `font_scan.bin` — `src/font_scan.rs:180-190`, `392-437`
- `catalog.bin` (FONT header) — `src/classifier.rs:1059-1072`, `src/font_scan.rs:177-190`, `src/main.rs:103-115`
- `geo-cache.bin` (BGEO v10) — `src/geo_cache.rs:1038-1146`
- `glyph-map.bin` (NGMP v3) — `src/glyph_map.rs:154-211`
- `lda-weights.bin` / `lda-weights-*.bin`, `ngram` models — `src/classifier.rs:xxx` (`NgramModel::save` does tmp+rename), `src/train.rs:1016-1053`
- Training feature manifests — `src/train.rs:1016-1053`

All do:

```rust
let tmp = tmp_for(path);
let mut w = File::create(&tmp)?;
w.write_all(...)?;
w.flush()?; drop(w);
std::fs::rename(&tmp, path)?; // atomic
```

## Validity, not corruption

Caches are not "corrupted" by partial writes — the rename guarantees either old file or new file is visible. A "stale" cache is detected via `catalog_hash`:

- `catalog_hash` is `hash(font_key sorted)` from current `font_scan` (5898 fonts × dedup).
- Every cache file stores its `catalog_hash` in header.
- Loader does: `if file_hash != current_catalog_hash { retrain/rebuild }`

What happened in the t64 investigation (Jul 20): we experimentally filtered `scan_fonts` to 8–523 fonts via `UNPRINT_FONT_ALLOWLIST`, which wrote a valid `catalog.bin` with hash `0xb43263893022f818` (523 fonts) vs previous `0xe24fde88439a3ac5` (5898 fonts). That triggered `LDA weights stale (catalog_hash 0xb432... != 0xe24f...), retraining...` — not file corruption, just a logical hash mismatch from a filtered catalog. The fix is to never filter inside `scan_fonts` (which controls `catalog_hash`), only filter at matching time.

Thus `lda-weights.bin` was not corrupted, it was correctly invalidated.

## Staging guarantees

- No cache is ever truncated in place.
- No reader sees a half-written file.
- If process crashes mid-write, `.tmp` remains and is ignored on next run; old cache remains valid.
- All `write_all` paths are covered by tests.
