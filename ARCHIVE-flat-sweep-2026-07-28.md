# Archive — flat sweep up to 2026-07-28 before restart

## Note
- Prior sweep mixed two repos (~/workspace/repos/unscan and tmp/bap-flat-*) with unscan-side.
- User strict order: only use ~/workspace/repos/unscan-side going forward.
- This archive preserves last known tables before clean restart of 0.5, 0.45, 0.4, 0.55.

## Last table from side chat before adding 0.52 (seq 757532) — OCR-clean filtered tot = expected_font!=null && ocr_correct!=false
Exact = hit + similarity_failure, major = hit+minor+similarity_failure, avgZ = mean similarity_score

| variant | tot | exact | major | avgZ |
|---|---|---|---|---|
| pre-prune | 389 | 241 | 317 | 0.9219 |
| base | 373 | 281 | 343 | 0.9226 |
| 0.28 | 369 | 280 | 341 | 0.9247 |
| 0.4 | 398 | 295 | 367 | 0.9245 |
| 0.45 | 389 | 240 | 316 | 0.9219 |
| 0.5 | 397 | 323 | 370 | 0.9300 |
| 0.6 | 402 | 295 | 369 | 0.9221 |
| 0.55 | pending | | | |

## 0.55-release result (from bap-flat-0.55-release, release 7.5M, old repo — saved for reference, invalid per strict order)
- tot 389, exact 242, major 318, avgZ 0.9219281099, hits 205, minor 76, simfail 37, majorMiss 71
- elapsed 528.02s, 505 entries raw, 6 pages
- Delta vs 0.45: +2 exact, +2 major, +2 hits, -2 major_miss

## Side-only runs currently in ~/workspace/repos/unscan-side/test-docs/audit/ (has_font definition, Total=494)

| run | Total | Exact | Major_ok | Primary | Hit | Minor | MajorMiss | SimFail | Avg ZNCC | Pct | mtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audit-before-midpoint.json | 494 | 319 | 417 | 362 | 264 | 98 | 77 | 55 | 0.89 | 73.3% | Jul 25 10:24 |
| audit-after-midpoint-thr2.json | 494 | 392 | 453 | 333 | 272 | 61 | 41 | 120 | 0.8957 | 67.4% | Jul 25 10:42 |
| audit-prev.json | 494 | 400 | 454 | 334 | 280 | 54 | 40 | 120 | 0.8956 | 67.6% | Jul 25 17:47 |
| audit.json | 494 | 372 | 434 | 346 | 284 | 62 | 60 | 88 | 0.8968 | 70.0% | Jul 25 17:59 |

All have entries=505, has_font=494, geo 694.

## Wrong-repo runs that were mixed in prior tables (to be discarded per strict order)
- ~/workspace/repos/unscan/test-docs/audit/ : 507 lines_vectorized era, has_font 495
- ~/workspace/tmp/bap-flat-0.45/audit.json : 505 vec, 693 geo, has_font 505, filtered 389
- ~/workspace/tmp/bap-flat-0.55-release/audit.json : 505 vec, 693 geo, has_font 505, filtered 389

## Intent for clean restart
Re-run sweep in unscan-side only, using release build (optimized) with UNPRINT_FLAT_TOP env var default 0.5:
- 0.5 (default, no env or UNPRINT_FLAT_TOP=0.5)
- 0.45
- 0.4
- 0.55

All runs serial, no timeouts, TMPDIR=$HOME/workspace/tmp, CARGO_BUILD_JOBS=1 MALLOC_ARENA_MAX=1, release binary.

