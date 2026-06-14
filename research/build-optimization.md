# Build Optimization Benchmark — unscan

**Date:** 2026-06-13  
**System:** AMD EPYC 9D25 (2 vCPUs), Rust 1.96.0 stable, x86_64-unknown-linux-gnu  
**Workload:** Clean `cargo build --release --bin unscan` + full audit run on `font-timeline-specimen-scanned.pdf`

## Results

| Configuration | Build (s) | Audit (s) | Total (s) | Audit Δ | Total Δ |
|---|---:|---:|---:|---:|---:|
| **debug** (`cargo build`, no release) | 640 | 361 | 1001 | +86% | +21% |
| **baseline** (no `[profile.release]`) | 635 | 194 | 829 | — | — |
| opt-level = 3 | 778 | 167 | 945 | −14% | +14% |
| codegen-units = 1 | 459 | 132 | 591 | −32% | −29% |
| lto = "thin" | 354 | 156 | 510 | −20% | −38% |
| lto = "thin" + cgu = 1 | 439 | 197 | 636 | +2% | −23% |
| RUSTFLAGS: target-cpu=native | 458 | 114 | 572 | **−41%** | −31% |
| lto = "thin" + cgu = 1 + native | 456 | 111 | 567 | **−43%** | −32% |
| lto = "thin" + cgu = 1 + native + mold | 431 | 106 | 537 | **−45%** | **−35%** |
| opt3 + lto-thin + cgu1 + native + mold | 431 | 106 | 537 | **−45%** | **−35%** |

## Analysis

### Debug profile

Debug builds (`cargo build` with no `--release`) compile in roughly the same time as baseline release (640s vs 635s) but run **86% slower** (361s vs 194s). The `dev` profile applies `opt-level = 0` plus full debuginfo, so it compiles just as many crates but skips all optimisation. The result is a binary that's useful for `gdb`/`lldb` but impractical for iteration on accuracy — a single test run takes 6 minutes. For fast iteration, `--release` with `target-cpu=native` is strictly better: similar build time, 3.4× faster runtime.

### What helps runtime (audit time)

1. **`target-cpu=native`** is the single biggest runtime win (194 → 114s, −41%). This EPYC 9D25 supports AVX-512 and other modern SIMD extensions that the default `generic` target doesn't use. The image processing and feature vector math in unscan benefits enormously from wider SIMD.

2. **`codegen-units = 1`** gives a solid −32% by enabling better inlining across the entire crate — the compiler sees the whole crate as one unit and can inline hot paths that would be opaque with 16 separate codegen units.

3. **`lto = "thin"`** alone gives −20% by enabling cross-crate inlining (into deps like `image`, `ab_glyph`).

4. **`opt-level = 3`** gives a modest −14% runtime improvement but costs +23% build time. When combined with LTO + cgu1 + native, it adds nothing — those flags already unlocked the same optimizations.

5. **mold linker** shaves ~25s off build time (link phase only). No runtime impact; it just links faster.

### Combinations

- `target-cpu=native` alone captures most of the runtime benefit.
- Adding LTO thin + cgu1 on top squeezes out another ~8s (114 → 106s).
- opt-level 3 on top of the full stack adds zero additional benefit.
- mold helps build speed modestly.

### Surprise: lto = "thin" + cgu = 1 WITHOUT native

This combo (439s build, 197s audit) performed *worse* at runtime than cgu=1 alone (459s build, 132s audit). Thin LTO with a single codegen unit may be pessimizing code layout — the two optimizations can interfere when thin LTO makes different inlining decisions than the single-CU pass alone. Adding `target-cpu=native` fixes this by giving the optimizer enough SIMD width to overcome the layout issues.

## Recommendation

### For maximum runtime performance (CI, production):

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

```sh
RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=mold"
```

**Result:** 431s build, 106s audit → 537s total (−35% vs baseline)

### For best build+run compromise (development):

```toml
# No [profile.release] section (use defaults)
```

```sh
RUSTFLAGS="-C target-cpu=native"
```

**Result:** 458s build, 114s audit → 572s total (−31% vs baseline)  
Simpler, nearly as fast at runtime, and avoids the LTO/cgu build overhead.

### Skip:

- **opt-level = 3**: no benefit over opt-level 2 when combined with LTO/cgu1/native
- **lto = "fat"**: not tested due to build failures (resource-hungry on 2 vCPU), and thin LTO captures most of the benefit
- **PGO**: would require nightly; skip on stable toolchain

## Notes

- All configurations produced identical audit results (402/480, 83.8%), confirming build flags don't affect correctness.
- Build times include all dependencies from clean. Incremental rebuilds (source-only changes) would be much faster for all configs.
- The `target-cpu=native` flag makes the binary non-portable (tied to this CPU family). Fine for local dev and CI on the same hardware; for distributed binaries, use `target-cpu=x86-64-v3` (AVX2) as a reasonable portable-but-fast compromise.
