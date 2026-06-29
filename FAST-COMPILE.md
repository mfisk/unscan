# Fast Compile Guide for unprint

## TL;DR — Iteration Workflow

```bash
# Edit code, then:
cargo build --bin unprint        # ~30s incremental debug build
./target/debug/unprint ...       # run with debug binary (opt-level=1)
```

**Use `cargo build --bin unprint`** (not bare `cargo build`). The project has 4+ binaries;
building them all takes ~6 minutes because each re-links the full library. Building just
the main binary skips that.

Use `cargo build --release` only for final accuracy runs where runtime performance matters.

### Typical test cycle

```bash
cargo build --bin unprint                      # ~30s debug build
./target/debug/unprint test-docs/font-timeline-specimen-scanned.pdf \
    --audit /tmp/audit-out -o /tmp/out.pdf \
    --test test-docs/font-timeline-specimen.pdf
# Report at /tmp/audit-out/report.html
```

## Measured Build Times (2026-05-25)

| Scenario | Command | Time |
|----------|---------|------|
| **Incremental, main bin only** | `cargo build --bin unprint` | **30s** |
| Incremental, all bins | `cargo build` | 5m 52s |
| Clean, all bins (debug) | `cargo clean && cargo build` | 33m 32s |
| Release (full) | `cargo build --release` | ~4m |

All times with mold linker, `[profile.dev] opt-level = 1`, no sccache.

## What's Configured

### mold linker (already installed: `/usr/bin/mold` v2.30.0)

```toml
# ~/.cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

mold is the fastest linker on Linux. Main benefit: faster link step on incremental builds.

### Dev profile (Cargo.toml)

```toml
[profile.dev]
opt-level = 1
```

`opt-level = 1` gives reasonable runtime performance without the full optimization cost.
The debug binary is ~1.5-2x slower than release, which is fine for segmentation iteration.

### sccache (installed but NOT enabled)

sccache v0.9.1 is installed at `~/.cargo/bin/sccache` but **not** in `config.toml`.
It actually slows down incremental builds (7m18s vs 5m52s) because it conflicts with
Rust's built-in incremental compilation. Only useful for clean builds across `cargo clean`
cycles.

To enable temporarily for a clean build:
```bash
RUSTC_WRAPPER=sccache cargo build
```

## Tips

- **Touch only what you need**: `touch src/classifier.rs` then `cargo build --bin unprint`
  recompiles only the unprint crate and re-links only the main binary.
- **Don't `cargo clean`** unless you need to. Incremental compilation is the win.
- **Debug binary location**: `./target/debug/unprint`
- **Release binary location**: `./target/release/unprint`
