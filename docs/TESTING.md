# End-to-End Font Testing

unscan's accuracy is measured by comparing its output against a vector PDF whose fonts are known. This document explains the testing pipeline, the tools, and how to run everything.

## Pipeline Overview

```
gen-specimen.py          →  font-timeline-specimen.pdf          (vector, source of truth)
                         →  font-timeline-specimen-scanned.pdf  (rasterized "scan")

rasterize.py rasterize   →  rasterized PDF from any vector PDF

unscan --audit           →  audit.json + SSIM images
       --test    →  ground-truth comparison + report.html
```

1. **gen-specimen.py** builds a multi-page vector PDF using 30+ font families. It calls `rasterize.py` to rasterize.
2. **rasterize.py** handles rasterization (with optional scan artifacts).
3. **unscan** processes the rasterized PDF, producing an audit log with per-line font matches.
4. **report.rs** (built into unscan) compares matches against the vector PDF's ground truth and generates `report.html`.

## Tools

### `tools/rasterize.py`

One script for rasterization:

```bash
# Rasterize a vector PDF to a grayscale raster PDF
python3 tools/rasterize.py rasterize INPUT.pdf OUTPUT.pdf [--dpi 300] [--no-aa] [--backend mupdf|poppler]

# Rasterize with next-step commands printed
python3 tools/rasterize.py prepare INPUT.pdf [-d output_dir] [--dpi 300] [--no-aa] [--backend mupdf|poppler]
```

**Rasterize options:**
- `--dpi N` — resolution (default 300)
- `--no-aa` — disable anti-aliasing (binary threshold, simulates photocopies)
- `--backend mupdf|poppler` — rendering engine (default mupdf)
- `--scan` — shorthand for `--skew 2.0 --noise --speckle --blur 0.7`
- `--skew`, `--noise`, `--speckle`, `--blur` — individual scan artifact controls

**Prepare options:**
- `-d` / `--output-dir` — output directory (default: same as input)
- `-o` / `--output` — explicit rasterized PDF path
- `--rasterize-only` — skip fontmap generation

### Built-in miss report

unscan generates a self-contained HTML miss report at `DIR/report.html` when
both `--audit DIR` and `--test VECTOR.pdf` are set. No separate script
needed.

Ground-truth font identification uses `/Widths` and `Tw` (word spacing) read
directly from the vector PDF's font dictionaries — no external fontmap file
required. On miss lines, the ground-truth font is injected into the CI candidate
list and per-char distances are computed automatically.

## Running Tests

### Rust Test Suite

Tests have strict ordering — **t55 must run before t60/t61/t62**:

```bash
# Build
cargo build --release

# Generate all fixtures (vector PDF, AA + no-AA rasters, char index)
cargo test --test t55_specimen_gen -- --nocapture

# Run accuracy tests (these assert fixtures exist, no fallback generation)
cargo test --test t60_specimen_accuracy_aa -- --nocapture     # AA, threshold 90%
cargo test --test t61_specimen_accuracy_noaa -- --nocapture   # no-AA, threshold 82%
cargo test --test t62_cross_renderer_accuracy -- --nocapture  # Poppler, threshold 82%
```

**t55** always regenerates from scratch — runs gen-specimen.py, then rasterizes AA and no-AA variants at 300 dpi via `rasterize.py`. This is deterministic: same fonts → same PDF → same rasters.

**t60/t61/t62** are pure measurement — they assert the fixture files exist and fail immediately if they don't. No fallback generation, no "if not exists" logic.

### Any Vector PDF (Manual)

The tools work on any vector PDF:

```bash
# 1. Rasterize
python3 tools/rasterize.py rasterize original.pdf /tmp/rasterized.pdf

# 2. Run unscan with audit + ground-truth comparison
./target/release/unscan /tmp/rasterized.pdf \
  -o /tmp/out.pdf --audit /tmp/audit \
  --test original.pdf

# Report at /tmp/audit/report.html
```

**Requirements:** The fonts embedded in the vector PDF must be installed on the system.

### Specimen Pipeline (Full Manual Regeneration)

```bash
# Regenerate specimen + scanned version
cd test-docs && python3 gen-specimen.py && cd ..

# Build
cargo build --release

# Run unscan with audit
./target/release/unscan test-docs/font-timeline-specimen-scanned.pdf \
  -o /tmp/out.pdf --audit /tmp/audit \
  --test test-docs/font-timeline-specimen.pdf

# Report at /tmp/audit/report.html
```

## Reading the Miss Report

The HTML report shows every line where unscan's match disagrees with the vector PDF. For each miss:

- **Page/Line**: Location in the document
- **Text**: The OCR'd text content
- **Expected**: Font from the vector PDF (determined by spatial overlap with ground-truth spans)
- **Got**: Font unscan matched
- **Per-char distances**: How far each character crop was from the correct font vs. the chosen font
- **SSIM images**: Side-by-side scan crop, render crop, and diff

### Accuracy Tracking

The summary line `Report: H/C (P%) — M misses (S SSIM)` is the headline metric.

- **Hit**: unscan's matched font agrees with the ground-truth font
- **Miss**: different font (font miss or SSIM failure)
- **NoGT**: lines where no ground-truth span overlaps the OCR bbox (excluded from denominator)

## Audit Images

The `--audit` directory contains per-line SSIM comparison images:

```
/tmp/audit/
  audit.json
  report.html             # Visual miss report (when --test is set)
  page_1/
    line_000/
      ssim_scan.png       # Scan crop (word-union bbox from OCR)
      ssim_render.png     # Render crop (ink-extent, tight to glyphs)
      ssim_diff.png       # Absolute difference
    line_001/
      ...
```

SSIM scoring uses `ssim_windowed_best_vshift` with ±12px vertical shift to handle baseline alignment differences.

## Font Resolution Pitfalls

`gen-specimen.py` resolves fonts via fontconfig (`fc-list`), the same system unscan uses. This section documents known pitfalls and how `fc_find()` handles them.

### Weight Mismatch (4-Font-Family Naming)

The OpenType spec has two naming models. In the 4-font-family model, SemiBold registers as "Regular" under a weight-qualified family name. Fontconfig's `style=Regular` filter matches all of them.

| Family | `fc_find("X", "Regular")` returned | Actual weight |
|---|---|---|
| Roboto | Roboto-Medium.ttf | 500 |
| Lato | Lato-Hairline.ttf | 250 |
| SourceSans3 | SourceSans3-Semibold.ttf | 600 |

**Fix:** `fc_find()` validates `OS/2.usWeightClass` via fontTools.

### Width Mismatch (Superfamilies)

Large families like Noto Serif ship dozens of width variants under the same family name.

| Entry | Returned file | Width class |
|---|---|---|
| NotoSerif-Bold | NotoSerif-ExtraCondensedBold.ttf | 2 (ExtraCondensed) |

**Fix:** `fc_find()` validates `OS/2.usWidthClass == 5` (Normal).

### Variable Font Axis Defaults

Some variable fonts set `OS/2.usWeightClass` to their lightest instance, not 400. This deprioritizes them vs static instances, which is generally desirable.

| Font file | OS/2 weight | Expected |
|---|---|---|
| SourceSans3[wght].ttf | 200 | 400 |
| Merriweather[opsz,wdth,wght].ttf | 300 | 400 |

### The `fc_find()` Scoring Algorithm

```
For each fontconfig candidate:
  +100  if OS/2.usWidthClass == 5 (normal width)
  +50   if OS/2.usWeightClass == expected_weight (exact match)
  +25   if weight within ±50 of expected (close match)
  +10   if NOT a variable font (static preferred)

Accept highest-scoring candidate with normal width.
Fallback: any candidate with normal width.
Last resort: highest-scoring candidate regardless of width.
```

## When to Regenerate the Specimen

Regenerate when:
- `gen-specimen.py` changes
- New fonts are installed or updated

```bash
cd test-docs && python3 gen-specimen.py
```

Or via the test suite:
```bash
cargo test --test t55_specimen_gen -- --nocapture
```
