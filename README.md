# unscan

Replace scanned (raster) text in documents with native **vector** text —
dramatic file-size reduction and quality improvement while maintaining
**zero information loss**.

## Philosophy

* **Aggressive vectorisation** of high-confidence text and geometry.
* **Conservative fallback** — when in doubt, keep the original pixels.
* **No compression artefacts** — remaining raster stays at full resolution with
  lossless (FlateDecode/PNG) encoding.
* **Full audit trail** — a JSON sidecar log documents every decision.

## How it works

1. **OCR** — Tesseract extracts word-level bounding boxes and confidence scores.
2. **Font matching** — each word is segmented into individual character crops
   (VP split + seam carving DP — see [`SEGMENTATION.md`](SEGMENTATION.md)),
   then each crop is converted to a 99-dimensional feature vector
   (see [`FEATURES.md`](FEATURES.md)). Per-character kd-tree nearest-neighbor
   search against a pre-built font index produces a ranked candidate list
   (the CI — Character Index). The top CI candidates are reranked using
   word-level SSIM: each candidate font renders the word, and SSIM against
   the scan crop picks the best visual match. OpenType feature variants
   (old-style figures, small caps, stylistic sets, etc.) are matched as
   separate candidates — the SSIM comparison naturally picks the correct
   variant without heuristics. Common Latin ligatures (ff, fi, fl, ffi, ffl)
   are handled via dual-path segmentation: plain OCR characters vs.
   ligature-collapsed characters, with the higher-scoring path winning.
3. **Decision matrix** — OCR confidence and font-match score are checked
   against user-configurable thresholds (`--min-ocr-confidence`,
   `--min-font-confidence`). Lines passing both thresholds are vectorised;
   all others keep the original raster.
4. **SSIM verification** — vector text is rendered back to raster and compared
   with the original region (ink-cropped to actual glyph extent). If
   SSIM < 0.3, the region reverts to raster.
5. **Geometry vectorisation** — horizontal/vertical lines, solid-colour fills,
   and rectangles are replaced with native PDF paths.
6. **PDF generation** — vector text + vector geometry + lossless raster fragments,
   laid on a native background-colour fill.

## Installation

```bash
# Build
cargo build --release

# The binary is at target/release/unscan
```

### Runtime dependencies

| Tool | Install | Purpose |
|------|---------|---------|
| `tesseract` | `apt install tesseract-ocr` | OCR engine |
| `pdftoppm` | `apt install poppler-utils` | PDF → raster |

### Install all recommended fonts

```bash
bash scripts/install-all-fonts.sh
```

The script is idempotent and installs MS Core Fonts, Google Fonts families,
typewriter fonts, and LaTeX fonts — everything needed for accurate matching.
See [`docs/POPULAR_FONTS.md`](docs/POPULAR_FONTS.md) for the full catalog of
supported font families (Word, typewriter, LaTeX, Google Fonts) and manual
install instructions.

## Font ground-truth map

When generating the specimen PDF, `test-docs/gen-specimen.py` emits a
`font-timeline-specimen-fontmap.json` mapping font names to the exact
TTF/OTF files used. This fontmap serves two purposes:

1. **Miss report rendering:** pass it to `tools/char-misses.py` so the report
   can render ground-truth characters from the correct font files.

2. **Audit inclusion:** pass it to unscan via `--include-fontmap` to ensure
   all ground-truth fonts appear in the CI candidate list for every line,
   even if CI would normally prune them.

```bash
# Run unscan with fontmap-injected candidates
unscan test-docs/font-timeline-specimen-rasterized.pdf \
  -o /tmp/out.pdf --audit /tmp/audit-out \
  --include-fontmap test-docs/font-timeline-specimen-fontmap.json

# Generate the visual miss report
python3 tools/char-misses.py /tmp/audit-out/audit.json \
  test-docs/font-timeline-specimen.pdf \
  -o /tmp/misses.html \
  --fontmap test-docs/font-timeline-specimen-fontmap.json
```

## Usage

```bash
unscan input.pdf -o output.pdf

# Override confidence thresholds
unscan input.pdf -o output.pdf \
  --min-ocr-confidence 85 \
  --min-font-confidence 0.65

# Supply extra fonts
unscan input.pdf -o output.pdf --font-dir ~/my-fonts --font-dir /mnt/win/Windows/Fonts

# Skip geometry detection
unscan input.pdf -o output.pdf --no-geometry

# Audit + diagnostics (writes audit.json and segmentation images to DIR)
unscan input.pdf -o output.pdf --audit /tmp/audit-out

# Debug overlay (semitransparent red vector text over original raster)
unscan input.pdf -o output.pdf --overlay

# Image input (PNG, JPEG, TIFF, BMP, GIF, WebP)
unscan scan.png -o output.pdf
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | *(required)* | Output PDF path |
| `--min-ocr-confidence` | 0 | Minimum Tesseract confidence (0–100) to attempt vectorisation |
| `--min-font-confidence` | 0.10 | Minimum CI score (0.0–1.0) to accept a font match |
| `--dpi` | 300 | DPI for rasterising PDF pages |
| `--font-dir` | *(system defaults)* | Extra font search directory (repeatable) |
| `--no-geometry` | off | Skip line / rectangle / fill vectorisation |
| `--overlay` | off | Debug mode: render vector text in semitransparent red over original raster |
| `--smooth` | off | Unify per-word font sizes within consecutive same-font runs to their median |
| `--audit` | *(none)* | Write audit JSON + per-line segmentation diagnostics to DIR |
| `--compare` | off | Generate side-by-side scan vs. render comparison images |
| `--include-font` | *(none)* | Force a font (case-insensitive substring) into CI candidate list for every line |
| `--include-fontmap` | *(none)* | Inject all fonts from a fontmap JSON into CI candidate list |
| `--thoroughness` | 1.0 | Scale CI thresholds — higher = more candidates survive, slower |
| `--index` | off | Scan fonts, update the character index cache, and exit |
| `--index-path` | `~/.cache/unscan/char-index.bin` | Path to the character index cache file |
| `--rebuild-index` | off | Force a full rebuild of the character index, ignoring cache |
| `--diag-ref-font` | *(none)* | Render each extracted character from this font file for comparison (requires `--audit`) |

## Font search paths

`unscan` searches these directories automatically:

**Linux:**
- `/usr/share/fonts/`
- `/usr/local/share/fonts/`
- `/usr/share/fonts/truetype/msttcorefonts/`
- `/usr/share/texlive/texmf-dist/fonts/opentype/`
- `/usr/share/texlive/texmf-dist/fonts/truetype/`
- `/usr/share/texmf/fonts/opentype/`
- `/usr/share/texmf/fonts/truetype/`
- `~/.fonts/`, `~/.local/share/fonts/`
- `~/texmf/fonts/`

**macOS:**
- `/Library/Fonts/`, `/System/Library/Fonts/`
- `~/Library/Fonts/`

**Windows:**
- `C:\Windows\Fonts`

**Custom:** `--font-dir <path>` (repeatable)

### Font name aliasing

Font files often have cryptic filenames (`arialbd.ttf`, `nimbussans-regular.otf`).
An alias table maps these to canonical family names with bold/italic metadata.
Notably, URW Nimbus clones (NimbusSans, NimbusRoman, NimbusMonoPS) are mapped
to the PDF Base-14 canonical names (Helvetica, Times-Roman, Courier) so output
PDFs reference standard names all viewers understand. See
[`docs/font-aliasing.md`](docs/font-aliasing.md) for details.

## Tools

| Script | Purpose |
|--------|---------|
| `tools/char-misses.py` | Generate visual HTML miss report from audit JSON + vector PDF ground truth |
| `tools/rasterize.py` | Rasterize vector PDFs, build fontmaps, or both (`rasterize`, `fontmap`, `prepare` subcommands) |
| `scripts/install-all-fonts.sh` | Install all recommended fonts (MS Core, Google Fonts, typewriter, LaTeX) |
| `test-docs/gen-specimen.py` | Generate the 6-page font timeline specimen PDF + fontmap |
| `test-docs/gen-resolution-series.py` | Generate resolution degradation series (600→fax DPI) |
| `test-docs/gen-ligature-test.py` | Generate the ligature test PDF |
| `test-docs/gen-mixed-font-specimen.py` | Generate the mixed-font (intra-line switching) specimen |

## Audit log

When `--audit DIR` is set, unscan writes `DIR/audit.json` with full pipeline
decisions, plus per-line directories containing per-word subdirectories with
segmentation overlays, character crops, and summary JSONs. Use
`tools/char-misses.py DIR/audit.json VECTOR.pdf` to generate a visual miss report.

```
DIR/
  audit.json                              # Top-level pipeline decisions
  p1_L000_A_Timeline_of/                  # Per-line directory
    ssim_scan.png                         # Scan crop used for SSIM (word-union bbox)
    ssim_render.png                       # Render crop used for SSIM (ink-cropped)
    ssim_diff.png                         # Absolute difference
    word_000_Timeline/                    # Per-word directory
      seg_plain/                          # Plain segmentation path
        overlay.png                       # VP + seam split visualization
        00_T.png, 01_i.png, ...          # Character crops (48px normalized)
      seg_lig/                            # Ligature segmentation path (if applicable)
        overlay.png
        00_T.png, ...
    line_summary.json                     # CI top candidates, font match
```

Without `--audit`, a default audit JSON is still written as `<output>.audit.json`.

```json
{
  "input_file": "scan.pdf",
  "output_file": "out.pdf",
  "input_size_bytes": 52428800,
  "output_size_bytes": 1048576,
  "compression_ratio": 50.0,
  "pages": [ ... ],
  "text_entries": [
    {
      "page": 1,
      "line_index": 0,
      "text": "PURCHASE AGREEMENT",
      "ocr_confidence": 96.0,
      "font_matched": "Arial",
      "font_confidence": 0.87,
      "ssim_score": 0.92,
      "decision": "vectorized",
      "reason": "Vectorised",
      "bbox": { "x": 150, "y": 80, "width": 800, "height": 45 }
    }
  ],
  "geometry_entries": [ ... ]
}
```

## License

MIT

## Test Suite

The `test-docs/` directory contains a comprehensive ground-truth corpus for
validating font detection accuracy. See [`test-docs/README.md`](test-docs/README.md)
for full documentation.

### Test font prerequisites

The test suite requires **Microsoft core TTF fonts** and the **specimen fonts**
(Google Fonts / OFL) to be installed.

```bash
# Install required fonts (requires sudo)
./scripts/install-all-fonts.sh
```

### Tests

Tests live in the `tests/` directory:

| Test | What it tests |
|------|---------------|
| `t10_char_index_roundtrip.rs` | Character index build + query roundtrip |
| `t20_distance_analysis.rs` | Feature-space distance analysis |
| `t30_regression_ssim.rs` | SSIM regression checks |
| `t40_bodoni_sentence.rs` | Single-font smoke test (Libre Bodoni 400) |
| `t40_ligature.sh` | 21 ligature lines (3 fonts × with/without ligatures), asserts 21/21 hits |
| `t50_output_quality.rs` | Output PDF quality validation |
| `t55_specimen_gen.rs` | Specimen generation test |
| `t58_word_segmentation.rs` | Word-level segmentation validation |
| `t60_specimen_accuracy_aa.rs` | 30 fonts, 6 pages — anti-aliased accuracy baseline |
| `t61_specimen_accuracy_noaa.rs` | Same content without anti-aliasing |
| `t62_cross_renderer_accuracy.rs` | Cross-renderer stability |

### Test tiers

| Tier | Source | What it tests |
|------|--------|---------------|
| **Font timeline specimen** | `test-docs/font-timeline-specimen.pdf` | 30 fonts, 6 pages, 500 years, OT variants — the vector ground truth |
| **Ligature test** | `test-docs/ligature-test.pdf` | 3 fonts × 7 ligature lines — dual-path CI validation |
| **Bodoni sentence** | `test-docs/bodoni-sentence-raster.pdf` | Single-font smoke test — must match Libre Bodoni 400 |
| **Mixed-font specimen** | `test-docs/mixed-font-specimen-raster.pdf` | Intra-line font switching (sans/serif/mono/bold/italic) |
| **Resolution series** | `test-docs/resolution-series/specimen-*.pdf` | Same content at 600→fax DPI — measures degradation tolerance |

## OpenType Feature Variant Detection

The font catalog doesn't just match base fonts — it matches specific
OpenType feature configurations. During font scanning, `rustybuzz` (a pure-Rust
harfbuzz port) probes each font for 25+ OT features and creates a separate
catalog entry for any feature that changes glyph shapes.

### How it works

```
Font file: SourceSerif4-Regular.ttf
  └─ Default entry:     "Source Serif 4"          (lining figures, normal lowercase)
  └─ Variant [onum]:    "Source Serif 4 [onum]"   (old-style figures — different 0-9 glyphs)
  └─ Variant [smcp]:    "Source Serif 4 [smcp]"   (small caps — different a-z glyphs)
  └─ Variant [ss01]:    "Source Serif 4 [ss01]"   (stylistic set 1)
  └─ Variant [ss02]:    "Source Serif 4 [ss02]"   (stylistic set 2)
```

Each variant entry carries a `glyph_overrides` map — only the characters whose
glyph IDs actually changed are stored. During rendering for SSIM comparison,
`resolve_glyph()` checks the override map first, falling back to the standard
cmap lookup. This means SSIM naturally picks `Source Serif 4 [onum]` over
plain `Source Serif 4` when the source document uses old-style figures —
no figure-style detection heuristic needed.

### Features probed

| Tag | Name | What changes |
|-----|------|-------------|
| `onum` | Old-style figures | Digits 0–9 get descenders/ascenders |
| `lnum` | Lining figures | Forces tabular-height digits (explicit) |
| `smcp` | Small caps | Lowercase a–z become small capitals |
| `c2sc` | Capitals to small caps | Uppercase A–Z become small capitals |
| `swsh` | Swash | Decorative alternate letterforms |
| `salt` | Stylistic alternates | General alternate glyphs |
| `titl` | Titling alternates | Capitals optimized for large sizes |
| `hist` | Historical forms | Archaic letterforms (e.g. long s) |
| `ss01`–`ss20` | Stylistic sets | Font-specific named alternate sets |
| `liga` | Standard ligatures | ff, fi, fl → single glyphs (probed for ligature detection) |
| `dlig` | Discretionary ligatures | ffi, ffl → single glyphs (probed for ligature detection) |

### Key code

OT variant detection is in `src/font_scan.rs`:

```rust
// detect_ot_variants() shapes a Latin probe string with each feature
// and compares glyph IDs against the default. Returns only features
// that actually change at least one glyph.
let variants = detect_ot_variants(&font_data);
for (tag, overrides) in &variants {
    // Each variant becomes a separate FontEntry with glyph_overrides
    let mut var_entry = base_entry.clone();
    var_entry.variant_tag = tag.clone();
    var_entry.glyph_overrides = Some(overrides.clone());
    catalog.push(var_entry);
}
```

Glyph override resolution is in `src/char_index.rs`:

```rust
// resolve_glyph() checks the override map before falling back to cmap
pub fn resolve_glyph<F: ab_glyph::Font>(
    font: &F, ch: char, overrides: Option<&[(char, u16)]>
) -> ab_glyph::GlyphId {
    if let Some(map) = overrides {
        if let Some(&(_, gid)) = map.iter().find(|(c, _)| *c == ch) {
            return ab_glyph::GlyphId(gid);
        }
    }
    font.glyph_id(ch)
}
```
