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
   then each crop is compared against every candidate font via per-character
   SSIM. The font with the best aggregate word-level SSIM wins. OpenType
   feature variants (old-style figures, small caps, stylistic sets, etc.) are
   matched as separate candidates — the SSIM comparison naturally picks the
   correct variant without heuristics.
3. **Decision matrix** —
   | OCR confidence | Font match | Action |
   |----------------|-----------|--------|
   | High (≥ 90 %)  | High (≥ 0.7) | Vectorise |
   | High           | Low        | **Keep raster** — log warning |
   | Low            | any        | **Keep raster** — log warning |
4. **SSIM verification** — vector text is rendered back to raster and compared
   with the original region. If SSIM < threshold, the region reverts to raster.
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
| `cabextract` | `apt install cabextract` | MS core font extraction |

### Install all recommended fonts

```bash
# MS Core Fonts (Arial, Times New Roman, Courier New, Georgia, Verdana, etc.)
sudo apt install ttf-mscorefonts-installer

# If SourceForge is blocked by proxy, download manually:
sudo apt install cabextract
mkdir -p /tmp/msfonts
for f in andale32 arial32 arialb32 comic32 courie32 georgi32 impact32 \
         times32 trebuc32 verdan32 webdin32; do
    wget "https://downloads.sourceforge.net/corefonts/\${f}.exe" -O /tmp/\${f}.exe
    cabextract -q -d /tmp/msfonts /tmp/\${f}.exe
done
sudo mkdir -p /usr/share/fonts/truetype/msttcorefonts
sudo cp /tmp/msfonts/*.ttf /tmp/msfonts/*.TTF /usr/share/fonts/truetype/msttcorefonts/
sudo fc-cache -f

# Calibri/Cambria metric-compatible replacements
sudo apt install fonts-crosextra-carlito fonts-crosextra-caladea

# Liberation fonts (metric-compatible Arial/Times/Courier replacements)
sudo apt install fonts-liberation

# IBM typewriter fonts (for scanned legal/govt documents)
sudo mkdir -p /usr/local/share/fonts/typewriter
# Copy OGCourier*.ttf and "IBM Selectric Light"*.ttf here -- see
# "Working with Typewriter & Vintage Documents" section below
sudo fc-cache -f /usr/local/share/fonts/typewriter/

# LaTeX fonts
sudo apt install fonts-lmodern texlive-fonts-recommended

# Wide Unicode coverage
sudo apt install fonts-noto
```

> **Automated installer:** For a one-shot setup (including EULA display for MS Core Fonts, typewriter fonts, and specimen fonts), run:
> ```bash
> sudo bash scripts/install-all-fonts.sh
> ```

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
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--min-ocr-confidence` | 0 | Minimum Tesseract confidence (0–100) to attempt vectorisation |
| `--min-font-confidence` | 0.10 | Minimum CI score (0.0–1.0) to accept a font match |
| `--dpi` | 300 | DPI for rasterising PDF pages |
| `--font-dir` | *(system defaults)* | Extra font search directory (repeatable) |
| `--no-geometry` | off | Skip line / rectangle / fill vectorisation |
| `--audit` | *(none)* | Write audit JSON + segmentation diagnostics to DIR |

## Working with Microsoft Word Documents

Most scanned documents were originally created in Word. To get the best font
matching, install the matching fonts.

### Core MS fonts (free, redistributable)

```bash
# Installs: Arial, Times New Roman, Courier New, Verdana, Georgia,
#           Trebuchet MS, Comic Sans MS, Impact, Andale Mono, Webdings
sudo apt install ttf-mscorefonts-installer
```

### Calibri, Cambria, Consolas, Segoe UI, Aptos …

These proprietary fonts are **not** included in the free package. Options:

1. **Copy from a Windows installation:**
   ```bash
   mkdir -p ~/ms-fonts
   # From a Windows machine or VM:
   cp /mnt/c/Windows/Fonts/calibri*.ttf ~/ms-fonts/
   cp /mnt/c/Windows/Fonts/cambria*.ttf ~/ms-fonts/
   cp /mnt/c/Windows/Fonts/consola*.ttf ~/ms-fonts/
   cp /mnt/c/Windows/Fonts/segoeui*.ttf ~/ms-fonts/
   # Then:
   unscan input.pdf -o out.pdf --font-dir ~/ms-fonts
   ```

2. **Extract from a Microsoft Office installer** (if licensed).

3. **Extract from a LibreOffice install** — the Carlito and Caladea fonts are
   metric-compatible with Calibri and Cambria:
   ```bash
   sudo apt install fonts-crosextra-carlito fonts-crosextra-caladea
   ```

### macOS / Windows

On macOS, system fonts under `/Library/Fonts/` and `~/Library/Fonts/` are
scanned automatically. On Windows, `C:\Windows\Fonts` is included.

## Working with Typewriter & Vintage Documents

Scanned legal documents, government records, and correspondence from the
1960s–1990s typically use IBM Selectric or Courier typewriter fonts.

### Install typewriter fonts

```bash
# Create a dedicated directory
sudo mkdir -p /usr/local/share/fonts/typewriter

# Prestige Elite — the condensed/narrow IBM typewriter font used in most
# legal and government documents from the 1970s–1990s
# Regular weight (from GitHub):
wget "https://raw.githubusercontent.com/maseyyi/font-prestige-elite/master/prestige.ttf" \
    -O /usr/local/share/fonts/typewriter/PrestigeElite-Regular.ttf
# Bold weight (from font.download):
wget "https://font.download/dl/font/prestige-elite-std.zip" -O /tmp/pe-bold.zip
unzip -j /tmp/pe-bold.zip "*.otf" -d /usr/local/share/fonts/typewriter/

# Letter Gothic — IBM's sans-serif narrow monospace (URW free version)
wget "https://font.download/dl/font/lettergothic.zip" -O /tmp/lg.zip
unzip -j /tmp/lg.zip "*.ttf" -d /usr/local/share/fonts/typewriter/

# OG Courier — IBM's original 1955 Courier (the actual typewriter font, not Courier New)
# Download from: https://github.com/ATypI/OriginalCourier
# Place Regular, Bold, Italic, BoldItalic .ttf files in:
sudo cp OGCourier*.ttf /usr/local/share/fonts/typewriter/

# IBM Selectric Light — faithful recreation of the Selectric II ball typeface
# Download from font archives / dafont.com
sudo cp "IBM Selectric Light"*.ttf /usr/local/share/fonts/typewriter/

# Additional typewriter-style fonts (available via apt)
sudo apt install fonts-courier-prime   # Google's Courier redesign (screen-optimized)

# Rebuild font cache
sudo fc-cache -f /usr/local/share/fonts/typewriter/
```

### Key typewriter font families

| Original typewriter | Best digital match | Notes |
|---|---|---|
| IBM Selectric condensed (12-pitch) | **Prestige Elite** | The standard legal/govt typewriter font — narrower than Courier |
| IBM Selectric (standard 10-pitch) | OG Courier, IBM Selectric Light | The dominant office typewriter 1961–1990 |
| IBM Selectric sans-serif | Letter Gothic | Clean, narrow, sans-serif monospace |
| IBM Executive | Courier Prime | Proportional spacing variant |
| Olympia / Adler | Courier 10 Pitch | European typewriter standard |
| Government / DOJ docs | **Prestige Elite**, OG Courier | Most US govt typed on IBM Selectrics at 12-pitch |

### Why not just Courier New?

Courier New (Microsoft's version) has noticeably different stroke weights
and letter spacing compared to actual IBM typewriter output. More importantly,
many government and legal documents used Prestige Elite (the condensed 12-pitch
element), not standard Courier. Prestige Elite is significantly narrower than
any Courier variant — using Courier to match a Prestige Elite document will
always fail. For scanned typewriter documents, install both Prestige Elite and
OG Courier to cover the two most common IBM typeball elements.

## Working with LaTeX Documents

Academic papers and textbooks often use Computer Modern / Latin Modern fonts.

### Quick install

```bash
# Latin Modern (the modern OTF version of Computer Modern)
sudo apt install fonts-lmodern

# Core TeX fonts (TeX Gyre family, STIX, Libertinus, etc.)
sudo apt install texlive-fonts-recommended

# Everything (large — includes hundreds of font families)
sudo apt install texlive-fonts-extra
```

### Key font families

| LaTeX family | OTF package | Notes |
|---|---|---|
| Computer Modern / Latin Modern | `fonts-lmodern` | Default LaTeX text font |
| TeX Gyre Termes | `texlive-fonts-recommended` | Times-compatible |
| TeX Gyre Heros | `texlive-fonts-recommended` | Helvetica-compatible |
| TeX Gyre Pagella | `texlive-fonts-recommended` | Palatino-compatible |
| TeX Gyre Cursor | `texlive-fonts-recommended` | Courier-compatible |
| STIX Two | `fonts-stix` | Math + text, wide Unicode coverage |
| Libertinus | `fonts-libertinus` | Popular modern serif family |

### Math-heavy regions

LaTeX math uses specialised symbol fonts (AMS, STIX, etc.) with unusual
glyph mappings. If the font matcher can't confidently match a math expression,
it falls back to raster — which is the safe choice. The audit log still records
the OCR text content for reference.

## Font search paths

`unscan` searches these directories automatically:

**Linux:**
- `/usr/share/fonts/`
- `/usr/local/share/fonts/`
- `/usr/share/fonts/truetype/msttcorefonts/`
- `/usr/share/texlive/texmf-dist/fonts/opentype/`
- `/usr/share/texlive/texmf-dist/fonts/truetype/`
- `~/.fonts/`, `~/.local/share/fonts/`
- `~/texmf/fonts/`

**macOS:**
- `/Library/Fonts/`, `/System/Library/Fonts/`
- `~/Library/Fonts/`

**Windows:**
- `C:\Windows\Fonts`

**Custom:** `--font-dir <path>` (repeatable)

## Audit log

When `--audit DIR` is set, unscan writes `DIR/audit.json` with full pipeline
decisions plus per-word segmentation diagnostics (crops, seams, overlays) into
the same directory. Use `tools/char-misses.py DIR VECTOR.pdf` to generate a
visual miss report.

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
(Google Fonts / OFL) to be installed. These are hard prerequisites — without
them, the ground truth sections for Times New Roman, Arial, Courier New,
Georgia, Verdana, Trebuchet MS, and Comic Sans MS cannot be scored.

```bash
# Check whether required fonts are installed
./scripts/check-fonts.sh

# Install missing fonts automatically (requires sudo)
./scripts/check-fonts.sh --install
```

**MS Core Fonts** (required for ground truth sections 7, 9, 11–15):

```bash
sudo apt install ttf-mscorefonts-installer
```

This installs Arial, Times New Roman, Courier New, Georgia, Verdana,
Trebuchet MS, Comic Sans MS, and others into
`/usr/share/fonts/truetype/msttcorefonts/`.

**Specimen fonts** (required for ground truth sections 0–6, 16–29):

```bash
# Downloaded automatically by gen-specimen.py on first run
cd test-docs && python3 gen-specimen.py
```

These install into `/usr/share/fonts/truetype/specimen-fonts/`.

### Test cases

| Test | Input | Ground truth | What it tests |
|------|-------|-------------|---------------|
| **Font timeline specimen** | `font-timeline-specimen-scanned.pdf` | `font-timeline-specimen.json` | 30 fonts across 500 years — the full ground truth |
| **Bodoni sentence** | `bodoni-sentence-raster.pdf` | `bodoni-sentence.json` | Single-font smoke test (Libre Bodoni 400, one sentence) |
| **Bodoni only** | `bodoni-only.pdf` | — | Multi-style Bodoni (regular/bold/italic) |
| **Mixed-font specimen** | `mixed-font-specimen-raster.pdf` | `mixed-font-ground-truth.json` | Intra-line font switching (sans/serif/mono/bold/italic) |

### Quick start

```bash
# Generate the font specimen (downloads fonts on first run)
cd test-docs/
python3 gen-specimen.py

# Generate the resolution degradation series
python3 gen-resolution-series.py

# Run unscan against the standard 300dpi scan
cd ..
cargo run --release -- test-docs/resolution-series/specimen-300dpi.pdf \
    -o /tmp/reconstructed.pdf

# Compare the audit log against ground truth
# (font-timeline-specimen.json has the expected font for each section)
```

### Test tiers

| Tier | Source | What it tests |
|------|--------|---------------|
| **Clean specimen** | `font-timeline-specimen.pdf` | 30 fonts, 500 years, OT variants — the vector ground truth |
| **Bodoni sentence** | `bodoni-sentence-raster.pdf` | Single-font smoke test — must match Libre Bodoni 400 |
| **Resolution series** | `resolution-series/specimen-*.pdf` | Same content at 600→fax dpi — measures degradation tolerance |
| **Historical scans** | `historical/*.pdf` | Real documents from archives — Baskerville's 1757 Virgil, CIA memos, NASA standards |
| **Existing test docs** | `berkeley-acceptance.pdf`, `irs-w4.pdf` | Real-world documents with known fonts |

## OpenType Feature Variant Detection

Scantext's font catalog doesn't just match base fonts — it matches specific
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

### Adding this to your font pipeline

The key code is in `src/font_scan.rs`:

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

And in `src/font_match.rs`:

```rust
// resolve_glyph() checks the override map before falling back to cmap
fn resolve_glyph(font: &FontRef, ch: char, overrides: Option<&[(char, u16)]>)
    -> ab_glyph::GlyphId
{
    if let Some(map) = overrides {
        if let Some(&(_, gid)) = map.iter().find(|(c, _)| *c == ch) {
            return ab_glyph::GlyphId(gid);
        }
    }
    font.glyph_id(ch)
}
```
