# Scantext Test Documents

Ground-truth test corpus for **unscan** — a tool that reconstructs
vector PDFs from scanned/raster originals by identifying fonts, matching
glyphs, and re-setting text as native outlines.

## Philosophy

The test suite works like a round-trip fidelity test:

1. **Start with a known-good vector PDF** (`font-timeline-specimen.pdf`) —
   every font, every glyph, every OpenType feature is documented in a
   machine-readable ground truth file.

2. **Degrade it** through a gauntlet of increasingly hostile scan simulations —
   from 600 dpi archival quality down to 98-dpi fax with 1-bit dithering.

3. **Run unscan** on each degraded version and measure how accurately it
   reconstructs the original vector document.

The goal is **zero information loss at reasonable scan quality** and
**graceful degradation** as conditions worsen. A perfect score means
unscan identified every font (including OpenType variants like old-style
figures), placed every glyph correctly, and produced a PDF that's
byte-for-byte identical in text content to the original.

## Test Corpus Structure

```
test-docs/
├── font-timeline-specimen.pdf          # Clean vector PDF (ground truth)
├── font-timeline-specimen-scanned.pdf  # Simulated scan (skewed, noisy)
├── font-timeline-specimen.json         # Ground truth: font per section + OT features
├── gen-specimen.py                     # Generator for the specimen PDF
├── gen-resolution-series.py            # Generator for the resolution series
│
├── resolution-series/                  # Degradation gauntlet
│   ├── specimen-600dpi.pdf             # Archival — should be near-perfect
│   ├── specimen-300dpi.pdf             # Standard — the baseline target
│   ├── specimen-200dpi.pdf             # MFP default — common in the wild
│   ├── specimen-150dpi.pdf             # Economy — font detection gets hard
│   ├── specimen-100dpi.pdf             # Fax-fine — survival mode
│   ├── specimen-fax-standard.pdf       # Group 3 fax — 204×98 dpi, 1-bit
│   └── manifest.json                   # Metadata for all resolution variants
│
├── historical/                         # Real scanned documents from archives
│   ├── 1545-garamond-estienne-bible.pdf
│   ├── 1734-caslon-type-specimen.pdf
│   ├── 1757-baskerville-virgil.pdf
│   ├── 1776-caslon-dunlap-broadside.pdf
│   ├── 1818-bodoni-manuale-tipografico.pdf
│   ├── 1870s-victorian-slab-serif-specimen.pdf
│   ├── 1923-bauhaus-weimar-catalog.pdf
│   ├── 1961-courier-eisenhower-farewell.pdf
│   ├── 1961-courier-cia-memo.pdf
│   ├── 1970-tnr-irs-1040.pdf
│   ├── 1976-helvetica-nasa-standards.pdf
│   ├── 1999-tnr-irs-w4.pdf
│   ├── 2000-tnr-irs-1040.pdf
│   ├── ground-truth.json
│   └── README.md                       # Provenance, source URLs, confidence
│
├── berkeley-acceptance.pdf             # UC Berkeley admissions letter (Source Serif 4 + onum)
└── irs-w4.pdf                          # IRS W-4 form (mixed TNR/Helvetica)
```

## Generating the Test Documents

### Prerequisites

```bash
# System packages
apt install poppler-utils pango1.0-tools ttf-mscorefonts-installer

# Python packages
pip install Pillow numpy img2pdf

# Fonts — the specimen generator installs these, or get them manually:
# Google Fonts: fonts.google.com (all OFL-licensed)
# Microsoft Core Fonts: apt install ttf-mscorefonts-installer
# URW Base35: ships with ghostscript/texlive on most Linux distros
```

### Step 1: Generate the clean specimen

```bash
cd test-docs/
python3 gen-specimen.py
```

This produces:
- `font-timeline-specimen.pdf` — 6-page vector PDF, 30 font sections spanning
  1530–2020, each rendered in the actual font described. Includes OT variant
  demos (onum, smcp, ss01–ss03, titl) where available.
- `font-timeline-specimen-scanned.pdf` — same content with simulated scan
  artifacts: ~2° skew, Gaussian blur, speckle noise, off-white paper.
  **This is the canonical rasterized input** for accuracy testing (6 pages).
- `font-timeline-specimen-fontmap.json` — maps font names to the exact
  TTF/OTF font file paths on disk. Used by `unscan --include-fontmap` for
  CI audit injection.
- `font-timeline-specimen.json` — machine-readable ground truth mapping each
  section index to its font family, pango font name, source URL, and which
  OpenType features are demonstrated.

The generator downloads fonts from the Google Fonts CDN on first run and
installs them to `/usr/share/fonts/truetype/specimen-fonts/`. Run `fc-cache -fv`
if fonts aren't found.

### Step 2: Generate the resolution degradation series

```bash
python3 gen-resolution-series.py [source.pdf] [output_dir]
# Defaults: font-timeline-specimen.pdf → resolution-series/
```

Produces six PDFs simulating scans at decreasing quality:

| File | DPI | Blur | Noise | JPEG | Dither | Difficulty |
|------|-----|------|-------|------|--------|------------|
| `specimen-600dpi.pdf` | 600 | 0.3px | σ=1.0 | — | — | Easy |
| `specimen-300dpi.pdf` | 300 | 0.7px | σ=1.5 | — | — | Standard |
| `specimen-200dpi.pdf` | 200 | 0.9px | σ=2.0 | Q92 | — | Medium |
| `specimen-150dpi.pdf` | 150 | 1.1px | σ=2.5 | Q85 | — | Hard |
| `specimen-100dpi.pdf` | 100 | 1.4px | σ=3.0 | Q75 | — | Very Hard |
| `specimen-fax-standard.pdf` | 204×98 | 1.6px | σ=4.0 | Q65 | 1-bit F-S | Brutal |

All files share a consistent random skew angle (~2°). A `manifest.json` records
the exact parameters used.

### Step 3 (optional): Fetch historical documents

The `historical/` directory contains real scanned documents from public
archives. See `historical/README.md` for provenance and source URLs.
These are not generated — they're downloaded from Internet Archive, Library
of Congress, IRS, CIA FOIA, NASA, and presidential libraries.

## Evaluating Scantext

### Basic round-trip test

```bash
# Run unscan on each degraded version
for pdf in resolution-series/specimen-*.pdf; do
    name=$(basename "$pdf" .pdf)
    unscan "$pdf" -o "resolution-series/output-${name}.pdf"
done
```

### Comparing against ground truth

The ground truth JSON maps section indices to expected font families:

```json
{
  "sections": [
    {
      "index": 0,
      "era": "c. 1530 — The Garamond",
      "font_family": "EB Garamond",
      "source": "fonts.google.com/specimen/EB+Garamond — OFL, Georg Mayr-Duffner",
      "features_demonstrated": ["onum", "smcp", "ss01"]
    },
    ...
  ]
}
```

Scantext's `--audit` flag produces an audit log JSON with the detected font
for each line. Compare `audit.font_family` against the ground truth to compute
accuracy at each resolution tier.

### What "success" looks like

| Resolution | Target accuracy | Notes |
|------------|----------------|-------|
| 600 dpi | >95% | Near-perfect — if it fails here, the algorithm is wrong |
| 300 dpi | >90% | The standard we optimize for |
| 200 dpi | >80% | Good enough for office document reconstruction |
| 150 dpi | >70% | Acceptable — some rare fonts may be confused |
| 100 dpi | >50% | Survival mode — common fonts should still match |
| Fax | >30% | Heroic — 1-bit dithering destroys most glyph detail |

### Historical documents

Historical scans have no synthetic degradation — they have *real* degradation:
200-year-old ink bleed, uneven paper, binding shadows, foxing stains. These
test unscan's robustness against conditions no simulator can replicate.

See `historical/ground-truth.json` for expected fonts and confidence levels.

## Font Coverage

The specimen includes **30 typefaces** from these sources:

| Source | Fonts | License | How to install |
|--------|-------|---------|----------------|
| Google Fonts | EB Garamond, Libre Caslon Text, Libre Baskerville, Libre Bodoni, Zilla Slab, Jost, Caladea, Roboto, Open Sans, Lato, Merriweather, Source Sans 3, Source Serif 4, Noto Serif, PT Serif, Playfair Display, IBM Plex Sans/Serif/Mono, Inter | OFL / Apache 2.0 | `gen-specimen.py` auto-downloads, or visit fonts.google.com |
| Microsoft Core Fonts | Times New Roman, Courier New, Arial, Georgia, Verdana, Comic Sans MS, Trebuchet MS | Restricted redistribution | `apt install ttf-mscorefonts-installer` |
| URW Base35 | Nimbus Sans (Helvetica clone) | AGPL | Ships with ghostscript |

### OpenType variants tested

The specimen exercises these OT features on fonts that support them:

- **onum** (old-style figures): EB Garamond, Caladea, Roboto, Source Sans 3, Source Serif 4, Noto Serif
- **smcp** (small caps): EB Garamond, Roboto, Source Sans 3, Source Serif 4, Noto Serif, Inter
- **ss01–ss03** (stylistic sets): EB Garamond, Source Sans 3, Source Serif 4, Inter
- **titl** (titling alternates): Source Sans 3

Scantext's font catalog automatically detects and creates variant entries
for every OT feature that produces different glyph shapes. See
`src/font_scan.rs` — the `detect_ot_variants()` function probes 25+ features
using rustybuzz (a pure-Rust harfbuzz port) and emits a separate catalog
entry for each variant that changes at least one Latin glyph.

## Contributing Test Documents

Good test documents have:

1. **Known font provenance** — you can state with confidence what typeface
   was used, either because it's a type specimen, a branded document, or
   a known historical artifact.
2. **Real scan artifacts** — actual scanner output, not synthetic degradation.
3. **Public domain or freely licensed** — government documents, expired
   copyright, Creative Commons, etc.
4. **Latin script** — unscan currently targets Latin-alphabet text.

Add new documents to `historical/` with an entry in `README.md` and
`ground-truth.json`. Include the source URL so others can verify provenance.
