# Font Installation for unprint

`unprint` is a font-accurate OCR-to-vector pipeline. It **must** have the exact fonts installed that appear in test documents, otherwise font matching and accuracy tests will fail.

## Never stub fonts

Do **not** copy DejaVu or other placeholder fonts to satisfy missing font files. The project is about getting fonts correct. Stubbing was previously done to unblock fixture generation, but it caused `t60_specimen_accuracy_aa` to fail with "No font matches found" and invalid vectorization counts.

If a font is missing, install the real font from Google Fonts or the upstream distributor.

## Quick install

Run the automated installer:

```bash
sudo bash tools/install-all-fonts.sh
```

### Install routes

The installer has two routes, selected by a single flag:

- Default (no flag): fonts from apt packages on Linux or Homebrew casks on
  macOS, plus direct downloads for fonts with no package.
- `--home` (Linux only): every manifest font is downloaded into
  `~/.local/share/fonts` instead of apt — for VMs where apt state is wiped
  by reboots but `$HOME` persists. Prerequisites (tesseract, weasyprint,
  Python packages) still install via apt.

On the Hatch VM, always run with `--home` (this is what `check-fonts.sh`
passes when fonts are missing).

This script:
1. Installs apt packages (`ttf-mscorefonts-installer`, `fonts-crosextra-caladea`, `fonts-liberation`, etc.)
2. Installs Microsoft Core Fonts via `cabextract` (EULA workaround)
3. Downloads Google Fonts specimen families directly from `https://raw.githubusercontent.com/google/fonts/main/...`
   - EB Garamond, Libre Baskerville, Libre Bodoni, Libre Caslon Text
   - Zilla Slab, Jost, Playfair Display
   - Roboto, Open Sans, Lato, Merriweather, Source Sans 3, Noto Serif, PT Serif
   - IBM Plex Sans/Serif/Mono, Inter, Source Serif 4
   - Special Elite (Apache)
4. Installs typewriter fonts (Prestige Elite, Letter Gothic)
5. Refreshes font cache
6. Generates `font-timeline-specimen.pdf` with real fonts

## Manifest coverage

Every family in the specimen ground truth is covered by the manifest, the
Microsoft Core Fonts section, or the typewriter section. The manifest also
covers the common families real documents use, in both install routes (apt /
Homebrew and `--home` downloads), including:

- DejaVu Sans / Serif / Sans Mono (the Linux default family; the `--home`
  route extracts them from the Debian packages, since upstream publishes no
  raw-file download — the Condensed variants and DejaVu Math TeX Gyre are
  deliberately not extracted)
- Liberation Serif and Liberation Mono (the metric clones of Times New Roman
  and Courier New; Liberation Sans was already covered)
- Noto Sans Mono (the monospace companion to Noto Sans / Noto Serif)
- Latin Modern Mono and Latin Modern Sans (the TeX default family, alongside
  Latin Modern Roman; Latin Modern Mono has no bold weight)
- Roboto Condensed and Inter Display (the width/display generations of
  Roboto and Inter)
- TeX Gyre Heros and TeX Gyre Pagella (the Helvetica and Palatino
  generations of the TeX Gyre family, alongside TeX Gyre Termes)

Deliberately not in the manifest:

- Noto CJK, Noto emoji, Noto UI — no canonical upstream tarball exists for
  the `--home` route
- TeX Gyre Cursor, Schola, Bonum, Adventor, Chorus — rarely used; the URW
  Base 35 section already installs near-twins (Nimbus Mono PS, C059, URW
  Bookman, URW Gothic, Z003)
- Source Sans Pro / Source Serif Pro — superseded generations; Adobe's
  current distribution is Source Sans 3 / Source Serif 4, and the Pro
  releases are only available as stale release zips
- Calibri, Cambria, Consolas, and other proprietary Microsoft / Apple
  families — no free source; Carlito and Caladea are the metric clones
- Arimo, Tinos, Cousine — metric clones of Arial, Times New Roman, and
  Courier New, which are already covered

## Specimen font directory

Real fonts are installed to:
- `/usr/share/fonts/truetype/specimen-fonts/` — Google Fonts OFL families used in `gen-specimen.py`
- `/usr/share/fonts/truetype/extra/` — Source Serif 4
- `/usr/local/share/fonts/typewriter/` — Special Elite, Prestige Elite, Letter Gothic
- `/usr/share/fonts/truetype/msttcorefonts/` — Microsoft Core Fonts

## Installation

```bash
./tools/install-all-fonts.sh
```

## Why this matters

The test suite includes:
- `t30_regression_ssim` — SSIM comparison against ground truth renders
- `t60_specimen_accuracy_aa` — multi-DPI font accuracy and vectorization counts

Both tests compare OCR output against known font names and metrics. If fonts are stubbed, the visual features don't match and tests fail with:
- `No font matches found in specimen output`
- `could not parse vectorized count from specimen`

Always install real fonts. Never stub.
