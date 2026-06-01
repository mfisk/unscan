# Font Installation for unscan

`unscan` is a font-accurate OCR-to-vector pipeline. It **must** have the exact fonts installed that appear in test documents, otherwise font matching and accuracy tests will fail.

## Never stub fonts

Do **not** copy DejaVu or other placeholder fonts to satisfy missing font files. The project is about getting fonts correct. Stubbing was previously done to unblock fixture generation, but it caused `t60_specimen_accuracy` to fail with "No font matches found" and invalid vectorization counts.

If a font is missing, install the real font from Google Fonts or the upstream distributor.

## Quick install

Run the automated installer:

```bash
sudo bash scripts/install-all-fonts.sh
```

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

## Specimen font directory

Real fonts are installed to:
- `/usr/share/fonts/truetype/specimen-fonts/` — Google Fonts OFL families used in `gen-specimen.py`
- `/usr/share/fonts/truetype/extra/` — Source Serif 4
- `/usr/local/share/fonts/typewriter/` — Special Elite, Prestige Elite, Letter Gothic
- `/usr/share/fonts/truetype/msttcorefonts/` — Microsoft Core Fonts

## Verification

Check installation:

```bash
bash scripts/check-fonts.sh
```

Expected output: `✓ All required fonts are installed.`

## Why this matters

The test suite includes:
- `t30_regression_ssim` — SSIM comparison against ground truth renders
- `t60_specimen_accuracy` — multi-DPI font accuracy and vectorization counts

Both tests compare OCR output against known font names and metrics. If fonts are stubbed, the visual features don't match and tests fail with:
- `No font matches found in specimen output`
- `could not parse vectorized count from specimen`

Always install real fonts. Never stub.
