# Popular Fonts for unscan

This guide covers the most common font families you'll encounter in scanned
documents, organised by document type. Install the right fonts and unscan
will match them accurately; miss one and that text stays raster.

> **Quick install:** `bash scripts/install-all-fonts.sh` handles everything
> listed here in one shot. See [`FONT_INSTALL.md`](FONT_INSTALL.md) for
> details on what the script does.

> **Reference specimen:** Our 6-page
> [`font-timeline-specimen.pdf`](../test-docs/font-timeline-specimen.pdf)
> covers five centuries of typefaces — 30 fonts from Garamond (c. 1530) to
> Aptos (2023), each rendered in the font it describes. It serves as both a
> visual reference and the ground-truth corpus for accuracy testing.

---

## Microsoft Word Documents

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

---

## Typewriter & Vintage Documents

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

---

## LaTeX Documents

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

---

## Google Fonts & Open-Source Families

The install script downloads these directly from the
[Google Fonts repository](https://github.com/google/fonts):

- **Serif:** EB Garamond, Libre Baskerville, Libre Bodoni, Libre Caslon Text,
  Zilla Slab, Merriweather, PT Serif, Noto Serif, Source Serif 4, Playfair Display
- **Sans-serif:** Roboto, Open Sans, Lato, Inter, Source Sans 3, IBM Plex Sans, Jost
- **Monospace:** IBM Plex Mono
- **Display:** Special Elite (typewriter-style)

These are installed to `~/.local/share/fonts/` preserving the upstream
`ofl/`/`apache/` directory structure.

---

## Font Search Paths

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
