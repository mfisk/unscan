# End-to-End Testing with Ground Truth

unscan's accuracy is measured by comparing its output against a vector PDF whose fonts are known. This document explains the testing pipeline, the fontmap ground truth format, fontconfig resolution pitfalls, and how to avoid them.

## Overview

```
gen-specimen.py
  ├── font-timeline-specimen.pdf          (vector PDF — the source of truth)
  ├── font-timeline-specimen-rasterized.pdf  (rasterized via Ghostscript)
  ├── font-timeline-specimen-fontmap.json (ground truth: font name → file path)
  └── font-timeline-specimen-scanned.pdf  (JBIG2 re-encoded "scanned" version)

unscan ─── rasterized.pdf + fontmap.json → audit.json + SSIM images

char-misses.py ─── audit.json + vector PDF + fontmap.json → HTML miss report
```

1. `gen-specimen.py` builds a multi-page vector PDF where each section uses a specific font family (Regular, Bold, Italic). It simultaneously writes a fontmap that records which font *file* was used for each PDF font name.
2. The vector PDF is rasterized to simulate a scanned document.
3. `unscan` processes the rasterized PDF, producing an audit log with per-line font matches and SSIM scores.
4. `char-misses.py` compares unscan's matches against the vector PDF's embedded font names, using the fontmap to resolve what file each font name should map to.

## The Fontmap

### Format

```json
{
  "EBGaramond": "/usr/share/fonts/truetype/specimen-fonts/eb-garamond-400.ttf",
  "EBGaramond-Bold": "/usr/share/fonts/truetype/specimen-fonts/eb-garamond-700.ttf",
  "EBGaramond-Italic": "/usr/share/fonts/truetype/specimen-fonts/eb-garamond-400i.ttf",
  "SourceSerif4": "/usr/share/fonts/truetype/extra/SourceSerif4[opsz,wght].ttf",
  ...
}
```

Keys are ReportLab registration names (`{Family}`, `{Family}-Bold`, `{Family}-Italic`). Values are absolute filesystem paths to the font files used to generate the PDF.

### How Tools Use It

**`unscan --include-fontmap`**: Injects the fontmap into the audit JSON under `font_file_map`. This gives downstream tools the file-level ground truth without needing to re-derive it from the vector PDF.

**`char-misses.py --fontmap`**: Uses the fontmap to resolve the "expected" font for each line. The vector PDF embeds font *names* (e.g., `SourceSerif4-SemiboldIt`); the fontmap maps the registration name to a *file*. The report compares unscan's matched file against the expected file.

**`gen-specimen.py`**: Generates the fontmap as a side effect of font registration. Every `register_font()` call records the family→file mapping.

### Why File Paths, Not Names

Font names are unreliable. A single font file can declare multiple names (PostScript name, full name, family name, preferred family). The 4-font-family naming model means a font's `subfamily` field might say "Regular" even when its weight is 600. File paths are unambiguous — either unscan found the same file or it didn't.

## Font Resolution via Fontconfig (`fc_find`)

`gen-specimen.py` resolves fonts through fontconfig (`fc-list`), the same system unscan uses at runtime. This is intentional — the ground truth should use the same font resolution infrastructure.

But fontconfig has pitfalls.

### Problem 1: Weight Mismatch (4-Font-Family Naming)

The OpenType spec defines two naming models:

- **4-font-family**: Only Regular, Bold, Italic, Bold Italic subfamilies exist. SemiBold registers as "Regular" under a weight-qualified family name (e.g., "Roboto Medium" with subfamily "Regular").
- **Extended family**: Each weight gets its own subfamily ("SemiBold", "Light", etc.) under the true family name.

When both models are present, fontconfig's `style=Regular` filter matches *all* fonts that declare subfamily "Regular" — including SemiBold, Light, ExtraLight, etc. under their alternate family names.

**Real examples caught in our specimen:**

| Family | `fc_find("X", "Regular")` returned | Actual weight | Should be |
|---|---|---|---|
| Roboto | Roboto-Medium.ttf | 500 | 400 |
| Lato | Lato-Hairline.ttf | 250 | 400 |
| SourceSans3 | SourceSans3-Semibold.ttf | 600 | 400 |
| SourceSerif4 | SourceSerif4-ExtraLight.ttf | 200 | 400 |
| IBMPlexSerif | IBMPlexSerif-Light.ttf | 300 | 400 |
| IBMPlexMono | IBMPlexMono-SemiBold.ttf | 600 | 400 |
| Inter | Inter-Black.otf | 900 | 400 |

**Fix**: `fc_find()` validates `OS/2.usWeightClass` via fontTools for every candidate.

### Problem 2: Width Mismatch (Superfamilies)

Large families like Noto Serif ship dozens of width variants — Condensed, SemiCondensed, ExtraCondensed — all registered under the same family name with `style=Regular` or `style=Bold`.

Fontconfig returns them in arbitrary order. Without width validation, `fc_find("Noto Serif", "Regular")` might return `NotoSerif-ExtraCondensedBold.ttf` or `NotoSerif-CondensedItalic.ttf`.

**Real examples caught:**

| Entry | Returned file | Width class | Should be |
|---|---|---|---|
| NotoSerif-Bold | NotoSerif-ExtraCondensedBold.ttf | 2 (ExtraCondensed) | 5 (Normal) |
| NotoSerif-Italic | NotoSerif-CondensedItalic.ttf | 3 (Condensed) | 5 (Normal) |

**Fix**: `fc_find()` validates `OS/2.usWidthClass == 5` (Normal) for all candidates.

### Problem 3: Italic Weight Mismatch

Same as Problem 1, but for italic styles. `fc_find("Source Serif 4", "Italic")` returned `SourceSerif4-SemiboldIt.ttf` (weight 600) instead of the regular-weight italic (weight 400).

**Real examples caught:**

| Family | `fc_find("X", "Italic")` returned | Actual weight |
|---|---|---|
| SourceSerif4 | SourceSerif4-SemiboldIt.ttf | 600 |
| IBMPlexMono | IBMPlexMono-LightItalic.ttf | 300 |
| IBMPlexSans | IBMPlexSans-LightItalic.ttf | 300 |

**Fix**: `fc_find()` maps `"Italic"` → expected weight 400, `"Bold Italic"` → expected weight 700.

### Problem 4: Variable Font Axis Defaults

Variable fonts store a default weight in both `OS/2.usWeightClass` and the `fvar` table's `wght` axis default. Some variable fonts set this to their lightest instance, not 400:

| Font file | OS/2 weight | fvar default | Axis range |
|---|---|---|---|
| SourceSans3[wght].ttf | 200 | 200 | 200–900 |
| Merriweather[opsz,wdth,wght].ttf | 300 | 300 | 300–900 |
| PlayfairDisplay[wght].ttf | 400 | 400 | 400–900 |

When `fc_find` checks `usWeightClass == 400`, SourceSans3's variable font fails (it reports 200). This is actually beneficial — it forces selection of the static instance (`SourceSans3-Regular.ttf`, weight 400) when available. Static fonts render more predictably in ReportLab.

**Current behavior**: Variable fonts with wrong-default weights are deprioritized. If no static alternative exists (e.g., PlayfairDisplay-Bold), the variable font is accepted as a fallback. This is a known limitation — the specimen will be generated with the variable font's default weight, not the requested weight.

**Mitigation**: `fc_find()` adds a +10 score bonus for static fonts over variable fonts when both satisfy the weight constraint.

### Problem 5: Missing Styles

Some fonts only ship one weight (SpecialElite, PrestigeElite) or are missing specific styles on the system (PT Serif has no Bold installed). When `fc_find("X", "Bold")` finds nothing matching weight 700, it falls back to the Regular file.

This is expected and correct — the specimen uses what's available, and the fontmap records what was actually used. The miss report will show these as "correct" because unscan is matching against the actual file used, not an ideal file that doesn't exist.

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

## Running Tests

### Full Pipeline

```bash
# 1. Regenerate specimen (if fonts or gen-specimen.py changed)
cd test-docs && python3 gen-specimen.py

# 2. Build unscan
cd .. && cargo build --release

# 3. Run unscan with audit + fontmap
./target/release/unscan test-docs/font-timeline-specimen-rasterized.pdf \
  -o /tmp/out.pdf \
  --audit /tmp/audit \
  --include-fontmap test-docs/font-timeline-specimen-fontmap.json

# 4. Generate miss report
python3 tools/char-misses.py /tmp/audit/audit.json \
  test-docs/font-timeline-specimen.pdf \
  --fontmap test-docs/font-timeline-specimen-fontmap.json \
  -o /tmp/misses.html
```

### Reading the Miss Report

The HTML report shows every line where unscan's font match disagrees with the vector PDF's font. For each miss:

- **Page/Line**: Location in the document
- **Text**: The OCR'd text content
- **Expected**: Font from the vector PDF (resolved via fontmap to a file path)
- **Got**: Font unscan matched (file path)
- **SSIM images**: Side-by-side scan crop, render crop, and diff — showing exactly what the SSIM comparison evaluated

### Accuracy Tracking

The summary line `Total: N  Hits: H  Misses: M  Skipped: S` is the headline metric. Track `H/N` as the accuracy percentage across changes.

**What counts as a hit**: unscan's matched font file == fontmap's expected font file (or their filenames match after stripping paths).

**What counts as a miss**: Different file. This includes cases where unscan found a close variant (e.g., Regular instead of Light) — the specimen generator should be using the exact file that unscan would realistically encounter.

**What counts as skipped**: Lines where the vector PDF's font couldn't be mapped through the fontmap (usually OCR artifacts or lines that span multiple fonts).

### Audit Images

The `--audit` directory contains per-line SSIM comparison images:

```
/tmp/audit/
  audit.json              # Structured results
  page_1/
    line_000/
      ssim_scan.png       # Scan crop (word-union bbox from OCR)
      ssim_render.png     # Render crop (ink-extent, tight to glyphs)
      ssim_diff.png       # Absolute difference between displayed scan/render
    line_001/
      ...
```

- **ssim_scan.png**: Cropped from the rasterized input using the union of OCR word bounding boxes. This avoids ink bleed from adjacent lines.
- **ssim_render.png**: The matched font rendered at the same size, then cropped to ink extent (threshold 240). Shows only where glyphs actually produced ink.
- **ssim_diff.png**: Pixel-level absolute difference between the displayed scan and render. Bright pixels = high disagreement.

SSIM scoring uses `ssim_windowed_best_vshift` with ±12px vertical shift to handle baseline alignment differences between OCR bbox placement and rendered glyph positioning.

## When to Regenerate the Specimen

Regenerate (`python3 test-docs/gen-specimen.py`) when:

- `gen-specimen.py` changes (font resolution, layout, content)
- New fonts are installed on the system
- Font packages are updated (new versions may change metrics or file paths)
- You suspect fontmap corruption (run the validation check below)

### Validating the Fontmap

```bash
python3 -c "
import json
from fontTools.ttLib import TTFont

fm = json.load(open('test-docs/font-timeline-specimen-fontmap.json'))
issues = 0
for k in sorted(fm):
    path = fm[k]
    try:
        tt = TTFont(path)
        wt = tt['OS/2'].usWeightClass
        wd = tt['OS/2'].usWidthClass
        tt.close()
        expected_wt = 700 if 'Bold' in k else 400
        if abs(wt - expected_wt) > 50 or wd != 5:
            issues += 1
            print(f'  {k}: wt={wt} wd={wd} ({path.split(\"/\")[-1]})')
    except Exception as e:
        issues += 1
        print(f'  {k}: {e}')

print(f'{len(fm)} entries, {issues} with unexpected weight/width')
"
```

Expected: single-weight fonts (SpecialElite, PrestigeElite) and variable fonts with non-400 defaults (SourceSans3, Merriweather) will show as "issues" — these are known and documented above.
