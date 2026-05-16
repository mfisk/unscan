# Text Matching Approach

How unscan identifies fonts and places vector text to match the original scan.

## Pipeline Overview

```
Scanned PDF
  → Rasterize page (pdftoppm at configured DPI)
  → OCR (Tesseract HOCR) → words with bounding boxes
  → Char Index search → candidate fonts per character
  → Word-level SSIM reranking → winning font per line
  → PDF output → subsetted font + positioned words
```

## 1. Character Index (Coarse Filter)

**File:** `src/char_index.rs`

Each installed font (~4000+) is rendered at a reference size for ~70 printable characters. A 59-dimensional feature vector is extracted per glyph capturing stroke widths, serif presence, x-height ratio, stress angle, and other geometric properties.

At query time, each OCR'd character's crop is feature-extracted and compared against the index via brute-force linear scan (flat `Vec` per character, not a KD-tree — at 59 dims KD-trees provide no pruning benefit). All fonts within `factor²` of the nearest distance are returned as candidates.

**Key design choice:** The char index is a *coarse filter* — it narrows ~4000 fonts to ~50 candidates. It does not need to be precise. Precision comes from word-level SSIM.

## 2. Word-Level SSIM Reranking (Precision Filter)

**File:** `src/word_match.rs`

For each candidate font from the char index, every OCR word is:
1. Cropped from the scan image
2. Rendered in the candidate font at the width-matched em size
3. Both images are whitespace-trimmed (all 4 edges) and resized to match
4. Compared via SSIM (structural similarity)

The font with the most word-level SSIM "wins" (majority vote across words). This is where close cousins (EB Garamond vs Libre Caslon, Bodoni vs SourceSerif4) get discriminated.

**Key design choices:**
- `trim_whitespace()` trims all 4 edges of both crop and render to align ink regions. This was the single biggest accuracy improvement — words like "brown" jumped from 0.918 to 0.977 SSIM.
- Render left-padding: glyphs with descenders extending left of origin (like 'j') get a first pass to find `min_px_x`, then all rendering is offset rightward. Fixed "jumps" from 0.745 to 0.963.
- `MAX_RERANK_CANDIDATES = 50` to prevent font family flooding (e.g. SourceSerif4 has many OT variants that can swamp the top-k).
- Font family dedup via `font_family_key()` collapses optical sizes / weight / style variants to one candidate per family.

## 3. Font Size Determination

**File:** `src/pdf_out.rs`, `src/layout.rs`

Font size is determined from the OCR bounding box **height**, not width:

```
em_px = bbox_height * reference_em / font_ink_height
```

Where `font_ink_height = ascent - descent` at the reference em size.

**Why height, not width:** Width-matching (`width_matched_em_px`) computes a font size that makes rendered text width equal the OCR bbox width. If the OCR bbox is even slightly tight or loose, this distorts natural letter spacing — characters get compressed or stretched. Height is more stable because OCR line heights are consistent and don't depend on word-level bbox precision.

Width-matching is still used in word_match.rs for SSIM comparison (where matching the crop dimensions is the goal), but not for final PDF output sizing.

## 4. PDF Text Placement

**File:** `src/pdf_out.rs`

Each word is placed at its OCR x-position with a uniform font size for the line:

- One `em_px` computed from line bbox height (shared by all words in the line)
- One baseline computed via `ink_centered_baseline_pt()` (shared by all words)
- Each word gets its own `BT/Tf/Td/Tj/ET` block at its OCR x-coordinate
- No character spacing adjustments (Tc), no horizontal scaling — natural font metrics only

The font is subsetted to contain only the glyphs actually used (via the `subsetter` crate), typically reducing a 50-200KB font to 2-5KB.

## 5. Raster Handling

**File:** `src/main.rs`, `src/color.rs`

After vectorizing text, the corresponding regions are erased from the raster image (filled with background color). The remaining raster is split into fragments via a cell-based content detector:

- 100px cells are classified as interesting/blank via `region_has_content()`
- Adjacent interesting cells are flood-fill grouped into fragments
- Each fragment is independently embedded in the output PDF
- Fully blank pages produce zero raster fragments (no wasted bytes)

Source image encoding is preserved where possible — JPEG passthrough avoids re-encoding when nothing was vectorized on a page.

## 6. What's NOT Done

- **No character spacing adjustment:** We don't set PDF Tc (character spacing) or Tz (horizontal scaling). The font's built-in metrics and kerning are trusted.
- **No verify stage:** There was previously a second SSIM check (using FreeType) after font selection. This was redundant with word-level SSIM and used a different renderer (FreeType vs ab_glyph), causing inconsistent results. Removed.
- **No ML classifier:** With ~4000 font classes and 1 training sample per class, nearest-neighbor is the right approach. A random forest module exists (`src/ml_forest.rs`) but is not integrated.

## Design Principles

1. **Fix inputs, not algorithms.** Every SSIM failure traced to bad inputs (illustration contamination, garbage OCR words, wrong crop geometry), not bad math.
2. **Coarse then precise.** Char index narrows cheaply; word SSIM discriminates accurately.
3. **Natural font metrics.** Don't distort spacing to match OCR bboxes. Place words at OCR positions, let the font's natural advance widths handle letter spacing.
4. **Smaller outputs.** Font subsetting + blank raster elimination should produce outputs smaller than inputs for text-heavy pages.

## Known Weaknesses

- **Baskerville vs EB Garamond**: Body text in Baskerville sections sometimes matches EB Garamond. These are historically related faces with similar proportions. The char index doesn't separate them reliably enough for word SSIM to differentiate.
- **Short fragments**: Single-word line wraps ("dogs.") match poorly — too little signal for SSIM voting.
- **Small/watermark text**: Attribution lines and "Font:" labels at low point sizes get unreliable OCR, leading to poor matches.
- **Paragraph regrouping**: Still uses a FreeType verify call to decide whether minority lines should switch to the majority font. This is the last remaining verify.rs dependency — could be replaced with word-match SSIM.
