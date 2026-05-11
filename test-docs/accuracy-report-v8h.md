# Unscan v8h Accuracy Report — Per-Stage Analysis

**Date:** 2026-05-10
**Test document:** specimen-clean-raster.pdf (page 1, 94 OCR lines)
**Ground truth:** font-timeline-specimen.json (5 sections visible on page 1 + title)

## Key Finding

**The char index is not pulling its weight.** It found the correct font in only 11.2% of lines,
while the coarse scorer found it in 34.8%. The char index's unique contribution was just 4 lines
where it found the font and coarse missed it — but coarse uniquely found the font in 25 lines.

The char index adds ~4 minutes of runtime (SSIM-ranking ~80 candidates instead of 30) for
minimal accuracy benefit. It needs significant improvement before it earns its compute cost.

## Aggregate Statistics

| Stage | Found / Total | Recall |
|-------|--------------|--------|
| Char Index Top-50 (logged top-10) | 10/89 | 11.2% |
| Coarse Top-30 | 31/89 | 34.8% |
| Union (either) | 35/89 | 39.3% |
| Final SSIM Correct | 30/89 | 33.7% |

### Unique Contributions
- Char index found the right font but coarse missed it: **4 lines**
- Coarse found the right font but char index missed it: **25 lines**
- Both found it: **6 lines**
- Neither found it: **54 lines** (correct font not in top-N of either stage)

### Important Caveats
1. We only logged the char-index top-10, but it returns top-50. The correct font
   may appear at ranks 11-50 more often than shown.
2. The "correct font" matching is by family name only (e.g., "EB Garamond" matches
   "EBGaramond12 Regular [ss02]"). Some matches are to different weights/optical sizes.
3. Lines 9-10 ("4", "Be)") are OCR fragments, not real text lines.
4. Many lines are italic or bold variants; the char index may struggle with these
   since it indexes regular weight primarily.

## Per-Section Breakdown

| Section | CharIdx | Coarse | SSIM Correct | Notes |
|---------|---------|--------|-------------|-------|
| Title (Playfair Display) | 1/3 (33%) | 2/3 (67%) | 2/3 (67%) | Italic subtitle mismatches |
| Garamond (EB Garamond) | 0/19 (0%) | 5/19 (26%) | 2/19 (11%) | Worst section — many serif confusions |
| Caslon (Libre Caslon Text) | 0/17 (0%) | 3/17 (18%) | 4/17 (24%) | Confused with P052, FreeSerif, Merriweather |
| Baskerville (Libre Baskerville) | 2/19 (11%) | 5/19 (26%) | 8/19 (42%) | SSIM rescues some from union pool |
| Bodoni (Libre Bodoni) | 6/19 (32%) | 11/19 (58%) | 12/19 (63%) | Best section — high contrast is distinctive |
| Slab Serif (Zilla Slab) | 1/14 (7%) | 5/14 (36%) | 2/14 (14%) | Slab trait not captured well |

## Analysis

### Why the char index underperforms

1. **Feature vector doesn't capture serif/contrast well enough for discrimination.**
   Garamond, Caslon, Baskerville, and Bodoni are all serifs. The 32-bin column density
   profile sees them as similar. Only Bodoni (high stroke contrast) separates clearly.

2. **The char index extracts chars from the longest OCR words.** Many short lines
   (section headings, attribution lines) don't yield enough characters for reliable matching.

3. **Italic/bold variants confuse extraction.** The char index renders chars in regular
   weight but scans may contain bold/italic text. The shape mismatch tanks cosine scores.

4. **Cosine scores are very high and tightly clustered.** Most char-index scores are 0.88-0.99,
   meaning all serif fonts look nearly identical to the feature extractor. The top-50 cutoff
   may be arbitrary when scores differ by < 0.01.

### Why coarse scoring does better

The coarse scorer uses IoU, NCC, Hu moments, and fill ratio on the full rendered line —
it captures global shape characteristics that the per-character approach misses:
- Overall line weight and density
- Word spacing and kerning patterns  
- Full-word silhouette matching

### Where the char index helps (4 unique saves)

Lines 52, 55, 61, 69: all Baskerville or Bodoni body text. The char index found the
correct font when the coarse scorer ranked it outside top-30. This suggests the char
index IS better at discriminating within the high-contrast serif space.

## Recommendations

1. **Don't remove the char index** — it does provide some unique value, especially for
   high-contrast serifs. But its current recall is too low to justify the runtime cost.

2. **Priority fixes:**
   - Improve serif sub-classification in the feature vector
   - Index bold/italic variants separately (currently indexes regular only)
   - Use stroke contrast as a hard pre-filter (slab vs modern vs transitional vs old-style)
   - Consider expanding char-index to top-100 or top-200 (scores are tightly clustered)

3. **Runtime optimization:** If the union approach adds ~50 candidates to SSIM reranking,
   and each costs ~5ms, that's ~250ms per line × 94 lines = ~24s overhead. Not 4 minutes.
   Investigate why it actually takes 4+ minutes extra.

## Appendix: Per-Line Detail

| # | Section | Expected | CharIdx | Coarse | SSIM Winner | Score | OK |
|---|---------|----------|---------|--------|-------------|-------|----|
| 0 | Title | Playfair Display | #1 | #1 | playfair display 400 [lnum] | 0.440 | ✓ |
| 1 | Title | Playfair Display | — | #1 | playfair display 400 [lnum] | 0.574 | ✓ |
| 2 | Title | Playfair Display | — | — | NotoSansDisplay MediumItalic [c2sc] | 0.391 | ✗ |
| 6 | Garamond | EB Garamond | — | — | P052 Bold [hist] | 0.294 | ✗ |
| 7 | Garamond | EB Garamond | — | — | C059 Italic [hist] | 0.255 | ✗ |
| 8 | Garamond | EB Garamond | — | — | FreeSerifItalic [onum] | 0.451 | ✗ |
| 9 | Garamond | EB Garamond | — | — | (none) | 0.000 | ✗ |
| 10 | Garamond | EB Garamond | — | — | (none) | 0.000 | ✗ |
| 11 | Garamond | EB Garamond | — | — | NotoSerif Condensed [onum] | 0.292 | ✗ |
| 12 | Garamond | EB Garamond | — | #1 | FreeSerif [hist] | 0.349 | ✗ |
| 13 | Garamond | EB Garamond | — | #1 | EBGaramond12 Regular [ss02] | 0.817 | ✓ |
| 14 | Garamond | EB Garamond | — | — | EBGaramond08 Regular [ss02] | 0.677 | ✓ |
| 15 | Garamond | EB Garamond | — | — | P052 Roman [hist] | 0.398 | ✗ |
| 16 | Garamond | EB Garamond | — | — | NotoSerifDisplay Condensed [onum] | 0.390 | ✗ |
| 17 | Garamond | EB Garamond | — | #2 | FreeSerif [ss10] | 0.275 | ✗ |
| 18 | Garamond | EB Garamond | — | — | libre bodoni 700 | 0.494 | ✗ |
| 19 | Garamond | EB Garamond | — | — | libre bodoni 700 | 0.551 | ✗ |
| 20 | Garamond | EB Garamond | — | #3 | libre bodoni 400i | 0.345 | ✗ |
| 21 | Garamond | EB Garamond | — | — | libre bodoni 700 | 0.551 | ✗ |
| 22 | Garamond | EB Garamond | — | — | Georgia | 0.441 | ✗ |
| 23 | Garamond | EB Garamond | — | #2 | libre bodoni 400 | 0.426 | ✗ |
| 24 | Garamond | EB Garamond | — | — | libre baskerville 700 | 0.311 | ✗ |
| 25 | Caslon | Libre Caslon Text | — | — | P052 Bold [hist] | 0.427 | ✗ |
| 26 | Caslon | Libre Caslon Text | — | — | merriweather 400 | 0.359 | ✗ |
| 27 | Caslon | Libre Caslon Text | — | — | libre caslon text 700 | 0.390 | ✓ |
| 28 | Caslon | Libre Caslon Text | — | — | NimbusRoman Regular | 0.555 | ✗ |
| 29 | Caslon | Libre Caslon Text | — | — | merriweather 400 | 0.288 | ✗ |
| 30 | Caslon | Libre Caslon Text | — | — | FreeSerif [onum] | 0.574 | ✗ |
| 31 | Caslon | Libre Caslon Text | — | #1 | libre caslon text 400 | 0.596 | ✓ |
| 32 | Caslon | Libre Caslon Text | — | #1 | libre caslon text 400 | 0.576 | ✓ |
| 33 | Caslon | Libre Caslon Text | — | — | P052 Roman | 0.427 | ✗ |
| 34 | Caslon | Libre Caslon Text | — | #1 | libre caslon text 400 | 0.595 | ✓ |
| 35 | Caslon | Libre Caslon Text | — | — | C059 Roman [hist] | 0.470 | ✗ |
| 36 | Caslon | Libre Caslon Text | — | — | libre baskerville 700 | 0.764 | ✗ |
| 37 | Caslon | Libre Caslon Text | — | — | libre bodoni 400i | 0.345 | ✗ |
| 38 | Caslon | Libre Caslon Text | — | — | libre bodoni 700 | 0.551 | ✗ |
| 39 | Caslon | Libre Caslon Text | — | — | Georgia | 0.441 | ✗ |
| 40 | Caslon | Libre Caslon Text | — | — | P052 Roman | 0.548 | ✗ |
| 41 | Caslon | Libre Caslon Text | — | — | DejaVuSerif [salt] | 0.205 | ✗ |
| 42 | Baskerville | Libre Baskerville | — | #1 | libre baskerville 700 | 0.577 | ✓ |
| 43 | Baskerville | Libre Baskerville | — | — | InterDisplay LightItalic [smcp] | 0.145 | ✗ |
| 44 | Baskerville | Libre Baskerville | — | #2 | libre caslon text 700 | 0.390 | ✗ |
| 45 | Baskerville | Libre Baskerville | — | — | libre baskerville 400 | 0.638 | ✓ |
| 46 | Baskerville | Libre Baskerville | — | #1 | libre baskerville 400 | 0.772 | ✓ |
| 47 | Baskerville | Libre Baskerville | — | #1 | libre baskerville 400 | 0.693 | ✓ |
| 48 | Baskerville | Libre Baskerville | — | — | libre baskerville 700 | 0.518 | ✓ |
| 49 | Baskerville | Libre Baskerville | — | — | SourceSerif4SmText Regular [onum] | 0.628 | ✗ |
| 50 | Baskerville | Libre Baskerville | — | — | libre baskerville 700 | 0.593 | ✓ |
| 51 | Baskerville | Libre Baskerville | — | — | P052 Roman | 0.497 | ✗ |
| 52 | Baskerville | Libre Baskerville | #3 | — | libre baskerville 400 | 0.637 | ✓ |
| 53 | Baskerville | Libre Baskerville | — | — | SourceSerif4SmText Regular [onum] | 0.643 | ✗ |
| 54 | Baskerville | Libre Baskerville | — | — | C059 Roman [hist] | 0.470 | ✗ |
| 55 | Baskerville | Libre Baskerville | #1 | — | libre baskerville 700 | 0.764 | ✓ |
| 56 | Baskerville | Libre Baskerville | — | — | libre bodoni 400i | 0.345 | ✗ |
| 57 | Baskerville | Libre Baskerville | — | — | Caladea Italic | 0.443 | ✗ |
| 58 | Baskerville | Libre Baskerville | — | #1 | EBGaramond08 Regular [lnum] | 0.409 | ✗ |
| 59 | Baskerville | Libre Baskerville | — | — | P052 Roman | 0.548 | ✗ |
| 60 | Baskerville | Libre Baskerville | — | — | DejaVuSerif [salt] | 0.205 | ✗ |
| 61 | Bodoni | Libre Bodoni | #1 | — | libre bodoni 700 | 0.647 | ✓ |
| 62 | Bodoni | Libre Bodoni | — | — | URWBookman LightItalic | 0.251 | ✗ |
| 63 | Bodoni | Libre Bodoni | — | — | libre baskerville 400 | 0.331 | ✗ |
| 64 | Bodoni | Libre Bodoni | #5 | #1 | libre bodoni 400 | 0.663 | ✓ |
| 65 | Bodoni | Libre Bodoni | — | #1 | libre bodoni 400 | 0.782 | ✓ |
| 66 | Bodoni | Libre Bodoni | #1 | #2 | libre bodoni 400 | 0.516 | ✓ |
| 67 | Bodoni | Libre Bodoni | — | #1 | libre bodoni 400 | 0.526 | ✓ |
| 68 | Bodoni | Libre Bodoni | — | #3 | libre bodoni 400 | 0.699 | ✓ |
| 69 | Bodoni | Libre Bodoni | #4 | — | libre bodoni 400 | 0.623 | ✓ |
| 70 | Bodoni | Libre Bodoni | — | #1 | libre bodoni 400 | 0.591 | ✓ |
| 71 | Bodoni | Libre Bodoni | #1 | #1 | libre bodoni 400 | 0.485 | ✓ |
| 72 | Bodoni | Libre Bodoni | — | — | libre baskerville 700 | 0.491 | ✗ |
| 73 | Bodoni | Libre Bodoni | #1 | #1 | libre bodoni 700 | 0.494 | ✓ |
| 74 | Bodoni | Libre Bodoni | — | #1 | libre bodoni 700 | 0.551 | ✓ |
| 75 | Bodoni | Libre Bodoni | — | #1 | libre bodoni 400i | 0.345 | ✓ |
| 76 | Bodoni | Libre Bodoni | — | #4 | Caladea Italic | 0.424 | ✗ |
| 77 | Bodoni | Libre Bodoni | — | — | EBGaramond08 Regular [lnum] | 0.409 | ✗ |
| 78 | Bodoni | Libre Bodoni | — | — | P052 Roman | 0.548 | ✗ |
| 79 | Bodoni | Libre Bodoni | — | — | DejaVuSerif [salt] | 0.205 | ✗ |
| 80 | Slab Serif | Zilla Slab | — | — | Inter ExtraBold [ss01] | 0.312 | ✗ |
| 81 | Slab Serif | Zilla Slab | — | — | NotoSerif MediumItalic [c2sc] | 0.346 | ✗ |
| 82 | Slab Serif | Zilla Slab | — | #2 | zilla slab 700 [lnum] | 0.529 | ✓ |
| 83 | Slab Serif | Zilla Slab | — | #3 | C059 Roman [hist] | 0.518 | ✗ |
| 84 | Slab Serif | Zilla Slab | — | — | zilla slab 700 [lnum] | 0.486 | ✓ |
| 85 | Slab Serif | Zilla Slab | — | #1 | P052 Roman | 0.384 | ✗ |
| 86 | Slab Serif | Zilla Slab | — | #3 | libre caslon text 700 | 0.557 | ✗ |
| 87 | Slab Serif | Zilla Slab | — | — | NotoSans ExtraBold [onum] | 0.439 | ✗ |
| 88 | Slab Serif | Zilla Slab | — | — | InterDisplay Italic [smcp] | 0.583 | ✗ |
| 89 | Slab Serif | Zilla Slab | #5 | #5 | libre baskerville 700 | 0.436 | ✗ |
| 90 | Slab Serif | Zilla Slab | — | — | InterDisplay Italic [smcp] | 0.583 | ✗ |
| 91 | Slab Serif | Zilla Slab | — | — | Georgia | 0.441 | ✗ |
| 92 | Slab Serif | Zilla Slab | — | — | P052 Roman | 0.548 | ✗ |
| 93 | Slab Serif | Zilla Slab | — | — | DejaVuSerif [salt] | 0.205 | ✗ |
