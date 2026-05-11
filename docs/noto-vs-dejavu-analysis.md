# NotoSansSymbols vs DejaVu Sans — Index Discrimination Analysis

## TL;DR

**The original "NotoSansSymbols beats DejaVu Sans" result was a test bug — the test searched for `"DejaVu Sans"` (with space) but the full index stores it as `"DejaVuSans"` (no space, from filename).** With the corrected name, DejaVuSans ranks #1-#2 overall (tied with its `[salt]` variant at 0.9937).

However, the investigation still revealed important architectural findings about the index's feature discrimination.

## Corrected Full-Index Results

With 4714 fonts:
- **Overall #1**: DejaVuSans [salt] (0.9937)
- **Overall #2**: DejaVuSans (0.9937)
- DejaVuSans is #1-#4 on every individual character except 's' (where it doesn't appear — likely excluded by the k-d tree radius)

### Per-character breakdown (DejaVu Sans vs NotoSansSymbols)

| Char | DejaVu Rank | DejaVu Score | NotoSymbols Rank | NotoSymbols Score |
|------|-------------|--------------|------------------|-------------------|
| h | #1 | 0.9957 | not in top results | — |
| a | #2 | 0.9933 | not in top results | — |
| m | #1 | 0.9966 | not in top results | — |
| b | #2 | 0.9922 | not in top results | — |
| u | #1 | 0.9970 | #3 | 0.9951 |
| r | #2 | 0.9961 | not in top results | — |
| g | #1 | 0.9926 | not in top results | — |
| e | #4 | 0.9937 | not in top results | — |
| f | #2 | 0.9960 | not in top results | — |
| o | #1 | 0.9940 | not in top results | — |
| n | #2 | 0.9960 | not in top results | — |
| t | #3 | 0.9895 | not in top results | — |
| s | N/A | — | not in top results | — |
| i | #2 | 0.9966 | not in top results | — |
| v | #1 | 0.9889 | not in top results | — |

**NotoSansSymbols only appeared once** (rank #3 on 'u' at 0.9951). The k-d tree search with tolerance bands correctly excludes it for most characters.

## Glyph-Level Comparison

Despite having different names, NotoSansSymbols includes real Latin glyphs. Fonttools analysis shows:

### Font Metrics
| Property | DejaVu Sans | NotoSansSymbols |
|----------|-------------|-----------------|
| UPM | 2048 | 1000 |
| Weight class | 400 | 400 |
| Width class | 5 | 5 |
| PANOSE weight | 6 | 5 |
| PANOSE proportion | 3 | 2 |
| hhea ascender | 1901 | 1480 |
| hhea descender | -483 | -570 |

### Glyph Dimensions (normalized to 1000 UPM)
| Char | DejaVu w×h | NotoSymbols w×h | Aspect DV | Aspect NS |
|------|-----------|-----------------|-----------|-----------|
| h | 458×760 | 452×760 | 0.603 | 0.595 |
| a | 462×574 | 434×555 | 0.804 | 0.782 |
| g | 489×768 | 475×786 | 0.636 | 0.604 |
| e | 507×574 | 458×556 | 0.883 | 0.824 |
| m | 798×560 | 769×546 | 1.425 | 1.408 |
| o | 502×574 | 496×556 | 0.874 | 0.892 |
| v | 532×547 | 508×536 | 0.973 | 0.948 |

Glyphs are similar but NOT identical. Width ratios range from 0.86-1.01× between fonts. The features DO distinguish them — DejaVuSans consistently scores ~0.002 higher than NotoSansSymbols.

## Visual Comparison (Pillow rendering)

Python/Pillow rendering showed dramatically different widths between the two fonts, with NotoSansSymbols appearing 50-100% wider. This was a **Pillow rendering artifact** — Pillow's text layout engine applies different scaling than ab_glyph's raw glyph-unit rendering. The actual font-unit glyph proportions are within a few percent.

Cosine similarity from Python feature extraction: 0.88-0.99 (much lower than Rust's 0.99+).
This confirms the Rust renderer produces more consistent feature vectors between similar fonts.

## Remaining Score Compression Problem

Even with the k-d tree, scores are extremely compressed:
- DejaVuSans: 0.9937 (winner)
- lato 400: 0.9948 on 'e' (beats DejaVu on that char)
- 50+ fonts score above 0.99 on individual chars

### Most Discriminating Features (from k-d tree diagnostics)

For **'e'**:
1. serif_score (dim 36): σ = 0.0494 ← **most useful**
2. profile[10]: σ = 0.0250
3. profile[9]: σ = 0.0237
4. profile[5]: σ = 0.0230
5. aspect (dim 32): σ = 0.0229

For **'g'**:
1. profile[28]: σ = 0.0673
2. profile[29]: σ = 0.0648
3. profile[27]: σ = 0.0635
4. profile[30]: σ = 0.0615
5. serif_score (dim 36): σ = 0.0502

For **'a'**:
1. profile[28]: σ = 0.0590
2. profile[29]: σ = 0.0566
3. profile[27]: σ = 0.0550
4. profile[30]: σ = 0.0526
5. serif_score (dim 36): σ = 0.0474

**Key finding: `serif_score` is the single most discriminating scalar feature.** It appears in the top-5 for all tested chars. The column profile bins 27-30 (right side of the glyph) are also highly discriminating for 'a' and 'g' (where counter shapes and tail structure vary between fonts).

## Features That Would Help More

### 1. Counter Shape Analysis
The enclosed space inside 'a', 'e', 'g', 'o' varies significantly between fonts. Features:
- Counter area / total ink area ratio
- Counter centroid position (x, y normalized)
- Counter aspect ratio
- Number of counters (single-story 'a' vs double-story)

### 2. Terminal Angle Classification
How strokes end (flat, rounded, angled, ball terminal):
- Detect stroke endpoints via contour tracing
- Classify terminal type (0=flat, 1=round, 2=angled, 3=ball)
- Average terminal angle across all endpoints

### 3. Inter-Character Metric Ratios
Rather than scoring each char independently, use ratios between chars:
- width('m') / width('n') — varies by font (1.3-1.7)
- width('o') / width('e') — circular vs oval tendency
- height('g' descender) / height('h' ascender) — descender/ascender ratio

These ratios are invariant to rendering size and more discriminating than absolute values.

### 4. PANOSE/OS2 Metadata (Low-Hanging Fruit)
The fonts already embed classification data:
- PANOSE proportion: 3 (DejaVu) vs 2 (NotoSymbols)
- PANOSE weight: 6 vs 5
- Weight class, width class, sFamilyClass

These could be used as hard pre-filters (only compare fonts with matching PANOSE family type) or as additional feature dimensions.

### 5. Contour Complexity
- Number of outline points per glyph (DejaVu 'v': 14 points, NotoSymbols 'v': 32 points)
- Point density (points per unit outline length)
- Ratio of on-curve to off-curve points

## What's Fundamentally Wrong with Column Density Profiles

The 32-bin column density profile compresses the entire glyph shape into a 1D histogram. It captures **where ink is distributed left-to-right** but loses:

1. **Vertical structure** — two fonts with the same column density but different crossbar heights (like different 'e' designs) score identically
2. **Interior detail** — counters, apertures, and junction shapes are invisible when projected to columns
3. **Stroke endpoints** — terminal shape differences disappear in the column sum
4. **Scale invariance is too aggressive** — normalizing to 32 bins means a narrow 'i' and wide 'm' use the same resolution, wasting bins on 'i' and under-sampling 'm'

The profile IS useful for gross shape discrimination (serif vs sans, mono vs proportional), but can't separate fonts within the same category. It's the right feature for the wrong task — it was designed to distinguish letter shapes, not font families.

### Better Alternatives
- **2D moment features** (Hu moments, Zernike moments) — capture shape without 1D projection
- **Radial profile** — density in concentric rings from centroid, captures interior structure
- **Stroke width distribution** — histogram of stroke widths, orthogonal to shape
- **Junction topology** — number and type of stroke intersections ('a' has 0 in single-story, 1 in double-story)
