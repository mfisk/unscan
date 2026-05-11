# Unscan Specimen Test Run Report
**Date**: 2025-05-10
**Input**: `font-timeline-specimen-scanned.pdf` (23.76 MB, 8 pages, 300 DPI)
**Tool version**: post-SSIM-windowed-grayscale fix, with height-only rendering fix in verify.rs
**Threshold**: `--min-font-confidence 0.40` (default)

## Summary

**CRASHED** on first run at page 2, line `Victor Lardent, Morison created Times New Roman —`
- **Bug**: UTF-8 boundary panic in `truncate_str()` at `src/main.rs:702`
- **Cause**: `&s[..max]` sliced into middle of 3-byte em dash `—` (bytes 48..51)
- **Fix applied**: Walk backwards from `max` to find valid char boundary using `is_char_boundary()`

**Second run (with fix) completed page 1**, killed during page 2 because:

### Page 1 Results (66 lines total)
- **Vectorized**: 5 lines (7.6%)
- **Raster fallback**: 61 lines (92.4%)
  - "No confident font match": ~49 lines
  - OCR confidence too low: ~12 lines

### Lines that DID vectorize (Page 1)
| Line Text | Matched Font | Score | Correct? |
|-----------|-------------|-------|----------|
| `popularized each typeface.` | P052 Roman [hist] | 0.460 | ❌ (should be Caladea/Cambria — intro text) |
| `dogs.` (Garamond section) | EBGaramond12 Regular [lnum] | 0.507 | ✅ (close enough — EB Garamond) |
| `lazy dogs.` (Caslon section) | C059 Italic [hist] | 0.411 | ❌ (should be Libre Caslon Text) |
| `lazy dogs.` (Baskerville section) | EBGaramond08 Italic [lnum] | 0.433 | ❌ (should be Libre Baskerville) |
| `dogs.` (Bodoni section) | EBGaramond12 Regular [lnum] | 0.465 | ❌ (should be Libre Bodoni) |

**Accuracy on vectorized lines**: 1/5 correct (20%) — and even that's debatable since it matched on just the word "dogs."

### Score Distribution (all lines, Page 1)
- Best scores for full body text lines: **0.17 – 0.38** (ALL below 0.40 threshold)
- Correct fonts DO appear in "best" results sometimes but still score below threshold:
  - EB Garamond body text: best 0.260–0.279 (should be 0.40+)
  - SpecialElite (typewriter): best 0.222–0.241
  - Libre Baskerville body: best 0.230
  - Libre Bodoni body: best 0.298

### First Run (crashed) — Additional Data Points from Page 2
The first run got partway through page 2 before the UTF-8 crash. Same pattern:
- SpecialElite section: SpecialElite appeared as best match twice but only at 0.222 and 0.241
- Jost/Futura section: best matches were Roboto, Inter, EBGaramond — wrong family, scores 0.27–0.39
- Times New Roman section: best match FreeSerif at 0.278–0.295 — reasonable family but below threshold

## Analysis — Why So Bad?

### 1. The scanned specimen is HARD
The gen-specimen.py `scan_pdf()` function applies:
- Random rotation (-0.7° to +0.7°)
- Gaussian blur (sigma ~0.3 at 300 DPI)
- Speckle noise
- Paper texture/yellowing
- Possible compression artifacts

This degrades SSIM severely — even comparing a page to itself after scanning would probably score < 0.8.

### 2. The windowed SSIM fix made matching STRICTER
The whole point of the windowed SSIM fix was to reject false positives (Merriweather matching "Congratulations" at 0.565 when it was wrong). But it also:
- Tanks scores across the board
- Body text at 10-11pt with scanning artifacts → SSIM 0.20-0.35 even for correct fonts
- Only tiny fragments ("dogs.") have enough signal to score above 0.40

### 3. Short words have inflated scores
"dogs." is 5 characters. With fewer characters, SSIM has less opportunity to penalize. The correct font and the wrong font both score similarly for such short text. This is why only tiny fragments pass the threshold — they're too short to differentiate properly but also too short to fail.

### 4. OCR degradation from scanning
Many lines have OCR errors from the scan simulation:
- "Sox" instead of "fox"
- Spaces in wrong places
- Confidence drops below 80% on many alphabet/figure lines
- This compounds with font matching — wrong characters mean wrong renderings

## Recommendations

1. **Lower the threshold for scanned documents** — maybe `--min-font-confidence 0.25` or 0.30, with a separate "scanned mode" that tolerates lower SSIM
2. **Pre-process the scan before matching** — deskew, denoise, sharpen. This would bring SSIM scores up significantly
3. **Use the coarse score (NCC/IoU/Hu moments) as a secondary signal** — if coarse score is high but SSIM is low, trust the coarse score more for noisy inputs
4. **Paragraph-level grouping is critical** — if the best line in a paragraph picks the right font at 0.35, other lines should inherit that font rather than each independently failing at 0.25
5. **Consider testing on a non-distorted scan** — render the specimen PDF at 300 DPI without noise/rotation, then test. This isolates the font-matching algorithm from the OCR/scan-artifact problem.

## Bug Fix Applied
```rust
// BEFORE (panics on multi-byte UTF-8):
fn truncate_str(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() <= max { s }
    else { format!("{}…", &s[..max]) }  // PANIC if max lands mid-character
}

// AFTER:
fn truncate_str(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() <= max { s }
    else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) { end -= 1; }
        format!("{}…", &s[..end])
    }
}
```

## Files
- Fixed source: `src/main.rs` (truncate_str UTF-8 fix)
- This report: `test-docs/specimen-run-report.md`
- No output PDF generated (killed during page 2 to save time)
