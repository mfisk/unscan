# Segmentation Algorithm: Three-Pass VP Cascade + Seam Carving

Reference: Seam carving DP from Avidan & Shamir, SIGGRAPH 2007.

## Input
Word image, N characters → need N-1 splits.

## Pass 1: Strict Zero-Ink VP
Find contiguous runs of columns with truly zero ink pixels (threshold 200).
Each interior run (not touching image edges) gives one split at its midpoint.

Zero-ink columns are definitive inter-character gaps — no ambiguity. This pass
produces geometrically centered split points that align well with how fonts
actually space characters.

If this yields ≥ N-1 splits: pick the N-1 widest runs, done.

## Pass 2: 5% Ink VP
If Pass 1 didn't produce enough splits, relax the threshold: a column counts
as "whitespace" if its total ink is ≤ 5% of the peak column (`max_ink / 20`).

**Why a second pass?** Serif fonts (especially Georgia uppercase) have serifs
that bridge inter-character gaps, creating 10-15% ink in every valley column
with zero pure-whitespace columns. The full alphabet line
`ABCDEFGHIJKLMNOPQRSTUVWXYZ` in Georgia noaa has **no** strict zero-ink
columns at all. The 5% cutoff catches most of these gaps.

**Why not use 5% as the only pass?** At large display sizes (36pt+), the 5%
cutoff over-segments — intra-character features like serif feet on 'T' or the
humps of 'm' create multi-pixel low-ink columns that VP mistakes for character
boundaries. "Timeline" at 36pt gets 12 VP splits for 8 characters with a 5%
cutoff, but exactly 7 with strict zero-ink. Running strict zero-ink first
avoids this.

Pass 2 only runs on segments that still need more splits after Pass 1.

## Pass 3: Greedy Seam Selection
For any remaining under-split segments, find the cheapest vertical seam path.

### Seam DP (per segment)
For a segment [x_start, x_end):
```
energy(r,c) = (255 - pixel) if pixel < threshold, else 0
M(0, c) = energy(0, c)
M(r, c) = energy(r, c) + min(M(r-1, c-1), M(r-1, c), M(r-1, c+1))
```
Minimum of last row = cheapest seam. Backtrace gives path.
Split point = column where seam crosses vertical midpoint of segment.

### Greedy loop
1. Compute one cheapest seam per segment. Put all in a min-heap keyed by cost.
2. Pop cheapest seam. Record its midpoint as a split.
3. That split divides its parent segment into two sub-segments.
4. Compute cheapest seam for each sub-segment. Push onto heap.
5. Repeat until K splits found.

### Why this works
- No guessing about chars-per-segment.
- Segments with multiple characters have cheap seams (whitespace between chars).
- Single-character segments have expensive seams (cutting through ink).
- Greedy always picks the easiest remaining cut first.
- Sub-segment recomputation ensures seams don't overlap.

## Fallback
If the algorithm can't produce enough splits (all remaining seams have
infinite cost), fall back to uniform_boundaries(w, n_chars).

## Key Test Cases

| Case | Behavior |
|------|----------|
| "Timeline" (36pt AA serif) | Pass 1 finds 7 zero-ink splits for 8 chars — exact. Old 5%-only gave 12 (over-seg) |
| Georgia uppercase `ABCDEFG...XYZ` (noaa) | Zero strict-zero columns. Pass 2 catches most serif-bridged gaps. Pass 3 handles the rest |
| Normal body text (10-12pt) | Pass 1 usually sufficient — inter-character whitespace is clear |
