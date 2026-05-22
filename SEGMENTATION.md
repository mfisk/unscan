# Segmentation Algorithm: VP + Greedy Seam Selection

Reference: Seam carving DP from Avidan & Shamir, SIGGRAPH 2007.

## Input
Word image, N characters → need N-1 splits.

## Pass 1: Vertical Projection (VP)
Find contiguous runs of zero-ink columns (threshold 200). Each interior run
(not touching edges) gives one split at its midpoint. Let F = count of these.

If F >= N-1: pick the N-1 widest runs, done.

## Pass 2: Greedy Seam Selection
Need K = (N-1) - F more splits.

VP splits divide the word into segments: [0, vp₁, vp₂, ..., w].

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
