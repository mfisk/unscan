# Segmentation Algorithm: VP Split + Greedy Seam Carving

Reference: Seam carving DP from Avidan & Shamir, SIGGRAPH 2007.

## Input
Word image, N characters (from OCR) → need N-1 interior splits.

## Pass 1: VP (Vertical Projection) Split
Find contiguous runs of columns with zero ink pixels (threshold 200).
Each interior run (not touching image edges) gives one split at its midpoint.

Zero-ink columns are definitive inter-character gaps — no ambiguity.

If this yields ≥ N-1 splits: pick the N-1 widest runs, done.

Both sides of every VP split must have at least `min_ink_for_symbol` total
column-ink or the split is rejected.  This threshold scales with the word
crop height: `(MIN_SYMBOL_FRAC × h)² × 255`, where `MIN_SYMBOL_FRAC = 0.07`
(a period is roughly 7% of line height).  The squared scaling tracks the
fact that ink area grows with the square of font size.

## Pass 2: Greedy Seam Carving

For any segments that still need more splits after VP, seam carving takes over.

### Energy Function

```
darkness(r, c) = 255.0 - pixel_value
```

White pixels have zero darkness; solid black pixels have 255. This is the
base per-pixel cost — seams seek paths through whitespace (zero cost) and
avoid ink.

### Entry Penalty

On top of the base darkness cost, the DP adds an **entry penalty** when the
path moves into a darker pixel than its predecessor:

```
entry_penalty = ENTRY_PENALTY_WEIGHT × max(0, darkness[r,c] - darkness[r-1,pc])
```

This directly encodes "stay in whitespace, don't wander into ink." A path
through the interior of a uniformly dark stroke pays the base darkness cost
but no entry penalty (the darkness isn't increasing). A path crossing from
a white gap into a stroke edge pays a heavy penalty.

`ENTRY_PENALTY_WEIGHT = 3.0`.

### Dual DP: Forward + Reverse

For each segment, two DP passes run simultaneously:

**Forward** (top → bottom):
```
cost_fwd[0][c] = darkness(0, c)
cost_fwd[r][c] = darkness(r, c) + min over pc ∈ {c, c-1, c+1} of:
    cost_fwd[r-1][pc] + entry_penalty(r, c, pc)
```

**Reverse** (bottom → top):
```
cost_rev[H-1][c] = darkness(H-1, c)
cost_rev[r][c]   = darkness(r, c) + min over pc ∈ {c, c-1, c+1} of:
    cost_rev[r+1][pc] + entry_penalty(r, c, pc)
```

### Candidate Generation

For each interior column `c` at the vertical midpoint (`mid_r = H/2`),
the combined cost of the best seam passing through `(mid_r, c)` is:

```
combined(c) = cost_fwd[mid_r][c] + cost_rev[mid_r][c] - darkness(mid_r, c)
```

Subtracting the mid-row darkness avoids double-counting. Multiple candidate
seams are generated per segment — all go onto the min-heap.

### Local-Minima Selection

Only local minima in DP cost enter the candidate set.  A local minimum is
a column (or run of consecutive equal-cost columns) whose DP cost is
strictly less than both neighbors.  This filters out plateau and shoulder
candidates that are not at a true cost valley, preventing the segment
penalty from pulling splits away from clean whitespace channels.

Among equal-cost runs that form a local minimum, the middle column is
selected, centering the split and maximizing distance from ink on both
sides.  For example, if columns 108–114 all have zero cost and both
neighbors are higher, only column 111 becomes a candidate.

Segment penalty is then added to each local-minimum candidate when it
enters the greedy heap.  Among multiple local minima in the same segment,
the heap picks the one with the best overall score (DP cost + segment
penalty).

### Straight-Path Preference

When tracing a seam path, if the pixel directly above/below has equal or
better cost than a diagonal neighbor, the straight path wins. The DP and
backtrace both iterate neighbors in `[c, c-1, c+1]` order with strict
less-than comparison, so ties always resolve to the vertical path.

### Path Tracing

The `SeamDp` struct retains both DP matrices so that `trace_path_through(col)`
can reconstruct the full seam path for any candidate without recomputation.
From the midpoint column:
- **Top half**: backtrace upward through `cost_fwd`, choosing the neighbor
  that produced the minimum cost at each row.
- **Bottom half**: backtrace downward through `cost_rev`, same logic.

Result: a `Vec<u32>` of length H — one column index per row.

### Diagonal Masking

When a seam splits a segment, child segments inherit the seam path as a
diagonal boundary. In the child's DP, pixels on the wrong side of the
inherited boundary get `f32::INFINITY` energy, preventing paths from crossing
into sibling territory. This replaces the old rectangular segment model where
children only used column ranges.

Each child inherits up to two boundaries via `SegBounds`:
- The **left child** gets the accepted seam as its right boundary
  (plus any inherited left boundary from the parent).
- The **right child** gets the accepted seam as its left boundary
  (plus any inherited right boundary from the parent).

### Greedy Loop

1. Compute candidate seams for each VP segment that needs more splits.
   Only local minima in DP cost become candidates (see Local-Minima
   Selection above).  Segment penalty is added to each, and all go onto
   a min-heap keyed by total cost (DP + segment penalty).
2. Pop the cheapest candidate. Validate:
   - **Ink on both sides**: the proposed left and right sub-segments must
     each have at least `min_ink_for_symbol` total ink (height-scaled;
     see Key Constants).
     Rejects splits that would create empty fragments.
3. Accept the split. Record its midpoint column and full seam path.
4. Drain all remaining candidates from the old segment (heap cleanup —
   replaces lazy stale-check filtering).
5. Compute new candidates for the two child segments (with diagonal masking
   from the accepted seam path) and push onto the heap.
6. Repeat until enough splits are found or the heap is exhausted.

### Why This Works

- **No guessing about chars-per-segment.** The greedy heap naturally picks
  the easiest remaining cut first.
- **Segments with multiple characters have cheap seams** (whitespace between
  chars = zero-cost paths).
- **Single-character segments have expensive seams** (cutting through ink).
- **Diagonal masking** prevents the classic seam-carving failure where a
  child's "cheapest" path wanders back into sibling territory through
  anti-aliased edges.
- **Midpoint tie-breaking** eliminates nondeterminism in zero-cost bands
  and produces geometrically centered splits.

## Fallback

If the algorithm can't produce enough splits (all remaining seams have
infinite cost or fail ink validation), fall back to
`uniform_boundaries(w, n_chars)`.

## Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `VP_THRESHOLD` | 200 | Grayscale values ≥ 200 are treated as white (no ink) |
| `ENTRY_PENALTY_WEIGHT` | 3.0 | Multiplier for dark-entry penalty in seam DP |
| `MIN_SYMBOL_FRAC` | 0.07 | Fraction of crop height for the smallest symbol (a period).  Minimum ink threshold = `(0.07 × h)² × 255`, scaling with DPI and font size. |
