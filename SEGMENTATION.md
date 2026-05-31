# Segmentation Algorithm: VP Split + Dual-DP Seam Carving

Reference: Seam carving DP from Avidan & Shamir, SIGGRAPH 2007.

## Input
Word image, N characters (from OCR) → need N-1 interior splits.

## Pass 1: VP (Vertical Projection) Split
Find contiguous runs of columns with zero ink pixels (threshold 200).
Each interior run (not touching image edges) gives one split at its midpoint.

Zero-ink columns are definitive inter-character gaps — no ambiguity.

If this yields ≥ N-1 splits: pick the N-1 widest runs, done.

## Pass 2: Greedy Seam Selection

For any VP segment that still needs more splits, seam carving takes over.

### Energy Function

```
energy(r, c) = 255 - pixel_value
```

White pixels have zero energy; black pixels have 255. This is "darkness" —
seams seek paths through whitespace (zero cost) and avoid ink.

### Dual DP: Forward + Reverse

For each segment, two DP passes run simultaneously:

**Forward** (top → bottom):
```
cost_fwd[0][c] = energy(0, c)
cost_fwd[r][c] = energy(r, c) + min over pc ∈ {c, c-1, c+1} of:
    cost_fwd[r-1][pc] + entry_penalty(r, c, pc)
```

**Reverse** (bottom → top):
```
cost_rev[H-1][c] = energy(H-1, c)
cost_rev[r][c]   = energy(r, c) + min over pc ∈ {c, c-1, c+1} of:
    cost_rev[r+1][pc] + entry_penalty(r, c, pc)
```

**Entry penalty** discourages paths that jump from light into dark pixels:
```
entry_penalty = max(0, current_darkness - neighbor_darkness) × ENTRY_PENALTY_WEIGHT
```
With `ENTRY_PENALTY_WEIGHT = 4.0`. This penalizes transitions from whitespace
into ink far more than traversal through already-dark areas.

### Candidate Generation

For each interior column `c` at the vertical midpoint (`mid_r = H/2`),
the combined cost of the best seam passing through `(mid_r, c)` is:

```
combined(c) = cost_fwd[mid_r][c] + cost_rev[mid_r][c] - energy(mid_r, c)
```

Subtracting the mid-row energy avoids double-counting.

### Midpoint Tie-Breaking

Consecutive columns with equal combined cost are collapsed into a single
candidate at the run's midpoint. For example, if columns 108–114 all have
zero cost, only column 111 becomes a candidate. This centers splits in
zero-cost bands, maximizing distance from ink on both sides.

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
   All candidates go onto a min-heap keyed by combined cost.
2. Pop the cheapest candidate. Validate:
   - **Ink on both sides**: the proposed left and right sub-segments must
     each have at least `MIN_INK_FOR_SYMBOL` total ink (16 × 255 = 4080).
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
| `ENTRY_PENALTY_WEIGHT` | 4.0 | Multiplier for dark-entry penalty in seam DP |
| `MIN_INK_FOR_SYMBOL` | 4080 (16×255) | Minimum ink in a sub-segment to count as a character |

## Accuracy

**473/486 (97.3%)** on `font-timeline-specimen.pdf` — 30 fonts spanning
500 years, including full uppercase/lowercase alphabet lines, serif fonts
with bridging serifs, and display-size anti-aliased text.
