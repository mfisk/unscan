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

#### Horizontal-Context Discount (Ink Discount)

Before the DP runs, darkness values are adjusted into an energy map. If a
pixel is lighter than the average of its two left neighbors AND the average
of its two right neighbors, it sits in a narrow gap between heavier ink.
Its energy is discounted to half its raw darkness:

```
energy(r, c) = darkness(r, c) * 0.5   if left_avg > d AND right_avg > d
             = darkness(r, c)          otherwise
```

where `left_avg = avg(darkness[r, c-1], darkness[r, c-2])` (clamped at edges).

This makes the DP prefer paths through thin gaps between strokes (e.g.
anti-aliased edges between touching glyphs) over paths through solid ink.

The **ink discount** for a seam path is the total energy saved by this
adjustment: `Σ (darkness[r,c] - energy[r,c])` along the path. This is
recorded in audit data as `ink_discount`.

### Entry Penalty

On top of the energy cost, the DP adds an **entry penalty** when the
path moves into a darker pixel than its predecessor:

```
entry_penalty = ENTRY_PENALTY_WEIGHT × max(0, darkness[r,c] - darkness[r-1,pc])
```

This directly encodes "stay in whitespace, don't wander into ink." A path
through the interior of a uniformly dark stroke pays the base energy cost
but no entry penalty (the darkness isn't increasing). A path crossing from
a white gap into a stroke edge pays a heavy penalty.

`ENTRY_PENALTY_WEIGHT = 3.0`.

### Dual DP: Forward + Reverse

For each segment, two DP passes run simultaneously:

**Forward** (top → bottom):
```
cost_fwd[0][c] = energy(0, c)
cost_fwd[r][c] = energy(r, c) + min over pc ∈ {c, c-1, c+1} of:
    cost_fwd[r-1][pc] + entry_penalty(r, c, pc) + (1.0 if diagonal)
```

**Reverse** (bottom → top):
```
cost_rev[H-1][c] = energy(H-1, c)
cost_rev[r][c]   = energy(r, c) + min over pc ∈ {c, c-1, c+1} of:
    cost_rev[r+1][pc] + entry_penalty(r, c, pc) + (1.0 if diagonal)
```

The `+1.0` for diagonal moves is the **horizontal cost** — it penalizes
seam paths that wander sideways, favoring straight vertical paths.

### Candidate Generation

For each interior column `c` at the vertical midpoint (`mid_r = H/2`),
the combined cost of the best seam passing through `(mid_r, c)` is:

```
combined(c) = cost_fwd[mid_r][c] + cost_rev[mid_r][c] - energy(mid_r, c)
```

Subtracting the mid-row energy avoids double-counting.

### Width Penalty

Each candidate's combined cost is augmented by a width penalty:

```
candidate_cost = combined(c) + 1.0 × seam_width
```

where `seam_width = max_col - min_col` of the traced path. This penalizes
seams that wander far horizontally — a perfectly vertical seam has width 0.

### Local-Minima Selection

Only local minima in candidate cost enter the greedy heap. A local minimum is
a column (or run of consecutive equal-cost columns) whose cost is strictly
less than both neighbors. This filters out plateau and shoulder candidates
that are not at a true cost valley, preventing the segment penalty from
pulling splits away from clean whitespace channels.

Among equal-cost runs that form a local minimum, the middle column is
selected, centering the split and maximizing distance from ink on both
sides.

### Segment-Size Penalty

When a candidate enters the greedy heap, a segment-size penalty is added
to its cost. This discourages splits that create a segment narrower than
the expected character width:

```
segment_penalty(seg_start, seg_end, col):
    if col has no ink → 0.0           (splitting at a clean gap is always free)
    left  = col - seg_start
    right = seg_end - col
    min_child = min(left, right)
    if min_child ≤ 0 → ∞
    penalty = (10.0 × avg_char_width / min_child)²
```

**The penalty is zero when the split column has no ink.** A clean
whitespace gap is a definitive character boundary regardless of
the resulting segment sizes — only splits through ink need the
size-balance heuristic.

The total cost on the heap is: `dp_cost + width_penalty + horizontal_cost + segment_size_penalty`.

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

Result: a `Vec<[u32; 2]>` of length H — one `[row, col]` per row.

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
   Only local minima in candidate cost become candidates (see Local-Minima
   Selection above). Segment-size penalty is added to each, and all go onto
   a min-heap keyed by total cost.
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

### Cost Breakdown (Audit)

Each seam's cost is decomposed in audit data as `SeamCost`:

| Field | Description |
|-------|-------------|
| `dp_cost` | Raw DP combined cost minus width and horizontal penalties |
| `seam_width_penalty` | `max_col - min_col` of the path |
| `segment_size_penalty` | Size-balance penalty (0 if no ink at split column) |
| `horizontal_cost` | Number of diagonal moves in the path |
| `ink_discount` | Energy saved by the horizontal-context discount |
| `total` | `dp_cost + seam_width_penalty + segment_size_penalty + horizontal_cost` |

### Why This Works

- **No guessing about chars-per-segment.** The greedy heap naturally picks
  the easiest remaining cut first.
- **Segments with multiple characters have cheap seams** (whitespace between
  chars = zero-cost paths).
- **Single-character segments have expensive seams** (cutting through ink).
- **No-ink columns skip segment penalty**, so clean gaps always win over
  cuts through ink regardless of segment size balance.
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
| `MIN_SYMBOL_FRAC` | 0.07 | Fraction of crop height for the smallest symbol (a period). Minimum ink threshold = `(0.07 × h)² × 255`, scaling with DPI and font size. |
| `ink_power` | 1 | Exponent on darkness in ink scoring (SeamParams) |
| `ink_norm` | 1 | Normalizer for ink scoring (SeamParams) |
| `delta_weight` | 1.0 | Weight on entry penalty relative to base ink cost |
| Horizontal cost | +1.0 | Per diagonal move in the DP |
| Width penalty | 1.0× | Multiplier on seam path width in candidate cost |

## Classifier Probability & Confidence

After segmentation produces character crops, each crop is classified against
per-character font models.  Two values control the output probability:
the **classifier score** (a softmax over font centroids) and the
**confidence score** (an out-of-distribution detector).

### Classifier Score (Softmax)

For a given character (e.g. `'A'`), the model stores one centroid per font
in LDA embedding space.  Given a query feature vector:

1. Embed the query into the same space (LDA projection).
2. Compute d²_i = squared Euclidean distance from query to each font
   centroid i.
3. Compute σ²_pairwise = median of all pairwise d² among centroids.
   This is the **kernel bandwidth** — it scales the softmax so that
   characters with widely-spread centroids don't produce artificially
   peaked distributions.
4. Softmax with Gaussian kernel (max-subtracted for numerical stability):

```
softmax_i = exp(-(d²_i - d²_min) / (2·σ²_pairwise))
          / Σ_j exp(-(d²_j - d²_min) / (2·σ²_pairwise))
```

This is a standard nearest-centroid classifier with a Gaussian kernel
whose bandwidth adapts per character.

### Confidence Score (OOD Detection)

A second variance, σ²_within, captures the **within-class scatter**:
the median d² from training samples to their own class centroid.  This
measures how far a typical in-distribution sample sits from its centroid.

The confidence score for a query is:

```
confidence = exp(-d²_min / (2·σ²_within))
```

where d²_min is the distance to the closest centroid.

- **In-distribution** (d²_min ≈ σ²_within): confidence ≈ 0.6 — the
  query is about as far as a typical training sample.  Softmax ranking
  is mostly preserved.
- **On a centroid** (d²_min ≈ 0): confidence ≈ 1.0 — full trust in the
  softmax distribution.
- **Out-of-distribution** (d²_min >> σ²_within): confidence → 0 — the
  query is far from everything the model has seen.

### Blending

The final probability blends softmax with uniform using confidence:

```
p_i = confidence · softmax_i + (1 - confidence) · (1/N)
```

where N is the number of fonts for that character.

**Effect:** An OOD observation (bad crop, unseen glyph, ligature fragment)
converges to uniform — it contributes no font preference.  An in-distribution
observation preserves its softmax ranking.  This prevents a single
overconfident misclassification from dominating the font score.

### Why Two Sigmas

σ²_pairwise and σ²_within serve different purposes and differ by orders
of magnitude (typically 400–12,000×):

| Sigma | Computed from | Typical scale | Purpose |
|-------|--------------|---------------|---------|
| σ²_pairwise | Median pairwise d² among centroids | ~7–127 | Softmax bandwidth: controls how peaked the probability distribution is |
| σ²_within | Median d² from training samples to own centroid | ~0.01 | Confidence: detects when the query is far from all known classes |

Using pairwise sigma for both would make the confidence check useless
(everything looks in-distribution relative to inter-centroid distances).
Using within-class sigma for both would make the softmax too peaked
(all probability on the nearest centroid).

### How Sigmas Are Computed (LDA Training)

**σ²_pairwise:** `pairwise_sigma_sq()` — enumerate all (N choose 2) pairs
of centroids, compute d² for each, take the median.

**σ²_within:** For each training sample, project it into LDA space and
compute d² to its own class centroid.  Collect all such distances across
all training samples for that character, take the median.

Both are stored per character in the weight file (IndexedEntryFixed, 32
bytes per entry).
