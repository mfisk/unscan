//! Character segmentation: VP whitespace splitting + dual-DP seam carving.
//!
//! Given a word image and the expected number of characters, produce N+1
//! boundaries that partition the image into N cells.

use image::GrayImage;
use std::collections::{HashMap, HashSet};

/// Entry penalty weight for seam carving.  When the seam path moves into
/// a darker pixel than the previous one, the darkness increase is
/// multiplied by this weight and added as extra cost.  This penalizes
/// seams that drift from whitespace into glyph strokes.
const ENTRY_PENALTY_WEIGHT: f32 = 4.0;

/// Minimum total column-ink on each side of a split for it to be
/// accepted.  Each side must contain at least this much ink to be
/// considered a real symbol — roughly the ink of a period (~12 fully-
/// black pixel rows = 12 × 255 = 3060).  Used by both VP and seam
/// passes.
pub(crate) const MIN_INK_FOR_SYMBOL: u32 = 16 * 255;


/// Segment a word image into N character cells.
///
/// Three-pass cascade:
///
/// **Pass 1 — Vertical Profile (VP):** find contiguous runs of zero-ink
/// columns (threshold 200).  Each interior run gives one split at its
/// midpoint.  If that yields ≥ N-1 splits, pick the N-1 widest runs.
/// Both sides of every VP split must have at least `MIN_INK_FOR_SYMBOL`
/// total column-ink or the split is rejected.
///
/// **Pass 2 — Seam carving:** for remaining splits, find the cheapest
/// vertical seam in each existing segment via DP, pick the globally
/// cheapest, split there, and repeat.  Energy is ink-based: each pixel's
/// cost is its darkness (0 white, 255 black) plus an entry penalty
/// (`ENTRY_PENALTY_WEIGHT × darkness_increase`) when the path moves into
/// a darker pixel — directly encoding "stay in whitespace, don't wander
/// into ink."  The same `MIN_INK_FOR_SYMBOL` threshold applies: both
/// children of every accepted seam split must contain meaningful ink.
pub fn segment_characters(img: &GrayImage, n_chars: usize) -> (Vec<u32>, HashMap<u32, Vec<u32>>) {
    segment_characters_inner(img, n_chars, None, None)
}

/// Same as `segment_characters` but dumps per-pass diagnostics when `diag_dir` is Some.
pub fn segment_characters_diag(
    img: &GrayImage,
    n_chars: usize,
    diag_dir: &std::path::Path,
    word_text: &str,
) -> (Vec<u32>, HashMap<u32, Vec<u32>>) {
    segment_characters_inner(img, n_chars, Some(diag_dir), Some(word_text))
}

fn segment_characters_inner(
    img: &GrayImage,
    n_chars: usize,
    diag_dir: Option<&std::path::Path>,
    word_text: Option<&str>,
) -> (Vec<u32>, HashMap<u32, Vec<u32>>) {
    let (w, h) = img.dimensions();
    if n_chars <= 1 {
        return (vec![0, w], HashMap::new());
    }
    if w < 2 || h < 2 {
        return (uniform_boundaries(w, n_chars), HashMap::new());
    }

    let need = n_chars - 1; // number of splits needed

    // Two-pass segmentation cascade:
    //   Pass 1 — VP strict zero-ink: split at columns with truly zero ink
    //   Pass 2 — Seam carving: cheapest vertical path for remaining splits
    //
    // VP midpoints give geometrically centered boundaries.  Seam carving is
    // last resort for genuinely connected characters (e.g. serif bridges
    // at 10%+ ink where no column qualifies as low-ink).

    let threshold = 200u8;

    // Compute total ink per column (sum of dark-pixel intensities).
    let col_ink: Vec<u32> = (0..w)
        .map(|x| {
            (0..h)
                .map(|y| {
                    let px = img.get_pixel(x, y).0[0];
                    if px < threshold { (255 - px) as u32 } else { 0 }
                })
                .sum()
        })
        .collect();

    let _max_ink = col_ink.iter().copied().max().unwrap_or(0);

    // --- VP splitting: iterative, ink-trimmed ---
    //
    // Find character boundaries by iteratively splitting at the widest
    // low-ink column run.  After each split, trim each resulting segment
    // to its horizontal ink extent so that the whitespace margins around
    // the split are excluded from future searches.  This prevents
    // near-duplicate splits (e.g. 173 and 174) from consuming slots
    // that should go to real inter-character gaps.

    let col_has_ink: Vec<bool> = col_ink.iter().map(|&v| v > 0).collect();

    let mut splits: Vec<u32> = Vec::with_capacity(need);

    /// Find the ink extent within [seg_start, seg_end) using the given
    /// per-column ink flags.  Returns (ink_left, ink_right_exclusive).
    fn ink_extent(col_ink_flags: &[bool], seg_start: u32, seg_end: u32) -> (u32, u32) {
        let left = (seg_start..seg_end)
            .find(|&x| col_ink_flags[x as usize])
            .unwrap_or(seg_end);
        let right = (seg_start..seg_end)
            .rev()
            .find(|&x| col_ink_flags[x as usize])
            .map(|x| x + 1)
            .unwrap_or(seg_start);
        (left, right)
    }

    /// Find the best low-ink valley within [search_start, search_end).
    /// Ranks runs by minimum ink value (deepest valley wins), breaking
    /// ties by width (wider wins).  Returns (run_start, run_end, split_col)
    /// where split_col is the column with minimum ink in the run.
    fn best_low_ink_valley(is_ink: &[bool], col_ink: &[u32], search_start: u32, search_end: u32) -> Option<(u32, u32, u32)> {
        if search_end <= search_start + 2 {
            return None;
        }
        let mut best: Option<(u32, u32, u32, u32, u32)> = None; // (min_ink, neg_width, split_col, start, end)
        let mut run_start: Option<u32> = None;
        for x in search_start..search_end {
            if !is_ink[x as usize] {
                if run_start.is_none() { run_start = Some(x); }
            } else if let Some(start) = run_start {
                // Interior runs only (not touching search edges)
                if start > search_start {
                    let width = x - start;
                    let (min_ink, min_col) = (start..x)
                        .map(|c| (col_ink[c as usize], c))
                        .min()
                        .unwrap_or((u32::MAX, start));
                    let neg_width = u32::MAX - width;
                    if best.map_or(true, |b| (min_ink, neg_width) < (b.0, b.1)) {
                        best = Some((min_ink, neg_width, min_col, start, x));
                    }
                }
                run_start = None;
            }
        }
        best.map(|(_, _, split, s, e)| (s, e, split))
    }

    // Segments are stored as (left_edge, right_edge) of the full segment,
    // plus (ink_left, ink_right) for the ink-trimmed region.
    // Initially: one segment spanning the whole word.
    struct Segment { left: u32, right: u32, ink_left: u32, ink_right: u32 }

    let initial_ink = ink_extent(&col_has_ink, 0, w);
    let mut segments: Vec<Segment> = vec![Segment {
        left: 0, right: w,
        ink_left: initial_ink.0, ink_right: initial_ink.1,
    }];

    // Minimum ink on each side of a split for it to count as a real
    // inter-character boundary.  A period is about the smallest symbol —
    // roughly 12 fully-black pixels worth of ink.  At small font sizes
    // the crops get scaled up so real characters always have substantial
    // ink, while anti-aliasing bleed scales down proportionally.


    // --- Pass 1: VP strict zero-ink only ---
    while splits.len() < need {
        let mut best_valley: Option<(u32, u32, usize)> = None; // (min_ink, split_col, segment_index)

        for (si, seg) in segments.iter().enumerate() {
            if seg.ink_right <= seg.ink_left + 2 { continue; }

            if let Some((_s, _e, split)) = best_low_ink_valley(&col_has_ink, &col_ink, seg.ink_left, seg.ink_right) {
                // Reject splits that don't have substantial ink on both
                // sides — each side should have at least as much ink as
                // a period (~6 fully-black pixels = 6×255 = 1530 ink).
                let ink_left_sum: u32 = (seg.left..split).map(|c| col_ink[c as usize]).sum();
                let ink_right_sum: u32 = (split + 1..seg.right).map(|c| col_ink[c as usize]).sum();
                if ink_left_sum < MIN_INK_FOR_SYMBOL || ink_right_sum < MIN_INK_FOR_SYMBOL {
                    continue;
                }

                let min_ink = col_ink[split as usize];
                if best_valley.map_or(true, |b: (u32, u32, usize)| min_ink < b.0) {
                    best_valley = Some((min_ink, split, si));
                }
            }
        }

        let (_min_ink, mid, si) = match best_valley {
            Some(v) => v,
            None => break, // no more zero-ink valleys
        };

        splits.push(mid);

        let old = &segments[si];
        let left_ink = ink_extent(&col_has_ink, old.left, mid);
        let right_ink = ink_extent(&col_has_ink, mid, old.right);
        let left_seg = Segment { left: old.left, right: mid, ink_left: left_ink.0, ink_right: left_ink.1 };
        let right_seg = Segment { left: mid, right: old.right, ink_left: right_ink.0, ink_right: right_ink.1 };
        segments.splice(si..=si, [left_seg, right_seg]);
    }

    splits.sort();
    splits.dedup();

    let vp_splits = splits.clone();

    // Diag: dump VP passes
    if let (Some(ddir), Some(wtext)) = (diag_dir, word_text) {
        let _ = std::fs::create_dir_all(ddir);
        let _ = img.save(ddir.join("word_crop.png"));
        eprintln!(
            "  DIAG-SEG VP: \"{}\" {}x{} — {} vp splits, need {} total",
            &wtext[..wtext.len().min(30)], w, h, vp_splits.len(), need,
        );
    }

    // --- Pass 2: greedy seam carving ---
    //
    // For remaining splits, find the cheapest vertical seam in each segment.
    // Greedy: pop cheapest, split, recompute children, repeat.
    let mut seam_paths: HashMap<u32, Vec<u32>> = HashMap::new();
    if splits.len() < need {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        // Ink-based energy for seam carving.  Each pixel's base cost is
        // its darkness (0 for white, 255 for black).  The DP adds an
        // "entry penalty" when the path moves into a darker pixel than
        // the previous one — this penalizes seams that drift from
        // whitespace into a glyph stroke, directly encoding "stay in
        // the gap, don't wander into ink."
        //
        // The entry penalty is proportional to the darkness increase:
        //   penalty = ENTRY_PENALTY_WEIGHT * max(0, darkness[r] - darkness[r-1])
        //
        // This replaces the Avidan & Shamir gradient-based energy, which
        // couldn't distinguish between the interior of a dark stroke
        // (zero gradient) and a white gap (also zero gradient).

        // Per-pixel darkness: 0.0 for white, 255.0 for black.
        let darkness: Vec<Vec<f32>> = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        255.0 - img.get_pixel(x, y).0[0] as f32
                    })
                    .collect()
            })
            .collect();

        // The energy map is just darkness — used for the base per-pixel
        // cost in the DP.  The entry penalty is applied during the DP
        // transition, not stored in the energy map.
        let energy = &darkness;

        // Min-heap entry: (cost, split_col, seg_start, seg_end).
        // BinaryHeap is a max-heap, so wrap cost in Reverse-style ordering.
        #[derive(PartialEq)]
        struct SeamEntry {
            cost: f32,
            col: u32,
            seg_start: u32,
            seg_end: u32,
            seg_id: u32,
        }
        impl Eq for SeamEntry {}
        impl PartialOrd for SeamEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for SeamEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse: lower cost = higher priority
                other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<SeamEntry> = BinaryHeap::new();
        // Cache DP matrices by segment ID so we can trace paths
        // from the same matrices that computed candidate costs.
        let mut dp_cache: std::collections::HashMap<u32, SeamDp> = std::collections::HashMap::new();
        // Diagonal bounds per segment: seam paths that bound each side.
        // Pixels at or beyond these paths are unusable in the DP.
        // left_path[r] = seam col; pixels with col <= left_path[r] are masked.
        // right_path[r] = seam col; pixels with col >= right_path[r] are masked.
        struct SegBounds {
            left_path: Option<Vec<u32>>,
            right_path: Option<Vec<u32>>,
        }
        let mut seg_bounds: std::collections::HashMap<u32, SegBounds> = std::collections::HashMap::new();
        let mut next_seg_id: u32 = 0;

        // Build initial segments from VP splits and seed the heap.
        // Use ink_extent to trim whitespace so seam carving searches
        // inside the ink region, not in adjacent whitespace.
        let mut initial_segs: Vec<(u32, u32)> = Vec::new();
        {
            let mut prev = 0u32;
            for &s in &splits {
                if s > prev {
                    initial_segs.push((prev, s));
                }
                prev = s;
            }
            if prev < w {
                initial_segs.push((prev, w));
            }
        }
        for &(seg_start, seg_end) in &initial_segs {
            let (ink_l, ink_r) = ink_extent(&col_has_ink, seg_start, seg_end);
            if ink_r > ink_l + 2 {
                let sid = next_seg_id; next_seg_id += 1;
                let (cands, dp) = candidate_seams(&energy, ink_l, ink_r, h, None, None);
                if word_text.map_or(false, |w| w.starts_with("tradition")) {
                    eprintln!("  SEED seg=[{},{}) ink=[{},{}) {} candidates sid={}", seg_start, seg_end, ink_l, ink_r, cands.len(), sid);
                    for (col, cost) in &cands {
                        eprintln!("    candidate col={} cost={:.1}", col, cost);
                    }
                }
                for (col, cost) in &cands {
                    heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid });
                }
                seg_bounds.insert(sid, SegBounds { left_path: None, right_path: None });
                dp_cache.insert(sid, dp);
            } else if word_text.map_or(false, |w| w.starts_with("tradition")) {
                eprintln!("  SKIP NARROW seg=[{},{}) ink=[{},{})", seg_start, seg_end, ink_l, ink_r);
            }
        }

        // Greedy loop: pop cheapest, split, recompute children.
        // Lazy deletion: stale seg_ids are skipped on pop instead of
        // draining and rebuilding the heap on every accepted seam.
        let mut dead_sids: HashSet<u32> = HashSet::new();
        while splits.len() < need {
            let entry = match heap.pop() {
                Some(e) => e,
                None => break, // no valid seams remain
            };

            // Skip candidates from dead segments (replaced or consumed).
            if dead_sids.contains(&entry.seg_id) { continue; }

            if word_text.map_or(false, |w| w.starts_with("tradition")) {
                eprintln!("  SEAM POP [{}]: col={} cost={:.1} seg=[{},{}) sid={} | accepted={:?}", word_text.unwrap_or("?"), entry.col, entry.cost, entry.seg_start, entry.seg_end, entry.seg_id, &splits);
            }

            // Skip if this exact split column was already accepted.
            if splits.contains(&entry.col) {
                if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    SKIP DUP col={}", entry.col); }
                continue;
            }

            // Validate: both children must have meaningful ink.
            let left_ink = ink_extent(&col_has_ink, entry.seg_start, entry.col);
            let right_ink = ink_extent(&col_has_ink, entry.col + 1, entry.seg_end);
            let left_ok = left_ink.1 > left_ink.0 + 2;
            let right_ok = right_ink.1 > right_ink.0 + 2;

            if !left_ok || !right_ok {
                if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    SKIP INK col={} left_ok={} right_ok={}", entry.col, left_ok, right_ok); }
                // Seam hugged an edge → retry with narrowed range.
                if !right_ok && left_ok {
                    let new_end = entry.col;
                    if new_end > entry.seg_start + 2 {
                        let sid = next_seg_id; next_seg_id += 1;
                        let parent_bounds = seg_bounds.get(&entry.seg_id);
                        let lp = parent_bounds.and_then(|b| b.left_path.clone());
                        let rp = parent_bounds.and_then(|b| b.right_path.clone());
                        let (cands, dp) = candidate_seams(&energy, entry.seg_start, new_end, h, lp.as_deref(), rp.as_deref());
                        for (col, cost) in &cands {
                            heap.push(SeamEntry { cost: *cost, col: *col, seg_start: entry.seg_start, seg_end: new_end, seg_id: sid });
                        }
                        seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                        dp_cache.insert(sid, dp);
                        // Kill old segment — retry replacement covers valid columns
                        dead_sids.insert(entry.seg_id);
                        dp_cache.remove(&entry.seg_id);
                        seg_bounds.remove(&entry.seg_id);
                    }
                } else if !left_ok && right_ok {
                    let new_start = entry.col + 1;
                    if entry.seg_end > new_start + 2 {
                        let sid = next_seg_id; next_seg_id += 1;
                        let parent_bounds = seg_bounds.get(&entry.seg_id);
                        let lp = parent_bounds.and_then(|b| b.left_path.clone());
                        let rp = parent_bounds.and_then(|b| b.right_path.clone());
                        let (cands, dp) = candidate_seams(&energy, new_start, entry.seg_end, h, lp.as_deref(), rp.as_deref());
                        for (col, cost) in &cands {
                            heap.push(SeamEntry { cost: *cost, col: *col, seg_start: new_start, seg_end: entry.seg_end, seg_id: sid });
                        }
                        seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                        dp_cache.insert(sid, dp);
                        // Kill old segment — retry replacement covers valid columns
                        dead_sids.insert(entry.seg_id);
                        dp_cache.remove(&entry.seg_id);
                        seg_bounds.remove(&entry.seg_id);
                    }
                }
                continue;
            }

            // Trace the seam path early — needed for diagonal ink check.
            let path = match dp_cache.get(&entry.seg_id) {
                Some(dp) => dp.trace_path_through(&energy, entry.col),
                None => {
                    // Segment was already consumed; skip stale candidate.
                    continue;
                }
            };

            // Reject seam splits without substantial ink on both sides.
            // Use the actual diagonal seam path and diagonal segment
            // boundaries — not vertical column positions.
            let bounds = seg_bounds.get(&entry.seg_id);
            let left_bound = bounds.and_then(|b| b.left_path.as_deref());
            let right_bound = bounds.and_then(|b| b.right_path.as_deref());
            let mut seam_ink_left: u32 = 0;
            let mut seam_ink_right: u32 = 0;
            for row in 0..h {
                let seam_col = path[row as usize];
                let lb = left_bound.map_or(entry.seg_start, |lp| lp[row as usize]);
                let rb = right_bound.map_or(entry.seg_end, |rp| rp[row as usize]);
                for c in lb..seam_col {
                    let px = img.get_pixel(c, row).0[0];
                    if px < 200 { seam_ink_left += (255 - px) as u32; }
                }
                for c in (seam_col + 1)..rb {
                    let px = img.get_pixel(c, row).0[0];
                    if px < 200 { seam_ink_right += (255 - px) as u32; }
                }
            }
            if seam_ink_left < MIN_INK_FOR_SYMBOL || seam_ink_right < MIN_INK_FOR_SYMBOL {
                if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    SKIP MIN_INK col={} left={} right={} min={}", entry.col, seam_ink_left, seam_ink_right, MIN_INK_FOR_SYMBOL); }
                continue;
            }

            if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    ACCEPT col={}", entry.col); }
            if word_text.map_or(false, |w| w.starts_with("abcdefgh")) { eprintln!("    ACCEPT col={} cost={:.1} seg=[{},{}) sid={}", entry.col, entry.cost, entry.seg_start, entry.seg_end, entry.seg_id); }
            splits.push(entry.col);
            seam_paths.insert(entry.col, path.clone());

            // Capture parent's diagonal bounds before removing.
            let parent_lp = seg_bounds.get(&entry.seg_id).and_then(|b| b.left_path.clone());
            let parent_rp = seg_bounds.get(&entry.seg_id).and_then(|b| b.right_path.clone());

            // Mark old segment as dead — stale entries skipped on pop.
            let old_sid = entry.seg_id;
            dead_sids.insert(old_sid);
            dp_cache.remove(&old_sid);
            seg_bounds.remove(&old_sid);

            // Left child: inherits parent's left boundary, seam path as right boundary.
            {
                let (ink_l, ink_r) = left_ink;
                if ink_r > ink_l + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp = parent_lp.clone();
                    let rp: Option<Vec<u32>> = Some(path.clone());
                    let (cands, dp) = candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref());
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid });
                    }
                    seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                    dp_cache.insert(sid, dp);
                }
            }

            // Right child: seam path as left boundary, inherits parent's right boundary.
            {
                let (ink_l, ink_r) = right_ink;
                if ink_r > ink_l + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp: Option<Vec<u32>> = Some(path.clone());
                    let rp = parent_rp.clone();
                    let (cands, dp) = candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref());
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid });
                    }
                    seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                    dp_cache.insert(sid, dp);
                }
            }
        }

        splits.sort();
    }

    let seam_splits: Vec<u32> = splits.iter().filter(|s| !vp_splits.contains(s)).copied().collect();

    if w == 492 && n_chars == 10 {
    }

    // Diag: dump seam pass
    if let Some(ddir) = diag_dir {
        let vp_mids: Vec<u32> = vp_splits.clone();
        crate::seg_diag::save_split_overlay_with_paths(img, &vp_mids, &seam_splits, &[], &seam_paths, &ddir.join("seam_overlay.png"));
        eprintln!(
            "  DIAG-SEG SEAM: {} seam splits added (total now {})",
            seam_splits.len(), splits.len(),
        );
    }

    // Build final boundaries: [0, split1, split2, ..., w]
    let mut bounds = Vec::with_capacity(n_chars + 1);
    bounds.push(0);
    for &s in &splits {
        if s > 0 && s < w {
            bounds.push(s);
        }
    }
    bounds.push(w);
    bounds.dedup();

    // Diag: dump final overlay
    if let (Some(ddir), Some(wtext)) = (diag_dir, word_text) {
        let vp_mids: Vec<u32> = vp_splits.clone();
        let empty: Vec<u32> = Vec::new();
        crate::seg_diag::save_split_overlay_with_paths(img, &vp_mids, &seam_splits, &empty, &seam_paths, &ddir.join("final_overlay.png"));

        // NOTE: char crops are saved by extract_chars_from_boundaries
        // (the actual CI code path), not here — so diag shows exact CI inputs.

        let n_segs = bounds.len().saturating_sub(1);
        eprintln!(
            "  DIAG-SEG FINAL: {} total splits, {} boundaries, {} segments (expected {})",
            splits.len(), bounds.len(), n_segs, n_chars,
        );
        if n_segs != n_chars {
            eprintln!("  *** MISMATCH: {} segments vs {} expected chars", n_segs, n_chars);
        }

        // Write summary JSON
        let summary = serde_json::json!({
            "word_text": wtext,
            "image_w": w,
            "image_h": h,
            "n_chars_expected": n_chars,
            "n_segments_produced": n_segs,
            "vp_splits": vp_splits,
            "seam_splits": seam_splits,
            "final_boundaries": bounds,
            "seam_paths": seam_paths,
            "mismatch": n_segs != n_chars,
        });
        let _ = std::fs::write(ddir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap_or_default());
    }

    (bounds, seam_paths)
}

/// Exhaustive (col, cost) candidates via dual-DP: for every interior column
/// at mid-row, compute the cost of the cheapest seam path that passes
/// through that column.  All candidates go on the heap so the greedy loop
/// sees every possible split point, not just bottom-row local minima
/// (which can miss clean inter-character gaps that backtrack to a
/// different mid-row column).
/// DP matrices from a seam candidate search.  Retaining these lets us
/// trace the optimal path through any mid-row column without recomputing.
struct SeamDp {
    cost_fwd: Vec<Vec<f32>>,
    cost_rev: Vec<Vec<f32>>,
    seg_start: u32,
    seg_end: u32,
    h: u32,
}

impl SeamDp {

    /// Backtrace the cheapest path constrained to pass through
    /// `target_col` at mid-row.
    fn trace_path_through(&self, energy: &[Vec<f32>], target_col: u32) -> Vec<u32> {
        let seg_w = (self.seg_end - self.seg_start) as usize;
        let base = self.seg_start as usize;
        let mid_r = (self.h / 2) as usize;
        let last_r = (self.h - 1) as usize;
        let tc = (target_col - self.seg_start) as usize;

        let mut path = vec![0u32; self.h as usize];
        path[mid_r] = target_col;

        // Top half: backtrace upward from (mid_r, tc) through cost_fwd
        {
            let mut c = tc;
            for r in (1..=mid_r).rev() {
                let cur_dark = energy[r][base + c];
                let mut best_cost = f32::INFINITY;
                let mut best_c = c;
                for &pc in &[c, c.wrapping_sub(1), c + 1] {
                    if pc < seg_w {
                        let prev_dark = energy[r - 1][base + pc];
                        let entry = if cur_dark > prev_dark {
                            (cur_dark - prev_dark) * ENTRY_PENALTY_WEIGHT
                        } else {
                            0.0
                        };
                        let cand = self.cost_fwd[r - 1][pc] + entry;
                        if cand < best_cost {
                            best_cost = cand;
                            best_c = pc;
                        }
                    }
                }
                c = best_c;
                path[r - 1] = self.seg_start + c as u32;
            }
        }

        // Bottom half: backtrace downward from (mid_r, tc) through cost_rev
        {
            let mut c = tc;
            for r in mid_r..last_r {
                let cur_dark = energy[r][base + c];
                let mut best_cost = f32::INFINITY;
                let mut best_c = c;
                for &pc in &[c, c.wrapping_sub(1), c + 1] {
                    if pc < seg_w {
                        let child_dark = energy[r + 1][base + pc];
                        let entry = if child_dark > cur_dark {
                            (child_dark - cur_dark) * ENTRY_PENALTY_WEIGHT
                        } else {
                            0.0
                        };
                        let cand = self.cost_rev[r + 1][pc] + entry;
                        if cand < best_cost {
                            best_cost = cand;
                            best_c = pc;
                        }
                    }
                }
                c = best_c;
                path[r + 1] = self.seg_start + c as u32;
            }
        }

        path
    }
}

fn candidate_seams(
    energy: &[Vec<f32>],
    seg_start: u32,
    seg_end: u32,
    h: u32,
    left_path: Option<&[u32]>,   // pixels with col <= left_path[r] are masked
    right_path: Option<&[u32]>,  // pixels with col >= right_path[r] are masked
) -> (Vec<(u32, f32)>, SeamDp) {
    let seg_w = (seg_end - seg_start) as usize;
    if seg_w < 3 || h < 1 {
        let dp = SeamDp { cost_fwd: Vec::new(), cost_rev: Vec::new(), seg_start, seg_end, h };
        return (Vec::new(), dp);
    }
    let base = seg_start as usize;
    let mid_r = (h / 2) as usize;

    // Masked energy: pixels outside diagonal boundaries are impassable.
    let masked_energy = |r: usize, c: usize| -> f32 {
        let abs_col = base + c;
        if let Some(lp) = left_path {
            if abs_col <= lp[r] as usize { return f32::INFINITY; }
        }
        if let Some(rp) = right_path {
            if abs_col >= rp[r] as usize { return f32::INFINITY; }
        }
        energy[r][abs_col]
    };

    // Forward DP: cost_fwd[r][c] = cheapest path from any top-row column
    // down to (r, c).  Cost = sum of ink darkness along the path, plus
    // an entry penalty each time the path moves into a darker pixel.
    let mut cost_fwd = vec![vec![0.0f32; seg_w]; h as usize];
    for c in 0..seg_w {
        cost_fwd[0][c] = masked_energy(0, c);
    }
    for r in 1..h as usize {
        for c in 0..seg_w {
            let cur_dark = masked_energy(r, c);
            let mut best = f32::INFINITY;
            for &pc in &[c, c.wrapping_sub(1), c + 1] {
                if pc < seg_w {
                    let prev_dark = masked_energy(r - 1, pc);
                    let entry = if cur_dark > prev_dark {
                        (cur_dark - prev_dark) * ENTRY_PENALTY_WEIGHT
                    } else {
                        0.0
                    };
                    let candidate = cost_fwd[r - 1][pc] + entry;
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            cost_fwd[r][c] = cur_dark + best;
        }
    }

    // Reverse DP: models downward continuation from (r, c) to bottom.
    let last_r = (h - 1) as usize;
    let mut cost_rev = vec![vec![0.0f32; seg_w]; h as usize];
    for c in 0..seg_w {
        cost_rev[last_r][c] = masked_energy(last_r, c);
    }
    for r in (0..last_r).rev() {
        for c in 0..seg_w {
            let cur_dark = masked_energy(r, c);
            let mut best = f32::INFINITY;
            for &pc in &[c, c.wrapping_sub(1), c + 1] {
                if pc < seg_w {
                    let child_dark = masked_energy(r + 1, pc);
                    let entry = if child_dark > cur_dark {
                        (child_dark - cur_dark) * ENTRY_PENALTY_WEIGHT
                    } else {
                        0.0
                    };
                    let candidate = cost_rev[r + 1][pc] + entry;
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            cost_rev[r][c] = cur_dark + best;
        }
    }

    // For each interior column at mid-row, the cheapest path through it
    // costs cost_fwd[mid][c] + cost_rev[mid][c] - energy[mid][c]
    // (subtract once to avoid double-counting the mid-row pixel).
    let mut raw_candidates: Vec<(u32, f32)> = Vec::with_capacity(seg_w.saturating_sub(2));
    for c in 1..seg_w - 1 {
        let me = masked_energy(mid_r, c);
        if me >= f32::INFINITY { continue; } // masked pixel, skip
        let combined = cost_fwd[mid_r][c] + cost_rev[mid_r][c] - me;
        let split_col = seg_start + c as u32;
        raw_candidates.push((split_col, combined));
    }

    // Collapse runs of consecutive columns with equal cost into a
    // single candidate at the run's midpoint.  This picks the center
    // of a zero-cost band, maximizing distance from ink on both sides.
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(raw_candidates.len());
    let mut i = 0;
    while i < raw_candidates.len() {
        let cost = raw_candidates[i].1;
        let run_start = i;
        while i < raw_candidates.len()
            && raw_candidates[i].1 == cost
            && (i == run_start || raw_candidates[i].0 == raw_candidates[i - 1].0 + 1)
        {
            i += 1;
        }
        let mid_idx = (run_start + i - 1) / 2;
        candidates.push(raw_candidates[mid_idx]);
    }

    let dp = SeamDp { cost_fwd, cost_rev, seg_start, seg_end, h };
    (candidates, dp)
}

/// Uniform character boundaries.
fn uniform_boundaries(width: u32, n: usize) -> Vec<u32> {
    let mut b = Vec::with_capacity(n + 1);
    for i in 0..=n {
        b.push((i as f32 * width as f32 / n as f32).round() as u32);
    }
    b
}

