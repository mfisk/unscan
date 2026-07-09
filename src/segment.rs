//! Character segmentation: whitespace splitting + dual-DP seam carving.
//!
//! Given a word image and the expected number of characters, produce N+1
//! boundaries that partition the image into N cells.

use image::GrayImage;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::features::{contrast_normalize_char, is_supported, normalize_to_ink_bounds, NORM_H};
use crate::verify::WordPlacement;

/// Seam carving scoring parameters, configurable via environment variables
/// for hill-climbing parameter search.  Defaults reproduce the original
/// linear scoring (ink_power=1, delta_weight=4, no row_ink influence).
struct SeamParams {
    ink_power: f32,         // exponent on darkness for base cost (1.0 = linear)
    ink_norm: f32,          // divisor after powering (1.0 = raw)
    ink_row_weight: f32,    // multiplier for row_ink factor (0.0 = ignore)
    ink_row_power: f32,     // exponent on row_ink
    delta_weight: f32,      // entry penalty weight (was 4.0)
    delta_power: f32,       // exponent on darkness delta
    delta_scale_power: f32, // exponent on cur_dark/max_ink scaling
    delta_row_weight: f32,  // row_ink multiplier in delta (0.0 = ignore)
    delta_row_power: f32,   // exponent on row_ink in delta
}

fn seam_params() -> &'static SeamParams {
    static PARAMS: OnceLock<SeamParams> = OnceLock::new();
    PARAMS.get_or_init(|| {
        fn env_f32(name: &str, default: f32) -> f32 {
            std::env::var(name).ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }
        let p = SeamParams {
            ink_power: env_f32("SEAM_INK_POWER", 1.0),
            ink_norm: env_f32("SEAM_INK_NORM", 1.0),
            ink_row_weight: env_f32("SEAM_INK_ROW_WEIGHT", 0.0),
            ink_row_power: env_f32("SEAM_INK_ROW_POWER", 1.0),
            delta_weight: env_f32("SEAM_DELTA_WEIGHT", 4.0),
            delta_power: env_f32("SEAM_DELTA_POWER", 1.0),
            delta_scale_power: env_f32("SEAM_DELTA_SCALE_POWER", 1.0),
            delta_row_weight: env_f32("SEAM_DELTA_ROW_WEIGHT", 0.0),
            delta_row_power: env_f32("SEAM_DELTA_ROW_POWER", 1.0),
        };
        p
    })
}

/// Per-pixel ink score: base traversal cost for the seam path.
#[inline]
fn ink_score(darkness: f32, row: usize, row_ink: &[f32]) -> f32 {
    let p = seam_params();
    let base = if p.ink_power == 1.0 { darkness } else { darkness.powf(p.ink_power) }
        / p.ink_norm;
    if p.ink_row_weight == 0.0 {
        base
    } else {
        let ri = if p.ink_row_power == 1.0 { row_ink[row] }
                 else { row_ink[row].powf(p.ink_row_power) };
        base * (1.0 + p.ink_row_weight * ri)
    }
}

/// Transition penalty: extra cost when the seam moves into darker ink.
#[inline]
fn delta_ink_score(
    dark_cur: f32, dark_prev: f32,
    _row_cur: usize, _row_prev: usize,
    _row_ink: &[f32], max_ink: f32,
) -> f32 {
    if dark_cur <= dark_prev { return 0.0; }
    if !dark_cur.is_finite() || !dark_prev.is_finite() { return f32::INFINITY; }
    let p = seam_params();
    // Linear edge penalty: k × (dc - dp).  Path-independent.
    let delta = dark_cur - dark_prev;
    p.delta_weight * delta
}

/// Fraction of the crop height that represents the smallest symbol
/// (a period).  A period is roughly 8% of line height.  The minimum
/// ink threshold is `(MIN_SYMBOL_FRAC * h)²` dark pixels (counted,
/// not intensity-weighted), making it scale with DPI and font size
/// without penalising grey/anti-aliased text vs solid black.
const MIN_SYMBOL_FRAC: f32 = 0.07;


/// Segment a word image into N character cells.
///
/// Two-pass cascade:
///
/// **Pass 1 — Whitespace splitting:** find contiguous runs of zero-ink
/// columns (threshold 200) within the ink extent and split at each run's
/// midpoint.
///
/// **Pass 2 — Seam carving:** for remaining splits, find the cheapest
/// vertical seam in each existing segment via DP, pick the globally
/// cheapest, split there, and repeat.  Energy is ink-based: each pixel's
/// cost is its darkness (0 white, 255 black) plus an entry penalty
/// (delta_ink_score) when the path moves into
/// a darker pixel — directly encoding "stay in whitespace, don't wander
/// into ink."  Both children of every accepted seam split must contain
/// meaningful ink (`min_ink_for_symbol`).
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

    // Minimum ink for a real symbol, scaled to crop height.
    // Count of pixels above ink threshold — not weighted by intensity,
    // so grey/anti-aliased text isn't penalised vs black text.
    let min_side = MIN_SYMBOL_FRAC * h as f32;
    let min_ink_for_symbol = (min_side * min_side) as u32;

    // Two-pass segmentation cascade:
    //   Pass 1 — Whitespace: split at midpoint of zero-ink column runs
    //   Pass 2 — Seam carving: cheapest vertical path for remaining splits

    let threshold = 200u8;

    // Compute total ink per column (count of pixels above ink threshold).
    let col_ink: Vec<u32> = (0..w)
        .map(|x| {
            (0..h)
                .map(|y| {
                    let px = img.get_pixel(x, y).0[0];
                    if px < threshold { 1u32 } else { 0 }
                })
                .sum()
        })
        .collect();

    let _max_ink = col_ink.iter().copied().max().unwrap_or(0);

    let col_has_ink_strict: Vec<bool> = col_ink.iter().map(|&v| v > 0).collect();

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

    let initial_ink = ink_extent(&col_has_ink_strict, 0, w);

    // --- Pass 1: whitespace splitter ---
    // Find runs of consecutive zero-ink columns within the ink extent
    // and split at the midpoint of each run.
    {
        let (ink_l, ink_r) = ink_extent(&col_has_ink_strict, 0, w);
        let mut run_start: Option<u32> = None;
        for c in ink_l..ink_r {
            if !col_has_ink_strict[c as usize] {
                if run_start.is_none() {
                    run_start = Some(c);
                }
            } else {
                if let Some(rs) = run_start {
                    let mid = (rs + c) / 2;
                    splits.push(mid);
                    run_start = None;
                }
            }
        }
    }

    splits.sort();
    splits.dedup();

    let vp_splits = splits.clone();

    // Diag: dump VP passes
    if let (Some(ddir), Some(_wtext)) = (diag_dir, word_text) {
        let _ = std::fs::create_dir_all(ddir);
        let _ = img.save(ddir.join("word_crop.png"));

    }

    // --- Pass 2: greedy seam carving ---
    //
    // For remaining splits, find the cheapest vertical seam in each segment.
    // Greedy: pop cheapest, split, recompute children, repeat.
    let mut seam_paths: HashMap<u32, Vec<u32>> = HashMap::new();
    // Seam instrumentation: capture seed candidates and greedy-loop events
    // for the audit summary JSON.
    let mut seam_seed_candidates: Vec<serde_json::Value> = Vec::new();
    let mut seam_greedy_log: Vec<serde_json::Value> = Vec::new();
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
        //   penalty = delta_ink_score(darkness[r], darkness[r-1], r, r-1, row_ink, max_ink)
        //
        // This replaces the Avidan & Shamir gradient-based energy, which
        // couldn't distinguish between the interior of a dark stroke
        // (zero gradient) and a white gap (also zero gradient).

        // Per-pixel darkness: 0.0 for white, 255.0 for black (raw).
        // ink_score() applies the parameterized transform during DP scoring.
        let darkness: Vec<Vec<f32>> = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        255.0 - img.get_pixel(x, y).0[0] as f32
                    })
                    .collect()
            })
            .collect();

        // Row ink fractions: what share of the word's total ink is in each
        // row.  Rows with heavy strokes are high; whitespace rows near zero.
        let total_ink: f32 = darkness.iter()
            .flat_map(|row| row.iter()).copied().sum();
        let row_ink: Vec<f32> = darkness.iter()
            .map(|row| {
                if total_ink > 0.0 { row.iter().copied().sum::<f32>() / total_ink }
                else { 0.0 }
            })
            .collect();

        // The energy map is just darkness — used for the base per-pixel
        // cost in the DP.  The entry penalty is applied during the DP
        // transition, not stored in the energy map.
        let energy = &darkness;

        // Word-level max ink (p95 of raw darkness): used by delta_ink_score
        // to scale the entry penalty proportionally.
        let mut ink_values: Vec<f32> = Vec::new();
        for row in &darkness {
            for &d in row {
                if d > 0.0 {
                    ink_values.push(d);
                }
            }
        }
        ink_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_ink = if ink_values.is_empty() {
            255.0
        } else {
            let p95_idx = (ink_values.len() as f64 * 0.95) as usize;
            let p95_idx = p95_idx.min(ink_values.len() - 1);
            ink_values[p95_idx].max(1.0) // avoid division by zero
        };

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
            let (ink_l, ink_r) = ink_extent(&col_has_ink_strict, seg_start, seg_end);
            if ink_r > ink_l + 2 {
                let sid = next_seg_id; next_seg_id += 1;
                let (cands, dp) = candidate_seams(&energy, ink_l, ink_r, h, None, None, max_ink, &row_ink);
                for (col, cost) in &cands {
                    heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid });
                    seam_seed_candidates.push(serde_json::json!({
                        "col": col, "cost": *cost as f64,
                        "seg": [ink_l, ink_r], "sid": sid,
                    }));
                }
                seg_bounds.insert(sid, SegBounds { left_path: None, right_path: None });
                dp_cache.insert(sid, dp);
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
            if dead_sids.contains(&entry.seg_id) {
                seam_greedy_log.push(serde_json::json!({
                    "action": "skip_dead", "col": entry.col, "cost": entry.cost as f64,
                    "seg": [entry.seg_start, entry.seg_end], "sid": entry.seg_id,
                }));
                continue;
            }

            // Skip if this exact split column was already accepted.
            if splits.contains(&entry.col) {
                seam_greedy_log.push(serde_json::json!({
                    "action": "skip_dup", "col": entry.col, "cost": entry.cost as f64,
                    "seg": [entry.seg_start, entry.seg_end], "sid": entry.seg_id,
                }));
                continue;
            }

            // Validate: both children must have meaningful ink.
            let left_ink = ink_extent(&col_has_ink_strict, entry.seg_start, entry.col);
            let right_ink = ink_extent(&col_has_ink_strict, entry.col + 1, entry.seg_end);
            let left_ok = left_ink.1 > left_ink.0 + 2;
            let right_ok = right_ink.1 > right_ink.0 + 2;

            if !left_ok || !right_ok {
                seam_greedy_log.push(serde_json::json!({
                    "action": "skip_ink", "col": entry.col, "cost": entry.cost as f64,
                    "seg": [entry.seg_start, entry.seg_end], "sid": entry.seg_id,
                    "left_ok": left_ok, "right_ok": right_ok,
                }));
                // Seam hugged an edge → retry with narrowed range.
                if !right_ok && left_ok {
                    let new_end = entry.col;
                    if new_end > entry.seg_start + 2 {
                        let sid = next_seg_id; next_seg_id += 1;
                        let parent_bounds = seg_bounds.get(&entry.seg_id);
                        let lp = parent_bounds.and_then(|b| b.left_path.clone());
                        let rp = parent_bounds.and_then(|b| b.right_path.clone());
                        let (cands, dp) = candidate_seams(&energy, entry.seg_start, new_end, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
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
                        let (cands, dp) = candidate_seams(&energy, new_start, entry.seg_end, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
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
            // Vertical-only winners get a straight vertical path; DP
            // winners get the cheapest diagonal path through their column.
            let path = match dp_cache.get(&entry.seg_id) {
                Some(dp) => dp.trace_path_through(&energy, entry.col, &row_ink),
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
                    if px < 200 { seam_ink_left += 1; }
                }
                for c in (seam_col + 1)..rb {
                    let px = img.get_pixel(c, row).0[0];
                    if px < 200 { seam_ink_right += 1; }
                }
            }
            if seam_ink_left < min_ink_for_symbol || seam_ink_right < min_ink_for_symbol {
                seam_greedy_log.push(serde_json::json!({
                    "action": "skip_min_ink", "col": entry.col, "cost": entry.cost as f64,
                    "seg": [entry.seg_start, entry.seg_end], "sid": entry.seg_id,
                    "ink_left": seam_ink_left, "ink_right": seam_ink_right,
                    "min_ink": min_ink_for_symbol,
                }));
                continue;
            }

            let final_col = entry.col;

            splits.push(final_col);
            seam_greedy_log.push(serde_json::json!({
                "action": "accept", "col": final_col, "cost": entry.cost as f64,
                "seg": [entry.seg_start, entry.seg_end], "sid": entry.seg_id,
                "splits_so_far": &splits,
            }));
            seam_paths.insert(final_col, path.clone());

            // Capture parent's diagonal bounds before removing.
            let parent_lp = seg_bounds.get(&entry.seg_id).and_then(|b| b.left_path.clone());
            let parent_rp = seg_bounds.get(&entry.seg_id).and_then(|b| b.right_path.clone());

            // Mark old segment as dead — stale entries skipped on pop.
            let old_sid = entry.seg_id;
            dead_sids.insert(old_sid);
            dp_cache.remove(&old_sid);
            seg_bounds.remove(&old_sid);

            // Recompute child ink extents.
            let child_left_ink = ink_extent(&col_has_ink_strict, entry.seg_start, final_col);
            let child_right_ink = ink_extent(&col_has_ink_strict, final_col + 1, entry.seg_end);

            // Left child: inherits parent's left boundary, seam path as right boundary.
            {
                let (ink_l, ink_r) = child_left_ink;
                if ink_r > ink_l + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp = parent_lp.clone();
                    let rp: Option<Vec<u32>> = Some(path.clone());
                    let (cands, dp) = candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid });
                    }
                    seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                    dp_cache.insert(sid, dp);
                }
            }

            // Right child: seam path as left boundary, inherits parent's right boundary.
            {
                let (ink_l, ink_r) = child_right_ink;
                if ink_r > ink_l + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp: Option<Vec<u32>> = Some(path.clone());
                    let rp = parent_rp.clone();
                    let (cands, dp) = candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
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
        let ws_mids: Vec<u32> = vp_splits.clone();
        crate::seg_diag::save_split_overlay_with_paths(img, &ws_mids, &seam_splits, &[], &seam_paths, &ddir.join("seam_overlay.png"));
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
        let ws_mids: Vec<u32> = vp_splits.clone();
        let empty: Vec<u32> = Vec::new();
        crate::seg_diag::save_split_overlay_with_paths(img, &ws_mids, &seam_splits, &empty, &seam_paths, &ddir.join("final_overlay.png"));

        // NOTE: char crops are saved by extract_chars_from_boundaries
        // (the actual CI code path), not here — so diag shows exact CI inputs.

        let n_segs = bounds.len().saturating_sub(1);

        let summary = serde_json::json!({
            "word_text": wtext,
            "image_w": w,
            "image_h": h,
            "n_chars_expected": n_chars,
            "n_segments_produced": n_segs,
            "ws_splits": vp_splits,
            "seam_splits": seam_splits,
            "final_boundaries": bounds,
            "seam_paths": seam_paths,
            "mismatch": n_segs != n_chars,
            "seam_seed_candidates": seam_seed_candidates,
            "seam_greedy_log": seam_greedy_log,
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
    _cost_fwd: Vec<f32>,   // flat [row * seg_w + col]
    _cost_rev: Vec<f32>,   // flat [row * seg_w + col]
    pred_fwd: Vec<u32>,   // flat [row * seg_w + col] — packed (r, c) predecessor
    pred_rev: Vec<u32>,   // flat [row * seg_w + col] — packed (r, c) predecessor
    seg_start: u32,
    _seg_end: u32,
    seg_w: usize,
    h: u32,
    _max_ink: f32,
    _row_ink: Vec<f32>,    // per-row ink fraction for ink_score/delta_ink_score
}

impl SeamDp {

    /// Backtrace the cheapest path constrained to pass through
    /// `target_col` at mid-row.
    fn trace_path_through(&self, _energy: &[Vec<f32>], target_col: u32, _row_ink: &[f32]) -> Vec<u32> {
        let seg_w = self.seg_w;
        let _base = self.seg_start as usize;
        let mid_r = (self.h / 2) as usize;
        let last_r = (self.h - 1) as usize;
        let tc = (target_col - self.seg_start) as usize;

        let mut path = vec![0u32; self.h as usize];
        path[mid_r] = target_col;

        // Top half: backtrace upward from (mid_r, tc) through pred_fwd
        {
            let mut c = tc;
            let mut r = mid_r;
            while r > 0 {
                let pred = self.pred_fwd[r * seg_w + c] as usize;
                let pr = pred / seg_w;
                let pc = pred % seg_w;
                if pr < r {
                    // Vertical step: predecessor is in row above
                    path[r - 1] = self.seg_start + pc as u32;
                    c = pc;
                    r = pr;
                    if r == 0 { break; }
                } else {
                    // Horizontal step: predecessor on same row, keep going
                    c = pc;
                }
            }
        }

        // Bottom half: backtrace downward from (mid_r, tc) through pred_rev
        {
            let mut c = tc;
            let mut r = mid_r;
            while r < last_r {
                let pred = self.pred_rev[r * seg_w + c] as usize;
                let pr = pred / seg_w;
                let pc = pred % seg_w;
                if pr > r {
                    // Vertical step: predecessor is in row below
                    path[r + 1] = self.seg_start + pc as u32;
                    c = pc;
                    r = pr;
                    if r >= last_r { break; }
                } else {
                    // Horizontal step: predecessor on same row, keep going
                    c = pc;
                }
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
    max_ink: f32,                // p95 ink darkness — scales entry penalty
    row_ink: &[f32],             // per-row ink fractions for scoring
) -> (Vec<(u32, f32)>, SeamDp) {
    let seg_w = (seg_end - seg_start) as usize;
    if seg_w < 3 || h < 1 {
        let dp = SeamDp { _cost_fwd: Vec::new(), _cost_rev: Vec::new(), pred_fwd: Vec::new(), pred_rev: Vec::new(), seg_start, _seg_end: seg_end, seg_w: 0, h, _max_ink: max_ink, _row_ink: row_ink.to_vec() };
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

    // Forward DP: cost_fwd[r * seg_w + c] = cheapest path from any top-row column
    // down to (r, c).  Cost = sum of ink darkness along the path, plus
    // an entry penalty each time the path moves into a darker pixel.
    // Row-ink discount removed from DP: it's gated on run length (≥11px),
    // which is column-dependent. VP uses it because it's a straight vertical
    // line; DP paths wander, so the discount doesn't apply.
    let p = seam_params();
    let n_cells = h as usize * seg_w;
    let mut cost_fwd = vec![0.0f32; n_cells];
    let mut pred_fwd = vec![0u32; n_cells];
    for c in 0..seg_w {
        let dark0 = masked_energy(0, c);
        cost_fwd[c] = ink_score(dark0, 0, row_ink)
            + delta_ink_score(dark0, 0.0, 0, 0, row_ink, max_ink);
        pred_fwd[c] = c as u32; // self
    }
    for r in 1..h as usize {
        let row_off = r * seg_w;
        let prev_off = (r - 1) * seg_w;
        // Step 1: vertical from row above (same column)
        for c in 0..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let prev_dark = masked_energy(r - 1, c);
            let entry = delta_ink_score(cur_dark, prev_dark, r, r - 1, row_ink, max_ink);
            cost_fwd[row_off + c] = cur_ink + entry + cost_fwd[prev_off + c];
            pred_fwd[row_off + c] = (prev_off + c) as u32;
        }
        // Step 2: diagonal from (r-1, c-1) → (r, c)
        // Diagonal costs dest ink + delta + pass-through cell ink.
        for c in 1..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let prev_dark = masked_energy(r - 1, c - 1);
            let entry = delta_ink_score(cur_dark, prev_dark, r, r - 1, row_ink, max_ink);
            let pass_dark = masked_energy(r, c - 1);
            let pass_ink = ink_score(pass_dark, r, row_ink);
            let via_diag = cost_fwd[prev_off + c - 1] + cur_ink + entry + pass_ink;
            if via_diag < cost_fwd[row_off + c] {
                cost_fwd[row_off + c] = via_diag;
                pred_fwd[row_off + c] = (prev_off + c - 1) as u32;
            }
        }
        // Step 3: diagonal from (r-1, c+1) → (r, c)
        for c in 0..seg_w - 1 {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let prev_dark = masked_energy(r - 1, c + 1);
            let entry = delta_ink_score(cur_dark, prev_dark, r, r - 1, row_ink, max_ink);
            let pass_dark = masked_energy(r, c + 1);
            let pass_ink = ink_score(pass_dark, r, row_ink);
            let via_diag = cost_fwd[prev_off + c + 1] + cur_ink + entry + pass_ink;
            if via_diag < cost_fwd[row_off + c] {
                cost_fwd[row_off + c] = via_diag;
                pred_fwd[row_off + c] = (prev_off + c + 1) as u32;
            }
        }
    }

    // Reverse DP: models downward continuation from (r, c) to bottom.
    let last_r = (h - 1) as usize;
    let mut cost_rev = vec![0.0f32; n_cells];
    let mut pred_rev = vec![0u32; n_cells];
    let last_off = last_r * seg_w;
    for c in 0..seg_w {
        let dark_last = masked_energy(last_r, c);
        cost_rev[last_off + c] = ink_score(dark_last, last_r, row_ink)
            + delta_ink_score(dark_last, 0.0, last_r, last_r, row_ink, max_ink);
        pred_rev[last_off + c] = (last_off + c) as u32; // self
    }
    for r in (0..last_r).rev() {
        let row_off = r * seg_w;
        let next_off = (r + 1) * seg_w;
        // Step 1: vertical from row below (same column)
        for c in 0..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let child_dark = masked_energy(r + 1, c);
            let entry = delta_ink_score(child_dark, cur_dark, r + 1, r, row_ink, max_ink);
            cost_rev[row_off + c] = cur_ink + entry + cost_rev[next_off + c];
            pred_rev[row_off + c] = (next_off + c) as u32;
        }
        // Step 2: diagonal from (r+1, c-1) → (r, c)
        for c in 1..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let child_dark = masked_energy(r + 1, c - 1);
            let entry = delta_ink_score(child_dark, cur_dark, r + 1, r, row_ink, max_ink);
            let pass_dark = masked_energy(r, c - 1);
            let pass_ink = ink_score(pass_dark, r, row_ink);
            let via_diag = cost_rev[next_off + c - 1] + cur_ink + entry + pass_ink;
            if via_diag < cost_rev[row_off + c] {
                cost_rev[row_off + c] = via_diag;
                pred_rev[row_off + c] = (next_off + c - 1) as u32;
            }
        }
        // Step 3: diagonal from (r+1, c+1) → (r, c)
        for c in 0..seg_w - 1 {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let child_dark = masked_energy(r + 1, c + 1);
            let entry = delta_ink_score(child_dark, cur_dark, r + 1, r, row_ink, max_ink);
            let pass_dark = masked_energy(r, c + 1);
            let pass_ink = ink_score(pass_dark, r, row_ink);
            let via_diag = cost_rev[next_off + c + 1] + cur_ink + entry + pass_ink;
            if via_diag < cost_rev[row_off + c] {
                cost_rev[row_off + c] = via_diag;
                pred_rev[row_off + c] = (next_off + c + 1) as u32;
            }
        }
    }

    // For each interior column at mid-row, the cheapest path through it
    // costs cost_fwd[mid][c] + cost_rev[mid][c] - energy[mid][c]
    // (subtract once to avoid double-counting the mid-row pixel).
    // Width penalty is now handled inside the DP via drift multiplier
    // on horizontal steps, so no post-hoc width penalty needed.
    let mid_off = mid_r * seg_w;
    let mut dp_candidates: Vec<(u32, f32)> = Vec::with_capacity(seg_w.saturating_sub(2));
    for c in 1..seg_w - 1 {
        let me = masked_energy(mid_r, c);
        if me >= f32::INFINITY { continue; } // masked pixel, skip
        let combined = cost_fwd[mid_off + c] + cost_rev[mid_off + c] - ink_score(me, mid_r, row_ink);
        // Width penalty: trace path to find horizontal extent.
        let mut min_c = c;
        let mut max_c = c;
        {
            let mut cur = mid_off + c;
            loop {
                let p = pred_fwd[cur] as usize;
                if p == cur { break; }
                let pc = p % seg_w;
                if pc < min_c { min_c = pc; }
                if pc > max_c { max_c = pc; }
                cur = p;
            }
        }
        {
            let mut cur = mid_off + c;
            loop {
                let p = pred_rev[cur] as usize;
                if p == cur { break; }
                let pc = p % seg_w;
                if pc < min_c { min_c = pc; }
                if pc > max_c { max_c = pc; }
                cur = p;
            }
        }
        let width = (max_c - min_c) as f32;
        let split_col = seg_start + c as u32;
        dp_candidates.push((split_col, combined + 50.0 * width));
    }

    // Second pass: trace each DP candidate's path and adjust cost

    // Sort DP candidates by column for collapse pass.
    dp_candidates.sort_by(|a, b| a.0.cmp(&b.0));

    // Collapse runs of consecutive columns with equal cost into a
    // single candidate at the run's midpoint.  This picks the center
    // of a zero-cost band, maximizing distance from ink on both sides.
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(dp_candidates.len());
    let mut i = 0;
    while i < dp_candidates.len() {
        let cost = dp_candidates[i].1;
        let run_start = i;
        while i < dp_candidates.len()
            && dp_candidates[i].1 == cost
            && (i == run_start || dp_candidates[i].0 == dp_candidates[i - 1].0 + 1)
        {
            i += 1;
        }
        let mid_idx = (run_start + i - 1) / 2;
        candidates.push(dp_candidates[mid_idx]);
    }

    let dp = SeamDp { _cost_fwd: cost_fwd, _cost_rev: cost_rev, pred_fwd, pred_rev, seg_start, _seg_end: seg_end, seg_w, h, _max_ink: max_ink, _row_ink: row_ink.to_vec() };
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


const MIN_WORD_LEN: usize = 3;

/// Per-word segmentation data retained for lazy bigram cropping.
pub struct WordSeg {
    /// Index of the originating word in the input `words` slice (i.e. line.words).
    pub source_word_idx: usize,
    pub word_img: GrayImage,
    pub chars: Vec<char>,
    pub boundaries: Vec<u32>,
    pub seam_paths: HashMap<u32, Vec<u32>>,
    pub crop_h: u32,
}

pub struct LineCrops {
    /// Per-word segmentation data for lazy unigram and bigram cropping (plain path).
    pub word_segs: Vec<WordSeg>,
    /// Per-word segmentation data for lazy cropping (ligature path).
    /// Only present when the line has ligature-eligible words.
    pub lig_word_segs: Option<Vec<WordSeg>>,
}

pub fn segment_line(
    page: &GrayImage,
    words: &[WordPlacement],
    word_height: u32,
    diag_seg_dir: Option<&std::path::Path>,
    render_params: &crate::char_render::RenderParams,
) -> LineCrops {
    let _ = render_params; // reserved for scan-side binarize
    if words.is_empty() || word_height == 0 {
        return LineCrops { word_segs: Vec::new(), lig_word_segs: None };
    }

    // Crop whole words (reliable bboxes from Tesseract) and split them
    // into individual characters using VP + seam carving.

    let mut sorted: Vec<(usize, &WordPlacement)> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.text.chars().count() >= MIN_WORD_LEN && w.width > 0)
        .collect();
    sorted.sort_by(|(_, a), (_, b)| b.text.chars().count().cmp(&a.text.chars().count()));

    let mut char_counts: HashMap<char, usize> = HashMap::new();
    let mut word_segs: Vec<WordSeg> = Vec::new();
    let mut lig_word_segs: Vec<WordSeg> = Vec::new();
    let mut any_ligatures = false;
    let mut words_with_ligatures: HashSet<usize> = HashSet::new();

    for (word_idx, &(orig_idx, word)) in sorted.iter().enumerate() {
        let chars_in_word: Vec<char> = word.text.chars().filter(|c| is_supported(*c)).collect();
        if chars_in_word.is_empty() {
            continue;
        }

        let need_any = chars_in_word.iter().any(|c| {
            char_counts.get(c).copied().unwrap_or(0) < 2
        });
        if !need_any {
            continue;
        }

        let wx = word.x_off;
        let wy = word.y_off;
        let ww = word.width;
        let wh = word.height;

        let (pw, ph) = page.dimensions();
        if wx >= pw || wy >= ph {
            continue;
        }

        let crop_w = ww.min(pw - wx);
        let crop_h = wh.min(ph - wy);
        if crop_w < 2 || crop_h < 2 {
            continue;
        }

        // Crop the word at its expanded bbox (ink-expanded by expand_words_to_ink).
        // Don't ink-trim — trimming picks up stray text from adjacent lines.
        let word_img = image::imageops::crop_imm(page, wx, wy, crop_w, crop_h).to_image();
        // Contrast-normalize the whole word before segmentation so the
        // segmenter sees consistent ink/background separation regardless
        // of scan brightness.  Individual char crops inherit this and are
        // not re-normalized downstream.
        let word_img = contrast_normalize_char(&word_img);

        let (_word_w, word_h) = word_img.dimensions();
        let all_chars: Vec<char> = word.text.chars().collect();

        // Build ligature-collapsed char array: replace sequences like
        // ['f','f','l'] with ['\u{FB04}'] so "affluent" becomes
        // ['a','\u{FB04}','u','e','n','t'] with n_chars=6.
        let lig_chars = collapse_ligature_chars(&all_chars);
        let has_ligatures = lig_chars.len() < all_chars.len();

        let word_diag_dir = diag_seg_dir.map(|ddir| {
            let word_slug = crate::seg_diag::sanitize_text(&word.text);
            ddir.join(format!("word_{:03}_{}", word_idx, word_slug))
        });

        // ── Path A: plain segmentation (OCR chars as-is) ────────────
        let (bounds_plain, seams_plain) = if let Some(ref wdir) = word_diag_dir {
            let pdir = wdir.join("seg_plain");
            segment_characters_diag(&word_img, all_chars.len(), &pdir, &word.text)
        } else {
            segment_characters(&word_img, all_chars.len())
        };

        // Update char counts (for the word-skip optimisation)
        for &c in &all_chars {
            if is_supported(c) {
                *char_counts.entry(c).or_insert(0) += 1;
            }
        }

        word_segs.push(WordSeg {
            source_word_idx: orig_idx,
            word_img: word_img.clone(),
            chars: all_chars.clone(),
            boundaries: bounds_plain.clone(),
            seam_paths: seams_plain.clone(),
            crop_h: word_h,
        });

        if has_ligatures {
            any_ligatures = true;
            words_with_ligatures.insert(word_segs.len() - 1);
            // ── Path B: ligature segmentation (reduced n_chars) ─────
            let (bounds_lig, seams_lig) = if let Some(ref wdir) = word_diag_dir {
                let ldir = wdir.join("seg_lig");
                segment_characters_diag(&word_img, lig_chars.len(), &ldir, &word.text)
            } else {
                segment_characters(&word_img, lig_chars.len())
            };

            lig_word_segs.push(WordSeg {
                source_word_idx: orig_idx,
                word_img: word_img.clone(),
                chars: lig_chars,
                boundaries: bounds_lig,
                seam_paths: seams_lig,
                crop_h: word_h,
            });
        }
    }

    // For the ligature path, non-ligature words use plain segmentation
    if any_ligatures {
        for (idx, seg) in word_segs.iter().enumerate() {
            if !words_with_ligatures.contains(&idx) {
                lig_word_segs.push(WordSeg {
                    source_word_idx: seg.source_word_idx,
                    word_img: seg.word_img.clone(),
                    chars: seg.chars.clone(),
                    boundaries: seg.boundaries.clone(),
                    seam_paths: seg.seam_paths.clone(),
                    crop_h: seg.crop_h,
                });
            }
        }
    }

    LineCrops {
        word_segs,
        lig_word_segs: if any_ligatures { Some(lig_word_segs) } else { None },
    }
}

/// Collapse ligature sequences into single Unicode ligature codepoints.
/// e.g. ['a','f','f','l','u','e','n','t'] → ['a','\u{FB04}','u','e','n','t']
/// Greedy longest-first matching (ffi/ffl before ff/fi/fl).
fn collapse_ligature_chars(chars: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut matched = false;
        for &(seq, lig_char) in LIGATURE_SEQUENCES {
            if i + seq.len() <= chars.len() && chars[i..i + seq.len()] == *seq {
                out.push(lig_char);
                i += seq.len();
                matched = true;
                break;
            }
        }
        if !matched {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Ligature sequences: (char_sequence, unicode_ligature_char).
/// Longer sequences first so ffi/ffl match before ff/fi/fl.
const LIGATURE_SEQUENCES: &[(&[char], char)] = &[
    (&['f', 'f', 'i'], '\u{FB03}'),  // ffi
    (&['f', 'f', 'l'], '\u{FB04}'),  // ffl
    (&['f', 'f'],      '\u{FB00}'),  // ff
    (&['f', 'i'],      '\u{FB01}'),  // fi
    (&['f', 'l'],      '\u{FB02}'),  // fl
];



// ---------------------------------------------------------------------------
// Bigram crop extraction (sliding window pairs)
// ---------------------------------------------------------------------------

/// Extract sliding-window bigram crops from a word image.
///
/// For chars [c0, c1, c2, c3] with boundaries [b0, b1, b2, b3, b4],
/// produces pairs: [(c0,c1, crop[b0..b2]), (c1,c2, crop[b1..b3]), (c2,c3, crop[b2..b4])].
///
/// Each pair crop spans from boundary[i] to boundary[i+2], ink-cropped
/// and normalized to NORM_H.  This preserves relative heights, kerning,
/// and spacing — the same signal the bigram training data captures.
///
/// `has_model` filters which pairs are worth cropping.  Pairs for which
/// it returns false are skipped entirely (no crop, no resize).
/// Crop a single character from a word image at position `i`.
/// Returns `None` if the region is too narrow or contains no ink.
/// Crop `n` adjacent characters starting at position `i` from a word image.
/// Spans boundaries[i] to boundaries[i+n], with seam masking at the
/// outer edges.  Returns `None` if the region is too narrow or has no ink.
pub fn crop_ngram(
    word_img: &GrayImage,
    i: usize,
    n: usize,
    boundaries: &[u32],
    seam_paths: &HashMap<u32, Vec<u32>>,
    crop_h: u32,
) -> Option<GrayImage> {
    let (ww, _) = word_img.dimensions();

    if i + n >= boundaries.len() {
        return None;
    }

    let b_left = boundaries[i];
    let b_right = boundaries[i + n];

    let left_seam = seam_paths.get(&b_left);
    let right_seam = seam_paths.get(&b_right);

    let x0 = if let Some(sp) = left_seam {
        sp.iter().copied().min().unwrap_or(b_left).min(b_left)
    } else {
        b_left
    }.min(ww);

    let x1 = if let Some(sp) = right_seam {
        sp.iter().copied().max().unwrap_or(b_right).max(b_right).saturating_add(1)
    } else {
        b_right
    }.min(ww);

    if x1 <= x0 || (x1 - x0) < 2 * (n as u32) {
        return None;
    }

    let mut crop = image::imageops::crop_imm(word_img, x0, 0, x1 - x0, crop_h).to_image();

    let crop_w = x1 - x0;
    for y in 0..crop_h.min(crop.height()) {
        if let Some(sp) = left_seam {
            if let Some(&seam_x) = sp.get(y as usize) {
                let limit = seam_x.saturating_sub(x0);
                for cx in 0..limit.min(crop_w) {
                    crop.put_pixel(cx, y, image::Luma([255u8]));
                }
            }
        }
        if let Some(sp) = right_seam {
            if let Some(&seam_x) = sp.get(y as usize) {
                let start = seam_x.saturating_sub(x0);
                for cx in start..crop_w {
                    crop.put_pixel(cx, y, image::Luma([255u8]));
                }
            }
        }
    }

    normalize_to_ink_bounds(&crop, NORM_H)
}

