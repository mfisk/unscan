//! Character segmentation: whitespace splitting + dual-DP seam carving.
//!
//! Given a word image and the expected number of characters, produce N+1
//! boundaries that partition the image into N cells.

use std::sync::Arc;
use image::GrayImage;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::cell::RefCell;

use crate::features::{audit_all_chars_enabled, contrast_normalize_char, is_supported, normalize_to_ink_bounds, NORM_H};

// Thread-local reuse for candidate_seams forward+reverse DP buffers (perf wins)
thread_local! {
    static CANDIDATE_SEAMS_FWD: RefCell<(Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    static CANDIDATE_SEAMS_REV: RefCell<(Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    static SEGMENT_TL: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
}
use crate::verify::WordPlacement;

/// Seam carving scoring parameters, configurable via environment variables
/// for hill-climbing parameter search.  Defaults reproduce the original
/// linear scoring (ink_power=1, delta_weight=2.5, no row_ink influence).
struct SeamParams {
    ink_power: f32,         // exponent on darkness for base cost (1.0 = linear)
    ink_norm: f32,          // divisor after powering (1.0 = raw)
    ink_row_weight: f32,    // multiplier for row_ink factor (0.0 = ignore)
    ink_row_power: f32,     // exponent on row_ink
    delta_weight: f32,      // entry penalty weight (was 4.0)
    #[allow(dead_code)]
    delta_power: f32,       // exponent on darkness delta
    #[allow(dead_code)]
    delta_scale_power: f32, // exponent on cur_dark/max_ink scaling
    #[allow(dead_code)]
    delta_row_weight: f32,  // row_ink multiplier in delta (0.0 = ignore)
    #[allow(dead_code)]
    delta_row_power: f32,   // exponent on row_ink in delta
    horizontal_cost: f32,   // cost per diagonal move
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
            delta_weight: env_f32("SEAM_DELTA_WEIGHT", 1.0),
            delta_power: env_f32("SEAM_DELTA_POWER", 1.0),
            delta_scale_power: env_f32("SEAM_DELTA_SCALE_POWER", 1.0),
            delta_row_weight: env_f32("SEAM_DELTA_ROW_WEIGHT", 0.0),
            delta_row_power: env_f32("SEAM_DELTA_ROW_POWER", 1.0),
            horizontal_cost: env_f32("SEAM_HORIZONTAL_COST", 0.01),
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
    _row_ink: &[f32], _max_ink: f32,
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
pub fn segment_characters(img: &GrayImage, n_chars: usize) -> (Vec<u32>, HashMap<u32, Vec<[u32; 2]>>, SegSummary) {
    segment_characters_inner(img, n_chars, None, None)
}

/// Same as `segment_characters` but dumps per-pass diagnostics when `diag_dir` is Some.
pub fn segment_characters_diag(
    img: &GrayImage,
    n_chars: usize,
    diag_dir: &std::path::Path,
    word_text: &str,
) -> (Vec<u32>, HashMap<u32, Vec<[u32; 2]>>, SegSummary) {
    segment_characters_inner(img, n_chars, Some(diag_dir), Some(word_text))
}

fn segment_characters_inner(
    img: &GrayImage,
    n_chars: usize,
    diag_dir: Option<&std::path::Path>,
    word_text: Option<&str>,
) -> (Vec<u32>, HashMap<u32, Vec<[u32; 2]>>, SegSummary) {
    let (w, h) = img.dimensions();
    if n_chars <= 1 {
        return (vec![0, w], HashMap::new(), SegSummary {
            image_w: w, image_h: h, n_chars_expected: n_chars as u32,
            n_segments_produced: 1, mismatch: n_chars != 1,
            ws_splits: Vec::new(), seam_splits: Vec::new(),
            seam_costs: HashMap::new(),
        });
    }
    if w < 2 || h < 2 {
        let bounds = uniform_boundaries(w, n_chars);
        let n_segs = bounds.len().saturating_sub(1) as u32;
        return (bounds, HashMap::new(), SegSummary {
            image_w: w, image_h: h, n_chars_expected: n_chars as u32,
            n_segments_produced: n_segs, mismatch: n_segs != n_chars as u32,
            ws_splits: Vec::new(), seam_splits: Vec::new(),
            seam_costs: HashMap::new(),
        });
    }

    let need = n_chars - 1; // number of splits needed

    // Minimum ink for a real symbol, scaled to crop height.
    // Count of pixels above ink threshold — not weighted by intensity,
    // so grey/anti-aliased text isn't penalised vs black text.
    let min_side = MIN_SYMBOL_FRAC * h as f32;
    let min_ink_for_symbol = (min_side * min_side) as u32;

    // Average character width — used to penalize splitting narrow segments.
    let avg_char_width = w as f32 / n_chars as f32;

    // Two-pass segmentation cascade:
    //   Pass 1 — Whitespace: split at midpoint of zero-ink column runs
    //   Pass 2 — Seam carving: cheapest vertical path for remaining splits

    let threshold = crate::INK_THRESH;

    // Compute total ink per column (count of pixels < thresh). Row-major single scan
    // merges col_has_ink_strict creation – cache-friendly vs column-major strided.
    let w_us = w as usize;
    let h_us = h as usize;
    let raw_img = img.as_raw();
    let mut col_ink: Vec<u32> = vec![0u32; w_us];
    let mut col_has_ink_strict: Vec<u8> = vec![0u8; w_us];
    for y in 0..h_us {
        let base = y * w_us;
        for x in 0..w_us {
            if raw_img[base + x] < threshold {
                col_ink[x] += 1;
                col_has_ink_strict[x] = 1;
            }
        }
    }

    let _max_ink = col_ink.iter().copied().max().unwrap_or(0);

    let mut splits: Vec<u32> = Vec::with_capacity(need);

    /// Find the ink extent within [seg_start, seg_end) using the given
    /// per-column ink flags.  Returns (ink_left, ink_right_exclusive).
    fn ink_extent(col_ink_flags: &[u8], seg_start: u32, seg_end: u32) -> (u32, u32) {
        let left = (seg_start..seg_end)
            .find(|&x| col_ink_flags[x as usize] != 0)
            .unwrap_or(seg_end);
        let right = (seg_start..seg_end)
            .rev()
            .find(|&x| col_ink_flags[x as usize] != 0)
            .map(|x| x + 1)
            .unwrap_or(seg_start);
        (left, right)
    }

    let _initial_ink = ink_extent(&col_has_ink_strict, 0, w);

    // --- Pass 1: whitespace splitter ---
    // Find runs of consecutive zero-ink columns within the ink extent.
    // Collect candidates, then greedily accept only those whose segment
    // penalty is acceptable — prevents over-splitting when glyphs like
    // " have multiple clean vertical gaps but only need one split.
    {
        let (ink_l, ink_r) = ink_extent(&col_has_ink_strict, 0, w);
        let mut vp_candidates: Vec<u32> = Vec::new();
        let mut run_start: Option<u32> = None;
        for c in ink_l..ink_r {
            if col_has_ink_strict[c as usize] == 0 {
                if run_start.is_none() {
                    run_start = Some(c);
                }
            } else {
                if let Some(rs) = run_start {
                    let mid = (rs + c) / 2;
                    vp_candidates.push(mid);
                    run_start = None;
                }
            }
        }

        // Fast path: if we don't have more VP candidates than needed,
        // accept them all — no selection required.
        if vp_candidates.len() <= need - splits.len() {
            splits.extend(&vp_candidates);
        } else {
            // Greedy selection: always pick the most balanced VP split
            // (smallest segment penalty), stop when we have enough.
            while splits.len() < need && !vp_candidates.is_empty() {
                let mut best_idx: Option<usize> = None;
                let mut best_min_child: f32 = -1.0;
                for (i, &col) in vp_candidates.iter().enumerate() {
                    let seg_start = splits.iter().filter(|&&s| s < col).copied()
                        .max().unwrap_or(0);
                    let seg_end = splits.iter().filter(|&&s| s > col).copied()
                        .min().unwrap_or(w);
                    let left = (col - seg_start) as f32;
                    let right = (seg_end - col) as f32;
                    let min_child = left.min(right);
                    if min_child > best_min_child {
                        best_min_child = min_child;
                        best_idx = Some(i);
                    }
                }
                match best_idx {
                    Some(i) => { splits.push(vp_candidates.remove(i)); }
                    None => break,
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

    // Segment-size penalty: discourages splits that create a segment
    // narrower than the expected character width.
    // (10 * avg_char_width / min_child_width)^2, applied additively to seam cost.
    // Penalizes based on the smallest segment CREATED by the split,
    // so splitting near the edge costs more than splitting in the middle.
    let trailing_narrow_punct = word_text.map_or(false, |t| {
        matches!(t.as_bytes().last(), Some(b'.' | b',' | b':' | b';'))
    });
    let segment_penalty = |seg_start: u32, seg_end: u32, col: u32, cost: f32| -> f32 {
        if cost < 1.0 { return 0.0; }
        let left = (col - seg_start) as f32;
        let right = (seg_end - col) as f32;
        let mut min_child = left.min(right);
        if min_child <= 0.0 { return f32::MAX; }
        // Don't penalize narrow trailing punctuation (. , : ;)
        if trailing_narrow_punct && right < left && seg_end == w {
            min_child = left;
        }
        let p = 10.0 * avg_char_width / min_child;
        p * p
    };

    // --- Pass 2: greedy seam carving ---
    //
    // For remaining splits, find the cheapest vertical seam in each segment.
    // Greedy: pop cheapest, split, recompute children, repeat.
    let mut seam_paths: HashMap<u32, Vec<[u32; 2]>> = HashMap::new();
    let mut seam_costs: HashMap<u32, SeamCost> = HashMap::new();
    // Paths for top candidates (including unused) — traced at generation time
    // so we have them before DPs are consumed. Keyed by (col, cost_bits) to
    // avoid overwriting when same col appears in different segments.
    let mut candidate_paths: Vec<(u32, SeamCost, Vec<[u32; 2]>)> = Vec::new();  // (col, SeamCost, path)
    // Seam instrumentation: capture seed candidates and greedy-loop events
    // for the audit summary JSON.
    if splits.len() < need {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        let audit_mode = diag_dir.is_some();

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

        // Per-pixel darkness + row ink + total ink + ink_values in one scan,
        // TLS-reused arenas to avoid per-word Vec allocation (perf win 1).
        let raw_dark = img.as_raw();
        let w_us = w as usize;
        let h_us = h as usize;
        let n = w_us * h_us;
        // Take 5 bufs from TLS (capacity preserved across calls)
        let (mut darkness, mut energy, mut row_sums, mut row_ink, mut ink_values): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) =
            SEGMENT_TL.with(|c| std::mem::replace(&mut *c.borrow_mut(), (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())));
        darkness.clear(); darkness.resize(n, 0.0);
        row_sums.clear(); row_sums.resize(h_us, 0.0);
        energy.clear(); energy.resize(n, 0.0);
        row_ink.clear(); row_ink.resize(h_us, 0.0);
        ink_values.clear(); ink_values.reserve(n / 2);
        let mut total_ink: f32 = 0.0;
        for y in 0..h_us {
            let base = y * w_us;
            let mut row_sum = 0.0f32;
            for x in 0..w_us {
                let d = 255.0 - raw_dark[base + x] as f32;
                darkness[base + x] = d;
                row_sum += d;
                if d > 0.0 {
                    ink_values.push(d);
                }
            }
            row_sums[y] = row_sum;
            total_ink += row_sum;
        }

        // Row ink fractions: what share of the word's total ink is in each row.
        if total_ink > 0.0 {
            let inv_total = 1.0 / total_ink;
            for y in 0..h_us {
                row_ink[y] = row_sums[y] * inv_total;
            }
        }

        // Energy map: darkness with horizontal-context discount.
        // Flat vec: same layout as darkness (already sized).
        for y in 0..h_us {
            let row_off = y * w_us;
            for c in 0..w_us {
                let d = darkness[row_off + c];
                if d > 0.0 {
                    let left_avg = if c >= 2 {
                        (darkness[row_off + c - 1] + darkness[row_off + c - 2]) * 0.5
                    } else if c >= 1 {
                        darkness[row_off + c - 1]
                    } else {
                        d
                    };
                    let right_avg = if c + 2 < w_us {
                        (darkness[row_off + c + 1] + darkness[row_off + c + 2]) * 0.5
                    } else if c + 1 < w_us {
                        darkness[row_off + c + 1]
                    } else {
                        d
                    };
                    energy[row_off + c] = if left_avg > d && right_avg > d { d * 0.5 } else { d };
                } else {
                    energy[row_off + c] = d;
                }
            }
        }


        // Ink discount along a path: sum of (raw darkness - discounted energy)
        // for each pixel on the path. Flat layout: idx = r*w + c.
        let ink_discount_for_path = |path: &[[u32; 2]]| -> f32 {
            path.iter().map(|p| {
                let r = p[0] as usize;
                let c = p[1] as usize;
                let idx = r * w_us + c;
                darkness[idx] - energy[idx]
            }).sum()
        };

        // Factor: trace all candidate seam paths and record their costs.
        // Energy is now flat &[f32]; trace_path_through ignores it but keep param for compat.
        // Perf: sort cands in-place to avoid to_vec() clone (cands is small Vec, but called many times).
        let trace_candidate_costs = |cands: &mut [(u32, f32)], dp: &SeamDp,
            seg_start: u32, seg_end: u32, energy_flat: &[f32], row_ink: &[f32],
            seg_pen: &dyn Fn(u32, u32, u32, f32) -> f32,
            ink_disc: &dyn Fn(&[[u32; 2]]) -> f32,
            h_cost: f32,
            paths: &mut Vec<(u32, SeamCost, Vec<[u32; 2]>)>|
        {
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(col, cost) in cands.iter() {
                let path = dp.trace_path_through(energy_flat, col, row_ink);
                let p_min = path.iter().map(|p| p[1]).min().unwrap_or(col);
                let p_max = path.iter().map(|p| p[1]).max().unwrap_or(col);
                let pw = (p_max - p_min) as f32;
                let sp = seg_pen(seg_start, seg_end, (p_min + p_max) / 2, cost);
                let hm = path.windows(2).filter(|w| w[0][0] == w[1][0]).count() as f32;
                let id = ink_disc(&path);
                paths.push((col, SeamCost {
                    dp_cost: cost - pw - hm * h_cost,
                    seam_width_penalty: pw,
                    segment_size_penalty: sp,
                    horizontal_cost: hm * h_cost,
                    ink_discount: id,
                    total: cost + sp,
                }, path));
            }
        };

        // Word-level max ink (p95 of raw darkness): reused ink_values from darkness scan above.
        // select_nth avoids full sort O(n log n).
        let max_ink = if ink_values.is_empty() {
            255.0
        } else {
            let p95_idx = (ink_values.len() as f64 * 0.95) as usize;
            let p95_idx = p95_idx.min(ink_values.len() - 1);
            // select_nth_unstable_by is O(n) vs sort O(n log n) – ~22k pixels per word
            ink_values.select_nth_unstable_by(p95_idx, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let mut dp_cache: FxHashMap<u32, SeamDp> = FxHashMap::default();
        // Diagonal bounds per segment: seam paths that bound each side.
        // Pixels at or beyond these paths are unusable in the DP.
        // left_path[r] = seam col; pixels with col <= left_path[r] are masked.
        // right_path[r] = seam col; pixels with col >= right_path[r] are masked.
        // Perf: store as Arc to make clone cheap (atomic inc vs Vec copy O(h)).
        struct SegBounds {
            left_path: Option<std::sync::Arc<Vec<[u32; 2]>>>,
            right_path: Option<std::sync::Arc<Vec<[u32; 2]>>>,
        }
        let mut seg_bounds: FxHashMap<u32, SegBounds> = FxHashMap::default();
        let mut next_seg_id: u32 = 0;

        // Build initial segments from VP splits and seed the heap.
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
            if seg_end > seg_start + 2 {
                let sid = next_seg_id; next_seg_id += 1;
                let (mut cands, dp) = candidate_seams(&energy, w_us, seg_start, seg_end, h, None, None, max_ink, &row_ink);
                for (col, cost) in &cands {
                    heap.push(SeamEntry { cost: *cost + segment_penalty(seg_start, seg_end, *col, *cost), col: *col, seg_start, seg_end, seg_id: sid });
                }
                if audit_mode {
                    trace_candidate_costs(&mut cands, &dp, seg_start, seg_end, &energy, &row_ink,
                        &segment_penalty, &ink_discount_for_path, seam_params().horizontal_cost, &mut candidate_paths);
                }
                seg_bounds.insert(sid, SegBounds { left_path: None, right_path: None });
                dp_cache.insert(sid, dp);
            }
        }

        // Greedy loop: pop cheapest, split, recompute children.
        // Lazy deletion: stale seg_ids are skipped on pop instead of
        // draining and rebuilding the heap on every accepted seam.
        // Perf: Vec<bool> dense O(1) vs HashSet<u32> hash.
        let mut dead_sids: Vec<bool> = Vec::new();
        while splits.len() < need {
            let entry = match heap.pop() {
                Some(e) => e,
                None => break, // no valid seams remain
            };

            // Skip candidates from dead segments (replaced or consumed).
            if (entry.seg_id as usize) < dead_sids.len() && dead_sids[entry.seg_id as usize] {
                continue;
            }

            // Skip if this exact split column was already accepted.
            if splits.contains(&entry.col) {
                continue;
            }

            // Validate: both children must have meaningful ink.
            let left_ink = ink_extent(&col_has_ink_strict, entry.seg_start, entry.col);
            let right_ink = ink_extent(&col_has_ink_strict, entry.col + 1, entry.seg_end);
            let left_ok = left_ink.1 > left_ink.0 + 2;
            let right_ok = right_ink.1 > right_ink.0 + 2;

            if !left_ok || !right_ok {
                // Seam hugged an edge → retry with narrowed range.
                if !right_ok && left_ok {
                    let new_end = entry.col;
                    if new_end > entry.seg_start + 2 {
                        let sid = next_seg_id; next_seg_id += 1;
                        let parent_bounds = seg_bounds.get(&entry.seg_id);
                        let lp = parent_bounds.and_then(|b| b.left_path.clone());
                        let rp = parent_bounds.and_then(|b| b.right_path.clone());
                        let (cands, dp) = candidate_seams(&energy, w_us, entry.seg_start, new_end, h, lp.as_ref().map(|a| a.as_slice()), rp.as_ref().map(|a| a.as_slice()), max_ink, &row_ink);
                        for (col, cost) in &cands {
                            heap.push(SeamEntry { cost: *cost + segment_penalty(entry.seg_start, new_end, *col, *cost), col: *col, seg_start: entry.seg_start, seg_end: new_end, seg_id: sid });
                        }
                        seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                        dp_cache.insert(sid, dp);
                        // Kill old segment — retry replacement covers valid columns
                        if dead_sids.len() <= entry.seg_id as usize { dead_sids.resize((entry.seg_id as usize)+1, false); }
                        dead_sids[entry.seg_id as usize] = true;
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
                        let (cands, dp) = candidate_seams(&energy, w_us, new_start, entry.seg_end, h, lp.as_ref().map(|a| a.as_slice()), rp.as_ref().map(|a| a.as_slice()), max_ink, &row_ink);
                        for (col, cost) in &cands {
                            heap.push(SeamEntry { cost: *cost + segment_penalty(new_start, entry.seg_end, *col, *cost), col: *col, seg_start: new_start, seg_end: entry.seg_end, seg_id: sid });
                        }
                        seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                        dp_cache.insert(sid, dp);
                        // Kill old segment — retry replacement covers valid columns
                        if dead_sids.len() <= entry.seg_id as usize { dead_sids.resize((entry.seg_id as usize)+1, false); }
                        dead_sids[entry.seg_id as usize] = true;
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
            // Build per-row column lookups from [row, col] pair paths.
            let mut seam_by_row = vec![entry.col; h as usize];
            for p in &path { seam_by_row[p[0] as usize] = p[1]; }
            let mut lb_by_row = vec![entry.seg_start; h as usize];
            if let Some(lp) = left_bound {
                for p in lp { if (p[0] as usize) < lb_by_row.len() {
                    lb_by_row[p[0] as usize] = lb_by_row[p[0] as usize].max(p[1]);
                }}
            }
            let mut rb_by_row = vec![entry.seg_end; h as usize];
            if let Some(rp) = right_bound {
                for p in rp { if (p[0] as usize) < rb_by_row.len() {
                    rb_by_row[p[0] as usize] = rb_by_row[p[0] as usize].min(p[1]);
                }}
            }
            let raw_seam = img.as_raw();
            let w_us_seam = w as usize;
            let mut seam_ink_left: u32 = 0;
            let mut seam_ink_right: u32 = 0;
            for row in 0..h as usize {
                let base = row * w_us_seam;
                let seam_col = seam_by_row[row] as usize;
                let lb = lb_by_row[row] as usize;
                let rb = rb_by_row[row] as usize;
                for c in lb..seam_col.min(w_us_seam) {
                    if raw_seam[base + c] < crate::INK_THRESH {
                        seam_ink_left += 1;
                    }
                }
                for c in (seam_col + 1)..rb.min(w_us_seam) {
                    if raw_seam[base + c] < crate::INK_THRESH {
                        seam_ink_right += 1;
                    }
                }
            }
            if seam_ink_left < min_ink_for_symbol || seam_ink_right < min_ink_for_symbol {
                continue;
            }

            let final_col = entry.col;

            splits.push(final_col);
            // Perf: compute min/max/h_moves/id from &path directly, single pass, no iterator clone or map lookup
            let mut path_min_col = u32::MAX;
            let mut path_max_col = 0u32;
            for p in &path {
                let c = p[1];
                if c < path_min_col { path_min_col = c; }
                if c > path_max_col { path_max_col = c; }
            }
            if path_min_col == u32::MAX {
                path_min_col = final_col;
                path_max_col = final_col;
            }
            let swp = (path_max_col - path_min_col) as f32;
            let seg_pen = segment_penalty(entry.seg_start, entry.seg_end, (path_min_col + path_max_col) / 2, entry.cost);
            let h_moves = path.windows(2)
                .filter(|w| w[0][0] == w[1][0])
                .count() as f32;
            let id = ink_discount_for_path(&path);
            seam_costs.insert(final_col, SeamCost {
                dp_cost: entry.cost - seg_pen - swp - h_moves * seam_params().horizontal_cost,
                seam_width_penalty: swp,
                segment_size_penalty: seg_pen,
                horizontal_cost: h_moves * seam_params().horizontal_cost,
                ink_discount: id,
                total: entry.cost,
            });

            // Capture parent's diagonal bounds and DP before removing (Arc clone = cheap).
            let parent_lp = seg_bounds.get(&entry.seg_id).and_then(|b| b.left_path.clone());
            let parent_rp = seg_bounds.get(&entry.seg_id).and_then(|b| b.right_path.clone());
            let parent_dp_cloned: Option<SeamDp> = dp_cache.get(&entry.seg_id).map(|dp_ref| SeamDp {
                pred_fwd: dp_ref.pred_fwd.clone(),
                pred_rev: dp_ref.pred_rev.clone(),
                cost_fwd: dp_ref.cost_fwd.clone(),
                cost_rev: dp_ref.cost_rev.clone(),
                seg_start: dp_ref.seg_start,
                seg_w: dp_ref.seg_w,
                h: dp_ref.h,
            });

            // Perf: wrap path in Arc once (moves Vec, no clone), then Arc clones for bounds.
            // seam_paths map stores Vec for external API compat — clone once from Arc (1 Vec copy vs 2 before).
            let arc_path = std::sync::Arc::new(path);
            seam_paths.insert(final_col, (*arc_path).clone());

            // Mark old segment as dead — stale entries skipped on pop.
            let old_sid = entry.seg_id;
            if dead_sids.len() <= old_sid as usize { dead_sids.resize((old_sid as usize)+1, false); }
            dead_sids[old_sid as usize] = true;
            dp_cache.remove(&old_sid);
            seg_bounds.remove(&old_sid);

            // Recompute child segments.  Use raw boundaries for segment
            // identity and penalty; only narrow the DP search range when a
            // child side has no ink at all.
            let child_left_start = entry.seg_start;
            let child_left_end = final_col;
            let child_right_start = final_col + 1;
            let child_right_end = entry.seg_end;

            // Left child: inherits parent's left boundary, seam path as right boundary.
            {
                if child_left_end > child_left_start + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp = parent_lp.clone();
                    let rp: Option<std::sync::Arc<Vec<[u32; 2]>>> = Some(std::sync::Arc::clone(&arc_path));
                    let (mut cands, dp) = if let Some(ref pdp) = parent_dp_cloned {
                        candidate_seams_child(pdp, pdp.seg_start, &energy, w_us, child_left_start, child_left_end, h, lp.as_ref().map(|a| a.as_slice()), rp.as_ref().map(|a| a.as_slice()), max_ink, &row_ink)
                    } else {
                        candidate_seams(&energy, w_us, child_left_start, child_left_end, h, lp.as_ref().map(|a| a.as_slice()), rp.as_ref().map(|a| a.as_slice()), max_ink, &row_ink)
                    };
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost + segment_penalty(child_left_start, child_left_end, *col, *cost), col: *col, seg_start: child_left_start, seg_end: child_left_end, seg_id: sid });
                    }
                    if audit_mode {
                        trace_candidate_costs(&mut cands, &dp, child_left_start, child_left_end, &energy, &row_ink,
                            &segment_penalty, &ink_discount_for_path, seam_params().horizontal_cost, &mut candidate_paths);
                    }
                    seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                    dp_cache.insert(sid, dp);
                }
            }

            // Right child: seam path as left boundary, inherits parent's right boundary.
            {
                if child_right_end > child_right_start + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp: Option<std::sync::Arc<Vec<[u32; 2]>>> = Some(std::sync::Arc::clone(&arc_path));
                    let rp = parent_rp.clone();
                    let (mut cands, dp) = if let Some(ref pdp) = parent_dp_cloned {
                        candidate_seams_child(pdp, pdp.seg_start, &energy, w_us, child_right_start, child_right_end, h, lp.as_ref().map(|a| a.as_slice()), rp.as_ref().map(|a| a.as_slice()), max_ink, &row_ink)
                    } else {
                        candidate_seams(&energy, w_us, child_right_start, child_right_end, h, lp.as_ref().map(|a| a.as_slice()), rp.as_ref().map(|a| a.as_slice()), max_ink, &row_ink)
                    };
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost + segment_penalty(child_right_start, child_right_end, *col, *cost), col: *col, seg_start: child_right_start, seg_end: child_right_end, seg_id: sid });
                    }
                    if audit_mode {
                        trace_candidate_costs(&mut cands, &dp, child_right_start, child_right_end, &energy, &row_ink,
                            &segment_penalty, &ink_discount_for_path, seam_params().horizontal_cost, &mut candidate_paths);
                    }
                    seg_bounds.insert(sid, SegBounds { left_path: lp, right_path: rp });
                    dp_cache.insert(sid, dp);
                }
            }
        }

        // After greedy loop: merge unused candidate paths into seam_paths
        // for diagnostics. Only needed when auditing (win 2). Skip when audit_mode false.
        if audit_mode {
            let extra_limit: Option<usize> = match std::env::var("UNPRINT_EXTRA_SEAMS").ok().as_deref() {
                Some("all") | Some("ALL") => None,
                Some(n) => n.parse().ok(),
                None => Some(10),
            };
            candidate_paths.sort_by(|a, b| a.1.total.partial_cmp(&b.1.total).unwrap_or(std::cmp::Ordering::Equal));
            let mut added = 0usize;
            for (col, sc, path) in candidate_paths {
                if let Some(limit) = extra_limit {
                    if added >= limit { break; }
                }
                if seam_paths.contains_key(&col) { continue; }
                seam_paths.insert(col, path);
                seam_costs.insert(col, sc);
                added += 1;
            }
        }

        // Return TLS arenas (keep capacity for next word)
        SEGMENT_TL.with(|c| *c.borrow_mut() = (darkness, energy, row_sums, row_ink, ink_values));

        splits.sort();
    }

    let seam_splits: Vec<u32> = splits.iter().filter(|s| !vp_splits.contains(s)).copied().collect();

    if w == 492 && n_chars == 10 {
    }

    // Diag: dump seam pass
    if let Some(ddir) = diag_dir {
        crate::seg_diag::save_split_overlay_with_paths(img, &vp_splits, &seam_splits, &[], &seam_paths, &ddir.join("seam_overlay.png"));
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
    if let (Some(ddir), Some(_wtext)) = (diag_dir, word_text) {
        crate::seg_diag::save_split_overlay_with_paths(img, &vp_splits, &seam_splits, &[], &seam_paths, &ddir.join("final_overlay.png"));

        // NOTE: char crops are saved by extract_chars_from_boundaries
        // (the actual CI code path), not here — so diag shows exact CI inputs.
    }

    let n_segs = bounds.len().saturating_sub(1) as u32;
    let seg_summary = SegSummary {
        image_w: w,
        image_h: h,
        n_chars_expected: n_chars as u32,
        n_segments_produced: n_segs,
        mismatch: n_segs != n_chars as u32,
        ws_splits: vp_splits.clone(),
        seam_splits: seam_splits.clone(),
        seam_costs: seam_costs,
    };

    (bounds, seam_paths, seg_summary)
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
    pred_fwd: Vec<u32>, // flat [row * seg_w + col] — packed (r, c) predecessor
    pred_rev: Vec<u32>, // flat [row * seg_w + col] — packed (r, c) predecessor
    cost_fwd: Vec<f32>,
    cost_rev: Vec<f32>,
    seg_start: u32,
    seg_w: usize,
    h: u32,
}

impl SeamDp {

    /// Backtrace the cheapest path constrained to pass through
    /// `target_col` at mid-row.
    fn trace_path_through(&self, _energy: &[f32], target_col: u32, _row_ink: &[f32]) -> Vec<[u32; 2]> {
        let seg_w = self.seg_w;
        let mid_r = (self.h / 2) as usize;
        let last_r = (self.h - 1) as usize;
        let tc = (target_col - self.seg_start) as usize;

        // Top half: backtrace upward from (mid_r, tc) through pred_fwd.
        // Collected in backtrace order then reversed to get top-to-bottom.
        let mut top: Vec<[u32; 2]> = Vec::new();
        {
            let mut c = tc;
            let mut r = mid_r;
            top.push([r as u32, self.seg_start + c as u32]);
            while r > 0 {
                let pred = self.pred_fwd[r * seg_w + c] as usize;
                let pr = pred / seg_w;
                let pc = pred % seg_w;
                if pr < r {
                    let delta = pc as i32 - c as i32;
                    if delta != 0 {
                        // Diagonal: pass-through at (pred_row, current_col)
                        top.push([pr as u32, self.seg_start + c as u32]);
                        if delta.abs() == 2 {
                            // double horizontal: intermediate pass at (pr, c+delta/2)
                            let mid_c = c as i32 + delta / 2;
                            top.push([pr as u32, self.seg_start + mid_c as u32]);
                        }
                    }
                    top.push([pr as u32, self.seg_start + pc as u32]);
                    c = pc;
                    r = pr;
                    if r == 0 { break; }
                } else {
                    // Horizontal step on same row
                    top.push([r as u32, self.seg_start + pc as u32]);
                    c = pc;
                }
            }
        }
        top.reverse();

        // Bottom half: backtrace downward from (mid_r, tc) through pred_rev.
        // Already in top-to-bottom order; pass-throughs at (current_row, pred_col).
        let mut bottom: Vec<[u32; 2]> = Vec::new();
        {
            let mut c = tc;
            let mut r = mid_r;
            while r < last_r {
                let pred = self.pred_rev[r * seg_w + c] as usize;
                let pr = pred / seg_w;
                let pc = pred % seg_w;
                if pr > r {
                    let delta = pc as i32 - c as i32;
                    if delta != 0 {
                        if delta.abs() == 2 {
                            // double: first intermediate on current row
                            let mid_c = c as i32 + delta / 2;
                            bottom.push([r as u32, self.seg_start + mid_c as u32]);
                        }
                        // Diagonal: pass-through at (current_row, pred_col)
                        bottom.push([r as u32, self.seg_start + pc as u32]);
                    }
                    bottom.push([pr as u32, self.seg_start + pc as u32]);
                    c = pc;
                    r = pr;
                    if r >= last_r { break; }
                } else {
                    // Horizontal step on same row
                    bottom.push([r as u32, self.seg_start + pc as u32]);
                    c = pc;
                }
            }
        }

        top.extend(bottom);
        top
    }
}

#[allow(unused_assignments)]
fn candidate_seams(
    energy: &[f32],
    img_w: usize, // word image width, for flat indexing r*img_w + col
    seg_start: u32,
    seg_end: u32,
    h: u32,
    left_path: Option<&[[u32; 2]]>,   // pixels with col <= left bound are masked
    right_path: Option<&[[u32; 2]]>,  // pixels with col >= right bound are masked
    max_ink: f32,                     // p95 ink darkness — scales entry penalty
    row_ink: &[f32],                  // per-row ink fractions for scoring
) -> (Vec<(u32, f32)>, SeamDp) {
    let seg_w = (seg_end - seg_start) as usize;
    if seg_w < 3 || h < 1 {
        let dp = SeamDp { pred_fwd: Vec::new(), pred_rev: Vec::new(), cost_fwd: Vec::new(), cost_rev: Vec::new(), seg_start, seg_w: 0, h };
        return (Vec::new(), dp);
    }
    let base = seg_start as usize;
    let mid_r = (h / 2) as usize;

    // Build per-row boundary arrays from [row, col] pairs.
    // Left boundary: max column per row (most conservative mask).
    // Right boundary: min column per row (most conservative mask).
    // Perf: avoid per-call heap allocation for typical h <= 1024 by using
    // stack buffers.  Falls back to Vec only for unusually tall words.
    let h_us = h as usize;
    const BOUND_STACK_MAX: usize = 1024;
    let mut left_stack = [0u32; BOUND_STACK_MAX];
    let mut right_stack = [u32::MAX; BOUND_STACK_MAX];
    let mut left_heap: Option<Vec<u32>> = None;
    let mut right_heap: Option<Vec<u32>> = None;

    let left_bound: Option<&[u32]> = if let Some(lp) = left_path {
        if h_us <= BOUND_STACK_MAX {
            for entry in lp {
                let r = entry[0] as usize;
                if r < h_us {
                    let col = entry[1];
                    if col > left_stack[r] {
                        left_stack[r] = col;
                    }
                }
            }
            Some(&left_stack[..h_us])
        } else {
            let mut bound = vec![0u32; h_us];
            for entry in lp {
                let r = entry[0] as usize;
                if r < bound.len() && entry[1] > bound[r] {
                    bound[r] = entry[1];
                }
            }
            left_heap = Some(bound);
            Some(left_heap.as_ref().unwrap().as_slice())
        }
    } else {
        None
    };
    let right_bound: Option<&[u32]> = if let Some(rp) = right_path {
        if h_us <= BOUND_STACK_MAX {
            for entry in rp {
                let r = entry[0] as usize;
                if r < h_us {
                    let col = entry[1];
                    if col < right_stack[r] {
                        right_stack[r] = col;
                    }
                }
            }
            Some(&right_stack[..h_us])
        } else {
            let mut bound = vec![u32::MAX; h_us];
            for entry in rp {
                let r = entry[0] as usize;
                if r < bound.len() && entry[1] < bound[r] {
                    bound[r] = entry[1];
                }
            }
            right_heap = Some(bound);
            Some(right_heap.as_ref().unwrap().as_slice())
        }
    } else {
        None
    };

    // Masked energy: pixels outside diagonal boundaries are impassable.
    // Energy already includes the horizontal-context ink discount.
    // Flat layout: energy[r*img_w + abs_col]
    let masked_energy = |r: usize, c: usize| -> f32 {
        let abs_col = base + c;
        if let Some(lb) = left_bound {
            if abs_col <= lb[r] as usize { return f32::INFINITY; }
        }
        if let Some(rb) = right_bound {
            if abs_col >= rb[r] as usize { return f32::INFINITY; }
        }
        energy[r * img_w + abs_col]
    };

    // Forward DP: fused 5-predecessor single pass — cuts 5 scans of h*seg_w
    // down to 1, precomputes cur row darkness/ink once per cell.
    // Identical costs to prior 5-pass version: same INFINITY handling,
    // same 0.01/0.02 horizontal penalties, same pass-through ordering.
    // Arena reuse: thread_local buffers to avoid allocs per call. 6 buffers now include prev_inks.
    let _p = seam_params(); // hoist OnceLock load
    let n_cells = h_us * seg_w;
    // Take reused buffers from thread-local (capacity preserved across calls)
    let (mut cost_fwd, mut pred_fwd, mut prev_darks, mut cur_darks, mut cur_inks, mut prev_inks) =
        CANDIDATE_SEAMS_FWD.with(|c| {
            let mut s = c.borrow_mut();
            (
                std::mem::take(&mut s.0),
                std::mem::take(&mut s.1),
                std::mem::take(&mut s.2),
                std::mem::take(&mut s.3),
                std::mem::take(&mut s.4),
                std::mem::take(&mut s.5),
            )
        });
    cost_fwd.resize(n_cells, 0.0f32);
    // Ensure exact len n_cells (resize keeps extra initialized to 0)
    // pred_fwd similarly
    pred_fwd.resize(n_cells, 0u32);
    prev_darks.resize(seg_w, 0.0f32);
    cur_darks.resize(seg_w, 0.0f32);
    cur_inks.resize(seg_w, 0.0f32);
    prev_inks.resize(seg_w, 0.0f32);
    // Row 0 cached for reuse as prev in r=1
    for c in 0..seg_w { prev_darks[c] = masked_energy(0, c); }
    for c in 0..seg_w { prev_inks[c] = ink_score(prev_darks[c], 0, row_ink); }
    for c in 0..seg_w {
        let dark0 = prev_darks[c];
        cost_fwd[c] = prev_inks[c]
            + delta_ink_score(dark0, 0.0, 0, 0, row_ink, max_ink);
        pred_fwd[c] = c as u32;
    }
    // cur_darks/cur_inks reused buffers already resized
    for r in 1..h_us {
        for c in 0..seg_w {
            let cd = masked_energy(r, c);
            cur_darks[c] = cd;
            cur_inks[c] = ink_score(cd, r, row_ink);
        }
        let row_off = r * seg_w;
        let prev_off = (r - 1) * seg_w;
        // Single fused predecessor evaluation – prev_inks reuse avoids 5-6 ink_score calls per cell
        for c in 0..seg_w {
            let cur_dark = cur_darks[c];
            // masked columns remain INF — keep downstream INF propagation same as before
            if !cur_dark.is_finite() {
                cost_fwd[row_off + c] = f32::INFINITY;
                pred_fwd[row_off + c] = (prev_off + c) as u32;
                continue;
            }
            let cur_ink = cur_inks[c];
            // vertical from (r-1,c)
            let prev_dark_c = prev_darks[c];
            let mut best = cur_ink + delta_ink_score(cur_dark, prev_dark_c, r, r - 1, row_ink, max_ink)
                + cost_fwd[prev_off + c];
            let mut best_pred = prev_off + c;
            // diag -1 : (r-1,c-1) via (r-1,c)
            if c >= 1 {
                let pd_m1 = prev_darks[c - 1];
                let pass_dark = prev_dark_c;
                let pass_ink = prev_inks[c]; // reuse – same as ink_score(pass_dark, r-1)
                let pass_entry = delta_ink_score(pass_dark, pd_m1, r - 1, r - 1, row_ink, max_ink);
                let cur_entry = delta_ink_score(cur_dark, pass_dark, r, r - 1, row_ink, max_ink);
                let via = cost_fwd[prev_off + c - 1] + pass_ink + pass_entry + cur_ink + cur_entry + 0.01;
                if via < best { best = via; best_pred = prev_off + c - 1; }
            }
            // diag +1 : (r-1,c+1) via (r-1,c)
            if c + 1 < seg_w {
                let pd_p1 = prev_darks[c + 1];
                let pass_dark = prev_dark_c;
                let pass_ink = prev_inks[c];
                let pass_entry = delta_ink_score(pass_dark, pd_p1, r - 1, r - 1, row_ink, max_ink);
                let cur_entry = delta_ink_score(cur_dark, pass_dark, r, r - 1, row_ink, max_ink);
                let via = cost_fwd[prev_off + c + 1] + pass_ink + pass_entry + cur_ink + cur_entry + 0.01;
                if via < best { best = via; best_pred = prev_off + c + 1; }
            }
            // double -2 : (r-1,c-2) via c-1,c
            if c >= 2 {
                let pd_m2 = prev_darks[c - 2];
                let p1_ink = prev_inks[c - 1];
                let p2_ink = prev_inks[c];
                let p1d = prev_darks[c - 1];
                let p2d = prev_dark_c;
                let p1_entry = delta_ink_score(p1d, pd_m2, r - 1, r - 1, row_ink, max_ink);
                let p2_entry = delta_ink_score(p2d, p1d, r - 1, r - 1, row_ink, max_ink);
                let cur_entry = delta_ink_score(cur_dark, p2d, r, r - 1, row_ink, max_ink);
                let via = cost_fwd[prev_off + c - 2] + p1_ink + p1_entry + p2_ink + p2_entry + cur_ink + cur_entry + 0.02;
                if via < best { best = via; best_pred = prev_off + c - 2; }
            }
            // double +2 : (r-1,c+2) via c+1,c
            if c + 2 < seg_w {
                let pd_p2 = prev_darks[c + 2];
                let p1_ink = prev_inks[c + 1];
                let p2_ink = prev_inks[c];
                let p1d = prev_darks[c + 1];
                let p2d = prev_dark_c;
                let p1_entry = delta_ink_score(p1d, pd_p2, r - 1, r - 1, row_ink, max_ink);
                let p2_entry = delta_ink_score(p2d, p1d, r - 1, r - 1, row_ink, max_ink);
                let cur_entry = delta_ink_score(cur_dark, p2d, r, r - 1, row_ink, max_ink);
                let via = cost_fwd[prev_off + c + 2] + p1_ink + p1_entry + p2_ink + p2_entry + cur_ink + cur_entry + 0.02;
                if via < best { best = via; best_pred = prev_off + c + 2; }
            }
            cost_fwd[row_off + c] = best;
            pred_fwd[row_off + c] = best_pred as u32;
        }
        std::mem::swap(&mut prev_darks, &mut cur_darks);
        std::mem::swap(&mut prev_inks, &mut cur_inks);
    }

    // Reverse DP fused single pass — symmetric to forward, now reused via thread_local
    let last_r = (h - 1) as usize;
    let (mut cost_rev, mut pred_rev, mut next_darks, mut cur_darks_rev, mut cur_inks_rev, mut next_inks) =
        CANDIDATE_SEAMS_REV.with(|c| {
            let mut s = c.borrow_mut();
            (
                std::mem::take(&mut s.0),
                std::mem::take(&mut s.1),
                std::mem::take(&mut s.2),
                std::mem::take(&mut s.3),
                std::mem::take(&mut s.4),
                std::mem::take(&mut s.5),
            )
        });
    cost_rev.resize(n_cells, 0.0f32);
    pred_rev.resize(n_cells, 0u32);
    next_darks.resize(seg_w, 0.0f32);
    cur_darks_rev.resize(seg_w, 0.0f32);
    cur_inks_rev.resize(seg_w, 0.0f32);
    next_inks.resize(seg_w, 0.0f32);
    for c in 0..seg_w { next_darks[c] = masked_energy(last_r, c); }
    let last_off = last_r * seg_w;
    for c in 0..seg_w {
        let dark_last = next_darks[c];
        cost_rev[last_off + c] = ink_score(dark_last, last_r, row_ink)
            + delta_ink_score(dark_last, 0.0, last_r, last_r, row_ink, max_ink);
        pred_rev[last_off + c] = (last_off + c) as u32;
    }
    // next row inks cached for pass costs
    // rebuild each iteration, start with last row values
    for c in 0..seg_w { next_inks[c] = ink_score(next_darks[c], last_r, row_ink); }
    for r in (0..last_r).rev() {
        for c in 0..seg_w {
            let cd = masked_energy(r, c);
            cur_darks_rev[c] = cd;
            cur_inks_rev[c] = ink_score(cd, r, row_ink);
        }
        let row_off = r * seg_w;
        let next_off = (r + 1) * seg_w;
        for c in 0..seg_w {
            let cur_dark = cur_darks_rev[c];
            if !cur_dark.is_finite() {
                cost_rev[row_off + c] = f32::INFINITY;
                pred_rev[row_off + c] = (next_off + c) as u32;
                continue;
            }
            let cur_ink = cur_inks_rev[c];
            // vertical from (r+1,c)
            let child_dark_v = next_darks[c];
            let entry_v = delta_ink_score(child_dark_v, cur_dark, r + 1, r, row_ink, max_ink);
            let mut best = cur_ink + entry_v + cost_rev[next_off + c];
            let mut best_pred = next_off + c;
            // diag -1 : (r+1,c-1) via (r,c-1)
            if c >= 1 {
                let pass_dark = cur_darks_rev[c - 1];
                let pass_ink = cur_inks_rev[c - 1];
                let pass_entry = delta_ink_score(pass_dark, cur_dark, r, r, row_ink, max_ink);
                let child_dark = next_darks[c - 1];
                let child_entry = delta_ink_score(child_dark, pass_dark, r + 1, r, row_ink, max_ink);
                let via = cost_rev[next_off + c - 1] + cur_ink + pass_ink + pass_entry + child_entry + 0.01;
                if via < best { best = via; best_pred = next_off + c - 1; }
            }
            // diag +1
            if c + 1 < seg_w {
                let pass_dark = cur_darks_rev[c + 1];
                let pass_ink = cur_inks_rev[c + 1];
                let pass_entry = delta_ink_score(pass_dark, cur_dark, r, r, row_ink, max_ink);
                let child_dark = next_darks[c + 1];
                let child_entry = delta_ink_score(child_dark, pass_dark, r + 1, r, row_ink, max_ink);
                let via = cost_rev[next_off + c + 1] + cur_ink + pass_ink + pass_entry + child_entry + 0.01;
                if via < best { best = via; best_pred = next_off + c + 1; }
            }
            // double -2 via (r,c-1),(r,c-2)
            if c >= 2 {
                let p1d = cur_darks_rev[c - 1];
                let p2d = cur_darks_rev[c - 2];
                let p1ink = cur_inks_rev[c - 1];
                let p2ink = cur_inks_rev[c - 2];
                let p1_entry = delta_ink_score(p1d, cur_dark, r, r, row_ink, max_ink);
                let p2_entry = delta_ink_score(p2d, p1d, r, r, row_ink, max_ink);
                let child_dark = next_darks[c - 2];
                let child_entry = delta_ink_score(child_dark, p2d, r + 1, r, row_ink, max_ink);
                let via = cost_rev[next_off + c - 2] + cur_ink + p1ink + p1_entry + p2ink + p2_entry + child_entry + 0.02;
                if via < best { best = via; best_pred = next_off + c - 2; }
            }
            // double +2
            if c + 2 < seg_w {
                let p1d = cur_darks_rev[c + 1];
                let p2d = cur_darks_rev[c + 2];
                let p1ink = cur_inks_rev[c + 1];
                let p2ink = cur_inks_rev[c + 2];
                let p1_entry = delta_ink_score(p1d, cur_dark, r, r, row_ink, max_ink);
                let p2_entry = delta_ink_score(p2d, p1d, r, r, row_ink, max_ink);
                let child_dark = next_darks[c + 2];
                let child_entry = delta_ink_score(child_dark, p2d, r + 1, r, row_ink, max_ink);
                let via = cost_rev[next_off + c + 2] + cur_ink + p1ink + p1_entry + p2ink + p2_entry + child_entry + 0.02;
                if via < best { best = via; best_pred = next_off + c + 2; }
            }
            cost_rev[row_off + c] = best;
            pred_rev[row_off + c] = best_pred as u32;
        }
        std::mem::swap(&mut next_darks, &mut cur_darks_rev);
        std::mem::swap(&mut next_inks, &mut cur_inks_rev);
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
        let split_col = seg_start + c as u32;
        dp_candidates.push((split_col, combined));
    }

    // Second pass: trace each DP candidate's path and adjust cost
    // dp_candidates already emitted in column order (1..seg_w-1), sort redundant.

    // Find local minima in DP cost (before segment penalty).
    // A local minimum is a column (or run of equal-cost consecutive columns)
    // whose cost is strictly less than both neighbors.  Among equal-cost runs
    // that form a local minimum, pick the middle column (maximizes distance
    // from ink on both sides).  Boundary columns are treated as having
    // infinite cost, so an edge candidate is a local minimum if the first
    // interior neighbor is higher.
    let mut candidates: Vec<(u32, f32)> = Vec::new();
    let n = dp_candidates.len();
    if n > 0 {
        let mut i = 0;
        while i < n {
            let cost = dp_candidates[i].1;
            let run_start = i;
            // Extend through consecutive columns with equal cost.
            while i < n
                && dp_candidates[i].1 == cost
                && (i == run_start || dp_candidates[i].0 == dp_candidates[i - 1].0 + 1)
            {
                i += 1;
            }
            // Check local-minimum condition: both neighbors must be strictly higher.
            let left_higher = run_start == 0 || dp_candidates[run_start - 1].1 > cost;
            let right_higher = i >= n || dp_candidates[i].1 > cost;
            if left_higher && right_higher {
                let mid_idx = (run_start + i - 1) / 2;
                candidates.push(dp_candidates[mid_idx]);
            }
        }
    }

    // Return intermediate scratch buffers to thread-local for reuse (keep max capacity)
    let cost_fwd_cap = cost_fwd.capacity();
    let pred_fwd_cap = pred_fwd.capacity();
    let cost_rev_cap = cost_rev.capacity();
    let pred_rev_cap = pred_rev.capacity();
    CANDIDATE_SEAMS_FWD.with(|c| {
        let mut s = c.borrow_mut();
        s.2 = prev_darks;
        s.3 = cur_darks;
        s.4 = cur_inks;
        s.5 = prev_inks;
        s.0 = Vec::with_capacity(cost_fwd_cap);
        s.1 = Vec::with_capacity(pred_fwd_cap);
    });
    CANDIDATE_SEAMS_REV.with(|c| {
        let mut s = c.borrow_mut();
        s.2 = next_darks;
        s.3 = cur_darks_rev;
        s.4 = cur_inks_rev;
        s.5 = next_inks;
        s.0 = Vec::with_capacity(cost_rev_cap);
        s.1 = Vec::with_capacity(pred_rev_cap);
    });

    let dp = SeamDp { pred_fwd, pred_rev, cost_fwd, cost_rev, seg_start, seg_w, h };
    (candidates, dp)
}

/// Incremental DP child reuse: given parent SeamDp (with costs), trim-copy to child
/// segment and invalidate cells whose costs are too low because they used an
/// out-of-bounds parent cell (masked or outside child range). Propagate invalidity
/// forward (and reverse), then recompute only invalid region.
#[allow(clippy::too_many_arguments)]
fn candidate_seams_child(
    parent: &SeamDp,
    _parent_seg_start: u32, // == parent.seg_start, kept for clarity
    energy: &[f32],
    img_w: usize,
    child_start: u32,
    child_end: u32,
    h: u32,
    left_path: Option<&[[u32; 2]]>,
    right_path: Option<&[[u32; 2]]>,
    max_ink: f32,
    row_ink: &[f32],
) -> (Vec<(u32, f32)>, SeamDp) {
    let seg_w = (child_end - child_start) as usize;
    let h_us = h as usize;
    if seg_w < 3 || h_us < 1 {
        let dp = SeamDp { pred_fwd: Vec::new(), pred_rev: Vec::new(), cost_fwd: Vec::new(), cost_rev: Vec::new(), seg_start: child_start, seg_w: 0, h };
        return (Vec::new(), dp);
    }
    let base = child_start as usize;
    let parent_w = parent.seg_w;
    let off_i32 = child_start as i32 - parent.seg_start as i32;
    // Even if child not inside parent, clamp off and continue via invalid-propagation recompute — no fallback
    let mut off = if off_i32 < 0 { 0usize } else { off_i32 as usize };
    let mut force_full_invalid = false;
    if parent_w == 0 || off + seg_w > parent_w {
        // parent empty or child straddles outside parent — force full recompute via invalid path
        force_full_invalid = true;
        if parent_w == 0 {
            off = 0;
        } else if off >= parent_w {
            off = 0;
        } else if off + seg_w > parent_w {
            // keep off as-is but we'll treat out-of-range copy as invalid
            // clamp for array indexing safety, remainder handled by invalid logic
            if off + seg_w > parent_w && parent_w >= seg_w {
                off = parent_w - seg_w;
            }
        }
    }

    // Build per-row boundary arrays same as candidate_seams
    const BOUND_STACK_MAX: usize = 1024;
    let mut left_stack = [0u32; BOUND_STACK_MAX];
    let mut right_stack = [u32::MAX; BOUND_STACK_MAX];
    let left_heap: Option<Vec<u32>> = if left_path.is_some() && h_us > BOUND_STACK_MAX {
        let lp = left_path.unwrap();
        let mut b = vec![0u32; h_us];
        for e in lp {
            let r = e[0] as usize;
            if r < b.len() && e[1] > b[r] { b[r] = e[1]; }
        }
        Some(b)
    } else { None };
    let right_heap: Option<Vec<u32>> = if right_path.is_some() && h_us > BOUND_STACK_MAX {
        let rp = right_path.unwrap();
        let mut b = vec![u32::MAX; h_us];
        for e in rp {
            let r = e[0] as usize;
            if r < b.len() && e[1] < b[r] { b[r] = e[1]; }
        }
        Some(b)
    } else { None };

    let left_bound: Option<&[u32]> = if let Some(lp) = left_path {
        if h_us <= BOUND_STACK_MAX {
            for e in lp {
                let r = e[0] as usize;
                if r < h_us && e[1] > left_stack[r] {
                    left_stack[r] = e[1];
                }
            }
            Some(&left_stack[..h_us])
        } else {
            Some(left_heap.as_ref().unwrap().as_slice())
        }
    } else { None };
    let right_bound: Option<&[u32]> = if let Some(rp) = right_path {
        if h_us <= BOUND_STACK_MAX {
            for e in rp {
                let r = e[0] as usize;
                if r < h_us && e[1] < right_stack[r] {
                    right_stack[r] = e[1];
                }
            }
            Some(&right_stack[..h_us])
        } else {
            Some(right_heap.as_ref().unwrap().as_slice())
        }
    } else { None };

    // masked_energy_child
    let masked_energy_child = |r: usize, c: usize| -> f32 {
        let abs_col = base + c;
        if let Some(lb) = left_bound {
            if abs_col <= lb[r] as usize { return f32::INFINITY; }
        }
        if let Some(rb) = right_bound {
            if abs_col >= rb[r] as usize { return f32::INFINITY; }
        }
        energy[r * img_w + abs_col]
    };

    let n_cells = h_us * seg_w;
    let mut cost_fwd = vec![f32::INFINITY; n_cells];
    let mut pred_fwd = vec![0u32; n_cells];
    let mut cost_rev = vec![f32::INFINITY; n_cells];
    let mut pred_rev = vec![0u32; n_cells];
    let mut invalid_fwd = vec![false; n_cells];
    let mut invalid_rev = vec![false; n_cells];

    // ----- Trim-copy forward -----
    // If parent empty or child outside parent, force full invalid — no fallback to separate full-DP path
    if !force_full_invalid {
    for r in 0..h_us {
        let parent_row_off = r * parent_w;
        let child_row_off = r * seg_w;
        for c in 0..seg_w {
            let pc = off + c;
            if pc >= parent_w { // out-of-range for this row due to clamping
                let c_idx = child_row_off + c;
                invalid_fwd[c_idx] = true;
                cost_fwd[c_idx] = f32::INFINITY;
                continue;
            }
            let p_idx = parent_row_off + pc;
            let c_idx = child_row_off + c;
            // mask in child?
            let child_dark = masked_energy_child(r, c);
            if !child_dark.is_finite() {
                // INF cell — keep INF, mark invalid so we know not to use as valid predecessor?
                // We keep invalid flag to propagate INF outward but skip recompute later.
                cost_fwd[c_idx] = f32::INFINITY;
                pred_fwd[c_idx] = (parent_row_off + pc) as u32; // dummy
                invalid_fwd[c_idx] = true; // treat as invalid source
                continue;
            }
            // parent cost: if parent INF (parent masked) but child not masked, need recompute
            if parent.cost_fwd.is_empty() {
                // no parent costs — mark invalid for recompute via single path
                invalid_fwd[c_idx] = true;
                continue;
            }
            let p_cost = parent.cost_fwd[p_idx];
            // check parent pred out-of-range
            let p_pred_flat = parent.pred_fwd[p_idx] as usize;
            let pr = p_pred_flat / parent_w;
            let p_pred_c = p_pred_flat % parent_w;
            let out_of_range = if r == 0 {
                false
            } else {
                p_pred_c < off || p_pred_c >= off + seg_w
            };
            if out_of_range || !p_cost.is_finite() {
                invalid_fwd[c_idx] = true;
                // keep placeholder cost (will recompute)
                cost_fwd[c_idx] = f32::INFINITY;
                continue;
            }
            // valid copy — remap pred flat
            let remapped_pred_c = p_pred_c - off;
            let remapped_pred = (pr * seg_w + remapped_pred_c) as u32;
            // For r>0, pr should be r-1; remapped row same
            cost_fwd[c_idx] = p_cost;
            pred_fwd[c_idx] = remapped_pred;
        }
    }
    } else {
        // force full invalid region — skip trim-copy, all forward cells need recompute
        for i in 0..n_cells { invalid_fwd[i] = true; }
    }

    // Propagate invalid forward: if predecessor invalid then child invalid (too-low cost via invalid source)
    for r in 1..h_us {
        let row_off = r * seg_w;
        let prev_off = (r - 1) * seg_w;
        for c in 0..seg_w {
            let idx = row_off + c;
            if invalid_fwd[idx] { continue; } // already invalid
            // masked already filtered
            let pred = pred_fwd[idx] as usize;
            let pr = pred / seg_w;
            let pc = pred % seg_w;
            if pr != r - 1 {
                // horizontal with same row not expected in our DP; if pr==r we look at same row? ignore
                if pr == r {
                    // pred within same row — dependency inside row, but our DP order is left-right? predecessor still from same row not computed earlier in copy? treat as invalid if that same-row pred invalid.
                    // For safety, if pc invalid flag in same row and that cell already marked invalid and appears earlier in evaluation order we could propagate, but we use strict < comparison later recompute will fix.
                    if invalid_fwd[pr * seg_w + pc] {
                        invalid_fwd[idx] = true;
                    }
                    continue;
                }
            }
            if invalid_fwd[prev_off + pc] {
                invalid_fwd[idx] = true;
            }
        }
    }

    // Recompute invalid forward cells row-major
    // Need per-row dark caches for speed similar to fused loop
    // We'll recompute on the fly with masked closure; skip if not invalid
    let mut prev_darks_cache: Vec<f32> = vec![0.0; seg_w];
    for c in 0..seg_w {
        prev_darks_cache[c] = masked_energy_child(0, c);
    }
    // Row 0: any invalid row0 cells recompute directly (no predecessor)
    for c in 0..seg_w {
        let idx = c;
        if !invalid_fwd[idx] { continue; }
        let cd = masked_energy_child(0, c);
        if !cd.is_finite() {
            cost_fwd[idx] = f32::INFINITY;
            pred_fwd[idx] = c as u32;
        } else {
            let ic = ink_score(cd, 0, row_ink) + delta_ink_score(cd, 0.0, 0, 0, row_ink, max_ink);
            cost_fwd[idx] = ic;
            pred_fwd[idx] = c as u32;
            invalid_fwd[idx] = false; // fixed
        }
    }
    let mut cur_darks = vec![0.0f32; seg_w];
    let mut cur_inks = vec![0.0f32; seg_w];
    for r in 1..h_us {
        // build cur row dark/ink caches
        for c in 0..seg_w {
            let cd = masked_energy_child(r, c);
            cur_darks[c] = cd;
            cur_inks[c] = if cd.is_finite() { ink_score(cd, r, row_ink) } else { f32::INFINITY };
        }
        let row_off = r * seg_w;
        let prev_off = (r - 1) * seg_w;
        // prev_darks_cache holds row r-1 darks (already)
        for c in 0..seg_w {
            let idx = row_off + c;
            if !invalid_fwd[idx] { continue; }
            let cur_dark = cur_darks[c];
            if !cur_dark.is_finite() {
                cost_fwd[idx] = f32::INFINITY;
                pred_fwd[idx] = (prev_off + c) as u32;
                // remain invalid? leave as true to block downstream but INF propagation is okay
                // keep true? Actually INF cells are considered invalid sources but we keep flag true.
                continue;
            }
            let cur_ink = cur_inks[c];
            let prev_dark_c = prev_darks_cache[c];
            // vertical option
            let mut best = f32::INFINITY;
            let mut best_pred = prev_off + c;
            if cost_fwd[prev_off + c].is_finite() {
                best = cur_ink + delta_ink_score(cur_dark, prev_dark_c, r, r - 1, row_ink, max_ink) + cost_fwd[prev_off + c];
            }
            // diag -1
            if c >= 1 {
                let pd_m1 = prev_darks_cache[c - 1];
                if pd_m1.is_finite() && cost_fwd[prev_off + c - 1].is_finite() {
                    let pass_dark = prev_dark_c;
                    if pass_dark.is_finite() {
                        let pass_ink = ink_score(pass_dark, r - 1, row_ink);
                        let pass_entry = delta_ink_score(pass_dark, pd_m1, r - 1, r - 1, row_ink, max_ink);
                        let cur_entry = delta_ink_score(cur_dark, pass_dark, r, r - 1, row_ink, max_ink);
                        let via = cost_fwd[prev_off + c - 1] + pass_ink + pass_entry + cur_ink + cur_entry + 0.01;
                        if via < best { best = via; best_pred = prev_off + c - 1; }
                    }
                }
            }
            if c + 1 < seg_w {
                let pd_p1 = prev_darks_cache[c + 1];
                if pd_p1.is_finite() && cost_fwd[prev_off + c + 1].is_finite() {
                    let pass_dark = prev_dark_c;
                    if pass_dark.is_finite() {
                        let pass_ink = ink_score(pass_dark, r - 1, row_ink);
                        let pass_entry = delta_ink_score(pass_dark, pd_p1, r - 1, r - 1, row_ink, max_ink);
                        let cur_entry = delta_ink_score(cur_dark, pass_dark, r, r - 1, row_ink, max_ink);
                        let via = cost_fwd[prev_off + c + 1] + pass_ink + pass_entry + cur_ink + cur_entry + 0.01;
                        if via < best { best = via; best_pred = prev_off + c + 1; }
                    }
                }
            }
            if c >= 2 {
                let pd_m2 = prev_darks_cache[c - 2];
                if pd_m2.is_finite() && cost_fwd[prev_off + c - 2].is_finite() {
                    let p1d = prev_darks_cache[c - 1];
                    let p2d = prev_dark_c;
                    if p1d.is_finite() && p2d.is_finite() {
                        let p1_ink = ink_score(p1d, r - 1, row_ink);
                        let p2_ink = ink_score(p2d, r - 1, row_ink);
                        let p1_entry = delta_ink_score(p1d, pd_m2, r - 1, r - 1, row_ink, max_ink);
                        let p2_entry = delta_ink_score(p2d, p1d, r - 1, r - 1, row_ink, max_ink);
                        let cur_entry = delta_ink_score(cur_dark, p2d, r, r - 1, row_ink, max_ink);
                        let via = cost_fwd[prev_off + c - 2] + p1_ink + p1_entry + p2_ink + p2_entry + cur_ink + cur_entry + 0.02;
                        if via < best { best = via; best_pred = prev_off + c - 2; }
                    }
                }
            }
            if c + 2 < seg_w {
                let pd_p2 = prev_darks_cache[c + 2];
                if pd_p2.is_finite() && cost_fwd[prev_off + c + 2].is_finite() {
                    let p1d = prev_darks_cache[c + 1];
                    let p2d = prev_dark_c;
                    if p1d.is_finite() && p2d.is_finite() {
                        let p1_ink = ink_score(p1d, r - 1, row_ink);
                        let p2_ink = ink_score(p2d, r - 1, row_ink);
                        let p1_entry = delta_ink_score(p1d, pd_p2, r - 1, r - 1, row_ink, max_ink);
                        let p2_entry = delta_ink_score(p2d, p1d, r - 1, r - 1, row_ink, max_ink);
                        let cur_entry = delta_ink_score(cur_dark, p2d, r, r - 1, row_ink, max_ink);
                        let via = cost_fwd[prev_off + c + 2] + p1_ink + p1_entry + p2_ink + p2_entry + cur_ink + cur_entry + 0.02;
                        if via < best { best = via; best_pred = prev_off + c + 2; }
                    }
                }
            }
            cost_fwd[idx] = best;
            pred_fwd[idx] = best_pred as u32;
            if best.is_finite() { invalid_fwd[idx] = false; }
        }
        // swap caches: cur becomes prev for next iteration
        std::mem::swap(&mut prev_darks_cache, &mut cur_darks);
    }

    // ---- Reverse DP trim-copy ----
    let last_r = h_us - 1;
    if !force_full_invalid {
    for r in 0..h_us {
        let pr_off = r * parent_w;
        let ch_off = r * seg_w;
        for c in 0..seg_w {
            let pc = off + c;
            if pc >= parent_w {
                let c_idx = ch_off + c;
                cost_rev[c_idx] = f32::INFINITY;
                invalid_rev[c_idx] = true;
                continue;
            }
            let p_idx = pr_off + pc;
            let c_idx = ch_off + c;
            let cd = masked_energy_child(r, c);
            if !cd.is_finite() {
                cost_rev[c_idx] = f32::INFINITY;
                invalid_rev[c_idx] = true;
                continue;
            }
            if parent.cost_rev.is_empty() {
                invalid_rev[c_idx] = true;
                continue;
            }
            let p_cost = parent.cost_rev[p_idx];
            let p_pred_flat = parent.pred_rev[p_idx] as usize;
            let pr_ = p_pred_flat / parent_w;
            let pc_ = p_pred_flat % parent_w;
            let out_of_range = if r == last_r { false } else { pc_ < off || pc_ >= off + seg_w };
            if out_of_range || !p_cost.is_finite() {
                invalid_rev[c_idx] = true;
                cost_rev[c_idx] = f32::INFINITY;
                continue;
            }
            let remapped_c = pc_ - off;
            let remapped = pr_ * seg_w + remapped_c;
            cost_rev[c_idx] = p_cost;
            pred_rev[c_idx] = remapped as u32;
        }
    }
    } else {
        for i in 0..n_cells { invalid_rev[i] = true; }
    }
    // propagate invalid upward
    for r in (0..last_r).rev() {
        let row_off = r * seg_w;
        let next_off = (r + 1) * seg_w;
        for c in 0..seg_w {
            let idx = row_off + c;
            if invalid_rev[idx] { continue; }
            let pred = pred_rev[idx] as usize;
            let pc = pred % seg_w;
            let pr = pred / seg_w;
            if pr == r + 1 {
                if invalid_rev[next_off + pc] { invalid_rev[idx] = true; }
            } else if pr == r {
                if invalid_rev[row_off + pc] { invalid_rev[idx] = true; }
            }
        }
    }
    // recompute invalid reverse bottom-up
    let mut next_darks_rev: Vec<f32> = vec![0.0; seg_w];
    for c in 0..seg_w { next_darks_rev[c] = masked_energy_child(last_r, c); }
    let mut next_inks_rev: Vec<f32> = vec![0.0; seg_w];
    for c in 0..seg_w { next_inks_rev[c] = if next_darks_rev[c].is_finite() { ink_score(next_darks_rev[c], last_r, row_ink) } else { f32::INFINITY }; }
    // last row already set for invalid cells
    for c in 0..seg_w {
        let idx = last_r * seg_w + c;
        if invalid_rev[idx] {
            let cd = next_darks_rev[c];
            if cd.is_finite() {
                cost_rev[idx] = ink_score(cd, last_r, row_ink) + delta_ink_score(cd, 0.0, last_r, last_r, row_ink, max_ink);
                pred_rev[idx] = idx as u32;
                invalid_rev[idx] = false;
            } else {
                cost_rev[idx] = f32::INFINITY;
            }
        }
    }
    let mut cur_darks_rev = vec![0.0f32; seg_w];
    let mut cur_inks_rev = vec![0.0f32; seg_w];
    for r in (0..last_r).rev() {
        for c in 0..seg_w {
            let cd = masked_energy_child(r, c);
            cur_darks_rev[c] = cd;
            cur_inks_rev[c] = if cd.is_finite() { ink_score(cd, r, row_ink) } else { f32::INFINITY };
        }
        let row_off = r * seg_w;
        let next_off = (r + 1) * seg_w;
        for c in 0..seg_w {
            let idx = row_off + c;
            if !invalid_rev[idx] { continue; }
            let cur_dark = cur_darks_rev[c];
            if !cur_dark.is_finite() {
                cost_rev[idx] = f32::INFINITY;
                pred_rev[idx] = (next_off + c) as u32;
                continue;
            }
            let cur_ink = cur_inks_rev[c];
            let child_dark_v = next_darks_rev[c];
            let mut best = f32::INFINITY;
            let mut best_pred = next_off + c;
            if child_dark_v.is_finite() && cost_rev[next_off + c].is_finite() {
                let entry_v = delta_ink_score(child_dark_v, cur_dark, r + 1, r, row_ink, max_ink);
                best = cur_ink + entry_v + cost_rev[next_off + c];
            }
            if c >= 1 {
                let pass_dark = cur_darks_rev[c - 1];
                if pass_dark.is_finite() && next_darks_rev[c - 1].is_finite() && cost_rev[next_off + c - 1].is_finite() {
                    let pass_ink = cur_inks_rev[c - 1];
                    let pass_entry = delta_ink_score(pass_dark, cur_dark, r, r, row_ink, max_ink);
                    let child_dark = next_darks_rev[c - 1];
                    let child_entry = delta_ink_score(child_dark, pass_dark, r + 1, r, row_ink, max_ink);
                    let via = cost_rev[next_off + c - 1] + cur_ink + pass_ink + pass_entry + child_entry + 0.01;
                    if via < best { best = via; best_pred = next_off + c - 1; }
                }
            }
            if c + 1 < seg_w {
                let pass_dark = cur_darks_rev[c + 1];
                if pass_dark.is_finite() && next_darks_rev[c + 1].is_finite() && cost_rev[next_off + c + 1].is_finite() {
                    let pass_ink = cur_inks_rev[c + 1];
                    let pass_entry = delta_ink_score(pass_dark, cur_dark, r, r, row_ink, max_ink);
                    let child_dark = next_darks_rev[c + 1];
                    let child_entry = delta_ink_score(child_dark, pass_dark, r + 1, r, row_ink, max_ink);
                    let via = cost_rev[next_off + c + 1] + cur_ink + pass_ink + pass_entry + child_entry + 0.01;
                    if via < best { best = via; best_pred = next_off + c + 1; }
                }
            }
            if c >= 2 {
                let p1d = cur_darks_rev[c - 1];
                let p2d = cur_darks_rev[c - 2];
                if p1d.is_finite() && p2d.is_finite() && next_darks_rev[c - 2].is_finite() && cost_rev[next_off + c - 2].is_finite() {
                    let p1ink = cur_inks_rev[c - 1];
                    let p2ink = cur_inks_rev[c - 2];
                    let p1_entry = delta_ink_score(p1d, cur_dark, r, r, row_ink, max_ink);
                    let p2_entry = delta_ink_score(p2d, p1d, r, r, row_ink, max_ink);
                    let child_dark = next_darks_rev[c - 2];
                    let child_entry = delta_ink_score(child_dark, p2d, r + 1, r, row_ink, max_ink);
                    let via = cost_rev[next_off + c - 2] + cur_ink + p1ink + p1_entry + p2ink + p2_entry + child_entry + 0.02;
                    if via < best { best = via; best_pred = next_off + c - 2; }
                }
            }
            if c + 2 < seg_w {
                let p1d = cur_darks_rev[c + 1];
                let p2d = cur_darks_rev[c + 2];
                if p1d.is_finite() && p2d.is_finite() && next_darks_rev[c + 2].is_finite() && cost_rev[next_off + c + 2].is_finite() {
                    let p1ink = cur_inks_rev[c + 1];
                    let p2ink = cur_inks_rev[c + 2];
                    let p1_entry = delta_ink_score(p1d, cur_dark, r, r, row_ink, max_ink);
                    let p2_entry = delta_ink_score(p2d, p1d, r, r, row_ink, max_ink);
                    let child_dark = next_darks_rev[c + 2];
                    let child_entry = delta_ink_score(child_dark, p2d, r + 1, r, row_ink, max_ink);
                    let via = cost_rev[next_off + c + 2] + cur_ink + p1ink + p1_entry + p2ink + p2_entry + child_entry + 0.02;
                    if via < best { best = via; best_pred = next_off + c + 2; }
                }
            }
            cost_rev[idx] = best;
            pred_rev[idx] = best_pred as u32;
            if best.is_finite() { invalid_rev[idx] = false; }
        }
        std::mem::swap(&mut next_darks_rev, &mut cur_darks_rev);
        std::mem::swap(&mut next_inks_rev, &mut cur_inks_rev);
    }

    // ---- Generate candidates same as candidate_seams final ----
    let mid_r = h as usize / 2;
    let mid_off = mid_r * seg_w;
    let mut dp_candidates: Vec<(u32, f32)> = Vec::with_capacity(seg_w.saturating_sub(2));
    for c in 1..seg_w - 1 {
        let me = masked_energy_child(mid_r, c);
        if !me.is_finite() { continue; }
        let combined = cost_fwd[mid_off + c] + cost_rev[mid_off + c] - ink_score(me, mid_r, row_ink);
        if !combined.is_finite() { continue; }
        dp_candidates.push((child_start + c as u32, combined));
    }
    let mut candidates: Vec<(u32, f32)> = Vec::new();
    let n = dp_candidates.len();
    if n > 0 {
        let mut i = 0;
        while i < n {
            let cost = dp_candidates[i].1;
            let run_start = i;
            while i < n && dp_candidates[i].1 == cost && (i == run_start || dp_candidates[i].0 == dp_candidates[i - 1].0 + 1) { i += 1; }
            let left_higher = run_start == 0 || dp_candidates[run_start - 1].1 > cost;
            let right_higher = i >= n || dp_candidates[i].1 > cost;
            if left_higher && right_higher {
                let mid_idx = (run_start + i - 1) / 2;
                candidates.push(dp_candidates[mid_idx]);
            }
        }
    }
    let dp = SeamDp { pred_fwd, pred_rev, cost_fwd, cost_rev, seg_start: child_start, seg_w, h };
    (candidates, dp)
}

fn uniform_boundaries(width: u32, n: usize) -> Vec<u32> {
    let mut b = Vec::with_capacity(n + 1);
    for i in 0..=n {
        b.push((i as f32 * width as f32 / n as f32).round() as u32);
    }
    b
}



/// Cost breakdown for a single seam, stored in audit data.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SeamCost {
    pub dp_cost: f32,
    pub seam_width_penalty: f32,
    pub segment_size_penalty: f32,
    pub horizontal_cost: f32,
    pub ink_discount: f32,
    pub total: f32,
}

/// Segmentation summary for a single word — returned from segmentation
/// for audit integration. Does not include large debug arrays
#[derive(Debug, Clone)]
pub struct SegSummary {
    pub image_w: u32,
    pub image_h: u32,
    pub n_chars_expected: u32,
    pub n_segments_produced: u32,
    pub mismatch: bool,
    pub ws_splits: Vec<u32>,
    pub seam_splits: Vec<u32>,
    pub seam_costs: HashMap<u32, SeamCost>,
}

/// Per-word segmentation data retained for lazy bigram cropping.
pub struct WordSeg {
    /// Index of the originating word in the input `words` slice (i.e. line.words).
    pub source_word_idx: usize,
    pub word_img: Arc<GrayImage>,
    pub chars: Vec<char>,
    pub boundaries: Vec<u32>,
    pub seam_paths: Arc<HashMap<u32, Vec<[u32; 2]>>>,
    pub seam_costs: Arc<HashMap<u32, SeamCost>>,
    pub crop_h: u32,
    // Segmentation summary fields (for audit integration)
    pub word_text: String,
    pub image_w: u32,
    pub image_h: u32,
    pub n_chars_expected: u32,
    pub n_segments_produced: u32,
    pub mismatch: bool,
    pub ws_splits: Vec<u32>,
    pub seam_splits: Vec<u32>,
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
        .filter(|(_, w)| w.width > 0)
        .collect();
    sorted.sort_by(|(_, a), (_, b)| b.text.chars().count().cmp(&a.text.chars().count()));

    let mut char_counts: HashMap<char, usize> = HashMap::new();
    let mut word_segs: Vec<WordSeg> = Vec::new();
    let mut lig_word_segs: Vec<WordSeg> = Vec::new();
    let mut any_ligatures = false;
    let mut words_with_ligatures: Vec<bool> = vec![false; words.len()];

    for (_word_idx, &(orig_idx, word)) in sorted.iter().enumerate() {
        let audit_all = audit_all_chars_enabled();
        let chars_in_word: Vec<char> = if audit_all {
            word.text.chars().collect()
        } else {
            word.text.chars().filter(|c| is_supported(*c)).collect()
        };
        // Include 2-letter words, only exclude single-letter (and empty) unless audit-all requested
        if !audit_all && chars_in_word.len() <= 1 {
            continue;
        }

        let need_any = if audit_all {
            true
        } else {
            chars_in_word.iter().any(|c| {
                char_counts.get(c).copied().unwrap_or(0) < 2
            })
        };
        // For 2-letter words, always keep them for geometry even if chars already seen
        if !audit_all && chars_in_word.len() > 2 && !need_any {
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
        let word_img = contrast_normalize_char(word_img);

        let (_word_w, word_h) = word_img.dimensions();
        let all_chars: Vec<char> = word.text.chars().collect();

        // Build ligature-collapsed char array: replace sequences like
        // ['f','f','l'] with ['\u{FB04}'] so "affluent" becomes
        // ['a','\u{FB04}','u','e','n','t'] with n_chars=6.
        let lig_chars = collapse_ligature_chars(&all_chars);
        let has_ligatures = lig_chars.len() < all_chars.len();

        let word_diag_dir = diag_seg_dir.map(|ddir| {
            let word_slug = crate::seg_diag::sanitize_text(&word.text);
            ddir.join(format!("word_{:03}_{}", orig_idx, word_slug))
        });

        // ── Path A: plain segmentation (OCR chars as-is) ────────────
        let (bounds_plain, seams_plain, seg_summary_plain) = if let Some(ref wdir) = word_diag_dir {
            let pdir = wdir.join("seg_plain");
            segment_characters_diag(&word_img, all_chars.len(), &pdir, &word.text)
        } else {
            segment_characters(&word_img, all_chars.len())
        };

        // Destructure summary to allow moves instead of clones
        let SegSummary {
            image_w: plain_image_w,
            image_h: plain_image_h,
            n_chars_expected: plain_n_chars_expected,
            n_segments_produced: plain_n_segments_produced,
            mismatch: plain_mismatch,
            ws_splits: plain_ws_splits,
            seam_splits: plain_seam_splits,
            seam_costs: plain_seam_costs,
        } = seg_summary_plain;

        // Update char counts (for the word-skip optimisation)
        for &c in &all_chars {
            if audit_all || is_supported(c) {
                *char_counts.entry(c).or_insert(0) += 1;
            }
        }

        if has_ligatures {
            // Both plain and lig share same GrayImage — Arc clone cheap, zero GrayImage copy
            let word_img_arc = Arc::new(word_img);
            word_segs.push(WordSeg {
                source_word_idx: orig_idx,
                word_img: word_img_arc.clone(),
                chars: all_chars,
                boundaries: bounds_plain,
                seam_paths: Arc::new(seams_plain),
                seam_costs: Arc::new(plain_seam_costs),
                crop_h: word_h,
                word_text: word.text.clone(),
                image_w: plain_image_w,
                image_h: plain_image_h,
                n_chars_expected: plain_n_chars_expected,
                n_segments_produced: plain_n_segments_produced,
                mismatch: plain_mismatch,
                ws_splits: plain_ws_splits,
                seam_splits: plain_seam_splits,
            });

            any_ligatures = true;
            if words_with_ligatures.len() > word_segs.len() - 1 {
                words_with_ligatures[word_segs.len() - 1] = true;
            } else {
                // Resize if sorted filtering made len larger than original words.len() (shouldn't happen, but be safe)
                words_with_ligatures.resize(word_segs.len(), false);
                words_with_ligatures[word_segs.len() - 1] = true;
            }
            // ── Path B: ligature segmentation (reduced n_chars) ─────
            let (bounds_lig, seams_lig, seg_summary_lig) = if let Some(ref wdir) = word_diag_dir {
                let ldir = wdir.join("seg_lig");
                segment_characters_diag(&word_img_arc, lig_chars.len(), &ldir, &word.text)
            } else {
                segment_characters(&word_img_arc, lig_chars.len())
            };

            lig_word_segs.push(WordSeg {
                source_word_idx: orig_idx,
                word_img: word_img_arc,
                chars: lig_chars,
                boundaries: bounds_lig,
                seam_paths: Arc::new(seams_lig),
                seam_costs: Arc::new(seg_summary_lig.seam_costs),
                crop_h: word_h,
                word_text: word.text.clone(),
                image_w: seg_summary_lig.image_w,
                image_h: seg_summary_lig.image_h,
                n_chars_expected: seg_summary_lig.n_chars_expected,
                n_segments_produced: seg_summary_lig.n_segments_produced,
                mismatch: seg_summary_lig.mismatch,
                ws_splits: seg_summary_lig.ws_splits,
                seam_splits: seg_summary_lig.seam_splits,
            });
        } else {
            // No ligatures: move everything, zero clones for image and segmentation data
            word_segs.push(WordSeg {
                source_word_idx: orig_idx,
                word_img: Arc::new(word_img),
                chars: all_chars,
                boundaries: bounds_plain,
                seam_paths: Arc::new(seams_plain),
                seam_costs: Arc::new(plain_seam_costs),
                crop_h: word_h,
                word_text: word.text.clone(),
                image_w: plain_image_w,
                image_h: plain_image_h,
                n_chars_expected: plain_n_chars_expected,
                n_segments_produced: plain_n_segments_produced,
                mismatch: plain_mismatch,
                ws_splits: plain_ws_splits,
                seam_splits: plain_seam_splits,
            });
        }

    }

    // For the ligature path, non-ligature words use plain segmentation
    if any_ligatures {
        for (idx, seg) in word_segs.iter().enumerate() {
            let has_lig = idx < words_with_ligatures.len() && words_with_ligatures[idx];
            if !has_lig {
                lig_word_segs.push(WordSeg {
                    source_word_idx: seg.source_word_idx,
                    word_img: seg.word_img.clone(),
                    chars: seg.chars.clone(),
                    boundaries: seg.boundaries.clone(),
                    seam_paths: seg.seam_paths.clone(),
                    seam_costs: seg.seam_costs.clone(),
                    crop_h: seg.crop_h,
                    word_text: seg.word_text.clone(),
                    image_w: seg.image_w,
                    image_h: seg.image_h,
                    n_chars_expected: seg.n_chars_expected,
                    n_segments_produced: seg.n_segments_produced,
                    mismatch: seg.mismatch,
                    ws_splits: seg.ws_splits.clone(),
                    seam_splits: seg.seam_splits.clone(),
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

/// Per-font ligature collapse: only collapse a sequence if the resulting
/// ligature unicode char is in `allowed` (font's supported lig set).
/// Greedy longest-first, same as `collapse_ligature_chars`.  Empty allowed → identity.
/// `allowed` is a bitmask: bit0=FB00,1=FB01,2=FB02,3=FB03,4=FB04.
pub fn collapse_ligature_chars_for_allowed(chars: &[char], allowed: u8) -> Vec<char> {
    if allowed == 0 {
        return chars.to_vec();
    }
    let mut out = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut matched = false;
        for &(seq, lig_char) in LIGATURE_SEQUENCES {
            let bit = match lig_char {
                '\u{FB00}' => 0, '\u{FB01}' => 1, '\u{FB02}' => 2, '\u{FB03}' => 3, '\u{FB04}' => 4, _ => continue,
            };
            if (allowed & (1u8 << bit)) == 0 { continue; }
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
pub(crate) const LIGATURE_SEQUENCES: &[(&[char], char)] = &[
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
    seam_paths: &HashMap<u32, Vec<[u32; 2]>>,
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

    // Expanded bounds to include winding seam excursions (pre-1cfca57 crop_ngram behavior),
    // but with clamped, ordered limits to avoid 5px h_err -> -690 ll.
    let x0 = if let Some(sp) = left_seam {
        sp.iter().map(|p| p[1]).min().unwrap_or(b_left).min(b_left)
    } else {
        b_left
    }.min(ww);

    let x1 = if let Some(sp) = right_seam {
        sp.iter().map(|p| p[1]).max().unwrap_or(b_right).max(b_right).saturating_add(1)
    } else {
        b_right
    }.min(ww);

    if x1 <= x0 || (x1 - x0) < 2 * (n as u32) {
        return None;
    }

    let mut crop = image::imageops::crop_imm(word_img, x0, 0, x1 - x0, crop_h).to_image();

    let crop_w = x1 - x0;
    {
        let cw_us = crop_w as usize;
        let ch_us = crop_h as usize;
        let raw = crop.as_mut();
        let stride = cw_us;
        for y in 0..ch_us {
            let base = y * stride;
            // Per-row limits – handles 1-3 px horizontal moves, darkest-adjacent ownership
            let mut left_limit = x0;
            let mut right_limit = x1;
            if let Some(sp) = left_seam {
                let cols: Vec<u32> = sp.iter().filter(|p| p[0] == y as u32).map(|p| p[1]).collect();
                if !cols.is_empty() {
                    let s_min = *cols.iter().min().unwrap();
                    let s_max = *cols.iter().max().unwrap();
                    let left_adj = if s_min > 0 {
                        word_img.get_pixel(s_min - 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let right_adj = if s_max + 1 < ww {
                        word_img.get_pixel(s_max + 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj;
                    left_limit = if assign_to_left {
                        (s_max + 1).max(x0).min(x1)
                    } else {
                        s_min.max(x0).min(x1)
                    };
                }
            }
            if let Some(sp) = right_seam {
                let cols: Vec<u32> = sp.iter().filter(|p| p[0] == y as u32).map(|p| p[1]).collect();
                if !cols.is_empty() {
                    let s_min = *cols.iter().min().unwrap();
                    let s_max = *cols.iter().max().unwrap();
                    let left_adj = if s_min > 0 {
                        word_img.get_pixel(s_min - 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let right_adj = if s_max + 1 < ww {
                        word_img.get_pixel(s_max + 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj;
                    let r = if assign_to_left {
                        s_max + 1
                    } else {
                        s_min
                    };
                    right_limit = r.max(x0).min(x1).max(left_limit);
                }
            }
            if left_limit >= x1 || right_limit <= x0 {
                raw[base..base + cw_us].fill(255);
                continue;
            }
            let l = (left_limit - x0) as usize;
            let r = (right_limit - x0) as usize;
            if l > 0 {
                raw[base..base + l.min(cw_us)].fill(255);
            }
            if r < cw_us {
                raw[base + r..base + cw_us].fill(255);
            }
        }
    }

    normalize_to_ink_bounds(&crop, NORM_H)
}

/// Crop a character and return its normalized image plus ink metrics in word coordinates.
/// Scan crop does seam handling (whitening outside winding divider); trim itself
/// finds ink bounds and returns center in caller's (word) coordinate system.
/// This is the single source of truth for both recognition crop and geometry midpoint,
/// so trim is called exactly once per character.
pub fn char_crop_and_metrics(
    word_img: &GrayImage,
    i: usize,
    boundaries: &[u32],
    seam_paths: &HashMap<u32, Vec<[u32; 2]>>,
    crop_h: u32,
) -> Option<(GrayImage, u32, u32, u32, u32, f64, f64)> {
    let (ww, _) = word_img.dimensions();
    if i + 1 >= boundaries.len() {
        return None;
    }
    let b_left = boundaries[i];
    let b_right = boundaries[i + 1];
    let left_seam = seam_paths.get(&b_left);
    let right_seam = seam_paths.get(&b_right);

    // Restore original clamped bounds: use nominal b_left/b_right, not expanded
    // seam excursions. Expanded bounds shift cx for narrow glyphs (v,w,x,l,n) and
    // Single source crop: expanded bounds to include winding seam excursions
    // (avoids clipping ffi etc), but with clamped, ordered seam limits to avoid
    // 5px h_err -> -690 ll. This merges pre-1cfca57 measure_char_ink_bounds (clamped)
    // with pre-1cfca57 crop_ngram (expanded) into one correct implementation.
    let x0 = if let Some(sp) = left_seam {
        sp.iter().map(|p| p[1]).min().unwrap_or(b_left).min(b_left)
    } else {
        b_left
    }.min(ww);
    let x1 = if let Some(sp) = right_seam {
        sp.iter().map(|p| p[1]).max().unwrap_or(b_right).max(b_right).saturating_add(1)
    } else {
        b_right
    }.min(ww);

    if x1 <= x0 || (x1 - x0) < 2 {
        return None;
    }

    let mut crop = image::imageops::crop_imm(word_img, x0, 0, x1 - x0, crop_h).to_image();
    let crop_w = x1 - x0;
    {
        let cw_us = crop_w as usize;
        let ch_us = crop_h as usize;
        let raw = crop.as_mut();
        let stride = cw_us;
        for y in 0..ch_us {
            let base = y * stride;
            // Per-row limits – handles 1-3 px horizontal moves, darkest-adjacent ownership (mirrors measure_char_ink_bounds)
            let mut left_limit = x0;
            let mut right_limit = x1;

            if let Some(sp) = left_seam {
                let cols: Vec<u32> = sp.iter().filter(|p| p[0] == y as u32).map(|p| p[1]).collect();
                if !cols.is_empty() {
                    let s_min = *cols.iter().min().unwrap();
                    let s_max = *cols.iter().max().unwrap();
                    let left_adj = if s_min > 0 {
                        word_img.get_pixel(s_min - 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let right_adj = if s_max + 1 < ww {
                        word_img.get_pixel(s_max + 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj;
                    left_limit = if assign_to_left {
                        (s_max + 1).max(x0).min(x1)
                    } else {
                        s_min.max(x0).min(x1)
                    };
                }
            }
            if let Some(sp) = right_seam {
                let cols: Vec<u32> = sp.iter().filter(|p| p[0] == y as u32).map(|p| p[1]).collect();
                if !cols.is_empty() {
                    let s_min = *cols.iter().min().unwrap();
                    let s_max = *cols.iter().max().unwrap();
                    let left_adj = if s_min > 0 {
                        word_img.get_pixel(s_min - 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let right_adj = if s_max + 1 < ww {
                        word_img.get_pixel(s_max + 1, y as u32).0[0]
                    } else {
                        255
                    };
                    let assign_to_left = left_adj <= right_adj;
                    let r = if assign_to_left { s_max + 1 } else { s_min };
                    right_limit = r.max(x0).min(x1).max(left_limit);
                }
            }
            // right_limit is at least left_limit by construction
            if left_limit >= x1 {
                // left seam at/ beyond right edge: no ink can satisfy
                raw[base..base + cw_us].fill(255);
                continue;
            }
            if right_limit <= x0 {
                raw[base..base + cw_us].fill(255);
                continue;
            }
            let l = (left_limit - x0) as usize;
            let r = (right_limit - x0) as usize;
            if l > 0 {
                raw[base..base + l.min(cw_us)].fill(255);
            }
            if r < cw_us {
                raw[base + r..base + cw_us].fill(255);
            }
        }
    }

    // Trim to ink — single scan, raw buffer, no get_pixel.
    const THRESH: u8 = 200;
    let (cw, ch) = crop.dimensions();
    if cw == 0 || ch == 0 {
        return None;
    }
    let cw_us = cw as usize;
    let ch_us = ch as usize;
    let raw_crop = crop.as_raw();
    let mut min_x = cw;
    let mut max_x = 0u32;
    let mut min_y = ch;
    let mut max_y = 0u32;
    for y in 0..ch_us {
        let base = y * cw_us;
        for x in 0..cw_us {
            if raw_crop[base + x] < THRESH {
                let xu = x as u32;
                let yu = y as u32;
                if xu < min_x { min_x = xu; }
                if xu > max_x { max_x = xu; }
                if yu < min_y { min_y = yu; }
                if yu > max_y { max_y = yu; }
            }
        }
    }
    if min_x > max_x || min_y > max_y {
        return None;
    }

    // Absolute bounds and center in word coordinates (caller's system)
    let x_min_abs = x0 + min_x;
    let x_max_abs = x0 + max_x;
    let y_min_abs = min_y;
    let y_max_abs = max_y;
    let cx = (x_min_abs as f64 + x_max_abs as f64) * 0.5;
    let cy = (y_min_abs as f64 + y_max_abs as f64) * 0.5;

    // Build normalized image from the same ink bounds (no second scan) - raw blit.
    let ink_w = max_x - min_x + 1;
    let ink_h = max_y - min_y + 1;
    let pad = 1u32;
    let canvas_w = ink_w + 2 * pad;
    let canvas_h = ink_h + 2 * pad;
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, image::Luma([255u8]));
    {
        let cw_us = cw as usize;
        let canvas_w_us = canvas_w as usize;
        let raw_crop = crop.as_raw();
        let raw_canvas = canvas.as_mut();
        for y in min_y..=max_y {
            let src_base = y as usize * cw_us + min_x as usize;
            let dst_base = (y - min_y + pad) as usize * canvas_w_us + pad as usize;
            let len = ink_w as usize;
            raw_canvas[dst_base..dst_base + len].copy_from_slice(&raw_crop[src_base..src_base + len]);
        }
    }
    let normalized = if canvas_h == NORM_H {
        // Identity: scaled_w == canvas_w when canvas_h == NORM_H
        canvas
    } else {
        let scaled_w = (canvas_w as f32 * NORM_H as f32 / canvas_h as f32).ceil() as u32;
        if scaled_w < 2 {
            return None;
        }
        image::imageops::resize(
            &canvas,
            scaled_w,
            NORM_H,
            image::imageops::FilterType::Lanczos3,
        )
    };


    Some((normalized, x_min_abs, x_max_abs, y_min_abs, y_max_abs, cx, cy))
}

