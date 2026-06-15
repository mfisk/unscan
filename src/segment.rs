//! Character segmentation: VP whitespace splitting + dual-DP seam carving.
//!
//! Given a word image and the expected number of characters, produce N+1
//! boundaries that partition the image into N cells.

use image::GrayImage;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

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
    vert_run_discount: f32,  // weight for short horizontal dark runs in vertical scoring
    vert_run_threshold: u32, // run length cutoff for discount (<=threshold gets discount)
    vert_row_ink_power: f32, // exponent on (row_ink/max_row_ink) discount in vertical scoring (0=off)
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
            vert_run_discount: env_f32("SEAM_VERT_RUN_DISCOUNT", 0.9),
            vert_run_threshold: std::env::var("SEAM_VERT_RUN_THRESHOLD").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(3u32),
            vert_row_ink_power: env_f32("SEAM_VERT_ROW_INK_POWER", 0.0),
        };
        eprintln!("[seam params] ink_power={} ink_norm={} ink_row_wt={} ink_row_pow={} \
delta_wt={} delta_pow={} delta_scale_pow={} delta_row_wt={} delta_row_pow={} \
vert_run_disc={} vert_run_thresh={} vert_row_ink_pow={}",
            p.ink_power, p.ink_norm, p.ink_row_weight, p.ink_row_power,
            p.delta_weight, p.delta_power, p.delta_scale_power,
            p.delta_row_weight, p.delta_row_power,
            p.vert_run_discount, p.vert_run_threshold, p.vert_row_ink_power);
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
    row_cur: usize, _row_prev: usize,
    row_ink: &[f32], max_ink: f32,
) -> f32 {
    if dark_cur <= dark_prev { return 0.0; }
    let p = seam_params();
    let delta = dark_cur - dark_prev;
    let base = if p.delta_power == 1.0 { delta } else { delta.powf(p.delta_power) };
    let scale = if p.delta_scale_power == 1.0 {
        dark_cur / max_ink
    } else {
        (dark_cur / max_ink).powf(p.delta_scale_power)
    };
    let row_factor = if p.delta_row_weight == 0.0 {
        1.0
    } else {
        let ri = if p.delta_row_power == 1.0 { row_ink[row_cur] }
                 else { row_ink[row_cur].powf(p.delta_row_power) };
        1.0 + p.delta_row_weight * ri
    };
    p.delta_weight * base * scale * row_factor
}

/// Fraction of the crop height that represents the smallest symbol
/// (a period).  A period is roughly 8% of line height.  The minimum
/// ink threshold is `(MIN_SYMBOL_FRAC * h)²` dark pixels (counted,
/// not intensity-weighted), making it scale with DPI and font size
/// without penalising grey/anti-aliased text vs solid black.
const MIN_SYMBOL_FRAC: f32 = 0.07;


/// Segment a word image into N character cells.
///
/// Three-pass cascade:
///
/// **Pass 1 — Vertical Profile (VP):** find contiguous runs of zero-ink
/// columns (threshold 200).  Each interior run gives one split at its
/// midpoint.  If that yields ≥ N-1 splits, pick the N-1 widest runs.
/// Both sides of every VP split must have at least `min_ink_for_symbol`
/// total column-ink or the split is rejected.
///
/// **Pass 2 — Seam carving:** for remaining splits, find the cheapest
/// vertical seam in each existing segment via DP, pick the globally
/// cheapest, split there, and repeat.  Energy is ink-based: each pixel's
/// cost is its darkness (0 white, 255 black) plus an entry penalty
/// (delta_ink_score) when the path moves into
/// a darker pixel — directly encoding "stay in whitespace, don't wander
/// into ink."  The same `min_ink_for_symbol` threshold applies: both
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
    if word_text.map_or(false, |w| w.starts_with("abcdefg")) {
        eprintln!("  SEGINNER ENTRY: n_chars={} diag={} wtext={:?}", n_chars, diag_dir.is_some(), word_text);
    }
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
    //   Pass 1 — VP strict zero-ink: split at columns with truly zero ink
    //   Pass 2 — Seam carving: cheapest vertical path for remaining splits
    //
    // VP midpoints give geometrically centered boundaries.  Seam carving is
    // last resort for genuinely connected characters (e.g. serif bridges
    // at 10%+ ink where no column qualifies as low-ink).

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

    // --- VP splitting: iterative, ink-trimmed ---
    //
    // Find character boundaries by iteratively splitting at the widest
    // low-ink column run.  After each split, trim each resulting segment
    // to its horizontal ink extent so that the whitespace margins around
    // the split are excluded from future searches.  This prevents
    // near-duplicate splits (e.g. 173 and 174) from consuming slots
    // that should go to real inter-character gaps.

    // VP valleys: strict zero-ink only.  Relaxed thresholds cause too many
    // false VP splits through genuine thin strokes (at h=42, even 2-pixel
    // columns can be real ink).  Low-ink-but-nonzero columns are handled by
    // the seam pass below with a VP-preference bias: when a seam's nominal
    // column is near a low-ink valley, the seam is snapped to the valley's
    // vertical line instead of its diagonal path.
    let col_has_ink_strict: Vec<bool> = col_ink.iter().map(|&v| v > 0).collect();
    let col_has_ink: Vec<bool> = col_has_ink_strict.clone();

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

    let initial_ink = ink_extent(&col_has_ink_strict, 0, w);
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
                // a period (~12 fully-black pixels = 12×255 = 3060 ink).
                let ink_left_sum: u32 = (seg.left..split).map(|c| col_ink[c as usize]).sum();
                let ink_right_sum: u32 = (split + 1..seg.right).map(|c| col_ink[c as usize]).sum();
                if ink_left_sum < min_ink_for_symbol || ink_right_sum < min_ink_for_symbol {
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
        let left_ink = ink_extent(&col_has_ink_strict, old.left, mid);
        let right_ink = ink_extent(&col_has_ink_strict, mid, old.right);
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
            &wtext.chars().take(30).collect::<String>(), w, h, vp_splits.len(), need,
        );
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
            is_vertical: bool,
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
                let (cands, dp, vert_wins) = candidate_seams(&energy, ink_l, ink_r, h, None, None, max_ink, &row_ink);
                if word_text.map_or(false, |w| w.starts_with("tradition")) {
                    eprintln!("  SEED seg=[{},{}) ink=[{},{}) {} candidates sid={}", seg_start, seg_end, ink_l, ink_r, cands.len(), sid);
                    for (col, cost) in &cands {
                        eprintln!("    candidate col={} cost={:.1}", col, cost);
                    }
                }
                for (col, cost) in &cands {
                    let is_vert = vert_wins.contains(col);
                    heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid, is_vertical: is_vert });
                    seam_seed_candidates.push(serde_json::json!({
                        "col": col, "cost": *cost as f64,
                        "seg": [ink_l, ink_r], "sid": sid,
                        "is_vertical": is_vert,
                    }));
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
            if dead_sids.contains(&entry.seg_id) {
                seam_greedy_log.push(serde_json::json!({
                    "action": "skip_dead", "col": entry.col, "cost": entry.cost as f64,
                    "seg": [entry.seg_start, entry.seg_end], "sid": entry.seg_id,
                }));
                continue;
            }

            if word_text.map_or(false, |w| w.starts_with("tradition") || w.starts_with("abcdefg")) {
                eprintln!("  SEAM POP [{}]: col={} cost={:.1} seg=[{},{}) sid={} | accepted={:?}", word_text.unwrap_or("?"), entry.col, entry.cost, entry.seg_start, entry.seg_end, entry.seg_id, &splits);
            }

            // Skip if this exact split column was already accepted.
            if splits.contains(&entry.col) {
                if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    SKIP DUP col={}", entry.col); }
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
                if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    SKIP INK col={} left_ok={} right_ok={}", entry.col, left_ok, right_ok); }
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
                        let (cands, dp, vert_wins) = candidate_seams(&energy, entry.seg_start, new_end, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
                        for (col, cost) in &cands {
                            heap.push(SeamEntry { cost: *cost, col: *col, seg_start: entry.seg_start, seg_end: new_end, seg_id: sid, is_vertical: vert_wins.contains(col) });
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
                        let (cands, dp, vert_wins) = candidate_seams(&energy, new_start, entry.seg_end, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
                        for (col, cost) in &cands {
                            heap.push(SeamEntry { cost: *cost, col: *col, seg_start: new_start, seg_end: entry.seg_end, seg_id: sid, is_vertical: vert_wins.contains(col) });
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
            let path = if entry.is_vertical {
                vec![entry.col; h as usize]
            } else {
                match dp_cache.get(&entry.seg_id) {
                    Some(dp) => dp.trace_path_through(&energy, entry.col, &row_ink),
                    None => {
                        // Segment was already consumed; skip stale candidate.
                        continue;
                    }
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
                if word_text.map_or(false, |w| w.starts_with("tradition")) { eprintln!("    SKIP MIN_INK col={} left={} right={} min={}", entry.col, seam_ink_left, seam_ink_right, min_ink_for_symbol); }
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
                "is_vertical": entry.is_vertical,
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

            // Recompute child ink extents using the final (possibly VP-snapped) column.
            let child_left_ink = ink_extent(&col_has_ink_strict, entry.seg_start, final_col);
            let child_right_ink = ink_extent(&col_has_ink_strict, final_col + 1, entry.seg_end);

            // Left child: inherits parent's left boundary, seam path as right boundary.
            {
                let (ink_l, ink_r) = child_left_ink;
                if ink_r > ink_l + 2 {
                    let sid = next_seg_id; next_seg_id += 1;
                    let lp = parent_lp.clone();
                    let rp: Option<Vec<u32>> = Some(path.clone());
                    let (cands, dp, vert_wins) = candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid, is_vertical: vert_wins.contains(col) });
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
                    let (cands, dp, vert_wins) = candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink);
                    for (col, cost) in &cands {
                        heap.push(SeamEntry { cost: *cost, col: *col, seg_start: ink_l, seg_end: ink_r, seg_id: sid, is_vertical: vert_wins.contains(col) });
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
    cost_fwd: Vec<f32>,   // flat [row * seg_w + col]
    cost_rev: Vec<f32>,   // flat [row * seg_w + col]
    pred_fwd: Vec<u32>,   // flat [row * seg_w + col] — packed (r, c) predecessor
    pred_rev: Vec<u32>,   // flat [row * seg_w + col] — packed (r, c) predecessor
    seg_start: u32,
    seg_end: u32,
    seg_w: usize,
    h: u32,
    max_ink: f32,
    row_ink: Vec<f32>,    // per-row ink fraction for ink_score/delta_ink_score
}

impl SeamDp {

    /// Backtrace the cheapest path constrained to pass through
    /// `target_col` at mid-row.
    fn trace_path_through(&self, energy: &[Vec<f32>], target_col: u32, row_ink: &[f32]) -> Vec<u32> {
        let seg_w = self.seg_w;
        let base = self.seg_start as usize;
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
) -> (Vec<(u32, f32)>, SeamDp, HashSet<u32>) {
    let seg_w = (seg_end - seg_start) as usize;
    if seg_w < 3 || h < 1 {
        let dp = SeamDp { cost_fwd: Vec::new(), cost_rev: Vec::new(), pred_fwd: Vec::new(), pred_rev: Vec::new(), seg_start, seg_end, seg_w: 0, h, max_ink, row_ink: row_ink.to_vec() };
        return (Vec::new(), dp, HashSet::new());
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
    let n_cells = h as usize * seg_w;
    let mut cost_fwd = vec![0.0f32; n_cells];
    let mut pred_fwd = vec![0u32; n_cells];
    for c in 0..seg_w {
        cost_fwd[c] = ink_score(masked_energy(0, c), 0, row_ink);
        pred_fwd[c] = c as u32; // self
    }
    for r in 1..h as usize {
        let row_off = r * seg_w;
        let prev_off = (r - 1) * seg_w;
        // Step 1: vertical-only from row above (same column)
        for c in 0..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let prev_dark = masked_energy(r - 1, c);
            let entry = delta_ink_score(cur_dark, prev_dark, r, r - 1, row_ink, max_ink);
            cost_fwd[row_off + c] = cur_ink + cost_fwd[prev_off + c] + entry;
            pred_fwd[row_off + c] = (prev_off + c) as u32;
        }
        // Step 2: horizontal propagation left-to-right
        for c in 1..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let nbr_dark = masked_energy(r, c - 1);
            let entry = delta_ink_score(cur_dark, nbr_dark, r, r, row_ink, max_ink);
            let via_left = cost_fwd[row_off + c - 1] + cur_ink + entry;
            if via_left < cost_fwd[row_off + c] {
                cost_fwd[row_off + c] = via_left;
                pred_fwd[row_off + c] = (row_off + c - 1) as u32;
            }
        }
        // Step 3: horizontal propagation right-to-left
        for c in (0..seg_w - 1).rev() {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let nbr_dark = masked_energy(r, c + 1);
            let entry = delta_ink_score(cur_dark, nbr_dark, r, r, row_ink, max_ink);
            let via_right = cost_fwd[row_off + c + 1] + cur_ink + entry;
            if via_right < cost_fwd[row_off + c] {
                cost_fwd[row_off + c] = via_right;
                pred_fwd[row_off + c] = (row_off + c + 1) as u32;
            }
        }
    }

    // Reverse DP: models downward continuation from (r, c) to bottom.
    let last_r = (h - 1) as usize;
    let mut cost_rev = vec![0.0f32; n_cells];
    let mut pred_rev = vec![0u32; n_cells];
    let last_off = last_r * seg_w;
    for c in 0..seg_w {
        cost_rev[last_off + c] = ink_score(masked_energy(last_r, c), last_r, row_ink);
        pred_rev[last_off + c] = (last_off + c) as u32; // self
    }
    for r in (0..last_r).rev() {
        let row_off = r * seg_w;
        let next_off = (r + 1) * seg_w;
        // Step 1: vertical-only from row below (same column)
        for c in 0..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let child_dark = masked_energy(r + 1, c);
            let entry = delta_ink_score(child_dark, cur_dark, r + 1, r, row_ink, max_ink);
            cost_rev[row_off + c] = cur_ink + cost_rev[next_off + c] + entry;
            pred_rev[row_off + c] = (next_off + c) as u32;
        }
        // Step 2: horizontal propagation left-to-right
        for c in 1..seg_w {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let nbr_dark = masked_energy(r, c - 1);
            let entry = delta_ink_score(cur_dark, nbr_dark, r, r, row_ink, max_ink);
            let via_left = cost_rev[row_off + c - 1] + cur_ink + entry;
            if via_left < cost_rev[row_off + c] {
                cost_rev[row_off + c] = via_left;
                pred_rev[row_off + c] = (row_off + c - 1) as u32;
            }
        }
        // Step 3: horizontal propagation right-to-left
        for c in (0..seg_w - 1).rev() {
            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let nbr_dark = masked_energy(r, c + 1);
            let entry = delta_ink_score(cur_dark, nbr_dark, r, r, row_ink, max_ink);
            let via_right = cost_rev[row_off + c + 1] + cur_ink + entry;
            if via_right < cost_rev[row_off + c] {
                cost_rev[row_off + c] = via_right;
                pred_rev[row_off + c] = (row_off + c + 1) as u32;
            }
        }
    }

    // For each interior column at mid-row, the cheapest path through it
    // costs cost_fwd[mid][c] + cost_rev[mid][c] - energy[mid][c]
    // (subtract once to avoid double-counting the mid-row pixel).
    let mid_off = mid_r * seg_w;
    let mut dp_candidates: Vec<(u32, f32)> = Vec::with_capacity(seg_w.saturating_sub(2));
    for c in 1..seg_w - 1 {
        let me = masked_energy(mid_r, c);
        if me >= f32::INFINITY { continue; } // masked pixel, skip
        let combined = cost_fwd[mid_off + c] + cost_rev[mid_off + c] - ink_score(me, mid_r, row_ink);
        let split_col = seg_start + c as u32;
        dp_candidates.push((split_col, combined));
    }

    // Vertical-only candidates: score each column as a straight vertical
    // cut, discounting ink where the horizontal dark run through the
    // candidate column is short (serif bridges).  Unlike the fixed
    // top/bottom band discount, this catches serifs at any height
    // (lowercase x-height, baseline, cap-height, etc.).
    let img_w = energy[0].len();
    let mut vert_candidates: Vec<(u32, f32)> = Vec::with_capacity(seg_w.saturating_sub(2));
    let max_row_ink = row_ink.iter().copied().fold(0.0f32, f32::max).max(1e-9);
    for c in 1..seg_w - 1 {
        let abs_col = base + c;
        let mut cost = 0.0f32;
        let mut masked = false;
        let mut prev_dark = 0.0f32;
        for r in 0..h as usize {
            let e = masked_energy(r, c);
            if e >= f32::INFINITY { masked = true; break; }
            // Measure horizontal dark run length through this column.
            // Short runs (1-3 px) are likely serif bridges → heavy discount.
            let mut run_len = 1u32;
            // Extend left
            {
                let mut cx = abs_col.wrapping_sub(1);
                while cx < img_w && energy[r][cx] > 0.0 {
                    run_len += 1;
                    if cx == 0 { break; }
                    cx = cx.wrapping_sub(1);
                }
            }
            // Extend right
            {
                let mut cx = abs_col + 1;
                while cx < img_w && energy[r][cx] > 0.0 {
                    run_len += 1;
                    cx += 1;
                }
            }
            let p = seam_params();
            let run_weight = if run_len <= p.vert_run_threshold {
                p.vert_run_discount
            } else {
                1.0
            };
            let row_weight = if p.vert_row_ink_power == 0.0 {
                1.0
            } else {
                (row_ink[r] / max_row_ink).powf(p.vert_row_ink_power)
            };
            let weight = run_weight * row_weight;
            // Same scoring as DP: ink_score + delta_ink_score
            let ink = ink_score(e, r, row_ink);
            let delta = if r == 0 { 0.0 } else {
                delta_ink_score(e, prev_dark, r, r - 1, row_ink, max_ink)
            };
            cost += (ink + delta) * weight;
            prev_dark = e;
        }
        if !masked {
            vert_candidates.push((seg_start + c as u32, cost));
        }
    }

    // Deduplicate: for each column keep only the cheapest candidate
    // (DP path or vertical cut, whichever wins).  Track which columns
    // were won by the vertical-only candidate.
    let mut vertical_winners: HashSet<u32> = HashSet::new();
    let mut raw_candidates: Vec<(u32, f32)>;
    {
        // Start with DP candidates as baseline
        let mut best: HashMap<u32, (f32, bool)> = HashMap::new(); // col -> (cost, is_vertical)
        for &(col, cost) in &dp_candidates {
            best.insert(col, (cost, false));
        }
        // Vertical candidates override if cheaper
        for &(col, cost) in &vert_candidates {
            let entry = best.entry(col).or_insert((f32::INFINITY, false));
            if cost < entry.0 {
                *entry = (cost, true);
            }
        }
        for (&col, &(_cost, is_vert)) in &best {
            if is_vert { vertical_winners.insert(col); }
        }
        raw_candidates = best.into_iter().map(|(col, (cost, _))| (col, cost)).collect();
        raw_candidates.sort_by(|a, b| a.0.cmp(&b.0));
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
        let mid_col = raw_candidates[mid_idx].0;
        // If any column in this run was a vertical winner, the midpoint inherits that.
        let any_vert = (run_start..i).any(|j| vertical_winners.contains(&raw_candidates[j].0));
        if any_vert { vertical_winners.insert(mid_col); }
        candidates.push(raw_candidates[mid_idx]);
    }

    let dp = SeamDp { cost_fwd, cost_rev, pred_fwd, pred_rev, seg_start, seg_end, seg_w, h, max_ink, row_ink: row_ink.to_vec() };
    (candidates, dp, vertical_winners)
}

/// Uniform character boundaries.
fn uniform_boundaries(width: u32, n: usize) -> Vec<u32> {
    let mut b = Vec::with_capacity(n + 1);
    for i in 0..=n {
        b.push((i as f32 * width as f32 / n as f32).round() as u32);
    }
    b
}

