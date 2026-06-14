#!/usr/bin/env python3
"""Patch segment.rs: parameterize ink scoring with ink_score/delta_ink_score + row_ink."""

import re

SRC = "src/segment.rs"

with open(SRC) as f:
    code = f.read()

# ─── 1. Add OnceLock import ───
code = code.replace(
    "use std::collections::{HashMap, HashSet};",
    "use std::collections::{HashMap, HashSet};\nuse std::sync::OnceLock;",
)

# ─── 2. Replace ENTRY_PENALTY_WEIGHT constant with SeamParams + functions ───
old_const = """/// Entry penalty weight for seam carving.  When the seam path moves into
/// a darker pixel than the previous one, the darkness increase is
/// multiplied by this weight and added as extra cost.  This penalizes
/// seams that drift from whitespace into glyph strokes.
const ENTRY_PENALTY_WEIGHT: f32 = 4.0;"""

new_params = """/// Seam carving scoring parameters, configurable via environment variables
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
        eprintln!("[seam params] ink_power={} ink_norm={} ink_row_wt={} ink_row_pow={} \\
delta_wt={} delta_pow={} delta_scale_pow={} delta_row_wt={} delta_row_pow={}",
            p.ink_power, p.ink_norm, p.ink_row_weight, p.ink_row_power,
            p.delta_weight, p.delta_power, p.delta_scale_power,
            p.delta_row_weight, p.delta_row_power);
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
}"""

assert old_const in code, "Could not find ENTRY_PENALTY_WEIGHT constant block"
code = code.replace(old_const, new_params)

# ─── 3. Darkness: revert squared back to raw + add row_ink ───
old_dark = """        // Per-pixel darkness, squared: heavier ink pays quadratically more,
        // so seams that cross even a few dark pixels are strongly penalised
        // relative to seams through lighter anti-aliased fringes.
        let darkness: Vec<Vec<f32>> = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let d = 255.0 - img.get_pixel(x, y).0[0] as f32;
                        d * d / 255.0
                    })
                    .collect()
            })
            .collect();"""

new_dark = """        // Per-pixel darkness: 0.0 for white, 255.0 for black (raw).
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
            .collect();"""

assert old_dark in code, "Could not find squared darkness block"
code = code.replace(old_dark, new_dark)

# ─── 4. Fix max_ink comment ───
code = code.replace(
    """        // Word-level max ink (p95 of squared darkness): scale the entry
        // penalty so that anti-aliased grey pixels pay proportionally
        // less than full strokes.""",
    """        // Word-level max ink (p95 of raw darkness): used by delta_ink_score
        // to scale the entry penalty proportionally.""",
)

# ─── 5. candidate_seams calls: add &row_ink ───
# Pattern: candidate_seams(&energy, ..., max_ink)  →  candidate_seams(&energy, ..., max_ink, &row_ink)
code = code.replace(
    "candidate_seams(&energy, ink_l, ink_r, h, None, None, max_ink)",
    "candidate_seams(&energy, ink_l, ink_r, h, None, None, max_ink, &row_ink)",
)
code = code.replace(
    "candidate_seams(&energy, entry.seg_start, new_end, h, lp.as_deref(), rp.as_deref(), max_ink)",
    "candidate_seams(&energy, entry.seg_start, new_end, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink)",
)
code = code.replace(
    "candidate_seams(&energy, new_start, entry.seg_end, h, lp.as_deref(), rp.as_deref(), max_ink)",
    "candidate_seams(&energy, new_start, entry.seg_end, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink)",
)
code = code.replace(
    "candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref(), max_ink)",
    "candidate_seams(&energy, ink_l, ink_r, h, lp.as_deref(), rp.as_deref(), max_ink, &row_ink)",
)

# ─── 6. trace_path_through calls: add &row_ink ───
code = code.replace(
    "dp.trace_path_through(&energy, entry.col)",
    "dp.trace_path_through(&energy, entry.col, &row_ink)",
)

# ─── 7. SeamDp struct: add row_ink field (as owned Vec) ───
# We'll store a clone of row_ink in SeamDp so the borrow checker is happy.
code = code.replace(
    """struct SeamDp {
    cost_fwd: Vec<f32>,   // flat [row * seg_w + col]
    cost_rev: Vec<f32>,   // flat [row * seg_w + col]
    seg_start: u32,
    seg_end: u32,
    seg_w: usize,
    h: u32,
    max_ink: f32,
}""",
    """struct SeamDp {
    cost_fwd: Vec<f32>,   // flat [row * seg_w + col]
    cost_rev: Vec<f32>,   // flat [row * seg_w + col]
    seg_start: u32,
    seg_end: u32,
    seg_w: usize,
    h: u32,
    max_ink: f32,
    row_ink: Vec<f32>,    // per-row ink fraction for ink_score/delta_ink_score
}""",
)

# ─── 8. trace_path_through signature + body ───
code = code.replace(
    "fn trace_path_through(&self, energy: &[Vec<f32>], target_col: u32) -> Vec<u32> {",
    "fn trace_path_through(&self, energy: &[Vec<f32>], target_col: u32, row_ink: &[f32]) -> Vec<u32> {",
)

# Top-half backtrace entry penalty
code = code.replace(
    """                        let entry = if cur_dark > prev_dark {
                            let scaled_weight = ENTRY_PENALTY_WEIGHT * (cur_dark / self.max_ink);
                            (cur_dark - prev_dark) * scaled_weight
                        } else {
                            0.0
                        };
                        let cand = self.cost_fwd[(r - 1) * seg_w + pc] + entry;""",
    """                        let entry = delta_ink_score(cur_dark, prev_dark, r, r - 1, row_ink, self.max_ink);
                        let cand = self.cost_fwd[(r - 1) * seg_w + pc] + entry;""",
)

# Bottom-half backtrace entry penalty
code = code.replace(
    """                        let entry = if child_dark > cur_dark {
                            let scaled_weight = ENTRY_PENALTY_WEIGHT * (child_dark / self.max_ink);
                            (child_dark - cur_dark) * scaled_weight
                        } else {
                            0.0
                        };
                        let cand = self.cost_rev[(r + 1) * seg_w + pc] + entry;""",
    """                        let entry = delta_ink_score(child_dark, cur_dark, r + 1, r, row_ink, self.max_ink);
                        let cand = self.cost_rev[(r + 1) * seg_w + pc] + entry;""",
)

# ─── 9. candidate_seams signature ───
code = code.replace(
    """fn candidate_seams(
    energy: &[Vec<f32>],
    seg_start: u32,
    seg_end: u32,
    h: u32,
    left_path: Option<&[u32]>,   // pixels with col <= left_path[r] are masked
    right_path: Option<&[u32]>,  // pixels with col >= right_path[r] are masked
    max_ink: f32,                // p95 ink darkness — scales entry penalty
) -> (Vec<(u32, f32)>, SeamDp, HashSet<u32>) {""",
    """fn candidate_seams(
    energy: &[Vec<f32>],
    seg_start: u32,
    seg_end: u32,
    h: u32,
    left_path: Option<&[u32]>,   // pixels with col <= left_path[r] are masked
    right_path: Option<&[u32]>,  // pixels with col >= right_path[r] are masked
    max_ink: f32,                // p95 ink darkness — scales entry penalty
    row_ink: &[f32],             // per-row ink fractions for scoring
) -> (Vec<(u32, f32)>, SeamDp, HashSet<u32>) {""",
)

# ─── 10. Early return SeamDp: add row_ink ───
code = code.replace(
    "let dp = SeamDp { cost_fwd: Vec::new(), cost_rev: Vec::new(), seg_start, seg_end, seg_w: 0, h, max_ink };",
    "let dp = SeamDp { cost_fwd: Vec::new(), cost_rev: Vec::new(), seg_start, seg_end, seg_w: 0, h, max_ink, row_ink: row_ink.to_vec() };",
)

# ─── 11. Forward DP: row 0 init ───
code = code.replace(
    """    for c in 0..seg_w {
        cost_fwd[c] = masked_energy(0, c);
    }""",
    """    for c in 0..seg_w {
        cost_fwd[c] = ink_score(masked_energy(0, c), 0, row_ink);
    }""",
)

# ─── 12. Forward DP: entry penalty + accumulation ───
old_fwd = """            let cur_dark = masked_energy(r, c);
            let mut best = f32::INFINITY;
            for &pc in &[c, c.wrapping_sub(1), c + 1] {
                if pc < seg_w {
                    let prev_dark = masked_energy(r - 1, pc);
                    let entry = if cur_dark > prev_dark {
                        let scaled_weight = ENTRY_PENALTY_WEIGHT * (cur_dark / max_ink);
                        (cur_dark - prev_dark) * scaled_weight
                    } else {
                        0.0
                    };
                    let candidate = cost_fwd[prev_off + pc] + entry;
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            cost_fwd[row_off + c] = cur_dark + best;"""

new_fwd = """            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let mut best = f32::INFINITY;
            for &pc in &[c, c.wrapping_sub(1), c + 1] {
                if pc < seg_w {
                    let prev_dark = masked_energy(r - 1, pc);
                    let entry = delta_ink_score(cur_dark, prev_dark, r, r - 1, row_ink, max_ink);
                    let candidate = cost_fwd[prev_off + pc] + entry;
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            cost_fwd[row_off + c] = cur_ink + best;"""

assert old_fwd in code, "Could not find forward DP block"
code = code.replace(old_fwd, new_fwd)

# ─── 13. Reverse DP: last row init ───
code = code.replace(
    """    for c in 0..seg_w {
        cost_rev[last_off + c] = masked_energy(last_r, c);
    }""",
    """    for c in 0..seg_w {
        cost_rev[last_off + c] = ink_score(masked_energy(last_r, c), last_r, row_ink);
    }""",
)

# ─── 14. Reverse DP: entry penalty + accumulation ───
old_rev = """            let cur_dark = masked_energy(r, c);
            let mut best = f32::INFINITY;
            for &pc in &[c, c.wrapping_sub(1), c + 1] {
                if pc < seg_w {
                    let child_dark = masked_energy(r + 1, pc);
                    let entry = if child_dark > cur_dark {
                        let scaled_weight = ENTRY_PENALTY_WEIGHT * (child_dark / max_ink);
                        (child_dark - cur_dark) * scaled_weight
                    } else {
                        0.0
                    };
                    let candidate = cost_rev[next_off + pc] + entry;
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            cost_rev[row_off + c] = cur_dark + best;"""

new_rev = """            let cur_dark = masked_energy(r, c);
            let cur_ink = ink_score(cur_dark, r, row_ink);
            let mut best = f32::INFINITY;
            for &pc in &[c, c.wrapping_sub(1), c + 1] {
                if pc < seg_w {
                    let child_dark = masked_energy(r + 1, pc);
                    let entry = delta_ink_score(child_dark, cur_dark, r + 1, r, row_ink, max_ink);
                    let candidate = cost_rev[next_off + pc] + entry;
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            cost_rev[row_off + c] = cur_ink + best;"""

assert old_rev in code, "Could not find reverse DP block"
code = code.replace(old_rev, new_rev)

# ─── 15. Mid-row candidate cost: subtract ink_score instead of raw energy ───
code = code.replace(
    "let combined = cost_fwd[mid_off + c] + cost_rev[mid_off + c] - me;",
    "let combined = cost_fwd[mid_off + c] + cost_rev[mid_off + c] - ink_score(me, mid_r, row_ink);",
)

# ─── 16. Vertical scoring: use ink_score ───
code = code.replace(
    "            cost += e * weight;",
    "            cost += ink_score(e, r, row_ink) * weight;",
)

# ─── 17. Final SeamDp construction: add row_ink ───
code = code.replace(
    "    let dp = SeamDp { cost_fwd, cost_rev, seg_start, seg_end, seg_w, h, max_ink };",
    "    let dp = SeamDp { cost_fwd, cost_rev, seg_start, seg_end, seg_w, h, max_ink, row_ink: row_ink.to_vec() };",
)

# ─── 18. Remove stale comment referencing ENTRY_PENALTY_WEIGHT ───
code = code.replace(
    "        //   penalty = ENTRY_PENALTY_WEIGHT * max(0, darkness[r] - darkness[r-1])",
    "        //   penalty = delta_ink_score(darkness[r], darkness[r-1], r, r-1, row_ink, max_ink)",
)
code = code.replace(
    "/// (`ENTRY_PENALTY_WEIGHT × darkness_increase`) when the path moves into",
    "/// (delta_ink_score) when the path moves into",
)

# ─── Write ───
with open(SRC, "w") as f:
    f.write(code)

print("Patch applied successfully.")
print(f"File size: {len(code)} chars, {code.count(chr(10))} lines")

# Sanity checks
assert "ENTRY_PENALTY_WEIGHT" not in code, "ENTRY_PENALTY_WEIGHT still present!"
assert "ink_score(" in code, "ink_score function missing"
assert "delta_ink_score(" in code, "delta_ink_score function missing"
assert "row_ink" in code, "row_ink missing"
assert "SeamParams" in code, "SeamParams missing"
assert "OnceLock" in code, "OnceLock missing"
print("All sanity checks passed.")
