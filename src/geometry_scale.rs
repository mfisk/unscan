//! Shared center-span scaling helpers.
//!
//! Geometry scoring and font-size estimation both need the same unbiased
//! scale: `obs_span / pred_span` where span = last center - first center.
//! This module is the single source of truth for that calculation and for
//! converting scale -> em.

/// Minimum span (px or font units) to be considered valid.
/// Below this we treat the word as too short for a reliable span.
pub const MIN_SPAN: f64 = 0.5;

/// Compute scale from first/last centers.
///
/// Returns `None` if either span is < MIN_SPAN.
/// Caller decides fallback (e.g. height ratio for single-char words).
#[inline]
pub fn center_span_scale(obs_first: f64, obs_last: f64, pred_first: f64, pred_last: f64) -> Option<f64> {
    let obs_span = (obs_last - obs_first).abs();
    let pred_span = (pred_last - pred_first).abs();
    if obs_span < MIN_SPAN || pred_span < MIN_SPAN {
        None
    } else {
        Some(obs_span / pred_span)
    }
}

/// Convert scale (px per font unit) to em_px using upem.
#[inline]
pub fn em_from_scale(scale: f64, upem: f64) -> f32 {
    (scale * upem) as f32
}

/// Median of a small slice without full sort.
/// Uses `select_nth_unstable` which is O(n) and allocation-free.
pub fn median_f32(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    // `select_nth_unstable_by` requires Ord; use partial_cmp wrapper
    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(values[mid])
}

/// Filter for sizing anchors: at least one alphabetic, length >=2, spans valid.
/// Pure numbers/punctuation are excluded here, matching the sizing heuristic.
#[inline]
pub fn is_sizing_anchor(chars: &[char]) -> bool {
    if chars.len() < 2 {
        return false;
    }
    chars.iter().any(|c| c.is_alphabetic())
}
