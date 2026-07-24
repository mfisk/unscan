use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bbox {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl Bbox {
    pub fn width(&self) -> f32 { (self.max_x - self.min_x).max(0.0) }
    pub fn height(&self) -> f32 { (self.max_y - self.min_y).max(0.0) }
    pub fn is_empty(&self) -> bool { self.width() <= 0.0 || self.height() <= 0.0 }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphBBox {
    pub gid: u16,
    pub bbox: Option<Bbox>,
}

/// Pure ink bounds for a single character in pixel space.
#[derive(Debug, Clone, PartialEq)]
pub struct CharInkBounds {
    pub cx: f64,
    pub cy: f64,
    pub width: f64,
    pub height: f64,
    pub x_min: u32,
    pub x_max: u32,
    pub y_min: u32,
    pub y_max: u32,
}

/// Batch result: one entry per char in a word.
#[derive(Debug, Clone)]
pub struct WordGeoMeasurement {
    pub chars: Vec<CharInkBounds>,
}

/// Batch API: compute bounding boxes for many GlyphIds at once.
/// This is the low-volume batch version - takes slice, returns Vec.
/// The inner per-glyph loop stays inline in this crate (hot path kept here).
pub fn glyph_bboxes_batch_pure<F>(gids: &[u16], mut bbox_fn: F) -> Vec<GlyphBBox>
where
    F: FnMut(u16) -> Option<Bbox>,
{
    gids.iter().map(|&gid| GlyphBBox { gid, bbox: bbox_fn(gid) }).collect()
}

/// Compute centre-span scale from observed vs predicted centres.
/// Pure math, no deps. Returns scale factor.
pub fn center_span_scale(obs_cx: &[f64], pred_cx: &[f64]) -> f64 {
    if obs_cx.len() >= 2 && pred_cx.len() >= 2 {
        let obs_span = (obs_cx.last().unwrap() - obs_cx.first().unwrap()).abs().max(0.5);
        let pred_span = (pred_cx.last().unwrap() - pred_cx.first().unwrap()).abs().max(0.5);
        obs_span / pred_span
    } else {
        1.0
    }
}

/// Batch version of centre calculation for many words.
/// Takes slices of slices, returns Vec of scales - one call, many results.
pub fn batch_center_span_scales(obs_sets: &[Vec<f64>], pred_sets: &[Vec<f64>]) -> Vec<f64> {
    obs_sets.iter().zip(pred_sets.iter())
        .map(|(obs, pred)| center_span_scale(obs, pred))
        .collect()
}
