// ---------------------------------------------------------------------------
// ZnccClassifier — pixel-level raster comparison via ZNCC
// ---------------------------------------------------------------------------
//
// Bypasses the feature vector entirely.  Stores the normalised reference
// glyph raster for each (char, font) pair and compares query crops
// directly using zero-mean normalised cross-correlation.
//
// ZNCC scores (in [−1, 1]) are converted to probabilities via softmax
// with a temperature parameter, so the output conforms to the Classifier
// trait's probability interface.

use std::collections::HashMap;
use image::GrayImage;
use crate::char_index::CharFeatures;
use crate::classifier::Classifier;
use crate::ssim::zncc_global_pub;

/// Temperature for ZNCC→probability conversion.
/// Higher = flatter distribution; lower = more peaked.
/// Tuned so that a ZNCC gap of ~0.05 produces a meaningful probability gap.
const ZNCC_TEMPERATURE: f32 = 10.0;

pub struct ZnccClassifier {
    /// Per-char reference rasters: char → [(font_id, raster)]
    refs: HashMap<char, Vec<(usize, GrayImage)>>,
    /// Number of distinct font IDs seen.
    n_fonts: usize,
}

impl ZnccClassifier {
    pub fn new() -> Self {
        Self {
            refs: HashMap::new(),
            n_fonts: 0,
        }
    }
}

impl Classifier for ZnccClassifier {
    fn classify(&self, ch: char, query: &CharFeatures, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self.probabilities(ch, query);
        probs.truncate(k);
        probs
    }

    fn probabilities(&self, ch: char, query: &CharFeatures) -> Vec<(usize, f32)> {
        let query_img = match query.raster.as_ref() {
            Some(img) => img,
            None => return Vec::new(),
        };
        let entries = match self.refs.get(&ch) {
            Some(e) => e,
            None => return Vec::new(),
        };
        if entries.is_empty() {
            return Vec::new();
        }

        // Compute ZNCC for each reference
        let mut scored: Vec<(usize, f32)> = entries.iter()
            .map(|(font_id, ref_img)| {
                let z = zncc_global_pub(query_img, ref_img);
                (*font_id, z)
            })
            .collect();

        // Softmax: convert ZNCC scores to probabilities
        // Numerically stable: subtract max before exp
        let max_z = scored.iter().map(|(_, z)| *z)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (_, z) in &mut scored {
            let e = ((*z - max_z) * ZNCC_TEMPERATURE).exp();
            *z = e;
            sum += e;
        }
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for (_, p) in &mut scored {
            *p *= inv_sum;
        }

        // Sort descending by probability
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn probability(&self, ch: char, query: &CharFeatures, font_id: usize) -> Option<f32> {
        // For a single font lookup, compute full probabilities and find the one.
        // Could be optimised to skip sort, but correctness first.
        self.probabilities(ch, query).iter()
            .find(|(id, _)| *id == font_id)
            .map(|(_, p)| *p)
    }

    fn name(&self) -> &str {
        "zncc"
    }

    fn font_count(&self) -> usize {
        self.n_fonts
    }

    fn needs_raster(&self) -> bool { true }

    fn add_font(&mut self, font_id: usize, ch: char, features: &CharFeatures) {
        if let Some(ref img) = features.raster {
            self.refs.entry(ch).or_default().push((font_id, img.clone()));
            if font_id >= self.n_fonts {
                self.n_fonts = font_id + 1;
            }
        }
    }
}
