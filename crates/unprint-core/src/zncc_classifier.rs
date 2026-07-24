// ---------------------------------------------------------------------------
// ZnccClassifier — pixel-level raster comparison via ZNCC
// ---------------------------------------------------------------------------
//
// No training step.  At classification time, iterates unique glyph
// images from the GlyphMap and computes pairwise ZNCC against the
// query crop.  Since images are hash-addressed, each unique render is
// compared exactly once regardless of how many fonts share it.
//
// ZNCC scores (in [−1, 1]) are converted to probabilities via softmax
// with a temperature parameter, so the output conforms to the Classifier
// trait's probability interface.
//
// Returns glyph_ids (indices into GlyphMap groups for the given char),
// not font_ids.

use crate::features::CropFeatures;
use crate::classifier::Classifier;
use crate::compare_rasters::zncc_global_pub;
use crate::glyph_map::NgramGlyphMap;
use crate::char_render::RenderParams;

/// Temperature for ZNCC→probability conversion.
const ZNCC_TEMPERATURE: f32 = 10.0;

pub struct ZnccClassifier {
    /// Shared glyph equivalence map (glyph_id → font_keys per char).
    glyph_map: NgramGlyphMap,
    /// Render parameters for loading cached glyphs.
    render_params: RenderParams,
}

impl ZnccClassifier {
    /// Build from a pre-built GlyphMap.  No training — reference rasters
    /// are loaded from the hash-addressed cache at classification time.
    pub fn from_glyph_map(
        glyph_map: NgramGlyphMap,
        render_params: &RenderParams,
    ) -> Self {
        let total = glyph_map.groups.values().map(|g| g.len()).sum::<usize>();
        let n_chars = glyph_map.groups.len();
        eprintln!("ZNCC: {total} unique glyphs across {n_chars} chars (lazy render)");
        Self {
            glyph_map,
            render_params: render_params.clone(),
        }
    }
}

impl Classifier for ZnccClassifier {
    fn classify(&self, seq: &[char], query: &CropFeatures, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self.probabilities(seq, query);
        probs.truncate(k);
        probs
    }

    fn probabilities(&self, seq: &[char], query: &CropFeatures) -> Vec<(usize, f32)> {
        let query_img = match query.raster.as_ref() {
            Some(img) => img,
            None => return Vec::new(),
        };

        let groups = match self.glyph_map.groups.get(seq) {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (glyph_id, group) in groups.iter().enumerate() {
            let ref_img = if group.hash != 0 {
                crate::char_render::load_cached_ngram(seq, group.hash, &self.render_params)
            } else {
                None
            };

            let ref_img = match ref_img {
                Some(img) => img,
                None => continue,
            };

            let z = zncc_global_pub(query_img, &ref_img);
            scored.push((glyph_id, z));
        }

        if scored.is_empty() {
            return Vec::new();
        }

        // Softmax: convert ZNCC scores to probabilities
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

    fn probability(&self, seq: &[char], query: &CropFeatures, glyph_id: usize) -> Option<f32> {
        self.probabilities(seq, query).iter()
            .find(|(id, _)| *id == glyph_id)
            .map(|(_, p)| *p)
    }

    fn name(&self) -> &str {
        "zncc"
    }

    fn glyph_count(&self, seq: &[char]) -> usize {
        self.glyph_map.glyph_count(seq)
    }

    fn add_glyph(&mut self, _glyph_id: usize, _seq: &[char], _features: &CropFeatures) {
        // No-op: ZNCC works from glyph_map + cached renders.
    }
}

