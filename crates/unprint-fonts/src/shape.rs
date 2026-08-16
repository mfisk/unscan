use crate::types::{ShapedGlyph, ShapedWord};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone)]
pub struct FaceHandle {
    data: Vec<u8>,
}

impl FaceHandle {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        rustybuzz::Face::from_slice(&bytes, 0).ok_or_else(|| "invalid font for shaping".to_string())?;
        Ok(Self { data: bytes })
    }
    pub fn from_slice(bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes(bytes.to_vec())
    }
}

pub type Features = Vec<rustybuzz::Feature>;

pub fn ot_features_for_variant(variant_tag: &str) -> Features {
    if variant_tag.is_empty() {
        return Vec::new();
    }
    let mut tag_bytes = [0u8; 4];
    let src = variant_tag.as_bytes();
    for i in 0..4.min(src.len()) {
        tag_bytes[i] = src[i];
    }
    let tag = rustybuzz::ttf_parser::Tag::from_bytes(&tag_bytes);
    vec![rustybuzz::Feature::new(tag, 1, ..)]
}

// Thread-local cache of ShapePlan by (font_data_ptr, combined_hash)
// Nightly profiling 2026-08-16: find_language_feature 377 samples 8.3% leaf,
// hb_ot_shape_plan_t::new 2.2% inclusive, collapsed_lig_set inclusive 33.9%
// calling rustybuzz::shape 5x per font with no cache, plus FS read per call.
// Cache reuses Arc<ShapePlan> across thousands of words per font.
type PlanKey = (usize, u64);

thread_local! {
    static SHAPE_PLAN_CACHE: RefCell<HashMap<PlanKey, Arc<rustybuzz::ShapePlan>>> =
        RefCell::new(HashMap::new());
}

#[inline]
fn plan_key_hash(
    dir: rustybuzz::Direction,
    script: rustybuzz::Script,
    lang: &Option<rustybuzz::Language>,
    features: &[rustybuzz::Feature],
) -> u64 {
    let mut h = DefaultHasher::new();
    dir.hash(&mut h);
    script.hash(&mut h);
    lang.hash(&mut h);
    for f in features {
        f.tag.hash(&mut h);
        f.value.hash(&mut h);
        f.start.hash(&mut h);
        f.end.hash(&mut h);
    }
    h.finish()
}

#[inline]
fn get_plan_cached(
    face: &rustybuzz::Face,
    data_ptr: usize,
    dir: rustybuzz::Direction,
    script_opt: Option<rustybuzz::Script>,
    script_for_hash: rustybuzz::Script,
    lang_opt: Option<&rustybuzz::Language>,
    lang_owned_for_hash: &Option<rustybuzz::Language>,
    features: &[rustybuzz::Feature],
) -> Arc<rustybuzz::ShapePlan> {
    let hash = plan_key_hash(dir, script_for_hash, lang_owned_for_hash, features);
    let key = (data_ptr, hash);
    if let Some(hit) = SHAPE_PLAN_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit;
    }
    let plan = Arc::new(rustybuzz::ShapePlan::new(
        face,
        dir,
        script_opt,
        lang_opt,
        features,
    ));
    SHAPE_PLAN_CACHE.with(|c| {
        c.borrow_mut().insert(key, plan.clone());
    });
    plan
}

pub fn shape_words(
    face_handle: &FaceHandle,
    words: &[&str],
    features: &[rustybuzz::Feature],
) -> Vec<Option<ShapedWord>> {
    let face = match rustybuzz::Face::from_slice(&face_handle.data, 0) {
        Some(f) => f,
        None => return vec![None; words.len()],
    };
    let data_ptr = face_handle.data.as_ptr() as usize;
    words
        .iter()
        .map(|text| shape_word_inner(&face, data_ptr, features, text))
        .collect()
}

pub fn shape_word(
    face_handle: &FaceHandle,
    features: &[rustybuzz::Feature],
    text: &str,
) -> Option<ShapedWord> {
    let face = rustybuzz::Face::from_slice(&face_handle.data, 0)?;
    let data_ptr = face_handle.data.as_ptr() as usize;
    shape_word_inner(&face, data_ptr, features, text)
}

fn is_lig_word(s: &str) -> bool {
    s.contains("ff")
        || s.contains("fi")
        || s.contains("fl")
        || s.contains('\u{FB00}')
        || s.contains('\u{FB01}')
}

fn shape_word_inner(
    face: &rustybuzz::Face,
    data_ptr: usize,
    features: &[rustybuzz::Feature],
    text: &str,
) -> Option<ShapedWord> {
    // Ligature disable only for ff/fi/fl words – 99% path stays zero-alloc
    let lig = is_lig_word(text);

    // Create buffer and guess properties first – mirrors rustybuzz::shape logic
    // which does guess before ShapePlan::new. This gives us dir/script/lang
    // for cache key and ensures plan matches buffer (debug_assert in shape_with_plan).
    let mut buffer = rustybuzz::UnicodeBuffer::new();
    buffer.push_str(text);
    buffer.guess_segment_properties();

    let dir = buffer.direction();
    let script = buffer.script(); // UNKNOWN if None – compare via unwrap_or UNKNOWN in plan assert
    let lang_owned = buffer.language(); // Option<Language> clone

    let script_opt = if script == rustybuzz::script::UNKNOWN {
        None
    } else {
        Some(script)
    };

    // Final feature set reused for plan key + plan creation; only alloc when lig word
    // to avoid features.to_vec() per call (previous hot path alloc).
    let final_features_owned: Option<Vec<rustybuzz::Feature>>;
    let final_features: &[rustybuzz::Feature] = if lig {
        let mut v = Vec::with_capacity(features.len() + 2);
        v.extend_from_slice(features);
        v.push(rustybuzz::Feature::new(
            rustybuzz::ttf_parser::Tag::from_bytes(b"liga"),
            0,
            ..,
        ));
        v.push(rustybuzz::Feature::new(
            rustybuzz::ttf_parser::Tag::from_bytes(b"dlig"),
            0,
            ..,
        ));
        final_features_owned = Some(v);
        final_features_owned.as_ref().unwrap().as_slice()
    } else {
        final_features_owned = None;
        features
    };

    let plan = get_plan_cached(
        face,
        data_ptr,
        dir,
        script_opt,
        script,
        lang_owned.as_ref(),
        &lang_owned,
        final_features,
    );

    let glyph_buffer = rustybuzz::shape_with_plan(face, &plan, buffer);
    let infos = glyph_buffer.glyph_infos();
    let positions = glyph_buffer.glyph_positions();

    let mut glyphs = Vec::with_capacity(infos.len());
    let mut total: f32 = 0.0;
    for (info, pos) in infos.iter().zip(positions.iter()) {
        let gid = info.glyph_id as u16;
        let adv = pos.x_advance as f32 / 64.0;
        total += adv;
        glyphs.push(ShapedGlyph {
            gid,
            cluster: info.cluster,
            x_advance: adv,
            y_advance: pos.y_advance as f32 / 64.0,
            x_offset: pos.x_offset as f32 / 64.0,
            y_offset: pos.y_offset as f32 / 64.0,
        });
    }
    // keep final_features_owned alive through shape (plan already built)
    drop(final_features_owned);
    Some(ShapedWord {
        text: text.to_string(),
        glyphs,
        total_advance: total,
    })
}

pub fn compute_em_px_batch(
    font_data: &[u8],
    sample_text: &str,
    features: &[rustybuzz::Feature],
) -> Option<f32> {
    let face = rustybuzz::Face::from_slice(font_data, 0)?;
    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(sample_text);
    let out = rustybuzz::shape(&face, features, buf);
    let positions = out.glyph_positions();
    if positions.is_empty() {
        return None;
    }
    let total_adv: i32 = positions.iter().map(|p| p.x_advance).sum();
    Some(total_adv as f32 / 64.0 / sample_text.chars().count() as f32)
}
