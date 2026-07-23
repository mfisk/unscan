use crate::types::{ShapedGlyph, ShapedWord};

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

pub fn shape_words(
    face_handle: &FaceHandle,
    words: &[&str],
    features: &[rustybuzz::Feature],
) -> Vec<Option<ShapedWord>> {
    let face = match rustybuzz::Face::from_slice(&face_handle.data, 0) {
        Some(f) => f,
        None => return vec![None; words.len()],
    };
    words.iter().map(|text| {
        shape_word_inner(&face, features, text)
    }).collect()
}

pub fn shape_word(
    face_handle: &FaceHandle,
    features: &[rustybuzz::Feature],
    text: &str,
) -> Option<ShapedWord> {
    let face = rustybuzz::Face::from_slice(&face_handle.data, 0)?;
    shape_word_inner(&face, features, text)
}

fn is_lig_word(s: &str) -> bool {
    s.contains("ff") || s.contains("fi") || s.contains("fl") || s.contains('\u{FB00}') || s.contains('\u{FB01}')
}

fn shape_word_inner(
    face: &rustybuzz::Face,
    features: &[rustybuzz::Feature],
    text: &str,
) -> Option<ShapedWord> {
    let features_for_shape: Vec<rustybuzz::Feature> = if is_lig_word(text) {
        let mut v = features.to_vec();
        v.push(rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(b"liga"), 0, ..));
        v.push(rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(b"dlig"), 0, ..));
        v
    } else {
        features.to_vec()
    };

    let mut buffer = rustybuzz::UnicodeBuffer::new();
    buffer.push_str(text);
    let glyph_buffer = rustybuzz::shape(face, &features_for_shape, buffer);
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
    Some(ShapedWord { text: text.to_string(), glyphs, total_advance: total })
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
    if positions.is_empty() { return None; }
    let total_adv: i32 = positions.iter().map(|p| p.x_advance).sum();
    Some(total_adv as f32 / 64.0 / sample_text.chars().count() as f32)
}
