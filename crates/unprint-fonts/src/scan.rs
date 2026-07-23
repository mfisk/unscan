use crate::types::FeatureTag;

#[derive(Clone, Debug)]
pub struct FontEntryMeta {
    pub family_name: String,
    pub postscript_name: Option<String>,
    pub is_monospace: bool,
    pub weight: u16,
    pub is_italic: bool,
}

pub fn scan_font_meta(font_data: &[u8]) -> Option<FontEntryMeta> {
    let face = ttf_parser::Face::parse(font_data, 0).ok()?;
    let family = face.names().into_iter()
        .find(|n| n.name_id == 1)
        .and_then(|n| n.to_string())
        .unwrap_or_else(|| "Unknown".to_string());
    let ps = face.names().into_iter()
        .find(|n| n.name_id == 6)
        .and_then(|n| n.to_string());
    let is_monospace = face.is_monospaced();
    let weight = face.weight().to_number();
    let is_italic = face.is_italic();
    Some(FontEntryMeta { family_name: family, postscript_name: ps, is_monospace, weight, is_italic })
}

pub fn detect_ot_features(font_data: &[u8]) -> Vec<FeatureTag> {
    let face = match rustybuzz::Face::from_slice(font_data, 0) {
        Some(f) => f,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    let ttf = match ttf_parser::Face::parse(font_data, 0) {
        Ok(tf) => tf,
        Err(_) => return out,
    };
    if let Some(gsub_data) = ttf.raw_face().table(ttf_parser::Tag::from_bytes(b"GSUB")) {
        let tags_to_check: &[&[u8;4]] = &[b"liga", b"dlig", b"smcp", b"c2sc", b"onum", b"lnum", b"frac", b"ss01", b"ss02", b"ss03"];
        for tag in tags_to_check {
            if gsub_data.windows(4).any(|w| w == *tag) {
                out.push(FeatureTag::from_bytes(tag));
            }
        }
    }
    let test_str = "fi fl ff";
    for tag_bytes in &[b"liga", b"dlig", b"smcp", b"onum", b"lnum"] {
        let tag = ttf_parser::Tag::from_bytes(*tag_bytes);
        let feat = rustybuzz::Feature::new(tag, 1, ..);
        let mut buf_default = rustybuzz::UnicodeBuffer::new();
        buf_default.push_str(test_str);
        let out_default = rustybuzz::shape(&face, &[], buf_default);

        let mut buf_feat = rustybuzz::UnicodeBuffer::new();
        buf_feat.push_str(test_str);
        let out_feat = rustybuzz::shape(&face, &[feat], buf_feat);

        if out_default.glyph_infos().len() != out_feat.glyph_infos().len() ||
           out_default.glyph_infos().iter().zip(out_feat.glyph_infos().iter()).any(|(a,b)| a.glyph_id != b.glyph_id)
        {
            out.push(FeatureTag::from_bytes(tag_bytes));
        }
    }
    out
}

pub fn detect_ligatures(font_data: &[u8]) -> Vec<String> {
    let face = match rustybuzz::Face::from_slice(font_data, 0) {
        Some(f) => f,
        None => return Vec::new(),
    };
    let liga_tag = ttf_parser::Tag::from_bytes(b"liga");
    let dlig_tag = ttf_parser::Tag::from_bytes(b"dlig");
    let features = [
        rustybuzz::Feature::new(liga_tag, 1, ..),
        rustybuzz::Feature::new(dlig_tag, 1, ..),
    ];
    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str("fi fl ff ffi ffl");
    let out = rustybuzz::shape(&face, &features, buf);
    let glyph_count = out.glyph_infos().len();
    if glyph_count < 11 {
        vec!["fi".into(), "fl".into(), "ff".into(), "ffi".into(), "ffl".into()]
    } else {
        Vec::new()
    }
}
