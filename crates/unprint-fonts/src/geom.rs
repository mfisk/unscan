use crate::types::{Bbox, GlyphBBox};

#[derive(Clone, Debug, Default)]
pub struct GlyphKernTable {
    pub pairs: Vec<(u16, u16, f32)>,
}

pub fn glyph_bboxes_batch(
    font_data: &[u8],
    gids: &[u16],
) -> Vec<GlyphBBox> {
    let face = match ttf_parser::Face::parse(font_data, 0) {
        Ok(f) => f,
        Err(_) => return gids.iter().map(|&gid| GlyphBBox { gid, bbox: None }).collect(),
    };
    gids.iter().map(|&gid| {
        let gid_t = ttf_parser::GlyphId(gid);
        let bbox = face.glyph_bounding_box(gid_t).map(|r| Bbox {
            min_x: r.x_min as f32,
            min_y: r.y_min as f32,
            max_x: r.x_max as f32,
            max_y: r.y_max as f32,
        });
        GlyphBBox { gid, bbox }
    }).collect()
}

pub fn kerning_table(_font_data: &[u8]) -> GlyphKernTable {
    GlyphKernTable::default()
}
