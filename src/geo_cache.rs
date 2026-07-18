//! Geometry cache for font glyph positions (BGEO v4).
//!
//! Caches predicted glyph positions for cacheable fonts to avoid
//! recomputing HarfBuzz shaping for every word. v4 rejects any font
//! with y_advance !=0, x_offset !=0, or y_offset !=0 on ASCII (GPOS
//! vertical/offsets would make cached static bbox wrong).

use std::collections::HashMap;
use std::path::PathBuf;

const BGEO_VERSION: u32 = 4;

#[derive(Clone)]
pub struct GlyphMetrics {
    pub advance: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

pub struct FontGeo {
    pub glyphs: HashMap<char, GlyphMetrics>, // ASCII 0x20-0x7E plus FB00-FB04
    pub kern_pairs: HashMap<(char, char), f64>, // pair kern in FU
    pub units_per_em: f64,
}

pub struct GeometryCache {
    fonts: HashMap<String, FontGeo>,
    _cache_path: PathBuf,
}

impl GeometryCache {
    /// Load or build the geometry cache.
    ///
    /// For now, builds in-memory from font registry (no disk persistence yet).
    /// Checks each font for GPOS vertical/offsets and ligatures.
    pub fn load_or_build(
        font_registry: &crate::font_scan::FontRegistry,
        font_cache: &crate::font_cache::FontCache,
    ) -> Self {
        let mut fonts = HashMap::new();
        let mut n_total = 0;
        let mut n_cacheable = 0;
        let mut n_reject_offset = 0;
        let mut n_reject_lig = 0;
        let n_reject_other = 0;

        for fe in font_registry.iter() {
            n_total += 1;
            let font_key = fe.font_key();
            let font_data = match font_cache.load(&fe.path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let face = match rustybuzz::Face::from_slice(&font_data, 0) {
                Some(f) => f,
                None => continue,
            };
            // Test ASCII singles and bigrams for GPOS offsets that would break cache.
            let mut has_offset = false;
            let mut has_lig = false;

            // Check singles 0x20-0x7E
            for c in 0x20u8..=0x7Eu8 {
                let ch = c as char;
                let text = ch.to_string();
                if let Some(shaped) = crate::layout::shape_word(&face, &[], &text) {
                    if shaped.y_offsets.iter().any(|&y| y != 0) {
                        has_offset = true;
                        break;
                    }
                    if shaped.x_offsets.iter().any(|&x| x != 0) {
                        has_offset = true;
                        break;
                    }
                    let mut buf = rustybuzz::UnicodeBuffer::new();
                    buf.push_str(&text);
                    let glyphs = rustybuzz::shape(&face, &[], buf);
                    for pos in glyphs.glyph_positions() {
                        if pos.y_advance != 0 {
                            has_offset = true;
                            break;
                        }
                        if pos.x_offset != 0 || pos.y_offset != 0 {
                            has_offset = true;
                            break;
                        }
                    }
                    if has_offset { break; }
                }
            }
            if has_offset {
                n_reject_offset += 1;
                continue;
            }

            // Use canonical ligature list (first 5 = ff/fi/fl/ffi/ffl -> FB00-FB04)
            for &(probe, _) in crate::font_scan::LIGATURE_PROBES.iter().filter(|(_, c)| matches!(c, '\u{FB00}'..='\u{FB04}')) {
                let mut buf = rustybuzz::UnicodeBuffer::new();
                buf.push_str(probe);
                let glyphs = rustybuzz::shape(&face, &[], buf);
                if glyphs.len() == 1 && probe.chars().count() > 1 {
                    has_lig = true;
                }
                for pos in glyphs.glyph_positions() {
                    if pos.y_advance != 0 || pos.x_offset != 0 || pos.y_offset != 0 {
                        has_offset = true;
                        break;
                    }
                }
                if has_offset { break; }
            }
            if has_offset {
                n_reject_offset += 1;
                continue;
            }

            let mut sampled_offsets = false;
            for a in [b'a', b'e', b'i', b'o', b'n', b't', b'T', b'A'] {
                for b in [b'a', b'e', b'i', b'o', b'n', b't', b'T', b'A', b' ', b'.', b','] {
                    let text = format!("{}{}", a as char, b as char);
                    let mut buf = rustybuzz::UnicodeBuffer::new();
                    buf.push_str(&text);
                    let glyphs = rustybuzz::shape(&face, &[], buf);
                    for pos in glyphs.glyph_positions() {
                        if pos.y_advance != 0 || pos.x_offset != 0 || pos.y_offset != 0 {
                            sampled_offsets = true;
                            break;
                        }
                    }
                    if sampled_offsets { break; }
                }
                if sampled_offsets { break; }
            }
            if sampled_offsets {
                n_reject_offset += 1;
                continue;
            }

            let ttfp = face.as_ref();
            let upem = face.units_per_em() as f64;
            let mut glyphs_map = HashMap::new();
            for c in 0x20u8..=0x7Eu8 {
                let ch = c as char;
                let gid = match ttfp.glyph_index(ch) {
                    Some(gid) => gid,
                    None => continue,
                };
                let adv = ttfp.glyph_hor_advance(gid).unwrap_or(0) as f64;
                let bbox = ttfp.glyph_bounding_box(gid);
                let (x_min, x_max, y_min, y_max) = if let Some(bb) = bbox {
                    (bb.x_min as f64, bb.x_max as f64, bb.y_min as f64, bb.y_max as f64)
                } else {
                    (0.0, adv, 0.0, 0.0)
                };
                glyphs_map.insert(ch, GlyphMetrics { advance: adv, x_min, x_max, y_min, y_max });
            }
            for &(_, ch) in crate::font_scan::LIGATURE_PROBES.iter().filter(|(_, c)| matches!(c, '\u{FB00}'..='\u{FB04}')) {
                if let Some(gid) = ttfp.glyph_index(ch) {
                    let adv = ttfp.glyph_hor_advance(gid).unwrap_or(0) as f64;
                    let bbox = ttfp.glyph_bounding_box(gid);
                    let (x_min, x_max, y_min, y_max) = if let Some(bb) = bbox {
                        (bb.x_min as f64, bb.x_max as f64, bb.y_min as f64, bb.y_max as f64)
                    } else {
                        (0.0, adv, 0.0, 0.0)
                    };
                    glyphs_map.insert(ch, GlyphMetrics { advance: adv, x_min, x_max, y_min, y_max });
                }
            }
            let kern_pairs = HashMap::new();

            fonts.insert(font_key, FontGeo { glyphs: glyphs_map, kern_pairs, units_per_em: upem });
            n_cacheable += 1;
            if has_lig {
                n_reject_lig += 1;
            }
            let _ = n_reject_other;
        }

        eprintln!("[geo-cache] Built v{} cache ({} fonts cacheable / {} total, {} reject offset, {} lig fonts)",
            BGEO_VERSION, n_cacheable, n_total, n_reject_offset, n_reject_lig);

        Self {
            fonts,
            _cache_path: PathBuf::from("/tmp/geo-cache-v4.bin"),
        }
    }

    pub fn has_font(&self, font_key: &str) -> bool {
        self.fonts.contains_key(font_key)
    }

    pub fn predict_glyph_x(&self, _seg_idx: usize, _orig_idx: usize) -> Option<f64> {
        None
    }
    pub fn predict_glyph_y(&self, _seg_idx: usize, _orig_idx: usize) -> Option<f64> {
        None
    }

    pub fn predict_glyph_positions(
        &self,
        font_key: &str,
        chars: &[char],
    ) -> Option<Vec<(f64, f64)>> {
        let fg = self.fonts.get(font_key)?;
        let mut out = Vec::with_capacity(chars.len());
        let mut cursor = 0.0f64;
        for (i, &c) in chars.iter().enumerate() {
            let gm = fg.glyphs.get(&c)?;
            let cx = cursor + (gm.x_min + gm.x_max) * 0.5;
            let cy = (gm.y_min + gm.y_max) * 0.5;
            out.push((cx, cy));
            cursor += gm.advance;
            if i + 1 < chars.len() {
                if let Some(kern) = fg.kern_pairs.get(&(c, chars[i+1])) {
                    cursor += kern;
                }
            }
        }
        Some(out)
    }

    pub fn predict_word_ink_extent(
        &self,
        font_key: &str,
        chars: &[char],
        _font_data: &[u8],
        _em_px: f64,
    ) -> Option<(f64, f64)> {
        let fg = self.fonts.get(font_key)?;
        let mut total_adv = 0.0f64;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &c in chars {
            let gm = fg.glyphs.get(&c)?;
            total_adv += gm.advance;
            min_y = min_y.min(gm.y_min);
            max_y = max_y.max(gm.y_max);
        }
        Some((total_adv, max_y - min_y))
    }
}
