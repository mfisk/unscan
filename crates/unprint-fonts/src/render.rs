use std::sync::Arc;

use ab_glyph::{Font, GlyphId as AbGlyphId, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};

use crate::types::{AaMode, RenderParams, RenderResult};

#[derive(Clone)]
pub struct FontHandle {
    data: Arc<[u8]>,
}

impl FontHandle {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        // Validate that ab_glyph can parse it
        let _ = ab_glyph::FontRef::try_from_slice(&bytes).map_err(|e| format!("parse font: {:?}", e))?;
        Ok(Self { data: Arc::from(bytes.into_boxed_slice()) })
    }

    pub fn from_arc(data: Arc<[u8]>) -> Result<Self, String> {
        let _ = ab_glyph::FontRef::try_from_slice(&data).map_err(|e| format!("parse font: {:?}", e))?;
        Ok(Self { data })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

pub fn hash_image(img: &GrayImage) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    img.width().hash(&mut hasher);
    img.height().hash(&mut hasher);
    img.as_raw().hash(&mut hasher);
    hasher.finish()
}

pub fn hash_hex(h: u64) -> String {
    format!("{:016x}", h)
}

// Copied from main crate features::normalize_to_ink_bounds
fn normalize_to_ink_bounds(img: &GrayImage, target_h: u32) -> Option<GrayImage> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    const THRESH: u8 = 200;
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y).0[0] < THRESH {
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }
    }
    if min_x > max_x || min_y > max_y {
        return None;
    }
    let ink_w = max_x - min_x + 1;
    let ink_h = max_y - min_y + 1;
    if ink_w < 1 || ink_h < 1 {
        return None;
    }
    let pad = 1u32;
    let canvas_w = ink_w + 2 * pad;
    let canvas_h = ink_h + 2 * pad;
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = img.get_pixel(x, y);
            canvas.put_pixel(x - min_x + pad, y - min_y + pad, *px);
        }
    }
    let scaled_w = (canvas_w as f32 * target_h as f32 / canvas_h as f32).ceil() as u32;
    if scaled_w < 2 {
        return None;
    }
    Some(image::imageops::resize(
        &canvas,
        scaled_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
    ))
}

fn binarize(img: &GrayImage, threshold: u8) -> GrayImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        p.0[0] = if p.0[0] < threshold { 0 } else { 255 };
    }
    out
}

fn apply_aa(img: &GrayImage, mode: AaMode) -> GrayImage {
    match mode {
        AaMode::Native => img.clone(),
        AaMode::None => img.clone(),
        AaMode::Mono => img.clone(), // Mono handled via binarize later; keep simple
    }
}

// Internal: render one ngram given already parsed font
fn render_ngram_fresh_inner<F: Font>(
    font: &F,
    seq: &[char],
    gid_overrides: &[Option<AbGlyphId>],
    params: &RenderParams,
) -> Option<GrayImage> {
    if seq.is_empty() {
        return None;
    }
    let gids: Vec<AbGlyphId> = seq.iter().enumerate().map(|(i, &c)| {
        gid_overrides.get(i).copied().flatten().unwrap_or_else(|| font.glyph_id(c))
    }).collect();

    if gids.iter().any(|g| g.0 == 0) {
        return None;
    }

    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);

    let measure_ink_h = |gid: AbGlyphId| -> Option<f32> {
        let g = gid.with_scale_and_position(ref_scale, point(0.0, sf_ref.ascent()));
        let outlined = font.outline_glyph(g)?;
        let b = outlined.px_bounds();
        Some(b.max.y - b.min.y)
    };

    let mut max_ink_h = 0.0f32;
    for &gid in &gids {
        let h = measure_ink_h(gid)?;
        if h > max_ink_h {
            max_ink_h = h;
        }
    }
    if max_ink_h < 1.0 {
        return None;
    }

    let target_ink_h = params.height * params.render_scale;
    let target_scale = ref_h * (target_ink_h as f32 / max_ink_h);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);
    let baseline_y = sf.ascent();

    let mut x_pos = Vec::with_capacity(gids.len());
    x_pos.push(0.0f32);
    for i in 1..gids.len() {
        let prev_x = x_pos[i - 1];
        x_pos.push(prev_x + sf.h_advance(gids[i - 1]) + sf.kern(gids[i - 1], gids[i]));
    }

    let mut outlines = Vec::with_capacity(gids.len());
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for (i, &gid) in gids.iter().enumerate() {
        let g = gid.with_scale_and_position(scale, point(x_pos[i], baseline_y));
        let outlined = font.outline_glyph(g)?;
        let b = outlined.px_bounds();
        min_x = min_x.min(b.min.x);
        min_y = min_y.min(b.min.y);
        max_x = max_x.max(b.max.x);
        max_y = max_y.max(b.max.y);
        outlines.push(outlined);
    }

    let img_w = (max_x - min_x).ceil() as u32 + 2;
    let img_h = (max_y - min_y).ceil() as u32 + 2;
    if img_w == 0 || img_h == 0 || img_w > 4000 || img_h > 2000 {
        return None;
    }

    let mut canvas = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));
    let ox = min_x.floor() as i32;
    let oy = min_y.floor() as i32;

    for outlined in &outlines {
        let ob = outlined.px_bounds();
        outlined.draw(|gx, gy, cov| {
            let px = gx as i32 + (ob.min.x.floor() as i32) - ox + 1;
            let py = gy as i32 + (ob.min.y.floor() as i32) - oy + 1;
            if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
                let val = (255.0 * (1.0 - cov)) as u8;
                let cur = canvas.get_pixel(px as u32, py as u32).0[0];
                canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
            }
        });
    }

    let normalized = normalize_to_ink_bounds(&canvas, params.height)?;
    let aa_applied = apply_aa(&normalized, params.aa);
    match params.binarize_threshold {
        Some(t) => Some(binarize(&aa_applied, t)),
        None => Some(aa_applied),
    }
}

/// Public batch API - renders many ngrams for one font in a tight loop.
/// Hot per-character work stays inside this crate for inlining.
pub fn render_ngrams_batch(
    handle: &FontHandle,
    seqs: &[&[char]],
    overrides_per_seq: &[Option<&[(char, u16)]>],
    params: &RenderParams,
) -> Vec<Option<RenderResult>> {
    let font = match ab_glyph::FontRef::try_from_slice(handle.as_slice()) {
        Ok(f) => f,
        Err(_) => return vec![None; seqs.len()],
    };

    let mut out = Vec::with_capacity(seqs.len());
    for (idx, seq) in seqs.iter().enumerate() {
        let ov = overrides_per_seq.get(idx).copied().flatten();
        let gid_overrides: Vec<Option<AbGlyphId>> = seq.iter().map(|c| {
            ov.and_then(|map| map.iter().find(|(ch, _)| *ch == *c).map(|(_, gid)| AbGlyphId(*gid)))
        }).collect();

        let img_opt = render_ngram_fresh_inner(&font, seq, &gid_overrides, params);
        if let Some(img) = img_opt {
            let h = hash_image(&img);
            out.push(Some(RenderResult::new(img, h)));
        } else {
            out.push(None);
        }
    }
    out
}

/// Single ngram helper built on batch impl
pub fn render_ngram_single(
    handle: &FontHandle,
    seq: &[char],
    overrides: Option<&[(char, u16)]>,
    params: &RenderParams,
) -> Option<RenderResult> {
    let seqs = [seq];
    let ovs = [overrides];
    let mut batch = render_ngrams_batch(handle, &seqs, &ovs, params);
    batch.pop().flatten()
}

/// Render from raw bytes without handle - for one-off use
pub fn render_ngram_from_bytes(
    font_data: &[u8],
    seq: &[char],
    gid_overrides_u16: &[Option<u16>],
    params: &RenderParams,
) -> Option<GrayImage> {
    let font = ab_glyph::FontRef::try_from_slice(font_data).ok()?;
    let gid_overrides: Vec<Option<AbGlyphId>> = gid_overrides_u16.iter().map(|o| o.map(AbGlyphId)).collect();
    render_ngram_fresh_inner(&font, seq, &gid_overrides, params)
}

pub fn render_glyph_at_ink_height(
    font_data: &[u8],
    gid_u16: u16,
    target_ink_h: u32,
) -> Option<GrayImage> {
    let font = ab_glyph::FontRef::try_from_slice(font_data).ok()?;
    let gid = AbGlyphId(gid_u16);
    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);
    let glyph = gid.with_scale_and_position(ref_scale, point(0.0, sf_ref.ascent()));
    let outlined = font.outline_glyph(glyph)?;
    let bounds = outlined.px_bounds();
    let ink_h_ref = bounds.max.y - bounds.min.y;
    if ink_h_ref < 1.0 {
        return None;
    }
    let target_scale = ref_h * (target_ink_h as f32 / ink_h_ref);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);
    let glyph2 = gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
    let outlined2 = font.outline_glyph(glyph2)?;
    let b2 = outlined2.px_bounds();
    let img_w = (b2.max.x - b2.min.x).ceil() as u32 + 2;
    let img_h = (b2.max.y - b2.min.y).ceil() as u32 + 2;
    if img_w == 0 || img_h == 0 || img_w > 2000 || img_h > 2000 {
        return None;
    }
    let mut canvas = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));
    let ox = b2.min.x.floor() as i32;
    let oy = b2.min.y.floor() as i32;
    outlined2.draw(|gx, gy, cov| {
        let px = gx as i32 + (b2.min.x.floor() as i32) - ox + 1;
        let py = gy as i32 + (b2.min.y.floor() as i32) - oy + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
        }
    });
    Some(canvas)
}

pub fn glyph_metric_ratios_batch(
    font_data: &[u8],
    chars: &[char],
    overrides: Option<&[(char, u16)]>,
) -> Vec<(char, f32, f32)> {
    let font = match ab_glyph::FontRef::try_from_slice(font_data) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let scale = PxScale::from(100.0);
    let sf = font.as_scaled(scale);
    let ascent = sf.ascent();
    let descent = sf.descent();
    let full_height = ascent - descent;
    if full_height < 1.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &ch in chars {
        let gid = if let Some(ovs) = overrides {
            if let Some(&(_, gid)) = ovs.iter().find(|(c, _)| *c == ch) {
                AbGlyphId(gid)
            } else {
                font.glyph_id(ch)
            }
        } else {
            font.glyph_id(ch)
        };
        if gid.0 == 0 { continue; }
        let glyph = gid.with_scale_and_position(scale, point(0.0, ascent));
        let outlined = match font.outline_glyph(glyph) {
            Some(o) => o,
            None => continue,
        };
        let b = outlined.px_bounds();
        let top_frac = b.min.y / full_height;
        let bottom_frac = b.max.y / full_height;
        out.push((ch, top_frac, bottom_frac));
    }
    out
}
