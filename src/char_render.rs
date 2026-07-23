//! Unified n-gram rendering + caching module.
//!
//! All callers that need rendered font glyph images go through
//! `render_ngram()` or its convenience wrapper `render_ngram_default()`.
//! Single characters are single-element sequences: `&['a']`.
//!
//! Pipeline (always this order):
//! 1. Render glyphs at shared em-size so tallest fills `height * render_scale`
//! 2. `normalize_to_ink_bounds()` — find ink, crop with 1px pad, Lanczos3 resize to `height`
//! 3. Apply AA variant
//! 4. If `binarize_threshold` is Some(t): binarize
//! 5. Hash → cache as PNG under `~/.cache/unprint/chars/`
//!
//! Cache path: `chars/h{H}_s{S}/{aa}_{binarize}/{seq_dir}/{hash}.png`
//! where `seq_dir` is `U+0061` for a single char or `U+0066_U+0069` for a bigram.

use std::path::PathBuf;

use unprint_fonts::ab_glyph::{Font, GlyphId, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};

use crate::features::{self as features, AaVariant, NORM_H};
use crate::glyph_map;


/// Parameters controlling how a character is rendered.
#[derive(Clone)]
pub struct RenderParams {
    /// Target normalized height in pixels (default: NORM_H = 24).
    pub height: u32,
    /// Multiplier for hi-res render before downscale (default: 1).
    pub render_scale: u32,
    /// Antialiasing variant applied after rendering (default: Native).
    pub aa: AaVariant,
    /// Binarize threshold after normalization.  None = keep grayscale.
    /// Some(128) = binarize at 128 (pixels < 128 → black, >= 128 → white).
    pub binarize_threshold: Option<u8>,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            height: NORM_H,
            render_scale: 1,
            aa: AaVariant::Native,
            binarize_threshold: None,
        }
    }
}

/// Return the cache directory for rendered characters.
fn cache_dir() -> PathBuf {
    crate::cache::paths::chars_dir()
}

fn dirs_cache_dir() -> PathBuf {
    crate::cache::cache_dir()
}

/// Build the sequence directory name: `U+0061` for len-1, `U+0066_U+0069` for len-2, etc.
fn seq_dir_name(seq: &[char]) -> String {
    seq.iter()
        .map(|&c| format!("U+{:04X}", c as u32))
        .collect::<Vec<_>>()
        .join("_")
}

/// Build the hash-addressed cache path for a rendered ngram image.
/// Structure: chars/h{H}_s{S}/{aa}_{binarize}/{seq_dir}/{hash}.png
pub fn ngram_cache_path(
    seq: &[char],
    img_hash: u64,
    params: &RenderParams,
) -> PathBuf {
    let params_dir = format!("h{}_s{}", params.height, params.render_scale);
    let binarize_tag = match params.binarize_threshold {
        Some(t) => format!("b{}", t),
        None => "bnone".to_string(),
    };
    let aa_dir = format!("{}_{}", params.aa.name(), binarize_tag);
    let fname = format!("{}.png", glyph_map::hash_hex(img_hash));

    cache_dir().join(params_dir).join(aa_dir).join(seq_dir_name(seq)).join(fname)
}

/// Load a cached ngram image by its content hash.
/// Returns `None` if no cached image exists for this hash.
pub fn load_cached_ngram(
    seq: &[char],
    img_hash: u64,
    params: &RenderParams,
) -> Option<GrayImage> {
    let path = ngram_cache_path(seq, img_hash, params);
    if path.exists() {
        image::open(&path).ok().map(|d| d.to_luma8())
    } else {
        None
    }
}

/// Render an n-gram (one or more adjacent characters) through the canonical
/// pipeline, with hash-addressed caching.
///
/// All glyphs are rendered at the same em-size — the tallest glyph fills
/// `params.height` and shorter glyphs appear proportionally smaller.
/// Adjacent glyphs include kerning. For a single character, this degenerates
/// to scaling that character's ink to fill the height.
///
/// Returns `None` if any glyph in the sequence is missing from the font.
/// Returns `Some((hash, image))` — the content hash uniquely identifies
/// the rendered image and is used as the glyph_id key.
/// Render an n-gram for a font entry, with full cache integration.
///
/// 1. Check glyph_map for a known hash → try image cache → return on hit.
/// 2. On miss: load font from fe.path, render, update glyph_map and image cache.
pub fn render_ngram(
    fe: &crate::font_scan::FontEntry,
    seq: &[char],
    glyph_map: &mut glyph_map::NgramGlyphMap,
    params: &RenderParams,
) -> Option<(u64, GrayImage)> {
    let fk = fe.font_key();

    // Cache read: check glyph_map for known hash, then image cache
    if let Some(hash) = glyph_map.hash_for_font(seq, &fk) {
        if let Some(img) = load_cached_ngram(seq, hash, params) {
            return Some((hash, img));
        }
    }

    // Cache miss: load font, render, update caches
    let font_data = std::fs::read(&fe.path).ok()?;
    let font = unprint_fonts::ab_glyph::FontRef::try_from_slice(&font_data).ok()?;
    let gid_overrides: Vec<Option<GlyphId>> = seq.iter().map(|c| {
        fe.glyph_overrides.as_deref()
            .and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| GlyphId(*g)))
    }).collect();

    let img = render_ngram_fresh(&font, seq, &gid_overrides, params)?;
    let img_hash = glyph_map::hash_image(&img);

    // Update glyph_map
    glyph_map.register(seq, &fk, img_hash);

    // Write image cache
    let path = ngram_cache_path(seq, img_hash, params);
    if !path.exists() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = img.save(&path);
    }

    Some((img_hash, img))
}

/// The actual rendering pipeline — no caching.
pub fn render_ngram_fresh<F: Font>(
    font: &F,
    seq: &[char],
    gid_overrides: &[Option<GlyphId>],
    params: &RenderParams,
) -> Option<GrayImage> {
    if seq.is_empty() {
        return None;
    }

    // Resolve glyph IDs
    let gids: Vec<GlyphId> = seq.iter().enumerate().map(|(i, &c)| {
        gid_overrides.get(i).copied().flatten().unwrap_or_else(|| font.glyph_id(c))
    }).collect();

    // All must have glyphs
    if gids.iter().any(|g| g.0 == 0) {
        return None;
    }

    // Measure ink heights at a reference em-size to find the tallest glyph
    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);

    let measure_ink_h = |gid: GlyphId| -> Option<f32> {
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

    // Scale so the tallest glyph's ink height = target
    let target_ink_h = params.height * params.render_scale;
    let target_scale = ref_h * (target_ink_h as f32 / max_ink_h);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);
    let baseline_y = sf.ascent();

    // Position glyphs with advances and kerning
    let mut x_pos = Vec::with_capacity(gids.len());
    x_pos.push(0.0f32);
    for i in 1..gids.len() {
        let prev_x = x_pos[i - 1];
        x_pos.push(prev_x + sf.h_advance(gids[i - 1]) + sf.kern(gids[i - 1], gids[i]));
    }

    // Outline all glyphs and compute combined bounding box
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

    // Normalize: ink-crop, 1px pad, resize to params.height
    let normalized = features::normalize_to_ink_bounds(&canvas, params.height)?;

    // Apply AA variant and optional binarization
    let aa_applied = params.aa.apply(&normalized);
    match params.binarize_threshold {
        Some(t) => Some(features::binarize(&aa_applied, t)),
        None => Some(aa_applied),
    }
}

/// Render a glyph at a specific ink height (in pixels).
/// Returns the raw rendered canvas (grayscale, white background, black ink).
pub fn render_glyph_at_ink_height<F: Font>(font: &F, gid: GlyphId, target_ink_h: u32) -> Option<GrayImage> {
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


/// Falls back to the font's default cmap lookup if no override exists for this character.
pub fn resolve_glyph<F: unprint_fonts::ab_glyph::Font>(font: &F, ch: char, overrides: Option<&[(char, u16)]>) -> unprint_fonts::ab_glyph::GlyphId {
    if let Some(ovs) = overrides {
        if let Some(&(_, gid)) = ovs.iter().find(|(c, _)| *c == ch) {
            return unprint_fonts::ab_glyph::GlyphId(gid);
        }
    }
    font.glyph_id(ch)
}
pub fn render_ref_chars(json_str: &str) {
    use unprint_fonts::ab_glyph::{Font, FontVec};

    #[derive(serde::Deserialize)]
    struct Req {
        font: String,
        chars: String,
        output_dir: String,
    }

    let req: Req = serde_json::from_str(json_str).unwrap_or_else(|e| {
        eprintln!("Invalid --render-ref-chars JSON: {e}");
        std::process::exit(1);
    });

    let font_data = std::fs::read(&req.font).unwrap_or_else(|e| {
        eprintln!("Cannot read font {:?}: {e}", req.font);
        std::process::exit(1);
    });
    let font = FontVec::try_from_vec(font_data).unwrap_or_else(|e| {
        eprintln!("Cannot parse font {:?}: {e}", req.font);
        std::process::exit(1);
    });

    let out = std::path::Path::new(&req.output_dir);
    std::fs::create_dir_all(out).unwrap_or_else(|e| {
        eprintln!("Cannot create output dir {:?}: {e}", req.output_dir);
        std::process::exit(1);
    });

    let mut rendered = 0u32;
    for c in req.chars.chars() {
        if font.glyph_id(c).0 == 0 {
            continue;
        }
        if let Some(img) = render_ngram_fresh(&font, &[c], &[None], &RenderParams::default()) {
            let fname = format!("U+{:04X}.png", c as u32);
            let _ = img.save(out.join(&fname));
            rendered += 1;
        }
    }
    eprintln!("Rendered {rendered} glyphs");
    std::process::exit(0);
}

/// Compute per-glyph vertical ink position ratios for a font.
///
/// Returns a map of char → (top_frac, bottom_frac) where fractions express
/// the glyph's ink bounding box relative to the font's full vertical extent
/// (ascent to descent).  0.0 = ascender line, 1.0 = descender line.
///
/// Uses `outline_glyph` + `px_bounds` — no rasterisation, just outline math.
pub fn glyph_metric_ratios<F: unprint_fonts::ab_glyph::Font>(
    font: &F,
    chars: &[char],
    overrides: Option<&[(char, u16)]>,
) -> std::collections::HashMap<char, (f32, f32)> {
    use unprint_fonts::ab_glyph::{PxScale, point};

    let scale = PxScale::from(100.0); // arbitrary reference scale; ratios are scale-invariant
    let sf = font.as_scaled(scale);
    let ascent = sf.ascent();
    let descent = sf.descent(); // negative value
    let full_height = ascent - descent;
    if full_height < 1.0 { return std::collections::HashMap::new(); }

    let mut result = std::collections::HashMap::new();
    for &ch in chars {
        let gid = resolve_glyph(font, ch, overrides);
        if gid.0 == 0 { continue; }

        let glyph = gid.with_scale_and_position(scale, point(0.0, ascent));
        let outlined = match font.outline_glyph(glyph) {
            Some(o) => o,
            None => continue,
        };
        let b = outlined.px_bounds();
        // b.min.y = top of ink, b.max.y = bottom of ink
        // Both in pixel coords where y=0 is top and baseline is at y=ascent
        let top_frac = b.min.y / full_height;
        let bottom_frac = b.max.y / full_height;
        result.insert(ch, (top_frac, bottom_frac));
    }
    result
}
