//! Single shared character rendering + caching module.
//!
//! All callers that need a rendered font character image go through
//! `get_rendered_char()` or its convenience wrapper `get_rendered_char_default()`.
//!
//! Pipeline (always this order):
//! 1. Render glyph at `height * render_scale` ink height via ab_glyph
//! 2. `normalize_to_ink_bounds()` — find ink, crop with 1px pad, Lanczos3 resize to `height`
//! 3. If `binarize_threshold` is Some(t): `binarize(&img, t)` — AFTER normalize
//! 4. Return image
//!
//! Rendered images are cached as individual PNGs under `~/.cache/unscan/chars/`.

use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

use ab_glyph::{Font, GlyphId, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};

use crate::char_index::{self, AaVariant, NORM_H};

/// Parameters controlling how a character is rendered.
#[derive(Clone)]
pub struct RenderParams {
    /// Target normalized height in pixels (default: NORM_H = 24).
    pub height: u32,
    /// Multiplier for hi-res render before downscale (default: 3, simulates ~300dpi).
    /// A value of 1 means render directly at `height` (no downscale).
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
    dirs_cache_dir().join("chars")
}

fn dirs_cache_dir() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".cache").join("unscan"))
        .unwrap_or_else(|_| PathBuf::from(".cache/unscan"))
}

/// Build the cache file path for a given render configuration.
/// Structure: chars/h{H}_s{S}/{aa}_{binarize}/{font_dir}/U+XXXX[_g{gid}].png
/// Font dir uses a sanitized (percent-encoded) font key for readability.
/// Fewest-values directories first, fan out into many fonts, then characters.
fn cache_path(
    font_key: &str,
    c: char,
    glyph_id_override: Option<GlyphId>,
    params: &RenderParams,
) -> PathBuf {
    let font_dir = sanitize_font_key(font_key);

    // Params dir: h24_s3
    let params_dir = format!("h{}_s{}", params.height, params.render_scale);

    // AA + binarize dir: native_b128 or sharpen_bnone
    let binarize_tag = match params.binarize_threshold {
        Some(t) => format!("b{}", t),
        None => "bnone".to_string(),
    };
    let aa_dir = format!("{}_{}", params.aa.name(), binarize_tag);

    // Filename: U+0041.png or U+0041_g123.png
    let mut fname = format!("U+{:04X}", c as u32);
    if let Some(gid) = glyph_id_override {
        write!(fname, "_g{}", gid.0).unwrap();
    }
    fname.push_str(".png");

    cache_dir().join(params_dir).join(aa_dir).join(font_dir).join(fname)
}

/// SHA-256 of a string, returning the first `n` hex characters.
/// Encode a font key for use as a directory name.
/// Keeps ASCII alphanumerics, dash, dot, space literal; encodes everything else
/// as _XX hex. Underscore is the escape prefix.
fn sanitize_font_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for b in key.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b' ' => {
                out.push(b as char);
            }
            b'_' => out.push_str("_5F"),
            _ => {
                write!(out, "_{:02X}", b).unwrap();
            }
        }
    }
    out
}


/// Render a character image through the canonical pipeline, with caching.
///
/// Returns `None` if the font doesn't contain the glyph.
pub fn get_rendered_char<F: Font>(
    font: &F,
    font_key: &str,
    c: char,
    glyph_id_override: Option<GlyphId>,
    params: &RenderParams,
) -> Option<GrayImage> {
    let path = cache_path(font_key, c, glyph_id_override, params);

    // Cache hit: read from disk
    if path.exists() {
        if let Ok(dyn_img) = image::open(&path) {
            return Some(dyn_img.to_luma8());
        }
        // Corrupted cache file — fall through to re-render
    }

    // Cache miss: render from scratch
    let img = render_char_fresh(font, c, glyph_id_override, params)?;

    // Save to cache (best-effort, don't fail on I/O errors)
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = img.save(&path);

    Some(img)
}

/// Convenience: render with default params (height=NORM_H, scale=1, native AA, no binarize).
pub fn get_rendered_char_default<F: Font>(
    font: &F,
    font_key: &str,
    c: char,
    glyph_id_override: Option<GlyphId>,
) -> Option<GrayImage> {
    get_rendered_char(font, font_key, c, glyph_id_override, &RenderParams::default())
}

/// The actual rendering pipeline — no caching.
fn render_char_fresh<F: Font>(
    font: &F,
    c: char,
    glyph_id_override: Option<GlyphId>,
    params: &RenderParams,
) -> Option<GrayImage> {
    let gid = match glyph_id_override {
        Some(g) => g,
        None => font.glyph_id(c),
    };
    if gid.0 == 0 {
        return None;
    }

    // Step 1: Render at hi-res (height * render_scale ink height)
    let target_ink_h = params.height * params.render_scale;
    let canvas = render_glyph_at_ink_height(font, gid, target_ink_h)?;

    // Step 2: normalize_to_ink_bounds — crop to ink, 1px pad, Lanczos3 resize to params.height
    // If render_scale == 1, the image is already at target height, but we still
    // run normalize to get consistent ink cropping and padding.
    let normalized = char_index::normalize_to_ink_bounds(&canvas, params.height)?;

    // Step 3: apply AA variant (AFTER normalize, BEFORE binarize)
    let aa_applied = params.aa.apply(&normalized);

    // Step 4: binarize if requested (AFTER AA)
    match params.binarize_threshold {
        Some(t) => Some(char_index::binarize(&aa_applied, t)),
        None => Some(aa_applied),
    }
}

/// Render a glyph at a specific ink height (in pixels).
/// Returns the raw rendered canvas (grayscale, white background, black ink).
pub fn render_glyph_at_ink_height<F: Font>(font: &F, gid: GlyphId, target_ink_h: u32) -> Option<GrayImage> {
    // Measure ink height at a reference scale
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

    // Scale to get target ink height
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

