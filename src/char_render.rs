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
//! Rendered images are cached as PNGs under `~/.cache/unprint/chars/`,
//! addressed by content hash (not font key) so identical renders are
//! stored exactly once.

use std::path::PathBuf;

use ab_glyph::{Font, GlyphId, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};

use crate::features::{self as features, AaVariant, NORM_H};
use crate::glyph_map;


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
        .map(|h| PathBuf::from(h).join(".cache").join("unprint"))
        .unwrap_or_else(|_| PathBuf::from(".cache/unprint"))
}

/// Build the hash-addressed cache path for a rendered glyph image.
/// Structure: chars/h{H}_s{S}/{aa}_{binarize}/U+XXXX/{hash}.png
fn hash_cache_path(
    c: char,
    img_hash: u64,
    params: &RenderParams,
) -> PathBuf {
    let params_dir = format!("h{}_s{}", params.height, params.render_scale);
    let binarize_tag = match params.binarize_threshold {
        Some(t) => format!("b{}", t),
        None => "bnone".to_string(),
    };
    let aa_dir = format!("{}_{}", params.aa.name(), binarize_tag);
    let char_dir = format!("U+{:04X}", c as u32);
    let fname = format!("{}.png", glyph_map::hash_hex(img_hash));

    cache_dir().join(params_dir).join(aa_dir).join(char_dir).join(fname)
}

/// Load a cached glyph image by its content hash.
/// Returns `None` if no cached image exists for this hash.
pub fn load_cached_glyph(
    c: char,
    img_hash: u64,
    params: &RenderParams,
) -> Option<GrayImage> {
    let path = hash_cache_path(c, img_hash, params);
    if path.exists() {
        image::open(&path).ok().map(|d| d.to_luma8())
    } else {
        None
    }
}

/// Return the filesystem path for a hash-addressed cached glyph image.
pub fn glyph_cache_path(
    c: char,
    img_hash: u64,
    params: &RenderParams,
) -> PathBuf {
    hash_cache_path(c, img_hash, params)
}


/// Render a character image through the canonical pipeline, with
/// hash-addressed caching.
///
/// Returns `None` if the font doesn't contain the glyph.
/// Returns `Some((hash, image))` — the content hash uniquely identifies
/// the rendered image and is used as the glyph_id key.
pub fn get_rendered_char<F: Font>(
    font: &F,
    c: char,
    glyph_id_override: Option<GlyphId>,
    params: &RenderParams,
) -> Option<(u64, GrayImage)> {
    // Always render (we need the image to compute the hash).
    // The cache deduplicates storage — identical images share a file.
    let img = render_char_fresh(font, c, glyph_id_override, params)?;
    let img_hash = glyph_map::hash_image(&img);

    // Save to hash-addressed cache (best-effort, skip if already exists)
    let path = hash_cache_path(c, img_hash, params);
    if !path.exists() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = img.save(&path);
    }

    Some((img_hash, img))
}

/// Convenience: render with default params (height=NORM_H, scale=1, native AA, no binarize).
pub fn get_rendered_char_default<F: Font>(
    font: &F,
    c: char,
    glyph_id_override: Option<GlyphId>,
) -> Option<(u64, GrayImage)> {
    get_rendered_char(font, c, glyph_id_override, &RenderParams::default())
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
    let normalized = features::normalize_to_ink_bounds(&canvas, params.height)?;

    // Step 3: apply AA variant (AFTER normalize, BEFORE binarize)
    let aa_applied = params.aa.apply(&normalized);

    // Step 4: binarize if requested (AFTER AA)
    match params.binarize_threshold {
        Some(t) => Some(features::binarize(&aa_applied, t)),
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


/// Falls back to the font's default cmap lookup if no override exists for this character.
pub fn resolve_glyph<F: ab_glyph::Font>(font: &F, ch: char, overrides: Option<&[(char, u16)]>) -> ab_glyph::GlyphId {
    if let Some(ovs) = overrides {
        if let Some(&(_, gid)) = ovs.iter().find(|(c, _)| *c == ch) {
            return ab_glyph::GlyphId(gid);
        }
    }
    font.glyph_id(ch)
}

// ---------------------------------------------------------------------------
// CLI subcommand: render reference characters
// ---------------------------------------------------------------------------

/// Render reference characters for a font and exit.
///
/// `json_str` is a JSON object: `{"font": "<path>", "chars": "<string>", "output_dir": "<path>"}`.
/// Each character with a glyph is rendered at the standard ink height and
/// saved as `U+XXXX.png` in the output directory.
pub fn render_ref_chars_and_exit(json_str: &str) -> ! {
    use ab_glyph::{Font, FontVec};

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
        if let Some((_hash, img)) = get_rendered_char_default(&font, c, None) {
            let fname = format!("U+{:04X}.png", c as u32);
            let _ = img.save(out.join(&fname));
            rendered += 1;
        }
    }
    eprintln!("Rendered {rendered} glyphs");
    std::process::exit(0);
}
