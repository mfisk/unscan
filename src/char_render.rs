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

use std::path::{Path, PathBuf};

use unprint_fonts::ab_glyph::{Font, GlyphId, PxScale, ScaleFont, point};
use unprint_fonts::KernProvider;
use image::{GrayImage, Luma};

/// Atomic PNG save: write to `.tmp` sibling then rename, so a SIGKILL never leaves a partial file.
/// Matches `atomic_file::tmp_for` pattern used for glyph_map and training features.
/// Uses explicit PngEncoder so the `.tmp` extension does not confuse `image::save`.
pub(crate) fn atomic_save_png(img: &GrayImage, final_path: &Path) -> bool {
    use std::fs::File;
    use std::io::BufWriter;
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    let tmp = crate::atomic_file::tmp_for(final_path);
    if let Some(parent) = tmp.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ok = (|| -> std::io::Result<()> {
        let file = File::create(&tmp)?;
        let mut w = BufWriter::new(file);
        let enc = PngEncoder::new(&mut w);
        enc.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::L8)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        // BufWriter flush on drop, but explicit flush for safety
        use std::io::Write;
        w.flush()?;
        Ok(())
    })()
    .is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    // On Unix, rename is atomic. If it fails, clean up tmp and report failure.
    if std::fs::rename(&tmp, final_path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

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
    /// Subpixel offset in x (fractional pixels, 0.0-1.0). For training
    /// robustness to MuPDF-style subpixel positioning during PDF rasterization.
    /// (Mike Nov 3, 2026: invasive subpixel training varies both x and y)
    pub subpixel_dx: f32,
    /// Subpixel offset in y (fractional pixels, 0.0-1.0).
    pub subpixel_dy: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            height: NORM_H,
            render_scale: 1,
            aa: AaVariant::Native,
            binarize_threshold: None,
            subpixel_dx: 0.0,
            subpixel_dy: 0.0,
        }
    }
}

/// Return the cache directory for rendered characters.
fn cache_dir() -> PathBuf {
    crate::cache::paths::chars_dir()
}

#[allow(dead_code)]
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
    // Include subpixel offsets in path (Mike Nov 4, 2026) — otherwise
    // dx=0.25 would collide with dx=0 in cache.
    let sp_dir = format!("dx{}_dy{}", params.subpixel_dx, params.subpixel_dy);
    let fname = format!("{}.png", glyph_map::hash_hex(img_hash));

    cache_dir().join(params_dir).join(aa_dir).join(sp_dir).join(seq_dir_name(seq)).join(fname)
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
        image::open(&path).ok().map(|d| d.into_luma8())
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
///
/// The caller owns glyph-map publication; this function never writes the
/// process-wide glyph-map cache implicitly.
/// `kern` is the REQUIRED GPOS pair-kern provider for this font (Finding 5).
pub fn render_ngram(
    fe: &crate::font_scan::FontEntry,
    seq: &[char],
    glyph_map: &mut glyph_map::NgramGlyphMap,
    params: &RenderParams,
    kern: &dyn KernProvider,
) -> Option<(u64, GrayImage)> {
    let fk = fe.font_key_ref();

    // Aggressive ligature use (Mike Nov 6, 2026): normalize the sequence
    // for this font entry — expand any ligature characters to components,
    // then collapse per the font's collapsed_lig_set(). "fi" renders as
    // the single FB01 glyph for lig variants; nolig variants (allowed==0)
    // render f, i separately, even when the input carries the combined
    // U+FB01 character. Idempotent for already-normalized sequences.
    // Cache lookup, overrides, render, and registration all use the
    // normalized sequence (the actual rendered content).
    let collapsed = normalize_seq_for_font(fe, seq);
    let seq = &collapsed[..];

    // Cache read: check glyph_map for known hash, then image cache
    if let Some(hash) = glyph_map.hash_for_font(seq, &fk) {
        if let Some(img) = load_cached_ngram(seq, hash, params) {
            return Some((hash, img));
        }
    }

    // Cache miss: load font, render, update caches
    let font_data = std::fs::read(&fe.path).ok()?;
    let mut font = unprint_fonts::ab_glyph::FontRef::try_from_slice(&font_data).ok()?;
    fe.apply_variations(&mut font);
    let gid_overrides: Vec<Option<GlyphId>> = seq.iter().map(|c| {
        fe.glyph_overrides.as_deref()
            .and_then(|ovs| ovs.iter().find(|(ch, _)| *ch == *c).map(|(_, g)| GlyphId(*g)))
    }).collect();

    let disable_kern = fe.variant_tag.contains("nokern");
    let img = render_ngram_fresh(&font, seq, &gid_overrides, params, disable_kern, kern)?;
    let img_hash = glyph_map::hash_image(&img);

    // Update glyph_map
    glyph_map.register(seq, &fk, img_hash);

    // Write image cache — always atomic, and always overwrite on miss path
    // to heal any previously partial/corrupt file (exists check alone would keep corrupt file).
    let path = ngram_cache_path(seq, img_hash, params);
    let _ = atomic_save_png(&img, &path);

    Some((img_hash, img))
}

/// Normalize a character sequence for rendering with a font entry.
/// Expands any ligature characters to their components, then collapses
/// per the font's `collapsed_lig_set()` — so a lig font renders "fi" as
/// the single U+FB01 glyph and a nolig font renders it as separate f, i
/// glyphs, regardless of whether the input sequence carries the combined
/// ligature character or the separate characters. Idempotent. This is
/// what makes the renderer honor the explicit |lig / |nolig distinction:
/// the image rendered for a (font, sequence) pair is always what that
/// font actually renders for those characters.
pub fn normalize_seq_for_font(fe: &crate::font_scan::FontEntry, seq: &[char]) -> Vec<char> {
    normalize_seq_for_lig_mask(seq, fe.collapsed_lig_set())
}

/// Normalize a character sequence for rendering with a ligature mask
/// (the font's `collapsed_lig_set()`). Same as `normalize_seq_for_font`
/// but takes the mask directly, for render paths that don't hold the
/// FontEntry (e.g. the full-line verify render, which shapes word text
/// via HarfBuzz — the text must already carry the FB01 character, since
/// HarfBuzz cannot invent a ligature the font has no GSUB rule for).
pub fn normalize_seq_for_lig_mask(seq: &[char], allowed: u8) -> Vec<char> {
    let expanded = crate::segment::expand_ligature_chars(seq);
    crate::segment::collapse_ligature_chars_for_allowed(&expanded, allowed)
}

/// The actual rendering pipeline — no caching.
/// `kern` is the REQUIRED GPOS pair-kern provider for this font (Finding 5):
/// the single source of pair kern for the render. There is no fallback to
/// the legacy `kern` table. When `disable_kern` is set (nokern variants)
/// the kern is 0 and the provider is not consulted.
pub fn render_ngram_fresh<F: Font>(
    font: &F,
    seq: &[char],
    gid_overrides: &[Option<GlyphId>],
    params: &RenderParams,
    disable_kern: bool,
    kern: &dyn KernProvider,
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

    // Position glyphs with advances and kerning (unless disabled for nokern variants)
    let mut x_pos = Vec::with_capacity(gids.len());
    x_pos.push(0.0f32);
    for i in 1..gids.len() {
        let prev_x = x_pos[i - 1];
        // GPOS pair kern (Finding 5): the provider's font-unit kern scaled
        // to px with the same factor ab_glyph uses for sf.kern()
        // (scale.x / height_unscaled). Never the legacy kern table.
        let kern_px = if disable_kern {
            0.0
        } else {
            kern.pair_kern_fu(gids[i - 1].0, gids[i].0) as f32 * scale.x / font.height_unscaled()
        };
        x_pos.push(prev_x + sf.h_advance(gids[i - 1]) + kern_px);
    }

    // Outline all glyphs and compute combined bounding box
    let mut outlines = Vec::with_capacity(gids.len());
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for (i, &gid) in gids.iter().enumerate() {
        // Apply subpixel offsets for training robustness (Mike Nov 3, 2026).
        // MuPDF-style rasterization positions glyphs at fractional pixel
        // coordinates; training must cover these variations.
        let g = gid.with_scale_and_position(
            scale,
            point(x_pos[i] + params.subpixel_dx, baseline_y + params.subpixel_dy)
        );
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

    // Raw-buffer min-blend: avoids get_pixel/put_pixel per glyph pixel.
    let canvas_w = img_w as usize;
    let canvas_raw = canvas.as_mut();
    for outlined in &outlines {
        let ob = outlined.px_bounds();
        outlined.draw(|gx, gy, cov| {
            let px = gx as i32 + (ob.min.x.floor() as i32) - ox + 1;
            let py = gy as i32 + (ob.min.y.floor() as i32) - oy + 1;
            if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
                let val = (255.0 * (1.0 - cov)) as u8;
                let idx = py as usize * canvas_w + px as usize;
                // SAFETY: bounds checked above, idx < canvas_raw.len()
                let cur = canvas_raw[idx];
                if val < cur {
                    canvas_raw[idx] = val;
                }
            }
        });
    }

    // Normalize: ink-crop, 1px pad, resize to params.height
    let normalized = features::normalize_to_ink_bounds(&canvas, params.height)?;

    // Optimized AA + binarize: apply_inplace avoids clone for Native (0 alloc) and
    // reuses buffer for Sharpen (1 alloc vs 2). Binarize uses in-place raw-buffer loop.
    let base = params.aa.apply_inplace(normalized);
    match params.binarize_threshold {
        Some(t) => Some(features::binarize_inplace(base, t)),
        None => Some(base),
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

    // Raw-buffer min-blend for glyph rasterization.
    let cw = img_w as usize;
    let craw = canvas.as_mut();
    outlined2.draw(|gx, gy, cov| {
        let px = gx as i32 + (b2.min.x.floor() as i32) - ox + 1;
        let py = gy as i32 + (b2.min.y.floor() as i32) - oy + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let idx = py as usize * cw + px as usize;
            let cur = craw[idx];
            if val < cur {
                craw[idx] = val;
            }
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
        // Single-glyph renders: kerning is undefined for one glyph (no
        // pair), so the provider is never consulted. NoKern is explicit
        // here, not a fallback — this dump has no geo cache available.
        if let Some(img) = render_ngram_fresh(&font, &[c], &[None], &RenderParams::default(), false, &unprint_fonts::NoKern) {
            let fname = format!("U+{:04X}.png", c as u32);
            let final_path = out.join(&fname);
            let _ = atomic_save_png(&img, &final_path);
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
