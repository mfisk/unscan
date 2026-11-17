//! SSIM verification — render vector text back to raster and compare with the
//! original to catch bad replacements before they make it into the output.
//!
//! v5: width-scaled SSIM for glyph-shape comparison + aspect-ratio penalty
//! for catching fonts whose proportions don't match the original.

use unprint_fonts::ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{GrayImage, Luma, RgbImage};
use crate::ocr::TextRegion;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// FreeType variable-font axis coordination helper
// ---------------------------------------------------------------------------

/// Apply variable-font design coordinates to a FreeType face.
///
/// Reads the fvar axis order from the raw font data (via ttf_parser) and maps
/// the provided tag→value pairs to `FT_Set_Var_Design_Coordinates`.
fn set_ft_variations<B>(ft_face: &unprint_fonts::freetype::Face<B>, font_data: &[u8], vars: &[([u8; 4], f32)]) {
    let parsed = match unprint_fonts::ttf_parser::Face::parse(font_data, 0) {
        Ok(f) => f,
        Err(_) => return,
    };

    // Get the axis list in fvar order
    let axes: Vec<unprint_fonts::ttf_parser::VariationAxis> = parsed.variation_axes().into_iter().collect();
    if axes.is_empty() { return; }

    // Build coordinate array: each axis gets its value from vars, or its default
    let coords: Vec<i64> = axes.iter().map(|ax| {
        let val = vars.iter()
            .find(|(tag, _)| *tag == ax.tag.to_bytes())
            .map(|(_, v)| *v)
            .unwrap_or(ax.def_value);
        // FT_Fixed is 16.16 fixed-point
        (val as f64 * 65536.0).round() as i64
    }).collect();

    // Call FT_Set_Var_Design_Coordinates via raw FFI
    let raw_face: unprint_fonts::freetype::freetype_sys::FT_Face = ft_face.raw() as *const _ as *mut _;
    unsafe {
        unprint_fonts::freetype::freetype_sys::FT_Set_Var_Design_Coordinates(
            raw_face,
            coords.len() as u32,
            coords.as_ptr() as *const unprint_fonts::freetype::freetype_sys::FT_Fixed,
        );
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct WordPlacement {
    pub text: String,
    pub x_off: u32,
    pub y_off: u32,
    pub width: u32,
    pub height: u32,
}

/// Result of verifying a text region against a font render.
pub struct VerifyResult {
    /// ZNCC similarity score (0–1).
    pub score: f32,
    /// Vertical shift that produced the best score.
    pub dy: i32,
    /// Ink-cropped render image (for display).
    pub render_ink: Option<GrayImage>,
    /// Colored diff image: red = scan-only ink, blue = render-only ink, white = agreement.
    pub diff: Option<RgbImage>,
}

/// Measure each word's ink width on the scan crop: the horizontal ink
/// extent of the word's box, falling back to the box width when the box
/// is off-crop or holds no ink. Depends only on the crop and the word
/// boxes — never on the font — so callers that verify the same line
/// against several fonts measure once and reuse the widths across calls
/// (redundancy F34). This is the single source of truth for the
/// computation; `verify_text_region` takes the widths as a required
/// parameter.
pub fn measure_word_widths(
    scan_crop: &GrayImage,
    words: &[TextRegion],
    crop_x: u32,
    crop_y: u32,
) -> Vec<u32> {
    let ink_thresh = crate::ocr::INK_THRESHOLD;
    let (crop_w, crop_h) = scan_crop.dimensions();
    words
        .iter()
        .map(|wr| {
            let x_off = wr.x.saturating_sub(crop_x);
            let y_off = wr.y.saturating_sub(crop_y);
            if x_off < crop_w && y_off < crop_h {
                let bw = wr.width.min(crop_w.saturating_sub(x_off));
                let bh = wr.height.min(crop_h.saturating_sub(y_off));
                if bw >= 1 && bh >= 1 {
                    let (il, ir) = crate::ocr::ink_horizontal_extent(
                        scan_crop, y_off, y_off + bh, x_off, x_off + bw, ink_thresh
                    );
                    let mw = ir.saturating_sub(il);
                    if mw >= 1 { mw } else { wr.width }
                } else { wr.width }
            } else { wr.width }
        })
        .collect()
}

/// Verify a vectorised text region by:
/// 1. Computing an aspect-ratio penalty (natural advance widths vs OCR bbox widths)
/// 2. Rendering with per-word width scaling (so ZNCC can compare glyph shapes)
/// 3. Computing windowed ZNCC on the width-scaled render
/// 4. Returning `zncc * aspect_penalty`
pub fn verify_text_region(
    scan_crop: &GrayImage,
    font_data: &[u8],
    // Path of the font file font_data was read from (F34): the cache key
    // for the FreeType face and shaped words. The path — not the font key —
    // identifies the bytes: two files can share a key (same name+version),
    // but one path is one file, and fonts never change mid-run.
    font_path: &std::path::Path,
    text: &str,
    words: &[TextRegion],
    crop_x: u32,
    crop_y: u32,
    overrides: Option<&[(char, u16)]>,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
    allow_liga: bool,
    allowed_lig: u8,
    audit_dir: Option<&std::path::Path>,
    bail_below: Option<f32>,
    override_em_px: Option<f32>,
    frac_width: Option<f32>,
    // Per-word ink widths from `measure_word_widths` on this same crop and
    // these same word boxes (F34): the widths depend only on the scan, not
    // the font, so callers verifying one line against several fonts measure
    // once and pass the result in. Required, not optional: every caller
    // measures (or reuses) explicitly, and the length must match `words`.
    measured_widths: &[u32],
) -> VerifyResult {
    let (w, h) = scan_crop.dimensions();

    if w < 3 || h < 3 {
        return VerifyResult { score: 0.0, dy: 0, render_ink: None, diff: None };
    }

    // Page-level Hough deskew already corrected the full page before we get
    // here, so no per-line rotation needed.
    //
    // Word boxes are already tight ink bboxes produced upstream:
    //   expand_words_to_ink -> fix_overlapping_words_by_ink -> trim_words_to_ink
    // in page_cache.rs.  Do NOT re-derive ink runs, split, or merge words here —
    // spacing and splitting are handled up front.  Use the boxes as-is with
    // uniform line scale (midpoint-derived override_em_px).
    // Compute word widths from our crop's ink measurement, not Tesseract's bbox.
    // Tesseract widths are approximate; our ink_horizontal_extent measures actual ink.
    // The widths arrive precomputed (see `measure_word_widths`): they depend
    // only on the scan crop and the word boxes, never on the font, so the
    // caller measures once per line and reuses them across verify calls (F34).
    assert_eq!(
        measured_widths.len(),
        words.len(),
        "measured_widths must cover every word"
    );
    let placements: Vec<WordPlacement> = words
        .iter()
        .zip(measured_widths.iter())
        .map(|(wr, &measured_w)| {
            let x_off = wr.x.saturating_sub(crop_x);
            let y_off = wr.y.saturating_sub(crop_y);
            WordPlacement {
                // Normalize the word text for this font (Mike Nov 6, 2026):
                // a lig font renders "fi" as the single U+FB01 glyph.
                // HarfBuzz cannot invent that ligature (Trebuchet has no
                // GSUB liga), so the text itself must carry the FB01
                // character before shaping. A nolig font keeps f, i
                // separate (allowed_lig == 0 → identity).
                text: crate::char_render::normalize_seq_for_lig_mask(
                    &wr.text.chars().collect::<Vec<char>>(), allowed_lig,
                ).into_iter().collect(),
                x_off,
                y_off,
                width: measured_w,
                height: wr.height,
            }
        })
        .collect();

    // Both scan and render use the word-union bbox — the union of all
    // final (post-processed) word bboxes.  This matches what the report
    // draws as the "final" cyan-dashed boxes.

    // If fractional (subpixel) line width is provided, use it to compute
    // override_em_px directly, bypassing the per-word median.
    // Normalize the full line text the same way as the word placements
    // (lig font: "fi" → FB01; nolig: unchanged) for the width match.
    let norm_text: String = crate::char_render::normalize_seq_for_lig_mask(
        &text.chars().collect::<Vec<char>>(), allowed_lig,
    ).into_iter().collect();
    let override_em_px = if let Some(fw) = frac_width {
        if fw.is_finite() && fw >= 4.0 && fw <= 5000.0 {
            if let Some(em) = crate::layout::width_matched_em_px_shaped(
                font_data, &norm_text, fw, variant_tag, variations, allow_liga,
            ) {
                Some(em)
            } else {
                override_em_px
            }
        } else {
            override_em_px
        }
    } else {
        override_em_px
    };

    let full_render = match render_via_freetype_scaled(
        font_data,
        font_path,
        &placements,
        w,
        h,
        1,
        overrides,
        variant_tag,
        variations,
        allow_liga,
        override_em_px,
    ) {
        Some(r) => r,
        None => return VerifyResult { score: 0.0, dy: 0, render_ink: None, diff: None },
    };

    // ZNCC scoring — no blur or contrast normalization needed,
    // ZNCC is inherently invariant to mean/contrast differences.
    let (score, dy) = crate::compare_rasters::zncc_windowed_best_vshift(
        scan_crop, &full_render, 12, bail_below,
    );

    // Hot path: match_lines callers only need score/dy (fast_path and font
    // selection). render_ink + colored diff (with Lanczos3 resize when dims
    // differ) are only needed for audit/visualization. Skip when audit_dir
    // is None – avoids two ink-extent scans + clone/resize per verify call.
    if audit_dir.is_none() {
        return VerifyResult { score, dy, render_ink: None, diff: None };
    }

    // Audit path (audit_dir Some): ink-crop render for display + colored diff.
    // Same threshold as scan box trim (single source of truth).
    let ink_threshold = crate::ocr::INK_THRESHOLD;
    let (rw, rh) = full_render.dimensions();
    // same functions as scan trim: `< threshold` is ink.
    // scan uses trim_words_to_ink -> ink_horizontal_extent; we use identical pair for render.
    let (r_top, r_bot) = crate::ocr::ink_vertical_extent(&full_render, 0, rw, 0, rh, ink_threshold);
    let (r_left, r_right) = crate::ocr::ink_horizontal_extent(&full_render, 0, rh, 0, rw, ink_threshold);
    let ink_h = r_bot.saturating_sub(r_top);
    let ink_w = r_right.saturating_sub(r_left);
    let render_ink = if ink_h >= 3 && ink_w >= 3 {
        image::imageops::crop_imm(&full_render, r_left, r_top, ink_w, ink_h).to_image()
    } else if ink_h >= 3 {
        image::imageops::crop_imm(&full_render, 0, r_top, rw, ink_h).to_image()
    } else {
        full_render
    };

    let diff = compute_colored_diff(scan_crop, &render_ink);

    // Save audit images.
    let audit_path = audit_dir.unwrap();
    let _ = scan_crop.save(audit_path.join("ssim_scan.png"));
    let _ = render_ink.save(audit_path.join("ssim_render.png"));
    let _ = diff.save(audit_path.join("ssim_diff.png"));

    VerifyResult { score, dy, render_ink: Some(render_ink), diff: Some(diff) }
}

/// Colored diff: red = scan-only ink, blue = render-only ink, white = agreement.
/// Images are resized to match dimensions if needed (using the scan crop size).
pub fn compute_colored_diff(a: &GrayImage, b: &GrayImage) -> RgbImage {
    let (aw, ah) = a.dimensions();
    let b_resized = if b.dimensions() != (aw, ah) {
        image::imageops::resize(b, aw, ah, image::imageops::FilterType::Lanczos3)
    } else {
        b.clone()
    };
    let ink_thresh: u8 = 180;
    let mut diff = RgbImage::new(aw, ah);
    // Raw-buffer version: no get_pixel/put_pixel.
    let aw_us = aw as usize;
    let ah_us = ah as usize;
    let a_raw = a.as_raw();
    let b_raw = b_resized.as_raw();
    let d_raw = diff.as_mut();
    for y in 0..ah_us {
        let row_base = y * aw_us;
        let d_row_base = row_base * 3;
        for x in 0..aw_us {
            let idx = row_base + x;
            let pa = a_raw[idx];
            let pb = b_raw[idx];
            let a_ink = pa < ink_thresh;
            let b_ink = pb < ink_thresh;
            let (r, g, bcol) = match (a_ink, b_ink) {
                (true, false) => (220, 30, 30),
                (false, true) => (30, 80, 220),
                _ => (255, 255, 255),
            };
            let di = d_row_base + x * 3;
            d_raw[di] = r;
            d_raw[di + 1] = g;
            d_raw[di + 2] = bcol;
        }
    }
    diff
}

// ---------------------------------------------------------------------------
// Word-by-word renderer — height-matched scale, natural advance width
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Word-by-word renderer — width-matched scale, natural height
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Verify caches (F34): verify_text_region runs up to 3x per line with the
// same font, and each call used to re-parse faces and re-shape every word.
// All caches are thread-local: match_lines runs lines on a rayon pool, and
// neither FreeType faces nor the shaping cache are shared across threads.
// ---------------------------------------------------------------------------

// Variation coordinates as (tag, f32 bits): f32 has no Hash/Eq, so cache
// keys carry the bit pattern. Two variation sets with equal bits shape and
// render identically.
fn variation_bits(variations: Option<&[([u8; 4], f32)]>) -> Vec<([u8; 4], u32)> {
    match variations {
        Some(vars) => vars.iter().map(|(tag, v)| (*tag, v.to_bits())).collect(),
        None => Vec::new(),
    }
}

// FreeType face per (font path, variations), unbounded like the rustybuzz
// FACE_CACHE in unprint-fonts/shape.rs. Keyed by path, not font key: the
// path identifies the bytes (two files can share a key). A cached face is
// re-sized on every use — size is per-call state; variations are in the key
// so they are applied once at creation.
// No 'static library is needed: Face<Vec<u8>> owns the font bytes, and
// Face::from_raw takes its own C-level library reference, so the face
// keeps the underlying FT_Library alive after the creating Library drops
// (Face::drop releases that reference). Faces are used in place via
// get_mut — never cloned, since a clone would share the same FT_Face
// and its mutable size/glyph-slot state.
type FtFaceKey = (std::path::PathBuf, Vec<([u8; 4], u32)>);
thread_local! {
    static FT_FACE_CACHE: RefCell<HashMap<FtFaceKey, unprint_fonts::freetype::Face<Vec<u8>>>> =
        RefCell::new(HashMap::new());
}

// Shaped words per (font path, variations, text, variant tag, allow_liga).
// Shaping outputs are font-unit values, independent of render size, so the
// em never enters the key. FIFO-capped (not LRU: promotion would scan the
// order deque per hit; insertion order is enough for the per-page working
// set, and eviction just re-shapes). Entries are small (a few short Vecs).
const SHAPE_CACHE_CAP: usize = 4096;
type ShapeKey = (std::path::PathBuf, Vec<([u8; 4], u32)>, String, String, bool);
thread_local! {
    static SHAPE_CACHE: RefCell<HashMap<ShapeKey, crate::layout::ShapedWord>> =
        RefCell::new(HashMap::new());
    static SHAPE_CACHE_ORDER: RefCell<VecDeque<ShapeKey>> = RefCell::new(VecDeque::new());
}

fn shape_word_cached(
    face: &unprint_fonts::rustybuzz::Face,
    features: &[unprint_fonts::rustybuzz::Feature],
    text: &str,
    allow_liga: bool,
    key: &ShapeKey,
) -> Option<crate::layout::ShapedWord> {
    if let Some(shaped) = SHAPE_CACHE.with(|cache| cache.borrow().get(key).cloned()) {
        return Some(shaped);
    }
    let shaped = crate::layout::shape_word(face, features, text, allow_liga)?;
    SHAPE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        SHAPE_CACHE_ORDER.with(|order| {
            let mut order = order.borrow_mut();
            while cache.len() >= SHAPE_CACHE_CAP {
                match order.pop_front() {
                    Some(old) => {
                        cache.remove(&old);
                    }
                    None => break,
                }
            }
            cache.insert(key.clone(), shaped.clone());
            order.push_back(key.clone());
        });
    });
    Some(shaped)
}

fn render_via_freetype_scaled(
    font_data: &[u8],
    font_path: &std::path::Path,
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    render_scale: u32,
    _overrides: Option<&[(char, u16)]>,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
    allow_liga: bool,
    override_em_px: Option<f32>,
) -> Option<GrayImage> {
    // NOTE: ab_glyph fallback disabled.  width_matched_em_px (ab_glyph) does
    // not compensate for sidebearings, so it systematically underestimates font
    // size (~3% too small).  The shaped path (rustybuzz + FreeType) handles
    // sidebearing correction and should be the only render path.
    render_via_freetype(
        font_data,
        font_path,
        words,
        canvas_w,
        canvas_h,
        render_scale,
        variant_tag,
        variations,
        allow_liga,
        override_em_px,
    )
}

/// Render text using rustybuzz (OT shaping) + FreeType (rasterisation).
/// Subpixel glyph positioning, no intermediate PDF, no subprocess.
fn render_via_freetype(
    font_data: &[u8],
    font_path: &std::path::Path,
    words: &[WordPlacement],
    canvas_w: u32,
    canvas_h: u32,
    render_scale: u32,
    variant_tag: &str,
    variations: Option<&[([u8; 4], f32)]>,
    allow_liga: bool,
    override_em_px: Option<f32>,
) -> Option<GrayImage> {
    // Compute font size from ab_glyph (consistent with coarse scoring).
    // Compute per-word em_px using rustybuzz (shaped advances with
    // sidebearing correction).  No ab_glyph fallback — it lacks sidebearing
    // compensation and systematically underestimates font size by ~3%.
    let mut font_ref = FontRef::try_from_slice(font_data).ok()?;
    // Apply variable-font axis coordinates
    if let Some(vars) = variations {
        use unprint_fonts::ab_glyph::VariableFont;
        for (tag, val) in vars {
            font_ref.set_variation(tag, *val);
        }
    }
    // One rustybuzz face for this call (F34): width_matched_em_px_shaped
    // used to parse a fresh face per word for the em, and the render parsed
    // another. Shaping is a pure function of (face, features, text), so one
    // face serves every word here.
    let mut buzz_face = unprint_fonts::rustybuzz::Face::from_slice(font_data, 0)?;
    if let Some(vars) = variations {
        for (tag, val) in vars {
            let t = unprint_fonts::ttf_parser::Tag::from_bytes(tag);
            buzz_face.set_variation(t, *val);
        }
    }
    let ot_features = crate::layout::ot_features(variant_tag);
    let var_bits = variation_bits(variations);
    // Shape each word once (F34): previously each word was shaped once for
    // the em (inside width_matched_em_px_shaped) and again for the render.
    // The shaped word is reused for both, and cached across calls.
    let shaped_words: Vec<Option<crate::layout::ShapedWord>> = words
        .iter()
        .map(|w| {
            if w.text.is_empty() || w.width < 1 {
                None
            } else {
                let key = (
                    font_path.to_path_buf(),
                    var_bits.clone(),
                    w.text.clone(),
                    variant_tag.to_string(),
                    allow_liga,
                );
                shape_word_cached(&buzz_face, &ot_features, &w.text, allow_liga, &key)
            }
        })
        .collect();
    // Font size: prefer midpoint-derived override (scale = obs_span/pred_span * upem)
    // which reuses the exact scale calculation from geometry scoring.
    // If no override is provided, fall back to width-matched median (legacy).
    let line_em_px: f32 = match override_em_px {
        Some(ov) if ov.is_finite() && (4.0..=500.0).contains(&ov) => ov,
        // Invalid or absent override — fall through to width-matched.
        // (Was two copy-pasted filter_map blocks; the branches were identical.)
        _ => {
            let mut ems: Vec<f32> = words
                .iter()
                .zip(shaped_words.iter())
                .filter_map(|(w, s)| s.as_ref()?.ink_matched_em_px(w.width as f32))
                .collect();
            if ems.is_empty() {
                return None;
            }
            ems.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ems[ems.len() / 2]
        }
    };

    // Pad the render canvas so italic overshoot / wide terminal strokes
    // aren't clipped.  After rendering, we trim back to the scan width.
    let render_pad: u32 = 20;
    let render_w = (canvas_w + render_pad) * render_scale;
    let _final_w = canvas_w * render_scale;  // target width after trim
    let render_em = line_em_px * render_scale as f32;

    // The caller's canvas_h is the OCR-expanded bbox — sized to the scan's
    // ink extent.  The rendered font's metric height (ascent − descent) may
    // exceed that, so use whichever is larger to avoid vertical clipping.
    // Add vertical padding so glyphs whose descenders exceed the font's
    // declared descent (e.g. Q tail, swash g) aren't clipped.
    let vert_pad: u32 = 10 * render_scale;
    let sf2 = font_ref.as_scaled(PxScale::from(render_em));
    let ink_h2 = sf2.ascent() - sf2.descent();
    let min_render_h = canvas_h * render_scale;
    let render_h = min_render_h.max(ink_h2.ceil() as u32) + vert_pad;

    // Baseline centred in the (possibly taller) canvas
    let baseline_y = ((render_h as f32 - ink_h2) / 2.0 + sf2.ascent()) as f64;

    let units_per_em = buzz_face.units_per_em() as f64;
    let px_per_unit = render_em as f64 / units_per_em;

    // FreeType face: cached per (font path, variation bits) (F34) for
    // STATIC fonts — a face parse per verify call was the redundant setup
    // this finding removes. The face is used in place via get_mut — never
    // cloned, since a clone would share the same FT_Face and its mutable
    // size/glyph-slot state. A thread-local Library creates faces on cache
    // miss; the face keeps the library alive through its own C-level
    // reference.
    //
    // VARIABLE fonts (variations is Some) do NOT use the cache: they get a
    // fresh face per call, exactly as the original code did. Reusing a
    // variable-font face across verify calls renders differently — BAP 840
    // moved 14 lines (all IBMPlexSans winners, max sim delta 0.0126) while
    // the lob stayed bit-identical. The face carries mutable variation /
    // hinter state that a fresh face does not, and re-applying the variation
    // coordinates after set_char_size on a reused face does not restore the
    // original render. Fresh-face-per-call is the known-good behavior here;
    // the shaping cache and shared buzz face still apply.
    thread_local! {
        static FT_LIB: RefCell<Option<unprint_fonts::freetype::Library>> = RefCell::new(None);
    }
    // Fresh face for variable fonts, created up front so the render below
    // can use one `ft_face` binding for both paths. Face creation failure
    // fails the render, as the original code did (`.ok()?`).
    let mut fresh_face: Option<unprint_fonts::freetype::Face<Vec<u8>>> = None;
    if variations.is_some() {
        fresh_face = Some(FT_LIB.with(|lib_cell| -> Option<unprint_fonts::freetype::Face<Vec<u8>>> {
            let mut borrow = lib_cell.borrow_mut();
            if borrow.is_none() {
                *borrow = unprint_fonts::freetype::Library::init().ok();
            }
            let lib = borrow.as_ref()?;
            lib.new_memory_face2(font_data.to_vec(), 0).ok()
        })?);
    }
    let ft_result: Option<GrayImage> = FT_FACE_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let ft_face: &mut unprint_fonts::freetype::Face<Vec<u8>> = if let Some(f) = fresh_face.as_mut() {
            f
        } else {
            let key: FtFaceKey = (font_path.to_path_buf(), var_bits.clone());
            if !cache.contains_key(&key) {
                let face = FT_LIB.with(|lib_cell| -> Option<unprint_fonts::freetype::Face<Vec<u8>>> {
                    let mut borrow = lib_cell.borrow_mut();
                    if borrow.is_none() {
                        *borrow = unprint_fonts::freetype::Library::init().ok();
                    }
                    let lib = borrow.as_ref()?;
                    lib.new_memory_face2(font_data.to_vec(), 0).ok()
                })?;
                cache.insert(key.clone(), face);
            }
            cache.get_mut(&key)?
        };
        // render_em is in pixels. Use 72 DPI so points==pixels (pixels-as-points).
        // This is intentional, not a hack: we need pixel-exact rasterization
        // matching the scan's pixel grid. Using 300 DPI with pt size causes
        // FreeType to use a different internal scale, introducing subpixel
        // differences that regress ZNCC (measured: -9 hits, -0.0082 ZNCC).
        // Size is per-call state: reset on every use, since a cached face
        // keeps the previous call's size.
        let size_26_6 = (render_em as f64 * 64.0) as isize;
        ft_face.set_char_size(size_26_6, size_26_6, 72, 72).ok()?;

        // Variations MUST be applied AFTER set_char_size, as the original
        // code did. Applying them before sizing (e.g. once at face
        // creation) renders variable fonts differently: lob p1 L9 (winner
        // IBMPlexSans-300Italic, a variable font) sim moved 0.730139 ->
        // 0.73023695 with variations-before-size, and back to 0.730139
        // with variations-after-size. (Only the fresh-face path above ever
        // has variations; cached faces are static fonts.)
        if let Some(vars) = variations {
            set_ft_variations(ft_face, font_data, vars);
        }

    let mut canvas = GrayImage::from_pixel(render_w, render_h, Luma([255u8]));
    {
        // Raw-buffer blit: hoist width and mutable slice for alpha blending.
        let cw_us = render_w as usize;
        let canvas_raw = canvas.as_mut();

        for (wi, word) in words.iter().enumerate() {
        if word.text.is_empty() || word.width < 1 {
            continue;
        }

        // Shaped once per call above (F34) — ligature choice propagated
        // from segmentation winner.
        let shaped = match shaped_words[wi].as_ref() {
            Some(s) => s,
            None => continue,
        };

        // Walk glyphs, accumulating pen position in subpixel floats
        // Start at ink edge: subtract first glyph's LSB (in pixels) so
        // rendered ink aligns with the OCR bbox edge.
        let lsb_px = shaped.first_lsb_fu * px_per_unit;
        let mut pen_x = word.x_off as f64 * render_scale as f64 - lsb_px;
        let pen_y = baseline_y;

        for i in 0..shaped.glyph_ids.len() {
            let glyph_id = shaped.glyph_ids[i];
            let x_offset = shaped.x_offsets[i] as f64 * px_per_unit;
            let y_offset = shaped.y_offsets[i] as f64 * px_per_unit;

            // Load glyph in FreeType (hinting enabled to match PDF rendering)
            ft_face.load_glyph(glyph_id, unprint_fonts::freetype::face::LoadFlag::RENDER).ok()?;
            let glyph = ft_face.glyph();
            let bitmap = glyph.bitmap();
            let bmp_w = bitmap.width() as usize;
            let bmp_h = bitmap.rows() as usize;
            let bmp_buf = bitmap.buffer();
            let bmp_pitch = bitmap.pitch().unsigned_abs() as usize;

            if bmp_w == 0 || bmp_h == 0 || bmp_buf.is_empty() {
                pen_x += shaped.x_advances[i] as f64 * px_per_unit;
                continue;
            }

            // Glyph bitmap origin: (pen_x + x_offset + bitmap_left, pen_y - y_offset - bitmap_top)
            let blit_x = (pen_x + x_offset + glyph.bitmap_left() as f64).round() as i32;
            let blit_y = (pen_y - y_offset - glyph.bitmap_top() as f64).round() as i32;

            // Blit the glyph bitmap onto the canvas - raw buffer alpha blend.
            for row in 0..bmp_h {
                for col in 0..bmp_w {
                    let cx = blit_x + col as i32;
                    let cy = blit_y + row as i32;
                    if cx < 0 || cy < 0 || cx >= render_w as i32 || cy >= render_h as i32 {
                        continue;
                    }
                    let alpha = bmp_buf[row * bmp_pitch + col] as f32 / 255.0;
                    if alpha < 0.01 {
                        continue;
                    }
                    let idx = cy as usize * cw_us + cx as usize;
                    let existing = canvas_raw[idx] as f32;
                    let blended = existing * (1.0 - alpha);
                    canvas_raw[idx] = blended as u8;
                }
            }

            pen_x += shaped.x_advances[i] as f64 * px_per_unit;
        }
    }
    } // end raw blit scope, release canvas_raw borrow

    // Trim to ink using shared canonical threshold — do NOT scale to fill w+20.
    // The +20 padding is safety margin for rasterization only. The correct scale
    // comes from override_em_px (midpoint-derived) or width_matched_em_px,
    // not from forcing ink to fill the padded width. We keep whatever ink exists
    // and crop to its bounds, preserving overhang that landed in padding.
    let ink_thresh = crate::ocr::INK_THRESHOLD;
    let (rend_ink_left, rend_ink_right) =
        crate::ocr::ink_horizontal_extent(&canvas, 0, canvas.height(), 0, canvas.width(), ink_thresh);
    let (rend_ink_top, rend_ink_bottom) =
        crate::ocr::ink_vertical_extent(&canvas, 0, canvas.width(), 0, canvas.height(), ink_thresh);
    let ink_w = rend_ink_right.saturating_sub(rend_ink_left);
    let ink_h = rend_ink_bottom.saturating_sub(rend_ink_top);
    if ink_w >= 3 && ink_h >= 3 {
        canvas = image::imageops::crop_imm(&canvas, rend_ink_left, rend_ink_top, ink_w, ink_h).to_image();
    } else {
        // Fallback: if ink measurement failed, keep at least the scan width
        // but don't force fill — preserve white padding on right if no ink there.
        let keep_w = rend_ink_right.max(canvas_w).min(canvas.width());
        let keep_h = rend_ink_bottom.max(canvas.height()).min(canvas.height());
        let kh = if keep_h < 3 { canvas.height() } else { keep_h };
        canvas = image::imageops::crop_imm(&canvas, 0, 0, keep_w, kh).to_image();
    }

    // Downsample from render resolution to 1x (keep expanded height if canvas was grown)
    if render_scale > 1 {
        let out_h = render_h / render_scale;
        Some(image::imageops::resize(&canvas, canvas_w, out_h, image::imageops::FilterType::Lanczos3))
    } else {
        Some(canvas)
    }
    }); // end FT_FACE_CACHE.with

    ft_result
}

// ---------------------------------------------------------------------------
// Public: render a line for visual comparison in the miss report
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
