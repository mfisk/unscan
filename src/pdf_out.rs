//! PDF output — generate hybrid vector text + vector geometry + raster.
//!
//! Raster fragments preserve the **same encoding as the source PDF**.
//! If the source stores page images as JPEG (DCTDecode), raster fragments
//! are passed through or re-encoded as JPEG.  If the source uses FlateDecode,
//! we keep FlateDecode.  No unnecessary transcoding.

use crate::color::Rgb;
use crate::error::ScanTextError;
use crate::font_match::FontMatchResult;
use crate::geometry::{DetectedFill, DetectedLine};
use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Document, Object, Stream};
use log::debug;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A text element that will (or won't) be placed as vector text.
#[derive(Debug, Clone)]
pub struct PlacedText {
    pub text: String,
    pub x: f32,
    #[allow(dead_code)]
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub font_size_pt: f32,
    pub font_match: Option<FontMatchResult>,
    pub keep_raster: bool,
    pub color: Rgb,
    #[allow(dead_code)]
    pub confidence: f32,
    /// Word-level bounding boxes from OCR for per-word positioning.
    pub words: Vec<WordBox>,
}

/// Minimal word bounding box for PDF placement.
#[derive(Debug, Clone)]
pub struct WordBox {
    pub text: String,
    pub x: f32,
    #[allow(dead_code)]
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// If set by the smoothing pass, this overrides the per-word width-matched
    /// em_px calculation.  All words in a same-font run share the median size.
    pub smoothed_em_px: Option<f32>,
}

/// The encoding/filter used by an image stream in the source PDF.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageFilter {
    /// JPEG — `/DCTDecode`
    DCTDecode,
    /// Lossless zlib — `/FlateDecode`
    FlateDecode,
    /// No filter — raw uncompressed
    None,
    /// Something else we don't special-case
    Other(String),
}

/// Metadata + compressed bytes extracted from the source PDF's page image.
#[derive(Debug, Clone)]
pub struct SourceImageInfo {
    /// The raw (possibly compressed) stream bytes exactly as stored in the PDF.
    pub stream_bytes: Vec<u8>,
    /// Filter applied to the stream.
    pub filter: ImageFilter,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// PDF ColorSpace name (e.g. "DeviceRGB", "DeviceGray") or empty if
    /// the source used an indirect reference we couldn't resolve simply.
    pub color_space: String,
    /// Bits per component (typically 8).
    pub bits_per_component: u32,
}

/// A raster fragment to embed in the output PDF.
#[derive(Debug, Clone)]
pub struct RasterFragment {
    /// Raw decoded RGB pixels — used when we had to modify the image
    /// (erase vectorised regions).  Empty when `passthrough` is `Some`.
    pub raw_rgb: Vec<u8>,
    pub width_px: u32,
    pub height_px: u32,
    pub x_pt: f32,
    pub y_pt: f32,
    pub width_pt: f32,
    pub height_pt: f32,
    /// If set, embed these pre-compressed bytes directly instead of `raw_rgb`.
    /// This is used for full-page pass-through when nothing was vectorised.
    pub passthrough: Option<SourceImageInfo>,
}

/// Everything needed to render one page.
#[derive(Debug)]
pub struct PageContent {
    pub width_px: u32,
    pub height_px: u32,
    pub dpi: u32,
    pub text_regions: Vec<PlacedText>,
    pub raster_fragments: Vec<RasterFragment>,
    pub lines: Vec<DetectedLine>,
    pub fills: Vec<DetectedFill>,
    pub bg_color: Rgb,
}

impl PageContent {
    pub fn width_pt(&self) -> f32 {
        self.width_px as f32 * 72.0 / self.dpi as f32
    }
    pub fn height_pt(&self) -> f32 {
        self.height_px as f32 * 72.0 / self.dpi as f32
    }
    pub fn px_to_pt_x(&self, px: f32) -> f32 {
        px * 72.0 / self.dpi as f32
    }
    /// Convert image-space Y (top-left origin) to PDF Y (bottom-left origin).
    pub fn px_to_pt_y(&self, py: f32) -> f32 {
        self.height_pt() - (py * 72.0 / self.dpi as f32)
    }
    fn scale(&self) -> f32 {
        72.0 / self.dpi as f32
    }
}

// ---------------------------------------------------------------------------
// Fragment extraction
// ---------------------------------------------------------------------------

/// Extract a raster fragment from the image at original resolution.
pub fn extract_raster_fragment(
    img: &image::DynamicImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    dpi: u32,
    page_height_px: u32,
) -> Option<RasterFragment> {
    let (iw, ih) = (img.width(), img.height());
    let x = x.min(iw);
    let y = y.min(ih);
    let w = w.min(iw - x);
    let h = h.min(ih - y);
    if w < 2 || h < 2 {
        return None;
    }

    let cropped = img.crop_imm(x, y, w, h);
    let rgb = cropped.to_rgb8();
    let raw = rgb.into_raw();

    let scale = 72.0 / dpi as f32;
    Some(RasterFragment {
        raw_rgb: raw,
        width_px: w,
        height_px: h,
        x_pt: x as f32 * scale,
        y_pt: (page_height_px as f32 - (y + h) as f32) * scale,
        width_pt: w as f32 * scale,
        height_pt: h as f32 * scale,
        passthrough: None,
    })
}

// ---------------------------------------------------------------------------
// PDF generation
// ---------------------------------------------------------------------------

pub fn generate_pdf(output_path: &Path, pages: &[PageContent], overlay: bool, font_cache: &crate::font_cache::FontCache) -> Result<(), ScanTextError> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();

    // ── Collect unique chars per font key (path|variant_tag) ──────────
    let mut font_chars: HashMap<String, HashSet<char>> = HashMap::new();
    let mut font_data_map: HashMap<String, std::sync::Arc<Vec<u8>>> = HashMap::new();
    let mut font_overrides_map: HashMap<String, Option<Vec<(char, u16)>>> = HashMap::new();
    for page in pages {
        for tr in &page.text_regions {
            if tr.keep_raster { continue; }
            if let Some(ref fm) = tr.font_match {
                let chars = font_chars.entry(fm.font_key.clone()).or_default();
                // Collect from per-word text if available, else whole line
                if !tr.words.is_empty() {
                    for w in &tr.words {
                        chars.extend(w.text.chars());
                    }
                } else {
                    chars.extend(tr.text.chars());
                }
                font_data_map.entry(fm.font_key.clone())
                    .or_insert_with(|| {
                        font_cache.load(&fm.font_path)
                            .unwrap_or_else(|_| std::sync::Arc::new(Vec::new()))
                    });
                font_overrides_map.entry(fm.font_key.clone())
                    .or_insert_with(|| fm.glyph_overrides.clone());
            }
        }
    }

    // ── Embed subsetted fonts ────────────────────────────────────────
    // font_map: font_key → (object_id, pdf_name, char→CID mapping)
    let mut font_map: HashMap<String, (lopdf::ObjectId, String, HashMap<char, u16>)> = HashMap::new();
    let mut font_counter = 0u32;

    for (font_key, chars) in &font_chars {
        let name = format!("F{font_counter}");
        font_counter += 1;
        let font_data = &font_data_map[font_key];
        let overrides = font_overrides_map.get(font_key).and_then(|o| o.as_deref());
        let (id, cid_map) = embed_subsetted_font(&mut doc, font_data.as_slice(), &name, chars, overrides)?;
        font_map.insert(font_key.clone(), (id, name, cid_map));
    }

    let mut page_ids = Vec::new();

    for (pi, page) in pages.iter().enumerate() {
        let pw = page.width_pt();
        let ph = page.height_pt();
        let scale = page.scale();

        let mut ops: Vec<Operation> = Vec::new();

        // 1. Background fill
        let (br, bg, bb) = page.bg_color;
        push_fill_color(&mut ops, br, bg, bb);
        ops.push(op("re", &[real(0.0), real(0.0), real(pw as f64), real(ph as f64)]));
        ops.push(op("f", &[]));

        // 2. Vector fills
        for fill in &page.fills {
            let (fr, fg, fb) = fill.color;
            push_fill_color(&mut ops, fr, fg, fb);
            let fx = fill.x as f64 * scale as f64;
            let fy = (page.height_px as f64 - (fill.y + fill.height) as f64) * scale as f64;
            let fw = fill.width as f64 * scale as f64;
            let fh = fill.height as f64 * scale as f64;
            ops.push(op("re", &[real(fx), real(fy), real(fw), real(fh)]));
            ops.push(op("f", &[]));
        }

        // 3. Raster fragments — preserve source encoding
        let mut img_resources: Vec<(String, lopdf::ObjectId)> = Vec::new();

        for (fi, frag) in page.raster_fragments.iter().enumerate() {
            let img_name = format!("Im{pi}_{fi}");

            let img_obj_id = if let Some(ref pt) = frag.passthrough {
                // Pass through pre-compressed bytes from source PDF
                embed_passthrough_image(&mut doc, pt)?
            } else {
                // Fall back to raw RGB (modified image)
                embed_raw_rgb_image(&mut doc, &frag.raw_rgb, frag.width_px, frag.height_px)?
            };

            img_resources.push((img_name.clone(), img_obj_id));

            ops.push(op("q", &[]));
            ops.push(op("cm", &[
                real(frag.width_pt as f64),
                real(0.0),
                real(0.0),
                real(frag.height_pt as f64),
                real(frag.x_pt as f64),
                real(frag.y_pt as f64),
            ]));
            ops.push(op("Do", &[Object::Name(img_name.into_bytes())]));
            ops.push(op("Q", &[]));
        }

        // 4. Vector lines
        for line in &page.lines {
            let (lr, lg, lb) = line.color;
            push_stroke_color(&mut ops, lr, lg, lb);
            let lw = (line.thickness as f64 * scale as f64).max(0.5);
            ops.push(op("w", &[real(lw)]));
            let x1 = line.x1 as f64 * scale as f64;
            let y1 = (page.height_px as f64 - line.y1 as f64) * scale as f64;
            let x2 = line.x2 as f64 * scale as f64;
            let y2 = (page.height_px as f64 - line.y2 as f64) * scale as f64;
            ops.push(op("m", &[real(x1), real(y1)]));
            ops.push(op("l", &[real(x2), real(y2)]));
            ops.push(op("S", &[]));
        }

        // 5. Vector text
        let default_font_name = "Fdef";
        let mut font_resources: Vec<(String, lopdf::ObjectId)> = Vec::new();
        let mut page_font_names: HashMap<String, String> = HashMap::new();

        for tr in &page.text_regions {
            if tr.keep_raster { continue; }
            if let Some(ref fm) = tr.font_match {
                if let Some((obj_id, ref name, _)) = font_map.get(&fm.font_key) {
                    if !page_font_names.contains_key(&fm.font_key) {
                        font_resources.push((name.clone(), *obj_id));
                        page_font_names.insert(fm.font_key.clone(), name.clone());
                    }
                }
            }
        }

for tr in &page.text_regions {
            if tr.keep_raster || tr.text.trim().is_empty() {
                continue;
            }
            let (cr, cg, cb) = tr.color;

            let (fname, cid_map) = if let Some(ref fm) = tr.font_match {
                let name = page_font_names.get(&fm.font_key)
                    .cloned().unwrap_or_else(|| default_font_name.to_string());
                let cmap = font_map.get(&fm.font_key).map(|(_, _, m)| m);
                (name, cmap)
            } else {
                (default_font_name.to_string(), None)
            };

            // Per-word placement: each word positioned at its OCR x-offset,
            // width-matched independently. This avoids em-dash / special char
            // width differences stealing space from regular letters.
            log::debug!("pdf_out: '{}...' — {} words, font_match={}", 
                &tr.text[..tr.text.len().min(30)], tr.words.len(), tr.font_match.is_some());
            if !tr.words.is_empty() && tr.font_match.is_some() {
                let fm = tr.font_match.as_ref().unwrap();
                let font_bytes = font_cache.load(&fm.font_path).ok();
                let font_ok = font_bytes.as_ref().and_then(|b| ab_glyph::FontRef::try_from_slice(b.as_slice()).ok());

                if overlay {
                    ops.push(op("gs", &[Object::Name(b"GS_overlay".to_vec())]));
                }

                if let Some(ref f) = font_ok {
                    // Compute ONE baseline for the entire line — all words share it.
                    // Use height-based em sizing: the OCR bbox height determines the
                    // font size, preserving natural letter spacing.  Width-matching
                    // would shrink/stretch spacing to fit OCR bbox widths.
                    let line_em_px = {
                        use ab_glyph::{Font, PxScale, ScaleFont};
                        let sf = f.as_scaled(PxScale::from(1000.0));
                        let ink_h = sf.ascent() - sf.descent();
                        if ink_h > 0.1 {
                            Some(tr.height * 1000.0 / ink_h)
                        } else {
                            None
                        }
                    };

                    let (line_baseline_offset_pt, _) = if let Some(em) = line_em_px {
                        crate::layout::ink_centered_baseline_pt(f, em, tr.height, page.dpi as f32)
                    } else {
                        continue;
                    };

                    let dy_pt = fm.best_dy as f32 * 72.0 / page.dpi as f32;
                    let pdf_y = page.px_to_pt_y(tr.y) - line_baseline_offset_pt - dy_pt;

                    // Use the line-level em size for all words — no per-word
                    // width matching that would distort natural letter spacing.
                    let line_pt = line_em_px.unwrap() * 72.0 / page.dpi as f32;

                    for word in &tr.words {
                        if word.text.is_empty() || word.width < 1.0 {
                            continue;
                        }

                        let word_pt = line_pt;

                        // Word x position (word.x is absolute page coords)
                        let pdf_x = page.px_to_pt_x(word.x);

                        ops.push(op("BT", &[]));
                        if overlay {
                            push_fill_color(&mut ops, 220, 0, 0);
                        } else {
                            push_fill_color(&mut ops, cr, cg, cb);
                        }
                        ops.push(op("Tf", &[Object::Name(fname.clone().into_bytes()), real(word_pt as f64)]));
                        ops.push(op("Td", &[real(pdf_x as f64), real(pdf_y as f64)]));
                        let encoded = encode_text_for_font(&word.text, cid_map);
                        ops.push(op("Tj", &[Object::String(encoded, lopdf::StringFormat::Hexadecimal)]));
                        ops.push(op("ET", &[]));
                    }
                }
            } else {
                // Fallback: single Tj for the whole line.
                if overlay {
                    ops.push(op("gs", &[Object::Name(b"GS_overlay".to_vec())]));
                }
                ops.push(op("BT", &[]));
                if overlay {
                    push_fill_color(&mut ops, 220, 0, 0);
                } else {
                    push_fill_color(&mut ops, cr, cg, cb);
                }
                ops.push(op("Tf", &[Object::Name(fname.into_bytes()), real(tr.font_size_pt as f64)]));
                let pdf_x = page.px_to_pt_x(tr.x);
                let pdf_y = page.px_to_pt_y(tr.y + tr.height);
                ops.push(op("Td", &[real(pdf_x as f64), real(pdf_y as f64)]));
                let encoded = encode_text_for_font(&tr.text, cid_map);
                ops.push(op("Tj", &[Object::String(encoded, lopdf::StringFormat::Hexadecimal)]));
                ops.push(op("ET", &[]));
            }
        }

        // Build content stream
        let content = Content { operations: ops };
        let content_bytes = content
            .encode()
            .map_err(|e| ScanTextError::PdfGen(format!("content encode: {e}")))?;
        let content_id = doc.add_object(Stream::new(dictionary! {}, content_bytes));

        // Resource dictionary
        let mut font_dict = lopdf::Dictionary::new();
        for (name, obj_id) in &font_resources {
            font_dict.set(name.as_bytes(), Object::Reference(*obj_id));
        }
        let helvetica_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "Encoding" => "WinAnsiEncoding",
        });
        font_dict.set(default_font_name.as_bytes(), Object::Reference(helvetica_id));

        let mut xobject_dict = lopdf::Dictionary::new();
        for (name, obj_id) in &img_resources {
            xobject_dict.set(name.as_bytes(), Object::Reference(*obj_id));
        }

        let mut resources = dictionary! {
            "Font" => Object::Dictionary(font_dict),
            "XObject" => Object::Dictionary(xobject_dict),
        };

        // Add transparency graphics state for overlay mode.
        if overlay {
            let gs_dict = dictionary! {
                "Type" => "ExtGState",
                "ca" => Object::Real(0.5),   // fill opacity 50%
                "CA" => Object::Real(0.5),   // stroke opacity 50%
            };
            let gs_id = doc.add_object(gs_dict);
            let mut ext_gstate = lopdf::Dictionary::new();
            ext_gstate.set(b"GS_overlay", Object::Reference(gs_id));
            resources.set(b"ExtGState", Object::Dictionary(ext_gstate));
        }

        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => Object::Reference(pages_id),
            "MediaBox" => vec![real(0.0), real(0.0), real(pw as f64), real(ph as f64)],
            "Resources" => Object::Dictionary(resources),
            "Contents" => Object::Reference(content_id),
        });
        page_ids.push(page_id);
    }

    // Pages node
    let kids: Vec<Object> = page_ids.iter().map(|id| Object::Reference(*id)).collect();
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(page_ids.len() as i64),
            "Kids" => Object::Array(kids),
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    });
    doc.trailer.set("Root", Object::Reference(catalog_id));

    doc.compress();
    doc.save(output_path)
        .map_err(|e| ScanTextError::PdfGen(format!("save: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

/// Embed a subsetted font as a CID-keyed Type0 font in the PDF.
/// Returns (font_object_id, char→CID mapping).
fn embed_subsetted_font(
    doc: &mut Document,
    font_data: &[u8],
    _name: &str,
    used_chars: &HashSet<char>,
    overrides: Option<&[(char, u16)]>,
) -> Result<(lopdf::ObjectId, HashMap<char, u16>), ScanTextError> {
    use ab_glyph::{Font, ScaleFont};

    let ab_font = ab_glyph::FontRef::try_from_slice(font_data)
        .map_err(|e| ScanTextError::PdfGen(format!("parse font: {e}")))?;

    // Map chars → original glyph IDs (with variant overrides)
    let mut char_to_orig_gid: Vec<(char, u16)> = Vec::new();
    for &ch in used_chars {
        let gid = crate::char_index::resolve_glyph(&ab_font, ch, overrides);
        let gid_val = gid.0;
        if gid_val != 0 {
            char_to_orig_gid.push((ch, gid_val));
        }
    }
    // Always include .notdef (GID 0)
    char_to_orig_gid.sort_by_key(|&(_, gid)| gid);
    char_to_orig_gid.dedup_by_key(|e| e.1);

    // Build subset via subsetter crate
    let mut remapper = subsetter::GlyphRemapper::new();
    for &(_, gid) in &char_to_orig_gid {
        remapper.remap(gid);
    }

    let (embed_bytes, char_to_cid) = match subsetter::subset(font_data, 0, &remapper) {
        Ok(subsetted) => {
            // Build char → remapped CID map
            let mut cid_map = HashMap::new();
            for &(ch, orig_gid) in &char_to_orig_gid {
                if let Some(new_gid) = remapper.get(orig_gid) {
                    cid_map.insert(ch, new_gid);
                }
            }
            log::debug!("pdf_out: subsetted font {} → {} bytes ({} glyphs)",
                font_data.len(), subsetted.len(), cid_map.len());
            (subsetted, cid_map)
        }
        Err(e) => {
            // Fallback: embed full font, use original GIDs as CIDs
            log::warn!("pdf_out: font subsetting failed ({e}), embedding full font");
            let mut cid_map = HashMap::new();
            for &(ch, orig_gid) in &char_to_orig_gid {
                cid_map.insert(ch, orig_gid);
            }
            (font_data.to_vec(), cid_map)
        }
    };

    let is_cff = font_data.len() >= 4 && &font_data[0..4] == b"OTTO";

    // ── Font stream ──────────────────────────────────────────────────
    let stream_dict = if is_cff {
        dictionary! { "Subtype" => "CIDFontType0C" }
    } else {
        dictionary! { "Length1" => Object::Integer(embed_bytes.len() as i64) }
    };
    let font_stream = Stream::new(stream_dict, embed_bytes);
    let file_id = doc.add_object(font_stream);

    // ── Font metrics ─────────────────────────────────────────────────
    let sf = ab_font.as_scaled(ab_glyph::PxScale::from(1000.0));
    let ascent = sf.ascent().round() as i64;
    let descent = sf.descent().round() as i64;
    let bbox = vec![
        Object::Integer(0), Object::Integer(descent),
        Object::Integer(1000), Object::Integer(ascent),
    ];

    // ── CID widths array (W entry) ───────────────────────────────────
    // Format: [cid [w1 w2 ...]] for each CID
    let mut w_entries: Vec<Object> = Vec::new();
    let mut cid_widths: Vec<(u16, i64)> = char_to_cid.iter()
        .map(|(&ch, &cid)| {
            let gid = crate::char_index::resolve_glyph(&ab_font, ch, overrides);
            let w = sf.h_advance(gid).round() as i64;
            (cid, w)
        })
        .collect();
    cid_widths.sort_by_key(|&(cid, _)| cid);
    for &(cid, w) in &cid_widths {
        w_entries.push(Object::Integer(cid as i64));
        w_entries.push(Object::Array(vec![Object::Integer(w)]));
    }

    // ── FontDescriptor ───────────────────────────────────────────────
    let mut desc_dict = dictionary! {
        "Type" => "FontDescriptor",
        "FontName" => "EmbeddedFont",
        "Flags" => Object::Integer(32),
        "FontBBox" => Object::Array(bbox),
        "ItalicAngle" => Object::Integer(0),
        "Ascent" => Object::Integer(ascent),
        "Descent" => Object::Integer(descent),
        "CapHeight" => Object::Integer((ascent * 7 / 10).max(500)),
        "StemV" => Object::Integer(80),
    };
    if is_cff {
        desc_dict.set("FontFile3", Object::Reference(file_id));
    } else {
        desc_dict.set("FontFile2", Object::Reference(file_id));
    }
    let desc_id = doc.add_object(desc_dict);

    // ── CIDFont dictionary ───────────────────────────────────────────
    let cid_subtype = if is_cff { "CIDFontType0" } else { "CIDFontType2" };
    let cid_font_dict = dictionary! {
        "Type" => "Font",
        "Subtype" => Object::Name(cid_subtype.as_bytes().to_vec()),
        "BaseFont" => "EmbeddedFont",
        "CIDSystemInfo" => Object::Dictionary(dictionary! {
            "Registry" => Object::String(b"Adobe".to_vec(), lopdf::StringFormat::Literal),
            "Ordering" => Object::String(b"Identity".to_vec(), lopdf::StringFormat::Literal),
            "Supplement" => Object::Integer(0),
        }),
        "FontDescriptor" => Object::Reference(desc_id),
        "DW" => Object::Integer(1000),
        "W" => Object::Array(w_entries),
    };
    let cid_font_id = doc.add_object(cid_font_dict);

    // ── ToUnicode CMap (enables copy/paste of text) ──────────────────
    let tounicode_id = build_tounicode_cmap(doc, &char_to_cid);

    // ── Type0 (composite) font ───────────────────────────────────────
    let font_dict = dictionary! {
        "Type" => "Font",
        "Subtype" => "Type0",
        "BaseFont" => "EmbeddedFont",
        "Encoding" => "Identity-H",
        "DescendantFonts" => Object::Array(vec![Object::Reference(cid_font_id)]),
        "ToUnicode" => Object::Reference(tounicode_id),
    };
    let dict_id = doc.add_object(font_dict);
    Ok((dict_id, char_to_cid))
}

/// Build a ToUnicode CMap stream so PDF viewers can extract text.
fn build_tounicode_cmap(doc: &mut Document, char_to_cid: &HashMap<char, u16>) -> lopdf::ObjectId {
    let mut entries: Vec<(u16, char)> = char_to_cid.iter().map(|(&ch, &cid)| (cid, ch)).collect();
    entries.sort_by_key(|&(cid, _)| cid);

    let mut cmap = String::new();
    cmap.push_str("/CIDInit /ProcSet findresource begin\n");
    cmap.push_str("12 dict begin\n");
    cmap.push_str("begincmap\n");
    cmap.push_str("/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n");
    cmap.push_str("/CMapName /Adobe-Identity-UCS def\n");
    cmap.push_str("/CMapType 2 def\n");
    cmap.push_str("1 begincodespacerange\n");
    cmap.push_str("<0000> <FFFF>\n");
    cmap.push_str("endcodespacerange\n");

    // Write in chunks of 100 (PDF spec limit)
    for chunk in entries.chunks(100) {
        cmap.push_str(&format!("{} beginbfchar\n", chunk.len()));
        for &(cid, ch) in chunk {
            cmap.push_str(&format!("<{:04X}> <{:04X}>\n", cid, ch as u32));
        }
        cmap.push_str("endbfchar\n");
    }

    cmap.push_str("endcmap\n");
    cmap.push_str("CMapName currentdict /CMap defineresource pop\n");
    cmap.push_str("end\nend\n");

    doc.add_object(Stream::new(dictionary! {}, cmap.into_bytes()))
}

/// Encode text as 2-byte CID values using the char→CID map.
/// Falls back to WinAnsi encoding if no CID map is available (e.g. default Helvetica).
fn encode_text_for_font(text: &str, cid_map: Option<&HashMap<char, u16>>) -> Vec<u8> {
    match cid_map {
        Some(map) => {
            let mut out = Vec::with_capacity(text.len() * 2);
            for ch in text.chars() {
                let cid = map.get(&ch).copied().unwrap_or(0);
                out.push((cid >> 8) as u8);
                out.push((cid & 0xFF) as u8);
            }
            out
        }
        None => encode_pdf_text(text),
    }
}

/// Embed raw RGB pixel data as an XObject Image.
/// `doc.compress()` will FlateDecode it later (lossless).
fn embed_raw_rgb_image(
    doc: &mut Document,
    raw_rgb: &[u8],
    w: u32,
    h: u32,
) -> Result<lopdf::ObjectId, ScanTextError> {
    let stream = Stream::new(
        dictionary! {
            "Type" => "XObject",
            "Subtype" => "Image",
            "Width" => Object::Integer(w as i64),
            "Height" => Object::Integer(h as i64),
            "ColorSpace" => "DeviceRGB",
            "BitsPerComponent" => Object::Integer(8),
        },
        raw_rgb.to_vec(),
    );
    Ok(doc.add_object(stream))
}

/// Embed pre-compressed image data from the source PDF, preserving the
/// original encoding (JPEG / FlateDecode / raw).  The bytes are copied
/// verbatim — no decode/re-encode cycle.
fn embed_passthrough_image(
    doc: &mut Document,
    info: &SourceImageInfo,
) -> Result<lopdf::ObjectId, ScanTextError> {
    let cs_name = if info.color_space.is_empty() {
        "DeviceRGB"
    } else {
        &info.color_space
    };

    let mut dict = lopdf::Dictionary::new();
    dict.set("Type", Object::Name(b"XObject".to_vec()));
    dict.set("Subtype", Object::Name(b"Image".to_vec()));
    dict.set("Width", Object::Integer(info.width as i64));
    dict.set("Height", Object::Integer(info.height as i64));
    dict.set("ColorSpace", Object::Name(cs_name.as_bytes().to_vec()));
    dict.set(
        "BitsPerComponent",
        Object::Integer(info.bits_per_component as i64),
    );

    // Set the filter so PDF readers know how to decode the stream.
    match &info.filter {
        ImageFilter::DCTDecode => {
            dict.set("Filter", Object::Name(b"DCTDecode".to_vec()));
            debug!(
                "  embed_passthrough: JPEG pass-through {}×{} ({} bytes)",
                info.width,
                info.height,
                info.stream_bytes.len()
            );
        }
        ImageFilter::FlateDecode => {
            dict.set("Filter", Object::Name(b"FlateDecode".to_vec()));
            debug!(
                "  embed_passthrough: FlateDecode pass-through {}×{} ({} bytes)",
                info.width,
                info.height,
                info.stream_bytes.len()
            );
        }
        ImageFilter::None => {
            // No filter — raw bytes
            debug!(
                "  embed_passthrough: raw pass-through {}×{} ({} bytes)",
                info.width,
                info.height,
                info.stream_bytes.len()
            );
        }
        ImageFilter::Other(name) => {
            dict.set("Filter", Object::Name(name.as_bytes().to_vec()));
            debug!(
                "  embed_passthrough: {} pass-through {}×{} ({} bytes)",
                name,
                info.width,
                info.height,
                info.stream_bytes.len()
            );
        }
    }

    // Create stream with the already-compressed bytes.
    // IMPORTANT: we must NOT let lopdf re-compress this — the bytes are
    // already in the format described by Filter.
    let stream = Stream::new(dict, info.stream_bytes.clone());
    Ok(doc.add_object(stream))
}

// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

fn op(name: &str, operands: &[Object]) -> Operation {
    Operation::new(name, operands.to_vec())
}

fn real(v: f64) -> Object {
    Object::Real(v as f32)
}

fn push_fill_color(ops: &mut Vec<Operation>, r: u8, g: u8, b: u8) {
    ops.push(op("rg", &[
        real(r as f64 / 255.0),
        real(g as f64 / 255.0),
        real(b as f64 / 255.0),
    ]));
}

fn push_stroke_color(ops: &mut Vec<Operation>, r: u8, g: u8, b: u8) {
    ops.push(op("RG", &[
        real(r as f64 / 255.0),
        real(g as f64 / 255.0),
        real(b as f64 / 255.0),
    ]));
}

pub fn encode_pdf_text(text: &str) -> Vec<u8> {
    // Map Unicode chars to WinAnsiEncoding byte values.
    // WinAnsi is mostly latin-1 but bytes 0x80–0x9F map to specific Unicode chars.
    text.chars()
        .map(|ch| unicode_to_winansi(ch).unwrap_or(b'?'))
        .collect()
}

/// Map a Unicode code point to its WinAnsiEncoding byte, if it exists.
fn unicode_to_winansi(ch: char) -> Option<u8> {
    let cp = ch as u32;
    // ASCII + latin-1 supplement (U+00A0–U+00FF map 1:1)
    if cp <= 0x7F || (0xA0..=0xFF).contains(&cp) {
        return Some(cp as u8);
    }
    // WinAnsi 0x80–0x9F range — specific Unicode mappings
    match cp {
        0x20AC => Some(0x80), // €
        0x201A => Some(0x82), // ‚
        0x0192 => Some(0x83), // ƒ
        0x201E => Some(0x84), // „
        0x2026 => Some(0x85), // …
        0x2020 => Some(0x86), // †
        0x2021 => Some(0x87), // ‡
        0x02C6 => Some(0x88), // ˆ
        0x2030 => Some(0x89), // ‰
        0x0160 => Some(0x8A), // Š
        0x2039 => Some(0x8B), // ‹
        0x0152 => Some(0x8C), // Œ
        0x017D => Some(0x8E), // Ž
        0x2018 => Some(0x91), // '
        0x2019 => Some(0x92), // '
        0x201C => Some(0x93), // "
        0x201D => Some(0x94), // "
        0x2022 => Some(0x95), // •
        0x2013 => Some(0x96), // –  (en dash)
        0x2014 => Some(0x97), // —  (em dash)
        0x02DC => Some(0x98), // ˜
        0x2122 => Some(0x99), // ™
        0x0161 => Some(0x9A), // š
        0x203A => Some(0x9B), // ›
        0x0153 => Some(0x9C), // œ
        0x017E => Some(0x9E), // ž
        0x0178 => Some(0x9F), // Ÿ
        _ => None,
    }
}
