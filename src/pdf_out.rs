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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A text element that will (or won't) be placed as vector text.
#[derive(Debug, Clone)]
pub struct PlacedText {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub font_size_pt: f32,
    pub font_match: Option<FontMatchResult>,
    pub keep_raster: bool,
    pub color: Rgb,
    pub confidence: f32,
    /// Word-level bounding boxes from OCR for per-word positioning.
    pub words: Vec<WordBox>,
}

/// Minimal word bounding box for PDF placement.
#[derive(Debug, Clone)]
pub struct WordBox {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
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

pub fn generate_pdf(output_path: &Path, pages: &[PageContent], overlay: bool) -> Result<(), ScanTextError> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();

    // Dedup embedded fonts by path.
    let mut font_map: HashMap<PathBuf, (lopdf::ObjectId, String)> = HashMap::new();
    let mut font_counter = 0u32;

    for page in pages {
        for tr in &page.text_regions {
            if tr.keep_raster {
                continue;
            }
            if let Some(ref fm) = tr.font_match {
                if !font_map.contains_key(&fm.font_path) {
                    let name = format!("F{font_counter}");
                    font_counter += 1;
                    let id = embed_truetype_font(&mut doc, &fm.font_data, &name)?;
                    font_map.insert(fm.font_path.clone(), (id, name));
                }
            }
        }
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
        let mut page_font_names: HashMap<PathBuf, String> = HashMap::new();

        for tr in &page.text_regions {
            if tr.keep_raster { continue; }
            if let Some(ref fm) = tr.font_match {
                if let Some((obj_id, ref name)) = font_map.get(&fm.font_path) {
                    if !page_font_names.contains_key(&fm.font_path) {
                        font_resources.push((name.clone(), *obj_id));
                        page_font_names.insert(fm.font_path.clone(), name.clone());
                    }
                }
            }
        }

for tr in &page.text_regions {
            if tr.keep_raster || tr.text.trim().is_empty() {
                continue;
            }
            let (cr, cg, cb) = tr.color;

            let fname = if let Some(ref fm) = tr.font_match {
                page_font_names.get(&fm.font_path).cloned().unwrap_or_else(|| default_font_name.to_string())
            } else {
                default_font_name.to_string()
            };

            // Whole-line placement: height from line bbox, width via Tz scaling.
            // This preserves the font's natural kerning and word spacing.
            if !tr.words.is_empty() && tr.font_match.is_some() {
                let fm = tr.font_match.as_ref().unwrap();
                let font_ok = ab_glyph::FontRef::try_from_slice(&fm.font_data).ok();

                // Font size from line bbox height mapped through ascent-descent.
                let (line_pt, h_scale) = if let Some(ref f) = font_ok {
                    use ab_glyph::{Font, PxScale, ScaleFont};
                    let ref_h = 100.0f32;
                    let sf_ref = f.as_scaled(PxScale::from(ref_h));
                    let ref_ink = sf_ref.ascent() - sf_ref.descent();

                    let pt = if tr.height > 1.0 {
                        let em_px = ref_h * (tr.height / ref_ink);
                        em_px * 72.0 / page.dpi as f32
                    } else {
                        tr.font_size_pt
                    };

                    // Compute Tz: advance width of full line text at this size vs OCR line width.
                    let em_px = pt * page.dpi as f32 / 72.0;
                    let sf = f.as_scaled(PxScale::from(em_px));
                    let mut adv = 0.0f32;
                    let mut prev: Option<ab_glyph::GlyphId> = None;
                    for c in tr.text.chars() {
                        let gid = f.glyph_id(c);
                        if let Some(p) = prev { adv += sf.kern(p, gid); }
                        adv += sf.h_advance(gid);
                        prev = Some(gid);
                    }
                    let tz = if adv > 0.1 {
                        ((tr.width / adv) * 100.0).clamp(70.0, 150.0)
                    } else {
                        100.0
                    };
                    (pt, tz)
                } else {
                    (tr.font_size_pt, 100.0)
                };

                let pdf_x = page.px_to_pt_x(tr.x);
                // Position baseline: bottom of bbox + |descent| to shift up
                // from descender line to actual baseline.
                let descent_pt = if let Some(ref f) = font_ok {
                    use ab_glyph::{Font, PxScale, ScaleFont};
                    let em_px = line_pt * page.dpi as f32 / 72.0;
                    let sf = f.as_scaled(PxScale::from(em_px));
                    sf.descent() * 72.0 / page.dpi as f32  // negative value
                } else {
                    0.0
                };
                let pdf_y = page.px_to_pt_y(tr.y + tr.height) - descent_pt;

                if overlay {
                    ops.push(op("gs", &[Object::Name(b"GS_overlay".to_vec())]));
                }
                ops.push(op("BT", &[]));
                if overlay {
                    push_fill_color(&mut ops, 220, 0, 0);
                } else {
                    push_fill_color(&mut ops, cr, cg, cb);
                }
                ops.push(op("Tf", &[Object::Name(fname.clone().into_bytes()), real(line_pt as f64)]));
                if (h_scale - 100.0).abs() > 0.5 {
                    ops.push(op("Tz", &[real(h_scale as f64)]));
                }
                ops.push(op("Td", &[real(pdf_x as f64), real(pdf_y as f64)]));
                let encoded = encode_pdf_text(&tr.text);
                ops.push(op("Tj", &[Object::String(encoded, lopdf::StringFormat::Literal)]));
                ops.push(op("ET", &[]));
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
                let encoded = encode_pdf_text(&tr.text);
                ops.push(op("Tj", &[Object::String(encoded, lopdf::StringFormat::Literal)]));
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

fn embed_truetype_font(
    doc: &mut Document,
    font_data: &[u8],
    _name: &str,
) -> Result<lopdf::ObjectId, ScanTextError> {
    // Detect font type from magic bytes.
    let is_cff = font_data.len() >= 4 && &font_data[0..4] == b"OTTO";

    // Build font stream — CFF fonts need Subtype in the stream dict.
    let stream_dict = if is_cff {
        dictionary! {
            "Subtype" => "OpenType"
        }
    } else {
        dictionary! {
            "Length1" => Object::Integer(font_data.len() as i64)
        }
    };
    let font_stream = Stream::new(stream_dict, font_data.to_vec());
    let file_id = doc.add_object(font_stream);

    // Compute per-glyph widths for WinAnsiEncoding (chars 32–255)
    // in PDF width units (thousandths of em-square).
    let widths: Vec<Object> = if let Ok(f) = ab_glyph::FontRef::try_from_slice(font_data) {
        use ab_glyph::{Font, PxScale, ScaleFont};
        let sf = f.as_scaled(PxScale::from(1000.0));
        (32u8..=255)
            .map(|code| {
                let ch = code as char;
                let gid = f.glyph_id(ch);
                let w = sf.h_advance(gid);
                Object::Integer(w.round() as i64)
            })
            .collect()
    } else {
        // Fallback: 500 for all glyphs.
        (32..=255).map(|_| Object::Integer(500)).collect()
    };

    // Compute actual font metrics from the font data.
    let (ascent, descent, bbox) = if let Ok(f) = ab_glyph::FontRef::try_from_slice(font_data) {
        use ab_glyph::{Font, PxScale, ScaleFont};
        let sf = f.as_scaled(PxScale::from(1000.0));
        let a = sf.ascent().round() as i64;
        let d = sf.descent().round() as i64;
        (a, d, vec![Object::Integer(0), Object::Integer(d), Object::Integer(1000), Object::Integer(a)])
    } else {
        (800, -200, vec![Object::Integer(0), Object::Integer(-200), Object::Integer(1000), Object::Integer(800)])
    };

    // FontDescriptor — use FontFile3 for CFF, FontFile2 for TrueType.
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

    // Font dictionary.
    let font_subtype = if is_cff { "Type1" } else { "TrueType" };
    let mut font_dict = dictionary! {
        "Type" => "Font",
        "Subtype" => Object::Name(font_subtype.as_bytes().to_vec()),
        "BaseFont" => "EmbeddedFont",
        "Encoding" => "WinAnsiEncoding",
        "FontDescriptor" => Object::Reference(desc_id),
        "FirstChar" => Object::Integer(32),
        "LastChar" => Object::Integer(255),
        "Widths" => Object::Array(widths),
    };
    let dict_id = doc.add_object(font_dict);
    Ok(dict_id)
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

fn encode_pdf_text(text: &str) -> Vec<u8> {
    // lopdf's StringFormat::Literal handles escaping of (, ), and \.
    // We just need to map chars to bytes (latin-1 range).
    text.chars()
        .map(|ch| if (ch as u32) <= 255 { ch as u8 } else { b'?' })
        .collect()
}
