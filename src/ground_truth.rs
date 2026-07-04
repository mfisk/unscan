//! Ground-truth extraction from a vector PDF for miss-only audit gating.
//!
//! When `--audit <path>` is given, unprint parses the vector
//! PDF upfront and builds a per-page spatial index of text spans with their font
//! names.  During per-line processing, each matched font is immediately compared
//! against ground truth: hits skip expensive audit I/O (crop PNGs, per-observation
//! distances, font ref glyphs); misses get full audit detail.

use lopdf::Document;
use std::collections::HashMap;
use std::path::Path;

/// A text span from the vector PDF with its font name and bounding box.
#[derive(Debug, Clone)]
pub struct VectorSpan {
    pub font_name: String,
    /// Bounding box in PDF points: (x0, y0, x1, y1).
    pub bbox: [f32; 4],
    /// Effective font size in PDF points (font_size × CTM y-scale).
    pub font_size_pt: f32,
    /// Raw text content extracted from Tj/TJ operands.
    pub text: String,
}

/// Ground truth for the entire document, keyed by 1-based page number.
pub struct GroundTruth {
    pub pages: HashMap<usize, Vec<VectorSpan>>,
}

// ── Font name matching ──────────────────────────────────────────────────────

/// Font alias map for canonical comparison.
fn font_aliases() -> HashMap<&'static str, &'static str> {
    let pairs: &[(&str, &str)] = &[
        ("arial", "helvetica"), ("arialmt", "helvetica"),
        ("nimbussans", "helvetica"), ("helvetica", "helvetica"),
        ("freesans", "helvetica"),
        ("texgyreheros", "helvetica"), ("texgyreheroscn", "helvetica"),
        ("timesnewroman", "times"), ("timesnewromanps", "times"),
        ("timesroman", "times"), ("nimbusroman", "times"),
        ("tinos", "times"), ("freeserif", "times"),
        ("texgyretermes", "times"),
        ("freeserifitalic", "times"), ("freeserifbold", "times"),
        ("freeserifbolditalic", "times"),
        ("p052", "times"), ("c059", "times"),
        ("couriernew", "courier"), ("couriernewps", "courier"),
        ("nimbusmonops", "courier"), ("freemono", "courier"),
        ("texgyrecursor", "courier"),
        ("carlito", "calibri"), ("caladea", "cambria"),
        ("sourcesanspro", "sourcesans"), ("sourcesans3", "sourcesans"),
        ("sourcesans", "sourcesans"),
        ("sourceserif4", "sourceserif"), ("sourceserif4subhead", "sourceserif"),
        ("sourceserif4smtext", "sourceserif"), ("sourceserif4caption", "sourceserif"),
        ("sourceserif4display", "sourceserif"),
        ("prestigeelite", "prestigeelite"), ("prestigeelitestd", "prestigeelite"),
        ("prestigeelitenormal", "prestigeelite"),
    ];
    pairs.iter().cloned().collect()
}

/// Strip everything but lowercase alphanumeric.
fn alphanum(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Strip weight/style suffixes to get the base family name.
fn base_family(name: &str) -> String {
    // Strip subset prefix (AAAAAA+FamilyName) BEFORE alphanumeric normalization
    let stripped = if let Some(pos) = name.find('+') {
        &name[pos + 1..]
    } else {
        name
    };
    let mut n = alphanum(stripped);
    let suffixes = [
        "mt", "ps",
        "bolditalic", "semibolditalic", "mediumitalic",
        "lightitalic", "thinitalic",
        "boldit", "semiboldit", "mediumit",
        "lightit", "thinit", "extraboldit",
        "blackit", "heavyit", "extralightit",
        "bold", "italic", "oblique",
        "regular", "medium", "light", "thin",
        "semibold", "extrabold", "demibold",
        "condensed", "semicondensed", "expanded",
        "book", "heavy", "black", "demi",
        "extralight",
        "roman", "normal",
        "display", "caption", "subhead", "smtext",
        "it", // short-form italic (SourceSerif4-It, SourceSans3-BoldIt, etc.)
        "400", "400i", "500", "600", "700", "800",
        "bd", "bi",  // MS TTC filenames: Arialbd, Arialbi, Timesbd, Courbi
    ];
    let mut changed = true;
    while changed {
        changed = false;
        for suf in &suffixes {
            if n.len() > suf.len() && n.ends_with(suf) {
                n.truncate(n.len() - suf.len());
                changed = true;
                break;
            }
        }
    }
    n
}

/// Canonicalize a font name through alias resolution.
fn canon(name: &str) -> String {
    let aliases = font_aliases();
    // Strip subset prefix before normalization
    let stripped = if let Some(pos) = name.find('+') { &name[pos + 1..] } else { name };
    let an = alphanum(stripped);
    if let Some(&c) = aliases.get(an.as_str()) {
        return c.to_string();
    }
    let bf = base_family(name);
    if let Some(&c) = aliases.get(bf.as_str()) {
        return c.to_string();
    }
    // Prefix matching
    let mut best: Option<&str> = None;
    let mut best_len = 0;
    for (&ak, &av) in &aliases {
        if bf.starts_with(ak) && ak.len() > best_len {
            best = Some(av);
            best_len = ak.len();
        } else if ak.starts_with(&*bf) && bf.len() > best_len {
            best = Some(av);
            best_len = bf.len();
        }
    }
    if let Some(b) = best {
        return b.to_string();
    }
    bf
}

/// Normalize a font key (may be a path like /usr/share/fonts/Foo.ttf|0).
fn normalize_font_key(name: &str) -> String {
    let mut n = name.rsplit('/').next().unwrap_or(name).to_string();
    for ext in &[".ttf", ".otf", ".TTF", ".OTF"] {
        if n.ends_with(ext) {
            n.truncate(n.len() - ext.len());
        }
    }
    if let Some(pos) = n.find('|') {
        n.truncate(pos);
    }
    if let Some(pos) = n.find('[') {
        n.truncate(pos);
    }
    alphanum(&n)
}

/// Strip subset prefix (e.g. "AAAAAA+FontName" → "FontName").
fn strip_subset_prefix(name: &str) -> &str {
    if let Some(pos) = name.find('+') { &name[pos + 1..] } else { name }
}

/// Strip subset prefix, returning an owned String.
pub fn strip_subset_prefix_str(name: &str) -> String {
    strip_subset_prefix(name).to_string()
}

/// Check whether two font names refer to the same font family.
/// Handles subset prefixes, weight/style suffixes, and metric-compatible clones.
pub fn fonts_match(matched: &str, actual: &str) -> bool {
    let na = alphanum(strip_subset_prefix(matched));
    let nb = alphanum(strip_subset_prefix(actual));
    if na == nb {
        return true;
    }
    let ba = base_family(matched);
    let bb = base_family(actual);
    if ba == bb {
        return true;
    }
    if !ba.is_empty() && !bb.is_empty() && (ba.contains(&*bb) || bb.contains(&*ba)) {
        return true;
    }
    if canon(matched) == canon(actual) {
        return true;
    }
    // Also try normalize_font_key for path-based keys
    let nna = normalize_font_key(matched);
    let nnb = normalize_font_key(actual);
    if !nna.is_empty() && !nnb.is_empty() && (nna == nnb || nna.contains(&*nnb) || nnb.contains(&*nna)) {
        return true;
    }
    let cna = canon(&nna);
    let cnb = canon(&nnb);
    if cna == cnb {
        return true;
    }
    false
}

/// Extract style flags from a font name: (is_italic, is_bold).
fn font_style_flags(name: &str) -> (bool, bool) {
    let n = alphanum(strip_subset_prefix(name));
    let is_italic = n.contains("italic") || n.ends_with("it") ||
                    n.contains("oblique") || n.ends_with("400i") ||
                    n.ends_with("bi") || n.ends_with("z"); // MS TTC: Georgiaz = bold-italic
    let is_bold = n.contains("bold") || n.contains("black") ||
                  n.contains("heavy") || n.contains("semibold") ||
                  n.contains("extrabold") || n.contains("demibold") ||
                  n.ends_with("600") || n.ends_with("700") ||
                  n.ends_with("800") || n.ends_with("900") ||
                  n.ends_with("bd") || n.ends_with("bi") ||
                  n.ends_with("z");  // MS TTC filenames: Arialbd, Arialbi, Georgiaz
    (is_italic, is_bold)
}

/// Strict font matching: same family AND same style (italic/bold) AND same variant tag.
/// DEPRECATED: use exact == comparison after canonicalize_names() instead.
/// Kept temporarily for reference during migration to canonical lookup.
#[allow(dead_code)]
pub fn fonts_match_strict(matched: &str, actual: &str) -> bool {
    // Split off variant tags: "FontName|hist" or "FontName [hist]"
    let (matched_base, matched_var) = split_variant(matched);
    let (actual_base, actual_var) = split_variant(actual);

    // Variant tags must agree, UNLESS the actual (GT) font has no variant —
    // PDF font dictionaries don't encode OT feature state, so an empty
    // variant on the GT side means "unknown", not "none".
    if matched_var != actual_var && !actual_var.is_empty() {
        return false;
    }

    let na = alphanum(strip_subset_prefix(matched_base));
    let nb = alphanum(strip_subset_prefix(actual_base));
    if na == nb {
        return true;
    }
    if !fonts_match(matched_base, actual_base) {
        return false;
    }
    // Family matches — now check that italic/bold flags agree
    let (ia, ba) = font_style_flags(matched_base);
    let (ib, bb) = font_style_flags(actual_base);
    ia == ib && ba == bb
}

/// Split a font name into (base, variant_tag).
/// Handles both "Name|tag" (font_key format) and "Name [tag]" (family_name format).
fn split_variant(name: &str) -> (&str, &str) {
    // Try pipe first: "FontName|hist"
    if let Some((base, var)) = name.split_once('|') {
        return (base, var);
    }
    // Try bracket: "FontName [hist]"
    if name.ends_with(']') {
        if let Some(open) = name.rfind(" [") {
            let var = &name[open + 2..name.len() - 1];
            let base = &name[..open];
            return (base, var);
        }
    }
    (name, "")
}

/// Public access to split_variant for report.rs.
pub fn split_variant_pub(name: &str) -> (&str, &str) {
    split_variant(name)
}

// ── Font metrics from PDF /Widths ───────────────────────────────────────────

/// Per-byte advance widths read from the PDF font dictionary's /Widths array.
/// Widths are in thousandths of a text-space unit (standard PDF convention).
struct PdfFontWidths {
    first_char: u32,
    /// widths[byte_value - first_char] in 1/1000 text-space units.
    widths: Vec<f32>,
    /// Average width for bytes outside the FirstChar..LastChar range.
    default_width: f32,
}

impl PdfFontWidths {
    /// Get width for a byte value in 1/1000 text-space units.
    fn width_for_byte(&self, b: u8) -> f32 {
        let idx = b as u32;
        if idx >= self.first_char && ((idx - self.first_char) as usize) < self.widths.len() {
            let w = self.widths[(idx - self.first_char) as usize];
            if w > 0.0 { w } else { self.default_width }
        } else {
            self.default_width
        }
    }
}

/// Extract /Widths, /FirstChar, /LastChar from a PDF font dictionary.
fn extract_pdf_font_widths(doc: &Document, font_dict: &lopdf::Dictionary) -> Option<PdfFontWidths> {
    let first_char = match font_dict.get(b"FirstChar") {
        Ok(obj) => {
            let resolved = doc.dereference(obj).ok().map(|(_, o)| o).unwrap_or(obj);
            as_f32(resolved).unwrap_or(0.0) as u32
        }
        Err(_) => return None,
    };

    let widths_obj = font_dict.get(b"Widths").ok()?;
    let widths_obj = doc.dereference(widths_obj).ok().map(|(_, o)| o).unwrap_or(widths_obj);
    let widths_arr = widths_obj.as_array().ok()?;

    let widths: Vec<f32> = widths_arr.iter()
        .map(|o| {
            let o2 = doc.dereference(o).ok().map(|(_, o)| o).unwrap_or(o);
            as_f32(o2).unwrap_or(0.0)
        })
        .collect();

    let total: f32 = widths.iter().filter(|&&w| w > 0.0).sum();
    let count = widths.iter().filter(|&&w| w > 0.0).count();
    let default_width = if count > 0 { total / count as f32 } else { 500.0 };

    Some(PdfFontWidths { first_char, widths, default_width })
}

/// Compute the width of text in a Tj/TJ operand list using PDF /Widths.
/// `word_spacing` is the current Tw value in text-space units.
/// Returns width in PDF points.
fn text_width_from_pdf(
    operands: &[lopdf::Object],
    fw: &PdfFontWidths,
    font_size: f32,
    ctm_x_scale: f32,
    word_spacing: f32,
) -> f32 {
    let mut width_thousandths = 0.0f32;
    let mut n_spaces = 0u32;

    for op in operands {
        match op {
            lopdf::Object::String(bytes, _) => {
                for &b in bytes.iter() {
                    width_thousandths += fw.width_for_byte(b);
                    if b == b' ' { n_spaces += 1; }
                }
            }
            lopdf::Object::Array(arr) => {
                for item in arr {
                    match item {
                        lopdf::Object::String(bytes, _) => {
                            for &b in bytes.iter() {
                                width_thousandths += fw.width_for_byte(b);
                                if b == b' ' { n_spaces += 1; }
                            }
                        }
                        // TJ kerning: value in thousandths of text space unit,
                        // positive = move left (reduce width).
                        lopdf::Object::Integer(n) => {
                            width_thousandths -= *n as f32;
                        }
                        lopdf::Object::Real(n) => {
                            width_thousandths -= *n as f32;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    let char_width = width_thousandths / 1000.0 * font_size.abs() * ctm_x_scale;
    // Tw is in text-space units (already sized), NOT thousandths.
    // PDF spec §9.4.4: tx = ((w0/1000)*Tfs + Tc + Tw) * Th
    let space_extra = n_spaces as f32 * word_spacing * ctm_x_scale;
    char_width + space_extra
}

/// Fallback width estimate when no /Widths available.
fn text_width_estimate(operands: &[lopdf::Object], font_size: f32, ctm_x_scale: f32) -> f32 {
    let n = text_length(operands);
    n as f32 * font_size.abs() * 0.5 * ctm_x_scale
}

// ── PDF parsing ─────────────────────────────────────────────────────────────

/// Resolve the canonical font name for a font resource on a given page.
/// Prefers /UnprintCanonical (our annotation with the unambiguous canonical
/// name) over /BaseFont (raw PS name from the font file).
fn resolve_font_name(doc: &Document, page_id: lopdf::ObjectId, resource_name: &[u8]) -> Option<String> {
    // Walk the page's Resources → Font → resource_name → font dict
    let page_dict = doc.get_dictionary(page_id).ok()?;

    // Get Resources dict (may be direct or indirect)
    let resources = match page_dict.get(b"Resources") {
        Ok(obj) => doc.dereference(obj).ok().map(|(_, o)| o),
        Err(_) => return None,
    }?;
    let resources_dict = resources.as_dict().ok()?;

    // Get Font dict
    let fonts_obj = match resources_dict.get(b"Font") {
        Ok(obj) => doc.dereference(obj).ok().map(|(_, o)| o),
        Err(_) => return None,
    }?;
    let fonts_dict = fonts_obj.as_dict().ok()?;

    // Get specific font entry
    let font_obj = match fonts_dict.get(resource_name) {
        Ok(obj) => doc.dereference(obj).ok().map(|(_, o)| o),
        Err(_) => return None,
    }?;
    let font_dict = font_obj.as_dict().ok()?;

    // Prefer /UnprintCanonical — our annotation with the unambiguous name.
    if let Ok(canonical) = font_dict.get(b"UnprintCanonical") {
        let name = match canonical {
            lopdf::Object::String(bytes, _) => Some(String::from_utf8_lossy(bytes).to_string()),
            lopdf::Object::Name(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
            lopdf::Object::Reference(_) => {
                if let Ok((_, obj)) = doc.dereference(canonical) {
                    match obj {
                        lopdf::Object::String(bytes, _) => Some(String::from_utf8_lossy(bytes).to_string()),
                        lopdf::Object::Name(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(n) = name {
            return Some(n);
        }
    }

    // Fall back to /BaseFont (raw PS name).
    let base_font = font_dict.get(b"BaseFont").ok()?;
    match base_font {
        lopdf::Object::Name(name) => Some(String::from_utf8_lossy(name).to_string()),
        lopdf::Object::Reference(_) => {
            if let Ok((_, obj)) = doc.dereference(base_font) {
                if let lopdf::Object::Name(name) = obj {
                    Some(String::from_utf8_lossy(name).to_string())
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Parse a PDF content stream into text spans with font names and positions.
///
/// Uses /Widths from the PDF font dictionaries for accurate span widths,
/// plus Tw (word spacing) tracking.  Falls back to character_count × 0.5 × font_size
/// when /Widths is not available for a font.
fn extract_page_spans(
    doc: &Document,
    page_id: lopdf::ObjectId,
    content_bytes: &[u8],
    page_height: f32,
) -> Vec<VectorSpan> {
    let content = match lopdf::content::Content::decode(content_bytes) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    // Build font resource name → (BaseFont name, PdfFontWidths) map for this page.
    let mut font_map: HashMap<Vec<u8>, String> = HashMap::new();
    let mut font_widths_map: HashMap<Vec<u8>, PdfFontWidths> = HashMap::new();
    let mut tounicode_map: HashMap<Vec<u8>, HashMap<u16, String>> = HashMap::new();
    // Pre-populate from Resources/Font if available.
    if let Ok(page_dict) = doc.get_dictionary(page_id) {
        if let Ok(res_obj) = page_dict.get(b"Resources") {
            if let Ok((_, res)) = doc.dereference(res_obj) {
                if let Ok(res_dict) = res.as_dict() {
                    if let Ok(font_obj) = res_dict.get(b"Font") {
                        if let Ok((_, fonts)) = doc.dereference(font_obj) {
                            if let Ok(fonts_dict) = fonts.as_dict() {
                                for (name, val) in fonts_dict.iter() {
                                    if let Some(bf) = resolve_font_name(doc, page_id, name) {
                                        font_map.insert(name.clone(), bf);
                                    }
                                    // Extract /Widths from font dict
                                    if let Ok((_, font_obj)) = doc.dereference(val) {
                                        if let Ok(fd) = font_obj.as_dict() {
                                            if let Some(fw) = extract_pdf_font_widths(doc, fd) {
                                                font_widths_map.insert(name.clone(), fw);
                                            }
                                            // Extract /ToUnicode CMap
                                            if let Some(tu) = extract_tounicode_map(doc, fd) {
                                                tounicode_map.insert(name.clone(), tu);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut spans = Vec::new();
    let mut current_font = String::new();
    let mut current_font_resource: Vec<u8> = Vec::new(); // resource name (e.g. F9+0)
    let mut font_size: f32 = 12.0;
    let mut word_spacing: f32 = 0.0; // Tw in text-space units (1/1000)
    // Text matrix [a, b, c, d, e, f] — (e, f) is the translation.
    let mut tm = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    // Current transformation matrix — we track it for coordinate transforms.
    let mut ctm_stack: Vec<[f32; 6]> = Vec::new();
    let mut ctm = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut leading: f32 = 0.0;

    for op in &content.operations {
        match op.operator.as_str() {
            // Graphics state
            "q" => ctm_stack.push(ctm),
            "Q" => { if let Some(prev) = ctm_stack.pop() { ctm = prev; } }
            "cm" => {
                if op.operands.len() >= 6 {
                    let m = extract_matrix(&op.operands);
                    ctm = multiply_matrix(&ctm, &m);
                }
            }

            // Text object
            "BT" => {
                tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            }

            // Set font
            "Tf" => {
                if op.operands.len() >= 2 {
                    if let Ok(name_bytes) = op.operands[0].as_name() {
                        current_font_resource = name_bytes.to_vec();
                        if let Some(bf) = font_map.get(name_bytes) {
                            current_font = bf.clone();
                        } else {
                            current_font = String::from_utf8_lossy(name_bytes).to_string();
                        }
                    }
                    font_size = as_f32(&op.operands[1]).unwrap_or(12.0);
                }
            }

            // Word spacing
            "Tw" => {
                if let Some(v) = op.operands.first().and_then(|o| as_f32(o)) {
                    word_spacing = v;
                }
            }

            // Text leading
            "TL" => {
                if let Some(v) = op.operands.first().and_then(|o| as_f32(o)) {
                    leading = v;
                }
            }

            // Text positioning
            "Td" => {
                if op.operands.len() >= 2 {
                    let tx = as_f32(&op.operands[0]).unwrap_or(0.0);
                    let ty = as_f32(&op.operands[1]).unwrap_or(0.0);
                    tm[4] += tx * tm[0] + ty * tm[2];
                    tm[5] += tx * tm[1] + ty * tm[3];
                }
            }
            "TD" => {
                if op.operands.len() >= 2 {
                    let tx = as_f32(&op.operands[0]).unwrap_or(0.0);
                    let ty = as_f32(&op.operands[1]).unwrap_or(0.0);
                    leading = -ty;
                    tm[4] += tx * tm[0] + ty * tm[2];
                    tm[5] += tx * tm[1] + ty * tm[3];
                }
            }
            "Tm" => {
                if op.operands.len() >= 6 {
                    tm = extract_matrix(&op.operands);
                }
            }
            "T*" => {
                tm[4] -= leading * tm[2];
                tm[5] -= leading * tm[3];
            }

            // Show text
            "Tj" | "'" | "\"" => {
                if op.operator == "\"" && op.operands.len() >= 3 {
                    // " sets word and char spacing then shows text
                    word_spacing = as_f32(&op.operands[0]).unwrap_or(0.0);
                    // char spacing (Tc) — tracked but not yet used for width
                }
                // Advance to next line for ' and "
                if op.operator == "'" || op.operator == "\"" {
                    tm[4] -= leading * tm[2];
                    tm[5] -= leading * tm[3];
                }
                let n_chars = text_length(&op.operands);
                if n_chars > 0 && !current_font.is_empty() {
                    let (x, y) = transform_point(tm[4], tm[5], &ctm);
                    let h = font_size.abs() * ctm[3].abs().max(0.1);
                    let ctm_x = ctm[0].abs().max(0.1);
                    let w = if let Some(fw) = font_widths_map.get(&current_font_resource) {
                        text_width_from_pdf(&op.operands, fw, font_size, ctm_x, word_spacing)
                    } else {
                        text_width_estimate(&op.operands, font_size, ctm_x)
                    };
                    spans.push(VectorSpan {
                        font_name: current_font.clone(),
                        bbox: [x, y, x + w, y + h],
                        font_size_pt: h,
                        text: text_content(&op.operands, tounicode_map.get(&current_font_resource)),
                    });
                }
            }
            "TJ" => {
                let n_chars = text_length(&op.operands);
                if n_chars > 0 && !current_font.is_empty() {
                    let (x, y) = transform_point(tm[4], tm[5], &ctm);
                    let h = font_size.abs() * ctm[3].abs().max(0.1);
                    let ctm_x = ctm[0].abs().max(0.1);
                    let w = if let Some(fw) = font_widths_map.get(&current_font_resource) {
                        text_width_from_pdf(&op.operands, fw, font_size, ctm_x, word_spacing)
                    } else {
                        text_width_estimate(&op.operands, font_size, ctm_x)
                    };
                    spans.push(VectorSpan {
                        font_name: current_font.clone(),
                        bbox: [x, y, x + w, y + h],
                        font_size_pt: h,
                        text: text_content(&op.operands, tounicode_map.get(&current_font_resource)),
                    });
                }
            }

            _ => {}
        }
    }

    // Flip Y coordinates from PDF space (origin bottom-left) to display space
    // (origin top-left) so they match the pixel coordinates unprint uses.
    for span in &mut spans {
        // PDF y is baseline from bottom; flip to top-down
        let pdf_y0 = span.bbox[1];
        let pdf_y1 = span.bbox[3];
        // In display coords: top = page_height - pdf_top, bottom = page_height - pdf_bottom
        span.bbox[1] = page_height - pdf_y1; // display top = page_height - pdf_top
        span.bbox[3] = page_height - pdf_y0; // display bottom = page_height - pdf_bottom
    }

    spans
}

fn extract_matrix(operands: &[lopdf::Object]) -> [f32; 6] {
    let mut m = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    for (i, op) in operands.iter().take(6).enumerate() {
        m[i] = as_f32(op).unwrap_or(if i == 0 || i == 3 { 1.0 } else { 0.0 });
    }
    m
}

fn multiply_matrix(a: &[f32; 6], b: &[f32; 6]) -> [f32; 6] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

fn transform_point(x: f32, y: f32, m: &[f32; 6]) -> (f32, f32) {
    (x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5])
}

fn as_f32(obj: &lopdf::Object) -> Option<f32> {
    match obj {
        lopdf::Object::Integer(i) => Some(*i as f32),
        lopdf::Object::Real(f) => Some(*f as f32),
        _ => None,
    }
}

/// Extract a /ToUnicode CMap from a font dictionary.
/// Returns a map from 2-byte glyph code to Unicode string.
fn extract_tounicode_map(doc: &Document, font_dict: &lopdf::Dictionary) -> Option<HashMap<u16, String>> {
    let tu_obj = font_dict.get(b"ToUnicode").ok()?;
    let tu_obj = doc.dereference(tu_obj).ok().map(|(_, o)| o).unwrap_or(tu_obj);

    // ToUnicode is a stream
    let stream = match tu_obj {
        lopdf::Object::Stream(ref s) => s,
        _ => return None,
    };
    let content = stream.decompressed_content().ok().unwrap_or_else(|| stream.content.clone());
    let text = String::from_utf8_lossy(&content);

    let mut map: HashMap<u16, String> = HashMap::new();

    // Parse beginbfchar / endbfchar blocks
    for section in text.split("beginbfchar") {
        if let Some(end) = section.find("endbfchar") {
            let block = &section[..end];
            for line in block.lines() {
                let line = line.trim();
                // Format: <XXXX> <YYYY> or <XXXX> <YYYYYYYY>
                let parts: Vec<&str> = line.split('<').filter(|s| s.contains('>')).collect();
                if parts.len() >= 2 {
                    let src = parts[0].split('>').next().unwrap_or("");
                    let dst = parts[1].split('>').next().unwrap_or("");
                    if let Ok(code) = u16::from_str_radix(src, 16) {
                        // dst can be multi-byte Unicode
                        let mut s = String::new();
                        let chars: Vec<char> = (0..dst.len() / 4).filter_map(|i| {
                            u16::from_str_radix(&dst[i*4..(i+1)*4], 16).ok()
                        }).filter_map(|cp| {
                            char::from_u32(cp as u32)
                        }).collect();
                        if chars.is_empty() {
                            // Try as a single 2-byte or 4-byte code
                            if let Ok(cp) = u32::from_str_radix(dst, 16) {
                                if let Some(c) = char::from_u32(cp) {
                                    s.push(c);
                                }
                            }
                        } else {
                            for c in chars { s.push(c); }
                        }
                        if !s.is_empty() {
                            map.insert(code, s);
                        }
                    }
                }
            }
        }
    }

    // Parse beginbfrange / endbfrange blocks
    for section in text.split("beginbfrange") {
        if let Some(end) = section.find("endbfrange") {
            let block = &section[..end];
            for line in block.lines() {
                let line = line.trim();
                let parts: Vec<&str> = line.split('<').filter(|s| s.contains('>')).collect();
                if parts.len() >= 3 {
                    let start_s = parts[0].split('>').next().unwrap_or("");
                    let end_s = parts[1].split('>').next().unwrap_or("");
                    let base_s = parts[2].split('>').next().unwrap_or("");
                    if let (Ok(start), Ok(end_code), Ok(base)) = (
                        u16::from_str_radix(start_s, 16),
                        u16::from_str_radix(end_s, 16),
                        u32::from_str_radix(base_s, 16),
                    ) {
                        for code in start..=end_code {
                            let cp = base + (code - start) as u32;
                            if let Some(c) = char::from_u32(cp) {
                                map.insert(code, c.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    if map.is_empty() { None } else { Some(map) }
}

/// Count the number of text characters in the operands of a text-showing op.
fn text_length(operands: &[lopdf::Object]) -> usize {
    let mut count = 0;
    for op in operands {
        match op {
            lopdf::Object::String(bytes, _) => count += bytes.len(),
            lopdf::Object::Array(arr) => {
                for item in arr {
                    if let lopdf::Object::String(bytes, _) = item {
                        count += bytes.len();
                    }
                }
            }
            _ => {}
        }
    }
    count
}

/// Extract the text content from Tj/TJ operands as a String.
/// If a ToUnicode CMap is available, use it to decode 2-byte glyph codes
/// to Unicode. Otherwise fall back to raw bytes as Latin-1.
fn text_content(operands: &[lopdf::Object], tounicode: Option<&HashMap<u16, String>>) -> String {
    let mut bytes_out = Vec::new();
    for op in operands {
        match op {
            lopdf::Object::String(bytes, _) => bytes_out.extend_from_slice(bytes),
            lopdf::Object::Array(arr) => {
                for item in arr {
                    if let lopdf::Object::String(bytes, _) = item {
                        bytes_out.extend_from_slice(bytes);
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(tu) = tounicode {
        // Decode 2-byte big-endian glyph codes via ToUnicode
        let mut result = String::new();
        let mut i = 0;
        while i + 1 < bytes_out.len() {
            let code = ((bytes_out[i] as u16) << 8) | (bytes_out[i + 1] as u16);
            if let Some(s) = tu.get(&code) {
                result.push_str(s);
            }
            i += 2;
        }
        // If ToUnicode produced nothing, fall back
        if result.is_empty() {
            String::from_utf8_lossy(&bytes_out).into_owned()
        } else {
            result
        }
    } else {
        String::from_utf8_lossy(&bytes_out).into_owned()
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

impl GroundTruth {
    /// Parse a vector PDF and extract all text spans with font names.
    /// Reads /Widths from the PDF font dictionaries for accurate span bounding
    /// boxes; otherwise estimates widths as character_count × 0.5 × font_size.
    pub fn load(path: &Path) -> Result<Self, String> {
        let doc = Document::load(path).map_err(|e| format!("failed to load vector PDF: {}", e))?;
        let _page_count = doc.get_pages().len();
        let mut pages: HashMap<usize, Vec<VectorSpan>> = HashMap::new();

        let page_ids: Vec<(u32, lopdf::ObjectId)> = doc.get_pages().into_iter().collect();

        for (page_num, page_id) in &page_ids {
            // Get page height from MediaBox for Y-coordinate flipping
            let page_height = doc.get_dictionary(*page_id).ok()
                .and_then(|d| d.get(b"MediaBox").ok())
                .and_then(|obj| {
                    if let Ok((_, resolved)) = doc.dereference(obj) {
                        resolved.as_array().ok().cloned()
                    } else {
                        obj.as_array().ok().cloned()
                    }
                })
                .and_then(|arr| arr.get(3).and_then(|o| as_f32(o)))
                .unwrap_or(792.0); // US Letter default

            // Get page content stream bytes
            let content_bytes = match doc.get_page_content(*page_id) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let spans = extract_page_spans(&doc, *page_id, &content_bytes, page_height);
            pages.insert(*page_num as usize, spans);
        }

        Ok(GroundTruth { pages })
    }

    /// Rewrite all span font names from raw PDF BaseFont to canonical names.
    /// For each span, strips the subset prefix, finds the matching catalog
    /// entry by raw_postscript_name, and replaces with the canonical
    /// (weight-explicit) postscript_name.  Names that don't match any catalog
    /// entry are left as-is (with subset prefix stripped).
    pub fn canonicalize_names(&mut self, catalog: &[crate::font_scan::FontEntry]) {
        for spans in self.pages.values_mut() {
            for span in spans.iter_mut() {
                let raw = strip_subset_prefix(&span.font_name).to_string();
                // If the name already matches a canonical postscript_name
                // (e.g., from /UnprintCanonical in an annotated PDF), keep it.
                if catalog.iter().any(|fe| fe.postscript_name == raw) {
                    span.font_name = raw;
                } else if let Some(fe) = catalog.iter().find(|fe| fe.raw_postscript_name == raw) {
                    // Wild PDF: map raw PS name → canonical via catalog.
                    span.font_name = fe.postscript_name.clone();
                } else {
                    span.font_name = raw;
                }
            }
        }
    }

    /// Look up the ground-truth font for a given audit bbox (in pixels at the
    /// given DPI).  Returns the font name of the best-overlapping span, or None
    /// if no span overlaps.
    pub fn lookup_font(&self, page: usize, bbox_px: &[f32; 4], dpi: u32) -> Option<&str> {
        self.lookup_span(page, bbox_px, dpi).map(|s| s.font_name.as_str())
    }

    /// Look up the ground-truth text for a given audit bbox (in pixels at the
    /// given DPI).  Returns the text of the best-overlapping span, or None.
    pub fn lookup_text(&self, page: usize, bbox_px: &[f32; 4], dpi: u32) -> Option<&str> {
        self.lookup_span(page, bbox_px, dpi).map(|s| s.text.as_str())
    }

    /// Find the best-overlapping VectorSpan for a given audit bbox.
    fn lookup_span(&self, page: usize, bbox_px: &[f32; 4], dpi: u32) -> Option<&VectorSpan> {
        let scale = dpi as f32 / 72.0;
        // Convert pixel bbox to PDF points.
        let px0 = bbox_px[0] / scale;
        let py0 = bbox_px[1] / scale;
        let px1 = bbox_px[2] / scale;
        let py1 = bbox_px[3] / scale;

        let spans = self.pages.get(&page)?;

        let mut best_span: Option<&VectorSpan> = None;
        let mut best_area: f32 = 0.0;

        for span in spans {
            let [sx0, sy0, sx1, sy1] = span.bbox;
            let ox0 = sx0.max(px0);
            let oy0 = sy0.max(py0);
            let ox1 = sx1.min(px1);
            let oy1 = sy1.min(py1);
            if ox0 < ox1 && oy0 < oy1 {
                let area = (ox1 - ox0) * (oy1 - oy0);
                if area > best_area {
                    best_area = area;
                    best_span = Some(span);
                }
            }
        }

        best_span
    }

    /// Look up the ground-truth font name and effective font size (in PDF
    /// points) for a given audit bbox (in pixels at the given DPI).
    pub fn lookup_font_and_size(&self, page: usize, bbox_px: &[f32; 4], dpi: u32) -> Option<(&str, f32)> {
        self.lookup_span(page, bbox_px, dpi).map(|s| (s.font_name.as_str(), s.font_size_pt))
    }

    /// Check whether a matched font is correct for the given position.
    /// `matched_ps` should be the PostScript name of the chosen font.
    /// Returns true if it's a hit (correct match), false if it's a miss.
    pub fn is_hit(&self, page: usize, bbox_px: &[f32; 4], dpi: u32, matched_ps: &str) -> bool {
        match self.lookup_font(page, bbox_px, dpi) {
            Some(actual) => {
                // After canonicalize_names(), span font names are already
                // canonical (weight-explicit).  Exact equality is correct.
                matched_ps == actual
            }
            None => true, // no ground truth available → assume hit (don't penalize)
        }
    }
}
