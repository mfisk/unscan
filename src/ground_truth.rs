//! Ground-truth extraction from a vector PDF for miss-only audit gating.
//!
//! When `--audit-vector <path>` is given alongside `--audit`, unscan parses the vector
//! PDF upfront and builds a per-page spatial index of text spans with their font
//! names.  During per-line processing, each matched font is immediately compared
//! against ground truth: hits skip expensive audit I/O (crop PNGs, fontmap
//! per-char distances, font ref glyphs); misses get full audit detail.

use lopdf::Document;
use std::collections::HashMap;
use std::path::Path;

/// A text span from the vector PDF with its font name and bounding box.
#[derive(Debug, Clone)]
pub struct VectorSpan {
    pub font_name: String,
    /// Bounding box in PDF points: (x0, y0, x1, y1).
    pub bbox: [f32; 4],
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
        "bold", "italic", "oblique",
        "regular", "medium", "light", "thin",
        "semibold", "extrabold", "demibold",
        "condensed", "semicondensed", "expanded",
        "book", "heavy", "black", "demi",
        "roman", "normal",
        "display", "caption", "subhead", "smtext",
        "400", "400i", "500", "600", "700", "800",
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

/// Check whether two font names refer to the same font family.
/// Mirrors char-misses.py `fonts_match` + `fonts_match_broad`.
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

// ── PDF parsing ─────────────────────────────────────────────────────────────

/// Resolve the /BaseFont name for a font resource name on a given page.
fn resolve_font_name(doc: &Document, page_id: lopdf::ObjectId, resource_name: &[u8]) -> Option<String> {
    // Walk the page's Resources → Font → resource_name → BaseFont
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

    // Get BaseFont name
    let base_font = font_dict.get(b"BaseFont").ok()?;
    match base_font {
        lopdf::Object::Name(name) => Some(String::from_utf8_lossy(name).to_string()),
        lopdf::Object::Reference(r) => {
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
/// We track the text state machine (BT/ET, Tf, Td, Tm, Tj, TJ) and record
/// a span whenever text is shown.  Bounding boxes are approximate (we use
/// font size for height and estimate width from character count × 0.5 × size)
/// but this is sufficient for spatial matching against unscan's line bboxes.
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

    // Build font resource name → BaseFont name map for this page.
    let mut font_map: HashMap<Vec<u8>, String> = HashMap::new();
    // Pre-populate from Resources/Font if available.
    if let Ok(page_dict) = doc.get_dictionary(page_id) {
        if let Ok(res_obj) = page_dict.get(b"Resources") {
            if let Ok((_, res)) = doc.dereference(res_obj) {
                if let Ok(res_dict) = res.as_dict() {
                    if let Ok(font_obj) = res_dict.get(b"Font") {
                        if let Ok((_, fonts)) = doc.dereference(font_obj) {
                            if let Ok(fonts_dict) = fonts.as_dict() {
                                for (name, _) in fonts_dict.iter() {
                                    if let Some(bf) = resolve_font_name(doc, page_id, name) {
                                        font_map.insert(name.clone(), bf);
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
    let mut font_size: f32 = 12.0;
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
                        if let Some(bf) = font_map.get(name_bytes) {
                            current_font = bf.clone();
                        } else {
                            current_font = String::from_utf8_lossy(name_bytes).to_string();
                        }
                    }
                    font_size = as_f32(&op.operands[1]).unwrap_or(12.0);
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
                    // We skip spacing for our purposes
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
                    let w = n_chars as f32 * font_size.abs() * 0.5 * ctm[0].abs().max(0.1);
                    spans.push(VectorSpan {
                        font_name: current_font.clone(),
                        bbox: [x, y, x + w, y + h],
                    });
                }
            }
            "TJ" => {
                let n_chars = text_length(&op.operands);
                if n_chars > 0 && !current_font.is_empty() {
                    let (x, y) = transform_point(tm[4], tm[5], &ctm);
                    let h = font_size.abs() * ctm[3].abs().max(0.1);
                    let w = n_chars as f32 * font_size.abs() * 0.5 * ctm[0].abs().max(0.1);
                    spans.push(VectorSpan {
                        font_name: current_font.clone(),
                        bbox: [x, y, x + w, y + h],
                    });
                }
            }

            _ => {}
        }
    }

    // Flip Y coordinates from PDF space (origin bottom-left) to display space
    // (origin top-left) so they match the pixel coordinates unscan uses.
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

// ── Public API ──────────────────────────────────────────────────────────────

impl GroundTruth {
    /// Parse a vector PDF and extract all text spans with font names.
    pub fn load(path: &Path) -> Result<Self, String> {
        let doc = Document::load(path).map_err(|e| format!("failed to load vector PDF: {}", e))?;
        let page_count = doc.get_pages().len();
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

        eprintln!("[ground_truth] loaded {} pages, {} total spans",
            page_count,
            pages.values().map(|v| v.len()).sum::<usize>());

        Ok(GroundTruth { pages })
    }

    /// Look up the ground-truth font for a given audit bbox (in pixels at the
    /// given DPI).  Returns the font name of the best-overlapping span, or None
    /// if no span overlaps.
    pub fn lookup_font(&self, page: usize, bbox_px: &[f32; 4], dpi: u32) -> Option<&str> {
        let scale = dpi as f32 / 72.0;
        // Convert pixel bbox to PDF points.
        let px0 = bbox_px[0] / scale;
        let py0 = bbox_px[1] / scale;
        let px1 = bbox_px[2] / scale;
        let py1 = bbox_px[3] / scale;

        let spans = self.pages.get(&page)?;

        let mut best_font: Option<&str> = None;
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
                    best_font = Some(&span.font_name);
                }
            }
        }

        // Fallback: if no overlap, find nearest span by vertical center distance
        if best_font.is_none() {
            let cy = (py0 + py1) / 2.0;
            let mut best_dist = f32::MAX;
            for span in spans {
                let span_cy = (span.bbox[1] + span.bbox[3]) / 2.0;
                let dist = (cy - span_cy).abs();
                if dist < best_dist && dist < (py1 - py0).max(5.0) {
                    best_dist = dist;
                    best_font = Some(&span.font_name);
                }
            }
        }

        best_font
    }

    /// Check whether a matched font is correct for the given position.
    /// Returns true if it's a hit (correct match), false if it's a miss.
    pub fn is_hit(&self, page: usize, bbox_px: &[f32; 4], dpi: u32, matched_font: &str) -> bool {
        match self.lookup_font(page, bbox_px, dpi) {
            Some(actual) => fonts_match(matched_font, actual),
            None => true, // no ground truth available → assume hit (don't penalize)
        }
    }
}
