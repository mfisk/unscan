//! Font scanning — discover .ttf / .otf fonts on all standard system paths,
//! with aliasing for Microsoft and LaTeX font families.
//!
//! For each font file, the scanner creates a base `FontEntry` plus one
//! additional entry per OpenType feature that changes glyph shapes for
//! common Latin characters. This means a font like Source Serif 4 with
//! `onum`, `smcp`, `ss01`, and `ss02` support produces 5 catalog entries:
//! the default plus one per feature variant.
//!
//! During matching (`font_match.rs`), each variant entry carries a
//! `glyph_overrides` map so the renderer uses the correct glyph IDs.
//! SSIM comparison naturally picks the best-matching variant without
//! needing explicit figure-style or small-caps detection heuristics.
//!
//! The OT feature detection uses `rustybuzz` (a pure-Rust harfbuzz port)
//! to shape a Latin probe string with each feature enabled, then compares
//! the resulting glyph IDs against the default shaping. Only features
//! that produce at least one different glyph ID are emitted as variants.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use log::debug;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontClass {
    Serif,
    Sans,
    Mono,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct FontEntry {
    pub path: PathBuf,
    pub family_name: String,
    /// PostScript name (name ID 6) read from the font's name table.
    /// This is the exact string that appears as BaseFont in PDF dictionaries,
    /// so GT comparison can use direct equality instead of heuristics.
    pub postscript_name: String,
    pub is_bold: bool,
    #[allow(dead_code)]
    pub is_italic: bool,
    pub class: FontClass,
    pub data: Vec<u8>,
    /// True if the font uses old-style (text/ranging) figures where digits
    /// like 3, 5, 7, 9 have descenders below the baseline.
    pub oldstyle_figures: bool,
    /// OT feature tag this variant represents (empty string for default entry).
    pub variant_tag: String,
    /// For variant entries: maps characters to their feature-specific glyph IDs.
    /// Only characters whose glyph ID differs from default are included.
    /// None for the default entry (use normal cmap lookup).
    pub glyph_overrides: Option<Vec<(char, u16)>>,
}

impl FontEntry {
    /// Unique key for this font entry in the char index.
    /// Encodes path + variant tag so each weight/style/OT-variant gets its own slot.
    pub fn font_key(&self) -> String {
        let p = self.path.display().to_string();
        if self.variant_tag.is_empty() {
            p
        } else {
            format!("{}|{}", p, self.variant_tag)
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the full set of font search directories for the current platform,
/// plus any user-supplied extra dirs.
pub fn default_font_dirs(extra: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    // ── Linux ────────────────────────────────────────────────────────
    dirs.push("/usr/share/fonts".into());
    dirs.push("/usr/local/share/fonts".into());
    dirs.push("/usr/share/fonts/truetype/msttcorefonts".into());

    // TeX Live (OTF + TTF)
    dirs.push("/usr/share/texlive/texmf-dist/fonts/opentype".into());
    dirs.push("/usr/share/texlive/texmf-dist/fonts/truetype".into());
    dirs.push("/usr/share/texmf/fonts/opentype".into());
    dirs.push("/usr/share/texmf/fonts/truetype".into());

    // ── macOS ────────────────────────────────────────────────────────
    dirs.push("/Library/Fonts".into());
    dirs.push("/System/Library/Fonts".into());

    // ── Windows ──────────────────────────────────────────────────────
    dirs.push("C:\\Windows\\Fonts".into());

    // ── User-level dirs ──────────────────────────────────────────────
    if let Some(home) = std::env::var_os("HOME") {
        let h = PathBuf::from(home);
        dirs.push(h.join(".fonts"));
        dirs.push(h.join(".local/share/fonts"));
        // macOS user
        dirs.push(h.join("Library/Fonts"));
        // User TeX fonts
        dirs.push(h.join("texmf/fonts"));
    }

    // User-supplied extras
    for d in extra {
        dirs.push(d.clone());
    }

    dirs
}

/// Walk the given directories for .ttf / .otf files and return a catalogue.
pub fn scan_fonts(dirs: &[PathBuf]) -> Vec<FontEntry> {
    let aliases = build_alias_table();
    let mut fonts = Vec::new();
    let mut seen_paths: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    for dir in dirs {
        if !dir.exists() {
            continue;
        }
        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext != "ttf" && ext != "otf" {
                continue;
            }
            // Canonicalize to avoid duplicates from overlapping dir walks
            let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if !seen_paths.insert(canon) {
                continue;
            }
            if let Some(fe) = load_font_entry(path, &aliases) {
                let fig_label = if fe.oldstyle_figures { "OLDSTYLE" } else { "lining" };
                debug!("  Font: {} [{}] [{}] {}", fe.family_name, class_label(fe.class), fig_label, fe.path.display());

                // Detect ligature glyphs (liga + dlig)
                let ligatures = detect_ligature_glyphs(&fe.data);
                if !ligatures.is_empty() {
                    debug!("    + {} ligature glyphs", ligatures.len());
                }

                // Probe all OT features — emit a variant entry for each that changes glyphs
                let variants = detect_ot_variants(&fe.data);
                for (tag, overrides) in &variants {
                    // Merge ligature overrides into each variant
                    let mut combined = overrides.clone();
                    for &(lig_c, gid) in &ligatures {
                        if !combined.iter().any(|(c, _)| *c == lig_c) {
                            combined.push((lig_c, gid));
                        }
                    }
                    let var_entry = FontEntry {
                        path: fe.path.clone(),
                        family_name: format!("{} [{}]", fe.family_name, tag),
                        postscript_name: fe.postscript_name.clone(),
                        is_bold: fe.is_bold,
                        is_italic: fe.is_italic,
                        class: fe.class,
                        data: Vec::new(), // bytes not retained
                        oldstyle_figures: fe.oldstyle_figures,
                        variant_tag: tag.clone(),
                        glyph_overrides: Some(combined),
                    };
                    debug!("    + variant [{}]: {} glyph overrides", tag, overrides.len());
                    fonts.push(var_entry);
                }
                // Drop font bytes — metadata + path is all we keep.
                // Index build and matching load from path on demand.
                let mut fe = fe;
                fe.data = Vec::new();
                // Add ligature overrides to base entry
                if !ligatures.is_empty() {
                    let mut base_overrides = fe.glyph_overrides.take().unwrap_or_default();
                    base_overrides.extend(ligatures);
                    fe.glyph_overrides = Some(base_overrides);
                }
                fonts.push(fe);
            }
        }
    }

    fonts
}

// ---------------------------------------------------------------------------
// Alias table
// ---------------------------------------------------------------------------

struct Alias {
    family: &'static str,
    bold: bool,
    italic: bool,
}

fn build_alias_table() -> HashMap<String, Alias> {
    let mut m = HashMap::new();

    macro_rules! a {
        ($stem:expr, $fam:expr, $b:expr, $i:expr) => {
            m.insert($stem.to_string(), Alias { family: $fam, bold: $b, italic: $i });
        };
    }

    // ── Microsoft core fonts ─────────────────────────────────────────
    a!("arial",       "Arial", false, false);
    a!("arialbd",     "Arial", true,  false);
    a!("ariali",      "Arial", false, true);
    a!("arialbi",     "Arial", true,  true);
    a!("arial_bold",  "Arial", true,  false);
    a!("arialn",      "Arial Narrow", false, false);
    a!("arialnb",     "Arial Narrow", true,  false);
    a!("arialni",     "Arial Narrow", false, true);
    a!("arialnbi",    "Arial Narrow", true,  true);
    a!("ariblk",      "Arial Black", false, false);

    a!("times",       "Times New Roman", false, false);
    a!("timesbd",     "Times New Roman", true,  false);
    a!("timesi",      "Times New Roman", false, true);
    a!("timesbi",     "Times New Roman", true,  true);

    a!("cour",        "Courier New", false, false);
    a!("courbd",      "Courier New", true,  false);
    a!("couri",       "Courier New", false, true);
    a!("courbi",      "Courier New", true,  true);

    a!("calibri",     "Calibri", false, false);
    a!("calibrib",    "Calibri", true,  false);
    a!("calibrii",    "Calibri", false, true);
    a!("calibriz",    "Calibri", true,  true);
    a!("calibril",    "Calibri Light", false, false);
    a!("calibrili",   "Calibri Light", false, true);

    a!("cambria",     "Cambria", false, false);
    a!("cambriab",    "Cambria", true,  false);
    a!("cambriai",    "Cambria", false, true);
    a!("cambriaz",    "Cambria", true,  true);

    a!("verdana",     "Verdana", false, false);
    a!("verdanab",    "Verdana", true,  false);
    a!("verdanai",    "Verdana", false, true);
    a!("verdanaz",    "Verdana", true,  true);

    a!("tahoma",      "Tahoma", false, false);
    a!("tahomabd",    "Tahoma", true,  false);

    a!("georgia",     "Georgia", false, false);
    a!("georgiab",    "Georgia", true,  false);
    a!("georgiai",    "Georgia", false, true);
    a!("georgiaz",    "Georgia", true,  true);

    a!("trebuc",      "Trebuchet MS", false, false);
    a!("trebucbd",    "Trebuchet MS", true,  false);
    a!("trebucit",    "Trebuchet MS", false, true);
    a!("trebucbi",    "Trebuchet MS", true,  true);

    a!("comic",       "Comic Sans MS", false, false);
    a!("comicbd",     "Comic Sans MS", true,  false);
    a!("comici",      "Comic Sans MS", false, true);

    a!("consola",     "Consolas", false, false);
    a!("consolab",    "Consolas", true,  false);
    a!("consolai",    "Consolas", false, true);
    a!("consolaz",    "Consolas", true,  true);

    a!("segoeui",     "Segoe UI", false, false);
    a!("segoeuib",    "Segoe UI", true,  false);
    a!("segoeuii",    "Segoe UI", false, true);
    a!("segoeuiz",    "Segoe UI", true,  true);
    a!("seguisb",     "Segoe UI Semibold", true, false);

    a!("garamond",    "Garamond", false, false);
    a!("aptos",       "Aptos", false, false);
    a!("aptosb",      "Aptos", true,  false);

    a!("gothicb",     "Century Gothic", true,  false);
    a!("gothic",      "Century Gothic", false, false);

    a!("bkant",       "Book Antiqua", false, false);
    a!("pala",        "Palatino Linotype", false, false);
    a!("palab",       "Palatino Linotype", true,  false);
    a!("palai",       "Palatino Linotype", false, true);
    a!("palabi",      "Palatino Linotype", true,  true);

    // ── LaTeX / TeX fonts ────────────────────────────────────────────
    a!("lmroman10-regular",     "Latin Modern Roman", false, false);
    a!("lmroman10-bold",        "Latin Modern Roman", true,  false);
    a!("lmroman10-italic",      "Latin Modern Roman", false, true);
    a!("lmroman10-bolditalic",  "Latin Modern Roman", true,  true);
    a!("lmroman12-regular",     "Latin Modern Roman", false, false);
    a!("lmroman12-bold",        "Latin Modern Roman", true,  false);
    a!("lmsans10-regular",      "Latin Modern Sans",  false, false);
    a!("lmsans10-bold",         "Latin Modern Sans",  true,  false);
    a!("lmsans10-oblique",      "Latin Modern Sans",  false, true);
    a!("lmmono10-regular",      "Latin Modern Mono",  false, false);
    a!("lmmono10-italic",       "Latin Modern Mono",  false, true);

    // STIX Two
    a!("stixtwotextregular",     "STIX Two Text", false, false);
    a!("stixtwotextbold",        "STIX Two Text", true,  false);
    a!("stixtwotextitalic",      "STIX Two Text", false, true);
    a!("stixtwotextbolditalic",  "STIX Two Text", true,  true);

    // TeX Gyre families
    a!("texgyretermes-regular",  "TeX Gyre Termes", false, false);
    a!("texgyretermes-bold",     "TeX Gyre Termes", true,  false);
    a!("texgyretermes-italic",   "TeX Gyre Termes", false, true);
    a!("texgyreheros-regular",   "TeX Gyre Heros",  false, false);
    a!("texgyreheros-bold",      "TeX Gyre Heros",  true,  false);
    a!("texgyrepagella-regular", "TeX Gyre Pagella", false, false);
    a!("texgyrepagella-bold",    "TeX Gyre Pagella", true,  false);
    a!("texgyrecursor-regular",  "TeX Gyre Cursor",  false, false);
    a!("texgyrecursor-bold",     "TeX Gyre Cursor",  true,  false);
    a!("texgyrebonum-regular",   "TeX Gyre Bonum",   false, false);
    a!("texgyreschola-regular",  "TeX Gyre Schola",  false, false);
    a!("texgyreadventor-regular","TeX Gyre Adventor", false, false);

    // ── PDF Base-14 (URW Nimbus clones → PDF canonical names) ────────
    a!("nimbussans-regular",       "Helvetica",    false, false);
    a!("nimbussans-bold",          "Helvetica",    true,  false);
    a!("nimbussans-italic",        "Helvetica",    false, true);
    a!("nimbussans-bolditalic",    "Helvetica",    true,  true);
    a!("nimbussansnarrow-regular", "Helvetica Narrow", false, false);
    a!("nimbussansnarrow-bold",    "Helvetica Narrow", true,  false);
    a!("nimbussansnarrow-oblique", "Helvetica Narrow", false, true);
    a!("nimbussansnarrow-boldoblique", "Helvetica Narrow", true, true);
    a!("nimbusroman-regular",      "Times-Roman",  false, false);
    a!("nimbusroman-bold",         "Times-Roman",  true,  false);
    a!("nimbusroman-italic",       "Times-Roman",  false, true);
    a!("nimbusroman-bolditalic",   "Times-Roman",  true,  true);
    a!("nimbusmonops-regular",     "Courier",      false, false);
    a!("nimbusmonops-bold",        "Courier",      true,  false);
    a!("nimbusmonops-italic",      "Courier",      false, true);
    a!("nimbusmonops-bolditalic",  "Courier",      true,  true);

    // Libertinus
    a!("libertinusserif-regular",  "Libertinus Serif", false, false);
    a!("libertinusserif-bold",     "Libertinus Serif", true,  false);
    a!("libertinusserif-italic",   "Libertinus Serif", false, true);
    a!("libertinussans-regular",   "Libertinus Sans",  false, false);

    // ── Typewriter fonts ─────────────────────────────────────────────
    a!("ogcourier",               "OGCourier", false, false);
    a!("ogcourier-bold",          "OGCourier", true,  false);
    a!("ogcourier-italic",        "OGCourier", false, true);
    a!("ogcourier-bolditalic",    "OGCourier", true,  true);
    a!("courierprime-regular",    "CourierPrime", false, false);
    a!("courierprime-bold",       "CourierPrime", true,  false);
    a!("courierprime-italic",     "CourierPrime", false, true);
    a!("courierprime-bolditalic", "CourierPrime", true,  true);
    a!("cutivemono-regular",      "CutiveMono", false, false);
    a!("specialelite-regular",    "SpecialElite", false, false);
    a!("ibmselectriclightregular","IBM Selectric Light", false, false);
    a!("ibmselectriclightitalic", "IBM Selectric Light", false, true);

    // Prestige Elite
    a!("prestigeelitestd-bd",     "Prestige Elite Std", true,  false);
    a!("prestigeelitestd-regular","Prestige Elite Std", false, false);
    a!("prestigeelitestd",        "Prestige Elite Std", false, false);

    // Letter Gothic (URW)
    a!("lettergothic-reg",        "Letter Gothic", false, false);
    a!("lettergothic-bol",        "Letter Gothic", true,  false);
    a!("lettergothic-ita",        "Letter Gothic", false, true);
    a!("lettergothic-bolita",     "Letter Gothic", true,  true);
    // letr45w is URW Letter Gothic Regular (URW naming convention)
    a!("letr45w",                 "Letter Gothic", false, false);

    m
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

const SERIF_HINTS: &[&str] = &[
    "times", "georgia", "garamond", "cambria", "palatino", "book antiqua",
    "bookman", "century schoolbook", "century", "computer modern",
    "latin modern roman", "cmu serif", "stix", "libertinus serif",
    "tex gyre termes", "tex gyre pagella", "tex gyre bonum", "tex gyre schola",
    "concrete", "minion", "caslon", "baskerville",
];

const SANS_HINTS: &[&str] = &[
    "arial", "helvetica", "calibri", "verdana", "tahoma", "segoe",
    "trebuchet", "comic sans", "aptos", "century gothic", "avant garde",
    "cmu sans", "latin modern sans", "computer modern sans",
    "libertinus sans", "tex gyre heros", "tex gyre adventor",
    "fira sans", "open sans", "roboto", "lato", "noto sans",
];

const MONO_HINTS: &[&str] = &[
    "courier", "consolas", "menlo", "monaco", "cmu typewriter",
    "latin modern mono", "computer modern typewriter", "tex gyre cursor",
    "fira code", "fira mono", "source code", "inconsolata", "lucida console",
    "dejavu sans mono", "liberation mono", "ubuntu mono",
    "freemono",
    // Typewriter fonts — monospaced by nature
    "prestige", "selectric", "letter gothic", "lettergothic",
    "cutive mono", "cutivemono", "special elite", "specialelite",
    "og courier", "ogcourier", "courier prime", "courierprime",
];

fn classify(family: &str) -> FontClass {
    let lower = family.to_lowercase();
    if MONO_HINTS.iter().any(|h| lower.contains(h)) {
        FontClass::Mono
    } else if SERIF_HINTS.iter().any(|h| lower.contains(h)) {
        FontClass::Serif
    } else if SANS_HINTS.iter().any(|h| lower.contains(h)) {
        FontClass::Sans
    } else {
        FontClass::Unknown
    }
}

fn class_label(c: FontClass) -> &'static str {
    match c {
        FontClass::Serif => "serif",
        FontClass::Sans  => "sans",
        FontClass::Mono  => "mono",
        FontClass::Unknown => "?",
    }
}

// ---------------------------------------------------------------------------
// Font loading
// ---------------------------------------------------------------------------

/// Detect whether a font uses old-style (text/ranging) figures.
///
/// Old-style figures have varying heights: digits like 3, 5, 7, 9 typically
/// descend below the baseline, while 6 and 8 ascend higher. Lining figures
/// are all the same height (cap-height) and sit on the baseline.
///
/// We check by rendering '0' and '3' and comparing their vertical bounds.
/// If '3' descends noticeably below '0', it's old-style.
fn detect_oldstyle_figures(data: &[u8]) -> bool {
    let font = match FontRef::try_from_slice(data) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let scale = PxScale::from(100.0);
    let ascent = font.as_scaled(scale).ascent();

    // Position glyphs at the baseline so we can compare their actual outlines.
    let gid_0 = font.glyph_id('0');
    let gid_3 = font.glyph_id('3');
    if gid_0.0 == 0 || gid_3.0 == 0 {
        return false;
    }

    let g0 = gid_0.with_scale_and_position(scale, ab_glyph::point(0.0, ascent));
    let g3 = gid_3.with_scale_and_position(scale, ab_glyph::point(0.0, ascent));

    let og0 = match font.outline_glyph(g0) {
        Some(o) => o,
        None => return false,
    };
    let og3 = match font.outline_glyph(g3) {
        Some(o) => o,
        None => return false,
    };

    let b0 = og0.px_bounds();
    let b3 = og3.px_bounds();

    let height_0 = b0.max.y - b0.min.y;
    if height_0 < 1.0 {
        return false;
    }

    // If '3' extends further below than '0', it has a descender (old-style).
    let descent_diff = b3.max.y - b0.max.y;

    // 15% of '0' height threshold
    descent_diff > height_0 * 0.15
}

/// OpenType features to probe for variant generation.
/// Only features that are OFF by default in most renderers — so they represent
/// deliberate typographic choices that change glyph shapes.
const VARIANT_FEATURES: &[&[u8; 4]] = &[
    b"onum",  // Old-style (text) numerals
    b"lnum",  // Lining numerals (explicit — some fonts default to onum)
    b"smcp",  // Small capitals
    b"c2sc",  // Capitals to small caps
    b"swsh",  // Swash alternates
    b"salt",  // Stylistic alternates
    b"titl",  // Titling alternates
    b"hist",  // Historical forms
    b"ss01", b"ss02", b"ss03", b"ss04", b"ss05",
    b"ss06", b"ss07", b"ss08", b"ss09", b"ss10",
    b"ss11", b"ss12", b"ss13", b"ss14", b"ss15",
    b"ss16", b"ss17", b"ss18", b"ss19", b"ss20",
];

/// Test string covering Latin alphanumerics + a few common punctuation marks.
const PROBE_STRING: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Use rustybuzz to detect which OT features produce different glyph IDs
/// for common Latin characters. Returns a vec of (feature_tag, glyph_overrides)
/// for each feature that changes at least one glyph.
fn detect_ot_variants(data: &[u8]) -> Vec<(String, Vec<(char, u16)>)> {
    let face = match rustybuzz::Face::from_slice(data, 0) {
        Some(f) => f,
        None => return Vec::new(),
    };

    // Shape with no extra features (default rendering)
    let mut buf_default = rustybuzz::UnicodeBuffer::new();
    buf_default.push_str(PROBE_STRING);
    let out_default = rustybuzz::shape(&face, &[], buf_default);
    let default_ids: Vec<u16> = out_default
        .glyph_infos()
        .iter()
        .map(|gi| gi.glyph_id as u16)
        .collect();

    let chars: Vec<char> = PROBE_STRING.chars().collect();
    if default_ids.len() != chars.len() {
        return Vec::new();
    }

    let mut variants = Vec::new();

    for tag_bytes in VARIANT_FEATURES {
        let tag = rustybuzz::ttf_parser::Tag::from_bytes(tag_bytes);
        let features = [rustybuzz::Feature::new(tag, 1, ..)];

        let mut buf = rustybuzz::UnicodeBuffer::new();
        buf.push_str(PROBE_STRING);
        let out = rustybuzz::shape(&face, &features, buf);
        let feat_ids: Vec<u16> = out
            .glyph_infos()
            .iter()
            .map(|gi| gi.glyph_id as u16)
            .collect();

        if feat_ids.len() != chars.len() {
            continue;
        }

        // Collect only characters whose glyph ID actually changed
        let overrides: Vec<(char, u16)> = chars
            .iter()
            .zip(default_ids.iter())
            .zip(feat_ids.iter())
            .filter(|((_, def), feat)| def != feat)
            .map(|((ch, _), feat)| (*ch, *feat))
            .collect();

        if !overrides.is_empty() {
            let tag_str = std::str::from_utf8(tag_bytes.as_slice())
                .unwrap_or("????")
                .to_string();
            variants.push((tag_str, overrides));
        }
    }

    variants
}

/// Ligature probe sequences: (input_chars, unicode_ligature_char).
/// We shape the input chars with liga/dlig features and check if the
/// shaper produces a single glyph (i.e. a ligature substitution fired).
const LIGATURE_PROBES: &[(&str, char)] = &[
    ("ff",  '\u{FB00}'),
    ("fi",  '\u{FB01}'),
    ("fl",  '\u{FB02}'),
    ("ffi", '\u{FB03}'),
    ("ffl", '\u{FB04}'),
];

/// Detect ligature glyph IDs by shaping probe strings with liga and dlig
/// features. Returns a vec of (unicode_ligature_char, glyph_id) for each
/// ligature the font supports.
fn detect_ligature_glyphs(data: &[u8]) -> Vec<(char, u16)> {
    let face = match rustybuzz::Face::from_slice(data, 0) {
        Some(f) => f,
        None => return Vec::new(),
    };

    let mut result = Vec::new();

    for &(probe, lig_char) in LIGATURE_PROBES {
        let input_len = probe.chars().count();

        // Try with both liga and dlig enabled
        let liga_tag = rustybuzz::ttf_parser::Tag::from_bytes(b"liga");
        let dlig_tag = rustybuzz::ttf_parser::Tag::from_bytes(b"dlig");
        let features = [
            rustybuzz::Feature::new(liga_tag, 1, ..),
            rustybuzz::Feature::new(dlig_tag, 1, ..),
        ];

        let mut buf = rustybuzz::UnicodeBuffer::new();
        buf.push_str(probe);
        let out = rustybuzz::shape(&face, &features, buf);
        let infos = out.glyph_infos();

        // If shaping produced exactly 1 glyph from N input chars, it's a ligature
        if infos.len() == 1 && input_len > 1 {
            let gid = infos[0].glyph_id as u16;
            if gid != 0 {
                result.push((lig_char, gid));
            }
        }
    }

    result
}

/// Read the PostScript name (name ID 6) from the font's name table.
/// Returns empty string if unavailable.
fn read_postscript_name(data: &[u8]) -> String {
    use rustybuzz::ttf_parser;
    let face = match ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    // name ID 6 = PostScript name. Prefer platformID 3 (Windows) / encodingID 1 (Unicode BMP),
    // fall back to platformID 1 (Macintosh) / encodingID 0 (Roman).
    for name in face.names() {
        if name.name_id == 6 {
            if let Some(s) = name.to_string() {
                return s;
            }
        }
    }
    String::new()
}

/// Ensure the PostScript name contains an explicit weight keyword.
///
/// If the PS name already contains a weight marker (Regular, Bold, Light, etc.),
/// it is returned unchanged.  Otherwise the OS/2 usWeightClass is mapped to a
/// keyword and inserted:
///   "Lato-Italic" (w400) → "Lato-RegularItalic"
///   "OpenSans"    (w400) → "OpenSans-Regular"
///   "Lato-Bold"   (w700) → "Lato-Bold" (already explicit)
///
/// The specimen generator (`gen-specimen.py`) applies the identical transform so
/// ground-truth PDF names and catalog names always agree.
pub fn make_weight_explicit(ps_name: &str, weight: u16) -> String {
    if ps_name.is_empty() {
        return ps_name.to_string();
    }

    let weight_keyword = match weight {
        0..=150   => "Thin",
        151..=250 => "ExtraLight",
        251..=350 => "Light",
        351..=450 => "Regular",
        451..=475 => "Text",
        476..=550 => "Medium",
        551..=650 => "SemiBold",
        651..=750 => "Bold",
        751..=850 => "ExtraBold",
        _         => "Black",
    };

    // Check if PS name already contains an explicit weight marker.
    let lower = ps_name.to_lowercase().replace('-', "");
    let markers = [
        "thin", "extralight", "ultralight", "light",
        "regular", "text", "book",
        "medium", "semibold", "demibold", "demi",
        "bold", "extrabold", "ultrabold",
        "black", "heavy",
    ];
    if markers.iter().any(|m| lower.contains(m)) {
        return ps_name.to_string();
    }

    // Insert weight: before "Italic"/"It" suffix, or append.
    if let Some(idx) = ps_name.to_lowercase().find("italic") {
        let prefix = ps_name[..idx].trim_end_matches('-');
        let italic_part = &ps_name[idx..];
        format!("{}-{}{}", prefix, weight_keyword, italic_part)
    } else if ps_name.ends_with("It") {
        let prefix = ps_name[..ps_name.len()-2].trim_end_matches('-');
        format!("{}-{}It", prefix, weight_keyword)
    } else {
        format!("{}-{}", ps_name, weight_keyword)
    }
}

/// Font identity for major/minor miss classification.
/// Read from the font's name table and OS/2 table — no string munging.
#[derive(Debug, Clone)]
pub struct FontIdentity {
    /// Typographic family: name ID 16 if present, else name ID 1.
    pub family: String,
    /// OS/2 usWeightClass (400 = Regular, 500 = Medium, 700 = Bold, etc.)
    pub weight: u16,
    /// OS/2 fsSelection italic bit.
    pub italic: bool,
}

impl FontIdentity {
    /// Weight bucket: 100-unit ranges (400–499 = Regular, 500–599 = Medium, etc.)
    pub fn weight_bucket(&self) -> u16 {
        self.weight / 100
    }

    /// Two fonts are a "major" difference if family, italic, or weight bucket differ.
    pub fn is_major_diff(&self, other: &FontIdentity) -> bool {
        self.family != other.family
            || self.italic != other.italic
            || self.weight_bucket() != other.weight_bucket()
    }
}

/// Read font identity from a font file path. Returns None on parse failure.
pub fn read_font_identity(path: &Path) -> Option<FontIdentity> {
    use rustybuzz::ttf_parser;
    let data = std::fs::read(path).ok()?;
    let face = ttf_parser::Face::parse(&data, 0).ok()?;

    // Name ID 16 (typographic family) if present, else name ID 1 (family).
    let mut nid1: Option<String> = None;
    let mut nid16: Option<String> = None;
    for name in face.names() {
        if name.name_id == 16 && nid16.is_none() {
            nid16 = name.to_string();
        }
        if name.name_id == 1 && nid1.is_none() {
            nid1 = name.to_string();
        }
    }
    let family = nid16.or(nid1)?;

    let weight = face.tables().os2.map(|os2| os2.weight().to_number()).unwrap_or(400);
    let italic = face.is_italic();

    Some(FontIdentity { family, weight, italic })
}

fn load_font_entry(path: &Path, aliases: &HashMap<String, Alias>) -> Option<FontEntry> {
    let data = std::fs::read(path).ok()?;

    // Verify ab_glyph can parse it (reject corrupt files)
    let _ = ab_glyph::FontRef::try_from_slice(&data).ok()?;

    let oldstyle_figures = detect_oldstyle_figures(&data);
    let raw_ps_name = read_postscript_name(&data);

    // Read OS/2 weight for make_weight_explicit
    let os2_weight = {
        use rustybuzz::ttf_parser;
        ttf_parser::Face::parse(&data, 0)
            .ok()
            .and_then(|face| face.tables().os2)
            .map(|os2| os2.weight().to_number())
            .unwrap_or(400)
    };
    let postscript_name = make_weight_explicit(&raw_ps_name, os2_weight);

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown");

    let stem_lower = stem.to_lowercase().replace(' ', "");

    // Try alias table first (exact match on lowercase stem)
    if let Some(alias) = aliases.get(&stem_lower) {
        let class = classify(alias.family);
        return Some(FontEntry {
            path: path.to_path_buf(),
            family_name: alias.family.to_string(),
            postscript_name,
            is_bold: alias.bold,
            is_italic: alias.italic,
            class,
            data,
            oldstyle_figures,
            variant_tag: String::new(),
            glyph_overrides: None,
        });
    }

    // Fallback: derive from filename
    let family_name = stem.replace('-', " ").replace('_', " ");
    let lower = family_name.to_lowercase();
    let is_bold = lower.contains("bold") || lower.contains("black") || lower.contains("heavy");
    let is_italic = lower.contains("italic") || lower.contains("oblique") || lower.contains("slant");
    let class = classify(&family_name);

    Some(FontEntry {
        path: path.to_path_buf(),
        family_name,
        postscript_name,
        is_bold,
        is_italic,
        class,
        data,
        oldstyle_figures,
        variant_tag: String::new(),
        glyph_overrides: None,
    })
}
