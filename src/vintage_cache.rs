//! Vintage font cache — locally build old-school era variants from modern fonts.
//!
//! Licensing: NO REDISTRIBUTION of generated fonts.
//!   Unprint builds these fonts on the user's machine from modern fonts the user
//!   already has (OFL / Apache 2.0 / system fonts). The vintage variants are written
//!   to `~/.cache/unprint/vintage/` and never checked into git or distributed.
//!   Only the *transformation code* (this module + era configs) is shipped.
//!
//! Design:
//!   - Early in `run()`, after scanning base fonts, we call `ensure_vintage_fonts()`
//!   - For each base font, we lookup its birth year via `font_history::font_birth_year`
//!   - If birth year is unknown => skip (prevents 1988 Comic Sans)
//!   - If birth year <= era.start_year => font existed at that era, so we generate variant
//!   - Cache path = `~/.cache/unprint/vintage/<sanitized>-<era>-<hash>.ttf`
//!   - If cached file exists and is newer than base, reuse; else regenerate
//!   - Currently stubbed as copy + log (safe to ship); TODO: replace with write-fonts transform
//!
//! Future write-fonts transforms (TODO markers below):
//!   PostScript: strip GPOS, keep kern fmt0 only, limit to ~300 pairs (1985)
//!   TrueType: empty kern/GPOS (Word 6 default kerning off) (1990)
//!   Word6: kern fmt0 only, coarse quantization (1993)
//!   OpenType: GPOS class kerning limited (1996)
//!   Pdf14: full GPOS + ToUnicode (2001)
//!   InDesignCS: optical kerning simulation (outline distance + jitter) (2003)
//!   Docx: Calibri-style class kerning, 1289 pairs quantize (2007)
//!   HarfBuzz: HarfBuzz-shaped, mark offsets, over-segmented TJ <50 merge (2013)
//!   Variable: gvar/avar tracking via Tc vs variation (2020)
//!
//! Naming: readable slugs like `word6`, `postscript`, `harfbuzz`.
//! Old cache files using year-suffixed names like `ps1985`, `word6_1993` will be orphaned
//! on upgrade — intentional migration to readable names. Delete old `*.ttf` in vintage dir
//! if you need to reclaim space.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use read_fonts::types::Tag;
use read_fonts::FontRef;
use write_fonts::FontBuilder;

use crate::font_history::{era_exists_for_font, font_birth_year};
use crate::font_scan::FontEntry;

/// Era definition — historical typesetting engine generations (last 40 years).
/// Variant names are readable (word6, postscript) — years are kept in start_year/end_year for gating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Era {
    /// PostScript Level 1 (1985-1989) — AFM KPX ~300 pairs, screen≠print
    PostScript,
    /// TrueType vs Type1 war (1990-1994) — Word kerning checkbox off by default
    TrueType,
    /// Word 6.0 / WordPerfect 5.1 era (1993-1995) — proportional table, Quark 1/200em quant
    Word6,
    /// OpenType 1.0, Core Web Fonts, pdfTeX (1996-1999)
    OpenType,
    /// PDF 1.4, Office 2000/XP (2001-2003) — ToUnicode, early ClearType draft
    Pdf14,
    /// InDesign CS optical kerning (2003-2008) — outline distance, not table
    InDesignCS,
    /// Office 2007 / Vista DOCX SaveAsPDF (2007-2010) — ClearType Collection, Calibri 1289 pairs
    Docx,
    /// LibreOffice→HarfBuzz, Chrome Skia PDF, Google Docs (2013-2019)
    HarfBuzz,
    /// PDF 2.0, variable fonts gvar/avar (2020-2026)
    Variable,
}

/// Full era definition with shipping metadata (for docs and was_shipped_with).
pub struct EraDef {
    pub era: Era,
    pub name: &'static str,
    pub start_year: u16,
    pub end_year: u16,
    /// Fonts actually bundled/shipped with this system, independent of birth year.
    /// Lower-case substrings matched via canonical_family().
    pub shipped: &'static [&'static str],
    pub description: &'static str,
}

impl Era {
    /// Human-readable slug without year — used in cache file names and variant tags.
    /// e.g. "word6", "postscript", "harfbuzz"
    pub fn name(&self) -> &'static str {
        match self {
            Era::PostScript => "postscript",
            Era::TrueType => "truetype",
            Era::Word6 => "word6",
            Era::OpenType => "opentype",
            Era::Pdf14 => "pdf14",
            Era::InDesignCS => "indesign_cs",
            Era::Docx => "docx",
            Era::HarfBuzz => "harfbuzz",
            Era::Variable => "variable",
        }
    }

    pub fn start_year(&self) -> u16 {
        match self {
            Era::PostScript => 1985,
            Era::TrueType => 1990,
            Era::Word6 => 1993,
            Era::OpenType => 1996,
            Era::Pdf14 => 2001,
            Era::InDesignCS => 2003,
            Era::Docx => 2007,
            Era::HarfBuzz => 2013,
            Era::Variable => 2020,
        }
    }

    pub fn end_year(&self) -> u16 {
        match self {
            Era::PostScript => 1989,
            Era::TrueType => 1994,
            Era::Word6 => 1995,
            Era::OpenType => 1999,
            Era::Pdf14 => 2003,
            Era::InDesignCS => 2008,
            Era::Docx => 2010,
            Era::HarfBuzz => 2019,
            Era::Variable => 2026,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Era::PostScript => "PostScript Level 1 (1985) — AFM KPX ~300 pairs, screen!=print, WYSINOTWYF",
            Era::TrueType => "TrueType vs Type1 war (1990) — Word6 kerning checkbox off by default",
            Era::Word6 => "Word 6.0 / WordPerfect 5.1 (1993) — proportional table, Quark 3.3 editable 1/200em",
            Era::OpenType => "OpenType 1.0 (1996) — GSUB/GPOS, pdfTeX, class kerning 10k combos",
            Era::Pdf14 => "PDF 1.4 (2001) — ToUnicode, Calibri draft, early ClearType",
            Era::InDesignCS => "InDesign CS (2003) — optical kerning (outline distance), metrics vs optical",
            Era::Docx => "DOCX SaveAsPDF (2007) — Calibri 1289 pairs+89 classes, Office 2007 ClearType",
            Era::HarfBuzz => "HarfBuzz era (2013) — LibreOffice→HarfBuzz, Chrome Skia PDF, Google Docs 0 TJ + absolute Tm",
            Era::Variable => "Variable fonts (2020) — PDF 2.0 CFF2/ActualText, gvar/avar, mark offsets",
        }
    }

    /// Fonts actually shipped with this era/system (cumulative bundle).
    /// Lower-case substrings, matched after canonicalization.
    pub fn shipped_fonts(&self) -> &'static [&'static str] {
        match self {
            Era::PostScript => SHIPPED_POSTSCRIPT,
            Era::TrueType => SHIPPED_TRUETYPE,
            Era::Word6 => SHIPPED_WORD6,
            Era::OpenType => SHIPPED_OPENTYPE,
            Era::Pdf14 => SHIPPED_PDF14,
            Era::InDesignCS => SHIPPED_INDESIGN_CS,
            Era::Docx => SHIPPED_DOCX,
            Era::HarfBuzz => SHIPPED_HARFBUZZ,
            Era::Variable => SHIPPED_VARIABLE,
        }
    }

    pub fn def(&self) -> EraDef {
        EraDef {
            era: *self,
            name: self.name(),
            start_year: self.start_year(),
            end_year: self.end_year(),
            shipped: self.shipped_fonts(),
            description: self.description(),
        }
    }
}

/// All eras in chronological order.
pub const ALL_ERAS: &[Era] = &[
    Era::PostScript,
    Era::TrueType,
    Era::Word6,
    Era::OpenType,
    Era::Pdf14,
    Era::InDesignCS,
    Era::Docx,
    Era::HarfBuzz,
    Era::Variable,
];

/// Minimal set used by default to keep cache size reasonable.
/// Covers biggest spacing deltas: AFM-only, Quark quant, modern class kern.
/// Readable names: postscript (1985 AFM), word6 (1993 quant), docx (2007 ClearType)
/// Active set used by `main.rs` by default: all eras except InDesignCS (optical kerning).
/// Leave out InDesignCS for now per Mike — its outline-distance model needs its own pipeline.
pub const DEFAULT_ERAS: &[Era] = &[
    Era::PostScript, // AFM-only, no GPOS
    Era::TrueType,   // Word 6 kerning-off default
    Era::Word6,      // Quark / WordPerfect quantization — THE user example
    Era::OpenType,   // GPOS class 64
    Era::Pdf14,      // full GPOS + ToUnicode
    // Era::InDesignCS omitted intentionally — optical kerning simulation deferred
    Era::Docx,       // modern but coarse class kerning (Calibri 1289 pairs)
    Era::HarfBuzz,   // LO→HarfBuzz, Chrome Skia, GPOS-only
    Era::Variable,   // PDF 2.0 CFF2/gvar, Tc tracking
];

// ---------------------------------------------------------------------------
// Shipping tables — what each system actually bundled, independent of birth.
// Cumulative: later eras bundle previous + new.
// ---------------------------------------------------------------------------

/// PostScript Base 13 (1985) + common LaserWriter additions.
/// Courier, Helvetica (incl. Oblique/Bold), Times (Times-Roman), Symbol, Zapf Dingbats.
/// Plus clones via canonical mapping: Nimbus -> Helvetica/Times/Courier, TeX Gyre -> same.
const SHIPPED_POSTSCRIPT: &[&str] = &[
    "courier",
    "helvetica",
    "times",
    "times-roman",
    "times new roman", // clone considered shipped via PS->Win mapping for free builder
    "symbol",
    "zapf dingbats",
    "itc zapf dingbats",
    "palatino", // frequent PS addition
    "bookman",
    "avant garde",
    "zapf chancery",
    "nimbus sans", // URW free PS clones - allow as stand-ins
    "nimbus roman",
    "nimbus mono",
    "tex gyre", // TeX Gyre as stand-in for PS base (free)
    "computer modern",
    "latin modern",
];

/// Windows 3.1 (1992) core TrueType: the actual .TTF files on disk for Word 6 era.
/// Ground truth from KB Q89652: 14 files = Arial 4, Courier New 4, Times New Roman 4, Symbol 1, Wingdings 1.
/// MS Sans Serif / MS Serif raster .FON also installed but not TrueType; listed for completeness.
const SHIPPED_TRUETYPE: &[&str] = &[
    // Core 5 families = 14 files (Q89652)
    "arial",
    "courier new",
    "times new roman",
    "symbol",
    "wingdings",
    // Raster System UI (not TTF) but shipped and usable for layout fallbacks
    "ms sans serif",
    "ms serif",
    // PS Base retained via ATM – allow if user has them
    "courier",
    "helvetica",
    "times",
    "times-roman",
    // Free stand-ins (build locally, no redistribution) – canonicalized to core families
    "nimbus sans",
    "nimbus roman",
    "nimbus mono",
    "tex gyre",
    "tex gyre heros",
    "tex gyre termes",
    "tex gyre cursor",
    "latin modern",
    "computer modern",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "arimo",
    "tinos",
    "cousine",
];

/// Word 6.0 (1993) / WordPerfect 5.1 (1989 DOS) era.
/// Word 6 for Windows shipped NO new TrueType fonts beyond Windows 3.1 core.
/// TrueType Font Pack (Lucida Sans Typewriter etc) was optional retail, not baseline.
/// So SHIPPED = same as TrueType strict core + free clones for local build.
///
/// THIS IS THE CANONICAL EXAMPLE USER ASKED FOR: Word only had certain fonts regardless of when they were made.
/// Hardcode from ~/workspace/goals/co-build-unprint-ocr/files/word6-shipped-fonts.md:
///
/// WORD6_SHIPPED = ["arial","courier new","times new roman","symbol","wingdings"]
/// WORD6_FILES = 14 files ARIAL.TTF, COUR.TTF, TIMES.TTF, SYMBOL.TTF, WINGDING.TTF etc 03-10-92
const SHIPPED_WORD6: &[&str] = &[
    // Strict core (5 families) — ground truth KB Q89652
    "arial",
    "courier new",
    "times new roman",
    "symbol",
    "wingdings",
    // Raster legacy
    "ms sans serif",
    "ms serif",
    // ATM PS base if present
    "courier",
    "helvetica",
    "times",
    "times-roman",
    "symbol",
    // Free stand-ins via HISTORICAL_CLONE_MAP canonicalization — buildable locally
    "nimbus sans",
    "nimbus roman",
    "nimbus mono",
    "tex gyre",
    "tex gyre heros",
    "tex gyre termes",
    "tex gyre cursor",
    "latin modern",
    "computer modern",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "arimo",
    "tinos",
    "cousine",
];

/// Core Web Fonts 1996 + Windows 95 / Office 95 additions.
/// MS Core Web 1996: Andale Mono, Arial Black, Comic Sans MS, Courier New, Georgia (design 1993 but web release 1996), Impact, Times New Roman, Trebuchet MS, Verdana, Webdings
/// Office 95 adds Book Antiqua (Palatino clone), Century Gothic, Haettenschweiler.
const SHIPPED_OPENTYPE: &[&str] = &[
    // Previous
    "courier",
    "helvetica",
    "times",
    "times new roman",
    "times-roman",
    "arial",
    "courier new",
    "symbol",
    "wingdings",
    "ms sans serif",
    "ms serif",
    // Core Web 1996
    "andale mono",
    "arial black",
    "comic sans ms",
    "comic sans",
    "georgia",
    "impact",
    "trebuchet ms",
    "trebuchet",
    "verdana",
    "webdings",
    // Office 95
    "book antiqua",
    "century gothic",
    "haettenschweiler",
    // Free stand-ins
    "nimbus sans",
    "nimbus roman",
    "nimbus mono",
    "tex gyre",
    "latin modern",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "dejavu",
    "dejavu sans",
    "dejavu serif",
];

/// PDF 1.4 / Office 2000 era (1999-2003). Adds Palatino Linotype, Bookman Old Style, etc.
const SHIPPED_PDF14: &[&str] = &[
    // Carry forward all OpenType
    "courier",
    "helvetica",
    "times",
    "times new roman",
    "times-roman",
    "arial",
    "courier new",
    "symbol",
    "wingdings",
    "ms sans serif",
    "ms serif",
    "andale mono",
    "arial black",
    "comic sans ms",
    "comic sans",
    "georgia",
    "impact",
    "trebuchet ms",
    "trebuchet",
    "verdana",
    "webdings",
    "book antiqua",
    "century gothic",
    "haettenschweiler",
    // Office 2000 new
    "palatino linotype",
    "palatino",
    "bookman old style",
    "bookman",
    "century schoolbook",
    "monotype corsiva",
    // Free
    "nimbus sans",
    "nimbus roman",
    "nimbus mono",
    "tex gyre",
    "tex gyre termes",
    "tex gyre heros",
    "tex gyre pagella",
    "tex gyre bonum",
    "tex gyre schola",
    "latin modern",
    "computer modern",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "dejavu",
];

/// InDesign CS (2003-2008): same bundle as Office 2003, plus OpenType Pro families
/// but no ClearType yet. For our model, same as Pdf14.
const SHIPPED_INDESIGN_CS: &[&str] = SHIPPED_PDF14;

/// Office 2007 / Windows Vista (2007): first ClearType Collection.
/// Calibri, Cambria, Candara, Consolas, Constantia, Corbel + Segoe UI.
/// This is first era where Calibri is valid, even though designed 2002-2004.
const SHIPPED_DOCX: &[&str] = &[
    // All previous
    "courier",
    "helvetica",
    "times",
    "times new roman",
    "times-roman",
    "arial",
    "courier new",
    "symbol",
    "wingdings",
    "ms sans serif",
    "ms serif",
    "andale mono",
    "arial black",
    "comic sans ms",
    "comic sans",
    "georgia",
    "impact",
    "trebuchet ms",
    "trebuchet",
    "verdana",
    "webdings",
    "book antiqua",
    "century gothic",
    "haettenschweiler",
    "palatino linotype",
    "palatino",
    "bookman old style",
    "bookman",
    "century schoolbook",
    // ClearType 2007 new
    "calibri",
    "cambria",
    "candara",
    "consolas",
    "constantia",
    "corbel",
    "segoe ui",
    "segoe",
    // Free equivalents that ship as metrics-compatible in LibreOffice/OOo
    "carlito", // Calibri clone, actually 2013 but allow as stand-in for Calibri in 2007+ via canonical
    "caladea", // Cambria clone
    "nimbus sans",
    "nimbus roman",
    "nimbus mono",
    "tex gyre",
    "latin modern",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "dejavu",
    "open sans",
    "source sans",
    "source serif",
];

/// HarfBuzz era 2013-2019: LibreOffice 4.1+ switches to HarfBuzz, Chrome Skia PDF.
/// Distros now bundle Carlito, Caladea, Source, Noto, etc. as default replacements.
const SHIPPED_HARFBUZZ: &[&str] = &[
    // All Docx
    "courier",
    "helvetica",
    "times",
    "times new roman",
    "times-roman",
    "arial",
    "courier new",
    "symbol",
    "wingdings",
    "andale mono",
    "arial black",
    "comic sans ms",
    "comic sans",
    "georgia",
    "impact",
    "trebuchet ms",
    "trebuchet",
    "verdana",
    "webdings",
    "book antiqua",
    "century gothic",
    "palatino linotype",
    "palatino",
    "bookman old style",
    "calibri",
    "cambria",
    "candara",
    "consolas",
    "constantia",
    "corbel",
    "segoe ui",
    // New free stack 2013+
    "carlito",
    "caladea",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "arimo",
    "tinos",
    "cousine",
    "dejavu",
    "dejavu sans",
    "dejavu serif",
    "source sans",
    "source sans pro",
    "source sans 3",
    "source serif",
    "source serif pro",
    "source serif 4",
    "source code pro",
    "noto",
    "noto sans",
    "noto serif",
    "tex gyre",
    "tex gyre termes",
    "tex gyre heros",
    "latin modern",
    "computer modern",
    "open sans",
    "lato",
    "ubuntu",
    "roboto",
    "fira sans",
];

/// Variable fonts era 2020-2026: everything previous plus Inter, IBM Plex, Jost, etc.
const SHIPPED_VARIABLE: &[&str] = &[
    // Carry all HarfBuzz
    "courier",
    "helvetica",
    "times",
    "times new roman",
    "times-roman",
    "arial",
    "courier new",
    "symbol",
    "wingdings",
    "andale mono",
    "arial black",
    "comic sans ms",
    "georgia",
    "impact",
    "trebuchet ms",
    "verdana",
    "webdings",
    "book antiqua",
    "century gothic",
    "palatino linotype",
    "calibri",
    "cambria",
    "candara",
    "consolas",
    "constantia",
    "corbel",
    "segoe ui",
    "carlito",
    "caladea",
    "liberation sans",
    "liberation serif",
    "liberation mono",
    "source sans",
    "source serif",
    "noto",
    "tex gyre",
    "latin modern",
    // 2020+ additions
    "inter",
    "ibm plex",
    "ibm plex sans",
    "ibm plex serif",
    "ibm plex mono",
    "jost",
    "merriweather",
    "playfair display",
    "eb garamond",
    "libre baskerville",
    "libre bodoni",
    "libre caslon",
    "fira code",
    "aptos",
    "bahnschrift", // variable 2017 but allow (fixed leading space typo from previous)
];

// ---------------------------------------------------------------------------
// Shipping check
// ---------------------------------------------------------------------------

/// Returns true if `font_family` was actually shipped/bundled with `era` system,
/// independent of birth year. Uses canonicalization for free clones (carlito -> calibri).
///
/// Logic: normalize + canonicalize family, then check if it contains or is contained
/// by any entry in era.shipped_fonts(). Longest-match wins, but for shipping we
/// accept substring match because "Times New Roman" should match shipped "times".
pub fn was_shipped_with(font_family: &str, era: Era) -> bool {
    use crate::font_history::canonical_family;

    let lower = font_family.to_lowercase();
    let canon = canonical_family(&lower); // already lowercased via function

    let shipped = era.shipped_fonts();

    // Check canonical first (e.g., carlito -> calibri)
    for &shipped_entry in shipped {
        let s = shipped_entry.to_lowercase();
        // Exact or substring in either direction to handle "times new roman" vs "times"
        if canon.contains(&s) || s.contains(&canon) {
            return true;
        }
        // Also check original lower for free fonts that ship as themselves (e.g., source sans ships in harfbuzz)
        if lower.contains(&s) || s.contains(&lower) {
            return true;
        }
    }
    false
}

/// Extended check that also considers PS name as fallback (for fonts where family is generic like "Sans" but PS name has "Arial").
pub fn was_shipped_with_family_or_ps(family: &str, ps_name: &str, era: Era) -> bool {
    if was_shipped_with(family, era) {
        return true;
    }
    if !ps_name.is_empty() && was_shipped_with(ps_name, era) {
        return true;
    }
    false
}

/// Vintage cache directory: ~/.cache/unprint/vintage
pub fn vintage_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("unprint").join("vintage")
}

/// Compute sanitized family string for filenames.
fn sanitize_family(family: &str) -> String {
    let mut s = String::with_capacity(family.len());
    for ch in family.chars() {
        if ch.is_ascii_alphanumeric() {
            s.push(ch.to_ascii_lowercase());
        } else if ch == ' ' || ch == '-' || ch == '_' {
            s.push('_');
        }
        // else skip
    }
    if s.is_empty() {
        "unknown".to_string()
    } else {
        // truncate to 32 chars
        if s.len() > 32 { s.truncate(32); }
        s
    }
}

/// Hash base identity + era for cache key stability.
/// Previously hashed base_path string + mtime/len, which caused duplicate
/// msttcorefonts files (Arial.TTF vs Arial.ttf vs symlink arial.ttf) to produce
/// different cache filenames for the same logical font, leading to 301 duplicate
/// font_keys and geo-cache Wrote 2748 vs 3049. Now hashes logical identity
/// (postscript_name + variant_tag) so same font maps to same cache file.
fn hash_base_era(base: &FontEntry, era: Era) -> u64 {
    let mut hasher = DefaultHasher::new();
    base.postscript_name.hash(&mut hasher);
    base.variant_tag.hash(&mut hasher);
    // vintage_era on base should be None (base is not vintage), but include for completeness
    if let Some(ref ve) = base.vintage_era {
        ve.hash(&mut hasher);
    }
    era.name().hash(&mut hasher);
    era.start_year().hash(&mut hasher);
    hasher.finish()
}

/// Full cache path for a given base font and era.
/// New naming uses readable slugs: e.g. `arial-word6-a1b2c3...ttf` not `arial-word6_1993-...`
/// Old files with year-suffixed slugs (`ps1985`, `word6_1993`) or path-hashed names are orphaned intentionally;
/// they will be left on disk and ignored after dedup, and can be GC'd separately.
pub fn vintage_cache_path(base: &FontEntry, era: Era) -> PathBuf {
    let dir = vintage_cache_dir();
    let fam = sanitize_family(&base.family_name);
    let h = hash_base_era(base, era);
    dir.join(format!("{}-{}-{:016x}.ttf", fam, era.name(), h))
}

/// Check if vintage cache exists and is newer than base font.
fn cache_valid(cache_path: &Path, base_path: &Path) -> bool {
    if !cache_path.exists() { return false; }
    let Ok(cache_meta) = std::fs::metadata(cache_path) else { return false; };
    let Ok(base_meta) = std::fs::metadata(base_path) else { return true; }; // if base missing, keep cache
    let Ok(cache_mtime) = cache_meta.modified() else { return false; };
    let Ok(base_mtime) = base_meta.modified() else { return true; };
    cache_mtime >= base_mtime
}

/// Real vintage transforms using read-fonts 0.1 + write-fonts 0.3 FontBuilder.
///
/// Caching is handled by `ensure_vintage_fonts` — this function only transforms bytes.
///
/// Strategy: parse original via FontRef, collect raw table bytes, then rebuild with
/// FontBuilder, skipping or rewriting tables per era.  This avoids needing high-level
/// write-fonts table builders which don't yet exist for `kern` in 0.3. We manipulate
/// `kern` at the byte level (format 0) which is stable since 1985.
///
/// - PostScript: strip GPOS, keep kern fmt0 only, limit to ~300 pairs
/// - TrueType: empty kern + GPOS
/// - Word6: keep kern fmt0 only, quantize to multiples of 5
/// - OpenType: strip kern, keep GPOS
/// - Pdf14/Docx/HarfBuzz/Variable: return original (modern, full GPOS)
///
/// For OTF/CFF fonts (sfnt version 'OTTO'), we patch the built font's first 4 bytes
/// to preserve 'OTTO', because FontBuilder hardcodes 0x00010000.
fn transform_font_stub(base_bytes: &[u8], era: Era, _family: &str) -> Result<Vec<u8>, String> {
    // Modern eras: no transform needed — preserve gvar/avar/CFF2/ToUnicode etc.
    match era {
        Era::Pdf14 | Era::InDesignCS | Era::Docx | Era::HarfBuzz | Era::Variable => {
            // Validate that input is at least parsable, but return original bytes unchanged.
            if FontRef::new(base_bytes).is_err() {
                return Err("base font parse failed".to_string());
            }
            return Ok(base_bytes.to_vec());
        }
        _ => {}
    }

    let font_ref = FontRef::new(base_bytes).map_err(|e| format!("font parse failed: {e:?}"))?;
    let sfnt_version = font_ref.table_directory.sfnt_version();

    // Collect all tables as owned Vec<u8>
    let mut orig_tables: Vec<(Tag, Vec<u8>)> = Vec::with_capacity(font_ref.table_directory.num_tables() as usize);
    for rec in font_ref.table_directory.table_records() {
        let tag = rec.tag();
        if let Some(data) = font_ref.table_data(tag) {
            // FontData implements AsRef<[u8]>
            orig_tables.push((tag, data.as_ref().to_vec()));
        }
    }

    // Helpers for Tag constants
    let tag_gpos = Tag::new(b"GPOS");
    let tag_kern = Tag::new(b"kern");
    let tag_gdef = Tag::new(b"GDEF");

    // Build filtered/rebuilt table list
    let mut out_tables: Vec<(Tag, Vec<u8>)> = Vec::with_capacity(orig_tables.len());

    // Pre-parse kern if needed
    let kern_entry = orig_tables.iter().find(|(t, _)| *t == tag_kern).map(|(_, d)| d.clone());

    let (new_kern, drop_kern, drop_gpos) = match era {
        Era::PostScript => {
            // Keep kern (truncated), drop GPOS, keep GDEF for safety (some buggy GPOS-less fonts need it)
            let rebuilt = if let Some(kdata) = &kern_entry {
                rebuild_kern_truncate(kdata, 300)
            } else {
                None
            };
            // If we have kern, use rebuilt if Some, else keep original truncated attempt; if None, no kern in output
            if kern_entry.is_some() && rebuilt.is_none() {
                // parsing failed: keep original kern but still strip GPOS; if original >300, try naive truncate fallback
                // naive fallback: attempt to truncate by reusing original data but we keep it as-is to avoid corruption
                (None, false, true) // keep original kern as-is, drop GPOS
            } else {
                (rebuilt, false, true)
            }
        }
        Era::TrueType => {
            // Empty kern and GPOS — Word 6 checkbox off
            (None, true, true)
        }
        Era::Word6 => {
            // Quantize kern to 1/200em, strip GPOS
            let rebuilt = if let Some(kdata) = &kern_entry {
                rebuild_kern_quantize(kdata)
            } else {
                None
            };
            if kern_entry.is_some() && rebuilt.is_none() {
                (None, false, true) // keep original if rebuild fails
            } else {
                (rebuilt, false, true)
            }
        }
        Era::OpenType => {
            // OT 1.0: GPOS class kerning, kern stripped
            (None, true, false)
        }
        _ => (None, false, false), // unreachable due to early return
    };

    // PostScript also historically had no GPOS; ensure GDEF not required but keep it to avoid breaking other tables
    // The spec says drop GPOS completely; we do that via drop_gpos.

    for (tag, data) in orig_tables {
        if tag == tag_gpos && drop_gpos {
            continue;
        }
        if tag == tag_kern {
            if drop_kern {
                continue;
            }
            if let Some(ref new_data) = new_kern {
                out_tables.push((tag, new_data.clone()));
                continue;
            } else {
                // keep original (or keep original if rebuild was None but we decided to keep)
                out_tables.push((tag, data));
                continue;
            }
        }
        // For PostScript we could also drop GDEF if it references GPOS, but keep for safety
        // Requirement says strip GPOS only, so leave GDEF.
        if era == Era::PostScript && tag == tag_gdef {
            // optional: keep GDEF — it doesn't hurt for 1985 emulation, and dropping can break some fonts
            out_tables.push((tag, data));
            continue;
        }
        out_tables.push((tag, data));
    }

    // If era expects a kern but original had none, that's fine — no kern = no kerning (acceptable for PostScript fallback)

    // Rebuild font
    let mut builder = FontBuilder::default();
    for (tag, data) in &out_tables {
        builder.add_table(*tag, data.as_slice());
    }
    let mut out = builder.build();

    // Patch sfnt version if original was OTTO (CFF)
    const OTTO: u32 = 0x4F54544F; // 'OTTO'
    if sfnt_version == OTTO && out.len() >= 4 {
        out[0..4].copy_from_slice(&OTTO.to_be_bytes());
    }

    // Validate output parses
    if FontRef::new(&out).is_err() {
        // Fallback: if our build broke somehow, return original for safety (or error for old eras)
        // For old eras we still want to surface error to caller so they don't cache broken font,
        // but to keep pipeline moving we warn and return original.
        eprintln!("[vintage] warning: rebuilt {} font failed to parse, falling back to original bytes", era.name());
        return Ok(base_bytes.to_vec());
    }

    Ok(out)
}

/// Parse kern format 0 subtable and return (coverage, pairs)
fn parse_kern_format0(data: &[u8]) -> Option<(u16, Vec<(u16, u16, i16)>)> {
    if data.len() < 4 {
        return None;
    }
    let _version = u16::from_be_bytes([data[0], data[1]]);
    let n_tables = u16::from_be_bytes([data[2], data[3]]) as usize;
    if n_tables == 0 || data.len() < 4 + 6 {
        return None;
    }
    let offset = 4;
    // Only first subtable for vintage emulation
    if data.len() < offset + 6 {
        return None;
    }
    let sub_version = u16::from_be_bytes([data[offset], data[offset + 1]]);
    let sub_length = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
    let coverage = u16::from_be_bytes([data[offset + 4], data[offset + 5]]);
    // sub_version 0 = format 0; coverage low byte 0x01 = horizontal
    if sub_version != 0 {
        return None;
    }
    if sub_length < 14 || data.len() < offset + sub_length {
        return None;
    }
    let n_pairs = u16::from_be_bytes([data[offset + 6], data[offset + 7]]) as usize;
    // searchRange, entrySelector, rangeShift ignored on read, recomputed on write
    if n_pairs > 20000 {
        // sanity guard — kern with insane nPairs likely not format0
        return None;
    }
    let pairs_start = offset + 14;
    if data.len() < pairs_start + n_pairs * 6 {
        return None;
    }
    let mut pairs = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        let o = pairs_start + i * 6;
        let left = u16::from_be_bytes([data[o], data[o + 1]]);
        let right = u16::from_be_bytes([data[o + 2], data[o + 3]]);
        let value = i16::from_be_bytes([data[o + 4], data[o + 5]]);
        pairs.push((left, right, value));
    }
    Some((coverage, pairs))
}

fn build_kern_table(pairs: &[(u16, u16, i16)], coverage: u16) -> Vec<u8> {
    let n = pairs.len();
    let (search_range, entry_selector, range_shift) = if n == 0 {
        (0u16, 0u16, 0u16)
    } else {
        // largest power of 2 <= n
        let mut pow2 = 1usize;
        while pow2 * 2 <= n {
            pow2 *= 2;
        }
        let sr = (pow2 * 6) as u16;
        let es = (pow2 as f64).log2() as u16; // floor log2(pow2)
        let rs = (n * 6 - pow2 * 6) as u16;
        (sr, es, rs)
    };
    let sub_len = 14 + n * 6;
    let total_len = 4 + sub_len;
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&0u16.to_be_bytes()); // version
    out.extend_from_slice(&1u16.to_be_bytes()); // nTables =1
    out.extend_from_slice(&0u16.to_be_bytes()); // subtable version
    out.extend_from_slice(&(sub_len as u16).to_be_bytes());
    out.extend_from_slice(&coverage.to_be_bytes());
    out.extend_from_slice(&(n as u16).to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_selector.to_be_bytes());
    out.extend_from_slice(&range_shift.to_be_bytes());
    for (l, r, v) in pairs {
        out.extend_from_slice(&l.to_be_bytes());
        out.extend_from_slice(&r.to_be_bytes());
        out.extend_from_slice(&v.to_be_bytes());
    }
    debug_assert_eq!(out.len(), total_len);
    out
}

fn rebuild_kern_truncate(orig: &[u8], limit: usize) -> Option<Vec<u8>> {
    let (coverage, pairs) = parse_kern_format0(orig)?;
    let truncated: Vec<_> = pairs.into_iter().take(limit).collect();
    Some(build_kern_table(&truncated, coverage))
}

fn rebuild_kern_quantize(orig: &[u8]) -> Option<Vec<u8>> {
    let (coverage, pairs) = parse_kern_format0(orig)?;
    let quantized: Vec<(u16, u16, i16)> = pairs
        .into_iter()
        .map(|(l, r, v)| {
            let q = ((v as f32 / 5.0).round() * 5.0) as i32;
            // clamp to i16 range
            let clamped = q.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            (l, r, clamped)
        })
        .collect();
    Some(build_kern_table(&quantized, coverage))
}

/// Ensure vintage fonts exist for given base fonts and eras.
/// Returns list of cache paths that now exist (both pre-existing and newly generated).
/// Never errors fatally — logs warnings and continues.
/// Checks BOTH:
///   1) birth_year <= era.start (font existed)
///   2) was_shipped_with(font, era) (font was actually bundled/shipped with that system)
/// This prevents 1988 Comic Sans (birth 1994) AND 1993 Word version of Calibri (birth 2007, not shipped)
/// AND 1988 Georgia (born 1993 but shipped 1996).
pub fn ensure_vintage_fonts(base_fonts: &[FontEntry], eras: &[Era]) -> Vec<PathBuf> {
    let dir = vintage_cache_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[vintage] warning: could not create {}: {e}", dir.display());
        return Vec::new();
    }

    // Deduplicate base_fonts by logical identity before generating vintage.
    // /usr/share/fonts/truetype/msttcorefonts contains duplicate files like
    // Arial.TTF + Arial.ttf + symlink arial.ttf -> Arial.ttf which would otherwise
    // generate 2-3 vintage cache files with different path-hashed names but identical font_key,
    // leading to 301 duplicate keys and geo-cache Wrote 2748 vs 3049 mismatch.
    let mut seen_base_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut deduped_bases: Vec<&FontEntry> = Vec::with_capacity(base_fonts.len());
    for b in base_fonts {
        // Use font_key as canonical identity (PostScript name + variant)
        if seen_base_keys.insert(b.font_key()) {
            deduped_bases.push(b);
        }
    }
    let base_fonts_deduped = deduped_bases;

    // Debug: count proprietary originals
    let mstt_total = base_fonts_deduped.iter().filter(|b| b.path.to_string_lossy().to_ascii_lowercase().contains("msttcorefonts")).count();
    eprintln!("[vintage] debug: scanned_bases={} (deduped from {}) msttcore_candidates={}", base_fonts_deduped.len(), base_fonts.len(), mstt_total);

    let mut generated: Vec<PathBuf> = Vec::new();
    let mut skipped_unknown = 0usize;
    let mut skipped_too_new = 0usize;
    let mut skipped_not_shipped = 0usize;
    let mut reused = 0usize;
    let mut created = 0usize;

    // Avoid generating vintage from vintage (prevent recursion)
    let dir_canon = dir.canonicalize().unwrap_or(dir.clone());

    for &base in &base_fonts_deduped {
        // Skip if base is already inside vintage cache dir
        if base.path.starts_with(&dir_canon) || base.path.to_string_lossy().contains("/vintage/") {
            continue;
        }
        // Skip OT feature variants to keep cache small; allow base and wght instances
        if !base.variant_tag.is_empty() && !base.variant_tag.starts_with("wght") {
            continue;
        }

        // Proprietary-originals-only (2026-07-31 directive): use real MS Core Fonts for pixel accuracy.
        // All MS core fonts installed via ttf-mscorefonts-installer live under msttcorefonts/.
        // Clones (Liberation, Arimo, Tinos, Cousine, Nimbus, TeX Gyre, DejaVu, Carlito, Caladea, etc.)
        // are metric-compatible only and are skipped for vintage generation.
        let path_lossy = base.path.to_string_lossy().to_ascii_lowercase();
        let is_msttcore = path_lossy.contains("msttcorefonts");
        if !is_msttcore {
            skipped_not_shipped += 1;
            continue;
        }

        let birth_opt = font_birth_year(&base.family_name, &base.postscript_name);
        let birth = match birth_opt {
            Some(y) => y,
            None => {
                skipped_unknown += 1;
                continue;
            }
        };

        for &era in eras {
            // 1) Historical existence
            if !era_exists_for_font(birth, era.start_year()) {
                skipped_too_new += 1;
                continue;
            }
            // 2) Was actually shipped/bundled with this system?
            if !was_shipped_with_family_or_ps(&base.family_name, &base.postscript_name, era) {
                skipped_not_shipped += 1;
                continue;
            }

            let cache_path = vintage_cache_path(base, era);
            if cache_valid(&cache_path, &base.path) {
                reused += 1;
                generated.push(cache_path);
                continue;
            }

            // Need to generate
            let base_bytes = match std::fs::read(&base.path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[vintage] warning: could not read base {}: {e}", base.path.display());
                    continue;
                }
            };

            match transform_font_stub(&base_bytes, era, &base.family_name) {
                Ok(out_bytes) => {
                    let tmp = crate::atomic_file::tmp_for(&cache_path);
                    match std::fs::write(&tmp, &out_bytes) {
                        Ok(_) => {
                            if let Err(e) = std::fs::rename(&tmp, &cache_path) {
                                eprintln!("[vintage] warning: rename failed for {}: {e}", cache_path.display());
                                let _ = std::fs::remove_file(&tmp);
                            } else {
                                eprintln!("[vintage] generated {} variant of {} (born {} <= {} shipped, {}) -> {}",
                                    era.name(), base.family_name, birth, era.start_year(), era.description(), cache_path.display());
                                created += 1;
                                generated.push(cache_path);
                            }
                        }
                        Err(e) => {
                            eprintln!("[vintage] warning: write failed for {}: {e}", cache_path.display());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[vintage] warning: transform failed for {} {}: {e}", base.family_name, era.name());
                }
            }
        }
    }

    eprintln!("[vintage] summary: created={} reused={} skipped_unknown={} skipped_too_new={} skipped_not_shipped={} total_cached={} scanned_bases={} (deduped from {})",
        created, reused, skipped_unknown, skipped_too_new, skipped_not_shipped, generated.len(), base_fonts_deduped.len(), base_fonts.len());

    generated
}

/// Scan vintage cache dir without touching global font_scan.bin cache.
/// Returns FontEntry vec for immediate use in same run.
/// Post-processes entries to set vintage_era from filename and restore true family name from font tables.
pub fn scan_vintage_uncached() -> Vec<FontEntry> {
    let dir = vintage_cache_dir();
    if !dir.exists() {
        return Vec::new();
    }
    let mut fonts = crate::font_scan::scan_fonts_uncached(&[dir]);
    for fe in &mut fonts {
        let file_name_os = fe.path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let file_name = file_name_os.to_lowercase();
        let mut detected: Option<&'static str> = None;
        for era in ALL_ERAS {
            let needle = format!("-{}-", era.name());
            if file_name.contains(&needle) {
                detected = Some(era.name());
                break;
            }
        }
        // Set vintage_era field so font_key includes it and prevents collision with base
        if let Some(era_str) = detected {
            fe.vintage_era = Some(era_str.to_string());
            // scan_fonts_uncached used file stem to derive family_name, which for
            // "liberationsans-word6-abc.ttf" would become "liberationsans word6 abc" – wrong.
            // Since we copied base bytes, the font's internal typographic_family / postscript_name
            // are still the base's. Use typographic_family as authoritative family when available.
            if !fe.typographic_family.is_empty() {
                fe.family_name = fe.typographic_family.clone();
            } else if !fe.family_name.is_empty() {
                // Fallback: strip the "-{era}-{hash}" suffix from file stem to recover sanitized family
                if let Some(idx) = file_name.find(&format!("-{}-", era_str)) {
                    let sanitized = &file_name[..idx];
                    // sanitized was lowercased alphanumeric + underscores – restore spaces for readability
                    // We keep original casing from typographic_family if possible, else use sanitized
                    let restored = sanitized.replace('_', " ");
                    // Only override if current family looks like it contains era/hash garbage
                    if fe.family_name.contains(era_str) || fe.family_name.len() > restored.len() + 10 {
                        fe.family_name = restored;
                    }
                }
            }
            // variant_tag remains as-is (preserves wght for variable instances), vintage_era field handles distinction
        } else {
            // Could not parse era from filename (old cache format) – still mark as vintage to avoid collision
            fe.vintage_era = Some("unknown".to_string());
        }
        // Recompute cached font_key to include |vintage=ERA|var for correct dedup/sorting (perf slice 8 cache)
        fe.recompute_font_key_cache();
    }
    fonts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize() {
        assert_eq!(sanitize_family("Times New Roman"), "times_new_roman");
        assert_eq!(sanitize_family(""), "unknown");
    }

    #[test]
    fn test_cache_path_stable() {
        let fe = FontEntry {
            path: PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            family_name: "DejaVu Sans".to_string(),
            postscript_name: "DejaVuSans-400".to_string(),
            raw_postscript_name: "DejaVuSans".to_string(),
            is_bold: false,
            is_italic: false,
            class: crate::font_scan::FontClass::Sans,
            data: vec![],
            oldstyle_figures: false,
            variant_tag: String::new(),
            glyph_overrides: None,
            variations: None,
            typographic_family: "DejaVu Sans".to_string(),
            vintage_era: None,
        };
        let p1 = vintage_cache_path(&fe, Era::PostScript);
        let p2 = vintage_cache_path(&fe, Era::PostScript);
        assert!(p1.to_string_lossy().contains("dejavusans"));
        assert!(p1.to_string_lossy().contains("postscript"));
        assert_eq!(p1.parent(), p2.parent());
    }

    #[test]
    fn test_names_readable() {
        assert_eq!(Era::Word6.name(), "word6");
        assert_eq!(Era::PostScript.name(), "postscript");
        assert_eq!(Era::TrueType.name(), "truetype");
        assert_eq!(Era::OpenType.name(), "opentype");
        assert_eq!(Era::Pdf14.name(), "pdf14");
        assert_eq!(Era::InDesignCS.name(), "indesign_cs");
        assert_eq!(Era::Docx.name(), "docx");
        assert_eq!(Era::HarfBuzz.name(), "harfbuzz");
        assert_eq!(Era::Variable.name(), "variable");
        // years still accessible for gating
        assert_eq!(Era::Word6.start_year(), 1993);
        assert_eq!(Era::PostScript.start_year(), 1985);
        assert_eq!(Era::Docx.start_year(), 2007);
    }

    #[test]
    fn test_shipping_respects_readable_names() {
        // Word6 should contain arial etc, NOT calibri
        assert!(was_shipped_with("Arial", Era::Word6));
        assert!(!was_shipped_with("Calibri", Era::Word6));
        assert!(was_shipped_with("Calibri", Era::Docx));
        assert!(!was_shipped_with("Comic Sans MS", Era::PostScript));
        assert!(was_shipped_with("Comic Sans MS", Era::OpenType));
    }

    #[test]
    fn test_kern_roundtrip() {
        let pairs = vec![(10u16, 20u16, -15i16), (30, 40, 25), (50, 60, 0)];
        let data = build_kern_table(&pairs, 0x0001);
        let parsed = parse_kern_format0(&data).expect("parse should succeed");
        assert_eq!(parsed.0, 0x0001);
        assert_eq!(parsed.1.len(), 3);
        assert_eq!(parsed.1[0], (10, 20, -15));
        assert_eq!(parsed.1[1], (30, 40, 25));
    }

    #[test]
    fn test_kern_truncate() {
        let pairs: Vec<(u16, u16, i16)> = (0..500).map(|i| (i, i + 1, (i as i16 % 100) - 50)).collect();
        let orig = build_kern_table(&pairs, 1);
        let truncated = rebuild_kern_truncate(&orig, 300).expect("truncate");
        let parsed = parse_kern_format0(&truncated).unwrap();
        assert_eq!(parsed.1.len(), 300);
    }

    #[test]
    fn test_kern_quantize() {
        let pairs = vec![(1, 2, 7), (3, 4, -7), (5, 6, 12), (7, 8, -12)];
        let orig = build_kern_table(&pairs, 1);
        let q = rebuild_kern_quantize(&orig).unwrap();
        let parsed = parse_kern_format0(&q).unwrap();
        assert_eq!(parsed.1[0].2 % 5, 0);
        assert_eq!(parsed.1[1].2 % 5, 0);
        assert_eq!(parsed.1[2].2 % 5, 0);
        assert_eq!(parsed.1[3].2 % 5, 0);
        // 7 -> 5, -7 -> -5, 12 -> 10, -12 -> -10
        assert_eq!(parsed.1[0].2, 5);
        assert_eq!(parsed.1[1].2, -5);
        assert_eq!(parsed.1[2].2, 10);
        assert_eq!(parsed.1[3].2, -10);
    }

    #[test]
    fn test_transform_valid_ttf() {
        // Use a real system font if available, otherwise skip gracefully
        let candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
        ];
        let mut base_bytes = None;
        for p in candidates {
            if let Ok(b) = std::fs::read(p) {
                base_bytes = Some(b);
                break;
            }
        }
        let base = match base_bytes {
            Some(b) => b,
            None => return, // skip test in CI without system fonts
        };
        // Validate base parses
        assert!(read_fonts::FontRef::new(&base).is_ok());

        for era in [Era::PostScript, Era::TrueType, Era::Word6, Era::OpenType, Era::Pdf14, Era::Docx, Era::HarfBuzz, Era::Variable] {
            let out = transform_font_stub(&base, era, "TestFamily").expect("transform should not error");
            assert!(read_fonts::FontRef::new(&out).is_ok(), "era {:?} output should parse", era);
            // Check GPOS stripping for eras that should strip
            let orig_ref = read_fonts::FontRef::new(&base).unwrap();
            let out_ref = read_fonts::FontRef::new(&out).unwrap();
            let has_gpos_orig = orig_ref.table_data(read_fonts::types::Tag::new(b"GPOS")).is_some();
            let has_gpos_out = out_ref.table_data(read_fonts::types::Tag::new(b"GPOS")).is_some();
            match era {
                Era::PostScript | Era::TrueType | Era::Word6 => {
                    // GPOS must be stripped
                    assert!(!has_gpos_out, "GPOS should be stripped for {:?}", era);
                }
                Era::OpenType => {
                    // kern stripped, GPOS kept if orig had it
                    let has_kern_out = out_ref.table_data(read_fonts::types::Tag::new(b"kern")).is_some();
                    assert!(!has_kern_out, "kern should be stripped for OpenType");
                    if has_gpos_orig {
                        // we keep GPOS as-is
                        assert!(has_gpos_out, "GPOS should be kept for OpenType if original has it");
                    }
                }
                Era::TrueType => {
                    let has_kern_out = out_ref.table_data(read_fonts::types::Tag::new(b"kern")).is_some();
                    assert!(!has_kern_out, "kern should be stripped for TrueType");
                }
                _ => {
                    // modern eras return unchanged bytes, so table presence should match
                    assert_eq!(has_gpos_orig, has_gpos_out, "modern era {:?} should preserve GPOS", era);
                }
            }
        }
    }
}
