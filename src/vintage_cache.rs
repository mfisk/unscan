//! Vintage font cache — locally build old-school era variants
//! from modern fonts.
//!
//! Licensing: NO REDISTRIBUTION of generated fonts.
//!   Unprint builds these fonts on the user's machine from modern fonts the user
//!   already has (OFL / Apache 2.0 / system fonts). The vintage variants are written
//!   to `~/.cache/unprint/vintage-fonts/` and never checked into git or distributed.
//!   Only the transformation code (this module + era configs) is shipped.
//!
//! Design:
//!   - Early in `run()`, after scanning base fonts, we call `ensure_vintage_fonts()`
//!   - For each base font, we lookup its birth year via `font_history::font_birth_year`
//!   - If birth year is unknown => skip (prevents 1988 Comic Sans)
//!   - If birth year <= era.start_year => font existed at that era, so we generate variant
//!   - Cache path = `~/.cache/unprint/vintage-fonts/<sanitized>-<era>-<hash>.ttf`
//!   - If cached file exists and is newer than base, reuse; else regenerate
//!
//! Implemented transforms (`transform_font`, read-fonts 0.1 + write-fonts 0.3
//! FontBuilder, `kern` format 0 manipulated at the byte level):
//!   PostScript: strip GPOS, kern fmt0 only, limited to the 300 largest-|value|
//!     pairs. The 300 limit is a hardcoded truncation in this transform, NOT a
//!     measured AFM property: real AFM files carry ~3800 kern pairs
//!     (NimbusRoman-Regular.afm declares 3845 KPX entries). Kern source is the
//!     font's own kern fmt0 table when present, else pairs derived from GPOS
//!     PairPos (XAdvance), composed first-match-wins within each lookup (as a
//!     shaper applies subtables). (1985)
//!   TrueType: empty kern and GPOS (Word 6 default kerning off). (1990)
//!   OpenType: strip kern, keep GPOS as-is. The module docs mention "GPOS class
//!     kerning limited" but give no concrete limit, so no limiting is applied. (1996)
//!   Pdf14 / Docx / HarfBuzz / Variable: return original bytes unchanged
//!     (modern, full GPOS). (2001 / 2007 / 2013 / 2020)
//!   InDesignCS: DEFERRED — optical kerning simulation (outline distance + jitter)
//!     needs its own pipeline (see DEFAULT_ERAS, which omits it per Mike). Returns
//!     original bytes unchanged. (2003)
//!
//! Recreation methods (Mike Nov 12, 2026): a vintage state is recreated one
//! of three ways, and the t66 specimen states which method each section uses:
//!   Clone: use a real clone font as-is, with no byte modification. The URW
//!     Base 35 fonts (Nimbus Roman, Nimbus Mono PS, ...) are metric-compatible
//!     clones of the Adobe PostScript core fonts — NOT the original Adobe fonts
//!     — downloaded as real OTF from the URW Base 35 release (20200910,
//!     ArtifexSoftware/urw-base35-fonts). Their standard catalog font keys are
//!     the expected keys in t66.
//!   Reconstructing: modify a real font's tables to match a historical state
//!     (this module's transforms: PostScript truncates the kern table,
//!     TrueType/OpenType strip tables). Currently UNUSED by t66: the dedup
//!     rule (below) shows these transforms produce variants identical to
//!     catalog variants, so generating them adds nothing.
//!   Quantizing: REMOVED (Nov 12, 2026). The Word6 transform quantized kern
//!     values to multiples of 5, but no primary source documents Word 6.0
//!     doing this — the real Win31 kern data (113 pairs for Times, values
//!     verbatim) contradicts it. The quantized variant was replaced by the
//!     real Win31 fonts (see the t66 specimen), downloaded via the install
//!     script. Do not re-add a quantization transform without a cited source.
//!
//! Upstream redundancy (Mike Nov 12, 2026): a vintage variant is NOT generated
//! when it would be identical to a variant already in the catalog. The check
//! (`era_redundant`) runs before the transform, from the base font's tables:
//! TrueType, OpenType, and the unchanged-byte eras (Pdf14/Docx/HarfBuzz/
//! Variable/InDesignCS) are never generated — TrueType empties kern and GPOS
//! (= nokern); OpenType strips kern and keeps GPOS, but GPOS and legacy kern
//! are mutually exclusive per font (the geo cache reads legacy kern only when
//! all GPOS tables are empty), so it duplicates the kern variant when GPOS
//! kern exists and the nokern variant when it does not; the unchanged-byte
//! eras return the base font itself. PostScript generates only when the kern
//! source has >300 pairs or the font has GPOS kern. For Times New Roman (867
//! kern pairs, no GPOS) only PostScript generates; TrueType and OpenType are
//! skipped as nokern duplicates.
//!
//! Naming: readable slugs like `postscript`, `truetype`, `harfbuzz`.
//! Old cache files using year-suffixed names like `ps1985` will be orphaned
//! on upgrade — intentional migration to readable names. Delete old `*.ttf` in vintage dir
//! if you need to reclaim space.

use std::hash::Hash;
use std::path::{Path, PathBuf};

use read_fonts::types::Tag;
use read_fonts::FontRef;
use write_fonts::FontBuilder;

use crate::font_history::{era_exists_for_font, font_birth_year};
use crate::font_scan::FontEntry;

/// Era definition — historical typesetting engine generations (last 40 years).
/// Variant names are readable (postscript, truetype) — years are kept in start_year/end_year for gating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Era {
    /// PostScript Level 1 (1985-1989) — kern-table-only (AFM-style), screen≠print.
    /// Real AFM files carry ~3800 kern pairs; the 300-pair truncation in the
    /// PostScript transform is hardcoded, not an AFM property (see module docs).
    PostScript,
    /// TrueType vs Type1 war (1990-1994) — Word kerning checkbox off by default
    TrueType,
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
    /// e.g. "postscript", "truetype", "harfbuzz"
    pub fn name(&self) -> &'static str {
        match self {
            Era::PostScript => "postscript",
            Era::TrueType => "truetype",
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
            Era::PostScript => "PostScript Level 1 (1985) — kern-table-only (AFM-style), screen!=print, WYSINOTWYF",
            Era::TrueType => "TrueType vs Type1 war (1990) — Word6 kerning checkbox off by default",
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
    Era::OpenType,
    Era::Pdf14,
    Era::InDesignCS,
    Era::Docx,
    Era::HarfBuzz,
    Era::Variable,
];

/// Minimal set used by default to keep cache size reasonable.
/// Covers biggest spacing deltas: AFM-only, kern-off, modern class kern.
/// Readable names: postscript (1985 AFM), truetype (1990), docx (2007 ClearType)
/// Active set used by `main.rs` by default: all eras except InDesignCS (optical kerning).
/// Leave out InDesignCS for now per Mike — its outline-distance model needs its own pipeline.
pub const DEFAULT_ERAS: &[Era] = &[
    Era::PostScript, // AFM-only, no GPOS
    Era::TrueType,   // Word 6 kerning-off default
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

/// Vintage cache directory: ~/.cache/unprint/vintage-fonts
pub fn vintage_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("unprint").join("vintage-fonts")
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
/// Uses FxHasher for deterministic, stable hashing across runs (DefaultHasher is randomized).
fn hash_base_era(base: &FontEntry, era: Era) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
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
/// New naming uses readable slugs: e.g. `arial-postscript-a1b2c3...ttf` not `arial-ps1985-...`
/// Old files with year-suffixed slugs (`ps1985`) or path-hashed names are orphaned intentionally;
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

/// Vintage era transform using read-fonts 0.1 + write-fonts 0.3 FontBuilder.
///
/// Caching is handled by `ensure_vintage_fonts` — this function only transforms bytes.
///
/// Strategy: parse original via FontRef, collect raw table bytes, then rebuild with
/// FontBuilder, skipping or rewriting tables per era.  This avoids needing high-level
/// write-fonts table builders which don't yet exist for `kern` in 0.3. We manipulate
/// `kern` at the byte level (format 0) which is stable since 1985.
///
/// Maximum pairs a kern format 0 subtable can hold: the subtable length is a u16,
/// so 14 + n*6 <= 65535, i.e. n <= 10920. This is a hard format constraint, not a
/// heuristic. A GPOS-derived kern source can expand past this capacity —
/// GPOS class kern can expand to ~10^5 glyph pairs (SourceSerif4 Italic:
/// ~124k excl. class 0, ~363k incl.), far past what format 0 can store.
const KERN_FORMAT0_MAX_PAIRS: usize = (u16::MAX as usize - 14) / 6;

/// Kern source for eras that need a kern fmt0 table (PostScript): the font's
/// own kern format 0 subtable when present, else pairs derived from GPOS PairPos
/// XAdvance (`gpos_kern_pairs`). A font with neither source has no kerning to keep,
/// so the output has no kern table — that is correct, not a fallback.
///
/// - PostScript: strip GPOS, keep kern fmt0 only, limit to the 300 largest-|value| pairs
/// - TrueType: empty kern + GPOS
/// - OpenType: strip kern, keep GPOS as-is (no concrete class-kerning limit is specified)
/// - Pdf14/Docx/HarfBuzz/Variable: return original (modern, full GPOS)
/// - InDesignCS: DEFERRED (optical kerning simulation needs its own pipeline);
///   returns original bytes unchanged, like the modern eras.
///
/// For OTF/CFF fonts (sfnt version 'OTTO'), we patch the built font's first 4 bytes
/// to preserve 'OTTO', because FontBuilder hardcodes 0x00010000.
///
/// Errors are returned loudly (no silent fallback to original bytes) when the kern
/// source cannot be parsed or the rebuilt font fails to parse, so a broken vintage
/// variant is never cached.
fn transform_font(base_bytes: &[u8], era: Era, _family: &str) -> Result<Vec<u8>, String> {
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

    // Build filtered/rebuilt table list
    let mut out_tables: Vec<(Tag, Vec<u8>)> = Vec::with_capacity(orig_tables.len());

    // Pre-parse kern if needed
    let kern_entry = orig_tables.iter().find(|(t, _)| *t == tag_kern).map(|(_, d)| d.clone());
    let gpos_entry = orig_tables.iter().find(|(t, _)| *t == tag_gpos).map(|(_, d)| d.clone());
    let n_glyphs = num_glyphs(&font_ref);

    // Resolve the kern source for eras that need a kern fmt0 table (PostScript).
    // Preference: the font's own kern format 0 subtable. Fallback: GPOS PairPos XAdvance.
    let kern_src = resolve_kern_source(kern_entry.as_deref(), gpos_entry.as_deref(), n_glyphs);

    let (new_kern, drop_kern, drop_gpos, keep_orig_kern) = match era {
        Era::PostScript => {
            // Keep kern (truncated to the 300 largest-|value| pairs), drop GPOS.
            match kern_src {
                KernSource::Pairs(pairs, coverage) => {
                    let truncated = truncate_pairs(pairs, 300);
                    match build_kern_table(&truncated, coverage) {
                        Some(t) => (Some(t), false, true, false),
                        None => return Err("kern table too large for format 0".to_string()),
                    }
                }
                KernSource::KeepOriginal => (None, false, true, true),
                KernSource::None => (None, false, true, false),
            }
        }
        Era::TrueType => {
            // Empty kern and GPOS — Word 6 checkbox off
            (None, true, true, false)
        }
        Era::OpenType => {
            // OT 1.0: GPOS class kerning, kern stripped
            (None, true, false, false)
        }
        _ => (None, false, false, false), // unreachable due to early return
    };

    let mut saw_kern = false;
    for (tag, data) in orig_tables {
        if tag == tag_gpos && drop_gpos {
            continue;
        }
        if tag == tag_kern {
            saw_kern = true;
            if drop_kern {
                continue;
            }
            if keep_orig_kern {
                // Kern table is not format 0; keep its bytes as-is (warned at resolve time).
                out_tables.push((tag, data));
                continue;
            }
            if let Some(ref new_data) = new_kern {
                out_tables.push((tag, new_data.clone()));
                continue;
            }
            // No kern table for this era (no kerning source).
            continue;
        }
        out_tables.push((tag, data));
    }
    // The kern source may be GPOS-derived for a font that had no kern table at all;
    // in that case the rebuilt table must be added explicitly.
    if let Some(ref new_data) = new_kern {
        if !saw_kern {
            out_tables.push((tag_kern, new_data.clone()));
        }
    }

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

    // Validate output parses. A rebuilt font that does not parse is a bug in the
    // transform — return an error loudly so the caller never caches a broken variant.
    if FontRef::new(&out).is_err() {
        return Err(format!("rebuilt {} font failed to parse", era.name()));
    }

    Ok(out)
}

/// Number of glyphs in the font, from the maxp table (numGlyphs is a u16 at offset 4).
/// Returns 0 if maxp is missing or malformed; callers treat 0 as "unknown" and skip
/// class-kern expansion that needs it.
fn num_glyphs(font_ref: &FontRef) -> usize {
    let Some(data) = font_ref.table_data(Tag::new(b"maxp")) else {
        return 0;
    };
    let b = data.as_ref();
    if b.len() < 6 {
        return 0;
    }
    u16::from_be_bytes([b[4], b[5]]) as usize
}

/// Coverage value for a horizontal kern format 0 subtable (format 0, horizontal).
const KERN_COVERAGE_HORIZONTAL: u16 = 0x0001;

/// How an era that needs a kern fmt0 table obtains its pairs.
enum KernSource {
    /// (pairs, coverage) to build a fresh kern fmt0 table from. Pairs are sorted
    /// by (left, right) and contain no zero values.
    Pairs(Vec<(u16, u16, i16)>, u16),
    /// A kern table exists but is not format 0; keep its bytes as-is (warned at resolve time).
    KeepOriginal,
    /// No kerning source; the output has no kern table.
    None,
}

/// Resolve the kern source for PostScript: the font's own kern format 0
/// subtable first, else pairs derived from GPOS PairPos XAdvance. A font with
/// neither source has no kerning, so `KernSource::None` is correct — not a fallback.
fn resolve_kern_source(
    kern_entry: Option<&[u8]>,
    gpos_entry: Option<&[u8]>,
    num_glyphs: usize,
) -> KernSource {
    if let Some(kdata) = kern_entry {
        if let Some((coverage, pairs)) = parse_kern_format0(kdata) {
            return KernSource::Pairs(pairs, coverage);
        }
        // Not a format 0 subtable (e.g. format 2) — try GPOS before giving up.
        if let Some(pairs) = gpos_entry.and_then(|g| gpos_kern_pairs(g, num_glyphs)) {
            if !pairs.is_empty() {
                return KernSource::Pairs(pairs, KERN_COVERAGE_HORIZONTAL);
            }
        }
        eprintln!("[vintage] warning: kern table is not format 0 and no GPOS kern pairs found; keeping original kern table as-is");
        return KernSource::KeepOriginal;
    }
    if let Some(pairs) = gpos_entry.and_then(|g| gpos_kern_pairs(g, num_glyphs)) {
        if !pairs.is_empty() {
            return KernSource::Pairs(pairs, KERN_COVERAGE_HORIZONTAL);
        }
    }
    KernSource::None
}

/// Keep the `limit` largest-|value| pairs (the pairs that actually move spacing),
/// re-sorted by (left, right) as a kern format 0 table requires. Deterministic:
/// ties on |value| break by (left, right). The 300-pair limit used by the
/// PostScript transform is a hardcoded truncation, NOT a measured AFM property:
/// real AFM files carry ~3800 kern pairs (NimbusRoman-Regular.afm declares 3845).
fn truncate_pairs(mut pairs: Vec<(u16, u16, i16)>, limit: usize) -> Vec<(u16, u16, i16)> {
    if pairs.len() <= limit {
        pairs.sort_unstable();
        return pairs;
    }
    pairs.sort_unstable_by(|a, b| {
        b.2.unsigned_abs()
            .cmp(&a.2.unsigned_abs())
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });
    pairs.truncate(limit);
    pairs.sort_unstable();
    pairs
}

// ---------------------------------------------------------------------------
// GPOS -> kern extraction (byte level, modeled on geo_cache::parse_gpos_inner)
// ---------------------------------------------------------------------------

/// Bounds-checked big-endian readers for the GPOS byte parser below.
fn gbu16(data: &[u8], off: usize) -> Option<u16> {
    data.get(off..off + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
}
fn gbi16(data: &[u8], off: usize) -> Option<i16> {
    gbu16(data, off).map(|v| v as i16)
}
fn gbu32(data: &[u8], off: usize) -> Option<u32> {
    data.get(off..off + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Size in bytes of a ValueRecord with the given ValueFormat bitmask.
fn gpos_value_format_size(fmt: u16) -> usize {
    let mut sz = 0usize;
    for bit in [0x0001u16, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080] {
        if fmt & bit != 0 {
            sz += 2;
        }
    }
    sz
}

/// XAdvance field of a ValueRecord at `off` with the given ValueFormat, or 0 if
/// the format has no XAdvance (bit 0x0004). Device-table offsets are skipped (2 bytes).
fn gpos_value_xadvance(data: &[u8], off: usize, fmt: u16) -> Option<i16> {
    let mut p = off;
    let mut xad = 0i16;
    // Field order: XPlacement, YPlacement, XAdvance, YAdvance, then 4 device offsets.
    if fmt & 0x0001 != 0 {
        p += 2;
    }
    if fmt & 0x0002 != 0 {
        p += 2;
    }
    if fmt & 0x0004 != 0 {
        xad = gbi16(data, p)?;
        p += 2;
    }
    // Remaining fields (YAdvance + device offsets) do not affect XAdvance; no need to read them.
    let _ = p;
    Some(xad)
}

/// Parse a Coverage table at `off`. Format 1 is an explicit glyph list; format 2 is
/// a list of (start, end) ranges. Returns glyph IDs sorted and deduped.
fn gpos_parse_coverage(data: &[u8], off: usize) -> Option<Vec<u16>> {
    let fmt = gbu16(data, off)?;
    if fmt == 1 {
        let count = gbu16(data, off + 2)? as usize;
        let mut v = Vec::with_capacity(count);
        for i in 0..count {
            v.push(gbu16(data, off + 4 + i * 2)?);
        }
        Some(v)
    } else if fmt == 2 {
        let range_count = gbu16(data, off + 2)? as usize;
        let mut v = Vec::new();
        for i in 0..range_count {
            let base = off + 4 + i * 6;
            let start = gbu16(data, base)?;
            let end = gbu16(data, base + 2)?;
            for gid in start..=end {
                v.push(gid);
            }
        }
        v.sort_unstable();
        v.dedup();
        Some(v)
    } else {
        None
    }
}

/// Parse a ClassDef table at `off` into a class-per-glyph array of length `num_glyphs`
/// (glyphs not listed are class 0). Format 1 is startGlyphID + class values;
/// format 2 is a list of (start, end, class) ranges.
fn gpos_parse_class_def(data: &[u8], off: usize, num_glyphs: usize) -> Option<Vec<u16>> {
    let fmt = gbu16(data, off)?;
    let mut classes = vec![0u16; num_glyphs];
    if fmt == 1 {
        let start = gbu16(data, off + 2)? as usize;
        let count = gbu16(data, off + 4)? as usize;
        for i in 0..count {
            let gid = start + i;
            if gid >= num_glyphs {
                break;
            }
            classes[gid] = gbu16(data, off + 6 + i * 2)?;
        }
        Some(classes)
    } else if fmt == 2 {
        let range_count = gbu16(data, off + 2)? as usize;
        for i in 0..range_count {
            let base = off + 4 + i * 6;
            let start = gbu16(data, base)? as usize;
            let end = gbu16(data, base + 2)? as usize;
            let cls = gbu16(data, base + 4)?;
            for gid in start..=end {
                if gid < num_glyphs {
                    classes[gid] = cls;
                }
            }
        }
        Some(classes)
    } else {
        None
    }
}

/// Extract effective XAdvance kerning pairs from a GPOS table (raw bytes).
///
/// Only PairPos (LookupType 2, including ExtensionPos LookupType 9 wrapping a
/// PairPos) contributes: kern = ValueRecord1.XAdvance + ValueRecord2.XAdvance,
/// the same convention the geometry predictor uses. SinglePos, mark, cursive and
/// contextual lookups do not produce pair kerning and are ignored; XPlacement is
/// not representable in a kern table and is ignored.
///
/// Feature gating follows the geometry predictor: lookups referenced only by the
/// `case`/`cpsp` features are excluded; lookups referenced by any other feature
/// (kern, mark, ...) are kept. A lookup shared between `case` and a default
/// feature stays in the default set.
///
/// Class pairs (Format 2) are expanded to glyph pairs. Within one lookup, subtables
/// are alternatives: the first subtable whose coverage contains the first glyph
/// claims it (first-match-wins, as a shaper applies it). When the same (left, right)
/// pair appears in more than one lookup, the later lookup wins (HashMap insert).
/// Zero values are dropped. The result is sorted by (left, right).
///
/// Returns None if the GPOS data is malformed (caller falls back to the kern table
/// or to no kerning); returns Some(vec) — possibly empty — when parsing succeeds.
fn gpos_kern_pairs(gpos_data: &[u8], num_glyphs: usize) -> Option<Vec<(u16, u16, i16)>> {
    if gpos_data.len() < 10 {
        return None;
    }
    let feature_list_off = gbu16(gpos_data, 6)? as usize;
    let lookup_list_off = gbu16(gpos_data, 8)? as usize;
    if lookup_list_off + 2 > gpos_data.len() {
        return None;
    }

    // FeatureList: split lookups into case/cpsp-only vs default-eligible.
    let mut case_lookups: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut default_lookups: std::collections::HashSet<usize> = std::collections::HashSet::new();
    if feature_list_off + 2 <= gpos_data.len() {
        if let Some(feature_count) = gbu16(gpos_data, feature_list_off) {
            let fc = feature_count as usize;
            for fi in 0..fc {
                let rec_off = feature_list_off + 2 + fi * 6;
                if rec_off + 6 > gpos_data.len() {
                    continue;
                }
                let tag = std::str::from_utf8(&gpos_data[rec_off..rec_off + 4]).unwrap_or("");
                let feat_off = gbu16(gpos_data, rec_off + 4)? as usize + feature_list_off;
                if feat_off + 4 > gpos_data.len() {
                    continue;
                }
                let lookup_count = gbu16(gpos_data, feat_off + 2)? as usize;
                for li in 0..lookup_count {
                    let pos = feat_off + 4 + li * 2;
                    if pos + 2 > gpos_data.len() {
                        break;
                    }
                    if let Some(lid) = gbu16(gpos_data, pos) {
                        let lid = lid as usize;
                        if tag == "case" || tag == "cpsp" {
                            case_lookups.insert(lid);
                        } else {
                            default_lookups.insert(lid);
                        }
                    }
                }
            }
        }
    }
    let case_only: std::collections::HashSet<usize> = case_lookups
        .difference(&default_lookups)
        .cloned()
        .collect();

    let lookup_count = gbu16(gpos_data, lookup_list_off)? as usize;
    let mut pairs: std::collections::HashMap<(u16, u16), i16> = std::collections::HashMap::new();

    for li in 0..lookup_count {
        if case_only.contains(&li) {
            continue;
        }
        let off_pos = lookup_list_off + 2 + li * 2;
        let Some(lookup_off) = gbu16(gpos_data, off_pos).map(|o| o as usize + lookup_list_off)
        else {
            continue;
        };
        if lookup_off + 6 > gpos_data.len() {
            continue;
        }
        let Some(lookup_type) = gbu16(gpos_data, lookup_off) else {
            continue;
        };
        if lookup_type != 2 && lookup_type != 9 {
            continue;
        }
        // Within one lookup, subtables are alternatives: the FIRST subtable whose
        // coverage contains a first glyph claims that glyph, and later subtables
        // in the same lookup must not override it (OpenType first-match-wins;
        // HarfBuzz applies the first matching subtable). A last-wins dedupe here
        // is wrong: for SourceSerif4 Italic it collapses the kern set to ~7.6k
        // pairs, while first-wins gives the shaper's ~109.5k. Claiming is by
        // coverage membership, not by nonzero value — a subtable claims a glyph
        // even when its kern value for that glyph is zero.
        let mut claimed: std::collections::HashSet<u16> = std::collections::HashSet::new();
        let Some(subtable_count) = gbu16(gpos_data, lookup_off + 4).map(|v| v as usize) else {
            continue;
        };
        for si in 0..subtable_count {
            let sub_off_pos = lookup_off + 6 + si * 2;
            let Some(sub_off) = gbu16(gpos_data, sub_off_pos).map(|o| o as usize + lookup_off)
            else {
                continue;
            };
            if sub_off + 2 > gpos_data.len() {
                continue;
            }
            // Unwrap ExtensionPos (Format 1, ExtensionLookupType 2 = PairPos).
            let mut inner_off = sub_off;
            if lookup_type == 9 {
                if sub_off + 8 > gpos_data.len() {
                    continue;
                }
                let Some(ext_fmt) = gbu16(gpos_data, sub_off) else {
                    continue;
                };
                if ext_fmt != 1 {
                    continue;
                }
                let Some(ext_type) = gbu16(gpos_data, sub_off + 2) else {
                    continue;
                };
                if ext_type != 2 {
                    continue;
                }
                let Some(ext_offset) = gbu32(gpos_data, sub_off + 4).map(|o| o as usize) else {
                    continue;
                };
                inner_off = sub_off + ext_offset;
                if inner_off + 2 > gpos_data.len() {
                    continue;
                }
            }
            let Some(fmt) = gbu16(gpos_data, inner_off) else {
                continue;
            };
            if fmt == 1 {
                // PairPos Format 1: coverage + pair sets of (second_gid, value1, value2).
                if inner_off + 10 > gpos_data.len() {
                    continue;
                }
                let Some(cov_off) =
                    gbu16(gpos_data, inner_off + 2).map(|o| o as usize + inner_off)
                else {
                    continue;
                };
                let Some(val_fmt1) = gbu16(gpos_data, inner_off + 4) else {
                    continue;
                };
                let Some(val_fmt2) = gbu16(gpos_data, inner_off + 6) else {
                    continue;
                };
                let Some(pair_set_count) =
                    gbu16(gpos_data, inner_off + 8).map(|v| v as usize)
                else {
                    continue;
                };
                let Some(coverage) = gpos_parse_coverage(gpos_data, cov_off) else {
                    continue;
                };
                let val1_size = gpos_value_format_size(val_fmt1);
                let val2_size = gpos_value_format_size(val_fmt2);
                let rec_size = 2 + val1_size + val2_size;
                for psi in 0..pair_set_count {
                    let Some(first_gid) = coverage.get(psi).copied() else {
                        continue;
                    };
                    // First-match-wins: skip first glyphs already claimed by an
                    // earlier subtable in this lookup. Claim on coverage
                    // membership, even if this PairSet yields no nonzero pairs.
                    if !claimed.insert(first_gid) {
                        continue;
                    }
                    let ps_off_pos = inner_off + 10 + psi * 2;
                    let Some(ps_off) =
                        gbu16(gpos_data, ps_off_pos).map(|o| o as usize + inner_off)
                    else {
                        continue;
                    };
                    if ps_off + 2 > gpos_data.len() {
                        continue;
                    }
                    let Some(pair_val_count) = gbu16(gpos_data, ps_off).map(|v| v as usize)
                    else {
                        continue;
                    };
                    let mut rp = ps_off + 2;
                    for _ in 0..pair_val_count {
                        if rp + rec_size > gpos_data.len() {
                            break;
                        }
                        let Some(second_gid) = gbu16(gpos_data, rp) else {
                            break;
                        };
                        let Some(v1) = gpos_value_xadvance(gpos_data, rp + 2, val_fmt1) else {
                            break;
                        };
                        let Some(v2) =
                            gpos_value_xadvance(gpos_data, rp + 2 + val1_size, val_fmt2)
                        else {
                            break;
                        };
                        let kern = (v1 as i32 + v2 as i32)
                            .clamp(i16::MIN as i32, i16::MAX as i32)
                            as i16;
                        if kern != 0 {
                            pairs.insert((first_gid, second_gid), kern);
                        }
                        rp += rec_size;
                    }
                }
            } else if fmt == 2 {
                // PairPos Format 2: coverage + class defs + class matrix.
                if num_glyphs == 0 {
                    continue;
                }
                if inner_off + 16 > gpos_data.len() {
                    continue;
                }
                let Some(cov_off) =
                    gbu16(gpos_data, inner_off + 2).map(|o| o as usize + inner_off)
                else {
                    continue;
                };
                let Some(val_fmt1) = gbu16(gpos_data, inner_off + 4) else {
                    continue;
                };
                let Some(val_fmt2) = gbu16(gpos_data, inner_off + 6) else {
                    continue;
                };
                let Some(cd1_off) =
                    gbu16(gpos_data, inner_off + 8).map(|o| o as usize + inner_off)
                else {
                    continue;
                };
                let Some(cd2_off) =
                    gbu16(gpos_data, inner_off + 10).map(|o| o as usize + inner_off)
                else {
                    continue;
                };
                let Some(c1_count) = gbu16(gpos_data, inner_off + 12).map(|v| v as usize)
                else {
                    continue;
                };
                let Some(c2_count) = gbu16(gpos_data, inner_off + 14).map(|v| v as usize)
                else {
                    continue;
                };
                let Some(coverage) = gpos_parse_coverage(gpos_data, cov_off) else {
                    continue;
                };
                let Some(class1) = gpos_parse_class_def(gpos_data, cd1_off, num_glyphs) else {
                    continue;
                };
                let Some(class2) = gpos_parse_class_def(gpos_data, cd2_off, num_glyphs) else {
                    continue;
                };
                // First-match-wins: this subtable claims every first glyph in its
                // coverage that no earlier subtable in this lookup claimed. Only
                // newly claimed glyphs get pairs from this subtable.
                let mut new_g1: std::collections::HashSet<u16> =
                    std::collections::HashSet::new();
                for &g1 in coverage.iter() {
                    if claimed.insert(g1) {
                        new_g1.insert(g1);
                    }
                }
                // Invert ClassDef2 into class -> glyphs.
                let mut class2_glyphs: Vec<Vec<u16>> = vec![Vec::new(); c2_count];
                for (gid, &cls) in class2.iter().enumerate() {
                    let cls = cls as usize;
                    if cls < c2_count {
                        class2_glyphs[cls].push(gid as u16);
                    }
                }
                let val1_size = gpos_value_format_size(val_fmt1);
                let val2_size = gpos_value_format_size(val_fmt2);
                let rec_size = val1_size + val2_size;
                let mut p = inner_off + 16;
                // Read the matrix row-major: [class1][class2].
                for cl1 in 0..c1_count {
                    for cl2 in 0..c2_count {
                        if p + rec_size > gpos_data.len() {
                            break;
                        }
                        let Some(v1) = gpos_value_xadvance(gpos_data, p, val_fmt1) else {
                            break;
                        };
                        let Some(v2) = gpos_value_xadvance(gpos_data, p + val1_size, val_fmt2)
                        else {
                            break;
                        };
                        let kern = (v1 as i32 + v2 as i32)
                            .clamp(i16::MIN as i32, i16::MAX as i32)
                            as i16;
                        if kern != 0 {
                            for &g1 in coverage.iter() {
                                if !new_g1.contains(&g1) {
                                    continue;
                                }
                                if (g1 as usize) < num_glyphs && class1[g1 as usize] as usize == cl1
                                {
                                    for &g2 in &class2_glyphs[cl2] {
                                        pairs.insert((g1, g2), kern);
                                    }
                                }
                            }
                        }
                        p += rec_size;
                    }
                }
            }
        }
    }

    let mut out: Vec<(u16, u16, i16)> = pairs
        .into_iter()
        .map(|((l, r), v)| (l, r, v))
        .collect();
    out.sort_unstable();
    Some(out)
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

/// Build a kern format 0 table (version 0, one subtable) from sorted (left, right, value)
/// pairs. Returns None if the pair count cannot fit in a u16 subtable length
/// (14 + n*6 bytes must fit in u16, so n <= KERN_FORMAT0_MAX_PAIRS).
fn build_kern_table(pairs: &[(u16, u16, i16)], coverage: u16) -> Option<Vec<u8>> {
    let n = pairs.len();
    // subtable length is u16: 14 + n*6 <= 65535
    if n > KERN_FORMAT0_MAX_PAIRS {
        return None;
    }
    let (search_range, entry_selector, range_shift) = if n == 0 {
        (0u16, 0u16, 0u16)
    } else {
        // largest power of 2 <= n
        let mut pow2 = 1usize;
        while pow2 * 2 <= n {
            pow2 *= 2;
        }
        let sr = (pow2 * 6) as u16;
        let es = pow2.trailing_zeros() as u16; // log2(pow2), exact integer math
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
    Some(out)
}

/// Matching-relevant kerning content of a font file: the dimensions of the
/// `kern` table the vintage transform can change, as the pipeline sees them.
///
/// The transform only ever touches two things: the `kern` table (drop /
/// truncate / rebuild) and GPOS presence (drop / keep). Every
/// other table — outlines, metrics, cmap, GSUB — passes through
/// byte-identical, so kerning is the only matching-relevant dimension a
/// vintage variant can differ on.
#[derive(Debug, PartialEq, Eq)]
enum KernContent {
    /// No `kern` table.
    Absent,
    /// Single format-0 subtable: its (left, right, value) pairs in table
    /// order. A rebuild that changes no pair compares equal even if the
    /// rebuilt header bytes differ from the original.
    Fmt0Pairs(Vec<(u16, u16, i16)>),
    /// Anything else (multi-subtable, non-format-0): raw table bytes.
    /// Compared opaquely — any byte change counts as distinct, which is
    /// conservative.
    Opaque(Vec<u8>),
}

fn kern_content(kern_data: Option<&[u8]>) -> KernContent {
    let Some(data) = kern_data else {
        return KernContent::Absent;
    };
    let n_tables = if data.len() >= 4 {
        u16::from_be_bytes([data[2], data[3]])
    } else {
        0
    };
    if n_tables == 1 {
        if let Some((_coverage, pairs)) = parse_kern_format0(data) {
            return KernContent::Fmt0Pairs(pairs);
        }
    }
    KernContent::Opaque(data.to_vec())
}

/// True if the font's GPOS table has a `kern` feature with at least one
/// lookup (same check as `font_scan::font_has_kern_feature`). A font with
/// GPOS pair kerning never consults the legacy `kern` table: the geometry
/// cache only falls back to `kern` when GPOS yields no pair tables
/// (geo_cache.rs), and HarfBuzz shaping follows the same OpenType
/// precedence. Unparseable or absent GPOS returns false.
fn gpos_has_kern_feature(face: &unprint_fonts::ttf_parser::Face) -> bool {
    let tables = face.tables();
    let Some(gpos) = &tables.gpos else {
        return false;
    };
    let kern_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"kern");
    if let Some(feature) = gpos.features.find(kern_tag) {
        return !feature.lookup_indices.is_empty();
    }
    false
}

/// Returns true if the transformed font `out_bytes` differs from its base
/// `base_bytes` in matching-relevant terms — i.e. whether the vintage
/// variant deserves to exist as a separate catalog entry.
///
/// The vintage transform only changes kerning, so the variant is distinct
/// exactly when the effective kerning changes:
///
///   DISTINCT iff GPOS presence differs,
///          OR the base has no GPOS `kern` feature and the kern-table
///             content differs (pairs dropped by truncation, or the
///             table dropped).
///
/// Consequences: the modern eras (Pdf14, InDesignCS, Docx, HarfBuzz,
/// Variable) return the base bytes unchanged and are never distinct; a
/// kern truncation is a no-op on a font with fewer than 300 pairs;
/// stripping the kern table is a no-op on a font with no kern table; and
/// any kern-table change is invisible on a font with GPOS pair kerning.
///
/// Unparseable input fails open (returns true): a font we cannot read is
/// kept rather than pruned on a guess.
pub fn vintage_transform_distinct(base_bytes: &[u8], out_bytes: &[u8]) -> bool {
    use unprint_fonts::ttf_parser::Tag;
    let Ok(base_face) = unprint_fonts::ttf_parser::Face::parse(base_bytes, 0) else {
        return true;
    };
    let Ok(out_face) = unprint_fonts::ttf_parser::Face::parse(out_bytes, 0) else {
        return true;
    };
    // GPOS presence: the transform drops GPOS wholesale or keeps it
    // wholesale; it never modifies GPOS content.
    let base_gpos = base_face.raw_face().table(Tag::from_bytes(b"GPOS")).is_some();
    let out_gpos = out_face.raw_face().table(Tag::from_bytes(b"GPOS")).is_some();
    if base_gpos != out_gpos {
        return true;
    }
    // With GPOS pair kerning present, the legacy kern table is invisible
    // to the pipeline and GPOS content is untouched by the transform, so
    // nothing matching-relevant can differ.
    if gpos_has_kern_feature(&base_face) {
        return false;
    }
    // Otherwise the legacy kern table is the kerning source: compare it.
    let base_kern = base_face.raw_face().table(Tag::from_bytes(b"kern"));
    let out_kern = out_face.raw_face().table(Tag::from_bytes(b"kern"));
    kern_content(base_kern) != kern_content(out_kern)
}

fn rebuild_kern_truncate(orig: &[u8], limit: usize) -> Option<Vec<u8>> {
    let (coverage, pairs) = parse_kern_format0(orig)?;
    let truncated = truncate_pairs(pairs, limit);
    build_kern_table(&truncated, coverage)
}

/// Kerning facts about a base font, parsed once per font for the upstream
/// redundancy check (`era_redundant`). The kern source resolution mirrors
/// `resolve_kern_source`'s preference (own kern fmt0 first, else
/// GPOS-derived pairs) but gathers facts instead of building a table, so
/// the check runs before any transform is attempted.
struct BaseKernFacts {
    /// Number of pairs in the resolved kern source (own kern fmt0, else
    /// GPOS-derived). None when there is no kern source, or when the kern
    /// table is present but not format 0 (`kern_keep_original`).
    kern_src_pair_count: Option<usize>,
    /// The kern table exists but is not format 0, so the transform keeps
    /// its bytes as-is (no truncation).
    kern_keep_original: bool,
    /// The base font has GPOS kern (at least one non-zero PairPos pair).
    has_gpos_kern: bool,
}

/// Parse a base font's kerning facts from its bytes. Returns None when the
/// font does not parse (the caller then treats the era as not redundant and
/// lets `transform_font` fail loudly, as before).
fn base_kern_facts(base_bytes: &[u8]) -> Option<BaseKernFacts> {
    let font_ref = FontRef::new(base_bytes).ok()?;
    let tag_kern = Tag::new(b"kern");
    let tag_gpos = Tag::new(b"GPOS");
    let kern_entry = font_ref
        .table_data(tag_kern)
        .map(|d| d.as_ref().to_vec());
    let gpos_entry = font_ref
        .table_data(tag_gpos)
        .map(|d| d.as_ref().to_vec());
    let n_glyphs = num_glyphs(&font_ref);

    // GPOS kern: at least one non-zero PairPos pair (a zero pair moves no spacing).
    let gpos_pairs = gpos_entry
        .as_deref()
        .and_then(|g| gpos_kern_pairs(g, n_glyphs))
        .unwrap_or_default();
    let has_gpos_kern = gpos_pairs.iter().any(|(_, _, v)| *v != 0);

    // Legacy kern fmt0 pairs (first subtable, as the transform reads it).
    let kern_fmt0 = kern_entry
        .as_deref()
        .and_then(parse_kern_format0);

    // Resolved kern source, mirroring resolve_kern_source's preference.
    let (kern_src_pair_count, kern_keep_original) =
        if let Some((_, pairs)) = kern_fmt0 {
            (Some(pairs.len()), false)
        } else if !gpos_pairs.is_empty() {
            (Some(gpos_pairs.len()), false)
        } else if kern_entry.is_some() {
            // Kern table exists but is not format 0; the transform keeps it as-is.
            (None, true)
        } else {
            (None, false)
        };

    Some(BaseKernFacts {
        kern_src_pair_count,
        kern_keep_original,
        has_gpos_kern,
    })
}

/// Whether the (font, era) vintage variant would be redundant with a variant
/// already in the catalog, so it must not be generated (Mike Nov 12, 2026:
/// "We don't need to make vintage fonts that are identical to variants we
/// already put in the catalog — like no kern"). Checked UPSTREAM from the
/// base font's tables and the era transform's definition, before calling
/// the transform — the variant is never generated and then inspected.
///
/// - TrueType: always redundant — it empties kern and GPOS, so the result
///   has no kerning, which is exactly the nokern catalog variant.
/// - OpenType: always redundant — it strips kern and keeps GPOS, but GPOS
///   and legacy kern are mutually exclusive per font: the geo cache reads
///   the legacy kern table only when all GPOS tables are empty
///   (geo_cache.rs cache-build path), so the pipeline never applies both.
///   For a font with GPOS kern, the catalog kern variant already applies
///   GPOS only (the kern table is ignored), so the OpenType variant is
///   identical to it. For a font without GPOS kern, stripping kern leaves
///   no kerning, so the OpenType variant is identical to nokern.
/// - Pdf14 / Docx / HarfBuzz / Variable / InDesignCS: always redundant —
///   they return the base bytes unchanged, so the variant is identical to
///   the catalog's kern (or nokern) variant for that font.
/// - PostScript: redundant unless the kern source has more than 300 pairs
///   or the font has GPOS kern. With <=300 pairs and no GPOS kern, the
///   truncation and the GPOS strip are both no-ops, so the result is the
///   catalog kern variant.
fn era_redundant(era: Era, facts: &BaseKernFacts) -> bool {
    match era {
        Era::TrueType | Era::OpenType => true,
        Era::Pdf14 | Era::InDesignCS | Era::Docx | Era::HarfBuzz | Era::Variable => true,
        Era::PostScript => {
            // GPOS kern present: stripping GPOS changes the font.
            if facts.has_gpos_kern {
                return false;
            }
            // Non-fmt0 kern kept as-is and no GPOS to strip: the result is
            // the base font itself, i.e. the catalog kern variant.
            if facts.kern_keep_original {
                return true;
            }
            match facts.kern_src_pair_count {
                // <=300 pairs: truncation is a no-op, GPOS strip is a
                // no-op (no GPOS kern), so the result is the catalog kern
                // variant. No kern source at all: no kerning, i.e. nokern.
                Some(n) => n <= 300,
                None => true,
            }
        }
    }
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
    let mut skipped_duplicate = 0usize;
    let mut skipped_redundant = 0usize;
    let mut reused = 0usize;
    let mut created = 0usize;

    // Avoid generating vintage from vintage (prevent recursion)
    let dir_canon = dir.canonicalize().unwrap_or(dir.clone());

    for &base in &base_fonts_deduped {
        // Skip if base is already inside vintage cache dir
        if base.path.starts_with(&dir_canon) || base.path.to_string_lossy().contains("/vintage/") {
            continue;
        }
        // Skip OT feature variants to keep cache small; allow the kerned
        // base entry and wght instances. Kern/nokern and lig/nolig are
        // explicit in every font key now, so the base entry carries
        // |kern|lig ("" for legacy entries) — the old empty-tag check
        // would skip every base entry.
        let tag = base.variant_tag.as_str();
        let is_wght = tag.split('|').any(|seg| seg.starts_with("wght"));
        if !(tag.is_empty() || tag == "kern" || tag == "kern|lig" || is_wght) {
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

        // Read the base font once per font (not once per era): the upstream
        // redundancy check needs its tables, and generation reuses the bytes.
        let base_bytes = match std::fs::read(&base.path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[vintage] warning: could not read base {}: {e}", base.path.display());
                continue;
            }
        };
        // Kerning facts for the upstream redundancy check. None when the
        // font does not parse — then no era is skipped as redundant and
        // transform_font fails loudly per era, as before.
        let kern_facts = base_kern_facts(&base_bytes);

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
            // 3) Upstream redundancy — do not create a vintage variant
            // identical to a catalog variant (Mike Nov 12). Checked from
            // the base font's tables before generating, and before reusing
            // a stale cache file, so redundant variants are never loaded.
            if let Some(facts) = &kern_facts {
                if era_redundant(era, facts) {
                    skipped_redundant += 1;
                    continue;
                }
            }

            let cache_path = vintage_cache_path(base, era);
            if cache_valid(&cache_path, &base.path) {
                reused += 1;
                generated.push(cache_path);
                continue;
            }

            // Need to generate (base_bytes read once above, reused here).
            match transform_font(&base_bytes, era, &base.family_name) {
                Ok(out_bytes) => {
                    // Uniqueness pruning: skip variants whose transform
                    // changes nothing matching-relevant (kern/GPOS identical
                    // to the base). Such entries are pure duplicates of the
                    // modern entry in the catalog and can only confuse
                    // matching via tie-breaks.
                    if !vintage_transform_distinct(&base_bytes, &out_bytes) {
                        skipped_duplicate += 1;
                        continue;
                    }
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

    eprintln!("[vintage] summary: created={} reused={} skipped_unknown={} skipped_too_new={} skipped_not_shipped={} skipped_redundant={} skipped_duplicate={} total_cached={} scanned_bases={} (deduped from {})",
        created, reused, skipped_unknown, skipped_too_new, skipped_not_shipped, skipped_redundant, skipped_duplicate, generated.len(), base_fonts_deduped.len(), base_fonts.len());

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
            // "liberationsans-postscript-abc.ttf" would become "liberationsans postscript abc" – wrong.
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
        // Recompute cached font_key to include |vintage=ERA and bracket notation for correct dedup/sorting (perf slice 8 cache)
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
            font_key_cache: "dejavusans-400".to_string(),
            glyph_overrides: None,
            variations: None,
            typographic_family: "DejaVu Sans".to_string(),
            vintage_era: None,
            opsz_range: None,
            has_gpos: false,
            has_kern: false,
            font_version: "1.00".to_string(),
        };
        let p1 = vintage_cache_path(&fe, Era::PostScript);
        let p2 = vintage_cache_path(&fe, Era::PostScript);
        // sanitize_family("DejaVu Sans") is "dejavu_sans" (space -> '_').
        assert!(p1.to_string_lossy().contains("dejavu_sans"));
        assert!(p1.to_string_lossy().contains("postscript"));
        assert_eq!(p1.parent(), p2.parent());
    }

    #[test]
    fn test_names_readable() {
        assert_eq!(Era::PostScript.name(), "postscript");
        assert_eq!(Era::TrueType.name(), "truetype");
        assert_eq!(Era::OpenType.name(), "opentype");
        assert_eq!(Era::Pdf14.name(), "pdf14");
        assert_eq!(Era::InDesignCS.name(), "indesign_cs");
        assert_eq!(Era::Docx.name(), "docx");
        assert_eq!(Era::HarfBuzz.name(), "harfbuzz");
        assert_eq!(Era::Variable.name(), "variable");
        // years still accessible for gating
        assert_eq!(Era::PostScript.start_year(), 1985);
        assert_eq!(Era::Docx.start_year(), 2007);
    }

    #[test]
    fn test_shipping_respects_readable_names() {
        // TrueType should contain arial etc, NOT calibri
        assert!(was_shipped_with("Arial", Era::TrueType));
        assert!(!was_shipped_with("Calibri", Era::TrueType));
        assert!(was_shipped_with("Calibri", Era::Docx));
        assert!(!was_shipped_with("Comic Sans MS", Era::PostScript));
        assert!(was_shipped_with("Comic Sans MS", Era::OpenType));
    }

    #[test]
    fn test_kern_roundtrip() {
        let pairs = vec![(10u16, 20u16, -15i16), (30, 40, 25), (50, 60, 0)];
        let data = build_kern_table(&pairs, 0x0001).expect("build kern table");
        let parsed = parse_kern_format0(&data).expect("parse should succeed");
        assert_eq!(parsed.0, 0x0001);
        assert_eq!(parsed.1.len(), 3);
        assert_eq!(parsed.1[0], (10, 20, -15));
        assert_eq!(parsed.1[1], (30, 40, 25));
    }

    #[test]
    fn test_kern_truncate() {
        let pairs: Vec<(u16, u16, i16)> = (0..500).map(|i| (i, i + 1, (i as i16 % 100) - 50)).collect();
        let orig = build_kern_table(&pairs, 1).expect("build kern table");
        let truncated = rebuild_kern_truncate(&orig, 300).expect("truncate");
        let parsed = parse_kern_format0(&truncated).unwrap();
        assert_eq!(parsed.1.len(), 300);
    }

    /// Path of a GPOS-only font (no `kern` table) for the GPOS-derived kern tests.
    /// SourceSerif4 Italic keeps its kerning in GPOS (ExtensionPos-wrapped class kern).
    fn gpos_only_font_bytes() -> Option<Vec<u8>> {
        let candidates = [
            "/home/hatch/.local/share/fonts/ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf",
            "/home/hatch/.local/share/fonts/ofl/ebgaramond/EBGaramond-Regular.ttf",
        ];
        for p in candidates {
            if let Ok(b) = std::fs::read(p) {
                return Some(b);
            }
        }
        None
    }

    #[test]
    fn test_gpos_kern_pairs_source_serif() {
        // SourceSerif4 Italic is GPOS-only (no kern table); its kern lives in
        // ExtensionPos-wrapped PairPos Format 1 + Format 2 (class kern). Ground
        // truth from fontTools with first-match-wins subtable composition at the
        // default instance: 109,502 nonzero pairs, max |value| 180.
        // NOTE: f(33)->o(42) is -25 only at the wght=400/opsz=10 instance; the
        // extractor reads the raw file (default instance, opsz=20), where f->o
        // is 0, so it is correctly absent. a(28)->U(22) is -20 at the default
        // instance and is used as the spot check instead.
        let path = "/home/hatch/.local/share/fonts/ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf";
        let Ok(base) = std::fs::read(path) else {
            return; // skip test in CI without this font
        };
        let font_ref = read_fonts::FontRef::new(&base).expect("base parses");
        let gpos_data = font_ref
            .table_data(read_fonts::types::Tag::new(b"GPOS"))
            .expect("GPOS present")
            .as_ref()
            .to_vec();
        let n = num_glyphs(&font_ref);
        assert!(n > 0, "num_glyphs from maxp");
        let pairs = gpos_kern_pairs(&gpos_data, n).expect("GPOS parses");
        // Substantial kern data extracted (fontTools first-wins: 109,502 pairs).
        assert!(pairs.len() > 1000, "expected >1000 GPOS kern pairs, got {}", pairs.len());
        // Sorted by (left, right) and unique.
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(pairs.len(), sorted.len(), "pairs must be sorted and unique");
        assert!(pairs.windows(2).all(|w| w[0] < w[1]), "pairs must be sorted");
        // a(28) -> U(22) is a real kern pair with value -20 at the default instance.
        let au = pairs.iter().find(|&&(l, r, _)| l == 28 && r == 22);
        assert!(au.is_some(), "a->U kern pair must be present");
        assert_eq!(au.unwrap().2, -20, "a->U kern value must be -20");
        // No zero values anywhere.
        assert!(pairs.iter().all(|&(_, _, v)| v != 0), "no zero-valued pairs");
    }

    #[test]
    fn test_postscript_gpos_derived_kern() {
        // A GPOS-only font must still get a kern fmt0 table under PostScript,
        // derived from GPOS, limited to 300 pairs, with GPOS stripped.
        let Some(base) = gpos_only_font_bytes() else {
            return; // skip test in CI without a GPOS-only font
        };
        let out = transform_font(&base, Era::PostScript, "TestFamily").expect("transform");
        let out_ref = read_fonts::FontRef::new(&out).expect("output parses");
        assert!(
            out_ref
                .table_data(read_fonts::types::Tag::new(b"GPOS"))
                .is_none(),
            "GPOS must be stripped for PostScript"
        );
        let kern_data = out_ref
            .table_data(read_fonts::types::Tag::new(b"kern"))
            .expect("PostScript output must have a kern table derived from GPOS");
        let (_coverage, pairs) =
            parse_kern_format0(kern_data.as_ref()).expect("kern must be format 0");
        assert!(!pairs.is_empty(), "kern table must not be empty");
        assert!(pairs.len() <= 300, "kern limited to 300 pairs, got {}", pairs.len());
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

        for era in [Era::PostScript, Era::TrueType, Era::OpenType, Era::Pdf14, Era::Docx, Era::HarfBuzz, Era::Variable] {
            let out = transform_font(&base, era, "TestFamily").expect("transform should not error");
            assert!(read_fonts::FontRef::new(&out).is_ok(), "era {:?} output should parse", era);
            // Check GPOS stripping for eras that should strip
            let orig_ref = read_fonts::FontRef::new(&base).unwrap();
            let out_ref = read_fonts::FontRef::new(&out).unwrap();
            let has_gpos_orig = orig_ref.table_data(read_fonts::types::Tag::new(b"GPOS")).is_some();
            let has_gpos_out = out_ref.table_data(read_fonts::types::Tag::new(b"GPOS")).is_some();
            match era {
                Era::PostScript | Era::TrueType => {
                    // GPOS must be stripped
                    assert!(!has_gpos_out, "GPOS should be stripped for {:?}", era);
                    match era {
                        Era::PostScript => {
                            // Kern kept, truncated to 300 pairs (DejaVu has 2727).
                            let kern_data = out_ref
                                .table_data(read_fonts::types::Tag::new(b"kern"))
                                .expect("PostScript output must have a kern table");
                            let (_, pairs) = parse_kern_format0(kern_data.as_ref())
                                .expect("kern must be format 0");
                            assert!(!pairs.is_empty(), "kern table must not be empty");
                            assert!(pairs.len() <= 300, "kern limited to 300 pairs");
                        }
                        Era::TrueType => {
                            // Kern stripped (covered again below; kept for clarity).
                            let has_kern_out = out_ref
                                .table_data(read_fonts::types::Tag::new(b"kern"))
                                .is_some();
                            assert!(!has_kern_out, "kern should be stripped for TrueType");
                        }
                        _ => unreachable!(),
                    }
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

    /// Distinctness of a transform output vs its base, for one era.
    fn distinct_for(base: &[u8], era: Era) -> bool {
        let out = transform_font(base, era, "TestFamily").expect("transform should not error");
        vintage_transform_distinct(base, &out)
    }

    #[test]
    fn test_distinct_modern_eras_are_noops() {
        // Modern eras return the base bytes unchanged: never distinct.
        let candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ];
        for path in candidates {
            let Ok(base) = std::fs::read(path) else { continue };
            for era in [Era::Pdf14, Era::InDesignCS, Era::Docx, Era::HarfBuzz, Era::Variable] {
                assert!(!distinct_for(&base, era), "{} {:?} must not be distinct", path, era);
            }
        }
    }

    #[test]
    fn test_distinct_gpos_kern_font() {
        // DejaVu Sans: GPOS with kern feature + 2727 kern pairs (fontTools).
        // Stripping GPOS changes effective kerning; stripping the kern table
        // alone does not (GPOS kern dominates, kern table invisible).
        let Ok(base) = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else { return };
        assert!(distinct_for(&base, Era::PostScript));
        assert!(distinct_for(&base, Era::TrueType));
        assert!(!distinct_for(&base, Era::OpenType));
    }

    #[test]
    fn test_distinct_gpos_no_kern_font() {
        // DejaVu Sans Mono: GPOS without a kern feature, no kern table.
        // Stripping GPOS is distinct (presence differs); OpenType keeps GPOS
        // and has no kern to strip, so it is a duplicate.
        let Ok(base) = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf") else { return };
        assert!(distinct_for(&base, Era::PostScript));
        assert!(distinct_for(&base, Era::TrueType));
        assert!(!distinct_for(&base, Era::OpenType));
    }

    #[test]
    fn test_distinct_msttcorefonts() {
        // No-GPOS bases: distinctness reduces to the kern table alone.
        // Pair counts measured with fontTools from the same files.
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let dir = std::path::PathBuf::from(home).join(".local/share/fonts/msttcorefonts");
        let read = |name: &str| std::fs::read(dir.join(name)).ok();
        // ArialMT: 909 kern pairs.
        if let Some(base) = read("arial.ttf") {
            assert!(distinct_for(&base, Era::PostScript), "Arial PostScript truncates 909 -> 300");
            assert!(distinct_for(&base, Era::TrueType), "Arial TrueType strips kern");
            assert!(distinct_for(&base, Era::OpenType), "Arial OpenType strips kern");
            assert!(!distinct_for(&base, Era::Pdf14), "Arial Pdf14 is a byte copy");
        }
        // CourierNewPSMT: no kern table at all.
        if let Some(base) = read("cour.ttf") {
            for era in [Era::PostScript, Era::TrueType, Era::OpenType] {
                assert!(!distinct_for(&base, era), "Courier {:?} has no kern to change", era);
            }
        }
        // TrebuchetMS: 118 kern pairs (< 300).
        if let Some(base) = read("trebuc.ttf") {
            assert!(!distinct_for(&base, Era::PostScript), "Trebuchet PostScript truncation is a no-op under 300 pairs");
            assert!(distinct_for(&base, Era::TrueType));
            assert!(distinct_for(&base, Era::OpenType));
        }
        // Georgia: no kern table at all.
        if let Some(base) = read("georgia.ttf") {
            for era in [Era::PostScript, Era::TrueType, Era::OpenType] {
                assert!(!distinct_for(&base, era), "Georgia {:?} has no kern to change", era);
            }
        }
        // Verdana: 534 kern pairs; Verdana-Bold: no kern table.
        if let Some(base) = read("verdana.ttf") {
            assert!(distinct_for(&base, Era::PostScript), "Verdana PostScript truncates 534 -> 300");
        }
        if let Some(base) = read("verdanab.ttf") {
            assert!(!distinct_for(&base, Era::OpenType), "Verdana-Bold OpenType has no kern to strip");
        }
    }

}
