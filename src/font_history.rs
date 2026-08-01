//! Font birth-year database for historically accurate vintage generation.
//!
//! Only generate a vintage variant if the font existed at the era start.
//! E.g. don't make a 1988 Comic Sans because it shipped in 1994.
//!
//! Matching is substring, case-insensitive, longest-key-wins so "times new roman"
//! beats "times". If family is unknown, we fall back to PostScript name.
//! Returns None => skip vintage generation for that font.

/// Metric-compatible free clones -> canonical historical family they stand in for.
/// Used for both birth-year projection and shipping checks.
/// Key is lower-case substring to match in family name; value is canonical family name.
/// If a modern free font is metric-compatible with a historical font, we treat its
/// *effective* birth year as the canonical's birth year, so we can generate vintage
/// variants from the free font file (no redistribution, built locally).
pub const HISTORICAL_CLONE_MAP: &[(&str, &str)] = &[
    // ClearType clones (LibreOffice)
    ("carlito", "calibri"),
    ("caladea", "cambria"),
    // Liberation / Croscore
    ("liberation sans", "arial"),
    ("liberation serif", "times new roman"),
    ("liberation mono", "courier new"),
    ("arimo", "arial"),
    ("tinos", "times new roman"),
    ("cousine", "courier new"),
    // URW Core 35 -> PS Base
    ("nimbus sans", "helvetica"),
    ("nimbus roman", "times new roman"),
    ("nimbus mono", "courier"),
    // TeX Gyre as stand-ins for PS base where metrics are close
    ("tex gyre heros", "helvetica"),
    ("tex gyre termes", "times new roman"),
    ("tex gyre cursor", "courier"),
    ("tex gyre pagella", "palatino"),
    ("tex gyre bonum", "bookman"),
    ("tex gyre schola", "century schoolbook"),
];

/// Canonicalize a family name for historical purposes.
/// Returns lower-case canonical name if alias matches, otherwise lower-cased original.
pub fn canonical_family(family: &str) -> String {
    let lower = family.to_lowercase();
    for &(alias, canon) in HISTORICAL_CLONE_MAP {
        if lower.contains(alias) {
            return canon.to_string();
        }
    }
    lower
}

/// (lowercase substring, birth year)
pub const KNOWN_FONT_BIRTH: &[(&str, u16)] = &[
    // Historical metal / early digital classics
    ("garamond", 1530),
    ("caslon", 1722),
    ("baskerville", 1757),
    ("bodoni", 1798),
    ("century schoolbook", 1919),
    ("futura", 1927),
    ("times new roman", 1931),
    ("times-roman", 1931),
    ("times", 1931), // alias, but after longer "times new roman"
    ("palatino", 1948),
    ("courier", 1955),
    ("courier new", 1955),
    ("helvetica", 1957),
    ("univers", 1957),
    ("optima", 1958),
    ("nimbus sans", 1984), // URW Helvetica clone lineage
    ("nimbus roman", 1982),
    ("nimbus mono", 1984),
    ("itc zapf dingbats", 1978),

    // 1980s desktop publishing
    ("arial", 1982),
    ("arial narrow", 1982),
    ("symbol", 1984),
    ("wingdings", 1990),

    // Early 90s core web / MS
    ("comic sans ms", 1994),
    ("comic sans", 1994),
    ("comic", 1994),
    ("georgia", 1993),
    ("trebuchet ms", 1996),
    ("trebuchet", 1996),
    ("verdana", 1996),
    ("tahoma", 1994),
    ("andale mono", 1995),
    ("impact", 1965), // actually 1965 Stephenson Blake, digital 1990s but allow early
    ("carlito", 2013), // metrics-compatible Calibri, modern free clone
    ("caladea", 2013), // Cambria clone
    ("liberation sans", 2007),
    ("liberation serif", 2007),
    ("liberation mono", 2007),
    ("dejavu sans", 2004),
    ("dejavu serif", 2004),
    ("dejavu", 2004),

    // Office era
    ("segoe ui", 2004),
    ("segoe", 2004),
    ("cambria", 2004),
    ("calibri", 2007),
    ("candara", 2006),
    ("corbel", 2005),
    ("consolas", 2005),
    ("constantia", 2005),
    ("aptos", 2023),

    // TeX Gyre / URW free families (GUST)
    ("tex gyre termes", 2006),
    ("tex gyre heros", 2006),
    ("tex gyre pagella", 2006),
    ("tex gyre bonum", 2006),
    ("tex gyre schola", 2006),
    ("tex gyre cursor", 2006),
    ("tex gyre adventor", 2006),
    ("texgyretermes", 2006),
    ("texgyreheros", 2006),

    // Adobe Source
    ("source sans pro", 2012),
    ("source sans 3", 2012),
    ("source sans", 2012),
    ("source serif pro", 2014),
    ("source serif 4", 2014),
    ("source serif", 2014),
    ("source code pro", 2012),
    ("source han sans", 2014),

    // Google Noto
    ("noto sans", 2013),
    ("noto serif", 2013),
    ("noto mono", 2013),
    ("noto", 2013),

    // IBM Plex
    ("ibm plex sans", 2017),
    ("ibm plex serif", 2017),
    ("ibm plex mono", 2017),
    ("ibm plex", 2017),

    // Open / libre families
    ("inter", 2017),
    ("roboto", 2011),
    ("open sans", 2010),
    ("lato", 2010),
    ("jost", 2018),
    ("fira sans", 2013),
    ("fira code", 2015),
    ("ubuntu", 2010),
    ("merriweather", 2011),
    ("playfair display", 2011),
    ("eb garamond", 2011),
    ("libre baskerville", 2012),
    ("libre bodoni", 2023),
    ("libre caslon", 2014),

    // Typewriter / vintage clones
    ("courier prime", 2013),
    ("special elite", 2010),
    ("cutive mono", 2011),
    ("ogcourier", 2000),
    ("letter gothic", 1956),
    ("prestige elite", 1953),
    ("consola", 2005), // fallback fragment
];

/// Return birth year for a font given family and PostScript name.
/// Case-insensitive substring match, longest key wins.
/// If family is a known free clone (carlito -> calibri), we use the canonical's birth year
/// so we can build vintage variants from the free file without redistributing.
/// If no match, returns None (caller should skip vintage generation).
pub fn font_birth_year(family: &str, ps_name: &str) -> Option<u16> {
    let fam = family.to_lowercase();
    let ps = ps_name.to_lowercase();
    let canon = canonical_family(&fam);
    // Prefer canonical match first (for clones), then original
    let search_targets = [&canon as &str, &fam as &str, &ps as &str];

    let mut best: Option<(usize, u16)> = None; // (match_len, year)

    for &(key, year) in KNOWN_FONT_BIRTH {
        for target in &search_targets {
            if target.contains(key) {
                let len = key.len();
                match best {
                    Some((blen, _)) if blen >= len => {},
                    _ => best = Some((len, year)),
                }
                break; // don't double-count same key across multiple targets
            }
        }
    }
    best.map(|(_, y)| y)
}

/// Returns true if a font born in `birth_year` existed at `era_start`.
/// Eras are inclusive start: font must have birth <= era_start.
#[inline]
pub fn era_exists_for_font(birth_year: u16, era_start: u16) -> bool {
    birth_year <= era_start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_birth() {
        assert_eq!(font_birth_year("Comic Sans MS", "ComicSansMS"), Some(1994));
        assert_eq!(font_birth_year("Times New Roman", "TimesNewRomanPSMT"), Some(1931));
        // Carlito is metric-compatible clone of Calibri, so effective birth is Calibri's 2007
        assert_eq!(font_birth_year("Carlito", "Carlito-Regular"), Some(2007));
        assert_eq!(font_birth_year("TeX Gyre Termes", "TeXGyreTermes-Regular"), Some(1931)); // canonicalized to times new roman via clone map -> 1931
        assert_eq!(font_birth_year("Nimbus Sans L", "NimbusSanL-Regu"), Some(1957)); // via helvetica
        assert_eq!(font_birth_year("Liberation Sans", "LiberationSans-Regular"), Some(1982)); // via arial
    }

    #[test]
    fn test_era_exists() {
        // Comic Sans 1994 did NOT exist in 1985 era
        assert!(!era_exists_for_font(1994, 1985));
        // But did exist in 1996 era
        assert!(era_exists_for_font(1994, 1996));
        // Times 1931 existed in all eras
        assert!(era_exists_for_font(1931, 1985));
        assert!(era_exists_for_font(1931, 2007));
    }

    #[test]
    fn test_unknown_skipped() {
        assert_eq!(font_birth_year("SomeFutureFont XYZ 2050", "SomeFutureFont"), None);
    }

    #[test]
    fn test_canonical() {
        assert_eq!(canonical_family("Carlito"), "calibri");
        assert_eq!(canonical_family("Nimbus Sans L"), "helvetica");
        assert_eq!(canonical_family("Source Sans 3"), "source sans 3"); // no alias, stays itself
    }
}
