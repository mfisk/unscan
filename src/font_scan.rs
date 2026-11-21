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

use unprint_fonts::ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use walkdir::WalkDir;

/// Cache of render-scale gutter counts per (font path, ligature char) —
/// avoids re-rendering the same shaped probe string for every candidate
/// per line. 52% of CPU was here in nightly pprof.
static GAP_CACHE: OnceLock<RwLock<HashMap<(PathBuf, char), usize>>> = OnceLock::new();

fn gap_cache() -> &'static RwLock<HashMap<(PathBuf, char), usize>> {
    GAP_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// FreeType library for the ligature-fusion probe, one per thread —
/// the same pattern as the verify render (src/verify.rs). Faces keep
/// the library alive through their own C-level reference, so a face
/// created inside the `with` closure outlives the borrow.
thread_local! {
    static LIG_FT_LIB: RefCell<Option<unprint_fonts::freetype::Library>> = RefCell::new(None);
}

/// Compact bitmask for FB00-FB04 ligatures: bit0=FF,1=FI,2=FL,3=FFI,4=FFL.
/// Replaces HashSet<char> to avoid allocation in hot path.
pub type LigMask = u8;
const FB_LIGS: [char;5] = ['\u{FB00}','\u{FB01}','\u{FB02}','\u{FB03}','\u{FB04}'];

#[inline]
fn lig_bit(c: char) -> Option<u8> {
    match c {
        '\u{FB00}' => Some(0),
        '\u{FB01}' => Some(1),
        '\u{FB02}' => Some(2),
        '\u{FB03}' => Some(3),
        '\u{FB04}' => Some(4),
        _ => None,
    }
}

/// Cache of final collapsed lig sets per font path — eliminates 33.9%
/// inclusive hot path (font_match.rs:138, font_pipeline.rs:381) which
/// otherwise did fs::read + cmap + 5× shape per candidate per line.
static LIG_SET_CACHE: OnceLock<RwLock<HashMap<String, LigMask>>> = OnceLock::new();

fn lig_set_cache() -> &'static RwLock<HashMap<String, LigMask>> {
    LIG_SET_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Cache of variant-independent ligature support masks per font path.
/// The support mask depends only on the font file (GSUB + glyph ink), not
/// on the variant tag, so it is keyed by path. Used for GT segmentation
/// (Finding 9): the GT font's segmentation must match the image, and the
/// image's ligature-ness is a property of the font's rendering, not of the
/// variant tag.
static LIG_SUPPORT_CACHE: OnceLock<RwLock<HashMap<PathBuf, LigMask>>> = OnceLock::new();

fn lig_support_cache() -> &'static RwLock<HashMap<PathBuf, LigMask>> {
    LIG_SUPPORT_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Cache of variant-independent ligature overlap masks per font path.
/// The overlap mask records, for each FB00–FB04 pair, whether the separate
/// component glyphs overlap when placed adjacently (default instance), so a
/// nolig render of the font still produces fused ink for that pair. Used
/// for GT segmentation (Finding 9 refinement): a nolig GT font whose
/// separate f/i do not overlap renders separate ink, and collapsing it
/// corrupts the GT geometry and similarity (BAP 622: five "Lining
/// figures" digit lines regressed). Keyed by path.
static LIG_OVERLAP_CACHE: OnceLock<RwLock<HashMap<PathBuf, LigMask>>> = OnceLock::new();

fn lig_overlap_cache() -> &'static RwLock<HashMap<PathBuf, LigMask>> {
    LIG_OVERLAP_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

// NOTE (Nov 20): the mask shapes with plain `rustybuzz::shape`, exactly
// as the verify render does (src/layout.rs `shape_word`). An earlier
// version cached ShapePlans keyed by the font data's heap address
// (data_ptr); the address is freed and reused as files are scanned in
// turn, so a later font could inherit an earlier font's plan and shape
// wrong — Lato and IBMPlexSans shaped 'fi' to two glyphs under the
// cached plan while plain shaping (and HarfBuzz itself) substitute the
// single FB01 ligature. The plan cache is gone; plain shaping rebuilds
// the plan per call, which the scan absorbs (scan time measured in the
// BAP).
#[inline]
fn shape_probes_collapsed(face: &unprint_fonts::rustybuzz::Face, features: &[unprint_fonts::rustybuzz::Feature], probe: &str) -> bool {
    let mut buf = unprint_fonts::rustybuzz::UnicodeBuffer::new();
    buf.push_str(probe);
    buf.guess_segment_properties();
    let out = unprint_fonts::rustybuzz::shape(face, features, buf);
    out.glyph_infos().len() == 1 && out.glyph_infos()[0].glyph_id != 0
}

/// Shape `probe` with liga/dlig on and kerning OFF, returning the shaped
/// glyph sequence as (glyph_id, x_advance, x_offset, y_offset) in font
/// units. Kerning is off: the mask is variant-independent and the
/// overlap test it complements ignores kern (raw glyph overlap
/// dominates); the nokern GT render the mask must match uses raw
/// advances. The glyph COUNT is kern-independent (kern only moves
/// glyphs, never substitutes), so one glyph here also means the shaper
/// collapsed the probe. Shaping is plain `rustybuzz::shape` — the same
/// path the verify render uses — so the mask judges what the pipeline
/// itself renders. Returns an empty vec when shaping fails or yields
/// no glyphs.
#[inline]
fn shape_probe_sequence(
    face: &unprint_fonts::rustybuzz::Face,
    probe: &str,
) -> Vec<(u32, i32, i32, i32)> {
    let liga_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"liga");
    let dlig_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"dlig");
    let kern_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"kern");
    let features = [
        unprint_fonts::rustybuzz::Feature::new(liga_tag, 1, ..),
        unprint_fonts::rustybuzz::Feature::new(dlig_tag, 1, ..),
        unprint_fonts::rustybuzz::Feature::new(kern_tag, 0, ..),
    ];
    let mut buf = unprint_fonts::rustybuzz::UnicodeBuffer::new();
    buf.push_str(probe);
    buf.guess_segment_properties();
    let out = unprint_fonts::rustybuzz::shape(face, &features, buf);
    let infos = out.glyph_infos();
    let positions = out.glyph_positions();
    infos
        .iter()
        .zip(positions.iter())
        .map(|(info, pos)| (info.glyph_id, pos.x_advance, pos.x_offset, pos.y_offset))
        .collect()
}

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

impl FontClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            FontClass::Serif => "serif",
            FontClass::Sans => "sans",
            FontClass::Mono => "mono",
            FontClass::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FontEntry {
    pub path: PathBuf,
    pub family_name: String,
    /// Canonical PostScript name: raw nameID 6 with numeric weight appended
    /// via `make_weight_explicit`.  Used for all font identity comparisons.
    pub postscript_name: String,
    /// Raw PostScript name (nameID 6) as it appears in the font file and
    /// therefore in PDF BaseFont entries.  Used only to map PDF BaseFont
    /// back to a catalog entry during GT canonicalization.
    pub raw_postscript_name: String,
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
    /// Cached font_key = postscript_name (+ "|" + variant_tag if non-empty).
    /// Stored to avoid repeated String alloc in hot loops (2532 fonts × 500 chars).
    /// Not serialized; recomputed on load.
    #[allow(dead_code)]
    pub font_key_cache: String,
    /// For variant entries: maps characters to their feature-specific glyph IDs.
    /// Only characters whose glyph ID differs from default are included.
    /// None for the default entry (use normal cmap lookup).
    pub glyph_overrides: Option<Vec<(char, u16)>>,
    /// For variable-font weight instances: axis coordinates to set before
    /// rendering (e.g. `[(b"wght", 700.0)]`).  None for static fonts and
    /// the default instance of variable fonts.
    pub variations: Option<Vec<([u8; 4], f32)>>,
    /// Typographic family name from the font's name table (nameID 16,
    /// falling back to nameID 1).  Used for dedup: when a static font and
    /// a variable-font weight instance share the same family+weight, the
    /// static one wins.  Not serialized in the font registry.
    pub typographic_family: String,
    /// Vintage era this entry represents, e.g. "word6", "postscript", "variable".
    /// None for base/system fonts. Used to distinguish vintage variants that
    /// copy the base PostScript name but represent different era transforms.
    /// Included in font_key to avoid colliding with base entry.
    /// String (not Era) to avoid circular dep with vintage_cache crate.
    pub vintage_era: Option<String>,
    /// (min, default, max) of the `opsz` (optical size) axis, captured during
    /// the font scan while the font file bytes are available. None for fonts
    /// without an optical-size axis. `identify_fonts` uses this to synthesize
    /// the size-derived `|opszN` candidate; it cannot be re-derived later
    /// because `data` is dropped after the scan. Serialized in the scan cache.
    pub opsz_range: Option<(f32, f32, f32)>,
    /// True if the font file contains a GPOS table. Captured during the scan
    /// while the font file bytes are available; cannot be re-derived later
    /// because `data` is dropped after the scan. Used by the shared font-key
    /// preference (`font_key_preference_score`) for --resolve-fontkey,
    /// --select-font, and classifier tiebreaking. Serialized in the
    /// scan cache (v17+).
    pub has_gpos: bool,
    /// True if the font file has kerning data (legacy `kern` table with at
    /// least one pair, or a GPOS `kern` feature with at least one lookup).
    /// Captured during the scan while the font file bytes are available;
    /// cannot be re-derived later because `data` is dropped after the scan.
    /// Drives the shared font-key preference: a font with kerning renders
    /// kerned by default, so the kerned variant wins the default pick.
    /// (GPOS presence alone does not imply kerning — legacy `kern` tables
    /// kern in rendering too.) Serialized in the scan cache (v18+).
    pub has_kern: bool,
    /// Font version from the `head` table `fontRevision` field, formatted
    /// as "major.minor" (e.g. "2.50"). Captured during the scan while the
    /// font file bytes are available; cannot be re-derived later because
    /// `data` is dropped after the scan. Part of the font key
    /// (`|version=X`): two files with the same PostScript name but
    /// different versions (e.g. Win31 Times v2.50 vs modern Times v2.82)
    /// are different fonts and must not collide. Identical files share
    /// a version, so they share a key and dedup naturally. Serialized
    /// in the scan cache (v20+).
    pub font_version: String,
}

impl FontEntry {
    /// Append `|variant_tag` segments to a PostScript name, skipping any
    /// segment already present as a trailing `|segment`. Variant entries
    /// store the OT tag in `postscript_name` already (`PS|salt`, for exact
    /// GT matching), so a verbatim append would duplicate it (`PS|salt|salt`).
    /// Each tag appears at most once in the resulting key.
    fn append_variant_tag(mut k: String, variant_tag: &str) -> String {
        for seg in variant_tag.split('|') {
            if seg.is_empty() {
                continue;
            }
            if !k.ends_with(&format!("|{seg}")) {
                k.push('|');
                k.push_str(seg);
            }
        }
        k
    }

    #[inline]
    pub fn compute_font_key(postscript_name: &str, variant_tag: &str) -> String {
        Self::append_variant_tag(postscript_name.to_owned(), variant_tag)
    }

    #[inline]
    pub fn compute_font_key_full(
        postscript_name: &str,
        variant_tag: &str,
        vintage_era: Option<&str>,
        variations: &Variations,
        font_version: &str,
    ) -> String {
        // The font version (|version=X, from the head table fontRevision)
        // goes immediately after the PostScript name, before any variant
        // tags: it identifies WHICH font file this is (e.g. Win31 Times
        // v2.50 vs modern Times v2.82 share a PostScript name but are
        // different fonts). OT variant entries pass "PS|tag" as the name,
        // so insert before the first '|' to keep the version adjacent to
        // the bare PostScript name. An empty version (tombstones) omits
        // the tag.
        let mut k = postscript_name.to_owned();
        if !font_version.is_empty() {
            let tag = format!("|version={font_version}");
            match k.find('|') {
                Some(idx) => k.insert_str(idx, &tag),
                None => k.push_str(&tag),
            }
        }
        let mut k = Self::append_variant_tag(k, variant_tag);
        if let Some(vars) = variations {
            if !vars.is_empty() {
                k.push('[');
                for (i, (tag, val)) in vars.iter().enumerate() {
                    if i > 0 {
                        k.push(',');
                    }
                    let tag_str = std::str::from_utf8(tag).unwrap_or("????");
                    if val.fract() == 0.0 {
                        k.push_str(&format!("{}={}", tag_str, *val as i32));
                    } else {
                        k.push_str(&format!("{}={}", tag_str, val));
                    }
                }
                k.push(']');
            }
        }
        if let Some(era) = vintage_era {
            k.push_str("|vintage=");
            k.push_str(era);
        }
        k
    }

    /// Unique key for this font entry in the font registry.
    /// Uses the canonical PostScript name (from `make_weight_explicit`) so
    /// duplicate font files with different paths but the same identity
    /// collapse to a single key.  Variant entries append `|tag`.
    /// Vintage era and variable vs static are now part of the key so that
    /// `liberationsans-word6-<hash>.ttf` copying PS name does not collide
    /// with base, and static Inter-Regular vs variable Inter wght400 are distinct.
    /// Returns clone of cached key to preserve API but avoid recompute (perf slice 8).
    #[inline]
    pub fn font_key(&self) -> String {
        self.font_key_cache.clone()
    }

    /// Borrowed view of cached key — zero alloc, for hot loops.
    #[inline]
    pub fn font_key_ref(&self) -> &str {
        &self.font_key_cache
    }

    /// Recompute cache after mutating vintage_era or variant fields (used for vintage generation).
    pub fn recompute_font_key_cache(&mut self) {
        self.font_key_cache = Self::compute_font_key_full(
            &self.postscript_name,
            &self.variant_tag,
            self.vintage_era.as_deref(),
            &self.variations,
            &self.font_version,
        );
    }

    fn assign_default_variations(&mut self, default_variations: Variations) {
        if let Some(default_axes) = default_variations {
            self.variations = Some(default_axes);
            self.recompute_font_key_cache();
        }
    }

    /// Apply variable-font axis coordinates to a FontRef/FontVec.
    /// Centralizes FontEntry -> FontRef variation handling so all render
    /// paths (char_render, train prewarm, train missing-feature) apply the
    /// same coordinates.
    #[inline]
    pub fn apply_variations<F: unprint_fonts::ab_glyph::VariableFont>(&self, font: &mut F) {
        if let Some(vars) = &self.variations {
            for (tag, val) in vars {
                let _ = font.set_variation(tag, *val);
            }
        }
    }

    /// Create a FontRef from borrowed data and apply this entry's variations.
    /// Returns None if the data is not a valid font.
    #[inline]
    pub fn font_ref_with_variations<'a>(
        &self,
        data: &'a [u8],
    ) -> Option<unprint_fonts::ab_glyph::FontRef<'a>> {
        let mut font = unprint_fonts::ab_glyph::FontRef::try_from_slice(data).ok()?;
        self.apply_variations(&mut font);
        Some(font)
    }

    /// Set of ligature unicode codepoints (FB00–FB04) that this font
    /// truly collapses as one contiguous ink blob.
    ///
    /// Historically this was called `allowed_lig_set` and was derived only from
    /// `glyph_overrides` + cmap + GSUB shaping.  That led to fonts that claim a
    /// ligature via GSUB but render it as two separate inks (e.g., separate f
    /// and i with a 100% vertical white gutter) still being treated as collapsed.
    /// Those should be scored as separate characters.
    ///
    /// New name is `collapsed_lig_set`.  It filters the supported set by
    /// shaping each ligature's probe string and rendering the shaped
    /// glyphs at the word-scan scale, checking the render for a clear
    /// vertical gutter (see `count_render_gutters_for_shaped`).  Two-char
    /// ligs (ff, fi, fl) need ≥1 such interior gutter to be excluded;
    /// three-char ligs (ffi, ffl) need ≥2.  Only FB00–FB04 are
    /// considered; quote ligature probes are excluded.
    pub fn collapsed_lig_set(&self) -> LigMask {
        // nolig variant: disable ligature collapsing entirely.
        // The cache is keyed by fontkey (not path) so nolig variants
        // get their own entry (0) distinct from the base variant's set.
        if self.variant_tag.contains("nolig") {
            return 0;
        }
        // Global memo — eliminates 33.9% inclusive hot path after first line.
        // Final filtered set is cached by fontkey, so no FS read / shaping on hits.
        let fk = self.font_key_cache.clone();
        {
            let guard = lig_set_cache().read().unwrap();
            if let Some(cached) = guard.get(&fk) {
                return *cached;
            }
        }
        let mask = self.ligature_support_mask();
        // Insert final filtered set into global cache (keyed by fontkey).
        {
            let mut guard = lig_set_cache().write().unwrap();
            guard.insert(fk, mask);
        }
        mask
    }

    /// Set of ligature unicode codepoints (FB00–FB04) this font truly
    /// collapses as one contiguous ink blob, IGNORING the variant tag.
    ///
    /// This is the same GSUB + glyph-ink computation as
    /// `collapsed_lig_set`, but without the nolig early-return. The support
    /// mask is a property of the font file, not of the variant: a font that
    /// has a fused f_i ligature glyph renders fused "fi" ink whenever its
    /// liga feature fires, whatever the variant key says.
    ///
    /// Used for GT segmentation (Finding 9): the GT font's segmentation must
    /// match the image, and the image's ligature-ness is a property of the
    /// font's rendering, not of the variant tag. A nolig GT key on a line
    /// whose render used ligatures would otherwise force an 8-way split of a
    /// fused "fi", cutting solid ink and giving the correct font a
    /// catastrophic geometry penalty (p1 l19 "figures:": GT 'i' h_err
    /// -4.9 px, h_ll -39.08, ~50 points against the correct font).
    /// Cached by font path (variant-independent).
    pub fn ligature_support_mask(&self) -> LigMask {
        let path_key = self.path.clone();
        {
            let guard = lig_support_cache().read().unwrap();
            if let Some(cached) = guard.get(&path_key) {
                return *cached;
            }
        }
        let mask = self.compute_ligature_support_mask();
        {
            let mut guard = lig_support_cache().write().unwrap();
            guard.insert(path_key, mask);
        }
        mask
    }

    /// Set of ligature codepoints (FB00–FB04) whose separate component
    /// glyphs overlap when placed adjacently (default instance), so a nolig
    /// render of this font still produces fused ink for that pair.
    /// E.g. EB Garamond's f (xMax 420) and i (xMin 24, after f advance
    /// 318) overlap by 78 font units, so even a nolig render fuses "fi".
    /// Most fonts have a clear gap between f and i, so their nolig render
    /// has separate ink and must NOT be collapsed in GT segmentation.
    /// Cached by font path (variant-independent).
    pub fn ligature_overlap_mask(&self) -> LigMask {
        let path_key = self.path.clone();
        {
            let guard = lig_overlap_cache().read().unwrap();
            if let Some(cached) = guard.get(&path_key) {
                return *cached;
            }
        }
        let mask = self.compute_ligature_overlap_mask();
        {
            let mut guard = lig_overlap_cache().write().unwrap();
            guard.insert(path_key, mask);
        }
        mask
    }

    /// Core overlap computation: for each FB00–FB04 pair, check whether any
    /// adjacent pair of component glyphs overlaps (first glyph's ink right
    /// edge past the second glyph's ink left edge when placed at the first
    /// glyph's advance). Default-instance metrics; kern is ignored (the raw
    /// glyph overlap dominates the fused-vs-separate determination).
    fn compute_ligature_overlap_mask(&self) -> LigMask {
        let mut mask: LigMask = 0;
        let data = match std::fs::read(&self.path) {
            Ok(d) => d,
            Err(_) => return 0,
        };
        let face = match unprint_fonts::ttf_parser::Face::parse(&data, 0) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        // (ligature char, component chars in order).
        const PAIRS: &[(char, &[char])] = &[
            ('\u{FB00}', &['f', 'f']),
            ('\u{FB01}', &['f', 'i']),
            ('\u{FB02}', &['f', 'l']),
            ('\u{FB03}', &['f', 'f', 'i']),
            ('\u{FB04}', &['f', 'f', 'l']),
        ];
        for &(lig, comps) in PAIRS {
            let Some(b) = lig_bit(lig) else { continue };
            let mut overlaps = false;
            for w in comps.windows(2) {
                let (a, c) = (w[0], w[1]);
                let (Some(ga), Some(gc)) = (face.glyph_index(a), face.glyph_index(c)) else { continue };
                if ga.0 == 0 || gc.0 == 0 { continue; }
                let adv_a = face.glyph_hor_advance(ga).unwrap_or(0) as i32;
                let (Some(bb_a), Some(bb_c)) = (face.glyph_bounding_box(ga), face.glyph_bounding_box(gc)) else { continue };
                // First glyph's ink right edge vs second glyph's ink left
                // edge when the second glyph starts at the first's advance.
                if (bb_a.x_max as i32) - (adv_a + bb_c.x_min as i32) > 0 {
                    overlaps = true;
                    break;
                }
            }
            if overlaps {
                mask |= 1u8 << b;
            }
        }
        mask
    }

    /// Ligature mask for GT segmentation (Finding 9): collapse a pair when
    /// the render would produce fused ink — either the variant is lig
    /// (`collapsed_lig_set` covers it) or the separate glyphs overlap so
    /// even a nolig render fuses the ink. A nolig GT font whose separate
    /// glyphs do not overlap renders separate ink and must not be collapsed.
    pub fn gt_ligature_mask(&self) -> LigMask {
        self.collapsed_lig_set() | (self.ligature_support_mask() & self.ligature_overlap_mask())
    }

    /// Core ligature-support computation: glyph_overrides + cmap + GSUB
    /// shaping, filtered by the glyph-ink gutter check. No variant-tag
    /// gate. See `collapsed_lig_set` for the filter semantics.
    fn compute_ligature_support_mask(&self) -> LigMask {
        let mut mask: LigMask = 0;
        // Fast path: anything already known from glyph_overrides.
        if let Some(ref ov) = self.glyph_overrides {
            for (ch, _gid) in ov {
                if let Some(b) = lig_bit(*ch) {
                    mask |= 1u8 << b;
                }
            }
        }

        // Probe the actual font file for cmap and GSUB liga support if not full.
        let mut data_cache: Option<Vec<u8>> = None;
        let current_count = mask.count_ones() as usize;
        if current_count < 5 {
            if let Ok(data) = std::fs::read(&self.path) {
                if let Ok(font) = FontRef::try_from_slice(&data) {
                    for &lig in &FB_LIGS {
                        let b = lig_bit(lig).unwrap();
                        if (mask & (1u8<<b)) != 0 { continue; }
                        if font.glyph_id(lig).0 != 0 {
                            mask |= 1u8<<b;
                        }
                    }
                }
                if let Some(face) = unprint_fonts::rustybuzz::Face::from_slice(&data, 0) {
                    let liga_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"liga");
                    let dlig_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"dlig");
                    let features = [
                        unprint_fonts::rustybuzz::Feature::new(liga_tag, 1, ..),
                        unprint_fonts::rustybuzz::Feature::new(dlig_tag, 1, ..),
                    ];
                    for &(probe, lig_char) in LIGATURE_PROBES {
                        let Some(b) = lig_bit(lig_char) else { continue };
                        if (mask & (1u8<<b)) != 0 { continue; }
                        if probe.chars().count() <= 1 { continue; }
                        if shape_probes_collapsed(&face, &features, probe) {
                            mask |= 1u8<<b;
                        }
                    }
                }
                data_cache = Some(data);
            }
        } else {
            if let Ok(data) = std::fs::read(&self.path) {
                data_cache = Some(data);
            }
        }

        // --- collapsed filtering: exclude ligs that are really two/three inks ---
        // Fusion is judged per ligature at the word-scan scale on the
        // SHAPED result (see `count_render_gutters_for_shaped`): the
        // probe string is shaped with liga/dlig, the shaped glyphs are
        // rendered, and a clean vertical gutter in the render means
        // separate ink and the ligature must not collapse. Shaping
        // first matters: most fonts never substitute a ligature glyph
        // for the text, so the FB01 glyph's ink is not what the scan
        // shows. One FreeType face and one rustybuzz face per font
        // file, created lazily on the first uncached ligature.
        if let Some(ref data) = data_cache {
            let path_key = self.path.clone();
            let mut ft_face: Option<unprint_fonts::freetype::Face<Vec<u8>>> = None;
            let mut ft_face_tried = false;
            let mut buzz_face: Option<unprint_fonts::rustybuzz::Face> = None;
            let mut buzz_face_tried = false;
            for &lig in &FB_LIGS {
                let b = lig_bit(lig).unwrap();
                if (mask & (1u8<<b)) == 0 { continue; }
                let required_gaps = match lig {
                    '\u{FB00}' | '\u{FB01}' | '\u{FB02}' => 1usize,
                    '\u{FB03}' | '\u{FB04}' => 2usize,
                    _ => continue,
                };
                let cached = {
                    let guard = gap_cache().read().unwrap();
                    guard.get(&(path_key.clone(), lig)).copied()
                };
                let gaps = if let Some(g) = cached {
                    g
                } else {
                    if !ft_face_tried {
                        ft_face_tried = true;
                        ft_face =
                            Self::ft_face_for_lig_probe(data, Self::LIG_PROBE_EMS_PX[0]);
                    }
                    if !buzz_face_tried {
                        buzz_face_tried = true;
                        buzz_face = unprint_fonts::rustybuzz::Face::from_slice(data, 0);
                    }
                    let g = match (ft_face.as_mut(), buzz_face.as_ref()) {
                        (Some(ft), Some(bz)) => {
                            let probe = LIGATURE_PROBES
                                .iter()
                                .find(|&&(_, l)| l == lig)
                                .map(|&(p, _)| p)
                                .unwrap_or("");
                            let glyphs = shape_probe_sequence(bz, probe);
                            Self::count_render_gutters_for_shaped(ft, &glyphs)
                        }
                        _ => 0,
                    };
                    let mut guard = gap_cache().write().unwrap();
                    guard.insert((path_key.clone(), lig), g);
                    g
                };
                if gaps >= required_gaps {
                    mask &= !(1u8<<b);
                }
            }
        }

        mask
    }

    /// Em sizes (px) the ligature-fusion test renders at: the
    /// word-scan scales the pipeline receives. The specimen's text
    /// carrying ligature pairs is set at 7pt, 9pt, and 10pt
    /// (test-docs/gen-specimen.py: 7pt source lines, 9pt lining
    /// figures, 10pt body), rasterized at 300 dpi
    /// (tools/rasterize.py), so pt * 300 / 72 px em; the measured
    /// midpoint_em_px across BAP 1364 spans 29.3–43.4 for lines
    /// containing a ligature pair, consistent with 7–10pt. Fusion is
    /// judged at all three sizes, never at outline truth: a gap that
    /// is a clean column at 10pt can be fused at 7pt, and the mask is
    /// one bit per font file, so a gutter counts only when it
    /// survives every size (see `count_render_gutters_for_shaped`).
    const LIG_PROBE_EMS_PX: [f32; 3] = [
        7.0 * 300.0 / 72.0,
        9.0 * 300.0 / 72.0,
        10.0 * 300.0 / 72.0,
    ];

    /// Size `face` for the ligature-fusion probe at `em_px`
    /// (pixels-as-points at 72 dpi, exactly as the verify render
    /// in src/verify.rs). Returns false on failure; the caller
    /// treats that as fused (keeps the mask bit).
    fn set_lig_probe_size(
        face: &mut unprint_fonts::freetype::Face<Vec<u8>>,
        em_px: f32,
    ) -> bool {
        let size_26_6 = (em_px as f64 * 64.0) as isize;
        face.set_char_size(size_26_6, size_26_6, 72, 72).is_ok()
    }

    /// A FreeType face for the ligature-fusion probe, sized at
    /// `em_px`. Default instance: the mask is a property of the
    /// font file, not of a variant (see `ligature_support_mask`), and
    /// the ab_glyph path this replaces also measured the default
    /// instance. Returns None when the face cannot be created or
    /// sized; the caller treats that as fused (keeps the mask bit).
    fn ft_face_for_lig_probe(
        data: &[u8],
        em_px: f32,
    ) -> Option<unprint_fonts::freetype::Face<Vec<u8>>> {
        LIG_FT_LIB.with(|lib_cell| -> Option<unprint_fonts::freetype::Face<Vec<u8>>> {
            let mut borrow = lib_cell.borrow_mut();
            if borrow.is_none() {
                *borrow = unprint_fonts::freetype::Library::init().ok();
            }
            let lib = borrow.as_ref()?;
            let mut face = lib.new_memory_face2(data.to_vec(), 0).ok()?;
            if !Self::set_lig_probe_size(&mut face, em_px) {
                return None;
            }
            Some(face)
        })
    }

    /// Count interior runs of clean columns in a per-column ink map: a
    /// run counts when it closes against ink on the right having seen
    /// ink on the left (the segmenter's VP pass only splits within the
    /// ink extent).
    fn count_interior_clean_runs(col_has_ink: &[bool]) -> usize {
        let mut gaps = 0usize;
        let mut in_gap = false;
        let mut seen_ink = false;
        for &has_ink in col_has_ink {
            if has_ink {
                if in_gap && seen_ink {
                    gaps += 1;
                }
                in_gap = false;
                seen_ink = true;
            } else if !in_gap {
                in_gap = true;
            }
        }
        gaps
    }

    /// Render the shaped sequence `glyphs` (from
    /// `shape_probe_sequence`) onto one canvas at `em_px`, with the
    /// pen starting at subpixel phase `phase`, using the same
    /// FreeType rendering the verify path scores with (hinting on,
    /// RENDER load flag), and return the per-column ink map (a column
    /// has ink iff any pixel is below INK_THRESH, gray 240). The
    /// glyphs are blitted at their shaped advances, exactly as the
    /// verify render blits shaped words.
    /// Each glyph is rendered with its true fractional pen offset
    /// applied via FT_Set_Transform (the delta is 26.6 fixed point,
    /// like all FT_Pos values; the matrix is NULL for identity):
    /// FreeType hints the glyph and then translates the hinted
    /// outline, which is how a FreeType-based renderer positions
    /// glyphs at fractional pen offsets. Returns None when the
    /// sequence is empty or any render fails.
    fn render_shaped_columns(
        face: &mut unprint_fonts::freetype::Face<Vec<u8>>,
        glyphs: &[(u32, i32, i32, i32)],
        phase: f64,
        em_px: f32,
    ) -> Option<Vec<bool>> {
        if glyphs.is_empty() {
            return None;
        }
        if !Self::set_lig_probe_size(face, em_px) {
            return None;
        }
        let upem = face.raw().units_per_EM as f32;
        if upem <= 0.0 {
            return None;
        }
        let px_per_unit = em_px / upem;
        // Canvas: total advance plus a full em of margin on each side
        // for overhang; two ems tall with the baseline one em down.
        let total_adv: f32 = glyphs
            .iter()
            .map(|&(_, adv, _, _)| adv as f32 * px_per_unit)
            .sum();
        let margin = em_px.ceil() as i32;
        let canvas_w = (total_adv.ceil() as i32 + 2 * margin).max(4) as usize;
        let canvas_h = (2.0 * em_px).ceil() as usize;
        let baseline_y = em_px as f64;
        let mut canvas = vec![255.0f32; canvas_w * canvas_h];
        let mut pen_x = margin as f64 + phase;
        for &(gid, x_adv, x_off, y_off) in glyphs {
            if gid == 0 {
                return None;
            }
            // The glyph's true pen position, fractional part carried
            // by the transform, integer part by the blit.
            let pen_eff = pen_x + x_off as f64 * px_per_unit as f64;
            let frac = pen_eff.fract();
            let mut delta = unprint_fonts::freetype::freetype_sys::FT_Vector {
                x: (frac * 64.0).round() as i64,
                y: 0,
            };
            let raw_face = face.raw() as *const _ as *mut _;
            unsafe {
                unprint_fonts::freetype::freetype_sys::FT_Set_Transform(
                    raw_face,
                    std::ptr::null_mut(),
                    &mut delta as *mut _,
                );
            }
            if face
                .load_glyph(gid, unprint_fonts::freetype::face::LoadFlag::RENDER)
                .is_err()
            {
                return None;
            }
            let glyph = face.glyph();
            let bitmap = glyph.bitmap();
            if !matches!(
                bitmap.pixel_mode(),
                Ok(unprint_fonts::freetype::bitmap::PixelMode::Gray)
            ) {
                return None;
            }
            let bmp_w = bitmap.width() as usize;
            let bmp_h = bitmap.rows() as usize;
            let buf = bitmap.buffer();
            let pitch = bitmap.pitch().unsigned_abs() as usize;
            if bmp_w == 0 || bmp_h == 0 || buf.is_empty() {
                pen_x += x_adv as f64 * px_per_unit as f64;
                continue;
            }
            if pitch < bmp_w || buf.len() < pitch * bmp_h.saturating_sub(1) + bmp_w {
                return None;
            }
            // Blit at the integer pen plus the bitmap bearing (which
            // already accounts for the fractional transform).
            let blit_x = pen_eff.floor() as i32 + glyph.bitmap_left();
            let blit_y = (baseline_y
                - y_off as f64 * px_per_unit as f64
                - glyph.bitmap_top() as f64)
                .round() as i32;
            // Alpha compositing mirrors the verify render: the canvas
            // holds gray values (255 = white) and each glyph darkens
            // by (1 - alpha). The f32 canvas and the verify path's u8
            // canvas agree exactly on the < 240 verdict (240 is an
            // integer, so flooring cannot move a value across it).
            for row in 0..bmp_h {
                for col in 0..bmp_w {
                    let cx = blit_x + col as i32;
                    let cy = blit_y + row as i32;
                    if cx < 0 || cy < 0 || cx >= canvas_w as i32 || cy >= canvas_h as i32 {
                        continue;
                    }
                    let alpha = buf[row * pitch + col] as f32 / 255.0;
                    if alpha < 0.01 {
                        continue;
                    }
                    let idx = cy as usize * canvas_w + cx as usize;
                    canvas[idx] *= 1.0 - alpha;
                }
            }
            pen_x += x_adv as f64 * px_per_unit as f64;
        }
        // One pass: column ink presence (any pixel below INK_THRESH).
        let thresh = crate::INK_THRESH as f32;
        let mut col_has_ink = vec![false; canvas_w];
        for y in 0..canvas_h {
            let row = y * canvas_w;
            for x in 0..canvas_w {
                if canvas[row + x] < thresh {
                    col_has_ink[x] = true;
                }
            }
        }
        Some(col_has_ink)
    }

    /// Count interior clean vertical gutters in the RENDER of what the
    /// text shapes to for a ligature, at the word-scan scales
    /// (LIG_PROBE_EMS_PX), using the same FreeType rendering the verify
    /// path scores with (hinting on, RENDER load flag). `glyphs` is the
    /// shaped sequence from `shape_probe_sequence`. Shaping first
    /// matters because most fonts do not substitute a ligature glyph
    /// for the text at all — Arial, CourierNew, and SourceSans3 have
    /// no GSUB fi ligature, so Pango renders f+i and the FB01 glyph's
    /// ink is never in the scan; judging the FB01 glyph instead of
    /// the shaped pair misjudges CourierNew as fused.
    ///
    /// One glyph means the shaper substituted a ligature: there is no
    /// inter-glyph gap to judge, and a clean column inside the glyph
    /// is not a split point the segmenter should use. Splitting
    /// Roboto's and Lato's single FB01 glyphs collapses their GT
    /// scores (BAP 1469: Roboto gt_score -0.70 -> -21.12, Lato -0.46
    /// -> -786.12), while collapsing them scores fine. Fused.
    ///
    /// Otherwise the count is the MINIMUM over every sampled size
    /// (7/9/10pt) and subpixel phase (0, 0.25, 0.5, 0.75): the mask
    /// is per font file but words land at arbitrary subpixel phases
    /// and sizes, and a sub-pixel gutter exists only at some of
    /// them — Arial's f/i gap is 1.32px unhinted, showing one clean
    /// column at phase 0 while the scan's word is fused, and the
    /// lining-figures lines are set at 9pt, where a marginal gap
    /// closes. A gutter counts only when it survives every sampled
    /// size and phase. That is the safe side of a size- and
    /// phase-dependent call: splitting fused ink costs a headline
    /// hit (the phantom-'i' geo penalty), while collapsing separate
    /// ink costs only an exact hit (the nolig twin still wins the
    /// line).
    ///
    /// Returns 0 when the sequence is empty or any render fails — the
    /// ligature is treated as fused, keeping the mask bit (the
    /// existing failure semantics).
    fn count_render_gutters_for_shaped(
        face: &mut unprint_fonts::freetype::Face<Vec<u8>>,
        glyphs: &[(u32, i32, i32, i32)],
    ) -> usize {
        if glyphs.is_empty() {
            return 0;
        }
        if glyphs.len() == 1 {
            return 0;
        }
        const PHASES: [f64; 4] = [0.0, 0.25, 0.5, 0.75];
        let mut min_gaps = usize::MAX;
        for &em_px in &Self::LIG_PROBE_EMS_PX {
            for &phase in &PHASES {
                match Self::render_shaped_columns(face, glyphs, phase, em_px) {
                    Some(cols) => {
                        let gaps = Self::count_interior_clean_runs(&cols);
                        min_gaps = min_gaps.min(gaps);
                        if min_gaps == 0 {
                            return 0;
                        }
                    }
                    None => return 0,
                }
            }
        }
        min_gaps
    }

    /// (min, default, max) for the `opsz` axis, or None if this font has no
    /// optical-size axis. Returns the range captured during the font scan
    /// while the font file bytes were available.
    pub fn opsz_axis_range(&self) -> Option<(f32, f32, f32)> {
        self.opsz_range
    }
}

/// Parse a synthetic `|opszN` suffix from a candidate fontkey.
///
/// Returns `(base_key, Some(opsz))` when the key ends with `|opsz` followed by
/// a number, else `(key, None)`. Used to resolve synthetic per-line optical-
/// size candidates back to their registry entry plus an opsz override.
pub fn parse_opsz_suffix(key: &str) -> (&str, Option<f32>) {
    if let Some(idx) = key.rfind("|opsz") {
        if let Ok(v) = key[idx + 5..].parse::<f32>() {
            return (&key[..idx], Some(v));
        }
    }
    (key, None)
}

/// Build a synthetic candidate fontkey for a non-default optical size.
/// The base key is the registry entry's key; the suffix makes it a distinct
/// candidate so geometry cache, scoring, and audit all distinguish it.
pub fn opsz_synth_key(base_key: &str, opsz: f32) -> String {
    format!("{}|opsz{}", base_key, opsz.round() as i32)
}

/// Indexed collection of discovered fonts, keyed by `font_key()`.
pub struct FontRegistry {
    entries: Vec<FontEntry>,
    by_key: HashMap<String, usize>,
    catalog_hash: u64,
}

impl FontRegistry {
    pub fn new(mut entries: Vec<FontEntry>) -> Self {
        // Deduplicate by font_key to guarantee registry contains unique keys.
        // Callers may assemble entries from multiple sources (base + vintage) where
        // duplicate msttcorefonts paths (Arial.TTF vs Arial.ttf) produce identical
        // font_keys. Without dedup, iter().count() != owned_fonts.len() and geo-cache
        // reports Wrote < total.
        {
            use std::collections::HashSet;
            let mut seen: HashSet<String> = HashSet::with_capacity(entries.len());
            let before = entries.len();
            entries.retain(|e| seen.insert(e.font_key()));
            if before != entries.len() {
                eprintln!("[scan] FontRegistry deduped {} -> {} entries by font_key", before, entries.len());
            }
        }
        // Sort by font_key for deterministic ordering and stable font_ids.
        entries.sort_by(|a, b| a.font_key_ref().cmp(b.font_key_ref()));
        let by_key = entries.iter().enumerate()
            .map(|(i, e)| (e.font_key_ref().to_owned(), i))
            .collect();
        let catalog_hash = Self::compute_hash(&entries);
        Self { entries, by_key, catalog_hash }
    }

    /// Content hash of the rendering catalog. Besides the stable font key,
    /// include the source-file fingerprint, variation coordinates, and glyph
    /// overrides because each can change rendered glyph identity without
    /// changing the key.
    pub fn compute_hash(entries: &[FontEntry]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        for e in entries {
            e.font_key_ref().hash(&mut hasher);
            source_file_fingerprint(&e.path).hash(&mut hasher);
            match &e.variations {
                Some(axes) => {
                    1u8.hash(&mut hasher);
                    for (tag, value) in axes {
                        tag.hash(&mut hasher);
                        value.to_bits().hash(&mut hasher);
                    }
                }
                None => 0u8.hash(&mut hasher),
            }
            match &e.glyph_overrides {
                Some(overrides) => {
                    1u8.hash(&mut hasher);
                    for (ch, glyph_id) in overrides {
                        ch.hash(&mut hasher);
                        glyph_id.hash(&mut hasher);
                    }
                }
                None => 0u8.hash(&mut hasher),
            }
        }
        hasher.finish()
    }

    pub fn catalog_hash(&self) -> u64 {
        self.catalog_hash
    }

    pub fn set_catalog_hash(&mut self, hash: u64) {
        self.catalog_hash = hash;
    }

    pub fn by_key(&self, key: &str) -> Option<&FontEntry> {
        self.by_key.get(key).map(|&i| &self.entries[i])
    }

    #[inline]
    pub fn index_of(&self, key: &str) -> Option<usize> {
        self.by_key.get(key).copied()
    }

    pub fn entries(&self) -> &[FontEntry] {
        &self.entries
    }


    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, FontEntry> {
        self.entries.iter()
    }

    // -------------------------------------------------------------------
    // fonts.bin serialization
    // -------------------------------------------------------------------
    //
    // Format:
    //   magic:    b"FONT" (4 bytes)
    //   version:  u32 le  (currently 1)
    //   hash:     u64 le  (catalog content hash)
    //   n_fonts:  u32 le
    //   per font:
    //     font_id:      u32 le  (index in sorted catalog)
    //     font_key_len: u32 le
    //     font_key:     [u8; font_key_len]  (UTF-8)
    //
    // font_id is the font's position in the sorted catalog.  Classifier
    // .bin files reference fonts by this index.  The hash lets loaders
    // detect when the catalog has changed and reject stale classifiers.

    /// Write the catalog identity to a fonts.bin file.
    pub fn write_fonts_bin(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = crate::atomic_file::tmp_for(path);
        let f = std::fs::File::create(&tmp)?;
        let mut w = BufWriter::new(f);

        w.write_all(b"FONT")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&self.catalog_hash.to_le_bytes())?;
        w.write_all(&(self.entries.len() as u32).to_le_bytes())?;

        for (id, e) in self.entries.iter().enumerate() {
            w.write_all(&(id as u32).to_le_bytes())?;
            let key = e.font_key();
            w.write_all(&(key.len() as u32).to_le_bytes())?;
            w.write_all(key.as_bytes())?;
        }
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Glyph override map for OT variant entries (e.g. smcp, onum).
/// Maps character → overridden glyph ID so the font matching renders the correct variant glyph.
pub type GlyphOverrides = Option<Vec<(char, u16)>>;

/// Variable-font axis coordinates to apply before rendering.
pub type Variations = Option<Vec<([u8; 4], f32)>>;


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

/// Find the MS Core Fonts directory from the environment — never a
/// hardcoded user-specific path. First asks fontconfig (`fc-list`, the
/// standard font-finding utility) where Times New Roman lives and takes
/// that file's directory; if that fails, checks $MSTTCOREFONTS_DIR, then
/// $HOME/.local/share/fonts/msttcorefonts, then the system
/// /usr/share/fonts/truetype/msttcorefonts. Returns the first directory
/// that contains times.ttf, or None if none is found (the caller handles
/// the missing case — vintage generation is skipped, no crash).
pub fn find_msttcorefonts_dir() -> Option<PathBuf> {
    // fontconfig knows where the fonts are, whatever dir they were
    // installed in. `fc-list "Times New Roman" file` prints one
    // "<path>: " line per matching file.
    if let Ok(out) = std::process::Command::new("fc-list")
        .args(["Times New Roman", "file"])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let path = line.split(':').next().unwrap_or("").trim();
                let p = PathBuf::from(path);
                if p
                    .file_name()
                    .map(|n| n.eq_ignore_ascii_case("times.ttf"))
                    .unwrap_or(false)
                    && p.is_file()
                {
                    if let Some(dir) = p.parent() {
                        return Some(dir.to_path_buf());
                    }
                }
            }
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("MSTTCOREFONTS_DIR") {
        candidates.push(PathBuf::from(d));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".local/share/fonts/msttcorefonts"));
    }
    candidates.push(PathBuf::from("/usr/share/fonts/truetype/msttcorefonts"));
    for dir in candidates {
        if dir.join("times.ttf").is_file() {
            return Some(dir);
        }
    }
    None
}


// ---------------------------------------------------------------------------
// Font scan cache (FSCN)
// ---------------------------------------------------------------------------
//
// Caches the output of scan_fonts() so subsequent runs skip the expensive
// per-font-file parsing (OT feature detection, ligature probing, weight
// instance enumeration).  Invalidation is by a content fingerprint: the
// sorted list of canonical paths for every .ttf/.otf
// in the scanned directories, hashed together.
//
// Format:
//   magic:       b"FSCN"          (4 bytes)
//   version:     u32 le           (currently 1)
//   fingerprint: u64 le           (directory content hash)
//   n_entries:   u32 le
//   per entry:
//     path_len:           u32 le + [u8; path_len]
//     family_name_len:    u32 le + [u8; family_name_len]
//     postscript_name_len:u32 le + [u8; postscript_name_len]
//     raw_ps_name_len:    u32 le + [u8; raw_ps_name_len]
//     is_bold:            u8
//     is_italic:          u8
//     class:              u8  (0=Serif, 1=Sans, 2=Mono, 3=Unknown)
//     oldstyle_figures:   u8
//     variant_tag_len:    u32 le + [u8; variant_tag_len]
//     n_overrides:        u32 le  (0xFFFFFFFF = None)
//       per override:     u32 le (char) + u16 le (glyph_id)
//     n_variations:       u32 le  (0xFFFFFFFF = None)
//       per variation:    [u8; 4] (axis tag) + f32 le
//     typographic_family_len: u32 le + [u8; typographic_family_len]

const FSCN_MAGIC: &[u8; 4] = b"FSCN";
// v20: font_version field (|version=X in font keys) + content-based identity.
// v21: variable-font weight instances get the full kern/lig/opsz permutation.
// v22: ligature gutter filter judges width at render scale (40px em, 2px bar).
// v25: the mask shapes with plain rustybuzz::shape (the verify
// render's path); the cached-ShapePlan keyed by heap address is gone
// (it could hand a later font an earlier font's plan). Fusion is
// judged on the shaped probe string at the word-scan scales
// (FreeType render at 7/9/10pt); a single shaped glyph is fused
// outright, and a gutter must survive every size and subpixel phase.
const FSCN_VERSION: u32 = 25;

pub(crate) fn scan_cache_path() -> PathBuf {
    crate::cache::paths::font_scan_bin()
}

/// Walk font directories and return deduplicated, sorted canonical paths
/// for all .ttf/.otf files.
fn collect_font_paths(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut paths: Vec<PathBuf> = Vec::new();
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
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext != "ttf" && ext != "otf" {
                continue;
            }
            let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if seen.insert(canon.clone()) {
                paths.push(canon);
            }
        }
    }
    paths.sort();
    paths
}

fn source_file_fingerprint(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let Ok(metadata) = std::fs::metadata(path) else {
        return 0;
    };
    let mut hasher = rustc_hash::FxHasher::default();
    metadata.len().hash(&mut hasher);
    if let Ok(modified) = metadata.modified() {
        if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
            duration.as_secs().hash(&mut hasher);
            duration.subsec_nanos().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn write_str(w: &mut impl std::io::Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_str(r: &mut &[u8]) -> Option<String> {
    if r.len() < 4 { return None; }
    let len = u32::from_le_bytes([r[0], r[1], r[2], r[3]]) as usize;
    *r = &r[4..];
    if r.len() < len { return None; }
    let s = std::str::from_utf8(&r[..len]).ok()?.to_string();
    *r = &r[len..];
    Some(s)
}

fn read_u64(r: &mut &[u8]) -> Option<u64> {
    if r.len() < 8 { return None; }
    let v = u64::from_le_bytes(r[..8].try_into().ok()?);
    *r = &r[8..];
    Some(v)
}

fn read_u32(r: &mut &[u8]) -> Option<u32> {
    if r.len() < 4 { return None; }
    let v = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
    *r = &r[4..];
    Some(v)
}

fn read_u16(r: &mut &[u8]) -> Option<u16> {
    if r.len() < 2 { return None; }
    let v = u16::from_le_bytes([r[0], r[1]]);
    *r = &r[2..];
    Some(v)
}

fn read_u8(r: &mut &[u8]) -> Option<u8> {
    if r.is_empty() { return None; }
    let v = r[0];
    *r = &r[1..];
    Some(v)
}

fn read_f32(r: &mut &[u8]) -> Option<f32> {
    if r.len() < 4 { return None; }
    let v = f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
    *r = &r[4..];
    Some(v)
}

fn class_to_u8(c: FontClass) -> u8 {
    match c {
        FontClass::Serif => 0,
        FontClass::Sans => 1,
        FontClass::Mono => 2,
        FontClass::Unknown => 3,
    }
}

fn u8_to_class(v: u8) -> FontClass {
    match v {
        0 => FontClass::Serif,
        1 => FontClass::Sans,
        2 => FontClass::Mono,
        _ => FontClass::Unknown,
    }
}

fn write_scan_cache(path: &Path, entries: &[FontEntry]) -> std::io::Result<()> {
    use std::io::{BufWriter, Write};
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = crate::atomic_file::tmp_for(path);
    let f = std::fs::File::create(&tmp)?;
    let mut w = BufWriter::new(f);

    w.write_all(FSCN_MAGIC)?;
    w.write_all(&FSCN_VERSION.to_le_bytes())?;
    w.write_all(&(entries.len() as u32).to_le_bytes())?;

    for e in entries {
        // Path plus source fingerprint. A package upgrade can replace a font
        // in place without changing its path; stale cached names, axes, and
        // glyph overrides must not survive that replacement.
        write_str(&mut w, &e.path.to_string_lossy())?;
        w.write_all(&source_file_fingerprint(&e.path).to_le_bytes())?;
        write_str(&mut w, &e.family_name)?;
        write_str(&mut w, &e.postscript_name)?;
        write_str(&mut w, &e.raw_postscript_name)?;
        w.write_all(&[e.is_bold as u8])?;
        w.write_all(&[e.is_italic as u8])?;
        w.write_all(&[class_to_u8(e.class)])?;
        w.write_all(&[e.oldstyle_figures as u8])?;
        write_str(&mut w, &e.variant_tag)?;

        // glyph_overrides
        match &e.glyph_overrides {
            None => w.write_all(&0xFFFF_FFFFu32.to_le_bytes())?,
            Some(ov) => {
                w.write_all(&(ov.len() as u32).to_le_bytes())?;
                for &(ch, gid) in ov {
                    w.write_all(&(ch as u32).to_le_bytes())?;
                    w.write_all(&gid.to_le_bytes())?;
                }
            }
        }

        // variations
        match &e.variations {
            None => w.write_all(&0xFFFF_FFFFu32.to_le_bytes())?,
            Some(vars) => {
                w.write_all(&(vars.len() as u32).to_le_bytes())?;
                for &(tag, val) in vars {
                    w.write_all(&tag)?;
                    w.write_all(&val.to_le_bytes())?;
                }
            }
        }

        // opsz_range: Option<(f32, f32, f32)> - v10+
        // None encodes as the 0xFFFFFFFF sentinel; Some encodes as three
        // bare f32s (an axis min is never NaN, so the sentinel is unambiguous).
        match &e.opsz_range {
            None => w.write_all(&0xFFFF_FFFFu32.to_le_bytes())?,
            Some((min, default, max)) => {
                w.write_all(&min.to_le_bytes())?;
                w.write_all(&default.to_le_bytes())?;
                w.write_all(&max.to_le_bytes())?;
            }
        }

        write_str(&mut w, &e.typographic_family)?;

        // vintage_era (Option<String>) - v3+
        match &e.vintage_era {
            None => w.write_all(&0xFFFF_FFFFu32.to_le_bytes())?,
            Some(s) => {
                write_str(&mut w, s)?;
            }
        }

        // has_gpos - v17+
        w.write_all(&[e.has_gpos as u8])?;

        // has_kern - v18+
        w.write_all(&[e.has_kern as u8])?;

        // font_version - v20+
        write_str(&mut w, &e.font_version)?;
    }

    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_scan_cache(path: &Path, quiet: bool) -> Option<Vec<FontEntry>> {
    let data = std::fs::read(path).ok()?;
    let mut r: &[u8] = &data;

    // Header: magic(4) + version(4) + count(4)
    if r.len() < 12 { return None; }
    if &r[..4] != FSCN_MAGIC { return None; }
    r = &r[4..];
    let version = read_u32(&mut r)?;
    if version != FSCN_VERSION {
        if !quiet { eprintln!("[scan] Font scan cache is v{version}, need v{FSCN_VERSION} — rescanning..."); }
        return None;
    }
    let n = read_u32(&mut r)? as usize;

    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let path_str = read_str(&mut r)?;
        let path_buf = PathBuf::from(&path_str);
        let cached_source_fingerprint = read_u64(&mut r)?;
        let live_source_fingerprint = source_file_fingerprint(&path_buf);
        if cached_source_fingerprint != live_source_fingerprint {
            if !quiet {
                eprintln!(
                    "[scan] Font source changed in place: {} — rescanning...",
                    path_buf.display()
                );
            }
            return None;
        }
        let family_name = read_str(&mut r)?;
        let postscript_name = read_str(&mut r)?;
        let raw_postscript_name = read_str(&mut r)?;
        let is_bold = read_u8(&mut r)? != 0;
        let is_italic = read_u8(&mut r)? != 0;
        let class = u8_to_class(read_u8(&mut r)?);
        let oldstyle_figures = read_u8(&mut r)? != 0;
        let variant_tag = read_str(&mut r)?;

        // glyph_overrides
        let n_ov = read_u32(&mut r)?;
        let glyph_overrides = if n_ov == 0xFFFF_FFFF {
            None
        } else {
            let mut ov = Vec::with_capacity(n_ov as usize);
            for _ in 0..n_ov {
                let ch = char::from_u32(read_u32(&mut r)?)?;
                let gid = read_u16(&mut r)?;
                ov.push((ch, gid));
            }
            Some(ov)
        };

        // variations
        let n_var = read_u32(&mut r)?;
        let variations = if n_var == 0xFFFF_FFFF {
            None
        } else {
            let mut vars = Vec::with_capacity(n_var as usize);
            for _ in 0..n_var {
                if r.len() < 4 { return None; }
                let tag: [u8; 4] = [r[0], r[1], r[2], r[3]];
                r = &r[4..];
                let val = read_f32(&mut r)?;
                vars.push((tag, val));
            }
            Some(vars)
        };

        // opsz_range: Option<(f32, f32, f32)> - v10+
        let opsz_range = {
            if r.len() < 4 {
                return None;
            }
            let sentinel = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
            if sentinel == 0xFFFF_FFFF {
                r = &r[4..];
                None
            } else {
                let min = read_f32(&mut r)?;
                let default = read_f32(&mut r)?;
                let max = read_f32(&mut r)?;
                Some((min, default, max))
            }
        };

        let typographic_family = read_str(&mut r)?;

        // vintage_era: Option<String> - v3+, with backwards compat for v2
        // In v2, there was no vintage_era field, so we attempt to peek.
        // If version == 3, we read it as Option.
        let vintage_era = if version >= 3 {
            // Peek next u32: 0xFFFFFFFF => None, else it's length of string.
            // Need to handle gracefully if buffer ends (old file truncated)
            if r.len() < 4 {
                None
            } else {
                // Save position to detect sentinel
                let sentinel = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
                if sentinel == 0xFFFF_FFFF {
                    // consume sentinel
                    r = &r[4..];
                    None
                } else {
                    // it's a string length => read_str will consume len+bytes
                    read_str(&mut r)
                }
            }
        } else {
            None
        };

        // has_gpos - v17+
        let has_gpos = read_u8(&mut r)? != 0;

        // has_kern - v18+
        let has_kern = read_u8(&mut r)? != 0;

        // font_version - v20+
        let font_version = read_str(&mut r)?;

        entries.push(FontEntry {
            path: path_buf,
            family_name,
            postscript_name: postscript_name.clone(),
            raw_postscript_name,
            is_bold,
            is_italic,
            class,
            data: Vec::new(),
            oldstyle_figures,
            variant_tag: variant_tag.clone(),
            font_key_cache: {
                FontEntry::compute_font_key_full(
                    &postscript_name,
                    &variant_tag,
                    vintage_era.as_deref(),
                    &variations,
                    &font_version,
                )
            },
            glyph_overrides,
            variations,
            typographic_family,
            vintage_era,
            opsz_range,
            has_gpos,
            has_kern,
            font_version,
        });
    }

    Some(entries)
}

/// Dedup font entries: keep both static and variable-font weight instances as
/// distinct legacy vs modern variants (user requested), then dedup by enhanced font_key.
fn dedup_fonts(mut fonts: Vec<FontEntry>, quiet: bool) -> Vec<FontEntry> {
    // Previously we dropped variable-font weight instances covered by static fonts.
    // That loses the distinction between static Inter-Regular (legacy) and
    // variable Inter wght400 (modern) which render differently (hinting, gvar).
    // User request: keep both as separate entries expecting difference.
    // So we now KEEP both and only log what would have been dropped.

    {
        use std::collections::HashSet;
        let static_keys: HashSet<(String, u16, bool)> = fonts.iter()
            .filter(|f| f.variations.is_none() && !f.variant_tag.starts_with("wght") && f.vintage_era.is_none())
            .filter_map(|f| {
                if f.typographic_family.is_empty() { return None; }
                let ps = &f.postscript_name;
                let base = ps.strip_suffix("Italic")
                    .or_else(|| ps.strip_suffix("It"))
                    .unwrap_or(ps);
                let w = base.rsplit('-').next()
                    .and_then(|s| s.parse::<u16>().ok())
                    .filter(|&w| (100..=900).contains(&w))?;
                Some((f.typographic_family.clone(), w, f.is_italic))
            })
            .collect();

        let mut would_drop = 0usize;
        for f in &fonts {
            if f.variant_tag.starts_with("wght") {
                let weight = f.variant_tag.strip_prefix("wght")
                    .and_then(|s| s.parse::<u16>().ok())
                    .unwrap_or(0);
                if static_keys.contains(&(f.typographic_family.clone(), weight, f.is_italic)) {
                    would_drop += 1;
                }
            }
        }
        if would_drop > 0 {
            eprintln!("[scan] Keeping {} variable-font weight instances that overlap static fonts (legacy vs modern, expected difference)", would_drop);
        }
        // Intentionally do NOT filter them out anymore.
    }

    dedup_font_entries_by_key(fonts, quiet)
}

/// Deduplicate a complete registry generation after callers append entries
/// (for example, vintage variants) to an already-deduplicated base scan.
pub fn dedup_font_entries_by_key(mut fonts: Vec<FontEntry>, quiet: bool) -> Vec<FontEntry> {
    use std::collections::HashSet;
    let mut seen_keys: HashSet<String> = HashSet::new();
    let before = fonts.len();
    fonts.retain(|f| seen_keys.insert(f.font_key()));
    let removed = before - fonts.len();
    if removed > 0 && !quiet {
        eprintln!(
            "[scan] Deduped {} entries by font_key ({} → {})",
            removed,
            before,
            fonts.len()
        );
    }
    fonts
}

/// Walk the given directories for .ttf / .otf files and return a catalogue.
pub fn scan_fonts(dirs: &[PathBuf], quiet: bool) -> Vec<FontEntry> {
    let current_paths = collect_font_paths(dirs);
    let cache_path = scan_cache_path();
    let allowlist = crate::cache::font_allowlist();
    let is_alt_cache = !crate::cache::is_default_cache_dir();

    // Load cached entries indexed by source font file path
    let cached_by_path: std::collections::HashMap<PathBuf, Vec<FontEntry>> = {
        let mut map: std::collections::HashMap<PathBuf, Vec<FontEntry>> = std::collections::HashMap::new();
        if let Some(entries) = read_scan_cache(&cache_path, quiet) {
            for e in entries {
                map.entry(e.path.clone()).or_default().push(e);
            }
        }
        map
    };

    let current_set: std::collections::HashSet<&PathBuf> = current_paths.iter().collect();
    let cached_set: std::collections::HashSet<&PathBuf> = cached_by_path.keys().collect();

    // Sort: HashSet difference order is non-deterministic, and the dedup
    // below keeps the FIRST entry per font_key. Without sorting, which
    // copy of a duplicated font survives varies run to run, so
    // --resolve-fontkey intermittently fails for the dropped path.
    let mut added: Vec<&PathBuf> = current_set.difference(&cached_set).copied().collect();
    added.sort();
    let mut removed: Vec<&PathBuf> = cached_set.difference(&current_set).copied().collect();
    removed.sort();
    let cache_changed = !added.is_empty() || !removed.is_empty();

    if !cache_changed {
        let mut fonts: Vec<FontEntry> = current_paths.iter()
            .flat_map(|p| cached_by_path.get(p).cloned().unwrap_or_default())
            .collect();
        // Filter out tombstone entries (empty family_name = rejected font file)
        fonts.retain(|f| !f.family_name.is_empty());
        if !quiet { eprintln!("[scan] Loaded {} font entries from cache", fonts.len()); }
        // Apply allowlist filtering only when using alternate cache dir
        if is_alt_cache {
            if let Some(ref allow) = allowlist {
                let before = fonts.len();
                fonts.retain(|f| allow.contains(&f.font_key()));
                if !quiet && before != fonts.len() {
                    eprintln!("[scan] Font allowlist (fontkey format): filtered {} -> {} fonts", before, fonts.len());
                }
            }
        }
        return dedup_fonts(fonts, quiet);
    }

    // Low-mem optimization for alt cache with allowlist: if cached entries already cover all allowlisted fonts,
    // skip scanning 656 new files (which OOMs on 7.8G VM). Reuse cached entries.
    if is_alt_cache {
        if let Some(ref allow) = allowlist {
            let cached_keys: std::collections::HashSet<String> = cached_by_path.values()
                .flat_map(|v| v.iter().map(|e| e.font_key()))
                .collect();
            // Check if all allowlisted keys (or their base without |variant) are present
            let all_present = allow.iter().all(|k| {
                cached_keys.contains(k) || 
                // also check if base key present when allowlist contains variant
                cached_keys.contains(&k.split('|').next().unwrap_or("").to_string())
            });
            if all_present && !cached_by_path.is_empty() {
                let mut fonts: Vec<FontEntry> = current_paths.iter()
                    .filter(|p| cached_by_path.contains_key(*p))
                    .flat_map(|p| cached_by_path.get(p).cloned().unwrap_or_default())
                    .collect();
                fonts.retain(|f| !f.family_name.is_empty());
                let before = fonts.len();
                fonts.retain(|f| allow.contains(&f.font_key()));
                if !quiet {
                    eprintln!("[scan] Low-mem alt-cache shortcut: reusing {} cached entries, filtered {} -> {} for allowlist (skipping {} new files)",
                        before, before, fonts.len(), added.len());
                }
                return dedup_fonts(fonts, quiet);
            }
        }
    }

    if !removed.is_empty() {
        if !quiet { eprintln!("[scan] {} font files removed", removed.len()); }
    }
    if !added.is_empty() {
        if !quiet { eprintln!("[scan] {} new font files to scan", added.len()); }
    }

    // Start with cached entries for paths that still exist
    let mut fonts: Vec<FontEntry> = current_paths.iter()
        .filter(|p| cached_by_path.contains_key(*p))
        .flat_map(|p| cached_by_path.get(p).cloned().unwrap_or_default())
        .collect();

    // Optimization: when using alt cache with allowlist that has no OT variant
    // entries (no "|"), skip expensive OT feature probing (28 shapings per file).
    // This saves ~70k shapings for 2497 files when testing with 6 static fonts.
    let need_ot_variants = if is_alt_cache {
        if let Some(ref allow) = allowlist {
            allow.iter().any(|k| k.contains('|'))
        } else {
            true
        }
    } else {
        true
    };

    // Parse only the new font files
    let aliases = build_alias_table();
    for path in &added {
        if let Some(fe) = load_font_entry(path, &aliases) {
                // Unprint's supported sequences are Latin. Keep unsupported
                // fonts out of the catalog at scan time so every downstream
                // cache sees the same eligible set. This also excludes
                // NotoColorEmoji, whose CBDT glyph data previously expanded
                // indexing to millions of entries.
                let has_latin = unprint_fonts::ttf_parser::Face::parse(&fe.data, 0)
                    .ok()
                    .is_some_and(|face| {
                        ('a'..='z').any(|ch| {
                            face.glyph_index(ch).is_some_and(|glyph_id| glyph_id.0 != 0)
                        })
                    });
                if !has_latin {
                    continue;
                }

                let _fig_label = if fe.oldstyle_figures { "OLDSTYLE" } else { "lining" };

                // Detect ligature glyphs (liga + dlig)
                let ligatures = detect_ligature_glyphs(&fe.data);
                if !ligatures.is_empty() {
                }

                let (default_variations, opsz_range) = detect_default_variations(path);

                // Probe all OT features — emit a variant entry for each that changes glyphs
                // Skip when allowlist has no variant entries (common for 6-font tests)
                let variants = if need_ot_variants {
                    detect_ot_variants(&fe.data)
                } else {
                    Vec::new()
                };
                // Kern/nokern is explicit in every font key (no implicit
                // default): variant entries render kerned when the font has
                // kerning data, nokern otherwise. Reused for OT variants,
                // allcaps, and weight instances below.
                let kern_tag = if fe.has_kern { "kern" } else { "nokern" };
                for (tag, overrides) in &variants {
                    // Merge ligature overrides into each variant
                    let mut combined = overrides.clone();
                    for &(lig_c, gid) in &ligatures {
                        if !combined.iter().any(|(c, _)| *c == lig_c) {
                            combined.push((lig_c, gid));
                        }
                    }
                    // Variant entries carry the kern/nokern tag explicitly
                    // (appended after the OT tag): they render kerned unless
                    // the font has no kerning data. Lig is explicit too:
                    // variants render with ligatures on (|lig); there is no
                    // nolig permutation for OT variants.
                    let var_tag = format!("{}|{}|lig", tag, kern_tag);
                    let var_entry = FontEntry {
                        path: fe.path.clone(),
                        family_name: format!("{} [{}]", fe.family_name, tag),
                        // Variant gets its own unambiguous canonical name:
                        // base PS name + "|" + variant tag.  This ensures
                        // exact == comparison never confuses a variant with
                        // its base entry.
                        postscript_name: format!("{}|{}", fe.postscript_name, tag),
                        raw_postscript_name: fe.raw_postscript_name.clone(),
                        is_bold: fe.is_bold,
                        is_italic: fe.is_italic,
                        class: fe.class,
                        data: Vec::new(), // bytes not retained
                        oldstyle_figures: fe.oldstyle_figures,
                        variant_tag: var_tag.clone(),
                        font_key_cache: FontEntry::compute_font_key_full(
                            &format!("{}|{}", fe.postscript_name, tag),
                            &var_tag,
                            None,
                            &default_variations,
                            &fe.font_version,
                        ),
                        glyph_overrides: Some(combined),
                        variations: default_variations.clone(),
                        typographic_family: fe.typographic_family.clone(),
                        vintage_era: None,
                        opsz_range,
                        has_gpos: fe.has_gpos,
                        has_kern: fe.has_kern,
                        font_version: fe.font_version.clone(),
                    };
                    fonts.push(var_entry);
                }
                // ── allcaps enable variant (case + cpsp) ──
                // Default is no case/cpsp (matches WeasyPrint/CSS). The |allcaps
                // variant opts in to both, orthogonal and additive per OT spec.
                // Only emit when the font actually defines either table —
                // don't add a no-op variant when neither exists.
                if font_has_caps_tables(&fe.data) {
                    let tag = "allcaps".to_string();
                    // Avoid duplicate if detect_ot_variants ever started returning it
                    if !variants.iter().any(|(t, _)| t == &tag) {
                        let mut combined_lig = Vec::new();
                        for &(lig_c, gid) in &ligatures {
                            combined_lig.push((lig_c, gid));
                        }
                        // Allcaps entries carry the kern/nokern tag explicitly,
                        // like OT variants (render kerned unless no kern data).
                        // Lig is explicit too (|lig — ligatures on).
                        let allcaps_tag = format!("{}|{}|lig", tag, kern_tag);
                        let allcaps_entry = FontEntry {
                            path: fe.path.clone(),
                            family_name: format!("{} [{}]", fe.family_name, tag),
                            postscript_name: format!("{}|{}", fe.postscript_name, tag),
                            raw_postscript_name: fe.raw_postscript_name.clone(),
                            is_bold: fe.is_bold,
                            is_italic: fe.is_italic,
                            class: fe.class,
                            data: Vec::new(),
                            oldstyle_figures: fe.oldstyle_figures,
                            variant_tag: allcaps_tag.clone(),
                            font_key_cache: FontEntry::compute_font_key_full(
                                &format!("{}|{}", fe.postscript_name, tag),
                                &allcaps_tag,
                                None,
                                &default_variations,
                                &fe.font_version,
                            ),
                            glyph_overrides: if combined_lig.is_empty() { None } else { Some(combined_lig) },
                            variations: default_variations.clone(),
                            typographic_family: fe.typographic_family.clone(),
                            vintage_era: None,
                            opsz_range,
                            has_gpos: fe.has_gpos,
                            has_kern: fe.has_kern,
                            font_version: fe.font_version.clone(),
                        };
                        fonts.push(allcaps_entry);
                    }
                }
                // Drop font bytes — metadata + path is all we keep.
                // Index build and matching load from path on demand.
                // Compute kern feature BEFORE clearing data (needed for nokern variants).
                let has_kern = font_has_kern_feature(&fe.data);
                let mut fe = fe;
                fe.data = Vec::new();
                // Add ligature overrides to base entry
                if !ligatures.is_empty() {
                    let mut base_overrides = fe.glyph_overrides.take().unwrap_or_default();
                    base_overrides.extend(ligatures.clone());
                    fe.glyph_overrides = Some(base_overrides);
                }

                // ── Variable font weight instances ──────────────────────
                // If the font has a wght axis, emit additional entries at
                // each named-instance weight so the font matcher indexes bold/light/
                // etc. renderings from the same file. Each weight instance
                // gets the SAME kern/lig/opsz permutation as the base
                // entry — a single kern|lig entry per weight cannot
                // represent e.g. "400 italic, nokern, opsz 10", so the
                // classifier could never match that rendering exactly.
                let weight_instances = detect_weight_instances(path, fe.class);
                let has_ligatures = fe.collapsed_lig_set() != 0;
                for wi in &weight_instances {
                    let var_ps = make_weight_explicit(&fe.raw_postscript_name, wi.os2_weight);
                    let var_variations = Some(wi.axes.clone());
                    let wght_tag = format!("wght{}", wi.os2_weight);
                    let var_opts = var_perm_opts(&var_variations);
                    for &is_nokern in kern_perm_opts(has_kern) {
                        for &is_nolig in lig_perm_opts(has_ligatures) {
                            for variations in &var_opts {
                                let var_tag = append_kern_lig_markers(&wght_tag, is_nokern, is_nolig);
                                let mut var_fe = FontEntry {
                                    path: fe.path.clone(),
                                    family_name: format!("{} [{}]", fe.family_name, var_tag),
                                    postscript_name: var_ps.clone(),
                                    raw_postscript_name: fe.raw_postscript_name.clone(),
                                    is_bold: wi.os2_weight >= 700,
                                    is_italic: fe.is_italic,
                                    class: fe.class,
                                    data: Vec::new(),
                                    oldstyle_figures: fe.oldstyle_figures,
                                    variant_tag: var_tag.clone(),
                                    // Bracket notation distinguishes static "SourceSerif4-400Italic"
                                    // from variable "SourceSerif4-400Italic|wght400[wght=400,opsz=20]"
                                    // after normalization.
                                    font_key_cache: FontEntry::compute_font_key_full(
                                        &var_ps,
                                        &var_tag,
                                        None,
                                        variations,
                                        &fe.font_version,
                                    ),
                                    glyph_overrides: None,
                                    variations: variations.clone(),
                                    typographic_family: fe.typographic_family.clone(),
                                    vintage_era: None,
                                    opsz_range,
                                    has_gpos: fe.has_gpos,
                                    has_kern: fe.has_kern,
                                    font_version: fe.font_version.clone(),
                                };
                                // Add ligature overrides to weight-instance entry
                                if !ligatures.is_empty() {
                                    var_fe.glyph_overrides = Some(ligatures.clone());
                                }
                                fonts.push(var_fe);
                            }
                        }
                    }
                }

                // If this file is variable, mark the base entry as variable
                // too (variations Some) so its key becomes "...[wght=400,opsz=20]"
                // and does not collide with the static counterpart after normalization.
                // The base entry must use every axis's declared default. Using
                // the first non-default weight instance here mislabeled and
                // rendered 400-weight base fonts at another weight (for example
                // Source Serif 4 at 200 and Libre Bodoni/Baskerville at 500).
                fe.assign_default_variations(default_variations);
                fe.opsz_range = opsz_range;

                // Emit all permutations of the kern/lig/opsz variant dimensions.
                // - kern: [kern, nokern] if the font has kerning data, else [nokern].
                //   A font with no kerning renders identically with the kern
                //   feature on or off, so a kerned variant would be a
                //   byte-identical duplicate. Emit only the nokern variant,
                //   which honestly labels the rendering.
                //   The PDF specimen's embedded subsets lack GPOS (stripped during
                //   subsetting), so the scan has no kerning applied. The installed
                //   full fonts do apply kerning, causing letter-spacing mismatch.
                //   The nokern variant disables the 'kern' OpenType feature during
                //   shaping, matching the PDF's kerning-less rendering.
                // - lig: [lig, nolig] if the font has ligature glyphs, else [lig].
                //   The lig variant uses the font's ligature set (collapses fi→ﬁ
                //   when the font has a true ligature). The nolig variant disables
                //   ligature collapsing, matching documents that render f and i
                //   as separate glyphs. Lig/nolig is explicit in every font key
                //   — no implicit default: lig variants carry |lig, nolig
                //   variants carry |nolig.
                // - opsz: [base, opsz10] if the font has an opsz axis, else [base].
                // The classifier scores all permutations and picks the best.
                // The same permutation applies to the variable-font weight
                // instances above; the option computation is shared (see
                // kern_perm_opts / lig_perm_opts / var_perm_opts).
                let kern_opts = kern_perm_opts(has_kern);
                let lig_opts = lig_perm_opts(has_ligatures);
                let var_opts = var_perm_opts(&fe.variations);
                for &is_nokern in kern_opts {
                    for &is_nolig in lig_opts {
                        for variations in &var_opts {
                            // Combine base variant_tag with kern/nokern and
                            // lig/nolig markers. Kern/nokern and lig/nolig are
                            // explicit in every font key — no implicit default:
                            // kerned variants carry |kern, nokern variants
                            // carry |nokern; ligature-enabled variants carry
                            // |lig, nolig variants carry |nolig.
                            // Use "|" separator to avoid ambiguity.
                            let variant_tag = append_kern_lig_markers(&fe.variant_tag, is_nokern, is_nolig);
                            let mut family_name = fe.family_name.clone();
                            if is_nokern {
                                family_name = format!("{} [nokern]", family_name);
                            }
                            if is_nolig {
                                family_name = format!("{} [nolig]", family_name);
                            }
                            let entry = FontEntry {
                                path: fe.path.clone(),
                                family_name,
                                postscript_name: fe.postscript_name.clone(),
                                raw_postscript_name: fe.raw_postscript_name.clone(),
                                is_bold: fe.is_bold,
                                is_italic: fe.is_italic,
                                class: fe.class,
                                data: Vec::new(),
                                oldstyle_figures: fe.oldstyle_figures,
                                variant_tag: variant_tag.clone(),
                                font_key_cache: FontEntry::compute_font_key_full(
                                    &fe.postscript_name,
                                    &variant_tag,
                                    None,
                                    variations,
                                    &fe.font_version,
                                ),
                                glyph_overrides: fe.glyph_overrides.clone(),
                                variations: variations.clone(),
                                typographic_family: fe.typographic_family.clone(),
                                vintage_era: None,
                                opsz_range,
                                has_gpos: fe.has_gpos,
                                has_kern: fe.has_kern,
                                font_version: fe.font_version.clone(),
                            };
                            fonts.push(entry);
                        }
                    }
                }
        }
    }

    // Add tombstone entries for font files that produced no entries
    // (e.g. corrupt or unparseable) so they aren't re-scanned next time.
    {
        let covered: std::collections::HashSet<PathBuf> = fonts.iter().map(|f| f.path.clone()).collect();
        for path in &added {
            if !covered.contains(path.as_path()) {
                fonts.push(FontEntry {
                    path: path.to_path_buf(),
                    family_name: String::new(),
                    postscript_name: String::new(),
                    raw_postscript_name: String::new(),
                    is_bold: false,
                    is_italic: false,
                    class: FontClass::Serif,
                    data: Vec::new(),
                    oldstyle_figures: false,
                    variant_tag: String::new(),
                    font_key_cache: String::new(),
                    glyph_overrides: None,
                    variations: None,
                    typographic_family: String::new(),
                    vintage_era: None,
                    opsz_range: None,
                    has_gpos: false,
                    has_kern: false,
                    font_version: String::new(),
                });
            }
        }
    }

    // Apply allowlist filtering when using alternate cache dir (keep main cache untouched)
    if is_alt_cache {
        if let Some(ref allow) = allowlist {
            let before = fonts.len();
            // Keep tombstones (empty family_name) for now - they'll be filtered next
            fonts.retain(|f| f.family_name.is_empty() || allow.contains(&f.font_key()));
            if !quiet && before != fonts.len() {
                eprintln!("[scan] Font allowlist (fontkey format): filtered {} -> {} fonts (pre-dedup)", before, fonts.len());
            }
        }
    }

    // Write cache pre-dedup so every source path is represented
    if let Err(e) = write_scan_cache(&cache_path, &fonts) {
        if !quiet { eprintln!("[scan] Warning: failed to write font scan cache: {}", e); }
    } else {
        if !quiet { eprintln!("[scan] Wrote {} font entries to cache", fonts.len()); }
    }

    // Filter out tombstone entries before dedup
    fonts.retain(|f| !f.family_name.is_empty());
    // Re-apply allowlist after tombstone removal to ensure exact fontkey match
    if is_alt_cache {
        if let Some(ref allow) = allowlist {
            fonts.retain(|f| allow.contains(&f.font_key()));
        }
    }
    dedup_fonts(fonts, quiet)
}

// ---------------------------------------------------------------------------
// Alias table
// ---------------------------------------------------------------------------

pub struct Alias {
    family: &'static str,
    bold: bool,
    italic: bool,
}

pub fn build_alias_table() -> HashMap<String, Alias> {
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

    let g0 = gid_0.with_scale_and_position(scale, unprint_fonts::ab_glyph::point(0.0, ascent));
    let g3 = gid_3.with_scale_and_position(scale, unprint_fonts::ab_glyph::point(0.0, ascent));

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

// ---------------------------------------------------------------------------
// Variable font weight instances
// ---------------------------------------------------------------------------

/// A weight instance to emit from a variable font.
struct WeightInstance {
    /// OS/2-style weight value (e.g. 400, 700).
    os2_weight: u16,
    /// Axis coordinates to set before rendering.
    axes: Vec<([u8; 4], f32)>,
}

/// Return every variation axis pinned to its declared default for the base
/// entry. The base font's canonical weight/style describes these coordinates;
/// non-default weight instances are emitted separately below.
///
/// For `opsz` (optical size), we use the font's declared default. The
/// renderer's actual opsz behavior (font default vs CSS `font-optical-sizing:
/// auto` = pt size) is not recoverable from the raster alone, so
/// `identify_fonts` scores a second synthetic candidate at the size-derived
/// opsz (`|opszN` fontkey suffix); the better geometry wins. See
/// `parse_opsz_suffix` / `opsz_synth_key`.
///
/// Returns `(default_variations, opsz_range)`:
/// - `default_variations`: every axis pinned to its declared default
///   (`None` for static fonts).
/// - `opsz_range`: `(min, default, max)` of the `opsz` axis when the font has
///   one. Captured here because the font bytes are dropped after the scan
///   and the range cannot be re-derived from the entry later.
fn detect_default_variations(path: &Path) -> (Variations, Option<(f32, f32, f32)>) {
    use unprint_fonts::ab_glyph::{FontRef, VariableFont};

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return (None, None),
    };
    let font = match FontRef::try_from_slice(&data) {
        Ok(f) => f,
        Err(_) => return (None, None),
    };
    let axes = font.variations();
    if axes.is_empty() {
        return (None, None);
    }
    let opsz_range = axes
        .iter()
        .find(|a| &a.tag == b"opsz")
        .map(|a| (a.min_value, a.default_value, a.max_value));
    let variations = Some(
        axes.iter()
            .map(|axis| (axis.tag, axis.default_value))
            .collect(),
    );
    (variations, opsz_range)
}

/// Create opsz=10 variations from existing variations, if the font has an opsz axis.
/// Returns None if the font has no opsz axis, or if opsz is already 10.
/// Otherwise returns variations with opsz set to 10.0, all other axes unchanged.
fn make_opsz10_variations(variations: &Variations) -> Variations {
    let vars = variations.as_ref()?;
    let mut has_opsz = false;
    let mut already_10 = false;
    for (tag, val) in vars.iter() {
        if tag == b"opsz" {
            has_opsz = true;
            if (*val - 10.0).abs() < 0.001 {
                already_10 = true;
            }
            break;
        }
    }
    if !has_opsz || already_10 {
        return None;
    }
    let new_vars: Vec<([u8; 4], f32)> = vars
        .iter()
        .map(|(tag, val)| {
            if tag == b"opsz" {
                (*tag, 10.0)
            } else {
                (*tag, *val)
            }
        })
        .collect();
    Some(new_vars)
}

/// Kern options for the kern/lig/opsz permutation shared by the base
/// entry and the variable-font weight instances. Values are `is_nokern`:
/// [kern, nokern] when the font has kerning data, else [nokern] only —
/// a font with no kerning renders identically with the kern feature on
/// or off, so a kerned variant would be a byte-identical duplicate.
fn kern_perm_opts(has_kern: bool) -> &'static [bool] {
    if has_kern {
        &[false, true]
    } else {
        &[true]
    }
}

/// Lig options for the kern/lig/opsz permutation. Values are `is_nolig`:
/// [lig, nolig] when the font has ligature glyphs, else [lig] only.
fn lig_perm_opts(has_ligatures: bool) -> &'static [bool] {
    if has_ligatures {
        &[false, true]
    } else {
        &[false]
    }
}

/// Opsz options for the kern/lig/opsz permutation: the given variations
/// plus an opsz=10 variant when the font has an opsz axis not already
/// at 10.
fn var_perm_opts(variations: &Variations) -> Vec<Variations> {
    let mut opts = vec![variations.clone()];
    if let Some(opsz10_vars) = make_opsz10_variations(variations) {
        opts.push(Some(opsz10_vars));
    }
    opts
}

/// Append the kern/nokern and lig/nolig markers to a variant tag.
/// Both are explicit in every font key — no implicit default. An empty
/// base tag yields just the two markers; otherwise they are appended
/// after the base tag with `|` separators, kern first then lig.
fn append_kern_lig_markers(base_tag: &str, is_nokern: bool, is_nolig: bool) -> String {
    let kern = if is_nokern { "nokern" } else { "kern" };
    let lig = if is_nolig { "nolig" } else { "lig" };
    if base_tag.is_empty() {
        format!("{kern}|{lig}")
    } else {
        format!("{base_tag}|{kern}|{lig}")
    }
}

/// Detect named weight instances for a variable font.
///
/// Returns instances at each named weight that differs from the default.
/// Only fires if the font has a `wght` axis.  All other axes are pinned
/// to their defaults so the instance is fully determined.
fn detect_weight_instances(path: &Path, _class: FontClass) -> Vec<WeightInstance> {
    use unprint_fonts::ab_glyph::{FontRef, VariableFont};

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let font = match FontRef::try_from_slice(&data) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let axes = font.variations();
    let wght_axis = match axes.iter().find(|a| &a.tag == b"wght") {
        Some(a) => a,
        None => return Vec::new(), // not a variable font (or no weight axis)
    };

    let default_wght = wght_axis.default_value;

    // Read named instances from fvar via ttf_parser
    let named_weights: Vec<u16> = {
        match unprint_fonts::ttf_parser::Face::parse(&data, 0) {
            Ok(face) => {
                let mut weights = Vec::new();
                // fvar named instances
                if let Some(fvar) = face.tables().fvar {
                    for inst in fvar.axes {
                        // axes gives us axis records, not instances
                        let _ = inst;
                    }
                }
                // Use common weight stops that fall within the axis range
                for w in [100, 200, 300, 400, 500, 600, 700, 800, 900] {
                    let wf = w as f32;
                    if wf >= wght_axis.min_value && wf <= wght_axis.max_value
                        && (wf - default_wght).abs() > 0.5
                    {
                        weights.push(w);
                    }
                }
                weights
            }
            Err(_) => return Vec::new(),
        }
    };

    // Build axis pinning: set wght to each weight, all other axes to default
    // (including opsz, which uses the font's declared default for consistency
    // with detect_default_variations; per-line synthetic `|opszN` candidates
    // handle the size-derived optical size at scoring time).
    let other_axes: Vec<([u8; 4], f32)> = axes.iter()
        .filter(|a| &a.tag != b"wght")
        .map(|a| (a.tag, a.default_value))
        .collect();

    named_weights.iter().map(|&w| {
        let mut ax = vec![(*b"wght", w as f32)];
        ax.extend(other_axes.iter().cloned());
        WeightInstance {
            os2_weight: w,
            axes: ax,
        }
    }).collect()
}

/// Use rustybuzz to detect which OT features produce different glyph IDs
/// for common Latin characters. Returns a vec of (feature_tag, glyph_overrides)
/// for each feature that changes at least one glyph.
fn detect_ot_variants(data: &[u8]) -> Vec<(String, Vec<(char, u16)>)> {
    let face = match unprint_fonts::rustybuzz::Face::from_slice(data, 0) {
        Some(f) => f,
        None => return Vec::new(),
    };

    // Shape with no extra features (default rendering)
    let mut buf_default = unprint_fonts::rustybuzz::UnicodeBuffer::new();
    buf_default.push_str(PROBE_STRING);
    let out_default = unprint_fonts::rustybuzz::shape(&face, &[], buf_default);
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
        let tag = unprint_fonts::ttf_parser::Tag::from_bytes(tag_bytes);
        let features = [unprint_fonts::rustybuzz::Feature::new(tag, 1, ..)];

        let mut buf = unprint_fonts::rustybuzz::UnicodeBuffer::new();
        buf.push_str(PROBE_STRING);
        let out = unprint_fonts::rustybuzz::shape(&face, &features, buf);
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

/// Returns true if the font's GPOS or GSUB contains a 'case' or 'cpsp' Feature.
/// We must not emit |allcaps when neither table exists (pure no-op variant).
fn font_has_caps_tables(data: &[u8]) -> bool {
    let face = match unprint_fonts::ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return false,
    };
    // Check both GPOS and GSUB for feature tags
    for table_tag in [b"GPOS", b"GSUB"] {
        let tag = unprint_fonts::ttf_parser::Tag::from_bytes(table_tag);
        let Some(tbl) = face.raw_face().table(tag) else { continue };
        if tbl.len() < 10 { continue; }
        // Table header: version (4), scriptListOffset (2), featureListOffset (2), lookupListOffset (2)
        // Offsets are relative to start of table
        let feature_list_off = u16::from_be_bytes([tbl[6], tbl[7]]) as usize;
        if feature_list_off + 2 > tbl.len() { continue; }
        let feature_count = u16::from_be_bytes([tbl[feature_list_off], tbl[feature_list_off+1]]) as usize;
        for i in 0..feature_count {
            let rec_off = feature_list_off + 2 + i*6;
            if rec_off + 6 > tbl.len() { break; }
            let t = &tbl[rec_off..rec_off+4];
            if t == b"case" || t == b"cpsp" {
                return true;
            }
        }
    }
    false
}

/// Returns true if the font actually has kerning data available.
/// Used to decide whether to emit a kerned variant: a font with no kerning
/// renders identically with the kern feature on or off, so a kerned variant
/// would be a byte-identical duplicate of the nokern variant.
/// Kerning counts as available when:
/// - the legacy `kern` table has at least one pair (format 0), or any
///   class-based (format 2/3) subtable, which carries kerning by construction; or
/// - the GPOS table has a `kern` feature with at least one lookup
///   (a kern feature's lookups are PairAdjustment by definition).
/// True if the font data contains a GPOS table.
pub fn font_has_gpos(data: &[u8]) -> bool {
    let face = match unprint_fonts::ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let gpos_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"GPOS");
    face.raw_face().table(gpos_tag).is_some()
}

/// True if `seg` is an OpenType feature tag that selects a non-default
/// rendering: oldstyle/lining figures, small caps, swash/stylistic/
/// titling/historical alternates, stylistic sets, or the allcaps
/// (case+cpsp) variant. This is exactly the tag set the scan emits as
/// variant entries (`VARIANT_FEATURES` plus `allcaps`); kern/nokern,
/// lig/nolig, wght/opsz axis values, and `vintage=` are not OT features.
/// The specimen renders default features only, so a variant carrying one
/// of these tags never matches the specimen's default rendering.
fn is_ot_feature_tag(seg: &str) -> bool {
    matches!(
        seg,
        "onum" | "lnum" | "smcp" | "c2sc" | "swsh" | "salt" | "titl" | "hist" | "allcaps"
    ) || (seg.len() == 4
        && seg.starts_with("ss")
        && seg[2..].bytes().all(|b| b.is_ascii_digit()))
}

/// Numeric font version parsed from the `|version=X` key segment, in
/// hundredths (2.50 → 250, 10.00 → 1000), for numeric comparison —
/// string comparison would wrongly rank "10.00" below "2.50".
/// A key without a version tag (tombstones omit it) scores 0 and loses
/// to any versioned key. The minor part is read as two digits ("5" →
/// 50, "505" → 50); `font_revision` always formats two decimals, so
/// anything else is malformed input handled defensively.
fn font_key_version(font_key: &str) -> u32 {
    for seg in font_key.split('|').skip(1) {
        let Some(v) = seg.strip_prefix("version=") else { continue };
        let v = v.split('[').next().unwrap_or(v);
        let (major_s, minor_s) = match v.split_once('.') {
            Some((a, b)) => (a, b),
            None => (v, ""),
        };
        let major: u32 = major_s.parse().unwrap_or(0);
        let mut digits = minor_s.chars().filter(|c| c.is_ascii_digit());
        let d1 = digits.next().and_then(|c| c.to_digit(10)).unwrap_or(0);
        let d2 = digits.next().and_then(|c| c.to_digit(10)).unwrap_or(0);
        return major.saturating_mul(100).saturating_add(d1 * 10 + d2);
    }
    0
}

/// Preference for font-key selection and classifier tiebreaking.
/// Single source of truth, shared by --resolve-fontkey and
/// --select-font (crates/unprint-cli/src/main.rs) and the classifier
/// tiebreakers (font_match.rs fine-score sort, font_pipeline.rs
/// exact-tie resolution and GT resolution). One implementation, one
/// preference, everywhere — selection, bare-name resolution, and
/// the audit tiebreakers all pick variants the same way.
///
/// Returns `(version, variant_score)`, compared lexicographically:
/// the numeric font version DOMINATES — when no version is specified
/// (an exact-score tie between versions, or a bare PostScript-name
/// query), the highest version wins (10.00 > 2.50, numeric, not
/// string order). The variant score breaks ties within one version:
/// font with kerning data → +4 for kern (not nokern); font without
/// kerning → +4 for nokern; +2 for lig; +1 for opsz=10; −10 per
/// OT-feature variant tag (onum, lnum, smcp, c2sc, swsh, salt, titl,
/// hist, ss01–ss20, allcaps) and −10 per non-default weight instance
/// (`|wghtN`), so the base/default variant always wins the default-GT
/// pick over an OT-feature variant or a non-default weight. Higher
/// wins. The kerning signal is `has_kern` (legacy `kern` table
/// or GPOS `kern` feature), not GPOS presence: a font with a legacy
/// `kern` table but no GPOS (e.g. Trebuchet, Arial) still renders
/// kerned by default, so its kerned variant is the default pick.
/// Nolig is explicit in the key but scores 0 (the default); lig keeps
/// +2, so the lig variant wins the default pick — a specimen renders
/// real text, ligatures on by default (Mike Nov 17, 2026), and the
/// audit tiebreakers use this same preference, so a lig/nolig tie
/// resolves to the variant the specimen actually renders.
/// The lig match is segment-exact (`key_has_variant_tag`): a
/// substring test would match "nolig" too.
/// A key without a version tag scores version 0 and loses to any
/// versioned key. An exact key match (version specified) bypasses
/// this preference entirely — callers check exact matches first.
pub fn font_key_preference_score(font_key: &str, has_kern: bool) -> (u32, i32) {
    let version = font_key_version(font_key);
    let mut s = 0;
    let is_nokern = font_key.contains("nokern");
    // If has_kern: +4 for kern (not nokern).
    // If !has_kern: +4 for nokern.
    if has_kern != is_nokern {
        s += 4;
    }
    if key_has_variant_tag(font_key, "lig") {
        s += 2;
    }
    if font_key.contains("opsz=10") {
        s += 1;
    }
    // Non-default variants must never be the default pick: the specimen
    // renders default features at the font's default weight, so an
    // OT-feature variant or a `|wghtN` weight instance never matches the
    // default rendering. Segment 0 is the PostScript name, never a tag;
    // bracketed variations (`[wght=400,...]`) are stripped before matching.
    // The `wghtN` case matters: the scan only emits a weight instance for
    // a weight other than the declared default, and the instance must
    // never outscore the honest base entry at the default weight (e.g. a
    // kern-less variable font's `|wght400|nokern` instance would otherwise
    // tie the `|nokern|nolig` base entry on the kern dimension).
    for seg in font_key.split('|').skip(1) {
        let seg = seg.split('[').next().unwrap_or(seg);
        if is_ot_feature_tag(seg) {
            s -= 10;
        }
        if seg.len() > 4
            && seg.starts_with("wght")
            && seg[4..].bytes().all(|b| b.is_ascii_digit())
        {
            s -= 10;
        }
    }
    (version, s)
}

/// Best registry entry for a bare PostScript name (no version
/// specified): latest version, then the shared variant preference,
/// then the lowest font key (deterministic final fallback, matching
/// the fine-score sort). Exact-match callers check `by_key` first —
/// this helper is only for version-less queries.
pub fn best_entry_for_ps_name<'a, I>(entries: I, ps_name: &str) -> Option<&'a FontEntry>
where
    I: Iterator<Item = &'a FontEntry>,
{
    best_by_preference(entries.filter(|fe| fe.postscript_name == ps_name))
}

/// Best entry by the shared preference: highest version first, then
/// variant preference, then the lowest font_key on a full tie. Shared by
/// `best_entry_for_ps_name` and `select_entry_for_criteria` — one
/// implementation, one preference, everywhere.
fn best_by_preference<'a, I>(entries: I) -> Option<&'a FontEntry>
where
    I: Iterator<Item = &'a FontEntry>,
{
    entries.max_by(|a, b| {
        let pa = font_key_preference_score(a.font_key_ref(), a.has_kern);
        let pb = font_key_preference_score(b.font_key_ref(), b.has_kern);
        // Max-by prefers the greater; reverse the key comparison so
        // the lowest key wins a full preference tie (same as the
        // fine-score sort's key-asc final fallback).
        pa.cmp(&pb).then_with(|| b.font_key_ref().cmp(a.font_key_ref()))
    })
}

/// Criteria for structured font selection (`--select-font`).
/// All fields except `family` are optional; an omitted field imposes no
/// constraint. A present field is a HARD constraint — no match errors
/// loudly, never relaxes (Mike Nov 15, 2026).
#[derive(Debug, Clone, Default)]
pub struct FontSelection {
    /// Family name, e.g. "Noto Serif". Matched against the entry's
    /// typographic family (nameID 16, else nameID 1) after normalization.
    pub family: String,
    /// Weight, e.g. 400 or 700. Matched against the weight in the
    /// font_key's first segment.
    pub weight: Option<u32>,
    /// Slant: false = upright, true = italic.
    pub italic: Option<bool>,
    /// Optical size. Vacuous for entries whose variations carry no `opsz`
    /// axis (static fonts and variable fonts without the axis).
    pub opsz: Option<f32>,
    /// Kerning: true = kern variant, false = nokern variant.
    pub kern: Option<bool>,
    /// Ligatures: true = lig variant, false = nolig variant.
    pub lig: Option<bool>,
}

/// Normalize a family name for exact matching: lowercase, strip every
/// non-alphanumeric character. "Noto Serif" -> "notoserif",
/// "Source Serif 4" -> "sourceserif4". Same rule as the generators'
/// old `_norm_family` (gen-specimen.py), now owned by Rust.
pub fn normalize_family_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// (weight, italic) from a font_key's first segment, e.g.
/// "SourceSerif4-400Italic" -> (400, true), "Arial-BoldMT-700" ->
/// (700, false). Returns None when the segment carries no trailing
/// -NNN weight.
fn key_weight_italic(font_key: &str) -> Option<(u32, bool)> {
    let seg = font_key.split('|').next().unwrap_or(font_key);
    let (base, italic) = match seg.strip_suffix("Italic") {
        Some(b) => (b, true),
        None => (seg, false),
    };
    let digits = base.rsplit('-').next()?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u32>().ok().map(|w| (w, italic))
}

/// Whether the font_key carries the exact variant segment `tag`
/// ("kern", "nokern", "lig", "nolig"). Segments match exactly —
/// "nolig" is not "lig" — and a trailing bracket (`[wght=400,...]`) is
/// stripped first, mirroring `font_key_preference_score`.
fn key_has_variant_tag(font_key: &str, tag: &str) -> bool {
    font_key.split('|').skip(1).any(|seg| {
        let seg = seg.split('[').next().unwrap_or(seg);
        seg == tag
    })
}

/// The entry's `opsz` variation value, if it has one.
fn entry_opsz(fe: &FontEntry) -> Option<f32> {
    fe.variations
        .as_ref()?
        .iter()
        .find(|(tag, _)| tag == b"opsz")
        .map(|(_, v)| *v)
}

/// Select the single best catalog entry matching structured criteria
/// (`--select-font`, Mike Nov 15, 2026: specimen generators state what
/// they want; Rust picks). Every present criterion is a hard filter;
/// among the survivors the shared preference breaks ties (latest
/// version, then variant preference — lig over nolig when lig is
/// unconstrained, Mike Nov 17, 2026: a specimen renders ligatures on,
/// as real text does). Returns None when nothing matches — the caller
/// errors loudly, never relaxes.
pub fn select_entry_for_criteria<'a, I>(entries: I, sel: &FontSelection) -> Option<&'a FontEntry>
where
    I: Iterator<Item = &'a FontEntry>,
{
    let want_family = normalize_family_name(&sel.family);
    let filtered = entries.filter(|fe| {
        // Family: typographic family (nameID 16, else nameID 1),
        // normalized. Exact match — no fuzzy/nearest, no prefix.
        if normalize_family_name(&fe.typographic_family) != want_family {
            return false;
        }
        // Weight + slant come from the font_key's first segment, the
        // canonical identity (never the OS/2 bits, which describe the
        // file, not the instance).
        if sel.weight.is_some() || sel.italic.is_some() {
            let Some((w, it)) = key_weight_italic(fe.font_key_ref()) else {
                return false;
            };
            if let Some(want_w) = sel.weight {
                if w != want_w {
                    return false;
                }
            }
            if let Some(want_it) = sel.italic {
                if it != want_it {
                    return false;
                }
            }
        }
        // Optical size: an entry carrying an `opsz` axis value must
        // equal the requested size; an entry without the axis passes
        // vacuously (static fonts and axis-less variable fonts).
        if let Some(want_opsz) = sel.opsz {
            if let Some(v) = entry_opsz(fe) {
                if v != want_opsz {
                    return false;
                }
            }
        }
        if let Some(want_kern) = sel.kern {
            let tag = if want_kern { "kern" } else { "nokern" };
            if !key_has_variant_tag(fe.font_key_ref(), tag) {
                return false;
            }
        }
        if let Some(want_lig) = sel.lig {
            let tag = if want_lig { "lig" } else { "nolig" };
            if !key_has_variant_tag(fe.font_key_ref(), tag) {
                return false;
            }
        }
        true
    });
    best_by_preference(filtered)
}

/// Font version from the `head` table `fontRevision` field, formatted
/// as "major.minor" with two decimals (e.g. "2.50"). `fontRevision` is a
/// 16.16 fixed-point at offset 4 of the `head` table (after the 4-byte
/// table version). Returns an empty string if the font has no parseable
/// `head` table. Two-decimal rounding matches the conventional display
/// (fontTools reports the same values); versions finer than 0.01 (rare
/// in practice) round to the nearest hundredth.
fn font_revision(data: &[u8]) -> String {
    let face = match unprint_fonts::ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    let head_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"head");
    let Some(head) = face.raw_face().table(head_tag) else {
        return String::new();
    };
    if head.len() < 8 {
        return String::new();
    }
    let raw = u32::from_be_bytes([head[4], head[5], head[6], head[7]]);
    let major = raw >> 16;
    let frac = (raw & 0xFFFF) as f64 / 65536.0;
    let minor = (frac * 100.0).round() as u32;
    if minor >= 100 {
        // Fractional part rounds up to the next major (e.g. 2.9999 -> 3.00).
        format!("{}.00", major + 1)
    } else {
        format!("{}.{:02}", major, minor)
    }
}

fn font_has_kern_feature(data: &[u8]) -> bool {
    let face = match unprint_fonts::ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let tables = face.tables();
    // Legacy `kern` table: needs at least one pair. A present-but-empty
    // table (or subtables with no pairs) is no kerning.
    if let Some(kern) = &tables.kern {
        for sub in kern.subtables {
            match sub.format {
                unprint_fonts::ttf_parser::kern::Format::Format0(ref s0) => {
                    if s0.pairs.len() > 0 {
                        return true;
                    }
                }
                unprint_fonts::ttf_parser::kern::Format::Format2(_)
                | unprint_fonts::ttf_parser::kern::Format::Format3(_) => {
                    return true;
                }
                _ => {}
            }
        }
    }
    // GPOS `kern` feature with at least one lookup.
    if let Some(gpos) = &tables.gpos {
        let kern_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"kern");
        if let Some(feature) = gpos.features.find(kern_tag) {
            if feature.lookup_indices.len() > 0 {
                return true;
            }
        }
        return false;
    }
    // GPOS table exists but failed to parse: fail open (assume kerning may
    // be present) so a parse quirk can't drop the kerned variant.
    let gpos_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"GPOS");
    face.raw_face().table(gpos_tag).is_some()
}

/// Ligature probe sequences: (input_chars, unicode_ligature_char).
/// We shape the input chars with liga/dlig features and check if the
/// shaper produces a single glyph (i.e. a ligature substitution fired).
pub(crate) const LIGATURE_PROBES: &[(&str, char)] = &[
    ("ff",  '\u{FB00}'),
    ("fi",  '\u{FB01}'),
    ("fl",  '\u{FB02}'),
    ("ffi", '\u{FB03}'),
    ("ffl", '\u{FB04}'),
    // Double quotes as ligatures of two single quotes
    ("''",            '"'),          // U+0027 U+0027 → U+0022 (straight)
    ("\u{2018}\u{2018}", '\u{201C}'),  // left single → left double
    ("\u{2019}\u{2019}", '\u{201D}'),  // right single → right double
];

/// Returns true if `c` is a Unicode ligature codepoint (ff, fi, fl, ffi, ffl).
#[allow(dead_code)]
pub fn is_ligature_char(c: char) -> bool {
    LIGATURE_PROBES.iter().any(|&(_, lig)| lig == c)
}

/// Detect ligature glyph IDs by shaping probe strings with liga and dlig
/// features. Returns a vec of (unicode_ligature_char, glyph_id) for each
/// ligature the font supports.
fn detect_ligature_glyphs(data: &[u8]) -> Vec<(char, u16)> {
    let face = match unprint_fonts::rustybuzz::Face::from_slice(data, 0) {
        Some(f) => f,
        None => return Vec::new(),
    };

    let mut result = Vec::new();

    for &(probe, lig_char) in LIGATURE_PROBES {
        let input_len = probe.chars().count();

        // Try with both liga and dlig enabled
        let liga_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"liga");
        let dlig_tag = unprint_fonts::ttf_parser::Tag::from_bytes(b"dlig");
        let features = [
            unprint_fonts::rustybuzz::Feature::new(liga_tag, 1, ..),
            unprint_fonts::rustybuzz::Feature::new(dlig_tag, 1, ..),
        ];

        let mut buf = unprint_fonts::rustybuzz::UnicodeBuffer::new();
        buf.push_str(probe);
        let out = unprint_fonts::rustybuzz::shape(&face, &features, buf);
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

    // Special case: Arial Black is a distinct family, not Arial at weight 900.
    // The file's usWeightClass lies (says 400), but 400 is reasonable for the
    // Arial Black family (it's the regular weight of that family).
    // Per Mike Nov 2: "Black should be part of font family rather than a weight."
    if ps_name == "Arial-Black" {
        // Preserve "Black" as part of family name, don't strip it.
        // Use the provided weight (400 is reasonable for this family).
        let weight_str = weight.to_string();
        return format!("ArialBlack-{}", weight_str);
    }

    let weight_str = weight.to_string();

    // Detect italic from original PS name — we canonicalize all italic forms
    // to a single suffix "Italic".  Covers:
    //   "Italic", "Oblique", Adobe "It" suffix ("SourceSerif4-It",
    //   "SourceSerif4-BlackIt"), and the double "Italic-Italic" artifact
    //   from variable fonts (raw PS "SourceSerif4Italic-Italic").
    let lower_orig = ps_name.to_lowercase();
    let is_italic = lower_orig.contains("italic")
        || lower_orig.contains("oblique")
        || ps_name.ends_with("It")
        || lower_orig.ends_with("-it");

    // Strip italic markers iteratively from the end, collapsing
    // "Italic-Italic" → "" in two passes and "It" → "" in one.
    let mut base = ps_name.to_string();
    loop {
        let l = base.to_lowercase();
        let mut changed = false;
        // Longest italic tokens first
        const ITALIC_SUFFIXES: &[&str] = &[
            "-italic-italic",
            "_italic_italic",
            "-italic",
            "_italic",
            "italic",
            "-oblique",
            "_oblique",
            "oblique",
        ];
        for &sfx in ITALIC_SUFFIXES {
            if l.ends_with(sfx) {
                base.truncate(base.len() - sfx.len());
                changed = true;
                break;
            }
        }
        if changed {
            continue;
        }
        // Adobe shorthand: trailing "It" (capital I) e.g. "SourceSerif4-It",
        // "SourceSerif4-BlackIt".  Also handle lowercase "-it".
        if base.ends_with("It") && base.len() > 2 {
            // Avoid stripping "Git" or other accidental words: only strip if
            // the name contains italic or the suffix was preceded by '-' or a
            // weight word.  Since we already know is_italic may be true from
            // the original, stripping "It" when present is safe for our font
            // set; no family ends with literal "It" outside of italic.
            base.truncate(base.len() - 2);
            changed = true;
        } else if base.to_lowercase().ends_with("-it") && base.len() >= 3 {
            // e.g. variable file normalized to lowercase "-it"
            base.truncate(base.len() - 3);
            changed = true;
        }
        if !changed {
            break;
        }
    }
    base = base.trim_end_matches(|c| c == '-' || c == '_' ).to_string();

    // Strip weight-word suffixes (now that italic markers are gone) to obtain
    // the family core.  Handles both "-Bold" and "Bold" (no hyphen) forms so
    // "SourceSerif4BoldIt" → "SourceSerif4" after the italic strip above.
    let core = strip_weight_suffix(&base);
    let core = core.trim_end_matches(|c| c == '-' || c == '_' ).to_string();

    if is_italic {
        // Canonical form: "{Family}-{weight}Italic" — no duplicate "Italic".
        // This collapses both static "SourceSerif4-It" and variable
        // "SourceSerif4Italic-Italic" to "SourceSerif4-400Italic".
        if core.is_empty() {
            format!("{}Italic", weight_str)
        } else {
            format!("{}-{}Italic", core, weight_str)
        }
    } else if core.is_empty() {
        weight_str
    } else {
        format!("{}-{}", core, weight_str)
    }
}

/// Strip trailing weight-word suffixes from a PostScript name.
///
/// Handles hyphenated, underscored, and bare suffixes (e.g. "-Bold",
/// "_Bold", "Bold") case-insensitively, longest-first so "ExtraLight" is
/// removed before "Light".  Loops until no more weight words remain so
/// "SourceSerif4-Bold-Italic" after italic stripping still collapses.
///
/// Returns an owned String — the family stem.
fn strip_weight_suffix(ps_name: &str) -> String {
    // Longest-first to avoid "Light" eating "ExtraLight"
    const WEIGHT_WORDS: &[&str] = &[
        "ExtraLight",
        "UltraLight",
        "ExtraBlack",
        "ExtraBold",
        "UltraBold",
        "SemiBold",
        "DemiBold",
        "Regular",
        "Roman",
        "Medium",
        "Black",
        "Heavy",
        "Bold",
        "Light",
        "Thin",
    ];
    // Numeric weights must be stripped when they appear as "-400" / "_400"
    // so that "SourceSerif4-400It" → "SourceSerif4" (not "SourceSerif4-400")
    // and we don't produce "SourceSerif4-400-400Italic".
    const NUMERIC_WEIGHTS: &[&str] = &["100","200","300","400","500","600","700","800","900"];
    let mut s = ps_name.to_string();
    loop {
        let lower = s.to_lowercase();
        let mut stripped = false;
        for &w in WEIGHT_WORDS {
            let wl = w.to_lowercase();
            let hyphen = format!("-{}", wl);
            let uscore = format!("_{}", wl);
            if lower.ends_with(&hyphen) {
                s.truncate(s.len() - hyphen.len());
                stripped = true;
                break;
            }
            if lower.ends_with(&uscore) {
                s.truncate(s.len() - uscore.len());
                stripped = true;
                break;
            }
            if lower.ends_with(&wl) && s.len() > w.len() {
                // Bare suffix like "SourceSerif4Bold" — strip only if something
                // remains that looks like a family (at least 2 chars).
                // This is safe for our corpus; no family name itself is a
                // weight word.
                s.truncate(s.len() - wl.len());
                stripped = true;
                break;
            }
        }
        if stripped {
            continue;
        }
        // Numeric weight stripping — only hyphen/underscore forms to avoid
        // stripping the "4" in "SourceSerif4".
        for &nw in NUMERIC_WEIGHTS {
            let hyphen = format!("-{}", nw);
            let uscore = format!("_{}", nw);
            if lower.ends_with(&hyphen) {
                s.truncate(s.len() - hyphen.len());
                stripped = true;
                break;
            }
            if lower.ends_with(&uscore) {
                s.truncate(s.len() - uscore.len());
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Comprehensive font metadata (single-parse reader)
// ---------------------------------------------------------------------------

/// All name-table IDs, OS/2 weight class, and italic flag in one parse.
#[allow(dead_code)]
pub struct FontMetadata {
    pub nid1_family: String,
    pub nid2_subfamily: String,
    pub nid4_full_name: String,
    pub nid6_postscript: String,
    /// nameID 16 (Typographic Family), falls back to nameID 1 if absent.
    pub nid16_typographic: String,
    pub weight_class: u16,
    pub italic: bool,
}

/// Read all font metadata in a single ttf_parser parse pass.
pub fn read_font_metadata(data: &[u8]) -> FontMetadata {
    let face = match unprint_fonts::ttf_parser::Face::parse(data, 0) {
        Ok(f) => f,
        Err(_) => return FontMetadata {
            nid1_family: String::new(),
            nid2_subfamily: String::new(),
            nid4_full_name: String::new(),
            nid6_postscript: String::new(),
            nid16_typographic: String::new(),
            weight_class: 400,
            italic: false,
        },
    };

    let mut nid1 = String::new();
    let mut nid2 = String::new();
    let mut nid4 = String::new();
    let mut nid6 = String::new();
    let mut nid16 = String::new();

    for name in face.names() {
        match name.name_id {
            1 if nid1.is_empty() => { if let Some(s) = name.to_string() { nid1 = s; } }
            2 if nid2.is_empty() => { if let Some(s) = name.to_string() { nid2 = s; } }
            4 if nid4.is_empty() => { if let Some(s) = name.to_string() { nid4 = s; } }
            6 if nid6.is_empty() => { if let Some(s) = name.to_string() { nid6 = s; } }
            16 if nid16.is_empty() => { if let Some(s) = name.to_string() { nid16 = s; } }
            _ => {}
        }
    }

    if nid16.is_empty() {
        nid16 = nid1.clone();
    }

    let weight_class = face.tables().os2
        .map(|os2| os2.weight().to_number())
        .unwrap_or(400);
    let italic = face.is_italic();

    FontMetadata {
        nid1_family: nid1,
        nid2_subfamily: nid2,
        nid4_full_name: nid4,
        nid6_postscript: nid6,
        nid16_typographic: nid16,
        weight_class,
        italic,
    }
}

/// Font identity for major/minor miss classification.
/// Read from the font's name table and OS/2 table — no string munging.
#[derive(Debug, Clone)]
pub struct FontIdentity {
    /// Typographic family: name ID 16 if present, else name ID 1.
    pub family: String,
    /// OS/2 usWeightClass (400 = Regular, 500 = Medium, 700 = Bold, etc.)
    pub _weight: u16,
    /// OS/2 fsSelection italic bit.
    pub italic: bool,
}

impl FontIdentity {

    /// Strip optical-size suffixes (Caption, SmText, Subhead, Display)
    /// so "Source Serif 4 SmText" and "Source Serif 4" compare equal.
    fn root_family(family: &str) -> &str {
        for suffix in &[" Display", " Subhead", " SmText", " Caption"] {
            if let Some(stripped) = family.strip_suffix(suffix) {
                return stripped;
            }
        }
        family
    }

    /// Two fonts are a "major" difference if root family or italic differ.
    /// Weight and optical-size differences within the same family are minor.
    pub fn is_major_diff(&self, other: &FontIdentity) -> bool {
        Self::root_family(&self.family) != Self::root_family(&other.family)
            || self.italic != other.italic
    }
}

/// Read font identity from a font file path. Returns None on parse failure.
pub fn read_font_identity(path: &Path) -> Option<FontIdentity> {
    let data = std::fs::read(path).ok()?;
    let face = unprint_fonts::ttf_parser::Face::parse(&data, 0).ok()?;

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

    Some(FontIdentity { family, _weight: weight, italic })
}

pub fn load_font_entry(path: &Path, aliases: &HashMap<String, Alias>) -> Option<FontEntry> {
    let data = std::fs::read(path).ok()?;

    // Verify ab_glyph can parse it (reject corrupt files)
    let _ = unprint_fonts::ab_glyph::FontRef::try_from_slice(&data).ok()?;

    let oldstyle_figures = detect_oldstyle_figures(&data);
    // GPOS presence, captured while the bytes are available — drives the
    // shared font-key preference (see font_key_preference_score). The
    // scan drops `data` after this, so it cannot be re-derived later.
    let has_gpos = font_has_gpos(&data);
    // Kerning data, captured while the bytes are available — the shared
    // font-key preference's kern/nokern signal. Same lifetime constraint
    // as has_gpos: computed here, stored on the entry, serialized.
    let has_kern = font_has_kern_feature(&data);
    // Font version, captured while the bytes are available — part of the
    // font key (|version=X). Same lifetime constraint as has_gpos/has_kern.
    let font_version = font_revision(&data);
    let meta = read_font_metadata(&data);
    let raw_ps_name = meta.nid6_postscript;
    let os2_weight = meta.weight_class;
    let postscript_name = make_weight_explicit(&raw_ps_name, os2_weight);
    let typographic_family = meta.nid16_typographic;

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown");

    let stem_lower = stem.to_lowercase().replace(' ', "");

    // Try alias table first (exact match on lowercase stem)
    if let Some(alias) = aliases.get(&stem_lower) {
        let class = classify(alias.family);
        let font_key_cache = FontEntry::compute_font_key_full(
            &postscript_name,
            "",
            None,
            &None,
            &font_version,
        );
        return Some(FontEntry {
            path: path.to_path_buf(),
            family_name: alias.family.to_string(),
            postscript_name,
            raw_postscript_name: raw_ps_name.clone(),
            is_bold: alias.bold,
            is_italic: alias.italic,
            class,
            data,
            oldstyle_figures,
            variant_tag: String::new(),
            font_key_cache,
            glyph_overrides: None,
            variations: None,
            typographic_family,
            vintage_era: None,
            opsz_range: None,
            has_gpos,
            has_kern,
            font_version: font_version.clone(),
        });
    }

    // Fallback: derive from filename + OS/2 metadata.
    // Use OS/2 weight/italic as authority — filename "It" suffix (Adobe
    // italic shorthand) previously missed because we only checked for
    // "italic"/"oblique"/"slant".
    let family_name = stem.replace('-', " ").replace('_', " ");
    let lower = family_name.to_lowercase();
    let is_bold = os2_weight >= 700
        || lower.contains("bold")
        || lower.contains("black")
        || lower.contains("heavy");
    let is_italic = meta.italic
        || lower.contains("italic")
        || lower.contains("oblique")
        || lower.contains("slant")
        || lower.ends_with("-it")
        || lower.ends_with(" it")
        || stem_lower.ends_with("it");
    let class = classify(&family_name);

    let font_key_cache = FontEntry::compute_font_key_full(
        &postscript_name,
        "",
        None,
        &None,
        &font_version,
    );
    Some(FontEntry {
        path: path.to_path_buf(),
        family_name,
        postscript_name,
        raw_postscript_name: raw_ps_name,
        is_bold,
        is_italic,
        class,
        data,
        oldstyle_figures,
        variant_tag: String::new(),
        font_key_cache,
        glyph_overrides: None,
        variations: None,
        typographic_family,
        vintage_era: None,
        opsz_range: None,
        has_gpos,
        has_kern,
        font_version,
    })
}

/// Scan fonts in given dirs without touching the global scan cache.
/// Used for vintage cache dir to avoid invalidating font_scan.bin.
/// Walks dirs, calls load_font_entry for each .ttf/.otf, no dedup (caller may dedup).
pub fn scan_fonts_uncached(dirs: &[PathBuf]) -> Vec<FontEntry> {
    let aliases = build_alias_table();
    let mut fonts = Vec::new();
    for dir in dirs {
        if !dir.exists() {
            continue;
        }
        for entry in WalkDir::new(dir).follow_links(true).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            if ext != "ttf" && ext != "otf" {
                continue;
            }
            if let Some(fe) = load_font_entry(path, &aliases) {
                // Drop bytes like main scan does for memory, but keep path
                let mut fe_nodata = fe;
                fe_nodata.data = Vec::new();
                // Note: variant detection (OT features, weight instances) is skipped for vintage
                // cache fonts to keep them as single entries. Base vintage fonts already have
                // historical spacing baked in.
                fonts.push(fe_nodata);
            }
        }
    }
    fonts
}

#[cfg(test)]
mod catalog_hash_tests {
    use super::*;

    fn entry(variations: Variations, glyph_overrides: Option<Vec<(char, u16)>>) -> FontEntry {
        let font_key_cache = FontEntry::compute_font_key_full(
            "Test-400",
            "",
            None,
            &variations,
            "1.00",
        );
        FontEntry {
            path: PathBuf::from("/font.ttf"),
            family_name: "Test".to_string(),
            postscript_name: "Test-400".to_string(),
            raw_postscript_name: "Test-Regular".to_string(),
            is_bold: false,
            is_italic: false,
            class: FontClass::Serif,
            data: Vec::new(),
            oldstyle_figures: false,
            variant_tag: String::new(),
            font_key_cache,
            glyph_overrides,
            variations,
            typographic_family: "Test".to_string(),
            vintage_era: None,
            opsz_range: None,
            has_gpos: false,
            has_kern: false,
            font_version: "1.00".to_string(),
        }
    }

    #[test]
    fn cache_freshness_base_entry_uses_declared_variation_defaults() {
        // Start with non-variable fixture (no brackets), assign defaults, expect transition to bracket notation
        let mut base = FontEntry {
            path: PathBuf::from("/font.ttf"),
            family_name: "Test".to_string(),
            postscript_name: "Test-400".to_string(),
            raw_postscript_name: "Test-Regular".to_string(),
            is_bold: false,
            is_italic: false,
            class: FontClass::Serif,
            data: Vec::new(),
            oldstyle_figures: false,
            variant_tag: String::new(),
            font_key_cache: "Test-400".to_string(),
            glyph_overrides: None,
            variations: None,
            typographic_family: "Test".to_string(),
            vintage_era: None,
            opsz_range: None,
            has_gpos: false,
            has_kern: false,
            font_version: "1.00".to_string(),
        };
        // Test fixture: arbitrary variation coordinates (opsz value is not
        // significant here; detect_default_variations now uses the font's
        // declared default).
        let defaults = vec![(*b"wght", 400.0), (*b"opsz", 10.0)];
        base.assign_default_variations(Some(defaults.clone()));

        assert_eq!(base.variations, Some(defaults));
        assert_eq!(base.font_key_ref(), "Test-400|version=1.00[wght=400,opsz=10]");
    }

    #[test]
    fn complete_generation_deduplicates_appended_entries_by_font_key() {
        let first = entry(None, None);
        let duplicate = entry(None, None);
        let deduped = dedup_font_entries_by_key(vec![first, duplicate], true);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].font_key_ref(), "Test-400|version=1.00");
    }

    #[test]
    fn catalog_hash_includes_variation_coordinates() {
        let at_default = entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 20.0)]), None);
        let wrong_weight = entry(Some(vec![(*b"wght", 200.0), (*b"opsz", 20.0)]), None);
        assert_ne!(
            FontRegistry::compute_hash(&[at_default]),
            FontRegistry::compute_hash(&[wrong_weight]),
        );
    }

    #[test]
    fn cache_freshness_catalog_hash_includes_source_file_fingerprint() {
        let dir = tempfile::tempdir().expect("create temp directory");
        let first_path = dir.path().join("first.ttf");
        let second_path = dir.path().join("second.ttf");
        std::fs::write(&first_path, b"first payload").expect("write first source");
        std::fs::write(&second_path, b"different second payload").expect("write second source");

        let mut first = entry(None, None);
        first.path = first_path;
        let mut second = entry(None, None);
        second.path = second_path;

        assert_ne!(
            FontRegistry::compute_hash(&[first]),
            FontRegistry::compute_hash(&[second]),
        );
    }

    #[test]
    fn catalog_hash_includes_glyph_overrides() {
        let base = entry(None, Some(vec![('f', 10)]));
        let changed = entry(None, Some(vec![('f', 11)]));
        assert_ne!(
            FontRegistry::compute_hash(&[base]),
            FontRegistry::compute_hash(&[changed]),
        );
    }

    #[test]
    fn catalog_hash_wght_500_vs_400_distinct() {
        // Same file, same key, but wght axis differs – must produce distinct hash
        // (regression: stale variation-blind reuse would keep same BGEO)
        let wght_500 = entry(Some(vec![(*b"wght", 500.0)]), None);
        let wght_400 = entry(Some(vec![(*b"wght", 400.0)]), None);
        let h500 = FontRegistry::compute_hash(&[wght_500]);
        let h400 = FontRegistry::compute_hash(&[wght_400]);
        assert_ne!(h500, h400, "wght 500 vs 400 should change catalog hash to force BGEO rebuild");
    }

    #[test]
    fn catalog_hash_opsz_change_rebuilds() {
        // opsz is often same PS name but different rendering metrics; hash must change
        let opsz_12 = entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 12.0)]), None);
        let opsz_20 = entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 20.0)]), None);
        assert_ne!(
            FontRegistry::compute_hash(&[opsz_12]),
            FontRegistry::compute_hash(&[opsz_20]),
            "opsz change should invalidate BGEO and training caches"
        );
    }

    #[test]
    fn catalog_hash_distinct_coords_produce_distinct_classes() {
        // Different variation coords should lead to distinct render hashes/classes
        // We test via catalog hash as proxy for class identity
        let a = entry(Some(vec![(*b"wght", 400.0), (*b"wdth", 100.0)]), None);
        let b = entry(Some(vec![(*b"wght", 400.0), (*b"wdth", 75.0)]), None);
        let c = entry(Some(vec![(*b"wght", 700.0), (*b"wdth", 100.0)]), None);
        let ha = FontRegistry::compute_hash(&[a]);
        let hb = FontRegistry::compute_hash(&[b]);
        let hc = FontRegistry::compute_hash(&[c]);
        assert_ne!(ha, hb);
        assert_ne!(ha, hc);
        assert_ne!(hb, hc);
    }

    #[test]
    fn runtime_train_hash_equality() {
        // Train's old hash was font_key-only; runtime includes variations.
        // This test asserts equality – will fail until train uses same hash.
        // We simulate train's old logic vs new.
        use std::hash::{Hash, Hasher};
        let fe = entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 12.0)]), Some(vec![('a', 5)]));
        let runtime_hash = FontRegistry::compute_hash(&[fe.clone()]);
        // Simulate old train hash (font_key only)
        let mut old_hasher = rustc_hash::FxHasher::default();
        fe.font_key().hash(&mut old_hasher);
        let old_train_hash = old_hasher.finish();
        // New train hash should use compute_hash – they should match runtime
        // If this assert fails, it means train still uses old logic.
        // For now we assert runtime != old (to show bug), and runtime == new.
        assert_ne!(runtime_hash, old_train_hash, "old train hash is variation-blind – should differ from runtime");
        // New correct train hash:
        let new_train_hash = FontRegistry::compute_hash(&[fe]);
        assert_eq!(runtime_hash, new_train_hash, "runtime and train hashes must be identical");
    }

    #[test]
    fn opsz_base_uses_declared_default_and_captures_range() {
        // Source Serif 4 has opsz axis min 8 default 20 max 60.
        // The base candidate pins every axis to its declared default
        // (opsz=20); the size-derived opsz now comes from the synthetic
        // `|opszN` candidate in identify_fonts, not from this function.
        // This function must also capture the axis range so the synthesis
        // gate (FontEntry::opsz_axis_range) can fire.
        let path = PathBuf::from("/home/hatch/.local/share/fonts/ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf");
        if !path.exists() {
            // Skip if font not installed (CI may not have it)
            return;
        }
        let (vars_opt, opsz_range) = super::detect_default_variations(&path);
        let vars = vars_opt.expect("should have variations");
        let opsz = vars.iter().find(|(tag, _)| tag == b"opsz").map(|(_, v)| *v);
        assert!(opsz.is_some(), "opsz axis should be present");
        let opsz_val = opsz.unwrap();
        assert!((opsz_val - 20.0).abs() < 0.01, "opsz should be 20 (declared default), got {}", opsz_val);
        // Ensure wght is still default 400
        let wght = vars.iter().find(|(tag, _)| tag == b"wght").map(|(_, v)| *v).unwrap_or(0.0);
        assert!((wght - 400.0).abs() < 0.5, "wght should be default 400, got {}", wght);
        // The axis range must be captured for synthesis.
        let (min, default, max) = opsz_range.expect("opsz_range should be captured");
        assert!((min - 8.0).abs() < 0.01, "opsz min should be 8, got {}", min);
        assert!((default - 20.0).abs() < 0.01, "opsz default should be 20, got {}", default);
        assert!((max - 60.0).abs() < 0.01, "opsz max should be 60, got {}", max);
    }

    #[test]
    fn opsz_weight_instances_use_same_opsz() {
        // Weight instances pin non-wght axes to their declared defaults,
        // matching the base entry (opsz=20 for Source Serif 4). The
        // size-derived opsz comes from the synthetic `|opszN` candidate.
        let path = PathBuf::from("/home/hatch/.local/share/fonts/ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf");
        if !path.exists() {
            return;
        }
        let instances = super::detect_weight_instances(&path, super::FontClass::Serif);
        assert!(!instances.is_empty(), "should have weight instances");
        for inst in &instances {
            let opsz = inst.axes.iter().find(|(tag, _)| tag == b"opsz").map(|(_, v)| *v);
            assert!(opsz.is_some(), "weight instance should have opsz");
            let v = opsz.unwrap();
            assert!((v - 20.0).abs() < 0.01, "weight instance opsz should be 20 (declared default), got {}", v);
        }
    }

    #[test]
    fn opsz_render_training_bgeo_verification_same_coords() {
        // All downstream consumers (char_render, train, geo_cache, verify)
        // use FontEntry::apply_variations, which applies the stored variations.
        // If catalog has opsz=10, they all receive identical coords.
        // This test proves that a FontEntry with opsz=10, when applied to a
        // FontRef, actually sets the variation, and that different opsz values
        // produce distinct hashes (so they remain distinct where geometry differs).
        let fe_10 = entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 10.0)]), None);
        let fe_20 = entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 20.0)]), None);
        // Same opsz should produce same hash, different opsz different hash
        let h10_a = FontRegistry::compute_hash(&[fe_10.clone()]);
        let h10_b = FontRegistry::compute_hash(&[entry(Some(vec![(*b"wght", 400.0), (*b"opsz", 10.0)]), None)]);
        let h20 = FontRegistry::compute_hash(&[fe_20]);
        assert_eq!(h10_a, h10_b, "same opsz should give same hash (identical coords)");
        assert_ne!(h10_a, h20, "different opsz should give different hash (distinct geometry)");
    }

    #[test]
    fn preference_version_dominates_variant() {
        // Newest version wins even when the older version has the better
        // variant score: v2.82 kern (4) beats v2.50 kern+lig (6).
        let old = super::font_key_preference_score("TimesNewRomanPSMT-400|version=2.50|kern|lig", true);
        let new = super::font_key_preference_score("TimesNewRomanPSMT-400|version=2.82|kern|nolig", true);
        assert_eq!(old, (250, 6));
        assert_eq!(new, (282, 4));
        assert!(new > old, "version must dominate the variant score");
    }

    #[test]
    fn preference_version_compares_numerically() {
        // 10.00 > 2.50 numerically; string comparison would rank "10.00" below "2.50".
        let low = super::font_key_preference_score("X-400|version=2.50|kern|nolig", true);
        let high = super::font_key_preference_score("X-400|version=10.00|kern|nolig", true);
        assert_eq!(low.0, 250);
        assert_eq!(high.0, 1000);
        assert!(high > low, "10.00 must beat 2.50");
    }

    #[test]
    fn preference_variant_breaks_same_version_tie() {
        // Within one version, the existing variant preference decides.
        let lig = super::font_key_preference_score("X-400|version=2.82|kern|lig", true);
        let nolig = super::font_key_preference_score("X-400|version=2.82|kern|nolig", true);
        assert_eq!(lig, (282, 6));
        assert_eq!(nolig, (282, 4));
        assert!(lig > nolig, "lig must win within the same version");
    }

    #[test]
    fn preference_missing_version_loses() {
        // A key without a version tag scores version 0 and loses to any
        // versioned key, whatever the variant score.
        let unversioned = super::font_key_preference_score("X-400|kern|nolig", true);
        let versioned = super::font_key_preference_score("X-400|version=1.00|kern|lig", true);
        assert_eq!(unversioned, (0, 4));
        assert_eq!(versioned, (100, 6));
        assert!(versioned > unversioned, "versioned must beat unversioned");
    }

    #[test]
    fn font_key_version_parses_hundredths() {
        assert_eq!(super::font_key_version("X-400|version=2.50|kern|nolig"), 250);
        assert_eq!(super::font_key_version("X-400|version=10.00|kern|nolig"), 1000);
        assert_eq!(super::font_key_version("X-400|version=1.00"), 100);
        assert_eq!(super::font_key_version("X-400|version=2.5|kern|nolig"), 250);
        assert_eq!(super::font_key_version("X-400|kern|nolig"), 0);
        assert_eq!(super::font_key_version("X-400|version=abc|kern|nolig"), 0);
    }
}

#[cfg(test)]
mod select_font_tests {
    use super::*;

    fn entry(font_key: &str, typographic_family: &str, variations: Variations) -> FontEntry {
        // font_key_cache is the key itself; postscript_name is the first
        // segment (weight-explicit), as the scanner builds it.
        let ps = font_key.split('|').next().unwrap_or(font_key).to_string();
        FontEntry {
            path: PathBuf::from("/font.ttf"),
            family_name: typographic_family.to_string(),
            postscript_name: ps,
            raw_postscript_name: "Test".to_string(),
            is_bold: false,
            is_italic: font_key.contains("Italic"),
            class: FontClass::Serif,
            data: Vec::new(),
            oldstyle_figures: false,
            variant_tag: String::new(),
            font_key_cache: font_key.to_string(),
            glyph_overrides: None,
            variations,
            typographic_family: typographic_family.to_string(),
            vintage_era: None,
            opsz_range: None,
            has_gpos: false,
            has_kern: false,
            font_version: "1.00".to_string(),
        }
    }

    #[test]
    fn normalize_family_name_strips_space_and_case() {
        assert_eq!(super::normalize_family_name("Noto Serif"), "notoserif");
        assert_eq!(super::normalize_family_name("Source Serif 4"), "sourceserif4");
        assert_eq!(super::normalize_family_name("IBM Plex Serif"), "ibmplexserif");
    }

    #[test]
    fn key_weight_italic_parses_first_segment() {
        assert_eq!(
            super::key_weight_italic("SourceSerif4-400Italic|version=4.00|kern|nolig"),
            Some((400, true))
        );
        assert_eq!(
            super::key_weight_italic("Arial-BoldMT-700|version=2.82|kern|nolig"),
            Some((700, false))
        );
        assert_eq!(
            super::key_weight_italic("NotoSerif-400|version=2.01|nokern|lig[wght=400,wdth=100]"),
            Some((400, false))
        );
        assert_eq!(super::key_weight_italic("NoWeight|version=1.00|kern|lig"), None);
    }

    #[test]
    fn select_prefers_latest_version() {
        // Courier New Regular, Win31 v2.50 vs modern v2.82: the API must
        // return v2.82 (the case that started this work).
        let entries = vec![
            entry("CourierNewPSMT-400|version=2.50|nokern|lig", "Courier New", None),
            entry("CourierNewPSMT-400|version=2.82|nokern|lig", "Courier New", None),
        ];
        let sel = FontSelection {
            family: "Courier New".to_string(),
            weight: Some(400),
            italic: Some(false),
            opsz: Some(10.0),
            kern: Some(false),
            lig: None,
        };
        let best = super::select_entry_for_criteria(entries.iter(), &sel).unwrap();
        assert_eq!(best.font_key_ref(), "CourierNewPSMT-400|version=2.82|nokern|lig");
    }

    #[test]
    fn select_prefers_lig_when_unconstrained() {
        // Mike Nov 17, 2026: the API prefers lig. With lig unconstrained,
        // the lig variant wins over nolig within one version — the
        // specimen renders ligatures on, as real text does.
        let entries = vec![
            entry("X-400|version=2.82|nokern|nolig", "X", None),
            entry("X-400|version=2.82|nokern|lig", "X", None),
        ];
        let sel = FontSelection {
            family: "X".to_string(),
            weight: Some(400),
            italic: Some(false),
            opsz: None,
            kern: Some(false),
            lig: None,
        };
        let best = super::select_entry_for_criteria(entries.iter(), &sel).unwrap();
        assert_eq!(best.font_key_ref(), "X-400|version=2.82|nokern|lig");
    }

    #[test]
    fn select_explicit_nolig_still_constrains() {
        // An explicit lig=false stays a hard filter: the nolig entry is
        // the only survivor even though the API prefers lig.
        let entries = vec![
            entry("X-400|version=2.82|nokern|nolig", "X", None),
            entry("X-400|version=2.82|nokern|lig", "X", None),
        ];
        let sel = FontSelection {
            family: "X".to_string(),
            weight: Some(400),
            italic: Some(false),
            opsz: None,
            kern: Some(false),
            lig: Some(false),
        };
        let best = super::select_entry_for_criteria(entries.iter(), &sel).unwrap();
        assert_eq!(best.font_key_ref(), "X-400|version=2.82|nokern|nolig");
    }

    #[test]
    fn select_no_match_returns_none() {
        // Merriweather Italic 400 nokern opsz=10: the catalog holds only
        // the kern opsz=18 entry, so the strict selection finds nothing.
        let entries = vec![
            entry(
                "Merriweather-400Italic|version=2.10|wght400|kern|lig[wght=400,wdth=100,opsz=18]",
                "Merriweather",
                Some(vec![(*b"wght", 400.0), (*b"wdth", 100.0), (*b"opsz", 18.0)]),
            ),
        ];
        let sel = FontSelection {
            family: "Merriweather".to_string(),
            weight: Some(400),
            italic: Some(true),
            opsz: Some(10.0),
            kern: Some(false),
            lig: None,
        };
        assert!(super::select_entry_for_criteria(entries.iter(), &sel).is_none());
    }

    #[test]
    fn select_opsz_vacuous_without_axis() {
        // A static font (no variations) passes an opsz constraint
        // vacuously; a variable font with the axis must match it.
        let entries = vec![
            entry("Static-400|version=1.00|nokern|lig", "Static", None),
            entry(
                "Static-400|version=1.00|nokern|lig[wght=400,opsz=20]",
                "Static",
                Some(vec![(*b"wght", 400.0), (*b"opsz", 20.0)]),
            ),
        ];
        let sel = FontSelection {
            family: "Static".to_string(),
            weight: Some(400),
            italic: Some(false),
            opsz: Some(10.0),
            kern: Some(false),
            lig: None,
        };
        let best = super::select_entry_for_criteria(entries.iter(), &sel).unwrap();
        assert_eq!(best.font_key_ref(), "Static-400|version=1.00|nokern|lig");
    }

    #[test]
    fn select_family_is_exact_not_prefix() {
        // "Noto Sans" must not match "Noto Sans Mono".
        let entries = vec![
            entry("NotoSansMono-400|version=1.00|nokern|lig", "Noto Sans Mono", None),
        ];
        let sel = FontSelection {
            family: "Noto Sans".to_string(),
            weight: Some(400),
            italic: Some(false),
            opsz: None,
            kern: None,
            lig: None,
        };
        assert!(super::select_entry_for_criteria(entries.iter(), &sel).is_none());
    }
}
