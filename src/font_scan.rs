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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use walkdir::WalkDir;

/// Cache of vertical-gap counts per (font path, ligature char) — avoids
/// re-rasterizing the same FB00-FB04 glyph for every candidate per line.
/// 52% of CPU was here in nightly pprof.
static GAP_CACHE: OnceLock<RwLock<HashMap<(PathBuf, char), usize>>> = OnceLock::new();

fn gap_cache() -> &'static RwLock<HashMap<(PathBuf, char), usize>> {
    GAP_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
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
#[inline]
#[allow(dead_code)]
fn lig_contains(mask: LigMask, c: char) -> bool {
    if let Some(b) = lig_bit(c) {
        (mask & (1u8 << b)) != 0
    } else { false }
}

/// Cache of final collapsed lig sets per font path — eliminates 33.9%
/// inclusive hot path (font_match.rs:138, font_pipeline.rs:381) which
/// otherwise did fs::read + cmap + 5× shape per candidate per line.
static LIG_SET_CACHE: OnceLock<RwLock<HashMap<PathBuf, LigMask>>> = OnceLock::new();

fn lig_set_cache() -> &'static RwLock<HashMap<PathBuf, LigMask>> {
    LIG_SET_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

// Thread-local ShapePlan cache for lig probe — mirrors
// crates/unprint-fonts/src/shape.rs which caches 8.3% leaf
// find_language_feature. Keyed by (font_data_ptr, features_hash).
type PlanKey = (usize, u64);
thread_local! {
    static PROBE_PLAN_CACHE: RefCell<HashMap<PlanKey, Arc<unprint_fonts::rustybuzz::ShapePlan>>> =
        RefCell::new(HashMap::new());
}

#[inline]
fn probe_plan_hash(features: &[unprint_fonts::rustybuzz::Feature]) -> u64 {
    let mut h = DefaultHasher::new();
    // Dir/Script/Lang are constant for Latin LTR probes (Directional guess
    // always LTR, Latn), so we only hash features; if guess ever diverges
    // the miss just rebuilds — still correct because shape_with_plan debug
    // asserts dir/script match.
    for f in features {
        f.tag.hash(&mut h);
        f.value.hash(&mut h);
        f.start.hash(&mut h);
        f.end.hash(&mut h);
    }
    h.finish()
}

#[inline]
fn probe_plan_cached(
    face: &unprint_fonts::rustybuzz::Face,
    data_ptr: usize,
    features: &[unprint_fonts::rustybuzz::Feature],
) -> Arc<unprint_fonts::rustybuzz::ShapePlan> {
    let hash = probe_plan_hash(features);
    let key = (data_ptr, hash);
    if let Some(hit) = PROBE_PLAN_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit;
    }
    // We guess inside shape_probe; but plan creation needs dir/script/lang.
    // All our probes are Latin LTR, no language — matches what UnicodeBuffer
    // guess returns for ff/fi/fl etc. Using explicit None avoids needing
    // buffer to build plan; shape_with_plan will assert dir/script match and
    // they do match because guess → LTR Latn.
    let plan = Arc::new(unprint_fonts::rustybuzz::ShapePlan::new(
        face,
        unprint_fonts::rustybuzz::Direction::LeftToRight,
        Some(unprint_fonts::rustybuzz::script::LATIN),
        None,
        features,
    ));
    PROBE_PLAN_CACHE.with(|c| {
        c.borrow_mut().insert(key, plan.clone());
    });
    plan
}

#[inline]
fn shape_probes_collapsed(face: &unprint_fonts::rustybuzz::Face, data_ptr: usize, features: &[unprint_fonts::rustybuzz::Feature], probe: &str) -> bool {
    let mut buf = unprint_fonts::rustybuzz::UnicodeBuffer::new();
    buf.push_str(probe);
    buf.guess_segment_properties();
    // If guess diverges from our cached plan's dir/script (unlikely for ASCII
    // probes) fall back to fresh shape — correctness over cache hit.
    let dir = buf.direction();
    let script = buf.script();
    let latin_ltr = dir == unprint_fonts::rustybuzz::Direction::LeftToRight
        && (script == unprint_fonts::rustybuzz::script::LATIN || script == unprint_fonts::rustybuzz::script::UNKNOWN);
    if !latin_ltr {
        let out = unprint_fonts::rustybuzz::shape(face, features, buf);
        return out.glyph_infos().len() == 1 && out.glyph_infos()[0].glyph_id != 0;
    }
    let plan = probe_plan_cached(face, data_ptr, features);
    let out = unprint_fonts::rustybuzz::shape_with_plan(face, &plan, buf);
    out.glyph_infos().len() == 1 && out.glyph_infos()[0].glyph_id != 0
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
}

impl FontEntry {
    #[inline]
    pub fn compute_font_key(postscript_name: &str, variant_tag: &str) -> String {
        if variant_tag.is_empty() {
            postscript_name.to_owned()
        } else {
            format!("{}|{}", postscript_name, variant_tag)
        }
    }

    #[inline]
    pub fn compute_font_key_full(
        postscript_name: &str,
        variant_tag: &str,
        vintage_era: Option<&str>,
        has_var: bool,
    ) -> String {
        let mut k = if variant_tag.is_empty() {
            postscript_name.to_owned()
        } else {
            format!("{}|{}", postscript_name, variant_tag)
        };
        if let Some(era) = vintage_era {
            k.push_str("|vintage=");
            k.push_str(era);
        }
        if has_var {
            k.push_str("|var");
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
        let has_var = self.variations.is_some();
        self.font_key_cache = Self::compute_font_key_full(
            &self.postscript_name,
            &self.variant_tag,
            self.vintage_era.as_deref(),
            has_var,
        );
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
    /// checking actual glyph ink for a clear vertical whitespace (no white
    /// padding, 100% top-to-bottom).  Two-char ligs (ff, fi, fl) need ≥1 such
    /// interior gutter to be excluded; three-char ligs (ffi, ffl) need ≥2.
    /// Only FB00–FB04 are considered; quote ligature probes are excluded.
    pub fn collapsed_lig_set(&self) -> LigMask {
        // Global memo — eliminates 33.9% inclusive hot path after first line.
        // Final filtered set is cached, so no FS read / shaping on hits.
        {
            let guard = lig_set_cache().read().unwrap();
            if let Some(cached) = guard.get(&self.path) {
                return *cached;
            }
        }

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
                    let data_ptr = data.as_ptr() as usize;
                    for &(probe, lig_char) in LIGATURE_PROBES {
                        let Some(b) = lig_bit(lig_char) else { continue };
                        if (mask & (1u8<<b)) != 0 { continue; }
                        if probe.chars().count() <= 1 { continue; }
                        if shape_probes_collapsed(&face, data_ptr, &features, probe) {
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
        if let Some(ref data) = data_cache {
            let path_key = self.path.clone();
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
                    let g = Self::count_vertical_gaps_for_lig(data, lig);
                    let mut guard = gap_cache().write().unwrap();
                    guard.insert((path_key.clone(), lig), g);
                    g
                };
                if gaps >= required_gaps {
                    mask &= !(1u8<<b);
                }
            }
        }

        // Insert final filtered set into global cache.
        {
            let mut guard = lig_set_cache().write().unwrap();
            guard.insert(self.path.clone(), mask);
        }
        mask
    }

    /// Back-compat shim: old name kept for any missed call sites.
    #[allow(dead_code)]
    pub fn allowed_lig_set(&self) -> LigMask {
        self.collapsed_lig_set()
    }

    /// Count interior 100% top-to-bottom white vertical gutters inside the glyph
    /// for `lig_char`. No white padding around glyph — tight ink bounds only.
    /// Returns number of whitespace regions separating ink (0,1,2...).
    fn count_vertical_gaps_for_lig(data: &[u8], lig_char: char) -> usize {
        // High-res 200px no-hint gives outline truth. Low-res 40px was previously
        // computed but discarded — skip it to halve CPU.
        Self::vertical_gaps_at_scale(data, lig_char, 200.0, 20.0 / 255.0)
    }

    fn vertical_gaps_at_scale(data: &[u8], lig_char: char, px: f32, cov_thresh: f32) -> usize {
        let font = match FontRef::try_from_slice(data) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let scale = PxScale::from(px);
        let sf = font.as_scaled(scale);
        let gid = font.glyph_id(lig_char);
        if gid.0 == 0 {
            return 0;
        }
        let glyph = gid.with_scale_and_position(scale, unprint_fonts::ab_glyph::point(0.0, sf.ascent()));
        let outlined = match font.outline_glyph(glyph) {
            Some(o) => o,
            None => return 0,
        };
        let bounds = outlined.px_bounds();
        let w_f = bounds.max.x - bounds.min.x;
        let h_f = bounds.max.y - bounds.min.y;
        if w_f < 1.0 || h_f < 1.0 {
            return 0;
        }
        let w = w_f.ceil() as usize;
        let h = h_f.ceil() as usize;
        if w < 2 || h < 2 || w > 4096 || h > 4096 {
            return 0;
        }
        // Tight bitmap, no padding — single allocation.
        let mut ink = vec![false; w * h];
        outlined.draw(|gx, gy, cov| {
            if cov <= cov_thresh {
                return;
            }
            let x = gx as usize;
            let y = gy as usize;
            if x < w && y < h {
                ink[y * w + x] = true;
            }
        });

        // One pass: column ink presence
        let mut col_has = vec![false; w];
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                if ink[row + x] {
                    col_has[x] = true;
                }
            }
        }

        // Gap columns = columns with no ink at all
        let mut gap_cols = Vec::new();
        for x in 0..w {
            if !col_has[x] {
                gap_cols.push(x);
            }
        }
        if gap_cols.is_empty() {
            return 0;
        }

        // Group consecutive empty columns
        let mut gaps: Vec<(usize, usize)> = Vec::new();
        let mut start = gap_cols[0];
        let mut prev = gap_cols[0];
        for &cx in gap_cols.iter().skip(1) {
            if cx == prev + 1 {
                prev = cx;
            } else {
                gaps.push((start, prev));
                start = cx;
                prev = cx;
            }
        }
        gaps.push((start, prev));

        // Prefix sums of col_has for O(1) left/right mass
        let mut prefix = vec![0usize; w + 1];
        for i in 0..w {
            prefix[i + 1] = prefix[i] + (col_has[i] as usize);
        }

        let mut valid = 0usize;
        for (gs, ge) in gaps {
            if gs == 0 || ge + 1 >= w {
                continue; // edge artifact from ceil
            }
            let left_mass = prefix[gs]; // columns with ink in [0, gs)
            if left_mass < 1 {
                continue;
            }
            let right_mass = prefix[w] - prefix[ge + 1]; // [ge+1, w)
            if right_mass < 1 {
                continue;
            }
            if px >= 100.0 && (left_mass < 3 || right_mass < 3) {
                continue;
            }
            valid += 1;
        }
        valid
    }
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

    /// Content hash of the catalog: hash of sorted font_keys.
    /// Changes when fonts are added, removed, or renamed.
    fn compute_hash(entries: &[FontEntry]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        for e in entries {
            e.font_key_ref().hash(&mut hasher);
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
const FSCN_VERSION: u32 = 6;

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
        // path (as UTF-8 lossy)
        write_str(&mut w, &e.path.to_string_lossy())?;
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

        write_str(&mut w, &e.typographic_family)?;

        // vintage_era (Option<String>) - v3+
        match &e.vintage_era {
            None => w.write_all(&0xFFFF_FFFFu32.to_le_bytes())?,
            Some(s) => {
                write_str(&mut w, s)?;
            }
        }
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

        entries.push(FontEntry {
            path: PathBuf::from(path_str),
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
                let has_var = variations.is_some();
                FontEntry::compute_font_key_full(
                    &postscript_name,
                    &variant_tag,
                    vintage_era.as_deref(),
                    has_var,
                )
            },
            glyph_overrides,
            variations,
            typographic_family,
            vintage_era,
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

    // Dedup by font_key (now includes vintage_era and |var marker, so static vs variable vs vintage distinct)
    {
        use std::collections::HashSet;
        let mut seen_keys: HashSet<String> = HashSet::new();
        let before = fonts.len();
        fonts.retain(|f| seen_keys.insert(f.font_key()));
        let removed = before - fonts.len();
        if removed > 0 {
            if !quiet { eprintln!("[scan] Deduped {} entries by font_key ({} → {})", removed, before, fonts.len()); }
        }
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

    let added: Vec<&PathBuf> = current_set.difference(&cached_set).copied().collect();
    let removed: Vec<&PathBuf> = cached_set.difference(&current_set).copied().collect();
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
                let _fig_label = if fe.oldstyle_figures { "OLDSTYLE" } else { "lining" };

                // Detect ligature glyphs (liga + dlig)
                let ligatures = detect_ligature_glyphs(&fe.data);
                if !ligatures.is_empty() {
                }

                // Probe all OT features — emit a variant entry for each that changes glyphs
                // Skip when allowlist has no variant entries (common for 6-font tests)
                let variants = if need_ot_variants {
                    detect_ot_variants(&fe.data)
                } else {
                    Vec::new()
                };
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
                        variant_tag: tag.clone(),
                        font_key_cache: format!("{}|{}|{}", fe.postscript_name, tag, tag),
                        glyph_overrides: Some(combined),
                        variations: None,
                        typographic_family: fe.typographic_family.clone(),
                        vintage_era: None,
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
                            variant_tag: tag.clone(),
                            font_key_cache: format!("{}|{}|{}", fe.postscript_name, tag, tag),
                            glyph_overrides: if combined_lig.is_empty() { None } else { Some(combined_lig) },
                            variations: None,
                            typographic_family: fe.typographic_family.clone(),
                            vintage_era: None,
                        };
                        fonts.push(allcaps_entry);
                    }
                }
                // Drop font bytes — metadata + path is all we keep.
                // Index build and matching load from path on demand.
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
                // etc. renderings from the same file.
                let weight_instances = detect_weight_instances(path, fe.class);
                let is_variable = !weight_instances.is_empty();
                for wi in &weight_instances {
                    let var_ps = make_weight_explicit(&fe.raw_postscript_name, wi.os2_weight);
                    let var_tag = format!("wght{}", wi.os2_weight);
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
                        // Include |var so static "SourceSerif4-400Italic" and
                        // variable "SourceSerif4-400Italic|wght400|var" remain
                        // distinct after normalization.
                        font_key_cache: FontEntry::compute_font_key_full(
                            &var_ps,
                            &var_tag,
                            None,
                            true,
                        ),
                        glyph_overrides: None,
                        variations: Some(wi.axes.clone()),
                        typographic_family: fe.typographic_family.clone(),
                        vintage_era: None,
                    };
                    // Add ligature overrides to weight-instance entry
                    if !ligatures.is_empty() {
                        var_fe.glyph_overrides = Some(ligatures.clone());
                    }
                    fonts.push(var_fe);
                }

                // If this file is variable, mark the base entry as variable
                // too (variations Some) so its key becomes "...|var" and does
                // not collide with the static counterpart after normalization.
                if is_variable {
                    // Use the first instance's axes as representative; base
                    // entry's variations presence is what matters for key.
                    fe.variations = Some(weight_instances[0].axes.clone());
                    fe.font_key_cache = FontEntry::compute_font_key_full(
                        &fe.postscript_name,
                        &fe.variant_tag,
                        fe.vintage_era.as_deref(),
                        true,
                    );
                }

                fonts.push(fe);
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
        let font_key_cache = postscript_name.clone();
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

    let font_key_cache = postscript_name.clone();
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
