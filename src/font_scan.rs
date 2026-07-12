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
}

impl FontEntry {
    /// Unique key for this font entry in the font registry.
    /// Uses the canonical PostScript name (from `make_weight_explicit`) so
    /// duplicate font files with different paths but the same identity
    /// collapse to a single key.  Variant entries append `|tag`.
    pub fn font_key(&self) -> String {
        if self.variant_tag.is_empty() {
            self.postscript_name.clone()
        } else {
            format!("{}|{}", self.postscript_name, self.variant_tag)
        }
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
        // Sort by font_key for deterministic ordering and stable font_ids.
        entries.sort_by(|a, b| a.font_key().cmp(&b.font_key()));
        let by_key = entries.iter().enumerate()
            .map(|(i, e)| (e.font_key(), i))
            .collect();
        let catalog_hash = Self::compute_hash(&entries);
        Self { entries, by_key, catalog_hash }
    }

    /// Content hash of the catalog: hash of sorted font_keys.
    /// Changes when fonts are added, removed, or renamed.
    fn compute_hash(entries: &[FontEntry]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for e in entries {
            e.font_key().hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn by_key(&self, key: &str) -> Option<&FontEntry> {
        self.by_key.get(key).map(|&i| &self.entries[i])
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
const FSCN_VERSION: u32 = 2;

pub(crate) fn scan_cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("unprint").join("font_scan.bin")
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
    }

    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_scan_cache(path: &Path) -> Option<Vec<FontEntry>> {
    let data = std::fs::read(path).ok()?;
    let mut r: &[u8] = &data;

    // Header: magic(4) + version(4) + count(4)
    if r.len() < 12 { return None; }
    if &r[..4] != FSCN_MAGIC { return None; }
    r = &r[4..];
    let version = read_u32(&mut r)?;
    if version != FSCN_VERSION {
        eprintln!("[scan] Font scan cache is v{version}, need v{FSCN_VERSION} — rescanning...");
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

        entries.push(FontEntry {
            path: PathBuf::from(path_str),
            family_name,
            postscript_name,
            raw_postscript_name,
            is_bold,
            is_italic,
            class,
            data: Vec::new(),
            oldstyle_figures,
            variant_tag,
            glyph_overrides,
            variations,
            typographic_family,
        });
    }

    Some(entries)
}

/// Dedup font entries: drop variable-font weight instances covered by static
/// fonts, then dedup by font_key.
fn dedup_fonts(mut fonts: Vec<FontEntry>) -> Vec<FontEntry> {
    // Prefer static fonts over variable-font weight instances
    {
        use std::collections::HashSet;
        let static_keys: HashSet<(String, u16, bool)> = fonts.iter()
            .filter(|f| f.variations.is_none() && !f.variant_tag.starts_with("wght"))
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

        let before = fonts.len();
        fonts.retain(|f| {
            if !f.variant_tag.starts_with("wght") {
                return true;
            }
            let weight = f.variant_tag.strip_prefix("wght")
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            !static_keys.contains(&(f.typographic_family.clone(), weight, f.is_italic))
        });
        let removed = before - fonts.len();
        if removed > 0 {
            eprintln!("[scan] Dropped {} variable-font weight instances covered by static fonts", removed);
        }
    }

    // Dedup by font_key
    {
        use std::collections::HashSet;
        let mut seen_keys: HashSet<String> = HashSet::new();
        let before = fonts.len();
        fonts.retain(|f| seen_keys.insert(f.font_key()));
        let removed = before - fonts.len();
        if removed > 0 {
            eprintln!("[scan] Deduped {} entries by font_key ({} → {})", removed, before, fonts.len());
        }
    }

    fonts
}

/// Walk the given directories for .ttf / .otf files and return a catalogue.
pub fn scan_fonts(dirs: &[PathBuf]) -> Vec<FontEntry> {
    let current_paths = collect_font_paths(dirs);
    let cache_path = scan_cache_path();

    // Load cached entries indexed by source font file path
    let cached_by_path: std::collections::HashMap<PathBuf, Vec<FontEntry>> = {
        let mut map: std::collections::HashMap<PathBuf, Vec<FontEntry>> = std::collections::HashMap::new();
        if let Some(entries) = read_scan_cache(&cache_path) {
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
        eprintln!("[scan] Loaded {} font entries from cache", fonts.len());
        return dedup_fonts(fonts);
    }

    if !removed.is_empty() {
        eprintln!("[scan] {} font files removed", removed.len());
    }
    if !added.is_empty() {
        eprintln!("[scan] {} new font files to scan", added.len());
    }

    // Start with cached entries for paths that still exist
    let mut fonts: Vec<FontEntry> = current_paths.iter()
        .filter(|p| cached_by_path.contains_key(*p))
        .flat_map(|p| cached_by_path.get(p).cloned().unwrap_or_default())
        .collect();

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
                        glyph_overrides: Some(combined),
                        variations: None,
                        typographic_family: fe.typographic_family.clone(),
                    };
                    fonts.push(var_entry);
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
                for wi in &weight_instances {
                    let var_ps = make_weight_explicit(&fe.raw_postscript_name, wi.os2_weight);
                    let var_tag = format!("wght{}", wi.os2_weight);
                    let mut var_fe = FontEntry {
                        path: fe.path.clone(),
                        family_name: format!("{} [{}]", fe.family_name, var_tag),
                        postscript_name: var_ps,
                        raw_postscript_name: fe.raw_postscript_name.clone(),
                        is_bold: wi.os2_weight >= 700,
                        is_italic: fe.is_italic,
                        class: fe.class,
                        data: Vec::new(),
                        oldstyle_figures: fe.oldstyle_figures,
                        variant_tag: var_tag,
                        glyph_overrides: None,
                        variations: Some(wi.axes.clone()),
                        typographic_family: fe.typographic_family.clone(),
                    };
                    // Add ligature overrides to weight-instance entry
                    if !ligatures.is_empty() {
                        var_fe.glyph_overrides = Some(ligatures.clone());
                    }
                    fonts.push(var_fe);
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
                    glyph_overrides: None,
                    variations: None,
                    typographic_family: String::new(),
                });
            }
        }
    }

    // Write cache pre-dedup so every source path is represented
    if let Err(e) = write_scan_cache(&cache_path, &fonts) {
        eprintln!("[scan] Warning: failed to write font scan cache: {}", e);
    } else {
        eprintln!("[scan] Wrote {} font entries to cache", fonts.len());
    }

    // Filter out tombstone entries before dedup
    fonts.retain(|f| !f.family_name.is_empty());
    dedup_fonts(fonts)
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
    use ab_glyph::{FontRef, VariableFont};

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
        use rustybuzz::ttf_parser;
        match ttf_parser::Face::parse(&data, 0) {
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

    // Strip any trailing weight-word suffix before appending the numeric
    // weight.  This ensures two copies of the same font with different PS
    // naming conventions (e.g. "IBMPlexSerif-Regular" vs "IBMPlexSerif")
    // collapse to the same canonical name ("IBMPlexSerif-400").
    // The OS/2 weight class is the authority; the word suffix is just a label.
    let stem = strip_weight_suffix(ps_name);

    // Separate italic suffix — insert weight number before it.
    if let Some(idx) = stem.to_lowercase().find("italic") {
        let prefix = stem[..idx].trim_end_matches('-');
        let italic_part = &stem[idx..];
        format!("{}-{}{}", prefix, weight_str, italic_part)
    } else if stem.ends_with("It") {
        let prefix = stem[..stem.len()-2].trim_end_matches('-');
        format!("{}-{}It", prefix, weight_str)
    } else {
        format!("{}-{}", stem, weight_str)
    }
}

/// Strip a trailing weight-word suffix (e.g. "-Regular", "-Bold") from a
/// PostScript name, returning the family stem.  The suffix must appear after
/// a hyphen; bare names without a hyphen are returned unchanged.
fn strip_weight_suffix(ps_name: &str) -> &str {
    const WEIGHT_SUFFIXES: &[&str] = &[
        "-Regular",
        "-Bold",
        "-Light",
        "-Medium",
        "-Thin",
        "-ExtraLight",
        "-UltraLight",
        "-SemiBold",
        "-DemiBold",
        "-ExtraBold",
        "-UltraBold",
        "-Heavy",
        "-Black",
    ];
    for suffix in WEIGHT_SUFFIXES {
        if ps_name.ends_with(suffix) {
            return &ps_name[..ps_name.len() - suffix.len()];
        }
    }
    ps_name
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
    use rustybuzz::ttf_parser;
    let face = match ttf_parser::Face::parse(data, 0) {
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

    Some(FontIdentity { family, _weight: weight, italic })
}

fn load_font_entry(path: &Path, aliases: &HashMap<String, Alias>) -> Option<FontEntry> {
    let data = std::fs::read(path).ok()?;

    // Verify ab_glyph can parse it (reject corrupt files)
    let _ = ab_glyph::FontRef::try_from_slice(&data).ok()?;

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
            glyph_overrides: None,
            variations: None,
            typographic_family,
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
        raw_postscript_name: raw_ps_name,
        is_bold,
        is_italic,
        class,
        data,
        oldstyle_figures,
        variant_tag: String::new(),
        glyph_overrides: None,
        variations: None,
        typographic_family,
    })
}
