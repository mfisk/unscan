// ---------------------------------------------------------------------------
// NgramGlyphMap — per-sequence equivalence classes for identical renders
// ---------------------------------------------------------------------------
//
// When multiple fonts produce identical rendered images for a character
// sequence, they share a glyph hash.  The NgramGlyphMap assigns dense
// glyph_ids per sequence and maps each back to the set of font_keys that
// produce that image.  Classifiers operate on glyph_ids; font_match
// expands back to font_keys via this map.
//
// Supports arbitrary sequence lengths: seq_len=1 for single characters,
// seq_len=2 for bigrams, etc.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A group of font_keys that produce an identical rendered image.
pub struct GlyphGroup {
    /// Content hash of the rendered image.
    pub hash: u64,
    /// Font keys that produce this exact image.
    pub font_keys: Vec<String>,
}

/// Per-sequence mapping from glyph_id → GlyphGroup.
///
/// `glyph_id` is dense per sequence (0..n_unique_glyphs_for_that_seq).
/// Every font_key appears in exactly one group per sequence.
pub struct NgramGlyphMap {
    /// seq → Vec<GlyphGroup>, where group index = glyph_id
    pub groups: HashMap<Vec<char>, Vec<GlyphGroup>>,
    /// Catalog hash at build time, for staleness detection.
    pub catalog_hash: u64,
    /// Whether the map has been modified since last write.
    dirty: bool,
}

impl NgramGlyphMap {
    pub fn new(catalog_hash: u64) -> Self {
        Self { groups: HashMap::new(), catalog_hash, dirty: false }
    }

    /// Look up which font_keys share a glyph_id for a given sequence.
    pub fn fonts_for_glyph(&self, seq: &[char], glyph_id: usize) -> &[String] {
        self.groups.get(seq)
            .and_then(|g| g.get(glyph_id))
            .map(|g| g.font_keys.as_slice())
            .unwrap_or(&[])
    }

    /// Get the image hash for a glyph_id.
    pub fn hash_for_glyph(&self, seq: &[char], glyph_id: usize) -> Option<u64> {
        self.groups.get(seq)
            .and_then(|g| g.get(glyph_id))
            .map(|g| g.hash)
    }

    /// Look up the image hash for a (sequence, font_key) pair.
    pub fn hash_for_font(&self, seq: &[char], font_key: &str) -> Option<u64> {
        let glyph_id = self.glyph_id_for_font(seq, font_key)?;
        self.hash_for_glyph(seq, glyph_id)
    }

    /// Number of unique glyph groups for a sequence.
    pub fn glyph_count(&self, seq: &[char]) -> usize {
        self.groups.get(seq).map_or(0, |g| g.len())
    }

    /// Find the glyph_id for a given (sequence, font_key).
    pub fn glyph_id_for_font(&self, seq: &[char], font_key: &str) -> Option<usize> {
        self.groups.get(seq)?
            .iter()
            .position(|group| group.font_keys.iter().any(|k| k == font_key))
    }

    /// All font_keys across all groups for a sequence.
    pub fn all_font_keys(&self, seq: &[char]) -> Vec<&str> {
        self.groups.get(seq)
            .map(|gs| gs.iter().flat_map(|g| g.font_keys.iter().map(|s| s.as_str())).collect())
            .unwrap_or_default()
    }

    /// All unique font_keys across all sequences (cached font_meta).
    pub fn cached_font_keys(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        for groups in self.groups.values() {
            for g in groups {
                for k in &g.font_keys {
                    set.insert(k.clone());
                }
            }
        }
        set
    }

    /// Update catalog_hash after incremental addition.
    pub fn set_catalog_hash(&mut self, hash: u64) {
        if self.catalog_hash != hash {
            self.catalog_hash = hash;
            self.dirty = true;
        }
    }

    /// Register a rendered glyph: associate a font_key with an image hash
    /// for a given sequence. If the hash matches an existing group, the
    /// font_key is added to that group. Otherwise a new group is created.
    /// Returns (glyph_id, is_new_group).
    pub fn register(&mut self, seq: &[char], font_key: &str, hash: u64) -> (usize, bool) {
        let groups = self.groups.entry(seq.to_vec()).or_default();
        for (idx, group) in groups.iter_mut().enumerate() {
            if group.hash == hash {
                if !group.font_keys.iter().any(|k| k == font_key) {
                    group.font_keys.push(font_key.to_string());
                    self.dirty = true;
                }
                return (idx, false);
            }
        }
        let new_id = groups.len();
        groups.push(GlyphGroup {
            hash,
            font_keys: vec![font_key.to_string()],
        });
        self.dirty = true;
        (new_id, true)
    }

    /// Default cache path for the glyph map.
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Path::new(&home).join(".cache").join("unprint").join("glyph-map.bin")
    }

    // ---------------------------------------------------------------
    // Binary format (NGMP v3)
    //
    //   magic:        b"NGMP"  (4 bytes)
    //   version:      u32 le   (3)
    //   catalog_hash: u64 le
    //   n_entries:    u32 le
    //   per entry:
    //     seq_len:    u32 le
    //     codepoints: [u32 le; seq_len]
    //     n_groups:   u32 le
    //     per group:
    //       hash:     u64 le
    //       n_fonts:  u32 le
    //       per font:
    //         key_len: u32 le
    //         key:     [u8; key_len]
    // ---------------------------------------------------------------

    pub fn write_bin(&mut self, path: &Path) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = crate::atomic_file::tmp_for(path);
        let f = std::fs::File::create(&tmp)?;
        let mut w = BufWriter::new(f);

        w.write_all(b"NGMP")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&self.catalog_hash.to_le_bytes())?;

        w.write_all(&(self.groups.len() as u32).to_le_bytes())?;
        let mut entries: Vec<_> = self.groups.iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (seq, groups) in &entries {
            w.write_all(&(seq.len() as u32).to_le_bytes())?;
            for ch in seq.iter() {
                w.write_all(&(*ch as u32).to_le_bytes())?;
            }
            w.write_all(&(groups.len() as u32).to_le_bytes())?;
            for group in groups.iter() {
                w.write_all(&group.hash.to_le_bytes())?;
                w.write_all(&(group.font_keys.len() as u32).to_le_bytes())?;
                for key in &group.font_keys {
                    let b = key.as_bytes();
                    w.write_all(&(b.len() as u32).to_le_bytes())?;
                    w.write_all(b)?;
                }
            }
        }
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, path)?;
        self.dirty = false;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let data = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        if data.len() < 20 { return Err("NGMP too small".into()); }
        if &data[0..4] != b"NGMP" {
            return Err(format!("bad magic: {:?} (expected NGMP)", &data[0..4]));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 3 {
            return Err(format!("NGMP version {version}, need v3 — rebuild required"));
        }
        let catalog_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());

        // v3: n_entries at offset 16, per-entry seq_len, with hashes
        let n_entries = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let mut pos = 20;
        let mut groups = HashMap::with_capacity(n_entries);

        for _ in 0..n_entries {
            let seq_len = read_u32(&data, &mut pos)? as usize;
            if seq_len == 0 || seq_len > 16 {
                return Err(format!("invalid seq_len: {seq_len}"));
            }
            let mut seq = Vec::with_capacity(seq_len);
            for _ in 0..seq_len {
                let cp = read_u32(&data, &mut pos)?;
                seq.push(char::from_u32(cp)
                    .ok_or_else(|| format!("invalid codepoint U+{cp:04X}"))?);
            }

            let n_groups = read_u32(&data, &mut pos)? as usize;
            let mut entry_groups = Vec::with_capacity(n_groups);
            for _ in 0..n_groups {
                let hash = read_u64(&data, &mut pos)?;
                let n_fonts = read_u32(&data, &mut pos)? as usize;
                let mut font_keys = Vec::with_capacity(n_fonts);
                for _ in 0..n_fonts {
                    let klen = read_u32(&data, &mut pos)? as usize;
                    if pos + klen > data.len() { return Err("truncated key".into()); }
                    let key = String::from_utf8(data[pos..pos+klen].to_vec())
                        .map_err(|e| format!("invalid key UTF-8: {e}"))?;
                    pos += klen;
                    font_keys.push(key);
                }
                entry_groups.push(GlyphGroup { hash, font_keys });
            }
            groups.insert(seq, entry_groups);
        }

        Ok(Self { groups, catalog_hash, dirty: false })
    }
}

impl Drop for NgramGlyphMap {
    fn drop(&mut self) {
        if self.dirty {
            let path = Self::default_path();
            if let Err(e) = self.write_bin(&path) {
                eprintln!("warning: failed to write dirty glyph map to {}: {e}", path.display());
            }
        }
    }
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() { return Err("truncated u32".into()); }
    let v = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() { return Err("truncated u64".into()); }
    let v = u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

/// Content hash for a rendered glyph image.
pub fn hash_image(img: &image::GrayImage) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    img.width().hash(&mut hasher);
    img.height().hash(&mut hasher);
    img.as_raw().hash(&mut hasher);
    hasher.finish()
}

/// Hash as a hex string for use in filenames.
pub fn hash_hex(h: u64) -> String {
    format!("{:016x}", h)
}
