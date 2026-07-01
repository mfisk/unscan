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

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Per-sequence mapping from glyph_id → Vec<font_key>.
///
/// `glyph_id` is dense per sequence (0..n_unique_glyphs_for_that_seq).
/// Every font_key appears in exactly one group per sequence.
pub struct NgramGlyphMap {
    /// seq → Vec<group>, where group index = glyph_id, group = Vec<font_key>
    pub groups: HashMap<Vec<char>, Vec<Vec<String>>>,
    /// Catalog hash at build time, for staleness detection.
    pub catalog_hash: u64,
}

impl NgramGlyphMap {
    pub fn new(catalog_hash: u64) -> Self {
        Self { groups: HashMap::new(), catalog_hash }
    }

    /// Look up which font_keys share a glyph_id for a given sequence.
    pub fn fonts_for_glyph(&self, seq: &[char], glyph_id: usize) -> &[String] {
        self.groups.get(seq)
            .and_then(|g| g.get(glyph_id))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of unique glyph groups for a sequence.
    pub fn glyph_count(&self, seq: &[char]) -> usize {
        self.groups.get(seq).map_or(0, |g| g.len())
    }

    /// Find the glyph_id for a given (sequence, font_key).
    pub fn glyph_id_for_font(&self, seq: &[char], font_key: &str) -> Option<usize> {
        self.groups.get(seq)?
            .iter()
            .position(|group| group.iter().any(|k| k == font_key))
    }

    /// All font_keys across all groups for a sequence.
    pub fn all_font_keys(&self, seq: &[char]) -> Vec<&str> {
        self.groups.get(seq)
            .map(|gs| gs.iter().flat_map(|g| g.iter().map(|s| s.as_str())).collect())
            .unwrap_or_default()
    }

    /// Default cache path for the glyph map.
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Path::new(&home).join(".cache").join("unprint").join("glyph-map.bin")
    }

    // ---------------------------------------------------------------
    // Binary format (NGMP v1)
    //
    //   magic:        b"NGMP"  (4 bytes)
    //   version:      u32 le   (1)
    //   catalog_hash: u64 le
    //   seq_len:      u32 le
    //   n_entries:    u32 le
    //   per entry:
    //     codepoints: [u32 le; seq_len]
    //     n_groups:   u32 le
    //     per group:
    //       n_fonts:  u32 le
    //       per font:
    //         key_len: u32 le
    //         key:     [u8; key_len]
    // ---------------------------------------------------------------

    pub fn write_bin(&self, path: &Path) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(b"NGMP")?;
        w.write_all(&2u32.to_le_bytes())?;  // v2: per-entry seq_len
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
                w.write_all(&(group.len() as u32).to_le_bytes())?;
                for key in group {
                    let b = key.as_bytes();
                    w.write_all(&(b.len() as u32).to_le_bytes())?;
                    w.write_all(b)?;
                }
            }
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if data.len() < 20 { return Err("NGMP too small".into()); }
        if &data[0..4] != b"NGMP" {
            return Err(format!("bad magic: {:?} (expected NGMP)", &data[0..4]));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 2 { return Err(format!("unsupported NGMP version: {version} (expected 2)")); }
        let catalog_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());
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
                entry_groups.push(font_keys);
            }
            groups.insert(seq, entry_groups);
        }

        Ok(Self { groups, catalog_hash })
    }
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() { return Err("truncated u32".into()); }
    let v = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap());
    *pos += 4;
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
