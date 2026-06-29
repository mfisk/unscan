// ---------------------------------------------------------------------------
// GlyphMap — per-character equivalence classes for identical renders
// ---------------------------------------------------------------------------
//
// When multiple fonts produce identical rendered images for a character,
// they share a glyph hash.  The GlyphMap assigns dense glyph_ids per
// character and maps each back to the set of font_keys that produce
// that image.  Classifiers operate on glyph_ids; font_match expands
// back to font_keys via this map.
//
// Stored as its own `.bin` file, shared by all classifiers.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Per-character mapping from glyph_id → Vec<font_key>.
///
/// `glyph_id` is dense per character (0..n_unique_glyphs_for_that_char).
/// Every font_key appears in exactly one group per character.
pub struct GlyphMap {
    /// char → Vec<group>, where group index = glyph_id, group = Vec<font_key>
    pub groups: HashMap<char, Vec<Vec<String>>>,
    /// Catalog hash at build time, for staleness detection.
    pub catalog_hash: u64,
}

impl GlyphMap {
    pub fn new(catalog_hash: u64) -> Self {
        Self { groups: HashMap::new(), catalog_hash }
    }

    /// Look up which font_keys share a glyph_id for a given character.
    pub fn fonts_for_glyph(&self, ch: char, glyph_id: usize) -> &[String] {
        self.groups.get(&ch)
            .and_then(|g| g.get(glyph_id))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of unique glyph groups for a character.
    pub fn glyph_count(&self, ch: char) -> usize {
        self.groups.get(&ch).map_or(0, |g| g.len())
    }

    /// Find the glyph_id for a given (char, font_key).
    pub fn glyph_id_for_font(&self, ch: char, font_key: &str) -> Option<usize> {
        self.groups.get(&ch)?
            .iter()
            .position(|group| group.iter().any(|k| k == font_key))
    }

    /// All font_keys across all groups for a character.
    pub fn all_font_keys(&self, ch: char) -> Vec<&str> {
        self.groups.get(&ch)
            .map(|gs| gs.iter().flat_map(|g| g.iter().map(|s| s.as_str())).collect())
            .unwrap_or_default()
    }

    /// Default cache path.
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Path::new(&home).join(".cache").join("unprint").join("glyph-map.bin")
    }

    // ---------------------------------------------------------------
    // Binary format (GMAP v1)
    //
    //   magic:        b"GMAP"  (4 bytes)
    //   version:      u32 le   (1)
    //   catalog_hash: u64 le
    //   n_chars:      u32 le
    //   per char:
    //     codepoint:  u32 le
    //     n_groups:   u32 le
    //     per group:
    //       n_fonts:  u32 le
    //       per font:
    //         key_len: u32 le
    //         key:     [u8; key_len]
    // ---------------------------------------------------------------

    pub fn write_bin(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(b"GMAP")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&self.catalog_hash.to_le_bytes())?;

        w.write_all(&(self.groups.len() as u32).to_le_bytes())?;
        // Sort by codepoint for deterministic output
        let mut chars: Vec<_> = self.groups.iter().collect();
        chars.sort_by_key(|(ch, _)| **ch as u32);

        for &(&ch, groups) in &chars {
            w.write_all(&(ch as u32).to_le_bytes())?;
            w.write_all(&(groups.len() as u32).to_le_bytes())?;
            for group in groups {
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
        let f = std::fs::File::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut r = BufReader::new(f);

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4).map_err(|e| format!("read magic: {e}"))?;
        if &buf4 != b"GMAP" {
            return Err(format!("bad magic: {:?}", buf4));
        }
        r.read_exact(&mut buf4).map_err(|e| format!("read version: {e}"))?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(format!("unsupported version: {version}"));
        }
        r.read_exact(&mut buf8).map_err(|e| format!("read catalog_hash: {e}"))?;
        let catalog_hash = u64::from_le_bytes(buf8);

        r.read_exact(&mut buf4).map_err(|e| format!("read n_chars: {e}"))?;
        let n_chars = u32::from_le_bytes(buf4) as usize;

        let mut groups = HashMap::with_capacity(n_chars);
        for _ in 0..n_chars {
            r.read_exact(&mut buf4).map_err(|e| format!("read codepoint: {e}"))?;
            let ch = char::from_u32(u32::from_le_bytes(buf4))
                .ok_or_else(|| "invalid codepoint".to_string())?;

            r.read_exact(&mut buf4).map_err(|e| format!("read n_groups: {e}"))?;
            let n_groups = u32::from_le_bytes(buf4) as usize;

            let mut char_groups = Vec::with_capacity(n_groups);
            for _ in 0..n_groups {
                r.read_exact(&mut buf4).map_err(|e| format!("read n_fonts: {e}"))?;
                let n_fonts = u32::from_le_bytes(buf4) as usize;

                let mut font_keys = Vec::with_capacity(n_fonts);
                for _ in 0..n_fonts {
                    r.read_exact(&mut buf4).map_err(|e| format!("read key_len: {e}"))?;
                    let key_len = u32::from_le_bytes(buf4) as usize;
                    let mut key_buf = vec![0u8; key_len];
                    r.read_exact(&mut key_buf).map_err(|e| format!("read key: {e}"))?;
                    let key = String::from_utf8(key_buf)
                        .map_err(|e| format!("invalid key UTF-8: {e}"))?;
                    font_keys.push(key);
                }
                char_groups.push(font_keys);
            }
            groups.insert(ch, char_groups);
        }

        Ok(Self { groups, catalog_hash })
    }
}

/// Hash a grayscale image's pixel data for dedup.
/// Returns a 64-bit hash suitable for grouping identical renders.
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
