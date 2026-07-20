//! Geometry cache BGEO v7 - Unicode, full GPOS (Single+Pair Format1/Format2), mmap zero-copy.
//!
//! Stores full GPOS value records (xPlacement, yPlacement, xAdvance, yAdvance)
//! with smallest types: u16/i16/u8, masked valueFormat (0x000F), only present
//! fields are stored (popcount determines stride per table). No rejection for
//! y/x placement offsets.
//!
//! File layout (all LE):
//!   header: magic b"BGEO" | version u32=7 | catalog_hash u64 | n_fonts u32
//!   per font:
//!     key_len u32 | key bytes | file_hash u64 | upem u16 | num_glyphs u16
//!     glyphs: [num_glyphs] * { advance i16, x_min i16, x_max i16, y_min i16, y_max i16 } (10 bytes each)
//!     cmap_len u32 | [cmap_len] * { codepoint u32, gid u16 } (6 bytes each, no pad)
//!     n_single u16 | n_format1 u16 | n_format2 u16
//!     single_tables: [n_single] * {
//!       coverage_len u16 | coverage[u16; coverage_len] |
//!       value_format u16 (masked 0xF) | is_single u8 |
//!       values: [ (is_single?1:coverage_len) * popcnt(value_format) ] i16
//!     }
//!     format1_tables: [n_format1] * {
//!       coverage_len u16 | coverage[u16; coverage_len] |
//!       val_fmt1 u16 | val_fmt2 u16 |
//!       pair_sets: [coverage_len] * { pair_count u16 | pairs[ pair_count * { second_gid u16, val1[popcnt(vf1) i16], val2[popcnt(vf2) i16] } ] }
//!     }
//!     format2_tables: [n_format2] * {
//!       coverage_len u16 | coverage[u16; coverage_len] | class1_count u16 | class2_count u16 |
//!       val_fmt1 u16 | val_fmt2 u16 |
//!       class_def1: { fmt u8, _pad u8, count u16, start u16, data... }
//!         fmt=1: start u16 | classes[u16; count]
//!         fmt=2: ranges[{start u16, end u16, class u16}; count] (6 bytes each)
//!       class_def2: same
//!       matrix: [class1_count*class2_count] * { val1[popcnt(vf1) i16], val2[popcnt(vf2) i16] }
//!     }

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const BGEO_MAGIC: &[u8; 4] = b"BGEO";
const BGEO_VERSION: u32 = 7;

// ---------- BE helpers for OT parsing ----------
#[inline]
fn be_u16(d: &[u8], off: usize) -> Option<u16> {
    if off + 2 > d.len() { return None; }
    Some(u16::from_be_bytes([d[off], d[off+1]]))
}
#[inline]
fn be_i16(d: &[u8], off: usize) -> Option<i16> { be_u16(d, off).map(|v| v as i16) }

// ---------- LE helpers for BGEO file ----------
#[inline] fn le_u16_at(d: &[u8], off: usize) -> u16 { u16::from_le_bytes(d[off..off+2].try_into().unwrap()) }
#[inline] fn le_i16_at(d: &[u8], off: usize) -> i16 { i16::from_le_bytes(d[off..off+2].try_into().unwrap()) }
#[inline] fn le_u32_at(d: &[u8], off: usize) -> u32 { u32::from_le_bytes(d[off..off+4].try_into().unwrap()) }
#[inline] fn le_u64_at(d: &[u8], off: usize) -> u64 { u64::from_le_bytes(d[off..off+8].try_into().unwrap()) }

fn read_u16_le(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() { return Err("trunc u16".into()); }
    let v = u16::from_le_bytes(data[*pos..*pos+2].try_into().unwrap()); *pos+=2; Ok(v)
}
fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() { return Err("trunc u32".into()); }
    let v = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()); *pos+=4; Ok(v)
}
fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() { return Err("trunc u64".into()); }
    let v = u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()); *pos+=8; Ok(v)
}

#[inline]
fn popcnt4(v: u16) -> usize {
    ((v & 0x000F).count_ones()) as usize
}

#[inline]
fn vf_mask(v: u16) -> u16 { v & 0x000F }

#[inline]
fn unpack_vals_4(d: &[u8], off: usize, vf: u16) -> ([i16;4], usize) {
    // vf is already masked to 0xF
    let mut out = [0i16;4];
    let mut p = off;
    if vf & 0x0001 != 0 { if p+2 <= d.len() { out[0]=le_i16_at(d,p); } p+=2; }
    if vf & 0x0002 != 0 { if p+2 <= d.len() { out[1]=le_i16_at(d,p); } p+=2; }
    if vf & 0x0004 != 0 { if p+2 <= d.len() { out[2]=le_i16_at(d,p); } p+=2; }
    if vf & 0x0008 != 0 { if p+2 <= d.len() { out[3]=le_i16_at(d,p); } p+=2; }
    (out, p-off)
}

#[inline]
fn pack_vals_4<W: std::io::Write>(w: &mut W, vals: &[i16;4], vf: u16) -> std::io::Result<()> {
    if vf & 0x0001 != 0 { w.write_all(&vals[0].to_le_bytes())?; }
    if vf & 0x0002 != 0 { w.write_all(&vals[1].to_le_bytes())?; }
    if vf & 0x0004 != 0 { w.write_all(&vals[2].to_le_bytes())?; }
    if vf & 0x0008 != 0 { w.write_all(&vals[3].to_le_bytes())?; }
    Ok(())
}

fn file_hash_bytes(data: &[u8]) -> u64 {
    // FNV-1a 64 - stable across processes (DefaultHasher is SipHash with random keys)
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ---------- Owned structures (build phase only) ----------
#[derive(Copy, Clone, Debug)]
struct GlyphMetricsOwned { advance: f64, x_min: f64, x_max: f64, y_min: f64, y_max: f64 }

type Val4 = [i16;4]; // [xPla, yPla, xAdv, yAdv] in order 0x1,0x2,0x4,0x8

#[derive(Clone, Debug)]
struct SingleTableOwned {
    coverage: Vec<u16>,
    value_format: u16, // masked 0xF
    values: Vec<Val4>, // len = coverage.len() (or 1 if is_single, expanded on read)
    is_single: bool, // true if original was Format1 single-value (one record for all)
}

#[derive(Clone, Debug)]
struct Format1TableOwned {
    coverage: Vec<u16>,
    val_fmt1: u16, // masked
    val_fmt2: u16, // masked
    pair_sets: Vec<Vec<(u16, Val4, Val4)>>, // per left gid: Vec<(second_gid, val1, val2)>
}

#[derive(Clone, Debug)]
enum ClassDefOwned {
    Format1 { start: u16, classes: Vec<u16> },
    Format2 { ranges: Vec<(u16,u16,u16)> },
}

#[derive(Clone, Debug)]
struct Format2TableOwned {
    coverage: Vec<u16>,
    class_def1: ClassDefOwned,
    class_def2: ClassDefOwned,
    class1_count: usize,
    class2_count: usize,
    val_fmt1: u16,
    val_fmt2: u16,
    matrix: Vec<(Val4, Val4)>, // len = class1_count*class2_count, each (val1,val2)
}

struct OwnedFont {
    file_hash: u64,
    units_per_em: f64,
    num_glyphs: usize,
    glyphs: Vec<GlyphMetricsOwned>,
    cmap: Vec<(u32,u16)>,
    single_tables: Vec<SingleTableOwned>,
    format1_tables: Vec<Format1TableOwned>,
    format2_tables: Vec<Format2TableOwned>,
}

// ---------- Mmap index (runtime) ----------
#[derive(Debug)]
struct SingleIndex {
    coverage_off: usize,
    coverage_len: usize,
    value_format: u16,
    is_single: bool,
    values_off: usize,
    stride: usize, // popcnt in i16 units
}

#[derive(Debug)]
struct PairSetIndex { pairs_off: usize, pair_count: usize }

#[derive(Debug)]
struct Format1Index {
    coverage_off: usize,
    coverage_len: usize,
    val_fmt1: u16,
    val_fmt2: u16,
    sz1: usize,
    sz2: usize,
    pair_sets: Vec<PairSetIndex>,
}

#[derive(Debug)]
enum ClassDefIndex {
    Format1 { start: u16, count: usize, classes_off: usize },
    Format2 { count: usize, ranges_off: usize },
}

#[derive(Debug)]
struct Format2Index {
    coverage_off: usize,
    coverage_len: usize,
    class1_count: usize,
    class2_count: usize,
    val_fmt1: u16,
    val_fmt2: u16,
    sz1: usize,
    sz2: usize,
    class1_def: ClassDefIndex,
    class2_def: ClassDefIndex,
    matrix_off: usize,
}

#[derive(Debug)]
struct FontMmapIndex {
    file_hash: u64,
    upem: f64,
    num_glyphs: usize,
    glyphs_off: usize,
    cmap_off: usize,
    cmap_len: usize,
    single_tables: Vec<SingleIndex>,
    format1_tables: Vec<Format1Index>,
    format2_tables: Vec<Format2Index>,
}

// ---------- Backward compat parse structs for test_gpos ----------
#[derive(Clone, Debug)]
pub struct Format1Parse { pub coverage: Vec<u16>, pub pair_sets: Vec<Vec<(u16,i16)>> }
#[derive(Clone, Debug)]
pub struct Format2Parse { pub coverage: Vec<u16>, pub class_def1: ClassDefParse, pub class_def2: ClassDefParse, pub class1_count: usize, pub class2_count: usize, pub matrix: Vec<i16> }
#[derive(Clone, Debug)]
pub enum ClassDefParse { Format1{start:u16,classes:Vec<u16>}, Format2{ranges:Vec<(u16,u16,u16)>} }

// ---------- Main cache (mmap zero-copy) ----------
pub struct GeometryCache {
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    fonts: HashMap<String, FontMmapIndex>,
    _cache_path: PathBuf,
}

impl GeometryCache {
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Path::new(&home).join(".cache").join("unprint").join("geo-cache.bin")
    }

    pub fn has_font(&self, font_key: &str) -> bool { self.fonts.contains_key(font_key) }

    #[inline] fn data(&self) -> &[u8] { &self.mmap }

    #[inline]
    fn glyph_metrics(&self, f: &FontMmapIndex, gid: usize) -> Option<(f64,f64,f64,f64,f64)> {
        if gid >= f.num_glyphs { return None; }
        let off = f.glyphs_off + gid*10;
        let d = self.data();
        if off+10 > d.len() { return None; }
        let adv = i16::from_le_bytes(d[off..off+2].try_into().unwrap()) as f64;
        let x0 = i16::from_le_bytes(d[off+2..off+4].try_into().unwrap()) as f64;
        let x1 = i16::from_le_bytes(d[off+4..off+6].try_into().unwrap()) as f64;
        let y0 = i16::from_le_bytes(d[off+6..off+8].try_into().unwrap()) as f64;
        let y1 = i16::from_le_bytes(d[off+8..off+10].try_into().unwrap()) as f64;
        Some((adv,x0,x1,y0,y1))
    }

    #[inline]
    fn cmap_gid(&self, f: &FontMmapIndex, ch: char) -> Option<u16> {
        let cp = ch as u32;
        let d = self.data();
        let mut lo = 0usize;
        let mut hi = f.cmap_len;
        while lo < hi {
            let mid = (lo+hi)/2;
            let off = f.cmap_off + mid*6;
            if off+6 > d.len() { break; }
            let mid_cp = le_u32_at(d, off);
            if mid_cp < cp { lo = mid+1; }
            else if mid_cp > cp { hi = mid; }
            else { return Some(le_u16_at(d, off+4)); }
        }
        None
    }

    fn class_get(&self, cd: &ClassDefIndex, gid: u16) -> usize {
        let d = self.data();
        match cd {
            ClassDefIndex::Format1 { start, count, classes_off } => {
                if gid < *start { return 0; }
                let idx = (gid - *start) as usize;
                if idx >= *count { return 0; }
                let off = *classes_off + idx*2;
                if off+2 > d.len() { return 0; }
                le_u16_at(d, off) as usize
            }
            ClassDefIndex::Format2 { count, ranges_off } => {
                for i in 0..*count {
                    let off = *ranges_off + i*6;
                    if off+6 > d.len() { break; }
                    let s = le_u16_at(d, off);
                    let e = le_u16_at(d, off+2);
                    let c = le_u16_at(d, off+4);
                    if gid >= s && gid <= e { return c as usize; }
                }
                0
            }
        }
    }

    #[inline]
    fn coverage_contains(&self, cov_off: usize, cov_len: usize, gid: u16) -> Option<usize> {
        let d = self.data();
        let mut lo = 0usize;
        let mut hi = cov_len;
        while lo < hi {
            let mid = (lo+hi)/2;
            let off = cov_off + mid*2;
            if off+2 > d.len() { break; }
            let v = le_u16_at(d, off);
            if v < gid { lo = mid+1; }
            else if v > gid { hi = mid; }
            else { return Some(mid); }
        }
        None
    }

    // single adjustment sum across all single tables, returns [xPla,yPla,xAdv,yAdv] as i32
    fn single_adjustment_full(&self, f: &FontMmapIndex, gid: u16) -> [i32;4] {
        let d = self.data();
        let mut acc = [0i32;4];
        for tbl in &f.single_tables {
            let idx_opt = if tbl.is_single {
                // single value applies to all in coverage - need to know if gid in coverage
                if self.coverage_contains(tbl.coverage_off, tbl.coverage_len, gid).is_none() { continue; }
                Some(0usize)
            } else {
                self.coverage_contains(tbl.coverage_off, tbl.coverage_len, gid)
            };
            let idx = match idx_opt { Some(i)=>i, None=>continue };
            let vf = tbl.value_format;
            let sz = tbl.stride;
            let off = tbl.values_off + idx*sz*2;
            if off + sz*2 > d.len() { continue; }
            // unpack in order
            let mut p = off;
            if vf & 0x0001 != 0 { acc[0] += le_i16_at(d,p) as i32; p+=2; }
            if vf & 0x0002 != 0 { acc[1] += le_i16_at(d,p) as i32; p+=2; }
            if vf & 0x0004 != 0 { acc[2] += le_i16_at(d,p) as i32; p+=2; }
            if vf & 0x0008 != 0 { acc[3] += le_i16_at(d,p) as i32; }
        }
        acc
    }

    // pair adjustment for adjacent gids, returns (val1, val2) each [xPla,yPla,xAdv,yAdv] summed
    fn pair_adjustment_full(&self, f: &FontMmapIndex, gid1: u16, gid2: u16) -> ([i32;4],[i32;4]) {
        let d = self.data();
        let mut v1_acc = [0i32;4];
        let mut v2_acc = [0i32;4];
        // Format1
        for tbl in &f.format1_tables {
            let cov_idx = match self.coverage_contains(tbl.coverage_off, tbl.coverage_len, gid1) {
                Some(i)=>i, None=>continue,
            };
            let ps = &tbl.pair_sets[cov_idx];
            // binary search second gid in pairs
            let mut lo = 0usize;
            let mut hi = ps.pair_count;
            let mut found_off: Option<usize> = None;
            while lo < hi {
                let mid = (lo+hi)/2;
                let rec_size = 2 + tbl.sz1*2 + tbl.sz2*2;
                let off = ps.pairs_off + mid*rec_size;
                if off+2 > d.len() { break; }
                let sg = le_u16_at(d, off);
                if sg < gid2 { lo = mid+1; }
                else if sg > gid2 { hi = mid; }
                else { found_off = Some(off); break; }
            }
            let off = match found_off { Some(o)=>o, None=>continue };
            let mut p = off + 2;
            let vf1 = tbl.val_fmt1;
            if vf1 & 0x0001 != 0 { v1_acc[0] += le_i16_at(d,p) as i32; p+=2; }
            if vf1 & 0x0002 != 0 { v1_acc[1] += le_i16_at(d,p) as i32; p+=2; }
            if vf1 & 0x0004 != 0 { v1_acc[2] += le_i16_at(d,p) as i32; p+=2; }
            if vf1 & 0x0008 != 0 { v1_acc[3] += le_i16_at(d,p) as i32; p+=2; }
            let vf2 = tbl.val_fmt2;
            if vf2 & 0x0001 != 0 { v2_acc[0] += le_i16_at(d,p) as i32; p+=2; }
            if vf2 & 0x0002 != 0 { v2_acc[1] += le_i16_at(d,p) as i32; p+=2; }
            if vf2 & 0x0004 != 0 { v2_acc[2] += le_i16_at(d,p) as i32; p+=2; }
            if vf2 & 0x0008 != 0 { v2_acc[3] += le_i16_at(d,p) as i32; }
        }
        // Format2
        for tbl in &f.format2_tables {
            if self.coverage_contains(tbl.coverage_off, tbl.coverage_len, gid1).is_none() { continue; }
            let c1 = self.class_get(&tbl.class1_def, gid1);
            let c2 = self.class_get(&tbl.class2_def, gid2);
            if c1 >= tbl.class1_count || c2 >= tbl.class2_count { continue; }
            let idx = c1 * tbl.class2_count + c2;
            let rec_size = (tbl.sz1 + tbl.sz2)*2;
            let off = tbl.matrix_off + idx*rec_size;
            if off + rec_size > d.len() { continue; }
            let mut p = off;
            let vf1 = tbl.val_fmt1;
            if vf1 & 0x0001 != 0 { v1_acc[0] += le_i16_at(d,p) as i32; p+=2; }
            if vf1 & 0x0002 != 0 { v1_acc[1] += le_i16_at(d,p) as i32; p+=2; }
            if vf1 & 0x0004 != 0 { v1_acc[2] += le_i16_at(d,p) as i32; p+=2; }
            if vf1 & 0x0008 != 0 { v1_acc[3] += le_i16_at(d,p) as i32; p+=2; }
            let vf2 = tbl.val_fmt2;
            if vf2 & 0x0001 != 0 { v2_acc[0] += le_i16_at(d,p) as i32; p+=2; }
            if vf2 & 0x0002 != 0 { v2_acc[1] += le_i16_at(d,p) as i32; p+=2; }
            if vf2 & 0x0004 != 0 { v2_acc[2] += le_i16_at(d,p) as i32; p+=2; }
            if vf2 & 0x0008 != 0 { v2_acc[3] += le_i16_at(d,p) as i32; }
        }
        (v1_acc, v2_acc)
    }

    #[inline]
    fn kern_for_gids(&self, f: &FontMmapIndex, gid1: u16, gid2: u16) -> i32 {
        // legacy: sum of xAdvance adjustments (val1[2] + val2[2])
        let (v1, v2) = self.pair_adjustment_full(f, gid1, gid2);
        // also include single xAdvance? previous impl only did pair kern, keep same
        v1[2] + v2[2]
    }

    pub fn predict_glyph_positions(&self, font_key: &str, chars: &[char]) -> Option<Vec<(f64,f64)>> {
        let f = self.fonts.get(font_key)?;
        let mut gids: Vec<Option<u16>> = Vec::with_capacity(chars.len());
        for &c in chars { gids.push(self.cmap_gid(f, c)); }
        // early exit if any missing
        for g in &gids { if g.is_none() { return None; } }
        let n = gids.len();
        if n==0 { return Some(Vec::new()); }

        // precompute singles
        let mut singles: Vec<[i32;4]> = Vec::with_capacity(n);
        for i in 0..n {
            let gid = gids[i].unwrap();
            singles.push(self.single_adjustment_full(f, gid));
        }
        // precompute pair adjs
        let mut pair_firsts: Vec<[i32;4]> = vec![[0;4]; n];
        let mut pair_seconds: Vec<[i32;4]> = vec![[0;4]; n];
        if n>1 {
            for i in 0..n-1 {
                let g1 = gids[i].unwrap();
                let g2 = gids[i+1].unwrap();
                let (v1,v2) = self.pair_adjustment_full(f, g1, g2);
                pair_firsts[i] = v1;
                pair_seconds[i+1] = v2;
                // also need cross? v1 belongs to i, v2 belongs to i+1
            }
        }

        let mut out = Vec::with_capacity(n);
        let mut cursor = 0.0f64;
        for i in 0..n {
            let gid = gids[i].unwrap() as usize;
            let (adv, x0, x1, y0, y1) = self.glyph_metrics(f, gid)?;
            let s = singles[i];
            let pf = pair_firsts[i];
            let ps = pair_seconds[i];
            let x_pla = (s[0] + pf[0] + ps[0]) as f64;
            let y_pla = (s[1] + pf[1] + ps[1]) as f64;
            let x_adv_adj = (s[2] + pf[2] + ps[2]) as f64;
            let total_adv = adv + x_adv_adj;
            let cx = cursor + x_pla + (x0 + x1)*0.5;
            let cy = y_pla + (y0 + y1)*0.5;
            out.push((cx,cy));
            cursor += total_adv;
        }
        Some(out)
    }

    pub fn predict_word_ink_extent(&self, font_key: &str, chars: &[char], _font_data: &[u8], _em_px: f64) -> Option<(f64,f64)> {
        let f = self.fonts.get(font_key)?;
        let mut gids: Vec<Option<u16>> = Vec::with_capacity(chars.len());
        for &c in chars { gids.push(self.cmap_gid(f, c)); }
        for g in &gids { if g.is_none() { return None; } }
        let n = gids.len();
        if n==0 { return Some((1.0,1.0)); }

        let mut singles: Vec<[i32;4]> = Vec::with_capacity(n);
        for i in 0..n { singles.push(self.single_adjustment_full(f, gids[i].unwrap())); }
        let mut pair_firsts: Vec<[i32;4]> = vec![[0;4]; n];
        let mut pair_seconds: Vec<[i32;4]> = vec![[0;4]; n];
        for i in 0..n.saturating_sub(1) {
            let (v1,v2)=self.pair_adjustment_full(f, gids[i].unwrap(), gids[i+1].unwrap());
            pair_firsts[i]=v1;
            pair_seconds[i+1]=v2;
        }

        let mut total_adv = 0.0f64;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..n {
            let gid = gids[i].unwrap() as usize;
            let (adv, _x0, _x1, y0, y1) = self.glyph_metrics(f, gid)?;
            let s = singles[i];
            let pf = pair_firsts[i];
            let ps = pair_seconds[i];
            let y_pla = (s[1] + pf[1] + ps[1]) as f64;
            let x_adv_adj = (s[2] + pf[2] + ps[2]) as f64;
            total_adv += adv + x_adv_adj;
            let y0a = y0 + y_pla;
            let y1a = y1 + y_pla;
            min_y = min_y.min(y0a.min(y1a));
            max_y = max_y.max(y0a.max(y1a));
        }
        if !min_y.is_finite() { min_y=0.0; max_y=1.0; }
        Some((total_adv.max(1.0), (max_y-min_y).max(1.0)))
    }

    pub fn predict_word_ink_width_sum(&self, font_key: &str, chars: &[char]) -> Option<f64> {
        let f = self.fonts.get(font_key)?;
        let mut total_ink_w = 0.0f64;
        for &c in chars {
            let gid = self.cmap_gid(f, c)? as usize;
            let (_adv, x0, x1, _y0, _y1) = self.glyph_metrics(f, gid)?;
            total_ink_w += (x1 - x0) as f64;
        }
        Some(total_ink_w.max(1.0))
    }

    pub fn predict_glyph_x(&self, _seg_idx: usize, _orig_idx: usize) -> Option<f64> { None }
    pub fn predict_glyph_y(&self, _seg_idx: usize, _orig_idx: usize) -> Option<f64> { None }

    // reconstruct OwnedFont from mmap index for incremental reuse
    fn mmap_index_to_owned(&self, findex: &FontMmapIndex) -> OwnedFont {
        let d = self.data();
        let num_glyphs = findex.num_glyphs;
        let mut glyphs = Vec::with_capacity(num_glyphs);
        for gid in 0..num_glyphs {
            let off = findex.glyphs_off + gid*10;
            if off+10 > d.len() { break; }
            let adv = le_i16_at(d, off) as f64;
            let x0 = le_i16_at(d, off+2) as f64;
            let x1 = le_i16_at(d, off+4) as f64;
            let y0 = le_i16_at(d, off+6) as f64;
            let y1 = le_i16_at(d, off+8) as f64;
            glyphs.push(GlyphMetricsOwned { advance: adv, x_min: x0, x_max: x1, y_min: y0, y_max: y1 });
        }
        let mut cmap = Vec::with_capacity(findex.cmap_len);
        for i in 0..findex.cmap_len {
            let off = findex.cmap_off + i*6;
            if off+6 > d.len() { break; }
            let cp = le_u32_at(d, off);
            let gid = le_u16_at(d, off+4);
            cmap.push((cp,gid));
        }
        // singles
        let mut single_tables = Vec::with_capacity(findex.single_tables.len());
        for tbl in &findex.single_tables {
            let mut coverage = Vec::with_capacity(tbl.coverage_len);
            for i in 0..tbl.coverage_len {
                let off = tbl.coverage_off + i*2;
                if off+2 > d.len() { break; }
                coverage.push(le_u16_at(d, off));
            }
            let vf = tbl.value_format;
            let mut values = Vec::new();
            let count = if tbl.is_single { 1 } else { tbl.coverage_len };
            for i in 0..count {
                let off = tbl.values_off + i*tbl.stride*2;
                if off + tbl.stride*2 > d.len() { break; }
                let (vals, _) = unpack_vals_4(d, off, vf);
                values.push(vals);
            }
            // expand single value to coverage_len copies for Owned convenience if is_single
            if tbl.is_single && values.len()==1 && coverage.len()>1 {
                let v = values[0];
                values = vec![v; coverage.len()];
            }
            single_tables.push(SingleTableOwned { coverage, value_format: vf, values, is_single: tbl.is_single });
        }
        // format1
        let mut format1_tables = Vec::with_capacity(findex.format1_tables.len());
        for tbl in &findex.format1_tables {
            let mut coverage = Vec::with_capacity(tbl.coverage_len);
            for i in 0..tbl.coverage_len {
                let off = tbl.coverage_off + i*2;
                if off+2 > d.len() { break; }
                coverage.push(le_u16_at(d, off));
            }
            let vf1 = tbl.val_fmt1;
            let vf2 = tbl.val_fmt2;
            let sz1 = tbl.sz1;
            let sz2 = tbl.sz2;
            let rec_sz = 2 + sz1*2 + sz2*2;
            let mut pair_sets = Vec::with_capacity(tbl.coverage_len);
            for ps in &tbl.pair_sets {
                let mut pairs = Vec::with_capacity(ps.pair_count);
                for i in 0..ps.pair_count {
                    let off = ps.pairs_off + i*rec_sz;
                    if off+rec_sz > d.len() { break; }
                    let second = le_u16_at(d, off);
                    let mut p = off+2;
                    let mut v1=[0i16;4];
                    let mut v2=[0i16;4];
                    if vf1 & 0x0001 !=0 { v1[0]=le_i16_at(d,p); p+=2; }
                    if vf1 & 0x0002 !=0 { v1[1]=le_i16_at(d,p); p+=2; }
                    if vf1 & 0x0004 !=0 { v1[2]=le_i16_at(d,p); p+=2; }
                    if vf1 & 0x0008 !=0 { v1[3]=le_i16_at(d,p); p+=2; }
                    if vf2 & 0x0001 !=0 { v2[0]=le_i16_at(d,p); p+=2; }
                    if vf2 & 0x0002 !=0 { v2[1]=le_i16_at(d,p); p+=2; }
                    if vf2 & 0x0004 !=0 { v2[2]=le_i16_at(d,p); p+=2; }
                    if vf2 & 0x0008 !=0 { v2[3]=le_i16_at(d,p); }
                    pairs.push((second, v1, v2));
                }
                pair_sets.push(pairs);
            }
            format1_tables.push(Format1TableOwned { coverage, val_fmt1: vf1, val_fmt2: vf2, pair_sets });
        }
        // format2
        let mut format2_tables = Vec::with_capacity(findex.format2_tables.len());
        for tbl in &findex.format2_tables {
            let mut coverage = Vec::with_capacity(tbl.coverage_len);
            for i in 0..tbl.coverage_len {
                let off = tbl.coverage_off + i*2;
                if off+2 > d.len() { break; }
                coverage.push(le_u16_at(d, off));
            }
            // class defs
            let cd1 = match &tbl.class1_def {
                ClassDefIndex::Format1 { start, count, classes_off } => {
                    let mut classes = Vec::with_capacity(*count);
                    for i in 0..*count {
                        let off = *classes_off + i*2;
                        if off+2 > d.len() { break; }
                        classes.push(le_u16_at(d, off));
                    }
                    ClassDefOwned::Format1 { start: *start, classes }
                }
                ClassDefIndex::Format2 { count, ranges_off } => {
                    let mut ranges = Vec::with_capacity(*count);
                    for i in 0..*count {
                        let off = *ranges_off + i*6;
                        if off+6 > d.len() { break; }
                        let s=le_u16_at(d, off);
                        let e=le_u16_at(d, off+2);
                        let c=le_u16_at(d, off+4);
                        ranges.push((s,e,c));
                    }
                    ClassDefOwned::Format2 { ranges }
                }
            };
            let cd2 = match &tbl.class2_def {
                ClassDefIndex::Format1 { start, count, classes_off } => {
                    let mut classes = Vec::with_capacity(*count);
                    for i in 0..*count {
                        let off = *classes_off + i*2;
                        if off+2 > d.len() { break; }
                        classes.push(le_u16_at(d, off));
                    }
                    ClassDefOwned::Format1 { start: *start, classes }
                }
                ClassDefIndex::Format2 { count, ranges_off } => {
                    let mut ranges = Vec::with_capacity(*count);
                    for i in 0..*count {
                        let off = *ranges_off + i*6;
                        if off+6 > d.len() { break; }
                        let s=le_u16_at(d, off);
                        let e=le_u16_at(d, off+2);
                        let c=le_u16_at(d, off+4);
                        ranges.push((s,e,c));
                    }
                    ClassDefOwned::Format2 { ranges }
                }
            };
            let vf1 = tbl.val_fmt1;
            let vf2 = tbl.val_fmt2;
            let mut matrix = Vec::with_capacity(tbl.class1_count*tbl.class2_count);
            let rec_sz = (tbl.sz1 + tbl.sz2)*2;
            for i in 0..tbl.class1_count*tbl.class2_count {
                let off = tbl.matrix_off + i*rec_sz;
                if off+rec_sz > d.len() { break; }
                let mut p = off;
                let mut v1=[0i16;4];
                let mut v2=[0i16;4];
                if vf1 & 0x0001 !=0 { v1[0]=le_i16_at(d,p); p+=2; }
                if vf1 & 0x0002 !=0 { v1[1]=le_i16_at(d,p); p+=2; }
                if vf1 & 0x0004 !=0 { v1[2]=le_i16_at(d,p); p+=2; }
                if vf1 & 0x0008 !=0 { v1[3]=le_i16_at(d,p); p+=2; }
                if vf2 & 0x0001 !=0 { v2[0]=le_i16_at(d,p); p+=2; }
                if vf2 & 0x0002 !=0 { v2[1]=le_i16_at(d,p); p+=2; }
                if vf2 & 0x0004 !=0 { v2[2]=le_i16_at(d,p); p+=2; }
                if vf2 & 0x0008 !=0 { v2[3]=le_i16_at(d,p); }
                matrix.push((v1,v2));
            }
            format2_tables.push(Format2TableOwned {
                coverage,
                class_def1: cd1,
                class_def2: cd2,
                class1_count: tbl.class1_count,
                class2_count: tbl.class2_count,
                val_fmt1: vf1,
                val_fmt2: vf2,
                matrix,
            });
        }

        OwnedFont {
            file_hash: findex.file_hash,
            units_per_em: findex.upem,
            num_glyphs,
            glyphs,
            cmap,
            single_tables,
            format1_tables,
            format2_tables,
        }
    }

    pub fn load_or_build(font_registry: &crate::font_scan::FontRegistry, font_cache: &crate::font_cache::FontCache) -> Self {
        let cache_path = Self::default_path();
        let catalog_hash = font_registry.catalog_hash();

        // Fast path: if cache exists and catalog hash matches, reuse directly (no cloning)
        if cache_path.exists() {
            match Self::load(&cache_path) {
                Ok((old_cache, old_catalog_hash)) => {
                    if old_catalog_hash == catalog_hash {
                        eprintln!("[geo-cache] Reusing valid v{} cache with {} fonts (hash match)", BGEO_VERSION, old_cache.fonts.len());
                        return old_cache;
                    }
                    // Hash mismatch — incremental reuse but keep old_cache for reuse
                    let mut old_owned: HashMap<String, OwnedFont> = HashMap::new();
                    for (k, idx) in &old_cache.fonts {
                        let owned = old_cache.mmap_index_to_owned(idx);
                        old_owned.insert(k.clone(), owned);
                    }
                    eprintln!("[geo-cache] Loaded old v{} cache with {} fonts for incremental reuse (hash mismatch)", BGEO_VERSION, old_owned.len());
                    return Self::build_incremental(font_registry, font_cache, catalog_hash, cache_path, old_owned);
                }
                Err(e) => {
                    eprintln!("[geo-cache] Could not load old cache for reuse ({}), full rebuild", e);
                }
            }
        }
        // No usable cache — full rebuild
        Self::build_incremental(font_registry, font_cache, catalog_hash, cache_path, HashMap::new())
    }

    fn build_incremental(
        font_registry: &crate::font_scan::FontRegistry,
        font_cache: &crate::font_cache::FontCache,
        catalog_hash: u64,
        cache_path: std::path::PathBuf,
        old_owned: HashMap<String, OwnedFont>,
    ) -> Self {

        let mut owned_fonts: HashMap<String, OwnedFont> = HashMap::new();
        let mut n_total = 0usize;
        let mut n_reused = 0usize;
        let mut n_built = 0usize;

        for fe in font_registry.iter() {
            n_total += 1;
            let font_key = fe.font_key();
            let font_data = match font_cache.load(&fe.path) { Ok(d) => d, Err(_) => continue, };
            let fhash = file_hash_bytes(&font_data);

            if let Some(old) = old_owned.get(&font_key) {
                if old.file_hash == fhash {
                    owned_fonts.insert(font_key.clone(), OwnedFont {
                        file_hash: fhash,
                        units_per_em: old.units_per_em,
                        num_glyphs: old.num_glyphs,
                        glyphs: old.glyphs.clone(),
                        cmap: old.cmap.clone(),
                        single_tables: old.single_tables.clone(),
                        format1_tables: old.format1_tables.clone(),
                        format2_tables: old.format2_tables.clone(),
                    });
                    n_reused += 1;
                    continue;
                }
            }

            let ttf_face = match ttf_parser::Face::parse(&font_data, 0) { Ok(f) => f, Err(_) => continue, };
            let upem = ttf_face.units_per_em() as f64;
            let num_glyphs = ttf_face.number_of_glyphs() as usize;

            let mut glyphs = Vec::with_capacity(num_glyphs);
            for gid in 0..num_glyphs {
                let gid_t = ttf_parser::GlyphId(gid as u16);
                let adv = ttf_face.glyph_hor_advance(gid_t).unwrap_or(0) as f64;
                let bbox = ttf_face.glyph_bounding_box(gid_t);
                let (x_min, x_max, y_min, y_max) = if let Some(bb) = bbox {
                    (bb.x_min as f64, bb.x_max as f64, bb.y_min as f64, bb.y_max as f64)
                } else { (0.0, adv, 0.0, 0.0) };
                glyphs.push(GlyphMetricsOwned { advance: adv, x_min, x_max, y_min, y_max });
            }

            let mut cmap_map: HashMap<u32, u16> = HashMap::new();
            if let Some(cmap) = ttf_face.tables().cmap.as_ref() {
                for subtable in cmap.subtables {
                    if !subtable.is_unicode() { continue; }
                    subtable.codepoints(|cp| {
                        if let Some(gid) = subtable.glyph_index(cp) {
                            if gid.0 != 0 { cmap_map.entry(cp).or_insert(gid.0); }
                        }
                    });
                }
            } else {
                for cp in 0x20u32..=0x7Eu32 {
                    if let Some(ch) = char::from_u32(cp) {
                        if let Some(gid) = ttf_face.glyph_index(ch) { cmap_map.insert(cp, gid.0); }
                    }
                }
                for &lig in &['\u{FB00}','\u{FB01}','\u{FB02}','\u{FB03}','\u{FB04}'] {
                    if let Some(gid) = ttf_face.glyph_index(lig) { cmap_map.insert(lig as u32, gid.0); }
                }
            }
            let mut cmap: Vec<(u32,u16)> = cmap_map.into_iter().collect();
            cmap.sort_by_key(|(cp,_)| *cp);

            let mut single_tables: Vec<SingleTableOwned> = Vec::new();
            let mut format1_tables: Vec<Format1TableOwned> = Vec::new();
            let mut format2_tables: Vec<Format2TableOwned> = Vec::new();

            if let Some(gpos_data) = ttf_face.raw_face().table(ttf_parser::Tag::from_bytes(b"GPOS")) {
                if let Some((s_tbls, f1_tbls, f2_tbls)) = Self::parse_gpos_full(gpos_data, num_glyphs) {
                    single_tables = s_tbls;
                    format1_tables = f1_tbls;
                    format2_tables = f2_tbls;
                }
            }

            if single_tables.is_empty() && format1_tables.is_empty() && format2_tables.is_empty() {
                if let Some(kern_tbl) = ttf_face.tables().kern.as_ref() {
                    for subtable in kern_tbl.subtables.clone().into_iter() {
                        if !subtable.horizontal || subtable.has_cross_stream || subtable.variable || subtable.has_state_machine { continue; }
                        let gids: Vec<u16> = cmap.iter().map(|(_, gid)| *gid).collect();
                        let mut uniq_gids = gids.clone(); uniq_gids.sort_unstable(); uniq_gids.dedup();
                        let mut map: BTreeMap<u16, Vec<(u16, Val4, Val4)>> = BTreeMap::new();
                        for &g1 in &uniq_gids {
                            for &g2 in &uniq_gids {
                                if let Some(k) = subtable.glyphs_kerning(ttf_parser::GlyphId(g1), ttf_parser::GlyphId(g2)) {
                                    if k!=0 {
                                        let v1 = [0,0,k,0];
                                        let v2 = [0,0,0,0];
                                        map.entry(g1).or_default().push((g2, v1, v2));
                                    }
                                }
                            }
                        }
                        if !map.is_empty() {
                            let mut coverage = Vec::with_capacity(map.len());
                            let mut pair_sets = Vec::with_capacity(map.len());
                            for (left, mut pairs) in map {
                                pairs.sort_by_key(|(r,_,_)| *r);
                                coverage.push(left);
                                pair_sets.push(pairs);
                            }
                            format1_tables.push(Format1TableOwned { coverage, val_fmt1: 0x0004, val_fmt2: 0x0000, pair_sets });
                            break;
                        }
                    }
                }
            }

            owned_fonts.insert(font_key, OwnedFont { file_hash: fhash, units_per_em: upem, num_glyphs, glyphs, cmap, single_tables, format1_tables, format2_tables });
            n_built+=1;
        }

        eprintln!("[geo-cache] Built v{} cache ({} total, {} reused, {} built, GPOS full incl singles, Unicode both formats)", BGEO_VERSION, n_total, n_reused, n_built);
        let tmp_cache = Self { mmap: unsafe { memmap2::MmapOptions::new().len(0).map_anon().unwrap().make_read_only().unwrap() }, fonts: HashMap::new(), _cache_path: cache_path.clone() };
        if let Err(e) = tmp_cache.write_bin_from_owned(&cache_path, catalog_hash, &owned_fonts) {
            eprintln!("warning: failed to write geo cache to {}: {e}", cache_path.display());
        } else {
            eprintln!("[geo-cache] Wrote {} fonts to {}", owned_fonts.len(), cache_path.display());
        }
        match Self::load(&cache_path) {
            Ok((mmap_cache,_)) => mmap_cache,
            Err(e) => {
                eprintln!("failed to reload just-written cache: {e}, returning empty");
                let empty_mmap = unsafe { memmap2::MmapOptions::new().len(0).map_anon().unwrap().make_read_only().unwrap() };
                Self { mmap: empty_mmap, fonts: HashMap::new(), _cache_path: cache_path }
            }
        }
    }

    fn write_bin_from_owned(&self, path: &Path, catalog_hash: u64, owned_fonts: &HashMap<String, OwnedFont>) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
        let tmp = crate::atomic_file::tmp_for(path);
        let f = std::fs::File::create(&tmp)?;
        let mut w = BufWriter::new(f);
        w.write_all(BGEO_MAGIC)?;
        w.write_all(&BGEO_VERSION.to_le_bytes())?;
        w.write_all(&catalog_hash.to_le_bytes())?;
        w.write_all(&(owned_fonts.len() as u32).to_le_bytes())?;
        let mut entries: Vec<_> = owned_fonts.iter().collect();
        entries.sort_by(|(a,_),(b,_)| a.cmp(b));
        for (font_key, of) in entries {
            let kb = font_key.as_bytes();
            w.write_all(&(kb.len() as u32).to_le_bytes())?; w.write_all(kb)?;
            w.write_all(&of.file_hash.to_le_bytes())?;
            w.write_all(&(of.units_per_em as u16).to_le_bytes())?;
            w.write_all(&(of.num_glyphs as u16).to_le_bytes())?;
            for gm in &of.glyphs {
                let adv = gm.advance.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                let x0 = gm.x_min.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                let x1 = gm.x_max.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                let y0 = gm.y_min.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                let y1 = gm.y_max.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                w.write_all(&adv.to_le_bytes())?; w.write_all(&x0.to_le_bytes())?; w.write_all(&x1.to_le_bytes())?; w.write_all(&y0.to_le_bytes())?; w.write_all(&y1.to_le_bytes())?;
            }
            w.write_all(&(of.cmap.len() as u32).to_le_bytes())?;
            for (cp,gid) in &of.cmap { w.write_all(&cp.to_le_bytes())?; w.write_all(&gid.to_le_bytes())?; }
            w.write_all(&(of.single_tables.len() as u16).to_le_bytes())?;
            w.write_all(&(of.format1_tables.len() as u16).to_le_bytes())?;
            w.write_all(&(of.format2_tables.len() as u16).to_le_bytes())?;
            // singles
            for tbl in &of.single_tables {
                w.write_all(&(tbl.coverage.len() as u16).to_le_bytes())?;
                for gid in &tbl.coverage { w.write_all(&gid.to_le_bytes())?; }
                w.write_all(&tbl.value_format.to_le_bytes())?;
                w.write_all(&[if tbl.is_single {1u8} else {0u8}])?;
                let cnt = if tbl.is_single { 1 } else { tbl.coverage.len() };
                for i in 0..cnt {
                    let v = if i < tbl.values.len() { &tbl.values[i] } else { &[0,0,0,0] };
                    pack_vals_4(&mut w, v, tbl.value_format)?;
                }
            }
            // format1
            for tbl in &of.format1_tables {
                w.write_all(&(tbl.coverage.len() as u16).to_le_bytes())?;
                for gid in &tbl.coverage { w.write_all(&gid.to_le_bytes())?; }
                w.write_all(&tbl.val_fmt1.to_le_bytes())?;
                w.write_all(&tbl.val_fmt2.to_le_bytes())?;
                for ps in &tbl.pair_sets {
                    w.write_all(&(ps.len() as u16).to_le_bytes())?;
                    for (sg, v1, v2) in ps {
                        w.write_all(&sg.to_le_bytes())?;
                        pack_vals_4(&mut w, v1, tbl.val_fmt1)?;
                        pack_vals_4(&mut w, v2, tbl.val_fmt2)?;
                    }
                }
            }
            // format2
            for tbl in &of.format2_tables {
                w.write_all(&(tbl.coverage.len() as u16).to_le_bytes())?;
                for gid in &tbl.coverage { w.write_all(&gid.to_le_bytes())?; }
                w.write_all(&(tbl.class1_count as u16).to_le_bytes())?;
                w.write_all(&(tbl.class2_count as u16).to_le_bytes())?;
                w.write_all(&tbl.val_fmt1.to_le_bytes())?;
                w.write_all(&tbl.val_fmt2.to_le_bytes())?;
                match &tbl.class_def1 {
                    ClassDefOwned::Format1 { start, classes } => {
                        w.write_all(&[1u8,0])?; w.write_all(&(classes.len() as u16).to_le_bytes())?;
                        w.write_all(&start.to_le_bytes())?;
                        for c in classes { w.write_all(&c.to_le_bytes())?; }
                    }
                    ClassDefOwned::Format2 { ranges } => {
                        w.write_all(&[2u8,0])?; w.write_all(&(ranges.len() as u16).to_le_bytes())?;
                        w.write_all(&[0,0])?;
                        for (s,e,c) in ranges { w.write_all(&s.to_le_bytes())?; w.write_all(&e.to_le_bytes())?; w.write_all(&c.to_le_bytes())?; }
                    }
                }
                match &tbl.class_def2 {
                    ClassDefOwned::Format1 { start, classes } => {
                        w.write_all(&[1u8,0])?; w.write_all(&(classes.len() as u16).to_le_bytes())?;
                        w.write_all(&start.to_le_bytes())?;
                        for c in classes { w.write_all(&c.to_le_bytes())?; }
                    }
                    ClassDefOwned::Format2 { ranges } => {
                        w.write_all(&[2u8,0])?; w.write_all(&(ranges.len() as u16).to_le_bytes())?;
                        w.write_all(&[0,0])?;
                        for (s,e,c) in ranges { w.write_all(&s.to_le_bytes())?; w.write_all(&e.to_le_bytes())?; w.write_all(&c.to_le_bytes())?; }
                    }
                }
                for (v1,v2) in &tbl.matrix {
                    pack_vals_4(&mut w, v1, tbl.val_fmt1)?;
                    pack_vals_4(&mut w, v2, tbl.val_fmt2)?;
                }
            }
        }
        w.flush()?; drop(w); std::fs::rename(&tmp, path)?; Ok(())
    }

    pub fn load(path: &Path) -> Result<(Self,u64),String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
        let data = &mmap[..];
        if data.len() < 20 { return Err("BGEO too small".into()); }
        if &data[0..4] != BGEO_MAGIC { return Err(format!("bad magic {:?}", &data[0..4])); }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != BGEO_VERSION { return Err(format!("BGEO version {version}, need v{BGEO_VERSION}")); }
        let catalog_hash = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let n_fonts = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let mut pos = 20usize;
        let mut fonts = HashMap::with_capacity(n_fonts);
        for _ in 0..n_fonts {
            let klen = read_u32_le(data, &mut pos)? as usize;
            if pos+klen > data.len() { return Err("trunc font_key".into()); }
            let font_key = String::from_utf8(data[pos..pos+klen].to_vec()).map_err(|e| format!("invalid key {e}"))?; pos+=klen;
            let file_hash = read_u64_le(data, &mut pos)?;
            if pos+4 > data.len() { return Err("trunc upem/num_glyphs".into()); }
            let upem = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as f64; pos+=2;
            let num_glyphs = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
            let glyphs_off = pos;
            let glyphs_bytes = num_glyphs*10;
            if pos+glyphs_bytes > data.len() { return Err("trunc glyphs".into()); }
            pos+=glyphs_bytes;
            let cmap_len = read_u32_le(data, &mut pos)? as usize;
            let cmap_off = pos;
            let cmap_bytes = cmap_len*6;
            if pos+cmap_bytes > data.len() { return Err("trunc cmap".into()); }
            pos+=cmap_bytes;
            if pos+6 > data.len() { return Err("trunc n_single/n_f1/n_f2".into()); }
            let n_single = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
            let n_f1 = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
            let n_f2 = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;

            let mut single_tables = Vec::with_capacity(n_single);
            for _ in 0..n_single {
                if pos+2 > data.len() { return Err("trunc single cov_len".into()); }
                let cov_len = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                let cov_off = pos;
                if pos+cov_len*2 > data.len() { return Err("trunc single coverage".into()); }
                pos+=cov_len*2;
                if pos+2 > data.len() { return Err("trunc single vf".into()); }
                let vf = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()); pos+=2;
                if pos+1 > data.len() { return Err("trunc single is_single".into()); }
                let is_single = data[pos]!=0; pos+=1;
                let sz = popcnt4(vf);
                let values_off = pos;
                let cnt = if is_single { 1 } else { cov_len };
                let bytes = cnt*sz*2;
                if pos+bytes > data.len() { return Err("trunc single values".into()); }
                pos+=bytes;
                single_tables.push(SingleIndex { coverage_off: cov_off, coverage_len: cov_len, value_format: vf, is_single, values_off, stride: sz });
            }

            let mut f1_tables = Vec::with_capacity(n_f1);
            for _ in 0..n_f1 {
                if pos+2 > data.len() { return Err("trunc f1 cov_len".into()); }
                let cov_len = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                let cov_off = pos;
                if pos+cov_len*2 > data.len() { return Err("trunc f1 coverage".into()); }
                pos+=cov_len*2;
                if pos+4 > data.len() { return Err("trunc f1 vf1/vf2".into()); }
                let vf1 = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()); pos+=2;
                let vf2 = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()); pos+=2;
                let sz1 = popcnt4(vf1);
                let sz2 = popcnt4(vf2);
                let rec_sz = 2 + sz1*2 + sz2*2;
                let mut pair_sets = Vec::with_capacity(cov_len);
                for _ in 0..cov_len {
                    if pos+2 > data.len() { return Err("trunc f1 pair_count".into()); }
                    let pc = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                    let pairs_off = pos;
                    let bytes = pc*rec_sz;
                    if pos+bytes > data.len() { return Err("trunc f1 pairs".into()); }
                    pos+=bytes;
                    pair_sets.push(PairSetIndex { pairs_off, pair_count: pc });
                }
                f1_tables.push(Format1Index { coverage_off: cov_off, coverage_len: cov_len, val_fmt1: vf1, val_fmt2: vf2, sz1, sz2, pair_sets });
            }

            let mut f2_tables = Vec::with_capacity(n_f2);
            for _ in 0..n_f2 {
                if pos+2 > data.len() { return Err("trunc f2 cov_len".into()); }
                let cov_len = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                let cov_off = pos;
                if pos+cov_len*2 > data.len() { return Err("trunc f2 coverage".into()); }
                pos+=cov_len*2;
                if pos+4 > data.len() { return Err("trunc f2 c1c/c2c".into()); }
                let c1c = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                let c2c = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                if pos+4 > data.len() { return Err("trunc f2 vf1/vf2".into()); }
                let vf1 = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                let vf2 = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                let vf1u = vf1 as u16;
                let vf2u = vf2 as u16;
                let sz1 = popcnt4(vf1u);
                let sz2 = popcnt4(vf2u);
                // cd1
                if pos+2 > data.len() { return Err("trunc f2 cd1 fmt".into()); }
                let fmt1 = data[pos]; pos+=2; // fmt + pad
                if pos+2 > data.len() { return Err("trunc f2 cd1 cnt".into()); }
                let cd1_cnt = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                if pos+2 > data.len() { return Err("trunc f2 cd1 start".into()); }
                let cd1_start = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()); pos+=2;
                let cd1 = if fmt1==1 {
                    let classes_off = pos;
                    if pos+cd1_cnt*2 > data.len() { return Err("trunc f2 cd1 classes".into()); }
                    pos+=cd1_cnt*2;
                    ClassDefIndex::Format1 { start: cd1_start, count: cd1_cnt, classes_off }
                } else {
                    let ranges_off = pos;
                    if pos+cd1_cnt*6 > data.len() { return Err("trunc f2 cd1 ranges".into()); }
                    pos+=cd1_cnt*6;
                    ClassDefIndex::Format2 { count: cd1_cnt, ranges_off }
                };
                // cd2
                if pos+2 > data.len() { return Err("trunc f2 cd2 fmt".into()); }
                let fmt2 = data[pos]; pos+=2;
                if pos+2 > data.len() { return Err("trunc f2 cd2 cnt".into()); }
                let cd2_cnt = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()) as usize; pos+=2;
                if pos+2 > data.len() { return Err("trunc f2 cd2 start".into()); }
                let cd2_start = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap()); pos+=2;
                let cd2 = if fmt2==1 {
                    let classes_off = pos;
                    if pos+cd2_cnt*2 > data.len() { return Err("trunc f2 cd2 classes".into()); }
                    pos+=cd2_cnt*2;
                    ClassDefIndex::Format1 { start: cd2_start, count: cd2_cnt, classes_off }
                } else {
                    let ranges_off = pos;
                    if pos+cd2_cnt*6 > data.len() { return Err("trunc f2 cd2 ranges".into()); }
                    pos+=cd2_cnt*6;
                    ClassDefIndex::Format2 { count: cd2_cnt, ranges_off }
                };
                let matrix_off = pos;
                let matrix_bytes = c1c*c2c*(sz1+sz2)*2;
                if pos+matrix_bytes > data.len() { return Err("trunc f2 matrix".into()); }
                pos+=matrix_bytes;
                f2_tables.push(Format2Index { coverage_off: cov_off, coverage_len: cov_len, class1_count: c1c, class2_count: c2c, val_fmt1: vf1u, val_fmt2: vf2u, sz1, sz2, class1_def: cd1, class2_def: cd2, matrix_off });
            }

            fonts.insert(font_key, FontMmapIndex { file_hash, upem, num_glyphs, glyphs_off, cmap_off, cmap_len, single_tables, format1_tables: f1_tables, format2_tables: f2_tables });
        }
        Ok((Self { mmap, fonts, _cache_path: path.to_path_buf() }, catalog_hash))
    }

    // compatibility for tests - returns old kern-only view
    pub fn test_parse_gpos(data: &[u8], num_glyphs: usize) -> Option<(Vec<Format1Parse>, Vec<Format2Parse>, bool)> {
        // call new full parser, then convert to old structs for display
        let (singles, f1_full, f2_full) = Self::parse_gpos_full(data, num_glyphs)?;
        // convert f1
        let mut f1_old = Vec::with_capacity(f1_full.len());
        for tbl in f1_full {
            let mut pair_sets = Vec::with_capacity(tbl.pair_sets.len());
            for ps in tbl.pair_sets {
                let mut pairs = Vec::with_capacity(ps.len());
                for (gid, v1, v2) in ps {
                    let kern = (v1[2] as i32 + v2[2] as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    if kern !=0 { pairs.push((gid, kern)); }
                }
                pair_sets.push(pairs);
            }
            f1_old.push(Format1Parse { coverage: tbl.coverage, pair_sets });
        }
        let mut f2_old = Vec::with_capacity(f2_full.len());
        for tbl in f2_full {
            let matrix = tbl.matrix.iter().map(|(v1,v2)| {
                let k = (v1[2] as i32 + v2[2] as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                k
            }).collect();
            let cd1 = match tbl.class_def1 {
                ClassDefOwned::Format1{start,classes} => ClassDefParse::Format1{start,classes},
                ClassDefOwned::Format2{ranges} => ClassDefParse::Format2{ranges},
            };
            let cd2 = match tbl.class_def2 {
                ClassDefOwned::Format1{start,classes} => ClassDefParse::Format1{start,classes},
                ClassDefOwned::Format2{ranges} => ClassDefParse::Format2{ranges},
            };
            f2_old.push(Format2Parse { coverage: tbl.coverage, class_def1: cd1, class_def2: cd2, class1_count: tbl.class1_count, class2_count: tbl.class2_count, matrix });
        }
        let _ = singles; // unused in old view
        Some((f1_old, f2_old, false))
    }

    fn parse_gpos_full(data: &[u8], num_glyphs: usize) -> Option<(Vec<SingleTableOwned>, Vec<Format1TableOwned>, Vec<Format2TableOwned>)> {
        Self::parse_gpos_inner(data, num_glyphs)
    }

    // old wrapper kept for internal callers that expected old signature - now maps to full
    fn parse_gpos(_data: &[u8], _num_glyphs: usize) -> Option<(Vec<Format1Parse>, Vec<Format2Parse>, bool)> {
        None
    }

    fn parse_gpos_inner(data: &[u8], num_glyphs: usize) -> Option<(Vec<SingleTableOwned>, Vec<Format1TableOwned>, Vec<Format2TableOwned>)> {
        if data.len() < 10 { return None; }
        let lookup_list_off = be_u16(data, 8)? as usize;
        if lookup_list_off + 2 > data.len() { return None; }
        let lookup_count = be_u16(data, lookup_list_off)? as usize;
        let mut single_tables = Vec::new();
        let mut f1_tables = Vec::new();
        let mut f2_tables = Vec::new();
        for li in 0..lookup_count {
            let off_pos = lookup_list_off + 2 + li*2;
            let lookup_off = be_u16(data, off_pos)? as usize + lookup_list_off;
            if lookup_off + 6 > data.len() { continue; }
            let lookup_type = be_u16(data, lookup_off)?;
            let subtable_count = be_u16(data, lookup_off+4)? as usize;
            for si in 0..subtable_count {
                let sub_off_pos = lookup_off + 6 + si*2;
                let sub_off = be_u16(data, sub_off_pos)? as usize + lookup_off;
                if sub_off + 2 > data.len() { continue; }
                if lookup_type == 1 {
                    // SinglePos
                    let fmt = be_u16(data, sub_off)?;
                    if fmt == 1 {
                        // SinglePos Format1: CoverageOffset, ValueFormat, ValueRecord (single)
                        if sub_off + 6 > data.len() { continue; }
                        let cov_off = be_u16(data, sub_off+2)? as usize + sub_off;
                        let val_fmt_full = be_u16(data, sub_off+4)?;
                        let val_fmt = vf_mask(val_fmt_full);
                        let coverage = Self::parse_coverage(data, cov_off)?;
                        let (xpl,ypl,xad,yad,_) = Self::parse_value_record(data, sub_off+6, val_fmt_full)?;
                        let val = [xpl,ypl,xad,yad];
                        single_tables.push(SingleTableOwned { coverage: coverage.clone(), value_format: val_fmt, values: vec![val; coverage.len()], is_single: true });
                    } else if fmt == 2 {
                        if sub_off + 8 > data.len() { continue; }
                        let cov_off = be_u16(data, sub_off+2)? as usize + sub_off;
                        let val_fmt_full = be_u16(data, sub_off+4)?;
                        let val_fmt = vf_mask(val_fmt_full);
                        let val_count = be_u16(data, sub_off+6)? as usize;
                        let coverage = Self::parse_coverage(data, cov_off)?;
                        let mut values = Vec::with_capacity(val_count);
                        let mut p = sub_off + 8;
                        let val_size = Self::value_format_size(val_fmt_full);
                        for _ in 0..val_count {
                            if p + val_size > data.len() { break; }
                            let (xpl,ypl,xad,yad,_) = Self::parse_value_record(data, p, val_fmt_full)?;
                            values.push([xpl,ypl,xad,yad]);
                            p += val_size;
                        }
                        single_tables.push(SingleTableOwned { coverage, value_format: val_fmt, values, is_single: false });
                    }
                } else if lookup_type == 2 {
                    let fmt = be_u16(data, sub_off)?;
                    if fmt == 1 {
                        if sub_off + 10 > data.len() { continue; }
                        let cov_off = be_u16(data, sub_off+2)? as usize + sub_off;
                        let val_fmt1_full = be_u16(data, sub_off+4)?;
                        let val_fmt2_full = be_u16(data, sub_off+6)?;
                        let val_fmt1 = vf_mask(val_fmt1_full);
                        let val_fmt2 = vf_mask(val_fmt2_full);
                        let pair_set_count = be_u16(data, sub_off+8)? as usize;
                        let coverage = Self::parse_coverage(data, cov_off)?;
                        let mut pair_sets: Vec<Vec<(u16,Val4,Val4)>> = Vec::with_capacity(pair_set_count);
                        for psi in 0..pair_set_count {
                            let ps_off_pos = sub_off + 10 + psi*2;
                            let ps_off = be_u16(data, ps_off_pos)? as usize + sub_off;
                            if ps_off + 2 > data.len() { pair_sets.push(Vec::new()); continue; }
                            let pair_val_count = be_u16(data, ps_off)? as usize;
                            let mut pairs = Vec::with_capacity(pair_val_count);
                            let val1_size = Self::value_format_size(val_fmt1_full);
                            let val2_size = Self::value_format_size(val_fmt2_full);
                            let rec_size = 2 + val1_size + val2_size;
                            let mut rp = ps_off + 2;
                            for _ in 0..pair_val_count {
                                if rp + rec_size > data.len() { break; }
                                let second_gid = be_u16(data, rp)?;
                                let (xpl1,ypl1,xad1,yad1,_) = Self::parse_value_record(data, rp+2, val_fmt1_full)?;
                                let (xpl2,ypl2,xad2,yad2,_) = Self::parse_value_record(data, rp+2+val1_size, val_fmt2_full)?;
                                let v1 = [xpl1,ypl1,xad1,yad1];
                                let v2 = [xpl2,ypl2,xad2,yad2];
                                // keep even if all zero? keep non-zero to save space, but keep all for correctness
                                if v1!=[0,0,0,0] || v2!=[0,0,0,0] {
                                    pairs.push((second_gid, v1, v2));
                                } else {
                                    // still need to keep zero kern? previous code dropped zero kern, we keep zero as well? To reduce size we can drop pure zero, but then pair search will miss zero (which is fine). Keep drop-zero to save space.
                                }
                                rp += rec_size;
                            }
                            pair_sets.push(pairs);
                        }
                        // pad pair_sets to coverage len if needed (should be equal)
                        while pair_sets.len() < coverage.len() { pair_sets.push(Vec::new()); }
                        f1_tables.push(Format1TableOwned { coverage, val_fmt1, val_fmt2, pair_sets });
                    } else if fmt == 2 {
                        if sub_off + 16 > data.len() { continue; }
                        let cov_off = be_u16(data, sub_off+2)? as usize + sub_off;
                        let val_fmt1_full = be_u16(data, sub_off+4)?;
                        let val_fmt2_full = be_u16(data, sub_off+6)?;
                        let val_fmt1 = vf_mask(val_fmt1_full);
                        let val_fmt2 = vf_mask(val_fmt2_full);
                        let cd1_off = be_u16(data, sub_off+8)? as usize + sub_off;
                        let cd2_off = be_u16(data, sub_off+10)? as usize + sub_off;
                        let c1_count = be_u16(data, sub_off+12)? as usize;
                        let c2_count = be_u16(data, sub_off+14)? as usize;
                        let coverage = Self::parse_coverage(data, cov_off)?;
                        let (cd1, cd1_parse) = Self::parse_class_def_with_raw(data, cd1_off, num_glyphs)?;
                        let (cd2, cd2_parse) = Self::parse_class_def_with_raw(data, cd2_off, num_glyphs)?;
                        // we need owned versions of class defs for later write
                        let cd1_owned = match cd1_parse {
                            ClassDefParse::Format1{start,classes} => ClassDefOwned::Format1{start,classes},
                            ClassDefParse::Format2{ranges} => ClassDefOwned::Format2{ranges},
                        };
                        let cd2_owned = match cd2_parse {
                            ClassDefParse::Format1{start,classes} => ClassDefOwned::Format1{start,classes},
                            ClassDefParse::Format2{ranges} => ClassDefOwned::Format2{ranges},
                        };
                        let _ = (cd1, cd2); // keep for count
                        let val1_size = Self::value_format_size(val_fmt1_full);
                        let val2_size = Self::value_format_size(val_fmt2_full);
                        let rec_size = val1_size + val2_size;
                        let mut matrix = Vec::with_capacity(c1_count * c2_count);
                        let mut p = sub_off + 16;
                        for _ in 0..c1_count*c2_count {
                            if p + rec_size > data.len() { break; }
                            let (xpl1,ypl1,xad1,yad1,_) = Self::parse_value_record(data, p, val_fmt1_full)?;
                            let (xpl2,ypl2,xad2,yad2,_) = Self::parse_value_record(data, p+val1_size, val_fmt2_full)?;
                            matrix.push(([xpl1,ypl1,xad1,yad1],[xpl2,ypl2,xad2,yad2]));
                            p += rec_size;
                        }
                        f2_tables.push(Format2TableOwned { coverage, class_def1: cd1_owned, class_def2: cd2_owned, class1_count: c1_count, class2_count: c2_count, val_fmt1, val_fmt2, matrix });
                    }
                }
            }
        }
        Some((single_tables, f1_tables, f2_tables))
    }

    fn value_format_size(fmt: u16) -> usize {
        let mut sz = 0usize;
        if fmt & 0x0001 != 0 { sz += 2; }
        if fmt & 0x0002 != 0 { sz += 2; }
        if fmt & 0x0004 != 0 { sz += 2; }
        if fmt & 0x0008 != 0 { sz += 2; }
        if fmt & 0x0010 != 0 { sz += 2; }
        if fmt & 0x0020 != 0 { sz += 2; }
        if fmt & 0x0040 != 0 { sz += 2; }
        if fmt & 0x0080 != 0 { sz += 2; }
        sz
    }

    fn parse_value_record(data: &[u8], off: usize, fmt: u16) -> Option<(i16,i16,i16,i16,bool)> {
        let mut p = off;
        let mut xpl = 0i16; let mut ypl = 0i16; let mut xad = 0i16; let mut yad = 0i16;
        let mut has_dev = false;
        if fmt & 0x0001 != 0 { xpl = be_i16(data, p)?; p+=2; }
        if fmt & 0x0002 != 0 { ypl = be_i16(data, p)?; p+=2; }
        if fmt & 0x0004 != 0 { xad = be_i16(data, p)?; p+=2; }
        if fmt & 0x0008 != 0 { yad = be_i16(data, p)?; p+=2; }
        if fmt & 0x0010 != 0 { has_dev = true; p+=2; }
        if fmt & 0x0020 != 0 { has_dev = true; p+=2; }
        if fmt & 0x0040 != 0 { has_dev = true; p+=2; }
        if fmt & 0x0080 != 0 { has_dev = true; p+=2; }
        Some((xpl,ypl,xad,yad,has_dev))
    }

    fn parse_coverage(data: &[u8], off: usize) -> Option<Vec<u16>> {
        if off + 2 > data.len() { return None; }
        let fmt = be_u16(data, off)?;
        if fmt == 1 {
            let count = be_u16(data, off+2)? as usize;
            if off + 4 + count*2 > data.len() { return None; }
            let mut v = Vec::with_capacity(count);
            for i in 0..count { v.push(be_u16(data, off+4+i*2)?); }
            Some(v)
        } else if fmt == 2 {
            let range_count = be_u16(data, off+2)? as usize;
            let mut v = Vec::new();
            for i in 0..range_count {
                let base = off+4+i*6;
                if base+6 > data.len() { return None; }
                let start = be_u16(data, base)?;
                let end = be_u16(data, base+2)?;
                for gid in start..=end { v.push(gid); }
            }
            v.sort_unstable(); v.dedup();
            Some(v)
        } else { None }
    }

    fn parse_class_def(data: &[u8], off: usize, _num_glyphs: usize) -> Option<ClassDefParse> {
        Self::parse_class_def_with_raw(data, off, _num_glyphs).map(|(_,p)| p)
    }

    fn parse_class_def_with_raw(data: &[u8], off: usize, _num_glyphs: usize) -> Option<(usize, ClassDefParse)> {
        if off + 2 > data.len() { return None; }
        let fmt = be_u16(data, off)?;
        if fmt == 1 {
            if off + 6 > data.len() { return None; }
            let start = be_u16(data, off+2)?;
            let count = be_u16(data, off+4)? as usize;
            if off + 6 + count*2 > data.len() { return None; }
            let mut classes = Vec::with_capacity(count);
            for i in 0..count { classes.push(be_u16(data, off+6+i*2)?); }
            Some((6+count*2, ClassDefParse::Format1 { start, classes }))
        } else if fmt == 2 {
            if off + 4 > data.len() { return None; }
            let range_count = be_u16(data, off+2)? as usize;
            let mut ranges = Vec::with_capacity(range_count);
            for i in 0..range_count {
                let base = off+4+i*6;
                if base+6 > data.len() { return None; }
                let start = be_u16(data, base)?;
                let end = be_u16(data, base+2)?;
                let cls = be_u16(data, base+4)?;
                ranges.push((start,end,cls));
            }
            Some((4+range_count*6, ClassDefParse::Format2 { ranges }))
        } else { None }
    }
}

