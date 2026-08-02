//! Per-character forward cache for lazy Unicode support.
//!
//! New format (v1):
//!   means/{code:08x}.bin = b"MEAN" | version | char_code | feat_dim | n_entries
//!     per entry: font_key_len u16 | font_key bytes | file_hash u64 | count u32 | mean[feat_dim] f32
//!   lda/{code:08x}.bin   = b"LDPC" | version | char_code | feat_dim | out_dim | sigma_sq f32 | med_nn f32 | catalog_hash u64 | proj[out_dim*feat_dim] f32
//!
//! All integers LE. Atomic write via tmp file + rename. file_hash is mtime+size FNV (same as geo_cache).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::cache::cache_dir;
use crate::atomic_file::tmp_for;

pub fn means_dir() -> PathBuf { cache_dir().join("means") }
pub fn lda_dir() -> PathBuf { cache_dir().join("lda") }

pub fn means_path(code: u32) -> PathBuf { means_dir().join(format!("{:08x}.bin", code)) }
pub fn lda_path(code: u32) -> PathBuf { lda_dir().join(format!("{:08x}.bin", code)) }

#[derive(Clone, Debug)]
pub struct MeanEntry {
    pub font_key: String,
    pub file_hash: u64,
    pub count: u32,
    pub mean: Vec<f32>,
}

pub fn file_meta_hash(path: &Path) -> u64 {
    if let Ok(meta) = fs::metadata(path) {
        let size = meta.len();
        let mtime = meta.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut h = FNV_OFFSET;
        for &b in &mtime.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(FNV_PRIME); }
        for &b in &size.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(FNV_PRIME); }
        h
    } else { 0 }
}

fn write_all_atomic(target: &Path, data: &[u8]) -> io::Result<()> {
    if let Some(parent) = target.parent() { fs::create_dir_all(parent)?; }
    let tmp = tmp_for(target);
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, target)?;
    Ok(())
}

// ---------- MEANS ----------

pub fn write_means_atomic(code: u32, feat_dim: usize, entries: &[MeanEntry]) -> io::Result<()> {
    let path = means_path(code);
    let mut buf = Vec::with_capacity(16 + entries.len() * (32 + feat_dim * 4));
    buf.extend_from_slice(b"MEAN");
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&code.to_le_bytes());
    buf.extend_from_slice(&(feat_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        let key_bytes = e.font_key.as_bytes();
        assert!(key_bytes.len() <= u16::MAX as usize);
        buf.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&e.file_hash.to_le_bytes());
        buf.extend_from_slice(&e.count.to_le_bytes());
        assert_eq!(e.mean.len(), feat_dim);
        for &v in &e.mean { buf.extend_from_slice(&v.to_le_bytes()); }
    }
    write_all_atomic(&path, &buf)
}

pub fn read_means(code: u32) -> io::Result<Option<(usize, Vec<MeanEntry>)>> {
    let path = means_path(code);
    if !path.exists() { return Ok(None); }
    let mut data = Vec::new();
    fs::File::open(&path)?.read_to_end(&mut data)?;
    if data.len() < 20 { return Ok(None); }
    if &data[0..4] != b"MEAN" { return Ok(None); }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 { return Ok(None); }
    let file_code = u32::from_le_bytes(data[8..12].try_into().unwrap());
    if file_code != code { /* allow mismatch but continue */ }
    let feat_dim = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let mut off = 20usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if off + 2 > data.len() { break; }
        let klen = u16::from_le_bytes(data[off..off+2].try_into().unwrap()) as usize; off+=2;
        if off + klen + 8 + 4 > data.len() { break; }
        let key = String::from_utf8_lossy(&data[off..off+klen]).to_string(); off+=klen;
        let file_hash = u64::from_le_bytes(data[off..off+8].try_into().unwrap()); off+=8;
        let count = u32::from_le_bytes(data[off..off+4].try_into().unwrap()); off+=4;
        if off + feat_dim*4 > data.len() { break; }
        let mut mean = Vec::with_capacity(feat_dim);
        for _ in 0..feat_dim {
            let v = f32::from_le_bytes(data[off..off+4].try_into().unwrap()); off+=4;
            mean.push(v);
        }
        out.push(MeanEntry{ font_key: key, file_hash, count, mean });
    }
    Ok(Some((feat_dim, out)))
}

// ---------- LDA ----------

#[derive(Clone, Debug)]
pub struct LdaPerChar {
    pub char_code: u32,
    pub feat_dim: usize,
    pub out_dim: usize,
    pub sigma_sq: f32,
    pub med_nn: f32,
    pub catalog_hash: u64,
    pub projection: Vec<f32>, // out_dim * feat_dim row-major
}

pub fn write_lda_atomic(entry: &LdaPerChar) -> io::Result<()> {
    let path = lda_path(entry.char_code);
    let mut buf = Vec::with_capacity(32 + entry.projection.len()*4);
    buf.extend_from_slice(b"LDPC");
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&entry.char_code.to_le_bytes());
    buf.extend_from_slice(&(entry.feat_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(entry.out_dim as u32).to_le_bytes());
    buf.extend_from_slice(&entry.sigma_sq.to_le_bytes());
    buf.extend_from_slice(&entry.med_nn.to_le_bytes());
    buf.extend_from_slice(&entry.catalog_hash.to_le_bytes());
    assert_eq!(entry.projection.len(), entry.feat_dim * entry.out_dim);
    for &v in &entry.projection { buf.extend_from_slice(&v.to_le_bytes()); }
    write_all_atomic(&path, &buf)
}

pub fn read_lda(code: u32) -> io::Result<Option<LdaPerChar>> {
    let path = lda_path(code);
    if !path.exists() { return Ok(None); }
    let mut data = Vec::new();
    fs::File::open(&path)?.read_to_end(&mut data)?;
    if data.len() < 32 { return Ok(None); }
    if &data[0..4] != b"LDPC" { return Ok(None); }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 { return Ok(None); }
    let char_code = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let feat_dim = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let out_dim = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let sigma_sq = f32::from_le_bytes(data[20..24].try_into().unwrap());
    let med_nn = f32::from_le_bytes(data[24..28].try_into().unwrap());
    let catalog_hash = u64::from_le_bytes(data[28..36].try_into().unwrap());
    let expected = out_dim.checked_mul(feat_dim).unwrap_or(0);
    if data.len() < 36 + expected*4 { return Ok(None); }
    let mut proj = Vec::with_capacity(expected);
    let mut off = 36usize;
    for _ in 0..expected {
        let v = f32::from_le_bytes(data[off..off+4].try_into().unwrap()); off+=4;
        proj.push(v);
    }
    Ok(Some(LdaPerChar{ char_code, feat_dim, out_dim, sigma_sq, med_nn, catalog_hash, projection: proj }))
}

/// Validate per-char entries against current font file hashes; returns filtered list.
/// If file_hash != current hash on disk, entry is dropped (lazy recreate).
pub fn filter_valid_means(entries: Vec<MeanEntry>, catalog: &[crate::font_scan::FontEntry], font_id_map: &HashMap<String, u32>) -> Vec<MeanEntry> {
    let mut out = Vec::with_capacity(entries.len());
    for e in entries {
        if let Some(&idx) = font_id_map.get(&e.font_key) {
            if let Some(fe) = catalog.get(idx as usize) {
                let cur_hash = file_meta_hash(&fe.path);
                if cur_hash != 0 && cur_hash == e.file_hash {
                    out.push(e);
                    continue;
                }
                // If hash mismatch, drop — will be regenerated on next train
            }
        }
        // Keep entry if we can't validate (conservative) ? For now drop stale
    }
    out
}
