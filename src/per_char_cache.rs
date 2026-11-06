//! Per-character forward cache for lazy Unicode support.
//!
//! New format (v1):
//!   lda/{code:08x}.bin   = b"LDPC" | version | char_code | feat_dim | out_dim | sigma_sq f32 | med_nn f32 | catalog_hash u64 | proj[out_dim*feat_dim] f32
//!
//! All integers LE. Atomic write via tmp file + rename. file_hash is mtime+size FNV (same as geo_cache).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::cache::cache_dir;
use crate::atomic_file::tmp_for;

pub fn lda_dir() -> PathBuf { cache_dir().join("lda-models-per-char") }

pub fn lda_path(code: u32) -> PathBuf { lda_dir().join(format!("{:08x}.bin", code)) }

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
