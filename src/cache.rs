//! Centralized cache directory and font allowlist handling.
//!
//! All persistent caches live under `cache_dir()`:
//! - default: `$HOME/.cache/unprint/`
//! - override: `$UNPRINT_CACHE_DIR` env or `--cache-dir` CLI
//!
//! Font allowlist (`UNPRINT_FONT_ALLOWLIST` / `--font-allowlist`) is a
//! comma-separated list of font_keys (or @file). When used with the default
//! cache dir, it filters at **matching time only** (does not rewrite the
//! main catalog). When used with an alternate cache dir, it filters at
//! **scan time** and writes a filtered catalog to the alt dir.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static CACHE_DIR_OVERRIDE: OnceLock<Option<PathBuf>> = OnceLock::new();
static ALLOWLIST_CACHE: OnceLock<Option<HashSet<String>>> = OnceLock::new();

fn default_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("unprint")
    } else {
        PathBuf::from("/tmp/unprint-cache")
    }
}

/// Returns the active cache directory.
/// Checks (in order):
/// 1. `CACHE_DIR_OVERRIDE` set via `init()` from CLI
/// 2. `UNPRINT_CACHE_DIR` env var
/// 3. `~/.cache/unprint`
pub fn cache_dir() -> PathBuf {
    if let Some(Some(dir)) = CACHE_DIR_OVERRIDE.get() {
        return dir.clone();
    }
    if let Ok(dir) = std::env::var("UNPRINT_CACHE_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    default_cache_dir()
}

/// Whether we're using the default cache dir (i.e., not an explicit alt dir).
/// Used to decide if allowlist filtering should happen at scan time.
pub fn is_default_cache_dir() -> bool {
    if CACHE_DIR_OVERRIDE.get().and_then(|o| o.as_ref()).is_some() {
        return false;
    }
    std::env::var("UNPRINT_CACHE_DIR").map_or(true, |s| s.is_empty())
}

/// Initialize cache dir from CLI args. Call once early in main.
pub fn init_cache_dir(cli_dir: Option<&Path>) {
    let _ = CACHE_DIR_OVERRIDE.set(cli_dir.map(|p| p.to_path_buf()));
    if let Some(dir) = cli_dir {
        // Also set env for child processes / libraries that check env directly
        std::env::set_var("UNPRINT_CACHE_DIR", dir);
    }
}

fn parse_allowlist_str(s: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    let s = s.trim();
    if s.is_empty() {
        return set;
    }
    // If it starts with @, treat as file path
    if let Some(path) = s.strip_prefix('@') {
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                for part in line.split(',') {
                    let p = part.trim();
                    if !p.is_empty() {
                        set.insert(p.to_string());
                    }
                }
            }
            return set;
        }
        // If @file not found, fall through to treat as literal
    }
    // Also check if s is a path to an existing file (without @)
    let p = Path::new(s);
    if p.is_file() {
        if let Ok(content) = std::fs::read_to_string(p) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                for part in line.split(',') {
                    let p = part.trim();
                    if !p.is_empty() {
                        set.insert(p.to_string());
                    }
                }
            }
            return set;
        }
    }
    // Otherwise comma-separated
    for part in s.split(',') {
        let p = part.trim();
        if !p.is_empty() {
            set.insert(p.to_string());
        }
    }
    set
}

/// Returns the font allowlist, if any.
/// Checks `UNPRINT_FONT_ALLOWLIST` env (or CLI via init).
pub fn font_allowlist() -> Option<HashSet<String>> {
    if let Some(cached) = ALLOWLIST_CACHE.get() {
        return cached.clone();
    }
    if let Ok(list) = std::env::var("UNPRINT_FONT_ALLOWLIST") {
        if !list.trim().is_empty() {
            return Some(parse_allowlist_str(&list));
        }
    }
    None
}

/// Initialize allowlist from CLI. Call once early.
pub fn init_allowlist(cli_allowlist: Option<&str>) {
    if let Some(s) = cli_allowlist {
        let set = parse_allowlist_str(s);
        let _ = ALLOWLIST_CACHE.set(Some(set.clone()));
        // Also set env for consistency
        std::env::set_var("UNPRINT_FONT_ALLOWLIST", s);
    } else {
        // Try to populate from env if present
        if let Ok(env_val) = std::env::var("UNPRINT_FONT_ALLOWLIST") {
            if !env_val.trim().is_empty() {
                let set = parse_allowlist_str(&env_val);
                let _ = ALLOWLIST_CACHE.set(Some(set));
            } else {
                let _ = ALLOWLIST_CACHE.set(None);
            }
        } else {
            let _ = ALLOWLIST_CACHE.set(None);
        }
    }
}

/// Filter helper: returns true if font_key should be kept given allowlist.
/// Allowlist is in fontkey format (exact match).
pub fn allowlist_keep(font_key: &str, allowlist: Option<&HashSet<String>>) -> bool {
    match allowlist {
        None => true,
        Some(set) => set.contains(font_key),
    }
}

/// Specific paths under cache_dir
pub mod paths {
    use super::cache_dir;
    use std::path::PathBuf;

    pub fn font_scan_bin() -> PathBuf { cache_dir().join("font_scan.bin") }
    pub fn catalog_bin() -> PathBuf { cache_dir().join("catalog.bin") }
    pub fn geo_cache_bin() -> PathBuf { cache_dir().join("geo-cache.bin") }
    pub fn glyph_map_bin() -> PathBuf { cache_dir().join("glyph-map.bin") }
    pub fn lda_weights_bin() -> PathBuf { cache_dir().join("lda-weights.bin") }
    pub fn fisher_weights_bin() -> PathBuf { cache_dir().join("fisher-weights.bin") }
    pub fn triplet_weights_bin() -> PathBuf { cache_dir().join("triplet-weights.bin") }
    pub fn mahalanobis_weights_bin() -> PathBuf { cache_dir().join("mahalanobis-weights.bin") }
    pub fn mlp_weights_bin() -> PathBuf { cache_dir().join("mlp-weights.bin") }
    pub fn per_font_lda_dir() -> PathBuf { cache_dir().join("per-font-lda") }
    pub fn training_dir() -> PathBuf { cache_dir().join("training") }
    pub fn chars_dir() -> PathBuf { cache_dir().join("chars") }
}
