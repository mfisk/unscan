//! Dump verbatim vintage font keys for Times New Roman (specimen generation utility).
//!
//! Generates the vintage variants of `times.ttf` via `ensure_vintage_fonts`
//! (PostScript, TrueType, Word6, OpenType) and writes
//! `test-docs/vintage-times-keys.json` with `{era: {path, font_key}}`.
//!
//! The keys are verbatim catalog keys, produced by the same code path as
//! `main.rs` `load_fonts` (`load_font_entry` + set `vintage_era` +
//! `recompute_font_key_cache`). The specimen generator reads this JSON; no
//! font key is ever constructed by hand.
//!
//! This test finds the MS Core Fonts dir from the environment — never a
//! hardcoded path. It asks fontconfig (`fc-list`, the standard
//! font-finding utility) where Times New Roman lives and takes that
//! file's directory; if that fails it checks $MSTTCOREFONTS_DIR, then
//! $HOME/.local/share/fonts/msttcorefonts, then the system
//! /usr/share/fonts/truetype/msttcorefonts, and uses the first dir that
//! contains times.ttf.
//!
//! The pipeline's runtime `load_fonts` looks only at
//! `/usr/share/fonts/truetype/msttcorefonts` (which does not exist here) and
//! therefore generates no vintage fonts at all.
//!
//! Run via the shared pueue queue `unprint`, never directly:
//!   cargo test --profile fastbuild -p unprint-cli --test vintage_key_dump -- --nocapture

use std::collections::BTreeMap;
use std::path::PathBuf;
use unprint::font_scan;
use unprint::vintage_cache::{self, Era};

/// Find the MS Core Fonts dir from the environment. No hardcoded paths:
/// first ask fontconfig (`fc-list`, the standard font-finding utility)
/// where Times New Roman lives and take that file's directory; if that
/// fails, check $MSTTCOREFONTS_DIR, then $HOME/.local/share/fonts/msttcorefonts,
/// then the system /usr/share/fonts/truetype/msttcorefonts. Returns the
/// first dir that contains times.ttf.
fn find_mstt_dir() -> PathBuf {
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
                        return dir.to_path_buf();
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
    for dir in &candidates {
        if dir.join("times.ttf").is_file() {
            return dir.clone();
        }
    }
    panic!(
        "times.ttf not found (fc-list found nothing, and none of {} contains it)",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn dump_vintage_times_keys() {
    let mstt_dir = find_mstt_dir();
    let bases = font_scan::scan_fonts_uncached(std::slice::from_ref(&mstt_dir));
    let times = bases
        .iter()
        .find(|b| b.path.file_name().map(|n| n == "times.ttf").unwrap_or(false))
        .unwrap_or_else(|| panic!("times.ttf not found in {}", mstt_dir.display()));
    let eras = [Era::PostScript, Era::TrueType, Era::Word6, Era::OpenType];
    let _paths = vintage_cache::ensure_vintage_fonts(std::slice::from_ref(times), &eras);
    let aliases = font_scan::build_alias_table();
    let mut out: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for era in eras {
        let vp = vintage_cache::vintage_cache_path(times, era);
        assert!(
            vp.exists(),
            "vintage file missing for era {:?}: {}",
            era,
            vp.display()
        );
        let mut fe = font_scan::load_font_entry(&vp, &aliases)
            .unwrap_or_else(|| panic!("load_font_entry failed for {}", vp.display()));
        fe.vintage_era = Some(era.name().to_string());
        fe.recompute_font_key_cache();
        out.insert(
            era.name().to_string(),
            serde_json::json!({
                "path": vp.to_string_lossy(),
                "font_key": fe.font_key(),
            }),
        );
    }
    let json = serde_json::to_string_pretty(&out).unwrap();
    let out_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-docs/vintage-times-keys.json");
    std::fs::write(&out_path, &json).unwrap();
    println!("wrote {}\n{}", out_path.display(), json);
}
