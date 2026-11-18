//! Dump verbatim vintage font keys for Times New Roman (specimen generation utility).
//!
//! Generates the vintage variants of `times.ttf` via `ensure_vintage_fonts`
//! (PostScript — the only era that produces a variant distinct from the
//! catalog for Times New Roman; TrueType and OpenType are skipped upstream as
//! redundant with the nokern catalog variant, Mike Nov 12, 2026; the Word6
//! quantized variant was removed Nov 12, 2026 as unsourced) and writes
//! `test-docs/vintage-times-keys.json` with `{era: {path, font_key}}`.
//!
//! The keys are verbatim catalog keys, produced by the same code path as
//! `main.rs` `load_fonts` (`load_font_entry` + set `vintage_era` +
//! `recompute_font_key_cache`). The specimen generator reads this JSON; no
//! font key is ever constructed by hand.
//!
//! This test finds the MS Core Fonts dir via the shared
//! `font_scan::find_msttcorefonts_dir()` helper — never a hardcoded path.
//! It asks fontconfig (`fc-list`, the standard font-finding utility)
//! where Times New Roman lives and takes that file's directory; if that
//! fails it checks $MSTTCOREFONTS_DIR, then
//! $HOME/.local/share/fonts/msttcorefonts, then the system
//! /usr/share/fonts/truetype/msttcorefonts, and uses the first dir that
//! contains times.ttf.
//!
//! The pipeline's runtime `load_fonts` uses the same helper, so this
//! test exercises the same code path.
//!
//! Run via the shared pueue queue `unprint`, never directly:
//!   cargo test --profile fastbuild -p unprint-cli --test vintage_key_dump -- --nocapture

use std::collections::BTreeMap;
use std::path::PathBuf;
use unprint::font_scan;
use unprint::vintage_cache::{self, Era};

#[test]
fn dump_vintage_times_keys() {
    let mstt_dir = font_scan::find_msttcorefonts_dir()
        .expect("times.ttf not found in any msttcorefonts dir (fc-list found nothing, and none of $MSTTCOREFONTS_DIR, ~/.local/share/fonts/msttcorefonts, /usr/share/fonts/truetype/msttcorefonts contains it)");
    let bases = font_scan::scan_fonts_uncached(std::slice::from_ref(&mstt_dir));
    let times = bases
        .iter()
        .find(|b| b.path.file_name().map(|n| n == "times.ttf").unwrap_or(false))
        .unwrap_or_else(|| panic!("times.ttf not found in {}", mstt_dir.display()));
    let eras = [Era::PostScript];
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
