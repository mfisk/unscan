// Verification for aggressive ligature rendering (Mike Nov 6, 2026):
// when the font key is not nolig, the renderer must replace regular
// character sequences with ligature glyphs — "fi" renders as the single
// FB01 glyph (gid 192 in Trebuchet). When the key is nolig, "fi" renders
// as separate f, i glyphs.
//
// Tests BOTH paths:
//  A. render_ngram (FontEntry-aware, collapses via collapsed_lig_set()).
//  B. render_ngram_fresh directly (the report path: seq + gid_overrides
//     from the entry, exactly as build_observation_table does).
// Prints the resolved glyph ID for U+FB01 and saves PNGs to /tmp for
// visual inspection.
//
// Usage: cargo run --release --example render_lig_check -- [BASE_KEY]
// Default base: DejaVuSerif-400 (static, real GSUB liga).
use unprint::char_render::{render_ngram, render_ngram_fresh, RenderParams};
use unprint::font_scan;
use unprint::glyph_map::NgramGlyphMap;
use unprint_fonts::ab_glyph::{Font, FontRef};

fn main() {
    let base = std::env::args().nth(1).unwrap_or_else(|| "DejaVuSerif-400".to_string());
    let dirs = font_scan::default_font_dirs(&[]);
    let catalog = font_scan::scan_fonts(&dirs, true);
    let lig_key = format!("{}|kern|lig", base);
    let nolig_key = format!("{}|kern|nolig", base);
    let lig_fe = catalog.iter().find(|fe| fe.font_key() == lig_key)
        .unwrap_or_else(|| panic!("lig entry not in catalog: {}", lig_key));
    let nolig_fe = catalog.iter().find(|fe| fe.font_key() == nolig_key)
        .unwrap_or_else(|| panic!("nolig entry not in catalog: {}", nolig_key));
    println!("lig   entry: {} lig_set={:#07b}", lig_fe.font_key(), lig_fe.collapsed_lig_set());
    println!("nolig entry: {} lig_set={:#07b}", nolig_fe.font_key(), nolig_fe.collapsed_lig_set());
    // Resolved glyph IDs for U+FB01: override first, then cmap.
    for (name, fe) in [("lig", lig_fe), ("nolig", nolig_fe)] {
        let ov = fe.glyph_overrides.as_ref()
            .and_then(|ovs| ovs.iter().find(|(c, _)| *c == '\u{FB01}').map(|(_, g)| *g));
        let data = std::fs::read(&fe.path).unwrap();
        let font = FontRef::try_from_slice(&data).unwrap();
        let cmap_gid = font.glyph_id('\u{FB01}').0;
        let f_gid = font.glyph_id('f').0;
        let i_gid = font.glyph_id('i').0;
        println!("{}: FB01 override={:?} cmap_gid={} f_gid={} i_gid={}", name, ov, cmap_gid, f_gid, i_gid);
    }

    // Path A: render_ngram (collapses via collapsed_lig_set()).
    let mut gmap = NgramGlyphMap::new(0);
    let params = RenderParams::default();
    let font_cache = unprint::font_cache::FontCache::new(unprint::font_cache::DEFAULT_CAPACITY);
    let mut render_a = |fe: &font_scan::FontEntry, seq: &[char], tag: &str| -> (u64, u32, u32) {
        let (hash, img) = render_ngram(fe, &font_cache, seq, &mut gmap, &params, &unprint_fonts::NoKern)
            .unwrap_or_else(|| panic!("render_ngram failed for {} {:?}", fe.font_key(), seq));
        let p = format!("/tmp/ligcheck_a_{}.png", tag);
        img.save(&p).unwrap();
        println!("A {} {:?} -> hash={:#x} {}x{} saved {}", fe.font_key(), seq, hash, img.width(), img.height(), p);
        (hash, img.width(), img.height())
    };
    let lig_fi = render_a(lig_fe, &['f', 'i'], "lig_fi");
    let lig_fb01 = render_a(lig_fe, &['\u{FB01}'], "lig_fb01");
    let nolig_fi = render_a(nolig_fe, &['f', 'i'], "nolig_fi");
    let nolig_fb01 = render_a(nolig_fe, &['\u{FB01}'], "nolig_fb01");

    // Path B: render_ngram_fresh directly (report path) with the entry's overrides.
    let render_b = |fe: &font_scan::FontEntry, seq: &[char], tag: &str| -> (u64, u32, u32) {
        let data = std::fs::read(&fe.path).unwrap();
        let mut font = FontRef::try_from_slice(&data).unwrap();
        fe.apply_variations(&mut font);
        let gid_overrides: Vec<Option<unprint_fonts::ab_glyph::GlyphId>> = seq.iter().map(|c| {
            fe.glyph_overrides.as_ref()
                .and_then(|ovs| ovs.iter().find(|(gc, _)| *gc == *c).map(|(_, g)| unprint_fonts::ab_glyph::GlyphId(*g)))
        }).collect();
        let disable_kern = fe.variant_tag.contains("nokern");
        let img = render_ngram_fresh(&font, seq, &gid_overrides, &params, disable_kern, &unprint_fonts::NoKern)
            .unwrap_or_else(|| panic!("render_ngram_fresh failed for {} {:?}", fe.font_key(), seq));
        let hash = unprint::glyph_map::hash_image(&img);
        let p = format!("/tmp/ligcheck_b_{}.png", tag);
        img.save(&p).unwrap();
        println!("B {} {:?} -> hash={:#x} {}x{} saved {}", fe.font_key(), seq, hash, img.width(), img.height(), p);
        (hash, img.width(), img.height())
    };
    let b_lig_fb01 = render_b(lig_fe, &['\u{FB01}'], "lig_fb01");
    let b_lig_fi = render_b(lig_fe, &['f', 'i'], "lig_fi");
    let b_nolig_fb01 = render_b(nolig_fe, &['\u{FB01}'], "nolig_fb01");
    let b_nolig_fi = render_b(nolig_fe, &['f', 'i'], "nolig_fi");

    // Path C: report flow — normalize_seq_for_font, then render_ngram_fresh
    // with overrides computed from the NORMALIZED sequence (exactly what
    // build_observation_table now does for both reference columns).
    let render_c = |fe: &font_scan::FontEntry, seq: &[char], tag: &str| -> (u64, u32, u32) {
        let render_seq = unprint::char_render::normalize_seq_for_font(fe, seq);
        let data = std::fs::read(&fe.path).unwrap();
        let mut font = FontRef::try_from_slice(&data).unwrap();
        fe.apply_variations(&mut font);
        let gid_overrides: Vec<Option<unprint_fonts::ab_glyph::GlyphId>> = render_seq.iter().map(|c| {
            fe.glyph_overrides.as_ref()
                .and_then(|ovs| ovs.iter().find(|(gc, _)| *gc == *c).map(|(_, g)| unprint_fonts::ab_glyph::GlyphId(*g)))
        }).collect();
        let disable_kern = fe.variant_tag.contains("nokern");
        let img = render_ngram_fresh(&font, &render_seq, &gid_overrides, &params, disable_kern, &unprint_fonts::NoKern)
            .unwrap_or_else(|| panic!("report-flow render failed for {} {:?}", fe.font_key(), seq));
        let hash = unprint::glyph_map::hash_image(&img);
        let p = format!("/tmp/ligcheck_c_{}.png", tag);
        img.save(&p).unwrap();
        println!("C {} {:?} (normalized {:?}) -> hash={:#x} {}x{} saved {}", fe.font_key(), seq, render_seq, hash, img.width(), img.height(), p);
        (hash, img.width(), img.height())
    };
    let c_lig_fb01 = render_c(lig_fe, &['\u{FB01}'], "lig_fb01");
    let c_lig_fi = render_c(lig_fe, &['f', 'i'], "lig_fi");
    let c_nolig_fb01 = render_c(nolig_fe, &['\u{FB01}'], "nolig_fb01");
    let c_nolig_fi = render_c(nolig_fe, &['f', 'i'], "nolig_fi");

    let mut pass = true;
    let check = |ok: bool, msg: &str, pass: &mut bool| {
        if ok { println!("PASS: {}", msg); } else { println!("FAIL: {}", msg); *pass = false; }
    };
    // A: lig "fi" renders as the FB01 glyph.
    check(lig_fi.0 == lig_fb01.0, "A: lig \"fi\" renders as FB01 glyph", &mut pass);
    // A: nolig renders "fi" as separate f, i glyphs even when the input
    // carries the combined FB01 character (normalize expands it).
    check(nolig_fi.0 != lig_fb01.0 && nolig_fb01.0 == nolig_fi.0, "A: nolig \"fi\" and nolig [FB01] both render as separate f, i glyphs", &mut pass);
    // A: lig vs nolig "fi" differ.
    check(lig_fi.0 != nolig_fi.0, "A: lig vs nolig \"fi\" renders differ", &mut pass);
    // B: direct [FB01] render matches the collapsed render (same glyph).
    check(b_lig_fb01.0 == lig_fb01.0, "B: report-path lig [FB01] matches collapsed render", &mut pass);
    // B: direct lig ['f','i'] (uncollapsed seq) does NOT use the ligature.
    check(b_lig_fi.0 != b_lig_fb01.0, "B: report-path lig ['f','i'] renders f,i separately (no collapse in fresh path)", &mut pass);
    // B: nolig [FB01] is still the FB01 glyph (single char renders as itself).
    check(b_nolig_fb01.0 == b_lig_fb01.0, "B: nolig [FB01] renders the FB01 glyph", &mut pass);
    // B: nolig ['f','i'] renders f,i separately (same as A nolig).
    check(b_nolig_fi.0 == nolig_fi.0, "B: nolig ['f','i'] matches A nolig render", &mut pass);
    // C: report flow — lig renders FB01 from either input form.
    check(c_lig_fb01.0 == lig_fb01.0 && c_lig_fi.0 == lig_fb01.0, "C: report-flow lig renders FB01 from [FB01] and from ['f','i']", &mut pass);
    // C: report flow — nolig renders f,i separately from either input form.
    check(c_nolig_fb01.0 == nolig_fi.0 && c_nolig_fi.0 == nolig_fi.0, "C: report-flow nolig renders separate f,i from [FB01] and from ['f','i']", &mut pass);
    // C: report flow lig vs nolig differ.
    check(c_lig_fb01.0 != c_nolig_fb01.0, "C: report-flow lig vs nolig renders differ", &mut pass);
    if !pass {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
