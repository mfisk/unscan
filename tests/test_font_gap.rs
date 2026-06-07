use ab_glyph::{Font, ScaleFont, PxScale, GlyphId};

#[test]
fn check_glyph_ids() {
    let data = std::fs::read("/usr/share/fonts/truetype/specimen-fonts/eb-garamond-700.ttf").unwrap();
    let font = ab_glyph::FontRef::try_from_slice(&data).unwrap();
    let scale = PxScale { x: 1000.0, y: 1000.0 };
    let sf = font.as_scaled(scale);
    
    println!("\nab_glyph glyph IDs and advances (at 1000px = UPM units):");
    for ch in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] {
        let gid = font.glyph_id(ch);
        let adv = sf.h_advance(gid);
        let lsb = sf.h_side_bearing(gid);
        println!("  '{}': glyph_id={:?}  advance={:.1}  lsb={:.1}", ch, gid, adv, lsb);
    }
    
    // Also check specific glyph IDs directly
    println!("\nDirect glyph ID checks:");
    for gid_raw in [142u16, 152] {
        let gid = GlyphId(gid_raw);
        let adv = sf.h_advance(gid);
        println!("  GlyphId({}): advance={:.1}", gid_raw, adv);
    }
}
