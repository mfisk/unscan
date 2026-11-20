// Probe: ab_glyph outline bounds vs ttf_parser glyph_bounding_box.
//
// Determines EMPIRICALLY how ab_glyph's outline bounds relate to
// ttf_parser's glyph bounding box for the same glyphs in a variable font:
// whether ab_glyph negates y, swaps min/max, or matches ttf_parser.
// This decides the correct conversion for the geo-cache variable-font
// build path (src/geo_cache.rs), which currently copies ab_glyph's
// b.min.y/b.max.y straight into the y_min/y_max fields.
//
// Usage: cargo run --example abglyph_ttf_bounds_probe -- [FONT_PATH]
// Default font: Inter variable (opsz,wght).
use unprint_fonts::ab_glyph::{Font, FontRef, VariableFont};
use unprint_fonts::ttf_parser;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/hatch/.local/share/fonts/ofl/inter/Inter[opsz,wght].ttf".to_string()
    });
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let face = ttf_parser::Face::parse(&data, 0).expect("ttf_parser parse");
    println!("font: {} upem={} glyphs={}", path, face.units_per_em(), face.number_of_glyphs());
    println!("note: ttf_parser glyph_bounding_box is the DEFAULT instance (no variation applied)");

    let chars = ['g', 'A', 'j', 'y', 'Q', 'p', 'x', 'f'];
    // ab_glyph default instance
    let font = FontRef::try_from_slice(&data).expect("ab_glyph FontRef");
    // ab_glyph varied instance: wght=700
    let mut font_w700 = FontRef::try_from_slice(&data).expect("ab_glyph FontRef w700");
    let set_res = font_w700.set_variation(b"wght", 700.0);
    println!("set_variation(wght, 700.0) -> {:?}", set_res);

    println!();
    println!("{:>4} {:>5} | {:>20} | {:>20} | {:>20}", "char", "gid", "ttf_parser (x0,x1,y0,y1)", "ab_glyph default", "ab_glyph wght=700");
    for c in chars {
        let gid = face.glyph_index(c).map(|g| g.0).unwrap_or(0);
        let ttf = face.glyph_bounding_box(ttf_parser::GlyphId(gid));
        let ab = font.outline(unprint_fonts::ab_glyph::GlyphId(gid));
        let ab7 = font_w700.outline(unprint_fonts::ab_glyph::GlyphId(gid));
        let ttf_s = match ttf {
            Some(bb) => format!("({},{},{},{})", bb.x_min, bb.x_max, bb.y_min, bb.y_max),
            None => "None".to_string(),
        };
        let ab_s = match &ab {
            Some(o) => format!("({:.1},{:.1},{:.1},{:.1})", o.bounds.min.x, o.bounds.max.x, o.bounds.min.y, o.bounds.max.y),
            None => "None".to_string(),
        };
        let ab7_s = match &ab7 {
            Some(o) => format!("({:.1},{:.1},{:.1},{:.1})", o.bounds.min.x, o.bounds.max.x, o.bounds.min.y, o.bounds.max.y),
            None => "None".to_string(),
        };
        println!("{:>4} {:>5} | {:>20} | {:>20} | {:>20}", c, gid, ttf_s, ab_s, ab7_s);
    }

    // Relationship analysis on the default instance.
    println!();
    println!("relationship analysis (default instance, ab_glyph vs ttf_parser):");
    let mut n_match = 0;
    let mut n_negate = 0;
    let mut n_swap = 0;
    let mut n_other = 0;
    for c in chars {
        let gid = face.glyph_index(c).map(|g| g.0).unwrap_or(0);
        let (Some(bb), Some(o)) = (
            face.glyph_bounding_box(ttf_parser::GlyphId(gid)),
            font.outline(unprint_fonts::ab_glyph::GlyphId(gid)),
        ) else {
            println!("  {} gid {}: missing bounds on one side", c, gid);
            continue;
        };
        let (tx0, tx1, ty0, ty1) = (bb.x_min as f32, bb.x_max as f32, bb.y_min as f32, bb.y_max as f32);
        let (ax0, ax1, ay0, ay1) = (o.bounds.min.x, o.bounds.max.x, o.bounds.min.y, o.bounds.max.y);
        let x_match = (tx0 - ax0).abs() < 0.5 && (tx1 - ax1).abs() < 0.5;
        let y_match = (ty0 - ay0).abs() < 0.5 && (ty1 - ay1).abs() < 0.5;
        let y_negate = (ty0 + ay1).abs() < 0.5 && (ty1 + ay0).abs() < 0.5;
        let y_swap = (ty0 - ay1).abs() < 0.5 && (ty1 - ay0).abs() < 0.5;
        let rel = if y_match {
            n_match += 1;
            "MATCH (ab y == ttf y)"
        } else if y_negate {
            n_negate += 1;
            "NEGATE (ab y == -ttf y, swapped)"
        } else if y_swap {
            n_swap += 1;
            "SWAP (ab y_min == ttf y_max and vice versa)"
        } else {
            n_other += 1;
            "OTHER"
        };
        println!(
            "  {} gid {}: x_match={} y: ttf=({},{}) ab=({:.1},{:.1}) -> {}",
            c, gid, x_match, ty0, ty1, ay0, ay1, rel
        );
    }
    println!();
    println!("summary: match={} negate={} swap={} other={}", n_match, n_negate, n_swap, n_other);
    if n_match == chars.len() {
        println!("CONCLUSION: ab_glyph outline bounds MATCH ttf_parser bounds (no conversion needed)");
    } else if n_negate == chars.len() {
        println!("CONCLUSION: ab_glyph y is NEGATED relative to ttf_parser (y_min = -ab.max.y, y_max = -ab.min.y)");
    } else if n_swap == chars.len() {
        println!("CONCLUSION: ab_glyph y min/max are SWAPPED relative to ttf_parser (y_min = ab.max.y, y_max = ab.min.y)");
    } else {
        println!("CONCLUSION: mixed or unexpected — inspect the table above");
    }
}
