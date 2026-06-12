//! Dump CI index render for a specific char in a specific font
use ab_glyph::FontRef;
use unscan::char_index::render_char_normalised;

fn main() {
    let font_path = std::env::args().nth(1).expect("usage: dump_ci_glyph <font_path> <char_hex>");
    let char_hex = std::env::args().nth(2).expect("usage: dump_ci_glyph <font_path> <char_hex>");
    
    let codepoint = u32::from_str_radix(char_hex.trim_start_matches("U+").trim_start_matches("0x"), 16)
        .expect("invalid hex codepoint");
    let c = char::from_u32(codepoint).expect("invalid char");
    
    let data = std::fs::read(&font_path).expect("can't read font");
    let font = FontRef::try_from_slice(&data).expect("can't parse font");
    
    match render_char_normalised(&font, c) {
        Some(img) => {
            let out = format!("/tmp/ci_render_U+{:04X}.png", codepoint);
            img.save(&out).unwrap();
            println!("Saved {} ({}x{})", out, img.width(), img.height());
        }
        None => {
            println!("render_char_normalised returned None for U+{:04X} — glyph not found in font", codepoint);
        }
    }
}
