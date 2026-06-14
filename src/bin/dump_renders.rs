use ab_glyph::{Font, FontRef};
use unscan::char_index;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ch: char = args[1].chars().next().unwrap();
    let font_path = &args[2];
    let out_path = &args[3];

    let data = std::fs::read(font_path).unwrap();
    let font = FontRef::try_from_slice(&data).unwrap();
    let img = char_index::render_char_normalised(&font, ch).unwrap();
    img.save(out_path).unwrap();
    eprintln!("Saved {}x{} to {}", img.width(), img.height(), out_path);
}
