use ab_glyph::{Font, FontRef};
use unprint::features;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ch: char = args[1].chars().next().unwrap();
    let font_path = &args[2];
    let out_path = &args[3];

    let data = std::fs::read(font_path).unwrap();
    let font = FontRef::try_from_slice(&data).unwrap();
    let img = unprint::char_render::render_glyph_at_ink_height(&font, font.glyph_id(ch), features::NORM_H)
        .and_then(|img| features::normalize_to_ink_bounds(&img, features::NORM_H)).unwrap();
    img.save(out_path).unwrap();
    eprintln!("Saved {}x{} to {}", img.width(), img.height(), out_path);
}
