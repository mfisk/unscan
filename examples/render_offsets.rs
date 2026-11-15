use unprint::char_render::{RenderParams, render_ngram_fresh};
use std::path::PathBuf;

fn main() {
    let font_path = PathBuf::from("/home/hatch/.local/share/fonts/msttcorefonts/trebuc.ttf");
    let font_data = std::fs::read(&font_path).unwrap();
    let font = unprint_fonts::ab_glyph::FontArc::try_from_vec(font_data).unwrap();
    
    for dx in [0.0f32, 0.25, 0.5, 0.75] {
        let mut params = RenderParams::default();
        params.height = 48;
        params.subpixel_dx = dx;
        params.subpixel_dy = 0.0;
        
        if let Some(img) = render_ngram_fresh(&font, &['a'], &[None], &params, false, &unprint_fonts::NoKern) {
            let path = format!("/tmp/a_dx{}.png", dx);
            img.save(&path).unwrap();
            println!("Saved {} ({}x{})", path, img.width(), img.height());
        }
    }
}
