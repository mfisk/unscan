//! Render two separate f glyphs side by side (no ligature) using ab_glyph
use unprint_fonts::ab_glyph::{FontRef, Font, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};

fn main() {
    let font_path = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf";
    let data = std::fs::read(font_path).unwrap();
    let font = FontRef::try_from_slice(&data).unwrap();
    
    let norm_h = 48u32;
    
    // First, figure out the scale to get ~48px ink height for 'f'
    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);
    
    let f_gid = font.glyph_id('f');
    let f_glyph = f_gid.with_scale_and_position(ref_scale, point(0.0, sf_ref.ascent()));
    let f_outlined = font.outline_glyph(f_glyph).unwrap();
    let f_bounds = f_outlined.px_bounds();
    let ink_h = f_bounds.max.y - f_bounds.min.y;
    
    let target_scale = ref_h * (norm_h as f32 / ink_h);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);
    
    // Render single f at target scale
    let f_glyph2 = f_gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
    let f_out2 = font.outline_glyph(f_glyph2).unwrap();
    let fb = f_out2.px_bounds();
    
    // Get advance width for f
    let f_advance = sf.h_advance(f_gid);
    
    // Get kern between f and f
    let kern = sf.kern(f_gid, f_gid);
    
    println!("f ink bounds: ({:.1}, {:.1}) to ({:.1}, {:.1})", fb.min.x, fb.min.y, fb.max.x, fb.max.y);
    println!("f ink size: {:.1} x {:.1}", fb.max.x - fb.min.x, fb.max.y - fb.min.y);
    println!("f advance: {:.1}", f_advance);
    println!("f+f kern: {:.1}", kern);
    
    // Now render two f's side by side
    let second_f_x = f_advance + kern;
    let f2_glyph = f_gid.with_scale_and_position(scale, point(second_f_x, sf.ascent()));
    let f2_out = font.outline_glyph(f2_glyph).unwrap();
    let fb2 = f2_out.px_bounds();
    
    // Canvas: from min_x of first f to max_x of second f
    let total_min_x = fb.min.x.floor() as i32;
    let total_max_x = fb2.max.x.ceil() as i32;
    let total_min_y = fb.min.y.min(fb2.min.y).floor() as i32;
    let total_max_y = fb.max.y.max(fb2.max.y).ceil() as i32;
    
    let img_w = (total_max_x - total_min_x + 2) as u32;
    let img_h = (total_max_y - total_min_y + 2) as u32;
    
    println!("Canvas: {}x{}", img_w, img_h);
    
    let mut canvas = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));
    
    // Draw first f
    f_out2.draw(|gx, gy, cov| {
        let px = gx as i32 + fb.min.x.floor() as i32 - total_min_x + 1;
        let py = gy as i32 + fb.min.y.floor() as i32 - total_min_y + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
        }
    });
    
    // Draw second f
    f2_out.draw(|gx, gy, cov| {
        let px = gx as i32 + fb2.min.x.floor() as i32 - total_min_x + 1;
        let py = gy as i32 + fb2.min.y.floor() as i32 - total_min_y + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
        }
    });
    
    canvas.save("/tmp/ci_render_nimbus_ff_nolig.png").unwrap();
    println!("Saved /tmp/ci_render_nimbus_ff_nolig.png");
    
    // Also render the ligature glyph for comparison
    let lig_gid = font.glyph_id('\u{FB00}');
    println!("\nU+FB00 glyph id: {:?}", lig_gid);
    println!("f glyph id: {:?}", f_gid);
    
    let lig_glyph = lig_gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
    let lig_out = font.outline_glyph(lig_glyph).unwrap();
    let lb = lig_out.px_bounds();
    println!("U+FB00 ink size: {:.1} x {:.1}", lb.max.x - lb.min.x, lb.max.y - lb.min.y);
    println!("ff manual ink size: {:.1} x {:.1}", 
        fb2.max.x - fb.min.x, 
        fb.max.y.max(fb2.max.y) - fb.min.y.min(fb2.min.y));
}
