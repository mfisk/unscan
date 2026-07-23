use image::GrayImage;

#[derive(Clone, Debug)]
pub struct VerifyParams {
    pub width: u32,
    pub height: u32,
    pub text: String,
}

pub fn verify_render_batch(
    font_data: &[u8],
    texts: &[&str],
    params: &VerifyParams,
) -> Vec<Option<GrayImage>> {
    use std::cell::RefCell;
    thread_local! {
        static FT_LIB: RefCell<Option<freetype::Library>> = RefCell::new(None);
    }

    let ft_lib_ok = FT_LIB.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = freetype::Library::init().ok();
        }
        borrow.is_some()
    });
    if !ft_lib_ok {
        return vec![None; texts.len()];
    }

    texts.iter().map(|text| {
        let face = match rustybuzz::Face::from_slice(font_data, 0) {
            Some(f) => f,
            None => return None,
        };
        let mut buf = rustybuzz::UnicodeBuffer::new();
        buf.push_str(text);
        let out = rustybuzz::shape(&face, &[], buf);
        let infos = out.glyph_infos();

        let ft_face = FT_LIB.with(|cell| {
            let b = cell.borrow();
            let lib = b.as_ref()?;
            lib.new_memory_face2(font_data.to_vec(), 0).ok()
        })?;

        let size_26_6 = (params.height as isize) * 64;
        let _ = ft_face.set_char_size(size_26_6, 0, 72, 0);

        let mut total_w = 0u32;
        let mut max_h = 0u32;
        let mut glyph_bitmaps: Vec<(u32, GrayImage)> = Vec::new();

        for info in infos {
            let gid = info.glyph_id;
            if ft_face.load_glyph(gid as u32, freetype::face::LoadFlag::RENDER | freetype::face::LoadFlag::NO_HINTING).is_err() {
                continue;
            }
            let glyph = ft_face.glyph();
            let bitmap = glyph.bitmap();
            let w = bitmap.width() as u32;
            let h = bitmap.rows() as u32;
            if w == 0 || h == 0 { continue; }
            let buffer = bitmap.buffer();
            let mut img = GrayImage::new(w, h);
            let pitch = bitmap.pitch().unsigned_abs() as usize;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let idx = y * pitch + x;
                    if idx < buffer.len() {
                        let v = buffer[idx];
                        img.put_pixel(x as u32, y as u32, image::Luma([255 - v]));
                    }
                }
            }
            total_w += w;
            if h > max_h { max_h = h; }
            glyph_bitmaps.push((w, img));
        }

        if glyph_bitmaps.is_empty() {
            return None;
        }
        let mut canvas = GrayImage::from_pixel(total_w.max(1), max_h.max(1), image::Luma([255u8]));
        let mut x_off = 0u32;
        for (w, img) in glyph_bitmaps {
            for y in 0..img.height().min(max_h) {
                for x in 0..w {
                    if x_off + x < canvas.width() {
                        let px = img.get_pixel(x, y);
                        canvas.put_pixel(x_off + x, y, *px);
                    }
                }
            }
            x_off += w;
        }
        Some(canvas)
    }).collect()
}
