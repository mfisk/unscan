//! Diagnostic: compare scan-extracted char crops vs index-time rendered chars
//! Outputs side-by-side PNGs to /tmp/ci_diag/

use image::{GrayImage, Luma, imageops};
use ab_glyph::{FontRef, Font, PxScale, ScaleFont, point};
use std::path::Path;

fn main() {
    let out_dir = Path::new("/tmp/ci_diag");
    std::fs::create_dir_all(out_dir).unwrap();

    // Specimen fonts and their paths
    let fonts = [
        ("EB Garamond 400", "/usr/share/fonts/truetype/specimen-fonts/eb-garamond-400.ttf"),
        ("Libre Caslon 400", "/usr/share/fonts/truetype/specimen-fonts/libre-caslon-text-400.ttf"),
        ("Libre Baskerville 400", "/usr/share/fonts/truetype/specimen-fonts/libre-baskerville-400.ttf"),
        ("Libre Bodoni 400", "/usr/share/fonts/truetype/specimen-fonts/libre-bodoni-400.ttf"),
        ("Zilla Slab 400", "/usr/share/fonts/truetype/specimen-fonts/zilla-slab-400.ttf"),
        // What the CI typically returns as #1
        ("Impact", "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"),
        ("Arial Bold", "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        ("Georgia Bold", "/usr/share/fonts/truetype/msttcorefonts/Georgia_Bold.ttf"),
        ("Times NR Bold Ital", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf"),
        ("Verdana Bold Ital", "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"),
    ];

    let norm_h: u32 = 48;
    let test_chars = ['e', 'a', 'o', 'n', 't', 'h', 'r'];

    // For each font, render each char at NORM_H and save
    for (name, path) in &fonts {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => { eprintln!("Skip {}: {}", name, e); continue; }
        };
        let font = match FontRef::try_from_slice(&data) {
            Ok(f) => f,
            Err(e) => { eprintln!("Skip {}: {}", name, e); continue; }
        };

        for &c in &test_chars {
            let img = match render_char_normalised(&font, c, norm_h) {
                Some(i) => i,
                None => continue,
            };
            let fname = format!("{}_{}.png", name.replace(' ', "_"), c);
            img.save(out_dir.join(&fname)).unwrap();
        }
    }

    // Now render the scan-extracted version from the specimen PDF
    // Use pdftoppm to get grayscale page, then extract chars the same way unscan does
    let pdf_path = "test-docs/specimen-clean-raster.pdf";
    let dpi = 150; // typical unscan default
    let status = std::process::Command::new("pdftoppm")
        .args(["-gray", "-r", &dpi.to_string(), "-f", "1", "-l", "1", pdf_path, "/tmp/ci_diag/page"])
        .status()
        .expect("pdftoppm failed");
    assert!(status.success(), "pdftoppm failed");

    // Find the output file
    let page_path = if Path::new("/tmp/ci_diag/page-1.pgm").exists() {
        "/tmp/ci_diag/page-1.pgm"
    } else if Path::new("/tmp/ci_diag/page-01.pgm").exists() {
        "/tmp/ci_diag/page-01.pgm"
    } else {
        eprintln!("Cannot find pdftoppm output");
        return;
    };

    let page_img = image::open(page_path).unwrap().to_luma8();
    let (pw, ph) = page_img.dimensions();
    eprintln!("Page: {}x{}", pw, ph);

    // Extract crops manually by finding dark regions and cutting characters
    // For a simpler approach: just crop the test chars from known positions
    // OR: use the unscan extract_line_chars function... but that needs OCR word boxes
    //
    // Simplest: manually crop a sample word from a known location in the PDF
    // The EB Garamond section starts roughly at y ~= 15% of page, first body text line
    // Let's just do a visual grid of the index-time renders

    // Create comparison grid: one row per char, columns = fonts
    let grid_cell_w = 64u32;
    let grid_cell_h = 64u32;
    let cols = fonts.len() as u32;
    let rows = test_chars.len() as u32;
    let label_w = 0u32;
    let grid_w = label_w + cols * grid_cell_w;
    let grid_h = rows * grid_cell_h;
    let mut grid = GrayImage::from_pixel(grid_w, grid_h, Luma([240u8]));

    for (col, (name, path)) in fonts.iter().enumerate() {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let font = match FontRef::try_from_slice(&data) {
            Ok(f) => f,
            Err(_) => continue,
        };
        for (row, &c) in test_chars.iter().enumerate() {
            if let Some(img) = render_char_normalised(&font, c, norm_h) {
                let (iw, ih) = img.dimensions();
                let x = label_w + col as u32 * grid_cell_w + (grid_cell_w.saturating_sub(iw)) / 2;
                let y = row as u32 * grid_cell_h + (grid_cell_h.saturating_sub(ih)) / 2;
                imageops::overlay(&mut grid, &img, x as i64, y as i64);
            }
        }
    }
    grid.save(out_dir.join("font_grid.png")).unwrap();
    eprintln!("Saved font_grid.png (cols: specimen fonts + CI top scorers)");

    // Now compute features for each and print the feature distances
    eprintln!("\n=== Feature distance analysis ===");
    eprintln!("For each char, showing L2 distance between specimen fonts and CI-top fonts:");

    for &c in &test_chars {
        eprintln!("\nChar '{}' — feature vectors:", c);
        let mut feat_vecs: Vec<(String, Vec<f32>)> = Vec::new();

        for (name, path) in &fonts {
            let data = match std::fs::read(path) { Ok(d) => d, Err(_) => continue };
            let font = match FontRef::try_from_slice(&data) { Ok(f) => f, Err(_) => continue };
            if let Some(img) = render_char_normalised(&font, c, norm_h) {
                let feats = compute_simple_features(&img);
                feat_vecs.push((name.to_string(), feats));
            }
        }

        // Print distances between each specimen font and each CI-top font
        let specimen_names: Vec<&str> = vec!["EB Garamond 400", "Libre Caslon 400", "Libre Baskerville 400", "Libre Bodoni 400", "Zilla Slab 400"];
        let ci_top_names: Vec<&str> = vec!["Impact", "Arial Bold", "Georgia Bold", "Times NR Bold Ital", "Verdana Bold Ital"];

        for spec in &specimen_names {
            let spec_feat = feat_vecs.iter().find(|(n,_)| n == spec);
            if let Some((_, sf)) = spec_feat {
                let mut dists: Vec<(String, f32)> = Vec::new();
                for (n, f) in &feat_vecs {
                    let d: f32 = sf.iter().zip(f.iter()).map(|(a,b)| (a-b)*(a-b)).sum::<f32>().sqrt();
                    dists.push((n.clone(), d));
                }
                dists.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
                let top5: Vec<String> = dists.iter().take(6).map(|(n,d)| format!("{}: {:.3}", n, d)).collect();
                eprintln!("  {} → {}", spec, top5.join(" | "));
            }
        }
    }

    // Now the real test: extract actual scan chars from the PDF and compare
    // Scan the page for words using simple connected-component analysis
    eprintln!("\n=== Scan-extracted vs rendered comparison ===");
    extract_scan_chars_and_compare(&page_img, &fonts, norm_h, out_dir);
}

fn render_char_normalised(font: &FontRef, c: char, norm_h: u32) -> Option<GrayImage> {
    let gid = font.glyph_id(c);
    if gid.0 == 0 {
        return None;
    }

    let ref_h = 200.0f32;
    let ref_scale = PxScale::from(ref_h);
    let sf_ref = font.as_scaled(ref_scale);

    let glyph = gid.with_scale_and_position(ref_scale, point(0.0, sf_ref.ascent()));
    let outlined = font.outline_glyph(glyph)?;
    let bounds = outlined.px_bounds();
    let ink_h_ref = bounds.max.y - bounds.min.y;
    if ink_h_ref < 1.0 {
        return None;
    }

    let target_scale = ref_h * (norm_h as f32 / ink_h_ref);
    let scale = PxScale::from(target_scale);
    let sf = font.as_scaled(scale);

    let glyph2 = gid.with_scale_and_position(scale, point(0.0, sf.ascent()));
    let outlined2 = font.outline_glyph(glyph2)?;
    let b2 = outlined2.px_bounds();

    let img_w = (b2.max.x - b2.min.x).ceil() as u32 + 2;
    let img_h = (b2.max.y - b2.min.y).ceil() as u32 + 2;
    if img_w == 0 || img_h == 0 || img_w > 500 || img_h > 500 {
        return None;
    }

    let mut canvas = GrayImage::from_pixel(img_w, img_h, Luma([255u8]));
    let ox = b2.min.x.floor() as i32;
    let oy = b2.min.y.floor() as i32;

    outlined2.draw(|gx, gy, cov| {
        let px = gx as i32 + (b2.min.x.floor() as i32) - ox + 1;
        let py = gy as i32 + (b2.min.y.floor() as i32) - oy + 1;
        if px >= 0 && py >= 0 && (px as u32) < img_w && (py as u32) < img_h {
            let val = (255.0 * (1.0 - cov)) as u8;
            let cur = canvas.get_pixel(px as u32, py as u32).0[0];
            canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
        }
    });

    Some(canvas)
}

fn compute_simple_features(img: &GrayImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let threshold = 200u8;
    let mut ink_pixels = 0u32;
    let mut total = 0u32;
    let mut wy_sum = 0.0f64;
    let mut min_x = w; let mut max_x = 0u32;
    let mut min_y = h; let mut max_y = 0u32;

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y).0[0];
            total += 1;
            if px < threshold {
                ink_pixels += 1;
                wy_sum += y as f64;
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }
    }

    if ink_pixels == 0 {
        return vec![0.0; 5];
    }

    let ink_w = (max_x - min_x + 1) as f32;
    let ink_h = (max_y - min_y + 1) as f32;
    let aspect = ink_w / ink_h.max(1.0);
    let density = ink_pixels as f32 / (ink_w * ink_h).max(1.0);
    let v_center = (wy_sum / ink_pixels as f64) as f32 / h as f32;
    let fill = ink_pixels as f32 / total as f32;

    vec![aspect, density, v_center, fill, ink_w / w as f32]
}

fn extract_scan_chars_and_compare(page: &GrayImage, fonts: &[(&str, &str); 10], norm_h: u32, out_dir: &Path) {
    // Find horizontal text lines by looking for rows with significant ink
    let (pw, ph) = page.dimensions();
    let threshold = 180u8;

    // Build row ink profile
    let mut row_ink = vec![0u32; ph as usize];
    for y in 0..ph {
        for x in 0..pw {
            if page.get_pixel(x, y).0[0] < threshold {
                row_ink[y as usize] += 1;
            }
        }
    }

    // Find text lines: contiguous runs of rows with ink > 5% of width
    let min_ink = pw / 20;
    let mut lines: Vec<(u32, u32)> = Vec::new(); // (top, bottom)
    let mut in_line = false;
    let mut line_top = 0u32;

    for y in 0..ph {
        if row_ink[y as usize] > min_ink {
            if !in_line {
                in_line = true;
                line_top = y;
            }
        } else if in_line {
            if y - line_top >= 5 {
                lines.push((line_top, y));
            }
            in_line = false;
        }
    }

    eprintln!("Found {} text line regions", lines.len());

    // For first 10 lines, extract 'e' crops and compare
    let target_char = 'e';
    let mut scan_crops: Vec<GrayImage> = Vec::new();

    for (idx, &(top, bot)) in lines.iter().take(15).enumerate() {
        let h = bot - top;
        // Find columns with ink in this line region
        let mut col_ink = vec![0u32; pw as usize];
        for y in top..bot {
            for x in 0..pw {
                if page.get_pixel(x, y).0[0] < threshold {
                    col_ink[x as usize] += 1;
                }
            }
        }

        // Find first word-ish region (consecutive columns with ink)
        let min_col_ink = h / 4;
        let mut in_word = false;
        let mut word_start = 0u32;
        let mut words: Vec<(u32, u32)> = Vec::new();

        for x in 0..pw {
            if col_ink[x as usize] > min_col_ink {
                if !in_word {
                    in_word = true;
                    word_start = x;
                }
            } else if in_word {
                if x - word_start >= 10 {
                    words.push((word_start, x));
                }
                in_word = false;
            }
        }

        if words.is_empty() { continue; }

        // Take first long word, crop it, try to extract a middle character
        for &(wx, wx2) in words.iter().take(3) {
            let ww = wx2 - wx;
            if ww < 20 { continue; }

            let word_crop = imageops::crop_imm(page, wx, top, ww, h).to_image();

            // Save the word crop
            let fname = format!("scan_line{}_word.png", idx);
            word_crop.save(out_dir.join(&fname)).ok();

            // Try to extract a single character from middle of word
            // Approximate: take a vertical strip ~1/5 of word width from center
            let char_w = (ww / 5).max(5);
            let char_x = ww / 3; // slightly off-center to hit a body char
            let char_crop = imageops::crop_imm(&word_crop, char_x, 0, char_w.min(ww - char_x), h).to_image();

            // Normalize to NORM_H
            let scaled = imageops::resize(
                &char_crop,
                ((char_w as f32 * norm_h as f32 / h as f32).ceil() as u32).max(1),
                norm_h,
                imageops::FilterType::Lanczos3,
            );

            let fname = format!("scan_line{}_char.png", idx);
            scaled.save(out_dir.join(&fname)).ok();
            scan_crops.push(scaled);
            break;
        }
    }

    // Build comparison strip: scan crop vs specimen rendered vs CI-top rendered
    if scan_crops.is_empty() {
        eprintln!("No scan crops extracted");
        return;
    }

    // Render 'e' for each font
    let mut font_renders: Vec<(String, GrayImage)> = Vec::new();
    for (name, path) in fonts {
        let data = match std::fs::read(path) { Ok(d) => d, Err(_) => continue };
        let font = match FontRef::try_from_slice(&data) { Ok(f) => f, Err(_) => continue };
        if let Some(img) = render_char_normalised(&font, target_char, norm_h) {
            font_renders.push((name.to_string(), img));
        }
    }

    // Build comparison image
    let cell_w = 64u32;
    let cell_h = 64u32;
    let n_scan = scan_crops.len().min(10) as u32;
    let n_fonts = font_renders.len() as u32;
    let total_cols = 1 + n_fonts; // scan + fonts
    let strip_w = total_cols * cell_w;
    let strip_h = n_scan * cell_h;
    let mut strip = GrayImage::from_pixel(strip_w, strip_h, Luma([240u8]));

    for (row, scan_img) in scan_crops.iter().take(n_scan as usize).enumerate() {
        // Scan crop in first column
        let (sw, sh) = scan_img.dimensions();
        let x = (cell_w.saturating_sub(sw)) / 2;
        let y = row as u32 * cell_h + (cell_h.saturating_sub(sh)) / 2;
        imageops::overlay(&mut strip, scan_img, x as i64, y as i64);

        // Draw separator
        for py in (row as u32 * cell_h)..((row as u32 + 1) * cell_h) {
            strip.put_pixel(cell_w - 1, py, Luma([100u8]));
        }

        // Font renders
        for (col, (_, fimg)) in font_renders.iter().enumerate() {
            let (fw, fh) = fimg.dimensions();
            let fx = (1 + col as u32) * cell_w + (cell_w.saturating_sub(fw)) / 2;
            let fy = row as u32 * cell_h + (cell_h.saturating_sub(fh)) / 2;
            imageops::overlay(&mut strip, fimg, fx as i64, fy as i64);
        }
    }

    strip.save(out_dir.join("scan_vs_rendered.png")).unwrap();
    eprintln!("Saved scan_vs_rendered.png (col 0 = scan, cols 1-{} = fonts)", n_fonts);

    // Print feature distances between scan crops and font renders
    eprintln!("\n=== Scan crop vs rendered feature distances ===");
    for (row, scan_img) in scan_crops.iter().take(5).enumerate() {
        let scan_feat = compute_simple_features(scan_img);
        eprintln!("Scan line {} features: aspect={:.2} density={:.2} vcenter={:.2} fill={:.2}",
            row, scan_feat[0], scan_feat[1], scan_feat[2], scan_feat[3]);
        for (name, fimg) in &font_renders {
            let font_feat = compute_simple_features(fimg);
            let dist: f32 = scan_feat.iter().zip(font_feat.iter()).map(|(a,b)| (a-b)*(a-b)).sum::<f32>().sqrt();
            eprint!("  {} d={:.3} (asp={:.2} den={:.2}) |", name, dist, font_feat[0], font_feat[1]);
        }
        eprintln!();
    }
}
