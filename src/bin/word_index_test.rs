//! Proof-of-concept: word-level font matching.
//!
//! Instead of chopping words into individual characters (which suffers from
//! Tesseract's imprecise character bounding boxes), we:
//!   1. Crop full word images from the scanned page (Tesseract word bboxes are reliable).
//!   2. Render the same word text in each candidate font, width-matched.
//!   3. Compute SSIM between the scan crop and each rendering.
//!   4. Vote: best SSIM font per word → majority vote per line.
//!
//! This is a standalone test — does NOT modify the main unscan pipeline.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use image::{DynamicImage, GrayImage, Luma};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

// ── Specimen fonts ──────────────────────────────────────────────────────────
const SPECIMEN_FONTS: &[(&str, &str)] = &[
    ("eb garamond 400", "/usr/share/fonts/truetype/specimen-fonts/eb-garamond-400.ttf"),
    ("libre caslon text 400", "/usr/share/fonts/truetype/specimen-fonts/libre-caslon-text-400.ttf"),
    ("libre baskerville 400", "/usr/share/fonts/truetype/specimen-fonts/libre-baskerville-400.ttf"),
    ("libre bodoni 400", "/usr/share/fonts/truetype/specimen-fonts/libre-bodoni-400.ttf"),
    ("zilla slab 400", "/usr/share/fonts/truetype/specimen-fonts/zilla-slab-400.ttf"),
];

// ── Section detection (matches /tmp/score.py logic) ─────────────────────────
fn detect_section(text: &str) -> Option<&'static str> {
    let t = text.to_lowercase();
    if t.contains("the garamond") { return Some("garamond"); }
    if t.contains("the caslon") { return Some("caslon"); }
    if t.contains("the baskerville") { return Some("baskerville"); }
    if t.contains("the bodoni") { return Some("bodoni"); }
    if t.contains("the slab serif") { return Some("zilla"); }
    None
}

fn font_matches_section(font_name: &str, section: &str) -> bool {
    let f = font_name.to_lowercase();
    match section {
        "garamond" => f.contains("garamond"),
        "caslon" => f.contains("caslon"),
        "baskerville" => f.contains("baskerville"),
        "bodoni" => f.contains("bodoni"),
        "zilla" => f.contains("zilla"),
        _ => false,
    }
}

// ── Word rendering ──────────────────────────────────────────────────────────

/// Render `text` in `font` onto a canvas of `canvas_w × canvas_h`, width-matched.
fn render_word(font: &FontRef, text: &str, canvas_w: u32, canvas_h: u32) -> Option<GrayImage> {
    if text.is_empty() || canvas_w < 4 || canvas_h < 4 {
        return None;
    }

    // Compute em size that makes the text fit canvas_w
    let em_px = width_matched_em(font, text, canvas_w as f32)?;
    let scale = PxScale::from(em_px);
    let sf = font.as_scaled(scale);

    // Vertical centering: ink-centered baseline
    let ink_h = sf.ascent() - sf.descent();
    let baseline = (canvas_h as f32 - ink_h) / 2.0 + sf.ascent();

    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));

    // Compute natural advance to get horizontal scale factor
    let natural_adv = {
        let mut adv = 0.0f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for c in text.chars() {
            let gid = font.glyph_id(c);
            if let Some(p) = prev {
                adv += sf.kern(p, gid);
            }
            adv += sf.h_advance(gid);
            prev = Some(gid);
        }
        adv
    };
    let h_scale = if natural_adv > 0.1 { canvas_w as f32 / natural_adv } else { 1.0 };

    let mut cx = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    let (cw, ch) = canvas.dimensions();

    for c in text.chars() {
        let gid = font.glyph_id(c);
        if let Some(p) = prev {
            cx += sf.kern(p, gid) * h_scale;
        }
        let glyph = gid.with_scale_and_position(scale, point(cx, baseline));
        if let Some(og) = font.outline_glyph(glyph) {
            let bounds = og.px_bounds();
            let bx = bounds.min.x as i32;
            let by = bounds.min.y as i32;
            og.draw(|gx, gy, cov| {
                let px = gx as i32 + bx;
                let py = gy as i32 + by;
                if px >= 0 && py >= 0 && (px as u32) < cw && (py as u32) < ch {
                    let val = (255.0 * (1.0 - cov)) as u8;
                    let cur = canvas.get_pixel(px as u32, py as u32).0[0];
                    canvas.put_pixel(px as u32, py as u32, Luma([cur.min(val)]));
                }
            });
        }
        cx += sf.h_advance(gid) * h_scale;
        prev = Some(gid);
    }

    Some(canvas)
}

fn width_matched_em(font: &FontRef, text: &str, target_w: f32) -> Option<f32> {
    let ref_h = 100.0f32;
    let sf = font.as_scaled(PxScale::from(ref_h));
    let mut adv = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for c in text.chars() {
        let gid = font.glyph_id(c);
        if let Some(p) = prev {
            adv += sf.kern(p, gid);
        }
        adv += sf.h_advance(gid);
        prev = Some(gid);
    }
    if adv < 0.1 { return None; }
    Some((ref_h * (target_w / adv)).clamp(4.0, 500.0))
}

// ── SSIM ────────────────────────────────────────────────────────────────────

/// Simple global SSIM between two grayscale images of the same size.
/// If sizes differ, the smaller is padded with white (255).
fn ssim(a: &GrayImage, b: &GrayImage) -> f32 {
    let w = a.width().max(b.width());
    let h = a.height().max(b.height());
    if w == 0 || h == 0 { return 0.0; }

    let get_a = |x: u32, y: u32| -> f64 {
        if x < a.width() && y < a.height() { a.get_pixel(x, y).0[0] as f64 } else { 255.0 }
    };
    let get_b = |x: u32, y: u32| -> f64 {
        if x < b.width() && y < b.height() { b.get_pixel(x, y).0[0] as f64 } else { 255.0 }
    };

    let n = (w * h) as f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;
    let mut sum_ab = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let va = get_a(x, y);
            let vb = get_b(x, y);
            sum_a += va;
            sum_b += vb;
            sum_a2 += va * va;
            sum_b2 += vb * vb;
            sum_ab += va * vb;
        }
    }

    let mu_a = sum_a / n;
    let mu_b = sum_b / n;
    let var_a = (sum_a2 / n) - mu_a * mu_a;
    let var_b = (sum_b2 / n) - mu_b * mu_b;
    let cov = (sum_ab / n) - mu_a * mu_b;

    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2);
    let den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
    (num / den) as f32
}

/// SSIM with vertical shift search (±max_shift pixels).
fn ssim_best_vshift(a: &GrayImage, b: &GrayImage, max_shift: i32) -> (f32, i32) {
    let mut best = f32::MIN;
    let mut best_dy = 0i32;

    for dy in -max_shift..=max_shift {
        let shifted = vshift_image(b, dy);
        let score = ssim(a, &shifted);
        if score > best {
            best = score;
            best_dy = dy;
        }
    }
    (best, best_dy)
}

fn vshift_image(img: &GrayImage, dy: i32) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::from_pixel(w, h, Luma([255u8]));
    for y in 0..h {
        let src_y = y as i32 - dy;
        if src_y >= 0 && src_y < h as i32 {
            for x in 0..w {
                out.put_pixel(x, y, *img.get_pixel(x, src_y as u32));
            }
        }
    }
    out
}

// ── TSV parsing (standalone, not importing from unscan) ─────────────────────

#[derive(Debug, Clone)]
struct TsvWord {
    text: String,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    conf: f32,
    block_num: u32,
    par_num: u32,
    line_num: u32,
}

fn parse_tsv(tsv: &str) -> Vec<TsvWord> {
    let mut words = Vec::new();
    for line in tsv.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 { continue; }
        let level: u32 = cols[0].parse().unwrap_or(0);
        if level != 5 { continue; } // word level
        let conf: f32 = cols[10].parse().unwrap_or(-1.0);
        let text = cols[11].trim().to_string();
        if text.is_empty() || conf < 0.0 { continue; }
        let x: u32 = cols[6].parse().unwrap_or(0);
        let y: u32 = cols[7].parse().unwrap_or(0);
        let w: u32 = cols[8].parse().unwrap_or(0);
        let h: u32 = cols[9].parse().unwrap_or(0);
        let block_num: u32 = cols[2].parse().unwrap_or(0);
        let par_num: u32 = cols[3].parse().unwrap_or(0);
        let line_num: u32 = cols[4].parse().unwrap_or(0);
        if w < 3 || h < 3 { continue; }
        words.push(TsvWord { text, x, y, width: w, height: h, conf, block_num, par_num, line_num });
    }
    words
}

// ── Assemble words into lines ───────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TextLine {
    text: String,
    words: Vec<TsvWord>,
    block_num: u32,
    par_num: u32,
    line_num: u32,
}

fn assemble_lines(words: &[TsvWord]) -> Vec<TextLine> {
    let mut map: HashMap<(u32, u32, u32), Vec<TsvWord>> = HashMap::new();
    for w in words {
        map.entry((w.block_num, w.par_num, w.line_num))
            .or_default()
            .push(w.clone());
    }
    let mut lines: Vec<TextLine> = map.into_iter().map(|((b, p, l), mut ws)| {
        ws.sort_by_key(|w| w.x);
        let text = ws.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        TextLine { text, words: ws, block_num: b, par_num: p, line_num: l }
    }).collect();
    lines.sort_by_key(|l| (l.block_num, l.par_num, l.line_num));
    lines
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let pdf_path = "test-docs/specimen-clean-raster.pdf";
    let dpi = 150u32;

    // 1. Convert PDF page to grayscale image
    eprintln!("Converting PDF to image...");
    let tmp_prefix = "/tmp/word_idx_test";
    let status = Command::new("pdftoppm")
        .args(["-gray", "-r", &dpi.to_string(), "-f", "1", "-l", "1", pdf_path, tmp_prefix])
        .status()
        .expect("pdftoppm failed");
    assert!(status.success(), "pdftoppm failed");

    let page_path = format!("{}-1.pgm", tmp_prefix);
    let page_img = image::open(&page_path).expect("Failed to load page image");
    let page_gray = page_img.to_luma8();
    let (pw, ph) = page_gray.dimensions();
    eprintln!("Page: {}×{}", pw, ph);

    // 2. Run Tesseract TSV to get word bboxes
    eprintln!("Running Tesseract TSV...");
    let tsv_output = Command::new("tesseract")
        .args([&page_path, "stdout", "--dpi", &dpi.to_string(), "-l", "eng", "tsv"])
        .output()
        .expect("tesseract failed");
    assert!(tsv_output.status.success(), "tesseract failed");
    let tsv = String::from_utf8_lossy(&tsv_output.stdout);
    let words = parse_tsv(&tsv);
    eprintln!("Found {} words", words.len());

    // 3. Load specimen fonts
    eprintln!("Loading fonts...");
    let mut fonts: Vec<(&str, FontRef)> = Vec::new();
    for (name, path) in SPECIMEN_FONTS {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("Cannot read {}: {}", path, e));
        // Leak the data so FontRef can have 'static lifetime
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        let font = FontRef::try_from_slice(data)
            .unwrap_or_else(|e| panic!("Cannot parse {}: {}", name, e));
        fonts.push((name, font));
    }

    // 4. Assemble lines and detect sections
    let lines = assemble_lines(&words);
    eprintln!("Assembled {} lines", lines.len());

    // Optional: dump word crops for inspection
    let dump_crops = std::env::var("DUMP_WORD_CROPS").is_ok();
    if dump_crops {
        let _ = std::fs::create_dir_all("/tmp/word_crops");
    }

    let mut current_section: Option<&'static str> = None;
    let mut section_correct: HashMap<&str, u32> = HashMap::new();
    let mut section_total: HashMap<&str, u32> = HashMap::new();
    let mut total_correct = 0u32;
    let mut total_scored = 0u32;
    let mut line_results: Vec<String> = Vec::new();

    for line in &lines {
        // Check for section header
        if let Some(section) = detect_section(&line.text) {
            current_section = Some(section);
            continue;
        }
        let section = match current_section {
            Some(s) => s,
            None => continue,
        };

        // Skip lines with no usable words (single-char fragments etc.)
        // But keep even single-word lines — they still get scored in the main pipeline.
        if line.words.is_empty() {
            continue;
        }

        // For each word in this line, find best font by SSIM
        let mut font_votes: HashMap<&str, u32> = HashMap::new();
        let mut word_count = 0u32;
        let mut best_details: Vec<String> = Vec::new();

        for word in &line.words {
            if word.text.len() < 2 || word.conf < 50.0 {
                continue;
            }

            // Crop word from page
            let x_end = (word.x + word.width).min(pw);
            let y_end = (word.y + word.height).min(ph);
            if word.x >= x_end || word.y >= y_end { continue; }
            let crop = image::imageops::crop_imm(
                &page_gray, word.x, word.y,
                x_end - word.x, y_end - word.y
            ).to_image();

            if dump_crops {
                let fname = format!("/tmp/word_crops/{}_{}.png", section, word.text.replace(' ', "_"));
                let _ = crop.save(&fname);
            }

            // Compare against each font's rendering
            let mut best_ssim = f32::MIN;
            let mut best_font = "";

            for (fname, font) in &fonts {
                let rendered = match render_word(font, &word.text, crop.width(), crop.height()) {
                    Some(r) => r,
                    None => continue,
                };

                // SSIM with vertical shift tolerance (±3px)
                let (score, _dy) = ssim_best_vshift(&crop, &rendered, 3);
                if score > best_ssim {
                    best_ssim = score;
                    best_font = fname;
                }
            }

            if !best_font.is_empty() {
                *font_votes.entry(best_font).or_insert(0) += 1;
                word_count += 1;
                best_details.push(format!("  '{}' → {} ({:.3})", word.text, best_font, best_ssim));
            }
        }

        if word_count == 0 { continue; }

        // Line winner = font with most votes
        let winner = font_votes.iter()
            .max_by_key(|(_, &v)| v)
            .map(|(&f, _)| f)
            .unwrap_or("");

        let correct = font_matches_section(winner, section);
        if correct {
            total_correct += 1;
            *section_correct.entry(section).or_insert(0) += 1;
        }
        total_scored += 1;
        *section_total.entry(section).or_insert(0) += 1;

        let mark = if correct { "✓" } else { "✗" };
        let line_text: String = line.text.chars().take(50).collect();
        line_results.push(format!(
            "{} [{}] \"{}...\" → {} ({}/{} votes)",
            mark, section, line_text, winner,
            font_votes.get(winner).unwrap_or(&0), word_count
        ));
    }

    // ── Print results ───────────────────────────────────────────────────────
    println!("\n═══ Word-Level Font Matching Results ═══\n");
    for r in &line_results {
        println!("{}", r);
    }

    println!("\n─── Per-Section Accuracy ───\n");
    for section in &["garamond", "caslon", "baskerville", "bodoni", "zilla"] {
        let c = section_correct.get(section).unwrap_or(&0);
        let t = section_total.get(section).unwrap_or(&0);
        let pct = if *t > 0 { *c as f32 / *t as f32 * 100.0 } else { 0.0 };
        println!("  {:<15} {}/{} ({:.0}%)", section, c, t, pct);
    }

    let pct = if total_scored > 0 { total_correct as f32 / total_scored as f32 * 100.0 } else { 0.0 };
    println!("\n  TOTAL: {}/{} = {:.1}%", total_correct, total_scored, pct);
    println!("\n  (char-level baseline: 42/88 = 47.7%)\n");
}
