//! Word-level geometry: expand, fix overlaps, trim to ink.
//! Batch APIs: one call per page/lines, inner per-word/per-pixel loops stay inline inside this crate.

use image::GrayImage;
use crate::text::TextLine;

fn walk_ink_edge(
    gray: &GrayImage,
    start: u32,
    limit: u32,
    y_top: u32,
    y_bot: u32,
    blur: u8,
    direction: i32,
) -> u32 {
    let page_w = gray.width();
    let mut edge = start;
    if direction > 0 {
        for col in start..limit.min(page_w) {
            if (y_top..y_bot).any(|row| gray.get_pixel(col, row).0[0] < blur) {
                edge = col + 1;
            } else { break; }
        }
    } else {
        for col in (limit..start).rev() {
            if (y_top..y_bot).any(|row| gray.get_pixel(col.min(page_w-1), row).0[0] < blur) {
                edge = col;
            } else { break; }
        }
    }
    edge
}

fn walk_ink_edge_vertical(
    gray: &GrayImage,
    start: u32,
    limit: u32,
    x_left: u32,
    x_right: u32,
    blur: u8,
    direction: i32,
) -> u32 {
    let page_w = gray.width();
    let page_h = gray.height();
    let mut edge = start;
    if direction > 0 {
        for row in start..limit.min(page_h) {
            if (x_left..x_right.min(page_w)).any(|col| gray.get_pixel(col, row).0[0] < blur) {
                edge = row + 1;
            } else { break; }
        }
    } else {
        for row in (limit..start).rev() {
            if (x_left..x_right.min(page_w)).any(|col| gray.get_pixel(col, row).0[0] < blur) {
                edge = row;
            } else { break; }
        }
    }
    edge
}

/// Batch API: expand word bboxes to ink.
/// One call per page: &mut [TextLine] + gray + thresholds -> mutates lines in place.
pub fn expand_words_to_ink(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8, blur: u8, margin: u32) {
    let (page_w, page_h) = gray.dimensions();
    for line in lines.iter_mut() {
        let n = line.words.len();
        for i in 0..n {
            {
                let w = &line.words[i];
                let right_edge = w.x + w.width;
                let limit = if i+1 < n { line.words[i+1].x } else { (right_edge+margin).min(page_w) };
                if right_edge < limit && right_edge < page_w {
                    let check_col = right_edge.saturating_sub(1);
                    let y_top = w.y; let y_bot = w.y+w.height;
                    let has_edge_ink = (y_top..y_bot).any(|row| gray.get_pixel(check_col.min(page_w-1), row).0[0] < ink_threshold);
                    if has_edge_ink {
                        let new_right = walk_ink_edge(gray, right_edge, limit, y_top, y_bot, blur, 1);
                        if new_right > right_edge { line.words[i].width = new_right - line.words[i].x; }
                    }
                }
            }
            {
                let left_edge = line.words[i].x;
                let word_y = line.words[i].y;
                let word_h = line.words[i].height;
                let mut limit = if i>0 { line.words[i-1].x + line.words[i-1].width } else { left_edge.saturating_sub(margin) };
                if i>0 {
                    let prev_x = line.words[i-1].x;
                    let prev_right = prev_x + line.words[i-1].width;
                    let prev_y_top = line.words[i-1].y;
                    let prev_y_bot = prev_y_top + line.words[i-1].height;
                    let mut shrink_to = prev_right;
                    for col in (prev_x..prev_right).rev() {
                        let col_has_ink = (prev_y_top..prev_y_bot).any(|row| gray.get_pixel(col.min(page_w-1), row).0[0] < ink_threshold);
                        if col_has_ink { shrink_to = col+1; break; }
                        shrink_to = col;
                    }
                    if shrink_to < prev_right { line.words[i-1].width = shrink_to.saturating_sub(prev_x); limit = shrink_to; }
                    else if left_edge <= prev_right {
                        let scan_y_top = prev_y_top.min(word_y);
                        let scan_y_bot = prev_y_bot.max(word_y+word_h);
                        let scan_left = prev_right.saturating_sub(prev_y_top).max(prev_x);
                        let mut gap_col: Option<u32> = None;
                        for col in (scan_left..prev_right).rev() {
                            let ink: u32 = (scan_y_top..scan_y_bot).map(|row| if gray.get_pixel(col.min(page_w-1), row).0[0] < ink_threshold {1} else {0}).sum();
                            if ink==0 { gap_col = Some(col); break; }
                        }
                        if let Some(gc)=gap_col { line.words[i-1].width = gc.saturating_sub(prev_x); limit = gc; }
                    }
                }
                if left_edge > limit {
                    let y_top = word_y; let y_bot = word_y+word_h;
                    let gap_has_ink = (limit..left_edge).any(|col| (y_top..y_bot).any(|row| gray.get_pixel(col.min(page_w-1), row).0[0] < ink_threshold));
                    let has_edge_ink = (y_top..y_bot).any(|row| gray.get_pixel(left_edge.min(page_w-1), row).0[0] < ink_threshold);
                    if has_edge_ink || gap_has_ink {
                        let new_left = walk_ink_edge(gray, left_edge, limit, y_top, y_bot, blur, -1);
                        if new_left < left_edge { let growth = left_edge-new_left; line.words[i].x = new_left; line.words[i].width += growth; }
                    }
                }
            }
            {
                let w = &line.words[i];
                let wx = w.x; let wr = (w.x+w.width).min(page_w);
                let word_top = w.y; let word_bot = w.y+w.height;
                let search_top = word_top.saturating_sub(margin);
                let search_bot = (word_bot+margin).min(page_h);
                let new_top = walk_ink_edge_vertical(gray, word_top, search_top, wx, wr, blur, -1);
                let new_bot = walk_ink_edge_vertical(gray, word_bot, search_bot, wx, wr, blur, 1);
                if new_top < word_top || new_bot > word_bot { line.words[i].y = new_top; line.words[i].height = new_bot-new_top; }
            }
        }
    }
    for line in lines.iter_mut() {
        if let (Some(x0),Some(y0),Some(x1),Some(y1)) = (
            line.words.iter().map(|w| w.x).min(),
            line.words.iter().map(|w| w.y).min(),
            line.words.iter().map(|w| w.x+w.width).max(),
            line.words.iter().map(|w| w.y+w.height).max(),
        ) { line.x=x0; line.y=y0; line.width=x1-x0; line.height=y1-y0; }
    }
}

/// Batch API: fix overlapping words by finding natural whitespace gap.
/// One call per page.
pub fn fix_overlapping_words_by_ink(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8) {
    let page_w = gray.width(); let page_h = gray.height();
    for line in lines.iter_mut() {
        let n = line.words.len(); if n<2 { continue; }
        for i in 0..n-1 {
            let a_x = line.words[i].x; let a_right = a_x+line.words[i].width;
            let b_x = line.words[i+1].x; let b_right = b_x+line.words[i+1].width;
            if a_right <= b_x { continue; }
            let a_center = a_x+line.words[i].width/2;
            let b_center = b_x+line.words[i+1].width/2;
            let search_left = a_center.min(page_w.saturating_sub(1));
            let mut search_right = b_center.min(page_w);
            if search_right <= search_left { let new_w = b_x.saturating_sub(a_x); if new_w>0 { line.words[i].width=new_w; } continue; }
            let union_left = a_x.min(b_x); let union_right = a_right.max(b_right).min(page_w);
            let search_left = search_left.max(union_left); search_right = search_right.min(union_right);
            if search_right <= search_left { let new_w=b_x.saturating_sub(a_x); if new_w>0 { line.words[i].width=new_w; } continue; }
            let y_top_full = line.words[i].y.min(line.words[i+1].y);
            let y_bot_full = (line.words[i].y+line.words[i].height).max(line.words[i+1].y+line.words[i+1].height).min(page_h);
            if y_top_full>=y_bot_full { let new_w=b_x.saturating_sub(a_x); if new_w>0 { line.words[i].width=new_w; } continue; }
            let y_top = y_top_full;
            let y_bot = y_bot_full;
            if y_top>=y_bot { let new_w=b_x.saturating_sub(a_x); if new_w>0 { line.words[i].width=new_w; } continue; }
            let mut col_has_ink = Vec::with_capacity((search_right-search_left) as usize);
            for col in search_left..search_right {
                let mut has=false;
                for row in y_top..y_bot { if gray.get_pixel(col,row).0[0] < ink_threshold { has=true; break; } }
                col_has_ink.push(has);
            }
            let mut runs: Vec<(u32,u32)> = Vec::new(); let mut run_start: Option<u32>=None;
            for (idx,has) in col_has_ink.iter().enumerate() { let col=search_left+idx as u32; if !*has { if run_start.is_none(){run_start=Some(col);} } else if let Some(rs)=run_start.take(){ runs.push((rs,col-1)); } }
            if let Some(rs)=run_start { runs.push((rs,search_right-1)); }
            if runs.is_empty(){ let new_w=b_x.saturating_sub(a_x); if new_w>0 { line.words[i].width=new_w; } continue; }
            let overlap_center=(a_right+b_x)/2;
            let mut best_idx=0; let mut best_dist=u32::MAX; let mut best_width=0;
            for (idx,(rs,re)) in runs.iter().enumerate() { let run_center=(rs+re)/2; let dist=run_center.abs_diff(overlap_center); let width=re-rs; if dist<best_dist || (dist==best_dist && width>best_width){best_dist=dist;best_width=width;best_idx=idx;} }
            let (best_rs,best_re)=runs[best_idx];
            let new_a_width=best_rs.saturating_sub(a_x); let new_b_x=best_re+1;
            if new_a_width==0 || new_b_x>=b_right || new_b_x<=a_x { let new_w=b_x.saturating_sub(a_x); if new_w>0 { line.words[i].width=new_w; } continue; }
            let new_b_width=b_right.saturating_sub(new_b_x); if new_b_width==0 { continue; }
            line.words[i].width=new_a_width; line.words[i+1].x=new_b_x; line.words[i+1].width=new_b_width;
        }
    }
}

/// Batch API: trim words to ink.
/// Removes trailing/leading whitespace that Tesseract included, but never
/// expands across a zero-ink column in the middle 80% band — that column is
/// true inter-word whitespace (e.g. Originally/for gap 529,530).
/// Must run after `expand_words_to_ink` and `fix_overlapping_words_by_ink`.
pub fn trim_words_to_ink(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8) {
    let page_w = gray.width(); let _page_h = gray.height();
    for line in lines.iter_mut() {
        for word in line.words.iter_mut() {
            if word.width<=2 || word.height<=2 { continue; }
            let wx = word.x.min(page_w.saturating_sub(1)); let wy = word.y.min(_page_h.saturating_sub(1));
            let ww = word.width.min(page_w-wx); let wh = word.height.min(_page_h-wy);
            if ww==0 || wh==0 { continue; }
            let y_top_full = wy; let y_bot_full = wy+wh;
            let scan_left = wx;
            let scan_right = (wx+ww).min(page_w);
            let mut left_full: Option<u32> = None;
            for col in scan_left..scan_right {
                if (y_top_full..y_bot_full).any(|row| gray.get_pixel(col.min(page_w-1), row.min(_page_h-1)).0[0] < ink_threshold) {
                    left_full = Some(col); break;
                }
            }
            let mut right_full: Option<u32> = None;
            for col in (scan_left..scan_right).rev() {
                if (y_top_full..y_bot_full).any(|row| gray.get_pixel(col.min(page_w-1), row.min(_page_h-1)).0[0] < ink_threshold) {
                    right_full = Some(col); break;
                }
            }
            let li = left_full.unwrap_or(wx);
            let ri = right_full.unwrap_or(wx+ww-1);
            if ri < li { continue; }
            let new_x = li;
            let new_w = ri - li + 1;
            word.x=new_x; word.width=new_w;
        }
    }
}

/// Scan grayscale for vertical ink extent - batch helper.
pub fn ink_vertical_extent(gray: &GrayImage, x: u32, w: u32, search_top: u32, search_bot: u32, threshold: u8) -> (u32,u32) {
    let mut first: Option<u32>=None; let mut last=search_top;
    for row in search_top..search_bot { for col in x..x+w { if gray.get_pixel(col,row).0[0] < threshold { if first.is_none(){first=Some(row);} last=row; break; } } }
    match first { Some(t)=>(t,last+1), None=>(search_top,search_top) }
}

/// Batch wrapper for all three refinements in order: expand -> fix -> trim -> fix.
/// Second fix enforces r <= x invariant because trim can expand up to 5px.
pub fn refine_words_batch(lines: &mut [TextLine], gray: &GrayImage, ink_threshold: u8, blur: u8, margin: u32) {
    expand_words_to_ink(lines, gray, ink_threshold, blur, margin);
    fix_overlapping_words_by_ink(lines, gray, ink_threshold);
    trim_words_to_ink(lines, gray, ink_threshold);
    // Second pass: trim can expand up to 5px and may re-introduce overlap (italic tail).
    fix_overlapping_words_by_ink(lines, gray, ink_threshold);
}
