// smooth.rs — Post-match font-size smoothing.
//
// Adjacent lines matched to the same font file that have slightly different
// per-word em_px values (due to OCR bbox noise) are unified to a single
// median size.  This prevents visually jarring size wobble within a paragraph.

use crate::layout;
use crate::pdf_out::PlacedText;

/// Smooth font sizes across consecutive same-font runs.
///
/// For each run of adjacent `PlacedText` entries that matched the **same font
/// file** (by path), compute per-word `em_px` via `layout::width_matched_em_px`,
/// drop outliers more than 1 pt (≈ dpi/72 px) from the mean, and store the
/// median of the survivors into `word.smoothed_em_px`.
///
/// The PDF renderer will prefer `smoothed_em_px` over recalculating per-word.
pub fn smooth_font_sizes(entries: &mut [PlacedText], dpi: f32, font_cache: &crate::font_cache::FontCache) {
    if entries.is_empty() {
        return;
    }

    // Identify consecutive runs of the same font_path.
    let mut run_start = 0usize;
    while run_start < entries.len() {
        // Skip entries with no font match.
        let run_path = match entries[run_start].font_match {
            Some(ref fm) => fm.font_path.clone(),
            None => {
                run_start += 1;
                continue;
            }
        };

        // Extend the run while the next entry has the same font_path.
        let mut run_end = run_start + 1;
        while run_end < entries.len() {
            if let Some(ref fm) = entries[run_end].font_match {
                if fm.font_path == run_path {
                    run_end += 1;
                    continue;
                }
            }
            break;
        }

        // Process this run: [run_start .. run_end)
        smooth_run(&mut entries[run_start..run_end], dpi, font_cache);
        run_start = run_end;
    }
}

/// Smooth a single run of entries that all share the same font file.
fn smooth_run(run: &mut [PlacedText], dpi: f32, font_cache: &crate::font_cache::FontCache) {
    if run.is_empty() {
        return;
    }

    // Load font data via shared cache.
    let (font_data, overrides_owned) = match run[0].font_match {
        Some(ref fm) => match font_cache.load(&fm.font_path) {
            Ok(d) => (d, fm.glyph_overrides.clone()),
            Err(_) => return,
        },
        None => return,
    };
    let font = match unprint_fonts::ab_glyph::FontRef::try_from_slice(&font_data) {
        Ok(f) => f,
        Err(_) => return,
    };
    let overrides = overrides_owned.as_deref();

    // Collect all per-word em_px values across the run.
    let mut all_em_px: Vec<f32> = Vec::new();
    for entry in run.iter() {
        for word in &entry.words {
            if word.text.is_empty() || word.width < 1.0 {
                continue;
            }
            if let Some(em) = layout::width_matched_em_px(&font, &word.text, word.width, overrides) {
                all_em_px.push(em);
            }
        }
    }

    if all_em_px.is_empty() {
        return;
    }

    // Compute mean.
    let mean: f32 = all_em_px.iter().sum::<f32>() / all_em_px.len() as f32;

    // Outlier threshold: 1 pt in pixels.
    let one_pt_px = dpi / 72.0;

    // Filter out outliers (> 1pt from mean).
    let mut survivors: Vec<f32> = all_em_px
        .iter()
        .copied()
        .filter(|&v| (v - mean).abs() <= one_pt_px)
        .collect();

    if survivors.is_empty() {
        // All outliers — fall back to unsmoothed.
        return;
    }

    // Compute median of survivors.
    survivors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if survivors.len() % 2 == 0 {
        (survivors[survivors.len() / 2 - 1] + survivors[survivors.len() / 2]) / 2.0
    } else {
        survivors[survivors.len() / 2]
    };

    // Apply smoothed em_px to all words in the run.
    for entry in run.iter_mut() {
        for word in entry.words.iter_mut() {
            if word.text.is_empty() || word.width < 1.0 {
                continue;
            }
            // Only smooth if this word's natural em_px is within the reasonable
            // range (i.e. was not itself a hard outlier worth keeping).
            if let Some(em) = layout::width_matched_em_px(&font, &word.text, word.width, overrides) {
                if (em - mean).abs() <= one_pt_px {
                    word.smoothed_em_px = Some(median);
                }
                // Outlier words keep smoothed_em_px = None → use their natural size.
            }
        }
    }
}
