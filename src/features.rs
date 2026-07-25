//! Character feature computation.
//!
//! Extracts a fixed-length feature vector from a normalised grayscale glyph
//! image.  The vector captures column/row ink profiles, density, aspect ratio,
//! stroke topology (crossings, terminals, skeleton), symmetry, corners, and
//! other shape descriptors.

use image::{GrayImage, Luma};


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Per-dimension Fisher weights from learn_weights analysis.
/// Normalised character height for all feature computation.
pub const NORM_H: u32 = 24;

/// Number of bins in the column ink density profile.
/// At NORM_H=24, typical glyph widths are 10–20 px, so 16 bins avoids
/// upsampling noise.  (Was 32 when NORM_H=48.)
pub const PROFILE_BINS: usize = 16;

/// Number of bins in the row ink density profile (horizontal).
/// 24 rows → 16 bins keeps roughly 1:1 sampling.  (Was 32.)
pub const ROW_PROFILE_BINS: usize = 16;

/// Anti-aliasing variant for reference character rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AaVariant {
    /// Native rasterizer AA (no post-processing).
    Native,
    /// Gaussian blur σ=0.5 post-process.
    Blur05,
    /// Sharpen (unsharp mask) post-process.
    Sharpen,
}

impl AaVariant {
    pub fn name(&self) -> &'static str {
        match self {
            AaVariant::Native => "native",
            AaVariant::Blur05 => "blur05",
            AaVariant::Sharpen => "sharpen",
        }
    }

    pub fn parse(s: &str) -> Option<AaVariant> {
        match s.to_lowercase().as_str() {
            "native" => Some(AaVariant::Native),
            "blur05" | "blur" => Some(AaVariant::Blur05),
            "sharpen" => Some(AaVariant::Sharpen),
            _ => None,
        }
    }

    pub fn all() -> &'static [AaVariant] {
        &[AaVariant::Native, AaVariant::Blur05, AaVariant::Sharpen]
    }

    /// Apply this AA variant to a greyscale character image.
    pub fn apply(&self, img: &image::GrayImage) -> image::GrayImage {
        match self {
            AaVariant::Native => img.clone(),
            AaVariant::Blur05 => image::imageops::blur(img, 0.5),
            AaVariant::Sharpen => {
                // Unsharp mask: original + (original - blurred) * amount
                let blurred = image::imageops::blur(img, 0.5);
                let mut out = img.clone();
                for (p, b) in out.pixels_mut().zip(blurred.pixels()) {
                    let diff = p.0[0] as f32 - b.0[0] as f32;
                    p.0[0] = (p.0[0] as f32 + diff * 0.5).clamp(0.0, 255.0) as u8;
                }
                out
            }
        }
    }
}

/// Binarize a greyscale image at the given threshold.
/// Pixels with value < threshold → 0 (black ink), >= threshold → 255 (white).
pub fn binarize(img: &image::GrayImage, threshold: u8) -> image::GrayImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        p.0[0] = if p.0[0] < threshold { 0 } else { 255 };
    }
    out
}

/// Number of horizontal crossing scan lines.
/// 24 px / 4 lines ≈ 6 px spacing — same density as 48 px / 8.  (Was 8.)
const CROSSING_BINS: usize = 4;

/// Number of terminal angle bins (up/right/down/left).
const TERMINAL_ANGLE_BINS: usize = 4;

/// Number of original scalar features (aspect..stroke_contrast).
pub const SCALAR_V1: usize = 6;

/// Number of new discriminative scalar features.
pub const SCALAR_V2: usize = 4 + TERMINAL_ANGLE_BINS + 2 + CROSSING_BINS;
// counter_area_ratio(1) + counter_centroid(2) + counter_aspect(1) = 4
// terminal_count is folded into terminal_angles normalization
// terminal_angles(4) + ink_perimeter(1) + compactness(1) = 6
// h_crossings(4)

/// Number of v3 feature dimensions.
pub const SCALAR_V3: usize = 1 + 1 + 1 + 2 + 1 + 4 + 1;
// hole_count(1) + h_symmetry(1) + v_symmetry(1) + skeleton(branch_pts, end_pts = 2) + corner_count(1) + quadrant_density(4) + mean_stroke_width(1)

/// Feature vector length: col_profile + row_profile + original scalars + v2 + v3.
pub const FEAT_LEN: usize = PROFILE_BINS + ROW_PROFILE_BINS + SCALAR_V1 + SCALAR_V2 + SCALAR_V3;

/// Uniform Fisher weights (placeholder until per-sequence learned weights are loaded).
pub const FISHER_WEIGHTS: [f32; FEAT_LEN] = [1.0 / FEAT_LEN as f32; FEAT_LEN];

/// Canonical feature dimension names, one per FEAT_LEN slot, in as_slice() order.
pub const FEAT_NAMES: [&str; FEAT_LEN] = [
    // Column profile (16)
    "col0","col1","col2","col3","col4","col5","col6","col7",
    "col8","col9","col10","col11","col12","col13","col14","col15",
    // Scalar v1 (6)
    "aspect","ink_density","v_center","h_balance","serif_score","stroke_contrast",
    // Counter features (4)
    "counter_area","counter_cx","counter_cy","counter_asp",
    // Terminal angles (4)
    "term0","term1","term2","term3",
    // Boundary (2)
    "ink_perim","compactness",
    // Horizontal crossings (4)
    "cross0","cross1","cross2","cross3",
    // Row profile (16)
    "row0","row1","row2","row3","row4","row5","row6","row7",
    "row8","row9","row10","row11","row12","row13","row14","row15",
    // Scalar v3 (11)
    "hole_count","h_symmetry","v_symmetry","skeleton_branch","skeleton_end",
    "corner_count","quad_tl","quad_tr","quad_bl","quad_br","mean_stroke_w",
];

/// Group boundary offsets for feature dimension ranges.
pub const GROUP_OFFSETS: [(usize, usize, &str); 5] = [
    (0, PROFILE_BINS, "Col profile"),
    (PROFILE_BINS, PROFILE_BINS + SCALAR_V1, "Scalar v1"),
    (PROFILE_BINS + SCALAR_V1, PROFILE_BINS + SCALAR_V1 + SCALAR_V2, "Scalar v2"),
    (PROFILE_BINS + SCALAR_V1 + SCALAR_V2, PROFILE_BINS + SCALAR_V1 + SCALAR_V2 + ROW_PROFILE_BINS, "Row profile"),
    (PROFILE_BINS + SCALAR_V1 + SCALAR_V2 + ROW_PROFILE_BINS, FEAT_LEN, "Scalar v3"),
];

/// Return the feature-group name for a given feature dimension index.
pub fn group_name_for_dim(i: usize) -> &'static str {
    for &(start, end, name) in &GROUP_OFFSETS {
        if i >= start && i < end {
            return name;
        }
    }
    "unknown"
}

/// Downscale an image by `scale_factor` (0..1) and re-normalize to NORM_H.
/// Shared by learn_weights (DPI simulation) and gen_training_data (native-height simulation).
pub fn degrade_and_renormalize(img: &GrayImage, scale_factor: f32) -> Option<GrayImage> {
    let (w, h) = img.dimensions();
    let small_w = ((w as f32 * scale_factor).round() as u32).max(3);
    let small_h = ((h as f32 * scale_factor).round() as u32).max(3);
    let small = image::imageops::resize(img, small_w, small_h, image::imageops::FilterType::Lanczos3);
    normalize_to_ink_bounds(&small, NORM_H)
}

// ---------------------------------------------------------------------------
// Indexed character set
// ---------------------------------------------------------------------------

/// Returns the full set of characters we index.
pub fn supported_chars() -> &'static [char] {
    static CHARS: std::sync::LazyLock<Vec<char>> = std::sync::LazyLock::new(|| {
        let mut v: Vec<char> = Vec::with_capacity(140);
        for c in 'a'..='z' { v.push(c); }
        for c in 'A'..='Z' { v.push(c); }
        for c in '0'..='9' { v.push(c); }
        for c in &[
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
            '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\',
            ']', '^', '_', '`', '{', '|', '}', '~',
        ] {
            v.push(*c);
        }
        v.push('\u{2014}'); // em dash
        v.push('\u{2013}'); // en dash
        v.push('\u{2018}'); // left single quote
        v.push('\u{2019}'); // right single quote
        v.push('\u{201C}'); // left double quote
        v.push('\u{201D}'); // right double quote
        v.push('\u{2026}'); // ellipsis
        // Standard and discretionary ligatures
        v.push('\u{FB00}'); // ff ligature
        v.push('\u{FB01}'); // fi ligature
        v.push('\u{FB02}'); // fl ligature
        v.push('\u{FB03}'); // ffi ligature
        v.push('\u{FB04}'); // ffl ligature
        v
    });
    &CHARS
}

/// Quick membership test.
pub fn is_supported(c: char) -> bool {
    supported_chars().contains(&c)
}

/// Common English bigrams (character pairs) for bigram classifiers.
/// Includes frequent letter pairs, mixed-case pairs for sentence starts,
/// and letter–punctuation pairs (period, comma) to capture relative size.
pub fn supported_sequences(n: usize) -> &'static [Vec<char>] {
    match n {
        2 => supported_bigrams_internal(),
        _ => &[],
    }
}

fn supported_bigrams_internal() -> &'static [Vec<char>] {
    static BIGRAMS: std::sync::LazyLock<Vec<Vec<char>>> = std::sync::LazyLock::new(|| {
        let mut v = Vec::with_capacity(400);

        // Top ~100 lowercase bigrams by English frequency
        let lc = [
            "th","he","in","er","an","re","on","at","en","nd",
            "ti","es","or","te","of","ed","is","it","al","ar",
            "st","to","nt","ng","se","ha","as","ou","io","le",
            "ve","co","me","de","hi","ri","ro","ic","ne","ea",
            "ra","ce","li","ch","ll","be","ma","si","om","ur",
            "ca","el","ta","la","ns","ge","ly","pe","ec","vi",
            "nc","no","tr","sh","di","ss","ag","ni","ct","pr",
            "pl","ad","mi","fo","ow","un","su","us","mo","ol",
            "wa","pa","do","gi","sa","rs","ex","im","po","we",
            "ac","il","em","fi","ab","ir","id","ho","lo","op",
        ];
        for s in &lc {
            let mut cs = s.chars();
            let c1 = cs.next().unwrap();
            let c2 = cs.next().unwrap();
            v.push(vec![c1, c2]);
        }

        // Mixed-case pairs (uppercase + lowercase) for word starts
        let mc = [
            "Th","He","In","An","St","Ar","Ma","Re","Al","Se",
            "Co","De","La","Pr","Ch","Le","Ca","Ha","Be","En",
            "El","Ne","Mo","Di","Me","Mi","No","Do","Ba","Lo",
            "To","Ro","Pa","Fo","Wi","Si","Li","Ho","Fi","Ea",
            "Wh","Gr","Tr","Fr","Cr","Cl","Bl","Fl","Br","Dr",
            "So","Po","Da","Te","We","Bo","Ri","Ra","Pe","Sh",
        ];
        for s in &mc {
            let mut cs = s.chars();
            let c1 = cs.next().unwrap();
            let c2 = cs.next().unwrap();
            v.push(vec![c1, c2]);
        }

        // Letter + punctuation pairs: every lowercase/uppercase letter
        // adjacent to period or comma (captures relative size of punctuation)
        for c in 'a'..='z' {
            v.push(vec![c, '.']);
            v.push(vec![c, ',']);
        }
        for c in 'A'..='Z' {
            v.push(vec![c, '.']);
            v.push(vec![c, ',']);
        }

        // Punctuation + uppercase letter (sentence boundaries: ". T")
        // Less common (usually separated by space) but included for
        // abbreviation patterns like "U.S." → "S."
        // Already covered by uppercase + '.' above.

        // Digit pairs for numbers
        let digits = [
            "01","10","12","19","20","23","30","45","50","67","89","99",
        ];
        for s in &digits {
            let mut cs = s.chars();
            let c1 = cs.next().unwrap();
            let c2 = cs.next().unwrap();
            v.push(vec![c1, c2]);
        }

        // Dedup (some bigrams may overlap between categories)
        v.sort();
        v.dedup();
        v
    });
    &BIGRAMS
}

/// Check if a sequence is in the supported set.
pub fn is_supported_sequence(seq: &[char]) -> bool {
    supported_sequences(seq.len()).iter().any(|s| s.as_slice() == seq)
}

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/// Compact feature vector for one character rendering.
#[derive(Debug, Clone)]
pub struct CropFeatures {
    /// 16-bin column ink density profile (each 0.0–1.0).
    pub profile: [f32; PROFILE_BINS],
    /// Ink width / ink height.
    pub aspect: f32,
    /// Total ink pixels / bounding-box area.
    pub ink_density: f32,
    /// Vertical centre of mass (0.0 = top, 1.0 = bottom).
    pub v_center: f32,
    /// Left-half ink / total ink.
    pub h_balance: f32,
    /// Serif confidence score (0.0 = sans-serif, 1.0 = definite serif).
    pub serif_score: f32,
    /// Stroke contrast: ratio of thickest to thinnest strokes.
    pub stroke_contrast: f32,

    // ── v2 discriminative features ──────────────────────────────────
    /// Counter (enclosed whitespace) area / total bbox area. 0 if no counter.
    pub counter_area_ratio: f32,
    /// Counter centroid X, normalised to [0,1] within ink bbox.
    pub counter_centroid_x: f32,
    /// Counter centroid Y, normalised to [0,1] within ink bbox.
    pub counter_centroid_y: f32,
    /// Counter aspect ratio (width/height of counter bbox). 0 if no counter.
    pub counter_aspect: f32,
    /// Terminal angle histogram: 4 bins (up/right/down/left), normalised.
    pub terminal_angles: [f32; TERMINAL_ANGLE_BINS],
    /// Ink perimeter / sqrt(ink_area). Normalised boundary complexity.
    pub ink_perimeter: f32,
    /// Compactness: 4π × area / perimeter². Circle=1.0, complex<1.0.
    pub compactness: f32,
    /// Horizontal crossings at 4 evenly-spaced scan lines. Each value is
    /// the number of ink↔white transitions, normalised by dividing by 20.
    pub h_crossings: [f32; CROSSING_BINS],

    // ── v3 discriminative features ──────────────────────────────────
    /// Row ink density profile (16 bins, horizontal analog of column profile).
    pub row_profile: [f32; ROW_PROFILE_BINS],
    /// Number of enclosed white regions (holes), normalised by dividing by 4.
    pub hole_count: f32,
    /// Horizontal mirror symmetry score (0.0 = asymmetric, 1.0 = perfect mirror).
    pub h_symmetry: f32,
    /// Vertical mirror symmetry score (0.0 = asymmetric, 1.0 = perfect mirror).
    pub v_symmetry: f32,
    /// Skeleton branch points (3+ neighbors), normalised by dividing by 10.
    pub skeleton_branch_pts: f32,
    /// Skeleton endpoints (1 neighbor), normalised by dividing by 10.
    pub skeleton_end_pts: f32,
    /// Corner count (sharp direction changes on ink boundary), normalised by dividing by 20.
    pub corner_count: f32,
    /// Ink density in each quadrant (TL, TR, BL, BR), each 0.0–1.0.
    pub quadrant_density: [f32; 4],
    /// Mean stroke width normalised by glyph height. Distinguishes weight classes
    /// (Light ≈ 0.08, Regular ≈ 0.12, Medium ≈ 0.14, Bold ≈ 0.18).
    pub mean_stroke_width: f32,

    // ── raster (optional) ──────────────────────────────────────────
    /// Normalised raster image, carried through for pixel-based classifiers
    /// (e.g. ZnccClassifier). `None` when the pipeline doesn't need it.
    pub raster: Option<GrayImage>,
}

impl CropFeatures {
    /// Raw feature vector for serialisation (no normalisation).
    pub fn as_slice(&self) -> [f32; FEAT_LEN] {
        let mut v = [0.0f32; FEAT_LEN];
        let mut i = 0;
        // Column profile (16)
        v[i..i + PROFILE_BINS].copy_from_slice(&self.profile);
        i += PROFILE_BINS;
        // Original scalars (6)
        v[i] = self.aspect;           i += 1;
        v[i] = self.ink_density;      i += 1;
        v[i] = self.v_center;         i += 1;
        v[i] = self.h_balance;        i += 1;
        v[i] = self.serif_score;      i += 1;
        v[i] = self.stroke_contrast;  i += 1;
        // v2: counter features (4)
        v[i] = self.counter_area_ratio;   i += 1;
        v[i] = self.counter_centroid_x;   i += 1;
        v[i] = self.counter_centroid_y;   i += 1;
        v[i] = self.counter_aspect;       i += 1;
        // v2: terminal angles (4)
        v[i..i + TERMINAL_ANGLE_BINS].copy_from_slice(&self.terminal_angles);
        i += TERMINAL_ANGLE_BINS;
        // v2: boundary features (2)
        v[i] = self.ink_perimeter;    i += 1;
        v[i] = self.compactness;      i += 1;
        // v2: horizontal crossings (4)
        v[i..i + CROSSING_BINS].copy_from_slice(&self.h_crossings);
        i += CROSSING_BINS;
        // v3: row profile (16)
        v[i..i + ROW_PROFILE_BINS].copy_from_slice(&self.row_profile);
        i += ROW_PROFILE_BINS;
        // v3: scalars (10)
        v[i] = self.hole_count;           i += 1;
        v[i] = self.h_symmetry;           i += 1;
        v[i] = self.v_symmetry;           i += 1;
        v[i] = self.skeleton_branch_pts;  i += 1;
        v[i] = self.skeleton_end_pts;     i += 1;
        v[i] = self.corner_count;         i += 1;
        v[i..i + 4].copy_from_slice(&self.quadrant_density);
        i += 4;
        v[i] = self.mean_stroke_width;
        // i += 1;  // last field
        debug_assert_eq!(i + 1, FEAT_LEN);
        v
    }

}

/// Detect whether a glyph image shows serifs.
fn detect_serif(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    if w < 3 || h < 8 {
        return 0.0;
    }

    let threshold = 200u8;
    let mut row_ink = vec![0u32; h as usize];
    let mut min_y = h;
    let mut max_y = 0u32;

    for y in 0..h {
        let mut count = 0u32;
        for x in 0..w {
            if img.as_raw()[(y) as usize * img.width() as usize + (x) as usize] < threshold {
                count += 1;
            }
        }
        row_ink[y as usize] = count;
        if count > 0 {
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }
    }

    let ink_h = max_y.saturating_sub(min_y) + 1;
    if ink_h < 8 {
        return 0.0;
    }

    let terminal_rows = (ink_h / 8).max(2).min(4);

    let top_ink: f32 = (min_y..min_y + terminal_rows)
        .map(|y| row_ink[y as usize] as f32)
        .sum::<f32>() / terminal_rows as f32;
    let bot_ink: f32 = ((max_y + 1 - terminal_rows)..=max_y)
        .map(|y| row_ink[y as usize] as f32)
        .sum::<f32>() / terminal_rows as f32;

    let mid_start = min_y + ink_h / 3;
    let mid_end = min_y + 2 * ink_h / 3;
    let mid_rows = (mid_end - mid_start).max(1);
    let mid_ink: f32 = (mid_start..mid_end)
        .map(|y| row_ink[y as usize] as f32)
        .sum::<f32>() / mid_rows as f32;

    if mid_ink < 1.0 {
        return 0.0;
    }

    let top_ratio = top_ink / mid_ink;
    let bot_ratio = bot_ink / mid_ink;
    let avg_ratio = (top_ratio + bot_ratio) / 2.0;

    ((avg_ratio - 1.0)).clamp(0.0, 1.0)
}

/// Compute the expected pixel gap between the ink of character `a` and the ink
/// of character `b` when typeset adjacently at `scale`.  Returns RSB(a) + LSB(b)
/// in pixels, i.e. the whitespace the font places between the two glyphs'
/// ink bounding boxes.  Returns 0.0 when either glyph has no outline.

// ---------------------------------------------------------------------------
// Feature computation
// ---------------------------------------------------------------------------

/// Compute features from a normalised grayscale glyph image.
///
/// When `pre_normalized` is true the caller asserts the image has already
/// been contrast-normalized (e.g. the word crop was normalized before
/// segmentation) and the internal normalization pass is skipped.
pub fn compute_features(img: &GrayImage, pre_normalized: bool) -> Option<CropFeatures> {
    let img = if pre_normalized {
        std::borrow::Cow::Borrowed(img)
    } else {
        // Contrast-normalize: ensures symmetric treatment between
        // training renders and inference crops.  On a clean render this
        // is effectively a no-op (p1≈0, p99≈255).
        std::borrow::Cow::Owned(contrast_normalize_char(img.clone()))
    };
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }

    let threshold = 200u8;
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;
    let mut total_ink = 0u64;
    let mut ink_pixels = 0u64;
    let mut wy_sum = 0.0f64;

    // ── Single pass: accumulate bounds, ink totals, col/row sums ──
    // We'll do a two-step approach: first pass gets bounds + totals,
    // then a second restricted pass over the ink bbox builds col_ink,
    // row_ink, left_ink, and ink_mask simultaneously.

    for y in 0..h {
        for x in 0..w {
            let px = img.as_raw()[(y) as usize * img.width() as usize + (x) as usize];
            if px < threshold {
                let ink_val = (255 - px) as u64;
                total_ink += ink_val;
                ink_pixels += 1;
                wy_sum += y as f64 * ink_val as f64;
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }
    }

    if ink_pixels == 0 || total_ink == 0 {
        return None;
    }

    let ink_w = (max_x - min_x + 1) as f32;
    let ink_h = (max_y - min_y + 1) as f32;
    let aspect = ink_w / ink_h.max(1.0);
    let bbox_area = ink_w * ink_h;
    let ink_density = ink_pixels as f32 / bbox_area.max(1.0);
    let v_center = (wy_sum / total_ink as f64) as f32 / h as f32;

    let ink_w_u = (max_x - min_x + 1) as usize;
    let ink_h_u = (max_y - min_y + 1) as usize;
    // h_balance: left-half ink / total ink, split at ink-bbox midpoint
    // (not image midpoint w/2 — that would bias for off-center glyphs).
    // Use local x within ink bbox: midpoint = (ink_w - 1)/2 ensures even
    // widths split 2|2 rather than 3|1.
    let ink_mid_lx = (ink_w_u - 1) / 2;

    // Second pass over ink bbox only: col_ink, row_ink, left_ink, ink_mask
    let mut col_ink = vec![0.0f32; ink_w_u];
    let mut row_ink = vec![0.0f32; ink_h_u];
    let mut left_ink = 0u64;
    let mut ink_mask = vec![false; ink_w_u * ink_h_u];

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = img.as_raw()[(y) as usize * img.width() as usize + (x) as usize];
            if px < threshold {
                let ink_val = (255 - px) as f32;
                let lx = (x - min_x) as usize;
                let ly = (y - min_y) as usize;
                col_ink[lx] += ink_val;
                row_ink[ly] += ink_val;
                ink_mask[ly * ink_w_u + lx] = true;
                if lx <= ink_mid_lx {
                    left_ink += ink_val as u64;
                }
            }
        }
    }
    let h_balance = left_ink as f32 / total_ink as f32;

    let col_max = col_ink.iter().copied().fold(0.0f32, f32::max);
    if col_max > 0.0 {
        for v in &mut col_ink {
            *v /= col_max;
        }
    }
    let profile = resample(&col_ink, PROFILE_BINS);

    let row_max = row_ink.iter().copied().fold(0.0f32, f32::max);
    if row_max > 0.0 {
        for v in &mut row_ink {
            *v /= row_max;
        }
    }
    let row_profile = resample(&row_ink, ROW_PROFILE_BINS);

    let serif_score = detect_serif(&img);
    let stroke_contrast_val = measure_stroke_contrast(&img);
    let mean_stroke_width_val = measure_mean_stroke_width(&img);

    // ── v2/v3 features (computed from ink_mask) ───────────────────

    // Shared flood-fill: compute reachable from edges once, used by
    // both counter-feature analysis and hole counting.
    let reachable = flood_fill_from_edges(&ink_mask, ink_w_u, ink_h_u);

    let (counter_area_ratio, counter_centroid_x, counter_centroid_y, counter_aspect) =
        counter_features_from_reachable(&ink_mask, &reachable, ink_w_u, ink_h_u);

    let terminal_angles = compute_terminal_angles(&ink_mask, ink_w_u, ink_h_u);

    let (ink_perimeter, compactness) =
        compute_boundary_features(&ink_mask, ink_w_u, ink_h_u, ink_pixels as f32);

    let h_crossings = compute_h_crossings(&ink_mask, ink_w_u, ink_h_u);

    let hole_count = hole_count_from_reachable(&ink_mask, &reachable, ink_w_u, ink_h_u);

    let (h_symmetry, v_symmetry) = compute_symmetry(&ink_mask, ink_w_u, ink_h_u);

    let (skeleton_branch_pts, skeleton_end_pts) = compute_skeleton_features(&ink_mask, ink_w_u, ink_h_u);

    let corner_count = compute_corner_count(&ink_mask, ink_w_u, ink_h_u);

    let quadrant_density = compute_quadrant_density(&ink_mask, ink_w_u, ink_h_u);

    Some(CropFeatures {
        profile,
        row_profile,
        aspect,
        ink_density,
        v_center,
        h_balance,
        serif_score,
        stroke_contrast: stroke_contrast_val,
        counter_area_ratio,
        counter_centroid_x,
        counter_centroid_y,
        counter_aspect,
        terminal_angles,
        ink_perimeter,
        compactness,
        h_crossings,
        hole_count,
        h_symmetry,
        v_symmetry,
        skeleton_branch_pts,
        skeleton_end_pts,
        corner_count,
        quadrant_density,
        mean_stroke_width: mean_stroke_width_val,
        raster: Some(img.into_owned()),
    })
}

// ---------------------------------------------------------------------------
// v2 discriminative feature helpers
// ---------------------------------------------------------------------------

/// Counter shape analysis via edge flood-fill.
///
/// Flood-fills white pixels from edges of the ink bounding box. Any white pixels
/// NOT reached are enclosed counters (e.g. the hole in 'o', 'a', 'e').
/// Edge flood-fill: marks all white pixels reachable from the image border.
/// Shared by counter-feature analysis and hole counting.
fn flood_fill_from_edges(ink_mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let total = w * h;
    let mut reachable = vec![false; total];
    if w == 0 || h == 0 { return reachable; }

    let mut queue = std::collections::VecDeque::with_capacity(w * 2 + h * 2);

    // Seed from all edge pixels that are NOT ink
    for x in 0..w {
        if !ink_mask[x] && !reachable[x] {
            reachable[x] = true;
            queue.push_back(x); // flat index
        }
        let idx = (h - 1) * w + x;
        if !ink_mask[idx] && !reachable[idx] {
            reachable[idx] = true;
            queue.push_back(idx);
        }
    }
    for y in 1..h.saturating_sub(1) {
        let idx = y * w;
        if !ink_mask[idx] && !reachable[idx] {
            reachable[idx] = true;
            queue.push_back(idx);
        }
        let idx = y * w + (w - 1);
        if !ink_mask[idx] && !reachable[idx] {
            reachable[idx] = true;
            queue.push_back(idx);
        }
    }

    // BFS flood fill using flat indices
    while let Some(idx) = queue.pop_front() {
        let x = idx % w;
        let y = idx / w;
        if x > 0 {
            let ni = idx - 1;
            if !ink_mask[ni] && !reachable[ni] { reachable[ni] = true; queue.push_back(ni); }
        }
        if x + 1 < w {
            let ni = idx + 1;
            if !ink_mask[ni] && !reachable[ni] { reachable[ni] = true; queue.push_back(ni); }
        }
        if y > 0 {
            let ni = idx - w;
            if !ink_mask[ni] && !reachable[ni] { reachable[ni] = true; queue.push_back(ni); }
        }
        if y + 1 < h {
            let ni = idx + w;
            if !ink_mask[ni] && !reachable[ni] { reachable[ni] = true; queue.push_back(ni); }
        }
    }

    reachable
}

/// Counter shape features from pre-computed reachable mask.
fn counter_features_from_reachable(ink_mask: &[bool], reachable: &[bool], w: usize, h: usize) -> (f32, f32, f32, f32) {
    if w == 0 || h == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let total = w * h;
    let mut counter_area = 0u32;
    let mut cx_sum = 0.0f64;
    let mut cy_sum = 0.0f64;
    let mut counter_min_x = w;
    let mut counter_max_x = 0usize;
    let mut counter_min_y = h;
    let mut counter_max_y = 0usize;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !ink_mask[idx] && !reachable[idx] {
                counter_area += 1;
                cx_sum += x as f64;
                cy_sum += y as f64;
                if x < counter_min_x { counter_min_x = x; }
                if x > counter_max_x { counter_max_x = x; }
                if y < counter_min_y { counter_min_y = y; }
                if y > counter_max_y { counter_max_y = y; }
            }
        }
    }

    if counter_area == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let area_ratio = counter_area as f32 / total as f32;
    let centroid_x = (cx_sum / counter_area as f64) as f32 / w.max(1) as f32;
    let centroid_y = (cy_sum / counter_area as f64) as f32 / h.max(1) as f32;
    let cw = (counter_max_x - counter_min_x + 1) as f32;
    let ch = (counter_max_y - counter_min_y + 1) as f32;
    let c_aspect = cw / ch.max(1.0);

    (area_ratio, centroid_x, centroid_y, c_aspect)
}

/// Hole count from pre-computed reachable mask (no duplicate BFS).
fn hole_count_from_reachable(ink_mask: &[bool], reachable: &[bool], w: usize, h: usize) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let total = w * h;
    let mut visited = vec![false; total];
    let mut hole_count = 0u32;

    for idx in 0..total {
        if !ink_mask[idx] && !reachable[idx] && !visited[idx] {
            hole_count += 1;
            // Flood-fill this hole to mark all its pixels
            let mut q = std::collections::VecDeque::new();
            visited[idx] = true;
            q.push_back(idx);
            while let Some(ci) = q.pop_front() {
                let cx = ci % w;
                let cy = ci / w;
                if cx > 0 {
                    let ni = ci - 1;
                    if !ink_mask[ni] && !reachable[ni] && !visited[ni] {
                        visited[ni] = true; q.push_back(ni);
                    }
                }
                if cx + 1 < w {
                    let ni = ci + 1;
                    if !ink_mask[ni] && !reachable[ni] && !visited[ni] {
                        visited[ni] = true; q.push_back(ni);
                    }
                }
                if cy > 0 {
                    let ni = ci - w;
                    if !ink_mask[ni] && !reachable[ni] && !visited[ni] {
                        visited[ni] = true; q.push_back(ni);
                    }
                }
                if cy + 1 < h {
                    let ni = ci + w;
                    if !ink_mask[ni] && !reachable[ni] && !visited[ni] {
                        visited[ni] = true; q.push_back(ni);
                    }
                }
            }
        }
    }

    (hole_count as f32 / 4.0).min(1.0)
}

/// Terminal / endpoint analysis.
///
/// A terminal pixel has exactly 1 ink neighbor in 8-connected. We classify
/// the direction based on where the single neighbor is relative to the terminal.
/// Returns a 4-bin normalised histogram: [up, right, down, left].
fn compute_terminal_angles(ink_mask: &[bool], w: usize, h: usize) -> [f32; TERMINAL_ANGLE_BINS] {
    let mut bins = [0.0f32; TERMINAL_ANGLE_BINS]; // up, right, down, left
    if w < 3 || h < 3 {
        return bins;
    }

    let deltas: [(i32, i32); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];

    let mut total_terminals = 0u32;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !ink_mask[idx] { continue; }

            // Count ink neighbors and find the single neighbor if count==1
            let mut neighbor_count = 0u32;
            let mut nb_dx = 0i32;
            let mut nb_dy = 0i32;

            for &(dx, dy) in &deltas {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    if ink_mask[ny as usize * w + nx as usize] {
                        neighbor_count += 1;
                        nb_dx = dx;
                        nb_dy = dy;
                    }
                }
            }

            if neighbor_count == 1 {
                total_terminals += 1;
                // The terminal points AWAY from its single neighbor
                // If neighbor is below (dy=1), terminal points up → bin 0
                // If neighbor is left (dx=-1), terminal points right → bin 1
                // If neighbor is above (dy=-1), terminal points down → bin 2
                // If neighbor is right (dx=1), terminal points left → bin 3
                let angle = (-nb_dy as f64).atan2(-nb_dx as f64); // direction away
                let deg = angle.to_degrees();
                // -180..180 → bin: up=[-135,-45), right=[-45,45), down=[45,135), left=else
                let bin = if deg >= -135.0 && deg < -45.0 {
                    0 // up
                } else if deg >= -45.0 && deg < 45.0 {
                    1 // right
                } else if deg >= 45.0 && deg < 135.0 {
                    2 // down
                } else {
                    3 // left
                };
                bins[bin] += 1.0;
            }
        }
    }

    // Normalise by total terminals (so histogram sums to 1.0)
    if total_terminals > 0 {
        for b in &mut bins {
            *b /= total_terminals as f32;
        }
    }

    bins
}

/// Boundary complexity features.
///
/// Returns (normalised_perimeter, compactness).
/// - normalised_perimeter = boundary_pixel_count / sqrt(ink_area)
///   Dividing by sqrt(area) makes it scale-independent.
/// - compactness = 4π × area / perimeter² (1.0 = circle).
fn compute_boundary_features(ink_mask: &[bool], w: usize, h: usize, ink_count: f32) -> (f32, f32) {
    if w == 0 || h == 0 || ink_count < 1.0 {
        return (0.0, 0.0);
    }

    let mut boundary_count = 0u32;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !ink_mask[idx] { continue; }

            // Ink pixel borders white if any 4-connected neighbor is white or out-of-bounds
            let is_boundary = (x == 0 || !ink_mask[idx - 1])
                || (x + 1 >= w || !ink_mask[idx + 1])
                || (y == 0 || !ink_mask[idx - w])
                || (y + 1 >= h || !ink_mask[idx + w]);

            if is_boundary {
                boundary_count += 1;
            }
        }
    }

    let perim = boundary_count as f32;
    let norm_perimeter = perim / ink_count.sqrt();
    let compactness = if perim > 0.0 {
        (4.0 * std::f32::consts::PI * ink_count) / (perim * perim)
    } else {
        0.0
    };

    // Cap normalised perimeter to a reasonable range for feature vector
    let norm_perimeter_capped = (norm_perimeter / 10.0).min(1.0);

    (norm_perimeter_capped, compactness)
}

/// Horizontal crossings at 4 evenly-spaced scan lines.
///
/// For each scan line, counts the number of ink→white or white→ink transitions.
/// Normalised by dividing by 20 (a reasonable cap for complex chars).
fn compute_h_crossings(ink_mask: &[bool], w: usize, h: usize) -> [f32; CROSSING_BINS] {
    let mut crossings = [0.0f32; CROSSING_BINS];
    if w == 0 || h == 0 {
        return crossings;
    }

    for bin in 0..CROSSING_BINS {
        // Evenly spaced: first at 1/(N+1), last at N/(N+1) of height
        let y = ((bin + 1) as f32 * h as f32 / (CROSSING_BINS + 1) as f32) as usize;
        let y = y.min(h - 1);

        let mut transitions = 0u32;
        let mut prev_ink = false;
        for x in 0..w {
            let is_ink = ink_mask[y * w + x];
            if is_ink != prev_ink && x > 0 {
                transitions += 1;
            }
            prev_ink = is_ink;
        }
        // Normalise (most chars have ≤10 transitions; 20 is generous cap)
        crossings[bin] = (transitions as f32 / 20.0).min(1.0);
    }

    crossings
}

// ---------------------------------------------------------------------------
// v3 discriminative feature helpers
// ---------------------------------------------------------------------------


/// Horizontal and vertical symmetry scores.
/// Measures how similar the ink mask is to its mirror image.
fn compute_symmetry(ink_mask: &[bool], w: usize, h: usize) -> (f32, f32) {
    if w < 2 || h < 2 {
        return (0.0, 0.0);
    }

    // Horizontal symmetry: compare left-right mirror
    let mut h_match = 0u32;
    let mut h_total = 0u32;
    for y in 0..h {
        for x in 0..w / 2 {
            let mirror_x = w - 1 - x;
            let left = ink_mask[y * w + x];
            let right = ink_mask[y * w + mirror_x];
            h_total += 1;
            if left == right {
                h_match += 1;
            }
        }
    }

    // Vertical symmetry: compare top-bottom mirror
    let mut v_match = 0u32;
    let mut v_total = 0u32;
    for y in 0..h / 2 {
        let mirror_y = h - 1 - y;
        for x in 0..w {
            let top = ink_mask[y * w + x];
            let bottom = ink_mask[mirror_y * w + x];
            v_total += 1;
            if top == bottom {
                v_match += 1;
            }
        }
    }

    let h_sym = if h_total > 0 { h_match as f32 / h_total as f32 } else { 0.0 };
    let v_sym = if v_total > 0 { v_match as f32 / v_total as f32 } else { 0.0 };
    (h_sym, v_sym)
}

/// Skeleton topology features via Zhang-Suen thinning.
/// Returns (branch_points / 10, endpoints / 10), both clamped to [0, 1].
fn compute_skeleton_features(ink_mask: &[bool], w: usize, h: usize) -> (f32, f32) {
    if w < 3 || h < 3 {
        return (0.0, 0.0);
    }

    // Copy mask to mutable binary image for thinning
    let total = w * h;
    let mut img: Vec<u8> = ink_mask.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();

    // Use a bitfield for marking pixels to remove (avoids Vec allocation per iteration)
    let flag_len = (total + 63) / 64;
    let mut to_remove = vec![0u64; flag_len];

    // Precompute lookup table for transition count (8-bit neighborhood pattern → transitions)
    let trans_lut: [u8; 256] = {
        let mut lut = [0u8; 256];
        for pattern in 0..256u16 {
            let mut t = 0u8;
            for i in 0..8 {
                let curr = (pattern >> i) & 1;
                let next = (pattern >> ((i + 1) % 8)) & 1;
                if curr == 0 && next == 1 { t += 1; }
            }
            lut[pattern as usize] = t;
        }
        lut
    };

    // Zhang-Suen thinning algorithm
    loop {
        let mut changed = false;

        // Step 1
        for w64 in to_remove.iter_mut() { *w64 = 0; }
        for y in 1..h - 1 {
            let row = y * w;
            let prev_row = (y - 1) * w;
            let next_row = (y + 1) * w;
            for x in 1..w - 1 {
                let idx = row + x;
                if img[idx] == 0 { continue; }
                let p2 = img[prev_row + x];
                let p3 = img[prev_row + x + 1];
                let p4 = img[row + x + 1];
                let p5 = img[next_row + x + 1];
                let p6 = img[next_row + x];
                let p7 = img[next_row + x - 1];
                let p8 = img[row + x - 1];
                let p9 = img[prev_row + x - 1];

                let neighbors = p2 as i32 + p3 as i32 + p4 as i32 + p5 as i32
                    + p6 as i32 + p7 as i32 + p8 as i32 + p9 as i32;
                if neighbors < 2 || neighbors > 6 { continue; }

                let pattern = (p2 as u16) | ((p3 as u16) << 1) | ((p4 as u16) << 2) | ((p5 as u16) << 3)
                    | ((p6 as u16) << 4) | ((p7 as u16) << 5) | ((p8 as u16) << 6) | ((p9 as u16) << 7);
                if trans_lut[pattern as usize] != 1 { continue; }

                if p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0 { continue; }
                to_remove[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        for idx in 0..total {
            if to_remove[idx / 64] & (1u64 << (idx % 64)) != 0 {
                img[idx] = 0;
                changed = true;
            }
        }

        // Step 2
        for w64 in to_remove.iter_mut() { *w64 = 0; }
        for y in 1..h - 1 {
            let row = y * w;
            let prev_row = (y - 1) * w;
            let next_row = (y + 1) * w;
            for x in 1..w - 1 {
                let idx = row + x;
                if img[idx] == 0 { continue; }
                let p2 = img[prev_row + x];
                let p3 = img[prev_row + x + 1];
                let p4 = img[row + x + 1];
                let p5 = img[next_row + x + 1];
                let p6 = img[next_row + x];
                let p7 = img[next_row + x - 1];
                let p8 = img[row + x - 1];
                let p9 = img[prev_row + x - 1];

                let neighbors = p2 as i32 + p3 as i32 + p4 as i32 + p5 as i32
                    + p6 as i32 + p7 as i32 + p8 as i32 + p9 as i32;
                if neighbors < 2 || neighbors > 6 { continue; }

                let pattern = (p2 as u16) | ((p3 as u16) << 1) | ((p4 as u16) << 2) | ((p5 as u16) << 3)
                    | ((p6 as u16) << 4) | ((p7 as u16) << 5) | ((p8 as u16) << 6) | ((p9 as u16) << 7);
                if trans_lut[pattern as usize] != 1 { continue; }

                if p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0 { continue; }
                to_remove[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        for idx in 0..total {
            if to_remove[idx / 64] & (1u64 << (idx % 64)) != 0 {
                img[idx] = 0;
                changed = true;
            }
        }

        if !changed { break; }
    }

    // Count endpoints (1 neighbor) and branch points (≥3 neighbors)
    let mut endpoints = 0u32;
    let mut branch_pts = 0u32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if img[y * w + x] == 0 { continue; }
            let mut neighbors = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    if img[ny * w + nx] == 1 {
                        neighbors += 1;
                    }
                }
            }
            if neighbors == 1 { endpoints += 1; }
            if neighbors >= 3 { branch_pts += 1; }
        }
    }

    ((branch_pts as f32 / 10.0).min(1.0), (endpoints as f32 / 10.0).min(1.0))
}

/// Corner count using boundary direction changes.
/// Walks the ink boundary and counts points where direction changes sharply.
fn compute_corner_count(ink_mask: &[bool], w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }

    // Collect boundary pixels (ink pixels with at least one non-ink 4-neighbor)
    let mut boundary: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !ink_mask[y * w + x] { continue; }
            let mut on_boundary = false;
            for (dx, dy) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || (nx as usize) >= w || (ny as usize) >= h {
                    on_boundary = true;
                    break;
                }
                if !ink_mask[ny as usize * w + nx as usize] {
                    on_boundary = true;
                    break;
                }
            }
            if on_boundary {
                boundary.push((x, y));
            }
        }
    }

    if boundary.len() < 5 {
        return 0.0;
    }

    // Simple corner detection: for each boundary pixel, look at neighbors
    // in the boundary that are within distance 3 and measure angle change.
    // More practical approach: count pixels where the local curvature is high
    // using a simple method based on the area of the triangle formed by
    // a point and its k-neighbors on the boundary.
    //
    // Use a simpler approach: count ink pixels that have exactly 1 or 2
    // diagonal ink neighbors but form an L-shape (corner of the glyph outline).
    let mut corners = 0u32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if !ink_mask[y * w + x] { continue; }
            // Check if this is a boundary pixel
            let up = if y > 0 { ink_mask[(y - 1) * w + x] } else { false };
            let down = if y + 1 < h { ink_mask[(y + 1) * w + x] } else { false };
            let left = if x > 0 { ink_mask[y * w + (x - 1)] } else { false };
            let right = if x + 1 < w { ink_mask[y * w + (x + 1)] } else { false };

            // L-shape patterns: exactly 2 adjacent cardinal neighbors that are
            // perpendicular (forming a corner)
            let cardinal = [up, right, down, left];
            let n_cardinal: usize = cardinal.iter().filter(|&&b| b).count();
            if n_cardinal != 2 { continue; }

            // Check if the two neighbors are perpendicular (not opposite)
            let is_corner = (up && right) || (right && down) || (down && left) || (left && up);
            if is_corner {
                // Check the diagonal to confirm it's a true corner (no diagonal fill)
                let diag_filled = match (up, right, down, left) {
                    (true, true, _, _) => ink_mask[(y - 1) * w + (x + 1)],
                    (_, true, true, _) => ink_mask[(y + 1) * w + (x + 1)],
                    (_, _, true, true) => ink_mask[(y + 1) * w + (x - 1)],
                    (true, _, _, true) => ink_mask[(y - 1) * w + (x - 1)],
                    _ => true,
                };
                if !diag_filled {
                    corners += 1;
                }
            }
        }
    }

    (corners as f32 / 20.0).min(1.0)
}

/// Ink density in each quadrant (TL, TR, BL, BR).
fn compute_quadrant_density(ink_mask: &[bool], w: usize, h: usize) -> [f32; 4] {
    if w == 0 || h == 0 {
        return [0.0; 4];
    }
    let mid_x = w / 2;
    let mid_y = h / 2;

    let mut counts = [0u32; 4];  // TL, TR, BL, BR
    let mut areas = [0u32; 4];

    for y in 0..h {
        for x in 0..w {
            let q = if y < mid_y {
                if x < mid_x { 0 } else { 1 }
            } else {
                if x < mid_x { 2 } else { 3 }
            };
            areas[q] += 1;
            if ink_mask[y * w + x] {
                counts[q] += 1;
            }
        }
    }

    let mut density = [0.0f32; 4];
    for i in 0..4 {
        density[i] = if areas[i] > 0 { counts[i] as f32 / areas[i] as f32 } else { 0.0 };
    }
    density
}

/// Linearly resample a 1-D signal to `n` bins.
fn resample(src: &[f32], n: usize) -> [f32; PROFILE_BINS] {
    let mut out = [0.0f32; PROFILE_BINS];
    if src.is_empty() || n == 0 {
        return out;
    }
    let src_len = src.len() as f32;
    for i in 0..n {
        let pos = i as f32 * (src_len - 1.0) / (n as f32 - 1.0).max(1.0);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(src.len() - 1);
        let frac = pos - lo as f32;
        out[i] = src[lo] * (1.0 - frac) + src[hi] * frac;
    }
    out
}

// ---------------------------------------------------------------------------
// Shared normalisation: tight ink crop → NORM_H scale
// ---------------------------------------------------------------------------

/// Per-character percentile-based contrast normalization.
///
/// Maps the 1st-percentile pixel value → 0 and the 99th-percentile → 255.
/// Applied to scan-side character crops so their dynamic range matches
/// Compute the p1/p99 percentile stretch range from a luminance histogram.
/// Returns `Some((p1, p99))` when a valid stretch exists, `None` when flat.
fn contrast_percentiles(luma: &[u8]) -> Option<(u8, u8)> {
    if luma.is_empty() { return None; }
    let mut hist = [0u32; 256];
    for &px in luma { hist[px as usize] += 1; }
    let n = luma.len() as u32;
    let p1_target = n / 100;
    let p99_target = n * 99 / 100;
    let mut cum = 0u32;
    let (mut p1, mut p99) = (0u8, 255u8);
    let mut found_p1 = false;
    for (val, &count) in hist.iter().enumerate() {
        cum += count;
        if !found_p1 && cum >= p1_target { p1 = val as u8; found_p1 = true; }
        if cum >= p99_target { p99 = val as u8; break; }
    }
    if p1 < p99 { Some((p1, p99)) } else { None }
}

/// Apply a p1/p99 stretch to a single channel value.
#[inline]
fn stretch(v: u8, p1: u8, range: f32) -> u8 {
    ((v as f32 - p1 as f32) * 255.0 / range).round().clamp(0.0, 255.0) as u8
}

/// Contrast-normalize a grayscale image via p1/p99 percentile stretch.
/// This ensures symmetric treatment between training renders and inference
/// crops: clean renders are effectively a no-op (p1≈0, p99≈255), while
/// scanned/compressed pages get stretched to match
/// the full-range black-on-white rendered index characters, regardless
/// of how the source PDF was rasterized or compressed.
pub fn contrast_normalize_char(mut img: GrayImage) -> GrayImage {
    let Some((p1, p99)) = contrast_percentiles(img.as_raw()) else {
        return img;
    };
    let range = (p99 - p1) as f32;
    for px in img.as_mut() {
        *px = stretch(*px, p1, range);
    }
    img
}

/// Contrast-normalize an RGBA image, preserving colour.
/// Computes the stretch from luminance, applies it to R/G/B channels.
pub fn contrast_normalize_rgba(img: &image::RgbaImage) -> image::RgbaImage {
    let gray = image::imageops::grayscale(img);
    let Some((p1, p99)) = contrast_percentiles(gray.as_raw()) else {
        return img.clone();
    };
    let range = (p99 - p1) as f32;
    let mut out = img.clone();
    for px in out.pixels_mut() {
        for c in 0..3 {
            px[c] = stretch(px[c], p1, range);
        }
    }
    out
}

fn collect_ink_runs(img: &GrayImage) -> Vec<u32> {
    let (w, h) = img.dimensions();
    let threshold = 200u8;
    let mut all_runs: Vec<u32> = Vec::new();

    for y in 0..h {
        let mut run = 0u32;
        for x in 0..w {
            if img.as_raw()[(y) as usize * img.width() as usize + (x) as usize] < threshold {
                run += 1;
            } else {
                if run >= 2 { all_runs.push(run); }
                run = 0;
            }
        }
        if run >= 2 { all_runs.push(run); }
    }

    for x in 0..w {
        let mut run = 0u32;
        for y in 0..h {
            if img.as_raw()[(y) as usize * img.width() as usize + (x) as usize] < threshold {
                run += 1;
            } else {
                if run >= 2 { all_runs.push(run); }
                run = 0;
            }
        }
        if run >= 2 { all_runs.push(run); }
    }

    all_runs
}

fn measure_stroke_contrast(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    if w < 4 || h < 4 {
        return 1.0;
    }

    let all_runs = collect_ink_runs(img);
    if all_runs.len() < 4 {
        return 1.0;
    }

    let mut sorted = all_runs;
    sorted.sort_unstable();
    let p10 = sorted[sorted.len() / 10].max(1);
    let p90 = sorted[sorted.len() * 9 / 10].max(1);

    let ratio = p90 as f32 / p10 as f32;
    // Normalize to [0,1]: 1.0 (uniform) → 0.0, high contrast → ~1.0
    1.0 - 1.0 / ratio
}

/// Mean stroke width normalised by image height.
///
/// Captures absolute stroke heaviness — the main signal that distinguishes
/// weight classes (Light 300 / Regular 400 / Medium 500 / Bold 700).
/// `stroke_contrast` captures the *ratio* of thick-to-thin strokes, which is
/// similar across weights of the same family; `mean_stroke_width` captures the
/// actual thickness.
fn measure_mean_stroke_width(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    if w < 4 || h < 4 {
        return 0.0;
    }

    let all_runs = collect_ink_runs(img);
    if all_runs.is_empty() {
        return 0.0;
    }

    let sum: f64 = all_runs.iter().map(|&r| r as f64).sum();
    let mean = sum / all_runs.len() as f64;

    // Normalise by image height so the feature is scale-invariant
    (mean / h as f64) as f32
}

/// Aggregate per-observation weighted log-probabilities into a single font score.
/// Used by both `identify_fonts` and GT-font injection — one formula,
/// one implementation.  Higher = better (higher probability = better match).
///
/// Input: `(ln(prob), weight)` pairs for each matched observation (unigram or bigram).
/// Missing glyphs (matched < n_windows) get probability 0 → ln(0) = −∞,
/// guaranteeing a font that can't render an observation can never win.

pub fn normalize_to_ink_bounds(img: &GrayImage, target_h: u32) -> Option<GrayImage> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    const THRESH: u8 = 200;
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;
    for y in 0..h {
        for x in 0..w {
            if img.as_raw()[(y) as usize * img.width() as usize + (x) as usize] < THRESH {
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }
    }
    if min_x > max_x || min_y > max_y {
        return None;
    }
    // Crop tightly to ink, then paste onto a white canvas with guaranteed
    // 1px padding on all sides — matching index-time render_char_normalised
    // which creates a canvas of (ink_w+2) × (ink_h+2).
    // Previous approach used saturating_sub which clipped at image boundary,
    // producing 0px padding when ink reached the edge of the raw slice.
    let ink_w = max_x - min_x + 1;
    let ink_h = max_y - min_y + 1;
    if ink_w < 1 || ink_h < 1 {
        return None;
    }
    let pad = 1u32;
    let canvas_w = ink_w + 2 * pad;
    let canvas_h = ink_h + 2 * pad;
    let mut canvas = GrayImage::from_pixel(canvas_w, canvas_h, Luma([255u8]));
    // Copy ink region into canvas at (pad, pad) — raw blit
    {
        let src_w = img.width() as usize;
        let dst_w = canvas_w as usize;
        let src_raw = img.as_raw();
        let dst_raw = canvas.as_mut();
        let copy_w = (max_x - min_x + 1) as usize;
        for y in min_y..=max_y {
            let src_off = y as usize * src_w + min_x as usize;
            let dst_off = (y - min_y + pad) as usize * dst_w + pad as usize;
            dst_raw[dst_off..dst_off+copy_w].copy_from_slice(&src_raw[src_off..src_off+copy_w]);
        }
    }
    let scaled_w = (canvas_w as f32 * target_h as f32 / canvas_h as f32).ceil() as u32;
    if scaled_w < 2 {
        return None;
    }
    Some(image::imageops::resize(
        &canvas,
        scaled_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
    ))
}

// ---------------------------------------------------------------------------
// Character extraction from scan
// ---------------------------------------------------------------------------
