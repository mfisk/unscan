//! Common tiny params that everything depends on.

use std::sync::OnceLock;

// Canonical ink threshold: `< INK_THRESHOLD` is ink, `>= INK_THRESHOLD` is paper/whitespace.
// Single source of truth for both scan-side trimming (`trim_words_to_ink`,
// `ink_horizontal_extent`, `ink_vertical_extent`) and render-side cropping
// (`verify.rs`). Must be 255 to keep 1px AA fringe (212,230) as ink and avoid
// the 1px left-shift ZNCC failure seen in p2:L84 (diff blue-left/red-right).
pub const INK_THRESHOLD: u8 = 255;

// Theoretical geo error model — subpixel measurement (Nov 20, 2026).
//
// The old model (1/sqrt(12) center, 1/sqrt(6) pitch) assumed whole-pixel
// uniform quantization: each edge measured to the nearest pixel. Subpixel
// measurement (stage 1, Nov 15) measures fractional edges from darkness
// ratios, so the quantization model no longer applies. The revised model
// derives each sigma from the actual estimator and imaging chain, and
// checks it against the measured error distribution (BAP 1364 audit,
// 12,203 ground-truth observations on ocr_correct lines; t64 suite).
//
// CENTER sigma — the error of one measured glyph center (midpoint of two
// fractional edges). Terms, added in quadrature:
//   - Gamma-mismatch bias (rendered case), sigma = 0.0599 px: the pipeline
//     linearizes page pixels with a gamma-2.2 power law, which is correct
//     for gamma-encoded scanner output but wrong for the already-linear
//     MuPDF/Poppler renders used in the specimen: a true coverage c is read
//     as c^2.2, so a left/top edge errs by (c - c^2.2) and a right/bottom
//     edge by (c^2.2 - c). For c ~ U(0,1): E[c - c^2.2] = 1/2 - 1/3.2,
//     Var = 1/3 - 2/4.2 + 1/5.4 - (1/2 - 1/3.2)^2 = 0.00717, so each edge
//     has sigma 0.0847 px; the center averages two independent edges, so
//     sigma_center = 0.0847/sqrt(2) = 0.0599 px. (Verified numerically.)
//   - Intensity quantization: Lyvers et al. 1989 RMS 1/(2*sqrt(3)*255) =
//     0.0011 px — negligible.
//   Total: sqrt(0.0599^2 + 0.0011^2) = 0.0599 px.
//   Check against measurement: BAP 1364 vertical filtered sd 0.0515,
//   t64 suite 0.0737; the model sits between them.
//   Scanned-print budget (the operational target; the gamma term does not
//   apply there because scanner output IS gamma-encoded, so linearization
//   is correct): scanner-PSF bias of the linear estimator (computed from
//   the Gaussian-PSF edge model: center RMS 0.038/0.074/0.089 px at
//   sigma_PSF 0.3/0.5/0.6 px) + ink raggedness 0.04-0.08 px/edge (ISO 13660;
//   /sqrt(2) for the center) + scanner-noise CRLB 0.01-0.03 px +
//   quantization 0.001 px -> 0.06-0.09 px. The adopted 0.0599 is at the low
//   end of the scan budget: exact for the renders the BAP measures,
//   slightly optimistic for the noisiest scans.
//
// PITCH sigma — the error of the pitch between two adjacent centers.
// Pitch = center_2 - center_1, so it carries the center error twice, plus
// two further terms measured from the BAP 1364 audit (pair decomposition
// of gt_geo_h_err over 9,843 observations; ~half the pitch variance is a
// stable per-pair property, half is per-occurrence noise):
//   - Stable per-glyph offset (sigma_glyph = 0.121 px): the ink center of
//     a glyph differs from its outline-bbox center by a stable per-glyph
//     amount (shape asymmetry; largest in italics, ~nil in monospace).
//     Two glyphs contribute 2*sigma_glyph^2. This is a prediction-model
//     limitation (the prediction uses bbox centers), not a measurement
//     error; predicting ink centers would remove it.
//   - Per-occurrence context noise (sigma_ctx = 0.150 px, pitch-level):
//     seam placement and word-context variation of the measured centers,
//     beyond the estimator term. One term per pitch.
//   Total: sqrt(2*0.0599^2 + 2*0.121^2 + 0.150^2) = 0.243 px.
//   Check: BAP 1364 pitch filtered sd 0.2428; t64 suite 0.2697.
//   Note the pitch is NOT sqrt(2) times the center (as in the old model):
//   the glyph and context terms dominate it, so the pitch sigma is ~4x the
//   center sigma, not 1.41x. The pitch error does not scale with pitch
//   magnitude (corr -0.008), so no scale-error term is included.
pub const SIGMA_CENTER_THEORETICAL: f64 = 0.0599;
pub const SIGMA_PITCH_THEORETICAL: f64 = 0.243;

pub const SIGMA_CENTER_PX: f64 = SIGMA_CENTER_THEORETICAL;
pub const SIGMA_PITCH_PX: f64 = SIGMA_PITCH_THEORETICAL;

pub const FLAT_CENTER_THEORETICAL: f64 = SIGMA_CENTER_THEORETICAL;
pub const FLAT_PITCH_THEORETICAL: f64 = SIGMA_PITCH_THEORETICAL;
pub const FLAT_TOP_DEFAULT: f64 = 0.3375; // 0.45 * 0.75
pub const FLAT_CENTER_DEFAULT: f64 = FLAT_TOP_DEFAULT;
pub const FLAT_PITCH_DEFAULT: f64 = FLAT_TOP_DEFAULT;

static FLAT_TOP_CACHE: OnceLock<f64> = OnceLock::new();
static FLAT_TOP_PITCH_CACHE: OnceLock<f64> = OnceLock::new();

#[inline]
pub fn quant_half_width_center_px() -> f64 {
    *FLAT_TOP_CACHE.get_or_init(|| {
        std::env::var("UNPRINT_FLAT_TOP")
            .or_else(|_| std::env::var("QUANT_HALF_WIDTH_PX"))
            .or_else(|_| std::env::var("FLAT_TOP"))
            .or_else(|_| std::env::var("QUANT_HALF_WIDTH"))
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&v| v > 0.0 && v < 10.0)
            .unwrap_or(FLAT_CENTER_DEFAULT)
    })
}

#[inline]
pub fn quant_half_width_pitch_px() -> f64 {
    *FLAT_TOP_PITCH_CACHE.get_or_init(|| {
        if let Ok(s) = std::env::var("UNPRINT_FLAT_TOP_PITCH") {
            if let Ok(v) = s.parse::<f64>() {
                if v > 0.0 && v < 10.0 {
                    return v;
                }
            }
        }
        FLAT_PITCH_DEFAULT
    })
}

#[inline]
pub fn quant_half_width_px() -> f64 {
    quant_half_width_center_px()
}

/// Conditional: inflection at 0.75*σ per dimension, inside 2σ (flat), outside 1σ (theoretical), continuous.
///
/// Center σ=0.0599 → thresh 0.0449, pitch σ=0.243 → thresh 0.1822
/// |e| < 0.75σ: -0.5*(e/2σ)²  → 0.04px center -0.06, pitch -0.003 (almost free)
/// |e| ≥ 0.75σ: -0.5*(e/σ)² + 0.2109375 → continuous at -0.0703, 0.2px center -5.36, pitch -0.13
#[inline]
pub fn quantized_ll(e: f64, sigma: f64, _half_width: f64) -> f64 {
    let sigma = sigma.max(1e-12);
    let thresh = 0.75 * sigma;
    let e_abs = e.abs();
    const K_INNER: f64 = 2.0;
    if e_abs < thresh {
        let si = K_INNER * sigma;
        -0.5 * (e / si) * (e / si)
    } else {
        // constants: inner(thresh) = -0.5*(0.75/K)² = -0.0703125, outer_raw(thresh) = -0.28125, offset = 0.2109375
        const OFFSET: f64 = 0.2109375;
        -0.5 * (e / sigma) * (e / sigma) + OFFSET
    }
}

static AUDIT_ALL_CHARS_CACHE: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn audit_all_chars_enabled() -> bool {
    *AUDIT_ALL_CHARS_CACHE.get_or_init(|| {
        let v = std::env::var("UNPRINT_AUDIT_ALL_CHARS").unwrap_or_default();
        matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}
