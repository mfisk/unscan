//! Common tiny params that everything depends on.

use std::sync::OnceLock;

// First-principles expected error from uniform quantization:
pub const SIGMA_CENTER_THEORETICAL: f64 = 0.28867513459481287;
pub const SIGMA_PITCH_THEORETICAL: f64 = 0.40824829046386302;
pub const SIGMA_CENTER_TUNED: f64 = 0.284;
pub const SIGMA_PITCH_TUNED: f64 = 0.435;

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
/// Center σ=0.2887 → thresh 0.2165, pitch σ=0.4082 → thresh 0.3062
/// |e| < 0.75σ: -0.5*(e/2σ)²  → 0.2px center -0.06, pitch -0.03 (almost free)
/// |e| ≥ 0.75σ: -0.5*(e/σ)² + 0.2109375 → continuous at -0.0703, 0.6px center -1.95, pitch -0.87
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
