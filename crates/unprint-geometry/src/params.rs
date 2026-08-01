//! Common tiny params that everything depends on.
//! Centralizes flat-top quantized likelihood constants and env flags
//! to avoid duplication across crates/unprint-core, src/, and
//! unprint-geometry itself. No heavy deps.

use std::sync::OnceLock;

// First-principles expected error from uniform quantization:
// Var[uniform -0.5..0.5] = 1/12  => sigma = 1/sqrt(12) ≈ 0.2887
// Pitch variance doubles => 1/6 => 1/sqrt(6) ≈ 0.4082
// Tuned values from sweep win over theory: 0.284 / 0.435 at default flat-top 0.45
pub const SIGMA_CENTER_PX: f64 = 0.284;
pub const SIGMA_PITCH_PX: f64 = 0.435;

static FLAT_TOP_CACHE: OnceLock<f64> = OnceLock::new();

/// Flat-top half-width in px. Env override `UNPRINT_FLAT_TOP` (compat: `QUANT_HALF_WIDTH_PX`|`FLAT_TOP`|`QUANT_HALF_WIDTH`), filtered 0<a<10, default 0.45.
#[inline]
pub fn quant_half_width_px() -> f64 {
    *FLAT_TOP_CACHE.get_or_init(|| {
        std::env::var("UNPRINT_FLAT_TOP")
            .or_else(|_| std::env::var("QUANT_HALF_WIDTH_PX"))
            .or_else(|_| std::env::var("FLAT_TOP"))
            .or_else(|_| std::env::var("QUANT_HALF_WIDTH"))
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&v| v > 0.0 && v < 10.0)
            .unwrap_or(0.45)
    })
}

/// Quantized likelihood: ln[ Φ((e+a)/σ) - Φ((e-a)/σ) ] - ln(2a)
/// Φ via libm::erf. a = half-width, σ = per-axis sigma.
#[inline]
pub fn quantized_ll(e: f64, sigma: f64, half_width: f64) -> f64 {
    let sigma = sigma.max(1e-12);
    let a = half_width;
    let upper = (e + a) / sigma;
    let lower = (e - a) / sigma;
    const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let phi_upper = 0.5 * (1.0 + libm::erf(upper * FRAC_1_SQRT_2));
    let phi_lower = 0.5 * (1.0 + libm::erf(lower * FRAC_1_SQRT_2));
    let prob = (phi_upper - phi_lower).max(1e-300);
    prob.ln() - (2.0 * a).ln()
}

static AUDIT_ALL_CHARS_CACHE: OnceLock<bool> = OnceLock::new();

/// Env toggle for full-char audit logging. Checks `UNPRINT_AUDIT_ALL_CHARS` then `UNPRINT_AUDIT_ALL`.
/// Accepts 1|true|yes|on case-insensitive. Defaults off.
#[inline]
pub fn audit_all_chars_enabled() -> bool {
    *AUDIT_ALL_CHARS_CACHE.get_or_init(|| {
        let v = std::env::var("UNPRINT_AUDIT_ALL_CHARS")
            .or_else(|_| std::env::var("UNPRINT_AUDIT_ALL"))
            .unwrap_or_default();
        matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}
