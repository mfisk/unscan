//! GPOS pair kerning for rendering (Finding 5).
//!
//! The ngram/word renderers must space adjacent glyphs with the same GPOS
//! pair kern the geometry predictor uses. The legacy `kern` table (what
//! `ab_glyph`'s `sf.kern()` reads) is empty for GPOS-only fonts, so renders
//! built on it are unkerned while the shaped text and the GT render are
//! kerned. This trait is the single source of pair kern for every render
//! path: a renderer takes a `&dyn KernProvider` (required, never `Option`)
//! and uses it for every adjacent pair. There is no fallback to the legacy
//! `kern` table anywhere in the render path.

/// Provider of GPOS pair kerning for one font, in font units.
///
/// The provider is bound to a single font (the font whose kern it reports);
/// a renderer that iterates many fonts constructs one provider per font.
/// The trait is object-safe and `Sync`, so a `&dyn KernProvider` can be
/// shared across rayon threads.
pub trait KernProvider: Sync {
    /// GPOS pair kern for adjacent glyphs `(gid1, gid2)`, in font units.
    ///
    /// This is the scalar added to `gid1`'s advance when followed by `gid2`:
    /// `gid1`'s XAdvance adjustment plus `gid2`'s XPlacement adjustment, as
    /// read from the font's GPOS PairPos tables (variation-aware: the value
    /// is for the font's instantiated variation, per Finding 1). It is the
    /// same quantity the geometry predictor folds into the inter-glyph pitch
    /// (`v1[2]` via the first glyph's advance adjustment, `v2[0]` via the
    /// second glyph's placement).
    ///
    /// Returns 0 when the pair has no kern or the font has no kern data.
    fn pair_kern_fu(&self, gid1: u16, gid2: u16) -> i32;
}

/// A provider that reports no kerning for any pair.
///
/// Use it only where kerning is genuinely undefined, never as a silent
/// fallback: single-glyph renders (a single glyph has no pair, so the
/// provider is never consulted) and display-only renders that have no geo
/// cache available. A render path that has kern data must pass the real
/// provider; passing `NoKern` there would silently unkern it.
pub struct NoKern;

impl KernProvider for NoKern {
    fn pair_kern_fu(&self, _gid1: u16, _gid2: u16) -> i32 {
        0
    }
}
