use image::GrayImage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphId(pub u16);

impl GlyphId {
    pub fn new(id: u16) -> Self { Self(id) }
    pub fn get(self) -> u16 { self.0 }
}

#[derive(Clone, Copy, Debug)]
pub struct Bbox {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GlyphBBox {
    pub gid: u16,
    pub bbox: Option<Bbox>,
}

#[derive(Clone, Debug)]
pub struct RenderParams {
    pub height: u32,
    pub render_scale: u32,
    pub aa: AaMode,
    pub binarize_threshold: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaMode {
    Native,
    None,
    Mono,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            height: 24,
            render_scale: 1,
            aa: AaMode::Native,
            binarize_threshold: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderResult {
    pub seq_hash: u64,
    pub image: GrayImage,
    pub width: u32,
    pub height: u32,
}

impl RenderResult {
    pub fn new(image: GrayImage, seq_hash: u64) -> Self {
        let (w, h) = (image.width(), image.height());
        Self { seq_hash, image, width: w, height: h }
    }
}

// Shaping types
#[derive(Clone, Debug)]
pub struct ShapedGlyph {
    pub gid: u16,
    pub cluster: u32,
    pub x_advance: f32,
    pub y_advance: f32,
    pub x_offset: f32,
    pub y_offset: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ShapedRun {
    pub glyphs: Vec<ShapedGlyph>,
}

#[derive(Clone, Debug)]
pub struct ShapedWord {
    pub text: String,
    pub glyphs: Vec<ShapedGlyph>,
    pub total_advance: f32,
}

pub type FontKey = String;

#[derive(Clone, Debug)]
pub struct FontMeta {
    pub family_name: String,
    pub postscript_name: Option<String>,
    pub is_monospace: bool,
    pub has_liga: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureTag(pub [u8; 4]);

impl FeatureTag {
    pub fn from_bytes(b: &[u8; 4]) -> Self { Self(*b) }
    pub fn to_string(&self) -> String { String::from_utf8_lossy(&self.0).to_string() }
}

#[derive(Clone, Debug)]
pub struct Variation {
    pub tag: [u8; 4],
    pub value: f32,
}

#[derive(Clone, Debug)]
pub struct KerningPair {
    pub left: u16,
    pub right: u16,
    pub kern: f32,
}

pub type KerningPairList = Vec<KerningPair>;

// Re-export misc
pub use RenderResult as ImageResult;
