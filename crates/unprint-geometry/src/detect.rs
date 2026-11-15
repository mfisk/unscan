pub type Rgb = (u8, u8, u8);

#[derive(Debug, Clone)]
pub struct DetectedLine {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
    pub thickness: u32,
    pub color: Rgb,
}

#[derive(Debug, Clone)]
pub struct DetectedFill {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub color: Rgb,
}

#[derive(Debug, Clone)]
pub struct GeometryResult {
    pub lines: Vec<DetectedLine>,
    pub fills: Vec<DetectedFill>,
}
