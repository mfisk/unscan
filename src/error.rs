use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScanTextError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image load error: {0}")]
    ImageLoad(String),

    #[error("Unsupported file format: .{0}")]
    UnsupportedFormat(String),

    #[error("OCR error: {0}")]
    Ocr(String),

    #[error("PDF generation error: {0}")]
    PdfGen(String),

    #[error(
        "No fonts found on the system.\n\
         For Microsoft fonts:  apt install ttf-mscorefonts-installer\n\
         For LaTeX fonts:      apt install fonts-lmodern texlive-fonts-recommended\n\
         Or supply --font-dir pointing at a directory of .ttf / .otf files."
    )]
    NoFonts,

    #[error("Serialization error: {0}")]
    Serialize(String),
}
