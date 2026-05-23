//! Shared helpers for unscan integration tests (t30, t40, t50).
//!
//! Contains CLI binary resolution, fixture generation, output parsers,
//! and run wrappers used across the end-to-end test suites.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

pub const EB_GARAMOND: &str = "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf";

// ── Shared index pre-build ───────────────────────────────────────────

static INDEX_ONCE: Once = Once::new();

/// Ensure the character index exists before any test spawns unscan.
/// Uses `Once` so parallel test threads only build it once; the rest block
/// until it's ready, then all hit the cached file.
pub fn ensure_index() {
    INDEX_ONCE.call_once(|| {
        let bin = unscan_bin();
        eprintln!("[test setup] Pre-building character index via {:?} --index", bin);
        let output = Command::new(&bin)
            .arg("--index")
            .env("RUST_LOG", "info")
            .output()
            .unwrap_or_else(|e| panic!("failed to run {:?} --index: {}", bin, e));
        if !output.status.success() {
            panic!(
                "Index pre-build failed (exit {}):\n{}{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        eprintln!("[test setup] Index ready.");
    });
}

// ── Binary & path helpers ────────────────────────────────────────────

/// Locate the built `unscan` binary (release or debug).
pub fn unscan_bin() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let release = manifest.join("target/release/unscan");
    if release.exists() {
        return release;
    }
    let debug = manifest.join("target/debug/unscan");
    if debug.exists() {
        return debug;
    }
    panic!(
        "unscan binary not found. Run `cargo build --release` first.\n\
         Checked: {:?} and {:?}",
        release, debug
    );
}

/// Path into the test-docs directory.
pub fn test_doc(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-docs")
        .join(name)
}

// ── Fixture generation helpers ───────────────────────────────────────

/// Run a Python one-liner to generate a fixture. Panics on failure.
pub fn python3(code: &str) {
    let status = Command::new("python3")
        .arg("-c")
        .arg(code)
        .status()
        .expect("failed to launch python3");
    assert!(status.success(), "python3 fixture generation failed");
}

/// Run ImageMagick `convert` to embed a PNG as a PDF at a given DPI.
pub fn convert_png_to_pdf(png: &Path, pdf: &Path, dpi: u32) {
    let status = Command::new("convert")
        .arg(png)
        .args(["-density", &dpi.to_string()])
        .arg(pdf)
        .status()
        .expect("failed to launch convert (ImageMagick)");
    assert!(status.success(), "convert failed for {:?}", pdf);
}

/// Ensure all test fixtures exist, generating any that are missing.
/// Returns false if prerequisites (EB Garamond, python3, convert) are absent.
pub fn ensure_fixtures() -> bool {
    if !Path::new(EB_GARAMOND).exists() {
        eprintln!("SKIP: EB Garamond 12 not installed at {}", EB_GARAMOND);
        return false;
    }

    // punch-hires.pdf — 600 DPI source, "punchcutter" at 120px
    let punch_hires = test_doc("punch-hires.pdf");
    if !punch_hires.exists() {
        eprintln!("  Generating punch-hires.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 120)
img = Image.new('L', (1800, 240), 255)
ImageDraw.Draw(img).text((0, 60), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-hires.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(Path::new("/tmp/_punch-hires.png"), &punch_hires, 600);
    }

    // punch-garamond.pdf — 200 DPI
    let punch_garamond = test_doc("punch-garamond.pdf");
    if !punch_garamond.exists() {
        eprintln!("  Generating punch-garamond.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 120)
img = Image.new('L', (1800, 240), 255)
ImageDraw.Draw(img).text((0, 60), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-garamond.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(
            Path::new("/tmp/_punch-garamond.png"),
            &punch_garamond,
            200,
        );
    }

    // punch-gold.png — native 60px, no pdftoppm
    let punch_gold = test_doc("punch-gold.png");
    if !punch_gold.exists() {
        eprintln!("  Generating punch-gold.png...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 60)
img = Image.new('L', (900, 120), 255)
ImageDraw.Draw(img).text((100, 30), 'punchcutter', font=font, fill=0)
img.save('{}')
"#,
            EB_GARAMOND,
            punch_gold.display()
        ));
    }

    // punch-100dpi-big.pdf — 600 DPI source, 40pt text, loaded at 100 DPI
    let punch_100dpi = test_doc("punch-100dpi-big.pdf");
    if !punch_100dpi.exists() {
        eprintln!("  Generating punch-100dpi-big.pdf...");
        python3(&format!(
            r#"
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('{}', 333)
img = Image.new('L', (5100, 1800), 255)
ImageDraw.Draw(img).text((600, 600), 'punchcutter', font=font, fill=0)
img.save('/tmp/_punch-100dpi-big.png')
"#,
            EB_GARAMOND
        ));
        convert_png_to_pdf(
            Path::new("/tmp/_punch-100dpi-big.png"),
            &punch_100dpi,
            600,
        );
    }

    // specimen-clean-raster.pdf — generated by gen-specimen.py
    let specimen = test_doc("specimen-clean-raster.pdf");
    if !specimen.exists() {
        let gen_script = test_doc("gen-specimen.py");
        if gen_script.exists() {
            eprintln!("  Generating specimen-clean-raster.pdf...");
            let status = Command::new("python3")
                .arg(&gen_script)
                .current_dir(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-docs"),
                )
                .status()
                .expect("python3 gen-specimen.py failed to launch");
            assert!(status.success(), "gen-specimen.py failed");
        } else {
            eprintln!("  SKIP: gen-specimen.py not found");
        }
    }

    true
}

/// Common preamble: ensure fixtures exist or skip.
/// Returns true if tests can proceed.
pub fn setup() -> bool {
    ensure_fixtures()
}

// ── Run wrappers ─────────────────────────────────────────────────────

/// Run unscan with output to /dev/null and return combined stdout+stderr.
pub fn run_unscan(input: &Path, extra_args: &[&str]) -> String {
    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(input)
        .args(["-o", "/dev/null"])
        .args(extra_args)
        .env("RUST_LOG", "info")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

/// Run unscan with a specific output path and return combined stdout+stderr.
pub fn run_unscan_to(input: &Path, output_path: &Path, extra_args: &[&str]) -> String {
    let bin = unscan_bin();
    let output = Command::new(&bin)
        .arg(input)
        .args(["-o", output_path.to_str().unwrap()])
        .args(extra_args)
        .env("RUST_LOG", "info")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

// ── Output parsers ───────────────────────────────────────────────────

/// Extract the first `ssim=X.XXX` value from unscan output.
pub fn parse_ssim(output: &str) -> Option<f64> {
    for line in output.lines() {
        if let Some(pos) = line.find("ssim=") {
            let rest = &line[pos + 5..];
            let num_end = rest
                .find(|c: char| c != '.' && !c.is_ascii_digit())
                .unwrap_or(rest.len());
            return rest[..num_end].parse::<f64>().ok();
        }
    }
    None
}

/// Extract the matched font name from the first `✓` line (the `→ NAME (` part).
pub fn parse_font_match(output: &str) -> Option<String> {
    for line in output.lines() {
        if !line.contains('✓') {
            continue;
        }
        if let Some(arrow_pos) = line.find('→') {
            // "→ " is 4 bytes (UTF-8 arrow + space)
            let after_arrow = &line[arrow_pos + 4..];
            if let Some(paren) = after_arrow.find('(') {
                let name = after_arrow[..paren].trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

/// Extract the "Text lines vectorised: N" count.
pub fn parse_vectorized_count(output: &str) -> Option<u32> {
    for line in output.lines() {
        // Handle both British and American spelling
        if line.contains("vectorised:") || line.contains("vectorized:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(last) = parts.last() {
                return last.parse::<u32>().ok();
            }
        }
    }
    None
}

// ── Rasterization helpers ────────────────────────────────────────────

/// Rasterize a vector PDF to a grayscale raster PDF at the given DPI.
/// `aa` controls anti-aliasing: true = default AA, false = hard pixel edges.
/// Uses PyMuPDF (fitz) for rasterization and img2pdf for reassembly.
/// Result is cached at `out_path`; regenerated only if the source is newer.
pub fn rasterize_pdf(src: &Path, out_path: &Path, dpi: u32, aa: bool) -> bool {
    // Check cache: skip if output exists and is newer than source
    if out_path.exists() {
        if let (Ok(src_meta), Ok(out_meta)) = (src.metadata(), out_path.metadata()) {
            if let (Ok(src_mtime), Ok(out_mtime)) = (src_meta.modified(), out_meta.modified()) {
                if out_mtime >= src_mtime {
                    eprintln!("  Using cached {:?}", out_path.file_name().unwrap());
                    return true;
                }
            }
        }
    }

    eprintln!("  Rasterizing {:?} → {:?} (dpi={}, aa={})",
        src.file_name().unwrap(), out_path.file_name().unwrap(), dpi, aa);

    // Use PyMuPDF for rasterization — same FreeType backend as the CI font index
    let aa_flag = if aa { "True" } else { "False" };
    let py_code = format!(
        r#"
import fitz, img2pdf, os, tempfile
doc = fitz.open('{src}')
dpi = {dpi}
aa = {aa_flag}
mat = fitz.Matrix(dpi/72, dpi/72)
tmpdir = tempfile.mkdtemp(prefix='unscan-raster-')
pngs = []
for i, page in enumerate(doc):
    if not aa:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False,
                              annots=False)
        # Force no-AA by thresholding to binary then back to gray
        import numpy as np
        from PIL import Image
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        arr = ((arr > 128) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
        path = os.path.join(tmpdir, f'page_{{i:03d}}.png')
        img.save(path, dpi=(dpi, dpi))
    else:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        path = os.path.join(tmpdir, f'page_{{i:03d}}.png')
        pix.save(path)
    pngs.append(path)
layout = img2pdf.get_layout_fun(pagesize=(img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11)))
with open('{out}', 'wb') as f:
    f.write(img2pdf.convert(pngs, layout_fun=layout))
for p in pngs:
    os.remove(p)
os.rmdir(tmpdir)
"#,
        src = src.display(),
        dpi = dpi,
        aa_flag = aa_flag,
        out = out_path.display(),
    );

    let status = Command::new("python3").arg("-c").arg(&py_code).status();
    let ok = status.map(|s| s.success()).unwrap_or(false);
    if ok {
        eprintln!("  Generated {:?}", out_path.file_name().unwrap());
    } else {
        eprintln!("  ERROR: MuPDF rasterization failed");
    }
    ok
}

/// Rasterize a vector PDF using Poppler's pdftoppm (Cairo rendering engine).
/// This uses a different rasterizer than the CI's FreeType/ab_glyph backend,
/// so accuracy against Poppler rasters tests rendering invariance.
/// Result is cached at `out_path`; regenerated only if the source is newer.
pub fn rasterize_pdf_poppler(src: &Path, out_path: &Path, dpi: u32, aa: bool) -> bool {
    // Check cache: skip if output exists and is newer than source
    if out_path.exists() {
        if let (Ok(src_meta), Ok(out_meta)) = (src.metadata(), out_path.metadata()) {
            if let (Ok(src_mtime), Ok(out_mtime)) = (src_meta.modified(), out_meta.modified()) {
                if out_mtime >= src_mtime {
                    eprintln!("  Using cached {:?}", out_path.file_name().unwrap());
                    return true;
                }
            }
        }
    }

    eprintln!("  Rasterizing (Poppler) {:?} → {:?} (dpi={}, aa={})",
        src.file_name().unwrap(), out_path.file_name().unwrap(), dpi, aa);

    // pdftoppm renders to PGM/PNG pages, then img2pdf reassembles into a PDF.
    // Poppler uses Cairo+FreeType with its own hinting — different from MuPDF.
    let aa_flag = if aa { "True" } else { "False" };
    let py_code = format!(
        r#"
import subprocess, img2pdf, os, tempfile, glob
dpi = {dpi}
aa = {aa}
tmpdir = tempfile.mkdtemp(prefix='unscan-raster-poppler-')
prefix = os.path.join(tmpdir, 'page')

# pdftoppm renders each page as a separate PGM file
cmd = ['pdftoppm', '-r', str(dpi), '-gray', '{src}', prefix]
subprocess.run(cmd, check=True)

pgms = sorted(glob.glob(os.path.join(tmpdir, 'page-*.pgm')))
assert pgms, f'pdftoppm produced no output in {{tmpdir}}'

# Convert PGMs to PNGs (img2pdf prefers PNG), optionally binarizing for no-AA
pngs = []
if not aa:
    import numpy as np
    from PIL import Image
for pgm in pgms:
    if not aa:
        img = Image.open(pgm).convert('L')
        arr = np.array(img)
        arr = ((arr > 128) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
        png = pgm.replace('.pgm', '.png')
        img.save(png, dpi=(dpi, dpi))
    else:
        from PIL import Image
        img = Image.open(pgm).convert('L')
        png = pgm.replace('.pgm', '.png')
        img.save(png, dpi=(dpi, dpi))
    pngs.append(png)

layout = img2pdf.get_layout_fun(pagesize=(img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11)))
with open('{out}', 'wb') as f:
    f.write(img2pdf.convert(pngs, layout_fun=layout))

for p in pngs + pgms:
    os.remove(p)
os.rmdir(tmpdir)
"#,
        src = src.display(),
        dpi = dpi,
        aa = aa_flag,
        out = out_path.display(),
    );

    let status = Command::new("python3").arg("-c").arg(&py_code).status();
    let ok = status.map(|s| s.success()).unwrap_or(false);
    if ok {
        eprintln!("  Generated {:?}", out_path.file_name().unwrap());
    } else {
        eprintln!("  ERROR: Poppler rasterization failed");
    }
    ok
}
