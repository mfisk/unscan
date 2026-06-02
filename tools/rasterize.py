#!/usr/bin/env python3
"""
rasterize.py — Single rasterization entry point for unscan.

Rasterizes a vector PDF to a grayscale raster PDF.  All scan artifacts
(skew, noise, blur, speckle) are **off by default** and opt-in via flags.

Usage:
    python3 tools/rasterize.py INPUT.pdf OUTPUT.pdf [OPTIONS]

Options:
    --dpi N          Resolution (default: 300)
    --no-aa          Disable anti-aliasing (binary threshold)
    --backend STR    'mupdf' (default) or 'poppler'
    --skew DEG       Apply rotational skew in degrees (default: 0)
    --noise          Add paper texture noise + slight darkening
    --speckle        Add speckle noise (dust on scanner glass)
    --blur R         Gaussian blur radius (default: 0, off)
    --scan           Shorthand for --skew 2.0 --noise --speckle --blur 0.7

All artifact flags are off by default, producing a clean raster.
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile


def rasterize(
    src: str,
    out: str,
    dpi: int = 300,
    aa: bool = True,
    backend: str = "mupdf",
    skew_deg: float = 0.0,
    noise: bool = False,
    speckle: bool = False,
    blur_radius: float = 0.0,
):
    """Rasterize *src* vector PDF → *out* raster PDF."""
    import img2pdf
    import numpy as np
    from PIL import Image, ImageFilter

    tmpdir = tempfile.mkdtemp(prefix="unscan-raster-")

    if backend == "mupdf":
        pages = _rasterize_mupdf(src, tmpdir, dpi, aa)
    elif backend == "poppler":
        pages = _rasterize_poppler(src, tmpdir, dpi, aa)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    need_artifacts = skew_deg != 0.0 or noise or speckle or blur_radius > 0

    final_pages = []
    for page_path in pages:
        im = Image.open(page_path).convert("L")

        if need_artifacts:
            orig_w, orig_h = im.size

            # 1. Skew
            if skew_deg != 0.0:
                im = im.rotate(skew_deg, resample=Image.BICUBIC,
                               expand=True, fillcolor=245)
                cx, cy = im.width // 2, im.height // 2
                left = cx - orig_w // 2
                top = cy - orig_h // 2
                im = im.crop((left, top, left + orig_w, top + orig_h))

            arr = np.array(im, dtype=np.float32)

            # 2. Paper noise + darkening
            if noise:
                paper_noise = np.random.normal(0, 1.5, arr.shape).astype(np.float32)
                arr = np.clip(arr + paper_noise, 0, 255)
                arr = arr * 0.96 + 8

            # 3. Speckle
            if speckle:
                sp = np.random.random(arr.shape)
                arr[sp < 0.0003] = np.random.randint(
                    40, 120, size=int(np.sum(sp < 0.0003))
                )

            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

            # 4. Blur
            if blur_radius > 0:
                im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        out_png = os.path.join(tmpdir, f"final_{len(final_pages):03d}.png")
        im.save(out_png, dpi=(dpi, dpi))
        final_pages.append(out_png)

    # Assemble raster PDF
    layout = img2pdf.get_layout_fun(
        pagesize=(img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11))
    )
    with open(out, "wb") as f:
        f.write(img2pdf.convert(final_pages, layout_fun=layout))

    # Cleanup
    for p in glob.glob(os.path.join(tmpdir, "*")):
        os.remove(p)
    os.rmdir(tmpdir)

    return out


# ── Backend: PyMuPDF ──────────────────────────────────────────────────

def _rasterize_mupdf(src, tmpdir, dpi, aa):
    import fitz
    import numpy as np
    from PIL import Image

    doc = fitz.open(src)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pngs = []
    for i, page in enumerate(doc):
        if not aa:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY,
                                  alpha=False, annots=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width
            )
            arr = ((arr > 128) * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
            path = os.path.join(tmpdir, f"page_{i:03d}.png")
            img.save(path, dpi=(dpi, dpi))
        else:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
            path = os.path.join(tmpdir, f"page_{i:03d}.png")
            pix.save(path)
        pngs.append(path)
    return pngs


# ── Backend: Poppler (pdftoppm) ──────────────────────────────────────

def _rasterize_poppler(src, tmpdir, dpi, aa):
    import numpy as np
    from PIL import Image

    prefix = os.path.join(tmpdir, "page")
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", src, prefix],
        check=True,
    )
    pgms = sorted(glob.glob(os.path.join(tmpdir, "page-*.pgm")))
    assert pgms, f"pdftoppm produced no output in {tmpdir}"

    pngs = []
    for pgm in pgms:
        img = Image.open(pgm).convert("L")
        if not aa:
            arr = np.array(img)
            arr = ((arr > 128) * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
        png = pgm.replace(".pgm", ".png")
        img.save(png, dpi=(dpi, dpi))
        pngs.append(png)
        os.remove(pgm)
    return pngs


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Rasterize a vector PDF")
    p.add_argument("input", help="Input vector PDF")
    p.add_argument("output", help="Output raster PDF")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-aa", action="store_true", help="Disable anti-aliasing")
    p.add_argument("--backend", choices=["mupdf", "poppler"], default="mupdf")
    p.add_argument("--skew", type=float, default=0.0, help="Skew degrees")
    p.add_argument("--noise", action="store_true", help="Paper noise + darkening")
    p.add_argument("--speckle", action="store_true", help="Dust speckle noise")
    p.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius")
    p.add_argument("--scan", action="store_true",
                   help="Shorthand: --skew 2.0 --noise --speckle --blur 0.7")
    args = p.parse_args()

    if args.scan:
        if args.skew == 0.0:
            args.skew = 2.0
        args.noise = True
        args.speckle = True
        if args.blur == 0.0:
            args.blur = 0.7

    rasterize(
        src=args.input,
        out=args.output,
        dpi=args.dpi,
        aa=not args.no_aa,
        backend=args.backend,
        skew_deg=args.skew,
        noise=args.noise,
        speckle=args.speckle,
        blur_radius=args.blur,
    )
    print(f"Rasterized: {args.output}")


if __name__ == "__main__":
    main()
