#!/usr/bin/env python3
"""
rasterize.py — Rasterize vector PDFs and build ground-truth fontmaps for unscan.

Subcommands:
    rasterize   Rasterize a vector PDF to a grayscale raster PDF.
    fontmap     Build a PS-name → file-path font map from a vector PDF.
    prepare     Do both: rasterize + fontmap in one shot, print next-step commands.

Usage:
    python3 tools/rasterize.py rasterize INPUT.pdf OUTPUT.pdf [OPTIONS]
    python3 tools/rasterize.py fontmap INPUT.pdf [-o MAP.json]
    python3 tools/rasterize.py prepare INPUT.pdf [OPTIONS]

Rasterize options:
    --dpi N          Resolution (default: 300)
    --no-aa          Disable anti-aliasing at the renderer level (8-bit output)
    --threshold      Binary threshold output to 1-bit (0/255)
    --color          Render in RGB color instead of grayscale
    --backend STR    'mupdf' (default) or 'poppler'
    --skew DEG       Apply rotational skew in degrees (default: 0)
    --noise          Add paper texture noise + slight darkening
    --speckle        Add speckle noise (dust on scanner glass)
    --blur R         Gaussian blur radius (default: 0, off)
    --scan           Shorthand for --skew 2.0 --noise --speckle --blur 0.7

Prepare options:
    --dpi, --no-aa, --color, --backend  (same as rasterize)
    -d / --output-dir          Output directory (default: same as input)
    -o / --output              Explicit rasterized PDF path
    --fontmap-only             Only build fontmap, skip rasterization
    --rasterize-only           Only rasterize, skip fontmap

All artifact flags are off by default, producing a clean raster.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ── Rasterization ────────────────────────────────────────────────────

def rasterize(
    src,
    out,
    dpi=300,
    aa=True,
    backend="mupdf",
    skew_deg=0.0,
    noise=False,
    speckle=False,
    blur_radius=0.0,
    color=False,
    threshold=False,
):
    """Rasterize *src* vector PDF → *out* raster PDF."""
    import img2pdf
    import numpy as np
    from PIL import Image, ImageFilter

    src, out = str(src), str(out)
    tmpdir = tempfile.mkdtemp(prefix="unscan-raster-")

    if backend == "mupdf":
        pages = _rasterize_mupdf(src, tmpdir, dpi, aa, color=color, threshold=threshold)
    elif backend == "poppler":
        pages = _rasterize_poppler(src, tmpdir, dpi, aa, color=color, threshold=threshold)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    need_artifacts = skew_deg != 0.0 or noise or speckle or blur_radius > 0

    final_pages = []
    for page_path in pages:
        im = Image.open(page_path)
        if not color:
            im = im.convert("L")

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


# ── Backend: PyMuPDF ─────────────────────────────────────────────────

def _rasterize_mupdf(src, tmpdir, dpi, aa, color=False, threshold=False):
    import fitz
    import numpy as np
    from PIL import Image

    if not aa:
        fitz.TOOLS.set_aa_level(0)

    doc = fitz.open(src)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    cs = fitz.csRGB if color else fitz.csGRAY
    pngs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=cs, alpha=False)
        if threshold:
            arr = np.frombuffer(pix.samples, dtype=np.uint8).copy()
            if color:
                arr = arr.reshape(pix.height, pix.width, 3)
            else:
                arr = arr.reshape(pix.height, pix.width)
            arr = ((arr > 128) * 255).astype(np.uint8)
            mode = "RGB" if color else "L"
            img = Image.fromarray(arr, mode=mode)
            path = os.path.join(tmpdir, f"page_{i:03d}.png")
            img.save(path, dpi=(dpi, dpi))
        else:
            path = os.path.join(tmpdir, f"page_{i:03d}.png")
            pix.save(path)
        pngs.append(path)

    if not aa:
        fitz.TOOLS.set_aa_level(8)  # restore default

    return pngs


# ── Backend: Poppler (pdftoppm) ──────────────────────────────────────

def _rasterize_poppler(src, tmpdir, dpi, aa, color=False, threshold=False):
    import numpy as np
    from PIL import Image

    prefix = os.path.join(tmpdir, "page")
    fmt_args = ["-png"] if color else ["-gray"]
    aa_args = [] if aa else ["-aa", "no", "-aaVector", "no"]
    subprocess.run(
        ["pdftoppm", "-r", str(dpi)] + fmt_args + aa_args + [src, prefix],
        check=True,
    )
    # Poppler outputs .pgm for -gray, .png for -png
    if color:
        outputs = sorted(glob.glob(os.path.join(tmpdir, "page-*.png")))
    else:
        outputs = sorted(glob.glob(os.path.join(tmpdir, "page-*.pgm")))
    assert outputs, f"pdftoppm produced no output in {tmpdir}"

    pngs = []
    for raw in outputs:
        img = Image.open(raw)
        if not color:
            img = img.convert("L")
        if threshold:
            arr = np.array(img)
            arr = ((arr > 128) * 255).astype(np.uint8)
            mode = "RGB" if color else "L"
            img = Image.fromarray(arr, mode=mode)
        png = raw.replace(".pgm", ".png") if raw.endswith(".pgm") else raw
        img.save(png, dpi=(dpi, dpi))
        pngs.append(png)
        if raw != png:
            os.remove(raw)
    return pngs


# ── Fontmap ──────────────────────────────────────────────────────────

def build_fontmap(pdf_path):
    """Extract font file map by introspecting a vector PDF.

    Returns (resolved, unresolved) where resolved is a dict mapping
    PostScript name → absolute file path, and unresolved is a list of
    PS names that couldn't be resolved (built-in PDF fonts).
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    fontmap = {}

    for page_num in range(len(doc)):
        for font_entry in doc[page_num].get_fonts(full=True):
            basefont = font_entry[3]
            # Strip subset prefix (AAAAAA+FontName → FontName)
            if '+' in basefont:
                basefont = basefont.split('+', 1)[1]
            if basefont in fontmap:
                continue

            # Reverse-resolve: PS name → file path via fontconfig
            r = subprocess.run(
                ['fc-list', f':postscriptname={basefont}', '--format=%{file}\n'],
                capture_output=True, text=True
            )
            files = [l for l in r.stdout.strip().split('\n') if l]
            if files:
                # Prefer .ttf (TrueType outlines)
                ttf = [f for f in files if f.lower().endswith('.ttf')]
                fontmap[basefont] = ttf[0] if ttf else files[0]
            else:
                fontmap[basefont] = None

    doc.close()

    resolved = {k: v for k, v in sorted(fontmap.items()) if v is not None}
    unresolved = [k for k, v in fontmap.items() if v is None]

    return resolved, unresolved


def write_fontmap(resolved, output_path):
    """Write a fontmap dict to a JSON file."""
    with open(str(output_path), 'w') as f:
        json.dump(resolved, f, indent=2, sort_keys=True)
        f.write('\n')


# ── CLI: rasterize subcommand ────────────────────────────────────────

def cmd_rasterize(args):
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
        color=args.color,
        threshold=args.threshold,
    )
    print(f"Rasterized: {args.output}")


# ── CLI: fontmap subcommand ──────────────────────────────────────────

def cmd_fontmap(args):
    resolved, unresolved = build_fontmap(args.pdf)

    if args.verbose or not args.output:
        print(f"{len(resolved)} fonts resolved, {len(unresolved)} unresolved",
              file=sys.stderr)
        if unresolved:
            for name in unresolved:
                print(f"  unresolved (builtin?): {name}", file=sys.stderr)

    output = json.dumps(resolved, indent=2, sort_keys=True)

    if args.output:
        write_fontmap(resolved, args.output)
        if args.verbose:
            print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output)


# ── CLI: prepare subcommand ──────────────────────────────────────────

def cmd_prepare(args):
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    stem = pdf_path.stem
    out_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    if args.output:
        rasterized_path = Path(args.output)
    else:
        suffix = "-noaa" if args.no_aa else ""
        if args.color:
            suffix += "-color"
        rasterized_path = out_dir / f"{stem}-rasterized{suffix}.pdf"
    fontmap_path = out_dir / f"{stem}-fontmap.json"

    # --- Fontmap ---
    if not args.rasterize_only:
        print(f"Building fontmap from {pdf_path.name}...")
        resolved, unresolved = build_fontmap(str(pdf_path))
        write_fontmap(resolved, fontmap_path)
        print(f"  {len(resolved)} fonts resolved → {fontmap_path}")
        if unresolved:
            print(f"  {len(unresolved)} unresolved (builtins): {', '.join(unresolved)}")

    # --- Rasterize ---
    if not args.fontmap_only:
        aa_label = "no-AA" if args.no_aa else "AA"
        threshold_label = "+threshold" if args.threshold else ""
        color_label = ", color" if args.color else ""
        print(f"Rasterizing at {args.dpi} DPI, {aa_label}{threshold_label}{color_label} ({args.backend})...")
        rasterize(pdf_path, rasterized_path,
                  dpi=args.dpi, backend=args.backend, aa=not args.no_aa,
                  color=args.color, threshold=args.threshold)
        print(f"  Rasterized: {rasterized_path}")

    # --- Summary ---
    print()
    print("Next steps:")
    if not args.rasterize_only:
        print(f"  # Run unscan with ground-truth audit")
        print(f"  ./target/release/unscan {rasterized_path} \\")
        print(f"    -o /tmp/out.pdf --audit /tmp/audit \\")
        print(f"    --audit-vector {pdf_path}")
        print()
        print(f"  # Report at /tmp/audit/report.html")


# ── Shared argument helpers ──────────────────────────────────────────

def _add_raster_args(p):
    """Add common rasterization arguments to a subparser."""
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-aa", action="store_true", help="Disable anti-aliasing at the renderer level (8-bit output)")
    p.add_argument("--threshold", action="store_true", help="Binary threshold output to 1-bit (0/255). Combine with --no-aa for true binary rasterization")
    p.add_argument("--color", action="store_true", help="Render in RGB color instead of grayscale")
    p.add_argument("--backend", choices=["mupdf", "poppler"], default="mupdf")


def _add_artifact_args(p):
    """Add scan-artifact arguments to a subparser."""
    p.add_argument("--skew", type=float, default=0.0, help="Skew degrees")
    p.add_argument("--noise", action="store_true", help="Paper noise + darkening")
    p.add_argument("--speckle", action="store_true", help="Dust speckle noise")
    p.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius")
    p.add_argument("--scan", action="store_true",
                   help="Shorthand: --skew 2.0 --noise --speckle --blur 0.7")


# ── Main CLI ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Rasterize vector PDFs and build fontmaps for unscan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # rasterize
    r = sub.add_parser("rasterize", help="Rasterize a vector PDF")
    r.add_argument("input", help="Input vector PDF")
    r.add_argument("output", help="Output raster PDF")
    _add_raster_args(r)
    _add_artifact_args(r)

    # fontmap
    f = sub.add_parser("fontmap", help="Build font map from a vector PDF")
    f.add_argument("pdf", help="Path to the vector PDF")
    f.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    f.add_argument("-v", "--verbose", action="store_true",
                   help="Print summary to stderr")

    # prepare
    pr = sub.add_parser("prepare",
                        help="Rasterize + fontmap in one shot")
    pr.add_argument("pdf", help="Path to the vector PDF")
    pr.add_argument("--output-dir", "-d",
                    help="Output directory (default: same as input PDF)")
    pr.add_argument("-o", "--output",
                    help="Explicit output path for rasterized PDF")
    _add_raster_args(pr)
    pr.add_argument("--fontmap-only", action="store_true",
                    help="Only build the fontmap, skip rasterization")
    pr.add_argument("--rasterize-only", action="store_true",
                    help="Only rasterize, skip fontmap generation")

    args = p.parse_args()

    if args.command == "rasterize":
        cmd_rasterize(args)
    elif args.command == "fontmap":
        cmd_fontmap(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
