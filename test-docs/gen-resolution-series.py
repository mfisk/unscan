#!/usr/bin/env python3
"""
Generate a resolution degradation series from a source PDF.

Takes a clean PDF (rendered at 300dpi) and produces simulated scans at:
  - 600 dpi  (archival quality)
  - 300 dpi  (standard office scanner)
  - 200 dpi  (default on many MFPs)
  - 150 dpi  (economy scan / email-optimized)
  - 100 dpi  (fax-fine mode, 200×200 but we use 100 for brutal test)
  -  98 dpi  (standard fax: 204×98, horizontally stretched)

Each output simulates realistic scan artifacts at that resolution:
  - Resolution-appropriate blur
  - Paper texture / off-white background
  - Speckle noise (dust on glass)
  - Slight skew (consistent across all, ~2°)
  - JPEG artifacts for fax-tier (real fax uses lossy compression)

For the fax resolutions, we also simulate the characteristic horizontal
stretch and vertical compression of Group 3 fax encoding.

Usage:
    python3 gen-resolution-series.py [source.pdf] [output_dir]

Defaults:
    source: font-timeline-specimen.pdf
    output: resolution-series/
"""

import subprocess, os, sys, shutil, random, math
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────

SOURCE_DPI = 300  # DPI of the source PDF pages (our gen-specimen renders at 300)

RESOLUTIONS = [
    {
        "name": "600dpi",
        "dpi": 600,
        "blur_radius": 0.3,
        "noise_sigma": 1.0,
        "speckle_rate": 0.0001,
        "paper_darken": 0.98,
        "paper_offset": 4,
        "jpeg_quality": None,  # lossless PNG
        "description": "Archival scan — 600 dpi, minimal artifacts",
    },
    {
        "name": "300dpi",
        "dpi": 300,
        "blur_radius": 0.7,
        "noise_sigma": 1.5,
        "speckle_rate": 0.0003,
        "paper_darken": 0.96,
        "paper_offset": 8,
        "jpeg_quality": None,
        "description": "Standard office scan — 300 dpi",
    },
    {
        "name": "200dpi",
        "dpi": 200,
        "blur_radius": 0.9,
        "noise_sigma": 2.0,
        "speckle_rate": 0.0005,
        "paper_darken": 0.95,
        "paper_offset": 10,
        "jpeg_quality": 92,
        "description": "Default MFP scan — 200 dpi, slight JPEG compression",
    },
    {
        "name": "150dpi",
        "dpi": 150,
        "blur_radius": 1.1,
        "noise_sigma": 2.5,
        "speckle_rate": 0.0008,
        "paper_darken": 0.94,
        "paper_offset": 12,
        "jpeg_quality": 85,
        "description": "Economy scan — 150 dpi, email-optimized",
    },
    {
        "name": "100dpi",
        "dpi": 100,
        "blur_radius": 1.4,
        "noise_sigma": 3.0,
        "speckle_rate": 0.001,
        "paper_darken": 0.93,
        "paper_offset": 14,
        "jpeg_quality": 75,
        "description": "Fax-fine mode — 100 dpi, heavy degradation",
    },
    {
        "name": "fax-standard",
        "dpi": (204, 98),  # fax standard: 204 horizontal × 98 vertical
        "blur_radius": 1.6,
        "noise_sigma": 4.0,
        "speckle_rate": 0.002,
        "paper_darken": 0.92,
        "paper_offset": 16,
        "jpeg_quality": 65,
        "fax_dither": True,  # simulate fax's 1-bit dithering
        "description": "Standard fax (Group 3) — 204×98 dpi, 1-bit dithered",
    },
]

# US Letter at 300dpi (source dimensions)
PAGE_W_300 = 2550
PAGE_H_300 = 3300


def rasterize_pdf(pdf_path, dpi, output_dir):
    """Use pdftoppm to rasterize a PDF at the given DPI. Returns list of PNG paths."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, "page")
    cmd = ["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf_path), prefix]
    subprocess.run(cmd, check=True, capture_output=True)
    pages = sorted(Path(output_dir).glob("page-*.png"))
    return [str(p) for p in pages]


def apply_scan_artifacts(img_path, cfg, skew_deg):
    """Apply resolution-appropriate scan simulation to a page image."""
    im = Image.open(img_path).convert("L")
    target_dpi = cfg["dpi"]

    # For anisotropic fax DPI, resize differently in x and y
    if isinstance(target_dpi, tuple):
        dpi_x, dpi_y = target_dpi
        # Source was rasterized at max(dpi_x, dpi_y) to avoid upscaling
        # Now scale to the actual fax pixel grid
        src_dpi = max(dpi_x, dpi_y)
        new_w = int(im.width * dpi_x / src_dpi)
        new_h = int(im.height * dpi_y / src_dpi)
        im = im.resize((new_w, new_h), Image.LANCZOS)
    
    w, h = im.size

    # 1. Skew
    im = im.rotate(skew_deg, resample=Image.BICUBIC, expand=True, fillcolor=245)
    # Crop back to original size (centered)
    cx, cy = im.width // 2, im.height // 2
    left = cx - w // 2
    top = cy - h // 2
    im = im.crop((left, top, left + w, top + h))

    # 2. Paper texture + noise
    arr = np.array(im, dtype=np.float32)
    paper_noise = np.random.normal(0, cfg["noise_sigma"], arr.shape).astype(np.float32)
    arr = np.clip(arr + paper_noise, 0, 255)
    arr = arr * cfg["paper_darken"] + cfg["paper_offset"]

    # 3. Speckle noise (scanner dust)
    speckle = np.random.random(arr.shape)
    n_speckle = np.sum(speckle < cfg["speckle_rate"])
    if n_speckle > 0:
        arr[speckle < cfg["speckle_rate"]] = np.random.randint(30, 130, size=n_speckle)

    # 4. Gaussian blur
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    if cfg["blur_radius"] > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=cfg["blur_radius"]))

    # 5. Fax dithering — convert to 1-bit with Floyd-Steinberg, then back to gray
    if cfg.get("fax_dither"):
        im = im.convert("1")  # PIL uses Floyd-Steinberg dithering by default
        im = im.convert("L")  # back to 8-bit gray (0 or 255)

    return im


def images_to_pdf(image_paths, out_pdf, dpi):
    """Combine images into a PDF at the specified DPI."""
    import img2pdf

    # For anisotropic DPI, use the horizontal DPI for layout (fax stretches on display)
    if isinstance(dpi, tuple):
        layout_dpi = dpi[0]
    else:
        layout_dpi = dpi

    # US Letter dimensions
    page_w_pt = 8.5 * 72  # 612 pt
    page_h_pt = 11 * 72   # 792 pt

    layout = img2pdf.get_layout_fun(
        pagesize=(img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11))
    )

    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(image_paths, layout_fun=layout))


def main():
    source_pdf = sys.argv[1] if len(sys.argv) > 1 else "font-timeline-specimen.pdf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "resolution-series"

    if not os.path.exists(source_pdf):
        print(f"Error: source PDF not found: {source_pdf}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Consistent skew across all resolutions
    skew_deg = random.uniform(1.5, 3.0)
    if random.random() < 0.5:
        skew_deg = -skew_deg
    print(f"Global skew: {skew_deg:.1f}°")

    results = []

    for cfg in RESOLUTIONS:
        name = cfg["name"]
        dpi = cfg["dpi"]
        desc = cfg["description"]

        # Rasterization DPI: for anisotropic, rasterize at the higher of the two
        raster_dpi = max(dpi) if isinstance(dpi, tuple) else dpi

        print(f"\n{'='*60}")
        print(f"  {name}: {desc}")
        print(f"  Rasterizing source at {raster_dpi} dpi...")

        # Rasterize the clean PDF at target DPI
        raster_dir = f"/tmp/resolution-series/{name}/raster"
        page_pngs = rasterize_pdf(source_pdf, raster_dpi, raster_dir)
        print(f"  {len(page_pngs)} pages rasterized")

        # Apply scan artifacts
        print(f"  Applying scan simulation...")
        out_pages = []
        scan_dir = f"/tmp/resolution-series/{name}/scan"
        os.makedirs(scan_dir, exist_ok=True)

        for i, ppng in enumerate(page_pngs):
            im = apply_scan_artifacts(ppng, cfg, skew_deg)

            # Save — JPEG for lower quality tiers, PNG for high quality
            if cfg.get("jpeg_quality"):
                out_path = f"{scan_dir}/page_{i:03d}.jpg"
                im.save(out_path, "JPEG", quality=cfg["jpeg_quality"])
                # Convert back to PNG for img2pdf (it handles both, but consistent)
                png_path = f"{scan_dir}/page_{i:03d}.png"
                Image.open(out_path).save(png_path)
                out_pages.append(png_path)
            else:
                out_path = f"{scan_dir}/page_{i:03d}.png"
                im.save(out_path)
                out_pages.append(out_path)

        # Assemble PDF
        out_pdf = os.path.join(output_dir, f"specimen-{name}.pdf")
        print(f"  Writing: {out_pdf}")
        images_to_pdf(out_pages, out_pdf, dpi)

        file_size = os.path.getsize(out_pdf)
        results.append({
            "name": name,
            "dpi": str(dpi),
            "description": desc,
            "file": f"specimen-{name}.pdf",
            "size_bytes": file_size,
        })

    # Write manifest
    import json
    manifest = {
        "source": source_pdf,
        "skew_degrees": round(skew_deg, 1),
        "description": "Resolution degradation series for font detection testing",
        "files": results,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Resolution series complete: {output_dir}/")
    print(f"{'─'*60}")
    for r in results:
        sz = r["size_bytes"]
        sz_str = f"{sz/1024:.0f}K" if sz < 1024*1024 else f"{sz/1024/1024:.1f}M"
        print(f"  {r['name']:20s}  {sz_str:>8s}  {r['description']}")
    print(f"{'─'*60}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
