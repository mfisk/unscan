#!/usr/bin/env python3
"""
Generate a single-line mixed-font regression test PDF.

Line: "Bits per second (bps) is the x-axis not the y-axis"

Font mixing:
  - "Bits per second" in regular, with B, p, s underlined (first letters
    spell out the abbreviation "bps")
  - "(bps)" in italic
  - "is the" in regular
  - "x" in italic, "-axis" in regular
  - "not" bold (whole word)
  - "the" in regular
  - "y" in italic, "-axis" in regular

Output:
  t40-mixed-underline.pdf          — vector PDF
  t40-mixed-underline-raster.pdf   — 300 DPI raster for unscan input

This tests that word splitting doesn't break on mixed-font lines with
underlines, italics, and intra-word font changes.
"""

import subprocess
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

SCRIPT_DIR = Path(__file__).resolve().parent

FONT_DIR = Path("/usr/share/fonts/truetype")

FONTS = {
    "Sans":        FONT_DIR / "liberation/LiberationSans-Regular.ttf",
    "Sans-Italic": FONT_DIR / "liberation/LiberationSans-Italic.ttf",
    "Sans-Bold":   FONT_DIR / "liberation/LiberationSans-Bold.ttf",
}

for name, path in FONTS.items():
    if not path.exists():
        print(f"ERROR: missing font {path}", file=sys.stderr)
        sys.exit(1)
    pdfmetrics.registerFont(TTFont(name, str(path)))

FONT_SIZE = 14
PAGE_W, PAGE_H = letter
Y_POS = PAGE_H - 1.5 * inch  # text baseline


def draw_underline(c, x, y, width, font_size):
    """Draw an underline beneath text at (x, y)."""
    # Underline sits ~2pt below baseline, thickness ~0.5pt
    ul_y = y - 2
    c.setLineWidth(0.7)
    c.line(x, ul_y, x + width, ul_y)


def draw_line(c):
    """Draw the mixed-font test line with underlines and italics."""
    x = 0.75 * inch
    y = Y_POS

    def regular(text):
        nonlocal x
        c.setFont("Sans", FONT_SIZE)
        c.drawString(x, y, text)
        x += c.stringWidth(text, "Sans", FONT_SIZE)

    def italic(text):
        nonlocal x
        c.setFont("Sans-Italic", FONT_SIZE)
        c.drawString(x, y, text)
        x += c.stringWidth(text, "Sans-Italic", FONT_SIZE)

    def regular_underline(text):
        nonlocal x
        c.setFont("Sans", FONT_SIZE)
        w = c.stringWidth(text, "Sans", FONT_SIZE)
        c.drawString(x, y, text)
        draw_underline(c, x, y, w, FONT_SIZE)
        x += w

    # "Bits per second" with B, p, s underlined
    regular_underline("B")
    regular("its ")
    regular_underline("p")
    regular("er ")
    regular_underline("s")
    regular("econd ")

    # "(bps)" in italic
    italic("(bps)")
    regular(" ")

    # "is the " in regular
    regular("is the ")

    # "x" italic + "-axis " regular
    italic("x")
    regular("-axis ")

    # "not" bold whole word
    def bold(text):
        nonlocal x
        c.setFont("Sans-Bold", FONT_SIZE)
        c.drawString(x, y, text)
        x += c.stringWidth(text, "Sans-Bold", FONT_SIZE)

    bold("not")
    regular(" ")

    # "the " regular
    regular("the ")

    # "y" italic + "-axis" regular
    italic("y")
    regular("-axis")


def main():
    vector_pdf = SCRIPT_DIR / "t40-mixed-underline.pdf"
    raster_pdf = SCRIPT_DIR / "t40-mixed-underline-raster.pdf"

    # Build vector PDF
    c_obj = canvas.Canvas(str(vector_pdf), pagesize=letter)
    draw_line(c_obj)
    c_obj.save()
    print(f"Generated {vector_pdf}")

    # Rasterize at 300 DPI
    png_path = "/tmp/t40-mixed-underline.png"
    subprocess.run([
        "pdftoppm", "-r", "300", "-png", "-singlefile",
        str(vector_pdf), "/tmp/t40-mixed-underline"
    ], check=True)

    # Wrap PNG in a PDF
    from PIL import Image as PILImage
    img = PILImage.open(png_path)
    w_px, h_px = img.size
    w_pt = w_px * 72.0 / 300.0
    h_pt = h_px * 72.0 / 300.0

    c2 = canvas.Canvas(str(raster_pdf), pagesize=(w_pt, h_pt))
    c2.drawImage(png_path, 0, 0, w_pt, h_pt)
    c2.showPage()
    c2.save()
    print(f"Generated {raster_pdf}")


if __name__ == "__main__":
    main()
