#!/usr/bin/env python3
"""Generate a single-line PDF using Inter Bold (OTF-only font).

Inter Bold is only available as .otf on this system — there is no .ttf.
This test fixture verifies that unscan can identify fonts distributed
exclusively as OpenType/CFF (.otf) files.

Output: test-docs/inter-bold-sentence-raster.pdf (rasterized at 300 DPI)
"""

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from importlib import import_module
rasterize_mod = import_module("rasterize")

OTF_PATH = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
OUT_DIR = Path(__file__).resolve().parent
SENTENCE = "The quick brown fox jumps over 1,234,567,890 lazy dogs."

def main():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # letter size in points

    font = fitz.Font(fontfile=OTF_PATH)
    tw = fitz.TextWriter(page.rect)
    tw.append((72, 96), SENTENCE, font=font, fontsize=24)
    tw.write_text(page)

    vector_pdf = OUT_DIR / "inter-bold-sentence.pdf"
    doc.save(str(vector_pdf))
    doc.close()
    print(f"Vector PDF: {vector_pdf}")

    # Verify the embedded font name
    doc2 = fitz.open(str(vector_pdf))
    page2 = doc2[0]
    for f in page2.get_fonts():
        print(f"  Embedded: basefont={f[3]} name={f[4]}")
    blocks = page2.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                print(f"  Span: font={span['font']} size={span['size']}")
    doc2.close()

    # Rasterize
    raster_pdf = OUT_DIR / "inter-bold-sentence-raster.pdf"
    rasterize_mod.rasterize(vector_pdf, raster_pdf, dpi=300)
    print(f"Raster PDF:  {raster_pdf}")


if __name__ == "__main__":
    main()
