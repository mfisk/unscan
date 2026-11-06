#!/usr/bin/env python3
"""Regenerate test-xpos-kern.pdf from the HTML source with current binary.

Renders /home/hatch/workspace/test_xpos_kern.html via WeasyPrint, annotates
/UnprintCanonical using the verbatim --resolve-fontkey output from the current
binary, then rasterizes at 300 DPI.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/hatch/workspace/repos/unscan")
sys.path.insert(0, str(REPO / "test-docs"))
sys.path.insert(0, str(REPO / "tools"))

from gen_common import render_html_to_pdf
from pdf_font_annotate import annotate_canonical_names

HTML = Path("/home/hatch/workspace/test_xpos_kern.html")
OUT_PDF = REPO / "test-docs" / "test-xpos-kern.pdf"
RAST_PDF = REPO / "test-docs" / "test-xpos-kern-rasterized.pdf"
TREBUC = "/home/hatch/.local/share/fonts/msttcorefonts/trebuc.ttf"
UNPRINT = REPO / "target" / "release" / "unprint"

# 1. Render
html = HTML.read_text()
render_html_to_pdf(html, str(OUT_PDF), base_url="/home/hatch/workspace")
print(f"wrote: {OUT_PDF}")

# 2. Resolve verbatim fontkey from the current binary
r = subprocess.run([str(UNPRINT), "--resolve-fontkey", TREBUC],
                   capture_output=True, text=True, timeout=60)
key = r.stdout.strip().splitlines()[-1]
print(f"resolved key: {key}")

# 3. Annotate: BaseFont in the PDF is "Trebuchet-MS" (subset-stripped)
canonical_map = {"Trebuchet-MS": key}
annotated, missing = annotate_canonical_names(str(OUT_PDF), canonical_map)
print(f"annotated {annotated} fonts, missing: {missing}")
if missing:
    print("ERROR: missing canonical mappings", file=sys.stderr)
    sys.exit(1)

# 4. Rasterize at 300 DPI via tools/rasterize.py
r = subprocess.run([sys.executable, str(REPO / "tools" / "rasterize.py"),
                    "rasterize", "--dpi", "300", str(OUT_PDF), str(RAST_PDF)],
                   capture_output=True, text=True, timeout=300)
print(r.stdout[-500:] if r.stdout else "")
if r.returncode != 0:
    print(r.stderr[-2000:], file=sys.stderr)
    sys.exit(r.returncode)
print(f"wrote: {RAST_PDF}")
print("Done!")
