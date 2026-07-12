#!/usr/bin/env python3
"""Generate a multi-line test PDF from scratch using shared font infrastructure.

Imports fc_find/SECTIONS from gen-specimen.py, canonical naming from
pdf_font_annotate.py, and rasterization from rasterize.py.

Usage:
    python3 test-docs/gen-line-test.py <page> <line> [<line2> ...] [--audit-ref PATH]

Reads BAP audit.json to find text/font for each line, then generates:
    test-docs/line-test-gt.pdf   — vector PDF with /UnprintCanonical metadata
    test-docs/line-test.pdf      — rasterized (300 DPI grayscale)
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import ParagraphStyle


def _draw_edge_rules(canvas, doc):
    """Draw horizontal rules at the very top and bottom edges of the page.
    Gives detect_skew strong horizontal signal so it won't hallucinate
    rotation on sparse pages. Placed far from text to avoid crop contamination."""
    w, h = letter
    canvas.setStrokeColor((0.3, 0.3, 0.3))
    canvas.setLineWidth(1.0)
    canvas.line(18, h - 18, w - 18, h - 18)  # top edge, 18pt from border
    canvas.line(18, 18, w - 18, 18)            # bottom edge, 18pt from border

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Import gen-specimen (hyphenated filename requires importlib)
_spec = importlib.util.spec_from_file_location("gen_specimen", SCRIPT_DIR / "gen-specimen.py")
_gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gs)
fc_find = _gs.fc_find
# Build family map from SECTIONS (FAMILIES is local to register_all_fonts)
FAMILIES = {s["rl_font"]: s["font_family"] for s in _gs.SECTIONS}

# Import shared tools
sys.path.insert(0, str(REPO_ROOT / "tools"))
from pdf_font_annotate import read_postscript_name, make_weight_explicit, annotate_canonical_names
from rasterize import rasterize


def resolve_font(expected_font):
    """Resolve expected_font string to (ttf_path, canonical_name)."""
    font_base = expected_font.split('-')[0] if '-' in expected_font else expected_font
    for suffix in ['Italic', 'It']:
        font_base = font_base.replace(suffix, '')

    fc_family = FAMILIES.get(font_base)
    if not fc_family:
        raise RuntimeError(f"'{font_base}' not in FAMILIES")

    style = "Regular"
    if "Bold" in expected_font or "-700" in expected_font or "-600" in expected_font:
        style = "Bold"
    elif "Italic" in expected_font or "It" in expected_font.split('-')[-1]:
        style = "Italic"

    ttf_path = fc_find(fc_family, style)
    _, canonical_name, _ = make_weight_explicit(ttf_path)
    return ttf_path, canonical_name


def main():
    # Parse args: page line1 [line2 ...] [--audit-ref PATH]
    args = sys.argv[1:]
    audit_ref = str(SCRIPT_DIR / "audit" / "audit.json")
    if "--audit-ref" in args:
        idx = args.index("--audit-ref")
        audit_ref = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if len(args) < 2:
        print("Usage: gen-line-test.py <page> <line> [<line2> ...] or <p:l> [<p:l> ...]")
        sys.exit(1)

    # Support both "page line1 line2" and "page:line page:line" formats
    if ':' in args[0]:
        page_lines = [(int(a.split(':')[0]), int(a.split(':')[1])) for a in args]
    else:
        page = int(args[0])
        page_lines = [(page, int(a)) for a in args[1:]]

    with open(audit_ref) as f:
        audit = json.load(f)

    # Collect entries and fonts for each line
    canonical_map = {}
    story = []
    for (page, li) in page_lines:
        entries = [e for e in audit['text_entries']
                   if e.get('page') == page and e.get('line_index') == li]
        if not entries:
            print(f"ERROR: No entries for p{page}:L{li}", file=sys.stderr)
            sys.exit(1)

        entry = entries[0]
        expected_font = entry.get('expected_font', '')
        text = entry.get('gt_text', entry.get('text', entry.get('ocr_text', ''))) + '.'
        print(f"p{page}:L{li}: text='{text}', expected={expected_font}")

        ttf_path, canonical_name = resolve_font(expected_font)
        print(f"  font: {ttf_path}, canonical: {canonical_name}")

        # Register if not already
        try:
            pdfmetrics.getFont(canonical_name)
        except KeyError:
            pdfmetrics.registerFont(TTFont(canonical_name, ttf_path))

        # Track for annotation
        ps_name = read_postscript_name(ttf_path)
        canonical_map[ps_name] = canonical_name
        canonical_map[canonical_name] = canonical_name

        style_obj = ParagraphStyle(
            f'Line-{li}', fontName=canonical_name,
            fontSize=9, leading=10.8)
        story.append(Paragraph(text, style_obj))

    # Build vector PDF
    gt_pdf = str(SCRIPT_DIR / "line-test-gt.pdf")
    doc = SimpleDocTemplate(gt_pdf, pagesize=letter,
                            leftMargin=72, rightMargin=72,
                            topMargin=72, bottomMargin=72)
    doc.build(story, onFirstPage=_draw_edge_rules, onLaterPages=_draw_edge_rules)
    print(f"wrote: {gt_pdf}")

    # Annotate /UnprintCanonical
    annotated, missing = annotate_canonical_names(gt_pdf, canonical_map)
    print(f"annotated {annotated} fonts, missing: {missing}")

    # Rasterize
    rast_pdf = str(SCRIPT_DIR / "line-test.pdf")
    rasterize(gt_pdf, rast_pdf)
    print(f"wrote: {rast_pdf}")


if __name__ == "__main__":
    main()
