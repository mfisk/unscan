#!/usr/bin/env python3
"""Generate a multi-line test PDF using WeasyPrint (Pango/HarfBuzz).

Imports fc_find/SECTIONS from gen-specimen.py, canonical naming from
pdf_font_annotate.py, and rasterization from rasterize.py.

Usage:
    python3 test-docs/gen-line-test.py <p:l> [<p:l> ...] [--audit-ref PATH]

Reads BAP audit.json to find text/font for each line, then generates:
    test-docs/line-test-gt.pdf   — vector PDF with /UnprintCanonical metadata
    test-docs/line-test.pdf      — rasterized (300 DPI grayscale)

Uses WeasyPrint (Pango/HarfBuzz backend) for proper OpenType kerning.
Each line is rendered as a whole <p> element, producing proper per-word
PDF text spans so ground-truth text lookup works correctly.
"""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Import gen-specimen (hyphenated filename requires importlib)
_spec = importlib.util.spec_from_file_location("gen_specimen", SCRIPT_DIR / "gen-specimen.py")
_gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gs)
fc_find = _gs.fc_find
# Build family map from SECTIONS
FAMILIES = {s["rl_font"]: s["font_family"] for s in _gs.SECTIONS}

# Import shared tools
sys.path.insert(0, str(REPO_ROOT / "tools"))
from pdf_font_annotate import read_postscript_name, make_weight_explicit, annotate_canonical_names
from rasterize import rasterize


def _escape(text):
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def resolve_font(expected_font):
    """Resolve expected_font string to (ttf_path, canonical_name, css_weight, css_style)."""
    font_base = expected_font.split('-')[0] if '-' in expected_font else expected_font
    for suffix in ['Italic', 'It']:
        font_base = font_base.replace(suffix, '')

    fc_family = FAMILIES.get(font_base)
    if not fc_family:
        raise RuntimeError(f"'{font_base}' not in FAMILIES")

    style = "Regular"
    css_weight = "400"
    css_style = "normal"
    if "Bold" in expected_font or "-700" in expected_font or "-600" in expected_font:
        style = "Bold"
        # The TTF is already the bold weight — tell Pango font-weight: normal
        # so it doesn't append "-Bold" to the embedded PS name.
        css_weight = "400"
    elif "Italic" in expected_font or "It" in expected_font.split('-')[-1]:
        style = "Italic"
        css_style = "italic"

    ttf_path = fc_find(fc_family, style)
    _, canonical_name, _ = make_weight_explicit(ttf_path)
    return ttf_path, canonical_name, css_weight, css_style


def main():
    # Parse args: p:l [p:l ...] [--audit-ref PATH]
    args = sys.argv[1:]
    audit_ref = str(SCRIPT_DIR / "audit" / "audit.json")
    if "--audit-ref" in args:
        idx = args.index("--audit-ref")
        audit_ref = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if len(args) < 1:
        print("Usage: gen-line-test.py <p:l> [<p:l> ...] [--audit-ref PATH]")
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
    lines = []  # (text, canonical_name, ttf_path, css_weight, css_style)
    font_face_rules = []
    seen_faces = set()

    for (page, li) in page_lines:
        entries = [e for e in audit['text_entries']
                   if e.get('page') == page and e.get('line_index') == li]
        if not entries:
            print(f"ERROR: No entries for p{page}:L{li}", file=sys.stderr)
            sys.exit(1)

        entry = entries[0]
        expected_font = entry.get('expected_font', '')
        text = entry.get('gt_text', entry.get('text', entry.get('ocr_text', '')))
        if not text.endswith('.'):
            text += '.'
        print(f"p{page}:L{li}: text='{text}', expected={expected_font}")

        ttf_path, canonical_name, css_weight, css_style = resolve_font(expected_font)
        print(f"  font: {ttf_path}, canonical: {canonical_name}")

        # Track for annotation
        ps_name = read_postscript_name(str(ttf_path))
        canonical_map[ps_name] = canonical_name
        canonical_map[canonical_name] = canonical_name

        # Generate @font-face CSS (deduplicated)
        face_key = (canonical_name, css_weight, css_style)
        if face_key not in seen_faces:
            seen_faces.add(face_key)
            font_face_rules.append(
                f"@font-face {{\n"
                f"  font-family: '{canonical_name}';\n"
                f"  src: url('file://{ttf_path}') format('truetype');\n"
                f"  font-weight: {css_weight};\n"
                f"  font-style: {css_style};\n"
                f"}}"
            )

        lines.append((text, canonical_name, str(ttf_path), css_weight, css_style))

    # Build HTML
    font_face_css = "\n".join(font_face_rules)
    font_size = 9  # pt

    lines_html = []
    for text, canonical_name, ttf_path, css_weight, css_style in lines:
        lines_html.append(
            f'<p style="font-family: \'{canonical_name}\'; '
            f'font-weight: {css_weight}; font-style: {css_style};">'
            f'{_escape(text)}</p>'
        )

    all_lines = "\n".join(lines_html)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{font_face_css}

@page {{
  size: 8.5in 11in;
  margin: 1in 1in;
}}

body {{
  font-size: {font_size}pt;
  line-height: 1.2;
  color: black;
}}

p {{
  margin: 0;
  padding: 0;
}}
</style>
</head>
<body>
{all_lines}
</body>
</html>"""

    # Render with WeasyPrint
    import weasyprint
    gt_pdf = str(SCRIPT_DIR / "line-test-gt.pdf")
    doc = weasyprint.HTML(string=html, base_url=str(SCRIPT_DIR))
    doc.write_pdf(gt_pdf)
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
