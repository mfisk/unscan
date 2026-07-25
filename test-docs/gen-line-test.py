#!/usr/bin/env python3
"""Generate a multi-line test PDF using WeasyPrint (Pango/HarfBuzz).

Two modes:
  1) Hardcoded (new, used by lob + t59):
     python3 test-docs/gen-line-test.py --hardcoded
     or
     python3 test-docs/gen-line-test.py "LibreBodoni-400=abcdefgh." "EBGaramond-400=Hello world."

     No audit.json dependency. No empty lines.

  2) Legacy audit mode (deprecated, fragile):
     python3 test-docs/gen-line-test.py 1:72 1:73 ...

Uses shared logic from gen_common.py (font registry, fontconfig lookup,
HTML escaping, @font-face CSS, PDF rendering, canonical annotation).
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Shared generation utilities — single source of truth
from gen_common import (
    resolve_expected_font,
    escape_html as _escape,
    build_font_face_css,
    render_html_to_pdf,
    build_canonical_map_from_pdf,
)
from pdf_font_annotate import annotate_canonical_names
from rasterize import rasterize

# ---------------------------------------------------------------------------
# Hardcoded 7-line seam test — no audit, no empty lines, deterministic.
# Matches tests/t59_seam_regression.rs EXPECTED and lob coverage.
# ---------------------------------------------------------------------------
HARDCODED_7 = [
    ("LibreBodoni-400", "abcdefghijklmnopqrstuvwxyz."),
    ("LibreBodoni-400", "ABCDEFGHIJKLMNOPQRSTUVWXYZ."),
    ("Georgia-400", "ABCDEFGHIJKLMNOPQRSTUVWXYZ."),
    ("OpenSans-400", "abcdefghijklmnopqrstuvwxyz."),
    ("LibreBodoni-400Italic", "dogs."),
    ("IBMPlexSans-400", "abcdefghijklmnopqrstuvwxyz."),
    ("SourceSerif4-400It", "Mayr-Duffner."),
    ("SourceSerif4-400It", "Type"),
    ("LibreBaskerville-400", "abcdefghijklmnopqrstuvwxyz."),

    ("Georgia-400", "Matthew Carter created Georgia in 1993."),
]

def parse_args():
    args = sys.argv[1:]
    audit_ref = str(SCRIPT_DIR / "audit" / "audit.json")
    if "--audit-ref" in args:
        idx = args.index("--audit-ref")
        audit_ref = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    hardcoded = False
    if "--hardcoded" in args:
        hardcoded = True
        args = [a for a in args if a != "--hardcoded"]
        if not args:
            args = [f"{font}={text}" for font, text in HARDCODED_7]

    # Detect hardcoded "Font=Text" form (contains '=')
    if args and any("=" in a for a in args):
        hardcoded = True

    return hardcoded, args, audit_ref

def main():
    hardcoded, args, audit_ref = parse_args()

    if len(args) < 1:
        print("Usage:")
        print("  gen-line-test.py --hardcoded")
        print("  gen-line-test.py 'LibreBodoni-400=abc.' 'EBGaramond-400=Hello.'")
        print("  gen-line-test.py 1:72 1:73 ...  (legacy audit mode)")
        sys.exit(1)

    ttf_paths_for_map = []
    lines = []  # (text, canonical_name, ttf_path, css_weight, css_style)
    font_face_entries = []

    if hardcoded:
        for item in args:
            if "=" not in item:
                print(f"ERROR: hardcoded mode expects 'Font=Text', got '{item}'", file=sys.stderr)
                sys.exit(1)
            expected_font, text = item.split("=", 1)
            expected_font = expected_font.strip()
            text = text.strip()
            if not text:
                print(f"ERROR: empty line for font '{expected_font}' — no empty lines allowed", file=sys.stderr)
                sys.exit(1)
            if not expected_font:
                print(f"ERROR: empty font for text '{text}'", file=sys.stderr)
                sys.exit(1)

            print(f"hardcoded: text='{text}', expected={expected_font}")

            ttf_path, canonical_name, css_weight, css_style = resolve_expected_font(expected_font)
            print(f"  font: {ttf_path}, canonical: {canonical_name}")

            ttf_paths_for_map.append(ttf_path)
            font_face_entries.append((canonical_name, ttf_path, css_weight, css_style))
            lines.append((text, canonical_name, str(ttf_path), css_weight, css_style))
    else:
        # Legacy audit mode (fragile, deprecated)
        if ':' in args[0]:
            page_lines = [(int(a.split(':')[0]), int(a.split(':')[1])) for a in args]
        else:
            page = int(args[0])
            page_lines = [(page, int(a)) for a in args[1:]]

        with open(audit_ref) as f:
            audit = json.load(f)

        for (page, li) in page_lines:
            entries = [e for e in audit['text_entries']
                       if e.get('page') == page and e.get('line_index') == li]
            if not entries:
                print(f"ERROR: No entries for p{page}:L{li}", file=sys.stderr)
                sys.exit(1)

            entry = entries[0]
            expected_font = entry.get('expected_font', '')
            text = entry.get('gt_text', entry.get('text', entry.get('ocr_text', '')))
            if not text:
                print(f"ERROR: p{page}:L{li} has empty text (no GT) — use hardcoded mode", file=sys.stderr)
                sys.exit(1)
            if not expected_font:
                print(f"ERROR: p{page}:L{li} has no expected_font — use hardcoded mode", file=sys.stderr)
                sys.exit(1)
            if not text.endswith('.'):
                text += '.'
            print(f"p{page}:L{li}: text='{text}', expected={expected_font}")

            ttf_path, canonical_name, css_weight, css_style = resolve_expected_font(expected_font)
            print(f"  font: {ttf_path}, canonical: {canonical_name}")

            ttf_paths_for_map.append(ttf_path)
            font_face_entries.append((canonical_name, ttf_path, css_weight, css_style))
            lines.append((text, canonical_name, str(ttf_path), css_weight, css_style))

    # Enforce no empty lines
    for text, _, _, _, _ in lines:
        if not text.strip():
            print("ERROR: empty line detected — no empty lines allowed", file=sys.stderr)
            sys.exit(1)

    # Build HTML — shared @font-face generation
    font_face_css = build_font_face_css(font_face_entries)
    font_size = 9  # pt

    PAGE_BREAK = '<div style="page-break-before: always"></div>'
    lines_html = []
    for idx, (text, canonical_name, ttf_path, css_weight, css_style) in enumerate(lines):
        if idx > 0 and idx % 8 == 0:
            lines_html.append(PAGE_BREAK)
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
  margin: 0.6in 0.75in;
}}

body {{
  font-size: {font_size}pt;
  line-height: 1.15;
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

    # Render with WeasyPrint (shared helper)
    gt_pdf = str(SCRIPT_DIR / "line-test-gt.pdf")
    render_html_to_pdf(html, gt_pdf, base_url=SCRIPT_DIR)
    print(f"wrote: {gt_pdf}")

    # Build canonical map from embedded font data (same as gen-specimen — robust for italic/variable)
    canonical_map = build_canonical_map_from_pdf(gt_pdf, ttf_paths_for_map)

    # Annotate /UnprintCanonical
    annotated, missing = annotate_canonical_names(gt_pdf, canonical_map)
    print(f"annotated {annotated} fonts, missing: {missing}")

    # Rasterize
    rast_pdf = str(SCRIPT_DIR / "line-test.pdf")
    rasterize(gt_pdf, rast_pdf)
    print(f"wrote: {rast_pdf}")


if __name__ == "__main__":
    main()