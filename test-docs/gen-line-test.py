#!/usr/bin/env python3
"""Generate a multi-line test PDF using WeasyPrint (Pango/HarfBuzz).

Two modes:
  1) Hardcoded (new, used by lob + t59):
     python3 test-docs/gen-line-test.py --hardcoded

  2) Legacy audit mode (deprecated, fragile)
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

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
# Hardcoded 11-line test — page-break isolation for Tesseract LSTM reset
# p1: 8 lines (L01-L08), p2: 2 lines (LibreBaskerville + PTSerif italic fox with numbers),
# p3: 1 line (Matthew Carter) isolated to avoid bbox shift.
# Matches t59 + extensions requested: LibreBaskerville lower, PTSerif italic numbers.
# ---------------------------------------------------------------------------
HARDCODED = [
    ("LibreBodoni-400", "abcdefghijklmnopqrstuvwxyz."),
    ("LibreBodoni-400", "ABCDEFGHIJKLMNOPQRSTUVWXYZ."),
    ("Georgia-400", "ABCDEFGHIJKLMNOPQRSTUVWXYZ."),
    ("OpenSans-400", "abcdefghijklmnopqrstuvwxyz."),
    ("LibreBodoni-400Italic", "dogs."),
    ("SourceSerif4-400It", "Mayr-Duffner."),
    ("SourceSerif4-400It", "Type"),
    ("LibreBaskerville-400", "abcdefghijklmnopqrstuvwxyz."),
    ("PTSerif-400Italic", "Italic: The quick brown fox jumps over 1,234,567,890 lazy"),
    ("SourceSerif4-400It", "Font: Originally for IBM Executive typewriters — 12 characters per inch"),
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
            args = [f"{font}={text}" for font, text in HARDCODED]
    if args and any("=" in a for a in args):
        hardcoded = True
    return hardcoded, args, audit_ref

def main():
    hardcoded, args, audit_ref = parse_args()
    if len(args) < 1:
        print("Usage:")
        print("  gen-line-test.py --hardcoded")
        sys.exit(1)
    ttf_paths_for_map = []
    lines = []
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
                print(f"ERROR: empty line for font '{expected_font}'", file=sys.stderr)
                sys.exit(1)
            print(f"hardcoded: text='{text}', expected={expected_font}")
            ttf_path, canonical_name, css_weight, css_style = resolve_expected_font(expected_font)
            print(f"  font: {ttf_path}, canonical: {canonical_name}")
            ttf_paths_for_map.append(ttf_path)
            font_face_entries.append((canonical_name, ttf_path, css_weight, css_style))
            lines.append((text, canonical_name, str(ttf_path), css_weight, css_style))
    else:
        if ':' in args[0]:
            page_lines = [(int(a.split(':')[0]), int(a.split(':')[1])) for a in args]
        else:
            page = int(args[0])
            page_lines = [(page, int(a)) for a in args[1:]]
        with open(audit_ref) as f:
            audit = json.load(f)
        for (page, li) in page_lines:
            entries = [e for e in audit['text_entries'] if e.get('page') == page and e.get('line_index') == li]
            if not entries:
                print(f"ERROR: No entries for p{page}:L{li}", file=sys.stderr)
                sys.exit(1)
            entry = entries[0]
            expected_font = entry.get('expected_font', '')
            text = entry.get('gt_text', entry.get('text', entry.get('ocr_text', '')))
            if not text:
                print(f"ERROR: p{page}:L{li} has empty text", file=sys.stderr)
                sys.exit(1)
            if not expected_font:
                print(f"ERROR: p{page}:L{li} has no expected_font", file=sys.stderr)
                sys.exit(1)
            if not text.endswith('.'):
                text += '.'
            ttf_path, canonical_name, css_weight, css_style = resolve_expected_font(expected_font)
            ttf_paths_for_map.append(ttf_path)
            font_face_entries.append((canonical_name, ttf_path, css_weight, css_style))
            lines.append((text, canonical_name, str(ttf_path), css_weight, css_style))

    for text, _, _, _, _ in lines:
        if not text.strip():
            print("ERROR: empty line detected", file=sys.stderr)
            sys.exit(1)

    font_face_css = build_font_face_css(font_face_entries)

    # Single-page large-gap: 120pt blank div resets Tesseract line grouping
    # without multi-page drift. Keeping p1+p2 on same page preserves fox line's
    # word-gap statistics, which raises Tesseract's word-gap threshold and
    # prevents alphabet j/k split (abcdefghij / klmn...). Page-break isolation
    # makes p1 gap stats too small, causing alphabet to split.
    LARGE_GAP = '<div style="height:120pt;"></div>'
    lines_html = []
    for idx, (text, canonical_name, ttf_path, css_weight, css_style) in enumerate(lines):
        stripped = text.rstrip('.').strip()
        is_alpha = stripped in ("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        pt = 9 if is_alpha else 10
        # Keep first 10 lines (8 + 2) together to share word-gap stats and prevent
        # alphabet j/k split. Only isolate the final Matthew Carter line.
        if idx == len(lines) - 1 and len(lines) > 8:
            lines_html.append(LARGE_GAP)
        lines_html.append(
            f'<p style="font-family: \'{canonical_name}\'; '
            f'font-weight: {css_weight}; font-style: {css_style}; '
            f'font-size: {pt}pt; white-space: nowrap; margin:0; padding:0;">'
            f'{_escape(text)}</p>'
        )
        # hrule between each line for visual separation / tesseract line isolation
        if idx < len(lines) - 1:
            lines_html.append('<hr style="border:none; border-top:0.5pt solid #999; margin:14pt 0; height:0;" />')

    all_lines = "\n".join(lines_html)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{font_face_css}
@page {{
  size: 8.5in 11in;
  margin: 0.7in 0.75in;
}}
body {{
  font-size: 9pt;
  line-height: 2.0;
  color: black;
}}
p {{
  margin: 0;
  padding: 0;
  line-height: 2.0;
}}
</style>
</head>
<body>
{all_lines}
</body>
</html>"""

    gt_pdf = str(SCRIPT_DIR / "line-test-gt.pdf")
    render_html_to_pdf(html, gt_pdf, base_url=SCRIPT_DIR)
    print(f"wrote: {gt_pdf}")
    canonical_map = build_canonical_map_from_pdf(gt_pdf, ttf_paths_for_map)
    annotated, missing = annotate_canonical_names(gt_pdf, canonical_map)
    print(f"annotated {annotated} fonts, missing: {missing}")
    rast_pdf = str(SCRIPT_DIR / "line-test.pdf")
    rasterize(gt_pdf, rast_pdf)
    print(f"wrote: {rast_pdf}")

if __name__ == "__main__":
    main()
