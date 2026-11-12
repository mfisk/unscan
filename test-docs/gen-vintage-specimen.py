#!/usr/bin/env python3
"""Generate the vintage specimen: real PostScript-era fonts (URW Base 35),
all on ONE page.

The specimen renders the SAME test text in every variant (apples-to-apples:
the only thing that varies between sections is the font/variant, not the
text). All 5 variants share a single letter page, one section per variant.
Each section is SELF-DOCUMENTING: a header at the top states the font name
and variant, the era, HOW that vintage is recreated (clone, reconstructing,
or quantizing — see src/vintage_cache.rs module docs), and the key
characteristics (kern pair count, kerning mechanism, outline format), so a
reader opening the PDF can see exactly what each section is and why it is
distinct, without consulting any other file. 5 sections x 8 lines (5 header
+ 3 test) = 40 lines. The Recreation line is long and may wrap to two
visual lines; the other header lines fit on one line each.

The header lines are NOT scored by the t66 test: they are rendered in
DejaVu Sans (a font distinct from the test fonts) and annotated with the
sentinel canonical "HEADER-NO-SCORE". The test skips audit entries whose
expected font is the sentinel. Only the repeated test text lines are scored.

Variants (in section order, top to bottom):
  Times New Roman Regular (URW clone: Nimbus Roman Regular)
  Times New Roman Bold (URW clone: Nimbus Roman Bold)
  Times New Roman Italic (URW clone: Nimbus Roman Italic)
  Courier Regular (URW clone: Nimbus Mono PS Regular) — NO kerning (nokern)
  Times New Roman Word6 — SYNTHETIC vintage variant (quantized kern),
                          the one era that still needs synthetic generation
                          because no real font has quantized kern.

The headers lead with the HISTORICAL font name (Times New Roman, Courier),
not the URW clone name — "Nimbus" is URW's name for their clones, not the
historical name. Each header notes the URW clone used as the
metric-compatible stand-in.

Font keys are resolved verbatim via --resolve-fontkey for the real URW
fonts (Rust is the single source of truth; Python never constructs font
keys). The Word6 key comes verbatim from test-docs/vintage-times-keys.json
(written by tests/vintage_key_dump.rs via the Rust vintage code path).
CSS is derived from the resolved key: font-kerning normal iff the key
carries |kern| (the Word6 key carries no kern tag — it has a kern table,
so kerning is on), font-variant-ligatures normal iff the key carries
|lig|.

Characteristic numbers (kern pair counts, mechanism, outline format) are
read from the actual font files with fontTools at generation time — never
hardcoded. Verified values: Nimbus Roman Regular has 3845 GPOS PairPos
pairs (matching the AFM's 3845 KPX entries); Word6 has 867 kern-table
pairs, all values multiples of 5 (the original Times has 704 of 867
values NOT multiples of 5).

Layout: test text at 11pt (large enough for the pipeline to segment and
match accurately), headers at 8pt, tight but legible spacing. The
generator asserts the output is exactly one page and fails loudly if
the content overflows.

Recreation lines (one per variant, stated in the header):
  URW fonts: clone — the real URW font is a metric-compatible clone of
    the historical font (Nimbus Roman -> Times New Roman, Nimbus Mono PS
    -> Courier), used as a stand-in for the historical font; real OTF
    from the URW Base 35 release (20200910). No modification.
  Word6: quantizing — real Times New Roman with kern values quantized to
    multiples of 5 (Word 6.0 behavior), via the vintage_cache.rs Word6
    transform. Zero-valued pairs dropped.

Usage:
  python3 test-docs/gen-vintage-specimen.py
Outputs:
  test-docs/vintage-specimen-gt.pdf      (single page, GT-annotated)
  test-docs/vintage-specimen-rasterized.pdf
  test-docs/vintage-font-keys.json       (resolved keys, for the record)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from gen_common import (
    escape_html as _escape,
    build_font_face_css,
    css_family_name,
    render_html_to_pdf,
    build_canonical_map_from_pdf,
)
from pdf_font_annotate import annotate_canonical_names
from rasterize import rasterize

# Same text for every variant. All three lines are kern-dense in Nimbus
# Roman (AV/VA, WA/YA, AT, Ry, Tr all kern in its 3845-pair GPOS table),
# so kerning is exercised on the kern fonts. The Nimbus Mono section
# renders the same lines with no kerning (nokern key).
LINES = [
    "AVAST! WAYWARD YOUTH TO YONDER VALLEY, AWAY AT DAWN. Ryan Tries.",
    "Ryan Will Write To Lynn At Dawn. Try To Turn.",
    "Mr. Ryan Water Is Warm, 0123456789. Try To Turn.",
]

# Era label for the real URW fonts (descriptive metadata, not part of the key).
ERA_LABEL = "PostScript Level 2 (1991)"

# Header font (distinct from every test font) and the sentinel canonical
# that marks header lines as unscored. The t66 test skips audit entries
# whose expected font is this sentinel.
HEADER_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
HEADER_SENTINEL = "HEADER-NO-SCORE"

# (slug, display name, OTF filename, CSS weight, CSS style, family name,
# original font this URW font clones). Order defines the section order on
# the single page, top to bottom. The display name leads with the
# HISTORICAL font name and notes the URW clone in parentheses — "Nimbus"
# is URW's name for their clones, not the historical name. The
# family/original names feed the "Recreation: clone" header line.
URW_DIR = Path(os.path.expanduser("~/.local/share/fonts/urw-base35"))
FONTS = [
    ("nimbus-roman-regular", "Times New Roman Regular (URW clone: Nimbus Roman Regular)", "NimbusRoman-Regular.otf", "400", "normal", "Nimbus Roman", "Times New Roman"),
    ("nimbus-roman-bold", "Times New Roman Bold (URW clone: Nimbus Roman Bold)", "NimbusRoman-Bold.otf", "700", "normal", "Nimbus Roman", "Times New Roman"),
    ("nimbus-roman-italic", "Times New Roman Italic (URW clone: Nimbus Roman Italic)", "NimbusRoman-Italic.otf", "400", "italic", "Nimbus Roman", "Times New Roman"),
    ("nimbus-mono-regular", "Courier Regular (URW clone: Nimbus Mono PS Regular)", "NimbusMonoPS-Regular.otf", "400", "normal", "Nimbus Mono PS", "Courier"),
]

# Word6 recreation line (the one synthetic variant: quantizing).
WORD6_RECREATION = (
    "quantizing — real Times New Roman with kern values quantized to "
    "multiples of 5 (Word 6.0 behavior), via the vintage_cache.rs Word6 "
    "transform. Zero-valued pairs dropped."
)

# URW Base 35 release the clone fonts were downloaded from (real OTF,
# ArtifexSoftware/urw-base35-fonts). Stated in each clone recreation line.
URW_RELEASE = "20200910"

# Single-page layout: test text stays at 11pt (large enough to segment
# and match accurately); headers at 8pt (metadata, still legible);
# compact but legible spacing.
FONT_SIZE_PT = 11
HEADER_SIZE_PT = 8


def find_unprint_bin():
    for bp in [
        REPO_ROOT / "target" / "fastbuild" / "unprint",
        REPO_ROOT / "target" / "release" / "unprint",
        REPO_ROOT / "target" / "debug" / "unprint",
    ]:
        if bp.exists():
            return str(bp)
    print("ERROR: unprint binary not found, cannot resolve verbatim fontkeys", file=sys.stderr)
    sys.exit(1)


def resolve_fontkey(unprint_bin, otf_path):
    """Verbatim font_key from Rust via --resolve-fontkey (source of truth)."""
    r = subprocess.run(
        [unprint_bin, "--resolve-fontkey", str(otf_path)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"ERROR: --resolve-fontkey failed for {otf_path}: {r.stderr[:300]}", file=sys.stderr)
        sys.exit(1)
    # --resolve-fontkey prints scan lines to stdout before the key; the key
    # is the last non-empty line.
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        print(f"ERROR: --resolve-fontkey returned no output for {otf_path}", file=sys.stderr)
        sys.exit(1)
    return lines[-1]


def font_characteristics(font_path, quantized=False):
    """Read kern/outline characteristics from the actual font file.

    Returns (kern_desc, outline_fmt) where kern_desc is a human-readable
    string like "GPOS, 3845 pairs" or "none (0 pairs)". Numbers come from
    fontTools reads of this file, never hardcoded.
    """
    from fontTools.ttLib import TTFont
    f = TTFont(str(font_path))
    # GPOS PairPos pair count.
    gpos_pairs = 0
    if "GPOS" in f:
        for lookup in f["GPOS"].table.LookupList.Lookup:
            if lookup.LookupType == 2:
                for st in lookup.SubTable:
                    if st.Format == 1:
                        for ps in st.PairSet:
                            gpos_pairs += len(ps.PairValueRecord)
                    elif st.Format == 2:
                        for row in st.Class1Record:
                            for rec in row.Class2Record:
                                if rec.Value1 or rec.Value2:
                                    gpos_pairs += 1
    # Legacy kern table pair count.
    kern_pairs = 0
    if "kern" in f:
        for st in f["kern"].kernTables:
            tab = getattr(st, "kernTable", None) or getattr(st, "kernTab", None)
            if tab:
                kern_pairs += len(tab)
    outline = "CFF" if "CFF " in f else ("TrueType" if "glyf" in f else "?")
    f.close()
    if gpos_pairs > 0:
        kern_desc = f"GPOS, {gpos_pairs} pairs"
    elif kern_pairs > 0:
        if quantized:
            kern_desc = f"legacy kern table, {kern_pairs} pairs, values quantized to multiples of 5"
        else:
            kern_desc = f"legacy kern table, {kern_pairs} pairs"
    else:
        kern_desc = "none (0 pairs)"
    return kern_desc, outline


def main():
    unprint_bin = find_unprint_bin()
    print(f"unprint: {unprint_bin}")
    if not Path(HEADER_FONT_PATH).exists():
        print(f"ERROR: header font missing: {HEADER_FONT_PATH}", file=sys.stderr)
        sys.exit(1)

    # Section specs: (slug, display_name, font_path, font_key, weight,
    # style, kern_on, lig_on, era, kern_desc, outline_fmt, recreation).
    # Real URW fonts resolve via --resolve-fontkey; the Word6 synthetic
    # variant comes verbatim from vintage-times-keys.json (Rust-written).
    # The recreation string names the method (clone / quantizing) and
    # briefly explains it; it is printed in the section header.
    sections = []
    for slug, display_name, otf_name, weight, style, family, original in FONTS:
        otf_path = URW_DIR / otf_name
        if not otf_path.exists():
            print(f"ERROR: OTF missing: {otf_path}", file=sys.stderr)
            sys.exit(1)
        font_key = resolve_fontkey(unprint_bin, otf_path)
        # CSS derived from the resolved key, not hardcoded per font.
        kern_on = "|kern|" in font_key
        lig_on = "|lig|" in font_key
        kern_desc, outline_fmt = font_characteristics(otf_path)
        recreation = (
            f"clone — URW {family} is a metric-compatible clone of {original}, "
            "used as a stand-in for the historical font. "
            f"Real OTF, URW Base 35 ({URW_RELEASE}). No modification."
        )
        print(f"{slug}: {font_key} (kern={'on' if kern_on else 'off'}, lig={'on' if lig_on else 'off'}; {kern_desc}; {outline_fmt})")
        sections.append((slug, display_name, str(otf_path), font_key, weight, style, kern_on, lig_on, ERA_LABEL, kern_desc, outline_fmt, recreation))

    # Word6 synthetic variant (Times New Roman, kern quantized to multiples
    # of 5). No real font exhibits quantized kern, so this era still needs
    # synthetic generation via vintage_cache.rs. Key + path verbatim from
    # vintage-times-keys.json (written by tests/vintage_key_dump.rs).
    keys_path = SCRIPT_DIR / "vintage-times-keys.json"
    if not keys_path.exists():
        print(f"ERROR: {keys_path} not found — run the vintage_key_dump test first", file=sys.stderr)
        sys.exit(1)
    with open(keys_path) as f:
        vintage_keys = json.load(f)
    if "word6" not in vintage_keys:
        print(f"ERROR: era 'word6' missing from {keys_path}", file=sys.stderr)
        sys.exit(1)
    w6_path = vintage_keys["word6"]["path"]
    w6_key = vintage_keys["word6"]["font_key"]
    if not Path(w6_path).exists():
        print(f"ERROR: Word6 TTF missing: {w6_path}", file=sys.stderr)
        sys.exit(1)
    w6_kern_desc, w6_outline = font_characteristics(w6_path, quantized=True)
    print(f"times-word6: {w6_key} (kern=on, lig=off; {w6_kern_desc}; {w6_outline})")
    sections.append(("times-word6", "Times New Roman — Word6 (quantized kern)", w6_path, w6_key, "400", "normal", True, False, "Word 6.0 (1993)", w6_kern_desc, w6_outline, WORD6_RECREATION))

    # Build the @font-face CSS for all test fonts + the header font.
    font_face_entries = [(key, path, weight, style) for _, _, path, key, weight, style, _, _, _, _, _, _ in sections]
    font_face_css = build_font_face_css(font_face_entries)
    header_face_css = build_font_face_css([("HEADER", HEADER_FONT_PATH, "400", "normal")])
    header_family = css_family_name("HEADER")

    # Render all 5 sections into one HTML document (single page).
    body_html = []
    for i, (slug, display_name, font_path, font_key, weight, style, kern_on, lig_on, era, kern_desc, outline_fmt, recreation) in enumerate(sections):
        family = css_family_name(font_key)
        kerning_css = "font-kerning: normal; " if kern_on else "font-kerning: none; "
        ligatures_css = "font-variant-ligatures: normal; " if lig_on else "font-variant-ligatures: none; "

        # Self-documenting header: variant, era, recreation method, and
        # characteristics. Rendered in the header font (DejaVu Sans).
        # These lines are annotated with the HEADER-NO-SCORE sentinel and
        # are not scored. The Recreation line is long (~150 chars) and
        # wraps to two visual lines; the other header lines fit on one
        # line each (white-space: nowrap).
        header_lines = [
            (f"Variant: {display_name}", True),
            (f"Era: {era}", True),
            (f"Recreation: {recreation}", False),
            (f"Kerning: {kern_desc}", True),
            (f"Outlines: {outline_fmt}", True),
        ]
        for htext, nowrap in header_lines:
            ws = "nowrap" if nowrap else "normal"
            body_html.append(
                f'<p style="font-family: \'{header_family}\'; '
                f'font-weight: 400; font-style: normal; '
                f'font-size: {HEADER_SIZE_PT}pt; white-space: {ws}; margin:0; padding:0; line-height: 1.15;">'
                f'{_escape(htext)}</p>'
            )
        body_html.append('<hr style="border:none; border-top:1pt solid #333; margin:3pt 0 5pt 0; height:0;" />')

        for idx, text in enumerate(LINES):
            body_html.append(
                f'<p style="font-family: \'{family}\'; '
                f'font-weight: {weight}; font-style: {style}; '
                f'{kerning_css}'
                f'font-variant-caps: normal; '
                f'font-variant-numeric: lining-nums; '
                f'{ligatures_css}'
                f'font-feature-settings: "ss01" 0, "ss02" 0, "case" 0, "cpsp" 0, "dlig" 0; '
                f'font-size: {FONT_SIZE_PT}pt; white-space: nowrap; margin:0; padding:0; line-height: 1.5;">'
                f'{_escape(text)}</p>'
            )
            if idx < len(LINES) - 1:
                body_html.append('<hr style="border:none; border-top:0.5pt solid #999; margin:2pt 0; height:0;" />')

        # Section separator (a thicker rule with breathing room), except
        # after the last section.
        if i < len(sections) - 1:
            body_html.append('<hr style="border:none; border-top:1.5pt solid #666; margin:4pt 0; height:0;" />')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{font_face_css}
{header_face_css}
@page {{
  size: 8.5in 11in;
  margin: 0.5in 0.75in;
}}
body {{
  font-size: {FONT_SIZE_PT}pt;
  line-height: 1.5;
  color: black;
}}
p {{
  margin: 0;
  padding: 0;
}}
</style>
</head>
<body>
{chr(10).join(body_html)}
</body>
</html>"""
    gt_pdf = str(SCRIPT_DIR / "vintage-specimen-gt.pdf")
    render_html_to_pdf(html, gt_pdf, base_url=SCRIPT_DIR)
    print(f"rendered: {gt_pdf}")

    # The specimen must fit on exactly ONE page. Fail loudly on overflow.
    import pikepdf
    probe = pikepdf.open(gt_pdf)
    n_pages = len(probe.pages)
    probe.close()
    if n_pages != 1:
        print(f"ERROR: specimen has {n_pages} pages, expected exactly 1 — tighten the layout", file=sys.stderr)
        sys.exit(1)
    print(f"  page count: 1 (OK)")

    # GT annotation: each test font gets its verbatim key; the header font
    # gets the HEADER-NO-SCORE sentinel. All 6 fonts have distinct PS
    # names, so the PS-name-keyed map is unambiguous.
    catalog_by_path = {HEADER_FONT_PATH: {"font_key": HEADER_SENTINEL, "path": HEADER_FONT_PATH}}
    all_font_files = [HEADER_FONT_PATH]
    for _, _, font_path, font_key, _, _, _, _, _, _, _, _ in sections:
        catalog_by_path[str(font_path)] = {"font_key": font_key, "path": str(font_path)}
        all_font_files.append(str(font_path))
    canonical_map = build_canonical_map_from_pdf(
        gt_pdf, all_font_files, catalog_by_path
    )
    annotated, missing = annotate_canonical_names(gt_pdf, canonical_map)
    print(f"  annotated {annotated} fonts, missing: {missing}")
    if annotated < 6:
        print(f"ERROR: expected 6 fonts annotated in {gt_pdf}, got {annotated}", file=sys.stderr)
        sys.exit(1)

    # Write the resolved keys for the record (the test reads expected keys
    # from the audit's expected_font, but this documents the mapping).
    resolved = {}
    for slug, display_name, font_path, font_key, weight, style, _, _, era, kern_desc, outline_fmt, recreation in sections:
        resolved[slug] = {
            "font_key": font_key,
            "path": str(font_path),
            "era": era,
            "weight": weight,
            "style": style,
            "kerning": kern_desc,
            "outlines": outline_fmt,
            "recreation": recreation,
        }
    keys_out = SCRIPT_DIR / "vintage-font-keys.json"
    with open(keys_out, "w") as f:
        json.dump(resolved, f, indent=2, sort_keys=True)
    print(f"wrote: {keys_out}")

    # Verify the single page carries all 5 test-font canonicals plus the
    # header sentinel.
    merged = pikepdf.open(gt_pdf)
    page_canon = []
    page = merged.pages[0]
    if "/Resources" in page and "/Font" in page.Resources:
        for font_ref in page.Resources.Font.values():
            font = merged.get_object(font_ref.objgen)
            if "/UnprintCanonical" in font:
                page_canon.append(str(font.UnprintCanonical))
    merged.close()
    print(f"  page 1 canonicals: {page_canon}")
    expected = [s[3] for s in sections]
    test_canon = [c for c in page_canon if c != HEADER_SENTINEL]
    if sorted(test_canon) != sorted(expected):
        print(
            f"ERROR: page test-font canonicals {test_canon} do not match expected {expected}",
            file=sys.stderr,
        )
        sys.exit(1)
    if HEADER_SENTINEL not in page_canon:
        print(
            f"ERROR: header sentinel missing from page canonicals: {page_canon}",
            file=sys.stderr,
        )
        sys.exit(1)

    rast_pdf = str(SCRIPT_DIR / "vintage-specimen-rasterized.pdf")
    rasterize(gt_pdf, rast_pdf)
    print(f"rasterized: {rast_pdf}")
    print("DONE")


if __name__ == "__main__":
    main()
