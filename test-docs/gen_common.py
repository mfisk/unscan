#!/usr/bin/env python3
"""gen_common.py — Shared PDF generation utilities for WeasyPrint-based specimen/test PDFs.

This module is the single source of truth for:
  - font family registry (FAMILIES: short code -> fontconfig family)
  - fontconfig resolution (fc_find)
  - HTML escaping, @font-face CSS, WeasyPrint rendering
  - PDF canonical-name annotation / rasterization helpers

Both gen-specimen.py and gen-line-test.py import from here to minimize
duplication and arbitrary divergence.
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Ensure tools/ is on path for pdf_font_annotate / rasterize
sys.path.insert(0, str(REPO_ROOT / "tools"))
from pdf_font_annotate import make_weight_explicit, annotate_canonical_names, read_postscript_name  # noqa: E402


# ---------------------------------------------------------------------------
# Font families (short code -> fontconfig family)
# ---------------------------------------------------------------------------
# Single definition used by gen-specimen (SECTIONS) and gen-line-test (resolve).
FAMILIES = {
    # Google Fonts
    "EBGaramond": "EB Garamond",
    "LibreCaslonText": "Libre Caslon Text",
    "LibreBaskerville": "Libre Baskerville",
    "LibreBodoni": "Libre Bodoni",
    "ZillaSlab": "Zilla Slab",
    "Jost": "Jost",
    "PlayfairDisplay": "Playfair Display",
    "Roboto": "Roboto",
    "OpenSans": "Open Sans",
    "Lato": "Lato",
    "Merriweather": "Merriweather",
    "SourceSans3": "Source Sans 3",
    "SourceSerif4": "Source Serif 4",
    "NotoSerif": "Noto Serif",
    "PTSerif": "PT Serif",
    "IBMPlexSans": "IBM Plex Sans",
    "IBMPlexSerif": "IBM Plex Serif",
    "IBMPlexMono": "IBM Plex Mono",
    "Inter": "Inter",
    "SpecialElite": "Special Elite",
    # System / MS Core Fonts
    "TimesNewRoman": "Times New Roman",
    "CourierNew": "Courier New",
    "NimbusSans": "Arial",
    "Arial": "Arial",
    "Georgia": "Georgia",
    "Verdana": "Verdana",
    "ComicSansMS": "Comic Sans MS",
    "TrebuchetMS": "Trebuchet MS",
    "Caladea": "Caladea",
    "PrestigeElite": "Prestige Elite",
    # Aliases that appear in audit expected_font values
    "ArialMT": "Arial",
    "Arial-BoldMT": "Arial",
    "CourierNewPSMT": "Courier New",
    "TimesNewRomanPSMT": "Times New Roman",
    "PrestigeEliteNormal": "Prestige Elite",
    "NimbusSansL": "Arial",  # common alias for Nimbus Sans on some systems
}

# Inverse for debugging / display
FAMILY_BY_FC = {v: k for k, v in FAMILIES.items()}


# ---------------------------------------------------------------------------
# Fontconfig resolution
# ---------------------------------------------------------------------------
def fc_find(family, style="Regular"):
    """Find a font file via fontconfig for the given family and style.

    Resolution is strict and deterministic:
    1. Query fontconfig for candidates matching family + style.
    2. Filter to normal-width (.ttf) candidates with the exact target weight.
       Prefer static fonts over variable fonts at this step.
    3. If no exact-weight static font, find a variable font that covers the
       target weight on its wght axis and return it (Pango handles variable).
    4. If none of the above, raise.

    No silent fallbacks — caller must handle fallbacks explicitly (e.g. Bold
    -> ExtraBold -> ... -> Regular as gen-specimen does).
    """
    from fontTools.ttLib import TTFont as FTFont

    STYLE_WEIGHTS = {
        "Regular": 400, "Bold": 700, "Light": 300,
        "Medium": 500, "SemiBold": 600, "ExtraBold": 800, "Black": 900,
        "Italic": 400, "Bold Italic": 700,
    }
    target_weight = STYLE_WEIGHTS.get(style, 400)
    NORMAL_WIDTH = 5  # usWidthClass: 5 = normal

    candidates = []
    for query in [f"{family}:style={style}", family]:
        r = subprocess.run(
            ["fc-list", query, "--format=%{file}\n"],
            capture_output=True, text=True,
        )
        for line in r.stdout.strip().split("\n"):
            if line and line.lower().endswith(".ttf") and line not in candidates:
                candidates.append(line)

    if not candidates:
        raise RuntimeError(f"fc_find: no .ttf candidates for '{family}' style='{style}'")

    exact_static = []
    variable = []

    for path in candidates:
        try:
            tt = FTFont(path)
            wt = tt["OS/2"].usWeightClass
            wd = tt["OS/2"].usWidthClass
            fvar = tt.get("fvar")
            is_var = fvar is not None
            wght_range = None
            if is_var:
                for axis in fvar.axes:
                    if axis.axisTag == "wght":
                        wght_range = (axis.minValue, axis.maxValue)
                        break
            tt.close()
        except Exception:
            continue

        if wd != NORMAL_WIDTH:
            continue

        if not is_var and wt == target_weight:
            exact_static.append(path)
        elif is_var and wght_range and wght_range[0] <= target_weight <= wght_range[1]:
            variable.append(path)

    if exact_static:
        for p in exact_static:
            if "/specimen-fonts/" not in p:
                return p
        return exact_static[0]

    if variable:
        return variable[0]

    raise RuntimeError(
        f"fc_find: no font for '{family}' style='{style}' (weight={target_weight}). "
        f"Candidates: {candidates}"
    )


def resolve_expected_font(expected_font):
    """Resolve an expected_font string from audit.json to (ttf_path, canonical, css_weight, css_style).

    expected_font examples:
      'EBGaramond-400', 'Arial-BoldMT-700', 'SourceSerif4-400It',
      'ArialMT-400', 'Inter-400'
    """
    # Strip weight/style suffixes to get base key for FAMILIES
    # Expected forms: Base-400, Base-400It, Base-BoldMT-700, etc.
    # First split on '-' to get first token as family key candidate
    font_base = expected_font.split('-')[0] if '-' in expected_font else expected_font
    # Handle known PS suffixes that contain family (e.g. Arial-BoldMT -> Arial)
    # Already covered by FAMILIES aliases, but also strip Italic/It markers
    for suffix in ['Italic', 'It']:
        # Only strip trailing occurrence, not middle
        if font_base.endswith(suffix):
            font_base = font_base[: -len(suffix)]
            break
    # Also handle if font_base still has -700 etc handled by earlier split; keep aliases
    fc_family = FAMILIES.get(font_base)
    if not fc_family:
        # Try removing MT/PS suffixes
        base2 = font_base.replace('MT', '').replace('PS', '')
        fc_family = FAMILIES.get(base2)
    if not fc_family:
        raise RuntimeError(f"'{font_base}' (from '{expected_font}') not in FAMILIES; add alias")

    style = "Regular"
    css_weight = "400"
    css_style = "normal"
    if "Bold" in expected_font or "-700" in expected_font or "-600" in expected_font:
        style = "Bold"
        # TTF is already bold — tell Pango font-weight: normal so it doesn't append "-Bold"
        css_weight = "400"
    elif "Italic" in expected_font or expected_font.endswith("It") or "-400It" in expected_font:
        style = "Italic"
        css_style = "italic"

    ttf_path = fc_find(fc_family, style)
    _, canonical_name, _ = make_weight_explicit(ttf_path)
    return ttf_path, canonical_name, css_weight, css_style


# ---------------------------------------------------------------------------
# HTML / CSS helpers
# ---------------------------------------------------------------------------
def escape_html(text):
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def font_face_rule(family, ttf_path, weight="400", style="normal"):
    return (
        f"@font-face {{\n"
        f"  font-family: '{family}';\n"
        f"  src: url('file://{ttf_path}') format('truetype');\n"
        f"  font-weight: {weight};\n"
        f"  font-style: {style};\n"
        f"}}"
    )


def build_font_face_css(entries):
    """entries: iterable of (family, ttf_path, weight, style) or dicts.

    Deduplicates by (family, weight, style).
    """
    seen = set()
    rules = []
    for e in entries:
        if isinstance(e, dict):
            family = e["family"]; path = e["path"]; weight = e.get("weight", "400"); style = e.get("style", "normal")
        else:
            family, path, weight, style = e
        key = (family, weight, style, str(path))
        if key in seen:
            continue
        seen.add(key)
        rules.append(font_face_rule(family, path, weight, style))
    return "\n".join(rules)


def render_html_to_pdf(html, out_pdf, base_url=None):
    """Render HTML string to PDF via WeasyPrint."""
    import weasyprint
    doc = weasyprint.HTML(string=html, base_url=str(base_url) if base_url else None)
    doc.write_pdf(str(out_pdf))
    return out_pdf


# ---------------------------------------------------------------------------
# PDF canonical map building (extract embedded font PS names + weights)
# ---------------------------------------------------------------------------
def _extract_embedded_ps_and_weight(font_dict):
    font_desc = font_dict.get("/FontDescriptor")
    if font_desc is not None:
        ps, w = _read_ps_and_weight_from_descriptor(font_desc)
        if ps:
            return ps, w
    descendants = font_dict.get("/DescendantFonts")
    if descendants is not None:
        for desc in descendants:
            fd = desc.get("/FontDescriptor")
            if fd is not None:
                ps, w = _read_ps_and_weight_from_descriptor(fd)
                if ps:
                    return ps, w
    return None, None


def _read_ps_and_weight_from_descriptor(font_desc):
    import io
    from fontTools.ttLib import TTFont as FTFont
    for key in ("/FontFile2", "/FontFile3", "/FontFile"):
        stream = font_desc.get(key)
        if stream is None:
            continue
        try:
            data = bytes(stream.read_bytes())
            tt = FTFont(io.BytesIO(data))
            ps_name = ""
            for rec in tt["name"].names:
                if rec.nameID == 6:
                    try:
                        ps_name = rec.toUnicode()
                        break
                    except Exception:
                        pass
            os2_weight = None
            try:
                os2_weight = tt["OS/2"].usWeightClass
            except Exception:
                pass
            tt.close()
            if ps_name:
                return ps_name, os2_weight
        except Exception:
            continue
    return None, None


def build_canonical_map_from_pdf(pdf_path, all_font_files):
    """Build canonical_map by extracting PS names from embedded font data in PDF.

    Returns dict: raw BaseFont (subset-stripped) -> canonical PS name.
    """
    import pikepdf
    from fontTools.ttLib import TTFont as FTFont

    ps_to_canonical = {}
    ps_to_orig_weight = {}
    for path in all_font_files:
        orig_ps, canonical_ps, _ = make_weight_explicit(path)
        if orig_ps and canonical_ps:
            ps_to_canonical[orig_ps] = canonical_ps
            try:
                tt = FTFont(path)
                ps_to_orig_weight[orig_ps] = tt['OS/2'].usWeightClass
                tt.close()
            except Exception:
                pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    unprint_bin = None
    for bin_path in [
        os.path.join(repo_root, "target", "release", "unprint"),
        os.path.join(repo_root, "target", "release", "unscan"),
        os.path.join(repo_root, "target", "debug", "unprint"),
        os.path.join(repo_root, "target", "debug", "unscan"),
    ]:
        if os.path.exists(bin_path):
            unprint_bin = bin_path
            break
    if unprint_bin is None:
        unprint_bin = "unprint"

    canonical_map = {}
    pdf = pikepdf.open(str(pdf_path))
    for page in pdf.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for res_name in list(fonts.keys()):
            font_dict = fonts[res_name]
            base_font = font_dict.get("/BaseFont")
            if base_font is None:
                continue
            bf_str = str(base_font).lstrip("/")
            raw_bf = bf_str[7:] if len(bf_str) > 7 and bf_str[6] == '+' else bf_str
            if raw_bf in canonical_map:
                continue
            embedded_ps, embedded_weight = _extract_embedded_ps_and_weight(font_dict)
            if not embedded_ps:
                continue
            orig_weight = ps_to_orig_weight.get(embedded_ps)
            if embedded_weight is not None and orig_weight is not None and embedded_weight != orig_weight:
                r = subprocess.run(
                    [unprint_bin, "--weight-explicit", f"{embedded_ps}:{embedded_weight}"],
                    capture_output=True, text=True
                )
                canonical_map[raw_bf] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else embedded_ps
            elif embedded_ps in ps_to_canonical:
                canonical_map[raw_bf] = ps_to_canonical[embedded_ps]
            else:
                if embedded_weight is not None:
                    r = subprocess.run(
                        [unprint_bin, "--weight-explicit", f"{embedded_ps}:{embedded_weight}"],
                        capture_output=True, text=True
                    )
                    canonical_map[raw_bf] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else embedded_ps
                else:
                    canonical_map[raw_bf] = embedded_ps
    pdf.close()
    return canonical_map


def build_simple_canonical_map(ttf_paths):
    """Simple PS->canonical map from a list of TTF paths (used by line-test)."""
    m = {}
    for p in ttf_paths:
        orig_ps = read_postscript_name(str(p))
        if not orig_ps:
            continue
        _, canon, _ = make_weight_explicit(str(p))
        if canon:
            m[orig_ps] = canon
            m[canon] = canon
    return m


# ---------------------------------------------------------------------------
# Common PDF pipeline: annotate + rasterize
# ---------------------------------------------------------------------------
def annotate_and_rasterize(vector_pdf, rasterized_pdf, canonical_map):
    from rasterize import rasterize
    annotated, missing = annotate_canonical_names(str(vector_pdf), canonical_map)
    # Caller prints; return for logging
    rasterize(str(vector_pdf), str(rasterized_pdf))
    return annotated, missing
