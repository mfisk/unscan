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


def resolve_fontkey_via_api(unprint_bin, font_path):
    """Get verbatim font_key from Rust catalog via --resolve-fontkey.

    This is the single source of truth for fontkeys. Never construct
    fontkeys in Python; always ask Rust.
    """
    result = subprocess.run(
        [unprint_bin, "--resolve-fontkey", font_path],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"--resolve-fontkey failed for {font_path}: {result.stderr[:200]}"
        )
    return result.stdout.strip()

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
            if line and line.lower().endswith((".ttf", ".otf")) and line not in candidates:
                candidates.append(line)

    if not candidates:
        raise RuntimeError(f"fc_find: no .ttf/.otf candidates for '{family}' style='{style}'")

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
    try:
        _, canonical_name, _ = make_weight_explicit(ttf_path)
    except FileNotFoundError:
        # fallback if unprint not built - use PS name directly
        canonical_name = read_postscript_name(ttf_path) or fc_family.replace(' ','')
    return ttf_path, canonical_name, css_weight, css_style


# ---------------------------------------------------------------------------
# HTML / CSS helpers
# ---------------------------------------------------------------------------
def escape_html(text):
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _font_face_rule(family, ttf_path, weight="400", style="normal"):
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
        rules.append(_font_face_rule(family, path, weight, style))
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


_gpos_cache = {}


def _font_has_gpos(path):
    """True if the font file at path contains a GPOS table.

    Cached per path. Mirrors font_scan::font_has_gpos in Rust. Used to
    give _resolve_score the same GPOS-aware preference as --resolve-fontkey.
    Unreadable files default to False (prefer nokern), matching Rust's
    fail-closed parse behavior.
    """
    if path in _gpos_cache:
        return _gpos_cache[path]
    has = False
    try:
        from fontTools.ttLib import TTFont as FTFont
        with FTFont(path, lazy=True) as tt:
            has = "GPOS" in tt
    except Exception:
        has = False
    _gpos_cache[path] = has
    return has


def _is_ot_feature_tag(seg):
    """True if seg is an OpenType feature tag selecting a non-default
    rendering (onum, lnum, smcp, c2sc, swsh, salt, titl, hist, ss01-ss20,
    allcaps). Mirrors font_scan::is_ot_feature_tag in Rust."""
    if seg in ("onum", "lnum", "smcp", "c2sc", "swsh", "salt", "titl",
               "hist", "allcaps"):
        return True
    return len(seg) == 4 and seg.startswith("ss") and seg[2:].isdigit()


def _resolve_score(font_key, has_gpos=False):
    """Mirror of the shared font-key preference in Rust
    (src/font_scan.rs font_key_preference_score, used by --resolve-fontkey
    and the classifier tiebreakers): GPOS font +4 kern, non-GPOS +4 nokern,
    nolig +2, opsz=10 +1, -10 per OT-feature variant tag (onum, lnum,
    smcp, c2sc, swsh, salt, titl, hist, ss01-ss20, allcaps) and -10 per
    non-default weight instance (`|wghtN`), so the base/default variant
    always wins the default-GT pick over an OT-feature variant or a
    non-default weight. Higher score wins. Keep in sync with Rust —
    Python must not invent its own preference."""
    s = 0
    is_nokern = "nokern" in font_key
    # If has_gpos: +4 for kern (not nokern).
    # If !has_gpos: +4 for nokern.
    if has_gpos != is_nokern:
        s += 4
    if "nolig" in font_key:
        s += 2  # prefer nolig
    if "opsz=10" in font_key:
        s += 1  # tiebreaker
    # Non-default variants must never be the default pick: the specimen
    # renders default features at the font's default weight. Segment 0 is
    # the PostScript name, never a tag; bracketed variations
    # (`[wght=400,...]`) are stripped before matching.
    for seg in font_key.split("|")[1:]:
        seg = seg.split("[", 1)[0]
        if _is_ot_feature_tag(seg):
            s -= 10
        if len(seg) > 4 and seg.startswith("wght") and seg[4:].isdigit():
            s -= 10
    return s


def build_canonical_map_from_pdf(pdf_path, all_font_files, catalog_json=None, font_entries=None, unprint_bin=None):
    """Build canonical_map by extracting PS names from embedded font data in PDF.

    Returns dict: raw BaseFont (subset-stripped) -> canonical PS name.

    If font_entries is provided (dict from gen-specimen.py: base -> {variant: entry}),
    uses it directly to map PDF fonts to the EXACT fontkeys used during generation.
    This is the correct approach for variable fonts where multiple weights share
    the same PS name — the PDF BaseFont name (e.g., "Source-Serif-4-Bold") is parsed
    to get the variant, then looked up in font_entries to get the full fontkey
    (including axes like [opsz=20]).

    If catalog_json is provided (path to unprint --export-catalog-json output),
    uses the catalog's font_key directly (preferring the --resolve-fontkey
    preference: kern, nolig, then opsz=10) instead
    of reconstructing via make_weight_explicit. This avoids duplicating the
    Rust catalog logic in Python.

    If neither is provided, falls back to legacy behavior (may be incorrect for
    variable fonts).
    """
    import pikepdf
    from fontTools.ttLib import TTFont as FTFont
    import json
    import re

    # (Removed _normalize_ps and _strip_weight per architectural change #3:
    #  font_key is opaque, use exact matching only. No heuristics.)

    # If font_entries provided, build direct mapping from PDF font names
    # This is the CORRECT approach — uses the exact entries from generation
    if font_entries:
        # Build a lookup: normalized family name -> base key in font_entries
        # font_entries keys are like "SourceSerif4", PDF names are like "Source-Serif-4-Bold"
        def _norm_pdf_family(pdf_name):
            # Remove variant suffixes: -Bold, -Italic, -BoldItalic, etc.
            # Also handle trailing comma (WeasyPrint quirk for regular)
            name = pdf_name.strip()
            # Remove trailing comma (WeasyPrint quirk)
            if name.endswith(','):
                name = name[:-1]
            # Remove variant suffixes
            for suffix in ['-BoldItalic', '-Bold', '-Italic', '-Oblique', '-Regular']:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            # Normalize: remove dashes, lowercase
            return re.sub(r'[^a-z0-9]', '', name.lower())
        
        def _parse_variant(pdf_name):
            # Extract variant from PDF font name
            # e.g., "Source-Serif-4-Bold" -> "bold", "Source-Serif-4-Italic" -> "italic"
            name_lower = pdf_name.lower()
            if 'bolditalic' in name_lower or ('bold' in name_lower and 'italic' in name_lower):
                return 'bold'  # Use bold entry (italic bold not separately tracked)
            elif 'bold' in name_lower:
                return 'bold'
            elif 'italic' in name_lower or 'oblique' in name_lower:
                return 'italic'
            else:
                return 'regular'
        
        # Build family lookup
        family_to_base = {}
        for base in font_entries.keys():
            # Normalize base: "SourceSerif4" -> "sourceserif4"
            norm_base = re.sub(r'[^a-z0-9]', '', base.lower())
            family_to_base[norm_base] = base
        
        # Parse PDF and build map
        pdf = pikepdf.open(pdf_path)
        canonical_map = {}
        pdf_font_names = set()
        for page in pdf.pages:
            if '/Resources' in page and '/Font' in page.Resources:
                for font_ref in page.Resources.Font.values():
                    try:
                        font = pdf.get_object(font_ref.objgen)
                    except:
                        font = font_ref
                    if '/BaseFont' in font:
                        raw_bf = str(font.BaseFont)
                        # Strip subset prefix
                        clean_bf = raw_bf.split('+')[-1]
                        pdf_font_names.add((raw_bf, clean_bf))
        pdf.close()
        
        for raw_bf, clean_bf in pdf_font_names:
            # Parse family and variant from clean_bf
            norm_family = _norm_pdf_family(clean_bf)
            variant = _parse_variant(clean_bf)
            
            base = family_to_base.get(norm_family)
            if not base:
                # PDF font not in specimen (e.g., DejaVu-Sans for UI/fallback).
                # Skip it — we only need mappings for specimen fonts.
                # The legacy catalog-based path below will handle it if needed.
                continue
            
            variants_dict = font_entries.get(base, {})
            entry = variants_dict.get(variant)
            if not entry:
                # Fallback to regular if variant not found (e.g., bolditalic -> bold)
                entry = variants_dict.get('regular')
            if not entry or not isinstance(entry, dict):
                raise RuntimeError(
                    f"No font entry for '{base}' variant '{variant}' (PDF font '{raw_bf}'). "
                    f"Fix the specimen."
                )

            # Use --resolve-fontkey (Rust as single source of truth) if available,
            # otherwise fall back to catalog's font_key (may be stale).
            if unprint_bin and entry.get('path'):
                fontkey = resolve_fontkey_via_api(unprint_bin, entry['path'])
            else:
                fontkey = entry.get('font_key', '')
            if not fontkey:
                raise RuntimeError(
                    f"Font entry for '{base}' variant '{variant}' has no font_key. Fix the catalog."
                )
            
            # Map both raw and clean names to the fontkey
            canonical_map[raw_bf] = fontkey
            canonical_map[clean_bf] = fontkey
        
        return canonical_map

    # Load catalog if provided (legacy path, may be incorrect for variable fonts)
    catalog_by_path = {}
    if catalog_json and os.path.exists(catalog_json):
        try:
            with open(catalog_json) as f:
                catalog = json.load(f)
            # Group by path, preferring the --resolve-fontkey preference
            # (GPOS-aware: kern +4 / nokern +4, nolig +2, opsz=10 +1).
            # Must match Rust scoring.
            for entry in catalog:
                path = entry.get('path')
                if not path:
                    continue
                existing = catalog_by_path.get(path)
                if existing is None:
                    catalog_by_path[path] = entry
                    continue
                gpos = _font_has_gpos(path)
                if _resolve_score(entry.get('font_key', ''), gpos) > _resolve_score(existing.get('font_key', ''), gpos):
                    catalog_by_path[path] = entry
        except Exception:
            pass

    ps_to_canonical = {}
    ps_to_orig_weight = {}
    
    # Build map from normalized catalog postscript_name -> font_key
    # For variable fonts, there are multiple entries per path (different weights).
    # We need to match PDF BaseFont names (e.g., "Playfair-Display-Bold") to
    # catalog entries (e.g., "PlayfairDisplay-700|wght700[wght=700]").
    # Prefer the --resolve-fontkey preference (kern +4, nolig +2, opsz=10 +1)
    # when multiple entries match.
    # Build exact PS-name -> catalog entry map (architectural change #3: opaque font_key,
    # no normalization heuristics). The catalog's postscript_name is the clean
    # raw_postscript_name from the font file, which should exactly match the
    # PDF's BaseFont (after subset prefix stripping).
    ps_to_entry = {}
    if catalog_json and os.path.exists(catalog_json):
        try:
            with open(catalog_json) as f:
                catalog = json.load(f)
            for entry in catalog:
                ps_name = entry.get('postscript_name', '')
                if not ps_name:
                    continue
                # Exact match only — no normalization, no guessing.
                # If multiple entries share the same PS name, prefer the
                # --resolve-fontkey preference (GPOS-aware: kern +4 / nokern
                # +4, nolig +2, opsz=10 +1).
                existing = ps_to_entry.get(ps_name)
                if existing is None:
                    ps_to_entry[ps_name] = entry
                elif _resolve_score(entry.get('font_key', ''), _font_has_gpos(entry.get('path'))) > _resolve_score(existing.get('font_key', ''), _font_has_gpos(existing.get('path'))):
                    ps_to_entry[ps_name] = entry
        except Exception as e:
            # Catalog load failure is fatal — cannot annotate without it
            raise RuntimeError(f"Failed to load catalog {catalog_json}: {e}")
    
    # For each file, also add the file's PS name mapping (for non-variable fonts)
    for path in all_font_files:
        catalog_entry = catalog_by_path.get(str(path))
        if catalog_entry:
            # Use the catalog's font_key directly.
            # The font_key already includes |nokern, [wght=...], etc.
            # Get the raw PS name from the file for mapping (exact match only)
            from pdf_font_annotate import read_postscript_name
            raw_ps = read_postscript_name(str(path))
            if raw_ps:
                ps_to_canonical[raw_ps] = catalog_entry['font_key']
        else:
            # Path not in catalog - try fallback for deduped files (same content, different path).
            # Rust dedupes by content hash; fontconfig may return the deduped path.
            # Fallback: find catalog entry in same directory with same file size.
            fallback_entry = None
            try:
                path_str = str(path)
                path_dir = os.path.dirname(path_str)
                path_size = os.path.getsize(path_str)
                for cat_path_str, cat_entry in catalog_by_path.items():
                    if os.path.dirname(cat_path_str) == path_dir:
                        try:
                            if os.path.getsize(cat_path_str) == path_size:
                                fallback_entry = cat_entry
                                break
                        except OSError:
                            pass
            except OSError:
                pass

            if fallback_entry:
                # Use the deduped catalog entry
                catalog_entry = fallback_entry
                from pdf_font_annotate import read_postscript_name
                raw_ps = read_postscript_name(str(path))
                if raw_ps:
                    ps_to_canonical[raw_ps] = catalog_entry['font_key']
                    norm = _normalize_ps(raw_ps)
                    if norm not in ps_to_canonical:
                        pass
            else:
                # Catalog doesn't have this font - ERROR OUT (per Mike: "Error out instead of a silent skip")
                raise RuntimeError(
                    f"Font file not in catalog: {path}. "
                    f"Specimen uses a font not in the catalog. Fix the catalog or specimen."
                )
        # Still need weight for the PDF parsing logic below
        if path not in ps_to_orig_weight:
            try:
                tt = FTFont(str(path))
                # Get orig_ps for weight mapping
                from pdf_font_annotate import read_postscript_name
                rps = read_postscript_name(str(path))
                if rps:
                    ps_to_orig_weight[rps] = tt['OS/2'].usWeightClass
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
            # Use catalog ONLY - no reconstruction via --weight-explicit (per Mike's directive).
            # Try exact match, then normalized matching.
            if embedded_ps in ps_to_canonical:
                canonical_map[raw_bf] = ps_to_canonical[embedded_ps]
            else:
                # Try normalized matching against catalog entries
                # (handles Playfair-Display-Bold vs PlayfairDisplay-700, etc.)
                # Try embedded_ps first, then raw_bf (PDF BaseFont name)
                # Exact match only (architectural change #3: no normalization heuristics).
                # Try embedded PS name first, then raw BaseFont (both with subset prefix stripped).
                ps_entry = None
                for name_to_try in [embedded_ps, raw_bf]:
                    if not name_to_try:
                        continue
                    # Strip subset prefix (e.g., "ABCDEF+DejaVuSans" -> "DejaVuSans")
                    # The catalog's postscript_name is the clean raw name.
                    clean_name = name_to_try.split('+')[-1]
                    ps_entry = ps_to_entry.get(clean_name)
                    if ps_entry:
                        break
                if ps_entry:
                    canonical_map[raw_bf] = ps_entry['font_key']
                else:
                    # No catalog match - ERROR OUT (do not silently skip)
                    # Per Mike: "Error out instead of a silent skip"
                    raise RuntimeError(
                        f"No catalog entry for PDF font '{raw_bf}' (embedded: '{embedded_ps}'). "
                        f"Specimen uses a font not in the catalog. Fix the catalog or specimen."
                    )
    pdf.close()
    return canonical_map


