#!/usr/bin/env python3
"""Shared PDF font annotation utilities.

Used by gen-specimen.py and test PDF generators to embed /UnprintCanonical
in font dictionaries, so ground-truth comparison uses canonical font names
instead of raw /BaseFont strings.
"""

import os
import subprocess


def read_postscript_name(ttf_path):
    """Read PostScript name (name ID 6) from a font file."""
    from fontTools.ttLib import TTFont as FTFont
    try:
        tt = FTFont(ttf_path)
        for rec in tt['name'].names:
            if rec.nameID == 6:
                try:
                    ps = rec.toUnicode()
                    if ps:
                        tt.close()
                        return ps
                except Exception:
                    pass
        tt.close()
    except Exception:
        pass
    return None


def make_weight_explicit(path):
    """Ensure the font's PS name always contains an explicit weight keyword.

    Delegates to `unprint --weight-explicit` so the logic lives in one place
    (Rust font_scan::make_weight_explicit).  The Rust catalog scanner applies
    the same function at index time, so GT and catalog names always agree.

    Returns (original_ps_name, canonical_ps_name, path).
    """
    from fontTools.ttLib import TTFont as FTFont
    if not path:
        return None, None, path

    orig_ps = read_postscript_name(path)
    if not orig_ps:
        return None, None, path

    try:
        tt = FTFont(path)
        weight = tt['OS/2'].usWeightClass
        tt.close()
    except Exception:
        return orig_ps, orig_ps, path

    # Try release binary first, then debug, then PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    for bin_path in [
        os.path.join(repo_root, "target", "release", "unprint"),
        os.path.join(repo_root, "target", "release", "unscan"),
        os.path.join(repo_root, "target", "debug", "unprint"),
        os.path.join(repo_root, "target", "debug", "unscan"),
    ]:
        if os.path.exists(bin_path):
            break
    else:
        bin_path = "unprint"

    try:
        r = subprocess.run(
            [bin_path, "--weight-explicit", f"{orig_ps}:{weight}"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            new_ps = r.stdout.strip()
        else:
            new_ps = orig_ps
    except (FileNotFoundError, OSError):
        new_ps = orig_ps

    return orig_ps, new_ps, path


def build_canonical_map_from_paths(font_paths):
    """Build a canonical_map from a list of font file paths.

    Returns dict mapping raw PS name → canonical (weight-explicit) PS name.
    """
    canonical_map = {}
    for path in font_paths:
        if not path or not os.path.exists(path):
            continue
        orig_ps, canonical_ps, _ = make_weight_explicit(path)
        if orig_ps and canonical_ps:
            canonical_map[orig_ps] = canonical_ps
    return canonical_map


def annotate_canonical_names(pdf_path, canonical_map):
    """Post-process a PDF: add /UnprintCanonical to every font dictionary.

    For each font dictionary in the PDF, reads /BaseFont (stripping subset
    prefix), looks it up in canonical_map, and writes /UnprintCanonical with
    the canonical (weight-explicit) name.

    Returns (annotated_count, missing_names_list).
    """
    import pikepdf

    pdf = pikepdf.open(pdf_path, allow_overwriting_input=True)
    annotated = 0
    missing = []

    for page in pdf.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for res_name in list(fonts.keys()):
            font_dict = fonts[res_name]
            if isinstance(font_dict, pikepdf.Object) and hasattr(font_dict, 'get'):
                pass
            else:
                continue
            base_font = font_dict.get("/BaseFont")
            if base_font is None:
                continue
            bf_str = str(base_font).lstrip("/")
            # Strip subset prefix (e.g. "AAAAAA+Lato-Italic" → "Lato-Italic")
            if len(bf_str) > 7 and bf_str[6] == '+':
                raw_ps = bf_str[7:]
            else:
                raw_ps = bf_str
            # For Type0 (CID) fonts, the top-level BaseFont may be a
            # human-readable name ("Libre Bodoni Regular") while the
            # CIDFont descendant carries the actual PS name
            # ("LibreBodoni-Regular").  Try the descendant first.
            canonical = canonical_map.get(raw_ps)
            if canonical is None:
                descendants = font_dict.get("/DescendantFonts")
                if descendants is not None:
                    for desc in descendants:
                        dbf = desc.get("/BaseFont")
                        if dbf is not None:
                            dbf_str = str(dbf).lstrip("/")
                            if len(dbf_str) > 7 and dbf_str[6] == '+':
                                dbf_str = dbf_str[7:]
                            canonical = canonical_map.get(dbf_str)
                            if canonical:
                                break
            if canonical:
                font_dict[pikepdf.Name("/UnprintCanonical")] = pikepdf.String(canonical)
                annotated += 1
            else:
                if raw_ps not in missing:
                    missing.append(raw_ps)

    pdf.save(pdf_path)
    pdf.close()
    return annotated, missing
