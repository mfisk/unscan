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

    Appends the full explicit variation form in brackets (e.g.
    `[wght=400,opsz=10]`) to match the catalog's explicit variant names.
    The opsz value uses the catalog's explicit optical-size choice (10)
    for fonts with an opsz axis, matching the CSS `font-variation-settings`
    emitted by the specimen generators.

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
        # Collect variation axes for the bracketed suffix.
        # Per Mike Nov 3: font keys must be as explicit as possible —
        # include ALL fvar axes with their values, not just wght/opsz.
        # This matches the Rust catalog (src/font_scan.rs::detect_default_variations)
        # which pins every axis to its default.
        var_axes = {}  # tag -> default value
        if 'fvar' in tt:
            for ax in tt['fvar'].axes:
                var_axes[ax.axisTag] = ax.defaultValue
        # Check for kern feature in GPOS (matches Rust font_has_kern_feature
        # in font_scan.rs, which only checks GPOS, not the old 'kern' table).
        # Only fonts with GPOS kern get a nokern variant in the catalog.
        has_kern = False
        if 'GPOS' in tt:
            try:
                gpos = tt['GPOS'].table
                # Walk FeatureList for 'kern' tag
                feature_list = gpos.FeatureList
                for rec in feature_list.FeatureRecord:
                    if rec.FeatureTag == 'kern':
                        has_kern = True
                        break
            except Exception:
                pass
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

    # Per Mike Nov 3: Python must use verbatim strings from the catalog,
    # not construct fontkeys. The bracketed variation suffix (e.g.,
    # [wght=400,opsz=10,wdth=100]) and the |nokern tag must come from the
    # catalog's font_key verbatim. This function only returns the
    # weight-explicit base name from Rust; the caller is responsible for
    # looking up the full verbatim font_key from the catalog.
    # DO NOT construct the bracket suffix here.
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
