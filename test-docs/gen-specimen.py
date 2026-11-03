#!/usr/bin/env python3
"""gen-specimen.py — Generate font-timeline-specimen.pdf using WeasyPrint.

Outputs:
  font-timeline-specimen.pdf        — vector PDF with embedded fonts
  font-timeline-specimen-rasterized.pdf — rasterized version

Uses WeasyPrint (Pango/HarfBuzz backend) for proper OpenType kerning.
Each section is rendered in the font it describes.  The PDF is then annotated
with /UnprintCanonical entries mapping raw PostScript names to weight-explicit
canonical names, and a rasterized version is generated.

Refactored to share font resolution, HTML escaping, @font-face CSS, and PDF
canonical-map logic via gen_common.py (single source of truth).
"""

from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR

# Shared utilities — single source of truth for font finding / PDF pipeline
from gen_common import (
    FAMILIES,
    fc_find,
    escape_html as _escape,
    build_font_face_css as _font_face_css_from_entries,
    build_canonical_map_from_pdf,
    render_html_to_pdf,
)
from pdf_font_annotate import annotate_canonical_names


# ---------------------------------------------------------------------------
# Font families to include in the specimen — primary families only
# ---------------------------------------------------------------------------
# NOTE: Full family registry (including aliases like ArialMT -> Arial) lives
# in gen_common.FAMILIES. We filter to primary specimen families here.

def _is_primary_family(base):
    return base not in {"ArialMT", "Arial-BoldMT", "CourierNewPSMT", "TimesNewRomanPSMT",
                         "PrestigeEliteNormal", "NimbusSansL"}

def resolve_all_fonts(catalog_json=None):
    """Resolve all font families via the font catalog (not fontconfig).

    Per architectural change #2: the specimen must pick catalog entries directly,
    not rely on fontconfig resolution which can disagree with the Rust scan.

    Returns (font_entries, all_font_files):
      font_entries[base] = {'regular': entry, 'bold': entry, 'italic': entry}
        where entry is the catalog dict with 'path' and 'font_key'
      all_font_files: deduplicated list of all resolved TTF paths

    Raises RuntimeError if a family/style has no catalog entry (per Mike's
    "error out instead of silent skip" rule).
    """
    import json
    import re

    if not catalog_json:
        raise RuntimeError("resolve_all_fonts requires catalog_json (catalog-based resolution)")

    with open(catalog_json) as f:
        catalog = json.load(f)

    # Build lookup: normalized family prefix -> list of entries
    # e.g., "dejavusans" -> [DejaVuSans-400, DejaVuSans-700, ...]
    def _norm_family(name):
        # "DejaVu Sans" -> "dejavusans", "EB Garamond" -> "ebgaramond"
        return re.sub(r'[^a-z0-9]', '', name.lower())

    # Index catalog by normalized font_key prefix
    # font_key like "DejaVuSans-400" -> prefix "dejavusans"
    # font_key like "SourceSans3-200[wght=200]" -> prefix "sourcesans3"
    prefix_to_entries = {}
    for entry in catalog:
        fk = entry.get('font_key', '')
        # Extract family prefix: everything before "-<digit>" (weight)
        # "SourceSans3-200[wght=200]" -> "SourceSans3"
        m = re.match(r'^(.+?)-\d', fk)
        if m:
            prefix_raw = m.group(1)
        else:
            prefix_raw = re.split(r'[|\[]', fk)[0]
        prefix = _norm_family(prefix_raw)
        prefix_to_entries.setdefault(prefix, []).append(entry)

    def _find_entry(family, style, base=None):
        """Find catalog entry for family+style. Errors out if not found or ambiguous.
        
        Tries family name first, then base name (if provided), then suffix variations.
        """
        candidates = []
        # Try family name, then base name
        names_to_try = [family]
        if base and base != family:
            names_to_try.append(base)
        for name in names_to_try:
            norm = _norm_family(name)
            candidates = prefix_to_entries.get(norm, [])
            if candidates:
                if name != family:
                    print(f"  NOTE: {family} mapped via base name '{name}'")
                break
            # Try suffix variations
            for suffix in ["ps", "mt", "psmt", "normal"]:
                candidates = prefix_to_entries.get(norm + suffix, [])
                if candidates:
                    print(f"  NOTE: {family} mapped to '{norm + suffix}' variant")
                    break
            if candidates:
                break
        if not candidates:
            norm_fam = _norm_family(family)
            raise RuntimeError(
                f"No catalog entry for family '{family}' (normalized: '{norm_fam}'). "
                f"Specimen requires a catalog entry. Fix the catalog or specimen."
            )

        # Filter by style
        if style == "Regular":
            # Want weight 400, not italic, prefer non-variant (or nokern per specimen policy)
            matches = [e for e in candidates
                       if '400' in e.get('font_key', '')
                       and 'italic' not in e.get('font_key', '').lower()]
        elif style == "Bold":
            matches = [e for e in candidates
                       if '700' in e.get('font_key', '')
                       and 'italic' not in e.get('font_key', '').lower()]
            if not matches:
                # Fallback to nearby weights
                for w in ("800", "900", "600", "500"):
                    matches = [e for e in candidates
                               if w in e.get('font_key', '')
                               and 'italic' not in e.get('font_key', '').lower()]
                    if matches:
                        print(f"  NOTE: {family} has no Bold — using weight {w}")
                        break
        elif style == "Italic":
            matches = [e for e in candidates
                       if 'italic' in e.get('font_key', '').lower()]
        else:
            matches = []

        if not matches:
            raise RuntimeError(
                f"No catalog entry for '{family}' style '{style}'. "
                f"Found {len(candidates)} candidates for family, none match style."
            )

        # Prefer nokern variant (specimen renders with font-kerning: none)
        nokern = [e for e in matches if 'nokern' in e.get('font_key', '')]
        if nokern:
            matches = nokern

        # Prefer opsz=10 if available
        opsz10 = [e for e in matches if 'opsz=10' in e.get('font_key', '')]
        if opsz10:
            matches = opsz10

        # Return first (deterministic: sorted by font_key)
        matches.sort(key=lambda e: e.get('font_key', ''))
        return matches[0]

    font_entries = {}
    all_font_files = []
    seen = set()

    # Pre-check: collect all families missing from catalog before starting.
    # Error out with the complete list (not one-by-one whack-a-mole).
    missing_families = []
    for base, family in FAMILIES.items():
        if not _is_primary_family(base):
            continue
        # Try family name first, then base name (e.g., "Prestige Elite" -> "prestigeelite",
        # but catalog has "prestigeelitenormal" from base "PrestigeEliteNormal")
        candidates = []
        for name_to_try in [family, base]:
            norm = _norm_family(name_to_try)
            candidates = prefix_to_entries.get(norm, [])
            if candidates:
                break
            # Try suffix variations
            for suffix in ["ps", "mt", "psmt", "normal"]:
                candidates = prefix_to_entries.get(norm + suffix, [])
                if candidates:
                    break
            if candidates:
                break
        if not candidates:
            missing_families.append((base, family))
    if missing_families:
        msg = "Specimen requires catalog entries for the following families, but they are missing:\n"
        for base, family in missing_families:
            msg += f"  - {base}: '{family}'\n"
        msg += "Fix the catalog (install fonts) or remove them from FAMILIES in gen_common.py."
        raise RuntimeError(msg)

    for base, family in FAMILIES.items():
        if not _is_primary_family(base):
            continue
        entries = {}
        reg = _find_entry(family, "Regular", base=base)
        entries['regular'] = reg

        try:
            bold = _find_entry(family, "Bold", base=base)
        except RuntimeError:
            # WARNING: No Bold in catalog — falling back to Regular (400).
            # The HTML renders with font-weight:bold, but GT annotation is 400
            # (the font file used). For families without a true bold variant
            # (e.g. IBM Plex Serif, Prestige Elite).
            print(f"  WARNING: {family} has no Bold — falling back to 400")
            bold = reg
        entries['bold'] = bold

        try:
            italic = _find_entry(family, "Italic", base=base)
        except RuntimeError:
            # WARNING: No Italic in catalog — falling back to Regular.
            # The HTML renders with font-style:italic, but GT annotation is the
            # regular fontkey (the font file used). For families without a true
            # italic variant.
            print(f"  WARNING: {family} has no Italic — falling back to Regular")
            italic = reg
        entries['italic'] = italic

        font_entries[base] = entries
        for e in [reg, bold, italic]:
            p = e['path']
            if p not in seen:
                seen.add(p)
                all_font_files.append(p)

    return font_entries, all_font_files


def _font_face_css(font_paths):
    """Generate @font-face CSS rules from resolved font paths.

    font_paths[base][variant] is either a path string (legacy) or a catalog
    entry dict with 'path' and 'font_key' (catalog-based resolution).
    """
    entries = []
    for base, paths in font_paths.items():
        for variant, path_or_entry in paths.items():
            # Extract path from entry dict or use string directly
            if isinstance(path_or_entry, dict):
                path = path_or_entry.get('path', '')
            else:
                path = path_or_entry
            weight = '700' if variant == 'bold' else '400'
            style = 'italic' if variant == 'italic' else 'normal'
            entries.append((base, path, weight, style))
    return _font_face_css_from_entries(entries)


def _img_tag(rel_path, css_class, max_w_pt=40, max_h_pt=55):
    """Generate an <img> tag for a logo or headshot, if the file exists."""
    full_path = SCRIPT_DIR / rel_path
    if not full_path.exists():
        return ""
    return (
        f'<img src="{rel_path}" class="{css_class}" '
        f'style="max-width: {max_w_pt}pt; max-height: {max_h_pt}pt;">'
    )


SECTIONS = [
    {
        "era": "c. 1530 — The Garamond",
        "font_family": "EB Garamond",
        "rl_font": "EBGaramond",
        "alignment": "justify",
        "source": "fonts.google.com/specimen/EB+Garamond — OFL, Georg Mayr-Duffner",
        "blurb": (
            "Claude Garamond was a Parisian punchcutter who broke the mold — literally. "
            "Before him, printers carved type into wood or imported it from Italy. Garamond created "
            "the first commercially available metal typefaces, and their elegant proportions "
            "became the template that defined Roman letterforms for 500 years."
        ),
        "logo_svg": "logos/gutenberg-press.svg",  # printing press
        "headshot": "logos/headshots/garamond.jpg",
    },
    {
        "era": "1722 — The Caslon",
        "font_family": "Libre Caslon Text",
        "rl_font": "LibreCaslonText",
        "alignment": "justify",
        "source": "fonts.google.com/specimen/Libre+Caslon+Text — OFL, Impallari Type",
        "blurb": (
            "William Caslon's types were the workhorses of the English-speaking world for a century. "
            "The American Declaration of Independence was first printed in Caslon. So was the first "
            "edition of Robinson Crusoe. Printers' saying: \"When in doubt, use Caslon.\""
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/caslon.jpg",
    },
    {
        "era": "1757 — The Baskerville",
        "font_family": "Libre Baskerville",
        "rl_font": "LibreBaskerville",
        "alignment": "justify",
        "source": "fonts.google.com/specimen/Libre+Baskerville — OFL, Impallari Type",
        "blurb": (
            "John Baskerville was a Birmingham industrialist who became obsessed with printing. "
            "He invented new inks, designed smoother paper, and created typefaces with unprecedented "
            "contrast between thick and thin strokes. His contemporaries called his types "
            "\"too sharp for the eye\" — Benjamin Franklin disagreed."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/baskerville.jpg",
    },
    {
        "era": "1798 — The Bodoni",
        "font_family": "Libre Bodoni",
        "rl_font": "LibreBodoni",
        "alignment": "justify",
        "source": "fonts.google.com/specimen/Libre+Bodoni — OFL, Impallari Type",
        "blurb": (
            "Giambattista Bodoni pushed the contrast dial to eleven. His types feature razor-thin "
            "hairlines and dramatic thick verticals — a style later called \"Modern.\" Bodoni's "
            "Manuale Tipografico contained 142 typefaces and remains one of the most beautiful "
            "type specimens ever printed. Vogue magazine still uses a Bodoni variant for its masthead."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/bodoni.jpg",
    },
    {
        "era": "1845 — The Slab Serif",
        "font_family": "Zilla Slab",
        "rl_font": "ZillaSlab",
        "alignment": "justify",
        "source": "fonts.google.com/specimen/Zilla+Slab — OFL, Typotheque for Mozilla",
        "blurb": (
            "The Industrial Revolution needed loud type. Slab serifs — with their blunt, "
            "unbracketed serifs of equal weight — were designed to scream from handbills, posters, "
            "and newspaper headlines. Mozilla's Zilla Slab carries the tradition forward."
        ),
        "logo_svg": None,  # was mozilla.svg — dropped (tiny text glyph, not a graphic mark)
        "headshot": None,  # Slab Serif is a style, not an individual designer
    },
    {
        "era": "c. 1895 — The Manual Typewriter",
        "font_family": "Special Elite",
        "rl_font": "SpecialElite",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Special+Elite — OFL, Astigmatic (manual typewriter recreation)",
        "blurb": (
            "Before electricity, every keystroke was muscle. The Remington No. 2 (1878) introduced "
            "the QWERTY layout and the shift key. Underwood No. 5 (1900) made the typebar visible. "
            "By 1900, thousands of typists hammered out correspondence on machines that cost a year's wages. "
            "The uneven impression, ink buildup, and worn ribbons were not defects — they were the texture "
            "of written communication for 80 years."
        ),
        "logo_svg": None,  # typewriter machine illustration to be added
        "headshot": None,
    },
    {
        "era": "1927 — Futura",
        "font_family": "Jost",
        "rl_font": "Jost",
        "alignment": "justify",
        "source": "fonts.google.com/specimen/Jost — OFL, Owen Earl (Futura stand-in)",
        "blurb": (
            "Paul Renner designed Futura in 1927 for the Bauer foundry. Its near-perfect geometric "
            "circles and triangles were radical. Stanley Kubrick used it in 2001. Wes Anderson uses "
            "it in everything. And in 1969, it went to the moon on the Apollo 11 plaque. "
            "Jost (by Owen Earl) is a faithful open-source stand-in."
        ),
        "logo_svg": None,  # was supreme.svg — dropped (just Futura text in a box)
        "headshot": "logos/headshots/renner.jpg",
    },
    {
        "era": "1931 — Times New Roman",
        "font_family": "Times New Roman",
        "rl_font": "TimesNewRoman",
        "alignment": "justify",
        "source": "Bundled with Windows/Office — Monotype",
        "blurb": (
            "In 1931, The Times of London commissioned Stanley Morison to redesign their newspaper "
            "type. Working with draftsman Victor Lardent, Morison created Times New Roman — optimized "
            "for narrow columns and cheap newsprint. Seven decades later, Microsoft bundled it with "
            "Windows, and every college student's essay defaulted to it."
        ),
        "logo_svg": None,  # was nytimes.svg — dropped (text-only wordmark)
        "headshot": "logos/headshots/morison.jpg",
    },
    {
        "era": "1953 — Prestige Elite",
        "font_family": "Prestige Elite",
        "rl_font": "PrestigeElite",
        "alignment": "ragged",
        "source": "Originally for IBM Executive typewriters — 12 characters per inch",
        "blurb": (
            "Prestige Elite was the other standard typewriter typeface — the smaller, tighter sibling "
            "to Courier's 10 characters per inch. At 12 cpi, Prestige Elite fit more words per line, "
            "making it the default for business correspondence, invoices, and government forms. "
            "If Courier was the Hollywood typewriter, Prestige Elite was the office workhorse."
        ),
        "logo_svg": None,
        "headshot": None,
    },
    {
        "era": "1955 — Courier (IBM)",
        "font_family": "Courier New",
        "rl_font": "CourierNew",
        "alignment": "ragged",
        "source": "IBM origin — Bundled with Windows/Office",
        "blurb": (
            "Howard \"Bud\" Kettler designed Courier in 1955 for IBM's Selectric typewriters. "
            "IBM deliberately chose not to trademark it, making it freely available. Every monospaced "
            "terminal, screenplay, and government form owes something to Courier. Its fixed-width "
            "design means every character occupies exactly the same horizontal space."
        ),
        "logo_svg": "logos/ibm.svg",
        "headshot": None,  # Kettler — no public portrait exists
    },
    {
        "era": "1957 — Helvetica / Nimbus Sans",
        "font_family": "Nimbus Sans",
        "rl_font": "NimbusSans",
        "alignment": "justify",
        "source": "URW++ Helvetica clone — fonts.urwpp.de",
        "blurb": (
            "Max Miedinger and Eduard Hoffmann designed Neue Haas Grotesk in 1957, renaming it "
            "Helvetica in 1960. It became the face of corporate modernism — used by American Airlines, "
            "Jeep, Toyota, and the NYC subway. Nimbus Sans is URW's Helvetica-compatible libre clone."
        ),
        "logo_svg": None,  # was americanairlines.svg — dropped (text-only wordmark)
        "headshot": "logos/headshots/miedinger.jpg",
    },
    {
        "era": "1982 — Arial (Microsoft)",
        "font_family": "Arial",
        "rl_font": "Arial",
        "alignment": "justify",
        "source": "Bundled with Windows/Office — Monotype",
        "blurb": (
            "Arial was Monotype's strategic masterstroke: a Helvetica substitute with matching metrics "
            "but just enough differences to avoid licensing. Microsoft bundled it with Windows 3.1 "
            "in 1992 and it conquered the world. Three billion people have used it."
        ),
        "logo_svg": "logos/microsoft.svg",
        "headshot": None,  # Monotype team effort
    },
    {
        "era": "1993 — Georgia (Microsoft)",
        "font_family": "Georgia",
        "rl_font": "Georgia",
        "alignment": "ragged",
        "source": "Bundled with Windows/Office — Matthew Carter",
        "blurb": (
            "Matthew Carter created Georgia in 1993 specifically for screen readability. Named after "
            "a tabloid headline ('Alien Heads Found in Georgia'), it was one of the first fonts "
            "designed from the pixel grid up rather than adapted from print. Carter also designed "
            "Verdana, Bell Centennial, and Miller."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/carter.jpg",
    },
    {
        "era": "1996 — Verdana (Microsoft)",
        "font_family": "Verdana",
        "rl_font": "Verdana",
        "alignment": "ragged",
        "source": "Bundled with Windows/Office — Matthew Carter",
        "blurb": (
            "Verdana is Georgia's sans-serif sibling, also by Matthew Carter, also designed for "
            "screens. The wide letterforms and tall x-height made it supremely legible on 640×480 "
            "monitors. At 10px on a CRT, Verdana was more readable than anything else alive."
        ),
        "logo_svg": None,  # was ikea.svg — dropped (text-based wordmark)
        "headshot": "logos/headshots/carter.jpg",
    },
    {
        "era": "1994 — Comic Sans (Microsoft)",
        "font_family": "Comic Sans MS",
        "rl_font": "ComicSansMS",
        "alignment": "ragged",
        "source": "Bundled with Windows/Office — Vincent Connare",
        "blurb": (
            "Vincent Connare designed Comic Sans in 1994 after seeing Times New Roman in a Microsoft "
            "Bob speech bubble. He based it on The Dark Knight Returns and Watchmen lettering. "
            "It was never intended for body text — but users put it everywhere: office memos, "
            "funeral programs, CERN's Higgs boson announcement. Typographers weep."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/connare.jpg",
    },
    {
        "era": "1996 — Trebuchet MS",
        "font_family": "Trebuchet MS",
        "rl_font": "TrebuchetMS",
        "alignment": "ragged",
        "source": "Bundled with Windows/Office — Vincent Connare",
        "blurb": (
            "Vincent Connare also designed Trebuchet MS. Named after a medieval siege engine because "
            "'it launches words across the internet,' it occupies a peculiar middle ground: more "
            "personality than Arial, less chaos than Comic Sans."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/connare.jpg",
    },
    {
        "era": "2004 — The ClearType Collection",
        "font_family": "Caladea",
        "rl_font": "Caladea",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Caladea — OFL, Carolina Giovagnoli (Cambria-compatible)",
        "blurb": (
            "When Microsoft developed ClearType subpixel rendering for LCD screens, they commissioned "
            "six new C-named font families: Calibri, Cambria, Candara, Consolas, Constantia, and Corbel. "
            "Calibri replaced Times New Roman as the Office default in 2007. "
            "Caladea is an open-source metric-compatible Cambria substitute."
        ),
        "logo_svg": "logos/microsoft.svg",
        "headshot": None,  # ClearType was a Microsoft team project
    },
    {
        "era": "2010 — Roboto (Google)",
        "font_family": "Roboto",
        "rl_font": "Roboto",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Roboto — Apache 2.0, Christian Robertson for Google",
        "blurb": (
            "When Google launched Google Fonts in 2010, it broke the foundry cartel overnight. "
            "Christian Robertson's Roboto became Android's system font and the most popular Google Font "
            "by a cosmic margin — it's on 27 million websites. Google's Material Design is built on "
            "Roboto's proportions."
        ),
        "logo_svg": "logos/google.svg",  # Google G multicolor icon (genuine graphic mark)
        "headshot": "logos/headshots/robertson.jpg",
    },
    {
        "era": "2010 — Open Sans (Google)",
        "font_family": "Open Sans",
        "rl_font": "OpenSans",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Open+Sans — OFL, Steve Matteson for Google",
        "blurb": (
            "Steve Matteson designed Open Sans in 2011, commissioned by Google. Its open apertures "
            "and neutral forms make it the typographic equivalent of clean water. WordPress.com and "
            "countless government sites use it. The second most popular Google Font."
        ),
        "logo_svg": "logos/wordpress.svg",
        "headshot": "logos/headshots/matteson.jpg",
    },
    {
        "era": "2010 — Lato",
        "font_family": "Lato",
        "rl_font": "Lato",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Lato — OFL, Łukasz Dziedzic",
        "blurb": (
            "Łukasz Dziedzic designed Lato ('Summer' in Polish) in 2010, originally as a corporate "
            "typeface for a client who went in a different direction. Their loss. Lato's semi-rounded "
            "details give it warmth without sacrificing seriousness. Third most popular Google Font."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/dziedzic.jpg",
    },
    {
        "era": "2011 — Merriweather",
        "font_family": "Merriweather",
        "rl_font": "Merriweather",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Merriweather — OFL, Eben Sorkin",
        "blurb": (
            "Eben Sorkin designed Merriweather for comfortable reading on screens — large x-height, "
            "slightly condensed letterforms, sturdy serifs that don't crumble at small sizes. Named "
            "after a 19th-century Kansas newspaper editor. The go-to serif for long blog posts."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/sorkin.jpg",
    },
    {
        "era": "2012 — Source Sans Pro (Adobe)",
        "font_family": "Source Sans 3",
        "rl_font": "SourceSans3",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Source+Sans+3 — OFL, Paul Hunt for Adobe",
        "blurb": (
            "Paul Hunt designed Source Sans Pro as Adobe's first open-source typeface, released in "
            "2012. It was a signal: even Adobe saw the value of open fonts. Source Sans is a clean "
            "humanist sans. Its companion, Source Code Pro, is beloved by programmers."
        ),
        "logo_svg": "logos/adobe.svg",
        "headshot": "logos/headshots/hunt.jpg",
    },
    {
        "era": "2014 — Source Serif 4 (Adobe)",
        "font_family": "Source Serif 4",
        "rl_font": "SourceSerif4",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Source+Serif+4 — OFL, Frank Grießhammer for Adobe",
        "blurb": (
            "Frank Grießhammer's Source Serif bridges the gap between old-style and transitional "
            "designs. UC Berkeley uses it as their official serif typeface. Source Serif 4 ships "
            "with extensive OpenType features — old-style figures, small caps, and stylistic sets."
        ),
        "logo_svg": "logos/adobe.svg",
        "headshot": None,  # Grießhammer — no public portrait found
    },
    {
        "era": "2014 — Noto Serif (Google + Monotype)",
        "font_family": "Noto Serif",
        "rl_font": "NotoSerif",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Noto+Serif — OFL, Google + Monotype",
        "blurb": (
            "The Noto project is Google's audacious attempt to create fonts for every writing system "
            "in Unicode — 'No Tofu' (the nickname for □ missing-glyph boxes). It covers 146 scripts "
            "and 800+ languages. The most democratic type project ever attempted."
        ),
        "logo_svg": None,
        "headshot": None,  # Noto is a massive team project
    },
    {
        "era": "2009 — PT Serif (ParaType)",
        "font_family": "PT Serif",
        "rl_font": "PTSerif",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/PT+Serif — OFL, Alexandra Korolkova at ParaType",
        "blurb": (
            "PT Serif was commissioned by the Russian government to create a public font family "
            "covering all the languages of the Russian Federation. Designed by Alexandra Korolkova "
            "at ParaType. State-funded type for the public good."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/korolkova.jpg",
    },
    {
        "era": "2011 — Playfair Display",
        "font_family": "Playfair Display",
        "rl_font": "PlayfairDisplay",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Playfair+Display — OFL, Claus Eggers Sørensen",
        "blurb": (
            "Claus Eggers Sørensen designed Playfair Display as a nod to the high-contrast types "
            "of the Enlightenment — Baskerville, Bodoni, and the Didots. Its delicate hairlines "
            "make it a display face par excellence: magazine headings, wedding invitations. "
            "Not great below 14px, but for titles at 36pt+, Playfair is magnificent."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/sorensen.jpg",
    },
    {
        "era": "2017 — IBM Plex Sans",
        "font_family": "IBM Plex Sans",
        "rl_font": "IBMPlexSans",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/IBM+Plex+Sans — OFL, Mike Abbink + Bold Monday for IBM",
        "blurb": (
            "When IBM replaced Helvetica Neue with IBM Plex in 2017, it was a $100B company saying "
            "'we need our own voice.' Plex threads a needle between Helvetica's neutrality and "
            "Futura's geometry. Corporate type that actually cares about craft."
        ),
        "logo_svg": "logos/ibm.svg",
        "headshot": "logos/headshots/abbink.jpg",
    },
    {
        "era": "2017 — IBM Plex Serif",
        "font_family": "IBM Plex Serif",
        "rl_font": "IBMPlexSerif",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/IBM+Plex+Serif — OFL, Mike Abbink + Bold Monday for IBM",
        "blurb": (
            "IBM Plex Serif is the quieter sibling — a contemporary slab-influenced serif that "
            "pairs naturally with Plex Sans. Together, the Plex family demonstrates that corporate "
            "identity fonts don't have to be boring."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/abbink.jpg",
    },
    {
        "era": "2017 — IBM Plex Mono",
        "font_family": "IBM Plex Mono",
        "rl_font": "IBMPlexMono",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/IBM+Plex+Mono — OFL, Mike Abbink + Bold Monday for IBM",
        "blurb": (
            "IBM Plex Mono completes the Plex trilogy with a monospaced design that nods to IBM's "
            "original Selectric typefaces. Its distinctive glyph shapes make zero/O and one/l/I "
            "instantly distinguishable — the single most important trait in a programming font."
        ),
        "logo_svg": None,
        "headshot": "logos/headshots/abbink.jpg",
    },
    {
        "era": "2018 — Inter (Rasmus Andersson)",
        "font_family": "Inter",
        "rl_font": "Inter",
        "alignment": "ragged",
        "source": "fonts.google.com/specimen/Inter — OFL, Rasmus Andersson. Also rsms.me/inter",
        "blurb": (
            "Rasmus Andersson, a designer at Figma, created Inter for user interfaces — specifically, "
            "for the awkward sizes between 11px and 16px. Its tall x-height and extensive kerning "
            "pairs make it the default for design tools and dashboards. It's what Helvetica would be "
            "if Helvetica were designed for Retina screens."
        ),
        "logo_svg": "logos/figma.svg",
        "headshot": "logos/headshots/andersson.jpg",
    },
]

# ---------------------------------------------------------------------------
# HTML + CSS generation
# ---------------------------------------------------------------------------

def generate_html(font_paths):
    """Generate the full HTML document for the specimen."""

    font_face_css = _font_face_css(font_paths)

    # SourceSerif4 for source lines — resolve its italic path
    # font_paths values are entry dicts (catalog-based) or path strings (legacy)
    def _get_path(base, variant):
        v = font_paths.get(base, {}).get(variant, "")
        if isinstance(v, dict):
            return v.get('path', '')
        return v
    ss4_it = _get_path("SourceSerif4", "italic")
    # PlayfairDisplay for title
    pd_reg = _get_path("PlayfairDisplay", "regular")

    sections_html = []
    for section in SECTIONS:
        rl = section["rl_font"]
        if rl not in font_paths:
            raise RuntimeError(
                f"Font '{rl}' (family '{section['font_family']}') not resolved — "
                f"run tools/install-all-fonts.sh to install missing fonts"
            )

        # Get kerning value from catalog entry (architectural: use the kerning
        # value returned by font catalog matching, which prefers nokern when available)
        # If entry is not a dict, ERROR OUT (per Mike: no silent fallback, be consistent).
        def _kerning_css(variant):
            entry = font_paths[rl].get(variant, {})
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Font entry for '{rl}' variant '{variant}' is not a dict: {type(entry)}. "
                    f"Cannot determine CSS from fontkey. Fix the catalog or specimen."
                )
            fk = entry.get('font_key', '')
            if 'nokern' in fk:
                return 'font-kerning: none;'
            return 'font-kerning: normal;'
        
        # Get font-weight CSS from catalog entry — MUST match the fontkey exactly.
        # Planned font -> fontkey -> render. Never hardcode.
        # If entry is not a dict, ERROR OUT (per Mike: no silent fallback).
        def _weight_css(variant):
            entry = font_paths[rl].get(variant, {})
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Font entry for '{rl}' variant '{variant}' is not a dict: {type(entry)}. "
                    f"Cannot determine CSS from fontkey. Fix the catalog or specimen."
                )
            fk = entry.get('font_key', '')
            import re
            # Check for weight in fontkey. Use precise patterns to avoid substring false positives.
            # Format examples:
            #   SourceSerif4-700|wght700[wght=700,opsz=20]
            #   SourceSerif4-400|nokern[wght=400,opsz=10]
            #   PrestigeEliteNormal-400 (no suffix)
            # Look for |wghtNNN or -NNN followed by | or [ or end-of-string
            if re.search(r'\|wght700\b', fk) or re.search(r'-700([\|\[]|$)', fk):
                return 'font-weight: bold;'
            elif re.search(r'\|wght400\b', fk) or re.search(r'-400([\|\[]|$)', fk):
                return 'font-weight: 400;'
            # For other weights, extract numeric value
            m = re.search(r'\|wght(\d+)\b', fk)
            if m:
                return f"font-weight: {m.group(1)};"
            # Try -NNN pattern at end or before |/[
            m = re.search(r'-(\d+)([\|\[]|$)', fk)
            if m:
                weight = m.group(1)
                # Map common weights to CSS
                if weight == '700':
                    return 'font-weight: bold;'
                elif weight == '400':
                    return 'font-weight: 400;'
                else:
                    return f'font-weight: {weight};'
            # Else: cannot determine — error out (no silent fallback)
            raise RuntimeError(
                f"Cannot determine font-weight for '{rl}' variant '{variant}': "
                f"fontkey '{fk}' has no recognizable weight info. Fix the catalog."
            )
        
        # Get font-style CSS from catalog entry — MUST match the fontkey exactly.
        # Planned font -> fontkey -> render. Never hardcode.
        # If entry is not a dict, ERROR OUT (per Mike: no silent fallback).
        def _style_css(variant):
            entry = font_paths[rl].get(variant, {})
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Font entry for '{rl}' variant '{variant}' is not a dict: {type(entry)}. "
                    f"Cannot determine CSS from fontkey. Fix the catalog or specimen."
                )
            fk = entry.get('font_key', '').lower()
            if 'italic' in fk:
                return 'font-style: italic;'
            # If variant is 'italic' but fontkey has no 'italic', it's a fallback to Regular.
            # CSS must match the fontkey (not italic), not the planned variant.
            return 'font-style: normal;'
        
        # Get font-variation-settings CSS from catalog entry — MUST match the fontkey exactly.
        # Planned font -> fontkey -> render. Never hardcode.
        # Parses [wght=...,opsz=...] from fontkey and generates CSS.
        # Note: wght is handled via font-weight (WeasyPrint bug #1981), so we skip it here.
        # Only opsz (and other non-wght axes) go in font-variation-settings.
        def _variation_css(variant):
            entry = font_paths[rl].get(variant, {})
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Font entry for '{rl}' variant '{variant}' is not a dict: {type(entry)}. "
                    f"Cannot determine CSS from fontkey. Fix the catalog or specimen."
                )
            fk = entry.get('font_key', '')
            # Extract variations from [tag=value,...]
            import re
            m = re.search(r'\[([^\]]+)\]', fk)
            if not m:
                return ''
            variations = []
            for part in m.group(1).split(','):
                part = part.strip()
                if '=' not in part:
                    continue
                tag, value = part.split('=', 1)
                tag = tag.strip()
                value = value.strip()
                # Skip wght (handled via font-weight per WeasyPrint bug #1981)
                if tag == 'wght':
                    continue
                variations.append(f"'{tag}' {value}")
            if variations:
                return f"font-variation-settings: {', '.join(variations)};"
            return ''
        
        kern_regular = _kerning_css('regular')
        kern_bold = _kerning_css('bold')
        weight_bold = _weight_css('bold')
        style_italic = _style_css('italic')
        kern_italic = _kerning_css('italic')
        var_regular = _variation_css('regular')
        var_bold = _variation_css('bold')
        var_italic = _variation_css('italic')

        align = "justify" if section.get("alignment") == "justify" else "left"

        # Build image sidebar
        imgs = []
        headshot_rel = section.get("headshot")
        if headshot_rel:
            tag = _img_tag(headshot_rel, "headshot")
            if tag:
                imgs.append(tag)
        logo_rel = section.get("logo_svg")
        if logo_rel:
            tag = _img_tag(logo_rel, "logo", max_w_pt=42, max_h_pt=30)
            if tag:
                imgs.append(tag)

        img_html = ""
        if imgs:
            img_html = '<div class="sidebar">' + "".join(imgs) + "</div>"

        blurb = _escape(section["blurb"])

        sections_html.append(f"""
<div class="section">
  {img_html}
  <h2 style="font-family: '{rl}'; {weight_bold} {kern_bold} {var_bold}">{_escape(section["era"])}</h2>
  <p class="source">Font: {_escape(section["source"])}</p>
  <p class="blurb" style="font-family: '{rl}'; text-align: {align}; {kern_regular} {var_regular}">{blurb}</p>
  <p class="sample" style="font-family: '{rl}'; {weight_bold} text-align: {align}; {kern_bold} {var_bold}">Bold: The quick brown fox jumps over 1,234,567,890 lazy dogs.</p>
  <p class="sample" style="font-family: '{rl}'; {style_italic} text-align: {align}; {kern_italic} {var_italic}">Italic: The quick brown fox jumps over 1,234,567,890 lazy dogs.</p>
  <p class="alpha" style="font-family: '{rl}'; {kern_regular} {var_regular}">ABCDEFGHIJKLMNOPQRSTUVWXYZ</p>
  <p class="alpha" style="font-family: '{rl}'; {kern_regular} {var_regular}">abcdefghijklmnopqrstuvwxyz</p>
  <p class="alpha" style="font-family: '{rl}'; {kern_regular} {var_regular}">Lining figures: 0 1 2 3 4 5 6 7 8 9</p>
</div>""")

    all_sections = "\n".join(sections_html)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{font_face_css}

@page {{
  size: 8.5in 11in;
  margin: 0.5in 0.6in;
}}

body {{
  columns: 2;
  column-gap: 0.3in;
  font-size: 10pt;
  line-height: 1.3;
  orphans: 3;
  widows: 3;
}}

.title-block {{
  column-span: all;
  margin-bottom: 8pt;
}}

.title-block h1 {{
  font-family: 'PlayfairDisplay';
  font-size: 22pt;
  line-height: 1.18;
  margin: 0 0 4pt 0;
}}

.title-block .subtitle {{
  font-family: 'SourceSerif4';
  font-style: italic;
  font-size: 10pt;
  line-height: 1.3;
  color: #444444;
  margin: 0 0 2pt 0;
}}

.title-block .intro {{
  font-family: 'SourceSerif4';
  font-size: 9pt;
  line-height: 1.33;
  margin: 0 0 10pt 0;
}}

.section {{
  break-inside: avoid;
  margin-bottom: 8pt;
}}

.section h2 {{
  font-size: 14pt;
  line-height: 1.21;
  margin: 6pt 0 2pt 0;
}}

.section .source {{
  font-family: 'SourceSerif4';
  font-style: italic;
  font-size: 7pt;
  line-height: 1.29;
  color: #777777;
  margin: 0 0 3pt 0;
}}

.section .blurb {{
  font-size: 10pt;
  line-height: 1.3;
  margin: 0 0 2pt 0;
}}

.section .sample {{
  font-size: 10pt;
  line-height: 1.3;
  margin: 0 0 1pt 0;
}}

.section .alpha {{
  font-size: 9pt;
  line-height: 1.22;
  margin: 0 0 1pt 0;
}}

.sidebar {{
  float: right;
  margin: 0 0 3pt 4pt;
  text-align: center;
}}

.sidebar .headshot {{
  display: block;
  margin-bottom: 3pt;
}}

.sidebar .logo {{
  display: block;
}}
</style>
</head>
<body>

<div class="title-block">
  <h1>A Timeline of Typography</h1>
  <p class="subtitle"><i>Five Centuries of Letterforms — A Font Specimen for OCR Audit</i></p>
  <p class="intro">Each section is rendered in a known font. Every blurb is set in the font
  it describes. SVG logos mark companies that made or popularized each typeface.</p>
</div>

{all_sections}

</body>
</html>"""

# ---------------------------------------------------------------------------
# PDF rendering + post-processing
# ---------------------------------------------------------------------------

def build_specimen(out_pdf, font_paths):
    """Render the specimen HTML to PDF with WeasyPrint (via shared helper)."""
    html = generate_html(font_paths)
    render_html_to_pdf(html, out_pdf, base_url=SCRIPT_DIR)
    return out_pdf


# Remaining original main / rasterize logic (uses shared canonical map builder)

def build_specimen(out_pdf, font_paths):
    """Render the specimen HTML to PDF with WeasyPrint."""
    import weasyprint
    html = generate_html(font_paths)
    doc = weasyprint.HTML(string=html, base_url=str(SCRIPT_DIR))
    doc.write_pdf(str(out_pdf))
    return out_pdf


# Rasterization — all logic lives in tools/rasterize.py.
from importlib import import_module as _imp
_rasterize_mod = _imp("rasterize")


def main():
    # Export font catalog FIRST (architectural change #2: specimen picks from catalog)
    import os
    import subprocess
    import json
    catalog_json = "/tmp/unprint-catalog.json"
    print("Exporting font catalog...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    unprint_bin = None
    for bp in [
        os.path.join(repo_root, "target", "release", "unprint"),
        os.path.join(repo_root, "target", "debug", "unprint"),
    ]:
        if os.path.exists(bp):
            unprint_bin = bp
            break
    if unprint_bin:
        try:
            result = subprocess.run(
                [unprint_bin, "--export-catalog-json", catalog_json, "-o", "/dev/null", "/dev/null"],
                capture_output=True, timeout=180)
            if result.returncode != 0:
                raise RuntimeError(f"Catalog export failed: {result.stderr.decode()[:500]}")
        except Exception as e:
            raise RuntimeError(f"Catalog export failed: {e}. Cannot build specimen without catalog.")
    else:
        raise RuntimeError("Unprint binary not found. Cannot build specimen without catalog.")

    print("Resolving fonts from catalog...")
    font_entries, all_font_files = resolve_all_fonts(catalog_json)
    print(f"  {len(font_entries)} font families resolved")
    print(f"  {len(all_font_files)} unique font files")

    out_pdf = OUT_DIR / "font-timeline-specimen.pdf"
    print(f"Building vector specimen: {out_pdf}")
    build_specimen(out_pdf, font_entries)

    # Build canonical map from actual PDF BaseFont names
    # Uses font_entries directly (the exact entries used during generation)
    # This ensures the annotation matches the render for variable fonts.
    print("Building canonical map from PDF font names...")
    canonical_map = build_canonical_map_from_pdf(
        str(out_pdf), all_font_files, catalog_json, font_entries, unprint_bin
    )
    print(f"  {len(canonical_map)} mappings built")

    # Annotate font dictionaries with /UnprintCanonical
    print("Annotating PDF with canonical font names...")
    annotated, missing = annotate_canonical_names(str(out_pdf), canonical_map)
    print(f"  {annotated} font dicts annotated")
    if missing:
        print(f"  WARNING: {len(missing)} BaseFont names not in canonical_map:")
        for m in missing:
            print(f"    {m}")

    # Rasterize
    rasterized_pdf = OUT_DIR / "font-timeline-specimen-rasterized.pdf"
    print("Creating rasterized version...")
    _rasterize_mod.rasterize(out_pdf, rasterized_pdf)
    print(f"  → {rasterized_pdf}")

    print("Done!")


if __name__ == "__main__":
    main()
