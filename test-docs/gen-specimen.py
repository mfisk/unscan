#!/usr/bin/env python3
"""
Generate "A Timeline of Typography" — a multi-page vector PDF specimen.

Output:
  font-timeline-specimen.pdf       — native vector PDF with embedded fonts + SVG logos
  font-timeline-specimen-scanned.pdf — rasterized with scan artifacts (skew, noise, blur)
  font-timeline-specimen-fontmap.json — font name → file path map for --include-fontmap

All text is real PDF text (not raster). SVG logos are placed as vector drawings.
The "scanned" version rasterizes entire pages then re-assembles as a raster PDF.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    FrameBreak, BaseDocTemplate, Frame, PageTemplate, KeepTogether,
    Flowable, Table, TableStyle, ImageAndFlowables,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
LOGO_DIR = SCRIPT_DIR / "logos"

PAGE_W, PAGE_H = letter  # 612 × 792 points (8.5 × 11 in)

# ---------------------------------------------------------------------------
# Font registration — fonts are resolved via fontconfig
# ---------------------------------------------------------------------------
def fc_find(family, style="Regular"):
    """Find a font file via fontconfig, validated against OS/2 metrics.

    Fontconfig's style matching is unreliable for two reasons:
    1. **4-font-family naming model**: SemiBold, Light, etc. declare subfamily
       "Regular" under an alternate family name. fc-list style=Regular matches them.
    2. **Large superfamilies**: Noto Serif has ~50 width/weight combos. fc-list
       returns Condensed/ExtraCondensed variants for "Noto Serif:style=Regular".

    We validate every candidate against OS/2.usWeightClass and OS/2.usWidthClass.
    Variable fonts whose OS/2 weight reflects the axis default (not the style we
    want) are deprioritized — static instances are preferred when available, since
    ReportLab doesn't support variable font instances anyway.

    Prefers .ttf over .otf (ReportLab needs TrueType outlines).
    """
    from fontTools.ttLib import TTFont as FTFont

    STYLE_WEIGHTS = {
        "Regular": 400,
        "Bold": 700,
        "Light": 300,
        "Medium": 500,
        "Italic": 400,
        "Bold Italic": 700,
    }
    expected_weight = STYLE_WEIGHTS.get(style)
    expected_width = 5  # usWidthClass: 5 = normal (not condensed/expanded)

    for query in [f"{family}:style={style}", family]:
        r = subprocess.run(
            ["fc-list", query, "--format=%{file}\n"],
            capture_output=True, text=True
        )
        candidates = [l for l in r.stdout.strip().split('\n') if l.lower().endswith(('.ttf', '.otf'))]
        # Prefer .ttf — ReportLab can't handle PostScript-outline .otf
        ttf = [c for c in candidates if c.lower().endswith('.ttf')]
        pool = ttf if ttf else candidates
        if not pool:
            continue

        def _check(path):
            """Return (weight_ok, width_ok, is_variable, weight, width)."""
            try:
                tt = FTFont(path)
                wt = tt['OS/2'].usWeightClass
                wd = tt['OS/2'].usWidthClass
                is_var = tt.get('fvar') is not None
                tt.close()
                return (wt, wd, is_var)
            except Exception:
                return None

        # Score and rank candidates
        scored = []
        for path in pool:
            info = _check(path)
            if info is None:
                continue
            wt, wd, is_var = info
            width_ok = (wd == expected_width)
            weight_ok = (wt == expected_weight) if expected_weight else True
            weight_close = (abs(wt - expected_weight) <= 50) if expected_weight else True
            # Priority: exact weight + normal width + static > variable
            # Condensed/expanded fonts are almost never what we want for specimen
            score = 0
            if width_ok:       score += 100
            if weight_ok:      score += 50
            elif weight_close: score += 25
            if not is_var:     score += 10  # prefer static over variable
            scored.append((score, path, wt, wd, is_var))

        if not scored:
            continue

        scored.sort(key=lambda x: -x[0])
        best_score, best_path, best_wt, best_wd, best_var = scored[0]

        # Accept if width is normal and weight is at least close
        if best_wd == expected_width:
            if not expected_weight or best_wt == expected_weight or abs(best_wt - expected_weight) <= 50:
                return best_path

        # Fallback: accept anything with normal width regardless of weight
        for s, path, wt, wd, is_var in scored:
            if wd == expected_width:
                return path

        # Last resort: return highest-scored candidate even if condensed
        return scored[0][1]

    return None
def register_font(rl_name, ttf_path):
    """Register a TTF with reportlab. Returns True on success."""
    if ttf_path and os.path.exists(ttf_path):
        try:
            pdfmetrics.registerFont(TTFont(rl_name, ttf_path))
            return True
        except Exception as e:
            print(f"  WARN: can't register {rl_name} from {ttf_path}: {e}")
    return False

# Font spec: (rl_base_name, regular_path, bold_path, italic_path)
# We'll register {base}, {base}-Bold, {base}-Italic

def register_all_fonts():
    """Register all specimen fonts with reportlab."""
    from reportlab.lib.fonts import addMapping

    # All fonts are resolved via fontconfig by family name.
    # This is the normal way fonts are installed on Linux — no hard-coded paths.
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
    }

    registered = {}
    font_file_map = {}

    for base, family in FAMILIES.items():
        reg = fc_find(family, "Regular") or fc_find(family)
        bold = fc_find(family, "Bold") or reg
        italic = fc_find(family, "Italic") or reg

        ok = register_font(base, reg)
        if ok:
            registered[base] = True
            if reg:
                font_file_map[base] = reg
        ok_b = register_font(f"{base}-Bold", bold)
        ok_i = register_font(f"{base}-Italic", italic)
        if not ok_b:
            register_font(f"{base}-Bold", reg)
        if not ok_i:
            register_font(f"{base}-Italic", reg)
        if ok_b and bold:
            font_file_map[f"{base}-Bold"] = bold
        if ok_i and italic:
            font_file_map[f"{base}-Italic"] = italic

        addMapping(base, 0, 0, base)
        addMapping(base, 1, 0, f"{base}-Bold")
        addMapping(base, 0, 1, f"{base}-Italic")
        addMapping(base, 1, 1, f"{base}-Bold")

    return registered, font_file_map




# ---------------------------------------------------------------------------
# Section data
# ---------------------------------------------------------------------------
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
# SVG logo flowable
# ---------------------------------------------------------------------------
class SVGLogo(Flowable):
    """Place an SVG as a vector drawing in the PDF."""
    def __init__(self, svg_path, max_width, max_height=50):
        Flowable.__init__(self)
        self.svg_path = svg_path
        self.max_width = max_width
        self.max_height = max_height
        self._drawing = None
        self._load()

    def _load(self):
        try:
            from svglib.svglib import svg2rlg
            drawing = svg2rlg(self.svg_path)
            if drawing is None:
                return
            # Scale to fit: scale up small SVGs to max_height, scale down large ones
            sx = self.max_width / drawing.width
            sy = self.max_height / drawing.height
            scale = min(sx, sy)
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
            self._drawing = drawing
            self.width = drawing.width
            self.height = drawing.height
        except Exception as e:
            print(f"  WARN: SVG load failed for {self.svg_path}: {e}")

    def wrap(self, availWidth, availHeight):
        if self._drawing:
            return self._drawing.width, self._drawing.height
        return 0, 0

    def draw(self):
        if self._drawing:
            self._drawing.drawOn(self.canv, 0, 0)


class StackedFlowables(Flowable):
    """Stack multiple flowables vertically into a single flowable.
    Used to combine headshot + logo into one image block for ImageAndFlowables."""
    def __init__(self, flowables, gap=3):
        Flowable.__init__(self)
        self._flowables = [f for f in flowables if f is not None]
        self._gap = gap
        self.drawWidth = 0
        self.drawHeight = 0

    def _restrictSize(self, aW, aH):
        """Mimic Image._restrictSize so ImageAndFlowables can use us."""
        if self.drawWidth > aW + 1e-6 or self.drawHeight > aH + 1e-6:
            self._oldDrawSize = self.drawWidth, self.drawHeight
            factor = min(float(aW) / max(self.drawWidth, 1e-6),
                         float(aH) / max(self.drawHeight, 1e-6))
            self.drawWidth *= factor
            self.drawHeight *= factor
        return self.drawWidth, self.drawHeight

    def _unRestrictSize(self):
        dwh = getattr(self, '_oldDrawSize', None)
        if dwh:
            self.drawWidth, self.drawHeight = dwh

    def wrap(self, availWidth, availHeight):
        self.drawWidth = 0
        self.drawHeight = 0
        for i, f in enumerate(self._flowables):
            w, h = f.wrap(availWidth, availHeight)
            self.drawWidth = max(self.drawWidth, w)
            self.drawHeight += h
            if i > 0:
                self.drawHeight += self._gap
        self.width = self.drawWidth
        self.height = self.drawHeight
        return self.drawWidth, self.drawHeight

    def draw(self):
        y = self.drawHeight
        for i, f in enumerate(self._flowables):
            w, h = f.wrap(self.drawWidth, self.drawHeight)
            y -= h
            f.drawOn(self.canv, 0, y)
            if i < len(self._flowables) - 1:
                y -= self._gap


# ---------------------------------------------------------------------------
# Build the specimen PDF
# ---------------------------------------------------------------------------
def build_specimen(out_pdf, registered):
    """Build a proper vector PDF with embedded fonts and SVG logos."""

    # Two-column layout
    margin_side = 0.6 * inch
    margin_top = 0.5 * inch
    margin_bottom = 0.5 * inch
    gutter = 0.3 * inch
    col_w = (PAGE_W - 2 * margin_side - gutter) / 2

    frame_left = Frame(
        margin_side, margin_bottom,
        col_w, PAGE_H - margin_top - margin_bottom,
        id='left', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
    )
    frame_right = Frame(
        margin_side + col_w + gutter, margin_bottom,
        col_w, PAGE_H - margin_top - margin_bottom,
        id='right', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
    )

    doc = BaseDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=margin_side,
        rightMargin=margin_side,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
    )
    doc.addPageTemplates([
        PageTemplate(id='TwoCol', frames=[frame_left, frame_right]),
    ])

    story = []

    # --- Title block ---
    title_style = ParagraphStyle(
        'Title', fontName='PlayfairDisplay', fontSize=22, leading=26,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', fontName='SourceSerif4-Italic',
        fontSize=10, leading=13, spaceAfter=2, textColor='#444444'
    )
    intro_style = ParagraphStyle(
        'Intro', fontName='SourceSerif4', fontSize=9, leading=12, spaceAfter=10
    )

    story.append(Paragraph("A Timeline of Typography", title_style))
    story.append(Paragraph(
        "<i>Five Centuries of Letterforms — A Font Specimen for Unscan</i>",
        subtitle_style
    ))
    story.append(Paragraph(
        "Each section is rendered in a known font. Every blurb is set in the font it describes. "
        "SVG logos mark companies that made or popularized each typeface.",
        intro_style
    ))
    story.append(Spacer(1, 8))

    # --- Sections ---
    # Image column width for headshot/logo sidebar
    IMG_COL_W = 52  # points — fits ~40pt image + padding

    for section in SECTIONS:
        rl = section["rl_font"]
        if rl not in registered:
            raise RuntimeError(
                f"Font '{rl}' (family '{section['font_family']}') not registered — "
                f"run scripts/install-all-fonts.sh to install missing fonts"
            )
        text_items = []  # left column: all text flowables
        img_items = []   # right column: headshot + logo stacked

        # --- Build right-column image stack ---

        # Headshot/portrait (aspect-ratio-preserving)
        headshot_rel = section.get("headshot")
        if headshot_rel:
            hs_path = SCRIPT_DIR / headshot_rel
            if hs_path.exists():
                try:
                    pil_img = PILImage.open(hs_path)
                    iw, ih = pil_img.size
                    target_w = 40  # points
                    max_h = 55     # points — cap very tall portraits
                    aspect = ih / iw
                    target_h = min(target_w * aspect, max_h)
                    # If capped by max_h, shrink width to preserve ratio
                    if target_w * aspect > max_h:
                        target_w = max_h / aspect
                    img_items.append(RLImage(str(hs_path), width=target_w, height=target_h))
                    img_items.append(Spacer(1, 3))
                except Exception as e:
                    print(f"  WARN: headshot load failed for {hs_path}: {e}")

        # SVG logo
        svg_rel = section.get("logo_svg")
        if svg_rel:
            svg_path = SCRIPT_DIR / svg_rel
            if svg_path.exists():
                logo = SVGLogo(str(svg_path), max_width=42, max_height=30)
                if logo._drawing:
                    img_items.append(logo)
                    img_items.append(Spacer(1, 3))

        has_images = len(img_items) > 0

        # --- Build left-column text flowables ---

        # Heading in the section's own font, bold
        hdr_style = ParagraphStyle(
            f'Hdr-{rl}', fontName=f'{rl}-Bold', fontSize=14, leading=17,
            spaceBefore=6, spaceAfter=2
        )
        text_items.append(Paragraph(section["era"], hdr_style))

        # Source line
        src_style = ParagraphStyle(
            f'Src-{rl}', fontName='SourceSerif4-Italic',
            fontSize=7, leading=9, textColor='#777777', spaceAfter=3
        )
        text_items.append(Paragraph(f"Font: {section['source']}", src_style))

        # Body blurb in the section's font
        text_align = TA_JUSTIFY if section.get("alignment") == "justify" else TA_LEFT
        body_style = ParagraphStyle(
            f'Body-{rl}', fontName=rl, fontSize=10, leading=13, spaceAfter=2,
            alignment=text_align
        )
        text_items.append(Paragraph(section["blurb"], body_style))

        # Bold sample
        bold_style = ParagraphStyle(
            f'Bold-{rl}', fontName=f'{rl}-Bold', fontSize=10, leading=13, spaceAfter=1,
            alignment=text_align
        )
        text_items.append(Paragraph(
            "Bold: The quick brown fox jumps over 1,234,567,890 lazy dogs.",
            bold_style
        ))

        # Italic sample
        italic_style = ParagraphStyle(
            f'Italic-{rl}', fontName=f'{rl}-Italic', fontSize=10, leading=13, spaceAfter=1,
            alignment=text_align
        )
        text_items.append(Paragraph(
            "Italic: The quick brown fox jumps over 1,234,567,890 lazy dogs.",
            italic_style
        ))

        # Alphabet + digits
        alpha_style = ParagraphStyle(
            f'Alpha-{rl}', fontName=rl, fontSize=9, leading=11, spaceAfter=1
        )
        text_items.append(Paragraph(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ  abcdefghijklmnopqrstuvwxyz",
            alpha_style
        ))
        text_items.append(Paragraph("Lining figures: 0 1 2 3 4 5 6 7 8 9", alpha_style))

        # --- Compose layout: ImageAndFlowables (text wraps around images) ---
        if has_images:
            # Stack headshot + logo into one composite image flowable
            # Filter out Spacer items from img_items — only keep actual images
            real_images = [f for f in img_items if not isinstance(f, Spacer)]
            if len(real_images) >= 1:
                side_img = StackedFlowables(real_images, gap=3)

            iaf = ImageAndFlowables(
                side_img, text_items,
                imageLeftPadding=4, imageRightPadding=0,
                imageTopPadding=0, imageBottomPadding=3,
                imageSide='right',
            )
            section_items = [iaf, Spacer(1, 8)]
        else:
            # No images — full-width text
            text_items.append(Spacer(1, 8))
            section_items = text_items

        # Try to keep section together
        story.append(KeepTogether(section_items))

    doc.build(story)
    return out_pdf


# ---------------------------------------------------------------------------
# Rasterization + fontmap — all logic lives in tools/rasterize.py.
# gen-specimen.py only builds the vector PDF; rasterize.py does the rest.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from importlib import import_module as _imp
_rasterize_mod = _imp("rasterize")


def main():
    print("Registering fonts...")
    registered, _font_file_map = register_all_fonts()
    print(f"  {len(registered)} font families registered")

    out_pdf = OUT_DIR / "font-timeline-specimen.pdf"
    print(f"Building vector specimen: {out_pdf}")
    build_specimen(out_pdf, registered)

    # Build fontmap by introspecting what's actually in the PDF
    print("Introspecting PDF for font map...")
    resolved, unresolved = _rasterize_mod.build_fontmap(str(out_pdf))
    out_fontmap = OUT_DIR / "font-timeline-specimen-fontmap.json"
    _rasterize_mod.write_fontmap(resolved, out_fontmap)
    print(f"  {len(resolved)} fonts → {out_fontmap}")
    if unresolved:
        print(f"  {len(unresolved)} unresolved (builtins): {', '.join(unresolved)}")

    # Rasterize
    scanned_pdf = OUT_DIR / "font-timeline-specimen-scanned.pdf"
    print("Creating scanned version...")
    _rasterize_mod.rasterize(out_pdf, scanned_pdf)
    print(f"  → {scanned_pdf}")

    print("Done!")


if __name__ == "__main__":
    main()
