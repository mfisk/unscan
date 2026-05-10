#!/usr/bin/env python3
"""
Generate "A Timeline of Typography" — a multi-page font specimen PDF.

This is the ground-truth document for unscan's font detection tests.
Each section is rendered IN the font it describes, with documented OpenType
feature variants. A machine-readable JSON sidecar maps every section to its
expected font family, source URL, and demonstrated OT features.

The script also produces a "scanned" variant with realistic degradation:
slight rotation (skew), Gaussian blur, speckle noise, and off-white paper.

Output:
    font-timeline-specimen.pdf          — clean vector PDF
    font-timeline-specimen-scanned.pdf  — simulated scan
    font-timeline-specimen.json         — ground truth

Prerequisites:
    apt install pango1.0-tools poppler-utils
    pip install Pillow numpy img2pdf

Fonts:
    On first run, download fonts from the Google Fonts CDN to
    /usr/share/fonts/truetype/specimen-fonts/ and run fc-cache -fv.
    The script doesn't auto-download; see test-docs/README.md for URLs.
    Microsoft Core Fonts: apt install ttf-mscorefonts-installer

Usage:
    python3 gen-specimen.py
"""
import subprocess, os, json, shutil, math, tempfile
from pathlib import Path

OUT_DIR = Path("/tmp/specimen-pages")
OUT_DIR.mkdir(exist_ok=True)

# Page dimensions at 300dpi (US Letter)
PAGE_W = 2550
PAGE_H = 3300

# ─── Font map: fc font family name → features available ────────────────────
# For pango, we use the fc family name directly.

SECTIONS = [
    # Each section: (era_label, font_family, pango_font_spec, blurb, features_dict)
    # features_dict: {"onum": True, "smcp": True, ...} for variants to demo
    {
        "era": "c. 1530 — The Garamond",
        "font_family": "EB Garamond",
        "pango_font": "EB Garamond 12",  # system OTF version has features
        "source": "fonts.google.com/specimen/EB+Garamond — OFL, Georg Mayr-Duffner",
        "blurb": (
            "Claude Garamond was a Parisian punchcutter who broke the mold — literally. "
            "Before him, printers carved type into wood or imported it from Italy. Garamond created "
            "the first commercially available metal typefaces, and their elegant proportions "
            "became the template that defined Roman letterforms for 500 years. "
            "This digital revival by Georg Mayr-Duffner faithfully recreates the warmth of the originals."
        ),
        "features": {"onum": True, "smcp": True, "swsh": False, "ss01": True},
    },
    {
        "era": "1722 — The Caslon",
        "font_family": "Libre Caslon Text",
        "pango_font": "Libre Caslon Text",
        "source": "fonts.google.com/specimen/Libre+Caslon+Text — OFL, Impallari Type",
        "blurb": (
            "William Caslon's types were the workhorses of the English-speaking world for a century. "
            "The American Declaration of Independence was first printed in Caslon. So was the first "
            "edition of Robinson Crusoe. His punches have a distinctly English warmth — slightly "
            "irregular, with a readable charm that more \"refined\" typefaces never quite matched. "
            "Printers' saying: \"When in doubt, use Caslon.\""
        ),
        "features": {},
    },
    {
        "era": "1757 — The Baskerville",
        "font_family": "Libre Baskerville",
        "pango_font": "Libre Baskerville",
        "source": "fonts.google.com/specimen/Libre+Baskerville — OFL, Impallari Type",
        "blurb": (
            "John Baskerville was a Birmingham industrialist who became obsessed with printing. "
            "He invented new inks, designed smoother paper, and created typefaces with unprecedented "
            "contrast between thick and thin strokes. His contemporaries called his types \"too sharp "
            "for the eye\" — Benjamin Franklin disagreed, and their correspondence about typography "
            "remains one of the great letters about letters. The 18th-century eye wasn't ready."
        ),
        "features": {},
    },
    {
        "era": "1798 — The Bodoni",
        "font_family": "Libre Bodoni",
        "pango_font": "Libre Bodoni",
        "source": "fonts.google.com/specimen/Libre+Bodoni — OFL, Impallari Type",
        "blurb": (
            "Giambattista Bodoni pushed the contrast dial to eleven. His types feature razor-thin "
            "hairlines and dramatic thick verticals — a style later called \"Modern\" (the irony of "
            "calling a 1798 design 'modern' in 2025 is not lost on us). Bodoni's Manuale Tipografico, "
            "published posthumously, contained 142 typefaces and remains one of the most beautiful "
            "type specimens ever printed. Vogue magazine still uses a Bodoni variant for its masthead."
        ),
        "features": {},
    },
    {
        "era": "1845 — The Slab Serif",
        "font_family": "Zilla Slab",
        "pango_font": "Zilla Slab",
        "source": "fonts.google.com/specimen/Zilla+Slab — OFL, Typotheque for Mozilla",
        "blurb": (
            "The Industrial Revolution needed loud type. Slab serifs — with their blunt, unbracketed "
            "serifs of equal weight — were designed to scream from handbills, posters, and newspaper "
            "headlines. Clarendon (1845) was the most famous, but the Egyptian style dates to 1815 "
            "when Vincent Figgins cast the first. Named 'Egyptian' not because of any actual Egyptian "
            "connection, but because Egyptomania was sweeping Europe after Napoleon's campaigns. "
            "Mozilla's Zilla Slab carries the tradition forward with a digital-native warmth."
        ),
        "features": {},
    },
    {
        "era": "1927 — Futura",
        "font_family": "Jost",
        "pango_font": "Jost",
        "source": "fonts.google.com/specimen/Jost — OFL, Owen Earl (Futura stand-in)",
        "blurb": (
            "Paul Renner designed Futura in 1927 for the Bauer foundry, and it became the defining "
            "typeface of modernism. Its near-perfect geometric circles and triangles were radical — "
            "a declaration that type should look forward, not backward. Stanley Kubrick used it in "
            "2001: A Space Odyssey. Wes Anderson uses it in everything. And in 1969, it literally "
            "went to the moon on the Apollo 11 commemorative plaque. "
            "Jost (by Owen Earl) is a faithful open-source geometric sans in the Futura tradition."
        ),
        "features": {},
    },
    {
        "era": "1931 — Times New Roman",
        "font_family": "Times New Roman",
        "pango_font": "Times New Roman",
        "source": "Bundled with Windows/Office — Monotype. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "In 1931, The Times of London commissioned Stanley Morison to redesign their newspaper "
            "type. Working with draftsman Victor Lardent, Morison created Times New Roman — optimized "
            "for narrow columns and cheap newsprint. It was never meant to be pretty. It was meant to "
            "be readable at small sizes on bad paper. Seven decades later, Microsoft bundled it with "
            "Windows, and every college student's essay defaulted to it. It's the typographic equivalent "
            "of khaki pants: unremarkable, inoffensive, everywhere."
        ),
        "features": {},
    },
    {
        "era": "1955 — Courier (IBM)",
        "font_family": "Courier New",
        "pango_font": "Courier New",
        "source": "Bundled with Windows/Office — IBM origin. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "Howard \"Bud\" Kettler designed Courier in 1955 for IBM's Selectric typewriters. "
            "IBM deliberately chose not to trademark it, making it freely available — a decision "
            "that ensured its ubiquity. Every monospaced terminal, screenplay, and government form "
            "owes something to Courier. Its fixed-width design means every character occupies exactly "
            "the same horizontal space, which is why programmers still reach for monospaced fonts "
            "and why court filings still specify Courier. Some traditions die hard."
        ),
        "features": {},
    },
    {
        "era": "1957 — Helvetica / Arial",
        "font_family": "Nimbus Sans",
        "pango_font": "Nimbus Sans",
        "source": "URW++ Helvetica clone — ships with ghostscript/texlive. fonts.urwpp.de",
        "blurb": (
            "Max Miedinger and Eduard Hoffmann designed Neue Haas Grotesk in 1957, renaming it "
            "Helvetica (Latin for 'Swiss') in 1960. It became the face of corporate modernism — "
            "used by American Airlines, Jeep, Toyota, and the NYC subway. In 1982, Robin Nicholas "
            "and Patricia Saunders at Monotype designed Arial as a metrically compatible alternative "
            "that Microsoft could license cheaply. Typographers can tell them apart by the 'a' tail, "
            "the 'G' bar, and the diagonal cut on the 't'. Everyone else just sees 'the normal font.' "
            "Nimbus Sans is URW's Helvetica-compatible libre clone, faithful to the Swiss original."
        ),
        "features": {},
    },
    {
        "era": "1982 — Arial (Microsoft)",
        "font_family": "Arial",
        "pango_font": "Arial",
        "source": "Bundled with Windows/Office — Monotype. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "Arial was Monotype's strategic masterstroke: a Helvetica substitute with matching metrics "
            "but just enough differences to avoid licensing fees. Microsoft bundled it with Windows 3.1 "
            "in 1992 and it conquered the world. Designers love to hate it. 'Arial is the font of "
            "people who don't care about fonts,' goes the saying — and that's exactly why it works. "
            "It's the path of least resistance, the default sans-serif, the typographic shrug emoji. "
            "Three billion people have used it. Probably including you, today."
        ),
        "features": {},
    },
    {
        "era": "1993 — Georgia (Microsoft)",
        "font_family": "Georgia",
        "pango_font": "Georgia",
        "source": "Bundled with Windows/Office — Matthew Carter. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "Matthew Carter — arguably the greatest living type designer — created Georgia in 1993 "
            "specifically for screen readability. Named after a tabloid headline ('Alien Heads Found "
            "in Georgia'), it was one of the first fonts designed from the pixel grid up rather than "
            "adapted from print. Its generous x-height and sturdy serifs made it the default 'readable "
            "serif' of the early web. Carter also designed Verdana, Bell Centennial (for phone books), "
            "and Miller (for newspapers). The man has literally shaped how billions of people read."
        ),
        "features": {},
    },
    {
        "era": "1996 — Verdana (Microsoft)",
        "font_family": "Verdana",
        "pango_font": "Verdana",
        "source": "Bundled with Windows/Office — Matthew Carter. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "Verdana is Georgia's sans-serif sibling, also by Matthew Carter, also designed for "
            "screens. Its name is a portmanteau of 'verdant' (the green of the Pacific Northwest, "
            "where Microsoft is headquartered) and 'Ana' (the name of designer Virginia Howlett's "
            "eldest daughter). The wide letterforms and tall x-height made it supremely legible on "
            "640×480 monitors. At 10px on a CRT, Verdana was more readable than anything else alive."
        ),
        "features": {},
    },
    {
        "era": "1994 — Comic Sans (Microsoft)",
        "font_family": "Comic Sans MS",
        "pango_font": "Comic Sans MS",
        "source": "Bundled with Windows/Office — Vincent Connare. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "Vincent Connare designed Comic Sans in 1994 after seeing Times New Roman in a Microsoft "
            "Bob speech bubble and thinking 'that's wrong.' He based it on the lettering in The Dark "
            "Knight Returns and Watchmen comics. It was never intended for body text — it was a UI "
            "font for children's software. But users discovered it, loved it, and put it everywhere: "
            "office memos, funeral programs, CERN's Higgs boson announcement. Typographers weep. "
            "Comic Sans doesn't care. It's having more fun than your serif ever will."
        ),
        "features": {},
    },
    {
        "era": "1996 — Trebuchet MS",
        "font_family": "Trebuchet MS",
        "pango_font": "Trebuchet MS",
        "source": "Bundled with Windows/Office — Vincent Connare. Linux: apt install ttf-mscorefonts-installer",
        "blurb": (
            "Vincent Connare also designed Trebuchet MS (yes, the same guy who made Comic Sans — "
            "range, right?). Named after a medieval siege engine because 'it launches words across "
            "the internet,' it was one of Microsoft's core web fonts. It occupies a peculiar middle "
            "ground: more personality than Arial, less chaos than Comic Sans. Its slightly humanist "
            "proportions and generous spacing made it a quiet workhorse of late-90s web design."
        ),
        "features": {},
    },
    {
        "era": "2004 — The ClearType Collection (Microsoft)",
        "font_family": "Caladea",
        "pango_font": "Caladea",
        "source": "fonts.google.com/specimen/Caladea — OFL, Carolina Giovagnoli (Cambria-compatible)",
        "blurb": (
            "When Microsoft developed ClearType subpixel rendering for LCD screens, they commissioned "
            "six new font families optimized for it: Calibri, Cambria, Candara, Consolas, Constantia, "
            "and Corbel (all C-names, because apparently Microsoft's font naming committee had one "
            "letter in mind). Calibri replaced Times New Roman as the Office default in 2007, which "
            "means more words have been set in Calibri than in any typeface in human history. "
            "Caladea is an open-source metric-compatible Cambria substitute from Carolina Giovagnoli."
        ),
        "features": {"onum": True},
    },
    {
        "era": "2010 — The Google Fonts Revolution: Roboto",
        "font_family": "Roboto",
        "pango_font": "Roboto",
        "source": "fonts.google.com/specimen/Roboto — Apache 2.0, Christian Robertson for Google",
        "blurb": (
            "When Google launched Google Fonts in 2010, it broke the foundry cartel overnight. "
            "Suddenly any designer could use quality typefaces for free, legally, on the web. "
            "Christian Robertson's Roboto (2011) became Android's system font and the most popular "
            "Google Font by a cosmic margin — it's on 27 million websites. Its dual nature (geometric "
            "skeleton, slightly humanist curves) makes it simultaneously mechanical and approachable. "
            "Google's design language, Material Design, is built on Roboto's proportions."
        ),
        "features": {"onum": True, "smcp": True},
    },
    {
        "era": "2010 — Open Sans",
        "font_family": "Open Sans",
        "pango_font": "Open Sans",
        "source": "fonts.google.com/specimen/Open+Sans — OFL, Steve Matteson for Google",
        "blurb": (
            "Steve Matteson designed Open Sans in 2011, commissioned by Google. Its open apertures "
            "and neutral forms make it the typographic equivalent of clean water — essential, "
            "invisible, everywhere. It's optimized for legibility across print, web, and mobile. "
            "WordPress.com, Google (itself), and countless government sites use it as their primary "
            "face. Open Sans is the second most popular Google Font, trailing only Roboto."
        ),
        "features": {},
    },
    {
        "era": "2010 — Lato",
        "font_family": "Lato",
        "pango_font": "Lato",
        "source": "fonts.google.com/specimen/Lato — OFL, Łukasz Dziedzic",
        "blurb": (
            "Łukasz Dziedzic designed Lato ('Summer' in Polish) in 2010, originally as a corporate "
            "typeface for a large client who ultimately went in a different direction. Their loss, "
            "everyone's gain. Lato's semi-rounded details give it warmth without sacrificing "
            "seriousness — it works for banking apps and wedding invitations alike. "
            "It's the third most popular Google Font and a strong contender for 'best free sans.'"
        ),
        "features": {},
    },
    {
        "era": "2011 — Merriweather",
        "font_family": "Merriweather",
        "pango_font": "Merriweather",
        "source": "fonts.google.com/specimen/Merriweather — OFL, Eben Sorkin",
        "blurb": (
            "Eben Sorkin designed Merriweather specifically for comfortable reading on screens — "
            "large x-height, slightly condensed letterforms, sturdy serifs that don't crumble at "
            "small sizes. Named after a 19th-century Kansas newspaper editor (because everything "
            "in type circles eventually loops back to print), it's become the go-to serif for long "
            "blog posts, online magazines, and Medium articles. Pairs beautifully with any neutral "
            "sans — the typographic buddy cop pairing of the decade."
        ),
        "features": {},
    },
    {
        "era": "2012 — Source Sans Pro (Adobe)",
        "font_family": "Source Sans 3",
        "pango_font": "Source Sans 3",
        "source": "fonts.google.com/specimen/Source+Sans+3 — OFL, Paul Hunt for Adobe",
        "blurb": (
            "Paul Hunt designed Source Sans Pro as Adobe's first open-source typeface, released in "
            "2012. It was a signal: even Adobe, the company that built its empire on proprietary type "
            "technology (PostScript, OpenType), saw the value of open fonts. Source Sans is a clean "
            "humanist sans with generous proportions — think 'Frutiger but free.' Its companion, "
            "Source Code Pro (monospaced), is beloved by programmers. Together they proved that "
            "open-source fonts could match commercial quality."
        ),
        "features": {"onum": True, "smcp": True, "ss01": True, "titl": True},
    },
    {
        "era": "2014 — Source Serif 4 (Adobe)",
        "font_family": "Source Serif 4",
        "pango_font": "Source Serif 4",
        "source": "fonts.google.com/specimen/Source+Serif+4 — OFL, Frank Grießhammer for Adobe",
        "blurb": (
            "Frank Grießhammer's Source Serif (2014, updated to v4 in 2021) is Adobe's open-source "
            "serif companion to Source Sans. Inspired by Pierre Simon Fournier's types from the 1740s, "
            "it bridges the gap between old-style and transitional designs. UC Berkeley uses it as "
            "their official serif typeface. Source Serif 4 is particularly interesting because it ships "
            "with extensive OpenType features — old-style figures, small caps, and stylistic sets — "
            "making it a rich test case for font detection tools like the one reading this page."
        ),
        "features": {"onum": True, "smcp": True, "ss01": True, "ss02": True},
    },
    {
        "era": "2014 — Noto Serif (Google + Monotype)",
        "font_family": "Noto Serif",
        "pango_font": "Noto Serif",
        "source": "fonts.google.com/specimen/Noto+Serif — OFL, Google + Monotype",
        "blurb": (
            "The Noto project is Google's audacious attempt to create fonts for every writing system "
            "in Unicode — 'No Tofu' (the nickname for □ missing-glyph boxes). It covers 146 scripts "
            "and 800+ languages. Noto Serif's Latin is a solid transitional design, but the real story "
            "is the scope: from Armenian to Zanabazar Square, Noto ensures no one sees a blank box "
            "where their language should be. It's the most democratic type project ever attempted."
        ),
        "features": {"onum": True, "smcp": True},
    },
    {
        "era": "2009 — PT Serif (ParaType)",
        "font_family": "PT Serif",
        "pango_font": "PT Serif",
        "source": "fonts.google.com/specimen/PT+Serif — OFL, Alexandra Korolkova at ParaType",
        "blurb": (
            "PT Serif was commissioned by the Russian government as part of a project to create "
            "a public font family covering all the languages of the Russian Federation — from Russian "
            "Cyrillic to Tatar, Bashkir, and dozens of minority scripts. Designed by Alexandra "
            "Korolkova at ParaType, it's a transitional serif that works beautifully in both Latin "
            "and Cyrillic. State-funded type for the public good: a concept as old as Garamond's "
            "royal commissions, updated for the digital commons."
        ),
        "features": {},
    },
    {
        "era": "2011 — Playfair Display",
        "font_family": "Playfair Display",
        "pango_font": "Playfair Display",
        "source": "fonts.google.com/specimen/Playfair+Display — OFL, Claus Eggers Sørensen",
        "blurb": (
            "Claus Eggers Sørensen designed Playfair Display as a nod to the high-contrast "
            "types of the European Enlightenment — Baskerville, Bodoni, and the Didots. Its "
            "delicate hairlines and generous proportions make it a display face par excellence: "
            "magazine headings, wedding invitations, that chic restaurant menu. Not great for "
            "small body text (those hairlines vanish below 14px), but for titles at 36pt+, "
            "Playfair is the free font that finally let indie designers stop pirating Didot."
        ),
        "features": {},
    },
    {
        "era": "2017 — IBM Plex (IBM)",
        "font_family": "IBM Plex Sans",
        "pango_font": "IBM Plex Sans",
        "source": "fonts.google.com/specimen/IBM+Plex+Sans — OFL, Mike Abbink + Bold Monday for IBM",
        "blurb": (
            "When IBM replaced Helvetica Neue with their custom IBM Plex family in 2017, it was "
            "a $100B company saying 'we need our own voice.' Mike Abbink and Bold Monday designed "
            "Plex in Sans, Serif, and Mono — all open-source. The design threads a needle between "
            "Helvetica's neutrality and Futura's geometry, with distinctive details like the slashed "
            "zero and the slab-serif-inflected terminals. IBM Plex is what corporate type looks like "
            "when the corporation actually cares about craft."
        ),
        "features": {},
    },
    {
        "era": "2017 — IBM Plex Serif",
        "font_family": "IBM Plex Serif",
        "pango_font": "IBM Plex Serif",
        "source": "fonts.google.com/specimen/IBM+Plex+Serif — OFL, Mike Abbink + Bold Monday for IBM",
        "blurb": (
            "IBM Plex Serif is the quieter sibling — a contemporary slab-influenced serif that "
            "pairs naturally with Plex Sans. Its slightly mechanical structure reflects IBM's "
            "engineering DNA while remaining warm and readable. Together, the Plex family demonstrates "
            "that corporate identity fonts don't have to be boring — they just need to be consistent. "
            "Every IBM product, from Watson to Cloud, uses Plex exclusively."
        ),
        "features": {},
    },
    {
        "era": "2017 — IBM Plex Mono",
        "font_family": "IBM Plex Mono",
        "pango_font": "IBM Plex Mono",
        "source": "fonts.google.com/specimen/IBM+Plex+Mono — OFL, Mike Abbink + Bold Monday for IBM",
        "blurb": (
            "IBM Plex Mono completes the Plex trilogy with a monospaced design that nods to "
            "IBM's original Selectric typefaces. Where Courier is a relic of the mechanical era, "
            "Plex Mono is designed for modern terminals and code editors. Its distinctive glyph "
            "shapes make zero/O and one/l/I instantly distinguishable — the single most important "
            "trait in a programming font. It's proof that even monospace can be beautiful."
        ),
        "features": {},
    },
    {
        "era": "2018 — Inter (Rasmus Andersson)",
        "font_family": "Inter",
        "pango_font": "Inter",
        "source": "fonts.google.com/specimen/Inter — OFL, Rasmus Andersson. Also rsms.me/inter",
        "blurb": (
            "Rasmus Andersson, a designer at Figma, created Inter as a typeface optimized for "
            "user interfaces — specifically, for the awkward sizes between 11px and 16px where most "
            "UI text lives. Its tall x-height, open apertures, and extensive kerning pairs make it "
            "the default choice for design tools, dashboards, and apps. With 5 stylistic sets, small "
            "caps, and a variable-font axis for weight, Inter is arguably the most polished open-source "
            "UI font available. It's what Helvetica would be if Helvetica were designed for Retina screens."
        ),
        # Use the system OTF which has features
        "features": {"smcp": True, "ss01": True, "ss02": True, "ss03": True},
    },
]


def render_pango_to_png(text, font_name, size_pt, out_png, width_px=1020,
                         features=None, is_bold=False, is_italic=False):
    """Render text with pango-view to a PNG file.
    
    Default width_px=1020 is half-column at 300dpi (for two-column layout).
    At 300dpi, 1pt ≈ 4.17px, so 10pt text ≈ 42px rendered height.
    All body text must be ≥10pt to remain fair at fax resolution.
    """
    markup_parts = []
    style_attrs = [f'font="{font_name} {size_pt}"']
    if is_bold:
        style_attrs.append('weight="bold"')
    if is_italic:
        style_attrs.append('style="italic"')
    if features:
        feat_str = ",".join(f"{f}=1" for f in features)
        style_attrs.append(f'font_features="{feat_str}"')

    attr_str = " ".join(style_attrs)
    # Escape text for pango markup
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    markup = f'<span {attr_str}>{safe}</span>'

    # pango-view requires a file argument for markup
    markup_file = out_png + ".markup"
    with open(markup_file, "w") as f:
        f.write(markup)

    cmd = [
        "pango-view", "--markup", "--no-display",
        f"--width={width_px}", "--wrap=word",
        "--margin=0",
        f"--output={out_png}",
        "--backend=cairo",
        markup_file,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.unlink(markup_file)
    except OSError:
        pass
    if proc.returncode != 0:
        print(f"  WARN: pango-view failed for {font_name}: {proc.stderr[:200]}")
        return False
    return os.path.exists(out_png)


def build_section_images(section, idx, tmpdir):
    """Build all image strips for a section. Returns list of PNG paths in order.
    
    Text sizes (all ≥10pt for fax-resolution fairness):
      - Section header: 14pt bold (in Jost)
      - Source URL: 8pt italic (Source Serif 4) — intentionally below 10pt as a
        metadata line, not a test target
      - Body blurb: 11pt (in the section's own font — the main test target)
      - Pangram/alphabet/digit lines: 10pt
      - OT variant demos: 10pt with features enabled
    
    All rendered at column width (1020px at 300dpi ≈ 3.4 inches) for
    two-column page composition.
    """
    images = []
    era = section["era"]
    font = section["pango_font"]
    blurb = section["blurb"]
    source = section.get("source", "")
    features = section.get("features", {})
    prefix = f"{tmpdir}/s{idx:02d}"

    COL_W = 1020  # half-page column width at 300dpi

    # 1. Era header (rendered in Jost bold — geometric, modern)
    hdr_file = f"{prefix}_00_header.png"
    render_pango_to_png(era, "Jost", 14, hdr_file, width_px=COL_W, is_bold=True)
    images.append(hdr_file)

    # 1b. Source/download info (small italic, in Source Serif 4)
    if source:
        src_file = f"{prefix}_00b_source.png"
        render_pango_to_png(f"Font: {source}", "Source Serif 4", 8, src_file,
                            width_px=COL_W, is_italic=True)
        images.append(src_file)

    # 2. Blurb paragraph in the actual font — 11pt body, the main test target
    blurb_file = f"{prefix}_01_blurb.png"
    render_pango_to_png(blurb, font, 11, blurb_file, width_px=COL_W)
    images.append(blurb_file)

    # 3. Pangram + digits line
    pangram = "The quick brown fox jumps over 1,234,567,890 lazy dogs."
    pang_file = f"{prefix}_02_pangram.png"
    render_pango_to_png(pangram, font, 10, pang_file, width_px=COL_W)
    images.append(pang_file)

    # 4. Uppercase alphabet
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ  abcdefghijklmnopqrstuvwxyz"
    alpha_file = f"{prefix}_03_alpha.png"
    render_pango_to_png(upper, font, 10, alpha_file, width_px=COL_W)
    images.append(alpha_file)

    # 5. Default digits line
    digits = "Lining figures: 0 1 2 3 4 5 6 7 8 9"
    dig_file = f"{prefix}_04_digits.png"
    render_pango_to_png(digits, font, 10, dig_file, width_px=COL_W)
    images.append(dig_file)

    # 6. OT variant demos — 10pt with features enabled
    if features.get("onum"):
        onum_text = "Old-style figures (onum): 0 1 2 3 4 5 6 7 8 9"
        onum_file = f"{prefix}_05_onum.png"
        render_pango_to_png(onum_text, font, 10, onum_file,
                            width_px=COL_W, features=["onum"])
        images.append(onum_file)

    if features.get("smcp"):
        smcp_text = "Small Caps (smcp): The Quick Brown Fox Jumps Over The Lazy Dog"
        smcp_file = f"{prefix}_06_smcp.png"
        render_pango_to_png(smcp_text, font, 10, smcp_file,
                            width_px=COL_W, features=["smcp"])
        images.append(smcp_file)

    for ss in ["ss01", "ss02", "ss03"]:
        if features.get(ss):
            ss_text = f"Stylistic Set {ss[-2:]} ({ss}): abcdefghijklmnopqrstuvwxyz 0123456789"
            ss_file = f"{prefix}_07_{ss}.png"
            render_pango_to_png(ss_text, font, 10, ss_file,
                                width_px=COL_W, features=[ss])
            images.append(ss_file)

    if features.get("titl"):
        titl_text = "Titling (titl): ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        titl_file = f"{prefix}_08_titl.png"
        render_pango_to_png(titl_text, font, 10, titl_file,
                            width_px=COL_W, features=["titl"])
        images.append(titl_file)

    return images


def compose_pages(all_section_images, tmpdir):
    """Stack section images into a two-column US Letter layout.
    
    Two-column layout at 300dpi:
      Page:    2550 × 3300 px (8.5" × 11")
      Margins: 150px left/right, 120px top/bottom
      Gutter:  80px between columns
      Column:  (2550 - 2×150 - 80) / 2 = 1035px each
    
    Sections flow left-to-right, top-to-bottom, snaking across columns.
    A section that won't fit in the remaining column space starts at the
    top of the next available column (which may be the right column on
    the same page, or the left column of a new page).
    """
    from PIL import Image

    MARGIN_TOP = 120
    MARGIN_SIDE = 150
    GUTTER = 80
    SECTION_GAP = 40
    INTRA_GAP = 6
    COL_W = (PAGE_W - 2 * MARGIN_SIDE - GUTTER) // 2  # ~1035px
    USABLE_H = PAGE_H - 2 * MARGIN_TOP

    pages = []          # list of finished page PNGs
    page_num = 0
    # Current page image (created lazily)
    cur_page = None
    # Two columns: track y position for each
    col_y = [0, 0]     # [left_col_y, right_col_y]
    cur_col = 0         # 0 = left, 1 = right

    def ensure_page():
        nonlocal cur_page
        if cur_page is None:
            cur_page = Image.new("L", (PAGE_W, PAGE_H), 255)

    def flush_page():
        nonlocal page_num, cur_page, col_y, cur_col
        if cur_page is not None:
            ppath = f"{tmpdir}/page_{page_num:03d}.png"
            cur_page.save(ppath)
            pages.append(ppath)
            page_num += 1
            cur_page = None
        col_y = [0, 0]
        cur_col = 0

    def col_x(col_idx):
        """Left edge of a column in page coordinates."""
        if col_idx == 0:
            return MARGIN_SIDE
        else:
            return MARGIN_SIDE + COL_W + GUTTER

    for section_idx, section_imgs in enumerate(all_section_images):
        # Load and scale all strips for this section
        loaded = []
        section_h = 0
        for imgpath in section_imgs:
            if not os.path.exists(imgpath):
                continue
            im = Image.open(imgpath).convert("L")
            # Scale to column width if needed
            if im.width > COL_W:
                ratio = COL_W / im.width
                im = im.resize((COL_W, int(im.height * ratio)), Image.LANCZOS)
            loaded.append(im)
            section_h += im.height + INTRA_GAP
        section_h += SECTION_GAP  # gap after section

        if not loaded:
            continue

        # Try to fit section in current column
        remaining = USABLE_H - col_y[cur_col]
        if col_y[cur_col] > 0 and section_h > remaining:
            # Won't fit — try next column
            if cur_col == 0:
                cur_col = 1
                remaining = USABLE_H - col_y[cur_col]
                if col_y[cur_col] > 0 and section_h > remaining:
                    # Right column also full — new page
                    flush_page()
            else:
                # Already in right column — new page
                flush_page()

        # Place strips in current column
        ensure_page()
        x = col_x(cur_col)
        for im in loaded:
            # If this strip alone overflows, advance column/page
            if col_y[cur_col] + im.height > USABLE_H:
                if cur_col == 0:
                    cur_col = 1
                else:
                    flush_page()
                    ensure_page()
                x = col_x(cur_col)

            cur_page.paste(im, (x, MARGIN_TOP + col_y[cur_col]))
            col_y[cur_col] += im.height + INTRA_GAP

        col_y[cur_col] += SECTION_GAP - INTRA_GAP  # section separator

    flush_page()
    return pages


def pages_to_pdf(page_pngs, out_pdf):
    """Convert page PNGs to PDF via img2pdf (lossless, exact DPI)."""
    import img2pdf
    # img2pdf handles PNGs natively and preserves DPI metadata
    with open(out_pdf, "wb") as f:
        # Set layout to US Letter at 300dpi
        layout = img2pdf.get_layout_fun(
            pagesize=(img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11))
        )
        f.write(img2pdf.convert(page_pngs, layout_fun=layout))


def build_ground_truth(sections, out_json):
    """Write ground truth mapping."""
    truth = {
        "description": "Ground truth for font-timeline-specimen.pdf",
        "generated": "2025-05-10",
        "sections": []
    }
    for idx, s in enumerate(sections):
        entry = {
            "index": idx,
            "era": s["era"],
            "font_family": s["font_family"],
            "pango_font": s["pango_font"],
            "source": s.get("source", ""),
            "features_demonstrated": list(k for k, v in s.get("features", {}).items() if v),
        }
        truth["sections"].append(entry)
    with open(out_json, "w") as f:
        json.dump(truth, f, indent=2)


def main():
    tmpdir = "/tmp/specimen-render"
    os.makedirs(tmpdir, exist_ok=True)

    # First: render title page (full width, spans both columns)
    FULL_W = PAGE_W - 2 * 150  # 2250px — full usable width
    title_imgs = []
    t1 = f"{tmpdir}/title_01.png"
    render_pango_to_png(
        "A Timeline of Typography",
        "Playfair Display", 36, t1, width_px=FULL_W, is_bold=False
    )
    title_imgs.append(t1)

    t2 = f"{tmpdir}/title_02.png"
    render_pango_to_png(
        "Five Centuries of Letterforms — A Font Specimen for Unscan",
        "Source Serif 4", 14, t2, width_px=FULL_W, is_italic=True
    )
    title_imgs.append(t2)

    t3 = f"{tmpdir}/title_03.png"
    render_pango_to_png(
        "\n\nThis document is a ground-truth test specimen. Each section is rendered in a known "
        "font, with OpenType variants demonstrated where available. Feed it to unscan and verify "
        "it identifies each correctly.\n\n"
        "Every blurb is set in the font it describes — so you're reading Garamond IN Garamond, "
        "Bodoni IN Bodoni, and yes, Comic Sans IN Comic Sans.\n\n"
        "Fonts are ordered chronologically by their original design date. Digital revivals stand "
        "in for historical originals where exact versions aren't freely available.\n\n"
        "Body text is set at 11pt — large enough to survive fax-resolution degradation, small "
        "enough to be a realistic test of font detection. Two-column layout increases density "
        "and tests the tool's ability to handle adjacent columns with different typefaces.\n\n"
        "Lining figures:  0 1 2 3 4 5 6 7 8 9\n"
        "Test string:     The quick brown fox jumps over 42 lazy dogs.",
        "Source Serif 4", 11, t3, width_px=FULL_W
    )
    title_imgs.append(t3)

    # Title page is special — full width, rendered as its own page
    all_sections = []  # section images for column flow (NOT title)

    print(f"Rendering {len(SECTIONS)} sections...")
    for idx, section in enumerate(SECTIONS):
        print(f"  [{idx+1}/{len(SECTIONS)}] {section['era']} — {section['font_family']}")
        imgs = build_section_images(section, idx, tmpdir)
        all_sections.append(imgs)

    print("Composing pages...")

    # Build title page first (full-width, centered)
    from PIL import Image
    title_page = Image.new("L", (PAGE_W, PAGE_H), 255)
    title_y = 300  # start title lower for visual balance
    for timg_path in title_imgs:
        if not os.path.exists(timg_path):
            continue
        tim = Image.open(timg_path).convert("L")
        # Center horizontally
        x = (PAGE_W - tim.width) // 2
        title_page.paste(tim, (x, title_y))
        title_y += tim.height + 12
    title_path = f"{tmpdir}/page_title.png"
    title_page.save(title_path)

    # Compose remaining sections into two-column pages
    col_pages = compose_pages(all_sections, tmpdir)
    all_page_pngs = [title_path] + col_pages
    print(f"  {len(all_page_pngs)} pages (1 title + {len(col_pages)} content)")

    out_pdf = "/home/hatch/workspace/repos/unscan/test-docs/font-timeline-specimen.pdf"
    print(f"Writing PDF: {out_pdf}")
    pages_to_pdf(all_page_pngs, out_pdf)

    out_json = "/home/hatch/workspace/repos/unscan/test-docs/font-timeline-specimen.json"
    print(f"Writing ground truth: {out_json}")
    build_ground_truth(SECTIONS, out_json)

    # Also create a "scanned" version — rasterize and reassemble
    # Simulates a real flatbed scan: slight rotation (skew), Gaussian blur,
    # speckle noise, and off-white paper background.
    print("Creating scanned version...")
    scanned_dir = f"{tmpdir}/scanned"
    os.makedirs(scanned_dir, exist_ok=True)

    from PIL import Image, ImageFilter
    import numpy as np
    import random

    # Consistent skew for all pages (as if the whole doc was placed crooked)
    skew_deg = random.uniform(1.5, 3.0)
    # Randomly pick CW or CCW
    if random.random() < 0.5:
        skew_deg = -skew_deg
    print(f"  Skew: {skew_deg:.1f}°")

    scanned_pages = []
    for i, ppng in enumerate(all_page_pngs):
        im = Image.open(ppng).convert("L")

        # 1. Rotate (skew) — fill exposed edges with near-white (like scanner lid)
        #    expand=True so we don't clip corners; PIL fills with fillcolor
        im = im.rotate(skew_deg, resample=Image.BICUBIC, expand=True,
                        fillcolor=245)

        # Crop back to original page size (centered)
        cx, cy = im.width // 2, im.height // 2
        left = cx - PAGE_W // 2
        top = cy - PAGE_H // 2
        im = im.crop((left, top, left + PAGE_W, top + PAGE_H))

        # 2. Off-white paper background — real scanned paper isn't pure 255
        arr = np.array(im, dtype=np.float32)
        # Paper tone: ~245 with slight per-pixel variation (paper texture)
        paper_noise = np.random.normal(0, 1.5, arr.shape).astype(np.float32)
        arr = np.clip(arr + paper_noise, 0, 255)
        # Darken overall slightly (scanner doesn't produce pure white)
        arr = arr * 0.96 + 8  # shifts white from 255 to ~253

        # 3. Speckle noise — random dark spots (dust on scanner glass)
        speckle = np.random.random(arr.shape)
        arr[speckle < 0.0003] = np.random.randint(40, 120, size=np.sum(speckle < 0.0003))

        # 4. Light Gaussian blur — scanner optics aren't perfectly sharp
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        im = im.filter(ImageFilter.GaussianBlur(radius=0.7))

        spath = f"{scanned_dir}/page_{i:03d}.png"
        im.save(spath)
        scanned_pages.append(spath)

    scanned_pdf = "/home/hatch/workspace/repos/unscan/test-docs/font-timeline-specimen-scanned.pdf"
    pages_to_pdf(scanned_pages, scanned_pdf)
    print(f"Scanned PDF: {scanned_pdf}")

    print("Done!")


if __name__ == "__main__":
    main()
