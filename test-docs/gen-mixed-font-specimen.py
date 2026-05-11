#!/usr/bin/env python3
"""
Generate a mixed-font specimen PDF for testing unscan's ability to recover
font changes within a line and within a word.

Output:
  mixed-font-specimen.pdf           — vector PDF with mixed fonts per line
  mixed-font-specimen-raster.pdf    — 300 DPI rasterized version for unscan
  mixed-font-ground-truth.json      — per-line font span ground truth

Uses Liberation family (metrically equivalent to Arial/Times/Courier) because
it provides Regular, Italic, Bold, BoldItalic for Sans, Serif, and Mono.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, KeepTogether, Flowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR

PAGE_W, PAGE_H = letter

# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------
FONT_DIR = Path("/usr/share/fonts/truetype")

FONTS = {
    # Liberation Sans family (sans-serif)
    "Sans":            FONT_DIR / "liberation/LiberationSans-Regular.ttf",
    "Sans-Italic":     FONT_DIR / "liberation/LiberationSans-Italic.ttf",
    "Sans-Bold":       FONT_DIR / "liberation/LiberationSans-Bold.ttf",
    "Sans-BoldItalic": FONT_DIR / "liberation/LiberationSans-BoldItalic.ttf",
    # Liberation Serif family (serif)
    "Serif":            FONT_DIR / "liberation/LiberationSerif-Regular.ttf",
    "Serif-Italic":     FONT_DIR / "liberation/LiberationSerif-Italic.ttf",
    "Serif-Bold":       FONT_DIR / "liberation/LiberationSerif-Bold.ttf",
    "Serif-BoldItalic": FONT_DIR / "liberation/LiberationSerif-BoldItalic.ttf",
    # Liberation Mono (monospace)
    "Mono":            FONT_DIR / "liberation/LiberationMono-Regular.ttf",
    "Mono-Italic":     FONT_DIR / "liberation/LiberationMono-Italic.ttf",
    "Mono-Bold":       FONT_DIR / "liberation/LiberationMono-Bold.ttf",
    # DejaVu for extra contrast tests
    "DejaVu":          FONT_DIR / "dejavu/DejaVuSans.ttf",
    "DejaVu-Bold":     FONT_DIR / "dejavu/DejaVuSans-Bold.ttf",
    "DejaVu-Serif":    FONT_DIR / "dejavu/DejaVuSerif.ttf",
    "DejaVu-Mono":     FONT_DIR / "dejavu/DejaVuSansMono.ttf",
}

for rl_name, ttf_path in FONTS.items():
    if ttf_path.exists():
        pdfmetrics.registerFont(TTFont(rl_name, str(ttf_path)))
    else:
        print(f"WARNING: missing font {ttf_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def font_tag(text, font_name):
    """Wrap text in a ReportLab <font> tag."""
    return f'<font name="{font_name}">{text}</font>'


def make_span(text, font):
    """Create a ground-truth span dict."""
    return {"text": text, "font": font}


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------
# Each entry: (category, display_line_markup, ground_truth_spans)
# Markup uses ReportLab XML tags. Ground truth uses real font names.

REGULAR = "Sans"
ITALIC = "Sans-Italic"
BOLD = "Sans-Bold"
BOLD_ITALIC = "Sans-BoldItalic"
SERIF = "Serif"
SERIF_ITALIC = "Serif-Italic"
SERIF_BOLD = "Serif-Bold"
MONO = "Mono"
MONO_BOLD = "Mono-Bold"

TEST_CASES = []

def add(category, markup, spans):
    TEST_CASES.append((category, markup, spans))


# ── LaTeX-style math mixing ──────────────────────────────────────────────

add("Math: italic variable in sans text",
    font_tag("Uses ", REGULAR) + font_tag("k", SERIF_ITALIC) + font_tag("-means clustering", REGULAR),
    [make_span("Uses ", REGULAR),
     make_span("k", SERIF_ITALIC),
     make_span("-means clustering", REGULAR)])

add("Math: italic variable mid-sentence",
    font_tag("The variable ", REGULAR) + font_tag("x", SERIF_ITALIC) + font_tag(" represents displacement", REGULAR),
    [make_span("The variable ", REGULAR),
     make_span("x", SERIF_ITALIC),
     make_span(" represents displacement", REGULAR)])

add("Math: expression with mixed fonts",
    font_tag("Let ", REGULAR) + font_tag("f", SERIF_ITALIC) + font_tag("(", REGULAR)
    + font_tag("x", SERIF_ITALIC) + font_tag(") = ", REGULAR)
    + font_tag("x", SERIF_ITALIC) + font_tag("² + 2", REGULAR)
    + font_tag("x", SERIF_ITALIC) + font_tag(" + 1", REGULAR),
    [make_span("Let ", REGULAR),
     make_span("f", SERIF_ITALIC), make_span("(", REGULAR),
     make_span("x", SERIF_ITALIC), make_span(") = ", REGULAR),
     make_span("x", SERIF_ITALIC), make_span("² + 2", REGULAR),
     make_span("x", SERIF_ITALIC), make_span(" + 1", REGULAR)])

add("Math: multiple variables",
    font_tag("Given ", REGULAR) + font_tag("n", SERIF_ITALIC) + font_tag(" samples with ", REGULAR)
    + font_tag("d", SERIF_ITALIC) + font_tag(" dimensions and ", REGULAR)
    + font_tag("k", SERIF_ITALIC) + font_tag(" clusters", REGULAR),
    [make_span("Given ", REGULAR), make_span("n", SERIF_ITALIC),
     make_span(" samples with ", REGULAR), make_span("d", SERIF_ITALIC),
     make_span(" dimensions and ", REGULAR), make_span("k", SERIF_ITALIC),
     make_span(" clusters", REGULAR)])

add("Math: subscripted-style variable",
    font_tag("Compute ", REGULAR) + font_tag("x", SERIF_ITALIC)
    + font_tag("₁ + ", REGULAR) + font_tag("x", SERIF_ITALIC)
    + font_tag("₂ for each pair", REGULAR),
    [make_span("Compute ", REGULAR), make_span("x", SERIF_ITALIC),
     make_span("₁ + ", REGULAR), make_span("x", SERIF_ITALIC),
     make_span("₂ for each pair", REGULAR)])


# ── Inline emphasis ──────────────────────────────────────────────────────

add("Emphasis: italic word",
    font_tag("Some things need ", REGULAR) + font_tag("emphasis", ITALIC) + font_tag(" more than others.", REGULAR),
    [make_span("Some things need ", REGULAR),
     make_span("emphasis", ITALIC),
     make_span(" more than others.", REGULAR)])

add("Emphasis: bold word",
    font_tag("And then there are ", REGULAR) + font_tag("bold", BOLD) + font_tag(" visions.", REGULAR),
    [make_span("And then there are ", REGULAR),
     make_span("bold", BOLD),
     make_span(" visions.", REGULAR)])

add("Emphasis: bold-italic phrase",
    font_tag("This is both ", REGULAR) + font_tag("bold and italic", BOLD_ITALIC) + font_tag(" together.", REGULAR),
    [make_span("This is both ", REGULAR),
     make_span("bold and italic", BOLD_ITALIC),
     make_span(" together.", REGULAR)])

add("Emphasis: italic phrase mid-sentence",
    font_tag("The report concluded that ", REGULAR) + font_tag("further investigation is warranted", ITALIC)
    + font_tag(" before proceeding.", REGULAR),
    [make_span("The report concluded that ", REGULAR),
     make_span("further investigation is warranted", ITALIC),
     make_span(" before proceeding.", REGULAR)])

add("Emphasis: bold at start",
    font_tag("Warning:", BOLD) + font_tag(" Do not proceed without authorization.", REGULAR),
    [make_span("Warning:", BOLD),
     make_span(" Do not proceed without authorization.", REGULAR)])

add("Emphasis: bold at end",
    font_tag("The default value is ", REGULAR) + font_tag("true", BOLD) + font_tag(".", REGULAR),
    [make_span("The default value is ", REGULAR),
     make_span("true", BOLD),
     make_span(".", REGULAR)])


# ── Monospace in running text ────────────────────────────────────────────

add("Code: command in text",
    font_tag("The ", REGULAR) + font_tag("grep", MONO) + font_tag(" command searches files.", REGULAR),
    [make_span("The ", REGULAR),
     make_span("grep", MONO),
     make_span(" command searches files.", REGULAR)])

add("Code: URL in text",
    font_tag("Visit ", REGULAR) + font_tag("example.com", MONO) + font_tag(" for more information.", REGULAR),
    [make_span("Visit ", REGULAR),
     make_span("example.com", MONO),
     make_span(" for more information.", REGULAR)])

add("Code: function name",
    font_tag("Call ", REGULAR) + font_tag("initialize()", MONO) + font_tag(" before any other method.", REGULAR),
    [make_span("Call ", REGULAR),
     make_span("initialize()", MONO),
     make_span(" before any other method.", REGULAR)])

add("Code: path in text",
    font_tag("Edit the config at ", REGULAR) + font_tag("/etc/unscan.conf", MONO) + font_tag(" to change defaults.", REGULAR),
    [make_span("Edit the config at ", REGULAR),
     make_span("/etc/unscan.conf", MONO),
     make_span(" to change defaults.", REGULAR)])

add("Code: multiple inline code spans",
    font_tag("Use ", REGULAR) + font_tag("--overlay", MONO)
    + font_tag(" or ", REGULAR) + font_tag("--compare", MONO)
    + font_tag(" for debugging.", REGULAR),
    [make_span("Use ", REGULAR), make_span("--overlay", MONO),
     make_span(" or ", REGULAR), make_span("--compare", MONO),
     make_span(" for debugging.", REGULAR)])

add("Code: variable assignment",
    font_tag("Set ", REGULAR) + font_tag("RUST_LOG=info", MONO)
    + font_tag(" in your environment.", REGULAR),
    [make_span("Set ", REGULAR),
     make_span("RUST_LOG=info", MONO),
     make_span(" in your environment.", REGULAR)])


# ── Serif / sans mixing ─────────────────────────────────────────────────

add("Serif+Sans: book title in serif",
    font_tag("As described in ", REGULAR) + font_tag("The Elements of Typographic Style", SERIF_ITALIC)
    + font_tag(" by Bringhurst.", REGULAR),
    [make_span("As described in ", REGULAR),
     make_span("The Elements of Typographic Style", SERIF_ITALIC),
     make_span(" by Bringhurst.", REGULAR)])

add("Serif+Sans: journal citation",
    font_tag("Smith, J. (2024). ", REGULAR) + font_tag("On the Nature of Font Matching", SERIF_ITALIC)
    + font_tag(". ", REGULAR) + font_tag("Journal of Typography", SERIF_ITALIC)
    + font_tag(", 12(3), 45–67.", REGULAR),
    [make_span("Smith, J. (2024). ", REGULAR),
     make_span("On the Nature of Font Matching", SERIF_ITALIC),
     make_span(". ", REGULAR),
     make_span("Journal of Typography", SERIF_ITALIC),
     make_span(", 12(3), 45–67.", REGULAR)])


# ── Legal / formal document patterns ────────────────────────────────────

add("Legal: defined term in bold",
    font_tag("The ", REGULAR) + font_tag("Licensor", BOLD)
    + font_tag(" grants the ", REGULAR) + font_tag("Licensee", BOLD)
    + font_tag(" a non-exclusive right.", REGULAR),
    [make_span("The ", REGULAR), make_span("Licensor", BOLD),
     make_span(" grants the ", REGULAR), make_span("Licensee", BOLD),
     make_span(" a non-exclusive right.", REGULAR)])

add("Legal: section reference bold",
    font_tag("See ", REGULAR) + font_tag("Section 4.2", BOLD)
    + font_tag(" for the complete terms.", REGULAR),
    [make_span("See ", REGULAR), make_span("Section 4.2", BOLD),
     make_span(" for the complete terms.", REGULAR)])


# ── Documentation patterns ───────────────────────────────────────────────

add("Docs: parameter name",
    font_tag("The ", REGULAR) + font_tag("timeout", MONO)
    + font_tag(" parameter accepts values in milliseconds.", REGULAR),
    [make_span("The ", REGULAR), make_span("timeout", MONO),
     make_span(" parameter accepts values in milliseconds.", REGULAR)])

add("Docs: return type",
    font_tag("Returns a ", REGULAR) + font_tag("Vec<(String, f32)>", MONO)
    + font_tag(" of font matches.", REGULAR),
    [make_span("Returns a ", REGULAR), make_span("Vec<(String, f32)>", MONO),
     make_span(" of font matches.", REGULAR)])

add("Docs: bold label + regular body",
    font_tag("Parameters: ", BOLD) + font_tag("crops — character image crops, index — the font index", REGULAR),
    [make_span("Parameters: ", BOLD),
     make_span("crops — character image crops, index — the font index", REGULAR)])


# ── Multi-font within a single word ─────────────────────────────────────

add("Intra-word: bold prefix",
    font_tag("un", BOLD) + font_tag("believable results", REGULAR),
    [make_span("un", BOLD), make_span("believable results", REGULAR)])

add("Intra-word: italic suffix",
    font_tag("pseudo", REGULAR) + font_tag("random", ITALIC) + font_tag(" numbers", REGULAR),
    [make_span("pseudo", REGULAR), make_span("random", ITALIC), make_span(" numbers", REGULAR)])


# ── Headers / labels mixed with body ────────────────────────────────────

add("Header+body: bold header same line",
    font_tag("Abstract. ", BOLD) + font_tag("This paper presents a novel approach to font matching in scanned documents.", REGULAR),
    [make_span("Abstract. ", BOLD),
     make_span("This paper presents a novel approach to font matching in scanned documents.", REGULAR)])

add("Header+body: bold-italic label",
    font_tag("Theorem 1. ", BOLD_ITALIC) + font_tag("For any finite set of fonts, there exists a feature vector of dimension ", REGULAR)
    + font_tag("d", SERIF_ITALIC) + font_tag(" that uniquely identifies each font.", REGULAR),
    [make_span("Theorem 1. ", BOLD_ITALIC),
     make_span("For any finite set of fonts, there exists a feature vector of dimension ", REGULAR),
     make_span("d", SERIF_ITALIC),
     make_span(" that uniquely identifies each font.", REGULAR)])

add("Header+body: mono label",
    font_tag("unscan v8i", MONO_BOLD) + font_tag(" — Font matching engine with per-character indexing", REGULAR),
    [make_span("unscan v8i", MONO_BOLD),
     make_span(" — Font matching engine with per-character indexing", REGULAR)])


# ── Complex multi-font lines ────────────────────────────────────────────

add("Complex: four fonts in one line",
    font_tag("The ", REGULAR) + font_tag("CharIndex", MONO)
    + font_tag(" struct stores a ", REGULAR) + font_tag("k-d tree", BOLD)
    + font_tag(" for ", REGULAR) + font_tag("O", SERIF_ITALIC)
    + font_tag("(log ", REGULAR) + font_tag("n", SERIF_ITALIC)
    + font_tag(") lookups.", REGULAR),
    [make_span("The ", REGULAR), make_span("CharIndex", MONO),
     make_span(" struct stores a ", REGULAR), make_span("k-d tree", BOLD),
     make_span(" for ", REGULAR), make_span("O", SERIF_ITALIC),
     make_span("(log ", REGULAR), make_span("n", SERIF_ITALIC),
     make_span(") lookups.", REGULAR)])

add("Complex: interleaved emphasis",
    font_tag("Both ", REGULAR) + font_tag("width", ITALIC) + font_tag(" and ", REGULAR)
    + font_tag("height", ITALIC) + font_tag(" must be ", REGULAR)
    + font_tag("positive", BOLD) + font_tag(".", REGULAR),
    [make_span("Both ", REGULAR), make_span("width", ITALIC),
     make_span(" and ", REGULAR), make_span("height", ITALIC),
     make_span(" must be ", REGULAR), make_span("positive", BOLD),
     make_span(".", REGULAR)])

add("Complex: three emphasis types",
    font_tag("Results were ", REGULAR) + font_tag("significant", ITALIC)
    + font_tag(" (", REGULAR) + font_tag("p", SERIF_ITALIC)
    + font_tag(" < 0.05), ", REGULAR) + font_tag("robust", BOLD)
    + font_tag(", and ", REGULAR) + font_tag("reproducible", BOLD_ITALIC)
    + font_tag(".", REGULAR),
    [make_span("Results were ", REGULAR), make_span("significant", ITALIC),
     make_span(" (", REGULAR), make_span("p", SERIF_ITALIC),
     make_span(" < 0.05), ", REGULAR), make_span("robust", BOLD),
     make_span(", and ", REGULAR), make_span("reproducible", BOLD_ITALIC),
     make_span(".", REGULAR)])


# ── Edge cases ───────────────────────────────────────────────────────────

add("Edge: single bold letter",
    font_tag("Press ", REGULAR) + font_tag("Q", BOLD) + font_tag(" to quit.", REGULAR),
    [make_span("Press ", REGULAR), make_span("Q", BOLD), make_span(" to quit.", REGULAR)])

add("Edge: single italic letter",
    font_tag("The coefficient ", REGULAR) + font_tag("a", SERIF_ITALIC)
    + font_tag(" determines the shape.", REGULAR),
    [make_span("The coefficient ", REGULAR), make_span("a", SERIF_ITALIC),
     make_span(" determines the shape.", REGULAR)])

add("Edge: adjacent different fonts no space",
    font_tag("input", MONO) + font_tag("→", REGULAR) + font_tag("output", MONO),
    [make_span("input", MONO), make_span("→", REGULAR), make_span("output", MONO)])

add("Edge: all-bold line",
    font_tag("This entire line is bold.", BOLD),
    [make_span("This entire line is bold.", BOLD)])

add("Edge: all-italic line",
    font_tag("This entire line is italic.", ITALIC),
    [make_span("This entire line is italic.", ITALIC)])

add("Edge: all-mono line",
    font_tag("fn main() { println!(\"hello\"); }", MONO),
    [make_span("fn main() { println!(\"hello\"); }", MONO)])

add("Edge: font change at punctuation",
    font_tag("The method ", REGULAR) + font_tag("returns", ITALIC)
    + font_tag(", but not always.", REGULAR),
    [make_span("The method ", REGULAR), make_span("returns", ITALIC),
     make_span(", but not always.", REGULAR)])

add("Edge: parenthetical italic",
    font_tag("The algorithm (", REGULAR) + font_tag("see appendix", ITALIC)
    + font_tag(") runs in linear time.", REGULAR),
    [make_span("The algorithm (", REGULAR), make_span("see appendix", ITALIC),
     make_span(") runs in linear time.", REGULAR)])

add("Edge: quoted bold",
    font_tag('The so-called "', REGULAR) + font_tag('best practice', BOLD)
    + font_tag('" is often debated.', REGULAR),
    [make_span('The so-called "', REGULAR), make_span('best practice', BOLD),
     make_span('" is often debated.', REGULAR)])


# ── Full paragraph with natural mixing ───────────────────────────────────

add("Paragraph: natural academic mixing",
    font_tag("In this paper, we introduce ", REGULAR)
    + font_tag("unscan", MONO)
    + font_tag(", a tool that converts scanned PDFs back to ", REGULAR)
    + font_tag("vector", ITALIC)
    + font_tag(" format. Our approach uses a ", REGULAR)
    + font_tag("per-character feature index", BOLD)
    + font_tag(" with ", REGULAR)
    + font_tag("k", SERIF_ITALIC)
    + font_tag("-d tree lookups for ", REGULAR)
    + font_tag("O", SERIF_ITALIC)
    + font_tag("(log ", REGULAR)
    + font_tag("n", SERIF_ITALIC)
    + font_tag(") candidate retrieval.", REGULAR),
    [make_span("In this paper, we introduce ", REGULAR),
     make_span("unscan", MONO),
     make_span(", a tool that converts scanned PDFs back to ", REGULAR),
     make_span("vector", ITALIC),
     make_span(" format. Our approach uses a ", REGULAR),
     make_span("per-character feature index", BOLD),
     make_span(" with ", REGULAR),
     make_span("k", SERIF_ITALIC),
     make_span("-d tree lookups for ", REGULAR),
     make_span("O", SERIF_ITALIC),
     make_span("(log ", REGULAR),
     make_span("n", SERIF_ITALIC),
     make_span(") candidate retrieval.", REGULAR)])


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
FONT_SIZE = 11
LEADING = 16

def build_pdf():
    pdf_path = OUT_DIR / "mixed-font-specimen.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    base_style = ParagraphStyle(
        "MixedBase",
        fontName=REGULAR,
        fontSize=FONT_SIZE,
        leading=LEADING,
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    category_style = ParagraphStyle(
        "Category",
        fontName=BOLD,
        fontSize=9,
        leading=12,
        textColor="#666666",
        spaceBefore=12,
        spaceAfter=2,
    )

    title_style = ParagraphStyle(
        "Title",
        fontName=BOLD,
        fontSize=16,
        leading=20,
        spaceBefore=0,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        fontName=REGULAR,
        fontSize=10,
        leading=14,
        textColor="#888888",
        spaceAfter=20,
    )

    story = []

    # Title
    story.append(Paragraph("Mixed-Font Specimen", title_style))
    story.append(Paragraph(
        f"{len(TEST_CASES)} test cases for recovering font changes within lines and words",
        subtitle_style
    ))

    current_category = None
    for category, markup, spans in TEST_CASES:
        cat_prefix = category.split(":")[0]
        if cat_prefix != current_category:
            current_category = cat_prefix
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"— {cat_prefix} —", category_style))

        story.append(Paragraph(markup, base_style))

    doc.build(story)
    print(f"Generated {pdf_path} with {len(TEST_CASES)} test lines")
    return pdf_path


def build_ground_truth():
    gt = []
    for category, markup, spans in TEST_CASES:
        # Reconstruct the plain text from spans
        plain_text = "".join(s["text"] for s in spans)
        gt.append({
            "category": category,
            "line": plain_text,
            "spans": spans,
        })

    gt_path = OUT_DIR / "mixed-font-ground-truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"Generated {gt_path} with {len(gt)} entries")
    return gt_path


def rasterize(pdf_path):
    """Rasterize at 300 DPI for unscan testing."""
    base = pdf_path.stem
    raster_pdf = OUT_DIR / f"{base}-raster.pdf"

    # First convert to PNGs
    png_base = str(OUT_DIR / f"{base}-page")
    subprocess.run([
        "pdftoppm", "-r", "300", "-png",
        str(pdf_path), png_base
    ], check=True)

    # Find generated PNGs
    pngs = sorted(OUT_DIR.glob(f"{base}-page-*.png"))
    if not pngs:
        print("ERROR: no PNGs generated", file=sys.stderr)
        return

    print(f"Rasterized {len(pngs)} pages at 300 DPI")

    # Also create a single-file raster PDF using img2pdf or reportlab
    try:
        from PIL import Image as PILImage
        from reportlab.lib.utils import ImageReader

        # Build a raster PDF from the PNGs
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(raster_pdf))
        for png in pngs:
            img = PILImage.open(png)
            w_px, h_px = img.size
            # 300 DPI → points: px * 72 / 300
            w_pt = w_px * 72.0 / 300.0
            h_pt = h_px * 72.0 / 300.0
            c.setPageSize((w_pt, h_pt))
            c.drawImage(str(png), 0, 0, w_pt, h_pt)
            c.showPage()
        c.save()
        print(f"Generated raster PDF: {raster_pdf}")
    except Exception as e:
        print(f"WARNING: could not build raster PDF: {e}", file=sys.stderr)

    return pngs


if __name__ == "__main__":
    pdf_path = build_pdf()
    build_ground_truth()
    pngs = rasterize(pdf_path)
