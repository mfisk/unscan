#!/usr/bin/env python3
"""
Generate a ligature test PDF using fpdf2 with HarfBuzz text shaping.

Renders text lines with ligatures enabled (liga+dlig) and the same text
without ligatures, using fonts known to have ligature substitutions.
The output is a vector PDF with real shaped ligature glyphs.

Output:
  ligature-test.pdf           — vector PDF with OT-shaped ligatures
  ligature-test-fontmap.json  — font name → file path map
"""

import json
import subprocess
import sys
from pathlib import Path

import uharfbuzz as hb
from fpdf import FPDF

SCRIPT_DIR = Path(__file__).resolve().parent

# Words/sentences with ligature sites
LIGATURE_WORDS = (
    "Duffner  offline  efficient  affluent  definition  "
    "difficult  shuffle  waffles  fifteen  flying  coffee  different"
)
LIGATURE_SENTENCE = (
    "The efficient office staff fixed fifteen difficult files "
    "before the official shuffle."
)


def fc_find(family, style="Regular"):
    r = subprocess.run(
        ["fc-list", f"{family}:style={style}", "--format=%{file}\n"],
        capture_output=True, text=True,
    )
    candidates = [l for l in r.stdout.strip().split("\n") if l.strip()]
    ttf = [c for c in candidates if c.lower().endswith(".ttf")]
    return ttf[0] if ttf else (candidates[0] if candidates else None)


def check_ligatures(font_path, text):
    """Check if shaping the text produces fewer glyphs (= ligatures fired)."""
    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    font = hb.Font(face)

    buf_with = hb.Buffer()
    buf_with.add_str(text)
    buf_with.guess_segment_properties()
    hb.shape(font, buf_with, {"liga": True, "dlig": True})

    buf_without = hb.Buffer()
    buf_without.add_str(text)
    buf_without.guess_segment_properties()
    hb.shape(font, buf_without, {"liga": False, "dlig": False})

    n_with = len(buf_with.glyph_infos)
    n_without = len(buf_without.glyph_infos)
    return n_without - n_with  # number of glyphs saved by ligatures


class LigaturePDF(FPDF):
    """FPDF subclass that can toggle text shaping features."""

    def draw_label(self, text):
        self.set_font("Helvetica", size=10)
        self.set_text_color(100, 100, 100)
        self.cell(text=text, new_x="LMARGIN", new_y="NEXT", h=6)
        self.ln(2)

    def draw_heading(self, text):
        self.set_font("Helvetica", "B", size=11)
        self.set_text_color(0, 0, 0)
        self.cell(text=text, new_x="LMARGIN", new_y="NEXT", h=7)
        self.ln(3)

    def draw_text(self, font_name, text, shaping=True):
        self.set_font(font_name, size=14)
        self.set_text_color(0, 0, 0)
        self.set_text_shaping(shaping)
        self.cell(text=text, new_x="LMARGIN", new_y="NEXT", h=8)
        self.ln(2)


def main():
    test_fonts = []
    for family, style in [
        ("EB Garamond", "Regular"),
        ("Libre Caslon Text", "Regular"),
        ("Noto Serif", "Regular"),
    ]:
        path = fc_find(family, style)
        if path:
            test_fonts.append((family, path))
            print(f"  Found: {family} -> {path}")
            # Report ligature counts
            for word in ["efficient", "Duffner", "flying", "coffee"]:
                n = check_ligatures(path, word)
                if n > 0:
                    print(f"    '{word}': {n} ligature substitution(s)")
        else:
            print(f"  SKIP: {family} not found")

    if not test_fonts:
        print("ERROR: no fonts found")
        sys.exit(1)

    # ── Build PDF ────────────────────────────────────────────────────
    out_pdf = SCRIPT_DIR / "ligature-test.pdf"
    fontmap = {}

    pdf = LigaturePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for family, path in test_fonts:
        # Register font — fpdf2 uses a single name per font file
        rl_name = family.replace(" ", "")
        try:
            pdf.add_font(rl_name, "", path)
        except Exception as e:
            print(f"  WARN: can't register {family}: {e}")
            continue
        fontmap[rl_name] = path

        pdf.draw_heading(f"Font: {family}")

        # ── With ligatures (shaped) ───────────────────────────────
        pdf.draw_label("With ligatures (liga+dlig):")
        pdf.draw_text(rl_name, LIGATURE_WORDS, shaping=True)
        pdf.draw_text(rl_name, LIGATURE_SENTENCE, shaping=True)

        # ── Without ligatures (unshaped) ──────────────────────────
        pdf.draw_label("Without ligatures:")
        pdf.draw_text(rl_name, LIGATURE_WORDS, shaping=False)
        pdf.draw_text(rl_name, LIGATURE_SENTENCE, shaping=False)

        pdf.ln(8)

    pdf.output(str(out_pdf))
    print(f"\nWrote: {out_pdf}")

    # ── Write fontmap ────────────────────────────────────────────────
    # Map PDF font names to the actual font files the system rendered with.
    # For built-in PDF fonts (Helvetica, etc.), resolve via fc-match.
    builtin_fonts = {
        "Helvetica": "Helvetica",
        "Helvetica-Bold": "Helvetica:style=Bold",
    }
    for pdf_name, fc_query in builtin_fonts.items():
        r = subprocess.run(
            ["fc-match", fc_query, "--format=%{file}"],
            capture_output=True, text=True,
        )
        if r.stdout.strip():
            fontmap[pdf_name] = r.stdout.strip()
            print(f"  Built-in: {pdf_name} -> {r.stdout.strip()}")

    fontmap_path = SCRIPT_DIR / "ligature-test-fontmap.json"
    with open(fontmap_path, "w") as f:
        json.dump(fontmap, f, indent=2)
    print(f"Wrote: {fontmap_path}")


if __name__ == "__main__":
    main()
