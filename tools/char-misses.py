#!/usr/bin/env python3
"""
char-misses.py — Visual miss report for unscan font identification.

Uses the original vector PDF as ground truth: extracts every text span
with its font and bbox, then spatially matches against unscan's audit log
to find genuine misses. No JSON metadata involved.

Usage:
    # 1. Run unscan with --audit:
    ./target/debug/unscan RASTERIZED.pdf \
        -o /dev/null --audit /tmp/audit-out

    # 2. Generate the report against the vector PDF:
    python3 tools/char-misses.py /tmp/audit-out VECTOR.pdf \
        -o /tmp/misses.html
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, ImageFont, ImageDraw
except ImportError:
    print("ERROR: Pillow required. pip install Pillow", file=sys.stderr)
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF required. pip install pymupdf", file=sys.stderr)
    sys.exit(1)

NORM_H = 48
SCALE = 300.0 / 72.0  # pixels per PDF point (300 DPI)

# Cache for raster PDF page images (page_num → PIL Image)
_raster_page_cache = {}
_raster_doc = None


def get_raster_page_image(raster_pdf_path, page_num):
    """Render a page from the raster PDF as a PIL Image, cached."""
    global _raster_doc
    if raster_pdf_path is None:
        return None
    if _raster_doc is None:
        if not os.path.exists(raster_pdf_path):
            return None
        _raster_doc = fitz.open(raster_pdf_path)
    if page_num not in _raster_page_cache:
        if page_num < 1 or page_num > len(_raster_doc):
            return None
        page = _raster_doc[page_num - 1]
        # Render at 300 DPI (same as unscan's rasterization)
        mat = fitz.Matrix(300.0 / 72.0, 300.0 / 72.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        _raster_page_cache[page_num] = img
    return _raster_page_cache[page_num]


def render_scan_line_with_word_boxes(raster_pdf_path, entry, diag_seg_root=None):
    """Crop the scanned line from the raster PDF and overlay word bounding boxes
    and character segmentation paths.

    The image is upscaled for crisp labels.  Draws:
    - Raw Tesseract word bboxes as dotted orange outlines
    - Final post-processed word bboxes as dashed cyan outlines
    - VP splits as thin blue vertical lines with column labels
    - Seam paths as magenta diagonal paths with column labels
    - A pixel-scale ruler at the top of each word box

    Returns a base64 data URI string for the image, or None if unavailable.
    """
    bbox = entry.get("bbox")
    word_bboxes = entry.get("word_bboxes", [])
    word_bboxes_raw = entry.get("word_bboxes_raw", [])
    if not bbox:
        return None

    page_img = get_raster_page_image(raster_pdf_path, entry["page"])
    if page_img is None:
        return None

    # Crop to the union of word bboxes (both raw and final) with minimal
    # padding — just enough for the ruler scales above and column labels below.
    pad = 4
    pad_bottom = 4

    all_word_boxes = list(word_bboxes) + list(word_bboxes_raw)
    if all_word_boxes:
        wx_left = min(wb["x"] for wb in all_word_boxes)
        wx_right = max(wb["x"] + wb["width"] for wb in all_word_boxes)
        wy_top = min(wb["y"] for wb in all_word_boxes)
        wy_bot = max(wb["y"] + wb["height"] for wb in all_word_boxes)
        lx = max(0, wx_left - pad)
        lr = min(page_img.width, wx_right + pad)
        ly = max(0, wy_top - pad)
        lb = min(page_img.height, wy_bot + pad_bottom)
    else:
        lx = max(0, bbox["x"] - pad)
        lr = min(page_img.width, bbox["x"] + bbox["width"] + pad)
        ly = max(0, bbox["y"] - pad)
        lb = min(page_img.height, bbox["y"] + bbox["height"] + pad_bottom)

    crop = page_img.crop((lx, ly, lr, lb)).convert("RGBA")
    cw, ch = crop.size

    # Upscale for crisp labels — 3× gives readable text at small font sizes
    SCALE = 3
    big = crop.resize((cw * SCALE, ch * SCALE), Image.NEAREST)

    # Add margin at top for the ruler and bottom for column labels
    margin_top = 18 * SCALE
    margin_bot = 14 * SCALE
    canvas = Image.new("RGBA", (big.width, big.height + margin_top + margin_bot), (255, 255, 255, 0))
    canvas.paste(big, (0, margin_top))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        label_font = ImageFont.load_default(size=10)
    except TypeError:
        label_font = ImageFont.load_default()
    try:
        ruler_font = ImageFont.load_default(size=9)
    except TypeError:
        ruler_font = ImageFont.load_default()

    # Raw Tesseract boxes — dotted orange
    raw_color = (255, 160, 0, 200)
    for wb in word_bboxes_raw:
        wx = (wb["x"] - lx) * SCALE
        wy = (wb["y"] - ly) * SCALE + margin_top
        wr = wx + wb["width"] * SCALE
        wbot = wy + wb["height"] * SCALE
        _draw_patterned_rect(draw, wx, wy, wr, wbot, raw_color, dash=2, gap=4, width=1)

    # Final post-processed boxes — dashed cyan
    final_color = (0, 200, 220, 220)
    for wb in word_bboxes:
        wx = (wb["x"] - lx) * SCALE
        wy = (wb["y"] - ly) * SCALE + margin_top
        wr = wx + wb["width"] * SCALE
        wbot = wy + wb["height"] * SCALE
        _draw_patterned_rect(draw, wx, wy, wr, wbot, final_color, dash=6, gap=3, width=1)

    # Draw pixel-scale ruler at top of each final word box
    for wb in word_bboxes:
        wx = (wb["x"] - lx) * SCALE
        wy = (wb["y"] - ly) * SCALE + margin_top
        wb_w = wb["width"]

        # Column numbers every 10, ticks every 5
        for col in range(0, wb_w + 1, 5):
            sx = wx + col * SCALE + SCALE // 2
            if col % 10 == 0:
                draw.line([(sx, wy - 6), (sx, wy)], fill=(140, 140, 140, 180), width=1)
                draw.text((sx - 8, wy - 18), str(col), fill=(120, 120, 120, 200),
                          font=ruler_font)
            else:
                draw.line([(sx, wy - 3), (sx, wy)], fill=(180, 180, 180, 140), width=1)

    # Load segmentation data from diag-seg and draw splits inside word boxes
    if diag_seg_root:
        diag_line_dir = find_diag_seg_dir(
            diag_seg_root, entry["page"], entry.get("text", ""),
            line_index=entry.get("line_index"))
        if diag_line_dir and os.path.isdir(diag_line_dir):
            _draw_seg_paths_on_scan(draw, diag_line_dir, word_bboxes, lx, ly,
                                    SCALE, margin_top, label_font)

    result = Image.alpha_composite(canvas, overlay).convert("RGB")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    data = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def _draw_patterned_rect(draw, x0, y0, x1, y1, color, dash=4, gap=3, width=1):
    """Draw a rectangle outline with a dash/gap pattern."""
    for side in [
        ((x0, y0), (x1, y0)),       # top
        ((x1, y0), (x1, y1)),       # right
        ((x1, y1), (x0, y1)),       # bottom
        ((x0, y1), (x0, y0)),       # left
    ]:
        sx, sy = side[0]
        ex, ey = side[1]
        dx = ex - sx
        dy = ey - sy
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        step = dash + gap
        for i in range(0, length, step):
            px0 = sx + dx * i // length
            py0 = sy + dy * i // length
            px1 = sx + dx * min(i + dash, length) // length
            py1 = sy + dy * min(i + dash, length) // length
            draw.line([(px0, py0), (px1, py1)], fill=color, width=width)


def _draw_seg_paths_on_scan(draw, diag_line_dir, word_bboxes, crop_lx, crop_ly,
                            scale, margin_top, label_font):
    """Draw VP splits and seam paths from diag-seg summary.json files
    inside the word bounding boxes on the scan line overlay.
    Labels each split with its column number below the word box."""
    word_dirs = sorted(
        d for d in os.listdir(diag_line_dir)
        if d.startswith("word_") and os.path.isdir(os.path.join(diag_line_dir, d))
    )

    for wd in word_dirs:
        wpath = os.path.join(diag_line_dir, wd)
        # Prefer seg_plain
        data_path = os.path.join(wpath, "seg_plain")
        if not os.path.isdir(data_path):
            data_path = wpath
        summary_path = os.path.join(data_path, "summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        word_text = summary.get("word_text", "")
        img_w = summary.get("image_w", 0)
        img_h = summary.get("image_h", 0)
        if img_w == 0 or img_h == 0:
            continue

        # Find the matching word bbox by text
        matching_wb = None
        for wb in word_bboxes:
            if wb["text"] == word_text:
                matching_wb = wb
                break
        if matching_wb is None:
            continue

        # Word bbox position in the scaled canvas coordinate system
        wx = (matching_wb["x"] - crop_lx) * scale
        wy = (matching_wb["y"] - crop_ly) * scale + margin_top
        wb_w = matching_wb["width"] * scale
        wb_h = matching_wb["height"] * scale

        # Scale factor: seg boundaries are in word image pixels,
        # word image was cropped at the word bbox dimensions
        sx = (matching_wb["width"] / img_w * scale) if img_w else scale
        sy = (matching_wb["height"] / img_h * scale) if img_h else scale

        vp_splits = summary.get("vp_splits", [])
        seam_splits = summary.get("seam_splits", [])
        seam_paths_raw = summary.get("seam_paths", {})

        label_y = wy + wb_h + 2  # just below the word box

        # VP splits — blue vertical lines + column label
        for col in vp_splits:
            cx = wx + int(col * sx)
            draw.line([(cx, wy), (cx, wy + wb_h)],
                      fill=(40, 100, 220, 200), width=1)
            draw.text((cx - 6, label_y), str(col), fill=(40, 100, 220, 240),
                      font=label_font)

        # Seam paths — magenta diagonal paths (one x per row) + column label
        seam_paths = {}
        if isinstance(seam_paths_raw, dict):
            seam_paths = seam_paths_raw
        for col_key, path in seam_paths.items():
            for row_idx in range(len(path)):
                px_x = wx + int(path[row_idx] * sx)
                px_y = wy + int(row_idx * sy)
                # Draw a small rect for visibility at scale
                pad = max(1, scale // 3)
                draw.rectangle(
                    [(px_x - pad, px_y), (px_x + pad, px_y + max(1, scale - 1))],
                    fill=(255, 0, 200, 200))
            # Label with the nominal column
            col_val = col_key if isinstance(col_key, int) else col_key
            cx = wx + int(int(col_val) * sx) if str(col_val).isdigit() else wx
            draw.text((cx - 6, label_y), str(col_val), fill=(255, 0, 200, 240),
                      font=label_font)

        # Seam splits that don't have paths — fall back to vertical lines + label
        seam_cols_with_paths = set(seam_paths.keys()) | set(str(c) for c in seam_paths.keys())
        for col in seam_splits:
            if str(col) not in seam_cols_with_paths and col not in seam_paths:
                cx = wx + int(col * sx)
                draw.line([(cx, wy), (cx, wy + wb_h)],
                          fill=(255, 0, 200, 180), width=1)
                draw.text((cx - 6, label_y), str(col), fill=(255, 0, 200, 240),
                          font=label_font)

# ---------------------------------------------------------------------------
# Font alias / clone map
# ---------------------------------------------------------------------------
FONT_ALIASES = {
    "arial": "helvetica", "arialmt": "helvetica",
    "nimbussans": "helvetica", "helvetica": "helvetica",
    "freesans": "helvetica",
    "texgyreheros": "helvetica", "texgyreheroscn": "helvetica",
    "timesnewroman": "times", "timesnewromanps": "times",
    "timesroman": "times", "nimbusroman": "times",
    "tinos": "times", "freeserif": "times",
    "texgyretermes": "times",
    "freeserifitalic": "times", "freeserifbold": "times",
    "freeserifbolditalic": "times",
    "p052": "times", "c059": "times",
    "couriernew": "courier", "couriernewps": "courier",
    "nimbusmonops": "courier", "freemono": "courier",
    "texgyrecursor": "courier",
    "carlito": "calibri", "caladea": "cambria",
    "sourcesanspro": "sourcesans", "sourcesans3": "sourcesans",
    "sourcesans": "sourcesans",
    "sourceserif4": "sourceserif", "sourceserif4subhead": "sourceserif",
    "sourceserif4smtext": "sourceserif", "sourceserif4caption": "sourceserif",
    "sourceserif4display": "sourceserif",
    "prestigeelite": "prestigeelite", "prestigeelitestd": "prestigeelite",
    "prestigeelitenormal": "prestigeelite",
}


def clean(name):
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def normalize_font_name(name):
    """Strip path, extension, OT variant suffixes, weight/style, clean."""
    n = name.rsplit("/", 1)[-1]
    for ext in (".ttf", ".otf", ".TTF", ".OTF"):
        if n.endswith(ext):
            n = n[:-len(ext)]
    if "|" in n:
        n = n[:n.index("|")]
    if "[" in n:
        n = n[:n.index("[")]
    return clean(n)


def base_family(name):
    n = re.sub(r'[^a-z0-9]', '', name.lower())
    if "+" in n:
        n = n.split("+", 1)[1]
    changed = True
    while changed:
        changed = False
        for suffix in ["mt", "ps",
                       "bolditalic", "semibolditalic", "mediumitalic",
                       "lightitalic", "thinitalic",
                       "bold", "italic", "oblique",
                       "regular", "medium", "light", "thin",
                       "semibold", "extrabold", "demibold",
                       "condensed", "semicondensed", "expanded",
                       "book", "heavy", "black", "demi",
                       "roman", "normal",
                       "display", "caption", "subhead", "smtext",
                       "400", "400i", "500", "600", "700", "800"]:
            if n.endswith(suffix) and len(n) > len(suffix):
                n = n[:-len(suffix)]
                changed = True
                break
    return n


def canon(name):
    n = re.sub(r'[^a-z0-9]', '', name.lower())
    if n in FONT_ALIASES:
        return FONT_ALIASES[n]
    bf = base_family(name)
    if bf in FONT_ALIASES:
        return FONT_ALIASES[bf]
    best_key, best_len = None, 0
    for ak in FONT_ALIASES:
        if bf.startswith(ak) and len(ak) > best_len:
            best_key, best_len = ak, len(ak)
        elif ak.startswith(bf) and len(bf) > best_len:
            best_key, best_len = ak, len(bf)
    if best_key:
        return FONT_ALIASES[best_key]
    return bf


def fonts_match(a, b):
    """Strict font match for hit/miss classification."""
    na = re.sub(r'[^a-z0-9]', '', a.lower())
    nb = re.sub(r'[^a-z0-9]', '', b.lower())
    if na == nb:
        return True
    ba, bb = base_family(a), base_family(b)
    if ba == bb:
        return True
    if ba in bb or bb in ba:
        return True
    # Also compare with spaces stripped (e.g. "eb garamond" vs "ebgaramond")
    sa, sb = ba.replace(' ', ''), bb.replace(' ', '')
    if sa == sb:
        return True
    if sa and sb and (sa in sb or sb in sa):
        return True
    return canon(a) == canon(b)


def fonts_match_broad(a, b):
    """Broader font match for CI candidate lookup — also handles path-based
    font keys and alias resolution through normalize_font_name."""
    if fonts_match(a, b):
        return True
    # Compare via normalize (handles path-based font keys)
    nna, nnb = normalize_font_name(a), normalize_font_name(b)
    if nna and nnb and (nna == nnb or nna in nnb or nnb in nna):
        return True
    # Compare via canon on normalized names (path key → stem → alias)
    ca, cb = canon(a), canon(b)
    cna = canon(nna) if nna else ca
    cnb = canon(nnb) if nnb else cb
    if cna == cnb:
        return True
    return False


# ---------------------------------------------------------------------------
# Vector PDF ground truth — spatial extraction
# ---------------------------------------------------------------------------

def extract_vector_spans(doc):
    """Extract every text span from the vector PDF with font and bbox."""
    page_spans = {}
    for pi in range(len(doc)):
        page = doc[pi]
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    spans.append((x0, y0, x1, y1, span["font"], span["text"]))
        page_spans[pi + 1] = spans
    return page_spans


def lookup_actual_font(page_spans, page, bbox_px, text=None):
    """Find the dominant font in the vector PDF at the given pixel bbox.
    
    When bbox overlap is ambiguous or empty, falls back to text-content
    matching against nearby spans (OCR bboxes can be off by a few points).
    """
    px0 = bbox_px["x"] / SCALE
    py0 = bbox_px["y"] / SCALE
    px1 = (bbox_px["x"] + bbox_px["width"]) / SCALE
    py1 = (bbox_px["y"] + bbox_px["height"]) / SCALE

    # Weight by overlap area, not text length — avoids a long adjacent-line
    # span dominating when the audit bbox slightly overshoots vertically.
    font_area = Counter()
    for sx0, sy0, sx1, sy1, sfont, stext in page_spans.get(page, []):
        ox0 = max(sx0, px0)
        oy0 = max(sy0, py0)
        ox1 = min(sx1, px1)
        oy1 = min(sy1, py1)
        if ox0 < ox1 and oy0 < oy1:
            font_area[sfont] += (ox1 - ox0) * (oy1 - oy0)

    # If bbox overlap found a clear winner, use it — it's the most reliable
    # signal since it's position-based, not content-based.
    if font_area:
        return font_area.most_common(1)[0][0]

    # Text-content fallback: only when bbox overlap found nothing (OCR bboxes
    # can be off by a few points).  Use Euclidean distance from bbox center
    # to span center so column layouts don't confuse the match.
    if text and len(text) >= 5:
        audit_prefix = re.sub(r'\s+', '', text[:20].lower())
        bbox_cx = (px0 + px1) / 2
        bbox_cy = (py0 + py1) / 2
        best_text_font = None
        best_text_dist = 999
        for sx0, sy0, sx1, sy1, sfont, stext in page_spans.get(page, []):
            span_prefix = re.sub(r'\s+', '', stext[:20].lower())
            if audit_prefix[:10] == span_prefix[:10]:
                span_cx = (sx0 + sx1) / 2
                span_cy = (sy0 + sy1) / 2
                dist = ((bbox_cx - span_cx)**2 + (bbox_cy - span_cy)**2) ** 0.5
                if dist < best_text_dist:
                    best_text_dist = dist
                    best_text_font = sfont
        if best_text_font:
            return best_text_font

    return None


# ---------------------------------------------------------------------------
# Font file resolution
# ---------------------------------------------------------------------------

def find_font_file(font_name):
    """Resolve a font name to a .ttf/.otf path on disk."""
    if not font_name:
        return None
    # Strip OT variant suffix [smcp], [lnum], etc. before matching filenames
    n = font_name
    if "[" in n:
        n = n[:n.index("[")]
    fn = clean(n)
    bf = base_family(n)  # stripped of weight/style suffixes (MT, Bold, etc.)
    # Detect requested weight/style from the input name
    nl = n.lower()
    want_bold = "bold" in nl or "bd" in nl
    want_italic = "italic" in nl or "oblique" in nl or nl.endswith("it")
    want_black = "black" in nl and "bold" not in nl
    candidates = []
    for fontdir in ["/usr/share/fonts/truetype", "/usr/share/fonts/opentype",
                    "/usr/share/fonts"]:
        if not os.path.isdir(fontdir):
            continue
        for root, _, files in os.walk(fontdir):
            for f in files:
                if not f.endswith((".ttf", ".otf")):
                    continue
                cf = clean(f)
                cf_base = base_family(f)
                # Match on full cleaned name OR base family
                if fn in cf or bf in cf or bf == cf_base:
                    candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    # Prefer files matching the requested weight/style
    def score(path):
        cf = clean(os.path.basename(path))
        cf_base = base_family(os.path.basename(path))
        family_match = (cf_base == bf)
        # Weight/style match
        has_bold = "bold" in cf or "bd" in cf
        has_italic = "italic" in cf or "oblique" in cf
        has_black = "black" in cf
        style_match = (has_bold == want_bold and has_italic == want_italic
                       and has_black == want_black)
        return (family_match, style_match)
    candidates.sort(key=score, reverse=True)
    return candidates[0]


def find_font_file_by_key(font_key):
    """Resolve a full font_key path to a file on disk."""
    if not font_key:
        return None
    base = font_key.split("|")[0]
    if os.path.exists(base):
        return base
    if os.path.exists(font_key):
        return font_key
    return None


def resolve_font_from_map(font_name, font_map):
    """Resolve a font name to a file path using the ground-truth font map.

    The map is keyed by PostScript names extracted from the vector PDF (e.g.
    'PlayfairDisplay-Regular', 'EBGaramond-Bold', 'SourceSerif4-It').
    PyMuPDF span font names use the same PostScript names, so direct match
    is the common case.  Prefix match handles minor naming discrepancies.
    """
    if not font_map or not font_name:
        return None
    target = clean(font_name)
    # Direct key match (common case with PS-name-keyed fontmap)
    for key, path in font_map.items():
        if clean(key) == target:
            return path
    # Prefix match: target starts with key or vice versa
    best = None
    best_len = 0
    for key, path in font_map.items():
        ck = clean(key)
        if target.startswith(ck) or ck.startswith(target):
            if len(ck) > best_len:
                best = path
                best_len = len(ck)
    return best


def find_correct_ci_candidate(entry, actual_font):
    """Find the CI candidate that matches the actual (vector PDF) font.

    Returns (font_key, score, rank) or (None, None, None).
    """
    if not actual_font:
        return None, None, None
    for i, c in enumerate(entry.get("ci_candidates", [])):
        if fonts_match_broad(c["font_key"], actual_font):
            return c["font_key"], c["score"], i + 1
    return None, None, None


# ---------------------------------------------------------------------------
# Image rendering — ab_glyph only via unscan --render-ref-chars
# ---------------------------------------------------------------------------

import subprocess
import tempfile

# Cache: font_path -> {char -> PIL.Image}
_unscan_ref_cache: dict[str, dict[str, "Image.Image"]] = {}

def _find_unscan_binary():
    """Find the unscan binary, preferring debug build."""
    script_dir = Path(__file__).resolve().parent.parent
    for candidate in [
        script_dir / "target" / "debug" / "unscan",
        script_dir / "target" / "release" / "unscan",
    ]:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None

def render_ref_chars_unscan(font_path: str, chars: set[str]) -> dict[str, "Image.Image"]:
    """Render characters using unscan's ab_glyph-based render_char_normalised().

    Returns a dict mapping each character to its PIL Image (grayscale, NORM_H tall).
    Results are cached per font_path so subsequent calls are instant.
    """
    font_path = str(font_path)
    if font_path not in _unscan_ref_cache:
        _unscan_ref_cache[font_path] = {}

    # Find chars we haven't rendered yet
    needed = chars - set(_unscan_ref_cache[font_path].keys())
    if not needed:
        return _unscan_ref_cache[font_path]

    unscan_bin = _find_unscan_binary()
    if not unscan_bin:
        print("WARNING: unscan binary not found, ref chars unavailable", file=sys.stderr)
        return _unscan_ref_cache[font_path]

    with tempfile.TemporaryDirectory(prefix="unscan-ref-") as tmpdir:
        req = json.dumps({
            "font": font_path,
            "chars": "".join(sorted(needed)),
            "output_dir": tmpdir,
        })
        try:
            subprocess.run(
                [unscan_bin, "--render-ref-chars", req],
                capture_output=True, timeout=30,
            )
        except Exception:
            return _unscan_ref_cache[font_path]

        # Load rendered PNGs: U+XXXX.png
        for c in needed:
            fname = f"U+{ord(c):04X}.png"
            fpath = os.path.join(tmpdir, fname)
            if os.path.exists(fpath):
                _unscan_ref_cache[font_path][c] = Image.open(fpath).convert("L").copy()

    return _unscan_ref_cache[font_path]


def img_to_b64(img_or_path):
    if isinstance(img_or_path, (str, Path)):
        with open(img_or_path, "rb") as f:
            data = f.read()
    else:
        buf = io.BytesIO()
        img_or_path.save(buf, format="PNG")
        data = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


# ---------------------------------------------------------------------------
# Segmentation visualisation
# ---------------------------------------------------------------------------

def find_diag_seg_dir(diag_seg_root, page, text, line_index=None):
    """Find the diag-seg line directory matching a page and line text."""
    if not diag_seg_root or not os.path.isdir(diag_seg_root):
        return None
    prefix = f"p{page}_"
    # Sanitise text the same way seg_diag.rs does (replace non-alnum with _)
    slug = re.sub(r'[^A-Za-z0-9]', '_', text)[:30]

    # New format: p{page}_L{line_index:03}_{slug} — exact match by line_index
    if line_index is not None:
        exact_prefix = f"p{page}_L{line_index:03d}_"
        for d in os.listdir(diag_seg_root):
            if d.startswith(exact_prefix):
                return os.path.join(diag_seg_root, d)

    # Legacy format: p{page}_{slug} — match by text slug, verify via line_summary.json
    candidates = []
    for d in os.listdir(diag_seg_root):
        if not d.startswith(prefix):
            continue
        dsuf = d[len(prefix):]
        if slug and slug in dsuf:
            candidates.append(os.path.join(diag_seg_root, d))

    # If we have a line_index, verify via line_summary.json
    if line_index is not None and candidates:
        for cpath in candidates:
            summary_path = os.path.join(cpath, "line_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path) as sf:
                        summary = json.load(sf)
                    if summary.get("line_index") == line_index:
                        return cpath
                except (json.JSONDecodeError, KeyError):
                    pass
        return None

    best = candidates[0] if candidates else None
    if best is None:
        # Fallback: try matching by line number embedded in dir name
        for d in sorted(os.listdir(diag_seg_root)):
            if d.startswith(prefix):
                dsuf = d[len(prefix):]
                # Accept if first word of text appears in dir name
                first_word = re.sub(r'[^A-Za-z0-9]', '', text.split()[0]) if text.strip() else ""
                if first_word and first_word in dsuf:
                    best = os.path.join(diag_seg_root, d)
                    break
    return best

def img_td(img_or_path, fallback="—"):
    if img_or_path is None:
        return fallback
    return f'<img src="{img_to_b64(img_or_path)}" class="ci">'


# ---------------------------------------------------------------------------
# Character selection
# ---------------------------------------------------------------------------

def pick_interesting_chars(chars, n_worst=4, n_normal=2):
    corrected = [(i, c) for i, c in enumerate(chars) if c.get("ocr_corrected_from")]
    by_dist = sorted(enumerate(chars), key=lambda x: x[1]["min_dist_sq"], reverse=True)
    corrected_idxs = {i for i, _ in corrected}
    worst = [(i, c) for i, c in by_dist if i not in corrected_idxs][:n_worst]
    used = corrected_idxs | {i for i, _ in worst}
    normal = [(i, c) for i, c in by_dist if i not in used and c["min_dist_sq"] < 0.008]
    normal = normal[-n_normal:]
    result = corrected + worst + normal
    result.sort(key=lambda x: x[0])
    return result


# ---------------------------------------------------------------------------
# Crop directory matching
# ---------------------------------------------------------------------------

def find_crop_dir(crops_root, page, line_index, diag_seg_root=None, line_text=None):
    """Find crop directory by page and line index.

    Checks the diag-seg line directory first (crops/ subdir created
    automatically by --audit).
    """
    # Try diag-seg crops/ subdir first (matched by text slug)
    if diag_seg_root and line_text:
        diag_line = find_diag_seg_dir(diag_seg_root, page, line_text, line_index=line_index)
        if diag_line:
            crop_subdir = os.path.join(diag_line, "crops")
            if os.path.isdir(crop_subdir):
                files = sorted(os.listdir(crop_subdir))
                if files:
                    return crop_subdir, files

    # Fall back to legacy crops dir
    if crops_root and os.path.isdir(crops_root):
        prefix = f"p{page}_L{line_index:03d}_"
        for d in sorted(os.listdir(crops_root)):
            if d.startswith(prefix):
                path = os.path.join(crops_root, d)
                return path, sorted(os.listdir(path))
    return None, []


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def dist_class(d2):
    if d2 > 0.05:
        return "bad"
    elif d2 > 0.01:
        return "warn"
    return "ok"


def build_miss_html(entry, chars_to_show, crop_dir, crop_files,
                    correct_font_path, correct_font_name, ci_rank, ci_score,
                    chosen_font_path, chosen_font_name,
                    diag_seg_root=None, raster_pdf_path=None):
    rows = []
    for idx, cv in chars_to_show:
        ch = cv["ch"]
        ocr_from = cv.get("ocr_corrected_from", "")
        original_ocr = ocr_from if ocr_from else ch
        d2 = cv["min_dist_sq"]

        crop_img = None
        crop_idx = cv.get("crop_index", idx)
        if crop_dir and crop_files:
            prefix = f"crop_{crop_idx:02d}_"
            for cf in crop_files:
                if cf.startswith(prefix):
                    crop_img = os.path.join(crop_dir, cf)
                    break
        ocr_label = f"'{original_ocr}'"
        if ocr_from:
            ocr_label = f"<span class='ocr-fix'>'{ocr_from}' → '{ch}'</span>"

        ref_img = None
        if correct_font_path:
            refs = render_ref_chars_unscan(correct_font_path, {ch})
            ref_img = refs.get(ch)

        # Find the CI distance for this character against the correct font.
        # Prefer fontmap_dists (computed for all fontmap fonts, always present
        # when --include-fontmap is used), fall back to nearest (top 3 only).
        correct_char_dist = None
        if correct_font_path:
            # First: check fontmap_dists by exact path match
            for fk, fd in cv.get("fontmap_dists", []):
                if fk == correct_font_path or fonts_match_broad(fk, correct_font_name):
                    correct_char_dist = fd
                    break
            # Fallback: check nearest (top 3 from global search)
            if correct_char_dist is None:
                for nf, nd in cv.get("nearest", []):
                    if fonts_match(nf, correct_font_name):
                        correct_char_dist = nd
                        break

        # Render chosen (wrong) font's reference via unscan --render-ref-chars.
        chosen_img = None
        if chosen_font_path:
            refs = render_ref_chars_unscan(chosen_font_path, {ch})
            chosen_img = refs.get(ch)
        dc = dist_class(d2)

        # OCR column: OCR char, best alt char + score on separate line
        near = cv.get("nearest", [])
        best_alt_ch = cv.get("best_alt_char")
        best_alt_dist = cv.get("best_alt_dist")

        ocr_parts = [f"OCR: <b>'{original_ocr}'</b>"]

        # Show best-scoring font for the OCR char
        if near:
            ocr_font = near[0][0].rsplit("/", 1)[-1]
            ocr_dist = near[0][1]
            ocr_dc = dist_class(ocr_dist)
            ocr_parts.append(f"<span class='font-mini'>{ocr_font}</span><br><span class='num {ocr_dc}'>{ocr_dist:.6f}</span>")

        # Show best alt char on a separate line (even if correction didn't fire)
        if best_alt_ch and best_alt_dist is not None:
            alt_dc = dist_class(best_alt_dist)
            ocr_parts.append(f"Alt: <b>'{best_alt_ch}'</b> <span class='num {alt_dc}'>{best_alt_dist:.6f}</span>")

        ocr_cell = "<br>".join(ocr_parts)

        # Per-char distance for the chosen (unscan pick) font
        chosen_d2 = cv.get("chosen_dist_sq")
        if chosen_d2 is not None:
            chosen_dc = dist_class(chosen_d2)
            chosen_score_label = f"<div class='sub'><span class='num {chosen_dc}'>{chosen_d2:.6f}</span></div>"
        else:
            chosen_score_label = ""

        # Per-char distance label for correct font
        if correct_char_dist is not None:
            cc_dc = dist_class(correct_char_dist)
            correct_score_label = f"<div class='sub'><span class='num {cc_dc}'>{correct_char_dist:.6f}</span></div>"
        else:
            correct_score_label = ""

        rows.append(f"""<tr>
  <td class="img-td">{img_td(crop_img)}</td>
  <td class="img-td">{img_td(ref_img)}{correct_score_label}</td>
  <td class="img-td">{img_td(chosen_img)}{chosen_score_label}</td>
  <td class="ocr-col">{ocr_cell}</td>
</tr>""")

    text_preview = entry["text"][:60]
    matched = entry.get("font_matched") or "?"
    # Normalize display name: when the pick is the same font family as the
    # correct answer (just a different filename variant), show the ground-truth
    # name for both so the report doesn't look like a mismatch.
    if correct_font_name and matched and matched != "?" and fonts_match(matched, correct_font_name):
        matched = correct_font_name
    rank_str = f"CI #{ci_rank}, score {ci_score:.10f}" if ci_rank else "not in CI"

    # SSIM verification info
    ssim_val = entry.get("ssim_score")
    ssim_pass = entry.get("ssim_pass")
    if ssim_val is not None:
        ssim_cls = "ssim-pass" if ssim_pass else "ssim-fail"
        ssim_label = "pass" if ssim_pass else "FAIL"
        ssim_html = f' <span class="{ssim_cls}">SSIM {ssim_val:.10f} ({ssim_label})</span>'
    else:
        ssim_html = ""

    chosen_score = None
    chosen_rank = None
    for i, c in enumerate(entry.get("ci_candidates", []), 1):
        if fonts_match_broad(c["font_key"], matched):
            chosen_score = c["score"]
            chosen_rank = i
            break

    correct_col_hdr = f"{correct_font_name}<br><span class='score'>{rank_str}</span>"
    chosen_rank_str = f"CI #{chosen_rank}, score {chosen_score:.10f}" if chosen_rank and chosen_score else ""
    chosen_col_hdr = f"{matched}<br><span class='score'>{chosen_rank_str}</span>"

    # Gather segmentation stats from diag-seg data for the scan line label
    seg_stats_html = ""
    diag_line_dir = find_diag_seg_dir(diag_seg_root, entry["page"], entry.get("text", ""),
                                       line_index=entry.get("line_index"))
    if diag_line_dir:
        word_dirs = sorted(
            d for d in os.listdir(diag_line_dir)
            if d.startswith("word_") and os.path.isdir(os.path.join(diag_line_dir, d))
        )
        # Build word text → x position map from the entry's word bboxes
        # so we can sort seg stats in left-to-right reading order.
        word_x_map = {}
        for wb in entry.get("word_bboxes", []):
            word_x_map[wb["text"]] = wb["x"]
        seg_parts = []
        for wd in word_dirs:
            wpath = os.path.join(diag_line_dir, wd)
            # Prefer seg_plain subdirectory, fall back to flat layout
            if os.path.isdir(os.path.join(wpath, "seg_plain")):
                data_path = os.path.join(wpath, "seg_plain")
            else:
                data_path = wpath
            summary_path = os.path.join(data_path, "summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            wtext = summary.get("word_text", wd)
            n_exp = summary.get("n_chars_expected", "?")
            n_got = summary.get("n_segments_produced", "?")
            mismatch = summary.get("mismatch", False)
            nvp = len(summary.get("vp_splits", []))
            nseam = len(summary.get("seam_splits", []))
            word_x = word_x_map.get(wtext, 999999)
            info = f'"{wtext}" {n_got}/{n_exp}'
            if mismatch:
                info += " ⚠"
            tags = []
            if nvp: tags.append(f"{nvp} vert")
            if nseam: tags.append(f"{nseam} seam")
            if tags:
                info += f" ({', '.join(tags)})"
            seg_parts.append((word_x, info))
        if seg_parts:
            seg_parts.sort(key=lambda t: t[0])
            seg_stats_html = f'Segmentation: {" | ".join(info for _, info in seg_parts)}'

    # SSIM comparison block: scan crop vs rendered, with diff image
    ssim_compare_html = ""
    if diag_line_dir:
        ssim_scan_path = os.path.join(diag_line_dir, "ssim_scan.png")
        ssim_render_path = os.path.join(diag_line_dir, "ssim_render.png")
        ssim_diff_path = os.path.join(diag_line_dir, "ssim_diff.png")
        if os.path.exists(ssim_scan_path) and os.path.exists(ssim_render_path):
            scan_b64 = img_to_b64(ssim_scan_path)
            render_b64 = img_to_b64(ssim_render_path)
            diff_b64 = img_to_b64(ssim_diff_path) if os.path.exists(ssim_diff_path) else None

            ssim_val_str = f"{ssim_val:.10f}" if ssim_val is not None else "—"
            diff_row = ""
            if diff_b64:
                diff_row = f"""<tr>
  <td class="ssim-label">Diff</td>
  <td colspan="2"><img src="{diff_b64}" class="ssim-compare-img"></td>
</tr>"""

            ssim_compare_html = f"""<div class="ssim-compare-block">
<table class="ssim-compare-table">
<tr>
  <th></th>
  <th>Correct</th>
  <th>Picked (SSIM verified)</th>
</tr>
<tr>
  <td class="ssim-label">Font</td>
  <td class="correct">{correct_font_name}</td>
  <td class="chosen">{matched}</td>
</tr>
<tr>
  <td class="ssim-label">Scan</td>
  <td colspan="2"><img src="{scan_b64}" class="ssim-compare-img"></td>
</tr>
<tr>
  <td class="ssim-label">Render</td>
  <td colspan="2"><img src="{render_b64}" class="ssim-compare-img"></td>
</tr>
{diff_row}
<tr>
  <td class="ssim-label">SSIM</td>
  <td colspan="2">{ssim_val_str}</td>
</tr>
</table>
</div>"""

    # Show alternate (lig) CI candidates when available
    # Scan line image with word boxes
    scan_line_html = ""
    scan_line_b64 = render_scan_line_with_word_boxes(raster_pdf_path, entry, diag_seg_root=diag_seg_root)
    if scan_line_b64:
        scan_line_html = f"""<div class="scan-line-block">
<div class="scan-line-label">Scan line: <span style="color:#ffa000">···</span> raw box · <span style="color:#00c8dc">- -</span> final box · <span style="color:#2864dc">│</span> v-whitespace · <span style="color:#ff00c8">╲</span> seam</div>
<img src="{scan_line_b64}" class="scan-line-img">
{f'<div class="scan-line-label">{seg_stats_html}</div>' if seg_stats_html else ''}
</div>"""

    # Skip the per-character comparison table when the font pick is correct
    # (i.e. SSIM-only failures) — the char table is only useful for real misses.
    font_is_correct = (matched == correct_font_name)
    char_table_html = "" if font_is_correct else f"""<table>
<tr>
  <th>Scan</th>
  <th class="correct">Correct: {correct_col_hdr}</th>
  <th class="chosen">Unscan pick: {chosen_col_hdr}</th>
  <th>OCR</th>
</tr>
{"".join(rows)}
</table>"""

    return f"""<div class="miss">
<h3>p{entry['page']}:L{entry['line_index']} — "{text_preview}"{ssim_html}</h3>
{scan_line_html}
{ssim_compare_html}
{char_table_html}
</div>"""


CSS = """<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
.ssim-pass { font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #d4edda; color: #155724; margin-left: 8px; }
.ssim-fail { font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #f8d7da; color: #721c24; margin-left: 8px; font-weight: bold; }
body {
  font-family: -apple-system, system-ui, sans-serif;
  font-size: 13px;
  color: #222;
  padding: 16px;
}
h2 { font-size: 16px; margin-bottom: 12px; color: #111; }
.summary { color: #555; font-size: 12px; margin-bottom: 8px; }
.score-legend { color: #666; font-size: 11px; margin-bottom: 16px; line-height: 1.6; }
.miss { margin-bottom: 28px; }
.miss h3 { font-size: 13px; margin-bottom: 6px; color: #111; }
table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
th {
  text-align: center; font-size: 11px; font-weight: 600;
  color: #444;
  padding: 4px 6px;
  border-bottom: 2px solid #ccc;
  line-height: 1.4;
}
th.correct { color: #2e7d32; }
th.chosen { color: #c62828; }
th .score { font-weight: 400; font-size: 10px; color: #666; }
td {
  padding: 6px; vertical-align: middle;
  border-bottom: 1px solid #eee;
}
.img-td { text-align: center; }
img.ci {
  height: 48px; image-rendering: pixelated; display: block; margin: 0 auto;
  border: 1px solid #ddd;
  background: #f5f5f5;
}
.sub { font-size: 10px; color: #777; margin-top: 2px; }
.ocr-fix { color: #c62828; }
.num { font-family: monospace; font-size: 12px; text-align: right; white-space: nowrap; }
.num.bad { color: #c62828; font-weight: bold; }
.num.warn { color: #e65100; }
.num.ok { color: #2e7d32; }
.ocr-col { text-align: center; font-size: 11px; vertical-align: middle; padding: 4px; }
.char-label { font-size: 14px; font-weight: 600; }
.font-mini { font-size: 9px; color: #888; word-break: break-all; max-width: 100px; display: inline-block; }
.dimmed { color: #bbb; font-size: 10px; }
.ratio { font-size: 11px; color: #888; }
.ssim-compare-block {
  margin: 8px 0 10px 0; padding: 8px; background: #f5f8ff;
  border: 1px solid #ccd; border-radius: 4px;
}
.ssim-compare-table { border-collapse: collapse; width: 100%; }
.ssim-compare-table th {
  text-align: center; font-size: 11px; font-weight: 600;
  padding: 4px 6px; border-bottom: 2px solid #ccc;
}
.ssim-compare-table td {
  padding: 4px 6px; border-bottom: 1px solid #dde; vertical-align: middle;
}
.ssim-compare-table .ssim-label {
  font-size: 10px; font-weight: 600; color: #555; width: 50px; text-align: right;
}
.ssim-compare-img {
  max-width: 100%; image-rendering: pixelated;
  border: 1px solid #ddd; display: block; margin: 2px 0;
}
.scan-line-block {
  margin: 6px 0 10px 0; padding: 8px; background: #f0f4f0;
  border: 1px solid #c0c8c0; border-radius: 4px; overflow-x: auto;
  box-sizing: border-box;
}
.scan-line-label { font-size: 10px; font-weight: 600; color: #555; margin-bottom: 4px; }
.scan-line-img {
  image-rendering: pixelated;
  border: 1px solid #ddd; display: block; margin: 2px 0;
}
</style>"""


def main():
    parser = argparse.ArgumentParser(description="Visual miss report for unscan")
    parser.add_argument("audit_dir", help="Path to --audit directory (contains audit.json and diag images)")
    parser.add_argument("vector_pdf", help="Path to the original vector PDF")
    parser.add_argument("-o", "--output", default="/tmp/misses.html",
                        help="Output HTML file path")
    parser.add_argument("--fontmap", help="Path to font file map JSON generated by gen-specimen.py")
    args = parser.parse_args()

    # Load font map if provided
    font_map = {}
    if args.fontmap and os.path.exists(args.fontmap):
        with open(args.fontmap) as f:
            font_map = json.load(f)

    # audit_dir contains both audit.json and diag-seg images
    audit_json = os.path.join(args.audit_dir, "audit.json")
    if not os.path.exists(audit_json):
        # Backward compat: if the argument is a JSON file itself, use it
        if args.audit_dir.endswith(".json") and os.path.exists(args.audit_dir):
            audit_json = args.audit_dir
            diag_seg_root = os.path.dirname(audit_json)
        else:
            print(f"ERROR: {audit_json} not found", file=sys.stderr)
            sys.exit(1)
    else:
        diag_seg_root = args.audit_dir

    with open(audit_json) as f:
        audit = json.load(f)

    # Get raster PDF path from audit metadata for scan line rendering
    raster_pdf_path = audit.get("input_file")
    if raster_pdf_path and not os.path.isabs(raster_pdf_path):
        # Try relative to audit dir
        candidate = os.path.join(os.path.dirname(audit_json), raster_pdf_path)
        if os.path.exists(candidate):
            raster_pdf_path = candidate
    if raster_pdf_path and not os.path.exists(raster_pdf_path):
        print(f"WARNING: raster PDF not found: {raster_pdf_path}", file=sys.stderr)
        raster_pdf_path = None

    doc = fitz.open(args.vector_pdf)
    page_spans = extract_vector_spans(doc)

    entries = audit["text_entries"]
    total = len(entries)

    # Classify each audit line against vector PDF ground truth
    misses = []
    ssim_failures = []
    hits = 0
    skipped = 0
    total_chars = 0
    corrected_chars = 0

    unmatched = 0

    for e in entries:
        # Count OCR corrections across all entries
        for cv in e.get("ci_char_votes", []):
            total_chars += 1
            if cv.get("ocr_corrected_from"):
                corrected_chars += 1

        matched = e.get("font_matched") or ""
        bbox = e.get("bbox")
        if not bbox:
            skipped += 1
            continue

        actual_font = lookup_actual_font(page_spans, e["page"], bbox, text=e.get("text"))
        if actual_font is None:
            skipped += 1
            continue

        if not matched:
            # Unmatched line — unscan found no font at all.  Count as a miss.
            unmatched += 1
            misses.append((e, actual_font, None, None, None))
            continue

        if fonts_match(matched, actual_font):
            if e.get("ssim_pass") is False:
                # Font correct but SSIM failed — count as miss
                gt_key, gt_score, gt_rank = find_correct_ci_candidate(e, actual_font)
                ssim_failures.append((e, actual_font, gt_key, gt_score, gt_rank))
            else:
                hits += 1
        else:
            # Find correct font's CI candidate for rendering
            gt_key, gt_score, gt_rank = find_correct_ci_candidate(e, actual_font)
            misses.append((e, actual_font, gt_key, gt_score, gt_rank))

    doc.close()

    all_misses = len(misses) + len(ssim_failures)
    unmatched_str = f" ({unmatched} unmatched)" if unmatched else ""
    ocr_corr_str = f" | OCR corrections: {corrected_chars}/{total_chars}" if total_chars else ""
    ssim_miss_str = f" ({len(ssim_failures)} SSIM)" if ssim_failures else ""
    print(f"Total: {total}  Hits: {hits}  Misses: {all_misses}{ssim_miss_str}{unmatched_str}  Skipped: {skipped}  OCR corrections: {corrected_chars}/{total_chars}",
          file=sys.stderr)

    if not misses and not ssim_failures:
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>unscan char-misses — 100%</title></head>
<body style="background: white; color: #222;">
{CSS}
<h2>unscan Miss Report</h2>
<div class="summary">{hits}/{hits} correct (100.0%) — no misses 🎉{ocr_corr_str}</div>
<div class="score-legend">
<b>Score key:</b>
<b>CI score</b> (per-line) = −mean(log(dist²)) across characters; <b>higher = better match</b>.
<b>CI dist²</b> (per-character) = squared Euclidean distance in normalized feature space between scan crop and rendered glyph; <b>lower = better</b> (good: &lt;1e-4, suspect: &gt;1e-3).
<b>SSIM</b> (per-line) = structural similarity between scanned line and re-render; <b>0–1, higher = more similar</b>.
</div>
</body>
</html>"""
        with open(args.output, "w") as f:
            f.write(html)
        print(f"Report written to {args.output}", file=sys.stderr)
        return

    # Build HTML
    miss_blocks = []
    for entry, actual_font, gt_key, gt_score, gt_rank in misses:
        crop_dir, crop_files = find_crop_dir(None, entry["page"], entry["line_index"],
                                              diag_seg_root=diag_seg_root,
                                              line_text=entry.get("text", ""))

        chars = entry.get("ci_char_votes", [])
        interesting = pick_interesting_chars(chars)

        # Correct font: prefer ground-truth fontmap, then CI candidate, then system search
        correct_font_path = None
        if font_map:
            correct_font_path = resolve_font_from_map(actual_font, font_map)
        if not correct_font_path:
            correct_font_path = find_font_file_by_key(gt_key)
        if not correct_font_path:
            correct_font_path = find_font_file(actual_font)
        correct_font_name = actual_font

        # Chosen (wrong) font
        matched = entry.get("font_matched") or ""
        chosen_font_path = None
        if font_map:
            chosen_font_path = resolve_font_from_map(matched, font_map)
        if not chosen_font_path:
            chosen_font_path = find_font_file(matched)

        block = build_miss_html(
            entry, interesting, crop_dir, crop_files,
            correct_font_path, correct_font_name, gt_rank, gt_score,
            chosen_font_path, matched,
            diag_seg_root=diag_seg_root,
            raster_pdf_path=raster_pdf_path,
        )
        miss_blocks.append(block)

    # Build SSIM failure blocks (correct font but SSIM failed)
    ssim_blocks = []
    for entry, actual_font, gt_key, gt_score, gt_rank in ssim_failures:
        crop_dir, crop_files = find_crop_dir(None, entry["page"], entry["line_index"],
                                              diag_seg_root=diag_seg_root,
                                              line_text=entry.get("text", ""))

        chars = entry.get("ci_char_votes", [])
        interesting = pick_interesting_chars(chars)

        correct_font_path = None
        if font_map:
            correct_font_path = resolve_font_from_map(actual_font, font_map)
        if not correct_font_path:
            correct_font_path = find_font_file_by_key(gt_key)
        if not correct_font_path:
            correct_font_path = find_font_file(actual_font)
        correct_font_name = actual_font

        matched = entry.get("font_matched") or ""
        chosen_font_path = None
        if font_map:
            chosen_font_path = resolve_font_from_map(matched, font_map)
        if not chosen_font_path:
            chosen_font_path = find_font_file(matched)

        block = build_miss_html(
            entry, interesting, crop_dir, crop_files,
            correct_font_path, correct_font_name, gt_rank, gt_score,
            chosen_font_path, matched,
            diag_seg_root=diag_seg_root,
            raster_pdf_path=raster_pdf_path,
        )
        ssim_blocks.append(block)

    unmatched_html = f" | {unmatched} unmatched" if unmatched else ""
    compared = hits + all_misses
    pct = hits / compared * 100 if compared else 0
    ssim_section = ""
    if ssim_blocks:
        ssim_section = f"""<h2 style="margin-top:2em; color:#c55;">SSIM Failures (correct font, SSIM rejected)</h2>
{"".join(ssim_blocks)}"""

    ssim_miss_html = f" ({len(ssim_failures)} SSIM)" if ssim_failures else ""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>unscan char-misses — {hits}/{compared} ({pct:.1f}%)</title>
</head>
<body style="background: white; color: #222;">
{CSS}
<h2>unscan Miss Report</h2>
<div class="summary">{hits}/{compared} correct ({pct:.1f}%) — {all_misses} misses shown below{ssim_miss_html}{unmatched_html}{ocr_corr_str}</div>
<div class="score-legend">
<b>Score key:</b>
<b>CI score</b> (per-line) = −mean(log(dist²)) across characters; <b>higher = better match</b>.
<b>CI dist²</b> (per-character) = squared Euclidean distance in normalized feature space between scan crop and rendered glyph; <b>lower = better</b> (good: &lt;1e-4, suspect: &gt;1e-3).
<b>SSIM</b> (per-line) = structural similarity between scanned line and re-render; <b>0–1, higher = more similar</b>.
</div>
{"".join(miss_blocks)}
{ssim_section}
</body>
</html>"""

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)

    print(f"Report written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
