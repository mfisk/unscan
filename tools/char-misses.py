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
    # Strip OT variant suffix [smcp], [lnum], etc. before matching filenames
    n = font_name
    if "[" in n:
        n = n[:n.index("[")]
    fn = clean(n)
    for fontdir in ["/usr/share/fonts/truetype", "/usr/share/fonts/opentype",
                    "/usr/share/fonts"]:
        if not os.path.isdir(fontdir):
            continue
        for root, _, files in os.walk(fontdir):
            for f in files:
                if f.endswith((".ttf", ".otf")) and fn in clean(f):
                    return os.path.join(root, f)
    return None


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

    The map keys are ReportLab names like 'PlayfairDisplay' or 'PlayfairDisplay-Bold'.
    PyMuPDF font names are like 'PlayfairDisplay-Regular'. We match by cleaned prefix.
    """
    if not font_map or not font_name:
        return None
    target = clean(font_name)
    # Direct key match
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
# Image rendering
# ---------------------------------------------------------------------------

def render_char(char, font_path, height=NORM_H):
    try:
        font = ImageFont.truetype(font_path, height)
    except Exception:
        return None
    bbox = font.getbbox(char)
    if not bbox:
        return None
    w = bbox[2] - bbox[0] + 4
    h = bbox[3] - bbox[1] + 4
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((-bbox[0] + 2, -bbox[1] + 2), char, font=font, fill=0)
    return img


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


def render_seg_picture(diag_line_dir, word_text=None, seg_subdir=None):
    """Render a labelled segmentation image for a word in diag-seg output.

    Returns a PIL Image with the word crop scaled up, split lines drawn in
    colour (red=VP, blue=seam), and column numbers along the
    top.  Returns None if data is missing.

    seg_subdir: if set (e.g. "seg_plain" or "seg_lig"), look for word_crop.png
    and summary.json inside that subdirectory of each word dir.  If None,
    auto-detect: prefer seg_plain/seg_lig structure, fall back to flat layout.
    """
    if not diag_line_dir or not os.path.isdir(diag_line_dir):
        return None

    # Find matching word directory
    word_dirs = sorted(
        d for d in os.listdir(diag_line_dir)
        if d.startswith("word_") and os.path.isdir(os.path.join(diag_line_dir, d))
    )

    results = []
    for wd in word_dirs:
        wpath = os.path.join(diag_line_dir, wd)

        # Resolve the actual data directory (supports seg_plain/seg_lig structure)
        if seg_subdir:
            data_path = os.path.join(wpath, seg_subdir)
        elif os.path.isdir(os.path.join(wpath, "seg_plain")):
            data_path = os.path.join(wpath, "seg_plain")
        else:
            data_path = wpath

        crop_path = os.path.join(data_path, "word_crop.png")
        summary_path = os.path.join(data_path, "summary.json")
        if not os.path.exists(crop_path) or not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        crop = Image.open(crop_path).convert("RGBA")
        w, h = crop.size

        scale = max(3, min(6, 400 // max(w, 1)))
        big = crop.resize((w * scale, h * scale), Image.NEAREST)
        bw, bh = big.size

        margin_top = 28
        margin_bottom = 22
        canvas = Image.new("RGBA", (bw + 2, bh + margin_top + margin_bottom), (255, 255, 255, 255))
        canvas.paste(big, (1, margin_top))
        draw = ImageDraw.Draw(canvas)

        # Column numbers every 10
        for x in range(0, w, 10):
            sx = x * scale + scale // 2 + 1
            draw.line([(sx, margin_top - 4), (sx, margin_top)], fill=(180, 180, 180), width=1)
            draw.text((sx - 6, 1), str(x), fill=(120, 120, 120))

        # Tick marks every 5
        for x in range(5, w, 10):
            sx = x * scale + scale // 2 + 1
            draw.line([(sx, margin_top - 2), (sx, margin_top)], fill=(210, 210, 210), width=1)

        vp_splits = summary.get("vp_splits", [])
        seam_splits = summary.get("seam_splits", [])
        seam_paths_raw = summary.get("seam_paths", {})
        # Support both old format (list of paths) and new (dict col→path)
        if isinstance(seam_paths_raw, dict):
            seam_paths = list(seam_paths_raw.values())
        else:
            seam_paths = seam_paths_raw

        # Draw split lines: VP=red, seam=blue (diagonal path)
        for s in vp_splits:
            sx = s * scale + scale // 2 + 1
            draw.line([(sx, margin_top), (sx, margin_top + bh)], fill=(220, 40, 40), width=2)
            draw.text((sx - 3, margin_top + bh + 2), str(s), fill=(220, 40, 40))

        # Draw actual seam paths as bright, thick diagonal overlays
        seam_colors = [(0, 80, 255, 220), (30, 120, 255, 220), (60, 150, 255, 220)]
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        pad = max(1, scale // 3)  # expand each pixel by pad on each side
        for pi, path in enumerate(seam_paths):
            color = seam_colors[pi % len(seam_colors)]
            for r in range(len(path)):
                sx = path[r] * scale + 1
                sy = margin_top + r * scale
                odraw.rectangle(
                    [(sx - pad, sy), (sx + scale - 1 + pad, sy + scale - 1)],
                    fill=color,
                )
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)

        # Label seam split columns at bottom
        for s in seam_splits:
            sx = s * scale + scale // 2 + 1
            draw.text((sx - 3, margin_top + bh + 2), str(s), fill=(0, 80, 255))

        wtext = summary.get("word_text", wd)
        n_exp = summary.get("n_chars_expected", "?")
        n_got = summary.get("n_segments_produced", "?")
        mismatch = summary.get("mismatch", False)

        results.append((canvas, wtext, n_exp, n_got, mismatch,
                         len(vp_splits), len(seam_splits)))

    if not results:
        return None

    # Combine all word images horizontally with gap
    gap = 8
    total_w = sum(c.width for c, *_ in results) + gap * (len(results) - 1)
    max_h = max(c.height for c, *_ in results)
    combined = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))
    x_off = 0
    for canvas, *_ in results:
        combined.paste(canvas, (x_off, 0))
        x_off += canvas.width + gap

    combined = combined.convert("RGB")

    # Build a caption
    parts = []
    for _, wtext, n_exp, n_got, mismatch, nvp, nseam in results:
        info = f'"{wtext}" {n_got}/{n_exp}'
        if mismatch:
            info += " ⚠"
        tags = []
        if nvp: tags.append(f"{nvp} VP")
        if nseam: tags.append(f"{nseam} seam")
        if tags:
            info += f" ({', '.join(tags)})"
        parts.append(info)

    caption = " | ".join(parts)
    return combined, caption


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
    automatically by --diag-seg), then falls back to the legacy
    UNSCAN_DUMP_CROPS directory.
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

    # Fall back to legacy UNSCAN_DUMP_CROPS dir
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
                    diag_seg_root=None):
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

        ref_img = render_char(ch, correct_font_path, NORM_H) if correct_font_path else None

        # Find the CI distance for this character against the correct font
        correct_char_dist = None
        if correct_font_path:
            for nf, nd in cv.get("nearest", []):
                if fonts_match(nf, correct_font_name):
                    correct_char_dist = nd
                    break

        # Use actual CI reference image from diag-seg if available (exact data
        # the character index compared against), fall back to PIL re-render.
        chosen_img = None
        if crop_dir:
            refs_dir = os.path.join(crop_dir, "..", "refs")
            if os.path.isdir(refs_dir):
                prefix = f"ref_{crop_idx:02d}_"
                for rf in sorted(os.listdir(refs_dir)):
                    if rf.startswith(prefix):
                        chosen_img = os.path.join(refs_dir, rf)
                        break
        if chosen_img is None:
            chosen_img = render_char(ch, chosen_font_path, NORM_H) if chosen_font_path else None
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
    matched = entry.get("font_matched", "?")
    rank_str = f"CI #{ci_rank}, score {ci_score:.4f}" if ci_rank else "not in CI"

    # SSIM verification info
    ssim_val = entry.get("ssim_score")
    ssim_pass = entry.get("ssim_pass")
    if ssim_val is not None:
        ssim_cls = "ssim-pass" if ssim_pass else "ssim-fail"
        ssim_label = "pass" if ssim_pass else "FAIL"
        ssim_html = f' <span class="{ssim_cls}">SSIM {ssim_val:.4f} ({ssim_label})</span>'
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
    chosen_rank_str = f"CI #{chosen_rank}, score {chosen_score:.4f}" if chosen_rank and chosen_score else ""
    chosen_col_hdr = f"{matched}<br><span class='score'>{chosen_rank_str}</span>"

    # Segmentation picture — show both plain and ligature when available
    seg_html = ""
    diag_line_dir = find_diag_seg_dir(diag_seg_root, entry["page"], entry.get("text", ""),
                                       line_index=entry.get("line_index"))
    if diag_line_dir:
        seg_winner = entry.get("seg_winner")
        has_lig_path = any(
            os.path.isdir(os.path.join(diag_line_dir, d, "seg_lig"))
            for d in os.listdir(diag_line_dir)
            if d.startswith("word_") and os.path.isdir(os.path.join(diag_line_dir, d))
        )

        if has_lig_path:
            # Show both paths side by side
            seg_plain_result = render_seg_picture(diag_line_dir, seg_subdir="seg_plain")
            seg_lig_result = render_seg_picture(diag_line_dir, seg_subdir="seg_lig")

            parts = []
            for label, result, is_winner in [
                ("Plain", seg_plain_result, seg_winner == "plain"),
                ("Ligature", seg_lig_result, seg_winner == "ligature"),
            ]:
                if result:
                    seg_img, seg_caption = result
                    winner_badge = ' <span style="color:#2e7d32;font-weight:bold">★ winner</span>' if is_winner else ''
                    parts.append(f"""<div style="flex:1;min-width:0">
<div style="font-weight:600;margin-bottom:4px">{label} segmentation{winner_badge}</div>
<img src="{img_to_b64(seg_img)}" class="seg-img" style="max-width:100%">
<div class="seg-caption">{seg_caption}</div>
</div>""")

            if parts:
                seg_html = f"""<div class="seg-block">
<div class="seg-legend">
  <span class="leg-vp">■ VP split</span>
  <span class="leg-seam">■ seam split</span>
</div>
<div style="display:flex;gap:16px;flex-wrap:wrap">
{"".join(parts)}
</div>
</div>"""
        else:
            seg_result = render_seg_picture(diag_line_dir)
            if seg_result:
                seg_img, seg_caption = seg_result
                seg_html = f"""<div class="seg-block">
<div class="seg-legend">
  <span class="leg-vp">■ VP split</span>
  <span class="leg-seam">■ seam split</span>
</div>
<img src="{img_to_b64(seg_img)}" class="seg-img">
<div class="seg-caption">{seg_caption}</div>
</div>"""

    # Show alternate (lig) CI candidates when available
    return f"""<div class="miss">
<h3>p{entry['page']}:L{entry['line_index']} — "{text_preview}"{ssim_html}</h3>
{seg_html}
<table>
<tr>
  <th>Scan</th>
  <th class="correct">Correct: {correct_col_hdr}</th>
  <th class="chosen">Unscan pick: {chosen_col_hdr}</th>
  <th>OCR</th>
</tr>
{"".join(rows)}
</table>
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
  max-width: 800px;
}
h2 { font-size: 16px; margin-bottom: 12px; color: #111; }
.summary { color: #555; font-size: 12px; margin-bottom: 16px; }
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
.seg-block {
  margin: 6px 0 10px 0; padding: 8px; background: #f9f9f9;
  border: 1px solid #e0e0e0; border-radius: 4px;
}
.seg-img {
  max-width: 100%; image-rendering: pixelated;
  border: 1px solid #ddd; display: block; margin: 4px 0;
}
.seg-caption { font-size: 10px; color: #666; margin-top: 2px; }
.seg-legend { font-size: 10px; margin-bottom: 4px; }
.seg-legend span { margin-right: 12px; }
.leg-vp { color: #dc2828; }
.leg-seam { color: #1e64ff; }
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

    doc = fitz.open(args.vector_pdf)
    page_spans = extract_vector_spans(doc)

    entries = audit["text_entries"]
    total = len(entries)

    # Classify each audit line against vector PDF ground truth
    misses = []
    ssim_failures = []
    hits = 0
    skipped = 0
    ssim_fail_count = 0
    total_chars = 0
    corrected_chars = 0

    unmatched = 0

    for e in entries:
        # Count OCR corrections across all entries
        for cv in e.get("ci_char_votes", []):
            total_chars += 1
            if cv.get("ocr_corrected_from"):
                corrected_chars += 1

        matched = e.get("font_matched", "")
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

        if e.get("ssim_pass") is False:
            ssim_fail_count += 1

        if fonts_match(matched, actual_font):
            hits += 1
            # Still track SSIM failures even when font is correct
            if e.get("ssim_pass") is False:
                gt_key, gt_score, gt_rank = find_correct_ci_candidate(e, actual_font)
                ssim_failures.append((e, actual_font, gt_key, gt_score, gt_rank))
        else:
            # Find correct font's CI candidate for rendering
            gt_key, gt_score, gt_rank = find_correct_ci_candidate(e, actual_font)
            misses.append((e, actual_font, gt_key, gt_score, gt_rank))

    doc.close()

    unmatched_str = f" ({unmatched} unmatched)" if unmatched else ""
    ocr_corr_str = f" | OCR corrections: {corrected_chars}/{total_chars}" if total_chars else ""
    print(f"Total: {total}  Hits: {hits}  Misses: {len(misses)}{unmatched_str}  Skipped: {skipped}  OCR corrections: {corrected_chars}/{total_chars}",
          file=sys.stderr)

    if not misses and not ssim_failures:
        ssim_fail_str_early = f" | {ssim_fail_count} SSIM failures" if ssim_fail_count else ""
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>unscan char-misses — 100%</title></head>
<body style="background: white; color: #222;">
{CSS}
<h2>unscan Miss Report</h2>
<div class="summary">{hits}/{hits + len(misses)} correct (100.0%) — no misses 🎉{ssim_fail_str_early}{ocr_corr_str}</div>
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
        correct_font_path = find_font_file_by_key(gt_key)
        if not correct_font_path and font_map:
            correct_font_path = resolve_font_from_map(actual_font, font_map)
        if not correct_font_path:
            correct_font_path = find_font_file(actual_font)
        correct_font_name = actual_font

        # Chosen (wrong) font
        matched = entry.get("font_matched", "")
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

        correct_font_path = find_font_file_by_key(gt_key)
        if not correct_font_path and font_map:
            correct_font_path = resolve_font_from_map(actual_font, font_map)
        if not correct_font_path:
            correct_font_path = find_font_file(actual_font)
        correct_font_name = actual_font

        matched = entry.get("font_matched", "")
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
        )
        ssim_blocks.append(block)

    ssim_fail_str = f" | {ssim_fail_count} SSIM failures" if ssim_fail_count else ""
    unmatched_html = f" | {unmatched} unmatched" if unmatched else ""
    compared = hits + len(misses)
    pct = hits / compared * 100 if compared else 0
    ssim_section = ""
    if ssim_blocks:
        ssim_section = f"""<h2 style="margin-top:2em; color:#c55;">SSIM Failures (correct font, SSIM rejected)</h2>
{"".join(ssim_blocks)}"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>unscan char-misses — {hits}/{compared} ({pct:.1f}%)</title>
</head>
<body style="background: white; color: #222;">
{CSS}
<h2>unscan Miss Report</h2>
<div class="summary">{hits}/{compared} correct ({pct:.1f}%) — {len(misses)} misses shown below{unmatched_html}{ssim_fail_str}{ocr_corr_str}</div>
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
