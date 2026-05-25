#!/usr/bin/env python3
"""
char-misses.py — Visual miss report for unscan font identification.

Uses the original vector PDF as ground truth: extracts every text span
with its font and bbox, then spatially matches against unscan's audit log
to find genuine misses. No JSON metadata involved.

Usage:
    # 1. Run unscan with crops + audit:
    rm -rf /tmp/unscan-crops
    UNSCAN_DUMP_CROPS=1 ./target/release/unscan RASTERIZED.pdf \
        -o /dev/null --audit-log /tmp/audit.json

    # 2. Generate the report against the vector PDF:
    python3 tools/char-misses.py /tmp/audit.json VECTOR.pdf \
        --crops /tmp/unscan-crops -o /tmp/misses.html
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
    "timesnewroman": "times", "timesnewromanps": "times",
    "timesroman": "times", "nimbusroman": "times",
    "tinos": "times", "freeserif": "times",
    "freeserifitalic": "times", "freeserifbold": "times",
    "freeserifbolditalic": "times",
    "p052": "times", "c059": "times",
    "couriernew": "courier", "couriernewps": "courier",
    "nimbusmonops": "courier", "freemono": "courier",
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
    na = re.sub(r'[^a-z0-9]', '', a.lower())
    nb = re.sub(r'[^a-z0-9]', '', b.lower())
    if na == nb:
        return True
    ba, bb = base_family(a), base_family(b)
    if ba == bb:
        return True
    if ba in bb or bb in ba:
        return True
    return canon(a) == canon(b)


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


def lookup_actual_font(page_spans, page, bbox_px):
    """Find the dominant font in the vector PDF at the given pixel bbox."""
    px0 = bbox_px["x"] / SCALE
    py0 = bbox_px["y"] / SCALE
    px1 = (bbox_px["x"] + bbox_px["width"]) / SCALE
    py1 = (bbox_px["y"] + bbox_px["height"]) / SCALE

    font_count = Counter()
    for sx0, sy0, sx1, sy1, sfont, stext in page_spans.get(page, []):
        if sx1 > px0 and sx0 < px1 and sy1 > py0 and sy0 < py1:
            font_count[sfont] += len(stext)

    if not font_count:
        return None
    return font_count.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Font file resolution
# ---------------------------------------------------------------------------

def find_font_file(font_name):
    """Resolve a font name to a .ttf/.otf path on disk."""
    fn = clean(font_name)
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


def find_correct_ci_candidate(entry, actual_font):
    """Find the CI candidate that matches the actual (vector PDF) font.

    Returns (font_key, score, rank) or (None, None, None).
    """
    for i, c in enumerate(entry.get("ci_candidates", [])):
        if fonts_match(c["font_key"], actual_font):
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

def find_crop_dir(crops_root, page, text):
    if not os.path.isdir(crops_root):
        return None, []
    text_clean = re.sub(r'[^a-zA-Z0-9]', '', text[:30]).lower()
    page_dirs = sorted(d for d in os.listdir(crops_root) if d.startswith(f"p{page}_"))
    for d in page_dirs:
        d_clean = re.sub(r'[^a-zA-Z0-9]', '', d).lower()
        if text_clean[:15] in d_clean:
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
                    chosen_font_path, chosen_font_name):
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
        chosen_img = render_char(ch, chosen_font_path, NORM_H) if chosen_font_path else None
        dc = dist_class(d2)

        ocr_best_dist = cv.get("ocr_char_best_dist")
        ocr_near = cv.get("ocr_char_nearest", cv.get("nearest", []))
        if ocr_from and ocr_near:
            ocr_font = ocr_near[0][0].rsplit("/", 1)[-1] if ocr_near else "?"
            ocr_d = ocr_near[0][1] if ocr_near else 0
            ocr_dc = dist_class(ocr_d)
            ocr_cell = f"<span class='char-label'>'{original_ocr}'</span><br><span class='font-mini'>{ocr_font}</span><br><span class='num {ocr_dc}'>{ocr_d:.4f}</span>"
        elif not ocr_from:
            near = cv.get("nearest", [])
            if near:
                ocr_font = near[0][0].rsplit("/", 1)[-1]
                ocr_d = near[0][1]
                ocr_dc = dist_class(ocr_d)
                ocr_cell = f"<span class='char-label'>'{ch}'</span><br><span class='font-mini'>{ocr_font}</span><br><span class='num {ocr_dc}'>{ocr_d:.4f}</span>"
            else:
                ocr_cell = "—"
        else:
            ocr_cell = "—"

        alt_ch = cv.get("alt_char")
        alt_dist = cv.get("alt_char_best_dist")
        if alt_ch and alt_dist is not None:
            if ocr_from:
                near = cv.get("nearest", [])
                alt_font = near[0][0].rsplit("/", 1)[-1] if near else "?"
            else:
                alt_font = ""
            alt_dc = dist_class(alt_dist)
            alt_cell = f"<span class='char-label'>'{alt_ch}'</span><br><span class='font-mini'>{alt_font}</span><br><span class='num {alt_dc}'>{alt_dist:.4f}</span>"
        else:
            alt_cell = "<span class='dimmed'>same</span>"

        if ocr_best_dist and alt_dist and alt_dist > 0:
            ratio = ocr_best_dist / alt_dist
            ratio_str = f"{ratio:.1f}×"
        else:
            ratio_str = ""

        rows.append(f"""<tr>
  <td class="img-td">{img_td(crop_img)}<div class="sub">OCR: {ocr_label}</div></td>
  <td class="img-td">{img_td(ref_img)}</td>
  <td class="img-td">{img_td(chosen_img)}</td>
  <td class="ocr-col">{ocr_cell}</td>
  <td class="alt-col">{alt_cell}</td>
  <td class="num ratio">{ratio_str}</td>
</tr>""")

    text_preview = entry["text"][:60]
    matched = entry.get("font_matched", "?")
    rank_str = f"CI #{ci_rank}, score {ci_score:.3f}" if ci_rank else "not in CI"

    chosen_score = None
    for c in entry.get("ci_candidates", []):
        if fonts_match(c["font_key"], matched):
            chosen_score = c["score"]
            break

    correct_col_hdr = f"{correct_font_name}<br><span class='score'>{rank_str}</span>"
    chosen_score_str = f"score {chosen_score:.3f}" if chosen_score else ""
    chosen_col_hdr = f"{matched}<br><span class='score'>{chosen_score_str}</span>"

    return f"""<div class="miss">
<h3>p{entry['page']}:L{entry['line_index']} — "{text_preview}"</h3>
<table>
<tr>
  <th>Scan + OCR</th>
  <th class="correct">{correct_col_hdr}</th>
  <th class="chosen">{chosen_col_hdr}</th>
  <th>OCR char</th>
  <th>Alt char</th>
  <th>Ratio</th>
</tr>
{"".join(rows)}
</table>
</div>"""


CSS = """<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
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
.ocr-col, .alt-col { text-align: center; font-size: 11px; vertical-align: middle; padding: 4px; }
.char-label { font-size: 14px; font-weight: 600; }
.font-mini { font-size: 9px; color: #888; word-break: break-all; max-width: 100px; display: inline-block; }
.dimmed { color: #bbb; font-size: 10px; }
.ratio { font-size: 11px; color: #888; }
</style>"""


def main():
    parser = argparse.ArgumentParser(description="Visual miss report for unscan")
    parser.add_argument("audit", help="Path to audit JSON from --audit-log")
    parser.add_argument("vector_pdf", help="Path to the original vector PDF")
    parser.add_argument("--crops", default="/tmp/unscan-crops",
                        help="Path to UNSCAN_DUMP_CROPS output dir")
    parser.add_argument("-o", "--output", default="/tmp/misses.html",
                        help="Output HTML file path")
    args = parser.parse_args()

    with open(args.audit) as f:
        audit = json.load(f)

    doc = fitz.open(args.vector_pdf)
    page_spans = extract_vector_spans(doc)

    entries = audit["text_entries"]
    total = len(entries)

    # Classify each audit line against vector PDF ground truth
    misses = []
    hits = 0
    skipped = 0

    for e in entries:
        matched = e.get("font_matched", "")
        bbox = e.get("bbox")
        if not matched or not bbox:
            skipped += 1
            continue

        actual_font = lookup_actual_font(page_spans, e["page"], bbox)
        if actual_font is None:
            skipped += 1
            continue

        if fonts_match(matched, actual_font):
            hits += 1
        else:
            # Find correct font's CI candidate for rendering
            gt_key, gt_score, gt_rank = find_correct_ci_candidate(e, actual_font)
            misses.append((e, actual_font, gt_key, gt_score, gt_rank))

    doc.close()

    print(f"Total: {total}  Hits: {hits}  Misses: {len(misses)}  Skipped: {skipped}",
          file=sys.stderr)

    if not misses:
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>unscan char-misses — 100%</title></head>
<body style="background: white; color: #222;">
{CSS}
<h2>unscan Miss Report</h2>
<div class="summary">{hits}/{hits + len(misses)} correct (100.0%) — no misses 🎉</div>
</body>
</html>"""
        with open(args.output, "w") as f:
            f.write(html)
        print(f"Report written to {args.output}", file=sys.stderr)
        return

    # Build HTML
    miss_blocks = []
    for entry, actual_font, gt_key, gt_score, gt_rank in misses:
        crop_dir, crop_files = find_crop_dir(args.crops, entry["page"], entry["text"])

        chars = entry.get("ci_char_votes", [])
        interesting = pick_interesting_chars(chars)

        # Correct font: use the CI candidate file if found, else search by name
        correct_font_path = find_font_file_by_key(gt_key)
        if not correct_font_path:
            correct_font_path = find_font_file(actual_font)
        correct_font_name = actual_font

        # Chosen (wrong) font
        matched = entry.get("font_matched", "")
        chosen_font_path = find_font_file(matched)

        block = build_miss_html(
            entry, interesting, crop_dir, crop_files,
            correct_font_path, correct_font_name, gt_rank, gt_score,
            chosen_font_path, matched,
        )
        miss_blocks.append(block)

    compared = hits + len(misses)
    pct = hits / compared * 100 if compared else 0
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>unscan char-misses — {hits}/{compared} ({pct:.1f}%)</title>
</head>
<body style="background: white; color: #222;">
{CSS}
<h2>unscan Miss Report</h2>
<div class="summary">{hits}/{compared} correct ({pct:.1f}%) — {len(misses)} misses shown below</div>
{"".join(miss_blocks)}
</body>
</html>"""

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)

    print(f"Report written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
