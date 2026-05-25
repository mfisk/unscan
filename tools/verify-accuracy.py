#!/usr/bin/env python3
"""Verify unscan accuracy against a vector PDF source of truth.

Extracts every text span from the vector PDF with its font and bbox,
then checks what font unscan assigned to the same region in the
rasterized version. The vector PDF spans are the ground truth.

Usage:
    ./target/release/unscan RASTERIZED.pdf -o /dev/null --audit-log /tmp/audit.json
    python3 tools/verify-accuracy.py /tmp/audit.json VECTOR.pdf [--verbose] [--misses-only]
"""
import argparse
import json
import re
import sys
from collections import Counter

import fitz  # PyMuPDF

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


def normalize(name):
    if "+" in name:
        name = name.split("+", 1)[1]
    return re.sub(r'[^a-z0-9]', '', name.lower())


def base_family(name):
    n = normalize(name)
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
                       "display", "caption", "subhead", "smtext"]:
            if n.endswith(suffix) and len(n) > len(suffix):
                n = n[:-len(suffix)]
                changed = True
                break
    return n


def canon(name):
    n = normalize(name)
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


def fonts_equivalent(a, b):
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    ba, bb = base_family(a), base_family(b)
    if ba == bb:
        return True
    if ba in bb or bb in ba:
        return True
    return canon(a) == canon(b)


# ---------------------------------------------------------------------------
# Extract ground truth spans from vector PDF
# ---------------------------------------------------------------------------

def extract_gt_spans(doc):
    """Extract every text span from the vector PDF.

    Returns list of dicts with page, font, text, bbox (in PDF points).
    """
    spans = []
    idx = 0
    for pi in range(len(doc)):
        page = doc[pi]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    spans.append({
                        "idx": idx,
                        "page": pi + 1,
                        "font": span["font"],
                        "text": text,
                        "bbox": (x0, y0, x1, y1),
                    })
                    idx += 1
    return spans


# ---------------------------------------------------------------------------
# Build spatial index of audit entries
# ---------------------------------------------------------------------------

def build_audit_index(entries):
    """Index audit entries by page for spatial lookup.

    Each entry's bbox is in pixel coords — convert to PDF points.
    """
    by_page = {}
    for e in entries:
        b = e.get("bbox")
        if not b:
            continue
        rec = {
            "font_matched": e.get("font_matched", ""),
            "text": e.get("text", ""),
            "line_index": e.get("line_index", -1),
            # bbox in PDF points
            "x0": b["x"] / SCALE,
            "y0": b["y"] / SCALE,
            "x1": (b["x"] + b["width"]) / SCALE,
            "y1": (b["y"] + b["height"]) / SCALE,
        }
        by_page.setdefault(e["page"], []).append(rec)
    return by_page


def lookup_unscan_font(audit_index, page, bbox_pts):
    """Find the unscan font that covers a vector PDF span's bbox.

    Returns (font_matched, line_index) or (None, None).
    """
    gx0, gy0, gx1, gy1 = bbox_pts
    best_overlap = 0
    best_entry = None

    for e in audit_index.get(page, []):
        # Intersection
        ix0 = max(gx0, e["x0"])
        iy0 = max(gy0, e["y0"])
        ix1 = min(gx1, e["x1"])
        iy1 = min(gy1, e["y1"])
        if ix0 < ix1 and iy0 < iy1:
            area = (ix1 - ix0) * (iy1 - iy0)
            if area > best_overlap:
                best_overlap = area
                best_entry = e

    if best_entry:
        return best_entry["font_matched"], best_entry["line_index"]
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Verify unscan accuracy against vector PDF ground truth")
    ap.add_argument("audit", help="unscan --audit-log JSON")
    ap.add_argument("vector_pdf", help="Original vector PDF")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--misses-only", "-m", action="store_true")
    ap.add_argument("--dump-gt", metavar="PATH",
                    help="Dump ground truth spans to JSON")
    args = ap.parse_args()

    with open(args.audit) as f:
        audit = json.load(f)

    doc = fitz.open(args.vector_pdf)
    gt_spans = extract_gt_spans(doc)
    audit_index = build_audit_index(audit["text_entries"])

    if args.dump_gt:
        with open(args.dump_gt, "w") as f:
            json.dump([{**s, "bbox": list(s["bbox"])} for s in gt_spans], f, indent=2)
        print(f"Dumped {len(gt_spans)} GT spans to {args.dump_gt}", file=sys.stderr)

    hits = 0
    misses = 0
    no_match = 0
    miss_list = []

    for s in gt_spans:
        unscan_font, line_idx = lookup_unscan_font(
            audit_index, s["page"], s["bbox"])

        if unscan_font is None:
            no_match += 1
            continue

        if fonts_equivalent(unscan_font, s["font"]):
            hits += 1
        else:
            misses += 1
            miss_list.append({
                "idx": s["idx"],
                "page": s["page"],
                "line": line_idx,
                "gt_font": s["font"],
                "unscan_font": unscan_font,
                "text": s["text"],
            })

    total = hits + misses
    pct = hits / total * 100 if total else 0

    print(f"\n{'=' * 60}")
    print(f"  unscan vs vector PDF  ({len(gt_spans)} spans)")
    print(f"{'=' * 60}")
    print(f"  GT spans:    {len(gt_spans)}")
    print(f"  Matched:     {total}  (no audit overlap: {no_match})")
    print(f"  Correct:     {hits}/{total}  ({pct:.1f}%)")
    print(f"  Wrong:       {misses}")
    print(f"{'=' * 60}\n")

    if miss_list:
        print("MISSES:")
        for m in miss_list:
            print(f"  #{m['idx']:3d}  p{m['page']}:L{m['line']:2d}  "
                  f"gt=\"{m['gt_font'][:28]:28s}\"  "
                  f"unscan=\"{m['unscan_font'][:28]:28s}\"  "
                  f"\"{m['text'][:40]}\"")
        print()

    if args.verbose:
        for s in gt_spans:
            uf, li = lookup_unscan_font(audit_index, s["page"], s["bbox"])
            if uf is None:
                tag = "SKIP"
                eq = False
            else:
                eq = fonts_equivalent(uf, s["font"])
                tag = " OK " if eq else "MISS"
            if args.misses_only and tag != "MISS":
                continue
            print(f"  [{tag}] #{s['idx']:3d}  p{s['page']}  "
                  f"gt=\"{s['font'][:25]:25s}\"  "
                  f"unscan=\"{(uf or '—')[:25]:25s}\"  "
                  f"\"{s['text'][:40]}\"")

    doc.close()
    sys.exit(0 if misses == 0 else 1)


if __name__ == "__main__":
    main()
