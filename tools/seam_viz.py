#!/usr/bin/env python3
"""Seam pixel visualizer: side-by-side zoomed panels with per-pixel cost inside each cell.

Reads all data from audit output (summary.json + word_crop.png). No recomputation.
The seam path comes from seam_paths in summary.json; per-pixel ink/delta are
read from the image along the recorded path.

Each path pixel is outlined in a color indicating cost type, with ink value
and delta penalty written inside the cell.

Color coding:
  blue   = ink only (delta=0)
  orange = ink + Δ
  red    = big Δ (>50)
  green  = free (dark=0)

Green column outline = target column for the seam.
Yellow horizontal line = mid row (forward/reverse DP meeting point).
Totals at bottom of each panel.

Usage:
  python3 tools/seam_viz.py <audit.json> <word> <col_a> <col_b> [-o output.png]
  python3 tools/seam_viz.py <audit.json> <word> <col_a> <col_b> --line 5
  python3 tools/seam_viz.py dummy WORD <col_a> <col_b> --seg-dir <seg_plain_dir>  # legacy
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse, json, sys, os

DELTA_WEIGHT = 1.0
SCALE = 32
PAD_COLS = 6
WIDTH_PENALTY = 1
PANEL_GAP = 40
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CELL_FONT_SIZE = 9
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16

# Colors
COL_FREE    = (50, 200, 50)
COL_INK     = (100, 180, 255)
COL_DELTA   = (30, 60, 180)
COL_TARGET  = (0, 200, 0, 128)
COL_MID     = (180, 180, 0)
COL_BG      = (20, 20, 20)
COL_TEXT    = (200, 200, 200)
COL_LABEL   = (255, 255, 255)


def delta_ink(dc, dp):
    if dc <= dp:
        return 0.0
    return DELTA_WEIGHT * (dc - dp)


def steps_from_recorded_path(dark, path_data):
    """Walk a recorded seam path over the image.
    Path is a list of [row, col] pairs from the audit data — including
    pass-through pixels.  Show exactly what the audit contains.
    Returns (steps, tot_ink, tot_delta)."""
    steps = []
    tot_ink = 0.0
    tot_delta = 0.0
    cum = 0.0
    prev_dark = 0.0  # boundary above

    for entry in path_data:
        r, c = int(entry[0]), int(entry[1])
        d = float(dark[r, c])
        dlt = delta_ink(d, prev_dark)
        cum += d + dlt
        tot_ink += d
        tot_delta += dlt
        steps.append(dict(r=r, c=c, dark=d, ink=d, delta=dlt, cum=cum))
        prev_dark = d

    return steps, tot_ink, tot_delta


def steps_from_vertical(dark, col):
    """Walk a straight vertical column — VP candidate. Includes run-length
    and row-ink discounts."""
    VERT_RUN_THRESHOLD = 11
    VERT_RUN_DISCOUNT = 0.02
    VERT_ROW_INK_DIVISOR = 8.0
    h, w_full = dark.shape
    row_ink = dark.sum(axis=1)
    max_row_ink = max(row_ink.max(), 1e-9)
    steps = []
    tot_ink = 0.0
    tot_delta = 0.0
    cum = 0.0
    prev_dark = 0.0
    for r in range(h):
        d = float(dark[r, col])
        run_len = 1
        cx = col - 1
        while cx >= 0 and dark[r, cx] > 0:
            run_len += 1; cx -= 1
        cx = col + 1
        while cx < w_full and dark[r, cx] > 0:
            run_len += 1; cx += 1
        if run_len >= VERT_RUN_THRESHOLD:
            run_wt = max(0.1, 1.0 - VERT_RUN_DISCOUNT * (run_len - VERT_RUN_THRESHOLD + 1))
        else:
            run_wt = 1.0
        row_wt = 1.0 - (row_ink[r] / max_row_ink) / VERT_ROW_INK_DIVISOR
        wt = run_wt * row_wt
        ink = d
        dlt = delta_ink(d, prev_dark)
        weighted = (ink + dlt) * wt
        cum += weighted
        tot_ink += ink * wt
        tot_delta += dlt * wt
        steps.append(dict(r=r, c=col, dark=d, ink=ink, delta=dlt,
                          run_len=run_len, run_weight=run_wt,
                          row_weight=row_wt, weight=wt, cum=cum))
        prev_dark = d
    return steps, tot_ink, tot_delta


def step_color(s):
    if s['dark'] == 0:
        return COL_FREE
    if s['delta'] > 0:
        return (255, 100, 100) if s['delta'] > 50 else COL_DELTA
    return COL_INK


def text_color_for_bg(orig_val):
    return (255, 255, 255) if orig_val < 128 else (0, 0, 0)


def load_fonts():
    try:
        cell_font = ImageFont.truetype(FONT_PATH, CELL_FONT_SIZE)
        label_font = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)
        title_font = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
    except IOError:
        cell_font = label_font = title_font = ImageFont.load_default()
    return cell_font, label_font, title_font


def render_panel(dark, orig, col, steps, combined, tot_ink, tot_delta,
                 cell_font, label_font, swp=0.0, ssp=0.0, hc=0.0):
    h = dark.shape[0]
    mid = h // 2

    path_cols = [s['c'] for s in steps]
    c_min = max(0, min(path_cols) - PAD_COLS)
    c_max = min(orig.shape[1], max(path_cols) + PAD_COLS + 1)
    cw = c_max - c_min

    step_map = {(s['r'], s['c']): s for s in steps}

    header_h = LABEL_FONT_SIZE + 8
    totals_h = LABEL_FONT_SIZE * 5 + 24
    panel_w = cw * SCALE
    panel_h = header_h + h * SCALE + totals_h

    img = Image.new('RGB', (panel_w, panel_h), COL_BG)
    draw = ImageDraw.Draw(img)

    path_width = max(path_cols) - min(path_cols)
    header = f"Col {col}: total={combined:.0f}"
    draw.text((4, 2), header, fill=COL_LABEL, font=label_font)
    y_off = header_h

    for r in range(h):
        for c in range(c_min, c_max):
            v = int(orig[r, c])
            lc = c - c_min
            x0 = lc * SCALE
            y0 = y_off + r * SCALE
            draw.rectangle([x0, y0, x0 + SCALE - 1, y0 + SCALE - 1],
                           fill=(v, v, v))

    my = y_off + mid * SCALE + SCALE // 2
    draw.line([(0, my), (cw * SCALE, my)], fill=COL_MID, width=1)

    for s in steps:
        lc = s['c'] - c_min
        x0 = lc * SCALE
        y0 = y_off + s['r'] * SCALE
        color = step_color(s)
        bg_val = int(orig[s['r'], s['c']])

        draw.rectangle([x0, y0, x0 + SCALE - 1, y0 + SCALE - 1],
                       outline=color, width=2)

        tc_ = text_color_for_bg(bg_val)

        has_wt = 'weight' in s and s['weight'] < 0.999
        ink_str = f"{s['dark']:.0f}"
        line1_w = cell_font.getlength(ink_str)
        tx = x0 + (SCALE - line1_w) / 2

        if has_wt:
            lines = [ink_str]
            if s['delta'] > 0:
                lines.append(f"+{s['delta']:.0f}")
            lines.append(f"\u00d7{s['weight']:.2f}")
            n = len(lines)
            ty = y0 + (SCALE - n * (CELL_FONT_SIZE + 1)) / 2
            for i, line in enumerate(lines):
                lw = cell_font.getlength(line)
                lx = x0 + (SCALE - lw) / 2
                c_ = (255, 200, 80) if line.startswith('\u00d7') else color
                draw.text((lx, ty + i * (CELL_FONT_SIZE + 1)), line, fill=c_, font=cell_font)
        elif s['delta'] > 0:
            ty = y0 + (SCALE - 2 * CELL_FONT_SIZE - 2) / 2
            draw.text((tx, ty), ink_str, fill=color, font=cell_font)
            dlt_str = f"+{s['delta']:.0f}"
            dlt_w = cell_font.getlength(dlt_str)
            dx = x0 + (SCALE - dlt_w) / 2
            draw.text((dx, ty + CELL_FONT_SIZE + 2), dlt_str, fill=color, font=cell_font)
        elif s['dark'] > 0:
            ty = y0 + (SCALE - CELL_FONT_SIZE) / 2
            draw.text((tx, ty), ink_str, fill=color, font=cell_font)

    ty = y_off + h * SCALE + 8
    draw.text((4, ty), f"col={col}  cells={len(steps)}", fill=COL_LABEL, font=label_font)
    ty += LABEL_FONT_SIZE + 4
    draw.text((4, ty), f"ink={tot_ink:.0f}  \u0394={tot_delta:.0f}  seam_w={swp:.0f}  seg_sz={ssp:.0f}  horiz={hc:.0f}",
              fill=COL_LABEL, font=label_font)
    ty += LABEL_FONT_SIZE + 4
    draw.text((4, ty), f"total={combined:.0f}",
              fill=COL_LABEL, font=label_font)

    return img, combined


def render(dark, orig, col_a, col_b, out_path,
           steps_a=None, steps_b=None,
           cost_a=0.0, cost_b=0.0,
           ink_a=0.0, ink_b=0.0,
           delta_a=0.0, delta_b=0.0,
           vp_a=False, vp_b=False,
           swp_a=0.0, swp_b=0.0,
           ssp_a=0.0, ssp_b=0.0,
           hc_a=0.0, hc_b=0.0):
    cell_font, label_font, title_font = load_fonts()

    panel_a, _ = render_panel(dark, orig, col_a, steps_a, cost_a, ink_a, delta_a,
                              cell_font, label_font, swp=swp_a, ssp=ssp_a, hc=hc_a)
    panel_b, _ = render_panel(dark, orig, col_b, steps_b, cost_b, ink_b, delta_b,
                              cell_font, label_font, swp=swp_b, ssp=ssp_b, hc=hc_b)

    title_h = TITLE_FONT_SIZE + 12
    legend_h = CELL_FONT_SIZE + 8
    header_h = title_h + legend_h

    total_w = panel_a.width + PANEL_GAP + panel_b.width
    total_h = header_h + max(panel_a.height, panel_b.height)
    combined = Image.new('RGB', (total_w, total_h), COL_BG)
    draw = ImageDraw.Draw(combined)

    winner = col_a if cost_a < cost_b else col_b
    tag_a = "VP" if vp_a else "seam"
    tag_b = "VP" if vp_b else "seam"
    title = f"Col {col_a} [{tag_a}] ({cost_a:.0f}) vs col {col_b} [{tag_b}] ({cost_b:.0f})  \u2014  winner: {winner}"
    draw.text((8, 4), title, fill=COL_LABEL, font=title_font)

    ly = title_h
    lx = 8
    for label, color in [("\u25a0 ink only", COL_INK), ("\u25a0 ink+\u0394", COL_DELTA),
                          ("\u25a0 free", COL_FREE),
                          ("\u2502 target", (0, 200, 0)), ("\u2500 mid", COL_MID)]:
        draw.text((lx, ly), label, fill=color, font=cell_font)
        lx += int(cell_font.getlength(label)) + 16

    combined.paste(panel_a, (0, header_h))
    combined.paste(panel_b, (panel_a.width + PANEL_GAP, header_h))

    combined.save(out_path)
    print(f"Written to {out_path}")


def sanitize_text(text, max_len=30):
    """Match the slug logic in Rust: alphanumeric pass-through, else underscore."""
    return ''.join(c if c.isalnum() else '_' for c in text[:max_len])


def find_word_crop(audit_dir, page, line_index, word_text, source_word_idx):
    """Locate word_crop.png on disk from audit dir structure."""
    # Line dirs: p{page}_L{line:03}_{slug}
    line_prefix = f"p{page}_L{line_index:03}_"
    for entry in sorted(os.listdir(audit_dir)):
        if entry.startswith(line_prefix) and os.path.isdir(os.path.join(audit_dir, entry)):
            line_dir = os.path.join(audit_dir, entry)
            # Word dirs: word_{idx:03}_{slug}
            word_prefix = f"word_{source_word_idx:03}_"
            for wentry in sorted(os.listdir(line_dir)):
                if wentry.startswith(word_prefix) and os.path.isdir(os.path.join(line_dir, wentry)):
                    seg_plain = os.path.join(line_dir, wentry, 'seg_plain')
                    if os.path.isdir(seg_plain):
                        crop = os.path.join(seg_plain, 'word_crop.png')
                        if os.path.exists(crop):
                            return crop, seg_plain
                    # Flat layout fallback
                    crop = os.path.join(line_dir, wentry, 'word_crop.png')
                    if os.path.exists(crop):
                        return crop, os.path.join(line_dir, wentry)
    return None, None


def find_seg_summary(seg_plain_dir):
    """Load summary.json from seg_plain dir for candidate_seam_paths and seam_seed_candidates."""
    if seg_plain_dir is None:
        return {}
    summary_path = os.path.join(seg_plain_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Seam path comparison visualizer (reads audit.json)")
    p.add_argument('audit_json', help='Path to audit.json')
    p.add_argument('word', help='Word text to visualize (matched against word_segmentation)')
    p.add_argument('col_a', type=int, help='First column')
    p.add_argument('col_b', type=int, help='Second column')
    p.add_argument('--line', type=int, default=None,
                   help='Line index (0-based) to disambiguate if word appears on multiple lines')
    p.add_argument('-o', '--output', default=None,
                   help='Output path (default: /tmp/seam_<a>_vs_<b>.png)')
    # Legacy mode: accept seg_plain dir directly
    p.add_argument('--seg-dir', default=None,
                   help='Legacy: seg_plain directory (bypasses audit.json lookup)')
    a = p.parse_args()

    if a.seg_dir:
        # Legacy mode: read from seg_plain dir directly
        summary_path = os.path.join(a.seg_dir, 'summary.json')
        crop_path = os.path.join(a.seg_dir, 'word_crop.png')
        if not os.path.exists(summary_path):
            print(f"ERROR: {summary_path} not found", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(crop_path):
            print(f"ERROR: {crop_path} not found", file=sys.stderr)
            sys.exit(1)
        with open(summary_path) as f:
            summary = json.load(f)

        seam_paths = {int(k): v for k, v in summary.get('seam_paths', {}).items()}
        seam_costs = {int(k): v for k, v in summary.get('seam_costs', {}).items()}
        ws_splits = set(summary.get('ws_splits', []))
        # Legacy fallback: derive costs from seam_seed_candidates if seam_costs absent
        if not seam_costs:
            for c in summary.get('seam_seed_candidates', []):
                seam_costs[c['col']] = c['cost']
    else:
        # Primary mode: read from audit.json
        if not os.path.exists(a.audit_json):
            print(f"ERROR: {a.audit_json} not found", file=sys.stderr)
            sys.exit(1)

        with open(a.audit_json) as f:
            audit = json.load(f)

        audit_dir = os.path.dirname(os.path.abspath(a.audit_json))

        # Find the matching word_segmentation entry
        match = None
        match_entry = None
        for entry in audit['text_entries']:
            if a.line is not None and entry['line_index'] != a.line:
                continue
            for ws in entry.get('word_segmentation', []):
                if ws['word_text'] == a.word:
                    match = ws
                    match_entry = entry
                    break
            if match:
                break

        if match is None:
            print(f"ERROR: word '{a.word}' not found in audit.json word_segmentation", file=sys.stderr)
            # List available words
            words = set()
            for entry in audit['text_entries']:
                for ws in entry.get('word_segmentation', []):
                    words.add(f"L{entry['line_index']:03}:{ws['word_text']}")
            if words:
                print(f"Available: {', '.join(sorted(words))}", file=sys.stderr)
            sys.exit(1)

        # Seam data from audit.json
        seam_paths = {int(k): v for k, v in match['seam_paths'].items()}
        seam_costs = {int(k): v for k, v in match.get('seam_costs', {}).items()}
        ws_splits = set(match['ws_splits'])

        # Find word_crop.png on disk (still needed for pixel data)
        crop_path, seg_plain_dir = find_word_crop(
            audit_dir, match_entry['page'], match_entry['line_index'],
            match['word_text'], match['source_word_idx'])

        if crop_path is None:
            print(f"ERROR: word_crop.png not found for word '{a.word}' "
                  f"(page {match_entry['page']}, line {match_entry['line_index']})", file=sys.stderr)
            sys.exit(1)

    img = Image.open(crop_path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    dark = 255.0 - arr.astype(float)

    results = {}
    for col in [a.col_a, a.col_b]:
        is_vp = col in ws_splits
        raw_cost = seam_costs.get(col)
        # Handle both old (float) and new (struct) formats
        if isinstance(raw_cost, dict):
            audit_cost = raw_cost.get('total')
            swp = raw_cost.get('seam_width_penalty', 0.0)
            ssp = raw_cost.get('segment_size_penalty', 0.0)
            hc = raw_cost.get('horizontal_cost', 0.0)
        else:
            audit_cost = raw_cost
            swp = 0.0
            ssp = 0.0
            hc = 0.0

        if is_vp:
            steps, tot_ink, tot_delta = steps_from_vertical(dark, col)
            cost = audit_cost if audit_cost is not None else steps[-1]['cum']
        elif col in seam_paths:
            path_cols = seam_paths[col]
            steps, tot_ink, tot_delta = steps_from_recorded_path(dark, path_cols)
            cost = audit_cost if audit_cost is not None else steps[-1]['cum']
        else:
            print(f"ERROR: col {col} has no audit data (not a VP split and no recorded seam path)", file=sys.stderr)
            sys.exit(1)

        results[col] = dict(steps=steps, cost=cost, ink=tot_ink, delta=tot_delta, is_vp=is_vp,
                            seam_width_penalty=swp, segment_size_penalty=ssp, horizontal_cost=hc)

    out = a.output or f'/tmp/seam_{a.col_a}_vs_{a.col_b}.png'
    ra, rb = results[a.col_a], results[a.col_b]
    render(dark, arr, a.col_a, a.col_b, out,
           steps_a=ra['steps'], steps_b=rb['steps'],
           cost_a=ra['cost'], cost_b=rb['cost'],
           ink_a=ra['ink'], ink_b=rb['ink'],
           delta_a=ra['delta'], delta_b=rb['delta'],
           vp_a=ra['is_vp'], vp_b=rb['is_vp'],
           swp_a=ra['seam_width_penalty'], swp_b=rb['seam_width_penalty'],
           ssp_a=ra['segment_size_penalty'], ssp_b=rb['segment_size_penalty'],
           hc_a=ra['horizontal_cost'], hc_b=rb['horizontal_cost'])
