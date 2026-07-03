#!/usr/bin/env python3
"""Seam pixel visualizer: side-by-side zoomed panels with per-pixel cost inside each cell.

Each path pixel is outlined in a color indicating cost type, with ink value
and delta penalty written inside the cell.  No circles, no connecting lines.

Color coding:
  blue   = ink only (delta=0)
  orange = ink + Δ
  red    = big Δ (>50)
  green  = free (dark=0)

Green column outline = target column for the seam.
Yellow horizontal line = mid row (forward/reverse DP meeting point).
Totals at bottom of each panel.

Usage:
  python3 tools/seam_viz.py <word_crop.png> <col_a> <col_b> \
    [--seg-start S --seg-end E] [--summary summary.json] [-o output.png]
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse, json, sys

DELTA_WEIGHT = 4.0
SCALE = 32
PAD_COLS = 6
GAP = 40
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CELL_FONT_SIZE = 9       # text inside each pixel cell
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


def build_dp(dark, seg_start, seg_end):
    h, w_full = dark.shape
    sw = seg_end - seg_start
    base = seg_start
    def d(r, c): return dark[r, base + c]

    n = h * sw
    cf = [0.0] * n; pf = [0] * n
    cr = [0.0] * n; pr = [0] * n

    for c in range(sw):
        cf[c] = d(0, c) + delta_ink(d(0, c), 0); pf[c] = c

    for r in range(1, h):
        ro, po = r * sw, (r - 1) * sw
        # Step 1: vertical from row above
        for c in range(sw):
            dc, dp_ = d(r, c), d(r - 1, c)
            cf[ro + c] = dc + delta_ink(dc, dp_) + cf[po + c]
            pf[ro + c] = po + c
        # Step 2-3: horizontal chaining (no per-step penalty)
        for c in range(1, sw):
            dc, dn = d(r, c), d(r, c - 1)
            v = cf[ro + c - 1] + dc + delta_ink(dc, dn)
            if v < cf[ro + c]:
                cf[ro + c] = v; pf[ro + c] = ro + c - 1
        for c in range(sw - 2, -1, -1):
            dc, dn = d(r, c), d(r, c + 1)
            v = cf[ro + c + 1] + dc + delta_ink(dc, dn)
            if v < cf[ro + c]:
                cf[ro + c] = v; pf[ro + c] = ro + c + 1

    last = h - 1
    for c in range(sw):
        i = last * sw + c; cr[i] = d(last, c) + delta_ink(d(last, c), 0); pr[i] = i

    for r in range(last - 1, -1, -1):
        ro, no = r * sw, (r + 1) * sw
        # Step 1: vertical from row below
        for c in range(sw):
            dc, ch_ = d(r, c), d(r + 1, c)
            cr[ro + c] = dc + delta_ink(ch_, dc) + cr[no + c]
            pr[ro + c] = no + c
        # Step 2-3: horizontal chaining (no per-step penalty)
        for c in range(1, sw):
            dc, dn = d(r, c), d(r, c - 1)
            v = cr[ro + c - 1] + dc + delta_ink(dn, dc)
            if v < cr[ro + c]:
                cr[ro + c] = v; pr[ro + c] = ro + c - 1
        for c in range(sw - 2, -1, -1):
            dc, dn = d(r, c), d(r, c + 1)
            v = cr[ro + c + 1] + dc + delta_ink(dn, dc)
            if v < cr[ro + c]:
                cr[ro + c] = v; pr[ro + c] = ro + c + 1

    return cf, pf, cr, pr, sw, base


def trace(pred, start, sw):
    path = []; idx = start; seen = set()
    while idx not in seen:
        seen.add(idx)
        path.append((idx // sw, idx % sw))
        p = pred[idx]
        if p == idx: break
        idx = p
    path.reverse()
    return path


def full_seam_path(dark, col, seg_start, seg_end):
    h = dark.shape[0]
    cf, pf, cr, pr, sw, base = build_dp(dark, seg_start, seg_end)
    mid = h // 2
    cl = col - base
    mi = mid * sw + cl

    fwd = trace(pf, mi, sw)                     # [row0, ..., mid]
    rev = trace(pr, mi, sw)                      # [last_row, ..., mid]
    rev_down = rev[::-1]                         # [mid, ..., last_row]

    fwd_cells = [(r, c + base) for r, c in fwd]
    rev_cells = [(r, c + base) for r, c in rev_down[1:]]
    cells = fwd_cells + rev_cells

    steps = []
    tot_ink = 0.0; tot_delta = 0.0; cum = 0.0

    for i, (r, c) in enumerate(cells):
        ink = float(dark[r, c])
        if i == 0:
            dlt = delta_ink(ink, 0)
        else:
            pr_, pc_ = cells[i - 1]
            dlt = delta_ink(ink, dark[pr_, pc_])
        cum += ink + dlt
        tot_ink += ink; tot_delta += dlt
        steps.append(dict(r=r, c=c, dark=ink, ink=ink, delta=dlt, cum=cum))

    # Last cell: exit to below (no ink out of bounds)
    last_ink = steps[-1]['dark']
    exit_dlt = delta_ink(last_ink, 0)
    if exit_dlt > 0:
        steps[-1]['delta'] += exit_dlt
        steps[-1]['cum'] += exit_dlt
        tot_delta += exit_dlt
        cum += exit_dlt

    # Width penalty: multiplicative (1 + width)
    all_cols = [c for _, c in cells]
    width = max(all_cols) - min(all_cols)

    mi = mid * sw + cl
    raw_cost = cf[mi] + cr[mi] - dark[mid, col]
    combined = raw_cost * (1.0 + width)
    return steps, cf[mi], cr[mi], combined, tot_ink, tot_delta


def step_color(s):
    if s['dark'] == 0:
        return COL_FREE
    if s['delta'] > 0:
        return COL_DELTA
    return COL_INK


def text_color_for_bg(orig_val):
    """Pick text color that's readable against the greyscale background."""
    return (255, 255, 255) if orig_val < 140 else (0, 0, 0)


def load_fonts():
    try:
        cell_font = ImageFont.truetype(FONT_PATH, CELL_FONT_SIZE)
    except (OSError, IOError):
        cell_font = ImageFont.load_default()
    try:
        label_font = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)
    except (OSError, IOError):
        label_font = cell_font
    try:
        title_font = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
    except (OSError, IOError):
        title_font = label_font
    return cell_font, label_font, title_font


def render_panel(dark, orig, col, seg_start, seg_end, cell_font, label_font,
                 is_vertical=False, audit_cost=None):
    h = dark.shape[0]
    mid = h // 2

    if is_vertical:
        # VP candidate: straight vertical line, per-row darkness + delta,
        # with run-length and row-ink discounts for serif/baseline bands.
        VERT_RUN_THRESHOLD = 11
        VERT_RUN_DISCOUNT = 0.02  # per-pixel rate
        VERT_ROW_INK_DIVISOR = 8.0
        w_full = dark.shape[1]
        row_ink = dark.sum(axis=1)
        max_row_ink = max(row_ink.max(), 1e-9)
        steps = []
        tot_ink = 0.0
        tot_delta = 0.0
        cum = 0.0
        prev_dark = 0.0  # boundary: no ink above
        for r in range(h):
            d = float(dark[r, col])
            # Measure horizontal dark run through this column
            run_len = 1
            cx = col - 1
            while cx >= 0 and dark[r, cx] > 0:
                run_len += 1
                cx -= 1
            cx = col + 1
            while cx < w_full and dark[r, cx] > 0:
                run_len += 1
                cx += 1
            # Run-length discount: long runs get scaled discount
            if run_len >= VERT_RUN_THRESHOLD:
                run_wt = max(0.1, 1.0 - VERT_RUN_DISCOUNT * (run_len - VERT_RUN_THRESHOLD + 1))
            else:
                run_wt = 1.0
            # Row-ink discount: high-ink rows discounted
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
        combined = audit_cost if audit_cost is not None else cum
        cf = combined
        cr = 0.0
    else:
        steps, cf, cr, combined, tot_ink, tot_delta = full_seam_path(
            dark, col, seg_start, seg_end)

    path_cols = [s['c'] for s in steps]
    c_min = max(0, min(path_cols) - PAD_COLS)
    c_max = min(orig.shape[1], max(path_cols) + PAD_COLS + 1)
    cw = c_max - c_min

    # Build lookup: (r,c) -> step
    step_map = {(s['r'], s['c']): s for s in steps}

    header_h = LABEL_FONT_SIZE + 8
    totals_h = LABEL_FONT_SIZE * 4 + 20
    panel_w = cw * SCALE
    panel_h = header_h + h * SCALE + totals_h

    img = Image.new('RGB', (panel_w, panel_h), COL_BG)
    draw = ImageDraw.Draw(img)

    # Header
    header = f"Col {col}: cost={combined:.0f}  ink={tot_ink:.0f}"
    draw.text((4, 2), header, fill=COL_LABEL, font=label_font)
    y_off = header_h

    # Draw all pixels as greyscale background
    for r in range(h):
        for c in range(c_min, c_max):
            v = int(orig[r, c])
            lc = c - c_min
            x0 = lc * SCALE
            y0 = y_off + r * SCALE
            draw.rectangle([x0, y0, x0 + SCALE - 1, y0 + SCALE - 1],
                           fill=(v, v, v))

    # Mid row line
    my = y_off + mid * SCALE + SCALE // 2
    draw.line([(0, my), (cw * SCALE, my)], fill=COL_MID, width=1)

    # Draw path pixels: colored outline + cost text inside cell
    for s in steps:
        lc = s['c'] - c_min
        x0 = lc * SCALE
        y0 = y_off + s['r'] * SCALE
        color = step_color(s)
        bg_val = int(orig[s['r'], s['c']])

        # Colored outline, 2px
        draw.rectangle([x0, y0, x0 + SCALE - 1, y0 + SCALE - 1],
                       outline=color, width=2)

        # Text inside the cell
        tc_ = text_color_for_bg(bg_val)

        has_wt = 'weight' in s and s['weight'] < 0.999
        ink_str = f"{s['dark']:.0f}"
        line1_w = cell_font.getlength(ink_str)
        tx = x0 + (SCALE - line1_w) / 2

        if has_wt:
            # VP cell with discount: show ink, delta, and weight
            lines = [ink_str]
            if s['delta'] > 0:
                lines.append(f"+{s['delta']:.0f}")
            lines.append(f"×{s['weight']:.2f}")
            n = len(lines)
            ty = y0 + (SCALE - n * (CELL_FONT_SIZE + 1)) / 2
            for i, line in enumerate(lines):
                lw = cell_font.getlength(line)
                lx = x0 + (SCALE - lw) / 2
                c_ = (255, 200, 80) if line.startswith('×') else color
                draw.text((lx, ty + i * (CELL_FONT_SIZE + 1)), line, fill=c_, font=cell_font)
        elif s['delta'] > 0:
            # Two lines: ink on top, delta below
            ty = y0 + (SCALE - 2 * CELL_FONT_SIZE - 2) / 2
            draw.text((tx, ty), ink_str, fill=color, font=cell_font)
            dlt_str = f"+{s['delta']:.0f}"
            dlt_w = cell_font.getlength(dlt_str)
            dx = x0 + (SCALE - dlt_w) / 2
            draw.text((dx, ty + CELL_FONT_SIZE + 2), dlt_str, fill=color, font=cell_font)
        elif s['dark'] > 0:
            # One line: ink centered
            ty = y0 + (SCALE - CELL_FONT_SIZE) / 2
            draw.text((tx, ty), ink_str, fill=color, font=cell_font)
        # else: free cell, outline only

    # Totals at bottom
    ty = y_off + h * SCALE + 8
    draw.text((4, ty), f"col={col}  cells={len(steps)}", fill=COL_LABEL, font=label_font)
    ty += LABEL_FONT_SIZE + 4
    draw.text((4, ty), f"ink={tot_ink:.0f}  Δ={tot_delta:.0f}  total={combined:.0f}",
              fill=COL_LABEL, font=label_font)
    ty += LABEL_FONT_SIZE + 4
    draw.text((4, ty), f"fwd={cf:.0f}  rev={cr:.0f}", fill=COL_TEXT, font=cell_font)

    return img, combined


def render(dark, orig, col_a, col_b, seg_start, seg_end, out_path,
           vp_a=False, vp_b=False, cost_a_audit=None, cost_b_audit=None,
           seg_a=None, seg_b=None):
    cell_font, label_font, title_font = load_fonts()
    ss_a, se_a = seg_a if seg_a else (seg_start, seg_end)
    ss_b, se_b = seg_b if seg_b else (seg_start, seg_end)

    panel_a, cost_a = render_panel(dark, orig, col_a, ss_a, se_a,
                                    cell_font, label_font,
                                    is_vertical=vp_a, audit_cost=cost_a_audit)
    panel_b, cost_b = render_panel(dark, orig, col_b, ss_b, se_b,
                                    cell_font, label_font,
                                    is_vertical=vp_b, audit_cost=cost_b_audit)

    title_h = TITLE_FONT_SIZE + 12
    legend_h = CELL_FONT_SIZE + 8
    header_h = title_h + legend_h

    total_w = panel_a.width + GAP + panel_b.width
    total_h = header_h + max(panel_a.height, panel_b.height)
    combined = Image.new('RGB', (total_w, total_h), COL_BG)
    draw = ImageDraw.Draw(combined)

    winner = col_a if cost_a < cost_b else col_b
    tag_a = "VP" if vp_a else "seam"
    tag_b = "VP" if vp_b else "seam"
    title = f"Col {col_a} [{tag_a}] ({cost_a:.0f}) vs col {col_b} [{tag_b}] ({cost_b:.0f})  —  winner: {winner}"
    draw.text((8, 4), title, fill=COL_LABEL, font=title_font)

    ly = title_h
    lx = 8
    for label, color in [("■ ink only", COL_INK), ("■ ink+Δ", COL_DELTA),
                          ("■ free", COL_FREE),
                          ("│ target", (0, 200, 0)), ("─ mid", COL_MID)]:
        draw.text((lx, ly), label, fill=color, font=cell_font)
        lx += int(cell_font.getlength(label)) + 16

    combined.paste(panel_a, (0, header_h))
    combined.paste(panel_b, (panel_a.width + GAP, header_h))

    combined.save(out_path)
    print(f"Written to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Seam path comparison visualizer")
    p.add_argument('image', help='word_crop.png path')
    p.add_argument('col_a', type=int, help='First seam column')
    p.add_argument('col_b', type=int, help='Second seam column')
    p.add_argument('--seg-start', type=int, default=None)
    p.add_argument('--seg-end', type=int, default=None)
    p.add_argument('--summary', default=None,
                   help='Path to summary.json for cost validation')
    p.add_argument('-o', '--output', default=None,
                   help='Output path (default: /tmp/seam_<a>_vs_<b>.png)')
    a = p.parse_args()

    img = Image.open(a.image)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    dark = 255.0 - arr.astype(float)
    w = arr.shape[1]
    ss = a.seg_start if a.seg_start is not None else max(0, min(a.col_a, a.col_b) - 30)
    se = a.seg_end if a.seg_end is not None else min(w, max(a.col_a, a.col_b) + 30)

    if a.summary:
        with open(a.summary) as f:
            summary = json.load(f)
        candidates = {c['col']: c for c in summary.get('seam_seed_candidates', [])}
        ok = True
        vp_flags = {}
        audit_costs = {}
        for col in [a.col_a, a.col_b]:
            if col not in candidates:
                print(f"WARN: col {col} not in seam_seed_candidates, skipping validation")
                vp_flags[col] = False
                continue
            cand = candidates[col]
            vp_flags[col] = cand.get('is_vertical', False)
            audit_costs[col] = cand.get('cost')
            if vp_flags[col]:
                print(f"VP col {col}: audit_cost={cand['cost']:.1f}")
                continue
            c_ss, c_se = cand['seg']
            steps, cf_, cr_, comb, ti, td = full_seam_path(dark, col, c_ss, c_se)
            audit_cost = cand['cost']
            if abs(comb - audit_cost) > 1.0:
                print(f"MISMATCH col {col}: python={comb:.1f} audit={audit_cost:.1f} "
                      f"(diff={comb - audit_cost:.1f})", file=sys.stderr)
                ok = False
            else:
                print(f"OK col {col}: python={comb:.1f} audit={audit_cost:.1f}")
        if not ok:
            print("VALIDATION WARNING: cost mismatch (scoring params changed since audit)", file=sys.stderr)
    else:
        vp_flags = {a.col_a: False, a.col_b: False}
        audit_costs = {}

    out = a.output or f'/tmp/seam_{a.col_a}_vs_{a.col_b}.png'
    seg_a_bounds = None
    seg_b_bounds = None
    if a.summary:
        for col, attr in [(a.col_a, 'seg_a_bounds'), (a.col_b, 'seg_b_bounds')]:
            if col in candidates:
                cand = candidates[col]
                bounds = tuple(cand['seg'])
                if col == a.col_a:
                    seg_a_bounds = bounds
                else:
                    seg_b_bounds = bounds
    render(dark, arr, a.col_a, a.col_b, ss, se, out,
           vp_a=vp_flags.get(a.col_a, False),
           vp_b=vp_flags.get(a.col_b, False),
           cost_a_audit=audit_costs.get(a.col_a),
           cost_b_audit=audit_costs.get(a.col_b),
           seg_a=seg_a_bounds,
           seg_b=seg_b_bounds)
