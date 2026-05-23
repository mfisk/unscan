#!/usr/bin/env python3
"""
Fisher discriminant weight analysis for unscan CI features.
Deduplicates per character — identical feature vectors collapse so OT variants
that render the same glyph don't dilute between-font variance. OT variants
that render differently (e.g., onum 'g' vs default 'g') are preserved.
"""
import numpy as np
import sys

FEAT_NAMES = [
    "prof0","prof1","prof2","prof3","prof4","prof5","prof6","prof7",
    "prof8","prof9","prof10","prof11","prof12","prof13","prof14","prof15",
    "prof16","prof17","prof18","prof19","prof20","prof21","prof22","prof23",
    "prof24","prof25","prof26","prof27","prof28","prof29","prof30","prof31",
    "aspect","ink_density","v_center","h_balance","serif_score","stroke_contrast","xh_cap_ratio",
    "counter_area","counter_cx","counter_cy","counter_asp",
    "term0","term1","term2","term3",
    "ink_perim","compactness",
    "cross0","cross1","cross2","cross3","cross4","cross5","cross6","cross7",
]
N = len(FEAT_NAMES)

def load_tsv(path):
    fonts = []
    chars = []
    feats = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < N + 2:
                continue
            feat_vals = parts[-N:]
            ch = parts[-(N+1)]
            font = '\t'.join(parts[:-(N+1)])
            try:
                fv = [float(x) for x in feat_vals]
            except ValueError:
                continue
            fonts.append(font)
            chars.append(ch)
            feats.append(fv)
    return fonts, chars, np.array(feats, dtype=np.float32)

print("Loading index features...", file=sys.stderr)
idx_fonts, idx_chars, idx_feats = load_tsv('/tmp/index_features.tsv')
print(f"  {len(idx_fonts)} entries, {len(set(idx_fonts))} fonts, {len(set(idx_chars))} chars", file=sys.stderr)

print("Loading scan features...", file=sys.stderr)
scan_files, scan_chars, scan_feats = load_tsv('/tmp/scan_features.tsv')
print(f"  {len(scan_files)} entries", file=sys.stderr)

# ── Dedup per character ──
# For each char, round features to 4 decimal places and dedup.
# This collapses identical OT variants while keeping genuinely different ones.
print("\nDeduplicating per character...", file=sys.stderr)
idx_chars_arr = np.array(idx_chars)
unique_chars = sorted(set(idx_chars))

total_before = 0
total_after = 0
# Build deduped index: char -> unique feature vectors
char_deduped = {}
for ch in unique_chars:
    mask = idx_chars_arr == ch
    ch_feats = idx_feats[mask]
    total_before += len(ch_feats)
    # Round to 4 decimals for dedup (handles float noise)
    rounded = np.round(ch_feats, 4)
    # Use tuple of rounded values as hash key
    seen = set()
    unique_rows = []
    for i, row in enumerate(rounded):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique_rows.append(ch_feats[i])  # keep original precision
    char_deduped[ch] = np.array(unique_rows)
    total_after += len(unique_rows)

print(f"  Before: {total_before} entries", file=sys.stderr)
print(f"  After:  {total_after} entries ({total_before - total_after} duplicates removed)", file=sys.stderr)
print(f"  Reduction: {(1 - total_after/total_before)*100:.1f}%", file=sys.stderr)

# ── 1. Between-font variance per feature (signal) ──
print("\nComputing between-font variance (signal) on deduped data...", file=sys.stderr)
char_vars = []
for ch, ch_feats in char_deduped.items():
    if len(ch_feats) < 10:
        continue
    char_vars.append(np.var(ch_feats, axis=0))
signal = np.mean(char_vars, axis=0)

# ── 2. Scan-vs-index noise ──
print("Computing scan-index noise...", file=sys.stderr)
# Build char->index lookup (use ALL index entries, not deduped — we want real nearest)
char_to_idx = {}
for i, ch in enumerate(idx_chars):
    char_to_idx.setdefault(ch, []).append(i)

noise_sq = np.zeros(N, dtype=np.float64)
n_noise = 0
for i in range(len(scan_files)):
    ch = scan_chars[i]
    if ch not in char_to_idx:
        continue
    indices = char_to_idx[ch]
    ch_idx_feats = idx_feats[indices]
    scan_f = scan_feats[i]
    diffs = ch_idx_feats - scan_f[np.newaxis, :]
    dists = (diffs ** 2).sum(axis=1)
    nearest = np.argmin(dists)
    noise_sq += diffs[nearest] ** 2
    n_noise += 1
noise = noise_sq / max(n_noise, 1)

# ── 3. Fisher ratio and optimal weights ──
fisher = signal / (noise + 1e-12)
raw_weights = np.sqrt(np.maximum(fisher, 0))
norm_weights = raw_weights / (raw_weights.sum() + 1e-12)

# ── 4. Output ──
print("\n" + "="*95)
print("FISHER DISCRIMINANT ANALYSIS — OPTIMAL FEATURE WEIGHTS (DEDUPED)")
print("="*95)
print(f"\n{'rank':>4} {'dim':>4} {'name':>16} {'signal':>10} {'noise':>10} {'fisher':>10} {'opt_wt':>8} {'cur_grp':>10}")
print("-"*82)

order = np.argsort(-fisher)
for rank, i in enumerate(order):
    name = FEAT_NAMES[i]
    if i < 32: cur = "prof"
    elif i < 39: cur = "scal"
    else: cur = "v2"
    print(f"{rank+1:>4} {i:>4} {name:>16} {signal[i]:>10.6f} {noise[i]:>10.6f} {fisher[i]:>10.2f} {norm_weights[i]:>8.4f} {cur:>10}")

print("\n" + "="*95)
print("GROUP WEIGHT COMPARISON — CURRENT vs OPTIMAL (DEDUPED)")
print("="*95)
prof_wt = norm_weights[:32].sum()
scal_wt = norm_weights[32:39].sum()
v2_wt = norm_weights[39:].sum()
print(f"  {'Profile (32 bins)':25s}  current: 0.4000  optimal: {prof_wt:.4f}")
print(f"  {'Scalars (7)':25s}  current: 0.3000  optimal: {scal_wt:.4f}")
print(f"  {'V2 features (18)':25s}  current: 0.3000  optimal: {v2_wt:.4f}")

print("\n" + "="*95)
print("SCALAR FEATURES")
print("="*95)
for j in range(7):
    i = 32 + j
    print(f"  {FEAT_NAMES[i]:>16}  signal={signal[i]:.6f}  noise={noise[i]:.6f}  fisher={fisher[i]:.2f}  wt={norm_weights[i]:.4f}")

# Emit weights as Rust array for copy-paste
print("\n" + "="*95)
print("RUST WEIGHTS ARRAY")
print("="*95)
print("const FISHER_WEIGHTS: [f32; FEAT_LEN] = [")
for start, end, label in [(0,32,"Profile"), (32,39,"Scalars"), (39,57,"V2")]:
    vals = ", ".join(f"{norm_weights[i]:.4f}" for i in range(start, end))
    print(f"    // {label}")
    print(f"    {vals},")
print("];")
