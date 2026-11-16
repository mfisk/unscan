#!/usr/bin/env python3
"""
Generate zero-centered histograms with flat-top scoring overlay.
- 100 bins, bw=0.03 px, edges ...-0.045,-0.015,0.015,0.045..., center bin [-0.015,0.015]
- Input: audit.json path (expects text_entries[].obs_votes[].chosen_geo_{h,v}_{err,ll})
- Output: PNG with v and h histograms + flat-top and Gaussian curves
- Computes sv, sh from filtered data (abs err <1.5 px) matching prior sv=0.2486 sh=0.49799 methodology
"""
import json, math, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

AUDIT = sys.argv[1] if len(sys.argv)>1 else "test-docs/audit-flat-0.45/audit.json"
OUT = sys.argv[2] if len(sys.argv)>2 else str(pathlib.Path.home()/"workspace/your_files/hist-0.45/hist-centered-zero-flat-top-20260729-fresh.png")

def erf_phi(x):
    # Phi(x) = 0.5*(1+erf(x/sqrt2))
    return 0.5*(1.0+math.erf(x*0.7071067811865475))

def quantized_ll(e, sigma, a):
    sigma = max(sigma,1e-12)
    upper = (e+a)/sigma
    lower = (e-a)/sigma
    phi_u = erf_phi(upper)
    phi_l = erf_phi(lower)
    prob = max(phi_u-phi_l, 1e-300)
    return math.log(prob) - math.log(2*a)

def gaussian_ll(e, sigma):
    return -e*e/(2*sigma*sigma)

data=json.load(open(AUDIT))
v_errs=[]
h_errs=[]
total_obs=0
total_geo=0
for te in data["text_entries"]:
    for v in te.get("obs_votes",[]):
        total_obs+=1
        # GT-only, no fallback: midpoint jitter stats must only use GT fonts.
        # If GT cannot render its own char, that observation is invalid for jitter;
        # falling back to chosen (which may be a different font that lacks the glyph)
        # would pollute GT stats with missing-glyph bias. So skip entirely when GT missing.
        ve=v.get("gt_geo_v_err")
        he=v.get("gt_geo_h_err")
        if ve is None and he is None:
            continue
        total_geo+=1
        if ve is not None:
            v_errs.append(ve)
        if he is not None:
            h_errs.append(he)

print(f"total_obs={total_obs} total_geo={total_obs and total_geo} pct={total_geo/max(1,total_obs)*100:.2f}%")
print(f"v count {len(v_errs)} h count {len(h_errs)}")

# filtered for sigma computation: |err|<1.5 as in prior analysis giving sv~0.255 sh~0.447
def sd(arr):
    m=sum(arr)/len(arr)
    return math.sqrt(sum((x-m)**2 for x in arr)/len(arr)), m

vf=[x for x in v_errs if abs(x)<1.5]
hf=[x for x in h_errs if abs(x)<1.5]
sv,_=sd(vf); sh,_=sd(hf)
print(f"filtered |<1.5 n_v={len(vf)} n_h={len(hf)} sv={sv:.5f} sh={sh:.5f}")

# Use tuned sigma for flat-top curves: 0.284, 0.435 as per code, but also report empirical
SIGMA_V_TUNED=0.284
SIGMA_H_TUNED=0.435
A_DEFAULT=0.45

# Binning: 100 bins, bw=0.03, zero-centered
bw=0.03
nbins=100
# edges = (i+0.5)*bw for i in range(-50,50) ??? gives 100 bins from -1.485 to 1.515?
# We want edges ...-0.045,-0.015,0.015,0.045... center bin [-0.015,0.015]
# So edges = k*bw + bw/2 for k=-50..49 gives 100 bins? Let's compute:
edges = np.array([(i+0.5)*bw for i in range(-50,50)], dtype=float)  # 100 edges? Actually need 101 edges for 100 bins
# Need 101 edges for 100 bins: i=-50..50 inclusive =101 values
edges = np.array([(i+0.5)*bw for i in range(-50,51)], dtype=float)  # -1.485 to 1.515 step 0.03
# Actually center bin is [-0.015,0.015] which is edges[50]= -0.015? Let's verify:
# i=0 => 0.015, i=-1 => -0.015, so bin between them is center. Good.
assert len(edges)==101

# histograms
v_hist, _ = np.histogram(v_errs, bins=edges, density=True)
h_hist, _ = np.histogram(h_errs, bins=edges, density=True)
centers = (edges[:-1]+edges[1:])/2

# curves
vs = np.linspace(-1.5,1.5,400)
def flat_curve(es, sigma):
    return [math.exp(quantized_ll(e, sigma, A_DEFAULT)) for e in es]  # prob density (not log) for visualization? Use ll scaled
# For display, we want log-likelihood shape or pdf? Use exp(ll) which is prob/(2a)
# Gaussian pdf for comparison: exp(-e^2/(2s^2))/(sqrt(2pi)s) but our Gaussian_ll was unnormalized -e^2/(2s^2). Use normalized for overlay.
def gauss_pdf(e, sigma):
    return math.exp(-e*e/(2*sigma*sigma))/(math.sqrt(2*math.pi)*sigma)

v_gauss = [gauss_pdf(e, SIGMA_V_TUNED) for e in centers]
h_gauss = [gauss_pdf(e, SIGMA_H_TUNED) for e in centers]
v_flat = [math.exp(quantized_ll(e, SIGMA_V_TUNED, A_DEFAULT)) for e in centers]
h_flat = [math.exp(quantized_ll(e, SIGMA_H_TUNED, A_DEFAULT)) for e in centers]

fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True)
ax[0].bar(centers, v_hist, width=bw*0.9, alpha=0.5, label=f"v_err n={len(v_errs)}")
ax[0].plot(centers, v_gauss, label=f"Gaussian sigma_v={SIGMA_V_TUNED}")
ax[0].plot(centers, v_flat, label=f"Flat-top a={A_DEFAULT} sigma_v={SIGMA_V_TUNED}")
ax[0].set_ylabel("density v")
ax[0].legend()
ax[0].set_title(f"Zero-centered bw={bw} bins={nbins} sv_emp={sv:.4f} (|<1.5 n={len(vf)})")

ax[1].bar(centers, h_hist, width=bw*0.9, alpha=0.5, label=f"h_err n={len(h_errs)}", color="orange")
ax[1].plot(centers, h_gauss, label=f"Gaussian sigma_h={SIGMA_H_TUNED}")
ax[1].plot(centers, h_flat, label=f"Flat-top a={A_DEFAULT} sigma_h={SIGMA_H_TUNED}")
ax[1].set_ylabel("density h")
ax[1].set_xlabel("error px")
ax[1].legend()

plt.tight_layout()
out_path=pathlib.Path(OUT)
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200)
print(f"Wrote {out_path} bw={bw} nbins={nbins} edges zero-centered center=[-0.015,0.015] sv={sv:.5f} sh={sh:.5f}")
