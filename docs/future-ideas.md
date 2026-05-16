
## Blur-Matching for Real Scans (2026-05-11)

Real scanned documents have optical blur from the scanner glass, focus distance,
and paper texture. Currently SSIM compares a crisp vector render against a blurry
scan crop — this inherently caps the achievable SSIM score.

Idea: detect the blur kernel of the scanned text (e.g. via edge analysis or
frequency domain) and apply a matching blur to the rendered reference before
SSIM comparison. This would let SSIM focus on letterform shape differences
rather than penalizing "too sharp" renders.

The gaussian_blur_3x3 and related functions in font_match.rs (currently dead
after removing the coarse scorer) could be repurposed for this. Keep them around
or rewrite as needed when tackling real scans.

Related: the verify.rs SSIM pipeline already applies gaussian blur to both
scan crop and rendered reference, but uses a fixed kernel. An adaptive kernel
matched to the actual scan quality would be more accurate.
