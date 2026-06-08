
## Blur-Matching for Real Scans (2026-05-11)

Real scanned documents have optical blur from the scanner glass, focus distance,
and paper texture. Currently SSIM compares a crisp vector render against a blurry
scan crop — this inherently caps the achievable SSIM score.

Idea: detect the blur kernel of the scanned text (e.g. via edge analysis or
frequency domain) and apply a matching blur to the rendered reference before
SSIM comparison. This would let SSIM focus on letterform shape differences
rather than penalizing "too sharp" renders.

The `gaussian_blur_3x3` function in `ssim.rs` applies a fixed 3×3 kernel to both
scan and render crops. An adaptive kernel matched to the actual scan quality
would be more accurate.
