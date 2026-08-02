#!/bin/bash
set -euo pipefail
if [ $# -lt 1 ] || [ $# -gt 2 ]; then echo "Usage: html2pdf.sh <input.html> [output.pdf]" >&2; exit 1; fi
INPUT="$1"; OUTPUT="${2:-${INPUT%.html}.pdf}"
CHROME=""; for c in google-chrome google-chrome-stable chromium-browser chromium /opt/meta-chromium/chrome /opt/google/chrome/chrome; do if command -v "$c" >/dev/null 2>&1; then CHROME="$c"; break; fi; if [ -x "$c" ]; then CHROME="$c"; break; fi; done
[ -z "$CHROME" ] && { echo "No chrome binary found" >&2; exit 1; }
export TMPDIR="${TMPDIR:-/home/hatch/workspace/tmp}"
export MALLOC_ARENA_MAX=1
export RAYON_NUM_THREADS=1
USER_DATA_DIR="${TMPDIR}/chrome-pdf-$$"; mkdir -p "$USER_DATA_DIR"
ulimit -H -v unlimited 2>/dev/null || true
ulimit -S -v unlimited 2>/dev/null || true
ulimit -H -d unlimited 2>/dev/null || true
ABS_INPUT="$(realpath -m "$INPUT" 2>/dev/null || echo "$INPUT")"
if [[ "$ABS_INPUT" != file://* && "$ABS_INPUT" != http://* && "$ABS_INPUT" != https://* ]]; then INPUT_URL="file://$ABS_INPUT"; else INPUT_URL="$ABS_INPUT"; fi
# Chrome-only, 30s virtual-time-budget version which reliably completes <1m for 75M HTML
# Use old headless mode which succeeded for 58M PDF, not headless=new
# Low-memory flags added to survive 7.8GB no-swap VM with 80M HTML (492 pages, many base64 PNGs)
# Unlimited VSZ required — previous 6GB clamp killed chrome renderer (1.4TB VSZ ghost)
"$CHROME" --headless --disable-gpu --no-sandbox \
    --disable-dev-shm-usage \
    --disable-software-rasterizer \
    --disable-extensions \
    --disable-background-networking \
    --disable-sync \
    --metrics-recording-only \
    --mute-audio \
    --no-first-run \
    --safebrowsing-disable-auto-update \
    --disable-dev-tools \
    --js-flags="--max-old-space-size=3072" \
    --virtual-time-budget=30000 \
    --user-data-dir="$USER_DATA_DIR" \
    --print-to-pdf="$OUTPUT" \
    --print-to-pdf-no-header \
    --run-all-compositor-stages-before-draw \
    "$INPUT_URL"
RET=$?; rm -rf "$USER_DATA_DIR" 2>/dev/null || true; [ $RET -eq 0 ] || exit $RET; echo "$OUTPUT"
