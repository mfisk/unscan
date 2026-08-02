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
# Working minimal flag set – verified 58M 470p success 2026-08-02 11:16 EDT:
# old headless, no virtual-time-budget, no run-all-compositor-stages-before-draw, no headless=new
# Monitor: ~2.9GB RSS then finish, Skia/PDF m146, HeadlessChrome/146.0.0.0
"$CHROME" --headless --disable-gpu --no-sandbox --disable-dev-shm-usage --print-to-pdf="$OUTPUT" --print-to-pdf-no-header "$INPUT_URL"
RET=$?; rm -rf "$USER_DATA_DIR" 2>/dev/null || true; [ $RET -eq 0 ] || exit $RET; echo "$OUTPUT"
