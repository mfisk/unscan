#!/bin/bash
# Convert HTML to PDF via headless Chrome — landscape A0, no network access.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: html2pdf.sh <input.html> [output.pdf]" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.html}.pdf}"

# Find chrome binary
CHROME=""
for c in google-chrome google-chrome-stable chromium-browser chromium /opt/meta-chromium/chrome /opt/google/chrome/chrome; do
    if command -v "$c" >/dev/null 2>&1; then CHROME="$c"; break; fi
    if [ -x "$c" ]; then CHROME="$c"; break; fi
done
if [ -z "$CHROME" ]; then
    echo "No chrome binary found" >&2
    exit 1
fi

USER_DATA_DIR="${TMPDIR:-/home/hatch/workspace/tmp}/chrome-pdf-$$"
mkdir -p "$USER_DATA_DIR"

"$CHROME" --headless --disable-gpu --no-sandbox \
    --user-data-dir="$USER_DATA_DIR" \
    --disable-dev-shm-usage \
    --host-resolver-rules="MAP * ~NOTFOUND" \
    --print-to-pdf="$OUTPUT" \
    --print-to-pdf-no-header \
    --run-all-compositor-stages-before-draw \
    --virtual-time-budget=15000 \
    "$INPUT" 2>/dev/null
RET=$?
rm -rf "$USER_DATA_DIR" 2>/dev/null || true
if [ $RET -ne 0 ]; then
  exit $RET
fi

echo "$OUTPUT"
