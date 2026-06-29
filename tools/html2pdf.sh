#!/bin/bash
# Convert HTML to PDF via headless Chrome with no network access.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: html2pdf.sh <input.html> [output.pdf]" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.html}.pdf}"

google-chrome --headless --disable-gpu --no-sandbox \
    --host-resolver-rules="MAP * ~NOTFOUND" \
    --print-to-pdf="$OUTPUT" "$INPUT" 2>/dev/null

echo "$OUTPUT"
