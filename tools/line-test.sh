#!/bin/bash
# Generate a single-line test PDF and run a quick lob on it.
# Usage: tools/line-test.sh <page> <line> [<line2> ...]
#
# Uses gen-line-test.py to create proper vector+rasterized PDFs from scratch
# (same font infrastructure as gen-specimen.py), avoiding coordinate-offset
# bugs from mediabox cropping.  Requires a prior BAP audit in test-docs/audit/.
#
# Examples:
#   tools/line-test.sh 1 73
#   tools/line-test.sh 1 72 73

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then echo "Usage: tools/line-test.sh <page> <line> [<line2> ...]"; exit 1; fi

AUDIT_DIR="test-docs/line-test-audit"

# Generate test PDFs from scratch
python3 test-docs/gen-line-test.py "$@"

# Copy to the filenames unprint expects
cp test-docs/line-test-gt.pdf test-docs/line-test-seams-gt.pdf
cp test-docs/line-test.pdf test-docs/line-test-seams.pdf

# Clear page cache for these PDFs
rm -rf /tmp/unprint-page-cache/line-test-seams*

# Build debug binary if needed
export PATH="$HOME/.cargo/bin:$PATH"
if [ ! -f target/debug/unprint ] || [ "$(find src -newer target/debug/unprint -name '*.rs' | head -1)" ]; then
    echo "Building debug binary..."
    AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    ulimit -v $((AVAIL_KB * 80 / 100))
    cargo build --bin unprint
fi

# Run unprint on the generated test PDFs
echo "Running unprint on line-test..."
./target/debug/unprint -o /dev/null \
    --test test-docs/line-test-seams-gt.pdf \
    --audit "$AUDIT_DIR" \
    test-docs/line-test-seams.pdf
