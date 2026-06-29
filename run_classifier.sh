#!/bin/bash
# Usage: ./run_classifier.sh <classifier_name>
set -e
CLASSIFIER="$1"
export PATH="$HOME/.cargo/bin:$PATH"
AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
ulimit -v $((AVAIL_KB * 80 / 100))

cd ~/workspace/unscan
OUTDIR="$HOME/workspace/your_files/classifier-comparison"

echo "=== Running classifier: $CLASSIFIER ==="
START=$(date +%s)

# Capture stdout (JSON perf data) separately from stderr (training logs)
./target/release/unprint --classifier "$CLASSIFIER" --test test-docs/font-timeline-specimen.pdf --audit . test-docs/font-timeline-specimen.pdf 2>/tmp/unprint_stderr.txt | tee "$OUTDIR/results-${CLASSIFIER}.json"

END=$(date +%s)
ELAPSED=$((END - START))
echo "" >&2
echo "=== Elapsed: ${ELAPSED}s ===" >&2

cp report.html "$OUTDIR/audit-${CLASSIFIER}.html"
cp audit.json "$OUTDIR/audit-${CLASSIFIER}.json"

echo "=== Training stderr (last 15 lines) ===" >&2
tail -15 /tmp/unprint_stderr.txt >&2
echo "=== Results ===" >&2
cat "$OUTDIR/results-${CLASSIFIER}.json" >&2
