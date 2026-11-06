#!/bin/bash
# Regenerate all three specimens with the current binary.
# Run via pueue -g unprint from the repo root.
set -e
set -o pipefail
cd /home/hatch/workspace/repos/unscan

echo "=== 1. font-timeline-specimen ==="
python3 test-docs/gen-specimen.py 2>&1 | tail -8

echo "=== 2. line-test ==="
python3 test-docs/gen-line-test.py --hardcoded 2>&1 | tail -8

echo "=== 3. test-xpos-kern ==="
python3 test-docs/regen-xpos-kern.py 2>&1 | tail -8

echo "=== ALL DONE ==="
ls -la test-docs/font-timeline-specimen.pdf test-docs/font-timeline-specimen-rasterized.pdf test-docs/line-test.pdf test-docs/line-test-gt.pdf test-docs/test-xpos-kern.pdf test-docs/test-xpos-kern-rasterized.pdf
