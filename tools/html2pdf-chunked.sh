#!/bin/bash
set -euo pipefail
# Chunked HTML -> PDF via headless Chrome + pdfunite
# Handles huge 75M+ reports with 10k+ base64 PNGs by splitting into ~60 miss blocks per chunk.
# Usage: html2pdf-chunked.sh <input.html> <output.pdf> [chunk_size]
if [ $# -lt 2 ]; then echo "Usage: $0 <input.html> <output.pdf> [miss_per_chunk=60]" >&2; exit 1; fi
INPUT="$1"
OUTPUT="$2"
PER_CHUNK="${3:-60}"

CHROME=""
for c in google-chrome google-chrome-stable chromium-browser chromium /opt/meta-chromium/chrome /opt/google/chrome/chrome; do
  if command -v "$c" >/dev/null 2>&1; then CHROME="$c"; break; fi
  if [ -x "$c" ]; then CHROME="$c"; break; fi
done
[ -z "$CHROME" ] && { echo "No chrome binary found" >&2; exit 1; }

export TMPDIR="${TMPDIR:-/home/hatch/workspace/tmp}"
export MALLOC_ARENA_MAX=1
export RAYON_NUM_THREADS=1
mkdir -p "$TMPDIR"

ABS_INPUT="$(realpath -m "$INPUT" 2>/dev/null || echo "$INPUT")"
WORKDIR="$(mktemp -d "${TMPDIR}/chunked-pdf-XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Chunking $INPUT -> $WORKDIR (per_chunk=$PER_CHUNK)"

python3 - "$ABS_INPUT" "$WORKDIR" "$PER_CHUNK" << 'PY'
import sys, re, pathlib, os
inp = sys.argv[1]
workdir = pathlib.Path(sys.argv[2])
per_chunk = int(sys.argv[3])

html = pathlib.Path(inp).read_text(encoding='utf-8', errors='ignore')

# Find prefix up to first <h2>Major Misses
m = re.search(r'<h2>Major Misses', html)
if not m:
    print("No Major Misses header found, treating whole file as single chunk", file=sys.stderr)
    header_end = 0
    prefix = html[:0]
else:
    header_end = m.start()
    prefix = html[:header_end]

# suffix is closing tags
suffix_match = re.search(r'</body>\s*</html>\s*\Z', html, re.DOTALL|re.IGNORECASE)
suffix = suffix_match.group(0) if suffix_match else "\n</body></html>"

# rest from header_end to before suffix
rest_start = header_end
rest_end = len(html) - len(suffix) if suffix_match else len(html)
rest = html[rest_start:rest_end]

# Tokenize rest into alternating h2 and miss blocks
tokens = []  # list of (type, html)
pos = 0
# precompile patterns
h2_pat = re.compile(r'<h2[^>]*>.*?</h2>', re.DOTALL|re.IGNORECASE)
miss_pat = re.compile(r'<div class="miss"', re.IGNORECASE)

while pos < len(rest):
    h2_m = h2_pat.search(rest, pos)
    miss_m = miss_pat.search(rest, pos)
    # determine which comes first
    next_h2 = h2_m.start() if h2_m else None
    next_miss = miss_m.start() if miss_m else None
    if next_h2 is None and next_miss is None:
        # trailing content
        tail = rest[pos:].strip()
        if tail:
            tokens.append(('tail', tail))
        break
    if next_h2 is not None and (next_miss is None or next_h2 < next_miss):
        # h2 header token
        tokens.append(('h2', h2_m.group(0)))
        pos = h2_m.end()
    else:
        # miss token: from this miss start to next h2 or next miss or end
        start = next_miss
        # find next boundary after start+1
        h2_next = h2_pat.search(rest, start+1)
        miss_next = miss_pat.search(rest, start+1)
        candidates = []
        if h2_next: candidates.append(h2_next.start())
        if miss_next: candidates.append(miss_next.start())
        if candidates:
            end = min(candidates)
        else:
            end = len(rest)
        block = rest[start:end]
        tokens.append(('miss', block))
        pos = end

# Now chunk misses
chunks = []
cur_headers = []
cur_misses = []
cur_h2 = None
last_seen_h2 = None

def flush_chunk():
    global cur_headers, cur_misses, chunks, last_seen_h2
    if not cur_misses and not cur_headers:
        return
    parts = [prefix]
    for h in cur_headers:
        parts.append(h)
    if not cur_headers and last_seen_h2:
        parts.append(last_seen_h2)
    for mb in cur_misses:
        parts.append(mb)
    parts.append(suffix)
    chunks.append('\n'.join(parts))
    cur_headers = []
    cur_misses = []

for typ, content in tokens:
    if typ == 'h2':
        # If current chunk already has misses, and we encounter new h2, we should flush or keep h2 in same chunk if not full
        # If adding this h2 would not overflow, keep it in current chunk
        # Otherwise flush current and start new with this h2
        if len(cur_misses) >= per_chunk:
            flush_chunk()
        cur_headers.append(content)
        last_seen_h2 = content
    elif typ == 'miss':
        cur_misses.append(content)
        if len(cur_misses) >= per_chunk:
            flush_chunk()
    else: # tail
        cur_headers.append(content)

if cur_misses or cur_headers:
    flush_chunk()

print(f"Total tokens: {len(tokens)} misses={sum(1 for t,_ in tokens if t=='miss')} chunks={len(chunks)}", file=sys.stderr)
for i, ch in enumerate(chunks):
    out = workdir / f"chunk-{i:03d}.html"
    out.write_text(ch, encoding='utf-8')
PY

echo "Generated $(ls -1 "$WORKDIR"/chunk-*.html | wc -l) chunks"

# Render each chunk via Chrome
set +e
CHUNK_PDFS=()
idx=0
for html in "$WORKDIR"/chunk-*.html; do
  pdf="$WORKDIR/chunk-$(printf "%03d" $idx).pdf"
  echo "Rendering chunk $idx: $html -> $pdf"
  USER_DATA_DIR="${TMPDIR}/chrome-pdf-chunk-$$-$idx"
  mkdir -p "$USER_DATA_DIR"
  ulimit -H -v unlimited 2>/dev/null || true
  ulimit -S -v unlimited 2>/dev/null || true
  # Use old headless, low-mem flags, 20s virtual budget per chunk
  "$CHROME" --headless --disable-gpu --no-sandbox --disable-dev-shm-usage --print-to-pdf="$pdf" --print-to-pdf-no-header "file://$(realpath -m "$html")" 2> "$WORKDIR/chrome-$idx.log"
  RET=$?
  rm -rf "$USER_DATA_DIR" 2>/dev/null || true
  if [ $RET -ne 0 ] || [ ! -f "$pdf" ]; then
    echo "Chunk $idx failed RET=$RET, log tail:"
    tail -n 30 "$WORKDIR/chrome-$idx.log" || true
    # retry once with more budget
    echo "Retrying chunk $idx with 30s budget..."
    mkdir -p "$USER_DATA_DIR"
    "$CHROME" --headless --disable-gpu --no-sandbox --disable-dev-shm-usage --print-to-pdf="$pdf" --print-to-pdf-no-header "file://$(realpath -m "$html")" 2> "$WORKDIR/chrome-${idx}-retry.log" || true
    rm -rf "$USER_DATA_DIR" 2>/dev/null || true
  fi
  if [ ! -f "$pdf" ]; then
    echo "ERROR: chunk $idx still no PDF after retry" >&2
    exit 1
  fi
  ls -lh "$pdf"
  CHUNK_PDFS+=("$pdf")
  idx=$((idx+1))
done
set -e

echo "Merging ${#CHUNK_PDFS[@]} PDFs -> $OUTPUT"
pdfunite "${CHUNK_PDFS[@]}" "$OUTPUT"
ls -lh "$OUTPUT"
pdfinfo "$OUTPUT" | head -n 20
echo "$OUTPUT"
