#!/bin/bash
set -euo pipefail
# WeasyPrint chunked HTML -> PDF for huge 75M+ reports
# Splits on <div class="miss" and renders each chunk with WeasyPrint, then merges via pdfunite
# Usage: html2pdf-weasy-chunked.sh <input.html> <output.pdf> [miss_per_chunk=30]

if [ $# -lt 2 ]; then echo "Usage: $0 <input.html> <output.pdf> [miss_per_chunk=30]" >&2; exit 1; fi
INPUT="$1"
OUTPUT="$2"
PER_CHUNK="${3:-30}"

export TMPDIR="${TMPDIR:-/home/hatch/workspace/tmp}"
export MALLOC_ARENA_MAX=1
export RAYON_NUM_THREADS=1
mkdir -p "$TMPDIR"

ABS_INPUT="$(realpath -m "$INPUT" 2>/dev/null || echo "$INPUT")"
WORKDIR="$(mktemp -d "${TMPDIR}/weasy-chunked-XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Chunking $INPUT -> $WORKDIR (per_chunk=$PER_CHUNK)"

python3 - "$ABS_INPUT" "$WORKDIR" "$PER_CHUNK" << 'PY'
import sys, re, pathlib
inp = sys.argv[1]
workdir = pathlib.Path(sys.argv[2])
per_chunk = int(sys.argv[3])

html = pathlib.Path(inp).read_text(encoding='utf-8', errors='ignore')

m = re.search(r'<h2>Major Misses', html)
if not m:
    header_end = 0
    prefix = html[:0]
else:
    header_end = m.start()
    prefix = html[:header_end]

suffix_match = re.search(r'</body>\s*</html>\s*\Z', html, re.DOTALL|re.IGNORECASE)
suffix = suffix_match.group(0) if suffix_match else "\n</body></html>"

rest_start = header_end
rest_end = len(html) - len(suffix) if suffix_match else len(html)
rest = html[rest_start:rest_end]

tokens = []
h2_pat = re.compile(r'<h2[^>]*>.*?</h2>', re.DOTALL|re.IGNORECASE)
miss_pat = re.compile(r'<div class="miss"', re.IGNORECASE)
pos = 0
while pos < len(rest):
    h2_m = h2_pat.search(rest, pos)
    miss_m = miss_pat.search(rest, pos)
    next_h2 = h2_m.start() if h2_m else None
    next_miss = miss_m.start() if miss_m else None
    if next_h2 is None and next_miss is None:
        tail = rest[pos:].strip()
        if tail:
            tokens.append(('tail', tail))
        break
    if next_h2 is not None and (next_miss is None or next_h2 < next_miss):
        tokens.append(('h2', h2_m.group(0)))
        pos = h2_m.end()
    else:
        start = next_miss
        h2_next = h2_pat.search(rest, start+1)
        miss_next = miss_pat.search(rest, start+1)
        candidates = []
        if h2_next: candidates.append(h2_next.start())
        if miss_next: candidates.append(miss_next.start())
        end = min(candidates) if candidates else len(rest)
        block = rest[start:end]
        tokens.append(('miss', block))
        pos = end

chunks = []
cur_headers = []
cur_misses = []
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
        if len(cur_misses) >= per_chunk:
            flush_chunk()
        cur_headers.append(content)
        last_seen_h2 = content
    elif typ == 'miss':
        cur_misses.append(content)
        if len(cur_misses) >= per_chunk:
            flush_chunk()
    else:
        cur_headers.append(content)

if cur_misses or cur_headers:
    flush_chunk()

print(f"Total tokens: {len(tokens)} misses={sum(1 for t,_ in tokens if t=='miss')} chunks={len(chunks)}", file=sys.stderr)
for i, ch in enumerate(chunks):
    out = workdir / f"chunk-{i:03d}.html"
    out.write_text(ch, encoding='utf-8')
PY

NUM_CHUNKS=$(ls -1 "$WORKDIR"/chunk-*.html 2>/dev/null | wc -l)
echo "Generated $NUM_CHUNKS chunks"
if [ "$NUM_CHUNKS" -eq 0 ]; then echo "No chunks generated" >&2; exit 1; fi

set +e
CHUNK_PDFS=()
idx=0
for html in "$WORKDIR"/chunk-*.html; do
  pdf="$WORKDIR/chunk-$(printf "%03d" $idx).pdf"
  echo "Rendering chunk $idx: $(stat -c%s "$html" 2>/dev/null || echo ?) bytes -> $pdf"
  # WeasyPrint with low memory settings
  weasyprint "$html" "$pdf" 2> "$WORKDIR/weasy-$idx.log"
  RET=$?
  if [ $RET -ne 0 ] || [ ! -f "$pdf" ]; then
    echo "Chunk $idx failed RET=$RET, log tail:"
    tail -n 40 "$WORKDIR/weasy-$idx.log" || true
    echo "ERROR: chunk $idx no PDF" >&2
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
