#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-test-docs/audit}"
PREFIX="${2:-report}"
TS="$(date +%Y%m%d-%H%M%S)"
OUTBASE="${HOME}/workspace/your_files"
# fallback if HOME not set
if [ ! -d "$OUTBASE" ]; then OUTBASE="/home/hatch/workspace/your_files"; fi
OUTDIR="${OUTBASE}/${PREFIX}-${TS}"
mkdir -p "$OUTDIR"

if [ -f "${SRC}/report.html" ]; then
  cp "${SRC}/report.html" "${OUTDIR}/index.html"
elif [ -f "${SRC}/index.html" ]; then
  cp "${SRC}/index.html" "${OUTDIR}/index.html"
else
  echo "publish-report: no report.html in $SRC" >&2; exit 2
fi

# copy assets
for d in images font_refs; do
  if [ -d "${SRC}/${d}" ]; then
    cp -a "${SRC}/${d}" "${OUTDIR}/"
  fi
done

# copy p dirs (p0_*, p1_*, p[0-9]_L* etc)
shopt -s nullglob
pdirs=()
for p in "${SRC}"/p[0-9]*; do
  [ -e "$p" ] || continue
  cp -a "$p" "${OUTDIR}/"
  pdirs+=("$(basename "$p")")
done
shopt -u nullglob

# enforce 0 base64
BASE64_COUNT=$(grep -c "base64" "${OUTDIR}/index.html" || true)
if [ "$BASE64_COUNT" -ne 0 ]; then
  echo "FAIL: index.html contains $BASE64_COUNT base64 occurrences (expected 0)" >&2
  exit 3
fi

IMG_COUNT=0; FONTREF_COUNT=0
[ -d "${OUTDIR}/images" ] && IMG_COUNT=$(find "${OUTDIR}/images" -type f | wc -l | tr -d ' ')
[ -d "${OUTDIR}/font_refs" ] && FONTREF_COUNT=$(find "${OUTDIR}/font_refs" -type f | wc -l | tr -d ' ')

PDIR_COUNT=$(find "${OUTDIR}" -maxdepth 1 -type d -name 'p[0-9]*' | wc -l | tr -d ' ')

# check missing src refs
MISSING=0
# extract src="..."
# ignore http, data:, //
TMP_REFS=$(grep -oE 'src="[^"]+"' "${OUTDIR}/index.html" | sed -E 's/src="//;s/"$//' | sort -u || true)
while IFS= read -r ref; do
  [ -z "$ref" ] && continue
  case "$ref" in http://*|https://*|data:*|//*) continue;; esac
  # strip query/fragment and leading ./
  clean=$(echo "$ref" | cut -d'?' -f1 | cut -d'#' -f1 | sed 's#^\./##')
  if [ ! -e "${OUTDIR}/${clean}" ]; then
    echo "missing ref: $clean" >&2
    MISSING=$((MISSING+1))
  fi
done <<< "$TMP_REFS"

SIZE_KB=$(du -sh "$OUTDIR" | cut -f1)
INDEX_BYTES=$(stat -c%s "${OUTDIR}/index.html" 2>/dev/null || wc -c < "${OUTDIR}/index.html")

echo "PUBLISH OK $OUTDIR"
echo "  size: $SIZE_KB  index: ${INDEX_BYTES} bytes"
echo "  images: $IMG_COUNT  font_refs: $FONTREF_COUNT  pdirs: $PDIR_COUNT"
echo "  base64: $BASE64_COUNT  missing_refs: $MISSING"
if [ "$MISSING" -ne 0 ]; then
  echo "FAIL: $MISSING missing src refs" >&2
  exit 4
fi
