#!/bin/bash
set -euo pipefail
cd /home/hatch/workspace/repos/unscan-side
export HOME=/home/hatch
export TMPDIR=/home/hatch/workspace/tmp
mkdir -p "$TMPDIR"
export MALLOC_ARENA_MAX=1
export CARGO_BUILD_JOBS=1
BIN=./target/release/unprint

# wait for current cargo build if still running
while pgrep -f "cargo build --profile release --bin unprint" > /dev/null; do
  echo "[$(date -u +%H:%M:%S)] waiting for release build..."
  sleep 10
done

if [ ! -x "$BIN" ]; then
  echo "no binary, building..."
  env -u LD_PRELOAD TMPDIR=$TMPDIR CARGO_BUILD_JOBS=1 MALLOC_ARENA_MAX=1 /root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo build --profile release --bin unprint -j1
fi

ls -lh $BIN
strings $BIN | grep -i "UNPRINT_FLAT_TOP\|QUANT_HALF" | head || echo "no strings (expected, env var is runtime)"

# corpus hashes
echo "=== corpus hashes ==="
sha256sum test-docs/font-timeline-specimen.pdf test-docs/font-timeline-specimen-rasterized.pdf | tee /home/hatch/workspace/tmp/corpus-hashes.txt

VARIANTS="0.5 0.45 0.4 0.55"
RESULT=/home/hatch/workspace/tmp/flat-sweep-results.csv
echo "variant,tot,exact,major,avgZ,hit,minor,simfail,majorMiss,elapsed" > $RESULT

for v in $VARIANTS; do
  echo "=== RUN flat=$v $(date -u) ==="
  AUDIT_DIR="test-docs/audit-flat-$v"
  rm -rf "$AUDIT_DIR"
  mkdir -p "$AUDIT_DIR"
  LOG="/home/hatch/workspace/tmp/bap-flat-$v.log"
  # snapshot prev if exists
  if [ -f "test-docs/audit-flat-0.5/audit.json" ] && [ "$v" != "0.5" ]; then
    echo "prev exists"
  fi
  # run
  set +e
  env -u LD_PRELOAD UNPRINT_FLAT_TOP=$v UNPRINT_EXTRA_SEAMS=all TMPDIR=$TMPDIR MALLOC_ARENA_MAX=1 $BIN -o /dev/null --test test-docs/font-timeline-specimen.pdf --audit "$AUDIT_DIR" test-docs/font-timeline-specimen-rasterized.pdf > "$LOG" 2>&1
  RC=$?
  set -e
  echo "exit $RC, log tail:"
  tail -n 50 "$LOG"
  if [ $RC -ne 0 ]; then
    echo "FAILED $v rc=$RC" | tee -a $RESULT
    continue
  fi
  AUDIT_JSON="$AUDIT_DIR/audit.json"
  if [ ! -f "$AUDIT_JSON" ]; then
    echo "missing $AUDIT_JSON"
    continue
  fi
  # compute metrics via jq - canonical filtered scoring
  # filtered = expected_font != null && ocr_correct != false
  # exact = hit + similarity_failure
  # major = hit+minor+similarity_failure
  # per 02:49 canonical: exact = hit + similarity_failure, major = hit + minor_miss + similarity_failure
  # also compute hit/minor etc from miss_type? We need to see fields.
  # Inspect fields: text_entries[].miss_type? Let's dump keys.
  python3 - <<PY
import json, pathlib, sys, subprocess, math
audit_path="$AUDIT_JSON"
with open(audit_path) as f:
    data=json.load(f)
entries=data.get("text_entries",[])
def get(e,k):
    return e.get(k)
# filtered
filtered=[e for e in entries if e.get("expected_font") is not None and e.get("ocr_correct")!=False]
# For filtered scoring, what fields exist for hit counting?
# Use fields from report: hit, minor_miss, similarity_failure ? Check keys in entries
# Let's discover miss_type values
from collections import Counter
c=Counter()
for e in filtered:
    mt=e.get("miss_type") or e.get("decision") or "?"
    c[mt]+=1
print("miss_type counts", c)
# Count using miss_type if present else use font matching logic?
# According to AGENTS: Hits = miss_type == 'hit' + miss_type == 'minor_miss' ??? Actually BAP reporting says Hits = miss_type == 'hit' + 'minor_miss' denominator = entries with ground truth, but exact = hit+similarity_failure, major = hit+minor+similarity_failure
# Let's look at sample entry keys
if entries:
    print(list(entries[0].keys())[:30])
    # find similarity_score
    import statistics
    scores=[e.get("similarity_score") for e in filtered if isinstance(e.get("similarity_score"), (int,float))]
    avg=sum(scores)/len(scores) if scores else 0
    print(f"filtered tot {len(filtered)} avgZ {avg:.6f}")

PY
  # Use jq for final numbers - attempt both schemas
  # Scheme A: fields hit/minor etc are in audit.json top-level? Check report summary?
  jq -r '
  def filtered: .text_entries | map(select(.expected_font != null and .ocr_correct != false));
  filtered as $f |
  {
    tot: ($f|length),
    avgZ: (if ($f|length)>0 then ([$f[] | select(.similarity_score!=null) | .similarity_score] | add / length) else 0 end),
    hit: ($f|map(select(.miss_type=="hit"))|length),
    minor: ($f|map(select(.miss_type=="minor_miss"))|length),
    simfail: ($f|map(select(.miss_type=="similarity_failure"))|length),
    majorMiss: ($f|map(select(.miss_type=="major_miss"))|length),
    noGT: ($f|map(select(.miss_type=="no_ground_truth"))|length)
  } | "\(.tot),\(.hit),\(.minor),\(.simfail),\(.majorMiss),\(.avgZ)"
  ' "$AUDIT_JSON" > /home/hatch/workspace/tmp/metrics-$v.json || true
  cat /home/hatch/workspace/tmp/metrics-$v.json
  # compute exact/major
  read TOT HIT MINOR SIMFAIL MAJOR_MISS AVGZ <<< $(jq -r '
  def filtered: .text_entries | map(select(.expected_font != null and .ocr_correct != false));
  filtered as $f |
  [
    ($f|length),
    ($f|map(select(.miss_type=="hit"))|length),
    ($f|map(select(.miss_type=="minor_miss"))|length),
    ($f|map(select(.miss_type=="similarity_failure"))|length),
    ($f|map(select(.miss_type=="major_miss"))|length),
    (if ($f|length)>0 then ([$f[] | select(.similarity_score!=null) | .similarity_score] | add / length) else 0 end)
  ] | @tsv' "$AUDIT_JSON")
  # exact = hit + simfail, major = hit+minor+simfail
  EXACT=$((HIT + SIMFAIL))
  MAJOR=$((HIT + MINOR + SIMFAIL))
  ELAPSED=$(jq -r '.elapsed_secs // 0' "$AUDIT_JSON")
  echo "$v,$TOT,$EXACT,$MAJOR,$AVGZ,$HIT,$MINOR,$SIMFAIL,$MAJOR_MISS,$ELAPSED" | tee -a $RESULT
done

cat $RESULT
echo "done sweep $(date -u)"
