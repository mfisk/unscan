#!/usr/bin/env bash
# regression_ligature.sh — Ligature font-identification regression test
#
# Runs unscan on ligature-test.pdf (3 font families × with/without ligatures
# × 2 text variants = 21 lines), then checks the built-in miss report for
# perfect font identification (0 misses).
#
# Usage:
#   ./tests/regression_ligature.sh
#   UNSCAN=./target/debug/unscan ./tests/regression_ligature.sh
#
# Prerequisites:
#   - Built unscan binary
#   - python3 with fpdf2, uharfbuzz, PyMuPDF (fitz)
#   - Fonts: EB Garamond, Libre Caslon Text, Noto Serif (system-installed)

set -uo pipefail

UNSCAN="${UNSCAN:-./target/debug/unscan}"
TESTDIR="test-docs"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[1;33m%s\033[0m\n' "$*"; }

echo "unscan ligature regression test"
echo "Binary: $UNSCAN"
echo ""

if [ ! -x "$UNSCAN" ]; then
    red "ERROR: $UNSCAN not found or not executable"
    echo "Build with: cargo build"
    exit 1
fi

PDF="$TESTDIR/ligature-test.pdf"

if [ ! -f "$PDF" ]; then
    red "ERROR: $PDF not found"
    echo "Generate with: python3 $TESTDIR/gen-ligature-test.py"
    exit 1
fi

# ── Run unscan with --audit-vector ────────────────────────────────────
echo "Running unscan on $PDF..."
AUDIT_DIR="$TMPDIR/audit"
OUTPUT="$TMPDIR/output.pdf"

$UNSCAN "$PDF" -o "$OUTPUT" --audit "$AUDIT_DIR" --audit-vector "$PDF" 2>"$TMPDIR/unscan.log"
rc=$?

if [ $rc -ne 0 ]; then
    red "FAIL: unscan exited with code $rc"
    tail -20 "$TMPDIR/unscan.log"
    exit 1
fi

if [ ! -f "$AUDIT_DIR/audit.json" ]; then
    red "FAIL: no audit.json produced"
    exit 1
fi

if [ ! -f "$AUDIT_DIR/report.html" ]; then
    red "FAIL: no report.html produced"
    exit 1
fi

# ── Parse report summary from unscan stderr ──────────────────────────
# Format: "Report: H/C (P%) — M misses ..."
REPORT_LINE=$(grep "Report:" "$TMPDIR/unscan.log" || true)
echo ""
echo "  $REPORT_LINE"
echo ""

# Extract hits and compared from "Report: H/C"
HITS=$(echo "$REPORT_LINE" | grep -oP 'Report: \K[0-9]+')
COMPARED=$(echo "$REPORT_LINE" | grep -oP 'Report: [0-9]+/\K[0-9]+')
MISSES_FROM_LINE=$(echo "$REPORT_LINE" | grep -oP '— \K[0-9]+(?= misses)')

# Total = compared (the report only covers compared lines)
TOTAL="${COMPARED:-0}"
HITS="${HITS:-0}"
MISSES="${MISSES_FROM_LINE:-0}"

EXPECTED_TOTAL=21
EXPECTED_MISSES=0

PASS=0
FAIL=0

# Check total lines detected
if [ "${TOTAL:-0}" -eq "$EXPECTED_TOTAL" ]; then
    green "PASS: $TOTAL/$EXPECTED_TOTAL lines detected"
    PASS=$((PASS+1))
else
    red "FAIL: $TOTAL/$EXPECTED_TOTAL lines detected"
    FAIL=$((FAIL+1))
fi

# Check zero misses
if [ "${MISSES:-1}" -eq "$EXPECTED_MISSES" ]; then
    green "PASS: 0 font identification misses"
    PASS=$((PASS+1))
else
    red "FAIL: $MISSES font identification misses (expected $EXPECTED_MISSES)"
    FAIL=$((FAIL+1))
fi

# Check all hits
if [ "${HITS:-0}" -eq "$EXPECTED_TOTAL" ]; then
    green "PASS: $HITS/$EXPECTED_TOTAL correct font identifications"
    PASS=$((PASS+1))
else
    red "FAIL: $HITS/$EXPECTED_TOTAL correct font identifications"
    FAIL=$((FAIL+1))
fi

echo ""
echo "════════════════════════════════════"
printf "  PASS: %d  FAIL: %d\n" "$PASS" "$FAIL"
echo "════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
