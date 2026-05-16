#!/usr/bin/env bash
#
# check-fonts.sh — Verify that required test fonts are installed.
#
# MS core TTF fonts are a hard prerequisite for the test suite.
# The font-timeline-specimen ground truth includes sections for
# Times New Roman, Arial, Courier New, Georgia, Verdana,
# Trebuchet MS, and Comic Sans MS.
#
# Usage:
#   ./scripts/check-fonts.sh          # check only
#   ./scripts/check-fonts.sh --install # attempt to install missing fonts

set -euo pipefail

MSFONTS_DIR="/usr/share/fonts/truetype/msttcorefonts"
SPECIMEN_DIR="/usr/share/fonts/truetype/specimen-fonts"

# ── Required MS core fonts ──────────────────────────────────────────
# Map: display name → expected filename in msttcorefonts/
declare -A MS_CORE_FONTS=(
  ["Times New Roman"]="Times_New_Roman.ttf"
  ["Times New Roman Bold"]="Times_New_Roman_Bold.ttf"
  ["Times New Roman Italic"]="Times_New_Roman_Italic.ttf"
  ["Times New Roman Bold Italic"]="Times_New_Roman_Bold_Italic.ttf"
  ["Arial"]="Arial.ttf"
  ["Arial Bold"]="Arial_Bold.ttf"
  ["Arial Italic"]="Arial_Italic.ttf"
  ["Arial Bold Italic"]="Arial_Bold_Italic.ttf"
  ["Courier New"]="Courier_New.ttf"
  ["Courier New Bold"]="Courier_New_Bold.ttf"
  ["Courier New Italic"]="Courier_New_Italic.ttf"
  ["Courier New Bold Italic"]="Courier_New_Bold_Italic.ttf"
  ["Georgia"]="Georgia.ttf"
  ["Georgia Bold"]="Georgia_Bold.ttf"
  ["Georgia Italic"]="Georgia_Italic.ttf"
  ["Georgia Bold Italic"]="Georgia_Bold_Italic.ttf"
  ["Verdana"]="Verdana.ttf"
  ["Verdana Bold"]="Verdana_Bold.ttf"
  ["Verdana Italic"]="Verdana_Italic.ttf"
  ["Verdana Bold Italic"]="Verdana_Bold_Italic.ttf"
  ["Comic Sans MS"]="Comic_Sans_MS.ttf"
  ["Comic Sans MS Bold"]="Comic_Sans_MS_Bold.ttf"
  ["Trebuchet MS"]="Trebuchet_MS.ttf"
  ["Trebuchet MS Bold"]="Trebuchet_MS_Bold.ttf"
  ["Trebuchet MS Italic"]="Trebuchet_MS_Italic.ttf"
  ["Trebuchet MS Bold Italic"]="Trebuchet_MS_Bold_Italic.ttf"
)

# ── Required specimen fonts (Google Fonts / OFL) ───────────────────
declare -A SPECIMEN_FONTS=(
  ["Libre Bodoni Regular"]="libre-bodoni-400.ttf"
  ["Libre Bodoni Italic"]="libre-bodoni-400i.ttf"
  ["Libre Bodoni Bold"]="libre-bodoni-700.ttf"
  ["EB Garamond Regular"]="eb-garamond-400.ttf"
  ["EB Garamond Italic"]="eb-garamond-400i.ttf"
  ["EB Garamond Bold"]="eb-garamond-700.ttf"
  ["Libre Caslon Text Regular"]="libre-caslon-text-400.ttf"
  ["Libre Caslon Text Italic"]="libre-caslon-text-400i.ttf"
  ["Libre Caslon Text Bold"]="libre-caslon-text-700.ttf"
  ["Libre Baskerville Regular"]="libre-baskerville-400.ttf"
  ["Libre Baskerville Italic"]="libre-baskerville-400i.ttf"
  ["Libre Baskerville Bold"]="libre-baskerville-700.ttf"
)

missing=()
found=0
total=0

echo "Checking MS core fonts in ${MSFONTS_DIR}/ ..."
for name in "${!MS_CORE_FONTS[@]}"; do
  file="${MSFONTS_DIR}/${MS_CORE_FONTS[$name]}"
  total=$((total + 1))
  if [[ -f "$file" ]]; then
    found=$((found + 1))
  else
    # Also check lowercase variants (some installs use different casing)
    lc_file="${MSFONTS_DIR}/$(echo "${MS_CORE_FONTS[$name]}" | tr '[:upper:]' '[:lower:]')"
    if [[ -f "$lc_file" ]]; then
      found=$((found + 1))
    else
      missing+=("MS: $name → ${MS_CORE_FONTS[$name]}")
    fi
  fi
done

echo "Checking specimen fonts in ${SPECIMEN_DIR}/ ..."
for name in "${!SPECIMEN_FONTS[@]}"; do
  file="${SPECIMEN_DIR}/${SPECIMEN_FONTS[$name]}"
  total=$((total + 1))
  if [[ -f "$file" ]]; then
    found=$((found + 1))
  else
    missing+=("Specimen: $name → ${SPECIMEN_FONTS[$name]}")
  fi
done

echo ""
echo "Found: ${found}/${total} required fonts"

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "✓ All required fonts are installed."
  exit 0
fi

echo ""
echo "✗ Missing ${#missing[@]} font(s):"
for m in "${missing[@]}"; do
  echo "  - $m"
done

if [[ "${1:-}" == "--install" ]]; then
  echo ""
  echo "Attempting to install missing fonts..."

  # Check for MS core fonts
  has_ms_missing=false
  for m in "${missing[@]}"; do
    if [[ "$m" == MS:* ]]; then
      has_ms_missing=true
      break
    fi
  done

  if $has_ms_missing; then
    echo ""
    echo "Installing MS core fonts via ttf-mscorefonts-installer..."
    if command -v apt-get &>/dev/null; then
      echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | \
        sudo debconf-set-selections 2>/dev/null || true
      sudo apt-get install -y ttf-mscorefonts-installer
    else
      echo "ERROR: apt-get not available. Install MS core fonts manually:"
      echo "  See README.md for manual installation instructions."
      exit 1
    fi
  fi

  # Specimen fonts need manual download — print instructions
  has_specimen_missing=false
  for m in "${missing[@]}"; do
    if [[ "$m" == Specimen:* ]]; then
      has_specimen_missing=true
      break
    fi
  done

  if $has_specimen_missing; then
    echo ""
    echo "Specimen fonts must be downloaded from Google Fonts and placed in:"
    echo "  ${SPECIMEN_DIR}/"
    echo ""
    echo "Run:  python3 test-docs/gen-specimen.py"
    echo "(it downloads and installs specimen fonts on first run)"
  fi

  # Refresh font cache
  sudo fc-cache -f 2>/dev/null || true
  echo ""
  echo "Done. Re-run this script to verify."
else
  echo ""
  echo "Run with --install to attempt automatic installation:"
  echo "  ./scripts/check-fonts.sh --install"
  exit 1
fi
