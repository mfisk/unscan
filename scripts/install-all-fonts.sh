#!/usr/bin/env bash
set -euo pipefail

# Install all fonts required by unscan.
# Works both as root (system-wide) and as a regular user (per-user ~/.local/share/fonts).
# For apt packages, root is required; otherwise those steps are skipped.

if [ "$(id -u)" -eq 0 ]; then
  FONT_BASE="/usr/share/fonts/truetype"
  DOC_BASE="/usr/share/doc"
  LOCAL_SHARE="/usr/local/share/fonts"
  IS_ROOT=1
else
  FONT_BASE="$HOME/.local/share/fonts"
  DOC_BASE="$HOME/.local/share/doc"
  LOCAL_SHARE="$HOME/.local/share/fonts"
  IS_ROOT=0
fi

export DEBIAN_FRONTEND=noninteractive

if [ "$IS_ROOT" -eq 1 ] && command -v debconf-set-selections >/dev/null 2>&1; then
  echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections || true
fi

echo "=== Installing apt font packages ==="
if [ "$IS_ROOT" -eq 1 ] && command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    ttf-mscorefonts-installer \
    fonts-crosextra-carlito \
    fonts-crosextra-caladea \
    fonts-liberation \
    fonts-lmodern \
    texlive-fonts-recommended \
    fonts-noto \
    fonts-courier-prime \
    cabextract \
    wget \
    unzip || true
else
  echo "Skipping apt packages (requires root). Install manually or run with sudo."
fi

echo "=== Installing typewriter / vintage fonts ==="
TYPEWRITER_DIR="$LOCAL_SHARE/typewriter"
mkdir -p "$TYPEWRITER_DIR"
cd "$TYPEWRITER_DIR"

wget -q -O PrestigeElite-Regular.ttf "https://raw.githubusercontent.com/maseyyi/font-prestige-elite/master/prestige.ttf" || true
wget -q -O /tmp/pe-bold.zip "https://font.download/dl/font/prestige-elite-std.zip" || true
unzip -j -o /tmp/pe-bold.zip "*.otf" -d "$TYPEWRITER_DIR/" || true
wget -q -O /tmp/lg.zip "https://font.download/dl/font/lettergothic.zip" || true
unzip -j -o /tmp/lg.zip "*.ttf" -d "$TYPEWRITER_DIR/" || true

fc-cache -f "$TYPEWRITER_DIR" || true

echo "=== Microsoft Core Fonts EULA ==="
mkdir -p "$DOC_BASE"
cat > "$DOC_BASE/msttcorefonts-eula.txt" << 'EULA'
MICROSOFT SOFTWARE LICENSE TERMS
MICROSOFT CORE FONTS
...
Source: https://corefonts.sourceforge.net/eula.htm
EULA
echo "EULA saved to $DOC_BASE/msttcorefonts-eula.txt. By continuing you accept the terms."
sleep 1

echo "=== Installing MS Core Fonts via cabextract (EULA workaround) ==="
MS_DIR="$FONT_BASE/msttcorefonts"
mkdir -p "$MS_DIR" /tmp/msfonts
cd /tmp
for f in andale32 arial32 arialb32 comic32 courie32 georgi32 impact32 times32 trebuc32 verdan32 webdin32; do
  wget -q "https://downloads.sourceforge.net/corefonts/${f}.exe" -O "/tmp/${f}.exe" || true
  cabextract -q -d /tmp/msfonts "/tmp/${f}.exe" || true
done
cp /tmp/msfonts/*.ttf /tmp/msfonts/*.TTF "$MS_DIR/" 2>/dev/null || true
fc-cache -f "$MS_DIR" || true

echo "=== Installing Google Fonts specimen families ==="
SPEC_DIR="$FONT_BASE/specimen-fonts"
EXTRA_DIR="$FONT_BASE/extra"
TMP_FONT_DIR="/tmp/google-fonts"
mkdir -p "$SPEC_DIR" "$EXTRA_DIR" "$TMP_FONT_DIR"
cd "$TMP_FONT_DIR"

dl() {
  url="$1"
  out="$2"
  if [ ! -f "$out" ]; then
    echo "Downloading $(basename "$out") ..."
    wget -q -O "$out" "$url" || echo "WARN: Failed to download $url"
  fi
}

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf" "EBGaramond[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ebgaramond/EBGaramond-Italic%5Bwght%5D.ttf" "EBGaramond-Italic[wght].ttf"
cp "EBGaramond[wght].ttf" "$SPEC_DIR/eb-garamond-400.ttf" 2>/dev/null || true
cp "EBGaramond[wght].ttf" "$SPEC_DIR/eb-garamond-700.ttf" 2>/dev/null || true
cp "EBGaramond-Italic[wght].ttf" "$SPEC_DIR/eb-garamond-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville%5Bwght%5D.ttf" "LibreBaskerville[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Italic%5Bwght%5D.ttf" "LibreBaskerville-Italic[wght].ttf"
cp "LibreBaskerville[wght].ttf" "$SPEC_DIR/libre-baskerville-400.ttf" 2>/dev/null || true
cp "LibreBaskerville[wght].ttf" "$SPEC_DIR/libre-baskerville-700.ttf" 2>/dev/null || true
cp "LibreBaskerville-Italic[wght].ttf" "$SPEC_DIR/libre-baskerville-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebodoni/LibreBodoni%5Bwght%5D.ttf" "LibreBodoni[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebodoni/LibreBodoni-Italic%5Bwght%5D.ttf" "LibreBodoni-Italic[wght].ttf"
cp "LibreBodoni[wght].ttf" "$SPEC_DIR/libre-bodoni-400.ttf" 2>/dev/null || true
cp "LibreBodoni[wght].ttf" "$SPEC_DIR/libre-bodoni-700.ttf" 2>/dev/null || true
cp "LibreBodoni-Italic[wght].ttf" "$SPEC_DIR/libre-bodoni-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librecaslontext/LibreCaslonText%5Bwght%5D.ttf" "LibreCaslonText[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librecaslontext/LibreCaslonText-Italic%5Bwght%5D.ttf" "LibreCaslonText-Italic[wght].ttf"
cp "LibreCaslonText[wght].ttf" "$SPEC_DIR/libre-caslon-text-400.ttf" 2>/dev/null || true
cp "LibreCaslonText[wght].ttf" "$SPEC_DIR/libre-caslon-text-700.ttf" 2>/dev/null || true
cp "LibreCaslonText-Italic[wght].ttf" "$SPEC_DIR/libre-caslon-text-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/zillaslab/ZillaSlab-Regular.ttf" "ZillaSlab-Regular.ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/zillaslab/ZillaSlab-Bold.ttf" "ZillaSlab-Bold.ttf"
cp "ZillaSlab-Regular.ttf" "$SPEC_DIR/zilla-slab-400.ttf" 2>/dev/null || true
cp "ZillaSlab-Bold.ttf" "$SPEC_DIR/zilla-slab-700.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/jost/Jost%5Bwght%5D.ttf" "Jost[wght].ttf"
cp "Jost[wght].ttf" "$SPEC_DIR/jost-400.ttf" 2>/dev/null || true
cp "Jost[wght].ttf" "$SPEC_DIR/jost-700.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf" "PlayfairDisplay[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay-Italic%5Bwght%5D.ttf" "PlayfairDisplay-Italic[wght].ttf"
cp "PlayfairDisplay[wght].ttf" "$SPEC_DIR/playfair-display-400.ttf" 2>/dev/null || true
cp "PlayfairDisplay-Italic[wght].ttf" "$SPEC_DIR/playfair-display-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth,wght%5D.ttf" "Roboto[wdth,wght].ttf"
cp "Roboto[wdth,wght].ttf" "$SPEC_DIR/roboto-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth,wght%5D.ttf" "OpenSans[wdth,wght].ttf"
cp "OpenSans[wdth,wght].ttf" "$SPEC_DIR/open-sans-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Regular.ttf" "Lato-Regular.ttf"
cp "Lato-Regular.ttf" "$SPEC_DIR/lato-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather%5Bopsz,wdth,wght%5D.ttf" "Merriweather[opsz,wdth,wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather-Italic%5Bopsz,wdth,wght%5D.ttf" "Merriweather-Italic[opsz,wdth,wght].ttf"
cp "Merriweather[opsz,wdth,wght].ttf" "$SPEC_DIR/merriweather-400.ttf" 2>/dev/null || true
cp "Merriweather-Italic[opsz,wdth,wght].ttf" "$SPEC_DIR/merriweather-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf" "SourceSans3[wght].ttf"
cp "SourceSans3[wght].ttf" "$SPEC_DIR/source-sans-pro-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf" "NotoSerif[wdth,wght].ttf"
cp "NotoSerif[wdth,wght].ttf" "$SPEC_DIR/noto-serif-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Regular.ttf" "PT_Serif-Web-Regular.ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Italic.ttf" "PT_Serif-Web-Italic.ttf"
cp "PT_Serif-Web-Regular.ttf" "$SPEC_DIR/pt-serif-400.ttf" 2>/dev/null || true
cp "PT_Serif-Web-Italic.ttf" "$SPEC_DIR/pt-serif-400i.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf" "IBMPlexSans[wdth,wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexserif/IBMPlexSerif-Regular.ttf" "IBMPlexSerif-Regular.ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf" "IBMPlexMono-Regular.ttf"
cp "IBMPlexSans[wdth,wght].ttf" "$SPEC_DIR/ibm-plex-sans-400.ttf" 2>/dev/null || true
cp "IBMPlexSerif-Regular.ttf" "$SPEC_DIR/ibm-plex-serif-400.ttf" 2>/dev/null || true
cp "IBMPlexMono-Regular.ttf" "$SPEC_DIR/ibm-plex-mono-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf" "Inter[opsz,wght].ttf"
cp "Inter[opsz,wght].ttf" "$SPEC_DIR/inter-400.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf" "SourceSerif4[opsz,wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/sourceserif4/SourceSerif4-Italic%5Bopsz,wght%5D.ttf" "SourceSerif4-Italic[opsz,wght].ttf"
cp "SourceSerif4[opsz,wght].ttf" "$EXTRA_DIR/SourceSerif4-Regular.ttf" 2>/dev/null || true
cp "SourceSerif4[opsz,wght].ttf" "$EXTRA_DIR/SourceSerif4-Bold.ttf" 2>/dev/null || true
cp "SourceSerif4-Italic[opsz,wght].ttf" "$EXTRA_DIR/SourceSerif4-It.ttf" 2>/dev/null || true

dl "https://raw.githubusercontent.com/google/fonts/main/apache/specialelite/SpecialElite-Regular.ttf" "SpecialElite-Regular.ttf"
cp "SpecialElite-Regular.ttf" "$TYPEWRITER_DIR/SpecialElite-Regular.ttf" 2>/dev/null || true

fc-cache -f "$SPEC_DIR" "$EXTRA_DIR" "$TYPEWRITER_DIR" || true

echo "=== Generating specimen fonts ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/test-docs"
if [ -f gen-specimen.py ]; then
  python3 gen-specimen.py || true
fi

echo "=== Final font cache refresh ==="
fc-cache -f || true

echo "=== Verification ==="
echo "MS Core fonts:"
ls -1 "$MS_DIR" 2>/dev/null | head
echo ""
echo "Typewriter fonts:"
ls -1 "$TYPEWRITER_DIR" 2>/dev/null | head
echo ""
echo "Specimen fonts:"
ls -1 "$SPEC_DIR" 2>/dev/null | head
echo ""
fc-list | grep -i "Arial" | head -n 3 || true
fc-list | grep -i "Prestige" | head -n 3 || true

echo ""
echo "Done. Fonts installed to $FONT_BASE (user mode: IS_ROOT=$IS_ROOT)."
