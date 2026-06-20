#!/usr/bin/env bash
set -euo pipefail

FONTS="$HOME/.local/share/fonts"
GF="https://raw.githubusercontent.com/google/fonts/main"

export DEBIAN_FRONTEND=noninteractive

# --- apt (MS Core Fonts + Google Fonts where available) ---
command -v debconf-set-selections >/dev/null 2>&1 &&
  echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" |
  sudo debconf-set-selections || true
sudo apt-get install -y \
  ttf-mscorefonts-installer fonts-crosextra-carlito fonts-crosextra-caladea \
  fonts-liberation fonts-lmodern texlive-fonts-recommended fonts-noto \
  fonts-courier-prime curl \
  fonts-ebgaramond fonts-ibm-plex fonts-inter fonts-lato \
  fonts-open-sans fonts-roboto unzip || true

# --- typewriter ---
TW="$FONTS/typewriter"; mkdir -p "$TW"
if ! [ -s "$TW/PrestigeElite-Regular.ttf" ]; then
  curl -fsSL -o "$TW/PrestigeElite-Regular.ttf" \
    "https://raw.githubusercontent.com/maseyyi/font-prestige-elite/master/prestige.ttf" || true
  curl -fsSL -o /tmp/pe-bold.zip "https://font.download/dl/font/prestige-elite-std.zip" || true
  unzip -j -o /tmp/pe-bold.zip "*.otf" -d "$TW/" || true
  curl -fsSL -o /tmp/lg.zip "https://font.download/dl/font/lettergothic.zip" || true
  unzip -j -o /tmp/lg.zip "*.ttf" -d "$TW/" || true
fi

# --- Google Fonts not in apt (upstream paths + filenames) ---
GF_PATHS=(
  ofl/librebaskerville/LibreBaskerville[wght].ttf
  ofl/librebaskerville/LibreBaskerville-Italic[wght].ttf
  ofl/librebodoni/LibreBodoni[wght].ttf
  ofl/librebodoni/LibreBodoni-Italic[wght].ttf
  ofl/librecaslontext/LibreCaslonText[wght].ttf
  ofl/librecaslontext/LibreCaslonText-Italic[wght].ttf
  ofl/zillaslab/ZillaSlab-Regular.ttf
  ofl/zillaslab/ZillaSlab-Bold.ttf
  ofl/jost/Jost[wght].ttf
  ofl/playfairdisplay/PlayfairDisplay[wght].ttf
  ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf
  ofl/merriweather/Merriweather[opsz,wdth,wght].ttf
  ofl/merriweather/Merriweather-Italic[opsz,wdth,wght].ttf
  ofl/sourcesans3/SourceSans3[wght].ttf
  ofl/notoserif/NotoSerif[wdth,wght].ttf
  ofl/ptserif/PT_Serif-Web-Regular.ttf
  ofl/ptserif/PT_Serif-Web-Italic.ttf
  ofl/ptserif/PT_Serif-Web-Bold.ttf
  ofl/ptsans/PT_Sans-Web-Regular.ttf
  ofl/ptsans/PT_Sans-Web-Bold.ttf
  ofl/sourceserif4/SourceSerif4[opsz,wght].ttf
  ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf
  apache/specialelite/SpecialElite-Regular.ttf
)

for p in "${GF_PATHS[@]}"; do
  dest="$FONTS/$p"
  [ -s "$dest" ] && continue
  mkdir -p "$(dirname "$dest")"
  url="$GF/${p//\[/%5B}"; url="${url//\]/%5D}"
  echo "  $p"
  curl -fsSL -o "$dest" "$url" || echo "  WARN: $p"
done

fc-cache -f "$FONTS" || true
echo "Done."
