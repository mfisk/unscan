#!/usr/bin/env bash
set -euo pipefail

FONTS="$HOME/.local/share/fonts"
GF="https://raw.githubusercontent.com/google/fonts/main"

export DEBIAN_FRONTEND=noninteractive

# --- apt (packaged fonts + install prerequisites) ---
command -v debconf-set-selections >/dev/null 2>&1 &&
  echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" |
  sudo debconf-set-selections || true

apt_packages=(
  fonts-crosextra-carlito fonts-crosextra-caladea
  fonts-liberation fonts-lmodern texlive-fonts-recommended fonts-texgyre
  fonts-noto fonts-noto-core fonts-noto-cjk fonts-noto-color-emoji
  fonts-noto-extra fonts-noto-mono fonts-noto-ui-core fonts-noto-ui-extra
  fonts-noto-unhinted
  fonts-courier-prime fontconfig curl cabextract tesseract-ocr weasyprint
  python3-pikepdf python3-fonttools python3-pil python3-fitz
  python3-img2pdf python3-numpy
  fonts-ebgaramond fonts-ibm-plex fonts-inter fonts-lato
  fonts-open-sans fonts-roboto fonts-roboto-unhinted unzip
)

# A reset can preserve dpkg's database while dropping package payloads.  A
# status-only check therefore reports "installed" even when the executable,
# Python module, or font files are gone.  Check every package-owned path and
# dpkg's recorded checksums, then reinstall only packages that are actually
# missing or damaged.
missing_packages=()
broken_packages=()
for package in "${apt_packages[@]}"; do
  if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != "install ok installed" ]]; then
    missing_packages+=("$package")
    continue
  fi

  missing_payload="$(dpkg-query -L "$package" 2>/dev/null |
    while IFS= read -r path; do
      case "$path" in
        /bin/*|/sbin/*|/lib/*|/usr/*)
          if [[ ! -e "$path" && ! -L "$path" ]]; then
            printf '%s\n' "$path"
            break
          fi
          ;;
      esac
    done)"
  verify_output="$(dpkg -V "$package" 2>&1 || true)"
  if [[ -n "$missing_payload" || -n "$verify_output" ]]; then
    echo "Repairing $package: package payload is missing or corrupt" >&2
    [[ -n "$missing_payload" ]] && echo "  missing: $missing_payload" >&2
    [[ -n "$verify_output" ]] && printf '%s\n' "$verify_output" >&2
    broken_packages+=("$package")
  fi
done

apt_install() {
  local mode="$1"
  shift
  ((${#@})) || return 0

  local -a args=(install -y --no-install-recommends)
  [[ "$mode" == reinstall ]] && args+=(--reinstall)
  if sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 \
      "${args[@]}" "$@"; then
    return 0
  fi

  # If package indexes vanished with the payload reset, refresh once.  Wait
  # only for apt's explicit lock contention; all other update failures are
  # real installation failures.
  while ! apt_output="$(sudo apt-get -o DPkg::Lock::Timeout=300 update 2>&1)"; do
    printf '%s\n' "$apt_output" >&2
    if [[ "$apt_output" != *"Could not get lock"* ]]; then
      return 1
    fi
    sleep 5
  done
  printf '%s\n' "$apt_output"
  sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 \
    "${args[@]}" "$@"
}

apt_install install "${missing_packages[@]}"
apt_install reinstall "${broken_packages[@]}"

# SourceForge redirects are unreliable behind some proxies. Install the
# original Microsoft Core Fonts archives from a pinned mirror and verify every
# archive before extraction.
MS_CORE="$FONTS/msttcorefonts"
MS_CACHE="$HOME/.cache/microsoft-core-fonts"
MS_BASE="https://raw.githubusercontent.com/pushcx/corefonts/8f8627b0004b5f1b952c09cf6390e4ae46475e59"
mkdir -p "$MS_CORE" "$MS_CACHE"

MS_ARCHIVES=(
  "0524fe42951adc3a7eb870e32f0920313c71f170c859b5f770d82b4ee111e970 andale32.exe"
  "85297a4d146e9c87ac6f74822734bdee5f4b2a722d7eaa584b7f2cbf76f478f6 arial32.exe"
  "a425f0ffb6a1a5ede5b979ed6177f4f4f4fdef6ae7c302a7b7720ef332fec0a8 arialb32.exe"
  "9c6df3feefde26d4e41d4a4fe5db2a89f9123a772594d7f59afd062625cd204e comic32.exe"
  "bb511d861655dde879ae552eb86b134d6fae67cb58502e6ff73ec5d9151f3384 courie32.exe"
  "2c2c7dcda6606ea5cf08918fb7cd3f3359e9e84338dc690013f20cd42e930301 georgi32.exe"
  "6061ef3b7401d9642f5dfdb5f2b376aa14663f6275e60a51207ad4facf2fccfb impact32.exe"
  "db56595ec6ef5d3de5c24994f001f03b2a13e37cee27bc25c58f6f43e8f807ab times32.exe"
  "5a690d9bb8510be1b8b4fe49f1f2319651fe51bbe54775ddddd8ef0bd07fdac9 trebuc32.exe"
  "c1cb61255e363166794e47664e2f21af8e3a26cb6346eb8d2ae2fa85dd5aad96 verdan32.exe"
  "64595b5abc1080fba8610c5c34fab5863408e806aafe84653ca8575bed17d75a webdin32.exe"
)

for entry in "${MS_ARCHIVES[@]}"; do
  read -r sha archive <<<"$entry"
  cached="$MS_CACHE/$archive"
  if ! printf '%s  %s\n' "$sha" "$cached" | sha256sum -c --status 2>/dev/null; then
    echo "  $archive"
    curl -fsSL -o "$cached.part" "$MS_BASE/$archive"
    printf '%s  %s\n' "$sha" "$cached.part" | sha256sum -c -
    mv "$cached.part" "$cached"
  fi
  cabextract -q -L -F '*.ttf' -d "$MS_CORE" "$cached"
done

# Georgia is part of the hardcoded audit fixture, so verify all four exact
# payloads instead of accepting fontconfig's fallback to another serif face.
printf '%s  %s\n' \
  "7d0bb20c632bb59e81a0885f573bd2173f71f73204de9058feb68ce032227072" "$MS_CORE/georgia.ttf" \
  "82d2fbadb88a8632d7f2e8ad50420c9fd2e7d3cbc0e90b04890213a711b34b93" "$MS_CORE/georgiab.ttf" \
  "1523f19bda6acca42c47c50da719a12dd34f85cc2606e6a5af15a7728b377b60" "$MS_CORE/georgiai.ttf" \
  "c983e037d8e4e694dd0fb0ba2e625bca317d67a41da2dc81e46a374e53d0ec8a" "$MS_CORE/georgiaz.ttf" |
  sha256sum -c -

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
  ofl/opensans/OpenSans[wdth,wght].ttf
  ofl/opensans/OpenSans-Italic[wdth,wght].ttf
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

fc-cache -f "$FONTS"
for style in Regular Bold Italic "Bold Italic"; do
  resolved="$(fc-match -f '%{family[0]}|%{style[0]}|%{file}' "Georgia:style=$style")"
  if [[ "$resolved" != "Georgia|$style|$MS_CORE/"* ]]; then
    echo "ERROR: Georgia $style resolved to $resolved" >&2
    exit 1
  fi
done

resolved="$(fc-match -f '%{family[0]}|%{style[0]}|%{file}' 'Open Sans:style=Regular')"
if [[ "$resolved" != "Open Sans|Regular|"* ]]; then
  echo "ERROR: Open Sans Regular resolved to $resolved" >&2
  exit 1
fi

# Verify the actual font payload owned by each apt package.  Checking only
# fc-match would allow a downloaded font with the same family to hide a broken
# package installation.
PACKAGED_FONT_FAMILIES=(
  "fonts-crosextra-carlito|Carlito"
  "fonts-crosextra-caladea|Caladea"
  "fonts-liberation|Liberation Sans"
  "fonts-lmodern|Latin Modern Roman"
  "fonts-texgyre|TeX Gyre Termes"
  "fonts-noto-core|Noto Sans"
  "fonts-courier-prime|Courier Prime"
  "fonts-ebgaramond|EB Garamond"
  "fonts-ibm-plex|IBM Plex Sans"
  "fonts-inter|Inter"
  "fonts-lato|Lato"
  "fonts-open-sans|Open Sans"
  "fonts-roboto-unhinted|Roboto"
)

for expectation in "${PACKAGED_FONT_FAMILIES[@]}"; do
  package="${expectation%%|*}"
  expected_family="${expectation#*|}"
  found_family=false
  found_payload=false
  while IFS= read -r font_file; do
    [[ -n "$font_file" ]] || continue
    found_payload=true
    [[ -s "$font_file" ]] || {
      echo "ERROR: $package font payload is missing: $font_file" >&2
      exit 1
    }
    if fc-scan --format '%{family}\n' "$font_file" 2>/dev/null |
        tr ',' '\n' | grep -Fxq "$expected_family"; then
      found_family=true
      break
    fi
  done < <(dpkg-query -L "$package" | grep -Ei '\.(otf|ttf|ttc|pfb)$' || true)

  if [[ "$found_payload" != true ]]; then
    echo "ERROR: $package owns no installed font payload" >&2
    exit 1
  fi
  if [[ "$found_family" != true ]]; then
    echo "ERROR: $package does not contain expected family $expected_family" >&2
    exit 1
  fi
done

# Verify every executable and Python dependency used by the fixture generator.
# Missing imports must fail installation now, not fail a lob later.
command -v tesseract >/dev/null
tesseract --version >/dev/null
python3 - <<'PY'
import fitz
import img2pdf
import numpy
import pikepdf
import weasyprint
from PIL import Image
from fontTools.ttLib import TTFont
PY

echo "Done."
