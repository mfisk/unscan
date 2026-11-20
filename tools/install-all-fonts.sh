#!/usr/bin/env bash
set -euo pipefail

FONTS="$HOME/.local/share/fonts"
GF="https://raw.githubusercontent.com/google/fonts/main"
OS="$(uname -s)"

# Install route: default (apt on Linux, Homebrew casks on macOS) or --home
# (Linux only: download every manifest font into $FONTS instead of
# apt-installing it). --home exists for VMs where apt state is wiped by
# reboots but $HOME persists. The flag is the only switch: no
# environment-variable fallback, no auto-detection. Unknown arguments are a
# hard error — a silently ignored flag once let an apt install run when a
# download route was intended (the --brew flag died in the Nov 1, 2026
# rewrite while its caller kept passing it).
usage() {
  cat <<'EOF'
Usage: install-all-fonts.sh [--home] [--help]

Install the fonts in the FONTS_TO_INSTALL manifest plus prerequisites.

  (default)  Fonts from apt packages (Linux) or Homebrew casks (macOS),
             plus direct downloads for fonts with no package.
  --home     Linux only. Download every manifest font into
             ~/.local/share/fonts instead of apt-installing it.
             Prerequisites (tesseract, weasyprint, Python packages)
             still install via apt.
  --help     Show this message and exit.
EOF
}

HOME_MODE=0
for arg in "$@"; do
  case "$arg" in
    --home) HOME_MODE=1 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done
if [[ "$HOME_MODE" == 1 && "$OS" == "Darwin" ]]; then
  echo "ERROR: --home is Linux-only; macOS installs fonts via Homebrew casks" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Single source of truth for fonts and prerequisites to install
# ---------------------------------------------------------------------------

# Unified font manifest:
#   "Expected Fontconfig Family|apt package(s)|brew cask(s)|Google Fonts path(s)|Home downloads"
#
# - Column 1: Family name verified via fc-scan (comma-separated if platform
#             versions use alternate family names).
# - Column 2: Debian/Ubuntu apt package(s) to install (first is verified via
#             dpkg-query -L + fc-scan). Skipped with --home.
# - Column 3: macOS Homebrew cask(s) to install (first is verified via
#             Caskroom + fc-scan).
# - Column 4: Upstream Google Fonts repo path(s) to download directly to
#             $FONTS/<path> and verify via fc-scan. Downloaded in both
#             routes: families with no package need it, and families with
#             one get the upstream variable file the specimen was built
#             against.
# - Column 5: Home-route downloads for fonts NOT in Google Fonts, used only
#             with --home (the apt route ignores this column):
#             space-separated tokens, either URL>DEST (single file to
#             $FONTS/DEST) or deb:URL>DESTDIR:GLOB (download a .deb, extract
#             its data tarball, and copy files whose basename matches GLOB
#             into $FONTS/DESTDIR — for fonts whose built files are only
#             distributed as distro packages). Every entry must have column
#             4 or column 5 coverage, or --home fails loudly in verification.
FONTS_TO_INSTALL=(
  "Carlito|fonts-crosextra-carlito|font-carlito|ofl/carlito/Carlito-Regular.ttf ofl/carlito/Carlito-Bold.ttf ofl/carlito/Carlito-Italic.ttf ofl/carlito/Carlito-BoldItalic.ttf|"
  "Caladea|fonts-crosextra-caladea|font-caladea|ofl/caladea/Caladea-Regular.ttf ofl/caladea/Caladea-Bold.ttf ofl/caladea/Caladea-Italic.ttf ofl/caladea/Caladea-BoldItalic.ttf|"
  "Liberation Sans|fonts-liberation|font-liberation||deb:https://deb.debian.org/debian/pool/main/f/fonts-liberation/fonts-liberation_2.1.5-3_all.deb>liberation:LiberationSans-*.ttf"
  "Latin Modern Roman,LM Roman 10|fonts-lmodern|font-latin-modern||https://mirrors.ctan.org/fonts/lm/fonts/opentype/public/lm/lmroman10-regular.otf>lm/lmroman10-regular.otf https://mirrors.ctan.org/fonts/lm/fonts/opentype/public/lm/lmroman10-bold.otf>lm/lmroman10-bold.otf https://mirrors.ctan.org/fonts/lm/fonts/opentype/public/lm/lmroman10-italic.otf>lm/lmroman10-italic.otf https://mirrors.ctan.org/fonts/lm/fonts/opentype/public/lm/lmroman10-bolditalic.otf>lm/lmroman10-bolditalic.otf"
  "TeX Gyre Termes|fonts-texgyre texlive-fonts-recommended|font-tex-gyre-termes font-tex-gyre-heros font-tex-gyre-pagella font-tex-gyre-cursor||https://mirrors.ctan.org/fonts/tex-gyre/opentype/texgyretermes-regular.otf>texgyre/texgyretermes-regular.otf https://mirrors.ctan.org/fonts/tex-gyre/opentype/texgyretermes-italic.otf>texgyre/texgyretermes-italic.otf https://mirrors.ctan.org/fonts/tex-gyre/opentype/texgyretermes-bold.otf>texgyre/texgyretermes-bold.otf https://mirrors.ctan.org/fonts/tex-gyre/opentype/texgyretermes-bolditalic.otf>texgyre/texgyretermes-bolditalic.otf"
  "Noto Sans|fonts-noto-core fonts-noto fonts-noto-cjk fonts-noto-color-emoji fonts-noto-extra fonts-noto-mono fonts-noto-ui-core fonts-noto-ui-extra fonts-noto-unhinted|font-noto-sans font-noto-serif font-noto-sans-mono font-noto-color-emoji|ofl/notosans/NotoSans[wdth,wght].ttf ofl/notosans/NotoSans-Italic[wdth,wght].ttf|"
  "Noto Serif|||ofl/notoserif/NotoSerif[wdth,wght].ttf|"
  "Courier Prime|fonts-courier-prime|font-courier-prime|ofl/courierprime/CourierPrime-Regular.ttf ofl/courierprime/CourierPrime-Bold.ttf ofl/courierprime/CourierPrime-Italic.ttf ofl/courierprime/CourierPrime-BoldItalic.ttf|"
  "EB Garamond|fonts-ebgaramond|font-eb-garamond|ofl/ebgaramond/EBGaramond[wght].ttf ofl/ebgaramond/EBGaramond-Italic[wght].ttf|"
  "IBM Plex Sans|fonts-ibm-plex|font-ibm-plex-sans|ofl/ibmplexsans/IBMPlexSans[wdth,wght].ttf ofl/ibmplexsans/IBMPlexSans-Italic[wdth,wght].ttf|"
  "IBM Plex Serif|fonts-ibm-plex|font-ibm-plex-serif|ofl/ibmplexserif/IBMPlexSerif-Regular.ttf|"
  "IBM Plex Mono|fonts-ibm-plex|font-ibm-plex-mono|ofl/ibmplexmono/IBMPlexMono-Regular.ttf|"
  "Inter|fonts-inter|font-inter|ofl/inter/Inter[opsz,wght].ttf ofl/inter/Inter-Italic[opsz,wght].ttf|"
  "Lato|fonts-lato|font-lato|ofl/lato/Lato-Regular.ttf ofl/lato/Lato-Bold.ttf ofl/lato/Lato-Italic.ttf ofl/lato/Lato-BoldItalic.ttf|"
  "Open Sans|fonts-open-sans|font-open-sans|ofl/opensans/OpenSans[wdth,wght].ttf ofl/opensans/OpenSans-Italic[wdth,wght].ttf|"
  "Roboto|fonts-roboto-unhinted fonts-roboto|font-roboto|ofl/roboto/Roboto[wdth,wght].ttf ofl/roboto/Roboto-Italic[wdth,wght].ttf|"
  "Libre Baskerville|||ofl/librebaskerville/LibreBaskerville[wght].ttf ofl/librebaskerville/LibreBaskerville-Italic[wght].ttf|"
  "Libre Bodoni|||ofl/librebodoni/LibreBodoni[wght].ttf ofl/librebodoni/LibreBodoni-Italic[wght].ttf|"
  "Libre Caslon Text|||ofl/librecaslontext/LibreCaslonText[wght].ttf ofl/librecaslontext/LibreCaslonText-Italic[wght].ttf|"
  "Zilla Slab|||ofl/zillaslab/ZillaSlab-Regular.ttf ofl/zillaslab/ZillaSlab-Bold.ttf|"
  "Jost|||ofl/jost/Jost[wght].ttf|"
  "Playfair Display|||ofl/playfairdisplay/PlayfairDisplay[wght].ttf ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf|"
  "Merriweather|||ofl/merriweather/Merriweather[opsz,wdth,wght].ttf ofl/merriweather/Merriweather-Italic[opsz,wdth,wght].ttf|"
  "Source Sans 3|||ofl/sourcesans3/SourceSans3[wght].ttf|"
  "PT Serif|||ofl/ptserif/PT_Serif-Web-Regular.ttf ofl/ptserif/PT_Serif-Web-Italic.ttf ofl/ptserif/PT_Serif-Web-Bold.ttf|"
  "PT Sans|||ofl/ptsans/PT_Sans-Web-Regular.ttf ofl/ptsans/PT_Sans-Web-Bold.ttf|"
  "Source Serif 4|||ofl/sourceserif4/SourceSerif4[opsz,wght].ttf ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf|"
  "Special Elite|||apache/specialelite/SpecialElite-Regular.ttf|"
)

# Microsoft Core Fonts archives (SHA-256 + filename from pinned mirror):
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

# Non-font CLI/library prerequisites:
#   "apt package|brew formula" (empty brew cell if built into macOS)
PREREQ_TOOLS=(
  "fontconfig|fontconfig"
  "curl|curl"
  "cabextract|cabextract"
  "tesseract-ocr|tesseract"
  "weasyprint|pango"
  "unzip|"
  "|poppler"
)

# Python dependencies used by fixture generators and tools:
#   "python import|apt package|pip package"
PYTHON_DEPS=(
  "fitz|python3-fitz|pymupdf"
  "img2pdf|python3-img2pdf|img2pdf"
  "numpy|python3-numpy|numpy"
  "pikepdf|python3-pikepdf|pikepdf"
  "weasyprint|weasyprint|weasyprint"
  "PIL|python3-pil|Pillow"
  "fontTools|python3-fonttools|fonttools"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

sha256_check() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c "$@"
  else
    sha256sum -c "$@"
  fi
}

append_unique() {
  local list_name="$1"
  shift
  local item existing
  for item in "$@"; do
    if [[ -z "$item" ]]; then
      continue
    fi
    eval "for existing in \${${list_name}[@]+\"\${${list_name}[@]}\"}; do
      if [[ \"\$existing\" == \"\$item\" ]]; then
        continue 2
      fi
    done"
    eval "${list_name}+=(\"\$item\")"
  done
}

# file_has_family <file> <expected-family>: exit 0 if fc-scan reports one of
# the comma-separated expected families for the file.
file_has_family() {
  local font_file="$1" expected_family="$2"
  fc-scan --format '%{family}\n' "$font_file" 2>/dev/null |
    tr ',' '\n' | grep -Fxq -f <(printf '%s\n' "$expected_family" | tr ',' '\n')
}

# ---------------------------------------------------------------------------
# Platform package installation (Homebrew on macOS, apt on Linux)
# ---------------------------------------------------------------------------

if [[ "$OS" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew (brew) is required on macOS" >&2
    exit 1
  fi

  unset HOMEBREW_ASK
  export HOMEBREW_NO_ASK=1
  export HOMEBREW_NO_AUTO_UPDATE=1
  export HOMEBREW_NO_INSTALL_CLEANUP=1

  BREW_PREFIX="$(brew --prefix)"
  CASKROOM="$(brew --caskroom)"
  BREW_FONTS="$FONTS/brew"
  [[ -d "$FONTS" ]] || mkdir -p "$FONTS"
  [[ -d "$BREW_FONTS" ]] || mkdir -p "$BREW_FONTS"

  brew_formulae=()
  for entry in "${PREREQ_TOOLS[@]}"; do
    formula="${entry#*|}"
    append_unique brew_formulae $formula
  done

  missing_formulae=()
  broken_formulae=()
  for formula in "${brew_formulae[@]}"; do
    opt_dir="$BREW_PREFIX/opt/$formula"
    if [[ ! -d "$opt_dir" ]]; then
      missing_formulae+=("$formula")
      continue
    fi
    broken_link="$(find "$opt_dir" -type l ! -exec test -e {} \; -print -quit 2>/dev/null || true)"
    if [[ -n "$broken_link" ]]; then
      echo "Repairing $formula: broken symlink $broken_link" >&2
      broken_formulae+=("$formula")
    fi
  done

  if ((${#missing_formulae[@]})); then
    brew install "${missing_formulae[@]}"
  fi
  if ((${#broken_formulae[@]})); then
    brew reinstall "${broken_formulae[@]}"
  fi

  brew_casks=()
  for entry in "${FONTS_TO_INSTALL[@]}"; do
    IFS='|' read -r _ _ casks _ _ <<<"$entry"
    append_unique brew_casks $casks
  done

  missing_casks=()
  broken_casks=()
  for cask in "${brew_casks[@]}"; do
    cask_dir="$CASKROOM/$cask"
    if [[ ! -d "$cask_dir" ]]; then
      missing_casks+=("$cask")
      continue
    fi

    has_fonts=false
    missing_payload=""
    while IFS= read -r font_path; do
      [[ -n "$font_path" ]] || continue
      has_fonts=true
      if [[ ! -e "$font_path" || ! -r "$font_path" || ! -s "$font_path" ]]; then
        missing_payload="$font_path"
        break
      fi
    done < <(find "$cask_dir" \( -type f -o -type l \) 2>/dev/null | grep -Ei '\.(otf|ttf|ttc|pfb)$' || true)

    if [[ "$has_fonts" != true || -n "$missing_payload" ]]; then
      echo "Repairing $cask: cask font payload is missing or unreadable" >&2
      if [[ -n "$missing_payload" ]]; then
        echo "  missing: $missing_payload" >&2
      fi
      broken_casks+=("$cask")
    fi
  done

  if ((${#missing_casks[@]})); then
    brew install --cask --fontdir="$BREW_FONTS" "${missing_casks[@]}"
  fi
  if ((${#broken_casks[@]})); then
    brew reinstall --cask --force --fontdir="$BREW_FONTS" "${broken_casks[@]}"
  fi

  missing_pip=()
  while IFS= read -r pip_pkg; do
    [[ -n "$pip_pkg" ]] && append_unique missing_pip "$pip_pkg"
  done < <(python3 - "${PYTHON_DEPS[@]}" <<'PY'
import importlib.util, sys
for entry in sys.argv[1:]:
    mod, _, pip_pkg = entry.split("|")
    if importlib.util.find_spec(mod) is None:
        print(pip_pkg)
PY
  )
  if ((${#missing_pip[@]})); then
    pip_args=()
    if python3 -m pip help install 2>&1 | grep -q -- --break-system-packages; then
      pip_args+=(--break-system-packages)
    fi
    python3 -m pip install ${pip_args[@]+"${pip_args[@]}"} "${missing_pip[@]}"
  fi
else
  export DEBIAN_FRONTEND=noninteractive

  command -v debconf-set-selections >/dev/null 2>&1 &&
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" |
    sudo debconf-set-selections || true

  apt_packages=()
  # With --home, fonts come from downloads (columns 4/5), so no font apt
  # packages are collected. Prerequisites and Python deps still install
  # via apt.
  if [[ "$HOME_MODE" == 0 ]]; then
    for entry in "${FONTS_TO_INSTALL[@]}"; do
      IFS='|' read -r _ pkgs _ _ _ <<<"$entry"
      append_unique apt_packages $pkgs
    done
  fi
  for entry in "${PREREQ_TOOLS[@]}"; do
    pkgs="${entry%%|*}"
    append_unique apt_packages $pkgs
  done
  for dep in "${PYTHON_DEPS[@]}"; do
    rest="${dep#*|}"
    pkgs="${rest%%|*}"
    append_unique apt_packages $pkgs
  done

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

  apt_install install ${missing_packages[@]+"${missing_packages[@]}"}
  apt_install reinstall ${broken_packages[@]+"${broken_packages[@]}"}
fi

# ---------------------------------------------------------------------------
# Microsoft Core Fonts (pinned mirror + SHA-256 verification)
# ---------------------------------------------------------------------------
# SourceForge redirects are unreliable behind some proxies. Install the
# original Microsoft Core Fonts archives from a pinned mirror and verify every
# archive before extraction.
MS_CORE="$FONTS/msttcorefonts"
MS_CACHE="$HOME/.cache/microsoft-core-fonts"
MS_BASE="https://raw.githubusercontent.com/pushcx/corefonts/8f8627b0004b5f1b952c09cf6390e4ae46475e59"
[[ -d "$MS_CORE" ]] || mkdir -p "$MS_CORE"
[[ -d "$MS_CACHE" ]] || mkdir -p "$MS_CACHE"

georgia_ok() {
  printf '%s  %s\n' \
    "7d0bb20c632bb59e81a0885f573bd2173f71f73204de9058feb68ce032227072" "$MS_CORE/georgia.ttf" \
    "82d2fbadb88a8632d7f2e8ad50420c9fd2e7d3cbc0e90b04890213a711b34b93" "$MS_CORE/georgiab.ttf" \
    "1523f19bda6acca42c47c50da719a12dd34f85cc2606e6a5af15a7728b377b60" "$MS_CORE/georgiai.ttf" \
    "c983e037d8e4e694dd0fb0ba2e625bca317d67a41da2dc81e46a374e53d0ec8a" "$MS_CORE/georgiaz.ttf"
}

need_ms_extract=false
if ! georgia_ok | sha256_check --status 2>/dev/null; then
  need_ms_extract=true
fi

for entry in "${MS_ARCHIVES[@]}"; do
  read -r sha archive <<<"$entry"
  cached="$MS_CACHE/$archive"
  if ! printf '%s  %s\n' "$sha" "$cached" | sha256_check --status 2>/dev/null; then
    echo "  $archive"
    curl -fsSL -o "$cached.part" "$MS_BASE/$archive"
    printf '%s  %s\n' "$sha" "$cached.part" | sha256_check -
    mv "$cached.part" "$cached"
    need_ms_extract=true
  fi
  if [[ "$need_ms_extract" == true ]]; then
    cabextract -q -L -F '*.ttf' -d "$MS_CORE" "$cached"
  fi
done

# Georgia is part of the hardcoded audit fixture, so verify all four exact
# payloads instead of accepting fontconfig's fallback to another serif face.
georgia_ok | sha256_check -

# ---------------------------------------------------------------------------
# Typewriter fonts
# ---------------------------------------------------------------------------
TW="$FONTS/typewriter"
[[ -d "$TW" ]] || mkdir -p "$TW"
if ! [ -s "$TW/PrestigeElite-Regular.ttf" ]; then
  curl -fsSL -o "$TW/PrestigeElite-Regular.ttf" \
    "https://raw.githubusercontent.com/maseyyi/font-prestige-elite/master/prestige.ttf" || true
  curl -fsSL -o /tmp/pe-bold.zip "https://font.download/dl/font/prestige-elite-std.zip" || true
  unzip -j -o /tmp/pe-bold.zip "*.otf" -d "$TW/" || true
  curl -fsSL -o /tmp/lg.zip "https://font.download/dl/font/lettergothic.zip" || true
  unzip -j -o /tmp/lg.zip "*.ttf" -d "$TW/" || true
fi

# ---------------------------------------------------------------------------
# URW Base 35 (PostScript Level 2 core fonts, OTF)
# ---------------------------------------------------------------------------
# Real vintage PostScript fonts (Nimbus Roman = Times clone, Nimbus Sans =
# Helvetica clone, etc.) in OTF format, downloaded directly from the
# ArtifexSoftware/urw-base35-fonts release. The OTF files carry the full
# authentic kern data (Nimbus Roman Regular: 3845 GPOS PairPos pairs,
# matching the AFM's 3845 KPX entries). Used by the t66 vintage test.
# Direct download per Mike (Oct 31, 2026): "Don't use apt for the fonts."
URW="$FONTS/urw-base35"
URW_VER="20200910"
URW_URL="https://github.com/ArtifexSoftware/urw-base35-fonts/archive/refs/tags/${URW_VER}.tar.gz"
if ! [ -s "$URW/NimbusRoman-Regular.otf" ]; then
  [[ -d "$URW" ]] || mkdir -p "$URW"
  echo "  urw-base35 $URW_VER"
  curl -fsSL -o /tmp/urw-base35.tar.gz "$URW_URL" || echo "  WARN: urw-base35 download"
  if [[ -s /tmp/urw-base35.tar.gz ]]; then
    tar xzf /tmp/urw-base35.tar.gz -C /tmp "urw-base35-fonts-${URW_VER}/fonts/" || echo "  WARN: urw-base35 extract"
    cp /tmp/urw-base35-fonts-${URW_VER}/fonts/*.otf "$URW/" 2>/dev/null || echo "  WARN: urw-base35 copy"
    rm -rf /tmp/urw-base35.tar.gz "/tmp/urw-base35-fonts-${URW_VER}"
  fi
fi

# ---------------------------------------------------------------------------
# Windows 3.11 fonts (Win31-era TrueType, version 2.50)
# ---------------------------------------------------------------------------
# Real Windows 3.1-era TrueType fonts from the Internet Archive Windows 3.11
# stock installation (win3_stock / win311-stock.zip). These are the genuine
# early-1990s Monotype/Microsoft fonts: TIMES.TTF v2.50 has 113 kern pairs
# (vs 867 in the modern v2.82), ARIAL.TTF v2.50 has 105 pairs, COUR.TTF has
# 0 pairs (monospace). Used by the t66 vintage test as the real Win31 variant
# (replacing the unsourced quantized-kern assumption — the real Win31
# distinction is pair-count, not quantized values).
# (c) Monotype Corporation / Microsoft — NOT freely redistributable.
# Downloaded at install time only; never committed to the repo.
WIN31="$FONTS/win31"
WIN31_URL="https://archive.org/download/win3_stock/win311-stock.zip"
if ! [ -s "$WIN31/TIMES.TTF" ]; then
  [[ -d "$WIN31" ]] || mkdir -p "$WIN31"
  echo "  win31 2.50"
  curl -fsSL -o /tmp/win311-stock.zip "$WIN31_URL" || echo "  WARN: win31 download"
  if [[ -s /tmp/win311-stock.zip ]]; then
    unzip -j -o /tmp/win311-stock.zip "WINDOWS/SYSTEM/TIMES.TTF" "WINDOWS/SYSTEM/ARIAL.TTF" "WINDOWS/SYSTEM/COUR.TTF" -d "$WIN31/" || echo "  WARN: win31 extract"
    rm -f /tmp/win311-stock.zip
  fi
fi

# ---------------------------------------------------------------------------
# Google Fonts (direct upstream downloads from FONTS_TO_INSTALL)
# ---------------------------------------------------------------------------
for entry in "${FONTS_TO_INSTALL[@]}"; do
  IFS='|' read -r _ _ _ gf_paths _ <<<"$entry"
  read -r -a gf_list <<<"$gf_paths"
  for p in ${gf_list[@]+"${gf_list[@]}"}; do
    dest="$FONTS/$p"
    if [[ -s "$dest" ]]; then
      continue
    fi
    mkdir -p "$(dirname "$dest")"
    url="$GF/${p//\[/%5B}"; url="${url//\]/%5D}"
    echo "  $p"
    curl -fsSL -o "$dest" "$url" || echo "  WARN: $p"
  done
done

# ---------------------------------------------------------------------------
# Home-route downloads (--home only): fonts not in Google Fonts
# ---------------------------------------------------------------------------
# Column 5 tokens: URL>DEST downloads a single file to $FONTS/DEST;
# deb:URL>DESTDIR:GLOB downloads a .deb, extracts its data tarball, and
# copies files whose basename matches GLOB into $FONTS/DESTDIR.
# Already-installed files are skipped, like the Google Fonts loop above.
if [[ "$HOME_MODE" == 1 ]]; then
  for entry in "${FONTS_TO_INSTALL[@]}"; do
    IFS='|' read -r _ _ _ _ home_downloads <<<"$entry"
    read -r -a dl_list <<<"$home_downloads"
    for token in ${dl_list[@]+"${dl_list[@]}"}; do
      if [[ "$token" == deb:* ]]; then
        rest="${token#deb:}"
        url="${rest%%>*}"
        spec="${rest#*>}"
        destdir="${spec%%:*}"
        glob="${spec#*:}"
        if compgen -G "$FONTS/$destdir/$glob" >/dev/null; then
          continue
        fi
        [[ -d "$FONTS/$destdir" ]] || mkdir -p "$FONTS/$destdir"
        tmp_deb="$(mktemp --suffix=.deb /tmp/home-fonts-XXXXXX)"
        tmp_dir="$(mktemp -d /tmp/home-fonts-deb-XXXXXX)"
        echo "  $url"
        if curl -fsSL -o "$tmp_deb" "$url" &&
            (cd "$tmp_dir" && ar x "$tmp_deb" && tar -xf data.tar.*) &&
            found=false &&
            while IFS= read -r font_file; do
              cp "$font_file" "$FONTS/$destdir/"
              found=true
            done < <(find "$tmp_dir" -name "$glob" -type f); then
          [[ "$found" == true ]] || echo "  WARN: no $glob in $url" >&2
        else
          echo "  WARN: $url" >&2
        fi
        rm -f "$tmp_deb"
        rm -rf "$tmp_dir"
      else
        url="${token%%>*}"
        dest="${token#*>}"
        if [[ -s "$FONTS/$dest" ]]; then
          continue
        fi
        mkdir -p "$(dirname "$FONTS/$dest")"
        echo "  $dest"
        curl -fsSL -o "$FONTS/$dest" "$url" || echo "  WARN: $url"
      fi
    done
  done
fi

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
fc-cache -f "$FONTS"
for style in Regular Bold Italic "Bold Italic"; do
  resolved="$(fc-match -f '%{family[0]}|%{style[0]}|%{file}' "Georgia:style=$style")"
  if [[ "$resolved" != "Georgia|$style|$MS_CORE/"* &&
        ( "$OS" != "Darwin" || ( "$resolved" != "Georgia|$style|/System/Library/Fonts/"* &&
                                 "$resolved" != "Georgia|$style|/Library/Fonts/"* ) ) ]]; then
    echo "ERROR: Georgia $style resolved to $resolved" >&2
    exit 1
  fi
done

resolved="$(fc-match -f '%{family[0]}|%{style[0]}|%{file}' 'Open Sans:style=Regular')"
if [[ "$resolved" != "Open Sans|Regular|"* ]]; then
  echo "ERROR: Open Sans Regular resolved to $resolved" >&2
  exit 1
fi

# Verify the actual font payloads declared in FONTS_TO_INSTALL.  Checking only
# fc-match would allow a downloaded font with the same family to hide a broken
# package installation.
if [[ "$OS" == "Darwin" ]]; then
  package_font_files() {
    find "$CASKROOM/$1" \( -type f -o -type l \) 2>/dev/null | grep -Ei '\.(otf|ttf|ttc|pfb)$' || true
  }
else
  package_font_files() {
    dpkg-query -L "$1" | grep -Ei '\.(otf|ttf|ttc|pfb)$' || true
  }
fi

for entry in "${FONTS_TO_INSTALL[@]}"; do
  IFS='|' read -r expected_family apt_pkgs brew_casks gf_paths home_downloads <<<"$entry"
  if [[ "$OS" == "Darwin" ]]; then
    pkgs="$brew_casks"
  else
    pkgs="$apt_pkgs"
  fi
  package="${pkgs%% *}"

  if [[ "$HOME_MODE" == 1 ]]; then
    # Home route: fonts come from downloads, so there is no package payload
    # to verify. Every entry must declare a download in column 4 or 5.
    if [[ -z "$gf_paths" && -z "$home_downloads" ]]; then
      echo "ERROR: $expected_family has no home-route download (columns 4/5)" >&2
      exit 1
    fi
  elif [[ -n "$package" ]]; then
    found_family=false
    found_payload=false
    while IFS= read -r font_file; do
      [[ -n "$font_file" ]] || continue
      found_payload=true
      [[ -s "$font_file" ]] || {
        echo "ERROR: $package font payload is missing: $font_file" >&2
        exit 1
      }
      if file_has_family "$font_file" "$expected_family"; then
        found_family=true
        break
      fi
    done < <(package_font_files "$package")

    if [[ "$found_payload" != true ]]; then
      echo "ERROR: $package owns no installed font payload" >&2
      exit 1
    fi
    if [[ "$found_family" != true ]]; then
      echo "ERROR: $package does not contain expected family $expected_family" >&2
      exit 1
    fi
  fi

  read -r -a gf_list <<<"$gf_paths"
  for p in ${gf_list[@]+"${gf_list[@]}"}; do
    font_file="$FONTS/$p"
    [[ -s "$font_file" ]] || {
      echo "ERROR: missing Google Font file: $font_file" >&2
      exit 1
    }
    if ! file_has_family "$font_file" "$expected_family"; then
      echo "ERROR: $font_file does not contain expected family $expected_family" >&2
      exit 1
    fi
  done

  if [[ "$HOME_MODE" == 1 ]]; then
    read -r -a dl_list <<<"$home_downloads"
    for token in ${dl_list[@]+"${dl_list[@]}"}; do
      if [[ "$token" == deb:* ]]; then
        rest="${token#deb:}"
        spec="${rest#*>}"
        destdir="${spec%%:*}"
        glob="${spec#*:}"
        found_family=false
        found_payload=false
        while IFS= read -r font_file; do
          [[ -n "$font_file" ]] || continue
          found_payload=true
          [[ -s "$font_file" ]] || {
            echo "ERROR: $expected_family file is missing: $font_file" >&2
            exit 1
          }
          if file_has_family "$font_file" "$expected_family"; then
            found_family=true
            break
          fi
        done < <(compgen -G "$FONTS/$destdir/$glob")

        if [[ "$found_payload" != true ]]; then
          echo "ERROR: no $expected_family files extracted to $FONTS/$destdir" >&2
          exit 1
        fi
        if [[ "$found_family" != true ]]; then
          echo "ERROR: $FONTS/$destdir does not contain expected family $expected_family" >&2
          exit 1
        fi
      else
        dest="${token#*>}"
        font_file="$FONTS/$dest"
        [[ -s "$font_file" ]] || {
          echo "ERROR: missing home-route font file: $font_file" >&2
          exit 1
        }
        if ! file_has_family "$font_file" "$expected_family"; then
          echo "ERROR: $font_file does not contain expected family $expected_family" >&2
          exit 1
        fi
      fi
    done
  fi
done

# Verify every executable and Python dependency used by the fixture generator.
# Missing imports must fail installation now, not fail a job later.
command -v tesseract >/dev/null
tesseract --version >/dev/null
python3 - "${PYTHON_DEPS[@]}" <<'PY'
import importlib, sys
for entry in sys.argv[1:]:
    mod = entry.split("|", 1)[0]
    importlib.import_module(mod)
PY

echo "Done."
