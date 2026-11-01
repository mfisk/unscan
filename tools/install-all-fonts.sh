#!/usr/bin/env bash
set -euo pipefail

FONTS="$HOME/.local/share/fonts"
GF="https://raw.githubusercontent.com/google/fonts/main"
OS="$(uname -s)"

# ---------------------------------------------------------------------------
# Single source of truth for fonts and prerequisites to install
# ---------------------------------------------------------------------------

# Unified font manifest:
#   "Expected Fontconfig Family|apt package(s)|brew cask(s)|Google Fonts path(s)"
#
# - Column 1: Family name verified via fc-scan (comma-separated if platform
#             versions use alternate family names).
# - Column 2: Debian/Ubuntu apt package(s) to install (first is verified via
#             dpkg-query -L + fc-scan).
# - Column 3: macOS Homebrew cask(s) to install (first is verified via
#             Caskroom + fc-scan).
# - Column 4: Upstream Google Fonts repo path(s) to download directly to
#             $FONTS/<path> and verify via fc-scan.
FONTS_TO_INSTALL=(
  "Carlito|fonts-crosextra-carlito|font-carlito|"
  "Caladea|fonts-crosextra-caladea|font-caladea|"
  "Liberation Sans|fonts-liberation|font-liberation|"
  "Latin Modern Roman,LMRoman10|fonts-lmodern|font-latin-modern|"
  "TeX Gyre Termes|fonts-texgyre texlive-fonts-recommended|font-tex-gyre-termes font-tex-gyre-heros font-tex-gyre-pagella font-tex-gyre-cursor|"
  "Noto Sans|fonts-noto-core fonts-noto fonts-noto-cjk fonts-noto-color-emoji fonts-noto-extra fonts-noto-mono fonts-noto-ui-core fonts-noto-ui-extra fonts-noto-unhinted|font-noto-sans font-noto-serif font-noto-sans-mono font-noto-color-emoji|"
  "Noto Serif|||ofl/notoserif/NotoSerif[wdth,wght].ttf"
  "Courier Prime|fonts-courier-prime|font-courier-prime|"
  "EB Garamond|fonts-ebgaramond|font-eb-garamond|"
  "IBM Plex Sans|fonts-ibm-plex|font-ibm-plex-sans|"
  "IBM Plex Serif|fonts-ibm-plex|font-ibm-plex-serif|"
  "IBM Plex Mono|fonts-ibm-plex|font-ibm-plex-mono|"
  "Inter|fonts-inter|font-inter|"
  "Lato|fonts-lato|font-lato|"
  "Open Sans|fonts-open-sans|font-open-sans|ofl/opensans/OpenSans[wdth,wght].ttf ofl/opensans/OpenSans-Italic[wdth,wght].ttf"
  "Roboto|fonts-roboto-unhinted fonts-roboto|font-roboto|"
  "Libre Baskerville|||ofl/librebaskerville/LibreBaskerville[wght].ttf ofl/librebaskerville/LibreBaskerville-Italic[wght].ttf"
  "Libre Bodoni|||ofl/librebodoni/LibreBodoni[wght].ttf ofl/librebodoni/LibreBodoni-Italic[wght].ttf"
  "Libre Caslon Text|||ofl/librecaslontext/LibreCaslonText[wght].ttf ofl/librecaslontext/LibreCaslonText-Italic[wght].ttf"
  "Zilla Slab|||ofl/zillaslab/ZillaSlab-Regular.ttf ofl/zillaslab/ZillaSlab-Bold.ttf"
  "Jost|||ofl/jost/Jost[wght].ttf"
  "Playfair Display|||ofl/playfairdisplay/PlayfairDisplay[wght].ttf ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf"
  "Merriweather|||ofl/merriweather/Merriweather[opsz,wdth,wght].ttf ofl/merriweather/Merriweather-Italic[opsz,wdth,wght].ttf"
  "Source Sans 3|||ofl/sourcesans3/SourceSans3[wght].ttf"
  "PT Serif|||ofl/ptserif/PT_Serif-Web-Regular.ttf ofl/ptserif/PT_Serif-Web-Italic.ttf ofl/ptserif/PT_Serif-Web-Bold.ttf"
  "PT Sans|||ofl/ptsans/PT_Sans-Web-Regular.ttf ofl/ptsans/PT_Sans-Web-Bold.ttf"
  "Source Serif 4|||ofl/sourceserif4/SourceSerif4[opsz,wght].ttf ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf"
  "Special Elite|||apache/specialelite/SpecialElite-Regular.ttf"
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
    IFS='|' read -r _ _ casks _ <<<"$entry"
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
  for entry in "${FONTS_TO_INSTALL[@]}"; do
    IFS='|' read -r _ pkgs _ _ <<<"$entry"
    append_unique apt_packages $pkgs
  done
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
# Google Fonts (direct upstream downloads from FONTS_TO_INSTALL)
# ---------------------------------------------------------------------------
for entry in "${FONTS_TO_INSTALL[@]}"; do
  IFS='|' read -r _ _ _ gf_paths <<<"$entry"
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
  IFS='|' read -r expected_family apt_pkgs brew_casks gf_paths <<<"$entry"
  if [[ "$OS" == "Darwin" ]]; then
    pkgs="$brew_casks"
  else
    pkgs="$apt_pkgs"
  fi
  package="${pkgs%% *}"

  if [[ -n "$package" ]]; then
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
          tr ',' '\n' | grep -Fxq -f <(printf '%s\n' "$expected_family" | tr ',' '\n'); then
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
    if ! fc-scan --format '%{family}\n' "$font_file" 2>/dev/null |
        tr ',' '\n' | grep -Fxq -f <(printf '%s\n' "$expected_family" | tr ',' '\n'); then
      echo "ERROR: $font_file does not contain expected family $expected_family" >&2
      exit 1
    fi
  done
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
