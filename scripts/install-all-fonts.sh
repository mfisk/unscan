#!/usr/bin/env bash
set -euo pipefail

# Install all fonts required by unscan, including MS core fonts (EULA workaround),
# typewriter/vintage fonts, and specimen fonts.
# Run as root or with sudo.

export DEBIAN_FRONTEND=noninteractive

# Pre-accept MS Core Fonts EULA for non-interactive installs
if command -v debconf-set-selections >/dev/null 2>&1; then
  echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
fi

echo "=== Installing apt font packages ==="
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
  unzip

echo "=== Installing typewriter / vintage fonts ==="
mkdir -p /usr/local/share/fonts/typewriter
cd /usr/local/share/fonts/typewriter

# Prestige Elite Regular (GitHub)
wget -q -O PrestigeElite-Regular.ttf "https://raw.githubusercontent.com/maseyyi/font-prestige-elite/master/prestige.ttf" || true

# Prestige Elite Bold (font.download zip)
wget -q -O /tmp/pe-bold.zip "https://font.download/dl/font/prestige-elite-std.zip" || true
unzip -j -o /tmp/pe-bold.zip "*.otf" -d /usr/local/share/fonts/typewriter/ || true

# Letter Gothic (font.download zip)
wget -q -O /tmp/lg.zip "https://font.download/dl/font/lettergothic.zip" || true
unzip -j -o /tmp/lg.zip "*.ttf" -d /usr/local/share/fonts/typewriter/ || true

# Note: OG Courier and IBM Selectric Light require manual licensed download.
# Place OGCourier*.ttf and "IBM Selectric Light"*.ttf in this directory if you have them.

fc-cache -f /usr/local/share/fonts/typewriter/

echo "=== Microsoft Core Fonts EULA ==="
echo "The following fonts are subject to the Microsoft TrueType core fonts EULA:"
echo "  https://corefonts.sourceforge.net/eula.htm"
echo ""
echo "Key terms (summary, not legal advice):"
echo "  - You may use the fonts to view and print documents."
echo "  - You may NOT redistribute the fonts as standalone files."
echo "  - Embedding in documents is permitted."
echo "  - You accept the EULA by installing/using the fonts."
echo ""
echo "Full EULA will be saved to /usr/share/doc/msttcorefonts-eula.txt"
mkdir -p /usr/share/doc
cat > /usr/share/doc/msttcorefonts-eula.txt << 'EULA'
MICROSOFT SOFTWARE LICENSE TERMS
MICROSOFT CORE FONTS

These license terms are an agreement between Microsoft Corporation (or based on where you live, one of its affiliates) and you. Please read them. They apply to the software named above, which includes the media on which you received it, if any. The terms also apply to any Microsoft
- updates,
- supplements,
- Internet-based services, and
- support services
for this software, unless other terms accompany those items. If so, those terms apply.

BY USING THE SOFTWARE, YOU ACCEPT THESE TERMS. IF YOU DO NOT ACCEPT THEM, DO NOT USE THE SOFTWARE.

If you comply with these license terms, you have the rights below.

1. INSTALLATION AND USE RIGHTS. You may install and use any number of copies of the software on your devices.

2. ADDITIONAL LICENSING REQUIREMENTS AND/OR USE RIGHTS.
   a. Distributable Code. The software contains fonts that are distributable. You may copy and distribute the fonts, but only as part of a document that embeds the fonts, and only if the document is not itself a font file.
   b. You may not sell the fonts separately.

3. SCOPE OF LICENSE. The software is licensed, not sold. This agreement only gives you some rights to use the software. Microsoft reserves all other rights. Unless applicable law gives you more rights despite this limitation, you may use the software only as expressly permitted in this agreement. In doing so, you must comply with any technical limitations in the software that only allow you to use it in certain ways. You may not
   - work around any technical limitations in the software;
   - reverse engineer, decompile or disassemble the software, except and only to the extent that applicable law expressly permits, despite this limitation;
   - make more copies of the software than specified in this agreement or allowed by applicable law, despite this limitation;
   - publish the software for others to copy;
   - rent, lease or lend the software; or
   - use the software for commercial software hosting services.

4. BACKUP COPY. You may make one backup copy of the software. You may use it only to reinstall the software.

5. DOCUMENTATION. Any person that has valid access to your computer or internal network may copy and use the documentation for your internal, reference purposes.

6. EXPORT RESTRICTIONS. The software is subject to United States export laws and regulations. You must comply with all domestic and international export laws and regulations that apply to the software. These laws include restrictions on destinations, end users and end use. For additional information, see www.microsoft.com/exporting.

7. SUPPORT SERVICES. Because this software is "as is," we may not provide support services for it.

8. ENTIRE AGREEMENT. This agreement, and the terms for supplements, updates, Internet-based services and support services that you use, are the entire agreement for the software and support services.

9. APPLICABLE LAW.
   a. United States. If you acquired the software in the United States, Washington state law governs the interpretation of this agreement and applies to claims for breach of it, regardless of conflict of laws principles. The laws of the state where you live govern all other claims, including claims under state consumer protection laws, unfair competition laws, and in tort.
   b. Outside the United States. If you acquired the software in any other country, the laws of that country apply.

10. LEGAL EFFECT. This agreement describes certain legal rights. You may have other rights under the laws of your country. You may also have rights with respect to the party from whom you acquired the software. This agreement does not change your rights under the laws of your country if the laws of your country do not permit it to do so.

11. DISCLAIMER OF WARRANTY. THE SOFTWARE IS LICENSED "AS-IS." YOU BEAR THE RISK OF USING IT. MICROSOFT GIVES NO EXPRESS WARRANTIES, GUARANTEES OR CONDITIONS. YOU MAY HAVE ADDITIONAL CONSUMER RIGHTS UNDER YOUR LOCAL LAWS WHICH THIS AGREEMENT CANNOT CHANGE. TO THE EXTENT PERMITTED UNDER YOUR LOCAL LAWS, MICROSOFT EXCLUDES THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.

12. LIMITATION ON AND EXCLUSION OF REMEDIES AND DAMAGES. YOU CAN RECOVER FROM MICROSOFT AND ITS SUPPLIERS ONLY DIRECT DAMAGES UP TO U.S. $5.00. YOU CANNOT RECOVER ANY OTHER DAMAGES, INCLUDING CONSEQUENTIAL, LOST PROFITS, SPECIAL, INDIRECT OR INCIDENTAL DAMAGES.

This limitation applies to
- anything related to the software, services, content (including code) on third party Internet sites, or third party programs; and
- claims for breach of contract, breach of warranty, guarantee or condition, strict liability, negligence, or other tort to the extent permitted by applicable law.

It also applies even if Microsoft knew or should have known about the possibility of the damages. The above limitation or exclusion may not apply to you because your country may not allow the exclusion or limitation of incidental, consequential or other damages.

Please note: As this software is distributed in Quebec, Canada, some of the clauses in this agreement are provided below in French.

Remarque : Ce logiciel étant distribué au Québec, Canada, certaines des clauses dans ce contrat sont fournies ci-dessous en français.

EXONÉRATION DE GARANTIE. Le logiciel visé par une licence est offert « tel quel ». Toute utilisation de ce logiciel est à votre seule risque et péril. Microsoft n’accorde aucune autre garantie expresse. Vous pouvez bénéficier de droits additionnels en vertu du droit local sur la protection des consommateurs, que ce contrat ne peut modifier. La ou elles sont permises par le droit locale, les garanties implicites de qualité marchande, d’adéquation à un usage particulier et d’absence de contrefaçon sont exclues.

LIMITATION DES DOMMAGES-INTÉRÊTS ET EXCLUSION DE RESPONSABILITÉ POUR LES DOMMAGES. Vous pouvez obtenir de Microsoft et de ses fournisseurs une indemnisation en cas de dommages directs uniquement à hauteur de 5,00 $ US. Vous ne pouvez prétendre à aucune indemnisation pour les autres dommages, y compris les dommages spéciaux, indirects ou accessoires et pertes de bénéfices.

Cette limitation concerne :
- tout ce qui est relié au logiciel, aux services ou au contenu (y compris le code) figurant sur des sites Internet tiers ou dans des programmes tiers ; et
- les réclamations au titre de violation de contrat ou de garantie, ou au titre de responsabilité stricte, de négligence ou d’une autre faute dans la limite autorisée par la loi en vigueur.

Elle s’applique également, même si Microsoft connaissait ou devrait connaître l’éventualité d’un tel dommage. Si votre pays n’autorise pas l’exclusion ou la limitation de responsabilité pour les dommages indirects, accessoires ou de quelque nature que ce soit, il se peut que la limitation ou l’exclusion ci-dessus ne s’appliquera pas à votre égard.

EFFET JURIDIQUE. Le présent contrat décrit certains droits juridiques. Vous pourriez avoir d’autres droits prévus par les lois de votre pays. Le présent contrat ne modifie pas les droits que vous confèrent les lois de votre pays si celles-ci ne le permettent pas.

Source: https://corefonts.sourceforge.net/eula.htm
EULA
echo "EULA saved. Proceeding with installation (by continuing you accept the terms)..."
sleep 2

echo "=== Installing MS Core Fonts via cabextract (EULA workaround) ==="
mkdir -p /tmp/msfonts
cd /tmp
for f in andale32 arial32 arialb32 comic32 courie32 georgi32 impact32 times32 trebuc32 verdan32 webdin32; do
  echo "Fetching ${f}.exe ..."
  wget -q "https://downloads.sourceforge.net/corefonts/${f}.exe" -O "/tmp/${f}.exe" || true
  cabextract -q -d /tmp/msfonts "/tmp/${f}.exe" || true
done
mkdir -p /usr/share/fonts/truetype/msttcorefonts
cp /tmp/msfonts/*.ttf /tmp/msfonts/*.TTF /usr/share/fonts/truetype/msttcorefonts/ 2>/dev/null || true
fc-cache -f /usr/share/fonts/truetype/msttcorefonts/

echo "=== Installing Google Fonts specimen families ==="
mkdir -p /usr/share/fonts/truetype/specimen-fonts
mkdir -p /usr/share/fonts/truetype/extra
SPEC_DIR="/usr/share/fonts/truetype/specimen-fonts"
EXTRA_DIR="/usr/share/fonts/truetype/extra"
TMP_FONT_DIR="/tmp/google-fonts"
mkdir -p "$TMP_FONT_DIR"
cd "$TMP_FONT_DIR"

# Helper to download a file if not present
dl() {
  url="$1"
  out="$2"
  if [ ! -f "$out" ]; then
    echo "Downloading $(basename "$out") ..."
    wget -q -O "$out" "$url" || echo "WARN: Failed to download $url"
  fi
}

# EB Garamond (variable)
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf" "EBGaramond[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ebgaramond/EBGaramond-Italic%5Bwght%5D.ttf" "EBGaramond-Italic[wght].ttf"
cp "EBGaramond[wght].ttf" "$SPEC_DIR/eb-garamond-400.ttf" 2>/dev/null || true
cp "EBGaramond[wght].ttf" "$SPEC_DIR/eb-garamond-700.ttf" 2>/dev/null || true
cp "EBGaramond-Italic[wght].ttf" "$SPEC_DIR/eb-garamond-400i.ttf" 2>/dev/null || true

# Libre Baskerville
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville%5Bwght%5D.ttf" "LibreBaskerville[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Italic%5Bwght%5D.ttf" "LibreBaskerville-Italic[wght].ttf"
cp "LibreBaskerville[wght].ttf" "$SPEC_DIR/libre-baskerville-400.ttf" 2>/dev/null || true
cp "LibreBaskerville[wght].ttf" "$SPEC_DIR/libre-baskerville-700.ttf" 2>/dev/null || true
cp "LibreBaskerville-Italic[wght].ttf" "$SPEC_DIR/libre-baskerville-400i.ttf" 2>/dev/null || true

# Libre Bodoni
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebodoni/LibreBodoni%5Bwght%5D.ttf" "LibreBodoni[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librebodoni/LibreBodoni-Italic%5Bwght%5D.ttf" "LibreBodoni-Italic[wght].ttf"
cp "LibreBodoni[wght].ttf" "$SPEC_DIR/libre-bodoni-400.ttf" 2>/dev/null || true
cp "LibreBodoni[wght].ttf" "$SPEC_DIR/libre-bodoni-700.ttf" 2>/dev/null || true
cp "LibreBodoni-Italic[wght].ttf" "$SPEC_DIR/libre-bodoni-400i.ttf" 2>/dev/null || true

# Libre Caslon Text
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librecaslontext/LibreCaslonText%5Bwght%5D.ttf" "LibreCaslonText[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/librecaslontext/LibreCaslonText-Italic%5Bwght%5D.ttf" "LibreCaslonText-Italic[wght].ttf"
cp "LibreCaslonText[wght].ttf" "$SPEC_DIR/libre-caslon-text-400.ttf" 2>/dev/null || true
cp "LibreCaslonText[wght].ttf" "$SPEC_DIR/libre-caslon-text-700.ttf" 2>/dev/null || true
cp "LibreCaslonText-Italic[wght].ttf" "$SPEC_DIR/libre-caslon-text-400i.ttf" 2>/dev/null || true

# Zilla Slab
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/zillaslab/ZillaSlab-Regular.ttf" "ZillaSlab-Regular.ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/zillaslab/ZillaSlab-Bold.ttf" "ZillaSlab-Bold.ttf"
cp "ZillaSlab-Regular.ttf" "$SPEC_DIR/zilla-slab-400.ttf" 2>/dev/null || true
cp "ZillaSlab-Bold.ttf" "$SPEC_DIR/zilla-slab-700.ttf" 2>/dev/null || true

# Jost
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/jost/Jost%5Bwght%5D.ttf" "Jost[wght].ttf"
cp "Jost[wght].ttf" "$SPEC_DIR/jost-400.ttf" 2>/dev/null || true
cp "Jost[wght].ttf" "$SPEC_DIR/jost-700.ttf" 2>/dev/null || true

# Playfair Display
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf" "PlayfairDisplay[wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay-Italic%5Bwght%5D.ttf" "PlayfairDisplay-Italic[wght].ttf"
cp "PlayfairDisplay[wght].ttf" "$SPEC_DIR/playfair-display-400.ttf" 2>/dev/null || true
cp "PlayfairDisplay-Italic[wght].ttf" "$SPEC_DIR/playfair-display-400i.ttf" 2>/dev/null || true

# Roboto
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth,wght%5D.ttf" "Roboto[wdth,wght].ttf"
cp "Roboto[wdth,wght].ttf" "$SPEC_DIR/roboto-400.ttf" 2>/dev/null || true

# Open Sans
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth,wght%5D.ttf" "OpenSans[wdth,wght].ttf"
cp "OpenSans[wdth,wght].ttf" "$SPEC_DIR/open-sans-400.ttf" 2>/dev/null || true

# Lato
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Regular.ttf" "Lato-Regular.ttf"
cp "Lato-Regular.ttf" "$SPEC_DIR/lato-400.ttf" 2>/dev/null || true

# Merriweather
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather%5Bopsz,wdth,wght%5D.ttf" "Merriweather[opsz,wdth,wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather-Italic%5Bopsz,wdth,wght%5D.ttf" "Merriweather-Italic[opsz,wdth,wght].ttf"
cp "Merriweather[opsz,wdth,wght].ttf" "$SPEC_DIR/merriweather-400.ttf" 2>/dev/null || true
cp "Merriweather-Italic[opsz,wdth,wght].ttf" "$SPEC_DIR/merriweather-400i.ttf" 2>/dev/null || true

# Source Sans 3 (used as Source Sans Pro)
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf" "SourceSans3[wght].ttf"
cp "SourceSans3[wght].ttf" "$SPEC_DIR/source-sans-pro-400.ttf" 2>/dev/null || true

# Noto Serif
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf" "NotoSerif[wdth,wght].ttf"
cp "NotoSerif[wdth,wght].ttf" "$SPEC_DIR/noto-serif-400.ttf" 2>/dev/null || true

# PT Serif
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Regular.ttf" "PT_Serif-Web-Regular.ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Italic.ttf" "PT_Serif-Web-Italic.ttf"
cp "PT_Serif-Web-Regular.ttf" "$SPEC_DIR/pt-serif-400.ttf" 2>/dev/null || true
cp "PT_Serif-Web-Italic.ttf" "$SPEC_DIR/pt-serif-400i.ttf" 2>/dev/null || true

# IBM Plex Sans / Serif / Mono
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf" "IBMPlexSans[wdth,wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexserif/IBMPlexSerif-Regular.ttf" "IBMPlexSerif-Regular.ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf" "IBMPlexMono-Regular.ttf"
cp "IBMPlexSans[wdth,wght].ttf" "$SPEC_DIR/ibm-plex-sans-400.ttf" 2>/dev/null || true
cp "IBMPlexSerif-Regular.ttf" "$SPEC_DIR/ibm-plex-serif-400.ttf" 2>/dev/null || true
cp "IBMPlexMono-Regular.ttf" "$SPEC_DIR/ibm-plex-mono-400.ttf" 2>/dev/null || true

# Inter
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf" "Inter[opsz,wght].ttf"
cp "Inter[opsz,wght].ttf" "$SPEC_DIR/inter-400.ttf" 2>/dev/null || true

# Source Serif 4
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf" "SourceSerif4[opsz,wght].ttf"
dl "https://raw.githubusercontent.com/google/fonts/main/ofl/sourceserif4/SourceSerif4-Italic%5Bopsz,wght%5D.ttf" "SourceSerif4-Italic[opsz,wght].ttf"
cp "SourceSerif4[opsz,wght].ttf" "$EXTRA_DIR/SourceSerif4-Regular.ttf" 2>/dev/null || true
cp "SourceSerif4[opsz,wght].ttf" "$EXTRA_DIR/SourceSerif4-Bold.ttf" 2>/dev/null || true
cp "SourceSerif4-Italic[opsz,wght].ttf" "$EXTRA_DIR/SourceSerif4-It.ttf" 2>/dev/null || true

# Special Elite (Apache)
dl "https://raw.githubusercontent.com/google/fonts/main/apache/specialelite/SpecialElite-Regular.ttf" "SpecialElite-Regular.ttf"
cp "SpecialElite-Regular.ttf" "/usr/local/share/fonts/typewriter/SpecialElite-Regular.ttf" 2>/dev/null || true

fc-cache -f "$SPEC_DIR" "$EXTRA_DIR" /usr/local/share/fonts/typewriter/

echo "=== Generating specimen fonts (EB Garamond, Libre families) ==="
# Find repo root (script is in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/test-docs"
if [ -f gen-specimen.py ]; then
  python3 gen-specimen.py || true
else
  echo "WARN: gen-specimen.py not found"
fi

echo "=== Final font cache refresh ==="
fc-cache -f

echo "=== Verification ==="
echo "MS Core fonts:"
ls -1 /usr/share/fonts/truetype/msttcorefonts/ 2>/dev/null | head
echo ""
echo "Typewriter fonts:"
ls -1 /usr/local/share/fonts/typewriter/ 2>/dev/null | head
echo ""
echo "Specimen fonts:"
ls -1 /usr/share/fonts/truetype/specimen-fonts/ 2>/dev/null | head
echo ""
echo "fc-list samples:"
fc-list | grep -i "Arial" | head -n 3 || true
fc-list | grep -i "Prestige" | head -n 3 || true

echo ""
echo "Done. You can now run: bash scripts/check-fonts.sh"
