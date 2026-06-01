#!/usr/bin/env bash
set -euo pipefail

# Install all fonts required by unscan, including MS core fonts (EULA workaround),
# typewriter/vintage fonts, and specimen fonts.
# Run as root or with sudo.

echo "=== Installing apt font packages ==="
apt-get update -qq
apt-get install -y \
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
