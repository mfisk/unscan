# Font Aliasing, Classification & Ligature Detection

**Module:** `src/font_scan.rs`

---

## Why Aliasing Exists

Font files on disk have cryptic filenames that bear little resemblance to the
font's canonical name: `arialbd.ttf`, `nimbussans-regular.otf`,
`lmroman10-bold.otf`, `letr45w.ttf`. Without aliasing, these would appear in
unscan's catalog as `Arialbd`, `Nimbussans Regular`, `Lmroman10 Bold`, and
`Letr45w` — unrecognizable and useless for producing well-labeled output PDFs.

The alias table maps lowercase filename stems to canonical family names with
explicit bold/italic metadata.

---

## How It Works

`build_alias_table()` creates a `HashMap<String, Alias>` where each key is a
lowercase filename stem (e.g. `"arialbd"`) and the value is:

```rust
struct Alias {
    family: &'static str,  // canonical family name ("Arial")
    bold: bool,            // true for bold weights
    italic: bool,          // true for italic/oblique styles
}
```

When `load_font_entry()` loads a font file, it:

1. Lowercases the filename stem and strips spaces.
2. Looks up the result in the alias table (exact match).
3. **If found:** the font gets the canonical family name and weight/style from
   the alias. No filename parsing needed.
4. **If not found:** falls back to deriving the family name from the filename
   by replacing hyphens and underscores with spaces, then detecting bold/italic
   from keywords in the lowercased name.

---

## PDF Base-14 / URW Mapping

The PDF specification defines 14 standard fonts that every conforming viewer
must support:

| PDF name | URW clone on Linux |
|----------|-------------------|
| Helvetica | NimbusSans |
| Helvetica-Bold | NimbusSans-Bold |
| Times-Roman | NimbusRoman |
| Times-Bold | NimbusRoman-Bold |
| Courier | NimbusMonoPS |
| Courier-Bold | NimbusMonoPS-Bold |
| (+ italic variants) | (+ italic variants) |

Most Linux systems don't have the original Adobe fonts. Instead they ship URW
Nimbus clones (typically under `/usr/share/fonts/opentype/urw-base35/`). The
alias table maps these URW filenames to the PDF canonical names:

```
nimbussans-regular    → Helvetica
nimbusroman-regular   → Times-Roman
nimbusmonops-regular  → Courier
```

This ensures unscan's output PDF references the standard names that all viewers
understand, regardless of which physical font file was installed on the
processing machine.

---

## Categories Covered

### Microsoft Core Fonts

Arial (including Narrow and Black), Times New Roman, Courier New, Calibri
(including Light), Cambria, Verdana, Tahoma, Georgia, Trebuchet MS, Comic
Sans MS, Consolas, Segoe UI (including Semibold), Garamond, Aptos, Century
Gothic, Book Antiqua, Palatino Linotype.

Covers the cryptic Windows naming convention where `arialbd.ttf` = Arial Bold,
`calibriz.ttf` = Calibri Bold Italic, `trebucbi.ttf` = Trebuchet MS Bold
Italic, etc.

### LaTeX / TeX Fonts

Latin Modern Roman / Sans / Mono (various optical sizes: 10pt, 12pt), STIX
Two Text, TeX Gyre families (Termes, Heros, Pagella, Cursor, Bonum, Schola,
Adventor), Libertinus Serif / Sans.

These fonts use a `familyNNpt-style` naming pattern (e.g.
`lmroman10-regular.otf`) that doesn't parse cleanly without aliasing.

### PDF Base-14 via URW Clones

NimbusSans → Helvetica, NimbusSans Narrow → Helvetica Narrow,
NimbusRoman → Times-Roman, NimbusMonoPS → Courier.

All four standard weight/style variants (regular, bold, italic, bold-italic).

### Typewriter Fonts

OGCourier, CourierPrime, CutiveMono, SpecialElite, IBM Selectric Light,
Prestige Elite Std, Letter Gothic (including URW variant `letr45w`).

---

## Font Classification

After aliasing, each font is classified into one of four categories:

| Class | Purpose |
|-------|---------|
| Serif | Fonts with serifs (Times, Garamond, Baskerville, etc.) |
| Sans | Sans-serif fonts (Arial, Helvetica, Calibri, etc.) |
| Mono | Monospaced fonts (Courier, Consolas, etc.) |
| Unknown | Anything that doesn't match the hint lists |

Classification uses keyword matching against three arrays of lowercase
substrings (`SERIF_HINTS`, `SANS_HINTS`, `MONO_HINTS`). The match order is
mono → serif → sans (mono is checked first because monospaced fonts often
contain "serif" or "sans" in their names, e.g. "Noto Sans Mono"). The class
is stored in each `FontEntry` in the catalog.

---

## Ligature Glyph Detection

**Function:** `detect_ligature_glyphs()`

Unscan needs to know which fonts have ligature substitutions so it can match
ligature crops against the correct glyph in the character index.

### How It Works

For each of five standard ligature probes:

| Input string | Expected ligature | Unicode codepoint |
|-------------|-------------------|-------------------|
| `"ff"` | ff ligature | U+FB00 |
| `"fi"` | fi ligature | U+FB01 |
| `"fl"` | fl ligature | U+FB02 |
| `"ffi"` | ffi ligature | U+FB03 |
| `"ffl"` | ffl ligature | U+FB04 |

The function uses `rustybuzz` (a pure-Rust HarfBuzz port) to shape the probe
string with both `liga` (standard ligatures) and `dlig` (discretionary
ligatures) features enabled. If shaping produces a **single glyph** from
multiple input characters, that's a ligature substitution. The resulting glyph
ID is recorded as a `(ligature_char, glyph_id)` pair in the font entry.

During index construction, these ligature glyph IDs are used to render the
ligature character at its Unicode codepoint (U+FB00–FB04), producing a correct
feature vector for matching against ligature crops extracted from scanned text.

### Interaction with Segmentation

The segmentation pipeline runs a **dual-path** comparison:

- **Path A (plain):** OCR characters as-is (e.g., `f`, `f`, `i`).
- **Path B (ligature-collapsed):** Adjacent characters that form known ligature
  sequences are merged (e.g., `f` + `f` + `i` → `ffi`), reducing the target
  character count and producing wider crops that span the full ligature.

Both paths are scored by the character index. The path with the better CI score
wins (recorded as `seg_winner` in the audit). Ligature sequences are greedily
matched longest-first (`ffi` before `fi`).
