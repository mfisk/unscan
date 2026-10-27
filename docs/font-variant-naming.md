# Font Variant Naming — Standards-Based Recipe

## OpenType Standard

Variable fonts follow the OpenType specification:
- Axis tags are 4-character lowercase codes: `wght`, `opsz`, `wdth`, `ital`, `slnt`
- Font filenames list supported axes in brackets: `FontName[axis1,axis2].ttf`
- Example: `SourceSerif4[opsz,wght].ttf`, `LibreBodoni[wght].ttf`

## Variant Naming Recipe

### Axes in Brackets with Values

Variable font entries use bracket notation with `axis=value` pairs:

```
BaseName[axis1=value1,axis2=value2]
```

Examples:
- `SourceSerif4-400Italic[wght=400,opsz=20]` — SourceSerif4 with wght 400, opsz 20
- `LibreBodoni-400[wght=400]` — LibreBodoni with wght 400 only (no opsz axis)

### Replace `|var` with Brackets

The bare `|var` suffix is eliminated. Bracket notation replaces it and makes all axis values explicit.

Old: `SourceSerif4-400Italic|var`
New: `SourceSerif4-400Italic[wght=400,opsz=20]`

### Features Keep `|` Separator

OpenType features (not axes) keep the `|` suffix:

```
BaseName[axes]|feature
```

Examples:
- `SourceSerif4-400Italic[wght=400,opsz=20]|nokern`
- `SourceSerif4-400Italic[wght=400,opsz=20]|onum`

### OpsZ Rule

For any font with an opsz axis, the opsz value MUST appear in the brackets. No variable entry may omit opsz if the font supports it.

### Static Fonts Unchanged

Non-variable fonts (no axes) keep their plain names:
- `Georgia-400`
- `OpenSans-400`

## Implementation Notes

- The `variations` field (structured list of axis-value pairs) remains separate from the key string
- The key string is derived from the variations: each axis in variations appears as `axis=value` in brackets
- Axis order in brackets follows the font's fvar table order
- Values are formatted as integers when whole, otherwise as decimal
