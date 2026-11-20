# Branch critique

Before every commit, always critique the change, including the errors and redundancies it is adding and how it could be better. Address those items before committing. A merge is a commit: the change under critique is the whole branch. (Mike Nov 20, 2026.)

## How to critique

- Read the full diff first, then read the code the diff touches.

- For each change ask what defect it fixes, at which layer, and whether the change removes the defect or layers over it.

- Check every measurement in the diff against the scale the pipeline actually sees, using the wrong-scale measurement class below.

- Check every proposal in the branch or in your own critique against the code: does it already exist, can it exist given the code's invariants, and does it contradict a design comment.

- Verify the BAP numbers from the frozen audits yourself with targeted extraction, recompute the deltas, and check per-line attribution for every differing line.

- Check the three regression axes below, each against the baseline BAP, for changes that can move the numbers. Verify with a BAP where applicable. Docs-only and test-only changes carry the unchanged baseline score in the commit message instead.

- Present three parts with file and line references and measured numbers: errors the branch adds, redundancies it adds, and how it could be better.

## Regression checks

Check all three axes against the baseline BAP and report each separately. They can move in different directions: the kern fix gained 63 font hits while losing 0.0030 mean ZNCC.

- Runtime: warm-cache audit time. Baseline BAP 1315 audit 254s. Identical code measured 254 to 390s, so judge by re-measurement and mechanism, not single runs. A runtime regression blocks merging.

- ZNCC: mean ZNCC with p50 and p90. Baseline BAP 1364: ZNCC 0.9334, p50 0.9582, p90 0.9782.

- Font matching: font hits (exact plus major-only, where exact is a key match at or above 0.9 and major-only is a key match below 0.9) and headline hits (hit plus minor miss). Baseline BAP 1364: font hits 289/506, headline hits 453/506.

## Principles

### Correctness and measurement

- Correctness is the universal goal in everything. Metrics measure correctness; they are not to be gamed with looser tolerances, sigma tweaks, or fuzzy matching. (AGENTS.md)

- Fix input quality at the source. When geometry or naming is wrong, fix the geometry or naming. Do not wallpaper over it. (AGENTS.md)

- Fix things right. Do the refactor. Do not ship a minimal workaround that leaves a design wart, such as an opt-in `Option` closure or a silently-skipped code path, to avoid the refactor. (AGENTS.md)

- Deliberate changes only. No speculative fixes. Only the main code path. No one-off diagnostics. Input quality over fancier algorithms. (AGENTS.md)

- Fixes must be general, not overfit to a particular case. (Mike Nov 20, 2026.) A fix that works for the reported case and fails the same class elsewhere is overfit: the 363a942 gutter fix fixed Arial on p3 L34/L48 and broke CourierNew, SourceSans3, and Merriweather on p3 L9, p3 L18, and p5 L15 by the same mechanism in reverse.

- All measurements in the context of the low-res word scans the pipeline receives, about 40px em at 300 dpi, never at high-res outline truth. (Mike Nov 20, 2026.)

- Index renders must match the scan's downsample ratio. Real scan crops come from 300 dpi rasterization where typical body text is about 40 to 50px before normalization to 24px, about a 2x downsample. Index entries rendered at 72px or 200px and downsampled to 24px have different Lanczos3 artifact profiles than real scan crops, and the classifier then learns the wrong pixel domain. The hires 200px experiment scored 14.3% primary hits against 18.8% for the matched render. (RENDERING-INVESTIGATION.md)

- Start with the character crops. If a crop is wrong, no scoring refinement can fix the match. Fix extraction first. (DEBUGGING.md)

- On clean input the correct font must be first or tied for first at every stage. Anything else is a bug to fix. (DEBUGGING.md)

- Narrow glyphs (I, l, t, i) are the canary for geometry mismatches. Check them first. (DEBUGGING.md)

- Use only actual tool output for diagnosis. Never reimplement feature computation or character rendering in Python. (DEBUGGING.md)

### One canonical implementation

- One canonical implementation of a thing. No duplication. (Mike Nov 20, 2026.) A second implementation, a mirror, or a hand-port is a critique finding.

- One canonical font lookup implementation resolves a name to a specific (file, variant, size). The Rust scanner, the specimen generator, and the ground-truth reader all use the same lookup, and no downstream code re-derives or normalizes names on its own. (DEBUGGING.md)

- Index build and scan lookup must produce identical images for the same glyph via the identical code path, not merely equivalent code. (DEBUGGING.md)

- Index-build and index-lookup must use identical crop geometry. (AGENTS.md)

- Rendering goes through one shared function, not three separate render functions across modules. (docs/cleanup-opportunities.md)

- Training, index, and audit must use the exact same rendering function (`render_glyph_normalised`), so all three live in the same image domain. (RENDERING-INVESTIGATION.md)

- Both scan and training/index paths share `normalize_to_ink_bounds` (ink bbox threshold 200, 1px pad, Lanczos3 resize to NORM_H) as the canonical downsample. (RENDERING-INVESTIGATION.md, FEATURES.md)

- Rust is the single source of truth. Python delegates through the resolve helper. No mirror. Never hand-port a change across branches. The stale branch gets main merged into it instead.

- Magic numbers must be named constants. (CLEANUP.md)

- Names must not lie. A function named `sha256_prefix` that uses DefaultHasher (SipHash) must be renamed. (CLEANUP.md)

### Scoring and geometry

- Geometry is midpoint-only: score observed ink centers and inter-character pitches against font predictions, with no other geometry term. (docs/midpoint-geometry.md)

- Vertical error is word-relative, zero-centered by construction, sum of v_err equals 0. (docs/midpoint-geometry.md)

- Sigmas are first-principles uniform-quantization values (sigma_center 1/sqrt(12), sigma_pitch 1/sqrt(6)) with inflection at 0.75 sigma per dimension, not a shared absolute threshold. (docs/midpoint-geometry.md)

- CI #1 wins directly. There is no word-level SSIM reranking. SSIM is a verification gate, not a selector. (docs/text-matching-approach.md, docs/char-index-methodology.md)

- The font score is the generative log-likelihood, sum of w times lp, which is order-preserving under subsetting. A score using a data-dependent best term breaks pruning soundness. (docs/text-matching-approach.md, docs/midpoint-geometry.md)

- The fast-path gate (0.90) must stay much higher than the verify gate (0.3) because the fast path accepts a font without CI evidence. Lower fast-path thresholds caused accuracy regressions. (docs/text-matching-approach.md)

- A font with no geometry data for a line is kept. It cannot be pruned for geometry. Midpoint pruning never prunes a font with no geometry data, re-adds the best-pruned fonts to MIN_KEEP=10, and never prunes to empty. (docs/text-matching-approach.md, docs/midpoint-geometry.md)

- A missing glyph is a categorical disqualification (MISSING_GLYPH_PENALTY is infinity), not a tunable threshold. A font that cannot render even one character in the line is a non-starter. (docs/char-index-methodology.md)

- Conjunction: gate on the seam's darkest pixel reaching 0.4 of the word's max_ink. Do not gate on dp_cost greater than 0: dp_cost includes the horizontal-movement penalty, so a seam moving diagonally through white space has dp_cost greater than 0 with no ink. The correction subtracts the full overlap (overlap_px is 1.0 plus max(-gap_px, 0)), clamps the corrected magnitude at 0, keeps the sign of the original h_err, and computes h_ll from the corrected h_err with the original sigma. No sigma widening. No halving: the shift is systematic, not uniform. (AGENTS.md)

- Seam semantics: the pixel at seam_x belongs to the RIGHT character (right seam exclusive, left inclusive). (docs/midpoint-geometry.md)

- Ink bounds use binary pixel less than 200. The weighted centroid uses pixel less than 250 with mass 255 minus p. (docs/midpoint-geometry.md)

- Font size is determined from OCR bounding-box height, not width, because height-matching is more stable. (docs/text-matching-approach.md)

- Natural font metrics: do not distort spacing to match OCR bboxes. (docs/text-matching-approach.md)

- Fix inputs, not algorithms: every SSIM failure traces to bad inputs (illustration contamination, garbage OCR, wrong crop geometry). (docs/text-matching-approach.md)

- L2 group normalization is load-bearing: it creates the shape-comparison geometry, and every attempt to replace it with per-feature weighting regressed AA accuracy. (docs/FEATURE_WEIGHTS.md)

- Structural fixes beat parametric tuning: the Noto dedup (structural) was worth 8x more misses than any weight miscalibration. (docs/FEATURE_WEIGHTS.md)

- Fisher analysis is diagnostic, not prescriptive: applying Fisher-optimal weights traded 300 DPI accuracy for DPI robustness. (docs/FEATURE_WEIGHTS.md)

- `v_center` is a trap: high Fisher ratio, but it does not discriminate between fonts in the same class and made the wrong font closer. (docs/FEATURE_WEIGHTS.md)

- Fonts sharing identical Latin outlines (the Noto superfamily) must be deduped into one candidate slot, or they crowd out the correct font after the sigma cutoff. (docs/FEATURE_WEIGHTS.md)

- Score-epsilon dedup (epsilon 1e-5) merges exact clones that produce bitwise-identical feature vectors. (docs/FEATURE_WEIGHTS.md)

- Out-of-distribution observations must be neutralized: confidence blending (softmax toward uniform as confidence goes to 0) and OOD observation weighting (min(1, med_nn / min_d)). (docs/ml-classifier-design.md)

- Sigma squared for the Gaussian kernel is the median pairwise squared distance between centroids, which is robust to outlier centroids. (docs/ml-classifier-design.md)

- Per-group L2 normalization plus Fisher-tuned weights is the current weighting, and group balance must be revisited whenever features are added. (docs/char-index-features-research.md)

- Scan crops and rendered references must be compared in the same feature domain. Systematic offsets between scan and render (anti-aliasing artifacts, skew, thresholds) can make the wrong font nearest. (docs/char-index-features-research.md)

- Brute-force linear scan is the production search method at this scale. Tree structures provide no pruning benefit at 99 dimensions. (docs/ml-classifier-design.md, docs/char-index-features-research.md)

- Top-5 accuracy is the relevant pre-filter metric, not top-1, since the pre-filter only needs the correct font in the candidate set. (docs/font-matching-research.md)

- VLMs are not useful for font recognition. The best barely beats the random baseline. (docs/font-matching-research.md)

- Font identification is a retrieval problem (top-50 from about 5k classes, 1 sample per class per character), not a classification problem. Nearest-neighbor with a good distance metric is the right tool. (docs/ml-classifier-design.md)

- Pure Rust, no Python in the hot path. The index trains at runtime during the indexing step. (docs/ml-classifier-design.md)

### Segmentation

- Zero-ink columns are definitive inter-character gaps. VP splits at run midpoints, and both sides of every split must have `min_ink_for_symbol` ink. (SEGMENTATION.md)

- `min_ink_for_symbol` scales with the square of crop height ((0.07 times h) squared times 255) because ink area grows with the square of font size. (SEGMENTATION.md)

- The segment-size penalty is zero when the split column has no ink. A clean gap is always free regardless of segment sizes. (SEGMENTATION.md)

- Trailing narrow punctuation (`. , : ;`) is exempt from the segment-size penalty. The left child width is used instead. (SEGMENTATION.md)

- Only local minima in candidate cost enter the greedy heap, and among equal-cost runs the middle column is selected. (SEGMENTATION.md)

- Ties resolve to the straight vertical path (neighbors iterated [c, c-1, c+1] with strict less-than comparison). (SEGMENTATION.md)

- Child segments inherit the accepted seam as a diagonal boundary with INFINITY energy on the wrong side. Paths must not cross into sibling territory. (SEGMENTATION.md)

- Seam-occupied ink belongs to one side by darkest-adjacent ownership and is never dropped. The same logic is mirrored in three places and must stay in lockstep. (SEGMENTATION.md)

- Every accepted split must have ink on both sides (`min_ink_for_symbol`), rejecting splits that would create empty fragments. (SEGMENTATION.md)

- If the algorithm cannot produce enough splits, fall back to uniform boundaries. (SEGMENTATION.md)

- Ligature collapse uses greedy longest-first matching: `ffi`/`ffl` (3 to 1) before `ff`/`fi`/`fl` (2 to 1). (docs/debugging-segmentation.md, docs/font-aliasing.md)

- The winning segmentation path is the one with the higher top CI score, and the losing path's CI results are kept in the audit for diagnostic comparison. (docs/debugging-segmentation.md)

- Ligature detection shapes probe strings with `liga` AND `dlig` enabled via rustybuzz. A single glyph from multiple input characters is a ligature substitution. (docs/font-aliasing.md)

- Do not fix ligature mismatches by disabling ligatures in the source PDF. The ground truth is correct, and the fix belongs in unprint's rendering. (DEBUGGING.md)

- Tesseract is used only for word bounding boxes and text. Character positions come from our own segmentation, because Tesseract character-level boxes are unreliable. (DEBUGGING.md)

- Always crop to ink bounds first. Before segmenting, matching, or computing features on a word or character image, trim it down to just the ink. Use `trim_whitespace()` or `normalize_to_ink_bounds()` before doing anything with the image. (AGENTS.md)

### Fonts and naming

- Font names must match exactly with `==`. Never fuzzy match, normalize, extract families, or use weight-keyword equivalence. If names do not match, fix the naming. (AGENTS.md, DEBUGGING.md)

- Font names must match exactly as stored in the index ("DejaVuSans", no space). The original "Noto beats DejaVu" result was a test bug from searching "DejaVu Sans". (docs/noto-vs-dejavu-analysis.md)

- Variable-font axis values go in brackets as `axis=value` pairs. The bare `|var` suffix is eliminated. (docs/font-variant-naming.md)

- OpenType features keep the `|` separator and come after the brackets: `BaseName[axes]|feature`. (docs/font-variant-naming.md)

- For any font with an opsz axis, the opsz value MUST appear in the brackets. (docs/font-variant-naming.md)

- Axis order in brackets follows the font's fvar table order. Values are integers when whole, otherwise decimal. (docs/font-variant-naming.md)

- Static (non-variable) fonts keep plain names with no brackets. (docs/font-variant-naming.md)

- Alias lookup is an exact match on the lowercase filename stem. If not found, fall back to a filename-derived name with keyword bold/italic detection. (docs/font-aliasing.md)

- URW Nimbus clones map to the PDF Base-14 canonical names (Helvetica, Times-Roman, Courier) so output PDFs reference standard names. (docs/font-aliasing.md)

- Classification match order is mono, then serif, then sans, because monospaced fonts often contain "serif" or "sans" in their names. (docs/font-aliasing.md)

- Never stub fonts: install the real font. Do not copy DejaVu or other placeholders to satisfy missing font files. (docs/FONT_INSTALL.md)

- The pipeline must have the exact fonts installed that appear in test documents, otherwise font matching and accuracy tests fail. (docs/FONT_INSTALL.md)

- OpenType feature variants are matched as separate candidates and scored in the CI like any other font entry. (README.md)

- Font resolution must validate OS/2 weight and width classes, never trust fontconfig name/style filters alone. Picking the first `fc-list` result for "Lato Regular" without checking weight returns Lato-Hairline (250). (docs/TESTING.md)

- Vintage fonts: never generate a vintage that could not have existed (the shipping gate and birth gate both apply; unknown birth year means skip). Never redistribute old fonts: vintages are built locally from modern free bases already on the box. Vintage variants do not collapse in dedup (dedup keys on the full `font_key` including `vintage_era`). `postscript_name` is never mangled; the era lives in the separate `vintage_era` field. A built vintage font must validate with `FontRef::new`; on failure, log a warning and fall back to the original bytes, never cache a broken file. (docs/vintage-font-spacing.md)

### Caches and state

- All persistent caches are read-only after creation and built via staging file plus atomic rename. Readers never see a partially-written file, and no cache is ever truncated in place. (docs/caching.md)

- Cache staleness is detected via `catalog_hash` (hash of sorted font_keys). A hash mismatch means rebuild or retrain, not corruption. (docs/caching.md)

- Never filter inside `scan_fonts` (which controls `catalog_hash`). Filter only at matching time. (docs/caching.md)

- All disk caches use minimal necessary precision: f32 floats, u32 counts and offsets, u64 only for hashes and counters. No f64 in any cache file. (docs/debugging-segmentation.md)

- In-memory caches follow the FontCache LRU contract (bounded capacity, promotion on get, eviction, hits and misses stats). Unbounded per-font caches OOM the VM. (docs/debugging-segmentation.md)

- All post-index font access goes through the shared FontCache. No direct file reads of font data. (docs/debugging-segmentation.md, docs/OPTIMIZATION-PLAN.md)

- `INDEX_VERSION` must be bumped when the feature vector or character set changes, so stale index files are rejected and rebuilt automatically. (docs/char-index-methodology.md, FEATURES.md)

- Do not write large files to `/tmp` (RAM-backed, can OOM). Use `~/.cache/unprint/pages/` instead. (CLEANUP.md)

- The training data cache lives in `~/.cache/unprint/training/`, not in the project directory. (RENDERING-INVESTIGATION.md)

### Tests and reporting

- t55 must run before t60, t61, and t62. t60 through t62 assert fixtures exist and fail immediately if they do not, with no fallback generation. (docs/TESTING.md)

- Ground truth comes from the vector PDF's own font dictionaries (`/Widths` and `Tw`), not from an external fontmap file. (docs/TESTING.md)

- Regenerate the specimen whenever `gen-specimen.py` changes or fonts are installed or updated. (docs/TESTING.md)

- Lines with no ground-truth overlap (NoGT) are excluded from the accuracy denominator. (docs/TESTING.md)

- After any accuracy test that is not 100%, examine the miss report and understand every miss before proposing fixes. (docs/debugging-segmentation.md)

- Always verify ground truth before investigating a CI miss (show the actual page image). (DEBUGGING.md)

- `--include-font` bypasses CI and can make the included font win lines it should not, so accuracy numbers from an `--include-font` run are not trustworthy. (DEBUGGING.md)

- Report ligatures: when showing characters in the report, always include ligature characters. If a character is part of a ligature, the report must show the ligature glyph (U+FB01), not just the separate f and i glyphs. (AGENTS.md)

- Report lig-vs-nolig comparison: when the report compares a lig font against a nolig font, show both renderings side by side, each side rendered according to its own font key's lig/nolig state. The comparison is shown IN ADDITION TO the top 6 deltas in the observation table, not instead of them. (AGENTS.md)

- Commit messages report the BAP score (hits out of the specimen line count, ZNCC, p50, p90). A child branch reads the BAP score from the commit message and uses it as its baseline instead of recomputing a BAP. (AGENTS.md)

- A line is vectorized only if it passes OCR confidence, font-match score, AND SSIM at or above 0.3. All others keep the original raster. (README.md)

- Aggressive vectorisation of high-confidence text, conservative fallback: when in doubt, keep the original pixels. (README.md)

- Zero information loss: remaining raster stays at full resolution with lossless (FlateDecode/PNG) encoding. (README.md)

- Full audit trail: a JSON sidecar log documents every decision. (README.md)

- Tesseract must use the `tessdata_best` eng model (15MB), not the apt model. (README.md)

- PDF placement uses one em per line and one baseline, with no character spacing (Tc) and no horizontal scaling, and fonts subsetted to used glyphs only. (docs/text-matching-approach.md)

- Do not reconstruct seam data by hand. The viz must run on real recorded data. (AGENTS.md)

- The OCR correction gate substitutes only when the OCR character's best match is greater than 0.1 AND greater than 10x worse than an alternative. Moderate distances check confusable pairs only, and catastrophic distances (greater than 0.5) scan all characters. (FEATURES.md)

- Character extraction takes the longest words first (3 or more characters), up to 3 samples per character, stopping when every observed character has 2 or more samples. 1 to 2 character words are excluded because segmentation is impossible. (docs/char-index-methodology.md)

- Height normalization uses the full word-crop height, not the individual character's ink height, so all characters from one word keep consistent scale. (docs/char-index-methodology.md)

- Normalization uses ink height, not em height, so features describe actual visual appearance independent of the font's internal metrics. (docs/char-index-methodology.md)

- Ligatures are indexed with weight 2.0 as a binary, highly distinctive signal. (docs/char-index-methodology.md)

- Dual-path ligature comparison uses OOD-weighted scores only, with no position weights. (docs/text-matching-approach.md)

- Pass 1.5 detects the dominant font but does not override individual line matches. (docs/text-matching-approach.md)

### Performance

- Optimizations change no algorithm and cause no regressions. (docs/OPTIMIZATION-PLAN.md)

- Audit-only work must be gated behind `args.audit.is_some()`. (docs/OPTIMIZATION-PLAN.md)

- Lines are independent: per-line font matching is parallelizable, and the paragraph-grouping pass must run after all line results as a sequential post-pass. (docs/performance-audit.md)

- The font catalog, page image, and char index are read-only during matching and safe to share across threads. (docs/performance-audit.md)

- Compute shared inputs once and thread them through (one `to_luma8`, one source blur, one font parse) rather than recomputing per line or per candidate. (docs/performance-audit.md)

- Do not store font bytes in results. Use `Arc` or references instead of cloning `Vec<u8>` font data. (docs/performance-audit.md)

- Do not clone font data per call. Use thread-local or shared resources (thread-local FreeType Library, no `font_data.to_vec()`). (docs/perf-opt-2026-05-11.md)

- Pass offsets into kernels instead of creating shifted image copies. (docs/scaling-alignment-methodology.md, docs/perf-opt-2026-05-11.md)

- Do not recompute what an earlier stage already computed: Pass 2 must reuse the SSIM score from matching (the `ssim_verified` flag) instead of re-verifying. (docs/perf-opt-2026-05-11.md)

- The SSIM render path and the PDF output path must share the same arithmetic via `layout.rs`. No drift between the two. (docs/scaling-alignment-methodology.md)

- Font-size smoothing (median em) applies only to PDF output, never to SSIM ranking. (docs/scaling-alignment-methodology.md)

- The vertical shift search runs center-outward with early exit at SSIM at or above 0.92, passing shift offsets into the SSIM kernel rather than creating shifted image copies. (docs/scaling-alignment-methodology.md)

- The ink-centered baseline model is documented as conceptually wrong (it systematically misaligns lines without descenders). The correct model is baseline-aligned. (docs/scaling-alignment-methodology.md)

- Sub-pixel shift interpolation and horizontal shift search are explicitly not worth adding. They would not change font ranking. (docs/scaling-alignment-methodology.md)

- OCR correction must early-exit or pre-filter. Checking all 106 characters with a full brute-force search per bad crop is the dominant cost. (docs/PROFILE-ANALYSIS.md)

- `split_wide_whitespace` should reuse the main-pass font instead of running its own CI search per line. (docs/PROFILE-ANALYSIS.md)

- Build only the binary you need (`cargo build --bin unprint`), not all binaries. Use release builds only for final accuracy runs. Do not `cargo clean` unless needed. sccache is NOT enabled because it slows incremental builds; use it only for clean builds. (FAST-COMPILE.md)

- Unused dependencies must be removed from Cargo.toml. (docs/cleanup-opportunities.md)

- Character segmentation and the full pipeline need isolated and integration tests. Changing `segment.rs` with no test covering the changed path is a finding. (docs/cleanup-opportunities.md)

- After each change, validate with t60 (at or above 93% accuracy). (docs/OPTIMIZATION-PLAN.md)

- The line bbox is derived from the word union. Word bboxes are authoritative (the separate line-bbox expansion was removed as redundant). (docs/OPTIMIZATION-PLAN.md)

## Mistake classes

Recorded from the November 20, 2026 gutter-fix exchange. Each class names the mistake, the evidence, and what to check.

### Wrong-scale measurement

- Measure pixel-level features in the context of the low-res word scans the pipeline receives, about 40px em at 300 dpi, never at high-res outline truth.

- A derived scale conversion on top of the wrong measurement does not fix it. The 363a942 gutter fix kept the 200px outline measurement and added a 10-column bar derived from a 2px render-scale rule, and Mike rejected it: no 200px, per character at render scale.

- The evidence for the class: Arial's U+FB01 gutter is 4px at 200px outline truth, which is 0.8px at render scale, so the ink is fused and the 200px test mismatches the render. The same fix misfired in reverse on CourierNew, SourceSans3, and Merriweather, where the mask collapsed 'fi' while the render kept 'f' and 'i' separate, losing three exact hits.

- The same class was already documented: RENDERING-INVESTIGATION.md shows hires-to-downsample rendering creates a domain mismatch against real scan crops.

### Proposing what already exists

- Check whether the proposed thing is already in the code with grep and git blame before proposing it.

- One k per candidate already existed when it was proposed as a fix: `let k = collapsed.len()` per word per font in build_for_font, one segmentation per candidate.

### Proposing what is impossible by construction

- Check the proposal against the code's invariants before presenting it.

- K is per-font by construction, collapsed.len() under each font's own ligature mask, so one shared k across candidates is impossible. Two fonts with different masks necessarily have different ks for the same word.

- A proposal that contradicts a documented design comment is wrong until the design itself is challenged with evidence. The line 1296 comment in src/font_pipeline.rs says k differs correctly, with Office k=4 versus k=5 as the example.

### Muddled framing

- Restate any proposal in the code's own terms and check the restatement before presenting it.

- "One k per candidate" and "one k across candidates" are different claims. The first exists, the second is impossible, and conflating them produced a proposal that removed nothing.

### Fixing the symptom, piling on

- A guard, conversion, or threshold layered over a defective measurement is not a fix. Ask whether the change removes the defect or adds a layer over it.

- The better fix removes the defective measurement. Judging fusion from the render deletes the scale conversion and the rasterizer disagreement instead of adding to them.

### A principled bar on the wrong quantity

- A bar derived from first principles is still wrong when the quantity it gates does not match the thing being judged.

- The 2px bar was principled, since a 2px interval always contains a full pixel column regardless of subpixel phase, but the 200px measurement it gated was not the render.

### Invented mechanisms and names

- Use the code's own terms and define them from the code when asked. Report measured numbers and equations, not paraphrase.

- Every mechanism claim must trace to code or measured audit data. When a claim has no measured evidence behind it, say so plainly and say "I do not have data" when unmeasured.

- Never propose new terminology. Two attempts to propose new metric names were both rejected as invented.

### Cross-candidate comparison across different observation sets

- Scores computed on different ks are compared by the ranking. When a change alters k for some fonts and not others, check the affected lines' gt_rank and gt_score in the frozen audits.

- The Arial case: Arial scored on 24 observations against Roboto on 23, and the phantom 'i' carried geo_ll -72.01.

### Rasterizer disagreement near a bar

- When two rasterizers disagree by 1 to 2px near a threshold, the threshold is fragile. SourceSans3 FB01 measured 11px under PIL and under 10 columns under ab_glyph.

- Measure with the pipeline's own renderer rather than a second rasterizer.

### Unverified claims

- Verify every subagent claim against pueue, git, and the filesystem firsthand. A subagent report is a lead, not a result.

- Personally examine the primary evidence for every claim: the code, the image, the numbers. Verify the output before reporting a fix as landed.

### Redundancy

- Flag duplicated logic, a second implementation of an existing helper, and a Python mirror of Rust logic. Rust is the single source of truth and Python delegates through the resolve helper.

### Metric chasing

- No threshold, tolerance, or sigma tweaks to push a metric. The fix must be a genuine correction at the owning layer, never a tolerance tweak.

- Metrics measure correctness. They are not the target.
