# Font Matching & Identification: State of the Art Survey

> Research survey for the `unprint` project — converting scanned PDFs back to vector text by identifying which fonts were used.
>
> Last updated: 2025-07-14  
> **Status note (June 2026):** Since this survey was written, the architecture
> has changed significantly. Word-level SSIM reranking has been removed; CI #1
> wins directly. The feature vector has grown from 36 to 99 dimensions with 5
> weighted groups. A parallel SSIM fast path (dominant font from previous page,
> threshold 0.90) was added. See `docs/text-matching-approach.md` for the
> current pipeline description.

---

## Table of Contents

1. [Our Current Approach](#1-our-current-approach)
2. [Academic Approaches](#2-academic-approaches)
   - 2.1 DeepFont (Adobe, 2015)
   - 2.2 CNN-based Font Classification
   - 2.3 Metric Learning & Embedding Approaches
   - 2.4 Prototypical Networks / Few-Shot Learning
   - 2.5 Traditional Feature Extraction
   - 2.6 Eigenfaces for Fonts
   - 2.7 FontCLIP and Vision-Language Models
   - 2.8 OCR-Integrated Font Detection (Tesseract)
3. [Commercial / Production Systems](#3-commercial--production-systems)
   - 3.1 WhatTheFont (MyFonts/Monotype)
   - 3.2 Adobe Photoshop Match Font
   - 3.3 Identifont
   - 3.4 Font Squirrel Matcherator / Fontspring Matcherator
   - 3.5 IDEO Font Map (Google Fonts)
4. [Indexing Approaches](#4-indexing-approaches)
   - 4.1 Perceptual Hashing
   - 4.2 Feature-Based Indexing & Taxonomic Features
   - 4.3 Hierarchical Classification
   - 4.4 Locality-Sensitive Hashing (LSH)
   - 4.5 Learned Embeddings + ANN
5. [Evaluation Benchmarks & Datasets](#5-evaluation-benchmarks--datasets)
6. [Comparison with Our Approach](#6-comparison-with-our-approach)
7. [Recommendations for unprint](#7-recommendations-for-unprint)

---

## 1. Our Current Approach

For context, here's what unprint currently does for font matching (updated June 2026):

**Per-character feature classification** (`classifier.rs`, `features.rs`):
- Renders ~106 printable characters in each of ~5048 fonts at 48px (NORM_H)
- Extracts a **99-float feature vector** per character, in 5 weighted groups:
  - Column ink profile (32 bins, weight 0.40)
  - Scalar v1 (7 dims, weight 0.30): aspect, ink_density, v_center, h_balance, serif_score, stroke_contrast, xh_cap_ratio
  - Scalar v2 (18 dims, weight 0.30): counters (4), terminal angles (4), shape (2), horizontal crossings (8)
  - Row ink profile (32 bins, weight 0.30)
  - Scalar v3 (10 dims, weight 0.20): holes, symmetry, skeleton topology, corners, quadrant density
- Per-group L2 normalization + Fisher-tuned weights
- Matching via brute-force linear scan with squared Euclidean distance
- **CI #1 wins directly** — no word-level SSIM reranking

**SSIM fast path** (dominant font acceleration):
- Before CI, each line tries the dominant font from the previous page via SSIM
- Threshold ≥ 0.90: accept and skip segmentation + CI entirely
- All lines run in parallel (`rayon::par_iter`)
- Candidate updated after each page by font-key frequency tally

**SSIM verification gate** (not reranking):
- After CI selects a font, `verify_text_region()` renders via FreeType and
  compares SSIM. If SSIM < 0.3 (`MIN_VERIFY_SSIM`), the line reverts to raster.
- This is a gate, not a selector — it doesn't choose between candidates.

**Key constraints**:
- Input: 300 DPI scanned documents (not wild scene text or photos)
- ~5048 candidate fonts (system fonts, not millions)
- Sub-second per-line matching after indexing
- Unlimited offline pre-computation budget
- Rust implementation (no Python ML in the hot path)
- Current accuracy: **454/480 (94.6%)** on 30-font specimen

---

## 2. Academic Approaches

### 2.1 DeepFont (Adobe, 2015)

**Paper**: Wang et al., "DeepFont: Identify Your Font from An Image," ACM MM 2015.
**URL**: https://arxiv.org/abs/1507.03196

The seminal work on deep-learning font recognition. Key details:

**Architecture**:
- CNN with shared low-level convolutional layers (Sub-CNN) feeding into class-specific higher layers
- Input: 105×105 pixel text patches (no character segmentation needed)
- Trained on synthetic data rendered from 2,383 font classes
- **Stacked Convolutional Auto-Encoder (SCAE)** for domain adaptation between synthetic training data and real-world images

**Results**:
- **>80% top-5 accuracy** on real-world AdobeVFR test set (2,383 classes)
- Significant improvement over prior art which relied on hand-crafted features
- Also produces a **font similarity measure** — the CNN's penultimate-layer features serve as a perceptual embedding space

**Key innovations**:
- Domain adaptation via SCAE: trains on synthetic rendered text but generalizes to real photos
- **6× model compression** via low-rank factorization without accuracy loss
- Training data augmentation: noise, blur, perspective distortion, varied backgrounds, different text strings

**Relevance to unprint**: ⭐⭐⭐ (High for ideas, medium for direct use)
- DeepFont targets "wild" text in photos — much harder than our 300 DPI scans
- The domain adaptation problem barely applies to us since we can render reference glyphs under the same conditions as the scan
- The **penultimate-layer CNN embedding as a similarity measure** is directly relevant — we could train a small CNN to produce per-character embeddings that capture font style better than hand-crafted features
- 2,383 classes is comparable to our 5,048 fonts

### 2.2 CNN-based Font Classification

**Tensmeyer et al., "Convolutional Neural Networks for Font Classification," ICDAR 2017**
**URL**: https://arxiv.org/abs/1708.03669

- CNN trained on **densely extracted patches** from text lines
- Achieves **98.8% line-level accuracy** on Arabic font classification and **86.6% page-level accuracy**
- Key insight: averaging CNN predictions over many patches from the same line/page dramatically improves accuracy vs single-patch prediction
- Features learned by CNN trained on Latin manuscripts transfer to identifying scribal script classes

**ResNet18 font identifier (Cselle, 2023)**
**URL**: https://github.com/gaborcselle/font-identifier

- Simple ResNet18 finetuned on rendered text images
- **95%+ accuracy** on a test set of 2,400 images
- Built in 1 day as a proof of concept
- Confusion primarily between genuinely similar fonts (Helvetica vs Arial, Trebuchet vs Verdana)
- Dataset: synthetic rendered images per font on HuggingFace

**Relevance to unprint**: ⭐⭐⭐⭐ (High)
- A small CNN (even ResNet18) achieves remarkably high accuracy on rendered font classification
- For our use case, we could train a CNN on our exact 5,048 fonts with rendered text and use it as the primary pre-filter
- The "average over many patches" insight from Tensmeyer maps directly to our per-line matching — score multiple characters and aggregate
- Key advantage: CNN features capture relationships humans describe (serif style, stroke contrast, letter proportions) without explicit feature engineering

### 2.3 Metric Learning & Embedding Approaches

**Adobe Triplet Loss Patent (US10515295B2, 2019)**
**URL**: https://patents.google.com/patent/US10515295B2/en

- Adobe patented font recognition using **triplet loss neural network training**
- Architecture: CNN produces an embedding; trained with triplets (anchor, positive=same font, negative=different font)
- Key advantage over classification: works for **open-set** recognition — can match fonts not seen during training by comparing embeddings
- Directly addresses font similarity (embeddings close = visually similar fonts)

**Font Representation Learning via Paired-Glyph Matching (Cho et al., 2022)**
**URL**: https://arxiv.org/abs/2211.10967

- Learns font embeddings by training on pairs of glyphs from same/different fonts
- Better generalization than classification-based approaches
- Enables font retrieval and style transfer
- Source code available

**Contrastive Learning for Font Style Classification (2023)**
**URL**: https://www.mdpi.com/2076-3417/13/6/3635

- Compares normalized temperature-scaled cross-entropy loss, triplet loss, and supervised contrastive loss for multilingual font classification
- Contrastive learning achieves comparable results to fully supervised methods with fewer labeled examples and training epochs
- Demonstrates robustness across Latin, Arabic, and other scripts

**Relevance to unprint**: ⭐⭐⭐⭐⭐ (Very high)
- **This is probably the most promising direction for improving our pre-filter**
- Train a CNN with triplet/contrastive loss on rendered characters from our 5,048 fonts
- Produces a fixed-dimensional embedding per rendered character
- At match time: render the scanned character's shape → compute embedding → nearest-neighbor search against pre-computed font embeddings
- Handles the "open set" problem — if we add new fonts, we just compute their embeddings without retraining
- Works at per-character level which maps directly to our pipeline

### 2.4 Prototypical Networks / Few-Shot Learning

**Font-ProtoNet (Goel et al., CVPR Workshop 2020)**
**URL**: https://openaccess.thecvf.com/content_CVPRW_2020/html/w34/Goel_Font-ProtoNet...

- Uses prototypical networks for font identification with minimal labeled data
- **59.86% word-level accuracy (1-shot)** and **71.01% (5-shot)** on AdobeVFR
- Trains on 200 font classes, tests on 100 novel classes
- Learns an embedding space where class prototypes are centroids

**Relevance to unprint**: ⭐⭐ (Limited)
- Few-shot is less relevant since we have unlimited synthetic data for all our fonts
- The prototypical network concept is useful though: represent each font as a centroid in embedding space, classify by nearest centroid
- The accuracy numbers are lower than classification-based approaches because of the extreme few-shot constraint

### 2.5 Traditional Feature Extraction

**Zramdini & Ingold, "Optical Font Recognition Using Typographical Features," IEEE TPAMI 1998**

Classic paper establishing the typographic features that discriminate fonts:

| Feature | What it measures | Discriminative power |
|---------|-----------------|---------------------|
| **Serif presence** | Whether strokes have terminal flourishes | Very high — separates two major font families |
| **Stroke contrast** | Ratio of thick to thin strokes | High — distinguishes modern from old-style |
| **x-height / cap-height ratio** | Relative size of lowercase to uppercase | Medium-high — varies significantly across families |
| **Weight** | Average stroke width relative to body size | High within families |
| **Width** | Character width relative to height | Medium |
| **Italic angle** | Slant of vertical strokes | High for style variants |
| **Letter spacing** | Inter-character gaps | Low-medium |

**Zhu, Tan & Wang, "Font Recognition Based on Global Texture Analysis," IEEE TPAMI 2001**
- Uses Gabor filters to capture text "texture" at multiple scales and orientations
- Treats a block of text as a texture and classifies the texture
- Works well for document-level font classification

**Bozkurt et al., "Classifying Fonts and Calligraphy Styles Using Complex Wavelet Transform," 2014**
**URL**: https://arxiv.org/abs/1407.2649
- Complex wavelet transform + SVM for font classification
- Achieves higher accuracy than Gabor-based approaches
- Applied to Ottoman manuscript classification

**Relevance to unprint**: ⭐⭐⭐ (Medium-high for specific features)
- We're already using some of these concepts (ink density ≈ weight, aspect ratio ≈ width)
- **Missing from our features**: serif detection, stroke contrast, x-height/cap-height ratio, counter shapes
- Adding serif detection and stroke contrast as discrete features could dramatically improve coarse filtering
- Gabor/wavelet texture features could be a good middle ground between hand-crafted and CNN — more discriminative than our column profiles but cheaper than a full CNN

### 2.6 Eigenfaces for Fonts (EigenFonts)

**Al-Khaffaf & Musa, "Optical English Font Recognition in Document Images Using Eigenfaces"**
**URL**: https://doi.org/10.15649/2346075X.466

- Applies PCA-based Eigenfaces (Turk & Pentland, 1991) to font recognition
- Treats each rendered character image as a high-dimensional vector; PCA extracts principal components
- **99% accuracy on synthetic data**, **97% on degraded data** (6,144 degraded samples)
- Only tested on 3 fonts (Comic Sans, DejaVu Sans Condensed, Times New Roman)
- Uses Euclidean distance in eigenspace for classification

**Relevance to unprint**: ⭐⭐⭐ (Medium)
- PCA on rendered character images is essentially what we'd get from a linear dimensionality reduction of pixel data
- The 97% on degraded data is impressive but on only 3 fonts — doesn't tell us about 5,048-font discrimination
- Our column ink density profile is a hand-designed version of this — PCA would find the optimal projection automatically
- **Practical idea**: for each character, render all 5,048 fonts, compute PCA on the rendered images, and use the top-K principal components as features. This is a drop-in improvement over our 32-bin profiles.

### 2.7 FontCLIP and Vision-Language Models

**FontCLIP (2024)**
**URL**: https://arxiv.org/abs/2403.06453
- Connects CLIP's vision-language knowledge with typographic expertise
- Enables semantic font retrieval ("find me a font that looks elegant and modern")
- Generalizes across languages

**VLMs for Font Recognition (2025)**
**URL**: https://arxiv.org/abs/2503.23768
- Evaluates GPT-4o, Claude, Gemini, Llama, Qwen on font recognition
- **VLMs are terrible at font recognition**: best model (Qwen2-VL-72B) barely beats random baseline
- Font style accuracy peaks at 23-34% vs 25% random baseline on style classification
- VLMs can read text perfectly but can't identify how it's rendered

**Relevance to unprint**: ⭐ (Low)
- VLMs are not useful for our task
- FontCLIP is interesting for semantic search but overkill for our exact-match identification problem

### 2.8 OCR-Integrated Font Detection (Tesseract)

Tesseract's approach to fonts:

- Tesseract does **not** identify fonts in its standard OCR pipeline
- It uses a fixed set of trained fonts during its training phase (defined in `training/language-specific.sh` and `langdata/font_properties`)
- The LSTM-based engine (Tesseract 4+) doesn't track which font produced a recognition — it's trained on multiple fonts to be font-invariant
- Older Tesseract (pre-LSTM) had per-font character prototypes and would implicitly select the best-matching font, but this was never exposed as an API

**Xerox/Google OCR Font Formatting Patent (US6741745B2)**
**URL**: https://patents.google.com/patent/US20020076111A1/en
- Method for determining typeface from scanned OCR text
- Key technique: **match word widths, not character widths** — word-level width matching is much more accurate
- For each candidate typeface, render each recognized word and compute scaling factors
- Look for **clusters of consistent scaling factors** — the correct typeface produces a tight cluster at the true font size
- Uses plausible size ranges to handle measurement noise

**Relevance to unprint**: ⭐⭐⭐⭐ (High — the width clustering insight)
- We already do width-matched scaling, but the **clustering of scaling factors** insight is powerful
- If the correct font is used, the font-size needed to match each word should be consistent across words in the same paragraph
- We could use **consistency of required font size** as an additional signal: the correct font should require the same pt size for every word, while wrong fonts will need varying scale factors
- This is essentially free to compute since we already calculate scale factors

---

## 3. Commercial / Production Systems

### 3.1 WhatTheFont (MyFonts / Monotype)

**How it works**:
- Image upload → text detection + segmentation → glyph isolation → ML matching
- Database: **130,000-230,000+ fonts** (numbers vary by source)
- Trained on **33 million images** (per FastCompany interview)
- Claims **~90% accuracy** on clean input
- Uses deep learning (likely CNN-based, details not public since Monotype acquisition)

**Key capabilities**:
- Handles multiple fonts in a single image
- Manual cropping for improved accuracy
- Returns ranked list of matches with confidence

**Limitations**:
- Struggles with low-quality images and rare/decorative fonts
- Requires reasonably clean, isolated text
- Black-box — no published technical details on current architecture

**Relevance to unprint**: ⭐⭐ (Their problem is harder than ours)
- WhatTheFont solves a harder problem (wild images, huge font space) but we need higher accuracy on a narrower problem
- Our advantage: we know exactly what characters we're matching, have 300 DPI input, and only need to search 5,048 fonts

### 3.2 Adobe Photoshop Match Font

**How it works**:
- User selects text region in an image → Photoshop identifies the font
- Powered by DeepFont technology (see §2.1) with Adobe Sensei
- Searches both installed local fonts and Adobe Fonts cloud library
- Can distinguish serif/sans-serif, weight, style

**Architecture** (inferred from patents):
- Adobe patent US10127199B2: "Automatic Measure of Visual Similarity Between Fonts"
  - Defines a **model** with: letterforms, keypoints on each letterform, detail shapes at specific areas, geometric measurements
  - For each character image: identify corresponding letterform → locate keypoints → classify detail shapes (serif terminals, ball terminals, etc.) → measure geometric ratios
  - Similarity = function of differences between visual descriptors
- Adobe patent US10515295B2: Triplet loss CNN for font embedding
  - CNN trained with triplet loss produces an embedding vector per text image
  - Similar fonts map to nearby points in embedding space

**Relevance to unprint**: ⭐⭐⭐⭐ (The patent details are directly applicable)
- The visual descriptor model (US10127199B2) is essentially a refined version of typographic features — we could implement the same keypoint + detail shape + measurement approach
- The triplet loss embedding (US10515295B2) is the most directly relevant approach

### 3.3 Identifont

**How it works**:
- **Question-based narrowing**: asks users a series of questions about typographic features
- Feature taxonomy includes:
  - Serif or sans-serif?
  - Serif type (bracketed, unbracketed, slab, wedge)?
  - x-height (small, medium, large)?
  - Weight (light, book, medium, bold, black)?
  - Width (condensed, normal, extended)?
  - Stroke contrast (none, low, medium, high)?
  - Specific letter shapes: 'a' single/double story? 'g' single/double story? 'Q' tail shape? 'R' leg shape? etc.

**Feature taxonomy** (approximately 20-30 categorical features):
- Serif: none / line / slab / wedge / bracketed / hairline
- Cap height vs ascender: equal / cap shorter
- x-height: small / medium / large  
- Weight: thin / light / regular / medium / bold / black
- Width: compressed / condensed / normal / wide / extended
- Contrast: none / low / medium / high / extreme
- Specific characters: 'a' stories, 'g' form, 'J' descender, 'Q' tail, etc.

**Relevance to unprint**: ⭐⭐⭐ (Feature taxonomy is gold)
- The specific **categorical features** Identifont uses are the distilled wisdom of typographers
- Many of these could be computed automatically from rendered glyphs:
  - Serif detection: render 'I' or 'l', check for horizontal protrusions at baseline and cap line
  - x-height ratio: render 'x' and 'H', measure relative heights
  - Weight: stroke width / em-square ratio
  - Contrast: max stroke width / min stroke width
  - Specific character shapes: 'a' single/double story detection via topology
- Building an **automatic Identifont** as a hierarchical pre-filter could be very effective

### 3.4 Font Squirrel Matcherator / Fontspring Matcherator

**How it works**:
- Upload image → glyph detection and segmentation → per-character matching
- Advanced glyph detection with OpenType feature matching
- Database: 900,000+ fonts (Fontspring)
- Users can manually input letters for small caps correction

**Technical approach** (inferred):
- Per-character glyph isolation (contour detection)
- Template matching against rendered glyph database
- May use outline/contour comparison rather than pixel comparison

**Relevance to unprint**: ⭐⭐ (Similar to our approach, but less documented)

### 3.5 IDEO Font Map (Google Fonts)

**How it works**:
- Used **VGG16 CNN** to extract features from rendered font specimens
- Applied **t-SNE** dimensionality reduction to create a 2D map of 800+ Google Fonts
- Fonts that look similar cluster together on the map

**Architecture**:
- Render each font as a specimen image (pangram or similar)
- Pass through pretrained VGG16 → extract activations from a deep layer
- These activations serve as a "font fingerprint"
- t-SNE or UMAP for visualization; cosine similarity for retrieval

**Relevance to unprint**: ⭐⭐⭐ (The pretrained-CNN-as-feature-extractor idea)
- Using a pretrained image CNN (VGG16, ResNet) as a feature extractor is simple and effective
- No font-specific training needed — just forward pass through the network
- Could replace our hand-crafted 99-float vector with a ~512-float CNN feature vector
- Fast: single forward pass per character, can be batched

---

## 4. Indexing Approaches

### 4.1 Perceptual Hashing

**Concept**: Generate a compact hash of each glyph image such that similar images produce similar (or identical) hashes.

**Common algorithms**:
- **aHash (average hash)**: Resize to 8×8, threshold against mean → 64-bit hash
- **dHash (difference hash)**: Resize to 9×8, compare adjacent pixels → 64-bit hash
- **pHash (perceptual hash)**: DCT of resized image, threshold against median → 64-bit hash

**Google Patent US7664323B2: Scalable Hash-Based Character Recognition**:
- Uses hash tables for character recognition at scale
- Generates hash from rendered character images
- Auxiliary recognition for hash collisions

**Applicability to font matching**:
- Hash the same character rendered in different fonts
- Same character + same font = identical hash (for clean renders)
- Same character + similar font = close hash (Hamming distance)
- Problem: perceptual hashes are designed for near-duplicate detection, not fine-grained similarity
- **pHash is the most promising** — DCT captures frequency information similar to our column profiles

**Relevance to unprint**: ⭐⭐⭐ (Good for fast coarse filtering)
- pHash of rendered glyphs at standard size could serve as an ultra-fast first-pass filter
- Hamming distance is trivially fast (XOR + popcount)
- Could reduce 5,048 → ~200 candidates in microseconds
- But likely too coarse for the fine discrimination we need between similar fonts
- **Best used as an additional coarse pre-filter layer before our current cosine similarity**

### 4.2 Feature-Based Indexing & Taxonomic Features

Based on typographic literature and the Identifont taxonomy, the most discriminative measurable features for font identification are:

| Feature | How to compute | Discrimination power | Implementation difficulty |
|---------|---------------|---------------------|--------------------------|
| **Serif presence/type** | Render 'I'/'l', detect horizontal protrusions at baseline & cap | Very high | Medium |
| **Stroke contrast** | Render 'O', measure max/min stroke width via distance transform | Very high | Medium |
| **x-height / cap-height** | Render 'x' and 'H', measure ink bbox heights | High | Easy |
| **Weight (stroke width)** | Average distance transform value inside ink regions | High | Easy |
| **Aspect ratio** | Character width / height | Medium-high | Easy (we have this) |
| **Counter area** | Measure enclosed white space in 'e', 'a', 'o', 'p' | Medium-high | Medium |
| **Ink density** | Ratio of ink pixels to total bbox | Medium | Easy (we have this) |
| **a-form** | Single-story vs double-story 'a' — topology check | High | Medium-hard |
| **g-form** | Single-story vs double-story 'g' — topology check | High | Medium-hard |
| **Terminal shapes** | Ball, flat, tapered — requires contour analysis at stroke endpoints | Medium | Hard |
| **Column ink profile** | Vertical projection histogram | Medium | Easy (we have this) |
| **Vertical center of mass** | y-centroid of ink pixels | Low-medium | Easy (we have this) |

**Key insight**: Serif detection + stroke contrast + weight + x-height ratio alone can partition fonts into ~20-50 coarse classes, reducing the search space by 100×.

### 4.3 Hierarchical Classification

A multi-stage pipeline:

```
Stage 1: Binary classifiers (microseconds)
├── Serif vs Sans-Serif
├── Monospace vs Proportional  
├── Italic vs Upright

Stage 2: Coarse class (sub-millisecond)  
├── Serif: Old-style / Transitional / Modern / Slab
├── Sans: Grotesque / Neo-grotesque / Geometric / Humanist
├── Weight: Light / Regular / Medium / Bold / Black
├── Width: Condensed / Normal / Extended

Stage 3: Fine matching (milliseconds)
├── Feature vector similarity (our current approach or CNN embeddings)
├── Top-N candidates

Stage 4: Verification (10s of ms)
├── SSIM reranking
```

**Relevance to unprint**: ⭐⭐⭐⭐⭐ (Very high — directly implementable)
- This maps perfectly to our pipeline: coarse → fine → verify
- We're missing Stage 1 and Stage 2 entirely — jumping straight to fine matching across all 5,048 fonts
- Adding even basic serif/sans-serif + weight classification could cut our search space in half

### 4.4 Locality-Sensitive Hashing (LSH)

**Concept**: Hash high-dimensional vectors such that similar vectors have high probability of colliding.

**How it works for font matching**:
1. Compute feature vector for each character in each font (our 99-float vector, or a CNN embedding)
2. Build LSH index with multiple hash tables
3. At query time, hash the query vector → retrieve candidates from same buckets
4. Score only the candidates (not all 5,048 fonts)

**Practical implementations**:
- Random projection LSH (SimHash): works for cosine similarity
- FAISS (Facebook): highly optimized library for similarity search
- Annoy (Spotify): tree-based approximate nearest neighbors
- ScaNN (Google): state-of-the-art ANN library

**Performance**: For 5,048 fonts × ~80 characters = ~400K vectors, ANN search returns top-K in <1ms even with brute force. LSH becomes more valuable at millions of vectors.

**Relevance to unprint**: ⭐⭐ (Marginal benefit at our scale)
- With only 5,048 fonts, brute-force linear scan across all fonts is already fast enough (~microseconds for 99-float vectors with SIMD auto-vectorization)
- LSH/ANN would be more useful if we had 100K+ fonts or if we moved to higher-dimensional embeddings (e.g., 256-dim CNN features)
- FAISS could be useful if we adopt CNN embeddings — but the query overhead of running the CNN dominates anyway

### 4.5 Learned Embeddings + ANN

**Concept**: Train a neural network to map font character images to a compact embedding space, then use ANN for retrieval.

**Pipeline**:
1. **Offline**: For each font × character, render image → CNN → D-dimensional embedding → store in ANN index
2. **Online**: For each scanned character, extract image → CNN → embedding → ANN query → top-K fonts

**Key design choices**:
- **Embedding dimension**: 64-256 is typical. Larger = more discriminative but slower search
- **Training loss**: Triplet loss, contrastive loss, or ArcFace
- **CNN architecture**: Even MobileNet-scale CNNs suffice for this task
- **ANN structure**: Flat index (brute force) is fine for 400K vectors; IVF or HNSW for larger

**Relevance to unprint**: ⭐⭐⭐⭐ (High, but requires ML infrastructure)
- This is the "modern" approach and likely the most accurate pre-filter possible
- Trade-off: requires training a CNN (one-time cost) and running inference per character (the hot path bottleneck)
- Could potentially be done in Rust via ONNX Runtime, but adds a dependency

---

## 5. Evaluation Benchmarks & Datasets

### AdobeVFR Dataset
- Created for the DeepFont paper
- **2,383 font classes** for classification
- Contains both synthetic rendered data and partially labeled real-world images
- Real-world images collected from the web with font labels
- The standard benchmark for visual font recognition

### Font Recognition Benchmark (FRB, 2025)
- 15-way classification benchmark for VLMs
- Tests font style recognition (not identification)
- Best results: Gemini-3-Flash at 40.5%, Claude-Sonnet at 22.9%
- Confirms that current VLMs are not useful for font recognition

### Synthetic Datasets
- Trivially generated by rendering text in known fonts with augmentation
- Common augmentations: Gaussian blur, salt-and-pepper noise, JPEG compression, rotation, perspective warp
- Most font recognition papers use synthetic training data since labeled real-world font data is extremely scarce

### Accuracy Summary Across Methods

| Method | Dataset/Setting | Top-1 Acc | Top-5 Acc | # Classes |
|--------|----------------|-----------|-----------|-----------|
| DeepFont | AdobeVFR (real-world) | ~53% | **>80%** | 2,383 |
| ResNet18 (Cselle) | Synthetic test | **95%+** | N/A | ~40 |
| CNN (Tensmeyer) | Arabic ICDAR | **98.8%** (line) | N/A | ~10 |
| Font-ProtoNet (5-shot) | AdobeVFR | N/A | **71%** | 100 novel |
| EigenFonts | Synthetic+degraded | **97%** | N/A | 3 |
| VLMs (best) | FRB | **40.5%** | N/A | 15 styles |

**Key takeaway**: Accuracy is highly dependent on the number of classes and input quality. On clean synthetic data with limited classes, even simple methods achieve >95%. On thousands of real-world classes, even DeepFont struggles for top-1 accuracy. **Top-5 accuracy is much more relevant for our pre-filter use case** since we only need the correct font to be in the candidate set.

---

## 6. Comparison with Our Approach

### What We're Doing Right

1. **Per-character matching**: This is the right granularity for document font identification. Most successful approaches (DeepFont, Matcherator, Font-ProtoNet) work at character or word level.

2. **Two-stage pipeline (CI → SSIM verify)**: The CI identifies the font; SSIM verification gates bad matches. Clean separation of concerns.

3. **Parallel SSIM fast path**: The dominant-font acceleration via SSIM (threshold 0.90) avoids CI entirely for the common case of single-font documents.

4. **Rich 99-dimensional feature set**: Five weighted groups with per-group L2 normalization and Fisher-tuned weights capture typographic properties (serif score, stroke contrast, counter shapes, skeleton topology) as well as pixel-level features (column/row profiles, ink density).

5. **Controlled rendering conditions**: We render reference glyphs under known conditions, avoiding the domain gap problem that plagues photo-based font recognition.

6. **Font-metric word splitting**: Using `outline_glyph().px_bounds()` for derived rendering scale and predicted inter-glyph gaps produces more accurate word boundaries than Tesseract alone.

### What We're Missing

1. **No hierarchical pre-filtering**: Every font is scored equally in the CI. A serif/sans-serif classifier alone would halve the work.

2. **No learned features**: Hand-crafted features inevitably miss discriminative information that a CNN would capture. The 94.6% accuracy ceiling may require learned embeddings to break through.

3. **No multi-resolution robustness**: Current testing is at 300 DPI only. Real scans vary widely.

4. **Column/row profile limitations**: The 32-bin profiles miss horizontal structure, curve shapes, and terminal details at a fine level.

### Feature Vector Comparison

| Feature | Ours (99-dim, 5 groups) | Typographic (est. 8-15 dim) | CNN Embedding (128-512 dim) | PCA/EigenFont (50-100 dim) |
|---------|------------------------|---------------------------|---------------------------|---------------------------|
| Computation speed | ⚡ Very fast | ⚡ Fast | 🔶 Medium (inference) | ⚡ Fast |
| Discrimination | ⭐ Medium-high | 🔶 Medium-high | ⭐ High | ⭐ Medium-high |
| Robustness to noise | 🔶 Medium | ⭐ High (discrete features) | ⭐ High | 🔶 Medium |
| Implementation effort | ✅ Done | 🔶 Moderate | 🔴 High (training + ONNX) | 🔶 Moderate |
| Pre-computation | ⚡ Fast | ⚡ Fast | 🔶 Medium | 🔶 Medium |

---

## 7. Recommendations for unprint

Ranked by **expected improvement / implementation effort**:

### Tier 1: Quick Wins (Days of work, significant improvement)

#### 1. Add Typographic Feature Pre-Filter
**Effort**: 2-3 days | **Expected improvement**: 2-5× speedup, better accuracy

Compute and index these features per font (at index build time):
- **Serif detection**: Render 'I' at large size, check for horizontal protrusions at baseline and cap-line using morphological analysis. Binary feature.
- **Stroke contrast**: Render 'O', compute distance transform inside ink, measure ratio of max to min. Continuous 0-1.
- **x-height ratio**: Render 'x' and 'H', measure ratio of ink bounding box heights. Continuous 0-1.
- **Weight**: Average stroke width from distance transform of 'o' or 'n'. Continuous.

At query time, compute the same features from the scanned text and filter fonts that don't match within tolerances. This alone could eliminate 50-80% of candidates before the cosine similarity step.

#### 2. Multi-Character Score Aggregation
**Effort**: 1 day | **Expected improvement**: Noticeable accuracy improvement

Instead of scoring a single representative character:
- Score multiple characters from the same text line against each candidate font
- Use **median or trimmed mean** across characters (robust to OCR errors on individual chars)
- Weight by character discriminativeness (e.g., 'g', 'a', 'e' are more discriminative than 'l', 'i', 'o')
- This is essentially free since we already have per-character scores

#### 3. Font Size Consistency Check
**Effort**: 0.5 days | **Expected improvement**: Better false-positive rejection

Per the Xerox/Google OCR patent insight:
- For the top-N candidates, compute the font size needed to match each word's width
- The correct font should produce a **tight cluster of consistent sizes** across all words in a paragraph
- Penalize candidates with high variance in required size → wrong fonts need different sizes for different words

### Tier 2: Medium Effort (1-2 weeks, major improvement)

#### 4. PCA-Based Character Embeddings (EigenFonts)
**Effort**: 3-5 days | **Expected improvement**: Better discrimination than column profiles

- Render each character at standard size (e.g., 48×48) in each font → flatten to vector
- Compute PCA across all fonts for each character
- Keep top 50-100 principal components as the feature vector
- Replace or supplement our 32-bin column profile with PCA features
- Pure linear algebra, no ML framework needed, trivial in Rust (use `nalgebra` or similar)

#### 5. Perceptual Hash Pre-Filter Layer
**Effort**: 1-2 days | **Expected improvement**: Faster coarse elimination

- Compute pHash for each character × font at index time (64-bit hash)
- At query time: pHash the scanned character, find fonts with Hamming distance < threshold
- Ultra-fast (XOR + popcount), eliminates obviously wrong fonts
- Stack before the cosine similarity step: pHash (5048→500) → cosine (500→50) → SSIM (50→1)

#### 6. Hierarchical Font Tree
**Effort**: 3-5 days | **Expected improvement**: Major speedup

Build an offline classification tree:
```
Root (5048 fonts)
├── Serif (2100 fonts)
│   ├── Old-style (400)
│   ├── Transitional (500)
│   ├── Modern/Didone (300)
│   ├── Slab (400)
│   └── Other (500)
├── Sans-serif (2200 fonts)
│   ├── Grotesque (600)
│   ├── Neo-grotesque (500)
│   ├── Geometric (400)
│   ├── Humanist (400)
│   └── Other (300)
├── Monospace (300 fonts)
├── Script/Handwriting (300 fonts)
└── Display/Decorative (148 fonts)
```

At query time, classify the scanned text into a branch and only search within that branch. Getting the top-level right (serif vs sans) is very reliable and immediately halves the search.

### Tier 3: High Effort, Highest Ceiling (Weeks, near-optimal)

#### 7. CNN Embedding with Triplet Loss
**Effort**: 2-3 weeks | **Expected improvement**: Likely the best possible pre-filter

- Train a small CNN (ResNet18 or MobileNetV2) with triplet/contrastive loss
- Training data: render each character in each font at multiple sizes with scan-like augmentation (blur, noise, binarization artifacts)
- Produces a 128-dim embedding per character image
- Pre-compute embeddings for all fonts; at query time, embed the scanned char and find nearest neighbors
- Export via ONNX; run inference in Rust via `ort` (ONNX Runtime crate)

**Why triplet loss over classification**:
- Classification requires retraining when fonts are added
- Triplet loss produces a general similarity metric that works for new fonts
- The embedding space naturally handles "this font isn't in our database, but here are the closest ones"

#### 8. Hybrid Feature Vector
**Effort**: 1-2 weeks (after implementing Tier 1-2 items)

Concatenate all useful features into a single vector:
```
[32 column_profile | 4 typographic | 50 PCA | 64 pHash_bits] = ~150 dim
```
Or, if CNN is available:
```
[128 CNN_embedding | 4 typographic | 32 column_profile] = ~164 dim
```
Use learned weights (simple logistic regression or MLP) to combine these into a single score, trained on synthetic matched/unmatched pairs.

---

### Priority Roadmap

| Phase | Items | Effort | Expected Impact |
|-------|-------|--------|-----------------|
| **Phase 1** | #1 (typographic features) + #2 (multi-char aggregation) + #3 (size consistency) | ~4 days | 2-5× faster, noticeably more accurate |
| **Phase 2** | #4 (PCA embeddings) + #6 (hierarchical tree) | ~1 week | Another 2-3× faster, more accurate |
| **Phase 3** | #7 (CNN embeddings) | ~2-3 weeks | Near-optimal pre-filter quality |
| **Phase 4** | #8 (hybrid vector) | ~1 week | Polish, marginal gains |

**Phase 1 is the clear priority** — the items are simple, don't require ML infrastructure, and address the most obvious gaps in our current approach.

---

## References

1. Wang et al., "DeepFont: Identify Your Font from An Image," ACM MM 2015. https://arxiv.org/abs/1507.03196
2. Tensmeyer et al., "CNNs for Font Classification," ICDAR 2017. https://arxiv.org/abs/1708.03669
3. Goel et al., "Font-ProtoNet," CVPR Workshop 2020. https://openaccess.thecvf.com/content_CVPRW_2020/html/w34/Goel_Font-ProtoNet...
4. Cho et al., "Font Representation Learning via Paired-glyph Matching," 2022. https://arxiv.org/abs/2211.10967
5. Zramdini & Ingold, "Optical Font Recognition Using Typographical Features," IEEE TPAMI 1998.
6. Zhu, Tan & Wang, "Font Recognition Based on Global Texture Analysis," IEEE TPAMI 2001.
7. Bozkurt et al., "Classifying Fonts Using Complex Wavelet Transform," 2014. https://arxiv.org/abs/1407.2649
8. Al-Khaffaf & Musa, "Optical English Font Recognition Using Eigenfaces." https://doi.org/10.15649/2346075X.466
9. Adobe Patent US10127199B2: "Automatic Measure of Visual Similarity Between Fonts"
10. Adobe Patent US10515295B2: "Font Recognition Using Triplet Loss Neural Network Training"
11. Xerox/Google Patent US6741745B2: "Method and Apparatus for Formatting OCR Text"
12. Cselle, "Font Identifier (ResNet18)," 2023. https://github.com/gaborcselle/font-identifier
13. IDEO Font Map, 2017. VGG16 + t-SNE for Google Fonts similarity. https://www.designboom.com/design/ideo-font-map...
14. FontCLIP, 2024. https://arxiv.org/abs/2403.06453
15. "VLMs Get Lost in Font Recognition," 2025. https://arxiv.org/abs/2503.23768
16. WhatTheFont (Monotype). Trained on 33M images, 130K+ fonts, ~90% accuracy. https://www.myfonts.com/pages/whatthefont
17. Identifont. Question-based font identification via typographic feature taxonomy. https://en.wikipedia.org/wiki/Identifont
18. "Contrastive Learning for Multilingual Font Style Classification," 2023. https://www.mdpi.com/2076-3417/13/6/3635
