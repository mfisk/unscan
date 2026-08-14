//! Font match result type and font identification
//!
//! Per-font ligature handling: we see what lig glyphs each font supports,
//! get (and save for other fonts) segmentation for that font’s number of
//! glyphs, and use that segmentation and those glyphs for any downstream
//! scoring.  Nothing in font scoring uses the original OCR characters or
//! word length — only the collapsed chars appropriate for a given font.
//! `WordSegs` cache is per-k (fresh map for each line/word), `WordSeg` is
//! per-font view on top of per-k cuts.

use rustc_hash::{FxHashMap, FxHashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use image::GrayImage;
use crate::features::compute_features;
use crate::classifier::{self, ObsStats};
use crate::segment::{WordSeg, SegSummary, collapse_ligature_chars_for_allowed, segment_characters, char_crop_and_metrics};
use crate::geometry_classifier::{CharInkBounds, WordGeoMeasurement};

const GEO_WEIGHT: f32 = 1.0;
const MIDPOINT_PRUNE_BASE: f32 = -2.0; // tighter still - avg logit typical -1 to -1.5, -3 was still loose

#[derive(Debug, Clone)]
pub struct FontMatchResult {
    pub font_name: String,
    pub font_path: PathBuf,
    pub font_key: String,
    pub variant_tag: String,
    pub glyph_overrides: crate::font_scan::GlyphOverrides,
    pub variations: crate::font_scan::Variations,
    pub score: f32,
    pub best_dy: i32,
}

pub struct ScoringWindow<'a> {
    pub ch: char,
    pub crop: &'a GrayImage,
    pub weight: f32,
}

pub struct WordImgInfo {
    pub img: Arc<GrayImage>,
    pub orig_chars: Vec<char>,
    pub source_idx: usize,
    pub text: String,
}

impl WordImgInfo {
    pub fn new(img: Arc<GrayImage>, orig_chars: Vec<char>, source_idx: usize, text: String) -> Self {
        Self { img, orig_chars, source_idx, text }
    }
}

#[derive(Debug, Clone)]
pub struct ObservationDetail {
    pub ch: char,
    pub weight: f32,
    pub crop_index: usize,
    pub best_prob: f32,
    pub passed_gate: bool,
    pub nearest: Vec<(usize, f32)>,
    pub ocr_corrected_from: Option<char>,
    pub best_alt_char: Option<char>,
    pub best_alt_dist: Option<f32>,
    pub pflda_top_char: Option<char>,
    pub pflda_top_p: Option<f32>,
    pub pflda_ocr_p: Option<f32>,
    pub pflda_replaced: bool,
    pub obs_stats: Option<ObsStats>,
    // mainline scoring temps – pulled directly from identify_fonts (no recompute)
    pub glyph_score: Option<f32>,   // raw logit for this font/glyph
    pub prob: Option<f32>,          // softmax prob for this font/glyph
    pub geo_h_ll: Option<f32>,
    pub geo_v_ll: Option<f32>,
    pub geo_h_err: Option<f32>,
    pub geo_v_err: Option<f32>,
}

#[allow(dead_code)]
const MIN_NGRAM_PROB: f32 = 0.001;

#[derive(Debug)]
pub struct FontIdResult {
    pub scores: Vec<(String, f32)>,
    pub observations: Vec<ObservationDetail>,      // winning font
    pub gt_observations: Vec<ObservationDetail>,   // GT font (from ensure list) if present
    pub path_score: f32,
}

struct SegCacheEntry {
    bounds: Vec<u32>,
    seams: HashMap<u32, Vec<[u32; 2]>>,
    summary: SegSummary,
    // per-char crop+metrics for this k, length = k – Arc to make per-font clone cheap
    char_crops: Vec<Option<(Arc<GrayImage>, u32, u32, u32, u32, f64, f64)>>,
}

pub fn identify_fonts(
    word_infos: &[WordImgInfo],
    classifier: &dyn crate::classifier::Classifier,
    glyph_map: &crate::glyph_map::NgramGlyphMap,
    thoroughness: f32,
    audit: bool,
    ensure_font_keys: &[&str],
    min_ngram_prob: f32,
    font_registry: &crate::font_scan::FontRegistry,
    font_cache: &crate::font_cache::FontCache,
    geo_cache: &crate::geo_cache::GeometryCache,
) -> FontIdResult {
    if word_infos.is_empty() {
        return FontIdResult { scores: Vec::new(), observations: Vec::new(), gt_observations: Vec::new(), path_score: f32::MIN };
    }

    let mut seg_caches: Vec<HashMap<usize, SegCacheEntry>> =
        (0..word_infos.len()).map(|_| HashMap::new()).collect();

    let mut ensure_set: FxHashSet<&str> = FxHashSet::default();
    for &fk in ensure_font_keys { ensure_set.insert(fk); }
    let allow_opt = crate::cache::font_allowlist();

    let prune_threshold = MIDPOINT_PRUNE_BASE * thoroughness.max(0.1);

    let mut scored: Vec<(String, f32)> = Vec::new();
    let mut best_observations: Vec<ObservationDetail> = Vec::new();
    let mut gt_observations: Vec<ObservationDetail> = Vec::new();
    let mut gt_best_score = f32::MIN;
    let mut best_path_score = f32::MIN;
    let mut best_score = f32::MIN;
    let mut pruned_count: usize = 0;

    for fe in font_registry.iter() {
        let fk: &str = fe.font_key_ref();
        if let Some(ref allow) = allow_opt {
            if !allow.contains(fk) && !ensure_set.contains(fk) { continue; }
        }
        let is_ensure = ensure_set.contains(fk);
        let allowed = fe.collapsed_lig_set();

        // Build per-font structures using cached cuts + char crops (per-k cache)
        let mut word_segs: Vec<WordSeg> = Vec::with_capacity(word_infos.len());
        struct WordBuild {
            collapsed: Vec<char>,
            bounds: Vec<u32>,
            seams: Arc<HashMap<u32, Vec<[u32;2]>>>,
            summary: SegSummary,
            char_crops: Vec<Option<(Arc<GrayImage>, u32,u32,u32,u32,f64,f64)>>, // Arc makes per-font clone cheap
            crop_h: u32,
            word_img: Arc<GrayImage>,
            word_text: String,
            source_idx: usize,
        }
        let mut builds: Vec<WordBuild> = Vec::with_capacity(word_infos.len());

        let mut feasible = true;
        for (wi_idx, wi) in word_infos.iter().enumerate() {
            let collapsed = collapse_ligature_chars_for_allowed(&wi.orig_chars, &allowed);
            let k = collapsed.len();
            if k == 0 { continue; }
            // get or create cache entry for this word k
            let cache = &mut seg_caches[wi_idx];
            if !cache.contains_key(&k) {
                let (b, s, sum) = segment_characters(&wi.img, k);
                let crop_h = wi.img.height();
                let mut char_crops: Vec<Option<(Arc<GrayImage>,u32,u32,u32,u32,f64,f64)>> = Vec::with_capacity(k);
                for i in 0..k {
                    let cc = char_crop_and_metrics(&wi.img, i, &b, &s, crop_h);
                    let cc_arc = cc.map(|(img, x1,x2,y1,y2,cx,cy)| (Arc::new(img), x1,x2,y1,y2,cx,cy));
                    char_crops.push(cc_arc);
                }
                cache.insert(k, SegCacheEntry { bounds: b, seams: s, summary: sum, char_crops });
            }
            let ent = cache.get(&k).unwrap();
            // Clone needed data for WordBuild (bounds clone cheap, seams Arc clone cheap)
            builds.push(WordBuild {
                collapsed,
                bounds: ent.bounds.clone(),
                seams: Arc::new(ent.seams.clone()),
                summary: ent.summary.clone(),
                char_crops: ent.char_crops.clone(),
                crop_h: wi.img.height(),
                word_img: wi.img.clone(),
                word_text: wi.text.clone(),
                source_idx: wi.source_idx,
            });
        }

        if builds.is_empty() {
            if !is_ensure { continue; }
            feasible = false;
        }
        if !feasible && builds.is_empty() {
            if !is_ensure { continue; }
        }

        // Build WordSegs + wib + scoring temps
        let mut wib: Vec<WordGeoMeasurement> = Vec::with_capacity(builds.len());
        struct Temp {
            ch: char,
            weight: f32,
            area_px: f32,
            ood: f32,
            best_prob: f32,
            nearest: Vec<(usize,f32)>,
            logit: Option<f32>,
            prob: Option<f32>,
            geo_h_ll: f32,
            geo_v_ll: f32,
            geo_h_err: f32,
            geo_v_err: f32,
        }
        // We'll collect per-word wib and temps
        let mut temps_collect: Vec<Temp> = Vec::new();

        for (_seg_idx, wb) in builds.iter().enumerate() {
            let mut char_bounds: Vec<CharInkBounds> = Vec::with_capacity(wb.collapsed.len());
            for (pos, _) in wb.collapsed.iter().enumerate() {
                if pos >= wb.char_crops.len() {
                    // fallback
                    let (b_l, b_r) = if pos+1 < wb.bounds.len() { (wb.bounds[pos], wb.bounds[pos+1]) } else { (0,wb.crop_h) };
                    let cx = (b_l as f64 + b_r as f64)*0.5;
                    let cy = wb.crop_h as f64 *0.5;
                    let cb = CharInkBounds { cx, cy, width: b_r.saturating_sub(b_l) as f64, height: wb.crop_h as f64, x_min: b_l, x_max: b_r, y_min: 0, y_max: wb.crop_h, frac_left: b_l as f64, frac_right: b_r as f64 };
                    char_bounds.push(cb);
                    continue;
                }
                if let Some((ref _norm, x_min, x_max, y_min, y_max, cx, cy)) = wb.char_crops[pos] {
                    let cb = CharInkBounds { cx, cy, width: (x_max - x_min +1) as f64, height: (y_max - y_min +1) as f64, x_min, x_max, y_min, y_max, frac_left: x_min as f64, frac_right: x_max as f64 +1.0 };
                    char_bounds.push(cb);
                } else {
                    let (b_l, b_r) = if pos+1 < wb.bounds.len() { (wb.bounds[pos], wb.bounds[pos+1]) } else { (0,wb.crop_h) };
                    let cx = (b_l as f64 + b_r as f64)*0.5;
                    let cy = wb.crop_h as f64 *0.5;
                    let cb = CharInkBounds { cx, cy, width: b_r.saturating_sub(b_l) as f64, height: wb.crop_h as f64, x_min: b_l, x_max: b_r, y_min:0, y_max: wb.crop_h, frac_left: b_l as f64, frac_right: b_r as f64 };
                    char_bounds.push(cb);
                }
            }
            wib.push(WordGeoMeasurement { chars: char_bounds });

            // For scoring, we will handle after wib built? Need temps now but wib already has cx
            // Actually temps need only crops, not wib
            for &c in wb.collapsed.iter() {
                if !crate::features::is_supported(c) { continue; }
                // Count only if crop exists – will filter later
                // (pos check done in scoring_temps)
                temps_collect.push(Temp {
                    ch: c,
                    weight: 1.0,
                    area_px: 0.0,
                    ood: 1.0,
                    best_prob: 0.0,
                    nearest: Vec::new(),
                    logit: None,
                    prob: None,
                    geo_h_ll: 0.0,
                    geo_v_ll: 0.0,
                    geo_h_err: 0.0,
                    geo_v_err: 0.0,
                });
            }
            // Build WordSeg for geo
            let seg = WordSeg {
                source_word_idx: wb.source_idx,
                word_img: wb.word_img.clone(),
                chars: wb.collapsed.clone(),
                boundaries: wb.bounds.clone(),
                seam_paths: wb.seams.clone(),
                seam_costs: Arc::new(wb.summary.seam_costs.clone()),
                crop_h: wb.crop_h,
                word_text: wb.word_text.clone(),
                image_w: wb.summary.image_w,
                image_h: wb.summary.image_h,
                n_chars_expected: wb.summary.n_chars_expected,
                n_segments_produced: wb.summary.n_segments_produced,
                mismatch: wb.summary.mismatch,
                ws_splits: wb.summary.ws_splits.clone(),
                seam_splits: wb.summary.seam_splits.clone(),
            };
            word_segs.push(seg);
        }

        if word_segs.is_empty() {
            if !is_ensure { continue; }
        }

        // Geometry LLs – keep h/v separate for per-char table
        let geo_opt = crate::geometry_classifier::per_char_geo_for_font(
            fk, &word_segs, &wib, font_cache, geo_cache, font_registry
        );
        // maps: (seg_idx, pos) -> ll / err
        let mut geo_h_map: FxHashMap<(usize,usize), f32> = FxHashMap::default();
        let mut geo_v_map: FxHashMap<(usize,usize), f32> = FxHashMap::default();
        let mut geo_h_err_map: FxHashMap<(usize,usize), f32> = FxHashMap::default();
        let mut geo_v_err_map: FxHashMap<(usize,usize), f32> = FxHashMap::default();
        match geo_opt {
            None => {
                if !is_ensure { pruned_count+=1; continue; }
            }
            Some(ref geos) if geos.is_empty() => {},
            Some(geos) => {
                let mut min_ll = f32::INFINITY;
                for g in &geos {
                    let ll = (g.h_ll + g.v_ll) as f32;
                    if ll < min_ll { min_ll = ll; }
                    geo_h_map.insert((g.seg_idx, g.orig_idx), g.h_ll as f32);
                    geo_v_map.insert((g.seg_idx, g.orig_idx), g.v_ll as f32);
                    geo_h_err_map.insert((g.seg_idx, g.orig_idx), g.h_err.unwrap_or(0.0) as f32);
                    geo_v_err_map.insert((g.seg_idx, g.orig_idx), g.v_err as f32);
                }
                if !is_ensure && min_ll < prune_threshold { pruned_count+=1; continue; }
            }
        };

        // Now compute features and logits for scoring temps
        // Need to map temps to actual crops: we have to retrieve norm images from builds
        // Build a flat list of crops corresponding to temps order is same as temps_collect order
        // We have temps_collect built in order of word_segs traversal, same as we need
        // But we still need features: compute_features on each cached norm image

        // Rebuild temps with features: we need to iterate again over builds to get norm images and match temps order

        let mut scoring_temps: Vec<Temp> = Vec::with_capacity(temps_collect.len());
        for (seg_idx, wb) in builds.iter().enumerate() {
            for (pos, &c) in wb.collapsed.iter().enumerate() {
                if !crate::features::is_supported(c) { continue; }
                if pos >= wb.char_crops.len() { continue; }
                let crop_opt = &wb.char_crops[pos];
                if crop_opt.is_none() { continue; }
                let (ref norm, _x_min,_x_max,_y_min,_y_max,_cx,_cy) = crop_opt.as_ref().unwrap();
                let area_px = ((_x_max - _x_min + 1) * (_y_max - _y_min + 1)) as f32;
                // Feature
                let feat = match compute_features(norm, false) {
                    Some(f) => f,
                    None => { continue; }
                };
                // Temp placeholder – classify below; pull geo from per-font maps
                let h_ll = geo_h_map.get(&(seg_idx, pos)).copied().unwrap_or(0.0);
                let v_ll = geo_v_map.get(&(seg_idx, pos)).copied().unwrap_or(0.0);
                let h_err = geo_h_err_map.get(&(seg_idx, pos)).copied().unwrap_or(0.0);
                let v_err = geo_v_err_map.get(&(seg_idx, pos)).copied().unwrap_or(0.0);
                let mut t = Temp {
                    ch: c,
                    weight: 1.0,
                    area_px,
                    ood: 1.0,
                    best_prob: 0.0,
                    nearest: Vec::new(),
                    logit: None,
                    prob: None,
                    geo_h_ll: h_ll,
                    geo_v_ll: v_ll,
                    geo_h_err: h_err,
                    geo_v_err: v_err,
                };
                // Classify for ood + best_prob + nearest
                let seq = [c];
                let picks = classifier.classify(&seq, &feat, 3);
                let ood = classifier::take_ood_weight();
                t.ood = ood;
                let best_prob = picks.iter().map(|(_,p)| *p).fold(0.0f32, f32::max);
                t.best_prob = best_prob;
                t.nearest = picks.iter().take(3).map(|&(id,p)| (id,p)).collect();
                // Logit for this font's glyph
                if let Some(gid) = glyph_map.glyph_id_for_font(&seq, fk) {
                    let logits = classifier.raw_logits(&seq, &feat);
                    if logits.is_empty() {
                        // cannot score
                    } else {
                        let mut max_logit = f32::NEG_INFINITY;
                        for (_, l) in &logits { if *l > max_logit { max_logit = *l; } }
                        let exps: Vec<f32> = logits.iter().map(|(_, l)| (*l - max_logit).exp()).collect();
                        let sum_exp: f32 = exps.iter().sum();
                        let mut logit_for_gid: Option<f32> = None;
                        let mut prob_for_gid: Option<f32> = None;
                        for (i, (gid2, logit)) in logits.iter().enumerate() {
                            if *gid2 == gid {
                                logit_for_gid = Some(*logit);
                                if sum_exp < 1e-30 {
                                    prob_for_gid = Some(1.0 / exps.len() as f32);
                                } else {
                                    prob_for_gid = Some(exps[i] / sum_exp);
                                }
                                break;
                            }
                        }
                        t.logit = logit_for_gid;
                        t.prob = prob_for_gid;
                    }
                }
                scoring_temps.push(t);
            }
        }

        let n_windows = scoring_temps.len();

        // Median ink-bbox pixel area across this line's scored crops; small
        // punctuation upscaled to the canonical cell gets downweighted by
        // area/median so its (noisy) glyph evidence counts less.
        let mut areas: Vec<f32> = scoring_temps.iter().map(|t| t.area_px).collect();
        areas.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let median_area = if areas.is_empty() { 1.0 } else { areas[areas.len()/2].max(1.0) };
        for t in scoring_temps.iter_mut() {
            t.weight = t.area_px / median_area;
        }
        let min_coverage = ((n_windows as f32 * 0.4).ceil() as usize).max(3).min(n_windows.max(1));

        let mut log_probs: Vec<(f32,f32)> = Vec::with_capacity(n_windows);
        let mut ood_probs: Vec<(f32,f32)> = Vec::with_capacity(n_windows);
        let mut obs_for_this: Vec<ObservationDetail> = Vec::with_capacity(n_windows);
        let mut skip_font = false;

        for (ti, t) in scoring_temps.iter().enumerate() {
            let logit = match t.logit {
                Some(v) => v,
                None => { if !is_ensure { skip_font=true; break; } else { continue; } }
            };
            let prob = match t.prob {
                Some(v) => v,
                None => { if !is_ensure { skip_font=true; break; } else { continue; } }
            };
            let seq = [t.ch];
            let n_glyphs = classifier.glyph_count(&seq).max(1) as f32;
            let thresh = min_ngram_prob / n_glyphs;
            if prob < thresh { continue; }
            let combined_geo = t.geo_h_ll + t.geo_v_ll;
            let lp = logit * t.weight + GEO_WEIGHT * combined_geo;
            log_probs.push((lp, 1.0));
            ood_probs.push((lp, 1.0));
            obs_for_this.push(ObservationDetail {
                ch: t.ch,
                weight: t.weight,
                crop_index: ti,
                best_prob: t.best_prob,
                passed_gate: true,
                nearest: t.nearest.clone(),
                ocr_corrected_from: None,
                best_alt_char: None,
                best_alt_dist: None,
                pflda_top_char: None,
                pflda_top_p: None,
                pflda_ocr_p: None,
                pflda_replaced: false,
                obs_stats: classifier::take_obs_stats(),
                glyph_score: Some(logit),
                prob: Some(prob),
                geo_h_ll: Some(t.geo_h_ll),
                geo_v_ll: Some(t.geo_v_ll),
                geo_h_err: Some(t.geo_h_err),
                geo_v_err: Some(t.geo_v_err),
            });
        }

        if skip_font { continue; }
        if log_probs.len() < min_coverage && !is_ensure { continue; }
        if log_probs.is_empty() && !is_ensure { continue; }

        let score = if log_probs.is_empty() { f32::MIN } else { log_probs.iter().map(|(lp,w)| lp * *w).sum() };
        let path_score = if ood_probs.is_empty() { f32::MIN } else { ood_probs.iter().map(|(lp,w)| lp * *w).sum() };

        if score > best_score {
            best_score = score;
            best_path_score = path_score;
            best_observations = obs_for_this.clone();
        } else if (score - best_score).abs() < 1e-6 && path_score > best_path_score {
            best_path_score = path_score;
            best_observations = obs_for_this.clone();
        }

        if is_ensure {
            if score > gt_best_score {
                gt_best_score = score;
                gt_observations = obs_for_this.clone();
            } else if gt_observations.is_empty() && !obs_for_this.is_empty() {
                // keep first ensure's obs if all MIN scores
                gt_observations = obs_for_this.clone();
            }
        }

        scored.push((fk.to_owned(), score));
    }

    if pruned_count > 0 && audit {
        eprintln!("midpoint prune: pruned {}/{} fonts at threshold {:.2} (base {:.1} * thoroughness {:.2})", pruned_count, font_registry.iter().count(), prune_threshold, MIDPOINT_PRUNE_BASE, thoroughness);
    }

    let mut present: std::collections::HashSet<String> = scored.iter().map(|(k,_)| k.clone()).collect();
    for &efk in ensure_font_keys {
        if !present.contains(efk) {
            if let Some(_fe) = font_registry.by_key(efk) {
                scored.push((efk.to_owned(), f32::MIN));
                present.insert(efk.to_owned());
            } else {
                scored.push((efk.to_owned(), f32::MIN));
            }
        }
    }

    if scored.is_empty() {
        return FontIdResult { scores: Vec::new(), observations: best_observations, gt_observations, path_score: f32::MIN };
    }

    scored.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

    FontIdResult { scores: scored, observations: best_observations, gt_observations, path_score: best_path_score }
}
