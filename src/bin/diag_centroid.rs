use unprint::classifier::PerCharModel;
use unprint::char_render::{self, RenderParams};
use unprint::features::{compute_features, FEAT_LEN};
use ab_glyph::FontVec;

fn main() {
    let home = std::env::var("HOME").unwrap();
    let path = format!("{}/.cache/unprint/lda-weights.bin", home);
    let data = std::fs::read(&path).expect("read model");
    let model = PerCharModel::read_bin(&data, b"LDAC", None).expect("parse model");

    let pf_path = "/home/hatch/.local/share/fonts/ofl/playfairdisplay/PlayfairDisplay[wght].ttf";
    let font_data = std::fs::read(pf_path).expect("read font");
    let font = FontVec::try_from_vec(font_data).expect("parse font");
    let params = RenderParams::default();

    let ch = 'e';
    let img = char_render::get_rendered_char(&font, pf_path, ch, None, &params).expect("render");
    let feats = compute_features(&img, false).expect("features");
    let raw = feats.as_slice();

    let cm = model.chars.get(&ch).expect("char model");
    let out_dim = cm.weights[0] as usize;
    let proj = &cm.weights[1..];

    // Project: same as dense_project
    let mut query_emb = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..FEAT_LEN {
            sum += proj[i * FEAT_LEN + j] * raw[j];
        }
        query_emb[i] = sum;
    }

    // Find PlayfairDisplay (fid=65) centroid
    let pf = cm.centroids.iter().find(|(fid, _)| *fid == 65);
    if let Some((_, centroid)) = pf {
        let mut dist = 0.0f32;
        for i in 0..out_dim {
            let d = query_emb[i] - centroid[i];
            dist += d * d;
        }
        eprintln!("=== char 'e' PlayfairDisplay ===");
        eprintln!("out_dim={}, FEAT_LEN={}", out_dim, FEAT_LEN);
        eprintln!("proj len={}, expected={}", proj.len(), out_dim * FEAT_LEN);
        eprintln!("sigma_sq = {:.6}", cm.sigma_sq);
        eprintln!("query→centroid sq_dist = {:.6}", dist);
        eprintln!("query[0..8]    = {:?}", &query_emb[..8]);
        eprintln!("centroid[0..8] = {:?}", &centroid[..8]);
        eprintln!("raw[0..8]      = {:?}", &raw[..8]);

        // Find nearest centroid
        let mut min_d = f32::MAX;
        let mut min_fid = 0u32;
        for (fid, v) in &cm.centroids {
            let mut d = 0.0f32;
            for i in 0..out_dim {
                let dd = query_emb[i] - v[i];
                d += dd * dd;
            }
            if d < min_d {
                min_d = d;
                min_fid = *fid;
            }
        }
        let near_name = if (min_fid as usize) < model.font_names.len() {
            model.font_names[min_fid as usize].rsplit('/').next().unwrap_or("?")
        } else { "?" };
        eprintln!("nearest centroid: fid={} dist={:.6} = {}", min_fid, min_d, near_name);

        // Recompute centroid from raw features to verify
        // (project the mean features through LDA — same as training does)
        // We can only verify by rendering all 3 AA variants and averaging
        let aa_variants = [
            unprint::features::AaVariant::Native,
            unprint::features::AaVariant::Blur05,
            unprint::features::AaVariant::Sharpen,
        ];
        let mut mean_feats = vec![0.0f64; FEAT_LEN];
        let mut count = 0usize;
        for aa in &aa_variants {
            let mut p = params.clone();
            p.aa = *aa;
            if let Some(img) = char_render::get_rendered_char(&font, pf_path, ch, None, &p) {
                if let Some(f) = compute_features(&img, false) {
                    let s = f.as_slice();
                    for j in 0..FEAT_LEN { mean_feats[j] += s[j] as f64; }
                    count += 1;
                }
            }
        }
        if count > 0 {
            for j in 0..FEAT_LEN { mean_feats[j] /= count as f64; }
            // Project mean features
            let mut recomputed = vec![0.0f32; out_dim];
            for i in 0..out_dim {
                let mut sum = 0.0f32;
                for j in 0..FEAT_LEN {
                    sum += proj[i * FEAT_LEN + j] * mean_feats[j] as f32;
                }
                recomputed[i] = sum;
            }
            let mut rdist = 0.0f32;
            for i in 0..out_dim {
                let d = recomputed[i] - centroid[i];
                rdist += d * d;
            }
            eprintln!("\nRecomputed centroid from {} AA variants:", count);
            eprintln!("  recomputed[0..8] = {:?}", &recomputed[..8]);
            eprintln!("  dist to stored centroid = {:.6}", rdist);
        }
    }
}
