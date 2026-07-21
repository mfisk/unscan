use std::path::Path;
use unprint::ground_truth::GroundTruth;

fn test_query(spans: &Vec<unprint::ground_truth::VectorSpan>, bbox_px: [f32;4], label: &str) {
    let scale = 300.0/72.0;
    let px0=bbox_px[0]/scale; let py0=bbox_px[1]/scale;
    let px1=bbox_px[2]/scale; let py1=bbox_px[3]/scale;
    println!("\nQuery {} bbox_px {:?} -> pt [{:.2},{:.2},{:.2},{:.2}]", label, bbox_px, px0,py0,px1,py1);
    let mut cands=Vec::new();
    for s in spans {
        let ox0=s.bbox[0].max(px0); let oy0=s.bbox[1].max(py0);
        let ox1=s.bbox[2].min(px1); let oy1=s.bbox[3].min(py1);
        if ox0<ox1 && oy0<oy1 {
            let area=(ox1-ox0)*(oy1-oy0);
            cands.push((area, s.font_name.clone(), s.bbox, s.text.clone()));
        }
    }
    cands.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
    println!("  Found {} overlapping", cands.len());
    for (area,font,bbox,text) in cands.iter().take(5) {
        println!("  area {:.1} font={} bbox=[{:.1},{:.1},{:.1},{:.1}] text={:?}", area,font,bbox[0],bbox[1],bbox[2],bbox[3],&text[..text.len().min(80)]);
    }
    if cands.is_empty() {
        for s in spans { if (s.bbox[1]-py0).abs()<25.0 { println!(" near y: {} {:?} [{:.1},{:.1},{:.1},{:.1}]", s.font_name, &s.text[..s.text.len().min(40)], s.bbox[0],s.bbox[1],s.bbox[2],s.bbox[3]); } }
    }
}

fn main() {
    let pdf_path = std::env::args().nth(1).unwrap_or_else(|| "/home/hatch/workspace/repos/unscan/test-docs/font-timeline-specimen.pdf".to_string());
    let gt = GroundTruth::load(Path::new(&pdf_path)).expect("load gt");
    for (page,spans) in &gt.pages {
        if *page==4 {
            println!("Page {} has {} spans", page, spans.len());
            for (i,s) in spans.iter().enumerate() {
                if s.bbox[1]>=310.0 && s.bbox[1]<=340.0 {
                    println!("p4 span {}: font={} bbox=[{:.1},{:.1},{:.1},{:.1}] w={:.1} text={:?}", i, s.font_name, s.bbox[0],s.bbox[1],s.bbox[2],s.bbox[3], s.bbox[2]-s.bbox[0], &s.text[..s.text.len().min(80)]);
                }
            }
        }
        if *page==3 {
            println!("\nPage {} has {} spans (y 200-280)", page, spans.len());
            for (i,s) in spans.iter().enumerate() {
                if s.bbox[1]>=200.0 && s.bbox[1]<=280.0 {
                    println!("p3 span {}: font={} bbox=[{:.1},{:.1},{:.1},{:.1}] w={:.1} text={:?}", i, s.font_name, s.bbox[0],s.bbox[1],s.bbox[2],s.bbox[3], s.bbox[2]-s.bbox[0], &s.text[..s.text.len().min(80)]);
                }
            }
        }
    }
    if let Some(spans)=gt.pages.get(&4) {
        test_query(spans, [1321.0,1324.0,1321.0+977.0,1324.0+39.0], "p4 L71");
    }
    if let Some(spans)=gt.pages.get(&3) {
        test_query(spans, [1321.0,915.0,1321.0+486.0,915.0+38.0], "p3 L65");
    }
}
