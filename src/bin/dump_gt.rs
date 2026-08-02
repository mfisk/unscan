use std::path::Path;
use unprint::ground_truth::GroundTruth;
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let pdf_path = if args.len() > 1 { &args[1] } else { "test-docs/font-timeline-specimen.pdf" };
    let page_filter: Option<usize> = if args.len() > 2 { args[2].parse().ok() } else { None };
    let gt = GroundTruth::load(Path::new(pdf_path))?;
    eprintln!("loaded {} pages from {}", gt.pages.len(), pdf_path);
    for (page_num, spans) in &gt.pages {
        if let Some(pf) = page_filter { if *page_num != pf { continue; } }
        println!("Page {}: {} spans", page_num, spans.len());
        for s in spans {
            // canonical name already resolved via /UnprintCanonical or catalog
            println!(" y {:.1} x {:.1}-{:.1} w {:.1} font {} text {:?}", s.bbox[1], s.bbox[0], s.bbox[2], s.bbox[2]-s.bbox[0], s.font_name, &s.text.chars().take(60).collect::<String>());
        }
    }
    Ok(())
}
