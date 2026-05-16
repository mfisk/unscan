use std::path::{Path, PathBuf};
use std::fs;
use std::io::Write;
use serde::Serialize;

/// Collects diagnostic data during a pipeline run and renders an HTML report.
#[derive(Debug)]
pub struct DiagCollector {
    pub dir: PathBuf,
    crops_dir: PathBuf,
    renders_dir: PathBuf,
    pub pages: Vec<PageDiag>,
    pub thoroughness: f32,
}

#[derive(Debug, Serialize)]
pub struct PageDiag {
    pub page_num: usize,
    pub lines: Vec<LineDiag>,
}

#[derive(Debug, Serialize)]
pub struct LineDiag {
    pub line_index: usize,
    pub text: String,
    pub ocr_confidence: f32,
    pub bbox: [u32; 4],  // x, y, w, h
    pub ci_candidates: Vec<CiCandidate>,
    pub words: Vec<WordDiag>,
    pub word_rerank_winner: Option<String>,
    pub final_font: Option<String>,
    pub final_score: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct CiCandidate {
    pub font_key: String,
    pub score: f32,
}

#[derive(Debug, Serialize, Clone)]
pub struct WordDiag {
    pub text: String,
    pub bbox: [u32; 4],
    pub crop_path: String,       // relative to diag dir
    pub candidates: Vec<WordCandidateScore>,
    pub winner: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct WordCandidateScore {
    pub font_key: String,
    pub ssim: f32,
    pub dy: i32,
    pub render_path: String,     // relative to diag dir
}

impl DiagCollector {
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        let crops_dir = dir.join("crops");
        let renders_dir = dir.join("renders");
        fs::create_dir_all(&crops_dir)?;
        fs::create_dir_all(&renders_dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            crops_dir,
            renders_dir,
            pages: Vec::new(),
            thoroughness: 1.0,
        })
    }

    pub fn start_page(&mut self, page_num: usize) {
        self.pages.push(PageDiag {
            page_num,
            lines: Vec::new(),
        });
    }

    pub fn current_page_mut(&mut self) -> &mut PageDiag {
        self.pages.last_mut().expect("call start_page first")
    }

    /// Save a word crop image, return relative path.
    pub fn save_crop(
        &self,
        page: usize,
        line: usize,
        word_idx: usize,
        text: &str,
        img: &image::GrayImage,
    ) -> String {
        let safe: String = text.chars().take(15)
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let rel = format!("crops/p{}_l{}_w{}_{}.png", page, line, word_idx, safe);
        let path = self.dir.join(&rel);
        let _ = img.save(&path);
        rel
    }

    /// Save a rendered word image, return relative path.
    pub fn save_render(
        &self,
        page: usize,
        line: usize,
        word_idx: usize,
        font_key: &str,
        img: &image::GrayImage,
    ) -> String {
        // Use font filename (after last /) for readability — no truncation
        let font_base = font_key.rsplit('/').next().unwrap_or(font_key);
        let safe_font: String = font_base.chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
            .collect();
        let rel = format!("renders/p{}_l{}_w{}_{}.png", page, line, word_idx, safe_font);
        let path = self.dir.join(&rel);
        let _ = img.save(&path);
        rel
    }

    /// Write the HTML report + JSON data.
    pub fn finish(&self) -> std::io::Result<()> {
        // Write JSON data
        let json_path = self.dir.join("data.json");
        let json = serde_json::to_string_pretty(&self.pages)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&json_path, &json)?;

        // Write HTML
        let html_path = self.dir.join("index.html");
        let mut f = fs::File::create(&html_path)?;
        self.write_html(&mut f)?;

        eprintln!("Diagnostic report: {}", html_path.display());
        Ok(())
    }

    fn write_html(&self, f: &mut fs::File) -> std::io::Result<()> {
        write!(f, r##"<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>unscan diagnostic</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 12px; }}
h2 {{ color: #f0883e; margin-top: 40px; }}

.line-block {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.line-block.correct {{ border-left: 4px solid #3fb950; }}
.line-block.wrong {{ border-left: 4px solid #f85149; }}
.line-block.unknown {{ border-left: 4px solid #8b949e; }}
.line-text {{ font-size: 14px; color: #e6edf3; font-weight: 600; margin-bottom: 8px; }}
.line-meta {{ font-size: 12px; color: #8b949e; margin-bottom: 12px; }}
.line-meta b {{ color: #c9d1d9; }}

.ci-table {{ border-collapse: collapse; font-size: 12px; margin: 8px 0; }}
.ci-table th {{ background: #21262d; color: #8b949e; padding: 4px 10px; text-align: left; border: 1px solid #30363d; }}
.ci-table td {{ padding: 4px 10px; border: 1px solid #30363d; }}
.ci-table tr.highlight {{ background: #1a3a1a; }}

details {{ margin: 8px 0; }}
summary {{ cursor: pointer; color: #58a6ff; font-size: 13px; }}
summary:hover {{ text-decoration: underline; }}

.words-grid {{ display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }}
.word-entry {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }}
.word-entry h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #58a6ff; }}
.word-images {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }}
.word-img-cell {{ text-align: center; }}
.word-img-cell img {{ display: block; border: 2px solid #30363d; background: #fff; image-rendering: auto;
    max-height: 60px; min-width: 40px; }}
.word-img-cell img.crop {{ border-color: #58a6ff; }}
.word-img-cell img.winner {{ border-color: #f85149; }}
.word-img-cell img.rank0 {{ border-color: #f0883e; }}
.word-img-cell .label {{ font-size: 10px; color: #8b949e; margin-top: 2px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.word-img-cell .ssim {{ font-size: 11px; font-weight: 700; }}
.word-img-cell .ssim.high {{ color: #3fb950; }}
.word-img-cell .ssim.mid {{ color: #d29922; }}
.word-img-cell .ssim.low {{ color: #f85149; }}

.vote-bar {{ display: flex; gap: 4px; margin-top: 8px; align-items: center; flex-wrap: wrap; }}
.vote-chip {{ font-size: 11px; padding: 2px 8px; border-radius: 12px; background: #21262d; border: 1px solid #30363d; }}
.vote-chip.top {{ background: #f851491a; border-color: #f85149; color: #f85149; }}

.filter-bar {{ position: sticky; top: 0; background: #0d1117; padding: 12px 0; z-index: 10; border-bottom: 1px solid #30363d; margin-bottom: 20px; }}
.filter-bar label {{ font-size: 13px; margin-right: 16px; cursor: pointer; }}
.filter-bar input {{ margin-right: 4px; }}

.stats {{ display: flex; gap: 24px; margin: 16px 0; font-size: 14px; }}
.stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 20px; }}
.stat .num {{ font-size: 28px; font-weight: 700; }}
.stat .lbl {{ font-size: 12px; color: #8b949e; }}
</style>
</head><body>
<h1>unscan diagnostic report</h1>
"##)?;

        // Thoroughness info
        let t = self.thoroughness;
        write!(f, r#"<div class="summary"><strong>thoroughness: {t:.1}</strong> &mdash;
kd-tree radius factor: {kd:.2}, quality gate: {qg:.2}, quorum divisor: {t:.1}</div>"#,
            t = t,
            kd = 1.5 * t,
            qg = 0.5 * t,
        )?;

        // Summary stats
        let total_lines: usize = self.pages.iter().map(|p| p.lines.len()).sum();
        let total_words: usize = self.pages.iter()
            .flat_map(|p| &p.lines)
            .map(|l| l.words.len())
            .sum();
        let matched: usize = self.pages.iter()
            .flat_map(|p| &p.lines)
            .filter(|l| l.final_font.is_some())
            .count();

        write!(f, r#"<div class="stats">
<div class="stat"><div class="num">{}</div><div class="lbl">lines</div></div>
<div class="stat"><div class="num">{}</div><div class="lbl">words scored</div></div>
<div class="stat"><div class="num">{}</div><div class="lbl">fonts matched</div></div>
</div>"#, total_lines, total_words, matched)?;

        // Filter bar
        write!(f, r#"<div class="filter-bar">
<label><input type="checkbox" id="show-all" checked onchange="toggleFilter()"> Show all lines</label>
<label><input type="checkbox" id="show-matched" checked onchange="toggleFilter()"> Matched</label>
<label><input type="checkbox" id="show-unmatched" checked onchange="toggleFilter()"> Unmatched</label>
</div>"#)?;

        for page in &self.pages {
            write!(f, "<h2>Page {}</h2>\n", page.page_num)?;

            for line in &page.lines {
                let status_class = if line.final_font.is_some() { "correct" } else { "unknown" };
                let font_display = line.final_font.as_deref().unwrap_or("—");
                let score_display = line.final_score.map(|s| format!("{:.4}", s)).unwrap_or_else(|| "—".to_string());

                write!(f, r#"<div class="line-block {sc}" data-matched="{m}">
<div class="line-text">{text}</div>
<div class="line-meta">
  OCR conf: <b>{conf:.0}</b> | Final font: <b>{font}</b> | Score: <b>{score}</b>
  {wrw}
</div>
"#,
                    sc = status_class,
                    m = if line.final_font.is_some() { "yes" } else { "no" },
                    text = html_escape(&line.text),
                    conf = line.ocr_confidence,
                    font = html_escape(font_display),
                    score = score_display,
                    wrw = line.word_rerank_winner.as_ref()
                        .map(|w| format!("| Word rerank: <b>{}</b>", html_escape(w)))
                        .unwrap_or_default(),
                )?;

                // CI candidates (collapsible)
                if !line.ci_candidates.is_empty() {
                    let top_ci_score = line.ci_candidates.first().map(|c| c.score).unwrap_or(0.0);
                    write!(f, r#"<details><summary>CI candidates ({} fonts passed quorum)</summary>
<table class="ci-table"><tr><th>#</th><th>Font</th><th>CI Score</th><th>Gap from #1</th><th>Min top-N to include</th></tr>"#,
                        line.ci_candidates.len())?;
                    for (i, c) in line.ci_candidates.iter().enumerate() {
                        let hl = if line.final_font.as_deref()
                            .map(|ff| c.font_key.contains(&ff.split(' ').next().unwrap_or("")))
                            .unwrap_or(false)
                        {
                            " class=\"highlight\""
                        } else { "" };
                        let short = c.font_key.split('/').last().unwrap_or(&c.font_key);
                        let gap = top_ci_score - c.score;
                        write!(f, "<tr{}><td>{}</td><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>top {}</td></tr>\n",
                            hl, i + 1, html_escape(short), c.score, gap, i + 1)?;
                    }
                    write!(f, "</table></details>\n")?;
                }

                // Word-level diagnostics
                if !line.words.is_empty() {
                    write!(f, "<details open><summary>Word SSIM ({} words)</summary>\n<div class=\"words-grid\">\n",
                        line.words.len())?;

                    for wd in &line.words {
                        write!(f, "<div class=\"word-entry\">\n")?;
                        write!(f, "<h4>\"{}\" ({}×{})</h4>\n",
                            html_escape(&wd.text), wd.bbox[2], wd.bbox[3])?;
                        write!(f, "<div class=\"word-images\">\n")?;

                        // Page crop
                        let crop_uri = img_data_uri(&self.dir, &wd.crop_path);
                        write!(f, r#"<div class="word-img-cell">
<img class="crop" src="{}">
<div class="label">page crop</div>
</div>"#, &crop_uri)?;

                        // Top candidates
                        let best_word_ssim = wd.candidates.first().map(|c| c.ssim).unwrap_or(0.0);
                        for (rank, cand) in wd.candidates.iter().enumerate() {
                            let short = cand.font_key.split('/').last().unwrap_or(&cand.font_key);
                            let is_winner = wd.winner.as_deref() == Some(&cand.font_key);
                            let img_class = if is_winner { "winner" } else if rank == 0 { "rank0" } else { "" };
                            let ssim_class = if cand.ssim > 0.7 { "high" } else if cand.ssim > 0.4 { "mid" } else { "low" };
                            let render_uri = img_data_uri(&self.dir, &cand.render_path);
                            let gap = best_word_ssim - cand.ssim;
                            let gap_str = if rank == 0 { String::new() } else { format!(" Δ{:.4}", gap) };
                            write!(f, r#"<div class="word-img-cell">
<img class="{ic}" src="{src}">
<div class="label">{name}</div>
<div class="ssim {sc}">SSIM: {ssim:.4} (dy={dy}){gap}</div>
</div>"#,
                                ic = img_class,
                                src = &render_uri,
                                name = html_escape(short),
                                sc = ssim_class,
                                ssim = cand.ssim,
                                dy = cand.dy,
                                gap = gap_str,
                            )?;
                        }

                        write!(f, "</div>\n")?; // word-images

                        // Vote summary
                        if let Some(ref winner) = wd.winner {
                            let short = winner.split('/').last().unwrap_or(winner);
                            write!(f, r#"<div class="vote-bar"><span class="vote-chip top">→ {}</span></div>"#,
                                html_escape(short))?;
                        }

                        write!(f, "</div>\n")?; // word-entry
                    }
                    write!(f, "</div></details>\n")?;
                }

                write!(f, "</div>\n")?; // line-block
            }
        }

        // JS filter
        write!(f, r#"<script>
function toggleFilter() {{
    const all = document.getElementById('show-all').checked;
    const matched = document.getElementById('show-matched').checked;
    const unmatched = document.getElementById('show-unmatched').checked;
    document.querySelectorAll('.line-block').forEach(el => {{
        const m = el.dataset.matched === 'yes';
        el.style.display = (all || (matched && m) || (unmatched && !m)) ? '' : 'none';
    }});
}}
</script>"#)?;

        write!(f, "</body></html>")?;
        Ok(())
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

/// Read an image file from the diagnostic dir and return a base64 data URI.
/// Falls back to the relative path if reading fails.
fn img_data_uri(dir: &Path, rel_path: &str) -> String {
    if rel_path.is_empty() {
        return String::new();
    }
    let full = dir.join(rel_path);
    match fs::read(&full) {
        Ok(bytes) => {
            let b64 = b64_encode(&bytes);
            format!("data:image/png;base64,{}", b64)
        }
        Err(_) => rel_path.to_string(),
    }
}

fn b64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARS[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
