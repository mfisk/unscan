#!/usr/bin/env python3
"""Convert unscan audit HTML report to PDF with low memory usage.

Strategy: parse HTML, decode images to temp files on disk one at a time,
reference them in ReportLab flowables. Temp files keep memory low while
avoiding the closed-BytesIO problem.
"""
import re, sys, os, base64, tempfile, gc
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from PIL import Image as PILImage


def safe_xml(text):
    return (text.replace('&','&amp;').replace('<','&lt;')
            .replace('>','&gt;').replace('"','&quot;'))


def extract_sections(html_path):
    with open(html_path, 'r') as f:
        content = f.read()
    title_m = re.search(r'<h2[^>]*>(.*?)</h2>', content)
    title = re.sub(r'<[^>]+>', '', title_m.group(1)) if title_m else 'Audit Report'
    yield ('title', title)
    h3_pattern = re.compile(
        r'<h3[^>]*>(.*?)</h3>(.*?)(?=<h3[^>]*>|<h2[^>]*>|\Z)', re.DOTALL)
    for m in h3_pattern.finditer(content):
        header = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        body_html = m.group(2)
        imgs = [(im.group(1), im.group(2))
                for im in re.finditer(
                    r'<img[^>]+src="data:image/(png|jpeg|jpg|gif);base64,([^"]+)"',
                    body_html, re.DOTALL)]
        text = re.sub(r'<img[^>]*/?>', '', body_html)
        text = re.sub(r'<br\s*/?>', '\n', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        yield ('entry', header, text, imgs)


def b64_to_tmp(fmt, b64data, tmpdir):
    try:
        raw = base64.b64decode(b64data)
        ext = 'jpg' if fmt in ('jpeg','jpg') else fmt
        fd, path = tempfile.mkstemp(suffix=f'.{ext}', dir=tmpdir)
        os.write(fd, raw); os.close(fd); del raw
        with PILImage.open(path) as pil:
            w, h = pil.size
        return path, w, h
    except Exception:
        return None, 0, 0


def mk_img(path, w, h, max_w, max_h):
    scale = min(max_w/w, max_h/h, 1.0)
    return Image(path, width=w*scale, height=h*scale)


def build_pdf(html_path, pdf_path):
    tmpdir = tempfile.mkdtemp(prefix='audit_imgs_')
    print(f"Temp dir: {tmpdir}", file=sys.stderr)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
        leftMargin=0.5*inch, rightMargin=0.5*inch,
        topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    ts = ParagraphStyle('T', parent=styles['Title'], fontSize=16, spaceAfter=12)
    hs = ParagraphStyle('H', parent=styles['Heading3'], fontSize=9,
        spaceAfter=4, spaceBefore=8, leading=11, textColor=colors.HexColor('#333'))
    bs = ParagraphStyle('B', parent=styles['Normal'], fontSize=7,
        spaceAfter=4, leading=9, textColor=colors.HexColor('#555'))

    story = []; tmp_files = []; ec = 0; ic = 0
    print(f"Parsing {html_path}...", file=sys.stderr)

    for section in extract_sections(html_path):
        if section[0] == 'title':
            story.append(Paragraph(safe_xml(section[1]), ts))
            story.append(Spacer(1, 12))
        elif section[0] == 'entry':
            _, header, text, images = section
            ec += 1
            story.append(Paragraph(safe_xml(header), hs))
            if text and len(text) > 5:
                story.append(Paragraph(safe_xml(text[:500]), bs))
            row = []
            for fmt, b64 in images:
                path, w, h = b64_to_tmp(fmt, b64, tmpdir)
                if path:
                    tmp_files.append(path)
                    row.append(mk_img(path, w, h, 3.3*inch, 1.5*inch))
                    ic += 1
                    if len(row) == 2:
                        t = Table([row], colWidths=[3.5*inch, 3.5*inch])
                        t.setStyle(TableStyle([
                            ('VALIGN',(0,0),(-1,-1),'TOP'),
                            ('LEFTPADDING',(0,0),(-1,-1),2),
                            ('RIGHTPADDING',(0,0),(-1,-1),2)]))
                        story.append(t); story.append(Spacer(1,4)); row = []
            if row:
                story.append(row[0]); story.append(Spacer(1,4))
            if ec % 50 == 0:
                print(f"  ...{ec} entries, {ic} images", file=sys.stderr)
                gc.collect()

    print(f"Building PDF: {ec} entries, {ic} images...", file=sys.stderr)
    doc.build(story)
    print(f"Done: {pdf_path}", file=sys.stderr)
    for f in tmp_files:
        try: os.unlink(f)
        except OSError: pass
    try: os.rmdir(tmpdir)
    except OSError: pass


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.html> <output.pdf>", file=sys.stderr)
        sys.exit(1)
    build_pdf(sys.argv[1], sys.argv[2])
