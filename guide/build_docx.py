"""Build guide/Assignment8_User_Guide_Koziol.docx from the markdown guide in the Assignment 5A/6 layout.
Set APP_URL in the environment to fill in the deployed link."""
import os
import re
import datetime
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
HERE = os.path.dirname(os.path.abspath(__file__))
APP_URL = os.environ.get("APP_URL", "{{APP_URL}}")


def path(*p):
    return os.path.join(HERE, *p)

MD = path("Assignment8_User_Guide.md")
OUT_DOCX = path("Assignment8_User_Guide_Koziol.docx")
BODY_FONT = "Times New Roman"

doc = Document()
st = doc.styles["Normal"]
st.font.name = BODY_FONT
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
st.paragraph_format.space_after = Pt(8)
st.paragraph_format.line_spacing = 1.16
for s in doc.sections:
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.left_margin = s.right_margin = Inches(1)
    s.top_margin = s.bottom_margin = Inches(1)


def add_runs(par, text, size=None):
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1])
        elif tok.startswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            r.italic = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size)


def add_table(block):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            add_runs(p, val, size=9)
            for r in p.runs:
                r.bold = r.bold or i == 0
    doc.add_paragraph()


def add_figure(src, caption):
    doc.add_picture(path("figures", os.path.basename(src)), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    add_runs(p, caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()


def add_code(lines):
    p = doc.add_paragraph()
    r = p.add_run("\n".join(lines))
    r.font.name = "Consolas"
    r.font.size = Pt(9)


def add_bullet(text):
    """5A style: the hyphen is part of the paragraph and shares the bold of a bold label."""
    p = doc.add_paragraph()
    if text.startswith("**"):
        add_runs(p, "**- " + text[2:])
    else:
        add_runs(p, "- " + text)


lines = open(MD, encoding="utf-8").read().replace("{{APP_URL}}", APP_URL).splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("```"):
        block = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            block.append(lines[i])
            i += 1
        add_code(block)
        i += 1
        continue
    if line.startswith("|"):
        block = []
        while i < len(lines) and lines[i].startswith("|"):
            block.append(lines[i])
            i += 1
        add_table(block)
        continue
    m = re.match(r"^(#{1,3})\s+(.*)", line)
    if m:
        level, text = len(m.group(1)), m.group(2)
        if level == 1:
            p = doc.add_paragraph()
            add_runs(p, text, size=14)
        else:
            doc.add_paragraph()
            p = doc.add_paragraph()
            add_runs(p, text)
            for r in p.runs:
                r.bold = True
        i += 1
        continue
    m = re.match(r"^!\[(.*)\]\((.*)\)", line)
    if m:
        add_figure(m.group(2), m.group(1))
        i += 1
        continue
    m = re.match(r"^\s*[-*]\s+(.*)", line)
    if m:
        add_bullet(m.group(1))
        i += 1
        continue
    if line.strip() and not line.startswith("<!--"):
        par = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|<!--|```|\s*[-*]\s)", lines[i + 1]):
            i += 1
            par.append(lines[i])
        add_runs(doc.add_paragraph(), " ".join(par))
    i += 1

cp = doc.core_properties                      # python-docx's template stamps itself as author with a 2013 date
cp.author = cp.last_modified_by = "Matthew Koziol"
cp.title = "Assignment 8 Flight Delay Risk Predictor User Guide"
cp.comments = ""
cp.created = cp.modified = datetime.datetime.now()
cp.revision = 1
doc.save(OUT_DOCX)
print("wrote", OUT_DOCX)
