"""Helpers for building the report with python-docx."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

plt.rcParams["mathtext.fontset"] = "cm"


def equation_png(latex, path, fontsize=11):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=False,
                facecolor="white")
    plt.close(fig)
    return path


def new_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.7)
    sec.top_margin, sec.bottom_margin = Cm(1.5), Cm(1.5)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(2.5)
    st.paragraph_format.line_spacing = 1.0
    for lvl, size in ((1, 11), (2, 10)):
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = RGBColor(0x1f, 0x3a, 0x68)
        h.paragraph_format.space_before = Pt(5 if lvl == 1 else 3)
        h.paragraph_format.space_after = Pt(1.5)
        h.paragraph_format.keep_with_next = True
    return doc


def two_columns(doc):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    cols = sec._sectPr.xpath("./w:cols")
    col = cols[0] if cols else OxmlElement("w:cols")
    col.set(qn("w:num"), "2")
    col.set(qn("w:space"), "340")
    if not cols:
        sec._sectPr.append(col)
    return sec


def one_column(doc):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    cols = sec._sectPr.xpath("./w:cols")
    col = cols[0] if cols else OxmlElement("w:cols")
    col.set(qn("w:num"), "1")
    if not cols:
        sec._sectPr.append(col)
    return sec


def para(doc, text, bold_lead=None, align=None, size=None, italic=False, after=None):
    """Paragraph with **bold** segments marked by double asterisks."""
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead + " ")
        r.bold = True
        if size:
            r.font.size = Pt(size)
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = i % 2 == 1
        r.italic = italic
        if size:
            r.font.size = Pt(size)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(doc, text, bold_lead=None):
    p = para(doc, text, bold_lead)
    p.style = doc.styles["List Bullet"]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Cm(0.45)
    return p


def equation(doc, png, width_cm, label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1.5)
    p.add_run().add_picture(str(png), width=Cm(width_cm))
    if label:
        p.add_run(f"   ({label})").font.size = Pt(8.5)
    return p


def figure(doc, png, width_cm, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(png), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(8)
    c.paragraph_format.space_after = Pt(3)
    return c


def _shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def table(doc, header, rows, widths_cm=None, font=7.5, caption=None, bold_rows=()):
    if caption:
        c = doc.add_paragraph()
        r = c.add_run(caption)
        r.italic = True
        r.font.size = Pt(8)
        c.paragraph_format.space_after = Pt(1)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(font)
        _shade(cell, "DCE4F2")
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
            run.bold = (i - 1) in bold_rows
    for row in t.rows:
        for j, cell in enumerate(row.cells):
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            if j > 0:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if widths_cm:
                cell.width = Cm(widths_cm[j])
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(1)
    return t
