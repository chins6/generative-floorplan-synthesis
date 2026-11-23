from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x1F, 0x3A, 0x68)
GREY = RGBColor(0x49, 0x50, 0x57)
LIGHT = RGBColor(0xF1, 0xF3, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MODEL = {"DAE": RGBColor(0x2A, 0x6F, 0xDB), "VAE": RGBColor(0xD9, 0x48, 0x0F),
         "GNN": RGBColor(0x2B, 0x8A, 0x3E)}


def new_deck():
    p = Presentation()
    p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
    return p


def slide(prs, title, number, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(11.6), Inches(0.75))
    tf = tb.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run()
    r.text = title
    r.font.size, r.font.bold, r.font.color.rgb = Pt(26), True, WHITE
    if subtitle:
        r2 = tf.paragraphs[0].add_run()
        r2.text = "   " + subtitle
        r2.font.size, r2.font.color.rgb = Pt(14), RGBColor(0xC5, 0xD3, 0xEA)
    nb = s.shapes.add_textbox(Inches(12.4), Inches(7.05), Inches(0.8), Inches(0.35))
    q = nb.text_frame.paragraphs[0]
    q.alignment = PP_ALIGN.RIGHT
    rr = q.add_run()
    rr.text = f"{number} / 7"
    rr.font.size, rr.font.color.rgb = Pt(10), GREY
    return s


def text(s, x, y, w, h, lines, size=14, color=GREY, bold_first=False, bullets=False,
         spacing=4, align=None):
    """lines: list of str or (str, dict(size=, bold=, color=)). '**x**' marks bold runs."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        opts = {}
        if isinstance(line, tuple):
            line, opts = line
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(opts.get("after", spacing))
        if align == "center":
            p.alignment = PP_ALIGN.CENTER
        prefix = "• " if bullets and not opts.get("nobullet") else ""
        parts = (prefix + line).split("**")
        for k, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", (bold_first and i == 0)) or k % 2 == 1
            r.font.color.rgb = opts.get("color", color)
    return tb


def card(s, x, y, w, h, title, lines, color=NAVY, size=12, title_size=13, fill=LIGHT):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.adjustments[0] = 0.05
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = color
    box.line.width = Pt(1.25)
    box.shadow.inherit = False
    text(s, x + 0.12, y + 0.06, w - 0.24, 0.4, [(title, dict(size=title_size, bold=True, color=color))])
    text(s, x + 0.12, y + 0.45, w - 0.24, h - 0.5, lines, size=size, bullets=True, spacing=3)
    return box


def picture(s, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def caption(s, x, y, w, txt, size=10):
    return text(s, x, y, w, 0.3, [(txt, dict(size=size, color=GREY))], align="center")


def table(s, x, y, w, h, header, rows, col_w=None, size=11, highlight_rows=(), colors=None):
    shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(h))
    t = shape.table
    for j, hd in enumerate(header):
        c = t.cell(0, j)
        c.text = hd
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        para = c.text_frame.paragraphs[0]
        para.runs[0].font.size, para.runs[0].font.bold = Pt(size), True
        para.runs[0].font.color.rgb = WHITE
        para.alignment = PP_ALIGN.CENTER
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0xE7, 0xF5, 0xFF) if (i - 1) in highlight_rows else (
                WHITE if i % 2 else LIGHT)
            para = c.text_frame.paragraphs[0]
            run = para.runs[0]
            run.font.size = Pt(size)
            run.font.bold = (i - 1) in highlight_rows or j == 0
            if j == 0 and colors and v in colors:
                run.font.color.rgb = colors[v]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            c.margin_top = c.margin_bottom = Emu(30000)
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    return t


def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt
