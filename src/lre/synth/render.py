"""Render a Jamabandi-style register page (report Figure 1, D.6 step 2).

The printed form (title band, header row, 12 column rules) is identical on every
page, as on a real pre-printed register. Handwritten values are drawn word by
word with a handwriting-style font, random slant, baseline jitter, stroke
weight and ink colour. Every cell and every text line is recorded with its box,
so pages double as training data for the layout analyser and the recogniser.

All coordinates here are in the *form frame* (the page before scan skew).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from ..schema import COLUMN_FIELDS
from . import vocab as V


class FontError(RuntimeError):
    pass


def check_complex_script_support() -> None:
    """Devanagari needs a shaping engine (libraqm); without it matras and conjuncts render wrongly."""
    if not features.check("raqm"):
        raise FontError(
            "Pillow was built without libraqm, so Devanagari would be shaped incorrectly. "
            "Install it (macOS: `brew install libraqm`; Ubuntu: `apt install libraqm0`) and retry.")


@lru_cache(maxsize=64)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    if not Path(path).exists():
        raise FontError(f"font not found: {path}. Run `lre fonts` to download the OFL fonts.")
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


# ----------------------------------------------------------------------------- form geometry


@dataclass
class Form:
    width: int
    height: int
    x_edges: list[int]          # 13 x positions of the column rules
    title_y: int
    table_top: int
    header_bottom: int
    table_bottom: int
    pad_x: int
    pad_y: int

    def col_box(self, c: int) -> tuple[int, int]:
        return self.x_edges[c], self.x_edges[c + 1]


def make_form(width: int, height: int) -> Form:
    # Margins leave room for scan skew (up to 5 deg) without rotating the table out of the image.
    left, right = int(width * 0.04), int(width * 0.96)
    usable = right - left
    xs = [left]
    for w in V.COLUMN_WIDTHS:
        xs.append(xs[-1] + w * usable)
    xs = [int(round(x)) for x in xs]
    xs[-1] = right
    table_top = int(height * 0.075)
    return Form(width=width, height=height, x_edges=xs, title_y=int(height * 0.03),
                table_top=table_top, header_bottom=table_top + int(height * 0.07),
                table_bottom=int(height * 0.94), pad_x=int(width * 0.004), pad_y=int(height * 0.006))


# ----------------------------------------------------------------------------- style


@dataclass
class HandStyle:
    """How one Patwari writes: fixed for a whole village register."""
    font_path: str
    font_px: int
    slant_deg: float
    line_h: int = 0

    def __post_init__(self):
        if not self.line_h:
            self.line_h = int(self.font_px * 1.5)


@dataclass
class PageInk:
    """What varies between pages of the same register."""
    ink: tuple[int, int, int]
    red_ink: tuple[int, int, int]
    paper: tuple[int, int, int]
    rule_color: tuple[int, int, int]
    heavy_stroke: bool
    seed: int


# ----------------------------------------------------------------------------- wrapping / layout


def _slant_extra(style: HandStyle) -> int:
    return int(abs(math.tan(math.radians(style.slant_deg))) * style.line_h) + 4


def wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = f"{cur} {w}" if cur else w
        if cur and font.getlength(cand) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


@dataclass
class HoldingLayout:
    lines: dict[int, list[str]]
    height: int


def layout_holding(texts: dict[int, list[str]], form: Form, style: HandStyle) -> HoldingLayout:
    font = load_font(style.font_path, style.font_px)
    extra = _slant_extra(style)
    lines: dict[int, list[str]] = {}
    for c in range(12):
        x0, x1 = form.col_box(c)
        max_w = (x1 - x0) - 2 * form.pad_x - extra
        out: list[str] = []
        for seg in texts.get(c, []):
            out.extend(wrap(seg, font, max_w))
        lines[c] = out
    n = max(2, max(len(v) for v in lines.values()))
    return HoldingLayout(lines=lines, height=n * style.line_h + 2 * form.pad_y + style.line_h // 3)


def paginate(layouts: list[HoldingLayout], form: Form) -> list[list[int]]:
    """Greedy: fill each page's table with whole holdings, top to bottom."""
    avail = form.table_bottom - form.header_bottom
    pages, cur, used = [], [], 0
    for i, lay in enumerate(layouts):
        if cur and used + lay.height > avail:
            pages.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += lay.height
    if cur:
        pages.append(cur)
    return pages


# ----------------------------------------------------------------------------- drawing


@dataclass
class DrawnLine:
    text: str
    bbox: tuple[int, int, int, int]          # x, y, w, h (form frame)


@dataclass
class DrawnCell:
    col: int
    field: str
    bbox: tuple[int, int, int, int]
    lines: list[DrawnLine] = field(default_factory=list)


def draw_hand_line(page: Image.Image, x: int, y: int, text: str, style: HandStyle,
                   color: tuple[int, int, int], heavy: bool, rng: np.random.Generator) -> tuple | None:
    """Draw one handwritten line with its top-left near (x, y). Returns the tight ink box."""
    font = load_font(style.font_path, style.font_px)
    words = text.split()
    if not words:
        return None
    space = font.getlength(" ")
    widths = [font.getlength(w) for w in words]
    total = sum(widths) + space * 1.3 * (len(words) - 1)
    h = style.line_h + style.font_px // 2
    shear = math.tan(math.radians(style.slant_deg))
    margin = int(abs(shear) * h) + 8
    layer = Image.new("L", (int(total * 1.15) + 2 * margin + 8, h), 0)
    d = ImageDraw.Draw(layer)
    cx = margin
    jitter_amp = max(1.0, style.font_px * 0.05)
    for w, wd in zip(words, widths):
        dy = rng.normal(0, jitter_amp)
        d.text((cx, style.font_px * 0.15 + dy), w, font=font, fill=int(rng.integers(215, 256)))
        cx += wd + space * rng.uniform(0.9, 1.6)
    # slant: shift rows horizontally in proportion to their height above the baseline
    base = h * 0.75
    layer = layer.transform(layer.size, Image.AFFINE, (1, shear, -shear * base, 0, 1, 0),
                            resample=Image.BILINEAR)
    if heavy:
        layer = layer.filter(ImageFilter.MaxFilter(3))
    # pen pressure / drying ink: smooth intensity variation along the line
    arr = np.asarray(layer, dtype=np.float32)
    ramp = np.interp(np.arange(arr.shape[1]), np.linspace(0, arr.shape[1], 4),
                     rng.uniform(0.7, 1.0, size=4)).astype(np.float32)
    arr = arr * ramp[None, :]
    layer = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.4))
    box = layer.point(lambda v: 255 if v > 60 else 0).getbbox()
    if box is None:
        return None
    ox, oy = x - margin, y
    page.paste(Image.new("RGB", layer.size, color), (ox, oy), layer)
    l, t, r, b = box
    return (ox + l, oy + t, r - l, b - t)


def draw_printed(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, anchor="mm") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_form(page: Image.Image, form: Form, rows_y: list[int], printed_font: str,
              title: str, ink: PageInk, rng: np.random.Generator) -> dict:
    """Printed title, header and column rules; hand-ruled lines between holdings.

    Returns the exact geometry of every ruling line (for U-Net masks and grid evaluation).
    """
    d = ImageDraw.Draw(page)
    rule_w = max(3, form.width // 1400)
    hfont = load_font(printed_font, max(18, form.width // 150))
    tfont = load_font(printed_font, max(24, form.width // 110))
    draw_printed(d, (form.width // 2, form.title_y + tfont.size), title, tfont, ink.rule_color)
    h_lines, v_lines = [], []
    for x in form.x_edges:
        d.line([(x, form.table_top), (x, form.table_bottom)], fill=ink.rule_color, width=rule_w)
        v_lines.append({"x": x, "y0": form.table_top, "y1": form.table_bottom, "w": rule_w})
    for y in (form.table_top, form.header_bottom, form.table_bottom):
        d.line([(form.x_edges[0], y), (form.x_edges[-1], y)], fill=ink.rule_color, width=rule_w)
        h_lines.append({"y": y, "x0": form.x_edges[0], "x1": form.x_edges[-1], "w": rule_w})
    for c in range(12):
        x0, x1 = form.col_box(c)
        cx = (x0 + x1) // 2
        lines = [str(c + 1)] + wrap(V.COLUMN_HEADERS[c], hfont, (x1 - x0) - 10)
        lh = int(hfont.size * 1.5)
        y = (form.table_top + form.header_bottom) // 2 - lh * (len(lines) - 1) // 2
        for i, t in enumerate(lines):
            draw_printed(d, (cx, y + i * lh), t, hfont, ink.rule_color)
    # Patwari's hand-ruled separators: ink colour, slightly uneven thickness
    for y in rows_y:
        w = int(rng.integers(2, 4))
        d.line([(form.x_edges[0], y), (form.x_edges[-1], y)], fill=ink.ink, width=w)
        h_lines.append({"y": y, "x0": form.x_edges[0], "x1": form.x_edges[-1], "w": w})
    return {"h_lines": h_lines, "v_lines": v_lines}


def render_page(holding_layouts: list[HoldingLayout], is_remark_red: list[bool], form: Form,
                style: HandStyle, ink: PageInk, printed_font: str, title: str):
    """Draw one page. Returns (PIL image, per-holding list of DrawnCell, row boxes, rule geometry)."""
    rng = np.random.default_rng(ink.seed)
    page = Image.new("RGB", (form.width, form.height), ink.paper)
    y = form.header_bottom
    row_spans = []
    for lay in holding_layouts:
        row_spans.append((y, y + lay.height))
        y += lay.height
    rules = draw_form(page, form, [b for _, b in row_spans], printed_font, title, ink, rng)

    all_cells: list[list[DrawnCell]] = []
    for lay, (y0, y1), red in zip(holding_layouts, row_spans, is_remark_red):
        cells = []
        for c in range(12):
            x0, x1 = form.col_box(c)
            cell = DrawnCell(col=c, field=COLUMN_FIELDS[c].value, bbox=(x0, y0, x1 - x0, y1 - y0))
            ty = y0 + form.pad_y + int(rng.integers(0, max(1, form.pad_y)))
            for text in lay.lines[c]:
                tx = x0 + form.pad_x + int(rng.integers(0, max(1, form.pad_x)))
                color = ink.red_ink if (c == 11 and red) else ink.ink
                box = draw_hand_line(page, tx, ty, text, style, color, ink.heavy_stroke, rng)
                if box:
                    cell.lines.append(DrawnLine(text=text, bbox=box))
                ty += style.line_h
            cells.append(cell)
        all_cells.append(cells)
    return page, all_cells, row_spans, rules
