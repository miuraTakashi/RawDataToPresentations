#!/usr/bin/env python3
"""
Shared PowerPoint layout helpers used by nd2*_to_pptx.py scripts.

Goal:
- 4:3 slide size
- white background
- all text black
- simple absolute-position helpers
"""

from __future__ import annotations

import os
from typing import List, Optional

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt, Emu


# 4:3 typical slide (PowerPoint default)
SLIDE_W = Inches(10.0)
SLIDE_H = Inches(7.5)

# White background, black text
BG_COLOR = RGBColor(0xFF, 0xFF, 0xFF)
TITLE_COLOR = RGBColor(0x00, 0x00, 0x00)
TEXT_COLOR = RGBColor(0x00, 0x00, 0x00)

ACCENT_RED = RGBColor(0xDC, 0x32, 0x32)
ACCENT_BLUE = RGBColor(0x00, 0x78, 0xFF)

FONT_NAME = "Helvetica Neue"


def new_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def find_blank_layout(prs: Presentation):
    for layout in prs.slide_layouts:
        if layout.name == "Blank":
            return layout
    return prs.slide_layouts[6]


def set_slide_bg(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR


def _add_textbox(slide, text: str, left, top, width, height,
                 font_size: int, color,
                 bold: bool = False,
                 alignment=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = FONT_NAME
    p.alignment = alignment
    return tb


def add_multiline_text(slide, lines: List[str], left, top, width, height,
                        font_size: int = 14,
                        color=TEXT_COLOR,
                        bold: bool = False) -> None:
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.clear()

    for idx, line in enumerate(lines):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        if line.startswith("\t"):
            p.text = line[1:]
            p.level = 1
        else:
            p.text = line
            p.level = 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.font.name = FONT_NAME
        p.alignment = PP_ALIGN.LEFT


def add_title(slide, title: str, left, top, width, height) -> None:
    _add_textbox(
        slide,
        title,
        left=left,
        top=top,
        width=width,
        height=height,
        font_size=24,
        color=TITLE_COLOR,
        bold=True,
        alignment=PP_ALIGN.LEFT,
    )


def _extract_poster_png_from_mp4(mp4_path: str, out_png: str) -> bool:
    """
    Try to extract a single poster frame using OpenCV.
    Returns True on success.
    """
    try:
        import cv2  # lazy import for speed

        cap = cv2.VideoCapture(mp4_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        import PIL.Image

        # BGR -> RGB
        rgb = frame[:, :, ::-1]
        PIL.Image.fromarray(rgb).save(out_png, "PNG")
        return True
    except Exception:
        return False


def add_movie(slide, mp4_path: str, left, top, width, height,
              poster_png: Optional[str] = None,
              try_extract_poster: bool = True) -> None:
    """
    Embed an MP4 movie into a slide.
    poster_png: if provided, use it; otherwise optional extraction.
    """
    if try_extract_poster and poster_png is not None and not os.path.exists(poster_png):
        _extract_poster_png_from_mp4(mp4_path, poster_png)

    if poster_png is not None and os.path.exists(poster_png):
        try:
            slide.shapes.add_movie(
                mp4_path,
                left,
                top,
                width,
                height,
                poster_frame_image=poster_png,
                mime_type="video/mp4",
            )
            return
        except Exception:
            pass

    # Fallback: try without poster
    slide.shapes.add_movie(
        mp4_path,
        left,
        top,
        width,
        height,
        mime_type="video/mp4",
    )


def add_picture(slide, img_path: str, left, top, width=None, height=None) -> None:
    if width is not None and height is not None:
        slide.shapes.add_picture(img_path, left, top, width=width, height=height)
    else:
        slide.shapes.add_picture(img_path, left, top)

