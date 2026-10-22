#!/usr/bin/env python3
"""
nd2FRAP_to_pptx.py – Create a PowerPoint (.pptx) from FRAP ND2 files.

For each ND2 file the script:
  1. Converts the time-series to an MP4 movie (embedded in the slide).
  2. Extracts the ROI recovery curve.
  3. Creates two slides:
       Slide A – Movie with poster frame, title, and FRAP summary.
       Slide B – Native editable XY-scatter chart + measurement data table.
  4. Exports a CSV file with raw time / intensity / recovery data.

Advantages over the Keynote version:
  - Chart and table are native PowerPoint objects (editable, linked data).
  - No AppleScript / macOS dependency for presentation creation.
  - CSV data files produced for every ND2 input.
"""

import argparse
import csv
import os
import sys
import tempfile
from typing import List, Tuple, Optional, Dict

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import nd2_utils
    from nd2_utils import (
        detect_fps_from_metadata,
        classify_channel_names,
        extract_side_length_um,
        get_pixel_size_um,
    )
except ImportError:
    nd2_utils = None
    detect_fps_from_metadata = None  # type: ignore
    classify_channel_names = None    # type: ignore
    extract_side_length_um = None    # type: ignore
    get_pixel_size_um = None         # type: ignore

try:
    import nd2_to_mp4
except Exception:
    sys.stderr.write("ERROR: nd2_to_mp4 module not found beside this script.\n")
    raise

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import XyChartData

# ━━ Design tokens (white background, black text) ━━━━━━━━━━━━━━━
BG_COLOR        = RGBColor(0xFF, 0xFF, 0xFF)
TITLE_COLOR     = RGBColor(0x00, 0x00, 0x00)
SUBTITLE_COLOR  = RGBColor(0x00, 0x00, 0x00)
ACCENT_RED      = RGBColor(0xDC, 0x32, 0x32)
ACCENT_BLUE     = RGBColor(0x00, 0x78, 0xFF)
TBL_HDR_BG      = RGBColor(0xF2, 0xF2, 0xF2)
TBL_ODD_BG      = RGBColor(0xFF, 0xFF, 0xFF)
TBL_EVEN_BG     = RGBColor(0xF8, 0xF8, 0xF8)
TBL_TEXT_COLOR  = RGBColor(0x00, 0x00, 0x00)
FONT_NAME      = "Helvetica Neue"

# 4:3 aspect ratio (typical PowerPoint default)
SLIDE_W = Inches(10.0)
SLIDE_H = Inches(7.5)

MAX_CHART_POINTS = 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ND2 file discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_nd2_files(path: str, non_recursive: bool) -> List[str]:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    result: List[str] = []
    if non_recursive:
        entries = os.listdir(path)
        for f in entries:
            if f.lower().endswith(".nd2") and not f.startswith((".", "._")):
                result.append(os.path.join(path, f))
    else:
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".nd2") and not f.startswith((".", "._")):
                    result.append(os.path.join(root, f))
    result.sort()
    return result


def _parse_roi(s: str) -> Tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi requires x,y,w,h")
    return (parts[0], parts[1], parts[2], parts[3])


def _clamp_roi(x: int, y: int, w: int, h: int,
               iw: int, ih: int) -> Tuple[int, int, int, int]:
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be > 0")
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))
    return x, y, w, h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Channel selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _select_channel(meta: dict, frame: np.ndarray, ch_arg: str) -> int:
    if frame.ndim < 3:
        return 0
    nc = frame.shape[2]
    if ch_arg == "auto":
        cmap: Dict = {}
        try:
            if classify_channel_names is not None:
                cmap = classify_channel_names(meta or {}, nc)
        except Exception:
            pass
        if isinstance(cmap, dict):
            for colour in ("red", "green", "blue"):
                if colour in cmap:
                    return int(cmap[colour])
    else:
        if ch_arg.isdigit():
            idx = int(ch_arg)
            if 0 <= idx < nc:
                return idx
        if ch_arg in ("red", "green", "blue"):
            try:
                cmap = classify_channel_names(meta or {}, nc)  # type: ignore
                if isinstance(cmap, dict) and ch_arg in cmap:
                    return int(cmap[ch_arg])
            except Exception:
                pass
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRAP recovery-curve extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_curve(
    nd2_path: str,
    roi: Tuple[int, int, int, int],
    frap_channel: str,
    pre_frames: int,
    bleach_frame: Optional[int],
    recovery_mode: str,
    default_fps: float,
    debug_time: bool = False,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    int,
    float,
    Optional[float],
    Optional[float],
]:
    """Return (times, intensities, recovery, bleach_idx, fps, roi_side_um, roi_area_um2)."""
    Reader = None
    if nd2_utils is not None and getattr(nd2_utils, "_HAS_ND2READER", False):
        Reader = nd2_utils.ND2Reader  # type: ignore
    if Reader is None:
        from nd2reader import ND2Reader as Reader  # type: ignore

    with Reader(nd2_path) as imgs:
        imgs.iter_axes = "t"
        sizes = getattr(imgs, "sizes", {}) or {}
        imgs.bundle_axes = "yxc" if "c" in sizes else "yx"

        num_frames = sizes.get("t", len(imgs))
        meta = getattr(imgs, "metadata", {}) or {}

        fps = float(default_fps) if default_fps and default_fps > 0 else 10.0
        if detect_fps_from_metadata is not None:
            try:
                v = detect_fps_from_metadata(meta)
                if v and v > 0:
                    fps = float(v)
            except Exception:
                pass
        if debug_time:
            print(f"  [debug-time] default_fps={default_fps} initial fps={fps}")

        # Try to derive real timestamps from ND2 coords['t'] (preferred).
        # This avoids interpreting metadata keys incorrectly (some ND2 files store
        # seconds-per-frame under names like frame_rate).
        frame_times_s: Optional[List[float]] = None
        try:
            coords = getattr(imgs, "coords", None)
            if isinstance(coords, dict) and "t" in coords:
                tvals = coords.get("t")
                if tvals is not None and hasattr(tvals, "__len__"):
                    t_list = [float(x) for x in list(tvals)]
                    if len(t_list) == int(num_frames) and np.all(np.isfinite(t_list)):
                        # Heuristic: if coords look like frame indices (0..N-1), ignore them.
                        if num_frames >= 3:
                            diffs = np.diff(t_list)
                            looks_like_frames = (
                                all(abs(float(d) - 1.0) < 1e-6 for d in diffs) and
                                abs(t_list[0] - 0.0) < 1e-6 and
                                abs(t_list[-1] - (num_frames - 1)) < 1e-2
                            )
                        else:
                            looks_like_frames = False

                        if not looks_like_frames:
                            diffs = np.diff(t_list)
                            med_dt = float(np.median(diffs)) if len(diffs) > 0 else 0.0
                            scale = 1.0
                            if med_dt > 100:      # likely ms
                                scale = 1e-3
                            elif med_dt > 1e4:    # likely us
                                scale = 1e-6

                            t_scaled = [t * scale for t in t_list]
                            t0 = t_scaled[0] if t_scaled else 0.0
                            frame_times_s = [float(t - t0) for t in t_scaled]

                            # Update effective fps from derived seconds-per-frame.
                            if len(frame_times_s) >= 2:
                                diffs2 = np.diff(frame_times_s)
                                med_dt2 = float(np.median(diffs2))
                                if med_dt2 > 0:
                                    fps = 1.0 / med_dt2
                        if debug_time:
                            print(
                                f"  [debug-time] coords['t'] found: "
                                f"num_frames={num_frames}, "
                                f"t0={t_list[0] if t_list else None}, "
                                f"t1={t_list[1] if len(t_list)>1 else None}, "
                                f"looks_like_frames={looks_like_frames}"
                            )
                            if frame_times_s is not None and len(frame_times_s) >= 2:
                                diffs2 = np.diff(frame_times_s)
                                print(
                                    f"  [debug-time] derived dt~{float(np.median(diffs2))} s "
                                    f"-> fps={fps}"
                                )
        except Exception:
            frame_times_s = None
            if debug_time:
                print("  [debug-time] coords['t'] not usable; falling back to fps")

        f0 = np.asarray(imgs[0])
        ih, iw = f0.shape[:2]
        x, y, w, h = _clamp_roi(*roi, iw, ih)
        ci = _select_channel(meta, f0, frap_channel)

        # Physical size estimates from metadata (best-effort).
        roi_side_um = None
        roi_area_um2 = None
        try:
            if extract_side_length_um is not None:
                roi_side_um = extract_side_length_um(meta or {}, h, w)
        except Exception:
            roi_side_um = None
        try:
            if get_pixel_size_um is not None:
                px_um = get_pixel_size_um(meta or {})
                if px_um is not None:
                    roi_area_um2 = float(w) * float(h) * float(px_um) * float(px_um)
        except Exception:
            roi_area_um2 = None
        if roi_area_um2 is None and roi_side_um is not None and w == h:
            roi_area_um2 = float(roi_side_um) * float(roi_side_um)

        intensities: List[float] = []
        for i in range(num_frames):
            fr = nd2_to_mp4.to_native_endian(np.asarray(imgs[i]))
            patch = fr[y:y + h, x:x + w, ci] if fr.ndim == 3 else fr[y:y + h, x:x + w]
            intensities.append(float(np.mean(patch)))

        if frame_times_s is not None and len(frame_times_s) == int(num_frames):
            times = [float(t) for t in frame_times_s]
        else:
            times = [float(i) / fps if fps > 0 else float(i) for i in range(num_frames)]

        if debug_time:
            print(
                f"  [debug-time] time model: "
                f"{'coords' if frame_times_s is not None and len(frame_times_s)==int(num_frames) else 'i/fps'}, "
                f"fps={fps}, "
                f"times[:5]={[round(t,6) for t in times[:5]]}"
            )

    bi = int(bleach_frame) if bleach_frame is not None else int(np.argmin(intensities))
    bi = max(0, min(bi, len(intensities) - 1))

    ps = max(0, bi - pre_frames)
    pe = bi
    if pe > ps:
        i_pre = float(np.mean(intensities[ps:pe]))
    else:
        n = min(pre_frames, len(intensities))
        i_pre = float(np.mean(intensities[:n])) if n > 0 else float(intensities[0])
    i_bleach = float(intensities[bi])

    if recovery_mode == "raw":
        recovery = list(intensities)
    else:
        denom = i_pre - i_bleach
        if abs(denom) < 1e-12:
            denom = 1e-12
        recovery = [(v - i_bleach) / denom for v in intensities]

    return times, intensities, recovery, bi, fps, roi_side_um, roi_area_um2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Poster-frame extraction (first frame of the rendered MP4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_poster(mp4_path: str, out_png: str) -> bool:
    try:
        import cv2
        import shutil

        ext = os.path.splitext(mp4_path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png"):
            # Already a still image — copy/convert to poster PNG.
            img = cv2.imread(mp4_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                shutil.copy2(mp4_path, out_png)
                return True
            cv2.imwrite(out_png, img)
            return True

        cap = cv2.VideoCapture(mp4_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(out_png, frame)
            return True
    except Exception:
        pass
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSV export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _write_csv(times, intensities, recovery, path, time_unit):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"Time({time_unit})", "Intensity", "Recovery"])
        for t, i, r in zip(times, intensities, recovery):
            w.writerow([f"{t:.6g}", f"{i:.6g}", f"{r:.6g}"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data down-sampling (chart & table)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _downsample_indices(n: int, maxn: int) -> List[int]:
    if n <= maxn:
        return list(range(n))
    step = max(1, n // maxn)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRAP diffusion coefficient estimation (simple, best-effort)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _estimate_t_half_and_D(
    times_s: List[float],
    recovery: List[float],
    bleach_idx: int,
    roi_area_um2: Optional[float],
    roi_side_um: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate t1/2 and diffusion coefficient D from a normalized FRAP recovery curve.

    - recovery is expected to be normalized such that bleach ~ 0 and prebleach ~ 1.
    - We estimate the plateau as mean of the last 10% of frames.
    - t1/2 is the time when recovery reaches 0.5 * plateau.
    - Effective radius is estimated from ROI:
        r_eff = sqrt(area/pi) (preferred) else side/2 for square-ish ROI.
    - D is approximated by: D ≈ r_eff^2 / (4 * t1/2)
      (This is a common rough approximation; exact FRAP models depend on bleach profile.)
    Returns: (plateau, t_half, D_um2_per_s)
    """
    n = min(len(times_s), len(recovery))
    if n < 3:
        return None, None, None

    # Plateau estimate
    tail_n = max(3, int(0.1 * n))
    plateau = float(np.mean(recovery[-tail_n:]))
    if not np.isfinite(plateau) or plateau <= 0:
        return None, None, None

    target = 0.5 * plateau

    # Search forward from bleach index for first crossing
    start = max(0, min(int(bleach_idx), n - 2))
    t_half: Optional[float] = None
    for i in range(start, n - 1):
        r0 = float(recovery[i])
        r1 = float(recovery[i + 1])
        if (r0 <= target <= r1) or (r1 <= target <= r0):
            t0 = float(times_s[i])
            t1 = float(times_s[i + 1])
            if not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            if abs(r1 - r0) < 1e-12:
                t_half = t0
            else:
                alpha = (target - r0) / (r1 - r0)
                t_half = t0 + alpha * (t1 - t0)
            break

    if t_half is None or not np.isfinite(t_half) or t_half <= 0:
        return plateau, None, None

    # Effective radius (µm)
    r_eff_um: Optional[float] = None
    if roi_area_um2 is not None and roi_area_um2 > 0:
        r_eff_um = float(np.sqrt(float(roi_area_um2) / np.pi))
    elif roi_side_um is not None and roi_side_um > 0:
        r_eff_um = float(roi_side_um) / 2.0

    if r_eff_um is None or r_eff_um <= 0:
        return plateau, t_half, None

    D = (r_eff_um * r_eff_um) / (4.0 * t_half)
    if not np.isfinite(D) or D <= 0:
        return plateau, t_half, None

    return plateau, t_half, float(D)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PowerPoint slide helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _set_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR


def _add_text(slide, text, left, top, width, height,
              font_size=18, color=TITLE_COLOR, bold=False,
              alignment=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = FONT_NAME
    p.alignment = alignment
    return tb


def _add_multiline_text(slide, lines, left, top, width, height,
                        font_size=14, color=SUBTITLE_COLOR):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for idx, line in enumerate(lines):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = FONT_NAME
    return tb


def _add_chart(slide, times, recovery, bleach_idx,
               left, top, width, height, time_unit):
    """Insert an XY-scatter line chart (native, editable)."""
    ds_idx = _downsample_indices(len(times), MAX_CHART_POINTS)
    ts = [times[i] for i in ds_idx]
    rs = [recovery[i] for i in ds_idx]

    cd = XyChartData()
    sr = cd.add_series("Recovery")
    for x, y in zip(ts, rs):
        sr.add_data_point(x, y)

    # Bleach marker as a vertical line (separate series)
    if 0 <= bleach_idx < len(times):
        rng = max(rs) - min(rs) if rs else 1.0
        ylo = min(rs) - 0.05 * rng
        yhi = max(rs) + 0.05 * rng
        bt = times[bleach_idx]
        sb = cd.add_series("Bleach")
        sb.add_data_point(bt, ylo)
        sb.add_data_point(bt, yhi)

    gf = slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
        left, top, width, height, cd,
    )
    chart = gf.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(9)
    # Ensure legend text is black (requested).
    try:
        chart.legend.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    except Exception:
        pass

    plot = chart.plots[0]

    s0 = plot.series[0]
    s0.format.line.color.rgb = ACCENT_RED
    s0.format.line.width = Pt(2.0)
    s0.smooth = True

    if len(plot.series) > 1:
        s1 = plot.series[1]
        s1.format.line.color.rgb = ACCENT_BLUE
        s1.format.line.width = Pt(1.5)

    va = chart.value_axis
    va.has_title = True
    va.axis_title.text_frame.paragraphs[0].text = "Recovery"
    va.axis_title.text_frame.paragraphs[0].font.size = Pt(10)
    va.axis_title.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    va.tick_labels.font.size = Pt(9)
    try:
        va.tick_labels.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    except Exception:
        pass
    try:
        va.major_gridlines.format.line.color.rgb = RGBColor(0xD0, 0xD0, 0xD0)
    except Exception:
        pass

    try:
        ca = chart.category_axis
        ca.has_title = True
        ca.axis_title.text_frame.paragraphs[0].text = f"Time ({time_unit})"
        ca.axis_title.text_frame.paragraphs[0].font.size = Pt(10)
        ca.axis_title.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x00, 0x00, 0x00)
        ca.tick_labels.font.size = Pt(9)
        try:
            ca.tick_labels.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
        except Exception:
            pass
    except Exception:
        pass

    return gf


def _add_table(slide, times, intensities, recovery,
               left, top, width, time_unit, max_rows):
    """Insert a native PowerPoint table (editable)."""
    n = min(len(times), len(intensities), len(recovery))
    if n == 0:
        return None

    idx = _downsample_indices(n, max_rows)
    nrows = len(idx) + 1  # header + data
    ncols = 3
    row_h = Inches(0.24)
    tbl_h = row_h * nrows

    shape = slide.shapes.add_table(nrows, ncols, left, top, width, tbl_h)
    tbl = shape.table

    # Scale column widths to fit the requested table width.
    # (Avoid hardcoded inches that can overflow on 4:3 slides.)
    total_w = int(width)
    ratios = [0.36, 0.36, 0.28]  # Time, Intensity, Recovery
    col_emu = [int(total_w * r) for r in ratios]
    # Fix rounding so sum == total_w
    col_emu[-1] = max(1, total_w - col_emu[0] - col_emu[1])
    for ci, cw in enumerate(col_emu):
        tbl.columns[ci].width = Emu(cw)

    headers = [f"Time ({time_unit})", "Intensity", "Recovery"]
    for ci, ht in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = ht
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = TITLE_COLOR
        p.font.name = FONT_NAME
        p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = TBL_HDR_BG

    for ri, di in enumerate(idx):
        row_bg = TBL_ODD_BG if ri % 2 == 0 else TBL_EVEN_BG
        vals = [
            # Table should reflect real measured seconds when time_unit == "s".
            f"{times[di]:.3f}" if time_unit == "s" else f"{times[di]:.3g}",
            f"{intensities[di]:.1f}",
            f"{recovery[di]:.4f}",
        ]
        for ci, v in enumerate(vals):
            cell = tbl.cell(ri + 1, ci)
            cell.text = v
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)
            p.font.color.rgb = TBL_TEXT_COLOR
            p.font.name = FONT_NAME
            p.alignment = PP_ALIGN.RIGHT
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bg

    return shape


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Presentation builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_pptx(entries: List[Dict], out_path: str) -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    blank_layout = None
    for layout in prs.slide_layouts:
        if layout.name == "Blank":
            blank_layout = layout
            break
    if blank_layout is None:
        blank_layout = prs.slide_layouts[6]

    for e in entries:
        name       = e["name"]
        mp4_path   = e["mp4_path"]
        poster_png = e["poster_png"]
        times      = e["times"]
        table_times = e.get("table_times", times)
        recovery   = e["recovery"]
        intensities = e["intensities"]
        bleach_idx = e["bleach_idx"]
        time_unit  = e["time_unit"]
        table_time_unit = e.get("table_time_unit", time_unit)
        max_rows   = e["max_rows"]
        bleach_time = e["bleach_time"]
        max_rec    = e["max_rec"]
        fps = e.get("fps")
        roi_side_um = e.get("roi_side_um")
        roi_area_um2 = e.get("roi_area_um2")
        roi_side_px = e.get("roi_side_px")
        roi_area_px = e.get("roi_area_px")
        t_half_s = e.get("t_half_s")
        D_um2_s = e.get("D_um2_s")

        # Common geometry (4:3)
        margin_l = Inches(0.5)
        margin_r = Inches(0.5)
        margin_t = Inches(0.2)
        usable_w = SLIDE_W - margin_l - margin_r

        # ── Slide 1: Movie ──────────────────────────────────────
        s1 = prs.slides.add_slide(blank_layout)
        _set_bg(s1)

        _add_text(s1, name,
                  margin_l, margin_t, usable_w, Inches(0.7),
                  font_size=28, bold=True, alignment=PP_ALIGN.LEFT)

        # Movie placeholder: keep square (1:1) and shift to the left.
        movie_w, movie_h = Inches(5.1), Inches(5.1)
        movie_l = margin_l
        movie_t = Inches(1.2)

        if os.path.isfile(mp4_path):
            poster = poster_png if os.path.isfile(poster_png) else None
            ext = os.path.splitext(mp4_path)[1].lower()
            try:
                if ext in (".jpg", ".jpeg", ".png"):
                    s1.shapes.add_picture(mp4_path, movie_l, movie_t, movie_w, movie_h)
                else:
                    s1.shapes.add_movie(
                        mp4_path, movie_l, movie_t, movie_w, movie_h,
                        poster_frame_image=poster,
                        mime_type="video/mp4",
                    )
            except Exception:
                if poster and os.path.isfile(poster):
                    s1.shapes.add_picture(poster, movie_l, movie_t, movie_w, movie_h)

        # Right-side info box (requested fields).
        if roi_side_um is not None:
            side_txt = f"{roi_side_um:.3g} um"
        else:
            side_txt = f"{roi_side_px} px"

        if roi_area_um2 is not None:
            area_txt = f"{roi_area_um2:.3g} um^2"
        else:
            area_txt = f"{roi_area_px} px^2"

        if fps is not None and float(fps) > 0:
            # User expectation: "1 frame = X seconds".
            sec_per_frame = 1.0 / float(fps)
            fps_txt = f"{sec_per_frame:.3g} s/frame ({float(fps):.3g} fps)"
        else:
            fps_txt = "unknown"

        info_lines = [
            f"- 一辺の長さ: {side_txt}",
            f"- ROIの面積: {area_txt}",
            f"- フレームレート: {fps_txt}",
        ]
        _add_multiline_text(
            s1,
            info_lines,
            Inches(6.2),
            Inches(1.35),
            Inches(3.6),
            Inches(3.5),
            font_size=18,
            color=TITLE_COLOR,
        )

        # ── Slide 2: Chart + Table ──────────────────────────────
        s2 = prs.slides.add_slide(blank_layout)
        _set_bg(s2)

        _add_text(s2, f"{name}  (Recovery)",
                  margin_l, margin_t, usable_w, Inches(0.7),
                  font_size=24, bold=True, alignment=PP_ALIGN.LEFT)

        # 4:3 layout: chart on left, diffusion box + table on right (no overlaps).
        top_y = Inches(1.15)
        gap_x = Inches(0.25)
        right_w = Inches(3.25)
        chart_w = usable_w - right_w - gap_x
        chart_h = Inches(5.85)

        chart_left = margin_l
        right_left = margin_l + chart_w + gap_x

        _add_chart(
            s2,
            times,
            recovery,
            bleach_idx,
            chart_left,
            top_y,
            chart_w,
            chart_h,
            time_unit,
        )

        # Diffusion coefficient summary (place above table, not overlapping).
        if t_half_s is not None and D_um2_s is not None:
            d_lines = [f"- t1/2: {float(t_half_s):.3g} s", f"- D: {float(D_um2_s):.3g} um^2/s"]
        elif t_half_s is not None:
            d_lines = [f"- t1/2: {float(t_half_s):.3g} s", "- D: N/A"]
        else:
            d_lines = ["- t1/2: N/A", "- D: N/A"]

        d_box_h = Inches(0.65)
        _add_multiline_text(
            s2,
            d_lines,
            right_left,
            top_y,
            right_w,
            d_box_h,
            font_size=14,
            color=TITLE_COLOR,
        )

        table_top = top_y + d_box_h + Inches(0.10)
        _add_table(
            s2,
            table_times,
            intensities,
            recovery,
            right_left,
            table_top,
            right_w,
            table_time_unit,
            max_rows,
        )

    prs.save(out_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create PowerPoint (.pptx) from FRAP ND2 files.",
    )
    ap.add_argument("--input", default=os.getcwd(),
                    help="ND2 file or directory (default: cwd)")
    ap.add_argument("--output", default="",
                    help="Output .pptx path (default: auto)")

    # FRAP
    ap.add_argument("--roi", default="",
                    help="ROI as x,y,w,h pixels (e.g. '100,120,60,60')")
    ap.add_argument("--roi-fraction", type=float, default=0.2,
                    help="Central ROI fraction if --roi is omitted (default: 0.2)")
    ap.add_argument("--pre-frames", type=int, default=5,
                    help="Baseline frames before bleach (default: 5)")
    ap.add_argument("--bleach-frame", type=int, default=-1,
                    help="Bleach frame index; <0 = auto-detect (default: -1)")
    ap.add_argument("--recovery-mode", choices=["normalized", "raw"],
                    default="normalized",
                    help="Recovery curve mode (default: normalized)")
    ap.add_argument("--frap-channel", default="auto",
                    help="Channel: auto / red / green / blue / index (default: auto)")
    ap.add_argument("--time-unit", choices=["sec", "frame"], default="sec",
                    help="X-axis unit (default: sec)")

    # Movie
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Fallback FPS (default: 10)")
    ap.add_argument("--sec-per-frame", type=float, default=10.0,
                    help="Time axis model: seconds per frame (default: 10).")
    ap.add_argument("--clip-limit", type=float, default=3.0,
                    help="CLAHE clip limit (default: 3.0)")
    ap.add_argument("--tile-grid", type=int, default=8,
                    help="CLAHE tile grid (default: 8)")
    ap.add_argument("--no-clahe", action="store_true",
                    help="Disable CLAHE for movie")
    ap.add_argument("--codec", default="mp4v",
                    help="Fourcc codec (default: mp4v)")
    ap.add_argument("--no-ffmpeg", action="store_true",
                    help="Use OpenCV VideoWriter instead of FFmpeg")

    # Output
    ap.add_argument("--non-recursive", action="store_true",
                    help="Do not recurse into subdirectories")
    ap.add_argument("--save-mp4", action="store_true",
                    help="Save MP4 files alongside the .pptx")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip CSV export")
    ap.add_argument("--keep-temp", action="store_true",
                    help="Keep all temporary files")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--table-max-rows", type=int, default=20,
                    help="Max data rows in the table (default: 20)")

    args = ap.parse_args()

    nd2_paths = _find_nd2_files(args.input, args.non_recursive)
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    use_clahe = not args.no_clahe
    use_ffmpeg = not args.no_ffmpeg
    override_fps: Optional[float] = None
    if args.sec_per_frame is not None and args.sec_per_frame > 0:
        override_fps = 1.0 / float(args.sec_per_frame)

    # Output directory
    if os.path.isfile(os.path.abspath(args.input)):
        out_dir = os.path.dirname(os.path.abspath(args.input))
    else:
        out_dir = os.path.abspath(args.input)

    if args.output.strip():
        out_name = args.output.strip()
    else:
        folder = os.path.basename(out_dir.rstrip(os.sep)) or "FRAPPresentation"
        out_name = f"{folder}_FRAP.pptx"
    if not out_name.lower().endswith(".pptx"):
        out_name += ".pptx"
    out_pptx = os.path.join(out_dir, out_name)

    # Data output directory (for CSV / MP4)
    data_dir = os.path.splitext(out_pptx)[0] + "_data"

    time_unit = "s" if args.time_unit == "sec" else "frame"
    entries: List[Dict] = []
    temp_dirs: List[str] = []

    for nd2_path in nd2_paths:
        base = os.path.splitext(os.path.basename(nd2_path))[0]
        print(f"\nProcessing FRAP: {os.path.basename(nd2_path)}")

        tmp = tempfile.mkdtemp(prefix=f"frap_{base}_", dir=out_dir)
        temp_dirs.append(tmp)

        try:
            mp4_path = os.path.join(tmp, f"{base}.mp4")
            poster_png = os.path.join(tmp, f"{base}_poster.png")

            # 1) ND2 → MP4 (or JPEG if single-frame)
            print("  Converting ND2 → media ...")
            created = nd2_to_mp4.convert_nd2_to_mp4(
                nd2_path=nd2_path,
                out_path=mp4_path,
                default_fps=float(args.fps),
                use_clahe=use_clahe,
                clip_limit=float(args.clip_limit),
                tile_grid=int(args.tile_grid),
                preferred_codec=args.codec or None,
                use_ffmpeg=use_ffmpeg,
                green_only=False,
            )
            if created:
                mp4_path = created[0]

            # 2) Poster frame
            _extract_poster(mp4_path, poster_png)

            # 3) Default ROI (central rectangle) if not specified
            roi_x, roi_y, roi_w, roi_h = 0, 0, 10, 10
            if args.roi:
                roi_x, roi_y, roi_w, roi_h = _parse_roi(args.roi)
            else:
                Reader = None
                if nd2_utils and getattr(nd2_utils, "_HAS_ND2READER", False):
                    Reader = nd2_utils.ND2Reader  # type: ignore
                if Reader is None:
                    from nd2reader import ND2Reader as Reader  # type: ignore
                with Reader(nd2_path) as imgs:
                    imgs.iter_axes = "t"
                    sizes = getattr(imgs, "sizes", {}) or {}
                    imgs.bundle_axes = "yxc" if "c" in sizes else "yx"
                    f0 = np.asarray(imgs[0])
                    fh, fw = f0.shape[:2]
                    frac = float(args.roi_fraction)
                    roi_w = int(max(5, fw * frac))
                    roi_h = int(max(5, fh * frac))
                    roi_x = (fw - roi_w) // 2
                    roi_y = (fh - roi_h) // 2

            bleach_arg = None if args.bleach_frame < 0 else int(args.bleach_frame)

            # 4) Recovery curve
            print("  Extracting recovery curve ...")
            times, ints, rec, bi, fps, roi_side_um, roi_area_um2 = extract_curve(
                nd2_path,
                (roi_x, roi_y, roi_w, roi_h),
                args.frap_channel,
                int(args.pre_frames),
                bleach_arg,
                "raw" if args.recovery_mode == "raw" else "normalized",
                float(override_fps if override_fps is not None else args.fps),
                debug_time=bool(args.verbose),
            )

            if args.time_unit == "frame":
                times_plot = [float(i) for i in range(len(ints))]
            else:
                times_plot = times

            bt = times[bi] if bi < len(times) else float(bi)
            mr = max(rec) if rec else float("nan")

            # Diffusion coefficient estimation (use real seconds axis)
            plateau, t_half_s, D_um2_s = _estimate_t_half_and_D(
                times_s=times,
                recovery=rec,
                bleach_idx=bi,
                roi_area_um2=roi_area_um2,
                roi_side_um=roi_side_um,
            )

            # 5) CSV export
            if not args.no_csv:
                os.makedirs(data_dir, exist_ok=True)
                csv_path = os.path.join(data_dir, f"{base}_recovery.csv")
                _write_csv(times_plot, ints, rec, csv_path, time_unit)
                print(f"  CSV → {csv_path}")

            # 6) Optionally copy media to data dir
            if args.save_mp4:
                os.makedirs(data_dir, exist_ok=True)
                import shutil
                media_ext = os.path.splitext(mp4_path)[1] or ".mp4"
                dst = os.path.join(data_dir, f"{base}{media_ext}")
                shutil.copy2(mp4_path, dst)
                print(f"  Media → {dst}")

            entries.append({
                "name": base,
                "mp4_path": mp4_path,
                "poster_png": poster_png,
                "times": times_plot,
                "table_times": times,  # always real seconds
                "intensities": ints,
                "recovery": rec,
                "bleach_idx": bi,
                "time_unit": time_unit,
                "table_time_unit": "s",
                "max_rows": int(args.table_max_rows),
                "bleach_time": bt,
                "max_rec": mr,
                "fps": float(fps),
                "roi_side_um": roi_side_um,
                "roi_area_um2": roi_area_um2,
                "roi_side_px": int(max(roi_w, roi_h)),
                "roi_area_px": int(roi_w * roi_h),
                "t_half_s": t_half_s,
                "D_um2_s": D_um2_s,
            })

        except Exception as exc:
            print(f"  ERROR processing {os.path.basename(nd2_path)}: {exc}",
                  file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()

    if not entries:
        print("No slides created.")
        sys.exit(1)

    # Build presentation
    print(f"\nCreating PowerPoint ({len(entries)} file(s)) ...")
    build_pptx(entries, out_pptx)
    print(f"  Saved: {out_pptx}")

    # Cleanup temp dirs
    if not args.keep_temp:
        for td in temp_dirs:
            try:
                if os.path.isdir(td):
                    for f in os.listdir(td):
                        try:
                            os.remove(os.path.join(td, f))
                        except OSError:
                            pass
                    os.rmdir(td)
            except Exception:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
