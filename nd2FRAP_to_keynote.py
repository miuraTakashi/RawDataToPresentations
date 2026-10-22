#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from FRAP ND2 files.

For each ND2 file:
- Convert ND2 time-series to an MP4 (using nd2_to_mp4.convert_nd2_to_mp4)
- Extract ROI recovery curve from the ND2 time-series
- Create a plot image (PNG) of recovery vs time
- Add a Keynote slide containing both:
  - the MP4 movie
  - the recovery curve plot image

Default ROI:
- If --roi is not provided, a central rectangle ROI is used.

Important:
- This script assumes the ND2 "t" axis corresponds to the FRAP time-series.
- If ND2 has multiple channels, the script auto-selects the channel by metadata
  (prefer red, then green, then blue, then channel 0).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add script directory to Python path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import nd2_utils
    from nd2_utils import detect_fps_from_metadata, extract_side_length_um, classify_channel_names
except ImportError:
    nd2_utils = None  # type: ignore
    detect_fps_from_metadata = None  # type: ignore
    extract_side_length_um = None  # type: ignore
    classify_channel_names = None  # type: ignore

try:
    import nd2_to_mp4
except Exception:
    sys.stderr.write("ERROR: Unable to import nd2_to_mp4. Ensure it exists beside this script.\n")
    raise


def _escape_applescript(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


def _find_nd2_files(input_path: str, non_recursive: bool) -> List[str]:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(input_path)

    nd2_paths: List[str] = []
    if non_recursive:
        for f in os.listdir(input_path):
            if f.lower().endswith(".nd2") and not f.startswith(".") and not f.startswith("._"):
                nd2_paths.append(os.path.join(input_path, f))
    else:
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(".nd2") and not f.startswith(".") and not f.startswith("._"):
                    nd2_paths.append(os.path.join(root, f))

    nd2_paths.sort()
    return nd2_paths


def _parse_roi(roi: str) -> Tuple[int, int, int, int]:
    # format: x,y,w,h
    parts = [p.strip() for p in roi.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be 'x,y,w,h'")
    x, y, w, h = [int(p) for p in parts]
    return x, y, w, h


def _clamp_roi(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be > 0")
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _select_channel_index(meta: dict, sample_frame: np.ndarray, frap_channel: str) -> int:
    """
    Choose a channel index for intensity measurement.
    """
    # sample_frame is either (H,W) or (H,W,C)
    if sample_frame.ndim == 2:
        return 0
    if sample_frame.ndim != 3:
        return 0
    num_channels = sample_frame.shape[2]

    if frap_channel == "auto":
        ch_map = {}
        try:
            if classify_channel_names is not None:
                ch_map = classify_channel_names(meta or {}, num_channels)  # type: ignore
        except Exception:
            ch_map = {}

        if isinstance(ch_map, dict):
            if "red" in ch_map:
                return int(ch_map["red"])
            if "green" in ch_map:
                return int(ch_map["green"])
            if "blue" in ch_map:
                return int(ch_map["blue"])
    else:
        c = frap_channel.lower()
        if c.isdigit():
            idx = int(c)
            if 0 <= idx < num_channels:
                return idx
        # if user gave a semantic name
        if frap_channel in ("red", "green", "blue"):
            try:
                ch_map = classify_channel_names(meta or {}, num_channels)  # type: ignore
                if isinstance(ch_map, dict) and frap_channel in ch_map:
                    return int(ch_map[frap_channel])
            except Exception:
                pass

    # fallback: first channel
    return 0


def _make_recovery_plot_png(
    times: List[float],
    intensities: List[float],
    recovery: List[float],
    bleach_idx: int,
    out_png_path: str,
    title: str,
    time_unit: str,
) -> None:
    """
    Create a simple line plot using PIL (no matplotlib dependency).
    """
    w, h = 620, 320
    margin_left, margin_right, margin_top, margin_bottom = 70, 20, 30, 55
    plot_w = w - margin_left - margin_right
    plot_h = h - margin_top - margin_bottom

    # Determine y-scale (use recovery range if normalized)
    y_vals = recovery if recovery else intensities
    y_min = float(min(y_vals)) if y_vals else 0.0
    y_max = float(max(y_vals)) if y_vals else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_to_px(i: int) -> int:
        if len(times) <= 1:
            return margin_left
        frac = i / (len(times) - 1)
        return margin_left + int(frac * plot_w)

    def y_to_px(y: float) -> int:
        frac = (y - y_min) / (y_max - y_min)
        # invert for image coordinates
        return margin_top + plot_h - int(frac * plot_h)

    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    try:
        font = ImageFont.truetype("Helvetica", 16)
    except Exception:
        font = None
    draw.text((margin_left, 8), title, fill=(0, 0, 0), font=font)

    # Axes
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(0, 0, 0), width=2)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(0, 0, 0), width=2)

    # Labels
    draw.text((margin_left - 5, margin_top + plot_h + 10), f"0", fill=(0, 0, 0), font=font)
    if times:
        t_end = times[-1]
        draw.text(
            (margin_left + plot_w - 10, margin_top + plot_h + 10),
            f"{t_end:.2g}",
            fill=(0, 0, 0),
            font=font,
        )
    draw.text((10, margin_top + 5), "Recovery", fill=(0, 0, 0), font=font)
    draw.text((margin_left + plot_w + 2, margin_top + plot_h + 5), f"({time_unit})", fill=(0, 0, 0), font=font)

    # Plot line
    if recovery:
        pts = [(x_to_px(i), y_to_px(recovery[i])) for i in range(len(recovery))]
        draw.line(pts, fill=(220, 30, 30), width=3)

    # Bleach vertical line
    if 0 <= bleach_idx < len(times):
        x_bleach = x_to_px(bleach_idx)
        draw.line(
            (x_bleach, margin_top, x_bleach, margin_top + plot_h),
            fill=(0, 120, 255),
            width=2,
        )
        # annotate
        if len(times) > 1:
            tb = times[bleach_idx]
            label = f"Bleach @ {tb:.2g}{time_unit}"
            draw.text((x_bleach + 5, margin_top + 5), label, fill=(0, 120, 255), font=font)

    img.save(out_png_path, "PNG")


def _format_recovery_table_lines(
    times: List[float],
    intensities: List[float],
    recovery: List[float],
    time_unit: str,
    table_max_rows: int,
) -> List[str]:
    """
    Build table-like lines for Keynote.

    Since AppleScript table editing is limited, we render a "table" as
    multi-line monospaced-like text:
      t, I, R
    and downsample to at most table_max_rows rows.
    """
    n = min(len(times), len(intensities), len(recovery))
    if n <= 0:
        return ["No data"]

    if table_max_rows <= 0:
        table_max_rows = 1

    # Downsample indices (always include first and last)
    if n <= table_max_rows:
        idxs = list(range(n))
    else:
        step = max(1, int(n / table_max_rows))
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)

    header = f"t({time_unit})    Intensity    Recovery"
    lines = [header]
    for i in idxs:
        t = times[i]
        it = intensities[i]
        rt = recovery[i]
        # Fixed-ish formatting for readability in a text box
        lines.append(f"{t:>7.3g}         {it:>10.4g}     {rt:>8.4g}")
    return lines


def _escape_applescript_multiline(lines: List[str]) -> str:
    """
    Create AppleScript expression that joins lines with "return".

    Example return value:
      "a" & return & "b" & return & "c"
    """
    def esc_one(s: str) -> str:
        return _escape_applescript(s)

    if not lines:
        return '""'
    expr = '"' + esc_one(lines[0]) + '"'
    for ln in lines[1:]:
        expr += ' & return & "' + esc_one(ln) + '"'
    return expr


def extract_recovery_curve_from_nd2(
    nd2_path: str,
    roi: Tuple[int, int, int, int],
    frap_channel: str,
    pre_frames: int,
    bleach_frame: Optional[int],
    recovery_mode: str,
) -> Tuple[List[float], List[float], List[float], int, float]:
    """
    Returns:
      times, intensities, recovery, bleach_idx, fps
    """
    # ND2 backends
    if nd2_utils is not None and getattr(nd2_utils, "_HAS_ND2READER", False):
        ND2Reader = nd2_utils.ND2Reader  # type: ignore
    else:
        from nd2reader import ND2Reader  # type: ignore

    with ND2Reader(nd2_path) as images:
        images.iter_axes = "t"
        if "c" in getattr(images, "sizes", {}):
            images.bundle_axes = "yxc"
        else:
            images.bundle_axes = "yx"

        sizes = getattr(images, "sizes", {}) or {}
        num_frames = sizes.get("t", len(images))
        meta = getattr(images, "metadata", {}) or {}

        fps = 10.0
        try:
            if detect_fps_from_metadata is not None:
                fps_candidate = detect_fps_from_metadata(meta)  # type: ignore
                if fps_candidate is not None and fps_candidate > 0:
                    fps = float(fps_candidate)
        except Exception:
            pass

        sample_frame = np.asarray(images[0])
        if sample_frame.ndim == 2:
            img_h, img_w = sample_frame.shape
        else:
            img_h, img_w = sample_frame.shape[:2]

        x, y, w, h = roi
        x, y, w, h = _clamp_roi(x, y, w, h, img_w=img_w, img_h=img_h)

        channel_idx = _select_channel_index(meta, sample_frame, frap_channel)

        intensities: List[float] = []
        times: List[float] = []

        # Extract mean intensity inside ROI for each frame
        for i in range(num_frames):
            frame = nd2_to_mp4.to_native_endian(np.asarray(images[i]))  # type: ignore
            if frame.ndim == 3:
                ch = frame[y : y + h, x : x + w, channel_idx]
            else:
                ch = frame[y : y + h, x : x + w]
            intensities.append(float(np.mean(ch)))
            times.append(i / fps if fps > 0 else float(i))

        # Bleach frame detection
        if bleach_frame is not None:
            bleach_idx = int(bleach_frame)
        else:
            # If FRAP starts with prebleach frames, bleach is typically the minimum intensity.
            bleach_idx = int(np.argmin(intensities))

        bleach_idx = max(0, min(bleach_idx, len(intensities) - 1))

        # Pre intensity
        pre_start = max(0, bleach_idx - int(pre_frames))
        pre_end = bleach_idx  # not inclusive
        if pre_end > pre_start:
            i_pre = float(np.mean(intensities[pre_start:pre_end]))
        else:
            # fallback: average first N frames
            n = min(int(pre_frames), len(intensities))
            i_pre = float(np.mean(intensities[:n])) if n > 0 else float(intensities[0])  # type: ignore

        i_bleach = float(intensities[bleach_idx])

        # Recovery normalization
        recovery: List[float] = []
        if recovery_mode == "raw":
            recovery = intensities[:]
        else:
            denom = (i_pre - i_bleach)
            if abs(denom) < 1e-12:
                denom = 1e-12
            # Common FRAP normalization: 0 at bleach, 1 at prebleach
            # I_norm = (I(t) - I_bleach) / (I_pre - I_bleach)
            recovery = [float((it - i_bleach) / denom) for it in intensities]

        return times, intensities, recovery, bleach_idx, fps


def build_applescript_with_curve(
    dir_path: str,
    slides: List[Dict[str, str]],
    theme_name: str,
    out_key_path: str,
) -> str:
    """
    slides items:
      {
        "title": str,
        "movie_path": str,
        "curve_path": str,
        "subtitle": str
      }
    """
    lines: List[str] = []
    lines.append('tell application "Keynote"')
    lines.append('    activate')

    lines.append('    try')
    lines.append('        set theDoc to make new document with properties {document theme:theme "ExperimentalDataR1"}')
    lines.append('    on error')
    lines.append('        try')
    lines.append(f'            set theDoc to make new document with properties {{document theme:theme "{theme_name}"}}')
    lines.append('        on error')
    lines.append('            set theDoc to make new document')
    lines.append('        end try')
    lines.append('    end try')
    lines.append('    delay 1.0')

    if len(slides) > 0:
        lines.append('    try')
        lines.append('        tell theDoc to delete slide 1')
        lines.append('    end try')

    for s in slides:
        slide_kind = s.get("kind", "movie")
        title_escaped = _escape_applescript(s["title"])
        subtitle_escaped = _escape_applescript(s.get("subtitle", ""))
        movie_posix = _escape_applescript(s.get("movie_path", ""))
        curve_posix = _escape_applescript(s.get("curve_path", ""))
        table_lines = s.get("table_lines", [])

        lines.append('    tell theDoc')
        lines.append('        set newSlide to make new slide')
        lines.append('        try')
        lines.append('            set base slide of newSlide to master slide "タイトル、箇条書き、画像Confocal" of theDoc')
        lines.append('        on error')
        lines.append('            try')
        lines.append('                set base slide of newSlide to master slide "Blank" of theDoc')
        lines.append('            on error')
        lines.append('                set base slide of newSlide to master slide "空白" of theDoc')
        lines.append('            end try')
        lines.append('        end try')
        lines.append('        set current slide of theDoc to newSlide')
        lines.append('        delay 0.2')
        lines.append('        tell newSlide')

        # Title
        lines.append('            try')
        lines.append('                if (count of text items) > 0 then')
        lines.append(f'                    set object text of text item 1 to "{title_escaped}"')
        lines.append('                else')
        lines.append(f'                    set titleShape to make new text item with properties {{object text:"{title_escaped}", position:{{50, 50}}}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append(f'                set titleShape to make new text item with properties {{object text:"{title_escaped}", position:{{50, 50}}}}')
        lines.append('            end try')

        # Subtitle/bullets
        lines.append('            try')
        lines.append('                if (count of text items) > 1 then')
        lines.append(f'                    set object text of text item 2 to "{subtitle_escaped}"')
        lines.append('                else')
        lines.append(f'                    set scaleShape to make new text item with properties {{object text:"{subtitle_escaped}", position:{{50, 100}}}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append(f'                set scaleShape to make new text item with properties {{object text:"{subtitle_escaped}", position:{{50, 100}}}}')
        lines.append('            end try')

        if slide_kind == "movie":
            # Insert movie (try placeholders)
            lines.append('            try')
            lines.append(f'                set movieFile to alias (POSIX file "{movie_posix}")')
            lines.append('                repeat with img in (get every image)')
            lines.append('                    try')
            lines.append('                        set file name of img to movieFile')
            lines.append('                        exit repeat')
            lines.append('                    on error')
            lines.append('                        try')
            lines.append('                            set file of img to movieFile')
            lines.append('                            exit repeat')
            lines.append('                        end try')
            lines.append('                    end try')
            lines.append('                end repeat')
            lines.append('                repeat with mov in (get every movie)')
            lines.append('                    try')
            lines.append('                        set file name of mov to movieFile')
            lines.append('                        exit repeat')
            lines.append('                    on error')
            lines.append('                        try')
            lines.append('                            set file of mov to movieFile')
            lines.append('                            exit repeat')
            lines.append('                        end try')
            lines.append('                    end try')
            lines.append('                end repeat')
            lines.append('            on error errMsg1')
            lines.append('                log "Movie insert failed: " & errMsg1')
            lines.append('            end try')
        else:
            # Analysis slide:
            # Left: recovery plot image
            lines.append('            try')
            lines.append(f'                set curveFile to alias (POSIX file "{curve_posix}")')
            lines.append('                -- Left side plot')
            lines.append('                set curveImg to make new image with properties {file:curveFile, position:{50, 170}, width:460, height:230}')
            lines.append('            on error errMsgPlot')
            lines.append('                log "Plot insert failed: " & errMsgPlot')
            lines.append('            end try')

            # Right side: a text-box "table" with measurement values
            if table_lines:
                table_expr = _escape_applescript_multiline(table_lines)
            else:
                table_expr = '""'
            lines.append('            try')
            lines.append(f'                set tableText to {table_expr}')
            lines.append('                set tableBox to make new text item with properties {object text:tableText, position:{530, 140}, width:420, height:260}')
            lines.append('            on error errMsgTable')
            lines.append('                log "Table insert failed: " & errMsgTable')
            lines.append('            end try')

        lines.append('        end tell')
        lines.append('    end tell')

    out_posix = _escape_applescript(out_key_path)
    lines.append(f'    save theDoc in (POSIX file "{out_posix}")')
    lines.append('end tell')
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Keynote from FRAP ND2 files.")
    parser.add_argument("--input", type=str, default=os.getcwd(), help="ND2 file or directory (default: cwd)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto)")

    # FRAP extraction options
    parser.add_argument("--roi", type=str, default="", help="ROI rectangle as x,y,w,h (pixels), e.g. '100,120,60,60'")
    parser.add_argument("--roi-fraction", type=float, default=0.2, help="If --roi not set, ROI size fraction (default: 0.2)")
    parser.add_argument("--pre-frames", type=int, default=5, help="Frames before bleach for baseline average (default: 5)")
    parser.add_argument("--bleach-frame", type=int, default=-1, help="Bleach frame index (0-based). If <0, auto-detect (default: -1)")
    parser.add_argument("--recovery-mode", choices=["normalized", "raw"], default="normalized", help="Recovery curve mode (default: normalized)")
    parser.add_argument("--frap-channel", type=str, default="auto", help="Channel for intensity: auto, red/green/blue, or index (default: auto)")
    parser.add_argument("--time-unit", type=str, choices=["sec", "frame"], default="sec", help="X-axis unit in plot (default: sec)")

    # Movie conversion options (mirrors nd2_to_mp4)
    parser.add_argument("--fps", type=float, default=10.0, help="Fallback FPS if not found in ND2 metadata (default: 10)")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit (default: 3.0)")
    parser.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (default: 8)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE for movie conversion")
    parser.add_argument("--codec", type=str, default="mp4v", help="Preferred fourcc codec (default: mp4v)")
    parser.add_argument("--no-ffmpeg", action="store_true", help="Disable FFmpeg and use OpenCV VideoWriter instead")

    # Keynote & housekeeping
    parser.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp files (mp4/png) for debugging")
    parser.add_argument("--verbose", action="store_true", help="Print extra logs")
    parser.add_argument("--table-max-rows", type=int, default=18, help="Max rows for recovery table text (default: 18)")

    args = parser.parse_args()

    nd2_paths = _find_nd2_files(args.input, non_recursive=args.non_recursive)
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    use_clahe = not args.no_clahe
    use_ffmpeg = not args.no_ffmpeg

    # Output keynote path
    if os.path.isfile(os.path.abspath(args.input)):
        out_base_dir = os.path.dirname(os.path.abspath(args.input))
    else:
        out_base_dir = os.path.abspath(args.input)

    if args.output.strip():
        out_name = args.output.strip()
    else:
        folder_name = os.path.basename(os.path.abspath(out_base_dir).rstrip(os.sep)) or "FRAPPresentation"
        out_name = f"{folder_name}_FRAP.key"
    if not out_name.lower().endswith(".key"):
        out_name += ".key"
    out_key_path = os.path.join(out_base_dir, out_name)

    slides: List[Dict[str, str]] = []
    temp_dirs_to_cleanup: List[str] = []

    for nd2_path in nd2_paths:
        base_name = os.path.splitext(os.path.basename(nd2_path))[0]
        print(f"\nProcessing FRAP: {os.path.basename(nd2_path)}")

        temp_dir = tempfile.mkdtemp(prefix=f"frap_{base_name}_", dir=out_base_dir)
        temp_dirs_to_cleanup.append(temp_dir)

        try:
            movie_path = os.path.join(temp_dir, f"{base_name}.mp4")
            curve_png_path = os.path.join(temp_dir, f"{base_name}_recovery.png")

            # 1) Convert ND2 -> MP4 (or JPEG if single-frame)
            print("  Converting ND2 -> media...")
            # nd2_to_mp4.convert_nd2_to_mp4 signature:
            # (nd2_path, out_path, default_fps, use_clahe, clip_limit, tile_grid, preferred_codec, use_ffmpeg, green_only)
            created = nd2_to_mp4.convert_nd2_to_mp4(
                nd2_path=nd2_path,
                out_path=movie_path,
                default_fps=float(args.fps),
                use_clahe=use_clahe,
                clip_limit=float(args.clip_limit),
                tile_grid=int(args.tile_grid),
                preferred_codec=str(args.codec) if args.codec else None,
                use_ffmpeg=use_ffmpeg,
                green_only=False,
            )
            if created:
                movie_path = created[0]

            # 2) Compute recovery curve
            # Determine ROI with one ND2 open if needed
            roi_x = 0
            roi_y = 0
            roi_w = 50
            roi_h = 50

            # Get image size for default ROI
            # We'll open ND2 once more for curve extraction anyway; create a best-effort ROI.
            # If --roi is provided, clamp later in extract_recovery_curve_from_nd2.
            if args.roi:
                roi_x, roi_y, roi_w, roi_h = _parse_roi(args.roi)
            else:
                # Use a placeholder; actual clamp happens after reading sample frame in extraction
                roi_x, roi_y, roi_w, roi_h = 0, 0, 10, 10

            bleach_frame = None if args.bleach_frame < 0 else int(args.bleach_frame)

            # Temporarily open to estimate default central ROI if user didn't specify --roi.
            # Keep this simple: load sample frame to size the central ROI.
            # Use nd2reader directly here.
            if not args.roi:
                ND2Reader = None
                if nd2_utils is not None and getattr(nd2_utils, "_HAS_ND2READER", False):
                    ND2Reader = nd2_utils.ND2Reader  # type: ignore
                else:
                    from nd2reader import ND2Reader  # type: ignore
                with ND2Reader(nd2_path) as images:
                    images.iter_axes = "t"
                    if "c" in getattr(images, "sizes", {}):
                        images.bundle_axes = "yxc"
                    else:
                        images.bundle_axes = "yx"
                    sample_frame = np.asarray(images[0])
                    if sample_frame.ndim == 2:
                        img_h, img_w = sample_frame.shape
                    else:
                        img_h, img_w = sample_frame.shape[:2]
                    frac = float(args.roi_fraction)
                    roi_w = int(max(5, img_w * frac))
                    roi_h = int(max(5, img_h * frac))
                    roi_x = (img_w - roi_w) // 2
                    roi_y = (img_h - roi_h) // 2

            print("  Extracting recovery curve from ROI...")
            times, intensities, recovery, bleach_idx, fps = extract_recovery_curve_from_nd2(
                nd2_path=nd2_path,
                roi=(roi_x, roi_y, roi_w, roi_h),
                frap_channel=args.frap_channel,
                pre_frames=int(args.pre_frames),
                bleach_frame=bleach_frame,
                recovery_mode="raw" if args.recovery_mode == "raw" else "normalized",
            )

            # 3) Make plot image
            time_unit = "s" if args.time_unit == "sec" else "frame"
            # If user wants frame unit, override x-axis values
            if args.time_unit == "frame":
                times_plot = list(map(float, range(len(intensities))))
            else:
                times_plot = times

            # Plot title
            plot_title = f"{base_name}"
            _make_recovery_plot_png(
                times=times_plot,
                intensities=intensities,
                recovery=recovery,
                bleach_idx=bleach_idx,
                out_png_path=curve_png_path,
                title=plot_title,
                time_unit=time_unit,
            )

            # Keynote will load these files by absolute path, so they must exist.
            if not os.path.exists(movie_path):
                raise RuntimeError(f"Movie file not created: {movie_path}")
            if not os.path.exists(curve_png_path):
                raise RuntimeError(f"Curve file not created: {curve_png_path}")

            # 4) Build analysis values for table/subtitle
            if times and bleach_idx < len(times):
                tb = times[bleach_idx]
            else:
                tb = bleach_idx
            max_rec = max(recovery) if recovery else float("nan")
            subtitle = f"Bleach: {tb:.3g} {time_unit}\nMax recovery: {max_rec:.3g}"

            # Table lines (downsampled)
            # Note: times_plot may be different from times if user asked frame unit.
            table_lines = _format_recovery_table_lines(
                times=times_plot,
                intensities=intensities,
                recovery=recovery,
                time_unit=time_unit,
                table_max_rows=int(args.table_max_rows),
            )

            # Slide 1: movie only
            slides.append(
                {
                    "kind": "movie",
                    "title": base_name,
                    "movie_path": movie_path,
                    "subtitle": subtitle,
                }
            )

            # Slide 2: plot + table
            slides.append(
                {
                    "kind": "analysis",
                    "title": f"{base_name} (Recovery)",
                    "movie_path": "",
                    "curve_path": curve_png_path,
                    "subtitle": "ROI recovery and measurements",
                    "table_lines": table_lines,
                }
            )

        finally:
            # Do not delete temp files here.
            # They are referenced by Keynote AppleScript and are needed after this loop.
            pass

    if not slides:
        print("No slides were created.")
        sys.exit(1)

    print("\nCreating Keynote presentation...")
    ascript = build_applescript_with_curve(
        dir_path=out_base_dir,
        slides=slides,
        theme_name=args.theme,
        out_key_path=out_key_path,
    )

    if args.verbose:
        print("\n--- AppleScript begin ---")
        print(ascript)
        print("--- AppleScript end ---\n")

    completed = subprocess.run(
        ["/usr/bin/osascript", "-s", "e", "-s", "o", "-"],
        input=ascript.encode("utf-8"),
        capture_output=True,
    )
    out = completed.stdout.decode("utf-8", errors="ignore") if completed.stdout else ""
    err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
    if out:
        print(out)
    if completed.returncode != 0:
        print("AppleScript failed. Details:\n" + err, file=sys.stderr)
        sys.exit(completed.returncode)

    print(f"✅ Saved Keynote: {out_key_path}")

    # Cleanup temp files only after Keynote has been created successfully.
    if not args.keep_temp:
        # Keynote may still be loading external media from disk briefly.
        time.sleep(2.0)
        for td in temp_dirs_to_cleanup:
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


if __name__ == "__main__":
    main()

