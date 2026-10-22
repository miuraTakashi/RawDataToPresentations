#!/usr/bin/env python3
"""
nd2images_to_pptx.py

Create a PowerPoint (.pptx) from ND2 fluorescence images.

This is the PPTX-native counterpart of `nd2images_to_keynote.py`:
- white background, black text
- 4:3 slide size
- one slide per extracted RGB/JPG image
- uses `nd2images_to_keynote.extract_fluorescence_channels()` to generate JPGs
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from typing import List

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pptx_layout_utils import (
    new_presentation,
    find_blank_layout,
    set_slide_bg,
    add_title,
    add_multiline_text,
    add_picture,
    SLIDE_W,
    TITLE_COLOR,
)


try:
    import nd2images_to_keynote  # reuse extraction logic (no AppleScript execution)
except Exception as exc:
    sys.stderr.write(f"ERROR: unable to import nd2images_to_keynote: {exc}\n")
    raise


def _find_nd2_files(root_dir: str, non_recursive: bool) -> List[str]:
    root_dir = os.path.abspath(root_dir)
    if os.path.isfile(root_dir):
        return [root_dir]
    out: List[str] = []
    if non_recursive:
        for f in os.listdir(root_dir):
            if f.lower().endswith(".nd2") and not f.startswith((".", "._")):
                out.append(os.path.join(root_dir, f))
    else:
        for dirpath, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".nd2") and not f.startswith((".", "._")):
                    out.append(os.path.join(dirpath, f))
    out.sort()
    return out


def build_pptx_from_images(nd2_paths: List[str], args, out_pptx_path: str) -> str:
    # Create temp directory for JPG files
    dir_path = os.path.dirname(out_pptx_path)
    temp_dir = os.path.join(dir_path, "temp_fluorescence_images_pptx")
    os.makedirs(temp_dir, exist_ok=True)

    # Compute global min-max if needed (reused from keynote script)
    global_min_val = None
    global_max_val = None
    if args.norm_mode == "global":
        global_min_val, global_max_val = nd2images_to_keynote.compute_global_minmax(nd2_paths)

    # Configure flags used by extract_fluorescence_channels
    nd2images_to_keynote.extract_fluorescence_channels._all_channels_flag = args.all_channels  # type: ignore
    nd2images_to_keynote.extract_fluorescence_channels._debug_channels = args.debug_channels  # type: ignore

    # Channel selection arguments
    selected_channels = None
    if args.channels:
        selected_channels = [ch.strip().lower() for ch in args.channels.split(",") if ch.strip()]

    jpg_data: list[tuple[str, float, str]] = []

    print(f"\nProcessing ND2 images -> JPG: {len(nd2_paths)} file(s)")
    for nd2_path in nd2_paths:
        result = nd2images_to_keynote.extract_fluorescence_channels(
            nd2_path,
            temp_dir,
            selected_channels,
            args.single_channel,
            args.norm_mode,
            args.lower_pct,
            args.upper_pct,
            args.clahe,
            args.clip_limit,
            args.tile_grid,
            global_min_val,
            global_max_val,
        )
        if result and result[0]:
            jpg_data.append(result)

    if not jpg_data:
        raise RuntimeError("No JPG files were created from ND2 inputs.")

    # Build pptx
    prs = new_presentation()
    blank_layout = find_blank_layout(prs)

    # Layout (approximate to Keynote)
    image_left = __import__("pptx").util.Inches(1.2)
    image_top = __import__("pptx").util.Inches(1.2)
    image_w = __import__("pptx").util.Inches(5.2)
    image_h = __import__("pptx").util.Inches(5.2)

    for jpg_path, side_length_um, original_nd2_path in jpg_data:
        s = prs.slides.add_slide(blank_layout)
        set_slide_bg(s)

        rel_path = os.path.relpath(original_nd2_path, os.path.abspath(args.input))
        slide_title = os.path.splitext(rel_path)[0]
        if not slide_title:
            slide_title = os.path.splitext(os.path.basename(original_nd2_path))[0]

        if side_length_um > 0:
            scale_text = f"{side_length_um:.0f}µm/side"
        else:
            scale_text = "Scale unknown"

        add_title(
            s,
            slide_title,
            left=__import__("pptx").util.Inches(0.5),
            top=__import__("pptx").util.Inches(0.2),
            width=__import__("pptx").util.Inches(9.0),
            height=__import__("pptx").util.Inches(0.7),
        )

        add_multiline_text(
            s,
            [scale_text],
            left=__import__("pptx").util.Inches(0.5),
            top=__import__("pptx").util.Inches(1.0),
            width=__import__("pptx").util.Inches(3.0),
            height=__import__("pptx").util.Inches(0.5),
            font_size=16,
            color=TITLE_COLOR,
        )

        add_picture(s, jpg_path, image_left, image_top, width=image_w, height=image_h)

    prs.save(out_pptx_path)
    print(f"Saved PPTX: {out_pptx_path}")

    # Cleanup
    if not args.keep_jpgs:
        try:
            for jpg_path, _, _ in jpg_data:
                os.remove(jpg_path)
        except Exception:
            pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

    return out_pptx_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Create PowerPoint (.pptx) from ND2 fluorescence images.")
    ap.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: cwd)")
    ap.add_argument("--theme", type=str, default="White", help="(unused) kept for compatibility")
    ap.add_argument("--output", type=str, default="", help="Output .pptx file name (default: auto)")
    ap.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    ap.add_argument("--subdirs-only", action="store_true", help="[DEPRECATED] alias of recursive mode")
    ap.add_argument("--verbose", action="store_true", help="Verbose")
    ap.add_argument("--keep-jpgs", action="store_true", help="Keep generated JPG files after creating PPTX")

    ap.add_argument("--channels", type=str, default="", help="Specific channels to extract (comma-separated): red,green,blue or channel indices (0,1,2)")
    ap.add_argument("--single-channel", action="store_true", help="Extract only single channel as grayscale (use with --channels)")
    ap.add_argument("--all-channels", action="store_true", help="Force extraction of all available channels")
    ap.add_argument("--debug-channels", action="store_true", help="Save individual channel images for debugging")

    ap.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
    ap.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit (default: 3.0)")
    ap.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (default: 8)")
    ap.add_argument(
        "--norm-mode",
        choices=["none", "minmax", "percentile", "global"],
        default="global",
        help="Normalization mode (default: global)",
    )
    ap.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for histogram clipping")
    ap.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile for histogram clipping")

    args = ap.parse_args()

    dir_path = os.path.abspath(args.input)
    if not os.path.isdir(dir_path):
        print("Directory not found:", dir_path, file=sys.stderr)
        sys.exit(1)

    non_recursive = bool(args.non_recursive) and not bool(args.subdirs_only)
    nd2_paths = _find_nd2_files(dir_path, non_recursive=non_recursive)
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    if args.output.strip():
        out_pptx_path = args.output.strip()
    else:
        base = os.path.basename(dir_path.rstrip(os.sep)) or "FluorescencePresentation"
        out_pptx_path = os.path.join(dir_path, f"{base}.pptx")

    build_pptx_from_images(nd2_paths, args, out_pptx_path)


if __name__ == "__main__":
    main()

