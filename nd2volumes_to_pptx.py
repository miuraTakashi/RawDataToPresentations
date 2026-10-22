#!/usr/bin/env python3
"""
nd2volumes_to_pptx.py

Create a PowerPoint (.pptx) from ND2 volume (Z-stack) files.

PPTX counterpart of `nd2volumes_to_keynote.py`:
- white background, black text
- 4:3 slide size
- one PPTX per ND2 volume file (mirrors nd2_to_keynote's per-volume behavior)
- embed Z-stack movie and two cross-sections (XZ, YZ) in one slide
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Optional, Tuple

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
    add_movie,
    add_picture,
    TITLE_COLOR,
)

try:
    import nd2volumes_to_keynote
except Exception as exc:
    sys.stderr.write(f"ERROR: unable to import nd2volumes_to_keynote: {exc}\n")
    raise

try:
    import nd2_utils
except Exception:
    nd2_utils = None  # type: ignore


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


def _parse_channels_arg(channels: str) -> List[str]:
    if not channels.strip():
        return []
    return [c.strip().lower() for c in channels.split(",") if c.strip()]


def _estimate_volume_physical_dims(nd2_path: str) -> Tuple[int, int, int, Optional[float], Optional[float], Optional[float]]:
    """
    Returns:
      nx, ny, nz, x_um, y_um, z_um
    """
    from nd2reader import ND2Reader

    with ND2Reader(nd2_path) as images:
        sizes = getattr(images, "sizes", {}) or {}
        nx = int(sizes.get("x", 0) or 0)
        ny = int(sizes.get("y", 0) or 0)
        nz = int(sizes.get("z", 0) or 0)
        meta = getattr(images, "metadata", {}) or {}

        # pixel size (best-effort, µm)
        px_um = None
        if nd2_utils is not None and hasattr(nd2_utils, "get_pixel_size_um"):
            try:
                px_um = nd2_utils.get_pixel_size_um(meta)  # type: ignore
            except Exception:
                px_um = None
        # If missing sizes, infer from first frame
        if nx == 0 or ny == 0:
            try:
                sample = np.asarray(images[0])
                ny, nx = sample.shape[:2]
            except Exception:
                pass

        x_um = float(nx) * float(px_um) if px_um is not None and nx > 0 else None
        y_um = float(ny) * float(px_um) if px_um is not None and ny > 0 else None

        z_um = None
        z_coords = meta.get("z_coordinates") if isinstance(meta, dict) else None
        if nz > 0 and z_coords is not None and hasattr(z_coords, "__len__") and len(z_coords) > 1:
            try:
                z_step = float(z_coords[1] - z_coords[0])
                if abs(z_step) < 1e-6:
                    z_step = z_step * 1e6
                z_um = abs(float(nz) * z_step)
            except Exception:
                z_um = None

        if z_um is None and nz > 0 and px_um is not None:
            z_um = float(nz) * float(px_um)

        return nx, ny, nz, x_um, y_um, z_um


def _image_aspect_height_for_width(img_path: str, target_w_in: float) -> float:
    from PIL import Image

    with Image.open(img_path) as im:
        w_px, h_px = im.size
    if w_px <= 0:
        return target_w_in
    return float(target_w_in) * (h_px / w_px)


def build_pptx_from_one_volume(nd2_path: str, args, out_pptx_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(nd2_path))[0]
    base_dir = os.path.dirname(nd2_path)

    nx, ny, nz, x_um, y_um, z_um = _estimate_volume_physical_dims(nd2_path)
    use_clahe = not args.no_clahe

    temp_dir = os.path.join(base_dir, f"temp_{base_name}_pptx")
    os.makedirs(temp_dir, exist_ok=True)
    movie_path = os.path.join(temp_dir, f"{base_name}_zstack.mp4")
    xz_image_path = os.path.join(temp_dir, f"{base_name}_xz.jpg")
    yz_image_path = os.path.join(temp_dir, f"{base_name}_yz.jpg")

    selected_channels = _parse_channels_arg(args.channels)
    single_channel_mode = bool(args.single_channel)

    # Create Z-stack movie and cross sections
    if not nd2volumes_to_keynote.create_z_stack_movie(
        nd2_path=nd2_path,
        output_path=movie_path,
        fps=float(args.fps),
        use_clahe=use_clahe,
        clip_limit=float(args.clip_limit),
        tile_grid=int(args.tile_grid),
        verbose=bool(args.verbose),
        selected_channels=selected_channels if selected_channels else None,
        single_channel_mode=single_channel_mode,
    ):
        raise RuntimeError(f"Failed to create Z-stack movie: {nd2_path}")

    y_mid = ny // 2 if ny > 0 else 0
    if not nd2volumes_to_keynote.extract_xz_cross_section(
        nd2_path=nd2_path,
        y_index=y_mid,
        output_path=xz_image_path,
        use_clahe=use_clahe,
        clip_limit=float(args.clip_limit),
        tile_grid=int(args.tile_grid),
        selected_channels=selected_channels if selected_channels else None,
        single_channel_mode=single_channel_mode,
    ):
        raise RuntimeError(f"Failed to extract XZ cross-section: {nd2_path}")

    x_mid = nx // 2 if nx > 0 else 0
    if not nd2volumes_to_keynote.extract_yz_cross_section(
        nd2_path=nd2_path,
        x_index=x_mid,
        output_path=yz_image_path,
        use_clahe=use_clahe,
        clip_limit=float(args.clip_limit),
        tile_grid=int(args.tile_grid),
        selected_channels=selected_channels if selected_channels else None,
        single_channel_mode=single_channel_mode,
    ):
        raise RuntimeError(f"Failed to extract YZ cross-section: {nd2_path}")

    # Build pptx
    prs = new_presentation()
    blank_layout = find_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide)

    add_title(
        slide,
        base_name,
        left=__import__("pptx").util.Inches(0.5),
        top=__import__("pptx").util.Inches(0.2),
        width=__import__("pptx").util.Inches(9.0),
        height=__import__("pptx").util.Inches(0.7),
    )

    # Layout approximation to match Keynote scaling.
    # Keynote uses:
    #  base_x=30, base_y=180, max fit in 512x512 region
    TARGET_PX = 512.0
    TARGET_IN = 5.2
    factor = TARGET_IN / TARGET_PX

    total_width = float(nx) + 10.0 + float(nz)
    total_height = float(ny) + 10.0 + float(nz)
    if total_width <= 0 or total_height <= 0:
        total_width, total_height = 512.0, 512.0
    scale = min(TARGET_PX / total_width, TARGET_PX / total_height)

    base_x_px = 30.0
    base_y_px = 180.0

    # choose desired movie placement for base_x/base_y mapping
    desired_movie_left_in = 0.5
    desired_movie_top_in = 1.3
    offset_x_in = desired_movie_left_in - base_x_px * factor
    offset_y_in = desired_movie_top_in - base_y_px * factor

    def px_to_in(px: float) -> float:
        return px * factor

    movie_w_px = float(nx) * scale
    movie_h_px = float(ny) * scale
    movie_x_in = offset_x_in + px_to_in(base_x_px)
    movie_y_in = offset_y_in + px_to_in(base_y_px)
    movie_w_in = px_to_in(movie_w_px)
    movie_h_in = px_to_in(movie_h_px)

    yz_w_px = float(nz) * scale
    yz_h_px = float(ny) * scale
    yz_x_px = base_x_px + (float(nx) + 10.0) * scale
    yz_y_px = base_y_px
    yz_x_in = offset_x_in + px_to_in(yz_x_px)
    yz_y_in = offset_y_in + px_to_in(yz_y_px)
    yz_w_in = px_to_in(yz_w_px)
    yz_h_in = px_to_in(yz_h_px)

    xz_w_px = float(nx) * scale
    xz_h_px = float(nz) * scale
    xz_x_px = base_x_px
    xz_y_px = base_y_px + (float(ny) + 10.0) * scale
    xz_x_in = offset_x_in + px_to_in(xz_x_px)
    xz_y_in = offset_y_in + px_to_in(xz_y_px)
    xz_w_in = px_to_in(xz_w_px)
    xz_h_in = px_to_in(xz_h_px)

    # Insert movie + images
    add_movie(
        slide,
        mp4_path=movie_path,
        left=__import__("pptx").util.Inches(movie_x_in),
        top=__import__("pptx").util.Inches(movie_y_in),
        width=__import__("pptx").util.Inches(movie_w_in),
        height=__import__("pptx").util.Inches(movie_h_in),
    )

    add_picture(
        slide,
        yz_image_path,
        left=__import__("pptx").util.Inches(yz_x_in),
        top=__import__("pptx").util.Inches(yz_y_in),
        width=__import__("pptx").util.Inches(yz_w_in),
        height=__import__("pptx").util.Inches(yz_h_in),
    )

    add_picture(
        slide,
        xz_image_path,
        left=__import__("pptx").util.Inches(xz_x_in),
        top=__import__("pptx").util.Inches(xz_y_in),
        width=__import__("pptx").util.Inches(xz_w_in),
        height=__import__("pptx").util.Inches(xz_h_in),
    )

    # Volume info box (top-right-ish)
    lines: List[str] = ["Volume Information:"]
    if x_um is not None:
        lines.append(f"X: {x_um:.1f} µm ({nx} px)")
    else:
        lines.append(f"X: N/A ({nx} px)")
    if y_um is not None:
        lines.append(f"Y: {y_um:.1f} µm ({ny} px)")
    else:
        lines.append(f"Y: N/A ({ny} px)")
    if z_um is not None:
        lines.append(f"Z: {z_um:.1f} µm ({nz} slices)")
    else:
        lines.append(f"Z: N/A ({nz} slices)")

    add_multiline_text(
        slide,
        lines,
        left=__import__("pptx").util.Inches(6.2),
        top=__import__("pptx").util.Inches(1.3),
        width=__import__("pptx").util.Inches(3.5),
        height=__import__("pptx").util.Inches(2.3),
        font_size=14,
        color=TITLE_COLOR,
    )

    prs.save(out_pptx_path)
    print(f"Saved PPTX: {out_pptx_path}")

    if not args.keep_temp:
        # cleanup temp
        try:
            if os.path.isdir(temp_dir):
                for f in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except Exception:
                        pass
                os.rmdir(temp_dir)
        except Exception:
            pass

    return out_pptx_path


def build_pptx_from_volumes(nd2_paths: List[str], args) -> List[str]:
    outputs: List[str] = []
    for nd2_path in nd2_paths:
        base_name = os.path.splitext(os.path.basename(nd2_path))[0]
        out_name = args.output.strip() if args.output.strip() else f"{base_name}_Volume.pptx"
        if args.output.strip():
            # If user explicitly gave --output, write to same directory but named like base_name.
            # (Simple behavior; avoids multiple overwrites.)
            out_name = os.path.splitext(args.output.strip())[0] + f"_{base_name}.pptx"
        out_path = os.path.join(os.path.dirname(nd2_path), out_name)
        outputs.append(build_pptx_from_one_volume(nd2_path, args, out_path))
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Create PowerPoint (.pptx) from ND2 volume files (Z-stacks).")
    ap.add_argument("--input", type=str, default=os.getcwd(), help="Input ND2 file or directory")
    ap.add_argument("--theme", type=str, default="White", help="(unused) kept for compatibility")
    ap.add_argument("--output", type=str, default="", help="Output pptx name (default: auto)")
    ap.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    ap.add_argument("--verbose", action="store_true", help="Verbose")

    # Movie options
    ap.add_argument("--fps", type=float, default=10.0, help="FPS for Z-stack movie (default: 10)")
    ap.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit (default: 3.0)")
    ap.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (default: 8)")
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    ap.add_argument("--codec", type=str, default="avc1", help="(unused; kept for compatibility)")

    # Normalization options (env vars consumed by nd2_to_mp4-like functions)
    ap.add_argument("--norm-mode", choices=["percentile", "minmax"], default="percentile", help="Normalization mode")
    ap.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for histogram clipping")
    ap.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile for histogram clipping")

    # Channel selection
    ap.add_argument("--channels", type=str, default="", help="Specific channels: red,green,blue or channel indices (0,1,2)")
    ap.add_argument("--single-channel", action="store_true", help="Extract only single channel as grayscale (use with --channels)")

    ap.add_argument("--no-overwrite", action="store_true", help="Skip existing files")
    ap.add_argument("--no-ffmpeg", action="store_true", help="(unused; kept for compatibility)")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary MP4 and JPG files")

    args = ap.parse_args()

    nd2_paths = _find_nd2_files(args.input, non_recursive=bool(args.non_recursive))
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    # normalization env vars
    if args.norm_mode == "percentile":
        os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
        os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
    else:
        os.environ.pop("ND2_NORM_LOWER_PCT", None)
        os.environ.pop("ND2_NORM_UPPER_PCT", None)

    build_pptx_from_volumes(nd2_paths, args)


if __name__ == "__main__":
    main()

