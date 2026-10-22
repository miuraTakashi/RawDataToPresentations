#!/usr/bin/env python3
"""
nd2_to_pptx.py

Auto-detect ND2 content type (image/movie/volume) and generate PPTX files.

This script mirrors `nd2_to_keynote.py` behavior as closely as possible:
- Images: create one PPTX for all image ND2s
- Movies: create one PPTX for all movie ND2s
- Volumes: create one PPTX per volume ND2 file

PPTX layout is white background / black text / 4:3.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import nd2movies_to_pptx
import nd2images_to_pptx
import nd2volumes_to_pptx


def _find_nd2_files_recursively(root_dir: str, non_recursive: bool) -> List[str]:
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


def detect_nd2_content_type(nd2_path: str) -> Tuple[str, int, int, int, Dict]:
    """
    Returns:
      (content_type, num_t, num_z, num_c, meta)
    """
    try:
        from nd2reader import ND2Reader

        with ND2Reader(nd2_path) as images:
            meta = getattr(images, "metadata", {}) or {}
            sizes = getattr(images, "sizes", {}) or {}
            num_t = int(sizes.get("t", 1))
            num_z = int(sizes.get("z", 1))
            num_c = int(sizes.get("c", 1))

            if "t" in sizes:
                try:
                    images.iter_axes = "t"
                    total_frames = len(images)
                except Exception:
                    total_frames = num_t
            else:
                total_frames = num_t

            if num_t == 1 and num_z == 1 and total_frames > 1:
                z_coords = meta.get("z_coordinates", None) if isinstance(meta, dict) else None
                if z_coords is not None and hasattr(z_coords, "__len__") and len(z_coords) > 1:
                    num_z = len(z_coords)
                else:
                    num_t = total_frames

            if num_z > 1:
                return "volume", num_t, num_z, num_c, meta
            if num_t > 1:
                return "movie", num_t, num_z, num_c, meta
            return "image", num_t, num_z, num_c, meta
    except Exception:
        return "movie", 0, 0, 0, {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-detect ND2 content type and create PPTX (PPTX version of nd2_to_keynote).")
    ap.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: cwd)")
    ap.add_argument("--theme", type=str, default="White", help="(unused; compatibility)")
    ap.add_argument("--output", type=str, default="", help="Output override (best-effort)")
    ap.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    ap.add_argument("--verbose", action="store_true", help="Verbose")

    # Force modes
    ap.add_argument("--image-only", action="store_true", help="Force all ND2 files to be treated as images")
    ap.add_argument("--movie-only", action="store_true", help="Force all ND2 files to be treated as movies")
    ap.add_argument("--volume-only", action="store_true", help="Force all ND2 files to be treated as volumes")

    # Common selection args (images/volumes)
    ap.add_argument("--channels", type=str, default="", help="Specific channels for image/volume extraction (comma-separated)")
    ap.add_argument("--single-channel", action="store_true", help="Extract only single channel as grayscale (use with --channels)")
    ap.add_argument("--all-channels", action="store_true", help="Force extraction of all available channels (images)")
    ap.add_argument("--debug-channels", action="store_true", help="Debug channels (images)")

    # Images processing options
    ap.add_argument("--keep-jpgs", action="store_true", help="Keep generated JPGs for images mode")
    ap.add_argument("--clahe", action="store_true", help="Enable CLAHE for images/volumes cross-sections")
    ap.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit")
    ap.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid")
    ap.add_argument("--norm-mode", choices=["none", "minmax", "percentile", "global"], default="global", help="Images normalization mode")
    ap.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile")
    ap.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile")

    # Movies processing options
    ap.add_argument("--movie-fps", type=float, default=10.0, help="Movies fallback FPS")
    ap.add_argument("--movie-no-clahe", action="store_true", help="Disable CLAHE for movies")
    ap.add_argument("--movie-codec", type=str, default="mp4v", help="Movies preferred codec")
    ap.add_argument("--movie-norm-mode", choices=["percentile", "minmax"], default="percentile", help="Movies normalization mode")
    ap.add_argument("--movie-no-overwrite", action="store_true", help="Skip existing movies")
    ap.add_argument("--movie-no-ffmpeg", action="store_true", help="Disable FFmpeg for movies")
    ap.add_argument("--separate-channels", action="store_true", help="Movies: export separate channel MP4s")

    # Volumes processing options
    ap.add_argument("--volume-fps", type=float, default=10.0, help="Volumes Z-stack FPS")
    ap.add_argument("--volume-no-clahe", action="store_true", help="Disable CLAHE for volumes")
    ap.add_argument("--volume-norm-mode", choices=["percentile", "minmax"], default="percentile", help="Volumes normalization mode")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary files for volumes")

    args = ap.parse_args()

    dir_path = os.path.abspath(args.input)
    if not os.path.isdir(dir_path):
        print("Directory not found:", dir_path, file=sys.stderr)
        sys.exit(1)

    if sum([args.image_only, args.movie_only, args.volume_only]) > 1:
        print("Error: cannot specify more than one of --image-only/--movie-only/--volume-only", file=sys.stderr)
        sys.exit(1)

    nd2_paths = _find_nd2_files_recursively(dir_path, non_recursive=bool(args.non_recursive))
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    image_files: List[str] = []
    movie_files: List[str] = []
    volume_files: List[str] = []

    for nd2_path in nd2_paths:
        if args.image_only:
            content_type = "image"
        elif args.movie_only:
            content_type = "movie"
        elif args.volume_only:
            content_type = "volume"
        else:
            content_type, _, _, _, _ = detect_nd2_content_type(nd2_path)

        if content_type == "image":
            image_files.append(nd2_path)
        elif content_type == "movie":
            movie_files.append(nd2_path)
        else:
            volume_files.append(nd2_path)

    base_dir_name = os.path.basename(dir_path.rstrip(os.sep)) or "ND2Presentation"

    created: List[str] = []

    # Images
    if image_files:
        out_images = os.path.join(dir_path, f"{base_dir_name}_Images.pptx")
        print(f"\nProcessing {len(image_files)} image file(s) -> {out_images}")
        # map args for nd2images_to_pptx
        args_img = argparse.Namespace(**vars(args))
        args_img.input = dir_path
        args_img.output = out_images
        nd2images_to_pptx.build_pptx_from_images(image_files, args_img, out_images)
        created.append(out_images)

    # Movies
    if movie_files:
        out_movies = os.path.join(dir_path, f"{base_dir_name}_Movies.pptx")
        print(f"\nProcessing {len(movie_files)} movie file(s) -> {out_movies}")
        args_mov = argparse.Namespace(**vars(args))
        args_mov.input = dir_path
        args_mov.output = out_movies
        # map movie-specific flags expected by nd2movies_to_pptx
        args_mov.fps = float(args.movie_fps)
        args_mov.no_clahe = bool(args.movie_no_clahe)
        args_mov.clip_limit = float(args.clip_limit)
        args_mov.tile_grid = int(args.tile_grid)
        args_mov.codec = args.movie_codec
        args_mov.norm_mode = args.movie_norm_mode
        args_mov.lower_pct = args.lower_pct
        args_mov.upper_pct = args.upper_pct
        args_mov.no_overwrite = bool(args.movie_no_overwrite)
        args_mov.no_ffmpeg = bool(args.movie_no_ffmpeg)
        args_mov.separate_channels = bool(args.separate_channels)
        args_mov.non_recursive = bool(args.non_recursive)
        nd2movies_to_pptx.build_pptx_from_movies(movie_files, args_mov, out_movies)
        created.append(out_movies)

    # Volumes
    if volume_files:
        print(f"\nProcessing {len(volume_files)} volume file(s) ...")
        args_vol = argparse.Namespace(**vars(args))
        args_vol.channels = args.channels
        args_vol.single_channel = bool(args.single_channel)
        args_vol.fps = float(args.volume_fps)
        args_vol.no_clahe = bool(args.volume_no_clahe)
        args_vol.clip_limit = float(args.clip_limit)
        args_vol.tile_grid = int(args.tile_grid)
        args_vol.norm_mode = args.volume_norm_mode
        args_vol.lower_pct = args.lower_pct
        args_vol.upper_pct = args.upper_pct
        args_vol.no_overwrite = False
        args_vol.no_ffmpeg = False
        args_vol.keep_temp = bool(args.keep_temp)
        args_vol.non_recursive = bool(args.non_recursive)
        args_vol.output = ""
        outs = nd2volumes_to_pptx.build_pptx_from_volumes(volume_files, args_vol)
        created.extend(outs)

    print("\nCreated PPTX files:")
    for p in created:
        print(" -", p)


if __name__ == "__main__":
    main()

