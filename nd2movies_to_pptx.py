#!/usr/bin/env python3
"""
nd2movies_to_pptx.py

Create a PowerPoint (.pptx) presentation from ND2 movie files.

This script mirrors `nd2movies_to_keynote.py` behavior as much as possible:
- white background, black text
- 4:3 slide size
- one slide per MP4 (or per fluorescence channel when --separate-channels)
- embed MP4 movies and show simple metadata text (µm/side, frame rate, total frames)

Note:
PowerPoint layout is approximated to match the Keynote version (native pptx has no Keynote master slides).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

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
    SLIDE_W,
    SLIDE_H,
    TITLE_COLOR,
)


# Import common ND2 helpers (best-effort)
try:
    import nd2_utils
    from nd2_utils import detect_fps_from_metadata, extract_side_length_um, classify_channel_names
except Exception:
    nd2_utils = None  # type: ignore
    detect_fps_from_metadata = None  # type: ignore
    extract_side_length_um = None  # type: ignore
    classify_channel_names = None  # type: ignore


try:
    import nd2_to_mp4
except Exception:
    sys.stderr.write("ERROR: nd2_to_mp4.py not found beside this script.\n")
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


def _nd2reader_first_frame_shape(nd2_path: str) -> Tuple[int, int]:
    # returns (h, w)
    from nd2reader import ND2Reader
    with ND2Reader(nd2_path) as images:
        images.iter_axes = "t"
        if "c" in getattr(images, "sizes", {}):
            images.bundle_axes = "yxc"
        else:
            images.bundle_axes = "yx"
        arr = np.asarray(images[0])
        h, w = arr.shape[:2]
        return int(h), int(w)


def extract_movie_metadata(nd2_path: str, default_fps: float) -> Dict[str, str]:
    """
    Returns keys:
      - length_um: string like "123 µm/side" or "N/A"
      - frame_rate: string like "0.1 s/frame" / "10 fps"
      - total_frames: string like "12"
    """
    metadata: Dict[str, str] = {
        "length_um": "N/A",
        "frame_rate": "N/A",
        "total_frames": "N/A",
    }

    try:
        from nd2reader import ND2Reader

        with ND2Reader(nd2_path) as images:
            images.iter_axes = "t"
            sizes = getattr(images, "sizes", {}) or {}
            num_frames = int(sizes.get("t", len(images)))
            metadata["total_frames"] = str(num_frames)

            meta = getattr(images, "metadata", {}) or {}
            described = None
            if nd2_utils is not None and getattr(images, "timesteps", None) is not None:
                try:
                    described = nd2_utils.describe_frame_intervals_ms(list(images.timesteps))
                except (TypeError, ValueError):
                    described = None
            if described:
                metadata["frame_rate"] = described
            else:
                fps = None
                if detect_fps_from_metadata is not None:
                    try:
                        fps = detect_fps_from_metadata(meta)
                    except Exception:
                        fps = None
                if fps is None or fps <= 0:
                    fps = float(default_fps)
                sec_per_frame = 1.0 / float(fps) if fps > 0 else float("nan")
                metadata["frame_rate"] = f"{sec_per_frame:.3g} s/frame ({float(fps):.3g} fps)"

            # side length in µm
            try:
                first_frame = np.asarray(images[0])
                h, w = first_frame.shape[:2]
                if extract_side_length_um is not None:
                    side_um = extract_side_length_um(meta or {}, h, w)
                    if side_um is not None:
                        metadata["length_um"] = f"{side_um:.1f} µm/side"
            except Exception:
                pass
    except Exception:
        # leave as N/A
        pass

    return metadata


def _format_slide_title_from_mp4(mp4_abs_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(mp4_abs_path))[0]
    folder_name = os.path.basename(os.path.dirname(mp4_abs_path))
    return f"{folder_name}/{base_name}"


def _derive_channel_suffix_title(mp4_abs_path: str) -> str:
    base = os.path.splitext(os.path.basename(mp4_abs_path))[0]
    folder_name = os.path.basename(os.path.dirname(mp4_abs_path))

    if base.endswith("_red"):
        core = base[: -len("_red")]
        return f"{folder_name}/{core} (Red/Alexa568)"
    if base.endswith("_green"):
        core = base[: -len("_green")]
        return f"{folder_name}/{core} (Green/Alexa488)"
    if base.endswith("_blue"):
        core = base[: -len("_blue")]
        return f"{folder_name}/{core} (Blue/DAPI)"
    if base.endswith("_brightfield"):
        core = base[: -len("_brightfield")]
        return f"{folder_name}/{core} (明視野)"
    return _format_slide_title_from_mp4(mp4_abs_path)


def build_pptx_from_movies(nd2_paths: List[str], args, out_pptx_path: str) -> str:
    prs = new_presentation()
    blank_layout = find_blank_layout(prs)

    # posters for movie thumbnails
    poster_root = os.path.join(os.path.dirname(out_pptx_path), "pptx_movie_posters")
    os.makedirs(poster_root, exist_ok=True)

    # geometry (4:3)
    movie_layout = (
        __import__("pptx").util.Inches(0.5),  # left
        __import__("pptx").util.Inches(1.2),  # top
        __import__("pptx").util.Inches(4.9),  # width
        __import__("pptx").util.Inches(4.9),  # height
    )

    # conversion options mirror nd2movies_to_keynote/nd2_to_mp4
    use_clahe = not args.no_clahe
    use_ffmpeg = not args.no_ffmpeg
    preferred_codec = str(args.codec) if args.codec else None

    # normalization env (only percentile is explicitly supported in nd2_to_mp4 CLI)
    if args.norm_mode == "percentile":
        os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
        os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
    else:
        os.environ.pop("ND2_NORM_LOWER_PCT", None)
        os.environ.pop("ND2_NORM_UPPER_PCT", None)

    separate_channels_enabled = bool(args.separate_channels)

    for nd2_path in nd2_paths:
        base = os.path.splitext(os.path.basename(nd2_path))[0]
        print(f"\nProcessing ND2 movie: {os.path.basename(nd2_path)}")

        meta = extract_movie_metadata(nd2_path, default_fps=float(args.fps))

        # Determine output mp4 location (same folder as nd2)
        subdir = os.path.dirname(nd2_path)
        out_color_mp4 = os.path.join(subdir, f"{base}.mp4")

        created_mp4s: List[str] = []
        created_titles: List[str] = []

        use_separate = separate_channels_enabled

        if use_separate:
            # classify channels so convert_individual_channels_to_mp4 can generate per-channel mp4s
            ch_map: Dict[str, int] = {}
            try:
                from nd2reader import ND2Reader

                with ND2Reader(nd2_path) as images:
                    sample = np.asarray(images[0])
                    num_channels = sample.shape[2] if sample.ndim == 3 else 1
                    ch_map = classify_channel_names(getattr(images, "metadata", {}) or {}, num_channels)  # type: ignore
            except Exception:
                ch_map = {}

            if not ch_map:
                print("  ⚠️  No channel mapping. Falling back to combined color MP4.")
                use_separate = False
            else:
                try:
                    from nd2movies_to_keynote import convert_individual_channels_to_mp4

                    created_pairs = convert_individual_channels_to_mp4(
                        nd2_path=nd2_path,
                        base_output_path=out_color_mp4,
                        ch_map=ch_map,
                        default_fps=float(args.fps),
                        use_clahe=use_clahe,
                        clip_limit=float(args.clip_limit),
                        tile_grid=int(args.tile_grid),
                        preferred_codec=preferred_codec,
                        use_ffmpeg=use_ffmpeg,
                    )
                    for channel_name, channel_path in created_pairs:
                        if os.path.exists(channel_path):
                            created_mp4s.append(channel_path)
                            created_titles.append(_derive_channel_suffix_title(channel_path))
                except Exception as exc:
                    print(f"  ERROR separate-channels conversion failed: {exc}")
                    created_mp4s = []
                    created_titles = []

        if not use_separate:
            # combined conversion (MP4, or JPEG when ND2 has a single frame)
            existing = nd2_to_mp4.find_existing_media_output(out_color_mp4)
            if existing is not None and args.no_overwrite:
                created_mp4s = [existing]
            else:
                created_mp4s = nd2_to_mp4.convert_nd2_to_mp4(
                    nd2_path=nd2_path,
                    out_path=out_color_mp4,
                    default_fps=float(args.fps),
                    use_clahe=use_clahe,
                    clip_limit=float(args.clip_limit),
                    tile_grid=int(args.tile_grid),
                    preferred_codec=preferred_codec,
                    use_ffmpeg=use_ffmpeg,
                    green_only=False,
                )
            for mp4p in created_mp4s:
                created_titles.append(_derive_channel_suffix_title(mp4p))

        # brightfield already included in convert_nd2_to_mp4 return value when present
        if use_separate:
            brightfield_mp4 = os.path.join(subdir, f"{base}_brightfield.mp4")
            bf_existing = nd2_to_mp4.find_existing_media_output(brightfield_mp4)
            if bf_existing is not None:
                created_mp4s.append(bf_existing)
                created_titles.append(_derive_channel_suffix_title(bf_existing))

        # Add slides
        for mp4_abs, slide_title in zip(created_mp4s, created_titles):
            if not os.path.exists(mp4_abs):
                continue
            s = prs.slides.add_slide(blank_layout)
            set_slide_bg(s)
            poster_png = os.path.join(poster_root, f"{os.path.basename(mp4_abs)}.png")
            add_title(
                s,
                slide_title,
                left=__import__("pptx").util.Inches(0.5),
                top=__import__("pptx").util.Inches(0.2),
                width=__import__("pptx").util.Inches(9.0),
                height=__import__("pptx").util.Inches(0.7),
            )
            ext = os.path.splitext(mp4_abs)[1].lower()
            if ext in (".jpg", ".jpeg", ".png"):
                add_picture(
                    s,
                    mp4_abs,
                    left=movie_layout[0],
                    top=movie_layout[1],
                    width=movie_layout[2],
                    height=movie_layout[3],
                )
            else:
                add_movie(
                    s,
                    mp4_path=mp4_abs,
                    left=movie_layout[0],
                    top=movie_layout[1],
                    width=movie_layout[2],
                    height=movie_layout[3],
                    poster_png=poster_png,
                )

            spatial = meta.get("length_um", "N/A")
            if spatial != "N/A" and spatial.endswith(" µm/side"):
                spatial = spatial.replace(" µm/side", "µm/side")
            rate_text = str(meta.get("frame_rate", "N/A"))
            rate_lines = [part for part in rate_text.splitlines() if part] or ["N/A"]
            if len(rate_lines) == 1:
                rate_lines = [f"Frame Rate: {rate_lines[0]}"]
            else:
                rate_lines = ["Frame Rate:"] + [f"\t{part}" for part in rate_lines]
            lines = [
                spatial if spatial != "N/A" else "Scale unknown",
                *rate_lines,
                f"Total Frames: {meta.get('total_frames','N/A')}",
            ]
            add_multiline_text(
                s,
                lines,
                left=__import__("pptx").util.Inches(6.0),
                top=__import__("pptx").util.Inches(1.3),
                width=__import__("pptx").util.Inches(3.6),
                height=__import__("pptx").util.Inches(3.0),
                font_size=14,
                color=TITLE_COLOR,
            )

    prs.save(out_pptx_path)
    print(f"Saved PPTX: {out_pptx_path}")
    return out_pptx_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Create PowerPoint from ND2 movie files.")
    ap.add_argument("--input", type=str, default=os.path.abspath(os.getcwd()), help="ND2 file or root directory")
    ap.add_argument("--theme", type=str, default="White", help="(unused) kept for compatibility")
    ap.add_argument("--output", type=str, default="", help="Output .pptx path (default: auto)")
    ap.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")

    # converter options (mirrors nd2movies_to_keynote)
    ap.add_argument("--fps", type=float, default=10.0, help="Fallback FPS if not found in ND2 metadata (default: 10)")
    ap.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit (default: 3.0)")
    ap.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (N -> NxN) (default: 8)")
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    ap.add_argument("--codec", type=str, default="mp4v", help="Preferred codec for MP4 (default: mp4v)")
    ap.add_argument("--norm-mode", choices=["percentile", "minmax"], default="percentile", help="Normalization mode")
    ap.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for histogram clipping")
    ap.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile for histogram clipping")
    ap.add_argument("--no-overwrite", action="store_true", help="Skip existing MP4 files")
    ap.add_argument("--no-ffmpeg", action="store_true", help="Disable FFmpeg (use OpenCV VideoWriter)")
    ap.add_argument("--separate-channels", action="store_true",
                    help="Save each fluorescence channel (red/green/blue) as separate grayscale MP4s")

    args = ap.parse_args()

    nd2_paths = _find_nd2_files(args.input, non_recursive=args.non_recursive)
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    if args.output.strip():
        out_pptx_path = args.output.strip()
    else:
        base = os.path.basename(os.path.abspath(args.input).rstrip(os.sep)) or "Presentation"
        out_pptx_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), f"{base}_Movies.pptx")

    build_pptx_from_movies(nd2_paths, args, out_pptx_path)


if __name__ == "__main__":
    main()

