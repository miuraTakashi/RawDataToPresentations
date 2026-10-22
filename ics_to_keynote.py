#!/usr/bin/env python3
"""
Create Keynote (.key) presentations from ICS microscopy files.

This script mirrors nd2_to_keynote.py but reads ICS/IDS datasets through
bioio + bioio-bioformats.

Features:
- Automatic detection of image vs movie vs volume content
- Fluorescence channel extraction as RGB or grayscale images
- MP4 conversion for time series with Keynote presentation generation
- Volume visualization with XY/XZ/YZ views
- Recursive directory scanning for .ics files

Requirements:
- macOS with Keynote installed
- Python 3
- bioio, bioio-bioformats, opencv-python, numpy, Pillow
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

# Add script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import nd2_utils  # type: ignore
except Exception:
    sys.stderr.write("ERROR: Unable to import nd2_utils.py beside this script.\n")
    raise

nd2images_to_keynote = None  # type: ignore
nd2_to_mp4 = None  # type: ignore
nd2movies_to_keynote = None  # type: ignore
_IMAGE_BUILDER_IMPORT_ERROR: Optional[Exception] = None
_ND2_TO_MP4_IMPORT_ERROR: Optional[Exception] = None
_MOVIE_BUILDER_IMPORT_ERROR: Optional[Exception] = None

try:
    from bioio import BioImage  # type: ignore
    import bioio_bioformats  # type: ignore

    _HAS_BIOIO = True
except Exception:
    BioImage = None  # type: ignore
    bioio_bioformats = None  # type: ignore
    _HAS_BIOIO = False


@dataclass
class IcsDataset:
    path: str
    data_tczyx: np.ndarray
    metadata: Dict[str, Any]
    channel_names: List[str]
    pixel_size_x_um: Optional[float]
    pixel_size_y_um: Optional[float]
    pixel_size_z_um: Optional[float]
    source_time_interval_s: Optional[float]

    @property
    def num_t(self) -> int:
        return int(self.data_tczyx.shape[0])

    @property
    def num_c(self) -> int:
        return int(self.data_tczyx.shape[1])

    @property
    def num_z(self) -> int:
        return int(self.data_tczyx.shape[2])

    @property
    def height(self) -> int:
        return int(self.data_tczyx.shape[3])

    @property
    def width(self) -> int:
        return int(self.data_tczyx.shape[4])

    @property
    def content_type(self) -> str:
        if self.num_z > 1:
            return "volume"
        if self.num_t > 1:
            return "movie"
        return "image"

    @property
    def side_length_um(self) -> Optional[float]:
        try:
            if self.pixel_size_x_um and self.pixel_size_y_um:
                return max(self.width * self.pixel_size_x_um, self.height * self.pixel_size_y_um)
        except Exception:
            pass
        return None


def require_bioio() -> None:
    if not _HAS_BIOIO:
        raise ImportError(
            "ICS support requires bioio and bioio-bioformats. "
            "Install with: pip install bioio bioio-bioformats"
        )


def require_image_builder() -> None:
    global nd2images_to_keynote, _IMAGE_BUILDER_IMPORT_ERROR
    if nd2images_to_keynote is None:
        try:
            import nd2images_to_keynote as imported_module  # type: ignore

            nd2images_to_keynote = imported_module  # type: ignore
            _IMAGE_BUILDER_IMPORT_ERROR = None
        except Exception as exc:
            _IMAGE_BUILDER_IMPORT_ERROR = exc
            raise ImportError(
                "Image Keynote support could not be imported. "
                f"Underlying error: {exc}"
            ) from exc


def require_movie_modules() -> None:
    global nd2_to_mp4, nd2movies_to_keynote, _ND2_TO_MP4_IMPORT_ERROR, _MOVIE_BUILDER_IMPORT_ERROR
    if nd2_to_mp4 is None:
        try:
            import nd2_to_mp4 as imported_converter  # type: ignore

            nd2_to_mp4 = imported_converter  # type: ignore
            _ND2_TO_MP4_IMPORT_ERROR = None
        except Exception as exc:
            _ND2_TO_MP4_IMPORT_ERROR = exc
            raise ImportError(
                "Movie and volume processing requires nd2_to_mp4.py and opencv-python. "
                f"Underlying error: {exc}"
            ) from exc
    if nd2movies_to_keynote is None:
        try:
            import nd2movies_to_keynote as imported_builder  # type: ignore

            nd2movies_to_keynote = imported_builder  # type: ignore
            _MOVIE_BUILDER_IMPORT_ERROR = None
        except Exception as exc:
            _MOVIE_BUILDER_IMPORT_ERROR = exc
            raise ImportError(
                "Movie Keynote builder could not be imported. "
                f"Underlying error: {exc}"
            ) from exc


def build_comment_tag(source_path: str, side_length_um: Optional[float], fps: float, note: Optional[str] = None) -> str:
    parts = [f"Source: {source_path}"]
    if side_length_um and side_length_um > 0:
        parts.append(f"Side: {side_length_um:.1f}µm")
    else:
        parts.append("Side: unknown")
    parts.append(f"FPS: {fps:.3f}")
    if note:
        parts.append(note)
    return " | ".join(parts)


def find_ics_files_recursively(root_dir: str) -> List[str]:
    ics_paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.startswith(".") or file_name.startswith("._"):
                continue
            if file_name.lower().endswith(".ics"):
                ics_paths.append(os.path.join(root, file_name))
    return sorted(ics_paths)


def _coerce_value_to_um(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "value") and not isinstance(value, (int, float)):
        try:
            value = value.value
        except Exception:
            pass
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric <= 0:
        return None
    if numeric < 1e-6:
        return numeric * 1e6
    return numeric


def _convert_length_to_um(value: Any, unit: str) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric <= 0:
        return None

    unit_l = unit.strip().lower().replace("µ", "u")
    if unit_l in {"m", "meter", "meters"}:
        return numeric * 1e6
    if unit_l in {"mm", "millimeter", "millimeters"}:
        return numeric * 1e3
    if unit_l in {"um", "micrometer", "micrometers"}:
        return numeric
    if unit_l in {"nm", "nanometer", "nanometers"}:
        return numeric / 1e3
    return _coerce_value_to_um(numeric)


def _convert_time_to_seconds(value: Any, unit: str) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric <= 0:
        return None

    unit_l = unit.strip().lower().replace("µ", "u")
    if unit_l in {"s", "sec", "secs", "second", "seconds"}:
        return numeric
    if unit_l in {"ms", "millisecond", "milliseconds"}:
        return numeric / 1e3
    if unit_l in {"us", "microsecond", "microseconds"}:
        return numeric / 1e6
    if unit_l in {"min", "mins", "minute", "minutes"}:
        return numeric * 60.0
    if unit_l in {"h", "hr", "hrs", "hour", "hours"}:
        return numeric * 3600.0
    return None


def parse_ics_header_metadata(ics_path: str) -> Dict[str, Any]:
    order: List[str] = []
    sizes: List[str] = []
    scales: List[str] = []
    units: List[str] = []
    channel_names: List[str] = []
    significant_bits: Optional[int] = None
    lower_limit: Optional[float] = None
    higher_limit: Optional[float] = None

    try:
        with open(ics_path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                fields = raw_line.rstrip("\r\n").split("\t")
                if len(fields) < 3:
                    continue
                group = fields[0].strip().lower()
                key = fields[1].strip().lower()
                values = [field.strip() for field in fields[2:]]

                if group == "layout" and key == "order":
                    order = values
                elif group == "layout" and key == "sizes":
                    sizes = values
                elif group == "layout" and key == "significant_bits" and values:
                    try:
                        significant_bits = int(float(values[0]))
                    except Exception:
                        pass
                elif group == "parameter" and key == "scale":
                    scales = values
                elif group == "parameter" and key == "units":
                    units = values
                elif group == "parameter" and key == "ch":
                    channel_names = values
                elif group == "parameter" and key == "lower_limit" and values:
                    try:
                        lower_limit = float(values[0])
                    except Exception:
                        pass
                elif group == "parameter" and key == "higher_limit" and values:
                    try:
                        higher_limit = float(values[0])
                    except Exception:
                        pass
    except OSError:
        return {}

    axis_info: Dict[str, Dict[str, Any]] = {}
    for idx, axis_name in enumerate(order):
        axis_key = axis_name.strip().lower()
        axis_info[axis_key] = {
            "size": sizes[idx].strip() if idx < len(sizes) else "",
            "scale": scales[idx].strip() if idx < len(scales) else "",
            "unit": units[idx].strip() if idx < len(units) else "",
        }

    x_um = _convert_length_to_um(axis_info.get("x", {}).get("scale"), axis_info.get("x", {}).get("unit", ""))
    y_um = _convert_length_to_um(axis_info.get("y", {}).get("scale"), axis_info.get("y", {}).get("unit", ""))
    z_um = _convert_length_to_um(axis_info.get("z", {}).get("scale"), axis_info.get("z", {}).get("unit", ""))
    t_s = _convert_time_to_seconds(axis_info.get("t", {}).get("scale"), axis_info.get("t", {}).get("unit", ""))

    return {
        "axis_info": axis_info,
        "channel_names": channel_names,
        "pixel_size_x_um": x_um,
        "pixel_size_y_um": y_um,
        "pixel_size_z_um": z_um,
        "time_interval_s": t_s,
        "significant_bits": significant_bits,
        "lower_limit": lower_limit,
        "higher_limit": higher_limit,
    }


def _coerce_to_tczyx(arr: np.ndarray, order: str) -> np.ndarray:
    target_order = "TCZYX"
    order_list = [dim.upper() for dim in order]
    out = np.asarray(arr)

    for axis in range(len(order_list) - 1, -1, -1):
        if order_list[axis] not in target_order:
            out = np.take(out, 0, axis=axis)
            order_list.pop(axis)

    for dim in target_order:
        if dim not in order_list:
            out = np.expand_dims(out, axis=out.ndim)
            order_list.append(dim)

    perm = [order_list.index(dim) for dim in target_order]
    return np.transpose(out, perm)


def _load_current_scene_tczyx(img: Any, source_path: str) -> np.ndarray:
    try:
        data_tczyx = np.asarray(img.get_image_data("TCZYX"))
    except Exception:
        raw_data = np.asarray(img.data)
        dims = getattr(getattr(img, "dims", None), "order", "TCZYX")
        data_tczyx = _coerce_to_tczyx(raw_data, dims)

    data_tczyx = np.asarray(data_tczyx)
    if data_tczyx.ndim != 5:
        raise RuntimeError(
            f"Unexpected data rank for {source_path}: expected 5D TCZYX-compatible array, got shape {data_tczyx.shape}"
        )
    return data_tczyx


def _load_scene_aware_tczyx(img: Any, source_path: str, scene_ids: Sequence[Any]) -> Tuple[np.ndarray, List[str]]:
    if not scene_ids:
        return _load_current_scene_tczyx(img, source_path), []

    merged_scene_names: List[str] = []
    merged_data: Optional[np.ndarray] = None

    for scene_id in scene_ids:
        try:
            img.set_scene(scene_id)
        except Exception:
            pass

        scene_data = _load_current_scene_tczyx(img, source_path)
        scene_name = str(scene_id)

        if merged_data is None:
            merged_data = scene_data
            merged_scene_names.append(scene_name)
            continue

        # Bio-Formats sometimes exposes time points as separate scenes.
        # If the non-time dimensions match, concatenate scenes along T.
        if scene_data.shape[1:] == merged_data.shape[1:]:
            merged_data = np.concatenate([merged_data, scene_data], axis=0)
            merged_scene_names.append(scene_name)
        else:
            print(
                f"Warning: Scene {scene_name} in {os.path.basename(source_path)} "
                "has incompatible dimensions and will be skipped."
            )

    if merged_data is None:
        raise RuntimeError(f"Could not load any scene data from {source_path}")

    try:
        img.set_scene(scene_ids[0])
    except Exception:
        pass

    return merged_data, merged_scene_names


def load_ics_dataset(ics_path: str) -> IcsDataset:
    require_bioio()

    img = BioImage(ics_path, reader=bioio_bioformats.Reader)

    try:
        scene_ids = list(getattr(img, "scenes", []) or [])
    except Exception:
        scene_ids = []

    data_tczyx, merged_scene_ids = _load_scene_aware_tczyx(img, ics_path, scene_ids)

    metadata = nd2_utils.metadata_to_dict(getattr(img, "metadata", {}) or {})
    if merged_scene_ids:
        metadata["scene_ids"] = merged_scene_ids
        metadata["num_source_scenes"] = len(merged_scene_ids)
        metadata["scenes_merged_into_time"] = len(merged_scene_ids) > 1

    header_metadata = parse_ics_header_metadata(ics_path)
    if header_metadata:
        metadata["ics_header"] = header_metadata.get("axis_info", {})
        if header_metadata.get("significant_bits") is not None:
            metadata["significant_bits"] = header_metadata.get("significant_bits")
        if header_metadata.get("lower_limit") is not None:
            metadata["lower_limit"] = header_metadata.get("lower_limit")
        if header_metadata.get("higher_limit") is not None:
            metadata["higher_limit"] = header_metadata.get("higher_limit")

    try:
        channel_names = list(getattr(img, "channel_names", []) or [])
    except Exception:
        channel_names = []
    header_channel_names = header_metadata.get("channel_names", [])
    if header_channel_names and len(header_channel_names) == int(data_tczyx.shape[1]):
        channel_names = [str(name) for name in header_channel_names]
    if not channel_names:
        channel_names = [f"Channel {idx}" for idx in range(int(data_tczyx.shape[1]))]

    metadata["channels"] = [{"label": name} for name in channel_names]

    pps = getattr(img, "physical_pixel_sizes", None)
    pixel_size_x_um = header_metadata.get("pixel_size_x_um") or _coerce_value_to_um(getattr(pps, "X", None) if pps is not None else None)
    pixel_size_y_um = header_metadata.get("pixel_size_y_um") or _coerce_value_to_um(getattr(pps, "Y", None) if pps is not None else None)
    pixel_size_z_um = header_metadata.get("pixel_size_z_um") or _coerce_value_to_um(getattr(pps, "Z", None) if pps is not None else None)
    source_time_interval_s = header_metadata.get("time_interval_s")
    metadata["pixel_size_x"] = pixel_size_x_um
    metadata["pixel_size_y"] = pixel_size_y_um
    metadata["pixel_size_z"] = pixel_size_z_um
    metadata["source_time_interval_s"] = source_time_interval_s

    return IcsDataset(
        path=ics_path,
        data_tczyx=data_tczyx,
        metadata=metadata,
        channel_names=channel_names,
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        pixel_size_z_um=pixel_size_z_um,
        source_time_interval_s=source_time_interval_s,
    )


def detect_ics_content_type(ics_path: str) -> Tuple[str, int, int, int, Dict[str, Any]]:
    try:
        dataset = load_ics_dataset(ics_path)
        return (
            dataset.content_type,
            dataset.num_t,
            dataset.num_z,
            dataset.num_c,
            dataset.metadata,
        )
    except Exception as exc:
        print(f"Warning: Could not analyze {ics_path}: {exc}")
        return "movie", 0, 0, 0, {}


def categorize_ics_files(ics_paths: Sequence[str], force_mode: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    image_files: List[str] = []
    movie_files: List[str] = []
    volume_files: List[str] = []

    for ics_path in ics_paths:
        if force_mode == "image":
            image_files.append(ics_path)
            continue
        if force_mode == "movie":
            movie_files.append(ics_path)
            continue
        if force_mode == "volume":
            volume_files.append(ics_path)
            continue

        content_type, num_t, num_z, num_c, _ = detect_ics_content_type(ics_path)
        rel_name = os.path.basename(ics_path)
        if content_type == "volume":
            print(f"  {rel_name}: volume (Z={num_z}, T={num_t}, C={num_c})")
            volume_files.append(ics_path)
        elif content_type == "movie":
            print(f"  {rel_name}: movie (T={num_t}, C={num_c})")
            movie_files.append(ics_path)
        else:
            print(f"  {rel_name}: image (C={num_c})")
            image_files.append(ics_path)

    return image_files, movie_files, volume_files


def _relative_stem(root_dir: str, file_path: str) -> str:
    rel_path = os.path.relpath(file_path, root_dir)
    rel_stem = os.path.splitext(rel_path)[0]
    return rel_stem.replace(os.sep, "__")


def _channels_argument(args: argparse.Namespace) -> Optional[List[str]]:
    if not args.channels:
        return None
    return [token.strip().lower() for token in args.channels.split(",") if token.strip()]


def build_channel_metadata(channel_names: Sequence[str]) -> Dict[str, Any]:
    return {"channels": [{"label": name} for name in channel_names]}


def complete_channel_map(channel_names: Sequence[str], num_channels: int) -> Dict[str, int]:
    meta = build_channel_metadata(channel_names)
    ch_map = nd2_utils.classify_channel_names(meta, num_channels)

    if num_channels <= 0:
        return ch_map

    if num_channels == 1:
        ch_map.setdefault("green", 0)
        return ch_map

    if num_channels == 2:
        if "green" not in ch_map:
            ch_map["green"] = 0
        if "red" not in ch_map:
            ch_map["red"] = 1 if ch_map.get("green") != 1 else 0
        return ch_map

    for color, idx in (("blue", 0), ("green", 1), ("red", 2)):
        if idx < num_channels and color not in ch_map:
            ch_map[color] = idx

    return ch_map


def resolve_channel_index(token: str, channel_names: Sequence[str], ch_map: Dict[str, int], num_channels: int) -> Optional[int]:
    if token.isdigit():
        idx = int(token)
        return idx if 0 <= idx < num_channels else None

    if token in ch_map:
        idx = ch_map[token]
        return idx if 0 <= idx < num_channels else None

    token_l = token.lower()
    for idx, name in enumerate(channel_names):
        name_l = str(name).lower()
        if token_l == name_l or token_l in name_l:
            return idx
    return None


def select_gray_index(
    selected_channels: Optional[Sequence[str]],
    channel_names: Sequence[str],
    ch_map: Dict[str, int],
    num_channels: int,
) -> int:
    if selected_channels:
        for token in selected_channels:
            idx = resolve_channel_index(token, channel_names, ch_map, num_channels)
            if idx is not None:
                return idx

    for color in ("green", "red", "blue"):
        if color in ch_map:
            return ch_map[color]
    return 0


def select_color_assignments(
    selected_channels: Optional[Sequence[str]],
    channel_names: Sequence[str],
    ch_map: Dict[str, int],
    num_channels: int,
) -> Dict[str, int]:
    if not selected_channels:
        return {color: idx for color, idx in ch_map.items() if color in {"red", "green", "blue"} and idx < num_channels}

    selected_indices: List[int] = []
    selected_semantics: Dict[int, str] = {}

    inverse_map = {idx: color for color, idx in ch_map.items() if color in {"red", "green", "blue"}}

    for token in selected_channels:
        idx = resolve_channel_index(token, channel_names, ch_map, num_channels)
        if idx is None or idx in selected_indices:
            continue
        selected_indices.append(idx)
        if token in {"red", "green", "blue"}:
            selected_semantics[idx] = token
        elif idx in inverse_map:
            selected_semantics[idx] = inverse_map[idx]

    if not selected_indices:
        return {color: idx for color, idx in ch_map.items() if color in {"red", "green", "blue"} and idx < num_channels}

    if len(selected_indices) == 1:
        idx = selected_indices[0]
        color = selected_semantics.get(idx, "green")
        return {color: idx}

    if len(selected_indices) == 2:
        fallback = ["green", "red"]
    else:
        fallback = ["blue", "green", "red"]

    color_assignments: Dict[str, int] = {}
    used_colors: Set[str] = set()
    for idx in selected_indices:
        color = selected_semantics.get(idx)
        if color and color not in used_colors:
            color_assignments[color] = idx
            used_colors.add(color)

    for idx in selected_indices:
        if idx in color_assignments.values():
            continue
        for color in fallback:
            if color not in used_colors:
                color_assignments[color] = idx
                used_colors.add(color)
                break
        if len(color_assignments) == min(len(selected_indices), 3):
            break

    return color_assignments


def compute_clip_ranges(
    data_tczyx: np.ndarray,
    norm_mode: str,
    lower_pct: float,
    upper_pct: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[int, Tuple[float, float]]:
    num_channels = int(data_tczyx.shape[1])
    clip_ranges: Dict[int, Tuple[float, float]] = {}

    for ch_idx in range(num_channels):
        channel_data = nd2_utils.to_native_endian(np.asarray(data_tczyx[:, ch_idx, :, :, :]))
        channel_f = channel_data.astype(np.float32, copy=False)

        if norm_mode == "none":
            cmin = 0.0
            cmax = 0.0
            if metadata is not None:
                try:
                    if metadata.get("lower_limit") is not None:
                        cmin = float(metadata["lower_limit"])
                except Exception:
                    cmin = 0.0
                try:
                    if metadata.get("higher_limit") is not None:
                        cmax = float(metadata["higher_limit"])
                except Exception:
                    cmax = 0.0
                if cmax <= cmin:
                    try:
                        significant_bits = int(metadata.get("significant_bits") or 0)
                        if significant_bits > 0:
                            cmax = float((1 << significant_bits) - 1)
                    except Exception:
                        pass
            if cmax <= cmin:
                if np.issubdtype(channel_data.dtype, np.integer):
                    info = np.iinfo(channel_data.dtype)
                    cmin = float(max(0, info.min))
                    cmax = float(info.max)
                else:
                    cmin = float(np.min(channel_f))
                    cmax = float(np.max(channel_f))
        elif norm_mode == "minmax":
            cmin = float(np.min(channel_f))
            cmax = float(np.max(channel_f))
        else:
            flat = channel_f.ravel()
            if flat.size == 0:
                cmin, cmax = 0.0, 1.0
            else:
                step = max(1, flat.size // 50000)
                sample = flat[::step]
                cmin = float(np.percentile(sample, lower_pct))
                cmax = float(np.percentile(sample, upper_pct))

        if cmax <= cmin:
            cmax = cmin + 1.0
        clip_ranges[ch_idx] = (cmin, cmax)

    return clip_ranges


def resolve_norm_mode(requested_norm_mode: Optional[str], content_type: str) -> str:
    if requested_norm_mode:
        return requested_norm_mode
    if content_type == "image":
        return "none"
    return "percentile"


def _channel_span(clip_ranges: Dict[int, Tuple[float, float]], idx: int) -> float:
    cmin, cmax = clip_ranges.get(idx, (0.0, 0.0))
    return float(cmax - cmin)


def select_meaningful_color_assignments(
    color_assignments: Dict[str, int],
    clip_ranges: Dict[int, Tuple[float, float]],
    brightfield_idx: Optional[int],
) -> Dict[str, int]:
    if not color_assignments:
        return {}

    brightfield_span = _channel_span(clip_ranges, brightfield_idx) if brightfield_idx is not None else 0.0
    min_required_span = max(16.0, brightfield_span * 0.01) if brightfield_span > 0 else 16.0

    filtered: Dict[str, int] = {}
    for color, idx in color_assignments.items():
        if _channel_span(clip_ranges, idx) >= min_required_span:
            filtered[color] = idx
    return filtered


def normalize_channel_to_u8(
    channel_data: np.ndarray,
    clip_range: Tuple[float, float],
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
) -> np.ndarray:
    cmin, cmax = clip_range
    gray_u8 = nd2_utils.normalize_to_uint8_with_clip(channel_data, cmin, cmax)
    return nd2_utils.apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)


def render_rgb_frame(
    frame_cyx: np.ndarray,
    color_assignments: Dict[str, int],
    clip_ranges: Dict[int, Tuple[float, float]],
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
) -> np.ndarray:
    channel_to_u8: Dict[str, np.ndarray] = {}
    for color, idx in color_assignments.items():
        if 0 <= idx < frame_cyx.shape[0]:
            channel_to_u8[color] = nd2_utils.normalize_to_uint8_with_clip(frame_cyx[idx], *clip_ranges[idx])
    if not channel_to_u8:
        channel_to_u8["green"] = nd2_utils.normalize_to_uint8_with_clip(frame_cyx[0], *clip_ranges[0])
    bgr = nd2_utils.build_rgb_from_channels(channel_to_u8, use_clahe, clip_limit, tile_grid)
    return bgr[:, :, ::-1]


def render_gray_frame(
    frame_cyx: np.ndarray,
    gray_index: int,
    clip_ranges: Dict[int, Tuple[float, float]],
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
) -> np.ndarray:
    gray_index = min(max(gray_index, 0), frame_cyx.shape[0] - 1)
    return normalize_channel_to_u8(frame_cyx[gray_index], clip_ranges[gray_index], use_clahe, clip_limit, tile_grid)


def write_rgb_image(image_rgb: np.ndarray, output_path: str) -> None:
    Image.fromarray(image_rgb, mode="RGB").save(output_path, "JPEG", quality=95)


def write_gray_image(image_gray: np.ndarray, output_path: str) -> None:
    Image.fromarray(image_gray, mode="L").save(output_path, "JPEG", quality=95)


def format_spatial_scale_text(side_um: Optional[float]) -> str:
    if side_um is None or side_um <= 0:
        return "Spatial Scale: unknown"
    if side_um >= 1000.0:
        return f"Spatial Scale: {side_um / 1000.0:.3f} mm/side ({side_um:.1f} µm)"
    return f"Spatial Scale: {side_um:.1f} µm/side"


def format_acquisition_interval_text(interval_s: Optional[float]) -> str:
    if interval_s is None or interval_s <= 0:
        return "Acquisition Interval: unknown"
    if interval_s >= 3600.0:
        return f"Acquisition Interval: {interval_s / 3600.0:.2f} hr/frame"
    if interval_s >= 60.0:
        return f"Acquisition Interval: {interval_s / 60.0:.1f} min/frame"
    if interval_s >= 1.0:
        return f"Acquisition Interval: {interval_s:.1f} s/frame"
    return f"Acquisition Interval: {interval_s * 1000.0:.1f} ms/frame"


def applescript_insert_movie_lines(
    posix_path: str,
    indent: str = "            ",
    position: str = "{120, 120}",
    width: int = 512,
    height: int = 512,
) -> List[str]:
    """AppleScript that embeds an MP4 as a Keynote movie, not as an image.

    Assigning a movie file to an image placeholder (`set file of img`) makes
    Keynote start a synchronous transcode / conversion dialog. That call often
    never returns when driven from osascript, which looks like a hang after
    the slide template has already been created.
    """
    return [
        f'{indent}set movieFile to alias (POSIX file "{posix_path}")',
        f"{indent}set moviePlaced to false",
        f"{indent}try",
        f"{indent}    if (count of movies) > 0 then",
        f"{indent}        repeat with mov in (get every movie)",
        f"{indent}            try",
        f"{indent}                with timeout of 60 seconds",
        f"{indent}                    set file of mov to movieFile",
        f"{indent}                end timeout",
        f"{indent}                set moviePlaced to true",
        f"{indent}                exit repeat",
        f"{indent}            on error",
        f"{indent}                try",
        f"{indent}                    with timeout of 60 seconds",
        f"{indent}                        set file name of mov to movieFile",
        f"{indent}                    end timeout",
        f"{indent}                    set moviePlaced to true",
        f"{indent}                    exit repeat",
        f"{indent}                end try",
        f"{indent}            end try",
        f"{indent}        end repeat",
        f"{indent}    end if",
        f"{indent}on error",
        f"{indent}end try",
        f"{indent}if moviePlaced is false then",
        f"{indent}    try",
        f"{indent}        with timeout of 60 seconds",
        f"{indent}            make new movie with properties {{file:movieFile, position:{position}, width:{width}, height:{height}}}",
        f"{indent}        end timeout",
        f"{indent}        set moviePlaced to true",
        f"{indent}    on error errMovie",
        f"{indent}        log \"make new movie failed: \" & errMovie",
        f"{indent}    end try",
        f"{indent}end if",
    ]


def run_keynote_applescript(ascript: str, timeout_s: int = 180) -> subprocess.CompletedProcess:
    """Run Keynote AppleScript with a timeout so a stalled movie import cannot hang forever."""
    print(
        "  Keynote が前面に出て変換ダイアログを出していないか確認してください。"
        " 画像プレースホルダへの動画代入はタイムアウトします。"
    )
    return subprocess.run(
        ["/usr/bin/osascript", "-s", "e", "-s", "o", "-"],
        input=ascript.encode("utf-8"),
        capture_output=True,
        timeout=timeout_s,
    )


def build_movie_metadata(dataset: IcsDataset, playback_fps: float, total_frames: int) -> Dict[str, str]:
    return {
        "scale_text": format_spatial_scale_text(dataset.side_length_um),
        "playback_text": f"Playback FPS: {playback_fps:.3f}",
        "acquisition_text": format_acquisition_interval_text(dataset.source_time_interval_s),
        "total_frames_text": f"Total Frames: {total_frames}",
    }


def build_movie_applescript(dir_path: str, mp4_paths: List[str], metadata_list: List[Dict[str, str]], theme_name: str, out_key_path: str) -> str:
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")

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
    if len(mp4_paths) > 0:
        lines.append('    try')
        lines.append('        tell theDoc to delete slide 1')
        lines.append('    end try')

    for idx, mp4_abs in enumerate(mp4_paths, start=1):
        posix_path = esc(mp4_abs)
        folder_name = os.path.basename(os.path.dirname(mp4_abs))
        file_name = os.path.splitext(os.path.basename(mp4_abs))[0]
        if file_name.endswith("_brightfield"):
            base_name = file_name.replace("_brightfield", "")
            slide_title = f"{folder_name}/{base_name} (明視野)"
        else:
            slide_title = f"{folder_name}/{file_name}"
        escaped_title = esc(slide_title)

        metadata = metadata_list[idx - 1] if idx - 1 < len(metadata_list) else {}
        scale_text = esc(metadata.get("scale_text", "Spatial Scale: unknown"))
        playback_text = esc(metadata.get("playback_text", "Playback FPS: unknown"))
        acquisition_text = esc(metadata.get("acquisition_text", "Acquisition Interval: unknown"))
        total_frames_text = esc(metadata.get("total_frames_text", "Total Frames: unknown"))

        lines.append('    tell theDoc')
        lines.append('        set newSlide to make new slide')
        lines.append('        try')
        lines.append('            set base slide of newSlide to master slide "タイトル、箇条書き、画像Confocal" of theDoc')
        lines.append('        on error')
        lines.append('            try')
        lines.append('                set base slide of newSlide to master slide "Blank" of theDoc')
        lines.append('            on error')
        lines.append('                try')
        lines.append('                    set base slide of newSlide to master slide "空白" of theDoc')
        lines.append('                end try')
        lines.append('            end try')
        lines.append('        end try')
        lines.append('        set current slide of theDoc to newSlide')
        lines.append('        delay 0.2')
        lines.append('        tell newSlide')
        lines.append('            try')
        lines.append('                if (count of text items) > 0 then')
        lines.append(f'                    set object text of text item 1 to "{escaped_title}"')
        lines.append('                else')
        lines.append(f'                    set titleShape to make new text item with properties {{object text:"{escaped_title}", position:{{50, 50}}}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append(f'                set titleShape to make new text item with properties {{object text:"{escaped_title}", position:{{50, 50}}}}')
        lines.append('            end try')
        lines.append('            try')
        lines.append('                if (count of text items) > 1 then')
        lines.append(f'                    set object text of text item 2 to "{scale_text}" & return & "{playback_text}" & return & "{acquisition_text}" & return & "{total_frames_text}"')
        lines.append('                else')
        lines.append(f'                    set metadataBox to make new text item with properties {{object text:"{scale_text}" & return & "{playback_text}" & return & "{acquisition_text}" & return & "{total_frames_text}", position:{{400, 150}}, width:260, height:130}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append(f'                set metadataBox to make new text item with properties {{object text:"{scale_text}" & return & "{playback_text}" & return & "{acquisition_text}" & return & "{total_frames_text}", position:{{400, 150}}, width:260, height:130}}')
        lines.append('            end try')
        lines.extend(applescript_insert_movie_lines(posix_path, indent="            "))
        lines.append('        end tell')
        lines.append('    end tell')

    out_posix = esc(out_key_path)
    lines.append(f'    save theDoc in (POSIX file "{out_posix}")')
    lines.append('end tell')
    return "\n".join(lines)


def build_volume_applescript(
    movie_path: str,
    xz_image_path: str,
    yz_image_path: str,
    slide_title: str,
    theme_name: str,
    out_key_path: str,
    nx: int,
    ny: int,
    nz: int,
    x_um: Optional[float],
    y_um: Optional[float],
    z_um: Optional[float],
) -> str:
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")

    movie_posix = esc(movie_path)
    xz_posix = esc(xz_image_path)
    yz_posix = esc(yz_image_path)
    title_escaped = esc(slide_title)
    out_posix = esc(out_key_path)

    def fmt_size(axis: str, size_um: Optional[float], size_px: int, unit_label: str) -> str:
        if size_um is None or size_um <= 0:
            return f"{axis}: unknown ({size_px} {unit_label})"
        return f"{axis}: {size_um:.1f} µm ({size_px} {unit_label})"

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
    lines.append('    try')
    lines.append('        tell theDoc to delete slide 1')
    lines.append('    end try')
    lines.append('    tell theDoc')
    lines.append('        set newSlide to make new slide')
    lines.append('        try')
    lines.append('            set base slide of newSlide to master slide "タイトル、箇条書き、画像Confocal" of theDoc')
    lines.append('        on error')
    lines.append('            try')
    lines.append('                set base slide of newSlide to master slide "Blank" of theDoc')
    lines.append('            on error')
    lines.append('                try')
    lines.append('                    set base slide of newSlide to master slide "空白" of theDoc')
    lines.append('                end try')
    lines.append('            end try')
    lines.append('        end try')
    lines.append('        set current slide of theDoc to newSlide')
    lines.append('        delay 0.2')
    lines.append('        tell newSlide')
    lines.append(f'            set object text of text item 1 to "{title_escaped}"')

    total_width = nx + 10 + nz
    total_height = ny + 10 + nz
    scale = min(512.0 / total_width, 512.0 / total_height)
    base_x = 30
    base_y = 180

    movie_w = int(nx * scale)
    movie_h = int(ny * scale)
    yz_w = int(nz * scale)
    yz_h = int(ny * scale)
    xz_w = int(nx * scale)
    xz_h = int(nz * scale)

    movie_x = base_x
    movie_y = base_y
    yz_x = base_x + int((nx + 10) * scale)
    yz_y = base_y
    xz_x = base_x
    xz_y = base_y + int((ny + 10) * scale)

    lines.extend(
        applescript_insert_movie_lines(
            movie_posix,
            indent="            ",
            position=f"{{{movie_x}, {movie_y}}}",
            width=movie_w,
            height=movie_h,
        )
    )
    lines.append(f'            set yzImageFile to alias (POSIX file "{yz_posix}")')
    lines.append(f'            set yzImage to make new image with properties {{file:yzImageFile, position:{{{yz_x}, {yz_y}}}, width:{yz_w}, height:{yz_h}}}')
    lines.append(f'            set xzImageFile to alias (POSIX file "{xz_posix}")')
    lines.append(f'            set xzImage to make new image with properties {{file:xzImageFile, position:{{{xz_x}, {xz_y}}}, width:{xz_w}, height:{xz_h}}}')

    volume_lines = [
        "Volume Information:",
        fmt_size("X", x_um, nx, "px"),
        fmt_size("Y", y_um, ny, "px"),
        fmt_size("Z", z_um, nz, "slices"),
    ]
    volume_info_as = " & return & ".join(f'"{esc(line)}"' for line in volume_lines)

    lines.append("            try")
    lines.append("                if (count of text items) > 1 then")
    lines.append(f"                    set object text of text item 2 to {volume_info_as}")
    lines.append("                else")
    lines.append(f"                    set volumeInfoBox to make new text item with properties {{object text:{volume_info_as}, position:{{600, 180}}, width:220, height:150}}")
    lines.append("                end if")
    lines.append("            on error")
    lines.append(f"                set volumeInfoBox to make new text item with properties {{object text:{volume_info_as}, position:{{600, 180}}, width:220, height:150}}")
    lines.append("            end try")
    lines.append("        end tell")
    lines.append("    end tell")
    lines.append(f'    save theDoc in (POSIX file "{out_posix}")')
    lines.append("end tell")
    return "\n".join(lines)


def write_rgb_movie(frames_rgb: List[np.ndarray], output_path: str, fps: float, preferred_codec: Optional[str], use_ffmpeg: bool, metadata_comment: str) -> None:
    require_movie_modules()
    if use_ffmpeg and nd2_to_mp4.create_high_quality_mp4_with_ffmpeg([frame[:, :, ::-1] for frame in frames_rgb], output_path, fps, metadata_comment=metadata_comment):
        return

    first_frame = frames_rgb[0]
    writer = nd2_to_mp4.ensure_video_writer(output_path, fps, (first_frame.shape[1], first_frame.shape[0]), preferred_codec)
    try:
        for frame in frames_rgb:
            writer.write(frame[:, :, ::-1])
    finally:
        writer.release()
    nd2_to_mp4.set_mp4_comment_with_ffmpeg(output_path, metadata_comment)


def write_gray_movie(frames_gray: List[np.ndarray], output_path: str, fps: float, preferred_codec: Optional[str], use_ffmpeg: bool, metadata_comment: str) -> None:
    require_movie_modules()
    if use_ffmpeg and nd2_to_mp4.create_grayscale_mp4_with_ffmpeg(frames_gray, output_path, fps, metadata_comment=metadata_comment):
        return

    first_frame = frames_gray[0]
    writer = nd2_to_mp4.ensure_grayscale_video_writer(output_path, fps, (first_frame.shape[1], first_frame.shape[0]), preferred_codec)
    try:
        for frame in frames_gray:
            writer.write(frame)
    finally:
        writer.release()
    nd2_to_mp4.set_mp4_comment_with_ffmpeg(output_path, metadata_comment)


def process_images_to_keynote(image_files: Sequence[str], args: argparse.Namespace, root_dir: str) -> Optional[str]:
    if not image_files:
        return None
    require_image_builder()

    print(f"\n📸 Processing {len(image_files)} ICS image file(s)...")
    temp_dir = os.path.join(root_dir, "temp_ics_images")
    os.makedirs(temp_dir, exist_ok=True)

    selected_channels = _channels_argument(args)
    jpg_data: List[Tuple[str, float, str]] = []
    norm_mode = resolve_norm_mode(args.norm_mode, "image")

    for ics_path in image_files:
        try:
            dataset = load_ics_dataset(ics_path)
            frame_cyx = nd2_utils.to_native_endian(np.asarray(dataset.data_tczyx[0, :, 0, :, :]))
            ch_map = complete_channel_map(dataset.channel_names, dataset.num_c)
            clip_ranges = compute_clip_ranges(
                dataset.data_tczyx[:, :, :1, :, :],
                norm_mode,
                args.lower_pct,
                args.upper_pct,
                dataset.metadata,
            )

            out_jpg = os.path.join(temp_dir, f"{_relative_stem(root_dir, ics_path)}.jpg")
            if args.single_channel:
                gray_index = select_gray_index(selected_channels, dataset.channel_names, ch_map, dataset.num_c)
                image_gray = render_gray_frame(frame_cyx, gray_index, clip_ranges, False, args.clip_limit, args.tile_grid)
                write_gray_image(image_gray, out_jpg)
            else:
                color_assignments = select_color_assignments(selected_channels, dataset.channel_names, ch_map, dataset.num_c)
                image_rgb = render_rgb_frame(frame_cyx, color_assignments, clip_ranges, False, args.clip_limit, args.tile_grid)
                write_rgb_image(image_rgb, out_jpg)

            jpg_data.append((out_jpg, float(dataset.side_length_um or 0.0), ics_path))
            print(f"  ✅ Created image preview: {os.path.relpath(out_jpg, root_dir)}")
        except Exception as exc:
            print(f"Error processing {ics_path}: {exc}")

    if not jpg_data:
        print("No image previews were created from ICS files.")
        return None

    out_name = args.output or f"{os.path.basename(root_dir.rstrip(os.sep)) or 'Images'}_Images.key"
    if not out_name.lower().endswith(".key"):
        out_name += ".key"
    out_key_path = os.path.join(root_dir, out_name)

    ascript = nd2images_to_keynote.build_applescript(root_dir, jpg_data, args.theme, out_key_path)

    print(f"\n🎯 Creating Keynote presentation: {os.path.basename(out_key_path)}")
    completed = subprocess.run(
        ["/usr/bin/osascript", "-s", "e", "-s", "o", "-"],
        input=ascript.encode("utf-8"),
        capture_output=True,
    )
    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
        print("AppleScript failed. Details:\n" + err, file=sys.stderr)
        return None

    if not args.keep_jpgs:
        for jpg_path, _, _ in jpg_data:
            try:
                os.remove(jpg_path)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    print(f"✅ Image Keynote created: {out_key_path}")
    return out_key_path


def process_movies_to_keynote(movie_files: Sequence[str], args: argparse.Namespace, root_dir: str) -> Optional[str]:
    if not movie_files:
        return None
    require_movie_modules()

    print(f"\n🎬 Processing {len(movie_files)} ICS movie file(s)...")
    selected_channels = _channels_argument(args)
    use_clahe = bool(args.use_clahe)
    use_ffmpeg = not args.no_ffmpeg and getattr(nd2_to_mp4, "check_ffmpeg_availability", lambda: False)()
    norm_mode = resolve_norm_mode(args.norm_mode, "movie")

    mp4_paths: List[str] = []
    metadata_list: List[Dict[str, str]] = []

    for ics_path in movie_files:
        try:
            dataset = load_ics_dataset(ics_path)
            ch_map = complete_channel_map(dataset.channel_names, dataset.num_c)
            clip_ranges = compute_clip_ranges(
                dataset.data_tczyx[:, :, :1, :, :],
                norm_mode,
                args.lower_pct,
                args.upper_pct,
                dataset.metadata,
            )
            fps = nd2_utils.detect_fps_from_metadata(dataset.metadata) or float(args.fps)
            base_name = os.path.splitext(os.path.basename(ics_path))[0]
            out_path = os.path.join(os.path.dirname(ics_path), f"{base_name}.mp4")
            brightfield_path = os.path.join(os.path.dirname(ics_path), f"{base_name}_brightfield.mp4")
            brightfield_idx = ch_map.get("brightfield")

            if args.no_overwrite and os.path.exists(out_path):
                print(f"  ⏭️  Skipping existing movie: {os.path.relpath(out_path, root_dir)}")
                mp4_paths.append(out_path)
                metadata_list.append(build_movie_metadata(dataset, fps, dataset.num_t))
                continue

            frames_rgb: List[np.ndarray] = []
            frames_gray: List[np.ndarray] = []

            if args.single_channel:
                gray_index = select_gray_index(selected_channels, dataset.channel_names, ch_map, dataset.num_c)
                for t_idx in range(dataset.num_t):
                    frame_cyx = nd2_utils.to_native_endian(np.asarray(dataset.data_tczyx[t_idx, :, 0, :, :]))
                    frames_gray.append(render_gray_frame(frame_cyx, gray_index, clip_ranges, use_clahe, args.clip_limit, args.tile_grid))
                note = f"Channel {gray_index} grayscale"
                metadata_comment = build_comment_tag(ics_path, dataset.side_length_um, fps, note=note)
                write_gray_movie(frames_gray, out_path, fps, args.codec, use_ffmpeg, metadata_comment)
                print(f"  ✅ Created movie: {os.path.relpath(out_path, root_dir)}")
                mp4_paths.append(out_path)
                metadata_list.append(build_movie_metadata(dataset, fps, dataset.num_t))
            else:
                color_assignments = select_color_assignments(selected_channels, dataset.channel_names, ch_map, dataset.num_c)
                meaningful_assignments = select_meaningful_color_assignments(color_assignments, clip_ranges, brightfield_idx)

                if meaningful_assignments:
                    for t_idx in range(dataset.num_t):
                        frame_cyx = nd2_utils.to_native_endian(np.asarray(dataset.data_tczyx[t_idx, :, 0, :, :]))
                        frames_rgb.append(render_rgb_frame(frame_cyx, meaningful_assignments, clip_ranges, use_clahe, args.clip_limit, args.tile_grid))
                    metadata_comment = build_comment_tag(ics_path, dataset.side_length_um, fps)
                    write_rgb_movie(frames_rgb, out_path, fps, args.codec, use_ffmpeg, metadata_comment)
                    print(f"  ✅ Created movie: {os.path.relpath(out_path, root_dir)}")
                    mp4_paths.append(out_path)
                    metadata_list.append(build_movie_metadata(dataset, fps, dataset.num_t))
                elif brightfield_idx is not None:
                    print(
                        f"  ℹ️  Fluorescence channels are weak in {os.path.basename(ics_path)}; "
                        "using brightfield channel instead."
                    )
                    for t_idx in range(dataset.num_t):
                        frame_cyx = nd2_utils.to_native_endian(np.asarray(dataset.data_tczyx[t_idx, :, 0, :, :]))
                        frames_gray.append(
                            render_gray_frame(frame_cyx, int(brightfield_idx), clip_ranges, use_clahe, args.clip_limit, args.tile_grid)
                        )
                    metadata_comment = build_comment_tag(ics_path, dataset.side_length_um, fps, note="Brightfield channel")
                    write_gray_movie(frames_gray, brightfield_path, fps, args.codec, use_ffmpeg, metadata_comment)
                    print(f"  ✅ Created movie: {os.path.relpath(brightfield_path, root_dir)}")
                    mp4_paths.append(brightfield_path)
                    metadata_list.append(build_movie_metadata(dataset, fps, dataset.num_t))
                else:
                    for t_idx in range(dataset.num_t):
                        frame_cyx = nd2_utils.to_native_endian(np.asarray(dataset.data_tczyx[t_idx, :, 0, :, :]))
                        frames_rgb.append(render_rgb_frame(frame_cyx, color_assignments, clip_ranges, use_clahe, args.clip_limit, args.tile_grid))
                    metadata_comment = build_comment_tag(ics_path, dataset.side_length_um, fps)
                    write_rgb_movie(frames_rgb, out_path, fps, args.codec, use_ffmpeg, metadata_comment)
                    print(f"  ✅ Created movie: {os.path.relpath(out_path, root_dir)}")
                    mp4_paths.append(out_path)
                    metadata_list.append(build_movie_metadata(dataset, fps, dataset.num_t))
        except Exception as exc:
            print(f"Error processing {ics_path}: {exc}")

    if not mp4_paths:
        print("No movies were created from ICS files.")
        return None

    out_name = args.output or f"{os.path.basename(root_dir.rstrip(os.sep)) or 'Movies'}_Movies.key"
    if not out_name.lower().endswith(".key"):
        out_name += ".key"
    out_key_path = os.path.join(root_dir, out_name)

    ascript = build_movie_applescript(root_dir, mp4_paths, metadata_list, args.theme, out_key_path)

    print(f"\n🎯 Creating Keynote presentation: {os.path.basename(out_key_path)}")
    print(f"  Inserting {len(mp4_paths)} movie(s) as Keynote movie objects (not image placeholders)")
    try:
        completed = run_keynote_applescript(ascript, timeout_s=max(180, 90 * len(mp4_paths)))
    except subprocess.TimeoutExpired:
        print(
            "ERROR: Keynote AppleScript timed out while inserting MP4 files.\n"
            "This usually means Keynote is waiting on a movie-conversion dialog, "
            "or an image placeholder import is stuck. Check the Keynote window and retry.",
            file=sys.stderr,
        )
        return None
    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
        print("AppleScript failed. Details:\n" + err, file=sys.stderr)
        return None

    print(f"✅ Movie Keynote created: {out_key_path}")
    return out_key_path


def process_volumes_to_keynote(volume_files: Sequence[str], args: argparse.Namespace, root_dir: str) -> List[str]:
    if not volume_files:
        return []
    require_movie_modules()

    print(f"\n📦 Processing {len(volume_files)} ICS volume file(s)...")
    selected_channels = _channels_argument(args)
    use_clahe = bool(args.use_clahe)
    use_ffmpeg = not args.no_ffmpeg and getattr(nd2_to_mp4, "check_ffmpeg_availability", lambda: False)()
    temp_dir = os.path.join(root_dir, "temp_ics_volumes")
    os.makedirs(temp_dir, exist_ok=True)
    norm_mode = resolve_norm_mode(args.norm_mode, "volume")

    created_presentations: List[str] = []
    created_temp_files: List[str] = []

    for ics_path in volume_files:
        try:
            dataset = load_ics_dataset(ics_path)
            if dataset.num_t > 1:
                print(f"  Warning: {os.path.basename(ics_path)} contains T and Z axes; using the first timepoint for volume rendering.")

            volume_czyx = nd2_utils.to_native_endian(np.asarray(dataset.data_tczyx[0, :, :, :, :]))
            ch_map = complete_channel_map(dataset.channel_names, dataset.num_c)
            clip_ranges = compute_clip_ranges(
                dataset.data_tczyx[:1, :, :, :, :],
                norm_mode,
                args.lower_pct,
                args.upper_pct,
                dataset.metadata,
            )

            rel_stem = _relative_stem(root_dir, ics_path)
            movie_path = os.path.join(temp_dir, f"{rel_stem}_zstack.mp4")
            xz_path = os.path.join(temp_dir, f"{rel_stem}_xz.jpg")
            yz_path = os.path.join(temp_dir, f"{rel_stem}_yz.jpg")
            created_temp_files.extend([movie_path, xz_path, yz_path])

            if args.single_channel:
                gray_index = select_gray_index(selected_channels, dataset.channel_names, ch_map, dataset.num_c)
                frames_gray: List[np.ndarray] = []
                for z_idx in range(dataset.num_z):
                    frame_cyx = volume_czyx[:, z_idx, :, :]
                    frames_gray.append(render_gray_frame(frame_cyx, gray_index, clip_ranges, use_clahe, args.clip_limit, args.tile_grid))
                write_gray_movie(
                    frames_gray,
                    movie_path,
                    float(args.fps),
                    args.codec,
                    use_ffmpeg,
                    build_comment_tag(ics_path, dataset.side_length_um, float(args.fps), note=f"Volume channel {gray_index}"),
                )

                gray_stack = np.stack([volume_czyx[gray_index, z_idx, :, :] for z_idx in range(dataset.num_z)], axis=0)
                center_y = gray_stack.shape[1] // 2
                center_x = gray_stack.shape[2] // 2
                xz_gray = np.transpose(gray_stack[:, center_y, :], (0, 1))
                yz_gray = np.transpose(gray_stack[:, :, center_x], (1, 0))
                xz_u8 = normalize_channel_to_u8(xz_gray, clip_ranges[gray_index], use_clahe, args.clip_limit, args.tile_grid)
                yz_u8 = normalize_channel_to_u8(yz_gray, clip_ranges[gray_index], use_clahe, args.clip_limit, args.tile_grid)
                write_gray_image(xz_u8, xz_path)
                write_gray_image(yz_u8, yz_path)
            else:
                color_assignments = select_color_assignments(selected_channels, dataset.channel_names, ch_map, dataset.num_c)
                frames_rgb: List[np.ndarray] = []
                for z_idx in range(dataset.num_z):
                    frame_cyx = volume_czyx[:, z_idx, :, :]
                    frames_rgb.append(render_rgb_frame(frame_cyx, color_assignments, clip_ranges, use_clahe, args.clip_limit, args.tile_grid))
                write_rgb_movie(
                    frames_rgb,
                    movie_path,
                    float(args.fps),
                    args.codec,
                    use_ffmpeg,
                    build_comment_tag(ics_path, dataset.side_length_um, float(args.fps), note="Volume z-stack"),
                )

                center_y = dataset.height // 2
                center_x = dataset.width // 2
                xz_channels = np.transpose(volume_czyx[:, :, center_y, :], (0, 1, 2))
                yz_channels = np.transpose(volume_czyx[:, :, :, center_x], (0, 2, 1))
                xz_rgb = render_rgb_frame(xz_channels, color_assignments, clip_ranges, use_clahe, args.clip_limit, args.tile_grid)
                yz_rgb = render_rgb_frame(yz_channels, color_assignments, clip_ranges, use_clahe, args.clip_limit, args.tile_grid)
                write_rgb_image(xz_rgb, xz_path)
                write_rgb_image(yz_rgb, yz_path)

            x_um = dataset.width * dataset.pixel_size_x_um if dataset.pixel_size_x_um else None
            y_um = dataset.height * dataset.pixel_size_y_um if dataset.pixel_size_y_um else None
            z_um = dataset.num_z * dataset.pixel_size_z_um if dataset.pixel_size_z_um else None

            rel_title = os.path.splitext(os.path.relpath(ics_path, root_dir))[0]
            out_name = f"{rel_stem}_Volume.key"
            out_key_path = os.path.join(root_dir, out_name)

            ascript = build_volume_applescript(
                movie_path=movie_path,
                xz_image_path=xz_path,
                yz_image_path=yz_path,
                slide_title=rel_title,
                theme_name=args.theme,
                out_key_path=out_key_path,
                nx=dataset.width,
                ny=dataset.height,
                nz=dataset.num_z,
                x_um=x_um,
                y_um=y_um,
                z_um=z_um,
            )

            print(f"\n🎯 Creating volume Keynote: {os.path.basename(out_key_path)}")
            try:
                completed = run_keynote_applescript(ascript, timeout_s=180)
            except subprocess.TimeoutExpired:
                print(
                    "ERROR: Keynote AppleScript timed out while inserting the volume movie.\n"
                    "Check the Keynote window for a conversion dialog and retry.",
                    file=sys.stderr,
                )
                continue
            if completed.returncode != 0:
                err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
                print("AppleScript failed. Details:\n" + err, file=sys.stderr)
                continue

            print(f"  ✅ Created volume Keynote: {os.path.basename(out_key_path)}")
            created_presentations.append(out_key_path)
        except Exception as exc:
            print(f"Error processing {ics_path}: {exc}")

    if not args.keep_temp:
        for temp_path in created_temp_files:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    return created_presentations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Keynote presentations from ICS files (auto-detects images, movies, and volumes)."
    )

    parser.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: current directory)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto-generated)")
    parser.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    parser.add_argument("--verbose", action="store_true", help="Reserved for compatibility with nd2_to_keynote.py")

    parser.add_argument("--image-only", action="store_true", help="Force all ICS files to be treated as images")
    parser.add_argument("--movie-only", action="store_true", help="Force all ICS files to be treated as movies")
    parser.add_argument("--volume-only", action="store_true", help="Force all ICS files to be treated as volumes")

    parser.add_argument("--keep-jpgs", action="store_true", help="Keep generated JPG files after creating Keynote")
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Specific channels to extract (comma-separated): red,green,blue or channel indices (0,1,2)",
    )
    parser.add_argument("--single-channel", action="store_true", help="Extract only one channel as grayscale")

    parser.add_argument("--fps", type=float, default=10.0, help="Fallback FPS if not found in metadata (default: 10)")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit when --clahe is enabled (default: 3.0)")
    parser.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (N -> NxN) when --clahe is enabled (default: 8)")
    parser.set_defaults(use_clahe=False)
    parser.add_argument("--clahe", dest="use_clahe", action="store_true", help="Enable CLAHE contrast enhancement (disabled by default)")
    parser.add_argument("--no-clahe", dest="use_clahe", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--codec", type=str, default="avc1", help="Preferred fourcc codec for MP4 (default: avc1)")
    parser.add_argument(
        "--norm-mode",
        choices=["none", "percentile", "minmax"],
        default=None,
        help="Normalization mode (default: images=none, movies/volumes=percentile)",
    )
    parser.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for histogram clipping")
    parser.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile for histogram clipping")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing MP4 files instead of overwriting them")
    parser.add_argument("--no-ffmpeg", action="store_true", help="Disable FFmpeg and use OpenCV VideoWriter instead")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary MP4 and JPG files for volumes")

    args = parser.parse_args()

    force_count = sum([args.image_only, args.movie_only, args.volume_only])
    if force_count > 1:
        print("Error: Cannot specify more than one of --image-only, --movie-only, --volume-only", file=sys.stderr)
        sys.exit(1)

    try:
        require_bioio()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    root_dir = os.path.abspath(args.input)
    if not os.path.isdir(root_dir):
        print("Directory not found:", root_dir, file=sys.stderr)
        sys.exit(1)

    if args.non_recursive:
        ics_paths = sorted(
            [
                os.path.join(root_dir, file_name)
                for file_name in os.listdir(root_dir)
                if file_name.lower().endswith(".ics") and not file_name.startswith(".") and not file_name.startswith("._")
            ]
        )
    else:
        ics_paths = find_ics_files_recursively(root_dir)

    if not ics_paths:
        print("No ICS files found.")
        sys.exit(0)

    print(f"Found {len(ics_paths)} ICS file(s):")
    for ics_file in ics_paths:
        print(f"  - {os.path.relpath(ics_file, root_dir)}")

    force_mode = None
    if args.image_only:
        force_mode = "image"
        print("\n🔧 Force mode: All files will be treated as images")
    elif args.movie_only:
        force_mode = "movie"
        print("\n🔧 Force mode: All files will be treated as movies")
    elif args.volume_only:
        force_mode = "volume"
        print("\n🔧 Force mode: All files will be treated as volumes")
    else:
        print("\n🔍 Auto-detecting content type...")

    image_files, movie_files, volume_files = categorize_ics_files(ics_paths, force_mode)

    print("\n📊 Summary:")
    print(f"  Images:  {len(image_files)} files")
    print(f"  Movies:  {len(movie_files)} files")
    print(f"  Volumes: {len(volume_files)} files")

    created_presentations: List[str] = []

    if image_files:
        image_keynote = process_images_to_keynote(image_files, args, root_dir)
        if image_keynote:
            created_presentations.append(image_keynote)

    if movie_files:
        movie_keynote = process_movies_to_keynote(movie_files, args, root_dir)
        if movie_keynote:
            created_presentations.append(movie_keynote)

    if volume_files:
        created_presentations.extend(process_volumes_to_keynote(volume_files, args, root_dir))

    if created_presentations:
        print(f"\n🎉 Successfully created {len(created_presentations)} Keynote presentation(s):")
        for presentation in created_presentations:
            print(f"  ✅ {os.path.basename(presentation)}")
        return

    print("\n❌ No Keynote presentations were created.")
    sys.exit(1)


if __name__ == "__main__":
    main()
