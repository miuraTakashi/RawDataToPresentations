#!/usr/bin/env python3
"""
Batch-convert ND2 movies in subfolders to MP4 with red-on-black coloring and optional CLAHE contrast enhancement.

Features:
- Recursively scans an input directory and all subdirectories for .nd2 files
- Reads frames via nd2reader (preferred) or pims (fallback)
- Converts grayscale intensity to red channel (black background)
- Applies 8-bit normalization using a single global movie-wide range, selectable by:
  - percentile-based histogram clipping (default, robust to outliers)
  - min-max (legacy)
- Optionally applies CLAHE contrast enhancement after normalization
- Writes MP4 (mp4v, with fallback codecs) per ND2 into the same subfolder
- If an ND2 has only one time frame, writes JPEG instead of MP4
- Creates separate grayscale MP4/JPEG for brightfield channels when detected

Usage examples:
  python nd2_to_mp4.py --input "/Volumes/KINGSTON/20250917PodocyteBis-T" --fps 10
  python nd2_to_mp4.py --input . --clip-limit 3.0 --tile-grid 8 --codec mp4v
  python nd2_to_mp4.py --input . --no-clahe  # Disable CLAHE to avoid flicker
  python nd2_to_mp4.py --norm-mode percentile --lower-pct 1.0 --upper-pct 99.5 --hist-bins 4096

Dependencies (install as needed):
  pip install nd2reader pims opencv-python numpy
"""

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Generator, List, Optional, Tuple

import numpy as np

# Add script directory to Python path for nd2_utils import
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import common utilities from nd2_utils
try:
    import nd2_utils
    _HAS_ND2_UTILS = True
except ImportError:
    _HAS_ND2_UTILS = False

# Use nd2_utils for backend detection if available, otherwise fallback to local
if _HAS_ND2_UTILS:
    _HAS_ND2READER = nd2_utils._HAS_ND2READER
    _HAS_PIMS = nd2_utils._HAS_PIMS
    ND2Reader = nd2_utils.ND2Reader
    pims = nd2_utils.pims
else:
    # Fallback: local imports
    try:
        from nd2reader import ND2Reader  # type: ignore
        _HAS_ND2READER = True
    except Exception:
        ND2Reader = None  # type: ignore
        _HAS_ND2READER = False

    try:
        import pims  # type: ignore
        _HAS_PIMS = True
    except Exception:
        pims = None  # type: ignore
        _HAS_PIMS = False

try:
    import cv2  # type: ignore
except Exception as exc:
    sys.stderr.write("ERROR: OpenCV (opencv-python) is required. Install with: pip install opencv-python\n")
    raise


# Re-export from nd2_utils for backward compatibility (other scripts import from nd2_to_mp4)
if _HAS_ND2_UTILS:
    find_nd2_files_recursively = nd2_utils.find_nd2_files_recursively
    detect_fps_from_metadata = nd2_utils.detect_fps_from_metadata
else:
    def find_nd2_files_recursively(root_dir: str) -> Generator[Tuple[str, str], None, None]:
        """Yield (parent_folder_path, nd2_file_path) for each .nd2 in root_dir and all subdirectories."""
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(".nd2"):
                    nd2_file_path = os.path.join(dirpath, filename)
                    yield dirpath, nd2_file_path


    def detect_fps_from_metadata(meta: dict) -> Optional[float]:
        """Attempt to detect frames-per-second from known metadata keys."""
        for key in ("frame_rate", "fps", "frames_per_second", "acquisition_frame_rate", "framerate"):
            val = meta.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
        # Some libraries store frame_time interval (seconds)
        for key in ("frame_time", "frame_interval", "dt"):
            val = meta.get(key)
            try:
                if isinstance(val, (int, float)) and val > 0:
                    return 1.0 / float(val)
            except Exception:
                pass
        return None


def extract_side_length_um(meta: dict, height_px: int, width_px: int) -> Optional[float]:
    """Estimate a representative side length (µm) of the field of view.

    Tries multiple metadata keys for pixel size. Returns the larger of
    (height_px * pixel_size_y_um, width_px * pixel_size_x_um) if available.
    """
    try:
        pixel_size_x = None
        pixel_size_y = None

        # Direct keys first
        for key in ("pixel_microns", "pixel_size", "pixelSize", "pixel_size_x", "pixelSizeX", "x_pixel_size"):
            if key in meta:
                v = meta[key]
                if isinstance(v, dict):
                    pixel_size_x = v.get('x') or v.get('width')
                    pixel_size_y = v.get('y') or v.get('height')
                else:
                    pixel_size_x = v
                    pixel_size_y = v
                break

        # Nested common containers
        if pixel_size_x is None:
            for key in ("experiment", "acquisition", "calibration", "microscope"):
                if key in meta and isinstance(meta[key], dict):
                    sub = meta[key]
                    for sk in ("pixel_microns", "pixel_size", "pixelSize", "pixel_size_x", "pixelSizeX"):
                        if sk in sub:
                            v = sub[sk]
                            if isinstance(v, dict):
                                pixel_size_x = v.get('x') or v.get('width')
                                pixel_size_y = v.get('y') or v.get('height')
                            else:
                                pixel_size_x = v
                                pixel_size_y = v
                            break
                    if pixel_size_x is not None:
                        break

        if pixel_size_x is None or pixel_size_y is None:
            return None

        # Convert to micrometers if values look like meters
        def to_um(val: float) -> float:
            try:
                if val < 1e-6:
                    return float(val) * 1e6
                return float(val)
            except Exception:
                return 0.0

        px_um = to_um(float(pixel_size_x))
        py_um = to_um(float(pixel_size_y))
        if px_um <= 0 or py_um <= 0 or height_px <= 0 or width_px <= 0:
            return None

        length_um = height_px * py_um
        width_um = width_px * px_um
        return max(length_um, width_um)
    except Exception:
        return None


def build_comment_tag(source_path: str, side_length_um: Optional[float], fps: float, note: Optional[str] = None) -> str:
    """Build comment string for MP4 metadata."""
    parts = [f"Source: {source_path}"]
    if side_length_um and side_length_um > 0:
        parts.append(f"Side: {side_length_um:.1f}µm")
    else:
        parts.append("Side: unknown")
    parts.append(f"FPS: {fps:.3f}")
    if note:
        parts.append(note)
    return " | ".join(parts)


def normalize_to_uint8_with_range(gray: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """Normalize image to 8-bit [0,255] using provided global min/max across the movie."""
    if gray.dtype == np.uint8 and global_min <= 0 and global_max >= 255:
        return gray
    if global_max <= global_min:
        return np.zeros_like(gray, dtype=np.uint8)
    scaled = (gray.astype(np.float32) - float(global_min)) * (255.0 / (float(global_max) - float(global_min)))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def normalize_to_uint8_with_clip(gray: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Clip image to [clip_min, clip_max] and normalize to 8-bit [0,255].

    More robust than pure min-max if clip_{min,max} are chosen from percentiles.
    """
    if clip_max <= clip_min:
        return np.zeros_like(gray, dtype=np.uint8)
    gray_f = gray.astype(np.float32)
    clipped = np.clip(gray_f, float(clip_min), float(clip_max))
    scaled = (clipped - float(clip_min)) * (255.0 / (float(clip_max) - float(clip_min)))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_native_endian(arr: np.ndarray) -> np.ndarray:
    """Ensure array uses native-endian dtype (avoids any byte-order interpretation issues).

    If the array dtype is non-native-endian (e.g., '>u2' on little-endian machines),
    byteswap to native and update dtype metadata. Numpy arithmetic generally handles
    endianness, but normalizing behavior is clearest with native-endian.
    """
    dt = arr.dtype
    # '|' means not applicable (byte order independent), '=' means native
    # '>' big-endian, '<' little-endian
    if dt.byteorder in ('>', '<') and not dt.isnative:
        return arr.byteswap().newbyteorder()
    return arr


def apply_clahe_if_needed(gray_u8: np.ndarray, use_clahe: bool, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Optionally apply CLAHE to an 8-bit grayscale image."""
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
        return clahe.apply(gray_u8)
    return gray_u8


def build_rgb_from_channels(
    channel_to_u8: dict,
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
) -> np.ndarray:
    """Construct BGR frame from available per-channel 8-bit arrays.

    channel_to_u8 keys: 'red', 'green', 'blue' mapped to 8-bit grayscale arrays.
    Unknown/missing channels are filled with zeros.
    """
    # Determine frame size from any available channel
    any_arr = next(iter(channel_to_u8.values()))
    h, w = any_arr.shape[:2]
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    if 'blue' in channel_to_u8:
        bgr[:, :, 0] = apply_clahe_if_needed(channel_to_u8['blue'], use_clahe, clip_limit, tile_grid)
    if 'green' in channel_to_u8:
        bgr[:, :, 1] = apply_clahe_if_needed(channel_to_u8['green'], use_clahe, clip_limit, tile_grid)
    if 'red' in channel_to_u8:
        bgr[:, :, 2] = apply_clahe_if_needed(channel_to_u8['red'], use_clahe, clip_limit, tile_grid)
    return bgr


def print_channel_info(nd2_path: str, meta: dict, num_channels: int, ch_map: dict) -> None:
    """Print detailed channel information for the ND2 file."""
    print(f"\n📊 Channel Information for: {os.path.basename(nd2_path)}")
    print(f"   Total channels: {num_channels}")
    
    try:
        channels_meta = meta.get('channels') if isinstance(meta, dict) else None
        if isinstance(channels_meta, (list, tuple)):
            print("   Channel details:")
            for idx, ch in enumerate(channels_meta):
                if idx >= num_channels:
                    break
                # Extract channel name
                if isinstance(ch, dict):
                    name = str(ch.get('label') or ch.get('name') or ch.get('channel') or f'Channel {idx}')
                    wavelength = ch.get('wavelength', 'N/A')
                    if wavelength != 'N/A':
                        print(f"     Channel {idx}: {name} (λ={wavelength}nm)")
                    else:
                        print(f"     Channel {idx}: {name}")
                else:
                    print(f"     Channel {idx}: {str(ch)}")
        else:
            print("   Channel details: Not available in metadata")
            for idx in range(num_channels):
                print(f"     Channel {idx}: Unknown")
    except Exception as e:
        print(f"   Channel details: Error reading metadata - {e}")
        for idx in range(num_channels):
            print(f"     Channel {idx}: Unknown")
    
    # Print color mapping
    if ch_map:
        print("   Color mapping:")
        for color, idx in ch_map.items():
            print(f"     {color.upper()} -> Channel {idx}")
        # Additional info for Alexa 488
        if 'green' in ch_map:
            print("     Note: Alexa 488 antibody will be displayed as GREEN channel")
        if 'brightfield' in ch_map:
            print("     Note: Brightfield channel will be saved as separate grayscale MP4")
    else:
        print("   Color mapping: No specific mapping detected (will use Channel 0 as GREEN for Alexa 488 compatibility)")


def classify_channel_names(meta: dict, num_channels: int) -> dict:
    """Return mapping from semantic colors to channel indices based on metadata.

    Uses:
    - Label-based heuristics (Alexa568/488, DAPI, Brightfield, Cy3/Cy3.5, etc.)
    - If available, wavelength-based inference (from channel metadata 'wavelength')
    """
    mapping: dict[str, int] = {}

    def _parse_wavelength_nm(val) -> Optional[float]:
        """Parse wavelength-like values into numeric nm."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            w = float(val)
            return w if w > 0 else None
        if isinstance(val, str):
            import re
            m = re.search(r"(\d+(?:\.\d+)?)", val)
            if m:
                w = float(m.group(1))
                return w if w > 0 else None
        return None

    try:
        channels_meta = meta.get('channels') if isinstance(meta, dict) else None
        if not isinstance(channels_meta, (list, tuple)):
            return mapping

        wavelength_candidates: dict[int, float] = {}

        # 1) Label-based detection + collect wavelength candidates
        for idx, ch in enumerate(channels_meta):
            if idx >= num_channels:
                break

            if isinstance(ch, dict):
                name = str(ch.get('label') or ch.get('name') or ch.get('channel') or '')
                # Try a few known keys for wavelength
                for key in (
                    'wavelength', 'wavelength_nm',
                    'emission_wavelength', 'emissionWavelength', 'em_wavelength',
                    'excitation_wavelength', 'excitationWavelength', 'ex_wavelength',
                ):
                    if key in ch:
                        w_nm = _parse_wavelength_nm(ch.get(key))
                        if w_nm is not None:
                            wavelength_candidates[idx] = w_nm
                            break
            else:
                name = str(ch)

            name_l = name.lower()

            # Brightfield detection (brightfield channel)
            if (('brightfield' in name_l or 'bright field' in name_l or 'phase' in name_l or
                 'dic' in name_l or 'transmitted' in name_l or 'trans' in name_l or
                 'bright' in name_l or 'bf' in name_l or 'td' in name_l) and 'brightfield' not in mapping):
                mapping['brightfield'] = idx
                continue

            # DAPI detection (blue channel)
            if ('dapi' in name_l) and 'blue' not in mapping:
                mapping['blue'] = idx
                continue

            # Alexa 488 / GFP / FITC detection (green channel)
            if (('488' in name_l or 'alexa488' in name_l or 'alexa 488' in name_l or
                 'alexa488 antibody' in name_l or 'alexa 488 antibody' in name_l or
                 'gfap' in name_l or 'green' in name_l or 'egfp' in name_l or
                 'gfp' in name_l or 'fitc' in name_l) and 'green' not in mapping):
                mapping['green'] = idx
                continue

            # Alexa 568 / Cy3 / Rhodamine-ish detection (red channel)
            if (('568' in name_l or 'alx568' in name_l or 'alexa568' in name_l or
                 'alexa 568' in name_l or 'alexa568 antibody' in name_l or
                 'alexa 568 antibody' in name_l or 'cy3' in name_l or 'cy3.5' in name_l or
                 'cyanine3' in name_l or 'cyanine3.5' in name_l or 'rhodamine' in name_l or
                 'tubulin' in name_l or
                 'mcherry' in name_l or 'm-cherry' in name_l or
                 'tdtomato' in name_l or
                 'tritc' in name_l or 'texas red' in name_l) and 'red' not in mapping):
                mapping['red'] = idx
                continue

        # 2) Wavelength-based fill for missing colors
        def _color_from_wavelength_nm(w_nm: float) -> str:
            # Rough emission wavelength ranges:
            # - <~500 nm: blue (e.g. DAPI ~461)
            # - 500-560 nm: green (e.g. Alexa488 ~520)
            # - >=~560 nm: red (e.g. Alexa568/Cy3 ~570-620)
            if w_nm < 500:
                return 'blue'
            if w_nm < 560:
                return 'green'
            return 'red'

        missing_red = 'red' not in mapping
        missing_green = 'green' not in mapping
        missing_blue = 'blue' not in mapping

        if missing_red or missing_green or missing_blue:
            for idx, w_nm in wavelength_candidates.items():
                if idx >= num_channels:
                    continue
                inferred = _color_from_wavelength_nm(w_nm)
                if inferred == 'red' and missing_red:
                    mapping['red'] = idx
                    missing_red = False
                elif inferred == 'green' and missing_green:
                    mapping['green'] = idx
                    missing_green = False
                elif inferred == 'blue' and missing_blue:
                    mapping['blue'] = idx
                    missing_blue = False
                if not (missing_red or missing_green or missing_blue):
                    break

    except Exception:
        pass

    return mapping


def check_ffmpeg_availability() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_grayscale_mp4_with_ffmpeg(frames: list, output_path: str, fps: float, metadata_comment: Optional[str] = None) -> bool:
    """Create QuickTime-compatible grayscale MP4 using FFmpeg.
    
    Args:
        frames: List of grayscale frames (numpy arrays)
        output_path: Output MP4 file path
        fps: Frame rate
    
    Returns:
        True if successful, False otherwise
    """
    if not check_ffmpeg_availability():
        return False
    
    # Create temporary directory for frame images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save frames as PNG images (lossless)
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # Use FFmpeg to create QuickTime-compatible grayscale MP4
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',  # H.264 codec
            '-profile:v', 'baseline',  # Most compatible profile
            '-level', '3.0',  # Standard level
            '-pix_fmt', 'yuv420p',  # Required for QuickTime
            '-crf', '18',  # High quality but compatible
            '-preset', 'medium',  # Balance between speed and quality
            '-movflags', '+faststart',  # Enable fast start
        ]
        # Add metadata comment if provided (QuickTime compatible)
        if metadata_comment:
            cmd.extend([
                '-metadata', f"comment={metadata_comment}",
                '-metadata', f"description={metadata_comment}",
                '-metadata', f"title={os.path.basename(output_path)}",
                '-metadata:s:v:0', f"comment={metadata_comment}"
            ])
        # Use MOV container for better macOS metadata compatibility if output is .mov
        if output_path.lower().endswith('.mov'):
            cmd.extend(['-f', 'mov', output_path])
        else:
            cmd.extend(['-f', 'mp4', output_path])  # Force MP4 container and output path
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}", file=sys.stderr)
            return False


def create_high_quality_mp4_with_ffmpeg(frames: list, output_path: str, fps: float, metadata_comment: Optional[str] = None) -> bool:
    """Create QuickTime-compatible MP4 using FFmpeg with high quality.
    
    Uses H.264 baseline profile with yuv420p pixel format for maximum
    compatibility with QuickTime Player and other Apple applications.
    
    Args:
        frames: List of BGR frames (numpy arrays)
        output_path: Output MP4 file path
        fps: Frame rate
    
    Returns:
        True if successful, False otherwise
    """
    if not check_ffmpeg_availability():
        return False
    
    # Create temporary directory for frame images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save frames as PNG images (lossless)
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # Use FFmpeg to create QuickTime-compatible MP4
        # QuickTime Player compatibility settings:
        # -c:v libx264: H.264 codec (QuickTime standard)
        # -profile:v baseline: Most compatible profile
        # -level 3.0: Standard level for compatibility
        # -pix_fmt yuv420p: Required for QuickTime compatibility
        # -movflags +faststart: Enable fast start for web streaming
        # -crf 18: High quality but compatible (instead of lossless)
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',  # H.264 codec
            '-profile:v', 'baseline',  # Most compatible profile
            '-level', '3.0',  # Standard level
            '-pix_fmt', 'yuv420p',  # Required for QuickTime
            '-crf', '18',  # High quality but compatible
            '-preset', 'medium',  # Balance between speed and quality
            '-movflags', '+faststart',  # Enable fast start
        ]
        # Add metadata comment if provided (QuickTime compatible)
        if metadata_comment:
            cmd.extend([
                '-metadata', f"comment={metadata_comment}",
                '-metadata', f"description={metadata_comment}",
                '-metadata', f"title={os.path.basename(output_path)}",
                '-metadata:s:v:0', f"comment={metadata_comment}"
            ])
        # Use MOV container for better macOS metadata compatibility if output is .mov
        if output_path.lower().endswith('.mov'):
            cmd.extend(['-f', 'mov', output_path])
        else:
            cmd.extend(['-f', 'mp4', output_path])  # Force MP4 container and output path
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}", file=sys.stderr)
            return False


def ensure_grayscale_video_writer(path: str, fps: float, frame_size: Tuple[int, int], preferred_codec: Optional[str]) -> cv2.VideoWriter:
    """Create a VideoWriter for grayscale videos with QuickTime-compatible codecs."""
    width, height = frame_size
    candidates = [preferred_codec] if preferred_codec else []
    # QuickTime-compatible codecs (ordered by compatibility preference)
    candidates += ["avc1", "H264", "mp4v", "X264"]
    for codec in candidates:
        if not codec:
            continue
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Create VideoWriter for grayscale (single channel)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=False)
        if writer.isOpened():
            # Set quality parameters for QuickTime compatibility
            try:
                # Use moderate quality settings for better compatibility
                writer.set(cv2.CAP_PROP_QUALITY, 90)  # High quality but compatible
                writer.set(cv2.CAP_PROP_COMPRESSION, 1)  # Moderate compression
            except Exception:
                # Some codecs may not support these parameters
                pass
            return writer
    raise RuntimeError("Failed to create a grayscale VideoWriter for path: %s" % path)


def ensure_video_writer(path: str, fps: float, frame_size: Tuple[int, int], preferred_codec: Optional[str]) -> cv2.VideoWriter:
    """Create a VideoWriter with QuickTime-compatible codecs."""
    width, height = frame_size
    candidates = [preferred_codec] if preferred_codec else []
    # QuickTime-compatible codecs (ordered by compatibility preference)
    candidates += ["avc1", "H264", "mp4v", "X264"]
    for codec in candidates:
        if not codec:
            continue
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Create VideoWriter with QuickTime-compatible settings
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            # Set quality parameters for QuickTime compatibility
            try:
                # Use moderate quality settings for better compatibility
                writer.set(cv2.CAP_PROP_QUALITY, 90)  # High quality but compatible
                writer.set(cv2.CAP_PROP_COMPRESSION, 1)  # Moderate compression
            except Exception:
                # Some codecs may not support these parameters
                pass
            return writer
    raise RuntimeError("Failed to create a VideoWriter for path: %s" % path)


def set_mp4_comment_with_ffmpeg(path: str, comment: str) -> bool:
    """Set MP4 'comment' tag by remuxing with FFmpeg (codec copy)."""
    if not check_ffmpeg_availability():
        return False
    try:
        tmp_path = path + ".tmp_meta.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-i', path,
            '-map', '0',
            '-c', 'copy',
            '-metadata', f"comment={comment}",
            '-metadata', f"description={comment}",
            '-metadata', f"title={os.path.basename(path)}",
            '-metadata:s:v:0', f"comment={comment}",
            tmp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        # Replace original
        os.replace(tmp_path, path)
        return True
    except subprocess.CalledProcessError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        sys.stderr.write(f"FFmpeg metadata set failed for {os.path.basename(path)}: {e}\n")
        return False


def generate_output_paths(base_path: str, has_brightfield: bool) -> Tuple[str, Optional[str]]:
    """Generate output paths for color and optional brightfield grayscale MP4s.
    
    Args:
        base_path: Base output path (e.g., "/path/to/file.mp4")
        has_brightfield: Whether brightfield channel exists
    
    Returns:
        Tuple of (color_output_path, brightfield_output_path)
        brightfield_output_path is None if has_brightfield is False
    """
    if not has_brightfield:
        return base_path, None
    
    # Generate brightfield path by inserting "_brightfield" before extension
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    
    brightfield_path = os.path.join(base_dir, f"{name_without_ext}_brightfield{ext}")
    return base_path, brightfield_path


def with_extension(path: str, ext: str) -> str:
    """Replace file extension (ext may include or omit the leading dot)."""
    root, _ = os.path.splitext(path)
    if not ext.startswith("."):
        ext = "." + ext
    return root + ext


def find_existing_media_output(base_mp4_path: str) -> Optional[str]:
    """Return existing .mp4 or .jpg sibling for a nominal MP4 output path."""
    for ext in (".mp4", ".jpg", ".jpeg"):
        candidate = with_extension(base_mp4_path, ext)
        if os.path.exists(candidate):
            return candidate
    return None


def save_frame_as_jpg(frame: np.ndarray, path: str, quality: int = 95) -> None:
    """Save a BGR color or grayscale uint8 frame as JPEG."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if not cv2.imwrite(path, frame, params):
        raise RuntimeError(f"Failed to write JPEG: {path}")


def write_grayscale_sequence(
    frames: list,
    out_path: str,
    fps: float,
    use_ffmpeg: bool,
    preferred_codec: Optional[str],
    metadata_comment: Optional[str] = None,
    label: str = "grayscale",
) -> Optional[str]:
    """
    Write grayscale frames as MP4, or as JPEG when only one frame is present.
    Returns the path actually written.
    """
    if not frames:
        return None
    if len(frames) <= 1:
        jpg_path = with_extension(out_path, ".jpg")
        print(f"Single frame detected — saving {label} JPEG: {os.path.basename(jpg_path)}")
        save_frame_as_jpg(frames[0], jpg_path)
        print(f"✅ {label} JPEG created: {os.path.basename(jpg_path)}")
        return jpg_path

    print(f"Creating {label} MP4: {os.path.basename(out_path)}")
    if use_ffmpeg:
        if create_grayscale_mp4_with_ffmpeg(frames, out_path, fps, metadata_comment=metadata_comment):
            print(f"✅ {label} MP4 created with FFmpeg: {os.path.basename(out_path)}")
            return out_path
        print("⚠️  FFmpeg not available or failed, falling back to OpenCV VideoWriter")

    h, w = frames[0].shape[:2]
    writer = ensure_grayscale_video_writer(out_path, fps, (w, h), preferred_codec)
    for frame in frames:
        writer.write(frame)
    writer.release()
    if metadata_comment:
        set_mp4_comment_with_ffmpeg(out_path, metadata_comment)
    print(f"✅ {label} MP4 created with OpenCV: {os.path.basename(out_path)}")
    return out_path


def restrict_channel_map(
    ch_map: dict,
    num_channels: int,
    tokens: List[str],
    single_channel: bool,
) -> Tuple[dict, Optional[int]]:
    """Keep only the channels named by the still-image --channels flag.

    Color names use the classified map. Numeric tokens are channel indices.
    single_channel returns that index for a grayscale write.
    """
    if not tokens:
        return ch_map, None
    resolved: List[Tuple[str, int]] = []
    fallback_color = {0: "blue", 1: "green", 2: "red"}
    for token in tokens:
        if token in ("red", "green", "blue"):
            idx = ch_map.get(token)
            if isinstance(idx, int) and 0 <= idx < num_channels:
                resolved.append((token, idx))
        elif token.isdigit():
            idx = int(token)
            if 0 <= idx < num_channels:
                color = next((name for name in ("red", "green", "blue") if ch_map.get(name) == idx), None)
                if color is None:
                    color = fallback_color.get(idx, "green")
                resolved.append((color, idx))
    if single_channel:
        if not resolved:
            return {}, None
        return ch_map, resolved[0][1]
    return {color: idx for color, idx in resolved}, None


def frames_as_single_color(frames: list, bgr_index: int) -> list:
    """Place grayscale frames on one BGR plane (0=blue, 1=green, 2=red)."""
    color_frames = []
    for gray in frames:
        if getattr(gray, "ndim", 0) == 2:
            bgr = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            bgr[:, :, bgr_index] = gray
            color_frames.append(bgr)
        else:
            color_frames.append(gray)
    return color_frames


def write_converted_media(
    *,
    all_frames: list,
    brightfield_frames: list,
    green_only_frames: list,
    color_out_path: str,
    brightfield_out_path: Optional[str],
    green_only: bool,
    fps: float,
    use_ffmpeg: bool,
    preferred_codec: Optional[str],
    nd2_path: str,
    side_um: Optional[float],
    base_comment: str,
    red_only_frames: Optional[list] = None,
    red_only: bool = False,
) -> List[str]:
    """
    Write color / brightfield / single-channel outputs.
    Single-frame sequences are saved as JPEG; multi-frame as MP4.
    Returns list of created file paths.
    """
    created: List[str] = []

    if all_frames and getattr(all_frames[0], "ndim", 0) == 2:
        gray_path = write_grayscale_sequence(
            all_frames,
            color_out_path,
            fps,
            use_ffmpeg,
            preferred_codec,
            metadata_comment=base_comment,
            label="Single channel",
        )
        return [gray_path] if gray_path else []

    if green_only or red_only:
        source = green_only_frames if green_only else (red_only_frames or [])
        if not source:
            return created
        # Same intensity as that plane of the composite, shown on black.
        bgr_index = 1 if green_only else 2
        note = "Green channel only" if green_only else "Red channel only"
        all_frames = frames_as_single_color(source, bgr_index)
        brightfield_frames = []
        brightfield_out_path = None
        base_comment = build_comment_tag(nd2_path, side_um, float(fps), note=note)

    if not all_frames:
        return created

    single_frame = len(all_frames) <= 1
    if single_frame:
        color_jpg = with_extension(color_out_path, ".jpg")
        print(f"Single frame detected — saving JPEG instead of MP4: {os.path.basename(color_jpg)}")
        save_frame_as_jpg(all_frames[0], color_jpg)
        print(f"✅ JPEG created: {os.path.basename(color_jpg)}")
        created.append(color_jpg)

        if brightfield_out_path and brightfield_frames:
            bf_jpg = with_extension(brightfield_out_path, ".jpg")
            print(f"Saving brightfield JPEG: {os.path.basename(bf_jpg)}")
            save_frame_as_jpg(brightfield_frames[0], bf_jpg)
            print(f"✅ Brightfield JPEG created: {os.path.basename(bf_jpg)}")
            created.append(bf_jpg)
        return created

    # Multi-frame: write MP4s
    if use_ffmpeg:
        print(f"Creating QuickTime-compatible MP4 with FFmpeg: {os.path.basename(color_out_path)}")
        if create_high_quality_mp4_with_ffmpeg(all_frames, color_out_path, fps, metadata_comment=base_comment):
            print(f"✅ QuickTime-compatible MP4 created with FFmpeg: {os.path.basename(color_out_path)}")
            created.append(color_out_path)
        else:
            print("⚠️  FFmpeg not available or failed, falling back to OpenCV VideoWriter")
            print(f"Creating MP4 with OpenCV VideoWriter: {os.path.basename(color_out_path)}")
            h, w = all_frames[0].shape[:2]
            writer = ensure_video_writer(color_out_path, fps, (w, h), preferred_codec)
            for frame in all_frames:
                writer.write(frame)
            writer.release()
            set_mp4_comment_with_ffmpeg(color_out_path, base_comment)
            print(f"✅ MP4 created with OpenCV: {os.path.basename(color_out_path)}")
            created.append(color_out_path)
    else:
        print(f"Creating MP4 with OpenCV VideoWriter: {os.path.basename(color_out_path)}")
        h, w = all_frames[0].shape[:2]
        writer = ensure_video_writer(color_out_path, fps, (w, h), preferred_codec)
        for frame in all_frames:
            writer.write(frame)
        writer.release()
        set_mp4_comment_with_ffmpeg(color_out_path, base_comment)
        print(f"✅ MP4 created with OpenCV: {os.path.basename(color_out_path)}")
        created.append(color_out_path)

    if brightfield_out_path and brightfield_frames:
        bf_comment = build_comment_tag(nd2_path, side_um, float(fps), note="Brightfield")
        bf_path = write_grayscale_sequence(
            brightfield_frames,
            brightfield_out_path,
            fps,
            use_ffmpeg,
            preferred_codec,
            metadata_comment=bf_comment,
            label="Brightfield grayscale",
        )
        if bf_path:
            created.append(bf_path)

    return created


def convert_nd2_to_mp4(
    nd2_path: str,
    out_path: str,
    default_fps: float,
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
    preferred_codec: Optional[str],
    use_ffmpeg: bool = False,
    green_only: bool = False,
    red_only: bool = False,
    selected_channels: Optional[List[str]] = None,
    single_channel: bool = False,
) -> List[str]:
    """Convert a single ND2 file to MP4 (or JPEG if only one frame) with red-black coloring and CLAHE.

    Returns:
        List of created output paths (.mp4 and/or .jpg).
    """
    selection_tokens = [str(token).strip().lower() for token in (selected_channels or []) if str(token).strip()]
    if not selection_tokens and green_only:
        selection_tokens = ["green"]
    elif not selection_tokens and red_only:
        selection_tokens = ["red"]
    if selection_tokens:
        green_only = False
        red_only = False
    single_channel_active = bool(single_channel and selection_tokens)

    if _HAS_ND2READER:
        with ND2Reader(nd2_path) as images:  # type: ignore
            # Configure axes
            images.iter_axes = "t"
            if "c" in getattr(images, "sizes", {}):
                images.bundle_axes = "yxc"
            else:
                images.bundle_axes = "yx"

            meta = getattr(images, "metadata", {}) or {}
            fps = detect_fps_from_metadata(meta) or float(default_fps)

            # Determine channel semantics if multi-channel
            num_channels = 1
            sample_shape = np.asarray(images[0])
            if sample_shape.ndim == 3:
                num_channels = sample_shape.shape[2]
            ch_map = classify_channel_names(getattr(images, "metadata", {}) or {}, num_channels)
            ch_map, gray_channel_index = restrict_channel_map(
                ch_map, num_channels, selection_tokens, single_channel_active
            )
            channel_selection_active = bool(selection_tokens)
            
            # Check if brightfield channel exists
            has_brightfield = 'brightfield' in ch_map and not channel_selection_active
            
            # Print channel information
            print_channel_info(nd2_path, meta, num_channels, ch_map)
            
            # Generate output paths
            color_out_path, brightfield_out_path = generate_output_paths(out_path, has_brightfield)

            # Determine field-of-view side length in µm for tagging
            # Try sizes from ND2Reader if available, else from first frame
            try:
                sizes = getattr(images, 'sizes', {}) or {}
                h_px = int(sizes.get('y')) if 'y' in sizes else None
                w_px = int(sizes.get('x')) if 'x' in sizes else None
            except Exception:
                h_px, w_px = None, None
            if h_px is None or w_px is None:
                sample = np.asarray(images[0])
                if sample.ndim == 3:
                    h_px, w_px = sample.shape[:2]
                else:
                    h_px, w_px = sample.shape[0], sample.shape[1]
            side_um = extract_side_length_um(meta, int(h_px or 0), int(w_px or 0))
            base_comment = build_comment_tag(nd2_path, side_um, float(fps))

            # First pass: compute per-channel global histogram percentiles
            per_channel_samples = {i: [] for i in range(num_channels)}
            num_frames = len(images)
            for idx in range(num_frames):
                arr = to_native_endian(np.asarray(images[idx]))
                if arr.ndim == 3 and arr.shape[2] > 1:
                    for ch in range(arr.shape[2]):
                        ch_arr = arr[:, :, ch]
                        if ch_arr.size > 0:
                            step = max(1, int(ch_arr.size / 100))
                            sample = ch_arr.ravel()[::step]
                            if sample.size > 0:
                                per_channel_samples[ch].append(sample[:50000])
                else:
                    gray = arr if arr.ndim == 2 else np.squeeze(arr)
                    if gray.size > 0:
                        step = max(1, int(gray.size / 100))
                        sample = gray.ravel()[::step]
                        if sample.size > 0:
                            per_channel_samples[0].append(sample[:50000])

            # Compute percentile ranges per channel
            lower_pct = float(os.environ.get("ND2_NORM_LOWER_PCT", 1.0))
            upper_pct = float(os.environ.get("ND2_NORM_UPPER_PCT", 99.5))
            lower_pct = max(0.0, min(100.0, lower_pct))
            upper_pct = max(0.0, min(100.0, upper_pct))
            if upper_pct <= lower_pct:
                upper_pct = min(100.0, lower_pct + 0.1)
            clip_ranges = {}
            for ch, samples in per_channel_samples.items():
                if len(samples) == 0:
                    clip_ranges[ch] = (0.0, 1.0)
                    continue
                sample_all = np.concatenate(samples).astype(np.float32)
                cmin = float(np.percentile(sample_all, lower_pct))
                cmax = float(np.percentile(sample_all, upper_pct))
                if cmax <= cmin:
                    cmax = cmin + 1.0
                clip_ranges[ch] = (cmin, cmax)

            # Process all frames
            all_frames = []
            brightfield_frames = []
            green_only_frames = []
            red_only_frames = []
            for idx in range(len(images)):
                frame = to_native_endian(np.asarray(images[idx]))
                if gray_channel_index is not None:
                    if frame.ndim == 3 and gray_channel_index < frame.shape[2]:
                        src = frame[:, :, gray_channel_index]
                        clip_idx = gray_channel_index
                    else:
                        src = frame if frame.ndim == 2 else np.squeeze(frame)
                        clip_idx = 0
                    cmin, cmax = clip_ranges.get(clip_idx, (0.0, 1.0))
                    gray = apply_clahe_if_needed(
                        normalize_to_uint8_with_clip(src, cmin, cmax),
                        use_clahe,
                        clip_limit,
                        tile_grid,
                    )
                    all_frames.append(gray)
                    continue
                # Initialize variables at the start of each iteration
                brightfield_gray = None
                green_gray = None
                red_gray = None
                bgr = None
                
                if frame.ndim == 3 and frame.shape[2] > 1:
                    channel_to_u8 = {}
                    
                    # Process brightfield channel separately if it exists
                    if has_brightfield and 'brightfield' in ch_map:
                        bf_idx = ch_map['brightfield']
                        if bf_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(bf_idx, (0.0, 1.0))
                            brightfield_gray = normalize_to_uint8_with_clip(frame[:, :, bf_idx], cmin, cmax)
                            brightfield_gray = apply_clahe_if_needed(brightfield_gray, use_clahe, clip_limit, tile_grid)
                    
                    # If green-only mode, extract green channel as grayscale
                    if green_only:
                        green_idx = ch_map.get('green')
                        if isinstance(green_idx, int) and green_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(green_idx, (0.0, 1.0))
                            green_gray = normalize_to_uint8_with_clip(frame[:, :, green_idx], cmin, cmax)
                            green_gray = apply_clahe_if_needed(green_gray, use_clahe, clip_limit, tile_grid)
                        elif frame.shape[2] > 0:
                            # Fallback: use first channel if green not detected
                            cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                            green_gray = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                            green_gray = apply_clahe_if_needed(green_gray, use_clahe, clip_limit, tile_grid)

                    # If red-only mode, extract red channel as grayscale
                    if red_only:
                        red_idx = ch_map.get('red')
                        if isinstance(red_idx, int) and red_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(red_idx, (0.0, 1.0))
                            red_gray = normalize_to_uint8_with_clip(frame[:, :, red_idx], cmin, cmax)
                            red_gray = apply_clahe_if_needed(red_gray, use_clahe, clip_limit, tile_grid)
                        elif frame.shape[2] > 1:
                            cmin, cmax = clip_ranges.get(1, (0.0, 1.0))
                            red_gray = normalize_to_uint8_with_clip(frame[:, :, 1], cmin, cmax)
                            red_gray = apply_clahe_if_needed(red_gray, use_clahe, clip_limit, tile_grid)
                    
                    # Process color channels (excluding brightfield) - skip in single-channel mode
                    if not green_only and not red_only:
                        for color, idx_map in (('blue', ch_map.get('blue')), ('green', ch_map.get('green')), ('red', ch_map.get('red'))):
                            if isinstance(idx_map, int) and idx_map < frame.shape[2]:
                                cmin, cmax = clip_ranges.get(idx_map, (0.0, 1.0))
                                channel_to_u8[color] = normalize_to_uint8_with_clip(frame[:, :, idx_map], cmin, cmax)
                        
                        # Fallback: if no names detected, try to map channels based on common patterns
                        if not channel_to_u8 and frame.shape[2] > 0 and not channel_selection_active:
                            # For single channel, use green (more common for Alexa 488)
                            if frame.shape[2] == 1:
                                cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                                channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                            else:
                                # For multiple channels, map first channel to green
                                cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                                channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                        
                        if channel_to_u8:
                            bgr = build_rgb_from_channels(channel_to_u8, use_clahe, clip_limit, tile_grid)
                        else:
                            bgr = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                else:
                    gray = frame if frame.ndim == 2 else np.squeeze(frame)
                    cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                    gray_u8 = normalize_to_uint8_with_clip(gray, cmin, cmax)
                    
                    if green_only:
                        # In green-only mode, use the single channel as green grayscale
                        green_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        bgr = None
                    elif red_only:
                        red_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        bgr = None
                    else:
                        bgr = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
                        # Map the single channel to the detected semantic color (red/green/blue).
                        # Previously this was hard-coded to green, which breaks "red-only" datasets
                        # when ND2 reports only one channel.
                        color_idx = 1  # default to green for backward compatibility
                        if isinstance(ch_map, dict):
                            if ch_map.get('red') is not None:
                                color_idx = 2
                            elif ch_map.get('blue') is not None:
                                color_idx = 0
                            elif ch_map.get('green') is not None:
                                color_idx = 1
                        bgr[:, :, color_idx] = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        # For single channel, also use as brightfield if no brightfield detected
                        if not has_brightfield:
                            brightfield_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                
                if not green_only and not red_only:
                    all_frames.append(bgr)
                if brightfield_gray is not None:
                    brightfield_frames.append(brightfield_gray)
                if green_gray is not None:
                    green_only_frames.append(green_gray)
                if red_gray is not None:
                    red_only_frames.append(red_gray)
            
            return write_converted_media(
                all_frames=all_frames,
                brightfield_frames=brightfield_frames,
                green_only_frames=green_only_frames,
                red_only_frames=red_only_frames,
                color_out_path=color_out_path,
                brightfield_out_path=brightfield_out_path,
                green_only=green_only,
                red_only=red_only,
                fps=fps,
                use_ffmpeg=use_ffmpeg,
                preferred_codec=preferred_codec,
                nd2_path=nd2_path,
                side_um=side_um,
                base_comment=base_comment,
            )

    # Fallback to PIMS if nd2reader is unavailable
    if _HAS_PIMS:
        with pims.open(nd2_path) as images:  # type: ignore
            try:
                meta = getattr(images, "metadata", {}) or {}
            except Exception:
                meta = {}
            fps = detect_fps_from_metadata(meta) or float(default_fps)

            # Determine channel semantics if multi-channel
            num_channels = 1
            sample_shape = np.asarray(images[0])
            if sample_shape.ndim == 3:
                num_channels = sample_shape.shape[2]
            ch_map = classify_channel_names(meta, num_channels)
            ch_map, gray_channel_index = restrict_channel_map(
                ch_map, num_channels, selection_tokens, single_channel_active
            )
            channel_selection_active = bool(selection_tokens)
            
            # Check if brightfield channel exists
            has_brightfield = 'brightfield' in ch_map and not channel_selection_active
            
            # Print channel information
            print_channel_info(nd2_path, meta, num_channels, ch_map)
            
            # Generate output paths
            color_out_path, brightfield_out_path = generate_output_paths(out_path, has_brightfield)

            # Determine field-of-view side length in µm for tagging (PIMS branch)
            try:
                first_frame = np.asarray(images[0])
                if first_frame.ndim == 3:
                    h_px, w_px = first_frame.shape[:2]
                else:
                    h_px, w_px = first_frame.shape[0], first_frame.shape[1]
            except Exception:
                h_px, w_px = 0, 0
            side_um = extract_side_length_um(meta, int(h_px), int(w_px))
            base_comment = build_comment_tag(nd2_path, side_um, float(fps))

            # First pass: compute per-channel global histogram percentiles
            per_channel_samples = {i: [] for i in range(num_channels)}
            num_frames = len(images)
            for idx in range(num_frames):
                arr = to_native_endian(np.asarray(images[idx]))
                if arr.ndim == 3 and arr.shape[2] > 1:
                    for ch in range(arr.shape[2]):
                        ch_arr = arr[:, :, ch]
                        if ch_arr.size > 0:
                            step = max(1, int(ch_arr.size / 100))
                            sample = ch_arr.ravel()[::step]
                            if sample.size > 0:
                                per_channel_samples[ch].append(sample[:50000])
                else:
                    gray = arr if arr.ndim == 2 else np.squeeze(arr)
                    if gray.size > 0:
                        step = max(1, int(gray.size / 100))
                        sample = gray.ravel()[::step]
                        if sample.size > 0:
                            per_channel_samples[0].append(sample[:50000])

            lower_pct = float(os.environ.get("ND2_NORM_LOWER_PCT", 1.0))
            upper_pct = float(os.environ.get("ND2_NORM_UPPER_PCT", 99.5))
            lower_pct = max(0.0, min(100.0, lower_pct))
            upper_pct = max(0.0, min(100.0, upper_pct))
            if upper_pct <= lower_pct:
                upper_pct = min(100.0, lower_pct + 0.1)
            clip_ranges = {}
            for ch, samples in per_channel_samples.items():
                if len(samples) == 0:
                    clip_ranges[ch] = (0.0, 1.0)
                    continue
                sample_all = np.concatenate(samples).astype(np.float32)
                cmin = float(np.percentile(sample_all, lower_pct))
                cmax = float(np.percentile(sample_all, upper_pct))
                if cmax <= cmin:
                    cmax = cmin + 1.0
                clip_ranges[ch] = (cmin, cmax)

            # Process all frames
            all_frames = []
            brightfield_frames = []
            green_only_frames = []
            red_only_frames = []
            for idx in range(len(images)):
                frame = to_native_endian(np.asarray(images[idx]))
                if gray_channel_index is not None:
                    if frame.ndim == 3 and gray_channel_index < frame.shape[2]:
                        src = frame[:, :, gray_channel_index]
                        clip_idx = gray_channel_index
                    else:
                        src = frame if frame.ndim == 2 else np.squeeze(frame)
                        clip_idx = 0
                    cmin, cmax = clip_ranges.get(clip_idx, (0.0, 1.0))
                    gray = apply_clahe_if_needed(
                        normalize_to_uint8_with_clip(src, cmin, cmax),
                        use_clahe,
                        clip_limit,
                        tile_grid,
                    )
                    all_frames.append(gray)
                    continue
                # Initialize variables at the start of each iteration
                brightfield_gray = None
                green_gray = None
                red_gray = None
                bgr = None
                
                if frame.ndim == 3 and frame.shape[2] > 1:
                    channel_to_u8 = {}
                    
                    # Process brightfield channel separately if it exists
                    if has_brightfield and 'brightfield' in ch_map:
                        bf_idx = ch_map['brightfield']
                        if bf_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(bf_idx, (0.0, 1.0))
                            brightfield_gray = normalize_to_uint8_with_clip(frame[:, :, bf_idx], cmin, cmax)
                            brightfield_gray = apply_clahe_if_needed(brightfield_gray, use_clahe, clip_limit, tile_grid)
                    
                    # If green-only mode, extract green channel as grayscale
                    if green_only:
                        green_idx = ch_map.get('green')
                        if isinstance(green_idx, int) and green_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(green_idx, (0.0, 1.0))
                            green_gray = normalize_to_uint8_with_clip(frame[:, :, green_idx], cmin, cmax)
                            green_gray = apply_clahe_if_needed(green_gray, use_clahe, clip_limit, tile_grid)
                        elif frame.shape[2] > 0:
                            # Fallback: use first channel if green not detected
                            cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                            green_gray = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                            green_gray = apply_clahe_if_needed(green_gray, use_clahe, clip_limit, tile_grid)

                    # If red-only mode, extract red channel as grayscale
                    if red_only:
                        red_idx = ch_map.get('red')
                        if isinstance(red_idx, int) and red_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(red_idx, (0.0, 1.0))
                            red_gray = normalize_to_uint8_with_clip(frame[:, :, red_idx], cmin, cmax)
                            red_gray = apply_clahe_if_needed(red_gray, use_clahe, clip_limit, tile_grid)
                        elif frame.shape[2] > 1:
                            cmin, cmax = clip_ranges.get(1, (0.0, 1.0))
                            red_gray = normalize_to_uint8_with_clip(frame[:, :, 1], cmin, cmax)
                            red_gray = apply_clahe_if_needed(red_gray, use_clahe, clip_limit, tile_grid)
                    
                    # Process color channels (excluding brightfield) - skip in single-channel mode
                    if not green_only and not red_only:
                        for color, idx_map in (('blue', ch_map.get('blue')), ('green', ch_map.get('green')), ('red', ch_map.get('red'))):
                            if isinstance(idx_map, int) and idx_map < frame.shape[2]:
                                cmin, cmax = clip_ranges.get(idx_map, (0.0, 1.0))
                                channel_to_u8[color] = normalize_to_uint8_with_clip(frame[:, :, idx_map], cmin, cmax)
                        
                        # Fallback: if no names detected, try to map channels based on common patterns
                        if not channel_to_u8 and frame.shape[2] > 0 and not channel_selection_active:
                            # For single channel, use green (more common for Alexa 488)
                            if frame.shape[2] == 1:
                                cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                                channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                            else:
                                # For multiple channels, map first channel to green
                                cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                                channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                        
                        if channel_to_u8:
                            bgr = build_rgb_from_channels(channel_to_u8, use_clahe, clip_limit, tile_grid)
                        else:
                            bgr = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                else:
                    gray = frame if frame.ndim == 2 else np.squeeze(frame)
                    cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                    gray_u8 = normalize_to_uint8_with_clip(gray, cmin, cmax)
                    
                    if green_only:
                        # In green-only mode, use the single channel as green grayscale
                        green_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        bgr = None
                    elif red_only:
                        red_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        bgr = None
                    else:
                        bgr = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
                        # Map the single channel to the detected semantic color (red/green/blue).
                        # Previously this was hard-coded to green, which breaks "red-only" datasets
                        # when ND2 reports only one channel.
                        color_idx = 1  # default to green for backward compatibility
                        if isinstance(ch_map, dict):
                            if ch_map.get('red') is not None:
                                color_idx = 2
                            elif ch_map.get('blue') is not None:
                                color_idx = 0
                            elif ch_map.get('green') is not None:
                                color_idx = 1
                        bgr[:, :, color_idx] = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        # For single channel, also use as brightfield if no brightfield detected
                        if not has_brightfield:
                            brightfield_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                
                if not green_only and not red_only:
                    all_frames.append(bgr)
                if brightfield_gray is not None:
                    brightfield_frames.append(brightfield_gray)
                if green_gray is not None:
                    green_only_frames.append(green_gray)
                if red_gray is not None:
                    red_only_frames.append(red_gray)
            
            return write_converted_media(
                all_frames=all_frames,
                brightfield_frames=brightfield_frames,
                green_only_frames=green_only_frames,
                red_only_frames=red_only_frames,
                color_out_path=color_out_path,
                brightfield_out_path=brightfield_out_path,
                green_only=green_only,
                red_only=red_only,
                fps=fps,
                use_ffmpeg=use_ffmpeg,
                preferred_codec=preferred_codec,
                nd2_path=nd2_path,
                side_um=side_um,
                base_comment=base_comment,
            )

    raise SystemExit(
        "No ND2 backend available. Please install at least one: 'pip install nd2reader pims'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ND2 movies to MP4 (or JPEG if single-frame) with red-black coloring and CLAHE."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.abspath(os.getcwd()),
        help="Root directory to recursively scan for ND2 files (default: current directory)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Fallback FPS if not found in ND2 metadata (default: 10)",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=3.0,
        help="CLAHE clip limit (default: 3.0)",
    )
    parser.add_argument(
        "--tile-grid",
        type=int,
        default=8,
        help="CLAHE tile grid size (N -> NxN) (default: 8)",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE contrast enhancement (may reduce flicker)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="avc1",
        help="Preferred fourcc codec for QuickTime-compatible MP4 (default: avc1; fallbacks: H264, mp4v, X264)",
    )
    parser.add_argument(
        "--norm-mode",
        choices=["percentile", "minmax"],
        default="percentile",
        help="Global normalization mode: percentile (robust) or minmax (legacy)",
    )
    parser.add_argument(
        "--lower-pct",
        type=float,
        default=1.0,
        help="Lower percentile for histogram clipping when norm-mode=percentile (default: 1.0)",
    )
    parser.add_argument(
        "--upper-pct",
        type=float,
        default=99.5,
        help="Upper percentile for histogram clipping when norm-mode=percentile (default: 99.5)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip existing MP4/JPEG files instead of overwriting them (default: overwrite)",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Disable FFmpeg and use OpenCV VideoWriter instead (FFmpeg is used by default for high-quality encoding)",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Channels to keep, same as still images: red,green,blue or indices (0,1,2)",
    )
    parser.add_argument(
        "--single-channel",
        action="store_true",
        help="Write the one channel given by --channels as grayscale",
    )
    parser.add_argument(
        "--green-only",
        action="store_true",
        help="Alias of --channels green",
    )
    parser.add_argument(
        "--red-only",
        action="store_true",
        help="Alias of --channels red",
    )

    args = parser.parse_args()
    if args.green_only and args.red_only:
        sys.stderr.write("ERROR: --green-only and --red-only cannot be used together\n")
        sys.exit(1)
    if args.channels and (args.green_only or args.red_only):
        sys.stderr.write("ERROR: use either --channels or --green-only/--red-only\n")
        sys.exit(1)
    selected_channels = [token.strip() for token in args.channels.split(",") if token.strip()] or None

    root_dir = os.path.abspath(args.input)
    if not os.path.isdir(root_dir):
        sys.stderr.write(f"Input directory does not exist: {root_dir}\n")
        sys.exit(1)

    if not _HAS_ND2READER and not _HAS_PIMS:
        sys.stderr.write(
            "ERROR: No ND2 backend available. Please install: pip install nd2reader pims\n"
        )
        sys.exit(2)
    
    # Check FFmpeg availability and inform user
    use_ffmpeg = not args.no_ffmpeg
    if use_ffmpeg:
        if check_ffmpeg_availability():
            print("✅ FFmpeg detected - will use QuickTime-compatible encoding with H.264 baseline profile")
        else:
            print("⚠️  FFmpeg not found - falling back to OpenCV VideoWriter")
            print("   To install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
            use_ffmpeg = False
    else:
        print("📹 Using OpenCV VideoWriter (FFmpeg disabled by --no-ffmpeg)")

    found_any = False
    use_clahe = not args.no_clahe
    for subdir, nd2_path in find_nd2_files_recursively(root_dir):
        found_any = True
        base = os.path.splitext(os.path.basename(nd2_path))[0]
        out_path = os.path.join(subdir, f"{base}.mp4")
        existing = find_existing_media_output(out_path)
        if existing is not None and args.no_overwrite:
            print(f"Skip existing: {existing}")
            continue
        print(f"Converting: {nd2_path}\n  -> {out_path} (JPEG if single frame)")
        try:
            # Export percentiles via env for the converter
            if args.norm_mode == "percentile":
                os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
                os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
            else:
                os.environ.pop("ND2_NORM_LOWER_PCT", None)
                os.environ.pop("ND2_NORM_UPPER_PCT", None)

            created = convert_nd2_to_mp4(
                nd2_path=nd2_path,
                out_path=out_path,
                default_fps=float(args.fps),
                use_clahe=use_clahe,
                clip_limit=float(args.clip_limit),
                tile_grid=int(args.tile_grid),
                preferred_codec=str(args.codec) if args.codec else None,
                use_ffmpeg=use_ffmpeg,
                green_only=args.green_only,
                red_only=args.red_only,
                selected_channels=selected_channels,
                single_channel=bool(args.single_channel),
            )
            for path in created:
                print(f"  Wrote: {path}")
        except Exception as exc:
            print(f"ERROR converting {nd2_path}: {exc}")

    if not found_any:
        print("No .nd2 files found in:", root_dir, "(searched recursively)")


if __name__ == "__main__":
    main()
