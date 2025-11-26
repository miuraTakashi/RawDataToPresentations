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
- Creates separate grayscale MP4 for brightfield channels when detected

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
from typing import Generator, Optional, Tuple

import numpy as np

# Optional imports; handled dynamically
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
    """Return mapping from semantic colors to channel indices based on metadata names.

    Tries to detect Alexa 568 -> red, Alexa 488 -> green, DAPI -> blue, Brightfield -> brightfield.
    If names are unavailable, returns empty mapping.
    """
    mapping: dict[str, int] = {}
    try:
        channels_meta = meta.get('channels') if isinstance(meta, dict) else None
        if isinstance(channels_meta, (list, tuple)):
            for idx, ch in enumerate(channels_meta):
                if idx >= num_channels:
                    break
                # channel label extraction heuristic
                if isinstance(ch, dict):
                    name = str(ch.get('label') or ch.get('name') or ch.get('channel') or '')
                else:
                    name = str(ch)
                name_l = name.lower()
                # Brightfield detection (brightfield channel)
                if (('brightfield' in name_l or 'bright field' in name_l or 'phase' in name_l or 
                     'dic' in name_l or 'transmitted' in name_l or 'trans' in name_l or
                     'bright' in name_l or 'bf' in name_l) and 'brightfield' not in mapping):
                    mapping['brightfield'] = idx
                # DAPI detection (blue channel)
                elif 'dapi' in name_l and 'blue' not in mapping:
                    mapping['blue'] = idx
                # Alexa 488 detection (green channel) - including antibody variants
                elif (('488' in name_l or 'alexa488' in name_l or 'alexa 488' in name_l or 
                     'alexa488 antibody' in name_l or 'alexa 488 antibody' in name_l or
                     'gfap' in name_l or 'green' in name_l or 'egfp' in name_l or
                     'gfp' in name_l or 'fitc' in name_l) and 'green' not in mapping):
                    mapping['green'] = idx
                # Alexa 568 detection (red channel)
                elif (('568' in name_l or 'alx568' in name_l or 'alexa568' in name_l or 
                     'alexa 568' in name_l or 'alexa568 antibody' in name_l or 
                     'alexa 568 antibody' in name_l) and 'red' not in mapping):
                    mapping['red'] = idx
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


def create_grayscale_mp4_with_ffmpeg(frames: list, output_path: str, fps: float) -> bool:
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
            '-f', 'mp4',  # Force MP4 container
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}", file=sys.stderr)
            return False


def create_high_quality_mp4_with_ffmpeg(frames: list, output_path: str, fps: float) -> bool:
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
            '-f', 'mp4',  # Force MP4 container
            output_path
        ]
        
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


def convert_nd2_to_mp4(
    nd2_path: str,
    out_path: str,
    default_fps: float,
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
    preferred_codec: Optional[str],
    use_ffmpeg: bool = False,
) -> None:
    """Convert a single ND2 file to MP4 with red-black coloring and CLAHE."""
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
            
            # Check if brightfield channel exists
            has_brightfield = 'brightfield' in ch_map
            
            # Print channel information
            print_channel_info(nd2_path, meta, num_channels, ch_map)
            
            # Generate output paths
            color_out_path, brightfield_out_path = generate_output_paths(out_path, has_brightfield)

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
            for idx in range(len(images)):
                frame = to_native_endian(np.asarray(images[idx]))
                if frame.ndim == 3 and frame.shape[2] > 1:
                    channel_to_u8 = {}
                    brightfield_gray = None
                    
                    # Process brightfield channel separately if it exists
                    if has_brightfield and 'brightfield' in ch_map:
                        bf_idx = ch_map['brightfield']
                        if bf_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(bf_idx, (0.0, 1.0))
                            brightfield_gray = normalize_to_uint8_with_clip(frame[:, :, bf_idx], cmin, cmax)
                            brightfield_gray = apply_clahe_if_needed(brightfield_gray, use_clahe, clip_limit, tile_grid)
                    
                    # Process color channels (excluding brightfield)
                    for color, idx_map in (('blue', ch_map.get('blue')), ('green', ch_map.get('green')), ('red', ch_map.get('red'))):
                        if isinstance(idx_map, int) and idx_map < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(idx_map, (0.0, 1.0))
                            channel_to_u8[color] = normalize_to_uint8_with_clip(frame[:, :, idx_map], cmin, cmax)
                    
                    # Fallback: if no names detected, try to map channels based on common patterns
                    if not channel_to_u8 and frame.shape[2] > 0:
                        # For single channel, use green (more common for Alexa 488)
                        if frame.shape[2] == 1:
                            cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                            channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                        else:
                            # For multiple channels, map first channel to green
                            cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                            channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                    
                    bgr = build_rgb_from_channels(channel_to_u8, use_clahe, clip_limit, tile_grid)
                else:
                    gray = frame if frame.ndim == 2 else np.squeeze(frame)
                    cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                    gray_u8 = normalize_to_uint8_with_clip(gray, cmin, cmax)
                    bgr = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
                    # Use green channel instead of red for better Alexa 488 compatibility
                    bgr[:, :, 1] = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                    # For single channel, also use as brightfield if no brightfield detected
                    if not has_brightfield:
                        brightfield_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                
                all_frames.append(bgr)
                if brightfield_gray is not None:
                    brightfield_frames.append(brightfield_gray)
            
            # Create color MP4
            if use_ffmpeg:
                print(f"Creating QuickTime-compatible MP4 with FFmpeg: {os.path.basename(color_out_path)}")
                if create_high_quality_mp4_with_ffmpeg(all_frames, color_out_path, fps):
                    print(f"✅ QuickTime-compatible MP4 created with FFmpeg: {os.path.basename(color_out_path)}")
                else:
                    print("⚠️  FFmpeg not available or failed, falling back to OpenCV VideoWriter")
                    # Fallback to OpenCV VideoWriter
                    print(f"Creating MP4 with OpenCV VideoWriter: {os.path.basename(color_out_path)}")
                    h, w = all_frames[0].shape[:2]
                    writer = ensure_video_writer(color_out_path, fps, (w, h), preferred_codec)
                    for frame in all_frames:
                        writer.write(frame)
                    writer.release()
                    print(f"✅ MP4 created with OpenCV: {os.path.basename(color_out_path)}")
            else:
                # Fallback to OpenCV VideoWriter
                print(f"Creating MP4 with OpenCV VideoWriter: {os.path.basename(color_out_path)}")
                h, w = all_frames[0].shape[:2]
                writer = ensure_video_writer(color_out_path, fps, (w, h), preferred_codec)
                for frame in all_frames:
                    writer.write(frame)
                writer.release()
                print(f"✅ MP4 created with OpenCV: {os.path.basename(color_out_path)}")
            
            # Create brightfield grayscale MP4 if brightfield channel exists
            if brightfield_out_path and brightfield_frames:
                print(f"Creating brightfield grayscale MP4: {os.path.basename(brightfield_out_path)}")
                if use_ffmpeg:
                    if create_grayscale_mp4_with_ffmpeg(brightfield_frames, brightfield_out_path, fps):
                        print(f"✅ Brightfield grayscale MP4 created with FFmpeg: {os.path.basename(brightfield_out_path)}")
                    else:
                        print("⚠️  FFmpeg not available or failed, falling back to OpenCV VideoWriter")
                        # Fallback to OpenCV VideoWriter
                        h, w = brightfield_frames[0].shape[:2]
                        writer = ensure_grayscale_video_writer(brightfield_out_path, fps, (w, h), preferred_codec)
                        for frame in brightfield_frames:
                            writer.write(frame)
                        writer.release()
                        print(f"✅ Brightfield grayscale MP4 created with OpenCV: {os.path.basename(brightfield_out_path)}")
                else:
                    # Fallback to OpenCV VideoWriter
                    h, w = brightfield_frames[0].shape[:2]
                    writer = ensure_grayscale_video_writer(brightfield_out_path, fps, (w, h), preferred_codec)
                    for frame in brightfield_frames:
                        writer.write(frame)
                    writer.release()
                    print(f"✅ Brightfield grayscale MP4 created with OpenCV: {os.path.basename(brightfield_out_path)}")
            
            return

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
            
            # Check if brightfield channel exists
            has_brightfield = 'brightfield' in ch_map
            
            # Print channel information
            print_channel_info(nd2_path, meta, num_channels, ch_map)
            
            # Generate output paths
            color_out_path, brightfield_out_path = generate_output_paths(out_path, has_brightfield)

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
            for idx in range(len(images)):
                frame = to_native_endian(np.asarray(images[idx]))
                if frame.ndim == 3 and frame.shape[2] > 1:
                    channel_to_u8 = {}
                    brightfield_gray = None
                    
                    # Process brightfield channel separately if it exists
                    if has_brightfield and 'brightfield' in ch_map:
                        bf_idx = ch_map['brightfield']
                        if bf_idx < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(bf_idx, (0.0, 1.0))
                            brightfield_gray = normalize_to_uint8_with_clip(frame[:, :, bf_idx], cmin, cmax)
                            brightfield_gray = apply_clahe_if_needed(brightfield_gray, use_clahe, clip_limit, tile_grid)
                    
                    # Process color channels (excluding brightfield)
                    for color, idx_map in (('blue', ch_map.get('blue')), ('green', ch_map.get('green')), ('red', ch_map.get('red'))):
                        if isinstance(idx_map, int) and idx_map < frame.shape[2]:
                            cmin, cmax = clip_ranges.get(idx_map, (0.0, 1.0))
                            channel_to_u8[color] = normalize_to_uint8_with_clip(frame[:, :, idx_map], cmin, cmax)
                    
                    # Fallback: if no names detected, try to map channels based on common patterns
                    if not channel_to_u8 and frame.shape[2] > 0:
                        # For single channel, use green (more common for Alexa 488)
                        if frame.shape[2] == 1:
                            cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                            channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                        else:
                            # For multiple channels, map first channel to green
                            cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                            channel_to_u8['green'] = normalize_to_uint8_with_clip(frame[:, :, 0], cmin, cmax)
                    
                    bgr = build_rgb_from_channels(channel_to_u8, use_clahe, clip_limit, tile_grid)
                else:
                    gray = frame if frame.ndim == 2 else np.squeeze(frame)
                    cmin, cmax = clip_ranges.get(0, (0.0, 1.0))
                    gray_u8 = normalize_to_uint8_with_clip(gray, cmin, cmax)
                    bgr = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
                    # Use green channel instead of red for better Alexa 488 compatibility
                    bgr[:, :, 1] = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                    # For single channel, also use as brightfield if no brightfield detected
                    if not has_brightfield:
                        brightfield_gray = apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                
                all_frames.append(bgr)
                if brightfield_gray is not None:
                    brightfield_frames.append(brightfield_gray)
            
            # Create color MP4
            if use_ffmpeg:
                print(f"Creating QuickTime-compatible MP4 with FFmpeg: {os.path.basename(color_out_path)}")
                if create_high_quality_mp4_with_ffmpeg(all_frames, color_out_path, fps):
                    print(f"✅ QuickTime-compatible MP4 created with FFmpeg: {os.path.basename(color_out_path)}")
                else:
                    print("⚠️  FFmpeg not available or failed, falling back to OpenCV VideoWriter")
                    # Fallback to OpenCV VideoWriter
                    print(f"Creating MP4 with OpenCV VideoWriter: {os.path.basename(color_out_path)}")
                    h, w = all_frames[0].shape[:2]
                    writer = ensure_video_writer(color_out_path, fps, (w, h), preferred_codec)
                    for frame in all_frames:
                        writer.write(frame)
                    writer.release()
                    print(f"✅ MP4 created with OpenCV: {os.path.basename(color_out_path)}")
            else:
                # Fallback to OpenCV VideoWriter
                print(f"Creating MP4 with OpenCV VideoWriter: {os.path.basename(color_out_path)}")
                h, w = all_frames[0].shape[:2]
                writer = ensure_video_writer(color_out_path, fps, (w, h), preferred_codec)
                for frame in all_frames:
                    writer.write(frame)
                writer.release()
                print(f"✅ MP4 created with OpenCV: {os.path.basename(color_out_path)}")
            
            # Create brightfield grayscale MP4 if brightfield channel exists
            if brightfield_out_path and brightfield_frames:
                print(f"Creating brightfield grayscale MP4: {os.path.basename(brightfield_out_path)}")
                if use_ffmpeg:
                    if create_grayscale_mp4_with_ffmpeg(brightfield_frames, brightfield_out_path, fps):
                        print(f"✅ Brightfield grayscale MP4 created with FFmpeg: {os.path.basename(brightfield_out_path)}")
                    else:
                        print("⚠️  FFmpeg not available or failed, falling back to OpenCV VideoWriter")
                        # Fallback to OpenCV VideoWriter
                        h, w = brightfield_frames[0].shape[:2]
                        writer = ensure_grayscale_video_writer(brightfield_out_path, fps, (w, h), preferred_codec)
                        for frame in brightfield_frames:
                            writer.write(frame)
                        writer.release()
                        print(f"✅ Brightfield grayscale MP4 created with OpenCV: {os.path.basename(brightfield_out_path)}")
                else:
                    # Fallback to OpenCV VideoWriter
                    h, w = brightfield_frames[0].shape[:2]
                    writer = ensure_grayscale_video_writer(brightfield_out_path, fps, (w, h), preferred_codec)
                    for frame in brightfield_frames:
                        writer.write(frame)
                    writer.release()
                    print(f"✅ Brightfield grayscale MP4 created with OpenCV: {os.path.basename(brightfield_out_path)}")
            
            return

    raise SystemExit(
        "No ND2 backend available. Please install at least one: 'pip install nd2reader pims'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ND2 movies to MP4 with red-black coloring and CLAHE.")
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
        help="Skip existing MP4 files instead of overwriting them (default: overwrite)",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Disable FFmpeg and use OpenCV VideoWriter instead (FFmpeg is used by default for high-quality encoding)",
    )

    args = parser.parse_args()

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
        if os.path.exists(out_path) and args.no_overwrite:
            print(f"Skip existing: {out_path}")
            continue
        print(f"Converting: {nd2_path}\n  -> {out_path}")
        try:
            # Export percentiles via env for the converter
            if args.norm_mode == "percentile":
                os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
                os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
            else:
                os.environ.pop("ND2_NORM_LOWER_PCT", None)
                os.environ.pop("ND2_NORM_UPPER_PCT", None)

            convert_nd2_to_mp4(
                nd2_path=nd2_path,
                out_path=out_path,
                default_fps=float(args.fps),
                use_clahe=use_clahe,
                clip_limit=float(args.clip_limit),
                tile_grid=int(args.tile_grid),
                preferred_codec=str(args.codec) if args.codec else None,
                use_ffmpeg=use_ffmpeg,
            )
        except Exception as exc:
            print(f"ERROR converting {nd2_path}: {exc}")

    if not found_any:
        print("No .nd2 files found in:", root_dir, "(searched recursively)")


if __name__ == "__main__":
    main()
