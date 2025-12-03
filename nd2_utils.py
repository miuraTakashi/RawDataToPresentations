#!/usr/bin/env python3
"""
Utility functions for ND2 file processing.

This module contains common functions shared across multiple ND2 processing scripts:
- nd2_to_mp4.py
- nd2images_to_keynote.py
- nd2movies_to_keynote.py
- nd2volumes_to_keynote.py
- nd2_to_keynote.py

Functions included:
- ND2 reader backend detection and import
- Metadata extraction and conversion
- Image normalization and processing
- Channel detection and color mapping
- File discovery utilities
"""

import os
import sys
from typing import Generator, Optional, Tuple, Dict, Any, List

import numpy as np

# =============================================================================
# ND2 Reader Backend Detection
# =============================================================================

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
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


def check_nd2_backends() -> Tuple[bool, bool]:
    """Check availability of ND2 reader backends.
    
    Returns:
        Tuple of (has_nd2reader, has_pims)
    """
    return _HAS_ND2READER, _HAS_PIMS


def require_nd2_backend() -> None:
    """Raise error if no ND2 backend is available."""
    if not _HAS_ND2READER and not _HAS_PIMS:
        raise ImportError(
            "No ND2 backend available. Please install: pip install nd2reader pims"
        )


def require_cv2() -> None:
    """Raise error if OpenCV is not available."""
    if not _HAS_CV2:
        raise ImportError(
            "OpenCV (opencv-python) is required. Install with: pip install opencv-python"
        )


# =============================================================================
# File Discovery
# =============================================================================

def find_nd2_files_recursively(root_dir: str) -> Generator[Tuple[str, str], None, None]:
    """Yield (parent_folder_path, nd2_file_path) for each .nd2 in root_dir and all subdirectories.
    
    Args:
        root_dir: Root directory to search
        
    Yields:
        Tuples of (directory_path, nd2_file_path)
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".nd2") and not filename.startswith('.') and not filename.startswith('._'):
                nd2_file_path = os.path.join(dirpath, filename)
                yield dirpath, nd2_file_path


def find_nd2_files_in_directory(directory: str, recursive: bool = True) -> List[str]:
    """Find all ND2 files in a directory.
    
    Args:
        directory: Directory to search
        recursive: If True, search subdirectories
        
    Returns:
        List of ND2 file paths
    """
    nd2_files = []
    if recursive:
        for _, nd2_path in find_nd2_files_recursively(directory):
            nd2_files.append(nd2_path)
    else:
        for f in os.listdir(directory):
            if f.lower().endswith('.nd2') and not f.startswith('.') and not f.startswith('._'):
                nd2_files.append(os.path.join(directory, f))
    return sorted(nd2_files)


# =============================================================================
# Metadata Extraction and Conversion
# =============================================================================

def metadata_to_dict(metadata_obj) -> Dict[str, Any]:
    """Convert nd2reader metadata object to dictionary.
    
    Args:
        metadata_obj: Metadata object from nd2reader or similar
        
    Returns:
        Dictionary representation of metadata
    """
    meta_dict = {}
    
    # Try direct dict conversion
    if isinstance(metadata_obj, dict):
        return metadata_obj
    
    # Try to access as dictionary-like object
    if hasattr(metadata_obj, 'keys'):
        try:
            return dict(metadata_obj)
        except Exception:
            pass
    
    # Try to get all attributes
    if hasattr(metadata_obj, '__dict__'):
        meta_dict.update(metadata_obj.__dict__)
    
    # Try to access common attributes
    for attr in dir(metadata_obj):
        if not attr.startswith('_') and not callable(getattr(metadata_obj, attr, None)):
            try:
                value = getattr(metadata_obj, attr)
                if not callable(value):
                    meta_dict[attr] = value
            except Exception:
                pass
    
    # Try to access as items
    if hasattr(metadata_obj, 'items'):
        try:
            for key, value in metadata_obj.items():
                meta_dict[key] = value
        except Exception:
            pass
    
    return meta_dict


def detect_fps_from_metadata(meta: dict) -> Optional[float]:
    """Attempt to detect frames-per-second from known metadata keys.
    
    Args:
        meta: Metadata dictionary
        
    Returns:
        FPS value if found, None otherwise
    """
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
    
    Args:
        meta: Metadata dictionary
        height_px: Image height in pixels
        width_px: Image width in pixels
        
    Returns:
        Side length in micrometers, or None if not determinable
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


def get_pixel_size_um(meta: dict) -> Optional[float]:
    """Get pixel size in micrometers from metadata.
    
    Args:
        meta: Metadata dictionary
        
    Returns:
        Pixel size in µm, or None if not found
    """
    for key in ("pixel_microns", "pixel_size", "pixelSize", "pixel_size_x", "pixelSizeX", "x_pixel_size"):
        if key in meta:
            v = meta[key]
            if isinstance(v, dict):
                val = v.get('x') or v.get('width')
            else:
                val = v
            if val is not None:
                try:
                    val = float(val)
                    # Convert if looks like meters
                    if val < 1e-6:
                        val = val * 1e6
                    if val > 0:
                        return val
                except Exception:
                    pass
    return None


# =============================================================================
# Channel Detection and Color Mapping
# =============================================================================

def classify_channel_names(meta: dict, num_channels: int) -> Dict[str, int]:
    """Return mapping from semantic colors to channel indices based on metadata names.

    Tries to detect:
    - Alexa 568 -> red
    - Alexa 488 -> green  
    - DAPI -> blue
    - Brightfield -> brightfield
    
    If names are unavailable, returns empty mapping.
    
    Args:
        meta: Metadata dictionary
        num_channels: Number of channels in the image
        
    Returns:
        Dictionary mapping color names to channel indices
    """
    mapping: Dict[str, int] = {}
    try:
        channels_meta = meta.get('channels') if isinstance(meta, dict) else None
        if isinstance(channels_meta, (list, tuple)):
            for idx, ch in enumerate(channels_meta):
                if idx >= num_channels:
                    break
                # Channel label extraction heuristic
                if isinstance(ch, dict):
                    name = str(ch.get('label') or ch.get('name') or ch.get('channel') or '')
                else:
                    name = str(ch)
                name_l = name.lower()
                
                # Brightfield detection
                if (('brightfield' in name_l or 'bright field' in name_l or 'phase' in name_l or 
                     'dic' in name_l or 'transmitted' in name_l or 'trans' in name_l or
                     'bright' in name_l or 'bf' in name_l or 'td' in name_l) and 'brightfield' not in mapping):
                    mapping['brightfield'] = idx
                # DAPI detection (blue channel)
                elif 'dapi' in name_l and 'blue' not in mapping:
                    mapping['blue'] = idx
                # Alexa 488 detection (green channel)
                elif (('488' in name_l or 'alexa488' in name_l or 'alexa 488' in name_l or 
                       'alexa488 antibody' in name_l or 'alexa 488 antibody' in name_l or
                       'gfap' in name_l or 'green' in name_l or 'egfp' in name_l or
                       'gfp' in name_l or 'fitc' in name_l) and 'green' not in mapping):
                    mapping['green'] = idx
                # Alexa 568 detection (red channel)
                elif (('568' in name_l or 'alx568' in name_l or 'alexa568' in name_l or 
                       'alexa 568' in name_l or 'alexa568 antibody' in name_l or 
                       'alexa 568 antibody' in name_l or 'cy3' in name_l or 'cy3.5' in name_l or
                       'cyanine3' in name_l or 'cyanine3.5' in name_l or 'rhodamine' in name_l or
                       'tubulin' in name_l) and 'red' not in mapping):
                    mapping['red'] = idx
    except Exception:
        pass
    return mapping


def print_channel_info(nd2_path: str, meta: dict, num_channels: int, ch_map: dict) -> None:
    """Print detailed channel information for the ND2 file.
    
    Args:
        nd2_path: Path to the ND2 file
        meta: Metadata dictionary
        num_channels: Number of channels
        ch_map: Channel to color mapping
    """
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
            emoji = {'red': '🔴', 'green': '🟢', 'blue': '🔵', 'brightfield': '⚪'}.get(color, '⭕')
            print(f"     {emoji} {color.upper()} -> Channel {idx}")
    else:
        print("   Color mapping: No specific mapping detected")


# =============================================================================
# Image Processing
# =============================================================================

def to_native_endian(arr: np.ndarray) -> np.ndarray:
    """Ensure array is in native byte order for compatibility.
    
    Args:
        arr: Input numpy array
        
    Returns:
        Array in native byte order
    """
    if arr.dtype.byteorder not in ('=', '|', '<' if sys.byteorder == 'little' else '>'):
        return arr.astype(arr.dtype.newbyteorder('='), copy=False)
    return arr


def normalize_to_uint8(data: np.ndarray, clip_min: Optional[float] = None, 
                       clip_max: Optional[float] = None) -> np.ndarray:
    """Normalize array to uint8 [0, 255] range with percentile-based clipping.
    
    Args:
        data: Input array
        clip_min: Minimum clip value (default: 1st percentile)
        clip_max: Maximum clip value (default: 99.5th percentile)
        
    Returns:
        Normalized uint8 array
    """
    data = data.astype(np.float32)
    
    if clip_min is None:
        clip_min = np.percentile(data, 1.0)
    if clip_max is None:
        clip_max = np.percentile(data, 99.5)
    
    if clip_max <= clip_min:
        clip_max = clip_min + 1.0
    
    clipped = np.clip(data, clip_min, clip_max)
    normalized = ((clipped - clip_min) / (clip_max - clip_min) * 255.0).astype(np.uint8)
    return normalized


def normalize_to_uint8_with_clip(data: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Normalize array to uint8 [0, 255] with explicit clipping.
    
    Args:
        data: Input array
        clip_min: Minimum clip value
        clip_max: Maximum clip value
        
    Returns:
        Normalized uint8 array
    """
    data = data.astype(np.float32)
    if clip_max <= clip_min:
        clip_max = clip_min + 1.0
    clipped = np.clip(data, clip_min, clip_max)
    return ((clipped - clip_min) / (clip_max - clip_min) * 255.0).astype(np.uint8)


def apply_clahe_if_needed(gray_u8: np.ndarray, use_clahe: bool, 
                          clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE contrast enhancement if enabled.
    
    Args:
        gray_u8: Input grayscale uint8 image
        use_clahe: Whether to apply CLAHE
        clip_limit: CLAHE clip limit
        tile_grid: CLAHE tile grid size
        
    Returns:
        Enhanced image (or original if CLAHE disabled)
    """
    require_cv2()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
        return clahe.apply(gray_u8)
    return gray_u8


def build_rgb_from_channels(
    channel_to_u8: Dict[str, np.ndarray],
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
) -> np.ndarray:
    """Construct BGR frame from available per-channel 8-bit arrays.

    Args:
        channel_to_u8: Dictionary mapping color names ('red', 'green', 'blue') to 8-bit grayscale arrays
        use_clahe: Whether to apply CLAHE
        clip_limit: CLAHE clip limit
        tile_grid: CLAHE tile grid size
        
    Returns:
        BGR image array
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


def create_colored_frame(frame: np.ndarray, ch_map: Dict[str, int], use_clahe: bool = True,
                         clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    """Create a colored BGR frame from multi-channel data.
    
    Args:
        frame: Input frame (2D grayscale or 3D multi-channel)
        ch_map: Channel to color mapping
        use_clahe: Whether to apply CLAHE
        clip_limit: CLAHE clip limit
        tile_grid: CLAHE tile grid size
        
    Returns:
        BGR colored frame
    """
    if frame.ndim == 2:
        # Single channel - use ch_map to determine color
        gray_u8 = normalize_to_uint8(frame)
        if use_clahe:
            gray_u8 = apply_clahe_if_needed(gray_u8, True, clip_limit, tile_grid)
        bgr = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        
        # Check ch_map for channel 0's color assignment
        color_for_ch0 = None
        for color, idx in ch_map.items():
            if idx == 0:
                color_for_ch0 = color
                break
        
        # Apply color based on ch_map
        if color_for_ch0 == 'red':
            bgr[:, :, 2] = gray_u8  # R in BGR
        elif color_for_ch0 == 'green':
            bgr[:, :, 1] = gray_u8  # G in BGR
        elif color_for_ch0 == 'blue':
            bgr[:, :, 0] = gray_u8  # B in BGR
        else:
            # Default to green if no mapping found
            bgr[:, :, 1] = gray_u8
        return bgr
    
    num_channels = frame.shape[2] if frame.ndim == 3 else 1
    h, w = frame.shape[:2]
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize each channel separately
    for ch_idx in range(num_channels):
        ch_data = frame[:, :, ch_idx]
        ch_u8 = normalize_to_uint8(ch_data)
        if use_clahe:
            ch_u8 = apply_clahe_if_needed(ch_u8, True, clip_limit, tile_grid)
        
        # Determine color for this channel
        color_assigned = False
        for color, idx in ch_map.items():
            if idx == ch_idx:
                if color == 'red':
                    bgr[:, :, 2] = np.maximum(bgr[:, :, 2], ch_u8)  # R in BGR
                    color_assigned = True
                elif color == 'green':
                    bgr[:, :, 1] = np.maximum(bgr[:, :, 1], ch_u8)  # G in BGR
                    color_assigned = True
                elif color == 'blue':
                    bgr[:, :, 0] = np.maximum(bgr[:, :, 0], ch_u8)  # B in BGR
                    color_assigned = True
                break
        
        # If no color assigned, use default mapping based on channel index
        if not color_assigned:
            if ch_idx == 0:
                bgr[:, :, 0] = np.maximum(bgr[:, :, 0], ch_u8)  # Blue
            elif ch_idx == 1:
                bgr[:, :, 1] = np.maximum(bgr[:, :, 1], ch_u8)  # Green
            elif ch_idx == 2:
                bgr[:, :, 2] = np.maximum(bgr[:, :, 2], ch_u8)  # Red
    
    return bgr


# =============================================================================
# FFmpeg Utilities
# =============================================================================

def check_ffmpeg_availability() -> bool:
    """Check if FFmpeg is available on the system.
    
    Returns:
        True if FFmpeg is available, False otherwise
    """
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# =============================================================================
# Content Type Detection
# =============================================================================

def detect_nd2_content_type(nd2_path: str) -> str:
    """Detect the content type of an ND2 file.
    
    Args:
        nd2_path: Path to the ND2 file
        
    Returns:
        One of: 'image', 'movie', 'volume', 'unknown'
    """
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                sizes = getattr(images, 'sizes', {}) or {}
                num_frames = sizes.get('t', 1)  # Time frames
                num_z = sizes.get('z', 1)       # Z slices
                
                # Volume: has Z-stack
                if num_z > 1:
                    return 'volume'
                # Movie: has multiple time frames
                elif num_frames > 1:
                    return 'movie'
                elif len(images) > 1:
                    # Could be a movie or volume depending on iteration
                    return 'movie'
                else:
                    return 'image'
        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                num_frames = len(images)
                if num_frames > 1:
                    return 'movie'  # Default to movie for multi-frame
                else:
                    return 'image'
        
        return 'unknown'
    
    except Exception:
        return 'unknown'

