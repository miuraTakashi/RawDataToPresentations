#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation directly from ND2 movie files.

This script combines two steps:
1) Convert ND2 files to MP4 movies (red-on-black, optional CLAHE) reusing nd2_to_mp4 logic
2) Insert the generated MP4s into a Keynote presentation (one slide per movie)

By default, recursively scans all subdirectories for .nd2 files under --input.

Requirements:
- macOS with Keynote installed
- Python 3
- nd2_to_mp4 dependencies: nd2reader or pims, opencv-python, numpy

Usage examples:
  python create_keynote_from_nd2_movies.py --input "." --fps 10 --theme "White"
  python create_keynote_from_nd2_movies.py --input "/path/to/nd2_root" --no-clahe --codec mp4v
"""

import argparse
import os
import subprocess
import sys
from typing import List, Dict, Tuple, Optional

# Add script directory to Python path to allow importing nd2_to_mp4
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Reuse converter implementation
try:
    import nd2_to_mp4  # type: ignore
except Exception as exc:
    sys.stderr.write("ERROR: Unable to import nd2_to_mp4. Ensure it exists beside this script.\n")
    raise

# Optional imports for metadata extraction
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
    import numpy as np
    import cv2
except Exception as exc:
    sys.stderr.write("ERROR: numpy and opencv-python are required. Install with: pip install numpy opencv-python\n")
    raise


def extract_nd2_metadata(nd2_path: str, default_fps: float = 10.0) -> Dict[str, str]:
    """Extract metadata from ND2 file including dimensions, pixel size, frame rate, and total frames."""
    metadata = {
        "length_um": "N/A",
        "width_um": "N/A", 
        "frame_rate": "N/A",
        "total_frames": "N/A"
    }
    
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                meta = getattr(images, "metadata", {}) or {}
                
                # Extract total frames
                try:
                    total_frames = len(images)
                    metadata["total_frames"] = f"{total_frames} frames"
                except Exception:
                    pass
                
                # Extract frame rate from multiple sources
                fps = None
                frame_rate_found = False
                
                # First try timesteps array (most reliable for ND2 files)
                if hasattr(images, 'timesteps') and images.timesteps is not None:
                    try:
                        timesteps = images.timesteps
                        if len(timesteps) > 1:
                            # Calculate frame interval from timesteps (assuming milliseconds)
                            frame_interval = timesteps[1] - timesteps[0]
                            if frame_interval > 0:
                                frame_interval_seconds = frame_interval / 1000.0  # Convert milliseconds to seconds
                                
                                # Determine appropriate unit based on interval
                                if frame_interval_seconds >= 60:  # 1分以上
                                    frame_interval_minutes = frame_interval_seconds / 60.0
                                    metadata["frame_rate"] = f"{frame_interval_minutes:.1f}min/frame"
                                    frame_rate_found = True
                                elif frame_interval_seconds >= 1:  # 1秒以上
                                    metadata["frame_rate"] = f"{frame_interval_seconds:.1f}s/frame"
                                    frame_rate_found = True
                                else:  # 1秒未満
                                    fps = 1.0 / frame_interval_seconds
                                    metadata["frame_rate"] = f"{fps:.1f}fps"
                                    frame_rate_found = True
                    except (ValueError, ZeroDivisionError, IndexError):
                        pass
                
                # Only try other methods if timesteps didn't work
                if not frame_rate_found:
                    # Try the frame_rate attribute
                    if hasattr(images, 'frame_rate') and images.frame_rate is not None:
                        try:
                            fps_candidate = 1.0 / float(images.frame_rate)
                            # Only use if it's reasonable (between 0.1 and 1000 fps)
                            if 0.1 <= fps_candidate <= 1000:
                                fps = fps_candidate
                                metadata["frame_rate"] = f"{fps:.1f} fps"
                                frame_rate_found = True
                        except (ValueError, ZeroDivisionError):
                            pass
                    
                    # Fallback to metadata-based detection
                    if not frame_rate_found:
                        fps = nd2_to_mp4.detect_fps_from_metadata(meta)
                        if fps and fps > 0:
                            metadata["frame_rate"] = f"{fps:.1f} fps"
                        else:
                            # Use default_fps for static images or when no frame rate is found
                            metadata["frame_rate"] = f"{default_fps:.1f} fps"
                
                # Extract pixel size and dimensions
                pixel_size_x = None
                pixel_size_y = None
                
                # Try different metadata keys for pixel size
                for key in ["pixel_microns", "pixel_size", "pixelSize", "pixel_size_x", "pixelSizeX", "x_pixel_size"]:
                    if key in meta:
                        pixel_size_value = meta[key]
                        # pixel_microns might be a single value or a dict with x,y
                        if isinstance(pixel_size_value, dict):
                            pixel_size_x = pixel_size_value.get('x') or pixel_size_value.get('width')
                            pixel_size_y = pixel_size_value.get('y') or pixel_size_value.get('height')
                        else:
                            # Single value, use for both dimensions
                            pixel_size_x = pixel_size_value
                            pixel_size_y = pixel_size_value
                        break
                
                # If not found in direct keys, try nested structures
                if pixel_size_x is None:
                    for key in ["experiment", "acquisition", "calibration", "microscope"]:
                        if key in meta and isinstance(meta[key], dict):
                            submeta = meta[key]
                            for subkey in ["pixel_microns", "pixel_size", "pixelSize", "pixel_size_x", "pixelSizeX"]:
                                if subkey in submeta:
                                    pixel_size_value = submeta[subkey]
                                    if isinstance(pixel_size_value, dict):
                                        pixel_size_x = pixel_size_value.get('x') or pixel_size_value.get('width')
                                        pixel_size_y = pixel_size_value.get('y') or pixel_size_value.get('height')
                                    else:
                                        pixel_size_x = pixel_size_value
                                        pixel_size_y = pixel_size_value
                                    break
                            if pixel_size_x is not None:
                                break
                
                # Get image dimensions
                if hasattr(images, 'sizes'):
                    sizes = images.sizes
                    height_px = sizes.get('y', 0)
                    width_px = sizes.get('x', 0)
                else:
                    # Fallback: get dimensions from first frame
                    try:
                        first_frame = images[0]
                        if hasattr(first_frame, 'shape'):
                            height_px, width_px = first_frame.shape[:2]
                        else:
                            height_px, width_px = 0, 0
                    except:
                        height_px, width_px = 0, 0
                
                # Calculate physical dimensions if pixel size is available
                if pixel_size_x and pixel_size_y and height_px > 0 and width_px > 0:
                    try:
                        # Convert pixel size to micrometers (assuming it's in meters)
                        if pixel_size_x < 1e-6:  # Likely in meters, convert to micrometers
                            pixel_size_x_um = pixel_size_x * 1e6
                        else:  # Already in micrometers
                            pixel_size_x_um = pixel_size_x
                            
                        if pixel_size_y < 1e-6:  # Likely in meters, convert to micrometers
                            pixel_size_y_um = pixel_size_y * 1e6
                        else:  # Already in micrometers
                            pixel_size_y_um = pixel_size_y
                        
                        length_um = height_px * pixel_size_y_um
                        width_um = width_px * pixel_size_x_um
                        
                        metadata["length_um"] = f"{length_um:.1f} µm"
                        metadata["width_um"] = f"{width_um:.1f} µm"
                    except Exception:
                        pass
                        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                try:
                    meta = getattr(images, "metadata", {}) or {}
                except Exception:
                    meta = {}
                
                # Extract total frames
                try:
                    total_frames = len(images)
                    metadata["total_frames"] = f"{total_frames} frames"
                except Exception:
                    pass
                
                # Extract frame rate from multiple sources
                fps = None
                frame_rate_found = False
                
                # First try timesteps array (most reliable for ND2 files)
                if hasattr(images, 'timesteps') and images.timesteps is not None:
                    try:
                        timesteps = images.timesteps
                        if len(timesteps) > 1:
                            frame_interval = timesteps[1] - timesteps[0]
                            if frame_interval > 0:
                                frame_interval_seconds = frame_interval / 1000.0  # Convert milliseconds to seconds
                                
                                # Determine appropriate unit based on interval
                                if frame_interval_seconds >= 60:  # 1分以上
                                    frame_interval_minutes = frame_interval_seconds / 60.0
                                    metadata["frame_rate"] = f"{frame_interval_minutes:.1f}min/frame"
                                    frame_rate_found = True
                                elif frame_interval_seconds >= 1:  # 1秒以上
                                    metadata["frame_rate"] = f"{frame_interval_seconds:.1f}s/frame"
                                    frame_rate_found = True
                                else:  # 1秒未満
                                    fps = 1.0 / frame_interval_seconds
                                    metadata["frame_rate"] = f"{fps:.1f}fps"
                                    frame_rate_found = True
                    except (ValueError, ZeroDivisionError, IndexError):
                        pass
                
                # Only try other methods if timesteps didn't work
                if not frame_rate_found:
                    # Try PIMS-specific attributes
                    if hasattr(images, 'frame_rate') and images.frame_rate is not None:
                        try:
                            fps_candidate = float(images.frame_rate)
                            # Only use if it's reasonable (between 0.1 and 1000 fps)
                            if 0.1 <= fps_candidate <= 1000:
                                fps = fps_candidate
                                metadata["frame_rate"] = f"{fps:.1f} fps"
                                frame_rate_found = True
                        except (ValueError, TypeError):
                            pass
                    
                    # Fallback to metadata-based detection
                    if not frame_rate_found:
                        fps = nd2_to_mp4.detect_fps_from_metadata(meta)
                        if fps and fps > 0:
                            metadata["frame_rate"] = f"{fps:.1f} fps"
                        else:
                            # Use default_fps for static images or when no frame rate is found
                            metadata["frame_rate"] = f"{default_fps:.1f} fps"
                
                # For PIMS, try to get dimensions from the first frame
                try:
                    first_frame = images[0]
                    if hasattr(first_frame, 'shape'):
                        height_px, width_px = first_frame.shape[:2]
                        # Note: PIMS metadata extraction for pixel size is more limited
                        # This is a simplified approach
                        metadata["length_um"] = f"{height_px} px"
                        metadata["width_um"] = f"{width_px} px"
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"Warning: Could not extract metadata from {nd2_path}: {e}")
    
    return metadata


def convert_individual_channels_to_mp4(
    nd2_path: str,
    base_output_path: str,
    ch_map: dict,
    default_fps: float,
    use_clahe: bool,
    clip_limit: float,
    tile_grid: int,
    preferred_codec: Optional[str],
    use_ffmpeg: bool,
) -> List[Tuple[str, str]]:
    """Convert individual fluorescence channels to separate grayscale MP4 files.
    
    Args:
        nd2_path: Path to ND2 file
        base_output_path: Base output path (e.g., "/path/to/file.mp4")
        ch_map: Channel mapping dictionary (e.g., {'red': 0, 'green': 1, 'blue': 2})
        default_fps: Default frame rate
        use_clahe: Whether to apply CLAHE
        clip_limit: CLAHE clip limit
        tile_grid: CLAHE tile grid size
        preferred_codec: Preferred codec
        use_ffmpeg: Whether to use FFmpeg
    
    Returns:
        List of tuples (channel_name, output_path) for successfully created MP4s
    """
    created_files: List[Tuple[str, str]] = []
    
    # Channel name mapping for display
    channel_display_names = {
        'red': 'Red (Alexa568)',
        'green': 'Green (Alexa488)',
        'blue': 'Blue (DAPI)',
    }
    
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                images.iter_axes = "t"
                if "c" in getattr(images, "sizes", {}):
                    images.bundle_axes = "yxc"
                else:
                    images.bundle_axes = "yx"
                
                meta = getattr(images, "metadata", {}) or {}
                fps = nd2_to_mp4.detect_fps_from_metadata(meta) or float(default_fps)
                
                # Get image dimensions
                sample_shape = np.asarray(images[0])
                num_channels = 1
                if sample_shape.ndim == 3:
                    num_channels = sample_shape.shape[2]
                
                # Determine side length for metadata
                try:
                    sizes = getattr(images, 'sizes', {}) or {}
                    h_px = int(sizes.get('y')) if 'y' in sizes else None
                    w_px = int(sizes.get('x')) if 'x' in sizes else None
                except Exception:
                    h_px, w_px = None, None
                if h_px is None or w_px is None:
                    if sample_shape.ndim == 3:
                        h_px, w_px = sample_shape.shape[:2]
                    else:
                        h_px, w_px = sample_shape.shape[0], sample_shape.shape[1]
                
                side_um = nd2_to_mp4.extract_side_length_um(meta, int(h_px or 0), int(w_px or 0))
                
                # First pass: compute per-channel global histogram percentiles
                per_channel_samples = {i: [] for i in range(num_channels)}
                num_frames = len(images)
                for idx in range(num_frames):
                    arr = nd2_to_mp4.to_native_endian(np.asarray(images[idx]))
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
                
                # Process each fluorescence channel separately
                for channel_name in ['red', 'green', 'blue']:
                    if channel_name not in ch_map:
                        continue
                    
                    channel_idx = ch_map[channel_name]
                    if channel_idx >= num_channels:
                        continue
                    
                    # Generate output path for this channel
                    base_dir = os.path.dirname(base_output_path)
                    base_name = os.path.basename(base_output_path)
                    name_without_ext = os.path.splitext(base_name)[0]
                    ext = os.path.splitext(base_name)[1]
                    channel_out_path = os.path.join(base_dir, f"{name_without_ext}_{channel_name}{ext}")
                    
                    # Process all frames for this channel
                    channel_frames = []
                    for idx in range(len(images)):
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[idx]))
                        if frame.ndim == 3 and frame.shape[2] > channel_idx:
                            channel_data = frame[:, :, channel_idx]
                        else:
                            # Single channel case
                            channel_data = frame if frame.ndim == 2 else np.squeeze(frame)
                        
                        cmin, cmax = clip_ranges.get(channel_idx, (0.0, 1.0))
                        gray_u8 = nd2_to_mp4.normalize_to_uint8_with_clip(channel_data, cmin, cmax)
                        gray_u8 = nd2_to_mp4.apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        channel_frames.append(gray_u8)
                    
                    # Create grayscale MP4 for this channel
                    if channel_frames:
                        channel_display = channel_display_names.get(channel_name, channel_name.capitalize())
                        print(f"Creating {channel_display} channel MP4: {os.path.basename(channel_out_path)}")
                        channel_comment = nd2_to_mp4.build_comment_tag(
                            nd2_path, side_um, float(fps), note=f"{channel_display} channel only"
                        )
                        
                        if use_ffmpeg:
                            if nd2_to_mp4.create_grayscale_mp4_with_ffmpeg(
                                channel_frames, channel_out_path, fps, metadata_comment=channel_comment
                            ):
                                print(f"✅ {channel_display} channel MP4 created with FFmpeg: {os.path.basename(channel_out_path)}")
                                created_files.append((channel_name, channel_out_path))
                            else:
                                print(f"⚠️  FFmpeg failed for {channel_name}, falling back to OpenCV")
                                h, w = channel_frames[0].shape[:2]
                                writer = nd2_to_mp4.ensure_grayscale_video_writer(channel_out_path, fps, (w, h), preferred_codec)
                                for frame in channel_frames:
                                    writer.write(frame)
                                writer.release()
                                nd2_to_mp4.set_mp4_comment_with_ffmpeg(channel_out_path, channel_comment)
                                print(f"✅ {channel_display} channel MP4 created with OpenCV: {os.path.basename(channel_out_path)}")
                                created_files.append((channel_name, channel_out_path))
                        else:
                            h, w = channel_frames[0].shape[:2]
                            writer = nd2_to_mp4.ensure_grayscale_video_writer(channel_out_path, fps, (w, h), preferred_codec)
                            for frame in channel_frames:
                                writer.write(frame)
                            writer.release()
                            nd2_to_mp4.set_mp4_comment_with_ffmpeg(channel_out_path, channel_comment)
                            print(f"✅ {channel_display} channel MP4 created with OpenCV: {os.path.basename(channel_out_path)}")
                            created_files.append((channel_name, channel_out_path))
        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                try:
                    meta = getattr(images, "metadata", {}) or {}
                except Exception:
                    meta = {}
                fps = nd2_to_mp4.detect_fps_from_metadata(meta) or float(default_fps)
                
                # Get image dimensions
                sample_shape = np.asarray(images[0])
                num_channels = 1
                if sample_shape.ndim == 3:
                    num_channels = sample_shape.shape[2]
                
                # Determine side length for metadata
                try:
                    first_frame = np.asarray(images[0])
                    if first_frame.ndim == 3:
                        h_px, w_px = first_frame.shape[:2]
                    else:
                        h_px, w_px = first_frame.shape[0], first_frame.shape[1]
                except Exception:
                    h_px, w_px = 0, 0
                
                side_um = nd2_to_mp4.extract_side_length_um(meta, int(h_px), int(w_px))
                
                # First pass: compute per-channel global histogram percentiles
                per_channel_samples = {i: [] for i in range(num_channels)}
                num_frames = len(images)
                for idx in range(num_frames):
                    arr = nd2_to_mp4.to_native_endian(np.asarray(images[idx]))
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
                
                # Process each fluorescence channel separately
                for channel_name in ['red', 'green', 'blue']:
                    if channel_name not in ch_map:
                        continue
                    
                    channel_idx = ch_map[channel_name]
                    if channel_idx >= num_channels:
                        continue
                    
                    # Generate output path for this channel
                    base_dir = os.path.dirname(base_output_path)
                    base_name = os.path.basename(base_output_path)
                    name_without_ext = os.path.splitext(base_name)[0]
                    ext = os.path.splitext(base_name)[1]
                    channel_out_path = os.path.join(base_dir, f"{name_without_ext}_{channel_name}{ext}")
                    
                    # Process all frames for this channel
                    channel_frames = []
                    for idx in range(len(images)):
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[idx]))
                        if frame.ndim == 3 and frame.shape[2] > channel_idx:
                            channel_data = frame[:, :, channel_idx]
                        else:
                            # Single channel case
                            channel_data = frame if frame.ndim == 2 else np.squeeze(frame)
                        
                        cmin, cmax = clip_ranges.get(channel_idx, (0.0, 1.0))
                        gray_u8 = nd2_to_mp4.normalize_to_uint8_with_clip(channel_data, cmin, cmax)
                        gray_u8 = nd2_to_mp4.apply_clahe_if_needed(gray_u8, use_clahe, clip_limit, tile_grid)
                        channel_frames.append(gray_u8)
                    
                    # Create grayscale MP4 for this channel
                    if channel_frames:
                        channel_display = channel_display_names.get(channel_name, channel_name.capitalize())
                        print(f"Creating {channel_display} channel MP4: {os.path.basename(channel_out_path)}")
                        channel_comment = nd2_to_mp4.build_comment_tag(
                            nd2_path, side_um, float(fps), note=f"{channel_display} channel only"
                        )
                        
                        if use_ffmpeg:
                            if nd2_to_mp4.create_grayscale_mp4_with_ffmpeg(
                                channel_frames, channel_out_path, fps, metadata_comment=channel_comment
                            ):
                                print(f"✅ {channel_display} channel MP4 created with FFmpeg: {os.path.basename(channel_out_path)}")
                                created_files.append((channel_name, channel_out_path))
                            else:
                                print(f"⚠️  FFmpeg failed for {channel_name}, falling back to OpenCV")
                                h, w = channel_frames[0].shape[:2]
                                writer = nd2_to_mp4.ensure_grayscale_video_writer(channel_out_path, fps, (w, h), preferred_codec)
                                for frame in channel_frames:
                                    writer.write(frame)
                                writer.release()
                                nd2_to_mp4.set_mp4_comment_with_ffmpeg(channel_out_path, channel_comment)
                                print(f"✅ {channel_display} channel MP4 created with OpenCV: {os.path.basename(channel_out_path)}")
                                created_files.append((channel_name, channel_out_path))
                        else:
                            h, w = channel_frames[0].shape[:2]
                            writer = nd2_to_mp4.ensure_grayscale_video_writer(channel_out_path, fps, (w, h), preferred_codec)
                            for frame in channel_frames:
                                writer.write(frame)
                            writer.release()
                            nd2_to_mp4.set_mp4_comment_with_ffmpeg(channel_out_path, channel_comment)
                            print(f"✅ {channel_display} channel MP4 created with OpenCV: {os.path.basename(channel_out_path)}")
                            created_files.append((channel_name, channel_out_path))
    
    except Exception as e:
        import traceback
        print(f"ERROR processing individual channels for {nd2_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    return created_files


def build_applescript(dir_path: str, mp4_paths: List[str], metadata_list: List[Dict[str, str]], theme_name: str, out_key_path: str) -> str:
    # Escape paths for AppleScript POSIX file
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")

    # Build AppleScript that creates a doc, inserts videos slide by slide, then saves
    lines: List[str] = []
    lines.append('tell application "Keynote"')
    lines.append('    activate')
    # Try to create document with ExperimentalDataR1 theme
    lines.append('    try')
    lines.append('        set theDoc to make new document with properties {document theme:theme "ExperimentalDataR1"}')
    lines.append('    on error')
    # If that fails, try with user-specified theme
    lines.append('        try')
    lines.append(f'            set theDoc to make new document with properties {{document theme:theme "{theme_name}"}}')
    lines.append('        on error')
    # Ultimate fallback: create without theme
    lines.append('            set theDoc to make new document')
    lines.append('        end try')
    lines.append('    end try')
    # Wait a moment to ensure document is fully created
    lines.append('    delay 1.0')
    # Remove the default slide if we will add our own slides
    if len(mp4_paths) > 0:
        lines.append('    try')
        lines.append('        tell theDoc to delete slide 1')
        lines.append('    end try')

    for idx, mp4_abs in enumerate(mp4_paths, start=1):
        posix_path = esc(mp4_abs)
        # title: parent folder / basename
        import os as path_os
        folder_name = path_os.path.basename(path_os.path.dirname(mp4_abs))
        file_name = path_os.path.splitext(path_os.path.basename(mp4_abs))[0]
        
        # Check if this is a brightfield file and modify title accordingly
        if file_name.endswith("_brightfield"):
            base_name = file_name.replace("_brightfield", "")
            slide_title = f"{folder_name}/{base_name} (明視野)"
        elif file_name.endswith("_red"):
            base_name = file_name.replace("_red", "")
            slide_title = f"{folder_name}/{base_name} (Red/Alexa568)"
        elif file_name.endswith("_green"):
            base_name = file_name.replace("_green", "")
            slide_title = f"{folder_name}/{base_name} (Green/Alexa488)"
        elif file_name.endswith("_blue"):
            base_name = file_name.replace("_blue", "")
            slide_title = f"{folder_name}/{base_name} (Blue/DAPI)"
        else:
            slide_title = f"{folder_name}/{file_name}"
        escaped_title = esc(slide_title)
        
        # Get metadata for this slide
        metadata = metadata_list[idx-1] if idx-1 < len(metadata_list) else {"length_um": "N/A", "width_um": "N/A", "frame_rate": "N/A"}
        
        # Format metadata text - use length_um for spatial scale, show as "µm/side"
        spatial_scale = metadata['length_um']
        if spatial_scale != "N/A" and spatial_scale.endswith(" µm"):
            # Extract the numeric value and format as "µm/side"
            try:
                value = spatial_scale.replace(" µm", "")
                spatial_scale = f"{value}µm/side"
            except:
                pass
        
        # Format for AppleScript - use separate variables for each line
        spatial_scale_escaped = esc(spatial_scale)
        frame_rate_escaped = esc(f"Frame Rate: {metadata['frame_rate']}")
        total_frames_escaped = esc(f"Total Frames: {metadata['total_frames']}")

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
        lines.append('            -- Set slide title in layout title placeholder')
        lines.append('            try')
        lines.append('                if (count of text items) > 0 then')
        lines.append(f'                    set object text of text item 1 to "{escaped_title}"')
        lines.append('                else')
        lines.append(f'                    set titleShape to make new text item with properties {{object text:"{escaped_title}", position:{{50, 50}}}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append(f'                set titleShape to make new text item with properties {{object text:"{escaped_title}", position:{{50, 50}}}}')
        lines.append('            end try')
        lines.append('            -- Try to populate existing text boxes with metadata')
        lines.append('            try')
        lines.append('                -- First, try to find and populate existing text placeholders')
        lines.append('                set textItems to (get every text item)')
        lines.append('                if (count of textItems) >= 2 then')
        lines.append('                    -- Use the second text item (usually right side content)')
        lines.append(f'                    set object text of text item 2 to "{spatial_scale_escaped}" & return & "{frame_rate_escaped}" & return & "{total_frames_escaped}"')
        lines.append('                else if (count of textItems) = 1 then')
        lines.append('                    -- Only title exists, try to find bullet point placeholder')
        lines.append('                    try')
        lines.append('                        -- Look for bullet point text items')
        lines.append('                        repeat with txtItem in textItems')
        lines.append('                            try')
        lines.append('                                if (object text of txtItem) is "" or (object text of txtItem) contains "•" then')
        lines.append(f'                                    set object text of txtItem to "{spatial_scale_escaped}" & return & "{frame_rate_escaped}" & return & "{total_frames_escaped}"')
        lines.append('                                    exit repeat')
        lines.append('                                end if')
        lines.append('                            on error')
        lines.append('                                -- Continue to next text item')
        lines.append('                            end try')
        lines.append('                        end repeat')
        lines.append('                    on error')
        lines.append('                        -- Create new text box for metadata')
        lines.append(f'                        set metadataBox to make new text item with properties {{object text:"{spatial_scale_escaped}" & return & "{frame_rate_escaped}" & return & "{total_frames_escaped}", position:{{400, 150}}, width:200, height:100}}')
        lines.append('                    end try')
        lines.append('                else')
        lines.append('                    -- No text items exist, create both title and metadata')
        lines.append(f'                    set titleBox to make new text item with properties {{object text:"{escaped_title}", position:{{50, 50}}}}')
        lines.append(f'                    set metadataBox to make new text item with properties {{object text:"{spatial_scale_escaped}" & return & "{frame_rate_escaped}" & return & "{total_frames_escaped}", position:{{400, 150}}, width:200, height:100}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append('                -- Fallback: create new text box for metadata')
        lines.append(f'                set metadataBox to make new text item with properties {{object text:"{spatial_scale_escaped}" & return & "{frame_rate_escaped}" & return & "{total_frames_escaped}", position:{{400, 150}}, width:200, height:100}}')
        lines.append('            end try')
        lines.append('            -- Insert movie/image into layout placeholder')
        lines.append('            try')
        lines.append(f'                set movieFile to alias (POSIX file "{posix_path}")')
        lines.append('                -- Try images placeholders first')
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
        lines.append('                -- Then try movie placeholders')
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
        lines.append('                -- If no placeholders, add as new image element')
        lines.append('                if (count of images) = 0 and (count of movies) = 0 then')
        lines.append('                    set newImage to make new image with properties {{file:movieFile, position:{{120, 120}}, width:512, height:512}}')
        lines.append('                end if')
        lines.append('            on error errMsg1')
        lines.append('                try')
        lines.append(f'                    set movieFile to alias (POSIX file "{posix_path}")')
        lines.append('                    set newShape to make new text item with properties {{object text:"Movie: " & (name of movieFile), position:{{120, 200}}}}')
        lines.append('                on error errMsg2')
        lines.append('                    log "All insert methods failed: " & errMsg1 & " / " & errMsg2')
        lines.append('                end try')
        lines.append('            end try')
        lines.append('        end tell')
        lines.append('    end tell')

    out_posix = esc(out_key_path)
    lines.append(f'    save theDoc in (POSIX file "{out_posix}")')
    lines.append('end tell')
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ND2s to MP4 and create a Keynote presentation.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.abspath(os.getcwd()),
        help="Root directory to recursively scan for ND2 files (default: current directory)",
    )
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto)")
    # Converter options (mirroring nd2_to_mp4)
    parser.add_argument("--fps", type=float, default=10.0, help="Fallback FPS if not found in ND2 metadata (default: 10)")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit (default: 3.0)")
    parser.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (N -> NxN) (default: 8)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    parser.add_argument("--codec", type=str, default="mp4v", help="Preferred fourcc codec for MP4 (default: mp4v)")
    parser.add_argument(
        "--norm-mode", choices=["percentile", "minmax"], default="percentile", help="Global normalization mode"
    )
    parser.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for histogram clipping")
    parser.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile for histogram clipping")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing MP4 files instead of overwriting them (default: overwrite)")
    parser.add_argument("--no-ffmpeg", action="store_true", help="Disable FFmpeg and use OpenCV VideoWriter instead (FFmpeg is used by default for high-quality encoding)")
    parser.add_argument("--separate-channels", action="store_true", help="Save each fluorescence channel (red, green, blue) as separate MP4 files and display them on separate Keynote slides")

    args = parser.parse_args()

    root_dir = os.path.abspath(args.input)
    if not os.path.isdir(root_dir):
        sys.stderr.write(f"Input directory does not exist: {root_dir}\n")
        sys.exit(1)

    # Check converter backends quickly (delegated errors will also appear later)
    if not getattr(nd2_to_mp4, "_HAS_ND2READER", False) and not getattr(nd2_to_mp4, "_HAS_PIMS", False):
        sys.stderr.write(
            "ERROR: No ND2 backend available. Please install: pip install nd2reader pims opencv-python numpy\n"
        )
        sys.exit(2)
    
    # Check FFmpeg availability and inform user
    use_ffmpeg = not args.no_ffmpeg
    if use_ffmpeg:
        if getattr(nd2_to_mp4, "check_ffmpeg_availability", lambda: False)():
            print("✅ FFmpeg detected - will use QuickTime-compatible encoding with H.264 baseline profile")
        else:
            print("⚠️  FFmpeg not found - falling back to OpenCV VideoWriter")
            print("   To install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
            use_ffmpeg = False
    else:
        print("📹 Using OpenCV VideoWriter (FFmpeg disabled by --no-ffmpeg)")

    # 1) Convert ND2 -> MP4 (collect paths for Keynote)
    mp4_paths: List[str] = []
    metadata_list: List[Dict[str, str]] = []
    use_clahe = not args.no_clahe
    found_any = False
    for subdir, nd2_path in nd2_to_mp4.find_nd2_files_recursively(root_dir):
        found_any = True
        base = os.path.splitext(os.path.basename(nd2_path))[0]
        out_path = os.path.join(subdir, f"{base}.mp4")
        
        # Extract metadata from ND2 file
        print(f"Extracting metadata from: {nd2_path}")
        metadata = extract_nd2_metadata(nd2_path, float(args.fps))
        metadata_list.append(metadata)
        
        # Get channel mapping for separate channels mode
        ch_map = {}
        if args.separate_channels:
            try:
                if getattr(nd2_to_mp4, "_HAS_ND2READER", False):
                    from nd2reader import ND2Reader
                    with ND2Reader(nd2_path) as images:
                        meta = getattr(images, "metadata", {}) or {}
                        sample_shape = np.asarray(images[0])
                        num_channels = 1
                        if sample_shape.ndim == 3:
                            num_channels = sample_shape.shape[2]
                        ch_map = nd2_to_mp4.classify_channel_names(meta, num_channels)
                elif getattr(nd2_to_mp4, "_HAS_PIMS", False):
                    import pims
                    with pims.open(nd2_path) as images:
                        try:
                            meta = getattr(images, "metadata", {}) or {}
                        except Exception:
                            meta = {}
                        sample_shape = np.asarray(images[0])
                        num_channels = 1
                        if sample_shape.ndim == 3:
                            num_channels = sample_shape.shape[2]
                        ch_map = nd2_to_mp4.classify_channel_names(meta, num_channels)
            except Exception as e:
                print(f"Could not extract channel mapping: {e}")
        
        # Print channel information if available
        try:
            if hasattr(nd2_to_mp4, 'print_channel_info'):
                # Try to get channel info from the converter
                try:
                    if getattr(nd2_to_mp4, "_HAS_ND2READER", False):
                        from nd2reader import ND2Reader
                        with ND2Reader(nd2_path) as images:
                            meta = getattr(images, "metadata", {}) or {}
                            sample_shape = np.asarray(images[0])
                            num_channels = 1
                            if sample_shape.ndim == 3:
                                num_channels = sample_shape.shape[2]
                            ch_map_info = nd2_to_mp4.classify_channel_names(meta, num_channels)
                            nd2_to_mp4.print_channel_info(nd2_path, meta, num_channels, ch_map_info)
                    elif getattr(nd2_to_mp4, "_HAS_PIMS", False):
                        import pims
                        with pims.open(nd2_path) as images:
                            try:
                                meta = getattr(images, "metadata", {}) or {}
                            except Exception:
                                meta = {}
                            sample_shape = np.asarray(images[0])
                            num_channels = 1
                            if sample_shape.ndim == 3:
                                num_channels = sample_shape.shape[2]
                            ch_map_info = nd2_to_mp4.classify_channel_names(meta, num_channels)
                            nd2_to_mp4.print_channel_info(nd2_path, meta, num_channels, ch_map_info)
                except Exception as e:
                    print(f"Could not extract channel info: {e}")
        except Exception:
            pass
        
        # If separate-channels mode, create individual channel MP4s
        if args.separate_channels:
            if not ch_map:
                print(f"⚠️  Warning: No channel mapping detected for {nd2_path}")
                print("   Falling back to normal mode (combined color MP4)")
            else:
                # Check if we have any fluorescence channels
                fluorescence_channels = [ch for ch in ['red', 'green', 'blue'] if ch in ch_map]
                if fluorescence_channels:
                    print(f"Creating separate MP4 files for {len(fluorescence_channels)} fluorescence channel(s)...")
                    # Export percentiles via env for the converter
                    if args.norm_mode == "percentile":
                        os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
                        os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
                    else:
                        os.environ.pop("ND2_NORM_LOWER_PCT", None)
                        os.environ.pop("ND2_NORM_UPPER_PCT", None)
                    
                    try:
                        created_channels = convert_individual_channels_to_mp4(
                            nd2_path=nd2_path,
                            base_output_path=out_path,
                            ch_map=ch_map,
                            default_fps=float(args.fps),
                            use_clahe=use_clahe,
                            clip_limit=float(args.clip_limit),
                            tile_grid=int(args.tile_grid),
                            preferred_codec=str(args.codec) if args.codec else None,
                            use_ffmpeg=use_ffmpeg,
                        )
                        
                        # Add created channel MP4s to the list
                        if created_channels:
                            for channel_name, channel_path in created_channels:
                                if os.path.exists(channel_path):
                                    mp4_paths.append(channel_path)
                                    # Add metadata for each channel (use same metadata as original)
                                    metadata_list.append(metadata)
                            # Skip creating the combined color MP4 in separate-channels mode
                            continue
                        else:
                            print(f"⚠️  Warning: No channel MP4s were created for {nd2_path}")
                            print("   Falling back to normal mode (combined color MP4)")
                    except Exception as exc:
                        print(f"ERROR in separate-channels mode for {nd2_path}: {exc}")
                        print("   Falling back to normal mode (combined color MP4)")
                else:
                    print(f"⚠️  Warning: No fluorescence channels (red/green/blue) detected for {nd2_path}")
                    print("   Falling back to normal mode (combined color MP4)")
        
        # Normal mode: create combined color MP4
        if os.path.exists(out_path) and args.no_overwrite:
            print(f"Skip existing: {out_path}")
        else:
            print(f"Converting: {nd2_path}\n  -> {out_path}")
            try:
                # Export percentiles via env for the converter
                if args.norm_mode == "percentile":
                    os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
                    os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
                else:
                    os.environ.pop("ND2_NORM_LOWER_PCT", None)
                    os.environ.pop("ND2_NORM_UPPER_PCT", None)

                nd2_to_mp4.convert_nd2_to_mp4(
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
        if os.path.exists(out_path):
            mp4_paths.append(out_path)
            
        # Check for brightfield grayscale MP4 file
        brightfield_path = os.path.join(subdir, f"{base}_brightfield.mp4")
        if os.path.exists(brightfield_path):
            mp4_paths.append(brightfield_path)
            # Add metadata for brightfield (use same metadata as original)
            metadata_list.append(metadata)

    if not found_any:
        print("No .nd2 files found in:", root_dir, "(searched recursively)")
        sys.exit(0)

    if not mp4_paths:
        print("No MP4 files produced. Aborting Keynote creation.")
        sys.exit(1)

    mp4_paths.sort()
    print(f"Prepared {len(mp4_paths)} MP4 file(s) for Keynote.")

    # 2) Build and run AppleScript to assemble Keynote
    default_base = os.path.basename(os.path.abspath(root_dir).rstrip(os.sep)) or "Presentation"
    out_name = args.output.strip() or f"{default_base}.key"
    if not out_name.lower().endswith('.key'):
        out_name += '.key'
    out_key_path = os.path.join(root_dir, out_name)

    ascript = build_applescript(root_dir, mp4_paths, metadata_list, args.theme, out_key_path)
    print("\nCreating Keynote presentation...")
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
    print("✅ Saved Keynote:", out_key_path)


if __name__ == "__main__":
    main()


