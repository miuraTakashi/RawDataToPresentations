#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from ND2 volume files.

This script processes ND2 volume files (3D stacks) to create:
- Center: Movie animation through Z-stack (Z-axis as time)
- Right: XY cross-section at the middle Z-slice
- Bottom: XZ cross-section at the middle Y-slice

Requirements:
- macOS with Keynote installed
- Python 3
- nd2reader or pims, opencv-python, numpy, PIL (Pillow)

Usage:
  python nd2volumes_to_keynote.py --input "BisT-x100x2Volumend2.nd2" --output "VolumePresentation.key"
  python nd2volumes_to_keynote.py --input "/path/to/dir" --theme "White"
"""

import argparse
import os
import subprocess
import sys
import tempfile
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

# Add script directory to Python path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import common utilities
try:
    import nd2_utils
    from nd2_utils import (
        normalize_to_uint8,
        classify_channel_names,
        apply_clahe_if_needed,
        create_colored_frame,
        get_pixel_size_um,
    )
    _HAS_ND2_UTILS = True
except ImportError:
    _HAS_ND2_UTILS = False

try:
    import nd2_to_mp4
except Exception as exc:
    sys.stderr.write("ERROR: Unable to import nd2_to_mp4. Ensure it exists beside this script.\n")
    raise

# Use nd2_utils for backend detection if available
if _HAS_ND2_UTILS:
    _HAS_ND2READER = nd2_utils._HAS_ND2READER
    _HAS_PIMS = nd2_utils._HAS_PIMS
    ND2Reader = nd2_utils.ND2Reader
    pims = nd2_utils.pims
else:
    # Fallback: local imports
    try:
        from nd2reader import ND2Reader
        _HAS_ND2READER = True
    except Exception:
        ND2Reader = None
        _HAS_ND2READER = False

    try:
        import pims
        _HAS_PIMS = True
    except Exception:
        pims = None
        _HAS_PIMS = False

try:
    import cv2
except Exception as exc:
    sys.stderr.write("ERROR: opencv-python is required. Install with: pip install opencv-python\n")
    raise


# Fallback definitions if nd2_utils not available
if not _HAS_ND2_UTILS:
    def normalize_to_uint8(data: np.ndarray, clip_min: Optional[float] = None, clip_max: Optional[float] = None) -> np.ndarray:
        """Normalize array to uint8 [0, 255] range."""
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
    
    def apply_clahe_if_needed(gray_u8: np.ndarray, use_clahe: bool, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
        """Apply CLAHE if enabled."""
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
            return clahe.apply(gray_u8)
        return gray_u8


def get_channel_colors(meta: dict, num_channels: int) -> dict:
    """Get color mapping for channels based on metadata."""
    ch_map = {}
    try:
        if _HAS_ND2_UTILS:
            ch_map = classify_channel_names(meta, num_channels)
        elif hasattr(nd2_to_mp4, 'classify_channel_names'):
            ch_map = nd2_to_mp4.classify_channel_names(meta, num_channels)
    except Exception:
        pass
    return ch_map


# Use nd2_utils.create_colored_frame if available, otherwise define locally
if not _HAS_ND2_UTILS:
    def create_colored_frame(frame: np.ndarray, ch_map: dict, use_clahe: bool = True,
                             clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
        """Create a colored BGR frame from multi-channel data."""
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


def create_z_stack_movie(nd2_path: str, output_path: str, fps: float = 10.0, 
                         use_clahe: bool = True, clip_limit: float = 3.0, 
                         tile_grid: int = 8, verbose: bool = False) -> bool:
    """Create MP4 movie from Z-stack (treating Z-axis as time) with color channels."""
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                # Configure to iterate over Z-axis with channels
                images.iter_axes = "z"
                if "c" in getattr(images, "sizes", {}):
                    images.bundle_axes = "yxc"
                else:
                    images.bundle_axes = "yx"
                
                sizes = getattr(images, 'sizes', {}) or {}
                num_z = sizes.get('z', len(images))
                
                if num_z == 0:
                    print(f"  Error: No Z-slices found in {nd2_path}")
                    return False
                
                # Get channel mapping
                meta = getattr(images, 'metadata', {}) or {}
                sample_frame = nd2_to_mp4.to_native_endian(np.asarray(images[0]))
                num_channels = sample_frame.shape[2] if sample_frame.ndim == 3 else 1
                ch_map = get_channel_colors(meta, num_channels)
                
                print(f"  Creating Z-stack movie: {num_z} slices, {num_channels} channel(s)")
                if ch_map:
                    print(f"    Channel mapping: {ch_map}")
                    if verbose:
                        for color, idx in ch_map.items():
                            emoji = {'red': '🔴', 'green': '🟢', 'blue': '🔵'}.get(color, '⭕')
                            print(f"      {emoji} Channel {idx} → {color.upper()}")
                
                # Read all Z-slices
                frames = []
                for z_idx in range(num_z):
                    try:
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_idx]))
                        frames.append(frame)
                    except Exception as e:
                        print(f"    Warning: Failed to read Z-slice {z_idx}: {e}")
                        continue
                
                if not frames:
                    print(f"  Error: No frames could be read")
                    return False
                
                # Create colored frames
                h, w = frames[0].shape[:2]
                colored_frames = []
                
                for frame in frames:
                    bgr = create_colored_frame(frame, ch_map, use_clahe, clip_limit, tile_grid)
                    colored_frames.append(bgr)
                
                # Create MP4 (color video)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)  # True for color
                
                if not writer.isOpened():
                    print(f"  Error: Could not open video writer for {output_path}")
                    return False
                
                for frame in colored_frames:
                    writer.write(frame)
                
                writer.release()
                print(f"  ✅ Created Z-stack movie: {os.path.basename(output_path)}")
                return True
        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                num_z = len(images)
                if num_z == 0:
                    print(f"  Error: No Z-slices found")
                    return False
                
                # Get channel info
                try:
                    meta = getattr(images, "metadata", {}) or {}
                except Exception:
                    meta = {}
                
                sample_frame = nd2_to_mp4.to_native_endian(np.asarray(images[0]))
                num_channels = sample_frame.shape[2] if sample_frame.ndim == 3 else 1
                ch_map = get_channel_colors(meta, num_channels)
                
                print(f"  Creating Z-stack movie: {num_z} slices, {num_channels} channel(s)")
                if ch_map and verbose:
                    for color, idx in ch_map.items():
                        emoji = {'red': '🔴', 'green': '🟢', 'blue': '🔵'}.get(color, '⭕')
                        print(f"      {emoji} Channel {idx} → {color.upper()}")
                
                frames = []
                for z_idx in range(num_z):
                    try:
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_idx]))
                        frames.append(frame)
                    except Exception as e:
                        print(f"    Warning: Failed to read Z-slice {z_idx}: {e}")
                        continue
                
                if not frames:
                    return False
                
                h, w = frames[0].shape[:2]
                colored_frames = []
                
                for frame in frames:
                    bgr = create_colored_frame(frame, ch_map, use_clahe, clip_limit, tile_grid)
                    colored_frames.append(bgr)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
                
                if not writer.isOpened():
                    return False
                
                for frame in colored_frames:
                    writer.write(frame)
                
                writer.release()
                print(f"  ✅ Created Z-stack movie: {os.path.basename(output_path)}")
                return True
        
        return False
    
    except Exception as e:
        print(f"  Error creating Z-stack movie: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_xy_cross_section(nd2_path: str, z_index: int, output_path: str, use_clahe: bool = True) -> bool:
    """Extract XY cross-section at specified Z-index."""
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                images.iter_axes = "z"
                images.bundle_axes = "yx"
                
                sizes = getattr(images, 'sizes', {}) or {}
                num_z = sizes.get('z', len(images))
                
                if z_index >= num_z:
                    z_index = num_z // 2
                
                frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_index]))
                if frame.ndim == 3:
                    frame = frame[:, :, 0] if frame.shape[2] > 0 else np.mean(frame, axis=2)
                
                # Normalize
                clip_min = np.percentile(frame, 1.0)
                clip_max = np.percentile(frame, 99.5)
                gray_u8 = nd2_to_mp4.normalize_to_uint8_with_clip(frame, clip_min, clip_max)
                
                if use_clahe:
                    gray_u8 = nd2_to_mp4.apply_clahe_if_needed(gray_u8, True, 3.0, 8)
                
                # Save as JPG
                pil_image = Image.fromarray(gray_u8, mode='L')
                pil_image.save(output_path, 'JPEG', quality=95)
                print(f"  ✅ Created XY cross-section (Z={z_index}): {os.path.basename(output_path)}")
                return True
        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                num_z = len(images)
                if z_index >= num_z:
                    z_index = num_z // 2
                
                frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_index]))
                if frame.ndim == 3:
                    frame = frame[:, :, 0] if frame.shape[2] > 0 else np.mean(frame, axis=2)
                
                clip_min = np.percentile(frame, 1.0)
                clip_max = np.percentile(frame, 99.5)
                gray_u8 = nd2_to_mp4.normalize_to_uint8_with_clip(frame, clip_min, clip_max)
                
                if use_clahe:
                    gray_u8 = nd2_to_mp4.apply_clahe_if_needed(gray_u8, True, 3.0, 8)
                
                pil_image = Image.fromarray(gray_u8, mode='L')
                pil_image.save(output_path, 'JPEG', quality=95)
                print(f"  ✅ Created XY cross-section (Z={z_index}): {os.path.basename(output_path)}")
                return True
        
        return False
    
    except Exception as e:
        print(f"  Error extracting XY cross-section: {e}")
        return False


def extract_xz_cross_section(nd2_path: str, y_index: int, output_path: str, 
                             use_clahe: bool = True, clip_limit: float = 3.0, 
                             tile_grid: int = 8) -> bool:
    """Extract XZ cross-section at specified Y-index with color channels."""
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                images.iter_axes = "z"
                if "c" in getattr(images, "sizes", {}):
                    images.bundle_axes = "yxc"
                else:
                    images.bundle_axes = "yx"
                
                sizes = getattr(images, 'sizes', {}) or {}
                num_z = sizes.get('z', len(images))
                num_y = sizes.get('y', 256)
                
                if y_index >= num_y:
                    y_index = num_y // 2
                
                # Get channel mapping
                meta = getattr(images, 'metadata', {}) or {}
                sample_frame = nd2_to_mp4.to_native_endian(np.asarray(images[0]))
                num_channels = sample_frame.shape[2] if sample_frame.ndim == 3 else 1
                ch_map = get_channel_colors(meta, num_channels)
                
                # Read all Z-slices and extract Y-row (with all channels)
                xz_slices = []
                for z_idx in range(num_z):
                    try:
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_idx]))
                        if y_index < frame.shape[0]:
                            row = frame[y_index, :]  # Shape: (X,) or (X, C)
                            xz_slices.append(row)
                    except Exception as e:
                        continue
                
                if not xz_slices:
                    print(f"  Error: Could not extract XZ cross-section")
                    return False
                
                # Stack rows: shape becomes (Z, X) or (Z, X, C)
                xz_image = np.array(xz_slices)
                
                # Create colored image
                bgr = create_colored_frame(xz_image, ch_map, use_clahe, clip_limit, tile_grid)
                
                # Save as JPG (convert BGR to RGB for PIL)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb, mode='RGB')
                pil_image.save(output_path, 'JPEG', quality=95)
                print(f"  ✅ Created XZ cross-section (Y={y_index}): {os.path.basename(output_path)}")
                return True
        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                num_z = len(images)
                if num_z == 0:
                    return False
                
                # Get channel info
                try:
                    meta = getattr(images, "metadata", {}) or {}
                except Exception:
                    meta = {}
                
                first_frame = nd2_to_mp4.to_native_endian(np.asarray(images[0]))
                num_channels = first_frame.shape[2] if first_frame.ndim == 3 else 1
                ch_map = get_channel_colors(meta, num_channels)
                num_y = first_frame.shape[0]
                
                if y_index >= num_y:
                    y_index = num_y // 2
                
                xz_slices_pims = []
                for z_idx in range(num_z):
                    try:
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_idx]))
                        if y_index < frame.shape[0]:
                            row = frame[y_index, :]
                            xz_slices_pims.append(row)
                    except Exception as e:
                        continue
                
                if not xz_slices_pims:
                    return False
                
                xz_image = np.array(xz_slices_pims)
                bgr = create_colored_frame(xz_image, ch_map, use_clahe, clip_limit, tile_grid)
                
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb, mode='RGB')
                pil_image.save(output_path, 'JPEG', quality=95)
                print(f"  ✅ Created XZ cross-section (Y={y_index}): {os.path.basename(output_path)}")
                return True
        
        return False
    
    except Exception as e:
        print(f"  Error extracting XZ cross-section: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_yz_cross_section(nd2_path: str, x_index: int, output_path: str, 
                             use_clahe: bool = True, clip_limit: float = 3.0, 
                             tile_grid: int = 8) -> bool:
    """Extract YZ cross-section at specified X-index with color channels."""
    try:
        if _HAS_ND2READER:
            with ND2Reader(nd2_path) as images:
                images.iter_axes = "z"
                if "c" in getattr(images, "sizes", {}):
                    images.bundle_axes = "yxc"
                else:
                    images.bundle_axes = "yx"
                
                sizes = getattr(images, 'sizes', {}) or {}
                num_z = sizes.get('z', len(images))
                num_x = sizes.get('x', 256)
                
                if x_index >= num_x:
                    x_index = num_x // 2
                
                # Get channel mapping
                meta = getattr(images, 'metadata', {}) or {}
                sample_frame = nd2_to_mp4.to_native_endian(np.asarray(images[0]))
                num_channels = sample_frame.shape[2] if sample_frame.ndim == 3 else 1
                ch_map = get_channel_colors(meta, num_channels)
                
                # Read all Z-slices and extract X-column (with all channels)
                yz_slices = []
                for z_idx in range(num_z):
                    try:
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_idx]))
                        if x_index < frame.shape[1]:
                            col = frame[:, x_index]  # Shape: (Y,) or (Y, C)
                            yz_slices.append(col)
                    except Exception as e:
                        continue
                
                if not yz_slices:
                    print(f"  Error: Could not extract YZ cross-section")
                    return False
                
                # Stack columns: shape becomes (Z, Y) or (Z, Y, C)
                yz_stacked = np.array(yz_slices)
                # Transpose to get (Y, Z) or (Y, Z, C) - Y is vertical, Z is horizontal
                if yz_stacked.ndim == 2:
                    yz_image = yz_stacked.T
                else:
                    yz_image = np.transpose(yz_stacked, (1, 0, 2))
                
                # Create colored image
                bgr = create_colored_frame(yz_image, ch_map, use_clahe, clip_limit, tile_grid)
                
                # Save as JPG (convert BGR to RGB for PIL)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb, mode='RGB')
                pil_image.save(output_path, 'JPEG', quality=95)
                print(f"  ✅ Created YZ cross-section (X={x_index}): {os.path.basename(output_path)}")
                return True
        
        elif _HAS_PIMS:
            with pims.open(nd2_path) as images:
                num_z = len(images)
                if num_z == 0:
                    return False
                
                # Get channel info
                try:
                    meta = getattr(images, "metadata", {}) or {}
                except Exception:
                    meta = {}
                
                first_frame = nd2_to_mp4.to_native_endian(np.asarray(images[0]))
                num_channels = first_frame.shape[2] if first_frame.ndim == 3 else 1
                ch_map = get_channel_colors(meta, num_channels)
                num_x = first_frame.shape[1]
                
                if x_index >= num_x:
                    x_index = num_x // 2
                
                yz_slices = []
                for z_idx in range(num_z):
                    try:
                        frame = nd2_to_mp4.to_native_endian(np.asarray(images[z_idx]))
                        if x_index < frame.shape[1]:
                            col = frame[:, x_index]
                            yz_slices.append(col)
                    except Exception as e:
                        continue
                
                if not yz_slices:
                    return False
                
                # Stack and transpose
                yz_stacked = np.array(yz_slices)
                if yz_stacked.ndim == 2:
                    yz_image = yz_stacked.T
                else:
                    yz_image = np.transpose(yz_stacked, (1, 0, 2))
                
                bgr = create_colored_frame(yz_image, ch_map, use_clahe, clip_limit, tile_grid)
                
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb, mode='RGB')
                pil_image.save(output_path, 'JPEG', quality=95)
                print(f"  ✅ Created YZ cross-section (X={x_index}): {os.path.basename(output_path)}")
                return True
        
        return False
    
    except Exception as e:
        print(f"  Error extracting YZ cross-section: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_applescript(
    dir_path: str,
    movie_path: str,
    xz_image_path: str,
    yz_image_path: str,
    slide_title: str,
    theme_name: str,
    out_key_path: str,
    nx: int,
    ny: int,
    nz: int,
    x_um: float,
    y_um: float,
    z_um: float
) -> str:
    """Build AppleScript to create Keynote slide with movie and cross-sections."""
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")
    
    movie_posix = esc(movie_path)
    xz_posix = esc(xz_image_path)
    yz_posix = esc(yz_image_path)
    title_escaped = esc(slide_title)
    out_posix = esc(out_key_path)
    
    lines = []
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
    
    # Set title
    lines.append(f'            set object text of text item 1 to "{title_escaped}"')
    
    # Calculate scale factor to fit all images in 512x512
    # Total layout size: width = nx + 10 + nz, height = ny + 10 + nz
    total_width = nx + 10 + nz
    total_height = ny + 10 + nz
    scale = min(512.0 / total_width, 512.0 / total_height)
    
    # Base position
    base_x = 30
    base_y = 180
    
    # Scaled dimensions and positions
    movie_w = int(nx * scale)
    movie_h = int(ny * scale)
    movie_x = base_x
    movie_y = base_y
    
    yz_w = int(nz * scale)
    yz_h = int(ny * scale)
    yz_x = base_x + int((nx + 10) * scale)
    yz_y = base_y
    
    xz_w = int(nx * scale)
    xz_h = int(nz * scale)
    xz_x = base_x
    xz_y = base_y + int((ny + 10) * scale)
    
    # Insert movie
    lines.append('            -- Insert movie (scaled)')
    lines.append(f'            set movieFile to alias (POSIX file "{movie_posix}")')
    lines.append(f'            set movieImage to make new image with properties {{file:movieFile, position:{{{movie_x}, {movie_y}}}, width:{movie_w}, height:{movie_h}}}')
    
    # Insert YZ cross-section
    lines.append('            -- Insert YZ cross-section (scaled)')
    lines.append(f'            set yzImageFile to alias (POSIX file "{yz_posix}")')
    lines.append(f'            set yzImage to make new image with properties {{file:yzImageFile, position:{{{yz_x}, {yz_y}}}, width:{yz_w}, height:{yz_h}}}')
    
    # Insert XZ cross-section
    lines.append('            -- Insert XZ cross-section (scaled)')
    lines.append(f'            set xzImageFile to alias (POSIX file "{xz_posix}")')
    lines.append(f'            set xzImage to make new image with properties {{file:xzImageFile, position:{{{xz_x}, {xz_y}}}, width:{xz_w}, height:{xz_h}}}')
    
    # Add volume information to text area (use AppleScript's return for line breaks)
    volume_line1 = esc("Volume Information:")
    volume_line2 = esc(f"X: {x_um:.1f} µm ({nx} px)")
    volume_line3 = esc(f"Y: {y_um:.1f} µm ({ny} px)")
    volume_line4 = esc(f"Z: {z_um:.1f} µm ({nz} slices)")
    volume_info_as = f'"{volume_line1}" & return & "{volume_line2}" & return & "{volume_line3}" & return & "{volume_line4}"'
    lines.append('            -- Add volume information to text area')
    lines.append('            try')
    lines.append('                if (count of text items) > 1 then')
    lines.append(f'                    set object text of text item 2 to {volume_info_as}')
    lines.append('                else')
    lines.append(f'                    set volumeInfoBox to make new text item with properties {{object text:{volume_info_as}, position:{{600, 180}}, width:200, height:150}}')
    lines.append('                end if')
    lines.append('            on error')
    lines.append(f'                set volumeInfoBox to make new text item with properties {{object text:{volume_info_as}, position:{{600, 180}}, width:200, height:150}}')
    lines.append('            end try')
    
    lines.append('        end tell')
    lines.append('    end tell')
    
    lines.append(f'    save theDoc in (POSIX file "{out_posix}")')
    lines.append('end tell')
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Keynote from ND2 volume files (3D Z-stacks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single ND2 volume file
  python nd2volumes_to_keynote.py --input "volume.nd2"
  
  # Process all ND2 files in a directory
  python nd2volumes_to_keynote.py --input "/path/to/dir"
  
  # Custom theme and output name
  python nd2volumes_to_keynote.py --input "volume.nd2" --theme "Black" --output "MyVolume.key"
  
  # Adjust Z-stack movie speed
  python nd2volumes_to_keynote.py --input "volume.nd2" --fps 15
  
  # Keep temporary files for inspection
  python nd2volumes_to_keynote.py --input "volume.nd2" --keep-temp
"""
    )
    
    # Input/Output options
    parser.add_argument("--input", type=str, default=os.getcwd(), 
                        help="Input ND2 file or directory (default: current directory)")
    parser.add_argument("--theme", type=str, default="White", 
                        help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", 
                        help="Output .key file name (default: auto-generated)")
    
    # Movie/image processing options
    parser.add_argument("--fps", type=float, default=10.0, 
                        help="FPS for Z-stack movie (default: 10)")
    parser.add_argument("--clip-limit", type=float, default=3.0, 
                        help="CLAHE clip limit (default: 3.0)")
    parser.add_argument("--tile-grid", type=int, default=8, 
                        help="CLAHE tile grid size (N -> NxN) (default: 8)")
    parser.add_argument("--no-clahe", action="store_true", 
                        help="Disable CLAHE contrast enhancement")
    parser.add_argument("--codec", type=str, default="avc1", 
                        help="Preferred fourcc codec for MP4 (default: avc1)")
    
    # Normalization options
    parser.add_argument("--norm-mode", choices=["percentile", "minmax"], default="percentile", 
                        help="Global normalization mode (default: percentile)")
    parser.add_argument("--lower-pct", type=float, default=1.0, 
                        help="Lower percentile for histogram clipping (default: 1.0)")
    parser.add_argument("--upper-pct", type=float, default=99.5, 
                        help="Upper percentile for histogram clipping (default: 99.5)")
    
    # File handling options
    parser.add_argument("--no-overwrite", action="store_true", 
                        help="Skip existing files instead of overwriting them")
    parser.add_argument("--no-ffmpeg", action="store_true", 
                        help="Disable FFmpeg and use OpenCV VideoWriter instead")
    parser.add_argument("--keep-temp", action="store_true", 
                        help="Keep temporary MP4 and JPG files")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose output including channel detection details")
    
    args = parser.parse_args()
    
    if not _HAS_ND2READER and not _HAS_PIMS:
        sys.stderr.write("ERROR: No ND2 backend available. Please install: pip install nd2reader pims\n")
        sys.exit(1)
    
    # Check FFmpeg availability
    use_ffmpeg = not args.no_ffmpeg
    if use_ffmpeg:
        if getattr(nd2_to_mp4, "check_ffmpeg_availability", lambda: False)():
            print("✅ FFmpeg detected - will use QuickTime-compatible encoding")
        else:
            print("⚠️  FFmpeg not found - falling back to OpenCV VideoWriter")
            use_ffmpeg = False
    else:
        print("📹 Using OpenCV VideoWriter (FFmpeg disabled by --no-ffmpeg)")
    
    # Set environment variables for normalization
    if args.norm_mode == "percentile":
        os.environ["ND2_NORM_LOWER_PCT"] = str(args.lower_pct)
        os.environ["ND2_NORM_UPPER_PCT"] = str(args.upper_pct)
    else:
        os.environ.pop("ND2_NORM_LOWER_PCT", None)
        os.environ.pop("ND2_NORM_UPPER_PCT", None)
    
    input_path = os.path.abspath(args.input)
    
    # Determine if input is file or directory
    if os.path.isfile(input_path):
        nd2_files = [input_path]
        base_dir = os.path.dirname(input_path)
    elif os.path.isdir(input_path):
        # Find all ND2 files recursively
        nd2_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.nd2') and not file.startswith('.'):
                    nd2_files.append(os.path.join(root, file))
        nd2_files.sort()
        base_dir = input_path
    else:
        sys.stderr.write(f"Input path does not exist: {input_path}\n")
        sys.exit(1)
    
    if not nd2_files:
        print("No ND2 files found.")
        sys.exit(0)
    
    print(f"Found {len(nd2_files)} ND2 file(s)")
    
    use_clahe = not args.no_clahe
    
    # Process each ND2 file
    for nd2_path in nd2_files:
        print(f"\nProcessing: {os.path.basename(nd2_path)}")
        
        # Create temporary directory for outputs
        base_name = os.path.splitext(os.path.basename(nd2_path))[0]
        temp_dir = os.path.join(base_dir, f"temp_{base_name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        movie_path = os.path.join(temp_dir, f"{base_name}_zstack.mp4")
        xz_image_path = os.path.join(temp_dir, f"{base_name}_xz.jpg")
        yz_image_path = os.path.join(temp_dir, f"{base_name}_yz.jpg")
        
        # Get volume dimensions and pixel size
        try:
            pixel_size_xy = 1.0  # Default pixel size in µm
            pixel_size_z = 1.0   # Default Z step size in µm
            
            if _HAS_ND2READER:
                with ND2Reader(nd2_path) as images:
                    sizes = getattr(images, 'sizes', {}) or {}
                    num_z = sizes.get('z', len(images))
                    num_y = sizes.get('y', 256)
                    num_x = sizes.get('x', 256)
                    
                    # Try to get pixel size from metadata
                    meta = getattr(images, 'metadata', {}) or {}
                    if 'pixel_microns' in meta:
                        pixel_size_xy = float(meta['pixel_microns'])
                    
                    # Try to get Z step from z_coordinates
                    z_coords = meta.get('z_coordinates', None)
                    if z_coords is not None and len(z_coords) > 1:
                        # Calculate Z step from coordinates (in µm)
                        z_step = abs(z_coords[1] - z_coords[0])
                        if z_step > 0:
                            pixel_size_z = z_step
            else:
                with pims.open(nd2_path) as images:
                    num_z = len(images)
                    if num_z > 0:
                        first_frame = np.asarray(images[0])
                        if first_frame.ndim == 3:
                            num_y, num_x = first_frame.shape[:2]
                        else:
                            num_y, num_x = first_frame.shape[0], first_frame.shape[1]
                    else:
                        num_y, num_x = 256, 256
            
            # Calculate physical dimensions
            x_um = num_x * pixel_size_xy
            y_um = num_y * pixel_size_xy
            z_um = num_z * pixel_size_z
            
            print(f"  Volume dimensions: X={num_x}, Y={num_y}, Z={num_z}")
            print(f"  Pixel size: XY={pixel_size_xy:.3f} µm, Z={pixel_size_z:.3f} µm")
            print(f"  Physical size: X={x_um:.1f} µm, Y={y_um:.1f} µm, Z={z_um:.1f} µm")
            
            # Create Z-stack movie
            if not create_z_stack_movie(nd2_path, movie_path, args.fps, use_clahe, 
                                        args.clip_limit, args.tile_grid, args.verbose):
                print(f"  ⚠️  Failed to create Z-stack movie")
                continue
            
            # Extract XZ cross-section (middle Y)
            y_mid = num_y // 2
            if not extract_xz_cross_section(nd2_path, y_mid, xz_image_path, use_clahe,
                                            args.clip_limit, args.tile_grid):
                print(f"  ⚠️  Failed to extract XZ cross-section")
                continue
            
            # Extract YZ cross-section (middle X)
            x_mid = num_x // 2
            if not extract_yz_cross_section(nd2_path, x_mid, yz_image_path, use_clahe,
                                            args.clip_limit, args.tile_grid):
                print(f"  ⚠️  Failed to extract YZ cross-section")
                continue
            
            # Create Keynote presentation
            slide_title = base_name
            default_output = f"{base_name}_volume.key"
            out_name = args.output.strip() if args.output else default_output
            if not out_name.lower().endswith('.key'):
                out_name += '.key'
            
            # If multiple files, append index
            if len(nd2_files) > 1:
                idx = nd2_files.index(nd2_path) + 1
                out_name = f"{os.path.splitext(out_name)[0]}_{idx}.key"
            
            out_key_path = os.path.join(base_dir, out_name)
            
            ascript = build_applescript(
                base_dir, movie_path, xz_image_path, yz_image_path,
                slide_title, args.theme, out_key_path,
                num_x, num_y, num_z,
                x_um, y_um, z_um
            )
            
            print(f"\nCreating Keynote presentation...")
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
                continue
            
            print(f"✅ Saved Keynote: {out_key_path}")
            
            # Clean up temporary files unless --keep-temp is specified
            if not args.keep_temp:
                print(f"Cleaning up temporary files...")
                for temp_file in [movie_path, xz_image_path, yz_image_path]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except OSError:
                        pass
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
                print("✅ Cleanup complete")
            else:
                print(f"Temporary files kept in: {temp_dir}")
        
        except Exception as e:
            print(f"  Error processing {nd2_path}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

