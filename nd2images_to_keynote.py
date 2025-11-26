#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from ND2 files by extracting fluorescence channels.

This script processes ND2 files to extract fluorescence channels:
- Blue channel: DAPI
- Green channel: Alexa488 antibody  
- Red channel: Alexa568

The channels are combined into RGB JPG files, and each JPG gets its own slide in Keynote.

Requirements:
- macOS with Keynote installed
- Python 3
- nd2, numpy, PIL (Pillow) packages

Usage:
  python create_keynote_from_nd2.py \
    [--input "/path/to/dir"] [--theme "White"] [--output "MyImages.key"] \
    [--subdirs-only]
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
# import nd2  # Not available in current environment


def extract_side_length_um(meta: Dict[str, Any], height_px: int, width_px: int) -> Optional[float]:
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


def metadata_to_dict(metadata_obj) -> Dict[str, Any]:
    """Convert nd2reader metadata object to dictionary."""
    meta_dict = {}
    
    # Try direct dict conversion
    if isinstance(metadata_obj, dict):
        return metadata_obj
    
    # Try to access as dictionary-like object
    if hasattr(metadata_obj, 'keys'):
        try:
            return dict(metadata_obj)
        except:
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
            except:
                pass
    
    # Try to access as items
    if hasattr(metadata_obj, 'items'):
        try:
            for key, value in metadata_obj.items():
                meta_dict[key] = value
        except:
            pass
    
    return meta_dict


def extract_fluorescence_channels(nd2_path: str, output_dir: str, selected_channels: list[str] = None, single_channel_mode: bool = False) -> tuple[str, float, str]:
    """
    Extract fluorescence channels from ND2 file and create RGB JPG.
    
    Args:
        nd2_path: Path to the ND2 file
        output_dir: Directory to save the JPG file
        selected_channels: List of channels to extract (e.g., ['red', 'green'] or ['0', '1'])
        single_channel_mode: If True, extract only single channel as grayscale
        
    Returns:
        Tuple of (path to the created JPG file, side length in micrometers, original ND2 path)
    """
    try:
        # Note: nd2 module not available, using nd2reader/pims instead
        # Open ND2 file with nd2reader
        if not globals().get('ND2Reader'):
            from nd2reader import ND2Reader
        with ND2Reader(nd2_path) as nd2_file:
            print(f"Processing {os.path.basename(nd2_path)}...")
            
            # Get channel information from metadata
            channel_metas = nd2_file.metadata.channels if hasattr(nd2_file.metadata, 'channels') else []
            print(f"  Channels: {len(channel_metas)} channels found")
            print(f"  Frames: {nd2_file.shape}")
            
            # Get metadata as dictionary (same method as nd2_to_mp4.py)
            # Try multiple ways to access metadata
            metadata = getattr(nd2_file, "metadata", {}) or {}
            
            # Try raw_metadata if available (nd2reader specific)
            if hasattr(nd2_file, 'raw_metadata'):
                try:
                    raw_meta = nd2_file.raw_metadata
                    if isinstance(raw_meta, dict):
                        metadata = raw_meta
                    elif hasattr(raw_meta, '__dict__'):
                        metadata = raw_meta.__dict__
                except:
                    pass
            
            # Convert to dictionary
            meta_dict = metadata_to_dict(metadata)
            
            # Debug: print available metadata keys
            if meta_dict:
                print(f"  Metadata keys found: {list(meta_dict.keys())[:20]}...")
                # Also try to print pixel-related keys
                pixel_keys = [k for k in meta_dict.keys() if 'pixel' in k.lower() or 'calibration' in k.lower() or 'scale' in k.lower()]
                if pixel_keys:
                    print(f"  Pixel/calibration keys: {pixel_keys[:10]}")
            else:
                print(f"  Warning: No metadata keys found")
            
            # Get the image data
            # ND2Reader doesn't have asarray(), use indexing instead
            if len(nd2_file) == 0:
                print(f"      → Error: No frames found in {nd2_path}")
                return None, 0.0, nd2_path
            elif len(nd2_file) == 1:
                # Single frame
                image_data = np.asarray(nd2_file[0])
                print(f"  Processing single frame")
            else:
                # Multiple frames - each frame might be a different channel
                print(f"  Multiple frames detected ({len(nd2_file)})")
                
                # Check if frames have different content (different channels)
                frame_means = []
                for i in range(min(len(nd2_file), 5)):  # Check first 5 frames max
                    frame = np.asarray(nd2_file[i])
                    frame_means.append(frame.mean())
                
                # If frames have significantly different means, treat as separate channels
                if len(frame_means) > 1 and (max(frame_means) - min(frame_means)) > 10:
                    print(f"    → Frames appear to be different channels (means: {[f'{m:.1f}' for m in frame_means]})")
                    # Stack frames as channels: (frames, height, width) -> (frames, height, width)
                    all_frames = []
                    for i in range(len(nd2_file)):
                        frame = np.asarray(nd2_file[i])
                        if frame.ndim == 2:  # Ensure 2D
                            all_frames.append(frame)
                    
                    if all_frames:
                        # Stack frames as channels: (channels, height, width)
                        image_data = np.stack(all_frames, axis=0)
                        print(f"    → Stacked {len(all_frames)} frames as channels")
                    else:
                        image_data = np.asarray(nd2_file[0])
                        print(f"    → Fallback to first frame")
                else:
                    # Frames are similar, probably time series - use first frame
                    image_data = np.asarray(nd2_file[0])
                    print(f"    → Frames appear similar, using first frame")
            
            # Handle different data shapes
            print(f"  Image shape: {image_data.shape}")
            
            if len(image_data.shape) == 2:
                # Shape: (height, width) - single channel grayscale
                channels = 1
                height, width = image_data.shape
                # Convert to 3D array for consistent processing
                image_data = image_data.reshape(1, height, width)
            elif len(image_data.shape) == 3:
                # Could be (channels, height, width) or (height, width, channels)
                # ND2Reader typically uses (height, width, channels) for multi-channel
                if image_data.shape[2] <= 10:  # Assume last dimension is channels if small
                    # Shape: (height, width, channels)
                    height, width, channels = image_data.shape
                    # Transpose to (channels, height, width) for consistent processing
                    image_data = np.transpose(image_data, (2, 0, 1))
                else:
                    # Shape: (channels, height, width)
                    channels, height, width = image_data.shape
            elif len(image_data.shape) == 4:
                # Shape: (frames, height, width, channels) or similar - take first frame
                if image_data.shape[3] <= 10:  # Assume last dimension is channels
                    # Shape: (frames, height, width, channels)
                    height, width, channels = image_data.shape[1], image_data.shape[2], image_data.shape[3]
                    image_data = image_data[0]  # Take first frame
                    # Transpose to (channels, height, width)
                    image_data = np.transpose(image_data, (2, 0, 1))
                else:
                    # Shape: (frames, channels, height, width)
                    channels = image_data.shape[1]
                    image_data = image_data[0]  # Take first frame
            else:
                raise ValueError(f"Unexpected image shape: {image_data.shape}")
            
            print(f"  Final shape: {image_data.shape}")
            print(f"  Number of channels: {channels}")
            
            # Get actual image dimensions
            height, width = image_data.shape[1], image_data.shape[2]
            
            # Calculate side length using extract_side_length_um (same as nd2_to_mp4.py)
            side_length_um = extract_side_length_um(meta_dict, height, width)
            if side_length_um is not None and side_length_um > 0:
                print(f"  Side length: {side_length_um:.1f} µm")
            else:
                print(f"  Warning: Could not determine spatial scale from metadata")
                side_length_um = 0.0
            
            # Initialize RGB channels
            red_channel = np.zeros((height, width), dtype=np.uint8)
            green_channel = np.zeros((height, width), dtype=np.uint8)
            blue_channel = np.zeros((height, width), dtype=np.uint8)
            
            # Initialize channel mapping flags
            dapi_found = False
            alexa488_found = False
            alexa568_found = False
            
            # Channel mapping dictionary
            channel_mapping = {}
            
            # Try to identify channels by name if available
            if channel_metas:
                for i, channel_meta in enumerate(channel_metas):
                    channel_name = channel_meta.channel.name if hasattr(channel_meta, 'channel') else str(i)
                    print(f"    Channel {i}: {channel_name}")
                    
                    channel_name_lower = channel_name.lower()
                    if 'dapi' in channel_name_lower or 'hoechst' in channel_name_lower or 'blue' in channel_name_lower:
                        channel_mapping['blue'] = i
                        dapi_found = True
                        print(f"      → Mapped to Blue (DAPI/Hoechst)")
                    elif ('488' in channel_name_lower or 'alexa488' in channel_name_lower or 
                          'green' in channel_name_lower or 'fitc' in channel_name_lower or
                          'gfp' in channel_name_lower):
                        channel_mapping['green'] = i
                        alexa488_found = True
                        print(f"      → Mapped to Green (Alexa488/FITC/GFP)")
                    elif ('568' in channel_name_lower or 'alx568' in channel_name_lower or 
                          'alexa568' in channel_name_lower or 'red' in channel_name_lower or
                          'cy3' in channel_name_lower or 'rhodamine' in channel_name_lower or
                          'tubulin' in channel_name_lower):
                        channel_mapping['red'] = i
                        alexa568_found = True
                        print(f"      → Mapped to Red (Alexa568/Cy3/Tubulin)")
                    elif 'td' in channel_name_lower or 'transmitted' in channel_name_lower or 'brightfield' in channel_name_lower:
                        # TD (Transmitted Detector) - typically used for brightfield
                        print(f"      → Skipped (Transmitted Detector/Brightfield)")
                    else:
                        print(f"      → Unrecognized channel type: {channel_name}")
                        # If we have unrecognized channels and no mapping yet, assign them
                        if not channel_mapping:
                            if i == 0 and 'blue' not in channel_mapping:
                                channel_mapping['blue'] = i
                                print(f"        → Auto-assigned to Blue (channel {i})")
                            elif i == 1 and 'green' not in channel_mapping:
                                channel_mapping['green'] = i
                                print(f"        → Auto-assigned to Green (channel {i})")
                            elif i == 2 and 'red' not in channel_mapping:
                                channel_mapping['red'] = i
                                print(f"        → Auto-assigned to Red (channel {i})")
            
            # If no channels mapped by name, use default mapping for fluorescence microscopy
            # For multi-frame ND2s: 0=DAPI(Blue), 1=Alexa488(Green), 2=Alexa568(Red)
            if not channel_mapping:
                print(f"    No channel names detected, using default fluorescence mapping:")
                if channels >= 1:
                    channel_mapping['blue'] = 0
                    print(f"      → Channel 0 → Blue (DAPI/Hoechst)")
                if channels >= 2:
                    channel_mapping['green'] = 1
                    print(f"      → Channel 1 → Green (Alexa488/FITC)")
                if channels >= 3:
                    channel_mapping['red'] = 2
                    print(f"      → Channel 2 → Red (Alexa568/Tubulin)")
                if channels > 3:
                    print(f"      → Note: {channels-3} additional channels will be ignored")
            
            # Process selected channels
            if selected_channels:
                print(f"    Selected channels: {', '.join(selected_channels)}")
                
                if single_channel_mode:
                    # Single channel mode - extract only one channel as grayscale
                    if len(selected_channels) != 1:
                        print(f"      → Warning: Single channel mode requires exactly one channel, using first: {selected_channels[0]}")
                    
                    channel_name = selected_channels[0].lower()
                    if channel_name in channel_mapping:
                        channel_idx = channel_mapping[channel_name]
                        gray_channel = normalize_channel(image_data[channel_idx])
                        # Create grayscale image
                        pil_image = Image.fromarray(gray_channel, mode='L')
                        print(f"      → Extracted single channel: {channel_name} (index {channel_idx})")
                    elif channel_name.isdigit() and int(channel_name) < channels:
                        channel_idx = int(channel_name)
                        gray_channel = normalize_channel(image_data[channel_idx])
                        pil_image = Image.fromarray(gray_channel, mode='L')
                        print(f"      → Extracted single channel: index {channel_idx}")
                    else:
                        print(f"      → Error: Channel '{channel_name}' not found or invalid")
                        return None, 0.0, nd2_path
                else:
                    # Multi-channel mode - extract selected channels
                    red_channel = np.zeros((height, width), dtype=np.uint8)
                    green_channel = np.zeros((height, width), dtype=np.uint8)
                    blue_channel = np.zeros((height, width), dtype=np.uint8)
                    
                    for channel_name in selected_channels:
                        channel_name = channel_name.lower()
                        if channel_name in channel_mapping:
                            channel_idx = channel_mapping[channel_name]
                            normalized_data = normalize_channel(image_data[channel_idx])
                            
                            if channel_name == 'red':
                                red_channel = normalized_data
                            elif channel_name == 'green':
                                green_channel = normalized_data
                            elif channel_name == 'blue':
                                blue_channel = normalized_data
                            print(f"      → Extracted channel: {channel_name} (index {channel_idx})")
                        elif channel_name.isdigit() and int(channel_name) < channels:
                            channel_idx = int(channel_name)
                            normalized_data = normalize_channel(image_data[channel_idx])
                            
                            # Map by position: 0=red, 1=green, 2=blue
                            if channel_idx == 0:
                                red_channel = normalized_data
                            elif channel_idx == 1:
                                green_channel = normalized_data
                            elif channel_idx == 2:
                                blue_channel = normalized_data
                            print(f"      → Extracted channel: index {channel_idx}")
                        else:
                            print(f"      → Warning: Channel '{channel_name}' not found, skipping")
                    
                # Debug: Check channel values before creating RGB image
                print(f"    Debug - Channel statistics:")
                print(f"      Red channel: min={red_channel.min()}, max={red_channel.max()}, mean={red_channel.mean():.1f}")
                print(f"      Green channel: min={green_channel.min()}, max={green_channel.max()}, mean={green_channel.mean():.1f}")
                print(f"      Blue channel: min={blue_channel.min()}, max={blue_channel.max()}, mean={blue_channel.mean():.1f}")
                
                # Create RGB image
                rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)
                print(f"    RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                pil_image = Image.fromarray(rgb_image)
            else:
                # Default behavior - extract all available channels using channel_mapping
                print(f"    Channel mapping: {channel_mapping}")
                print(f"    Available channels: {channels}")
                
                # Extract channels based on mapping
                if 'red' in channel_mapping:
                    red_channel = normalize_channel(image_data[channel_mapping['red']])
                    print(f"      → Extracted Red channel (index {channel_mapping['red']})")
                    print(f"        Red stats: min={red_channel.min()}, max={red_channel.max()}, mean={red_channel.mean():.1f}")
                else:
                    print(f"      → Red channel not mapped, using zeros")
                
                if 'green' in channel_mapping:
                    green_channel = normalize_channel(image_data[channel_mapping['green']])
                    print(f"      → Extracted Green channel (index {channel_mapping['green']})")
                    print(f"        Green stats: min={green_channel.min()}, max={green_channel.max()}, mean={green_channel.mean():.1f}")
                else:
                    print(f"      → Green channel not mapped, using zeros")
                
                if 'blue' in channel_mapping:
                    blue_channel = normalize_channel(image_data[channel_mapping['blue']])
                    print(f"      → Extracted Blue channel (index {channel_mapping['blue']})")
                    print(f"        Blue stats: min={blue_channel.min()}, max={blue_channel.max()}, mean={blue_channel.mean():.1f}")
                else:
                    print(f"      → Blue channel not mapped, using zeros")
                
                # If no mapping was created (no channel names detected) or --all-channels is specified
                force_all_channels = getattr(extract_fluorescence_channels, '_all_channels_flag', False)
                if (not channel_mapping and channels > 0) or force_all_channels:
                    if force_all_channels:
                        print(f"    Force all channels mode: using all {channels} channels:")
                    else:
                        print(f"    No channel mapping found, using all {channels} channels:")
                    if channels >= 1:
                        blue_channel = normalize_channel(image_data[0])
                        print(f"      → Channel 0 → Blue")
                    if channels >= 2:
                        green_channel = normalize_channel(image_data[1])
                        print(f"      → Channel 1 → Green")
                    if channels >= 3:
                        red_channel = normalize_channel(image_data[2])
                        print(f"      → Channel 2 → Red")
                    # If more than 3 channels, use the first 3
                    if channels > 3:
                        print(f"      → Note: Using first 3 of {channels} channels")
                
                # Debug: Check channel values before creating RGB image
                print(f"    Debug - Channel statistics:")
                print(f"      Red channel: min={red_channel.min()}, max={red_channel.max()}, mean={red_channel.mean():.1f}")
                print(f"      Green channel: min={green_channel.min()}, max={green_channel.max()}, mean={green_channel.mean():.1f}")
                print(f"      Blue channel: min={blue_channel.min()}, max={blue_channel.max()}, mean={blue_channel.mean():.1f}")
                
                # Create RGB image
                rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)
                print(f"    RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                pil_image = Image.fromarray(rgb_image)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(nd2_path))[0]
            jpg_path = os.path.join(output_dir, f"{base_name}_fluorescence.jpg")
            
            # Save with high quality
            pil_image.save(jpg_path, 'JPEG', quality=95)
            print(f"  Saved: {os.path.basename(jpg_path)}")
            
            # Debug: Save individual channels for inspection
            if hasattr(extract_fluorescence_channels, '_debug_channels') and extract_fluorescence_channels._debug_channels:
                debug_dir = os.path.join(output_dir, "debug_channels")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save individual channels
                Image.fromarray(red_channel).save(os.path.join(debug_dir, f"{base_name}_red.jpg"))
                Image.fromarray(green_channel).save(os.path.join(debug_dir, f"{base_name}_green.jpg"))
                Image.fromarray(blue_channel).save(os.path.join(debug_dir, f"{base_name}_blue.jpg"))
                print(f"  Debug: Individual channels saved to {debug_dir}")
            
            return jpg_path, side_length_um, nd2_path
            
    except Exception as e:
        print(f"Error processing {nd2_path}: {e}")
        return None, 0.0, nd2_path

def normalize_channel(channel_data: np.ndarray) -> np.ndarray:
    """
    Normalize a single channel to 0-255 range for display.
    
    Args:
        channel_data: Raw channel data
        
    Returns:
        Normalized channel data as uint8
    """
    # Remove any NaN or infinite values
    channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to 0-255 range
    if channel_data.max() > channel_data.min():
        normalized = ((channel_data - channel_data.min()) / 
                     (channel_data.max() - channel_data.min()) * 255)
    else:
        normalized = np.zeros_like(channel_data)
    
    return normalized.astype(np.uint8)

def build_applescript(dir_path: str, jpg_data: list[tuple[str, float, str]], theme_name: str, out_key_path: str) -> str:
    # Escape paths for AppleScript POSIX file
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")

    # Build AppleScript that creates a doc, inserts images slide by slide, then saves
    lines: list[str] = []
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
    if len(jpg_data) > 0:
        lines.append('    try')
        lines.append('        tell theDoc to delete slide 1')
        lines.append('    end try')

    for idx, (jpg_abs, side_length_um, original_nd2_path) in enumerate(jpg_data, start=1):
        posix_path = esc(jpg_abs)
        # Extract relative path for title from original ND2 file (without extension)
        rel_path = os.path.relpath(original_nd2_path, dir_path)
        slide_title = os.path.splitext(rel_path)[0]  # Remove extension
        escaped_title = esc(slide_title)
        
        # Create scale information text
        if side_length_um > 0:
            scale_text = f"{side_length_um:.0f}µm/side"
        else:
            scale_text = "Scale unknown"
        escaped_scale = esc(scale_text)
        
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
        lines.append('            -- Add scale information as bullet point')
        lines.append('            try')
        lines.append('                if (count of text items) > 1 then')
        lines.append(f'                    set object text of text item 2 to "{escaped_scale}"')
        lines.append('                else')
        lines.append(f'                    set scaleShape to make new text item with properties {{object text:"{escaped_scale}", position:{{50, 100}}}}')
        lines.append('                end if')
        lines.append('            on error')
        lines.append(f'                set scaleShape to make new text item with properties {{object text:"{escaped_scale}", position:{{50, 100}}}}')
        lines.append('            end try')
        lines.append('            -- Insert image into layout placeholder')
        lines.append('            try')
        lines.append(f'                set imageFile to alias (POSIX file "{posix_path}")')
        lines.append('                -- Method 1: Try to replace existing image placeholder')
        lines.append('                set imagePlaced to false')
        lines.append('                repeat with img in (get every image)')
        lines.append('                    try')
        lines.append('                        set file name of img to imageFile')
        lines.append('                        set imagePlaced to true')
        lines.append('                        exit repeat')
        lines.append('                    on error')
        lines.append('                        try')
        lines.append('                            set file of img to imageFile')
        lines.append('                            set imagePlaced to true')
        lines.append('                            exit repeat')
        lines.append('                        end try')
        lines.append('                    end try')
        lines.append('                end repeat')
        lines.append('                -- Method 2: Create new image if no placeholder found')
        lines.append('                if not imagePlaced then')
        lines.append('                    set newImage to make new image with properties {{file:imageFile, position:{{120, 120}}, width:512, height:512}}')
        lines.append('                end if')
        lines.append('            on error errMsg1')
        lines.append('                try')
        lines.append(f'                    set imageFile to alias (POSIX file "{posix_path}")')
        lines.append('                    set newShape to make new text item with properties {{object text:"Image: " & (name of imageFile), position:{{120, 200}}}}')
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


def find_nd2_files_recursively(root_dir: str) -> list[str]:
    """
    Recursively find all ND2 files in directory and subdirectories.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of ND2 file paths sorted by path
    """
    nd2_paths = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Skip hidden and AppleDouble files
            if file.startswith('.') or file.startswith('._'):
                continue
            
            # Check if file has .nd2 extension
            if file.lower().endswith('.nd2'):
                nd2_paths.append(os.path.join(root, file))
    
    return sorted(nd2_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Keynote from ND2 fluorescence images.")
    parser.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: cwd)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto)")
    parser.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    parser.add_argument("--subdirs-only", action="store_true", help="[DEPRECATED] Use --non-recursive instead")
    parser.add_argument("--verbose", action="store_true", help="Print generated AppleScript for debugging")
    parser.add_argument("--keep-jpgs", action="store_true", help="Keep generated JPG files after creating Keynote")
    parser.add_argument("--channels", type=str, default="", 
                       help="Specific channels to extract (comma-separated): red,green,blue or channel indices (0,1,2)")
    parser.add_argument("--single-channel", action="store_true", 
                       help="Extract only single channel as grayscale (use with --channels)")
    parser.add_argument("--all-channels", action="store_true", 
                       help="Force extraction of all available channels (overrides automatic mapping)")
    parser.add_argument("--debug-channels", action="store_true", 
                       help="Save individual channel images for debugging")
    args = parser.parse_args()

    dir_path = os.path.abspath(args.input)
    if not os.path.isdir(dir_path):
        print("Directory not found:", dir_path, file=sys.stderr)
        sys.exit(1)

    # Find ND2 files
    if args.non_recursive or args.subdirs_only:
        # Non-recursive: only current directory
        if args.subdirs_only:
            print("Warning: --subdirs-only is deprecated. The script now searches recursively by default.", file=sys.stderr)
            print("         Use --non-recursive to search only the current directory.", file=sys.stderr)
        nd2_paths = sorted(
            [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
             if f.lower().endswith('.nd2') and not f.startswith('.') and not f.startswith('._')]
        )
    else:
        # Recursive: walk all subdirectories
        nd2_paths = find_nd2_files_recursively(dir_path)
    
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    print(f"Found {len(nd2_paths)} ND2 file(s):")
    for nd2_file in nd2_paths:
        print(f"  - {os.path.basename(nd2_file)}")

    # Create temporary directory for JPG files
    temp_dir = os.path.join(dir_path, "temp_fluorescence_images")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process channel selection arguments
    selected_channels = None
    if args.channels:
        selected_channels = [ch.strip().lower() for ch in args.channels.split(',')]
        print(f"Channel selection: {selected_channels}")
        if args.single_channel:
            print("Single channel mode enabled")
    
    # Process ND2 files and create JPG files
    jpg_data: list[tuple[str, float, str]] = []
    print(f"\nProcessing ND2 files and creating fluorescence images...")
    
    for nd2_path in nd2_paths:
        # Set global flags for options
        extract_fluorescence_channels._all_channels_flag = args.all_channels
        extract_fluorescence_channels._debug_channels = args.debug_channels
        result = extract_fluorescence_channels(nd2_path, temp_dir, selected_channels, args.single_channel)
        if result[0]:  # jpg_path is not None
            jpg_data.append(result)
    
    if not jpg_data:
        print("No JPG files were created from ND2 files.")
        sys.exit(1)

    print(f"\nCreated {len(jpg_data)} fluorescence image(s):")
    for jpg_path, side_length, original_path in jpg_data:
        rel_path = os.path.relpath(original_path, dir_path)
        print(f"  - {rel_path} (side: {side_length:.1f}µm)")

    # Default output: current folder name.key if --output not provided
    default_base = os.path.basename(os.path.abspath(dir_path).rstrip(os.sep)) or "FluorescencePresentation"
    out_name = args.output.strip() or f"{default_base}.key"
    if not out_name.lower().endswith('.key'):
        out_name += '.key'
    out_key_path = os.path.join(dir_path, out_name)

    ascript = build_applescript(dir_path, jpg_data, args.theme, out_key_path)
    if args.verbose:
        print("--- AppleScript begin ---")
        print(ascript)
        print("--- AppleScript end ---")

    # Always run osascript and print outputs to aid diagnosis
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

    # Clean up temporary JPG files unless --keep-jpgs is specified
    if not args.keep_jpgs:
        print(f"\nCleaning up temporary files...")
        for jpg_path, _, _ in jpg_data:
            try:
                os.remove(jpg_path)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass
        print("✅ Cleanup complete")
    else:
        print(f"\nJPG files kept in: {temp_dir}")


if __name__ == "__main__":
    main()
