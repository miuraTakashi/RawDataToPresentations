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
        
        # Print channel information if available
        try:
            if hasattr(nd2_to_mp4, 'print_channel_info'):
                # Try to get channel info from the converter
                import numpy as np
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
                            nd2_to_mp4.print_channel_info(nd2_path, meta, num_channels, ch_map)
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
                            nd2_to_mp4.print_channel_info(nd2_path, meta, num_channels, ch_map)
                except Exception as e:
                    print(f"Could not extract channel info: {e}")
        except Exception:
            pass
        
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


