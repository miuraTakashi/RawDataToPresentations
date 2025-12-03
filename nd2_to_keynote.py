#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from ND2 files with automatic format detection.

This script automatically detects whether ND2 files contain:
- Single frames (images): Extract fluorescence channels as RGB images
- Time series (movies): Convert to MP4 movies with colored channels
- Z-stacks (volumes): Create volume visualization with XZ/YZ cross-sections

Features:
- Automatic detection of image vs movie vs volume content
- Fluorescence channel extraction (DAPI=Blue, Alexa488=Green, Alexa568=Red)
- MP4 conversion with metadata embedding
- Volume visualization with orthogonal views
- Keynote presentation generation with proper slide layouts
- Recursive directory scanning

Requirements:
- macOS with Keynote installed
- Python 3
- nd2reader or pims, opencv-python, numpy, PIL (Pillow)

Usage examples:
  python nd2_to_keynote.py --input "." --theme "White"
  python nd2_to_keynote.py --input "/path/to/nd2_files" --fps 10 --output "MyPresentation.key"
  python nd2_to_keynote.py --input "." --image-only   # Force image mode
  python nd2_to_keynote.py --input "." --movie-only   # Force movie mode
  python nd2_to_keynote.py --input "." --volume-only  # Force volume mode
"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image

# Add script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import common utilities
try:
    import nd2_utils
    from nd2_utils import (
        find_nd2_files_recursively,
        detect_nd2_content_type as utils_detect_content_type,
        classify_channel_names,
    )
    _HAS_ND2_UTILS = True
except ImportError:
    _HAS_ND2_UTILS = False

# Import existing modules
try:
    import nd2_to_mp4  # type: ignore
    import nd2images_to_keynote  # type: ignore
    import nd2movies_to_keynote  # type: ignore
    import nd2volumes_to_keynote  # type: ignore
except Exception as exc:
    sys.stderr.write("ERROR: Unable to import required modules. Ensure nd2_to_mp4.py, nd2images_to_keynote.py, nd2movies_to_keynote.py, and nd2volumes_to_keynote.py exist.\n")
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

# Note: nd2 module is not available in current environment
_HAS_ND2 = False
nd2 = None


def detect_nd2_content_type(nd2_path: str) -> Tuple[str, int, int, int, Dict]:
    """
    Detect whether ND2 file contains image, movie, or volume data.
    
    Args:
        nd2_path: Path to the ND2 file
        
    Returns:
        Tuple of (content_type, num_t, num_z, num_c, metadata)
        content_type: 'image', 'movie', or 'volume'
        num_t: Number of time frames
        num_z: Number of Z slices
        num_c: Number of channels
        metadata: ND2 metadata dictionary
    """
    try:
        if _HAS_ND2READER:
            # Try nd2reader
            with ND2Reader(nd2_path) as images:
                metadata = getattr(images, "metadata", {}) or {}
                sizes = getattr(images, "sizes", {}) or {}
                
                # Get dimensions
                num_t = sizes.get('t', 1)
                num_z = sizes.get('z', 1)
                num_c = sizes.get('c', 1)
                
                # If sizes doesn't have t or z, check frame count
                total_frames = len(images)
                if num_t == 1 and num_z == 1 and total_frames > 1:
                    # Try to determine if it's time series or z-stack from metadata
                    z_coords = metadata.get('z_coordinates', None)
                    if z_coords is not None and len(z_coords) > 1:
                        num_z = len(z_coords)
                    else:
                        num_t = total_frames
                
                # Determine content type
                if num_z > 1:
                    content_type = 'volume'
                elif num_t > 1:
                    content_type = 'movie'
                else:
                    content_type = 'image'
                
                return content_type, num_t, num_z, num_c, metadata
                
        elif _HAS_PIMS:
            # Fallback to pims
            with pims.open(nd2_path) as images:
                try:
                    metadata = getattr(images, "metadata", {}) or {}
                except Exception:
                    metadata = {}
                
                total_frames = len(images)
                num_t = 1
                num_z = 1
                num_c = 1
                
                # Try to get dimensions from metadata
                if hasattr(images, 'sizes'):
                    sizes = images.sizes
                    num_t = sizes.get('t', 1)
                    num_z = sizes.get('z', 1)
                    num_c = sizes.get('c', 1)
                
                # If still single frame but many total frames, check z_coordinates
                if num_t == 1 and num_z == 1 and total_frames > 1:
                    z_coords = metadata.get('z_coordinates', None)
                    if z_coords is not None and len(z_coords) > 1:
                        num_z = len(z_coords)
                    else:
                        num_t = total_frames
                
                # Determine content type
                if num_z > 1:
                    content_type = 'volume'
                elif num_t > 1:
                    content_type = 'movie'
                else:
                    content_type = 'image'
                
                return content_type, num_t, num_z, num_c, metadata
                
        else:
            raise RuntimeError("No ND2 backend available")
            
    except Exception as e:
        print(f"Warning: Could not analyze {nd2_path}: {e}")
        # Default to movie if we can't determine
        return 'movie', 0, 0, 0, {}


def find_nd2_files_recursively(root_dir: str) -> List[str]:
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


def categorize_nd2_files(nd2_paths: List[str], force_mode: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize ND2 files into images, movies, and volumes.
    
    Args:
        nd2_paths: List of ND2 file paths
        force_mode: Force all files to be treated as 'image', 'movie', or 'volume', or None for auto-detection
        
    Returns:
        Tuple of (image_files, movie_files, volume_files)
    """
    image_files = []
    movie_files = []
    volume_files = []
    
    for nd2_path in nd2_paths:
        if force_mode == 'image':
            image_files.append(nd2_path)
        elif force_mode == 'movie':
            movie_files.append(nd2_path)
        elif force_mode == 'volume':
            volume_files.append(nd2_path)
        else:
            # Auto-detection
            content_type, num_t, num_z, num_c, metadata = detect_nd2_content_type(nd2_path)
            
            if content_type == 'volume':
                print(f"  {os.path.basename(nd2_path)}: {content_type} (Z={num_z}, T={num_t}, C={num_c})")
            elif content_type == 'movie':
                print(f"  {os.path.basename(nd2_path)}: {content_type} (T={num_t}, C={num_c})")
            else:
                print(f"  {os.path.basename(nd2_path)}: {content_type} (C={num_c})")
            
            if content_type == 'image':
                image_files.append(nd2_path)
            elif content_type == 'movie':
                movie_files.append(nd2_path)
            else:  # volume
                volume_files.append(nd2_path)
    
    return image_files, movie_files, volume_files


def process_images_to_keynote(image_files: List[str], args) -> Optional[str]:
    """
    Process image files and create Keynote presentation.
    
    Args:
        image_files: List of ND2 image file paths
        args: Command line arguments
        
    Returns:
        Path to created Keynote file or None if failed
    """
    if not image_files:
        return None
        
    print(f"\n📸 Processing {len(image_files)} image file(s)...")
    
    # Create temporary directory for images
    temp_dir = os.path.join(args.input, "temp_fluorescence_images")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each ND2 image file
    jpg_data = []
    for nd2_path in image_files:
        try:
            # Extract fluorescence channels
            result = nd2images_to_keynote.extract_fluorescence_channels(
                nd2_path, temp_dir, 
                selected_channels=args.channels.split(',') if args.channels else None,
                single_channel_mode=args.single_channel
            )
            if result[0]:  # jpg_path is not None
                jpg_data.append(result)
        except Exception as e:
            print(f"Error processing {nd2_path}: {e}")
    
    if not jpg_data:
        print("No images were created from ND2 files.")
        return None
    
    # Generate output filename
    if args.output:
        out_name = args.output
    else:
        base_name = os.path.basename(os.path.abspath(args.input).rstrip(os.sep)) or "Images"
        out_name = f"{base_name}_Images.key"
    
    if not out_name.lower().endswith('.key'):
        out_name += '.key'
    out_key_path = os.path.join(args.input, out_name)
    
    # Build and run AppleScript
    ascript = nd2images_to_keynote.build_applescript(args.input, jpg_data, args.theme, out_key_path)
    
    print(f"\n🎯 Creating Keynote presentation: {out_name}")
    completed = subprocess.run(
        ["/usr/bin/osascript", "-s", "e", "-s", "o", "-"],
        input=ascript.encode("utf-8"),
        capture_output=True,
    )
    
    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
        print("AppleScript failed. Details:\n" + err, file=sys.stderr)
        return None
    
    # Clean up temporary files unless --keep-jpgs is specified
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


def process_movies_to_keynote(movie_files: List[str], args) -> Optional[str]:
    """
    Process movie files and create Keynote presentation.
    
    Args:
        movie_files: List of ND2 movie file paths
        args: Command line arguments
        
    Returns:
        Path to created Keynote file or None if failed
    """
    if not movie_files:
        return None
        
    print(f"\n🎬 Processing {len(movie_files)} movie file(s)...")
    
    # Convert ND2 files to MP4
    mp4_paths = []
    metadata_list = []
    use_clahe = not args.no_clahe
    use_ffmpeg = not args.no_ffmpeg
    
    for nd2_path in movie_files:
        base = os.path.splitext(os.path.basename(nd2_path))[0]
        subdir = os.path.dirname(nd2_path)
        out_path = os.path.join(subdir, f"{base}.mp4")
        
        # Extract metadata
        metadata = nd2movies_to_keynote.extract_nd2_metadata(nd2_path, float(args.fps))
        metadata_list.append(metadata)
        
        if os.path.exists(out_path) and args.no_overwrite:
            print(f"Skip existing: {out_path}")
        else:
            print(f"Converting: {nd2_path}\n  -> {out_path}")
            try:
                # Set environment variables for normalization
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
            
        # Check for brightfield MP4
        brightfield_path = os.path.join(subdir, f"{base}_brightfield.mp4")
        if os.path.exists(brightfield_path):
            mp4_paths.append(brightfield_path)
            metadata_list.append(metadata)  # Use same metadata
    
    if not mp4_paths:
        print("No MP4 files were created.")
        return None
    
    # Generate output filename
    if args.output:
        out_name = args.output
    else:
        base_name = os.path.basename(os.path.abspath(args.input).rstrip(os.sep)) or "Movies"
        out_name = f"{base_name}_Movies.key"
    
    if not out_name.lower().endswith('.key'):
        out_name += '.key'
    out_key_path = os.path.join(args.input, out_name)
    
    # Build and run AppleScript
    ascript = nd2movies_to_keynote.build_applescript(args.input, mp4_paths, metadata_list, args.theme, out_key_path)
    
    print(f"\n🎯 Creating Keynote presentation: {out_name}")
    completed = subprocess.run(
        ["/usr/bin/osascript", "-s", "e", "-s", "o", "-"],
        input=ascript.encode("utf-8"),
        capture_output=True,
    )
    
    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
        print("AppleScript failed. Details:\n" + err, file=sys.stderr)
        return None
    
    print(f"✅ Movie Keynote created: {out_key_path}")
    return out_key_path


def process_volumes_to_keynote(volume_files: List[str], args) -> List[str]:
    """
    Process volume files and create Keynote presentations.
    
    Args:
        volume_files: List of ND2 volume file paths
        args: Command line arguments
        
    Returns:
        List of paths to created Keynote files
    """
    if not volume_files:
        return []
        
    print(f"\n📦 Processing {len(volume_files)} volume file(s)...")
    
    created_keynotes = []
    use_clahe = not args.no_clahe
    
    for nd2_path in volume_files:
        print(f"\nProcessing volume: {os.path.basename(nd2_path)}")
        
        base_name = os.path.splitext(os.path.basename(nd2_path))[0]
        base_dir = os.path.dirname(nd2_path)
        
        # Create temporary directory
        temp_dir = os.path.join(base_dir, f"temp_{base_name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        movie_path = os.path.join(temp_dir, f"{base_name}_zstack.mp4")
        xz_image_path = os.path.join(temp_dir, f"{base_name}_xz.jpg")
        yz_image_path = os.path.join(temp_dir, f"{base_name}_yz.jpg")
        
        try:
            # Get volume dimensions and metadata
            if nd2volumes_to_keynote._HAS_ND2READER:
                from nd2reader import ND2Reader
                with ND2Reader(nd2_path) as images:
                    sizes = getattr(images, 'sizes', {}) or {}
                    num_z = sizes.get('z', len(images))
                    num_y = sizes.get('y', 256)
                    num_x = sizes.get('x', 256)
                    
                    # Get pixel size from metadata
                    meta = getattr(images, 'metadata', {}) or {}
                    pixel_size_xy = 1.0
                    pixel_size_z = 1.0
                    
                    if 'pixel_microns' in meta:
                        pixel_size_xy = float(meta['pixel_microns'])
                    
                    z_coords = meta.get('z_coordinates', None)
                    if z_coords is not None and len(z_coords) > 1:
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
                    pixel_size_xy = 1.0
                    pixel_size_z = 1.0
            
            # Calculate physical dimensions
            x_um = num_x * pixel_size_xy
            y_um = num_y * pixel_size_xy
            z_um = num_z * pixel_size_z
            
            print(f"  Volume dimensions: X={num_x}, Y={num_y}, Z={num_z}")
            print(f"  Physical size: X={x_um:.1f} µm, Y={y_um:.1f} µm, Z={z_um:.1f} µm")
            
            # Create Z-stack movie
            if not nd2volumes_to_keynote.create_z_stack_movie(nd2_path, movie_path, args.fps, use_clahe):
                print(f"  ⚠️  Failed to create Z-stack movie")
                continue
            
            # Extract XZ cross-section (middle Y)
            y_mid = num_y // 2
            if not nd2volumes_to_keynote.extract_xz_cross_section(nd2_path, y_mid, xz_image_path, use_clahe):
                print(f"  ⚠️  Failed to extract XZ cross-section")
                continue
            
            # Extract YZ cross-section (middle X)
            x_mid = num_x // 2
            if not nd2volumes_to_keynote.extract_yz_cross_section(nd2_path, x_mid, yz_image_path, use_clahe):
                print(f"  ⚠️  Failed to extract YZ cross-section")
                continue
            
            # Generate output filename
            out_name = f"{base_name}_Volume.key"
            out_key_path = os.path.join(base_dir, out_name)
            
            # Build and run AppleScript
            ascript = nd2volumes_to_keynote.build_applescript(
                base_dir, movie_path, xz_image_path, yz_image_path,
                base_name, args.theme, out_key_path,
                num_x, num_y, num_z,
                x_um, y_um, z_um
            )
            
            print(f"\n🎯 Creating Keynote presentation: {out_name}")
            completed = subprocess.run(
                ["/usr/bin/osascript", "-s", "e", "-s", "o", "-"],
                input=ascript.encode("utf-8"),
                capture_output=True,
            )
            
            if completed.returncode != 0:
                err = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
                print("AppleScript failed. Details:\n" + err, file=sys.stderr)
                continue
            
            print(f"✅ Volume Keynote created: {out_key_path}")
            created_keynotes.append(out_key_path)
            
            # Clean up temporary files unless --keep-temp is specified
            if not args.keep_temp:
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
        
        except Exception as e:
            print(f"  Error processing {nd2_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return created_keynotes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Keynote presentations from ND2 files (auto-detects images, movies, and volumes).")
    
    # Input/Output options
    parser.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: current directory)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto-generated)")
    parser.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    parser.add_argument("--verbose", action="store_true", help="Print generated AppleScript for debugging")
    
    # Content type forcing
    parser.add_argument("--image-only", action="store_true", help="Force all ND2 files to be treated as images")
    parser.add_argument("--movie-only", action="store_true", help="Force all ND2 files to be treated as movies")
    parser.add_argument("--volume-only", action="store_true", help="Force all ND2 files to be treated as volumes")
    
    # Image-specific options
    parser.add_argument("--keep-jpgs", action="store_true", help="Keep generated JPG files after creating Keynote")
    parser.add_argument("--channels", type=str, default="", 
                       help="Specific channels to extract (comma-separated): red,green,blue or channel indices (0,1,2)")
    parser.add_argument("--single-channel", action="store_true", 
                       help="Extract only single channel as grayscale (use with --channels)")
    
    # Movie-specific options (mirroring nd2_to_mp4)
    parser.add_argument("--fps", type=float, default=10.0, help="Fallback FPS if not found in ND2 metadata (default: 10)")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE clip limit (default: 3.0)")
    parser.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (N -> NxN) (default: 8)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    parser.add_argument("--codec", type=str, default="avc1", help="Preferred fourcc codec for MP4 (default: avc1)")
    parser.add_argument("--norm-mode", choices=["percentile", "minmax"], default="percentile", help="Global normalization mode")
    parser.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for histogram clipping")
    parser.add_argument("--upper-pct", type=float, default=99.5, help="Upper percentile for histogram clipping")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing MP4 files instead of overwriting them")
    parser.add_argument("--no-ffmpeg", action="store_true", help="Disable FFmpeg and use OpenCV VideoWriter instead")
    
    # Volume-specific options
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary MP4 and JPG files for volumes")
    
    args = parser.parse_args()

    # Validate arguments
    force_count = sum([args.image_only, args.movie_only, args.volume_only])
    if force_count > 1:
        print("Error: Cannot specify more than one of --image-only, --movie-only, --volume-only", file=sys.stderr)
        sys.exit(1)

    dir_path = os.path.abspath(args.input)
    if not os.path.isdir(dir_path):
        print("Directory not found:", dir_path, file=sys.stderr)
        sys.exit(1)

    # Check backends
    if not _HAS_ND2READER and not _HAS_PIMS:
        print("ERROR: No ND2 backend available. Please install: pip install nd2reader pims", file=sys.stderr)
        sys.exit(2)

    # Find ND2 files
    if args.non_recursive:
        nd2_paths = sorted(
            [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
             if f.lower().endswith('.nd2') and not f.startswith('.') and not f.startswith('._')]
        )
    else:
        nd2_paths = find_nd2_files_recursively(dir_path)
    
    if not nd2_paths:
        print("No ND2 files found.")
        sys.exit(0)

    print(f"Found {len(nd2_paths)} ND2 file(s):")
    for nd2_file in nd2_paths:
        print(f"  - {os.path.relpath(nd2_file, dir_path)}")

    # Determine force mode
    force_mode = None
    if args.image_only:
        force_mode = 'image'
        print("\n🔧 Force mode: All files will be treated as images")
    elif args.movie_only:
        force_mode = 'movie'
        print("\n🔧 Force mode: All files will be treated as movies")
    elif args.volume_only:
        force_mode = 'volume'
        print("\n🔧 Force mode: All files will be treated as volumes")
    else:
        print("\n🔍 Auto-detecting content type...")

    # Categorize files
    image_files, movie_files, volume_files = categorize_nd2_files(nd2_paths, force_mode)
    
    print(f"\n📊 Summary:")
    print(f"  Images:  {len(image_files)} files")
    print(f"  Movies:  {len(movie_files)} files")
    print(f"  Volumes: {len(volume_files)} files")

    # Process files and create presentations
    created_presentations = []
    
    if image_files:
        image_keynote = process_images_to_keynote(image_files, args)
        if image_keynote:
            created_presentations.append(image_keynote)
    
    if movie_files:
        movie_keynote = process_movies_to_keynote(movie_files, args)
        if movie_keynote:
            created_presentations.append(movie_keynote)
    
    if volume_files:
        volume_keynotes = process_volumes_to_keynote(volume_files, args)
        created_presentations.extend(volume_keynotes)
    
    # Summary
    if created_presentations:
        print(f"\n🎉 Successfully created {len(created_presentations)} Keynote presentation(s):")
        for presentation in created_presentations:
            print(f"  ✅ {os.path.basename(presentation)}")
    else:
        print("\n❌ No Keynote presentations were created.")
        sys.exit(1)


if __name__ == "__main__":
    main()
