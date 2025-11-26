#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from ND2 files with automatic format detection.

This script automatically detects whether ND2 files contain single frames (images) or 
multiple frames (movies) and creates appropriate Keynote presentations:
- Single frame ND2s: Extract fluorescence channels as RGB images
- Multi-frame ND2s: Convert to MP4 movies with red-on-black coloring

Features:
- Automatic detection of image vs movie content
- Fluorescence channel extraction (DAPI=Blue, Alexa488=Green, Alexa568=Red)
- MP4 conversion with metadata embedding
- Keynote presentation generation with proper slide layouts
- Recursive directory scanning

Requirements:
- macOS with Keynote installed
- Python 3
- nd2reader or pims, opencv-python, numpy, PIL (Pillow)

Usage examples:
  python nd2_to_keynote.py --input "." --theme "White"
  python nd2_to_keynote.py --input "/path/to/nd2_files" --fps 10 --output "MyPresentation.key"
  python nd2_to_keynote.py --input "." --image-only  # Force image mode
  python nd2_to_keynote.py --input "." --movie-only  # Force movie mode
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

# Import existing modules
try:
    import nd2_to_mp4  # type: ignore
    import nd2images_to_keynote  # type: ignore
    import nd2movies_to_keynote  # type: ignore
except Exception as exc:
    sys.stderr.write("ERROR: Unable to import required modules. Ensure nd2_to_mp4.py, nd2images_to_keynote.py, and nd2movies_to_keynote.py exist.\n")
    raise

# Optional imports for ND2 reading
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


def detect_nd2_content_type(nd2_path: str) -> Tuple[str, int, Dict]:
    """
    Detect whether ND2 file contains single frame (image) or multiple frames (movie).
    
    Args:
        nd2_path: Path to the ND2 file
        
    Returns:
        Tuple of (content_type, frame_count, metadata)
        content_type: 'image' or 'movie'
        frame_count: Number of frames
        metadata: ND2 metadata dictionary
    """
    try:
        if _HAS_ND2READER:
            # Try nd2reader
            with ND2Reader(nd2_path) as images:
                frame_count = len(images)
                metadata = getattr(images, "metadata", {}) or {}
                content_type = 'image' if frame_count <= 1 else 'movie'
                return content_type, frame_count, metadata
                
        elif _HAS_PIMS:
            # Fallback to pims
            with pims.open(nd2_path) as images:
                frame_count = len(images)
                try:
                    metadata = getattr(images, "metadata", {}) or {}
                except Exception:
                    metadata = {}
                content_type = 'image' if frame_count <= 1 else 'movie'
                return content_type, frame_count, metadata
                
        else:
            raise RuntimeError("No ND2 backend available")
            
    except Exception as e:
        print(f"Warning: Could not analyze {nd2_path}: {e}")
        # Default to movie if we can't determine
        return 'movie', 0, {}


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


def categorize_nd2_files(nd2_paths: List[str], force_mode: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Categorize ND2 files into images and movies.
    
    Args:
        nd2_paths: List of ND2 file paths
        force_mode: Force all files to be treated as 'image' or 'movie', or None for auto-detection
        
    Returns:
        Tuple of (image_files, movie_files)
    """
    image_files = []
    movie_files = []
    
    for nd2_path in nd2_paths:
        if force_mode == 'image':
            image_files.append(nd2_path)
        elif force_mode == 'movie':
            movie_files.append(nd2_path)
        else:
            # Auto-detection
            content_type, frame_count, metadata = detect_nd2_content_type(nd2_path)
            print(f"  {os.path.basename(nd2_path)}: {content_type} ({frame_count} frames)")
            
            if content_type == 'image':
                image_files.append(nd2_path)
            else:
                movie_files.append(nd2_path)
    
    return image_files, movie_files


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Keynote presentations from ND2 files (auto-detects images vs movies).")
    
    # Input/Output options
    parser.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: current directory)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto-generated)")
    parser.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    parser.add_argument("--verbose", action="store_true", help="Print generated AppleScript for debugging")
    
    # Content type forcing
    parser.add_argument("--image-only", action="store_true", help="Force all ND2 files to be treated as images")
    parser.add_argument("--movie-only", action="store_true", help="Force all ND2 files to be treated as movies")
    
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
    
    args = parser.parse_args()

    # Validate arguments
    if args.image_only and args.movie_only:
        print("Error: Cannot specify both --image-only and --movie-only", file=sys.stderr)
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
    else:
        print("\n🔍 Auto-detecting content type...")

    # Categorize files
    image_files, movie_files = categorize_nd2_files(nd2_paths, force_mode)
    
    print(f"\n📊 Summary:")
    print(f"  Images: {len(image_files)} files")
    print(f"  Movies: {len(movie_files)} files")

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
