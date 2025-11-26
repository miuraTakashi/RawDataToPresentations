#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from image files in a directory and its subdirectories.

Each image gets its own slide with the "タイトル、画像、箇条書きKeyence" layout.
The slide title is set to the relative path of the image.
Automatically handles 16-bit images with byte order issues and applies optional contrast enhancement.
The resulting .key file is saved into the same directory.

Features:
- Automatic 16-bit image byte order correction
- Percentile-based normalization for better display
- Optional contrast enhancement

Requirements:
- macOS with Keynote installed
- Python 3
- PIL (Pillow) and numpy for image processing

Usage:
  python create_keynote_with_keyence_layout.py \
    [--input "/path/to/dir"] [--theme "White"] [--output "MyImages.key"] \
    [--extensions ".jpg,.jpeg,.png,.tif,.tiff"] [--contrast 1.0]
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np


# Keyence microscope magnification to side length mapping (µm)
KEYENCE_MAGNIFICATION = {
    'x4': 3490,
    'x10': 1454,
    'x20': 711,
    'x40': 360,
    '4x': 3490,
    '10x': 1454,
    '20x': 711,
    '40x': 360,
}


def detect_magnification_from_path(file_path: str) -> tuple[float, str]:
    """
    Detect magnification from file path (filename or folder name).
    
    Args:
        file_path: Full path to the image file
        
    Returns:
        Tuple of (side_length_um, magnification_text) or (0, "") if not found
    """
    import re
    
    # Get filename and all folder names in the path
    path_parts = file_path.split(os.sep)
    filename = os.path.basename(file_path)
    all_names = path_parts + [filename]
    
    # Search for magnification pattern (x4, x10, x20, x40, 4x, 10x, 20x, 40x)
    # Priority: check for longer patterns first (x40, x20, x10 before x4)
    sorted_magnifications = sorted(KEYENCE_MAGNIFICATION.items(), key=lambda x: len(x[0]), reverse=True)
    
    for name in all_names:
        # Case-insensitive search
        name_lower = name.lower()
        for mag_key, side_length in sorted_magnifications:
            # Look for magnification in the name
            # Pattern: x4, x10, x20, x40 or 4x, 10x, 20x, 40x
            # Allow it to be part of a word (e.g., "x20x4" should match "x20")
            mag_lower = mag_key.lower()
            if mag_lower in name_lower:
                # Check if it's a valid match (not part of a larger number like x400)
                # Use word boundary or check that it's followed by non-digit or end of string
                pattern = re.escape(mag_lower) + r'(?![0-9])'
                if re.search(pattern, name_lower):
                    return side_length, mag_key.upper()
    
    return 0, ""


def get_image_info(image_path: str) -> tuple[int, int, str]:
    """
    Get basic information about an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height, mode)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            return width, height, mode
    except Exception as e:
        print(f"Warning: Could not read {image_path}: {e}")
        return 0, 0, "unknown"


def build_applescript(dir_path: str, image_data: list[tuple[str, str]], theme_name: str, out_key_path: str, show_image_info: bool = False) -> str:
    """
    Build AppleScript to create Keynote presentation.
    
    Args:
        dir_path: Base directory for relative path calculation
        image_data: List of tuples (original_path, processed_path) where:
                   - original_path: Original image path (used for title)
                   - processed_path: Processed image path (used for insertion)
        theme_name: Keynote theme name
        out_key_path: Output Keynote file path
        show_image_info: Whether to show image info on slides
    """
    # Escape paths for AppleScript POSIX file
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")

    # Build AppleScript that creates a doc, inserts images slide by slide, then saves
    lines: list[str] = []
    lines.append('tell application "Keynote"')
    lines.append('    activate')
    # Use ExperimentalDataR1 theme
    lines.append('    try')
    lines.append('        set theDoc to make new document with properties {document theme:theme "ExperimentalDataR1"}')
    lines.append('    on error')
    lines.append('        try')
    lines.append(f'            set theDoc to make new document with properties {{document theme:theme "{theme_name}"}}')
    lines.append('        on error')
    lines.append('            set theDoc to make new document')
    lines.append('        end try')
    lines.append('    end try')
    # Wait a moment to ensure document is fully created
    lines.append('    delay 1.0')
    # Remove the default slide if we will add our own slides
    if len(image_data) > 0:
        lines.append('    try')
        lines.append('        tell theDoc to delete slide 1')
        lines.append('    end try')

    for idx, (original_path, processed_path) in enumerate(image_data, start=1):
        # Use processed path for image insertion
        posix_path = esc(processed_path)
        # Use original path for title (to preserve folder hierarchy)
        rel_path = os.path.relpath(original_path, dir_path)
        slide_title = os.path.splitext(rel_path)[0]  # Remove extension
        escaped_title = esc(slide_title)
        
        # Detect magnification from file path
        side_length_um, magnification_text = detect_magnification_from_path(original_path)
        scale_text = ""
        escaped_scale = ""
        if side_length_um > 0:
            scale_text = f"{side_length_um:.0f} µm / horizontal side"
            escaped_scale = esc(scale_text)
        
        # Get image info if requested (use processed path for info)
        info_text = ""
        if show_image_info:
            width, height, mode = get_image_info(processed_path)
            if width > 0 and height > 0:
                info_text = f"{width}×{height}px, {mode}"
                escaped_info = esc(info_text)
        
        lines.append('    tell theDoc')
        lines.append('        set newSlide to make new slide')
        lines.append('        try')
        lines.append('            set base slide of newSlide to master slide "タイトル、箇条書き、画像Keyence" of theDoc')
        lines.append('        on error')
        lines.append('            try')
        lines.append('                set base slide of newSlide to master slide "タイトル、画像、箇条書きKeyence" of theDoc')
        lines.append('            on error')
        lines.append('                try')
        lines.append('                    set base slide of newSlide to master slide "タイトル、画像、箇条書き" of theDoc')
        lines.append('                on error')
        lines.append('                    try')
        lines.append('                        set base slide of newSlide to master slide "Blank" of theDoc')
        lines.append('                    on error')
        lines.append('                        try')
        lines.append('                            set base slide of newSlide to master slide "空白" of theDoc')
        lines.append('                        end try')
        lines.append('                    end try')
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
        # Add scale information if magnification detected
        text_item_index = 1  # Track which text item index to use
        if scale_text:
            lines.append('            -- Add scale information (magnification)')
            lines.append('            try')
            lines.append(f'                if (count of text items) > {text_item_index} then')
            lines.append(f'                    set object text of text item {text_item_index + 1} to "{escaped_scale}"')
            lines.append('                else')
            lines.append(f'                    set scaleShape to make new text item with properties {{object text:"{escaped_scale}", position:{{50, 100}}}}')
            lines.append('                end if')
            lines.append('            on error')
            lines.append(f'                set scaleShape to make new text item with properties {{object text:"{escaped_scale}", position:{{50, 100}}}}')
            lines.append('            end try')
            text_item_index += 1
        # Add image info if available
        if show_image_info and info_text:
            lines.append('            -- Add image info as bullet point')
            lines.append('            try')
            lines.append(f'                if (count of text items) > {text_item_index} then')
            lines.append(f'                    set object text of text item {text_item_index + 1} to "{escaped_info}"')
            lines.append('                else')
            lines.append(f'                    set infoShape to make new text item with properties {{object text:"{escaped_info}", position:{{50, {100 + text_item_index * 30}}}}}')
            lines.append('                end if')
            lines.append('            on error')
            lines.append(f'                set infoShape to make new text item with properties {{object text:"{escaped_info}", position:{{50, {100 + text_item_index * 30}}}}}')
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


def fix_endianness(arr: np.ndarray) -> np.ndarray:
    """Fix byte order for 16-bit images that may have incorrect endianness."""
    if arr.dtype == np.uint16:
        # Check if the array appears to have wrong endianness
        # If most values are very small (< 256), it might be byte-swapped
        if np.percentile(arr, 95) < 256:
            print(f"  Detected possible byte order issue, swapping bytes...")
            return arr.byteswap()
    return arr


def enhance_image_contrast(image_path: str, contrast_factor: float, output_dir: str) -> str:
    """
    Enhance the contrast of an image and save it to the output directory.
    Handles 16-bit images with potential byte order issues and TIFF metadata problems.
    
    Args:
        image_path: Path to the original image
        contrast_factor: Contrast enhancement factor (1.0 = no change, >1.0 = increase contrast)
        output_dir: Directory to save the enhanced image
        
    Returns:
        Path to the enhanced image file
    """
    try:
        # Open the original image with specific handling for problematic TIFFs
        with Image.open(image_path) as img:
            print(f"  Processing image: {os.path.basename(image_path)} (mode: {img.mode})")
            
            # Handle various image modes that might cause display issues
            if img.mode in ('I;16', 'I;16B', 'I;16L', 'I;32', 'F') or '16' in str(img.mode) or '32' in str(img.mode):
                print(f"  Converting high bit-depth image to 8-bit...")
                
                # Convert to numpy array for processing
                img_array = np.array(img)
                
                # Fix byte order if needed
                img_array = fix_endianness(img_array)
                
                # Handle different data types
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    # Float images: normalize to 0-1 range first
                    if img_array.max() > 1.0:
                        img_array = img_array / img_array.max()
                    img_array = (img_array * 65535).astype(np.uint16)
                
                # Normalize to 8-bit using robust percentile method
                if img_array.max() > 0:
                    # Use more conservative percentiles for better results
                    p1, p99 = np.percentile(img_array, (1, 99))
                    if p99 > p1:
                        img_array = np.clip((img_array - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                    else:
                        img_array = np.zeros_like(img_array, dtype=np.uint8)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)
                
                # Convert back to PIL Image as grayscale first
                img = Image.fromarray(img_array, mode='L')
                
            elif img.mode in ('L', 'P', 'LA'):
                # Grayscale or palette images
                if img.mode == 'P':
                    img = img.convert('L')
                print(f"  Converting grayscale image...")
                
            else:
                print(f"  Converting color image...")
            
            # Convert to RGB for consistent handling
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast if needed
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                ext = '.jpg'  # Default to JPG if extension is not recognized
            
            output_filename = f"{base_name}_processed{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save with specific parameters to ensure compatibility
            if ext in ['.tif', '.tiff']:
                # Save as TIFF with specific parameters
                img.save(output_path, format='TIFF', compression='lzw', optimize=True)
            else:
                # Save as JPEG with high quality
                img.save(output_path, format='JPEG', quality=95, optimize=True)
            
            print(f"  Saved: {os.path.basename(output_path)}")
            return output_path
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path  # Return original path if processing fails


def find_images_recursively(dir_path: str, extensions: list[str], recursive: bool = True) -> list[str]:
    """
    Find all image files in directory and optionally subdirectories.
    
    Args:
        dir_path: Root directory to search
        extensions: List of file extensions to search for
        recursive: If True, search subdirectories; if False, only current directory
        
    Returns:
        List of image file paths sorted by path
    """
    image_paths = []
    
    if recursive:
        # Recursive: walk all subdirectories
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # Skip hidden and AppleDouble files
                if file.startswith('.') or file.startswith('._'):
                    continue
                
                # Check if file has one of the specified extensions
                file_lower = file.lower()
                if any(file_lower.endswith(ext.lower()) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        # Non-recursive: only current directory
        for file in os.listdir(dir_path):
            # Skip hidden and AppleDouble files
            if file.startswith('.') or file.startswith('._'):
                continue
            
            # Check if file has one of the specified extensions
            file_lower = file.lower()
            if any(file_lower.endswith(ext.lower()) for ext in extensions):
                image_paths.append(os.path.join(dir_path, file))
    
    return sorted(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Keynote from image files in a directory and subdirectories.")
    parser.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: cwd)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto)")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.tif,.tiff", 
                       help="Comma-separated list of image extensions (default: .jpg,.jpeg,.png,.tif,.tiff)")
    parser.add_argument("--contrast", type=float, default=1.0, 
                       help="Contrast enhancement factor (1.0 = no change, >1.0 = increase contrast, default: 1.0)")
    parser.add_argument("--force-process", action="store_true", 
                       help="Force processing of all images (useful for problematic TIFF files)")
    parser.add_argument("--non-recursive", action="store_true", 
                       help="Search only current directory (not subdirectories)")
    parser.add_argument("--show-image-info", action="store_true",
                       help="Display image dimensions and mode on each slide")
    parser.add_argument("--verbose", action="store_true", help="Print generated AppleScript for debugging")
    args = parser.parse_args()

    dir_path = os.path.abspath(args.input)
    if not os.path.isdir(dir_path):
        print("Directory not found:", dir_path, file=sys.stderr)
        sys.exit(1)

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(',') if ext.strip()]
    if not extensions:
        print("No valid extensions provided.", file=sys.stderr)
        sys.exit(1)

    # Find all image files (recursively or not based on --non-recursive flag)
    image_paths = find_images_recursively(dir_path, extensions, recursive=not args.non_recursive)
    
    if not image_paths:
        print(f"No image files found with extensions: {', '.join(extensions)}")
        sys.exit(0)

    print(f"Found {len(image_paths)} image file(s):")
    for img in image_paths:
        rel_path = os.path.relpath(img, dir_path)
        print(f"  - {rel_path}")

    # Process images for 16-bit handling and optional contrast enhancement
    # Create list of tuples: (original_path, processed_path)
    image_data: list[tuple[str, str]] = []
    temp_dir = None
    try:
        if args.force_process or any(img.lower().endswith(('.tif', '.tiff')) for img in image_paths):
            print(f"\nProcessing images for compatibility and contrast enhancement...")
            # Create temporary directory for processed images
            temp_dir = os.path.join(dir_path, "temp_processed_images")
            os.makedirs(temp_dir, exist_ok=True)
            
            processed_count = 0
            for img_path in image_paths:
                # Process images to handle various compatibility issues
                enhanced_path = enhance_image_contrast(img_path, args.contrast, temp_dir)
                # Store tuple: (original_path, processed_path)
                image_data.append((img_path, enhanced_path))
                if enhanced_path != img_path:  # Only print if processing was applied
                    rel_enhanced = os.path.relpath(enhanced_path, dir_path)
                    rel_original = os.path.relpath(img_path, dir_path)
                    print(f"  Processed: {rel_original} -> {rel_enhanced}")
                    processed_count += 1
            
            print(f"  Successfully processed {processed_count} image(s)")
        else:
            print(f"\nSkipping image processing (use --force-process to process all images)")
            # No processing: use original paths for both
            image_data = [(img_path, img_path) for img_path in image_paths]

        # Default output: target directory name.key if --output not provided
        if args.output.strip():
            out_name = args.output.strip()
        else:
            # Use the target directory name as the base filename
            default_base = os.path.basename(os.path.abspath(dir_path).rstrip(os.sep)) or "Presentation"
            out_name = f"{default_base}.key"
        if not out_name.lower().endswith('.key'):
            out_name += '.key'
        out_key_path = os.path.join(dir_path, out_name)

        ascript = build_applescript(dir_path, image_data, args.theme, out_key_path, show_image_info=args.show_image_info)
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
    finally:
        # Clean up temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"✅ Cleaned up temporary directory: {os.path.basename(temp_dir)}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
