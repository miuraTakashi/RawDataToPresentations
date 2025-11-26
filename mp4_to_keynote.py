#!/usr/bin/env python3
"""
Create a Keynote (.key) presentation from MP4 files in a directory.

Each MP4 gets its own slide with the video placed on the slide. The resulting
.key file is saved into the same directory.

By default, recursively searches all subdirectories for MP4 files.
Use --non-recursive to search only the target directory.

Requirements:
- macOS with Keynote installed
- Python 3

Usage:
  python create_keynote_from_mp4s.py \
    [--input "/path/to/dir"] [--theme "White"] [--output "MyVideos.key"] \
    [--non-recursive]
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def build_applescript(dir_path: str, mp4_paths: list[str], theme_name: str, out_key_path: str) -> str:
    # Escape paths for AppleScript POSIX file
    def esc(path: str) -> str:
        return path.replace("\\", "\\\\").replace("\"", "\\\"")

    # Build AppleScript that creates a doc, inserts videos slide by slide, then saves
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
    # Remove the default slide if we will add our own slides
    if len(mp4_paths) > 0:
        lines.append('    try')
        lines.append('        tell theDoc to delete slide 1')
        lines.append('    end try')

    for idx, mp4_abs in enumerate(mp4_paths, start=1):
        posix_path = esc(mp4_abs)
        # Extract folder name and file name for title
        import os as path_os
        folder_name = path_os.path.basename(path_os.path.dirname(mp4_abs))
        file_name = path_os.path.splitext(path_os.path.basename(mp4_abs))[0]
        slide_title = f"{folder_name}/{file_name}"
        escaped_title = esc(slide_title)
        
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
        lines.append('            -- Insert movie/image into layout placeholder')
        lines.append('            try')
        lines.append(f'                set movieFile to alias (POSIX file "{posix_path}")')
        lines.append('                -- Method 1: Try to replace existing image placeholder')
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
        lines.append('                -- Method 2: Try movie placeholders if image failed')
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
        lines.append('                -- Method 3: Create new image as fallback')
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
    parser = argparse.ArgumentParser(description="Create Keynote from MP4s in a directory.")
    parser.add_argument("--input", type=str, default=os.getcwd(), help="Target directory (default: cwd)")
    parser.add_argument("--theme", type=str, default="White", help="Keynote theme name (default: White)")
    parser.add_argument("--output", type=str, default="", help="Output .key file name (default: auto)")
    parser.add_argument("--non-recursive", action="store_true", help="Search only current directory (not subdirectories)")
    parser.add_argument("--subdirs-only", action="store_true", help="[DEPRECATED] Use --non-recursive instead")
    parser.add_argument("--verbose", action="store_true", help="Print generated AppleScript for debugging")
    args = parser.parse_args()

    dir_path = os.path.abspath(args.input)
    if not os.path.isdir(dir_path):
        print("Directory not found:", dir_path, file=sys.stderr)
        sys.exit(1)

    # Collect MP4 files recursively by default
    mp4_paths: list[str] = []
    if args.non_recursive or args.subdirs_only:
        # Non-recursive: only current directory
        if args.subdirs_only:
            print("Warning: --subdirs-only is deprecated. The script now searches recursively by default.", file=sys.stderr)
            print("         Use --non-recursive to search only the current directory.", file=sys.stderr)
        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            # Skip hidden and AppleDouble files (like ._*) that Keynote can't use
            if os.path.isfile(full_path) and filename.lower().endswith(".mp4") and not filename.startswith('.') and not filename.startswith('._'):
                mp4_paths.append(full_path)
        mp4_paths.sort()
    else:
        # Recursive: walk all subdirectories
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                # Skip hidden and AppleDouble files (like ._*) that Keynote can't use
                if filename.lower().endswith(".mp4") and not filename.startswith('.') and not filename.startswith('._'):
                    full_path = os.path.join(dirpath, filename)
                    mp4_paths.append(full_path)
        mp4_paths.sort()
    
    if not mp4_paths:
        search_mode = "recursively" if not (args.non_recursive or args.subdirs_only) else "in current directory"
        print(f"No MP4 files found {search_mode} in: {dir_path}")
        sys.exit(0)
    
    search_mode = "recursively" if not (args.non_recursive or args.subdirs_only) else "non-recursively"
    print(f"Found {len(mp4_paths)} MP4 file(s) {search_mode} in: {dir_path}")

    # Default output: current folder name.key if --output not provided
    default_base = os.path.basename(os.path.abspath(dir_path).rstrip(os.sep)) or "Presentation"
    out_name = args.output.strip() or f"{default_base}.key"
    if not out_name.lower().endswith('.key'):
        out_name += '.key'
    out_key_path = os.path.join(dir_path, out_name)

    ascript = build_applescript(dir_path, mp4_paths, args.theme, out_key_path)
    if args.verbose:
        print("--- AppleScript begin ---")
        print(ascript)
        print("--- AppleScript end ---")

    # Always run osascript and print outputs to aid diagnosis
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
    print("Saved Keynote:", out_key_path)


if __name__ == "__main__":
    main()


