"""
Interactive GUI Preview for VAE Noise Fix — entry point.

The implementation lives in the ``gui`` sub-package.  This file only
parses the command-line argument and launches the application, keeping
the entry point thin and trivially testable.

Usage:
    python gui_preview.py                         # file dialog
    python gui_preview.py path/to/image.png       # direct open
    python gui_preview.py path/to/folder/         # browse folder

Controls:
    Trackbars            gradient_sensitivity, max_noise_size, mask_dilate
    1                    Original image
    2                    Mask overlay (red)
    3                    Mask only (white on black, green circles)
    4                    Repaired image
    5                    Side-by-side (original | repaired)
    6                    Laplacian binary energy map
    7                    Median residual binary map
    8                    Combined seed mask (high sensitivity)
    9                    Context mask (low sensitivity)
    0                    Filtered candidates (post structural filter)
    -                    Final verified mask (post LAB; pre-dilation)
    Ctrl + Scroll        Zoom canvas in/out at mouse position
    R                    Reset canvas zoom
    Z                    Toggle zoom lens on/off
    Scroll               Adjust zoom lens magnification
    S                    Save current view
    A / D                Previous / Next image (folder mode)
    Q / ESC              Quit
"""

from __future__ import annotations

import sys

from gui import PreviewApp, collect_image_paths, open_file_dialog


def main() -> None:
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = open_file_dialog()
        if not target:
            print("No file selected. Exiting.")
            return

    paths = collect_image_paths(target)
    if not paths:
        print(f"No images found at: {target}")
        return

    print(f"Loaded {len(paths)} image(s). Launching preview...")
    print()
    print("  Controls:")
    print("    1-5          Switch main view (Original / Overlay / Mask / Repaired / Compare)")
    print("    6-0, -       Switch debug view (Laplacian / Median / Seed / Context / Filtered / Verified)")
    print("    Ctrl+Scroll  Zoom canvas in/out at mouse position")
    print("    R            Reset canvas zoom")
    print("    Z            Toggle zoom lens")
    print("    Scroll       Adjust zoom lens magnification")
    print("    S            Save current view")
    print("    A / D        Previous / Next image")
    print("    Q / ESC      Quit")
    print()

    PreviewApp(paths).run()


if __name__ == "__main__":
    main()
