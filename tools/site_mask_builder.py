import cv2
import json
import os
from typing import Dict, List, Tuple

def collect_polygon_points(image_path: str) -> List[Tuple[int, int]]:
    """Open an image and let the user click polygon vertices.

    Left-click     – add a point
    Right-click    – remove last point
    ENTER / SPACE  – finish polygon
    ESC            – abort
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    points: List[Tuple[int, int]] = []
    clone = img.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
        # redraw
        clone = img.copy()
        for p in points:
            cv2.circle(clone, p, 3, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(clone, [np.array(points, np.int32)], False, (0,255,0), 1)

    import numpy as np
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse)

    while True:
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            break
        elif key == 27:  # ESC
            points = []
            break
    cv2.destroyAllWindows()
    return points


def build_site_masks(minimap_image: str, map_name: str, output_dir: str = "site_masks"):
    os.makedirs(output_dir, exist_ok=True)
    sites = {}
    for site in ["A Site", "B Site", "A Main", "B Main", "T Spawn", "CT Spawn", "Mid 1", "Mid 2", "Mid 3", "Middle", "Garden", "Tree", "A Heaven", "Market"]:
        print(f"Draw polygon for site {site}. When done press ENTER/SPACE. Right-click removes last point.")
        pts = collect_polygon_points(minimap_image)
        if not pts:
            print("Aborted.")
            return
        sites[site] = pts
    output_path = os.path.join(output_dir, f"{map_name}.json")
    with open(output_path, "w") as f:
        json.dump(sites, f, indent=2)
    print(f"✅ Saved mask to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive tool to create bomb-site polygons for a Valorant map.")
    parser.add_argument("--image", required=True, help="Path to 640×640 minimap reference image")
    parser.add_argument("--map", required=True, help="Map name, e.g. bind, ascent")
    args = parser.parse_args()

    build_site_masks(args.image, args.map)
