import json
import cv2
import argparse
import os
import random
from shapely.geometry import Polygon, Point

COLOR = {
    "A Site": (0, 255, 0),  # Green
    "B Site": (0, 0, 255),  # Red
    "A Main": (0, 255, 255),  # Yellow
    "B Main": (255, 255, 0),  # Cyan
    "T Spawn": (255, 0, 255),  # Magenta
    "CT Spawn": (255, 0, 0),  # Red
    "Mid 1": (0, 255, 0),  # Green
    "Mid 2": (0, 0, 255),  # Red
    "Mid 3": (255, 255, 0),  # Cyan
    "Garden": (255, 0, 255),  # Magenta
    "Tree": (255, 0, 0),  # Red
    "A Heaven": (0, 255, 0),  # Green
    "Market": (0, 0, 255),  # Red
    "Unknown": (200, 200, 200)
}

def load_polygons(mask_path):
    import json
    from shapely.geometry import Polygon
    with open(mask_path) as f:
        raw = json.load(f)
    return {k: Polygon(v) for k, v in raw.items()}


def draw_polygons(img, polygons):
    for name, poly in polygons.items():
        pts = [(int(x), int(y)) for x, y in list(poly.exterior.coords)]
        cv2.polylines(img, [np.array(pts, dtype=int)], True, COLOR.get(name, (255,255,255)), 1)


def main():
    parser = argparse.ArgumentParser("Visual check of site polygons vs detections")
    parser.add_argument("--frame", required=True, help="Path to 640x640 minimap crop image (or full frame, will crop top-left 640x640)")
    parser.add_argument("--events", required=True, help="Path to events JSON produced by extractor")
    parser.add_argument("--mask", required=True, help="Path to site mask JSON")
    parser.add_argument("--out", default="overlay.png", help="Output image path")
    parser.add_argument("--timestamp", type=float, help="Filter events at given video time (seconds)")
    args = parser.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise FileNotFoundError(args.frame)
    if img.shape[0] > 640 or img.shape[1] > 640:
        img = img[52:406, 70:405]

    polys = load_polygons(args.mask)

    import numpy as np
    draw_polygons(img, polys)

    with open(args.events) as f:
        events = json.load(f)

    # Filter detections
    agents = [e for e in events if e.get("event_type") == "agent_detection"]
    if args.timestamp is not None:
        target_ms = int(args.timestamp*1000)
        agents = [e for e in agents if abs(e["timestamp_ms"] - target_ms) < 40]  # ±40 ms window
    # if no timestamp, keep all but cap visual clutter
    if args.timestamp is None and len(agents) > 300:
        agents = random.sample(agents, 300)

    for e in agents:
        cx = int(e["details"]["center_x"])
        cy = int(e["details"]["center_y"])
        p = Point(cx, cy)
        site = "Unknown"
        for name, poly in polys.items():
            if poly.contains(p):
                site = name
                break
        cv2.circle(img, (cx, cy), 3, COLOR.get(site, (255,255,255)), -1)

    cv2.imwrite(args.out, img)
    print(f"Overlay saved to {args.out}")

if __name__ == "__main__":
    import numpy as np
    main()
