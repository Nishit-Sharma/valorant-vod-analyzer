import json
import argparse
import os


def convert_points(points, offset_x, offset_y, src_w, src_h, dst_w, dst_h, dx=0, dy=0):
    """Scale/shift a list of [x,y] points"""
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    conv = []
    for x, y in points:
        new_x = (x - offset_x) * scale_x + dx
        new_y = (y - offset_y) * scale_y + dy
        conv.append([round(new_x, 2), round(new_y, 2)])
    return conv


def main():
    parser = argparse.ArgumentParser("Convert site mask coordinates from a cropped image size to 640x640 minimap coordinates")
    parser.add_argument("--mask", required=True, help="Path to original JSON mask")
    parser.add_argument("--offset", nargs=2, type=int, metavar=("X0", "Y0"), required=True, help="Top-left pixel of the minimap inside the original screenshot")
    parser.add_argument("--size", nargs=2, type=int, metavar=("SRC_W", "SRC_H"), required=True, help="Width/height of the image you drew on")
    parser.add_argument("--dst", nargs=2, type=int, metavar=("DST_W", "DST_H"), default=[640,640],
                        help="Target minimap size (default 640 640). If your code crops 335×354 pass those numbers")
    parser.add_argument("--shift", nargs=2, type=float, metavar=("DX","DY"), default=[0,0],
                        help="Extra pixels to add AFTER scaling – positive moves right/down")
    args = parser.parse_args()

    with open(args.mask, "r") as f:
        data = json.load(f)

    off_x, off_y = args.offset
    src_w, src_h = args.size

    new_data = {}
    dst_w, dst_h = args.dst
    dx, dy = args.shift
    for name, pts in data.items():
        new_data[name] = convert_points(pts, off_x, off_y, src_w, src_h, dst_w, dst_h, dx, dy)

    backup = args.mask + ".bak"
    os.rename(args.mask, backup)
    with open(args.mask, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Converted mask written to {args.mask} (backup saved to {backup})")


if __name__ == "__main__":
    main()
