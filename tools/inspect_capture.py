#!/usr/bin/env python3
"""Inspect a single screenshot: region/channel stats, color-anomaly localization,
and exposure-boosted crops. Replaces ad-hoc one-off image scripts.

Examples:
  # whole-image per-channel stats
  python tools/inspect_capture.py shot.png

  # stats over a sub-rectangle (x0,y0,x1,y1 in pixels)
  python tools/inspect_capture.py shot.png --region 1500,500,1600,620

  # locate blue-dominant pixels (count, bounding box, summed energy)
  python tools/inspect_capture.py shot.png --find-color b --threshold 0.06

  # write a 4x exposure-boosted crop, and a copy with found pixels marked red
  python tools/inspect_capture.py shot.png --find-color b --crop out.png --mark marked.png --exposure 4

Add --json for machine-readable output.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def parse_region(text, width, height):
    try:
        x0, y0, x1, y1 = (int(v) for v in text.split(","))
    except ValueError:
        sys.exit("--region must be 'x0,y0,x1,y1' integer pixels")
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    y0, y1 = sorted((max(0, y0), min(height, y1)))
    if x0 == x1 or y0 == y1:
        sys.exit("--region is empty after clamping to the image bounds")
    return x0, y0, x1, y1


def channel_stats(pixels):
    flat = pixels.reshape(-1, 3)
    return {
        "mean": [round(float(v), 4) for v in flat.mean(0)],
        "min":  [round(float(v), 4) for v in flat.min(0)],
        "max":  [round(float(v), 4) for v in flat.max(0)],
        "std":  [round(float(v), 4) for v in flat.std(0)],
    }


def find_color(pixels, channel, threshold):
    c = CHANNEL_INDEX[channel]
    others = [i for i in (0, 1, 2) if i != c]
    mask = (pixels[:, :, c] > pixels[:, :, others[0]] + threshold) & \
           (pixels[:, :, c] > pixels[:, :, others[1]] + threshold)
    ys, xs = np.where(mask)
    result = {
        "channel": channel,
        "threshold": threshold,
        "count": int(mask.sum()),
        "energy": round(float(pixels[:, :, c][mask].sum()), 3),
    }
    if len(xs):
        result["bbox"] = {"x0": int(xs.min()), "y0": int(ys.min()),
                          "x1": int(xs.max()), "y1": int(ys.max())}
        result["maxValue"] = round(float(pixels[:, :, c][mask].max()), 3)
    return mask, result


def save_image(array, path, exposure):
    out = np.clip(array * exposure, 0.0, 1.0)
    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description="Inspect one screenshot: stats, color anomalies, crops.")
    parser.add_argument("image", help="input image (PNG)")
    parser.add_argument("--region", help="x0,y0,x1,y1 sub-rectangle (default: whole image)")
    parser.add_argument("--find-color", choices=list(CHANNEL_INDEX), help="locate pixels where this channel dominates")
    parser.add_argument("--threshold", type=float, default=0.06, help="dominance margin for --find-color (default 0.06)")
    parser.add_argument("--crop", help="write an exposure-scaled crop of the region to this path")
    parser.add_argument("--mark", help="write a full image with found pixels marked red to this path")
    parser.add_argument("--exposure", type=float, default=1.0, help="multiplier applied before saving --crop/--mark")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    image = load_rgb(args.image)
    height, width, _ = image.shape

    region = parse_region(args.region, width, height) if args.region else (0, 0, width, height)
    x0, y0, x1, y1 = region
    sub = image[y0:y1, x0:x1]

    report = {"image": args.image, "size": [width, height], "region": list(region),
              "stats": channel_stats(sub)}

    if args.find_color:
        mask, found = find_color(sub, args.find_color, args.threshold)
        report["findColor"] = found
        if args.mark:
            marked = np.clip(image * args.exposure, 0.0, 1.0)
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y0:y1, x0:x1] = mask
            marked[full_mask] = [1.0, 0.0, 0.0]
            save_image(marked, args.mark, 1.0)
            report["markPath"] = args.mark

    if args.crop:
        save_image(sub, args.crop, args.exposure)
        report["cropPath"] = args.crop

    if args.json:
        print(json.dumps(report, indent=2))
        return

    s = report["stats"]
    print(f"Image:  {args.image}  ({width}x{height})")
    print(f"Region: {region}")
    print(f"  mean rgb {s['mean']}  min {s['min']}  max {s['max']}  std {s['std']}")
    if "findColor" in report:
        f = report["findColor"]
        print(f"Find '{f['channel']}'-dominant (thr {f['threshold']}): count={f['count']} energy={f['energy']}")
        if f["count"]:
            print(f"  bbox {f['bbox']}  maxValue={f['maxValue']}")
    if "cropPath" in report: print(f"Crop:   {report['cropPath']}  (exposure {args.exposure})")
    if "markPath" in report: print(f"Marked: {report['markPath']}")


if __name__ == "__main__":
    main()
