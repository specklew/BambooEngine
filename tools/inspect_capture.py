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


# Palette of GuidedPathTracingPass debug view 4 (GuideAcceptance), straight from the
# colour branch at the end of guidedPathTracing.hlsl. Counting these classes is how a
# guided run reports where its first-bounce samples go.
GUIDE_ACCEPTANCE_PALETTE = [
    ("accepted",        (0.0, 1.0, 0.0), "hit inside the sampled voxel, sample used"),
    ("blocked-short",   (1.0, 0.0, 0.0), "occluder between the shading point and the voxel"),
    ("no-cluster",      (0.0, 1.0, 1.0), "chain found no cluster"),
    ("pdf-nonpositive", (1.0, 0.5, 0.0), "guide pdf <= 0"),
    ("below-horizon",   (0.5, 0.0, 1.0), "sampled direction under the surface"),
    ("zero-brdf",       (1.0, 1.0, 0.0), "BRDF zero at the sampled direction"),
    ("crossed-empty",   (1.0, 1.0, 1.0), "reached the voxel and hit nothing"),
    ("guide-dead",      (0.0, 0.0, 1.0), "no live parent / dead branch"),
    ("background",      (0.0, 0.0, 0.0), "no shading point (primary miss)"),
]

PALETTES = {"guide-acceptance": GUIDE_ACCEPTANCE_PALETTE}


def aces_gamma(values):
    """The engine's display transform (postprocess.hlsl): ACES filmic (Narkowicz 2015)
    then linear->gamma. A debug view's palette colours arrive in a PNG through it, so a
    classifier has to compare against the transformed palette, not the authored one."""
    x = np.asarray(values, dtype=np.float32)
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)


def classify(pixels, palette, tolerance, transform=None):
    """Per-pixel nearest palette entry; -1 where nothing is within `tolerance`."""
    colors = np.array([entry[1] for entry in palette], dtype=np.float32)
    if transform is not None:
        colors = transform(colors)
    flat = pixels.reshape(-1, 3)
    distance = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
    ids = distance.argmin(axis=1)
    ids[distance.min(axis=1) > tolerance ** 2] = -1
    return ids.reshape(pixels.shape[:2])


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
    parser.add_argument("--classify", choices=list(PALETTES), help="count pixels per class of a known debug-view palette")
    parser.add_argument("--display-transform", choices=["none", "aces"], default="aces",
                        help="display curve the capture already went through (default aces, as postprocess.hlsl)")
    parser.add_argument("--tolerance", type=float, default=0.12, help="max per-pixel distance to a palette colour (default 0.12)")
    parser.add_argument("--class-mask", help="with --classify: write the image with one class marked red")
    parser.add_argument("--mask-class", help="which class --class-mask marks (default: blocked-short)")
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

    if args.classify:
        palette = PALETTES[args.classify]
        ids = classify(sub, palette, args.tolerance,
                       aces_gamma if args.display_transform == "aces" else None)
        total = ids.size
        counts = []
        for index, (name, _color, meaning) in enumerate(palette):
            n = int((ids == index).sum())
            counts.append({"class": name, "count": n, "share": round(100.0 * n / total, 3), "meaning": meaning})
        unmatched = int((ids < 0).sum())
        counts.append({"class": "unmatched", "count": unmatched,
                       "share": round(100.0 * unmatched / total, 3),
                       "meaning": "no palette colour within the tolerance (blended pixel)"})
        report["classify"] = {"palette": args.classify, "tolerance": args.tolerance,
                              "pixels": total, "classes": counts}
        if unmatched:
            # An unmatched population is usually the palette scaled or blended, not a new
            # class, so the commonest unmatched colours say which of the two it is.
            flat = sub.reshape(-1, 3)[ids.reshape(-1) < 0]
            quantized = np.round(flat * 32.0) / 32.0
            uniques, hits = np.unique(quantized, axis=0, return_counts=True)
            order = np.argsort(-hits)[:5]
            report["classify"]["unmatchedColors"] = [
                {"rgb": [round(float(v), 3) for v in uniques[i]], "count": int(hits[i])} for i in order]
        if args.class_mask:
            wanted = args.mask_class or "blocked-short"
            names = [entry[0] for entry in palette]
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = ids == names.index(wanted)
            marked = np.clip(image * args.exposure, 0.0, 1.0)
            marked[mask] = [1.0, 0.0, 0.0]
            save_image(marked, args.class_mask, 1.0)
            report["classMaskPath"] = args.class_mask

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
    if "classify" in report:
        c = report["classify"]
        print(f"Classify '{c['palette']}' over {c['pixels']} px (tolerance {c['tolerance']}):")
        for entry in c["classes"]:
            if entry["count"]:
                print(f"  {entry['class']:<16} {entry['count']:>9}  {entry['share']:>6.2f} %  {entry['meaning']}")
        for entry in c.get("unmatchedColors", []):
            print(f"  unmatched rgb {entry['rgb']}  x{entry['count']}")
        if "classMaskPath" in report: print(f"Class mask: {report['classMaskPath']}")
    if "cropPath" in report: print(f"Crop:   {report['cropPath']}  (exposure {args.exposure})")
    if "markPath" in report: print(f"Marked: {report['markPath']}")


if __name__ == "__main__":
    main()
