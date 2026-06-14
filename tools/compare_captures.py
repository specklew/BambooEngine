#!/usr/bin/env python3
"""Compare two screenshots: per-pixel error (mean/std/RMSE) and LDR-FLIP.

Usage: python tools/compare_captures.py A.png B.png [--out heatmap.png] [--no-map]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    import flip_evaluator as flip
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def per_pixel_error(reference, test):
    error = reference - test
    mean_abs = float(np.abs(error).mean(axis=(0, 1)).mean())
    std = float(error.std(axis=(0, 1)).mean())
    rmse = float(np.sqrt((error ** 2).mean(axis=(0, 1))).mean())
    return mean_abs, std, rmse


def main():
    parser = argparse.ArgumentParser(description="Per-pixel error + LDR-FLIP between two screenshots.")
    parser.add_argument("reference", help="reference image (PNG)")
    parser.add_argument("test", help="test image (PNG)")
    parser.add_argument("--out", help="heatmap output path (default: cwd <ref>-vs-<test>.flip.png)")
    parser.add_argument("--no-map", action="store_true", help="skip writing the FLIP error-map heatmap")
    args = parser.parse_args()

    reference = load_rgb(args.reference)
    test = load_rgb(args.test)
    if reference.shape != test.shape:
        sys.exit(f"Resolution mismatch: {reference.shape[1::-1]} vs {test.shape[1::-1]}")

    mean_abs, std, rmse = per_pixel_error(reference, test)
    error_map, mean_flip, _ = flip.evaluate(reference, test, "LDR")

    ref_stem = Path(args.reference).stem
    test_stem = Path(args.test).stem
    heatmap_path = Path(args.out) if args.out else Path.cwd() / f"{ref_stem}-vs-{test_stem}.flip.png"
    json_path = heatmap_path.with_suffix(".json")

    results = {
        "reference": args.reference,
        "test": args.test,
        "meanAbsError": mean_abs,
        "stdError": std,
        "rmse": rmse,
        "flipMean": float(mean_flip),
    }

    if not args.no_map:
        Image.fromarray((error_map * 255.0 + 0.5).astype(np.uint8)).save(heatmap_path)
        results["heatmap"] = str(heatmap_path)

    json_path.write_text(json.dumps(results, indent=2))

    print(f"Mean |error|: {mean_abs:.6f}")
    print(f"Std error:    {std:.6f}")
    print(f"RMSE:         {rmse:.6f}")
    print(f"FLIP (mean):  {mean_flip:.6f}")
    if not args.no_map:
        print(f"Heatmap:      {heatmap_path}")
    print(f"Results:      {json_path}")


if __name__ == "__main__":
    main()
