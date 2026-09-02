#!/usr/bin/env python3
"""Compare two screenshots: per-pixel error (mean/std/RMSE) and LDR-FLIP.

Usage: python tools/compare_captures.py A.png B.png [--out heatmap.png] [--no-map]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

# One implementation of the metric for every tool here — the weighted statistics have to
# come out of the same map the mean did, or two documents disagree about one image.
from bench_report import (flip_error_map, flip_settings, load_rgb,  # noqa: E402
                          magma, weighted_quantiles)


def per_pixel_error(reference, test):
    error = reference - test
    mean_abs = float(np.abs(error).mean(axis=(0, 1)).mean())
    std = float(error.std(axis=(0, 1)).mean())
    mse = float((error ** 2).mean())
    rmse = float(np.sqrt((error ** 2).mean(axis=(0, 1))).mean())
    return mean_abs, std, mse, rmse


def verify_against_tool(reference_path, test_path, ours):
    """Run the FLIP console tool on the same pair and print the disagreement.

    The pooled statistics here are computed from the map rather than shelled out per
    image, because scoring a campaign is thousands of pairs and the tool reloads both
    images every time. This is the check that keeps that shortcut honest — and it is a
    check, not a test suite: it prints, it does not assert, because the tool pools into
    100 buckets and a difference of ~1e-5 is expected rather than wrong."""
    try:
        completed = subprocess.run(["flip", "-r", str(reference_path), "-t", str(test_path),
                                    "-v", "2", "-nexm", "-nerm"],
                                   capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"--verify-flip: could not run the FLIP tool ({exc})")
        return

    labels = {"mean": r"Mean:\s+([\d.]+)", "median": r"Weighted median:\s+([\d.]+)",
              "q1": r"1st weighted quartile:\s+([\d.]+)", "q3": r"3rd weighted quartile:\s+([\d.]+)"}
    print("\n--verify-flip — ours vs the FLIP tool")
    for key, pattern in labels.items():
        found = re.search(pattern, completed.stdout)
        if not found:
            print(f"  {key:<7} tool output did not carry this value")
            continue
        theirs = float(found.group(1))
        print(f"  {key:<7} {ours[key]:.6f}  tool {theirs:.6f}  delta {abs(ours[key] - theirs):.2e}")


def main():
    parser = argparse.ArgumentParser(description="Per-pixel error + LDR-FLIP between two screenshots.")
    parser.add_argument("reference", help="reference image (PNG)")
    parser.add_argument("test", help="test image (PNG)")
    parser.add_argument("--out", help="heatmap output path (default: cwd <ref>-vs-<test>.flip.png)")
    parser.add_argument("--no-map", action="store_true", help="skip writing the FLIP error-map heatmap")
    parser.add_argument("--verify-flip", action="store_true",
                        help="also run the FLIP console tool on this pair and print the disagreement")
    args = parser.parse_args()

    reference = load_rgb(args.reference)
    test = load_rgb(args.test)
    if reference.shape != test.shape:
        sys.exit(f"Resolution mismatch: {reference.shape[1::-1]} vs {test.shape[1::-1]}")

    mean_abs, std, mse, rmse = per_pixel_error(reference, test)
    error_map, mean_flip = flip_error_map(reference, test)
    # Mean is the headline (see docs/adr/0022); the rest describe the distribution
    # the mean came from, which a single number cannot. The weighted trio is what the
    # FLIP tool itself pools and what section 8.1 asks to report beside the mean.
    flat = error_map.ravel()
    flip_q1, flip_median, flip_q3, flip_p95 = np.quantile(flat, [0.25, 0.5, 0.75, 0.95])
    weighted_q1, weighted_median, weighted_q3 = weighted_quantiles(error_map, [0.25, 0.5, 0.75])

    ref_stem = Path(args.reference).stem
    test_stem = Path(args.test).stem
    heatmap_path = Path(args.out) if args.out else Path.cwd() / f"{ref_stem}-vs-{test_stem}.flip.png"
    json_path = heatmap_path.with_suffix(".json")

    results = {
        "reference": args.reference,
        "test": args.test,
        "meanAbsError": mean_abs,
        "stdError": std,
        "mse": mse,
        "rmse": rmse,
        "flipMean": float(mean_flip),
        "flipMedian": float(flip_median),
        "flipQ1": float(flip_q1),
        "flipQ3": float(flip_q3),
        "flipP95": float(flip_p95),
        "flipMax": float(flat.max()),
        "flipWeightedMedian": weighted_median,
        "flipWeightedQ1": weighted_q1,
        "flipWeightedQ3": weighted_q3,
        "flip": flip_settings(),
    }

    if not args.no_map:
        Image.fromarray(magma(error_map)).save(heatmap_path)
        results["heatmap"] = str(heatmap_path)

    json_path.write_text(json.dumps(results, indent=2))

    print(f"Mean |error|: {mean_abs:.6f}")
    print(f"Std error:    {std:.6f}")
    print(f"MSE:          {mse:.6f}")
    print(f"RMSE:         {rmse:.6f}")
    print(f"FLIP (mean):  {mean_flip:.6f}   [{flip_settings()['version']}, LDR, "
          f"{flip_settings()['ppd']:.2f} ppd]")
    print(f"FLIP weighted median: {weighted_median:.6f}   Q1 {weighted_q1:.6f}  Q3 {weighted_q3:.6f}")
    print(f"FLIP median:  {flip_median:.6f}   Q1 {flip_q1:.6f}  Q3 {flip_q3:.6f}  p95 {flip_p95:.6f}")

    if args.verify_flip:
        verify_against_tool(args.reference, args.test,
                            {"mean": mean_flip, "median": weighted_median,
                             "q1": weighted_q1, "q3": weighted_q3})
    if not args.no_map:
        print(f"Heatmap:      {heatmap_path}")
    print(f"Results:      {json_path}")


if __name__ == "__main__":
    main()
