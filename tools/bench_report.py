#!/usr/bin/env python3
"""Score a headless run against a reference and aggregate it.

The engine's --images/--checkpoints protocol writes many images per configuration:
M independent images (accumulation reset between them, RNG stream running on) and
optionally K checkpoints inside each image's accumulation. This script scores every
one of them and reports the two aggregates those axes are for:

  across images     -> the error bar on a single number (mean, std, CI95, quartiles)
  across checkpoints-> the convergence curve, aggregated per checkpoint over images

Per image it computes MSE/RMSE/mean|err| plus the FLIP map's mean (the headline —
see below), median, quartiles, p95 and max. The mean is the headline statistic
because the weighted median that the FLIP tool prints is blind to fireflies, and
fireflies are exactly what separates a guided integrator from an unguided one.

Usage:
  python tools/bench_report.py <runDir> --reference ref.png
  python tools/bench_report.py <runDir> --reference ref.png --maps out/
  python tools/bench_report.py <runDir> --reference ref.png --json report.json
"""

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    import flip_evaluator as flip
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_exe(build=None):
    """Which build of the engine the harness drives.

    Order: the --build flag, then BAMBOO_BUILD, then Release. Release is the default
    because a measurement should describe the program that ships; a diagnostic build is
    not a configuration anyone runs. Debug remains selectable for one purpose only —
    correctness work with the D3D12 debug layer or GPU-based validation, which is a
    different axis entirely (5.1 ms -> 1118 ms/frame) and produces no timing numbers.
    Nothing is lost by the switch: Debug and Release frame cost were measured to agree
    within 1% for both techniques (R9), because the frame is GPU-work-bound.
    """
    name = build or os.environ.get("BAMBOO_BUILD") or "Release"
    exe = REPO_ROOT / "x64" / name / "Raytracer.exe"
    if not exe.exists():
        sys.exit(f"No engine at {exe}. Build the {name} configuration, or pass --build/BAMBOO_BUILD.")
    return exe


def add_build_argument(parser):
    parser.add_argument("--build", choices=["Debug", "Release"],
                        help="engine build to drive (default: $BAMBOO_BUILD, else Release)")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# The measurement conditions of the metric itself, recorded because PLAN_BADAWCZY 16.7
# asks for them and because neither is visible in a number that was already computed.
# The pixels-per-degree comes back from FLIP's own parameter dictionary — asking the
# library beats restating its default, which is what would silently go stale.
FLIP_VERSION = importlib.metadata.version("flip_evaluator")
FLIP_PARAMETERS = {}


def weighted_quantiles(gray_map, quantiles):
    """FLIP's own pooled statistics: quantiles of the error map WEIGHTED BY THE ERROR.

    This is not the median of the pixel values — it is the error level below which a
    given share of the total error MASS lies, which is what the FLIP tool prints as
    'weighted median' and 'weighted quartiles' and what section 8.1 asks to report.
    The tool pools into 100 histogram buckets and this is exact, so the two agree to
    about 1e-5 rather than exactly; verified against flip.exe 1.7 on veach-ajar
    (0.114183 vs 0.114192 median, 0.071876 vs 0.071869 Q1, 0.165719 vs 0.165681 Q3).
    `compare_captures.py --verify-flip` re-runs that comparison on any pair."""
    ordered = np.sort(np.asarray(gray_map, dtype=np.float64).ravel())
    cumulative = np.cumsum(ordered)
    total = cumulative[-1]
    if total <= 0.0:
        return [0.0 for _ in quantiles]
    return [float(ordered[min(int(np.searchsorted(cumulative, q * total)), ordered.size - 1)])
            for q in quantiles]


def flip_settings():
    """The metric's settings, available before anything has been scored.

    A report header is written before the first image is read, so the lazily captured
    dictionary would still be empty there. FLIP's pixels-per-degree is a viewing
    condition, not a property of the image, so one evaluation of a dummy pair answers it
    for every pair that follows."""
    if not FLIP_PARAMETERS:
        dummy = np.zeros((8, 8, 3), dtype=np.float32)
        flip_error_map(dummy, dummy)
    return dict(FLIP_PARAMETERS, dynamicRange="LDR")


def flip_error_map(reference, test):
    """The FLIP error map as ERROR, not as a picture of it.

    `applyMagma=True` (the library default) returns the magma-COLOURED map, and every
    statistic taken from that reads the colour ramp instead of the error: it is not
    monotone per channel and its range is the palette's, not the metric's. Only the
    mean survived, because FLIP computes it internally before colouring."""
    error_map, flip_mean, parameters = flip.evaluate(reference, test, "LDR", applyMagma=False)
    if not FLIP_PARAMETERS:
        FLIP_PARAMETERS.update({"ppd": parameters.get("ppd"), "version": FLIP_VERSION})
    return np.asarray(error_map, dtype=np.float64).reshape(error_map.shape[:2]), float(flip_mean)


def score_image(reference, test_path):
    test = load_rgb(test_path)
    if reference.shape != test.shape:
        sys.exit(f"Resolution mismatch on {test_path}: {test.shape[1::-1]} vs {reference.shape[1::-1]}")

    error = reference - test
    error_map, flip_mean = flip_error_map(reference, test)
    flat = error_map.ravel()
    q1, median, q3, p95 = np.quantile(flat, [0.25, 0.5, 0.75, 0.95])
    weighted_q1, weighted_median, weighted_q3 = weighted_quantiles(error_map, [0.25, 0.5, 0.75])

    return {
        "mse": float((error ** 2).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "meanAbsError": float(np.abs(error).mean()),
        "flipMean": float(flip_mean),
        "flipMedian": float(median),
        "flipQ1": float(q1),
        "flipQ3": float(q3),
        "flipP95": float(p95),
        # 8.1 reports the metric paper's own set, and that set has both ends of the range.
        "flipMin": float(flat.min()),
        "flipMax": float(flat.max()),
        "flipWeightedMedian": weighted_median,
        "flipWeightedQ1": weighted_q1,
        "flipWeightedQ3": weighted_q3,
    }, error_map


# FLIP's own display palette, so a map written by these tools looks like one written by
# the tool. Falls back to grayscale rather than failing: the colour is presentation.
def magma(gray_map):
    array = np.clip(np.asarray(gray_map, dtype=np.float64), 0.0, 1.0)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import colormaps
        return (colormaps["magma"](array)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    except ImportError:
        return (np.repeat(array[..., None], 3, axis=2) * 255.0 + 0.5).astype(np.uint8)


def sidecar_for(png_path):
    json_path = png_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text())


def aggregate(values):
    """Mean is the reported number; the spread is what says whether to believe it."""
    if not values:
        return {}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) > 1 else (mean, mean, mean)
    return {
        "n": len(values),
        "mean": mean,
        "stdev": stdev,
        "ci95": half_width,
        "min": min(values),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": max(values),
    }


def find_crossing(points, target):
    """Budget at which a falling series first reaches target, or None if it never does.

    Interpolated in log-log, because error against budget is a power law (~1/sqrt(N)
    for an unbiased estimator) and linear interpolation across a decade of budget
    would put the crossing in the wrong place by tens of percent.
    """
    for (budget_a, value_a), (budget_b, value_b) in zip(points, points[1:]):
        if value_a > target >= value_b:
            if min(budget_a, budget_b, value_a, value_b) > 0:
                log_a, log_b = math.log(budget_a), math.log(budget_b)
                value_log_a, value_log_b = math.log(value_a), math.log(value_b)
                t = (math.log(target) - value_log_a) / (value_log_b - value_log_a)
                return math.exp(log_a + t * (log_b - log_a))
            return budget_a + (budget_b - budget_a) * (value_a - target) / (value_a - value_b)
    if points and points[0][1] <= target:
        return points[0][0]  # already below the target at the first checkpoint
    return None


def series_by_arm(entries, value_key, budget_key):
    """(state, technique) -> [(budget, value)] ascending, from per-checkpoint entries."""
    series = defaultdict(list)
    for entry in entries:
        value = entry.get(value_key)
        if isinstance(value, dict):
            value = value.get("mean")
        if value is None or entry.get(budget_key) in (None, 0):
            continue
        series[(entry["state"], entry["technique"])].append((entry[budget_key], value))
    for key in series:
        series[key].sort()
    return series


def report_crossings(entries, value_key, label, baseline_technique=None, target=None):
    """Equal-error (or equal-variance) readout: what budget each arm needs to reach
    the same level. A technique whose curve plateaus above the target does not reach
    it — reported as such, never extrapolated, because the plateau IS the result."""
    by_seconds = series_by_arm(entries, value_key, "seconds")
    by_frames = series_by_arm(entries, value_key, "frames")
    if not by_seconds or all(len(points) < 2 for points in by_seconds.values()):
        return []

    rows = []
    for state in sorted({state for state, _ in by_seconds}):
        arms = {technique: points for (arm_state, technique), points in by_seconds.items() if arm_state == state}
        # The tightest level every arm actually reaches. Choosing anything lower
        # only produces "not reached" rows; anything higher throws away resolution.
        state_target = target if target is not None else max(min(v for _, v in points) for points in arms.values())

        baseline_seconds = None
        if baseline_technique and baseline_technique in arms:
            baseline_seconds = find_crossing(arms[baseline_technique], state_target)

        for technique in sorted(arms):
            seconds = find_crossing(arms[technique], state_target)
            frames = find_crossing(by_frames[(state, technique)], state_target)
            speedup = (baseline_seconds / seconds) if (baseline_seconds and seconds) else None
            rows.append({"state": state, "technique": technique, "metric": label,
                         "target": state_target, "seconds": seconds, "frames": frames,
                         "speedupVsBaseline": speedup})

    if rows:
        print()
        print(f"Equal-{label} readout — budget each arm needs to reach the same level"
              + (f" (baseline {baseline_technique})" if baseline_technique else ""))
        print("| state | technique | target | seconds | frames | speedup |")
        print("|" + "---|" * 6)
        for row in rows:
            seconds = f"{row['seconds']:.3f}" if row["seconds"] else "not reached"
            frames = f"{row['frames']:.0f}" if row["frames"] else "-"
            speedup = f"{row['speedupVsBaseline']:.2f}x" if row["speedupVsBaseline"] else "-"
            print(f"| {row['state']} | {row['technique']} | {row['target']:.6f} | {seconds} | {frames} | {speedup} |")
    return rows


def plot_curves(entries, metric, out_dir, crossings=None):
    """Convergence plots: metric against time and against samples, plus the
    estimator's own variance. Log-log, because that is where a power law is a line
    and a plateau is unmistakable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install -r tools/requirements.txt)")
        return []

    written = []
    states = sorted({entry["state"] for entry in entries})
    for state in states:
        arms = sorted({entry["technique"] for entry in entries if entry["state"] == state})
        has_variance = any("varianceMean" in entry for entry in entries if entry["state"] == state)
        panels = 3 if has_variance else 2

        figure, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.2))
        for technique in arms:
            points = sorted((entry for entry in entries
                             if entry["state"] == state and entry["technique"] == technique),
                            key=lambda e: e["frames"])
            if len(points) < 2:
                continue
            seconds = [p["seconds"] for p in points]
            frames = [p["frames"] for p in points]
            values = [p["metric"]["mean"] for p in points]
            errors = [p["metric"]["ci95"] for p in points]

            axes[0].plot(seconds, values, marker="o", markersize=3, label=technique)
            axes[0].fill_between(seconds,
                                 [v - e for v, e in zip(values, errors)],
                                 [v + e for v, e in zip(values, errors)], alpha=0.2)
            axes[1].plot(frames, values, marker="o", markersize=3, label=technique)
            if has_variance:
                # The first checkpoint has no variance — one sample has nothing to
                # deviate from — so filter per point rather than gating on points[0].
                measured = [p for p in points if "varianceMean" in p]
                if measured:
                    axes[2].plot([p["seconds"] for p in measured],
                                 [p["varianceMean"]["mean"] for p in measured],
                                 marker="o", markersize=3, label=technique)

        # Only this metric's target belongs on these axes; the variance readout has
        # its own scale (linear HDR) and would draw a meaningless line here.
        for target in {row["target"] for row in (crossings or [])
                       if row["state"] == state and row.get("metric") == metric}:
            axes[0].axhline(target, color="grey", linewidth=0.8, linestyle="--")
            axes[1].axhline(target, color="grey", linewidth=0.8, linestyle="--")

        titles = [f"{metric} vs render time", f"{metric} vs samples", "estimator variance vs render time"]
        labels = ["seconds", "frames", "seconds"]
        for index in range(panels):
            axes[index].set_xscale("log")
            axes[index].set_yscale("log")
            axes[index].set_xlabel(labels[index])
            axes[index].set_title(titles[index], fontsize=10)
            axes[index].grid(True, which="both", alpha=0.25, linewidth=0.5)
        axes[0].set_ylabel(metric)
        axes[0].legend(fontsize=8)
        if has_variance:
            # Linear HDR radiance, not the tonemapped display space the other two
            # panels live in — the axis label has to say so or the numbers invite
            # a comparison that is not valid.
            axes[2].set_ylabel("variance of the mean (linear HDR)")
        figure.suptitle(state)
        figure.tight_layout()

        stem = Path(out_dir) / f"curve-{state.replace(' ', '_')}"
        for suffix in (".png", ".pdf"):
            figure.savefig(stem.with_suffix(suffix), dpi=140)
            written.append(str(stem.with_suffix(suffix)))
        plt.close(figure)

    if written:
        print("\nPlots: " + ", ".join(w for w in written if w.endswith(".png")))
    return written


def replot(report_path, metric, baseline, target):
    """Redraw crossings and figures from a finished report — the captures are gone
    by then, but every number the plots need is in the JSON."""
    report = json.loads(Path(report_path).read_text())
    crossings = report_crossings(report["groups"], "metric", metric, baseline_technique=baseline, target=target)
    crossings += report_crossings(report["groups"], "varianceMean", "variance", baseline_technique=baseline)
    report["crossings"] = crossings
    report["plots"] = plot_curves(report["groups"], metric, Path(report_path).parent, crossings)
    Path(report_path).write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Aggregate a headless run against a reference image.")
    parser.add_argument("run_dir", nargs="?", help="run folder written by --out (its run-<timestamp> subfolder)")
    parser.add_argument("--replot", help="regenerate crossings and figures from an existing report.json/bench.json")
    parser.add_argument("--reference", help="converged reference image (PNG); single-state runs")
    parser.add_argument("--reference-dir",
                        help="folder holding '<state>-Path Tracing.png' per state; multi-state runs")
    parser.add_argument("--metric", default="flipMean",
                        help="per-image statistic to aggregate (default flipMean)")
    parser.add_argument("--maps", help="directory to write FLIP maps for the best/median/worst image of each group")
    parser.add_argument("--json", help="write the full report here (default <runDir>/report.json)")
    parser.add_argument("--baseline", help="technique the equal-error speedups are measured against")
    parser.add_argument("--target", type=float,
                        help="error level for the equal-error readout (default: the tightest level every arm reaches)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.replot:
        replot(args.replot, args.metric, args.baseline, args.target)
        return
    if not args.run_dir:
        sys.exit("Pass a run folder, or --replot <report.json>")

    run_dir = Path(args.run_dir)
    # Skip what this tool itself leaves in a run directory — the convergence plot lands
    # beside the captures, and a second pass over the same directory would try to score
    # the plot as an image and stop on the resolution mismatch.
    images = sorted(p for p in run_dir.rglob("*.png")
                    if not p.name.endswith(".flip.png") and not p.name.startswith("curve-"))
    if not images:
        sys.exit(f"No captures under {run_dir}")

    if not args.reference and not args.reference_dir:
        sys.exit("Pass --reference (one image) or --reference-dir (one per state)")

    single_reference = load_rgb(args.reference) if args.reference else None
    references = {}

    def reference_for(state):
        """Each state has its own converged image; a run may span several."""
        if single_reference is not None:
            return single_reference
        if state not in references:
            candidates = sorted(Path(args.reference_dir).rglob(f"{state}-Path Tracing.png"))
            if not candidates:
                sys.exit(f"No reference for state '{state}' under {args.reference_dir}")
            references[state] = load_rgb(candidates[-1])
        return references[state]

    # (state, technique, view) x checkpoint. Images of one configuration differ only
    # by their RNG stream, so they are the repeats; checkpoints are positions on the
    # convergence curve and must never be averaged together with them.
    groups = defaultdict(list)
    for png in images:
        meta = sidecar_for(png)
        scene = meta.get("scene", {})
        bench = meta.get("benchmark", {})
        rt = meta.get("raytracing", {})
        state = scene.get("place", "?")
        key = (state, meta.get("technique", "?"), bench.get("checkpointIndex", 0))

        scores, error_map = score_image(reference_for(state), png)
        scores["path"] = str(png)
        scores["frames"] = rt.get("frameIndex", 0)
        scores["seconds"] = rt.get("accumulatedTime", 0.0)
        scores["meanFrameMs"] = bench.get("meanFrameMs", 0.0)
        scores["varianceMean"] = bench.get("varianceMean")
        scores["varianceRelative"] = bench.get("varianceRelative")
        groups[key].append((scores, error_map if args.maps else None))

    report = {"reference": args.reference or args.reference_dir, "runDir": str(run_dir),
              "metric": args.metric, "groups": []}
    rows = []

    for (place, technique, checkpoint), entries in sorted(groups.items()):
        per_image = [scores for scores, _ in entries]
        values = [scores[args.metric] for scores in per_image]
        summary = aggregate(values)

        variances = [s["varianceMean"] for s in per_image if s.get("varianceMean") is not None]
        group = {
            "state": place,
            "technique": technique,
            "checkpoint": checkpoint,
            "frames": statistics.fmean([s["frames"] for s in per_image]),
            "seconds": statistics.fmean([s["seconds"] for s in per_image]),
            "meanFrameMs": statistics.fmean([s["meanFrameMs"] for s in per_image]),
            "metric": summary,
            "mse": aggregate([s["mse"] for s in per_image]),
            "flipMedian": aggregate([s["flipMedian"] for s in per_image]),
            "flipQ1": aggregate([s["flipQ1"] for s in per_image]),
            "flipQ3": aggregate([s["flipQ3"] for s in per_image]),
            "flipP95": aggregate([s["flipP95"] for s in per_image]),
            "flipWeightedMedian": aggregate([s["flipWeightedMedian"] for s in per_image]),
            "flipWeightedQ1": aggregate([s["flipWeightedQ1"] for s in per_image]),
            "flipWeightedQ3": aggregate([s["flipWeightedQ3"] for s in per_image]),
            "images": per_image,
        }
        if variances:
            group["varianceMean"] = aggregate(variances)
        report["groups"].append(group)

        rows.append((place, technique, checkpoint, group))

        if args.maps:
            maps_dir = Path(args.maps)
            maps_dir.mkdir(parents=True, exist_ok=True)
            ordered = sorted(entries, key=lambda e: e[0][args.metric])
            picks = {"best": ordered[0], "median": ordered[len(ordered) // 2], "worst": ordered[-1]}
            for label, (scores, error_map) in picks.items():
                stem = Path(scores["path"]).stem
                out = maps_dir / f"{stem}-{label}.flip.png"
                Image.fromarray(magma(error_map)).save(out)

    out_json = Path(args.json) if args.json else run_dir / "report.json"
    # Travels with the data, not only with the printed table: a later reader has to be
    # able to tell which metric build produced a number without trusting a screenshot.
    report["flip"] = flip_settings()

    # 16.7: the metric's own settings, beside the numbers they produced.
    print(f"FLIP {flip_settings()['version']}, LDR, {flip_settings()['ppd']:.2f} pixels per degree")
    header = (f"| state | technique | cp | frames | s | ms/frame | {args.metric} | ci95 | "
              f"w-median | w-Q1 | w-Q3 | median | p95 | MSE |")
    print(header)
    print("|" + "---|" * 14)
    for place, technique, checkpoint, group in rows:
        metric = group["metric"]
        print(f"| {place} | {technique} | {checkpoint} | {group['frames']:.0f} | {group['seconds']:.2f} | "
              f"{group['meanFrameMs']:.2f} | {metric['mean']:.6f} | +-{metric['ci95']:.6f} | "
              f"{group['flipWeightedMedian']['mean']:.6f} | {group['flipWeightedQ1']['mean']:.6f} | "
              f"{group['flipWeightedQ3']['mean']:.6f} | {group['flipMedian']['mean']:.6f} | "
              f"{group['flipP95']['mean']:.6f} | {group['mse']['mean']:.6f} |")

    crossings = report_crossings(report["groups"], "metric", args.metric,
                                 baseline_technique=args.baseline, target=args.target)
    crossings += report_crossings(report["groups"], "varianceMean", "variance",
                                  baseline_technique=args.baseline)
    if crossings:
        report["crossings"] = crossings
    if not args.no_plots:
        report["plots"] = plot_curves(report["groups"], args.metric, run_dir, crossings)
    out_json.write_text(json.dumps(report, indent=2))

    print(f"\nImages scored: {len(images)}   report: {out_json}")


if __name__ == "__main__":
    main()
