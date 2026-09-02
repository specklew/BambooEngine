#!/usr/bin/env python3
"""Run a benchmark matrix and report it, unattended.

One arm = (state, technique). The matrix is run in ROUNDS, every arm once per
round, so the arms interleave in time — the only ordering that survives the GPU's
thermal drift (ADR 0014). Everything else follows ADR 0022: warm up, render M
independent images per arm per round, score each one against a converged
reference, and report the mean with its error bar.

Presets:
  quick   M=5   x1 round, 1 s budget      - working iteration, ~1 min
  point   M=100 x4 rounds, 1 s budget     - table numbers with an error bar
  curve   M=10  x2 rounds, 5 s, 16 cps    - convergence curves and equal-error

Examples:
  python tools/bench.py --scene veach-ajar --states "Standard Look" \
      --techniques "Path Tracing" "Guided Path Tracing (VXPG)" \
      --preset quick --reference-dir Raytracer/SavedUserData/Screenshots/ref-2026-08-19

  python tools/bench.py --scene veach-ajar --states "Standard Look" \
      --techniques "Guided Path Tracing (VXPG)" --preset point --budget frames:1 \
      --reference-dir <dir>
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_report import (add_build_argument, aggregate, load_rgb,  # noqa: E402
                          plot_curves, report_crossings, resolve_exe,
                          score_image, sidecar_for)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAYTRACER_DIR = REPO_ROOT / "Raytracer"
EXE = None

PRESETS = {
    "quick": {"images": 5,   "rounds": 1, "budget": "seconds:1",  "warmup": 4.0, "checkpoints": ""},
    "point": {"images": 100, "rounds": 4, "budget": "seconds:1",  "warmup": 8.0, "checkpoints": ""},
    # STALE as of 2026-08-23, kept only because these presets are what past runs used:
    # time-spaced checkpoints do NOT collapse. Renderer::Update subtracts
    # ConsumeLastCaptureCostSeconds() from the frame delta, so the capture is already
    # outside accumulated time — measured, "--budget seconds:8 --checkpoints log:12"
    # yields twelve distinct captures. A seconds budget is preferable for new work
    # (tools/evaluate.py uses one): every arm then shares one x axis and equal-time is
    # read directly instead of interpolated.
    "curve": {"images": 10,  "rounds": 2, "budget": "frames:512", "warmup": 8.0, "checkpoints": "log:16"},
}


def underscore(value):
    """The headless CLI takes '_' for ' ' so quoting survives every launcher."""
    return value.replace(" ", "_")


def find_reference(reference_dir, state):
    """A state's reference is its long path-traced capture, wherever it was written."""
    candidates = sorted(Path(reference_dir).rglob(f"{state}-Path Tracing.png"))
    if not candidates:
        sys.exit(f"No reference for state '{state}' under {reference_dir}")
    return candidates[-1]


def run_arm(state, technique, levers, out_dir, settings, extra_cvars):
    command = [
        str(EXE), "--headless",
        "--scene", settings["scene"],
        "--states", underscore(state),
        "--techniques", underscore(technique),
        "--warmup", str(settings["warmup"]),
        "--images", str(settings["imagesPerRound"]),
        "--budget", settings["budget"],
        "--out", str(out_dir),
    ]
    if levers is not None:
        # Always stated in full, so a row cannot inherit the previous row's levers.
        # "none" is the baseline; the engine treats it as the empty set.
        command += ["--levers", levers.replace("+", ",") if levers != "none" else "none"]
    if settings.get("config"):
        command += ["--config", settings["config"]]
    if settings.get("statesKey"):
        command += ["--states-key", settings["statesKey"]]
    if settings["checkpoints"]:
        command += ["--checkpoints", settings["checkpoints"]]
    for assignment in extra_cvars:
        command += ["--cvar", assignment]

    started = time.time()
    result = subprocess.run(command, cwd=RAYTRACER_DIR)
    if result.returncode != 0:
        sys.exit(f"Arm failed ({state} / {technique}), exit {result.returncode}")
    return time.time() - started


def main():
    parser = argparse.ArgumentParser(description="Run and report a benchmark matrix.")
    add_build_argument(parser)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--techniques", nargs="+", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--reference-dir", required=True, help="folder holding '<state>-Path Tracing.png'")
    parser.add_argument("--config", help="headless config the arms render with; must be the one the "
                                         "reference was rendered with, or the FLIP compares two grades")
    parser.add_argument("--states-key", help="states.json key when it is not the scene's file stem")
    parser.add_argument("--images", type=int, help="override the preset's images per arm per round")
    parser.add_argument("--rounds", type=int, help="override the preset's round count")
    parser.add_argument("--budget", help="frames:N or seconds:T")
    parser.add_argument("--checkpoints", help="log:K | every:N | list:a,b,c")
    parser.add_argument("--warmup", type=float)
    parser.add_argument("--cvar", action="append", default=[], help="name=value, repeatable")
    parser.add_argument("--levers-matrix",
                        help="comma-separated vendor lever sets, '+' inside a set: 'none,swizzle,noviews+swizzle'")
    parser.add_argument("--metric", default="flipMean")
    parser.add_argument("--target", type=float,
                        help="error level for the equal-error readout (default: the tightest level every arm reaches)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out", help="output root (default SavedUserData/Screenshots/bench-<timestamp>)")
    parser.add_argument("--keep-images", action="store_true",
                        help="keep every capture; by default only the report survives")
    args = parser.parse_args()

    global EXE
    EXE = resolve_exe(getattr(args, "build", None))

    settings = dict(PRESETS[args.preset])
    settings["scene"] = args.scene
    settings["config"] = args.config
    settings["statesKey"] = args.states_key
    for key in ("images", "rounds", "budget", "checkpoints", "warmup"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    # images is the TOTAL per arm; rounds only decide how it is sliced in time.
    rounds = max(1, settings["rounds"])
    settings["imagesPerRound"] = max(1, settings["images"] // rounds)

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out) if args.out else Path("SavedUserData/Screenshots") / f"bench-{stamp}"
    variance_on = any(c.startswith("renderer.accumulation.variance=") for c in args.cvar)
    cvars = args.cvar if variance_on else args.cvar + ["renderer.accumulation.variance=1"]

    print(f"Matrix: {len(args.states)} state(s) x {len(args.techniques)} technique(s) x {rounds} round(s), "
          f"{settings['images']} image(s) per arm ({settings['imagesPerRound']}/round), budget {settings['budget']}")

    # A lever set is a third arm dimension (ADR 0020): "none,swizzle,ser+swizzle"
    # is three arms, and each one restates the whole set.
    lever_sets = args.levers_matrix.split(",") if args.levers_matrix else [None]

    # Arms interleave within a round; a technique measured entirely before another
    # is measured against a different GPU temperature, not a different algorithm.
    # Arm directories are SHORT codes, not descriptive names. Windows still enforces
    # a 260-character path, and the engine's own capture filename already spends ~55
    # of it: a descriptive arm name pushed the guided integrator's captures past the
    # limit, every save failed silently, and the grid reported an empty table after
    # running for half an hour. Identity lives in each capture's sidecar, which is
    # what the scoring pass reads — the directory name only has to be unique.
    for round_index in range(rounds):
        for state_index, state in enumerate(args.states):
            for technique_index, technique in enumerate(args.techniques):
                for lever_index, levers in enumerate(lever_sets):
                    tag = f"s{state_index}t{technique_index}a{lever_index:02d}"
                    out_dir = out_root / f"r{round_index}" / tag
                    seconds = run_arm(state, technique, levers, out_dir, settings, cvars)
                    print(f"  round {round_index}: {state} / {technique}"
                          f"{' / levers ' + levers if levers else ''}  ({seconds:.1f}s)")

    references = {state: load_rgb(find_reference(args.reference_dir, state)) for state in args.states}

    # (state, technique, checkpoint) -> per-image scores, pooled across rounds; the
    # round is kept per image so drift can be read back out of the same data.
    groups = defaultdict(list)
    absolute_root = RAYTRACER_DIR / out_root
    for png in sorted(absolute_root.rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        meta = sidecar_for(png)
        state = meta.get("scene", {}).get("place", "?")
        technique = meta.get("technique", "?")
        bench = meta.get("benchmark", {})
        rt = meta.get("raytracing", {})
        if state not in references:
            continue

        scores, _ = score_image(references[state], png)
        # The engine adds its own run-<timestamp> level under --out, so the round
        # marker is somewhere up the path rather than at a fixed depth.
        round_parts = [p.name for p in png.parents if len(p.name) > 1 and p.name[0] == "r" and p.name[1:].isdigit()]
        scores["round"] = int(round_parts[0][1:]) if round_parts else 0
        scores["frames"] = rt.get("frameIndex", 0)
        scores["seconds"] = rt.get("accumulatedTime", 0.0)
        scores["meanFrameMs"] = bench.get("meanFrameMs", 0.0)
        scores["varianceMean"] = bench.get("varianceMean")
        levers = bench.get("levers", "") or "none"
        groups[(state, technique, levers, bench.get("checkpointIndex", 0))].append(scores)

    report = {"scene": args.scene, "preset": args.preset, "settings": settings,
              "metric": args.metric, "groups": []}
    print()
    print(f"| state | technique | levers | cp | frames | s | ms/frame | {args.metric} | ci95 | median | p95 | MSE | var | drift |")
    print("|" + "---|" * 14)

    baseline = {}
    for (state, technique, levers, checkpoint), scored in sorted(groups.items()):
        values = [s[args.metric] for s in scored]
        summary = aggregate(values)

        # Drift: the same arm measured in different rounds. If this is large the
        # comparison below is describing the room, not the renderer.
        by_round = defaultdict(list)
        for s in scored:
            by_round[s["round"]].append(s[args.metric])
        round_means = [statistics.fmean(v) for v in by_round.values()]
        drift = (max(round_means) - min(round_means)) / summary["mean"] * 100.0 if len(round_means) > 1 else 0.0

        variances = [s["varianceMean"] for s in scored if s.get("varianceMean") is not None]
        entry = {
            "state": state, "technique": technique, "levers": levers, "checkpoint": checkpoint,
            "frames": statistics.fmean([s["frames"] for s in scored]),
            "seconds": statistics.fmean([s["seconds"] for s in scored]),
            "meanFrameMs": statistics.fmean([s["meanFrameMs"] for s in scored]),
            "metric": summary,
            "mse": aggregate([s["mse"] for s in scored]),
            # Two different spreads, and they answer different questions. The
            # aggregates ON each key below are across RUNS (how repeatable the arm
            # is); the keys themselves are quantiles WITHIN an image's error map (how
            # the error is distributed over the picture). Reporting one as the other
            # is the easy mistake, so both are kept.
            "flipQ1": aggregate([s["flipQ1"] for s in scored]),
            "flipMedian": aggregate([s["flipMedian"] for s in scored]),
            "flipQ3": aggregate([s["flipQ3"] for s in scored]),
            "flipP95": aggregate([s["flipP95"] for s in scored]),
            "driftPercent": drift,
        }
        if variances:
            entry["varianceMean"] = aggregate(variances)
        report["groups"].append(entry)
        # The baseline is the FIRST technique and the FIRST lever set on the command
        # line, not the first in sort order — otherwise the ratio silently reports
        # the comparison backwards.
        baseline_levers = (lever_sets[0] or "none") if lever_sets[0] else "none"
        if technique == args.techniques[0] and levers == baseline_levers:
            baseline[(state, checkpoint)] = entry

        variance_text = f"{entry['varianceMean']['mean']:.4f}" if variances else "-"
        print(f"| {state} | {technique} | {levers} | {checkpoint} | {entry['frames']:.0f} | {entry['seconds']:.2f} | "
              f"{entry['meanFrameMs']:.2f} | {summary['mean']:.6f} | +-{summary['ci95']:.6f} | "
              f"{entry['flipMedian']['mean']:.6f} | {entry['flipP95']['mean']:.6f} | "
              f"{entry['mse']['mean']:.6f} | {variance_text} | {drift:.1f}% |")

    # Ratios against the first technique listed, which is what "wins at equal time"
    # actually means once both arms carry an error bar.
    if len(args.techniques) > 1 or len(lever_sets) > 1:
        print()
        print(f"Baseline: {args.techniques[0]}"
              f"{' / levers ' + (lever_sets[0] or 'none') if lever_sets[0] else ''}."
              "  >100% = the arm beats it, <100% = it loses.")
        print("| state | cp | technique | levers | vs baseline (metric) | vs baseline (MSE) | frames |")
        print("|" + "---|" * 7)
        for entry in report["groups"]:
            base = baseline.get((entry["state"], entry["checkpoint"]))
            if base is None or (entry["technique"] == base["technique"] and entry["levers"] == base["levers"]):
                continue
            ratio = base["metric"]["mean"] / entry["metric"]["mean"] * 100.0
            mse_ratio = base["mse"]["mean"] / entry["mse"]["mean"] * 100.0
            frame_ratio = entry["frames"] / base["frames"] * 100.0
            print(f"| {entry['state']} | {entry['checkpoint']} | {entry['technique']} | {entry['levers']} | "
                  f"{ratio:.1f}% | {mse_ratio:.1f}% | {frame_ratio:.1f}% |")
            entry["breakEvenPercent"] = ratio

    # Only meaningful once a run carries checkpoints: a single budget is one point
    # on the curve, and one point cannot be crossed.
    # Crossings and plots key on (state, technique), so a lever matrix has to fold
    # its set into the technique label or all four arms collapse into one series.
    if len(lever_sets) > 1:
        curve_entries = [dict(entry, technique=f"{entry['technique']} [{entry['levers']}]")
                         for entry in report["groups"]]
        curve_baseline = f"{args.techniques[0]} [{lever_sets[0] or 'none'}]"
    else:
        curve_entries = report["groups"]
        curve_baseline = args.techniques[0]

    crossings = report_crossings(curve_entries, "metric", args.metric,
                                 baseline_technique=curve_baseline, target=args.target)
    crossings += report_crossings(curve_entries, "varianceMean", "variance",
                                  baseline_technique=curve_baseline)
    if crossings:
        report["crossings"] = crossings

    # Captures go before the plots are written, so the cleanup cannot eat them.
    if not args.keep_images:
        for png in absolute_root.rglob("*.png"):
            png.unlink()
        print("Captures deleted (--keep-images to keep them)")

    report_path = absolute_root / "bench.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        report["plots"] = plot_curves(curve_entries, args.metric, absolute_root, crossings)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
