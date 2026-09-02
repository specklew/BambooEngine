#!/usr/bin/env python3
"""Convergence and temporal-stability evaluation of one scene across several arms.

An ARM is a technique plus the CVars that configure it, so "VXPG" and "VXPG with
same-frame injection reuse" are two arms of the same technique. Three things get measured, and
they answer three different questions:

  convergence  FLIP against a converged reference at K checkpoints inside ONE
               accumulation, so a single run yields the whole error-vs-time curve.
  stability    how much the SAME arm moves when nothing about it changed. Two
               readings: the estimator's own per-pixel variance (Welford, from the
               accumulation pass) and the spread of FLIP across independent images.
  flicker      per-pixel standard deviation across N independent SINGLE-frame
               images. This is temporal stability in the interactive sense - what
               an unaccumulated viewer would see move between frames.

A time budget is used rather than a frame budget on purpose: the arms then share
one x axis, so equal-time is read directly instead of being interpolated. The
engine already excludes the PNG encode from its accumulated time (Renderer.cpp
"renderElapsed"), so time-spaced checkpoints do not collapse into the start of the
run the way a naive time budget would.

Usage:
  # references (long, once per scene/state - everything else is scored against them)
  python tools/evaluate.py reference --scene veach-ajar --state "Tungsten Default" \
      --config SavedUserData/headless.eval.json --seconds 1800 \
      --arm "PT=Path Tracing" --arm "VXPG=Guided Path Tracing (VXPG)" \
      --out SavedUserData/Screenshots/eval-refs

  # study
  python tools/evaluate.py study --scene veach-ajar --state "Tungsten Default" \
      --config SavedUserData/headless.eval.json \
      --arm "PT=Path Tracing" \
      --arm "VXPG=Guided Path Tracing (VXPG)" \
      --arm "VXPG-reuse=Guided Path Tracing (VXPG):vxpg.injection.reuseInMis=1" \
      --reference-dir SavedUserData/Screenshots/eval-refs --primary-reference PT \
      --seconds 30 --checkpoints log:16 --images 3 --rounds 2 --flicker-images 32

  # temporal stability (M1/M2 consecutive-frame difference, M6 variance homogeneity)
  python tools/evaluate.py temporal --scene veach-ajar --state "Tungsten Default" \
      --config SavedUserData/headless.eval.json --frames 60 --homogeneity-frames 128 \
      --arm "PT=Path Tracing" --arm "VXPG=Guided Path Tracing (VXPG)" \
      --reference-dir SavedUserData/Screenshots/eval-refs --primary-reference PT

  # re-score / re-plot a finished study without re-rendering
  python tools/evaluate.py report --run SavedUserData/Screenshots/eval-<stamp> \
      --reference-dir SavedUserData/Screenshots/eval-refs --primary-reference PT

The temporal pass needs CONSECUTIVE frames, which the harness gives without any special
mode: "--budget frames:N --checkpoints every:1" walks one accumulation and captures every
frame of it, and "--budget frames:1 --images N" renders N consecutive raw frames. Capturing
every frame inflates the reported ms/frame (~50-120% at 1080p); it does not touch the
images, so a frame-indexed metric is unaffected, but frame cost and equal-time readings
must come from a run that captures rarely.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

from bench_report import (add_build_argument, aggregate, find_crossing,  # noqa: E402
                          load_rgb, resolve_exe, score_image, sidecar_for)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAYTRACER_DIR = REPO_ROOT / "Raytracer"
# Bound once in main() from --build / BAMBOO_BUILD; see bench_report.resolve_exe.
EXE = None


# ---------------------------------------------------------------- arms and paths

class Arm:
    """LABEL=Technique Name[:cvar=value,cvar=value]"""

    def __init__(self, spec):
        if "=" not in spec:
            sys.exit(f"--arm expects LABEL=Technique[:cvar=value], got '{spec}'")
        self.label, rest = spec.split("=", 1)
        # The technique name is everything up to the first colon; ':' cannot appear
        # in a registered technique name, and every CVar assignment needs one '='.
        technique, _, cvar_list = rest.partition(":")
        self.technique = technique
        self.cvars = [c for c in cvar_list.split(",") if c]

    def as_dict(self):
        return {"label": self.label, "technique": self.technique, "cvars": self.cvars}


def underscore(value):
    """The headless CLI takes '_' for ' ' so quoting survives every launcher."""
    return value.replace(" ", "_")


def arm_dir(root, index):
    """Short directory codes, because Windows still enforces MAX_PATH and the
    engine's own capture filename already spends ~55 characters of it."""
    return root / f"a{index:02d}"


def newest_capture(directory):
    """The engine writes its own run-<timestamp> level under --out."""
    captures = sorted(p for p in Path(directory).rglob("*.png") if not p.name.endswith(".flip.png"))
    if not captures:
        sys.exit(f"No capture written under {directory}")
    return captures[-1]


def run_engine(scene, state, arm, out_dir, config, budget, checkpoints, images, warmup, extra_cvars):
    command = [
        str(EXE), "--headless",
        "--scene", scene,
        "--states", underscore(state),
        "--techniques", underscore(arm.technique),
        "--budget", budget,
        "--images", str(images),
        "--warmup", str(warmup),
        "--out", str(out_dir),
    ]
    if config:
        command += ["--config", config]
    if checkpoints:
        command += ["--checkpoints", checkpoints]
    for assignment in list(arm.cvars) + list(extra_cvars):
        command += ["--cvar", assignment]

    started = time.time()
    result = subprocess.run(command, cwd=RAYTRACER_DIR)
    if result.returncode != 0:
        sys.exit(f"Arm '{arm.label}' failed, exit {result.returncode}")
    return time.time() - started


# ------------------------------------------------------------------- references

def render_references(args):
    arms = [Arm(spec) for spec in args.arm]
    out_root = Path(args.out)
    absolute_root = RAYTRACER_DIR / out_root
    absolute_root.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        target = absolute_root / f"{arm.label}.png"
        if target.exists() and not args.force:
            print(f"{arm.label}: already rendered ({target.name}) - --force to redo")
            continue
        print(f"{arm.label}: rendering {args.seconds:.0f}s reference ...")
        raw = out_root / f"raw-{arm.label}"
        seconds = run_engine(args.scene, args.state, arm, raw, args.config,
                             f"seconds:{args.seconds}", "", 1, args.warmup, [])
        source = newest_capture(RAYTRACER_DIR / raw)
        target.write_bytes(source.read_bytes())
        (absolute_root / f"{arm.label}.json").write_bytes(source.with_suffix(".json").read_bytes())
        meta = sidecar_for(source).get("raytracing", {})
        print(f"{arm.label}: {meta.get('frameIndex', 0)} frames in "
              f"{meta.get('accumulatedTime', 0.0):.0f}s accumulated "
              f"({seconds / 60.0:.1f} min wall) -> {target}")

    print(f"\nReferences in {absolute_root}")


# ------------------------------------------------------------------------ study

def run_study(args):
    arms = [Arm(spec) for spec in args.arm]
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out) if args.out else Path("SavedUserData/Screenshots") / f"eval-{stamp}"
    absolute_root = RAYTRACER_DIR / out_root
    absolute_root.mkdir(parents=True, exist_ok=True)

    # The Welford variance the accumulation pass keeps is the estimator's own noise
    # level, which is the stability reading that does not need a reference image.
    variance_cvars = ["renderer.accumulation.variance=1"]

    manifest = {
        "scene": args.scene, "state": args.state, "config": args.config,
        "seconds": args.seconds, "checkpoints": args.checkpoints,
        "images": args.images, "rounds": args.rounds, "warmup": args.warmup,
        "flickerImages": args.flicker_images,
        "arms": [arm.as_dict() for arm in arms],
        "started": stamp,
    }
    (absolute_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Study: {len(arms)} arm(s) x {args.rounds} round(s) x {args.images} image(s), "
          f"budget seconds:{args.seconds}, checkpoints {args.checkpoints}")

    # Arms interleave within a round. An arm measured entirely before another is
    # measured against a different GPU temperature, not a different algorithm.
    for round_index in range(args.rounds):
        for index, arm in enumerate(arms):
            out_dir = arm_dir(out_root / "raw" / f"r{round_index}", index)
            seconds = run_engine(args.scene, args.state, arm, out_dir, args.config,
                                 f"seconds:{args.seconds}", args.checkpoints,
                                 args.images, args.warmup, variance_cvars)
            print(f"  round {round_index}: {arm.label}  ({seconds / 60.0:.1f} min)")

    # Flicker is a separate pass because it needs the opposite budget: many
    # independent ONE-frame images rather than one long accumulation.
    if args.flicker_images > 0:
        for index, arm in enumerate(arms):
            out_dir = arm_dir(out_root / "flicker", index)
            seconds = run_engine(args.scene, args.state, arm, out_dir, args.config,
                                 "frames:1", "", args.flicker_images, args.warmup, [])
            print(f"  flicker: {arm.label}  ({seconds:.0f}s)")

    return out_root




def run_temporal(args):
    """Three passes per arm; see the table in the module docstring for what each is for."""
    arms = [Arm(spec) for spec in args.arm]
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out) if args.out else Path("SavedUserData/Screenshots") / f"temporal-{stamp}"
    absolute_root = RAYTRACER_DIR / out_root
    absolute_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene": args.scene, "state": args.state, "config": args.config,
        "frames": args.frames, "homogeneityFrames": args.homogeneity_frames,
        "warmup": args.warmup, "arms": [arm.as_dict() for arm in arms], "started": stamp,
        "kind": "temporal",
    }
    (absolute_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Temporal: {len(arms)} arm(s), {args.frames}-frame sequences, "
          f"{args.homogeneity_frames} independent frames")

    for index, arm in enumerate(arms):
        # Sub-pixel jitter OFF for the two sequences. SSPG renders "the static scene
        # with disabled jittering of pixel centers" for exactly this reason: with it
        # on, a frame-to-frame difference metric mostly reports the jitter. Only the
        # guided integrator jitters (vxpg.vbufferJitter), so leaving it on would also
        # make the arms incomparable.
        no_jitter = ["vxpg.vbufferJitter=0"]

        seconds = run_engine(args.scene, args.state, arm, arm_dir(out_root / "accumulated", index),
                             args.config, f"frames:{args.frames}", "every:1", 1, args.warmup, no_jitter)
        print(f"  {arm.label}: accumulated sequence ({seconds:.0f}s)")

        seconds = run_engine(args.scene, args.state, arm, arm_dir(out_root / "raw", index),
                             args.config, "frames:1", "", args.frames, args.warmup, no_jitter)
        print(f"  {arm.label}: raw sequence ({seconds:.0f}s)")

        # Jitter stays ON here: these frames are scored against a reference rendered
        # with it, and a silhouette mismatch would land in every frame equally - a
        # constant that deflates exactly the relative spread M6 is asking about.
        seconds = run_engine(args.scene, args.state, arm, arm_dir(out_root / "homogeneity", index),
                             args.config, "frames:1", "", args.homogeneity_frames, args.warmup, [])
        print(f"  {arm.label}: {args.homogeneity_frames} independent frames ({seconds:.0f}s)")

    return out_root


def score_temporal(run_root, reference_dir, primary):
    root = Path(run_root)
    absolute_root = root if root.is_absolute() else RAYTRACER_DIR / root
    manifest = json.loads((absolute_root / "manifest.json").read_text())

    reference_root = Path(reference_dir)
    if not reference_root.is_absolute():
        reference_root = RAYTRACER_DIR / reference_root
    reference_path = reference_root / f"{primary}.png"
    if not reference_path.exists():
        sys.exit(f"No reference '{primary}.png' in {reference_root}")
    reference = load_rgb(reference_path)

    report = {"manifest": manifest, "primaryReference": primary, "arms": []}
    for index, arm in enumerate(manifest["arms"]):
        accumulated = sequence_metrics(arm_dir(absolute_root / "accumulated", index))
        raw = sequence_metrics(arm_dir(absolute_root / "raw", index))
        frames = homogeneity_metrics(arm_dir(absolute_root / "homogeneity", index), reference)

        report["arms"].append({
            "label": arm["label"], "technique": arm["technique"], "cvars": arm["cvars"],
            "accumulated": accumulated,
            "raw": raw,
            "homogeneityFrames": frames,
            "homogeneity": {
                "rmae": spread([f["rmae"] for f in frames]),
                "flipMean": spread([f["flipMean"] for f in frames]),
            },
        })
    return report, absolute_root


# ---------------------------------------------------------------------- scoring

def arm_index_from_path(path):
    for parent in path.parents:
        name = parent.name
        if len(name) == 3 and name[0] == "a" and name[1:].isdigit():
            return int(name[1:])
    return None


def round_index_from_path(path):
    for parent in path.parents:
        name = parent.name
        if len(name) > 1 and name[0] == "r" and name[1:].isdigit():
            return int(name[1:])
    return 0


def flicker_stats(directory, reference=None):
    """Per-pixel standard deviation across independent single-frame images.

    Streamed with Welford rather than stacking the images: 32 frames of 1080p RGB
    is 800 MB as one array and 50 MB as a running mean plus M2.

    The deviation alone is NOT a stability score, and reporting it alone inverts the
    result. These images are tonemapped and CLAMPED, so an estimator whose single
    sample usually misses the light writes the same black pixel every frame: perfect
    stability, entirely wrong. Hence staticFraction (pixels that never moved) and
    meanLevel against the reference's own level - a low deviation is only good news
    when those two say the pixels were alive in the first place.
    """
    images = sorted(p for p in Path(directory).rglob("*.png") if not p.name.endswith(".flip.png"))
    if len(images) < 2:
        return None

    count = 0
    mean = None
    m2 = None
    for path in images:
        sample = load_rgb(path)
        count += 1
        if mean is None:
            mean = np.zeros_like(sample)
            m2 = np.zeros_like(sample)
        delta = sample - mean
        mean += delta / count
        m2 += delta * (sample - mean)

    stdev = np.sqrt(m2 / max(count - 1, 1))
    per_pixel = stdev.mean(axis=2)  # average the channels, keep the pixel layout
    stats = {
        "images": count,
        "stdMean": float(per_pixel.mean()),
        "stdMedian": float(np.median(per_pixel)),
        "stdP95": float(np.quantile(per_pixel, 0.95)),
        "stdMax": float(per_pixel.max()),
        # Relative flicker: the same deviation is far more visible on a dark wall
        # than inside a blown-out doorway, so it is also reported against the mean.
        "relativeStdMean": float((per_pixel / np.maximum(mean.mean(axis=2), 1e-3)).mean()),
        "staticFraction": float((per_pixel < 1e-6).mean()),
        "meanLevel": float(mean.mean()),
    }

    if reference is not None:
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        reference_luminance = reference @ weights
        # Pixels the reference says carry light AND are not pinned at the top of the
        # display range: the only ones where a deviation is both meaningful and visible.
        midtones = (reference_luminance > 0.02) & (reference_luminance < 0.98)
        if midtones.any():
            stats["stdMidtones"] = float(per_pixel[midtones].mean())
        stats["referenceLevel"] = float(reference.mean())

    return stats


def score_run(run_root, reference_dir, primary, metric):
    root = Path(run_root)
    absolute_root = root if root.is_absolute() else RAYTRACER_DIR / root
    manifest = json.loads((absolute_root / "manifest.json").read_text())
    arms = manifest["arms"]

    reference_root = Path(reference_dir)
    if not reference_root.is_absolute():
        reference_root = RAYTRACER_DIR / reference_root
    references = {p.stem: p for p in reference_root.glob("*.png")}
    if primary not in references:
        sys.exit(f"No reference '{primary}.png' in {reference_root} (have: {sorted(references)})")
    primary_image = load_rgb(references[primary])

    report = {"manifest": manifest, "metric": metric, "primaryReference": primary,
              "referenceCrossCheck": [], "groups": [], "flicker": []}

    # How far apart the references are IS the systematic-difference number: two
    # unbiased estimators of the same integral converge to the same image, so
    # whatever separates them after half an hour is not noise.
    for label, path in sorted(references.items()):
        if label == primary:
            continue
        scores, _ = score_image(primary_image, path)
        scores["reference"] = label
        report["referenceCrossCheck"].append(scores)

    groups = defaultdict(list)
    for png in sorted((absolute_root / "raw").rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        index = arm_index_from_path(png)
        if index is None or index >= len(arms):
            continue
        meta = sidecar_for(png)
        bench = meta.get("benchmark", {})
        raytracing = meta.get("raytracing", {})
        scores, _ = score_image(primary_image, png)
        scores["round"] = round_index_from_path(png)
        scores["frames"] = raytracing.get("frameIndex", 0)
        scores["seconds"] = raytracing.get("accumulatedTime", 0.0)
        scores["meanFrameMs"] = bench.get("meanFrameMs", 0.0)
        scores["varianceMean"] = bench.get("varianceMean")
        scores["varianceRelative"] = bench.get("varianceRelative")
        groups[(index, bench.get("checkpointIndex", 0))].append(scores)

    for (index, checkpoint), scored in sorted(groups.items()):
        values = [s[metric] for s in scored]
        summary = aggregate(values)
        variances = [s["varianceMean"] for s in scored if s.get("varianceMean") is not None]
        relatives = [s["varianceRelative"] for s in scored if s.get("varianceRelative") is not None]

        by_round = defaultdict(list)
        for s in scored:
            by_round[s["round"]].append(s[metric])
        round_means = [statistics.fmean(v) for v in by_round.values()]
        drift = (max(round_means) - min(round_means)) / summary["mean"] * 100.0 if len(round_means) > 1 else 0.0

        entry = {
            "arm": arms[index]["label"], "technique": arms[index]["technique"],
            "cvars": arms[index]["cvars"], "checkpoint": checkpoint,
            "frames": statistics.fmean([s["frames"] for s in scored]),
            "seconds": statistics.fmean([s["seconds"] for s in scored]),
            "meanFrameMs": aggregate([s["meanFrameMs"] for s in scored]),
            "metric": summary,
            "mse": aggregate([s["mse"] for s in scored]),
            "flipMedian": aggregate([s["flipMedian"] for s in scored]),
            "flipP95": aggregate([s["flipP95"] for s in scored]),
            # Run-to-run stability: how much the SAME arm at the SAME budget moves
            # between independent images. Reported relative, so arms at different
            # error levels stay comparable.
            "repeatabilityPercent": (summary["stdev"] / summary["mean"] * 100.0) if summary["mean"] else 0.0,
            "driftPercent": drift,
        }
        if variances:
            entry["varianceMean"] = aggregate(variances)
        if relatives:
            entry["varianceRelative"] = aggregate(relatives)
        report["groups"].append(entry)

    flicker_root = absolute_root / "flicker"
    if flicker_root.exists():
        for index, arm in enumerate(arms):
            stats = flicker_stats(arm_dir(flicker_root, index), primary_image)
            if stats:
                stats["arm"] = arm["label"]
                report["flicker"].append(stats)

    return report, absolute_root



# ------------------------------------------------------- temporal stability

LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def ordered_captures(directory):
    """Captures in the order they were rendered.

    Both sequence recipes make the filename sort agree with time: an --images run
    numbers them -i0000, -i0001, ..., and a checkpointed one appends -f<frameIndex>.
    """
    return sorted(p for p in Path(directory).rglob("*.png") if not p.name.endswith(".flip.png"))


def sequence_metrics(directory):
    """Temporal error and temporal MSE over consecutive frames.

    M1 is SVGF's definition - "the average luminance of the difference between
    consecutive frames" (Schied et al. 2017, Sec. 5.4), read as the mean over pixels
    of the luminance of the ABSOLUTE difference, since a signed difference averages
    to zero over a converging sequence and would report stability that is not there.
    M2 is SSPG's "MSE between every consequent frame" (Derevyannykh 2022, Sec. 4.1).

    Frames are streamed one at a time: 60 frames of 1080p RGB is 1.5 GB stacked and
    two frames held at once.
    """
    captures = ordered_captures(directory)
    if len(captures) < 2:
        return []

    rows = []
    previous = load_rgb(captures[0])
    for index, path in enumerate(captures[1:], start=1):
        current = load_rgb(path)
        difference = np.abs(current - previous)
        rows.append({
            "pair": index,
            "temporalError": float((difference @ LUMINANCE).mean()),
            "temporalMse": float((difference ** 2).mean()),
        })
        previous = current
    return rows


def relative_mean_absolute_error(reference, image, floor=0.01):
    """ReSTIR's error metric (Bitterli et al. 2020, Sec. 7): "Relative Mean Absolute
    Error (RMAE), which we found less sensitive to isolated outliers than mean squared
    error". The floor keeps near-black pixels from dominating the ratio."""
    return float((np.abs(image - reference) / (reference + floor)).mean())


def homogeneity_metrics(directory, reference, limit=None):
    """Per-frame error over independent single-frame renders.

    The spread of these - not their mean - is M6: the paper says "different voxel
    clustering can lead to different amounts of variance across frames, especially in
    scenes with complex visibility" (Lu et al. 2024, Sec. 7) and never measures it.
    Path tracing is the control: it has no clustering, so its spread is what Monte
    Carlo noise alone produces at this budget.
    """
    captures = ordered_captures(directory)
    if limit is not None:
        captures = captures[:limit]
    if len(captures) < 2:
        return []

    rows = []
    for index, path in enumerate(captures):
        scores, _ = score_image(reference, path)
        rows.append({
            "frame": index,
            "rmae": relative_mean_absolute_error(reference, load_rgb(path)),
            "flipMean": scores["flipMean"],
            "mse": scores["mse"],
        })
    return rows


def spread(values):
    """Mean, standard deviation and the coefficient of variation that is the M6 answer."""
    if len(values) < 2:
        return {}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    return {
        "n": len(values),
        "mean": mean,
        "stdev": stdev,
        "cvPercent": (stdev / mean * 100.0) if mean else 0.0,
        "min": min(values),
        "max": max(values),
        "spreadPercent": ((max(values) - min(values)) / mean * 100.0) if mean else 0.0,
    }


# --------------------------------------------------------------------- readouts

def series(report, key, budget_key="seconds"):
    """arm -> [(budget, value)] ascending."""
    out = defaultdict(list)
    for entry in report["groups"]:
        value = entry.get(key)
        if isinstance(value, dict):
            value = value.get("mean")
        if value is None or not entry.get(budget_key):
            continue
        out[entry["arm"]].append((entry[budget_key], value))
    for arm in out:
        out[arm].sort()
    return out


def equal_error(report, baseline, target=None):
    """What budget each arm needs to reach the same error level. An arm whose curve
    plateaus above the target has not reached it - said so, never extrapolated,
    because the plateau IS the result."""
    by_seconds = series(report, "metric")
    by_frames = series(report, "metric", "frames")
    if not by_seconds:
        return []

    level = target if target is not None else max(min(v for _, v in pts) for pts in by_seconds.values())
    baseline_seconds = find_crossing(by_seconds.get(baseline, []), level) if baseline else None

    rows = []
    for arm in sorted(by_seconds):
        seconds = find_crossing(by_seconds[arm], level)
        frames = find_crossing(by_frames.get(arm, []), level)
        rows.append({
            "arm": arm, "target": level, "seconds": seconds, "frames": frames,
            "speedupVsBaseline": (baseline_seconds / seconds) if (baseline_seconds and seconds) else None,
        })
    return rows


def equal_time(report, baseline):
    """FLIP ratio against the baseline at every checkpoint. >100% = the arm is
    ahead of the baseline at that moment of the accumulation."""
    baseline_points = {e["checkpoint"]: e for e in report["groups"] if e["arm"] == baseline}
    rows = []
    for entry in report["groups"]:
        base = baseline_points.get(entry["checkpoint"])
        if base is None or entry["arm"] == baseline:
            continue
        rows.append({
            "arm": entry["arm"], "checkpoint": entry["checkpoint"], "seconds": entry["seconds"],
            "flipRatioPercent": base["metric"]["mean"] / entry["metric"]["mean"] * 100.0,
            "mseRatioPercent": base["mse"]["mean"] / entry["mse"]["mean"] * 100.0,
            "frameRatioPercent": entry["frames"] / base["frames"] * 100.0 if base["frames"] else 0.0,
        })
    return rows


def attach_equal_time_flicker(report):
    """Flicker is measured at ONE frame per image, which is equal SAMPLES, not equal
    time - and the arms here differ by 4x in frame cost. Averaging n independent
    frames divides the standard deviation by sqrt(n), so the deviation an arm would
    still show after spending one second is std * sqrt(ms / 1000). That is the
    number an interactive comparison actually wants."""
    frame_ms = {}
    for entry in report["groups"]:
        frame_ms[entry["arm"]] = entry["meanFrameMs"]["mean"]
    for row in report["flicker"]:
        ms = frame_ms.get(row["arm"])
        if ms:
            row["stdAfterOneSecond"] = row["stdMean"] * math.sqrt(ms / 1000.0)
            row["frameMs"] = ms


# ----------------------------------------------------------------------- output

def write_tables(report, crossings, ratios, metric, out_path):
    lines = []
    add = lines.append
    manifest = report["manifest"]

    add(f"# Evaluation - {manifest['scene']} / {manifest['state']}")
    add("")
    add(f"Budget `seconds:{manifest['seconds']}`, checkpoints `{manifest['checkpoints']}`, "
        f"{manifest['images']} image(s) x {manifest['rounds']} round(s), warm-up {manifest['warmup']}s. "
        f"Scored against the **{report['primaryReference']}** reference.")
    add("")
    add("| arm | technique | cvars |")
    add("|---|---|---|")
    for arm in manifest["arms"]:
        add(f"| {arm['label']} | {arm['technique']} | {', '.join(arm['cvars']) or '-'} |")
    add("")

    if report["referenceCrossCheck"]:
        add("## Reference cross-check")
        add("")
        add("Two converged references of the same integral. What separates them is systematic, not noise.")
        add("")
        add(f"| reference | FLIP vs {report['primaryReference']} | median | p95 | MSE | RMSE |")
        add("|---|---|---|---|---|---|")
        for row in report["referenceCrossCheck"]:
            add(f"| {row['reference']} | {row['flipMean']:.6f} | {row['flipMedian']:.6f} | "
                f"{row['flipP95']:.6f} | {row['mse']:.6f} | {row['rmse']:.6f} |")
        add("")

    add("## Convergence")
    add("")
    add("`est. var` is the variance of the accumulated estimate in LINEAR HDR; `rel. var` divides it by the "
        "pixel's squared mean. FLIP and MSE are read off the TONEMAPPED image, so a blown-out region that "
        "dominates the HDR variance contributes nothing to them - the two can and do disagree.")
    add("")
    add(f"| arm | cp | s | frames | ms/frame | {metric} | ci95 | median | p95 | MSE | est. var | rel. var | "
        "repeatability | drift |")
    add("|" + "---|" * 14)
    for entry in report["groups"]:
        variance = f"{entry['varianceMean']['mean']:.3e}" if "varianceMean" in entry else "-"
        relative = f"{entry['varianceRelative']['mean']:.3e}" if "varianceRelative" in entry else "-"
        add(f"| {entry['arm']} | {entry['checkpoint']} | {entry['seconds']:.3f} | {entry['frames']:.0f} | "
            f"{entry['meanFrameMs']['mean']:.3f} | {entry['metric']['mean']:.6f} | "
            f"+-{entry['metric']['ci95']:.6f} | {entry['flipMedian']['mean']:.6f} | "
            f"{entry['flipP95']['mean']:.6f} | {entry['mse']['mean']:.6f} | {variance} | {relative} | "
            f"{entry['repeatabilityPercent']:.1f}% | {entry['driftPercent']:.1f}% |")
    add("")

    if ratios:
        add("## Equal time")
        add("")
        add(f"Ratio against **{report['baseline']}** at each checkpoint. "
            ">100% = ahead of it at that moment.")
        add("")
        add("| arm | cp | s | FLIP vs baseline | MSE vs baseline | frames vs baseline |")
        add("|" + "---|" * 6)
        for row in ratios:
            add(f"| {row['arm']} | {row['checkpoint']} | {row['seconds']:.3f} | "
                f"{row['flipRatioPercent']:.1f}% | {row['mseRatioPercent']:.1f}% | "
                f"{row['frameRatioPercent']:.1f}% |")
        add("")

    if crossings:
        add("## Equal error")
        add("")
        add(f"Budget each arm needs to reach FLIP {crossings[0]['target']:.6f} "
            "(the tightest level every arm actually reaches).")
        add("")
        add("| arm | seconds | frames | speedup vs baseline |")
        add("|" + "---|" * 4)
        for row in crossings:
            seconds = f"{row['seconds']:.3f}" if row["seconds"] else "not reached"
            frames = f"{row['frames']:.0f}" if row["frames"] else "-"
            speedup = f"{row['speedupVsBaseline']:.2f}x" if row["speedupVsBaseline"] else "-"
            add(f"| {row['arm']} | {seconds} | {frames} | {speedup} |")
        add("")

    if report["flicker"]:
        add("## Temporal stability (flicker, 1 spp, no accumulation)")
        add("")
        add("Per-pixel standard deviation across independent single-frame images, in display units.")
        add("")
        add("`std after 1 s` rescales it to equal TIME: averaging n frames divides the deviation by "
            "sqrt(n), so an arm that costs `ms` per frame still shows `std * sqrt(ms/1000)` after a "
            "second of averaging. The raw columns are equal SAMPLES, which is not the same comparison.")
        add("")
        add("**Read `static` and `mean level` before reading `std`.** These images are clamped, so an "
            "estimator whose single sample usually misses the light writes the same black pixel every "
            "frame - zero deviation, entirely wrong. `static` is the fraction of pixels that never moved "
            "across the whole set; `mean level` is the set's average against the reference's own.")
        add("")
        add("| arm | images | ms/frame | std mean | std median | std midtones | std p95 | static | "
            "mean level | ref level | std after 1 s |")
        add("|" + "---|" * 11)
        for row in report["flicker"]:
            rescaled = f"{row['stdAfterOneSecond']:.5f}" if "stdAfterOneSecond" in row else "-"
            frame_ms = f"{row['frameMs']:.3f}" if "frameMs" in row else "-"
            midtones = f"{row['stdMidtones']:.5f}" if "stdMidtones" in row else "-"
            reference_level = f"{row['referenceLevel']:.4f}" if "referenceLevel" in row else "-"
            add(f"| {row['arm']} | {row['images']} | {frame_ms} | {row['stdMean']:.5f} | "
                f"{row['stdMedian']:.5f} | {midtones} | {row['stdP95']:.5f} | "
                f"{row['staticFraction'] * 100.0:.1f}% | {row['meanLevel']:.4f} | {reference_level} | "
                f"{rescaled} |")
        add("")

    add("## Frame cost")
    add("")
    add("| arm | ms/frame | ci95 across images | frames in the full budget |")
    add("|" + "---|" * 4)
    last = {}
    for entry in report["groups"]:
        last[entry["arm"]] = entry
    for arm, entry in last.items():
        add(f"| {arm} | {entry['meanFrameMs']['mean']:.3f} | "
            f"+-{entry['meanFrameMs']['ci95']:.3f} | {entry['frames']:.0f} |")
    add("")

    text = "\n".join(lines)
    Path(out_path).write_text(text, encoding="utf-8")
    return text


def plot(report, metric, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (pip install -r tools/requirements.txt)")
        return []

    arms = []
    for entry in report["groups"]:
        if entry["arm"] not in arms:
            arms.append(entry["arm"])

    def points(arm):
        return sorted((e for e in report["groups"] if e["arm"] == arm), key=lambda e: e["seconds"])

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    axes = axes.ravel()

    for arm in arms:
        pts = points(arm)
        if len(pts) < 2:
            continue
        seconds = [p["seconds"] for p in pts]
        frames = [p["frames"] for p in pts]
        values = [p["metric"]["mean"] for p in pts]
        errors = [p["metric"]["ci95"] for p in pts]

        axes[0].plot(seconds, values, marker="o", markersize=3, label=arm)
        axes[0].fill_between(seconds, [v - e for v, e in zip(values, errors)],
                             [v + e for v, e in zip(values, errors)], alpha=0.2)
        axes[1].plot(frames, values, marker="o", markersize=3, label=arm)

        # Relative rather than absolute: the absolute variance is linear HDR and is
        # dominated by whichever region is brightest, which on this scene is the
        # blown-out doorway that the tonemapped metrics cannot even see.
        measured = [p for p in pts if "varianceRelative" in p]
        if measured:
            axes[2].plot([p["seconds"] for p in measured],
                         [p["varianceRelative"]["mean"] for p in measured],
                         marker="o", markersize=3, label=arm)

        axes[3].plot(seconds, [p["repeatabilityPercent"] for p in pts], marker="o", markersize=3, label=arm)

    baseline = report.get("baseline")
    for arm in arms:
        if arm == baseline:
            continue
        rows = [r for r in report.get("equalTime", []) if r["arm"] == arm]
        if len(rows) >= 2:
            axes[4].plot([r["seconds"] for r in rows], [r["flipRatioPercent"] for r in rows],
                         marker="o", markersize=3, label=f"{arm} vs {baseline}")
    if baseline:
        axes[4].axhline(100.0, color="grey", linewidth=0.8, linestyle="--")

    if report["flicker"]:
        labels = [row["arm"] for row in report["flicker"]]
        positions = np.arange(len(labels))
        axes[5].bar(positions - 0.25, [row["stdMean"] for row in report["flicker"]],
                    width=0.25, label="std, 1 spp")
        axes[5].bar(positions, [row["stdP95"] for row in report["flicker"]],
                    width=0.25, label="p95, 1 spp")
        axes[5].bar(positions + 0.25, [row.get("stdAfterOneSecond", 0.0) for row in report["flicker"]],
                    width=0.25, label="std after 1 s")
        axes[5].set_xticks(positions)
        axes[5].set_xticklabels(labels, fontsize=8)
        axes[5].set_ylabel("per-pixel std (display units)")
        axes[5].set_title("flicker at 1 spp, no accumulation", fontsize=10)
        axes[5].grid(True, axis="y", alpha=0.25, linewidth=0.5)
        # The static fraction is drawn on the same panel because it is the reason a
        # low bar may be bad news, and a reader who sees only the bars will misread it.
        static_axis = axes[5].twinx()
        static_axis.plot(positions, [row["staticFraction"] * 100.0 for row in report["flicker"]],
                         color="black", marker="s", markersize=5, linestyle="--",
                         label="pixels that never moved")
        static_axis.set_ylabel("pixels that never moved (%)")
        static_axis.set_ylim(0, 100)
        handles, names = axes[5].get_legend_handles_labels()
        extra_handles, extra_names = static_axis.get_legend_handles_labels()
        axes[5].legend(handles + extra_handles, names + extra_names, fontsize=7)

    settings = [
        (f"{metric} vs accumulated time", "seconds", metric, "log", "log"),
        (f"{metric} vs samples", "frames", metric, "log", "log"),
        ("relative estimator variance vs time", "seconds", "var(mean) / mean^2", "log", "log"),
        ("run-to-run spread vs time", "seconds", "stdev / mean (%)", "log", "linear"),
        (f"equal-time {metric} ratio vs baseline", "seconds", "baseline / arm (%)", "log", "linear"),
    ]
    for index, (title, xlabel, ylabel, xscale, yscale) in enumerate(settings):
        axes[index].set_title(title, fontsize=10)
        axes[index].set_xlabel(xlabel)
        axes[index].set_ylabel(ylabel)
        axes[index].set_xscale(xscale)
        axes[index].set_yscale(yscale)
        axes[index].grid(True, which="both", alpha=0.25, linewidth=0.5)
        axes[index].legend(fontsize=8)

    manifest = report["manifest"]
    figure.suptitle(f"{manifest['scene']} / {manifest['state']} - scored against {report['primaryReference']}")
    figure.tight_layout()

    written = []
    stem = Path(out_dir) / "evaluation"
    for suffix in (".png", ".pdf"):
        figure.savefig(stem.with_suffix(suffix), dpi=140)
        written.append(str(stem.with_suffix(suffix)))
    plt.close(figure)
    return written


def build_report(args, run_root):
    report, absolute_root = score_run(run_root, args.reference_dir, args.primary_reference, args.metric)
    baseline = args.baseline or report["manifest"]["arms"][0]["label"]
    report["baseline"] = baseline
    attach_equal_time_flicker(report)
    report["equalTime"] = equal_time(report, baseline)
    report["equalError"] = equal_error(report, baseline, args.target)
    report["plots"] = plot(report, args.metric, absolute_root)

    text = write_tables(report, report["equalError"], report["equalTime"], args.metric,
                        absolute_root / "report.md")
    (absolute_root / "report.json").write_text(json.dumps(report, indent=2))
    print()
    print(text)
    print(f"Report: {absolute_root / 'report.md'}")
    if report["plots"]:
        print("Plots: " + ", ".join(p for p in report["plots"] if p.endswith(".png")))


def write_temporal_tables(report, out_path):
    lines = []
    add = lines.append
    manifest = report["manifest"]

    add(f"# Temporal stability - {manifest['scene']} / {manifest['state']}")
    add("")
    add(f"{manifest['frames']}-frame sequences, {manifest['homogeneityFrames']} independent frames "
        f"per arm, warm-up {manifest['warmup']}s. Static scene, camera and lights throughout.")
    add("")
    add("| arm | technique | cvars |")
    add("|---|---|---|")
    for arm in manifest["arms"]:
        add(f"| {arm['label']} | {arm['technique']} | {', '.join(arm['cvars']) or '-'} |")
    add("")

    add("## M1 / M2 - consecutive-frame difference")
    add("")
    add("**M1** is SVGF's temporal error: the mean luminance of |I_t - I_t+1|. **M2** is SSPG's "
        "temporal MSE over the same pairs. Sub-pixel jitter is off for both, or the metric would "
        "mostly report the jitter - and only the guided integrator jitters, so leaving it on would "
        "make the arms incomparable.")
    add("")
    add("`accumulated` is the running average a viewer sees, so its temporal error DECAYS as history "
        "builds - that decay is the result. `raw` is 1 spp with no history: a flat line whose height "
        "is the flicker amplitude.")
    add("")
    add("| arm | pass | M1 first pair | M1 last pair | M1 median | decay (first/last) |")
    add("|" + "---|" * 6)
    for arm in report["arms"]:
        for kind in ("accumulated", "raw"):
            rows = arm[kind]
            if len(rows) < 2:
                continue
            errors = [r["temporalError"] for r in rows]
            decay = errors[0] / errors[-1] if errors[-1] > 0 else float("inf")
            add(f"| {arm['label']} | {kind} | {errors[0]:.6f} | {errors[-1]:.6f} | "
                f"{statistics.median(errors):.6f} | {decay:.1f}x |")
    add("")
    add("| arm | pass | M2 first pair | M2 last pair | M2 median |")
    add("|" + "---|" * 5)
    for arm in report["arms"]:
        for kind in ("accumulated", "raw"):
            rows = arm[kind]
            if len(rows) < 2:
                continue
            values = [r["temporalMse"] for r in rows]
            add(f"| {arm['label']} | {kind} | {values[0]:.6e} | {values[-1]:.6e} | "
                f"{statistics.median(values):.6e} |")
    add("")

    add("## M6 - is the variance the same every frame?")
    add("")
    add(f"Per-frame error over {manifest['homogeneityFrames']} INDEPENDENT single-frame renders, "
        f"scored against the **{report['primaryReference']}** reference. The headline is not the mean "
        "but the **coefficient of variation**: how much the error level itself moves from frame to "
        "frame. Path tracing is the control - it has no clustering, so its CV is what Monte Carlo "
        "noise alone produces at this budget. Anything above that is the clustering lottery the paper "
        "describes in Sec. 7 and never measures.")
    add("")
    add("RMAE is ReSTIR's metric, chosen there for being less sensitive to isolated outliers than MSE.")
    add("")
    add("| arm | RMAE mean | RMAE CV | RMAE min-max | FLIP mean | FLIP CV | FLIP min-max |")
    add("|" + "---|" * 7)
    for arm in report["arms"]:
        rmae = arm["homogeneity"]["rmae"]
        flip = arm["homogeneity"]["flipMean"]
        if not rmae or not flip:
            continue
        add(f"| {arm['label']} | {rmae['mean']:.5f} | {rmae['cvPercent']:.2f}% | "
            f"{rmae['spreadPercent']:.1f}% | {flip['mean']:.5f} | {flip['cvPercent']:.2f}% | "
            f"{flip['spreadPercent']:.1f}% |")
    add("")

    rendered = "\n".join(lines)
    Path(out_path).write_text(rendered, encoding="utf-8")
    return rendered


def plot_temporal(report, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return []

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    axes = axes.ravel()

    for arm in report["arms"]:
        for index, kind in enumerate(("accumulated", "raw")):
            rows = arm[kind]
            if len(rows) < 2:
                continue
            axes[index].plot([r["pair"] for r in rows], [r["temporalError"] for r in rows],
                             marker="o", markersize=2.5, label=arm["label"])

        frames = arm["homogeneityFrames"]
        if frames:
            axes[2].plot([f["frame"] for f in frames], [f["rmae"] for f in frames],
                         marker="o", markersize=2.5, linewidth=0.8, label=arm["label"])

    labels = [arm["label"] for arm in report["arms"]]
    positions = np.arange(len(labels))
    axes[3].bar(positions - 0.2,
                [arm["homogeneity"]["rmae"].get("cvPercent", 0.0) for arm in report["arms"]],
                width=0.4, label="RMAE")
    axes[3].bar(positions + 0.2,
                [arm["homogeneity"]["flipMean"].get("cvPercent", 0.0) for arm in report["arms"]],
                width=0.4, label="FLIP")
    axes[3].set_xticks(positions)
    axes[3].set_xticklabels(labels, fontsize=8)
    axes[3].set_ylabel("coefficient of variation (%)")
    axes[3].set_title("M6: spread of the per-frame error level", fontsize=10)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, axis="y", alpha=0.25, linewidth=0.5)

    settings = [
        ("M1: temporal error, accumulated output", "frame pair", "mean luminance of |I_t - I_t+1|", "log"),
        ("M1: temporal error, raw 1 spp", "frame pair", "mean luminance of |I_t - I_t+1|", "linear"),
        ("M6: per-frame RMAE over independent frames", "frame", "RMAE vs reference", "linear"),
    ]
    for index, (title, xlabel, ylabel, yscale) in enumerate(settings):
        axes[index].set_title(title, fontsize=10)
        axes[index].set_xlabel(xlabel)
        axes[index].set_ylabel(ylabel)
        axes[index].set_yscale(yscale)
        axes[index].grid(True, which="both", alpha=0.25, linewidth=0.5)
        axes[index].legend(fontsize=8)

    manifest = report["manifest"]
    figure.suptitle(f"{manifest['scene']} / {manifest['state']} - temporal stability")
    figure.tight_layout()

    written = []
    stem = Path(out_dir) / "temporal"
    for suffix in (".png", ".pdf"):
        figure.savefig(stem.with_suffix(suffix), dpi=140)
        written.append(str(stem.with_suffix(suffix)))
    plt.close(figure)
    return written


def build_temporal_report(args, run_root):
    report, absolute_root = score_temporal(run_root, args.reference_dir, args.primary_reference)
    report["plots"] = plot_temporal(report, absolute_root)
    rendered = write_temporal_tables(report, absolute_root / "temporal.md")
    (absolute_root / "temporal.json").write_text(json.dumps(report, indent=2))
    print()
    print(rendered)
    print(f"Report: {absolute_root / 'temporal.md'}")
    if report["plots"]:
        print("Plots: " + ", ".join(p for p in report["plots"] if p.endswith(".png")))


# -------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_build_argument(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    reference = sub.add_parser("reference", help="render one long converged reference per arm")
    reference.add_argument("--scene", required=True)
    reference.add_argument("--state", required=True)
    reference.add_argument("--config", default="")
    reference.add_argument("--arm", action="append", required=True, help="LABEL=Technique[:cvar=value]")
    reference.add_argument("--seconds", type=float, default=1800.0)
    reference.add_argument("--warmup", type=float, default=45.0)
    reference.add_argument("--out", required=True)
    reference.add_argument("--force", action="store_true", help="re-render references that already exist")

    study = sub.add_parser("study", help="render the convergence, stability and flicker passes, then report")
    study.add_argument("--scene", required=True)
    study.add_argument("--state", required=True)
    study.add_argument("--config", default="")
    study.add_argument("--arm", action="append", required=True, help="LABEL=Technique[:cvar=value]")
    study.add_argument("--seconds", type=float, default=30.0)
    study.add_argument("--checkpoints", default="log:16")
    study.add_argument("--images", type=int, default=3, help="independent images per arm per round")
    study.add_argument("--rounds", type=int, default=2)
    study.add_argument("--warmup", type=float, default=45.0)
    study.add_argument("--flicker-images", type=int, default=32, help="0 to skip the flicker pass")
    study.add_argument("--reference-dir", required=True)
    study.add_argument("--primary-reference", required=True,
                       help="reference label everything is scored against")
    study.add_argument("--baseline", help="arm the ratios are measured against (default: the first --arm)")
    study.add_argument("--metric", default="flipMean")
    study.add_argument("--target", type=float, help="error level for the equal-error readout")
    study.add_argument("--out")
    study.add_argument("--no-report", action="store_true")

    temporal = sub.add_parser("temporal", help="M1/M2 consecutive-frame difference and M6 variance homogeneity")
    temporal.add_argument("--scene", required=True)
    temporal.add_argument("--state", required=True)
    temporal.add_argument("--config", default="")
    temporal.add_argument("--arm", action="append", required=True, help="LABEL=Technique[:cvar=value]")
    temporal.add_argument("--frames", type=int, default=60, help="length of each sequence (SVGF uses 60)")
    temporal.add_argument("--homogeneity-frames", type=int, default=128,
                          help="independent single-frame renders for M6")
    temporal.add_argument("--warmup", type=float, default=45.0)
    temporal.add_argument("--reference-dir", required=True)
    temporal.add_argument("--primary-reference", required=True)
    temporal.add_argument("--out")
    temporal.add_argument("--no-report", action="store_true")

    report = sub.add_parser("report", help="re-score and re-plot a finished study")
    report.add_argument("--run", required=True)
    report.add_argument("--reference-dir", required=True)
    report.add_argument("--primary-reference", required=True)
    report.add_argument("--baseline")
    report.add_argument("--metric", default="flipMean")
    report.add_argument("--target", type=float)

    args = parser.parse_args()
    global EXE
    if args.command != "report":
        EXE = resolve_exe(getattr(args, "build", None))

    if args.command == "reference":
        render_references(args)
    elif args.command == "temporal":
        run_root = run_temporal(args)
        if not args.no_report:
            build_temporal_report(args, run_root)
        else:
            print(f"\nCaptures in {RAYTRACER_DIR / run_root}")
    elif args.command == "study":
        run_root = run_study(args)
        if not args.no_report:
            build_report(args, run_root)
        else:
            print(f"\nCaptures in {RAYTRACER_DIR / run_root}")
    else:
        build_report(args, args.run)


if __name__ == "__main__":
    main()
