#!/usr/bin/env python3
"""The measurement campaign: scene x light type x arm x budget, repeated.

This is the run whose numbers go into the thesis, which is what separates it from
`recon.py` — the reconnaissance ranks cells, this one measures them. Four things
follow from that and shape the whole file:

* **A job is (cell, arm, protocol); a category is a VIEW over jobs.** K1 and K2 both
  want veach-ajar under its own light and it is measured once, because measuring it
  twice would put two different numbers for one condition into the same document. The
  runner takes the union of what the enabled categories ask for; the report slices.

* **Rounds interleave.** The runner walks every job once per round instead of finishing
  a job's ten images back to back. A campaign takes hours, the GPU drifts over hours,
  and a drift that lands on one arm is indistinguishable from an effect.

* **Repeats re-walk the whole grid in fresh processes.** Images of one process share a
  warm-up, a clock state and a thermal state; on this harness their confidence interval
  understates the between-run spread 3-10x. Section 10.2 wants the honest figure, and
  the only way to get it is to measure the same arm twice from scratch. The report's
  noise-floor table is that comparison, and it is what every other table is judged
  against.

* **Resumable at the leaf.** Every (cell, arm, repeat, round) writes its own directory
  and a `done.json`; a completed one is skipped. An interrupted campaign resumes rather
  than restarting hours of rendering.

Scene identity, camera state, exposure and tone curve are NOT declared here. They were
frozen for the reconnaissance and the campaign must not drift away from them, so the
manifest points at `recon-manifest.json` for scenes and adds only the campaign's own
axes. The engine-facing helpers are imported from `recon.py` for the same reason: the
path convention (`--out` is Raytracer-relative, every filesystem read goes through
`under()`) has one implementation, not two.

Usage:
  python tools/campaign.py configs
  python tools/campaign.py references
  python tools/campaign.py run
  python tools/campaign.py report
"""

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np  # noqa: F401  (imported for the same failure message as elsewhere)
    from PIL import Image  # noqa: F401
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

import recon  # noqa: E402
from bench_report import (add_build_argument, aggregate, find_crossing,  # noqa: E402
                          flip_settings, load_rgb, resolve_exe, score_image, sidecar_for)
from evaluate import (flicker_stats, homogeneity_metrics,  # noqa: E402
                      sequence_metrics, spread)
from recon import RAYTRACER_DIR, build_light, cell_id, engine_checked, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ the manifest

def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    scenes_from = manifest.get("scenesFrom")
    if not scenes_from:
        sys.exit("Campaign manifest must name 'scenesFrom': scenes are frozen in the recon manifest")
    source = recon.load_manifest(REPO_ROOT / scenes_from)
    manifest["scenes"] = source["scenes"]
    manifest["defaultLights"] = source.get("defaultLights", {})
    manifest["defaultGridResolutions"] = source.get("defaultGridResolutions", [32, 64])
    manifest["_scenesFromResolved"] = str(scenes_from)
    return manifest


def scene_by_id(manifest, scene_id):
    for scene in manifest["scenes"]:
        if scene["id"] == scene_id:
            return scene
    sys.exit(f"Category names scene '{scene_id}', which {manifest['_scenesFromResolved']} does not declare")


def enabled_categories(manifest, only):
    names = only or list(manifest["categories"])
    for name in names:
        if name not in manifest["categories"]:
            sys.exit(f"Unknown category '{name}'. Known: {', '.join(manifest['categories'])}")
    return {name: manifest["categories"][name] for name in names}


def jobs(manifest, only=None):
    """The union of what the enabled categories ask for, keyed by (cell, arm).

    Returns a dict {(cell, arm): {"scene", "light", "arm", "protocol", "categories"}}.
    A job asked for by two categories under different protocols is a manifest error and
    is refused here rather than silently measured under one of them.
    """
    out = {}
    for name, category in enabled_categories(manifest, only).items():
        scene_ids = ([s["id"] for s in manifest["scenes"]]
                     if category["scenes"] == "all" else category["scenes"])
        for scene_id in scene_ids:
            scene = scene_by_id(manifest, scene_id)
            for light in category["lights"]:
                for arm in category["arms"]:
                    key = (cell_id(scene, light), arm)
                    entry = out.get(key)
                    if entry is None:
                        out[key] = {"scene": scene, "light": light, "arm": arm,
                                    "protocol": category["protocol"], "categories": [name]}
                    elif entry["protocol"] != category["protocol"]:
                        sys.exit(f"{key[0]} / {key[1]} is asked for by {entry['categories'][0]} under "
                                 f"protocol '{entry['protocol']}' and by {name} under "
                                 f"'{category['protocol']}' — one condition cannot have two protocols")
                    else:
                        entry["categories"].append(name)
    return out


def campaign_cells(manifest, only=None):
    """Every distinct cell the campaign touches, in a stable order."""
    seen = {}
    for (cell, _arm), job in jobs(manifest, only).items():
        seen.setdefault(cell, (job["scene"], job["light"]))
    return [(cell, scene, light) for cell, (scene, light) in seen.items()]


def config_path(manifest, scene, light):
    return Path(manifest["configDir"]) / f"campaign.{cell_id(scene, light)}.json"


# ------------------------------------------------------------------ configs

def write_configs(manifest, only=None):
    """One config per cell. The config is the record of a measurement's conditions, so
    the reference image and every arm scored against it read the same file."""
    out_dir = RAYTRACER_DIR / manifest["configDir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    strict = {name for name, c in enabled_categories(manifest, only).items()
              if c.get("_requiresExplicitRig")}

    written = []
    for cell, scene, light in campaign_cells(manifest, only):
        # A category that puts light type on its axis must not fall back to the shared
        # default rig: the default is one scene's sun, and using it elsewhere would
        # measure "some directional light" rather than this scene's substituted source.
        if strict and light in ("point", "directional") and light not in scene.get("states", {}):
            has_own = light in scene.get("lights", {})
            if not has_own:
                sys.exit(f"{cell}: category {sorted(strict)} puts light type on its axis, so this "
                         f"cell needs its OWN '{light}' rig in {manifest['_scenesFromResolved']} "
                         f"under scenes[].lights — the shared defaultLights entry is another "
                         f"scene's rig and would measure the wrong thing")

        config = dict(manifest["renderDefaults"])
        config["exposure"] = scene["exposure"]
        config.update(scene.get("overrides", {}))
        rig = build_light(scene, light, manifest)
        if rig is not None:
            config["lights"] = rig
        if light in ("point", "directional"):
            config["emissiveGeometry"] = False

        path = RAYTRACER_DIR / config_path(manifest, scene, light)
        path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        written.append(path)

    print(f"{len(written)} configs written under {out_dir}")
    for path in written:
        print(f"  {path.relative_to(RAYTRACER_DIR).as_posix()}")


# ------------------------------------------------------------------ references

def run_references(args, manifest):
    """One long path-traced reference per CELL, not per job: the arms differ in how they
    sample the integral, not in which integral it is, so they share an image."""
    root = Path(args.out)
    (RAYTRACER_DIR / root).mkdir(parents=True, exist_ok=True)
    seconds = args.seconds or manifest["referenceSeconds"]

    for cell, scene, light in campaign_cells(manifest, args.only):
        target = under(root) / f"{cell}.png"
        if target.exists():
            print(f"exists  {cell}.png")
            continue
        out = root / "raw" / cell
        code, took = engine_checked(scene, light, manifest, "Path Tracing", [], out,
                                    f"seconds:{seconds}", 1, args.warmup,
                                    log_path=under(root) / f"{cell}.log",
                                    config=config_path(manifest, scene, light))
        if code != 0:
            print(f"FAILED  {cell} exit={code}")
            continue
        produced = sorted(p for p in under(out).rglob("*.png") if not p.name.endswith(".flip.png"))
        if not produced:
            print(f"FAILED  {cell}: the run produced no capture")
            continue
        produced[0].replace(target)
        sidecar = produced[0].with_suffix(".json")
        if sidecar.exists():
            sidecar.replace(target.with_suffix(".json"))
        print(f"ok      {cell}.png in {took / 60:.1f} min")


# ------------------------------------------------------------------ the campaign

def leaf(root, cell, arm, repeat, round_index):
    return Path(root) / cell / arm / f"p{repeat}r{round_index}"


# ------------------------------------------------------- per-scene technique settings

def load_parameters(path):
    """The per-scene technique settings phase 1 chose, or {} when there are none yet.

    Kept OUT of the manifest on purpose: the manifest is written by hand and states what
    the measurement is, while this file is produced by a sweep and states what the sweep
    found. Mixing them would make it impossible to re-run the sweep without editing the
    description of the measurement."""
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        sys.exit(f"No parameters file at {resolved}. Run 'campaign.py sweep' first, or pass "
                 f"--parameters '' to measure on the engine defaults (and say so in the report).")
    return json.loads(resolved.read_text(encoding="utf-8-sig")).get("scenes", {})


def scene_cvars(cell, parameters):
    """The cell's own settings as --cvar assignments, in a stable order."""
    entry = parameters.get(cell, {})
    return [f"{name}={value}" for name, value in sorted(entry.items())
            if not name.startswith("_")]


def load_frame_pairs(path):
    """{cell: {arm: frames}} chosen so both arms spend the same TIME.

    A time budget cannot equalise the time at this scale: the engine stops at the first
    frame REACHING the budget, so each arm overshoots by a partial frame and the two
    overshoots differ. Measured at 24 ms, that left the arms up to 20 % apart in time -
    always in the guided arm's favour, because its frame is the larger one. Stating the
    frame counts instead removes the quantisation: the counts are picked so that
    N1*cost1 and N2*cost2 land within a few percent of each other and of the target.

    The costs behind them must come from the regime being measured. Frame cost under a
    10 s load is 2-42 % higher than in a 24 ms burst, because the card clocks down - and
    the guided arm loses more to that than the path-traced one.
    """
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        sys.exit(f"No frame-pair table at {resolved}. The protocol asks for one; produce it "
                 f"from a measured run or point 'framesFrom' elsewhere.")
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def arm_budget(protocol, arm, manifest, cell=None, frame_pairs=None):
    """The stopping condition for ONE arm.

    Three shapes. A plain time budget is one value for both arms. A SAMPLE COUNT is not:
    the guided technique draws two path samples per iteration (two-sample MIS at the first
    vertex) where path tracing draws one, so the same sample count is a different frame
    count per arm. A FRAME-PAIR table is per cell and per arm, and is how equal time is
    actually realised at a budget of one display frame.
    """
    if "framesFrom" in protocol:
        entry = (frame_pairs or {}).get(cell, {})
        if arm not in entry:
            sys.exit(f"No frame count for {cell} / {arm} in the frame-pair table")
        return f"frames:{entry[arm]}"
    if "samples" in protocol:
        per_frame = manifest["arms"][arm].get("samplesPerFrame", 1)
        frames = protocol["samples"] // per_frame
        if frames * per_frame != protocol["samples"]:
            sys.exit(f"protocol asks for {protocol['samples']} samples, which arm {arm} "
                     f"cannot hit exactly at {per_frame} sample(s) per frame")
        return f"frames:{frames}"
    return protocol["budget"]


def run_campaign(args, manifest):
    """Interleaved rounds inside repeated passes.

    The loop order is repeat -> round -> job, and that order is the measurement design:
    every job is touched once before any job is touched twice, so a GPU that drifts over
    the campaign drifts across all of them together instead of favouring whichever arm
    happened to run while it was cold.
    """
    root = Path(args.out)
    (RAYTRACER_DIR / root).mkdir(parents=True, exist_ok=True)
    plan = jobs(manifest, args.only)
    protocols = manifest["protocols"]
    rounds = args.rounds or manifest["rounds"]
    repeats = args.repeats or manifest["repeats"]
    parameters = load_parameters(getattr(args, "parameters", None))
    frame_pairs = {name: load_frame_pairs(spec["framesFrom"])
                   for name, spec in protocols.items() if "framesFrom" in spec}

    total = len(plan) * rounds * repeats
    print(f"{len(plan)} jobs x {rounds} round(s) x {repeats} repeat(s) = {total} runs")

    if args.dry_run:
        # The union is the whole point of the job model, so it has to be inspectable
        # before hours of rendering rather than inferred from the output tree afterwards.
        protocol_seconds = {}
        for name, spec in protocols.items():
            kind, _, value = spec.get("budget", "").partition(":")
            protocol_seconds[name] = float(value) if kind == "seconds" else 0.0
        estimate = 0.0
        for (cell, arm), job in sorted(plan.items()):
            protocol = protocols[job["protocol"]]
            per_run = protocol_seconds[job["protocol"]] * (protocol["images"] // rounds) \
                + protocol["warmup"]
            estimate += per_run * rounds * repeats
            print(f"  {cell:<26} {arm:<6} {job['protocol']:<6} "
                  f"{'+'.join(sorted(job['categories']))}")
        print(f"\nrendering alone, excluding scene loads: {estimate / 3600:.1f} h")
        return

    done = skipped = failed = 0
    for repeat in range(repeats):
        for round_index in range(rounds):
            for (cell, arm), job in plan.items():
                protocol = protocols[job["protocol"]]
                out = leaf(root, cell, arm, repeat, round_index)
                if (under(out) / "done.json").exists():
                    skipped += 1
                    continue

                # Ten images per point, split across the rounds rather than repeated in
                # each: the protocol asks for ten, and two rounds of ten would be twenty.
                images = protocol["images"] // rounds
                if images < 1:
                    sys.exit(f"protocol '{job['protocol']}' asks for {protocol['images']} images "
                             f"across {rounds} rounds, which leaves a round with none")

                spec = manifest["arms"][arm]
                label = f"p{repeat}r{round_index} {cell} {arm}"
                print(f"  {label} ...", flush=True)
                code, took = engine_checked(
                    job["scene"], job["light"], manifest, spec["technique"],
                    list(spec["cvars"]) + scene_cvars(cell, parameters),
                    out, arm_budget(protocol, arm, manifest, cell,
                                    frame_pairs.get(job["protocol"])),
                    images, protocol["warmup"],
                    checkpoints=protocol["checkpoints"], settle=protocol.get("settle", 0),
                    log_path=under(root) / f"{cell}-{arm}-p{repeat}r{round_index}.log",
                    config=config_path(manifest, job["scene"], job["light"]))
                if code != 0:
                    print(f"  FAILED {label} exit={code}")
                    failed += 1
                    continue
                (under(out) / "done.json").write_text(json.dumps({
                    "seconds": took, "images": images, "arm": arm, "cell": cell,
                    "repeat": repeat, "round": round_index,
                    "categories": job["categories"]}))
                done += 1
                print(f"  ok     {label} in {took:.0f}s")

    print(f"\n{done} run(s), {skipped} already present, {failed} failed")


# ------------------------------------------------------------------ scoring

# 8.1 and Q14: the whole set the metric's own paper reports, not a chosen one. The mean
# stays the headline - it is the only one of these that counts a firefly - but a reader
# who wants to know whether a difference lives in the tail or in the bulk needs the rest,
# and the two medians disagree on purpose: one pools error mass, the other does not.
FLIP_STATISTICS = ("flipMedian", "flipQ1", "flipQ3", "flipP95", "flipMin", "flipMax",
                   "flipWeightedQ1", "flipWeightedQ3")

def score_arm(directory, reference):
    """Every capture under one leaf, aggregated per checkpoint ordinal."""
    per_checkpoint = {}
    for png in sorted(Path(directory).rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        sidecar = sidecar_for(png)
        index = sidecar.get("benchmark", {}).get("checkpointIndex", 0)
        scores, _ = score_image(reference, png)
        entry = per_checkpoint.setdefault(index, {"flip": [], "wmedian": [], "mse": [],
                                                  "ms": [], "seconds": 0.0, "frames": 0})
        entry["flip"].append(scores["flipMean"])
        # 8.2 wants a second, independent metric so a conclusion never rests on one
        # perceptual model. Taken on the same displayed image the metric sees (D5).
        entry["mse"].append(scores["mse"])
        for key in FLIP_STATISTICS:
            entry.setdefault(key, []).append(scores[key])
        # 16.6 asks for the weighted median as its own column, not as a replacement: it
        # pools the error mass and is therefore blind to the fireflies the mean counts.
        entry["wmedian"].append(scores["flipWeightedMedian"])
        entry["ms"].append(sidecar.get("benchmark", {}).get("meanFrameMs", 0.0))
        entry["seconds"] = sidecar.get("raytracing", {}).get("accumulatedTime", 0.0)
        entry["frames"] = sidecar.get("raytracing", {}).get("frameIndex", 0)
    return per_checkpoint


def gather(manifest, args):
    """{cell: {arm: {repeat: {checkpoint: aggregate}}}}, plus the reference each used."""
    root = Path(args.out)
    references = Path(args.reference_dir)
    rounds = args.rounds or manifest["rounds"]
    repeats = args.repeats or manifest["repeats"]

    data = {}
    for (cell, arm), _job in jobs(manifest, args.only).items():
        reference_path = under(references) / f"{cell}.png"
        if not reference_path.exists():
            print(f"skip {cell}: no reference")
            continue
        reference = load_rgb(reference_path)
        for repeat in range(repeats):
            merged = {}
            for round_index in range(rounds):
                directory = under(leaf(root, cell, arm, repeat, round_index))
                if not directory.exists():
                    continue
                # Rounds are halves of one sample of ten images, so they merge; repeats
                # are separate measurements and must not.
                for index, entry in score_arm(directory, reference).items():
                    into = merged.setdefault(index, {"flip": [], "wmedian": [], "mse": [],
                                                     "ms": [], "seconds": 0.0, "frames": 0})
                    into["flip"] += entry["flip"]
                    into["wmedian"] += entry["wmedian"]
                    into["mse"] += entry["mse"]
                    for key in FLIP_STATISTICS:
                        into.setdefault(key, []).extend(entry.get(key, []))
                    into["ms"] += entry["ms"]
                    into["seconds"] = entry["seconds"] or into["seconds"]
                    into["frames"] = entry["frames"] or into["frames"]
            if not merged:
                continue
            checkpoints = {}
            for index, e in sorted(merged.items()):
                point = {"flip": aggregate(e["flip"]), "ms": aggregate(e["ms"]),
                         "wmedian": aggregate(e["wmedian"]), "mse": aggregate(e["mse"]),
                         "seconds": e["seconds"], "frames": e["frames"]}
                for key in FLIP_STATISTICS:
                    point[key] = aggregate(e.get(key, []))
                checkpoints[index] = point
            data.setdefault(cell, {}).setdefault(arm, {})[repeat] = checkpoints
    return data


def curve(passes, value_key="flip", axis="seconds", repeat=0):
    """One pass of one arm as [(x, y)] ascending: the error-against-budget series of 8.1."""
    points = passes.get(repeat)
    if not points:
        return []
    series = []
    for index in sorted(points):
        entry = points[index]
        x = entry["seconds"] if axis == "seconds" else float(entry["frames"])
        y = entry.get(value_key, {}).get("mean", 0.0)
        if x > 0 and y > 0:
            series.append((x, y))
    return series


def value_at(series, x):
    """The series read at x, interpolated in log-log. None outside the measured span.

    Same power-law argument as find_crossing: error falls like a power of the budget, so
    a straight line between two checkpoints belongs in log-log, not in the raw axes.
    """
    if not series or x < series[0][0] or x > series[-1][0]:
        return None
    for (x_a, y_a), (x_b, y_b) in zip(series, series[1:]):
        if x_a <= x <= x_b:
            if x_b == x_a:
                return y_a
            t = (math.log(x) - math.log(x_a)) / (math.log(x_b) - math.log(x_a))
            return math.exp(math.log(y_a) + t * (math.log(y_b) - math.log(y_a)))
    return series[-1][1]


def show(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def full_statistics(data):
    """8.1's whole statistic set at the final checkpoint, one row per arm."""
    lines = ["", "## FLIP, the full statistic set (8.1)", "",
             "Final checkpoint, pass 0. The mean is the headline and stays the number every "
             "other table quotes; it is also the only one here that counts a firefly at its "
             "true weight. The two medians are both reported because they answer different "
             "questions - the weighted one pools error mass over the image, the plain one is "
             "read off the error map - and a conclusion that needs them to agree is not a "
             "conclusion. Q1/Q3/p95/max say whether a difference lives in the bulk or in the "
             "tail.", "",
             "| cell | arm | mean | w-median | w-Q1 | w-Q3 | median | Q1 | Q3 | p95 | "
             "min | max | MSE |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell, arms in sorted(data.items()):
        for arm, passes in sorted(arms.items()):
            if 0 not in passes:
                continue
            point = passes[0][max(passes[0])]

            def mean_of(key):
                return point.get(key, {}).get("mean", 0.0)

            lines.append(f"| {cell} | {arm} | {point['flip']['mean']:.5f} | "
                         f"{point['wmedian']['mean']:.5f} | "
                         f"{mean_of('flipWeightedQ1'):.5f} | {mean_of('flipWeightedQ3'):.5f} | "
                         f"{mean_of('flipMedian'):.5f} | {mean_of('flipQ1'):.5f} | "
                         f"{mean_of('flipQ3'):.5f} | {mean_of('flipP95'):.5f} | "
                         f"{mean_of('flipMin'):.5f} | {mean_of('flipMax'):.5f} | "
                         f"{mean_of('mse'):.6f} |")
    return lines


def comparison_modes(data, manifest, args):
    """10.1 asks for three modes and they are three different questions.

    Equal time is the category tables above. Equal error asks how long each arm needs to
    reach the same error - the mode a practitioner actually buys. Equal sample removes the
    frame-rate difference altogether and leaves only the estimator, which is the only mode
    in which the guide can be judged apart from what it costs.
    """
    pairs = []
    for _name, category in enabled_categories(manifest, args.only).items():
        for arm in category["arms"][1:]:
            pairs.append((category["arms"][0], arm))
    pairs = list(dict.fromkeys(pairs))

    lines = ["", "## Equal error and equal sample (10.1)", "",
             "Equal time is the table above. Here the other two modes of 10.1, both read off "
             "the 16-checkpoint curve with logarithmic interpolation. The equal-error target is "
             "the higher of the two arms' final errors - the easiest error both actually reach, "
             "so neither time column is an extrapolation. Equal sample is read at the lower of "
             "the two final frame counts, for the same reason.", "",
             "| pair | cell | target FLIP | t(base) s | t(arm) s | speed-up | frames | "
             "FLIP base | FLIP arm | per-sample |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for baseline, arm in pairs:
        for cell, arms in sorted(data.items()):
            if baseline not in arms or arm not in arms:
                continue
            base_time, arm_time = curve(arms[baseline]), curve(arms[arm])
            base_frames = curve(arms[baseline], axis="frames")
            arm_frames = curve(arms[arm], axis="frames")
            if not (base_time and arm_time and base_frames and arm_frames):
                continue

            target = max(base_time[-1][1], arm_time[-1][1])
            t_base, t_arm = find_crossing(base_time, target), find_crossing(arm_time, target)
            speedup = f"{t_base / t_arm:.2f}x" if t_base and t_arm else "—"

            common = min(base_frames[-1][0], arm_frames[-1][0])
            flip_base, flip_arm = value_at(base_frames, common), value_at(arm_frames, common)
            per_sample = f"{flip_base / flip_arm:.2f}x" if flip_base and flip_arm else "—"

            lines.append(f"| {baseline}/{arm} | {cell} | {target:.5f} | {show(t_base, 2)} | "
                         f"{show(t_arm, 2)} | {speedup} | {common:.0f} | {show(flip_base)} | "
                         f"{show(flip_arm)} | {per_sample} |")

    lines += ["", "## Estimator spread across images (8.3)", "",
              "Standard deviation of FLIP over the ten independent images of the final "
              "checkpoint, and the same relative to the mean. This is the estimator's own "
              "spread at a fixed budget - the quantity P3 watches part company with the error "
              "curve when an arm is biased, because bias moves the mean and leaves the spread "
              "where it was.", "",
              "| cell | arm | FLIP | stdev | rel. stdev | MSE | MSE stdev | n |",
              "|---|---|---|---|---|---|---|---|"]
    for cell, arms in sorted(data.items()):
        for arm, passes in sorted(arms.items()):
            if 0 not in passes:
                continue
            point = passes[0][max(passes[0])]
            flip, mse = point["flip"], point.get("mse", {})
            rel = flip["stdev"] / flip["mean"] if flip["mean"] > 0 else 0.0
            lines.append(f"| {cell} | {arm} | {flip['mean']:.5f} | {flip['stdev']:.6f} | "
                         f"{rel * 100:.2f} % | {mse.get('mean', 0.0):.6f} | "
                         f"{mse.get('stdev', 0.0):.6f} | {flip['n']} |")
    return lines


def write_curves(data, path):
    """The full 16-point curve of every arm, as data rather than as prose.

    The tables quote the last checkpoint; 8.1 is about the whole curve, and its figures are
    drawn outside this tool. Written once here so no figure is ever redrawn from a number
    retyped out of a table.
    """
    rows = ["cell,arm,pass,checkpoint,seconds,frames,msFrame,flipMean,flipStdev,flipCi95,"
            "wMedian,mse"]
    for cell, arms in sorted(data.items()):
        for arm, passes in sorted(arms.items()):
            for repeat in sorted(passes):
                for index in sorted(passes[repeat]):
                    e = passes[repeat][index]
                    rows.append(
                        f"{cell},{arm},{repeat},{index},{e['seconds']:.4f},{e['frames']},"
                        f"{e['ms']['mean']:.4f},{e['flip']['mean']:.6f},"
                        f"{e['flip']['stdev']:.6f},{e['flip'].get('ci95', 0.0):.6f},"
                        f"{e['wmedian']['mean']:.6f},{e.get('mse', {}).get('mean', 0.0):.6f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows) - 1


def reference_note(args):
    """Which reference these numbers were scored against, and how long it ran.

    E5 turned the reference into a parameter of the comparison rather than a technical
    detail, so no table may leave this tool without saying which one produced it.
    """
    directory = under(Path(args.reference_dir))
    seconds = set()
    for sidecar in sorted(directory.glob("*.json")):
        if sidecar.stem.endswith(".stale-i55p3"):
            continue
        try:
            budget = json.loads(sidecar.read_text(encoding="utf-8"))["benchmark"]
        except (OSError, ValueError, KeyError):
            continue
        seconds.add(float(budget.get("budgetValue", 0.0)))
    span = "/".join(f"{value:.0f}" for value in sorted(seconds)) or "unknown"
    return f"Reference: `{args.reference_dir}`, path traced, {span} s per image."


def run_report(args, manifest):
    data = gather(manifest, args)
    if not data:
        sys.exit("Nothing scored. Run 'references' and 'run' first.")

    lines = ["# Campaign", "",
             f"Build: {recon.EXE.parent.name}. Protocol: PLAN_BADAWCZY 7.9. "
             f"Scenes from `{manifest['_scenesFromResolved']}`.",
             f"Metric: FLIP {flip_settings()['version']}, LDR, "
             f"{flip_settings()['ppd']:.2f} pixels per degree (16.7). "
             "`FLIP` is the mean, `w-median` the tool's weighted median (16.6) — the mean is the "
             "headline because the weighted median pools error mass and is blind to fireflies.",
             reference_note(args), ""]

    # --- the noise floor, first, because every other table is read against it -------
    lines += ["## Between-run noise floor (K10)", "",
              "The same arm, measured twice from scratch in separate processes. This is the "
              "spread a difference has to clear before it is a difference; the confidence "
              "interval inside one process is not, and understates it.", "",
              "| cell | arm | FLIP pass 0 | FLIP pass 1 | spread | ci95 within pass 0 |",
              "|---|---|---|---|---|---|"]
    floors = {}
    for cell, arms in sorted(data.items()):
        for arm, passes in sorted(arms.items()):
            if len(passes) < 2:
                continue
            last = [passes[p][max(passes[p])] for p in sorted(passes)[:2]]
            a, b = last[0]["flip"]["mean"], last[1]["flip"]["mean"]
            spread = abs(a - b) / max(a, b) if max(a, b) > 0 else 0.0
            floors[(cell, arm)] = spread
            lines.append(f"| {cell} | {arm} | {a:.5f} | {b:.5f} | {spread * 100:.2f} % "
                         f"| ±{last[0]['flip'].get('ci95', 0.0):.5f} |")
    if not floors:
        lines.append("| — | — | — | — | — | *only one pass on disk* |")

    # --- per-category results ------------------------------------------------------
    for name, category in enabled_categories(manifest, args.only).items():
        lines += ["", f"## {name} — {category.get('_what', '')}", "",
                  "Last checkpoint of the curve, averaged over rounds, pass 0.", "",
                  "| cell | arm | ms/frame | frames | FLIP | ci95 | w-median | vs first arm |",
                  "|---|---|---|---|---|---|---|---|"]
        scene_ids = ([s["id"] for s in manifest["scenes"]]
                     if category["scenes"] == "all" else category["scenes"])
        baseline_arm = category["arms"][0]
        for scene_id in scene_ids:
            scene = scene_by_id(manifest, scene_id)
            for light in category["lights"]:
                cell = cell_id(scene, light)
                arms = data.get(cell, {})
                base = arms.get(baseline_arm, {}).get(0)
                base_flip = base[max(base)]["flip"]["mean"] if base else None
                for arm in category["arms"]:
                    passes = arms.get(arm, {})
                    if 0 not in passes:
                        continue
                    point = passes[0][max(passes[0])]
                    ratio = (f"{base_flip / point['flip']['mean'] * 100:.0f} %"
                             if base_flip and arm != baseline_arm and point["flip"]["mean"] > 0
                             else "—")
                    lines.append(f"| {cell} | {arm} | {point['ms']['mean']:.3f} | "
                                 f"{point['frames']} | {point['flip']['mean']:.5f} | "
                                 f"±{point['flip'].get('ci95', 0.0):.5f} | "
                                 f"{point['wmedian']['mean']:.5f} | {ratio} |")

        # A category with a light-type axis is asking one question and it is not "how do
        # these cells differ": it substitutes the source INTO the same scene, under the
        # same camera, and asks how the technique's advantage moves. That is a difference
        # of differences, so it needs its own table — reading it off two independent rows
        # invites comparing veach-ajar's point cell against staircase's own cell, which
        # answers nothing.
        if len(category["lights"]) > 1:
            first, *rest = category["lights"]
            lines += ["", f"### {name} — advantage under a substituted source", "",
                      f"Each scene under its own source and under each substitute, geometry and "
                      f"camera unchanged. The column that carries the category is the last one: "
                      f"how much of the technique's advantage survives the substitution.", "",
                      "| scene | source | "
                      + " | ".join(f"{a} ms" for a in category["arms"]) + " | "
                      + f"{category['arms'][-1]}/{baseline_arm} | vs own source |",
                      "|---" * (4 + len(category["arms"])) + "|"]
            for scene_id in scene_ids:
                scene = scene_by_id(manifest, scene_id)
                own_ratio = None
                for light in [first] + rest:
                    arms = data.get(cell_id(scene, light), {})
                    points = {}
                    for arm in category["arms"]:
                        passes = arms.get(arm, {})
                        if 0 in passes:
                            points[arm] = passes[0][max(passes[0])]
                    if len(points) < len(category["arms"]):
                        continue
                    base_flip = points[baseline_arm]["flip"]["mean"]
                    test = points[category["arms"][-1]]["flip"]["mean"]
                    ratio = base_flip / test * 100 if test > 0 else 0.0
                    if light == first:
                        own_ratio = ratio
                    shift = ("—" if light == first or not own_ratio
                             else f"{ratio - own_ratio:+.0f} pp")
                    costs = " | ".join(f"{points[a]['ms']['mean']:.3f}" for a in category["arms"])
                    lines.append(f"| {scene_id} | {light} | {costs} | {ratio:.0f} % | {shift} |")

    lines += full_statistics(data)
    lines += comparison_modes(data, manifest, args)

    out = under(Path(args.out)) / f"{args.name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")

    curves = under(Path(args.out)) / f"{args.name}-curves.csv"
    print(f"written: {curves} ({write_curves(data, curves)} points)")
    raw = under(Path(args.out)) / f"{args.name}.json"
    raw.write_text(json.dumps({"flip": flip_settings(), "data": data}, indent=1),
                   encoding="utf-8")
    print(f"written: {raw}")

    # The same run in the plan's own shape, next to the tool's native one: the chapter's
    # plotting scripts read this file and never have to know how gather() nests.
    flat = flatten_for_schema(data)
    write_measurement_json(under(Path(args.out)) / f"{args.name}-wyniki.json", args.name, flat,
                           args, load_parameters(getattr(args, "parameters", None)))
    write_measurement_csv(under(Path(args.out)) / f"{args.name}-wyniki.csv", flat)



# ------------------------------------------------------------------ error-map analysis

# The cluster debug view paints hue = frac(cluster * 0.618034) through a triangle wave
# (debugViewPaint.hlsl). A saturated hue always leaves one channel at zero, so white
# (a seed voxel), the near-black "unlit" and the magenta "corrupt" marker stay
# distinguishable from every cluster colour and the mapping inverts by nearest match.
CLUSTER_COUNT = 32


def cluster_palette():
    hues = (np.arange(CLUSTER_COUNT) * 0.618034) % 1.0
    h6 = hues * 6.0
    r = np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0)
    g = np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0)
    b = np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def cluster_ids(view_rgb):
    """Per-pixel cluster id; -1 where the view says the pixel carries no cluster."""
    palette = cluster_palette()
    flat = view_rgb.reshape(-1, 3)
    marker = (flat.min(axis=1) > 0.85) | (flat.max(axis=1) < 0.25)
    distance = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    ids = distance.argmin(axis=1)
    ids[marker] = -1
    # A colour far from every palette entry is not a cluster either.
    ids[distance.min(axis=1) > 0.15] = -1
    return ids.reshape(view_rgb.shape[:2])


def write_heat_png(values, path, vmax):
    """Magnitude map on the black-red-yellow-white ramp the engine's own views use."""
    t = np.clip(values / vmax if vmax > 0 else values, 0.0, 1.0)
    r = np.clip(t * 3.0, 0.0, 1.0)
    g = np.clip((t - 1.0 / 3.0) * 3.0, 0.0, 1.0)
    b = np.clip((t - 2.0 / 3.0) * 3.0, 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=2) * 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def write_diverging_png(values, path, limit):
    """Signed map: red where the biased arm is worse, blue where it is better, white at
    zero. A magnitude map cannot show that, and the sign is the whole question."""
    t = np.clip(values / limit if limit > 0 else values, -1.0, 1.0)
    positive = np.clip(t, 0.0, 1.0)
    negative = np.clip(-t, 0.0, 1.0)
    r = 1.0 - negative
    g = 1.0 - positive - negative
    b = 1.0 - positive
    rgb = (np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def averaged_map(root, cell, arm, repeat, rounds, reference, checkpoint):
    """One arm's error map at one checkpoint, averaged over the rounds of one repeat.

    Rounds are halves of the same sample and merge; repeats are separate measurements and
    must not, which is what makes repeat 1 usable as an independent copy of repeat 0."""
    total, count = None, 0
    for round_index in range(rounds):
        partial, n = mean_error_map(under(leaf(root, cell, arm, repeat, round_index)),
                                    reference, checkpoint)
        if partial is None:
            continue
        total = partial * n if total is None else total + partial * n
        count += n
    return (total / count, count) if count else (None, 0)


def noise_floor(args, root, cell, arm, repeat_zero_map, rounds, reference, checkpoint):
    """The per-pixel level below which a difference is not evidence.

    'auto' derives it from the SAME arm measured twice from scratch — the between-run
    spread PLAN_BADAWCZY 10.2 asks for, taken at the scale the threshold actually works
    at. That matters: the K10 table's spread is on the image MEAN, which averages two
    million pixels and is therefore some three orders of magnitude tighter than the
    per-pixel noise this threshold has to clear. Using the K10 number here would call
    almost every pixel evidence.

    Returns (floor, how it was obtained)."""
    if args.noise != "auto":
        fraction = float(args.noise)
        return fraction * float(repeat_zero_map.mean()), f"{fraction * 100:.0f} % of the arm's own mean error"

    second, _count = averaged_map(root, cell, arm, 1, rounds, reference, checkpoint)
    if second is None:
        fallback = 0.10
        return (fallback * float(repeat_zero_map.mean()),
                f"{fallback * 100:.0f} % of the arm's own mean error (no second pass on disk)")
    spread = np.abs(repeat_zero_map - second)
    return (float(np.quantile(spread, 0.95)),
            "p95 of the same arm measured twice from scratch")


def mean_error_map(directory, reference, checkpoint):
    """The images of one checkpoint, scored and AVERAGED as maps.

    Averaging the maps rather than picking one is the point of the exercise: bias is
    systematic and survives the average, Monte Carlo noise does not. One image would show
    mostly noise, and the difference of two of them almost entirely so."""
    total, count = None, 0
    for png in sorted(Path(directory).rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        if sidecar_for(png).get("benchmark", {}).get("checkpointIndex", 0) != checkpoint:
            continue
        _, error_map = score_image(reference, png)
        total = error_map if total is None else total + error_map
        count += 1
    return (total / count, count) if count else (None, 0)


def per_image_flip(directory, reference, checkpoint):
    """(flip mean, path) for every image of one checkpoint, for step 4."""
    out = []
    for png in sorted(Path(directory).rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        if sidecar_for(png).get("benchmark", {}).get("checkpointIndex", 0) != checkpoint:
            continue
        scores, _ = score_image(reference, png)
        out.append((scores["flipMean"], png))
    return sorted(out)


def render_cluster_view(scene, light, manifest, out_dir, seconds=2.0):
    """The cluster-assignment view for this cell's camera, so step 2 has something to
    bucket by. Rendered on demand rather than demanded as an input: it is a property of
    the cell, not of the measurement, and re-rendering it costs seconds."""
    target = under(out_dir) / "clusters.png"
    if target.exists():
        return target
    raw = Path(out_dir) / "clusters-raw"
    code, _ = engine_checked(scene, light, manifest, "Guided Path Tracing (VXPG)", [], raw,
                             f"seconds:{seconds}", 1, 0,
                             checkpoints=None,
                             log_path=under(out_dir) / "clusters.log",
                             config=config_path(manifest, scene, light),
                             debug_views="1003")
    if code != 0:
        return None
    produced = sorted(p for p in under(raw).rglob("*.png") if not p.name.endswith(".flip.png"))
    if not produced:
        return None
    produced[0].replace(target)
    return target


def write_histogram(difference, floor, path):
    """Step 3 wants the DISTRIBUTION, not just its tails. Optional, like every other plot
    in these tools: matplotlib is a convenience here and the numbers above it are not."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    flat = difference.ravel()
    limit = float(np.quantile(np.abs(flat), 0.999)) or 1e-6
    figure, axis = plt.subplots(figsize=(7.0, 3.6))
    axis.hist(np.clip(flat, -limit, limit), bins=256, color="#41628E")
    axis.axvline(0.0, color="#191713", linewidth=1.0)
    for sign in (-1.0, 1.0):
        axis.axvline(sign * floor, color="#9A3A2E", linewidth=1.0, linestyle="--")
    axis.set_xlabel("per-pixel FLIP difference (biased - unbiased); dashed = noise floor")
    axis.set_ylabel("pixels")
    axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return True


def run_diffmap(args, manifest):
    """The four steps of PLAN_BADAWCZY 8.5.1: where the reuse variant's bias lives.

    Lives here rather than in evaluate.py, which the plan names, because every input is
    the campaign's — its cells, its arms, its reference and its leaf layout. Teaching a
    second tool that layout would give one tree two readers.
    """
    root = Path(args.out)
    references = Path(args.reference_dir)
    out_root = Path(args.maps_out)
    (RAYTRACER_DIR / out_root).mkdir(parents=True, exist_ok=True)
    rounds = args.rounds or manifest["rounds"]

    lines = ["# Where two arms differ, pixel by pixel (8.5.1)", "",
             "Every category with exactly two arms gets a section, and each section names its "
             "pair — the same cell appears under several categories against different arms. "
             "**Only the category whose second arm is the reuse variant reads as a BIAS study**; "
             "elsewhere this is a difference map between two unbiased estimators, where the sign "
             "says which one is closer to the reference, not which one is wrong.", "",
             "A pixel counts as moved only past a stated noise floor; below it a difference is "
             "not evidence. The floor is printed with every checkpoint together with how it was "
             "obtained, because a threshold that does not say where it came from is a claim.", ""]

    for name, category in enabled_categories(manifest, args.only).items():
        if len(category["arms"]) != 2:
            continue
        baseline, biased = category["arms"]
        scene_ids = ([s["id"] for s in manifest["scenes"]]
                     if category["scenes"] == "all" else category["scenes"])
        for scene_id in scene_ids:
            scene = scene_by_id(manifest, scene_id)
            for light in category["lights"]:
                cell = cell_id(scene, light)
                reference_path = under(references) / f"{cell}.png"
                if not reference_path.exists():
                    print(f"skip {cell}: no reference")
                    continue
                reference = load_rgb(reference_path)
                cell_out = out_root / cell
                (RAYTRACER_DIR / cell_out).mkdir(parents=True, exist_ok=True)

                directories = [under(leaf(root, cell, arm, 0, r))
                               for arm in (baseline, biased) for r in range(rounds)]
                if not any(d.exists() for d in directories):
                    print(f"skip {cell}: nothing rendered")
                    continue

                # Step 1 wants two budget points, and which two is not free: the bias is
                # supposed to show only above the crossover, so an early checkpoint and
                # the last one bracket it.
                indices = sorted({int(sidecar_for(p).get("benchmark", {}).get("checkpointIndex", 0))
                                  for d in directories if d.exists()
                                  for p in d.rglob("*.png") if not p.name.endswith(".flip.png")})
                if not indices:
                    continue
                points = sorted({indices[len(indices) // 3], indices[-1]})

                # The cell alone does not identify a section: veach-ajar--own is asked for
                # by three categories under three different arm pairs, and a reader cannot
                # tell which one a table describes from the numbers.
                print(f"  {name} {cell} ({biased} vs {baseline}): checkpoints {points}")
                lines += [f"## {name} · {cell} · {biased} vs {baseline}", ""]

                view = render_cluster_view(scene, light, manifest, cell_out)
                cluster_map = cluster_ids(load_rgb(view)) if view is not None else None

                for point in points:
                    maps = {}
                    for arm in (baseline, biased):
                        averaged, count = averaged_map(root, cell, arm, 0, rounds, reference, point)
                        if count:
                            maps[arm] = (averaged, count)
                    if baseline not in maps or biased not in maps:
                        continue

                    a_map, a_count = maps[baseline]
                    b_map, b_count = maps[biased]
                    difference = b_map - a_map
                    limit = float(np.quantile(np.abs(difference), 0.995)) or 1e-6
                    stem = f"c{point:02d}"
                    floor, floor_source = noise_floor(args, root, cell, baseline, a_map,
                                                      rounds, reference, point)
                    write_heat_png(a_map, RAYTRACER_DIR / cell_out / f"{stem}-{baseline}.png",
                                   float(np.quantile(a_map, 0.99)))
                    write_heat_png(b_map, RAYTRACER_DIR / cell_out / f"{stem}-{biased}.png",
                                   float(np.quantile(b_map, 0.99)))
                    write_diverging_png(difference,
                                        RAYTRACER_DIR / cell_out / f"{stem}-diff.png", limit)

                    worse = float((difference > floor).mean())
                    better = float((difference < -floor).mean())
                    histogram = RAYTRACER_DIR / cell_out / f"{stem}-histogram.png"
                    has_histogram = write_histogram(difference, floor, histogram)

                    lines += [f"### checkpoint {point} ({a_count} + {b_count} images)", "",
                              "| quantity | value |", "|---|---|",
                              f"| mean {baseline} | {a_map.mean():.5f} |",
                              f"| mean {biased} | {b_map.mean():.5f} |",
                              f"| mean difference | {difference.mean():+.5f} |",
                              f"| noise floor | {floor:.5f} ({floor_source}) |",
                              f"| pixels worse than floor | {worse * 100:.2f} % |",
                              f"| pixels better than floor | {better * 100:.2f} % |",
                              f"| p99.5 of the absolute difference | {limit:.5f} |", ""]
                    if has_histogram:
                        lines += [f"Distribution: `{histogram.name}` (log counts, dashed lines "
                                  f"at the noise floor).", ""]

                    # Step 2: concentrated in particular clusters, or spread evenly?
                    if cluster_map is not None and cluster_map.shape == difference.shape:
                        rows = []
                        for cluster in range(CLUSTER_COUNT):
                            mask = cluster_map == cluster
                            share = float(mask.mean())
                            if share < 0.002:  # a cluster covering nothing says nothing
                                continue
                            rows.append((float(difference[mask].mean()), cluster, share))
                        rows.sort(reverse=True)
                        if rows:
                            lines += ["Difference by cluster, worst first. Concentration rather "
                                      "than an even spread is what 8.5.1 predicts.", "",
                                      "| cluster | share of pixels | mean difference |",
                                      "|---|---|---|"]
                            for value, cluster, share in rows[:8]:
                                lines.append(f"| {cluster} | {share * 100:.1f} % | {value:+.5f} |")
                            lines.append("")
                    elif cluster_map is not None:
                        lines += ["*Cluster view resolution does not match the captures, so step 2 "
                                  "is skipped for this cell rather than aligned by resampling — a "
                                  "resampled assignment is not an assignment.*", ""]

                # Step 4: best, median and worst of the series, both arms.
                lines += ["### Best, median and worst of the series", "",
                          "| arm | rank | FLIP | error map |", "|---|---|---|---|"]
                for arm in (baseline, biased):
                    ranked = []
                    for r in range(rounds):
                        ranked += per_image_flip(under(leaf(root, cell, arm, 0, r)),
                                                 reference, points[-1])
                    ranked.sort()
                    if not ranked:
                        continue
                    for label, (value, png) in (("best", ranked[0]),
                                                ("median", ranked[len(ranked) // 2]),
                                                ("worst", ranked[-1])):
                        _, error_map = score_image(reference, png)
                        target = RAYTRACER_DIR / cell_out / f"series-{arm}-{label}.png"
                        write_heat_png(error_map, target, float(np.quantile(error_map, 0.99)))
                        lines.append(f"| {arm} | {label} | {value:.5f} | `{target.name}` |")
                lines.append("")

    out = RAYTRACER_DIR / out_root / "diffmap.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")

# ------------------------------------------------------------------ entry point

# ------------------------------------------------------------------ temporal stability

# 8.5 measures the three passes on a static scene, static camera and static light. The two
# sequence passes also switch sub-pixel jitter off: only the guided integrator jitters, so
# leaving it on would make a frame-to-frame difference metric report the jitter on one arm
# and nothing on the other. The independent pass leaves it on, because those frames are
# scored against a reference that was rendered with it.
NO_JITTER = ["vxpg.vbufferJitter=0"]

TEMPORAL_PASSES = ("accumulated", "raw", "independent", "equalsample")


def temporal_leaf(root, cell, arm, pass_name):
    return root / cell / arm / pass_name


def run_temporal(args, manifest):
    """One accumulated sequence, one unaccumulated sequence and N independent frames."""
    root = Path(args.out)
    (RAYTRACER_DIR / root).mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    cells = [(cell, scene, light) for cell, scene, light in campaign_cells(manifest, args.only)]
    print(f"{len(cells)} cell(s) x {len(arms)} arm(s) x 3 passes")

    failed = 0
    for cell, scene, light in cells:
        for arm in arms:
            spec = manifest["arms"][arm]
            recipes = {
                # M1/M2 with history: the error falls as the history grows and the rate of
                # that fall is the result, not the level.
                "accumulated": (f"frames:{args.frames}", "every:1", 1, spec["cvars"] + NO_JITTER),
                # The same pair of metrics with no history at all: a flat line whose height
                # is the flicker amplitude an interactive viewer would see.
                "raw": ("frames:1", None, args.frames, spec["cvars"] + NO_JITTER),
                # S3 and S4 both read this one: per-pixel spread across the set, and the
                # spread of the per-frame error level over the same set.
                "independent": ("frames:1", None, args.independent, list(spec["cvars"])),
                # M6 of the evaluation plan: a sequence of images at EQUAL SAMPLE COUNT,
                # which is a different number of frames per arm - the guided arm draws two
                # path samples per iteration where path tracing draws one. Each image is a
                # fresh accumulation of that many frames, so the series is comparable
                # image-for-image across the arms and not frame-for-frame.
                "equalsample": (f"frames:{max(1, args.samples // spec.get('samplesPerFrame', 1))}",
                                None, args.images, spec["cvars"] + NO_JITTER),
            }
            for pass_name, (budget, checkpoints, images, cvars) in recipes.items():
                out = temporal_leaf(root, cell, arm, pass_name)
                if (under(out) / "done.json").exists():
                    continue
                label = f"{cell} {arm} {pass_name}"
                print(f"  {label} ...", flush=True)
                code, took = engine_checked(
                    scene, light, manifest, spec["technique"], cvars, out, budget, images,
                    args.warmup, checkpoints=checkpoints,
                    log_path=under(root) / f"{cell}-{arm}-{pass_name}.log",
                    config=config_path(manifest, scene, light))
                if code != 0:
                    print(f"  FAILED {label} exit={code}")
                    failed += 1
                    continue
                (under(out) / "done.json").write_text(json.dumps(
                    {"seconds": took, "cell": cell, "arm": arm, "pass": pass_name,
                     "images": images, "budget": budget}))
                print(f"  ok     {label} in {took:.0f}s")
    print(f"\n{failed} failed")


def temporal_rows(root, cell, arm, reference):
    """The three passes of one arm, reduced to the numbers S1-S4 are made of."""
    accumulated = sequence_metrics(under(temporal_leaf(root, cell, arm, "accumulated")))
    raw = sequence_metrics(under(temporal_leaf(root, cell, arm, "raw")))
    independent = under(temporal_leaf(root, cell, arm, "independent"))
    frames = homogeneity_metrics(independent, reference) if reference is not None else []
    return {
        "accumulated": accumulated,
        "raw": raw,
        "flicker": flicker_stats(independent, reference),
        "frames": frames,
        "flipSpread": spread([f["flipMean"] for f in frames]) if frames else {},
        "rmaeSpread": spread([f["rmae"] for f in frames]) if frames else {},
    }


def write_stability_csv(args, manifest):
    """wyniki-stabilnosc/stabilnosc.csv: cell,arm,index,flipMean,mse, in capture order.

    Order is the content here - the chart is error against image NUMBER - so the rows are
    sorted by the image index the engine wrote into the file name, never by directory
    listing order.
    """
    root = Path(args.out)
    references = Path(args.reference_dir)
    rows = ["cell,arm,index,flipMean,mse"]
    for cell, _scene, _light in campaign_cells(manifest, args.only):
        reference_path = under(references) / f"{cell}.png"
        if not reference_path.exists():
            continue
        reference = load_rgb(reference_path)
        for arm in args.arms.split(","):
            directory = under(temporal_leaf(root, cell, arm, "equalsample"))
            if not directory.exists():
                continue
            captures = [png for png in sorted(directory.rglob("*.png"))
                        if not png.name.endswith(".flip.png")]
            for index, png in enumerate(captures):
                scores, _ = score_image(reference, png)
                rows.append(f"{cell},{arm},{index},{scores['flipMean']:.6f},{scores['mse']:.8f}")
    out = under(root) / "stabilnosc.csv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"written: {out} ({len(rows) - 1} row(s))")


def report_temporal(args, manifest):
    root = Path(args.out)
    references = Path(args.reference_dir)
    arms = args.arms.split(",")
    data = {}
    for cell, _scene, _light in campaign_cells(manifest, args.only):
        reference_path = under(references) / f"{cell}.png"
        reference = load_rgb(reference_path) if reference_path.exists() else None
        for arm in arms:
            if not (under(temporal_leaf(root, cell, arm, "raw"))).exists():
                continue
            data.setdefault(cell, {})[arm] = temporal_rows(root, cell, arm, reference)
    if not data:
        sys.exit("Nothing to score. Run 'temporal' first.")

    lines = ["# Temporal stability (K4, PLAN_BADAWCZY 8.5)", "",
             f"Build: {recon.EXE.parent.name}. Static scene, static camera, static light. "
             f"Sub-pixel jitter is off in both sequence passes and on in the independent "
             f"pass. {args.frames}-frame sequences, {args.independent} independent frames.",
             reference_note(args), "",
             "## S1 and S2 - consecutive-frame difference", "",
             "`S1` is the mean luminance of the absolute difference between consecutive "
             "frames, `S2` the mean squared difference over the same pair. Accumulated: the "
             "level falls as history builds, so the pair reported is the first and the last "
             "of the sequence and the ratio between them is the rate. Unaccumulated: a flat "
             "line whose height is the flicker amplitude itself.", "",
             "| cell | arm | S1 acc. first | S1 acc. last | fall | S1 raw | S2 raw |",
             "|---|---|---|---|---|---|---|"]
    for cell, arms_data in sorted(data.items()):
        for arm, row in sorted(arms_data.items()):
            acc, raw = row["accumulated"], row["raw"]
            if not acc or not raw:
                continue
            first, last = acc[0]["temporalError"], acc[-1]["temporalError"]
            fall = f"{first / last:.1f}x" if last > 0 else "—"
            raw_s1 = statistics.fmean([r["temporalError"] for r in raw])
            raw_s2 = statistics.fmean([r["temporalMse"] for r in raw])
            lines.append(f"| {cell} | {arm} | {first:.6f} | {last:.6f} | {fall} | "
                         f"{raw_s1:.6f} | {raw_s2:.6f} |")

    lines += ["", "## S3 - per-pixel spread across independent frames", "",
              "The deviation alone inverts the ranking, so it never travels alone: these "
              "images are clamped, and an estimator whose single sample misses the light "
              "writes the same black pixel every frame - perfect stability, entirely wrong. "
              "`static` is the share of pixels that never moved and `level` the series' own "
              "brightness against the reference's; a low deviation counts only when both say "
              "the pixels were alive. `equal-time` divides by the square root of the frames "
              "one second of averaging buys at that arm's frame cost.", "",
              "| cell | arm | std | midtones | p95 | static | level | ref level | equal-time |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cell, arms_data in sorted(data.items()):
        for arm, row in sorted(arms_data.items()):
            flicker = row["flicker"]
            if not flicker:
                continue
            done = under(temporal_leaf(root, cell, arm, "independent")) / "done.json"
            per_frame = json.loads(done.read_text())["seconds"] / max(flicker["images"], 1) \
                if done.exists() else 0.0
            equal = flicker["stdMean"] * math.sqrt(per_frame) if per_frame > 0 else 0.0
            lines.append(f"| {cell} | {arm} | {flicker['stdMean']:.5f} | "
                         f"{flicker.get('stdMidtones', 0.0):.5f} | {flicker['stdP95']:.5f} | "
                         f"{flicker['staticFraction'] * 100:.1f} % | {flicker['meanLevel']:.4f} | "
                         f"{flicker.get('referenceLevel', 0.0):.4f} | {equal:.5f} |")

    lines += ["", "## S4 - homogeneity of the per-frame error level (P4)", "",
              "Coefficient of variation of the per-frame error over the independent frames. "
              "Path tracing is the control: it has no clustering, so its coefficient is what "
              "Monte Carlo noise alone produces at this budget. The paper's own sentence - "
              "different voxel clustering leading to different amounts of variance across "
              "frames - is what this table either finds or does not.", "",
              "| cell | arm | FLIP mean | FLIP cv | FLIP spread | RMAE mean | RMAE cv | n |",
              "|---|---|---|---|---|---|---|---|"]
    for cell, arms_data in sorted(data.items()):
        control = arms_data.get(args.arms.split(",")[0], {}).get("flipSpread", {})
        for arm, row in sorted(arms_data.items()):
            flip, rmae = row["flipSpread"], row["rmaeSpread"]
            if not flip:
                continue
            versus = (f" ({flip['cvPercent'] / control['cvPercent']:.1f}x control)"
                      if control.get("cvPercent") and arm != args.arms.split(",")[0] else "")
            lines.append(f"| {cell} | {arm} | {flip['mean']:.5f} | "
                         f"{flip['cvPercent']:.2f} %{versus} | {flip['spreadPercent']:.2f} % | "
                         f"{rmae.get('mean', 0.0):.5f} | {rmae.get('cvPercent', 0.0):.2f} % | "
                         f"{flip['n']} |")

    out = under(root) / "temporal.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")
    raw_out = under(root) / "temporal.json"
    raw_out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"written: {raw_out}")


# ------------------------------------------------------------------ frame cost breakdown

# "    <node name>   0.1234 ms mean   0.5678 ms max   n=1234", as Renderer.cpp formats it -
# but every line in the log also carries spdlog's own "[timestamp] [info] " in front of it.
LOG_PREFIX = re.compile(r"^\[[^\]]*\]\s*\[[^\]]*\]\s?")
TIMING_LINE = re.compile(r"^\s+(\S.*?)\s{2,}([\d.]+) ms mean\s+([\d.]+) ms max\s+n=(\d+)\s*$")


def parse_timings(log_path):
    """The last node-cost block in a log, as {node: (mean, max, executions)}.

    The last one, not the first: the dump is taken after the captures, when every root
    signature has been built and the table covers the whole capture window instead of a
    cold first frame.
    """
    blocks, current = [], None
    for raw in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = LOG_PREFIX.sub("", raw)
        if "[RDG] node GPU cost:" in line:
            current = {}
            blocks.append(current)
            continue
        if current is None:
            continue
        match = TIMING_LINE.match(line)
        if match:
            name = match.group(1).strip()
            current[name] = (float(match.group(2)), float(match.group(3)), int(match.group(4)))
        elif line.strip() and not line.startswith(" "):
            current = None
    return blocks[-1] if blocks else {}


def per_frame_cost(timings):
    """{node: (per-execution mean, contribution to the average frame, executions)}.

    Not every node runs every frame, and the difference is not a detail: the voxel bake ran
    ONCE in eleven thousand frames at 24 ms. Charged to every frame it would swamp a 4 ms
    one and make the residual meaningless; charged at the rate it actually ran it is what it
    is, a build cost amortised to nearly nothing.
    """
    executions = [value[2] for value in timings.values()]
    per_frame = max(executions) if executions else 0
    if not per_frame:
        return {}
    return {name: (mean, mean * count / per_frame, count)
            for name, (mean, _peak, count) in timings.items()}


def run_breakdown(args, manifest):
    """One short timed run per cell and arm, then one table per cell."""
    root = Path(args.out)
    (RAYTRACER_DIR / root).mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    rows = {}

    for cell, scene, light in campaign_cells(manifest, args.only):
        for arm in arms:
            spec = manifest["arms"][arm]
            out = root / cell / arm
            log_path = under(root) / f"{cell}-{arm}.log"
            if not (under(out) / "done.json").exists():
                print(f"  {cell} {arm} ...", flush=True)
                code, took = engine_checked(
                    scene, light, manifest, spec["technique"], spec["cvars"], out,
                    f"seconds:{args.seconds}", 1, args.warmup, log_path=log_path,
                    config=config_path(manifest, scene, light), rdg_timings=True)
                if code != 0:
                    print(f"  FAILED {cell} {arm} exit={code}")
                    continue
                (under(out) / "done.json").write_text(json.dumps({"seconds": took}))
                print(f"  ok     {cell} {arm} in {took:.0f}s")
            timings = parse_timings(log_path)
            frame_ms = 0.0
            for png in under(out).rglob("*.png"):
                frame_ms = sidecar_for(png).get("benchmark", {}).get("meanFrameMs", 0.0)
                break
            if timings:
                rows.setdefault(cell, {})[arm] = (timings, frame_ms)

    lines = ["# Frame cost by graph node (8.4, P2)", "",
             f"Build: {recon.EXE.parent.name}. Mean GPU cost of every render-graph node over "
             f"a {args.seconds} s window after a {args.warmup} s warm-up, as the engine's own "
             f"`--rdg-timings` reports it.", "",
             "`ms` is the cost of one execution. `per frame` is that cost charged at the rate "
             "the node actually ran, because not every node runs every frame - the voxel bake "
             "runs once for the whole session, and charging it to every frame would swamp the "
             "table with a build cost. `unattributed` is the measured frame time minus the sum "
             "of the per-frame column: submission, presentation and whatever gap the graph does "
             "not own. It is a residual, not a node, and a large one is a question rather than "
             "a result.", ""]
    for cell, arms_data in sorted(rows.items()):
        costs = {arm: per_frame_cost(arms_data.get(arm, ({}, 0.0))[0]) for arm in arms}
        frames = [arms_data.get(arm, ({}, 0.0))[1] for arm in arms]
        lines += [f"## {cell}", "",
                  "| node | " + " | ".join(f"{a} ms" for a in arms) + " | "
                  + " | ".join(f"{a} per frame" for a in arms) + " | "
                  + " | ".join(f"{a} %" for a in arms) + " |",
                  "|---" * (1 + 3 * len(arms)) + "|"]

        names = []
        for arm in arms:
            for name in costs.get(arm, {}):
                if name not in names:
                    names.append(name)
        names.sort(key=lambda n: -max(costs.get(a, {}).get(n, (0, 0, 0))[1] for a in arms))

        for name in names:
            per_execution, amortised, shares = [], [], []
            for arm, frame_ms in zip(arms, frames):
                entry = costs.get(arm, {}).get(name)
                if entry is None:
                    per_execution.append("\u2014")
                    amortised.append("\u2014")
                    shares.append("\u2014")
                    continue
                mean, contribution, count = entry
                once = " (x1)" if count <= 1 else ""
                per_execution.append(f"{mean:.4f}{once}")
                amortised.append(f"{contribution:.4f}")
                shares.append(f"{contribution / frame_ms * 100:.1f}" if frame_ms > 0 else "\u2014")
            lines.append(f"| {name} | " + " | ".join(per_execution) + " | "
                         + " | ".join(amortised) + " | " + " | ".join(shares) + " |")

        totals = [sum(entry[1] for entry in costs.get(arm, {}).values()) for arm in arms]
        blank = " | ".join("" for _ in arms)
        lines += ["| **sum of nodes** | " + blank + " | "
                  + " | ".join(f"**{t:.4f}**" for t in totals) + " | "
                  + " | ".join(f"**{t / f * 100:.1f}**" if f > 0 else "\u2014"
                               for t, f in zip(totals, frames)) + " |",
                  "| **frame** | " + blank + " | "
                  + " | ".join(f"**{f:.4f}**" for f in frames) + " | "
                  + " | ".join("**100.0**" if f > 0 else "\u2014" for f in frames) + " |",
                  "| **unattributed** | " + blank + " | "
                  + " | ".join(f"{f - t:+.4f}" for t, f in zip(totals, frames)) + " | "
                  + " | ".join(f"{(f - t) / f * 100:+.1f}" if f > 0 else "\u2014"
                               for t, f in zip(totals, frames)) + " |", ""]

    out_path = under(root) / "breakdown.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")


# ------------------------------------------------------------------ memory against the grid

MIB = 1024.0 ** 2


def voxel_metres(log_path):
    """The absolute voxel size the run used, in scene metres.

    The resolution alone does not compare across scenes: 64^3 is a 0.3 m cell in a bedroom
    and an 8 m cell in a temple, and it is the metres that decide what the grid can tell
    apart. The engine logs it once per grid resize."""
    if not Path(log_path).exists():
        return None
    found = None
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"voxelSize=([\d.]+)", line)
        if match:
            found = float(match.group(1))
    return found


def run_memory(args, manifest):
    """The guided arm at every grid rung its scene can take, for footprint and frame cost.

    Path tracing has no grid and therefore no curve here; it appears in the report only as
    the frame cost the guided arm is paying its footprint against. The rungs are per scene
    because the leaf cap bites at a different resolution in each (R17/R18), and a rung above
    the cap would measure the cap.
    """
    root = Path(args.out)
    (RAYTRACER_DIR / root).mkdir(parents=True, exist_ok=True)
    references = Path(args.reference_dir)
    rows = {}

    for cell, scene, light in campaign_cells(manifest, args.only):
        spec = manifest["arms"][args.arm]
        for resolution in recon.grid_resolutions(scene, manifest):
            out = root / cell / f"g{resolution}"
            if not (under(out) / "done.json").exists():
                print(f"  {cell} {resolution}^3 ...", flush=True)
                code, took = engine_checked(
                    scene, light, manifest, spec["technique"],
                    list(spec["cvars"]) + [f"voxel.gridDim={resolution}"], out,
                    f"seconds:{args.seconds}", 1, args.warmup,
                    log_path=under(root) / f"{cell}-g{resolution}.log",
                    config=config_path(manifest, scene, light))
                if code != 0:
                    print(f"  FAILED {cell} {resolution}^3 exit={code}")
                    continue
                (under(out) / "done.json").write_text(
                    json.dumps({"seconds": took, "gridDim": resolution}))
                print(f"  ok     {cell} {resolution}^3 in {took:.0f}s")

            captures = [png for png in sorted(under(out).rglob("*.png"))
                        if not png.name.endswith(".flip.png")]
            if not captures:
                continue
            sidecar = sidecar_for(captures[0])
            benchmark = sidecar.get("benchmark", {})
            reference_path = under(references) / f"{cell}.png"
            flip = None
            if reference_path.exists():
                flip = score_image(load_rgb(reference_path), captures[0])[0]["flipMean"]
            rows.setdefault(cell, {})[resolution] = {
                "memory": benchmark.get("memory", {}),
                # M8 asks for the footprint of the WHOLE method: the per-stage inventory
                # above cannot see the scene buffers, the textures or the driver-side
                # acceleration structures, and this can.
                "videoMemoryBytes": benchmark.get("videoMemoryBytes", 0),
                "adapter": benchmark.get("adapter", ""),
                "voxelMetres": voxel_metres(under(root) / f"{cell}-g{resolution}.log"),
                "ms": benchmark.get("meanFrameMs", 0.0),
                "frames": sidecar.get("raytracing", {}).get("frameIndex", 0),
                "flip": flip,
            }

    # Path tracing has no grid, so it is one run per cell rather than a curve - but M8
    # asks for its total as the point the guided total is read against, and without it the
    # guided figure says nothing about what the method costs OVER path tracing.
    baseline = {}
    for cell, scene, light in campaign_cells(manifest, args.only):
        spec = manifest["arms"][args.baseline_arm]
        out = root / cell / "baseline"
        if not (under(out) / "done.json").exists():
            print(f"  {cell} {args.baseline_arm} ...", flush=True)
            code, took = engine_checked(
                scene, light, manifest, spec["technique"], list(spec["cvars"]), out,
                f"seconds:{args.seconds}", 1, args.warmup,
                log_path=under(root) / f"{cell}-baseline.log",
                config=config_path(manifest, scene, light))
            if code != 0:
                print(f"  FAILED {cell} baseline exit={code}")
                continue
            (under(out) / "done.json").write_text(json.dumps({"seconds": took}))
        captures = [png for png in sorted(under(out).rglob("*.png"))
                    if not png.name.endswith(".flip.png")]
        if captures:
            benchmark = sidecar_for(captures[0]).get("benchmark", {})
            baseline[cell] = {"videoMemoryBytes": benchmark.get("videoMemoryBytes", 0),
                              "ms": benchmark.get("meanFrameMs", 0.0)}

    lines = ["# Guiding chain footprint against grid resolution (8.6, P5)", "",
             f"Build: {recon.EXE.parent.name}. Guided arm only - path tracing has no grid. "
             f"One {args.seconds} s run per rung after a {args.warmup} s warm-up; the footprint "
             f"is the engine's own per-stage accounting, taken from the capture's sidecar, and "
             f"the frame cost is the same run's mean. The rung ladder is per scene: above a "
             f"scene's cap the measurement would report the cap and not the grid.",
             reference_note(args), ""]
    for cell, rungs in sorted(rows.items()):
        stages = []
        for entry in rungs.values():
            for stage in entry["memory"].get("byStage", {}):
                if stage not in stages:
                    stages.append(stage)
        stages.sort(key=lambda st: -max(r["memory"].get("byStage", {}).get(st, 0)
                                        for r in rungs.values()))
        lines += [f"## {cell}", "",
                  "| stage | " + " | ".join(f"{r}^3 MiB" for r in sorted(rungs)) + " |",
                  "|---" * (1 + len(rungs)) + "|"]
        for stage in stages:
            cells_out = []
            for resolution in sorted(rungs):
                value = rungs[resolution]["memory"].get("byStage", {}).get(stage)
                cells_out.append(f"{value / MIB:.2f}" if value else "\u2014")
            lines.append(f"| {stage} | " + " | ".join(cells_out) + " |")
        totals = [rungs[r]["memory"].get("totalBytes", 0) / MIB for r in sorted(rungs)]
        lines += ["| **inwentarz razem** | " + " | ".join(f"**{t:.2f}**" for t in totals) + " |",
                  "| **cała metoda (karta)** | " + " | ".join(
                      (f"**{rungs[r].get('videoMemoryBytes', 0) / MIB:.0f}**"
                       if rungs[r].get("videoMemoryBytes") else "—")
                      for r in sorted(rungs)) + " |",
                  "| rozmiar woksela [m] | " + " | ".join(
                      (f"{rungs[r]['voxelMetres']:.3f}" if rungs[r].get("voxelMetres") else "—")
                      for r in sorted(rungs)) + " |",
                  "| ms/frame | " + " | ".join(f"{rungs[r]['ms']:.3f}"
                                               for r in sorted(rungs)) + " |",
                  "| FLIP | " + " | ".join(
                      ("\u2014" if rungs[r]["flip"] is None else f"{rungs[r]['flip']:.5f}")
                      for r in sorted(rungs)) + " |", ""]
        if cell in baseline:
            lines += [f"Punkt odniesienia, {args.baseline_arm}: "
                      f"{baseline[cell]['videoMemoryBytes'] / MIB:.0f} MiB na karcie, "
                      f"{baseline[cell]['ms']:.3f} ms/klatkę.", ""]

    out_path = under(root) / "memory.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")
    raw = under(root) / "memory.json"
    raw.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"written: {raw}")


# ------------------------------------------------------- phase 2: the testing parameters

# Budgets a chapter is willing to print, smallest first. The budget is chosen from this
# list rather than quoted to the millisecond, because a budget stated as 223 ms invites the
# reader to believe the third digit means something.
ROUND_BUDGETS_MS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 45, 50, 60,
                    70, 80, 90, 100, 120, 150, 180, 200, 230, 250, 280, 300, 350, 400, 450,
                    500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 5000]


def time_spent(budget_ms, frame_ms):
    """Frames rendered and time ACTUALLY spent under the engine's stopping rule.

    The capture triggers on the first frame whose accumulated time reaches the budget, so an
    arm overshoots by up to one frame rather than stopping short: an arm costing 25.5 ms
    handed a 32 ms budget renders two frames and spends 51 ms. At short budgets that
    overshoot is a large share of the budget AND it differs between the arms, which is what
    can make a nominally equal-time comparison unequal.
    """
    frames = max(1, math.ceil(budget_ms / frame_ms))
    return frames, frames * frame_ms


def run_budget(args, manifest):
    """Frame cost of both arms per scene, and the equal-time budget that follows from it.

    The budget is the one at which the two arms accumulate a SIMILAR NUMBER OF FRAMES on
    as many scenes as possible - the plan's condition - and it is one value for the whole
    set, not one per scene. Rounded up.

    The same run writes the per-cell frame cost that the equal-variance reading needs: its
    own run saves every frame and its clock is inflated by that, so the honest time for a
    frame index has to come from a sparse run like this one.
    """
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    parameters = load_parameters(args.parameters)
    arms = args.arms.split(",")
    costs = {}

    for cell, scene, light in campaign_cells(manifest, args.only):
        for arm in arms:
            spec = manifest["arms"][arm]
            out = root / cell / arm
            if not (under(out) / "done.json").exists():
                print(f"  {cell} {arm} ...", flush=True)
                code, took = engine_checked(
                    scene, light, manifest, spec["technique"],
                    list(spec["cvars"]) + scene_cvars(cell, parameters), out,
                    f"seconds:{args.seconds}", args.images, args.warmup,
                    log_path=under(root) / f"{cell}-{arm}.log",
                    config=config_path(manifest, scene, light))
                if code != 0:
                    print(f"  FAILED {cell} {arm} exit={code}")
                    continue
                (under(out) / "done.json").write_text(json.dumps({"seconds": took}))
            values = []
            for png in sorted(under(out).rglob("*.png")):
                if png.name.endswith(".flip.png"):
                    continue
                values.append(sidecar_for(png).get("benchmark", {}).get("meanFrameMs", 0.0))
            if values:
                costs.setdefault(cell, {})[arm] = statistics.fmean(values)

    if len(arms) != 2:
        sys.exit("The equal-time budget is defined between exactly two arms")
    first, second = arms

    def mismatch(budget_ms):
        """Per-cell gap between the two arms' ACTUALLY SPENT time at one budget.

        Not a frame-count gap: at equal time the frame counts differ by exactly the frame
        cost ratio (2.4-6.1x here), so asking them to agree asks the arms to cost the same,
        which is the thing being measured. What has to agree is the time.
        """
        rows = []
        for cell, per_arm in sorted(costs.items()):
            if first not in per_arm or second not in per_arm:
                continue
            n1, t1 = time_spent(budget_ms, per_arm[first])
            n2, t2 = time_spent(budget_ms, per_arm[second])
            rows.append((cell, per_arm[first], per_arm[second], n1, t1, n2, t2,
                         abs(t1 - t2) / max(t1, t2)))
        return rows

    # The SMALLEST budget whose worst-case time gap clears the tolerance. Smallest, because
    # the chapter is about real-time rendering and a budget is only as useful as it is short;
    # worst-case rather than average, because one cell measured unequally is enough to make
    # the table wrong.
    chosen_ms = None
    for budget_ms in ROUND_BUDGETS_MS:
        rows = mismatch(budget_ms)
        if rows and max(row[7] for row in rows) <= args.tolerance:
            chosen_ms = budget_ms
            break
    if chosen_ms is None:
        sys.exit(f"No budget up to {ROUND_BUDGETS_MS[-1]} ms keeps both arms within "
                 f"{100 * args.tolerance:.0f} % of each other - raise --tolerance")
    rows = mismatch(chosen_ms)
    within = sum(1 for row in rows if row[7] <= args.tolerance)

    lines = ["# Parametry testowania (faza 2)", "",
             f"Koszt klatki zmierzony przy budzecie {args.seconds} s, {args.images} "
             f"obraz(y) na punkt, rozgrzewka {args.warmup} s, nastawy z `parameters.json`. "
             f"Zapis rzadki - gesty zapis zawyza koszt klatki i nie opisuje techniki.", "",
             f"## Budzet rownego czasu: **{chosen_ms} ms**", "",
             "Wybrany jako NAJMNIEJSZY budzet, przy ktorym oba ramiona zuzywaja zblizony "
             "FAKTYCZNY czas. Silnik przerywa na pierwszej klatce siegajacej budzetu, wiec "
             "kazde ramie przestrzeliwuje o niepelna klatke, a przy krotkim budzecie te "
             "przestrzelenia sa rozne. Liczby klatek roznia sie o stosunek kosztow klatki i "
             "tak ma byc - to jest mierzona wielkosc, nie wada pomiaru.", "",
             f"Rozjazd czasu ponizej {100 * args.tolerance:.0f} % na {within} z {len(rows)} scen.", "",
             f"| scena | {first} [ms] | {second} [ms] | N({first}) | czas {first} | "
             f"N({second}) | czas {second} | rozjazd |",
             "|---|---|---|---|---|---|---|---|"]
    for cell, ms1, ms2, n1, t1, n2, t2, diff in rows:
        lines.append(f"| {cell} | {ms1:.3f} | {ms2:.3f} | {n1} | {t1:.1f} ms | {n2} | "
                     f"{t2:.1f} ms | {100 * diff:.1f} % |")

    lines += ["", "## Koszt klatki na scene i ramie", "",
              "| scena | " + " | ".join(arms) + " |", "|---" * (1 + len(arms)) + "|"]
    for cell, per_arm in sorted(costs.items()):
        lines.append(f"| {cell} | " + " | ".join(f"{per_arm.get(a, float('nan')):.3f}"
                                                 for a in arms) + " |")

    out_md = under(root) / "parametry-testowania.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {out_md}")

    cost_json = under(root) / "koszt-klatki.json"
    cost_json.write_text(json.dumps(costs, indent=1), encoding="utf-8")
    print(f"written: {cost_json}")

    # Written back into the parameters file so one document carries both halves of the
    # run configuration: what the technique is set to, and what the test is set to.
    if args.parameters:
        path = Path(args.parameters)
        if not path.is_absolute():
            path = REPO_ROOT / path
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        payload["testing"] = {
            "equalTimeBudgetMs": chosen_ms,
            "frameCostMs": costs,
            "tolerance": args.tolerance,
            "_criterion": "smallest budget whose worst-case gap between the arms' ACTUALLY "
                          "SPENT time clears the tolerance; frame counts differ by the frame "
                          "cost ratio and are not asked to agree",
            "_note": "The sample count and the MSE ceiling are added by their own steps.",
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"updated: {path}")


# ------------------------------------------------------- one shape for every measurement

def head_commit():
    """The commit the engine was built from, for the header of every result file."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def measured_adapter(root):
    """Which card produced these captures, read from a sidecar rather than assumed.

    Two machines run this harness and their numbers must never be mixed up; the engine
    writes the adapter name into every capture, so the result file can state it as fact.
    """
    for sidecar in sorted(Path(root).rglob("*.json")):
        if sidecar.name == "done.json":
            continue
        try:
            adapter = json.loads(sidecar.read_text(encoding="utf-8-sig")) \
                .get("benchmark", {}).get("adapter", "")
        except (OSError, ValueError):
            continue
        if adapter:
            return adapter
    return "unknown"


def flatten_for_schema(data):
    """gather()'s nesting reduced to the plan's own shape: cell -> arm -> repeat -> point.

    The last checkpoint of each pass, because that is the image the tables quote; the whole
    ladder stays in the curves file, which is what the plots read.
    """
    out = {}
    for cell, arms in data.items():
        for arm, passes in arms.items():
            for repeat, points in passes.items():
                if not points:
                    continue
                entry = points[max(points)]
                flip = {"mean": entry["flip"].get("mean")}
                flip["median"] = entry.get("flipMedian", {}).get("mean")
                flip["q1"] = entry.get("flipQ1", {}).get("mean")
                flip["q3"] = entry.get("flipQ3", {}).get("mean")
                flip["p95"] = entry.get("flipP95", {}).get("mean")
                flip["min"] = entry.get("flipMin", {}).get("mean")
                flip["max"] = entry.get("flipMax", {}).get("mean")
                flip["wmedian"] = entry.get("wmedian", {}).get("mean")
                flip["ci95"] = entry["flip"].get("ci95")
                out.setdefault(cell, {}).setdefault(arm, {})[str(repeat)] = {
                    "ms": entry["ms"].get("mean"),
                    "frames": entry["frames"],
                    "seconds": entry["seconds"],
                    "flip": flip,
                    "mse": entry["mse"].get("mean"),
                }
    return out


def write_measurement_json(path, measurement, data, args, parameters=None):
    """The machine-readable half of a result directory, in the shape plan 7.2 fixes.

    One shape for every measurement, so the plotting scripts do not have to know which
    measurement they are reading. Every number a chapter quotes has to be findable here;
    a number that is not in this file was not measured, and the chapter says so rather
    than deriving it.
    """
    payload = {
        "measurement": measurement,
        "gpu": measured_adapter(under(Path(args.out))),
        "build": recon.EXE.parent.name,
        "commit": head_commit(),
        "date": time.strftime("%Y-%m-%d"),
        "flip": flip_settings(),
        "parameters": parameters or {},
        "data": data,
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"written: {path}")


def write_measurement_csv(path, data):
    """The same numbers, flat, for the plots."""
    rows = ["cell,arm,repeat,ms,frames,seconds,flipMean,flipMedian,flipQ1,flipQ3,flipP95,mse"]
    for cell, arms in sorted(data.items()):
        for arm, repeats in sorted(arms.items()):
            for repeat, entry in sorted(repeats.items()):
                flip = entry["flip"]
                rows.append(",".join(str(v) for v in [
                    cell, arm, repeat, entry["ms"], entry["frames"], entry["seconds"],
                    flip["mean"], flip["median"], flip["q1"], flip["q3"], flip["p95"],
                    entry["mse"]]))
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"written: {path} ({len(rows) - 1} row(s))")


# ------------------------------------------------------- the image matrices (plan 7.3)

# The file name the chapter's LaTeX expects, per arm. Kept here rather than derived from
# the arm name because the thesis never uses the arm names and these two spellings must
# not drift apart.
MATRIX_NAMES = {"BSDF": "pt", "WIE": "vxpg", "WIE-R": "vxpg-r"}


def run_matrix(args, manifest):
    """One row of images per scene: reference, path tracing, guided - at 1:1, uncropped.

    Copied rather than re-rendered: these are the SAME images the table above them was
    scored from, and re-rendering would put a different sample of the estimator under a
    caption that quotes the table's numbers.
    """
    root = Path(args.out)
    references = Path(args.reference_dir)
    out_dir = under(root / "macierz")
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = {}

    for cell, _scene, _light in campaign_cells(manifest, args.only):
        reference_path = under(references) / f"{cell}.png"
        if not reference_path.exists():
            print(f"  {cell}: no reference, skipped")
            continue
        shutil.copyfile(reference_path, out_dir / f"{cell}-ref.png")
        reference = load_rgb(reference_path)
        for arm in args.arms.split(","):
            captures = final_checkpoint_captures(under(leaf(root, cell, arm, 0, 0)))
            if not captures:
                print(f"  {cell} {arm}: no capture, skipped")
                continue
            # The first image of the first round, not the best of the set: a matrix that
            # showed each arm's luckiest image would flatter both arms unequally.
            png = sorted(captures)[0]
            target = out_dir / f"{cell}-{MATRIX_NAMES.get(arm, arm.lower())}.png"
            shutil.copyfile(png, target)
            scores, _ = score_image(reference, png)
            captions[target.name] = {
                "cell": cell, "arm": arm,
                "mse": scores["mse"], "flipMean": scores["flipMean"],
                "source": png.relative_to(RAYTRACER_DIR).as_posix(),
            }
            print(f"  {cell} {arm}: FLIP {scores['flipMean']:.5f}, MSE {scores['mse']:.3e}")

    (out_dir / "podpisy.json").write_text(json.dumps(captions, indent=1), encoding="utf-8")
    print(f"written: {out_dir / 'podpisy.json'}")


# ------------------------------------------------------- phase 1: per-scene parameters


def sweep_leaf(root, cell, name, value):
    return Path(root) / cell / name.replace(".", "-") / str(value)


def run_sweep(args, manifest):
    """One factor at a time, per scene, guided arm only - phase 1 of the evaluation plan.

    Single-factor and not the cartesian product, and that is a decision rather than a
    shortcut: the product costs days and the goal is a good setting, not a provably optimal
    one. The chapter has to say so, because the two are different claims.

    Path tracing has no setting swept here - its only parameter is the bounce count, which
    both arms share - so the tuning is one-sided, and that goes in the chapter too.
    """
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    references = Path(args.reference_dir)
    spec = manifest["arms"][args.arm]
    sweeps = manifest["sweeps"]
    results = {}

    # A second pass holds the OTHER factors at what the first pass chose instead of at the
    # engine defaults. It matters where a default sits outside a scene's own admissible
    # range: San Miguel's guide accepts 0.1 % of its samples at the default 64^3, so a first
    # pass measures its remaining factors in a regime that scene never runs in.
    base = load_parameters(args.base) if args.base else {}

    for cell, scene, light in campaign_cells(manifest, args.only):
        reference_path = under(references) / f"{cell}.png"
        if not reference_path.exists():
            sys.exit(f"No reference for {cell} at {reference_path}. The sweep scores against "
                     f"the SAME reference as the measurement, so references come first.")
        reference = load_rgb(reference_path)
        for sweep in sweeps:
            name = sweep["cvar"]
            # The grid ladder is a property of the SCENE, not of the sweep: a rung above a
            # scene's lit-voxel cap measures the cap, and the ladders were set per scene in
            # the scene manifest for exactly that reason.
            values = (recon.grid_resolutions(scene, manifest) if name == "voxel.gridDim"
                      else sweep["values"])
            for value in values:
                out = sweep_leaf(root, cell, name, value)
                marker = under(out) / "done.json"
                if not marker.exists():
                    pinned = [assignment for assignment in scene_cvars(cell, base)
                              if not assignment.startswith(name + "=")]
                    print(f"  {cell} {name}={value}"
                          + (f" [{' '.join(pinned)}]" if pinned else "") + " ...", flush=True)
                    code, took = engine_checked(
                        scene, light, manifest, spec["technique"],
                        list(spec["cvars"]) + pinned + [f"{name}={value}"], out,
                        f"seconds:{args.seconds}", args.images, args.warmup,
                        log_path=under(root) / f"{cell}-{name}-{value}.log",
                        config=config_path(manifest, scene, light))
                    under(out).mkdir(parents=True, exist_ok=True)
                    if code != 0:
                        marker.write_text(json.dumps({"failed": True, "exit": code}))
                        print(f"  FAILED {cell} {name}={value} exit={code}")
                        continue
                    marker.write_text(json.dumps({"seconds": took}))
                done = json.loads(marker.read_text())
                if done.get("failed"):
                    results.setdefault(cell, {}).setdefault(name, {})[value] = None
                    continue
                scores = []
                for png in sorted(under(out).rglob("*.png")):
                    if png.name.endswith(".flip.png"):
                        continue
                    scores.append(score_image(reference, png)[0]["flipMean"])
                log = under(root) / f"{cell}-{name}-{value}.log"
                truncated = False
                if log.exists():
                    truncated = any("exceeds the" in line for line
                                    in log.read_text(encoding="utf-8", errors="replace").splitlines())
                results.setdefault(cell, {}).setdefault(name, {})[value] = {
                    "flip": statistics.fmean(scores) if scores else None,
                    "images": len(scores),
                    "truncated": truncated,
                }

    chosen = {}
    for cell, factors in results.items():
        entry = {"_sweep": {}}
        for name, readings in factors.items():
            entry["_sweep"][name] = {str(v): (r["flip"] if r else None)
                                     for v, r in readings.items()}
            usable = [(v, r["flip"]) for v, r in readings.items()
                      if r and r["flip"] is not None and not r["truncated"]]
            if not usable:
                continue
            best = min(f for _v, f in usable)
            # A tie inside the noise floor goes to the SMALLER value: a difference that
            # cannot be resolved is not a reason to pay for the more expensive setting.
            within = [v for v, f in usable if f <= best * (1.0 + args.tolerance)]
            entry[name] = min(within, key=lambda v: float(v))
        chosen[cell] = entry

    payload = {
        "_comment": "Per-scene technique settings, produced by 'campaign.py sweep'. _sweep "
                    "carries every reading and not only the winner, because the chapter "
                    "justifies the choice rather than merely stating it.",
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "sweepBudgetSeconds": args.seconds,
        "images": args.images,
        "tolerance": args.tolerance,
        "referenceDir": str(references),
        "scenes": chosen,
    }
    out_json = under(root) / "parameters.json"
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    lines = ["# Dobor parametrow per scena (faza 1)", "",
             f"Ramie naprowadzane, budzet {args.seconds} s, {args.images} obraz(y) na punkt, "
             f"rozgrzewka {args.warmup} s. Kryterium: najnizsza srednia FLIP; przy roznicy "
             f"ponizej {100 * args.tolerance:.0f} % wygrywa wartosc mniejsza. Przemiatanie "
             f"jednoczynnikowe - pozostale parametry "
             + (f"na wartosciach wybranych w `{args.base}`." if args.base
                else "na wartosciach domyslnych.") + "", ""]
    for sweep in sweeps:
        name = sweep["cvar"]
        # The column set is the union of what the scenes actually ran, so a scene whose
        # ladder stops early leaves a gap instead of shifting every column left.
        columns = sorted({value for cell in results for value in results[cell].get(name, {})},
                         key=lambda v: float(v))
        lines += [f"## {sweep.get('_label', name)} (`{name}`)", "",
                  "| scena | " + " | ".join(str(v) for v in columns) + " | wybor |",
                  "|---" * (2 + len(columns)) + "|"]
        for cell in sorted(results):
            readings = results[cell].get(name, {})
            cells_out = []
            for value in columns:
                reading = readings.get(value)
                if not reading or reading["flip"] is None:
                    cells_out.append("-")
                    continue
                text = f"{reading['flip']:.5f}"
                if reading["truncated"]:
                    text += " (obciecie)"
                if str(chosen.get(cell, {}).get(name, "")) == str(value):
                    text = f"**{text}**"
                cells_out.append(text)
            lines.append(f"| {cell} | " + " | ".join(cells_out) + " | "
                         + str(chosen.get(cell, {}).get(name, "-")) + " |")
        lines.append("")

    out_md = under(root) / "parametry.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {out_json}")
    print(f"written: {out_md}")


# ------------------------------------------------------- M3: the equal-variance reading


def run_equal_variance(args, manifest):
    """First frame under an MSE ceiling, read off the saved frames - no interpolation.

    The run this reads saved EVERY frame, which the engine charges for: a dense-capture run
    reports 6.8-10.4 ms against a 4.63 ms warm-up at 1080p. The IMAGES are unaffected (same
    frame index, same RNG, same accumulation), so the frame NUMBER is sound and the clock is
    not. The time column is therefore the frame number times the frame cost of a sparse run,
    passed in with --frame-cost, and the table says so.
    """
    root = Path(args.out)
    references = Path(args.reference_dir)
    costs = {}
    if args.frame_cost:
        costs = json.loads(under(Path(args.frame_cost)).read_text(encoding="utf-8-sig"))
    rows = []

    for (cell, arm), _job in sorted(jobs(manifest, args.only).items()):
        reference_path = under(references) / f"{cell}.png"
        if not reference_path.exists():
            continue
        reference = load_rgb(reference_path)
        directory = under(root / cell / arm)
        captures = []
        for png in sorted(directory.rglob("*.png")):
            if png.name.endswith(".flip.png"):
                continue
            captures.append((sidecar_for(png).get("raytracing", {}).get("frameIndex", 0), png))
        found = None
        for frame, png in sorted(captures):
            scores, _ = score_image(reference, png)
            if scores["mse"] <= args.mse:
                found = (frame, scores)
                break
        ms = costs.get(cell, {}).get(arm)
        if found is None:
            rows.append((cell, arm, None, None, None, len(captures)))
            continue
        frame, scores = found
        rows.append((cell, arm, frame, scores,
                     (frame * ms / 1000.0) if ms else None, len(captures)))

    lines = ["# Porownanie przy rownej wariancji (M3)", "",
             f"Pulap MSE {args.mse:g}. Wybrana jest klatka o **najmniejszym numerze**, przy "
             f"ktorej MSE spada ponizej pulapu - bez interpolacji. Czas to numer klatki razy "
             f"koszt klatki z przebiegu bez gestego zapisu; zegar gestego przebiegu nie "
             f"opisuje techniki.", "",
             "| komorka | ramie | klatka | czas [s] | FLIP | MSE | klatek zapisanych |",
             "|---|---|---|---|---|---|---|"]
    for cell, arm, frame, scores, seconds, count in rows:
        if frame is None:
            lines.append(f"| {cell} | {arm} | nie osiagnieto | - | - | - | {count} |")
            continue
        lines.append(f"| {cell} | {arm} | {frame} | "
                     + (f"{seconds:.3f}" if seconds else "-")
                     + f" | {scores['flipMean']:.5f} | {scores['mse']:.3e} | {count} |")

    out_md = under(root) / "rowna-wariancja.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {out_md}")


# ------------------------------------------------------------------ E5: reference length


def final_checkpoint_captures(directory):
    """The captures of a leaf's last checkpoint, found from the sidecars alone.

    Opening six thousand PNGs to discover that fifteen sixteenths of them are not wanted
    is an hour of FLIP per reference; the sidecar says which checkpoint a capture belongs
    to and costs nothing to read.
    """
    by_index = {}
    for png in sorted(Path(directory).rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        index = int(sidecar_for(png).get("benchmark", {}).get("checkpointIndex", 0))
        by_index.setdefault(index, []).append(png)
    return by_index[max(by_index)] if by_index else []


def score_final(root, references, manifest, args):
    """{cell: {arm: mean FLIP at the last checkpoint}} against one reference set."""
    rounds = args.rounds or manifest["rounds"]
    scored = {}
    for (cell, arm), _job in jobs(manifest, args.only).items():
        reference_path = under(references) / f"{cell}.png"
        if not reference_path.exists():
            continue
        reference = load_rgb(reference_path)
        values = []
        for round_index in range(rounds):
            directory = under(leaf(root, cell, arm, 0, round_index))
            if not directory.exists():
                continue
            for png in final_checkpoint_captures(directory):
                values.append(score_image(reference, png)[0]["flipMean"])
        if values:
            scored.setdefault(cell, {})[arm] = statistics.fmean(values)
    return scored


def run_reference_length(args, manifest):
    """One set of captures, two references, one table of what changed."""
    root = Path(args.out)
    scorings = {}
    for label, directory in (("A", args.reference_dir), ("B", args.reference_dir_b)):
        print(f"scoring against {directory} ...", flush=True)
        scorings[label] = (directory, score_final(root, Path(directory), manifest, args))
    if not scorings["A"][1] or not scorings["B"][1]:
        sys.exit("Nothing scored. Both reference sets must cover the rendered cells.")

    def note(directory):
        return reference_note(argparse.Namespace(reference_dir=directory))

    lines = ["# Reference length as a parameter of the comparison (E5)", "",
             "The same captures, scored twice. Nothing was rendered for this table.", "",
             f"- A: {note(scorings['A'][0])}",
             f"- B: {note(scorings['B'][0])}", "",
             "A reference is itself a path-traced image with residual noise. Where that noise "
             "is correlated with an arm's own - and it is, for the arm built from the same "
             "estimator - the measured distance between them is too small, and the shorter the "
             "reference the more so. The column to read is the last one: if the shift were a "
             "scoring offset shared by both arms it would cancel in their ratio.", "",
             "Both columns are the mean over the ten images of the final checkpoint, the "
             "same number the main tables quote.", "",
             "| cell | arm | FLIP A | FLIP B | change | ratio A | ratio B | shift |",
             "|---|---|---|---|---|---|---|---|"]

    def final(scoring, cell, arm):
        return scoring.get(cell, {}).get(arm)

    cells = sorted(set(scorings["A"][1]) | set(scorings["B"][1]))
    for cell in cells:
        arms = sorted(set(scorings["A"][1].get(cell, {})) | set(scorings["B"][1].get(cell, {})))
        baseline = arms[0] if arms else None
        base_a, base_b = final(scorings["A"][1], cell, baseline), final(scorings["B"][1], cell, baseline)
        for arm in arms:
            value_a, value_b = final(scorings["A"][1], cell, arm), final(scorings["B"][1], cell, arm)
            if value_a is None or value_b is None:
                continue
            change = f"{(value_b - value_a) / value_a * 100:+.2f} %" if value_a else "\u2014"
            ratio_a = base_a / value_a * 100 if base_a and value_a and arm != baseline else None
            ratio_b = base_b / value_b * 100 if base_b and value_b and arm != baseline else None
            shift = (f"{ratio_b - ratio_a:+.1f} pp"
                     if ratio_a is not None and ratio_b is not None else "\u2014")
            text_a = "\u2014" if ratio_a is None else f"{ratio_a:.0f} %"
            text_b = "\u2014" if ratio_b is None else f"{ratio_b:.0f} %"
            lines.append(f"| {cell} | {arm} | {value_a:.5f} | {value_b:.5f} | {change} | "
                         f"{text_a} | {text_b} | {shift} |")

    out = under(Path(args.out)) / "reference-length.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


# ------------------------------------------------------------------ vendor levers (Q13)

# One row per lever, plus the baseline that every row is read against. `ser` is absent on
# purpose: it needs hardware shader-execution reordering, which the RDNA card does not have
# and the Ampere laptop does not either, so measuring it would produce a row of zeros.
DEFAULT_LEVERS = ["none", "noviews", "swizzle", "wave32", "wave64", "lib69"]


def run_levers(args, manifest):
    """Every lever alone, both arms, on the named cells."""
    root = Path(args.out)
    (RAYTRACER_DIR / root).mkdir(parents=True, exist_ok=True)
    references = Path(args.reference_dir)
    arms = args.arms.split(",")
    levers = args.levers.split(",")
    wanted = args.cells.split(",") if args.cells else None
    rows = {}

    for cell, scene, light in campaign_cells(manifest, args.only):
        if wanted and cell not in wanted:
            continue
        reference_path = under(references) / f"{cell}.png"
        reference = load_rgb(reference_path) if reference_path.exists() else None
        for arm in arms:
            spec = manifest["arms"][arm]
            for lever in levers:
                out = root / cell / arm / lever
                if not (under(out) / "done.json").exists():
                    print(f"  {cell} {arm} {lever} ...", flush=True)
                    code, took = engine_checked(
                        scene, light, manifest, spec["technique"], spec["cvars"], out,
                        f"seconds:{args.seconds}", args.images, args.warmup,
                        log_path=under(root) / f"{cell}-{arm}-{lever}.log",
                        config=config_path(manifest, scene, light),
                        levers=[] if lever == "none" else [lever])
                    if code != 0:
                        print(f"  FAILED {cell} {arm} {lever} exit={code}")
                        continue
                    (under(out) / "done.json").write_text(
                        json.dumps({"seconds": took, "lever": lever}))
                    print(f"  ok     {cell} {arm} {lever} in {took:.0f}s")

                captures = [png for png in sorted(under(out).rglob("*.png"))
                            if not png.name.endswith(".flip.png")]
                if not captures:
                    continue
                flips, milliseconds, frames = [], [], 0
                for png in captures:
                    if reference is not None:
                        flips.append(score_image(reference, png)[0]["flipMean"])
                    sidecar = sidecar_for(png)
                    milliseconds.append(sidecar.get("benchmark", {}).get("meanFrameMs", 0.0))
                    frames = sidecar.get("raytracing", {}).get("frameIndex", frames)
                rows.setdefault((cell, arm), {})[lever] = {
                    "flip": aggregate(flips), "ms": aggregate(milliseconds), "frames": frames}

    lines = ["# Vendor levers, one factor at a time (Q13, 7.4)", "",
             f"Build: {recon.EXE.parent.name}. Every lever alone against the same baseline, "
             f"{args.images} image(s) of {args.seconds} s after a {args.warmup} s warm-up. "
             f"`--levers` states the whole set, so no row inherits the row before it. Shader "
             f"execution reordering is absent from the sweep: it needs hardware this work does "
             f"not have on either machine, and that is a stated limitation, not a result.",
             reference_note(args), "",
             "A lever is a result only where it clears the between-run noise floor of the "
             "campaign, 0.1 % on FLIP. Everything smaller is the machine, not the lever.", "",
             "| cell | arm | lever | ms/frame | vs none | frames | FLIP | vs none |",
             "|---|---|---|---|---|---|---|---|"]
    for (cell, arm), by_lever in sorted(rows.items()):
        base = by_lever.get("none")
        for lever in levers:
            entry = by_lever.get(lever)
            if not entry:
                continue
            ms = entry["ms"].get("mean", 0.0)
            flip = entry["flip"].get("mean", 0.0)
            ms_delta = flip_delta = "\u2014"
            if base and lever != "none":
                base_ms = base["ms"].get("mean", 0.0)
                base_flip = base["flip"].get("mean", 0.0)
                if base_ms > 0:
                    ms_delta = f"{(ms - base_ms) / base_ms * 100:+.1f} %"
                if base_flip > 0:
                    flip_delta = f"{(flip - base_flip) / base_flip * 100:+.2f} %"
            lines.append(f"| {cell} | {arm} | {lever} | {ms:.3f} | {ms_delta} | "
                         f"{entry['frames']} | {flip:.5f} | {flip_delta} |")

    out_path = under(root) / "levers.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_build_argument(parser)
    parser.add_argument("--manifest", default="tools/campaign-manifest.json")
    # Comma-separated rather than nargs="*": a greedy list swallows the subcommand name,
    # so "--only K2 run" would parse "run" as a category and fail asking for a command.
    parser.add_argument("--only", metavar="K1,K2", type=lambda v: v.split(","),
                        help="restrict to these categories (default: all in the manifest)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("configs", help="write one headless config per cell")

    references = sub.add_parser("references", help="one path-traced reference per cell")
    references.add_argument("--seconds", type=float, default=None)
    references.add_argument("--warmup", type=float, default=45.0)
    references.add_argument("--out", default="SavedUserData/Screenshots/campaign-refs")

    measure = sub.add_parser("run", help="the campaign")
    measure.add_argument("--rounds", type=int, default=None)
    measure.add_argument("--repeats", type=int, default=None)
    measure.add_argument("--dry-run", action="store_true", help="list the job union and the time it costs, render nothing")
    measure.add_argument("--out", default="SavedUserData/Screenshots/campaign")
    # Per-scene technique settings, produced by 'sweep'. Passing an empty string measures
    # on the engine defaults, which is a different measurement and has to be stated as one.
    measure.add_argument("--parameters",
                         default="Raytracer/SavedUserData/Screenshots/parametry/parameters.json")

    budget = sub.add_parser("budget", help="phase 2: frame cost per arm and the equal-time budget")
    budget.add_argument("--arms", default="BSDF,WIE")
    budget.add_argument("--seconds", type=float, default=10.0)
    budget.add_argument("--images", type=int, default=3)
    budget.add_argument("--warmup", type=float, default=45.0)
    # How far the two arms' actually-spent times may differ. 10 %, because the overshoot is
    # quantised by the frame cost and no budget below a few hundred ms can do better.
    budget.add_argument("--tolerance", type=float, default=0.10)
    budget.add_argument("--out", default="SavedUserData/Screenshots/parametry-testowania")
    budget.add_argument("--parameters",
                        default="Raytracer/SavedUserData/Screenshots/parametry/parameters.json")

    matrix = sub.add_parser("matrix", help="plan 7.3: reference/PT/guided image row per scene")
    matrix.add_argument("--arms", default="BSDF,WIE")
    matrix.add_argument("--out", default="SavedUserData/Screenshots/wyniki-czas")
    matrix.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")

    sweep = sub.add_parser("sweep", help="phase 1: one factor at a time, per scene")
    sweep.add_argument("--arm", default="WIE")
    sweep.add_argument("--seconds", type=float, default=5.0)
    sweep.add_argument("--images", type=int, default=3)
    # 35 s and not less: the warm-up settle criterion needs a 30 s floor, and below it every
    # point of the sweep would be recorded as unsettled.
    sweep.add_argument("--warmup", type=float, default=35.0)
    # How close two readings have to be before the cheaper setting wins. Default 2 %,
    # which is above the between-run noise floor measured on this harness (0.01-0.21 % on
    # the image mean, but a 5 s point is far noisier than a 30 s one).
    sweep.add_argument("--tolerance", type=float, default=0.02)
    sweep.add_argument("--out", default="SavedUserData/Screenshots/parametry")
    sweep.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")
    # Coordinate descent: hold the other factors at what a previous pass chose, rather than
    # at the engine defaults.
    sweep.add_argument("--base", default=None,
                       help="parameters.json whose choices pin the factors not being swept")

    variance = sub.add_parser("equal-variance", help="M3: first frame under an MSE ceiling")
    variance.add_argument("--mse", type=float, required=True)
    variance.add_argument("--out", default="SavedUserData/Screenshots/wyniki-wariancja")
    variance.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")
    # {cell: {arm: meanFrameMs}} from a run that captured RARELY. The dense run's own clock
    # is inflated by the per-capture machinery and does not describe the technique.
    variance.add_argument("--frame-cost", default=None,
                          help="JSON of per-cell, per-arm ms/frame from a sparse-capture run")

    maps = sub.add_parser("diffmap", help="8.5.1: where the reuse variant's bias lives")
    maps.add_argument("--rounds", type=int, default=None)
    maps.add_argument("--out", default="SavedUserData/Screenshots/campaign")
    maps.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")
    maps.add_argument("--maps-out", default="SavedUserData/Screenshots/campaign-maps")
    # The level below which a pixel difference is noise. 'auto' derives it from the same
    # arm measured twice from scratch, which is the between-run spread 10.2 asks for taken
    # at the PER-PIXEL scale the threshold works at — the K10 table's spread is on the
    # image mean and is orders of magnitude tighter. A float instead means that fraction
    # of the unbiased arm's own mean error.
    maps.add_argument("--noise", default="auto",
                      help="'auto' (same-arm between-run p95) or a fraction of the arm's mean error")

    report = sub.add_parser("report", help="score everything into one table")
    report.add_argument("--rounds", type=int, default=None)
    report.add_argument("--repeats", type=int, default=None)
    report.add_argument("--out", default="SavedUserData/Screenshots/campaign")
    report.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")
    # E5 scores one set of captures against two references of different length, so the
    # second scoring must not land on the first one's file.
    report.add_argument("--name", default="campaign",
                        help="stem for campaign.md / -curves.csv / .json")
    report.add_argument("--parameters", default=None,
                        help="parameters.json to record in the result header (plan 7.2)")

    temporal = sub.add_parser("temporal", help="S1-S4 temporal stability (K4, 8.5)")
    temporal.add_argument("--arms", default="BSDF,WIE")
    temporal.add_argument("--frames", type=int, default=64)
    temporal.add_argument("--independent", type=int, default=32)
    # M6: the sample count one image of the series carries, the same for both arms.
    temporal.add_argument("--samples", type=int, default=2)
    temporal.add_argument("--images", type=int, default=64,
                          help="images in the equal-sample series (M6)")
    temporal.add_argument("--warmup", type=int, default=45)
    temporal.add_argument("--out", default="SavedUserData/Screenshots/campaign-temporal")

    temporal_report = sub.add_parser("temporal-report", help="score a finished temporal run")
    temporal_report.add_argument("--arms", default="BSDF,WIE")
    temporal_report.add_argument("--frames", type=int, default=64)
    temporal_report.add_argument("--independent", type=int, default=32)
    temporal_report.add_argument("--out", default="SavedUserData/Screenshots/campaign-temporal")
    temporal_report.add_argument("--reference-dir",
                                 default="SavedUserData/Screenshots/campaign-refs")
    temporal_report.add_argument("--stability-csv", action="store_true",
                                 help="also write stabilnosc.csv from the equal-sample series (M6)")

    breakdown = sub.add_parser("breakdown", help="per-node frame cost (8.4)")
    breakdown.add_argument("--arms", default="BSDF,WIE")
    breakdown.add_argument("--seconds", type=int, default=20)
    breakdown.add_argument("--warmup", type=int, default=45)
    breakdown.add_argument("--out", default="SavedUserData/Screenshots/campaign-breakdown")

    memory = sub.add_parser("memory", help="footprint against grid resolution (8.6, P5)")
    memory.add_argument("--arm", default="WIE")
    memory.add_argument("--baseline-arm", default="BSDF",
                        help="the arm whose footprint the guided total is read against (M8)")
    memory.add_argument("--seconds", type=int, default=30)
    memory.add_argument("--warmup", type=int, default=45)
    memory.add_argument("--out", default="SavedUserData/Screenshots/campaign-memory")
    memory.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")

    e5 = sub.add_parser("reference-length", help="E5: score one set of captures against two references")
    e5.add_argument("--rounds", type=int, default=None)
    e5.add_argument("--repeats", type=int, default=None)
    e5.add_argument("--out", default="SavedUserData/Screenshots/campaign")
    e5.add_argument("--reference-dir", default="SavedUserData/Screenshots/recon-refs")
    e5.add_argument("--reference-dir-b", default="SavedUserData/Screenshots/campaign-refs")

    levers = sub.add_parser("levers", help="single-factor vendor lever sweep (Q13)")
    levers.add_argument("--arms", default="BSDF,WIE")
    levers.add_argument("--levers", default=",".join(DEFAULT_LEVERS))
    levers.add_argument("--cells", default="veach-ajar--own,bistro-exterior--own")
    levers.add_argument("--seconds", type=int, default=30)
    levers.add_argument("--images", type=int, default=3)
    levers.add_argument("--warmup", type=int, default=45)
    levers.add_argument("--out", default="SavedUserData/Screenshots/campaign-levers")
    levers.add_argument("--reference-dir", default="SavedUserData/Screenshots/campaign-refs")

    args = parser.parse_args()
    manifest = load_manifest(REPO_ROOT / args.manifest)

    # recon.py's engine helpers read its module-global EXE, which its own main() binds.
    # Importing them means binding it here instead; doing it explicitly beats a shared
    # mutable default that neither file owns.
    recon.EXE = resolve_exe(args.build)

    if args.command == "configs":
        write_configs(manifest, args.only)
    elif args.command == "references":
        run_references(args, manifest)
    elif args.command == "run":
        run_campaign(args, manifest)
    elif args.command == "budget":
        run_budget(args, manifest)
    elif args.command == "matrix":
        run_matrix(args, manifest)
    elif args.command == "sweep":
        run_sweep(args, manifest)
    elif args.command == "equal-variance":
        run_equal_variance(args, manifest)
    elif args.command == "levers":
        run_levers(args, manifest)
    elif args.command == "reference-length":
        run_reference_length(args, manifest)
    elif args.command == "memory":
        run_memory(args, manifest)
    elif args.command == "breakdown":
        run_breakdown(args, manifest)
    elif args.command == "temporal":
        run_temporal(args, manifest)
    elif args.command == "temporal-report":
        report_temporal(args, manifest)
        if getattr(args, "stability_csv", False):
            write_stability_csv(args, manifest)
    elif args.command == "report":
        run_report(args, manifest)
    elif args.command == "diffmap":
        run_diffmap(args, manifest)


if __name__ == "__main__":
    main()
