#!/usr/bin/env python3
"""Reconnaissance pass over the research grid: scene x light type, three arms.

This is NOT the final campaign and its numbers do not go into the thesis. It exists
to decide WHERE the campaign should spend its 72 hours, by answering four questions
per cell:

  * does the cell render at all, and is the image non-degenerate?
  * is the difference between arms resolvable against the run-to-run noise floor?
  * what is the sign and order of magnitude of the effect?
  * does anything truncate silently (light-tree leaf cap, compaction buffer)?

Because it ranks cells rather than measuring them, it runs on PNG captures and LDR-FLIP;
moving to floating-point captures and HDR-FLIP changes the values but not which cells
matter. It does NOT economise on the build: every timing comes from Release, the
configuration that ships, because a diagnostic build is not a program anyone runs.
(Debug and Release frame cost were measured to agree within 1% for both techniques —
the frame is GPU-work-bound — so the switch costs nothing and removes a caveat.)

A CELL is (scene, light type), and by default a scene has exactly one: 'own', the
light the scene was authored with, which lives in its camera state. The three light
types become an axis only in category K2, on two scenes, because substituting a
source into a scene that has none changes the scene rather than a parameter.

A cell's rig, exposure and tone curve live in a generated headless config, so that
the reference image and every arm scored against it share one description of the
conditions - which is the whole reason `indirectOnly` is a config key rather than a
command-line flag.

Subcommands, in the order they are meant to be run:

  configs     write one headless config per cell from the manifest
  smoke       5 s per cell per technique; catches a scene that does not load, a
              light rig that leaves the scene black, and any silent truncation
  references  one long path-traced reference per cell (skips what exists)
  run         the measurement: every cell, every arm
  grid        K5: the guided arm again at every voxel-grid rung the cell can take
  direct      16.9: both arms with and without direct light, one protocol
  importance  16.10: the guided arm under each top-level weighting strategy
  ratio       the same cells at a handful of samples, at a fixed frame proportion
  report      one table over everything produced

Usage:
  python tools/recon.py configs    --manifest tools/recon-manifest.json
  python tools/recon.py smoke      --manifest tools/recon-manifest.json
  python tools/recon.py references --manifest tools/recon-manifest.json --seconds 600
  python tools/recon.py run        --manifest tools/recon-manifest.json --seconds 30
  python tools/recon.py report     --manifest tools/recon-manifest.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

from bench_report import (add_build_argument, aggregate, flip_settings,  # noqa: E402
                          load_rgb, resolve_exe, score_image, sidecar_for)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAYTRACER_DIR = REPO_ROOT / "Raytracer"

EXE = None  # bound in main() from --build / BAMBOO_BUILD

# The three arms of PLAN_BADAWCZY section 5. The biased one is here because the recon
# has to know whether its crossover is visible at all in a given cell, not because it
# belongs in the main results.
ARMS = [
    ("PT",    "Path Tracing",                 []),
    ("VXPG",  "Guided Path Tracing (VXPG)",   []),
    ("REUSE", "Guided Path Tracing (VXPG)",   ["vxpg.injection.reuseInMis=1"]),
]

# How far a within-run confidence interval is trusted. Images of one process share a
# warm-up, a clock state and a thermal state, and on this harness that has been measured
# to understate the spread between separate runs by 3-10x. The recon uses the pessimistic
# end, because its job is to reject cells, and a cell wrongly kept costs campaign hours.
NOISE_INFLATION = 10


# ------------------------------------------------------------------ the manifest

def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for scene in manifest["scenes"]:
        for key in ("id", "state", "exposure"):
            if key not in scene:
                sys.exit(f"Manifest scene {scene.get('id', '?')} is missing '{key}'")
    return manifest


def cells(manifest):
    """(scene, light) pairs, skipping any the manifest marks as not applicable.

    Light type is a per-scene list, not a global axis: PLAN_BADAWCZY 7.2 rules that
    substituting a source into a scene that does not have one is a change of scene,
    not of parameter, so every scene is measured under its OWN light and only the K2
    category puts the three types side by side, on two scenes."""
    default = manifest.get("defaultLightTypes", ["own"])
    for scene in manifest["scenes"]:
        for light in scene.get("lightTypes", default):
            if light in scene.get("skipLights", []):
                continue
            yield scene, light


def cell_id(scene, light):
    return f"{scene['id']}--{light}"


def config_path(manifest, scene, light):
    return Path(manifest["configDir"]) / f"recon.{cell_id(scene, light)}.json"


def scene_argument(scene):
    """A bare id resolves inside the engine; anything else is spelled out."""
    return scene.get("scenePath", scene["id"])


def states_key(scene):
    """Which states.json key holds this scene's cameras. Spelling out a scenePath makes
    the engine stop deriving the key from the name, so the recon always states it."""
    return scene.get("statesKey", scene["id"])


def state_for(scene, light):
    """The camera state a cell renders from. A substituted light type gets its own
    state because the rig is authored IN the state, next to the camera it was balanced
    against — the alternative, a rig in the generated config, would let the two drift
    apart while both files still looked right."""
    return scene.get("states", {}).get(light, scene["state"])


# ------------------------------------------------------------------ configs

def build_light(scene, light, manifest):
    """The rig for one light type, or None for 'no lights key at all'.

    Three distinct statements, and the difference is load-bearing:
      'own'      -> omit the key, so the camera state's own `lights` array rules and
                    the scene is lit the way it was authored;
      'emissive' -> the empty list, which drops every analytic light and leaves the
                    emissive meshes;
      point/directional -> the rig, unless the cell has a state of its own, in which
                    case the state carries it and the config keeps out of the way.
                    A rig given here still wins: the engine applies config lights
                    AFTER the state precisely so a substitution can be stated once."""
    if light == "own":
        return None
    if light == "emissive":
        return []
    if light in scene.get("states", {}):
        return None
    rig = scene.get("lights", {}).get(light)
    if rig is None:
        rig = manifest.get("defaultLights", {}).get(light)
    if rig is None:
        sys.exit(f"No '{light}' rig for scene {scene['id']}: add it under scenes[].lights "
                 f"or defaultLights in the manifest")
    rigs = [rig] if isinstance(rig, dict) else list(rig)
    # Manifest annotations start with '_'. They are notes to a person, and a generated
    # config is the record of a measurement's conditions, so they do not belong in it.
    return [{k: v for k, v in entry.items() if not k.startswith("_")} for entry in rigs]


def write_configs(manifest):
    out_dir = RAYTRACER_DIR / manifest["configDir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scene, light in cells(manifest):
        config = dict(manifest["renderDefaults"])
        config["exposure"] = scene["exposure"]
        config.update(scene.get("overrides", {}))
        rig = build_light(scene, light, manifest)
        if rig is not None:
            config["lights"] = rig
        # Substituting a light type means REPLACING the scene's own source, so the
        # emitters go off with it; leaving them on would compare one source against two.
        if light in ("point", "directional"):
            config["emissiveGeometry"] = False
        path = RAYTRACER_DIR / config_path(manifest, scene, light)
        path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        written.append(path)
    print(f"{len(written)} configs written under {out_dir}")
    for path in written:
        print(f"  {path.relative_to(RAYTRACER_DIR).as_posix()}")


# ------------------------------------------------------------------ running

def wrong_size_captures(directory, manifest):
    """Captures whose pixel size is not what the config asked for.

    Seen once in the wild: a capture came out 1280x720 in a 1920x1080 run, taken in the
    first frames after a shader rebuild. Harmless in itself — but silent, and it would
    either crash the scorer or, worse, be resized by some other tool and averaged in. A
    run that produces one is thrown away and repeated rather than reported."""
    want = (int(manifest["renderDefaults"]["width"]), int(manifest["renderDefaults"]["height"]))
    bad = []
    for png in Path(directory).rglob("*.png"):
        if png.name.endswith(".flip.png"):
            continue
        try:
            with Image.open(png) as image:
                if image.size != want:
                    bad.append((png, image.size))
        except OSError:
            bad.append((png, None))
    return bad


CAPTURES_UNUSABLE = 90

MIN_FREE_BYTES = 8 * 1024 ** 3


def require_disk_space():
    """Refuse to start a run that the drive cannot hold.

    A full disk does not announce itself: the engine keeps rendering, every capture is
    written short, and the scorer meets a PNG it cannot open. Cheaper to stop here than
    to spend hours discovering it one truncated image at a time.
    """
    free = shutil.disk_usage(RAYTRACER_DIR).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"only {free / 1024 ** 3:.1f} GiB free on the capture drive, "
                         f"{MIN_FREE_BYTES / 1024 ** 3:.0f} GiB required - free space and resume")


def engine(scene, light, manifest, technique, cvars, out_dir, budget, images, warmup,
           checkpoints=None, log_path=None, config=None, debug_views=None,
           rdg_timings=False, levers=None, settle=0):
    command = [
        str(EXE), "--headless",
        "--scene", scene_argument(scene),
        "--states-key", states_key(scene),
        "--states", state_for(scene, light).replace(" ", "_"),
        "--techniques", technique.replace(" ", "_"),
        "--config", (config or config_path(manifest, scene, light)).as_posix(),
        "--budget", budget,
        "--images", str(images),
        "--warmup", str(warmup),
        "--out", Path(out_dir).as_posix(),
    ]
    if checkpoints:
        command += ["--checkpoints", checkpoints]
    # Frames pumped between images, outside every measured window: at a budget of one display
    # frame the leftover of the previous capture can otherwise eat a whole image.
    if settle:
        command += ["--settle", str(settle)]
    # A buffer debug view has to arrive as --debug-views: SelectDebugView wipes the CVar,
    # so `--cvar renderer.bufferDebugView=...` silently renders the ordinary image.
    if debug_views:
        command += ["--debug-views", debug_views]
    # The node cost table 8.4 reads is only produced when the graph is timed, and the dump
    # that prints it is taken after the captures, when the graph is warm.
    if rdg_timings:
        command += ["--rdg-timings", "--rdg-dump"]
    # --levers states the whole set, and "none" is how a matrix row asks for the baseline:
    # an empty value would be indistinguishable from the flag being absent.
    if levers is not None:
        command += ["--levers", ",".join(levers) if levers else "none"]
    for assignment in cvars:
        command += ["--cvar", assignment]

    started = time.time()
    if log_path:
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(" ".join(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=RAYTRACER_DIR, stdout=log,
                                    stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(command, cwd=RAYTRACER_DIR)
    return result.returncode, time.time() - started


def engine_checked(scene, light, manifest, technique, cvars, out_dir, budget, images, warmup,
                   checkpoints=None, log_path=None, config=None, debug_views=None,
                   rdg_timings=False, levers=None, settle=0):
    """engine(), then throw the run away and repeat it once if any capture came out at the
    wrong resolution. Seen once in the wild: a 1280x720 image inside a 1920x1080 run,
    written in the first frames after a shader rebuild. Rare and harmless on its own —
    the danger is that it is silent, and a scorer that resizes rather than refuses would
    average a different image into the result."""
    for attempt in (1, 2):
        require_disk_space()
        code, seconds = engine(scene, light, manifest, technique, cvars, out_dir, budget,
                               images, warmup, checkpoints, log_path, config, debug_views,
                               rdg_timings, levers, settle)
        if code != 0:
            # A nonzero exit is not always a defect in the run. Seen 2026-09-01: bistro-exterior
            # failed its BLAS build with E_OUTOFMEMORY because the previous job's process had not
            # released its device memory yet, and the same cell loaded fine seconds later. Give
            # the driver a moment and take one more shot before writing the cell off.
            if attempt == 1:
                print(f"        engine exited {code} — waiting for the device to settle and repeating")
                time.sleep(20)
                continue
            return code, seconds
        bad = wrong_size_captures(under(out_dir), manifest)
        if not bad:
            return code, seconds
        print(f"        {len(bad)} capture(s) at the wrong resolution (first {bad[0][1]}) "
              f"— discarding and repeating (attempt {attempt})")
        for png, _ in bad:
            png.unlink(missing_ok=True)
            png.with_suffix(".json").unlink(missing_ok=True)
    # A run whose captures still will not open is not a run. Reporting it as a success
    # would leave the leaf short of images and let the campaign average a smaller sample
    # into the result without saying so — which is exactly what a full disk did on
    # 2026-09-01, when every truncated write came back as an unreadable PNG.
    print(f"        {len(bad)} capture(s) still unreadable or mis-sized after a repeat — failing the run")
    return CAPTURES_UNUSABLE, seconds


def under(path):
    """--out and --reference-dir are RAYTRACER-relative, because that is what the engine
    is handed and what its own logs echo. Every filesystem access from here has to
    resolve them the same way or the tool reads a directory the engine never wrote to."""
    path = Path(path)
    return path if path.is_absolute() else RAYTRACER_DIR / path


def newest_capture(directory):
    captures = sorted(p for p in Path(directory).rglob("*.png") if not p.name.endswith(".flip.png"))
    return captures[-1] if captures else None


def lit_fraction(path, threshold=0.02):
    """How much of the image carries signal. A near-black render scores as almost
    perfect against a near-black reference, so 'looks converged' and 'is empty' are
    indistinguishable downstream - this is the check that separates them."""
    luminance = np.asarray(load_rgb(path), dtype=np.float64) @ (0.2126, 0.7152, 0.0722)
    return float((luminance > threshold).mean())


def truncation_warnings(log_path):
    if not Path(log_path).exists():
        return []
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if "exceeds the" in line]


# ------------------------------------------------------------------ subcommands

def run_smoke(args, manifest):
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene, light in cells(manifest):
        for label, technique, cvars in ARMS[:2]:  # PT and VXPG are enough to smoke it
            out = root / cell_id(scene, light) / label
            log = absolute_root / f"{cell_id(scene, light)}-{label}.log"
            # The cluster probe is what surfaces a silent light-tree truncation; it is
            # one-shot and free on every other frame, so a smoke run is where it belongs.
            code, seconds = engine_checked(scene, light, manifest, technique, list(cvars) + ["vxpg.cluster.dumpStats=1"],
                                   out, f"seconds:{args.seconds}", 1, 2, log_path=log)
            capture = newest_capture(under(out))
            lit = lit_fraction(capture) if capture else 0.0
            warnings = truncation_warnings(log)
            rows.append({"cell": cell_id(scene, light), "arm": label, "exit": code,
                         "seconds": seconds, "litFraction": lit, "warnings": warnings})
            status = "ok" if code == 0 and lit >= args.min_lit else "PROBLEM"
            print(f"{status:8} {cell_id(scene, light):32} {label:6} exit={code} lit={lit * 100:5.1f}%")
            for warning in warnings:
                print(f"         ! {warning}")
    (absolute_root / "smoke.json").write_text(json.dumps(rows, indent=2))
    bad = [r for r in rows if r["exit"] != 0 or r["litFraction"] < args.min_lit]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} cells usable")
    return 1 if bad else 0


def run_references(args, manifest):
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    for scene, light in cells(manifest):
        marker = absolute_root / f"{cell_id(scene, light)}.png"
        if marker.exists():
            print(f"exists  {marker.name}")
            continue
        out = root / "raw" / cell_id(scene, light)
        code, seconds = engine_checked(scene, light, manifest, "Path Tracing", [], out,
                               f"seconds:{args.seconds}", 1, 10,
                               log_path=absolute_root / f"{cell_id(scene, light)}.log")
        capture = newest_capture(under(out))
        if code != 0 or capture is None:
            print(f"FAILED  {cell_id(scene, light)} exit={code}")
            continue
        lit = lit_fraction(capture)
        if lit < args.min_lit:
            # Kept on disk under a different name: the image is the evidence of what
            # went wrong, but it must not sit where the scoring pass will find it.
            capture.replace(absolute_root / f"{cell_id(scene, light)}.black.png")
            print(f"BLACK   {cell_id(scene, light)} only {lit * 100:.2f}% lit — rig is wrong for this scene")
            continue
        sidecar = capture.with_suffix(".json")
        capture.replace(marker)
        if sidecar.exists():
            sidecar.replace(marker.with_suffix(".json"))
        print(f"ok      {marker.name}  {lit * 100:.1f}% lit  in {seconds / 60:.1f} min")


def run_measurement(args, manifest):
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    for scene, light in cells(manifest):
        for label, technique, cvars in ARMS:
            out = root / cell_id(scene, light) / label
            if (under(out) / "done.json").exists():
                print(f"exists  {cell_id(scene, light)} {label}")
                continue
            code, seconds = engine_checked(scene, light, manifest, technique, cvars, out,
                                   f"seconds:{args.seconds}", args.images, args.warmup,
                                   checkpoints=args.checkpoints,
                                   log_path=absolute_root / f"{cell_id(scene, light)}-{label}.log")
            if code != 0:
                print(f"FAILED  {cell_id(scene, light)} {label} exit={code}")
                continue
            (under(out) / "done.json").write_text(json.dumps({"seconds": seconds}))
            print(f"ok      {cell_id(scene, light)} {label} in {seconds:.0f}s")


# ------------------------------------------------------------------ K5: grid ladder

def grid_resolutions(scene, manifest):
    """How far this scene's grid may be pushed. The ladder is per scene because the
    32768-leaf cap bites at a different resolution in each one (R17/R18): a rung above
    the cap would measure the cap, not the grid."""
    return scene.get("gridResolutions", manifest.get("defaultGridResolutions", [32, 64]))


def run_grid(args, manifest):
    """Quality AND cost against voxel grid resolution, guided arm only.

    Path tracing has no grid, so its curve is a constant already measured by `run`, and
    a path-traced reference is grid-independent — no new reference is needed per rung.

    `--indirect` switches to the indirect-only configs and to a checkpoint ladder that
    starts at ONE OR TWO guided samples. Both matter for this question: the source paper's
    grid ablation is measured without direct light and without NEE, in a regime where its
    FLIP is 0.8-0.97 (noise-dominated). Ours at 30 s with direct light sits at 0.016-0.026,
    where every estimator has converged and no sampling change can show.
    """
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        config = curve_config_path(manifest, scene, light) if args.indirect else None
        checkpoints = (curve_checkpoints(name, args.seconds, args.first_frames, args.points)
                       if args.indirect else args.checkpoints)
        for resolution in grid_resolutions(scene, manifest):
            out = root / name / f"g{resolution}"
            if (under(out) / "done.json").exists():
                print(f"exists  {name} {resolution}^3")
                continue
            code, seconds = engine_checked(
                scene, light, manifest, "Guided Path Tracing (VXPG)",
                [f"voxel.gridDim={resolution}", "vxpg.cluster.dumpStats=1"], out,
                f"seconds:{args.seconds}", args.images, args.warmup,
                checkpoints=checkpoints, config=config,
                log_path=absolute_root / f"{name}-g{resolution}.log")
            if code != 0:
                print(f"FAILED  {cell_id(scene, light)} {resolution}^3 exit={code}")
                continue
            (under(out) / "done.json").write_text(json.dumps({"seconds": seconds,
                                                              "gridDim": resolution}))
            print(f"ok      {cell_id(scene, light)} {resolution}^3 in {seconds:.0f}s")


# ---------------------------------------------- section 16.9: the direct-light control

# The two lighting conditions of the control. The main measurement runs the first: the
# whole image, direct light included. The paper's own ablations run the second, where the
# integrator's advantage is not diluted by a component it does not touch — direct light
# reaches NEE identically in both arms. Each condition scores against its OWN reference,
# because indirect-only is a different integral and not a darker version of the same one.
DIRECT_CONDITIONS = [
    ("with-direct", "SavedUserData/Screenshots/recon-refs"),
    ("indirect-only", "SavedUserData/Screenshots/recon-curve-refs"),
]

DIRECT_ARMS = [("PT", "Path Tracing"), ("VXPG", "Guided Path Tracing (VXPG)")]


def selected_cells(manifest, wanted):
    """cells(), filtered by cell id, so a control run can name one cell instead of eight."""
    chosen = [(scene, light) for scene, light in cells(manifest)
              if not wanted or cell_id(scene, light) in wanted]
    unknown = set(wanted or []) - {cell_id(scene, light) for scene, light in cells(manifest)}
    if unknown:
        sys.exit(f"Unknown cell(s): {', '.join(sorted(unknown))}")
    return chosen


def run_direct(args, manifest):
    """Both arms, both lighting conditions, one protocol — 16.9's control comparison.

    Both configurations already exist (the cell's own, and the indirect-only one the curve
    study writes), and so do both 600 s references, so this costs four renders per cell and
    no new reference. The budget, image count and checkpoint ladder are shared across the
    four legs, because the number this produces is a RATIO OF RATIOS and any asymmetry in
    the protocol would land in it."""
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    for scene, light in selected_cells(manifest, args.cells):
        name = cell_id(scene, light)
        write_curve_config(manifest, scene, light)
        for condition, _references in DIRECT_CONDITIONS:
            config = (curve_config_path(manifest, scene, light)
                      if condition == "indirect-only" else None)
            for label, technique in DIRECT_ARMS:
                out = root / name / condition / label
                if (under(out) / "done.json").exists():
                    print(f"exists  {name} {condition} {label}")
                    continue
                code, seconds = engine_checked(
                    scene, light, manifest, technique, [], out,
                    f"seconds:{args.seconds}", args.images, args.warmup,
                    checkpoints=args.checkpoints, config=config,
                    log_path=absolute_root / f"{name}-{condition}-{label}.log")
                if code != 0:
                    print(f"FAILED  {name} {condition} {label} exit={code}")
                    continue
                (under(out) / "done.json").write_text(json.dumps({"seconds": seconds,
                                                                  "condition": condition}))
                print(f"ok      {name} {condition} {label} in {seconds:.0f}s")


def run_direct_report(args, manifest):
    """How much of the guided advantage direct light dilutes, per cell."""
    root = under(args.out)
    lines = ["# Direct-light control (section 16 item 9)", "",
             "The same cell, the same budget, twice: once as the main measurement renders it",
             "(direct light included) and once indirect-only, the regime the source paper",
             "ablates in. Each condition is scored against its own path-traced reference —",
             "indirect-only is a different integral, not a darker image. 'VXPG vs PT' above",
             f"100 % means the guided arm is ahead at equal time. FLIP {flip_settings()['version']} "
             f"LDR at {flip_settings()['ppd']:.2f} ppd.", "",
             "| cell | condition | arm | ms/frame | FLIP | ci95 | VXPG vs PT |",
             "|---|---|---|---|---|---|---|"]

    dilution = []
    for scene, light in selected_cells(manifest, args.cells):
        name = cell_id(scene, light)
        ratios, frames = {}, {}
        for condition, reference_dir in DIRECT_CONDITIONS:
            reference_path = under(reference_dir) / f"{name}.png"
            if not reference_path.exists():
                lines.append(f"| {name} | {condition} | — | — | — | — | no reference |")
                continue
            reference = load_rgb(reference_path)

            measured = {}
            for label, _technique in DIRECT_ARMS:
                directory = root / name / condition / label
                if not directory.exists():
                    continue
                by_checkpoint = score_arm(directory, reference)
                if by_checkpoint:
                    last = by_checkpoint[max(by_checkpoint)]
                    measured[label] = {"flip": aggregate(last["flip"]), "ms": aggregate(last["ms"])}

            baseline = measured.get("PT", {}).get("flip", {}).get("mean")
            for label, _technique in DIRECT_ARMS:
                if label not in measured:
                    continue
                flip, ms = measured[label]["flip"], measured[label]["ms"]
                ratio = "—"
                if baseline and label != "PT" and flip["mean"] > 0:
                    ratios[condition] = baseline / flip["mean"] * 100
                    ratio = f"{ratios[condition]:.0f} %"
                lines.append(f"| {name} | {condition} | {label} | {ms['mean']:.3f} | "
                             f"{flip['mean']:.5f} | ±{flip['ci95']:.5f} | {ratio} |")
                frames[(condition, label)] = ms["mean"]
        if len(ratios) == 2:
            # How much cheaper each arm's frame got when direct light left the image. The
            # two are not equal, and the difference moves the equal-time ratio on its own:
            # dropping direct light removes NEE work, which is a larger share of the path
            # tracer's frame than of the guided one, so the ablation hands PT more frames.
            cheaper = {label: frames[("indirect-only", label)] / frames[("with-direct", label)] - 1.0
                       for label, _t in DIRECT_ARMS
                       if ("indirect-only", label) in frames and ("with-direct", label) in frames}
            dilution.append((name, ratios["indirect-only"], ratios["with-direct"],
                             cheaper.get("PT"), cheaper.get("VXPG")))

    if dilution:
        lines += ["", "## What direct light costs the comparison", "",
                  "The expectation this control tests is that direct light DILUTES the comparison:",
                  "it is sampled identically in both arms, so it is a component the guide cannot",
                  "improve while the metric still averages over it. A positive 'dilution' would",
                  "mean the guided arm looks better once that component is removed.", "",
                  "The last two columns are why the answer is not that simple. Removing direct",
                  "light also removes NEE work, which is a larger share of the path tracer's frame",
                  "than of the guided one, so the ablation hands PT more frames and moves the",
                  "equal-time ratio on its own — in the opposite direction.", "",
                  "| cell | indirect-only | with direct | dilution | PT frame | VXPG frame |",
                  "|---|---|---|---|---|---|"]
        for name, indirect, whole, pt_cheaper, guided_cheaper in sorted(
                dilution, key=lambda d: d[1] - d[2], reverse=True):
            def percent(value):
                return "—" if value is None else f"{value * 100:+.1f} %"
            lines.append(f"| {name} | {indirect:.0f} % | {whole:.0f} % | "
                         f"{indirect - whole:+.0f} points | {percent(pt_cheaper)} | "
                         f"{percent(guided_cheaper)} |")

    out_path = root / "direct.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")


# ------------------------------------------------- section 16.10: weighting strategy

# The three ways the top-level tree can weight a cluster for a superpixel. The first two
# are SIByL's own pair and both consult visibility; IntensityOnly is PLAN_BADAWCZY 11's
# "proportional to power" strategy, which does not. Its cheaper frame — the whole
# cluster-visibility stage leaves the graph — is part of the answer rather than a
# confound: a strategy that ignores a signal does not pay for producing it.
IMPORTANCE_MODES = ["BinaryVisibility", "AverageVisibility", "IntensityOnly"]

# What the engine ships, and therefore the row every other mode is read against.
IMPORTANCE_BASELINE = "AverageVisibility"


def run_importance(args, manifest):
    """Quality AND cost against the top-level weighting strategy, guided arm only.

    Same shape as `grid`, for the same reason: the reference is path-traced and so is
    independent of anything the guide does, and one budget for all three modes makes the
    comparison equal-time by construction."""
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        for mode in IMPORTANCE_MODES:
            out = root / name / mode
            if (under(out) / "done.json").exists():
                print(f"exists  {name} {mode}")
                continue
            code, seconds = engine_checked(
                scene, light, manifest, "Guided Path Tracing (VXPG)",
                [f"vxpg.topLevelTree.importance={mode}"], out,
                f"seconds:{args.seconds}", args.images, args.warmup,
                checkpoints=args.checkpoints,
                log_path=absolute_root / f"{name}-{mode}.log")
            if code != 0:
                print(f"FAILED  {name} {mode} exit={code}")
                continue
            (under(out) / "done.json").write_text(json.dumps({"seconds": seconds,
                                                              "importance": mode}))
            print(f"ok      {name} {mode} in {seconds:.0f}s")


def run_importance_report(args, manifest):
    """One table per cell: cost, quality and the ratio against the shipped mode."""
    root = under(args.out)
    references = under(args.reference_dir)
    lines = ["# Top-level weighting strategy (section 16 item 10)", "",
             "Guided arm only, one budget for every mode, scored against the cell's own",
             f"path-traced reference with FLIP {flip_settings()['version']} LDR at "
             f"{flip_settings()['ppd']:.2f} ppd.",
             f"'vs {IMPORTANCE_BASELINE}' above 100 % means the mode is",
             "BETTER than what the engine ships; the ms/frame column is where the",
             "visibility-free strategy earns its keep.", "",
             f"| cell | mode | ms/frame | FLIP (last) | ci95 | vs {IMPORTANCE_BASELINE} |",
             "|---|---|---|---|---|---|"]

    verdicts = []
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        reference_path = references / f"{name}.png"
        if not reference_path.exists():
            lines.append(f"| {name} | — | — | — | — | no reference |")
            continue
        reference = load_rgb(reference_path)

        measured = {}
        for mode in IMPORTANCE_MODES:
            directory = root / name / mode
            if not directory.exists():
                continue
            by_checkpoint = score_arm(directory, reference)
            if not by_checkpoint:
                continue
            last = by_checkpoint[max(by_checkpoint)]
            measured[mode] = {"flip": aggregate(last["flip"]), "ms": aggregate(last["ms"])}

        baseline = measured.get(IMPORTANCE_BASELINE, {}).get("flip", {}).get("mean")
        for mode in (mode for mode in IMPORTANCE_MODES if mode in measured):
            flip, ms = measured[mode]["flip"], measured[mode]["ms"]
            ratio = f"{baseline / flip['mean'] * 100:.0f}%" if baseline else "—"
            lines.append(f"| {name} | {mode} | {ms['mean']:.3f} | {flip['mean']:.5f} | "
                         f"±{flip['ci95']:.5f} | {ratio} |")

        if baseline and "IntensityOnly" in measured:
            free = measured["IntensityOnly"]
            floor = NOISE_INFLATION * max(free["flip"]["ci95"],
                                          measured[IMPORTANCE_BASELINE]["flip"]["ci95"])
            gap = baseline - free["flip"]["mean"]
            verdicts.append((name, gap, floor,
                             free["ms"]["mean"] / measured[IMPORTANCE_BASELINE]["ms"]["mean"] - 1.0))

    if verdicts:
        lines += ["", "## Verdict per cell", "",
                  "The frame saving is real everywhere — the cluster-visibility stage leaves the "
                  "graph — so the only question is whether the quality difference at equal time "
                  f"clears {NOISE_INFLATION}x the within-run confidence interval. A positive gap "
                  "means the visibility-free strategy is BETTER.", "",
                  "| cell | frame | FLIP gap | noise floor | verdict |", "|---|---|---|---|---|"]
        for name, gap, floor, frame in sorted(verdicts, key=lambda v: -v[1]):
            if abs(gap) <= floor:
                verdict = "tie"
            else:
                verdict = "visibility-free wins" if gap > 0 else "visibility gate earns its cost"
            lines.append(f"| {name} | {frame * 100:+.1f}% | {gap:+.5f} | {floor:.5f} | {verdict} |")

    out_path = root / "importance.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")


# ------------------------------------------------------------------ minimal-sample ratio

# One guided frame costs about this many path-traced frames. 5 is the figure to test;
# the biased variant drops one ray chain per pixel and measured 2.3-4.9x, so it is paired
# at 4. Both are stated rather than derived per scene, because the point of this run is a
# FIXED proportion whose wall-time cost is then read off, not a proportion tuned to hide
# the difference.
RATIO_VXPG = 5
RATIO_REUSE = 4


def run_ratio(args, manifest):
    """The same comparison at a handful of samples instead of a 30 s accumulation.

    The guided arm renders `step, 2*step, ... , frames` frames; path tracing renders the
    same ladder multiplied by the arm's ratio, so every checkpoint pairs EXACTLY. Wall
    time is recorded at each rung, which is what turns a frame proportion back into the
    only currency that matters.
    """
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    plan = [("VXPG", "Guided Path Tracing (VXPG)", [], 1),
            ("REUSE", "Guided Path Tracing (VXPG)", ["vxpg.injection.reuseInMis=1"], 1),
            ("PT-x5", "Path Tracing", [], RATIO_VXPG),
            ("PT-x4", "Path Tracing", [], RATIO_REUSE)]
    for scene, light in cells(manifest):
        for label, technique, cvars, multiple in plan:
            out = root / cell_id(scene, light) / label
            if (under(out) / "done.json").exists():
                print(f"exists  {cell_id(scene, light)} {label}")
                continue
            frames = args.frames * multiple
            step = args.step * multiple
            code, seconds = engine_checked(
                scene, light, manifest, technique, cvars, out,
                f"frames:{frames}", args.images, args.warmup,
                checkpoints=f"every:{step}",
                log_path=absolute_root / f"{cell_id(scene, light)}-{label}.log")
            if code != 0:
                print(f"FAILED  {cell_id(scene, light)} {label} exit={code}")
                continue
            (under(out) / "done.json").write_text(
                json.dumps({"seconds": seconds, "frames": frames, "multiple": multiple}))
            print(f"ok      {cell_id(scene, light)} {label} {frames} frames in {seconds:.0f}s")


# ------------------------------------------------------------------ indirect-only curves

# Measured guided-arm frame cost per cell (ms), from the 2026-08-27 measurement run. It is
# here so the first checkpoint of a convergence curve can be placed at ONE OR TWO guided
# samples rather than at a round fraction of the budget: the interesting end of a real-time
# curve is its first few frames, and a log ladder anchored at budget/100 skips straight past
# it (0.3 s of a 30 s budget is already 64 guided frames on veach-ajar).
GUIDED_MS = {
    "veach-ajar--own": 4.72, "veach-ajar--point": 4.61,
    "staircase--own": 7.66, "staircase--point": 7.58,
    "kitchen--own": 9.24, "bedroom--own": 9.57,
    "sponza--own": 14.30, "bistro-exterior--own": 36.52,
}


def curve_config_path(manifest, scene, light):
    return Path(manifest["configDir"]) / f"curve.{cell_id(scene, light)}.json"


def write_curve_config(manifest, scene, light):
    """The cell's config with indirect-only turned on.

    Indirect-only renders a DIFFERENT integral, so it needs its own config AND its own
    reference; scoring an indirect-only arm against the full-image reference would make
    every arm equally and enormously wrong, with nothing in the table to show why."""
    source = RAYTRACER_DIR / config_path(manifest, scene, light)
    config = json.loads(source.read_text(encoding="utf-8-sig"))
    config["indirectOnly"] = True
    path = RAYTRACER_DIR / curve_config_path(manifest, scene, light)
    path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    return path


def curve_checkpoints(cell, seconds, first_frames, count):
    """Geometric ladder from `first_frames` guided samples up to the full budget."""
    first = first_frames * GUIDED_MS[cell] / 1000.0
    if first <= 0 or first >= seconds:
        return f"log:{count}"
    ratio = (seconds / first) ** (1.0 / (count - 1))
    points = [first * ratio ** i for i in range(count)]
    return "list:" + ",".join(f"{p:.4f}" for p in points)


def run_curve(args, manifest):
    """One equal-time curve per cell with indirect illumination only.

    Two phases in one command because the second is worthless without the first: an
    indirect-only reference per cell, then the three arms against it.
    """
    root = Path(args.out)
    absolute_root = under(root)
    absolute_root.mkdir(parents=True, exist_ok=True)
    reference_root = Path(args.reference_out)
    absolute_references = under(reference_root)
    absolute_references.mkdir(parents=True, exist_ok=True)

    chosen = selected_cells(manifest, getattr(args, "cells", []))
    for scene, light in chosen:
        write_curve_config(manifest, scene, light)

    for scene, light in chosen:
        name = cell_id(scene, light)
        marker = absolute_references / f"{name}.png"
        if marker.exists():
            print(f"exists  reference {name}")
            continue
        out = reference_root / "raw" / name
        code, seconds = engine_checked(scene, light, manifest, "Path Tracing", [], out,
                               f"seconds:{args.reference_seconds}", 1, args.warmup,
                               config=curve_config_path(manifest, scene, light),
                               log_path=absolute_references / f"{name}.log")
        if code != 0:
            print(f"FAILED  reference {name} exit={code}")
            continue
        capture = newest_capture(under(out))
        if capture is None:
            print(f"FAILED  reference {name}: no capture")
            continue
        lit = lit_fraction(capture)
        sidecar = capture.with_suffix(".json")
        capture.replace(marker)
        if sidecar.exists():
            sidecar.replace(marker.with_suffix(".json"))
        print(f"ok      reference {name}  {lit * 100:.1f}% lit  in {seconds / 60:.1f} min")

    # A reference is worth rendering on its own: the direct-light control scores against
    # this set too, so one stale cell can be replaced without re-running every arm.
    if getattr(args, "references_only", False):
        return

    for scene, light in chosen:
        name = cell_id(scene, light)
        checkpoints = curve_checkpoints(name, args.seconds, args.first_frames, args.points)
        for label, technique, cvars in ARMS:
            out = root / name / label
            if (under(out) / "done.json").exists():
                print(f"exists  {name} {label}")
                continue
            code, seconds = engine_checked(scene, light, manifest, technique, cvars, out,
                                   f"seconds:{args.seconds}", args.images, args.warmup,
                                   checkpoints=checkpoints,
                                   config=curve_config_path(manifest, scene, light),
                                   log_path=absolute_root / f"{name}-{label}.log")
            if code != 0:
                print(f"FAILED  {name} {label} exit={code}")
                continue
            (under(out) / "done.json").write_text(
                json.dumps({"seconds": seconds, "checkpoints": checkpoints}))
            print(f"ok      {name} {label} in {seconds:.0f}s")


# ------------------------------------------------------------------ reporting

def score_arm(directory, reference):
    """Every capture of one arm, scored and aggregated per checkpoint ordinal."""
    by_checkpoint = {}
    for png in sorted(directory.rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        sidecar = sidecar_for(png)
        benchmark = sidecar.get("benchmark", {})
        checkpoint = benchmark.get("checkpointIndex", 0)
        scores, _ = score_image(reference, png)
        entry = by_checkpoint.setdefault(checkpoint, {"flip": [], "ms": [], "seconds": 0.0})
        entry["flip"].append(scores["flipMean"])
        entry["ms"].append(benchmark.get("meanFrameMs", 0.0))
        entry["seconds"] = sidecar.get("raytracing", {}).get("accumulatedTime", 0.0)
    return by_checkpoint


def build_name():
    """Which build produced the data, for a report header.

    A report scores images that already exist, so it must not require an engine on disk —
    but it should still say which build it is describing when one is there."""
    return EXE.parent.name if EXE else "unknown"


def run_report(args, manifest):
    root = under(args.out)
    references = under(args.reference_dir)
    lines = ["# Reconnaissance — which cells are worth a campaign", "",
             f"Orientation only: {build_name()} build, PNG captures, "
             f"FLIP {flip_settings()['version']} LDR at "
             f"{flip_settings()['ppd']:.2f} pixels per degree. Values move under",
             "the final protocol; the ranking is what this table is for.", ""]
    lines.append("| cell | arm | ms/frame | FLIP (last) | ci95 | vs PT |")
    lines.append("|---|---|---|---|---|---|")

    verdicts = []
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        reference_path = references / f"{name}.png"
        if not reference_path.exists():
            lines.append(f"| {name} | — | — | — | — | no reference |")
            continue
        reference = load_rgb(reference_path)

        measured = {}
        for label, _, _ in ARMS:
            directory = root / name / label
            if not directory.exists():
                continue
            by_checkpoint = score_arm(directory, reference)
            if not by_checkpoint:
                continue
            last = by_checkpoint[max(by_checkpoint)]
            measured[label] = {"flip": aggregate(last["flip"]), "ms": aggregate(last["ms"])}

        baseline = measured.get("PT", {}).get("flip", {}).get("mean")
        for label in (label for label, _, _ in ARMS if label in measured):
            flip, ms = measured[label]["flip"], measured[label]["ms"]
            ratio = f"{baseline / flip['mean'] * 100:.0f}%" if baseline and label != "PT" else "—"
            lines.append(f"| {name} | {label} | {ms['mean']:.3f} | {flip['mean']:.5f} | "
                         f"±{flip['ci95']:.5f} | {ratio} |")

        if baseline and "VXPG" in measured:
            guided = measured["VXPG"]["flip"]
            gap = abs(baseline - guided["mean"])
            # The CI is computed across images WITHIN one process, which share a warm-up,
            # a clock state and a thermal state. Measured on this harness, that understates
            # the spread between separate runs by 3-10x, so a cell has to clear the inflated
            # floor, not the reported one, before its effect counts as real.
            floor = NOISE_INFLATION * max(guided["ci95"], measured["PT"]["flip"]["ci95"])
            verdicts.append((name, baseline / guided["mean"] * 100, gap, floor))

    if verdicts:
        lines += ["", "## Verdict per cell", "",
                  f"A cell earns campaign time when the gap between the arms clears "
                  f"{NOISE_INFLATION}x the within-run confidence interval — the factor is there because "
                  "a CI taken across images of one process understates the run-to-run spread by "
                  "3-10x on this harness. A cell below the floor is one where the campaign would "
                  "be measuring its own noise.", "",
                  "| cell | VXPG vs PT | gap | noise floor | verdict |", "|---|---|---|---|---|"]
        for name, ratio, gap, floor in sorted(verdicts, key=lambda v: -v[1]):
            verdict = "measure" if gap > floor else "below noise"
            lines.append(f"| {name} | {ratio:.0f}% | {gap:.5f} | {floor:.5f} | {verdict} |")

    out_path = root / "recon.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")


# ------------------------------------------------------------------ entry point

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_build_argument(parser)
    parser.add_argument("--manifest", default="tools/recon-manifest.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("configs", help="write one headless config per cell")

    smoke = sub.add_parser("smoke", help="5 s per cell: does it load, is it lit, does anything truncate")
    smoke.add_argument("--seconds", type=float, default=5.0)
    smoke.add_argument("--out", default="SavedUserData/Screenshots/recon-smoke")
    smoke.add_argument("--min-lit", type=float, default=0.02)

    references = sub.add_parser("references", help="one path-traced reference per cell")
    references.add_argument("--seconds", type=float, default=600.0)
    references.add_argument("--out", default="SavedUserData/Screenshots/recon-refs")
    references.add_argument("--min-lit", type=float, default=0.02)

    measure = sub.add_parser("run", help="the reconnaissance measurement")
    measure.add_argument("--seconds", type=float, default=30.0)
    measure.add_argument("--images", type=int, default=2)
    measure.add_argument("--warmup", type=float, default=45.0)
    measure.add_argument("--checkpoints", default="log:6")
    measure.add_argument("--out", default="SavedUserData/Screenshots/recon")

    report = sub.add_parser("report", help="score everything into one table")
    report.add_argument("--out", default="SavedUserData/Screenshots/recon")
    report.add_argument("--reference-dir", default="SavedUserData/Screenshots/recon-refs")

    grid = sub.add_parser("grid", help="K5: quality and cost against voxel grid resolution")
    grid.add_argument("--seconds", type=float, default=30.0)
    grid.add_argument("--images", type=int, default=2)
    grid.add_argument("--warmup", type=float, default=45.0)
    grid.add_argument("--checkpoints", default="log:6")
    grid.add_argument("--out", default="SavedUserData/Screenshots/recon-grid")
    grid.add_argument("--indirect", action="store_true",
                      help="use the indirect-only configs and a ladder starting at 1-2 guided samples")
    grid.add_argument("--first-frames", type=int, default=2)
    grid.add_argument("--points", type=int, default=9)

    direct = sub.add_parser("direct", help="section 16.9: both arms with and without direct light")
    direct.add_argument("--seconds", type=float, default=30.0)
    direct.add_argument("--images", type=int, default=2)
    direct.add_argument("--warmup", type=float, default=45.0)
    direct.add_argument("--checkpoints", default="log:6")
    direct.add_argument("--cells", type=lambda v: v.split(","), default=[],
                        help="cell ids to run (default: all)")
    direct.add_argument("--out", default="SavedUserData/Screenshots/recon-direct")

    direct_report = sub.add_parser("direct-report", help="score the direct-light control")
    direct_report.add_argument("--cells", type=lambda v: v.split(","), default=[])
    direct_report.add_argument("--out", default="SavedUserData/Screenshots/recon-direct")

    importance = sub.add_parser("importance", help="section 16.10: the three top-level weighting strategies")
    importance.add_argument("--seconds", type=float, default=30.0)
    importance.add_argument("--images", type=int, default=2)
    importance.add_argument("--warmup", type=float, default=45.0)
    importance.add_argument("--checkpoints", default="log:6")
    importance.add_argument("--out", default="SavedUserData/Screenshots/recon-importance")

    importance_report = sub.add_parser("importance-report", help="score the weighting-strategy run")
    importance_report.add_argument("--out", default="SavedUserData/Screenshots/recon-importance")
    importance_report.add_argument("--reference-dir", default="SavedUserData/Screenshots/recon-refs")

    ratio = sub.add_parser("ratio", help="minimal-sample comparison at a fixed frame proportion")
    ratio.add_argument("--frames", type=int, default=32, help="guided-arm frame budget")
    ratio.add_argument("--step", type=int, default=4, help="guided-arm checkpoint spacing")
    ratio.add_argument("--images", type=int, default=4)
    ratio.add_argument("--warmup", type=float, default=45.0)
    ratio.add_argument("--out", default="SavedUserData/Screenshots/recon-ratio")

    curve = sub.add_parser("curve", help="indirect-only equal-time curves from 1-2 guided samples")
    curve.add_argument("--seconds", type=float, default=30.0)
    curve.add_argument("--reference-seconds", type=float, default=600.0)
    curve.add_argument("--first-frames", type=int, default=2,
                       help="guided samples the first checkpoint should contain")
    curve.add_argument("--points", type=int, default=9)
    curve.add_argument("--images", type=int, default=2)
    curve.add_argument("--warmup", type=float, default=45.0)
    curve.add_argument("--out", default="SavedUserData/Screenshots/recon-curve")
    curve.add_argument("--reference-out", default="SavedUserData/Screenshots/recon-curve-refs")
    curve.add_argument("--cells", type=lambda v: v.split(","), default=[],
                       help="cell ids to run (default: all)")
    curve.add_argument("--references-only", action="store_true",
                       help="stop after the indirect-only references")

    args = parser.parse_args()
    manifest = load_manifest(args.manifest)

    global EXE
    # A report only reads images, so a missing engine must not stop it — but when one is
    # present its name belongs in the header. Everything else needs an engine and says so.
    if args.command in ("report", "importance-report", "direct-report"):
        try:
            EXE = resolve_exe(getattr(args, "build", None))
        except SystemExit:
            EXE = None
    else:
        EXE = resolve_exe(getattr(args, "build", None))

    if args.command == "configs":
        return write_configs(manifest)
    if args.command == "smoke":
        return run_smoke(args, manifest)
    if args.command == "references":
        return run_references(args, manifest)
    if args.command == "run":
        return run_measurement(args, manifest)
    if args.command == "grid":
        return run_grid(args, manifest)
    if args.command == "direct":
        return run_direct(args, manifest)
    if args.command == "direct-report":
        return run_direct_report(args, manifest)
    if args.command == "importance":
        return run_importance(args, manifest)
    if args.command == "importance-report":
        return run_importance_report(args, manifest)
    if args.command == "ratio":
        return run_ratio(args, manifest)
    if args.command == "curve":
        return run_curve(args, manifest)
    if args.command == "report":
        return run_report(args, manifest)


if __name__ == "__main__":
    sys.exit(main() or 0)
