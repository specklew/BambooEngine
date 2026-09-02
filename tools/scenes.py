#!/usr/bin/env python3
"""Phase 0 of the second evaluation round: which scenes the chapter measures.

PLAN_EWALUACJI_v2 section 3 picks scenes along two axes, size and the SHARE OF THE SCENE
THAT CARRIES IRRADIANCE, and defines the second one as a number: lit voxels over voxels
holding geometry. Nothing printed that ratio before — the engine knew its lit-voxel count
and nothing counted the denominator — so this tool exists to produce it, per scene and per
grid rung, next to the screenshot that shows what the ratio means.

It ranks candidates and stops there. The chosen six are the user's decision, and until it
is made no reference image may be rendered: a reference is bound to a scene, a state and a
tone curve, and a list that moves afterwards invalidates every image scored against it.

Three facts shape the file:

* **One process per (scene, rung).** `--cvar-matrix` would sweep the rungs inside one
  process and save the big scenes their 60-90 s load, but a coarse rung leaves its buffers
  behind for the next one, and the top rungs allocate in gigabytes. One process per rung
  makes an out-of-memory failure land on the rung that caused it instead of the one after.

* **The census rides `vxpg.guiding.probe`.** Counting occupied cells walks the whole grid,
  so it is armed with the probe rather than run every frame; disarmed, the compaction
  kernel does not read the occupancy texture at all.

* **Every rung records its truncation warnings.** Past 131072 lit voxels the light tree
  silently drops the excess, so a fine rung on a big scene measures the cap and not the
  grid. A rung that truncates is reported as such rather than quietly averaged in.

Usage:
  python tools/scenes.py configs
  python tools/scenes.py probe
  python tools/scenes.py irradiance
  python tools/scenes.py report
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from PIL import Image
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

import recon  # noqa: E402
from bench_report import add_build_argument, resolve_exe, sidecar_for  # noqa: E402
from recon import RAYTRACER_DIR, engine_checked, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# The report is written in Polish and the Windows console defaults to cp1252, which cannot
# encode it. The FILE is always UTF-8; this only stops the echo from killing the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VXPG = "Guided Path Tracing (VXPG)"

# The census and the cluster dump are both one-shot and both disarm themselves; they are
# armed together because the two numbers are read from the same frame.
PROBE_CVARS = ["vxpg.guiding.probe=1", "vxpg.cluster.dumpStats=1"]

# Constants::Graphics::VOXEL_GUIDING_CAPACITY. A rung whose census passes it measures the
# cap and not the grid, whatever the log did or did not warn about.
VOXEL_GUIDING_CAPACITY = 131072


# ------------------------------------------------------------------ the manifest

def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for scene in manifest["scenes"]:
        for key in ("id", "state", "exposure"):
            if key not in scene:
                sys.exit(f"Manifest scene {scene.get('id', '?')} is missing '{key}'")
    return manifest


def config_path(manifest, scene):
    return Path(manifest["configDir"]) / f"scenes.{scene['id']}.json"


def rungs(scene, manifest):
    return scene.get("gridResolutions", manifest["defaultGridResolutions"])


def write_configs(manifest):
    out_dir = RAYTRACER_DIR / manifest["configDir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in manifest["scenes"]:
        config = dict(manifest["renderDefaults"])
        config["exposure"] = scene["exposure"]
        config.update(scene.get("overrides", {}))
        path = RAYTRACER_DIR / config_path(manifest, scene)
        path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        print(f"  {path.relative_to(RAYTRACER_DIR).as_posix()}")
    print(f"{len(manifest['scenes'])} configs written under {out_dir}")


# ------------------------------------------------------------------ reading the log

CENSUS = re.compile(r"\[VXPG census\] (\d+) lit voxels of (\d+) occupied by geometry")
CLUSTER = re.compile(r"\[VXPG cluster\] (\d+) lit voxels over (\d+)/(\d+) occupied clusters")
AABB = re.compile(r"Scene AABB: min=\(([-\d.,]+)\) max=\(([-\d.,]+)\)")
GEOMETRY = re.compile(r"Scene geometry: (\d+) triangles in (\d+) primitive")
VOXEL = re.compile(r"VoxelizationPass: gridMin=\S+ gridMax=\S+ voxelSize=([\d.]+)")


def triple(text):
    return [float(v) for v in text.split(",")]


def read_log(path):
    """What one probe run said about the scene and the grid.

    The scene is loaded twice in a headless run (the built-in default scene, then the one
    asked for), so every per-scene line is taken as the LAST match rather than the first.
    """
    out = {"warnings": []}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = CENSUS.search(line)
        if match:
            out["litVoxels"] = int(match.group(1))
            out["occupiedVoxels"] = int(match.group(2))
        match = CLUSTER.search(line)
        if match:
            out["clustersOccupied"] = int(match.group(2))
        match = AABB.search(line)
        if match:
            out["aabbMin"] = triple(match.group(1))
            out["aabbMax"] = triple(match.group(2))
        match = GEOMETRY.search(line)
        if match:
            out["triangles"] = int(match.group(1))
            out["primitives"] = int(match.group(2))
        match = VOXEL.search(line)
        if match:
            out["voxelMetres"] = float(match.group(1))
        if "exceeds the" in line:
            out["warnings"].append(line.split("] ", 1)[-1].strip())
    if "litVoxels" in out and out.get("occupiedVoxels"):
        out["litShare"] = out["litVoxels"] / out["occupiedVoxels"]
    # Derived from the census rather than from the log line, because the engine only grew
    # that warning partway through this probe: the number is the evidence, the line is not.
    if out.get("litVoxels", 0) > VOXEL_GUIDING_CAPACITY:
        out["truncated"] = True
    if "aabbMin" in out:
        out["extent"] = [b - a for a, b in zip(out["aabbMin"], out["aabbMax"])]
    return out


def frame_ms(directory):
    captures = [png for png in sorted(Path(directory).rglob("*.png"))
                if not png.name.endswith(".flip.png")]
    if not captures:
        return None
    return sidecar_for(captures[0]).get("benchmark", {}).get("meanFrameMs")


# ------------------------------------------------------------------ probe

def run_probe(args, manifest):
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)

    for scene in manifest["scenes"]:
        if args.only and scene["id"] not in args.only:
            continue
        for resolution in rungs(scene, manifest):
            out = root / scene["id"] / f"g{resolution}"
            marker = under(out) / "done.json"
            if marker.exists():
                print(f"exists  {scene['id']} {resolution}^3")
                continue
            log_path = under(root) / f"{scene['id']}-g{resolution}.log"
            print(f"  {scene['id']} {resolution}^3 ...", flush=True)
            code, took = engine_checked(
                scene, "own", manifest, VXPG,
                PROBE_CVARS + [f"voxel.gridDim={resolution}"], out,
                f"seconds:{args.seconds}", 1, args.warmup,
                log_path=log_path, config=config_path(manifest, scene))
            if code != 0:
                # A rung that will not run is a result: the plan says to drop a resolution a
                # scene cannot take rather than to measure it at different settings.
                (under(out)).mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"gridDim": resolution, "failed": True,
                                              "exit": code, "seconds": took}))
                print(f"  FAILED {scene['id']} {resolution}^3 exit={code} — rung dropped")
                continue
            marker.write_text(json.dumps({"gridDim": resolution, "seconds": took}))
            reading = read_log(log_path)
            share = reading.get("litShare")
            print(f"  ok     {scene['id']} {resolution}^3 in {took:.0f}s  "
                  f"lit {reading.get('litVoxels', '?')}/{reading.get('occupiedVoxels', '?')}"
                  + (f" = {100 * share:.1f}%" if share else ""))


# ------------------------------------------------------------------ irradiance screenshots

def run_irradiance(args, manifest):
    """One BufferDebugView::VoxelIrradiance capture per candidate.

    The view replaces the whole technique and bypasses tone mapping and accumulation, so
    the colours in the PNG are the ramp the debug shader writes and nothing else. The
    budget is short for the same reason: nothing accumulates.
    """
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    for scene in manifest["scenes"]:
        if args.only and scene["id"] not in args.only:
            continue
        resolution = args.grid or scene.get("probeGrid", 64)
        # The ramp saturates at a scene-dependent irradiance, so the scale is per scene:
        # one value for all of them would blow out the bright interiors or leave the large
        # exteriors black, and either way the screenshot would show nothing.
        heat = scene.get("heatScale", args.heat)
        out = root / scene["id"]
        marker = under(out) / "done.json"
        if marker.exists():
            print(f"exists  {scene['id']}")
            continue
        log_path = under(root) / f"{scene['id']}.log"
        print(f"  {scene['id']} at {resolution}^3, heatScale {heat} ...", flush=True)
        code, took = engine_checked(
            scene, "own", manifest, VXPG,
            [f"voxel.gridDim={resolution}", f"voxel.heatScale={heat}"], out,
            f"seconds:{args.seconds}", 1, args.warmup,
            log_path=log_path, config=config_path(manifest, scene),
            debug_views="VoxelIrradiance")
        if code != 0:
            print(f"  FAILED {scene['id']} exit={code}")
            continue
        marker.write_text(json.dumps({"gridDim": resolution, "heatScale": heat,
                                      "seconds": took}))
        print(f"  ok     {scene['id']} in {took:.0f}s  {describe_shot(under(out))}")


def describe_shot(directory):
    """The two acceptance numbers of the irradiance view, counted rather than eyeballed.

    White is the top of the ramp, so a large white area is lost information; the flat grey
    0.05 is a cell with no injected vertex, and the caption in the chapter talks about it,
    so it has to be visible.
    """
    captures = [png for png in sorted(Path(directory).rglob("*.png"))
                if not png.name.endswith(".flip.png")]
    if not captures:
        return "no capture"
    with Image.open(captures[-1]) as image:
        pixels = list(image.convert("RGB").getdata())
    total = len(pixels)
    white = sum(1 for p in pixels if min(p) >= 250)
    common = Counter(pixels).most_common(3)
    return (f"white {100.0 * white / total:.2f}%, top colours "
            + ", ".join(f"{c}={100.0 * k / total:.1f}%" for c, k in common))


def shot_stats(directory):
    captures = [png for png in sorted(Path(directory).rglob("*.png"))
                if not png.name.endswith(".flip.png")]
    if not captures:
        return None
    with Image.open(captures[-1]) as image:
        pixels = list(image.convert("RGB").getdata())
    total = len(pixels)
    grey = sum(1 for p in pixels if max(p) - min(p) <= 2 and 8 <= p[0] <= 20)
    return {
        "path": captures[-1].relative_to(RAYTRACER_DIR).as_posix(),
        "whiteShare": sum(1 for p in pixels if min(p) >= 250) / total,
        "emptyCellShare": grey / total,
        "topColours": [[list(c), k / total] for c, k in Counter(pixels).most_common(3)],
    }


# ------------------------------------------------------------------ estimator-side probe


def run_strategy(args, manifest):
    """Two estimator-side readings per candidate, from views the engine already paints.

    The lit-voxel share describes the SCENE. These two describe what the technique does with
    it, and they are the pair that explains a result rather than predicting it:

    * MisWeights paints R = the BSDF strategy's MIS weight and G = the guide's, so
      mean(G) / (mean(R) + mean(G)) is the share of the first-bounce estimate the guide
      carries. Read off the weights and not the two radiance views, because those are tone
      mapped and both of them also carry the direct light.
    * GuideAcceptance classifies every guided sample: accepted, traced and rejected for
      landing outside the chosen voxel, or rejected before tracing (below the horizon, zero
      pdf). The accepted share is the funnel's throughput, and it is the number that moved
      most between scenes in the reconnaissance.
    """
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    for scene in manifest["scenes"]:
        if args.only and scene["id"] not in args.only:
            continue
        out = root / scene["id"]
        marker = under(out) / "done.json"
        if marker.exists():
            print(f"exists  {scene['id']}")
            continue
        print(f"  {scene['id']} at {args.grid}^3 ...", flush=True)
        code, took = engine_checked(
            scene, "own", manifest, VXPG, [f"voxel.gridDim={args.grid}"], out,
            f"seconds:{args.seconds}", 1, args.warmup,
            log_path=under(root) / f"{scene['id']}.log",
            config=config_path(manifest, scene),
            debug_views="MisWeights,GuideAcceptance")
        if code != 0:
            print(f"  FAILED {scene['id']} exit={code}")
            continue
        marker.write_text(json.dumps({"gridDim": args.grid, "seconds": took}))
        print(f"  ok     {scene['id']} in {took:.0f}s  {strategy_stats(under(out))}")


def strategy_stats(directory):
    """The two numbers of one strategy run, or None when the captures are missing."""
    import numpy as np

    mis = next(iter(Path(directory).rglob("*MisWeights.png")), None)
    acceptance = next(iter(Path(directory).rglob("*GuideAcceptance.png")), None)
    out = {}
    if mis:
        pixels = np.asarray(Image.open(mis).convert("RGB"), dtype=np.float64) / 255.0
        red, green = pixels[..., 0].mean(), pixels[..., 1].mean()
        out["guideMisShare"] = float(green / (red + green)) if (red + green) > 0 else None
    if acceptance:
        pixels = np.asarray(Image.open(acceptance).convert("RGB"), dtype=np.float64) / 255.0
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        marked = (red + green + blue) > 0.02
        counts = [int(((green > red) & (green > blue) & marked).sum()),
                  int(((red > green) & (red > blue) & marked).sum()),
                  int(((blue > red) & (blue >= green) & marked).sum())]
        total = max(1, sum(counts))
        out["accepted"] = counts[0] / total
        out["gateRejected"] = counts[1] / total
        out["rejectedBeforeTrace"] = counts[2] / total
    if not out:
        return None
    return ", ".join(f"{k} {v:.3f}" for k, v in out.items() if v is not None)


# ------------------------------------------------------------------ report

def size_label(extent):
    """Three size classes by the longest side of the scene bound, in metres.

    A threshold, not a judgement: the chapter states the numbers next to the label, so a
    scene near a boundary can be read for what it is instead of arguing with the label.
    """
    longest = max(extent)
    if longest < 8.0:
        return "mała"
    if longest < 40.0:
        return "średnia"
    return "duża"


def run_report(args, manifest):
    root = Path(args.out)
    data = {}
    for scene in manifest["scenes"]:
        entry = {"state": scene["state"], "rungs": {}}
        for resolution in rungs(scene, manifest):
            marker = under(root / scene["id"] / f"g{resolution}") / "done.json"
            if not marker.exists():
                continue
            done = json.loads(marker.read_text())
            reading = read_log(under(root) / f"{scene['id']}-g{resolution}.log")
            if done.get("failed"):
                entry["rungs"][str(resolution)] = {"failed": True, "exit": done.get("exit")}
                continue
            reading["ms"] = frame_ms(under(root / scene['id'] / f"g{resolution}"))
            entry["rungs"][str(resolution)] = reading
            for key in ("triangles", "primitives", "aabbMin", "aabbMax", "extent"):
                if key in reading:
                    entry[key] = reading[key]
        shot = shot_stats(under(Path(args.irradiance_dir) / scene["id"]))
        if shot:
            entry["irradiance"] = shot
        if entry["rungs"]:
            data[scene["id"]] = entry

    lines = ["# Kandydaci na sceny — sonda proporcji oświetlonych wokseli", "",
             "Wytworzone przez `tools/scenes.py`. Proporcja = woksele z niezerowym licznikiem "
             "wstrzykniętych wierzchołków / woksele zawierające geometrię, oba liczone na tej "
             "samej siatce w tej samej klatce. Rozmiar woksela jest bezwzględny (metry sceny), "
             "bo to on, a nie rozdzielczość, mówi co siatka jest w stanie rozróżnić.", "",
             "| scena | stan | trójkąty | rozpiętość [m] | rozmiar |",
             "|---|---|---|---|---|"]
    for name, entry in data.items():
        extent = entry.get("extent")
        span = " × ".join(f"{e:.1f}" for e in extent) if extent else "—"
        lines.append(f"| {name} | {entry['state']} | {entry.get('triangles', '—')} | {span} | "
                     f"{size_label(extent) if extent else '—'} |")

    lines += ["", "## Proporcja oświetlonych wokseli", "",
              "| scena | " + " | ".join(f"{r}³" for r in manifest["defaultGridResolutions"]) + " |",
              "|---" * (1 + len(manifest["defaultGridResolutions"])) + "|"]
    for name, entry in data.items():
        cells = []
        for resolution in manifest["defaultGridResolutions"]:
            rung = entry["rungs"].get(str(resolution))
            if rung is None:
                cells.append("—")
            elif rung.get("failed"):
                cells.append("nie wystartowała")
            elif "litShare" in rung:
                mark = "!" if (rung.get("warnings") or rung.get("truncated")) else ""
                cells.append(f"{100 * rung['litShare']:.1f}%{mark}")
            else:
                cells.append("?")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"`!` = szczebel, na którym liczba oświetlonych wokseli przekracza "
                  f"{VOXEL_GUIDING_CAPACITY}-elementowy bufor kompaktacji: mierzy pułap, nie siatkę.", ""]

    lines += ["## Rozmiar woksela [m] i koszt klatki [ms]", "",
              "| scena | " + " | ".join(f"{r}³" for r in manifest["defaultGridResolutions"]) + " |",
              "|---" * (1 + len(manifest["defaultGridResolutions"])) + "|"]
    for name, entry in data.items():
        cells = []
        for resolution in manifest["defaultGridResolutions"]:
            rung = entry["rungs"].get(str(resolution)) or {}
            if rung.get("failed"):
                cells.append("—")
            elif "voxelMetres" in rung:
                ms = rung.get("ms")
                cells.append(f"{rung['voxelMetres']:.3f} / {ms:.2f}" if ms else f"{rung['voxelMetres']:.3f}")
            else:
                cells.append("—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    warned = [(name, resolution, w)
              for name, entry in data.items()
              for resolution, rung in entry["rungs"].items()
              for w in rung.get("warnings", [])]
    lines += ["", "## Obcięcia zgłoszone przez silnik", ""]
    if warned:
        lines += ["| scena | siatka | ostrzeżenie |", "|---|---|---|"]
        lines += [f"| {n} | {r}³ | {w} |" for n, r, w in warned]
    else:
        lines.append("Żadne.")

    lines += ["", "## Zrzuty irradiancji", "",
              "| scena | biel | komórki puste | plik |", "|---|---|---|---|"]
    for name, entry in data.items():
        shot = entry.get("irradiance")
        if not shot:
            lines.append(f"| {name} | — | — | — |")
            continue
        lines.append(f"| {name} | {100 * shot['whiteShare']:.2f}% | "
                     f"{100 * shot['emptyCellShare']:.1f}% | `{shot['path']}` |")

    out_md = under(root) / "scenes.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = under(root) / "scenes.json"
    out_json.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_md}\nwritten: {out_json}")


# ------------------------------------------------------------------ entry point

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "tools" / "scenes-manifest.json"))
    add_build_argument(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("configs", help="write one headless config per candidate")

    probe = sub.add_parser("probe", help="lit/occupied voxel census at every grid rung")
    probe.add_argument("--out", default="SavedUserData/Screenshots/scenes-probe")
    probe.add_argument("--seconds", type=float, default=2.0)
    probe.add_argument("--warmup", type=float, default=3.0)
    probe.add_argument("--only", nargs="*", default=None, help="candidate ids")

    shots = sub.add_parser("irradiance", help="VoxelIrradiance capture per candidate")
    shots.add_argument("--out", default="SavedUserData/Screenshots/scenes-irradiance")
    shots.add_argument("--grid", type=int, default=None)
    shots.add_argument("--heat", type=float, default=1.0)
    shots.add_argument("--seconds", type=float, default=2.0)
    shots.add_argument("--warmup", type=float, default=5.0)
    shots.add_argument("--only", nargs="*", default=None)

    strategy = sub.add_parser("strategy", help="guide MIS share and guided-sample acceptance")
    strategy.add_argument("--out", default="SavedUserData/Screenshots/scenes-strategy")
    strategy.add_argument("--grid", type=int, default=64)
    strategy.add_argument("--seconds", type=float, default=2.0)
    strategy.add_argument("--warmup", type=float, default=5.0)
    strategy.add_argument("--only", nargs="*", default=None)

    report = sub.add_parser("report", help="one table over the probe and the screenshots")
    report.add_argument("--out", default="SavedUserData/Screenshots/scenes-probe")
    report.add_argument("--irradiance-dir", default="SavedUserData/Screenshots/scenes-irradiance")

    args = parser.parse_args()
    recon.EXE = resolve_exe(getattr(args, "build", None))
    manifest = load_manifest(args.manifest)

    if args.command == "configs":
        write_configs(manifest)
    elif args.command == "probe":
        run_probe(args, manifest)
    elif args.command == "irradiance":
        run_irradiance(args, manifest)
    elif args.command == "strategy":
        run_strategy(args, manifest)
    elif args.command == "report":
        run_report(args, manifest)


if __name__ == "__main__":
    main()
