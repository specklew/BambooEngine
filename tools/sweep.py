"""Settings sweep: every combination of the runtime render settings, per scene.

Different from tools/campaign.py, and the difference is the point: campaign.py sweeps
COMPILE-TIME vendor levers, so every arm pays a pipeline rebuild and a process launch.
Everything here is a runtime CVar, so one engine process walks the whole cross product
(--cvar-matrix) and a settings point costs about a second instead of twenty.

References are per (bounces, skyLighting), not per scene. Both change what the image
converges to, so scoring a 1-bounce render against a 4-bounce reference would measure
the missing bounces and call it noise. Everything else in the grid — samples per pixel,
the guide's own switches — only moves noise and speed, so those share one reference.
"""

import argparse
import json
import statistics
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_report import aggregate, load_rgb, resolve_exe, score_image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXE = resolve_exe()
CWD = REPO / "Raytracer"

# scene -> (--scene argument, state, --config). The config carries the LIGHTS, and one
# light rig cannot serve both a small interior and Sponza: the default rig is a point
# light placed for veach-ajar, which leaves Sponza black rather than failing loudly.
SCENES = {
    "veach-ajar": ("veach-ajar", "Standard Look", "SavedUserData/headless.json"),
    "sponza": ("resources/models/Sponza/glTF/Sponza.gltf", "Standard Look",
               "SavedUserData/headless.sponza.json"),
}

# Runtime settings, split by whether they move the target or only the noise.
TARGET_DIMS = ["renderer.numBounces", "pathtracing.skyLighting"]
SHARED_NOISE_DIMS = {"renderer.samplesPerPixel": ["1", "2"]}
VXPG_DIMS = {
    "vxpg.tree.weightMode": ["0", "1"],
}

PT = "Path Tracing"
VXPG = "Guided Path Tracing (VXPG)"


def run(args, log, timeout=None):
    log.write(f"\n$ {' '.join(str(a) for a in args)}\n")
    log.flush()
    started = time.time()
    result = subprocess.run(args, cwd=str(CWD), stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    log.write(f"[exit {result.returncode} in {time.time() - started:.0f}s]\n")
    log.flush()
    return result.returncode


def newest_run_dir(after):
    shots = CWD / "SavedUserData" / "Screenshots"
    runs = [d for d in shots.glob("run-*") if d.is_dir() and d.stat().st_mtime >= after]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def ensure_reference(scene, state, bounces, sky, seconds, out_root, log, indirect_only=False):
    # The reference must carry the SAME illumination scope as what it scores, or an
    # indirect-only render is compared against a full one and every pixel of direct
    # light reads as error. Hence a separate key, not a separate flag on the same one.
    ref_dir = out_root / "refs" / scene / (f"b{bounces}_s{sky}" + ("_i" if indirect_only else ""))
    marker = ref_dir / "ref.png"
    if marker.exists():
        log.write(f"[reference exists] {marker}\n")
        return marker

    scene_arg, _, config = SCENES[scene]
    started = time.time()
    code = run([str(EXE), "--headless", "--config", config, "--scene", scene_arg, "--states", state,
                "--techniques", PT, "--budget", f"seconds:{seconds}", "--images", "1",
                "--warmup", "5", "--cvar", f"renderer.numBounces={bounces}",
                "--cvar", f"pathtracing.skyLighting={sky}",
                "--cvar", f"pathtracing.indirectOnly={1 if indirect_only else 0}"],
               log, timeout=seconds * 3 + 900)
    produced = newest_run_dir(started)
    if code != 0 or produced is None:
        log.write(f"[reference FAILED] {scene} b{bounces} s{sky}\n")
        return None

    ref_dir.mkdir(parents=True, exist_ok=True)
    png = next(produced.glob("*.png"), None)
    if png is None:
        log.write(f"[reference MISSING IMAGE] {scene} b{bounces} s{sky}\n")
        return None
    png.replace(marker)
    for leftover in produced.glob("*"):
        leftover.replace(ref_dir / leftover.name)

    # A reference that is essentially black scores every test image as near-perfect:
    # FLIP compares against it, and both are empty. That reads as a superb result, not
    # as a failure, and it is how a Sponza lit for the wrong scene produced 102
    # "identical" settings points. Refuse it here instead.
    if not reference_has_signal(marker, log):
        return None
    return marker


def reference_has_signal(path, log, min_lit_fraction=0.02, threshold=0.02):
    """Lit fraction rather than mean luminance: a reference can be legitimately dark
    overall (a night interior) but cannot be dark EVERYWHERE, and a single blown
    highlight would satisfy a mean test on its own."""
    import numpy as np
    image = load_rgb(path)
    luminance = np.asarray(image, dtype=np.float64) @ (0.2126, 0.7152, 0.0722)
    lit = float((luminance > threshold).mean())
    if lit < min_lit_fraction:
        log.write(f"[reference IS BLACK] {path} — only {lit * 100:.2f}% of pixels above "
                  f"{threshold}; the scene is probably lit for a different one\n")
        # Renamed, not left in place: the scoring pass keys off the file existing, so a
        # rejected reference that stays put is worse than no check at all. Kept on disk
        # under a new name because the image is the evidence of what went wrong.
        path.replace(path.with_name("ref.black.png"))
        return False
    log.write(f"[reference ok] {path} — {lit * 100:.1f}% lit\n")
    return True


def matrix_for(technique, bounces, skies, spp=None, vxpg_dims=None, indirect_only=False):
    dims = {"renderer.numBounces": bounces, "pathtracing.skyLighting": skies}
    if indirect_only:
        dims["pathtracing.indirectOnly"] = ["1"]
    dims.update({"renderer.samplesPerPixel": spp} if spp else SHARED_NOISE_DIMS)
    if technique == VXPG:
        dims.update(vxpg_dims if vxpg_dims else VXPG_DIMS)
    return ";".join(f"{name}={','.join(values)}" for name, values in dims.items())


def sweep(scene, state, technique, bounces, skies, out_root, images, seconds, log, run_tag="",
          spp=None, vxpg_dims=None, indirect_only=False):
    scene_arg, _, config = SCENES[scene]
    # run_tag separates one grid from another under a SHARED reference tree: a wider
    # grid reuses the expensive references but must not inherit the narrower grid's
    # "done" marker, and its captures must not be pooled with the narrower grid's.
    tag = ("pt" if technique == PT else "vx") + (f"-{run_tag}" if run_tag else "")
    out = out_root / "s" / scene / tag
    if (out / "done.json").exists():
        log.write(f"[sweep exists] {out}\n")
        return out

    started = time.time()
    code = run([str(EXE), "--headless", "--config", config, "--scene", scene_arg, "--states", state,
                "--techniques", technique, "--budget", f"seconds:{seconds}",
                "--images", str(images), "--warmup", "1",
                "--cvar-matrix", matrix_for(technique, bounces, skies, spp, vxpg_dims, indirect_only),
                "--out", out.relative_to(CWD).as_posix()], log)
    if code != 0:
        log.write(f"[sweep FAILED] {scene} / {technique}\n")
        return None
    (out / "done.json").write_text(json.dumps({"seconds": time.time() - started}))
    return out


def values_of(sidecar):
    return dict(pair.split("=", 1) for pair in
                sidecar.get("benchmark", {}).get("settings", "").split(";") if "=" in pair)


def settings_of(sidecar):
    """(bounces, sky) from the sidecar, for picking the matching reference. Read from
    the settings tag when present and from the render state otherwise, so a capture
    taken without a sweep still scores."""
    bench = sidecar.get("benchmark", {})
    values = dict(pair.split("=", 1) for pair in bench.get("settings", "").split(";") if "=" in pair)
    bounces = values.get("renderer.numBounces") or str(sidecar.get("raytracing", {}).get("bounces", 1))
    sky = values.get("pathtracing.skyLighting", "0")
    return bounces, sky


def score_sweep(scene, out_root, log, run_tag=""):
    rows = []
    references = {}
    roots = [d for d in (out_root / "s" / scene).glob("*")
             if d.is_dir() and (d.name.endswith(f"-{run_tag}") if run_tag else "-" not in d.name)]
    for png in sorted(png for root in roots for png in root.rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        sidecar = json.loads(png.with_suffix(".json").read_text())
        bounces, sky = settings_of(sidecar)
        indirect = values_of(sidecar).get("pathtracing.indirectOnly", "0") == "1"
        ref_path = out_root / "refs" / scene / (f"b{bounces}_s{sky}" + ("_i" if indirect else "")) / "ref.png"
        if not ref_path.exists():
            log.write(f"[no reference] {png.name} wants b{bounces} s{sky}\n")
            continue
        if ref_path not in references:
            references[ref_path] = load_rgb(ref_path)

        scores, _ = score_image(references[ref_path], png)
        bench = sidecar.get("benchmark", {})
        rt = sidecar.get("raytracing", {})
        scores["technique"] = sidecar.get("technique", "?")
        scores["settings"] = bench.get("settings", "")
        scores["frames"] = rt.get("frameIndex", 0)
        scores["meanFrameMs"] = bench.get("meanFrameMs", 0.0)
        rows.append(scores)
    return rows


def report(scene, rows, out_root, run_tag=""):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["technique"], row["settings"])].append(row)

    summarised = []
    for (technique, settings), scored in groups.items():
        summarised.append({
            "technique": technique,
            "settings": settings,
            "frames": statistics.fmean([s["frames"] for s in scored]),
            "meanFrameMs": statistics.fmean([s["meanFrameMs"] for s in scored]),
            "flip": aggregate([s["flipMean"] for s in scored]),
            "mse": aggregate([s["mse"] for s in scored]),
            "flipQ1": aggregate([s["flipQ1"] for s in scored]),
            "flipQ2": aggregate([s["flipMedian"] for s in scored]),
            "flipQ3": aggregate([s["flipQ3"] for s in scored]),
        })

    summarised.sort(key=lambda r: (r["technique"], r["flip"]["mean"]))
    suffix = f"-{run_tag}" if run_tag else ""
    (out_root / f"sweep-{scene}{suffix}.json").write_text(json.dumps(summarised, indent=1))

    print(f"\n## {scene} — {len(summarised)} settings points\n")
    print("| technique | settings | frames | ms | FLIP mean | ci95 | Q1 | Q2 | Q3 | MSE |")
    print("|" + "---|" * 10)
    for row in summarised[:20]:
        flip, mse = row["flip"], row["mse"]
        print(f"| {row['technique']} | {row['settings']} | {row['frames']:.0f} | "
              f"{row['meanFrameMs']:.2f} | {flip['mean']:.6f} | +-{flip['ci95']:.6f} | "
              f"{row['flipQ1']['mean']:.5f} | {row['flipQ2']['mean']:.5f} | "
              f"{row['flipQ3']['mean']:.5f} | {mse['mean']:.6f} |")


def prune_captures(scene, out_root, run_tag, log):
    """Delete a scene's captures once they are scored. 204 settings points x 10 images
    of 1920x1080 PNG is ~18 GB per grid, and filling the disk does not fail loudly — it
    truncates a PNG somewhere in the middle of the run and the scoring pass dies on an
    unreadable file hours later. The report JSON is the artifact worth keeping; the
    references stay because rescoring needs them."""
    freed = 0
    for root in (out_root / "s" / scene).glob("*"):
        if not root.is_dir():
            continue
        if run_tag and not root.name.endswith(f"-{run_tag}"):
            continue
        if not run_tag and "-" in root.name:
            continue
        for capture_dir in root.glob("run-*"):
            for f in capture_dir.rglob("*"):
                if f.is_file():
                    freed += f.stat().st_size
            shutil.rmtree(capture_dir, ignore_errors=True)
    log.write(f"[pruned] {scene} captures, freed {freed / 2**30:.1f} GB\n")


def main():
    parser = argparse.ArgumentParser(description="Sweep runtime render settings.")
    parser.add_argument("--out", default="SavedUserData/Screenshots/sweep")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--bounces", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--sky", nargs="+", default=["0"])
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=0.1)
    parser.add_argument("--ref-seconds", type=int, default=600)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--tag", default="", help="separates this grid from another sharing the references")
    parser.add_argument("--spp", nargs="+", help="override the samples-per-pixel axis")
    parser.add_argument("--vxpg-dim", action="append", default=[],
                        help="replace the VXPG axes: name=v1,v2 (repeatable)")
    parser.add_argument("--indirect-only", action="store_true",
                        help="render and reference indirect illumination only (paper Sec. 6)")
    args = parser.parse_args()

    out_root = (CWD / args.out).resolve()
    if CWD.resolve() not in out_root.parents:
        sys.exit(f"--out must live under {CWD}")
    out_root.mkdir(parents=True, exist_ok=True)

    with (out_root / "sweep.log").open("a", encoding="utf-8") as log:
        log.write(f"\n=== sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for scene in args.scenes:
            _, state, _config = SCENES[scene]
            if not args.score_only:
                for bounces in args.bounces:
                    for sky in args.sky:
                        ensure_reference(scene, state, bounces, sky, args.ref_seconds, out_root, log,
                                         args.indirect_only)
                custom = {}
                for spec in args.vxpg_dim:
                    name, _, values = spec.partition("=")
                    custom[name] = values.split(",")
                for technique in (PT, VXPG):
                    sweep(scene, state, technique, args.bounces, args.sky, out_root,
                          args.images, args.seconds, log, args.tag, args.spp, custom or None,
                          args.indirect_only)
            report(scene, score_sweep(scene, out_root, log, args.tag), out_root, args.tag)
            prune_captures(scene, out_root, args.tag, log)


if __name__ == "__main__":
    main()
