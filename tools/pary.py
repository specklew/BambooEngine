#!/usr/bin/env python3
"""Equal-time frame pairs, and the check that they really are equal-time.

A time budget cannot equalise time at the scale of one display frame: the engine stops at the
first frame REACHING the budget, so each arm overshoots by a partial frame and the two
overshoots differ. The protocols therefore state FRAME COUNTS, and this is what produces them.

  plan    N1, N2 per cell from the measured frame costs, chosen so N1*c1 and N2*c2 land as
          close together as possible inside the plan's real-time window;
  verify  runs those counts and reads back what each arm actually spent, which is the only
          thing that settles whether the pair is equal-time;
  solve   re-solves from what verify measured, closing the loop in the regime that matters.

Why the loop exists: the cost table is measured under a 10 s load, and a 25 ms burst is a
different regime (the burst is ~2 % cheaper per frame on the path-traced arm and ~5 % on the
guided one). An earlier version of this table was built from burst measurements taken before
the frame-window accounting was fixed, and those were 3 % low on one arm and 17 % low on the
other - which handed the guided arm 21 % more frames than equal time allows on Cornell Box.
That is the failure this script's `verify` step exists to catch.

  python tools/pary.py plan --costs .../koszt-klatki.json --out .../klatki-rownego-czasu.json
  python tools/pary.py verify --pairs .../klatki-rownego-czasu.json
  python tools/pary.py solve --measured .../weryfikacja.json --out .../klatki-rownego-czasu.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign  # noqa: E402
import recon  # noqa: E402
from bench_report import add_build_argument, resolve_exe, sidecar_for  # noqa: E402
from recon import engine_checked, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def solve_pair(costFirst, costSecond, low, high, surcharge):
    """Frame counts whose spent times match best inside [low, high] milliseconds.

    `surcharge` is the opening cost of a window, in frames. The window is armed on a flushed
    queue, so its first frame has nothing to overlap with and runs about 20 % slower than the
    rest - measured as +0.85 ms on a 4.09 ms guided frame. It is a fixed cost per window, so as
    a share of the total it is large exactly where an arm has few frames, which is the guided
    arm. Ignoring it would leave the guided arm spending ~3 % more time than the pair promises.
    """
    best = None
    for first in range(1, 401):
        timeFirst = (first + surcharge) * costFirst
        if timeFirst > high:
            break
        if timeFirst < low:
            continue
        for second in range(1, 201):
            timeSecond = (second + surcharge) * costSecond
            if timeSecond > high:
                break
            if timeSecond < low:
                continue
            gap = abs(timeFirst - timeSecond) / max(timeFirst, timeSecond)
            # Ties go to the cheaper pair: the chapter is about real-time budgets, and a
            # shorter window is worth more than a marginally tighter match.
            key = (round(gap, 5), timeFirst + timeSecond)
            if best is None or key < best[0]:
                best = (key, first, second, timeFirst, timeSecond, gap)
    return best


def plan(args):
    costs = json.loads(Path(args.costs).read_text(encoding="utf-8-sig"))
    pairs = {}
    print(f"{'scena':<20}{'BSDF':>6}{'WIE':>6}{'t(BSDF)':>10}{'t(WIE)':>9}{'rozjazd':>9}"
          f"{'WIE-R':>7}{'t':>9}")
    for cell, perArm in sorted(costs.items()):
        if "BSDF" not in perArm or "WIE" not in perArm:
            continue
        best = solve_pair(perArm["BSDF"], perArm["WIE"], args.low, args.high,
                          args.surcharge)
        if not best:
            print(f"{cell:<20}  brak rozwiazania w oknie {args.low}-{args.high} ms")
            continue
        _key, first, second, timeFirst, timeSecond, gap = best
        entry = {"BSDF": first, "WIE": second,
                 "_msBSDF": round(timeFirst, 2), "_msWIE": round(timeSecond, 2),
                 "_gap": round(gap, 4)}
        line = (f"{cell:<20}{first:>6}{second:>6}{timeFirst:>9.2f}m{timeSecond:>8.2f}"
                f"{100 * gap:>8.1f}%")
        if "WIE-R" in perArm:
            # Matched to the guided arm's time, which is what M7 compares it against.
            target = timeSecond
            reuse = max(1, round(target / perArm["WIE-R"] - args.surcharge))
            entry["WIE-R"] = reuse
            entry["_msWIER"] = round((reuse + args.surcharge) * perArm["WIE-R"], 2)
            line += f"{reuse:>7}{entry['_msWIER']:>9.2f}"
        pairs[cell] = entry
        print(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    print(f"\nwritten: {out}")

    for factor in (2, 4, 8):
        scaled = {cell: {arm: value * factor for arm, value in entry.items()
                         if not arm.startswith("_")}
                  for cell, entry in pairs.items() if cell in args.ladder}
        for entry in scaled.values():
            entry["_note"] = (f"{factor}x the equal-time pair; the ratio between the arms is "
                              "preserved, so the time stays matched while the budget grows")
        target = out.parent / f"klatki-x{factor}.json"
        target.write_text(json.dumps(scaled, indent=1), encoding="utf-8")
        print(f"written: {target}")


def verify(args, manifest):
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8-sig"))
    parameters = campaign.load_parameters(args.parameters)
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    measured = {}

    for cell, scene, light in campaign.campaign_cells(manifest, args.only):
        if cell not in pairs:
            continue
        for arm in args.arms.split(","):
            frames = pairs[cell].get(arm)
            if not frames:
                continue
            out = root / cell / arm
            if not (under(out) / "done.json").exists():
                spec = manifest["arms"][arm]
                print(f"  {cell} {arm} x{frames} ...", flush=True)
                code, _took = engine_checked(
                    scene, light, manifest, spec["technique"],
                    list(spec["cvars"]) + campaign.scene_cvars(cell, parameters), out,
                    f"frames:{frames}", args.images, args.warmup,
                    log_path=under(root) / f"{cell}-{arm}.log", settle=args.settle,
                    config=campaign.config_path(manifest, scene, light))
                if code != 0:
                    print(f"  FAILED {cell} {arm} exit={code}")
                    continue
                (under(out) / "done.json").write_text(json.dumps({"frames": frames}))
            times = []
            for png in sorted(under(out).rglob("*.png")):
                if png.name.endswith(".flip.png"):
                    continue
                times.append(sidecar_for(png).get("raytracing", {})
                             .get("accumulatedTime", 0.0) * 1000.0)
            if times:
                mean = sum(times) / len(times)
                measured.setdefault(cell, {})[arm] = {"frames": frames, "ms": mean,
                                                      "msPerFrame": mean / frames}

    print(f"\n{'scena':<20}{'t(BSDF)':>10}{'t(WIE)':>10}{'rozjazd':>10}")
    for cell, perArm in sorted(measured.items()):
        if "BSDF" in perArm and "WIE" in perArm:
            first, second = perArm["BSDF"]["ms"], perArm["WIE"]["ms"]
            gap = abs(first - second) / max(first, second)
            print(f"{cell:<20}{first:>9.2f}m{second:>9.2f}m{100 * gap:>9.1f}%")
    target = under(root) / "weryfikacja.json"
    target.write_text(json.dumps(measured, indent=1), encoding="utf-8")
    print(f"\nwritten: {target}")


def solve(args):
    measured = json.loads(Path(args.measured).read_text(encoding="utf-8-sig"))
    costs = {cell: {arm: entry["msPerFrame"] for arm, entry in perArm.items()}
             for cell, perArm in measured.items()}
    scratch = Path(args.out).parent / "koszt-klatki-burst.json"
    scratch.write_text(json.dumps(costs, indent=1), encoding="utf-8")
    args.costs = str(scratch)
    plan(args)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="tools/ewaluacja-campaign.json")
    parser.add_argument("--only", metavar="M1", type=lambda v: v.split(","), default=None)
    parser.add_argument("--costs",
                        default="Raytracer/SavedUserData/Screenshots/parametry-pass3/koszt-klatki.json")
    parser.add_argument("--pairs",
                        default="Raytracer/SavedUserData/Screenshots/parametry-testowania/klatki-rownego-czasu.json")
    parser.add_argument("--measured",
                        default="Raytracer/SavedUserData/Screenshots/pary-weryfikacja/weryfikacja.json")
    parser.add_argument("--out",
                        default="Raytracer/SavedUserData/Screenshots/parametry-testowania/klatki-rownego-czasu.json")
    parser.add_argument("--parameters",
                        default="Raytracer/SavedUserData/Screenshots/parametry-pass2/parameters.json")
    parser.add_argument("--arms", default="BSDF,WIE,WIE-R")
    parser.add_argument("--surcharge", type=float, default=0.2,
                        help="opening cost of a measurement window, in frames")
    parser.add_argument("--low", type=float, default=17.0)
    parser.add_argument("--high", type=float, default=33.0)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--warmup", type=float, default=45.0)
    parser.add_argument("--settle", type=int, default=2)
    parser.add_argument("--ladder", type=lambda v: v.split(","),
                        default=["staircase--own", "san-miguel--own"])
    add_build_argument(parser)
    parser.add_argument("command", choices=["plan", "verify", "solve"])

    args = parser.parse_args()
    if args.command == "plan":
        plan(args)
        return
    recon.EXE = resolve_exe(getattr(args, "build", None))
    manifest = campaign.load_manifest(REPO_ROOT / args.manifest)
    if args.command == "verify":
        args.out = "SavedUserData/Screenshots/pary-weryfikacja"
        verify(args, manifest)
    else:
        solve(args)


if __name__ == "__main__":
    main()
