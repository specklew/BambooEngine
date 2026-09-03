#!/usr/bin/env python3
"""Does the concentration of irradiance predict where the guided technique wins?

The scene table describes each scene by the SHARE of voxels that carry light. That number
turned out to be a descriptor and not a predictor: Cornell Box has the highest share in the
set and loses hardest, Veach Ajar has one of the lowest and wins hardest.

This asks a different question, one the guide's own mathematics cares about: not how MANY
voxels are lit, but how UNEVENLY the light is spread among them. A guide is only worth its
cost when sampling proportional to irradiance differs from sampling uniformly, and that
difference is exactly what a concentration measure captures.

Reported per scene, from the engine's one-shot probe:

  * the share of total irradiance held by the brightest 1 %, 5 % and 10 % of lit voxels,
  * the participation ratio (1 / sum of squared shares) - the number of voxels the
    distribution behaves as if it spread over evenly, which is 1 for a single spike and the
    full count for a flat distribution.

Measured on the target grid of each scene, because concentration depends on cell size: the
same light in coarser cells is by definition more concentrated.

  python tools/koncentracja.py run
  python tools/koncentracja.py report
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign  # noqa: E402
import recon  # noqa: E402
from bench_report import add_build_argument, resolve_exe  # noqa: E402
from recon import RAYTRACER_DIR, engine_checked, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VXPG = "Guided Path Tracing (VXPG)"

# The equal-time result each scene actually produced, so the report can put the candidate
# predictor next to the thing it is supposed to predict rather than in a table of its own.
M1_RATIO = {"cornell-box--own": 83, "bedroom--own": 82, "staircase--own": 110,
            "veach-ajar--own": 125, "san-miguel--own": 115, "zero-day--own": 85}

CONCENTRATION = re.compile(
    r"top 1% holds ([\d.]+)%, top 5% ([\d.]+)%, top 10% ([\d.]+)% of total irradiance; "
    r"effective support ([\d.]+) of (\d+) lit voxels \(([\d.]+)%\)")
CENSUS = re.compile(r"\[VXPG census\] (\d+) lit voxels of (\d+) occupied")


def read_log(path):
    if not Path(path).exists():
        return None
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = CONCENTRATION.search(line)
        if match:
            out.update(top1=float(match.group(1)), top5=float(match.group(2)),
                       top10=float(match.group(3)), effective=float(match.group(4)),
                       lit=int(match.group(5)), effectiveShare=float(match.group(6)))
        match = CENSUS.search(line)
        if match:
            out.update(litVoxels=int(match.group(1)), occupied=int(match.group(2)))
    return out or None


def run(args, manifest, parameters):
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    for cell, scene, light in campaign.campaign_cells(manifest, args.only):
        out = root / cell
        log_path = under(root) / f"{cell}.log"
        if (under(out) / "done.json").exists():
            print(f"exists  {cell}")
            continue
        cvars = ["vxpg.guiding.probe=1"] + campaign.scene_cvars(cell, parameters)
        print(f"  {cell} [{' '.join(cvars)}] ...", flush=True)
        code, took = engine_checked(
            scene, light, manifest, VXPG, cvars, out,
            f"seconds:{args.seconds}", 1, args.warmup, log_path=log_path,
            config=campaign.config_path(manifest, scene, light))
        if code != 0:
            print(f"  FAILED {cell} exit={code}")
            continue
        (under(out) / "done.json").write_text(json.dumps({"seconds": took}))
        reading = read_log(log_path)
        print(f"  ok     {cell}: " + (f"1% -> {reading['top1']:.1f}%" if reading else "brak odczytu"))


def report(args, manifest):
    root = Path(args.out)
    rows = []
    for cell, _scene, _light in campaign.campaign_cells(manifest, args.only):
        reading = read_log(under(root) / f"{cell}.log")
        if not reading:
            continue
        reading["cell"] = cell
        reading["ratio"] = M1_RATIO.get(cell)
        rows.append(reading)

    rows.sort(key=lambda r: -(r["ratio"] or 0))
    lines = ["# Koncentracja irradiancji jako kandydat na wskaźnik", "",
             "Udział całkowitej irradiancji przypadający na najjaśniejsze woksele, mierzony na "
             "nastawie docelowej każdej sceny. `wsparcie efektywne` to odwrotność sumy kwadratów "
             "udziałów: liczba wokseli, po których rozkład zachowuje się, jakby rozłożył się "
             "równo. Kolumna `VXPG/PT` to wynik pomiaru M1 przy równym czasie.", "",
             "| scena | VXPG/PT | 1 % | 5 % | 10 % | wsparcie efektywne | oświetlonych |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['cell'].replace('--own', '')} | "
            f"{row['ratio'] if row['ratio'] else '—'} % | "
            f"{row['top1']:.1f} % | {row['top5']:.1f} % | {row['top10']:.1f} % | "
            f"{row['effective']:.0f} ({row['effectiveShare']:.2f} %) | {row['lit']} |")

    winners = [r for r in rows if (r["ratio"] or 0) > 100]
    losers = [r for r in rows if (r["ratio"] or 0) <= 100]
    lines += ["", "## Czy to rozdziela wygrane od przegranych", ""]
    for label, group in (("wygrywające", winners), ("przegrywające", losers)):
        if not group:
            continue
        for key, name in (("top1", "1 %"), ("top5", "5 %"), ("top10", "10 %"),
                          ("effectiveShare", "wsparcie efektywne [%]")):
            values = [r[key] for r in group]
            lines.append(f"- {label}, {name}: od {min(values):.2f} do {max(values):.2f}")
    lines += ["", "Wskaźnik rozdziela zbiór wtedy i tylko wtedy, gdy przedziały wyżej **nie "
              "nachodzą na siebie**. Jeżeli nachodzą, jest opisem sceny, a nie predyktorem — "
              "tak jak proporcja oświetlonych wokseli."]

    out_md = under(root) / "koncentracja.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = under(root) / "koncentracja.json"
    out_json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_md}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="tools/ewaluacja-campaign.json")
    parser.add_argument("--only", metavar="M1", type=lambda v: v.split(","), default=["M1"])
    parser.add_argument("--out", default="SavedUserData/Screenshots/koncentracja")
    parser.add_argument("--parameters",
                        default="Raytracer/SavedUserData/Screenshots/parametry-pass2/parameters.json")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    add_build_argument(parser)
    parser.add_argument("command", choices=["run", "report"])

    args = parser.parse_args()
    recon.EXE = resolve_exe(getattr(args, "build", None))
    manifest = campaign.load_manifest(REPO_ROOT / args.manifest)
    if args.command == "run":
        run(args, manifest, campaign.load_parameters(args.parameters))
    else:
        report(args, manifest)


if __name__ == "__main__":
    main()
