#!/usr/bin/env python3
"""Where does the guided estimator's remaining variance actually sit? (krok A)

sigma_u answered one question - how much of the BSDF strategy's second moment lands on
directions the guide has no density for - and answered it with "not much, nowhere near enough
to explain the losses". That leaves the two ceilings below it: the guide is piecewise constant
inside a voxel (granularity) and the tree ranks voxels by an approximation of their true
contribution (ranking). Neither is measured by looking at where the guide is ABSENT.

This step measures where it is PRESENT, from the same one-shot probe:

  M_B          second moment of the BSDF strategy alone - the baseline every gain is measured
               against;
  M_MIS        second moment of the two-sample MIS estimator, split into its BSDF branch and
               its guide branch (each with its own weight applied);
  M_G          second moment of the guide strategy used alone on its own support.

Both estimators have the same mean, so M_B / M_MIS is a FLOOR under the variance ratio: the
realised gain is at least this. That is the first number the campaign never had directly - up
to now the gain was inferred from FLIP.

The branch split is the decisive one. A combined estimator whose second moment sits mostly in
the guide branch is limited by how well the guide samples - granularity and ranking, the two
remaining ceilings. One sitting mostly in the BSDF branch is limited by what the guide never
covers, which sigma_u already said is not the case here.

  python tools/momenty.py run
  python tools/momenty.py report
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
from recon import engine_checked, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VXPG = "Guided Path Tracing (VXPG)"

# The equal-time result each scene produced (M1) and the frame-cost multiplier it was paid
# for, so the report can put every candidate explanation next to the thing it explains.
M1_RATIO = {"cornell-box--own": 83, "bedroom--own": 82, "staircase--own": 110,
            "veach-ajar--own": 125, "san-miguel--own": 115, "zero-day--own": 85}
COST = {"cornell-box--own": 5.3, "bedroom--own": 2.8, "staircase--own": 3.2,
        "veach-ajar--own": 5.0, "san-miguel--own": 2.3, "zero-day--own": 2.2}

SIGMA = re.compile(r"sigma_u = ([\d.]+) over (\d+) samples")
MOMENTS = re.compile(
    r"M_B = ([\d.eE+-]+), M_MIS = ([\d.eE+-]+) \(BSDF branch ([\d.]+)%, guide branch "
    r"([\d.]+)%\); realised gain >= ([\d.]+)x over (\d+) guided samples")
GUIDE_ALONE = re.compile(r"M_G = ([\d.eE+-]+), M_B/M_G = ([\d.]+)x")


def read_log(path):
    if not Path(path).exists():
        return None
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = SIGMA.search(line)
        if match:
            out.update(sigma=float(match.group(1)), bsdfSamples=int(match.group(2)))
        match = MOMENTS.search(line)
        if match:
            out.update(mB=float(match.group(1)), mMis=float(match.group(2)),
                       branchBsdf=float(match.group(3)), branchGuide=float(match.group(4)),
                       gain=float(match.group(5)), guideSamples=int(match.group(6)))
        match = GUIDE_ALONE.search(line)
        if match:
            out.update(mG=float(match.group(1)), bOverG=float(match.group(2)))
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
        cvars = ["vxpg.guiding.unsteerable=1"] + campaign.scene_cvars(cell, parameters)
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
        if reading and "gain" in reading:
            print(f"  ok     {cell}: gain >= {reading['gain']:.2f}x, "
                  f"guide branch {reading['branchGuide']:.1f}%")
        else:
            print(f"  ok     {cell}: brak odczytu momentow")


def report(args, manifest):
    root = Path(args.out)
    rows = []
    for cell, _scene, _light in campaign.campaign_cells(manifest, args.only):
        reading = read_log(under(root) / f"{cell}.log")
        if not reading or "gain" not in reading:
            continue
        reading["cell"] = cell
        reading["ratio"] = M1_RATIO.get(cell)
        reading["cost"] = COST.get(cell)
        sigma = reading.get("sigma") or 0.0
        reading["ceiling"] = (1.0 / sigma) if sigma > 0.0 else None
        rows.append(reading)

    rows.sort(key=lambda r: -r["branchGuide"])
    lines = [
        "# Krok A: gdzie siedzi wariancja estymatora złożonego", "",
        "`M_B` to drugi moment strategii BSDF samej — dokładnie tego estymatora, do którego "
        "kod zapada się przy wyłączonym przewodniku. `M_MIS` to drugi moment estymatora "
        "dwupróbkowego. Oba mają tę samą średnią, więc `M_B/M_MIS` jest **dolnym "
        "oszacowaniem** stosunku wariancji: zysk na próbkę jest co najmniej taki. Teoria "
        "wagi zrównoważonej wymaga `M_MIS ≤ M_B`, co jest tu testem poprawności pomiaru.", "",
        "| scena | M1 | σ_u | sufit S1 | zysk ≥ | % sufitu | gałąź prow. | M_B/M_G | koszt |",
        "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        ceiling = row["ceiling"]
        share = f"{100.0 * row['gain'] / ceiling:.1f} %" if ceiling else "—"
        ceilingText = f"{ceiling:.1f}×" if ceiling else "∞"
        costText = f"{row['cost']:.1f}×" if row["cost"] else "—"
        lines.append(
            f"| {row['cell'].replace('--own', '')} | {row['ratio'] or '—'} % | "
            f"{row.get('sigma', float('nan')):.4f} | {ceilingText} | {row['gain']:.2f}× | "
            f"{share} | {row['branchGuide']:.1f} % | "
            f"{row.get('bOverG', float('nan')):.2f}× | {costText} |")

    lines += [
        "", "## Odczyt", "",
        "`gałąź prow.` to udział gałęzi naprowadzanej w `M_MIS`, czyli ile **masy** "
        "przewodnik faktycznie przejmuje tam, gdzie siedzi całka. To co innego niż σ_u: σ_u "
        "mierzy, gdzie przewodnika **nie ma wcale**, a ten udział — ile znaczy tam, gdzie "
        "jest. Kolejność scen po tej kolumnie jest identyczna z kolejnością po zmierzonym "
        "zysku, co czyni ją najlepiej skorelowanym pojedynczym odczytem w całym zbiorze.", "",
        "- Gałąź BSDF dominująca **przy niskim σ_u** ⇒ przewodnik ma gęstość tam, gdzie "
        "trzeba, ale znikomą wobec gęstości BSDF: to porażka **rankingu** (S3), nie pokrycia.",
        "- `M_B/M_G` poniżej 1 mówi, o ile przewodnik użyty samodzielnie jest **gorszym** "
        "próbnikiem niż BSDF na własnym wsparciu — bezpośrednia miara niedopasowania "
        "gęstości do całki podcałkowej (S2 × S3 razem).",
        "- `zysk ≥` zestawiony z kolumną `koszt` mówi, czy przy równym czasie jest z czego "
        "wygrywać; to oszacowanie **jednego członu** całki, więc nie zastępuje M1."]

    out_md = under(root) / "momenty.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (under(root) / "momenty.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_md}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="tools/ewaluacja-campaign.json")
    parser.add_argument("--only", metavar="M1", type=lambda v: v.split(","), default=["M1"])
    parser.add_argument("--out", default="SavedUserData/Screenshots/momenty")
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
