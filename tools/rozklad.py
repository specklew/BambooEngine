#!/usr/bin/env python3
"""Krok B: rozbicie nieefektywności przewodnika na granulację i ranking.

Step A said the limiter is the SHAPE of the guide's density, not its support. Two things can
make that shape wrong, and they multiply:

  S2, granulation - inside one voxel the guide's density is constant by construction, so any
                    structure of the integrand WITHIN a voxel is invisible to it;
  S3, ranking     - the probability p_i of choosing voxel i comes from the light tree's
                    weighting, which is an approximation of the voxel's true contribution.

Write W = f cos L / pdfG for one guided draw. With the guide choosing voxel i with probability
p_i and then sampling uniformly by solid angle inside it:

    E[W]   = sum_i F1_i            = the integral itself, whatever p is  (M_cont = E[W]^2)
    E[W^2] = sum_i |Omega_i| F2_i / p_i                                  (M_act)
    M_opt  = (sum_i sqrt(|Omega_i| F2_i))^2   at the best possible p_i

and M_cont <= M_opt <= M_act, so the total inefficiency M_act/M_cont factors exactly into

    S2 = M_opt / M_cont     (granulation: what no choice of p_i can remove)
    S3 = M_act / M_opt      (ranking: what THIS choice of p_i costs on top)

Neither needs the solid angles. Grouping the recorded draws by the voxel that produced them
gives, per voxel, the empirical p_i and the first two moments m1_i, m2_i of W, and

    F1_i = p_i m1_i        |Omega_i| F2_i = p_i^2 m2_i

so S2 = (sum p_i sqrt(m2_i))^2 / (sum p_i m1_i)^2 and S3 = (sum p_i m2_i) / (sum p_i sqrt(m2_i))^2.
Both are >= 1 by Cauchy-Schwarz, which is the correctness test on the whole pipeline.

Reads guide-records.bin as written by `vxpg.guiding.records`.

  python tools/rozklad.py run
  python tools/rozklad.py report
"""

import argparse
import json
import struct
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign  # noqa: E402
import recon  # noqa: E402
from bench_report import add_build_argument, resolve_exe  # noqa: E402
from recon import RAYTRACER_DIR, engine_checked, under  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VXPG = "Guided Path Tracing (VXPG)"
RECORD_FILE = Path(RAYTRACER_DIR) / "guide-records.bin"
MAGIC = 0x31524742
CHAIN_OK = 0
OUTCOME_ABSENT = 0xFFFFFFFF

M1_RATIO = {"cornell-box--own": 83, "bedroom--own": 82, "staircase--own": 110,
            "veach-ajar--own": 125, "san-miguel--own": 115, "zero-day--own": 85}


def load_records(path):
    """Per (shading point, voxel) counts and the first two moments of W."""
    raw = Path(path).read_bytes()
    magic, stratifiedWidth, stratifiedHeight, slices = struct.unpack_from("<4I", raw, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: not a guide-record file (magic {magic:#x})")
    pixels = stratifiedWidth * stratifiedHeight
    slots = pixels * slices
    words = np.frombuffer(raw, dtype=np.uint32, offset=16)
    frames = len(words) // (slots * 4)
    words = words[:frames * slots * 4].reshape(frames * slots, 4)

    chain = words[:, 1]
    outcome = words[:, 3]
    live = (chain == CHAIN_OK) & (outcome != OUTCOME_ABSENT)
    absent = int(((chain == CHAIN_OK) & (outcome == OUTCOME_ABSENT)).sum())

    # The slice index is dropped: the slices are independent draws at the same shading point,
    # which is exactly what the decomposition wants pooled. Indices are materialised only for
    # the live rows - a 30 s run holds ~50 M records, and an int64 column over all of them is
    # 400 MB that is thrown away one line later.
    liveIndex = np.flatnonzero(live)
    pixel = (liveIndex % slots) % pixels
    voxel = words[liveIndex, 0].astype(np.int64)
    value = words[liveIndex, 2].view(np.float32).astype(np.float64)
    # Which half of the run each draw came from, for the cross-fitted estimate below.
    second = (liveIndex // slots) >= (frames // 2)

    group = (pixel << np.int64(32)) | voxel
    unique, inverse = np.unique(group, return_inverse=True)
    size = len(unique)
    counts = np.bincount(inverse, minlength=size).astype(np.float64)
    sums = np.bincount(inverse, weights=value, minlength=size)
    squares = np.bincount(inverse, weights=value * value, minlength=size)
    half = {}
    for name, mask in (("first", ~second), ("second", second)):
        half[name + "Count"] = np.bincount(inverse[mask], minlength=size).astype(np.float64)
        half[name + "Sum"] = np.bincount(inverse[mask], weights=value[mask], minlength=size)
        half[name + "Square"] = np.bincount(inverse[mask], weights=(value * value)[mask],
                                            minlength=size)
    return {"pixel": (unique >> np.int64(32)).astype(np.int64), "count": counts,
            "sum": sums, "square": squares, **half}, \
           {"frames": int(frames), "slots": int(slots), "drawn": int(live.sum()),
            "absent": absent, "stratifiedWidth": int(stratifiedWidth),
            "stratifiedHeight": int(stratifiedHeight)}


def _plugin(index, pixelCount, count, total, square):
    """(M_cont, M_opt, M_act) summed over shading points, from one set of per-voxel sums."""
    # F1_i = sum_i / N and |Omega_i| F2_i = p_i^2 m2_i, so p_i sqrt(m2_i) = count_i sqrt(m2_i) / N.
    # Nothing here divides by a probability, which is what made the cross-fitted version blow up.
    contSum = np.bincount(index, weights=total) / pixelCount
    actSum = np.bincount(index, weights=square) / pixelCount
    optSum = np.bincount(index, weights=count * np.sqrt(square / count)) / pixelCount
    usable = (contSum > 0.0) & (optSum > 0.0)
    return (float((contSum[usable] ** 2).sum()), float((optSum[usable] ** 2).sum()),
            float(actSum[usable].sum()), int(usable.sum()))


def richardson(grouped):
    """The split with the square-root bias extrapolated away.

    sqrt(m2) from n draws is biased low by O(1/n) - sqrt is concave - which understates M_opt
    and hands the difference to the ranking factor. The bias is visible directly: raising the
    per-voxel threshold cuts the reported ranking fivefold on some scenes. Since it goes as
    1/n, evaluating the same estimator on the full run and on half of it brackets it, and
    2 S(n) - S(n/2) cancels the leading term. Both evaluations use the SAME voxel set, so the
    difference is the bias and not a change of population.

    An earlier attempt evaluated an oracle policy fitted on one half against the other. It
    failed: with W as heavy-tailed as it is here, a policy fitted on half the draws is itself
    so noisy that its evaluated cost exceeded the actual policy's on every scene, and the
    defensive mixture that kept it finite was what the answer ended up measuring.
    """
    both = (grouped["firstCount"] > 0) & (grouped["secondCount"] > 0)
    if not both.any():
        return None
    index = np.unique(grouped["pixel"][both], return_inverse=True)[1]

    full = _plugin(index, np.bincount(index, weights=grouped["count"][both]),
                   grouped["count"][both], grouped["sum"][both], grouped["square"][both])
    halves = []
    for name in ("first", "second"):
        count = grouped[name + "Count"][both]
        halves.append(_plugin(index, np.bincount(index, weights=count), count,
                              grouped[name + "Sum"][both], grouped[name + "Square"][both]))
    if full[0] <= 0.0 or full[1] <= 0.0:
        return None

    granulationFull = full[1] / full[0]
    granulationHalf = 0.5 * sum(h[1] / h[0] for h in halves if h[0] > 0.0)
    # The extrapolation can only raise M_opt (the bias is downward); clamping keeps a scene
    # whose halves happened to land the wrong way from reporting a granulation below the
    # measured one.
    granulation = max(granulationFull, 2.0 * granulationFull - granulationHalf)
    total = full[2] / full[0]
    return {"granulation": granulation, "ranking": total / granulation, "total": total,
            "granulationPlugin": granulationFull, "granulationHalf": granulationHalf,
            "pixels": full[3],
            "voxelsPerPixel": float(int(both.sum())) / float(len(np.unique(index)))}


def decompose(grouped, minPerVoxel):
    """Image-level S2 (granulation) and S3 (ranking), summed over shading points."""
    # A voxel seen a handful of times gives a noisy m2, and sqrt is concave, so such entries
    # push M_opt down and blame the ranking for what is really sampling noise.
    keep = grouped["count"] >= minPerVoxel
    pixel, count = grouped["pixel"][keep], grouped["count"][keep]
    total, square = grouped["sum"][keep], grouped["square"][keep]
    if len(count) == 0:
        return None

    index = np.unique(pixel, return_inverse=True)[1]
    perPixelCount = np.bincount(index, weights=count)
    # p_i m1_i = sum_i / N, p_i m2_i = square_i / N, p_i sqrt(m2_i) = count_i sqrt(m2_i) / N
    contSum = np.bincount(index, weights=total) / perPixelCount
    actSum = np.bincount(index, weights=square) / perPixelCount
    optSum = np.bincount(index, weights=count * np.sqrt(square / count)) / perPixelCount

    usable = (contSum > 0.0) & (optSum > 0.0)
    if not usable.any():
        return None
    totalCont = float((contSum[usable] ** 2).sum())
    totalOpt = float((optSum[usable] ** 2).sum())
    totalAct = float(actSum[usable].sum())
    if totalCont <= 0.0 or totalOpt <= 0.0:
        return None
    return {"granulation": totalOpt / totalCont, "ranking": totalAct / totalOpt,
            "total": totalAct / totalCont, "pixels": int(usable.sum()),
            "voxelsPerPixel": float(len(count)) / float(len(perPixelCount))}


def analyse(path, thresholds):
    grouped, stats = load_records(path)
    stats["byThreshold"] = {}
    for threshold in thresholds:
        result = decompose(grouped, threshold)
        if result:
            stats["byThreshold"][str(threshold)] = result
    headline = richardson(grouped)
    if headline:
        stats.update(headline)
    return stats


def run(args, manifest, parameters):
    root = Path(args.out)
    under(root).mkdir(parents=True, exist_ok=True)
    for cell, scene, light in campaign.campaign_cells(manifest, args.only):
        out = root / cell
        target = under(root) / f"{cell}.json"
        if target.exists():
            print(f"exists  {cell}")
            continue
        if RECORD_FILE.exists():
            RECORD_FILE.unlink()
        cvars = ["vxpg.guiding.records=1"] + campaign.scene_cvars(cell, parameters)
        print(f"  {cell} [{' '.join(cvars)}] ...", flush=True)
        code, _took = engine_checked(
            scene, light, manifest, VXPG, cvars, out,
            f"seconds:{args.seconds}", 1, args.warmup,
            log_path=under(root) / f"{cell}.log",
            config=campaign.config_path(manifest, scene, light))
        if code != 0 or not RECORD_FILE.exists():
            print(f"  FAILED {cell} exit={code}")
            continue
        stats = analyse(RECORD_FILE, args.thresholds)
        # Kept, not deleted: a 30 s run is minutes of GPU time and the analysis of it is
        # seconds, so re-deriving the split from a different estimator must not cost a re-run.
        RECORD_FILE.replace(under(root) / f"{cell}.bin")
        target.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        if "granulation" in stats:
            print(f"  ok     {cell}: granulacja {stats['granulation']:.2f}x, "
                  f"ranking {stats['ranking']:.2f}x ({stats['frames']} klatek)")
        else:
            print(f"  ok     {cell}: brak wystarczajacych danych")


def report(args, manifest):
    root = Path(args.out)
    rows = []
    for cell, _scene, _light in campaign.campaign_cells(manifest, args.only):
        path = under(root) / f"{cell}.json"
        if not path.exists():
            continue
        stats = json.loads(path.read_text(encoding="utf-8"))
        if "granulation" not in stats:
            continue
        stats["cell"] = cell
        stats["ratio"] = M1_RATIO.get(cell)
        rows.append(stats)

    rows.sort(key=lambda r: -r["ranking"])
    lines = [
        "# Krok B: granulacja czy ranking", "",
        "Rozbicie całkowitej nieefektywności przewodnika `M_act/M_cont` na dwa czynniki, "
        "które się **mnożą**. `granulacja` to koszt tego, że wewnątrz woksela gęstość jest "
        "stała — czego nie usunie żaden wybór `p_i`. `ranking` to koszt tego, jak drzewo "
        "faktycznie wybiera woksle, ponad tamto minimum. Oba są ≥ 1 z nierówności "
        "Cauchy'ego-Schwarza; złamanie tego oznaczałoby błąd pomiaru.", "",
        "`√m2` z `n` próbek jest obciążone w dół (pierwiastek jest wklęsły), co zaniża `M_opt` "
        "i przypisuje różnicę rankingowi. Obciążenie jest rzędu `1/n`, więc znika w "
        "**ekstrapolacji Richardsona**: ten sam estymator liczony na pełnym przebiegu i na "
        "jego połowie, złożony jako `2·S(n) − S(n/2)`, na tym samym zbiorze wokseli.", "",
        "| scena | M1 | granulacja | ranking | razem | wtyczkowa | połówka | wokseli/px | klatek |",
        "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['cell'].replace('--own', '')} | {row['ratio'] or '—'} % | "
            f"{row['granulation']:.2f}× | {row['ranking']:.1f}× | {row['total']:.0f}× | "
            f"{row['granulationPlugin']:.2f}× | {row['granulationHalf']:.2f}× | "
            f"{row['voxelsPerPixel']:.0f} | {row['frames']} |")

    lines += [
        "", "## Odczyt", "",
        "- `ranking` ≫ `granulacja` ⇒ lewarem jest **waga w drzewie**, a nie drobniejsza "
        "siatka; zgodne z tym, że przemiatanie rozdzielczości (M9, K5) nic nie dało.",
        "- `granulacja` ≫ `ranking` ⇒ odwrotnie: reprezentacja jest za gruba i ranking nie ma "
        "czego szeregować.",
        "", "Kolumny `wtyczkowa` i `połówka` pokazują skalę poprawki: pierwsza to estymator "
        "bez korekty na pełnej próbie, druga — ten sam estymator na połowie danych. Różnica "
        "między nimi **jest** obciążeniem pierwiastka, a `granulacja` to ekstrapolacja "
        "`2·pełna − połówka`.", "",
        "## Wrażliwość na próg", "",
        "Ten sam rozkład przy różnych progach odrzucania. Jeżeli `ranking` maleje "
        "monotonicznie wraz z progiem, część przypisanej mu wartości jest szumem "
        "próbkowania, a nie rzeczywistym niedopasowaniem wag.", ""]
    thresholds = sorted({int(t) for row in rows for t in row.get("byThreshold", {})})
    lines.append("| scena | " + " | ".join(f"≥{t}" for t in thresholds) + " |")
    lines.append("|---" * (len(thresholds) + 1) + "|")
    for row in rows:
        cells = []
        for threshold in thresholds:
            entry = row.get("byThreshold", {}).get(str(threshold))
            cells.append(f"{entry['granulation']:.2f} × {entry['ranking']:.2f}" if entry else "—")
        lines.append(f"| {row['cell'].replace('--own', '')} | " + " | ".join(cells) + " |")

    out_md = under(root) / "rozklad.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (under(root) / "rozklad.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_md}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="tools/ewaluacja-campaign.json")
    parser.add_argument("--only", metavar="M1", type=lambda v: v.split(","), default=["M1"])
    parser.add_argument("--out", default="SavedUserData/Screenshots/rozklad")
    parser.add_argument("--parameters",
                        default="Raytracer/SavedUserData/Screenshots/parametry-pass2/parameters.json")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--thresholds", type=lambda v: [int(x) for x in v.split(",")],
                        default=[1, 4, 8, 16, 32],
                        help="min draws per voxel at a shading point; the last one is the headline")
    parser.add_argument("--file", help="analyse an existing guide-records.bin and exit")
    add_build_argument(parser)
    parser.add_argument("command", choices=["run", "analyse", "report"])

    args = parser.parse_args()
    if args.file:
        print(json.dumps(analyse(args.file, args.thresholds), indent=1))
        return
    recon.EXE = resolve_exe(getattr(args, "build", None))
    manifest = campaign.load_manifest(REPO_ROOT / args.manifest)
    if args.command == "run":
        run(args, manifest, campaign.load_parameters(args.parameters))
    elif args.command == "analyse":
        root = Path(args.out)
        for cell, _scene, _light in campaign.campaign_cells(manifest, args.only):
            stored = under(root) / f"{cell}.bin"
            if not stored.exists():
                continue
            stats = analyse(stored, args.thresholds)
            (under(root) / f"{cell}.json").write_text(json.dumps(stats, indent=1),
                                                      encoding="utf-8")
            print(f"  {cell}: granulacja {stats.get('granulation', 0):.2f}x, "
                  f"ranking {stats.get('ranking', 0):.2f}x")
    else:
        report(args, manifest)


if __name__ == "__main__":
    main()
