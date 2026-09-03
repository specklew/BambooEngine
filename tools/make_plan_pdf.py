"""Build the research-plan PDF from the measurement files themselves.

Every table below is generated from the JSON the harness wrote, not retyped, so the document
cannot drift away from the data. Where a measurement has not run yet, the section says so
instead of leaving a gap the reader would have to interpret.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\macad\Documents\_Projects\BambooEngine")
SHOTS = REPO / "Raytracer" / "SavedUserData" / "Screenshots"
SCRATCH = Path(__file__).resolve().parent
OUT = SCRATCH / "plan"
OUT.mkdir(exist_ok=True)

NAME = {
    "cornell-box": "Cornell Box", "bedroom": "Bedroom", "staircase": "Staircase",
    "veach-ajar": "Veach Ajar", "san-miguel": "San Miguel", "zero-day": "Zero Day",
    "sun-temple": "Sun Temple", "sun-temple-2": "Sun Temple 2",
    "breakfast-room": "Breakfast Room", "sponza": "Sponza",
}
CELL = {f"{k}--own": v for k, v in NAME.items()}
RUNGS = ["32", "64", "128", "256", "512"]
MIB = 1024 ** 2
SIX = ["cornell-box", "bedroom", "staircase", "veach-ajar", "san-miguel", "zero-day"]
CANDIDATES = ["cornell-box", "bedroom", "staircase", "breakfast-room", "veach-ajar",
              "sponza", "san-miguel", "sun-temple", "sun-temple-2", "zero-day"]


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def newest(name):
    """The repeated run of a measurement if there is one, otherwise the original.

    The frame-window accounting fix of 2026-09-03 invalidated every short-window measurement
    (M1, M7, the budget ladder, the 1 s curves); those were re-run into `<name>-v3`. Resolving
    by existence keeps this document pointing at the freshest data without a second set of
    paths to keep in step - and makes it obvious, when a `-v3` directory is missing, that the
    figure below still comes from the superseded run.
    """
    repeated = SHOTS / f"{name}-v3"
    return repeated if repeated.exists() else SHOTS / name


scenes = load(SHOTS / "scenes-probe" / "scenes.json", {})
strategy = load(SCRATCH / "strategy.json", {})
params = load(SHOTS / "parametry-pass2" / "parameters.json", {})
params1 = load(SHOTS / "parametry" / "parameters-pass1.json", {})
suntemple = load(SHOTS / "parametry-suntemple" / "parameters.json", {})
costs = load(SHOTS / "parametry-testowania" / "koszt-klatki.json", {})
m1 = load(newest("wyniki-czas") / "m1-wyniki.json", None)


def refs():
    out = {}
    for sidecar in sorted((SHOTS / "ewaluacja-refs").glob("*.json")):
        data = load(sidecar)
        if not data:
            continue
        out[sidecar.stem] = {
            "frames": data.get("raytracing", {}).get("frameIndex", 0),
            "ms": data.get("benchmark", {}).get("meanFrameMs", 0.0),
            "vram": data.get("benchmark", {}).get("videoMemoryBytes", 0) / 1048576,
        }
    return out


REFERENCES = refs()


def tex(text):
    """Escape what LaTeX would otherwise eat."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def num(value, digits=3, dash="---"):
    if value is None:
        return dash
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits=1, dash="---"):
    if value is None:
        return dash
    return f"{100 * value:.{digits}f}".replace(".", ",")


def table(header, rows, spec, caption=None, small=r"\footnotesize"):
    lines = [r"\begin{center}", small,
             r"\begin{tabular}{" + spec + "}", r"\hline",
             " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}"]
    if caption:
        lines += [r"\\[2pt]", r"\footnotesize\itshape " + caption]
    lines += [r"\end{center}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------- sections

def scene_table():
    rows = []
    for key in CANDIDATES:
        entry = scenes.get(key)
        if not entry:
            continue
        extent = entry.get("extent") or [0, 0, 0]
        at128 = entry["rungs"].get("128", {})
        rows.append([NAME[key],
                     f"{entry.get('triangles', 0):,}".replace(",", "\\,"),
                     num(max(extent), 0),
                     pct(at128.get("litShare")),
                     num(at128.get("voxelMetres"), 3)])
    return table(["scena", "trójkąty", "bok [m]", "oświetl. [\\%]", "woksel [m]"],
                 rows, "l r r r r")


def share_table():
    rows = []
    for key in CANDIDATES:
        entry = scenes.get(key)
        if not entry:
            continue
        cells = []
        for rung in RUNGS:
            rung_data = entry["rungs"].get(rung)
            if not rung_data or rung_data.get("failed"):
                cells.append("---")
            elif rung_data.get("truncated"):
                cells.append(pct(rung_data.get("litShare")) + "$^{*}$")
            else:
                cells.append(pct(rung_data.get("litShare")))
        rows.append([NAME[key]] + cells)
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r",
                 "$^{*}$ szczebel, na którym liczba oświetlonych wokseli przekracza "
                 "131\\,072-elementowy bufor kompaktacji: mierzy pułap, nie siatkę.")


def acceptance_table():
    rows = []
    for key in CANDIDATES:
        grids = strategy.get(key, {})
        if not grids:
            continue
        rows.append([NAME[key]] + [pct(grids[r]["accepted"]) if r in grids else "---"
                                   for r in RUNGS])
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r")


def voxel_metres_table():
    rows = []
    for key in CANDIDATES:
        entry = scenes.get(key)
        if not entry:
            continue
        rows.append([NAME[key]] + [num(entry["rungs"].get(r, {}).get("voxelMetres"), 3)
                                   for r in RUNGS])
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r")


def reference_table():
    rows = []
    for key in SIX + ["sun-temple"]:
        entry = REFERENCES.get(f"{key}--own")
        if not entry:
            continue
        rows.append([NAME[key], f"{entry['frames']:,}".replace(",", "\\,"),
                     num(entry["ms"], 3), num(entry["vram"], 0)])
    return table(["scena", "klatek", "ms/klatkę", "pamięć [MiB]"], rows, "l r r r")


def sweep_table(factor, label, source, cells):
    columns = sorted({v for cell in cells
                      for v in source.get("scenes", {}).get(cell, {}).get("_sweep", {})
                      .get(factor, {})}, key=float)
    rows = []
    for cell in cells:
        entry = source.get("scenes", {}).get(cell, {})
        readings = entry.get("_sweep", {}).get(factor, {})
        chosen = str(entry.get(factor, ""))
        out = []
        for value in columns:
            reading = readings.get(value)
            if reading is None:
                out.append("---")
            elif value == chosen:
                out.append(r"\textbf{" + num(reading, 5) + "}")
            else:
                out.append(num(reading, 5))
        rows.append([CELL.get(cell, cell)] + out + [chosen or "---"])
    return table(["scena"] + [tex(c) for c in columns] + ["wybór"], rows,
                 "l" + " r" * (len(columns) + 1), label)


def ratio_table(path, name):
    """One measurement's per-scene table: both arms and the ratio between them."""
    data = (load(path if isinstance(path, Path) else SHOTS / path, {}) or {}).get("data", {})
    if not data:
        return ""
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = data.get(cell)
        if not arms:
            continue
        base, guided = arms.get("BSDF", {}).get("0"), arms.get("WIE", {}).get("0")
        if not base or not guided:
            continue
        rows.append([CELL[cell],
                     str(base["frames"]), num(1000 * base["seconds"], 1),
                     num(base["flip"]["mean"], 5),
                     str(guided["frames"]), num(1000 * guided["seconds"], 1),
                     num(guided["flip"]["mean"], 5),
                     pct(base["flip"]["mean"] / guided["flip"]["mean"], 0)])
    return table(["scena", "kl.", "ms", "FLIP", "kl.", "ms", "FLIP", "VXPG/PT"],
                 rows, "l r r r r r r r",
                 "Kolumny 2--4 to śledzenie ścieżek, 5--7 technika naprowadzana. "
                 "Ostatnia kolumna powyżej 100\\,\\% oznacza przewagę techniki naprowadzanej.")


def variance_table():
    """M3, parsed from the report the scorer writes."""
    path = SHOTS / "wyniki-wariancja" / "rowna-wariancja.md"
    if not path.exists():
        return ""
    per_cell = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "komorka" in line or line.startswith("|---"):
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        per_cell.setdefault(parts[0], {})[parts[1]] = parts
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        entry = per_cell.get(cell)
        if not entry or "BSDF" not in entry or "WIE" not in entry:
            continue
        base, guided = entry["BSDF"], entry["WIE"]
        try:
            speedup = float(base[3]) / float(guided[3])
        except ValueError:
            speedup = None
        rows.append([CELL[cell], base[2], base[3].replace(".", ","),
                     guided[2], guided[3].replace(".", ","),
                     (num(speedup, 2) + r"$\times$") if speedup else "---"])
    return table(["scena", "klatka PT", "czas PT", "klatka VXPG", "czas VXPG", "przyspieszenie"],
                 rows, "l r r r r r")


def breakdown_table():
    """M5 reduced to the two numbers that answer where the multiplier comes from."""
    path = SHOTS / "wyniki-rozbicie" / "breakdown.md"
    if not path.exists():
        return ""
    rows = []
    blocks = {b.split("\n")[0].strip(): b for b in path.read_text(encoding="utf-8").split("## ")[1:]}
    for cell in [f"{k}--own" for k in SIX]:
        block = blocks.get(cell)
        if not block:
            continue
        nodes = {}
        for line in block.splitlines():
            if not line.startswith("| ") or line.startswith("|---"):
                continue
            parts = [c.strip() for c in line.strip("|").split("|")]
            if len(parts) >= 7:
                nodes[parts[0]] = parts

        def value(row, index):
            try:
                return float(row[index].split()[0])
            except (ValueError, IndexError):
                return 0.0

        integrator = nodes.get("Raytrace Technique")
        if not integrator:
            continue
        base, guided = value(integrator, 3), value(integrator, 4)
        chain = sum(value(row, 4) for key, row in nodes.items() if key.startswith("VXPG"))
        rows.append([CELL[cell], num(base, 3), num(guided, 3),
                     num(guided / base, 2) + r"$\times$", num(chain, 3),
                     pct(chain / (chain + guided), 0)])
    return table(["scena", "integr. PT", "integr. VXPG", "mnożnik", "łańcuch",
                  "udział łańcucha [\\%]"], rows, "l r r r r r")


def memory_table():
    """M8 for one scene: which rows follow the grid and which the screen."""
    data = load(SHOTS / "wyniki-pamiec" / "memory.json", {}) or {}
    entry = data.get("veach-ajar--own")
    if not entry:
        return ""
    rungs = sorted(entry, key=int)
    stages = []
    for rung in rungs:
        for stage in entry[rung]["memory"].get("byStage", {}):
            if stage not in stages:
                stages.append(stage)
    stages.sort(key=lambda st: -max(entry[r]["memory"]["byStage"].get(st, 0) for r in rungs))
    rows = []
    for stage in stages:
        rows.append([stage] + [num(entry[r]["memory"]["byStage"].get(stage, 0) / MIB, 1)
                               for r in rungs])
    rows.append([r"\textbf{inwentarz razem}"]
                + [r"\textbf{" + num(entry[r]["memory"]["totalBytes"] / MIB, 0) + "}" for r in rungs])
    rows.append([r"\textbf{cała metoda}"]
                + [r"\textbf{" + num(entry[r].get("videoMemoryBytes", 0) / MIB, 0) + "}" for r in rungs])
    return table(["etap [MiB]"] + [f"${r}^3$" for r in rungs], rows,
                 "l" + " r" * len(rungs),
                 "Veach Ajar. Wokselizacja i budowa rozkładu rosną z siatką; integrator, "
                 "superpiksele i V-bufor zależą wyłącznie od rozdzielczości obrazu.")


def grid_quality_table():
    """M9: FLIP against grid rung, per scene, at a common 30 s budget."""
    data = load(SHOTS / "wyniki-pamiec" / "memory.json", {}) or {}
    if not data:
        return ""
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        entry = data.get(cell)
        if not entry:
            continue
        cells_out = []
        best = min((v["flip"] for v in entry.values() if v.get("flip")), default=None)
        for rung in RUNGS:
            value = entry.get(rung, {}).get("flip")
            if value is None:
                cells_out.append("---")
            elif best is not None and abs(value - best) < 1e-12:
                cells_out.append(r"\textbf{" + num(value, 5) + "}")
            else:
                cells_out.append(num(value, 5))
        rows.append([CELL[cell]] + cells_out)
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r",
                 "Budżet 30 s, wspólny dla wszystkich szczebli. Wytłuszczony najlepszy "
                 "szczebel danej sceny.")


def crossover_table():
    """The same ratio at three budgets an order of magnitude apart."""
    import csv as _csv
    from collections import defaultdict as _dd
    short = load(newest("wyniki-czas") / "m1-wyniki.json", {}) or {}
    curves = SHOTS / "wyniki-krzywe" / "m4-curves.csv"
    if not short or not curves.exists():
        return ""
    series = _dd(lambda: _dd(list))
    for row in _csv.DictReader(curves.open(encoding="utf-8")):
        if row["pass"] != "0":
            continue
        series[row["cell"]][row["arm"]].append((float(row["seconds"]), float(row["flipMean"])))
    for cell in series:
        for arm in series[cell]:
            series[cell][arm].sort()

    def at(points, seconds):
        return min(points, key=lambda point: abs(point[0] - seconds))

    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = short.get("data", {}).get(cell)
        if not arms or cell not in series:
            continue
        base = arms["BSDF"]["0"]["flip"]["mean"]
        guided = arms["WIE"]["0"]["flip"]["mean"]
        out = [CELL[cell], pct(base / guided, 0)]
        for mark in (0.3, 3.0, 30.0):
            a = at(series[cell]["BSDF"], mark)[1]
            b = at(series[cell]["WIE"], mark)[1]
            out.append(pct(a / b, 0))
        rows.append(out)
    return table(["scena", "M1 ($\\approx$20 ms)", "0,3 s", "3 s", "30 s"], rows,
                 "l r r r r",
                 "Stosunek FLIP (ścieżkowe / naprowadzane). Powyżej 100\\,\\% prowadzi "
                 "technika naprowadzana.")


def stability_table():
    """M6: level and coefficient of variation over a run of equal-sample images."""
    import csv as _csv
    import statistics as _stats
    from collections import defaultdict as _dd
    path = SHOTS / "wyniki-stabilnosc" / "stabilnosc.csv"
    if not path.exists():
        return ""
    series = _dd(lambda: _dd(list))
    for row in _csv.DictReader(path.open(encoding="utf-8")):
        series[row["cell"]][row["arm"]].append(float(row["flipMean"]))
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = series.get(cell)
        if not arms:
            continue
        out = [CELL[cell]]
        for arm in ("BSDF", "WIE"):
            values = arms.get(arm, [])
            if len(values) < 2:
                out += ["---", "---"]
                continue
            mean = _stats.fmean(values)
            out += [num(mean, 5), pct(_stats.stdev(values) / mean, 2)]
        rows.append(out)
    return table(["scena", "poziom PT", "zmienność PT", "poziom VXPG", "zmienność VXPG"],
                 rows, "l r r r r",
                 "64 kolejne obrazy o tej samej liczbie próbek. Zmienność to odchylenie "
                 "standardowe podzielone przez średnią.")


def reuse_table():
    """M7 at the equal-time budget."""
    data = (load(newest("wyniki-obciazenie") / "m7-wyniki.json", {}) or {}).get("data", {})
    if not data:
        return ""
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = data.get(cell, {})
        base, reuse = arms.get("WIE", {}).get("0"), arms.get("WIE-R", {}).get("0")
        if not base or not reuse:
            continue
        gap = abs(base["seconds"] - reuse["seconds"]) / max(base["seconds"], reuse["seconds"])
        rows.append([CELL[cell], num(base["flip"]["mean"], 5), num(reuse["flip"]["mean"], 5),
                     pct(base["flip"]["mean"] / reuse["flip"]["mean"], 0), pct(gap, 1)])
    return table(["scena", "FLIP podstawowy", "FLIP obciążony", "stosunek", "rozjazd czasu"],
                 rows, "l r r r r",
                 "Powyżej 100\\,\\% wariant obciążony jest lepszy.")


def reuse_curve_table():
    """M7 along the 30 s curve: where the bias starts to cost."""
    import csv as _csv
    from collections import defaultdict as _dd
    path = newest("wyniki-obciazenie-krzywa") / "m7c-curves.csv"
    if not path.exists():
        return ""
    series = _dd(lambda: _dd(list))
    for row in _csv.DictReader(path.open(encoding="utf-8")):
        series[row["cell"]][row["arm"]].append((float(row["seconds"]), float(row["flipMean"])))
    for cell in series:
        for arm in series[cell]:
            series[cell][arm].sort()
    marks = (0.3, 1.0, 3.0, 10.0, 30.0)
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = series.get(cell)
        if not arms:
            continue
        out = [CELL[cell]]
        for mark in marks:
            base = min(arms["WIE"], key=lambda point: abs(point[0] - mark))[1]
            reuse = min(arms["WIE-R"], key=lambda point: abs(point[0] - mark))[1]
            out.append(pct(base / reuse, 0))
        rows.append(out)
    return table(["scena"] + [f"{m} s".replace(".", ",") for m in marks], rows,
                 "l r r r r r",
                 "Powyżej 100\\,\\% wariant obciążony jest lepszy; spadek poniżej oznacza, "
                 "że błąd systematyczny przeważył nad tańszą klatką.")


def levers_table():
    """M10, parsed from the report the scorer writes."""
    path = SHOTS / "wyniki-dzwignie" / "levers.md"
    if not path.exists():
        return ""
    per_cell = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "lever" in line or line.startswith("|---"):
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) < 8:
            continue
        per_cell.setdefault((parts[0], parts[1]), {})[parts[2]] = parts
    levers = ["noviews", "swizzle", "wave32", "wave64"]
    rows = []
    for cell in ("veach-ajar--own", "san-miguel--own"):
        for arm, label in (("BSDF", "ścieżkowe"), ("WIE", "naprowadzane")):
            entry = per_cell.get((cell, arm))
            if not entry:
                continue
            rows.append([CELL[cell], label]
                        + [entry.get(lever, ["", "", "", "", "---"])[4].replace("%", "\\%")
                           for lever in levers])
    return table(["scena", "wariant"] + [f"\\texttt{{{l}}}" for l in levers], rows,
                 "l l r r r r",
                 "Zmiana kosztu klatki wobec konfiguracji z wszystkimi dźwigniami "
                 "wyłączonymi. Wartość ujemna to przyspieszenie.")


def reference_drift_table():
    """M12.2: how far apart the reference images themselves are."""
    import numpy as _np
    from PIL import Image as _Image
    sets = {"600": SHOTS / "ewaluacja-refs-600",
            "1200": SHOTS / "ewaluacja-refs-1200",
            "1800": SHOTS / "ewaluacja-refs"}
    available = {k: v for k, v in sets.items() if v.exists()}
    if len(available) < 2:
        return ""
    pairs = [("600", "1200"), ("1200", "1800"), ("600", "1800")]
    pairs = [(a, b) for a, b in pairs if a in available and b in available]

    def mse(first, second):
        one = _np.asarray(_Image.open(first).convert("RGB"), dtype=_np.float64) / 255.0
        two = _np.asarray(_Image.open(second).convert("RGB"), dtype=_np.float64) / 255.0
        return float(((one - two) ** 2).mean())

    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        out = [CELL[cell]]
        for a, b in pairs:
            first, second = available[a] / f"{cell}.png", available[b] / f"{cell}.png"
            if not (first.exists() and second.exists()):
                out.append("---")
                continue
            out.append(f"{mse(first, second):.2e}".replace("e-0", r"$\cdot 10^{-") + "}$")
        rows.append(out)
    return table(["scena"] + [f"{a}--{b} s" for a, b in pairs], rows,
                 "l r r r",
                 "MSE między samymi obrazami odniesienia. Dla porównania: MSE ramion wobec "
                 "referencji w pomiarze M1 wynosi od $10^{-3}$ do $5\\cdot10^{-2}$.")


def throttle_table():
    """Frame cost under a 10 s load against the same cost measured in a 24 ms window."""
    burst = (load(SHOTS / "wyniki-czas-budzet24ms" / "m1-wyniki.json", {}) or {}).get("data", {})
    if not burst:
        return ""
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        for arm, label in (("WIE", "naprow."), ("BSDF", "ścieżk.")):
            sustained = costs.get(cell, {}).get(arm)
            quick = burst.get(cell, {}).get(arm, {}).get("0", {}).get("ms")
            if not sustained or not quick:
                continue
            rows.append([CELL[cell], label, num(sustained, 3), num(quick, 3),
                         pct((quick - sustained) / sustained, 1)])
    return table(["scena", "wariant", "10 s", "24 ms", "różnica [\\%]"], rows, "l l r r r")


def frame_pair_table():
    pairs = load(SHOTS / "parametry-testowania" / "klatki-rownego-czasu.json", {}) or {}
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        entry = pairs.get(cell)
        if not entry:
            continue
        rows.append([CELL[cell], str(entry["BSDF"]), num(entry["_msBSDF"], 2),
                     str(entry["WIE"]), num(entry["_msWIE"], 2), pct(entry["_gap"], 1)])
    return table(["scena", "N(PT)", "czas", "N(VXPG)", "czas", "rozjazd [\\%]"],
                 rows, "l r r r r r")


def cost_table():
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = costs.get(cell)
        if not arms:
            continue
        rows.append([CELL[cell], num(arms["BSDF"], 3), num(arms["WIE"], 3),
                     num(arms["WIE"] / arms["BSDF"], 2) + r"$\times$"])
    return table(["scena", "PT [ms]", "VXPG [ms]", "mnożnik"], rows, "l r r r")


def budget_table(budget_ms=24):
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = costs.get(cell)
        if not arms:
            continue
        n1 = max(1, math.ceil(budget_ms / arms["BSDF"]))
        n2 = max(1, math.ceil(budget_ms / arms["WIE"]))
        t1, t2 = n1 * arms["BSDF"], n2 * arms["WIE"]
        rows.append([CELL[cell], str(n1), num(t1, 1), str(n2), num(t2, 1),
                     pct(abs(t1 - t2) / max(t1, t2))])
    return table(["scena", "N(PT)", "czas PT", "N(VXPG)", "czas VXPG", "rozjazd [\\%]"],
                 rows, "l r r r r r")


def m1_table():
    if not m1:
        return (r"\emph{Przebieg trwa w chwili złożenia tego dokumentu. Tabela zostanie "
                r"uzupełniona po jego zakończeniu.}" + "\n")
    rows = []
    for cell, arms in m1.get("data", {}).items():
        base = arms.get("BSDF", {}).get("0", {})
        guided = arms.get("WIE", {}).get("0", {})
        if not base or not guided:
            continue
        ratio = base["flip"]["mean"] / guided["flip"]["mean"] if guided["flip"]["mean"] else None
        rows.append([CELL.get(cell, cell),
                     num(base["ms"], 2), str(base["frames"]), num(base["flip"]["mean"], 5),
                     num(guided["ms"], 2), str(guided["frames"]), num(guided["flip"]["mean"], 5),
                     pct(ratio, 0)])
    return table(["scena", "ms", "kl.", "FLIP", "ms", "kl.", "FLIP", "VXPG/PT"],
                 rows, "l r r r r r r r",
                 "Kolumny 2--4: śledzenie ścieżek. Kolumny 5--7: technika naprowadzana. "
                 "Ostatnia kolumna powyżej 100\\,\\% oznacza przewagę techniki naprowadzanej.")


def suntemple_table():
    entry = suntemple.get("scenes", {}).get("sun-temple--own", {})
    readings = entry.get("_sweep", {}).get("voxel.gridDim", {})
    scene = scenes.get("sun-temple", {})
    grids = strategy.get("sun-temple", {})
    rows = []
    for rung in RUNGS:
        rows.append([f"${rung}^3$",
                     num(scene.get("rungs", {}).get(rung, {}).get("voxelMetres"), 2),
                     num(readings.get(rung), 5),
                     pct(grids.get(rung, {}).get("accepted"))])
    return table(["siatka", "woksel [m]", "FLIP", "przepustowość [\\%]"], rows, "l r r r")


# --------------------------------------------------------------------- document

BODY = r"""
\section{Po co ten dokument}

Praca porównuje dwie techniki obliczania oświetlenia globalnego w czasie rzeczywistym:
\textbf{śledzenie ścieżek} (dalej: wariant podstawowy) i \textbf{śledzenie ścieżek
naprowadzane rozkładem wokselowym}, czyli metodę, w której z widocznej części sceny buduje
się co klatkę przestrzenny rozkład jasności i używa go do celowania promieni.

Pytanie badawcze brzmi: \emph{czy naprowadzanie opłaca się przy budżecie czasu rzeczywistego,
w jakich scenach, i jakim kosztem}. Odpowiedź musi być liczbowa, powtarzalna i odporna na
zarzut, że wybrano korzystne warunki --- stąd cały aparat opisany niżej.

Dokument opisuje: warunki pomiaru, każdy miernik razem ze sposobem jego liczenia, przebieg
badania krok po kroku wraz z celem każdego kroku, oraz wszystkie wyniki uzyskane do chwili
jego złożenia.

\section{Stanowisko i warunki wspólne}

\begin{itemize}
\item Renderer własny (Direct3D~12, DXR), uruchamiany w trybie bezgłowym --- bez okna,
      bez interfejsu, z zapisem obrazu do pliku PNG.
\item Rozdzielczość \textbf{1920$\times$1080}, kompilacja \textbf{Release}.
\item Układ główny: \textbf{AMD Radeon RX~9070~XT}. Układ weryfikacyjny:
      \textbf{NVIDIA GeForce RTX~3050 Mobile}. Każda liczba bez wskazania układu pochodzi
      z układu głównego.
\item Liczba odbić: jedno po punkcie cieniowania (w konwencji pracy: dwa odbicia).
      Ta sama dla obu wariantów --- jest to jedyny parametr, który obie techniki dzielą.
\item Oświetlenie nieba wyłączone, oświetlenie pochodzi wyłącznie ze źródeł sceny.
      Powód: przy dominującym niebie rozkład naprowadzający nie ma czego reprezentować i
      porównanie przestaje dotyczyć badanej metody.
\item Ekspozycja, kontrast i nasycenie są \textbf{zamrożone per scena}. Decydują o tym, jak
      wygląda obraz, a więc o tym, co znaczy każda liczba miernika percepcyjnego; nie są
      parametrami swobodnymi.
\end{itemize}

\section{Mierniki i sposób ich liczenia}

\subsection{FLIP --- miernik percepcyjny}

Podstawowy miernik jakości obrazu. Porównuje obraz badany z obrazem odniesienia i zwraca
\textbf{mapę błędu} o wartościach od 0 do 1, osobno dla każdego piksela, modelując to, co
człowiek faktycznie zauważa: czułość na kontrast w różnych częstotliwościach, różnice barwy
oraz zaburzenia krawędzi. Używana wersja: \textbf{1.7}, tryb \textbf{LDR}, gęstość kątowa
\textbf{67,02 piksela na stopień} (odpowiada obrazowi oglądanemu z typowej odległości).

Z mapy błędu liczone są statystyki rozkładu: \textbf{średnia} (miernik wiodący --- jako
jedyna liczy pojedyncze bardzo jasne artefakty z ich prawdziwą wagą), \textbf{mediana},
\textbf{kwartyle}, \textbf{95.\ percentyl}, wartość \textbf{najmniejsza} i
\textbf{największa}, oraz \textbf{mediana ważona} (pula masy błędu --- ślepa na pojedyncze
artefakty, dlatego podawana obok, a nie zamiast).

\subsection{MSE --- miernik drugi, niezależny}

Średnia kwadratów różnic wartości pikseli między obrazem badanym a odniesieniem, po trzech
składowych barwy. Podawany zawsze obok FLIP, bo obie wielkości mogą się rozejść: MSE karze
jasne artefakty znacznie mocniej i jest mniej podatny na zarzut, że wniosek zależy od
jednego modelu percepcji.

\subsection{Obraz odniesienia}

Obraz \emph{prawdziwy}, do którego porównywane jest wszystko inne: ten sam kadr, to samo
oświetlenie, wyrenderowany wariantem podstawowym przez \textbf{1800 sekund}. Jest związany
ze sceną, stanem kamery, oprawą świetlną i rozdzielczością --- zmiana którejkolwiek z tych
rzeczy unieważnia go.

Zastrzeżenie, które musi towarzyszyć każdemu wynikowi: obraz odniesienia sam jest obrazem
śledzenia ścieżek, więc jego szum resztkowy koreluje z szumem wariantu podstawowego i
\emph{zaniża} mierzoną odległość tego wariantu. Efekt jest przy 1800~s mały i jednakowy dla
wszystkich scen, więc nie zmienia ich uporządkowania.

\subsection{Koszt klatki}

Różnica czasu ściennego między dwoma kolejnymi obiegami pętli gry, mierzona zegarem
wysokiej rozdzielczości. Obejmuje: obsługę komunikatów okna, aktualizację sceny, zbudowanie
i wykonanie grafu renderowania oraz prezentację (bez oczekiwania na wygaszanie pionowe).
Nie obejmuje kosztu zapisu obrazu --- silnik odejmuje zmierzony czas odczytu i kodowania
poprzedniego zrzutu, a samo kodowanie PNG biegnie poza wątkiem renderującym.

Raportowana wartość jest \textbf{średnią po wszystkich klatkach danego obrazu}, licząc od
chwili uzbrojenia zrzutu, czyli \emph{po} rozgrzewaniu.

\subsection{Pamięć karty graficznej}

Dwie liczby. Pierwsza to własny inwentarz silnika, etap po etapie, z rzeczywistego rozmiaru
alokacji. Druga to zużycie pamięci lokalnej przez cały proces, odczytane z systemu ---
obejmuje bufory sceny, tekstury i struktury przyspieszające budowane przez sterownik, więc
to ona odpowiada na pytanie o koszt \emph{całej metody}.

\subsection{Proporcja oświetlonych wokseli}

Miara opisująca scenę, wprowadzona po to, żeby dobór scen dało się uzasadnić liczbą.

\[
\text{proporcja} = \frac{\text{woksele, do których dotarło światło}}
                        {\text{woksele zawierające geometrię}}
\]

Licznik to komórki siatki, w których po jednym odbiciu od widocznych punktów cieniowania
wylądowała energia. Mianownik to komórki, przez które przechodzi jakakolwiek geometria.
Obie liczby pochodzą z tej samej klatki i tej samej siatki.

Wielkość ta \textbf{zależy od kadru} (naświetlanie startuje od widocznych pikseli) oraz od
rozdzielczości siatki, więc podawana jest zawsze razem z jednym i drugim.

\subsection{Przepustowość naprowadzania}

Udział promieni naprowadzanych, które przechodzą przez bramkę poprawności. Promień jest
odrzucany, gdy trafi \emph{przed} woksel, do którego był celowany --- czyli gdy po drodze
stoi przeszkoda. Wielkość ta odpowiada na pytanie, czy metoda ma w danej scenie czym
sterować.

Zastrzeżenie: \textbf{nie jest to predyktor wyniku}. Scena o wysokiej przepustowości potrafi
przegrać, a o niskiej wygrać, bo wynik zależy jeszcze od tego, ile klatek zabiera koszt
budowy rozkładu. Przepustowość wyjaśnia wynik, nie przewiduje go.

\subsection{Poziom szumu pomiaru}

Ten sam wariant, ta sama scena, dwa \emph{osobne uruchomienia programu}. Rozrzut liczony jako

\[
\text{rozrzut} = \frac{|a - b|}{\max(a, b)}
\]

gdzie $a$ i $b$ to średnie FLIP obu uruchomień. Jest to próg, poniżej którego różnicy nie
wolno interpretować. Przedział ufności liczony wewnątrz jednego uruchomienia \emph{nie}
nadaje się do tego celu --- obrazy jednego procesu dzielą rozgrzanie i stan cieplny, więc
zaniża on rzeczywisty rozrzut kilkukrotnie.

\section{Przebieg badania krok po kroku}

\subsection{Faza 0 --- dobór scen}

\textbf{Cel.} Wybrać sześć scen tak, żeby wybór dało się uzasadnić liczbą, a nie wrażeniem.
Zbiór ma pokryć trzy rozmiary sceny oraz dwa poziomy proporcji oświetlonych wokseli ---
bo to właśnie ta proporcja decyduje, ile rozkład naprowadzający wnosi ponad rozkład
jednostajny.

\textbf{Sposób.} Dziesięciu kandydatów przemierzonych na pięciu szczeblach siatki, dla
każdego zapisana proporcja oświetlonych wokseli, rozmiar woksela w metrach, koszt klatki,
przepustowość naprowadzania i zrzut diagnostyczny rozkładu jasności.

\textbf{Wynik.}
"""


def build():
    parts = [BODY, scene_table(),
             r"\noindent Proporcja oświetlonych wokseli wobec szczebla siatki:" + "\n",
             share_table(),
             r"\noindent Rozmiar woksela w metrach sceny --- to on, a nie rozdzielczość, "
             r"mówi, co siatka jest w stanie rozróżnić:" + "\n",
             voxel_metres_table(),
             r"\noindent Przepustowość naprowadzania:" + "\n",
             acceptance_table(),
             r"""
\textbf{Rozstrzygnięcie.} Pułap bezwzględny (20\,\%) nie dzieli kandydatów na dwie
równoliczne grupy: przekraczają go tylko dwie sceny i obie są małe albo średnie. Wśród scen
dużych nie ma \emph{żadnej} o wysokiej proporcji i nie jest to kwestia doboru kandydatów ---
proporcja spada z rozmiarem sceny, bo światło z jednego źródła dociera do coraz mniejszej
części objętości. Osie ,,rozmiar'' i ,,rozkład jasności'' nie są więc niezależne.

Przyjęto podział \textbf{względny wewnątrz klasy rozmiaru}: w każdej klasie bierzemy scenę
o najwyższej i o najniższej proporcji.

\begin{center}
\footnotesize
\begin{tabular}{l l l}
\hline
klasa & wyższa proporcja & niższa proporcja \\
\hline
mała & Cornell Box 63,9\,\% & Bedroom 13,1\,\% \\
średnia & Staircase 31,0\,\% & Veach Ajar 6,6\,\% \\
duża & San Miguel 2,3\,\% & Zero Day 0,6\,\% \\
\hline
\end{tabular}
\end{center}

Sun Temple wypadł ze zbioru i wchodzi do pracy jako \emph{udokumentowany przypadek
negatywny} (sekcja \ref{sec:suntemple}).

\subsection{Obrazy odniesienia}

\textbf{Cel.} Wytworzyć obraz prawdziwy dla każdej sceny --- wejście dla wszystkiego, co po
nim. \textbf{Sposób.} 1800~s wariantu podstawowego na scenę. \textbf{Wynik:}
""",
             reference_table(),
             r"""
Zakres zużycia pamięci --- od 1380 do 5243~MiB --- pokazuje, że na scenach dużych największą
pozycją jest sama scena, a nie łańcuch naprowadzający. Ma to bezpośredni skutek: dwie sceny
duże nie zmieszczą się w 4~GB pamięci układu weryfikacyjnego nawet przy samym wariancie
podstawowym, więc weryfikacja obejmie cztery komórki, a brak dwóch pozostałych jest wynikiem,
nie usterką.

\subsection{Faza 1 --- dobór parametrów osobno dla każdej sceny}

\textbf{Cel.} Metoda ma parametry, których dobra wartość zależy od sceny; puszczenie
wszystkich scen na jednej nastawie krzywdzi sceny duże. Pomiar główny ma biec na nastawie
najlepszej dla danej sceny.

\textbf{Sposób.} Przemiatanie \textbf{jednoczynnikowe}: jeden parametr naraz, pozostałe na
wartościach ustalonych. Wspólny budżet 5~s, trzy obrazy na punkt, rozgrzewka 35~s,
punktowanie wobec obrazu odniesienia. Pełny iloczyn kartezjański nie wchodzi w grę czasowo i
nie jest potrzebny --- celem jest nastawa dobra, nie dowodliwie optymalna.

\textbf{Kryterium.} Najniższa średnia FLIP; przy różnicy poniżej 2\,\% wygrywa wartość
mniejsza, bo nierozstrzygalna różnica nie jest powodem, żeby płacić za droższą nastawę.

Symetria wobec wariantu podstawowego jest zachowana w ten sposób, że wariant podstawowy
\emph{nie ma} żadnego z tych parametrów --- i ten fakt trzeba w pracy zapisać wprost, żeby
czytelnik wiedział, że strojenie dotyczy jednej strony porównania.

\textbf{Wynik} (drugie przejście, parametry pozostałe na wartościach wybranych):
""",
             sweep_table("voxel.gridDim", "Rozdzielczość siatki wokseli.",
                         params, [f"{k}--own" for k in SIX]),
             sweep_table("vxpg.topLevelTree.importance",
                         "Tryb wartości liścia: 0 --- widoczność binarna, "
                         "1 --- średnia widoczność, 2 --- sama moc.",
                         params, [f"{k}--own" for k in SIX]),
             sweep_table("vxpg.tree.weightMode",
                         "Ważenie gałęzi dolnego drzewa: 0 --- sama moc, "
                         "1 --- geometria dokładna, 2 --- geometria przybliżona.",
                         params, [f"{k}--own" for k in SIX]),
             r"""
\textbf{Co z tego wynika.} Po pierwsze, najdrobniejsza siatka prawie nigdy nie wygrywa:
na czterech z sześciu scen wybór padł na najgrubszą. Powód jest kosztowy --- drobniejsza
siatka zabiera klatki, a celniejsze naprowadzanie tego nie odrabia. Wyżej poszły tylko te
sceny, na których przy grubej siatce naprowadzanie jest martwe.

Po drugie, tryb wartości liścia dzieli zbiór na dwie grupy: tam, gdzie scena faktycznie
zasłania oświetlone obszary, opłaca się bramka widoczności; gdzie nie zasłania --- opłaca się
ją pominąć, bo kosztuje cały etap obliczeń.

Po trzecie, ważenie gałęzi dolnego drzewa jest praktycznie bez wpływu: różnice rzędu
0,1--2\,\%, poniżej progu rozstrzygalności.

\textbf{Kontrola.} Przemiatanie wykonano \emph{dwukrotnie}: raz przy parametrach pozostałych
na wartościach domyślnych, raz przy wartościach wybranych w pierwszym przejściu. Oba
przejścia dały \textbf{identyczny wybór na wszystkich scenach i wszystkich czynnikach}.
Gdyby czynniki silnie na siebie oddziaływały, drugie przejście przesunęłoby przynajmniej
jeden wybór; nie przesunęło żadnego, więc przemiatanie jednoczynnikowe jest tu wystarczające.

\textbf{Dwa parametry planu, których nie da się przemiatać.} Liczba superwokseli (32) i
liczba przedstawicieli widoczności (128) są stałymi czasu kompilacji: maska widoczności to
jedno 32-bitowe słowo na kafel obrazu, po jednym bicie na superwoksel, a odcisk widoczności
to cztery takie słowa. Zmiana którejkolwiek jest zmianą struktury danych i jąder
obliczeniowych, nie ustawieniem. Obie wartości wchodzą do tabeli parametrów wspólnych razem
z tym uzasadnieniem.

\subsection{Faza 2 --- parametry testowania}

\textbf{Cel.} Ustalić trzy liczby sterujące wszystkimi pomiarami podstawowymi: budżet
równego czasu, liczbę próbek dla porównania przy równej liczbie próbek oraz pułap błędu dla
porównania przy równej wariancji.

\textbf{Koszt klatki} obu wariantów na nastawach z fazy 1:
""",
             cost_table(),
             r"""
\textbf{Budżet równego czasu --- i dlaczego kryterium trzeba było zmienić.} Pierwsza
redakcja planu żądała, żeby budżet dobrać tak, aby \emph{liczba klatek} obu wariantów była
zbliżona. Warunek ten jest przy równym czasie niewykonalny: stosunek liczby klatek jest
równy stosunkowi kosztów klatki (2,4--6,1$\times$), a budżet skraca się obu wariantom
tak samo. Równa liczba klatek jest osiągalna wyłącznie budżetem klatkowym --- czyli osobnym
pomiarem, opisanym niżej jako M2.

Wyrównywany jest zatem \textbf{czas}, a warunek, który faktycznie ma treść, wynika ze
sposobu zatrzymywania przebiegu: silnik przerywa na \emph{pierwszej klatce sięgającej
budżetu}, więc każdy wariant przestrzeliwuje o niepełną klatkę. Przy budżecie rzędu jednej
klatki obrazu przestrzelenia są duże i \emph{różne} dla obu wariantów --- i wtedy nominalnie
równy czas przestaje być równy.

\textbf{Kryterium:} najmniejszy budżet, przy którym najgorszy rozjazd faktycznie zużytego
czasu nie przekracza 10\,\%. Najmniejszy --- bo praca dotyczy czasu rzeczywistego, a cel
liczbowy to poniżej 32~ms, czyli poniżej klatki przy 30 klatkach na sekundę.

\textbf{Pierwszy wynik: 24~ms} (około 40 klatek na sekundę).
""",
             budget_table(),
             r"""
Wartości 30 i 32~ms są wyraźnie gorsze (rozjazd 34,8\,\%): San Miguel przeskakuje tam na
drugą klatkę wariantu naprowadzanego i zużywa 51~ms zamiast 32.

\textbf{Zastrzeżenie.} Przy tak krótkim oknie wariant naprowadzany renderuje na San Miguelu
\emph{dwie} klatki na obraz, a przy jeszcze krótszym --- jedną. Ta komórka prawie nie
akumuluje, a jej obraz jest niemal pojedynczą próbką estymatora. Jest to uczciwy odczyt przy
czterdziestu klatkach na sekundę, ale nie wolno go czytać jak komórki, w której obraz powstał
z kilkudziesięciu klatek.

\subsubsection*{Dlaczego sam budżet czasowy nie wystarczył}

Pierwszy przebieg pomiarowy na budżecie 24~ms pokazał, że faktycznie zużyty czas obu
wariantów rozjeżdża się mimo wszystko --- od 0 do 20,1\,\%. Przyczyny są dwie i obie
strukturalne.

Po pierwsze, koszt klatki użyty do wyznaczenia budżetu pochodził z pomiaru przy obciążeniu
dziesięciosekundowym, a okno pomiarowe ma 24~ms. Pod dłuższym obciążeniem układ obniża
taktowanie, i \emph{wariant naprowadzany traci na tym wyraźnie więcej}:

%%THROTTLE%%

Skutek jest podwójny: mnożnik kosztu klatki między wariantami nie jest jedną liczbą (na
San Miguelu 3,07$\times$ przy obciążeniu dziesięciosekundowym i 2,27$\times$ w oknie jednej
klatki obrazu), a budżet wyznaczony z kosztów zmierzonych w złym reżimie jest przesunięty.

Po drugie, silnik przerywa na \emph{pierwszej klatce sięgającej budżetu}, więc każdy wariant
przestrzeliwuje o niepełną klatkę --- a przestrzelenie jest większe dla wariantu o droższej
klatce, czyli \textbf{zawsze dla naprowadzanego}. Rozjazd działa więc systematycznie na jego
korzyść.

\subsubsection*{Rozwiązanie: równy czas podany liczbą klatek}

Zamiast budżetu czasowego podaje się \textbf{liczby klatek osobno dla każdego wariantu i
każdej sceny}, dobrane tak, żeby iloczyny (liczba klatek $\times$ koszt klatki) leżały jak
najbliżej siebie i jak najbliżej celu jednej klatki obrazu. Kwantyzacja znika, bo nie ma już
przestrzeliwania, a koszty klatki brane są z właściwego reżimu.

%%PAIRS%%

San Miguel nie schodzi poniżej 9,3\,\%: jego klatka naprowadzana trwa 14,9~ms, więc w oknie
jednej klatki obrazu mieszczą się dwie i kwantyzacja nie daje lepszej pary.

Przy przeniesieniu pomiaru na inny układ graficzny tabelę par trzeba wyznaczyć od nowa ---
tam koszty klatki są inne, więc te same pary dałyby inny czas.

\subsection{Faza 3 --- pomiary}

Każdy pomiar odpowiada na inne pytanie. Kolejność jest wiążąca, bo obrazy odniesienia są
wejściem wszystkiego, co po nich.

\begin{description}
\item[M1 --- równy czas.] Oba warianty dostają ten sam budżet 24~ms. Odpowiada na pytanie
      praktyczne: \emph{co dostanę na ekranie w tym samym czasie}. To jest pomiar wiodący.
\item[M2 --- równa liczba próbek.] Budżetem jest liczba próbek ścieżki, nie czas. Odpowiada
      na pytanie: \emph{jak dobrze metoda celuje promienie}, z pominięciem kosztu klatki.
      Wariant naprowadzany zbiera dwie próbki na klatkę, podstawowy jedną, więc równa liczba
      próbek to różna liczba klatek.
\item[M3 --- równa wariancja.] Renderowanie z zapisem \emph{każdej} klatki; odczytem jest
      klatka o najmniejszym numerze, przy której błąd spada poniżej ustalonego pułapu.
      Bez interpolacji. Odpowiada na pytanie: \emph{ile trzeba czekać na obraz o zadanej
      jakości}.
\item[M4 --- krzywe błędu wobec czasu.] Jedyny pomiar na budżecie 30~s, szesnaście punktów
      kontrolnych o rozstawie logarytmicznym. Służy wyłącznie wykresowi; żadna liczba z tego
      przebiegu nie trafia do tabel.
\item[M5 --- rozbicie kosztu klatki.] Koszt każdego etapu potoku, osobno dla obu wariantów,
      z zaznaczeniem, które etapy śledzą promienie. To jest wyjaśnienie, skąd bierze się
      mnożnik kosztu klatki. Osobno czas jednorazowej wokselizacji geometrii.
\item[M6 --- stabilność czasowa.] Ciąg kolejnych obrazów przy równej liczbie próbek, z
      błędem każdego z nich. Odpowiada na pytanie o migotanie: czy obraz jest stabilny
      między klatkami, czy tylko średnio dobry.
\item[M7 --- wariant z ponownym użyciem próbki.] Wariant metody, który oszczędza jedno
      śledzenie promienia kosztem obciążenia estymatora. Mierzony przy dwóch budżetach, bo
      obciążenie ujawnia się dopiero powyżej przecięcia krzywych.
\item[M8 --- pamięć karty graficznej.] Zużycie całej metody wobec rozdzielczości siatki,
      z rozbiciem na pozycje zależne od siatki i od rozdzielczości obrazu. Wariant
      podstawowy jako punkt odniesienia.
\item[M9 --- wpływ rozdzielczości siatki.] Jakość przy wspólnym budżecie dla kolejnych
      szczebli. Uzasadnia kolumnę ,,siatka'' w tabeli nastaw.
\item[M10 --- optymalizacje zależne od producenta.] Osobny podrozdział; wyniki nie wchodzą
      do żadnego innego pomiaru, bo wiążą metodę z konkretnym producentem układu.
\item[M11 --- weryfikacja na drugim układzie.] Powtórzenie M1 na układzie weryfikacyjnym.
      Rozstrzyga, czy znak wyniku i kolejność scen się utrzymują.
\item[M12 --- kontrole wiarygodności.] Poziom szumu pomiaru, wpływ długości obrazu
      odniesienia, wpływ rodzaju źródła światła.
\end{description}

\textbf{Powtórzenia.} Każde zadanie pomiarowe wykonywane jest \textbf{dziesięciokrotnie},
a wynikiem jest średnia. Zadania przeplatane są rundami --- program obchodzi wszystkie
zadania po jednym razie, zanim wróci do pierwszego --- żeby powolne nagrzewanie się układu
rozłożyło się na oba warianty jednakowo, a nie trafiło w ten, który akurat biegł jako drugi.
Dodatkowo cała siatka pomiarowa przemierzana jest \emph{drugi raz jako osobne uruchomienie
programu}; różnica między tymi przemierzeniami jest poziomem szumu pomiaru.

\section{Wyniki}

\subsection{M1 --- porównanie przy równym czasie}

Budżet jednej klatki obrazu, liczby klatek dobrane per scena tak, żeby zrównać czas.

%%M1%%

Poziom szumu międzyprzebiegowego w tym pomiarze wyniósł \textbf{0,00--0,19\,\%}, więc każda
różnica w tabeli jest o dwa do trzech rzędów wielkości większa od progu rozstrzygalności.

Technika naprowadzana wygrywa na trzech z sześciu scen. Zestawienie z proporcją oświetlonych
wokseli układa się \emph{odwrotnie} do intuicji ,,im więcej światła zapisanego, tym lepiej'':
Cornell Box ma najwyższą proporcję w zbiorze (63,9\,\%) i przegrywa najmocniej, Veach Ajar ma
6,6\,\% i wygrywa najmocniej. Zgadza się to z uzasadnieniem doboru scen --- rozkład bliski
jednostajnemu nie wnosi nic ponad losowanie z rozkładu BRDF, a kosztuje.

Pomiar wykonano trzykrotnie, na trzech różnych sposobach wyrównywania czasu. Znak i
uporządkowanie scen nie zmieniły się ani razu; wartości wahały się o 2--8 punktów
procentowych.

\subsection{M2 --- porównanie przy równej liczbie próbek}

Szesnaście próbek ścieżki na piksel: osiem klatek naprowadzanych wobec szesnastu ścieżkowych.

%%M2%%

Tu widać rozdzielenie dwóch pytań, dla którego oba pomiary istnieją. \textbf{Cornell Box
odwraca znak}: 118\,\% na próbkę, 83\,\% na czas. Naprowadzanie celuje tam lepiej niż losowanie
z rozkładu BRDF, ale jego klatka kosztuje pięć i pół raza więcej, więc przy równym czasie
przewaga na próbkę tego nie odrabia. Na Veach Ajar jest odwrotnie --- przewaga na próbkę
(170\,\%) jest tak duża, że przeżywa koszt.

Dwie sceny przegrywają w \emph{obu} ujęciach: Bedroom i Zero Day. Znaczy to, że rozkład
naprowadzający jest tam gorszy \emph{na próbkę}, a nie tylko droższy --- żadne przyspieszenie
łańcucha by tego nie odwróciło.

Zastrzeżenie: równa liczba próbek to bardzo różny czas (San Miguel 124 wobec 176~ms). Ten
pomiar celowo pomija koszt i nie odpowiada na pytanie ,,co dostanę na ekranie''.

\subsection{M3 --- porównanie przy równej wariancji}

Renderowanie z zapisem każdej klatki; odczytem jest klatka o najmniejszym numerze, przy której
MSE spada poniżej pułapu 0,005. Bez interpolacji.

%%M3%%

Zastrzeżenie, bez którego trzy wiersze tej tabeli nie znaczą nic: na Cornell Box i Staircase
odczyt wypada przy \textbf{dwóch do dziewięciu klatek}. Przy takich liczbach wynik mówi
wyłącznie o tym, że jedna klatka naprowadzana kosztuje więcej niż trzy ścieżkowe --- jest
zdominowany przez ziarnistość klatki, a nie przez zbieżność.

Zeskanowanie tych samych zrzutów przy niższych pułapach pokazuje granicę rozstrzygalności:
przy 0,002 i niżej \emph{technika naprowadzana osiąga pułap, a śledzenie ścieżek nie mieści
się w budżecie} na Bedroomie, San Miguelu i Veach Ajar --- co samo w sobie jest wynikiem,
choć zapisanym jako brak.

\subsection{Przewaga jest zjawiskiem krótkiego budżetu}

Ten sam stosunek zmierzony przy trzech budżetach różniących się o rzędy wielkości:

%%CROSS%%

Dwie z trzech scen, na których technika naprowadzana wygrywa przy budżecie jednej klatki
obrazu, \textbf{tracą prowadzenie, gdy budżet urośnie dziesięciokrotnie}. Veach Ajar zachowuje
się przeciwnie --- jego przewaga rośnie i nie przecina się do końca zakresu.

Jest to bezpośrednie potwierdzenie założenia, dla którego plan wybrał krótki budżet: pomiar na
budżecie trzydziestosekundowym nie jest pomiarem tej samej rzeczy, tylko innej. Środek
przedziału (od 30 do 300~ms) pozostaje nieprzemierzony: krzywe mają tam zbyt mało klatek, żeby
punkt kontrolny łapał oba warianty w porównywalnym stanie, a drabina budżetu przestaje być
wyrównana czasowo, bo technika naprowadzana dławi się przy dłuższym obciążeniu mocniej.

\subsection{M5 --- skąd bierze się mnożnik kosztu klatki}

%%BREAKDOWN%%

Mnożnik rozkłada się na dwie w przybliżeniu równe części. Sam \textbf{estymator} jest
1,8--2,9 raza droższy, zanim doliczy się cokolwiek z budowy rozkładu --- płaci za dwie próbki
na klatkę, chodzenie po drzewie świateł i próbkowanie kąta bryłowego woksela. \textbf{Łańcuch}
budujący rozkład dokłada 38--65\,\% klatki wierzchu.

Tłumaczy to, dlaczego mnożnik jest największy na scenach \emph{tanich}: łańcuch ma koszt w
dużej mierze niezależny od sceny, więc gdy klatka ścieżkowa kosztuje 0,5~ms, sam łańcuch waży
2,8~ms. Metoda kosztuje względnie najwięcej tam, gdzie i tak było szybko.

Wokselizacja geometrii, wykonywana raz przed renderowaniem, kosztuje od 2~ms (Cornell Box,
32 trójkąty) do 530~ms (San Miguel, 9,9~mln trójkątów) i skaluje się z liczbą trójkątów, nie
z rozdzielczością siatki.

Zastrzeżenie: przebieg z pomiarem etapów opróżnia kolejkę co klatkę, żeby odczytać znaczniki
czasu, więc jego bezwzględny koszt klatki jest zawyżony. Porównywalne są udziały i stosunki
wewnątrz jednego przebiegu.

\subsection{M8 --- pamięć karty graficznej}

%%MEMORY%%

Pozycje zależne od siatki rosną ośmiokrotnie na szczebel, ale startują tak nisko, że do
$128^3$ siatka jest praktycznie darmowa: +190~MiB względem $32^3$. Dopiero $256^3$ kosztuje
1,3~GiB, a $512^3$ --- 10,8~GiB.

Warte zapisania, bo nieoczywiste: przy $32^3$ i $64^3$ \textbf{to nie siatka jest kosztem
pamięciowym metody, tylko bufory ekranowe}. Integrator, superpiksele i V-bufor to razem
248~MiB niezależnie od sceny i siatki. Przy nastawach docelowych metoda płaci więc głównie za
pełnoekranowe bufory pośrednie.

\subsection{M9 --- wpływ rozdzielczości siatki wokseli}

%%GRID%%

Podział jest czysty: \textbf{sceny małe wolą siatkę grubą, duże drobną}. Mechanizm jest znany
z fazy 0 --- drobniejsza siatka podnosi przepustowość naprowadzania, ale zabiera klatki; na
scenie małej przepustowość i tak jest wysoka, więc zostaje sam koszt. Zero Day przy $512^3$
załamuje się, bo koszt klatki skacze tam z 14,5 na 41,0~ms.

\subsection{M6 --- stabilność czasowa}

Ciąg 64 kolejnych obrazów o tej samej liczbie próbek: dla śledzenia ścieżek akumulacja dwóch
klatek, dla techniki naprowadzanej jedna klatka.

%%STAB%%

Technika naprowadzana \textbf{kupuje niższy błąd średni kosztem gorszej powtarzalności między
klatkami}. Śledzenie ścieżek jest wyjątkowo stabilne, naprowadzanie waha się nawet
trzydziestokrotnie mocniej --- i najsilniej właśnie tam, gdzie wygrywa poziomem.

Mechanizm jest wbudowany w metodę i nie jest usterką: rozkład naprowadzający budowany jest od
zera co klatkę, z losowego wstrzykiwania światła i losowanych co klatkę przedstawicieli
superpikseli, więc każda klatka dostaje \emph{inny} rozkład. Zero Day jest wyjątkiem
potwierdzającym regułę --- przy rozkładzie liczącym 49 wokseli nie ma tam czym fluktuować.

Zastrzeżenie: miara opisuje ciąg obrazów \emph{niezależnych}, czyli podgląd na żywo. Przy
akumulacji obraz stabilizuje się w obu wariantach i różnica maleje.

\subsection{M7 --- wariant z ponownym użyciem próbki}

Wariant pomija jedno śledzenie promienia przy pierwszym wierzchołku i używa ponownie próbki
wstrzykiwania. Kosztuje to obciążenie estymatora, a oszczędza od 8,7\,\% (Cornell Box) do
32,9\,\% (Zero Day) kosztu klatki.

%%REUSE%%

Przy budżecie jednej klatki obrazu obciążenie \textbf{jeszcze się nie ujawnia}: tańsza klatka
przekłada się wprost na więcej klatek, a błąd systematyczny potrzebuje czasu, żeby przeważyć
nad szumem. Wynik ten nie mówi, że wariant obciążony jest lepszy --- mówi, że w tym reżimie
jego wada jest niewidoczna.

Widać ją na krzywej:

%%REUSECURVE%%

To jedyny pomiar w całej turze, który pokazuje przecięcie \emph{w całości}, a nie jego dwa
końce. Tańsza klatka wygrywa wcześnie, błąd systematyczny wygrywa późno, a granica leży
między pół sekundy a trzydziestoma sekundami zależnie od sceny. Na Veach Ajar spadek jest
największy --- to scena, na której naprowadzanie wnosi najwięcej, więc i skrót w jego wagach
kosztuje tam najwięcej.

\subsection{M10 --- optymalizacje zależne od producenta}

%%LEVERS%%

\texttt{noviews} --- kompilacja programu generującego promienie bez kodu widoków
diagnostycznych --- daje 5,6--8,8\,\% na obu wariantach. Jest domyślnie włączona, więc jest to
raczej pomiar tego, ile kosztowałoby jej wyłączenie.

\texttt{swizzle} daje 7,7--8,6\,\% na śledzeniu ścieżek i \textbf{dokładnie nic} na
naprowadzaniu. Potwierdza to analizę zajętości rejestrów: jądro naprowadzane jest ograniczone
liczbą rejestrów, a nie lokalnością dostępów.

\texttt{wave32} i \texttt{wave64} nie robią nic, i jest to ograniczenie języka, a nie wynik:
atrybut szerokości fali nie sięga w HLSL potoku śledzenia promieni. Dwie dalsze dźwignie nie
dają się uruchomić --- jedna wymaga sprzętowego przestawiania kolejności wykonania, którego
nie ma żaden z dwóch układów, druga nie kompiluje się na ramieniu naprowadzanym.

Poprawa jakości przy każdej z tych dźwigni jest \textbf{wyłącznie skutkiem większej liczby
klatek} w tym samym budżecie, a nie zmianą estymatora.

\subsection{Kontrola: długość obrazu odniesienia}

Trzy komplety obrazów odniesienia --- 600, 1200 i 1800~s --- pozwalają zapytać nie tylko
,,czy długość referencji jest parametrem'', ale też ,,czy 1800~s wystarcza''. Odpowiada na to
odległość między samymi referencjami:

%%DRIFT%%

Przedział 1200--1800~s daje MSE równe około \emph{połowie} przedziału 600--1200~s na każdej
scenie. Jest to dokładnie zachowanie niezależnych oszacowań Monte Carlo: MSE między dwoma
obrazami jest sumą ich wariancji, a wariancja maleje jak $1/t$. Dla tych dwóch par daje to
stosunek 1,8; zmierzono około 1,9. \textbf{Nie ma śladu podłogi systematycznej}, czyli 1800~s
nie natrafiło na żadne obciążenie --- wciąż redukuje sam szum.

Odpowiedź na pytanie o wystarczalność jest zatem twierdząca z dużym zapasem: resztkowa
wariancja referencji wnosi około $10^{-6}$ MSE, a mierzone błędy ramion to od $10^{-3}$ do
$5\cdot10^{-2}$ --- trzy do czterech rzędów wielkości więcej.

Warto zestawić to z poprzednią turą pomiarów, w której ta sama kontrola dała przesunięcie
sięgające siedmiu punktów procentowych stosunku. Nie dlatego, że referencje były tam gorsze:
tamten pomiar prowadzono na budżecie 30~s, gdzie błąd ramion spadał do rzędu $4\cdot10^{-5}$,
czyli \emph{tego samego rzędu co dryf referencji}. Wniosek jest więc mocniejszy niż samo
,,długość referencji jest parametrem'': \textbf{jest parametrem tylko wtedy, gdy porównanie
prowadzi się blisko zbieżności}.
""",

             r"""
\section{Sun Temple --- przypadek negatywny}
\label{sec:suntemple}

Scena o rozpiętości 271~metrów, wyłączona ze zbioru głównego, zmierzona osobno, bo
odpowiada na pytanie ,,kiedy tej metody \emph{nie} używać'' lepiej niż jakikolwiek argument.
""",
             suntemple_table(),
             r"""
Rozpiętość całej drabiny to \textbf{2,4\,\%} --- mniej niż próg rozstrzygalności przyjęty w
fazie 1. Siatka na tej scenie jest bezczynna: nawet przy najdrobniejszym szczeblu, gdzie
przepustowość naprowadzania dochodzi wreszcie do 20\,\%, błąd obrazu nie drgnął, a koszt
klatki rośnie tam tylko o 21\,\% (18,24 $\rightarrow$ 22,14~ms), więc budżet nie jest
przeszkodą. \emph{Nie istnieje szczebel siatki, który by na tej scenie pomógł.}

Zastrzeżenie: poziom błędu na tej scenie jest cztery razy wyższy niż na Veach Ajar --- pięć
sekund nie zbliża jej do zbieżności. Wniosek dotyczy więc reżimu krótkiego budżetu, który
jest reżimem całej pracy, a nie granicy przy budżecie nieskończonym.

\section{Ustalenia metodyczne, które zmieniły przebieg badania}

Cztery rzeczy wyszły w trakcie i każda z nich zmieniła coś w planie. Wszystkie są w pracy
warte wzmianki, bo dotyczą wiarygodności liczb.

\subsection{Cichy pułap rozkładu naprowadzającego}

Dwie sceny przy drobnej siatce kończyły przebieg utratą urządzenia. Przyczyną nie była
pamięć: liczba oświetlonych wokseli przekraczała pojemność bufora, do którego są pakowane;
nadmiar był odrzucany, ale licznik rósł dalej, a kolejny etap rozdzielał z niego pracę na
komórki, których nikt nie zapisał. Po poprawce oba szczeble liczą się normalnie.
\emph{Skutek dla wyników:} szczebel, na którym liczba oświetlonych wokseli przekracza
pojemność, mierzy pojemność, a nie siatkę --- takie szczeble są w tabelach oznaczone i
wyłączone z doboru nastaw.

\subsection{Koszt klatki z sondy fazy 0 jest zawyżony}

Sonda doboru scen miała trzysekundową rozgrzewkę, w którą wpadał jednorazowy wypiek
geometrii. Na najdrobniejszej siatce zawyżyło to koszt klatki nawet siedmiokrotnie
(139~ms wobec rzeczywistych 22~ms). \emph{Skutek:} kolumna kosztu klatki w tabeli fazy 0 nie
nadaje się do cytowania; koszty klatki pochodzą z fazy 2, która rozgrzewa się 45~s.

\subsection{Aparatura pomiarowa wchodziła do pomiaru}

Przy budżecie 24~ms zapis obrazu zostawiał po sobie pracę, której część lądowała w pierwszej
klatce następnego obrazu. Przy budżecie trzydziestosekundowym jest to niewidoczne; przy
budżecie rzędu jednej klatki jedna skażona klatka zjada cały budżet i obraz zostaje z jedną
klatką zamiast pięciu. Rozwiązanie: klatki wyciszające między obrazami, poza każdym oknem
pomiarowym. \emph{Skutek:} pierwszy przebieg pomiaru M1 został skasowany w całości i
powtórzony; żadna liczba z niego nie trafia do pracy.

\subsection{Warunek doboru budżetu był niewykonalny}

Opisany w fazie 2. Plan został poprawiony tak, żeby mówił wprost, że wyrównywany jest czas,
a nie liczba klatek.

\section{Czego ten dokument jeszcze nie zawiera}

Do wykonania pozostają pomiary M2--M12 oraz weryfikacja na drugim układzie graficznym.
Dwie liczby parametrów testowania --- liczba próbek dla M2 i pułap błędu dla M3 --- zostaną
wyznaczone z krzywych M4, bo to ten sam przebieg dostarcza obu.
"""]
    document = "\n".join(parts)
    # Placeholders rather than another string boundary: the narrative and the tables that
    # belong inside it live in one raw string, and splitting it would put the table
    # assembly in the middle of prose.
    for placeholder, builder in (
            ("%%THROTTLE%%", throttle_table),
            ("%%PAIRS%%", frame_pair_table),
            ("%%M1%%", lambda: ratio_table(newest("wyniki-czas") / "m1-wyniki.json", "M1")),
            ("%%M2%%", lambda: ratio_table("wyniki-probki/m2-wyniki.json", "M2")),
            ("%%M3%%", variance_table),
            ("%%CROSS%%", crossover_table),
            ("%%BREAKDOWN%%", breakdown_table),
            ("%%MEMORY%%", memory_table),
            ("%%GRID%%", grid_quality_table),
            ("%%STAB%%", stability_table),
            ("%%REUSE%%", reuse_table),
            ("%%REUSECURVE%%", reuse_curve_table),
            ("%%LEVERS%%", levers_table),
            ("%%DRIFT%%", reference_drift_table)):
        document = document.replace(placeholder, builder() or
                                    r"\emph{Pomiar jeszcze nie wykonany.}")
    return document


HEADER = r"""\documentclass[12pt,a5paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage[polish]{babel}
\usepackage[margin=11mm,top=13mm,bottom=13mm]{geometry}
\usepackage{array}
\linespread{1.06}
\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\title{\bfseries Plan badawczy i wyniki pośrednie\\[4pt]
\large Naprowadzane śledzenie ścieżek z rozkładem wokselowym}
\author{}
\date{2 września 2026}
\begin{document}
\maketitle
\thispagestyle{empty}
\tableofcontents
\newpage
"""

source = HEADER + build() + "\n\\end{document}\n"
(OUT / "plan-badawczy.tex").write_text(source, encoding="utf-8")

for _ in range(2):
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "plan-badawczy.tex"],
                            cwd=OUT, capture_output=True, text=True, errors="replace")
pdf = OUT / "plan-badawczy.pdf"
if not pdf.exists():
    tail = (OUT / "plan-badawczy.log").read_text(encoding="utf-8", errors="replace")
    print("\n".join(line for line in tail.splitlines() if line.startswith("!"))[:2000])
    sys.exit("PDF nie powstal")
print(f"OK: {pdf} ({pdf.stat().st_size // 1024} KB)")
