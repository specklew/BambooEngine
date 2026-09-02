#!/usr/bin/env python3
"""Turn a finished reconnaissance into one self-contained HTML page.

Reads what `recon.py run` and `recon.py references` left on disk — captures, their
sidecars and the reference images — scores every checkpoint against its reference and
writes a page with the frame-cost tables, the equal-time result, convergence curves and
the images themselves. Everything is inlined (thumbnails as data URIs, charts as SVG)
because the page is meant to be handed to someone, not served.

  python tools/recon_report_html.py --out <file.html>

Static measurements from the grid-resolution study (R17/R18 in
docs/plan-badawczy-realizacja.md) are carried in this file as constants: they are not
recoverable from the recon's own artefacts, and the page is where they belong next to
the frame costs they explain.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r tools/requirements.txt")

from bench_report import aggregate, load_rgb, score_image, sidecar_for  # noqa: E402
from recon import (RAYTRACER_DIR, ARMS, NOISE_INFLATION,  # noqa: E402
                   RATIO_VXPG, cell_id, cells, grid_resolutions, load_manifest,
                   state_for, under)

# The biased reuse variant is measured but never shown: this page reports unbiased arms only.
ARM_LABEL = {"PT": "Śledzenie ścieżek", "VXPG": "VXPG",
             "V15": "widok 15 (bez naprowadzania)", "NOGUIDE": "bez maszynerii",
             "before": "VXPG przed", "after": "VXPG po", "pt": "Śledzenie ścieżek"}
ARM_COLOR = {"PT": "var(--arm-pt)", "VXPG": "var(--arm-vxpg)",
             "V15": "var(--arm-base)", "NOGUIDE": "var(--arm-nog)",
             "before": "var(--arm-nog)", "after": "var(--arm-vxpg)", "pt": "var(--arm-pt)"}
SHOWN_ARMS = ("PT", "VXPG")

# Measured 2026-08-27, evaluation configs. Lit voxels via vxpg.cluster.dumpStats.
CENSUS = {
    "veach-ajar": [284, 848, 2247, 6246],
    "staircase": [2335, 8758, 31803, 101382],
    "kitchen": [2409, 8938, 33249, 110281],
    "bedroom": [1699, 5640, 16200, 34541],
    "sponza": [382, 1074, 3226, 10257],
    "bistro-exterior": [662, 1983, 5521, 13924],
}
MAX_EXTENT = {"veach-ajar": 17.076, "staircase": 10.338, "kitchen": 8.655,
              "bedroom": 7.204, "sponza": 29.767, "bistro-exterior": 184.419}
# Grid resolution vs frame cost, VXPG, 8 s runs without timestamps. None = over the leaf cap.
GRID_COST = {
    "veach-ajar": [4.630, 4.708, 4.820, 5.185],
    "sponza": [13.990, 14.273, 14.600, 15.300],
    "staircase": [7.361, 7.633, 8.491, None],
    "kitchen": [8.896, 9.193, None, None],
    "bedroom": [9.364, 9.521, 9.938, None],
    "bistro-exterior": [35.573, 36.362, 36.880, 38.271],
}
# Cluster occupancy at 64 cubed under the final regime (smoke logs of this recon).
CLUSTERS = {
    "veach-ajar--own": (848, 16, 21.7), "veach-ajar--point": (714, 3, 99.6),
    "staircase--own": (8758, 31, 13.7), "staircase--point": (7639, 32, 11.5),
    "kitchen--own": (8938, 27, 12.9), "bedroom--own": (5640, 21, 34.9),
    "sponza--own": (1074, 32, 20.8), "bistro-exterior--own": (1983, 32, 41.4),
}
LEAF_CAP = 32768

# The previous reconnaissance, 2026-08-27, read off its own recon.md. Same protocol, same
# manifest, same machine, but a DIAGNOSTIC build — this run is Release. R9 measured the two
# builds agreeing within 1% for both techniques, and the PT column of the delta table is the
# control that shows it: path tracing is untouched by everything that changed between the
# runs, so whatever it moves by bounds the build change plus bench repeatability together.
# FLIP is carried for orientation only: this run scored against a reference rendered fresh,
# so the two FLIP columns are not the same measurement to five decimals.
PREVIOUS_RECON = {
    "veach-ajar--own":      {"PT": 0.814,  "VXPG": 4.714,  "flipPT": 0.02061, "flipVXPG": 0.01611},
    "veach-ajar--point":    {"PT": 0.884,  "VXPG": 4.582,  "flipPT": 0.02188, "flipVXPG": 0.01884},
    "staircase--own":       {"PT": 1.686,  "VXPG": 7.643,  "flipPT": 0.00614, "flipVXPG": 0.00643},
    "staircase--point":     {"PT": 1.766,  "VXPG": 7.558,  "flipPT": 0.00706, "flipVXPG": 0.00613},
    "kitchen--own":         {"PT": 2.001,  "VXPG": 9.216,  "flipPT": 0.01170, "flipVXPG": 0.01597},
    "bedroom--own":         {"PT": 2.321,  "VXPG": 9.543,  "flipPT": 0.01567, "flipVXPG": 0.02262},
    "sponza--own":          {"PT": 3.829,  "VXPG": 14.262, "flipPT": 0.01160, "flipVXPG": 0.01443},
    "bistro-exterior--own": {"PT": 11.241, "VXPG": 36.336, "flipPT": 0.01752, "flipVXPG": 0.02601},
}
PREVIOUS_LABEL = "27.08.2026"


# ------------------------------------------------------------------ data gathering

def arm_checkpoints(directory, reference):
    """Every capture of one arm, aggregated per checkpoint ordinal."""
    per_checkpoint = {}
    for png in sorted(directory.rglob("*.png")):
        if png.name.endswith(".flip.png"):
            continue
        sidecar = sidecar_for(png)
        benchmark = sidecar.get("benchmark", {})
        index = benchmark.get("checkpointIndex", 0)
        scores, _ = score_image(reference, png)
        entry = per_checkpoint.setdefault(index, {"flip": [], "ms": [], "seconds": 0.0,
                                                  "frames": 0, "path": png})
        entry["flip"].append(scores["flipMean"])
        entry["ms"].append(benchmark.get("meanFrameMs", 0.0))
        entry["seconds"] = sidecar.get("raytracing", {}).get("accumulatedTime", 0.0)
        entry["frames"] = sidecar.get("raytracing", {}).get("frameIndex", 0)
    out = []
    for index in sorted(per_checkpoint):
        entry = per_checkpoint[index]
        out.append({"index": index, "seconds": entry["seconds"], "path": entry["path"],
                    "frames": entry["frames"],
                    "flip": aggregate(entry["flip"]), "ms": aggregate(entry["ms"])})
    return out


def gather(manifest, run_root, reference_root):
    data = []
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        reference_path = reference_root / f"{name}.png"
        if not reference_path.exists():
            print(f"skip {name}: no reference")
            continue
        reference = load_rgb(reference_path)
        arms = {}
        for label, _, _ in ARMS:
            if label not in SHOWN_ARMS:
                continue
            directory = run_root / name / label
            if not directory.exists():
                continue
            checkpoints = arm_checkpoints(directory, reference)
            if checkpoints:
                arms[label] = checkpoints
        data.append({"cell": name, "scene": scene, "light": light,
                     "state": state_for(scene, light), "reference": reference_path,
                     "exposure": scene["exposure"],
                     "contrast": scene.get("overrides", {}).get("contrast", 1.0),
                     "arms": arms})
    return data


# ------------------------------------------------------------------ rendering helpers

def thumbnail(path, width=1280, quality=86):
    image = Image.open(path).convert("RGB")
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def bar_chart(rows, unit="ms", width=760, row_height=30):
    """Horizontal bars. rows = [(label, value, css_colour_var), ...]."""
    top = max((value for _, value, _ in rows), default=1.0) or 1.0
    label_width, pad = 190, 74
    height = row_height * len(rows) + 8
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="bars">']
    for i, (label, value, colour) in enumerate(rows):
        y = i * row_height + 4
        span = (width - label_width - pad) * (value / top)
        parts.append(f'<text x="0" y="{y + 15}" class="bar-label">{label}</text>')
        parts.append(f'<rect x="{label_width}" y="{y + 4}" width="{max(span, 1):.1f}" '
                     f'height="15" rx="2" fill="{colour}"/>')
        parts.append(f'<text x="{label_width + span + 7:.1f}" y="{y + 16}" class="bar-value">'
                     f'{value:.3f}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def convergence_chart(arms, width=430, height=210):
    """FLIP against accumulated seconds, log x, one line per arm."""
    points = [(c["seconds"], c["flip"]["mean"]) for arm in arms.values() for c in arm]
    if not points:
        return ""
    xs = [x for x, _ in points if x > 0]
    ys = [y for _, y in points if y > 0]
    if not xs or not ys:
        return ""
    x0, x1 = np.log10(min(xs)), np.log10(max(xs))
    y0, y1 = np.log10(min(ys)), np.log10(max(ys))
    if x1 - x0 < 1e-9:
        x1 = x0 + 1e-9
    if y1 - y0 < 1e-9:
        y1 = y0 + 1e-9
    left, right, top, bottom = 46, 10, 12, 30

    def sx(v):
        return left + (width - left - right) * (np.log10(v) - x0) / (x1 - x0)

    def sy(v):
        return top + (height - top - bottom) * (1 - (np.log10(v) - y0) / (y1 - y0))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="curve">']
    for frac in (0.0, 0.5, 1.0):
        y = top + (height - top - bottom) * frac
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                     f'class="grid"/>')
    parts.append(f'<text x="{left - 6}" y="{top + 9}" class="axis" text-anchor="end">'
                 f'{max(ys):.4f}</text>')
    parts.append(f'<text x="{left - 6}" y="{height - bottom + 4}" class="axis" '
                 f'text-anchor="end">{min(ys):.4f}</text>')
    parts.append(f'<text x="{left}" y="{height - 8}" class="axis">{min(xs):.2f} s</text>')
    parts.append(f'<text x="{width - right}" y="{height - 8}" class="axis" '
                 f'text-anchor="end">{max(xs):.0f} s</text>')
    for label, checkpoints in arms.items():
        path = " ".join(
            f'{"M" if i == 0 else "L"}{sx(c["seconds"]):.1f},{sy(c["flip"]["mean"]):.1f}'
            for i, c in enumerate(checkpoints) if c["seconds"] > 0 and c["flip"]["mean"] > 0)
        if not path:
            continue
        parts.append(f'<path d="{path}" fill="none" stroke="{ARM_COLOR[label]}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        for c in checkpoints:
            if c["seconds"] > 0 and c["flip"]["mean"] > 0:
                parts.append(f'<circle cx="{sx(c["seconds"]):.1f}" '
                             f'cy="{sy(c["flip"]["mean"]):.1f}" r="2.6" '
                             f'fill="{ARM_COLOR[label]}"/>')
    parts.append("</svg>")
    return "".join(parts)


def verdict_class(ratio):
    if ratio >= 105:
        return "win"
    if ratio >= 95:
        return "tie"
    return "loss"


# ------------------------------------------------------------------ page

# ------------------------------------------------------------------ K5: grid ladder

def gather_grid(manifest, grid_root, reference_root):
    """Every voxel-grid rung of every cell, scored against the cell's own reference.

    The reference is path-traced and therefore grid-independent, so one 600 s image
    serves every rung — that is why this study needed no new references."""
    out = []
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        reference_path = reference_root / f"{name}.png"
        if not reference_path.exists():
            continue
        reference = load_rgb(reference_path)
        rungs = []
        for resolution in grid_resolutions(scene, manifest):
            directory = grid_root / name / f"g{resolution}"
            if not directory.exists():
                continue
            checkpoints = arm_checkpoints(directory, reference)
            if not checkpoints:
                continue
            last = checkpoints[-1]
            rungs.append({"dim": resolution, "flip": last["flip"], "ms": last["ms"],
                          "path": last["path"], "voxel": MAX_EXTENT[scene["id"]] / resolution,
                          "lit": CENSUS[scene["id"]][[32, 64, 128, 256].index(resolution)]})
        if rungs:
            out.append({"cell": name, "scene": scene, "rungs": rungs})
    return out


def line_chart(points, width=430, height=200, colour="var(--arm-vxpg)", fmt="{:.4f}"):
    """One series against a log-spaced integer x (the grid ladder)."""
    if len(points) < 2:
        return ""
    xs = [np.log2(x) for x, _ in points]
    ys = [y for _, y in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if y1 - y0 < 1e-12:
        y1 = y0 + 1e-12
    left, right, top, bottom = 54, 12, 14, 30

    def sx(v):
        return left + (width - left - right) * (np.log2(v) - x0) / max(x1 - x0, 1e-9)

    def sy(v):
        return top + (height - top - bottom) * (1 - (v - y0) / (y1 - y0))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="curve">']
    for frac in (0.0, 0.5, 1.0):
        y = top + (height - top - bottom) * frac
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
    parts.append(f'<text x="{left - 6}" y="{top + 9}" class="axis" text-anchor="end">{fmt.format(y1)}</text>')
    parts.append(f'<text x="{left - 6}" y="{height - bottom + 4}" class="axis" text-anchor="end">{fmt.format(y0)}</text>')
    path = " ".join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(y):.1f}'
                    for i, (x, y) in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>')
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{colour}"/>')
        parts.append(f'<text x="{sx(x):.1f}" y="{height - 8}" class="axis" text-anchor="middle">{x}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grid_section(grid_data):
    if not grid_data:
        return ""
    rows = []
    for entry in grid_data:
        base = next((r for r in entry["rungs"] if r["dim"] == 64), entry["rungs"][0])
        best = min(entry["rungs"], key=lambda r: r["flip"]["mean"])
        for rung in entry["rungs"]:
            delta = (base["flip"]["mean"] / rung["flip"]["mean"] - 1) * 100
            mark = "best" if rung is best else ""
            lit = f"{rung['lit']:,}".replace(",", " ")
            rows.append(
                f"<tr class='{mark}'><td class='cell'>{entry['cell']}</td>"
                f"<td class='num'>{rung['dim']}&#179;</td>"
                f"<td class='num'>{rung['voxel']:.3f}</td>"
                f"<td class='num'>{lit}</td>"
                f"<td class='num'>{rung['flip']['mean']:.5f}</td>"
                f"<td class='num'>{rung['ms']['mean']:.3f}</td>"
                f"<td class='num'><span class='pill {verdict_class(100 + delta)}'>"
                f"{delta:+.1f} %</span></td></tr>")
    charts = "".join(
        f"<figure class='curve-block'><figcaption>{e['cell']} &mdash; FLIP</figcaption>"
        + line_chart([(r["dim"], r["flip"]["mean"]) for r in e["rungs"]])
        + f"<figcaption style='margin-top:10px'>{e['cell']} &mdash; ms/klatkę</figcaption>"
        + line_chart([(r["dim"], r["ms"]["mean"]) for r in e["rungs"]],
                     colour="var(--arm-nog)", fmt="{:.2f}")
        + "</figure>" for e in grid_data)
    best_lines = "".join(
        f"<tr><td class='cell'>{e['cell']}</td>"
        f"<td class='num'>{min(e['rungs'], key=lambda r: r['flip']['mean'])['dim']}&#179;</td>"
        f"<td class='num'>{max(r['dim'] for r in e['rungs'])}&#179;</td></tr>" for e in grid_data)
    return f"""
<section id="k5">
  <h2>K5 &mdash; jakość i koszt wobec rozdzielczości siatki</h2>
  <p>Ramię naprowadzane na każdym szczeblu, jaki komórka może udźwignąć przy limicie
  32 768 liści drzewa świateł. Kolumna procentowa porównuje z 64&#179;, czyli z rozdzielczością,
  na której biegł rekonans; wiersz najlepszy w komórce jest wyróżniony.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>komórka</th><th>siatka</th><th>woksel [m]</th><th>oświetlone woksele</th>
      <th>FLIP</th><th>ms/klatkę</th><th>wobec 64&#179;</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div>
  <h3>Najlepszy szczebel wobec sufitu limitu</h3>
  <div class="scroll">
  <table>
    <thead><tr><th>komórka</th><th>najlepszy FLIP</th><th>sufit limitu</th></tr></thead>
    <tbody>{best_lines}</tbody>
  </table>
  </div>
  <div class="curve-grid">{charts}</div>
</section>"""


# ------------------------------------------------------------------ minimal-sample ratio

def gather_ratio(manifest, ratio_root, reference_root):
    """Guided rungs paired with the path-traced rung at exactly the arm's frame ratio."""
    out = []
    for scene, light in cells(manifest):
        name = cell_id(scene, light)
        reference_path = reference_root / f"{name}.png"
        if not reference_path.exists():
            continue
        reference = load_rgb(reference_path)
        series = {}
        for label in ("VXPG", "PT-x5"):
            directory = ratio_root / name / label
            if directory.exists():
                checkpoints = arm_checkpoints(directory, reference)
                if checkpoints:
                    series[label] = {c["frames"]: c for c in checkpoints}
        pairs = []
        for guided, baseline, multiple in (("VXPG", "PT-x5", RATIO_VXPG),):
            if guided not in series or baseline not in series:
                continue
            for frames in sorted(series[guided]):
                partner = series[baseline].get(frames * multiple)
                if partner is None:
                    continue
                g, p = series[guided][frames], partner
                pairs.append({"arm": guided, "frames": frames, "ratio": multiple,
                              "pt_frames": frames * multiple,
                              "g_flip": g["flip"]["mean"], "p_flip": p["flip"]["mean"],
                              "g_time": g["seconds"], "p_time": p["seconds"]})
        if pairs:
            out.append({"cell": name, "pairs": pairs})
    return out


def ratio_section(ratio_data):
    if not ratio_data:
        return ""
    blocks = []
    for entry in ratio_data:
        rows = "".join(
            f"<tr><td class='cell'>{p['arm']}</td>"
            f"<td class='num'>{p['frames']}</td><td class='num'>{p['pt_frames']}</td>"
            f"<td class='num'>{p['g_time'] * 1000:.1f}</td>"
            f"<td class='num'>{p['p_time'] * 1000:.1f}</td>"
            f"<td class='num muted'>{(p['g_time'] / p['p_time'] - 1) * 100:+.0f} %</td>"
            f"<td class='num'>{p['g_flip']:.5f}</td><td class='num'>{p['p_flip']:.5f}</td>"
            f"<td class='num'><span class='pill "
            f"{verdict_class(p['p_flip'] / p['g_flip'] * 100)}'>"
            f"{p['p_flip'] / p['g_flip'] * 100:.0f} %</span></td></tr>"
            for p in entry["pairs"])
        blocks.append(f"""
<article class="cell-block">
  <h3>{entry['cell']}</h3>
  <div class="scroll">
  <table>
    <thead><tr><th>ramię</th><th>klatek</th><th>klatek PT</th><th>czas ramienia [ms]</th>
      <th>czas PT [ms]</th><th>rozjazd czasu</th><th>FLIP ramienia</th><th>FLIP PT</th>
      <th>ramię wobec PT</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
</article>""")
    return f"""
<section id="proporcja">
  <h2>Reżim minimalnych próbek przy ustalonej proporcji klatek</h2>
  <p>Zamiast równego czasu &mdash; <b>równa proporcja klatek</b>. Ramię naprowadzane renderuje
  4, 8, &hellip; 32 klatki; śledzenie ścieżek dostaje dokładnie {RATIO_VXPG}&times; tyle, bo tyle
  wynosi zmierzony mnożnik kosztu naprowadzania. Kolumna
  <b>rozjazd czasu</b> mówi, o ile ustalona proporcja rozminęła się z prawdziwym kosztem
  w tej scenie: wartość dodatnia znaczy, że proporcja była dla naprowadzania zbyt hojna,
  ujemna &mdash; że zbyt skąpa.</p>
  <p class="note">Ten reżim jest bliższy zastosowaniu czasu rzeczywistego niż trzydziestosekundowa
  akumulacja: przy czterech klatkach obraz jest zdominowany szumem, a to właśnie tam
  naprowadzanie ma dawać przewagę.</p>
  {''.join(blocks)}
</section>"""


# ------------------------------------------------------------------ cost of guiding

# veach-ajar, own light, INDIRECT ONLY, 1080p, spp 1, 1 bounce, grid 64, 30 s budget,
# 2 images, 9 time checkpoints, quiet machine, 28.08.2026, Debug build.
#
# Four arms, all unbiased, all scored against the same 600 s path-traced reference:
#   PT      - the separate path-tracing shader
#   VXPG    - the guided integrator, guiding on
#   V15     - the guided integrator with the guide strategy skipped AT RUNTIME (debug
#             view 15). Weight-1 BSDF sample; the guide code is still in the kernel.
#   NOGUIDE - the same estimator with the guide strategy REMOVED FROM THE KERNEL (a temporary
#             compile switch, since deleted). V15 and NOGUIDE differ only in compilation, which
#             is why their fitted convergence curves land on top of each other.
STEP2_ARMS = [
    # label, ms/frame, frames in 30 s, FLIP, image mean, integrator node ms
    ("PT",      0.772, 38872, 0.02254, 0.34453, 0.5943),
    ("VXPG",    4.663,  6434, 0.01825, 0.34561, 2.5104),
    ("V15",     2.847, 10540, 0.02718, 0.34518, 0.7776),
    ("NOGUIDE", 2.654, 11306, 0.02654, 0.34522, 0.5786),
]
STEP2_REFERENCE_MEAN = 0.34570
# Per-sample error, from a log-log fit over each arm's own ladder (frames >= 100),
# read at 6434 frames - the guided arm's own 30 s frame count.
STEP2_PER_SAMPLE = [("PT", -0.3351, 0.04130, 1.000), ("VXPG", -0.3456, 0.01778, 2.323),
                    ("V15", -0.3718, 0.03236, 1.276), ("NOGUIDE", -0.3713, 0.03242, 1.274)]
STEP2 = {
    "PT": [(0.010, 12, 0.31735), (0.026, 34, 0.23421), (0.071, 94, 0.16966), (0.194, 255, 0.12205), (0.533, 699, 0.08685), (1.458, 1912, 0.06176), (3.996, 5236, 0.04431), (10.949, 14234, 0.03181), (30.000, 38872, 0.02254)],
    "VXPG": [(0.015, 2, 0.34496), (0.028, 4, 0.25116), (0.072, 14, 0.16461), (0.195, 40, 0.10955), (0.534, 113, 0.07355), (1.460, 312, 0.05008), (4.000, 857, 0.03484), (10.952, 2349, 0.02480), (30.002, 6434, 0.01825)],
    "V15": [(0.011, 2, 0.49566), (0.027, 8, 0.34864), (0.072, 24, 0.24592), (0.196, 68, 0.17403), (0.533, 186, 0.12117), (1.459, 512, 0.08316), (3.997, 1404, 0.05662), (10.949, 3847, 0.03886), (30.001, 10540, 0.02718)],
    "NOGUIDE": [(0.010, 3, 0.46913), (0.028, 10, 0.33593), (0.073, 26, 0.23963), (0.196, 72, 0.17017), (0.533, 199, 0.11832), (1.459, 546, 0.08112), (3.996, 1504, 0.05525), (10.950, 4126, 0.03791), (30.001, 11306, 0.02654)],
}
# Structure passes of one guided frame (node means, same runs). Everything that is not
# the integrator and not present at all in the path-traced frame.
STEP2_STRUCTURE = [("VXPG LightInjection", 0.8931), ("VXPG ClusterVisibility Check", 0.2527),
                   ("VXPG Superpixel Gather", 0.1446), ("VXPG Superpixel Associate", 0.1113),
                   ("VXPG ClusterVisibility Gather", 0.1016), ("VXPG VBuffer", 0.0991),
                   ("VXPG Cluster Seed", 0.0504), ("pozostałe 18 węzłów", 0.0975)]

STEP2_IMAGES = [
    (RAYTRACER_DIR / r"SavedUserData/Screenshots/recon-curve-refs/veach-ajar--own.png",
     "odniesienie &middot; 600 s śledzenia ścieżek"),
    (RAYTRACER_DIR / r"SavedUserData/Screenshots/step2/PT2/run-2026-08-28_14-04-07/Test State-Path Tracing-i0001-f038744.png",
     "śledzenie ścieżek &middot; 30 s &middot; 38 744 klatek"),
    (RAYTRACER_DIR / r"SavedUserData/Screenshots/step2/VX2/run-2026-08-28_14-05-18/Test State-Guided Path Tracing (VXPG)-i0001-f006433.png",
     "VXPG &middot; 30 s &middot; 6 433 klatki"),
    (RAYTRACER_DIR / r"SavedUserData/Screenshots/step2/V15T/run-2026-08-28_14-06-33/Test State-Guided Path Tracing (VXPG)-i0001-f010530.png",
     "widok 15 &middot; bez naprowadzania, kod obecny &middot; 10 530 klatek"),
    (RAYTRACER_DIR / r"SavedUserData/Screenshots/step2/NOG2/run-2026-08-28_14-07-49/Test State-Guided Path Tracing (VXPG)-i0001-f011296.png",
     "bez maszynerii &middot; kod wykompilowany &middot; 11 296 klatek"),
]


def step2_curve(width=880, height=300):
    """The four arms on one pair of log axes, FLIP against accumulated seconds."""
    arms = {label: [{"seconds": s, "flip": {"mean": f}} for s, _, f in points]
            for label, points in STEP2.items()}
    return convergence_chart(arms, width=width, height=height)


def cost_section():
    pt_flip = STEP2_ARMS[0][3]
    rows = "".join(
        "<tr class='" + ("best" if label == "VXPG" else "") + f"'><td>{ARM_LABEL[label]}</td>"
        f"<td class='num'>{ms:.3f}</td><td class='num'>{frames}</td>"
        f"<td class='num'>{node:.4f}</td>"
        f"<td class='num'>{flip:.5f}</td>"
        f"<td class='num'><span class='pill {verdict_class(pt_flip / flip * 100)}'>"
        f"{pt_flip / flip * 100:.0f} %</span></td>"
        f"<td class='num muted'>{mean:.5f}</td></tr>"
        for label, ms, frames, flip, mean, node in STEP2_ARMS)
    persample = "".join(
        f"<tr><td>{ARM_LABEL[label]}</td><td class='num'>{slope:+.3f}</td>"
        f"<td class='num'>{at:.5f}</td>"
        f"<td class='num strong'>{gain:.2f}&times;</td></tr>"
        for label, slope, at, gain in STEP2_PER_SAMPLE)
    structure = "".join(
        f"<tr><td class='cell'>{name}</td><td class='num'>{ms:.4f}</td></tr>"
        for name, ms in STEP2_STRUCTURE)
    structure_total = sum(ms for _, ms in STEP2_STRUCTURE)
    tiles = "".join(
        f"<figure><img src='{thumbnail(path)}' alt='{caption}' loading='lazy'>"
        f"<figcaption>{caption}</figcaption></figure>" for path, caption in STEP2_IMAGES)
    legend = "".join(
        f"<span><i style='background:{ARM_COLOR[label]}'></i>{ARM_LABEL[label]}</span>"
        for label in STEP2)
    bars = bar_chart([(ARM_LABEL[label], node, ARM_COLOR[label])
                      for label, _, _, _, _, node in STEP2_ARMS], unit=" ms")
    return f"""
<section id="krok2">
  <h2>Koszt naprowadzania, rozłożony na części</h2>
  <p>Cztery ramiona jednej sceny, ten sam budżet, ten sam obraz odniesienia. Trzy z nich
  całkują <b>ten sam estymator</b> i różnią się wyłącznie tym, ile maszynerii naprowadzania
  jest obecne w jądrze &mdash; to pozwala rozdzielić koszt <i>wykonywania</i> rozkładu od
  kosztu jego samej <i>obecności</i> w rejestrach, i osobno wycenić, co naprowadzanie daje.</p>

  <h3>Cztery poprawki, które musiały wejść pierwsze</h3>
  <p><b>Tryb tylko-pośredni pomija teraz wywołanie, a nie odrzuca wynik.</b> Oba ramiona
  próbkowały pulę świateł i śledziły promień cienia, po czym wyrzucały rezultat; to samo
  dotyczyło emisji pierwszego wierzchołka i jej kwerendy pdf. Śledzenie ścieżek zyskało na tym
  4&ndash;6 % klatki. <b>Widok 15 honoruje <code>IndirectOnly()</code></b> &mdash; wcześniej
  całkował inny obraz niż ramię, dla którego jest bazą. <b>Widok 4</b> przestał wracać przed
  swoją gałęzią klasyfikacji. <b>Czwarta, znaleziona dopiero teraz i najpoważniejsza:</b>
  <code>HeadlessRunner::ResolveDebugViews</code> zwracał na sztywno widok 0, kiedy nie podano
  <code>--debug-views</code> &mdash; i robił to <i>po</i> zastosowaniu konfiguracji, więc
  <code>guidingDebugView</code> z pliku konfiguracyjnego był po cichu kasowany.</p>
  <p class="note"><b>Sprostowanie.</b> Poprzednia wersja tej strony podawała, że wyniesienie
  łańcucha rozkładu z raygenu zdejmuje 68 % kosztu integratora, i że pominięcie łańcucha
  <i>w czasie wykonania</i> nie daje nic. Obie tezy opierały się na przebiegu opisanym jako
  &bdquo;widok 15&rdquo;, który przez powyższy błąd był w rzeczywistości <b>pełnym VXPG</b>
  &mdash; stąd jego podejrzana zgodność z ramieniem VXPG co do trzeciego miejsca po przecinku.
  Po naprawie liczby wyglądają inaczej, a wniosek jest odwrotny; tabele poniżej są tymi
  poprawionymi.</p>

  <h3>Wynik przy równym czasie</h3>
  <div class="scroll">
  <table>
    <thead><tr><th>ramię</th><th>ms/klatkę</th><th>klatek w 30 s</th>
      <th>węzeł integratora [ms]</th><th>FLIP</th><th>wobec PT</th>
      <th>średnia obrazu</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p class="note">Wszystkie średnie leżą w paśmie 0,3445&ndash;0,3456 przy odniesieniu
  {STEP2_REFERENCE_MEAN:.5f}, więc żadne ramię nie jest obciążone &mdash; różnice to szum,
  nie przesunięcie estymatora. <b>Naprowadzanie wygrywa przy równym czasie</b>: o 24 % wobec
  śledzenia ścieżek i o 45 % wobec własnej bazy bez naprowadzania.</p>

  <h3>Zbieżność</h3>
  <div class="legend">{legend}</div>
  <figure class="curve-block">{step2_curve()}</figure>
  <p>Obie osie logarytmiczne, dziewięć punktów kontrolnych od dwóch klatek do trzydziestu
  sekund. Linie <b>widoku 15</b> i <b>wariantu bez maszynerii</b> biegną równolegle i niemal
  się pokrywają &mdash; ten sam estymator przy różnej liczbie klatek. Linia VXPG leży pod nimi
  na całej długości: przewaga rozkładu nie zależy od budżetu.</p>

  <h3>Ile daje jedna próbka</h3>
  <div class="scroll">
  <table>
    <thead><tr><th>ramię</th><th>wykładnik zbieżności</th><th>FLIP przy 6434 klatkach</th>
      <th>wobec PT</th></tr></thead>
    <tbody>{persample}</tbody>
  </table>
  </div>
  <p>Dopasowanie log-log do własnej drabinki każdego ramienia, odczytane przy liczbie klatek,
  jaką w 30 s osiąga VXPG. Dwie liczby są tu istotne. Po pierwsze, <b>widok 15 i wariant bez
  maszynerii dają 0,03236 i 0,03242</b> &mdash; zgodność co do 0,2 %, czyli dowód, że
  wykompilowanie rozkładu nie zmieniło estymatora, tylko tempo. Po drugie, ten sam shader
  <b>bez naprowadzania</b> jest już 1,28&times; lepszy od śledzenia ścieżek na próbkę &mdash;
  to nie zasługa rozkładu, tylko wspólnego V-bufora. Samo naprowadzanie dokłada 2,32/1,28 =
  <b>1,82&times;</b>.</p>

  <h3>Gdzie idzie czas</h3>
  <figure class="bar-block"><figcaption>węzeł &bdquo;Raytrace Technique&rdquo;</figcaption>{bars}</figure>
  <p>Integrator VXPG kosztuje 2,5104 ms. Ten sam integrator z rozkładem pominiętym w czasie
  wykonania &mdash; 0,7776 ms. Ten sam bez rozkładu w jądrze &mdash; 0,5786 ms. Czyli
  <b>wykonywanie rozkładu kosztuje 1,733 ms, a jego sama obecność 0,199 ms</b>: dziewięć
  dziesiątych rachunku to praca, jedna dziesiąta to podatek rejestrowy. To <b>unieważnia plan
  kroku 1</b> &mdash; wyniesienie łańcucha do osobnego przebiegu może odzyskać najwyżej te
  0,199 ms, czyli 4,3 % klatki VXPG, a nie 1,7 ms.</p>
  <p>Warto też zauważyć, że integrator bez naprowadzania (0,5786 ms) jest <i>tańszy</i> od
  integratora śledzenia ścieżek (0,5943 ms) przy identycznej pracy &mdash; V-bufor zdejmuje
  z niego promień pierwotny. Cała nadwyżka klatki VXPG bez naprowadzania (2,654 wobec
  0,772 ms) siedzi więc w przebiegach budujących strukturę, nie w całkowaniu:</p>
  <div class="scroll">
  <table>
    <thead><tr><th>przebieg</th><th>ms/klatkę</th></tr></thead>
    <tbody>{structure}<tr class="best"><td class="cell"><b>razem</b></td>
      <td class="num"><b>{structure_total:.4f}</b></td></tr></tbody>
  </table>
  </div>
  <p class="note">Wstrzykiwanie światła to <b>0,893 ms</b> &mdash; jedna trzecia klatki bez
  naprowadzania i więcej niż cały integrator. Śledzi jeden promień na piksel ekranu, więc
  skaluje się rozdzielczością obrazu, nie liczbą wokseli, i jest pierwszym kandydatem do
  optymalizacji, przed czymkolwiek w raygenie.</p>

  <div class="tiles">{tiles}</div>
</section>"""


# ------------------------------------------------------------------ optimization sweep

# All eight cells re-measured 2026-08-28 on one binary, indirect-only, 1080p, spp 1,
# 1 bounce, 30 s, 2 images, 9 time checkpoints. 'before' restores the two shaders as they
# stood at 7b9ab9c, 'after' is HEAD; the C++ is identical in both, so the difference is
# exactly the four behaviour-changing commits.
PROGRESS = {
    "veach-ajar--own": {
        "reference_mean": 0.34570,
        "before": [(0.014, 2, 0.34458, 9.800), (0.027, 4, 0.24905, 6.127), (0.073, 15, 0.15952, 4.891), (0.198, 42, 0.10757, 4.703), (0.536, 116, 0.07279, 4.639), (1.460, 316, 0.04977, 4.612), (3.998, 866, 0.03471, 4.616), (10.950, 2371, 0.02474, 4.618), (30.004, 6485, 0.01823, 4.627)],
        "after": [(0.015, 2, 0.35553, 9.910), (0.028, 4, 0.25617, 6.302), (0.073, 14, 0.16554, 5.260), (0.197, 40, 0.11008, 4.853), (0.535, 113, 0.07369, 4.732), (1.460, 312, 0.05021, 4.687), (3.998, 856, 0.03490, 4.668), (10.952, 2349, 0.02489, 4.662), (30.002, 6432, 0.01829, 4.664)],
        "pt": [(0.010, 12, 0.31742, 0.780), (0.026, 34, 0.23300, 0.770), (0.071, 96, 0.16841, 0.743), (0.195, 262, 0.12113, 0.744), (0.533, 716, 0.08601, 0.745), (1.459, 1956, 0.06130, 0.746), (3.996, 5348, 0.04397, 0.747), (10.949, 14572, 0.03156, 0.751), (30.000, 39895, 0.02234, 0.752)],
    },
    "veach-ajar--point": {
        "reference_mean": 0.36085,
        "before": [(0.014, 2, 0.40760, 9.486), (0.027, 4, 0.29297, 6.094), (0.071, 14, 0.18607, 4.867), (0.194, 42, 0.12158, 4.563), (0.529, 118, 0.08113, 4.482), (1.448, 326, 0.05526, 4.447), (3.973, 894, 0.03823, 4.442), (10.918, 2460, 0.02699, 4.439), (30.002, 6753, 0.01959, 4.443)],
        "after": [(0.014, 2, 0.38422, 9.254), (0.026, 4, 0.28780, 5.827), (0.070, 14, 0.18650, 4.849), (0.192, 42, 0.12123, 4.519), (0.528, 120, 0.08091, 4.418), (1.447, 330, 0.05517, 4.384), (3.973, 910, 0.03807, 4.368), (10.919, 2502, 0.02684, 4.364), (30.003, 6872, 0.01950, 4.366)],
        "pt": [(0.009, 12, 0.33791, 0.794), (0.026, 32, 0.25248, 0.798), (0.070, 90, 0.18483, 0.773), (0.191, 248, 0.13203, 0.770), (0.526, 685, 0.09114, 0.768), (1.446, 1879, 0.06323, 0.770), (3.972, 5140, 0.04469, 0.773), (10.916, 14030, 0.03175, 0.778), (30.000, 38536, 0.02229, 0.779)],
    },
    "staircase--own": {
        "reference_mean": 0.11217,
        "before": [(0.023, 2, 0.09742, 15.315), (0.043, 4, 0.06664, 9.607), (0.106, 14, 0.04363, 7.593), (0.266, 36, 0.03130, 7.486), (0.679, 92, 0.02265, 7.426), (1.753, 237, 0.01656, 7.398), (4.515, 611, 0.01232, 7.390), (11.635, 1575, 0.00936, 7.388), (30.003, 4058, 0.00732, 7.393)],
        "after": [(0.023, 2, 0.09628, 15.298), (0.043, 4, 0.06622, 9.603), (0.106, 14, 0.04389, 7.855), (0.267, 36, 0.03116, 7.533), (0.682, 92, 0.02244, 7.412), (1.753, 238, 0.01643, 7.364), (4.514, 614, 0.01223, 7.351), (11.636, 1583, 0.00930, 7.351), (30.006, 4066, 0.00732, 7.381)],
        "pt": [(0.016, 10, 0.07969, 1.561), (0.040, 26, 0.05738, 1.549), (0.103, 67, 0.03868, 1.538), (0.264, 172, 0.02632, 1.534), (0.679, 445, 0.01854, 1.526), (1.749, 1150, 0.01358, 1.521), (4.511, 2982, 0.01034, 1.512), (11.632, 7704, 0.00813, 1.510), (30.000, 19896, 0.00643, 1.508)],
    },
    "staircase--point": {
        "reference_mean": 0.09045,
        "before": [(0.022, 2, 0.06616, 15.134), (0.042, 4, 0.04854, 9.477), (0.105, 14, 0.03130, 7.479), (0.266, 36, 0.02261, 7.392), (0.678, 92, 0.01692, 7.326), (1.747, 240, 0.01322, 7.293), (4.501, 618, 0.01063, 7.290), (11.622, 1594, 0.00892, 7.291), (30.002, 4112, 0.00784, 7.296)],
        "after": [(0.022, 2, 0.06438, 15.081), (0.042, 4, 0.04415, 9.455), (0.105, 14, 0.02971, 7.524), (0.264, 36, 0.02270, 7.346), (0.678, 93, 0.01695, 7.290), (1.746, 240, 0.01316, 7.261), (4.505, 621, 0.01053, 7.254), (11.622, 1602, 0.00888, 7.255), (30.003, 4136, 0.00781, 7.255)],
        "pt": [(0.016, 10, 0.06628, 1.600), (0.039, 24, 0.04613, 1.604), (0.102, 64, 0.03049, 1.596), (0.261, 164, 0.02141, 1.588), (0.675, 427, 0.01635, 1.582), (1.742, 1106, 0.01363, 1.576), (4.498, 2866, 0.01213, 1.570), (11.617, 7411, 0.01077, 1.568), (30.001, 19154, 0.00944, 1.566)],
    },
    "kitchen--own": {
        "reference_mean": 0.23986,
        "before": [(0.027, 2, 0.24332, 18.008), (0.050, 4, 0.18246, 11.378), (0.125, 14, 0.12237, 8.896), (0.300, 34, 0.08799, 8.832), (0.750, 86, 0.06233, 8.770), (1.879, 215, 0.04433, 8.741), (4.730, 542, 0.03200, 8.735), (11.909, 1363, 0.02358, 8.737), (30.004, 3431, 0.01798, 8.745)],
        "after": [(0.026, 2, 0.24405, 17.580), (0.052, 5, 0.17648, 10.825), (0.121, 14, 0.12135, 8.661), (0.300, 35, 0.08629, 8.572), (0.749, 88, 0.06113, 8.516), (1.883, 222, 0.04346, 8.482), (4.733, 559, 0.03142, 8.467), (11.915, 1408, 0.02324, 8.465), (30.002, 3542, 0.01778, 8.469)],
        "pt": [(0.019, 11, 0.16780, 1.751), (0.047, 27, 0.12564, 1.741), (0.118, 68, 0.09044, 1.732), (0.296, 172, 0.06370, 1.726), (0.746, 434, 0.04490, 1.719), (1.876, 1095, 0.03208, 1.714), (4.728, 2764, 0.02333, 1.711), (11.908, 6972, 0.01740, 1.708), (30.001, 17567, 0.01338, 1.708)],
    },
    "bedroom--own": {
        "reference_mean": 0.11680,
        "before": [(0.027, 2, 0.25230, 18.453), (0.051, 4, 0.21281, 11.548), (0.128, 14, 0.15807, 9.107), (0.306, 34, 0.12013, 9.012), (0.761, 85, 0.08701, 8.957), (1.906, 214, 0.06100, 8.929), (4.772, 535, 0.04290, 8.919), (11.965, 1342, 0.03070, 8.919), (30.004, 3360, 0.02244, 8.928)],
        "after": [(0.026, 2, 0.25170, 17.782), (0.053, 5, 0.20764, 11.043), (0.126, 14, 0.15738, 9.015), (0.308, 35, 0.11833, 8.807), (0.764, 88, 0.08564, 8.729), (1.903, 219, 0.06008, 8.690), (4.774, 551, 0.04227, 8.665), (11.963, 1382, 0.03032, 8.656), (30.004, 3464, 0.02217, 8.662)],
        "pt": [(0.020, 10, 0.18609, 1.922), (0.049, 26, 0.14593, 1.903), (0.121, 64, 0.11103, 1.897), (0.303, 160, 0.08017, 1.894), (0.759, 402, 0.05610, 1.890), (1.901, 1006, 0.03948, 1.890), (4.769, 2512, 0.02836, 1.899), (11.960, 6298, 0.02063, 1.899), (30.001, 15792, 0.01538, 1.900)],
    },
    "sponza--own": {
        "reference_mean": 0.08259,
        "before": [(0.042, 2, 0.14593, 28.094), (0.077, 4, 0.10846, 17.409), (0.167, 12, 0.07772, 13.942), (0.395, 28, 0.05812, 13.850), (0.931, 68, 0.04408, 13.787), (2.221, 162, 0.03366, 13.752), (5.277, 384, 0.02591, 13.741), (12.585, 916, 0.02005, 13.740), (30.000, 2182, 0.01578, 13.749)],
        "after": [(0.039, 2, 0.13970, 26.096), (0.078, 5, 0.10139, 16.209), (0.169, 13, 0.07307, 13.035), (0.394, 30, 0.05512, 12.916), (0.932, 72, 0.04199, 12.848), (2.210, 172, 0.03222, 12.814), (5.281, 413, 0.02476, 12.788), (12.579, 984, 0.01924, 12.783), (30.009, 2346, 0.01520, 12.789)],
        "pt": [(0.031, 9, 0.10966, 3.445), (0.069, 20, 0.08543, 3.448), (0.164, 48, 0.06490, 3.454), (0.389, 112, 0.04905, 3.455), (0.928, 269, 0.03683, 3.451), (2.211, 642, 0.02773, 3.447), (5.274, 1532, 0.02100, 3.443), (12.577, 3654, 0.01611, 3.442), (30.002, 8718, 0.01264, 3.441)],
    },
    "bistro-exterior--own": {
        "reference_mean": 0.04772,
        "before": [(0.100, 2, 0.13602, 67.152), (0.185, 4, 0.12055, 41.906), (0.335, 10, 0.10191, 33.463), (0.701, 21, 0.08168, 33.359), (1.496, 45, 0.06136, 33.253), (3.155, 95, 0.04508, 33.210), (6.684, 202, 0.03323, 33.171), (14.160, 427, 0.02499, 33.162), (30.008, 904, 0.01924, 33.176)],
        "after": [(0.086, 2, 0.13589, 57.646), (0.174, 5, 0.11824, 36.128), (0.349, 12, 0.09687, 29.053), (0.724, 25, 0.07669, 28.972), (1.499, 52, 0.05762, 28.835), (3.150, 110, 0.04239, 28.764), (6.678, 232, 0.03135, 28.723), (14.146, 493, 0.02369, 28.694), (30.005, 1045, 0.01836, 28.713)],
        "pt": [(0.077, 8, 0.11011, 9.111), (0.160, 18, 0.09178, 9.155), (0.334, 36, 0.07205, 9.161), (0.701, 76, 0.05408, 9.161), (1.484, 162, 0.03997, 9.158), (3.145, 344, 0.02988, 9.155), (6.668, 728, 0.02256, 9.153), (14.145, 1545, 0.01720, 9.155), (30.007, 3276, 0.01332, 9.161)],
    },
}


def progress_curve(cell, width=430, height=210):
    arms = {arm: [{"seconds": s, "flip": {"mean": f}} for s, _, f, _ in points]
            for arm, points in PROGRESS[cell].items() if arm != "reference_mean"}
    return convergence_chart(arms, width=width, height=height)


def progress_section():
    rows = []
    for cell, entry in PROGRESS.items():
        before, after = entry["before"][-1], entry["after"][-1]
        pt = entry["pt"][-1] if "pt" in entry else None
        rows.append((cell, before, after, pt))
    rows.sort(key=lambda r: -(r[1][2] / r[2][2] - 1))

    body = "".join(
        f"<tr><td class='cell'>{cell}</td>"
        f"<td class='num'>{before[3]:.3f}</td><td class='num'>{after[3]:.3f}</td>"
        f"<td class='num'><span class='pill {'win' if after[3] < before[3] * 0.995 else 'tie'}'>"
        f"{(after[3] / before[3] - 1) * 100:+.1f} %</span></td>"
        f"<td class='num'>{before[2]:.5f}</td><td class='num'>{after[2]:.5f}</td>"
        f"<td class='num'><span class='pill {'win' if after[2] < before[2] * 0.995 else 'tie'}'>"
        f"{(before[2] / after[2] - 1) * 100:+.1f} %</span></td>"
        + (f"<td class='num'>{pt[2]:.5f}</td>"
           f"<td class='num'><span class='pill {verdict_class(pt[2] / after[2] * 100)}'>"
           f"{pt[2] / after[2] * 100:.0f} %</span></td>" if pt else
           "<td class='num muted'>—</td><td class='num muted'>—</td>")
        + "</tr>"
        for cell, before, after, pt in rows)

    legend = "".join(
        f"<span><i style='background:{ARM_COLOR[a]}'></i>{ARM_LABEL[a]}</span>"
        for a in ("before", "after", "pt"))
    curves = "".join(
        f"<figure class='curve-block'><figcaption>{cell}</figcaption>{progress_curve(cell)}</figure>"
        for cell, _, _, _ in rows)

    return f"""
<section id="postep">
  <h2>Co dały optymalizacje</h2>
  <p>Wszystkie osiem komórek zmierzone od nowa na jednym pliku wykonywalnym, tym samym budżetem
  i tym samym obrazem odniesienia. Kolumna <b>przed</b> to dwa shadery przywrócone do stanu
  wyjściowego, <b>po</b> to stan obecny — kod C++ jest w obu przypadkach ten sam, więc różnica to
  dokładnie cztery zmiany zachowania: skrócenie promienia naprowadzanego do bryły woksela, znaki
  ścian zgodne z tym, z czego zbudowano kwadraty sferyczne, konserwatywne wypalanie geometrii
  i dwufazowy promień naprowadzany.</p>
  <div class="scroll">
  <table>
    <thead><tr><th rowspan="2">komórka</th><th colspan="3">ms/klatkę</th>
      <th colspan="3">FLIP po 30 s</th><th colspan="2">wobec śledzenia ścieżek</th></tr>
      <tr><th>przed</th><th>po</th><th>zmiana</th><th>przed</th><th>po</th><th>zysk</th>
      <th>PT</th><th>VXPG/PT</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
  </div>
  <p class="note">Zysk jakości jest w całości zyskiem tempa: żadna z tych zmian nie dotyka
  estymatora, więc błąd na próbkę zostaje ten sam, a spada tylko koszt klatki. Dlatego kolumny
  <b>zmiana</b> i <b>zysk</b> idą zgodnie, a średnie obrazów pozostają na swoich miejscach.</p>
  <h3>Krzywe zbieżności</h3>
  <div class="legend">{legend}</div>
  <div class="curve-grid">{curves}</div>
</section>"""


def previous_section(data, last):
    """ms/frame of this run against the previous reconnaissance, cell by cell."""
    rows = []
    for cell in data:
        name = cell["cell"]
        before = PREVIOUS_RECON.get(name)
        arms = last.get(name, {})
        pt = arms.get("PT", {}).get("ms", {}).get("mean")
        vx = arms.get("VXPG", {}).get("ms", {}).get("mean")
        if not before or pt is None or vx is None:
            continue
        rows.append({"cell": name, "pt_before": before["PT"], "pt_after": pt,
                     "vx_before": before["VXPG"], "vx_after": vx,
                     "ratio_before": before["VXPG"] / before["PT"], "ratio_after": vx / pt,
                     "gain": (1 - vx / before["VXPG"]) * 100})
    if not rows:
        return ""
    rows.sort(key=lambda r: -r["gain"])

    def pill(value):
        klass = "tie" if abs(value) <= 0.5 else ("win" if value < 0 else "loss")
        return f"<span class='pill {klass}'>{value:+.1f} %</span>"

    # The PT column is a control, not a contest: a plain number, because pilling it red
    # would read as "path tracing got worse" when what it measures is the margin.
    body = "".join(
        f"<tr><td class='cell'>{r['cell']}</td>"
        f"<td class='num muted'>{r['pt_before']:.3f}</td><td class='num'>{r['pt_after']:.3f}</td>"
        f"<td class='num muted'>{(r['pt_after'] / r['pt_before'] - 1) * 100:+.1f} %</td>"
        f"<td class='num muted'>{r['vx_before']:.3f}</td><td class='num'>{r['vx_after']:.3f}</td>"
        f"<td class='num'>{pill((r['vx_after'] / r['vx_before'] - 1) * 100)}</td>"
        f"<td class='num muted'>{r['ratio_before']:.2f}×</td>"
        f"<td class='num strong'>{r['ratio_after']:.2f}×</td></tr>"
        for r in rows)

    mean_vxpg = sum(r["vx_after"] / r["vx_before"] for r in rows) / len(rows)
    mean_pt = sum(r["pt_after"] / r["pt_before"] for r in rows) / len(rows)
    best = rows[0]
    worst = rows[-1]

    return f"""
<section id="poprzedni">
  <h2>Wobec poprzedniego rekonesansu ({PREVIOUS_LABEL})</h2>
  <p>Ten sam manifest, ten sam budżet i ta sama maszyna. Zmieniło się to, co powstało
  pomiędzy przebiegami: łańcuch naprowadzania w osobnym rozesłaniu, kąt bryłowy woksela
  z iloczynu potrójnego, wspólny pomocnik kwadratu sferycznego i bezwarunkowe wykonanie
  łańcucha. Zmieniła się też kompilacja — poprzedni przebieg był diagnostyczny, ten jest
  wynikowy — więc <b>kolumna PT jest kontrolą</b>: śledzenia ścieżek nie dotyka żadna
  z tych zmian, a zatem to, o ile ona się przesuwa, ogranicza z góry łączny wkład zmiany
  kompilacji i powtarzalności stanowiska. Zysk VXPG liczy się jako realny dopiero powyżej
  tego marginesu.</p>
  <div class="facts">
    <div><span>VXPG średnio</span><b>{(mean_vxpg - 1) * 100:+.1f} %</b></div>
    <div><span>PT średnio</span><b>{(mean_pt - 1) * 100:+.1f} %</b></div>
    <div><span>Największy zysk</span><b>{best['cell']} {best['gain']:.1f} %</b></div>
    <div><span>Najmniejszy zysk</span><b>{worst['cell']} {worst['gain']:.1f} %</b></div>
  </div>
  <div class="scroll">
  <table>
    <thead><tr><th rowspan="2">komórka</th><th colspan="3">PT ms/klatkę</th>
      <th colspan="3">VXPG ms/klatkę</th><th colspan="2">VXPG/PT</th></tr>
      <tr><th>{PREVIOUS_LABEL}</th><th>teraz</th><th>zmiana</th>
      <th>{PREVIOUS_LABEL}</th><th>teraz</th><th>zmiana</th>
      <th>{PREVIOUS_LABEL}</th><th>teraz</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
  </div>
  <p class="note">Siedem komórek trzyma kolumnę PT w przedziale 0,0…+1,4 %, co wyznacza
  margines. Ósma, <b>bedroom, ma tam −6,5 %</b> i to nie jest szum: zasłony w tej scenie
  to jedyny (obok żaluzji w kitchen) materiał o stałej przezroczystości bez tekstury,
  a przejście na regułę „alfa wycina tylko tam, gdzie tekstura jest w pełni przezroczysta"
  zdjęło je ze ścieżki any-hit i uczyniło zwykłą geometrią nieprzezroczystą. Zasłona
  stoi dokładnie tam, skąd przychodzi całe światło sceny, więc trafiał w nią każdy
  promień cienia. FLIP nie jest tu zestawiony, bo ten przebieg punktowano wobec świeżo
  policzonego obrazu odniesienia — porównywalne są koszty klatki, nie piąte miejsce po
  przecinku błędu. Zysk jakości i tak jest zyskiem tempa: żadna ze zmian nie dotyka
  estymatora, więc błąd na próbkę zostaje ten sam, a spada koszt klatki.</p>
</section>"""


def build_page(data, manifest, commands, grid_data, ratio_data):
    last = {}
    for cell in data:
        last[cell["cell"]] = {label: arm[-1] for label, arm in cell["arms"].items()}

    html = [HEAD]

    # --- run conditions ---------------------------------------------------
    defaults = manifest["renderDefaults"]
    html.append(f"""
<section id="warunki">
  <h2>Warunki przebiegu</h2>
  <div class="facts">
    <div><span>Rozdzielczość</span><b>{defaults['width']}×{defaults['height']}</b></div>
    <div><span>Próbek na piksel</span><b>{defaults['spp']}</b></div>
    <div><span>Odbić</span><b>{defaults['bounces']}</b></div>
    <div><span>Niebo oświetla</span><b>{'tak' if defaults['skyLighting'] else 'nie'}</b></div>
    <div><span>Tylko pośrednie</span><b>{'tak' if defaults['indirectOnly'] else 'nie'}</b></div>
    <div><span>Siatka wokseli</span><b>64³</b></div>
    <div><span>Obraz odniesienia</span><b>600 s, PT</b></div>
    <div><span>Budżet pomiaru</span><b>30 s × 2 obrazy</b></div>
    <div><span>Punkty kontrolne</span><b>6, logarytmiczne</b></div>
    <div><span>Kompilacja</span><b>wynikowa (Release)</b></div>
    <div><span>Metryka</span><b>FLIP, tryb LDR</b></div>
    <div><span>Komórek</span><b>{len(data)}</b></div>
  </div>
  <p class="note">Mierzymy kompilację wynikową, bo pomiar ma opisywać program, który
  faktycznie działa; kompilacja diagnostyczna zostaje wyłącznie do pracy nad poprawnością
  z warstwą sprawdzającą, co jest osobną osią (5,1 ms → 1118 ms/klatkę) i nie daje
  czasów. Poprzedni rekonans szedł na kompilacji diagnostycznej — przejście nic nie
  psuje, bo oba warianty zmierzono jako zgodne w granicach 1 % dla obu technik (R9):
  klatka jest związana pracą układu graficznego, nie kodem gospodarza. Warstwa
  sprawdzająca wyłączona. Liczby są <b>orientacyjne</b> — rekonans ustala kolejność
  komórek, nie dostarcza wyników do pracy.</p>
</section>""")

    # --- performance ------------------------------------------------------
    rows = []
    for cell in data:
        name = cell["cell"]
        arms = last[name]
        pt = arms.get("PT", {}).get("ms", {}).get("mean")
        vx = arms.get("VXPG", {}).get("ms", {}).get("mean")
        rows.append((name, pt, vx))
    rows.sort(key=lambda r: r[1] or 0)

    body = "".join(
        f"<tr><td class='cell'>{n}</td>"
        f"<td class='num'>{pt:.3f}</td><td class='num'>{vx:.3f}</td>"
        f"<td class='num strong'>{vx / pt:.2f}×</td></tr>"
        for n, pt, vx in rows)

    bars = "".join(
        f"<figure class='bar-block'><figcaption>{n}</figcaption>"
        + bar_chart([(ARM_LABEL['PT'], pt, ARM_COLOR['PT']),
                     (ARM_LABEL['VXPG'], vx, ARM_COLOR['VXPG'])]) + "</figure>"
        for n, pt, vx in rows)

    html.append(f"""
<section id="wydajnosc">
  <h2>Wydajność</h2>
  <p>Milisekundy na klatkę, średnia z okna pomiarowego ostatniego punktu kontrolnego.
  Kolumna <b>VXPG/PT</b> to mnożnik kosztu, jaki naprowadzanie nakłada na klatkę. Rozbiór
  tego mnożnika na maszynerię i sam rozkład jest w sekcji <a href="#krok2">Koszt naprowadzania</a><a href="#postep">Optymalizacje</a>.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>komórka</th><th>PT</th><th>VXPG</th><th>VXPG/PT</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
  </div>
  <div class="bar-grid">{bars}</div>
</section>""")

    html.append(previous_section(data, last))

    # --- equal sample -----------------------------------------------------
    sample_rows = []
    for cell in data:
        arms = cell["arms"]
        if "PT" not in arms or "VXPG" not in arms:
            continue
        pt_last, vx_last = arms["PT"][-1], arms["VXPG"][-1]
        # The nearest PT checkpoint by frame count, so both arms are compared after a
        # similar number of samples rather than a similar amount of time. The checkpoints
        # are placed by time, so the match is approximate and the mismatch is shown: a PT
        # column above the VXPG frame count flatters PT, below it flatters the guide.
        near = min(arms["PT"], key=lambda c: abs(c["frames"] - vx_last["frames"]))
        sample_rows.append({
            "cell": cell["cell"],
            "cost": vx_last["ms"]["mean"] / pt_last["ms"]["mean"],
            "pt_frames": near["frames"], "vx_frames": vx_last["frames"],
            "drift": (near["frames"] - vx_last["frames"]) / vx_last["frames"] * 100,
            "eq_time": pt_last["flip"]["mean"] / vx_last["flip"]["mean"] * 100,
            "eq_sample": near["flip"]["mean"] / vx_last["flip"]["mean"] * 100})
    sample_rows.sort(key=lambda r: -r["eq_sample"])
    sbody = "".join(
        f"<tr><td class='cell'>{r['cell']}</td>"
        f"<td class='num'>{r['cost']:.2f}×</td>"
        f"<td class='num'>{r['vx_frames']:,}</td>".replace(",", " ")
        + f"<td class='num'>{r['pt_frames']:,}</td>".replace(",", " ")
        + f"<td class='num muted'>{r['drift']:+.0f} %</td>"
        f"<td class='num'><span class='pill {verdict_class(r['eq_sample'])}'>"
        f"{r['eq_sample']:.0f} %</span></td>"
        f"<td class='num'><span class='pill {verdict_class(r['eq_time'])}'>"
        f"{r['eq_time']:.0f} %</span></td></tr>"
        for r in sample_rows)
    html.append(f"""
<section id="probka">
  <h2>Na próbkę wobec na czas</h2>
  <p>To jest rozstrzygnięcie, którego sama tabela równoczasowa nie daje: czy naprowadzanie
  <b>nie działa</b>, czy <b>działa, ale za drogo</b>. Kolumna „na próbkę" porównuje oba ramiona
  po zbliżonej liczbie klatek zamiast po zbliżonym czasie — powyżej 100 % pojedyncza próbka
  naprowadzana jest warta więcej niż próbka bez naprowadzania.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>komórka</th><th>koszt klatki</th><th>klatek VXPG</th><th>klatek PT</th>
      <th>rozjazd</th><th>na próbkę</th><th>na czas</th></tr></thead>
    <tbody>{sbody}</tbody>
  </table>
  </div>
  <p class="note">Punkty kontrolne są rozstawiane po czasie, nie po liczbie klatek, więc
  dopasowanie jest przybliżone — kolumna <b>rozjazd</b> podaje, o ile klatek PT wypadło obok.
  Rozjazd dodatni działa na korzyść śledzenia ścieżek, ujemny na korzyść naprowadzania.
  Przy Bistro rozjazd wynosi +28 %, więc tamtejsze 94 % jest zaniżone.</p>
</section>""")

    # --- equal time -------------------------------------------------------
    quality_rows = []
    for cell in data:
        name = cell["cell"]
        arms = last[name]
        pt = arms.get("PT")
        if not pt:
            continue
        base = pt["flip"]["mean"]
        entry = {"cell": name, "pt": base, "pt_ci": pt["flip"]["ci95"]}
        for label in ("VXPG",):
            if label in arms:
                entry[label] = arms[label]["flip"]["mean"]
                entry[label + "_ci"] = arms[label]["flip"]["ci95"]
                entry[label + "_ratio"] = base / arms[label]["flip"]["mean"] * 100
        quality_rows.append(entry)
    quality_rows.sort(key=lambda r: -r.get("VXPG_ratio", 0))

    qbody = "".join(
        f"<tr><td class='cell'>{r['cell']}</td>"
        f"<td class='num'>{r['pt']:.5f}</td>"
        f"<td class='num'>{r.get('VXPG', float('nan')):.5f}</td>"
        f"<td class='num'><span class='pill {verdict_class(r.get('VXPG_ratio', 0))}'>"
        f"{r.get('VXPG_ratio', 0):.0f} %</span></td></tr>"
        for r in quality_rows)

    qbars = "".join(
        f"<figure class='bar-block'><figcaption>{r['cell']}</figcaption>"
        + bar_chart([(ARM_LABEL['VXPG'], r.get('VXPG_ratio', 0), ARM_COLOR['VXPG'])],
                    unit=" %", row_height=27) + "</figure>"
        for r in quality_rows)

    html.append(f"""
<section id="rownoczasowo">
  <h2>Wynik przy równym czasie</h2>
  <p>FLIP wobec obrazu odniesienia po tym samym budżecie 30 s — mniej znaczy bliżej
  odniesienia. Kolumny procentowe to <b>błąd śledzenia ścieżek podzielony przez błąd
  ramienia</b>: powyżej 100 % naprowadzanie wygrywa przy równym czasie, poniżej przegrywa.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>komórka</th><th>PT</th><th>VXPG</th><th>VXPG wobec PT</th></tr></thead>
    <tbody>{qbody}</tbody>
  </table>
  </div>
  <div class="bar-grid">{qbars}</div>
</section>""")

    # --- convergence ------------------------------------------------------
    curves = "".join(
        f"<figure class='curve-block'><figcaption>{cell['cell']}</figcaption>"
        + convergence_chart(cell["arms"]) + "</figure>" for cell in data)
    html.append(f"""
<section id="zbieznosc">
  <h2>Zbieżność</h2>
  <p>FLIP wobec czasu akumulacji, obie osie logarytmiczne, sześć punktów kontrolnych na
  przebieg. Linia niżej to obraz bliższy odniesieniu; krzyżujące się linie oznaczają, że
  przewaga zależy od budżetu.</p>
  <div class="legend">
    {''.join(f'<span><i style="background:{ARM_COLOR[k]}"></i>{v}</span>' for k, v in ARM_LABEL.items())}
  </div>
  <div class="curve-grid">{curves}</div>
</section>""")

    # --- images -----------------------------------------------------------
    blocks = []
    for cell in data:
        name = cell["cell"]
        tiles = [f"<figure><img src='{thumbnail(cell['reference'])}' alt='odniesienie {name}' "
                 f"loading='lazy'><figcaption>odniesienie · 600 s</figcaption></figure>"]
        for label, _, _ in ARMS:
            if label not in cell["arms"]:
                continue
            checkpoint = cell["arms"][label][-1]
            tiles.append(
                f"<figure><img src='{thumbnail(checkpoint['path'])}' alt='{label} {name}' "
                f"loading='lazy'><figcaption>{ARM_LABEL[label]} · {checkpoint['seconds']:.0f} s · "
                f"{checkpoint['frames']} klatek</figcaption></figure>")
        lit, occupied, largest = CLUSTERS.get(name, (0, 0, 0.0))
        blocks.append(f"""
<article class="cell-block">
  <h3>{name}</h3>
  <p class="meta">stan <b>{cell['state']}</b> · ekspozycja {cell['exposure']} ·
     kontrast {cell['contrast']} · {lit} oświetlonych wokseli ·
     {occupied}/32 zajętych skupisk, największe {largest} %</p>
  <div class="tiles">{''.join(tiles)}</div>
</article>""")
    html.append(f"""
<section id="obrazy">
  <h2>Obrazy</h2>
  <p>Po lewej obraz odniesienia z 600 s śledzenia ścieżek, dalej każde ramię po 30 s.
  Miniatury 1280 px; różnice szumu na pełnej rozdzielczości są jeszcze wyraźniejsze.</p>
  {''.join(blocks)}
</section>""")

    html.append(cost_section())
    html.append(progress_section())
    html.append(grid_section(grid_data))
    html.append(ratio_section(ratio_data))

    # --- grid context -----------------------------------------------------
    census_rows = "".join(
        f"<tr><td class='cell'>{scene}</td>"
        + "".join(f"<td class='num{' over' if v > LEAF_CAP else ''}'>{v:,}</td>".replace(",", " ")
                  for v in counts)
        + "".join(f"<td class='num'>{MAX_EXTENT[scene] / d:.3f}</td>" for d in (32, 64, 128, 256))
        + "</tr>"
        for scene, counts in sorted(CENSUS.items(), key=lambda kv: MAX_EXTENT[kv[0]]))

    # Cost per rung comes from THIS run's grid captures wherever they exist, so the table
    # cannot drift away from the frame costs above it; GRID_COST is only the fallback for a
    # scene the grid phase did not cover.
    measured_cost = {}
    for entry in grid_data:
        scene_id = entry["scene"]["id"]
        # A scene with a K2 rig has two cells on the same geometry; the 'own' one is the
        # scene as authored, so that is the row.
        if scene_id in measured_cost and not entry["cell"].endswith("--own"):
            continue
        by_dim = {rung["dim"]: rung["ms"]["mean"] for rung in entry["rungs"]}
        measured_cost[scene_id] = [by_dim.get(d) for d in (32, 64, 128, 256)]

    cost_source = {scene: measured_cost.get(scene) or costs for scene, costs in GRID_COST.items()}
    cost_rows = "".join(
        f"<tr><td class='cell'>{scene}</td>"
        + "".join(f"<td class='num'>{v:.3f}</td>" if v else "<td class='num muted'>—</td>"
                  for v in costs)
        + f"<td class='num strong'>{(max(v for v in costs if v) / costs[0] - 1) * 100:+.1f} %</td></tr>"
        for scene, costs in sorted(cost_source.items(), key=lambda kv: MAX_EXTENT[kv[0]]))

    html.append(f"""
<section id="siatka">
  <h2>Kontekst siatki wokseli</h2>
  <p>Rekonans biegł na 64³ w każdej scenie. Ten spis mówi, co ta jedna liczba znaczy
  fizycznie w każdej z nich i gdzie leży sufit narzucony limitem 32 768 liści drzewa
  świateł (pola przekroczone zaznaczone).</p>
  <div class="scroll">
  <table>
    <thead>
      <tr><th rowspan="2">scena</th><th colspan="4">oświetlone woksele</th>
          <th colspan="4">krawędź woksela [m]</th></tr>
      <tr><th>32³</th><th>64³</th><th>128³</th><th>256³</th>
          <th>32³</th><th>64³</th><th>128³</th><th>256³</th></tr>
    </thead>
    <tbody>{census_rows}</tbody>
  </table>
  </div>
  <p>Koszt klatki wobec rozdzielczości, ramię naprowadzane, z tych samych przebiegów co
  sekcja K5 powyżej — nie z osobnego, starszego pomiaru. Puste pole =
  rozdzielczość przekracza limit liści, więc mierzyłaby limit, nie siatkę.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>scena</th><th>32³</th><th>64³</th><th>128³</th><th>256³</th>
      <th>32³ → sufit</th></tr></thead>
    <tbody>{cost_rows}</tbody>
  </table>
  </div>
</section>""")

    # --- verdict ----------------------------------------------------------
    verdict_rows = []
    for r in quality_rows:
        if "VXPG" not in r:
            continue
        gap = abs(r["pt"] - r["VXPG"])
        floor = NOISE_INFLATION * max(r["VXPG_ci"], r["pt_ci"])
        verdict_rows.append((r["cell"], r["VXPG_ratio"], gap, floor, gap > floor))
    vbody = "".join(
        f"<tr><td class='cell'>{n}</td>"
        f"<td class='num'><span class='pill {verdict_class(ratio)}'>{ratio:.0f} %</span></td>"
        f"<td class='num'>{gap:.5f}</td><td class='num'>{floor:.5f}</td>"
        f"<td>{'rozdzielcza' if ok else 'poniżej szumu'}</td></tr>"
        for n, ratio, gap, floor, ok in verdict_rows)
    html.append(f"""
<section id="werdykt">
  <h2>Werdykt na komórkę</h2>
  <p>Komórka zasługuje na czas kampanii, gdy odstęp między ramionami przekracza
  {NOISE_INFLATION}-krotność przedziału ufności liczonego wewnątrz przebiegu. Mnożnik jest
  tam dlatego, że przedział z obrazów jednego procesu zaniża rozrzut między osobnymi
  uruchomieniami 3–10× — a komórka niesłusznie zatrzymana kosztuje godziny.
  <b>Rozdzielcza nie znaczy korzystna</b>: kolumna procentowa mówi, w którą stronę.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>komórka</th><th>VXPG wobec PT</th><th>odstęp</th><th>próg szumu</th>
      <th>ocena</th></tr></thead>
    <tbody>{vbody}</tbody>
  </table>
  </div>
</section>""")

    # --- commands ---------------------------------------------------------
    html.append(f"""
<section id="komendy">
  <h2>Uruchomione komendy</h2>
  <pre><code>{commands}</code></pre>
</section>""")

    html.append("</main>")
    return "\n".join(html)


HEAD = """<title>Rekonans VXPG</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{
  --bg:#F6F4F0; --surface:#FFFFFF; --sunken:#EFEBE4;
  --ink:#191713; --ink-2:#4A453D; --muted:#7C756A; --rule:#DFD9CF;
  --arm-pt:#41628E; --arm-vxpg:#C07C10; --arm-base:#5C7A6B; --arm-nog:#8A5B4A;
  --win:#2F6B45; --win-bg:#DFEDE3; --tie:#7C756A; --tie-bg:#E8E4DC;
  --loss:#9A3A2E; --loss-bg:#F3E0DB;
  --shadow:0 1px 2px rgba(25,23,19,.06),0 8px 24px -12px rgba(25,23,19,.18);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#14150F; --surface:#1D1E17; --sunken:#25261E;
  --ink:#EEEAE0; --ink-2:#C3BDB0; --muted:#948D80; --rule:#33352A;
  --arm-pt:#8AA9D6; --arm-vxpg:#E3A83F; --arm-base:#8FB8A2; --arm-nog:#C98F79;
  --win:#8FCFA5; --win-bg:#1E3226; --tie:#948D80; --tie-bg:#2A2C22;
  --loss:#E8A092; --loss-bg:#3A2320;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --bg:#14150F; --surface:#1D1E17; --sunken:#25261E;
  --ink:#EEEAE0; --ink-2:#C3BDB0; --muted:#948D80; --rule:#33352A;
  --arm-pt:#8AA9D6; --arm-vxpg:#E3A83F; --arm-base:#8FB8A2; --arm-nog:#C98F79;
  --win:#8FCFA5; --win-bg:#1E3226; --tie:#948D80; --tie-bg:#2A2C22;
  --loss:#E8A092; --loss-bg:#3A2320;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:17px;line-height:1.62;
  -webkit-font-smoothing:antialiased}
main{max-width:1180px;margin:0 auto;padding:0 24px 96px;
  display:flex;flex-direction:column;gap:64px}
h1,h2,h3{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;text-wrap:balance;
  letter-spacing:-.015em;margin:0}
header{padding:72px 24px 40px;max-width:1180px;margin:0 auto;border-bottom:1px solid var(--rule)}
header .eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);margin:0 0 14px}
h1{font-size:clamp(34px,5.2vw,58px);font-weight:700;line-height:1.04}
header p.lede{font-size:20px;color:var(--ink-2);max-width:64ch;margin:18px 0 0}
h2{font-size:clamp(23px,2.6vw,30px);font-weight:600;padding-bottom:10px;
  border-bottom:2px solid var(--ink);display:inline-block;margin-bottom:6px}
h3{font-size:18px;font-weight:600;margin-bottom:2px}
section>p{max-width:70ch;color:var(--ink-2);margin:10px 0 22px}
p.note{background:var(--sunken);border-left:3px solid var(--arm-vxpg);
  padding:14px 18px;border-radius:0 6px 6px 0;font-size:15.5px;max-width:76ch}
nav{position:sticky;top:0;z-index:9;background:color-mix(in srgb,var(--bg) 92%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);padding:10px 24px}
nav div{max-width:1180px;margin:0 auto;display:flex;gap:20px;flex-wrap:wrap;
  font-family:"IBM Plex Mono",monospace;font-size:12.5px;letter-spacing:.03em}
nav a{color:var(--muted);text-decoration:none;padding:2px 0;border-bottom:1px solid transparent}
nav a:hover,nav a:focus-visible{color:var(--ink);border-bottom-color:var(--arm-vxpg);outline:none}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:8px;overflow:hidden}
.facts>div{background:var(--surface);padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.facts span{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted)}
.facts b{font-family:Archivo,sans-serif;font-size:17px;font-weight:600}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface);
  box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{font-family:Archivo,sans-serif;font-size:11.5px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--muted);font-weight:600;
  background:var(--sunken);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
tbody tr.best td{background:var(--win-bg)}
tbody tr.best td.cell,tbody tr.best td.num{font-weight:600}
td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;text-align:right}
td.cell{font-family:"IBM Plex Mono",monospace;font-size:13px}
td.strong{font-weight:600;color:var(--ink)}
td.muted{color:var(--muted)}
td.over{color:var(--loss);font-weight:600}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-weight:600;font-size:13px}
.pill.win{background:var(--win-bg);color:var(--win)}
.pill.tie{background:var(--tie-bg);color:var(--tie)}
.pill.loss{background:var(--loss-bg);color:var(--loss)}
.bar-grid,.curve-grid{display:grid;gap:20px;margin-top:24px;
  grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.bar-block,.curve-block{background:var(--surface);border:1px solid var(--rule);
  border-radius:8px;padding:14px 16px;margin:0}
figcaption{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);
  margin-bottom:8px}
svg.bars,svg.curve{width:100%;height:auto;display:block}
.bar-label{font-family:Archivo,sans-serif;font-size:12.5px;fill:var(--ink-2)}
.bar-value{font-family:"IBM Plex Mono",monospace;font-size:12px;fill:var(--muted)}
.axis{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--muted)}
.grid{stroke:var(--rule);stroke-width:1}
.legend{display:flex;gap:22px;flex-wrap:wrap;font-size:14px;color:var(--ink-2);margin-bottom:4px}
.legend span{display:flex;align-items:center;gap:7px}
.legend i{width:16px;height:3px;border-radius:2px;display:inline-block}
.cell-block{margin:34px 0 0;padding-top:22px;border-top:1px solid var(--rule)}
.cell-block h3{font-family:"IBM Plex Mono",monospace;font-size:15px}
p.meta{font-size:13.5px;color:var(--muted);margin:4px 0 14px;max-width:none}
.tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.tiles figure{margin:0}
.tiles img{width:100%;height:auto;display:block;border-radius:6px;border:1px solid var(--rule)}
.tiles figcaption{margin:6px 0 0}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px;
  overflow-x:auto;font-size:12.5px;line-height:1.6}
code{font-family:"IBM Plex Mono",monospace}
@media (max-width:640px){body{font-size:16px}main{padding:0 16px 64px;gap:48px}
  header{padding:48px 16px 28px}}
</style>
<header>
  <p class="eyebrow">Rekonans · 27.08.2026 · osiem komórek · trzy ramiona</p>
  <h1>Rekonans VXPG</h1>
  <p class="lede">Który z ośmiu układów scena–światło zasługuje na czas kampanii, jaki
  koszt klatki nakłada naprowadzanie i gdzie ten koszt się zwraca. Liczby są
  orientacyjne — rekonans ustala kolejność, nie dostarcza wyników do pracy.</p>
</header>
<nav><div>
  <a href="#warunki">Warunki</a><a href="#wydajnosc">Wydajność</a><a href="#poprzedni">Wobec poprzedniego</a>
  <a href="#probka">Próbka vs czas</a><a href="#rownoczasowo">Równy czas</a><a href="#zbieznosc">Zbieżność</a>
  <a href="#obrazy">Obrazy</a><a href="#krok2">Koszt naprowadzania</a><a href="#k5">K5 siatka</a><a href="#proporcja">Proporcja 5:1</a><a href="#siatka">Spis siatek</a>
  <a href="#werdykt">Werdykt</a><a href="#komendy">Komendy</a>
</div></nav>
<main>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tools/recon-manifest.json")
    parser.add_argument("--run", default="SavedUserData/Screenshots/recon")
    parser.add_argument("--references", default="SavedUserData/Screenshots/recon-refs")
    parser.add_argument("--commands", default="SavedUserData/recon-log/commands.md")
    parser.add_argument("--grid", default="SavedUserData/Screenshots/recon-grid")
    parser.add_argument("--ratio", default="SavedUserData/Screenshots/recon-ratio")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    data = gather(manifest, under(args.run), under(args.references))
    if not data:
        sys.exit("no scored cells — is the run finished?")

    commands_path = under(args.commands)
    commands = commands_path.read_text(encoding="utf-8") if commands_path.exists() else ""
    commands = commands.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    grid_data = gather_grid(manifest, under(args.grid), under(args.references))
    ratio_data = gather_ratio(manifest, under(args.ratio), under(args.references))
    print(f"scored {len(grid_data)} grid cells, {len(ratio_data)} ratio cells")

    page = build_page(data, manifest, commands, grid_data, ratio_data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"written: {out}  ({out.stat().st_size / 1e6:.2f} MB, {len(data)} cells)")


if __name__ == "__main__":
    main()
