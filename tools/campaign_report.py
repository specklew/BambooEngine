"""Aggregate a finished (or half-finished) campaign into readable tables.

Reads every bench.json the campaign wrote and answers three questions, in this order,
because they are what the numbers are actually for:

1. Per scene+state, which lever combination wins — and by how much against the plain
   build, at equal wall-clock time.
2. Across all states, which levers hold up. A lever that wins on one camera and loses
   on another is a scene-dependent trick, not an optimization; the per-lever roll-up
   is what tells them apart.
3. Path tracing against the guided integrator, per state, on the same reference.

Two spreads are reported side by side and they are NOT the same thing:
  * Q1/Q2/Q3 of the FLIP error MAP — how the error sits across the picture, averaged
    over the runs.
  * ci95 on the mean — how repeatable the arm was across its runs.
A tight ci95 with a wide Q1..Q3 means a reliable measurement of an uneven image.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT = REPO / "Raytracer" / "SavedUserData" / "Screenshots" / "campaign"


def load_groups(root):
    for report in sorted(root.glob("g/*/*/*/bench.json")):
        data = json.loads(report.read_text())
        scene = report.parts[-4]
        for group in data.get("groups", []):
            group["scene"] = scene
            yield group


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


def row(group):
    metric, mse = group["metric"], group["mse"]
    q1 = group.get("flipQ1", {}).get("mean")
    q2 = group.get("flipMedian", {}).get("mean")
    q3 = group.get("flipQ3", {}).get("mean")
    quartiles = " | ".join(fmt(v, 5) if v is not None else "-" for v in (q1, q2, q3))
    return (f"| {group['levers']} | {group['frames']:.0f} | {group['meanFrameMs']:.2f} | "
            f"{fmt(metric['mean'])} | +-{fmt(metric['ci95'])} | {quartiles} | {fmt(mse['mean'])} |")


def main():
    parser = argparse.ArgumentParser(description="Summarise a lever campaign.")
    parser.add_argument("--root", default=str(DEFAULT))
    parser.add_argument("--top", type=int, default=5, help="arms to show per state")
    args = parser.parse_args()

    root = Path(args.root)
    groups = list(load_groups(root))
    if not groups:
        print(f"No reports under {root}")
        return

    by_state = defaultdict(list)
    for group in groups:
        by_state[(group["scene"], group["state"], group["technique"])].append(group)

    print(f"# Campaign: {len(by_state)} scene/state/technique blocks, {len(groups)} arms\n")

    # 1. Per block, best arms and the plain build for comparison.
    for (scene, state, technique), arms in sorted(by_state.items()):
        arms.sort(key=lambda g: g["metric"]["mean"])
        plain = next((g for g in arms if g["levers"] == "none"), None)
        print(f"\n## {scene} / {state} / {technique}")
        print("\n| levers | frames | ms/frame | FLIP mean | ci95 | Q1 | Q2 | Q3 | MSE |")
        print("|" + "---|" * 9)
        shown = arms[: args.top]
        if plain is not None and plain not in shown:
            shown.append(plain)
        for group in shown:
            print(row(group))
        if plain is not None and arms[0] is not plain:
            gain = (plain["metric"]["mean"] / arms[0]["metric"]["mean"] - 1.0) * 100.0
            print(f"\nBest arm beats the plain build by {gain:.1f}% on FLIP mean "
                  f"({arms[0]['levers']}).")

    # 2. Per lever set across every state: mean ratio against that state's plain build.
    ratios = defaultdict(list)
    for (scene, state, technique), arms in by_state.items():
        plain = next((g for g in arms if g["levers"] == "none"), None)
        if plain is None:
            continue
        for group in arms:
            ratios[(technique, group["levers"])].append(
                plain["metric"]["mean"] / group["metric"]["mean"])

    print("\n\n## Levers across every state (ratio to the plain build; >1 is better)\n")
    print("| technique | levers | states | mean ratio | worst | best |")
    print("|" + "---|" * 6)
    for (technique, levers), values in sorted(ratios.items(),
                                              key=lambda kv: -statistics.fmean(kv[1])):
        print(f"| {technique} | {levers} | {len(values)} | {statistics.fmean(values):.3f} | "
              f"{min(values):.3f} | {max(values):.3f} |")

    # 3. The technique comparison the project exists to make, plain build only.
    print("\n\n## Path Tracing vs VXPG, plain build, equal time\n")
    print("| scene | state | PT FLIP | VXPG FLIP | VXPG/PT | PT ms | VXPG ms |")
    print("|" + "---|" * 7)
    for (scene, state) in sorted({(s, st) for s, st, _ in by_state}):
        pt = next((g for g in by_state.get((scene, state, "Path Tracing"), [])
                   if g["levers"] == "none"), None)
        vx = next((g for g in by_state.get((scene, state, "Guided Path Tracing (VXPG)"), [])
                   if g["levers"] == "none"), None)
        if pt is None or vx is None:
            continue
        ratio = pt["metric"]["mean"] / vx["metric"]["mean"]
        print(f"| {scene} | {state} | {fmt(pt['metric']['mean'])} | {fmt(vx['metric']['mean'])} | "
              f"{ratio:.3f} | {pt['meanFrameMs']:.2f} | {vx['meanFrameMs']:.2f} |")


if __name__ == "__main__":
    main()
