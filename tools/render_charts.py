"""Render the headline charts to committed SVGs, from the experiment JSON.

Data-bound like the prose: the charts read docs/experiments/*.json at render
time, and CI regenerates them and fails on any diff. The first chart is the
argument for the gate's existence in one picture; the second is the
resolution ladder a golden set has to climb before it can support a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "docs" / "experiments"
OUT = ROOT / "docs"

INK = "#1d2733"
TEAL = "#2b6777"
WARM = "#9a3b3b"
GOLD = "#b8860b"
BG = "#fcfdfe"
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def text(x, y, s, size=12, anchor="middle", fill=INK, weight="normal"):
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}">{s}</text>')


def frame(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>',
            text(w / 2, 26, title, size=15, weight="bold")]


def chart_false_alarms() -> None:
    data = json.loads((EXP / "noise_floor.json").read_text())
    naive = data["naive"]
    gate = data["gate"]
    w, h = 820, 340
    left, top, bottom = 80, 56, 72
    px, py = w - left - 40, h - top - bottom

    def sy(v):
        return top + py * (1 - v / 110.0)

    svg = frame(w, h, "False alarm rate on identical systems: the naive rule vs the gate")
    for v in range(0, 101, 25):
        svg.append(f'<line x1="{left}" y1="{sy(v)}" x2="{w - 40}" y2="{sy(v)}" '
                   f'stroke="#dde5ec"/>')
        svg.append(text(left - 8, sy(v) + 4, f"{v}%", size=11, anchor="end"))
    svg.append(f'<line x1="{left}" y1="{sy(gate["alpha_pct"])}" x2="{w - 40}" '
               f'y2="{sy(gate["alpha_pct"])}" stroke="{GOLD}" stroke-width="2" '
               f'stroke-dasharray="7 5"/>')
    svg.append(text(w - 44, sy(gate["alpha_pct"]) - 7,
                    f"declared alpha {gate['alpha_pct']:g}%", anchor="end",
                    fill=GOLD, size=12))
    cols = [*naive, None]
    bw = px / len(cols) * 0.44
    for i, r in enumerate(naive):
        cx = left + px * (i + 0.5) / len(cols)
        y = sy(r["fp_rate_pct"])
        svg.append(f'<rect x="{cx - bw / 2}" y="{y}" width="{bw}" '
                   f'height="{h - bottom - y}" fill="{WARM}"/>')
        svg.append(text(cx, y - 8, f"{r['fp_rate_pct']:g}%", weight="bold", fill=WARM))
        svg.append(text(cx, h - bottom + 16, f"naive, {r['subset_size']}-query", size=12))
        svg.append(text(cx, h - bottom + 32, "resamples", size=12))
    cx = left + px * (len(cols) - 0.5) / len(cols)
    y = sy(gate["fp_rate_pct"])
    svg.append(f'<rect x="{cx - bw / 2}" y="{y}" width="{bw}" '
               f'height="{h - bottom - y}" fill="{TEAL}"/>')
    svg.append(text(cx, y - 8, f"{gate['fp_rate_pct']:g}%", weight="bold", fill=TEAL))
    svg.append(text(cx, h - bottom + 16, "the paired", size=12))
    svg.append(text(cx, h - bottom + 32, "sign-test gate", size=12))
    svg.append(text(left + px / 2, h - 8,
                    f"{gate['trials']} seeded trials per bar; the naive rule is a "
                    f"{data['naive_threshold_points']:g}-point mean threshold on "
                    "an identical system", size=12))
    svg.append("</svg>")
    (OUT / "chart-false-alarms.svg").write_text("\n".join(svg))


def chart_resolution() -> None:
    data = json.loads((EXP / "resolution.json").read_text())
    rows = data["required"]
    w, h = 820, 330
    left, top, bottom = 80, 56, 64
    px, py = w - left - 40, h - top - bottom
    ymax = max(r["required_queries"] for r in rows) * 1.18

    def sy(v):
        return top + py * (1 - v / ymax)

    svg = frame(w, h, "Queries a golden set needs before the gate can call a regression")
    for v in range(0, int(ymax) + 1, 25):
        svg.append(f'<line x1="{left}" y1="{sy(v)}" x2="{w - 40}" y2="{sy(v)}" '
                   f'stroke="#dde5ec"/>')
        svg.append(text(left - 8, sy(v) + 4, str(v), size=11, anchor="end"))
    bw = px / len(rows) * 0.46
    pop = json.loads((EXP / "noise_floor.json").read_text())["population_queries"]
    for i, r in enumerate(rows):
        cx = left + px * (i + 0.5) / len(rows)
        y = sy(r["required_queries"])
        over = r["required_queries"] > pop
        svg.append(f'<rect x="{cx - bw / 2}" y="{y}" width="{bw}" '
                   f'height="{h - bottom - y}" fill="{TEAL if not over else WARM}"/>')
        svg.append(text(cx, y - 8, str(r["required_queries"]), weight="bold"))
        svg.append(text(cx, h - bottom + 16,
                        f"{r['regression_rate_pct']}% of queries", size=12))
        svg.append(text(cx, h - bottom + 32, "regress", size=12))
    svg.append(f'<line x1="{left}" y1="{sy(pop)}" x2="{w - 40}" y2="{sy(pop)}" '
               f'stroke="{GOLD}" stroke-width="2" stroke-dasharray="7 5"/>')
    svg.append(text(w - 44, sy(pop) - 7,
                    f"this repo's own set: {pop} queries", anchor="end",
                    fill=GOLD, size=12))
    svg.append(text(left + px / 2, h - 8,
                    f"exact sign-test power at {int(data['power'] * 100)}%, alpha "
                    f"{data['alpha']:g}; red bars exceed the set this repository "
                    "itself ships", size=12))
    svg.append("</svg>")
    (OUT / "chart-resolution.svg").write_text("\n".join(svg))


def main() -> None:
    chart_false_alarms()
    chart_resolution()
    print(f"wrote {OUT / 'chart-false-alarms.svg'} and {OUT / 'chart-resolution.svg'}")


if __name__ == "__main__":
    main()
