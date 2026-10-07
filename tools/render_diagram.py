"""Render the architecture diagram to a committed SVG, deterministically.

The diagram is generated rather than drawn so a reviewer can regenerate it,
and committed as an SVG with explicit width and height so it renders
identically on GitHub, in an editor preview, and offline. The background is
painted light on purpose: a transparent background puts dark edge labels on a
near-black page in dark themes.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagram.svg"

W, H = 920, 360
BOX_FILL = "#eef2f7"
BOX_EDGE = "#4a5a6a"
TEXT = "#1d2733"
ACCENT = "#9a3b3b"


def box(x, y, w, h, lines, fill=BOX_FILL):
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
        f'fill="{fill}" stroke="{BOX_EDGE}" stroke-width="1.5"/>'
    ]
    line_h = 18
    start = y + h / 2 - line_h * (len(lines) - 1) / 2 + 5
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        parts.append(
            f'<text x="{x + w / 2}" y="{start + i * line_h}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="13" '
            f'font-weight="{weight}" fill="{TEXT}">{line}</text>'
        )
    return "".join(parts)


def arrow(x1, y1, x2, y2, label=""):
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{BOX_EDGE}" '
        f'stroke-width="1.5" marker-end="url(#arrow)"/>'
    ]
    if label:
        parts.append(
            f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11" '
            f'fill="{TEXT}">{label}</text>'
        )
    return "".join(parts)


def main() -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{BOX_EDGE}"/></marker></defs>',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fcfdfe"/>',
        box(20, 140, 170, 80, ["seeded corpus", "58 docs, 36 queries", "labelled qrels"]),
        box(250, 50, 180, 70, ["baseline run", "embedder v1", "ExactIndex, top k"]),
        box(250, 240, 180, 70, ["candidate run", "v2 or lower dim", "ExactIndex, top k"]),
        box(490, 140, 200, 90, ["gate", "paired per-query deltas", "exact sign test",
                                "resolution floor"]),
        box(750, 40, 150, 60, ["exit 0 PASS", "no evidence of", "regression"]),
        box(750, 150, 150, 60, ["exit 1 REGRESSION", "p at or below alpha"], fill="#f7e9e9"),
        box(750, 260, 150, 60, ["exit 2 UNDERPOWERED", "set too small", "to rule"],
            fill="#f3eedd"),
        arrow(190, 165, 250, 95, "embed + retrieve"),
        arrow(190, 195, 250, 265, "embed + retrieve"),
        arrow(430, 85, 490, 165, "run A json"),
        arrow(430, 275, 490, 205, "run B json"),
        arrow(690, 160, 750, 80),
        arrow(690, 185, 750, 180),
        arrow(690, 210, 750, 280),
        f'<text x="20" y="25" font-family="Helvetica, Arial, sans-serif" font-size="14" '
        f'font-weight="bold" fill="{ACCENT}">rag-retrieval-gate: two runs in, '
        'one of three verdicts out</text>',
        "</svg>",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(svg))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
