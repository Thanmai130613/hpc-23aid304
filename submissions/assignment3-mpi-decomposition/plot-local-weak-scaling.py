#!/usr/bin/env python3
"""Render the measured local-only weak-scaling CSV as a labelled SVG."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "local-weak-scaling.csv"
OUTPUT = HERE / "local-weak-scaling.svg"

WIDTH, HEIGHT = 760, 470
LEFT, RIGHT, TOP, BOTTOM = 90, 35, 55, 85
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM
Y_MAX = 22.0
XS = (LEFT + 35, LEFT + 35 + PLOT_W // 3,
      LEFT + 35 + 2 * PLOT_W // 3, WIDTH - RIGHT - 35)

with DATA.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

def y(value: float) -> float:
    return TOP + PLOT_H * (1.0 - value / Y_MAX)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="380" y="27" text-anchor="middle" font-family="sans-serif" font-size="17">Local weak scaling (single virtualized machine)</text>',
]

for tick in (0, 5, 10, 15, 20):
    yy = y(tick)
    parts.append(f'<line x1="{LEFT}" y1="{yy:.1f}" x2="{WIDTH-RIGHT}" y2="{yy:.1f}" stroke="#dddddd"/>')
    parts.append(f'<text x="{LEFT-12}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>')

parts.extend([
    f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT-BOTTOM}" stroke="#222"/>',
    f'<line x1="{LEFT}" y1="{HEIGHT-BOTTOM}" x2="{WIDTH-RIGHT}" y2="{HEIGHT-BOTTOM}" stroke="#222"/>',
    f'<text x="{(LEFT+WIDTH-RIGHT)/2:.1f}" y="{HEIGHT-25}" text-anchor="middle" font-family="sans-serif" font-size="14">MPI ranks</text>',
    f'<text x="22" y="{HEIGHT/2:.1f}" text-anchor="middle" transform="rotate(-90 22 {HEIGHT/2:.1f})" font-family="sans-serif" font-size="14">Minimum runtime (ms)</text>',
])

for x, row in zip(XS, rows):
    parts.append(f'<text x="{x}" y="{HEIGHT-BOTTOM+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["ranks"]}</text>')

series = (
    ("blocking", "blocking_min_ms", "blocking_spread_ms", "#1769aa"),
    ("nonblocking with overlap", "overlap_min_ms", "overlap_spread_ms", "#c43b31"),
)
for label, min_key, spread_key, color in series:
    points = []
    for x, row in zip(XS, rows):
        minimum = float(row[min_key])
        maximum = minimum + float(row[spread_key])
        yy = y(minimum)
        points.append(f"{x},{yy:.1f}")
        parts.append(f'<line x1="{x}" y1="{y(maximum):.1f}" x2="{x}" y2="{yy:.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x-5}" y1="{y(maximum):.1f}" x2="{x+5}" y2="{y(maximum):.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x}" cy="{yy:.1f}" r="5" fill="{color}"/>')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')

parts.extend([
    '<line x1="420" y1="54" x2="448" y2="54" stroke="#1769aa" stroke-width="2"/><circle cx="434" cy="54" r="4" fill="#1769aa"/><text x="455" y="59" font-family="sans-serif" font-size="12">blocking</text>',
    '<line x1="545" y1="54" x2="573" y2="54" stroke="#c43b31" stroke-width="2"/><circle cx="559" cy="54" r="4" fill="#c43b31"/><text x="580" y="59" font-family="sans-serif" font-size="12">nonblocking + overlap</text>',
    '<text x="380" y="450" text-anchor="middle" font-family="sans-serif" font-size="11">Points: minimum of 5 timed runs; bars: min-to-max. Local result only; not ASA multi-node evidence.</text>',
    '</svg>',
])

OUTPUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(OUTPUT)
