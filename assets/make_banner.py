"""Generate assets/banner.svg (deterministic). Run: python assets/make_banner.py"""

from __future__ import annotations

import random
from pathlib import Path

W, H = 1280, 400
BG = "#0a0c10"
INK = "#e6edf3"
MUTED = "#7d8590"
A1, A2 = "#5eead4", "#818cf8"  # teal -> indigo accent

rng = random.Random(7)
parts: list[str] = []
add = parts.append

add(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">'
)
add('<title id="t">skill-issue</title>')
add(
    '<desc id="d">A catalog of skills on the left flows through a single gate; one or two matching skills come out on the right.</desc>'
)
add("<defs>")
add(
    f'<linearGradient id="acc" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{A1}"/><stop offset="1" stop-color="{A2}"/></linearGradient>'
)
add(
    f'<linearGradient id="accv" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{A1}"/><stop offset="1" stop-color="{A2}"/></linearGradient>'
)
add(
    '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#818cf8" stop-opacity="0.35"/><stop offset="1" stop-color="#818cf8" stop-opacity="0"/></radialGradient>'
)
add(
    '<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#ffffff" fill-opacity="0.045"/></pattern>'
)
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
add(f'<rect width="{W}" height="{H}" fill="url(#dots)"/>')

# Left: the catalog, many faint skill pills in columns.
gate_x, gate_y = 760, 200
cols = [(470, 26), (540, 22), (610, 18)]
pill_centers = []
for ci, (x, n) in enumerate(cols):
    for i in range(n):
        y = 40 + i * (320 / (n - 1))
        w = rng.choice([34, 42, 50, 58])
        op = 0.10 + 0.10 * ci + rng.random() * 0.06
        add(
            f'<rect x="{x - w / 2:.1f}" y="{y - 4:.1f}" width="{w}" height="8" rx="4" fill="{MUTED}" fill-opacity="{op:.2f}"/>'
        )
        pill_centers.append((x + w / 2, y, op))

# Faint candidate lines from the last column into the gate (retrieval shortlist).
for x, y, _ in rng.sample([p for p in pill_centers if p[0] > 600], 12):
    add(
        f'<path d="M{x:.1f},{y:.1f} C{x + 70:.1f},{y:.1f} {gate_x - 70},{gate_y} {gate_x - 8},{gate_y}" fill="none" stroke="{A2}" stroke-opacity="0.18" stroke-width="1.2"/>'
    )

# The gate.
add(f'<circle cx="{gate_x}" cy="{gate_y}" r="90" fill="url(#glow)"/>')
add(f'<rect x="{gate_x - 5}" y="{gate_y - 70}" width="10" height="140" rx="5" fill="url(#accv)"/>')

# Right: the selected skills.
outs = [(gate_y - 46, "pdf-tools", "0.94", 1.0), (gate_y + 6, "sql-tuning", "0.81", 0.85)]
for y, name, p, op in outs:
    add(
        f'<path d="M{gate_x + 8},{gate_y} C{gate_x + 60},{gate_y} {gate_x + 60},{y + 12} {gate_x + 110},{y + 12}" fill="none" stroke="url(#acc)" stroke-opacity="{op}" stroke-width="2"/>'
    )
    add(
        f'<rect x="{gate_x + 110}" y="{y}" width="208" height="24" rx="12" fill="#11161d" stroke="url(#acc)" stroke-opacity="{op}" stroke-width="1.5"/>'
    )
    add(
        f'<text x="{gate_x + 126}" y="{y + 16.5}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="13" fill="{INK}" fill-opacity="{op}">{name}</text>'
    )
    add(
        f'<text x="{gate_x + 302}" y="{y + 16.5}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="13" fill="{A1}" fill-opacity="{op}">{p}</text>'
    )
y = gate_y + 58
add(
    f'<rect x="{gate_x + 110}" y="{y}" width="208" height="24" rx="12" fill="none" stroke="{MUTED}" stroke-opacity="0.35" stroke-dasharray="3 4"/>'
)
add(
    f'<text x="{gate_x + 126}" y="{y + 16.5}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="13" fill="{MUTED}" fill-opacity="0.8">or nothing at all</text>'
)

# Wordmark.
add(
    f'<text x="64" y="176" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="54" font-weight="700" fill="{INK}" letter-spacing="-1">skill<tspan fill="url(#acc)">-</tspan>issue</text>'
)
add(
    f'<text x="66" y="220" font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="19" fill="{MUTED}">The right Agent Skill for every prompt.</text>'
)
add(
    f'<text x="66" y="246" font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="19" fill="{MUTED}">Or none. Local, fast, calibrated.</text>'
)
add("</svg>")

Path(__file__).with_name("banner.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")
print("wrote assets/banner.svg")
