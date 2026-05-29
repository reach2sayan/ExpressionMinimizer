#!/usr/bin/env python3
"""
Optimization path visualizer.

Reads the JSON produced by the playground executable and plots the contour
map of the objective function with each algorithm's iteration path overlaid.

Usage:
    python3 scripts/plot_paths.py [path/to/optimization_data.json]

The data file is written by the playground to build/optimization_data.json
by default.  The script is fully generic — it works for any 2D function.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
data_file = sys.argv[1] if len(sys.argv) > 1 else "optimization_data.json"
data = json.loads(Path(data_file).read_text())

title  = data.get("title", "Optimization paths")
grid   = data["grid"]
paths  = data["paths"]

nx, ny   = grid["nx"], grid["ny"]
xlo, xhi = grid["xlo"], grid["xhi"]
ylo, yhi = grid["ylo"], grid["yhi"]
Z = np.array(grid["z"], dtype=float).reshape(ny, nx)

X = np.linspace(xlo, xhi, nx)
Y = np.linspace(ylo, yhi, ny)

# ── Contour levels ────────────────────────────────────────────────────────────
zmin, zmax = Z.min(), Z.max()
if zmax / max(zmin, 1e-12) > 100:
    levels      = np.exp(np.linspace(np.log(max(zmin, 1e-3)), np.log(zmax), 30))
    line_levels = np.exp(np.linspace(np.log(max(zmin, 1e-3)), np.log(zmax), 12))
    norm        = mcolors.LogNorm(vmin=max(zmin, 1e-3), vmax=zmax)
else:
    levels      = np.linspace(zmin, zmax, 30)
    line_levels = np.linspace(zmin, zmax, 12)
    norm        = None

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
ax.set_facecolor("white")

# Filled contour — light, low-saturation colormap so paths stand out
cf = ax.contourf(X, Y, Z, levels=levels, cmap="YlGnBu", alpha=0.72, norm=norm)
cbar = plt.colorbar(cf, ax=ax, label="f(x, y)")
cbar.ax.yaxis.label.set_color("#111111")
cbar.ax.tick_params(colors="#111111")

# Iso-contour lines — dark, clearly separated, with value labels
cs = ax.contour(X, Y, Z, levels=line_levels, colors="#1a1a1a",
                linewidths=0.9, alpha=0.75, norm=norm)
ax.clabel(cs, cs.levels, inline=True, fontsize=7, fmt="%.3g",
          colors="#1a1a1a", inline_spacing=5)

# Grid — subtle, doesn't compete with paths
ax.grid(True, color="#888888", linewidth=0.4, alpha=0.4, linestyle="--")
ax.set_axisbelow(True)

# ── Algorithm paths ───────────────────────────────────────────────────────────
# High-contrast, distinct hues that read well on a light background
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
          "#9467bd", "#8c564b", "#17becf"]

for i, algo in enumerate(paths):
    name  = algo["name"]
    pts   = algo["points"]
    start = algo.get("start", pts[0]  if pts else None)
    final = algo.get("final", pts[-1] if pts else None)

    if not pts:
        print(f"  warning: no path points for {name}, skipping")
        continue

    xs, ys = zip(*pts)
    c = COLORS[i % len(COLORS)]

    ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.95,
            label=f"{name} ({len(pts)} iters)", zorder=3)
    ax.scatter(xs, ys, color=c, s=18, zorder=4, alpha=0.75,
               edgecolors="white", linewidths=0.4)

    if start:
        ax.scatter(*start, color=c, s=150, marker="o", zorder=6,
                   edgecolors="#111111", linewidths=1.5)

    if final:
        ax.scatter(*final, color=c, s=260, marker="*", zorder=6,
                   edgecolors="#111111", linewidths=1.0)

# ── Labels & legend ───────────────────────────────────────────────────────────
ax.set_xlabel("x", color="#111111", fontsize=11)
ax.set_ylabel("y", color="#111111", fontsize=11)
ax.set_title(title, color="#111111", fontsize=14, pad=10)
ax.tick_params(colors="#111111")
for spine in ax.spines.values():
    spine.set_edgecolor("#aaaaaa")

ax.legend(loc="upper right", fontsize=9, framealpha=0.90,
          facecolor="white", edgecolor="#aaaaaa")
plt.tight_layout()

out = Path(data_file).with_suffix("").name + "_plot.png"
plt.savefig(out, dpi=150, facecolor="white")
print(f"Saved {out}")
plt.show()
