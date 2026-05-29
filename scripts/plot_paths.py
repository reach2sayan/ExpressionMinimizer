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

title   = data.get("title", "Optimization paths")
grid    = data["grid"]
paths   = data["paths"]

nx, ny  = grid["nx"], grid["ny"]
xlo, xhi = grid["xlo"], grid["xhi"]
ylo, yhi = grid["ylo"], grid["yhi"]
Z = np.array(grid["z"], dtype=float).reshape(ny, nx)

X = np.linspace(xlo, xhi, nx)
Y = np.linspace(ylo, yhi, ny)

# ── Contour levels ─────────────────────────────────────────────────────────────
zmin, zmax = Z.min(), Z.max()
if zmax / max(zmin, 1e-12) > 100:
    # log-scale levels for functions with large dynamic range (e.g. Rosenbrock)
    levels = np.exp(np.linspace(np.log(max(zmin, 1e-3)), np.log(zmax), 60))
    norm   = mcolors.LogNorm(vmin=max(zmin, 1e-3), vmax=zmax)
else:
    levels = 60
    norm   = None

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.80, norm=norm)
plt.colorbar(cf, ax=ax, label="f(x, y)")
ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.35)

COLORS  = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#fabed4"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

for i, algo in enumerate(paths):
    name   = algo["name"]
    pts    = algo["points"]       # [[x0,y0], [x1,y1], ...]
    start  = algo.get("start",  pts[0]  if pts else None)
    final  = algo.get("final",  pts[-1] if pts else None)

    if not pts:
        print(f"  warning: no path points for {name}, skipping")
        continue

    xs, ys = zip(*pts)
    c = COLORS[i % len(COLORS)]
    m = MARKERS[i % len(MARKERS)]

    ax.plot(xs, ys, color=c, linewidth=1.8, alpha=0.9, label=f"{name} ({len(pts)} iters)", zorder=3)
    ax.scatter(xs, ys, color=c, s=20, zorder=4, alpha=0.7)
    if start:
        ax.scatter(*start, color=c, s=140, marker="o", edgecolors="white",
                   linewidths=1.5, zorder=6, label=f"{name} start")
    if final:
        ax.scatter(*final, color=c, s=220, marker="*", edgecolors="white",
                   linewidths=1.2, zorder=6)

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(title)
ax.legend(loc="upper right", fontsize=8, framealpha=0.88)
plt.tight_layout()

out = Path(data_file).with_suffix("").name + "_plot.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
