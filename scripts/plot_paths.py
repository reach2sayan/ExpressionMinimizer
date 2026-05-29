#!/usr/bin/env python3
"""
Optimization path visualizer (Plotly).

Reads the JSON produced by the playground executable and renders an
interactive contour map with each algorithm's iteration path overlaid.
Hover over any point to see its coordinates and f value.  Click legend
entries to toggle individual paths on/off.

Usage:
    python3 scripts/plot_paths.py [path/to/optimization_data.json]

The data file is written by the playground to build/optimization_data.json
by default.  The script is fully generic — it works for any 2D function.

Install deps (once):
    pip install plotly numpy
"""

import sys
import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

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

# ── Contour level strategy ────────────────────────────────────────────────────
zmin, zmax = float(Z.min()), float(Z.max())
log_scale = zmax / max(zmin, 1e-12) > 100

if log_scale:
    raw = np.exp(np.linspace(np.log(max(zmin, 1e-3)), np.log(zmax), 18))
    contour_kwargs = dict(
        contours=dict(
            coloring="fill",
            showlines=True,
            start=float(raw[0]),
            end=float(raw[-1]),
            size=float(np.diff(raw).mean()),
        ),
        colorscale="Plasma",
        zmin=float(raw[0]),
        zmax=float(raw[-1]),
    )
else:
    n_levels = 22
    step = (zmax - zmin) / n_levels
    contour_kwargs = dict(
        contours=dict(
            coloring="fill",
            showlines=True,
            start=zmin,
            end=zmax,
            size=step,
        ),
        colorscale="Plasma",
        zmin=zmin,
        zmax=zmax,
    )

# ── Build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()

# Filled contour surface
fig.add_trace(go.Contour(
    x=X, y=Y, z=Z,
    name="f(x,y)",
    showscale=True,
    colorbar=dict(
        title=dict(text="f(x,y)", side="right"),
        tickfont=dict(color="white"),
        titlefont=dict(color="white"),
        outlinecolor="rgba(255,255,255,0.3)",
        outlinewidth=1,
    ),
    line=dict(width=1.2, color="rgba(255,255,255,0.55)"),
    contours_showlabels=True,
    contours_labelfont=dict(size=9, color="white"),
    hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>f=%{z:.4g}<extra>f(x,y)</extra>",
    **contour_kwargs,
))

# ── Algorithm paths ───────────────────────────────────────────────────────────
COLORS = [
    "#ff4d6d", "#43e97b", "#4cc9f0", "#f77f00",
    "#c77dff", "#f9c74f", "#80ffdb",
]

for i, algo in enumerate(paths):
    name  = algo["name"]
    pts   = algo["points"]
    start = algo.get("start", pts[0]  if pts else None)
    final = algo.get("final", pts[-1] if pts else None)

    if not pts:
        print(f"  warning: no path points for {name}, skipping")
        continue

    xs, ys = zip(*pts)
    # Evaluate f at each recorded iterate for the hover tooltip
    xi_idx = np.clip(((np.array(xs) - xlo) / (xhi - xlo) * (nx - 1)).astype(int), 0, nx - 1)
    yi_idx = np.clip(((np.array(ys) - ylo) / (yhi - ylo) * (ny - 1)).astype(int), 0, ny - 1)
    fs = Z[yi_idx, xi_idx]

    c = COLORS[i % len(COLORS)]

    # Path line + iteration dots
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        name=f"{name} ({len(pts)} iters)",
        line=dict(color=c, width=2.2),
        marker=dict(size=5, color=c, opacity=0.8,
                    line=dict(width=0.5, color="white")),
        customdata=np.stack([fs], axis=-1),
        hovertemplate=(
            f"<b>{name}</b><br>"
            "iter %{pointNumber}<br>"
            "x=%{x:.4f}  y=%{y:.4f}<br>"
            "f=%{customdata[0]:.4g}"
            "<extra></extra>"
        ),
        legendgroup=name,
    ))

    # Start marker
    if start:
        fig.add_trace(go.Scatter(
            x=[start[0]], y=[start[1]],
            mode="markers",
            name=f"{name} start",
            marker=dict(size=13, symbol="circle", color=c,
                        line=dict(width=2, color="white")),
            hovertemplate=f"<b>{name} start</b><br>x={start[0]:.4f}  y={start[1]:.4f}<extra></extra>",
            legendgroup=name,
            showlegend=False,
        ))

    # Final marker
    if final:
        f_final = float(Z[
            int(np.clip((final[1] - ylo) / (yhi - ylo) * (ny - 1), 0, ny - 1)),
            int(np.clip((final[0] - xlo) / (xhi - xlo) * (nx - 1), 0, nx - 1)),
        ])
        fig.add_trace(go.Scatter(
            x=[final[0]], y=[final[1]],
            mode="markers",
            name=f"{name} final",
            marker=dict(size=18, symbol="star", color=c,
                        line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{name} final</b><br>"
                f"x={final[0]:.4f}  y={final[1]:.4f}<br>"
                f"f={f_final:.4g}"
                "<extra></extra>"
            ),
            legendgroup=name,
            showlegend=False,
        ))

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text=title, font=dict(size=18, color="white"), x=0.5),
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#0f0f0f",
    xaxis=dict(
        title="x",
        gridcolor="rgba(255,255,255,0.12)",
        gridwidth=1,
        griddash="dot",
        zerolinecolor="rgba(255,255,255,0.25)",
        color="white",
        showgrid=True,
    ),
    yaxis=dict(
        title="y",
        gridcolor="rgba(255,255,255,0.12)",
        gridwidth=1,
        griddash="dot",
        zerolinecolor="rgba(255,255,255,0.25)",
        color="white",
        showgrid=True,
        scaleanchor="x",
        scaleratio=1,
    ),
    legend=dict(
        bgcolor="rgba(20,20,20,0.80)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color="white", size=11),
    ),
    width=950,
    height=720,
    margin=dict(l=60, r=20, t=60, b=60),
)

# ── Output ────────────────────────────────────────────────────────────────────
out_html = Path(data_file).with_suffix("").name + "_plot.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved {out_html}  (open in any browser)")
fig.show()
