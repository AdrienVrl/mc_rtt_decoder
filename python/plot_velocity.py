"""Figure: 10 s of true vs decoded velocity on the test set, for Wiener, Kalman and GRU.

Run from the repo root:  .venv\\Scripts\\python.exe python/plot_velocity.py [--start 20]
Writes figures/velocity.png. Trains each decoder once (GRU: seed 0), so the R²
in the panels can differ slightly from the 3-seed mean in the results table.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data import REPO_ROOT, causal_split, load_session, search_lag
from decoders import (GRU_PARAMS, WIENER_GRID, GRUDecoder, Kalman, Wiener,
                      evaluate, tune)

# Categorical slots 1-3 of the reference palette (fixed order); truth in neutral ink.
COLORS = {"Wiener": "#2a78d6", "Kalman": "#eb6834", "GRU": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def main(start_s: float, window_s: float) -> None:
    sess = load_session()
    lag, _ = search_lag(sess)
    sp = causal_split(sess, lag_bins=lag)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    print("training Wiener, Kalman, GRU ...")
    decoders = {
        "Wiener": tune(Wiener, WIENER_GRID, tr, va)[0],
        "Kalman": Kalman(steady_state=True, use_pos=True).fit(tr),
        "GRU": GRUDecoder(seed=0, **GRU_PARAMS).fit(tr, va),
    }
    # Decode the whole test block (recurrent decoders need the full history),
    # then show only the window.
    preds = {name: dec.predict(te.X) for name, dec in decoders.items()}
    scores = {name: evaluate(dec, te) for name, dec in decoders.items()}

    t = te.t - te.t[0]
    win = (t >= start_s) & (t < start_s + window_s)
    tw = t[win] - start_s

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(3, 2, figsize=(11, 6.5), sharex=True, sharey="col")
    for row, name in enumerate(decoders):
        for col, axis in enumerate(["x", "y"]):
            ax = axes[row, col]
            ax.plot(tw, te.Y[win, col], color=INK, lw=1.6, label="true")
            ax.plot(tw, preds[name][win, col], color=COLORS[name], lw=1.6, label=name)
            ax.set_title(f"{name}, v{axis}   R² = {scores[name][col]:.2f}",
                         loc="left", fontsize=9, color=INK)
            ax.grid(True, color=GRID, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("velocity")
            if row == 2:
                ax.set_xlabel("time (s)")
        axes[row, 1].legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(f"Decoded vs true finger velocity, {window_s:.0f} s of the test set "
                 f"(R² over the whole test set)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    out = REPO_ROOT / "figures" / "velocity.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=float, default=20.0, help="window start, seconds into the test set")
    p.add_argument("--window", type=float, default=10.0, help="window length in seconds")
    a = p.parse_args()
    main(a.start, a.window)
