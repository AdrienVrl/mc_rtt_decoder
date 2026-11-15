"""Export the steady-state Kalman decoder and the test set for the C++ port.

Run from the repo root:  .venv\\Scripts\\python.exe python/export.py
Writes export/ (overwritten each run):

  export/meta.json                sizes, lag, formats, ranges, Python test R²
  export/kalman/M.csv   (d, d)    state transition of the fixed recurrence
  export/kalman/K.csv   (d, N)    steady-state Kalman gain
  export/kalman/b.csv   (d,)      constant term, b = -K mu_y (centring folded in)
  export/kalman/v_offset.csv (2,) mean velocity, added to the output
  export/kalman/A.csv, C.csv, W.csv, Q.csv   the fitted model, for reference
  export/test/counts.csv     (T, N) integer spike counts, one bin per row
  export/test/valid.csv      (T,)   1 if the bin is real data, 0 in a gap
  export/test/vel_true.csv   (T, 2) true finger velocity
  export/test/vel_decoded.csv (T, 2) Python steady-state decoder output (float64)
  export/test/state_decoded.csv (T, d) full decoder state, for debugging

The recurrence the C++ decoder must implement, with raw integer counts y_t:

    x_0 = 0
    x_t = M x_{t-1} + K y_t + b          (x: centred state, d numbers)
    v_t = x_t[d-2:d] + v_offset          (velocity = last two state entries)

This is algebraically the same as Kalman.predict with steady_state=True:
there, x_t = M x_{t-1} + K (y_t - mu_y); expanding gives b = -K mu_y.

Floats are written with 17 significant digits, so reading them back gives the
exact same float64 values (needed for the ~1e-9 C++ vs Python test).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data import REPO_ROOT, causal_split, load_session, r2, search_lag
from decoders import Kalman

OUT = REPO_ROOT / "export"


def recurrence(M, K, b, Y):
    """The exported decoder, step by step, exactly as C++ will run it. Returns states (T, d)."""
    x = np.zeros(M.shape[0])
    out = np.zeros((len(Y), M.shape[0]))
    for t in range(len(Y)):
        x = M @ x + K @ Y[t] + b
        out[t] = x
    return out


def save(path: Path, a: np.ndarray, integer: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.atleast_1d(a), fmt="%d" if integer else "%.17g", delimiter=",")


def main() -> None:
    sess = load_session()
    lag, _ = search_lag(sess)
    sp = causal_split(sess, lag_bins=lag)
    tr, te = sp["train"], sp["test"]

    kf = Kalman(steady_state=True, use_pos=True).fit(tr)
    M, K = kf.M, kf.K
    b = -K @ kf.mu_y
    v_offset = kf.mu_x[-2:]
    d, N = M.shape[0], K.shape[1]

    # Stability: the recurrence must forget its past.
    rho = np.abs(np.linalg.eigvals(M)).max()
    assert rho < 1, f"unstable recurrence, spectral radius {rho:.4f}"

    # The folded recurrence must reproduce the Python decoder.
    states = recurrence(M, K, b, te.X.astype(np.float64))
    vel_decoded = states[:, -2:] + v_offset
    diff = np.abs(vel_decoded - kf.predict(te.X)).max()
    assert diff < 1e-9, f"folded recurrence differs from Kalman.predict by {diff:.2e}"

    # Counts must fit the uint8 input of Decoder::step.
    assert te.X.min() >= 0 and te.X.max() <= 255

    # Value ranges seen on TRAIN, to choose fixed-point integer bits
    # (plan: max |value| x 1.25, then enough integer bits to hold it).
    train_states = recurrence(M, K, b, tr.X.astype(np.float64))
    ranges = {
        "state_max_abs_train": float(np.abs(train_states).max()),
        "state_max_abs_train_per_dim": np.abs(train_states).max(0).tolist(),
        "M_max_abs": float(np.abs(M).max()),
        "K_max_abs": float(np.abs(K).max()),
        "b_max_abs": float(np.abs(b).max()),
        "counts_max_train": int(tr.X.max()),
    }

    for name, a in {"M": M, "K": K, "b": b, "v_offset": v_offset,
                    "A": kf.A, "C": kf.C, "W": kf.W, "Q": kf.Q}.items():
        save(OUT / "kalman" / f"{name}.csv", a)
    save(OUT / "test" / "counts.csv", te.X, integer=True)
    save(OUT / "test" / "valid.csv", te.valid.astype(int), integer=True)
    save(OUT / "test" / "vel_true.csv", te.Y)
    save(OUT / "test" / "vel_decoded.csv", vel_decoded)
    save(OUT / "test" / "state_decoded.csv", states)

    r2_test = r2(te.Y[te.valid], vel_decoded[te.valid])
    meta = {
        "dataset": "MC_RTT (DANDI 000129), train file, test = last 20% of the session",
        "bin_ms": sess.bin_ms,
        "lag_bins": int(lag),
        "n_units": int(N),
        "state_dim": int(d),
        "state_layout": ["pos_x", "pos_y", "vel_x", "vel_y"],
        "n_test_bins": int(len(te.X)),
        "recurrence": "x_0 = 0; x_t = M x_{t-1} + K y_t + b; v_t = x_t[d-2:d] + v_offset",
        "spectral_radius_M": float(rho),
        "python_test_r2": {"x": float(r2_test[0]), "y": float(r2_test[1]),
                           "mean": float(r2_test.mean())},
        "check_folded_vs_predict_max_abs_diff": float(diff),
        "ranges": ranges,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {OUT}")
    print(f"  d = {d}, N = {N}, test bins = {len(te.X)}, spectral radius = {rho:.4f}")
    print(f"  folded recurrence vs Kalman.predict: max diff {diff:.1e}")
    print(f"  test R2: x {r2_test[0]:.3f}  y {r2_test[1]:.3f}  mean {r2_test.mean():.3f}")
    print(f"  ranges: {json.dumps(ranges)}")


if __name__ == "__main__":
    main()
