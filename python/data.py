"""Load, bin and causally split the MC_RTT dataset.

Typical use:

    from data import load_session, causal_split

    sess = load_session(bin_ms=20)
    splits = causal_split(sess, lag_bins=5)          # 100 ms lag at 20 ms bins
    tr, va, te = splits["train"], splits["val"], splits["test"]
    # tr.X: (T, n_units) int32 spike counts at time t
    # tr.Y: (T, 2) finger velocity at time t + lag
    # tr.valid: (T,) bool, False where the recording had a gap

Gaps: about 0.05% of the session is missing. Rows are NOT dropped, because
dropping them would glue together bins that are not neighbours in time, and the
decoders (spike history, Kalman, GRU) all rely on consecutive bins. Instead,
missing values are filled with 0 and flagged in `valid`. Exclude invalid bins
when fitting and when scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "000129" / "sub-Indy"


@dataclass
class Session:
    """The whole session, binned, in time order."""

    counts: np.ndarray  # (T, n_units) int32 spike counts per bin
    vel: np.ndarray  # (T, 2) finger velocity (x, y), 0 where invalid
    pos: np.ndarray  # (T, 2) finger position (x, y), 0 where invalid
    valid: np.ndarray  # (T,) bool, True where spikes and behaviour are present
    t: np.ndarray  # (T,) bin start time in seconds
    unit_ids: np.ndarray  # (n_units,) unit identifiers from the NWB file
    bin_ms: int

    @property
    def n_bins(self) -> int:
        return self.counts.shape[0]

    @property
    def n_units(self) -> int:
        return self.counts.shape[1]


@dataclass
class Split:
    """One contiguous block of the session, with the lag already applied."""

    X: np.ndarray  # (T, n_units) int32 spike counts at time t
    Y: np.ndarray  # (T, 2) velocity at time t + lag
    valid: np.ndarray  # (T,) bool, both X[t] and Y[t] are real data
    t: np.ndarray  # (T,) time of X[t] in seconds
    pos: np.ndarray  # (T, 2) finger position at time t + lag (same timing as Y)


def load_session(data_dir: str | Path = DEFAULT_DATA_DIR, bin_ms: int = 20) -> Session:
    """Load the MC_RTT train file with nlb_tools and bin it to `bin_ms`."""
    from nlb_tools.nwb_interface import NWBDataset

    data_dir = Path(data_dir)
    if not any(data_dir.glob("*train*.nwb")):
        raise FileNotFoundError(
            f"No *train*.nwb in {data_dir}. From the repo root run:\n"
            "  mkdir data\n  uvx dandi download DANDI:000129/draft -o data"
        )

    ds = NWBDataset(str(data_dir), "*train", split_heldout=False)
    # Spikes are stored as float16, which cannot sum long columns correctly.
    # Cast before resampling (resample() keeps the dtype it finds).
    ds.data = ds.data.astype({c: np.float32 for c in ds.data.columns if c[0] == "spikes"})
    ds.resample(bin_ms)
    df = ds.data

    spikes = df["spikes"].to_numpy(dtype=np.float64)
    vel = df["finger_vel"][["x", "y"]].to_numpy(dtype=np.float64)
    pos = df["finger_pos"][["x", "y"]].to_numpy(dtype=np.float64)

    valid = ~(np.isnan(spikes).any(axis=1) | np.isnan(vel).any(axis=1) | np.isnan(pos).any(axis=1))

    counts = np.nan_to_num(spikes, nan=0.0)
    if not np.all(counts == np.round(counts)):
        raise ValueError("binned spike counts are not integers")
    counts = counts.astype(np.int32)

    return Session(
        counts=counts,
        vel=np.where(valid[:, None], vel, 0.0),
        pos=np.where(valid[:, None], pos, 0.0),
        valid=valid,
        t=df.index.total_seconds().to_numpy(),
        unit_ids=np.asarray(df["spikes"].columns),
        bin_ms=bin_ms,
    )


def apply_lag(sess: Session, lag_bins: int) -> Split:
    """Pair spikes at time t with velocity at t + lag_bins (neural activity leads movement)."""
    if lag_bins < 0:
        raise ValueError("lag_bins must be >= 0")
    T = sess.n_bins - lag_bins
    return Split(
        X=sess.counts[:T],
        Y=sess.vel[lag_bins:],
        valid=sess.valid[:T] & sess.valid[lag_bins:],
        t=sess.t[:T],
        pos=sess.pos[lag_bins:],
    )


def split_bounds(n_bins: int, fractions=(0.7, 0.1, 0.2)) -> dict[str, slice]:
    """Contiguous train / val / test index ranges, in time order, no shuffling."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    a = int(round(fractions[0] * n_bins))
    b = int(round((fractions[0] + fractions[1]) * n_bins))
    return {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n_bins)}


def causal_split(sess: Session, lag_bins: int, fractions=(0.7, 0.1, 0.2)) -> dict[str, Split]:
    """Apply the lag, then cut the session into train (first 70%), val (next 10%), test (last 20%)."""
    full = apply_lag(sess, lag_bins)
    out = {}
    for name, s in split_bounds(len(full.X), fractions).items():
        out[name] = Split(X=full.X[s], Y=full.Y[s], valid=full.valid[s], t=full.t[s], pos=full.pos[s])
    return out


def r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """R² per output column (per velocity axis)."""
    return 1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def search_lag(sess: Session, lags=range(0, 11), alpha: float = 1.0) -> tuple[int, np.ndarray]:
    """Pick the spike-to-velocity lag on the validation split.

    For each lag, fit a ridge regression from the spike counts of one bin to the
    velocity `lag` bins later on train, and score R² on val. The test split is
    never touched. Returns (best lag in bins, scores of shape (n_lags, 2)).
    """
    from sklearn.linear_model import Ridge

    lags = list(lags)
    scores = np.zeros((len(lags), 2))
    for i, lag in enumerate(lags):
        sp = causal_split(sess, lag_bins=lag)
        tr, va = sp["train"], sp["val"]
        model = Ridge(alpha=alpha).fit(tr.X[tr.valid], tr.Y[tr.valid])
        scores[i] = r2(va.Y[va.valid], model.predict(va.X[va.valid]))
    best = lags[int(scores.mean(1).argmax())]
    return best, scores


if __name__ == "__main__":
    sess = load_session()
    print(f"{sess.n_units} units, {sess.n_bins} bins of {sess.bin_ms} ms "
          f"({sess.n_bins * sess.bin_ms / 60000:.1f} min), "
          f"{(~sess.valid).sum()} invalid bins, max count {sess.counts.max()}")
    for name, sp in causal_split(sess, lag_bins=5).items():
        print(f"{name:5s}: {len(sp.X):6d} bins, t = {sp.t[0]:7.1f}..{sp.t[-1]:7.1f} s, "
              f"{(~sp.valid).sum()} invalid")

    lags = range(0, 11)
    best, scores = search_lag(sess, lags)
    print("\nlag search (val R2, single-bin ridge):")
    for lag, (rx, ry) in zip(lags, scores):
        mark = "  <- best" if lag == best else ""
        print(f"  {lag * sess.bin_ms:4d} ms: x {rx:.3f}  y {ry:.3f}  mean {(rx + ry) / 2:.3f}{mark}")
