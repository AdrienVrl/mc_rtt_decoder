"""Causal velocity decoders for MC_RTT: Wiener, Kalman, GRU.

Every decoder has the same two methods:

    dec.fit(train)           # train: data.Split, returns self
    dec.predict(X) -> Yhat   # X: (T, n_units) spikes of ONE contiguous block,
                             # Yhat: (T, 2) velocity; row t may only use X[:t+1]

`predict` always receives a whole contiguous block (val or test) and returns a
prediction for every row, so recurrent decoders can carry their state through
time. Scoring then ignores rows where `valid` is False (see `evaluate`).

Run `python decoders.py` to load the data, pick the lag, tune each decoder on
val and print the results table on test.
"""

from __future__ import annotations

import copy  # noqa: F401  (for copy.deepcopy in GRUDecoder.fit)
import itertools

import numpy as np
from sklearn.linear_model import Ridge

from data import Split, causal_split, load_session, r2, search_lag


class Decoder:
    def fit(self, train: Split) -> "Decoder":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def evaluate(dec: Decoder, split: Split) -> np.ndarray:
    """R² per axis on the valid rows of `split`."""
    Yhat = dec.predict(split.X)
    return r2(split.Y[split.valid], Yhat[split.valid])


def tune(make, grid: dict, train: Split, val: Split) -> tuple[Decoder, dict, float]:
    """Try every combination in `grid`, fit on train, score mean R² on val.

    make: function taking the grid's keys as keyword arguments, returning a Decoder.
    Returns (best decoder, fitted on train), its parameters and its val score.
    """
    best = (None, None, -np.inf)
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        dec = make(**params).fit(train)
        score = evaluate(dec, val).mean()
        if score > best[2]:
            best = (dec, params, score)
    return best


# --------------------------------------------------------------------------
# Reference: the single-bin ridge used by data.search_lag. Already working,
# shown here as an example of the interface. This is what Wiener must beat.
# --------------------------------------------------------------------------
class SingleBinRidge(Decoder):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, train: Split) -> "SingleBinRidge":
        self.model = Ridge(alpha=self.alpha).fit(train.X[train.valid], train.Y[train.valid])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# --------------------------------------------------------------------------
# 1. Wiener filter
# --------------------------------------------------------------------------
def history(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Stack the last `n_bins` bins of spikes into one feature vector per row.

    X: (T, n_units)  ->  H: (T, n_bins * n_units), where row t contains
    X[t], X[t-1], ..., X[t-n_bins+1]. Only the past and present: never X[t+1].
    For the first n_bins-1 rows, where the past is not available, pad with zeros.

    Column block k (columns k*n_units .. (k+1)*n_units) holds X[t-k], the spikes
    from k bins ago. A 1-D input (e.g. a valid mask) is treated as one column.
    """
    if X.ndim == 1:
        X = X[:, None]
    T, n = X.shape
    H = np.zeros((T, n_bins * n), dtype=X.dtype)
    for k in range(n_bins):
        H[k:, k * n:(k + 1) * n] = X[:T - k]  # row t gets X[t-k]; rows t<k stay 0
    return H


class Wiener(Decoder):
    """Ridge regression from the last `n_bins` bins of spikes to velocity.
    """

    def __init__(self, n_bins: int = 5, alpha: float = 1.0):
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, train: Split) -> "Wiener":
        H = history(train.X, self.n_bins)
        valid = history(train.valid, self.n_bins).all(1)  # only rows where every bin in the window is valid
        self.model = Ridge(alpha=self.alpha).fit(H[valid], train.Y[valid])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = history(X, self.n_bins)
        return self.model.predict(H)

WIENER_GRID = {"n_bins": [5, 8, 10], "alpha": [1e-1, 1e0, 1e1, 1e2, 1e3]}


# --------------------------------------------------------------------------
# 2. Kalman filter  (your turn)
# --------------------------------------------------------------------------
# Notation (Wu et al. 2006, same letters as the project plan):
#   x_t in R^d : hidden state = finger velocity at time t (+lag), d = 2,
#                or [position, velocity] with use_pos=True, d = 4
#   y_t in R^N : observation  = spike counts in bin t, N = 130
#
#   dynamics     x_t = A x_{t-1} + w_t,   w_t ~ N(0, W)    A: d x d, W: d x d
#   observation  y_t = C x_t     + q_t,   q_t ~ N(0, Q)    C: N x d, Q: N x N
#
# In a Split, row t already pairs X[t] (spikes) with Y[t] (velocity lag bins
# later), so: state x_t = Y[t], observation y_t = X[t].
#
# Offsets: the model above has no constant term. Work in centred coordinates:
# subtract the train means (mu_x of velocity, mu_y of spike counts) before
# fitting, and add mu_x back to the decoded velocity. (Folding the means into
# a constant bias for the C++ port comes on Day 5.)


class Kalman(Decoder):
    """Kalman filter decoder of 2-D velocity from spike counts.

    steady_state=False : full filter, the gain K_t is recomputed every bin.
    steady_state=True  : fixed gain K from the Riccati equation, so inference is
                         the recurrence  x_t = M x_{t-1} + K y_t,  M = (I - K C) A.
                         This is the version ported to C++ in week 2.
    """

    def __init__(self, steady_state: bool = False, use_pos: bool = False):
        self.steady_state = steady_state
        self.use_pos = use_pos

    def _state(self, split: Split) -> np.ndarray:
        """State trajectory (T, d): velocity, or [pos_x, pos_y, vel_x, vel_y]. Velocity is always last."""
        return np.hstack([split.pos, split.Y]) if self.use_pos else split.Y

    # ---- fitting ---------------------------------------------------------
    def fit(self, train: Split) -> "Kalman":
        # 1. Centre. Using only valid rows, store
        S = self._state(train)  # (T, d)
        self.mu_x = np.mean(S[train.valid], axis=0)
        self.mu_y = np.mean(train.X[train.valid], axis=0)
        Xs = S - self.mu_x  # (T, d)
        Ys = train.X - self.mu_y  # (T, N)

        # 2. Dynamics A, W, by least squares on consecutive pairs.
        #   Use only t where BOTH row t-1 and row t are valid:
        pair = train.valid[1:] & train.valid[:-1]
        X1 = Xs[:-1][pair]
        X2 = Xs[1:][pair]
        self.A = np.linalg.lstsq(X1, X2, rcond=None)[0].T
        self.W = np.cov((X2 - X1 @ self.A.T).T, bias=True)  # bias=True for 1/n_pairs

        # Observation C, Q, by least squares on valid rows
        self.C = np.linalg.lstsq(Xs[train.valid], Ys[train.valid], rcond=None)[0].T
        R = Ys[train.valid] - Xs[train.valid] @ self.C.T
        self.Q = np.cov(R.T, bias=True)

        # 4. Initial state for predict: x0 = 0 (centred mean)
        self.x0 = np.zeros(Xs.shape[1])
        self.P0 = np.cov(Xs[train.valid].T, bias=True)

        # steady_state only: compute the fixed gain K and M = (I - K C) A
        if self.steady_state:
            self.K, self.M = self.steady_state_gain()

        d, N = S.shape[1], train.X.shape[1]
        assert self.A.shape == (d, d) and self.W.shape == (d, d)
        assert self.C.shape == (N, d) and self.Q.shape == (N, N)
        return self

    def steady_state_gain(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (K, M): the fixed gain K (d x N) and M = (I - K C) A (d x d).
        """

        if self.steady_state:
            from scipy.linalg import solve_discrete_are

            # Solve the discrete algebraic Riccati equation for P
            P = solve_discrete_are(self.A.T, self.C.T, self.W, self.Q)

            # Compute the steady-state gain K
            S = self.C @ P @ self.C.T + self.Q
            K = np.linalg.solve(S, self.C @ P).T  # K = P C^T S^{-1}

            # Compute M = (I - K C) A
            M = (np.eye(self.A.shape[0]) - K @ self.C) @ self.A

            return K, M
        else:
            raise ValueError("steady_state_gain should only be called when steady_state is True.")

    # ---- inference -------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decode a contiguous block X (T, N) -> velocity (T, 2), causally.
        """
        Ys = X - self.mu_y
        x = self.x0.copy()
        P = self.P0.copy()
        T = X.shape[0]
        d = x.shape[0]
        decoded = np.zeros((T, d))

        for t in range(T):
            if self.steady_state:
                x = self.M @ x + self.K @ Ys[t]
            else:
                x_minus = self.A @ x
                P_minus = self.A @ P @ self.A.T + self.W
                K = np.linalg.solve(self.C @ P_minus @ self.C.T + self.Q, self.C @ P_minus).T
                x = x_minus + K @ (Ys[t] - self.C @ x_minus)
                P = (np.eye(d) - K @ self.C) @ P_minus
            decoded[t] = x + self.mu_x
        return decoded[:, -2:]  # velocity is the last two state components


# --------------------------------------------------------------------------
# 3. GRU  (your turn)
# --------------------------------------------------------------------------
# Model: a one-layer GRU reads the spike counts bin by bin and keeps a hidden
# state h_t (H numbers, H = 64..128) that summarises the past; a linear readout
# maps h_t to velocity.
#
#   h_t     = GRU(h_{t-1}, x_t)        x_t: (N,) spikes,  h_t: (H,)
#   v_hat_t = W_out h_t + b_out        W_out: (2, H)
#
# The GRU is unidirectional, so v_hat_t only depends on x_1..x_t: causal by
# construction. Same role as the Kalman state, but the update is nonlinear and
# learned by gradient descent instead of derived from a Gaussian model.
#
# Training: cut the train split into sequences of a few seconds, start each one
# from h = 0, minimise masked MSE, and stop when val R² stops improving.
# Inference: run the whole val/test block as ONE long sequence (batch of 1).
#
# What is done for you: the Scaler (z-scoring), seeding, and the wiring in
# __main__ (3 seeds, mean ± std on test). What is yours: TODO 1-5.

import torch
from torch import nn


class Scaler:
    """Per-column z-scoring fitted on valid train rows. Neural nets train badly
    on raw inputs of very different scales (counts 0-3 vs velocities ~100s)."""

    def fit(self, A: np.ndarray, valid: np.ndarray) -> "Scaler":
        self.mean = A[valid].mean(0)
        self.std = A[valid].std(0) + 1e-6
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        return ((A - self.mean) / self.std).astype(np.float32)

    def inverse(self, A: np.ndarray) -> np.ndarray:
        return A * self.std + self.mean


class GRUNet(nn.Module):

    def __init__(self, n_in: int, hidden: int, n_out: int = 2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        gru_out, h = self.gru(x, h0)
        v = self.readout(gru_out)
        return v, h


def make_sequences(X: np.ndarray, Y: np.ndarray, mask: np.ndarray, seq_len: int, stride: int):
    Xs = np.stack([X[i:i + seq_len] for i in range(0, len(X) - seq_len + 1, stride)])
    Ys = np.stack([Y[i:i + seq_len] for i in range(0, len(Y) - seq_len + 1, stride)])
    Ms = np.stack([mask[i:i + seq_len] for i in range(0, len(mask) - seq_len + 1, stride)])
    return Xs, Ys, Ms


class GRUDecoder(Decoder):
    needs_val = True  # fit(train, val): val is used for early stopping

    def __init__(self, hidden: int = 64, seq_len: int = 150, stride: int = 50,
                 warmup: int = 25, batch_size: int = 32, lr: float = 1e-3,
                 weight_decay: float = 1e-4, max_epochs: int = 200,
                 patience: int = 15, seed: int = 0):
        # seq_len 150 bins = 3 s; warmup 25 bins = 0.5 s (see TODO 3)
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def fit(self, train: Split, val: Split) -> "GRUDecoder":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.x_scaler = Scaler().fit(train.X, train.valid)
        self.y_scaler = Scaler().fit(train.Y, train.valid)
        Xn = self.x_scaler.transform(train.X)
        Yn = self.y_scaler.transform(train.Y)

        # Loss mask
        mask = train.valid.copy()
        Xs, Ys, Ms = make_sequences(Xn, Yn, mask, self.seq_len, self.stride)
        Ms[:, :self.warmup] = False
        Xs = torch.tensor(Xs, dtype=torch.float32)
        Ys = torch.tensor(Ys, dtype=torch.float32)
        Ms = torch.tensor(Ms, dtype=torch.bool)

        self.net = GRUNet(n_in=train.X.shape[1], hidden=self.hidden)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_score = -np.inf
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(Xs.size(0))
            Xs_shuffled = Xs[perm]
            Ys_shuffled = Ys[perm]
            Ms_shuffled = Ms[perm]

            for i in range(0, Xs.size(0), self.batch_size):
                xb = Xs_shuffled[i:i + self.batch_size]
                yb = Ys_shuffled[i:i + self.batch_size]
                mb = Ms_shuffled[i:i + self.batch_size]

                v, _ = self.net(xb)
                loss = ((v - yb) ** 2)[mb].mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            score = evaluate(self, val).mean()
            if score > best_score:
                best_score = score
                best_state_dict = copy.deepcopy(self.net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.net.load_state_dict(best_state_dict)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xn = self.x_scaler.transform(X)
        X_tensor = torch.tensor(Xn, dtype=torch.float32).unsqueeze(0)  # (1, T, N)
        with torch.no_grad():
            v, _ = self.net(X_tensor)
        v_np = v[0].numpy()  # (T, 2)
        return self.y_scaler.inverse(v_np)


# Fixed settings, no hyperparameter hunting (see the project plan).
GRU_PARAMS = {"hidden": 64}
GRU_SEEDS = [0, 1, 2]


if __name__ == "__main__":
    sess = load_session()
    lag, _ = search_lag(sess)
    splits = causal_split(sess, lag_bins=lag)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"lag = {lag} bins ({lag * sess.bin_ms} ms)\n")

    candidates = {
        "single-bin ridge": (SingleBinRidge, {"alpha": [1.0]}),
        "Wiener": (Wiener, WIENER_GRID),
        "Kalman vel": (Kalman, {"steady_state": [False]}),
        "Kalman pos+vel": (Kalman, {"steady_state": [False], "use_pos": [True]}),
        "Kalman p+v steady": (Kalman, {"steady_state": [True], "use_pos": [True]}),
    }

    print(f"{'decoder':18s} {'R2 x':>6s} {'R2 y':>6s} {'mean':>6s}  best params (on val)")
    for name, (make, grid) in candidates.items():
        try:
            dec, params, _ = tune(make, grid, tr, va)
        except NotImplementedError:
            print(f"{name:18s} not implemented yet")
            continue
        rx, ry = evaluate(dec, te)
        print(f"{name:18s} {rx:6.3f} {ry:6.3f} {(rx + ry) / 2:6.3f}  {params}")

    # GRU: fixed settings, 3 seeds, early stopping on val, mean ± std on test.
    scores = []
    try:
        for seed in GRU_SEEDS:
            dec = GRUDecoder(seed=seed, **GRU_PARAMS).fit(tr, va)
            scores.append(evaluate(dec, te))
    except NotImplementedError:
        print(f"{'GRU':18s} not implemented yet")
    if scores:
        s = np.array(scores)  # (n_seeds, 2)
        m, sd = s.mean(0), s.std(0)
        mm, msd = s.mean(1).mean(), s.mean(1).std()
        print(f"{'GRU':18s} {m[0]:6.3f} {m[1]:6.3f} {mm:6.3f}  ± {msd:.3f} over {len(s)} seeds "
              f"(x ± {sd[0]:.3f}, y ± {sd[1]:.3f})  {GRU_PARAMS}")
