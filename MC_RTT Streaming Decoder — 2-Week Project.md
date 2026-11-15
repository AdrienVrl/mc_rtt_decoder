# MC\_RTT Streaming Decoder — 2-Week Project

Nov 8, 2026 · @Adrien

## Goal

In 10 working days, decode finger velocity from motor-cortex spikes on MC\_RTT, compare three causal decoders, and run the best linear one as a fixed-point C++ real-time decoder.

**Question.** How much accuracy does a recurrent decoder gain over the classic Kalman filter on continuous reaching, and what does the Kalman filter cost to run in real time in fixed point?

**Target result, one line.** A results table (velocity R² for Wiener, Kalman, GRU), one figure of decoded vs true velocity, and a C++ decoder that matches Python within a stated tolerance at a measured latency in microseconds per bin.

**Out of scope.** FPGA, the compiler, hyperparameter hunting. Those belong to the larger project; this one must finish in two weeks.

## Data and setup

Use the MC\_RTT dataset from the Neural Latents Benchmark: one monkey reaching continuously to random targets, roughly 130 sorted units, with finger velocity as the decoding target.

- **Loading.** Download from DANDI and load with `nlb_tools` (it returns a pandas DataFrame of spikes and behaviour). Confirm the unit count and the name of the velocity field when you load it.
- **Binning.** 20 ms bins, spike counts kept as integers. Optionally compare 10 ms in one extra run.
- **Causal split.** First 70% of the session for training, next 10% for validation, last 20% for test. No shuffling: the decoder must predict the future from the past.
- **Lag.** Neural activity leads movement by roughly 100 ms; test lags of 0–200 ms on the validation set and keep the best one for all decoders.
- **Metric.** Velocity R² on the test set, per axis and averaged. Report the standard deviation over 3 seeds for the GRU.
- **Tools.** Python: numpy, scikit-learn, PyTorch, matplotlib. C++17: Eigen (for the double-precision reference), CMake, Catch2 or GoogleTest.

## Week 1: decoders in Python

Three decoders of increasing capacity, all strictly causal, all tuned only on the validation split.

1. **Wiener filter.** Ridge regression from the last 5–10 bins of spike counts to velocity. Tune the window and ridge penalty. This is the floor.
2. **Kalman filter.** State = 2-D velocity (add position or a constant term if it helps). Fit the dynamics A and observation model C by least squares on training data, with noise covariances W and Q from the residuals. Run it causally, then compute the steady-state gain K so inference becomes a fixed recurrence:

```latex
\hat{x}_t = (I - KC)A\,\hat{x}_{t-1} + K\,y_t
```

Check that the steady-state version loses almost no R² against the full filter; that is the version you port in week 2.

3. **Small GRU.** One layer, 64–128 hidden units, trained on sequences of a few seconds, run causally at test time. Use early stopping on the validation split and 3 seeds.

**Outputs of week 1.** A results table (decoder, R² x, R² y, mean), a 10-second plot of true vs decoded velocity for all three, and the exported Kalman matrices (A, C, K) plus test-set inputs and outputs as files for week 2.

**Expected pattern.** The GRU should beat the Kalman filter, which should beat or match Wiener. If not, check the lag and the split before anything else.

## Week 2: fixed-point C++ decoder

Port the steady-state Kalman decoder to C++ twice, in double and in fixed point, and prove both match Python.

**Implementation**

- A header-only `Decoder` class with one method, `step(const uint8_t* counts) -> Velocity`, called once per bin. No heap allocation inside `step`.
- Double version with Eigen as the reference. Fixed-point version with plain `int32_t` arithmetic and `int64_t` accumulators, formats as compile-time template parameters (e.g. Q16.16 for state, Q2.14 for matrix entries).
- Pick integer bits from the ranges seen on the training set (max absolute value × 1.25). Keep spike counts as integers: K·y needs no conversion.
- Fold M = (I − KC)A offline in Python, so the recurrence is just x ← M x + K y.

**Validation**

- Python exports the test-set spike counts and the double-precision decoded velocity.
- Unit tests check the C++ double version against Python to \~1e-9, and the fixed-point version against a small numpy integer emulation bit-exactly.
- Report R² of the fixed-point decoder vs double, and the R² loss as you sweep fractional bits (8 to 20). That sweep is a nice extra figure.

**Latency**

- Time `step` with `std::chrono` over the full test set, after warm-up, in a release build. Report median and 99th percentile in µs per bin, and the machine.
- Compare against the 20 ms bin period: the point is the margin, and how it would shrink on a microcontroller. If you have a Cortex-M board, running the same code on it is an optional bonus.

## Day-by-day plan

Ten working days. If a day slips, cut the GRU seeds or the bit-width sweep, never the validation tests.

- [ ] Day 1: environment, download MC\_RTT, load with `nlb_tools`, plot raster and velocity to understand the data
- [ ] Day 2: binning, causal split, lag search, Wiener filter baseline
- [ ] Day 3: Kalman filter (fit, causal run, steady-state gain), compare full vs steady-state
- [ ] Day 4: GRU training, 3 seeds, early stopping
- [ ] Day 5: results table, velocity figure, export matrices and test data for C++
- [ ] Day 6: CMake project, double-precision Eigen decoder, test against Python
- [ ] Day 7: fixed-point decoder and numpy integer emulation
- [ ] Day 8: bit-exact tests pass, R² of fixed point vs double, fractional-bit sweep
- [ ] Day 9: latency benchmark, optional microcontroller run
- [ ] Day 10: README, clean notebooks, final figures, CV line

## Deliverables

A public repository that a reviewer understands in two minutes from the README alone.

**Repository layout**

```
mc-rtt-decoder/
  README.md            question, results table, figures, how to reproduce
  python/
    data.py            loading, binning, causal split
    decoders.py        Wiener, Kalman, GRU
    export.py          matrices + test data for C++
    notebooks/01_results.ipynb
  cpp/
    include/decoder.hpp   double and fixed-point decoders
    tests/                Catch2 or GoogleTest
    bench/latency.cpp
    CMakeLists.txt
  figures/
```

**README contents.** The question in two sentences. The results table. The velocity figure. The fixed-point R² vs bits figure. Latency (median, p99, machine). One paragraph on next steps: streaming SSM decoders and an FPGA deployment.

**CV line**

> Streaming motor-BCI decoding (Neural Latents Benchmark, MC\_RTT): compared Wiener, Kalman and GRU decoders on causal velocity decoding; ported the Kalman decoder to fixed-point C++ (bit-exact vs Python, X µs/bin).

Fill in the R² and latency numbers once you have them; concrete numbers are what make the line credible.
