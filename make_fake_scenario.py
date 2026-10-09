#!/usr/bin/env python3
"""
make_fake_scenario.py - synthetic historian data for a polymer waste tank.

Hidden "truth" built into the data (the analysis script does NOT know this):

  CAUSE    XV-201 / FIT-201  Reactor dump valve + dump-line flow to the waste tank.
                             Each startup/shutdown dump carries residual initiator +
                             monomer. Polymer buildup accelerates ~14-40 h later
                             (peak ~20 h), scaled by a random per-dump "activity".
  RESPONSE M-301_AMPS        Waste-tank agitator motor amps. Rises as buildup
                             accumulates; drops back at each tank cleanout.
  SYMPTOM  PIT-301           Recirc pump discharge pressure. Rises ~8 h AFTER the
                             buildup (lags the response -> should show negative lag).
  DECOYS   LIT-101           Feed tank level - slow random walk, unrelated.
           TIT-301           Tank temperature - seasonal drift + daily cycle, unrelated.

Also includes a 30 h PIT-301 instrument outage (NaNs) to exercise gap handling.
"""
import argparse
import numpy as np
import pandas as pd


def ar1(n, phi, sigma, rng):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def main(out="fake_waste_tank.csv", days=150, seed=7):
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2026-03-01", periods=n, freq="h")

    # --- Startup/shutdown dumps (XV-201 opens for 2-4 h every ~2.5-6 days)
    xv = np.zeros(n)
    activity = np.zeros(n)
    t = 48
    while t < n - 100:
        dur = rng.integers(2, 5)
        xv[t:t + dur] = 1
        activity[t:t + dur] = rng.lognormal(0, 0.5)  # how "live" this dump was
        t += rng.integers(60, 140)

    # --- FIT-201 dump-line flow (m3/h): small rinse flow + dump flow
    fit = 0.8 + ar1(n, 0.6, 0.15, rng) + xv * rng.normal(9, 1.5, n).clip(4)
    fit = fit.clip(0)

    # --- Delayed fouling response kernel (zero for 14 h, peaks ~20 h, tails to ~40 h)
    k = np.arange(73, dtype=float)
    s = (k - 14).clip(0)
    kernel = s ** 2 * np.exp(-s / 3)
    kernel /= kernel.sum()

    dump_vol = (fit - 0.8).clip(0) * xv
    growth = 0.004 + 0.08 * np.convolve(dump_vol * activity, kernel)[:n]
    growth += rng.normal(0, 0.003, n)                    # process noise
    growth = growth.clip(0)

    # --- Buildup state with cleanouts every 18-24 days
    F = np.zeros(n)
    next_clean = rng.integers(18, 25) * 24
    for t in range(1, n):
        if t == next_clean:
            F[t] = 0.3
            next_clean += rng.integers(18, 25) * 24
        else:
            F[t] = F[t - 1] + growth[t]

    amps = 40 + F + ar1(n, 0.9, 0.01, rng) + rng.normal(0, 0.03, n)

    # --- PIT-301: symptom, follows buildup with an 8 h delay
    F_lag8 = np.r_[np.zeros(8), F[:-8]]
    pit = 310 + 1.8 * F_lag8 + ar1(n, 0.85, 0.3, rng)
    pit[2000:2030] = np.nan                              # instrument outage

    # --- Decoys
    lit = 55 + np.cumsum(rng.normal(0, 0.6, n))
    lit = np.clip(lit, 10, 95)
    hours = np.arange(n)
    tit = (21 + 5 * np.sin(2 * np.pi * hours / (24 * 120))
           + 1.5 * np.sin(2 * np.pi * hours / 24) + ar1(n, 0.95, 0.15, rng))

    df = pd.DataFrame({
        "timestamp": idx,
        "LIT-101": lit.round(2),
        "PIT-301": np.round(pit, 2),
        "FIT-201": fit.round(3),
        "XV-201": xv.astype(int),
        "TIT-301": tit.round(2),
        "M-301_AMPS": amps.round(3),
    })
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} rows, {int((np.diff(xv) > 0).sum())} dump events")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake_waste_tank.csv")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.out, a.days, a.seed)
