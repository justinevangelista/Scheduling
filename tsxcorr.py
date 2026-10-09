#!/usr/bin/env python3
"""
tsxcorr.py - Lagged cross-correlation with prewhitening for process historian data.

Pipeline
  1. Load CSV, resample to a common interval (default 1 h).
  2. Response (fouling proxy, e.g. agitator amps): converted to a RATE (first
     difference). Cleanout resets (large negative jumps) are auto-detected and
     all tags are masked +/- --reset-pad steps around them, so the sawtooth
     doesn't create fake correlations.
  3. Each candidate tag: made stationary (ADF + KPSS -> difference if needed).
  4. Prewhitening: fit an AR model to the tag (order by BIC), filter BOTH the tag
     and the response with that same filter, then cross-correlate the residuals.
     This removes autocorrelation so the +/-1.96/sqrt(N) bands are valid.
  5. CCF over lags -K..+K. Convention: r(k) = corr(x[t], y[t+k]).
        peak at k > 0  -> tag LEADS response by k   (candidate cause)
        peak at k < 0  -> tag LAGS response by |k|  (likely symptom)
  6. Granger causality F-test (does the tag's past improve prediction of the
     response beyond the response's own past?) at the CCF peak lag.
  7. Optional event-triggered average: align the response on every event
     (e.g. dump-valve openings) with a permutation test against random times.
  8. Also computes the "naive" CCF (raw levels, no prewhitening) for comparison,
     to show how misleading it can be.

Outputs (in --out): ranking.csv, overview.png, ccf_grid.png, event_triggered.png

Usage
  python tsxcorr.py data.csv --response M-301_AMPS --event-tag XV-201
  python tsxcorr.py data.csv --response M-301_AMPS --tags FIT-201 PIT-301 --max-lag 120
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import lfilter
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.ar_model import ar_select_order, AutoReg

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data prep
def load(csv, time_col, freq):
    df = pd.read_csv(csv)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.resample(freq).mean()


def detect_resets(y, z=8.0, pad=2):
    """Flag large negative jumps (cleanouts). Returns bool mask over the diff series."""
    d = y.diff()
    med = d.median()
    mad = 1.4826 * (d - med).abs().median()
    resets = d < med - z * max(mad, 1e-12)
    mask = resets.copy()
    for s in range(1, pad + 1):
        mask |= resets.shift(s, fill_value=False) | resets.shift(-s, fill_value=False)
    return mask, y.index[resets.values]


def pad_mask(index, times, pad):
    """Bool mask True within +/-pad steps of each time in `times`."""
    m = np.zeros(len(index), bool)
    for t in times:
        i = index.get_loc(t)
        m[max(0, i - pad):i + pad + 1] = True
    return m


def make_stationary(s, alpha=0.05, max_d=2):
    """Difference until the series looks stationary by BOTH tests:
    ADF rejects a unit root AND KPSS does not reject level-stationarity.
    Using both catches drifting/sawtooth tags that ADF alone lets through."""
    out, d = s.copy(), 0
    while d < max_d:
        clean = out.dropna()
        if len(clean) < 50 or clean.std() == 0:
            break
        try:
            p_adf = adfuller(clean, autolag="AIC")[1]
            p_kpss = kpss(clean, regression="c", nlags="auto")[1]
        except Exception:
            break
        if p_adf < alpha and p_kpss > alpha:
            break
        out, d = out.diff(), d + 1
    return out, d


# ----------------------------------------------------------------------------- prewhitening
def _propagate_nan(invalid, p):
    """A filtered value at t is invalid if any input t-p..t was invalid."""
    if p == 0:
        return invalid
    return pd.Series(invalid.astype(int)).rolling(p + 1, min_periods=1).max().values.astype(bool)


def prewhiten(x, y, max_ar=48):
    """Fit AR(p) to x, apply the same filter to x and y. Returns xw, yw, p."""
    xc = (x - x.mean()).values
    yc = (y - y.mean()).values
    invalid = np.isnan(xc) | np.isnan(yc)
    xf = np.nan_to_num(xc)
    yf = np.nan_to_num(yc)

    sel = ar_select_order(xf, maxlag=max_ar, ic="bic", trend="n")
    lags = sel.ar_lags or []
    p = max(lags) if lags else 0
    phi = np.zeros(p)
    if p:
        res = AutoReg(xf, lags=lags, trend="n").fit()
        for lag, c in zip(lags, res.params):
            phi[lag - 1] = c
    a = np.r_[1.0, -phi]
    xw = lfilter(a, [1.0], xf)
    yw = lfilter(a, [1.0], yf)

    bad = _propagate_nan(invalid, p)
    bad[:p] = True
    xw[bad] = np.nan
    yw[bad] = np.nan
    return xw, yw, p


# ----------------------------------------------------------------------------- CCF
def ccf(x, y, max_lag):
    """r(k) = corr(x[t], y[t+k]) for k = -max_lag..max_lag, with pairwise NaN removal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.full(len(lags), np.nan)
    nk = np.zeros(len(lags), int)
    for i, k in enumerate(lags):
        if k >= 0:
            a, b = x[:n - k], y[k:]
        else:
            a, b = x[-k:], y[:n + k]
        m = ~(np.isnan(a) | np.isnan(b))
        nk[i] = m.sum()
        if nk[i] > 10 and a[m].std() > 0 and b[m].std() > 0:
            r[i] = np.corrcoef(a[m], b[m])[0, 1]
    return lags, r, nk


def granger_p(y, x, lag):
    """p-value of the F-test that x's past (up to `lag`) helps predict y."""
    data = pd.concat([y, x], axis=1).dropna()
    data = data - data.mean()
    if len(data) < 5 * lag + 20 or lag < 1:
        return np.nan
    try:
        res = grangercausalitytests(data.values, maxlag=[lag], verbose=False)
    except TypeError:
        res = grangercausalitytests(data.values, maxlag=[lag])
    return res[lag][0]["ssr_ftest"][1]


# ----------------------------------------------------------------------------- event-triggered
def event_triggered(y, events_bool, pre, post, n_perm=2000, win=(0, None), seed=0):
    """Average response around event onsets + permutation test vs random times."""
    yv = y.values
    onsets = np.where(np.diff(events_bool.astype(int)) > 0)[0] + 1
    onsets = onsets[(onsets - pre >= 0) & (onsets + post < len(yv))]
    offs = np.arange(-pre, post + 1)
    M = np.array([yv[o + offs] for o in onsets])
    mean = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), axis=0))

    lo, hi = win[0], (post if win[1] is None else win[1])
    sel = (offs >= lo) & (offs <= hi)
    base = offs < 0

    def stat(onset_idx):
        W = np.array([yv[o + offs] for o in onset_idx])
        return np.nanmean(W[:, sel]) - np.nanmean(W[:, base])

    obs = stat(onsets)
    rng = np.random.default_rng(seed)
    valid = np.arange(pre, len(yv) - post)
    null = np.array([stat(rng.choice(valid, len(onsets), replace=False)) for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return offs, mean, se, len(onsets), obs, p


# ----------------------------------------------------------------------------- main
def run(args):
    os.makedirs(args.out, exist_ok=True)
    df = load(args.csv, args.time_col, args.freq)
    resp = args.response
    tags = args.tags or [c for c in df.columns if c != resp]
    step_h = pd.Timedelta(args.freq) / pd.Timedelta("1h")

    # --- response -> rate, mask cleanout resets
    y_raw = df[resp]
    reset_mask, reset_times = detect_resets(y_raw, z=args.reset_z, pad=2)
    y = y_raw.diff() if args.response_transform == "diff" else y_raw.copy()
    y[reset_mask] = np.nan
    # data around cleanouts is unreliable for every tag (tank drained, flushed, restarted)
    clean_mask = pad_mask(df.index, reset_times, args.reset_pad)
    y[clean_mask] = np.nan
    print(f"Response {resp}: transform={args.response_transform}, "
          f"{len(reset_times)} reset(s) masked, N={y.notna().sum()}")

    K = args.max_lag
    n_tests = len(tags) * (2 * K + 1)
    rows, curves = [], {}
    for tag in tags:
        x_raw = df[tag]
        if x_raw.dropna().std() == 0:
            continue
        x, d = make_stationary(x_raw.mask(clean_mask))
        xw, yw, p = prewhiten(x, y, max_ar=args.max_ar)
        lags, r, nk = ccf(xw, yw, K)
        band = 1.96 / np.sqrt(nk)
        lags_n, r_n, _ = ccf(x_raw.values, y_raw.values, K)     # naive, for comparison
        curves[tag] = (lags, r, band, lags_n, r_n)

        i = np.nanargmax(np.abs(r))
        k_pk, r_pk = int(lags[i]), r[i]
        z = r_pk * np.sqrt(nk[i])
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_adj = min(1.0, p_raw * n_tests)                      # Bonferroni across tags*lags
        if p_adj >= 0.05:
            role = "not significant"
        elif k_pk > 0:
            role = "LEADS response (candidate cause)"
        elif k_pk < 0:
            role = "LAGS response (likely symptom)"
        else:
            role = "simultaneous"
        g_lag = max(k_pk + 2, 2) if k_pk > 0 else args.granger_default_lag
        g_p = granger_p(y, x, g_lag)

        rows.append({
            "tag": tag, "diff_order": d, "AR_order": p,
            "peak_lag_steps": k_pk, "peak_lag_hours": k_pk * step_h,
            "peak_r": round(r_pk, 3), "band_95": round(band[i], 3),
            "p_bonferroni": p_adj, "granger_lag": g_lag, "granger_p": g_p,
            "naive_max_|r|": round(np.nanmax(np.abs(r_n)), 3), "verdict": role,
        })

    rank = pd.DataFrame(rows)
    prio = {"LEADS response (candidate cause)": 0, "simultaneous": 1,
            "LAGS response (likely symptom)": 2, "not significant": 3}
    rank["_p"] = rank["verdict"].map(prio)
    rank["_r"] = -rank["peak_r"].abs()
    rank = rank.sort_values(["_p", "_r"]).drop(columns=["_p", "_r"])
    rank.to_csv(os.path.join(args.out, "ranking.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3g}".format):
        print("\n=== Ranked tags (prewhitened CCF) ===")
        print(rank[["tag", "peak_lag_hours", "peak_r", "band_95", "p_bonferroni",
                    "granger_p", "naive_max_|r|", "verdict"]].to_string(index=False))

    # --- plots
    plot_overview(df, resp, tags, reset_times, args)
    plot_ccf_grid(curves, rank, step_h, args)

    if args.event_tag:
        ev = (df[args.event_tag].fillna(0) > args.event_threshold).values
        pre, post = int(args.et_pre / step_h), int(args.et_post / step_h)
        offs, mean, se, n_ev, obs, p = event_triggered(y, ev, pre, post, n_perm=args.n_perm)
        plot_event(offs * step_h, mean, se, n_ev, obs, p, resp, args)
        print(f"\nEvent-triggered ({args.event_tag}, {n_ev} events): "
              f"post-minus-pre change in {resp} rate = {obs:.4g}, permutation p = {p:.4g}")
    print(f"\nOutputs written to {args.out}/")


def plot_overview(df, resp, tags, reset_times, args):
    cols = [resp] + tags
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 1.7 * len(cols)), sharex=True)
    for ax, c in zip(axes, cols):
        ax.plot(df.index, df[c], lw=0.6, color="C3" if c == resp else "C0")
        ax.set_ylabel(c, rotation=0, ha="right", fontsize=8)
        for t in reset_times:
            ax.axvline(t, color="k", ls=":", lw=0.8)
        if args.event_tag:
            on = df[args.event_tag].fillna(0) > args.event_threshold
            ax.fill_between(df.index, *ax.get_ylim(), where=on, color="orange", alpha=0.3, lw=0)
    axes[0].set_title("Historian overview  (orange = events, dotted = detected cleanouts/resets)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "overview.png"), dpi=130)
    plt.close(fig)


def plot_ccf_grid(curves, rank, step_h, args):
    order = [t for t in rank["tag"] if t in curves]
    fig, axes = plt.subplots(len(order), 2, figsize=(13, 2.3 * len(order)), squeeze=False)
    for row, tag in zip(axes, order):
        lags, r, band, lags_n, r_n = curves[tag]
        info = rank.set_index("tag").loc[tag]
        h = lags * step_h
        ax = row[0]
        ax.bar(lags_n * step_h, r_n, width=step_h, color="grey")
        ax.set_ylim(-1, 1)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{tag}: NAIVE (raw levels, no prewhitening)", fontsize=9)
        ax = row[1]
        ax.bar(h, r, width=step_h, color="C0")
        ax.plot(h, band, "r--", lw=0.8)
        ax.plot(h, -band, "r--", lw=0.8)
        ax.axvline(0, color="k", lw=0.5)
        ax.plot(info["peak_lag_hours"], info["peak_r"], "v", color="C3", ms=8)
        ax.set_title(f"{tag}: PREWHITENED  peak {info['peak_lag_hours']:+.0f} h, "
                     f"r={info['peak_r']:.2f}  -> {info['verdict']}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("lag (h)   [+ = tag leads response]")
    fig.suptitle(f"Cross-correlation vs {args.response} (rate)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ccf_grid.png"), dpi=130)
    plt.close(fig)


def plot_event(h, mean, se, n_ev, obs, p, resp, args):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h, mean, color="C3")
    ax.fill_between(h, mean - 1.96 * se, mean + 1.96 * se, color="C3", alpha=0.2)
    ax.axvline(0, color="k", ls="--")
    ax.axhline(np.nanmean(mean[h < 0]), color="grey", lw=0.8)
    ax.set_xlabel(f"hours from {args.event_tag} onset")
    ax.set_ylabel(f"{resp} rate")
    ax.set_title(f"Event-triggered average ({n_ev} events)   "
                 f"post-pre = {obs:.3g}, permutation p = {p:.3g}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "event_triggered.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--response", required=True, help="fouling proxy tag (e.g. agitator amps)")
    ap.add_argument("--tags", nargs="*", help="candidate tags (default: all other columns)")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--freq", default="1h", help="resample interval")
    ap.add_argument("--max-lag", type=int, default=96, help="max lag in resampled steps")
    ap.add_argument("--max-ar", type=int, default=48, help="max AR order for prewhitening")
    ap.add_argument("--response-transform", choices=["diff", "none"], default="diff")
    ap.add_argument("--reset-z", type=float, default=8.0, help="robust z for cleanout detection")
    ap.add_argument("--reset-pad", type=int, default=12,
                    help="steps masked on ALL tags around each detected cleanout/reset")
    ap.add_argument("--granger-default-lag", type=int, default=24)
    ap.add_argument("--event-tag", help="binary/threshold tag for event-triggered average")
    ap.add_argument("--event-threshold", type=float, default=0.5)
    ap.add_argument("--et-pre", type=float, default=24, help="hours before event")
    ap.add_argument("--et-post", type=float, default=72, help="hours after event")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--out", default="tsxcorr_results")
    run(ap.parse_args())
