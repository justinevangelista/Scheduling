"""
Chemical tank level forecaster.

Takes the current level (kg) of each tank, consumes it continuously at a fixed
rate and/or in batches, and tops it up whenever it reaches the lower cushion. Plots one panel per
chemical with LSL / LSH alarm lines and the lower / upper cushion band.

Level ordering expected per tank:
    0 <= LSL < lower_cushion < upper_cushion <= LSH <= capacity

Top-up behaviour:
    - When the level falls to the lower cushion a top-up is triggered.
    - The delivery arrives after `lead_time_h` hours (0 = instantaneous).
    - If `delivery_kg` is None the tank is filled to the upper cushion,
      otherwise exactly `delivery_kg` is added (an overfill above LSH is flagged).

Consumption (both can be combined):
    - Continuous: `rate_kg_h` kg every hour (0 for batch-only chemicals).
    - Batches: `batch_kg` per batch, repeating every `batch_every_h` hours starting
      `first_batch_h` hours from now, and/or at explicit `batch_times_h`.
      `batch_duration_h` = 0 takes the batch instantly (step drop), otherwise it is
      drawn evenly over that many hours.

Downtimes (no consumption, deliveries still arrive):
    - Plant-wide: DOWNTIMES below and/or --downtime on the command line.
    - Per chemical: the `downtimes` field.
    Each downtime is (start, end), where start/end are either hours from now
    (e.g. 48) or a date string "YYYY-MM-DD HH:MM". Any batch overlapping a
    downtime loses the overlapping part.

Usage:
    python main.py                                # uses CHEMICALS below
    python main.py --level Caustic=820 --level HCl=300 --hours 336
    python main.py --config tanks.json --save forecast.png
    python main.py --downtime 48-72 --downtime "2026-10-10 06:00/2026-10-11 18:00"

tanks.json is a list of objects with the same fields as `Chemical`, e.g.
    [{"name": "Caustic", "level_kg": 820, "rate_kg_h": 4.5, "lsl": 150,
      "lower_cushion": 300, "upper_cushion": 1600, "lsh": 1800, "capacity": 2000},
     {"name": "Coagulant", "level_kg": 900, "rate_kg_h": 0, "batch_kg": 150,
      "batch_every_h": 12, "batch_duration_h": 2, "first_batch_h": 3, "lsl": 100,
      "lower_cushion": 250, "upper_cushion": 1200, "lsh": 1400,
      "downtimes": [["2026-10-08 06:00", "2026-10-09 06:00"]]}]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration – edit these or pass --config / --level on the command line
# --------------------------------------------------------------------------- #
@dataclass
class Chemical:
    name: str
    level_kg: float            # current tank amount
    rate_kg_h: float           # continuous consumption rate
    lsl: float                 # level switch low (alarm)
    lower_cushion: float       # top-up trigger
    upper_cushion: float       # top-up target
    lsh: float                 # level switch high (alarm)
    capacity: float | None = None
    supplier: str = ""
    delivery_kg: float | None = None   # None -> fill to upper cushion
    lead_time_h: float = 0.0
    # batch consumption
    batch_kg: float = 0.0
    batch_every_h: float | None = None        # repeat interval; None = no repeating batches
    first_batch_h: float | str = 0.0          # first repeating batch: hours from now or "YYYY-MM-DD HH:MM"
    batch_duration_h: float = 0.0             # 0 = instant draw
    batch_times_h: list = field(default_factory=list)  # extra one-off batch starts (hours or dates)
    # chemical-specific downtimes: [(start, end), ...] as hours from now or "YYYY-MM-DD HH:MM"
    downtimes: list = field(default_factory=list)

    def batch_starts(self, horizon_h: float) -> list[float]:
        starts = list(self.batch_times_h)
        if self.batch_kg > 0 and self.batch_every_h:
            t = self.first_batch_h
            while t < horizon_h:
                starts.append(t)
                t += self.batch_every_h
        return sorted(s for s in starts if 0 <= s < horizon_h)

    def describe_use(self) -> str:
        parts = []
        if self.rate_kg_h > 0:
            parts.append(f"{self.rate_kg_h:g} kg/h")
        if self.batch_kg > 0 and (self.batch_every_h or self.batch_times_h):
            b = f"{self.batch_kg:g} kg batches"
            if self.batch_every_h:
                b += f" every {self.batch_every_h:g} h"
            if self.batch_duration_h > 0:
                b += f" over {self.batch_duration_h:g} h"
            parts.append(b)
        return " + ".join(parts) or "no consumption"

    def validate(self) -> None:
        cap = self.capacity if self.capacity is not None else float("inf")
        if not (0 <= self.lsl < self.lower_cushion < self.upper_cushion <= self.lsh <= cap):
            raise ValueError(
                f"{self.name}: levels must satisfy "
                f"0 <= LSL < lower cushion < upper cushion <= LSH <= capacity "
                f"(got {self.lsl}, {self.lower_cushion}, {self.upper_cushion}, {self.lsh}, {self.capacity})"
            )
        if self.rate_kg_h < 0:
            raise ValueError(f"{self.name}: consumption rate must be >= 0")
        if self.delivery_kg is not None and self.delivery_kg <= 0:
            raise ValueError(f"{self.name}: delivery_kg must be > 0")
        if self.lead_time_h < 0:
            raise ValueError(f"{self.name}: lead_time_h must be >= 0")
        if self.batch_kg < 0 or self.batch_duration_h < 0:
            raise ValueError(f"{self.name}: batch_kg and batch_duration_h must be >= 0")
        if self.batch_every_h is not None and self.batch_every_h <= 0:
            raise ValueError(f"{self.name}: batch_every_h must be > 0")
        if self.batch_every_h and self.batch_duration_h > self.batch_every_h:
            raise ValueError(f"{self.name}: batch_duration_h longer than batch_every_h")


CHEMICALS: list[Chemical] = [
    Chemical("Caustic (NaOH)", level_kg=820, rate_kg_h=9.0, lsl=150, lower_cushion=300,
             upper_cushion=1600, lsh=1800, capacity=2000, lead_time_h=12),
    Chemical("Hydrochloric acid", level_kg=450, rate_kg_h=5.0, lsl=80, lower_cushion=200,
             upper_cushion=900, lsh=1000, capacity=1100, delivery_kg=600),
    Chemical("Sodium hypochlorite", level_kg=1300, rate_kg_h=12.0, lsl=200, lower_cushion=400,
             upper_cushion=1700, lsh=1900, capacity=2100, lead_time_h=24),
    # batch-only: 150 kg drawn over 2 h, every 12 h, first batch in 3 h
    Chemical("Coagulant", level_kg=900, rate_kg_h=0, lsl=100, lower_cushion=250,
             upper_cushion=1200, lsh=1400, capacity=1500, lead_time_h=8,
             batch_kg=150, batch_every_h=12, batch_duration_h=2, first_batch_h=3,
             downtimes=[(200, 236)]),  # coagulant-only stop
    # small continuous dose plus an instant 60 kg shot every 8 h
    Chemical("Antiscalant", level_kg=400, rate_kg_h=0.8, lsl=40, lower_cushion=100,
             upper_cushion=500, lsh=560, capacity=600,
             batch_kg=60, batch_every_h=8, first_batch_h=1),
]

HORIZON_H = 24 * 14  # two weeks
STEP_MIN = 1.0       # simulation time step in minutes

# Plant-wide downtimes (apply to every chemical): (start, end) as hours from now
# or "YYYY-MM-DD HH:MM" strings.
DOWNTIMES: list = [
    (96, 120),   # e.g. 24 h shutdown starting 4 days from now
]


def to_hours(value, start: datetime) -> float:
    """Hours from `start` for a number (already hours) or a date string."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return (datetime.fromisoformat(value.strip()) - start).total_seconds() / 3600


def resolve_downtimes(windows, start: datetime) -> list[tuple[float, float]]:
    out = []
    for w in windows:
        a, b = to_hours(w[0], start), to_hours(w[1], start)
        if b <= a:
            raise ValueError(f"downtime {w}: end must be after start")
        out.append((a, b))
    return sorted(out)


def parse_downtime_arg(text: str) -> tuple[str, str]:
    """'48-72' (hours) or 'YYYY-MM-DD HH:MM/YYYY-MM-DD HH:MM'."""
    sep = "/" if "/" in text else "-"
    a, _, b = text.partition(sep)
    if not b:
        raise argparse.ArgumentTypeError(f"bad --downtime '{text}' (use 48-72 or 'start/end')")
    return a.strip(), b.strip()


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    chem: Chemical
    times_h: list[float] = field(default_factory=list)
    levels: list[float] = field(default_factory=list)
    topups: list[tuple[float, float, float]] = field(default_factory=list)  # (t, before, after)
    orders_h: list[float] = field(default_factory=list)
    hours_below_lsl: float = 0.0
    hours_empty: float = 0.0
    overfills: list[float] = field(default_factory=list)  # times level exceeded LSH
    downtimes: list[tuple[float, float]] = field(default_factory=list)


def consumption_per_step(c: Chemical, horizon_h: float, dt: float, n: int,
                         downtimes: list[tuple[float, float]] = ()) -> list[float]:
    """kg drawn during each step [i*dt, (i+1)*dt): continuous rate plus batch overlap,
    with nothing drawn during downtimes."""
    cons = [c.rate_kg_h * dt] * n
    instant = []
    for s in c.batch_starts(horizon_h):
        if c.batch_duration_h == 0:
            instant.append(s)
            continue
        e = s + c.batch_duration_h
        rate = c.batch_kg / c.batch_duration_h
        for i in range(int(s / dt), min(math.ceil(e / dt), n)):
            overlap = min(e, (i + 1) * dt) - max(s, i * dt)
            if overlap > 0:
                cons[i] += rate * overlap

    # scale each step by the fraction of it that is not in a downtime
    for a, b in downtimes:
        for i in range(max(0, int(a / dt)), min(math.ceil(b / dt), n)):
            down = min(b, (i + 1) * dt) - max(a, i * dt)
            if down > 0:
                cons[i] *= max(0.0, 1 - down / dt)

    # instant batches are skipped entirely if they fall in a downtime
    for s in instant:
        if not any(a <= s < b for a, b in downtimes):
            cons[min(int(s / dt), n - 1)] += c.batch_kg
    return cons


def simulate(c: Chemical, horizon_h: float, step_min: float = STEP_MIN,
             downtimes: list[tuple[float, float]] = ()) -> Result:
    """Fixed time-step simulation (handles continuous and batch draws)."""
    dt = step_min / 60.0
    n = max(1, math.ceil(horizon_h / dt))
    cons = consumption_per_step(c, horizon_h, dt, n, downtimes)

    r = Result(c, downtimes=list(downtimes))
    level = float(c.level_kg)
    arrival: float | None = None

    def check_and_deliver(t: float) -> None:
        nonlocal level, arrival
        if arrival is None and level <= c.lower_cushion:
            arrival = t + c.lead_time_h
            r.orders_h.append(t)
        if arrival is not None and t >= arrival - 1e-9:
            before = level
            level = c.upper_cushion if c.delivery_kg is None else level + c.delivery_kg
            if c.capacity is not None:
                level = min(level, c.capacity)
            if level > c.lsh:
                r.overfills.append(t)
            r.topups.append((t, before, level))
            r.times_h.append(t)
            r.levels.append(level)
            arrival = None

    r.times_h.append(0.0)
    r.levels.append(level)
    check_and_deliver(0.0)

    for i in range(n):
        t1 = min((i + 1) * dt, horizon_h)
        step = t1 - i * dt
        demand = cons[i]
        if demand > level:  # tank runs dry part-way through the step
            r.hours_empty += step * (1 - level / demand)
            level = 0.0
        else:
            level -= demand
        if level < c.lsl:
            r.hours_below_lsl += step
        r.times_h.append(t1)
        r.levels.append(level)
        check_and_deliver(t1)

    return r


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
LEVEL = "#2a78d6"
CRITICAL, WARNING = "#d03b3b", "#e39b00"


def plot(results: list[Result], start: datetime, horizon_h: float, save: str | None) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)
    to_dt = lambda h: start + timedelta(hours=h)
    end = to_dt(horizon_h)

    for ax, r in zip(axes, results):
        c = r.chem
        x = [to_dt(h) for h in r.times_h]
        ax.set_facecolor(SURFACE)

        # operating band between cushions
        ax.axhspan(c.lower_cushion, c.upper_cushion, color=LEVEL, alpha=0.06, lw=0)

        # downtime windows (no consumption)
        for a, b in r.downtimes:
            if b > 0 and a < horizon_h:
                ax.axvspan(to_dt(max(a, 0)), to_dt(min(b, horizon_h)), facecolor=MUTED,
                           alpha=0.15, hatch="//", edgecolor=AXIS, lw=0, zorder=0.5)

        # threshold lines
        lines = [
            (c.lsh, "LSH", CRITICAL, (0, (5, 3))),
            (c.upper_cushion, "Upper cushion", WARNING, (0, (2, 2))),
            (c.lower_cushion, "Lower cushion", WARNING, (0, (2, 2))),
            (c.lsl, "LSL", CRITICAL, (0, (5, 3))),
        ]
        for y, label, col, ls in lines:
            ax.axhline(y, color=col, lw=1.2, ls=ls, zorder=2)
            ax.annotate(f"{label}  {y:,.0f}", xy=(1, y), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK_2)

        # level trace
        ax.fill_between(x, r.levels, 0, color=LEVEL, alpha=0.12, lw=0, zorder=1)
        ax.plot(x, r.levels, color=LEVEL, lw=2, zorder=3)

        # shade any time below LSL
        ax.fill_between(x, r.levels, c.lsl, where=[v < c.lsl for v in r.levels],
                        interpolate=True, color=CRITICAL, alpha=0.25, lw=0, zorder=2)

        # top-up markers
        if r.topups:
            ax.scatter([to_dt(t) for t, _, a in r.topups], [a for _, _, a in r.topups],
                       marker="^", s=55, color=LEVEL, edgecolor=SURFACE, lw=1.5, zorder=4)
        # order (trigger) markers when lead time > 0
        if c.lead_time_h > 0 and r.orders_h:
            ax.scatter([to_dt(t) for t in r.orders_h], [c.lower_cushion] * len(r.orders_h),
                       marker="o", s=40, facecolor=SURFACE, edgecolor=LEVEL, lw=1.5, zorder=4)

        top = c.capacity if c.capacity is not None else c.lsh * 1.1
        ax.set_ylim(0, max(top, max(r.levels)) * 1.04)
        ax.set_xlim(start, end)
        ax.set_ylabel("kg", color=MUTED, fontsize=9)

        sub = f"{c.describe_use()}  ·  start {c.level_kg:,.0f} kg  ·  {len(r.topups)} top-up(s)"
        if r.hours_below_lsl > 0:
            sub += f"  ·  {r.hours_below_lsl:.1f} h below LSL"
        if r.overfills:
            sub += f"  ·  {len(r.overfills)} overfill(s) above LSH"
        ax.set_title(c.name, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.text(1, 1.02, sub, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=INK_2)

        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # real dates under every panel
    def date_fmt(x, _):
        d = mdates.num2date(x)
        return d.strftime("%a %d %b") if (d.hour, d.minute) == (0, 0) else d.strftime("%H:%M\n%d %b")

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(date_fmt))
        ax.tick_params(axis="x", labelbottom=True)

    # shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=LEVEL, lw=2, label="Tank level"),
        Line2D([], [], color=LEVEL, marker="^", ls="", ms=7, label="Top-up"),
        Line2D([], [], color=LEVEL, marker="o", mfc=SURFACE, ls="", ms=6, label="Order placed"),
        Line2D([], [], color=WARNING, ls=(0, (2, 2)), lw=1.2, label="Cushions"),
        Line2D([], [], color=CRITICAL, ls=(0, (5, 3)), lw=1.2, label="LSL / LSH"),
        Patch(facecolor=MUTED, alpha=0.3, hatch="//", edgecolor=AXIS, lw=0, label="Downtime"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=6, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))

    if save:
        fig.savefig(save, dpi=150, facecolor=SURFACE)
        print(f"Saved plot to {save}")
    else:
        plt.show()


def print_summary(results: list[Result], start: datetime) -> None:
    print(f"{'Chemical':<22}{'Now kg':>9}{'Next order':>18}{'Next top-up':>18}{'Top-ups':>9}{'h<LSL':>8}")
    for r in results:
        c = r.chem
        fmt = lambda h: (start + timedelta(hours=h)).strftime("%a %d %b %H:%M")
        nxt_o = fmt(r.orders_h[0]) if r.orders_h else "-"
        nxt_t = fmt(r.topups[0][0]) if r.topups else "-"
        print(f"{c.name[:21]:<22}{c.level_kg:>9,.0f}{nxt_o:>18}{nxt_t:>18}"
              f"{len(r.topups):>9}{r.hours_below_lsl:>8.1f}")
        if r.hours_empty > 0:
            print(f"  ! {c.name} runs EMPTY for {r.hours_empty:.1f} h - increase lead time margin or lower cushion")
        if r.overfills:
            print(f"  ! {c.name} delivery overfills above LSH {len(r.overfills)} time(s)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Forecast chemical tank levels with top-ups.")
    p.add_argument("--config", help="JSON file with a list of chemicals (overrides CHEMICALS)")
    p.add_argument("--level", action="append", default=[], metavar="NAME=KG",
                   help="override current level for a chemical (repeatable, name prefix match)")
    p.add_argument("--downtime", action="append", default=[], type=parse_downtime_arg,
                   metavar="START-END", help="plant-wide downtime, hours from now ('48-72') or "
                   "dates ('2026-10-10 06:00/2026-10-11 18:00'); repeatable")
    p.add_argument("--hours", type=float, help=f"forecast horizon in hours (default {HORIZON_H})")
    p.add_argument("--step-min", type=float, help=f"simulation step in minutes (default {STEP_MIN:g})")
    p.add_argument("--save", help="save the plot to this file instead of showing it")
    args = p.parse_args()

    plant_cfg = list(DOWNTIMES)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        # either a plain list of chemicals, or the web page's export:
        # {"horizon_h": .., "step_min": .., "downtimes": [..], "chemicals": [..]}
        if isinstance(cfg, dict):
            plant_cfg = cfg.get("downtimes", [])
            if args.hours is None:
                args.hours = cfg.get("horizon_h")
            if args.step_min is None:
                args.step_min = cfg.get("step_min")
            cfg = cfg["chemicals"]
        chems = [Chemical(**d) for d in cfg]
    else:
        chems = [Chemical(**vars(c)) for c in CHEMICALS]
    args.hours = args.hours or HORIZON_H
    args.step_min = args.step_min or STEP_MIN

    for item in args.level:
        name, _, kg = item.partition("=")
        matches = [c for c in chems if c.name.lower().startswith(name.strip().lower())]
        if not matches:
            p.error(f"--level: no chemical matching '{name}'")
        for c in matches:
            c.level_kg = float(kg)

    start = datetime.now().replace(second=0, microsecond=0)
    for c in chems:
        c.validate()
        # batch times may be hours from now or date strings (as the web page saves them)
        c.first_batch_h = to_hours(c.first_batch_h, start)
        c.batch_times_h = [to_hours(v, start) for v in c.batch_times_h]

    plant_down = plant_cfg + args.downtime
    results = [simulate(c, args.hours, args.step_min,
                        resolve_downtimes(plant_down + list(c.downtimes), start))
               for c in chems]
    print_summary(results, start)
    plot(results, start, args.hours, args.save)


if __name__ == "__main__":
    main()
