"""Look at the CSI signal: live while recording, as a plot of finished blocks, or per subcarrier.

Three commands (run from SmartRoomSense/, see also `python -m src.collect --help`):

  live   watch the recording that logger.py is writing (reads its CSV, not the serial port)
         uv run python -m src.collect live                      # follows the newest recording
         uv run python -m src.collect live FILE.csv --calibrate 120
         uv run python -m src.collect live FILE.csv --snapshot view.png   # one picture, then exit
         Top plot: activity indicator over the last 60 s (higher = more change in the signal).
         Bottom plot: the 52 kept subcarriers over the last 30 s, each minus its own mean;
         movement shows up as streaks, an empty quiet room is flat.
         --calibrate S: treat the first S seconds as empty room and draw a threshold line.

  plot   activity of finished recordings, one panel per file on a shared axis (pilot step 2A.7)
         uv run python -m src.collect plot A.csv B.csv C.csv --out reports/figures/pilot.png
         It also prints the median and 90th percentile of each file after the first 5 s.

  map    mean amplitude per subcarrier position of one file (pilot step 2A.6)
         uv run python -m src.collect map FILE.csv
         For a 20 MHz channel the guide predicts empty positions at 0 and 27-37, which keeps
         1-26 and 38-63 (52 positions). Saves a plot and says whether the prediction holds.
"""
import argparse
import csv
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np

from src.collect.common import (CSV_COLUMNS, DATA_DIR, EXPECTED_LEN, FIGURES_DIR, KEPT_POSITIONS,
                                N_POSITIONS, SKIP_SECONDS, TIMESTAMP_WRAP, activity_series, amplitudes,
                                load_recording, threshold_from_baseline, unwrap_timestamp)

COL = {name: i for i, name in enumerate(CSV_COLUMNS)}
HISTORY_S = 60          # live: seconds kept for the activity line
HEATMAP_S = 30          # live: seconds shown in the heatmap
HEATMAP_BIN_S = 0.25    # live: heatmap column width
REFRESH_S = 0.5
START_TAIL_BYTES = 5_000_000   # live: when opening a big file, only read this much from its end
SKIP_NAMES = {"session_log.csv", "manifest.csv"}
PREDICTED_EMPTY = [0] + list(range(27, 38))  # map: 12 positions
N_LOWEST = 12


def get_plt(interactive=False):
    """matplotlib's pyplot. Everything except the live window draws to files only (no window)."""
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ================================ live ====================================================

def watched_dirs():
    raw = DATA_DIR / "raw"
    session_folders = [d for d in raw.glob("room*_d*_s*") if d.is_dir()] if raw.exists() else []
    return [DATA_DIR / "pilot", DATA_DIR / "tests", *session_folders]


def latest_csv():
    """The most recently modified recording CSV, or None."""
    files = [p for d in watched_dirs() if d.exists() for p in d.rglob("*.csv")
             if p.name not in SKIP_NAMES and "bad" not in p.parts]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


class CsvTail:
    """Reads the complete new rows of a CSV that another program is still writing."""

    def __init__(self, path, tail_bytes=START_TAIL_BYTES):
        self.path = Path(path)
        self.f = open(self.path, "r", newline="", encoding="utf-8")
        self.rest = ""
        size = self.path.stat().st_size
        if size > tail_bytes:                       # a big file: start near its end
            self.f.seek(size - tail_bytes)
            self.f.readline()                       # drop the partly read first line

    def read_rows(self):
        chunk = self.f.read()
        if not chunk:
            return []
        lines = (self.rest + chunk).split("\n")
        self.rest = lines[-1]                       # an unfinished last line waits for next time
        lines = [ln.rstrip("\r") for ln in lines[:-1] if ln.strip()]
        return [r for r in csv.reader(lines) if r and r[0] != CSV_COLUMNS[0]]

    def close(self):
        self.f.close()


def parse_row(fields):
    """One CSV row -> (id, local_timestamp, amplitude[64]) or None if it is not usable."""
    if len(fields) != len(CSV_COLUMNS):
        return None
    try:
        row_id = int(fields[COL["id"]])
        ts = int(fields[COL["local_timestamp"]])
        first_word = int(fields[COL["first_word"]])
        nums = np.fromstring(fields[COL["data"]].strip()[1:-1], sep=",", dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if len(nums) != EXPECTED_LEN:
        return None
    amp = amplitudes(nums[None, :])[0]
    if first_word == 1:
        amp[:2] = np.nan                            # the first 4 numbers are invalid
    return row_id, ts, amp


class LiveState:
    """Rows of the last HISTORY_S seconds, with a running time axis that survives the 32-bit wrap."""

    def __init__(self, calibrate_s=0):
        self.calibrate_s = calibrate_s
        self.reset()

    def reset(self):
        self.t = deque()
        self.amp = deque()
        self.offset = 0
        self.last_id = None
        self.last_ts = None
        self.t0 = None
        self.restarts = 0
        self.threshold = None
        self.calib_t, self.calib_amp = [], []

    def add(self, row_id, ts, amp):
        if self.last_id is not None and row_id < self.last_id:
            keep = self.restarts + 1                 # the board restarted: start the time axis again
            self.reset()
            self.restarts = keep
        elif self.last_ts is not None and ts < self.last_ts:
            self.offset += TIMESTAMP_WRAP            # 32-bit microsecond counter wrapped
        self.last_id, self.last_ts = row_id, ts
        if self.t0 is None:
            self.t0 = ts + self.offset
        t = (ts + self.offset - self.t0) / 1e6
        self.t.append(t)
        self.amp.append(amp)
        while self.t[0] < t - HISTORY_S:
            self.t.popleft()
            self.amp.popleft()
        if self.calibrate_s and self.threshold is None:   # collect the empty-room start
            if t <= self.calibrate_s:
                self.calib_t.append(t)
                self.calib_amp.append(amp)
            else:
                _, act = activity_series(np.array(self.calib_t), np.array(self.calib_amp))
                self.threshold = threshold_from_baseline(act)

    def arrays(self):
        return np.array(self.t), (np.array(self.amp) if self.amp else np.empty((0, 64)))


def heatmap_data(t, amp):
    """Last HEATMAP_S seconds -> (52 x bins) matrix, each subcarrier minus its own mean."""
    if len(t) < 2:
        return None
    sel = t >= t[-1] - HEATMAP_S
    t, kept = t[sel], amp[sel][:, KEPT_POSITIONS]
    bins = np.floor((t - t[0]) / HEATMAP_BIN_S).astype(int)
    cols = [np.nanmean(kept[bins == b], axis=0) for b in range(bins.max() + 1) if (bins == b).any()]
    if not cols:
        return None
    m = np.array(cols).T
    return m - np.nanmean(m, axis=1, keepdims=True)


def draw(fig, axes, state, title):
    """Redraw both live plots from the current state."""
    t, amp = state.arrays()
    ax1, ax2 = axes
    ax1.clear()
    ax2.clear()
    rate = int(((t > t[-1] - 1).sum())) if len(t) else 0
    now = None
    if len(t) > 1:
        tt, act = activity_series(t, amp)
        if len(tt):
            ax1.plot(tt - t[-1], act, color="#2a6fbb")
            valid = act[~np.isnan(act)]
            now = float(valid[-1]) if len(valid) else None
    if state.threshold is not None:
        ax1.axhline(state.threshold, color="#d1432f", linestyle="--", label="empty mean + 3 std")
        ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax1.set_xlim(-HISTORY_S, 0)
    ax1.set_ylabel("activity (amplitude std)")
    ax1.set_xlabel("seconds ago")
    status = ""
    if now is not None:
        status = f"   now {now:.2f}"
        if state.threshold is not None:
            status += "  ABOVE threshold" if now > state.threshold else "  below threshold"
    flag = f"   RESTARTS {state.restarts}" if state.restarts else ""
    ax1.set_title(f"{title}   {rate} rows/s{status}{flag}", fontsize=10)

    m = heatmap_data(t, amp)
    if m is not None:
        lim = max(float(np.nanpercentile(np.abs(m), 98)), 1e-6)
        ax2.imshow(m, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower",
                   extent=[-m.shape[1] * HEATMAP_BIN_S, 0, 0, m.shape[0]], interpolation="nearest")
    ax2.set_ylabel("subcarrier (52 kept)")
    ax2.set_xlabel("seconds ago (amplitude minus its mean; red/blue streaks = change)")
    fig.tight_layout()


def feed(state, tail):
    n = 0
    for fields in tail.read_rows():
        parsed = parse_row(fields)
        if parsed:
            state.add(*parsed)
            n += 1
    return n


def run_live(a):
    plt = get_plt(interactive=not a.snapshot)
    follow = a.path is None
    path = a.path or latest_csv()
    if path is None:
        if a.snapshot:
            sys.exit("No recording found. Give a file path.")
        print("Waiting for a recording CSV (start `record`)...")
        while path is None:
            time.sleep(1)
            path = latest_csv()
    if not Path(path).is_file():
        sys.exit(f"Not found: {path}")

    state = LiveState(a.calibrate)
    tail = CsvTail(path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 2]})
    feed(state, tail)

    if a.snapshot:
        draw(fig, axes, state, Path(path).name)
        a.snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.snapshot, dpi=120)
        print(f"Saved {a.snapshot}")
        return 0

    print(f"Following {path}. Close the window or press Ctrl+C to stop.")
    plt.ion()
    try:
        while plt.fignum_exists(fig.number):
            feed(state, tail)
            if follow:                                # a new block file started: switch to it
                newest = latest_csv()
                if newest is not None and newest != path and newest.stat().st_mtime > path.stat().st_mtime:
                    tail.close()
                    path, tail = newest, CsvTail(newest)
                    state = LiveState(a.calibrate)
                    print(f"Now following {path}")
            draw(fig, axes, state, Path(path).name)
            plt.pause(REFRESH_S)
    except KeyboardInterrupt:
        pass
    finally:
        tail.close()
    return 0


# ================================ plot (finished blocks) ===================================

def block_activity(path):
    """Return (t, activity) for one file, t in seconds from its first row."""
    df, amp, _ = load_recording(path)
    if len(df) < 2:
        raise ValueError("fewer than 2 good rows")
    ts = unwrap_timestamp(df["local_timestamp"].to_numpy(), df["id"].to_numpy())
    seconds = (ts - ts[0]) / 1e6
    return activity_series(seconds, amp)


def summary(t, act):
    """(median, 90th percentile) of the activity after the first SKIP_SECONDS, or None."""
    values = act[(t >= SKIP_SECONDS) & ~np.isnan(act)]
    if len(values) == 0:
        return None
    return float(np.median(values)), float(np.percentile(values, 90))


def run_plot(a):
    plt = get_plt()
    results = []
    for f in a.files:
        if not f.is_file():
            sys.exit(f"Not found: {f}")
        try:
            t, act = block_activity(f)
        except ValueError as exc:
            sys.exit(f"Cannot plot {f.name}: {exc}")
        if len(t) == 0:
            sys.exit(f"{f.name} is shorter than the activity window, nothing to plot.")
        results.append((f, t, act))

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.2 * len(results) + 1), sharey=True,
                             squeeze=False)
    print(f"{'file':60s} {'median':>8s} {'p90':>8s}   (activity after the first {SKIP_SECONDS} s)")
    for ax, (f, t, act) in zip(axes[:, 0], results):
        ax.plot(t, act, color="#2a6fbb", linewidth=1)
        ax.axvspan(0, SKIP_SECONDS, color="#dddddd")
        ax.set_title(f.name, fontsize=9, loc="left")
        ax.set_ylabel("activity")
        s = summary(t, act)
        print(f"{f.name:60s} " + (f"{s[0]:8.2f} {s[1]:8.2f}" if s else "     n/a      n/a"))
    axes[-1, 0].set_xlabel("seconds from the start of the file (grey = first 5 s, skipped)")
    fig.tight_layout()

    out = a.out or FIGURES_DIR / f"blocks_{a.files[0].stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot: {out}")
    return 0


# ================================ map (subcarrier profile) =================================

def mean_profile(path):
    """Return (mean amplitude per position, rows used) for one file."""
    df, amp, _ = load_recording(path)
    if len(df) < 2:
        raise ValueError("fewer than 2 good rows")
    ts = unwrap_timestamp(df["local_timestamp"].to_numpy(), df["id"].to_numpy())
    seconds = (ts - ts[0]) / 1e6
    used = amp[seconds >= SKIP_SECONDS]
    if len(used) == 0:
        raise ValueError(f"the file is shorter than the {SKIP_SECONDS} s that are skipped")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a position that is NaN in every row
        return np.nanmean(used, axis=0), len(used)


def lowest_positions(means, k=N_LOWEST):
    return sorted(int(i) for i in np.argsort(means)[:k])


def plot_profile(plt, means, lowest, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = np.arange(N_POSITIONS)
    kept = np.isin(pos, KEPT_POSITIONS)
    ax.bar(pos[kept], means[kept], color="#2a6fbb", label="provisional keep (1-26, 38-63)")
    ax.bar(pos[~kept], means[~kept], color="#c8c8c8", label="predicted empty (0, 27-37)")
    for p in lowest:
        ax.plot(p, means[p], marker="v", color="#d1432f", markersize=5, linestyle="none")
    ax.plot([], [], marker="v", color="#d1432f", linestyle="none", label=f"{N_LOWEST} lowest measured")
    ax.set_xlabel("subcarrier position")
    ax.set_ylabel("mean amplitude")
    ax.set_title(title)
    ax.set_xlim(-1, N_POSITIONS)
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_map(a):
    plt = get_plt()
    if not a.path.is_file():
        sys.exit(f"Not found: {a.path}")
    try:
        means, n_rows = mean_profile(a.path)
    except ValueError as exc:
        sys.exit(f"Cannot profile {a.path.name}: {exc}")

    low = lowest_positions(means)
    out = a.out or FIGURES_DIR / f"profile_{a.path.stem}.png"
    plot_profile(plt, means, low, out, f"Mean amplitude per position - {a.path.name} ({n_rows} rows)")

    print(f"{a.path.name}: {n_rows} rows used (first {SKIP_SECONDS} s skipped; positions 0 and 1 "
          f"skipped in rows with first_word == 1)")
    print(f"{N_LOWEST} lowest mean amplitude positions: {low}")
    print("  " + ", ".join(f"{p}: {means[p]:.2f}" for p in np.argsort(means)[:N_LOWEST]))
    missing = sorted(set(PREDICTED_EMPTY) - set(low))
    extra = sorted(set(low) - set(PREDICTED_EMPTY))
    if not missing and not extra:
        print("Matches the predicted empty positions {0, 27-37}: keep 1-26 and 38-63 (52 positions).")
    else:
        print("Does NOT match the predicted empty positions {0, 27-37}.")
        print(f"  predicted but not among the lowest: {missing}")
        print(f"  among the lowest but not predicted: {extra}")
        print("  Decide the 52 kept positions from the plot before the config freeze.")
    print(f"Plot: {out}")
    return 0


# ================================ command line =============================================

def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.collect", description="Look at the CSI signal.")
    sub = p.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="live plot of the recording that is being written")
    live.add_argument("path", nargs="?", type=Path, help="CSV file (default: follow the newest recording)")
    live.add_argument("--calibrate", type=int, default=0, metavar="SECONDS",
                      help="treat the first SECONDS of the file as empty room and draw a threshold line")
    live.add_argument("--snapshot", type=Path, metavar="PNG", help="read the file once, save a picture, exit")

    plot = sub.add_parser("plot", help="activity plot of finished recordings, one panel per file")
    plot.add_argument("files", nargs="+", type=Path)
    plot.add_argument("--out", type=Path, help="PNG file (default reports/figures/blocks_<first file>.png)")

    smap = sub.add_parser("map", help="mean amplitude per subcarrier position of one file")
    smap.add_argument("path", type=Path)
    smap.add_argument("--out", type=Path, help="PNG file (default reports/figures/profile_<name>.png)")

    a = p.parse_args(argv)
    return {"live": run_live, "plot": run_plot, "map": run_map}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
