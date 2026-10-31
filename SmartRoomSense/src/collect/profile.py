"""Program 3: show which of the 64 subcarrier positions carry signal.

    uv run python -m src.collect.profile data/tests/roomA_d0_s1_n0_testA_r5.csv

Averages the amplitude at each position over one file (first 5 s skipped; positions 0 and 1
skipped in rows with first_word == 1), saves a plot to reports/figures/, and prints the 12
positions with the lowest mean amplitude. For a 20 MHz channel the guide predicts empty
positions at 0 and 27-37, which keeps 1-26 and 38-63 (52 positions).
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.collect.common import (FIGURES_DIR, KEPT_POSITIONS, N_POSITIONS, SKIP_SECONDS,  # noqa: E402
                                load_recording, unwrap_timestamp)

PREDICTED_EMPTY = [0] + list(range(27, 38))  # 12 positions
N_LOWEST = 12


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


def plot_profile(means, lowest, out_path, title):
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Mean amplitude per subcarrier position for one file.")
    p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path, help="plot file (default reports/figures/profile_<name>.png)")
    a = p.parse_args(argv)

    if not a.path.is_file():
        sys.exit(f"Not found: {a.path}")
    try:
        means, n_rows = mean_profile(a.path)
    except ValueError as exc:
        sys.exit(f"Cannot profile {a.path.name}: {exc}")

    low = lowest_positions(means)
    out = a.out or FIGURES_DIR / f"profile_{a.path.stem}.png"
    plot_profile(means, low, out, f"Mean amplitude per position - {a.path.name} ({n_rows} rows)")

    print(f"{a.path.name}: {n_rows} rows used (first {SKIP_SECONDS} s skipped)")
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


if __name__ == "__main__":
    sys.exit(main())
