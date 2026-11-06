"""Shared constants and helpers for the CSI collection programs (guide v2, step 3)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# The 25 columns printed by firmware/csi_recv_router (header line of the CSI_DATA output).
FIRMWARE_COLUMNS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
    "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
    "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_format", "len",
    "first_word", "data",
]
CSV_COLUMNS = ["pc_time"] + FIRMWARE_COLUMNS
N_FIELDS = len(FIRMWARE_COLUMNS)
NUMERIC_COLUMNS = [c for c in FIRMWARE_COLUMNS if c not in ("type", "mac", "data")]
LEN_INDEX = FIRMWARE_COLUMNS.index("len")

# Frozen router values. Changing any of these (router, channel, width) means a new dataset version.
ROUTER_BSSID = "6c:14:6e:2c:71:4c"
ROUTER_CHANNEL = 2
EXPECTED_LEN = 128
N_POSITIONS = EXPECTED_LEN // 2  # 64 subcarrier positions (imag, real pairs)

# Provisional kept positions (guide step 3, program 3): confirm with `map` on the pilot.
KEPT_POSITIONS = list(range(1, 27)) + list(range(38, 64))

# Checker PASS rules, first version (guide step 3, program 2). Revise after the pilot.
MIN_ROWS_PER_SECOND = 80
MAX_GAP_MS = 500
GAP_WARN_MS = 100
MAX_RESTARTS = 0
MAX_MISSING_FRACTION = 0.01  # lost serial lines (gaps in the firmware id) allowed, as a fraction of all lines
SKIP_SECONDS = 5

TIMESTAMP_WRAP = 2 ** 32

ROOMS = ("roomA", "roomB")
KINDS = ("pilot", "test", "data")

# Label lists for pilot and data recordings (plan 2B manifest). "test" recordings (link tests) are free text.
ACTIVITIES = ("quiet", "disturbance", "still", "walk", "desk", "mixed")  # E1 / E2-E3 / S / D, T still / T dynamic
POSTURES = ("none", "sit", "stand", "lie")
POSITIONS = ("none", "near-link", "away", "edge")
DISTURBANCES = ("none", "fan", "door", "chair", "other")


def parse_csi_line(line: str):
    """Classify one serial line.

    Returns ("noise", None) when it is not a CSI_DATA line, ("damaged", None) when it
    starts like one but is broken, or ("ok", fields) with the 25 values unchanged
    (the data field keeps its [..] but loses the outer quotes; csv adds them back).
    """
    if not line.startswith("CSI_DATA,"):
        return "noise", None
    parts = line.split(",", N_FIELDS - 1)
    if len(parts) != N_FIELDS:
        return "damaged", None
    data = parts[-1]
    if not (data.startswith('"[') and data.endswith(']"')):
        return "damaged", None
    inner = data[2:-2]
    try:
        for name, value in zip(FIRMWARE_COLUMNS[:-1], parts):
            if name in NUMERIC_COLUMNS:
                int(value)  # log text spliced into a row ends up here
        declared = int(parts[LEN_INDEX])
        numbers = inner.split(",")
        for n in numbers:
            int(n)
    except ValueError:
        return "damaged", None
    if len(numbers) != declared:
        return "damaged", None
    parts[-1] = "[" + inner + "]"
    return "ok", parts


def parse_data_column(series: pd.Series, declared_len: pd.Series):
    """Turn the data strings into a (rows, 128) int array.

    Returns (values, bad) where bad marks rows whose number count differs from len
    (or that cannot be parsed); their values are left as NaN.
    """
    n = len(series)
    values = np.full((n, EXPECTED_LEN), np.nan)
    bad = np.zeros(n, dtype=bool)
    for i, (text, declared) in enumerate(zip(series.to_numpy(), declared_len.to_numpy())):
        try:
            nums = np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float64)
        except (ValueError, AttributeError):
            bad[i] = True
            continue
        if len(nums) != declared or len(nums) != EXPECTED_LEN:
            bad[i] = True
            continue
        values[i] = nums
    return values, bad


def amplitudes(values: np.ndarray) -> np.ndarray:
    """(rows, 128) raw CSI -> (rows, 64) amplitude. The pairs are (imag, real)."""
    pairs = values.reshape(len(values), N_POSITIONS, 2)
    return np.hypot(pairs[:, :, 0], pairs[:, :, 1])


def unwrap_timestamp(ts: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """local_timestamp is a 32-bit microsecond counter: add 2^32 when it drops while id keeps counting."""
    ts = ts.astype(np.int64)
    if len(ts) < 2:
        return ts
    wrapped = (np.diff(ts) < 0) & (np.diff(ids) > 0)
    return ts + np.concatenate(([0], np.cumsum(wrapped))) * TIMESTAMP_WRAP


def load_recording(path):
    """Read one recording CSV.

    Returns (df, amp, n_damaged). Rows with a non-numeric field or a number count that differs
    from len are dropped and counted. amp is (rows, 64); in rows with first_word == 1,
    positions 0 and 1 are NaN (the first 4 numbers are invalid).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    numeric = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_fields = numeric.isna().any(axis=1).to_numpy()
    df[NUMERIC_COLUMNS] = numeric
    values, bad_data = parse_data_column(df["data"], numeric["len"].fillna(-1))
    bad = bad_fields | bad_data
    good = ~bad
    df = df[good].reset_index(drop=True)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].astype("int64")
    amp = amplitudes(values[good])
    amp[df["first_word"].to_numpy() == 1, :2] = np.nan
    return df, amp, int(bad.sum())


# ---- activity indicator (used by the live view and the block plots) ------------------------
# For each kept subcarrier take the standard deviation of its amplitude over a short window, then
# average over the 52 subcarriers. A still, empty room is low and flat; a person moving gives spikes.

WINDOW_S = 2.0   # length of the window the std is taken over
STEP_S = 0.5     # one activity value every 0.5 s
MIN_ROWS = 5     # fewer rows than this in a window -> no value (NaN)


def activity_series(seconds, amp, window_s=WINDOW_S, step_s=STEP_S):
    """Rolling activity indicator.

    seconds: (rows,) time of each row in seconds, increasing.
    amp: (rows, 64) amplitude per position (see amplitudes()).
    Returns (t, activity): the end time of each window and its value (NaN when too few rows).
    """
    seconds = np.asarray(seconds, dtype=float)
    if len(seconds) < 2 or seconds[-1] - seconds[0] < window_s:
        return np.array([]), np.array([])
    kept = amp[:, KEPT_POSITIONS]
    ends = np.arange(seconds[0] + window_s, seconds[-1] + 1e-9, step_s)
    lo = np.searchsorted(seconds, ends - window_s, side="left")
    hi = np.searchsorted(seconds, ends, side="right")
    out = np.full(len(ends), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a position that is NaN in the whole window
        for k, (a, b) in enumerate(zip(lo, hi)):
            if b - a >= MIN_ROWS:
                out[k] = np.nanmean(np.nanstd(kept[a:b], axis=0))
    return ends, out


def threshold_from_baseline(activity):
    """Empty-room threshold = mean + 3 std of the activity values (plan step 2C.5)."""
    values = activity[~np.isnan(activity)]
    if len(values) < 3:
        return None
    return float(values.mean() + 3 * values.std())
