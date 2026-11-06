"""Program 2: judge one recording (or every recording in a folder) and print PASS or FAIL.

    uv run python -m src.collect check data/tests/roomA_d0_s1_n0_testA_r1.csv
    uv run python -m src.collect check data/raw/roomA_d1_s1 --brief
    uv run python -m src.collect check FILE --chunks 300        # per 5-minute chunk (test C)
    uv run python -m src.collect check FILE --log-test A        # add a line to docs/link_quality_log.md
    uv run python -m src.collect check data/pilot --auto        # another router/room: files only have to agree with each other

Checking a folder also checks that every file used the same router settings (channel, bandwidth,
mac, ...). Nothing is written to the data folders unless you add --save (full report as check.txt).
Lost serial lines are found from gaps in the firmware's id counter (it counts every printed row).
"""
import argparse
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np 

from src.collect.common import (DOCS_DIR, EXPECTED_LEN, GAP_WARN_MS, KEPT_POSITIONS, MAX_GAP_MS,
                                MAX_MISSING_FRACTION, MAX_RESTARTS, MIN_ROWS_PER_SECOND,
                                ROUTER_BSSID, ROUTER_CHANNEL, SKIP_SECONDS, load_recording,
                                unwrap_timestamp)

LINK_LOG = DOCS_DIR / "link_quality_log.md"
LINK_LOG_HEADER = ("| date | test | file | rows/s | largest gap ms | restarts | activity indicator "
                   "| result |\n|---|---|---|---|---|---|---|---|\n")

# setting name -> the single value it must have (None = only "one value" is required)
REQUIRED_SETTINGS = {"mac": ROUTER_BSSID, "bandwidth": 0, "channel": ROUTER_CHANNEL,
                     "secondary_channel": 0, "len": EXPECTED_LEN, "ant": 0}
INFO_COLUMNS = ("rate", "sig_mode", "mcs")


def activity_of(amp, seconds):
    """Std of each kept subcarrier's amplitude over time, averaged. First SKIP_SECONDS skipped."""
    rows = amp[seconds >= SKIP_SECONDS][:, KEPT_POSITIONS]
    if (~np.isnan(rows).all(axis=1)).sum() < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(np.nanstd(rows, axis=0)))


def chunk_stats(seconds, amp, size):
    """Rows per second and mean amplitude for each chunk of `size` seconds."""
    out = []
    n_chunks = int(seconds[-1] // size) + 1
    for k in range(n_chunks):
        lo, hi = k * size, (k + 1) * size
        mask = (seconds >= lo) & (seconds < hi)
        if mask.sum() < 2 or (hi > seconds[-1] and seconds[mask][-1] - seconds[mask][0] < 1):
            continue  # nothing there, or a last chunk too short to give a rate
        span = size if hi <= seconds[-1] else seconds[mask][-1] - seconds[mask][0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_amp = float(np.nanmean(amp[mask][:, KEPT_POSITIONS]))
        out.append({"index": k, "start": lo, "rows": int(mask.sum()), "span": float(span),
                    "rate": mask.sum() / span if span else 0.0, "mean_amp": mean_amp,
                    "full": hi <= seconds[-1]})
    return out


def check_file(path, chunk=None, expected=None):
    """Analyse one CSV and return a dict of measurements and the PASS rules.

    expected: setting name -> required value (default REQUIRED_SETTINGS, the frozen router values).
    A value of None means "any value, but only one" (used by --auto).
    """
    expected = REQUIRED_SETTINGS if expected is None else expected
    path = Path(path)
    res = {"file": path.name, "path": str(path), "reasons": [], "rules": []}
    df, amp, n_damaged = load_recording(path)
    res["rows"] = len(df)
    res["damaged"] = n_damaged
    if len(df) < 2:
        res.update(passed=False, reasons=["fewer than 2 good rows"])
        return res

    ids = df["id"].to_numpy()
    ts = unwrap_timestamp(df["local_timestamp"].to_numpy(), ids)
    restarted = np.diff(ids) < 0
    dt = np.diff(ts).astype(float)
    typical = np.median(dt[~restarted]) if (~restarted).any() else 0.0
    dt[restarted] = typical  # the timestamp may reset at a restart; it is a restart, not a gap
    seconds = np.concatenate(([0.0], np.cumsum(dt))) / 1e6
    gaps = dt[~restarted] / 1000.0

    res["duration"] = float(seconds[-1])
    res["rate"] = len(df) / seconds[-1] if seconds[-1] > 0 else 0.0
    res["largest_gap_ms"] = float(gaps.max()) if len(gaps) else 0.0
    res["gaps_over_100"] = int((gaps > GAP_WARN_MS).sum())
    res["restarts"] = int(restarted.sum())
    # The firmware id counts every printed row, so a jump in id = lines lost on the way to the PC.
    id_step = np.diff(ids)
    res["missing_ids"] = int((id_step[id_step > 1] - 1).sum())
    res["missing_pct"] = 100.0 * res["missing_ids"] / (len(df) + res["missing_ids"])
    res["settings"] = {c: sorted(df[c].astype(str).unique()) for c in REQUIRED_SETTINGS}
    res["info"] = {c: df[c].value_counts().sort_index().to_dict() for c in INFO_COLUMNS}
    res["rssi"] = (float(df["rssi"].mean()), int(df["rssi"].min()), int(df["rssi"].max()))
    res["activity"] = activity_of(amp, seconds)
    res["chunks"] = chunk_stats(seconds, amp, chunk) if chunk else []

    rules = res["rules"]
    rules.append(("rate", res["rate"] >= MIN_ROWS_PER_SECOND,
                  f"{res['rate']:.1f} rows/s (need >= {MIN_ROWS_PER_SECOND})"))
    rules.append(("largest gap", res["largest_gap_ms"] <= MAX_GAP_MS,
                  f"{res['largest_gap_ms']:.0f} ms (need <= {MAX_GAP_MS})"))
    rules.append(("restarts", res["restarts"] <= MAX_RESTARTS,
                  f"{res['restarts']} (need <= {MAX_RESTARTS})"))
    rules.append(("lost lines", res["missing_pct"] <= 100 * MAX_MISSING_FRACTION,
                  f"{res['missing_ids']} missing ids = {res['missing_pct']:.2f}% "
                  f"(need <= {100 * MAX_MISSING_FRACTION:g}%)"))
    for name, want in expected.items():
        seen = res["settings"][name]
        ok = len(seen) == 1 if want is None else seen == [str(want)]
        rules.append((f"setting {name}", ok,
                      f"{', '.join(seen)} (need {'one value' if want is None else want})"))
    rules.append(("damaged rows", res["damaged"] == 0, f"{res['damaged']} (need 0)"))
    if "_partial" in path.stem:
        rules.append(("complete recording", False, "file is _partial (stopped early)"))

    res["reasons"] = [f"{n}: {d}" for n, ok, d in rules if not ok]
    res["passed"] = not res["reasons"]
    return res


def report_lines(r):
    """The full report of one file as a list of text lines."""
    out = [f"\n=== {r['file']} ==="]
    if "duration" not in r:
        out.append(f"rows {r['rows']}\nFAIL: {'; '.join(r['reasons'])}")
        return out
    out.append(f"1. rows {r['rows']}, duration {r['duration']:.1f} s, {r['rate']:.1f} rows/s")
    out.append(f"2. largest gap {r['largest_gap_ms']:.0f} ms, gaps over {GAP_WARN_MS} ms: {r['gaps_over_100']}")
    out.append(f"3. restarts: {r['restarts']}")
    out.append(f"   lost lines (gaps in id): {r['missing_ids']} = {r['missing_pct']:.2f}%")
    out.append("4. settings:")
    for name, seen in r["settings"].items():
        out.append(f"     {name}: {', '.join(seen)}")
    for name, counts in r["info"].items():
        out.append(f"     {name} (info): {counts}")
    out.append(f"5. damaged rows: {r['damaged']}")
    mean, lo, hi = r["rssi"]
    out.append(f"6. RSSI avg {mean:.1f}, lowest {lo}, highest {hi}")
    act = r["activity"]
    out.append("7. activity indicator: " + (f"{act:.3f}" if act is not None else "n/a (too short)"))
    if r["chunks"]:
        out.append("   chunks (start s, rows/s, mean amplitude):")
        for c in r["chunks"]:
            note = "" if c["full"] else "  (last, incomplete)"
            out.append(f"     {c['start']:6.0f}  {c['rate']:6.1f}  {c['mean_amp']:8.3f}{note}")
        full = [c for c in r["chunks"] if c["full"]]
        if len(full) >= 2 and full[0]["mean_amp"]:
            drift = 100 * (full[-1]["mean_amp"] - full[0]["mean_amp"]) / full[0]["mean_amp"]
            out.append(f"   mean amplitude, last vs first full chunk: {drift:+.1f}% (test C: within 10%)")
    if r["passed"]:
        out.append("PASS")
    else:
        out.append("FAIL")
        for reason in r["reasons"]:
            out.append(f"  - {reason}")
    return out


def print_report(r):
    print("\n".join(report_lines(r)))


def consistency_issues(results):
    """Settings that are not the same in every checked file (plan: every session must match)."""
    issues = []
    good = [r for r in results if "settings" in r]
    for name in REQUIRED_SETTINGS:
        seen = {}
        for r in good:
            for value in r["settings"][name]:
                seen.setdefault(value, []).append(r["file"])
        if len(seen) > 1:
            parts = "; ".join(f"{v} in {len(fs)} file(s), e.g. {fs[0]}" for v, fs in sorted(seen.items()))
            issues.append(f"setting {name} differs between files: {parts}")
    return issues


def save_report(folder, results, issues):
    """Write the full report of a folder to folder/check.txt (a report, not raw data: it is rewritten)."""
    lines = [f"Checker report, {datetime.now():%Y-%m-%d %H:%M}, folder {folder}"]
    for r in results:
        lines += report_lines(r)
    lines.append("\n=== consistency between files ===")
    lines += [f"FAIL: {i}" for i in issues] or ["OK: every file has the same settings"]
    failed = sum(1 for r in results if not r["passed"])
    lines.append(f"\n{len(results) - failed} PASS, {failed} FAIL of {len(results)} files"
                 + (", CONSISTENCY FAIL" if issues else ""))
    out = Path(folder) / "check.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def brief_line(r):
    if "duration" not in r:
        return f"FAIL  {r['file']}  ({'; '.join(r['reasons'])})"
    act = f"{r['activity']:.2f}" if r["activity"] is not None else "n/a"
    return (f"{'PASS' if r['passed'] else 'FAIL'}  {r['file']}  {r['rate']:.1f} rows/s  "
            f"gap {r['largest_gap_ms']:.0f} ms  restarts {r['restarts']}  "
            f"lost {r['missing_ids']}  activity {act}")


def log_test(r, name):
    LINK_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LINK_LOG.exists()
    act = f"{r['activity']:.3f}" if r.get("activity") is not None else "n/a"
    line = (f"| {date.today().isoformat()} | {name} | {r['file']} | {r.get('rate', 0):.1f} | "
            f"{r.get('largest_gap_ms', 0):.0f} | {r.get('restarts', 0)} | {act} | "
            f"{'PASS' if r['passed'] else 'FAIL'} |\n")
    with open(LINK_LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("# Link quality log (guide step 5)\n\n" + LINK_LOG_HEADER)
        f.write(line)
    print(f"Logged to {LINK_LOG}")


def find_files(target: Path):
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.csv")
                  if "bad" not in p.relative_to(target).parts and p.name not in ("session_log.csv", "manifest.csv"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.collect check", description="Judge CSI recordings: PASS or FAIL.")
    p.add_argument("path", type=Path, help="one CSV file or a folder")
    p.add_argument("--brief", action="store_true", help="one line per file")
    p.add_argument("--chunks", type=int, metavar="SECONDS", help="also report per chunk (test C: 300)")
    p.add_argument("--log-test", metavar="NAME", help="append a line to docs/link_quality_log.md (one file)")
    p.add_argument("--auto", action="store_true",
                   help="do not compare with the frozen router values; each setting only needs one value "
                        "(for another router or room). Files must still agree with each other.")
    p.add_argument("--save", action="store_true", help="also write the full report to check.txt in the folder")
    a = p.parse_args(argv)
    expected = {name: None for name in REQUIRED_SETTINGS} if a.auto else None

    if not a.path.exists():
        sys.exit(f"Not found: {a.path}")
    files = find_files(a.path)
    if not files:
        sys.exit(f"No CSV files under {a.path}")
    if a.log_test and len(files) != 1:
        sys.exit("--log-test works on one file at a time.")

    results = []
    for f in files:
        r = check_file(f, a.chunks, expected)
        results.append(r)
        if a.brief:
            print(brief_line(r))
        else:
            print_report(r)
    failed = [r for r in results if not r["passed"]]
    issues = consistency_issues(results) if len(results) > 1 else []
    if len(results) > 1:
        print(f"\n{len(results) - len(failed)} PASS, {len(failed)} FAIL of {len(results)} files")
        for issue in issues:
            print(f"CONSISTENCY FAIL: {issue}")
    if a.path.is_dir() and a.save:
        print(f"Report saved: {save_report(a.path, results, issues)}")
    if a.log_test:
        log_test(results[0], a.log_test)
    return 1 if (failed or issues) else 0

