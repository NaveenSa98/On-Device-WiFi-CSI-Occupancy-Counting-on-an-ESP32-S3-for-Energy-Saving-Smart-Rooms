"""Tests for the viewer (live, plot, map) (synthetic recordings, no board needed)."""
import csv

import numpy as np

from src.collect import viewer
from src.collect.common import activity_series, threshold_from_baseline
from test_collect import write_recording


def test_activity_is_higher_when_the_signal_moves(tmp_path):
    quiet = viewer.block_activity(write_recording(tmp_path / "q.csv", n=1000, jitter=1.0))
    busy = viewer.block_activity(write_recording(tmp_path / "b.csv", n=1000, jitter=6.0))
    assert np.nanmedian(busy[1]) > 3 * np.nanmedian(quiet[1])


def test_activity_series_too_short_is_empty():
    t, act = activity_series(np.arange(10) / 100, np.ones((10, 64)))
    assert len(t) == 0 and len(act) == 0


def test_threshold_needs_enough_values():
    assert threshold_from_baseline(np.array([1.0, 2.0])) is None
    assert threshold_from_baseline(np.array([1.0, 1.0, 1.0, 1.0])) == 1.0


def test_tail_reads_only_complete_rows(tmp_path):
    path = write_recording(tmp_path / "f.csv", n=20)
    text = path.read_text()
    cut = text[:-30]                                  # last row unfinished
    live = tmp_path / "live.csv"
    live.write_text(cut)
    tail = viewer.CsvTail(live)
    assert len(tail.read_rows()) == 19
    with open(live, "a") as f:
        f.write(text[-30:])                           # the logger finishes the row
    assert len(tail.read_rows()) == 1
    tail.close()


def test_parse_row_rejects_wrong_length():
    assert viewer.parse_row(["x"] * 5) is None


def test_live_state_survives_timestamp_wrap_and_counts_restart(tmp_path):
    state = viewer.LiveState()
    amp = np.ones(64)
    state.add(0, 2 ** 32 - 10_000, amp)
    state.add(1, 5_000, amp)                          # wrapped, id keeps counting
    assert state.t[-1] == (5_000 + 2 ** 32 - (2 ** 32 - 10_000)) / 1e6
    state.add(0, 100, amp)                            # id went back: board restarted
    assert state.restarts == 1 and len(state.t) == 1


def test_calibrate_gives_threshold(tmp_path):
    path = write_recording(tmp_path / "f.csv", n=1000)
    state = viewer.LiveState(calibrate_s=5)
    tail = viewer.CsvTail(path)
    viewer.feed(state, tail)
    assert state.threshold is not None and state.threshold > 0


def test_snapshot_writes_png(tmp_path):
    path = write_recording(tmp_path / "f.csv", n=1000)
    out = tmp_path / "view.png"
    assert viewer.main(["live", str(path), "--snapshot", str(out), "--calibrate", "5"]) == 0
    assert out.stat().st_size > 1000


def test_plot_block_writes_png_and_table(tmp_path, capsys):
    a = write_recording(tmp_path / "a.csv", n=1000, jitter=1.0)
    b = write_recording(tmp_path / "b.csv", n=1000, jitter=6.0)
    out = tmp_path / "blocks.png"
    assert viewer.main(["plot", str(a), str(b), "--out", str(out)]) == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "a.csv" in printed and "b.csv" in printed


def test_single_entry_point_dispatches_every_command(tmp_path, capsys):
    from src.collect import __main__ as entry
    assert entry.main([]) == 0 and "record" in capsys.readouterr().out
    assert entry.main(["nonsense"]) == 2
    path = write_recording(tmp_path / "f.csv", n=1000)
    assert entry.main(["check", str(path), "--brief"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert entry.main(["plot", str(path), "--out", str(tmp_path / "p.png")]) == 0
    assert entry.main(["map", str(path), "--out", str(tmp_path / "m.png")]) == 0
    assert entry.main(["live", str(path), "--snapshot", str(tmp_path / "l.png")]) == 0
    assert all((tmp_path / n).exists() for n in ("p.png", "m.png", "l.png"))


def test_live_follows_newest_file_in_session_folders_and_skips_wicount(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(viewer, "DATA_DIR", tmp_path)
    for folder in ("raw/roomA_d1_s1", "raw/wicount"):
        (tmp_path / folder).mkdir(parents=True)
    a = write_recording(tmp_path / "raw/roomA_d1_s1/roomA_d1_s1_E1.csv", n=10)
    b = write_recording(tmp_path / "raw/roomA_d1_s1/roomA_d1_s1_S1.csv", n=10)
    other = write_recording(tmp_path / "raw/wicount/x.csv", n=10)       # not a recording folder
    (tmp_path / "raw/manifest.csv").write_text("file\n")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    os.utime(other, (3000, 3000))
    assert viewer.latest_csv() == b
