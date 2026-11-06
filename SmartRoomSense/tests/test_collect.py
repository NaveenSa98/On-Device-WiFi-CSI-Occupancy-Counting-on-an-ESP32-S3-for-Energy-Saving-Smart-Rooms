"""Tests for src/collect, using generated rows only (no board needed)."""
import csv
import time

import pytest

from src.collect import logger
from src.collect.common import (CSV_COLUMNS, ROUTER_BSSID, ROUTER_CHANNEL, parse_csi_line)


def make_line(row_id, ts=None, length=128, count=None, first_word=0, bandwidth=0, channel=ROUTER_CHANNEL,
              mac=ROUTER_BSSID, amp_scale=1):
    """One CSI_DATA line in the firmware's format."""
    count = length if count is None else count
    ts = row_id * 10_000 if ts is None else ts
    nums = ",".join(str(((row_id + i) % 7 + 1) * amp_scale) for i in range(count))
    return (f"CSI_DATA,{row_id},{mac},-30,11,1,7,{bandwidth},0,0,1,0,0,1,-97,0,{channel},0,"
            f"{ts},0,76,1,{length},{first_word},\"[{nums}]\"")


# ---- parser -------------------------------------------------------------------------------

def test_parse_good_line():
    kind, fields = parse_csi_line(make_line(5))
    assert kind == "ok"
    assert len(fields) == 25
    assert fields[1] == "5" and fields[2] == ROUTER_BSSID
    assert fields[-1].startswith("[") and fields[-1].endswith("]")


def test_parse_noise_and_damaged():
    assert parse_csi_line("I (123) wifi: connected")[0] == "noise"
    assert parse_csi_line("type,id,mac,rssi")[0] == "noise"
    assert parse_csi_line(make_line(1, count=100))[0] == "damaged"       # count != len
    assert parse_csi_line(make_line(1)[:-5])[0] == "damaged"              # cut off
    assert parse_csi_line("CSI_DATA,1,aa")[0] == "damaged"                # too few fields
    assert parse_csi_line(make_line(1).replace('"[', '"[x', 1))[0] == "damaged"  # not a number


# ---- logger -------------------------------------------------------------------------------

class FakeSerial:
    """Feeds ~100 lines/s: some noise, one damaged line, then optionally goes silent."""

    def __init__(self, n_rows, silent_after=True, rate=100):
        self.payload = []
        self.payload.append("I (1) boot: hello")
        for i in range(n_rows):
            self.payload.append(make_line(i))
            if i == 3:
                self.payload.append(make_line(i, count=50))
        self.rate = rate
        self.sent = 0
        self.t0 = time.monotonic()
        self.closed = False

    @property
    def in_waiting(self):
        return 1

    def read(self, n):
        due = int((time.monotonic() - self.t0) * self.rate) + 1
        if self.sent < min(due, len(self.payload)):
            line = self.payload[self.sent]
            self.sent += 1
            return (line + "\r\n").encode()
        time.sleep(0.005)
        return b""

    def close(self):
        self.closed = True


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


def run_logger(monkeypatch, fake, *extra):
    monkeypatch.setattr(logger, "open_port", lambda port, baud: fake)
    argv = ["--port", "FAKE", "--kind", "data", "--room", "roomA", "--day", "1", "--session", "2",
            "--people", "2", "--activity", "walk", "--block", "T4", "--subjects", "P1+P2",
            "--countdown", "0", *extra]
    return logger.main(argv)


def test_build_path(data_dir):
    p = logger.build_path("data", "roomA", 1, 2, 2, "walk", 1, block="T4")
    assert p == data_dir / "raw/roomA_d1_s2/roomA_d1_s2_T4.csv"                  # first take: no suffix
    assert logger.build_path("data", "roomA", 1, 2, 2, "walk", 2, block="T4").name == "roomA_d1_s2_T4_r2.csv"
    assert logger.build_path("data", "roomA", 1, 2, 2, "walk", 1, partial=True, block="T4").name ==         "roomA_d1_s2_T4_partial.csv"
    assert logger.build_path("pilot", "roomA", 0, 1, 0, "quiet", 1, block="E1") == data_dir / "pilot/roomA_d0_s1_E1.csv"
    assert logger.build_path("test", "roomA", 0, 1, 0, "testA", 1) == data_dir / "tests/roomA_d0_s1_n0_testA_r1.csv"


def test_logger_full_recording(data_dir, monkeypatch):
    fake = FakeSerial(n_rows=400)
    assert run_logger(monkeypatch, fake, "--duration", "2") == 0
    out = data_dir / "raw/roomA_d1_s2/roomA_d1_s2_T4.csv"
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_COLUMNS
    assert 100 <= len(rows) - 1 <= 260
    assert len(rows[1]) == 26 and rows[1][0][:2] == "20"       # pc_time then 25 fields
    assert rows[1][3] == ROUTER_BSSID
    with open(data_dir / "raw" / "manifest.csv", newline="") as f:
        log = list(csv.DictReader(f))
    assert list(log[0]) == logger.MANIFEST_COLUMNS
    row = log[0]
    assert len(log) == 1 and row["skipped"] == "1" and row["count"] == "2" and row["presence"] == "1"
    assert row["file"] == "roomA_d1_s2/roomA_d1_s2_T4.csv"                     # relative to manifest.csv
    assert (row["block"], row["subjects"], row["day"], row["partial"]) == ("T4", "P1+P2", "1", "0")
    assert (row["channel"], row["bandwidth"]) == (str(ROUTER_CHANNEL), "0")    # from the first row
    assert fake.closed


def test_logger_rejects_incomplete_labels_before_opening_the_port(data_dir, monkeypatch):
    def refuse(port, baud):
        raise AssertionError("port must not be opened")
    monkeypatch.setattr(logger, "open_port", refuse)
    base = ["--port", "FAKE", "--kind", "pilot", "--countdown", "0"]
    bad = [
        (["--people", "1", "--activity", "still", "--block", "S1", "--subjects", "P1"], "posture"),
        (["--people", "1", "--activity", "walk", "--subjects", "P1"], "--block"),
        (["--people", "2", "--activity", "walk", "--block", "T4", "--subjects", "P1"], "--subjects"),
        (["--people", "0", "--activity", "disturbance", "--block", "E2", "--subjects", "none"], "disturbance"),
        (["--people", "1", "--activity", "quiet", "--block", "E1", "--subjects", "P1"], "empty room"),
        (["--people", "0", "--activity", "dancing", "--block", "E1", "--subjects", "none"], "--activity"),
        (["--people", "1", "--activity", "walk", "--block", "D1", "--subjects", "P1", "--posture", "fly"], "--posture"),
    ]
    for args, word in bad:
        with pytest.raises(SystemExit) as exc:
            logger.main(base + args)
        assert word in str(exc.value), (args, str(exc.value))


def test_logger_pilot_manifest_with_two_people_labels(data_dir, monkeypatch):
    monkeypatch.setattr(logger, "open_port", lambda port, baud: FakeSerial(300))
    assert logger.main(["--port", "FAKE", "--kind", "pilot", "--countdown", "0", "--duration", "1",
                        "--people", "2", "--activity", "still", "--block", "T1", "--subjects", "P1+P2",
                        "--posture", "sit+lie", "--position", "near-link+away", "--notes", "door open"]) == 0
    with open(data_dir / "pilot" / "manifest.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["file"] == "roomA_d0_s1_T1.csv"
    assert (row["posture"], row["position"], row["notes"]) == ("sit+lie", "near-link+away", "door open")


def test_link_tests_write_only_the_csv(data_dir, monkeypatch):
    monkeypatch.setattr(logger, "open_port", lambda port, baud: FakeSerial(300))
    assert logger.main(["--port", "FAKE", "--kind", "test", "--people", "0", "--activity", "testA",
                        "--countdown", "0", "--duration", "1"]) == 0
    assert [p.name for p in (data_dir / "tests").iterdir()] == ["roomA_d0_s1_n0_testA_r1.csv"]


def test_logger_never_overwrites_and_retake_gets_r2(data_dir, monkeypatch):
    assert run_logger(monkeypatch, FakeSerial(300), "--duration", "1") == 0
    with pytest.raises(SystemExit) as exc:
        run_logger(monkeypatch, FakeSerial(300), "--duration", "1")
    assert "never overwrite" in str(exc.value)
    assert run_logger(monkeypatch, FakeSerial(300), "--duration", "1", "--retry", "2") == 0
    assert (data_dir / "raw/roomA_d1_s2/roomA_d1_s2_T4_r2.csv").exists()


def test_logger_partial_after_silence(data_dir, monkeypatch, capsys):
    fake = FakeSerial(n_rows=50)                 # 0.5 s of data, then silence
    assert run_logger(monkeypatch, fake, "--duration", "20") == 1
    partial = data_dir / "raw/roomA_d1_s2/roomA_d1_s2_T4_partial.csv"
    assert partial.exists()
    assert not partial.with_name("roomA_d1_s2_T4.csv").exists()


def test_logger_no_rows_saves_nothing(data_dir, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_logger(monkeypatch, FakeSerial(n_rows=0), "--duration", "20")
    assert "No rows" in str(exc.value)
    assert not list(data_dir.rglob("*.csv"))


def test_logger_port_error(data_dir, monkeypatch):
    def boom(port, baud):
        raise logger.serial.SerialException("could not open port 'COM9': PermissionError(13)")
    monkeypatch.setattr(logger, "open_port", boom)
    with pytest.raises(SystemExit) as exc:
        logger.main(["--port", "COM9", "--kind", "test", "--people", "0", "--activity", "x",
                     "--countdown", "0"])
    assert "Cannot open COM9" in str(exc.value)


# ---- checker ------------------------------------------------------------------------------

import numpy as np

from src.collect import checker


def write_recording(path, n=1000, rate=100, jitter=2.0, ids=None, ts0=0, bandwidth=0,
                    gap_at=None, damaged=(), seed=0, zero_positions=(), first_word_rows=()):
    """Write a synthetic recording CSV. gap_at=(index, extra_us) opens a hole in the timestamps."""
    rng = np.random.default_rng(seed)
    step = int(1e6 / rate)
    ts, rows = ts0, []
    for i in range(n):
        if gap_at and i == gap_at[0]:
            ts += gap_at[1]
        row_id = i if ids is None else ids[i]
        line = make_line(row_id, ts=ts % 2 ** 32, bandwidth=bandwidth,
                         first_word=1 if i in first_word_rows else 0)
        head, _ = line.split(',"[', 1)
        nums = np.round(20 + rng.normal(0, jitter, 128)).astype(int)
        for z in zero_positions:
            nums[2 * z:2 * z + 2] = 0
        if i in first_word_rows:
            nums[:4] = 999
        if i in damaged:
            nums = nums[:50]
        rows.append(["2026-10-31 10:00:00.000"] + head.split(",") + ["[" + ",".join(map(str, nums)) + "]"])
        ts += step
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)
    return path


def test_checker_pass(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "ok.csv"))
    assert r["passed"], r["reasons"]
    assert 99 <= r["rate"] <= 101
    assert r["restarts"] == 0 and r["damaged"] == 0
    assert r["largest_gap_ms"] == pytest.approx(10, abs=1)
    assert r["activity"] > 0


def test_checker_low_rate_fails(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "slow.csv", n=200, rate=20))
    assert not r["passed"] and any(x.startswith("rate") for x in r["reasons"])


def test_checker_gap_fails(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "gap.csv", gap_at=(500, 1_000_000)))
    assert not r["passed"]
    assert r["largest_gap_ms"] == pytest.approx(1010, abs=5)
    assert r["gaps_over_100"] == 1
    assert any(x.startswith("largest gap") for x in r["reasons"])


def test_checker_restart_counted_not_as_gap(tmp_path):
    ids = list(range(500)) + list(range(500))
    r = checker.check_file(write_recording(tmp_path / "restart.csv", ids=ids))
    assert r["restarts"] == 1
    assert not r["passed"]
    assert r["largest_gap_ms"] < 20
    assert 9 < r["duration"] < 11


def test_checker_timestamp_wrap_is_not_an_error(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "wrap.csv", ts0=2 ** 32 - 3_000_000))
    assert r["passed"], r["reasons"]
    assert r["duration"] == pytest.approx(10, abs=0.1)
    assert r["restarts"] == 0


def test_checker_wrong_bandwidth_and_damaged_rows(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "bw.csv", bandwidth=1, damaged={5, 6}))
    assert r["damaged"] == 2
    assert any("bandwidth" in x for x in r["reasons"])
    assert any("damaged" in x for x in r["reasons"])


def test_checker_partial_never_passes(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "x_r1_partial.csv"))
    assert not r["passed"] and any("partial" in x for x in r["reasons"])


def test_checker_activity_indicator_follows_movement(tmp_path):
    still = checker.check_file(write_recording(tmp_path / "a.csv", jitter=0.5))["activity"]
    moving = checker.check_file(write_recording(tmp_path / "b.csv", jitter=5))["activity"]
    assert moving >= 2 * still


def test_checker_chunks(tmp_path):
    r = checker.check_file(write_recording(tmp_path / "c.csv", n=2000), chunk=10)
    assert len(r["chunks"]) == 2 and r["chunks"][0]["full"]
    assert r["chunks"][0]["rate"] == pytest.approx(100, abs=2)


def test_checker_folder_skips_bad_and_sets_exit_code(tmp_path, capsys):
    write_recording(tmp_path / "good.csv")
    (tmp_path / "bad").mkdir()
    write_recording(tmp_path / "bad" / "old.csv", n=100, rate=10)
    assert checker.main([str(tmp_path), "--brief"]) == 0
    write_recording(tmp_path / "slow.csv", n=100, rate=10)
    assert checker.main([str(tmp_path), "--brief"]) == 1
    assert "1 PASS, 1 FAIL" in capsys.readouterr().out


def test_checker_log_test_line(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "LINK_LOG", tmp_path / "docs" / "link_quality_log.md")
    f = write_recording(tmp_path / "t.csv")
    checker.main([str(f), "--log-test", "A"])
    text = (tmp_path / "docs" / "link_quality_log.md").read_text()
    assert "| A | t.csv |" in text and "PASS" in text


def test_parse_rejects_log_text_spliced_into_a_row():
    line = make_line(7, ts=34065946)
    spliced = line.replace("34065946", "34065ping_ok=93 ping_timeout=2 csi_rows=1019")
    assert parse_csi_line(spliced)[0] == "damaged"


def test_checker_counts_spliced_row_as_damaged_instead_of_crashing(tmp_path):
    path = write_recording(tmp_path / "spliced.csv", n=500)
    text = path.read_text().replace(",50000,", ",ping_ok=93 csi_rows=5,", 1)  # row 5's timestamp
    assert "ping_ok" in text
    path.write_text(text)
    r = checker.check_file(path)
    assert r["damaged"] == 1 and r["rows"] == 499
    assert not r["passed"]


def test_checker_finds_lost_lines_from_id_gaps(tmp_path):
    one_lost = [i + (1 if i >= 500 else 0) for i in range(1000)]        # one id missing
    r = checker.check_file(write_recording(tmp_path / "one.csv", ids=one_lost))
    assert r["missing_ids"] == 1 and r["passed"], r["reasons"]
    many_lost = [i + 3 * (i // 100) for i in range(1000)]               # 3 ids lost every 100 rows
    r = checker.check_file(write_recording(tmp_path / "many.csv", ids=many_lost))
    assert r["missing_ids"] == 27
    assert not r["passed"] and any(x.startswith("lost lines") for x in r["reasons"])


def test_checker_restart_is_not_counted_as_lost_lines(tmp_path):
    ids = list(range(500)) + list(range(500))
    assert checker.check_file(write_recording(tmp_path / "r.csv", ids=ids))["missing_ids"] == 0


def test_checker_auto_accepts_other_settings_but_not_a_mix(tmp_path):
    other = write_recording(tmp_path / "other.csv", bandwidth=1)         # not the frozen router setting
    assert not checker.check_file(other)["passed"]
    assert checker.check_file(other, expected={"bandwidth": None})["passed"]
    folder = tmp_path / "s1"
    folder.mkdir()
    write_recording(folder / "a.csv", bandwidth=1)
    write_recording(folder / "b.csv", bandwidth=1)
    assert checker.main([str(folder), "--brief", "--auto"]) == 0        # agree with each other
    write_recording(folder / "c.csv", bandwidth=0)
    assert checker.main([str(folder), "--brief", "--auto"]) == 1        # one differs


def test_checker_folder_reports_inconsistent_settings_and_saves_check_txt_only_on_request(tmp_path, capsys):
    folder = tmp_path / "roomA_d1_s1"
    folder.mkdir()
    write_recording(folder / "a.csv", bandwidth=0)
    write_recording(folder / "b.csv", bandwidth=1)
    assert checker.main([str(folder), "--brief", "--auto"]) == 1
    assert "CONSISTENCY FAIL: setting bandwidth differs" in capsys.readouterr().out
    assert not (folder / "check.txt").exists()                           # nothing saved unless asked
    checker.main([str(folder), "--brief", "--auto", "--save"])
    text = (folder / "check.txt").read_text()
    assert "=== a.csv ===" in text and "FAIL: setting bandwidth differs" in text


# ---- profile ------------------------------------------------------------------------------

from src.collect import viewer


def test_profile_finds_the_predicted_empty_positions(tmp_path):
    path = write_recording(tmp_path / "p.csv", n=1500, zero_positions=viewer.PREDICTED_EMPTY)
    means, n_rows = viewer.mean_profile(path)
    assert n_rows == pytest.approx(1500 - 500, abs=2)               # first 5 s skipped
    assert viewer.lowest_positions(means) == viewer.PREDICTED_EMPTY
    assert means[1] == pytest.approx(28, abs=2)                      # hypot(20, 20)


def test_profile_ignores_invalid_first_values_when_first_word_is_1(tmp_path):
    path = write_recording(tmp_path / "fw.csv", n=1500, first_word_rows=set(range(600, 1500)))
    means, _ = viewer.mean_profile(path)
    assert means[0] < 40 and means[1] < 40        # 999s would give ~1400 if they were used


def test_profile_main_writes_plot_and_reports_mismatch(tmp_path, capsys):
    path = write_recording(tmp_path / "q.csv", n=1500, zero_positions=[3])
    out = tmp_path / "fig" / "q.png"
    assert viewer.main(["map", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.exists() and "Does NOT match" in text and "3" in text


def test_profile_too_short_file(tmp_path):
    path = write_recording(tmp_path / "s.csv", n=300)               # 3 s, all skipped
    with pytest.raises(SystemExit) as exc:
        viewer.main(["map", str(path)])
    assert "shorter" in str(exc.value)
