"""Program 1: save one recording from the receiver as one CSV file (guide v2, step 3).

Run from SmartRoomSense/ with idf.py monitor CLOSED (only one program can use the port):

    uv run python -m src.collect.logger --port COM4 --kind test --room roomA \
        --people 0 --activity testA --duration 30 --countdown 5
"""
import argparse
import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime

import serial

from src.collect.common import (CSV_COLUMNS, DATA_DIR, KINDS, MIN_ROWS_PER_SECOND, ROOMS,
                                parse_csi_line)

SESSION_LOG = DATA_DIR / "session_log.csv"
SESSION_LOG_COLUMNS = ["file", "kind", "room", "session_id", "people", "activity",
                       "start", "end", "rows", "rows_per_second", "skipped"]
NO_DATA_STOP_S = 5
NO_DATA_WARN_S = 2


def pc_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def build_path(kind, room, day, session, people, activity, retry, partial=False):
    session_id = f"{room}_d{day}_s{session}"
    suffix = "_partial" if partial else ""
    name = f"{session_id}_n{people}_{activity}_r{retry}{suffix}.csv"
    if kind == "data":
        return DATA_DIR / "raw" / room / session_id / str(people) / name
    return DATA_DIR / ("pilot" if kind == "pilot" else "tests") / name


def open_port(port, baud):
    """Open the port without toggling DTR/RTS: a reset would redo the gain lock and restart id."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


class Reader(threading.Thread):
    """Reads the port in the background so no data is lost while the CSV is written."""

    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.lines = queue.Queue()
        self.stop_flag = threading.Event()
        self.error = None

    def run(self):
        buf = b""
        try:
            while not self.stop_flag.is_set():
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if not chunk:
                    continue
                stamp = pc_time()
                buf += chunk
                *done, buf = buf.split(b"\n")
                for raw in done:
                    self.lines.put((stamp, raw.decode("utf-8", errors="replace").strip()))
        except (serial.SerialException, OSError) as exc:
            self.error = exc


def append_session_log(row: dict):
    SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not SESSION_LOG.exists()
    with open(SESSION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SESSION_LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def record(reader, out_path, duration, status=print):
    """Write rows to out_path until duration ends, Ctrl+C, or 5 s without data.

    Returns a dict with rows, skipped, start, end, elapsed and partial (True when stopped early).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = skipped = noise = restarts = 0
    last_id = None
    partial = False
    start = pc_time()
    t0 = time.monotonic()
    last_row_t = t0
    next_tick = t0 + 1
    rows_at_tick = 0
    warned_gap = False

    with open(out_path, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        try:
            while True:
                now = time.monotonic()
                if now - t0 >= duration:
                    break
                try:
                    stamp, line = reader.lines.get(timeout=0.05)
                except queue.Empty:
                    stamp = None
                if stamp is not None:
                    kind, fields = parse_csi_line(line)
                    if kind == "ok":
                        writer.writerow([stamp] + fields)
                        rows += 1
                        last_row_t = time.monotonic()
                        warned_gap = False
                        row_id = int(fields[1])
                        if last_id is not None and row_id < last_id:
                            restarts += 1
                            status("WARNING: id went back, the board restarted. "
                                   "Stop and record this file again.")
                        last_id = row_id
                    elif kind == "damaged":
                        skipped += 1
                    else:
                        noise += 1
                now = time.monotonic()
                if now >= next_tick:
                    f.flush()
                    last_sec = rows - rows_at_tick
                    rows_at_tick = rows
                    next_tick += 1
                    msg = (f"{int(now - t0):4d}s  rows {rows}  last second {last_sec}  "
                           f"skipped {skipped}")
                    if last_sec < MIN_ROWS_PER_SECOND:
                        msg += f"  WARNING: under {MIN_ROWS_PER_SECOND} rows in the last second"
                    status(msg)
                gap = now - last_row_t
                if gap >= NO_DATA_WARN_S and not warned_gap:
                    status(f"WARNING: no row for {NO_DATA_WARN_S} s")
                    warned_gap = True
                if gap >= NO_DATA_STOP_S:
                    status(f"No data for {NO_DATA_STOP_S} s, stopping.")
                    partial = True
                    break
                if reader.error is not None:
                    status(f"Serial error: {reader.error}")
                    partial = True
                    break
        except KeyboardInterrupt:
            status("Interrupted (Ctrl+C), saving what exists.")
            partial = True
    elapsed = time.monotonic() - t0
    return {"rows": rows, "skipped": skipped, "noise": noise, "restarts": restarts,
            "start": start, "end": pc_time(), "elapsed": elapsed, "partial": partial}


def countdown(seconds, reader, say=print):
    for n in range(seconds, 0, -1):
        say(f"Leave the room. Recording starts in {n} s.")
        time.sleep(1)
    while not reader.lines.empty():  # drop what arrived during the countdown
        reader.lines.get_nowait()


def main(argv=None):
    p = argparse.ArgumentParser(description="Save one CSI recording as one CSV file.")
    p.add_argument("--port", required=True, help="e.g. COM4")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--room", choices=ROOMS, default="roomA")
    p.add_argument("--day", type=int, default=0, help="1-4 for data; 0 for pilot/test")
    p.add_argument("--session", type=int, choices=(1, 2), default=1)
    p.add_argument("--people", type=int, choices=(0, 1, 2, 3), required=True)
    p.add_argument("--activity", required=True, help="e.g. baseline, still, walk, mixed, fan, door")
    p.add_argument("--duration", type=int, default=180, help="seconds")
    p.add_argument("--countdown", type=int, default=30, help="seconds")
    p.add_argument("--retry", type=int, default=1)
    a = p.parse_args(argv)

    if a.kind == "data" and a.day < 1:
        sys.exit("kind=data needs --day 1 or more.")

    out = build_path(a.kind, a.room, a.day, a.session, a.people, a.activity, a.retry)
    out_partial = build_path(a.kind, a.room, a.day, a.session, a.people, a.activity, a.retry,
                             partial=True)
    for path in (out, out_partial):
        if path.exists():
            sys.exit(f"File already exists, never overwrite: {path}\n"
                     f"Use a new --retry number (move bad files to data/bad/).")

    try:
        ser = open_port(a.port, a.baud)
    except (serial.SerialException, OSError) as exc:
        sys.exit(f"Cannot open {a.port}: {exc}\n"
                 f"Is idf.py monitor (or another program) still using the port? Is the cable plugged in?")

    reader = Reader(ser)
    reader.start()
    try:
        try:
            countdown(a.countdown, reader)
        except KeyboardInterrupt:
            sys.exit("Cancelled during the countdown, nothing saved.")
        print(f"Recording {out.name} for {a.duration} s. Ctrl+C stops and saves a _partial file.")
        result = record(reader, out, a.duration)
    finally:
        reader.stop_flag.set()
        reader.join(timeout=1)
        ser.close()

    if result["rows"] == 0:
        out.unlink(missing_ok=True)
        sys.exit("No rows received, nothing saved. Check the board is running and the baud rate.")

    final = out
    if result["partial"]:
        os.replace(out, out_partial)
        final = out_partial

    rate = result["rows"] / result["elapsed"] if result["elapsed"] else 0.0
    append_session_log({
        "file": final.relative_to(DATA_DIR.parent).as_posix(),
        "kind": a.kind, "room": a.room, "session_id": f"{a.room}_d{a.day}_s{a.session}",
        "people": a.people, "activity": a.activity, "start": result["start"],
        "end": result["end"], "rows": result["rows"], "rows_per_second": f"{rate:.1f}",
        "skipped": result["skipped"],
    })
    print(f"Saved {final}\nrows {result['rows']}, {rate:.1f} rows/s, skipped {result['skipped']}"
          + (", PARTIAL" if result["partial"] else ""))
    return 1 if result["partial"] else 0


if __name__ == "__main__":
    sys.exit(main())
