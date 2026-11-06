"""Program 1: save one recording from the receiver as one CSV file (guide v2, step 3).

Run from SmartRoomSense/ with idf.py monitor CLOSED (only one program can use the port):

    uv run python -m src.collect record --port COM4 --kind test --room roomA \
        --people 0 --activity testA --duration 30 --countdown 5

A pilot or data block also needs its labels (they go into manifest.csv next to the recordings):

    uv run python -m src.collect record --port COM4 --kind pilot --block S1 --people 1 \
        --activity still --posture sit --position near-link --subjects P1 --duration 120

Where files go (kind -> folder):
    data   data/raw/<session>/<session>_<block>.csv        e.g. data/raw/roomA_d1_s1/roomA_d1_s1_E1.csv
    pilot  data/pilot/<session>_<block>.csv
    test   data/tests/<session>_n<people>_<activity>_r<retry>.csv   (link tests, no manifest)
A re-take of a block gets _r2, _r3 ... (--retry 2). A file is never overwritten.
manifest.csv (pilot and data only) sits in data/raw/ or data/pilot/, one row per recording.
"""
import argparse
import csv
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime

import serial

from src.collect.common import (ACTIVITIES, CSV_COLUMNS, DATA_DIR, DISTURBANCES, FIRMWARE_COLUMNS, KINDS,
                                MIN_ROWS_PER_SECOND, POSITIONS, POSTURES, ROOMS, parse_csi_line)

# One row per recording. "file" is relative to the folder that holds manifest.csv.
MANIFEST_COLUMNS = ["file", "kind", "session_id", "room", "day", "block", "count", "presence",
                    "activity", "posture", "position", "disturbance", "subjects", "start", "end",
                    "channel", "bandwidth", "rows", "rows_per_second", "skipped", "partial", "notes"]
BLOCK_PATTERN = re.compile(r"^[EDST]\d+$")   # E1 empty, E2/E3 disturbance, S1-S3 still 1, D1-D3 dynamic 1, T1-T6 two people
NO_DATA_STOP_S = 5
NO_DATA_WARN_S = 2


def pc_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def build_path(kind, room, day, session, people, activity, retry, partial=False, block=None):
    session_id = f"{room}_d{day}_s{session}"
    suffix = "_partial" if partial else ""
    if kind == "test":                                   # link tests: free text, no block
        return DATA_DIR / "tests" / f"{session_id}_n{people}_{activity}_r{retry}{suffix}.csv"
    take = f"_r{retry}" if retry > 1 else ""            # the first take has no suffix
    name = f"{session_id}_{block}{take}{suffix}.csv"
    if kind == "data":
        return DATA_DIR / "raw" / session_id / name
    return DATA_DIR / "pilot" / name


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


def manifest_path(kind):
    folder = {"data": DATA_DIR / "raw", "pilot": DATA_DIR / "pilot", "test": DATA_DIR / "tests"}[kind]
    return folder / "manifest.csv"


def append_manifest(kind, row: dict):
    path = manifest_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def label_error(a):
    """Return a message when the labels of a pilot/data block are incomplete or inconsistent, else None."""
    if a.kind == "test":
        return None                                   # link tests use free text
    if a.activity not in ACTIVITIES:
        return f"--activity must be one of {', '.join(ACTIVITIES)} for pilot/data (got '{a.activity}')."
    if not a.block or not BLOCK_PATTERN.match(a.block):
        return "--block is required for pilot/data, like E1, E2, S1, D1, T1."
    for name, allowed in (("posture", POSTURES), ("position", POSITIONS), ("disturbance", DISTURBANCES)):
        value = getattr(a, name)               # two people can differ: --posture sit+lie
        if not all(part in allowed for part in value.split("+")):
            return f"--{name} must be {'/'.join(allowed)} (joined with + for several people), got '{value}'."
    people = [s for s in a.subjects.split("+") if s] if a.subjects != "none" else []
    if len(people) != a.people:
        return f"--subjects must name {a.people} person(s) joined by +, like P1+P2 (or 'none' for 0); got '{a.subjects}'."
    if a.people == 0 and a.activity in ("still", "walk", "desk", "mixed"):
        return f"--activity {a.activity} needs --people 1 or more."
    if a.people > 0 and a.activity in ("quiet", "disturbance"):
        return f"--activity {a.activity} means an empty room, but --people is {a.people}."
    if a.activity == "still" and (a.posture == "none" or a.position == "none"):
        return "A still block needs --posture (sit/stand/lie) and --position (near-link/away/edge)."
    if a.activity == "disturbance" and a.disturbance == "none":
        return "A disturbance block needs --disturbance (fan/door/chair/other)."
    return None


def record(reader, out_path, duration, status=print):
    """Write rows to out_path until duration ends, Ctrl+C, or 5 s without data.

    Returns a dict with rows, skipped, start, end, elapsed and partial (True when stopped early).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = skipped = noise = restarts = 0
    last_id = None
    partial = False
    settings = {"channel": "", "bandwidth": ""}   # from the first good row, saved in the manifest
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
                        if rows == 0:
                            for name in settings:
                                settings[name] = fields[FIRMWARE_COLUMNS.index(name)]
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
            "start": start, "end": pc_time(), "elapsed": elapsed, "partial": partial, **settings}


def countdown(seconds, reader, say=print):
    for n in range(seconds, 0, -1):
        say(f"Leave the room. Recording starts in {n} s.")
        time.sleep(1)
    while not reader.lines.empty():  # drop what arrived during the countdown
        reader.lines.get_nowait()


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.collect record", description="Save one CSI recording as one CSV file.")
    p.add_argument("--port", required=True, help="e.g. COM4")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--room", choices=ROOMS, default="roomA")
    p.add_argument("--day", type=int, default=0, help="1-4 for data; 0 for pilot/test")
    p.add_argument("--session", type=int, choices=(1, 2), default=1)
    p.add_argument("--people", type=int, choices=(0, 1, 2, 3), required=True)
    p.add_argument("--activity", required=True,
                   help=f"pilot/data: {', '.join(ACTIVITIES)}; test: free text such as testA")
    p.add_argument("--block", help="block code for pilot/data: E1, E2, S1, D1, T1 ... (see the session table)")
    p.add_argument("--posture", default="none", help="sit / stand / lie (required for still); sit+lie for two people")
    p.add_argument("--position", default="none",
                   help="near-link / away / edge (required for still); join with + for two people")
    p.add_argument("--disturbance", default="none", help="fan / door / chair / other (required for disturbance)")
    p.add_argument("--subjects", default="none", help="who is in the room, e.g. P1 or P1+P2 ('none' when empty)")
    p.add_argument("--notes", default="", help="anything unusual, e.g. door left open")
    p.add_argument("--duration", type=int, default=180, help="seconds")
    p.add_argument("--countdown", type=int, default=30, help="seconds")
    p.add_argument("--retry", type=int, default=1)
    a = p.parse_args(argv)

    if a.kind == "data" and a.day < 1:
        sys.exit("kind=data needs --day 1 or more.")
    problem = label_error(a)
    if problem:
        sys.exit(problem)

    out = build_path(a.kind, a.room, a.day, a.session, a.people, a.activity, a.retry, block=a.block)
    out_partial = build_path(a.kind, a.room, a.day, a.session, a.people, a.activity, a.retry,
                             partial=True, block=a.block)
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
    if a.kind != "test":   # link tests are free text, the file name is enough; only pilot/data get a manifest
        append_manifest(a.kind, {
            "file": final.relative_to(manifest_path(a.kind).parent).as_posix(),
            "kind": a.kind, "session_id": f"{a.room}_d{a.day}_s{a.session}", "room": a.room,
            "day": a.day, "block": a.block or "", "count": a.people, "presence": int(a.people > 0),
            "activity": a.activity, "posture": a.posture, "position": a.position,
            "disturbance": a.disturbance, "subjects": a.subjects,
            "start": result["start"], "end": result["end"],
            "channel": result["channel"], "bandwidth": result["bandwidth"],
            "rows": result["rows"], "rows_per_second": f"{rate:.1f}", "skipped": result["skipped"],
            "partial": int(result["partial"]), "notes": a.notes,
        })
    print(f"Saved {final}\nrows {result['rows']}, {rate:.1f} rows/s, skipped {result['skipped']}"
          + (", PARTIAL" if result["partial"] else ""))
    return 1 if result["partial"] else 0

