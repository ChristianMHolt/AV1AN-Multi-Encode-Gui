#!/usr/bin/env python3
"""
AV1 GUI Runner (Qt) — grid UI, 4 jobs at once, live per-job progress + FPS

This version fixes the issues you hit:
- No ffprobe at startup (nothing blocks before the window appears).
- Parses live stdout/stderr from av1an; does not rely on sparse .log files.
- Ignores source "Video: … 24 fps" lines, so FPS reflects encoder speed.
- Progress heuristics: `progress:N`, `Encoded X/Y`, `chunk X/Y`.
- Clean JobTile (no broken strings or duplicated code).

Arch packages (choose ONE Qt binding):
  sudo pacman -S --needed pyside6        # or: python-pyqt6
  sudo pacman -S --needed python-matplotlib av1an mkvtoolnix-cli numactl util-linux

Run:
  python av1an_gui.py
"""
from __future__ import annotations
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Deque, Dict, List, Optional

# ----------------------------- CONFIG (edit) ---------------------------------
SVT_OPTS = "--preset 3 --rc 0 --crf 12 --lp 3 --aq-mode 2 --keyint 240 --enable-tf 0"
PASSES = 1
W_PER_JOB = 1               # av1an workers per file (keep 1 per CCX)
USE_CHUNK_METHOD = "bestsource"  # requires VapourSynth + BestSource plugin
TMPDIR = "/var/tmp/av1tmp"  # fast local SSD/NVMe
MAX_PAR = 4                 # number of concurrent files (tiles)
RELIABILITY_OPTS = ["--resume", "--max-tries", "3"]
INPUT_GLOBS = ["*.mkv", "*.mp4", "*.mov", "*.avi", "*.m2ts", "*.ts", "*.webm"]
LOG_DIR = (Path.cwd() / ".log").resolve()  # absolute (we still write it, but we parse stdout)
OUT_DIR = Path("output")
FPS_WINDOW = 120            # samples kept per job
GUI_REFRESH_HZ = 8
# -----------------------------------------------------------------------------

# ------------------------------ Regex Parsers --------------------------------
# Ignore input/stream description fps like: "Video: h264, 24 fps"
IGNORE_FPS_LINE = re.compile(r"\b(Video:|Stream #|Input #)\b", re.IGNORECASE)
# Prefer encoder-speed lines that actually mention speed/encoding
SPEED_FPS_RE = re.compile(r"\b(speed|enc|encoding)[^\n]*?(\d+(?:\.\d+)?)\s*fps\b", re.IGNORECASE)
# Fallback fps if no speed line matched (used cautiously)
ANY_FPS_RE   = re.compile(r"\b(\d+(?:\.\d+)?)\s*fps\b", re.IGNORECASE)
# Percent formats: "37.5%" or "progress: 37.5"
PCT_RE = re.compile(r"(?:(\d{1,3})(?:\.(\d+))?%)|progress\s*[:=]\s*(\d+(?:\.\n\d+)?)", re.IGNORECASE)
# Encoded frames count "Encoded X / Y"
ENC_RE   = re.compile(r"encoded\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
# av1an sometimes logs "chunk X / Y"
CHUNK_RE = re.compile(r"chunk\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

# -------------------------- System capability checks -------------------------

def check_tool(name: str, hint: str):
    if shutil.which(name) is None:
        raise SystemExit(f"{name} not found: {hint}")


def detect_bestsource() -> bool:
    if USE_CHUNK_METHOD != "bestsource":
        return False
    try:
        import importlib
        vs = importlib.import_module("vapoursynth")
        return hasattr(vs.core, "bs")
    except Exception:
        return False


def read_numa() -> tuple[list[int], dict[int, str], dict[int, str]]:
    out = subprocess.check_output(["numactl", "--hardware"], text=True)
    mem_nodes: List[int] = []
    for line in out.splitlines():
        if line.startswith("node ") and " size:" in line:
            parts = line.split()
            node = int(parts[1])
            try:
                size_mb = int(parts[3])
            except Exception:
                size_mb = 0
            if size_mb > 0:
                mem_nodes.append(node)
    if not mem_nodes:
        raise SystemExit("No usable memory nodes found; aborting.")
    node_ccx0: Dict[int, str] = {}
    node_ccx1: Dict[int, str] = {}
    for n in mem_nodes:
        cpus_line = None
        for ln in out.splitlines():
            if ln.startswith(f"node {n} cpus:"):
                cpus_line = ln
                break
        if not cpus_line:
            for ln in out.splitlines():
                if ln.startswith("node ") and " cpus:" in ln and f"node {n} " in ln:
                    cpus_line = ln
                    break
        parts = cpus_line.split(":", 1)[1].strip().split() if cpus_line else []
        prim = parts[:6] if len(parts) >= 6 else parts
        ccx0 = prim[:3]
        ccx1 = prim[3:6]
        node_ccx0[n] = ",".join(ccx0)
        node_ccx1[n] = ",".join(ccx1)
    return mem_nodes, node_ccx0, node_ccx1

# ------------------------------- Qt Imports ----------------------------------
try:
    from PySide6 import QtCore, QtGui, QtWidgets  # pacman: pyside6
    QT_LIB = "PySide6"
except Exception:  # pragma: no cover
    from PyQt6 import QtCore, QtGui, QtWidgets   # pacman: python-pyqt6
    QT_LIB = "PyQt6"

# Matplotlib optional (for nice FPS plots); otherwise we show text sparkline
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ------------------------------- Data model ----------------------------------
@dataclass
class Job:
    idx: int
    infile: Path
    out_mkv: Path
    tempdir: Path
    log_file: Path
    node: int
    ccx_idx: int
    cpus: str
    proc: Optional[subprocess.Popen] = None
    pct: float = 0.0
    fps_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=FPS_WINDOW))
    started_ts: float = field(default_factory=time.time)
    done: bool = False
    failed: bool = False
    returncode: Optional[int] = None
    line_queue: Queue[str] = field(default_factory=Queue)
    reader_thread: Optional[threading.Thread] = None

# ------------------------------- GUI Widgets ---------------------------------
class JobTile(QtWidgets.QFrame):
    def __init__(self, job: Job, parent=None):
        super().__init__(parent)
        self.job = job
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #555; border-radius: 10px; padding: 8px; }")

        self.title = QtWidgets.QLabel(job.infile.name)
        f = QtGui.QFont("Sans", 10); f.setBold(True)
        self.title.setFont(f)
        self.sub = QtWidgets.QLabel(f"Node {job.node} / CCX{job.ccx_idx}  |  CPUs {job.cpus}")
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 1000)
        self.status = QtWidgets.QLabel("Idle")

        if HAS_MPL:
            self.fig = Figure(figsize=(4, 1.6))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("FPS", fontsize=9)
            self.ax.set_xlabel("time", fontsize=8)
            self.ax.set_ylabel("fps", fontsize=8)
            (self.line,) = self.ax.plot([], [])
            self.ax.grid(True, alpha=0.3)
        else:
            self.canvas = QtWidgets.QLabel("(matplotlib not installed)\nFPS: —")
            self.canvas.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.canvas.setStyleSheet("QLabel { font-family: monospace; }")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)
        lay.addWidget(self.progress)
        lay.addWidget(self.canvas)
        lay.addWidget(self.status)

    def update_from_job(self):
        self.progress.setValue(int(self.job.pct * 10))
        if self.job.done:
            self.status.setText(f"Failed (rc={self.job.returncode})" if self.job.failed else "Done")
        else:
            last_fps = self.job.fps_hist[-1] if self.job.fps_hist else 0.0
            elapsed = time.time() - self.job.started_ts
            self.status.setText(f"{self.job.pct:5.1f}%  |  fps {last_fps:0.1f}  |  elapsed {int(elapsed)}s")
        if HAS_MPL:
            y = list(self.job.fps_hist); x = list(range(len(y)))
            self.line.set_data(x, y)
            if y:
                self.ax.set_xlim(0, max(10, len(y))); self.ax.set_ylim(0, max(y) * 1.2)
            self.canvas.draw_idle()
        else:
            blocks = "▁▂▃▄▅▆▇█"
            vals = list(self.job.fps_hist)[-40:]
            if vals:
                vmin, vmax = min(vals), max(vals)
                if vmax == vmin:
                    spar = blocks[0] * len(vals)
                else:
                    out = []
                    for v in vals:
                        idx = int(round((v - vmin) / (vmax - vmin) * (len(blocks) - 1)))
                        out.append(blocks[idx])
                    spar = "".join(out)
                self.canvas.setText(f"FPS: {vals[-1]:.1f}\n{spar}")
            else:
                self.canvas.setText("FPS: —")

# ------------------------------- Runner core ---------------------------------
class Runner(QtCore.QObject):
    job_updated = QtCore.Signal(int)
    job_finished = QtCore.Signal(int)

    def __init__(self, files: List[Path], parent=None):
        super().__init__(parent)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.makedirs(TMPDIR, exist_ok=True)

        check_tool("av1an", "install av1an")
        check_tool("mkvmerge", "install mkvtoolnix-cli")
        check_tool("numactl", "install numactl")
        check_tool("taskset", "install util-linux")

        self.chunk_method = USE_CHUNK_METHOD if detect_bestsource() else "ffmpeg"
        self.mem_nodes, self.node_ccx0, self.node_ccx1 = read_numa()
        if not self.mem_nodes:
            raise SystemExit("No usable memory nodes found; aborting.")

        self.jobs: List[Job] = []
        run_nodes = self.mem_nodes[:]
        for i, f in enumerate(files):
            base = f.stem
            j = Job(
                idx=i,
                infile=f,
                out_mkv=OUT_DIR / f"{base}-svt-av1.mkv",
                tempdir=Path(TMPDIR) / base,
                log_file=LOG_DIR / f"{base}.log",
                node=run_nodes[i % len(run_nodes)],
                ccx_idx=(i // len(run_nodes)) % 2,
                cpus=(
                    self.node_ccx0.get(run_nodes[i % len(run_nodes)], "")
                    if (i // len(run_nodes)) % 2 == 0
                    else self.node_ccx1.get(run_nodes[i % len(run_nodes)], "")
                ),
            )
            j.tempdir.mkdir(parents=True, exist_ok=True)
            if not j.cpus:
                raise SystemExit(f"No CPU list for node {j.node}/CCX{j.ccx_idx}; aborting.")
            self.jobs.append(j)

        self.queue: List[int] = list(range(len(self.jobs)))  # indices waiting to run
        self.running: Dict[int, Job] = {}
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / GUI_REFRESH_HZ))

    def _build_cmd(self, job: Job) -> List[str]:
        # Keep a chattier log-level to get parseable output (we still write a .log file, too).
        return [
            "bash", "-lc",
            shlex.join([
                "taskset", "-c", job.cpus,
                "numactl", f"--cpunodebind={job.node}", f"--membind={job.node}",
                "av1an", "-i", str(job.infile),
                "--encoder", "svt-av1",
                "-v", SVT_OPTS,
                "-w", str(W_PER_JOB),
                "-p", str(PASSES),
                "--log-level", "debug",
                "--keep",
                "--log-file", str(job.log_file),
                "--pix-format", "yuv420p10le",
                "--chunk-method", self.chunk_method,
                *RELIABILITY_OPTS,
                "--temp", str(job.tempdir),
                "--concat", "mkvmerge",
                "-o", str(job.out_mkv),
            ])
        ]

    def _start_next_if_possible(self):
        while len(self.running) < MAX_PAR and self.queue:
            idx = self.queue.pop(0)
            job = self.jobs[idx]
            cmd = self._build_cmd(job)
            try:
                # Capture stdout/stderr for live parsing
                job.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
            except Exception as e:
                job.failed = True
                job.done = True
                job.returncode = -1
                print(f"Failed to start {job.infile}: {e}")
                continue
            job.reader_thread = threading.Thread(
                target=self._drain_stdout, args=(job,), daemon=True
            )
            job.reader_thread.start()
            self.running[idx] = job

    def _drain_stdout(self, job: Job):
        proc = job.proc
        if not proc or not proc.stdout:
            return
        while True:
            try:
                line = proc.stdout.readline()
            except Exception:
                break
            if not line:
                break
            job.line_queue.put(line)

    def _parse_line_into_job(self, job: Job, line: str):
        # FPS (ignore stream descriptors)
        if not IGNORE_FPS_LINE.search(line):
            m = SPEED_FPS_RE.search(line)
            if m:
                try:
                    job.fps_hist.append(float(m.group(2)))
                except Exception:
                    pass
            else:
                m2 = ANY_FPS_RE.search(line)
                if m2 and not IGNORE_FPS_LINE.search(line):
                    try:
                        job.fps_hist.append(float(m2.group(1)))
                    except Exception:
                        pass
        # Percent (explicit or inferred)
        m = PCT_RE.search(line)
        if m:
            if m.group(1):
                whole = m.group(1)
                frac = m.group(2) or "0"
                try:
                    job.pct = max(0.0, min(100.0, float(f"{whole}.{frac}")))
                except Exception:
                    pass
            else:
                try:
                    job.pct = max(0.0, min(100.0, float(m.group(3))))
                except Exception:
                    pass
        else:
            m2 = ENC_RE.search(line)
            if m2:
                try:
                    x = float(m2.group(1))
                    y = float(m2.group(2))
                    if y > 0:
                        job.pct = (x / y) * 100.0
                except Exception:
                    pass
            else:
                m3 = CHUNK_RE.search(line)
                if m3:
                    try:
                        x = float(m3.group(1))
                        y = float(m3.group(2))
                        if y > 0:
                            job.pct = (x / y) * 100.0
                    except Exception:
                        pass

    def _read_stream_increment(self, job: Job):
        if not job.proc:
            return
        # Read up to N lines per tick so we don’t block the UI
        for _ in range(200):
            try:
                line = job.line_queue.get_nowait()
            except Empty:
                break
            self._parse_line_into_job(job, line)

    def _tick(self):
        # Start new jobs if slots available
        self._start_next_if_possible()

        # Update running jobs
        finished: List[int] = []
        for idx, job in list(self.running.items()):
            self._read_stream_increment(job)
            rc = job.proc.poll() if job.proc else 0
            if rc is not None:
                # Drain trailing lines
                self._read_stream_increment(job)
                job.done = True
                job.returncode = rc
                job.failed = (rc != 0)
                finished.append(idx)
            self.job_updated.emit(idx)

        # Reap finished and free slots for queued
        for idx in finished:
            self.running.pop(idx, None)
            self.job_finished.emit(idx)
        if finished:
            self._start_next_if_possible()

# --------------------------------- Main UI -----------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AV1 GUI Runner")
        self.resize(1200, 820)

        # Discover files
        files: List[Path] = []
        for g in INPUT_GLOBS:
            files.extend(Path.cwd().glob(g))
        files = sorted([f for f in files if f.is_file()])
        if not files:
            QtWidgets.QMessageBox.information(self, "No files", "No input files found in this folder.")
        self.runner = Runner(files)

        # Grid (2×2 when MAX_PAR=4)
        central = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(central)
        grid.setSpacing(10)
        self.setCentralWidget(central)

        cols = int(MAX_PAR ** 0.5)
        while cols and MAX_PAR % cols != 0:
            cols -= 1
        if cols == 0:
            cols = min(2, MAX_PAR)

        self.visible_slots: List[Optional[int]] = [None] * MAX_PAR  # slot -> job idx
        self.slot_widgets: List[Optional[JobTile]] = [None] * MAX_PAR
        self.grid = grid
        self.cols = cols

        # Pre-place placeholders
        for slot in range(MAX_PAR):
            ph = QtWidgets.QFrame()
            ph.setFrameShape(QtWidgets.QFrame.NoFrame)
            r = slot // cols
            c = slot % cols
            grid.addWidget(ph, r, c)

        # Signals
        self.runner.job_updated.connect(self.on_job_updated)
        self.runner.job_finished.connect(self.on_job_finished)

    def _ensure_tile_for(self, job_idx: int):
        if job_idx in self.visible_slots:
            return  # already placed
        # Find a free slot
        try:
            slot = self.visible_slots.index(None)
        except ValueError:
            # Reuse first finished slot
            for s, idx in enumerate(self.visible_slots):
                if idx is not None and self.runner.jobs[idx].done:
                    slot = s
                    break
            else:
                return
        job = self.runner.jobs[job_idx]
        tile = JobTile(job)
        self.slot_widgets[slot] = tile
        self.visible_slots[slot] = job_idx
        r = slot // self.cols
        c = slot % self.cols
        item = self.grid.itemAtPosition(r, c)
        if item:
            w = item.widget()
            if w:
                w.setParent(None)
        self.grid.addWidget(tile, r, c)

    @QtCore.Slot(int)
    def on_job_updated(self, job_idx: int):
        self._ensure_tile_for(job_idx)
        for slot, idx in enumerate(self.visible_slots):
            if idx == job_idx:
                tile = self.slot_widgets[slot]
                if tile:
                    tile.update_from_job()
                break

    @QtCore.Slot(int)
    def on_job_finished(self, job_idx: int):
        for slot, idx in enumerate(self.visible_slots):
            if idx == job_idx:
                tile = self.slot_widgets[slot]
                if tile:
                    tile.update_from_job()
                break

# ---------------------------------- entry ------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    def handle_sigint(*_):
        app.quit()
    signal.signal(signal.SIGINT, handle_sigint)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
