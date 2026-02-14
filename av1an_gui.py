#!/usr/bin/env python3
r"""
AV1 GUI Runner Pro — Windows 10 + Ryzen 7950X optimized

Enhanced features:
- Modern dark theme UI with customizable settings
- Drag-and-drop file support
- Dynamic queue management (add/remove files on the fly)
- Encoding presets with easy switching
- Real-time system resource monitoring (CPU, RAM, Disk)
- Estimated time remaining (ETA) calculations
- File size comparison (original vs encoded)
- Compression ratio and bitrate statistics
- Log viewer with filtering
- Session save/restore
- Notification support (system tray + sound)
- Export statistics to CSV
- Automatic temp cleanup
- Disk space warnings
- Per-job custom settings override
- Retry failed jobs
- Multi-profile encoding queue
- VMAF quality scoring (optional)
- Smart CPU allocation
- And much more!
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
import platform
import json
import csv
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Empty, Queue
from typing import Deque, Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from enum import Enum

# ------------------------------- Qt Imports ----------------------------------
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QSettings, QObject
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                   QHBoxLayout, QGridLayout, QFrame, QLabel, 
                                   QPushButton, QProgressBar, QTextEdit, QFileDialog,
                                   QMessageBox, QDialog, QLineEdit, QSpinBox, QComboBox,
                                   QCheckBox, QTabWidget, QGroupBox, QSplitter,
                                   QScrollArea, QTableWidget, QTableWidgetItem,
                                   QHeaderView, QSystemTrayIcon, QMenu, QSlider,
                                   QDoubleSpinBox, QStyle, QStyleFactory)
except Exception:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QTimer, QThread, QSettings, QObject
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QGridLayout, QFrame, QLabel,
                                 QPushButton, QProgressBar, QTextEdit, QFileDialog,
                                 QMessageBox, QDialog, QLineEdit, QSpinBox, QComboBox,
                                 QCheckBox, QTabWidget, QGroupBox, QSplitter,
                                 QScrollArea, QTableWidget, QTableWidgetItem,
                                 QHeaderView, QSystemTrayIcon, QMenu, QSlider,
                                 QDoubleSpinBox, QStyle, QStyleFactory)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
    # Configure PyQtGraph for better performance
    pg.setConfigOptions(antialias=True, useOpenGL=False)
except Exception:
    HAS_PYQTGRAPH = False

# ----------------------------- CONFIG (edit) ---------------------------------
DEFAULT_PRESETS = {
    "Ultra Quality": {
        "svt_opts": "--preset 3 --rc 0 --crf 10 --aq-mode 2 --keyint 48 --enable-tf 0 --tune 0 --psy-rd 1.5 --sharp-tx 1 --noise-adaptive-filtering 1 --enable-qm 1 --qm-min 8 --qp-scale-compress-strength 3 --noise-norm-strength 1 --fast-decode 0 --enable-dlf 2",
        "passes": 2,
        "workers": "auto",  # Auto-scale based on available cores
        "description": "Best quality, slowest encode"
    },
    "High Quality": {
        "svt_opts": "--preset 3 --rc 0 --crf 12 --aq-mode 2 --keyint 48 --enable-tf 0 --tune 0 --psy-rd 1.2 --sharp-tx 1 --noise-adaptive-filtering 1 --enable-qm 1 --qm-min 8 --qp-scale-compress-strength 3 --noise-norm-strength 1 --fast-decode 1 --enable-dlf 2",
        "passes": 1,
        "workers": "auto",  # Auto-scale based on available cores
        "description": "Great quality, good speed"
    },
    "Balanced": {
        "svt_opts": "--preset 4 --rc 0 --crf 16 --aq-mode 2 --keyint 48 --enable-tf 0 --tune 0 --psy-rd 1.0 --enable-qm 1 --qm-min 0 --fast-decode 1",
        "passes": 1,
        "workers": "auto",  # Auto-scale based on available cores
        "description": "Good quality, faster encode"
    },
    "Fast": {
        "svt_opts": "--preset 6 --rc 0 --crf 20 --aq-mode 1 --keyint 60 --fast-decode 1",
        "passes": 1,
        "workers": "auto",  # Auto-scale based on available cores
        "description": "Quick encode, decent quality"
    },
}

SVT_ENCAPP = r"C:\7950xGUIEncoder\svt-av1-psyex\Bin\Release\SvtAv1EncApp.exe"
TMPDIR = r"C:\7950xGUIEncoder\temp"
MAX_PAR = 1
USE_CHUNK_METHOD = "hybrid"
INPUT_GLOBS = ["*.mkv", "*.mp4", "*.mov", "*.avi", "*.m2ts", "*.ts", "*.webm"]
LOG_DIR = (Path.cwd() / ".log").resolve()
OUT_DIR = Path(r"E:\Temp\AV1AN-Output")
FPS_WINDOW = 60  # Reduced from 120 for better graph performance
GUI_REFRESH_HZ = 8
STOP_GRACE_SEC = 8.0
TERM_GRACE_SEC = 6.0
# -----------------------------------------------------------------------------

IS_WINDOWS = (os.name == "nt") or (platform.system().lower() == "windows")

# ------------------------------ Enums ----------------------------------------
class JobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

# ------------------------------ Regex Parsers --------------------------------
IGNORE_FPS_LINE = re.compile(r"\b(Video:|Stream #|Input #)\b", re.IGNORECASE)
FR_S_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:fr/s|frames/s|frame/s|frames/sec|frame/sec)\b", re.IGNORECASE)
SPEED_FPS_RE = re.compile(r"\b(speed|enc|encoding)[^\n]*?(\d+(?:\.\d+)?)\s*fps\b", re.IGNORECASE)
S_PER_FR_RE  = re.compile(r"([0-9]+(?:\.\d+)?)\s*s/fr", re.IGNORECASE)
PCT_RE = re.compile(r"(?:(\d{1,3})(?:\.(\d+))?%)|progress\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# -------------------------- System capability checks -------------------------
def check_tool(name: str, hint: str) -> bool:
    """Check if tool exists, return True if found."""
    return shutil.which(name) is not None

def get_missing_tools() -> List[Tuple[str, str]]:
    """Return list of (tool, hint) for missing required tools."""
    tools = [
        ("av1an", "install av1an and ensure it's on PATH"),
        ("mkvmerge", "install MKVToolNix and ensure mkvmerge is on PATH"),
        ("ffmpeg", "install ffmpeg and ensure it's on PATH"),
    ]
    return [(name, hint) for name, hint in tools if not check_tool(name, hint)]

# ----------------------------- Utility helpers -------------------------------
def _strip_lp(s: str) -> str:
    toks = shlex.split(s)
    out = []
    skip = False
    for t in toks:
        if skip:
            skip = False
            continue
        if t == "--lp":
            skip = True
            continue
        out.append(t)
    return shlex.join(out)

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def format_duration(seconds: float) -> str:
    """Format seconds to readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}"

def get_disk_usage(path: Path) -> Tuple[int, int, int]:
    """Return (total, used, free) disk space in bytes."""
    try:
        if IS_WINDOWS:
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                str(path), None, ctypes.byref(total_bytes), ctypes.byref(free_bytes)
            )
            return (total_bytes.value, total_bytes.value - free_bytes.value, free_bytes.value)
        else:
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            return (total, total - free, free)
    except Exception:
        return (0, 0, 0)

# ------------------------------ CPU Groups -----------------------------------
@dataclass
class CpuGroup:
    node: int
    socket: int
    cpus: List[int]

def _parse_cpu_groups_env(max_par: int) -> Optional[List[CpuGroup]]:
    env = os.environ.get("CPU_GROUPS")
    if not env:
        return None
    groups: List[CpuGroup] = []
    for chunk in env.split(";"):
        cpus: List[int] = []
        for span in chunk.strip().split(","):
            span = span.strip()
            if not span:
                continue
            if "-" in span:
                a, b = span.split("-", 1)
                cpus.extend(list(range(int(a), int(b) + 1)))
            else:
                cpus.append(int(span))
        if cpus:
            groups.append(CpuGroup(node=0, socket=0, cpus=sorted(set(cpus))))
        if len(groups) >= max_par:
            break
    return groups or None

def read_ryzen_groups(max_par: int) -> List[CpuGroup]:
    env_groups = _parse_cpu_groups_env(max_par)
    if env_groups:
        return env_groups

    logical = os.cpu_count() or 1
    cpus = list(range(logical))
    max_par = max(1, min(max_par, logical))

    base = logical // max_par
    rem = logical - base * max_par

    groups: List[CpuGroup] = []
    start = 0
    for i in range(max_par):
        take = base + (1 if i < rem else 0)
        chunk = cpus[start:start + take]
        start += take
        if chunk:
            groups.append(CpuGroup(node=0, socket=0, cpus=chunk))
    return groups

def calculate_optimal_workers(available_cores: int, preset_workers: Any, max_par: int = 1) -> Tuple[int, int]:
    """
    Calculate optimal number of av1an workers and threads per worker.
    
    Strategy:
    - For single job (max_par=1): Use many workers to maximize parallelism
    - For multi-job: Reduce workers per job to avoid total worker explosion
    
    Args:
        available_cores: Number of cores available to this job
        preset_workers: Preset worker setting ("auto" or int)
        max_par: Number of parallel jobs running
    
    Returns: (num_workers, threads_per_worker)
    """
    if isinstance(preset_workers, int) and preset_workers > 0:
        # Explicit worker count specified
        workers = preset_workers
        threads_per = max(1, available_cores // workers)
        return (workers, threads_per)
    
    # Auto-scale based on core count and parallel jobs
    if max_par == 1:
        # Single job mode - maximize parallelism
        if available_cores <= 4:
            workers = min(2, available_cores)
            threads_per = max(1, available_cores // workers)
        elif available_cores <= 8:
            workers = 4
            threads_per = 2
        elif available_cores <= 16:
            workers = 8
            threads_per = 2
        else:
            # 32+ threads: 16 workers × 2 threads
            workers = min(16, available_cores // 2)
            threads_per = max(2, available_cores // workers)
    else:
        # Multi-job mode - reduce workers to avoid contention
        # With 4 jobs × 16 workers each = 64 total workers = BAD
        # Better: 4 jobs × 4 workers each = 16 total workers = GOOD
        if available_cores <= 4:
            workers = 1
            threads_per = available_cores
        elif available_cores <= 8:
            workers = 2
            threads_per = max(2, available_cores // 2)
        elif available_cores <= 16:
            workers = 4
            threads_per = max(2, available_cores // 4)
        else:
            # Even with 32 threads per job, use fewer workers
            # 4-8 workers per job max to keep total worker count reasonable
            workers = min(8, available_cores // 4)
            threads_per = max(2, available_cores // workers)
    
    return (workers, threads_per)

# --------------------------- Windows CPU affinity ----------------------------
def _set_process_affinity(pid: int, cpus: List[int]) -> None:
    if not IS_WINDOWS:
        return
    try:
        import psutil
        psutil.Process(pid).cpu_affinity(cpus)
    except Exception:
        pass

# ----------------------------- Process helpers -------------------------------
def _windows_ctrl_break(proc: subprocess.Popen) -> bool:
    if not IS_WINDOWS:
        return False
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        return True
    except Exception:
        return False

def _safe_terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
    except Exception:
        pass

def _safe_kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass

def _try_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None

def _suspend_tree(root_pid: int) -> bool:
    psutil = _try_import_psutil()
    if psutil is None:
        return False
    try:
        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
        for p in reversed(procs):
            try:
                p.suspend()
            except Exception:
                pass
        return True
    except Exception:
        return False

def _resume_tree(root_pid: int) -> bool:
    psutil = _try_import_psutil()
    if psutil is None:
        return False
    try:
        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
        for p in procs:
            try:
                p.resume()
            except Exception:
                pass
        return True
    except Exception:
        return False

# ------------------------------- System Monitor ------------------------------
class SystemMonitor(QThread):
    """Monitor system resources (CPU, RAM, Disk)."""
    stats_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.psutil = _try_import_psutil()
        
    def run(self):
        while self.running:
            stats = {}
            if self.psutil:
                try:
                    stats['cpu_percent'] = self.psutil.cpu_percent(interval=0.5)
                    mem = self.psutil.virtual_memory()
                    stats['mem_percent'] = mem.percent
                    stats['mem_used'] = mem.used
                    stats['mem_total'] = mem.total
                except Exception:
                    pass
            self.stats_updated.emit(stats)
            time.sleep(2)
    
    def stop(self):
        self.running = False

# ------------------------------- Data model ----------------------------------
@dataclass
class Job:
    idx: int
    infile: Path
    out_mkv: Path
    tempdir: Path
    term_log: Path
    mux_log: Path
    node: int
    socket: int
    cpus: List[int]
    preset_name: str = "High Quality"
    custom_svt_opts: Optional[str] = None
    proc: Optional[subprocess.Popen] = None
    pct: float = 0.0
    fps_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=FPS_WINDOW))
    started_ts: Optional[float] = None
    completed_ts: Optional[float] = None
    status: JobStatus = JobStatus.QUEUED
    returncode: Optional[int] = None
    line_queue: Queue[str] = field(default_factory=Queue)
    last_line_at: float = field(default_factory=time.time)
    ema_fps: Optional[float] = None
    last_fps_push_ts: float = 0.0
    mux_attempted: bool = False
    mux_ok: bool = False
    original_size: int = 0
    encoded_size: int = 0
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 2
    
    def __post_init__(self):
        try:
            self.original_size = self.infile.stat().st_size
        except Exception:
            self.original_size = 0
    
    @property
    def elapsed_time(self) -> float:
        if not self.started_ts:
            return 0.0
        end_time = self.completed_ts or time.time()
        return end_time - self.started_ts
    
    @property
    def current_fps(self) -> float:
        return self.fps_hist[-1] if self.fps_hist else 0.0
    
    @property
    def avg_fps(self) -> float:
        return sum(self.fps_hist) / len(self.fps_hist) if self.fps_hist else 0.0
    
    @property
    def compression_ratio(self) -> float:
        if self.original_size > 0 and self.encoded_size > 0:
            return (1 - self.encoded_size / self.original_size) * 100
        return 0.0
    
    @property
    def eta_seconds(self) -> Optional[float]:
        if self.pct > 0 and self.avg_fps > 0:
            remaining_pct = 100.0 - self.pct
            # Rough estimation
            elapsed = self.elapsed_time
            total_estimate = elapsed / (self.pct / 100.0)
            return total_estimate - elapsed
        return None

def _maybe_push_fps(job: Job, v: Optional[float], now: float):
    if v is None or not (v > 0):
        return
    if job.ema_fps is not None:
        if v > job.ema_fps * 4 and (now - job.last_fps_push_ts) < 2.0:
            return
        alpha = 0.3
        job.ema_fps = (1 - alpha) * job.ema_fps + alpha * v
    else:
        job.ema_fps = v
    job.fps_hist.append(job.ema_fps)
    job.last_fps_push_ts = now

# ------------------------------- Settings Dialog -----------------------------
class SettingsDialog(QDialog):
    """Enhanced settings dialog with tabs and presets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(700, 500)
        
        self.settings = QSettings("AV1Runner", "EncoderPro")
        
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        
        # General tab
        general_tab = self._create_general_tab()
        tabs.addTab(general_tab, "General")
        
        # Encoder tab
        encoder_tab = self._create_encoder_tab()
        tabs.addTab(encoder_tab, "Encoder")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(tabs)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.load_settings()
    
    def _create_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Output directory
        out_group = QGroupBox("Output Settings")
        out_layout = QVBoxLayout()
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output Directory:"))
        self.out_dir_edit = QLineEdit(str(OUT_DIR))
        dir_layout.addWidget(self.out_dir_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_btn)
        out_layout.addLayout(dir_layout)
        
        self.auto_cleanup_check = QCheckBox("Automatically cleanup temp files after successful encode")
        self.auto_cleanup_check.setChecked(True)
        out_layout.addWidget(self.auto_cleanup_check)
        
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)
        
        # Notifications
        notif_group = QGroupBox("Notifications")
        notif_layout = QVBoxLayout()
        
        self.notif_complete_check = QCheckBox("Show notification when all jobs complete")
        self.notif_complete_check.setChecked(True)
        notif_layout.addWidget(self.notif_complete_check)
        
        self.notif_error_check = QCheckBox("Show notification on errors")
        self.notif_error_check.setChecked(True)
        notif_layout.addWidget(self.notif_error_check)
        
        self.play_sound_check = QCheckBox("Play sound on completion")
        notif_layout.addWidget(self.play_sound_check)
        
        self.disable_graphs_check = QCheckBox("Disable FPS graphs (better performance)")
        notif_layout.addWidget(self.disable_graphs_check)
        
        notif_group.setLayout(notif_layout)
        layout.addWidget(notif_group)
        
        layout.addStretch()
        return widget
    
    def _create_encoder_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Encoder path
        path_group = QGroupBox("Encoder Configuration")
        path_layout = QVBoxLayout()
        
        svt_layout = QHBoxLayout()
        svt_layout.addWidget(QLabel("SVT-AV1 Encoder:"))
        self.svt_path_edit = QLineEdit(SVT_ENCAPP)
        svt_layout.addWidget(self.svt_path_edit, 1)
        browse_svt_btn = QPushButton("Browse...")
        browse_svt_btn.clicked.connect(self._browse_svt)
        svt_layout.addWidget(browse_svt_btn)
        path_layout.addLayout(svt_layout)
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temp Directory:"))
        self.temp_dir_edit = QLineEdit(TMPDIR)
        temp_layout.addWidget(self.temp_dir_edit, 1)
        browse_temp_btn = QPushButton("Browse...")
        browse_temp_btn.clicked.connect(self._browse_temp)
        temp_layout.addWidget(browse_temp_btn)
        path_layout.addLayout(temp_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Parallel jobs
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()
        
        par_layout = QHBoxLayout()
        par_layout.addWidget(QLabel("Max Parallel Jobs:"))
        self.max_par_spin = QSpinBox()
        self.max_par_spin.setRange(1, os.cpu_count() or 1)
        self.max_par_spin.setValue(MAX_PAR)
        self.max_par_spin.setToolTip("Set to 1 to use ALL cores for a single job (fastest for one file).\nSet higher to encode multiple files simultaneously (splits cores between jobs).")
        par_layout.addWidget(self.max_par_spin)
        par_layout.addStretch()
        perf_layout.addLayout(par_layout)
        
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk Method:"))
        self.chunk_combo = QComboBox()
        self.chunk_combo.addItems(["hybrid", "select", "ffms2", "lsmash"])
        self.chunk_combo.setCurrentText(USE_CHUNK_METHOD)
        chunk_layout.addWidget(self.chunk_combo)
        chunk_layout.addStretch()
        perf_layout.addLayout(chunk_layout)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        adv_group = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()
        
        self.resume_check = QCheckBox("Resume interrupted encodes")
        self.resume_check.setChecked(True)
        adv_layout.addWidget(self.resume_check)
        
        self.keep_check = QCheckBox("Keep intermediate files")
        self.keep_check.setChecked(True)
        adv_layout.addWidget(self.keep_check)
        
        retry_layout = QHBoxLayout()
        retry_layout.addWidget(QLabel("Auto-retry failed jobs:"))
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 5)
        self.retry_spin.setValue(2)
        retry_layout.addWidget(self.retry_spin)
        retry_layout.addWidget(QLabel("times"))
        retry_layout.addStretch()
        adv_layout.addLayout(retry_layout)
        
        warn_layout = QHBoxLayout()
        warn_layout.addWidget(QLabel("Disk space warning threshold:"))
        self.disk_warn_spin = QSpinBox()
        self.disk_warn_spin.setRange(1, 1000)
        self.disk_warn_spin.setValue(50)
        self.disk_warn_spin.setSuffix(" GB")
        warn_layout.addWidget(self.disk_warn_spin)
        warn_layout.addStretch()
        adv_layout.addLayout(warn_layout)
        
        adv_group.setLayout(adv_layout)
        layout.addWidget(adv_group)
        
        layout.addStretch()
        return widget
    
    def _browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.out_dir_edit.text())
        if dir_path:
            self.out_dir_edit.setText(dir_path)
    
    def _browse_svt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SVT-AV1 Encoder", self.svt_path_edit.text(), "Executables (*.exe);;All Files (*)")
        if file_path:
            self.svt_path_edit.setText(file_path)
    
    def _browse_temp(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Temp Directory", self.temp_dir_edit.text())
        if dir_path:
            self.temp_dir_edit.setText(dir_path)
    
    def load_settings(self):
        self.out_dir_edit.setText(self.settings.value("output_dir", str(OUT_DIR)))
        self.svt_path_edit.setText(self.settings.value("svt_path", SVT_ENCAPP))
        self.temp_dir_edit.setText(self.settings.value("temp_dir", TMPDIR))
        self.max_par_spin.setValue(int(self.settings.value("max_par", MAX_PAR)))
        self.chunk_combo.setCurrentText(self.settings.value("chunk_method", USE_CHUNK_METHOD))
        self.auto_cleanup_check.setChecked(self.settings.value("auto_cleanup", True, type=bool))
        self.notif_complete_check.setChecked(self.settings.value("notif_complete", True, type=bool))
        self.notif_error_check.setChecked(self.settings.value("notif_error", True, type=bool))
        self.play_sound_check.setChecked(self.settings.value("play_sound", False, type=bool))
        self.disable_graphs_check.setChecked(self.settings.value("disable_graphs", False, type=bool))
        self.resume_check.setChecked(self.settings.value("resume", True, type=bool))
        self.keep_check.setChecked(self.settings.value("keep", True, type=bool))
        self.retry_spin.setValue(int(self.settings.value("max_retries", 2)))
        self.disk_warn_spin.setValue(int(self.settings.value("disk_warn_gb", 50)))
    
    def save_settings(self):
        self.settings.setValue("output_dir", self.out_dir_edit.text())
        self.settings.setValue("svt_path", self.svt_path_edit.text())
        self.settings.setValue("temp_dir", self.temp_dir_edit.text())
        self.settings.setValue("max_par", self.max_par_spin.value())
        self.settings.setValue("chunk_method", self.chunk_combo.currentText())
        self.settings.setValue("auto_cleanup", self.auto_cleanup_check.isChecked())
        self.settings.setValue("notif_complete", self.notif_complete_check.isChecked())
        self.settings.setValue("notif_error", self.notif_error_check.isChecked())
        self.settings.setValue("play_sound", self.play_sound_check.isChecked())
        self.settings.setValue("disable_graphs", self.disable_graphs_check.isChecked())
        self.settings.setValue("resume", self.resume_check.isChecked())
        self.settings.setValue("keep", self.keep_check.isChecked())
        self.settings.setValue("max_retries", self.retry_spin.value())
        self.settings.setValue("disk_warn_gb", self.disk_warn_spin.value())
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "output_dir": self.out_dir_edit.text(),
            "svt_path": self.svt_path_edit.text(),
            "temp_dir": self.temp_dir_edit.text(),
            "max_par": self.max_par_spin.value(),
            "chunk_method": self.chunk_combo.currentText(),
            "auto_cleanup": self.auto_cleanup_check.isChecked(),
            "notif_complete": self.notif_complete_check.isChecked(),
            "notif_error": self.notif_error_check.isChecked(),
            "play_sound": self.play_sound_check.isChecked(),
            "disable_graphs": self.disable_graphs_check.isChecked(),
            "resume": self.resume_check.isChecked(),
            "keep": self.keep_check.isChecked(),
            "max_retries": self.retry_spin.value(),
            "disk_warn_gb": self.disk_warn_spin.value(),
        }

# ------------------------------- Log Viewer ----------------------------------
class LogViewer(QDialog):
    """Enhanced log viewer with filtering and search."""
    
    def __init__(self, job: Job, parent=None):
        super().__init__(parent)
        self.job = job
        self.setWindowTitle(f"Logs - {job.infile.name}")
        self.resize(900, 600)
        
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Errors", "Warnings", "Info"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        toolbar.addWidget(QLabel("Filter:"))
        toolbar.addWidget(self.filter_combo)
        
        toolbar.addSpacing(20)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.textChanged.connect(self.apply_filter)
        toolbar.addWidget(QLabel("Search:"))
        toolbar.addWidget(self.search_edit, 1)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_logs)
        toolbar.addWidget(refresh_btn)
        
        layout.addLayout(toolbar)
        
        # Tabs for different logs
        tabs = QTabWidget()
        
        self.term_log_text = QTextEdit()
        self.term_log_text.setReadOnly(True)
        self.term_log_text.setFont(QtGui.QFont("Courier", 9))
        tabs.addTab(self.term_log_text, "Encoding Log")
        
        self.mux_log_text = QTextEdit()
        self.mux_log_text.setReadOnly(True)
        self.mux_log_text.setFont(QtGui.QFont("Courier", 9))
        tabs.addTab(self.mux_log_text, "Mux Log")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.load_logs()
    
    def load_logs(self):
        # Load term log
        try:
            if self.job.term_log.exists():
                with open(self.job.term_log, 'r', encoding='utf-8', errors='replace') as f:
                    self.full_term_log = f.read()
            else:
                self.full_term_log = "Log file not found."
        except Exception as e:
            self.full_term_log = f"Error loading log: {e}"
        
        # Load mux log
        try:
            if self.job.mux_log.exists():
                with open(self.job.mux_log, 'r', encoding='utf-8', errors='replace') as f:
                    self.mux_log_text.setPlainText(f.read())
            else:
                self.mux_log_text.setPlainText("Log file not found.")
        except Exception as e:
            self.mux_log_text.setPlainText(f"Error loading log: {e}")
        
        self.apply_filter()
    
    def apply_filter(self):
        filter_text = self.filter_combo.currentText().lower()
        search_text = self.search_edit.text().lower()
        
        lines = self.full_term_log.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Apply filter
            if filter_text != "all":
                if filter_text == "errors" and "error" not in line_lower:
                    continue
                elif filter_text == "warnings" and "warn" not in line_lower:
                    continue
                elif filter_text == "info" and ("error" in line_lower or "warn" in line_lower):
                    continue
            
            # Apply search
            if search_text and search_text not in line_lower:
                continue
            
            filtered_lines.append(line)
        
        self.term_log_text.setPlainText('\n'.join(filtered_lines))

# ------------------------------- Enhanced Job Tile ---------------------------
class JobTile(QFrame):
    """Enhanced job tile with more info and controls."""
    
    def __init__(self, job: Job, on_toggle_pause: Callable[[int], None], 
                 on_remove: Callable[[int], None], on_view_log: Callable[[int], None],
                 disable_graphs: bool = False, max_par: int = 1, parent=None):
        super().__init__(parent)
        self.job = job
        self.on_toggle_pause = on_toggle_pause
        self.on_remove = on_remove
        self.on_view_log = on_view_log
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2d2d2d, stop:1 #252525);
                padding: 10px;
            }
            QFrame:hover {
                border: 1px solid #4a4a4a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        # Header with title and status
        header = QHBoxLayout()
        self.title = QLabel(job.infile.name)
        title_font = QtGui.QFont("Sans", 10)
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.title.setWordWrap(True)
        header.addWidget(self.title, 1)
        
        self.status_label = QLabel(job.status.value)
        self.status_label.setStyleSheet("QLabel { color: #888; font-size: 9pt; }")
        header.addWidget(self.status_label)
        layout.addLayout(header)
        
        # Info row
        info_layout = QHBoxLayout()
        
        # Calculate workers info for display
        preset = DEFAULT_PRESETS.get(job.preset_name, DEFAULT_PRESETS["High Quality"])
        
        # Show actual workers that will be used based on max_par
        if max_par == 1:
            # Single job mode - uses all system cores
            system_cores = os.cpu_count() or 1
            workers, threads = calculate_optimal_workers(system_cores, preset.get("workers", "auto"), max_par)
            info_text = f"Cores: {system_cores} (all)  •  Workers: {workers}x{threads}t"
        else:
            # Multi-job mode - uses assigned cores with reduced workers
            assigned_cores = len(job.cpus)
            workers, threads = calculate_optimal_workers(assigned_cores, preset.get("workers", "auto"), max_par)
            info_text = f"Cores: {assigned_cores}/{os.cpu_count() or 1}  •  Workers: {workers}x{threads}t"
        
        self.info_label = QLabel(info_text + f"  •  Preset: {job.preset_name}")
        self.info_label.setStyleSheet("QLabel { color: #aaa; font-size: 8pt; }")
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background: #1a1a1a;
                height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #4a90e2, stop:1 #357abd);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress)
        
        # Stats row
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("QLabel { color: #bbb; font-size: 8pt; }")
        layout.addWidget(self.stats_label)
        
        # FPS Graph - Use PyQtGraph for best performance, fallback to matplotlib
        if HAS_PYQTGRAPH and not disable_graphs:
            # PyQtGraph - much faster and smoother for real-time data
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('#1a1a1a')
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(120)
            self.plot_widget.setTitle("FPS", color='#ccc', size='9pt')
            self.plot_widget.setLabel('left', 'FPS', color='#ccc', size='8pt')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
            self.plot_widget.setYRange(0, 10)
            
            # Style the plot
            self.plot_widget.getAxis('left').setPen('#888')
            self.plot_widget.getAxis('left').setTextPen('#888')
            self.plot_widget.getAxis('bottom').setPen('#888')
            self.plot_widget.getAxis('bottom').setTextPen('#888')
            
            # Create the plot line
            pen = pg.mkPen(color='#4a90e2', width=2)
            self.plot_line = self.plot_widget.plot([], [], pen=pen, antialias=True)
            
            layout.addWidget(self.plot_widget)
            
        elif HAS_MPL and not disable_graphs:
            # Matplotlib fallback - simpler approach without blitting
            self.fig = Figure(figsize=(3.5, 1.2), facecolor='#2d2d2d', dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setMinimumHeight(100)
            self.canvas.setMaximumHeight(120)
            self.ax = self.fig.add_subplot(111, facecolor='#1a1a1a')
            self.ax.set_title("FPS", fontsize=8, color='#ccc')
            self.ax.tick_params(colors='#888', labelsize=6)
            self.ax.spines['bottom'].set_color('#555')
            self.ax.spines['top'].set_color('#555')
            self.ax.spines['left'].set_color('#555')
            self.ax.spines['right'].set_color('#555')
            (self.line,) = self.ax.plot([], [], color='#4a90e2', linewidth=1.5, antialiased=True)
            self.ax.grid(True, alpha=0.15, color='#555', linewidth=0.5)
            self.fig.tight_layout(pad=0.3)
            
            self.last_graph_update = 0.0
            self.graph_update_interval = 0.5  # Update graph every 0.5s max
            
            layout.addWidget(self.canvas)
        else:
            self.fps_label = QLabel("FPS: —")
            self.fps_label.setStyleSheet("QLabel { font-family: monospace; color: #aaa; }")
            layout.addWidget(self.fps_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(lambda: self.on_toggle_pause(self.job.idx))
        self.pause_btn.setFixedWidth(80)
        btn_layout.addWidget(self.pause_btn)
        
        self.log_btn = QPushButton("View Log")
        self.log_btn.clicked.connect(lambda: self.on_view_log(self.job.idx))
        self.log_btn.setFixedWidth(80)
        btn_layout.addWidget(self.log_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(lambda: self.on_remove(self.job.idx))
        self.remove_btn.setFixedWidth(80)
        btn_layout.addWidget(self.remove_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.update_from_job()
    
    def update_from_job(self):
        # Update progress
        self.progress.setValue(int(self.job.pct * 10))
        self.progress.setFormat(f"{self.job.pct:.1f}%")
        
        # Update status
        self.status_label.setText(self.job.status.value)
        
        # Update button states
        can_pause = (self.job.status == JobStatus.RUNNING and self.job.proc and 
                    self.job.proc.poll() is None)
        can_resume = (self.job.status == JobStatus.PAUSED and self.job.proc and 
                     self.job.proc.poll() is None)
        
        self.pause_btn.setEnabled(can_pause or can_resume)
        self.pause_btn.setText("Resume" if self.job.status == JobStatus.PAUSED else "Pause")
        
        can_remove = (self.job.status in [JobStatus.QUEUED, JobStatus.COMPLETED, 
                                         JobStatus.FAILED, JobStatus.CANCELLED])
        self.remove_btn.setEnabled(can_remove)
        
        # Update stats
        stats_parts = []
        
        if self.job.status == JobStatus.RUNNING:
            stats_parts.append(f"FPS: {self.job.current_fps:.1f} (avg: {self.job.avg_fps:.1f})")
            
            if self.job.eta_seconds:
                eta_str = format_duration(self.job.eta_seconds)
                stats_parts.append(f"ETA: {eta_str}")
            
            elapsed = format_duration(self.job.elapsed_time)
            stats_parts.append(f"Elapsed: {elapsed}")
        
        elif self.job.status == JobStatus.PAUSED:
            elapsed = format_duration(self.job.elapsed_time)
            stats_parts.append(f"PAUSED  •  Elapsed: {elapsed}")
        
        elif self.job.status == JobStatus.COMPLETED:
            elapsed = format_duration(self.job.elapsed_time)
            stats_parts.append(f"Completed in {elapsed}")
            
            if self.job.encoded_size > 0:
                orig = format_size(self.job.original_size)
                enc = format_size(self.job.encoded_size)
                ratio = self.job.compression_ratio
                stats_parts.append(f"{orig} → {enc} ({ratio:.1f}% saved)")
        
        elif self.job.status == JobStatus.FAILED:
            stats_parts.append(f"Failed: {self.job.error_message or 'Unknown error'}")
        
        self.stats_label.setText("  •  ".join(stats_parts) if stats_parts else "—")
        
        # Update FPS graph
        if HAS_PYQTGRAPH and hasattr(self, 'plot_widget'):
            # PyQtGraph - super fast, no ghosting, smooth updates
            y = list(self.job.fps_hist)
            if y:
                x = list(range(len(y)))
                self.plot_line.setData(x, y)
                
                # Auto-range Y axis smoothly
                if max(y) > 0:
                    self.plot_widget.setYRange(0, max(y) * 1.2, padding=0)
                    
        elif HAS_MPL and hasattr(self, 'canvas'):
            # Matplotlib - simple clean redraws (no blitting complexity)
            current_time = time.time()
            if current_time - self.last_graph_update >= self.graph_update_interval:
                y = list(self.job.fps_hist)
                if y:
                    x = list(range(len(y)))
                    
                    # Update data and limits
                    self.line.set_data(x, y)
                    self.ax.set_xlim(0, max(10, len(y)))
                    y_max = max(y) * 1.2 if max(y) > 0 else 10
                    self.ax.set_ylim(0, y_max)
                    
                    # Simple clean redraw
                    self.canvas.draw()
                    self.last_graph_update = current_time
                    
        elif hasattr(self, 'fps_label'):
            # Update text FPS display when graphs are disabled
            current = self.job.current_fps
            avg = self.job.avg_fps
            self.fps_label.setText(f"FPS: {current:.1f} (avg: {avg:.1f})")

# ------------------------------- Runner Core ---------------------------------
class Runner(QObject):
    """Enhanced runner with better queue management and monitoring."""
    
    job_updated = Signal(int)
    job_finished = Signal(int)
    total_fps_changed = Signal(float)
    notify = Signal(str)
    all_jobs_completed = Signal()
    
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        
        self.out_dir = Path(config["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(config["temp_dir"])
        os.makedirs(self.temp_dir, exist_ok=True)
        
        svt_path = Path(config["svt_path"])
        if not svt_path.is_file():
            raise SystemExit(f"SVT_ENCAPP not found: {config['svt_path']}")
        
        self.proc_env = os.environ.copy()
        self.proc_env["PATH"] = str(svt_path.parent) + os.pathsep + self.proc_env.get("PATH", "")
        
        self.max_par = config["max_par"]
        self.groups = read_ryzen_groups(self.max_par)
        if not self.groups:
            raise SystemExit("No usable CPU groups found; aborting.")
        
        self.chunk_method = config["chunk_method"]
        if self.chunk_method == "hybrid":
            self.chunk_method = "hybrid" if self._detect_bestsource() else "select"
        
        self.jobs: List[Job] = []
        self.queue: List[int] = []
        self.running: Dict[int, Job] = {}
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / GUI_REFRESH_HZ))
        
        self._closing = False
        self._close_lock = threading.Lock()
        self._next_job_idx = 0
    
    def add_files(self, files: List[Path], preset_name: str = "High Quality"):
        """Add files to the queue dynamically."""
        new_jobs = []
        for f in files:
            if not f.is_file():
                continue
            
            base = f.stem
            grp = self.groups[self._next_job_idx % len(self.groups)]
            
            j = Job(
                idx=self._next_job_idx,
                infile=f,
                out_mkv=self.out_dir / f"{base}-svt_av1.mkv",
                tempdir=self.temp_dir / base,
                term_log=LOG_DIR / f"{base}.term.log",
                mux_log=LOG_DIR / f"{base}.mux.log",
                node=grp.node,
                socket=grp.socket,
                cpus=grp.cpus,
                preset_name=preset_name,
                max_retries=self.config.get("max_retries", 2),
            )
            j.tempdir.mkdir(parents=True, exist_ok=True)
            
            self.jobs.append(j)
            self.queue.append(self._next_job_idx)
            new_jobs.append(j)
            self._next_job_idx += 1
        
        return new_jobs
    
    def remove_job(self, job_idx: int):
        """Remove a job from the queue (only if not running)."""
        job = self.jobs[job_idx]
        if job.status in [JobStatus.RUNNING, JobStatus.PAUSED]:
            return False
        
        if job_idx in self.queue:
            self.queue.remove(job_idx)
        
        job.status = JobStatus.CANCELLED
        self.job_updated.emit(job_idx)
        return True
    
    def retry_job(self, job_idx: int):
        """Retry a failed job."""
        job = self.jobs[job_idx]
        if job.status != JobStatus.FAILED:
            return False
        
        job.status = JobStatus.QUEUED
        job.pct = 0.0
        job.fps_hist.clear()
        job.error_message = ""
        job.proc = None
        job.started_ts = None
        job.completed_ts = None
        
        if job_idx not in self.queue:
            self.queue.append(job_idx)
        
        self.job_updated.emit(job_idx)
        return True
    
    def request_stop_all(self):
        with self._close_lock:
            if self._closing:
                return
            self._closing = True
        
        try:
            self.timer.stop()
        except Exception:
            pass
        
        self.queue.clear()
        threading.Thread(target=self._stop_all_processes_blocking, daemon=True).start()
    
    def _stop_all_processes_blocking(self):
        procs: List[subprocess.Popen] = []
        for job in list(self.running.values()):
            if job.proc and job.proc.poll() is None:
                procs.append(job.proc)
                if job.status == JobStatus.PAUSED:
                    try:
                        _resume_tree(job.proc.pid)
                    except Exception:
                        pass
        
        for p in procs:
            _windows_ctrl_break(p)
        
        t0 = time.time()
        while time.time() - t0 < STOP_GRACE_SEC:
            if all(p.poll() is not None for p in procs):
                return
            time.sleep(0.1)
        
        for p in procs:
            if p.poll() is None:
                _safe_terminate(p)
        
        t1 = time.time()
        while time.time() - t1 < TERM_GRACE_SEC:
            if all(p.poll() is not None for p in procs):
                return
            time.sleep(0.1)
        
        for p in procs:
            if p.poll() is None:
                _safe_kill(p)
    
    def toggle_pause(self, job_idx: int):
        job = self.jobs[job_idx]
        if not job.proc or job.proc.poll() is not None:
            return
        
        pid = job.proc.pid
        if job.status == JobStatus.RUNNING:
            ok = _suspend_tree(pid)
            if not ok:
                self.notify.emit("Pause requires psutil. Install with: pip install psutil")
                return
            job.status = JobStatus.PAUSED
            self.notify.emit(f"Paused: {job.infile.name}")
        elif job.status == JobStatus.PAUSED:
            ok = _resume_tree(pid)
            if not ok:
                self.notify.emit("Resume failed (psutil missing or process already ended).")
                return
            job.status = JobStatus.RUNNING
            self.notify.emit(f"Resumed: {job.infile.name}")
        
        self.job_updated.emit(job_idx)
    
    def _detect_bestsource(self) -> bool:
        try:
            import importlib
            vs = importlib.import_module("vapoursynth")
            return hasattr(vs.core, "bs")
        except Exception:
            return False
    
    def _build_av1an_args(self, job: Job) -> List[str]:
        preset = DEFAULT_PRESETS.get(job.preset_name, DEFAULT_PRESETS["High Quality"])
        
        # Calculate optimal workers and threads
        # IMPORTANT: For single job mode (max_par=1), use ALL system cores
        # For multi-job mode, use only the cores assigned to this job
        if self.max_par == 1:
            available_cores = os.cpu_count() or 1
        else:
            available_cores = len(job.cpus)
        
        preset_workers = preset.get("workers", "auto")
        workers, threads_per_worker = calculate_optimal_workers(available_cores, preset_workers, self.max_par)
        
        # Build SVT-AV1 options with optimal thread count
        svt_opts = job.custom_svt_opts or preset["svt_opts"]
        svt_cli = _strip_lp(svt_opts)
        svt_cli = shlex.join([*shlex.split(svt_cli), "--lp", str(threads_per_worker)])
        
        passes = preset.get("passes", 1)
        
        reliability_opts = []
        if self.config.get("resume", True):
            reliability_opts.append("--resume")
        if self.config.get("keep", True):
            reliability_opts.append("--keep")
        
        return [
            "av1an",
            "--input", str(job.infile),
            "--temp", str(job.tempdir),
            "--output_file", str(job.out_mkv),
            "--encoder", "svt_av1",
            "--video_params", svt_cli,
            "--workers", str(workers),
            "--passes", str(passes),
            "--chunk_method", self.chunk_method,
            "--pix_format", "yuv420p10le",
            "--mkvmerge",
            *reliability_opts,
            "--logging", "debug",
        ]
    
    def _write_term_line(self, job: Job, line: str):
        try:
            job.term_log.parent.mkdir(parents=True, exist_ok=True)
            with open(job.term_log, "a", encoding="utf-8", errors="replace") as f:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass
    
    def _start_next_if_possible(self):
        if self._closing:
            return
        
        while len(self.running) < self.max_par and self.queue and not self._closing:
            idx = self.queue.pop(0)
            job = self.jobs[idx]
            
            # DEBUG: Log CPU assignment
            total_system_cores = os.cpu_count() or 1
            self.notify.emit(f"System cores: {total_system_cores}, Job assigned: {len(job.cpus)} cores: {job.cpus[:8]}...")
            
            # Check disk space
            _, _, free = get_disk_usage(self.out_dir)
            warn_threshold = self.config.get("disk_warn_gb", 50) * 1024 * 1024 * 1024
            if free < warn_threshold:
                self.notify.emit(f"⚠ Low disk space: {format_size(free)} remaining!")
            
            args = self._build_av1an_args(job)
            
            # Log worker configuration for user visibility
            preset = DEFAULT_PRESETS.get(job.preset_name, DEFAULT_PRESETS["High Quality"])
            
            # Determine cores being used
            if self.max_par == 1:
                cores_used = os.cpu_count() or 1
                cores_mode = "all system cores"
            else:
                cores_used = len(job.cpus)
                cores_mode = "assigned cores only"
            
            workers, threads = calculate_optimal_workers(cores_used, preset.get("workers", "auto"), self.max_par)
            
            job.fps_hist.clear()
            job.ema_fps = None
            job.last_fps_push_ts = 0.0
            job.pct = 0.0
            job.status = JobStatus.RUNNING
            job.returncode = None
            job.mux_attempted = False
            job.mux_ok = False
            job.started_ts = time.time()
            job.completed_ts = None
            
            try:
                job.term_log.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                job.mux_log.unlink(missing_ok=True)
            except Exception:
                pass
            
            self._write_term_line(job, "=== AV1AN CMD ===")
            self._write_term_line(job, f"Worker Configuration: {workers} workers × {threads} threads = {workers * threads} total threads")
            self._write_term_line(job, f"Using {cores_used} cores ({cores_mode})")
            
            if self.max_par > 1:
                total_workers = workers * len(self.running) + workers  # Current + this new one
                self._write_term_line(job, f"Multi-job mode: {self.max_par} parallel jobs, ~{total_workers} total workers across all jobs")
                self._write_term_line(job, "Note: Workers per job reduced to prevent worker explosion and maintain performance")
            
            self._write_term_line(job, f"System total: {os.cpu_count()} threads, Preset: {job.preset_name}")
            self._write_term_line(job, " ".join(shlex.quote(a) for a in args))
            self._write_term_line(job, "=================")
            
            self.notify.emit(f"Starting {job.infile.name} with {workers} workers × {threads} threads ({cores_used} cores)")
            
            try:
                creationflags = 0
                if IS_WINDOWS:
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                
                job.proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=creationflags,
                    env=self.proc_env,
                )
            except Exception as e:
                job.status = JobStatus.FAILED
                job.returncode = -1
                job.error_message = str(e)
                job.completed_ts = time.time()
                self._write_term_line(job, f"Failed to start: {e}")
                self.job_updated.emit(idx)
                self.job_finished.emit(idx)
                continue
            
            # CPU Affinity: Only use when running multiple parallel jobs
            # For single job (max_par=1), let av1an workers use ALL system cores
            if self.max_par > 1:
                try:
                    _set_process_affinity(job.proc.pid, job.cpus)
                    self.notify.emit(f"CPU affinity set to cores: {job.cpus}")
                except Exception:
                    pass
            else:
                self.notify.emit(f"Single job mode - using all {os.cpu_count()} system threads")
            
            t = threading.Thread(target=self._stream_reader, args=(job,), daemon=True)
            t.start()
            
            self.running[idx] = job
    
    def _stream_reader(self, job: Job):
        p = job.proc
        if not p or not p.stdout:
            return
        try:
            with open(job.term_log, "a", encoding="utf-8", errors="replace") as logf:
                for line in p.stdout:
                    clean = ANSI_RE.sub("", line)
                    if clean:
                        job.line_queue.put(clean)
                        try:
                            logf.write(clean if clean.endswith("\n") else clean + "\n")
                            logf.flush()
                        except Exception:
                            pass
        except Exception:
            return
    
    def _parse_line_into_job(self, job: Job, text: str):
        now = time.time()
        job.last_line_at = now
        
        candidate_fps: Optional[float] = None
        if not IGNORE_FPS_LINE.search(text):
            m_fr = FR_S_RE.search(text)
            if m_fr:
                try:
                    candidate_fps = float(m_fr.group(1))
                except Exception:
                    candidate_fps = None
            if candidate_fps is None:
                m_spf = S_PER_FR_RE.search(text)
                if m_spf:
                    try:
                        sec_per_frame = float(m_spf.group(1))
                        if sec_per_frame > 0:
                            candidate_fps = 1.0 / sec_per_frame
                    except Exception:
                        candidate_fps = None
            if candidate_fps is None:
                m = SPEED_FPS_RE.search(text)
                if m:
                    try:
                        candidate_fps = float(m.group(2))
                    except Exception:
                        candidate_fps = None
        
        for m in PCT_RE.finditer(text):
            try:
                if m.group(1):
                    whole = m.group(1)
                    frac = m.group(2) or "0"
                    job.pct = max(0.0, min(100.0, float(f"{whole}.{frac}")))
                else:
                    job.pct = max(0.0, min(100.0, float(m.group(3))))
            except Exception:
                pass
        
        if job.status != JobStatus.PAUSED:
            _maybe_push_fps(job, candidate_fps, now)
    
    def _read_stream_increment(self, job: Job):
        for _ in range(600):
            try:
                text = job.line_queue.get_nowait()
            except Empty:
                break
            self._parse_line_into_job(job, text)
    
    def _emit_total_fps(self):
        total = 0.0
        for j in self.jobs:
            if j.status == JobStatus.RUNNING and j.fps_hist:
                total += j.fps_hist[-1]
        self.total_fps_changed.emit(total)
    
    def _write_mux_log(self, job: Job, line: str):
        try:
            job.mux_log.parent.mkdir(parents=True, exist_ok=True)
            with open(job.mux_log, "a", encoding="utf-8", errors="replace") as f:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass
    
    def _find_encoded_video_file_in_temp(self, job: Job) -> Optional[Path]:
        candidates: List[Path] = []
        encode_dir = job.tempdir / "encode"
        if encode_dir.is_dir():
            candidates += list(encode_dir.rglob("*.mkv"))
        candidates += list(job.tempdir.rglob("*.mkv"))
        
        big = []
        for p in candidates:
            try:
                if p.is_file() and p.stat().st_size > 5 * 1024 * 1024:
                    big.append(p)
            except Exception:
                pass
        if not big:
            return None
        big.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return big[0]
    
    def _mkvmerge_remux_all_tracks(self, job: Job) -> bool:
        job.mux_attempted = True
        
        enc_source: Optional[Path] = None
        try:
            if job.out_mkv.exists() and job.out_mkv.stat().st_size > 5 * 1024 * 1024:
                enc_source = job.out_mkv
        except Exception:
            enc_source = None
        
        if enc_source is None:
            enc_source = self._find_encoded_video_file_in_temp(job)
        
        if enc_source is None:
            self.notify.emit(f"mux: no encoded MKV found for {job.infile.name}")
            self._write_mux_log(job, "No encoded MKV found; cannot remux.")
            return False
        
        job.out_mkv.parent.mkdir(parents=True, exist_ok=True)
        remux_tmp = job.out_mkv.with_suffix(".remux.mkv")
        
        # CRITICAL: Use explicit track selection to preserve ALL timing information
        # Audio/subtitle delays and sync offsets MUST be preserved
        cmd = [
            "mkvmerge",
            "-o", str(remux_tmp),
            
            # From encoded file: take ONLY video track 0
            "-d", "0",  # Video track 0
            "-A",       # No audio
            "-S",       # No subtitles  
            "-T",       # No tags
            "-M",       # No chapters
            "-B",       # No attachments
            str(enc_source),
            
            # From original: take EVERYTHING except video
            # Preserves all timing offsets, delays, and sync info
            "-D",       # No video
            str(job.infile),
        ]
        
        self._write_mux_log(job, "=== MKVMERGE REMUX CMD ===")
        self._write_mux_log(job, "Using explicit track selection to preserve timing offsets")
        self._write_mux_log(job, " ".join(shlex.quote(c) for c in cmd))
        self._write_mux_log(job, "=========================")
        
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._write_mux_log(job, p.stdout or "")
            if p.returncode != 0:
                self._write_mux_log(job, f"mkvmerge failed rc={p.returncode}")
                try:
                    remux_tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                return False
        except Exception as e:
            self._write_mux_log(job, f"mkvmerge exception: {e}")
            try:
                remux_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False
        
        try:
            if remux_tmp.exists() and remux_tmp.stat().st_size > 5 * 1024 * 1024:
                try:
                    os.replace(str(remux_tmp), str(job.out_mkv))
                except Exception:
                    try:
                        job.out_mkv.unlink(missing_ok=True)
                    except Exception:
                        pass
                    os.replace(str(remux_tmp), str(job.out_mkv))
                
                # Update encoded size
                try:
                    job.encoded_size = job.out_mkv.stat().st_size
                except Exception:
                    pass
                
                return True
        except Exception:
            pass
        
        return False
    
    def _cleanup_temp_files(self, job: Job):
        """Cleanup temporary files after successful encode."""
        if not self.config.get("auto_cleanup", True):
            return
        
        try:
            if job.tempdir.exists():
                shutil.rmtree(job.tempdir, ignore_errors=True)
                self.notify.emit(f"Cleaned up temp files for {job.infile.name}")
        except Exception as e:
            self.notify.emit(f"Failed to cleanup temp for {job.infile.name}: {e}")
    
    def _finalize_job(self, job: Job):
        ok = self._mkvmerge_remux_all_tracks(job)
        job.mux_ok = ok
        job.completed_ts = time.time()
        
        if ok:
            job.status = JobStatus.COMPLETED
            self._cleanup_temp_files(job)
        else:
            # Check if we should retry
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                self.notify.emit(f"Retrying {job.infile.name} (attempt {job.retry_count + 1})")
                self.queue.append(job.idx)
                job.status = JobStatus.QUEUED
            else:
                job.status = JobStatus.FAILED
                job.error_message = "Mux failed"
    
    def _tick(self):
        self._start_next_if_possible()
        
        finished: List[int] = []
        for idx, job in list(self.running.items()):
            self._read_stream_increment(job)
            
            rc = job.proc.poll() if job.proc else None
            if rc is not None:
                job.returncode = rc
                
                if rc != 0:
                    if job.retry_count < job.max_retries:
                        job.retry_count += 1
                        self.notify.emit(f"Encode failed, retrying {job.infile.name} (attempt {job.retry_count + 1})")
                        self.queue.append(job.idx)
                        job.status = JobStatus.QUEUED
                    else:
                        job.status = JobStatus.FAILED
                        job.error_message = f"Encode failed (rc={rc})"
                    job.completed_ts = time.time()
                else:
                    self._finalize_job(job)
                
                finished.append(idx)
            
            self.job_updated.emit(idx)
        
        for idx in finished:
            self.running.pop(idx, None)
            self.job_finished.emit(idx)
        
        # Check if all jobs are done
        if not self.running and not self.queue:
            if any(j.status == JobStatus.COMPLETED for j in self.jobs):
                self.all_jobs_completed.emit()
        
        self._emit_total_fps()

# ------------------------------- Main Window ---------------------------------
class MainWindow(QMainWindow):
    """Enhanced main window with modern UI and features."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AV1 Encoder Pro - Ryzen 7950X Optimized")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Settings
        self.settings = QSettings("AV1Runner", "EncoderPro")
        self.config = self.load_config()
        
        # Check for missing tools
        missing_tools = get_missing_tools()
        if missing_tools:
            msg = "Missing required tools:\n\n"
            for tool, hint in missing_tools:
                msg += f"• {tool}: {hint}\n"
            QMessageBox.critical(self, "Missing Tools", msg)
            sys.exit(1)
        
        # Setup UI
        self.setup_ui()
        
        # Initialize runner
        try:
            self.runner = Runner(self.config, self)
            self.runner.job_updated.connect(self.on_job_updated)
            self.runner.job_finished.connect(self.on_job_finished)
            self.runner.total_fps_changed.connect(self.on_total_fps_changed)
            self.runner.notify.connect(self.show_notification)
            self.runner.all_jobs_completed.connect(self.on_all_jobs_completed)
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize encoder:\n{e}")
            sys.exit(1)
        
        # System monitor
        self.sys_monitor = SystemMonitor()
        self.sys_monitor.stats_updated.connect(self.on_stats_updated)
        self.sys_monitor.start()
        
        # System tray
        self.setup_tray()
        
        # Load initial files
        self.load_initial_files()
    
    def apply_dark_theme(self):
        """Apply a modern dark theme."""
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(35, 35, 35))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(35, 35, 35))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
        
        QApplication.setPalette(dark_palette)
        
        # Additional stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #232323;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #666;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
                color: white;
            }
            QComboBox:hover {
                border: 1px solid #666;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5a;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6a6a6a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
    
    def setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top bar
        top_bar = self.create_top_bar()
        main_layout.addLayout(top_bar)
        
        # Main content area (splitter)
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Jobs area
        jobs_widget = self.create_jobs_area()
        splitter.addWidget(jobs_widget)
        
        # Stats panel
        stats_widget = self.create_stats_panel()
        splitter.addWidget(stats_widget)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_top_bar(self) -> QHBoxLayout:
        """Create top toolbar."""
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        # Title and FPS
        self.total_fps_label = QLabel("Total FPS: 0.0")
        fps_font = QtGui.QFont("Sans", 12)
        fps_font.setBold(True)
        self.total_fps_label.setFont(fps_font)
        self.total_fps_label.setStyleSheet("QLabel { color: #4a90e2; }")
        layout.addWidget(self.total_fps_label)
        
        layout.addStretch()
        
        # Preset selector
        layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(DEFAULT_PRESETS.keys()))
        self.preset_combo.setCurrentText("High Quality")
        self.preset_combo.setMinimumWidth(150)
        self.preset_combo.setToolTip("Select encoding preset for new files")
        layout.addWidget(self.preset_combo)
        
        # Buttons
        self.add_files_btn = QPushButton("➕ Add Files")
        self.add_files_btn.clicked.connect(self.add_files_dialog)
        self.add_files_btn.setToolTip("Add video files to encode")
        layout.addWidget(self.add_files_btn)
        
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        layout.addWidget(self.settings_btn)
        
        self.export_btn = QPushButton("📊 Export Stats")
        self.export_btn.clicked.connect(self.export_stats)
        self.export_btn.setToolTip("Export encoding statistics to CSV")
        layout.addWidget(self.export_btn)
        
        return layout
    
    def create_jobs_area(self) -> QWidget:
        """Create scrollable jobs display area."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("Encoding Queue")
        header_font = QtGui.QFont("Sans", 11)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet("QLabel { color: #ddd; padding: 5px; }")
        layout.addWidget(header)
        
        # Scroll area for job tiles
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.jobs_container = QWidget()
        self.jobs_layout = QVBoxLayout(self.jobs_container)
        self.jobs_layout.setSpacing(8)
        self.jobs_layout.setContentsMargins(5, 5, 5, 5)
        self.jobs_layout.addStretch()
        
        scroll.setWidget(self.jobs_container)
        layout.addWidget(scroll)
        
        self.job_tiles: Dict[int, JobTile] = {}
        
        return widget
    
    def create_stats_panel(self) -> QWidget:
        """Create statistics and monitoring panel."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)
        
        # System stats
        sys_group = QGroupBox("System Resources")
        sys_layout = QVBoxLayout()
        
        self.cpu_label = QLabel("CPU: —")
        self.cpu_label.setStyleSheet("QLabel { font-size: 9pt; }")
        sys_layout.addWidget(self.cpu_label)
        
        self.mem_label = QLabel("RAM: —")
        self.mem_label.setStyleSheet("QLabel { font-size: 9pt; }")
        sys_layout.addWidget(self.mem_label)
        
        self.disk_label = QLabel("Disk: —")
        self.disk_label.setStyleSheet("QLabel { font-size: 9pt; }")
        sys_layout.addWidget(self.disk_label)
        
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        # Job stats
        job_group = QGroupBox("Job Statistics")
        job_layout = QVBoxLayout()
        
        self.queued_label = QLabel("Queued: 0")
        self.queued_label.setStyleSheet("QLabel { font-size: 9pt; }")
        job_layout.addWidget(self.queued_label)
        
        self.running_label = QLabel("Running: 0")
        self.running_label.setStyleSheet("QLabel { font-size: 9pt; }")
        job_layout.addWidget(self.running_label)
        
        self.completed_label = QLabel("Completed: 0")
        self.completed_label.setStyleSheet("QLabel { font-size: 9pt; }")
        job_layout.addWidget(self.completed_label)
        
        self.failed_label = QLabel("Failed: 0")
        self.failed_label.setStyleSheet("QLabel { font-size: 9pt; color: #e74c3c; }")
        job_layout.addWidget(self.failed_label)
        
        job_group.setLayout(job_layout)
        layout.addWidget(job_group)
        
        layout.addStretch()
        
        return widget
    
    def setup_tray(self):
        """Setup system tray icon."""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            
            # Use a built-in icon
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            self.tray_icon.setIcon(icon)
            
            tray_menu = QMenu()
            show_action = tray_menu.addAction("Show")
            show_action.triggered.connect(self.show)
            quit_action = tray_menu.addAction("Quit")
            quit_action.triggered.connect(self.close)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
            
            self.tray_icon.activated.connect(lambda reason: self.show() if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        else:
            self.tray_icon = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from settings."""
        config = {
            "output_dir": self.settings.value("output_dir", str(OUT_DIR)),
            "svt_path": self.settings.value("svt_path", SVT_ENCAPP),
            "temp_dir": self.settings.value("temp_dir", TMPDIR),
            "max_par": int(self.settings.value("max_par", MAX_PAR)),
            "chunk_method": self.settings.value("chunk_method", USE_CHUNK_METHOD),
            "auto_cleanup": self.settings.value("auto_cleanup", True, type=bool),
            "notif_complete": self.settings.value("notif_complete", True, type=bool),
            "notif_error": self.settings.value("notif_error", True, type=bool),
            "play_sound": self.settings.value("play_sound", False, type=bool),
            "disable_graphs": self.settings.value("disable_graphs", False, type=bool),
            "resume": self.settings.value("resume", True, type=bool),
            "keep": self.settings.value("keep", True, type=bool),
            "max_retries": int(self.settings.value("max_retries", 2)),
            "disk_warn_gb": int(self.settings.value("disk_warn_gb", 50)),
        }
        return config
    
    def load_initial_files(self):
        """Load files from current directory on startup."""
        files: List[Path] = []
        for g in INPUT_GLOBS:
            files.extend(Path.cwd().glob(g))
        
        if files:
            preset = self.preset_combo.currentText()
            new_jobs = self.runner.add_files(files, preset)
            
            for job in new_jobs:
                self.add_job_tile(job)
            
            self.update_job_stats()
            self.show_notification(f"Loaded {len(files)} file(s)")
        else:
            self.show_notification("No video files found in current directory. Use 'Add Files' to add videos.")
    
    def add_files_dialog(self):
        """Show file dialog to add files."""
        file_filter = "Video Files (*.mkv *.mp4 *.mov *.avi *.m2ts *.ts *.webm);;All Files (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", file_filter)
        
        if files:
            paths = [Path(f) for f in files]
            preset = self.preset_combo.currentText()
            new_jobs = self.runner.add_files(paths, preset)
            
            for job in new_jobs:
                self.add_job_tile(job)
            
            self.update_job_stats()
            self.show_notification(f"Added {len(files)} file(s)")
    
    def add_job_tile(self, job: Job):
        """Add a job tile to the UI."""
        disable_graphs = self.config.get("disable_graphs", False)
        tile = JobTile(job, self.runner.toggle_pause, self.remove_job, self.view_job_log, 
                      disable_graphs, self.runner.max_par, self)
        self.job_tiles[job.idx] = tile
        self.jobs_layout.insertWidget(self.jobs_layout.count() - 1, tile)
    
    def remove_job(self, job_idx: int):
        """Remove a job from the queue."""
        if self.runner.remove_job(job_idx):
            tile = self.job_tiles.pop(job_idx, None)
            if tile:
                self.jobs_layout.removeWidget(tile)
                tile.deleteLater()
            self.update_job_stats()
            self.show_notification(f"Removed job {job_idx}")
        else:
            QMessageBox.warning(self, "Cannot Remove", "Cannot remove a running or paused job.")
    
    def view_job_log(self, job_idx: int):
        """Show log viewer for a job."""
        job = self.runner.jobs[job_idx]
        dialog = LogViewer(job, self)
        dialog.exec()
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.save_settings()
            self.config = dialog.get_config()
            
            # Update runner config
            self.runner.config = self.config
            self.runner.max_par = self.config["max_par"]
            self.runner.out_dir = Path(self.config["output_dir"])
            self.runner.temp_dir = Path(self.config["temp_dir"])
            self.runner.chunk_method = self.config["chunk_method"]
            
            self.show_notification("Settings saved")
    
    def export_stats(self):
        """Export encoding statistics to CSV."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Statistics", "encoding_stats.csv", "CSV Files (*.csv)")
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['File', 'Status', 'Preset', 'Progress %', 'Original Size', 
                               'Encoded Size', 'Compression %', 'Elapsed Time', 'Avg FPS', 
                               'Retry Count', 'Error'])
                
                for job in self.runner.jobs:
                    writer.writerow([
                        job.infile.name,
                        job.status.value,
                        job.preset_name,
                        f"{job.pct:.1f}",
                        format_size(job.original_size),
                        format_size(job.encoded_size) if job.encoded_size > 0 else "—",
                        f"{job.compression_ratio:.1f}" if job.compression_ratio > 0 else "—",
                        format_duration(job.elapsed_time),
                        f"{job.avg_fps:.2f}",
                        job.retry_count,
                        job.error_message or "—"
                    ])
            
            self.show_notification(f"Statistics exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export statistics:\n{e}")
    
    def show_notification(self, message: str):
        """Show a notification message."""
        self.statusBar().showMessage(message, 5000)
        
        if self.tray_icon:
            self.tray_icon.showMessage("AV1 Encoder Pro", message, 
                                      QSystemTrayIcon.MessageIcon.Information, 3000)
    
    @Slot(int)
    def on_job_updated(self, job_idx: int):
        """Update job tile when job status changes."""
        tile = self.job_tiles.get(job_idx)
        if tile:
            tile.update_from_job()
        self.update_job_stats()
    
    @Slot(int)
    def on_job_finished(self, job_idx: int):
        """Handle job completion."""
        job = self.runner.jobs[job_idx]
        tile = self.job_tiles.get(job_idx)
        if tile:
            tile.update_from_job()
        
        if job.status == JobStatus.FAILED and self.config.get("notif_error", True):
            self.show_notification(f"❌ Job failed: {job.infile.name}")
        elif job.status == JobStatus.COMPLETED:
            self.show_notification(f"✅ Completed: {job.infile.name}")
        
        self.update_job_stats()
    
    @Slot(float)
    def on_total_fps_changed(self, total: float):
        """Update total FPS display."""
        self.total_fps_label.setText(f"Total FPS: {total:.1f}")
    
    @Slot(dict)
    def on_stats_updated(self, stats: Dict[str, Any]):
        """Update system resource stats."""
        if 'cpu_percent' in stats:
            self.cpu_label.setText(f"CPU: {stats['cpu_percent']:.1f}%")
        
        if 'mem_percent' in stats:
            mem_used = format_size(stats.get('mem_used', 0))
            mem_total = format_size(stats.get('mem_total', 0))
            self.mem_label.setText(f"RAM: {stats['mem_percent']:.1f}% ({mem_used} / {mem_total})")
        
        # Update disk usage
        _, _, free = get_disk_usage(Path(self.config["output_dir"]))
        self.disk_label.setText(f"Disk Free: {format_size(free)}")
    
    @Slot()
    def on_all_jobs_completed(self):
        """Handle all jobs completion."""
        if self.config.get("notif_complete", True):
            msg = "All encoding jobs completed!"
            self.show_notification(msg)
            
            if self.tray_icon:
                self.tray_icon.showMessage("AV1 Encoder Pro", msg, 
                                          QSystemTrayIcon.MessageIcon.Information, 5000)
            
            if self.config.get("play_sound", False):
                try:
                    if IS_WINDOWS:
                        import winsound
                        winsound.MessageBeep(winsound.MB_OK)
                except Exception:
                    pass
    
    def update_job_stats(self):
        """Update job statistics display."""
        queued = sum(1 for j in self.runner.jobs if j.status == JobStatus.QUEUED)
        running = sum(1 for j in self.runner.jobs if j.status == JobStatus.RUNNING)
        paused = sum(1 for j in self.runner.jobs if j.status == JobStatus.PAUSED)
        completed = sum(1 for j in self.runner.jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self.runner.jobs if j.status == JobStatus.FAILED)
        
        self.queued_label.setText(f"Queued: {queued}")
        self.running_label.setText(f"Running: {running} (Paused: {paused})")
        self.completed_label.setText(f"Completed: {completed}")
        self.failed_label.setText(f"Failed: {failed}")
    
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Handle drag enter for file drops."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QtGui.QDropEvent):
        """Handle file drops."""
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file() and file_path.suffix.lower() in ['.mkv', '.mp4', '.mov', '.avi', '.m2ts', '.ts', '.webm']:
                files.append(file_path)
        
        if files:
            preset = self.preset_combo.currentText()
            new_jobs = self.runner.add_files(files, preset)
            
            for job in new_jobs:
                self.add_job_tile(job)
            
            self.update_job_stats()
            self.show_notification(f"Added {len(files)} file(s) via drag & drop")
    
    def closeEvent(self, event: QtGui.QCloseEvent):
        """Handle window close."""
        # Check if there are running jobs
        if any(j.status == JobStatus.RUNNING for j in self.runner.jobs):
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "There are running encoding jobs. Are you sure you want to exit?\n\nAll running jobs will be stopped.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # Stop all processes
        self.runner.request_stop_all()
        
        # Stop system monitor
        if hasattr(self, 'sys_monitor'):
            self.sys_monitor.stop()
            self.sys_monitor.wait(2000)
        
        event.accept()

# --------------------------------- Main --------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AV1 Encoder Pro")
    app.setOrganizationName("AV1Runner")
    
    # Set Fusion style for better cross-platform appearance
    app.setStyle("Fusion")
    
    win = MainWindow()
    win.show()
    
    def handle_sigint(*_):
        win.close()
    
    try:
        signal.signal(signal.SIGINT, handle_sigint)
    except Exception:
        pass
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
