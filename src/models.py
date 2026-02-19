from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Deque
from collections import deque
from queue import Queue
import subprocess
import time
from config import FPS_WINDOW

class JobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    MUXING = "Muxing"
    VMAF = "VMAF"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    RESTARTING = "Restarting"

@dataclass
class CpuGroup:
    node: int
    socket: int
    cpus: List[int]

@dataclass
class Job:
    idx: int
    infile: Path
    out_mkv: Path
    tempdir: Path
    
    # Log Paths
    term_log: Path
    mux_log: Path
    vmaf_log: Path  # <--- NEW FIELD
    
    preset_name: str = "High Quality"
    custom_svt_opts: Optional[str] = None
    
    initial_workers: int = 0
    initial_threads: int = 0

    # Runtime State
    total_frames: int = 0
    frames_done: int = 0
    cpus: List[int] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = None
    mux_proc: Optional[subprocess.Popen] = None
    vmaf_proc: Optional[subprocess.Popen] = None
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
    status_text: str = ""
    retry_count: int = 0
    max_retries: int = 2
    
    # VMAF Stats
    vmaf_score: float = 0.0
    vmaf_1_percent: float = 0.0
    vmaf_01_percent: float = 0.0
    
    log_read_offset: int = 0
    
    def __post_init__(self):
        try:
            self.original_size = self.infile.stat().st_size
        except Exception:
            self.original_size = 0
    
    @property
    def elapsed_time(self) -> float:
        if not self.started_ts: return 0.0
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
        if self.status not in [JobStatus.RUNNING, JobStatus.VMAF]:
            return None

        if self.total_frames > 0 and self.avg_fps > 0.1:
            remaining = self.total_frames - self.frames_done
            if remaining < 0: remaining = 0
            return remaining / self.avg_fps

        if self.pct > 0 and self.avg_fps > 0:
            elapsed = self.elapsed_time
            total_estimated = elapsed / (self.pct / 100.0)
            return max(0.0, total_estimated - elapsed)
            
        return None