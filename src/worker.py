import os
import sys
import datetime
import shutil
import signal
import subprocess
import threading
import time
import shlex
import stat
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from queue import Empty

from ui.qt import QObject, Signal, QThread, QTimer

try:
    import psutil
except ImportError:
    psutil = None

from config import (
    DEFAULT_PRESETS, MAX_JOBS_CAP, BLOCKED_CPUS, IS_WINDOWS,
    DEFAULT_AV1AN_PATH, AV1AN_PROGRESS_RE, FFMPEG_PROGRESS_RE,
    LOG_DIR, GUI_REFRESH_HZ
)
from models import Job, JobStatus, CpuGroup

# --- Utility Functions ---
def format_size(size_bytes: int) -> str:
    if size_bytes <= 0: return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def check_tool(name: str, hint: str = "") -> bool:
    return shutil.which(name) is not None

def get_missing_tools() -> List[Tuple[str, str]]:
    tools = [
        ("mkvmerge", "install MKVToolNix and ensure mkvmerge is on PATH"),
        ("ffmpeg", "install ffmpeg and ensure it's on PATH"),
    ]
    return [(name, hint) for name, hint in tools if not check_tool(name, hint)]

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

def get_valid_cpu_pool() -> List[int]:
    total_threads = os.cpu_count() or 1
    return [i for i in range(total_threads) if i not in BLOCKED_CPUS]

def calculate_dynamic_chunks(num_jobs: int) -> List[List[int]]:
    if num_jobs < 1: return []
    pool = get_valid_cpu_pool()
    total_valid = len(pool)
    base = total_valid // num_jobs
    rem = total_valid % num_jobs
    chunks = []
    start = 0
    for i in range(num_jobs):
        take = base + (1 if i < rem else 0)
        chunk = pool[start : start + take]
        start += take
        if chunk:
            chunks.append(chunk)
    return chunks

def calculate_optimal_workers(chunk_size: int, preset_workers: Any) -> Tuple[int, int]:
    if isinstance(preset_workers, int) and preset_workers > 0:
        return (preset_workers, max(1, chunk_size // preset_workers))
    if chunk_size >= 32:
        return 8, 4  
    elif chunk_size >= 16:
        return 8, chunk_size // 8
    elif chunk_size >= 8:
        return 4, chunk_size // 4
    else:
        return 1, chunk_size

def _set_process_affinity(pid: int, cpus: List[int]) -> None:
    if not IS_WINDOWS or not cpus: return
    try:
        if psutil:
            p = psutil.Process(pid)
            p.cpu_affinity(cpus)
            for child in p.children(recursive=True):
                try: child.cpu_affinity(cpus)
                except: pass
    except Exception: pass

def _safe_kill(proc: subprocess.Popen):
    try: proc.kill()
    except: pass

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    try: func(path)
    except: pass

def _suspend_tree(root_pid: int) -> bool:
    if not psutil: return False
    try:
        root = psutil.Process(root_pid)
        for p in reversed([root] + root.children(recursive=True)):
            try: p.suspend()
            except: pass
        return True
    except: return False

def _resume_tree(root_pid: int) -> bool:
    if not psutil: return False
    try:
        root = psutil.Process(root_pid)
        for p in [root] + root.children(recursive=True):
            try: p.resume()
            except: pass
        return True
    except: return False

# --- Classes ---

class SystemMonitor(QThread):
    stats_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.psutil = psutil
        
    def run(self):
        while self.running:
            stats = {}
            if self.psutil:
                try:
                    stats['cpu_percent'] = self.psutil.cpu_percent(interval=None)
                    mem = self.psutil.virtual_memory()
                    stats['mem_percent'] = mem.percent
                    stats['mem_used'] = mem.used
                    stats['mem_total'] = mem.total
                except: pass
            self.stats_updated.emit(stats)
            time.sleep(2)
    
    def stop(self):
        self.running = False

class ProbeWorker(QThread):
    file_probed = Signal(object, object, object) # path, data, error

    def __init__(self, files: List[Path], parent=None):
        super().__init__(parent)
        self.files = files

    def run(self):
        for f in self.files:
            if not f.is_file(): continue
            abs_path = f.resolve()
            data = None
            error = None
            try:
                si = subprocess.STARTUPINFO() if IS_WINDOWS else None
                if IS_WINDOWS: si.dwFlags |= subprocess.STARTF_USESHOWWINDOW; si.wShowWindow = subprocess.SW_HIDE

                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=nb_frames,duration,r_frame_rate",
                       "-show_entries", "format=duration",
                       "-of", "json", str(abs_path)]

                raw = subprocess.check_output(cmd, startupinfo=si, text=True, timeout=5)
                data = json.loads(raw)
            except Exception as e:
                error = str(e)

            self.file_probed.emit(abs_path, data, error)

class Runner(QObject):
    job_updated = Signal(int)
    job_finished = Signal(int)
    job_added = Signal(object)
    total_fps_changed = Signal(float)
    notify = Signal(str)
    all_jobs_completed = Signal()
    shutdown_complete = Signal()
    
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        
        self.out_dir = Path(config["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(config["temp_dir"])
        os.makedirs(self.temp_dir, exist_ok=True)
        
        svt_path = Path(config["svt_path"])
        if not svt_path.exists():
             if check_tool("SvtAv1EncApp", ""):
                 svt_path = Path(shutil.which("SvtAv1EncApp")) # type: ignore
        
        self.proc_env = os.environ.copy()
        if svt_path.parent.exists():
            self.proc_env["PATH"] = str(svt_path.parent) + os.pathsep + self.proc_env.get("PATH", "")
        
        self.chunk_method = "ffms2"
        self.jobs: List[Job] = []
        self.queue: List[int] = []
        self.running: Dict[int, Job] = {}
        self.run_lock = threading.Lock()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / GUI_REFRESH_HZ))
        
        self._closing = False
        self._next_job_idx = 0
        self._probe_workers = []

    def update_config(self, config: Dict[str, Any]):
        self.config = config
        self.out_dir = Path(config["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(config["temp_dir"])
        os.makedirs(self.temp_dir, exist_ok=True)

    def add_files(self, files: List[Path], preset_name: str = "High Quality"):
        worker = ProbeWorker(files)
        worker.file_probed.connect(lambda p, d, e: self._on_file_probed(p, d, e, preset_name))
        worker.finished.connect(lambda: self._on_probe_finished(worker))
        self._probe_workers.append(worker)
        worker.start()

    def _on_file_probed(self, abs_path, data, error, preset_name):
        base = abs_path.stem
        t_frames = 0
        debug_log = [f"--- INIT DEBUG: {abs_path.name} ---"]

        if error:
            debug_log.append(f"Probe Failed: {error}")
        elif data:
            stream = data.get("streams", [])[0] if data.get("streams") else {}
            if "nb_frames" in stream and stream["nb_frames"] != "N/A":
                try: t_frames = int(stream["nb_frames"])
                except: pass

        j = Job(
            idx=self._next_job_idx,
            infile=abs_path,
            out_mkv=self.out_dir / f"{base}-svt_av1.mkv",
            tempdir=self.temp_dir / base,
            
            # --- LOG PATHS ---
            term_log=LOG_DIR / f"{base}.term.log",
            mux_log=LOG_DIR / f"{base}.mux.log",
            vmaf_log=LOG_DIR / f"{base}-vmaf.json",
            
            preset_name=preset_name,
            total_frames=t_frames,
            max_retries=self.config.get("max_retries", 2),
        )

        j.chunk_count_cache = 0
        j.log_file_handle = None

        try: j.tempdir.mkdir(parents=True, exist_ok=True)
        except: pass

        try:
            j.log_file_handle = open(j.term_log, "w", encoding="utf-8")
            j.log_file_handle.write("\n".join(debug_log) + "\n\n")
        except: pass

        self.jobs.append(j)
        self.queue.append(self._next_job_idx)
        self._next_job_idx += 1
        self.job_added.emit(j)
        self._start_next_if_possible()

    def _on_probe_finished(self, worker):
        if worker in self._probe_workers:
            self._probe_workers.remove(worker)
        try: worker.deleteLater()
        except: pass

    def remove_job(self, job_idx: int):
        with self.run_lock:
            job = self.jobs[job_idx]
            if job.status in [JobStatus.RUNNING, JobStatus.MUXING, JobStatus.VMAF, JobStatus.PAUSED]:
                return False
            if job_idx in self.queue: self.queue.remove(job_idx)
            job.status = JobStatus.CANCELLED
            if hasattr(job, 'log_file_handle') and job.log_file_handle:
                try: job.log_file_handle.close()
                except: pass
            self.job_updated.emit(job_idx)
            return True

    def retry_job(self, job_idx: int):
        job = self.jobs[job_idx]
        if job.status != JobStatus.FAILED: return False
        job.status = JobStatus.QUEUED
        job.pct = 0.0
        job.vmaf_score = 0.0
        job.vmaf_1_percent = 0.0
        job.vmaf_01_percent = 0.0
        job.fps_hist.clear()
        job.error_message = ""
        job.proc = None
        job.started_ts = None
        job.completed_ts = None
        if job_idx not in self.queue: self.queue.append(job_idx)
        try:
            if hasattr(job, 'log_file_handle') and job.log_file_handle:
                try: job.log_file_handle.close()
                except: pass
            job.log_file_handle = open(job.term_log, "w", encoding="utf-8")
        except: pass
        self.job_updated.emit(job_idx)
        return True

    def request_stop_all(self):
        if self._closing: return
        self._closing = True
        self.timer.stop()
        self.queue.clear()
        threading.Thread(target=self._stop_all_processes_blocking, daemon=True).start()

    def _stop_all_processes_blocking(self):
        with self.run_lock:
            procs = []
            for job in self.running.values():
                if job.proc and job.proc.poll() is None: procs.append(job.proc)
                if job.mux_proc and job.mux_proc.poll() is None: procs.append(job.mux_proc)
                if job.vmaf_proc and job.vmaf_proc.poll() is None: procs.append(job.vmaf_proc)
        for p in procs: _safe_kill(p)
        self.shutdown_complete.emit()

    def toggle_pause(self, job_idx: int):
        with self.run_lock:
            job = self.jobs[job_idx]
            if job.status == JobStatus.RUNNING:
                if job.proc and _suspend_tree(job.proc.pid):
                    job.status = JobStatus.PAUSED
                    self.notify.emit(f"Paused {job.infile.name}")
                    self.job_updated.emit(job_idx)
                    self._rebalance_affinity() 
            elif job.status == JobStatus.PAUSED:
                if job.proc and _resume_tree(job.proc.pid):
                    job.status = JobStatus.RUNNING
                    self.notify.emit(f"Resumed {job.infile.name}")
                    self.job_updated.emit(job_idx)
                    self._rebalance_affinity()

    def _build_av1an_args(self, job: Job, assigned_chunk: List[int]) -> List[str]:
        preset = DEFAULT_PRESETS.get(job.preset_name, DEFAULT_PRESETS["High Quality"])
        chunk_size = len(assigned_chunk)
        workers, threads = calculate_optimal_workers(chunk_size, preset.get("workers", "auto"))
        svt_opts = job.custom_svt_opts or preset["svt_opts"]
        svt_cli = _strip_lp(svt_opts)
        svt_cli = shlex.join(shlex.split(svt_cli) + ["--lp", str(threads)])
        def to_posix(p):
            return str(p).replace("\\", "/")

        log_file = LOG_DIR / f"av1an.log.{datetime.date.today()}"

        args = [
            str(DEFAULT_AV1AN_PATH),
            "--log-file", to_posix(log_file),
            "-i", to_posix(job.infile),          
            "--temp", to_posix(job.tempdir),
            "-o", to_posix(job.out_mkv),         
            "-e", "svt-av1",
            "-v", svt_cli,                  
            "-w", str(workers),             
            "-m", "ffms2",
            "--pix-format", "yuv420p10le",  
            "--concat", "mkvmerge"
        ]
        if self.config.get("resume", True): args.append("-r")
        if self.config.get("keep", True): args.append("--keep")
        return args

    def _rebalance_affinity(self):
        if not self.running: return
        active_jobs = [j for j in self.running.values() if j.status == JobStatus.RUNNING and j.proc]
        if not active_jobs: return
        chunks = calculate_dynamic_chunks(len(active_jobs))
        for i, job in enumerate(active_jobs):
            if i < len(chunks):
                new_cpus = chunks[i]
                job.cpus = new_cpus
                _set_process_affinity(job.proc.pid, new_cpus)

    def _start_next_if_possible(self):
        if self._closing: return
        with self.run_lock:
            active_count = len(self.queue) + len(self.running)
            dynamic_limit = max(1, min(active_count, MAX_JOBS_CAP))
            started_new = False
            while len(self.running) < dynamic_limit and self.queue:
                idx = self.queue.pop(0)
                job = self.jobs[idx]
                temp_chunks = calculate_dynamic_chunks(len(self.running) + 1)
                my_chunk = temp_chunks[-1]
                args = self._build_av1an_args(job, my_chunk)
                
                job.fps_hist.clear()
                job.ema_fps = None
                job.pct = 0.0
                job.chunk_count_cache = 0
                job.status = JobStatus.RUNNING
                job.started_ts = time.time()
                job.completed_ts = None
                job.vmaf_score = 0.0
                
                lock_file = job.tempdir / "lock"
                try:
                    if lock_file.exists(): lock_file.unlink()
                except: pass
                
                self.notify.emit(f"Starting {job.infile.name}")
                try:
                    startupinfo = None
                    if IS_WINDOWS:
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = subprocess.SW_HIDE

                    job.proc = subprocess.Popen(
                        args,
                        stdout=job.log_file_handle,
                        stderr=job.log_file_handle,
                        text=True,
                        startupinfo=startupinfo,
                        env=self.proc_env,
                        cwd=LOG_DIR,
                    )
                    self.running[idx] = job
                    started_new = True
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    self.job_updated.emit(idx)
                    self.job_finished.emit(idx)
            
            if started_new:
                self._rebalance_affinity()

    def _start_muxing(self, job: Job):
        job.status = JobStatus.MUXING
        self.job_updated.emit(job.idx)
        self._rebalance_affinity()
        
        enc_source = job.out_mkv
        if not enc_source.exists():
             if (job.tempdir / "encode").exists():
                 for f in (job.tempdir / "encode").glob("*.mkv"):
                     enc_source = f
                     break
        
        if not enc_source or not enc_source.exists():
            job.status = JobStatus.FAILED
            job.error_message = "No encoded video found"
            return
            
        remux_tmp = job.out_mkv.with_suffix(".remux.mkv")
        
        cmd = [
            "mkvmerge", "-o", str(remux_tmp), "-D", str(job.infile),
            "--no-audio", "--no-subtitles", "--no-chapters", str(enc_source)
        ]
        
        try:
             job.log_file_handle.write(f"\n=== MUXING ===\n{shlex.join(cmd)}\n")
             job.log_file_handle.flush()
        except: pass

        try:
            startupinfo = None
            if IS_WINDOWS:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            job.mux_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, startupinfo=startupinfo, encoding='utf-8', errors='replace'
            )
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = f"Mux error: {e}"

    def _start_vmaf(self, job: Job):
        job.status = JobStatus.VMAF
        job.fps_hist.clear()
        job.started_ts = time.time()
        self.job_updated.emit(job.idx)
        self._rebalance_affinity()
        
        vmaf_json = job.vmaf_log
        if vmaf_json.exists(): vmaf_json.unlink()
        
        try:
            if not job.log_file_handle or job.log_file_handle.closed:
                job.log_file_handle = open(job.term_log, "a", encoding="utf-8")
                job.log_read_offset = job.log_file_handle.tell()
        except Exception as e:
            print(f"Failed to reopen log for VMAF: {e}")
            self._finalize(job)
            return

        if psutil:
            phys_cores = psutil.cpu_count(logical=False) or 16
        else:
            phys_cores = (os.cpu_count() or 4) // 2
        threads = phys_cores

        # --- FIX: ESCAPE SPECIAL CHARS FOR FFMPEG FILTER ---
        # 1. Escape backslashes first
        json_path_str = str(job.vmaf_log.resolve()).replace("\\", "\\\\")
        # 2. Escape ':' because it separates filter options
        json_path_str = json_path_str.replace(":", "\\:") 
        # 3. Escape '[' and ']' because they are stream specifiers in filter graphs
        json_path_str = json_path_str.replace("[", "\\[").replace("]", "\\]")
        # 4. Escape spaces
        json_path_str = json_path_str.replace(" ", "\\ ")
        # 5. Escape commas (option separators) and single quotes
        json_path_str = json_path_str.replace(",", "\\,").replace("'", "\\'")
        # 6. Escape semicolons
        json_path_str = json_path_str.replace(";", "\\;")

        cmd = [
            "ffmpeg", "-stats", 
            "-threads", str(threads), "-i", str(job.out_mkv), 
            "-threads", str(threads), "-i", str(job.infile),
            "-filter_complex_threads", str(threads),
            # Path is now heavily sanitized
            "-filter_complex", f"libvmaf=log_path={json_path_str}:log_fmt=json:n_threads={threads}:n_subsample=4",
            "-f", "null", "-"
        ]

        try:
             job.log_file_handle.write(f"\n=== VMAF CALCULATION (Threads={threads}) ===\n{shlex.join(cmd)}\n")
             job.log_file_handle.flush()
        except: pass

        try:
            startupinfo = None
            if IS_WINDOWS:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            job.vmaf_proc = subprocess.Popen(
                cmd, stdout=job.log_file_handle, stderr=job.log_file_handle,
                text=True, startupinfo=startupinfo, env=self.proc_env
            )
        except Exception as e:
            print(f"VMAF Start failed: {e}")
            self._finalize(job)

    def _tick(self):
        self._start_next_if_possible()
        finished = []
        rebalance = False
        
        with self.run_lock:
            for idx, job in list(self.running.items()):
                
                if job.status == JobStatus.RUNNING:
                    self._update_progress_from_disk(job)
                    if job.proc:
                        rc = job.proc.poll()
                        if rc is not None:
                            try: job.log_file_handle.close()
                            except: pass
                            if rc == 0:
                                self._start_muxing(job)
                                rebalance = True
                            else:
                                job.status = JobStatus.FAILED
                                job.error_message = f"Encode RC={rc}"
                                finished.append(idx)
                                rebalance = True
                            
                elif job.status == JobStatus.MUXING and job.mux_proc:
                    rc = job.mux_proc.poll()
                    if rc is not None:
                        if rc == 0:
                            self._finalize_mux(job) 
                            if self.config.get("calc_vmaf", True):
                                self._start_vmaf(job)
                            else:
                                self._finalize_complete(job)
                                finished.append(idx)
                        else:
                            self._fail_job(job, f"Mux RC={rc}")
                            finished.append(idx)

                elif job.status == JobStatus.VMAF:
                    self._update_progress_from_disk(job) 
                    if job.vmaf_proc:
                        rc = job.vmaf_proc.poll()
                        if rc is not None:
                            try: job.log_file_handle.close()
                            except: pass
                            
                            if rc == 0:
                                self._read_vmaf_score(job)
                            self._finalize_complete(job)
                            finished.append(idx)
                
                self.job_updated.emit(idx)
                        
            if finished:
                for idx in finished:
                    if idx in self.running: del self.running[idx]
                    self.job_finished.emit(idx)
                rebalance = True
                
            if rebalance:
                self._rebalance_affinity()
                
        self._emit_total_fps()
        if not self.running and not self.queue:
             if any(j.status == JobStatus.COMPLETED for j in self.jobs):
                 self.all_jobs_completed.emit()

    def _update_progress_from_disk(self, job: Job):
        if job.term_log.exists():
            try:
                with open(job.term_log, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(job.log_read_offset)
                    new_data = f.read()
                    if new_data:
                        job.log_read_offset = f.tell()
                        regex = AV1AN_PROGRESS_RE if job.status == JobStatus.RUNNING else FFMPEG_PROGRESS_RE
                        for line in new_data.splitlines():
                            m = regex.search(line)
                            if m:
                                try:
                                    if job.status == JobStatus.RUNNING:
                                        pct_val = float(m.group(1))
                                        frames_done = int(m.group(2))
                                        frames_total = int(m.group(3))
                                        fps_val = float(m.group(6))
                                        job.pct = pct_val
                                        job.frames_done = frames_done
                                        job.total_frames = frames_total
                                        job.ema_fps = fps_val
                                        job.fps_hist.append(fps_val)
                                    elif job.status == JobStatus.VMAF:
                                        frames_done = int(m.group(1))
                                        fps_val = float(m.group(2))
                                        job.frames_done = frames_done
                                        job.ema_fps = fps_val
                                        job.fps_hist.append(fps_val)
                                        if job.total_frames > 0:
                                            job.pct = (frames_done / job.total_frames) * 100.0
                                except ValueError:
                                    pass
            except Exception:
                pass
        if job.status == JobStatus.RUNNING and job.chunk_count_cache == 0:
            chunks_file = job.tempdir / "chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r') as f:
                        data = json.load(f)
                        job.chunk_count_cache = len(data)
                except: pass

    def _emit_total_fps(self):
        total = sum(j.fps_hist[-1] for j in self.jobs if j.status == JobStatus.RUNNING and j.fps_hist)
        self.total_fps_changed.emit(total)

    def _finalize_mux(self, job: Job):
        remux_tmp = job.out_mkv.with_suffix(".remux.mkv")
        if remux_tmp.exists():
            try:
                if job.out_mkv.exists(): job.out_mkv.unlink()
                os.replace(remux_tmp, job.out_mkv)
                job.encoded_size = job.out_mkv.stat().st_size
            except Exception as e:
                self._fail_job(job, f"Finalize error: {e}")

    def _read_vmaf_score(self, job: Job):
        print(f"[DEBUG] Attempting to read VMAF from: {job.vmaf_log}")
        for attempt in range(3):
            if job.vmaf_log.exists():
                size = job.vmaf_log.stat().st_size
                print(f"[DEBUG] File exists. Size: {size} bytes (Attempt {attempt+1})")
                if size > 0:
                    break
            else:
                print(f"[DEBUG] File does not exist yet (Attempt {attempt+1})")
            time.sleep(1.0)
            
        scores = []
        if job.vmaf_log.exists():
            try:
                with open(job.vmaf_log, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    print(f"[DEBUG] Read {len(content)} chars from file.")
                    matches = re.findall(r'"vmaf":\s*([\d\.]+)', content)
                    if matches:
                        scores = [float(m) for m in matches]
                        print(f"[DEBUG] Regex found {len(scores)} frame scores.")
                    else:
                        print("[DEBUG] Regex found 0 matches.")
            except Exception as e:
                print(f"[DEBUG] Error reading text content: {e}")

        if scores:
            # Filter out erroneous 0.0 scores
            valid_scores = [s for s in scores if s > 0.01]
            print(f"[DEBUG] Filtered {len(scores) - len(valid_scores)} zero-scores.")
            
            if valid_scores:
                valid_scores.sort()
                count = len(valid_scores)
                
                if job.vmaf_score == 0.0:
                    job.vmaf_score = sum(valid_scores) / count
                    print(f"[DEBUG] Calculated Mean from frames: {job.vmaf_score}")
                
                idx_1 = max(0, int(count * 0.01))
                idx_01 = max(0, int(count * 0.001))
                job.vmaf_1_percent = valid_scores[idx_1]
                job.vmaf_01_percent = valid_scores[idx_01]
                print(f"[DEBUG] Calculated Lows: 1%={job.vmaf_1_percent}, 0.1%={job.vmaf_01_percent}")
            else:
                print("[DEBUG] No valid scores remaining after filtering.")
        else:
            print("[DEBUG] No scores extracted.")

        if job.vmaf_score == 0.0 and job.term_log.exists():
            print(f"[DEBUG] Checking term log for fallback score: {job.term_log}")
            try:
                with open(job.term_log, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    m = re.search(r"VMAF score:\s+(\d+(?:\.\d+)?)", content)
                    if m:
                        job.vmaf_score = float(m.group(1))
                        print(f"[DEBUG] Found fallback score in log: {job.vmaf_score}")
                    else:
                        print("[DEBUG] No score found in term log.")
            except Exception as e:
                print(f"[DEBUG] Error reading term log: {e}")

    def _finalize_complete(self, job: Job):
        job.status = JobStatus.COMPLETED
        job.completed_ts = time.time()
        if self.config.get("auto_cleanup", True):
            shutil.rmtree(job.tempdir, onerror=on_rm_error)

    def _finalize(self, job: Job):
        self._finalize_mux(job)
        self._finalize_complete(job)

    def _fail_job(self, job: Job, msg: str):
        if hasattr(job, 'log_file_handle') and job.log_file_handle:
             try: job.log_file_handle.close()
             except: pass
        if job.retry_count < job.max_retries:
            job.retry_count += 1
            self.notify.emit(f"Retrying {job.infile.name} ({job.retry_count})")
            self.queue.append(job.idx)
            job.status = JobStatus.QUEUED
        else:
            job.status = JobStatus.FAILED
            job.error_message = msg
        job.completed_ts = time.time()

    def _write_log(self, path: Path, text: str):
        pass

    def _find_encoded_video(self, job: Job) -> Optional[Path]:
        return job.out_mkv
