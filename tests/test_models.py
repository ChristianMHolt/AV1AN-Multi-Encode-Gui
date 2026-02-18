import sys
import os
from pathlib import Path
import time
from collections import deque
import pytest

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models import Job, JobStatus

def create_job(**kwargs):
    defaults = {
        'idx': 0,
        'infile': Path('dummy_input.mkv'),
        'out_mkv': Path('dummy_output.mkv'),
        'tempdir': Path('dummy_temp'),
        'term_log': Path('dummy_term.log'),
        'mux_log': Path('dummy_mux.log'),
        'vmaf_log': Path('dummy_vmaf.log'),
        'status': JobStatus.QUEUED
    }
    defaults.update(kwargs)
    return Job(**defaults)

def test_eta_seconds_not_running():
    job = create_job(status=JobStatus.QUEUED)
    assert job.eta_seconds is None

    job.status = JobStatus.PAUSED
    assert job.eta_seconds is None

    job.status = JobStatus.COMPLETED
    assert job.eta_seconds is None

    job.status = JobStatus.FAILED
    assert job.eta_seconds is None

    job.status = JobStatus.CANCELLED
    assert job.eta_seconds is None

def test_eta_seconds_running_frames():
    job = create_job(status=JobStatus.RUNNING, total_frames=1000)

    # Simulate some progress
    job.frames_done = 100
    # Simulate avg_fps > 0.1
    # avg_fps property uses fps_hist
    job.fps_hist.append(10.0)

    # Expected: (1000 - 100) / 10.0 = 90.0 seconds
    assert job.eta_seconds == 90.0

    # Test with VMAF status
    job.status = JobStatus.VMAF
    assert job.eta_seconds == 90.0

def test_eta_seconds_running_frames_low_fps():
    # If avg_fps <= 0.1, it should fall through to pct check
    job = create_job(status=JobStatus.RUNNING, total_frames=1000)
    job.frames_done = 100
    job.fps_hist.append(0.1)

    # Fallback to pct logic if pct > 0 and avg_fps > 0

    job.pct = 50.0
    job.started_ts = time.time() - 100 # 100 seconds elapsed

    # total_estimated = 100 / (50/100) = 200
    # remaining = 200 - 100 = 100

    assert job.eta_seconds is not None
    assert abs(job.eta_seconds - 100.0) < 1.0

def test_eta_seconds_running_pct():
    # Scenario where total_frames is 0 (e.g. unknown)
    job = create_job(status=JobStatus.RUNNING, total_frames=0)

    job.pct = 25.0
    job.started_ts = time.time() - 60 # 60 seconds elapsed
    job.fps_hist.append(1.0) # avg_fps > 0

    # total_estimated = 60 / (25/100) = 240
    # remaining = 240 - 60 = 180

    assert abs(job.eta_seconds - 180.0) < 1.0

def test_eta_seconds_remaining_negative():
    job = create_job(status=JobStatus.RUNNING, total_frames=100)
    job.frames_done = 150 # More than total
    job.fps_hist.append(10.0)

    # remaining = 100 - 150 = -50 -> 0
    # 0 / 10.0 = 0.0

    assert job.eta_seconds == 0.0

def test_eta_seconds_none():
    job = create_job(status=JobStatus.RUNNING)

    # total_frames=0, pct=0
    assert job.eta_seconds is None

    # avg_fps=0
    job.total_frames = 100
    job.frames_done = 10
    # fps_hist empty -> avg_fps = 0
    assert job.eta_seconds is None
