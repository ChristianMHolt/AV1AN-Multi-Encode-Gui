import sys
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import os
import shutil

# --- MOCKS SETUP ---
mock_qt_core = MagicMock()
class MockQObject:
    def __init__(self, parent=None): pass
class MockQThread:
    def __init__(self, parent=None): pass
    def start(self): pass
    def quit(self): pass
    def wait(self): pass
    def deleteLater(self): pass

class MockQTimer:
    def __init__(self):
        self.timeout = MagicMock()
    def start(self, interval): pass
    def stop(self): pass

def MockSignal(*args):
    m = MagicMock()
    m.connect = MagicMock()
    m.emit = MagicMock()
    return m

mock_qt_core.QObject = MockQObject
mock_qt_core.QThread = MockQThread
mock_qt_core.QTimer = MockQTimer
mock_qt_core.Signal = MockSignal

sys.modules['ui.qt_core'] = mock_qt_core
sys.modules['src.ui.qt_core'] = mock_qt_core
sys.modules['psutil'] = MagicMock()

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_PATH))

from worker import Runner
from models import Job, JobStatus

class TestRebalance(unittest.TestCase):
    def setUp(self):
        self.config = {
            "output_dir": "out",
            "temp_dir": "temp",
            "svt_path": "svt",
            "max_retries": 1
        }
        with patch('pathlib.Path.mkdir'), patch('os.makedirs'), patch('shutil.which'):
            self.runner = Runner(self.config)
        self.runner.jobs = []
        self.runner.queue = []
        self.runner.running = {}

        # Inject lists if they don't exist yet (simulating partial implementation state if run early)
        if not hasattr(self.runner, 'restarting_jobs'):
            self.runner.restarting_jobs = []

    @patch('os.cpu_count', return_value=16)
    @patch('subprocess.Popen')
    def test_restart_trigger(self, mock_popen, mock_cpu):
        # 1. Setup: 1 Running Job that is suboptimal.
        job = Job(0, Path("i1"), Path("o1"), Path("t1"), Path("l1"), Path("m1"), Path("v1"))
        job.status = JobStatus.RUNNING
        job.initial_workers = 4
        job.initial_threads = 2

        p = MagicMock()
        p.poll.return_value = None
        p.pid = 999
        job.proc = p

        self.runner.jobs = [job]
        self.runner.running = {0: job}
        self.runner.queue = [] # Empty queue triggers check

        # 2. Trigger Check
        self.runner._check_optimization()

        # 3. Verify Restart Triggered
        self.assertEqual(job.status, JobStatus.RESTARTING)
        p.kill.assert_called_once()
        self.assertIn(job, self.runner.restarting_jobs)

        # 4. Simulate Process Exit
        p.poll.return_value = -9 # Killed

        # 5. Trigger Tick to handle restart completion
        with patch.object(self.runner, '_start_next_if_possible') as mock_start:
            self.runner._tick()
            self.assertIn(0, self.runner.queue)
            self.assertNotIn(job, self.runner.restarting_jobs)
            mock_start.assert_called()

    @patch('os.cpu_count', return_value=16)
    def test_vmaf_blocks_restart(self, mock_cpu):
        # 1. Setup: 1 Running Job (Suboptimal) + 1 VMAF Job
        job_run = Job(0, Path("i1"), Path("o1"), Path("t1"), Path("l1"), Path("m1"), Path("v1"))
        job_run.status = JobStatus.RUNNING
        job_run.initial_workers = 4 # Should trigger restart if alone
        job_run.initial_threads = 2
        job_run.proc = MagicMock()

        job_vmaf = Job(1, Path("i2"), Path("o2"), Path("t2"), Path("l2"), Path("m2"), Path("v2"))
        job_vmaf.status = JobStatus.VMAF
        job_vmaf.proc = MagicMock() # Mock process

        self.runner.jobs = [job_run, job_vmaf]
        self.runner.running = {0: job_run, 1: job_vmaf}
        self.runner.queue = []

        # 2. Trigger Check
        self.runner._check_optimization()

        # 3. Verify NO Restart
        self.assertEqual(job_run.status, JobStatus.RUNNING)
        job_run.proc.kill.assert_not_called()
        self.assertNotIn(job_run, self.runner.restarting_jobs)

if __name__ == '__main__':
    unittest.main()
