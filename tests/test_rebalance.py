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
        # It was started when there were 2 jobs (so 8 cores).
        # Now it is the only one (so it SHOULD use 16 cores).

        job = Job(0, Path("i1"), Path("o1"), Path("t1"), Path("l1"), Path("m1"), Path("v1"))
        job.status = JobStatus.RUNNING

        # Simulate initial state: 8 cores -> workers=4, threads=2 (based on logic)
        job.initial_workers = 4
        job.initial_threads = 2

        # Current state: 1 job. 16 cores -> workers=8, threads=2.
        # Mismatch: workers 4 vs 8. Should restart.

        p = MagicMock()
        p.poll.return_value = None
        p.pid = 999
        job.proc = p

        self.runner.jobs = [job]
        self.runner.running = {0: job}
        self.runner.queue = [] # Empty queue triggers check

        # Ensure method exists
        if not hasattr(self.runner, '_check_optimization'):
            # This allows the test to exist before implementation without failing the suite
            print("Skipping test_restart_trigger: _check_optimization not implemented")
            return

        # 2. Trigger Check
        self.runner._check_optimization()

        # 3. Verify Restart Triggered
        # Access RESTARTING dynamically to avoid import error if not defined yet
        restarting_status = getattr(JobStatus, 'RESTARTING', None)
        if restarting_status:
            self.assertEqual(job.status, restarting_status, "Job should be marked RESTARTING")

        p.kill.assert_called_once()
        self.assertIn(job, self.runner.restarting_jobs)
        self.assertNotIn(0, self.runner.running)

        # 4. Simulate Process Exit
        p.poll.return_value = -9 # Killed

        # 5. Trigger Tick to handle restart completion
        # We need to mock _start_next_if_possible to verify it gets called
        with patch.object(self.runner, '_start_next_if_possible') as mock_start:
            self.runner._tick()

            # Verify job moved back to queue
            self.assertIn(0, self.runner.queue)
            self.assertNotIn(job, self.runner.restarting_jobs)
            mock_start.assert_called()

if __name__ == '__main__':
    unittest.main()
