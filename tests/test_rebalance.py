import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock PySide6 and psutil
mock_qt = MagicMock()
class MockQObject:
    def __init__(self, parent=None):
        pass
mock_qt.QObject = MockQObject
mock_qt.Signal = MagicMock()
mock_qt.QThread = MagicMock()
mock_qt.QTimer = MagicMock()

sys.modules["PySide6"] = mock_qt
sys.modules["PySide6.QtCore"] = mock_qt
sys.modules["PyQt6"] = mock_qt
sys.modules["PyQt6.QtCore"] = mock_qt
sys.modules["psutil"] = MagicMock()

# Import project modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from worker import Runner, JobStatus
from models import Job

class TestRebalance(unittest.TestCase):
    def setUp(self):
        self.config = {
            "output_dir": "out",
            "temp_dir": "tmp",
            "svt_path": "svt",
            "max_retries": 2,
            "resume": True,
            "keep": True,
            "auto_cleanup": True,
            "calc_vmaf": False
        }
        with patch("shutil.which", return_value="/bin/ls"), patch("worker.check_tool", return_value=True):
            self.runner = Runner(self.config)

        # Mock valid cpu pool so chunks work
        with patch("worker.get_valid_cpu_pool", return_value=[0, 1, 2, 3]):
             pass

    def test_rebalance_triggers(self):
        """Test that jobs restart when running count drops below initial concurrency."""
        # 1. Setup 2 jobs running with MAX_JOBS_CAP = 2
        # Mock MAX_JOBS_CAP in worker module
        with patch("worker.MAX_JOBS_CAP", 2):
            # Create 2 jobs
            job1 = MagicMock(spec=Job)
            job1.idx = 0
            job1.status = JobStatus.RUNNING
            job1.initial_concurrent_count = 2
            job1.proc = MagicMock()
            job1.proc.poll.return_value = None
            job1.infile = MagicMock()
            job1.infile.name = "job1"
            job1.term_log = MagicMock()
            job1.term_log.exists.return_value = False
            job1.chunk_count_cache = 0
            job1.tempdir = MagicMock()
            job1.fps_hist = [0]

            job2 = MagicMock(spec=Job)
            job2.idx = 1
            job2.status = JobStatus.RUNNING
            job2.initial_concurrent_count = 2
            job2.proc = MagicMock()
            job2.proc.poll.return_value = None
            job2.infile = MagicMock()
            job2.infile.name = "job2"
            job2.term_log = MagicMock()
            job2.term_log.exists.return_value = False
            job2.chunk_count_cache = 0
            job2.tempdir = MagicMock()
            job2.fps_hist = [0]

            self.runner.running = {0: job1, 1: job2}
            self.runner.jobs = [job1, job2]

            # 2. Simulate Job 1 finishing
            del self.runner.running[0]
            job1.status = JobStatus.COMPLETED

            # Queue is empty
            self.runner.queue = []

            # 3. Call _tick -> should trigger optimization
            # Mock _start_next_if_possible to avoid side effects
            with patch.object(self.runner, '_start_next_if_possible'):
                with patch("worker._safe_kill") as mock_kill:
                     self.runner._tick()

                     # Check if job2 is restarting
                     self.assertEqual(job2.status, JobStatus.RESTARTING)
                     mock_kill.assert_called_with(job2.proc)
                     self.assertIn(1, self.runner._restart_pending)

    def test_rebalance_finishes(self):
        """Test that restarting jobs are requeued once stopped."""
        job = MagicMock(spec=Job)
        job.idx = 1
        job.status = JobStatus.RESTARTING
        job.proc = MagicMock()
        job.proc.poll.return_value = 0 # Stopped

        self.runner.running = {1: job}
        self.runner.jobs = [job]
        self.runner._restart_pending = [1]
        self.runner.queue = []

        with patch.object(self.runner, '_start_next_if_possible') as mock_start:
             self.runner._tick()

             # Should move to queue
             self.assertNotIn(1, self.runner.running)
             self.assertEqual(self.runner.queue[0], 1)
             self.assertEqual(job.status, JobStatus.QUEUED)
             self.assertEqual(self.runner._restart_pending, [])
             mock_start.assert_called_once()

    def test_no_rebalance_without_resume(self):
        """Test that rebalance is skipped if resume is disabled."""
        self.runner.config["resume"] = False

        job = MagicMock(spec=Job)
        job.idx = 0
        job.status = JobStatus.RUNNING
        job.initial_concurrent_count = 2
        job.proc = MagicMock()
        job.proc.poll.return_value = None
        job.term_log = MagicMock()
        job.term_log.exists.return_value = False
        job.chunk_count_cache = 0
        job.tempdir = MagicMock()
        job.fps_hist = [0]

        self.runner.running = {0: job}
        self.runner.queue = []

        with patch("worker.MAX_JOBS_CAP", 2):
            self.runner._tick()
            self.assertNotEqual(job.status, JobStatus.RESTARTING)

if __name__ == '__main__':
    unittest.main()
