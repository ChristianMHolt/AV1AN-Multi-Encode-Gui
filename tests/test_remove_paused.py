import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock PySide6
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

# Mock psutil
sys.modules["psutil"] = MagicMock()

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from worker import Runner, JobStatus

class TestRemovePaused(unittest.TestCase):
    def test_remove_paused(self):
        config = {
            "output_dir": "out",
            "temp_dir": "tmp",
            "svt_path": "svt",
        }
        with patch("shutil.which", return_value="/bin/ls"), patch("worker.check_tool", return_value=True):
            runner = Runner(config)

        # Create a mock paused job
        job = MagicMock()
        job.idx = 0
        job.status = JobStatus.PAUSED
        job.proc = MagicMock() # Mock process
        job.log_file_handle = MagicMock()
        job.mux_proc = None
        job.vmaf_proc = None

        runner.jobs = [job]
        runner.running = {0: job}

        # New behavior: should return True and kill process
        result = runner.remove_job(0)

        self.assertTrue(result, "Should return True for paused jobs")
        self.assertEqual(job.status, JobStatus.CANCELLED)
        self.assertTrue(job.proc.kill.called, "Should kill the process")
        self.assertNotIn(0, runner.running, "Should remove from running dict")
        self.assertTrue(job.log_file_handle.close.called, "Should close log file")

if __name__ == '__main__':
    unittest.main()
