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

# Now import worker
from worker import Runner, JobStatus

class TestRunner(unittest.TestCase):
    def test_runner_init(self):
        config = {
            "output_dir": "out",
            "temp_dir": "tmp",
            "svt_path": "svt", # fake
            "chunk_method": "ffms2"
        }
        with patch("shutil.which", return_value="/bin/ls"):
            with patch("worker.check_tool", return_value=True):
                 r = Runner(config)
                 self.assertEqual(str(r.out_dir), "out")

    def test_eta_includes_queued_jobs(self):
        """Verify that ETA calculation includes queued jobs if they have valid frame counts."""
        config = {
            "output_dir": "out",
            "temp_dir": "tmp",
            "svt_path": "svt",
        }
        with patch("shutil.which", return_value="/bin/ls"), patch("worker.check_tool", return_value=True):
            runner = Runner(config)

        # Job 1: Running, 1000 frames total, 500 done, 50 fps
        job1 = MagicMock()
        job1.status = JobStatus.RUNNING
        job1.total_frames = 1000
        job1.frames_done = 500
        job1.fps_hist = [50.0]
        job1.avg_fps = 50.0
        job1.global_fps = 50.0
        job1.infile.name = "video1.mkv"

        # Job 2: Queued, 1000 frames total, 0 done
        job2 = MagicMock()
        job2.status = JobStatus.QUEUED
        job2.total_frames = 1000
        job2.frames_done = 0
        job2.fps_hist = []
        job2.infile.name = "video2.mkv"

        runner.jobs = [job1, job2]

        # Trigger calculation
        runner._emit_total_fps()

        # Expected calculation:
        # Remaining: 500 (run) + 1000 (queue) = 1500
        # FPS: 50
        # ETA = 1500 / 50 = 30s
        runner.total_eta_changed.emit.assert_called_with("ETA: 30s")

    def test_probe_fallback(self):
        """Verify fallback calculation when nb_frames is missing."""
        config = {
            "output_dir": "out",
            "temp_dir": "tmp",
            "svt_path": "svt",
        }
        with patch("shutil.which", return_value="/bin/ls"), patch("worker.check_tool", return_value=True):
            runner = Runner(config)

        # nb_frames missing, but duration and r_frame_rate present
        data = {
            "streams": [{
                "duration": "100.0",
                "r_frame_rate": "30/1"
            }],
            "format": {
                "duration": "100.0"
            }
        }
        path = MagicMock()
        path.stem = "test_video_fallback"
        path.name = "test_video_fallback.mkv"
        path.resolve.return_value = path

        runner._on_file_probed(path, data, None, "High Quality")

        job = runner.jobs[-1]
        # Should be calculated: 100 * 30 = 3000
        self.assertEqual(job.total_frames, 3000)

if __name__ == '__main__':
    unittest.main()
