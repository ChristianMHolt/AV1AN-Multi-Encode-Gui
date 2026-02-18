import sys
import unittest
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models import Job, JobStatus

class TestJob(unittest.TestCase):
    def test_job_defaults(self):
        j = Job(0, Path("in.mkv"), Path("out.mkv"), Path("tmp"), Path("t.log"), Path("m.log"), Path("v.json"))
        self.assertEqual(j.status_text, "")
        self.assertIsNone(j.eta_seconds)

    def test_eta_calculation(self):
        j = Job(0, Path("in.mkv"), Path("out.mkv"), Path("tmp"), Path("t.log"), Path("m.log"), Path("v.json"))
        j.status = JobStatus.RUNNING
        j.total_frames = 1000
        j.frames_done = 500
        # Mock fps_hist
        j.fps_hist.append(50.0)

        # remaining = 500, fps = 50 -> 10s
        self.assertAlmostEqual(j.eta_seconds, 10.0)

    def test_compression_ratio(self):
        j = Job(0, Path("in.mkv"), Path("out.mkv"), Path("tmp"), Path("t.log"), Path("m.log"), Path("v.json"))
        j.original_size = 100
        j.encoded_size = 50
        self.assertAlmostEqual(j.compression_ratio, 50.0)

if __name__ == '__main__':
    unittest.main()
