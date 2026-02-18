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

        # Set started_ts to produce 50 fps global average
        # 500 frames done / 10s = 50 fps
        import time
        j.started_ts = time.time() - 10.0

        # remaining = 500, fps = 50 -> 10s
        self.assertAlmostEqual(j.eta_seconds, 10.0, delta=0.5)

    def test_compression_ratio(self):
        j = Job(0, Path("in.mkv"), Path("out.mkv"), Path("tmp"), Path("t.log"), Path("m.log"), Path("v.json"))
        j.original_size = 100
        j.encoded_size = 50
        self.assertAlmostEqual(j.compression_ratio, 50.0)

    def test_global_fps(self):
        j = Job(0, Path("in.mkv"), Path("out.mkv"), Path("tmp"), Path("t.log"), Path("m.log"), Path("v.json"))
        j.status = JobStatus.RUNNING

        # Mock time
        import time
        now = time.time()
        j.started_ts = now - 100.0 # 100 seconds elapsed
        j.frames_done = 1000 # 1000 frames done

        # global_fps should be 10.0
        self.assertAlmostEqual(j.global_fps, 10.0, delta=0.5)

        # eta_seconds should be remaining / global_fps
        j.total_frames = 2000
        # remaining = 1000
        # 1000 / 10 = 100 seconds
        self.assertAlmostEqual(j.eta_seconds, 100.0, delta=5.0)

        # Test 0 elapsed
        j.started_ts = None
        self.assertEqual(j.global_fps, 0.0)

if __name__ == '__main__':
    unittest.main()
