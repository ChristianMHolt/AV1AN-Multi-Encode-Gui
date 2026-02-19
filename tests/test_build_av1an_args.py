import sys
import unittest
import shlex
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup mocks
mock_qt = MagicMock()
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

mock_qt.QObject = MockQObject
mock_qt.QThread = MockQThread
mock_qt.QTimer = MockQTimer
mock_qt.Signal = MagicMock()

sys.modules["ui.qt_core"] = mock_qt
sys.modules["src.ui.qt_core"] = mock_qt
sys.modules["psutil"] = MagicMock()

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_PATH))

from worker import Runner, Job, JobStatus, _strip_lp
from config import IS_WINDOWS

class TestBuildAv1anArgs(unittest.TestCase):
    def setUp(self):
        self.config = {
            "output_dir": "out",
            "temp_dir": "temp",
            "svt_path": "svt",
            "max_retries": 1
        }
        with patch('pathlib.Path.mkdir'), patch('os.makedirs'), patch('shutil.which'):
            self.runner = Runner(self.config)

    def test_strip_lp_simple(self):
        s = "--preset 6 --lp 4 --crf 20"
        res = _strip_lp(s)
        self.assertIn("--preset 6", res)
        self.assertIn("--crf 20", res)
        self.assertNotIn("--lp", res)

    def test_strip_lp_windows_path_handling(self):
        s = r"--my-path C:\foo\bar"
        res = _strip_lp(s)
        self.assertEqual(res, s)

    @patch('worker.IS_WINDOWS', True)
    def test_strip_lp_windows_simulation(self):
        s = r"--my-path C:\foo\bar"
        res = _strip_lp(s)
        self.assertEqual(res, s)

    def test_build_args_basic(self):
        job = Job(0, Path("in.mkv"), Path("out.mkv"), Path("temp"), Path("t.log"), Path("m.log"), Path("v.json"))
        with patch("worker.calculate_optimal_workers", return_value=(4, 2)):
             args = self.runner._build_av1an_args(job, [0, 1, 2, 3])

        e_idx = args.index("-e")
        e_arg = args[e_idx + 1]

        # 'svt' isn't quoted by shlex.quote because it's simple string
        self.assertIn("svt", e_arg)
        self.assertIn("--lp 2", e_arg)
        self.assertIn("-i stdin", e_arg)
        self.assertIn("--output {}", e_arg)

        w_idx = args.index("-w")
        self.assertEqual(args[w_idx + 1], "4")

if __name__ == '__main__':
    unittest.main()
