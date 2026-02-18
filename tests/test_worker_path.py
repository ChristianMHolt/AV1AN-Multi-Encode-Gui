import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Define a simple Mock QObject to avoid MagicMock inheritance issues
class MockQObject:
    def __init__(self, parent=None):
        pass

# Mock modules
mock_qt = MagicMock()
mock_qt.QObject = MockQObject
sys.modules["PySide6"] = MagicMock()
sys.modules["PySide6.QtCore"] = mock_qt
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
sys.modules["psutil"] = MagicMock()

# Import worker
sys.path.append(str(os.path.abspath("src")))
from worker import Runner

class TestWorkerPath(unittest.TestCase):
    def setUp(self):
        # Patch Path in worker
        self.patcher_path = patch("worker.Path")
        self.mock_path_cls = self.patcher_path.start()

        # Patch shutil.which
        self.patcher_which = patch("worker.shutil.which")
        self.mock_which = self.patcher_which.start()

        # Patch os.makedirs
        self.patcher_makedirs = patch("worker.os.makedirs")
        self.mock_makedirs = self.patcher_makedirs.start()

    def tearDown(self):
        self.patcher_path.stop()
        self.patcher_which.stop()
        self.patcher_makedirs.stop()

    def test_runner_path_relative(self):
        # Setup initial path instance
        mock_path_instance = MagicMock()
        self.mock_path_cls.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        # Setup parent of initial path (for before fix logic, but now irrelevant if resolved)
        # But wait, if resolve fails, it falls back. So keep it just in case.
        mock_parent = MagicMock()
        mock_path_instance.parent = mock_parent
        mock_parent.exists.return_value = True
        mock_parent.__str__.return_value = "src"

        # Setup resolved path
        mock_abs_path = MagicMock()
        mock_path_instance.resolve.return_value = mock_abs_path
        mock_abs_path.exists.return_value = True # Should also exist

        # Setup parent of resolved path
        mock_abs_parent = MagicMock()
        mock_abs_path.parent = mock_abs_parent
        mock_abs_parent.exists.return_value = True
        mock_abs_parent.__str__.return_value = "/abs/src"

        config = {
            "output_dir": "out",
            "temp_dir": "temp",
            "svt_path": "src/SvtAv1EncApp.exe",
            "calc_vmaf": False
        }

        # We also need to mock LOG_DIR.mkdir
        with patch("worker.LOG_DIR") as mock_log_dir:
             runner = Runner(config)

        path_env = runner.proc_env.get("PATH", "")
        added_path = path_env.split(os.pathsep)[0]

        print(f"Added PATH: {added_path}")

        self.assertEqual(added_path, "/abs/src", f"PATH entry '{added_path}' is not absolute! Expected '/abs/src'.")

if __name__ == "__main__":
    unittest.main()
