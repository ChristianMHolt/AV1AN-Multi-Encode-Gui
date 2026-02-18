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
from worker import Runner

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

if __name__ == '__main__':
    unittest.main()
