import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

# 1. Setup psutil mock BEFORE importing worker
mock_psutil = MagicMock()
class MockPsutilError(Exception):
    pass
mock_psutil.Error = MockPsutilError
mock_psutil.NoSuchProcess = MockPsutilError
mock_psutil.AccessDenied = MockPsutilError

sys.modules["psutil"] = mock_psutil

# 2. Setup Qt mock
mock_qt = MagicMock()
sys.modules["PySide6.QtCore"] = mock_qt
sys.modules["PyQt6.QtCore"] = mock_qt
sys.modules["PySide6"] = mock_qt
sys.modules["PyQt6"] = mock_qt

# 3. Add src to path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import worker

class TestWorkerSuspend(unittest.TestCase):
    def setUp(self):
        # Reset mocks and clear side_effects
        mock_psutil.reset_mock()
        mock_psutil.Process.side_effect = None
        mock_psutil.Process.return_value = MagicMock() # Default return value

        # Ensure sys.modules["psutil"] is OUR mock
        sys.modules["psutil"] = mock_psutil

        # Reload worker to ensure it picks up the correct psutil mock
        importlib.reload(worker)

    def test_suspend_tree_psutil_error(self):
        # Simulate psutil.Process raising NoSuchProcess (a psutil.Error)
        mock_psutil.Process.side_effect = mock_psutil.NoSuchProcess("No process")

        # Should return False gracefully
        result = worker._suspend_tree(123)
        self.assertFalse(result)

    def test_suspend_tree_other_error(self):
        # Simulate a non-psutil error (e.g. RuntimeError)
        mock_psutil.Process.side_effect = RuntimeError("Something bad happened")

        # Should raise RuntimeError, NOT return False
        with self.assertRaises(RuntimeError):
            worker._suspend_tree(123)

    def test_suspend_children_error(self):
        # Simulate root process ok, but child suspend fails
        mock_proc = MagicMock()
        mock_child = MagicMock()
        mock_proc.children.return_value = [mock_child]
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.Process.side_effect = None # Ensure no side effect

        # Child suspend raises AccessDenied
        mock_child.suspend.side_effect = mock_psutil.AccessDenied("Denied")

        # Should catch the error and continue, returning True
        result = worker._suspend_tree(123)
        self.assertTrue(result)
        mock_child.suspend.assert_called_once()

    def test_resume_tree_other_error(self):
        # Simulate a non-psutil error (e.g. RuntimeError)
        mock_psutil.Process.side_effect = RuntimeError("Something bad happened")

        # Should raise RuntimeError, NOT return False
        with self.assertRaises(RuntimeError):
            worker._resume_tree(123)

if __name__ == '__main__':
    unittest.main()
