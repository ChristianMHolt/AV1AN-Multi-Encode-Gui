import sys
import unittest
import shlex
from unittest.mock import MagicMock
from pathlib import Path

# --- Mocking dependencies ---
# We mock PySide6/PyQt6 and psutil to avoid import errors and side effects
mock_qt = MagicMock()
sys.modules["PySide6"] = mock_qt
sys.modules["PySide6.QtCore"] = mock_qt
sys.modules["PyQt6"] = mock_qt
sys.modules["PyQt6.QtCore"] = mock_qt
sys.modules["psutil"] = MagicMock()

# --- Setup sys.path ---
# Add 'src' directory to sys.path so we can import 'worker' and its local dependencies
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from worker import _strip_lp
except ImportError as e:
    # If import fails, we can't run tests properly.
    # This might happen if 'config.py' or 'models.py' have issues,
    # but based on the code, they seem standard.
    raise ImportError(f"Failed to import worker module: {e}")


class TestStripLp(unittest.TestCase):
    def test_basic_removal(self):
        """Test removing --lp with a simple integer value."""
        cmd = "av1an -i input.mkv --lp 4 -o output.mkv"
        # tokens: ['av1an', '-i', 'input.mkv', '--lp', '4', '-o', 'output.mkv']
        # stripped: ['av1an', '-i', 'input.mkv', '-o', 'output.mkv']
        expected = shlex.join(['av1an', '-i', 'input.mkv', '-o', 'output.mkv'])
        self.assertEqual(_strip_lp(cmd), expected)

    def test_no_removal(self):
        """Test a string that doesn't contain --lp."""
        cmd = "av1an -i input.mkv -o output.mkv"
        # Since input doesn't have quotes or special chars that need escaping,
        # shlex.join(shlex.split(cmd)) should be equal to cmd if cmd is simple.
        # But to be safe, we compare against processed expectation.
        expected = shlex.join(shlex.split(cmd))
        self.assertEqual(_strip_lp(cmd), expected)

    def test_multiple_occurrences(self):
        """Test removing multiple --lp flags."""
        cmd = "--lp 1 --other --lp 2"
        # tokens: ['--lp', '1', '--other', '--lp', '2']
        # stripped: ['--other']
        expected = "--other"
        self.assertEqual(_strip_lp(cmd), expected)

    def test_lp_at_start(self):
        """Test --lp at the beginning of the string."""
        cmd = "--lp 8 -v 'some opts'"
        # tokens: ['--lp', '8', '-v', 'some opts']
        # stripped: ['-v', 'some opts']
        expected = shlex.join(['-v', 'some opts'])
        self.assertEqual(_strip_lp(cmd), expected)

    def test_lp_at_end(self):
        """Test --lp at the end of the string."""
        cmd = "-v opts --lp 16"
        # tokens: ['-v', 'opts', '--lp', '16']
        # stripped: ['-v', 'opts']
        expected = shlex.join(['-v', 'opts'])
        self.assertEqual(_strip_lp(cmd), expected)

    def test_quoted_values(self):
        """Test --lp with quoted values."""
        cmd = "--lp '4' --other"
        # tokens: ['--lp', '4', '--other']
        # stripped: ['--other']
        expected = "--other"
        self.assertEqual(_strip_lp(cmd), expected)

    def test_empty_string(self):
        """Test empty string."""
        self.assertEqual(_strip_lp(""), "")

    def test_lp_without_value(self):
        """Test --lp as the last token (no value)."""
        cmd = "command --lp"
        # tokens: ['command', '--lp']
        # stripped: ['command']
        expected = "command"
        self.assertEqual(_strip_lp(cmd), expected)

    def test_preservation_of_complex_quotes(self):
        """Test that other quoted arguments are preserved correctly."""
        cmd = "--lp 4 --vf 'scale=1920:1080, format=yuv420p'"
        # tokens: ['--lp', '4', '--vf', 'scale=1920:1080, format=yuv420p']
        # stripped: ['--vf', 'scale=1920:1080, format=yuv420p']
        expected = shlex.join(["--vf", "scale=1920:1080, format=yuv420p"])
        self.assertEqual(_strip_lp(cmd), expected)

if __name__ == "__main__":
    unittest.main()
