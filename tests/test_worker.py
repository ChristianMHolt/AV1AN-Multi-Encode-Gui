import sys
import os
import unittest
from unittest.mock import MagicMock

# --- Mocking dependencies before importing worker ---

# Mock PySide6
QtCore = MagicMock()
QtCore.QObject = object  # QObject must be a class
QtCore.Signal = MagicMock() # Signal can be anything
QtCore.QThread = object # QThread must be a class
QtCore.QTimer = MagicMock() # QTimer can be anything

PySide6 = MagicMock()
PySide6.QtCore = QtCore

sys.modules["PySide6"] = PySide6
sys.modules["PySide6.QtCore"] = QtCore

# Mock psutil
sys.modules["psutil"] = MagicMock()

# --- Add src to sys.path ---
# Get absolute path to src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import function to test ---
try:
    from worker import calculate_optimal_workers
except ImportError as e:
    print(f"Failed to import worker: {e}")
    sys.exit(1)

class TestCalculateOptimalWorkers(unittest.TestCase):

    def test_preset_workers_int(self):
        """Test with specific integer preset_workers (Happy Path)."""
        # If preset_workers is int > 0, it should return (preset_workers, max(1, chunk_size // preset_workers))
        self.assertEqual(calculate_optimal_workers(100, 4), (4, 25))
        self.assertEqual(calculate_optimal_workers(10, 2), (2, 5))
        self.assertEqual(calculate_optimal_workers(1, 1), (1, 1))

        # Test where chunk_size // preset_workers < 1, should default to 1
        self.assertEqual(calculate_optimal_workers(3, 4), (4, 1))

    def test_preset_workers_auto_large_chunk(self):
        """Test with 'auto' preset and chunk_size >= 32."""
        # chunk_size >= 32 -> (8, 4)
        self.assertEqual(calculate_optimal_workers(32, "auto"), (8, 4))
        self.assertEqual(calculate_optimal_workers(100, "auto"), (8, 4))
        self.assertEqual(calculate_optimal_workers(33, "auto"), (8, 4))

    def test_preset_workers_auto_medium_chunk(self):
        """Test with 'auto' preset and 16 <= chunk_size < 32."""
        # 16 <= chunk_size < 32 -> (8, chunk_size // 8)
        self.assertEqual(calculate_optimal_workers(16, "auto"), (8, 2))
        self.assertEqual(calculate_optimal_workers(24, "auto"), (8, 3))
        self.assertEqual(calculate_optimal_workers(31, "auto"), (8, 3))

    def test_preset_workers_auto_small_chunk(self):
        """Test with 'auto' preset and 8 <= chunk_size < 16."""
        # 8 <= chunk_size < 16 -> (4, chunk_size // 4)
        self.assertEqual(calculate_optimal_workers(8, "auto"), (4, 2))
        self.assertEqual(calculate_optimal_workers(12, "auto"), (4, 3))
        self.assertEqual(calculate_optimal_workers(15, "auto"), (4, 3))

    def test_preset_workers_auto_tiny_chunk(self):
        """Test with 'auto' preset and chunk_size < 8."""
        # chunk_size < 8 -> (1, chunk_size)
        self.assertEqual(calculate_optimal_workers(7, "auto"), (1, 7))
        self.assertEqual(calculate_optimal_workers(4, "auto"), (1, 4))
        self.assertEqual(calculate_optimal_workers(1, "auto"), (1, 1))
        self.assertEqual(calculate_optimal_workers(0, "auto"), (1, 0))

    def test_preset_workers_negative_or_zero(self):
        """Test with preset_workers <= 0 (should fall back to auto logic)."""
        # If preset_workers is int but <= 0, logic falls through to auto logic
        # chunk_size=32 -> (8, 4)
        self.assertEqual(calculate_optimal_workers(32, 0), (8, 4))
        self.assertEqual(calculate_optimal_workers(32, -5), (8, 4))

    def test_chunk_size_negative(self):
        """Test with negative chunk_size."""
        # If chunk_size < 8, returns (1, chunk_size)
        self.assertEqual(calculate_optimal_workers(-10, "auto"), (1, -10))

if __name__ == '__main__':
    unittest.main()
