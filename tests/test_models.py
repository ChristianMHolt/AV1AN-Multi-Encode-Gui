import unittest
import sys
import os
from pathlib import Path

# Add src to sys.path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models import Job

class TestJobCompressionRatio(unittest.TestCase):
    def setUp(self):
        # Create a dummy job instance
        # We use a non-existent path so stat() fails and original_size becomes 0 initially
        self.job = Job(
            idx=0,
            infile=Path("dummy_in.mkv"),
            out_mkv=Path("dummy_out.mkv"),
            tempdir=Path("dummy_temp"),
            term_log=Path("dummy_term.log"),
            mux_log=Path("dummy_mux.log"),
            vmaf_log=Path("dummy_vmaf.log")
        )

    def test_compression_ratio_standard(self):
        """Test standard compression scenario (50% reduction)."""
        self.job.original_size = 1000
        self.job.encoded_size = 500
        self.assertEqual(self.job.compression_ratio, 50.0)

    def test_compression_ratio_no_change(self):
        """Test no compression scenario (0% reduction)."""
        self.job.original_size = 1000
        self.job.encoded_size = 1000
        self.assertEqual(self.job.compression_ratio, 0.0)

    def test_compression_ratio_expansion(self):
        """Test expansion scenario (negative reduction)."""
        self.job.original_size = 1000
        self.job.encoded_size = 2000
        self.assertEqual(self.job.compression_ratio, -100.0)

    def test_compression_ratio_zero_original(self):
        """Test zero original size (avoid division by zero)."""
        self.job.original_size = 0
        self.job.encoded_size = 500
        self.assertEqual(self.job.compression_ratio, 0.0)

    def test_compression_ratio_zero_encoded(self):
        """Test zero encoded size."""
        self.job.original_size = 1000
        self.job.encoded_size = 0
        # If encoded size is 0 and original > 0, ratio is 100% theoretically?
        # No, the code says:
        # if self.original_size > 0 and self.encoded_size > 0:
        #     return (1 - self.encoded_size / self.original_size) * 100
        # return 0.0
        # So if encoded_size is 0, it returns 0.0.
        self.assertEqual(self.job.compression_ratio, 0.0)

    def test_compression_ratio_both_zero(self):
        """Test both zero sizes."""
        self.job.original_size = 0
        self.job.encoded_size = 0
        self.assertEqual(self.job.compression_ratio, 0.0)

if __name__ == '__main__':
    unittest.main()
