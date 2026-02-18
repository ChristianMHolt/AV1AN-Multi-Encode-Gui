import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

# Append src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Import PySide6 before importing widgets to ensure it's available
try:
    from PySide6.QtWidgets import QApplication, QLabel
    from ui.widgets import JobTile
    from models import Job, JobStatus
except ImportError:
    QApplication = None
    JobTile = None
    Job = None
    JobStatus = None

def create_dummy_job():
    j = Job(
        idx=0,
        infile=Path("test.mkv"),
        out_mkv=Path("out.mkv"),
        tempdir=Path("/tmp"),
        term_log=Path("/tmp/term.log"),
        mux_log=Path("/tmp/mux.log"),
        vmaf_log=Path("/tmp/vmaf.log")
    )
    j.status = JobStatus.RUNNING
    j.fps_hist.append(24.0)
    j.pct = 50.0
    # Mock file existence for post_init logic if needed, but post_init uses stat()
    # We can just set original_size manually if post_init fails or sets 0
    j.original_size = 1000
    return j

class TestJobTile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication is None:
            raise unittest.SkipTest("PySide6 or dependencies not installed")
        # Create QApplication if it doesn't exist
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def test_job_tile_structure(self):
        job = create_dummy_job()
        try:
            tile = JobTile(job, MagicMock(), MagicMock(), MagicMock(), disable_graphs=True)
        except Exception as e:
            self.fail(f"Failed to instantiate JobTile: {e}")

        # Check initial structure
        self.assertTrue(hasattr(tile, 'info'), "JobTile should have info label")
        self.assertTrue(hasattr(tile, 'base_info_text'), "JobTile should have base_info_text")
        self.assertFalse(hasattr(tile, 'stats'), "JobTile should NOT have stats label anymore")
        self.assertTrue(hasattr(tile, 'bar'), "JobTile should have progress bar")

        # Verify base info
        self.assertIn("Managed Auto-Scaling", tile.base_info_text)

        # Stats are updated in __init__ and merged into info
        info_text_initial = tile.info.text()
        self.assertIn(tile.base_info_text, info_text_initial)
        self.assertIn("FPS:", info_text_initial)

        # Update UI (redundant but harmless)
        tile.update_ui()

        # Check that info text contains stats
        info_text = tile.info.text()
        print(f"Current Info Text: {info_text}")
        self.assertIn("FPS:", info_text)
        self.assertIn(" : ", info_text)

if __name__ == '__main__':
    unittest.main()
