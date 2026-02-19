import unittest
import shlex
import subprocess
import os

class TestWindowsQuoting(unittest.TestCase):
    def test_shlex_quote_backslash_posix(self):
        # shlex.quote creates POSIX-compatible strings (single quotes)
        # This is what _strip_lp was doing, and what caused problems on Windows
        s = r"C:\path\to\file"
        quoted = shlex.quote(s)
        self.assertEqual(quoted, "'C:\\path\\to\\file'") # Single quotes around literal backslashes

    def test_list2cmdline_backslash_windows(self):
        # subprocess.list2cmdline creates Windows-compatible strings (double quotes)
        # This is what we WANT on Windows
        s = r"C:\path\to\file"
        # list2cmdline only quotes if needed (space/tab).
        # A simple path shouldn't be quoted unless it has spaces.
        quoted = subprocess.list2cmdline([s])
        self.assertEqual(quoted, r"C:\path\to\file")

    def test_list2cmdline_space_windows(self):
        s = r"C:\Program Files\App"
        quoted = subprocess.list2cmdline([s])
        self.assertEqual(quoted, r'"C:\Program Files\App"') # Double quotes!

    def test_shlex_split_windows_backslash(self):
        # Verify shlex.split behavior with posix=False
        s = r"prog --opt C:\path\to\file"
        toks = shlex.split(s, posix=False)
        self.assertEqual(toks, ["prog", "--opt", r"C:\path\to\file"])

if __name__ == '__main__':
    unittest.main()
