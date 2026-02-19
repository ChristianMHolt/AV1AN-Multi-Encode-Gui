import unittest
from pathlib import PureWindowsPath

class TestPathNormalization(unittest.TestCase):
    def test_windows_path_str(self):
        # PureWindowsPath should normalize slashes to backslashes when converted to string
        p = PureWindowsPath("C:/foo/bar")
        s = str(p)
        self.assertEqual(s, r"C:\foo\bar")

    def test_explicit_replace(self):
        # Just to be absolutely sure what our proposed fix does
        p_str = "C:/foo/bar"
        fixed = p_str.replace("/", "\\")
        self.assertEqual(fixed, r"C:\foo\bar")

if __name__ == '__main__':
    unittest.main()
