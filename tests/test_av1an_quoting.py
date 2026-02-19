import unittest
import shlex
import subprocess

class TestAv1anQuoting(unittest.TestCase):
    def test_shlex_join_windows_path(self):
        # shlex.join escapes backslashes
        # av1an (POSIX shell parser) consumes 'C:\\path' -> C:\path
        # This is correct for av1an input
        s = r"C:\path\to\file"
        joined = shlex.join([s])
        self.assertEqual(joined, r"'C:\path\to\file'")
        # shlex.join wraps in single quotes. Inside single quotes, backslash is literal?
        # On POSIX: 'C:\path' -> C:\path. YES.

    def test_list2cmdline_windows_path(self):
        # list2cmdline does NOT escape backslashes (unless before quote)
        # C:\path -> C:\path
        # av1an (POSIX shell parser) consumes C:\path -> C:path (escape sequence)
        # This causes mangling if passed to av1an!
        s = r"C:\path\to\file"
        joined = subprocess.list2cmdline([s])
        self.assertEqual(joined, r"C:\path\to\file")

    def test_manual_split_replace(self):
        # Verify the proposed fix strategy: replace backslashes before split
        s = r"--path C:\foo\bar"
        # If we replace backslashes with double backslashes:
        s_escaped = s.replace("\\", "\\\\")
        # shlex.split (POSIX) will see C:\\foo\\bar -> C:\foo\bar
        toks = shlex.split(s_escaped)
        self.assertEqual(toks, ["--path", r"C:\foo\bar"])

if __name__ == '__main__':
    unittest.main()
