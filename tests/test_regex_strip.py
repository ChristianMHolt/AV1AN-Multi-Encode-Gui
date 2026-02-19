import unittest
import re

def strip_lp_regex(s):
    # Matches --lp followed by spaces or =, then digits, surrounded by word boundary or space
    # (?:^|\s) ensures we don't match foo--lp
    # --lp\s*=?\s*\d+ matches --lp 4, --lp=4, --lp  4
    # (?:\s|$) ensures we don't match --lp42 (unless digits)
    return re.sub(r'(?:^|\s)--lp\s*=?\s*\d+(?:\s|$)', ' ', s).strip()

class TestRegexStrip(unittest.TestCase):
    def test_strip_lp_simple(self):
        s = "--preset 6 --lp 4 --crf 20"
        res = strip_lp_regex(s)
        # Expected: --preset 6 --crf 20 (spaces might remain, strip handles ends)
        self.assertIn("--preset 6", res)
        self.assertIn("--crf 20", res)
        self.assertNotIn("--lp", res)

    def test_strip_lp_equals(self):
        s = "--preset 6 --lp=4 --crf 20"
        res = strip_lp_regex(s)
        self.assertIn("--preset 6", res)
        self.assertNotIn("--lp", res)

    def test_strip_lp_start(self):
        s = "--lp 4 --preset 6"
        res = strip_lp_regex(s)
        self.assertIn("--preset 6", res)
        self.assertNotIn("--lp", res)

    def test_strip_lp_end(self):
        s = "--preset 6 --lp 4"
        res = strip_lp_regex(s)
        self.assertIn("--preset 6", res)
        self.assertNotIn("--lp", res)

    def test_windows_path_preserved(self):
        # Crucial test: Does regex mangle backslashes?
        s = r"--path C:\foo\bar --lp 4"
        res = strip_lp_regex(s)
        # Verify backslashes are intact
        self.assertIn(r"C:\foo\bar", res)
        self.assertNotIn("--lp", res)

    def test_quoted_path_preserved(self):
        s = r'--path "C:\foo\bar" --lp 4'
        res = strip_lp_regex(s)
        self.assertIn(r'"C:\foo\bar"', res)

if __name__ == '__main__':
    unittest.main()
