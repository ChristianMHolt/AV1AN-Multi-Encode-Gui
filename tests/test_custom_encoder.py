import unittest
import shlex
from pathlib import Path

class TestCustomEncoder(unittest.TestCase):
    def test_encoder_cmd_construction(self):
        svt_path = r"C:\Program Files\SVT\SvtAv1EncApp.exe"
        svt_cli = "--preset 6 --lp 4"

        # We need to quote the path because of spaces
        quoted_path = shlex.quote(svt_path)
        # Note: shlex.quote on Linux (sandbox) produces single quotes: 'path'
        # av1an parses this correctly.

        cmd = f"{quoted_path} {svt_cli} -i stdin --output {{}}"

        self.assertIn("'C:\\Program Files\\SVT\\SvtAv1EncApp.exe'", cmd)
        self.assertIn("--preset 6", cmd)
        self.assertIn("-i stdin", cmd)
        self.assertIn("--output {}", cmd)

if __name__ == '__main__':
    unittest.main()
