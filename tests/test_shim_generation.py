import unittest
import sys
import subprocess
from pathlib import Path

# Simulation of the shim script logic
def shim_logic(args, real_exe):
    new_args = []
    i = 0
    while i < len(args):
        if args[i] == '-b':
            new_args.append('--output')
        else:
            new_args.append(args[i])
        i += 1
    return [real_exe] + new_args

class TestShimGeneration(unittest.TestCase):
    def test_shim_argument_replacement(self):
        # Simulate args passed to the shim
        original_args = ["--preset", "6", "-i", "input.ivf", "-b", "output.ivf"]
        real_exe = "C:\\SVT\\SvtAv1EncApp.exe"

        cmd = shim_logic(original_args, real_exe)

        self.assertEqual(cmd[0], real_exe)
        self.assertIn("--output", cmd)
        self.assertNotIn("-b", cmd)

        # Verify order and structure
        output_idx = cmd.index("--output")
        self.assertEqual(cmd[output_idx + 1], "output.ivf")

if __name__ == '__main__':
    unittest.main()
