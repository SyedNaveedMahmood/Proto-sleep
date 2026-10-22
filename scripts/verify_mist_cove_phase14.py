"""Generated-only correctness; actual source records are never CI fixtures."""
import subprocess
import sys
if __name__=='__main__':raise SystemExit(subprocess.call([sys.executable,'-m','pytest','tests/mist_cove/test_phase14.py','-ra']))
