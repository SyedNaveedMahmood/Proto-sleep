"""Generated fixtures only; never load private data or fitted states."""
import subprocess
import sys
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','tests/mist_cove/test_phase13_foundation.py',
        'tests/mist_cove/test_phase13_annotations.py','tests/mist_cove/test_phase13_training.py','tests/mist_cove/test_phase13_analysis.py','-ra']))
