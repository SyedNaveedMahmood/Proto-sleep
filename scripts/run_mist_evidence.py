#!/usr/bin/env python3
"""Entry point for the separate MIST-Evidence research branch."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protosleep.evidence.cli import main

if __name__ == "__main__":
    main()
