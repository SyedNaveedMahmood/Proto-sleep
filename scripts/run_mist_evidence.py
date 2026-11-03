#!/usr/bin/env python3
"""Run the isolated MIST-Evidence candidate without importing legacy globals."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from mist_evidence.cli import main

if __name__ == "__main__":
    main()
