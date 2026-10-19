#!/usr/bin/env python3
"""Explicit prospective synthetic identity; never loads Phase VI weights."""
import argparse
import signal
from mist_cove.phase7.prospective import run_diagnostics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    def timeout(*_):raise TimeoutError('900-second prospective diagnostic ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    run_diagnostics(a.output)

if __name__=='__main__':main()
