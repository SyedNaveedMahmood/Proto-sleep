#!/usr/bin/env python3
import argparse
import signal
from mist_cove.phase7.robustness import run

def main():
    p=argparse.ArgumentParser(description='Frozen prospective robustness controls; no real data');p.add_argument('--output',required=True);args=p.parse_args()
    def timeout(*_):raise TimeoutError('900-second robustness ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900);run(args.output)

if __name__=='__main__':main()
