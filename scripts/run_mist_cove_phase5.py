"""Run frozen Phase V synthetic-only sequential gates; never real EEG."""
import argparse
import signal
from pathlib import Path
from mist_cove.phase5.experiment import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def expired(signum,frame): raise TimeoutError('external Phase V 1800-second wall ceiling')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
    try:
        report=run(a.output)
        print('COMPLETE',report['gates'],flush=True)
    finally: signal.alarm(0)


if __name__=='__main__': main()
