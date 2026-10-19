"""Execute frozen localized synthetic gates only; no source EEG interface."""
import argparse
from pathlib import Path
import signal
from mist_cove.phase6.experiment import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    def expired(signum,frame):raise TimeoutError('Phase VI external 1800-second ceiling')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
    try:
        result=run(a.output);print('COMPLETE',result['gates'],flush=True)
    finally:signal.alarm(0)


if __name__=='__main__':main()
