"""Create the Phase XV exploratory lock from the explicitly supplied prompt."""
import argparse
from mist_cove.phase15.governance import prepare

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorized-prompt',required=True)
    args=parser.parse_args()
    prepare(args.authorized_prompt)
