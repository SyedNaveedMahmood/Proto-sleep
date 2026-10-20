"""MIST-WIC: defaults to metadata verification; synthetic tracks are explicit."""
import argparse
import json
from mist_cove.phase9.experiments import execute,pilot,plan_proof,frozen

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='verify',choices=['verify','pilot','circuits','heads','robustness','stability','operators','reliability','report'])
    p.add_argument('--output');a=p.parse_args()
    if a.stage=='verify':print(json.dumps(plan_proof()))
    elif a.stage=='pilot':
        if not a.output:p.error('pilot requires a new ignored --output')
        pilot(a.output)
    elif a.stage=='report':
        from mist_cove.phase9.reporting import publish_report
        publish_report()
    else:execute(a.stage)
