"""Isolated detector transport and fixed generated plausibility/resource controls."""
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from .common import ROOT,PRIVATE,PUBLIC,write_new,read_json,sha256,measured,ledger
from .signal_preprocessing import frequency_checks


class Detector:
    def __init__(self):
        self.stderr=(PRIVATE/'detector_worker.log').open('a')
        python=PRIVATE/'detector_environment/bin/python'
        self.worker=subprocess.Popen([str(python),str(Path(__file__).with_name('detector_worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True)
        self.ordinal=0
    def run(self,arrays,config,start,core_start,core_stop,duration):
        self.ordinal+=1;label=f'worker_{os.getpid()}_{self.ordinal}'
        input_path=PRIVATE/(label+'.npz');output=PRIVATE/(label+'.json')
        np.savez(input_path,**arrays)
        request={'input':str(input_path),'input_sha256':sha256(input_path.read_bytes()),'output':str(output),'config':config,
                 'start':start,'core_start':core_start,'core_stop':core_stop,'duration':duration}
        self.worker.stdin.write(json.dumps(request)+'\n');self.worker.stdin.flush();line=self.worker.stdout.readline()
        if not line:raise RuntimeError('isolated detector stopped; see private log')
        status=json.loads(line)
        if status['status']!='SUCCESS':raise RuntimeError('detector transport failure: '+status.get('reason',''))
        data=output.read_bytes()
        if sha256(data)!=status['output_sha256']:raise ValueError('worker result identity differs')
        result=json.loads(data);input_path.unlink();output.unlink();return result
    def close(self):
        self.worker.stdin.close();code=self.worker.wait(timeout=30);self.stderr.close()
        if code:raise RuntimeError('detector process failed at close')


def pilot(stage='generated_detector_and_resource_pilot'):
    with measured(stage) as counts:
        config=read_json(ROOT/'configs/mist_cove_phase14_v1.json');rng=np.random.default_rng(14018);n=186000;t=np.arange(n)/100
        x=rng.normal(0,2,n)
        for center in range(30,1800,20):x+=25*np.exp(-.5*((t-center)/.3)**2)*np.sin(2*np.pi*13*t)
        for center in range(40,1800,60):
            a=(t>=center)&(t<center+4);x[a]+=90*np.sin(2*np.pi*(t[a]-center))
        arrays={'EEG Fpz-Cz':x.astype('float32'),'EEG Pz-Oz':(x*.9).astype('float32'),'EOG horizontal':(60*np.sin(2*np.pi*.5*t)).astype('float32'),
                'EMG submental':np.ones(1860,dtype='float32')*5,'EMG submental__fs':np.array(1.),'EEG Fpz-Cz__bounds':np.array([-1000,1000]),'EEG Pz-Oz__bounds':np.array([-1000,1000])}
        worker=Detector();start=time.perf_counter();result=worker.run(arrays,config,0,0,1800,1860);elapsed=time.perf_counter()-start
        negative={k:np.zeros_like(v) if k in ('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal') else v for k,v in arrays.items()}
        null=worker.run(negative,config,0,0,1800,1860);worker.close()
        positives={kind:len(result['events']['EEG Fpz-Cz|'+kind]) for kind in ('spindle','slow_wave')}
        negatives={kind:len(null['events']['EEG Fpz-Cz|'+kind]) for kind in ('spindle','slow_wave')}
        geometry=read_json(PRIVATE/'geometry.json');total_seconds=sum(r['duration'] for r in geometry['rows']);estimate=elapsed*total_seconds/1800
        passed=all(positives.values()) and not any(negatives.values()) and not result['failures'] and estimate<18000
        write_new(PUBLIC/'detector_manifest.json',{'status':'PASS' if passed else 'BLOCKED','YASA_version':result['YASA_version'],'detector_config':config['detector'],
            'generated_positive_event_counts':positives,'generated_flat_event_counts':negatives,'generated_detector_failures':result['failures'],
            'generated_30minute_pilot_seconds':elapsed,'conservative_projected_all_sources_detector_seconds':estimate,
            'frequency_response_checks':frequency_checks(),'stage_blind':True,'pretrained_weights':False,
            'published_methods':['YASA0.6.5spindles/Lacourse2018inspired','YASA0.6.5slowwaves/Massimini2004andCarrier2011inspired'],
            'human_event_accuracy':'NOT_TESTED','threshold_tuning':'NONE; primary defaults and alternatives frozen before source outcomes',
            'montage_limit':'BipolarFpz-Cz/Pz-Oz differs from common scoring derivations; amplitude cutoffs are unvalidated proxies',
            'runtime_interpretation':'conservative includes worker imports/initial JIT; no waveform inference on ST'})
        counts.update(generated_only=True,pilot_seconds=elapsed,projected_detector_seconds=estimate)
        ledger([9,10,11,12,13,14,15,16,18],'COMPLETE','Secure calibrated multichannel loader and stage-blind YASA/spectral/EOG/EMG/artifact proxies implemented; generated positive/flat and filter controls executed',['reports/mist_cove/phase14_v1/detector_manifest.json'],science='NOT_TESTED')
        if not passed:raise RuntimeError('detector preflight failed; do not run scientific detectors')
        print(json.dumps({'detector_pilot':passed,'seconds':elapsed,'projected':estimate,'positive':positives,'flat':negatives}),flush=True)
