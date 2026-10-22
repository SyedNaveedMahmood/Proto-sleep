"""Read-only inheritance checks and the prospective lock for retrospective analyses."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT/'reports/mist_cove/phase15_v1'
PRIVATE = ROOT/'mist_transfer_runs/mist_cove/phase15_v1'
OLD_PUBLIC = ROOT/'reports/mist_cove/phase14_v1'
OLD_PRIVATE = ROOT/'mist_transfer_runs/mist_cove/phase14_v1'
CONFIG = ROOT/'configs/mist_cove_phase15_v1.json'
BASE = '74ce0b0bac7e8fd9f53fb6d7adcf209ee9b4f5ce'
TREE = '3a539db6a5ba232cd6b042a7063e6f50d98125eb'
BRANCH = 'research/mist-cove-phase15-rem-proxy-method-value-v1'
LABEL = 'EXPLORATORY_REUSED_ST'


def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path): return json.loads(path.read_text())
def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def assert_base():
    if git('rev-parse','HEAD')!=BASE or git('rev-parse','HEAD^{tree}')!=TREE:
        raise ValueError('exact Phase XIV starting commit/tree required')
    if git('branch','--show-current')!=BRANCH:raise ValueError('Phase XV branch required')
    if git('rev-parse','origin/research/mist-cove-phase14-multichannel-st-transfer-v1')!=BASE:
        raise ValueError('origin Phase XIV identity differs')


def inherit():
    old = read(OLD_PUBLIC/'frozen_st_protocol.json')
    model = read(OLD_PUBLIC/'model_authentication.json')
    original = read(OLD_PUBLIC/'st_classification_metrics.json')
    gates = read(OLD_PUBLIC/'scientific_gates.json')
    cache_receipt = read(OLD_PUBLIC/'ST_EVAL_witness_execution.json')
    manifest = read(OLD_PRIVATE/'source_manifest.json')
    geometry = read(OLD_PRIVATE/'geometry.json')
    if sha(canonical(manifest))!=old['source_manifest_sha256'] or sha(canonical(geometry))!=old['geometry_sha256']:
        raise ValueError('Phase XIV private source identities differ')
    if model['status']!='PASS' or model['checkpoint_byte_sha256']!=old['original_checkpoint_byte_sha256'] or model['named_state_digest']!=old['original_state_digest']:
        raise ValueError('historical source model identity differs')
    if sum(r['record']['role']=='SC_TRAIN' for r in geometry['rows'])!=29 or sum(r['record']['role']=='ST_EVAL' for r in geometry['rows'])!=44:
        raise ValueError('geometry role count differs')
    if set(r['record']['role'] for r in geometry['rows'])!={'SC_TRAIN','ST_EVAL'}:
        raise ValueError('unapproved role in inherited geometry')
    rem = original['metrics']['confusion_matrix'][4]
    if rem!=[2322,1570,2648,4,1805] or original['metrics']['n_epochs']!=42691:
        raise ValueError('published REM confusion or scored grid differs')
    if cache_receipt['resources']['tables']!=1095 or not cache_receipt['historical_state_unchanged']:
        raise ValueError('Phase XIV witness cache missing')
    if gates['ST_proxy']['spectral_increment']['status']!='FAIL' or gates['SC_proxy']['proxy_primary']['status']!='NOT_ESTIMABLE':
        raise ValueError('inherited negative gates differ')
    release = read(OLD_PUBLIC/'publication_manifest.json')
    for relative,expected in release['file_sha256'].items():
        if sha((ROOT/relative).read_bytes())!=expected:raise ValueError('Phase XIV public artifact differs: '+relative)
    return {'source_manifest_canonical_sha256':sha(canonical(manifest)),'geometry_canonical_sha256':sha(canonical(geometry)),
            'checkpoint_byte_sha256':model['checkpoint_byte_sha256'],'named_state_sha256':model['named_state_digest'],
            'approved_SC_recordings':29,'approved_ST_recordings':44,'scored_ST_epochs':original['metrics']['n_epochs'],
            'ST_REM_row':rem,'ST_witness_tables':1095,'Phase_XIV_public_manifest_files_checked':len(release['file_sha256']),
            'ST_spectral_increment_gate':'FAIL','SC_primary_proxy_gate':'NOT_ESTIMABLE'}


def verify_plan_commit():
    receipt=read(PUBLIC/'analysis_protocol.json')
    if receipt['config_sha256']!=sha(CONFIG.read_bytes()):raise ValueError('Phase XV protocol changed')
    commit=git('log','-1','--format=%H','--',str(CONFIG.relative_to(ROOT)))
    if not commit:raise ValueError('Phase XV config must be committed before new analysis')
    committed=subprocess.check_output(['git','show',commit+':'+str(CONFIG.relative_to(ROOT))],cwd=ROOT)
    if committed!=CONFIG.read_bytes():
        raise ValueError('Phase XV committed protocol bytes differ')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True,capture_output=True)
    config=read(CONFIG)
    if config['all_ST_analysis_label']!=LABEL or config['resources']['new_neural_fits'] or config['resources']['new_optimizer_updates']:
        raise ValueError('retrospective/zero-training contract differs')
    return commit


def prepare(prompt_path):
    assert_base();inherited=inherit();config=read(CONFIG)
    if config['base_commit']!=BASE or config['status']!=LABEL:raise ValueError('config identity/status differs')
    text=Path(prompt_path).read_text();titles=re.findall(r'^## WP(\d\d) — (.+)$',text,re.M)
    if len(titles)!=48 or [int(i) for i,_ in titles]!=list(range(1,49)):
        raise ValueError('Phase XV authoritative package list differs')
    from importlib import metadata
    starting={'base_commit':BASE,'base_tree':TREE,'branch':BRANCH,'working_tree_clean_before_branch':True,
              'origin_phase14_matches':True,'phase14_pr':'https://github.com/SyedNaveedMahmood/Proto-sleep/pull/16',
              'python':sys.version.split()[0],
              'packages':{k:metadata.version(k) for k in ('torch','numpy','mne','pyedflib')},
              'baseline_CPU_passed':1139,'baseline_CUDA_skipped':11,
              'historical_experiments_modified':False}
    authorization={'status':LABEL,'new_raw_SC_VAL_reads':0,'new_SC_RESERVED_reads':0,'external_EEG_reads':0,'ST_training_updates':0,
                   'approved_SC_TRAIN_recordings':29,'approved_ST_recordings':44,
                   'allowed_channels':['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental'],
                   'dataset_root_source':'Phase XIV ignored local authorization; digest pinned; no new public literal',
                   'owner_accepted_earlier_root_path_in_Git_history':True,
                   'private_values_may_not_be_published':True,'Phase_XVI_execution':'NOT_AUTHORIZED'}
    exposure={'status':LABEL,'all_new_ST_analysis_label':LABEL,'UNTOUCHED_TEST_permitted':False,
              'previously_exposed':[
                  {'scope':'SC_TRAIN','measurement':'full grid original P3B classification','interpretation':'IN_SAMPLE_DEVELOPMENT'},
                  {'scope':'SC_TRAIN','measurement':'150 stage-balanced and 150 prevalence witness cases','interpretation':'DEVELOPMENTAL'},
                  {'scope':'SC_VAL','measurement':'historical approved aggregate only','interpretation':'NO_NEW_RAW_ACCESS'},
                  {'scope':'ST_EVAL','measurement':'44-night complete-grid classification and calibration','interpretation':LABEL},
                  {'scope':'ST_EVAL','measurement':'219 witness cases, 1095 five-view truth tables','interpretation':LABEL},
                  {'scope':'ST_EVAL','measurement':'Pz-Oz spindle proxy random PASS, sigma increment FAIL; detector thresholds and zero/.01 neutral compared','interpretation':LABEL},
                  {'scope':'SC_TRAIN','measurement':'five-person spindle developmental gate with 9 eligible cases','interpretation':'NOT_ESTIMABLE'},
                  {'scope':'SC_TRAIN','measurement':'three-seed C0/C1 transition gate FAIL','interpretation':'DEVELOPMENTAL'}],
              'new_phase15_panels':'selected after plan commit; descriptive/exploratory regardless of locking'}
    oldwp=next(r for r in read(OLD_PUBLIC/'progress.json')['work_packages'] if r['id']=='WP23')
    corrigendum={'phase':'XV','historical_phase':'XIV','work_package':'WP23',
                'published_ledger_status':oldwp['status'],'published_scientific_status':oldwp['scientific_status'],
                'published_witness_execution_status':read(OLD_PUBLIC/'SC_TRAIN_witness_execution.json')['status'],
                'published_SC_tables':read(OLD_PUBLIC/'SC_TRAIN_witness_execution.json')['resources']['tables'],
                'published_aggregate_exists':(OLD_PUBLIC/'sc_alignment_results.json').exists(),
                'interpretation':'status-aggregation field was left PENDING_AGGREGATION despite completed SC witness run and published aggregate; SC physiological primary gate remains NOT_ESTIMABLE',
                'Phase_XIV_ledger_modified':False,'scientific_gate_changed':False}
    nodes=[
        ('exact15','PROVEN_COMPUTATIONAL','15-atom closure, full identity and exhaustive finite tables'),
        ('phase9','SYNTHETIC','clean additive versus conventional and C2 versus C1 gates failed'),
        ('phase11','PROVEN_COMPUTATIONAL','historical source bytes/state and restricted loading authenticated'),
        ('phase12','TRAIN_DEVELOPMENT','source operator and real-model dependency audit'),
        ('phase13','TRAIN_DEVELOPMENT','exposure matters; width and routing incremental gates failed'),
        ('sc_full','TRAIN_IN_SAMPLE','79,025 scored original-model classification epochs'),
        ('sc_proxy','TRAIN_DEVELOPMENT','nine eligible cases, four people; primary gate not estimable'),
        ('st_full','FROZEN_PHASE_XIV_ST','42,691 scored epochs; REM recall 0.2162 and ECE15 0.1339'),
        ('st_proxy','FROZEN_PHASE_XIV_ST','25 eligible cases; random PASS, sigma increment FAIL'),
        ('context','TRAIN_DEVELOPMENT','six fits; transition gate failed all three seeds'),
        ('human','UNVERIFIED_PHYSIOLOGY','no qualified expert event validation'),
        ('phase15','EXPLORATORY_REUSED_ST','new REM/proxy/operator analyses cannot restore independent test status')]
    evidence={'nodes':[{'id':i,'category':c,'claim':f} for i,c,f in nodes],
              'question_links':{'REM_failure':['st_full','sc_full','phase11','phase15','human'],
                                'sigma_challenge':['exact15','st_proxy','sc_proxy','phase15','human'],
                                'strategy':['phase9','phase13','context','st_full','st_proxy','human']},
              'no_historical_fail_to_pass_promotion':True}
    ledger={'version':'mist-cove-phase15-progress-v1','base_commit':BASE,'analysis_status':LABEL,
            'config_sha256':sha(CONFIG.read_bytes()),'work_packages':[]}
    for i,title in titles:
        n=int(i);done=n<=7
        ledger['work_packages'].append({'id':'WP'+i,'title':title,'dependencies':([] if n==1 else ['WP'+str(n-1).zfill(2)]),
                                        'status':'COMPLETE' if done else 'PENDING',
                                        'engineering_status':'PASS' if done else 'PENDING',
                                        'scientific_status':('EXPLORATORY_GOVERNANCE' if done else 'NOT_RUN'),
                                        'code_configuration_identity':{'base_commit':BASE,'config_sha256':sha(CONFIG.read_bytes())},
                                        'tests':(['exact inherited source and publication identity checks'] if done else []),
                                        'artifacts':(['reports/mist_cove/phase15_v1/analysis_protocol.json'] if done else []),
                                        'findings':(['bounded retrospective governance; no new research outcomes'] if done else []),
                                        'limitations':(['ST outcomes were previously examined; protocol is an exploratory lock'] if done else []),
                                        'block_reason':None})
    for name,value in [('starting_commit.json',starting),('authorization_receipt.json',authorization),('exposure_ledger.json',exposure),
                       ('phase14_WP23_corrigendum.json',corrigendum),('historical_evidence_map.json',evidence),('progress.json',ledger),
                       ('analysis_protocol.json',{'status':LABEL,'config_sha256':sha(CONFIG.read_bytes()),'source':'configs/mist_cove_phase15_v1.json',
                                                  'lock_type':'exploratory reproducibility, not untouched prospective ST confirmation',
                                                  'commit_required_before_new_research_queries':True,'inherited_identity':inherited})]:
        new(PUBLIC/name,value)
    print(json.dumps({'governance':'COMPLETE','packages':48,'config_sha256':sha(CONFIG.read_bytes()),'inherited_public_files_checked':inherited['Phase_XIV_public_manifest_files_checked']}))
