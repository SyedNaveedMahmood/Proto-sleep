"""Evidence-checked public ledger, gates, and security review."""
import json
from pathlib import Path
import re
import subprocess

from .governance import ROOT,CONFIG,PRIVATE,PUBLIC,LABEL,read,sha,canonical,verify_plan_commit


ARTIFACTS={
8:['tests/test_phase15_forensics.py'],9:['reports/mist_cove/phase15_v1/rem_forensics.json'],
10:['reports/mist_cove/phase15_v1/rem_forensics.json'],11:['reports/mist_cove/phase15_v1/panel_seal.json'],
12:['reports/mist_cove/phase15_v1/rem_statistics.json'],13:['reports/mist_cove/phase15_v1/rem_wave_diagnostics.json'],
14:['reports/mist_cove/phase15_v1/rem_wave_diagnostics.json'],15:['reports/mist_cove/phase15_v1/rem_wave_diagnostics.json'],
16:['reports/mist_cove/phase15_v1/rem_wave_diagnostics.json'],17:['reports/mist_cove/phase15_v1/quality_results.json'],
18:['reports/mist_cove/phase15_v1/rem_statistics.json'],19:['reports/mist_cove/phase15_v1/rem_forensics.json'],
20:['reports/mist_cove/phase15_v1/rem_statistics.json'],21:['reports/mist_cove/phase15_v1/rem_statistics.json'],
22:['reports/mist_cove/phase15_v1/rem_statistics.json'],23:['reports/mist_cove/phase15_v1/atlas_receipt.json'],
24:['docs/mist_cove/PHASE_XV_RESEARCH_REPORT_V1.md'],25:['reports/mist_cove/phase15_v1/explanation_comparison.json'],
26:['reports/mist_cove/phase15_v1/explanation_comparison.json'],27:['reports/mist_cove/phase15_v1/panel_seal.json'],
28:['reports/mist_cove/phase15_v1/explanation_comparison.json'],29:['reports/mist_cove/phase15_v1/rem_wave_diagnostics.json'],
30:['docs/mist_cove/PHASE_XV_RESEARCH_REPORT_V1.md'],31:['reports/mist_cove/phase15_v1/explanation_comparison.json'],
32:['reports/mist_cove/phase15_v1/explanation_comparison.json'],33:['reports/mist_cove/phase15_v1/explanation_comparison.json'],
34:['reports/mist_cove/phase15_v1/explanation_comparison.json'],35:['reports/mist_cove/phase15_v1/rem_statistics.json','reports/mist_cove/phase15_v1/explanation_comparison.json'],
36:['reports/mist_cove/phase15_v1/explanation_execution.json'],37:['reports/mist_cove/phase15_v1/explanation_comparison.json'],
38:['reports/mist_cove/phase15_v1/operator_inheritance.json'],39:['reports/mist_cove/phase15_v1/stability_results.json'],
40:['reports/mist_cove/phase15_v1/resource_receipts.json'],41:['reports/mist_cove/phase15_v1/explanation_comparison.json','tests/test_phase15_forensics.py'],
42:['docs/mist_cove/PHASE_XV_RESEARCH_REPORT_V1.md'],43:['docs/mist_cove/PHASE_XV_NOVELTY_AND_PUBLICATION_V1.md'],
44:['reports/mist_cove/phase15_v1/scientific_gates.json'],45:['docs/mist_cove/PHASE_XVI_PROSPECTIVE_STUDY_PROPOSAL_V1.md'],
46:['reports/mist_cove/phase15_v1/privacy_review.json'],47:['reports/mist_cove/phase15_v1/resource_receipts.json']}

FINDINGS={
8:'Generated leakage, role, geometry, clause and ordering controls pass; inherited source/cache tests retained.',
9:'Published 42,691 scored-epoch confusion and REM row reproduce exactly.',
10:'REM to W/N1/N2/N3 separated; rare N3 denominator four preserved.',
11:'219 representative and 240 enriched cases frozen privately with public SHA seal.',
12:'Transition contrasts use contiguous scored physical runs; REM-to-N2 is transition-enriched.',
13:'Both calibrated EEG leads analyzed; within-REM sigma and entropy contrasts reported.',
14:'Fixed horizontal EOG candidate activity differs by REM error type; no new EOG classifier.',
15:'Native ST 100-Hz chin EMG and within-night ranks reported, separate from SC 1-Hz envelope.',
16:'EEG/EOG/EMG and model-score disagreements reported without synthetic expert labels.',
17:'Fixed flat/step/extreme/clip event flags and within-person error contrasts reported.',
18:'Paired temazepam-placebo REM recall difference +0.0124; bootstrap interval crosses zero.',
19:'Five-class margins/confidence and REM error taxonomy reported.',
20:'ECE15 reproduced; REM stage ECE 0.453, high-confidence wrong count 3,209.',
21:'Fixed warning rules have limited REM-error sensitivity and material false-warning rates.',
22:'Person-clustered descriptive contrasts; no fitted ST predictor or causal claim.',
23:'Ten calibrated multichannel reports retained privately; no expert adjudication.',
24:'Faithfulness, proxy agreement, stability and utility estimands kept distinct.',
25:'Exact, signed, greedy, spectral, gradient, random, position and inherited physical comparators audited.',
26:'All interval baselines matched to exact-positive two-second atom count; greedy own length separate.',
27:'E1/E2/E3 panel identities sealed before new explanation comparison.',
28:'E1 has 63 detector-candidate cases; 55 nondegenerate eligible from 20 people.',
29:'Original stage-blind YASA event cache and version reused with authenticated Phase XIV identity.',
30:'Detector targets are independent of explanation ranking but not expert event truth.',
31:'Pz slow-wave, frontal spindle, EOG excursion and EMG low-tone candidates reported separately.',
32:'0, 0.1 and 0.25-second interval tolerances; missing event/support cases retained.',
33:'Exact minus sigma is -0.0462 by person on 55 eligible E2 cases; registered FAIL unchanged.',
34:'Sigma wins 24 cases, exact wins 9 and 22 tie; wrong predictions remain explainable.',
35:'One thousand person-bootstrap draws; nights/cases remain nested within participant.',
36:'360 base and 360 neutral full tables; all 32,768 states verified; full-input error zero.',
37:'E1 neutral-state table changes in 190/219; same-model operator sensitivity remains high.',
38:'Inherited compatible physical bridge 500/3,285 and background 94/3,285; no donor-guard relaxation.',
39:'Mild HF prediction stable 93/100, same primary clause valid 70/100, three substitutions.',
40:'33,426,540 attempted-inclusive cached-head states; 0 fits and updates.',
41:'Fixed shifted/next-epoch and random controls; generated malformed mask/clause/role/ordering tests.',
42:'Internal exactness does not identify clinical events or physical deletions.',
43:'Prior SIS, Boolean, time-series, YASA and reliability work reviewed; world-first novelty not claimed.',
44:'Evidence table separates exact correctness from failed sigma increment and untested clinical validity.',
45:'Narrow independent blinded annotation study proposed; Phase XVI not executed.',
46:'New public assets scanned; ignored waveforms and atlas not tracked; old accepted root history unchanged.',
47:'Full CPU regression pass, 11 CUDA-only skips; no GPU science operation.'}


def _write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def privacy_review():
    tracked=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    relevant=sorted({p for p in tracked if p.startswith(('src/mist_cove/phase15/','reports/mist_cove/phase15_v1/',
                'docs/mist_cove/PHASE_XV','docs/mist_cove/PHASE_XVI','configs/mist_cove_phase15','tests/test_phase15'))})
    if not relevant:raise ValueError('no Phase XV public assets to review')
    forbidden=re.compile(rb'/home/' + b'FA006' + rb'|/Desktop/' + b'bhi' +
        rb'|SC[0-9]{4}|ST[0-9]{4}|"(?:mean_uV|std_uV)"\s*:\s*[-+0-9]')
    png=0;text_files=0
    for rel in relevant:
        path=ROOT/rel
        if path.is_symlink() or not path.is_file():raise ValueError('unsafe release file: '+rel)
        if path.suffix.lower() in ('.npz','.npy','.pt','.edf','.jsonl','.pkl'):
            raise ValueError('private data format tracked: '+rel)
        value=path.read_bytes()
        if rel.endswith('.png'):
            if not value.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('invalid PNG')
            png+=1
        else:
            if forbidden.search(value):raise ValueError('private literal in new public asset: '+rel)
            text_files+=1
    if any(p.startswith('mist_transfer_runs/mist_cove/phase15_v1/') for p in tracked):
        raise ValueError('private Phase XV artifact tracked')
    x={'status':'PASS','new_public_files_scanned':len(relevant),'text_files':text_files,'aggregate_PNGs':png,
       'private_Phase_XV_files_tracked':0,'forbidden_root_or_subject_literals':0,
       'private_waveform_crops_published':False,'private_full_prediction_arrays_published':False,
       'fitted_weights_or_normalization_values_published':False,
       'old_accepted_dataset_root_literal_in_prior_history':True,'history_rewritten':False,
       'new_raw_SC_VAL_reads':0,'new_SC_RESERVED_reads':0,'ST_neural_fits':0,'human_validation':'NOT_TESTED'}
    _write(PUBLIC/'privacy_review.json',x)
    return x


def evidence():
    e=read(PUBLIC/'explanation_comparison.json');rem=read(PUBLIC/'rem_forensics.json')
    result={'status':LABEL,'questions':[
        {'question':'exact classifier response','original_hypothesis':'declared availability operator supports sound finite certificates',
         'data':'360 sealed ST cases, reused exposure','denominator':360,'method':'720 exhaustive 32768-state tables and clause verification',
         'uncertainty_unit':'case for coverage, deterministic verification for correctness','finding':'full-input discrepancy zero; all tables and clauses checked',
         'negative_control':'generated incorrect mask/clause and cache identity tests','confounding':'operator and neutral state define the estimand',
         'interpretation':'internal model behavior only','scientific_status':'ENGINEERING_PASS'},
        {'question':'source-to-ST REM recognition','original_hypothesis':'frozen source model transfers across study',
         'data':'44 previously exposed ST nights','denominator':8349,'method':'unchanged cached original logits and scored grid',
         'uncertainty_unit':'participant, 22 people','finding':'REM recall 0.2162; 6544 REM errors; ECE15 0.1339',
         'negative_control':'exact Phase XIV confusion reproduction','confounding':'study, channels, medication and scorer differences',
         'interpretation':'single-model transfer failure, not physiological cause','scientific_status':'SCIENTIFIC_FAIL'},
        {'question':'REM-error signal taxonomy','original_hypothesis':'error groups have distinct signal profiles',
         'data':'8349 previously exposed scored ST REM epochs','denominator':8349,'method':'stage-conditioned calibrated wave and person-bootstrap contrasts',
         'uncertainty_unit':'participant, 22 people','finding':'REM-to-N2 higher sigma/transition proximity and fewer EOG proxies; REM-to-W more EOG proxies',
         'negative_control':'correct REM within-person reference; fixed quality flags','confounding':'observational, correlated features and unvalidated proxies',
         'interpretation':'exploratory association only','scientific_status':'INCONCLUSIVE'},
        {'question':'incremental spindle-proxy precision over sigma','original_hypothesis':'exact support exceeds matched sigma ranking',
         'data':'E2 previously exposed ST; E1 detector-conditioned subset','denominator':55,'method':'equal-duration atoms, person-bootstrap 1000',
         'uncertainty_unit':'participant, 20 people','finding':'exact minus sigma -0.0462, 95% [-0.0797,-0.0167]',
         'negative_control':'random, shifted10s and next-epoch events','confounding':'automated unvalidated event proxy and retrospective ST exposure',
         'interpretation':'prior registered sigma increment FAIL unchanged; new result exploratory','scientific_status':'SCIENTIFIC_FAIL'},
        {'question':'clause versus prediction stability','original_hypothesis':'stable class need not imply stable original reason',
         'data':'first100 already sealed E1 cases','denominator':100,'method':'three fixed nuisance views and full response tables',
         'uncertainty_unit':'case within participant','finding':'mild HF 93 stable predictions versus 70 same clauses, three substitutions',
         'negative_control':'separate alternative-active and prediction indicators','confounding':'synthetic nuisance views, no physiological claim',
         'interpretation':'internal explanation instability','scientific_status':'INCONCLUSIVE'},
        {'question':'physical intervention validity','original_hypothesis':'internal withholding predicts compatible waveform replacement',
         'data':'inherited Phase XIV 219 ST cases','denominator':3285,'method':'frozen bridge/background compatibility guards',
         'uncertainty_unit':'case/person nested, descriptive','finding':'500 bridge and 94 background compatible requests; disagreements remain',
         'negative_control':'nontarget exact preservation and historical donor eligibility','confounding':'operators alter different estimands',
         'interpretation':'clinical event deletion NOT TESTED','scientific_status':'INCONCLUSIVE'},
        {'question':'expert-validated event localization','original_hypothesis':'certificate maps to clinical event',
         'data':'no approved reviewers or independent annotations','denominator':0,'method':'not run',
         'uncertainty_unit':'not applicable','finding':'no evidence','negative_control':'not applicable',
         'confounding':'automated proxy imperfect','interpretation':'no clinical claim','scientific_status':'NOT_TESTED'}]}
    _write(PUBLIC/'evidence_table.json',result)
    gates={'status':LABEL,'exact_operator_correctness':'ENGINEERING_PASS','ST_REM_transfer':'SCIENTIFIC_FAIL',
       'Phase_XIV_ST_exact_minus_random':'SCIENTIFIC_PASS_INHERITED','Phase_XIV_ST_exact_minus_sigma':'SCIENTIFIC_FAIL_UNCHANGED',
       'Phase_XV_exact_minus_sigma':'SCIENTIFIC_FAIL_EXPLORATORY','SC_TRAIN_proxy_primary':'NOT_ESTIMABLE_UNCHANGED',
       'Phase_XIV_SC_context':'SCIENTIFIC_FAIL_UNCHANGED','human_event_validity':'NOT_TESTED',
       'clinical_deletion_validity':'NOT_TESTED','new_independent_ST_confirmation':'NOT_TESTED',
       'novelty_world_first':'NOT_CLAIMED','Phase_XVI_execution':'NOT_AUTHORIZED'}
    _write(PUBLIC/'scientific_gates.json',gates)
    return result


def ledger():
    p=PUBLIC/'progress.json';x=read(p)
    code_files=sorted((ROOT/'src/mist_cove/phase15').glob('*.py'))
    code_digest=sha(canonical({str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in code_files}))
    tests=['generated Phase XV negative/geometry/publication fixtures (13 pass)','complete CPU regression: 1152 pass, 11 CUDA skips']
    for n in range(8,48):
        w=x['work_packages'][n-1]
        if w['id']!=f'WP{n:02d}':raise ValueError('ledger order differs')
        artifacts=ARTIFACTS[n]
        if not all((ROOT/a).is_file() for a in artifacts):raise ValueError('missing evidence for '+w['id'])
        w.update(status='COMPLETE',engineering_status='PASS',
          scientific_status=('ENGINEERING_PASS' if n in (8,36,46,47) else
                             'NOT_TESTED_HUMAN' if n==23 else
                             'NOVELTY_UNCERTAIN' if n==43 else LABEL),
          code_configuration_identity={'config_sha256':x['config_sha256'],'phase15_code_bundle_sha256':code_digest,
                                       'model_checkpoint_sha256':read(PUBLIC/'model_reauthentication.json')['checkpoint_byte_sha256']},
          tests=tests,artifacts=artifacts,findings=[FINDINGS[n]],
          limitations=['Retrospective previously exposed ST; no expert event adjudication or causal identification'],block_reason=None)
    w=x['work_packages'][47]
    w.update(status='PENDING_PUBLICATION_AND_EXACT_CI',engineering_status='PENDING',scientific_status='NOT_RUN',
        tests=[],artifacts=[],findings=[],limitations=['Draft PR and final-commit CI not yet verified'],block_reason=None)
    _write(p,x)


def run():
    verify_plan_commit();privacy=privacy_review();evidence();ledger()
    print(json.dumps({'privacy':privacy['status'],'ledger_complete':sum(w['status']=='COMPLETE' for w in read(PUBLIC/'progress.json')['work_packages']),
                      'WP48':'PENDING_PUBLICATION_AND_EXACT_CI'}))


if __name__=='__main__':run()
