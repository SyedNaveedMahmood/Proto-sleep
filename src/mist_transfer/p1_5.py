"""P1.5 blocker resolution from immutable P1 proofs and small annotation files."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import mne
import pyedflib
import pandas as pd
from .annotation_compat import paired_annotations, read_annotation_profile, fixed_header
from .audit import discover
from .integrity import STAGES, index_runs, validate_indices, annotation_grid
from .provenance import AuditRun, atomic_json, atomic_csv, atomic_text, fingerprint, sha256_file


def verify_reuse(prior, proof_path):
    """Revalidate hashes once; reuse expensive waveform checks only on identical inputs.

    A locally generated proof permits later invocations to reuse current digests
    only while device/inode/size/mtime/ctime remain unchanged. It is not an
    adversarial tamper-proof cache and must remain local/trusted.
    """
    expected = prior['provenance']['source_recording_hashes']
    _, current = discover(prior['provenance']['config'])
    if set(current) != set(expected):
        raise ValueError('source inventory changed; new P1 audit required')
    old = json.loads(proof_path.read_text()) if proof_path.exists() else {}
    cache = old.get('files', {}) if old.get('prior_signature') == prior['signature'] else {}
    results, reads = {}, 0
    for name,digest in sorted(expected.items()):
        p = Path(name); stat = p.stat()
        signature = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        previous = cache.get(name,{})
        if previous.get('stat') == signature and previous.get('sha256') == digest:
            actual = digest
        else:
            actual = sha256_file(p); reads += 1
        if actual != digest:
            raise ValueError('source hash changed; cached waveform proof invalid: '+name)
        results[name] = {'stat':signature,'sha256':actual}
    atomic_json({'prior_signature':prior['signature'],'files':results},proof_path)
    return {'status':'VERIFIED','input_count':len(results),'files_rehashed':reads,
            'waveform_realignments':0,'npz_arrays_reopened':0,
            'verified_recording_hashes':dict(expected), 'proof_sha256':sha256_file(proof_path)}


def recovered_indices(row):
    runs = row['original_epoch_runs']
    cursor, pieces = 0, []
    for run in runs:
        if run['array_start'] != cursor or run['array_stop']-cursor != run['original_stop']-run['original_start']:
            raise ValueError('invalid cached epoch run topology')
        pieces.append(np.arange(run['original_start'],run['original_stop'],dtype=np.int64))
        cursor = run['array_stop']
    idx = validate_indices(np.concatenate(pieces) if pieces else [], row['n_epochs'])
    digest = hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()
    if digest != row['original_epoch_index_sha256']:
        raise ValueError('cached recovered epoch digest mismatch')
    return idx


def selection_analysis(labels, indices):
    """Describe measured inclusion and test candidate rules; never certify lineage."""
    n = len(labels); idx = validate_indices(indices,len(indices))
    if len(idx) and idx[-1] >= n:
        raise ValueError('retained original index outside recording')
    keep = np.zeros(n,dtype=bool); keep[idx] = True
    if np.any(labels[idx] < 0):
        raise ValueError('NPZ retains movement/unknown/unannotated epoch')
    scored = np.flatnonzero(labels >= 0)
    nonwake = np.flatnonzero(labels[scored] > 0)
    candidates = {'all_scored_epochs':scored}
    if len(nonwake):
        lo,hi = max(0,int(nonwake[0])-60), min(len(scored),int(nonwake[-1])+61)
        candidates['remove_unscored_then_60_scored_epochs_each_edge'] = scored[lo:hi]
        physical = np.flatnonzero(labels > 0)
        candidates['60_original_epochs_each_edge_then_remove_unscored'] = np.flatnonzero(
            (labels>=0) & (np.arange(n)>=max(0,int(physical[0])-60)) &
            (np.arange(n)<=min(n-1,int(physical[-1])+60)))
    counts = {name:{'retained':int(np.sum(keep & (labels==i))),
                    'excluded':int(np.sum(~keep & (labels==i)))} for i,name in enumerate(STAGES)}
    counts['unscored'] = {'retained':int(np.sum(keep & (labels<0))), 'excluded':int(np.sum(~keep & (labels<0)))}
    matches = {name:bool(np.array_equal(idx,want)) for name,want in candidates.items()}
    return {'observed_mask_status':'VERIFIED','counts_by_stage':counts,
            'retained_original_epoch_runs':index_runs(idx),
            'excluded_original_epoch_runs':index_runs(np.flatnonzero(~keep)),
            'candidate_rule_matches':matches,
            'label_dependent_candidate_fit':any(v for k,v in matches.items() if k!='all_scored_epochs'),
            'label_dependence_causal_status':'UNKNOWN; exact match to label-conditioned rule is evidence of compatibility, not execution lineage',
            'exact_upstream_recipe_status':'UNKNOWN',
            'zero_shot_evaluation_use':'BLOCKED; retained mask is label-conditioned-compatible; rebuild from raw'}


def manifest_analysis(config, archive_path=None):
    root = Path(config['sleep_edf_78_npz']); path = root/'MANIFEST.TXT'
    listed = {}
    for line in path.read_text().splitlines():
        m = re.fullmatch(r'([^ ]+\.npz) \(application/octet-stream\) (\d+) bytes\.',line)
        if not m or m[1] in listed:
            raise ValueError('invalid or duplicate distribution manifest entry')
        listed[m[1]] = int(m[2])
    actual = {p.name:p.stat().st_size for p in root.glob('*.npz')}
    missing = sorted(set(listed)-set(actual))
    size_conflicts = [k for k in set(listed)&set(actual) if listed[k] != actual[k]]
    if missing or size_conflicts:
        raise ValueError('listed manifest files/sizes disagree')
    result = {'coverage_status':'VERIFIED_PARTIAL_INDEX','listed_files':len(listed),'actual_files':len(actual),
              'unlisted_files':sorted(set(actual)-set(listed)), 'missing_listed_files':missing,'size_conflicts':size_conflicts,
              'scope_intent_status':'UNKNOWN; no author statement or generation log available',
              'manifest_sha256':sha256_file(path),
              'data_corruption_evidence':False,'source_traceability':'153 NPZ files previously waveform-aligned; re-used only after hash validation'}
    if archive_path and Path(archive_path).is_file():
        with zipfile.ZipFile(archive_path) as z:
            infos = z.infolist(); names=[i.filename for i in infos]
            if len(names)!=len(set(names)):
                raise ValueError('duplicate archive member identity')
            name = root.name+'/MANIFEST.TXT'
            member = z.getinfo(name)
            if member.file_size > 100000:
                raise ValueError('oversized archive manifest')
            data = z.read(name)  # checks member CRC; never extract patient NPZ payloads
            members = {Path(i.filename).name:i.file_size for i in infos if i.filename.startswith(root.name+'/') and i.filename.endswith('.npz')}
            result['archive_evidence'] = {'path':str(Path(archive_path).resolve()),
                'member_count':len(infos),'edf78_npz_members':len(members),
                'npz_names_and_uncompressed_sizes_equal_directory':members==actual,
                'manifest_member_identical':data==path.read_bytes(),
                'toc_sha256':fingerprint([(i.filename,i.file_size,i.compress_size,i.CRC) for i in infos]),
                'manifest_member_sha256':hashlib.sha256(data).hexdigest(),
                'archive_payloads_extracted':False,'archive_npz_content_crc_checked':False}
            if not result['archive_evidence']['manifest_member_identical'] or members != actual:
                raise ValueError('adjacent archive does not corroborate directory index/coverage')
    return result


INFERENCE_POLICY = {
    'status':'PRESPECIFIED_FOR_REVIEW; specification only, no P2 pipeline executed',
    'channel':'calibrated EEG Fpz-Cz; Pz-Oz only as separately declared montage sensitivity scope',
    'input_selection':'all complete physical 30-second epochs of each recording; signal duration only',
    'timing':'retain original_epoch_index and start_seconds = 30 * index plus original fixed clock; no concatenation across records',
    'partial_tail':'record actual trailing duration/samples; exclude incomplete 30-second tail solely by signal geometry',
    'annotation_access':'inference epoch enumeration, signal loading and normalization do not consult stage labels',
    'scoring_mask':'attach W/1/2/3+4/R labels after input enumeration; unknown/movement/unannotated are unscored, not removed from inference',
    'gaps':'retain scoring/exclusion reasons and original positions; absent signal/discontinuous recording breaks context; scoring-mask gaps do not compress time',
    'units':'preserve native EEG calibration, fs, units, montage; any conversion/normalization explicit and source-TRAIN fitted only',
    'identity':'canonical study/person/night/full PSG ID; all nights of person stay in one split; source/annotation SHA256 recorded',
    'st_clock_policy':'same fixed PSG/hyp clock + patient/recording fields; explicit previous-day Recordingfield exception only after strict TAL/MNE agreement',
    'prohibited':'no sleep-onset/last-sleep bounds, label-dependent Wake trimming, target-fitted scaling or inferred continuity',
    'p2_acceptance_required':'synthetic fixtures, differential checks and ten participant-night audits before any model training'}


def scope_gates(prior, resolutions, reuse_verified=True):
    raw = {study:[r for r in resolutions if r['study']==study] for study in ['SC','ST']}
    # A questionable night excludes every night of that person from the proposed
    # smallest scope. Annotation decoding and person eligibility are separate.
    excluded = {r.get('subject_id',r['recording_id']) for r in raw['SC'] if r['status']!='VERIFIED'}
    eligible = [r for r in raw['SC'] if r['status']=='VERIFIED' and r.get('subject_id',r['recording_id']) not in excluded]
    sc = reuse_verified and len(raw['SC'])==prior['inventory']['raw_sc']['recordings'] and bool(eligible)
    st = reuse_verified and len(raw['ST'])==prior['inventory']['raw_st']['recordings'] and all(r['status']=='VERIFIED' for r in raw['ST'])
    for row in raw['SC']:
        row['proposed_scope_eligibility']='BLOCKED' if row.get('subject_id',row['recording_id']) in excluded else 'VERIFIED'
    for row in raw['ST']:
        row['proposed_scope_eligibility']='CONDITIONAL' if row['status']=='VERIFIED' else 'BLOCKED'
    extension=set(prior['subject_disjoint_partitions']['edf78_sc_extension_subjects'])-excluded
    return {
      'status_vocabulary':{'VERIFIED':'directly validated evidence; scoped to stated measurements',
                           'CONDITIONAL':'permitted scope only with listed prerequisites',
                           'BLOCKED':'must not proceed', 'UNKNOWN':'evidence insufficient; no assumption substituted'},
      'sc_within_cohort':{'data_status':'VERIFIED' if sc else 'BLOCKED',
          'p2_eligibility':'VERIFIED' if sc else 'BLOCKED', 'scope':'raw SC person-quarantined subset only',
          'full_78_person_scope_status':'BLOCKED' if excluded else 'VERIFIED',
          'quarantined_subjects':sorted(excluded),
          'eligible_subjects':sorted({r.get('subject_id',r['recording_id']) for r in eligible}),
          'eligible_recordings':sorted(r['recording_id'] for r in eligible),
          'experiment_status':'CONDITIONAL' if sc else 'BLOCKED',
          'conditions':['apply prespecified full-recording label-independent inference policy',
                        'EDF20/78 is same cohort; remove all shared subjects/nights from extension',
                        'new subject-grouped split guards and P2 acceptance tests before experiments',
                        'quarantine every night of SC people with unresolved pair demographic disagreement; no demographic adjudication from labels'],
          'existing_npz_zero_shot_eligibility':'BLOCKED',
          'eligible_extension_subjects':sorted(extension)},
      'sc_to_st_proxy':{'data_status':'VERIFIED' if st else 'CONDITIONAL' if any(r['status']=='VERIFIED' for r in raw['ST']) else 'BLOCKED',
          'p2_eligibility':'CONDITIONAL' if st and sc else 'BLOCKED',
          'experiment_status':'CONDITIONAL' if st and sc else 'BLOCKED',
          'conditions':['include explicit validated annotation compatibility path and preserve original malformed header evidence',
                        'review/accept fixed-clock policy; never shift TAL onsets by 24 hours',
                        'discard incomplete signal tail by geometry, preserve all original annotation overhang diagnostics',
                        'freeze ST development/locked-evaluation role before any score; keep participant nights grouped',
                        'report placebo/temazepam, setting/population confounds and unknown cross-study person linkage',
                        'complete independent ST P2 fixture and sample differential checks'],
          'cross_study_person_linkage':'UNKNOWN', 'independent_external_cohort_claim':'BLOCKED',
          'quarantined_recordings':[r['recording_id'] for r in raw['ST'] if r['status']!='VERIFIED']},
      'shhs_external':{'data_status':'UNKNOWN','access_authorization':'UNKNOWN','p2_eligibility':'BLOCKED',
          'experiment_status':'BLOCKED','data_opened':False,
          'conditions':['confirmed authorized access','usable paired EDF/XML and participant/visit identifiers',
                        'separate unit/montage/stage/alignment audit','frozen source-only selection and locked target subject split']},
      'recommendation':'SC_RAW_PERSON_QUARANTINED_SUBSET_FOR_NEXT_P2_REVIEW' if sc else 'NO_PREPROCESSING',
      'p2_executed':False,'training_performed':False}


def resolve(prior_path, output, archive_path=None, seed=123, resume=False, reference_paths=()):
    prior_path=Path(prior_path); output=Path(output)
    prior=json.loads(prior_path.read_text())
    if prior['provenance']['git']['sha'] != '3f770c6c65c107eb171425142eefce91dd107bf0':
        raise ValueError('unexpected historical P1 implementation revision')
    print('Revalidating input hashes; no waveform re-alignment or NPZ array opening.',flush=True)
    proof=Path('mist_transfer_runs/p1_5_input_verification.json')
    reuse=verify_reuse(prior,proof)
    print(f'Input proofs: {reuse["input_count"]}; hashes read: {reuse["files_rehashed"]}; waveform checks reused.',flush=True)
    manifest=manifest_analysis(prior['provenance']['config'],archive_path)
    references=[{'path':str(Path(p).resolve()),'sha256':sha256_file(Path(p)),
                 'role':'candidate preprocessing reference only; UNKNOWN execution lineage; never executed'} for p in reference_paths]
    inputs={**reuse['verified_recording_hashes'],str(prior_path.resolve()):sha256_file(prior_path)}
    # Archive evidence is limited to CRC-checked manifest + central directory,
    # not an invented whole-archive/NPZ content verification.
    if manifest.get('archive_evidence'):
        inputs['archive_toc:'+str(Path(archive_path).resolve())]=manifest['archive_evidence']['toc_sha256']
    inputs.update({r['path']:r['sha256'] for r in references})
    run=AuditRun(output,{'phase':'P1.5','historical_audit':str(prior_path.resolve()),
                         'inference_policy':INFERENCE_POLICY},inputs,seed,resume)
    resolutions, selections, labels_by_record, errors = [],[],{},[]
    sc_table=pd.read_excel(Path(prior['provenance']['config']['sleep_edfx_original_root'])/'SC-subjects.xls')
    sc_demo={(f'SC:{int(row[0]):02d}',int(row[1])):{'age':int(row[2]),'sex':'F' if int(row[3])==1 else 'M'} for row in sc_table[['subject','night','age','sex (F=1)']].itertuples(index=False,name=None)}
    for row in [r for r in prior['records'] if r['dataset'].startswith('raw_')]:
        result={k:row[k] for k in ['study','subject_id','night','recording_id','path','source_sha256','annotation_source','annotation_sha256']}
        result['historical_p1_status']=row['indices_status']
        try:
            labels,detail=paired_annotations(row['annotation_source'],row['path'],row['duration_seconds'])
            counts={name:int(np.sum(labels==i)) for i,name in enumerate(STAGES)}
            # Existing successful P1 grids must retain exactly the same labels/counts.
            if row['indices_status']=='original_raw_grid' and (counts!=row['mapped_stage_counts'] or detail['raw_stage_counts']!=row['raw_stage_counts']):
                raise ValueError('new reader disagrees with historical annotation grid')
            if row['study']=='ST' and row.get('treatment') not in {'placebo','temazepam'}:
                raise ValueError('ST treatment mapping unverified')
            result.update(status='VERIFIED',resolution='READ_ONLY_COMPATIBILITY' if detail['pyedflib_comparison']['error'] else 'READERS_AGREE',
                          validation=detail,mapped_stage_counts=counts,
                          treatment=row.get('treatment'),n_full_epochs=len(labels),
                          unscored_epochs=int(np.sum(labels<0)))
            labels_by_record[row['recording_id']]=labels
        except (ValueError,OSError,UnicodeError) as exc:
            result.update(status='BLOCKED',resolution='QUARANTINE',error=str(exc))
            # Characterize SC metadata conflicts without assigning the decoded
            # stages to an experimentally eligible person/night. This permits
            # diagnostic retention analysis of all cached NPZ masks.
            if row['study']=='SC':
                try:
                    h,start,arrays,detail=read_annotation_profile(row['annotation_source'])
                    p,pstart=fixed_header(row['path'])
                    ann=mne.read_annotations(row['annotation_source'])
                    with pyedflib.EdfReader(row['annotation_source']) as reader:
                        po,pd_,pt=reader.readAnnotations()
                    if start!=pstart or not all(np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(arrays,(ann.onset,ann.duration,ann.description),(po,pd_,pt))):
                        raise ValueError('quarantine diagnostic reader/clock disagreement')
                    labels,grid=annotation_grid(*arrays,row['duration_seconds'])
                    result['payload_status']='VERIFIED_DIAGNOSTIC_ONLY'
                    result['demographic_conflict']={'hypnogram_sex':h[8:88].decode().split()[1],
                        'psg_sex':p[8:88].decode().split()[1],
                        'spreadsheet':sc_demo[(row['subject_id'],row['night'])],
                        'cause':'UNKNOWN; no header repair or automatic identity adjudication',
                        'proposed_policy':'quarantine all nights of this SC subject'}
                    result['validation']={**detail,**grid}
                    labels_by_record[row['recording_id']]=labels
                except (ValueError,OSError,UnicodeError) as diagnostic:
                    result['diagnostic_error']=str(diagnostic)
            errors.append({'recording_id':row['recording_id'],'error':str(exc)})
        resolutions.append(result)
        if row['study']=='ST':
            print(f'{row["recording_id"]}: {result["status"]} {result["resolution"]}',flush=True)
    for row in [r for r in prior['records'] if r['dataset'].endswith('_npz')]:
        labels=labels_by_record.get(row['recording_id'])
        if labels is None:
            errors.append({'recording_id':row['recording_id'],'error':'cannot assess retention without validated raw annotations'})
            continue
        idx=recovered_indices(row);detail=selection_analysis(labels,idx)
        counts={name:detail['counts_by_stage'][name]['retained'] for name in STAGES}
        if counts != row['mapped_stage_counts']:
            raise ValueError('cached NPZ mapping/counts disagree with reconstructed labels')
        selections.append({**{k:row[k] for k in ['dataset','study','subject_id','night','recording_id','source_sha256']},**detail})
    # Cheap postflight detects changes while the annotation diagnostics ran.
    postflight=verify_reuse(prior,proof)
    if postflight['proof_sha256']!=reuse['proof_sha256']:
        raise ValueError('inputs changed during audit; review required')
    gates=scope_gates(prior,resolutions)
    outcome={'phase':'P1.5','historical_p1_gate':'FAIL (immutable historical report)',
             'status':'SCOPED_ELIGIBILITY' if not errors else 'SCOPED_QUARANTINE',
             'historical_audit_sha256':sha256_file(prior_path),'reuse_validation':reuse,
             'annotation_resolutions':resolutions,'manifest_resolution':manifest,
             'selection_analysis':selections,'candidate_reference_evidence':references,'scope_gates':gates,'inference_policy':INFERENCE_POLICY,
             'quarantine_policy':'every night of an SC person with unresolved pair metadata disagreement; no automated correction',
             'errors':errors,'scientific_unknowns':['exact Fpz-Cz NPZ preprocessing execution lineage',
                 'optional manifest author/generation intent','causal use of stage labels by the unknown original program',
                 'SC/ST person-level linkage','SHHS authorization and usable data'],
             'provenance':run.identity,'signature':run.signature}
    atomic_json(outcome,run.output/'revised_audit.json')
    atomic_json(gates,run.output/'gate_decisions.json')
    atomic_json(INFERENCE_POLICY,run.output/'inference_policy.json')
    atomic_json(selections,run.output/'selection_analysis.json')
    atomic_json(manifest,run.output/'manifest_resolution.json')
    atomic_csv([{k:r.get(k,'') for k in ['study','subject_id','night','recording_id','historical_p1_status','status','resolution','proposed_scope_eligibility','error']} for r in resolutions],
               ['study','subject_id','night','recording_id','historical_p1_status','status','resolution','proposed_scope_eligibility','error'],run.output/'per_recording_resolution.csv')
    flat=[]
    for r in selections:
        flat.append({**{k:r[k] for k in ['dataset','recording_id']},
                     **{f'{name}_{action}':r['counts_by_stage'][name][action] for name in STAGES+['unscored'] for action in ['retained','excluded']},
                     **r['candidate_rule_matches']})
    fields=['dataset','recording_id']+[f'{name}_{a}' for name in STAGES+['unscored'] for a in ['retained','excluded']]+[
        'all_scored_epochs','remove_unscored_then_60_scored_epochs_each_edge','60_original_epochs_each_edge_then_remove_unscored']
    atomic_csv(flat,fields,run.output/'selection_counts.csv')
    tally=Counter(r['resolution'] for r in resolutions)
    fit=Counter(k for r in selections for k,v in r['candidate_rule_matches'].items() if v)
    lines=['MIST-Transfer v3 P1.5; STOP FOR REVIEW',f'Input code SHA: {run.identity["git"]["sha"]}',
           f'Historical P1 FAIL remains immutable. Input hashes unchanged: {reuse["input_count"]}.',
           f'Waveform re-alignments: 0; NPZ arrays reopened: 0.',
           f'Annotation resolutions: {dict(tally)}; quarantines: {len(errors)}.',
           f'NPZ masks described: {len(selections)}; candidate matches: {dict(fit)}.',
           f'Manifest: {manifest["listed_files"]}/{manifest["actual_files"]}, {len(manifest["unlisted_files"])} unlisted; no listed size conflict.',
           'Partial bundled index is not a full cohort manifest; author intent UNKNOWN.',
           'Label-conditioned candidate fit does not prove upstream execution lineage.',
           'Prespecified evaluation policy: infer on every complete physical epoch before consulting labels; score annotated stages afterwards.',
           f'SC raw P2 eligibility: {gates["sc_within_cohort"]["p2_eligibility"]}',
           f'SC -> ST P2 eligibility: {gates["sc_to_st_proxy"]["p2_eligibility"]}',
           f'SHHS P2 eligibility: {gates["shhs_external"]["p2_eligibility"]}',
           'Existing trimmed NPZ zero-shot evaluation: BLOCKED.',
           'Recommended next scope: SC-only raw preprocessing after review; no P2 action taken.',
           f'SC eligible raw subset: {len(gates["sc_within_cohort"]["eligible_subjects"])} subjects / {len(gates["sc_within_cohort"]["eligible_recordings"])} recordings.',
           f'SC quarantined subjects: {gates["sc_within_cohort"]["quarantined_subjects"]}',
           'No models, preprocessing arrays, downloads, target metrics or SHHS access.',
           'NEXT SINGLE SAFE COMMAND: git status --short --branch']
    atomic_text('\n'.join(lines)+'\n',run.output/'FINAL_COPY_PASTE.txt')
    outputs=['revised_audit.json','gate_decisions.json','inference_policy.json','selection_analysis.json',
             'manifest_resolution.json','per_recording_resolution.csv','selection_counts.csv','FINAL_COPY_PASTE.txt']
    run.finish(outputs,outcome['status'])
    print('\n'.join(lines),flush=True)
    return 2 if errors else 0
