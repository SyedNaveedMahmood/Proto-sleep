"""Sequential independent scientific tracks under one cumulative CPU budget."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from torch import nn
from mist_cove.phase2.provenance import file_sha,digest,environment
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_cove.phase5.experiment import CachedHead
from mist_cove.phase5.runtime import Budget,predict,paired_summary
from mist_cove.phase6.generator import generate,compose,class_swap
from mist_cove.phase6.evaluation import location,selected
from mist_cove.phase7.prospective import encode,optimizer,steps,seal,restore
from mist_cove.phase8.provenance import trusted_reference
from mist_cove.phase8.generator import intervene
from mist_cove.phase8.calibration import fit_temperature
from mist_cove.phase8.evaluation import reliability as historical_reliability,paired_accuracy_interval
from mist_evidence.runtime import atomic_json,atomic_torch_save,tensor_digest
from .operators import internal_logits,response_table,margin,sample_query_masks,masks_numpy,component_contract
from .circuits import compile_robdd,nonmonotone_edges
from .clauses import extract_readable_clauses,greedy_sufficient_mask,verify_clause
from .objectives import paired_objective

ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'reports/mist_cove/phase9_v1'
PRIVATE=ROOT/'mist_transfer_runs/mist_cove/phase9_v1'
CONFIG=ROOT/'configs/mist_cove_phase9_v1.json'
PLAN_COMMIT='99ee93d4343970a948b6b29a064728b6c82d0ae0'
PLAN_BLOB='e40e646cb027de163d43220e47e51a491316bce2'


def seed_for(namespace,task,seed):
    return int.from_bytes(hashlib.sha256(f'phase9/{namespace}/{task}/{seed}'.encode()).digest()[:4],'little')


def config():return json.loads(CONFIG.read_text())


def reliability(logits,y,temperature=1.):
    # Undefined constant-confidence correlations are null, never a fabricated
    # zero or a reason to suppress a failed classification result.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='invalid value encountered in divide',category=RuntimeWarning)
        result=historical_reliability(logits,y,temperature)
    value=result.get('error_confidence_correlation')
    if value is not None and not np.isfinite(value):result['error_confidence_correlation']=None
    return result


def require_scope(scope):
    if scope!='SYNTHETIC_ONLY':raise ValueError('synthetic-only role required before filesystem access')


def output(path,*,scope):
    require_scope(scope);path=Path(path)
    if not path.resolve().is_relative_to(PRIVATE.resolve()) or path.resolve()==PRIVATE.resolve():raise ValueError('ignored Phase IX output required')
    from mist_cove.phase7.integrity import confined_file
    confined_file(ROOT,str(path.resolve().relative_to(ROOT)));path.mkdir(parents=True,exist_ok=False);return path


def plan_proof():
    if subprocess.check_output(['git','hash-object','next_steps.md'],cwd=ROOT,text=True).strip()!=PLAN_BLOB:raise ValueError('authoritative plan changed')
    subprocess.check_call(['git','merge-base','--is-ancestor',PLAN_COMMIT,'HEAD'],cwd=ROOT)
    return {'plan_commit':PLAN_COMMIT,'plan_blob':PLAN_BLOB}


def frozen():
    plan_proof();p=ROOT/'experiments/mist_cove/phase9_reporting_freeze_v2.json'
    if subprocess.check_output(['git','show',f'HEAD:{p.relative_to(ROOT)}'],cwd=ROOT)!=p.read_bytes():raise ValueError('preregistration must be committed')
    r=json.loads(p.read_text())
    for name,sha in r['files'].items():
        if file_sha(ROOT/name)!=sha:raise ValueError('source/config freeze changed: '+name)
    return {'preregistration_sha256':file_sha(p),'inventory_sha256':digest(r['files'])}


def update(wp,status,*,artifacts=None,tests=None,findings=None,block=None):
    p=PUBLIC/'progress.json';r=json.loads(p.read_text());v=r['work_packages'][wp-1]
    v['status']=status;v['code_configuration_identity']={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config_sha256':file_sha(CONFIG)}
    if artifacts is not None:v['produced_artifacts']=artifacts
    if tests is not None:v['tests_performed']=tests
    if findings is not None:v['scientific_findings']=findings
    v['block_reason']=block;atomic_json(r,p)


def subset(draw,indices):
    indices=torch.as_tensor(indices,dtype=torch.long)
    return replace(draw,**{name:getattr(draw,name)[indices] for name in ('x','y','base','background','noise','nuisance','artifact','events')},
        **{name:[getattr(draw,name)[i] for i in indices.tolist()] for name in ('intervals','nuisance_intervals','latents')})


def draw_for(role,task,seed,n,condition='clean'):
    return generate(n,seed_for(role,task,seed),task,condition)


def add_view(draw,task,seed,namespace,kind):
    other=generate(len(draw.y),seed_for(namespace,task,seed),task)
    corrupted=intervene(other,kind,1.);extra=corrupted.base-other.base
    return replace(draw,x=compose(draw.base+extra,draw.events),base=draw.base+extra,nuisance=draw.nuisance+extra,
        nuisance_intervals=[a+b[len(old):] for a,b,old in zip(draw.nuisance_intervals,corrupted.nuisance_intervals,other.nuisance_intervals)],
        condition='phase9_'+namespace)


def audit_draw(task,seed,c):
    d=draw_for('circuit_discovery',task,seed,c['audit_generation_examples'])
    indices=sorted(i for k in range(5) for i in (d.y==k).nonzero()[:c['audit_per_class'],0].tolist())
    return subset(d,indices),indices


def discovery_views(draw,task,seed):
    return [draw,add_view(draw,task,seed,'discovery_high_frequency','high_frequency'),add_view(draw,task,seed,'discovery_transient','transient')]


def verification_views(draw,task,seed):
    return [add_view(draw,task,seed,'circuit_verification/'+str(j),kind) for j,kind in enumerate(('high_frequency','transient','high_frequency'))]


def state_path(task,seed,method):return PRIVATE/'core_v1'/f'{task}_{seed}_{method}.pt'


def load_new(task,seed,method,record):
    m=make_model(config()['head_methods'].get(method,'M1'))
    if method.startswith('E'):m.encoder.requires_grad_(False)
    creation=record['creation'];p=state_path(task,seed,method)
    if json.loads(p.with_suffix('.creation.json').read_text())!=creation:raise ValueError('creation receipt mismatch')
    restore(p,creation,creation['identity'],m,optimizer(m,config()),torch.Generator());m.eval()
    return m


def saved_receipt(name):
    p=PRIVATE/'core_v1'/name
    return json.loads(p.read_text())


def cumulative_budget(c):
    paths=list((PRIVATE/'core_v1').glob('wp*_receipt.json'))+list((PRIVATE/'core_v1').glob('STOPPED_WP*.json'))
    spent=sum(json.loads(p.read_text()).get('wall_seconds',0.) for p in paths)
    return Budget(max(0.,c['scientific_wall_seconds']-spent))


def finish_track(wp,receipt,budget,findings):
    receipt['wall_seconds']=budget.elapsed;receipt['peak_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['status']='COMPLETE';receipt['frozen']=frozen();receipt['real_data_accessed']=False
    p=PRIVATE/'core_v1'/f'wp{wp:02}_receipt.json';atomic_json(receipt,p)
    update(wp,'COMPLETE',artifacts=[str(p.relative_to(ROOT))],findings=findings)
    return receipt


def identity(task,seed,method,train_digest):
    return {'experiment':'mist-wic-phase9-v1','task':task,'seed':seed,'method':method,'frozen':frozen(),
        'config_sha256':file_sha(CONFIG),'environment':environment('cpu'),'train_digest':train_digest}


def pilot(path):
    c=config();configure_precision('cpu');torch.set_num_threads(2);out=output(path,scope=c['scope']);rows=[]
    data=draw_for('pilot','single',123,5)
    for i,method in enumerate(('M1','M2')):
        torch.manual_seed(2700+i);m=make_model(method).eval();started=time.perf_counter();f,r=encode(m,data.x[i:i+1]);target=int(m(data.x[i:i+1]).argmax(1))
        margins,prob=response_table(m,f,r,target,batch_size=c['mask_batch']);truth=margins>c['margin_threshold']
        bdd=compile_robdd(truth);summary=extract_readable_clauses(truth,margins=margins)
        rows.append({'method':method,'seconds':time.perf_counter()-started,'mask_queries':32768,'encoded_examples':1,'cache_bytes':f.nelement()*f.element_size()+r.nelement()*r.element_size()})
    result={'scope':'TWO_GENERATED_DEVELOPMENT_EXAMPLES_RESOURCE_ONLY','rows':rows,
        'projected_exact_table_calls':1272,'projected_exact_seconds':max(v['seconds'] for v in rows)*1272,
        'conservative_training_and_other_seconds':600,'ceiling_seconds':3600,'scientific_outcomes_inspected':False}
    result['conservative_projection_seconds']=result['projected_exact_seconds']*2+600
    atomic_json(result,out/'pilot.json');print(json.dumps(result));return result


def training_pair(draw,task,seed,role='train_nuisance'):
    hf=add_view(draw,task,seed,role+'/hf','high_frequency');art=add_view(draw,task,seed,role+'/artifact','transient')
    return torch.where((torch.arange(len(draw.y))%2==0)[:,None,None],hf.x,art.x)


@torch.no_grad()
def calibration(model,task,seed,c,budget):
    d=draw_for('calibration',task,seed,c['calibration_examples'])
    x=training_pair(d,task,seed,'calibration_nuisance')
    mixed=torch.where((torch.arange(len(d.y))%2==0)[:,None,None],d.x,x)
    logits=predict(model,mixed,budget)
    return {'temperature':fit_temperature(logits,d.y,role='SYNTHETIC_CALIBRATION'),
        'split':'phase9/calibration','seed':seed_for('calibration',task,seed),'examples':len(d.y),
        'logits_sha256':tensor_digest({'calibration_logits':logits,'calibration_y':d.y})}


def atom_localization(bits,draw,index):
    intervals=[(j*200,(j+1)*200) for j in range(15) if bits&(1<<j)]
    return location(intervals,draw.intervals[index],draw.nuisance_intervals[index])


def candidate_bits(indices):
    bits=0
    for j in indices:bits|=(1<<int(j))|(1<<(int(j)+1))
    return bits


@torch.no_grad()
def analyze_predictor(model,task,seed,name,c,budget,cal):
    """Discover on three views; unchanged-clause verification is deferred to WP10."""
    draw,original_indices=audit_draw(task,seed,c);views=discovery_views(draw,task,seed)
    checkpoint_sha=(trusted_reference(task,seed,'P0')[1]['checkpoint_sha256'] if name=='P0' else file_sha(state_path(task,seed,name)))
    details=PRIVATE/'core_v1'/f'{task}_{seed}_{name}_circuits.pt';prior_archive=None;prior_archive_sha=None
    if details.exists():
        recovery=json.loads((ROOT/'experiments/mist_cove/phase9_reporting_freeze_v2.json').read_text())['recovery']
        if (task,seed,name)!=('single',123,'E0') or file_sha(details)!=recovery['circuits_sha256']:
            raise FileExistsError('preserve existing unregistered circuit archive')
        from mist_evidence.runtime import load_trusted
        prior_archive=load_trusted(details);prior_archive_sha=file_sha(details)
        details=details.with_name(details.stem+'_continued_v2.pt')
        if details.exists():raise FileExistsError('preserve continued circuit archive')
    encoded=[encode(model,v.x) for v in views];native_logits=model(draw.x);rows=[];archives=[];exact_queries=0
    for i in range(len(draw.y)):
        budget.check();f,r=encoded[0][0][i:i+1],encoded[0][1][i:i+1]
        original=model.decision(f,r)[0];native=native_logits[i:i+1];target=int(native.argmax(1))
        if not torch.allclose(native,original,rtol=0,atol=c['full_identity_atol']):raise ValueError('native/cached full identity')
        tables=[];probs=[];circuits=[]
        for ef,er in encoded:
            margins,p=response_table(model,ef[i:i+1],er[i:i+1],target,batch_size=c['mask_batch'],temperature=cal['temperature'],budget=budget)
            tables.append(margins);probs.append(p);circuits.append(compile_robdd(margins>c['margin_threshold']));exact_queries+=32768
        tables=np.stack(tables);probs=np.stack(probs);truth=tables>c['margin_threshold']
        if prior_archive is not None:
            if not np.array_equal(tables,prior_archive[i]['margins']) or [v['sha256'] for v in circuits]!=[v['sha256'] for v in prior_archive[i]['circuits']]:
                raise ValueError('preserved completed E0 response changed during reporting recovery')
        clean=extract_readable_clauses(truth[0],margins=tables[0]);signed=extract_readable_clauses(truth,margins=tables)
        positive=extract_readable_clauses(truth,positive_only=True,margins=tables)
        greedy=greedy_sufficient_mask(truth[0])
        ranking=model.conventional_ranking(f,r) if model.cfg.method=='M1' else model.decision(f,r)[-1]
        ix=ranking.argsort(dim=1,descending=True,stable=True)[0,:4];native_bits=candidate_bits(ix)
        rms_bits=candidate_bits(r[0].argsort(descending=True,stable=True)[:4])
        gen=torch.Generator().manual_seed(seed_for('random_intervals/'+str(i),task,seed));random_bits=candidate_bits(torch.randperm(14,generator=gen)[:4])
        for summary in (clean,signed,positive):
            for clause in summary['clauses']:
                clause['localization']=atom_localization(clause['positive_bits'],draw,i)
                neg=[j for j in range(15) if clause['negative_bits']&(1<<j)]
                clause['negative_nuisance_overlap_fraction']=float(np.mean([any(j*200<b and j*200+200>a for a,b in draw.nuisance_intervals[i]) for j in neg])) if neg else None
        compact=any(v['active_at_original_full_mask'] and not clean['prior_only'] and v['retained_seconds']<=c['max_retained_seconds'] and v['signed_literals']<=c['max_summary_literals'] for v in clean['clauses'])
        neutral_margins,_=response_table(model,f,r,target,batch_size=c['mask_batch'],neutral=c['neutral_control'],budget=budget)
        exact_queries+=32768
        record={'task':task,'seed':seed,'predictor':name,'generation_index':original_indices[i],
            'target_class':target,'original_prediction':target,'correct':target==int(draw.y[i]),
            'input_sha256':tensor_digest({'generated_input':draw.x[i:i+1]}),'model_sha256':tensor_digest(model.state_dict()),
            'checkpoint_sha256':checkpoint_sha,
            'operator_sha256':file_sha(Path(__file__).with_name('operators.py')),'config_sha256':file_sha(CONFIG),
            'geometry':{'atoms':15,'samples_per_atom':200,'candidate_support':'adjacent atom pairs','candidate_count':14},
            'neutral_state':'zero local features and RMS; native mean denominator14','threshold':c['margin_threshold'],
            'calibration':cal,'mask_search_size':32768,'exact':True,'discovery_views':['base','added_HF','added_transient'],
            'circuits':[{'sha256':v['sha256'],'nonterminal_nodes':len(v['nodes'])-2,'variable_order':v['variable_order'],'exact_equality':True} for v in circuits],
            'clean_summary':clean,'signed_summary':signed,'positive_summary':positive,'greedy':greedy,
            'compact_active_clean_found':compact,'full_input_discovery_stable':bool(truth[:,-1].all()),
            'native':{'mask':native_bits,'canonical_response':bool(truth[0,native_bits]),'universal_valid':verify_clause(truth,native_bits,0),'retained_seconds':native_bits.bit_count()*2,'localization':atom_localization(native_bits,draw,i)},
            'rms':{'mask':rms_bits,'localization':atom_localization(rms_bits,draw,i)},
            'random':{'mask':random_bits,'localization':atom_localization(random_bits,draw,i)},
            'nonmonotone_edges':nonmonotone_edges(truth[0]),
            'neutral_response_disagreement':float(np.mean((neutral_margins>c['margin_threshold'])!=truth[0])),
            'zero_mask_response':bool(truth[0,0]),'zero_mask_target_probability':float(probs[0,0]),'full_margin':float(tables[0,-1]),
            'high_confidence_response_fraction':float(((probs[0]>=c['high_confidence_probability'])&truth[0]).mean()),
            'clinical_validity':'NOT TESTED','verification_views_status':'NOT YET TESTED'}
        archives.append({'margins':tables,'probabilities':probs,'circuits':circuits,'generation_index':original_indices[i],'target':target})
        rows.append(record)
    atomic_torch_save(archives,details)
    return {'task':task,'seed':seed,'predictor':name,'rows':rows,'private_circuits_sha256':file_sha(details),
        'private_circuits_file':str(details.relative_to(ROOT)),'encoded_waveform_examples':len(draw.y)*3,
        'head_mask_queries':exact_queries,'all_masks_verified':True,'calibration':cal,
        'preserved_prior_circuit_sha256':prior_archive_sha,'preserved_prior_margins_exact':prior_archive is not None}


def track_a1(c,budget):
    update(7,'IN_PROGRESS');groups=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            m,proof=trusted_reference(task,seed,'P0');cal=calibration(m,task,seed,c,budget)
            group=analyze_predictor(m,task,seed,'P0',c,budget,cal);group['provenance']=proof;groups.append(group)
            atomic_json({'groups':groups,'status':'IN_PROGRESS'},PRIVATE/'core_v1'/'a1_partial.json');print('A1',task,seed,flush=True)
    return finish_track(7,{'groups':groups,'new_fits':0,'optimizer_updates':0},budget,['Exact same-model response tables and bounded summaries complete; no clinical inference'])


@torch.no_grad()
def localization_metrics(model,draw,budget):
    records=[]
    for start in range(0,len(draw.y),64):
        budget.check();sel=selected(model,draw.x[start:start+64])
        for j,v in enumerate(sel):
            i=start+j;records.append(location(v,draw.intervals[i],draw.nuisance_intervals[i]))
    return {k:float(np.mean([v[k] for v in records if v[k] is not None])) if any(v[k] is not None for v in records) else None for k in records[0]}


@torch.no_grad()
def model_evaluation(model,task,seed,c,budget,cal=None):
    cal=cal or calibration(model,task,seed,c,budget);conditions={};private={}
    for condition in ('clean','competitive','removed','weak','artifact','eventless'):
        draw=draw_for('locked_evaluation',task,seed,c['evaluation_examples'],condition)
        logits=predict(model,draw.x,budget);private[condition]=logits
        conditions[condition]={'raw':reliability(logits,draw.y),'calibrated':reliability(logits,draw.y,cal['temperature'])}
        if condition in ('clean','competitive'):conditions[condition]['localization']=localization_metrics(model,draw,budget)
    pairs={}
    for condition in ('clean','competitive'):
        pair=draw_for('paired_evaluation',task,seed,100,condition);swapped=class_swap(pair)
        other=draw_for('paired_other',task,seed,100,condition)
        a=predict(model,pair.x,budget).argmax(1);b=predict(model,swapped.x,budget).argmax(1)
        v=predict(model,compose(other.base,pair.events),budget).argmax(1)
        pairs[condition]={'diagnostic_swap':paired_summary(((b!=a)&(b==swapped.y)).float().tolist(),seed),
            'nuisance_agreement':paired_summary((a==v).float().tolist(),seed)}
    d=draw_for('locked_evaluation',task,seed,c['evaluation_examples']);view=add_view(d,task,seed,'locked_added','high_frequency')
    altered=predict(model,view.x,budget);conditions['added']={'raw':reliability(altered,d.y),'calibrated':reliability(altered,d.y,cal['temperature'])}
    return {'conditions':conditions,'pairs':pairs,'calibration':cal,'paired_added_accuracy_interval':paired_accuracy_interval(private['clean'],altered,d.y,seed)}


def track_heads(c,budget):
    update(8,'IN_PROGRESS');rows=[];circuit_groups=[];eligibility=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            p0,proof=trusted_reference(task,seed,'P0');dev=draw_for('development',task,seed,c['development_examples'])
            dev_score=reliability(predict(p0,dev.x,budget),dev.y);eligible=dev_score['macro_f1']>=c['criteria']['parent_clean_f1']
            eligibility.append({'task':task,'seed':seed,'eligible':eligible,'development_macro_f1':dev_score['macro_f1'],'parent_encoder_sha256':tensor_digest(p0.encoder.state_dict())})
            if not eligible:continue
            train=draw_for('train',task,seed,c['train_examples'],'train');f,r=encode(p0,train.x);packed=torch.cat((f,r[:,:,None]),-1)
            encoder=copy.deepcopy(p0.encoder.state_dict());encoder_hash=tensor_digest(encoder)
            for name,method in c['head_methods'].items():
                preserved=state_path(task,seed,name).exists()
                if preserved:
                    amendment=json.loads((ROOT/'experiments/mist_cove/phase9_reporting_freeze_v2.json').read_text())
                    recovery=amendment['recovery']
                    if (task,seed,name)!=('single',123,'E0'):raise FileExistsError('unregistered partial fit preserved')
                    p=state_path(task,seed,name)
                    if file_sha(p.with_suffix('.creation.json'))!=recovery['creation_sha256'] or file_sha(p)!=recovery['checkpoint_sha256']:raise ValueError('preserved fit proof mismatch')
                    creation=json.loads(p.with_suffix('.creation.json').read_text())
                    if creation['identity']['train_digest']!=tensor_digest({'features':packed,'y':train.y}):raise ValueError('preserved TRAIN features mismatch')
                    m=load_new(task,seed,name,{'creation':creation});training_seconds=None;loss=None
                else:
                    torch.manual_seed(seed+402);m=make_model(method);m.encoder.load_state_dict(encoder);m.encoder.requires_grad_(False)
                    opt=optimizer(m,c);g=torch.Generator().manual_seed(seed+711);started=time.perf_counter()
                    loss=steps(CachedHead(m),opt,g,packed,train.y,c,0,c['head_updates'],budget)
                    training_seconds=time.perf_counter()-started;ident=identity(task,seed,name,tensor_digest({'features':packed,'y':train.y}));ident['encoder_sha256']=encoder_hash
                    creation=seal(state_path(task,seed,name),m,opt,g,c['head_updates'],ident)
                if tensor_digest(m.encoder.state_dict())!=encoder_hash:raise ValueError('frozen encoder changed')
                restored=load_new(task,seed,name,{'creation':creation})
                if tensor_digest(restored.state_dict())!=tensor_digest(m.state_dict()):raise ValueError('head exact restore mismatch')
                evaluation=model_evaluation(restored,task,seed,c,budget)
                row={'task':task,'seed':seed,'method':name,'encoder_sha256':encoder_hash,'creation':creation,'evaluation':evaluation,
                    'training':{'updates':c['head_updates'],'cached_feature_examples':c['head_updates']*16,'full_network_training_examples':0,
                        'common_parent_encoding_examples':1000 if name=='E0' else 0,'parameters':3593,'trainable_head_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
                        'training_seconds':training_seconds,'final_loss':loss,'preserved_completed_fit':preserved}}
                rows.append(row);print('A2',task,seed,name,flush=True)
                if name in ('E0','E1'):circuit_groups.append(analyze_predictor(restored,task,seed,name,c,budget,evaluation['calibration']))
                atomic_json({'rows':rows,'circuit_groups':circuit_groups,'eligibility':eligibility,'status':'IN_PROGRESS'},PRIVATE/'core_v1'/'heads_partial.json')
    checks=[]
    for parent in eligibility:
        if not parent['eligible']:continue
        local={v['method']:v for v in rows if (v['task'],v['seed'])==(parent['task'],parent['seed'])};e0,e1,e3=[local[k]['evaluation']['conditions']['clean'] for k in ('E0','E1','E3')]
        clauses={'clean_retention':e1['raw']['macro_f1']>=e0['raw']['macro_f1']-.03,
            'single_precision_gain':parent['task']!='single' or e1['localization']['precision']>=e3['localization']['precision']+.10,
            'single_coverage_retention':parent['task']!='single' or e1['localization']['coverage']>=e3['localization']['coverage']}
        checks.append({'task':parent['task'],'seed':parent['seed'],'clauses':clauses,'pass':all(clauses.values())})
    gate={'status':'PASS' if checks and all(v['pass'] for v in checks) else 'FAIL' if checks else 'NOT TESTED','checks':checks}
    return finish_track(8,{'rows':rows,'circuit_groups':circuit_groups,'eligibility':eligibility,'gate':gate,
        'new_fits':len(rows),'optimizer_updates':len(rows)*c['head_updates'],'severe_robustness_prerequisite':False},budget,['Clean matched-head study executed independently of Track B'])


def train_paired_steps(m,opt,g,x,plus,y,c,arm,start,end,budget):
    totals={'classification':0.,'full_js':0.,'query_js':0.};m.train()
    for step in range(start,end):
        budget.check();ix=torch.randperm(len(y),generator=g)[:c['batch_size']]
        opt.zero_grad(set_to_none=True);f,r=m.encode_candidates(torch.cat((x[ix],plus[ix])))
        masks=sample_query_masks(len(ix),device=x.device)
        loss,parts=paired_objective(m,f[:len(ix)],r[:len(ix)],f[len(ix):],r[len(ix):],y[ix],masks,arm,step,c['updates'])
        if not torch.isfinite(loss):raise FloatingPointError('paired loss nonfinite')
        loss.backward();nn.utils.clip_grad_norm_(m.parameters(),c['gradient_clip'],error_if_nonfinite=True);opt.step()
        if any(not torch.isfinite(p).all() for p in m.parameters()):raise FloatingPointError('paired parameters nonfinite')
        for key in totals:totals[key]+=parts[key]
    return {k:v/max(1,end-start) for k,v in totals.items()}


def track_robustness(c,budget):
    update(9,'IN_PROGRESS');rows=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            train=draw_for('train',task,seed,c['train_examples'],'train');plus=training_pair(train,task,seed)
            for arm in ('C0','C1','C2'):
                if state_path(task,seed,arm).exists():raise FileExistsError('preserve partial paired fit before training')
                torch.manual_seed(seed);m=make_model('M1');opt=optimizer(m,c);g=torch.Generator().manual_seed(seed+711)
                started=time.perf_counter();losses=train_paired_steps(m,opt,g,train.x,plus,train.y,c,arm,0,c['updates'],budget);seconds=time.perf_counter()-started
                ident=identity(task,seed,arm,tensor_digest({'clean':train.x,'corrupted':plus,'y':train.y}))
                creation=seal(state_path(task,seed,arm),m,opt,g,c['updates'],ident);m=load_new(task,seed,arm,{'creation':creation})
                evaluation=model_evaluation(m,task,seed,c,budget)
                record={'task':task,'seed':seed,'method':arm,'creation':creation,'evaluation':evaluation,
                    'training':{'updates':300,'base_examples':4800,'view_forwards':9600,'head_query_forwards':19200 if arm=='C2' else 0,
                        'head_full_view_forwards':9600,'geometry_mask_draws':9600,'parameters':3593,'trainable_parameters':3593,
                        'training_seconds':seconds,'loss_means':losses,'encoder_sha256':tensor_digest(m.encoder.state_dict())}}
                rows.append(record);atomic_json({'rows':rows,'status':'IN_PROGRESS'},PRIVATE/'core_v1'/'robustness_partial.json');print('B',task,seed,arm,flush=True)
    checks=[];deltas=[];positive=0
    for row in rows:
        if row['method']!='C2':continue
        c0=next(v for v in rows if (v['task'],v['seed'],v['method'])==(row['task'],row['seed'],'C0'))
        c1=next(v for v in rows if (v['task'],v['seed'],v['method'])==(row['task'],row['seed'],'C1'))
        clean=row['evaluation']['conditions']['clean']['raw'];comp=row['evaluation']['conditions']['competitive']['raw']
        clauses={'clean_floor':clean['macro_f1']>=.8,'clean_retention':clean['macro_f1']>=c0['evaluation']['conditions']['clean']['raw']['macro_f1']-.03,
            'competitive':comp['macro_f1']>=.7,'diagnostic_swap':all(v['diagnostic_swap']['mean']>=.8 for v in row['evaluation']['pairs'].values()),
            'nuisance_agreement':all(v['nuisance_agreement']['mean']>=.8 for v in row['evaluation']['pairs'].values())}
        delta=comp['macro_f1']-c1['evaluation']['conditions']['competitive']['raw']['macro_f1'];deltas.append(delta);positive+=int(delta>0)
        checks.append({'task':row['task'],'seed':row['seed'],'clauses':clauses,'pass':all(clauses.values()),'competitive_delta_C2_minus_C1':delta})
    incremental={'mean_delta':float(np.mean(deltas)),'positive_cells':positive,'clean_retention_all':all(v['clauses']['clean_retention'] and v['clauses']['clean_floor'] for v in checks)}
    incremental['status']='PASS' if incremental['mean_delta']>=.03 and positive>=4 and incremental['clean_retention_all'] else 'FAIL'
    gate={'status':'PASS' if all(v['pass'] for v in checks) else 'FAIL','checks':checks,'incremental':incremental}
    return finish_track(9,{'rows':rows,'gate':gate,'new_fits':18,'optimizer_updates':5400},budget,['C2 primary retained regardless of scores; clean/robust/query benefits assessed separately'])


def track_stability(c,budget):
    update(10,'IN_PROGRESS');groups=saved_receipt('wp07_receipt.json')['groups']+saved_receipt('wp08_receipt.json')['circuit_groups'];results=[]
    head_rows=saved_receipt('wp08_receipt.json')['rows']
    for group in groups:
        task,seed,name=group['task'],group['seed'],group['predictor'];draw,_=audit_draw(task,seed,c)
        m=trusted_reference(task,seed,'P0')[0] if name=='P0' else load_new(task,seed,name,next(v for v in head_rows if (v['task'],v['seed'],v['method'])==(task,seed,name)))
        views=verification_views(draw,task,seed);encoded=[encode(m,d.x) for d in views];records=[]
        for i,original in enumerate(group['rows']):
            target=original['target_class'];truth=[]
            for f,r in encoded:
                margins,_=response_table(m,f[i:i+1],r[i:i+1],target,batch_size=c['mask_batch'],budget=budget)
                truth.append(margins>c['margin_threshold'])
            truth=np.stack(truth);stable=bool(truth[:,-1].all());clauses=original['signed_summary']['clauses']
            matrix=[[verify_clause(t,cl['positive_bits'],cl['negative_bits']) for cl in clauses] for t in truth]
            active=[j for j,cl in enumerate(clauses) if cl['active_at_original_full_mask'] and cl['waveform_support_claim']]
            fixed=any(all(v[j] for v in matrix) for j in active)
            substitution=all(any(v[j] for j in active) for v in matrix)
            signed_fixed=any(all(v[j] for v in matrix) for j in range(len(clauses)))
            records.append({'task':task,'seed':seed,'predictor':name,'generation_index':original['generation_index'],
                'correct_original':original['correct'],'full_input_stable':stable,
                'fixed_active_clause_survives':fixed,'active_route_each_view':substitution,
                'fixed_signed_intervention_state_survives':signed_fixed,
                'all_case_fixed_waveform_explanation':stable and fixed,'all_case_substitute_waveform_explanation':stable and substitution,
                'prior_only_discovery':original['signed_summary']['prior_only'],'displayed_clauses':len(clauses),
                'unchanged_clause_view_validity':matrix,'independent_views':3,'refit_literals':False,
                'certificate_scope':'exact completion checks on three new sampled views; not all perturbations'})
        results+=records;print('stability',task,seed,name,flush=True)
    decisions={}
    for predictor in ('P0','E0','E1'):
        selected_rows=[r for r in results if r['predictor']==predictor];stable=[r for r in selected_rows if r['full_input_stable']]
        rate=sum(r['fixed_active_clause_survives'] for r in stable)/len(stable) if stable else None
        decisions[predictor]={'examples':len(selected_rows),'stable_predictions':len(stable),
            'prediction_stability':len(stable)/len(selected_rows) if selected_rows else None,
            'fixed_active_conditional':rate,'substitution_active_conditional':sum(r['active_route_each_view'] for r in stable)/len(stable) if stable else None,
            'fixed_all_case':sum(r['all_case_fixed_waveform_explanation'] for r in selected_rows)/len(selected_rows) if selected_rows else None,
            'substitution_all_case':sum(r['all_case_substitute_waveform_explanation'] for r in selected_rows)/len(selected_rows) if selected_rows else None,
            'status':'PASS' if rate is not None and rate>=.8 else 'FAIL' if rate is not None else 'INCONCLUSIVE'}
    return finish_track(10,{'rows':results,'decisions':decisions,'head_mask_queries':len(results)*3*32768,'new_fits':0,'optimizer_updates':0},budget,['Unchanged discovered clauses evaluated; full-input stability kept separate from favorable masks'])


@torch.no_grad()
def operator_records(model,draw,other,index,c):
    from mist_cove.phase2.audit import replace_group,union_intervals
    from mist_cove.phase6.audit import donor_diagnostics
    x=draw.x[index:index+1];baseline=draw.base[index:index+1];diagnostic=draw.events[index:index+1].sum(1);truth=draw.intervals[index]
    original=model(x);target=int(original.argmax(1));before=margin(original,target)[0]
    positive_bits=0
    for a,b in truth:
        for j in range(15):
            if j*200<b and (j+1)*200>a:positive_bits|=1<<j
    atoms=torch.ones(1,15,dtype=torch.bool)
    for j in range(15):
        if positive_bits&(1<<j):atoms[:,j]=False
    f,r=encode(model,x);internal=internal_logits(model,f,r,atoms)
    visible=(atoms[:,:-1]&atoms[:,1:])[0];candidate_truth=torch.tensor([any(j*200<b and j*200+400>a for a,b in truth) for j in range(14)])
    records=[{'operator':'internal_availability','target':'diagnostic support features','target_removal_fraction':None,
        'declared_candidate_withholding_fraction':float((~visible[candidate_truth]).float().mean()),
        'nontarget_max_residual':0.,'nontarget_metric':'retained feature/RMS identity; not waveform preservation',
        'collateral_disjoint_candidates_hidden':int((~visible&~candidate_truth).sum()),
        'physical_compatibility':None,'margin_effect':float(before-margin(internal,target)[0]),'physiological_validity':'NOT TESTED'}]
    for name in ('identity','diagnostic','nuisance'):
        modified,contract=component_contract(draw,index,name)
        diagnostics=donor_diagnostics(x,modified,truth,c['donor_guard'])
        records.append({'operator':'identity' if name=='identity' else 'exact_'+name+'_removal','target':'diagnostic' if name=='identity' else name,
            **{k:v for k,v in contract.items() if k!='operator'},'physical_compatibility':diagnostics,
            'margin_effect':float(before-margin(model(modified),target)[0]),'physiological_validity':'NOT TESTED'})
    changed,_,_=replace_group(model,x,truth,strategy='bridge')
    residual_d,_,_=replace_group(model,diagnostic,truth,strategy='bridge')
    changed_baseline,_,_=replace_group(model,baseline,truth,strategy='bridge')
    bridge={'operator':'historical_bridge','target':'diagnostic','target_removal_fraction':float(1-residual_d.norm()/diagnostic.norm().clamp_min(1e-12)),
        'nontarget_max_residual':float((changed_baseline-baseline).abs().max()),'nontarget_metric':'same linear bridge applied to stored nontarget baseline',
        'physical_compatibility':donor_diagnostics(x,changed,truth,c['donor_guard']),'margin_effect':float(before-margin(model(changed),target)[0]),'physiological_validity':'NOT TESTED'}
    replacement=x.clone();non=baseline.clone()
    for lo,hi in union_intervals(truth):replacement[:,:,lo:hi]=other.base[index:index+1,:,lo:hi];non[:,:,lo:hi]=other.base[index:index+1,:,lo:hi]
    independent={'operator':'historical_independent_baseline','target':'diagnostic','target_removal_fraction':1.,
        'nontarget_max_residual':float((non-baseline).abs().max()),'nontarget_metric':'changed original baseline on target union',
        'physical_compatibility':donor_diagnostics(x,replacement,truth,c['donor_guard']),
        'margin_effect':float(before-margin(model(replacement),target)[0]),'physiological_validity':'NOT TESTED'}
    records += [bridge,independent]
    accepted=[v for v in (bridge,independent) if v['physical_compatibility']['accepted']]
    return {'records':records,'original_target_class':target,'correct':target==int(draw.y[index]),
        'accepted_donors':len(accepted),'accepted_donor_margin_spread':abs(accepted[0]['margin_effect']-accepted[1]['margin_effect']) if len(accepted)==2 else None,
        'conditional_donor_status':'DESCRIPTIVE' if len(accepted)==2 else 'NOT TESTED'}


def track_operators(c,budget):
    update(11,'IN_PROGRESS');rows=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            m,_=trusted_reference(task,seed,'P0');draw,indices=audit_draw(task,seed,c)
            other=draw_for('operator_independent_baseline',task,seed,len(draw.y))
            for i in range(len(draw.y)):
                budget.check();rows.append({'task':task,'seed':seed,'generation_index':indices[i],**operator_records(m,draw,other,i,c)})
    correctness=all(v['target_removal_fraction']==1. and v['nontarget_max_residual']<=2e-5 for row in rows for v in row['records'] if v['operator']=='exact_diagnostic_removal')
    return finish_track(11,{'rows':rows,'oracle_correctness':'PASS' if correctness else 'FAIL','historical_guard':c['donor_guard'],
        'donor_attempts':len(rows)*2,'accepted_donors':sum(r['accepted_donors'] for r in rows),'new_fits':0,'optimizer_updates':0},budget,['Target removal, nontarget preservation, compatibility and response reported separately; five waveform controls plus internal operator'])


def track_reliability(c,budget):
    update(12,'IN_PROGRESS');rows=[];sanity=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            m,_=trusted_reference(task,seed,'P0')
            prior_cal=next(v['calibration'] for v in saved_receipt('wp07_receipt.json')['groups'] if (v['task'],v['seed'])==(task,seed))
            evaluation=model_evaluation(m,task,seed,c,budget,prior_cal)
            rows.append({'task':task,'seed':seed,'method':'P0','evaluation':evaluation})
            draw,_=audit_draw(task,seed,c);l=predict(m,draw.x,budget)
            gen=torch.Generator().manual_seed(seed_for('random_labels',task,seed));permuted=draw.y[torch.randperm(len(draw.y),generator=gen)]
            torch.manual_seed(seed_for('model_randomization',task,seed));random=make_model('M1').eval();a=draw.x[:1]
            f,r=encode(m,a);q,s=encode(random,a);target=int(l[0].argmax())
            old,_=response_table(m,f,r,target,batch_size=c['mask_batch'],budget=budget);new,_=response_table(random,q,s,target,batch_size=c['mask_batch'],budget=budget)
            sanity.append({'task':task,'seed':seed,'original_accuracy':float((l.argmax(1)==draw.y).float().mean()),
                'permuted_label_accuracy':float((l.argmax(1)==permuted).float().mean()),'response_query_uses_labels':False,
                'randomized_model_response_disagreement':float(np.mean((old>1e-6)!=(new>1e-6))),
                'randomized_model_full_prediction':int(random(a).argmax(1)),'target_remains_original_prediction':True})
    return finish_track(12,{'rows':rows,'sanity_controls':sanity,'new_fits':0,'optimizer_updates':0,'additional_sanity_head_mask_queries':12*32768},budget,['Wrong predictions retain their own predicted-class explanations; calibrated/null/removed/error metrics retained'])


def execute(stage):
    c=config();require_scope(c['scope']);proof=frozen();configure_precision('cpu');torch.set_num_threads(2)
    stages={'circuits':(7,track_a1),'heads':(8,track_heads),'robustness':(9,track_robustness),'stability':(10,track_stability),'operators':(11,track_operators),'reliability':(12,track_reliability)}
    if stage not in stages:raise ValueError('unknown scientific stage')
    wp,fn=stages[stage];progress=json.loads((PUBLIC/'progress.json').read_text())
    if progress['work_packages'][wp-2]['status']!='COMPLETE':raise ValueError('previous work package must be complete')
    if wp==7:output(PRIVATE/'core_v1',scope=c['scope'])
    receipt=PRIVATE/'core_v1'/f'wp{wp:02}_receipt.json'
    if receipt.exists():raise FileExistsError('completed track preserved; no repeat fits/evaluation')
    stopped=PRIVATE/'core_v1'/f'STOPPED_WP{wp:02}.json'
    if stopped.exists():
        amendment=json.loads((ROOT/'experiments/mist_cove/phase9_reporting_freeze_v2.json').read_text())
        if wp!=8 or file_sha(stopped)!=amendment['recovery']['stopped_receipt_sha256']:
            raise FileExistsError('stopped experiment preserved; no unregistered retry')
    budget=cumulative_budget(c)
    try:
        result=fn(c,budget);budget.check()
        if frozen()!=proof:raise ValueError('freeze changed during scientific execution')
        return result
    except BaseException as error:
        update(wp,'BLOCKED',block=repr(error));atomic_json({'status':'STOPPED','stage':stage,'error':repr(error),'wall_seconds':budget.elapsed},PRIVATE/'core_v1'/f'STOPPED_WP{wp:02}.json');raise
