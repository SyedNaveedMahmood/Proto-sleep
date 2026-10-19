"""Additional generated engineering checks, separate from confirmatory fits."""
import copy
import json
import torch

from mist_cove.phase2.synthetic_runtime import assert_exact
from mist_cove.phase2.training import configure_precision
from mist_cove.phase5.runtime import Budget
from mist_cove.phase6.generator import generate
from mist_cove.phase7.prospective import optimizer,steps,seal,restore,CONFIG
from mist_cove.phase7.robustness import ClippedModel
from mist_evidence.runtime import rng_state,restore_rng


def test_exact_clipped_model_resume_includes_fixed_statistics_and_optimizer(tmp_path):
    configure_precision('cpu');torch.set_num_threads(2)
    c=json.loads(CONFIG.read_text());draw=generate(10,9472,'multiple');identity={'prospective_test':'clipped-resume'}
    torch.manual_seed(9472);model=ClippedModel(torch.linspace(-.2,.2,18),torch.linspace(.1,1.,18))
    initial=copy.deepcopy(model.state_dict());initial_rng=rng_state()
    def fresh():
        model=ClippedModel(torch.zeros(18),torch.ones(18));model.load_state_dict(initial)
        opt=optimizer(model,c);g=torch.Generator().manual_seed(9472+711);restore_rng(initial_rng)
        return model,opt,g
    m,o,g=fresh();steps(m,o,g,draw.x,draw.y,c,0,6,Budget(60))
    expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),g.get_state(),rng_state())
    m,o,g=fresh();steps(m,o,g,draw.x,draw.y,c,0,3,Budget(60))
    p=tmp_path/'clipped.pt';creation=seal(p,m,o,g,3,identity)
    resumed=ClippedModel(torch.zeros(18),torch.ones(18));opt=optimizer(resumed,c);gen=torch.Generator()
    assert restore(p,creation,identity,resumed,opt,gen)==3
    steps(resumed,opt,gen,draw.x,draw.y,c,3,6,Budget(60))
    assert_exact(expected,(resumed.state_dict(),opt.state_dict(),gen.get_state(),rng_state()))
