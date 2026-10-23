"""Fit-only linear stage probe on frozen morphology representations."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from mist_sleep.experiment import new_model, verify_protocol
from mist_sleep.metrics import fixed_five
from mist_sleep.source import PRIVATE, PUBLIC, load_full_train


@torch.no_grad()
def embeddings(model, wave, selected, device):
    model.eval()
    blocks=[]
    for start in range(0,len(selected),64):
        x=torch.from_numpy(np.asarray(wave[selected[start:start+64]]).copy()).to(device)
        blocks.append(model(x)["embedding"].float().cpu().numpy())
    return np.concatenate(blocks)


def main():
    verify_protocol()
    target=PUBLIC/"stage_information_probe.json"
    if target.exists():raise FileExistsError("probe already completed")
    torch.set_num_threads(2)
    wave,index=load_full_train()
    rng=np.random.default_rng(4242)
    selected={}
    for role,name,per_class in ((0,"fit",1000),(1,"development",400)):
        selected[name]=np.concatenate([rng.choice(np.flatnonzero((index["role"]==role)&
            (index["stage"]==stage)),size=per_class,replace=False) for stage in range(5)])
    if set(index["person"][selected["fit"]]) & set(index["person"][selected["development"]]):
        raise RuntimeError("probe participant overlap")
    results={}
    for seed in (123,456):
        for arm in ("M0","M1","M2","M3"):
            name=f"{arm}_{seed}"
            train=json.loads((PUBLIC/f"{name}.json").read_text())
            path=PRIVATE/"models"/name/"step_1200.pt"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=train["trajectory"][-1]["checkpoint_sha256"]:
                raise RuntimeError("checkpoint digest differs")
            state=torch.load(path,map_location="cpu",weights_only=True)
            model=new_model(arm)
            model.load_state_dict(state["model"],strict=True)
            model.cuda()
            a=embeddings(model,wave,selected["fit"],torch.device("cuda"))
            b=embeddings(model,wave,selected["development"],torch.device("cuda"))
            probe=LogisticRegression(C=1,max_iter=200,random_state=4242)
            probe.fit(a,index["stage"][selected["fit"]])
            logits=probe.decision_function(b)
            results[name]={"fit_examples":len(a),"development_examples":len(b),
                           "participant_overlap":0,
                           "metrics":fixed_five(index["stage"][selected["development"]],
                                                logits,index["person"][selected["development"]])}
            print(name,results[name]["metrics"]["mean_participant_macro_f1"],flush=True)
    target.write_text(json.dumps({"status":"COMPLETE","fit_only_probe":True,
        "selection":"fixed seed 4242; 1000 fit and 400 development examples per class",
        "results":results},indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
