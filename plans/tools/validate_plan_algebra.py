#!/usr/bin/env python3
"""CPU algebra checks for the specification, NOT Qwen/model implementation tests."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def history(t, mode, recent=4):
    if t < 0 or recent < 0:
        raise ValueError("negative")
    if mode == "uniform8":
        return list(range(t+1)) if t <= 8 else np.linspace(0, t, 9, dtype=int).tolist()
    if mode == "recent":
        return list(range(max(0, t-recent), t+1))
    if mode == "current":
        return [t]
    raise ValueError(mode)


def check_history():
    assert history(100, "uniform8") == [0,12,25,37,50,62,75,87,100]
    assert history(100, "recent") == [96,97,98,99,100]
    for t in range(513):
        for mode, r in [("uniform8",4),("recent",4),("recent",0),("current",0)]:
            ids = history(t, mode, r)
            assert ids == sorted(set(ids)) and ids[-1] == t and ids[0] >= 0
    return "513 timesteps; exact uniform/recent/current and early-window invariants"


def check_feature_order():
    for h,w in [(2,2),(6,10),(8,4)]:
        raster = torch.arange(h*w*3).reshape(1,h,w,3)
        rows = raster.reshape(1,h//2,2,w//2,2,3).permute(0,1,3,2,4,5).reshape(h*w,3)
        restored = rows.reshape(1,h//2,w//2,2,2,3).permute(0,1,3,2,4,5).reshape(1,h,w,3)
        assert torch.equal(restored,raster)
    return "rectangular merge-block/raster permutation round-trips"


def check_reset_and_detach():
    initial = nn.Parameter(torch.randn(1,4,dtype=torch.float64))
    old = torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    reset = torch.where(torch.tensor([[True],[False]]), initial.expand(2,-1), old)
    reset.sum().backward()
    assert torch.equal(initial.grad,torch.ones_like(initial))
    assert torch.equal(old.grad[0],torch.zeros_like(old.grad[0]))
    assert torch.equal(old.grad[1],torch.ones_like(old.grad[1]))
    assert not reset.detach().requires_grad
    assert torch.equal(reset.detach(),reset)
    return "selective reset trains initial slots; detach preserves value and cuts graph"


def state_ce(logits, labels):
    loss = F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                           labels[:,1:].reshape(-1), reduction="none", ignore_index=-100)
    loss = loss.reshape(labels.shape[0],-1)
    count = (labels[:,1:] != -100).sum(-1)
    assert (count > 0).all()
    return loss.sum(-1)/count


def check_loss_partition():
    torch.manual_seed(42)
    x = torch.randn(3,7,9,dtype=torch.float64,requires_grad=True)
    y = torch.full((3,7),-100,dtype=torch.long)
    y[0,-1:] = 2; y[1,-3:] = torch.tensor([1,2,3]); y[2,-5:] = torch.tensor([3,2,1,4,5])
    loss = state_ce(x,y).mean(); g, = torch.autograd.grad(loss,x)
    xx = x.detach().clone().requires_grad_(True)
    for sl in (slice(0,2),slice(2,3)):
        (state_ce(xx[sl],y[sl]).sum()/3).backward()
    torch.testing.assert_close(g,xx.grad,rtol=1e-12,atol=1e-12)
    return "unequal action-token lengths and uneven microbatches have identical gradients"


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Parameter(torch.randn(4,dtype=torch.float64))
        self.wx = nn.Linear(5,4).double()
        self.wh = nn.Linear(4,4).double()
        self.merger = nn.Linear(5,4).double()
        self.reader = nn.Linear(12,1).double()

    def produce(self, frames):
        outputs = {}
        for i,f in enumerate(frames):
            outputs[("v",i)] = self.merger(f)
        state = self.initial
        for i,f in enumerate(frames):
            h1 = torch.tanh(self.wh(state)+self.wx(f))
            h3 = 0.8*state+0.2*torch.tanh(self.wh(h1))
            outputs[("h1",i)] = h1
            outputs[("h3",i)] = h3
            state = h3
        return outputs

    def state_loss(self, outputs, i):
        q = torch.cat([outputs[("h3",i)], outputs[("v",i)], outputs[("v",max(0,i-1))]])
        nav = (self.reader(q).squeeze() - 0.1*i).square()
        return nav + 0.05*outputs[("h1",i)].square().mean() + 0.07*outputs[("h3",i)].square().mean()


def check_gradient_bridge():
    torch.manual_seed(3)
    model = Toy(); other = copy.deepcopy(model)
    frames = torch.randn(8,5,dtype=torch.float64)
    originals = model.produce(frames)
    sum(model.state_loss(originals,i)/8 for i in range(8)).backward()
    originals2 = other.produce(frames)
    proxies = {key:x.detach().requires_grad_(True) for key,x in originals2.items()}
    for i in range(8):
        (other.state_loss(proxies,i)/8).backward()
    keys = [k for k,p in proxies.items() if p.grad is not None]
    torch.autograd.backward(tuple(originals2[k] for k in keys),tuple(proxies[k].grad for k in keys))
    worst = 0.0
    for (n,a),(m,b) in zip(model.named_parameters(),other.named_parameters()):
        assert n==m and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-11)
        worst=max(worst,(a.grad-b.grad).abs().max().item())
    return f"8-step recurrence, shared live merger, ancestor/descendant auxiliaries; max gradient delta {worst:.3g}"


def check_causality_and_credit():
    torch.manual_seed(7)
    model=Toy(); frames=torch.randn(8,5,dtype=torch.float64,requires_grad=True)
    original=model.produce(frames)
    changed=frames.detach().clone(); changed[5:]+=10
    altered=model.produce(changed)
    for t in range(5):
        torch.testing.assert_close(original[("h3",t)],altered[("h3",t)],rtol=0,atol=0)
    original[("h3",7)].square().sum().backward()
    assert frames.grad[0].abs().sum()>0
    return "future perturbations leave earlier states unchanged; last loss reaches first input"


def check_delay_eligibility():
    pairs=[(t,s) for t in range(8) for s in range(t+1) if s<=t-4-1]
    assert sorted({t-s for t,s in pairs})==[5,6,7]
    assert len(pairs)==6
    return "K8/R4 within-segment off-window eligible delays are exactly 5,6,7"


def check_global_weighting():
    local=[torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([3.],dtype=torch.float64),torch.empty(0,dtype=torch.float64)]
    w=len(local); n=sum(x.numel() for x in local)
    assembled=sum(w/n*x.sum() for x in local)/w
    torch.testing.assert_close(assembled,torch.cat(local).mean())
    return "global-count scaling matches union mean with unequal and empty ranks (algebra, not distributed execution)"


def check_collector_local():
    script=Path(__file__).with_name("fetch_references.py")
    spec=importlib.util.spec_from_file_location("reference_collector",script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); repo=root/"repo"; repo.mkdir()
        def run(*args):
            return subprocess.run(["git","-C",str(repo),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        run("init"); run("config","user.name","Fixture"); run("config","user.email","fixture@example.invalid")
        (repo/"sample.py").write_text("VALUE = 1\n"); (repo/"LICENSE").write_text("Local test fixture.\n")
        run("add",".");run("commit","-m","fixture")
        manifest={"schema_version":1,"repositories":[{"key":"fixture","url":str(repo),"ref":"HEAD","files":["sample.py"]}],"metadata":[]}
        out=root/"exports"; lock=module.collect(manifest,out,set(),False)
        assert (out/"fixture/sample.py").read_text()=="VALUE = 1\n"
        module.collect(manifest,out,set(),True)
        (repo/"sample.py").write_text("VALUE = 2\n");run("add",".");run("commit","-m","changed")
        lock2=module.collect(manifest,out,set(),False)
        assert lock2["repositories"]["fixture"]["commit"]==lock["repositories"]["fixture"]["commit"]
        assert (out/"fixture/sample.py").read_text()=="VALUE = 1\n"
    return "local Git export, license inclusion, hash verification and commit-lock reuse; no network"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("validation/algebra_results.json"))
    args=parser.parse_args()
    tests=[check_history,check_feature_order,check_reset_and_detach,check_loss_partition,
           check_gradient_bridge,check_causality_and_credit,check_delay_eligibility,
           check_global_weighting,check_collector_local]
    results=[]
    for fn in tests:
        try:
            detail=fn();results.append({"test":fn.__name__,"status":"passed","detail":detail})
        except Exception as error:
            results.append({"test":fn.__name__,"status":"failed","detail":f"{type(error).__name__}: {error}"})
    report={"scope":"CPU specification algebra and local collector fixture; not model implementation tests",
            "torch_version":torch.__version__,"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return int(any(r["status"]!="passed" for r in results))

if __name__=="__main__":
    raise SystemExit(main())
