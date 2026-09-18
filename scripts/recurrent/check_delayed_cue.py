"""Delayed-only toy supervision checks that recurrent credit can be learned."""
import json
import torch
from qwen_vl.models.memory_writer import MemoryWriter


def run():
    torch.set_num_threads(1);torch.manual_seed(3)
    writer=MemoryWriter(slots=4,width=16,layers=3,heads=2,ffn_width=32)
    head=torch.nn.Linear(16,4)
    optimizer=torch.optim.AdamW(list(writer.parameters())+list(head.parameters()),lr=.003)
    labels=torch.arange(4).repeat(2)
    cue=torch.nn.functional.one_hot(labels,16).float()[:,None,:]
    empty=torch.zeros_like(cue);instruction=torch.zeros(8,8,16)
    losses=[]
    for update in range(200):
        optimizer.zero_grad(set_to_none=True)
        state=writer.initial(8,torch.device('cpu'))
        for step in range(8):
            state=writer(state,cue if step==0 else empty,instruction,
                         torch.ones(8,1,dtype=torch.bool),torch.ones(8,8,dtype=torch.bool)).state
        logits=head(state.mean(1))
        loss=torch.nn.functional.cross_entropy(logits,labels)
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]*.2,(losses[0],losses[-1])
    accuracy=float((logits.argmax(-1)==labels).float().mean())
    assert accuracy==1.
    print(json.dumps(dict(initial_loss=losses[0],final_loss=losses[-1],updates=200,accuracy=accuracy,supervision='Only write7, cue only at write0; no detach inside K8'),indent=2))

if __name__=='__main__':run()
