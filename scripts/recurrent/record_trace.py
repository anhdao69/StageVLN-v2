"""Record a deterministic real-image M64/R4/K8 reference for later execution work."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.episode_trainer import EpisodeTrainer
from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
    torch.set_num_threads(1);torch.manual_seed(42)
    _install_qwen35_flash_attention_fix()
    policy,processor,recent=NavigationPolicy.from_export(args.export,device='cuda',dtype=torch.float32,attn_implementation='flash_attention_2')
    assert recent==4 and policy.memory_slots==64
    policy.backbone.model.visual.requires_grad_(False)
    policy.backbone.model.visual.merger.requires_grad_(True)
    assert all(p.dtype==torch.float32 for p in policy.parameters() if p.requires_grad)
    policy.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    policy.train()
    all_episodes=load_manifest('/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl')
    short=min(all_episodes,key=lambda ep:len(ep.frames))
    episodes=(short,next(ep for ep in all_episodes if ep.episode!=short.episode))
    loader=FrameLoader('/groups/yshang/an221229/data/JanusVLN_data',processor.image_processor)
    trainer=EpisodeTrainer(policy,episodes,loader,recent=4,reader_microbatch=4)
    writes={};rows=[];segments=[];gates=[]
    handles=[]
    for index,block in enumerate(policy.writer.blocks):
        def capture(module,inputs,output,index=index):
            value=output.detach().sigmoid().float()
            gates.append(dict(block=index,mean=float(value.mean()),minimum=float(value.min()),maximum=float(value.max())))
        handles.append(block.gate.register_forward_hook(capture))
    original_write=policy.write
    def write(previous,feature,instruction):
        gates.clear()
        memory=original_write(previous,feature,instruction)
        writes[feature.key]=dict(memory_norm=float(memory.detach().norm()),memory_dtype=str(memory.dtype),
                                 reset=previous is None,gate_statistics=list(gates))
        return memory
    policy.write=write
    original_read=policy.reader.forward
    logit_ids=[policy.tokenizer.encode(a,add_special_tokens=False)[0] for a in ('MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','STOP')]
    def read(states,features,memories=None):
        losses,output=original_read(states,features,memories)
        for i,state in enumerate(states):
            key=state.frame_keys[-1]
            rows.append(dict(frame=asdict(key),reader_frames=[asdict(k) for k in state.frame_keys],
                             image_grids=[list(span.grid_thw) for span in state.image_spans],
                             token_length=len(state.input_ids),target_length=int(state.labels.ne(-100).sum()),
                             loss=float(losses[i].detach()),selected_logit_ids=logit_ids,
                             selected_logits=output.logits[i,:,logit_ids].detach().float().cpu().tolist(),**writes[key]))
        return losses,output
    policy.reader.forward=read
    gradient_names=['writer.initial_slots','writer.blocks.0.gate.weight','memory_adapter.output_projection.weight',
                    'backbone.model.visual.merger.linear_fc2.weight','backbone.model.language_model.layers.3.self_attn.q_proj.weight']
    try:
        for ep_index,episode in enumerate(episodes):
            stop=len(episode.frames) if ep_index==0 else 2
            for start in range(0,stop,8):
                steps=tuple(range(start,min(start+8,stop)))
                segment=SimpleNamespace(episode_index=ep_index,steps=steps,first=start==0,last=steps[-1]==len(episode.frames)-1)
                policy.zero_grad(set_to_none=True)
                metrics=trainer.segment(segment,1/len(steps))
                checksums={}
                for name,p in policy.named_parameters():
                    if name in gradient_names:
                        if p.grad is None:
                            assert name=='writer.initial_slots' and not segment.first,name
                            checksums[name]=dict(unused=True,sha256=None,norm=None)
                            continue
                        assert bool(torch.isfinite(p.grad).all()),name
                        value=p.grad.detach().float().cpu().contiguous()
                        checksums[name]=dict(sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),norm=float(value.norm()))
                segments.append(dict(episode=episode.episode,steps=steps,first=segment.first,last=segment.last,
                                     detached=trainer.carry['memory'] is None or trainer.carry['memory'].grad_fn is None,
                                     gradients=checksums,**metrics))
    finally:
        for handle in handles:handle.remove()
        loader.close()
    assert any(row['frame']['step']==8 for row in rows)
    assert rows[-2]['reset'] and len(rows[-2]['reader_frames'])==1
    assert all(s['detached'] for s in segments)
    report=dict(format_version=1,export=str(Path(args.export).resolve()),K=8,M=64,R=4,reader_microbatch=4,
                precision='FP32 trainable/state, BF16 autocast, deterministic FA2',optimizer_steps=0,
                scenario='One complete real episode, then reset and first two observations of another; fixed exported weights',
                segments=segments,observations=rows)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,observations=len(rows),segments=len(segments),passed=True)))

if __name__=='__main__':main()
