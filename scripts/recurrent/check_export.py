"""Reload actual export, check every saved tensor, actor actions and memory probes."""
import argparse
import json
import os
from pathlib import Path
import torch
from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.data.history import history_indices
from qwen_vl.data.prompting import build_state
from qwen_vl.eval.memory_probes import MemoryProbe,MemorySnapshot
from qwen_vl.eval.session import PolicySession
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run)
    os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
    torch.set_num_threads(1)
    _install_qwen35_flash_attention_fix()
    policy,processor,recent=NavigationPolicy.from_export(run/'export',device='cuda',dtype=torch.float32,attn_implementation='flash_attention_2')
    reference=torch.load(run/f'checkpoint-{args.step}'/'model.pt',map_location='cpu',weights_only=True)
    tensors=0
    for name,value in policy.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(),reference[name].to(dtype=value.dtype),rtol=0,atol=0)
        tensors+=1
    del reference
    assert policy.backbone.get_input_embeddings().weight is policy.backbone.get_output_embeddings().weight
    loader=FrameLoader('/groups/yshang/an221229/data/JanusVLN_data',processor.image_processor)
    episodes=load_manifest('/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl')[:2]
    actor=PolicySession(policy,loader,recent=recent)
    actor.reset(episodes[0].episode,episodes[0].instruction)
    actions=[]
    for t in range(3):
        action=actor.observe(t,episodes[0].frames[t]);actions.append(action)
        before=actor.memory.clone()
        assert actor.observe(t,episodes[0].frames[t])==action
        assert torch.equal(before,actor.memory)
    def feature(frame):
        pixels,grid=loader.get(frame)
        return policy.visual.encode(pixels.cuda(),grid.unsqueeze(0).cuda(),[frame.key])[0]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        other=episodes[1]
        source=policy.write(None,feature(other.frames[0]),policy.encode_instruction(other.instruction))
        snapshot=MemorySnapshot.capture(other.episode,0,source)
        protocols=[MemoryProbe('baseline'),MemoryProbe('reset_at_5',reset_steps={5}),
                   MemoryProbe('freeze_after_4',freeze_after=4),MemoryProbe('replace_from_observed_episode',replacements={5:snapshot}),
                   MemoryProbe('remove_recent_cross_protocol',remove_recent=True)]
        episode=episodes[0];instruction=policy.encode_instruction(episode.instruction)
        memories={p.label:None for p in protocols};rows=[];features={}
        for t in range(10):
            current=episode.frames[t];features[current.key]=feature(current)
            frames=tuple(episode.frames[j] for j in history_indices(t,'recent',recent))
            for probe in protocols:
                result=probe.step(episode.episode,t,memories[probe.label],frames,
                                  lambda previous:policy.write(previous,features[current.key],instruction),device=policy.device)
                memories[probe.label]=result.state
                state=build_state(episode,result.frames,policy.tokenizer,[features[f.key].grid_thw for f in result.frames],policy.memory_slots)
                losses,_=policy.reader.forward([state],features,[policy.memory_adapter(result.state)[0]])
                assert bool(torch.isfinite(losses).all())
                rows.append(dict(protocol=probe.label,episode=episode.episode,step=t,loss=float(losses[0]),
                                 memory_norm=float(result.state.norm()),reader_frames=[f.key.step for f in result.frames],events=result.events))
            features={f.key:features[f.key] for f in frames}
    loader.close()
    report=dict(export=str(run/'export'),checkpoint_step=args.step,exact_saved_tensors=tensors,
                tied_embeddings_preserved=True,actor_actions=actions,actor_duplicate_idempotence=True,
                probe_observations=rows,
                limitation='Open-loop teacher-forced dependence diagnostics after a short smoke; no navigation benefit or simulator result.')
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,exact_saved_tensors=tensors,actor_actions=actions,output=args.output)))

if __name__=='__main__':main()
