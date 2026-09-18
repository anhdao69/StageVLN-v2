"""Chronological K-observation training, with direct and exact bridged execution."""
from contextlib import nullcontext
from dataclasses import replace
import torch
from qwen_vl.contracts import FrameFeatures
from qwen_vl.data.prompting import build_state
from qwen_vl.data.history import history_indices
from qwen_vl.train.gradient_bridge import GradientBridge

class EpisodeTrainer:
    def __init__(self,policy,episodes,loader,recent=0,execution='bridge',reader_microbatch=1,bf16=True):
        if recent not in (0,4):raise ValueError('v2/v3 support R0/R4 only')
        if execution not in ('direct','bridge'):raise ValueError(execution)
        if type(reader_microbatch) is not int or reader_microbatch<1:raise ValueError('Reader microbatch must be a positive integer')
        if execution=='direct' and reader_microbatch!=1:raise ValueError('Direct reference uses reader batch1')
        self.policy=policy;self.episodes=episodes;self.loader=loader
        self.recent=recent;self.execution=execution;self.reader_microbatch=reader_microbatch;self.bf16=bf16
        self.carry={'episode':None,'memory':None,'recent_keys':[]}
        self.raw_cache={}
    def autocast(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.bf16 and self.policy.device.type=='cuda' else nullcontext()
    def _frames(self,episode,steps):
        indices=sorted({j for t in steps for j in history_indices(t,'recent',self.recent)})
        return [episode.frames[j] for j in indices]
    def prefetch(self,segments):
        self.loader.prefetch([f for s in segments for f in self._frames(self.episodes[s.episode_index],s.steps)])
    def _features(self,frames):
        policy=self.policy;features={};new=[]
        for frame in frames:
            old=self.raw_cache.get(frame.key)
            if old is None:new.append(frame)
            else:
                merged=policy.backbone.model.visual.merger(old.premerge)
                features[frame.key]=replace(old,merged=merged)
        if new:
            loaded=[self.loader.get(f) for f in new]
            pixels=torch.cat([p for p,g in loaded]).to(policy.device)
            grids=torch.stack([g for p,g in loaded]).to(policy.device)
            features.update({f.key:f for f in policy.visual.encode(pixels,grids,[f.key for f in new])})
        return features
    def segment(self,segment,scale):
        """Backward one live segment; no optimizer operation occurs here."""
        policy=self.policy;episode=self.episodes[segment.episode_index]
        if segment.first:
            self.carry={'episode':episode.episode,'memory':None,'recent_keys':[]};self.raw_cache={}
        elif self.carry['episode']!=episode.episode:
            raise ValueError('Missing episode state: streams must start at step0 or resume')
        frames=self._frames(episode,segment.steps)
        # No target in this segment means there is no temporal loss before its
        # detach boundary. Advance forward values without retaining a graph.
        has_loss=bool(scale) and any(episode.frames[t].action is not None for t in segment.steps)
        with torch.set_grad_enabled(has_loss), self.autocast():
            features=self._features(frames)
            instruction=policy.encode_instruction(episode.instruction)
            memory=self.carry['memory'];states=[];adapted=[]
            for t in segment.steps:
                frame=episode.frames[t]
                memory=policy.write(memory,features[frame.key],instruction)
                if frame.action is not None:
                    selected=[episode.frames[j] for j in history_indices(t,'recent',self.recent)]
                    grids=[features[f.key].grid_thw for f in selected]
                    states.append(build_state(episode,selected,policy.tokenizer,grids,policy.memory_slots,merge_size=policy.visual.merge_size))
                    adapted.append(policy.memory_adapter(memory)[0] if policy.memory_enabled else None)
        losses=[]
        if states and scale:
            if self.execution=='direct':
                for state,m in zip(states,adapted):
                    with self.autocast():per_state,_=policy.reader.forward([state],features,[m])
                    losses.append(per_state.sum())
                (torch.stack(losses).sum()*scale).backward()
                loss_sum=sum(float(x.detach()) for x in losses)
            else:
                bridge=GradientBridge()
                leaves={key:replace(f,merged=bridge.leaf(f.merged)) for key,f in features.items()}
                memory_leaves=[bridge.leaf(m) if m is not None else None for m in adapted]
                loss_sum=0.
                for start in range(0,len(states),self.reader_microbatch):
                    with self.autocast():
                        per_state,_=policy.reader.forward(states[start:start+self.reader_microbatch],leaves,memory_leaves[start:start+self.reader_microbatch])
                        loss=per_state.sum()*scale
                    loss.backward();loss_sum+=float(per_state.detach().sum())
                bridge.backward()
        else:loss_sum=0.
        # A segment boundary cuts temporal gradients, never forward values.
        t=segment.steps[-1]
        recent_frames=episode.frames[max(0,t-self.recent+1):t+1] if self.recent else ()
        self.carry={'episode':episode.episode,'memory':memory.detach() if memory is not None else None,'recent_keys':[f.key for f in recent_frames]}
        self.raw_cache={f.key:replace(features[f.key],merged=None) for f in recent_frames}
        if segment.last:
            self.carry={'episode':None,'memory':None,'recent_keys':[]};self.raw_cache={}
        return {'loss_sum':loss_sum,'states':len(states),'observations':len(segment.steps)}
    def state_dict(self):
        return {'episode':self.carry['episode'],'memory':None if self.carry['memory'] is None else self.carry['memory'].detach().cpu().float(),'recent_keys':self.carry['recent_keys']}
    def load_state_dict(self,state):
        self.carry=dict(state)
        if self.carry['memory'] is not None:self.carry['memory']=self.carry['memory'].to(self.policy.device)
        self.raw_cache={}
