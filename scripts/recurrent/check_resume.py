"""Actual tiny-Qwen continuation across a live episode checkpoint boundary."""
import copy
import json
from pathlib import Path
import tempfile
import torch
from transformers import AutoTokenizer
from check_adapter import tiny_model, BASE
from check_temporal_parity import Loader
from qwen_vl.contracts import EpisodeRecord, FrameRecord, FrameKey
from qwen_vl.data.episode_stream import EpisodeScheduler
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.checkpointing import save_checkpoint, load_checkpoint
from qwen_vl.train.episode_trainer import EpisodeTrainer
from qwen_vl.train.optimizer import build_optimizer


def run():
    torch.set_num_threads(1); torch.manual_seed(18)
    tokenizer=AutoTokenizer.from_pretrained(BASE)
    base=NavigationPolicy(tiny_model(),tokenizer,True,dict(slots=4,width=16,heads=2,ffn_width=32,layers=3))
    frames=tuple(FrameRecord(FrameKey('r2r','0',i),'unused','TURN_LEFT' if i<11 else 'STOP',str(i)) for i in range(12))
    episodes=(EpisodeRecord('0','Turn left at the door.','',frames),)
    def setup(policy):
        trainer=EpisodeTrainer(policy,episodes,Loader(),recent=4,reader_microbatch=2,bf16=False)
        stream=EpisodeScheduler(episodes,K=8,target_budget=8)
        optimizer=build_optimizer(policy)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:1.)
        return trainer,stream,optimizer,scheduler
    live=setup(base)
    def update(parts):
        trainer,stream,optimizer,scheduler=parts
        schedule=stream.plan_update(); optimizer.zero_grad(set_to_none=True)
        loss=sum(trainer.segment(s,1/schedule.global_labels)['loss_sum'] for s in schedule.local_segments)/schedule.global_labels
        optimizer.step();scheduler.step()
        return loss
    loss1=update(live)
    with tempfile.TemporaryDirectory(prefix='stagevln-resume-') as directory:
        trainer,stream,optimizer,scheduler=live
        save_checkpoint(directory,base,optimizer,scheduler,stream,trainer.state_dict(),{'test':'actual-Qwen-R4'},1)
        loss2=update(live)
        resumed=setup(copy.deepcopy(base))
        other,other_stream,other_opt,other_lr=resumed
        restored=load_checkpoint(directory,other.policy,other_opt,other_lr,other_stream,expected_manifest={'test':'actual-Qwen-R4'})
        other.load_state_dict(restored['cursors'])
        repeated=update(resumed)
        assert loss2==repeated,(loss2,repeated)
        max_parameter_error=0.;max_adam_error=0.
        for lhs,rhs in zip(base.parameters(),other.policy.parameters()):
            # Re-encoding retained frozen frames in a different image batch can
            # change the last FP32 bits. No trainable cache is serialized.
            torch.testing.assert_close(lhs,rhs,rtol=1e-5,atol=1e-7)
            max_parameter_error=max(max_parameter_error,float((lhs.detach()-rhs.detach()).abs().max()))
        for lhs,rhs in zip(optimizer.state.values(),other_opt.state.values()):
            for key in lhs:
                torch.testing.assert_close(lhs[key],rhs[key],rtol=1e-5,atol=1e-7)
                max_adam_error=max(max_adam_error,float((lhs[key]-rhs[key]).abs().max()))
        assert stream.state_dict()==other_stream.state_dict()
        assert trainer.carry==other.carry  # both end at the episode boundary
    print(json.dumps(dict(first_loss=loss1,next_loss=loss2,resumed_loss=repeated,max_parameter_error=max_parameter_error,max_adam_error=max_adam_error,exact_cursor=True,precision='FP32',recent=4),indent=2))

if __name__=='__main__':run()
