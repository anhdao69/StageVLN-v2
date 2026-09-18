"""Full-checkpoint BF16/FA2 direct versus bridge K2 gradient check."""
import json,os
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from check_adapter import BASE
from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.episode_trainer import EpisodeTrainer
from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix


def run():
    os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
    torch.set_num_threads(1);torch.manual_seed(42)
    _install_qwen35_flash_attention_fix()
    backbone=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,dtype=torch.float32,attn_implementation='flash_attention_2',device_map={'':0})
    backbone.model.visual.requires_grad_(False)
    backbone.model.visual.bfloat16();backbone.model.visual.merger.float().requires_grad_(True)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    processor=AutoProcessor.from_pretrained(BASE)
    processor.image_processor.size.update(longest_edge=576*28*28,shortest_edge=16*28*28)
    policy=NavigationPolicy(backbone,processor.tokenizer,True).cuda().train()
    episodes=load_manifest('/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl')[:1]
    loader=FrameLoader('/groups/yshang/an221229/data/JanusVLN_data',processor.image_processor)
    names=['writer.initial_slots','writer.blocks.0.self_attention.in_proj_weight',
           'instruction_encoder.projection.weight','visual_projector.projection.weight',
           'memory_adapter.output_projection.weight','backbone.model.visual.merger.linear_fc2.weight',
           'backbone.model.language_model.layers.3.self_attn.q_proj.weight']
    report=[]
    for recent in (0,4):
        references={};losses=[];errors={}
        for trial,execution in enumerate(('direct','direct','bridge')):
            policy.zero_grad(set_to_none=True)
            trainer=EpisodeTrainer(policy,episodes,loader,recent,execution,1)
            result=trainer.segment(SimpleNamespace(episode_index=0,steps=(0,1),first=True,last=False),.5)
            losses.append(result['loss_sum']/2)
            for name,p in policy.named_parameters():
                if name not in names:continue
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
                if trial==0:references[name]=p.grad.detach().float().cpu().clone()
                else:
                    a,b=p.grad.detach().float().flatten().cpu(),references[name].flatten()
                    cosine=float(F.cosine_similarity(a.double(),b.double(),dim=0))
                    relative=float((a.double()-b.double()).norm()/b.double().norm())
                    assert cosine>.999 and relative<.02,(execution,name,cosine,relative)
                    errors[f'{execution}_{trial}/{name}']=dict(cosine=cosine,max_absolute=float((a-b).abs().max()),relative_l2=relative)
        assert max(losses)-min(losses)<.02,losses
        assert len(errors)==2*len(names),(len(errors),len(names))
        report.append(dict(recent=recent,K=2,direct_loss=losses[0],repeat_loss=losses[1],bridge_loss=losses[2],flash_attention_deterministic=True,gradients=errors))
    loader.close()
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
