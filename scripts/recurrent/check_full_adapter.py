"""Real Qwen3.5-4B BF16/FA2 native-reader parity on real R2R observations."""
import json,os
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from check_adapter import BASE
from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.data.prompting import build_state
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.losses import navigation_loss_per_state
from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix


def run():
    os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
    torch.set_num_threads(1);torch.manual_seed(42)
    _install_qwen35_flash_attention_fix()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,dtype=torch.float32,attn_implementation='flash_attention_2',device_map={'':0})
    model.model.visual.requires_grad_(False)
    model.model.visual.bfloat16();model.model.visual.merger.float().requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    processor=AutoProcessor.from_pretrained(BASE)
    processor.image_processor.size.update(longest_edge=576*28*28,shortest_edge=16*28*28)
    policy=NavigationPolicy(model,processor.tokenizer,False)
    eps=load_manifest('/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl')
    loader=FrameLoader('/groups/yshang/an221229/data/JanusVLN_data',processor.image_processor)
    report=[]
    for counts in [(1,2),(5,9)]:
        frames=[f for ep,n in zip(eps,counts) for f in ep.frames[:n]]
        loaded=[loader.get(f) for f in frames]
        pixels=torch.cat([p for p,g in loaded]).cuda();grids=torch.stack([g for p,g in loaded]).cuda()
        states=[];at=0
        for ep,n in zip(eps,counts):
            states.append(build_state(ep,ep.frames[:n],processor.tokenizer,grids[at:at+n].tolist()))
            at+=n
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features={f.key:f for f in policy.visual.encode(pixels,grids,[f.key for f in frames])}
            kwargs,labels,native=policy.reader.prepare(states,features)
            out=model(**native,pixel_values=pixels,use_cache=False)
            native_loss=navigation_loss_per_state(out.logits,labels).mean()
        selected=['model.visual.merger.linear_fc2.weight','model.language_model.layers.3.self_attn.q_proj.weight']
        native_loss.backward()
        ref_logits=out.logits.detach();reference={n:p.grad.detach().clone() for n,p in model.named_parameters() if n in selected}
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            losses,out=policy.reader.forward(states,features)
        losses.mean().backward()
        torch.testing.assert_close(out.logits,ref_logits,atol=.02,rtol=.02)
        errors={}
        for n,p in model.named_parameters():
            if n in reference:
                a,b=p.grad.flatten().float(),reference[n].flatten().float()
                cosine=float(F.cosine_similarity(a.double(),b.double(),dim=0))
                relative=float((a.double()-b.double()).norm()/b.double().norm())
                assert cosine>.999 and relative<.02,(n,cosine,relative)
                errors[n]=dict(max_abs=float((a-b).abs().max()),cosine=cosine,relative_l2=relative)
        report.append(dict(image_counts=counts,deterministic_attention=True,max_logit_error=float((out.logits.detach()-ref_logits).abs().max()),native_loss=float(native_loss.detach()),adapter_loss=float(losses.detach().mean()),gradients=errors))
        model.zero_grad(set_to_none=True)
        del features,out,ref_logits,reference
        torch.cuda.empty_cache()
    loader.close()
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
