"""Actual Qwen API parity; tiny weights exercise image layout, padding and gradients."""
import json,copy,hashlib
from pathlib import Path
import torch
from transformers import AutoConfig,AutoTokenizer,Qwen3_5ForConditionalGeneration
from qwen_vl.contracts import EpisodeRecord,FrameRecord,FrameKey
from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE
from qwen_vl.data.prompting import build_state
from qwen_vl.models.qwen_adapter import QwenVisualAdapter,QwenReaderAdapter
from qwen_vl.train.losses import navigation_loss_per_state

BASE='/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

def tiny_model(device='cpu'):
    cfg=AutoConfig.from_pretrained(BASE)
    t=cfg.text_config
    t.hidden_size=64;t.intermediate_size=128;t.num_hidden_layers=2;t.num_attention_heads=2;t.num_key_value_heads=1;t.head_dim=32
    t.linear_key_head_dim=16;t.linear_value_head_dim=16;t.linear_num_key_heads=2;t.linear_num_value_heads=2
    t.layer_types=['linear_attention','full_attention'];t.rope_parameters['mrope_section']=[2,1,1]
    v=cfg.vision_config
    v.hidden_size=32;v.intermediate_size=64;v.num_heads=4;v.depth=1;v.out_hidden_size=64
    cfg._attn_implementation='sdpa'
    if str(device)=='cpu':
        from unittest.mock import patch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
        # Use the official CPU reference functions in a CUDA-enabled install.
        with patch.multiple(qm,causal_conv1d_fn=None,causal_conv1d_update=None,
                            chunk_gated_delta_rule=None,fused_recurrent_gated_delta_rule=None,
                            FusedRMSNormGated=None):
            model=Qwen3_5ForConditionalGeneration(cfg).to(device)
    else:
        model=Qwen3_5ForConditionalGeneration(cfg).to(device)
    # The source config carries nested BF16 dtype defaults. Our reference
    # contract requires FP32 trainable parameters, including in tiny tests.
    model.float()
    model.model.visual.requires_grad_(False);model.model.visual.merger.requires_grad_(True)
    return model

def run():
    torch.manual_seed(5);torch.set_num_threads(1)
    model=tiny_model();model.eval()
    tok=AutoTokenizer.from_pretrained(BASE);tok.chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE
    visual=QwenVisualAdapter(model);reader=QwenReaderAdapter(model,tok);results=[]
    for counts in [(1,2),(5,9)]:
        keys=[];grids=[];states=[]
        for b,n in enumerate(counts):
            frames=tuple(FrameRecord(FrameKey('r2r',str(b),i),'unused','TURN_LEFT',str(i)) for i in range(n))
            ep=EpisodeRecord(str(b),'Turn left.'+' Find the door.'*b,'',frames)
            gs=[(1,4,6) if i%2==0 else (1,6,4) for i in range(n)]
            states.append(build_state(ep,frames,tok,gs));keys.extend(f.key for f in frames);grids.extend(gs)
            prefix=build_state(ep,frames,tok,gs,include_target=False)
            assert torch.equal(states[-1].input_ids[:states[-1].action_start],prefix.input_ids)
        grids=torch.tensor(grids)
        pixels=torch.randn(int(grids.prod(-1).sum()),3*2*16*16)
        features={f.key:f for f in visual.encode(pixels,grids,keys)}
        kwargs,labels,native=reader.prepare(states,features)
        selected=['model.visual.merger.linear_fc2.weight','model.language_model.layers.1.self_attn.q_proj.weight']
        out=model(**native,pixel_values=pixels,use_cache=False)
        native_loss=navigation_loss_per_state(out.logits,labels).mean();native_loss.backward()
        ref_logits=out.logits.detach();ref_grad={n:p.grad.clone() for n,p in model.named_parameters() if n in selected}
        model.zero_grad(set_to_none=True)
        losses,out=reader.forward(states,features);losses.mean().backward()
        torch.testing.assert_close(out.logits,ref_logits,atol=1e-5,rtol=1e-4)
        errors={}
        for n,p in model.named_parameters():
            if n in ref_grad:
                torch.testing.assert_close(p.grad,ref_grad[n],atol=1e-5,rtol=1e-4)
                errors[n]=float((p.grad-ref_grad[n]).abs().max())
        results.append(dict(image_counts=counts,max_logit_error=float((out.logits.detach()-ref_logits).abs().max()),gradient_max_errors=errors))
        model.zero_grad(set_to_none=True)
    print(json.dumps({'adapter_parity':results},indent=2))

if __name__=='__main__':run()
