"""Single-owner Qwen policy with an independent observation-only memory writer."""
import json
from pathlib import Path
import torch
from torch import nn
from qwen_vl.models.memory_writer import MemoryWriter
from qwen_vl.models.instruction_encoder import InstructionEncoder
from qwen_vl.models.memory_adapter import MemoryAdapter
from qwen_vl.models.visual_tokens import VisualTokenProjector
from qwen_vl.models.qwen_adapter import QwenVisualAdapter,QwenReaderAdapter
from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE

class NavigationPolicy(nn.Module):
    def __init__(self,backbone,tokenizer,memory_enabled=True,memory_config=None):
        super().__init__()
        self.backbone=backbone;self.tokenizer=tokenizer
        self.tokenizer.chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        self.memory_enabled=memory_enabled
        self.memory_config=dict(slots=64,width=512,layers=3,heads=8,ffn_width=2048)
        if memory_config:self.memory_config.update(memory_config)
        self.visual=QwenVisualAdapter(backbone)
        self.reader=QwenReaderAdapter(backbone,tokenizer)
        if memory_enabled:
            cfg=self.memory_config;text_width=backbone.config.text_config.hidden_size
            self.writer=MemoryWriter(**cfg)
            self.instruction_encoder=InstructionEncoder(text_width,width=cfg['width'],heads=cfg['heads'],ffn_width=cfg['ffn_width'])
            self.visual_projector=VisualTokenProjector(backbone.config.vision_config.hidden_size,width=cfg['width'],merge_size=self.visual.merge_size)
            self.memory_adapter=MemoryAdapter(text_width,width=cfg['width'])
        self.backbone.config.use_cache=False
        self.backbone.model.language_model.config.use_cache=False

    @property
    def device(self):return self.backbone.get_input_embeddings().weight.device
    @property
    def memory_slots(self):return self.memory_config['slots'] if self.memory_enabled else 0

    def encode_instruction(self,instruction):
        if not self.memory_enabled:return None
        ids=self.tokenizer.encode(instruction,add_special_tokens=False)
        if not 0<len(ids)<=512:raise ValueError('Instruction must contain 1..512 lexical tokens')
        ids=torch.tensor([ids],dtype=torch.long,device=self.device)
        embeddings=self.backbone.get_input_embeddings()(ids).detach()
        return self.instruction_encoder(embeddings,torch.ones_like(ids,dtype=torch.bool))

    def write(self,previous,feature,instruction_features):
        if not self.memory_enabled:return None
        if previous is None:previous=self.writer.initial(1,self.device)
        tokens,mask=self.visual_projector(feature.premerge,feature.grid_thw)
        instruction_mask=torch.ones(instruction_features.shape[:2],dtype=torch.bool,device=self.device)
        return self.writer(previous,tokens,instruction_features,mask,instruction_mask).state

    def export(self,path,processor,recent):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        # Native model + small modules, avoiding a duplicated multi-billion state dict.
        self.backbone.save_pretrained(path/'backbone',safe_serialization=True,max_shard_size='5GB')
        processor.tokenizer.chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        processor.chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        processor.save_pretrained(path/'backbone')
        extras={n:t.detach().cpu() for n,t in self.state_dict().items() if not n.startswith('backbone.')}
        torch.save(extras,path/'memory.pt')
        (path/'navigation_config.json').write_text(json.dumps(dict(memory_enabled=self.memory_enabled,memory=self.memory_config,recent=recent,format_version=1),indent=2)+'\n')
        (path/'prompt_protocol.json').write_text(json.dumps(dict(chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE,actions=['MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','STOP'],recent=recent,decoder='remove tokenizer special tokens; strip surrounding whitespace; require exact primitive action; invalid output raises'),indent=2)+'\n')

    @classmethod
    def from_export(cls,path,device='cpu',dtype=torch.float32,attn_implementation='sdpa'):
        from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
        path=Path(path);cfg=json.loads((path/'navigation_config.json').read_text())
        processor=AutoProcessor.from_pretrained(path/'backbone')
        backbone=Qwen3_5ForConditionalGeneration.from_pretrained(path/'backbone',dtype=dtype,attn_implementation=attn_implementation)
        policy=cls(backbone,processor.tokenizer,cfg['memory_enabled'],cfg['memory'])
        state=torch.load(path/'memory.pt',map_location='cpu',weights_only=True)
        missing,unexpected=policy.load_state_dict(state,strict=False)
        if unexpected or any(not n.startswith('backbone.') for n in missing):raise ValueError('Incompatible memory export')
        return policy.to(device).eval(),processor,cfg['recent']
