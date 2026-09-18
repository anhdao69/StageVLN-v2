"""Native Qwen visual/reader calls with explicit spans and no duplicate ownership."""
import torch
from torch.nn.utils.rnn import pad_sequence
from qwen_vl.contracts import FrameFeatures
from qwen_vl.train.losses import navigation_loss_per_state
from qwen_vl.models.visual_tokens import premerge_to_raster

class QwenVisualAdapter:
    def __init__(self, backbone):
        self.backbone=backbone  # Plain helper, not a registered nn.Module.
        self.merge_size=backbone.config.vision_config.spatial_merge_size

    def encode(self,pixel_values,image_grid_thw,keys):
        if len(keys)!=len(image_grid_thw):raise ValueError('Frame/grid count mismatch')
        grids=[tuple(map(int,g)) for g in image_grid_thw.tolist()]
        if any(t!=1 or h%self.merge_size or w%self.merge_size for t,h,w in grids):
            raise ValueError('Only independent image grids are supported')
        visual=self.backbone.model.visual
        output=visual(pixel_values.to(dtype=visual.dtype),grid_thw=image_grid_thw,return_dict=True)
        raw_counts=[t*h*w for t,h,w in grids]
        raw=output.last_hidden_state.split(raw_counts)
        merged=output.pooler_output.split([n//self.merge_size**2 for n in raw_counts])
        return [FrameFeatures(key,r.detach(),grid,m) for key,r,grid,m in zip(keys,raw,grids,merged)]

class QwenReaderAdapter:
    def __init__(self,backbone,tokenizer):
        self.backbone=backbone
        self.tokenizer=tokenizer

    def prepare(self,states,features_by_key,memories=None,supervised=True):
        device=self.backbone.get_input_embeddings().weight.device
        memories=[None]*len(states) if memories is None else memories
        if len(memories)!=len(states):raise ValueError('Memory/state count mismatch')
        ids=pad_sequence([s.input_ids for s in states],batch_first=True,padding_value=self.tokenizer.pad_token_id).to(device)
        labels=pad_sequence([s.labels for s in states],batch_first=True,padding_value=-100).to(device)
        lengths=torch.tensor([len(s.input_ids) for s in states],device=device)
        mask=torch.arange(ids.shape[1],device=device)[None,:]<lengths[:,None]
        mm=torch.zeros_like(ids,dtype=torch.int32)
        embeddings=self.backbone.get_input_embeddings()(ids)
        # Each replacement is differentiable to its unique native merger/memory.
        rows=[];grids=[]
        for i,(state,memory) in enumerate(zip(states,memories)):
            row=embeddings[i]
            for span in state.image_spans:
                feature=features_by_key[span.key]
                if feature.grid_thw!=span.grid_thw or feature.merged is None or len(feature.merged)!=span.stop-span.start:
                    raise ValueError('Per-image feature/span mismatch')
                positions=torch.arange(span.start,span.stop,device=device)
                row=row.index_copy(0,positions,feature.merged.to(device=device,dtype=row.dtype))
                mm[i,span.start:span.stop]=1;grids.append(span.grid_thw)
            if state.memory_span is not None:
                a,b=state.memory_span
                if memory is None or memory.shape!=(b-a,row.shape[-1]):raise ValueError('Memory span mismatch')
                row=row.index_copy(0,torch.arange(a,b,device=device),memory.to(device=device,dtype=row.dtype))
            elif memory is not None:raise ValueError('Memory supplied without span')
            rows.append(row)
        grids=torch.tensor(grids,dtype=torch.long,device=device)
        positions,_=self.backbone.model.get_rope_index(ids,mm,image_grid_thw=grids,attention_mask=mask)
        inputs=dict(inputs_embeds=torch.stack(rows),attention_mask=mask,position_ids=positions)
        native=dict(input_ids=ids,attention_mask=mask,mm_token_type_ids=mm,image_grid_thw=grids)
        if supervised:
            prediction_positions=labels[:,1:].ne(-100).any(0).nonzero().flatten()
            if len(prediction_positions)==0:raise ValueError('No supervised target')
            keep=torch.cat((prediction_positions,prediction_positions.new_tensor([ids.shape[1]-1])))
            labels=torch.cat((labels.new_full((len(states),1),-100),labels[:,prediction_positions+1]),1)
            inputs['logits_to_keep']=keep;native['logits_to_keep']=keep
        else:inputs['logits_to_keep']=1;native['logits_to_keep']=1
        return inputs,labels,native

    def forward(self,states,features_by_key,memories=None):
        inputs,labels,_=self.prepare(states,features_by_key,memories)
        outputs=self.backbone(**inputs,use_cache=False,return_dict=True)
        return navigation_loss_per_state(outputs.logits,labels),outputs
