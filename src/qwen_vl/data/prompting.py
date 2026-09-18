"""One exact training/inference renderer, preserving the v0 Janus wording."""
import torch
from qwen_vl.contracts import ImageSpan,TokenizedState
from qwen_vl.data.data_qwen import tokenize_conversation,_apply_chat_template,IGNORE_INDEX

USER_PROMPT=('You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task.\n This is your historical observation:{history}\n This is your current observation:<image>\n Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP.')

def build_state(episode, frames, tokenizer, grids, memory_slots=0, include_target=True, merge_size=2, max_length=12800):
    if not frames or len(frames)!=len(grids):raise ValueError('Frame/grid mismatch')
    if any(f.key.episode!=episode.episode for f in frames):raise ValueError('Cross-episode prompt')
    if [f.key.step for f in frames] != sorted(set(f.key.step for f in frames)):
        raise ValueError('History must be unique and chronological')
    action=frames[-1].action
    if include_target and action is None:raise ValueError('Cannot fabricate a target')
    counts=[]
    for grid in grids:
        t,h,w=map(int,grid)
        if t!=1 or h%merge_size or w%merge_size:raise ValueError('Unsupported image grid')
        counts.append(t*h*w//merge_size**2)
    user=USER_PROMPT.format(history='<image>'*(len(frames)-1),instruction=episode.instruction)
    ids,labels=tokenize_conversation([{'from':'human','value':user},{'from':'gpt','value':action or 'STOP'}],tokenizer,counts)
    if memory_slots:
        if memory_slots<0:raise ValueError('Negative memory length')
        at=len(_apply_chat_template(tokenizer,[{'role':'system','content':'You are a helpful assistant.'}]))
        before=tokenizer.encode('Memory state:\n',add_special_tokens=False)
        after=tokenizer.encode('\n',add_special_tokens=False)
        placeholder=tokenizer.encode('x',add_special_tokens=False)[0]
        if placeholder in tokenizer.all_special_ids:raise ValueError('Memory placeholder must be ordinary text')
        inserted=torch.tensor(before+[placeholder]*memory_slots+after,dtype=torch.long)
        ids=torch.cat((ids[:at],inserted,ids[at:]))
        labels=torch.cat((labels[:at],torch.full_like(inserted,IGNORE_INDEX),labels[at:]))
        memory_span=(at+len(before),at+len(before)+memory_slots)
    else:memory_span=None
    action_start=int(labels.ne(IGNORE_INDEX).nonzero()[0])
    if not include_target:ids,labels=ids[:action_start],labels[:action_start]
    if len(ids)>max_length:raise ValueError('Overlength prompt; no truncation allowed')
    image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
    positions=ids.eq(image_id).nonzero().flatten().tolist();spans=[];offset=0
    for frame,grid,n in zip(frames,grids,counts):
        subset=positions[offset:offset+n]
        if len(subset)!=n or subset!=list(range(subset[0],subset[0]+n)):
            raise ValueError('Image spans do not match per-image grids')
        spans.append(ImageSpan(frame.key,subset[0],subset[-1]+1,tuple(map(int,grid)),frame is frames[-1]))
        offset+=n
    if offset!=len(positions):raise ValueError('Extra image tokens')
    return TokenizedState(ids,labels,tuple(spans),memory_span,action_start,tuple(f.key for f in frames))
