"""Real-tokenizer training/generation-prefix and saved-template parity."""
import tempfile
import torch
from transformers import AutoTokenizer
from qwen_vl.data.data_qwen import tokenize_conversation,QWEN3_5_NON_THINKING_CHAT_TEMPLATE
path='/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
tok=AutoTokenizer.from_pretrained(path);tok.chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE
for n in [1,5,9]:
 user='History and current: '+'<image>'*n+'\nGo to the door.'
 expanded='History and current: '+('<|vision_start|>'+'<|image_pad|>'*4+'<|vision_end|>')*n+'\nGo to the door.'
 msgs=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':expanded}]
 pref=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
 if hasattr(pref,'keys'):pref=pref['input_ids']
 for action in ['STOP','MOVE_FORWARD','TURN_LEFT','TURN_RIGHT']:
  ids,labels=tokenize_conversation([{'from':'human','value':user},{'from':'gpt','value':action}],tok,[4]*n)
  start=int(labels.ne(-100).nonzero()[0]);assert ids[:start].tolist()==pref
  assert tok.decode(labels[start:])==action+'<|im_end|>\n'
 with tempfile.TemporaryDirectory() as d:
  tok.save_pretrained(d);again=AutoTokenizer.from_pretrained(d)
  assert again.chat_template==tok.chat_template
print('PASS: all four actions x 1/5/9 images, exact generation prefix and save/reload template')
