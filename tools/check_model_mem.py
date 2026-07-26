#!/usr/bin/env python3
"""Check model memory before training."""
import torch, sys
sys.path.insert(0, '/data3/wgq/StreamVAD/StreamMind')

def show(tag):
    a = torch.cuda.memory_allocated()/1e9
    r = torch.cuda.memory_reserved()/1e9
    print(f'[{tag}] alloc={a:.1f}GB res={r:.1f}GB')

show('start')
from transformers import BitsAndBytesConfig, AutoConfig
from streammind.model.language_model.videollama2_mistral import Videollama2MistralForCausalLM

cfg = AutoConfig.from_pretrained('/data3/wgq/models/VideoLLaMA2-7B', trust_remote_code=True)
cfg._attn_implementation = 'flash_attention_2'

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

m = Videollama2MistralForCausalLM.from_pretrained(
    '/data3/wgq/models/VideoLLaMA2-7B',
    config=cfg, quantization_config=bnb,
    torch_dtype=torch.float16, device_map={'': 0},
)
show('model loaded')
print(f'params: {sum(p.numel() for p in m.parameters())/1e9:.2f}B')

# Now load vision tower
from streammind.model import *
m.get_model().initialize_vision_modules(
    type('Args', (), {
        'vision_tower': '/data3/wgq/models/clip-vit-large-patch14-336',
        'mm_projector_type': 'stc_connector',
        'mm_vision_select_layer': -2,
        'mm_vision_select_feature': 'patch',
        'mm_use_im_start_end': False,
        'mm_use_im_patch_token': False,
    })(),
)
vt = m.get_vision_tower()
vt.to(dtype=torch.float16, device=0)
show('vision tower loaded')
print(f'VT params: {sum(p.numel() for p in vt.parameters())/1e9:.2f}B')
