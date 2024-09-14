import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
import torch


class Chameleon(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='facebook/chameleon-7b', **kwargs):
        try:
            from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
        except:
            warnings.warn('Please install the latest transformers.')
            
        if 'cfg' in kwargs:
            cfg = kwargs['cfg']
            if cfg['model_params'] is not None:
                self.model_params = cfg['model_params']
            else:
                self.model_params = 'default'

        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        self.model = model.cuda().eval()
        self.processor = processor

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for x in message:
            if x['type'] == 'text':
                content += x['value']
            elif x['type'] == 'image':
                content += '<image>\n'
                images.append(Image.open(x['value']))

        inputs = self.processor(
            text=[content],
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(device='cuda', dtype=torch.bfloat16)
        
        if self.params == 'default':
            generate_ids = self.model.generate(**inputs, max_new_tokens=512)
        else:
            generate_ids = self.model.generate(**inputs, **self.model_params)
            
        input_token_len = inputs.input_ids.shape[1]
        text = self.processor.batch_decode(
            generate_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return text
