from .base import BaseModel
from ..utils import *
from ..config import supported_VLM

from mmengine.config import Config

class Prism(BaseModel):
    
    def __init__(self, **kwargs):
        
        cfg = Config(kwargs)
        if 'cfg' in kwargs:
            if kwargs['cfg'] is not None:
                cfg = kwargs['cfg']
        self.cfg = cfg
        
        assert self.cfg.perception_module is not None, 'Prism requires a perception module'
        self.perception_module_name = self.cfg.perception_module
        
        self.reasoning_module_name = getattr(self.cfg, 'reasoning_module', 'chatgpt-0125')
        self.prompt_version = getattr(self.cfg, 'prompt_version', 'generic')
        
        if self.perception_module_name not in supported_VLM:
            raise ValueError(f'Perception module {self.perception_module_name} is not supported')
        # elif self.cfg is not None:
        #     self.perception_module = supported_VLM[self.perception_module_name](cfg=self.cfg)
        else:
            self.perception_module = supported_VLM[self.perception_module_name]()
            
        if self.reasoning_module_name not in gpt_version_map and self.reasoning_module_name not in reasoning_mapping:
            raise ValueError(f'Reasoning module {self.reasoning_module_name} is not supported')
        elif self.cfg is not None:
            self.reasoning_module = ReasoningWrapper(self.reasoning_module_name, cfg=self.cfg)
        else:
            self.reasoning_module = ReasoningWrapper(self.reasoning_module_name)
            
        if self.perception_module.is_api:
            self.is_api = True
        else:
            self.is_api = False
    
    def build_perception_prompt(self, question):
        
        if self.prompt_version in prompt_mapping:
            return prompt_mapping[self.prompt_version]
        
        elif self.prompt_version in ['query-specific' ,'qs']:
            qs_part = fetch_qs_part(self.reasoning_module, question)
            return merge_qs_part(qs_part)
        
        else:
            raise ValueError(f'Prompt version {self.prompt_version} is not supported')
    
    @staticmethod
    def build_infer_prompt(question, des):
        
        if not question.endswith('\n'):
            question += '\n'
        if not question.lower().startswith('question:') and not question.lower().startswith('hint:'):
            question = 'Question: ' + question 
        if not des.endswith('\n'):
            des += '\n'
        
        description = 'Description: ' + des
        role = 'You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.\n\n'
        
        prompt =  role + description + question
        return prompt
    
    def generate_inner(self, message, dataset=None):
        
        content, images = '', []
        for x in message:
            if x['type'] == 'text':
                content += x['value']
            elif x['type'] == 'image':
                content += '<image>\n'
                images.append(x['value'])

        pprompt = self.build_perception_prompt(content)
        pmessages = [[pprompt, image] for image in images]
        
        des = ''
        for i, pmessage in enumerate(pmessages):
            if len(pmessages) > 1:
                des = f'Image {i+1}:\n'
            des += self.perception_module.generate(pmessage, dataset='coco')
            
        iprompt = self.build_infer_prompt(content, des)
        response = self.reasoning_module.generate(iprompt)
        return response