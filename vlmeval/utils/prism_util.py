# all config & utils for prism
import os

from ..api import OpenAIWrapper

# remap the gpt model name
gpt_version_map = {
    'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
    'gpt-4-0125': 'gpt-4-0125-preview',
    'gpt-4-turbo': 'gpt-4-1106-preview', 
    'gpt-4-0613': 'gpt-4-0613',
    'chatgpt-1106': 'gpt-3.5-turbo-1106',
    'chatgpt-0613': 'gpt-3.5-turbo-0613',
    'chatgpt-0125': 'gpt-3.5-turbo-0125',
    'gpt-4o': 'gpt-4o-2024-05-13'
}

# map the model name to the api type
reasoning_mapping = {
    'llama3-70b-chat':'vllm',
    'Mixtral-8x22B-chat':'vllm',
    'deepseek-chat':'deepseek',
}

# stop_tokens for deploying vllm
stop_tokens = {
    'llama3-70b-chat': ["<|eot_id|>"],
}

# prompt_mapping
prompt_human1 = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present.'
prompt_human2 = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, background and any text present. Please skip generating statements for non-existent contents and describe all you see. '
prompt_gpt1 = 'Given the image below, please provide a detailed description of what you see.'
prompt_gpt2 = 'Analyze the image below and describe the main elements and their relationship.'
prompt_cot = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present. Let\'s think step by step.'
prompt_decompose = 'Decompose the image into several parts and describe the fine-grained content of the image part by part, including scenes, objects, relationships, instance location, and any text present.'

prompt_mapping = {
    'generic':prompt_human1,
    'human1':prompt_human1,
    'gpt1':prompt_gpt1,
    'gpt2':prompt_gpt2,
    'human2':prompt_human2,
    'cot': prompt_cot,
    'decompose': prompt_decompose,
}

def fetch_qs_part(reasoning_module, question):
    qs_example =  '''
    Question: In which period the number of full time employees is the maximum?
    Contents to observe: the number of full time employees
    Question: What is the value of the smallest bar?
    Contents to observe: the heights of all bars and their values
    Question: What is the main subject of the image?
    Contents to observe: the central theme or object
    Question: What is the position of the catcher relative to the home plate?
    Contents to observe: the spatial arrangement of the objects 
    Question: What is the expected ratio of offspring with white spots to offspring with solid coloring? Choose the most likely ratio.
    Contents to observe: the genetic information
    '''
    task_prompt = f'Your task is to give an concise instruction about what basic elements are needed to be described based on the given question. Ensure that your instructions do not cover the raw question, options or thought process of answering the question.\n'
    prompt = task_prompt + qs_example + 'Question: ' + question + '\nContents to observe: ' 
    res = reasoning_module.generate(prompt)
    return res

def merge_qs_part(qs_part):
    generic_prompt = prompt_mapping['generic']
    prompt = generic_prompt + 'Especially, pay attention to ' + qs_part
    if not prompt.endswith('.'):
        prompt += '.'
    return prompt

class ReasoningWrapper:
    def __init__(self, model_name, **kwargs):
        
        self.deepseek_api_base = 'https://api.deepseek.com/v1/chat/completions'

        # server settings of vllm
        self.PORT = 8080
        self.vllm_api_base = f'http://localhost:{self.PORT}/v1/chat/completions'

        default_params = {
            'max_tokens': 512,
            'verbose': False,
            'retry': 5
        }
        self.llm_params = default_params
        self.llm_params.update(kwargs)
        
        if 'cfg' in kwargs:
            cfg = kwargs['cfg']
            if getattr(cfg, 'model_params', None) is not None:
                self.llm_params.update(cfg['model_params'])
        max_tokens = self.llm_params['max_tokens']
        verbose = self.llm_params['verbose']
        retry = self.llm_params['retry']
        
        if model_name in gpt_version_map:
            gpt_version = gpt_version_map[model_name]
            model = OpenAIWrapper(gpt_version, max_tokens=max_tokens, verbose=verbose, retry=retry)
        
        elif reasoning_mapping[model_name] == 'vllm':
            model = OpenAIWrapper(model_name, api_base=self.vllm_api_base, max_tokens=max_tokens, verbose=verbose, retry=retry, system_prompt='You are a helpful assistant.', **self.llm_params, stop=stop_tokens[model_name])
            
        elif reasoning_mapping[model_name] == 'deepseek':
            deepseek_key = os.environ['DEEPSEEK_API_KEY']
            model = OpenAIWrapper(model_name, api_base=self.deepseek_api_base, key=deepseek_key, max_tokens=max_tokens, verbose=verbose, retry=retry, system_prompt='You are a helpful assistant.', **self.llm_params)
        else:
            raise ValueError(f'Model {model_name} is not supported')
        
        self.model = model

    def generate(self, prompt, **kwargs):
        response = self.model.generate(prompt, **kwargs)
        return response