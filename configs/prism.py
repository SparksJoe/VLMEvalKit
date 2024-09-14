# VLMEvalkit Config 
model = ['prism']
data = ['MMStar']
nproc = 16
work_dir = './outputs'
ignore = 1

# Prism Config
perception_module = 'GPT4o'
reasoning_module = 'chatgpt-0125'
prompt_version = 'generic'
postproc = 1

# Model Parameters
model_params = {
    'temperature': 0.5,
    'top_k': 50,
    'verbose': 1,
}

# Judge
verbose = 1
retry = 1