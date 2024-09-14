import inspect

def func_with_cfg(func, params_dict):
    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())
    
    kwargs = {key: value for key, value in params_dict.items() if key in param_names}
    return func(**kwargs)

