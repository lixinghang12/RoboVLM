import transformers
import copy
import torch

# from llava.model.language_model.llava_mpt import LlavaMptForCausalLM
# from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# from llava import conversation as conversation_lib
# from llava.mm_utils import get_model_name_from_path
# from llava.model.builder import load_pretrained_model

LLAVA1_6_CFG = {
    'VICUNA7b': {
        'pretrained_model_base': None,
        'pretrained_model_name_or_path': '/mnt/bn/robotics-data-lxh-lq/LLaVA/llava-v1.6-vicuna-7b',  
    },

    'VICUNA13b': {
        'pretrained_model_base': None,
        'pretrained_model_name_or_path': 'liuhaotian/llava-v1.6-vicuna-13b'
    },

    'MISTRAL7b': {
        'pretrained_model_base': None,
        'pretrained_model_name_or_path': '/mnt/bn/robotics-data-lxh-lq/LLaVA/llava-v1.6-mistral-7b'
    },

    'YI34b': {
        'pretrained_model_base': None,
        'pretrained_model_name_or_path': 'liuhaotian/llava-v1.6-34b'
    },

    'V1.57b': {
        'pretrained_model_base': None,
        'pretrained_model_name_or_path': '/mnt/bn/robotics-data-lxh-lq/LLaVA/llava-v1.5-7b'
    },

    "MPT7b":{
        'pretrained_model_base': None,
        'pretrained_model_name_or_path': '/mnt/bn/robotics-data-lxh-lq/LLaVA/LLaVA-Lightning-MPT-7B-preview'
    }
}

def default_llava_cfgs(llm_name):
    llava_cfg = {}

    llava_cfg['llava1.6_vicuna_7b'] = LLAVA1_6_CFG['VICUNA7b']
    llava_cfg['llava1.6_vicuna_13b'] = LLAVA1_6_CFG['VICUNA13b']

    llava_cfg['llava1.6_mistral_7b'] = LLAVA1_6_CFG['MISTRAL7b']
    llava_cfg['llava1.6_yi_34b'] = LLAVA1_6_CFG['YI34b']
    
    llava_cfg['llava-v1.5-7b'] = LLAVA1_6_CFG['V1.57b']
    llava_cfg['llava-mpt-7b'] = LLAVA1_6_CFG['MPT7b']
    # TODO: I think we may only have to test llava 1.6
    # llava_cfg['llava1.5_vicuna_7b'] = LLAVA1_5_CFG['VICUNA7b']
    # llava_cfg['llava1.5_vicuna_13b'] = LLAVA1_5_CFG['VICUNA13b']

    assert llm_name in llava_cfg, f'Unknown llm_name: {llm_name}'
    return llava_cfg[llm_name]

def build_llava(llava_config, precision='bf16'):
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path
    # disable_torch_init()
    if isinstance(llava_config, str):
        llava_config = default_llava_cfgs(llava_config)
    
    llava_config = copy.deepcopy(llava_config)
    model_base = llava_config.pop('pretrained_model_base') # default is None
    model_path = llava_config.pop('pretrained_model_name_or_path')
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, _, __ = load_pretrained_model(model_path, model_base, model_name, use_flash_attn=('16' in precision and 'mpt' not in model_name.lower()), device_map="cpu")
    # import pdb; pdb.set_trace()
    return tokenizer, model

if __name__ == "__main__":
    llava_config = "llava1.6_vicuna_7b"
    llava_config = "llava-v1.5-7b"
    # llava_config = "llava-mpt-7b"
    # llava_config = "llava1.6_mistral_7b"
    tokenizer, model = build_llava(llava_config)
    import pdb; pdb.set_trace()