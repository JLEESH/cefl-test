import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from types import SimpleNamespace

DEFAULT_OLM_PATH="../models/open_llama_3b_v2"

class OpenLLaMAv2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = config.model
        self.llama = self.model
        self.llamav2 = self.model
        self.tokenizer = config.tokenizer
    
    def forward(self, x):
        return self.model(x)
    
    default_config_dict = {
        'model'     :   LlamaForCausalLM.from_pretrained(DEFAULT_OLM_PATH),
        'tokenizer' :   LlamaTokenizer.from_pretrained(DEFAULT_OLM_PATH),
    }
    default_config = SimpleNamespace(default_config_dict)
# Alias:
# (can also be understood to mean Open Language Model, Original Language Model, etc.)
OLM = OpenLLaMAv2Model

def llm_input_test(args_list=None, prompt=None):
    import argparse
    from types import SimpleNamespace
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', '-p', default=None)
    parser.add_argument('--mode', '-m', default=None)
    args = parser.parse_args(args_list)
    model_path = args.model_path
    
    if model_path is None:
        model_path = DEFAULT_OLM_PATH
        #raise ValueError(f"``model_path`` is None. Please pass a directory.")
    
    # configure & create OLM model
    config = SimpleNamespace()
    config.model = LlamaForCausalLM.from_pretrained(model_path)
    config.tokenizer = LlamaTokenizer.from_pretrained(model_path)
    llm_model = OpenLLaMAv2Model(config)
    
    # test prompts
    if prompt is None:
        prompt = 'Q: What is the largest animal?\nA:'
    input_ids = llm_model.tokenizer(prompt, return_tensors="pt").input_ids
    
    generation_output = llm_model.model.generate(
        input_ids=input_ids, max_new_tokens=32
    )
    
    ret_out = llm_model.tokenizer.decode(generation_output[0])
    #print(ret_out)
    #input()
    return ret_out

if __name__ == '__main__':
    print(llm_input_test())
    print()
    print(llm_input_test(prompt="Make up a new word and explain what it means. The word and its meaning are: "))
    