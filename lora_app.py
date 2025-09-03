import torch
import torch.nn as nn
from lora_gen import EHLG
from llm import OLM

class EHLGOLM(nn.Module): #
    def __init__(self, config):
        self.olm = config.olm
        self.ehlg = config.ehlg
        # self.olm.tokenizer
        # self.ehlg.tokenizer
        self.scale_ehlg = config.scale_ehlg
        self.scale_olm = config.scale_olm
    
    def forward(self, x):
        # by default, the input to Gollem should be:
        # 1. list of indices and types for HN (can have default values)
        # 2. input for llm (data)
        
        ehlg_output = self.ehlg(x)
        # add LoRA matrix outputs to OLM
        #self.adapt(olm=self.olm, ehlg=self.ehlg)
        #self._adapt_olm(ehlg_output)
        #self._adapt_olm(self.ehlg(x))
        for k, v in ehlg_output.items():
            self.olm[k] += v
        
        olm_output = self.olm(x)
        return olm_output
    
    # def adapt(self, olm, ehlg):
    #     pass
GOHLLEM = EHLGOLM
Golem = GOHLLEM
Gollem = EHLGOLM
GOLLEM = Gollem
EHL = EHLGOLM # Embedding, Hypernetwork, LLMs
# EHL-FT-FL : Fine-tuning Models through Federated Learning of Hypernetwork Embeddings
# CEFL: Communication-Efficient Federated learning

def llm_apply_lora():
    # obtain output from LoRA-adapted OLM
    ...
    
    return None

def test_gollem_forward_pass():
    # load EHLG and OLM
    # instantiate Gollem
    # pass data through Gollem instance
    from types import SimpleNamespace
    
    layer_index = 1
    weight_type = 1
    
    # load EHLG
    ehlg_config = None
    ehlg = EHLG(ehlg_config)
    
    # obtain EHLG output LoRA matrices
    x = ehlg.make_input(torch.tensor(layer_index), torch.tensor(weight_type))
    ehlg_output = ehlg(x)
    
    # load OLM and add LoRA matrices to OLM
    olm_config = None
    olm = OLM(olm_config)
    
    gollem_config = SimpleNamespace()
    gollem_config.ehlg = ehlg
    gollem_config.olm = olm
    gollem_config.scale_ehlg = 1
    gollem_config.scale_olm = 1
    gollem = Gollem(gollem_config)
    
    gollem_output = gollem(1, 1)
    return gollem_output

# def llm_ehlg_train():
#     # train the pipeline/embeddings
#     return
def test_gollem_train():
    # Gollem training pipeline
    pass

if __name__ == "__main__":
    llm_apply_lora()
    # llm_apply_lora() to eventually be identical to instantiating Gollem
    # i.e.
    # '''
    # ...
    # model = Gollem(config)
    # output = model(x)
    # print(model.decode(output))
    # '''
    # test_gollem_train() # llm_ehlg_train()
    