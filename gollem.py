import torch
import torch.nn as nn
from ehlg import EHLG
from olm import OLM
from types import SimpleNamespace

# EHL-FT-FL : Fine-tuning Models through Federated Learning of Hypernetwork Embeddings
# CEFL: Communication-Efficient Federated learning

class EHLGOLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.olm = config.olm
        self.ehlg = config.ehlg
        # self.olm.tokenizer
        # self.ehlg.tokenizer
        self.scale_ehlg = config.scale_ehlg
        self.scale_olm = config.scale_olm
        self.nlayers_olm = config.nlayers_olm
        self.ntypes = config.ntypes
    
    def forward(self, prompt, layer_indices=None, weight_types=None):
        # by default, the input to Gollem should be:
        # 1. list of indices and types for HN (can have default values)
        # 2. input for llm (data)
        
        # train all indices by default
        if layer_indices is None:
            layer_indices = [n for n in range(self.nlayers_olm)]
        if weight_types is None:
            weight_types = [[i for i in range(self.ntypes)] for _ in range(len(layer_indices))]
        if len(layer_indices) != len(weight_types):
            raise ValueError(
                "``EHLGOLM.forward()``: the lengths of ``layer_indices`` and ``weight_types`` do not match."
            )
        
        ehlg_output_ab_dict = self.ehlg(layer_indices, weight_types, data_emb=None)
        ehlg_output = ehlg_output_ab_dict
        
        # add LoRA matrix outputs to OLM
        self.adapt_olm(lora_dict=ehlg_output_ab_dict)
        
        #self.adapt(olm=self.olm, ehlg=self.ehlg)
        #self._adapt_olm(ehlg_output)
        #self._adapt_olm(self.ehlg(x))
        # for k, v in ehlg_output.items():
        #     self.olm[k] += v
        #olm_output = self.olm(x)
        #return olm_output
        
        # pass prompt to OLM
        enc_olm = self.olm.tokenizer(prompt, return_tensors="pt")
        #out_olm = self.olm.model.generate(input_ids=enc_olm.input_ids, max_new_tokens=128)
        out_olm = self.olm.model.generate(input_ids=enc_olm.input_ids, max_new_tokens=12)
        dec_olm = self.olm.tokenizer.decode(out_olm[0])
        
        # return decoded text with the raw output
        return [dec_olm, out_olm]
    
    def adapt_olm(self, lora_dict):
        return self.olm
    
    default_config_dict = {
        'ehlg'          :   EHLG(EHLG.default_config),
        'olm'           :   OLM(OLM.default_config),
        'scale_ehlg'    :   1.0,
        'scale_olm'     :   1.0,
        'nlayers_olm'   :   26,
        'ntypes'        :   2
    }
    default_config = SimpleNamespace(default_config_dict)
# Aliases (perhaps to decide on just a few):
# Naming Rationale: Embedding, Hypernetwork, LLM
# EHLGOLM <- EHLG + OLM
Gollem = EHLGOLM
GOHLLEM = EHLGOLM
Golem = GOHLLEM
GOLLEM = Gollem
EHL = EHLGOLM

def gollem_test():
    gollem_model = Gollem(Gollem.default_config)
    gollem = gollem_model
    #[dec, raw] = gollem("This is a test sentence for Gollem.")
    [dec, raw] = gollem("Make up a new word and explain what it means. The word and its meaning are: ")
    print(dec)
    return [dec, raw]

def main():
    gollem_test()

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

def test_gollem_train():
    # Gollem training pipeline
    pass


if __name__ == "__main__":
    main()