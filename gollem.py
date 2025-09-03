import torch
import torch.nn as nn
from ehlg import EHLG
from olm import OLM
from lora import LinearLoRA
from types import SimpleNamespace

# EHL-FT-FL : Fine-tuning Models through Federated Learning of Hypernetwork Embeddings
# CEFL: Communication-Efficient Federated learning

class EHLGOLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.olm = config.olm
        self.olm.freeze()
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
        for li, wt in lora_dict.items():
            for wt_i in wt:
                if wt_i == 0:
                    #self.olm.model.model.layers[li].self_attn.v_proj.weight += lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B']
                    org = self.olm.model.model.layers[li].self_attn.k_proj
                    w = torch.nn.Parameter(lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B'])
                    self.olm.model.model.layers[li].self_attn.k_proj = LinearLoRA(org, w)
                    #self.olm.model.model.layers[li].self_attn.k_proj.weight = torch.nn.Parameter(torch.zeros((3200, 3200)))
                elif wt_i == 1:
                    #self.olm.model.model.layers[li].self_attn.v_proj.weight += lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B']
                    org = self.olm.model.model.layers[li].self_attn.v_proj
                    w = torch.nn.Parameter(lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B'])
                    self.olm.model.model.layers[li].self_attn.v_proj = LinearLoRA(org, w)
                    #self.olm.model.model.layers[li].self_attn.v_proj.weight = torch.nn.Parameter(torch.zeros((3200, 3200)))
                    # other adjustments tried: amplifying w (x1000); setting ``torch.manual_seed(42)``; removing the adaptation process
                else:
                    raise NotImplementedError
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
    torch.manual_seed(42)
    gollem_test()

def gollem_training_test():
    # Gollem training pipeline
    pass

if __name__ == "__main__":
    main()