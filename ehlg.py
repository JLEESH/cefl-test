import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace

class EMBGPT2LoRAGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # unique embedding vector for each layer and type combination;
        # could be changed to be more parameter-efficient
        self.nlayers = config.nlayers
        self.ntypes = config.ntypes
        self.n_emb_types = self.nlayers * self.ntypes
        self.emb = nn.Embedding(self.n_emb_types, config.emb_output_dim)
        
        # Hypernetwork and aliases
        self.gpt2 = config.gpt2 # centralised-fine-tuned or pretrained gpt2
        self.hypernetwork = self.gpt2 # alias
        self.inner_model = self.hypernetwork
        self.model = self.hypernetwork
        
        # Tokenizer and aliases
        self.gpt2_tokenizer = config.gpt2_tokenizer
        self.tokenizer = self.gpt2_tokenizer
        
        # TODO: Reduce size of EHLG.o2l?
        # Currently, the size of the matrix is 768-by-(3200*rank*2)
        # (in the case of using gpt2 as the HN and openllama as the LLM),
        # which may be less than ideal;
        # note that, however, (the gradient for) this matrix is not communicated
        # during the federated learning stage;
        # it may be fine-tuned during the centralised fine-tuning stage, however.
        # (Perhaps some sort of initialisation trick will do.)
        # We can also consider generating just one of the two LoRA matrices, in addition.
        # (Less gradients and slightly smaller model size)
        self.o2l = nn.Linear(config.lora_gen_in_dim, config.lora_gen_out_dim)
        
        # LoRA configurations
        self.lora_rank = config.lora_rank
        self.lora_in_dim = config.lora_in_dim # based on OLM
        self.lora_out_dim = config.lora_out_dim # likewise
    
    def forward(self, layer_indices, weight_types, data_emb=None):
        x = self.make_input_batch(layer_indices, weight_types, data_emb)
        x = self._inner_forward(x)
        ab_dict = self.reshape_output(x, layer_indices, weight_types)
        return ab_dict
    
    def _inner_forward(self, x):
        # pass embeddings to the hypernetwork and the resulting output to the lora generator
        for layer_index in range(len(self.gpt2.transformer.h)):
            x = self.gpt2.transformer.h[layer_index](x)[0]
        x = self.gpt2.transformer.ln_f(x)
        x = self.o2l(x)
        
        return x
    forward_old=_inner_forward
    
    def reshape_output(self, x, layer_indices, weight_types):
        # construct dictionary of A and B LoRA matrices by layer index and weight type
        ab_dict = {}
        counter = 0
        for i, layer_index in enumerate(layer_indices):
            ab_dict[layer_index] = {}
            for wt in weight_types[i]:
                # extract relevant row
                x_i_t = x[:,counter,:]
                
                # reshape to lora dimensions
                A_size = self.lora_rank * self.lora_in_dim
                # B_size = self.lora_rank * self.lora_out_dim
                
                A = x_i_t[:,:A_size].reshape(self.lora_in_dim, self.lora_rank)
                B = x_i_t[:,A_size:].reshape(self.lora_rank, self.lora_out_dim)
                
                # add A and B matrices to dict
                ab_dict[layer_index][wt] = {'A': A, 'B': B}
                counter += 1
        
        return ab_dict
    
    def make_input(self, layer_index, weight_type, data_emb=None):
        x = self.emb(layer_index * self.ntypes + weight_type)
        # TODO: test appending data_emb
        if data_emb is not None:
            x = torch.cat((x, data_emb))
        x = x.reshape((1, 1, x.shape[-1]))
        return x
    
    def make_input_batch(self, layer_indices, weight_types, data_emb=None):
        if len(layer_indices) != len(weight_types):
            raise ValueError(
                "``make_input_batch()``: length of ``layer_indices`` and ``weight_types`` do not match."
            )
        
        # extract relevant weights by index combination (after flattening the combinations)
        # cf. test input
        indices_list = [li * self.ntypes + wt for index, li in enumerate(layer_indices) for wt in weight_types[index]]
        x = self.emb(torch.tensor(indices_list))
        
        # append data_emb if any
        # TODO: test appending data_emb
        if data_emb is not None:
            x = torch.cat((x, data_emb))
        x = x.reshape((1, x.shape[-2], x.shape[-1]))
        return x
    
    def _freeze_hypernetwork(self, requires_grad):
        for param in self.model.model.parameters():
            param.requires_grad = requires_grad
    
    def freeze_hypernetwork(self):
        self._freeze_hypernetwork(requires_grad=False)
    
    def unfreeze_hypernetwork(self):
        self._freeze_hypernetwork(requires_grad=True)
    
    def _freeze_o2l(self, requires_grad):
        for param in self.o2l.parameters():
            param.requires_grad = requires_grad
    
    def freeze_o2l(self):
        self._freeze_o2l(requires_grad=False)
    
    def unfreeze_o2l(self):
        self._freeze_o2l(requires_grad=True)
    
    def freeze_for_fl(self):
        self.freeze_hypernetwork()
        self.freeze_o2l()
    
    def freeze_for_cft(self, freeze_hypernetwork=False):
        if freeze_hypernetwork:
            self.freeze_hypernetwork()
        else:
            self.unfreeze_hypernetwork()
        self.unfreeze_o2l()
    
    def count_params(self, trainable_only=False):
        # excludes gpt2 vocab embed
        n_p_emb = self.emb.parameters().numel()
        
        if trainable_only is True:
            n_p_o2l = sum(p.numel() for p in self.o2l.parameters() if p.requires_grad is True)
            n_p_model = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad is True)
        else:
            n_p_o2l = sum(p.numel() for p in self.o2l.parameters())
            n_p_model = sum(p.numel() for p in self.model.model.parameters())
            
        n_p_total = n_p_emb + n_p_o2l + n_p_model
        return n_p_total

    DEFAULT_EMB_DIM = 768 # same as the (wte)/embedding dimensions of the hypernetwork in EHLG
    DEFAULT_LORA_DIM = 3200 # should be the same as OLM weight dimensions (ehlg.o2l could be reduced in size)
    DEFAULT_LORA_RANK = 8
    default_config_dict = {
        
        'gpt2'              :   AutoModelForCausalLM.from_pretrained('gpt2'),
        'gpt2_tokenizer'    :   AutoTokenizer.from_pretrained('gpt2'),
        'nlayers'           :   26,#12,
        'ntypes'            :   2,
        'emb_output_dim'    :   DEFAULT_EMB_DIM,
        'lora_gen_in_dim'   :   DEFAULT_EMB_DIM, # (=gpt2 output dim.)
        'lora_gen_out_dim'  :   DEFAULT_LORA_DIM * DEFAULT_LORA_RANK * 2,
        'lora_rank'         :   DEFAULT_LORA_RANK,
        'lora_in_dim'       :   DEFAULT_LORA_DIM,
        'lora_out_dim'      :   DEFAULT_LORA_DIM
    }
    default_config = SimpleNamespace(default_config_dict)
# Aliases:
# EMB-Hypernetwork-LoRAGenerator <- EHLG
EHLG_GPT2 = EMBGPT2LoRAGen
EHLG = EHLG_GPT2

def ehlg_test_bare():
    ehlg_model = EHLG(EHLG.default_config)
    ehlg = ehlg_model
    
    layer_indices = [0, 1, 2, 4, 8]
    weight_types = [[0, 1], [0, 1], [0], [1], [0, 1]]
    output_ab_dict = ehlg(layer_indices, weight_types)
    
    return output_ab_dict

def ehlg_test():
    # test implementation
    print("Running some basic tests...")
    print("Creating model (``ehlg``)...")
    
    ehlg_model = EHLG(EHLG.default_config)
    ehlg = ehlg_model
    
    print("Model ``ehlg`` created.")
    
    layer_indices   =   [0, 1, 2, 4, 8]
    weight_types    =   [[0, 1], [0, 1], [0], [1], [0, 1]] # for each layer, decide which weight types to adapt
    
    print(f"Testing forward pass with ``li={layer_indices}`` and ``wt={weight_types}``...")
    
    output_ab_dict = ehlg(layer_indices, weight_types)
    
    print(f"Output from ``ehlg``:\n{output_ab_dict}")
    
    return output_ab_dict

def main():
    #print(ehlg_test_bare())
    ehlg_test()

if __name__ == '__main__':
    main()