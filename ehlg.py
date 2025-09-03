import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
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
        
        # # TODO: consider doing away with making the input and output separately
        # # pass embeddings thru model
        # for layer_index in range(len(self.gpt2.transformer.h)):
        #     x = self.gpt2.transformer.h[layer_index](x)[0]
        # x = self.gpt2.transformer.ln_f(x)
        # x = self.o2l(x)
        
        # return x
    
    def _inner_forward(self, x):
        # pass embeddings thru model
        for layer_index in range(len(self.gpt2.transformer.h)):
            x = self.gpt2.transformer.h[layer_index](x)[0]
        x = self.gpt2.transformer.ln_f(x)
        x = self.o2l(x)
        
        return x
    forward_old=_inner_forward
    
    def reshape_output(self, x, layer_indices, weight_types):
        ab_dict = {}
        counter = 0
        for i, layer_index in enumerate(layer_indices):
            ab_dict[layer_index] = {}
            for wt in weight_types[i]:
                x_i_t = x[:,counter,:]
                        
                # reshape to lora dimensions
                A_size = self.lora_rank * self.lora_in_dim
                # B_size = self.lora_rank * self.lora_out_dim
                A = x_i_t[:,:A_size].reshape(self.lora_in_dim, self.lora_rank)
                B = x_i_t[:,A_size:].reshape(self.lora_rank, self.lora_out_dim)
                
                ab_dict[layer_index][wt] = {'A': A, 'B': B}
                counter += 1
        
        return ab_dict
    
    def make_input(self, layer_index, weight_type, data_emb=None):
        x = self.emb(layer_index * self.ntypes + weight_type)
        if data_emb is not None:
            x = torch.cat((x, data_emb))
        x = x.reshape((1, 1, x.shape[-1]))
        return x
    
    def make_input_batch(self, layer_indices, weight_types, data_emb=None):
        if len(layer_indices) != len(weight_types):
            raise ValueError(
                "``make_input_batch()``: length of ``layer_indices`` and ``weight_types`` do not match."
            )
        
        # extract relevant weights by index combination
        indices_list = [li * self.ntypes + wt for i, li in enumerate(layer_indices) for wt in weight_types[i]]
        x = self.emb(torch.tensor(indices_list))
        
        # append data_emb if any
        if data_emb is not None:
            x = torch.cat((x, data_emb))
        x = x.reshape((1, x.shape[-2], x.shape[-1]))
        return x

    DEFAULT_EMB_DIM = 768 # same as the (wte)/embedding dimensions of the hypernetwork in EHLG
    DEFAULT_LORA_DIM = 3200 # should be the same as OLM weight dimensions (ehlg.o2l could be reduced in size)
    DEFAULT_LORA_RANK = 8
    default_config_dict = {
        
        'gpt2'              :   AutoModelForCausalLM.from_pretrained('gpt2'),
        'gpt2_tokenizer'    :   AutoTokenizer.from_pretrained('gpt2'),
        'nlayers'           :   12,
        'ntypes'            :   2,
        'emb_output_dim'    :   DEFAULT_EMB_DIM,
        'lora_gen_in_dim'   :   DEFAULT_EMB_DIM, # (=gpt2 output dim.)
        'lora_gen_out_dim'  :   DEFAULT_LORA_DIM * DEFAULT_LORA_RANK * 2,
        'lora_rank'         :   DEFAULT_LORA_RANK,
        'lora_in_dim'       :   DEFAULT_LORA_DIM,
        'lora_out_dim'      :   DEFAULT_LORA_DIM
    }
    default_config = SimpleNamespace(default_config_dict)
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
    
    layer_indices = [0, 1, 2, 4, 8]
    weight_types = [[0, 1], [0, 1], [0], [1], [0, 1]]
    
    print(f"Testing forward pass with ``li={layer_indices}`` and ``wt={weight_types}``...")
    
    output_ab_dict = ehlg(layer_indices, weight_types)
    
    print(f"Output from ``ehlg``:\n{output_ab_dict}")
    
    return output_ab_dict

def main():
    #print(ehlg_test_bare())
    ehlg_test()

def lora_gen_test():
    # TODO: to turn the following into a class function?
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from types import SimpleNamespace
    
    nLayers = 12
    nWeightTypes = 2
    
    config = SimpleNamespace()
    config.gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
    config.gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config.nlayers = nLayers
    config.ntypes = nWeightTypes
    
    EMB_DIM = 768 # same as the (wte)/embedding dimensions of the hypernetwork in EHLG
    LORA_DIM = 3200 # should be the same as OLM weight dimensions (ehlg.o2l could be reduced in size)
    LORA_RANK = 8
    config.emb_output_dim = EMB_DIM
    
    config.lora_gen_in_dim = EMB_DIM # (=gpt2 output dim.)
    #config.lora_gen_out_dim = 6144 * 2 # (=A_size + B_size) # TODO: fix A_size calculation redundancies
    config.lora_gen_out_dim = LORA_DIM * LORA_RANK * 2 # (no. of params grows by almost x20... to fix)
    config.lora_rank = LORA_RANK
    config.lora_in_dim = LORA_DIM
    config.lora_out_dim = config.lora_in_dim
    
    #eg_model = EHLG(config)
    #return eg_model
    
    ehlg_model = EHLG(config)
    return ehlg_model

def lora_gen_test_p2(ehlg_model, layer_indices, weight_types):
    # TODO: likewise, perhaps turn it into a class function
    
    #eg_model = ehlg_model
    #layer_index = 1
    #weight_type = 1
    #x = eg_model.make_input(torch.tensor(layer_index), torch.tensor(weight_type))
    #output = eg_model(x)
    
    ehlg = ehlg_model
    x = ehlg.make_input_batch(layer_indices, weight_types)
    #output = ehlg(x)
    output = ehlg.forward_old(x)
    output_ab_dict = ehlg.reshape_output(output, layer_indices, weight_types)
    return output_ab_dict

def old_main():
    # TODO: Instantiation of EHLG using default settings as described in tests
    
    # test implementation
    print("running some basic tests...")
    print("creating model (ehlg)...")
    eg_model = lora_gen_test() # should be a class function (constructor? configuration method?)
    print("ehlg(=eg_model) created.")
    
    # print("testing forward pass with li=1 and wt=1...")
    # output = lora_gen_test_p2(eg_model)
    # print("output from eg_model...")
    # #print(f"output: {output}")
    # print("A:%s\n B:%s" % (output[0].shape, output[1].shape))
    
    layer_indices = [0, 1, 2, 4, 8]
    weight_types = [[0, 1], [0, 1], [0], [1], [0, 1]]
    print(f"testing forward pass with li={layer_indices} and wt={weight_types}...")
    
    # likewise, perhaps the testing function below should be made part of the model
    output_ab_dict = lora_gen_test_p2(eg_model, layer_indices, weight_types)
    
    print("output from eg_model...")
    print(f"output: {output_ab_dict}")


if __name__ == '__main__':
    main()
    #old_main()