import torch
import torch.nn as nn

class EMBGPT2LoRAGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.emb_types, config.emb_output_dim)
        # unique embed for each layer and type combination
        self.gpt2 = config.gpt2 # fine-tuned or pretrained gpt2
        self.gpt2_tokenizer = config.gpt2_tokenizer
        self.tokenizer = self.gpt2_tokenizer
        self.o2l = nn.Linear(config.lora_gen_in_dim, config.lora_gen_out_dim)
        
        self.lora_rank = config.lora_rank
        self.lora_in_dim = config.lora_in_dim
        self.lora_out_dim = config.lora_out_dim
    
    def forward(self, x):
    #def forward(self, layer_index, weight_type, data_emb=None):
        #x = self._make_input(layer_index, weight_type, data_emb=data_emb)
        
        # pass embeddings thru model
        for layer_index in range(len(self.gpt2.transformer.h)):
            x = self.gpt2.transformer.h[layer_index](x)[0]
        x = self.gpt2.transformer.ln_f(x)
        x = self.o2l(x)
        
        # reshape to lora dimensions
        A_size = self.lora_rank * self.lora_in_dim
        # B_size = self.lora_rank * self.lora_out_dim
        
        # reshape and return output
        x = x.reshape(-1)
        A = x[:A_size].reshape(self.lora_in_dim, self.lora_rank)
        B = x[A_size:].reshape(self.lora_rank, self.lora_out_dim)
        
        # consider an alternative output format
        return [A, B]
    
    def make_input(self, layer_index, weight_type, data_emb=None):
        # TODO: move to forward()?
        x = self.emb(layer_index * weight_type)
        if data_emb is not None:
            x = torch.cat((x, data_emb))
        x = x.reshape((1, 1, x.shape[-1]))
        return x
    #_make_input = make_input
    #make_input = _make_input

# EMB-Hypernetwork-LoRAGenerator <- EHLG
EHLG_GPT2 = EMBGPT2LoRAGen
EHLG = EHLG_GPT2

def lora_gen_test():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from types import SimpleNamespace
    
    nLayers = 12
    nWeightTypes = 2
    
    config = SimpleNamespace()
    config.gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
    config.gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config.emb_types = nLayers * nWeightTypes # i.e. EHLG does not currently know how the embeddings came about
    
    EMB_DIM = 768
    LORA_DIM = 768
    config.emb_output_dim = EMB_DIM
    
    config.lora_gen_in_dim = EMB_DIM # (=gpt2 output dim.)
    config.lora_gen_out_dim = 6144 * 2 # (=A_size + B_size) # TODO: fix A_size calculation redundancies
    config.lora_rank = 8
    config.lora_in_dim = LORA_DIM
    config.lora_out_dim = config.lora_in_dim
    
    eg_model = EHLG(config)
    return eg_model

def lora_gen_test_p2(eg_model):
    layer_index = 1
    weight_type = 1
    x = eg_model.make_input(torch.tensor(layer_index), torch.tensor(weight_type)) # TODO: maybe move everything to forward()?
    output = eg_model(x)
    return output

if __name__ == '__main__':
    
    # debug
    print("running some basic tests...")
    print("creating model (ehlg)...")
    eg_model = lora_gen_test()
    print("ehlg(=eg_model) created.")
    
    print("testing forward pass with li=1 and wt=1...")
    output = lora_gen_test_p2(eg_model)
    print("output from eg_model...")
    #print(f"output: {output}")
    print("A:%s\n B:%s" % (output[0].shape, output[1].shape))
    
    pass
