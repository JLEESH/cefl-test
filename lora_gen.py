import torch
import torch.nn as nn

class EMBGPT2LoRAGen(nn.Module):
    def __init__(self, config):
        """
        EMBGPT2LoRAGen(config)
        
        ```
        config={
            emb_types, # (no. of layer_index * weight_type combinations)
            emb_output_dim, # (=gpt2.transformer.h[0] input dim.)
            lora_gen_in_dim, # (=gpt2.transformer.h[11] output dim.)
            lora_gen_out_dim, # (=A_size + B_size)
            lora_rank,
            lora_in_dim,
            lora_out_dim
        }
        ```
        
        A_size = lora_rank * lora_in_dim
        B_size = lora_rank * lora_out_dim
        
        ``forward(x)``:
            usage:
            ```
            # set LLM.weights to be the parameters to be fine-tuned
            ...
            for layer_index in range(nLayers):
                for weight_type in range(nWeightTypes):
                    x = make_input(layer_index, weight_type, data=None)
                    A, B = EHLG_GPT2.forward(x)
                    LLM.weights['{layer_index}']['{weight_type}'] += A @ B
            ```
        
        ``make_input(layer_index, weight_type, data)``:
            ``layer_index``
            ``weight_type`` (e.g. k or v)
            ``data`` (defaults to ``None``)
        
        The output is then added to the appropriate LLM matrix before the forward pass with the data.
        """
        super().__init__()
        self.emb = nn.Embedding(config.emb_types, config.emb_output_dim) # unique embed for each layer and type combin.
        self.gpt2 = config.gpt2 # fine-tuned or pretrained gpt2
        self.gpt2_tokenizer = config.gpt2_tokenizer
        self.tokenizer = self.gpt2_tokenizer
        self.o2l = nn.Linear(config.lora_gen_in_dim, config.lora_gen_out_dim)
        
        self.lora_rank = config.lora_rank
        self.lora_in_dim = config.lora_in_dim
        self.lora_out_dim = config.lora_out_dim
    
    def forward(self, x):
        # pass embeddings thru model
        #x = self.gpt2(x)
        for layer_index in range(len(self.gpt2.transformer.h)):
            x = self.gpt2.transformer.h[layer_index](x)[0] # TODO: fix leaving out (ln_f)
        x = self.gpt2.transformer.ln_f(x)
        # vs self.gpt2.lm_head(x)
        x = self.o2l(x)
        
        # reshape to lora dimensions
        A_size = self.lora_rank * self.lora_in_dim # B_size = self.lora_rank * self.lora_out_dim
        
        # reshape and return output
        x = x.reshape(-1)
        A = x[:A_size].reshape(self.lora_in_dim, self.lora_rank)
        B = x[A_size:].reshape(self.lora_rank, self.lora_out_dim)
        
        # consider an alternative output format
        return [A, B]
    
    def make_input(self, layer_index, weight_type, data_emb=None):
        x = self.emb(layer_index * weight_type)
        if data_emb is not None:
            x = torch.cat((x, data_emb))
        x = x.reshape((1, 1, 768))
        return x
        
        

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
    DIM = 768
    config.emb_output_dim = 768
    
    config.lora_gen_in_dim = 768 # (=gpt2 output dim.)
    config.lora_gen_out_dim = 6144 * 2 # (=A_size + B_size)
    config.lora_rank = 8
    config.lora_in_dim = 768
    config.lora_out_dim = config.lora_in_dim
    
    eg_model = EHLG(config)
    return eg_model

def lora_gen_test_p2(eg_model):
    x = eg_model.make_input(torch.tensor(1), torch.tensor(1))
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