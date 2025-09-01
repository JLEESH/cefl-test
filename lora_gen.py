import torch
import torch.nn as nn

class EMBGPT2LoRAGen(nn.Module):
    def __init__(self, config):
        """
            EMBGPT2LoRAGen(config)
            
            ```
            config={
                ###input_encoder, # maps layer_index, etc. to embeddings###
                ###emb_input_dim,###
                emb_types, # (no. of layer_index * weight_type combinations)
                emb_output_dim, # (=gpt2 input dim.)
                lora_gen_in_dim, # (=gpt2 output dim.)
                lora_gen_out_dim, # (=A_size + B_size)
                lora_rank
                lora_in_dim
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
        #self.input_encoder = config.input_encoder
        
        #self.emb = nn.Linear(config.emb_input_dim, config.emb_output_dim)
        self.emb = nn.Embedding(config.emb_types, config.emb_output_dim)
        self.gpt2 = config.gpt2
        self.o2l = nn.Linear(config.lora_gen_in_dim, config.lora_gen_out_dim)
        
        self.lora_rank = config.lora_rank
        self.lora_in_dim = config.lora_in_dim
        self.lora_out_dim = config.lora_out_dim
    
    def forward(self, x):
        # pass inputs thru model
        #x = self.emb(x)
        x = self.gpt2(x)
        x = self.o2l(x)
        
        # + reshape to lora dimensions
        # e.g. 
        A_size = self.lora_rank * self.lora_in_dim
        # B_size = self.lora_rank * self.lora_out_dim
        
        # reshape and return output
        A = x[:A_size].reshape(self.lora_in_dim, self.lora_rank)
        B = x[A_size:].reshape(self.lora_rank, self.lora_out_dim)
        
        # consider an alternative output format
        return [A, B]
    
    def make_input(layer_index, weight_type, data=None):
        #combined_input = ' <sep> '.join([layer_index, weight_type, data])
        #combined_input = ', '.join([layer_index, weight_type, data])
        #self.input_encoder(combined_input)
        x = self.emb(layer_index * weight_type)
        if data is not None:
            x.concatenate(data)
        
        return x
        
        

# EMB-Hypernetwork-LoRAGenerator <- EHLG
EHLG_GPT2 = EMBGPT2LoRAGen
EHLG = EHLG_GPT2