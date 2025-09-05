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
        #self.olm.freeze_for_fl()
        self.ehlg = config.ehlg
        #self.ehlg.freeze_for_fl()
        # self.olm.tokenizer
        # self.ehlg.tokenizer
        self.scale_ehlg = config.scale_ehlg
        self.scale_olm = config.scale_olm
        self.nlayers_olm = config.nlayers_olm
        self.ntypes = config.ntypes

    def forward(self, prompt=None, input_ids=None, layer_indices=None, weight_types=None, **kwargs):
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
        if prompt is not None:
            enc_olm = self.olm.tokenizer(prompt, return_tensors="pt")
            #out_olm = self.olm.model.generate(input_ids=enc_olm.input_ids, max_new_tokens=128)
            out_olm = self.olm.model.generate(input_ids=enc_olm.input_ids, max_new_tokens=12)
            dec_olm = self.olm.tokenizer.decode(out_olm[0])
            return [dec_olm, out_olm]
        elif input_ids is not None:
            out_olm = self.olm.model.generate(input_ids=input_ids, max_new_tokens=12)
            dec_olm = self.olm.tokenizer.decode(out_olm[0])
            return out_olm
        else:
            raise ValueError("Provide ``prompt`` or ``input_ids`` for ``EHLGOLM.forward()``.")
        
        # return decoded text with the raw output
        return [dec_olm, out_olm]

    def adapt_olm(self, lora_dict):
        for li, wt in lora_dict.items():
            for wt_i in wt:
                if wt_i == 0:
                    #self.olm.model.model.layers[li].self_attn.v_proj.weight += lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B']
                    org = self.olm.model.model.layers[li].self_attn.k_proj
                    w = torch.nn.Parameter(lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B'], requires_grad=False)
                    self.olm.model.model.layers[li].self_attn.k_proj = LinearLoRA(org, w)
                    #self.olm.model.model.layers[li].self_attn.k_proj.weight = torch.nn.Parameter(torch.zeros((3200, 3200)))
                elif wt_i == 1:
                    #self.olm.model.model.layers[li].self_attn.v_proj.weight += lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B']
                    org = self.olm.model.model.layers[li].self_attn.v_proj
                    w = torch.nn.Parameter(lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B'], requires_grad=False)
                    self.olm.model.model.layers[li].self_attn.v_proj = LinearLoRA(org, w)
                    #self.olm.model.model.layers[li].self_attn.v_proj.weight = torch.nn.Parameter(torch.zeros((3200, 3200)))
                    # other adjustments tried: amplifying w (x1000); setting ``torch.manual_seed(42)``; removing the adaptation process
                else:
                    raise NotImplementedError
        return self.olm

    def _freeze_all(self, requires_grad):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def freeze_all(self):
        self.ehlg.freeze_all()
        self.olm.freeze_all()

    def unfreeze_all(self):
        self.ehlg.unfreeze_all()
        self.olm.unfreeze_all()

    def freeze_for_fl(self):
        self.ehlg.freeze_for_fl()
        self.olm.freeze_for_fl()

    def freeze_for_cft(self):
        self.ehlg.freeze_for_cft()
        self.olm.freeze_for_cft()
        
    def freeze_for_pe_cft(self):
        self.ehlg.freeze_for_pe_cft()
        self.olm.freeze_for_pe_cft()
    
    def count_params(self, trainable_only=True):
        n_p_dict = {}
        n_p_ehlg_dict = self.ehlg.count_params(trainable_only)
        n_p_olm = self.olm.count_params(trainable_only)
        n_p_total = n_p_ehlg_dict[0] + n_p_olm
        
        n_p_dict['ehlg'] = n_p_ehlg_dict
        n_p_dict['olm'] = n_p_olm
        n_p_dict['total'] = n_p_total
        return n_p_total, n_p_dict
    
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
#GOHLLEM = EHLGOLM
#Golem = GOHLLEM
#GOLLEM = Gollem
#EHL = EHLGOLM

def gollem_test():
    gollem_model = Gollem(Gollem.default_config)
    gollem = gollem_model
    #[dec, raw] = gollem("This is a test sentence for Gollem.")
    [dec, raw] = gollem("Make up a new word and explain what it means. The word and its meaning are: ")
    print(dec)
    return [dec, raw]

def gollem_freezing_test(verbose=True):
    # test various freezing configurations
    gollem = Gollem(Gollem.default_config)
    
    gollem_params_count = {}
    gpc = gollem_params_count
    gpc['init'] = gollem.count_params()
    
    gollem.freeze_all()
    gpc['f_a'] = gollem.count_params()
    
    gollem.unfreeze_all()
    gpc['uf_a'] = gollem.count_params()
    
    gollem.freeze_for_fl()
    gpc['f_fl'] = gollem.count_params()
    
    gollem.freeze_for_cft()
    gpc['f_cft'] = gollem.count_params()
    
    gollem.freeze_for_pe_cft()
    gpc['f_pe_cft'] = gollem.count_params()
    
    if verbose:
        print(gpc)
    return gpc

def gollem_training_test():
    print("Warning: ``gollem_training_test()`` is not yet implemented.")
    return
    raise NotImplementedError
    import tqdm
    
    nEpochs = 3
    nIterPerEpoch = 5
    evalIter = 100
    
    # instantiate model
    gollem = Gollem(Gollem.default_config)
    
    # generate a synthetic dataset
    import random
    
    def generate_synth_data(causal=True):
        nQuestions = 1000
        qn_template = "What is the answer to the following mathematics equation?\n{}\nThe answer is: "
        synth_data_mq = {}
        for i in range(nQuestions):
            a1 = random.randrange(100)
            a2 = random.randrange(100)
            op = '*'
            r1 = a1 * a2
            qn_str = "{} {} {} = " .format(a1, op, a2)
            synth_data_mq[i] = {
                'Q': qn_template.format(qn_str),
                'A': str(r1)
            }
        if causal is True:
            synth_data_mq_causal = {k : ''.join([v['Q'], v['A']]) for k, v in synth_data_mq.items()}
            return synth_data_mq_causal
        return synth_data_mq
    synth_data_mq_causal = generate_synth_data(causal=True)
    
    tbar = tqdm(nEpochs * nIterPerEpoch)
    for epoch in range(nEpochs):
        for iter in range(nIterPerEpoch):
            
            gollem(synth_data_mq_causal[iter])
            
            if iter % evalIter == 0:
                res_eval = gollem.evaluate_perf()
                
            tbar.update(nEpochs * nIterPerEpoch + iter)
    pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-basic", action="store_true")
    parser.add_argument("--test-freezing", action="store_true")
    parser.add_argument("--test-training", action="store_true")
    parser.add_argument("--test", "--test-all", action="store_true")
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    torch_seed = args.seed
    #test_basic = args.test
    if args.test:
        TEST_BASE = True
        TEST_FREEZE = True
        TEST_TRAIN = True
    else:
        TEST_BASE = args.test_basic
        TEST_FREEZE = args.test_freezing
        TEST_TRAIN = args.test_training
    
    torch.manual_seed(torch_seed)
    
    if TEST_BASE:
        gollem_test()
    if TEST_FREEZE:
        gollem_freezing_test()
    if TEST_TRAIN:
        gollem_training_test()

if __name__ == "__main__":
    main()