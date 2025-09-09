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
        self.adapt_olm()

    def forward(self, prompt=None, input_ids=None, layer_indices=None, weight_types=None, **kwargs):
        # by default, the input to Gollem should be:
        # 1. list of indices and types for HN (can have default values)
        # 2. input for llm (data)

        if prompt is not None and input_ids is not None:
            raise ValueError("``EHLGOLM.forward()``: Please only provide one of ``prompt`` or ``input_ids``, but not both")
        elif prompt is None and input_ids is None:
            raise ValueError("``EHLGOLM.forward()``: Provide ``prompt`` or ``input_ids``.")

        # train all indices by default
        if layer_indices is None:
            layer_indices = [n for n in range(self.nlayers_olm)]
        if weight_types is None:
            weight_types = [[i for i in range(self.ntypes)] for _ in range(len(layer_indices))]
        if len(layer_indices) != len(weight_types):
            raise ValueError(
                "``EHLGOLM.forward()``: the lengths of ``layer_indices`` and ``weight_types`` do not match."
            )

        # obtain LoRA matrix weights and reshape
        ehlg_output = self.ehlg(layer_indices, weight_types, data_emb=None)
        ehlg_output_ab_dict = self.ehlg.reshape_output(ehlg_output, layer_indices, weight_types)

        # add LoRA matrix outputs to OLM
        self.olm_add_lora(lora_dict=ehlg_output_ab_dict)

        # # uncomment as needed
        # if verbose:
        #     import torchviz
        #     dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        #     #dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
        #     dot.format = 'png'
        #     dot.render('dot_gollem_1_ignore')
        #     return model

        # pass prompt to OLM
        if prompt is not None:
            enc_olm = self.olm.tokenizer(prompt, return_tensors="pt")
            input_ids = enc_olm.input_ids
        out_olm = self.olm.model(input_ids) # get CausalLMOutputWithPast
        return out_olm

    def generate(self,
        prompt=None,
        input_ids=None,
        layer_indices=None,
        weight_types=None,
        adapted=True,
        max_new_tokens=12,
        output_scores=False,
        return_dict_in_generate=False,
        do_sample=False,
        **kwargs):

        # by default, the input to Gollem should be:
        # 1. list of indices and types for HN (can have default values)
        # 2. input for llm (data)

        if prompt is not None and input_ids is not None:
            raise ValueError("``EHLGOLM.generate()``: Please only provide one of ``prompt`` or ``input_ids``, but not both")
        elif prompt is None and input_ids is None:
            raise ValueError("``EHLGOLM.generate()``: Provide ``prompt`` or ``input_ids``.")

        # train all indices by default
        if layer_indices is None:
            layer_indices = [n for n in range(self.nlayers_olm)]
        if weight_types is None:
            weight_types = [[i for i in range(self.ntypes)] for _ in range(len(layer_indices))]
        if len(layer_indices) != len(weight_types):
            raise ValueError(
                "``EHLGOLM.generate()``: the lengths of ``layer_indices`` and ``weight_types`` do not match."
            )

        # obtain LoRA matrix weights and reshape
        ehlg_output = self.ehlg(layer_indices, weight_types, data_emb=None)
        ehlg_output_ab_dict = self.ehlg.reshape_output(ehlg_output, layer_indices, weight_types)

        # add LoRA matrix outputs to OLM
        if adapted is True:
            self.olm_add_lora(lora_dict=ehlg_output_ab_dict)
        else:
            self.olm_remove_lora()

        # pass prompt to OLM
        if prompt is not None:
            enc_olm = self.olm.tokenizer(prompt, return_tensors="pt")
            input_ids = enc_olm.input_ids
        out_olm = self.olm.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=do_sample)
        return out_olm

    def adapt_olm(self):
        for li in range(self.ehlg.nlayers):
            for wt_i in range(self.ehlg.ntypes):
                if wt_i == 0:
                    org = self.olm.model.model.layers[li].self_attn.k_proj
                    self.olm.model.model.layers[li].self_attn.k_proj = LinearLoRA(org)
                elif wt_i == 1:
                    org = self.olm.model.model.layers[li].self_attn.v_proj
                    self.olm.model.model.layers[li].self_attn.v_proj = LinearLoRA(org)
                else:
                    raise NotImplementedError
        return self.olm

    def olm_add_lora(self, lora_dict):
        for li, wt in lora_dict.items():
            for wt_i in wt:
                if wt_i == 0:
                    self.olm.model.model.layers[li].self_attn.k_proj.attach_w(lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B'])
                elif wt_i == 1:
                    self.olm.model.model.layers[li].self_attn.v_proj.attach_w(lora_dict[li][wt_i]['A'] @ lora_dict[li][wt_i]['B'])
                else:
                    raise NotImplementedError
        return self.olm

    def olm_remove_lora(self):
        for li in range(self.ehlg.nlayers):
            for wt_i in range(self.ehlg.ntypes):
                if wt_i == 0:
                    self.olm.model.model.layers[li].self_attn.k_proj.remove_w()
                elif wt_i == 1:
                    self.olm.model.model.layers[li].self_attn.v_proj.remove_w()
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
# Naming Rationale: Embedding, Hypernetwork, LoRA Generator, LLM
# EHLGOLM <- EHLG + OLM
Gollem = EHLGOLM

def gollem_test():
    gollem_model = Gollem(Gollem.default_config)
    gollem = gollem_model
    out = gollem("Make up a new word and explain what it means. The word and its meaning are: ")
    dec = gollem.olm.tokenizer.decode(out[0])
    print(dec)
    return [dec, out]

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