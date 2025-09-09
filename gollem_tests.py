import gollem
import torch
import torch.nn as nn
import torch.nn.functional as F

layer_indices_default_all = [n for n in range(26)]
weight_types_default_all = [[i for i in range(2)] for _ in range(len(layer_indices_default_all))]
lia = layer_indices_default_all
wta = weight_types_default_all
default_prompt = "This is a sample sentence."


def test_gollem_train_to_dummy_sequence(nSteps=1000, model=None, verbose=False, freeze_func=None):
    if model is None:
        model = gollem.Gollem(gollem.Gollem.default_config)
    if type(model) is not gollem.Gollem:
        raise ValueError("``model`` passed to ``test_ehlg_train_to_dummy_sequence`` is of the wrong type.")

    if freeze_func is None:
        model.freeze_for_pe_cft()
    else:
        freeze_func()

    model.train()
    print(f"param count: {model.count_params()}")
    print(f"freezing config: {freeze_func.__func__}")
    print()

    #optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for i in range(nSteps):
        out = model(#input_ids=input_ids,
                    prompt=default_prompt,
                    layer_indices=layer_indices_default_all,
                    weight_types=weight_types_default_all)
        out_s = out
        inds_r = F.softmax(out_s, dim=2)
        #inds = torch.argmax(inds_r, dim=2)
        target = torch.zeros(torch.Size(inds_r.shape))
        for ti in range(len(target[0])):
            #target[0][ti][42] = 1 # model.olm.tokenizer.decode(torch.argmax(target)) : "'"
            target[0][ti][36] = 1 # model.olm.tokenizer.decode(torch.argmax(target)) : "!"
        loss = F.cross_entropy(inds_r[0], target[0])
        loss.backward()
        optim.step()

        if verbose:
            if (i % (nSteps / 10)) == 0:
                print(f"Step {i+1:5d} of {nSteps:5d} complete...")
            print(f"loss: {float(loss.detach())}")

    if verbose:
            print(f"Step {i+1:5d} of {nSteps:5d} complete...")
    return model


def test_gollem_train_to_zero(nSteps=1000, model=None, verbose=False, freeze_func=None):
    if model is None:
        model = gollem.Gollem(gollem.Gollem.default_config)
    if type(model) is not gollem.Gollem:
        raise ValueError("``model`` passed to ``test_ehlg_train_to_zero`` is of the wrong type.")

    if freeze_func is None:
        model.freeze_for_pe_cft()
    else:
        freeze_func()

    model.train()
    print(f"param count: {model.count_params()}")
    print(f"freezing config: {freeze_func.__func__}")
    print()

    #optim = torch.optim.SGD(model.parameters(), lr=1e-2)#, momentum=0.9)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.1)
    #optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for i in range(nSteps):
        out = model(#input_ids=input_ids,
                    prompt=default_prompt,
                    layer_indices=layer_indices_default_all,
                    weight_types=weight_types_default_all)
        #target = torch.zeros(torch.Size([3200, 8]))
        #out_s = out['scores'][0]
        #out_s = out['logits']
        out_s = out
        target = torch.zeros(torch.Size(out_s.shape))
        loss = F.mse_loss(out_s, target=target)
        loss.backward()
        optim.step()

        # # uncomment as needed
        # if verbose:
        #     import torchviz
        #     dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        #     #dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
        #     dot.format = 'png'
        #     dot.render('dot_gollem_1_ignore')
        #     return model

        if verbose:
            if (i % (nSteps / 10)) == 0:
                print(f"Step {i+1:5d} of {nSteps:5d} complete...")
            print(f"loss: {float(loss.detach())}")

    if verbose:
            print(f"Step {i+1:5d} of {nSteps:5d} complete...")
    return model

def get_gollem_output(model, prompt=default_prompt, adapted=True, do_sample=False, verbose=True):
    #out = model(prompt=default_prompt, layer_indices=lia, weight_types=wta)
    #dec = model.olm.tokenizer.decode(out['sequences'][0])
    #dec = model.olm.tokenizer.decode(out['logits'][0])
    #print(dec)
    out = model.generate(
        prompt=prompt,
        layer_indices=lia,
        weight_types=wta,
        adapted=adapted,
        do_sample=do_sample)
    out = model.olm.tokenizer.decode(out[0])

    if verbose:
        print(out)
    return out

def gollem_train_test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default=default_prompt)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--mode", "-m", type=str, default="train_to_dummy_sequence")
    parser.add_argument("--freezing-mode", "-f", type=str, default="freeze_for_pe_cft")
    args = parser.parse_args()
    prompt = args.prompt
    do_sample = args.do_sample
    mode = args.mode
    freezing_mode = args.freezing_mode

    modes = ["train_to_zero", "train_to_dummy_sequence"]

    if mode not in modes:
        raise ValueError("Invalid ``mode`` provided.")

    model = gollem.Gollem(gollem.Gollem.default_config)

    if freezing_mode == "freeze_for_fl":
        freeze_func = model.freeze_for_fl
    elif freezing_mode == "freeze_for_pe_cft":
        freeze_func = model.freeze_for_pe_cft # NOTE: training o2l as well here; training just the embeddings seems less effective
    elif freezing_mode == "freeze_for_cft":
        freeze_func = model.freeze_for_cft # crashes a computer with 32gb ram
    elif freezing_mode == "freeze_for_ehlg_tuning":
        freeze_func = model.freeze_for_ehlg_tuning
    else:
        raise ValueError("Please provide the name of a valid freezing function.")

    print("\nchecking model...")
    get_gollem_output(model, prompt=prompt, do_sample=do_sample)

    print("\nchecking model (unadapted output)...")
    get_gollem_output(model, prompt=prompt, adapted=False, do_sample=do_sample)

    print(f"\nmock training model (mode: {mode})...")
    #model = test_gollem_train_to_zero(nSteps=100, model=model, verbose=True)
    if mode == "train_to_zero":
        model = test_gollem_train_to_zero(nSteps=10, model=model, verbose=True, freeze_func=freeze_func)
    elif mode == "train_to_dummy_sequence":
        model = test_gollem_train_to_dummy_sequence(nSteps=100, model=model, verbose=True, freeze_func=freeze_func)
    else:
        raise ValueError("Invalid ``mode`` provided. Please check the code.")

    print("\nchecking mock trained model...")
    get_gollem_output(model, prompt=prompt, do_sample=do_sample)

    print("\nchecking mock trained model (unadapted output)...")
    get_gollem_output(model, prompt=prompt, adapted=False, do_sample=do_sample)

if __name__ == "__main__":
    gollem_train_test()