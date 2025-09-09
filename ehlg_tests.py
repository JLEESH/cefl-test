import ehlg
import torch
import torch.nn as nn
import torch.nn.functional as F

layer_indices_default_all = [n for n in range(26)] # NOTE: for use in this file only; no. of layers must match model!
weight_types_default_all = [[i for i in range(2)] for _ in range(len(layer_indices_default_all))] # likewise, but with weight types
lia = layer_indices_default_all
wta = weight_types_default_all

def test_ehlg_train_to_zero(nSteps=1000, model=None, verbose=False):
    if model is None:
        model = ehlg.EHLG(ehlg.EHLG.default_config)
    if type(model) is not ehlg.EHLG:
        raise ValueError("``model`` passed to ``test_ehlg_train_to_zero`` is of the wrong type.")

    model.freeze_for_pe_cft()
    #model.freeze_for_fl()
    model.train()
    print(f"param count: {model.count_params()}")

    #optim = torch.optim.SGD(model.parameters(), lr=1e-2)#, momentum=0.9)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.1)
    #optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for i in range(nSteps):
        out = model(layer_indices_default_all, weight_types_default_all)
        #target = torch.zeros(torch.Size([3200, 8]))
        target = torch.zeros(torch.Size(out.shape))
        loss = F.mse_loss(out, target=target)
        loss.backward()
        optim.step()

        # # uncomment as needed
        # if verbose:
        #     import torchviz
        #     dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        #     #dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
        #     dot.format = 'png'
        #     dot.render('dot_9_ignore')
        #     return model

        if verbose:
            if (i % (nSteps / 10)) == 0:
                print(f"Step {i+1:5d} of {nSteps:5d} complete...")
            print(f"loss: {float(loss.detach())}") # tab as needed

    if verbose:
            print(f"Step {i+1:5d} of {nSteps:5d} complete...")
    return model

def check_ehlg_training_mock(model, n=3):
    out = model(lia, wta)
    out = model.reshape_output(out, lia, wta)
    for i in range(n):
        print(f"i={i}")
        print(f"``out[{i}][0]['A'][0][0]``: {out[i][0]['A'][0][0]}")
        print(f"``out[{i}][0]['B'][0][0]``: {out[i][0]['B'][0][0]}")
        print(f"``out[{i}][1]['A'][0][0]``: {out[i][1]['A'][0][0]}")
        print(f"``out[{i}][0]['B'][0][0]``: {out[i][1]['B'][0][0]}")
        print()

def main():
    model = ehlg.EHLG(ehlg.EHLG.default_config)

    print("checking model...")
    check_ehlg_training_mock(model)

    print("mock training model (to output zeros)...")
    model = test_ehlg_train_to_zero(nSteps=100, model=model, verbose=True)

    print("checking mock trained model...")
    check_ehlg_training_mock(model)

if __name__ == "__main__":
    main()