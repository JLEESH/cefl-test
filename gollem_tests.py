#import ehlg
import gollem
import torch
import torch.nn as nn
import torch.nn.functional as F

layer_indices_default_all = [n for n in range(26)]
weight_types_default_all = [[i for i in range(2)] for _ in range(len(layer_indices_default_all))]
lia = layer_indices_default_all
wta = weight_types_default_all

def test_gollem_train_to_zero(nSteps=1000, model=None, verbose=False):
    if model is None:
        #model = ehlg.EHLG(ehlg.EHLG.default_config)
        model = gollem.Gollem(gollem.Gollem.default_config)
    if type(model) is not gollem.Gollem:
        raise ValueError("``model`` passed to ``test_ehlg_train_to_zero`` is of the wrong type.")

    #model.freeze_for_pe_cft()
    model.freeze_for_fl()
    model.train()
    print(f"param count: {model.count_params()}")

    #optim = torch.optim.SGD(model.parameters(), lr=1e-2)#, momentum=0.9)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.1)
    #optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # accum_loss = 0.0

    for i in range(nSteps):
        out = model(#input_ids=input_ids,
                    prompt="This is a sample sentence.",
                    layer_indices=layer_indices_default_all,
                    weight_types=weight_types_default_all)
        #target = torch.zeros(torch.Size([3200, 8]))
        #target = torch.zeros(torch.Size([3200, 8]))
        target = torch.zeros(torch.Size(out.shape))

        #loss = (target - out[0][0]['A']).sum()
        #loss = (out[0][0]['A'] - target).sum() / (3200 * 8)
        #loss = (out[0][0]['A'] - target).sum()
        #loss = F.mse_loss(out[0][0]['A'], target=target)
        loss = F.mse_loss(out, target=target)

        # # uncomment as needed
        # if verbose:
        #     import torchviz
        #     dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        #     #dot = torchviz.make_dot(loss.mean(), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
        #     dot.format = 'png'
        #     dot.render('dot_gollem_1_ignore')
        #     return model

        loss.backward()

        #accum_loss += float(loss)

        optim.step() #gradient descent

        if verbose:
            if (i % (nSteps / 10)) == 0:
                print(f"Step {i+1:5d} of {nSteps:5d} complete...")
            #avg_loss = accum_loss / (i + 1)
            print(f"loss: {float(loss.detach())}")

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
    #model = ehlg.EHLG(ehlg.EHLG.default_config)
    model = gollem.Gollem(gollem.Gollem.default_config)

    print("checking model...")
    check_ehlg_training_mock(model)

    print("mock training model (to output zeros)...")
    model = test_gollem_train_to_zero(nSteps=100, model=model, verbose=True)

    print("checking mock trained model...")
    check_ehlg_training_mock(model)

if __name__ == "__main__":
    main()