#import ehlg
import gollem
import torch
import torch.nn as nn
import torch.nn.functional as F

layer_indices_default_all = [n for n in range(26)]
weight_types_default_all = [[i for i in range(2)] for _ in range(len(layer_indices_default_all))]
lia = layer_indices_default_all
wta = weight_types_default_all
default_prompt = "This is a sample sentence."

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

    for i in range(nSteps):
        out = model(#input_ids=input_ids,
                    prompt=default_prompt,
                    layer_indices=layer_indices_default_all,
                    weight_types=weight_types_default_all)
        #target = torch.zeros(torch.Size([3200, 8]))
        #out_s = out['scores'][0]
        out_s = out['logits']
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

def check_gollem_training_mock(model, n=3):
    out = model(prompt=default_prompt, layer_indices=lia, weight_types=wta)
    #dec = model.olm.tokenizer.decode(out['sequences'][0])
    #dec = model.olm.tokenizer.decode(out['logits'][0])
    #print(dec)
    print(out)

def main():
    #model = ehlg.EHLG(ehlg.EHLG.default_config)
    model = gollem.Gollem(gollem.Gollem.default_config)

    print("checking model...")
    check_gollem_training_mock(model)

    print("mock training model (to output zeros)...")
    model = test_gollem_train_to_zero(nSteps=100, model=model, verbose=True)

    print("checking mock trained model...")
    check_gollem_training_mock(model)

if __name__ == "__main__":
    main()