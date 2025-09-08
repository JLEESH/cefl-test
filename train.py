import gollem
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
#from transformers import get_scheduler, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import tqdm
#import wandb

#wandb.init(mode="disabled")

DEFAULT_DATASET_PATH = "../datasets/" + "openai/gsm8k"
DEFAULT_DATASET_DOWNLOAD_CONFIGURATION = "main" #"socratic"

def download_dataset():
    dataset = load_dataset(DEFAULT_DATASET_PATH, DEFAULT_DATASET_DOWNLOAD_CONFIGURATION)
    dataset.save_to_disk(DEFAULT_DATASET_PATH)
    return dataset

def load_dataset_from_disk():
    dataset = load_from_disk(DEFAULT_DATASET_PATH, "default")
    return dataset

def get_dataloaders(batch_size=1):
    ds = load_from_disk(DEFAULT_DATASET_PATH, "default")
    train_dataloader = DataLoader(ds['train'], shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(ds['test'], shuffle=True, batch_size=batch_size)
    return { 'train' : train_dataloader, 'test' : test_dataloader }

def tokenize(tokenizer, example):
    return tokenizer(example["text"])

def preprocess_dataset_dict(tokenizer, ds_dict):
    for k, v in ds_dict.items():
        text_col = []
        for ind, sample in enumerate(ds_dict[k]):
            text_col.append(''.join((sample["question"], sample["answer"])))
        #ds_dict[k] = v.rename_column("question", "input_ids")
        #ds_dict[k] = ds_dict[k].rename_column("answer", "label")

        ds_dict[k] = ds_dict[k].add_column(name="text", column=text_col)
        ds_dict[k] = ds_dict[k].map(lambda x: tokenize(tokenizer, x), batched=True)
    return ds_dict

def gollem_test_train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", default=1)

    nEpochs = 3
    nIterPerEpoch = 5
    evalIter = 100

    learning_rate = 1e-6
    num_training_steps = nEpochs * nIterPerEpoch

    model = gollem.Gollem(gollem.Gollem.default_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device
    model.freeze_for_pe_cft()

    model.olm.tokenizer.pad_token = model.olm.tokenizer.eos_token

    dataloaders_dict = get_dataloaders()
    ds = load_from_disk(DEFAULT_DATASET_PATH, "default")
    # for k, v in ds.items():
    #     ds[k] = v.rename_column("question", "prompt")
    #     #ds[k] = v.rename_column("answer", "label")
    ds_prec = preprocess_dataset_dict(model.olm.tokenizer, ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    #metric =

    progress_bar = tqdm.tqdm(range(num_training_steps))

    # eval_results = trainer.evaluate()
    # return eval_results

    # for epoch in range(nEpochs):
    #     for iter in range(nIterPerEpoch):
    #         batch = dataloaders_dict['train'].next()
    #         #input_prompts = ['\n'.join([batch[0][i], batch[1][i]]) for i in len(batch[0])]
    #         input_prompts = batch[0]
    #         model.olm.tokenizer()

    #         # TODO: don't learn the questions?
    #         for prompt in input_prompts:
    #             output = model(prompt)
    #             loss = output[1]







if __name__ == "__main__":
    gollem_test_train()