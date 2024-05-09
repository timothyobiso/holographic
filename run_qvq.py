import argparse
import os
import sys

from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
from torch import no_grad

def main():
    parser = argparse.ArgumentParser(description='Run QVQ')

    parser.add_argument('--model_size', type=str, help='Size of the BART model', default='base')
    parser.add_argument('--dataset', type=str, help='Folder containing the (train and) test file(s)')

    parser.add_argument('clean', type=bool, help='Whether to clean the dataset', default=False)

    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--eval_every', type=int, help='Evaluate every n steps')

    parser.add_argument('--verbose', type=bool, help='Print outputs', default=False)

    # parser.add_argument('train', type=bool, help='Whether to train the model', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset folder ({args.dataset}) does not exist")
        sys.exit(1)

    # if eval_every was not set, set it to the number of epochs
    if args.eval_every is None:
        args.eval_every = args.epochs

    # Set up model

    tokenizer = BartTokenizer.from_pretrained(f"facebook/bart-{args.model_size}")
    model = BartForConditionalGeneration.from_pretrained(f"facebook/bart-{args.model_size}")

    # Set up data

    train_bind = pd.read_csv(f"{args.dataset}/train_binds.txt", sep="\t", header=None)
    train_sent = pd.read_csv(f"{args.dataset}/train.txt", sep="\t", header=None)

    test_bind = pd.read_csv(f"{args.dataset}/test_binds.txt", sep="\t", header=None)
    test_sent = pd.read_csv(f"{args.dataset}/test.txt", sep="\t", header=None)

    train_df = pd.concat([train_sent, train_bind], axis=1)
    test_df = pd.concat([test_sent, test_bind], axis=1)

    train_df.columns = ["sent", "labels"]
    test_df.columns = ["sent", "labels"]

    # remove rows with "skip" or "SKIP" anywhere in the label
    train_df = train_df[~train_df["labels"].str.contains("skip|SKIP")]
    test_df = test_df[~test_df["labels"].str.contains("skip|SKIP")]

    # load into huggingface dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # tokenize the data
    def tokenize_function(text):
        return tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].squeeze()

    train_inputs = torch.stack(list(map(tokenize_function, train_dataset["sent"])))
    train_labels = torch.stack(list(map(tokenize_function, train_dataset["labels"])))
    test_inputs = torch.stack(list(map(tokenize_function, test_dataset["sent"])))
    test_labels = torch.stack(list(map(tokenize_function, test_dataset["labels"])))
    train_labels[train_labels == tokenizer.pad_token_id] = -100
    test_labels[test_labels == tokenizer.pad_token_id] = -100
    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # train and evaluate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    def e():
        model.eval()
        with no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, labels = [x.to(model.device) for x in batch]
                outputs = model.generate(input_ids=input_ids, labels=labels)

                if args.verbose:
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
                    print(f"Output: {output_text}")

                # Output to File
                for output in outputs:
                    output_text = tokenizer.decode(output, skip_special_tokens=True)
                    with open(f"/content/drive/MyDrive/T5/bartbasenoskip{epoch + 1}epochs.txt", "a") as f:
                        f.write(f"{output_text}\n")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            input_ids, labels = [x.to(model.device) for x in batch]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Average Loss: {total_loss / len(train_loader)}")
        if (epoch + 1) % args.eval_every == 0 and epoch != 0:
            e()
            model.train()








