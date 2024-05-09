import argparse
import os
import sys

import torch
import torchhd

from scripts.run_amr import load_graphs, get_vocab, train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="../data/amr/")
    parser.add_argument("--vocab", type=str)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset {args.dataset} does not exist.")
        sys.exit(1)

    if not os.path.exists(args.vocab):
        print(f"Vocab {args.vocab} does not exist.")
        sys.exit(1)

    graphs = load_graphs(args.dataset)
    if args.vocab is not None and args.dataset != args.vocab:
        vocab = get_vocab(load_graphs(args.vocab))
    else:
        vocab = get_vocab(graphs)

    hvs = torchhd.HRRTensor.random(len(vocab), args.embedding_size, device=args.device)

    train(graphs, hvs, vocab, d=args.embedding_size, n=args.n, device=args.device)

