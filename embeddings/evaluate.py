from typing import Dict

import pandas as pd
from scipy.stats.mstats import spearmanr, pearsonr
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torchtext
from tqdm import tqdm

from utils import conv, corr

from embed import EmbeddingSet


def load_eval_dataset(file):
    # detect and fix differences so it is in the format of
    # word1, word2, score
    return pd.read_csv(file)


# 1 if similar, 0 if not similar
def cosine_similarity(embed1, embed2):
    # 1 if similar, 0 if not similar
    return torch.dot(embed1, embed2) / (torch.norm(embed1) * torch.norm(embed2))


def euclidean_distance(embed1, embed2):
    # compute the euclidean distance between two embeddings
    return torch.norm(embed1 - embed2)


def hyperbolic_distance(embed1, embed2):
    x = torch.clamp(embed1, -1 + 1e-15, 1 - 1e-15)
    y = torch.clamp(embed2, -1 + 1e-15, 1 - 1e-15)

    # Calculate the squared hyperbolic distance
    d = torch.sum((x - y) ** 2) / (1 - torch.sum(x ** 2)) / (1 - torch.sum(y ** 2))

    # Get the hyperbolic distance
    return torch.arccosh(1 + 2 * d)

def spherical_distance(embed1, embed2):
    return torch.acos(torch.dot(embed1, embed2) / (torch.norm(embed1) * torch.norm(embed2)))


sim_fns = [cosine_similarity, euclidean_distance, hyperbolic_distance, spherical_distance]

datasets = ["wordsim353-sim.csv", "wordsim353-rel.csv", "simlex999.csv", "men.csv"]
            # "sentence_similarity": []


def run_similarity(embedding_file, sim_path="../data/similarity/"):
    embs = EmbeddingSet.from_embedding_file(embedding_file)

    for data in datasets:
        print("Dataset:\t", data)
        dataset = load_eval_dataset(f"{sim_path}{data}")

        for similarity in sim_fns:
            print("Simiarity Function:\t", similarity.__name__)

            model_sim = []
            human_sim = []
            for i in range(len(dataset)):
                word1 = str(dataset.loc[i]["word1"])
                word2 = str(dataset.loc[i]["word2"])

                # men.csv has words with -n, -v, -j at the end for POS
                if word1.endswith("-n") or word1.endswith("-v") or word1.endswith("-j"):
                    word1 = word1[:-2]
                if word2.endswith("-n") or word2.endswith("-v") or word2.endswith("-j"):
                    word2 = word2[:-2]

                score = dataset.loc[i]["similarity"]
                if np.isnan(score):
                    continue

                vocab_only = True  # SET HERE

                if word1 in embs.embeddings and word2 in embs.embeddings:
                    model_sim.append(float(similarity(embs.embeddings[word1], embs.embeddings[word2])))
                    human_sim.append(score)
                else:
                    if not vocab_only:
                        model_sim.append(0.0)
                        human_sim.append(score)

            # compute spearman and pearson correlation
            spearman = spearmanr(model_sim, human_sim)
            pearson = pearsonr(model_sim, human_sim)
            print(f"Spearman: {spearman.statistic}")
            print(f"Pearson: {pearson.statistic}\n\n")


def pearson_correlation(x, y):
    x1 = torch.stack(x)
    y1 = torch.stack(y)
    # x2 = x1.reshape(1, x1.size()[0])
    # y2 = y1.reshape(1, y1.size()[0])
    mean_x = torch.mean(x1)
    mean_y = torch.mean(y1)
    xm, ym = x1 - mean_x, y1 - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    return r


def n_nearest_neighbors(embeddings_dict, target_embedding, n):
    # get the n nearest neighbors to the word

    if not isinstance(target_embedding, torch.Tensor):
        target_embedding = torch.tensor(target_embedding)
    if target_embedding.dim() == 1:
        target_embedding = target_embedding.unsqueeze(0)

        # Convert embeddings dictionary to a tensor
    embeddings = torch.stack(list(embeddings_dict.values()))
    keys = list(embeddings_dict.keys())

    # Calculate distances from the target_embedding to all other embeddings
    distances = torch.cdist(target_embedding, embeddings).squeeze(0)

    # Find the indices of the n nearest neighbors
    _, indices = distances.topk(n, largest=False, sorted=True)

    # Map indices back to keys and their distances
    neighbors = [(keys[idx], distances[idx].item()) for idx in indices]

    return neighbors


if __name__ == "__main__":

    # g6b300 = torchtext.vocab.GloVe(name="6B", dim=300)
    # ft_en = torchtext.vocab.FastText(language="en")

    run_similarity(embedding_file="ft_en_g6b300_conv_1.txt")
