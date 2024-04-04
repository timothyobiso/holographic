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

# def spherical_distance(embed1, embed2):
#     return torch.acos(torch.dot(embed1, embed2) / (torch.norm(embed1) * torch.norm(embed2)))


# sim_fns = [cosine_similarity, euclidean_distance, hyperbolic_distance]
# this is the only one worth looking at....
sim_fns = [euclidean_distance]

# embedding_files = ["ft_en_g6b300_conv_1.txt", "ft_en_g6b300_corr_1.txt", "cbow_amr3_5ns_deep_TOY.txt", "cbow_amr3_5ns_near_TOY.txt"]
embedding_files = [".vector_cache/glove.6B.50d.txt", ".vector_cache/glove.6B.100d.txt", ".vector_cache/glove.6B.300d.txt"]
# embedding_files = [".vector_cache/wiki.en.vec"]
# embedding_files = ["cbow running on tarski and shannon rn"]
# write a model to obtain the best dummy vector

datasets = {"similarity": ["wordsim353-sim.csv", "wordsim353-rel.csv", "simlex999.csv", "men.csv"],
            "analogy": ["semeval.csv", "sat.csv", "google.csv"],
            "clustering": ["bm.csv", "ap.csv", "bless.csv"],
            "outlier": ["888.csv", "wordsim500.csv"]}
            # "sentence_similarity": []


def run_similarity():
    for f in embedding_files:
        print("Embedding File:\t", f)
        embs = EmbeddingSet.from_embedding_file(f)

        for data in datasets["similarity"]:
            print("Dataset:\t", data)
            dataset = load_eval_dataset(f"../data/similarity/{data}")

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
                            model_sim.append(0.0)  # DEFAULT CASE FOR NOW
                            human_sim.append(score)

                # compute spearman and pearson correlation
                spearman = spearmanr(model_sim, human_sim)
                pearson = pearsonr(model_sim, human_sim)
                print(f"Spearman: {spearman.statistic}")
                print(f"Pearson: {pearson.statistic}\n\n")

def run_analogy():
    # ANALOGY TASK
    # load in dataset
    for f in embedding_files:
        print("Embedding File:\t", f)
        embs = EmbeddingSet.from_embedding_file(f)

        for data in datasets["analogy"]:
            print("Dataset:\t", data)
            dataset = load_eval_dataset(f"../data/analogy/{data}")

            for similarity in sim_fns:

                print("Simiarity Function:\t", similarity.__name__)
                correct = 0
                total = 0
                ranks = []
                for i in range(len(dataset)):
                    word1 = str(dataset.loc[i]["word1"])
                    word2 = str(dataset.loc[i]["word2"])
                    word3 = str(dataset.loc[i]["word3"])
                    word4 = str(dataset.loc[i]["target"])

                    if word1 in embs.embeddings and word2 in embs.embeddings and word3 in embs.embeddings and word4 in embs.embeddings:
                        w1 = embs.embeddings[word1]
                        w2 = embs.embeddings[word2]
                        w3 = embs.embeddings[word3]

                        result = w2 - w1 + w3
                        closest_word = ""
                        closest_score = torch.inf
                        ranking = []
                        for e in embs.embeddings:
                            if (e != word1 and e != word2 and e != word3) or e == word4: # stupid dataset.....
                                score = similarity(embs.embeddings[e], result)
                                if score < closest_score:
                                    closest_score = score
                                    closest_word = e
                                ranking.append((e, score))
                        ranking = sorted(ranking, key=lambda x: x[1])

                        # without knowing the actual closest score
                        ranks.append([x[0] for x in ranking].index(word4) + 1)
                        if closest_word == word4:
                            correct += 1
                        # print(closest_word, word4)
                        total += 1

                print(f"Correct: {correct}/{total} ({correct/total})")
                print(f"Average Rank of Correct Answer: {sum(ranks)/len(ranks)}\n\n")

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

class PositionalModel(nn.Module):
    def __init__(self, embedding_dim, s1, s2, bind="conv"):
        super(PositionalModel, self).__init__()
        self.source1 = s1
        self.source2 = s2
        self.p1 = nn.Parameter(torch.randn(embedding_dim))
        self.p2 = nn.Parameter(torch.randn(embedding_dim))
        if bind == "conv":
            self.bind = conv
        else:
            self.bind = corr

    def forward(self, word1, word2, sim_fn):
        e1 = self.bind(self.source1[self.source1.itos[word1]], self.p1)
        e2 = self.bind(self.source1[self.source1.itos[word1]], self.p2)
        w1 = e1 + e2
        e1 = self.bind(self.source2[self.source1.itos[word2]], self.p1)
        e2 = self.bind(self.source2[self.source1.itos[word2]], self.p2)
        w2 = e1 + e2

        return sim_fn(w1, w2)

class SimplePositionalModel(nn.Module):
    def __init__(self, embedding_dim, s1, s2, bind="conv"):
        super(SimplePositionalModel, self).__init__()
        self.source1 = s1
        self.source2 = s2
        self.p1 = nn.Parameter(torch.randn(embedding_dim))
        self.p2 = nn.Parameter(torch.randn(embedding_dim))
        if bind == "conv":
            self.bind = conv
        else:
            self.bind = corr

    # beacuse of success from doubling up
    def forward(self, word1, word2, sim_fn):
        e1 = self.bind(self.source1[self.source1.itos[word1]], self.p1)
        w1 = e1

        e2 = self.bind(self.source2[self.source2.itos[word2]], self.p2)
        w2 = e2

        return sim_fn(w1, w2)

class SimpleModel(nn.Module):
    def __init__(self, embedding_dim, s1, s2, bind="conv"):
        super(SimpleModel, self).__init__()
        self.source1 = s1
        self.source2 = s2
        self.dummy_vector = nn.Parameter(torch.randn(embedding_dim))
        if bind == "conv":
            self.bind = conv
        else:
            self.bind = corr

    # beacuse of success from doubling up
    def forward(self, word1, word2, sim_fn):
        e1 = self.bind(self.source1[self.source1.itos[word1]], self.dummy_vector)
        w1 = e1

        e2 = self.bind(self.source2[self.source2.itos[word2]], self.dummy_vector)
        w2 = e2

        return sim_fn(w1, w2)


# this model is the typo where i did this actually had good results lol
class SimpleDoubleModel(nn.Module):
    def __init__(self, embedding_dim, s1, s2, bind="conv"):
        super(SimpleDoubleModel, self).__init__()
        self.source1 = s1
        self.source2 = s2
        self.dummy_vector = nn.Parameter(torch.randn(embedding_dim))
        if bind == "conv":
            self.bind = conv
        else:
            self.bind = corr

        # beacuse of success from doubling up

    def forward(self, word1, word2, sim_fn):
        e1 = self.bind(self.source1[self.source1.itos[word1]], self.dummy_vector)
        e2 = self.bind(self.source1[self.source1.itos[word1]], self.dummy_vector)
        w1 = e1 + e2

        e1 = self.bind(self.source2[self.source2.itos[word2]], self.dummy_vector)
        e2 = self.bind(self.source2[self.source2.itos[word2]], self.dummy_vector)

        w2 = e1 + e2

        return sim_fn(w1, w2)

    def convolved_embeddings(self) -> Dict[str, torch.Tensor]:
        embs = {}
        for e in self.source1.itos:
            e1 = self.bind(self.source1[e], self.dummy_vector)
            e2 = self.bind(self.source1[e], self.dummy_vector)
            w1 = e1 + e2
            embs[e] = w1
        return embs



class SimilarityModel(nn.Module):
    def __init__(self, embedding_dim, s1, s2, bind="conv"):
        super(SimilarityModel, self).__init__()
        self.source1 = s1
        self.source2 = s2
        self.dummy_vector = nn.Parameter(torch.randn(embedding_dim))
        if bind == "conv":
            self.bind = conv
        else:
            self.bind = corr

    def forward(self, word1, word2, sim_fn):
        e1 = self.bind(self.source1[self.source1.itos[word1]], self.dummy_vector)
        e2 = self.bind(self.source2[self.source2.itos[word1]], self.dummy_vector)
        w1 = e1 + e2

        e1 = self.bind(self.source1[self.source1.itos[word2]], self.dummy_vector)
        e2 = self.bind(self.source2[self.source2.itos[word2]], self.dummy_vector)
        w2 = e1 + e2

        return sim_fn(w1, w2)


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



def train_similarity(s1, s2):

    count = 0
    for data in datasets["similarity"]:
        print("Dataset:\t", data)
        dataset = load_eval_dataset(f"../data/similarity/{data}")

        similarity = sim_fns[0]
        for m in PositionalModel, SimpleModel, SimpleDoubleModel, SimplePositionalModel, SimilarityModel:
            print("Model:\t", m.__name__)
            model = m(300, s1, s2, "conv")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


            num_epochs = 250
            batch_size = 16

            X1 = dataset["word1"].apply(lambda x: s1.stoi[x] if x in s1.stoi else s1.stoi["the"])
            X2 = dataset["word2"].apply(lambda x: s2.stoi[x] if x in s2.stoi else s2.stoi["the"])
            y = dataset["similarity"]

            X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2)
            train_data = TensorDataset(torch.tensor([x for x in X1_train]), torch.tensor([x for x in X2_train]), torch.tensor([y for y in y_train]))
            test_data = TensorDataset(torch.tensor([x for x in X1_test]), torch.tensor([x for x in X2_test]), torch.tensor([y for y in y_test]))

            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
            torch.autograd.set_detect_anomaly(True)
            model.train()
            loss_fn = nn.MSELoss()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = []
                ys = []
                epoch_loss = 0

                for i, (x1, x2, y) in enumerate(train_loader):

                    output = model(x1, x2, similarity)
                    if (not torch.isnan(output)) and (not torch.isnan(y)):
                        outputs.append(output)
                        ys.append(y)
                        if (i+1) % batch_size == 0 or i == len(train_loader) - 1:
                            # this was so crazy.... v
                            # loss = 1-pearson_correlation(outputs, ys)

                            loss = loss_fn(torch.stack(outputs), torch.stack(ys))
                            epoch_loss += loss.item()
                            # print(f"Epoch {epoch} Batch {(i+1) // batch_size} Loss: {loss.item()}")
                            loss.backward()
                            optimizer.step()
                            outputs = []
                            ys = []
                print(f"Epoch {epoch} Loss: {epoch_loss}")

                if epoch % 50 == 0:
                    model.eval()
                    with torch.no_grad():
                        outputs = []
                        ys = []
                        for i, (x1, x2, y) in enumerate(test_loader):
                            output = model(x1, x2, similarity)
                            if torch.isnan(output) or torch.isnan(y):
                                continue
                            outputs.append(output.item())
                            ys.append(y.item())

                        spearman = spearmanr(outputs, ys)
                        pearson = pearsonr(outputs, ys)
                        print(f"Spearman: {spearman.statistic}")
                        print(f"Pearson: {pearson.statistic}\n\n")

                        if spearman.statistic > .4 or pearson.statistic > .4:
                            with open(f"good_models{count}.txt", "a") as f:
                                count += 1
                                print("BINGO!")
                                if model.p1 is not None:
                                    f.write(f"{m.__name__} {model.p1} {model.p2} {spearman.statistic} {pearson.statistic}\n")
                                else:
                                    f.write(f"{m.__name__} {model.dummy_vector} {spearman.statistic} {pearson.statistic}\n")
                        #print(model.dummy_vector)
            model.eval()
            with torch.no_grad():
                outputs = []
                ys = []
                for i, (x1, x2, y) in enumerate(test_loader):
                    output = model(x1, x2, similarity)
                    if torch.isnan(output) or torch.isnan(y):
                        continue
                    outputs.append(output.item())
                    ys.append(y.item())

                spearman = spearmanr(outputs, ys)
                pearson = pearsonr(outputs, ys)
                print(f"Spearman: {spearman.statistic}")
                print(f"Pearson: {pearson.statistic}\n\n")
                # print(model.dummy_vector)


if __name__ == "__main__":

    # SIMILARITY TASK
    # run_similarity()
    g6b300 = torchtext.vocab.GloVe(name="6B", dim=300)
    ft_en = torchtext.vocab.FastText(language="en")

    m = SimpleDoubleModel(300, g6b300, ft_en, "conv")
    p = m.convolved_embeddings()

    with open("nearest_glove_neighbors.txt", "w") as f:
        embs = {}
        for e in tqdm(g6b300.itos):
            embs[e] = g6b300.vectors[g6b300.stoi[e]]
        for e in tqdm(g6b300.itos):
            f.write(f"{e} {n_nearest_neighbors(embs, embs[e], 10)}\n")

    # put all nearest neighbors in a file
    with open("nearest_neighbors.txt", "w") as f:
        for e in tqdm(p):
            f.write(f"{e} {n_nearest_neighbors(embs, embs[e], 10)}\n")

    # train_similarity(g6b300, ft_en)

    # run_analogy()



    # get embeddings for each word in analogy and answer choices
    # choose closest word to the result
    # score accuracy, precision, recall, f1
    # return scores

    # CLUSTERING TASK
    # load in dataset
    # get embeddings for each word
    # cluster words
    # compare to human clustering
    # return scores

    # OUTLIER DETECTION TASK
    # load in dataset
    # get embeddings for each word
    # detect outliers
    # compare to human outliers
    # return scores

    # SENTENCE SIMILARITY TASK
    # load in dataset
    # get embeddings for each sentence
    # compute similarity between the two embeddings
    # compare to human similarity ratings
    # return scores


