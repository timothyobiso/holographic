#Visualize emeddings using PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchtext
from embeddings.embed import EmbeddingSet

def main():
    # load embedding sets from file
    s = EmbeddingSet.from_embedding_file("../embeddings/recur_conv_corr/ft_en_g6b300_corr_1.txt")
    for dataset in "wordsim353-rel", "wordsim353-sim", "simlex999", "men":
        df = pd.read_csv(f"../data/similarity/{dataset}.csv", header=0, sep=",", names=["word1", "word2", "similarity"])

        # turn the columns word1 and word2 into a list
        words = list(set(df["word1"].tolist() + df["word2"].tolist()))
        if dataset == "men":
            words = [w[:-2] for w in words]

        # s = torchtext.vocab.GloVe(name="6B", dim=300, cache="../embeddings/.vector_cache")
        # for file in "embeddings/conv.txt", "embeddings/corr.txt":
        #     s = EmbeddingSet.from_embedding_file(file)
        print("hi")
        pca = PCA(n_components=2)

        # only visualize words in words
        relevant_vectors = {k: np.array(s.embeddings[k]) for k in words if k in s.embeddings}

        vals = list(relevant_vectors.values())

        # visualize embeddings
        pca.fit(vals)
        X = pca.transform(vals)
        plt.scatter(X[:, 0], X[:, 1], s=5)

        for i, word in enumerate(relevant_vectors):
            if i % 2 == 0:
                plt.annotate(word, (X[:, 0][i], X[:, 1][i]))

        for a in plt.gca().texts:
            a.set_size(5)

        plt.title(f"PCA of {dataset}")
        fig = plt.gcf()
        # save plot
        plt.savefig(f"corr_{dataset}_half.png", dpi=1500)

        fig.show()


if __name__ == "__main__":
    main()



