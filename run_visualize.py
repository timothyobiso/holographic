import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from embeddings.embed import EmbeddingSet


# example call to this file: python run_visualize.py data/holographic.txt --sim data/word_similarity --output_folder word_similarity_visualization

def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings using PCA')
    parser.add_argument('embedding_file', type=str, help='Path to embedding file')
    parser.add_argument('--sim', type=str, help='Path to word similarity data', default='data/word_similarity')
    parser.add_argument('--output_folder', type=str, default='word_similarity_visualization', help='Output folder')

    parser.add_argument('--every_n', type=int, default=1, help='Visualize every n-th word')
    parser.add_argument('--font_size', type=int, default=5, help='Font size of the words in the plot')
    parser.add_argument('--dpi', type=int, default=1000, help='DPI of the plot')
    parser.add_argument('--no_show', type=bool, default=False, help='Don\'t show the plot')
    args = parser.parse_args()


    # load embedding sets from file
    s = EmbeddingSet.from_embedding_file(args.embedding_file)
    for dataset in "wordsim353-rel", "wordsim353-sim", "simlex999", "men":
        df = pd.read_csv(f"{args.sim}{dataset}.csv", header=0, sep=",", names=["word1", "word2", "similarity"])

        # turn the columns word1 and word2 into a list
        words = list(set(df["word1"].tolist() + df["word2"].tolist()))
        if dataset == "men":
            words = [w[:-2] for w in words]

        pca = PCA(n_components=2)

        # only visualize words in words
        relevant_vectors = {k: np.array(s.embeddings[k]) for k in words if k in s.embeddings}

        vals = list(relevant_vectors.values())

        # visualize embeddings
        pca.fit(vals)
        X = pca.transform(vals)
        plt.scatter(X[:, 0], X[:, 1], s=5)

        for i, word in enumerate(relevant_vectors):
            if i % args.every_n == 0:
                plt.annotate(word, (X[:, 0][i], X[:, 1][i]))

        for a in plt.gca().texts:
            a.set_size(args.font_size)

        plt.title(f"PCA of {dataset}")
        fig = plt.gcf()
        # save plot
        plt.savefig(f"{args.output_folder}/{dataset}.png", dpi=args.dpi)

        if args.no_show:
            plt.close(fig)
        else:
            fig.show()


if __name__ == "__main__":
    main()