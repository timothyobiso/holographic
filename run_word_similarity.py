import argparse
import os
import sys

from embeddings.evaluate import run_similarity

# example call to this file: python run_word_similarity.py data/holographic.txt --sim data/word_similarity

def main():
    parser = argparse.ArgumentParser(description="Run word similarity evaluation")
    parser.add_argument("embedding_file", type=str, help="Path to embedding file")
    parser.add_argument("--sim", type=str, help="Path to word similarity data", default="data/word_similarity")

    args = parser.parse_args()

    if not os.path.exists(args.embedding_file):
        print(f"Embedding file ({args.embedding_file}) does not exist")
        sys.exit(1)

    if not os.path.exists(args.sim):
        print(f"Word similarity data folder ({args.sim}) does not exist")
        sys.exit(1)

    run_similarity(embedding_file=args.embedding_file, sim_path=args.sim)



if __name__ == "__main__":
    main()