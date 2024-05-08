import argparse
import os
import sys

from embeddings.embed import *

def main():
    parser = argparse.ArgumentParser(description="Generate Holographic Embeddings")

    parser.add_argument("embedding_set1", type=str, help="Path to embedding set file")
    parser.add_argument("embedding_set2", type=str, help="Path to embedding set file")
    parser.add_argument("--ins", type=str, help="Correlation Instructions", default="1")
    parser.add_argument("--method", type=str, help="Embedding Method", default="recursive")
    parser.add_argument("--bind", type=str, help="Binding Method", default="conv")
    parser.add_argument("--output_file", type=str, help="Output file", default=f"placeholder")

    args = parser.parse_args()

    if not os.path.exists(args.embedding_set1):
        print(f"Embedding file 1 ({args.embedding_set1}) does not exist")
        sys.exit(1)

    if not os.path.exists(args.embedding_set2):
        print(f"Embedding file 2 ({args.embedding_set2}) does not exist")
        sys.exit(1)

    if args.method == "recursive":
        m = RecursiveEmbedding
    elif args.method == "dummy":
        m = DummyEmbedding
    elif args.method == "position":
        m = PositionEmbedding
    elif args.method == "offset":
        m = OffsetEmbedding
    elif args.method == "cyclic":
        m = CyclicEmbedding
    elif args.method == "sequence":
        m = SequenceEmbedding
    else:
        m = RecursiveEmbedding

    es1 = EmbeddingSet([args.embedding_set1, args.embedding_set2], m, args.bind)
    es1.embed_set(CorrelationInstructions([int(i) for i in args.ins.split(",")]))
    es1.export_embeddings(args.output_file)

if __name__ == "__main__":
    main()

