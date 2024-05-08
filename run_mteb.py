# run scripts/run_mteb.py
# EX call: python run_mteb.py ft_en_g6b300_conv_1.txt --output_folder results

import argparse
import os
import sys

from mteb import MTEB, MTEB_MAIN_EN

from scripts.run_mteb import NaiveModel

def main():
    parser = argparse.ArgumentParser(description='Run MTEB')
    parser.add_argument('embedding_set', type=str, help='Path to embedding set file')
    parser.add_argument('--output_folder', type=str, default='results', help='Output folder')
    args = parser.parse_args()

    if not os.path.exists(args.embedding_set):
        print(f"Embedding file 1 ({args.embedding_set}) does not exist")
        sys.exit(1)

    model = NaiveModel(args.embedding_set)
    evaluation = MTEB(tasks=MTEB_MAIN_EN, task_langs=["en"])
    evaluation.run(model, output_folder=args.output_folder)


if __name__ == "__main__":
    main()