import numpy as np
import torch
from mteb import MTEB
from mteb import MTEB_MAIN_EN
from tqdm import tqdm

class MyAMRModel():
	def __init__(self, file):
		self.file = file
		self.embs = {}

class MyModel():

	def __init__(self, file):
		self.file = file
		self.embs = {}
		with open(self.file, "r") as f:
                        for line in tqdm(f, desc="Loading Embedding"):
                                tokens = line.strip().split(" ")
                                self.embs["".join(tokens[:-300]).lower()] = torch.tensor(list(map(float, tokens[-300:])))

	def encode(self, sentences: list[str], **kwargs) -> list[np.ndarray] | list[torch.Tensor]:
		embs = []
		for text in sentences:
			embs.append(1/len(text) * torch.sum(torch.stack([self.embs[token.lower()] if token.lower() in self.embs else self.embs["the"] for token in text.split(" ")]), 0))
		# emb = [self.embs[token.lower()] if token.lower() in self.embs else self.embs["the"] for text in sentences for token in text.split(" ")]
		return embs


if __name__ == "__main__":
	file = "ft_en_g6b300_conv_1.txt"
	model = MyModel(file)
	evaluation = MTEB(tasks=MTEB_MAIN_EN, task_langs=["en"])
	evaluation.run(model, output_folder=f"results/{file}")
