import io
from tqdm import tqdm
import torch

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin, total=999994):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(torch.tensor, tokens[1:])
    return data