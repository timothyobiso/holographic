# import io
from tqdm import tqdm
import penman
from torch.fft import rfft, irfft
import numpy as np
import torch

# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in tqdm(fin, total=999994):
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(torch.tensor, tokens[1:])
#     return data

def conv(a, b):
    return irfft(torch.mul(rfft(a), rfft(b)))


def corr(a, b):
    return irfft(torch.mul(np.conj(rfft(a)), rfft(b)))

def parents(g: penman.Graph, child):
    return [(p, r) for p, r, c in g.triples if c == child]


def children(g: penman.Graph, parent):
    return [(r, c) for p, r, c in g.triples if p == parent and r != ":instance"]

def grandchildren(g: penman.Graph, grandparent):
    return [children(g, _) for _ in children(g, grandparent)]

def has_children(g: penman.Graph, parent):
    return len(children(g, parent)) > 0

def has_grandchildren(g: penman.Graph, grandparent):
    return len(grandchildren(g, grandparent)) > 0

def relu(x):
    result = x.copy()
    result[result < 0] = 0
    return result

def softmax(x):
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / s