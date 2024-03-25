from typing import List, Dict
from tqdm import tqdm
import pickle

from torch import nn
import torch
from torch.fft import rfft, irfft
import numpy as np

class CorrelationInstructions:
    def __init__(self, instructions: List[int]):
        self.instructions = instructions # [space, space, ...]

    def __iter__(self):
        return iter(self.instructions)

def conv(a, b):
    return irfft(np.multiply(rfft(a), rfft(b)))

def corr(a, b):
    return irfft(np.multiply(np.conj(rfft(a)), rfft(b)))


class HolographicWordEmbedding:
    def __init__(self, sources: list, binder="conv"):
        self.embedding = torch.zeros_like(sources[0])
        self.sources = sources # list of vectors of equal length
        if binder == "conv": # is convolution
            self.bind = conv
        else: # is correlation
            self.bind = corr

    def embed(self, instructions: CorrelationInstructions = None):
        raise NotImplementedError("Embedding must be implemented in subclass")
class EmbeddingSet:
    def __init__(self, sources, strategy, binder="conv", vocab_index=0, max_vocab=-1):

        self.embeddings: Dict[HolographicWordEmbedding] = {}
        self.strategy = strategy
        self.binder = binder
        self.sources = sources
        self.stoi = self.sources[vocab_index].stoi
        self.itos = self.sources[vocab_index].itos
        if max_vocab != -1:
            self.itos = self.itos[:max_vocab]


        for e in tqdm(self.itos, desc="Creating Embedding Set"):
            self.embeddings[e] = self.strategy([s[e] for s in sources], binder)

    def embed_set(self, instructions: CorrelationInstructions = None):
        for e in tqdm(self.itos, desc="Embedding..."):
            self.embeddings[e].embed(instructions)

    def export_embeddings(self, outfile):
        with open(outfile, "w") as f:
            for e in tqdm(self.itos, desc="Exporting Embeddings..."):
                f.write(f"{e} {' '.join(map(lambda a: str(float(a)), self.embeddings[e].embedding))}\n")

    def pickle_embeddings(self, outfile):
        torch.save(self.embeddings, outfile)

class HolographicSentenceEmbedding:
    def __init__(self, amr, binder="conv"):
        self.embedding = torch.zeros_like()
        self.amr = amr
        if binder == "conv":
            self.bind = conv
        else:
            self.bind = corr

    def embed(self, embedding: EmbeddingSet):
        raise NotImplementedError("Embedding must be implemented in subclass")

class RecursiveEmbedding(HolographicWordEmbedding):
    def __init__(self, sources, binder):
        super().__init__(sources, binder)

    def embed(self, instructions: CorrelationInstructions = None):
        t_emb = self.sources.copy()
        if instructions is None:
            instructions = CorrelationInstructions(list(range(1, len(t_emb))))
        for gap in instructions:
            t_emb[gap-1] = self.bind(t_emb[gap-1], t_emb[gap])
            del t_emb[gap]
        self.embedding = t_emb[0]

        return self.embedding


class DummyEmbedding(HolographicWordEmbedding):
    def __init__(self, sources, binder, dummy):
        super().__init__(sources, binder)
        self.dummy = dummy

    def embed(self, instructions: CorrelationInstructions = None):
        t_emb = self.sources.copy()
        for e in t_emb:
            self.embedding += self.bind(self.dummy, e)
        return self.embedding


class PositionEmbedding(HolographicWordEmbedding):
    def __init__(self, sources, binder, pos):
        super().__init__(sources, binder)
        self.pos = pos

    def embed(self, instructions: CorrelationInstructions = None):
        t_emb = self.sources.copy()
        for p, e in zip(self.pos, t_emb):
            self.embedding += self.bind(p, e)
        return self.embedding


class OffsetEmbedding(HolographicWordEmbedding):
    def __init__(self, sources, binder, offset: int=1, finish=False):
        super().__init__(sources, binder)
        self.offset = offset
        self.finish = finish

    def embed(self, instructions: CorrelationInstructions = None):
        t_emb = self.sources.copy()
        # a + b + a * c + b * d...
        for i in range(self.offset):
            self.embedding += t_emb[i]
        for i in range(self.offset, len(t_emb)):
            self.embedding += self.bind(t_emb[i-self.offset], t_emb[i])
        if self.finish:
            for i in range(self.offset, len(t_emb)):
                self.embedding += t_emb[i]
        return self.embedding


class CyclicEmbedding(HolographicWordEmbedding):
    def __init__(self, sources, binder, offset: int=1, rev=False):
        super().__init__(sources, binder)
        self.offset = offset
        self.rev = rev

    def embed(self, instructions: CorrelationInstructions = None):
        t_emb = self.sources.copy()
        for i in range(len(t_emb)):
            if self.rev:
                self.embedding += self.bind(t_emb[i], t_emb[self.offset-i % len(t_emb)])
            else:
                self.embedding += self.bind(t_emb[i], t_emb[i+self.offset % len(t_emb)])
        return self.embedding


class SequenceEmbedding(HolographicWordEmbedding):
    def __init__(self, sources, binder):
        super().__init__(sources, binder)

    def embed(self, instructions: CorrelationInstructions = None):
        t_emb = self.sources.copy()
        for i in range(len(t_emb)):
            e = t_emb[0]
            for j in range(i):
                e = self.bind(e, t_emb[j])
            self.embedding += e
        return self.embedding


class FullyNested(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # go through amr, bind relation to child, then bundle siblings, then work up to top
        # TODO
        pass

class VerbFrame(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass

class PseudoLinearized(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass

class Unwrapped(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass


class Linearized(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass


class DualEncoding(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass

class LayeredEncoding(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass


class SubgraphEncoding(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # TODO
        pass


