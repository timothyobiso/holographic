from typing import List, Dict
from tqdm import tqdm
import pickle
from utils import *

from torch import nn
import torch


class CorrelationInstructions:
    def __init__(self, instructions: List[int]):
        self.instructions = instructions  # [space, space, ...]

    def __iter__(self):
        return iter(self.instructions)

def childless_siblings(g: penman.Graph):
    siblings = {}
    names = {}
    childless = set()
    has_children = set()
    for parent, relation, child in g.triples:
        if relation == ":instance":
            names[parent] = child
            continue
        if parent in siblings:
            siblings[parent].append((relation, child))
        else:
            siblings[parent] = [(relation, child)]
        childless.add(child)
        has_children.add(parent)
        if parent in childless:
            childless.remove(parent)
    return childless, siblings, names


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
    def __init__(self, sources=None, strategy=None, binder="conv", vocab_index=0, max_vocab=-1, file=None):
        if file is not None:
            # read in embeddings from file
            self.embeddings = {}
            self.stoi = {}
            self.itos = []
            with open(file, "r") as f:
                for line in tqdm(f, desc="Reading Embeddings"):
                    if len(line.split(" ")) == 2:
                        continue
                    tokens = line.strip().split(" ")
                    # if tokens[1] is not a float
                    self.embeddings[tokens[0]] = torch.tensor(list(map(float, tokens[1:])))
                    self.itos.append(tokens[0])
                    self.stoi[tokens[0]] = len(self.itos) - 1
        else:
            self.embeddings: Dict[HolographicWordEmbedding] = {}
            self.strategy = strategy
            self.sources = sources
            self.stoi = self.sources[vocab_index].stoi
            self.itos = self.sources[vocab_index].itos
            for e in tqdm(self.itos, desc="Creating Embedding Set"):
                # if not in embedding source, tensor of zeros
                self.embeddings[e] = self.strategy([s[e] if e in s else torch.rand(s[e].size()) for s in sources], binder)

        if max_vocab != -1:
            self.itos = self.itos[:max_vocab]
        self.binder = binder

        if binder == "conv":
            self.bind = conv
        else:
            self.bind = corr

    def embed_set(self, instructions: CorrelationInstructions = None):
        for e in tqdm(self.itos, desc="Embedding..."):
            self.embeddings[e].embed(instructions)

    def export_embeddings(self, outfile):
        with open(outfile, "w") as f:
            for e in tqdm(self.itos, desc="Exporting Embeddings..."):
                try:
                    f.write(f"{e} {' '.join(map(lambda a: str(float(a)), self.embeddings[e].embedding))}\n")
                except:
                    print(e, self.embeddings[e].embedding)

    def pickle_embeddings(self, outfile):
        torch.save(self.embeddings, outfile)

    @classmethod
    def from_embedding_file(cls, file, binder="conv"):
        return cls(file=file, binder=binder)

class HolographicSentenceEmbedding:
    def __init__(self, amr, binder="conv", length=300, dim=1):
        if dim == 1:
            self.embedding = torch.zeros((dim,length))
        else:
            self.embedding = torch.zeros(length)
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
            instructions = CorrelationInstructions([1] * (len(t_emb)-1))
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
    def __init__(self, sources, binder, offset=1, finish=False):
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


# TODO: deal with reentrant nodes
class FullyNested(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # childless, siblings, names = childless_siblings(self.amr)
        def _embed(rel, top):
            if has_children(self.amr, top):
                return self.bind(embeddings.embeddings[top].embedding,
                                 np.sum([_embed(r, c) for r, c in children(self.amr, top)], axis=0))
            return self.bind(embeddings.embeddings[rel].embedding, embeddings.embeddings[top].embedding)

        self.embedding = _embed(":snt", self.amr.top)


class VerbFrame(HolographicSentenceEmbedding):
    def __init__(self, amr, binder="conv"):
        super().__init__(amr, binder)

    def embed(self, embeddings: EmbeddingSet):
        # same as fully nested, and bundle with any verb frames
        def _embed(rel, top):
            if has_children(self.amr, top):
                return self.bind(embeddings.embeddings[top].embedding,
                                 np.sum([_embed(r, c) for r, c in children(self.amr, top)], axis=0))
            return self.bind(embeddings.embeddings[rel].embedding, embeddings.embeddings[top].embedding)

        verbs = [v for v in self.amr.nodes if v.is_verb()]
        self.embedding = np.sum([_embed(":snt", self.amr.top)] + [_embed(v) for v in verbs], axis=0)

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