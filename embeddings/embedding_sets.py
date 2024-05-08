from embed import EmbeddingSet, CorrelationInstructions, RecursiveEmbedding, DummyEmbedding, PositionEmbedding, OffsetEmbedding, CyclicEmbedding, SequenceEmbedding, FullyNested
import torchtext
import penman


def recursive_word_embedding(source1, source2):
    instructions = CorrelationInstructions([1])
    r1 = RecursiveEmbedding([source1, source2], "conv")
    e1 = r1.embed(instructions)

    r2 = RecursiveEmbedding([source1, source2], "corr")
    e2 = r2.embed(instructions)

    r3 = RecursiveEmbedding([source2, source1], "corr")
    e3 = r3.embed(instructions)

    return e1, e2, e3


if __name__ == "__main__":
    # obtain the embeddings
    g6b300 = torchtext.vocab.GloVe(name="6B", dim=300)  # 400,000 words
    # g42b300 = torchtext.vocab.GloVe(name="42B", dim=300) # 1.9M words
    # g840b300 = torchtext.vocab.GloVe(name="840B", dim=300) # 2.2M words

    ft_en = torchtext.vocab.FastText(language="en")  # 2.5M words

    i1 = CorrelationInstructions([1])

    # All Recursive Sets of 2
    # es1 = EmbeddingSet([ft_en, g6b300], RecursiveEmbedding, "conv")
    # es2 = EmbeddingSet([ft_en, g6b300], RecursiveEmbedding, "corr")
    # es3 = EmbeddingSet([g6b300, ft_en], RecursiveEmbedding, "corr")

    method = "conv"  # "corr"

    es1 = EmbeddingSet([ft_en, g6b300], RecursiveEmbedding, method)
    es1.embed_set(i1)
    es1.export_embeddings(f"ft_en_g6b300_{method}_1.txt")

    # s = """(c / choose-01
    #       :ARG1 (c2 / concept
    #             :quant 100
    #             :ARG1-of (i / innovate-01))
    #       :li 2
    #       :purpose (e / encourage-01
    #             :ARG0 c2
    #             :ARG1 (p / person
    #                   :ARG1-of (e2 / employ-01))
    #             :ARG2 (a / and
    #                   :op1 (r / research-01
    #                         :ARG0 p)
    #                   :op2 (d / develop-02
    #                         :ARG0 p)
    #                   :time (o / or
    #                         :op1 (w / work-01
    #                               :ARG0 p)
    #                         :op2 (t2 / time
    #                               :poss p
    #                               :mod (s / spare))))))"""
    #
    # p = FullyNested(penman.decode(s))
    # p.embed(es1)
