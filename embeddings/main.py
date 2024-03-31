from embed import EmbeddingSet, CorrelationInstructions, RecursiveEmbedding, DummyEmbedding, PositionEmbedding, OffsetEmbedding, CyclicEmbedding, SequenceEmbedding, FullyNested
import torchtext
import penman

def recursive2(source1, source2):
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
    # there are 4 sets of FT vectors
    # 1. wiki+news 1M (16B)
    # 2. wiki+news+subword 1M (16B)
    # 3. crawl 2M (600B)
    # 4. crawl+subword 2M (600B)

    #wiki_news = load_vectors("../data/wiki-news-300d-1M.vec")
    #print(len(wiki_news))
    #print(len(wiki_news["the"]))
    # wiki_news_sub = load_vectors("data/wiki-news-300d-1M-subword.vec")
    # # crawl files are HUGE!! use lab computer
    # # crawl = load_vectors("data/crawl-300d-2M.vec")
    # # crawl_sub = load_vectors("data/crawl-300d-2M-subword.vec")
    #
    # # g6b50 = torchtext.vocab.GloVe(name="6B", dim=50)
    # # g6b100 = torchtext.vocab.GloVe(name="6B", dim=100)
    #
    g6b300 = torchtext.vocab.GloVe(name="6B", dim=300) # 400,000 words
    # g42b300 = torchtext.vocab.GloVe(name="42B", dim=300) # 1.9M words
    # g840b300 = torchtext.vocab.GloVe(name="840B", dim=300) # 2.2M words

    ft_en = torchtext.vocab.FastText(language="en") # 2.5M words
    # ft_simp = torchtext.vocab.FastText(language="simple")


    # sources = [ft_en, ft_simp, g6b300, g42b300, g840b300]
    i1 = CorrelationInstructions([1])

    # Loop to Get All Recursive Sets of 2
    # es1 = EmbeddingSet([ft_en, g6b300], RecursiveEmbedding, "conv")
    # es2 = EmbeddingSet([ft_en, g6b300], RecursiveEmbedding, "corr")
    # es3 = EmbeddingSet([g6b300, ft_en], RecursiveEmbedding, "corr")

    es1 = EmbeddingSet([ft_en, g6b300], OffsetEmbedding, "conv", max_vocab=10000)
    es1.embed_set(i1)

    s = """(c / choose-01
          :ARG1 (c2 / concept :quant 100
                :ARG1-of (i / innovate-01))
          :li 2
          :purpose (e / encourage-01
                :ARG0 c2
                :ARG1 (p / person
                      :ARG1-of (e2 / employ-01))
                :ARG2 (a / and
                      :op1 (r / research-01
                            :ARG0 p)
                      :op2 (d / develop-02
                            :ARG0 p)
                      :time (o / or
                            :op1 (w / work-01
                                  :ARG0 p)
                            :op2 (t2 / time
                                  :poss p
                                  :mod (s / spare))))))"""

    p = FullyNested(penman.decode(s))
    p.embed(es1)
    # es1.export_embeddings("ft_en_g6b300_conv_1.txt")
    # es2.pickle_embeddings("ft_en_g6b300_conv_1.pkl")

    # # Recursive Sets of 3
    # for s1 in sources:
    #     for s2 in sources:
    #         for s3 in sources:
    #             if s1 == s2 or s2 == s3 or s1 == s3:
    #                 continue
    #
    #             # conv is associative and commutative
    #             i1 = CorrelationInstructions([1, 1])
    #             i2 = CorrelationInstructions([2, 1])
    #
    #             # convolution is commutative and associative
    #             r1 = RecursiveEmbedding([s1, s2, s3], "conv")
    #             e1 = r1.embed(i1)
    #
    #             # correlation is NOT commutative and NOT associative (i think, double check paper)
    #             r2 = RecursiveEmbedding([s1, s2, s3], "corr")
    #             e2 = r2.embed(i1) # ((s1 s2) s3)
    #             e3 = r2.embed(i2) # (s1 (s2 s3))
    #
    #             r3 = RecursiveEmbedding([s1, s3, s2], "corr") #
    #             e4 = r3.embed(i1) # ((s1 s3) s2)
    #             e5 = r3.embed(i2) # (s1 (s3 s2))
    #
    #             r4 = RecursiveEmbedding([s2, s1, s3], "corr")
    #             e6 = r4.embed(i1) # ((s2 s1) s3)
    #             e7 = r4.embed(i2) # (s2 (s1 s3))
    #
    #             r5 = RecursiveEmbedding([s2, s3, s1], "corr")
    #             e8 = r5.embed(i1) # ((s2 s3) s1)
    #             e9 = r5.embed(i2) # (s2 (s3 s1))
    #
    #             r6 = RecursiveEmbedding([s3, s1, s2], "corr")
    #             e10 = r6.embed(i1) # ((s3 s1) s2)
    #             e11 = r6.embed(i2) # (s3 (s1 s2))
    #
    #             r7 = RecursiveEmbedding([s3, s2, s1], "corr")
    #             e12 = r7.embed(i1) # ((s3 s2) s1)
    #             e13 = r7.embed(i2) # (s3 (s2 s1))
    #
    # # Recursive Sets of 4
    # for s1 in sources:
    #     for s2 in sources:
    #         for s3 in sources:
    #             for s4 in sources:
    #                 if s1 == s2 or s2 == s3 or s1 == s3 or s1 == s4 or s2 == s4 or s3 == s4:
    #                     continue
    #
    #                 # conv is associative and commutative
    #                 i1 = CorrelationInstructions([1, 1, 1])
    #                 i2 = CorrelationInstructions([2, 1, 1])
    #                 i3 = CorrelationInstructions([2, 2, 1])
    #                 i4 = CorrelationInstructions([3, 2, 1])
    #                 i5 = CorrelationInstructions([1, 2, 1]) # also equivalent to [3, 1, 1]
    #                 # i6 = CorrelationInstructions([3, 1, 1])
    #
    #                 ins_set = [i1, i2, i3, i4, i5]
    #
    #                 # convolution is commutative and associative
    #                 r1 = RecursiveEmbedding([s1, s2, s3, s4], "conv")
    #                 e1 = r1.embed(i1)
    #
    #                 # correlation is NOT commutative and NOT associative (i think, double check paper)
    #                 r2 = RecursiveEmbedding([s1, s2, s3, s4], "corr")
    #                 e2 = [r2.embed(i) for i in ins_set]
    #
    #                 # e2 = r2.embed(i1)
    #                 # e3 = r2.embed(i2)
    #                 # e4 = r2.embed(i3)
    #                 # e5 = r2.embed(i4)
    #                 # e6 = r2.embed(i5)
    #                 # e7 = r2.embed(i6)
    #
    #                 r3 = RecursiveEmbedding([s1, s2, s4, s3], "corr")
    #                 e3 = [r3.embed(i) for i in ins_set]
    #
    #                 r4 = RecursiveEmbedding([s1, s3, s2, s4], "corr")
    #                 e4 = [r4.embed(i) for i in ins_set]
    #
    #                 r5 = RecursiveEmbedding([s1, s3, s4, s2], "corr")
    #                 e5 = [r5.embed(i) for i in ins_set]
    #
    #                 r6 = RecursiveEmbedding([s1, s4, s2, s3], "corr")
    #                 e6 = [r6.embed(i) for i in ins_set]
    #
    #                 r7 = RecursiveEmbedding([s1, s4, s3, s2], "corr")
    #                 e7 = [r7.embed(i) for i in ins_set]
    #
    #                 r8 = RecursiveEmbedding([s2, s1, s3, s4], "corr")
    #                 e8 = [r8.embed(i) for i in ins_set]
    #
    #                 r9 = RecursiveEmbedding([s2, s1, s4, s3], "corr")
    #                 e9 = [r9.embed(i) for i in ins_set]
    #
    #                 r10 = RecursiveEmbedding([s2, s3, s1, s4], "corr")
    #                 e10 = [r10.embed(i) for i in ins_set]
    #
    #                 r11 = RecursiveEmbedding([s2, s3, s4, s1], "corr")
    #                 e11 = [r11.embed(i) for i in ins_set]
    #
    #                 r12 = RecursiveEmbedding([s2, s4, s1, s3], "corr")
    #                 e12 = [r12.embed(i) for i in ins_set]
    #
    #                 r13 = RecursiveEmbedding([s2, s4, s3, s1], "corr")
    #                 e13 = [r13.embed(i) for i in ins_set]
    #
    #                 r14 = RecursiveEmbedding([s3, s1, s2, s4], "corr")
    #                 e14 = [r14.embed(i) for i in ins_set]
    #
    #                 r15 = RecursiveEmbedding([s3, s1, s4, s2], "corr")
    #                 e15 = [r15.embed(i) for i in ins_set]
    #
    #                 r16 = RecursiveEmbedding([s3, s2, s1, s4], "corr")
    #                 e16 = [r16.embed(i) for i in ins_set]
    #
    #                 r17 = RecursiveEmbedding([s3, s2, s4, s1], "corr")
    #                 e17 = [r17.embed(i) for i in ins_set]
    #
    #                 r18 = RecursiveEmbedding([s3, s4, s1, s2], "corr")
    #                 e18 = [r18.embed(i) for i in ins_set]
    #
    #                 r19 = RecursiveEmbedding([s3, s4, s2, s1], "corr")
    #                 e19 = [r19.embed(i) for i in ins_set]
    #
    #                 r20 = RecursiveEmbedding([s4, s1, s2, s3], "corr")
    #                 e20 = [r20.embed(i) for i in ins_set]
    #
    #                 r21 = RecursiveEmbedding([s4, s1, s3, s2], "corr")
    #                 e21 = [r21.embed(i) for i in ins_set]
    #
    #                 r22 = RecursiveEmbedding([s4, s2, s1, s3], "corr")
    #                 e22 = [r22.embed(i) for i in ins_set]
    #
    #                 r23 = RecursiveEmbedding([s4, s2, s3, s1], "corr")
    #                 e23 = [r23.embed(i) for i in ins_set]
    #
    #                 r24 = RecursiveEmbedding([s4, s3, s1, s2], "corr")
    #                 e24 = [r24.embed(i) for i in ins_set]
    #
    #                 r25 = RecursiveEmbedding([s4, s3, s2, s1], "corr")
    #                 e25 = [r25.embed(i) for i in ins_set]