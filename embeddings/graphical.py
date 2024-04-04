from embed import EmbeddingSet
from utils import *
class CBOW:
    def __init__(self, dataset, method, window_size, embedding_size, learning_rate, epochs, batch_size, neg_samples, subsampling, seed):
        self.dataset = penman.load(dataset)
        self.method = method
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.subsampling = subsampling
        self.seed = seed
        self.vocab = set()

        self.var2vocab = {}
        self.vocab2var = {}

        for graph in self.dataset:
            for triple in graph.triples:
                if triple[1] == ":instance":
                    self.var2vocab[triple[0]] = triple[2]
                    self.vocab2var[triple[2]] = triple[0]
            for e in list(graph.variables()):
                self.vocab.add(self.var2vocab[e])
            for e in graph.edges():
                self.vocab.add(e.role)
            for e in graph.attributes():
                self.vocab.add(e.role)
            self.vocab.add(":snt")
            self.vocab.add("<UNK>")
        print("Vocab Size:", len(self.vocab))

        self.embeddings = np.random.uniform(-1, 1, (len(self.vocab), self.embedding_size)) # experiment with different distributions
        self.W1 = np.random.uniform(-1, 1, (self.embedding_size, len(self.vocab)))
        self.b1 = np.random.uniform(-1, 1, self.embedding_size)
        self.W2 = np.random.uniform(-1, 1, (len(self.vocab), self.embedding_size))
        self.b2 = np.random.uniform(-1, 1, len(self.vocab))

        self.word2index = {}
        self.index2word = {}
        for i, word in enumerate(self.vocab):
            self.word2index[word] = i
            self.index2word[i] = word

    def one_hot(self, word):
        vec = np.zeros(len(self.vocab))
        if word not in self.word2index:
            vec[self.word2index["<UNK>"]] = 1
        else:
            vec[self.word2index[word]] = 1
        return vec

    def context_to_vector(self, context):
        vec = np.zeros(len(self.vocab))
        for word in context:
            vec += self.one_hot(word)
        return vec / len(context)
    @staticmethod
    def fill_near(p_list, c_list, p_window_size, c_window_size, graph):
        if len(p_list) == 0:
            before_context = [":snt"] * p_window_size
        else:
            before_context = []
        if len(c_list) == 0:
            after_context = ["<UNK>"] * c_window_size
        else:
            after_context = []


        if len(p_list) > 0:
            while len(before_context) < p_window_size:
                for p in p_list:
                    before_context.insert(0, p[1])
                for p in p_list:
                    before_context.insert(0, p[0])
                np_list = []
                for p in p_list:
                    for np in parents(graph, p[0]):
                        np_list.append(np)
                p_list = np_list

        before_context = before_context[::-1][:p_window_size][::-1]

        if len(c_list) > 0:
            while len(after_context) < c_window_size:
                for c in c_list:
                    after_context.append(c[0])
                for c in c_list:
                    after_context.append(c[1])
                nc_list = []
                for c in c_list:
                    for nc in children(graph, c[1]):
                        nc_list.append(nc)
                c_list = nc_list

        after_context = after_context[:c_window_size]

        context_window = before_context + after_context

        return context_window

    @staticmethod
    def fill_deep(e, p_window_size, c_window_size, graph):
        before_context = []
        p = parents(graph, e)
        if len(p) == 0:
            while len(before_context) < p_window_size:
                before_context.insert(0, ":snt")
        else:
            p = p[0]
        count = 0
        while len(before_context) < p_window_size:

            before_context.insert(0, p[1])
            before_context.insert(0, p[0])
            p = parents(graph, p[0])
            if len(p) == 0:
                if len(parents(graph, e)) > count+1:
                    p = parents(graph, e)[count+1]
                    count += 1
                else:
                    break
            else:
                p = p[0]
        while len(before_context) < p_window_size:
            before_context.insert(0, before_context[0])

        before_context = before_context[::-1][:p_window_size][::-1]

        after_context = []
        c = children(graph, e)
        if len(c) == 0:
            while len(after_context) < p_window_size:
                after_context.append("random")
        else:
            c = c[0]
        count = 0
        while len(after_context) < c_window_size:

            after_context.append(c[0])
            after_context.append(c[1])
            c = children(graph, c[1])
            if len(c) == 0:
                if len(children(graph, e)) > count+1:
                    c = children(graph, e)[count+1]
                    count += 1
                else:
                    break
            else:
                c = c[0]
        while len(after_context) < c_window_size:
            after_context.append(after_context[-1])

        after_context = after_context[:c_window_size]
        context_window = before_context + after_context

        return context_window

    def forward(self, context_window, center_token):
        z1 = np.dot(self.W1, context_window) + self.b1
        h = relu(z1)
        z2 = np.dot(self.W2, h) + self.b2
        y_hat = softmax(z2)
        pred = np.argmax(y_hat)
        # loss = -np.sum(center_token * np.log(pred))

        # backpropagation
        dL_dz2 = y_hat - np.array(center_token)
        dL_dW2 = np.outer(dL_dz2, h)
        dL_db2 = dL_dz2
        dL_dh = np.dot(self.W2.T, dL_dz2)
        dL_dz1 = dL_dh * (z1 > 0)
        dL_dW1 = np.outer(dL_dz1, context_window)
        dL_db1 = dL_dz1

        # update weights
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2

        # negative samples
        for i in range(self.neg_samples):
            # randomly pick a word from the vocab
            neg_sample = np.random.choice(list(self.vocab))
            if neg_sample == center_token:
                continue

            neg_loss = -np.log(1-y_hat[self.word2index[neg_sample]])

            dL_dz2_neg = y_hat.copy()
            dL_dz2_neg[self.word2index[neg_sample]] -= 1
            dL_dW2_neg = np.outer(dL_dz2_neg, h)
            dL_db2_neg = dL_dz2_neg
            dL_dh_neg = np.dot(self.W2.T, dL_dz2_neg)
            dL_dz1_neg = dL_dh_neg * (z1 > 0)
            dL_dW1_neg = np.outer(dL_dz1_neg, context_window)
            dL_db1_neg = dL_dz1_neg

            self.W1 -= self.learning_rate * dL_dW1_neg
            self.b1 -= self.learning_rate * dL_db1_neg
            self.W2 -= self.learning_rate * dL_dW2_neg
            self.b2 -= self.learning_rate * dL_db2_neg


    def average_weights(self):
        self.embeddings = (self.W1 + self.W2.T)/2

    def dump_embeddings(self, outfile):
        with open(outfile, "w", encoding="utf-8") as f:
            for e in tqdm(self.index2word, desc="Exporting Embeddings..."):
                f.write(f"{self.index2word[e]} {' '.join(map(str, self.embeddings[:, e]))}\n")


    def embed(self, method=None):
        if method is not None:
            self.method = method

        for graph in tqdm(self.dataset, desc="Embedding...", total=len(self.dataset)):
            # based on method, update embeddings for the batch
            if self.method == "random":
                # any within sentence graph
                pass
            elif self.method == "nearest":
                # get parents, get children, sort
                for e in list(graph.variables()):  # things that could have children
                    c_list = children(graph, e)
                    p_list = parents(graph, e)
                    context_window = self.fill_near(p_list, c_list, self.window_size, self.window_size, graph)
                    self.forward(self.context_to_vector(context_window), self.one_hot(e))

                for e in graph.edges(): # relations
                    relation = e.role
                    p_node = e.source
                    c_node = e.target
                    gp_list = parents(graph, p_node)
                    gc_list = children(graph, c_node)
                    context_window = self.fill_near(gp_list, gc_list, self.window_size-1, self.window_size-1, graph)
                    # put p_node and c_node in the middle
                    context_window.insert(self.window_size-1, p_node)
                    context_window.insert(self.window_size, c_node)
                    self.forward(self.context_to_vector(context_window), self.one_hot(relation))

                for e in graph.attributes():  # no children possible
                    relation = e.role
                    p_node = e.source
                    c_node = e.target
                    gp_list = parents(graph, p_node)
                    gc_list = children(graph, c_node)
                    context_window = self.fill_near(gp_list, gc_list, self.window_size - 1, self.window_size - 1, graph)
                    # put p_node and c_node in the middle
                    context_window.insert(self.window_size - 1, p_node)
                    context_window.insert(self.window_size, c_node)
                    self.forward(self.context_to_vector(context_window), self.one_hot(relation))

            elif self.method == "deepest":
                # get parents sort + repeat, get children sort + repeat until full
                for e in list(graph.variables()):  # could have children
                    context_window = self.fill_deep(e, self.window_size, self.window_size, graph)
                    self.forward(self.context_to_vector(context_window), self.one_hot(e))
                for e in graph.edges():  # relation
                    # fill deep operates from a node, so i will overshoot the window size and then shrink it
                    relation = e.role
                    p_node = e.source
                    c_node = e.target
                    context_window = self.fill_deep(p_node, self.window_size-1, self.window_size+1, graph)
                    context_window[self.window_size-1] = p_node
                    self.forward(self.context_to_vector(context_window), self.one_hot(relation))

                for e in graph.attributes():  # no children possible
                    continue # TODO: implement
                    # print("ATTR:", e)
            elif self.method == "mixed":
                pass
            elif self.method == "duplication":
                # every path is its own update
                for e in list(graph.variables()):
                    pass
                for e in graph.edges():
                    pass
                for e in graph.attributes():
                    pass
            elif self.method == "average":
                # every path is used in one update
                for e in list(graph.variables()):
                    pass
                for e in graph.edges():
                    pass
                for e in graph.attributes():
                    pass
            elif self.method == "variable":
                # large size
                for e in list(graph.variables()):
                    pass
                for e in graph.edges():
                    pass
                for e in graph.attributes():
                    pass
            else:
                raise ValueError("Invalid embedding method")


class Skipgram:
    def __init__(self, dataset, method, window_size, embedding_size, learning_rate, epochs, batch_size, neg_samples, subsampling, seed):
        self.dataset = penman.load(dataset)
        self.method = method
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.subsampling = subsampling
        self.seed = seed

    def embed(self, method=None):
        if method is not None:
            self.method = method

        for graph in self.dataset:
        # based on method, update embeddings for the batch
            if self.method == "generation":
                pass
            elif self.method == "nearest":
                pass
            elif self.method == "crawling":
                pass
            else:
                raise ValueError("Invalid embedding method")


if __name__ == "__main__":
    s = """(f / foolish 
      :polarity (a / amr-unknown)
      :domain (i / i
             :mod (h/ happy))
      :condition (d / do-02
            :ARG0 i
            :ARG1 (t / this)))"""
    g = penman.decode(s)
    test_file_path = "/Users/timothyobiso/Downloads/amr_annotation_3.0/data/amrs/split/training/amr-release-3.0-amrs-training-bolt.txt"
    cbemb = CBOW(test_file_path, "nearest", 5, 300, 0.01, 10, 100, 5, 0.001, 42)
    cbemb.embed()
    cbemb.average_weights()
    cbemb.dump_embeddings("cbow_amr3_5ns_near_TOY.txt")
