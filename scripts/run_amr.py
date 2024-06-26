import torch
import torchhd
from torchhd import structures
import penman
from tqdm import tqdm
import numpy as np

def load_graphs(filepath):
	return penman.load(filepath)

def get_variables(graph):
	return {triple[0]: triple[2] for triple in graph.triples if triple[1] == ":instance"}

def get_graph_vocab(graph):
	return list(set([triple[2] for triple in graph.triples if triple[1] == ":instance"] +
		[triple[1] for triple in graph.triples if triple[1] != ":instance"] +
		[attr.role for attr in graph.attributes()] + 
		[attr.source for attr in graph.attributes()] +
		[attr.target for attr in graph.attributes()]
		))

def get_vocab(graphs):
	all_vocab = []
	for graph in tqdm(graphs, desc="Getting vocab..."):
		all_vocab += get_graph_vocab(graph)
	
	return list(set(all_vocab))

def train(graphs, hvs, vocab, d=1000, device="cpu", n=-1) -> structures.HashTable:
	v2i = {v: i for i, v in enumerate(vocab)}
	accs = []
	if n == -1:
		n = len(graphs)
	for graph in tqdm(graphs[:n], desc="Training"):
		r1 = structures.HashTable(d, device=device)
		variables = get_variables(graph)
		for triple in graph.triples:
			if triple[1] == ":instance":
				continue

			if triple[0] in variables:
				parent = variables[triple[0]]
			else:
				parent = triple[0]

			if triple[2] in variables:
				child = variables[triple[2]]
			else:
				child = triple[2]

			key = hvs[v2i[parent]].bind(torch.tensor(hvs[v2i[triple[1]]])).to(device)
			r1.add(torch.tensor(key), torch.tensor(hvs[v2i[child]]).to(device))
		accs.append(test(r1, graph, hvs, vocab, device))

	# print average accuracy
	accs = np.array(accs)
	print(f"Average accuracy: {np.mean(accs[:,0])*100:.3f}%\n")

def test(model, graph, hvs, vocab, device):
	v2i = {v: i for i, v in enumerate(vocab)}
	total = 0
	correct = 0
		
	for triple in tqdm(graph.triples, desc="Testing..."):

		if triple[1] == ":instance":
			continue
		variables = get_variables(graph)	
		if triple[0] in variables:
			parent = variables[triple[0]]
		else:
			parent = triple[0]
		
		if triple[2] in variables:
			child = variables[triple[2]]
		else:
			child = triple[2]
		
		key = hvs[v2i[parent]].bind(torch.tensor(hvs[v2i[triple[1]]]))
		
		result = model.get(torchhd.tensors.map.MAPTensor(key).to(device))
		
		sim = float(result.cosine_similarity(torch.tensor(hvs[v2i[child]],device=device)))
		scores = [float(result.cosine_similarity(torch.tensor(hvs[v2i[c]],device=device))) for c in vocab]
		
		if max(scores) == sim:
			correct += 1
		total += 1

	print(f"\nGraph Accuracy: {correct/total*100}.3f%\n")
	return correct/total, total

if __name__ == "__main__":
	filepath = "/home/timothyobiso/amr_annotation_3.0/data/amrs/split/training/train.txt" # put amr file in here
	d = 500
	graphs = load_graphs(filepath)
	vocab = get_vocab(graphs)
	
	hvs = torchhd.HRRTensor.random(len(vocab), d, device="cuda:0")
	model = train(graphs, hvs, vocab, d=d, device="cuda:0")


	s = """(c / choose-01
          :ARG1 (c2 / concept 
                :quant 100
                :ARG1-of (i / innovate-01))
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

