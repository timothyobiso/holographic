import torch
import torchhd
from torchhd import structures
import penman
from tqdm import tqdm


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
	all_vocab = [":ROOT", ":rel1", ":rel2", ":rel3", ":rel4", ":rel5", ":rel6", ":rel7"]
	for graph in tqdm(graphs, desc="Getting vocab..."):
		all_vocab += get_graph_vocab(graph)
	
	return list(set(all_vocab))

def prepare_memories(graph, variables, node, rel, hvs, v2i):
	mems = []
	if rel == ":ROOT":
		mems.append((hvs[v2i[rel]], hvs[v2i[variables[node]]]))
	else:
		mems.append((rel, hvs[v2i[variables[node]]]))

	for i, child in enumerate(graph.edges(source=node)):
		mems.append((hvs[v2i[variables[node]]].bind(hvs[v2i[f":rel{i+1}"]]), hvs[v2i[child.role]]))
		mems += prepare_memories(graph, variables, child.target, hvs[v2i[variables[node]]].bind(hvs[v2i[child.role]]), hvs, v2i)
	return mems
	

def store_memories(mems, device, d):
	r1 = structures.HashTable(d, device=device, vsa="HRR")
	for _ in range(5):
		for mem in tqdm(mems, desc="Storing"):
			r1.add(torchhd.HRRTensor(mem[0]), torchhd.HRRTensor(mem[1]))
	return r1


def train(graphs, hvs, vocab, d=1000, device="cpu", n=-1):
	v2i = {v: i for i, v in enumerate(vocab)}
	accs = []
	if n == -1:
		n = len(graphs)
	for graph in graphs[:n]:
		variables = get_variables(graph)
		mems = prepare_memories(graph, variables, graph.top, ":ROOT", hvs, v2i) 
		r1 = store_memories(mems,  device, d)
		
		acc, total = test(r1, mems, hvs, v2i, device)
		accs.append(acc)

	# print average accuracy native python
	print(f"Average Accuracy: {sum(accs)/len(accs*100)}%\n")


def test(model, mems, hvs, v2i, device):
	total = 0
	correct = 0
		
	for mem in tqdm(mems, desc="Testing..."):
		result = model.get(torchhd.HRRTensor(mem[0]))
		
		sim = float(result.cosine_similarity(torchhd.HRRTensor(mem[1])))
		scores = list(map(float, [result.cosine_similarity(torch.tensor(hvs[c], device=device)) for c in range(len(v2i))]))

		if max(scores) == sim:
			correct += 1

		total += 1
		model.remove(torchhd.HRRTensor(mem[0]), torchhd.HRRTensor(mem[1]))

	print(f"Graph Accuracy: {correct/total*100}%\n")
	return correct/total, total


if __name__ == "__main__":
	filepath = "/home/timothyobiso/amr_annotation_3.0/data/amrs/split/training/train.txt" # put amr file in here
	d = 10000
	graphs = load_graphs(filepath)
	vocab = get_vocab(graphs)
	
	hvs = torchhd.HRRTensor.random(len(vocab), d, device="cpu")
	
	train(graphs, hvs, vocab, d=d, device="cpu")
