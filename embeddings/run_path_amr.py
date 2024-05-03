import torch
import torchhd
from torchhd import structures
import penman
from penman.models.amr import model
import numpy as np
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
	all_vocab = [":ROOT", ":rel1", ":rel2", ":rel3", ":rel4", ":rel5", ":rel6", ":rel7"]
	for graph in tqdm(graphs, desc="Getting vocab..."):
		all_vocab += get_graph_vocab(graph)
	
	return list(set(all_vocab))

def prepare_memories(graph, variables, node, rel, hvs, v2i):
	mems = []
	if rel == ":ROOT":
		mems.append((hvs[v2i[rel]], hvs[v2i[variables[node]]]))
		#print(rel, node)
	else:
		mems.append((rel, hvs[v2i[variables[node]]]))
		#print("into", node)

	for i, child in enumerate(graph.edges(source=node)):
		mems.append((hvs[v2i[variables[node]]].bind(hvs[v2i[f":rel{i+1}"]]), hvs[v2i[child.role]]))
		#print("APPENDED", node, f"rel{i}", child.role)
		mems += prepare_memories(graph, variables, child.target, hvs[v2i[variables[node]]].bind(hvs[v2i[child.role]]), hvs, v2i)
	return mems
	

def store_memories(mems, device):
	r1 = structures.HashTable(d, device=device, vsa="HRR")
	#print(len(mems))
	#print(len(mems[0]))
	#print(mems[0])
	#print(mems[1])
	for _ in range(5):
		for mem in tqdm(mems, desc="Storing"):
			r1.add(torchhd.HRRTensor(mem[0]), torchhd.HRRTensor(mem[1]))
	return r1


def train(graphs, hvs, vocab, d=1000, epochs=1, device="cpu"):
	v2i = {v: i for i, v in enumerate(vocab)}
	accs = []
	for graph in graphs:
		variables = get_variables(graph)
		mems = prepare_memories(graph, variables, graph.top, ":ROOT", hvs, v2i) 
		r1 = store_memories(mems,  device)
		
		test(r1, mems, variables, hvs, v2i, device)


def test(model, mems, variables, hvs, v2i, device):
	total = 0
	correct = 0
		
	for mem in tqdm(mems, desc="Testing..."):

		# print("correct word:", child)
		
		result = model.get(torchhd.HRRTensor(mem[0]))
		
		sim = float(result.cosine_similarity(torchhd.HRRTensor(mem[1])))
		scores = list(map(float, [result.cosine_similarity(torch.tensor(hvs[c], device=device)) for c in range(len(v2i))]))
		
		# print("SIM:", sim)
		# print("SCORES:", scores)
		
		# print(sim, max(scores))
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
	
	train(graphs, hvs, vocab, d=d, epochs=1)

	test(model, filepath, hvs)



	s = """(c / choose-01
          :ARG1 (c2 / concept
                :quant 100
                :mod (i / innovate-01)))"""
	s = """(c / choose-01
          :ARG1 (c2 / concept 
                :quant 100
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

