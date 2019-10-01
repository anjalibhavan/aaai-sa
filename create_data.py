import numpy as np
import pandas as pd
import csv
from pprint import pprint
import networkx as nx
from chardet import detect
import os
import pickle as pkl

def create_data():

	data = open('handeset.csv',encoding='utf-8')
	debates = list(csv.reader(data))[1:]

	edges = []

	with open('edgelistparty.txt','r') as f:
		count = 0
		for line in f:
			edges.append(tuple(line.rstrip().split(', ')))

	G = nx.Graph(edges)
	A = nx.adjacency_matrix(G)
	features = np.identity(A.shape[0])

	nodess = list(G.nodes())

	labels = []
	y = np.zeros((len(nodess),2))
	for row in debates:
		labels.append(int(row[11]))
	labels = np.array(labels)
	y[np.arange(0,1251),labels] = 1

	return y,A,features

