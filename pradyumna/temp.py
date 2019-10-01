import numpy as np
import csv
import json
from chardet import detect

import os
import pickle
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile


path = 'C:/users/anjali/environments/acl'
#with open('C:/users/anjali/environments/acl/data/name_to_ind.pkl','rb') as f:
#	name_to_ind = pickle.load(f)

data = open('C:/users/anjali/environments/acl/data/HanDeSeT.csv',encoding='utf-8')
debates = list(csv.reader(data))[1:]
embeddings = []
#names = []
df = pd.read_csv('C:/users/anjali/environments/acl/data/HanDeSeT.csv')
names = np.unique(df['name'].values)
name_to_ind = dict(map(lambda x:(x[1], x[0]), enumerate(names)))
namess = []


vectors = open('C:/Users/ANJALI/environments/acl/data/danmf_embeddings/agres_64-128.csv')
vecs = list(csv.reader(vectors))[1:]
print(type(vecs[10]))

with open('C:/users/anjali/environments/acl/data/danmf_embeddings/agres_64-128.vec','w') as f:
	count = 0
	#f.write('599 128'+'\n')
	for vec in vecs:
		newlist = []
		newlist.append(int(float(vec[0])))
		for i in vec[1:]:
			newlist.append(float(i))
		for i in newlist:
			f.write(str(i)+' ')
		f.write('\n')


'''


fname = get_tmpfile("C:/users/anjali/environments/acl/data/danmf_embeddings/danmf_supres.vec")
word_vectors = KeyedVectors.load_word2vec_format(fname)
for row in debates:
	namess.append(row[14])
	embeddings.append(word_vectors[str(name_to_ind[row[14]])])
	
print(len(embeddings))

with open('C:/users/anjali/environments/acl/data/danmf_embeddings/danmf_sup.vec','w') as f:
	f.write(str(embeddings))
#print(word_vectors[str(name_to_ind[names[0]])])
'''
'''
es1 = []
es2 = []
ea1 = []
ea2 = []
with open(path+'/data/sup_0.txt','r') as f:
	for line in f:
		es1.append(line.rstrip().split()[0])
		es2.append(line.rstrip().split()[1])

with open(path+'/data/against_0.txt','r') as f:
	for line in f:
		ea1.append(line.rstrip().split()[0])
		ea2.append(line.rstrip().split()[1])


sup = pd.DataFrame()
sup['es1'] = es1
sup['es2'] = es2
ag = pd.DataFrame()
ag['ea1'] = ea1
ag['ea2'] = ea2

sup.to_csv(path+'/data/sup_0.csv')
ag.to_csv(path+'/data/ag_0.csv')
'''



