import numpy as np
from chardet import detect
import pandas as pd
import collections
import pickle
import csv
path='C:/users/anjali/environments/acl/data/handeset.csv'
df = pd.read_csv(path)
names = np.unique(df['name'].values)

with open('C:/users/anjali/environments/acl/theo.pkl','rb') as f:
	edge_list,users_list = pickle.load(f)

name_to_ind = dict(map(lambda x:(x[1], x[0]), enumerate(users_list))) 
#name_to_ind = dict(map(lambda x:(x[1], x[0]), enumerate(names)))
ind_to_name = dict(map(lambda x:(x[1], x[0]), name_to_ind.items()))

def word2dict(filename):
	embed =collections.defaultdict()
	orig_embed  = collections.defaultdict()
	ind = 0
	count = 0
	with open(filename) as f:
		lines = f.read().split('\n')
		no = int(lines[0].split(' ')[0])
		dim = int(lines[0].split(' ')[1])
	for line in lines[1:]:
		sp = line.strip().split(' ')
		node = sp[0]
		if node == '':
			continue
		#node = ind_to_name[int(node)]
		vec = np.array(list(map(float, sp[1:])))
		embed[node] = vec
		orig_embed[node] = vec

	vec = np.zeros((dim, ))
	for user in names:
		if user not in embed:
			embed[user] = vec

	return embed, orig_embed


embed,orig_embed = word2dict('C:/users/anjali/environments/node2vec/emb/finalres.emb')

res = []
data = open('C:/users/anjali/environments/acl/data/handeset.csv',encoding='utf-8')
debates = list(csv.reader(data))[1:]
for row in debates:
	res.append(embed[row[14]])

res = np.array(res)
with open('theoemb.pkl','wb') as f:
	pickle.dump(res,f)