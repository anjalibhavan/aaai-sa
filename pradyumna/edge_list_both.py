import numpy as np 
import pandas as pd
import pickle
import json
from pprint import pprint
from tqdm import tqdm
path = 'C:/users/anjali/environments/acl/data/HanDeSeT.csv'
output_path  = 'C:/users/anjali/environments/acl/adj_list.json'
df = pd.read_csv(path)
names = np.unique(df['name'].values)
name_to_ind = dict(map(lambda x:(x[1], x[0]), enumerate(names)))
with open('C:/users/anjali/environments/acl/data/name_to_ind.pkl', 'wb') as f:
	pickle.dump(name_to_ind, f)


#with open('C:/users/anjali/environments/acl/name_to_ind.pkl','rb') as f:#
	#name_to_ind = pickle.load(f)


groups = {
	'party_basis': {},
	'motion_basis': {}
} 

# Populating the groups
for id, row in df.iterrows():
	party = row['party affiliation']
	title = row['title']
	name  = row['name']
	if not party in groups['party_basis']:
		groups['party_basis'][party] = []
	groups['party_basis'][party].append(name)
	if not title in groups['motion_basis']:
		groups['motion_basis'][title] = []
	groups['motion_basis'][title].append((name, row['manual speech']))

adj_lists = [dict(map(lambda x: (x, {}), df['name'].values)), dict(map(lambda x: (x, {}), df['name'].values)), dict(map(lambda x: (x, {}), df['name'].values))]
total_edges = [0, 0, 0]
tot = 0
s = 0
for title, title_group in groups['motion_basis'].items():
	prop_group = {}
	for a, b in title_group:
		prop_group[a] = b
	prop_group = list(prop_group.items())
	#pprint(prop_group)
	l = len(prop_group)
	s+=l*(l-1)
	curr = 0
	for name1, pol1 in prop_group:
		for name2, pol2 in prop_group:
			if name1 == name2:
				continue
			curr+=1
			tot+=1
			pol = (int(pol1)^int(pol2))
			adj_lists[pol][name1][name2]  = adj_lists[pol][name1].get(name2, 0)+1
		

fnames = ['sup', 'against', 'against_complement']

#setting complement
max_wt = 0
for a, dic in adj_lists[1].items():
	for b, wt in dic.items():
		max_wt = max(max_wt, wt)
ww = set()
avg_wt = np.array([0, 0], dtype = np.float64)
for i in range(2):
	for a, dic in adj_lists[i].items():
		for b, wt in dic.items():
			total_edges[i]+=1
			avg_wt[i]+=wt
			ww.add(wt)

# print('possible', ww)
avg_wt=avg_wt/2
# print(avg_wt)

for a in df['name'].values:
	for b in df['name'].values:
		if a==b:
			continue
		val = -adj_lists[1][a].get(b, 0) +  max_wt
		if val == 0:
			continue
		adj_lists[2][a][b] = val
		total_edges[2]+=1

total_edges = np.array(total_edges)

total_edges = total_edges/2
n = len(np.unique(df['name'].values))
print('Edges', total_edges)
print('Nodes', n)
poss = (n*(n-1))/2
print('Total poss', poss)
print('Density', total_edges/poss)
print('avg_wt', avg_wt/total_edges[:-1])

for t in range(0, 2):
	st = ''
	wts = set()
	for a, dic in tqdm(adj_lists[t].items()):
		for b, wt in dic.items():
			wts.add(wt)
			st = '{}{} {} {}\n'.format(st, name_to_ind[a], name_to_ind[b], wt)
	print(fnames[t], wts)
	with open('C:/users/anjali/environments/acl/data/{}_0_weighted.txt'.format(fnames[t]), 'w') as f:
		f.write(st)
