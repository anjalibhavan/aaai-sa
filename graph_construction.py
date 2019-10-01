import numpy as np 
import pandas as pd
import pickle
import json
path = 'C:/users/anjali/environments/acl/data/HanDeSeT.csv'
output_path  = 'C:/users/anjali/environments/acl/data/adj_list_ok.json'
df = pd.read_csv(path)

adj_list = {}
# Format 
# {
# 	u1: {
# 	'pol_party(graph_type): {{'neighbour', 'weight'}, {'neighbour', 'weight'}, {'neighbour', 'weight'},{'neighbour', 'weight'}},
#   'motion1(graph_type): {{'neighbour', 'weight'}, {'neighbour', 'weight'}, {'neighbour', 'weight'},{'neighbour', 'weight'}}
# 	}
# }

groups = {
	'party_basis': {},
	'motion_basis': {}
} 
# groups = {
# 	'party_basis': {
# 		'party1': ['u1', 'u2', 'u3'], 
# 		'party2': ['u4', 'u5', 'u6']
# 	}, 
# 	'motion_basis': {
# 		'title1': ['u1', 'u2', 'u3'], 
# 		'title2': ['u4', 'u5', 'u6']
# 	}, 	
# }

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


adj_list = dict(map(lambda x: (x, {}), df['name'].values))
#Constructing the graph

# For Party
for party, party_group in groups['party_basis'].items():
	for name1 in party_group:
		for name2 in party_group:
			if name1 == name2:
				continue
			if 'party' not in adj_list[name1]:
				adj_list[name1]['party'] = {}
			if 'party' not in adj_list[name2]:
				adj_list[name2]['party'] = {}
			adj_list[name1]['party'][name2] = 1
			adj_list[name2]['party'][name1] = 1

# For Motions
for title, title_group in groups['motion_basis'].items():
	for name1, pol1 in title_group:
		for name2, pol2 in title_group:
			if name1 == name2:
				continue
			if title not in adj_list[name1]:
				adj_list[name1][title] = {}
			if title not in adj_list[name2]:
				adj_list[name2][title] = {}
			pol = 1-int(pol1)^int(pol2)
			adj_list[name1][title][name2]  = 2*pol - 1
			adj_list[name2][title][name1]  = 2*pol - 1

with open(output_path, 'w') as f:
	json.dump(adj_list, f)


				
				