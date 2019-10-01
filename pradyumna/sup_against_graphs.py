import numpy as np 
import pandas as pd
import pickle
import json
path = '../Data/HanDeSeT1.csv'
output_path  = '../Data/adj_list.json'
df = pd.read_csv(path)
df1 = df.drop_duplicates()
print(len(df), len(df1))
print('Your grace')
# x = df.loc[df['motion'] == 'That this House regrets the continuing lack of balance in the UK economy and the UK Government’s over-reliance on unsustainable consumer debt to support economic growth; notes in particular the UK’s poor export performance, which resulted in a trade deficit in goods of £123 billion in 2014; further notes the UK’s continuing poor productivity record and the lack of a credible long-term plan to improve it; and is deeply concerned by the UK Government’s change to Innovate UK funding of innovation from grants to loans, which this House believes will result in a deterioration of private sector research and development.']
# x = df.loc[df['name'] == 'David Mowat']
# print(x.columns)
# # print(x[['id', 'manual motion', 'govt/opp motion','motion party affiliation', 'manual speech', 'vote speech', 'party affiliation', 'name']])
# print(x[['name', 'utt1', 'manual speech', 'vote speech']].values)

adj_lists = [{}, {}]
adj_list = dict(map(lambda x: (x, {}), df['name'].values))
groups = {
	'party_basis': {},
	'motion_basis': {}
} 


# Populating the groups
for id, row in df.iterrows():
	party = row['party affiliation']
	title = row['motion']
	name  = row['name']
	if not party in groups['party_basis']:
		groups['party_basis'][party] = []
	groups['party_basis'][party].append(name)
	if not title in groups['motion_basis']:
		groups['motion_basis'][title] = []
	groups['motion_basis'][title].append((name, row['manual speech']))

pos, neg, s = 0, 0, 0
for title, title_group in groups['motion_basis'].items():
	l = len(title_group)
	# print('Length', ' - len ->', l)
	s+=l*(l-1)
	curr = 0
	for name1, pol1 in title_group:
		for name2, pol2 in title_group:
			if name1 == name2:
				continue
			curr+=1
			if title not in adj_list[name1]:
				adj_list[name1][title] = {}
			# if title not in adj_list[name2]:
			# 	adj_list[name2][title] = {}
			pol = 1-(int(pol1)^int(pol2))
			adj_list[name1][title][name2]  = 2*pol - 1
			# adj_list[name2][title][name1]  = 2*pol - 1
			if 2*pol-1 == -1:
				neg+=1
			else :
				pos+=1
	if curr!=l*(l-1):
		mm = {}
		for a,b in title_group:
			if a not in mm:
				mm[a] = list()
			mm[a].append(b)

		print(title)
		print('Hello')
		print(curr, l, l*(l-1))
		print('*'*20)
		print(title_group)
		print('*'*20)
		print(sorted(mm.items(), reverse  = True))
		print('*'*20)
		print(np.unique(title_group, return_counts = True))


		for a,b in mm.items():
			if len(b)>1 and a=='David Mowat':

				x = df.loc[(df['name'] == a )& (df['motion'] == title)]
				print(x.values)
				a = input('Lets go ?')

print('Positives->', pos, '| Negatives->', neg, 'Tot: ', s, pos+neg)
# with open(output_path, 'w') as f:
# 	json.dump(adj_list, f)

