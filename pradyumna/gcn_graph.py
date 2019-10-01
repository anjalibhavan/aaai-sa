import numpy as np
import pandas as pd
import collections
with open('C:/users/anjali/environments/acl/data/adj_list.json') as json_data:
    adj_list = json.load(json_data)

users_list=[] # Made this for flag matrix ahead
for name in adj_list:
	users_list.append(name)

edge_list = []

# Party connection
flag_matrix = np.zeros((607,607))   # Since there are 607 unique users
for name1 in adj_list:		 
	key = 'party'
	if key in adj_list[name1]:
		for name2 in adj_list[name1]['party']:   # Going through all users in the same party as name1.
			if flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
				temp_list = [name1, name2, 0]
				flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1  # Marking the two users so their weights 
				flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1  # aren't altered ever again.
				edge_list.append(temp_list)


# User speech connection
speech_index = np.arange(0,1251)
speeches = []
for i in speech_index:
	speeches.append('S'+str(i))

data = pd.read_csv('C:/users/anjali/environments/acl/data/handeset.csv')

userdict = collections.defaultdict()
for i in speeches:
	userdict[i]=users_list.index(data.iloc[i,14])

for i in speeches:
	edge_list.append([userdict[i],i,0])

# Speech speech connection
