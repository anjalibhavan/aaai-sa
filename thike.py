import numpy as np
import pickle
with open('theo.pkl','rb') as f:
	edge_list,users_list = pickle.load(f)

print(edge_list[1])
with open('edges_theo.txt','w') as f:
	for edge in edge_list:
		f.write(str(edge[0])+' '+str(edge[1])+ ' '+str(edge[2])+'\n')
