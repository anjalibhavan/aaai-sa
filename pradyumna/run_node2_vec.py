import sys
import os
import easydict
#sys.path.insert(0, '../node2vec(grover)/src/')
#import main 
#print(main)
import time


# win_values = [5, 10]

pq_values = [1, 0.1, 10]
dim_values = [100]
walk_values = [10, 15]
nwalks_values = [5]
win_values = [10]
weighted_values = ['unweighted', 'weighted']

rootdir = 'C:/users/anjali/environments/acl/data/node2vec_embeddings/node2vecs/'
supdir = rootdir+'sup'
agdir = rootdir+'against'

for idx1,supname in enumerate(os.listdir(supdir)):
	#for idx2, agname in os.listdir(agdir):
	if supname in os.listdir(agdir):
		ressup = supdir+'/'+supname
		resag = agdir+'/'+supname
		os.system('python combine_embeddings.py ' + ' ' +ressup+' '+resag+' '+str(idx1))








'''
for p in pq_values:
	for q in pq_values:
		for dim in dim_values:
			for walk in walk_values:
				for nwalks in nwalks_values:
					for win in win_values:
						start_time = time.time()
						ind = ind+1
						print(ind)
						i_filename = '../Data/edge_lists/{}_0.txt'.format(type_name)
						o_filename = '../Data/node2vecs/{}/p{}_q{}_dim{}_walk{}_nwalks{}_win{}_weighted{}.txt'.format(str(type_name), str(p), str(q), str(dim), str(walk), str(nwalks), str(win), str(int(weighted)))

						if not os.path.exists(o_filename):
							try:
								main.main(args)
							except Exception as e:
								print(e)
								print("--- %s seconds ---" % (time.time() - start_time))
								continue
						print("--- %s seconds ---" % (time.time() - start_time))
'''

