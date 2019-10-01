import json
import numpy as np

with open('adj_list.json') as json_data:
    adj_list = json.load(json_data)

users_list=[] # Made this for flag matrix ahead
for name in adj_list:
	users_list.append(name)


def gen_edge_list(type, adj_list, include_weight = True): 
	edge_list = []
	if type=='party':                		# For people in the same party. 
		flag_matrix = np.zeros((607,607))   # Since there are 607 unique users
		for name1 in adj_list:		 
			key = 'party'
			if key in adj_list[name1]:
				for name2 in adj_list[name1]['party']:   # Going through all users in the same party as name1.
					if flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
						temp_list = [name1, name2]
						flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1  # Marking the two users so their weights 
						flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1  # aren't altered ever again.
						edge_list.append(temp_list)
		with open('edgelistparty.txt', 'w') as text_file:
			for row in edge_list:
				text_file.write(str(row[0])+', '+str(row[1])+'\n')					
						

	if type=='motion':               # For across-motion polarities summation  
		count=0
		flag_matrix = np.zeros((607,607))
		for name1 in adj_list: 		 # Iterating over each username
			for key in adj_list[name1]:  
				if key != 'party':	 # Since motion-based
					for name2 in adj_list[name1][key]:  # Iterating through each user in outer user's motion which is 'key'.
						sumwt = 0
						for key1 in adj_list[name1]: # Now going through all the motions one by one to sum polarities.
							if key1 != 'party':	
								if name2 in adj_list[name1][key1] and flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
									
									#Summing polarities only if the two users haven't been marked before
									# which is checked by going into flag_matrix.
									
									if adj_list[name1][key1][name2]==1:     # If users name1 and name2 agree on motion 'key'
										sumwt+=2*0.1
									elif adj_list[name1][key1][name2]==-1:  # If users name1 and name2 disagree on motion 'key'
										sumwt+=1*0.1
													
						if flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:		
							if sumwt!=0:
								count+=1
								temp_list = [name1, name2]
								flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1  #Marking the two users so their weights aren't altered ever again.
								flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1
								edge_list.append(temp_list)	
					break
		print('number of nonzero weight edges',count)
		with open('edgelist2.txt', 'w') as text_file:
			for row in edge_list:
				text_file.write(str(row[0])+', '+str(row[1])+'\n')

					# Rest of the cases ahead follow the same nomenclature and iteration patterns.

	if type == 'smotion':             # For across-motion polarities summation only for people with same polarities on common motions.
		if include_weight == True:	  # Weighted case	
			flag_matrix = np.zeros((607,607))
			for name1 in adj_list:
				for key in adj_list[name1]:
					if key!='party':
						for name2 in adj_list[name1][key]:
							sumwt = 0
							for key1 in adj_list[name1]:
								if key1 != 'party':
									if (name2 in adj_list[name1][key1]) and adj_list[name1][key1][name2]!=-1 and flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
										sumwt += adj_list[name1][key1][name2]

							temp_list = [users_list.index(name1), users_list.index(name2), sumwt]
							flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1
							flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1
							if sumwt!=0:
								edge_list.append(temp_list)	
						break

		else:						  # Unweighted case
			flag_matrix = np.zeros((607,607))
			for name1 in adj_list:
				for key in adj_list[name1]:
					if key != 'party':
						for name2 in adj_list[name1][key]:
							for key1 in adj_list[name1]:
								if key1!='party':
									#print(name1)
									if name2 in adj_list[name1][key1] and adj_list[name1][key1][name2]!=-1 and flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
										temp_list = [name1, name2] # If there exists atleast one motion where users name1 and name2 have identical sentiments, create edge
										flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1
										flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1
										edge_list.append(temp_list)	
										break
		with open('edgelistS.txt', 'w') as text_file:
			for row in edge_list:
				text_file.write(str(row[0])+', '+str(row[1])+'\n')


	if type == 'dmotion':             # For across-motion polarities summation only for people with different polarities on common motions.
		if include_weight == True:    # Weighted case
			flag_matrix = np.zeros((607,607))
			for name1 in adj_list:
				for key in adj_list[name1]:
					if key != 'party':
						for name2 in adj_list[name1][key]:
							sumwt = 0
							for key1 in adj_list[name1]:
								if key1!='party':
									if (name2 in adj_list[name1][key1]) and adj_list[name1][key1][name2]!=1 and flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
										sumwt += adj_list[name1][key1][name2]

							temp_list = [users_list.index(name1), users_list.index(name2), np.abs(sumwt)]
							flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1
							flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1
							if sumwt!=0:
								edge_list.append(temp_list)	
						break

		else:						# Unweighted case
			flag_matrix = np.zeros((607,607))
			for name1 in adj_list:
				for key in adj_list[name1]:
					if key != 'party':
						for name2 in adj_list[name1][key]:
							for key1 in adj_list[name1]:
								if key1!='party':
									if name2 in adj_list[name1][key1] and adj_list[name1][key1][name2]!=1 and flag_matrix[users_list.index(name1)][users_list.index(name2)]!=1 and flag_matrix[users_list.index(name2)][users_list.index(name1)]!=1:
										temp_list = [name1, name2]
										flag_matrix[users_list.index(name1)][users_list.index(name2)] = 1
										flag_matrix[users_list.index(name2)][users_list.index(name1)] = 1
										edge_list.append(temp_list)	
										break
		with open('edgelistD.txt', 'w') as text_file:
			for row in edge_list:
				text_file.write(str(row[0])+', '+str(row[1])+'\n')			

print('enter type: party, motion, smotion or dmotion')
p=input()
gen_edge_list(p,adj_list,False)