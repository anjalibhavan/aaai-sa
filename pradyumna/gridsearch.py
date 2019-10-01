from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from gridearch_helper import EstimatorSelectionHelper
import os 
import csv
import pickle

number_of_walks = [5,10,15,20]
walk_length = [10,20,30,40,50,60,70,80,90,100]

# python src/main.py --edge-path C:/Users/ANJALI/environments/acl/data/sup_
#ments/acl/attentionwalk_embeddings/supres

for idx1, i in enumerate(number_of_walks):
	for idx2, j in enumerate(walk_length):
		os.system('deepwalk --input example_graphs/against_0.txt --representation-size 128 --number-walks' + ' ' + str(i) + ' ' + '--walk-length' + ' ' + str(j) + ' ' + ' --output C:/Users/ANJALI/environments/acl/data/deepwalk_embeddings/agres_{}{}'.format(str(idx1),str(idx2)))


pathsup = 'C:/Users/ANJALI/environments/acl/data/deepwalk_embeddings/supres_'
pathag = 'C:/Users/ANJALI/environments/acl/data/deepwalk_embeddings/agres_'

for i in range(0,4):
	for j in range(0,10):
		filesup = pathsup+str(i)+str(j)
		fileag = pathag+str(i)+str(j)
		os.system('python combine_embeddings.py' + ' '+ filesup + ' '+ fileag + ' ' + str(i) + ' ' + str(j))


'''
data = open('C:/users/anjali/environments/acl/data/HanDeSeT.csv',encoding='utf-8')
debates = list(csv.reader(data))[1:]
labels = []
for row in debates:
	labels.append(row[11])

rootdir = 'C:/Users/ANJALI/environments/acl/data/deepwalk_vecs'

#models = [MLPClassifier(), SVC(), RandomForestClassifier(), GradientBoostingClassifier(), DecisionTreeClassifier(), LogisticRegression()]
#params1 = 


models1 = {
    'RandomForestClassifier': RandomForestClassifier(),
    'MLPClassifier': MLPClassifier(verbose = 0),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = {
    'RandomForestClassifier': { 'n_estimators': [100,150,200,300] },
    'MLPClassifier':  { 'alpha': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], 'max_iter': [500] },
    'GradientBoostingClassifier': { 'n_estimators': [100,150,200,300], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [0.1,1, 10]},
        {'kernel': ['rbf'], 'C': [0.1,1, 10], 'gamma': [0.001, 0.0001]},
    ]
}

helper1 = EstimatorSelectionHelper(models1, params1)

for file in os.listdir(rootdir):
	full_name = rootdir+'/'+file
	with open(full_name,'rb') as f:
		X = pickle.load(f)
	helper1.fit(X, labels,scoring='accuracy')
	print(helper1.score_summary(sort_by='max_score'))
	#print(helper1.score_summary(sort_by='max_score')[0:1])
	with open('gridresultsfinal.txt','a') as f:
		f.write(str(helper1.score_summary(sort_by='max_score').iloc[0,0:4]))
		f.write('\n')

'''