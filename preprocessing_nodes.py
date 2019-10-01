import pandas as pd
from sklearn import decomposition, ensemble, model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import numpy as np
from scipy import sparse, stats
import pickle
import imblearn
import _pickle as cPickle
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, StratifiedKFold
from IPython.display import clear_output
import sys
import os
import pickle
import json
import argparse
#import easydict
import multiprocessing
from gensim import models
from gensim.models import Word2Vec
import gensim.models.keyedvectors as modelloader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.utils import class_weight

seed = 42

with open('adj_list.json') as json_data:
    adj_list = json.load(json_data)

users_list=[] #Made this for flag matrix ahead
for name in adj_list:
	users_list.append(name)


data = pd.read_csv('HanDeSeT.csv')
X = data.drop('manual speech', axis=1)
for name in X['name']:
  if name in users_list:
    X.name[X.name==name] = users_list.index(name)
y = data['manual speech']
#print(X.head())r'C:\users\anjali\environments\acl\codes'
#print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
   

embedfile_party=modelloader.KeyedVectors.load_word2vec_format(r'C:\users\anjali\environments\acl\codes\nodetovec\emb\party.emd')
embedfile_motion=modelloader.KeyedVectors.load_word2vec_format(r'C:\users\anjali\environments\acl\codes\nodetovec\emb\motion.emd')
#embedfile_opposing=modelloader.KeyedVectors.load_word2vec_format(r'C:\users\anjali\environments\acl\codes\nodetovec\emb\fourth.emd')

#print(embedfile_party['182'])
#print(type(embedfile_party['182']))
#print(embedfile_party.shape)

res_party_x = np.empty((X.shape[0],128,1))
res_motion_x = np.empty((X.shape[0],128,1))
#train_opposing_x = np.empty((X_train.shape[0],128,1))

#val_party_x = np.empty((X_val.shape[0],128,1))
#val_motion_x = np.empty((X_val.shape[0],128,1))
#val_opposing_x = np.empty((X_val.shape[0],128,1))

#test_party_x = np.empty((X_test.shape[0],128,1))
#test_motion_x = np.empty((X_test.shape[0],128,1))
#test_opposing_x = np.empty((X_test.shape[0],128,1))

zeros_array = np.zeros((128,))
tick_array = np.zeros((607,1))
tick_array_party = np.zeros((607,1))
tick_array_test = np.zeros((607,1))

count=0

for i in X['name']:
  if str(i) in embedfile_party.vocab and tick_array_party[i]!=1:
    for j in range(0,128):
      res_party_x[count,j,0] = embedfile_party[str(i)][j]
    tick_array_party[i]=1
    count+=1
  else:
    for j in range(0,128):
      res_party_x[i,j,0] = 0
    tick_array_party[i]=1
    count+=1

count=0
for i in X['name']:
  if str(i) in embedfile_motion.vocab and tick_array[i]!=1:
    for j in range(0,128):
      res_motion_x[i,j,0] = embedfile_motion[str(i)][j]
    tick_array[i]=1
    count+=1
  else:
    for j in range(0,128):
      res_motion_x[i,j,0]=0
    tick_array[i]=1
    count+=1
'''
count=0
for i in X_train['name']:
  if str(i) in embedfile_opposing.vocab and tick_array_train[i]!=1:
    for j in range(0,128):
      train_opposing_x[count,j,0] = embedfile_opposing[str(i)][j]
    tick_array_train[i]=1
    count+=1
  else:
    for j in range(0,128):
      train_opposing_x[count,j,0]=0
    tick_array_train[i]=1
    count+=1

count=0
for i in X_val['name']:
  if str(i) in embedfile_party.vocab and tick_array_val[i]!=1:
    for j in range(0,128):
      val_party_x[count,j,0] = embedfile_party[str(i)][j]
    tick_array_val[i]=1
    count+=1

  else:
    for j in range(0,128):
      val_party_x[count,j,0] = 0
    tick_array_val[i]=1
    count+=1


count=0
for i in X_val['name']:
  if str(i) in embedfile_motion.vocab and tick_array_val[i]!=1:
    for j in range(0,128):
      val_motion_x[count,j,0] = embedfile_motion[str(i)][j]
    tick_array_val[i]=1
    count+=1

  else:
    for j in range(0,128):
      val_motion_x[count,j,0]=0
    tick_array_val[i]=1
    count+=1


count=0
for i in X_val['name']:
  if str(i) in embedfile_opposing.vocab and tick_array_val[i]!=1:
    for j in range(0,128):
      val_opposing_x[count,j,0] = embedfile_opposing[str(i)][j]
    tick_array_val[i]=1
    count+=1

  else:
    for j in range(0,128):
      val_opposing_x[count,j,0]=0
    tick_array_val[i]=1
    count+=1


count=0
for i in X_test['name']:
  if str(i) in embedfile_party.vocab and tick_array_test[i]!=1:
    for j in range(0,128):
      test_party_x[count,j,0] = embedfile_party[str(i)][j]
    tick_array_test[i]=1
    count+=1

  else:
    for j in range(0,128):
      test_party_x[count,j,0] = 0
    tick_array_test[i]=1
    count+=1

count=0
for i in X_test['name']:
  if str(i) in embedfile_motion.vocab and tick_array_test[i]!=1:
    for j in range(0,128): 
      test_motion_x[count,j,0] = embedfile_motion[str(i)][j]
    tick_array_test[i]=1
    count+=1

  else:
    for j in range(0,128):
      test_motion_x[count,j,0]=0
    tick_array_test[i]=1
    count+=1

count=0
for i in X_test['name']:
  if str(i) in embedfile_opposing.vocab and tick_array_test[i]!=1:
    for j in range(0,128):
      test_opposing_x[count,j,0] = embedfile_opposing[str(i)][j]
    tick_array_test[i]=1
    count+=1

  else:
    for j in range(0,128):
      test_opposing_x[count,j,0]=0
    tick_array_test[i]=1
    count+=1



train_y = y_train
val_y = y_val
test_y = y_test

#res_motion_x = np.vstack((train_motion_x,val_motion_x,test_motion_x))
#res_motion_y = np.vstack((train_y,val_y,test_y))
'''
res_y = y
res_motion_x = np.squeeze(res_motion_x)
res_party_x = np.squeeze(res_party_x)

print(res_motion_x.shape)
f= open('embeddingsnew.pkl','wb')
pickle.dump((res_motion_x, res_party_x, res_y),f)