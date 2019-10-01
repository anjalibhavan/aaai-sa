import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, metrics
import pprint
import pickle
from operator import itemgetter
import sys
import os

from keras import backend as K
from keras.regularizers import l1, l2

from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Activation, LeakyReLU
from keras.models import Model
from keras.utils import to_categorical
import pandas as pd
from sklearn import decomposition, ensemble
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
from scipy import sparse, stats
import pickle
import imblearn
from sklearn.utils import class_weight
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from IPython.display import clear_output
import sys
import os
import pickle
import json
import argparse
import easydict
import multiprocessing
from gensim import models
from gensim.models import Word2Vec
import gensim.models.keyedvectors as modelloader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.text import one_hot
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import regularizers

seed = 3
np.random.seed(seed)
scaler = StandardScaler()
import warnings
warnings.filterwarnings("ignore")

with open('C:/users/anjali/environments/acl/data/name_to_ind.pkl', 'rb') as f:
	name_to_ind = pickle.load(f)
ind_to_name = dict(map(lambda x:(x[1], x[0]), name_to_ind.items()))
path = 'C:/users/anjali/environments/acl/data/HanDeSeT.csv'
df = pd.read_csv(path)
X = df['name'].values 
y = df['manual speech'].values
all_users = np.unique(df['name'].values)

def word2dict(filename):
	embed ={}
	orig_embed  = {}
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
		node = ind_to_name[int(node)]
		vec = np.array(list(map(float, sp[1:])))
		embed[node] = vec
		orig_embed[node] = vec

	vec = np.zeros((dim, ))
	for user in all_users:
		if user not in embed:
			embed[user] = vec

	return embed, orig_embed

def get_model(embedding_size = 100):
	sup = Input(shape = (embedding_size, ), name = 'similar')
	against = Input(shape = (embedding_size, ), name = 'opposing')
	s1 = Dense(100, activation='relu', kernel_regularizer=l2(0.001))(sup)
	s1 = Dropout(0.2)(s1)
	ag1 = Dense(100, activation='relu', kernel_regularizer=l2(0.001))(against)
	ag1 = Dropout(0.2)(ag1)
	combined = keras.layers.concatenate([s1, ag1])
	# combined = keras.layers.concatenate([sup, against])
	
	x = Dense(100, activation = 'relu', kernel_regularizer=l2(0.001))(combined)
	x = Dropout(0.2)(x)
	x = Dense(100, activation = 'relu', kernel_regularizer=l2(0.001))(x)
	x = Dropout(0.5)(x)
	# x = Dense(25, activation = 'relu')(x)
	# x = Dropout(0.5)(x)

	output = Dense(2, activation = 'sigmoid', kernel_regularizer=l2(1))(x)
	# output = Dense(2, activation = 'sigmoid')(combined)
	model = Model(inputs=[sup, against], outputs=[output])
	pen_model = Model(inputs = [sup, against], outputs = [x])
	return model, pen_model 

path1 = sys.argv[1]
#'C:/users/anjali/environments/acl/data/danmf_embeddings/supres_64-128.vec'
path2 = sys.argv[2]
#'C:/users/anjali/environments/acl/data/danmf_embeddings/agres_64-128.vec'
# path2 = '../Data/node2vecs/against_complement/p0.1_q0.5_dim100_walk10_nwalks15_win10_weighted1.txt'

embed1, o1 = word2dict(path1)
#print(embed1.shape)
embed2, o2 = word2dict(path2)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)
j = 0
best_sp = (0, '', '')
accs = 0
f1s = 0
aps = 0
for train_index, test_index in skf.split(X, y):
	K.clear_session()
	j+=1
	train_x, train_y = (X[train_index], y[train_index])
	test_x, test_y = (X[test_index], y[test_index])
	train_x1 = np.array(list(map(lambda x: embed1[x], train_x)))
	train_x2 = np.array(list(map(lambda x: embed2[x], train_x)))
	test_x1 = np.array(list(map(lambda x: embed1[x], test_x)))
	test_x2 = np.array(list(map(lambda x: embed2[x], test_x)))

	t1newlist = []
	t2newlist = []
	t3newlist = []
	t4newlist = []
	for i in range(0,1125):
		t1newlist.append(train_x1[0].reshape(1,-1))
		t2newlist.append(train_x2[0].reshape(1,-1))

	for i in range(0,126):
		t3newlist.append(test_x1[0].reshape(1,-1))
		t4newlist.append(test_x2[0].reshape(1,-1))

	train_x1_new = np.vstack(t1newlist)
	train_x2_new = np.vstack(t2newlist)
	test_x1_new = np.vstack(t3newlist)
	test_x2_new = np.vstack(t4newlist)

	model, pen_model = get_model(100)

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	#model.summary()
	weights_path = './{}_best.hdf5'.format(j)
	checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
	early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
	callbacks_list = [checkpoint, early]

	class_weights = class_weight.compute_class_weight('balanced',np.unique(train_y),train_y)
	#print(class_weights)
	#print(train_x1_new.shape, train_x2_new.shape, train_y.shape,  test_x1_new.shape, test_x2_new.shape, test_y.shape)
	#print(np.unique(train_y, return_counts = True))
	#print(np.unique(test_y, return_counts = True))
	train_y = to_categorical(train_y, 2)
	test_y = to_categorical(test_y, 2)
	model.fit([train_x1_new, train_x2_new], [train_y], epochs=100, batch_size=64, verbose=1, validation_data=([test_x1_new, test_x2_new], [test_y]),  class_weight=class_weights, callbacks=callbacks_list)
	model.load_weights(weights_path)
	probs = model.predict([test_x1_new, test_x2_new], verbose = 1)
	preds = np.argmax(probs, axis=1)
	# print('Acc:', np.sum(preds == np.argmax(test_y, axis = 1))/preds.shape[0])
	y_t = np.argmax(test_y, axis = 1)
	#print(y_t.shape, preds.shape)
	acc = metrics.accuracy_score(y_t, preds)
	#print('Acc: ', acc)
	accs+=acc
	avg_precision_score = round(metrics.average_precision_score(np.argmax(test_y, axis = 1), probs[:,1], average='weighted'),5)
	f1_score = round(metrics.f1_score(np.argmax(test_y, axis = 1), preds, average='weighted'),5)
	f1s+=f1_score
	aps+=avg_precision_score
	#print (f1_score, " ", avg_precision_score)
	#print (metrics.confusion_matrix(np.argmax(test_y, axis = 1), preds))
	#print (metrics.classification_report(np.argmax(test_y, axis = 1), preds))

	x1 = np.array(list(map(lambda x: embed1[x], X)))
	x2 = np.array(list(map(lambda x: embed2[x], X)))
	#print(x1.shape, x2.shape)
	vecs = pen_model.predict([x1, x2])
	#print(vecs.shape)
	with open('C:/users/anjali/environments/acl/data/node2vec_vecs/node2vec_{}{}.pkl'.format(j,str(sys.argv[3])), 'wb') as f:
		pickle.dump(vecs, f)
	if best_sp[0]<acc:
		best_sp = (acc, weights_path, './vecs/vecs_{}.pkl'.format(j))
	# xx = input('Continue?')

print('Best is ', best_sp)
print('Avg is', accs/10.0, f1s/10.0, aps/10.0)
	# break
	

