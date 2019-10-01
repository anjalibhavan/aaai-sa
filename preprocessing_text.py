import sys
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import csv
import pandas as pd
import numpy as np
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin


hansard_stopwords = ({'friend', 'honourable', 'hon', 'gentleman', 'lady', 
					  'give', 'way', 'house', 'amendment', 'beg', 'move', 
					  'member', 'question', 'green', 'white', 'paper', 
					  'bill', 'statement', 'government', 'prime', 'minister', 
					  'opposition', 'party', 'mr', 'rose'})
sklhansard_stopwords = text.ENGLISH_STOP_WORDS.union(hansard_stopwords)

n_gram_min, n_gram_max = 1, 3

#vectorizer = TfidfVectorizer(min_df=5, max_df = 1.0, ngram_range=(n_gram_min,n_gram_max), sublinear_tf=True, use_idf =True, stop_words=sklhansard_stopwords)


data = open('HanDeSeT.csv',encoding='utf-8')
debates = list(csv.reader(data))[1:]


# empty lists for features:
#X_speeches = []
y_speeches = []



for row in debates:
    y_speeches.append(row[11])

    speech_len = 0
    speech_words = " "
    
    for utterance in row[6:11]:
        for sent in sent_tokenize(utterance):
            for token in word_tokenize(sent):
                  
                token = token.lower() 
                token = token.strip() 
                token = token.strip('_') 
                token = token.strip('*') 

                if token in sklhansard_stopwords:
                    continue
                    
                if all(char in set(string.punctuation) for char in token):
                    continue
               
                speech_len += 1
                
        speech_words=speech_words+" "+utterance
        ut.append(speech_words)    
    '''
    speech_len = 0
    speech_words = " "
    
    for sent in sent_tokenize(row[7]):
        for token in word_tokenize(sent):

            token = token.lower() 
            token = token.strip() 
            token = token.strip('_') 
            token = token.strip('*') 

            if token in sklhansard_stopwords:
                continue
                
            if all(char in set(string.punctuation) for char in token):
                continue

            speech_len += 1
            speech_words=speech_words+token+" "
    ut2.append(speech_words)  

    speech_len = 0
    speech_words = " "
    
    for sent in sent_tokenize(row[8]):
        for token in word_tokenize(sent):

            token = token.lower() 
            token = token.strip() 
            token = token.strip('_') 
            token = token.strip('*') 

            if token in sklhansard_stopwords:
                continue
                
            if all(char in set(string.punctuation) for char in token):
                continue

            speech_len += 1
            speech_words=speech_words+token+" "
    ut3.append(speech_words)  

    speech_len = 0
    speech_words = " "
    
    for sent in sent_tokenize(row[9]):
        for token in word_tokenize(sent):

            token = token.lower() 
            token = token.strip() 
            token = token.strip('_') 
            token = token.strip('*') 

            if token in sklhansard_stopwords:
                continue
                
            if all(char in set(string.punctuation) for char in token):
                continue

            speech_len += 1
            speech_words=speech_words+token+" "
    ut4.append(speech_words)  

    speech_len = 0
    speech_words = " "
    
    for sent in sent_tokenize(row[10]):
        for token in word_tokenize(sent):

            token = token.lower() 
            token = token.strip() 
            token = token.strip('_') 
            token = token.strip('*') 

            if token in sklhansard_stopwords:
                continue
                
            if all(char in set(string.punctuation) for char in token):
                continue

            speech_len += 1
            speech_words=speech_words+token+" "
    ut5.append(speech_words)  

debates=np.asarray(debates)
#print(debates.shape)
X_speeches=np.zeros((1251,))

for i in range(6,11):
    print(debates[:,i].shape)
    print(X_speeches.shape)
    X_speeches=np.hstack((X_speeches,debates[:,i]))

#X_speeches=np.asarray(X_speeches)
print(debates[:,6].shape)
print(X_speeches.shape)

X_speeches=np.concatenate((X_speeches,debates[:,6],debates[:,7],debates[:,8],debates[:,9],debates[:,10]),axis=1)
y_speeches=np.asarray(y_speeches)
print(X_speeches.shape)
print(y_speeches.shape)
df=pd.DataFrame(data=(debates[:,6],debates[:,7],debates[:,8],debates[:,9],debates[:,10],y_speeches))

df.to_csv('preprocessed_dataset.csv') 
'''
with open('temp.csv','w') as f:
    f.write(ut)

#print(speeches_train_corpus[1]) 

