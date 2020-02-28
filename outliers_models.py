#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:05:39 2017

@author: prudhvi
"""

import simplejson as json 
import os 
from nltk.corpus import RegexpTokenizer as regextoken
import string 
import csv 
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,classification_report 
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize  
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
import collections, itertools


os.chdir("/home/prudhvi/Desktop/data science/Icecream/Nlp") 


""" extracting the data""" 

data = []
with open('text_with_machine.json') as f:
    for line in f:
        data.append(json.loads(line))

tokenizer = regextoken(r'\w+')
stop = stopwords.words('english') + ['b','B','BR','s','li','Li','w','ul']


""" extracting the title and short description """ 

des = []
for i in range(len(data)):
    tokens = tokenizer.tokenize(data[i]['title']+data[i]['short_description'])
    tokens = [token.lower() for token in tokens if token not in stop]
    des.append(((tokens))) 
    

ldes = []
for i in range(len(data)):
    tokens = tokenizer.tokenize(data[i]['title']+data[i]['short_description'])
    tokens = [token.lower() for token in tokens if token not in stop]
    ldes.append((str(tokens))) 
    
    
""" count vectoriser """

def count_model(data,mfeatures,target):
    
    count = CountVectorizer(analyzer='word',max_features = mfeatures)
    final_counts = count.fit_transform(data) 
    frs = count.get_feature_names()
    counts_dense = final_counts.todense()
    df = pd.DataFrame(counts_dense)
    words_count= df.sum(axis=1)
    sorted_counts = sorted(words_count)
    length = sorted_counts[int(0.30*len(words_count))]
    out = [i for i in range(len(words_count)) if words_count[i] < length]
    pred_out = [0]*len(words_count)
    for i in out:
        pred_out[i] = 1
    
    accuracy = accuracy_score(target,pred_out)
    recall = recall_score(target,pred_out)
    precision = precision_score(target,pred_out)
    conf = confusion_matrix(target,pred_out)
   ### features = count.get_feature_names()
    
    print 'precision:' , precision*100
    print 'recall : ' , recall*100
    print 'accuracy :' , accuracy*100
    print 'confusion matrix :' ,conf
    
    print classification_report(target,pred_out,target_names = ['Inliers','outliers'])
    
    return pred_out,frs


k,frs = count_model(title,20,act_out)
p = count_model(des,5000,act_out)


    
""" tfidf matrix"""

vect = TfidfVectorizer(ngram_range = (1,1),analyzer='word',max_features = 1000,use_idf = True) 

tfidf = vect.fit_transform(ldes) 
feature_names = vect.get_feature_names() 
dense = tfidf.todense() 
dense.shape 
df = pd.DataFrame(dense)

new_att  = [0]*56398 
           
for i in range(len(des)):
    if len(list(set(final).intersection(des[i]))) == 0:
        new_att[i] = 1

df['new_att'] = new_att 
import random
import numpy as np 
random.seed(800)
df['target'] = act_out 



df = df.iloc[np.random.permutation(len(df))]
train = df[0:int(len(df)*0.7)]
test = df[int(len(df)*0.7):]




""" Isolation forest code """ 

iforest = IsolationForest(max_samples = 40000, contamination = 0.3)

iforest.fit(train.ix[:,:-1])
#if_pre = iforest.predict(train.ix[:,:-1])
#if_pre = [0 if x == 1 else 1 for x in if_pre ]
if_test_pre = iforest.predict(test.ix[:,:-1])
if_test_pre = [0 if x == 1 else 1 for x in if_test_pre ]


if_test_conf = confusion_matrix(test.ix[:,-1],if_test_pre)
if_test_accuracy = accuracy_score(test.ix[:,-1],if_test_pre)
recall = recall_score(test.ix[:,-1],if_test_pre)
precision = precision_score(test.ix[:,-1],if_test_pre)

print 'precision:' , precision*100
print 'recall : ' , recall*100
print 'accuracy :' , if_test_accuracy*100
print 'confusion matrix :' , if_test_conf

print classification_report(test.ix[:,-1],if_test_pre,target_names = ['Inliers','outliers'])





"""word2vec""" 

from gensim.models import Word2Vec

w_v = Word2Vec(des,min_count = 2,size = 10)



def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0 :
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1 
    return reviewFeatureVecs

trainVecs = getAvgFeatureVecs(des,w_v,10) 


""" indetifying similarity between word pairs generated from pos tagging"""



"""creating pairwise similarity scores/matrix """ 



sim_matrix = []
for i in range(len(key_words)):
    sim_array = [] 
    for j in range(len(key_words)):
        sim_array.append(1/(Word2Vec.similarity(w_v,key_words[i],key_words[j])+0.01))
    
    sim_matrix.append(sim_array)
    
        
        """w2v_labels = []
        if Word2Vec.similarity(w_v,key_words[i],key_words[j]) > 0.8 and i != j :
            print key_words[i], key_words[j] , Word2Vec.similarity(w_v,key_words[i],key_words[j])
            w2v_labels.append(key_words[i])
            w2v_labels.append(key_words[j])
            
w2v_labels = list(set(w2v_labels)) """


""" Models based on word2vec """ 

import numpy as np 
sim_matrix = np.matrix(sim_matrix)

""" DB scan """ 

from sklearn.cluster import DBSCAN
model  = DBSCAN()
model.fit(sim_matrix)
kk = DBSCAN(metric = 'precomputed').fit_predict(sim_matrix)
              
                 
""" K Means clustering """ 

from sklearn.cluster import KMeans 
model = KMeans(n_clusters = 5 , init = 'k-means++', )
kk = model.fit_predict(sim_matrix)
clstr = {}
for i in range(len(kk)):
    clstr.update(map(key_words[i],kk[i]))



""" Spectral clustering """ 

from sklearn.cluster import SpectralClustering 
sc = SpectralClustering.fit_predict(sim_matrix)





""" random forest model """ 
# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100 )
import pandas as pd
df = pd.DataFrame(trainVecs)
df['target'] = act_out 

df = df.iloc[np.random.permutation(len(df))]
train = df[0:int(len(df)*0.7)]
test = df[int(len(df)*0.7):]

forest = forest.fit(train.ix[:,:-1],train.ix[:,-1])
# Test & extract results 
result = forest.predict(test.ix[:,:-1])
print classification_report(test.ix[:,-1],result,target_names = ['Inliers','outliers'])

rforest_conf = confusion_matrix(test.ix[:,-1],result)
