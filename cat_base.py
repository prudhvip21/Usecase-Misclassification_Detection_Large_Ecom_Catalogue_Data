# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 18:12:20 2017

@author: Prudhvi
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
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score 
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize  
from sklearn.cluster import KMeans

os.chdir("C:\Users\Prudhvi\Desktop\data science\Icecream\Nlp") 


data = []

with open('text_file.json') as f:
    for line in f:
        data.append(json.loads(line))

tokenizer = regextoken(r'\w+')
stop = stopwords.words('english') + ['b','B','BR','s','li','Li','w','ul']


### regex = re.compile('([A-Z]|[a-z])\w+')

### number is not removed


ldes = []


for i in range(len(data)):
    tokens = tokenizer.tokenize(data[i]['title'])
    tokens = [token for token in tokens if token not in stop]
    ldes.append(str(tokens))
    
    
""" count vectoriser """


count = CountVectorizer(analyzer='word',max_features = 500)

final_counts = count.fit_transform(ldes) 

counts_dense = final_counts.todense()

features = count.get_feature_names()
counts_dense.shape()
    
""" tfidf matrix"""

vect = TfidfVectorizer(ngram_range = (1,2),analyzer='word',max_features = 300) 

tfidf = vect.fit_transform(ldes) 
    
feature_names = vect.get_feature_names() 

dense = tfidf.todense() 
dense.shape 
df = pd.DataFrame(dense)

model = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

