#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:37:59 2017

@author: prudhvi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 18:12:20 2017

@author: Prudhvi """

os.chdir("/home/prudhvi/Desktop/data science/Icecream/Nlp") 



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




def create_labels(json_file)
    data = []
    with open('json_file') as f:
        for line in f:
            data.append(json.loads(line))
    tokenizer = regextoken(r'\w+')
    stop = stopwords.words('english')

    """ extracting the titles """ 
    des = []
    for i in range(len(data)):
        tokens = tokenizer.tokenize(data[i]['title'])
        tokens = [token.lower() for token in tokens if token not in stop]
        des.append((tokens))
    
    pos_tokens = []
    for i in range(len(data)):
        pos_tokens.append(nltk.pos_tag(tokenizer.tokenize(data[i]['title'])))
    pos_tokens = [item for sublist in pos_tokens for item in sublist]
    pos_tokens = list(set(pos_tokens))
    final_pos_tokens = [pos_tokens[i][0] for i in range(len(pos_tokens)) if pos_tokens[i][1] == 'NN' or pos_tokens[i][1] == 'NNS' ]
    counter = collections.Counter(final_pos_tokens)
    final_tags = counter.most_common(300)
    final_tags = [final_tags[i][0] for i in range(len(final_tags))]
    
    clusters= { }  
    for i in range(len(final_tags)):
        f_words = wn.synsets(final_tags[i])
        f_words = [item for item in f_words if item.pos() == 'n']
        f_words = [str(item)[8:-2] for item in f_words]
        for first in f_words :
            for j in range(len(final_tags)):
                if i != j :
                    s_words = wn.synsets(final_tags[j])
                    s_words = [item for item in s_words if item.pos() == 'n']
                    s_words = [str(item)[8:-2] for item in s_words]
                    for second in s_words :
                        try : 
                            group= wn.synset(first).lowest_common_hypernyms(wn.synset(second))
                            k = str(group[0])
                            k = k[8:-7]
                            if k in clusters.keys() :
                                clusters[k].append(first)
                                clusters[k].append(second)
                            else :
                                clusters[k] = []
                                clusters[k].append(first)
                                clusters[k].append(second)
                        except :
                            pass 
                            
    clusters[item] = [list(set(clusters[item])) for item in clusters.keys() 
    for item in clusters.keys():
        clusters[item] = [word.lower() for word in clusters[item]]
    
    with open('file_name','w') as f:
        fields = ['category','words']
        writer = csv.DictWriter(f, fields)
        for item in clusters.keys() :   
            writer.writerow('category' : item , 'words' : clusters[item])
    
    
    return clusters 
        
        
        

    
    
        
                    
