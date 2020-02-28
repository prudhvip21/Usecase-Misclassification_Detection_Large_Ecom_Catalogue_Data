# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 15:28:17 2017

@author: Prudhvi
"""
import os 
os.chdir("C:\Users\Prudhvi\Desktop\data science\Icecream\Nlp") 


data_mh = []

with open('text_dump_with_mh.json') as f:
    for line in f:
        data_mh.append(json.loads(line))
        
        
