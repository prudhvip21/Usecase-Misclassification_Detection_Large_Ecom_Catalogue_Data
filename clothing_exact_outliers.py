#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:29:50 2017

@author: prudhvi
"""

import os  
os.chdir("/home/prudhvi/Desktop/data science/Icecream/Nlp") 


import csv

outliers_data = [ ]
with open('Redirecting.txt', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        outliers_data.append(row)