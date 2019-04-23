# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

datafile = 'DScasestudy.txt'
rdf = pd.read_csv(datafile, sep = '\t')
udf = rdf.drop(['response'], axis = 1)

import seaborn as sns
import matplotlib.pyplot as plt
corr = udf.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)