# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:15:05 2018

@author: adityabiswas
"""


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import plot

import sys
root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
sys.path.append(root)


fileLoc = os.path.join('G:', 'Data', 'The Sugar', 'glucose static.csv')
data = pd.read_csv(fileLoc)



predictors = "ethnicity, race, sex, age, initial_creatinine, initial_sodium, \
initial_chloride, initial_potassium, initial_bicarbonate, initial_systolic, \
initial_diastolic, initial_pulse, initial_spo2, congestive_heart_failure, \
diabetes, liver_disease, ckd".split(sep = ', ')

outcome = 'hypo40'

############################################################################
## xgboost model

import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import roc_auc_score as AUC
import graphviz

cutoff = int(np.round(len(data)*0.7))
iters = 100
aucs = np.zeros(iters, dtype = np.float32)
for i in range(iters):
    data = data.sample(frac = 1.)
    train, test = data.iloc[:cutoff,:], data.iloc[cutoff:,:]
    
    dtrain = xgb.DMatrix(train[predictors], label = train[outcome])
    dtest = xgb.DMatrix(test[predictors], label = test[outcome])
    
    param = {'max_depth': 6, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    #param['eval_metric'] = 'auc'
    #evallist = [ (dtrain, 'train'), (dtest, 'eval') ]
    num_round = 150
    
    bst = xgb.train(param, dtrain, num_round)#, evallist)
    Ptest = bst.predict(dtest)
    aucs[i] = AUC(test[outcome].values, Ptest)
    if (i+1) % 10 == 0:
        print(i+1, ' iterations completed')
    #bst.get_score()

# mean: 0.775, std: 0.013

##################################################################################
## logistic regression model

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from Helper.utilities import showCoef
from Helper.preprocessing import Transformer

cutoff = int(np.round(len(data)*0.7))
iters = 100
aucs = np.zeros(iters, dtype = np.float32)
for i in range(iters):
    data = data.sample(frac = 1.)
    train, test = data.iloc[:cutoff,:], data.iloc[cutoff:,:]
    
    Xtrain = train[predictors].values
    Xtest = test[predictors].values
    Ytrain = train[outcome].values
    Ytest = test[outcome].values
    
    transformer = Transformer()
    Xtrain = transformer.fit_transform(Xtrain)
    Xtest = transformer.transform(Xtest)
    
    
    model = LR(class_weight = 'balanced', C = 1e-1)
    model.fit(Xtrain, Ytrain)
    #showCoef(model.coef_[0], predictors)
    
    #Ptrain = model.predict_proba(Xtrain)[:,1]
    Ptest = model.predict_proba(Xtest)[:,1]
    #P = np.concatenate([Ptrain, Ptest])
    aucs[i] = AUC(Ytest, Ptest)
    if (i+1) % 10 == 0:
        print(i+1, ' iterations completed')

## mean: 0.755, std = 0.013