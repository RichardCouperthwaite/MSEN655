# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:21:04 2019

@author: jaylen_james
Function to evaluate hyper-parameters for Random Forrest regression
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LassoCV
import numpy as np
import h5py
import csv


def lassoopt(x_train, y_train, x_test, y_test, multiplier):
    
    clf = linear_model.Lasso(alpha = multiplier, fit_intercept = True, normalize = True)
    
    clf.fit(x_train, y_train)
    
    score = clf.score(x_test, y_test)
    print(clf.coef_)
    
    reg = LassoCV().fit(x_train, y_train)
    scoreCV = reg.score(x_test, y_test)
    print(reg.coef_)
    print(scoreCV)
    
    return score, multiplier
    
    

if __name__ == "__main__":
    
    #import temperature training data
    f = h5py.File("Final_temp.hdf5", "r")
    x_data_train = np.array(f["train_features"])
    y_data_train = np.array(f["train_output"])
    
    x_data_test = np.array(f["test_features"])
    y_data_test = np.array(f["test_output"])
    
    
    #Open file to write random forest results to
    with open('lasso_temp_results.csv', 'w') as lasso_temp_results_file:
        lasso_writer = csv.writer(lasso_temp_results_file, delimiter=',')
        
        #Loop through number of tree values i.e. struct_matrix columns
        for idx in range(0,51,5):
            if idx == 0:
                idx = 1
            #Save to csv file
            result = lassoopt(x_data_train, y_data_train, x_data_test, y_data_test, idx)
            lasso_writer.writerow([result[0], result[1]])
        
        
    #import time training and test data
    f = h5py.File("Final_time.hdf5", "r")
    x_data_train = np.array(f["train_features"])
    y_data_train = np.array(f["train_output"])
    
    x_data_test = np.array(f["test_features"])
    y_data_test = np.array(f["test_output"])
    
    #Open file to write random forest results to
    with open('lasso_time_results.csv', 'w') as lasso_time_results_file:
        lasso_writer = csv.writer(lasso_time_results_file, delimiter=',')
        
        #Loop through number of tree values i.e. struct_matrix columns
        for idx in range(0,51,5):
            if idx == 0:
                idx = 1
            #Save to csv file
            result = lassoopt(x_data_train, y_data_train, x_data_test, y_data_test, idx)
            lasso_writer.writerow([result[0], result[1]])