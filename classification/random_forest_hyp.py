# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:21:04 2019

@author: jaylen_james
Function to evaluate hyper-parameters for Random Forrest regression
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import h5py
import csv


def randforestopt(x_train, y_train, x_test, y_test, numberoftrees, maxtreedepth):
    
    clf = RandomForestClassifier(n_estimators = numberoftrees, max_depth=maxtreedepth)
    
    clf.fit(x_train, y_train)
    
    score = clf.score(x_test, y_test)
    
    
    return score, numberoftrees, maxtreedepth
    

    

if __name__ == "__main__":
    
    #import temperature training data
    f = h5py.File("Final_temp.hdf5", "r")
    x_data_train = np.array(f["train_features"])
    y_data_train = np.array(f["train_output"])
    
    x_data_test = np.array(f["test_features"])
    y_data_test = np.array(f["test_output"])
    
    #Open file to write random forest results to
    with open('rnd_forest_hpy_temp_results.csv', 'w') as rnd_forest_hyp_temp_results_file:
        forest_writer = csv.writer(rnd_forest_hyp_temp_results_file, delimiter=',')
        
        #Loop through number of tree values
        for i in range(0,201,5):
            if i == 0:
                i = 1
            #Loop through max tree depth values
            for j in range(0,51,5):
                if j == 0:
                    j = 1
                #Save to csv file
                result = randforestopt(x_data_train, y_data_train, x_data_test, y_data_test, i, j)
                forest_writer.writerow([result[0], result[1], result[2]])
            
            
    #import time training and test data
    f = h5py.File("Final_time.hdf5", "r")
    x_data_train = np.array(f["train_features"])
    y_data_train = np.array(f["train_output"])
    
    x_data_test = np.array(f["test_features"])
    y_data_test = np.array(f["test_output"])
    
    #Open file to write random forest results to
    with open('rnd_forest_hpy_time_results.csv', 'w') as rnd_forest_hyp_time_results_file:
        forest_writer = csv.writer(rnd_forest_hyp_time_results_file, delimiter=',')
        
        #Loop through number of tree values
        for i in range(0,201,5):
            if i == 0:
                i = 1
            #Loop through max tree depth values
            for j in range(0,51,5):
                if j == 0:
                    j = 1
                #Save to csv file
                result = randforestopt(x_data_train, y_data_train, x_data_test, y_data_test, i, j)
                forest_writer.writerow([result[0], result[1], result[2]])
        
    