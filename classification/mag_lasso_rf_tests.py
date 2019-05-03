# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:39:53 2019

@author: jaylen_james

Magnification Scores
"""
from sklearn.model_selection import train_test_split
#from lasso_hyp.py import lassoopt
#from random_forest_hyp.py import randforestopt
import pandas as pd
import numpy as np
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
    
    return scoreCV

def randforestopt(x_train, y_train, x_test, y_test, numberoftrees, maxtreedepth):
    
    clf = RandomForestClassifier(n_estimators = numberoftrees, max_depth=maxtreedepth)
    
    clf.fit(x_train, y_train)
    
    score = clf.score(x_test, y_test)
    
    
    return score

#Import Data
temp_arr = np.genfromtxt('x_out_Final_temp_opt.csv', delimiter=',')
time_arr = np.genfromtxt('x_out_Final_time_opt.csv', delimiter=',')
mags_arr = np.genfromtxt('mags.csv', delimiter=',')
temp_target_arr = np.genfromtxt('y_temp_reg.csv')
time_target_arr = np.genfromtxt('y_time_reg.csv')

#Extract Low and High Magnification values for parameters
mags_low_bool_arr = mags_arr >= 1.5 
mags_high_bool_arr = mags_arr <= 0.05 

temp_low_arr = temp_arr[mags_low_bool_arr]
temp_high_arr = temp_arr[mags_high_bool_arr]
time_low_arr = time_arr[mags_low_bool_arr]
time_high_arr = time_arr[mags_high_bool_arr]

#Extract Low and High Magnification values for targets
temp_low_targets_arr = temp_target_arr[mags_low_bool_arr]
temp_high_targets_arr = temp_target_arr[mags_high_bool_arr]
time_low_targets_arr = time_target_arr[mags_low_bool_arr]
time_high_targets_arr = time_target_arr[mags_high_bool_arr]


#Create training and test set with low magnifications for temp and time
temp_x_low_train, temp_x_low_test, temp_y_low_train, temp_y_low_test = train_test_split(temp_low_arr,
                                temp_low_targets_arr, test_size=0.1 )

time_x_low_train, time_x_low_test, time_y_low_train, time_y_low_test = train_test_split(time_low_arr,
                                time_low_targets_arr, test_size=0.1 )

#Send values to rand forest and record scores of each
rf_temp_lowmag_score = randforestopt(temp_x_low_train, temp_y_low_train,
                              temp_x_low_test, temp_y_low_test, 20, 15)

rf_time_lowmag_score = randforestopt(time_x_low_train, time_y_low_train,
                              time_x_low_test, time_y_low_test, 10, 5)


#Send values to LassoCV and record R^2 scores of each
lassocv_temp_lowmag_score = lassoopt(temp_x_low_train, temp_y_low_train,
                              temp_x_low_test, temp_y_low_test, 1)

lassocv_time_lowmag_score = lassoopt(time_x_low_train, time_y_low_train,
                              time_x_low_test, time_y_low_test, 1)



#Create training and test set with high magnifications for temp and time
temp_x_high_train, temp_x_high_test, temp_y_high_train, temp_y_high_test = train_test_split(temp_high_arr,
                                temp_high_targets_arr, test_size=0.1 )

time_x_high_train, time_x_high_test, time_y_high_train, time_y_high_test = train_test_split(time_high_arr,
                                time_high_targets_arr, test_size=0.1 )

#Send values to rand forest and record scores of each
rf_temp_highmag_score = randforestopt(temp_x_high_train, temp_y_high_train,
                              temp_x_high_test, temp_y_high_test, 20, 15)

rf_time_highmag_score = randforestopt(time_x_high_train, time_y_high_train,
                              time_x_high_test, time_y_high_test, 10, 5)


#Send values to LassoCV and record R^2 scores of each
lassocv_temp_highmag_score = lassoopt(temp_x_high_train, temp_y_high_train,
                              temp_x_high_test, temp_y_high_test, 1)

lassocv_time_highmag_score = lassoopt(time_x_high_train, time_y_high_train,
                              time_x_high_test, time_y_high_test, 1)



#Calculate control scores for temp and time
temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(temp_arr,
                                temp_target_arr, test_size=0.1 )

time_x_train, time_x_test, time_y_train, time_y_test = train_test_split(time_arr,
                                time_target_arr, test_size=0.1 )

#Random Forest temp and time scores
rf_temp_score = randforestopt(temp_x_train, temp_y_train,
                              temp_x_test, temp_y_test, 20, 15)

rf_time_score = randforestopt(time_x_train, time_y_train,
                              time_x_test, time_y_test, 10, 5)

#LassoCV temp and time scores
lassocv_temp_score = lassoopt(temp_x_train, temp_y_train,
                              temp_x_test, temp_y_test, 1)

lassocv_time_score = lassoopt(time_x_train, time_y_train,
                              time_x_test, time_y_test, 1)
