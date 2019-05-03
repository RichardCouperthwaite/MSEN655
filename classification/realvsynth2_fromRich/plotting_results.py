# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:10:32 2019

@author: jaylen_james

Plotting the results
"""

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pylab as pl
import pickle





with open('lassocv_time_scores.pickle', 'rb') as f:
    lassocv_Time_orig_all_score, lassocv_one_of_4_Time_score, 
    lassocv_two_of_4_Time_score, lassocv_three_of_4_Time_score, 
    lassocv_four_of_4_Time_score, rf_Time_orig_all_score, 
    rf_one_of_4_Time_score, rf_two_of_4_Time_score, 
    rf_three_of_4_Time_score, rf_four_of_4_Time_score, 
    lassocv_Temp_orig_all_score, lassocv_one_of_4_Temp_score, 
    lassocv_two_of_4_Temp_score, lassocv_three_of_4_Temp_score, 
    lassocv_four_of_4_Temp_score, rf_Temp_orig_all_score, 
    rf_one_of_4_Temp_score, rf_two_of_4_Temp_score, 
    rf_three_of_4_Temp_score, rf_four_of_4_Temp_score = pickle.load(f)
                 
                 
                 
#Plot Lasso and Random Forest Results for Time Predictions
lassoCV_time = [lassocv_Time_orig_all_score, lassocv_one_of_4_Time_score, lassocv_two_of_4_Time_score, lassocv_three_of_4_Time_score, lassocv_four_of_4_Time_score]
rf_time = [rf_Time_orig_all_score, rf_one_of_4_Time_score, rf_two_of_4_Time_score, rf_three_of_4_Time_score, rf_four_of_4_Time_score]
x_axis = [1,2,3,4,5]

fig, ax = plt.subplots()
ax.plot(x_axis, lassoCV_time, label="Lasso CV")
ax.plot(x_axis, rf_time, label="Random Forest")
ax.ylim(0,1)
ax.legend()

plt.show()

#Plot Lasso and Random Forest Results for Temperature Predictions
lassoCV_temp = [lassocv_Temp_orig_all_score, lassocv_one_of_4_Temp_score, lassocv_two_of_4_Temp_score, lassocv_three_of_4_Temp_score, lassocv_four_of_4_Temp_score]
rf_temp = [rf_Temp_orig_all_score, rf_one_of_4_Temp_score, rf_two_of_4_Temp_score, rf_three_of_4_Temp_score, rf_four_of_4_Temp_score]

fig, ax2 = plt.subplots()
ax2.plot(x_axis, lassoCV_temp, label="Lasso CV")
ax2.plot(x_axis, rf_temp, label="Random Forest")
ax2.plot(x_axis, lassoCV_time, label="Lasso CV-Time")
ax2.plot(x_axis, rf_time, label="Random Forest-Time")
ax2.legend()

plt.show()