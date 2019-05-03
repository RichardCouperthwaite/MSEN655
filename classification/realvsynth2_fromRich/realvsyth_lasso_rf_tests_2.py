# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:15:31 2019

@author: jaylen_james
Real vs. Synthetic Tests
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


#Returns score of optimized LassoCV given data
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


#Returns score of Random Forest given data. (Best structure found no. of trees = )
def randforestopt(x_train, y_train, x_test, y_test, numberoftrees, maxtreedepth):
    
    clf = RandomForestClassifier(n_estimators = numberoftrees, max_depth=maxtreedepth)
    
    clf.fit(x_train, y_train)
    
    score = clf.score(x_test, y_test)
    
    
    return score



Temp_orig_all = genfromtxt('Temp_orig_synthMatch.csv', delimiter=' ')
Time_orig_all = genfromtxt('Time_orig_synthMatch.csv', delimiter=' ')
Temp_synth_all = genfromtxt('Temp_synth_synthMatch.csv', delimiter=' ')
Time_synth_all = genfromtxt('Time_synth_synthMatch.csv', delimiter=' ')
xFTime_orig_all = genfromtxt('xFTime_orig_synthMatch.csv', delimiter=' ')
xFTemp_orig_all = genfromtxt('xFTemp_orig_synthMatch.csv', delimiter=' ')
xT1M2_orig_all = genfromtxt('xT1M2_orig_synthMatch.csv', delimiter=' ')
xT2M3_orig_all = genfromtxt('xT2M3_orig_synthMatch.csv', delimiter=' ')
xT5M4_orig_all = genfromtxt('xT5M4_orig_synthMatch.csv', delimiter=' ')
xFTime_synth_all = genfromtxt('xFTime_synth_synthMatch.csv', delimiter=' ')
xFTemp_synth_all = genfromtxt('xFTemp_synth_synthMatch.csv', delimiter=' ')
xT1M2_synth_all = genfromtxt('xT1M2_synth_synthMatch.csv', delimiter=' ')
xT2M3_synth_all = genfromtxt('xT2M3_synth_synthMatch.csv', delimiter=' ')
xT5M4_synth_all = genfromtxt('xT5M4_synth_synthMatch.csv', delimiter=' ')

Temp_synth_partial = genfromtxt('Temp_synth_0.csv', delimiter=' ')
Time_synth_partial = genfromtxt('Time_synth_0.csv', delimiter=' ')
xFTime_synth_0 = genfromtxt('xFTime_synth_0.csv', delimiter=' ')
xFTemp_synth_0 = genfromtxt('xFTemp_synth_0.csv', delimiter=' ')
xT1M2_synth_0 = genfromtxt('xT1M2_synth_0.csv', delimiter=' ')
xT2M3_synth_0 = genfromtxt('xT2M3_synth_0.csv', delimiter=' ')
xT5M4_synth_0 = genfromtxt('xT5M4_synth_0.csv', delimiter=' ')

FTemp_synth_1 = genfromtxt('Temp_synth_1.csv', delimiter=' ')
FTime_synth_1 = genfromtxt('Time_synth_1.csv', delimiter=' ')
xFTime_synth_1 = genfromtxt('xFTime_synth_1.csv', delimiter=' ')
xFTemp_synth_1 = genfromtxt('xFTemp_synth_1.csv', delimiter=' ')
xT1M2_synth_1 = genfromtxt('xT1M2_synth_1.csv', delimiter=' ')
xT2M3_synth_1 = genfromtxt('xT2M3_synth_1.csv', delimiter=' ')
xT5M4_synth_1 = genfromtxt('xT5M4_synth_1.csv', delimiter=' ')

FTemp_synth_2 = genfromtxt('Temp_synth_2.csv', delimiter=' ')
FTime_synth_2 = genfromtxt('Time_synth_2.csv', delimiter=' ')
xFTime_synth_2 = genfromtxt('xFTime_synth_2.csv', delimiter=' ')
xFTemp_synth_2 = genfromtxt('xFTemp_synth_2.csv', delimiter=' ')
xT1M2_synth_2 = genfromtxt('xT1M2_synth_2.csv', delimiter=' ')
xT2M3_synth_2 = genfromtxt('xT2M3_synth_2.csv', delimiter=' ')
xT5M4_synth_2 = genfromtxt('xT5M4_synth_2.csv', delimiter=' ')

FTemp_synth_3 = genfromtxt('Temp_synth_3.csv', delimiter=' ')
FTime_synth_3 = genfromtxt('Time_synth_3.csv', delimiter=' ')
xFTime_synth_3 = genfromtxt('xFTime_synth_3.csv', delimiter=' ')
xFTemp_synth_3 = genfromtxt('xFTemp_synth_3.csv', delimiter=' ')
xT1M2_synth_3 = genfromtxt('xT1M2_synth_3.csv', delimiter=' ')
xT2M3_synth_3 = genfromtxt('xT2M3_synth_3.csv', delimiter=' ')
xT5M4_synth_3 = genfromtxt('xT5M4_synth_3.csv', delimiter=' ')

FTemp_synth_4 = genfromtxt('Temp_synth_4.csv', delimiter=' ')
FTime_synth_4 = genfromtxt('Time_synth_4.csv', delimiter=' ')
xFTime_synth_4 = genfromtxt('xFTime_synth_4.csv', delimiter=' ')
xFTemp_synth_4 = genfromtxt('xFTemp_synth_4.csv', delimiter=' ')
xT1M2_synth_4 = genfromtxt('xT1M2_synth_4.csv', delimiter=' ')
xT2M3_synth_4 = genfromtxt('xT2M3_synth_4.csv', delimiter=' ')
xT5M4_synth_4 = genfromtxt('xT5M4_synth_4.csv', delimiter=' ')

#Calculate R^2 with RF and LassoCV with real data only
xFTemp_orig_all_train, xFTemp_orig_all_test, Temp_orig_all_train, Temp_orig_all_test = train_test_split(xFTemp_orig_all,
                                Temp_orig_all, test_size=0.1 )


xFTime_orig_all_train, xFTime_orig_all_test, Time_orig_all_train, Time_orig_all_test = train_test_split(xFTime_orig_all,
                                Time_orig_all, test_size=0.1 )


lassocv_Temp_orig_all_score = lassoopt(xFTemp_orig_all_train, Temp_orig_all_train, xFTemp_orig_all_test, Temp_orig_all_test, 1 )
lassocv_Time_orig_all_score = lassoopt(xFTime_orig_all_train, Time_orig_all_train, xFTime_orig_all_test, Time_orig_all_test, 1 )

rf_Temp_orig_all_score = randforestopt(xFTemp_orig_all_train, Temp_orig_all_train, xFTemp_orig_all_test, Temp_orig_all_test, 20, 15)
rf_Time_orig_all_score = randforestopt(xFTime_orig_all_train, Time_orig_all_train, xFTime_orig_all_test, Time_orig_all_test, 10, 5)


#Calculate R^2 with 1 of 5 images from each type###############################
xzero_of_4_Temp = np.concatenate((xFTemp_orig_all, xFTemp_synth_0))
xzero_of_4_Time = np.concatenate((xFTime_orig_all, xFTime_synth_0))

zero_of_4_Temp = np.concatenate((Temp_orig_all, Temp_synth_partial))
zero_of_4_Time = np.concatenate((Time_orig_all, Time_synth_partial))


xzero_of_4_Temp_train, xzero_of_4_Temp_test, zero_of_4_Temp_train, zero_of_4_Temp_test = train_test_split(xzero_of_4_Temp,
                                zero_of_4_Temp, test_size=0.1 )

xzero_of_4_Time_train, xzero_of_4_Time_test, zero_of_4_Time_train, zero_of_4_Time_test = train_test_split(xzero_of_4_Time,
                                zero_of_4_Time, test_size=0.1 )

lassocv_zero_of_4_Temp_score = lassoopt(xzero_of_4_Temp_train, zero_of_4_Temp_train, xzero_of_4_Temp_test, zero_of_4_Temp_test, 1 )
lassocv_zero_of_4_Time_score = lassoopt(xzero_of_4_Time_train, zero_of_4_Time_train, xzero_of_4_Time_test, zero_of_4_Time_test, 1 )

rf_zero_of_4_Temp_score = randforestopt(xzero_of_4_Temp_train, zero_of_4_Temp_train, xzero_of_4_Temp_test, zero_of_4_Temp_test, 20, 15)
rf_zero_of_4_Time_score = randforestopt(xzero_of_4_Time_train, zero_of_4_Time_train, xzero_of_4_Time_test, zero_of_4_Time_test, 10,15)

#Calculate R^2 with 2 of 5 images from each type###############################
xone_of_4_Temp = np.concatenate((xFTemp_orig_all, xFTemp_synth_0, xFTemp_synth_1))
xone_of_4_Time = np.concatenate((xFTime_orig_all, xFTime_synth_0, xFTime_synth_1))

one_of_4_Temp = np.concatenate((Temp_orig_all, Temp_synth_partial, FTemp_synth_1))
one_of_4_Time = np.concatenate((Time_orig_all, Time_synth_partial, FTime_synth_1))


xone_of_4_Temp_train, xone_of_4_Temp_test, one_of_4_Temp_train, one_of_4_Temp_test = train_test_split(xone_of_4_Temp,
                                one_of_4_Temp, test_size=0.1 )

xone_of_4_Time_train, xone_of_4_Time_test, one_of_4_Time_train, one_of_4_Time_test = train_test_split(xone_of_4_Time,
                                one_of_4_Time, test_size=0.1 )

lassocv_one_of_4_Temp_score = lassoopt(xone_of_4_Temp_train, one_of_4_Temp_train, xone_of_4_Temp_test, one_of_4_Temp_test, 1 )
lassocv_one_of_4_Time_score = lassoopt(xone_of_4_Time_train, one_of_4_Time_train, xone_of_4_Time_test, one_of_4_Time_test, 1 )

rf_one_of_4_Temp_score = randforestopt(xone_of_4_Temp_train, one_of_4_Temp_train, xone_of_4_Temp_test, one_of_4_Temp_test, 20, 15)
rf_one_of_4_Time_score = randforestopt(xone_of_4_Time_train, one_of_4_Time_train, xone_of_4_Time_test, one_of_4_Time_test, 10,15)


#Calculate R^2 with 3 of 5 images from each type###############################
xtwo_of_4_Temp = np.concatenate((xFTemp_orig_all, xFTemp_synth_0, xFTemp_synth_1, xFTemp_synth_2))
xtwo_of_4_Time = np.concatenate((xFTime_orig_all, xFTime_synth_0, xFTime_synth_1, xFTime_synth_2))

two_of_4_Temp = np.concatenate((Temp_orig_all, Temp_synth_partial, FTemp_synth_1, FTemp_synth_2))
two_of_4_Time = np.concatenate((Time_orig_all, Time_synth_partial, FTime_synth_1, FTime_synth_2))


xtwo_of_4_Temp_train, xtwo_of_4_Temp_test, two_of_4_Temp_train, two_of_4_Temp_test = train_test_split(xtwo_of_4_Temp,
                                two_of_4_Temp, test_size=0.1 )

xtwo_of_4_Time_train, xtwo_of_4_Time_test, two_of_4_Time_train, two_of_4_Time_test = train_test_split(xtwo_of_4_Time,
                                two_of_4_Time, test_size=0.1 )

lassocv_two_of_4_Temp_score = lassoopt(xtwo_of_4_Temp_train, two_of_4_Temp_train, xtwo_of_4_Temp_test, two_of_4_Temp_test, 1 )
lassocv_two_of_4_Time_score = lassoopt(xtwo_of_4_Time_train, two_of_4_Time_train, xtwo_of_4_Time_test, two_of_4_Time_test, 1 )

rf_two_of_4_Temp_score = randforestopt(xtwo_of_4_Temp_train, two_of_4_Temp_train, xtwo_of_4_Temp_test, two_of_4_Temp_test, 20, 15)
rf_two_of_4_Time_score = randforestopt(xtwo_of_4_Time_train, two_of_4_Time_train, xtwo_of_4_Time_test, two_of_4_Time_test, 10,15)


#Calculate R^2 with 4 of 5 images from each type###############################
xthree_of_4_Temp = np.concatenate((xFTemp_orig_all, xFTemp_synth_0, xFTemp_synth_1, xFTemp_synth_2, xFTemp_synth_3))
xthree_of_4_Time = np.concatenate((xFTime_orig_all, xFTime_synth_0, xFTime_synth_1, xFTime_synth_2, xFTime_synth_3))

three_of_4_Temp = np.concatenate((Temp_orig_all, Temp_synth_partial, FTemp_synth_1, FTemp_synth_2, FTemp_synth_3))
three_of_4_Time = np.concatenate((Time_orig_all, Time_synth_partial, FTime_synth_1, FTime_synth_2, FTime_synth_3))


xthree_of_4_Temp_train, xthree_of_4_Temp_test, three_of_4_Temp_train, three_of_4_Temp_test = train_test_split(xthree_of_4_Temp,
                                three_of_4_Temp, test_size=0.1 )

xthree_of_4_Time_train, xthree_of_4_Time_test, three_of_4_Time_train, three_of_4_Time_test = train_test_split(xthree_of_4_Time,
                                three_of_4_Time, test_size=0.1 )

lassocv_three_of_4_Temp_score = lassoopt(xthree_of_4_Temp_train, three_of_4_Temp_train, xthree_of_4_Temp_test, three_of_4_Temp_test, 1 )
lassocv_three_of_4_Time_score = lassoopt(xthree_of_4_Time_train, three_of_4_Time_train, xthree_of_4_Time_test, three_of_4_Time_test, 1 )

rf_three_of_4_Temp_score = randforestopt(xthree_of_4_Temp_train, three_of_4_Temp_train, xthree_of_4_Temp_test, three_of_4_Temp_test, 20, 15)
rf_three_of_4_Time_score = randforestopt(xthree_of_4_Time_train, three_of_4_Time_train, xthree_of_4_Time_test, three_of_4_Time_test, 10,15)

#Calculate R^2 with 1 of 5 images from each type###############################
xfour_of_4_Temp = np.concatenate((xFTemp_orig_all, xFTemp_synth_0, xFTemp_synth_1, xFTemp_synth_2, xFTemp_synth_3, xFTemp_synth_4))
xfour_of_4_Time = np.concatenate((xFTime_orig_all, xFTime_synth_0, xFTime_synth_1, xFTime_synth_2, xFTime_synth_3, xFTime_synth_4))

four_of_4_Temp = np.concatenate((Temp_orig_all, Temp_synth_partial, FTemp_synth_1, FTemp_synth_2, FTemp_synth_3, FTemp_synth_4))
four_of_4_Time = np.concatenate((Time_orig_all, Time_synth_partial, FTime_synth_1, FTime_synth_2, FTime_synth_3, FTime_synth_4))


xfour_of_4_Temp_train, xfour_of_4_Temp_test, four_of_4_Temp_train, four_of_4_Temp_test = train_test_split(xfour_of_4_Temp,
                                four_of_4_Temp, test_size=0.1 )

xfour_of_4_Time_train, xfour_of_4_Time_test, four_of_4_Time_train, four_of_4_Time_test = train_test_split(xfour_of_4_Time,
                                four_of_4_Time, test_size=0.1 )

lassocv_four_of_4_Temp_score = lassoopt(xfour_of_4_Temp_train, four_of_4_Temp_train, xfour_of_4_Temp_test, four_of_4_Temp_test, 1 )
lassocv_four_of_4_Time_score = lassoopt(xfour_of_4_Time_train, four_of_4_Time_train, xfour_of_4_Time_test, four_of_4_Time_test, 1 )

rf_four_of_4_Temp_score = randforestopt(xfour_of_4_Temp_train, four_of_4_Temp_train, xfour_of_4_Temp_test, four_of_4_Temp_test, 20, 15)
rf_four_of_4_Time_score = randforestopt(xfour_of_4_Time_train, four_of_4_Time_train, xfour_of_4_Time_test, four_of_4_Time_test, 10,15)

###############################################################################


with open("lassocv_time_scores.pickle", 'wb') as f:
    pickle.dump([lassocv_Time_orig_all_score, lassocv_one_of_4_Time_score, 
                 lassocv_two_of_4_Time_score, lassocv_three_of_4_Time_score, 
                 lassocv_four_of_4_Time_score, rf_Time_orig_all_score, 
                 rf_one_of_4_Time_score, rf_two_of_4_Time_score, 
                 rf_three_of_4_Time_score, rf_four_of_4_Time_score, 
                 lassocv_Temp_orig_all_score, lassocv_one_of_4_Temp_score, 
                 lassocv_two_of_4_Temp_score, lassocv_three_of_4_Temp_score, 
                 lassocv_four_of_4_Temp_score, rf_Temp_orig_all_score, 
                 rf_one_of_4_Temp_score, rf_two_of_4_Temp_score, 
                 rf_three_of_4_Temp_score, rf_four_of_4_Temp_score], f)





#Plot Lasso and Random Forest Results for Time Predictions
#lassoCV_time = [lassocv_Time_orig_all_score, lassocv_one_of_4_Time_score, lassocv_two_of_4_Time_score, lassocv_three_of_4_Time_score, lassocv_four_of_4_Time_score]
#rf_time = [rf_Time_orig_all_score, rf_one_of_4_Time_score, rf_two_of_4_Time_score, rf_three_of_4_Time_score, rf_four_of_4_Time_score]
#x_axis = np.arange(1,6, step=1)#[1,2,3,4,5]
#
#fig, ax = plt.subplots()
#ax.plot(x_axis, lassoCV_time, label="Lasso CV")
#ax.plot(x_axis, rf_time, label="Random Forest")
#ax.legend()
#
#plt.show()

#Plot Lasso and Random Forest Results for Temperature Predictions
lassoCV_temp = [lassocv_Temp_orig_all_score, lassocv_one_of_4_Temp_score, lassocv_two_of_4_Temp_score, lassocv_three_of_4_Temp_score, lassocv_four_of_4_Temp_score]
rf_temp = [rf_Temp_orig_all_score, rf_one_of_4_Temp_score, rf_two_of_4_Temp_score, rf_three_of_4_Temp_score, rf_four_of_4_Temp_score]
x_axis = [0,1,2,3,4]

fig, ax2 = plt.subplots()
ax2.plot(x_axis, lassoCV_temp, '--o' , label="Lasso CV-Temp")
ax2.plot(x_axis, rf_temp, '--o' ,label="Random Forest-Temp")
ax2.plot(x_axis, lassoCV_time, '--o' , label="Lasso CV-Time")
ax2.plot(x_axis, rf_time, '--o' , label="Random Forest-Time")
#ax2.ylim(0,1)
ax2.legend()

plt.title("Effect of Using Different Amounts of Synthetic Data \n on CoD for Lasso CV and Random Forest")
plt.ylim(0,1)
plt.xlabel("Number of Synthetic Batches Used in Training")
plt.ylabel("Coefficient of Determination")
plt.show()




