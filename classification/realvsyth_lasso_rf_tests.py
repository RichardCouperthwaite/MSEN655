# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:44:36 2019

@author: jaylen_james

Real versus Synthetic Scores
"""
from sklearn.model_selection import train_test_split
import numpy as np

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


#Import all data, real and synthetic
temp_arr = np.genfromtxt('x_out_Final_temp_opt.csv', delimiter=',')
#time_arr = np.genfromtxt('x_out_Final_time_opt.csv', delimiter=',')
labels = np.genfromtxt('labels.csv', dtype=str)

temp_synth_arr = np.genfromtxt('x_out_synth_Final_temp_opt.csv', delimiter=',')
#time_arr = np.genfromtxt('x_out_synth_Final_time_opt.csv', delimiter=',')
synthlabels = np.genfromtxt('synthlabels.csv', dtype=str)

#Create separate image identifiers from real and synthetic data labels
labels_id_arr = np.array([])

for i in range(len(labels)):
    labels_id = labels[i].split('-')
    #print(labels_id[0])
    labels_id_arr = np.append(labels_id_arr, labels_id[0])


synthlabels_id_1_arr= np.array([])
    
for i in range(len(synthlabels)):
    synthlabels_id_1 = synthlabels[i].split('_')
    #print(synthlabels_id_1[0])
    synthlabels_id_1_arr = np.append(synthlabels_id_1_arr, synthlabels_id_1[0])
    
    
synthlabels_id_arr = np.array([])
 
for i in range(len(synthlabels_id_1_arr)):    
    synthlabels_id = synthlabels_id_1_arr[i].split('o')
    #print(synthlabels_id[1])
    synthlabels_id_arr = np.append(synthlabels_id_arr, synthlabels_id[1])


#Check which values from real image identifiers match those of synthetic image identifiers
intersect_labels_list = list(set(labels_id_arr).intersection(set(synthlabels_id_arr)))

intersect_labels_list = [int(x) for x in intersect_labels_list]


#Obtain array of index values that are contained in both real & synth data sets
matching_labels = np.array([])
labels_id_arr_int = [int(x) for x in labels_id_arr]

for i in range(len(intersect_labels_list)):
    if intersect_labels_list[i] in  labels_id_arr_int:
          print(labels_id_arr_int.index(intersect_labels_list[i]))
          matching_labels = np.append(matching_labels, labels_id_arr_int.index(intersect_labels_list[i]))
          
matching_labels = [int(x) for x in matching_labels]


#Obtain x and y outputs of real images that match sythetic images
real_of_synth_x_temp = temp_arr[matching_labels]
real_of_synth_y_temp = labels[matching_labels]

#Split data into testing and training
real_of_synth_x_temp_train, real_of_synth_x_temp_test, real_of_synth_y_temp_train, real_of_synth_y_temp_test = train_test_split(real_of_synth_x_temp, 
                                                                                                                real_of_synth_y_temp, test_size=0.1 )

#Calculate R^2 with RF and LassoCV with real data only




#Calculate R^2 with 1 of 5 images from each type

#Calculate R^2 with 2 of 5 images from each type, etc. to 5

