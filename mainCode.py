# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:45:21 2019

@author: Richard Couperthwaite

Python File to Run all the individual tests
"""

import Test1
import Test2
import Test3
import Test4
import Test5
import Test6
from keras.backend import set_image_dim_ordering
from keras.preprocessing.image import load_img, img_to_array, image
from keras.applications.vgg19 import preprocess_input, VGG19
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import optimizers, losses
from keras.utils import to_categorical, print_summary
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import json
import time


def load_micrograph(img_path):
    set_image_dim_ordering('tf')
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def import_data():
    start = time.time()
    # obtain the dictionary of filenames for the cropped images in the UHCS dataset
    with open('Micrograph_data.json', 'r') as f:
        cropsData = json.load(f)
        
    # function can be run to confirm the levels hard coded below
    #micros, cool, temp, time = get_parameter_levels(cropsData)
    
    # Hard coded levels for the parameters of interest
    micros = ['spheroidite', 'pearlite+spheroidite', 'network', 'spheroidite+widmanstatten', 'pearlite', 'pearlite+widmanstatten', 'martensite']
    cool = ['Q', 'N/A', 'AR', '650-1H', 'FC']
    temp = [0, 800.0, 970.0, 1100.0, 900.0, 1000.0, 700.0, 750.0]
    times = [0, 90.0, 180.0, 1440.0, 5100.0, 60.0, 5.0, 480.0, 2880.0]
    
    index = 0
    count = 0
    
    # retrieve the meta-data and images
    for label in cropsData:
        print("\r{}% Completed | {}/{} images skipped | Current Label: {}        ".format((round(index/len(cropsData)*100, 1)), count, len(cropsData), label), end='')
        if label not in ['_defaultTraceback', '_default', 'No-Treatment']:
            if label not in cropsData['No-Treatment']:
                new_img = load_micrograph(cropsData[label]['Path'])
                try:
                    inputs = np.r_[inputs, new_img]
                    y_micro.append(micros.index(cropsData[label]['Primary_Microconstituent']))
                    y_cool.append(cool.index(cropsData[label]['Cool Method']))
                    y_time.append(times.index(cropsData[label]['Anneal Time']))
                    y_temp.append(temp.index(cropsData[label]['Anneal Temperature']))
                except NameError:
                    inputs = new_img
                    y_micro = [micros.index(cropsData[label]['Primary_Microconstituent'])]
                    y_cool = [cool.index(cropsData[label]['Cool Method'])]
                    y_time =[times.index(cropsData[label]['Anneal Time'])]
                    y_temp = [temp.index(cropsData[label]['Anneal Temperature'])]
            else:
                count += 1
        index += 1
#        # stop loop before all samples run to speed up testing
#        if index == 500:
#            break
        
    end = time.time()
    print("\nTime Taken (min): ", round((end-start)/60,2))
    # convert the outputs to categorical values for use in the training
    y_micro = to_categorical(np.array(y_micro))
    y_cool = to_categorical(np.array(y_cool))
    # keep a copy of the time and temperature as actual values for use in regression
    y_time_reg = np.array(y_time)
    y_time = to_categorical(np.array(y_time))
    y_temp_reg = np.array(y_temp)
    y_temp = to_categorical(np.array(y_temp))
    
    return inputs, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp

def get_parameter_levels(cropsData):
    micros = []
    cool = []
    temp = []
    time = []
    
    for label in cropsData:
        if label != '_default':
            if cropsData[label]['Primary_Microconstituent'] not in micros:
                micros.append(cropsData[label]['Primary_Microconstituent'])
            if cropsData[label]['Cool Method'] not in cool:
                cool.append(cropsData[label]['Cool Method'])
            if cropsData[label]['Anneal Time'] not in time:
                time.append(cropsData[label]['Anneal Time'])
            if cropsData[label]['Anneal Temperature'] not in temp:
                temp.append(cropsData[label]['Anneal Temperature'])
    return micros, cool, temp, time

def split_data(x_data, y_data, size=0.1):
    """
    Split the x and y data into training and test sets. Default test set size is 10% of the
    dataset
    """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=size, random_state=None)
    return x_train, x_test, y_train, y_test

def train_and_test_model(model, x_data, y_data, epoch_num=500, batch_num=20):
    # split the data into train and test sets
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, 0.1)
    # Use a stochastic gradient descent optimizer, and train the model
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy)
    model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num, shuffle=True)
    # obtain the test results
    test_result = model.predict(x_test)
    
    print(test_result)
    print(y_test)
    pass 

if __name__ == "__main__":
    # retrieve the data for the testing
    x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp = import_data()
    
    base_model = Test6.test_structure(True, y_micro.shape[1])
    print(print_summary(base_model))
    outlayer='dense_3'
    
    train_and_test_model(base_model, x_data, y_micro, 100)
    
    
    
    
    
#    x_train, x_test, y_train, y_test = split_data(inputs, np.array(y_time), 0.3)
#
#    print(x_train.shape)
#    print(x_test.shape)
#    print(y_train.shape)
#    print(y_test.shape)
#    
#    base_model = Test6.test_structure()
#    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    base_model.compile(optimizer=sgd, loss=losses.mean_absolute_error)
#    base_model.fit(x_train, y_train, epochs=10, batch_size=10)
##    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#    
##    block4_pool_features = model.predict(x_test)
#    block4_pool_features = base_model.predict(x_test)
#
#    print(block4_pool_features.shape)
#    plt.hist(block4_pool_features[1,:])
#    plt.hist(block4_pool_features[10,:])
#    plt.hist(block4_pool_features[20,:])
#    plt.hist(block4_pool_features[30,:])
#    plt.hist(block4_pool_features[40,:])
#    plt.hist(block4_pool_features[50,:])
#    
#    pca = PCA(n_components=5)
#    pca.fit(block4_pool_features)
#    
#    print(pca.explained_variance_)
#    print(pca.singular_values_)
                