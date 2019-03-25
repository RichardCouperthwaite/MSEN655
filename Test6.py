# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:45:21 2019

@author: Richard Couperthwaite

Python File to Construct the Models required for Test 6
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import h5py

def get_data(name):
    f = h5py.File(name, 'r')
    x_train = np.array(f["train_features"])
    y_train = np.array(f["train_output"])
    x_test = np.array(f["test_features"])
    y_test = np.array(f["test_output"])
    
    return x_train, x_test, y_train, y_test

def test6_method1():
    filenames = ['test1_model1.hdf5', 'test1_model2.hdf5', 'test1_model3.hdf5',
                 'test1_model4.hdf5', 'test1_model5.hdf5', 'test1_model6.hdf5',
                 'test1_model7.hdf5', 'test1_model8.hdf5', 'test1_model9.hdf5',
                 'test1_model10.hdf5', 'test1_model11.hdf5', 'test1_model12.hdf5',
                 'test2_model1.hdf5', 'test2_model2.hdf5', 'test2_model3.hdf5',
                 'test2_model4.hdf5', 'test2_model5.hdf5', 'test2_model6.hdf5',
                 'test3_model1.hdf5', 'test3_model2.hdf5', 'test4_model1.hdf5',
                 'test5_model1.hdf5', 'test5_model2.hdf5', 'test5_model3.hdf5']









def test_structure(categorical, n):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    if categorical:
        model.add(Dense(n, activation='softmax'))
    else:
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n, activation='softmax'))

    return model

if __name__ == "__main__":
    #run some code
    pass