# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:45:21 2019

@author: Richard Couperthwaite

Python File to Construct the Models required for Test 2

Six functions are defined that return convolutional neural network models with varying kernel and step size
The values are:
    Kernel Size: 3, 4, 5
    Stride Size: 1, 2
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import plot_model
import numpy as np

def test2_model1(categorical, n):
    kern_sz = 3
    str_sz = 1
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
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

    return model, 'dense_3'

def test2_model2(categorical, n):
    kern_sz = 4
    str_sz = 1
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
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

    return model, 'dense_3'

def test2_model3(categorical, n):
    kern_sz = 5
    str_sz = 1
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
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

    return model, 'dense_3'

def test2_model4(categorical, n):
    kern_sz = 3
    str_sz = 2
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

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

    return model, 'dense_3'

def test2_model5(categorical, n):
    kern_sz = 4
    str_sz = 2
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

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

    return model, 'dense_3'

def test2_model6(categorical, n):
    kern_sz = 5
    str_sz = 2
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

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

    return model, 'dense_3'



if __name__ == "__main__":
    #run some code
    pass