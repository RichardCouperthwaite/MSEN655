# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:42:18 2019

@author: richardcouperthwaite
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import plot_model
import numpy as np

def final_structure(categorical, n):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2, 2 ), activation='relu'))

    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    if categorical:
        model.add(Dense(n, activation='softmax'))
    else:
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n, activation='softmax'))

    return model