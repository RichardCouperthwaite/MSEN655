# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:42:29 2019

@author: richardcouperthwaite
"""

def final():
    def test2_model1(categorical, n):
    kern_sz = 4
    str_sz = 2
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kern_sz, kern_sz), strides=(str_sz, str_sz), activation='relu'))

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

    return model, 'dense_3'