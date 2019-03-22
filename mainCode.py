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
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input
import numpy as np
import json
import time

def load_micrograph(img_path):
    set_image_dim_ordering('tf')
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

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
    
    

if __name__ == "__main__":
    start = time.time()
    # obtain the dictionary of filenames for the cropped images in the UHCS dataset
    with open('Micrograph_data.json', 'r') as f:
        cropsData = json.load(f)
        
    # function can be run to confirm the levels hard coded below
    #micros, cool, temp, time = get_parameter_levels(cropsData)
    
    # Hard coded levels for the parameters of interest
    #micros = ['spheroidite', 'pearlite+spheroidite', 'network', 'spheroidite+widmanstatten', 'pearlite', 'pearlite+widmanstatten', 'martensite']
    #cool = ['Q', 'N/A', 'AR', '650-1H', 'FC']
    #temp = [800.0, 0, 970.0, 1100.0, 900.0, 1000.0, 700.0, 750.0]
    #time = [90.0, 0, 180.0, 1440.0, 5100.0, 60.0, 5.0, 480.0, 2880.0]
    
    index = 0
    count = 0
    
    for label in cropsData:
        print("\r{}% Completed | {} images skipped | Current Label: {}        ".format((round(index/len(cropsData)*100, 1)), count, label), end='')
        if label not in ['_defaultTraceback', '_default', 'No-Treatment']:
            if label not in cropsData['No-Treatment']:
                new_img = load_micrograph(cropsData[label]['Path'])
                if index == 0:
                    inputs = new_img
                    y_micro = [cropsData[label]['Primary_Microconstituent']]
                    y_cool = [cropsData[label]['Cool Method']]
                    y_time =[cropsData[label]['Anneal Time']]
                    y_temp = [cropsData[label]['Anneal Temperature']]
                    
                else:
                    inputs = np.r_[inputs, new_img]
                    y_micro.append(cropsData[label]['Primary_Microconstituent'])
                    y_cool.append(cropsData[label]['Cool Method'])
                    y_time.append(cropsData[label]['Anneal Time'])
                    y_temp.append(cropsData[label]['Anneal Temperature'])
            else:
                count += 1
        index += 1
                
    end = time.time()
    print("\n Time Taken (min): ", round((end-start)/60,2))
                