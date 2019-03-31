# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:45:21 2019

@author: Richard Couperthwaite

Python File to Run all the individual tests
"""

from Test1 import test1_model1, test1_model2, test1_model3, test1_model4, test1_model5, test1_model6, test1_model7, test1_model8, test1_model9, test1_model10, test1_model11, test1_model12
from Test2 import test2_model1, test2_model2, test2_model3, test2_model4, test2_model5, test2_model6
from Test3 import test3_model1, test3_model2
from Test4 import test4_model1
from Test5 import test5_model1, test5_model2, test5_model3, test5_model4, test5_model5, test5_model6
from Test6 import test6_regression_test
from keras.backend import set_image_dim_ordering
from keras.preprocessing.image import load_img, img_to_array, image
from keras.applications.vgg19 import preprocess_input, VGG19
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import optimizers, losses
from keras.utils import to_categorical, print_summary
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import h5py


def load_micrograph(img_path):
    set_image_dim_ordering('tf')
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def import_data():
    start = time.time()
    obtain the dictionary of filenames for the cropped images in the UHCS dataset
    with open('Micrograph_data.json', 'r') as f:
        cropsData = json.load(f)
        
    function can be run to confirm the levels hard coded below
    #micros, cool, temp, time = get_parameter_levels(cropsData)
    
    Hard coded levels for the parameters of interest
    micros = ['spheroidite', 'pearlite+spheroidite', 'network', 'spheroidite+widmanstatten', 'pearlite', 'pearlite+widmanstatten', 'martensite']
    cool = ['Q', 'N/A', 'AR', '650-1H', 'FC']
    temp = [0, 800.0, 970.0, 1100.0, 900.0, 1000.0, 700.0, 750.0]
    times = [0, 90.0, 180.0, 1440.0, 5100.0, 60.0, 5.0, 480.0, 2880.0]
    
    index = 0
    count = 0
    
    retrieve the meta-data and images
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
#        stop loop before all samples run to speed up testing
#        if index == 100:
#            break
        
    end = time.time()
    print("\nTime Taken (min): ", round((end-start)/60,2))
    convert the outputs to categorical values for use in the training
    y_micro = to_categorical(np.array(y_micro))
    y_cool = to_categorical(np.array(y_cool))
    keep a copy of the time and temperature as actual values for use in regression
    y_time_reg = np.array(y_time)
    y_time = to_categorical(np.array(y_time))
    y_temp_reg = np.array(y_temp)
    y_temp = to_categorical(np.array(y_temp))
    
    return inputs, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp

def import_from_hdf5():
    start = time.time()
    f = h5py.File("processed_data.hdf5", "r")
    x_data = np.array(f["x_data"])
    y_micro = np.array(f["y_micro"])
    y_cool = np.array(f["y_cool"])
    y_time_reg = np.array(f["y_time_reg"])
    y_time = np.array(f["y_time"])
    y_temp_reg = np.array(f["y_temp_reg"])
    y_temp = np.array(f["y_temp"])
    end = time.time()
    print("Time Taken: ", end-start)
    return x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp

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
    split the data into train and test sets
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, 0.1)
    Use a stochastic gradient descent optimizer, and train the model
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num, shuffle=True)
    obtain the test results
    test_result = model.evaluate(x_test, y_test, batch_size=20)
#    print(test_result)
    return model, test_result

def test_regression(name, base_model, x_data, y_data, layername):
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, 0.1)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("dense_{}".format(layername)).output)
    reg_features = model.predict(x_train)
    gpr = GaussianProcessRegressor(kernel=RBF(), random_state=0).fit(reg_features, y_train)
    test_features = model.predict(x_test)
    f = h5py.File("results/{}.hdf5".format(name), 'w')
    f.create_dataset("train_features", data=reg_features)
    f.create_dataset("train_output", data=y_train)
    f.create_dataset("test_features", data=test_features)
    f.create_dataset("test_output", data=y_test)
    return(gpr.score(test_features, y_test))
    
def record_result(name, class_result, score):
    with open('results/results_TEX_output.txt', 'a') as f:
        f.write("{} & {} & {} & {} \\\\ \n".format(name, round(class_result[0],2), round(class_result[1],2), round(score,2)))
    with open('results/results_csv_output.csv', 'a') as f:
        f.write("{}, {}, {}, {} \n".format(name, class_result[0], class_result[1], score))


if __name__ == "__main__":
#    retrieve the data for the testing
#    data has also been formatted into an hdf5 file for easy importing
#    x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp = import_data()
    start = time.time()
    
    x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp = import_from_hdf5()
    epochs = 80

    samplename = "test1_model1"
    name = samplename+"_micro"
    base_model, lname = test1_model1(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model1(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model1(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    samplename = "test1_model2"
    name = samplename+"_micro"
    base_model, lname = test1_model2(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model2(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model2(False, y_temp.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)

    
    samplename = "test1_model3"
    name = samplename+"_micro"
    base_model, lname = test1_model3(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model3(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model3(False, y_temp.shape[1])
    layername = 7
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model4"
    name = samplename+"_micro"
    base_model, lname = test1_model4(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model4(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model4(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    samplename = "test1_model5"
    name = samplename+"_micro"
    base_model, lname = test1_model5(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model5(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model5(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model6"
    name = samplename+"_micro"
    base_model, lname = test1_model6(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model6(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model6(False, y_temp.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model7"
    name = samplename+"_micro"
    base_model, lname = test1_model7(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model7(False, y_time.shape[1])
    layername = 72
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model7(False, y_temp.shape[1])
    layername = 76
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model8"
    name = samplename+"_micro"
    base_model, lname = test1_model8(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model8(False, y_time.shape[1])
    layername = 83
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model8(False, y_temp.shape[1])
    layername = 87
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model9"
    name = samplename+"_micro"
    base_model, lname = test1_model9(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model9(False, y_time.shape[1])
    layername = 94
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model9(False, y_temp.shape[1])
    layername = 98
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model10"
    name = samplename+"_micro"
    base_model, lname = test1_model10(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model10(False, y_time.shape[1])
    layername = 105
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model10(False, y_temp.shape[1])
    layername = 109
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model11"
    name = samplename+"_micro"
    base_model, lname = test1_model11(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model11(False, y_time.shape[1])
    layername = 116
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model11(False, y_temp.shape[1])
    layername = 120
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test1_model12"
    name = samplename+"_micro"
    base_model, lname = test1_model12(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test1_model12(False, y_time.shape[1])
    layername = 127
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test1_model12(False, y_temp.shape[1])
    layername = 131
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test2_model1"
    name = samplename+"_micro"
    base_model, lname = test2_model1(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test2_model1(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test2_model1(False, y_temp.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
   
   
    samplename = "test2_model2"
    name = samplename+"_micro"
    base_model, lname = test2_model2(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test2_model2(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test2_model2(False, y_temp.shape[1])
    layername = 7
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
   

    samplename = "test2_model3"
    name = samplename+"_micro"
    base_model, lname = test2_model3(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test2_model3(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test2_model3(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)


    samplename = "test2_model4"
    name = samplename+"_micro"
    base_model, lname = test2_model4(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test2_model4(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test2_model4(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)


    samplename = "test2_model5"
    name = samplename+"_micro"
    base_model, lname = test2_model5(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test2_model5(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test2_model5(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)


    samplename = "test2_model6"
    name = samplename+"_micro"
    base_model, lname = test2_model6(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test2_model6(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test2_model6(False, y_temp.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
     
    
    samplename = "test3_model1"
    name = samplename+"_micro"
    base_model, lname = test3_model1(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test3_model1(False, y_time.shape[1])
    layername = 204
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test3_model1(False, y_temp.shape[1])
    layername = 208
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test3_model2"
    name = samplename+"_micro"
    base_model, lname = test3_model2(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test3_model2(False, y_time.shape[1])
    layername = 215
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test3_model2(False, y_temp.shape[1])
    layername = 219
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test4_model1"
    name = samplename+"_micro"
    base_model, lname = test4_model1(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test4_model1(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test4_model1(False, y_temp.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test5_model1"
    name = samplename+"_micro"
    base_model, lname = test5_model1(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test5_model1(False, y_time.shape[1])
    layername = 235
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test5_model1(False, y_temp.shape[1])
    layername = 238
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test5_model2"
    name = samplename+"_micro"
    base_model, lname = test5_model2(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test5_model2(False, y_time.shape[1])
    layername = 243
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test5_model2(False, y_temp.shape[1])
    layername = 246
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test5_model3"
    name = samplename+"_micro"
    base_model, lname = test5_model3(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test5_model3(False, y_time.shape[1])
    layername = 6
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test5_model3(False, y_temp.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test5_model4"
    name = samplename+"_micro"
    base_model, lname = test5_model4(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test5_model4(False, y_time.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test5_model4(False, y_temp.shape[1])
    layername = 3
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test5_model5"
    name = samplename+"_micro"
    base_model, lname = test5_model5(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test5_model5(False, y_time.shape[1])
    layername = 10
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test5_model5(False, y_temp.shape[1])
    layername = 5
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    samplename = "test5_model6"
    name = samplename+"_micro"
    base_model, lname = test5_model6(True, y_micro.shape[1])
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    record_result(name, test_result, -20)
    name = samplename+"_time"
    base_model, lname = test5_model6(False, y_time.shape[1])
    layername = 5
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
    score = test_regression(name, model, x_data, y_time_reg, layername)
    record_result(name, test_result, score)
    name = samplename+"_temp"
    base_model, lname = test5_model6(False, y_temp.shape[1])
    layername = 5
    print(print_summary(base_model))
    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
    score = test_regression(name, model, x_data, y_temp_reg, layername)
    record_result(name, test_result, score)
    
    
    test6_regression_test()
    
    
    
    end = time.time()
    
    print("###########################################################################")
    print("###########################################################################")
    print("###########################################################################")
    print("###########################################################################")
    print("Time Taken: ", (end-start)/60)
