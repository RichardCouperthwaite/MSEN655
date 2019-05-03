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
from TestFinal import final
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
    # load a single micrograph by using the img path and do the necessary
    # preprocessing required to ensure that it is in the correct format for
    # input into the CNN (require size is 224x224x3)
    set_image_dim_ordering('tf')
    img = load_img(img_path)
    x = img_to_array(img)[0:224,0:224,:]
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
                    mag.append(cropsData[label]["Effective_magnification"])
                    lab.append([label,])
                except NameError:
                    inputs = new_img
                    y_micro = [micros.index(cropsData[label]['Primary_Microconstituent'])]
                    y_cool = [cool.index(cropsData[label]['Cool Method'])]
                    y_time =[times.index(cropsData[label]['Anneal Time'])]
                    y_temp = [temp.index(cropsData[label]['Anneal Temperature'])]
                    mag = [cropsData[label]["Effective_magnification"]]
                    lab = [[label,]]
            else:
                count += 1
        index += 1
#        stop loop before all samples run to speed up testing
#        if index == 100:
#            break
        
    end = time.time()
    print("\nTime Taken (min): ", round((end-start)/60,2))
    # convert the outputs to categorical values for use in the training
    y_micro = to_categorical(np.array(y_micro))
    y_cool = to_categorical(np.array(y_cool))
    # keep a copy of the time and temperature as actual values for use in regression
    # This records the index of the temperature and time in the lists above
    y_time_reg = np.array(y_time)
    y_time = to_categorical(np.array(y_time))
    y_temp_reg = np.array(y_temp)
    y_temp = to_categorical(np.array(y_temp))
    
    return inputs, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp, mag, lab

def import_from_hdf5():
    # to speed up importing it is possible to save all the information data to an
    # hdf5 file. This function imports the required data from the hdf5 file
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
    # function to find and return the list of possible values for the microconstituent
    # cooling method, annealing time, and annealing temperature 
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
    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num, shuffle=True)
    # obtain the test results
    test_result = model.evaluate(x_test, y_test, batch_size=20)
#    print(test_result)
    return model, test_result

def test_regression(name, base_model, x_data, y_data, layername):
    """
    This function takes the input data, along with a trained CNN (base_model) and can
    return the output from any given dense layer in the network.
    The code then performs a GPR on the output from that layer using an RBF kernel
    """
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, 0.1)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("dense_{}".format(layername)).output)
    reg_features = model.predict(x_train)
    gpr = GaussianProcessRegressor(kernel=RBF(), random_state=0).fit(reg_features, y_train)
    test_features = model.predict(x_test)
    # save the output from the CNN for both training and testing datasets in order
    # to make it easier to repeat the results
    f = h5py.File("results/{}.hdf5".format(name), 'w')
    f.create_dataset("train_features", data=reg_features)
    f.create_dataset("train_output", data=y_train)
    f.create_dataset("test_features", data=test_features)
    f.create_dataset("test_output", data=y_test)
    return(gpr.score(test_features, y_test))
    
def record_result(name, class_result, score):
    """
    Function records the classification and regression result and outputs it in 
    csv format as well as a text format that can be directly copied into a LATEX table
    """
    with open('results/results_TEX_output.txt', 'a') as f:
        f.write("{} & {} & {} & {} \\\\ \n".format(name, round(class_result[0],2), round(class_result[1],2), round(score,2)))
    with open('results/results_csv_output.csv', 'a') as f:
        f.write("{}, {}, {}, {} \n".format(name, class_result[0], class_result[1], score))

        
def model_analysis():
    """
    Function Handles the running of the various models. The output from the 20 neuron
    layer needs to be corrected for the models if selected models are being run
    """

    # retrieve the data for the testing
    x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp = import_data()
    
    # data has also been formatted into an hdf5 file for easy importing
    # only use if the hdf5 file is present
    # x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp = import_from_hdf5()
    epochs = 80


    """
    It is noted that when running this code, it was not possible to run all the structures
    in series like this. As a result, the layername number is not necessarily the layer
    number that would be required if the models were run in series.
    However, the number provided in this code is the number that is required if the model is 
    run and tested separately.
    Each model has three tests, classification with microconstituent, and regression
    with annealing temperature and time.
    The current version of the code has commented all lines of code except that for running the
    Test1_Model1 classification of microconstituent model.
    Comments in the code indicate which lines to uncomment/comment in order to change
    which model is being tested.
    A line of ********* indicates the top of the model test
    A line of ~~~~~~~~~ indicates the bottom of the model test
    
    
    
    
    ***It is suggested that the file is opened in Spyder and ctrl+1 is used to comment or uncomment lines***
    
    
    
    """
    # Required Line for Test1_Model1
    samplename = "test1_model1"
    #*************************************************************************#
    # define the sample name
    name = samplename+"_micro"
    # retrieve the base model structure
    base_model, lname = test1_model1(True, y_micro.shape[1])
    # display a summary of the CNN structure
    print(print_summary(base_model))
    # train the CNN and then test to get a classification score
    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
    # record the final results
    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model1(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model1(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    # Required Line for Test1_Model2
    samplename = "test1_model2"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model2(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model2(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model2(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    
    # Required Line for Test1_Model3
    samplename = "test1_model3"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model3(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model3(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model3(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model4
    samplename = "test1_model4"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model4(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model4(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model4(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    # Required Line for Test1_Model5
    samplename = "test1_model5"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model5(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model5(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model5(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model6
    samplename = "test1_model6"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model6(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model6(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model6(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model7
    samplename = "test1_model7"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model7(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model7(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model7(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model8
    samplename = "test1_model8"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model8(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model8(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model8(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model9
    samplename = "test1_model9"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model9(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model9(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model9(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model10
    samplename = "test1_model10"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model10(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model10(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model10(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model11
    samplename = "test1_model11"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model11(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model11(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model11(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test1_Model12
    samplename = "test1_model12"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test1_model12(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test1_model12(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test1_model12(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test2_Model1
    samplename = "test2_model1"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test2_model1(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test2_model1(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test2_model1(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
   
   
    # Required Line for Test2_Model2
    samplename = "test2_model2"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test2_model2(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test2_model2(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test2_model2(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
   

    # Required Line for Test2_Model3
    samplename = "test2_model3"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test2_model3(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test2_model3(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test2_model3(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # Required Line for Test2_Model4
    samplename = "test2_model4"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test2_model4(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test2_model4(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test2_model4(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # Required Line for Test2_Model4
    samplename = "test2_model5"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test2_model5(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test2_model5(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test2_model5(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # Required Line for Test2_Model6
    samplename = "test2_model6"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test2_model6(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test2_model6(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test2_model6(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
     
    
    # Required Line for Test3_Model1
    samplename = "test3_model1"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test3_model1(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test3_model1(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test3_model1(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test3_Model2
    samplename = "test3_model2"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test3_model2(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test3_model2(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test3_model2(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test4_Model1
    samplename = "test4_model1"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test4_model1(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test4_model1(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test4_model1(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test5_Model1
    samplename = "test5_model1"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test5_model1(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test5_model1(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 2
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test5_model1(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 2
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test5_Model2
    samplename = "test5_model2"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test5_model2(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test5_model2(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 2
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test5_model2(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 2
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test5_Model3
    samplename = "test5_model3"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test5_model3(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test5_model3(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test5_model3(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test5_Model4
    samplename = "test5_model4"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test5_model4(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test5_model4(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test5_model4(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 3
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test5_Model5
    samplename = "test5_model5"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test5_model5(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test5_model5(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 5
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test5_model5(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 5
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
    # Required Line for Test5_Model6
    samplename = "test5_model6"
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_micro"
#    # retrieve the base model structure
#    base_model, lname = test5_model6(True, y_micro.shape[1])
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_micro, epochs)
#    # record the final results
#    record_result(name, test_result, -20)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_time"
#    # retrieve the base model structure
#    base_model, lname = test5_model6(False, y_time.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 5
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_time, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_time_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #*************************************************************************#
#    # define the sample name
#    name = samplename+"_temp"
#    # retrieve the base model structure
#    base_model, lname = test5_model6(False, y_temp.shape[1])
#    # define the layer number for the penultimate layer
#    layername = 5
#    # display a summary of the CNN structure
#    print(print_summary(base_model))
#    # train the CNN and then test to get a classification score
#    model, test_result = train_and_test_model(base_model, x_data, y_temp, epochs)
#    # obtain the regression result from a GPR
#    score = test_regression(name, model, x_data, y_temp_reg, layername)
#    # record the final results
#    record_result(name, test_result, score)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
def train_final_model(model, x_data, y_data, epoch_num=500, batch_num=20):
    # Use a stochastic gradient descent optimizer, and train the model
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    model.fit(x_data, y_data, epochs=epoch_num, batch_size=batch_num, shuffle=True)
    return model

def Final_Model_output(base_model, x_data, layername):
    # function to optain the outputs from the final model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("dense_{}".format(layername)).output)
    features = model.predict(x_data)
    return features

def get_synth_images():
    """
    A small selection of synthetic images have been obtained and this code is used to
    obtain the data. The directory where the files are stored will need to be updated
    if this is not run on the original PC
    
    To aid in this, an hdf5 file of the image data was created. This hdf5 file can be used
    to obtain the image data.
    """
    
    # code to store synthetic image data in an h5 file
#    import os
#    files = os.listdir("C:/Users/richardcouperthwaite/Documents/GitHub/MSEN655/classification/data/Cropped_images")
#    print(files)
#    hf = h5py.File('SynthData.hdf5', 'w')
#    for file in files:
#        x = load_micrograph("C:/Users/richardcouperthwaite/Documents/GitHub/MSEN655/classification/data/Cropped_images/"+file)
#        hf.create_dataset(file, data=x)
    
    # code to import the synthetic images from an h5 file
    hf = h5py.File('SynthData.hdf5', 'r')
    for key in hf.keys():
        x = hf.get(key)
        try:
            labels.append([key])
            x_input = np.r_[x_input, x]
        except NameError:
            labels = [[key]]
            x_input = x
    
    return labels, x_input
    
def finalTests():
    # retrieve the data for the testing
    # data has also been formatted into an hdf5 file for easy importing
#    x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp = import_data()
    
#    x_data, y_micro, y_cool, y_time_reg, y_time, y_temp_reg, y_temp, mags, labels = import_data()
#    synthlabels, x_data_synth = get_synth_images()
#    hf = h5py.File('inputdata.hdf5', 'w')
#    hf.create_dataset('OriginalData', data=x_data)
#    hf.create_dataset('SynthData', data=x_data_synth)
#    hf.create_dataset('y_time', data=y_time)
#    hf.create_dataset('y_temp', data=y_temp)
#    
#    np.savetxt('finalTest/y_micro.csv', y_micro)
#    np.savetxt('finalTest/y_cool.csv', y_cool)
#    np.savetxt('finalTest/y_time_reg.csv', y_time_reg)
#    np.savetxt('finalTest/y_time.csv', y_time)
#    np.savetxt('finalTest/y_temp_reg.csv', y_temp_reg)
#    np.savetxt('finalTest/y_temp.csv', y_temp)
#    np.savetxt('finalTest/mags.csv', mags)
#    
#    import csv
#    with open('finalTest/labels.csv', 'w') as csvFile:
#        writer = csv.writer(csvFile)
#        writer.writerows(labels)
#    with open('finalTest/synthlabels.csv', 'w') as csvFile:
#        writer = csv.writer(csvFile)
#        writer.writerows(synthlabels)
#    csvFile.close()
    
    hf = h5py.File('inputdata.hdf5', 'r')
    x_data = np.array(hf.get('OriginalData'))
    x_data_synth = np.array(hf.get('SynthData'))
    y_time = np.array(hf.get('y_time'))
    y_temp = np.array(hf.get('y_temp'))
    
    epochs = 80
    setnum = 5
    
    if setnum == 1:
        #***************************************************#
        # Required Line for Test1_Model1
        samplename = "Final"
        # define the sample name
        name = samplename+"_time_opt"
        # retrieve the base model structure
        base_model, lname = final(False, y_time.shape[1])
        # define the layer number for the penultimate layer
        layername = 3
        # display a summary of the CNN structure
        print(print_summary(base_model))
        model = train_final_model(base_model, x_data, y_time, epochs)
        feat = Final_Model_output(model, x_data, layername)
        synthfeat = Final_Model_output(model, x_data_synth, layername)
        np.savetxt('finalTest/x_out_{}.csv'.format(name), feat)
        np.savetxt('finalTest/x_out_synth_{}.csv'.format(name), synthfeat)
        print('done')
        #***************************************************#
    elif setnum == 2:
        #***************************************************#
        # Required Line for Test1_Model1
        samplename = "Final"
        # define the sample name
        name = samplename+"_temp_opt"
        # retrieve the base model structure
        base_model, lname = final(False, y_temp.shape[1])
        # define the layer number for the penultimate layer
        layername = 3
        # display a summary of the CNN structure
        print(print_summary(base_model))
        model = train_final_model(base_model, x_data, y_temp, epochs)
        feat = Final_Model_output(model, x_data, layername)
        synthfeat = Final_Model_output(model, x_data_synth, layername)
        np.savetxt('finalTest/x_out_{}.csv'.format(name), feat)
        np.savetxt('finalTest/x_out_synth_{}.csv'.format(name), synthfeat)
        #***************************************************#
    elif setnum == 3:
        #***************************************************#
        # Required Line for Test1_Model1
        samplename = "Optimization"
        # define the sample name
        name = samplename+"_T1_M2_time"
        # retrieve the base model structure
        base_model, lname = test1_model2(False, y_time.shape[1])
        # define the layer number for the penultimate layer
        layername = 3
        # display a summary of the CNN structure
        print(print_summary(base_model))
        model = train_final_model(base_model, x_data, y_time, epochs)
        feat = Final_Model_output(model, x_data, layername)
        synthfeat = Final_Model_output(model, x_data_synth, layername)
        np.savetxt('finalTest/x_out_{}.csv'.format(name), feat)
        np.savetxt('finalTest/x_out_synth_{}.csv'.format(name), synthfeat)
        #***************************************************#
    elif setnum == 4:
        #***************************************************#
        # Required Line for Test1_Model1
        samplename = "Optimization"
        # define the sample name
        name = samplename+"_T5_M4_time"
        # retrieve the base model structure
        base_model, lname = test5_model4(False, y_time.shape[1])
        # define the layer number for the penultimate layer
        layername = 3
        # display a summary of the CNN structure
        print(print_summary(base_model))
        model = train_final_model(base_model, x_data, y_time, epochs)
        feat = Final_Model_output(model, x_data, layername)
        synthfeat = Final_Model_output(model, x_data_synth, layername)
        np.savetxt('finalTest/x_out_{}.csv'.format(name), feat)
        np.savetxt('finalTest/x_out_synth_{}.csv'.format(name), synthfeat)
        #***************************************************#
    elif setnum == 5:
        #***************************************************#
        # Required Line for Test1_Model1
        samplename = "Optimization"
        # define the sample name
        name = samplename+"_T2_M3_temp"
        # retrieve the base model structure
        base_model, lname = test2_model3(False, y_temp.shape[1])
        # define the layer number for the penultimate layer
        layername = 3
        # display a summary of the CNN structure
        print(print_summary(base_model))
        model = train_final_model(base_model, x_data, y_temp, epochs)
        feat = Final_Model_output(model, x_data, layername)
        synthfeat = Final_Model_output(model, x_data_synth, layername)
        np.savetxt('finalTest/x_out_{}.csv'.format(name), feat)
        np.savetxt('finalTest/x_out_synth_{}.csv'.format(name), synthfeat)
        #***************************************************#
    

        
    

if __name__ == "__main__":
    # Run the analysis of the models to get the prediction accuracy and the results
    # from using a GP regression
    model_analysis()
    
    # Run the regression tests to get the correlations, PCA results and the 
    # Regression with different methods
    # this requires the hdf5 files from each of the structures
    test6_regression_test()
    
    # Run the testing of the final tests to obtain the activations for both the original
    # and synthetic images from the final structure as well as 3 of the other structures   
    # see report for more details
    finalTests()
    