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
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

def correlation_matrix_plot(filename, data):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cmap_name = 'my_list'
    df = pd.DataFrame(data)
    fig, axs = plt.subplots(1,1, figsize=(6,6))
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    im = axs.matshow(df.corr(), vmin=-1, vmax=1, interpolation='nearest', origin='lower', cmap=cm)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.grid(which='both')
    fig.colorbar(im, ax=axs)
    fig.tight_layout()
    fig.savefig("results/plots/corr_plot_{}.png".format(filename))
    plt.close(fig)
    

def principle_component(filename, data):
    pca = PCA(n_components=4)
    pca.fit(data)
    with open("results/PCA_results.txt", 'a') as f:
        out = pca.explained_variance_ratio_
        f.write("{} & {} & {} & {} & {} \\\\ \n".format(filename, round(out[0],2), round(out[1],2), round(out[2],2), round(out[3],2)))

def get_data(name):
    f = h5py.File(name, 'r')
    x_train = np.array(f["train_features"]).astype('float64')
    y_train = np.array(f["train_output"]).astype('float64').transpose()
    x_test = np.array(f["test_features"]).astype('float64')
    y_test = np.array(f["test_output"]).astype('float64').transpose()
    
    return x_train, x_test, y_train, y_test

def test6_regression_test():
    filenames = ['test1_model1_temp.hdf5', 'test1_model2_temp.hdf5', 'test1_model3_temp.hdf5',
                 'test1_model4_temp.hdf5', 'test1_model5_temp.hdf5', 'test1_model6_temp.hdf5',
                 'test1_model7_temp.hdf5', 'test1_model8_temp.hdf5', 'test1_model9_temp.hdf5',
                 'test1_model10_temp.hdf5', 'test1_model11_temp.hdf5', 'test1_model12_temp.hdf5',
                 'test2_model1_temp.hdf5', 'test2_model2_temp.hdf5', 'test2_model3_temp.hdf5',
                 'test2_model4_temp.hdf5', 'test2_model5_temp.hdf5', 'test2_model6_temp.hdf5',
                 'test3_model1_temp.hdf5', 'test3_model2_temp.hdf5', 'test4_model1_temp.hdf5',
                 'test5_model1_temp.hdf5', 'test5_model2_temp.hdf5', 'test5_model3_temp.hdf5',
                 'test5_model4_temp.hdf5', 'test5_model5_temp.hdf5', 'test5_model6_temp.hdf5',
                 'test1_model1_time.hdf5', 'test1_model2_time.hdf5', 'test1_model3_time.hdf5',
                 'test1_model4_time.hdf5', 'test1_model5_time.hdf5', 'test1_model6_time.hdf5',
                 'test1_model7_time.hdf5', 'test1_model8_time.hdf5', 'test1_model9_time.hdf5',
                 'test1_model10_time.hdf5', 'test1_model11_time.hdf5', 'test1_model12_time.hdf5',
                 'test2_model1_time.hdf5', 'test2_model2_time.hdf5', 'test2_model3_time.hdf5',
                 'test2_model4_time.hdf5', 'test2_model5_time.hdf5', 'test2_model6_time.hdf5',
                 'test3_model1_time.hdf5', 'test3_model2_time.hdf5', 'test4_model1_time.hdf5',
                 'test5_model1_time.hdf5', 'test5_model2_time.hdf5', 'test5_model3_time.hdf5',
                 'test5_model4_time.hdf5', 'test5_model5_time.hdf5', 'test5_model6_time.hdf5']
    for name in filenames:
        print(name)
        x_train, x_test, y_train, y_test = get_data("results/"+name)
        print(x_train.shape)
        print(y_train.shape)
        
        correlation_matrix_plot(name, x_train)
        principle_component(name, x_train)
        
        lasso_reg = Lasso(alpha=0.1)
        lasso_reg.fit(x_train, y_train)
        score2 = lasso_reg.score(x_test, y_test)
        
        # svr_reg = SVR(gamma='scale', C=1.0, epsilon=0.2)
        # svr_reg.fit(x_train, y_train)
        # score3 = svr_reg.score(x_test, y_test)
        score3 = -20
        
        rf_reg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        rf_reg.fit(x_train, y_train)
        score4 = rf_reg.score(x_test, y_test)
        
        with open('results/regressiontest.txt', 'a') as f:
            f.write("{}, {}, {}, {} \n".format(name, score2, score3, score4))

if __name__ == "__main__":
    #run some code
    pass