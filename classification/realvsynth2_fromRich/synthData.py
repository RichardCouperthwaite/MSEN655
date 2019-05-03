# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:10:02 2019

@author: richardcouperthwaite
"""

from numpy import genfromtxt

def get_synthData():
    Temp_orig_all = genfromtxt('Temp_orig_synthMatch', delimiter=' ')
    Time_orig_all = genfromtxt('Time_orig_synthMatch', delimiter=' ')
    Temp_synth_all = genfromtxt('Temp_synth_synthMatch', delimiter=' ')
    Time_synth_all = genfromtxt('Time_synth_synthMatch', delimiter=' ')
    xFTime_orig_all = genfromtxt('xFTime_orig_synthMatch', delimiter=' ')
    xFTemp_orig_all = genfromtxt('xFTemp_orig_synthMatch', delimiter=' ')
    xT1M2_orig_all = genfromtxt('xT1M2_orig_synthMatch', delimiter=' ')
    xT2M3_orig_all = genfromtxt('xT2M3_orig_synthMatch', delimiter=' ')
    xT5M4_orig_all = genfromtxt('xT5M4_orig_synthMatch', delimiter=' ')
    xFTime_synth_all = genfromtxt('xFTime_synth_synthMatch', delimiter=' ')
    xFTemp_synth_all = genfromtxt('xFTemp_synth_synthMatch', delimiter=' ')
    xT1M2_synth_all = genfromtxt('xT1M2_synth_synthMatch', delimiter=' ')
    xT2M3_synth_all = genfromtxt('xT2M3_synth_synthMatch', delimiter=' ')
    xT5M4_synth_all = genfromtxt('xT5M4_synth_synthMatch', delimiter=' ')
    
    Temp_synth_partial = genfromtxt('Temp_synth_0', delimiter=' ')
    Time_synth_partial = genfromtxt('Time_synth_0', delimiter=' ')
    xFTime_synth_0 = genfromtxt('xFTime_synth_0', delimiter=' ')
    xFTemp_synth_0 = genfromtxt('xFTemp_synth_0', delimiter=' ')
    xT1M2_synth_0 = genfromtxt('xT1M2_synth_0', delimiter=' ')
    xT2M3_synth_0 = genfromtxt('xT2M3_synth_0', delimiter=' ')
    xT5M4_synth_0 = genfromtxt('xT5M4_synth_0', delimiter=' ')
    
    xFTime_synth_1 = genfromtxt('xFTime_synth_1', delimiter=' ')
    xFTemp_synth_1 = genfromtxt('xFTemp_synth_1', delimiter=' ')
    xT1M2_synth_1 = genfromtxt('xT1M2_synth_1', delimiter=' ')
    xT2M3_synth_1 = genfromtxt('xT2M3_synth_1', delimiter=' ')
    xT5M4_synth_1 = genfromtxt('xT5M4_synth_1', delimiter=' ')
    
    xFTime_synth_2 = genfromtxt('xFTime_synth_2', delimiter=' ')
    xFTemp_synth_2 = genfromtxt('xFTemp_synth_2', delimiter=' ')
    xT1M2_synth_2 = genfromtxt('xT1M2_synth_2', delimiter=' ')
    xT2M3_synth_2 = genfromtxt('xT2M3_synth_2', delimiter=' ')
    xT5M4_synth_2 = genfromtxt('xT5M4_synth_2', delimiter=' ')
    
    xFTime_synth_3 = genfromtxt('xFTime_synth_3', delimiter=' ')
    xFTemp_synth_3 = genfromtxt('xFTemp_synth_3', delimiter=' ')
    xT1M2_synth_3 = genfromtxt('xT1M2_synth_3', delimiter=' ')
    xT2M3_synth_3 = genfromtxt('xT2M3_synth_3', delimiter=' ')
    xT5M4_synth_3 = genfromtxt('xT5M4_synth_3', delimiter=' ')
    
    xFTime_synth_4 = genfromtxt('xFTime_synth_4', delimiter=' ')
    xFTemp_synth_4 = genfromtxt('xFTemp_synth_4', delimiter=' ')
    xT1M2_synth_4 = genfromtxt('xT1M2_synth_4', delimiter=' ')
    xT2M3_synth_4 = genfromtxt('xT2M3_synth_4', delimiter=' ')
    xT5M4_synth_4 = genfromtxt('xT5M4_synth_4', delimiter=' ')
    
    return Temp_orig_all, Time_orig_all, Temp_synth_all, Time_synth_all, \
            xFTime_orig_all, xFTemp_orig_all, xT1M2_orig_all, xT2M3_orig_all, xT5M4_orig_all, \
            xFTime_synth_all, xFTemp_synth_all, xT1M2_synth_all, xT2M3_synth_all, xT5M4_synth_all,  \
            Temp_synth_partial, Time_synth_partial, \
            xFTime_synth_0, xFTemp_synth_0, xT1M2_synth_0, xT2M3_synth_0, xT5M4_synth_0, \
            xFTime_synth_1, xFTemp_synth_1, xT1M2_synth_1, xT2M3_synth_1, xT5M4_synth_1,  \
            xFTime_synth_2, xFTemp_synth_2, xT1M2_synth_2, xT2M3_synth_2, xT5M4_synth_2, \
            xFTime_synth_3, xFTemp_synth_3, xT1M2_synth_3, xT2M3_synth_3, xT5M4_synth_3, \
            xFTime_synth_4, xFTemp_synth_4, xT1M2_synth_4, xT2M3_synth_4, xT5M4_synth_4
            
Temp_orig_all = genfromtxt('Temp_orig_synthMatch.csv', delimiter=' ')
Time_orig_all = genfromtxt('Time_orig_synthMatch.csv', delimiter=' ')
Temp_synth_all = genfromtxt('Temp_synth_synthMatch.csv', delimiter=' ')
Time_synth_all = genfromtxt('Time_synth_synthMatch.csv', delimiter=' ')
xFTime_orig_all = genfromtxt('xFTime_orig_synthMatch.csv', delimiter=' ')
xFTemp_orig_all = genfromtxt('xFTemp_orig_synthMatch.csv', delimiter=' ')
xT1M2_orig_all = genfromtxt('xT1M2_orig_synthMatch.csv', delimiter=' ')
xT2M3_orig_all = genfromtxt('xT2M3_orig_synthMatch.csv', delimiter=' ')
xT5M4_orig_all = genfromtxt('xT5M4_orig_synthMatch.csv', delimiter=' ')
xFTime_synth_all = genfromtxt('xFTime_synth_synthMatch.csv', delimiter=' ')
xFTemp_synth_all = genfromtxt('xFTemp_synth_synthMatch.csv', delimiter=' ')
xT1M2_synth_all = genfromtxt('xT1M2_synth_synthMatch.csv', delimiter=' ')
xT2M3_synth_all = genfromtxt('xT2M3_synth_synthMatch.csv', delimiter=' ')
xT5M4_synth_all = genfromtxt('xT5M4_synth_synthMatch.csv', delimiter=' ')

Temp_synth_partial = genfromtxt('Temp_synth_0.csv', delimiter=' ')
Time_synth_partial = genfromtxt('Time_synth_0.csv', delimiter=' ')
xFTime_synth_0 = genfromtxt('xFTime_synth_0.csv', delimiter=' ')
xFTemp_synth_0 = genfromtxt('xFTemp_synth_0.csv', delimiter=' ')
xT1M2_synth_0 = genfromtxt('xT1M2_synth_0.csv', delimiter=' ')
xT2M3_synth_0 = genfromtxt('xT2M3_synth_0.csv', delimiter=' ')
xT5M4_synth_0 = genfromtxt('xT5M4_synth_0.csv', delimiter=' ')

xFTime_synth_1 = genfromtxt('xFTime_synth_1.csv', delimiter=' ')
xFTemp_synth_1 = genfromtxt('xFTemp_synth_1.csv', delimiter=' ')
xT1M2_synth_1 = genfromtxt('xT1M2_synth_1.csv', delimiter=' ')
xT2M3_synth_1 = genfromtxt('xT2M3_synth_1.csv', delimiter=' ')
xT5M4_synth_1 = genfromtxt('xT5M4_synth_1.csv', delimiter=' ')

xFTime_synth_2 = genfromtxt('xFTime_synth_2.csv', delimiter=' ')
xFTemp_synth_2 = genfromtxt('xFTemp_synth_2.csv', delimiter=' ')
xT1M2_synth_2 = genfromtxt('xT1M2_synth_2.csv', delimiter=' ')
xT2M3_synth_2 = genfromtxt('xT2M3_synth_2.csv', delimiter=' ')
xT5M4_synth_2 = genfromtxt('xT5M4_synth_2.csv', delimiter=' ')

xFTime_synth_3 = genfromtxt('xFTime_synth_3.csv', delimiter=' ')
xFTemp_synth_3 = genfromtxt('xFTemp_synth_3.csv', delimiter=' ')
xT1M2_synth_3 = genfromtxt('xT1M2_synth_3.csv', delimiter=' ')
xT2M3_synth_3 = genfromtxt('xT2M3_synth_3.csv', delimiter=' ')
xT5M4_synth_3 = genfromtxt('xT5M4_synth_3.csv', delimiter=' ')

xFTime_synth_4 = genfromtxt('xFTime_synth_4.csv', delimiter=' ')
xFTemp_synth_4 = genfromtxt('xFTemp_synth_4.csv', delimiter=' ')
xT1M2_synth_4 = genfromtxt('xT1M2_synth_4.csv', delimiter=' ')
xT2M3_synth_4 = genfromtxt('xT2M3_synth_4.csv', delimiter=' ')
xT5M4_synth_4 = genfromtxt('xT5M4_synth_4.csv', delimiter=' ')