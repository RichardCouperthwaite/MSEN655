# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:00:51 2019

@author: richardcouperthwaite
"""

import csv
import numpy as np

temps = [0, 800.0, 970.0, 1100.0, 900.0, 1000.0, 700.0, 750.0]
times = [0, 90.0, 180.0, 1440.0, 5100.0, 60.0, 5.0, 480.0, 2880.0]

labels = []
with open('labels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        labels.append(row[0].split('-')[0])
      
synthlabels = []
with open('synthlabels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x = row[0].split('_')[0]
        synthlabels.append(x.split('o')[1])
        
mags = []
with open('mags.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        mags.append(float(row[0]))

time = {}
rowcounter = 0
with open('y_time_reg.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        time[labels[rowcounter]] = times[int(float(row[0]))]
        rowcounter += 1
        
temp = {}
rowcounter = 0
with open('y_temp_reg.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        temp[labels[rowcounter]] = temps[int(float(row[0]))]
        rowcounter += 1
       
labelsset = set(labels)
synthlabelsset = set(synthlabels)       

#Original Data
x_out_Final_temp_opt = []
rowcounter = 0
with open('x_out_Final_temp_opt.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_Final_temp_opt.append(row)
        
x_out_Final_time_opt = []
rowcounter = 0
with open('x_out_Final_time_opt.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_Final_time_opt.append(row)
        
x_out_Optimization_T1_M2_time = []
rowcounter = 0
with open('x_out_Optimization_T1_M2_time.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_Optimization_T1_M2_time.append(row)
        
x_out_Optimization_T2_M3_temp = []
rowcounter = 0
with open('x_out_Optimization_T2_M3_temp.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_Optimization_T2_M3_temp.append(row)
        
x_out_Optimization_T5_M4_time = []
rowcounter = 0
with open('x_out_Optimization_T5_M4_time.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_Optimization_T5_M4_time.append(row)
 
#Synthetic Data       
x_out_synth_Final_temp_opt = []
rowcounter = 0
with open('x_out_synth_Final_temp_opt.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_synth_Final_temp_opt.append(row)
        
x_out_synth_Final_time_opt = []
rowcounter = 0
with open('x_out_synth_Final_time_opt.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_synth_Final_time_opt.append(row)
        
x_out_synth_Optimization_T1_M2_time = []
rowcounter = 0
with open('x_out_synth_Optimization_T1_M2_time.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_synth_Optimization_T1_M2_time.append(row)
        
x_out_synth_Optimization_T2_M3_temp = []
rowcounter = 0
with open('x_out_synth_Optimization_T2_M3_temp.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_synth_Optimization_T2_M3_temp.append(row)
        
x_out_synth_Optimization_T5_M4_time = []
rowcounter = 0
with open('x_out_synth_Optimization_T5_M4_time.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_out_synth_Optimization_T5_M4_time.append(row)
        
        
        
synthindexes = []
synthtemps = []
synthtimes = []
origtimes = []
origtemps = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []

sx1 = []
sx2 = []
sx3 = []
sx4 = []
sx5 = []

for i in range(len(synthlabels)):
    try:
        synthtemps.append(temp[synthlabels[i]])
        synthtimes.append(time[synthlabels[i]])
        sx1.append(x_out_synth_Final_temp_opt[i])
        sx2.append(x_out_synth_Final_time_opt[i])
        sx3.append(x_out_synth_Optimization_T1_M2_time[i])
        sx4.append(x_out_synth_Optimization_T2_M3_temp[i])
        sx5.append(x_out_synth_Optimization_T5_M4_time[i])
    except KeyError:
        pass
for i in range(len(labels)):
    if labels[i] in synthlabelsset:
        origtemps.append(temp[labels[i]])
        origtimes.append(time[labels[i]])
        x1.append(x_out_Final_temp_opt[i])
        x2.append(x_out_Final_time_opt[i])
        x3.append(x_out_Optimization_T1_M2_time[i])
        x4.append(x_out_Optimization_T2_M3_temp[i])
        x5.append(x_out_Optimization_T5_M4_time[i])
        
        
npsynthTemp = np.array(synthtemps)
npsynthtime = np.array(synthtimes)
nporigtime = np.array(origtimes)
nporigtemp = np.array(origtemps)
x1 = np.array(x1, dtype=np.float64)
x2 = np.array(x2, dtype=np.float64)
x3 = np.array(x3, dtype=np.float64)
x4 = np.array(x4, dtype=np.float64)
x5 = np.array(x5, dtype=np.float64)
sx1 = np.array(sx1, dtype=np.float64)
sx2= np.array(sx2, dtype=np.float64)
sx3 = np.array(sx3, dtype=np.float64)
sx4 = np.array(sx4, dtype=np.float64)
sx5 = np.array(sx5, dtype=np.float64)

np.savetxt('Temp_synth_synthMatch.csv', npsynthTemp)
np.savetxt('Time_synth_synthMatch.csv', npsynthtime)
np.savetxt('Time_orig_synthMatch.csv', nporigtime)
np.savetxt('Temp_orig_synthMatch.csv', nporigtemp)
np.savetxt('xFTemp_orig_synthMatch.csv', x1)
np.savetxt('xFTime_orig_synthMatch.csv', x2)
np.savetxt('xT1M2_orig_synthMatch.csv', x3)
np.savetxt('xT2M3_orig_synthMatch.csv', x4)
np.savetxt('xT5M4_orig_synthMatch.csv', x5)
np.savetxt('xFTemp_synth_synthMatch.csv', sx1)
np.savetxt('xFTime_synth_synthMatch.csv', sx2)
np.savetxt('xT1M2_synth_synthMatch.csv', sx3)
np.savetxt('xT2M3_synth_synthMatch.csv', sx4)
np.savetxt('xT5M4_synth_synthMatch.csv', sx5)

npsynthTemp = np.array(synthtemps)
npsynthtime = np.array(synthtimes)
nporigtime = np.array(origtimes)
nporigtemp = np.array(origtemps)

np.savetxt('npsynthTemp.csv', npsynthTemp)
np.savetxt('npsynthtime.csv', npsynthtime)
np.savetxt('nporigtime.csv', nporigtime)
np.savetxt('nporigtemp.csv', nporigtemp)

for i in range(5):
    count = 0
    npsx1 = []
    npsx2 = []
    npsx3 = []
    npsx4 = []
    npsx5 = []
    sortedsynthTemp = []
    sortedsynthTime = []
    
    for j in range(len(sx1)):
        if count == i:
            sortedsynthTemp.append(npsynthTemp[j])
            sortedsynthTime.append(npsynthtime[j])
            npsx1.append(sx1[j])
            npsx2.append(sx2[j])
            npsx3.append(sx3[j])
            npsx4.append(sx4[j])
            npsx5.append(sx5[j])
            print('match')
        count += 1
        if count == 5:
            count = 0
    
    sortedsynthTemp = np.array(sortedsynthTemp)
    sortedsynthTime = np.array(sortedsynthTime)
    npsx1 = np.array(npsx1, dtype=np.float64)
    npsx2 = np.array(npsx2, dtype=np.float64)
    npsx3 = np.array(npsx3, dtype=np.float64)
    npsx4 = np.array(npsx4, dtype=np.float64)
    npsx5 = np.array(npsx5, dtype=np.float64)
    
    np.savetxt('Temp_synth_{}.csv'.format(i), sortedsynthTemp)
    np.savetxt('Time_synth_{}.csv'.format(i), sortedsynthTime)
    np.savetxt('xFTemp_synth_{}.csv'.format(i), npsx1)
    np.savetxt('xFTime_synth_{}.csv'.format(i), npsx2)
    np.savetxt('xT1M2_synth_{}.csv'.format(i), npsx3)
    np.savetxt('xT2M3_synth_{}.csv'.format(i), npsx4)
    np.savetxt('xT5M4_synth_{}.csv'.format(i), npsx5)
    
highmagx1 = []
highmagx2 = []
highmagx3 = []
highmagx4 = []
highmagx5 = []
highmagytemp = []
highmagytime = []

lowmagx1 = []
lowmagx2 = []
lowmagx3 = []
lowmagx4 = []
lowmagx5 = []
lowmagytemp = []
lowmagytime = []
    
for i in range(len(mags)):
    if mags[i] > 1.5:
        highmagx1.append(x_out_Final_temp_opt[i])
        highmagx2.append(x_out_Final_time_opt[i])
        highmagx3.append(x_out_Optimization_T1_M2_time[i])
        highmagx4.append(x_out_Optimization_T2_M3_temp[i])
        highmagx5.append(x_out_Optimization_T5_M4_time[i])
        highmagytemp.append(temp[labels[i]])
        highmagytime.append(time[labels[i]])
    if mags[i] < 0.05:
        lowmagx1.append(x_out_Final_temp_opt[i])
        lowmagx2.append(x_out_Final_time_opt[i])
        lowmagx3.append(x_out_Optimization_T1_M2_time[i])
        lowmagx4.append(x_out_Optimization_T2_M3_temp[i])
        lowmagx5.append(x_out_Optimization_T5_M4_time[i])
        lowmagytemp.append(temp[labels[i]])
        lowmagytime.append(time[labels[i]])
        
nphighmagx1 = np.array(highmagx1, dtype=np.float64)
nphighmagx2 = np.array(highmagx2, dtype=np.float64)
nphighmagx3 = np.array(highmagx3, dtype=np.float64)
nphighmagx4 = np.array(highmagx4, dtype=np.float64)
nphighmagx5 = np.array(highmagx5, dtype=np.float64)
nphighmagytemp = np.array(highmagytemp, dtype=np.float64)
nphighmagytime = np.array(highmagytime, dtype=np.float64)

nplowmagx1 = np.array(lowmagx1, dtype=np.float64)
nplowmagx2 = np.array(lowmagx2, dtype=np.float64)
nplowmagx3 = np.array(lowmagx3, dtype=np.float64)
nplowmagx4 = np.array(lowmagx4, dtype=np.float64)
nplowmagx5 = np.array(lowmagx5, dtype=np.float64)
nplowmagytemp = np.array(lowmagytemp, dtype=np.float64)
nplowmagytime = np.array(lowmagytime, dtype=np.float64)

np.savetxt('highmagx1.csv', nphighmagx1)
np.savetxt('highmagx2.csv', nphighmagx2)
np.savetxt('highmagx3.csv', nphighmagx3)
np.savetxt('highmagx4.csv', nphighmagx4)
np.savetxt('highmagx5.csv', nphighmagx5)
np.savetxt('highmagxtemp.csv', nphighmagytemp)
np.savetxt('highmagxtime.csv', nphighmagytime)

np.savetxt('lowmagx1.csv', nplowmagx1)
np.savetxt('lowmagx2.csv', nplowmagx2)
np.savetxt('lowmagx3.csv', nplowmagx3)
np.savetxt('lowmagx4.csv', nplowmagx4)
np.savetxt('lowmagx5.csv', nplowmagx5)
np.savetxt('lowmagxtemp.csv', nplowmagytemp)
np.savetxt('lowmagxtime.csv', nplowmagytime)


