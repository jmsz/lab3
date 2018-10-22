from __future__ import division, print_function
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import tables
import sys
import csv
import time
from lab2_analysis_functions import *

def calculate_signal_rise_time(signal, plot=False):
    signal = signal[900:1090]
    sig = np.convolve(signal, np.ones((30,))/30, mode='valid')
    maxval = np.amax(sig)
    tenval = maxval* 0.10
    ninetyval = maxval * 0.9

    tenindex = 0
    ninetyindex = 0
    for i in range(0, np.argmax(signal), 1):
        if signal[i] <= tenval:
            tenindex = i
    for i in range(tenindex, len(signal), 1):
        if signal[i] >= ninetyval:
            ninetyindex = i
            break

    risetime = (ninetyindex - tenindex) * 10
    if plot==True:
        #print(risetime)
        #plt.plot(x[minindex - 2: maxindex + 2], signal[minindex - 2: maxindex + 2])
        plt.plot(signal)
        plt.plot(sig)
        #plt.plot(x[190:200], fit)
        plt.plot(ninetyindex, ninetyval, 'o')
        plt.plot(tenindex, tenval, 'o')
        plt.show()
    return risetime

filename = '/home/anp/Desktop/lab2/data/Cs_70cm_2.h5'
hf = tables.open_file(filename, "r")
cs_raw_data_2 = import_data(filename)
cs_event_data_2 = hf.root.EventData.read()
hf.close()
mask_2 = (cs_event_data_2['retrigger'] == 1)
cs_raw_data_2[mask_2] = 0

filename = '/home/anp/Desktop/lab2/data/Cs_70cm.h5'
hf = tables.open_file(filename, "r")
cs_raw_data_0 = import_data(filename)
cs_event_data_0 = hf.root.EventData.read()
hf.close()
mask_0 = (cs_event_data['retrigger'] == 1)
cs_raw_data_0[mask_0] = 0

filename = '/home/anp/Desktop/lab2/data/Cs_70cm_3.h5'
hf = tables.open_file(filename, "r")
cs_raw_data_3 = import_data(filename)
cs_event_data_3 = hf.root.EventData.read()
hf.close()
mask_3 = (cs_event_data['retrigger'] == 1)
cs_raw_data_3[mask_3] = 0

f = open('../data/risetimes_0.txt','w')
cs_risetimes= []
i = 0
print(len(cs_raw_data_0))
for sig in cs_raw_data_0:
    sig = baseline_correction(sig)
    #sig = sig[800:1100]
    dt = calculate_signal_rise_time(sig)
    if dt > 20 and dt < 1500:
        cs_risetimes.append(dt)
        E = cs_event_data['ADC_value'][i]
        f.write(str(E) + ', ' + str(dt) + '\n')
    i += 1
f.close()

f = open('../data/risetimes_1.txt','w')
cs_risetimes= []
i = 0
print(len(cs_raw_data_1))
for sig in cs_raw_data_1:
    sig = baseline_correction(sig)
    #sig = sig[800:1100]
    dt = calculate_signal_rise_time(sig)
    if dt > 20 and dt < 1500:
        cs_risetimes.append(dt)
        E = cs_event_data['ADC_value'][i]
        f.write(str(E) + ', ' + str(dt) + '\n')
    i += 1
f.close()

f = open('../data/risetimes_2.txt','w')
cs_risetimes= []
i = 0
print(len(cs_raw_data_2))
for sig in cs_raw_data_2:
    sig = baseline_correction(sig)
    #sig = sig[800:1100]
    dt = calculate_signal_rise_time(sig)
    if dt > 20 and dt < 1500:
        cs_risetimes.append(dt)
        E = cs_event_data['ADC_value'][i]
        f.write(str(E) + ', ' + str(dt) + '\n')
    i += 1
f.close()
