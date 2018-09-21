# Import pytables
from __future__ import division
from math import floor as floor
import tables
import matplotlib.pyplot as plt
import numpy as np
import sys

# Open the data file in 'read' mode
hf = tables.open_file("./data/DATA.h5", "r")
event_data = hf.root.EventData.read()
raw_data = hf.root.RawData.read()
hf.close()
# event_data

# BASELINE CORRECTION
def baseline_correction(data):
    n_data = len(data)
    baseline_correction_value = np.mean(data[0:99])
    bc_corrected_signal = []
    for i in range(0, n_data, 1):
        bc_corrected_signal.append(data[i] - baseline_correction_value )
    return bc_corrected_signal

def test_baseline_correction():
    data = []
    for i in range(0, 100, 1):
        data.append(100)
    for i in range(0, 100, 1):
        data.append(200)
    data = baseline_correction(data)
    assert data[10] == 0
    assert data[140] == 100

test_baseline_correction()

# POLE-ZERO CORRECTION
def pz_correction(data, tau):
    pz_correction = [0]
    pz_corrected = [0]
    for i in range(1, n_data, 1):
        #pz = np.sum(bc_corrected_signal[50:(i-1)])
        pz = np.sum(pz_corrected[i - 1] + data[i] - data[i - 1] + data[i - 1] / tau)
        #pz = np.sum(bc_corrected_signal[1:i - 1])
        pz_correction.append(pz)
        pz_corrected.append(data[i] + pz / tau)
    #    pz_corrected.append(data[i])
        #Tr_prime.append(pz)
    return pz_corrected

# TRAPEZOIDAL FILTER

def trapezoidal_filter(data, gap_time, peaking_time):
    filtered_signal = [0]*4096
    #print(filtered_signal[0:10])
    for j in range(k + 1, k + l, 1):
        sig = np.sum(data[j - k: j])
        filtered_signal[j] = sig
    for j in range(k + l + 1, n_data, 1):
        sig = np.sum(data[j - k: j]) - np.sum(data[j - l - k: j - l])
       # if j%1000 == 0:
       #     plt.plot(data)
       #     plt.plot(np.linspace(j-k, j, k), data[j - k: j], 'ob')
       #     plt.plot(np.linspace(j- l - k, j - l, k), data[j - l - k: j - l], 'or')
       #     plt.show()
        filtered_signal[j] = sig
    #print(filtered_signal[0:10])
    #print(filtered_signal[-10:-1])
    return filtered_signal

def get_energy_value(data):
    energy = max(data)
    return energy

def get_time_values(length_of_data, sampling_time):
    time_values = np.linspace(0, length_of_data * sampling_time, length_of_data)
    return time_values
n_data = 4096
x_values = np.linspace(0, n_data * 1, n_data)
x_values

#pz_correction = [0]
#Tr_prime = [0]

#for i in range(1, n_data, 1):
    #pz = np.sum(bc_corrected_signal[50:(i-1)])
#    pz = 0
#    for k in range(1, i, 1):
#        pz += np.sum(bc_corrected_signal[i] - bc_corrected_signal[i - 1] + bc_corrected_signal[i - 1] / tau)

#    pz_correction.append(pz)
    #Tr_prime.append(bc_corrected_signal[i] + pz / tau)

#print(len(pz_correction))
#print(Tr_prime[0:10])

sampling_time = 10.0
k = 100 # peaking time
l = 500 # peaking time + gap? gap?
tau = 30000
raw_signal = raw_data[2,:]
print(np.mean(raw_signal))
signal = baseline_correction(raw_signal)
plt.plot(signal, label='base')
print("=***")
print(np.mean(signal))
signal = pz_correction(signal, tau)
print("=**")
print(np.mean(signal))
plt.plot(signal, 'o', label ='pz')
plt.legend()
plt.show()
signal = trapezoidal_filter(signal, l, k)
print("=")
print(np.mean(signal))
plt.cla()
plt.clf()
plt.plot(x_values, signal)
plt.show()

sys.exit()

plt.plot(x_values[950:1100], signal[950:1100], label ='trap')
plt.plot(1060, signal[1060], 'o')
plt.show()

energy = get_energy_value(signal)
print(energy)

def filter_and_get_energy(data, peak, gap):
    signal = data
    #print(np.mean(signal))
    signal = baseline_correction(signal)
    #print("=***")
    #print(np.mean(signal))
    t = 100
    signal = pz_correction(signal, t)
    #plt.plot(signal)
    #print("=**")
    #print(np.mean(signal))
    #plt.show()
    signal = trapezoidal_filter(signal, gap, peak)
    #print("=")
    #print(np.mean(signal))
    #plt.cla()
    #plt.clf()
    #plt.plot(x_values, signal)

    #plt.plot(x_values[900:1100], signal[900:1100], label ='trap')
    #plt.plot(1060, signal[1060], 'o')
    #plt.show()

    energy_value = get_energy_value(signal)
    #print(energy_value)
    return energy_value

nrgs = []
number_of_events = len(raw_data)
number_of_events = 500
k = 100 # peaking time
l = 50 # gap
#R = 100 * 10**6
#C = 10 * 10**(-12)
#tau = R * C

for i in range(0, number_of_events, 1):
   nrg = filter_and_get_energy(raw_data[i,:], k, l)
   #print(raw_data[i,1060:1070])
   #print(nrg)
   nrgs.append(round(nrg, 7))

print(max(nrgs))
print(min(nrgs))

plt.hist(nrgs, 500)
plt.xlim([0, 200000])
plt.show()
