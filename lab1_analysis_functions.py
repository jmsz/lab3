from __future__ import division, print_function
from numba import jit
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import tables
import sys
import csv
import time

@jit(parallel = True)
def baseline_correction(data):
    n_data = len(data)
    baseline_correction_value = np.mean(data[0:799])
    for i in range(0, n_data, 1):
        data[i] = data[i] - baseline_correction_value
    return data

def fast_baseline_correction(data):  # for one signal
    num=len(data)
    baseline = np.mean(data[:num,0:99],1).reshape((num,1))
    baseline_corrected = data[:num,] - baseline
    return(baseline_corrected)

def test_baseline_correction():
    data = []
    for i in range(0, 800, 1): # TODO NOT WORKING
        data.append(100)
    for i in range(800, 810, 1):
        data.append(500)
    data = baseline_correction(data)
    assert data[10] == 0

def pz_correction(data, tau):
    pz_corrected = []
    pz_correction = []
    for i in range(0, n_data, 1):
        pz = np.sum(data[1:i-1])
        pz_correction.append(pz)
        pz_corrected.append(data[i] + pz / tau)
    return pz_corrected

@jit(parallel = True)
def trapezoidal_filter(signal, m, k, M):
    sum_3 = np.zeros(len(signal), dtype=np.float32)
    sum_4 = np.zeros(len(signal), dtype=np.float32)
    acc_1 = np.zeros(len(signal), dtype=np.float32)
    acc_2 = np.zeros(len(signal), dtype=np.float32)

    for i in range(2 * k + m, len(signal)):  # following block diagram in Jordanov paper
        sum1 = signal[i] - signal[i - k]
        sum2 = signal[i - k - m - k] - signal[i - k - m]
        sum_3[i] = sum1 + sum2
        acc_1[i] = sum_3[i] + acc_1[i - 1] # S(n-1) + sum3(n)
        sum_4[i] = acc_1[i] + M * sum_3[i] #PZ correction?? same?
        acc_2[i] = sum_4[i] + acc_2[i-1]
    return acc_2

@jit(parallel = True)
def get_energy_value(data):
    energy = max(data)
    return energy

@jit(parallel = True)
def trapezoidal_filter_energy(signal, m, k, M):
    sum_3 = np.zeros(len(signal), dtype=np.float32)
    sum_4 = np.zeros(len(signal), dtype=np.float32)
    acc_1 = np.zeros(len(signal), dtype=np.float32)
    acc_2 = np.zeros(len(signal), dtype=np.float32)

    for i in range(2 * k + m, len(signal)):  # following block diagram in Jordanov paper
        sum1 = signal[i] - signal[i - k]
        sum2 = signal[i - k - m - k] - signal[i - k - m]
        sum_3[i] = sum1 + sum2
        acc_1[i] = sum_3[i] + acc_1[i - 1]
        sum_4[i] = acc_1[i] + M * sum_3[i]
        acc_2[i] = sum_4[i] + acc_2[i-1]
    nrg = round(get_energy_value(acc_2), 7)
    return nrg

def fast_trapezoidal_filter(data, k, m, M=4400):
    ndata = len(data)
    l = 2*k+m
    pad = np.pad(data, (l, 0), 'constant', constant_values=(0))
    sum3 = pad[l:ndata+l] - pad[l-k-m:ndata+l-k-m] - pad[l-k:ndata+l-k] + pad[:ndata]
    acc1 = np.cumsum(sum3)
    sum4 = acc1 + sum3 * M
    acc2 = np.cumsum(sum4)
    return acc2

def fast_trapezoidal_filter_energy(data, k, m, M=4400):
    ndata = len(data)
    l = 2*k+m
    pad = np.pad(data, (l, 0), 'constant', constant_values=(0))
    sum3 = pad[l:ndata+l] - pad[l-k-m:ndata+l-k-m] - pad[l-k:ndata+l-k] + pad[:ndata]
    acc1 = np.cumsum(sum3)
    sum4 = acc1 + sum3 * M
    acc2 = np.cumsum(sum4)
    nrg = round(acc2.max(), 7)
    return nrg

def get_time_values(length_of_data, sampling_time):
    time_values = np.linspace(0, length_of_data * sampling_time, length_of_data)
    return time_values

def fit_exponential(x, y):
    exp = lmfit.models.ExponentialModel(prefix='exp_')
    pars = exp.guess(y, x=x)
    mod = exp
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    return out.params

def fit_exponential_with_plot(x, y):
    exp = lmfit.models.ExponentialModel(prefix='exp_')
    pars = exp.guess(y, x=x)
    mod = exp
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    print(out.fit_report())
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, out.best_fit, 'r--')
    plt.show()
    return out.params

def fit_gaussian_with_plot(x, y):
    mod = lmfit.models.GaussianModel()
    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))
    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.show()
    fwhm = out.params['fwhm'].value
    center = out.params['center'].value
    return fwhm, center

def fit_gaussian(x,y):
    mod = lmfit.models.GaussianModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    fwhm = out.params['fwhm'].value
    center = out.params['center'].value
    return fwhm, center

def import_data(filename):
    # open file
    hf = tables.open_file(filename, "r")
    event_data = hf.root.EventData.read()
    raw_data = hf.root.RawData.read()
    hf.close()

    # mask retriggered events
    mask = (event_data['retrigger'] == 1)
    raw_data[mask] = 0

    return raw_data

def get_bin_centers_from_edges(edges):
    edges = np.array(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers

def find_M_value(signals): # use baseline corrected signal input
    tau_values = []
    for i in range(0, len(signals), 1):
        signal = signals[i,:]
        x = np.asarray(range(np.argmax(signal),len(signal), 1))
        y = np.asarray(signal[np.argmax(signal):len(signal)])
        out = fit_exponential(x, y)
        tau_values.append(out['exp_decay'].value)
    tau = np.mean(tau_values)
    return tau

def make_ten_event_file():
    f = open('tenevents.txt','w')
    for i in range(10):
        data = raw_data[i,:]
        for j in data:
            f.write(str(j) + ', ')
    f.close()

def read_ten_event_file():
    events = np.zeros((10, 4096))
    with open('tenevents.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            sample_count = 0
            for j in range(0, 4096, 1):
                #print(sample_count, float(row[j]))
                events[i][sample_count] = float(row[j])
                sample_count += 1
            i += 1
    return events

def find_peaks(data_x, data_y, low_index, high_index):
    cut_data_y = data_y[low_index:high_index]
    for i in range (low_index,high_index):
        if data_y[i] == max(cut_data_y):
            centerindex =  data_x[i]
            minindex=low_index
            maxindex=high_index
            amplitude = max(cut_data_y)
    ROIrange = [minindex, maxindex]
    return centerindex, amplitude, minindex, maxindex

def calculate_coefficients(calibration_channels, calibration_energies, polynomial_order):
    return np.polyfit(calibration_channels, calibration_energies, polynomial_order)

def calculate_energies(x, m, b):
    energies = m * x + b
    return energies

def get_energy_resolution_co_1173(x, y):
    ROI_low =0
    ROI_high = int(2048 / 2)
    fwhm, center = fit_gaussian(x[ROI_low:ROI_high], y[ROI_low:ROI_high])

    return fwhm, center

def get_energy_resolution_co_1332(x, y):
    ROI_low = int(2048 / 2)
    ROI_high =  2048
    fwhm, center =  fit_gaussian(x[ROI_low:ROI_high], y[ROI_low:ROI_high])

    return fwhm, center

def get_energy_resolution_cs_662(x, y):
    ROI_low = 0
    ROI_high =  int(2048/2)
    i = np.argmax(y[ROI_low:ROI_high])
    ROI_low = i - 200
    ROI_high =  i + 200
    fwhm, center =  fit_gaussian_with_plot(x[ROI_low:ROI_high], y[ROI_low:ROI_high])
    return fwhm, center

def get_energy_resolution_cs_pulser(x, y):
    ROI_low = int(2048/2)
    ROI_high =  int(2048)
    i = np.argmax(y[ROI_low:ROI_high])
    ROI_low = i - 200
    ROI_high =  i + 200
    fwhm, center =  fit_gaussian_with_plot(x[ROI_low:ROI_high], y[ROI_low:ROI_high])
    return fwhm, center

def calibrate_co_spectrum(nrgs):# Get first calibration point, ROI HARDCODED
    nbins = 2048
    counts, bin_edges = np.histogram(nrgs, bins=2048, range=[2.3e8, 2.9e8]) # WORKS
    bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges
    x = bins
    y = counts

    fwhm1, center1 = get_energy_resolution_co_1173(x, y)
    fwhm2, center2 = get_energy_resolution_co_1332(x, y)

    # from above fits:
    Co_energies = [1173.2, 1332.5] # MeV
    Co_indices = [center1, center2]
    print (Co_indices)

    f = calculate_coefficients(Co_indices, Co_energies, 1)
    print(f)
    m = f[0]
    b = f[1]
    energies = calculate_energies(x, m, b)

    return energies
