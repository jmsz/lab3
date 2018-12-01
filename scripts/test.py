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
from lab3_analysis_functions import *

def exp_func(x,a,t):
    return a*np.exp((-x/t))

def test_fitting():
    am_nrgs = read_nrg_sample_file('./data/am_energies.txt')
    am_counts, bin_edges = np.histogram(am_nrgs, bins=2048, range=[0, 400])
    am_bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges
    am_bins = am_bins[10:250]
    am_counts = am_counts[10:250]
    peak_fwhm, peak_center ,err = fit_gaussian_peak_linear_background(am_bins, am_counts)
    peak_res = round(peak_fwhm/peak_center, 4)
    print('resolution fitting Am-241 peak: ' + str(peak_res) + ' percent')
    am_counts, bin_edges = np.histogram(am_nrgs, bins=2048, range=[0, 400])
    am_bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges
    am_bins = am_bins[1000:2000]
    am_counts = am_counts[1000:2000]
    pulser_fwhm, pulser_center ,err = fit_gaussian_peak_linear_background(am_bins, am_counts)
    pulser_res = round(pulser_fwhm/pulser_center, 4)
    print('resolution fitting Am-241 pulser: ' + str(pulser_res) + ' percent')
    assert(pulser_res <= peak_res)

def test_baseline_correction():
    x = [100] * 800 + [200] * (4048-800)
    y = baseline_correction(x)
    assert(y[0] == 0)
    assert(y[200] == 0)
    assert(y[2000] == 100)

def test_fast_baseline_correction():
    x = [100] * 800 + [200] * (4048-800)
    y = baseline_correction(x)
    assert(y[0] == 0)
    assert(y[200] == 0)
    assert(y[2000] == 100)

def test_array_sorting():
    arr = np.zeros((5,3))
    #print(arr)
    arr[:,0] = [5, 4, 3, 2, 1]
    arr[:,1] = [20, 21, 27, 40, 5]
    arr[:,2] = arr[:,1]
    arr[0][2] = 2
    arr[3][2] = -1

    arr = arr[np.argsort(arr[:, 2])]

    assert(arr[0][0] == 2)
    assert(arr[0][1] == 40)
    assert(arr[0][2] == -1)

print('testing ...')
test_array_sorting()
print('testing complete')
