from __future__ import division, print_function
from numba import jit
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import tables
import sys

@jit(parallel = True)
def baseline_correction(data):
    n_data = len(data)
    baseline_correction_value = np.mean(data[0:799])
    for i in range(0, n_data, 1):
        data[i] = data[i] - baseline_correction_value
    return data

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

# def trapezoidal_filter(data, gap_time, peaking_time):
#     m = gap_time
#     k = peaking_time
#     l = peaking_time + gap_time
#     filtered_signal = [0]*4096
#     for j in range(k + 1, k + m, 1):
#         sig = np.sum(data[j - k: j])
#         filtered_signal[j] = sig
#     for j in range(k + l + 1, n_data, 1):
#         sig = np.sum(data[j - k: j]) - np.sum(data[j - l - k: j - l])
#         filtered_signal[j] = sig
#     return filtered_signal

def find_M_value(data, num):
    tau_values = []
    for i in range(0, num, 1):
        signal = raw_data[2,:]
        x = np.asarray(range(np.argmax(signal), n_data, 1))
        y = np.asarray(signal[np.argmax(signal):n_data])
        out = FitExponential(x, y)
        tau_values.append(out['exp_decay'].value)
    tau = np.mean(tau_values)
    return tau

@jit(parallel = True)
def trapezoidal_filter(signal, m, k):
    M = 4400
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
def get_energy_value(data): # TODO
    energy = max(data)
    return energy

def get_time_values(length_of_data, sampling_time):
    time_values = np.linspace(0, length_of_data * sampling_time, length_of_data)
    return time_values

def FitExponential(x, y):
    exp = lmfit.models.ExponentialModel(prefix='exp_')
    pars = exp.guess(y, x=x)
    mod = exp
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    # print(out.fit_report())
    # plt.figure()
    # plt.plot(x, y, 'o')
    # plt.plot(x, out.best_fit, 'r--')
    # plt.show()
    return out.params

def filter_and_get_energy(signal, peak, gap):
    signal = baseline_correction(signal)
    signal = trapezoidal_filter(signal, gap, peak)
    energy_value = get_energy_value(signal)
    energy_value = round(energy_value, 7)
    #energy_value = energy_value / 440000
    return energy_value

def FitGaussian(x, y):
    mod = lmfit.models.GaussianModel()
    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))
    plt.figure(figsize=(20, 10))
    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.show()

def FitGaussianPeakLinearBackground(x, y, peak_amplitude, peak_centroid, peak_sigma):

    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    gauss1 = lmfit.models.GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    pars['g1_center'].set((peak_centroid), min=(peak_centroid - 200), max=(peak_centroid + 200))
    pars['g1_sigma'].set(peak_sigma, min=0.1)
    pars['g1_amplitude'].set(peak_amplitude, min=10)

    #gauss2  = GaussianModel(prefix='g2_')
    #pars.update(gauss2.make_params())
    #pars['g2_center'].set(155, min=125, max=175)
    #pars['g2_sigma'].set(15, min=3)
    #pars['g2_amplitude'].set(2000, min=10)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    #print("fwhm:")
    fit_fwhm = (out.params['g1_fwhm'].value)
    fit_fwhm_err = (out.params['g1_fwhm'].stderr)
    fit_center = (out.params['g1_center'].value)
    fit_sigma = (out.params['g1_sigma'].value)
    fit_height = (out.params['g1_height'].value)
    print("=========FIT HEIGHT=========")
    print(fit_height)
    fit_amplitude = (out.params['g1_amplitude'].value)
    chisqr = out.chisqr
    redchi = out.redchi

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10, 5))

    plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2 * out.params['g1_fwhm'].value), 0, (1.2 * out.params['g1_height'].value)])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, comps['g1_'], 'b--')
    #plt.plot(x, comps['g2_'], 'b--')
    plt.plot(x, comps['lin_'], 'g--')

    myfile = open('testfile.txt', 'w')
    myfile.write(out.fit_report())
    myfile.close()

    #   plt.plot(x, y, 'bo')
    #   plt.plot(x, out.init_fit, 'k--')
    #   plt.plot(x, out.best_fit, 'r-')
    plt.show()

    return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude, chisqr, redchi

def ImportData(filename):
    # open file
    hf = tables.open_file(filename, "r")
    event_data = hf.root.EventData.read()
    raw_data = hf.root.RawData.read()
    hf.close()
    h5file = tables.open_file("sample.h5", mode="w")

    # mask retriggered events
    mask = (event_data['retrigger'] == 1)
    raw_data[mask] = 0
    return raw_data

if __name__ == "__main__":

    # filename = 'data/co60.h5'
    filename = 'data/cs137_co60.h5'
    raw_data = ImportData(filename)
    print('len ', len(raw_data))
    #sys.exit()
    sampling_time = 10.0
    n_data = 4096
    x_values = np.linspace(0, n_data, n_data)

    peaking_time = 100 # peaking
    gap = 100 # gap
    raw_signal = raw_data[2,:]
    M = find_M_value(raw_data, 100)
    print('M= ', M)
    print("RAW ", len(raw_signal))
    plt.plot(x_values, raw_signal)
    signal = baseline_correction(raw_signal)
    print(signal)
    print(gap)
    print(peaking_time)
    signal = trapezoidal_filter(signal, gap, peaking_time)
    plt.figure(41)
    plt.cla()
    plt.clf()
    plt.plot(x_values, signal)
    plt.show()

    energy = get_energy_value(signal)
    print(energy)

    nrgs = []
    number_of_events = len(raw_data)
    #number_of_events = 10000
    assert number_of_events <= len(raw_data)
    k = 100 # peaking time
    l = 100 # gap

    for i in range(0, number_of_events, 1):
        signal = raw_data[i,:]
        nrg = filter_and_get_energy(signal, l, k)
        nrgs.append(round(nrg, 7))

    print('max ', max(nrgs))
    print('min ', min(nrgs))

    #histrange = [1.0e8, 3.5e8]
    nbins = 2048
    #histrange = np.linspace(histrange[0], histrange[1], nbins)
    #bins = GetBinCentersFromEdges(histrange)
    counts, bin_edges = np.histogram(nrgs, bins=2048, range=[0.1e9,1e9]) #, range=[1.4e8, 3.06e8])
    bins = (bin_edges[1:]+bin_edges[:-1])/2
    #return bin_centers, counts
    plt.figure()
    plt.plot(bins, counts)
    #plt.hist(nrgs, 500, histrange)
    plt.title('Cs Spectrum')
    plt.ylabel('counts')
    plt.xlabel('channel')
    plt.savefig('cs_spectrum_pulser.pdf')
    plt.show()
    sys.exit()
    numberofROIs = int(raw_input('how many ROIs?'))
    print (numberofROIs)

    if numberofROIs < 1:
        print('no ROIs. Exiting...')
        sys.exit()
    for i in range (0, (numberofROIs), 1):
        ROI_low, ROI_high = selectROI()
        centroidguess, amplitudeguess, ROIrangemin, ROIrangemax = findpeaks(spec.energies_kev, spec.counts_vals, ROI_low, ROI_high)
        plot_spectrum_raw(spec.energies_kev[ROI_low:ROI_high], spec.counts_vals[ROI_low:ROI_high])

        FitGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1)

    sys.exit()

def apply_calibration(self, cal):
    n_edges = len(self.channels) + 1
    channel_edges = np.linspace(-0.5, float(self.channels[-1] + 0.5), num=n_edges)
    self.bin_edges_kev = cal.ch2kev(channel_edges)
    return self.slope * ch + self.offset

def calculate_coefficients(calibration_channels, calibration_energies, polynomial_order):
    return np.polyfit(calibration_channels, calibration_energies, polynomial_order)

def calculate_energies(x, m, b):
    energies = m * x + b
    return energies

def GetBinCentersFromEdges(edges):
    edges = np.array(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers

def selectROI():
    lowindex = raw_input("ROI low index (channel #)")
    #lowindex = 8000
    if lowindex == '':
        lowindex = 0
    highindex = raw_input("ROI high index (channel #)")
    #highindex = 11000
    if highindex == '':
        highindex = spec.channels(max)
    lowindex = int(lowindex)
    highindex = int(highindex)
    return lowindex, highindex

def FitGaussian(x,y):
    plt.figure(figsize=(20,10))

    mod = lmfit.models.GaussianModel()

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.show()

# Switch to Jupyter Notebook
# Import Data
# Make a special data file with only 10 events or so
# Plot the raw signals from those 10 events
# Plot resulting Trapezoids
# Plot energy spectrum of those 10
# figure out calibration
# Plot the full, calibrated energy spectra for Co, Cs, Am (to check for re-takes)
# Fit peaks and get FWHM Values
# Plot FWHM vs gap time
# Plot FWHM vs peak time
# Set optimal peak and gap times and save spectrum
# Make electronic noise plot
# calculate FANO factor
# Write text
# What else?

# REDO DATA Cs = low stats, Co maybe okay, Am/ Co peaks?/ pulser?
