from __future__ import division, print_function
from numba import jit
from math import floor
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from lmfit import Model, CompositeModel
import lmfit
from lmfit.lineshapes import step, gaussian
from operator import truediv
import matplotlib.pyplot as plt
import numpy as np
#import lmfit
import tables
import sys
import csv
import time


def calculate_signal_rise_time_interpolation_TEST(signal, plot=False):
  signal = signal[910:1070]
  x = np.linspace(0, 1070-910, 1070-910)
  sig = savgol_filter(signal, 25, 3) # window size 51, polynomial order 3

  if sig.all() == 0:
    return -1, -1
  elif np.argmax(sig) == len(sig):
    return -1, -1
  else:
    grad = np.amax(np.gradient(sig)) / 10

    maxval = np.amax(sig)
    tenval = maxval* 0.10
    ninetyval = maxval * 0.9
    tenindex = 0
    ninetyindex = 0

    for i in range(0, np.argmax(sig), 1):
        if sig[i] <= tenval:
            tenindex = i
    for i in range(tenindex, len(sig), 1):
        if sig[i] >= ninetyval:
            ninetyindex = i
            break

    x_fit_low = x[int(tenindex - 1): int(tenindex + 2)]
    sig_fit_low = sig[int(tenindex - 1): int(tenindex + 2)]
    x_fit_low = np.array(x_fit_low)
    sig_fit_low = np.array(sig_fit_low)
    x_fit_low = np.array(x_fit_low)
    sig_fit_low = np.array(sig_fit_low)
    if len(x_fit_low) < 1:
        print('x empty')
        #plt.plot(signal)
        #plt.plot(sig)
        #plt.show()
        return -1,-1
    else:
       # print(x_fit_low)
       # print(sig_fit_low)
        m, b = np.polyfit(x_fit_low, sig_fit_low, deg=1)
        fit_low = b + m * x_fit_low
        rise_low = ((tenval - b )/ m)

    x_fit_high = x[ninetyindex - 1 : ninetyindex + 2]
    sig_fit_high = sig[ninetyindex - 1: ninetyindex + 2]
    x_fit_high = np.array(x_fit_high)
    sig_fit_high = np.array(sig_fit_high)
    x_fit_high = np.array(x_fit_high)
    sig_fit_high = np.array(sig_fit_high)
    if len(x_fit_high) < 1:
        print('x emp')
        #plt.plot(signal)
        #plt.plot(sig)
        #plt.show()
        return -1, -1
    else:
 #       print(x_fit_high)
 #       print(sig_fit_high)
        m, b = np.polyfit(x_fit_high, sig_fit_high, deg=1)
        fit_high = b + m * x_fit_high
        rise_high = ((ninetyval - b) / m)

    risetime = (rise_high - rise_low) * 10# ns
    #print('fit')
    #print(rise_high)
    #print(rise_low)
    #print(risetime)
    #print('basic')
    #print(ninetyindex)
    #print(tenindex)
    #print(ninetyindex - tenindex)
    if plot==True:
        plt.figure(figsize=(10,5))
        plt.plot(signal, '-')
        plt.plot(sig)
        plt.plot(x_fit_high, fit_high,'o')
        plt.plot(x_fit_low, fit_low,'o')
        plt.show()
    return risetime, grad


def calculate_t50(signal, plot=False):
  signal = signal[910:1070]
  x = np.linspace(0, 1070-910, 1070-910)
  sig = savgol_filter(signal, 25, 3) # window size 51, polynomial order 3

  if sig.all() == 0:
    return -1, -1
  elif np.argmax(sig) == len(sig):
    return -1, -1
  else:
    grad = np.amax(np.gradient(sig)) / 10

    maxval = np.amax(sig)
    fiftyval = maxval* 0.5

    for i in range(0, np.argmax(sig), 1):
        if sig[i] <= fiftyval:
            fiftyindex = i

    x_fit_low = x[int(fiftyindex - 3): int(fiftyindex + 3)]
    sig_fit_low = sig[int(fiftyindex - 3): int(fiftyindex + 3)]
    x_fit_low = np.array(x_fit_low)
    sig_fit_low = np.array(sig_fit_low)
    if len(x_fit_low) < 1:
        print('x empty')
        #plt.plot(signal)
        #plt.plot(sig)
        #plt.show()
        return -1,-1
    else:
       # print(x_fit_low)
       # print(sig_fit_low)
        m, b = np.polyfit(x_fit_low, sig_fit_low, deg=1)
        fit_low = b + m * x_fit_low
        rise_low = ((tenval - b )/ m)

    t50 = (rise_low) * 10# ns
    #print('fit')
    #print(rise_high)
    #print(rise_low)
    #print(risetime)
    #print('basic')
    #print(ninetyindex)
    #print(tenindex)
    #print(ninetyindex - tenindex)
    if plot==True:
        plt.figure(figsize=(10,5))
        plt.plot(signal, '-')
        plt.plot(sig)
        plt.plot(x_fit_low, fit_low,'o')
        plt.show()
    return t50

def calculate_signal_rise_time_interpolation(signal, plot=False):
    signal = signal[910:1090]
    x = np.linspace(0, 1090-910, 1090-910)
    sig = savgol_filter(signal, 15, 3) # window size 51, polynomial order 3

    maxval = np.amax(sig)
    tenval = maxval* 0.10
    ninetyval = maxval * 0.9
    tenindex = 0
    ninetyindex = 0

    for i in range(0, np.argmax(sig), 1):
        if sig[i] <= tenval:
            tenindex = i
    for i in range(tenindex, len(sig), 1):
        if sig[i] >= ninetyval:
            ninetyindex = i
            break

    x_fit_low = x[int(tenindex - 1): int(tenindex + 2)]
    sig_fit_low = sig[int(tenindex - 1): int(tenindex + 2)]
    m, b = np.polyfit(x_fit_low, sig_fit_low, deg=1)
    fit_low = b + m * x_fit_low
    rise_low = ((tenval - b )/ m)

    x_fit_high = x[ninetyindex - 1 : ninetyindex + 2]
    sig_fit_high = sig[ninetyindex - 1: ninetyindex + 2]
    m, b = np.polyfit(x_fit_high, sig_fit_high, deg=1)
    fit_high = b + m * x_fit_high
    rise_high = ((ninetyval - b) / m)

    risetime = (rise_high - rise_low) # ns

    maxgrad = np.amax(np.gradient(sig))

    print(ninetyindex - tenindex)
    if plot==True:
        plt.figure(figsize=(10,5))
        plt.cla()
        plt.clf()
        plt.plot(signal, '-')
        plt.plot(sig)
        plt.plot(x_fit_high, fit_high,'o')
        plt.plot(x_fit_low, fit_low,'o')
        plt.savefig('../figures/smoothing.pdf')
        #plt.show()
    return risetime, maxgrad


def calculate_coefficients(calibration_channels, calibration_energies, polynomial_order):
    return np.polyfit(calibration_channels, calibration_energies, polynomial_order)

def gaussianfunction(x, h, mu, sigma):
    return h*np.exp(-((x-mu)/sigma)**2)

def calculate_energies(x, m, b):
    energies = m * x + b
    return energies

def baseline_correction(data):
    n_data = len(data)
    baseline_correction_value = np.mean(data[0:15])
    for i in range(0, n_data, 1):
        data[i] = data[i] - baseline_correction_value
    return data

def fast_baseline_correction(data):  # for one signal
    num=len(data)
    baseline = np.mean(data[:num,0:15],1).reshape((num,1))
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
        acc_2[i] = acc_2[i] / ((M+1)*k)
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
    nrg = round(get_energy_value(acc_2) / ((M+1)*k), 7)
    if nrg >= 2048:
        nrg = 0
    return nrg

def fast_trapezoidal_filter(data, k, m, M=4400):
    ndata = len(data)
    l = 2*k+m
    pad = np.pad(data, (l, 0), 'constant', constant_values=(0))
    sum3 = pad[l:ndata+l] - pad[l-k-m:ndata+l-k-m] - pad[l-k:ndata+l-k] + pad[:ndata]
    acc1 = np.cumsum(sum3)
    sum4 = acc1 + sum3 * M
    acc2 = np.cumsum(sum4)
    acc2 = acc2 / ((M+1)*k)
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
    nrg = nrg / ((M+1)*k)
    if nrg >= 2048:
        nrg = 0
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

def get_tau_exponential(x, y):
    tau_values = []
    out = fit_exponential_with_plot(x, y)
    if out['exp_decay'].value < 1e4 and out['exp_decay'].stderr < 100:
        tau_values.append(out['exp_decay'].value)
    tau = np.mean(tau_values)
    return tau


def fit_gaussian(x,y):
    mod = lmfit.models.GaussianModel()
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    fwhm = out.params['fwhm'].value
    center = out.params['center'].value
    amp = out.params['amplitude'].value
    return fwhm, center, amp

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
        if out['exp_decay'].value < 1e4 and out['exp_decay'].stderr < 100:
            tau_values.append(out['exp_decay'].value)
    tau = np.mean(tau_values)
    return tau

def make_ten_event_file():
    f = open('../data/tenevents.txt','w')
    for i in range(10):
        data = raw_data[i,:]
        for j in data:
            f.write(str(j) + ', ')
    f.close()

def read_ten_event_file():
    events = np.zeros((10, 4096))
    with open('../data/tenevents.txt') as csv_file:
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

def make_nrg_sample_file(nrgs, name):
    filename = str(name)
    f = open(name,'w')
    num = len(nrgs)
    j = 0
    while j < num:
        for i in nrgs:
            f.write(str(i) + '\n')
            j += 1
    f.close()

def read_nrg_sample_file(name):
    values = []
    values = np.loadtxt(str(name), dtype='float', delimiter=' ')
    return values

def save_parameters(m, k, M):
    filename = './figures/gap.dat'
    f = open(filename,'w')
    f.write(str(m))
    f.close()

    filename = './figures/peak.dat'
    f = open(filename,'w')
    f.write(str(k))
    f.close()

    filename = './figures/decay.dat'
    f = open(filename,'w')
    f.write(str(M))
    f.close()

def save_fano(f, err):
    filename = './figures/fano.dat'
    f = open(filename,'w')
    f.write(str(f))
    f.close()

    filename = './figures/fano_err.dat'
    f = open(filename,'w')
    f.write(str(err))
    f.close()

def fit_gaussian_peak_linear_background_wide(x,y):
    peak_amplitude = max(y)
    peak_centroid = x[np.argmax(y)] - 5000
    peak_sigma = 8000
    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)
    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)
    gauss1  = lmfit.models.GaussianModel(prefix='g1_')
    pars.update( gauss1.make_params())
    pars['g1_center'].set((peak_centroid), min=(peak_centroid-100), max=(peak_centroid+100))
    pars['g1_sigma'].set(peak_sigma, min=5000)
    pars['g1_amplitude'].set(peak_amplitude, min=10)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    comps = out.eval_components(x=x)
    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_center = (out.params['g1_center'].value)
    fwhm_err = (out.params['g1_fwhm'].stderr )
    #print(out.fit_report(min_correl=0.5))
    #plt.figure()
    #ax = plt.gca()
    #plt.plot(x, y, 'bo')
    #plt.plot(x, out.init_fit, 'k--')
    #plt.plot(x, out.best_fit, 'r-')
    #plt.plot(x, comps['g1_'], 'b--')
    #plt.plot(x, comps['lin_'], 'g--')
    #plt.show()
    amp = out.params['g1_amplitude'].value
    return fit_fwhm, fit_center, fwhm_err, amp

def fit_gaussian_peak_linear_background(x,y,plot=False):
    peak_amplitude = max(y)
    peak_centroid = x[np.argmax(y)]
    peak_sigma = 1
    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)
    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)
    gauss1  = lmfit.models.GaussianModel(prefix='g1_')
    pars.update( gauss1.make_params())
    pars['g1_center'].set((peak_centroid), min=(peak_centroid-100), max=(peak_centroid+100))
    pars['g1_sigma'].set(peak_sigma, min=0.5)
    pars['g1_amplitude'].set(peak_amplitude, min=10)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    comps = out.eval_components(x=x)
    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_center = (out.params['g1_center'].value)
    fwhm_err = (out.params['g1_fwhm'].stderr )

    if plot==True:
        print(out.fit_report(min_correl=0.5))
        plt.figure()
        ax = plt.gca()
        plt.plot(x, y, 'bo')
        plt.plot(x, out.init_fit, 'k--')
        plt.plot(x, out.best_fit, 'r-')
        plt.plot(x, comps['g1_'], 'b--')
        plt.plot(x, comps['lin_'], 'g--')
        plt.show()
    amp = out.params['g1_amplitude'].value
    return fit_fwhm, fit_center, fwhm_err, amp

def fit_gaussian_peak_step_background(x,y):
    peak_amplitude = max(y)
    peak_centroid = x[np.argmax(y)]
    peak_sigma = 1
    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)
    bkg_mod = lmfit.models.StepModel(prefix='step_')
    # pars = bkg_mod.guess(y)
    pars = bkg_mod.guess(y, x=x, center=6e4)
    # pars = bkg_mod.guess(y, x=x)
    pars.update(bkg_mod.make_params())
    pars['step_center'].set(20)
    pars['step_sigma'].set(50)
    pars['step_amplitude'].set(10)
    # pars['step_center'].set(peak_centroid)
    # pars['step_sigma'].set(peak_sigma, min=0.1)
    # pars['step_amplitude'].set(peak_amplitude, min=5)
    gauss1  = lmfit.models.GaussianModel(prefix='g1_')
    pars.update( gauss1.make_params())
    pars['g1_center'].set((peak_centroid), min=(peak_centroid-100), max=(peak_centroid+100))
    pars['g1_sigma'].set(peak_sigma, min=0.1)
    pars['g1_amplitude'].set(peak_amplitude, min=10)
    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    comps = out.eval_components(x=x)
    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_center = (out.params['g1_center'].value)
    fwhm_err = (out.params['g1_fwhm'].stderr )
    print(out.fit_report(min_correl=0.5))
    plt.figure()
    ax = plt.gca()
    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'ro')
    plt.plot(x, comps['g1_'], 'b--')
    plt.plot(x, comps['step_'], 'g--')
    plt.show()
    return fit_fwhm, fit_center, fwhm_err


from lmfit import Model, CompositeModel
from lmfit.lineshapes import step, gaussian

def jump(x, mid):
    "heaviside step function"
    o = np.zeros(len(x))
    imid = max(np.where(x<=mid)[0])
    o[imid:] = 1.0
    return o

def convolve(arr, kernel):
    # simple convolution of two arrays
    npts = min(len(arr), len(kernel))
    pad  = np.ones(npts)
    tmp  = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out  = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts)/2)
    return out[noff:noff+npts]

def fit_gaussian_peak_step_background_2(x,y):
    # create data from broadened step
    npts = 201
    x = np.linspace(0, 10, npts)
    y = step(x, amplitude=12.5, center=4.5, sigma=0.88, form='erf')
    y = y + np.random.normal(size=npts, scale=0.35)
    # create Composite Model using the custom convolution operator
    mod  = CompositeModel(Model(jump), Model(gaussian), convolve)
    pars = mod.make_params(amplitude=1, center=3.5, sigma=1.5, mid=5.0)
    # 'mid' and 'center' should be completely correlated, and 'mid' is
    # used as an integer index, so a very poor fit variable:
    pars['mid'].vary = False

    # fit this model to data array y
    result =  mod.fit(y, params=pars, x=x)

    print(result.fit_report())

    plot_components = True

    # plot results
    plt.plot(x, y,'bo')
    if plot_components:
        # generate components
        comps = result.eval_components(x=x)
        plt.plot(x, 10*comps['jump'], 'k--')
        plt.plot(x, 10*comps['gaussian'], 'r-')
    else:
        plt.plot(x, result.init_fit, 'k--')
        plt.plot(x, result.best_fit, 'r-')
    plt.show()
    # #<end examples/model_doc3.py>

    return fit_fwhm, fit_center, fwhm_err

def make_test_file(masked_data, rawdata):
    f = open('test.csv', 'w')
    for i in event_data_cs1[mask_1]:
        #print(i)
        f.write(str(i[0]) +','+ str(i[1]) + ','+str(i[2]) + ','+ str(i[3]) + ','+ str(i[4]) + ','+ str(i[5]) + ','+ str(i[6]) + '\n')
    f.close()

    f = open('test_trace.csv', 'w')
    for i in event_data_cs1['rid'][mask_1]:
        x = rawdata[i]
        for j in x:
            f.write(str(j) + ' ')
        f.write('\n')
    f.close()
