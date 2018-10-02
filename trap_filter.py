# Import pytables
from __future__ import division
from math import floor
from lmfit.models import LinearModel
from lmfit.models import ExponentialModel
from lmfit.models import GaussianModel
import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.optimize as opt

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

def exp_func(x,a,b):
    return a*np.exp(-x/b)
def fit_exponential(data):
    xdata = np.arange(len(data))
    ydata = np.array(data)
    fit,fit2 = opt.curve_fit(exp_func,xdata,ydata)
    return int(np.round(fit[1]))

def pz_correction(data, tau):
    pz_corrected = []
    pz_correction = []
    for i in range(0, n_data, 1):
        pz = np.sum(data[1:i-1])
        pz_correction.append(pz)
        #tau = 30000
        pz_corrected.append(data[i] + pz / tau)
    return pz_corrected

def trapezoidal_filter(data, gap_time, peaking_time):
    m = gap_time
    k = peaking_time
    l = peaking_time + gap_time
    filtered_signal = [0]*4096
    #print(filtered_signal[0:10])
    for j in range(k + 1, k + m, 1):
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

def trapezoidal_filter_2(data, gap_time, peaking_time):
    # ------- Apply exponential fit for the M multiplication --------------#
    # out = exp_fit(signal, plot_comps = None)
    # tau = out['exp_decay'].value #in nanoseconds
    # tau_std = out['exp_decay'].stderr
    # T_sp = 1 #nano seconds
    # M= tau
    # print('M ',M)
    ##---------Start of filter---------------------------#
    filtered_signal = [0]*4096  # 4096 = signal length in this data set, change as needed
    k = peaking_time
    m = gap_time

    M = 4400

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

    return acc_2

def get_energy_value(data): # TODO
    #if point == 'max':
    #    energy = max(data)
    #elif point == 'mid':
    #    energy = max(data)
    #else:
    #
    energy = max(data)
    return energy

def get_time_values(length_of_data, sampling_time):
    time_values = np.linspace(0, length_of_data * sampling_time, length_of_data)
    return time_values

def exp_fit(signal, plot_comps=None):
    '''
    out = exp_fit takes a range of data that and fits an exponential to the tail
    of the amplitude pulse returning the parameters of the peak
    with out.params

    To access the data do out['exp_amplitude'].value or out['exp_amplitude'].stderr

    signal: input pulse from digitizer
    '''
    x_lbound = 1100 #np.argmax(signal)
    x_ubound = 4096 #signal.shape[0]

    x_exp = range(x_lbound, x_ubound)
    #exp_data = signal[x_exp]
    exp_data = signal[1100:4096]
    y = np.asarray(exp_data)
    x = np.asarray(x_exp)

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(y, x=x)

    mod = exp_mod

    init = mod.eval(pars, x=x)

    out = mod.fit(y, pars, x=x)
    #print(out.fit_report())

    if plot_comps == True:
        plt.figure()
        plt.plot(signal, 'b')
        plt.plot(x, out.best_fit, 'r--')
        amp = out.params['exp_amplitude'].value
        decay = out.params['exp_decay'].value

        plt.annotate('$%0.2f*e^{-x/%0.4f \mu s} ---add units$' % (amp , decay), xy=(1700, 600), xytext=(2000, 650),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

        plt.show()

    return out.params

# START HERE FOR PASTED

def energy_value(filtered_signal, pickoff=None):
    if pickoff is None:
        reading = np.max(filtered_signal)
    else:
        if pickoff == 'Midpoint':  # i.e. FWHM
            half_max = np.max(filtered_signal)/2
            d = np.sign(half_max - np.array(filtered_signal[0:-1])) - np.sign(half_max - np.array(filtered_signal[1:]))
            left_idx = np.argwhere(d > 0)[0]
            right_idx = np.argwhere(d < 0)[-1]
            if (left_idx + right_idx) % 2 == 0:
                reading = (filtered_signal[(left_idx + right_idx)/2])
            else:
                shift = left_idx + ((left_idx + right_idx-1)/2)
                reading = (((filtered_signal[shift] + filtered_signal[shift+1])/2)+0.5)
        else:
            ValueError('{pick} is not None or Midpoint'.format(pick=pickoff))
    return int(reading)


def filter_and_get_energy(data, peak, gap):
    signal = data
    signal = baseline_correction(signal)
    #t = 100
    #signal = pz_correction(signal, t)
    signal = trapezoidal_filter_2(signal, gap, peak)
    energy_value = get_energy_value(signal)
    return energy_value

if __name__ == "__main__":

    filename = 'data/DATA_co60_2.h5'
    hf = tables.open_file(filename, "r")

    event_data = hf.root.EventData.read()
    raw_data = hf.root.RawData.read()
    hf.close()

    sampling_time = 10.0
    n_data = 4096
    x_values = np.linspace(0, n_data, n_data)
    print("XVAL" , len(x_values))

    k = 100 # peaking
    m = 100 # gap
    raw_signal = raw_data[2,:]
    print("RAW ", len(raw_signal))
    plt.plot(x_values, raw_signal)
    signal = baseline_correction(raw_signal)
    signal = trapezoidal_filter_2(signal, m, k)

    plt.figure(41)
    plt.cla()
    plt.clf()
    plt.plot(x_values, signal)
    plt.show()

    energy = get_energy_value(signal)
    print(energy)

    nrgs = []
    number_of_events = len(raw_data)
    number_of_events = 1000
    assert number_of_events < len(raw_data)
    k = 100 # peaking time
    l = 100 # gap

    for i in range(0, number_of_events, 1):
        gap = 100
        peak = 100
        signal = raw_data[i,:]
        signal = baseline_correction(signal)
        signal = trapezoidal_filter_2(signal, gap, peak)
        nrg = get_energy_value(signal)
        #nrg = filter_and_get_energy(raw_data[i,:], gap, peak)
        nrgs.append(round(nrg, 7))

    print(max(nrgs))
    print(min(nrgs))

    plt.hist(nrgs, 100, log=True)
    plt.show()

    sys.exit()
