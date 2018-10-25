def calculate_signal_rise_time_interpolation(signal, plot=False):
    signal = signal[910:1070]
    x = np.linspace(0, 1070-910, 1070-910)
    sig = savgol_filter(signal, 51, 3) # window size 51, polynomial order 3

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
    print('fit')
    print(rise_high)
    print(rise_low)
    print(risetime)
    print('basic')
    print(ninetyindex)
    print(tenindex)
    print(ninetyindex - tenindex)
    if plot==True:
        plt.figure(figsize=(10,5))
        plt.plot(signal, '-')
        plt.plot(sig)
        plt.plot(x_fit_high, fit_high,'o')
        plt.plot(x_fit_low, fit_low,'o')
        plt.show()
    return risetime
