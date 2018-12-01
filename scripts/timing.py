from lab3_analysis_functions import *

# In[2]:


filename = './data/cs2.h5'
hf = tables.open_file(filename, "r")
raw_data_1 = import_data(filename)
event_data_cs1= hf.root.EventData.read()
hf.close()


# In[3]:


# filename = '/Users/Asia/Desktop/204/lab3/lab3/data/am1.h5'
# hf = tables.open_file(filename, "r")
# raw_data_1 = import_data(filename)
# event_data_am1= hf.root.EventData.read()
# hf.close()

# filename = '/Users/Asia/Desktop/204/lab3/lab3/data/am2.h5'
# hf = tables.open_file(filename, "r")
# raw_data_2 = import_data(filename)
# event_data_am2= hf.root.EventData.read()
# hf.close()

# filename = '/Users/Asia/Desktop/204/lab3/lab3/data/cs1.h5'
# hf = tables.open_file(filename, "r")
# raw_data_1 = import_data(filename)
# event_data_cs1= hf.root.EventData.read()
# hf.close()

# filename = '/Users/Asia/Desktop/204/lab3/lab3/data/cs2.h5'
# hf = tables.open_file(filename, "r")
# raw_data_2 = import_data(filename)
# event_data_cs2= hf.root.EventData.read()
# hf.close()


# In[4]:


# event_data_am = np.concatenate((event_data_am1, event_data_am2))
# event_data_cs = np.concatenate((event_data_cs1, event_data_cs2))


# In[5]:


# sort events by timestamp
event_data_cs1 = event_data_cs1[np.argsort(event_data_cs1['timestamp'])]


# In[6]:


time_btwn_events = np.diff(event_data_cs1['timestamp'])
counts, bin_edges = np.histogram(time_btwn_events, bins=30, range = [0, 30])
bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges
for i in range(0, len(bins), 1):
    bins[i] = i*10
plt.figure(4, figsize=(7, 5))
plt.cla()
plt.clf()
plt.plot(bins, counts, 'ko')
plt.xlabel('time to next event (ns)')
plt.ylabel('counts')
plt.title("Time-to-Next-Event Histogram")
plt.xlim([0, 300])
plt.savefig("./figures/time-to-next-event.pdf")
#plt.show()


# In[7]:


print('applying energy calibration to data files...')
print('(this part takes a bit of time)')

filename = './data/calibration_long.txt'
calibration = np.genfromtxt(filename,delimiter=' ')
slopes = calibration[:,0]
intercepts = calibration[:,1]

for i in range(0, 152, 1):
    mask = (event_data_cs1['detector'] == i)
    event_data_cs1['ADC_value'][mask] = calculate_energies(event_data_cs1['ADC_value'][mask], slopes[i], intercepts[i])
    if i == 90:
        print('still going... almost done')
    if i == 151:
        print('phew! done with that part')



# In[8]:


counts, bin_edges = np.histogram(event_data_cs1['ADC_value'], bins=2048, range = [300, 700])
bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

plt.cla()
plt.clf()
plt.plot(bins, counts)
plt.savefig('./figures/energy-spectrum.pdf')
#plt.show()


mask_1 = ((660 < event_data_cs1['ADC_value']) & (event_data_cs1['ADC_value'] < 663))
counts, bin_edges = np.histogram(event_data_cs1['ADC_value'][mask_1], bins=500, range = [657, 665])
bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

plt.cla()
plt.clf()
plt.plot(bins, counts)
plt.savefig('./figures/energy-spectrum-cut.pdf')
#plt.show()


# In[11]:


raw_data_1 = fast_baseline_correction(raw_data_1)


# In[12]:


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

#make_test_file(event_data_cs1[mask_1], raw_data_1)


# In[13]:


# full energy only
mask_1 = ((640 < event_data_cs1['ADC_value']) & (event_data_cs1['ADC_value'] < 680))
#print(event_data_cs1['ADC_value'][mask_1][0:10])


# In[14]:


#dc1, dc2, ac1, ac2
face1 = np.arange(0, 38) # dc1
face2 = np.arange(38, 76) # dc2
face3 = np.arange(76, 114) # ac1
face4 = np.arange(114, 152) # ac2


# In[15]:


def calculate_t50(signal, plot=False):
    signal = signal[:-100]
    x = np.linspace(0, len(signal) -100, len(signal) -99)
   # print(x)
    sig = savgol_filter(signal, 15, 3) # window size 51, polynomial order 3

    if sig.all() == 0:
        return -1000
    elif np.argmax(sig) == len(sig):
        return -1000
    else:
        maxval = np.amax(sig)
        fiftyval = maxval* 0.5
        fiftyindex = 0
        for i in range(0, len(sig), 1):
            if sig[i] <= fiftyval:
                fiftyindex = i

        if (fiftyindex + 3) >= len(sig):
            x_fit_low = np.linspace((fiftyindex - 2), len(sig), len(sig)-2 - fiftyindex)
            sig_fit_low = sig[int(fiftyindex - 2): len(sig)]
        else:
            x_fit_low = np.linspace((fiftyindex - 2), int(fiftyindex + 3), 5)
            sig_fit_low = sig[int(fiftyindex - 2): int(fiftyindex + 3)]
        if len(x_fit_low) != len(sig_fit_low):
            #print(fiftyindex)
            #print(len(x_fit_low))
            #print(len(sig_fit_low))
            #print(x_fit_low)
            #print(sig_fit_low)
            #plt.plot(signal)
            #plt.plot(sig)
            #plt.show()
            #plt.plot(signal[int(fiftyindex - 4):int(fiftyindex + 5)], 'o')
            #plt.plot(sig[int(fiftyindex -4): int(fiftyindex + 5)], 'o')
            #plt.show()
            x_fit_low = x_fit_low = np.linspace((fiftyindex - 1), int(fiftyindex + 2), 4)
            sig_fit_low = sig[int(fiftyindex -1): int(fiftyindex +2)]
            #print(len(x_fit_low))
            #print(len(sig_fit_low))
        x_fit_low = np.array(x_fit_low)
        sig_fit_low = np.array(sig_fit_low)
        if len(x_fit_low) < 1:
            #print('x empty')
            #plt.plot(signal)
            #plt.plot(sig)
            #plt.show()
            return -1000
        else:
            m, b = np.polyfit(x_fit_low, sig_fit_low, deg=1)
            fit_low = b + m * x_fit_low
            rise_low = ((fiftyval - b )/ m)

        t50 = (rise_low) * 10# ns

        if plot==True:
            plt.figure(figsize=(10,5))
            plt.plot(signal, '-', label = 'raw signal')
            plt.plot(sig, label = 'smoothed signal')
            plt.plot(x_fit_low, fit_low,'-', linewidth = 5.0,alpha=0.7, label = 'fit')
            plt.plot(t50/10, fiftyval, 'ro', label='t50')
            plt.title('T50 Fitting')
            plt.ylabel('ADC value')
            plt.xlabel('ADC sample (10 ns sampling time)')
            plt.legend()
            #plt.savefig('t50_fitting.pdf')
            plt.show()
        return t50


# In[16]:


#x = 7000
diff1vals = []
flag = 0 # 1 = detector1, 2= detector2, 3=detector1 neighbors, 4=detector2 neigbors, 5=other
delta_t50_values_1 = []
delta_t50_values_2 = []
for t in range(0, len(event_data_cs1['timestamp'][mask_1]), 1):
#for t in range(0, x, 1):
    diff0 = np.abs(event_data_cs1['timestamp'][mask_1][t] - event_data_cs1['timestamp'][mask_1][t-1])
    diff1 = np.abs(event_data_cs1['timestamp'][mask_1][t] - event_data_cs1['timestamp'][mask_1][t-2])
    # diff2 = np.abs(event_data_cs1['timestamp'][mask_1][t+1] - event_data_cs1['timestamp'][mask_1][t])
    # diff3 = np.abs(event_data_cs1['timestamp'][mask_1][t+2] - event_data_cs1['timestamp'][mask_1][t])
    if diff0 < 50 and diff1 > 50:
        # print('0', diff0)
        # print('1', diff1)
        # print('2', diff2)
        # print('3', diff3)
        detector1 = (event_data_cs1['detector'][mask_1][t-1])
        detector2 = (event_data_cs1['detector'][mask_1][t])
        #print('-------------')

        if detector1 in face1 and detector2 in face1:
            # print('face1 neighbors')
            flag = 3
        elif detector1 in face2 and detector2 in face2:
            # print('face2 neighbors')
            flag = 3
        elif detector1 in face3 and detector2 in face3:
            # print('face3 neighbors')
            flag = 4
        elif detector1 in face4 and detector2 in face4:
            # print('face4 neighbors')
            flag = 4

        elif detector1 in face1 and detector2 in face3:
            # print('detector1')
            flag  = 1
        elif detector1 in face3 and detector2 in face1:
            # print('detector1')
            flag  = 1
        elif detector1 in face2 and detector2 in face4:
            # print('detector2')
            flag  = 2
        elif detector1 in face4 and detector2 in face2:
            # print('detector2')
            flag  = 2
        else:
            flag= 5
            # print('other')

        if flag == 1:
            rid1 = (event_data_cs1['rid'][mask_1][t-1])
            rid2 = (event_data_cs1['rid'][mask_1][t])
            adc1 = (event_data_cs1['ADC_value'][mask_1][t-1])
            adc2 = (event_data_cs1['ADC_value'][mask_1][t])

            t501 = calculate_t50(raw_data_1[rid1], plot=False)
            t502 = calculate_t50(raw_data_1[rid2], plot=False)
            if rid1 > rid2: # electrons - holes = ac - dc
                #deltat50 = -(diff0 * 10) - t502 + t501 #- diff0 * 10
                deltat50 = (diff0 * 10) + t502 - t501
            else:  # rid1 < rid2
                #deltat50 = (diff0 * 10) + t502 - t501 #- diff0 * 10
                deltat50 = -(diff0 * 10) - t502 + t501
            deltat50 = round(float(deltat50),4)
            delta_t50_values_1.append(deltat50)
            diff1vals.append(diff0)

        elif flag == 2:
            rid1 = (event_data_cs1['rid'][mask_1][t-1])
            rid2 = (event_data_cs1['rid'][mask_1][t])
            adc1 = (event_data_cs1['ADC_value'][mask_1][t-1])
            adc2 = (event_data_cs1['ADC_value'][mask_1][t])
            t501 = calculate_t50(raw_data_1[rid1], plot=False)
            t502 = calculate_t50(raw_data_1[rid2], plot=False)

            if rid1 > rid2: # electrons - holes = ac - dc
                #deltat50 = -(diff0 * 10) - t502 + t501 #- diff0 * 10
                deltat50 = (diff0 * 10) + t502 - t501
            else:  # rid1 < rid2
                #deltat50 = (diff0 * 10) + t502 - t501 #- diff0 * 10
                deltat50 = -(diff0 * 10) - t502 + t501

            deltat50 = round(float(deltat50),4)
            delta_t50_values_2.append(deltat50)
            diff1vals.append(diff0)
        else:
            continue


# In[20]:


#print(len(delta_t50_values_1))
#print(len(delta_t50_values_2))
delta_t50_1 = np.asarray(delta_t50_values_1)#, dtype=float)
#print(type(delta_t50_1))
counts_det1, bin_edges = np.histogram(delta_t50_1, bins=200, range = [-225, 275])
bins_det1 = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

plt.cla()
plt.clf()
#plt.axvline(-160,color='k')
#plt.axvline(210,color='k')
plt.plot(bins_det1, counts_det1)
plt.title('$\Delta$t50 Values for Detector 1')
plt.ylabel('counts')
plt.xlabel('$\Delta$t50 (ns)')
plt.savefig('./figures/t50s_det1.pdf')
#plt.show()

delta_t50_2 = np.asarray(delta_t50_values_2)#, dtype=float)
#print(type(delta_t50_1))
counts_det2, bin_edges = np.histogram(delta_t50_2, bins=200, range = [-225, 275])
bins_det2 = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

plt.cla()
plt.clf()
#plt.axvline(-160,color='k')
#plt.axvline(210,color='k')
plt.plot(bins_det2, counts_det2)
plt.title('$\Delta$t50 Values for Detector 2')
plt.ylabel('counts')
plt.xlabel('$\Delta$t50 (ns)')
plt.savefig('./figures/t50s_det2.pdf')
#plt.show()

cut_counts_det1, bin_edges = np.histogram(delta_t50_1, bins=200, range = [-160, 210])
cut_bins_det1 = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

cut_counts_det2, bin_edges = np.histogram(delta_t50_2, bins=200, range=[-160, 210])
cut_bins_det2 = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges


# In[21]:


# method 1 = linear fit
x = (np.linspace(0, 15, len(cut_bins_det1)))
m, b = np.polyfit(cut_bins_det1, x, deg=1)
x_values = b + m * bins_det1
y_values = counts_det1
plt.xlim([-3,18])

plt.cla()
plt.clf()
plt.plot(x_values, y_values,'b--', label='method 1')
#plt.show()

# method 2
z_0 = 5.2  # mm
z_0 = 5.2 + 0.75# mm
k_c = 0.04  # mm/ns
c = 3*10**8 # m/s
c = 299.792 # mm/ns
z_coord_eq_1 = []
#print(len(delta_t50_values_2))
for i in delta_t50_values_2:
    z = z_0 + 0.04*(i)
    z_coord_eq_1.append(z)

z_coord_eq_1= np.asarray(z_coord_eq_1)#, dtype=float)
#print(type(delta_t50_1))
z_coord_eq_1_y, bin_edges = np.histogram(z_coord_eq_1, bins=200)
z_coord_eq_1_x = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

plt.plot(z_coord_eq_1_x, z_coord_eq_1_y,'r--', label = 'method 2')
plt.xlim([-3,18])
plt.title('Interaction Depth Profile')
plt.xlabel('depth (mm)')
plt.ylabel('counts')
plt.legend()
plt.savefig('./figures/interactiondepths.pdf')
#plt.show()


# In[ ]:





# In[82]:


#Geant4 comparison

hf = tables.open_file("./data/hits.h5", "r")
event_pointers = hf.root.EventPointers.read()
event_lengths = hf.root.EventLengths.read()
idata = hf.root.InteractionData.read()

# energy weighted z coordinates for full E deposition in a signel crystal

event_energies = []
z_values = []
z_values_all = []
for i in range(0, len(event_pointers), 1):
    #print('---', i)
    pointer = event_pointers[i]
    length = event_lengths[i]
    energy = np.sum(idata['energy'][pointer:pointer+length])
    z_coords = (idata['z'][pointer:pointer+length])
    if (energy > 661.6):
        neg = 0
        pos = 0
        for j in z_coords:
            z_values_all.append(j)
            if j > 0:
                pos = 1
            if j < 0:
                neg = 1
        if pos == 1 and neg == 0:
            event_energies.append(energy)
            z_val = []
            for j in idata[pointer:pointer+length]:
                z_val.append(j['z'] * j['energy'] / energy)
            z_coord_1 = np.sum(np.asarray(z_val))
            z_values.append(z_coord_1)
        elif pos == 0 and neg == 1:
            event_energies.append(energy)
            z_val = []
            for j in idata[pointer:pointer+length]:
                z_val.append(j['z'] * j['energy'] / energy)
            z_coord_1 = np.sum(np.asarray(z_val))
            z_values.append(z_coord_1)

event_energies = np.asarray(event_energies)
z_values = np.asarray(z_values)


# In[83]:


counts_weight, bin_edges = np.histogram(z_values, bins=200, range=[-25,0])
bins_weight = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges
counts_all, bin_edges = np.histogram(z_values_all, bins=200, range=[-25,0])
bins_all = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

# single interaction z coordinates for full E deposition in a signle crystal

event_energies = []
z_values = []
for i in range(0, len(event_pointers), 1):
    pointer = event_pointers[i]
    length = event_lengths[i]
    energy = np.sum(idata['energy'][pointer:pointer+length])
    if energy > 661.6:
        event_energies.append(energy)
        if length ==1:
            z_values.append(idata['z'][pointer:pointer+length])
        #elif length > 1:
            #print(length)

event_energies = np.asarray(event_energies)
z_values = np.asarray(z_values)

counts, bin_edges = np.histogram(z_values, bins=200,range=[-25,0])
bins = (bin_edges[1:]+bin_edges[:-1])/2 # bin centers from bin edges

for i in range(0, len(counts), 1):
    bins[i] = bins[i] + 20

for i in range(0, len(counts_weight), 1):
    bins_weight[i] = bins_weight[i] + 20

plt.cla()
plt.clf()
plt.plot(bins, counts,'b')
#plt.plot(bins_all,counts_all,'r')
#plt.plot(bins_weight,counts_weight,'g')
plt.xlim([-5, 20])
plt.savefig('./figures/detector1_g4_zpos.pdf')
#plt.show()

plt.cla()
plt.clf()
#plt.plot(bins,counts,'b')
#plt.plot(bins_all,counts_all,'r')
plt.plot(bins_weight, counts_weight,'g')
plt.xlim([-5, 20])
plt.savefig('./figures/detector1_g4_zpos_weighted.pdf')
#plt.show()


# In[89]:


y1 = z_coord_eq_1_y / np.sum(z_coord_eq_1_y)
y2 = counts / np.sum(counts)

plt.cla()
plt.clf()
plt.plot(z_coord_eq_1_x, y1,'r--', label = 'method 2')
plt.plot(bins, y2,'b')
plt.xlim([-3,18])
plt.title('Comparison with Simulated Data')
plt.xlabel('depth (mm)')
plt.ylabel('counts')
plt.legend()
plt.savefig('./figures/g4_comp.pdf')
#plt.show()


# In[ ]:
