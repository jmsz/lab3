import numpy as np
import matplotlib.pyplot as plt

def GetT50vsDepthTable(filename):
    with open(filename) as myfile:
        values = np.genfromtxt(filename, delimiter=' ', skip_header=2, dtype=float, autostrip=True)
        depths = values[:,0:1]
        T50values = (values[:,3:4])
        header = np.genfromtxt(filename, delimiter=' ', skip_footer=len(depths), dtype=float, autostrip=True)
        #print header
        height = header[0:1,1:2]
        width = header[1:2,1:2]
        #print float(height)
        #print float(width)
        #print(min(T50values))
        #print(np.mean(T50values))
        #print(max(T50values))
    return depths, T50values, height, width

#=========================================================================================================#
if __name__ == "__main__":

#=========================================================================================================#
# get look-up tables for t50 vs depth for different geometries
    print('Calculating coincidence window...')
    tvshfile = './data/tvsh_z15_x2.txt'
    height = 0
    width = 0
    x2_depths, x2_T50values, detector_depth, strip_width = GetT50vsDepthTable(tvshfile)
    x2_T50values = -1 * x2_T50values
    print('max T50 difference: ', round(np.amax(abs(x2_T50values)),5))
    print('mean T50 difference: ', round(np.mean(abs(x2_T50values)),5))
    plt.plot(x2_depths * 10, x2_T50values)
    #plt.plot(x2_depths[0] * 10, x2_T50values[0], 'bo')
    #plt.plot(x2_depths[-1] * 10, x2_T50values[-1], 'ro')
    #plt.ylim([])
    plt.xlim([0, 16])
    plt.xticks(np.linspace(0, 16, 17))
    plt.title('T50 vs depth')
    plt.xlabel('depth (mm)')
    plt.ylabel('$\Delta$T50 (ns)')
    plt.savefig('./figures/deltat50_vs_depth.pdf')
    print('Finished calculating coincidence window...')
    #plt.show()

#=========================================================================================================#
#=========================================================================================================#
