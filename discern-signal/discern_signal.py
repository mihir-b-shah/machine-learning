
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial, signal

def lowpass(my_signal, sampl_freq, passband_frac = 0.5, stopband_frac = 0.75, filter_order = 21):
  nyquist_rate = sampl_freq / 2.
  desired = (1, 1, 0, 0)
  bands = (0, passband_frac*nyquist_rate, stopband_frac*nyquist_rate, nyquist_rate)
  filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
  filtered = signal.filtfilt(filter_coefs, [1], my_signal)
  return filtered

#returns (merged_x, merged_y, noise_x, noise_y, signal_x, signal_y)
def gen_samples(FRAC_NOISE = 0.1, FRAC_SIGNAL = 0.01):

    # Arbitrary choices
    X_MIN = 0
    X_MAX = 1000
    Y_MIN = -50
    Y_MAX = 50
    AREA = (X_MAX-X_MIN)*(Y_MAX-Y_MIN);
    
    BASE_FREQUENCY = 0.001

    NUM_NOISE_SAMPL = int(FRAC_NOISE*AREA)
    x_noise = BASE_FREQUENCY*np.random.randint(X_MIN*int(1/BASE_FREQUENCY), X_MAX*int(1/BASE_FREQUENCY), size=NUM_NOISE_SAMPL)
    y_noise = BASE_FREQUENCY*np.random.randint(Y_MIN*int(1/BASE_FREQUENCY), Y_MAX*int(1/BASE_FREQUENCY), size=NUM_NOISE_SAMPL)
    
    SIGNAL_NUM_CYCLES = 5
    NUM_SIGNAL_SAMPL = int(FRAC_SIGNAL*AREA)
    
    x_signal = BASE_FREQUENCY*np.random.randint(0,2*SIGNAL_NUM_CYCLES*np.pi*int(1/BASE_FREQUENCY), NUM_SIGNAL_SAMPL)
    y_signal = Y_MAX*np.sin(x_signal)
    x_signal = X_MAX/(2*SIGNAL_NUM_CYCLES*np.pi)*x_signal
    
    return np.concatenate((x_signal,x_noise)), np.concatenate((y_signal,y_noise)), x_noise, y_noise, x_signal, y_signal

def gen_vector_field():
    x_merged, y_merged, x_noise, y_noise, x_signal, y_signal = gen_samples()
    merged = np.vstack((x_merged, y_merged)).T
    kdtree = spatial.cKDTree(merged)
    
    NUM_NEAREST_NEIGHBORS = 3
    nn_distances_all, nn_indices_all = kdtree.query(merged, NUM_NEAREST_NEIGHBORS)

    stable_filter = np.logical_or(nn_distances_all > 1e6, nn_distances_all == np.nan)
    nn_distances_all[stable_filter] = 0
    
    print(nn_distances_all)

    # go for a general solution using masked arrays.
    vects = merged[nn_indices_all]-merged[:,np.newaxis,:]
    #print(vects.shape)
    casted_distances = nn_distances_all[:,:,np.newaxis]
    #print(casted_distances)
    field = np.sum(vects*casted_distances,axis=1)
    
    plt.scatter(x_noise, y_noise, color='blue', s=1)
    plt.scatter(x_signal, y_signal, color='teal', s=1)
    plt.quiver(merged[:,0], merged[:,1], field[:,0], field[:,1])
    plt.show()
    
def gen_cluster_field(CLUSTER_RADIUS = 2):
    x_merged, y_merged, x_noise, y_noise, x_signal, y_signal = gen_samples()
    merged = np.vstack((x_merged, y_merged)).T
    kdtree = spatial.cKDTree(merged)

    weights = kdtree.query_ball_point(merged, CLUSTER_RADIUS, return_length=True);
    plt.figure(1)
    plt.scatter(x_merged, y_merged, s=1)
    plt.figure(2)
    print(weights/CLUSTER_RADIUS)
    
    #horribly inefficient.
    x_merged_new = []
    y_merged_new = []
    weights_new = []
    
    for i in range(len(x_merged)):
        if(weights[i] > 3):
            x_merged_new.append(x_merged[i])
            y_merged_new.append(y_merged[i])
            weights_new.append(weights[i])
    
    plt.scatter(x_merged_new, y_merged_new, s=weights_new)
    plt.show()
    
gen_cluster_field()