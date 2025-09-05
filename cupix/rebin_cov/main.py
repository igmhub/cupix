import numpy as np
import configparser
from cupix.rebin_cov.healpix_px import Px_meas
from cupix.rebin_cov.lib_funcs import bin_func_k, bin_func_theta, rebin_k, rebin_theta, average_px, compute_covariance, calculate_window_matrix, bin_window, save_to_hdf5, calculate_V_zh_AM, get_sum_over_healpix
import matplotlib.pyplot as plt 
#root = './data/px_measurements/raw_mocks/'
root = '/Users/ssatyavolu/projects/DESI/Y3_Lya_Px/mocks/raw_mocks/'
datafile = 'px-nhp_41-zbins_4-thetabins_40.hdf5'

# Read data from HDF5 files and store px_data object

px_data = Px_meas(root+datafile)
F_zh_am, W_zh_am, k_m = px_data.unpack_healpix(positive_frequencies=True)
print(k_m)
print('Shape of F is',np.shape(F_zh_am)) # (N_z, N_theta, N_hp, N_k)
assert ~np.isnan(F_zh_am).any()
assert ~np.isnan(W_zh_am).any()

# set average resolution
R2_m = np.ones_like(k_m)
sigma = 0.1
R2_m = np.exp(-(k_m*sigma)**2)

# compute the normalisation
V_zh_am = calculate_V_zh_AM(W_zh_am,R2_m,px_data.L_fft)

# define rebin parameters 
bin_info = {} 

k_bins_ratio = 4 # number of k bins after rebinning will be Nk/k_bins_ratio
max_k = px_data.k_Nyq # maximum frequency to consider, in 1/A
k_max_ratio = 4 # maximum frequency will be max_k/k_max_ratio

# we will rebin the wavenumbers to make them more independent, and better measured

B_M_m, k_M_edges = bin_func_k(k_m,px_data.k_fund,k_bins_ratio,max_k,k_max_ratio,bin_func_type='top_hat') # B_M_m has shape (NK, Nk) and B_A_a has shape (Ntheta_rebin, Ntheta_bin)

# Rebin in k per healpix
F_zh_aM =  rebin_k(F_zh_am,B_M_m,healpix=True)
V_zh_aM = rebin_k(V_zh_am,B_M_m,healpix=True)

# Test everything is in order
#for nhp in range(3):
    #plt.plot(k_m,F_zh_am[1,10,nhp,:])
    #plt.plot(k_M,F_zh_aM[1,10,nhp,:],color='black')
    #z = px_data.z_bin_centers[1]
    #theta_min = px_data.theta_bin_min[10]
    #theta_max = px_data.theta_bin_max[10]
    #key = (z,theta_min,theta_max)
    #plt.plot(px_data.k_arr[:px_data.N_fft//2],px_data.px[key][nhp,:px_data.N_fft//2],linestyle='--')
    #assert np.isclose(px_data.px[key][nhp,:px_data.N_fft//2],F_zh_am[1,10,nhp,:]).any() # works only if max_k is k_Nyq
    #plt.show()

# define rebin parameters
theta_bins_ratio = 4
# we need an average over all theta_bins belonging to the new theta_rebins, which will give the array (z,nhp,Nk) for each theta_rebins
B_A_a, theta_min_A, theta_max_A = bin_func_theta(px_data.theta_bin_min,px_data.theta_bin_max,theta_bins_ratio,bin_func_type='top_hat')
#plt.plot(px_data.theta_bin_min,B_A_a[8,:])
#plt.xscale('log')
#plt.show()

# Rebin in theta per healpix
F_zh_AM = rebin_theta(F_zh_aM,B_A_a,healpix=True)
V_zh_AM = rebin_theta(V_zh_aM,B_A_a,healpix=True)

# Test everything is in order
#for nhp in range(1):
#    plt.plot(k_m,F_zh_am[1,10,nhp,:])
#    print(px_data.theta_bin_min[10],px_data.theta_bin_max[10])
#    print(theta_min_A[1],theta_max_A[1])
#    plt.plot(k_M,F_zh_AM[1,1,nhp,:],color='black')
#    plt.show()


# Measure average 

#R_zh_AM = np.ones_like(W_zh_AM, dtype=float)
P_z_AM, V_z_AM = average_px(F_zh_AM,W_zh_am,R2_m,px_data.L_fft,B_A_a,B_M_m)

#for i in range(len(theta_min_A)):
#    plt.plot(k_M,P_z_AM[0,i,:],label=theta_min_A[i])
#plt.legend()
#plt.show()

# Measure covariance
C_z_AMN, Php_z_AM = compute_covariance(F_zh_AM,V_zh_AM)

# Plot covariance
#print(C_z_AMN)
#plt.imshow(C_z_AMN[2,8,:,:])
#plt.show()

# Compare averages
#for i in range(len(theta_min_A)):
#    plt.plot(k_M,P_z_AM[0,i,:],label=theta_min_A[i])
#    plt.plot(k_M,Php_z_AM[0,i,:],linestyle='--')
#plt.legend()
#plt.show()

# Window matrix 

W_z_am  = get_sum_over_healpix(W_zh_am)
#print(np.shape(W_zh_am),np.shape(R_zh_am),'passed')
U_z_amn  = calculate_window_matrix(W_z_am,R2_m)


# bin window matrix
U_z_aMn, V_z_aM, V_z_am = bin_window(U_z_amn,B_M_m,W_z_am,R2_m,px_data.L_fft)


# Save to new hdf5 file with metadata and binning information for theory
outfile= root+'output_data_for_cupix_zbins_4_thetabins_40_nhp41.hdf5'
save_to_hdf5(outfile,P_z_AM,C_z_AMN,U_z_aMn,B_A_a,V_z_aM,k_m,k_M_edges,px_data.theta_bin_min,px_data.theta_bin_max,theta_min_A,theta_max_A,px_data.N_fft,px_data.L_fft,px_data.z_bin_centers)


