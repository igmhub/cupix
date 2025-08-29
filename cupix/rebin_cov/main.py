import numpy as np
import configparser
from cupix.rebin_cov.healpix_px import Px_meas
from cupix.rebin_cov.lib_funcs import bin_func_k, bin_func_theta, rebin_k, rebin_theta, average_over_hp, compute_covariance, calculate_window_matrix, bin_window, save_to_hdf5
import matplotlib.pyplot as plt 
root = '/Users/ssatyavolu/projects/DESI/Y3_Lya_Px/mocks/raw_mocks/'
datafile = 'px-nhp_41-zbins_4-thetabins_40.hdf5'

# Read data from HDF5 files and store px_data object

px_data = Px_meas(root+datafile)
F_zh_am, W_zh_am, k_m = px_data.unpack_healpix(positive_frequencies=True,max_k=2.0)
print('Shape of F is',np.shape(F_zh_am)) # (N_z, N_theta, N_hp, N_k)
assert ~np.isnan(F_zh_am).any()
assert ~np.isnan(W_zh_am).any()

# define rebin parameters 
bin_info = {} 

k_bins_ratio = 4 # number of k bins after rebinning will be Nk/k_bins_ratio
k_max_ratio = 1 # maximum frequency will be max_k/k_max_ratio

# we will rebin the wavenumbers to make them more independent, and better measured

B_M_m, k_M = bin_func_k(k_m,px_data.k_fund,k_bins_ratio,k_max_ratio,bin_func_type='top_hat') # B_M_m has shape (NK, Nk) and B_A_a has shape (Ntheta_rebin, Ntheta_bin)

# Rebin in k 
F_zh_aM =  rebin_k(F_zh_am,B_M_m)
W_zh_aM = rebin_k(W_zh_am,B_M_m)

# Test everything is in order
#for nhp in range(3):
#    plt.plot(k_m,F_zh_am[1,10,nhp,:])
#    plt.plot(k_M,F_zh_aM[1,10,nhp,:],color='black')
#    z = px_data.z_bin_centers[1]
#    theta_min = px_data.theta_bin_min[10]
#    theta_max = px_data.theta_bin_max[10]
#    key = (z,theta_min,theta_max)
#    plt.plot(px_data.k_arr[:px_data.N_fft//2],px_data.px[key][nhp,:px_data.N_fft//2],linestyle='--')
#    #assert np.isclose(px_data.px[key][nhp,:px_data.N_fft//2],F_zh_am[1,10,nhp,:]).any() # works only if max_k is k_Nyq
#    plt.show()

# define rebin parameters
theta_bins_ratio = 4
# we need an average over all theta_bins belonging to the new theta_rebins, which will give the array (z,nhp,Nk) for each theta_rebins
B_A_a, theta_min_A, theta_max_A = bin_func_theta(px_data.theta_bin_min,px_data.theta_bin_max,theta_bins_ratio,bin_func_type='top_hat')
#plt.plot(px_data.theta_bin_min,B_A_a[8,:])
#plt.xscale('log')
#plt.show()

# Rebin in theta  

F_zh_AM = rebin_theta(F_zh_aM,B_A_a)
W_zh_AM = rebin_theta(W_zh_aM,B_A_a)

# Test everything is in order
#for nhp in range(1):
#    plt.plot(k_m,F_zh_am[1,10,nhp,:])
#    print(px_data.theta_bin_min[10],px_data.theta_bin_max[10])
#    print(theta_min_A[1],theta_max_A[1])
#    plt.plot(k_M,F_zh_AM[1,1,nhp,:],color='black')
#    plt.show()


# Measure average 

R_zh_AM = np.ones_like(W_zh_AM, dtype=float)
P_z_AM, W_z_AM, R_z_AM = average_over_hp(F_zh_AM,W_zh_AM,R_zh_AM,px_data.L_fft)

#for i in range(len(theta_min_A)):
#    plt.plot(k_M,P_Z_AM[0,i,:])
#plt.show()

# Measure covariance
C_z_AMN = compute_covariance(F_zh_AM,W_zh_AM,R_zh_AM,px_data.L_fft)
print(np.shape(C_z_AMN),'covariance shape')
print(C_z_AMN)
plt.imshow(C_z_AMN[0,1,:,:])
plt.show()

# Window matrix 
R_zh_am = np.ones_like(W_zh_am, dtype=float)
R_zh_aM = np.ones_like(W_zh_aM, dtype=float)
#print(np.shape(W_zh_am),np.shape(R_zh_am),'passed')
U_z_amn  = calculate_window_matrix(W_zh_am,R_zh_am,px_data.L_fft)
print(np.shape(U_z_amn))

# bin window matrix
U_z_aMn, V_z_aM, V_z_am = bin_window(U_z_amn,B_M_m,W_zh_am,R_zh_am,R_zh_aM,px_data.L_fft)
print(np.shape(U_z_aMn))

# Save to new hdf5 file with metadata and binning information for theory
outfile= root+'output_data_for_cupix_zbins_4_thetabins_40_nhp41.hdf5'
save_to_hdf5(outfile,P_z_AM,C_z_AMN,U_z_aMn,B_A_a,V_z_aM,V_z_am,k_m,k_M,px_data.theta_bin_min,px_data.theta_bin_max,theta_min_A,theta_max_A,px_data.N_fft,px_data.L_fft,px_data.z_bin_centers)
